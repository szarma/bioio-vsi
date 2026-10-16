"""Extract image metadata from a parsed VSI tree."""

from __future__ import annotations

import glob
import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bioio_vsi import errors
from bioio_vsi._accessors import (
    UNIT_MICROMETRES,
    UNIT_MILLISECONDS,
    UNIT_NANOMETRES,
    PixelSizeUm,
    camera_bit_depth_from_path,
    kids_unique_by_second,
    node_value,
    prop_value,
    read_boundary_rect,
    read_dimension_kind_code,
    read_dimension_sizes,
    read_format_version_raw,
    read_optional_int,
    read_optional_string,
    read_optional_unix_time,
    read_pixel_size_pair,
    read_stage_origin,
    unique_child,
)
from bioio_vsi._paths import resolve_path
from bioio_vsi._tree import (
    FieldNode,
    VsiTree,
    collection_root,
    document_root,
    load_tree,
)

# Decision 8: warn on unseen format versions that pass structural checks.
TESTED_FORMAT_VERSIONS: frozenset[tuple[int, int]] = frozenset(
    {
        (3, 22),
        (3, 26),
        (3, 27),
        (3, 30),
        (3, 32),
        (3, 33),
    }
)

DIM_KIND: dict[int, str] = {1: "Z", 2: "T", 4: "C"}
DIM_TO_AXIS = {"Z": "z", "T": "t", "C": "c"}
UNSUPPORTED_DIM_KINDS = {3: "lambda", 9: "phase"}


def check_pixel_size_unit(sp: FieldNode | None, vsi_path: str) -> None:
    unit = node_value(sp, path=f"{vsi_path}/2005", tag=2020)
    if unit is not None:
        read_optional_string(unit, path=f"{vsi_path}/2005/2020", field="2020")
    if unit is not None and str(unit) != UNIT_MICROMETRES:
        raise errors.UnknownUnitError(
            f"pixel size unit 2020 is {unit!r}, expected {UNIT_MICROMETRES!r}",
            path=vsi_path,
        )


def format_version_tuple(raw: Sequence[int] | None) -> tuple[int, int] | None:
    if not raw or len(raw) < 1:
        return None
    packed = raw[0]
    if isinstance(packed, bool) or not isinstance(packed, int):
        return None
    return (packed >> 16, packed & 0xFFFF)


@dataclass
class ChannelInfo:
    index: int
    name: str | None
    alt_name: str | None
    emission_nm: float | None


@dataclass
class DimensionSlot:
    slot_index: int
    kind: str | None
    z_start_um: float | None = None
    z_step_um: float | None = None
    channels: list[ChannelInfo] = field(default_factory=list)


@dataclass
class ImageVolumeMeta:
    stack_id: int
    name: str | None
    image_type: int | None
    image_type_name: str | None
    width: int
    height: int
    sizes_2003: list[int] | None
    pixel_size_um: PixelSizeUm | None
    stage_origin: Sequence[float] | None
    acquisition_time: float | None
    camera_bit_depth: int | None
    format_version: tuple[int, int] | None
    format_build: int | None
    writer_build: int | None
    slots: list[DimensionSlot]
    plane_z_um: np.ndarray | None
    plane_time_ms: np.ndarray | None
    ets_path: str | None
    tree: VsiTree
    volume_node: FieldNode


def _external_frame(volume: FieldNode, *, vsi_path: str, stack_id: int) -> FieldNode | None:
    base = f"{vsi_path}/2001[{stack_id}]"
    with_boundary: list[FieldNode] = []
    for fr in kids_unique_by_second(volume, 2002, path=base, member="frame"):
        frame_path = f"{base}/2002[{fr.second}]"
        if unique_child(fr, 2018, path=frame_path) is not None:
            with_boundary.append(fr)
    if len(with_boundary) > 1:
        raise errors.AmbiguousMetadataError(
            f"{base}/2002: {len(with_boundary)} frames declare boundary tag 2018",
            path=vsi_path,
        )
    return with_boundary[0] if with_boundary else None


def _find_ets_path(vsi_path: str, stack_id: int) -> str | None:
    base = os.path.join(
        os.path.dirname(vsi_path),
        "_" + os.path.splitext(os.path.basename(vsi_path))[0] + "_",
        f"stack{stack_id}",
    )
    candidates = sorted(glob.glob(glob.escape(base) + "/frame_t*.ets"))
    if not candidates:
        return None
    if len(candidates) > 1:
        raise errors.MultipleEtsCandidatesError(
            f"stack{stack_id} has {len(candidates)} frame_t*.ets files "
            f"({candidates}), expected one",
            path=vsi_path,
        )
    return candidates[0]


def _frame_indices(
    frame_index: int,
    kinds_order: list[str],
    sizes: list[int],
) -> dict[str, int]:
    rem = frame_index
    out: dict[str, int] = {}
    for kind, size in zip(kinds_order, sizes, strict=True):
        out[kind] = rem % size
        rem //= size
    return out


def _axis_size(kinds_order: list[str], sizes: list[int], axis: str) -> int:
    if axis not in kinds_order:
        return 1
    return sizes[kinds_order.index(axis)]


def _z_slot_params(slots: list[DimensionSlot]) -> tuple[float | None, float | None]:
    for slot in slots:
        if slot.kind == "Z":
            return slot.z_start_um, slot.z_step_um
    return None, None


def _z_positions_from_step(
    t: int,
    c: int,
    z: int,
    z_start: float,
    z_step: float,
) -> np.ndarray:
    positions = z_start + np.arange(z, dtype=np.float64) * z_step
    out: np.ndarray = np.broadcast_to(positions, (t, c, z)).copy()
    return out


def _plane_array_or_none(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None or np.all(np.isnan(arr)):
        return None
    return arr


def _resolve_plane_z_um(
    plane_z: np.ndarray | None,
    kinds_order: list[str],
    sizes: list[int],
    slots: list[DimensionSlot],
) -> np.ndarray | None:
    t = _axis_size(kinds_order, sizes, "T")
    c = _axis_size(kinds_order, sizes, "C")
    z = _axis_size(kinds_order, sizes, "Z")
    if plane_z is not None and np.any(~np.isnan(plane_z)):
        return plane_z
    if z > 1:
        z_start, z_step = _z_slot_params(slots)
        if z_start is not None and z_step is not None:
            return _z_positions_from_step(t, c, z, z_start, z_step)
    return _plane_array_or_none(plane_z)


def _expected_plane_count(kinds_order: list[str], sizes: list[int]) -> int:
    n = 1
    for kind in kinds_order:
        n *= sizes[kinds_order.index(kind)]
    return n


def _build_plane_arrays(
    frames_with_props: list[FieldNode],
    kinds_order: list[str],
    sizes: list[int],
    vsi_path: str,
    stack_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    t = _axis_size(kinds_order, sizes, "T")
    c = _axis_size(kinds_order, sizes, "C")
    z = _axis_size(kinds_order, sizes, "Z")
    arr_z = np.full((t, c, z), np.nan, dtype=np.float64)
    arr_t = np.full((t, c, z), np.nan, dtype=np.float64)
    expected_planes = _expected_plane_count(kinds_order, sizes)
    seen_frames: set[int] = set()
    for fr in frames_with_props:
        fp = unique_child(fr, 2006, path=f"{vsi_path}/2001[{stack_id}]/2002")
        if fp is None:
            continue
        fi = read_optional_int(
            fr.second,
            path=f"{vsi_path}/2001[{stack_id}]/2002",
            field="frame id",
        )
        if fi is None:
            raise errors.InvalidIdentifierError(
                f"image {stack_id} frame with plane metadata has no frame id",
                path=vsi_path,
            )
        if not (0 <= fi < expected_planes):
            raise errors.InvalidIdentifierError(
                f"image {stack_id} frame id {fi} out of range 0..{expected_planes - 1}",
                path=vsi_path,
            )
        if fi in seen_frames:
            raise errors.InvalidIdentifierError(
                f"image {stack_id} duplicate plane metadata for frame {fi}",
                path=vsi_path,
            )
        seen_frames.add(fi)
        idx = _frame_indices(fi, kinds_order, sizes)
        ti = idx.get("T", 0)
        ci = idx.get("C", 0)
        zi = idx.get("Z", 0)
        zv = prop_value(
            unique_child(fp, 2014, path=f"{vsi_path}/2001[{stack_id}]/2002/2006"),
            path=f"{vsi_path}/2001[{stack_id}]/2002/2006/2014",
            expected_unit=UNIT_MICROMETRES,
            allow_nan=True,
        )
        tv = prop_value(
            unique_child(fp, 2017, path=f"{vsi_path}/2001[{stack_id}]/2002/2006"),
            path=f"{vsi_path}/2001[{stack_id}]/2002/2006/2017",
            expected_unit=UNIT_MILLISECONDS,
            allow_nan=True,
        )
        arr_z[ti, ci, zi] = zv if zv is not None else np.nan
        arr_t[ti, ci, zi] = tv if tv is not None else np.nan
    return arr_z, arr_t


def _parse_volume_node(tree: VsiTree, volume: FieldNode) -> ImageVolumeMeta:
    vsi_path = tree.path
    stack_id = read_optional_int(
        volume.second,
        path=f"{vsi_path}/2001[{volume.second}]",
        field="stack id",
    )
    if stack_id is None:
        stack_id = 0
    vol_path = f"{vsi_path}/2001[{stack_id}]"
    sp = unique_child(volume, 2005, path=vol_path)
    fr = _external_frame(volume, vsi_path=vsi_path, stack_id=stack_id)
    width = height = 0
    has_ext: int | None = None
    if fr is not None:
        frame_path = f"{vol_path}/2002[{fr.second}]"
        ext = unique_child(fr, 2018, path=frame_path)
        if ext is not None:
            ext_path = f"{frame_path}/2018"
            rect = node_value(ext, path=ext_path, tag=2053)
            if rect is not None:
                _x0, _y0, width, height = read_boundary_rect(
                    rect,
                    path=f"{ext_path}/2053",
                )
            raw_ext_flag = node_value(ext, path=ext_path, tag=20005)
            if raw_ext_flag is not None:
                has_ext = read_optional_int(
                    raw_ext_flag,
                    path=f"{ext_path}/20005",
                    field="20005",
                )
    if width <= 0 or height <= 0:
        raise errors.InvalidStructureError(
            f"image {stack_id} has no valid 2002/2018/2053 width and height",
            path=vsi_path,
        )

    raw_sizes = node_value(volume, path=f"{vsi_path}/2001[{stack_id}]", tag=2003)
    sizes_2003 = (
        None
        if raw_sizes is None
        else read_dimension_sizes(
            raw_sizes,
            path=f"{vsi_path}/2001[{stack_id}]/2003",
        )
    )

    coll = collection_root(tree)
    raw_2035 = node_value(coll, path=vsi_path, tag=2035)
    fmt_ver = None
    fmt_build = None
    if raw_2035 is not None:
        parsed_2035 = read_format_version_raw(raw_2035, path=f"{vsi_path}/2000/2035")
        fmt_ver = format_version_tuple(parsed_2035)
        fmt_build = parsed_2035[1]

    doc = document_root(tree)
    writer = (
        read_optional_int(
            node_value(doc, path=vsi_path, tag=21),
            path=f"{vsi_path}/2/21",
            field="build",
        )
        if doc
        else None
    )

    slots: list[DimensionSlot] = []
    kinds_order: list[str] = []
    for ord_i, dd in enumerate(
        kids_unique_by_second(volume, 2007, path=vol_path, member="dimension slot")
    ):
        slot_path = f"{vol_path}/2007[{ord_i}]"
        ps = kids_unique_by_second(dd, 2008, path=slot_path, member="channel")
        slot_id = read_optional_int(
            dd.second,
            path=f"{vsi_path}/2001[{stack_id}]/2007[{ord_i}]",
            field="slot identifier",
        )
        if ps:
            if slot_id is None or slot_id != ord_i:
                raise errors.InvalidIdentifierError(
                    f"image {stack_id} dimension slot {ord_i} has identifier "
                    f"{dd.second!r}, expected {ord_i}",
                    path=vsi_path,
                )
        elif slot_id is not None and slot_id != ord_i:
            raise errors.InvalidIdentifierError(
                f"image {stack_id} dimension slot {ord_i} has identifier "
                f"{dd.second}, expected {ord_i}",
                path=vsi_path,
            )
        slot_idx = ord_i
        slot = DimensionSlot(slot_index=slot_idx, kind=None)
        if ps:
            kind_codes: list[int] = []
            for p_node in ps:
                kind_node = unique_child(
                    p_node,
                    2023,
                    path=f"{vsi_path}/2001[{stack_id}]/2007[{ord_i}]/2008",
                )
                if kind_node is None:
                    continue
                kc = read_dimension_kind_code(
                    kind_node.value,
                    path=f"{vsi_path}/2001[{stack_id}]/2007[{ord_i}]/2008/2023",
                    field="2023",
                )
                if kc in UNSUPPORTED_DIM_KINDS:
                    raise errors.UnknownDimensionKindError(
                        f"image {stack_id} declares unsupported "
                        f"{UNSUPPORTED_DIM_KINDS[kc]} dimension (kind code {kc})",
                        path=vsi_path,
                    )
                if kc not in DIM_KIND:
                    raise errors.UnknownDimensionKindError(
                        f"image {stack_id} declares dimension kind code {kc}",
                        path=vsi_path,
                    )
                kind_codes.append(kc)
            if not kind_codes:
                raise errors.InvalidStructureError(
                    f"image {stack_id} dimension slot {ord_i} "
                    "declares no dimension kind (tag 2023)",
                    path=vsi_path,
                )
            if len(set(kind_codes)) != 1:
                raise errors.UnknownDimensionKindError(
                    f"image {stack_id} dimension slot {ord_i} has "
                    f"conflicting kind codes {kind_codes}",
                    path=vsi_path,
                )
            slot.kind = DIM_KIND[kind_codes[0]]
            kinds_order.append(slot.kind)
            p0 = ps[0]
            z12 = unique_child(
                p0,
                2012,
                path=f"{vsi_path}/2001[{stack_id}]/2007[{ord_i}]/2008[0]",
            )
            z13 = unique_child(
                p0,
                2013,
                path=f"{vsi_path}/2001[{stack_id}]/2007[{ord_i}]/2008[0]",
            )
            slot.z_start_um = prop_value(
                z12,
                path=f"{vsi_path}/2001[{stack_id}]/2007[{ord_i}]/2008[0]/2012",
                expected_unit=UNIT_MICROMETRES,
            )
            slot.z_step_um = prop_value(
                z13,
                path=f"{vsi_path}/2001[{stack_id}]/2007[{ord_i}]/2008[0]/2013",
                expected_unit=UNIT_MICROMETRES,
            )
            for ch_ord, p in enumerate(ps):
                ch_id = read_optional_int(
                    p.second,
                    path=f"{vsi_path}/2001[{stack_id}]/2007[{ord_i}]/2008[{ch_ord}]",
                    field="channel identifier",
                )
                if ch_id is None:
                    raise errors.InvalidIdentifierError(
                        f"image {stack_id} channel {ch_ord} has no identifier",
                        path=vsi_path,
                    )
                if ch_id != ch_ord:
                    raise errors.InvalidIdentifierError(
                        f"image {stack_id} channel {ch_ord} has identifier "
                        f"{p.second}, expected {ch_ord}",
                        path=vsi_path,
                    )
                ch_idx = ch_ord
                em_node = unique_child(
                    p,
                    2417,
                    path=f"{vsi_path}/2001[{stack_id}]/2007[{ord_i}]/2008[{ch_ord}]",
                )
                slot.channels.append(
                    ChannelInfo(
                        index=ch_idx,
                        name=read_optional_string(
                            node_value(
                                p,
                                path=f"{vsi_path}/2001[{stack_id}]/2007[{ord_i}]/2008[{ch_ord}]",
                                tag=2021,
                            ),
                            path=f"{vsi_path}/2001[{stack_id}]/2007[{ord_i}]/2008[{ch_ord}]/2021",
                            field="2021",
                        ),
                        alt_name=read_optional_string(
                            node_value(
                                p,
                                path=f"{vsi_path}/2001[{stack_id}]/2007[{ord_i}]/2008[{ch_ord}]",
                                tag=2419,
                            ),
                            path=f"{vsi_path}/2001[{stack_id}]/2007[{ord_i}]/2008[{ch_ord}]/2419",
                            field="2419",
                        ),
                        emission_nm=prop_value(
                            em_node,
                            path=f"{vsi_path}/2001[{stack_id}]/2007[{ord_i}]/2008[{ch_ord}]/2417",
                            expected_unit=UNIT_NANOMETRES,
                        ),
                    )
                )
        slots.append(slot)

    plane_z: np.ndarray | None = None
    plane_t: np.ndarray | None = None
    if sizes_2003 and kinds_order and len(sizes_2003) == len(kinds_order):
        frames_with_props: list[FieldNode] = []
        for fr in kids_unique_by_second(volume, 2002, path=vol_path, member="frame"):
            frame_path = f"{vol_path}/2002[{fr.second}]"
            if unique_child(fr, 2006, path=frame_path) is not None:
                frames_with_props.append(fr)
        if frames_with_props:
            plane_z, plane_t = _build_plane_arrays(
                frames_with_props, kinds_order, sizes_2003, vsi_path, stack_id
            )
        plane_z = _resolve_plane_z_um(plane_z, kinds_order, sizes_2003, slots)
        plane_t = _plane_array_or_none(plane_t)

    check_pixel_size_unit(sp, vsi_path)
    scale = node_value(sp, path=f"{vsi_path}/2001[{stack_id}]/2005", tag=2019) if sp else None
    pixel_size: PixelSizeUm | None = None
    if scale is not None:
        pixel_size = read_pixel_size_pair(
            scale,
            path=f"{vsi_path}/2001[{stack_id}]/2005/2019",
        )

    bit_depth = camera_bit_depth_from_path(sp, vsi_path)

    ets_path = _find_ets_path(vsi_path, stack_id)
    if has_ext == 1 and ets_path is None:
        raise errors.MissingPixelFileError(
            f"image {stack_id} declares an external pixel file but "
            f"no frame_t*.ets was found under stack{stack_id}/",
            path=vsi_path,
        )

    return ImageVolumeMeta(
        stack_id=stack_id,
        name=read_optional_string(
            node_value(sp, path=f"{vsi_path}/2001[{stack_id}]/2005", tag=2030) if sp else None,
            path=f"{vsi_path}/2001[{stack_id}]/2005/2030",
            field="2030",
        ),
        image_type=read_optional_int(
            node_value(sp, path=f"{vsi_path}/2001[{stack_id}]/2005", tag=2074) if sp else None,
            path=f"{vsi_path}/2001[{stack_id}]/2005/2074",
            field="2074",
        ),
        image_type_name=None,
        width=width,
        height=height,
        sizes_2003=sizes_2003,
        pixel_size_um=pixel_size,
        stage_origin=read_stage_origin(
            node_value(sp, path=f"{vsi_path}/2001[{stack_id}]/2005", tag=2018) if sp else None,
            path=f"{vsi_path}/2001[{stack_id}]/2005/2018",
        ),
        acquisition_time=read_optional_unix_time(
            node_value(sp, path=f"{vsi_path}/2001[{stack_id}]/2005", tag=2015) if sp else None,
            path=f"{vsi_path}/2001[{stack_id}]/2005/2015",
            field="2015",
        ),
        camera_bit_depth=bit_depth,
        format_version=fmt_ver,
        format_build=fmt_build,
        writer_build=writer,
        slots=slots,
        plane_z_um=plane_z,
        plane_time_ms=plane_t,
        ets_path=ets_path,
        tree=tree,
        volume_node=volume,
    )


def list_image_volumes(tree: VsiTree) -> list[FieldNode]:
    coll = collection_root(tree)
    return kids_unique_by_second(
        coll,
        2001,
        path=f"{tree.path}/2000",
        member="image stack",
    )


def parse_single_image(tree: VsiTree) -> ImageVolumeMeta:
    volumes = list_image_volumes(tree)
    if not volumes:
        raise errors.NoImageError("no image volumes (tag 2001)", path=tree.path)
    if len(volumes) > 1:
        ids = [v.second for v in volumes]
        raise errors.MultiImageError(
            f"{len(volumes)} images (stack ids {ids}); "
            "only single-image cellSens files are supported",
            path=tree.path,
        )
    meta = _parse_volume_node(tree, volumes[0])
    if meta.format_version and meta.format_version not in TESTED_FORMAT_VERSIONS:
        warnings.warn(
            f"{tree.path}: format version {meta.format_version[0]}.{meta.format_version[1]} "
            f"has not been tested; reading because structural checks passed",
            UserWarning,
            stacklevel=2,
        )
    return meta


def load_vsi(path: str | Path) -> ImageVolumeMeta:
    tree = load_tree(resolve_path(path))
    return parse_single_image(tree)


def document_properties(tree: VsiTree) -> dict[str, Any]:
    doc = document_root(tree)
    if doc is None:
        return {}
    doc_path = f"{tree.path}/2"
    return {
        "product": node_value(doc, path=doc_path, tag=34),
        "product_version": node_value(doc, path=doc_path, tag=35),
        "build": node_value(doc, path=doc_path, tag=21),
        "name": node_value(doc, path=doc_path, tag=11),
    }
