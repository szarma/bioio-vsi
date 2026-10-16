"""Recreate a readable .vsi + .ets tree from anonymised fixtures.

Pixels come from `expected_tile_pixels`, drawn at each pyramid level's own
size (not subsampled from level 0), so tests can compare against the same
helper. The `.ets` is sparse: headers and unfilled tiles stay on the zero
region of the original length. JPEG 2000 payloads are appended after that end,
then the tile table is copied there and `tabpos` updated. By default only the
first plane is filled, at levels of at most 64 MiB per plane plus the
smallest level.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import BinaryIO, Literal

from bioio_vsi._anonymize import read_skel
from bioio_vsi._ets import EtsIndex, read_ets_index
from bioio_vsi._fixture_pixels import expected_tile_pixels
from bioio_vsi._stack import AxisLayout
from bioio_vsi._vsi import DIM_TO_AXIS, ImageVolumeMeta, load_vsi
from bioio_vsi.errors import MissingCodecsExtraError, VsiError

DEFAULT_SPARSE_LIMIT = 1_000_000_000

PlanesArg = Literal["default", "all"]


def _sparse_supported() -> bool:
    return sys.platform != "win32"


def _write_ets_shell(skel_path: Path, ets_path: Path) -> int:
    nominal, ranges = read_skel(skel_path)
    if not _sparse_supported() and nominal > DEFAULT_SPARSE_LIMIT:
        raise VsiError(
            f"nominal ETS size {nominal} exceeds sparse limit {DEFAULT_SPARSE_LIMIT} "
            "on this platform"
        )
    ets_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ets_path, "wb") as f:
        if nominal > 0:
            f.seek(nominal - 1)
            f.write(b"\0")
        for offset, chunk in ranges:
            f.seek(offset)
            f.write(chunk)
    return nominal


def _axis_order_from_meta(meta: ImageVolumeMeta) -> list[str]:
    order: list[str] = []
    for slot in meta.slots:  # type: ignore[attr-defined]
        if slot.kind and slot.kind in DIM_TO_AXIS:
            order.append(slot.kind)
    return order


def _layout_from_meta(meta: ImageVolumeMeta, idx: object) -> AxisLayout:
    slot_by_axis: dict[str, int] = {}
    sizes: dict[str, int] = {}
    for slot in meta.slots:  # type: ignore[attr-defined]
        if slot.kind and slot.kind in DIM_TO_AXIS:
            ax = DIM_TO_AXIS[slot.kind]
            slot_by_axis[ax] = slot.slot_index
            if meta.sizes_2003 is not None:  # type: ignore[attr-defined]
                sizes[ax] = int(meta.sizes_2003[slot.slot_index])
    return AxisLayout(
        slot_by_axis=slot_by_axis,
        sizes=sizes,
        n_levels=idx.n_levels,  # type: ignore[attr-defined]
    )


# The default fills the first plane only at levels up to this size, so a large
# stitched scan does not grow gigabytes of real pixels on disk.
DEFAULT_PLANE_BYTES = 64 * 1024 * 1024


def _plane_indices_to_fill(
    layout: AxisLayout,
    *,
    planes: PlanesArg,
    level_bytes: list[int],
) -> list[tuple[int, int, int, int]]:
    t_max = layout.sizes.get("t", 1)
    c_max = layout.sizes.get("c", 1)
    z_max = layout.sizes.get("z", 1)
    out: list[tuple[int, int, int, int]] = []
    if planes == "all":
        for level in range(layout.n_levels):
            for t in range(t_max):
                for c in range(c_max):
                    for z in range(z_max):
                        out.append((t, c, z, level))
    else:
        for level in range(layout.n_levels):
            if level_bytes[level] <= DEFAULT_PLANE_BYTES or level == layout.n_levels - 1:
                out.append((0, 0, 0, level))
    return out


def _coord_from_indices(
    layout: AxisLayout,
    meta: ImageVolumeMeta,
    t: int,
    c: int,
    z: int,
    level: int,
) -> tuple[int, ...]:
    n_slots = len(meta.sizes_2003) if meta.sizes_2003 else len(layout.slot_by_axis)  # type: ignore[attr-defined]
    slot_vals = [0] * n_slots
    for ax, slot_i in layout.slot_by_axis.items():
        slot_vals[slot_i] = {"t": t, "c": c, "z": z}[ax]
    return (0, 0, *slot_vals, level)


def materialize_ets(
    skel_path: Path,
    ets_path: Path,
    meta: ImageVolumeMeta,
    *,
    planes: PlanesArg = "default",
) -> None:
    nominal, _ = read_skel(skel_path)
    idx = read_ets_index(ets_path)
    idx.validate_grid()
    h = idx.header
    layout = _layout_from_meta(meta, idx)
    axis_order = _axis_order_from_meta(meta)
    height = meta.height
    width = meta.width

    level_bytes = [
        (height >> lv) * (width >> lv) * h.dtype.itemsize for lv in range(layout.n_levels)
    ]
    to_fill = _plane_indices_to_fill(layout, planes=planes, level_bytes=level_bytes)
    fill_coords = {_coord_from_indices(layout, meta, t, c, z, lv) for t, c, z, lv in to_fill}

    if h.compression == 3 and importlib.util.find_spec("imagecodecs") is None:
        raise MissingCodecsExtraError(
            "JPEG 2000 materialisation requires the codecs extra (imagecodecs)"
        )

    with open(ets_path, "r+b") as f:
        _fill_tiles(f, idx, layout, fill_coords, axis_order, height, width, nominal)


def _fill_tiles(
    f: BinaryIO,
    idx: EtsIndex,
    layout: AxisLayout,
    fill_coords: set[tuple[int, ...]],
    axis_order: list[str],
    height: int,
    width: int,
    nominal: int,
) -> None:
    """Write each filled tile as it is rendered; JPEG 2000 tiles go after the
    nominal end with a relocated copy of the tile table."""
    h = idx.header
    table = bytearray()
    if h.compression == 3:
        f.seek(h.tile_table_offset)
        table = bytearray(f.read(h.table_end - h.tile_table_offset))
    cursor = nominal
    relocated = False

    for ti in range(h.ntiles):
        row = idx.coordinates[ti]
        coord = tuple(int(x) for x in row)
        if coord not in fill_coords:
            continue
        tx, ty = coord[0], coord[1]
        level = coord[-1]
        slot_part = coord[2:-1]
        t = c = z = 0
        for ax, slot_i in layout.slot_by_axis.items():
            val = slot_part[slot_i] if slot_i < len(slot_part) else 0
            if ax == "t":
                t = val
            elif ax == "c":
                c = val
            elif ax == "z":
                z = val

        tile = expected_tile_pixels(
            height=height,
            width=width,
            dtype=h.dtype,
            t=t,
            c=c,
            z=z,
            level=level,
            axis_order=axis_order,
            tile_x=tx,
            tile_y=ty,
            tile_w=h.tile_width,
            tile_h=h.tile_height,
            origin_x=h.origin_x,
            origin_y=h.origin_y,
        )

        if h.compression == 3:
            import imagecodecs

            enc = bytes(imagecodecs.jpeg2k_encode(tile, level=0))
            f.seek(cursor)
            f.write(enc)
            base = ti * h.record_size + 4 + 4 * h.ndim
            struct.pack_into("<QI", table, base, cursor, len(enc))
            cursor += len(enc)
            relocated = True
        else:
            f.seek(int(idx.offsets[ti]))
            f.write(tile.tobytes())

    if relocated:
        f.seek(cursor)
        f.write(table)
        f.seek(32)
        f.write(struct.pack("<I", cursor))


def materialize(
    fixture_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    planes: PlanesArg = "default",
) -> Path:
    fixture = Path(fixture_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    anon_src = fixture / "anon.vsi"
    if not anon_src.is_file():
        raise VsiError(f"missing fixture {anon_src}")
    anon_dest = out / "anon.vsi"
    shutil.copy2(anon_src, anon_dest)

    skel_root = fixture / "_anon_"
    if not skel_root.is_dir():
        return anon_dest

    meta: ImageVolumeMeta | None = None
    for skel in skel_root.glob("stack*/*.skel"):
        stack_name = skel.parent.name
        frame_name = skel.stem
        ets_dest = out / "_anon_" / stack_name / frame_name
        _write_ets_shell(skel, ets_dest)
        if meta is None:
            meta = load_vsi(str(anon_dest))
        materialize_ets(skel, ets_dest, meta, planes=planes)

    return anon_dest


__all__ = ["materialize", "materialize_ets"]
