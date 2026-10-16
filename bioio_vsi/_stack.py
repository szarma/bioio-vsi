"""Read pixels for one VSI image (coordinate slots, pyramid, tiles)."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bioio_vsi import errors
from bioio_vsi._accessors import PixelSizeUm
from bioio_vsi._codecs import decode_tile
from bioio_vsi._ets import (
    EtsIndex,
    check_extents_match,
    check_orphan_header_consistency,
    read_ets_index,
    slot_sizes_from_index,
)
from bioio_vsi._paths import resolve_path
from bioio_vsi._vsi import DIM_TO_AXIS, ImageVolumeMeta


@dataclass(frozen=True)
class TileReadSpec:
    """Enough information to read one tile without reparsing the ETS index."""

    ets_path: str
    offset: int
    length: int
    compression: int
    dtype: np.dtype
    shape: tuple[int, ...]


def read_tile_from_spec(spec: TileReadSpec) -> np.ndarray:
    with open(spec.ets_path, "rb") as f:
        f.seek(spec.offset)
        raw = f.read(spec.length)
    if len(raw) < spec.length:
        raise errors.TruncatedFileError(
            f"truncated tile at offset {spec.offset}",
            path=spec.ets_path,
        )
    return decode_tile(raw, spec.compression, spec.dtype, spec.shape)


def level_shape(height: int, width: int, level: int) -> tuple[int, int]:
    """Decision 6: floor division by 2**level (slideio convention)."""
    return height >> level, width >> level


def _index_arg(value: object, *, what: str) -> int:
    try:
        return operator.index(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{what} must be an integer index, got {value!r}") from exc


@dataclass(frozen=True)
class AxisLayout:
    """Maps logical T/C/Z indices to ETS coordinate slots."""

    slot_by_axis: dict[str, int]
    sizes: dict[str, int]
    n_levels: int


def _axis_extent(layout: AxisLayout, axis: str) -> int:
    return layout.sizes.get(axis, 1)


def _normalize_plane_indices(
    layout: AxisLayout,
    *,
    t: object = 0,
    c: object = 0,
    z: object = 0,
    level: object = 0,
) -> tuple[int, int, int, int]:
    t_i = _index_arg(t, what="t")
    c_i = _index_arg(c, what="c")
    z_i = _index_arg(z, what="z")
    level_i = _index_arg(level, what="level")
    if level_i < 0 or level_i >= layout.n_levels:
        raise IndexError(f"level={level_i} out of range 0..{layout.n_levels - 1}")
    for axis, idx in (("t", t_i), ("c", c_i), ("z", z_i)):
        size = _axis_extent(layout, axis)
        if not (0 <= idx < size):
            raise IndexError(f"{axis}={idx} out of range 0..{size - 1}")
    return t_i, c_i, z_i, level_i


def _normalize_region_args(
    layout: AxisLayout,
    y: object,
    x: object,
    height: object,
    width: object,
    *,
    meta_height: int,
    meta_width: int,
    t: object = 0,
    c: object = 0,
    z: object = 0,
    level: object = 0,
) -> tuple[int, int, int, int, int, int, int, int]:
    t_i, c_i, z_i, level_i = _normalize_plane_indices(layout, t=t, c=c, z=z, level=level)
    y_i = _index_arg(y, what="y")
    x_i = _index_arg(x, what="x")
    h_i = _index_arg(height, what="height")
    w_i = _index_arg(width, what="width")
    ly, lx = level_shape(meta_height, meta_width, level_i)
    if h_i <= 0 or w_i <= 0:
        raise IndexError(f"height and width must be positive, got {h_i}x{w_i}")
    if y_i < 0 or x_i < 0 or y_i + h_i > ly or x_i + w_i > lx:
        raise IndexError(
            f"region ({y_i}, {x_i}, {h_i}, {w_i}) is outside level {level_i} shape ({ly}, {lx})"
        )
    return t_i, c_i, z_i, level_i, y_i, x_i, h_i, w_i


def _origin_is_zero(index: EtsIndex) -> bool:
    return index.header.origin_x == 0 and index.header.origin_y == 0


def _build_layout(meta: ImageVolumeMeta, index: EtsIndex) -> AxisLayout:
    vsi_path = meta.tree.path
    kinds: list[str | None] = [s.kind for s in meta.slots]
    declared = [k for k in kinds if k is not None]
    if any(k is None for k in kinds[: len(declared)]):
        raise errors.InvalidStructureError(
            f"gap in declared dimensions {kinds}",
            path=vsi_path,
        )
    if len(set(declared)) != len(declared):
        raise errors.InvalidStructureError(
            f"duplicate declared dimension in {declared}",
            path=vsi_path,
        )

    nd = index.header.ndim
    cmax = index.coordinates.max(axis=0)
    # Lower levels of a stitched scan are shifted by origin / 2**level, not a whole pixel.
    n_levels = index.n_levels if _origin_is_zero(index) else 1

    slot_by_axis: dict[str, int] = {}
    for i, kind in enumerate(declared):
        if kind not in DIM_TO_AXIS:
            raise errors.UnknownDimensionKindError(
                f"unsupported declared dimension {kind!r}",
                path=vsi_path,
            )
        slot_idx = 2 + i
        slot_by_axis[DIM_TO_AXIS[kind]] = slot_idx

    for j in range(2 + len(declared), nd - 1):
        if int(cmax[j]) != 0:
            raise errors.UndeclaredCoordinateError(
                f"coordinate slot {j} varies (max {int(cmax[j])}) but is not declared in the VSI",
                path=vsi_path,
            )
    if 2 + len(declared) > nd - 1:
        raise errors.InvalidStructureError(
            "more declared dimensions than ETS coordinates",
            path=vsi_path,
        )

    sizes = {axis: int(cmax[slot]) + 1 for axis, slot in slot_by_axis.items()}

    if meta.sizes_2003 is not None:
        expected = [sizes[DIM_TO_AXIS[k]] for k in declared]
        if list(meta.sizes_2003) != expected:
            raise errors.SizeMismatchError(
                f"VSI 2003 sizes {meta.sizes_2003} != tile table {expected}",
                path=vsi_path,
            )

    check_extents_match(
        vsi_path,
        index,
        meta.width,
        meta.height,
        [sizes[DIM_TO_AXIS[k]] for k in declared],
    )

    return AxisLayout(slot_by_axis=slot_by_axis, sizes=sizes, n_levels=n_levels)


class VsiImage:
    """One image from a single-image cellSens VSI file."""

    def __init__(self, meta: ImageVolumeMeta) -> None:
        if meta.ets_path is None:
            raise errors.MissingPixelFileError(
                f"no pixel file for image {meta.stack_id}",
                path=meta.tree.path,
            )
        self._meta = meta
        self._path = resolve_path(meta.ets_path)
        self._index = read_ets_index(self._path)
        self._layout = _build_layout(meta, self._index)
        h = self._index.header
        self._dtype = h.dtype
        self._tw = h.tile_width
        self._th = h.tile_height
        self._ncomp = h.ncomp
        self._shift_y = -h.origin_y
        self._shift_x = -h.origin_x

    @property
    def is_shifted(self) -> bool:
        """True when the image starts inside the stored grid (non-zero stitch origin)."""
        return self._shift_x != 0 or self._shift_y != 0

    @property
    def path(self) -> str:
        return self._meta.tree.path

    @property
    def ets_path(self) -> str:
        return self._path

    @property
    def tree(self) -> Any:
        return self._meta.tree

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def sizes(self) -> dict[str, int]:
        return dict(self._layout.sizes)

    @property
    def shape_tcz(self) -> tuple[int, int, int]:
        s = self._layout.sizes
        return s.get("t", 1), s.get("c", 1), s.get("z", 1)

    @property
    def n_levels(self) -> int:
        return self._layout.n_levels

    def level_shape(self, level: object = 0) -> tuple[int, int, int, int, int]:
        """Return (T, C, Z, Y, X) at the given pyramid level."""
        level_i = _index_arg(level, what="level")
        if level_i < 0 or level_i >= self.n_levels:
            raise IndexError(f"pyramid level {level_i} out of range 0..{self.n_levels - 1}")
        y, x = level_shape(self._meta.height, self._meta.width, level_i)
        t, c, z = self.shape_tcz
        return t, c, z, y, x

    @property
    def pixel_size_um(self) -> PixelSizeUm | None:
        return self._meta.pixel_size_um

    @property
    def z_step_um(self) -> float | None:
        for s in self._meta.slots:
            if s.kind == "Z":
                return s.z_step_um
        return None

    @property
    def z_start_um(self) -> float | None:
        for s in self._meta.slots:
            if s.kind == "Z":
                return s.z_start_um
        return None

    @property
    def plane_z_positions_um(self) -> np.ndarray | None:
        """Per-plane z in µm, shape (T, C, Z); ``None`` when absent.

        Planes without a stored property are ``nan`` at their index.
        """
        return None if self._meta.plane_z_um is None else self._meta.plane_z_um.copy()

    @property
    def plane_times_ms(self) -> np.ndarray | None:
        """Per-plane times in ms, shape (T, C, Z); ``None`` when entirely absent.

        Planes without a stored property are ``nan`` at their index.
        """
        return None if self._meta.plane_time_ms is None else self._meta.plane_time_ms.copy()

    @property
    def channel_names(self) -> list[str | None]:
        for s in self._meta.slots:
            if s.kind == "C":
                return [ch.name for ch in sorted(s.channels, key=lambda c: c.index)]
        return []

    @property
    def channel_alt_names(self) -> list[str | None]:
        for s in self._meta.slots:
            if s.kind == "C":
                return [ch.alt_name for ch in sorted(s.channels, key=lambda c: c.index)]
        return []

    @property
    def emission_wavelengths_nm(self) -> list[float | None]:
        for s in self._meta.slots:
            if s.kind == "C":
                return [ch.emission_nm for ch in sorted(s.channels, key=lambda c: c.index)]
        return []

    @property
    def camera_bit_depth(self) -> int | None:
        return self._meta.camera_bit_depth

    @property
    def image_name(self) -> str | None:
        return self._meta.name

    @property
    def acquisition_time(self) -> float | None:
        return self._meta.acquisition_time

    @property
    def format_version(self) -> tuple[int, int] | None:
        return self._meta.format_version

    @property
    def format_build(self) -> int | None:
        return self._meta.format_build

    @property
    def writer_build(self) -> int | None:
        return self._meta.writer_build

    def _tile_shape(self) -> tuple[int, ...]:
        suffix = (self._ncomp,) if self._ncomp > 1 else ()
        return (self._th, self._tw) + suffix

    def _read_tile_bytes(self, coord: tuple[int, ...]) -> bytes:
        idx = self._index.coord_to_index.get(coord)
        if idx is None:
            raise errors.IncompleteTileGridError(
                f"missing tile at coordinates {coord}",
                path=self._path,
            )
        off = int(self._index.offsets[idx])
        length = int(self._index.lengths[idx])
        if off + length > self._index.header.file_size:
            raise errors.TruncatedFileError(
                f"tile at {coord} extends past file end",
                path=self._path,
            )
        with open(self._path, "rb") as f:
            f.seek(off)
            raw = f.read(length)
        if len(raw) < length:
            raise errors.TruncatedFileError(
                f"truncated tile data at {coord}",
                path=self._path,
            )
        return raw

    def _decode_tile(self, coord: tuple[int, ...]) -> np.ndarray:
        return read_tile_from_spec(self.tile_read_spec(coord))

    def tile_read_spec(self, coord: tuple[int, ...]) -> TileReadSpec:
        idx = self._index.coord_to_index.get(coord)
        if idx is None:
            raise errors.IncompleteTileGridError(
                f"missing tile at coordinates {coord}",
                path=self._path,
            )
        h = self._index.header
        return TileReadSpec(
            ets_path=self._path,
            offset=int(self._index.offsets[idx]),
            length=int(self._index.lengths[idx]),
            compression=h.compression,
            dtype=self._dtype,
            shape=self._tile_shape(),
        )

    def plane_tile_coords(
        self,
        t: int,
        c: int,
        z: int,
        level: int,
    ) -> list[tuple[int, ...]]:
        """All tile coordinates for one logical plane at the given pyramid level."""
        layout = self._layout
        ly, lx = level_shape(self._meta.height, self._meta.width, level)
        th, tw = self._th, self._tw
        nd = self._index.header.ndim
        pos = {"t": t, "c": c, "z": z}
        coords: list[tuple[int, ...]] = []
        for ty in range((ly + th - 1) // th):
            for tx in range((lx + tw - 1) // tw):
                coord = [0] * nd
                coord[0], coord[1], coord[-1] = tx, ty, level
                for axis, slot in layout.slot_by_axis.items():
                    coord[slot] = pos[axis]
                coords.append(tuple(coord))
        return coords

    def read_plane(
        self,
        t: int = 0,
        c: int = 0,
        z: int = 0,
        level: int = 0,
    ) -> np.ndarray:
        t_i, c_i, z_i, level_i = _normalize_plane_indices(self._layout, t=t, c=c, z=z, level=level)
        y_size, x_size = level_shape(self._meta.height, self._meta.width, level_i)
        return self.read_region(0, 0, y_size, x_size, t=t_i, c=c_i, z=z_i, level=level_i)

    def read_region(
        self,
        y: int,
        x: int,
        height: int,
        width: int,
        *,
        t: int = 0,
        c: int = 0,
        z: int = 0,
        level: int = 0,
    ) -> np.ndarray:
        layout = self._layout
        t_i, c_i, z_i, level_i, y_i, x_i, h_i, w_i = _normalize_region_args(
            layout,
            y,
            x,
            height,
            width,
            meta_height=self._meta.height,
            meta_width=self._meta.width,
            t=t,
            c=c,
            z=z,
            level=level,
        )
        y, x, height, width = y_i, x_i, h_i, w_i

        pos = {"t": t_i, "c": c_i, "z": z_i}
        h_ets = self._index.header
        nd = h_ets.ndim
        th, tw = self._th, self._tw
        suffix = (self._ncomp,) if self._ncomp > 1 else ()
        out = np.zeros((height, width) + suffix, dtype=self._dtype)

        # Image pixel (y, x) is stored-grid pixel (y + shift_y, x + shift_x).
        sy = y + self._shift_y
        sx = x + self._shift_x
        y_end = sy + height
        x_end = sx + width
        for ty in range(sy // th, (y_end - 1) // th + 1):
            for tx in range(sx // tw, (x_end - 1) // tw + 1):
                coord = [0] * nd
                coord[0], coord[1], coord[-1] = tx, ty, level_i
                for axis, slot in layout.slot_by_axis.items():
                    coord[slot] = pos[axis]
                tile = self._decode_tile(tuple(coord))
                ya = max(sy, ty * th)
                xa = max(sx, tx * tw)
                yb = min(y_end, (ty + 1) * th)
                xb = min(x_end, (tx + 1) * tw)
                out[ya - sy : yb - sy, xa - sx : xb - sx] = tile[
                    ya - ty * th : yb - ty * th, xa - tx * tw : xb - tx * tw
                ]
        return out

    def __getstate__(self) -> dict[str, Any]:
        return {"meta": self._meta, "path": self._path}

    def __setstate__(self, state: dict[str, Any]) -> None:
        VsiImage.__init__(self, state["meta"])  # type: ignore[misc]


class OrphanEtsImage:
    """Pixel file without a matching VSI.

    Geometry (width, height, slot sizes) comes from the ETS header only. Which
    slot is z, channel, or time cannot be known without the ``.vsi`` metadata.

    The ETS header may declare a stitch ``origin_offset``; the image then starts
    at stored-grid pixel (-origin_x, -origin_y) and only level 0 is exposed.
    Missing per-plane calibration and axis labels remain unknown.
    """

    def __init__(self, path: str) -> None:
        self._path = resolve_path(path)
        self._index = read_ets_index(self._path)
        check_orphan_header_consistency(self._index)
        h = self._index.header
        self._dtype = h.dtype
        self._width = h.extent_width
        self._height = h.extent_height
        self._slot_sizes = slot_sizes_from_index(self._index)
        self._ndim = h.ndim
        self._n_levels = self._index.n_levels if _origin_is_zero(self._index) else 1
        self._tw = h.tile_width
        self._th = h.tile_height
        self._ncomp = h.ncomp
        self._origin_offset = (h.origin_x, h.origin_y)
        self._shift_y = -h.origin_y
        self._shift_x = -h.origin_x

    @property
    def origin_offset(self) -> tuple[int, int]:
        """ETS stitch origin as stored in the header (zero or negative)."""
        return self._origin_offset

    @property
    def path(self) -> str:
        return self._path

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def slot_sizes(self) -> list[int]:
        return list(self._slot_sizes)

    @property
    def n_levels(self) -> int:
        return self._n_levels

    def level_shape(self, level: object = 0) -> tuple[int, ...]:
        level_i = _index_arg(level, what="level")
        if level_i < 0 or level_i >= self._n_levels:
            raise IndexError(f"level={level_i} out of range 0..{self._n_levels - 1}")
        y, x = level_shape(self._height, self._width, level_i)
        sizes = tuple(self._slot_sizes) + (y, x)
        return sizes

    def read_plane_slots(self, slot_coords: Sequence[int], level: object = 0) -> np.ndarray:
        """Read a full frame at ``level`` using raw ETS dimension slot indices."""
        n_slots = self._ndim - 3  # exclude tile x, tile y, pyramid level
        if len(slot_coords) != n_slots:
            raise ValueError(f"expected {n_slots} slot indices, got {len(slot_coords)}")
        level_i = _index_arg(level, what="level")
        if level_i < 0 or level_i >= self._n_levels:
            raise IndexError(f"level={level_i} out of range 0..{self._n_levels - 1}")
        self._check_slot_coords(slot_coords)
        y, x = level_shape(self._height, self._width, level_i)
        return self.read_region(0, 0, y, x, slot_coords=slot_coords, level=level_i)

    def _check_slot_coords(self, slot_coords: Sequence[int]) -> None:
        for i, (coord, size) in enumerate(zip(slot_coords, self._slot_sizes, strict=True)):
            idx = _index_arg(coord, what=f"slot {i} coordinate")
            if not (0 <= idx < size):
                raise IndexError(
                    f"slot {i} coordinate {idx} out of range 0..{size - 1} "
                    f"(header/table sizes {self._slot_sizes})"
                )

    def read_region(
        self,
        y: object,
        x: object,
        height: object,
        width: object,
        *,
        slot_coords: Sequence[int],
        level: object = 0,
    ) -> np.ndarray:
        nd = self._ndim
        level_i = _index_arg(level, what="level")
        if level_i < 0 or level_i >= self._n_levels:
            raise IndexError(f"level={level_i} out of range 0..{self._n_levels - 1}")
        y_i = _index_arg(y, what="y")
        x_i = _index_arg(x, what="x")
        h_i = _index_arg(height, what="height")
        w_i = _index_arg(width, what="width")
        ly, lx = level_shape(self._height, self._width, level_i)
        if h_i <= 0 or w_i <= 0:
            raise IndexError(f"height and width must be positive, got {h_i}x{w_i}")
        if y_i < 0 or x_i < 0 or y_i + h_i > ly or x_i + w_i > lx:
            raise IndexError(
                f"region ({y_i}, {x_i}, {h_i}, {w_i}) is outside level {level_i} shape ({ly}, {lx})"
            )
        height, width = h_i, w_i
        y, x = y_i + self._shift_y, x_i + self._shift_x
        self._check_slot_coords(slot_coords)
        th, tw = self._th, self._tw
        suffix = (self._ncomp,) if self._ncomp > 1 else ()
        out = np.zeros((height, width) + suffix, dtype=self._dtype)
        y_end, x_end = y + height, x + width
        for ty in range(y // th, (y_end - 1) // th + 1):
            for tx in range(x // tw, (x_end - 1) // tw + 1):
                coord = [tx, ty]
                coord.extend(_index_arg(v, what="slot coordinate") for v in slot_coords)
                coord.append(level_i)
                if len(coord) != nd:
                    raise ValueError("internal coordinate length mismatch")
                tile = self._decode(tuple(coord))
                ya, xa = max(y, ty * th), max(x, tx * tw)
                yb, xb = min(y_end, (ty + 1) * th), min(x_end, (tx + 1) * tw)
                out[ya - y : yb - y, xa - x : xb - x] = tile[
                    ya - ty * th : yb - ty * th, xa - tx * tw : xb - tx * tw
                ]
        return out

    def _decode(self, coord: tuple[int, ...]) -> np.ndarray:
        idx = self._index.coord_to_index.get(coord)
        if idx is None:
            raise errors.IncompleteTileGridError(
                f"missing tile at coordinates {coord}",
                path=self._path,
            )
        off = int(self._index.offsets[idx])
        length = int(self._index.lengths[idx])
        with open(self._path, "rb") as f:
            f.seek(off)
            raw = f.read(length)
        h = self._index.header
        shape = (self._th, self._tw) + ((self._ncomp,) if self._ncomp > 1 else ())
        return decode_tile(raw, h.compression, self._dtype, shape)

    def __getstate__(self) -> dict[str, Any]:
        return {"path": self._path}

    def __setstate__(self, state: dict[str, Any]) -> None:
        OrphanEtsImage.__init__(self, state["path"])  # type: ignore[misc]
