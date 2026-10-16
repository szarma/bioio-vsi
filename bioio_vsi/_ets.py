"""ETS pixel-file headers and tile index."""

from __future__ import annotations

import itertools
import os
import struct
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bioio_vsi._codecs import SUPPORTED_COMPRESSION
from bioio_vsi._paths import resolve_path
from bioio_vsi.errors import (
    DuplicateTileError,
    IncompleteTileGridError,
    InvalidStructureError,
    SizeMismatchError,
    TruncatedFileError,
    UnsupportedColourSpaceError,
    UnsupportedCompressionError,
    UnsupportedPixelTypeError,
    UnsupportedTileLayoutError,
)

MAIN_HDR = struct.Struct("<4sIIIQIIQIIQII")
ETS_HDR_PREFIX = struct.Struct("<4s9I")

# File encoding is little-endian; decode explicitly, expose native-endian arrays.
_FILE_DTYPES: dict[int, np.dtype] = {
    2: np.dtype("<u1"),
    4: np.dtype("<u2"),
}
PIXEL_TYPES: dict[int, np.dtype] = {
    2: np.dtype(np.uint8),
    4: np.dtype(np.uint16),
}

SUPPORTED_COLOUR_SPACE = 1
SUPPORTED_TILE_DEPTH = 1
SUPPORTED_SAMPLES_PER_TILE = 1


def _level_extent(height: int, width: int, level: int) -> tuple[int, int]:
    """Pyramid level size (floor division by 2**level; docs/design.md)."""
    return height >> level, width >> level


def _level_extent_stored(height: int, width: int, level: int) -> tuple[int, int]:
    """Stored pyramid size: halve with rounding up at each step (docs/design.md)."""
    h, w = height, width
    for _ in range(level):
        h = (h + 1) >> 1
        w = (w + 1) >> 1
    return h, w


def _levels_for_extent_rule(
    height: int,
    width: int,
    tile_h: int,
    tile_w: int,
    extent_at_level: Callable[[int, int, int], tuple[int, int]],
) -> int:
    """Levels 0 .. L until the level extent fits in one tile, under either size rule."""
    level = 0
    while True:
        ly, lx = extent_at_level(height, width, level)
        ty = (ly + tile_h - 1) // tile_h
        tx = (lx + tile_w - 1) // tile_w
        if ty == 1 and tx == 1:
            return level + 1
        level += 1
        if ly < 1 or lx < 1:
            return level


def _pyramid_level_count(height: int, width: int, tile_h: int, tile_w: int) -> int:
    """Levels through the first single-tile grid under floor or stored-padding extents."""
    n_floor = _levels_for_extent_rule(height, width, tile_h, tile_w, _level_extent)
    n_stored = _levels_for_extent_rule(height, width, tile_h, tile_w, _level_extent_stored)
    return max(n_floor, n_stored)


def _spatial_tile_set(height: int, width: int, tile_h: int, tile_w: int) -> set[tuple[int, int]]:
    return {
        (tx, ty)
        for ty in range((height + tile_h - 1) // tile_h)
        for tx in range((width + tile_w - 1) // tile_w)
    }


def _expected_plane_keys(extent_slots: tuple[int, ...]) -> list[tuple[int, ...]]:
    if not extent_slots:
        return [()]
    ranges = [range(int(s)) for s in extent_slots]
    return [tuple(p) for p in itertools.product(*ranges)]


def _validate_ets_layout(
    path: str,
    fsize: int,
    *,
    main_end: int,
    addpos: int,
    addsize: int,
    tabpos: int,
    table_end: int,
) -> None:
    """Main header, second header, pixel data, and tile table must not overlap."""
    if addpos < main_end:
        raise InvalidStructureError(
            f"second header at {addpos} overlaps main header (ends at {main_end})",
            path=path,
        )
    second_end = addpos + addsize
    if second_end > fsize:
        raise TruncatedFileError(
            f"second header extends past file end ({second_end} > {fsize})",
            path=path,
        )
    if tabpos < second_end:
        raise InvalidStructureError(
            f"tile table at {tabpos} overlaps second header (ends at {second_end})",
            path=path,
        )
    data_end = tabpos
    if data_end > fsize:
        raise TruncatedFileError(
            f"pixel data region ends at {data_end}, past file size {fsize}",
            path=path,
        )
    if table_end > fsize:
        raise TruncatedFileError(
            f"tile table ends at {table_end}, past file size {fsize}",
            path=path,
        )


def _validate_header_geometry(header: EtsHeader) -> None:
    path = header.path
    if header.ntiles == 0:
        raise InvalidStructureError("tile table is empty", path=path)
    if header.tile_width <= 0:
        raise InvalidStructureError(
            f"tile width must be positive, got {header.tile_width}", path=path
        )
    if header.tile_height <= 0:
        raise InvalidStructureError(
            f"tile height must be positive, got {header.tile_height}", path=path
        )
    if header.extent_width <= 0 or header.extent_height <= 0:
        raise InvalidStructureError(
            f"image extent must be positive, got {header.extent_width}x{header.extent_height}",
            path=path,
        )
    if header.ndim < 3:
        raise InvalidStructureError(
            f"coordinate dimensionality must be at least 3, got {header.ndim}", path=path
        )


@dataclass(frozen=True)
class EtsHeader:
    path: str
    file_size: int
    ndim: int
    tile_table_offset: int
    ntiles: int
    pixel_type: int
    ncomp: int
    colour_space: int
    compression: int
    quality: int
    tile_width: int
    tile_height: int
    tile_depth: int
    use_pyramid: int
    origin_x: int
    origin_y: int
    extent_width: int
    extent_height: int
    extent_slots: tuple[int, ...]
    table_end: int
    record_size: int

    @property
    def dtype(self) -> np.dtype:
        return PIXEL_TYPES[self.pixel_type]

    @property
    def stored_width(self) -> int:
        """Width of the stored level-0 grid: the image starts at (-origin_x, -origin_y)."""
        return self.extent_width - self.origin_x

    @property
    def stored_height(self) -> int:
        return self.extent_height - self.origin_y

    @property
    def samples_shape_suffix(self) -> tuple[int, ...]:
        return (self.ncomp,) if self.ncomp > 1 else ()


@dataclass
class EtsIndex:
    header: EtsHeader
    coordinates: np.ndarray  # shape (ntiles, ndim), int32
    offsets: np.ndarray  # uint64
    lengths: np.ndarray  # uint32
    coord_to_index: dict[tuple[int, ...], int]
    # Observed pyramid depth (max level index + 1); set by validate_grid().
    n_levels: int = 1

    def validate_grid(self) -> None:
        """Complete spatial grid per level and plane, no duplicates."""
        h = self.header
        path = h.path
        _validate_header_geometry(h)
        if len(self.coord_to_index) != h.ntiles:
            raise DuplicateTileError(
                f"tile table lists duplicate coordinates "
                f"({h.ntiles} records, {len(self.coord_to_index)} unique)",
                path=path,
            )
        coords = self.coordinates
        if coords.shape[0] != h.ntiles:
            raise InvalidStructureError(
                f"tile table has {coords.shape[0]} coordinate rows, header declares {h.ntiles}",
                path=path,
            )
        if np.any(coords < 0):
            bad = coords[coords < 0]
            raise InvalidStructureError(
                f"tile coordinates must be non-negative, got {bad[0].tolist()}",
                path=path,
            )

        nd = h.ndim
        th, tw = h.tile_height, h.tile_width
        # A complete grid needs at least this many tiles. Checking the count first
        # keeps a header that declares an absurd extent from being enumerated.
        base_tiles = -(-h.stored_height // th) * -(-h.stored_width // tw)
        for slot_size in h.extent_slots:
            base_tiles *= int(slot_size)
        if base_tiles > h.ntiles:
            raise IncompleteTileGridError(
                f"header extents need at least {base_tiles} tiles, tile table has {h.ntiles}",
                path=path,
            )
        if h.use_pyramid:
            n_floor = _levels_for_extent_rule(
                h.stored_height, h.stored_width, th, tw, _level_extent
            )
            n_stored = _levels_for_extent_rule(
                h.stored_height, h.stored_width, th, tw, _level_extent_stored
            )
            observed_max = int(coords[:, -1].max())
            n_levels = observed_max + 1
            if n_levels not in (n_floor, n_stored):
                raise IncompleteTileGridError(
                    "tile table pyramid depth does not match a supported termination "
                    f"(have {n_levels} levels, expected {n_floor} by floor extents or "
                    f"{n_stored} by stored-padding extents)",
                    path=path,
                )
        else:
            n_levels = 1
            if np.any(coords[:, -1] != 0):
                raise IncompleteTileGridError(
                    "tile table lists pyramid levels but header declares no pyramid",
                    path=path,
                )
        plane_keys = _expected_plane_keys(h.extent_slots)
        expected_pairs = {
            (level_i, plane_key) for level_i in range(n_levels) for plane_key in plane_keys
        }

        level_spatial_options: dict[int, list[set[tuple[int, int]]]] = {}
        for level_i in range(n_levels):
            ly_floor, lx_floor = _level_extent(h.stored_height, h.stored_width, level_i)
            if ly_floor == 0 or lx_floor == 0:
                from bioio_vsi.errors import UnsupportedFeatureError

                raise UnsupportedFeatureError(
                    f"pyramid level {level_i} has zero floor size ({ly_floor}x{lx_floor})",
                    path=path,
                )
            if level_i == 0:
                level_spatial_options[level_i] = [
                    _spatial_tile_set(ly_floor, lx_floor, th, tw),
                ]
            else:
                ly_store, lx_store = _level_extent_stored(h.stored_height, h.stored_width, level_i)
                level_spatial_options[level_i] = [
                    _spatial_tile_set(ly_floor, lx_floor, th, tw),
                    _spatial_tile_set(ly_store, lx_store, th, tw),
                ]

        groups: dict[tuple[int, tuple[int, ...]], set[tuple[int, int]]] = defaultdict(set)
        for row in coords:
            level_i = int(row[-1])
            plane_key = tuple(int(x) for x in row[2 : nd - 1])
            groups[(level_i, plane_key)].add((int(row[0]), int(row[1])))

        observed_pairs = set(groups.keys())
        if observed_pairs != expected_pairs:
            missing_pairs = sorted(expected_pairs - observed_pairs)
            extra_pairs = sorted(observed_pairs - expected_pairs)
            raise IncompleteTileGridError(
                "tile table does not match declared volume: "
                f"expected {len(expected_pairs)} (level, plane) groups, "
                f"have {len(observed_pairs)}"
                + (f"; missing {missing_pairs[:3]}" if missing_pairs else "")
                + (f"; unexpected {extra_pairs[:3]}" if extra_pairs else ""),
                path=path,
            )

        for (level_i, plane_key), actual in groups.items():
            options = level_spatial_options[level_i]
            if not any(actual == expected for expected in options):
                ly_floor, lx_floor = _level_extent(h.stored_height, h.stored_width, level_i)
                missing = sorted(options[0] - actual)
                extra = sorted(actual - options[0])
                raise IncompleteTileGridError(
                    f"level {level_i} plane {plane_key}: "
                    f"expected spatial grid for {ly_floor}x{lx_floor} "
                    f"(or stored-padding variant at this level), have {len(actual)} tiles"
                    + (f"; missing tile positions {missing[:5]}" if missing else "")
                    + (f"; unexpected positions {extra[:5]}" if extra else ""),
                    path=path,
                )

        self.n_levels = n_levels


def _read_main_header(path: str, fsize: int, blob: bytes) -> dict[str, Any]:
    if len(blob) < 64:
        raise TruncatedFileError("ETS main header shorter than 64 bytes", path=path)
    h = MAIN_HDR.unpack(blob[:64])
    magic, hsize, version, ndim, addpos, addsize = h[0], h[1], h[2], h[3], h[4], h[5]
    tabpos, ntiles = h[7], h[8]
    if magic != b"SIS\x00":
        raise InvalidStructureError(f"bad ETS magic {magic!r}", path=path)
    if hsize != 64:
        raise InvalidStructureError(f"unexpected main header size {hsize}", path=path)
    if addpos + addsize > fsize:
        raise TruncatedFileError("second header extends past file end", path=path)
    if tabpos > fsize:
        raise TruncatedFileError(f"tile table offset {tabpos} past file end", path=path)
    return {
        "version": version,
        "ndim": ndim,
        "addpos": addpos,
        "addsize": addsize,
        "tabpos": tabpos,
        "ntiles": ntiles,
        "fsize": fsize,
    }


def _parse_second_header(path_str: str, second: bytes, *, for_reading: bool) -> dict[str, Any]:
    if len(second) < 40:
        raise TruncatedFileError("second header too short", path=path_str)
    try:
        hdr = ETS_HDR_PREFIX.unpack_from(second, 0)
    except struct.error as exc:
        raise InvalidStructureError("cannot unpack ETS embedded header", path=path_str) from exc
    emagic, _eversion, ptype, ncomp, cspace, comp, quality, tw, th, td = hdr
    if emagic != b"ETS\x00":
        raise InvalidStructureError(f"bad embedded header magic {emagic!r}", path=path_str)

    if ptype not in PIXEL_TYPES:
        raise UnsupportedPixelTypeError(
            f"pixel type code {ptype} is not uint8 (2) or uint16 (4)",
            path=path_str,
        )
    if for_reading:
        if cspace != SUPPORTED_COLOUR_SPACE:
            raise UnsupportedColourSpaceError(
                f"colour space {cspace} is not supported "
                f"(release 1 expects {SUPPORTED_COLOUR_SPACE})",
                path=path_str,
            )
        if int(td) != SUPPORTED_TILE_DEPTH:
            raise UnsupportedTileLayoutError(
                f"tile depth {td} is not supported (release 1 expects {SUPPORTED_TILE_DEPTH})",
                path=path_str,
            )
        if int(ncomp) != SUPPORTED_SAMPLES_PER_TILE:
            raise UnsupportedTileLayoutError(
                f"{ncomp} samples per tile is not supported "
                f"(release 1 expects {SUPPORTED_SAMPLES_PER_TILE})",
                path=path_str,
            )
        if comp not in SUPPORTED_COMPRESSION:
            if comp == 2:
                raise UnsupportedCompressionError(
                    "compression code 2 (JPEG) is not supported in release 1",
                    path=path_str,
                )
            raise UnsupportedCompressionError(
                f"compression code {comp} is not supported",
                path=path_str,
            )

    if len(second) < 188:
        raise TruncatedFileError("second header too short for extent count", path=path_str)
    n_extents = struct.unpack_from("<I", second, 184)[0]
    extent_end = 188 + 4 * n_extents
    if len(second) < extent_end:
        raise InvalidStructureError(
            f"extent count {n_extents} needs {extent_end} bytes in second header, "
            f"have {len(second)}",
            path=path_str,
        )
    try:
        extent_vals = list(struct.unpack_from(f"<{n_extents}I", second, 188))
    except struct.error as exc:
        raise InvalidStructureError("cannot unpack extent list", path=path_str) from exc
    if len(extent_vals) < 2:
        raise InvalidStructureError(f"extent list too short ({extent_vals})", path=path_str)

    use_pyramid = 0
    if len(second) >= 152:
        rest_count = (len(second) - 40) // 4
        if rest_count > 28:
            use_pyramid = struct.unpack_from(f"<{rest_count}I", second, 40)[28]
    origin_x = struct.unpack_from("<i", second, 172)[0] if len(second) >= 176 else 0
    origin_y = struct.unpack_from("<i", second, 176)[0] if len(second) >= 180 else 0

    return {
        "ptype": int(ptype),
        "ncomp": int(ncomp),
        "cspace": int(cspace),
        "comp": int(comp),
        "quality": int(quality),
        "tw": int(tw),
        "th": int(th),
        "td": int(td),
        "use_pyramid": int(use_pyramid),
        "origin_x": int(origin_x),
        "origin_y": int(origin_y),
        "extent_w": int(extent_vals[0]),
        "extent_h": int(extent_vals[1]),
        "extent_slots": tuple(int(x) for x in extent_vals[2:]),
    }


def inspect_ets_header(path: str | Path) -> EtsHeader:
    """Read ETS headers only (no tile payloads). Does not enforce decoding policy."""
    path_str = str(path)
    fsize = os.path.getsize(path_str)
    with open(path_str, "rb") as f:
        main = f.read(64)
        main_d = _read_main_header(path_str, fsize, main)
        addpos = int(main_d["addpos"])
        addsize = int(main_d["addsize"])
        f.seek(addpos)
        second = f.read(addsize)
        if len(second) < addsize:
            raise TruncatedFileError("truncated second header", path=path_str)
        parsed = _parse_second_header(path_str, second, for_reading=False)
        tabpos = int(main_d["tabpos"])
        ntiles = int(main_d["ntiles"])
        rec = 4 + 4 * int(main_d["ndim"]) + 8 + 4 + 4
        table_end = tabpos + rec * ntiles
        _validate_ets_layout(
            path_str,
            fsize,
            main_end=64,
            addpos=addpos,
            addsize=addsize,
            tabpos=tabpos,
            table_end=table_end,
        )
        header = EtsHeader(
            path=path_str,
            file_size=fsize,
            ndim=int(main_d["ndim"]),
            tile_table_offset=tabpos,
            ntiles=ntiles,
            pixel_type=parsed["ptype"],
            ncomp=parsed["ncomp"],
            colour_space=parsed["cspace"],
            compression=parsed["comp"],
            quality=parsed["quality"],
            tile_width=parsed["tw"],
            tile_height=parsed["th"],
            tile_depth=parsed["td"],
            use_pyramid=parsed["use_pyramid"],
            origin_x=parsed["origin_x"],
            origin_y=parsed["origin_y"],
            extent_width=parsed["extent_w"],
            extent_height=parsed["extent_h"],
            extent_slots=parsed["extent_slots"],
            table_end=table_end,
            record_size=rec,
        )
        _validate_header_geometry(header)
        return header


def _validate_tile_payloads(
    path_str: str,
    header: EtsHeader,
    tab: np.ndarray,
    data_start: int,
    tabpos: int,
) -> None:
    if ntiles := header.ntiles:
        offs = tab["off"]
        lens = tab["len"]
        for i in range(ntiles):
            off = int(offs[i])
            length = int(lens[i])
            if off < data_start:
                raise InvalidStructureError(
                    f"tile {i} offset {off} is before pixel data region (starts at {data_start})",
                    path=path_str,
                )
            if off + length > tabpos:
                raise InvalidStructureError(
                    f"tile {i} payload [{off}, {off + length}) extends into tile table at {tabpos}",
                    path=path_str,
                )
            if header.compression == 0:
                itemsize = PIXEL_TYPES[header.pixel_type].itemsize
                expected = (
                    header.tile_height
                    * header.tile_width
                    * header.tile_depth
                    * header.ncomp
                    * itemsize
                )
                if length != expected:
                    raise InvalidStructureError(
                        f"raw tile {i} length {length} != expected {expected} "
                        f"for {header.tile_height}x{header.tile_width} "
                        f"x {header.ncomp} sample(s) x depth {header.tile_depth}",
                        path=path_str,
                    )


def read_ets_index(path: str | Path) -> EtsIndex:
    """Read headers and tile table only (no tile payloads)."""
    path_str = resolve_path(path)
    fsize = os.path.getsize(path_str)
    with open(path_str, "rb") as f:
        main = f.read(64)
        main_d = _read_main_header(path_str, fsize, main)
        ndim = int(main_d["ndim"])
        addpos = int(main_d["addpos"])
        addsize = int(main_d["addsize"])
        data_start = addpos + addsize
        f.seek(addpos)
        second = f.read(addsize)
        if len(second) < addsize:
            raise TruncatedFileError("truncated second header", path=path_str)

        parsed = _parse_second_header(path_str, second, for_reading=True)
        extent_w, extent_h = parsed["extent_w"], parsed["extent_h"]
        extent_slots = parsed["extent_slots"]

        if parsed["origin_x"] > 0 or parsed["origin_y"] > 0:
            raise InvalidStructureError(
                f"ETS origin offset ({parsed['origin_x']}, {parsed['origin_y']}) is positive; "
                "only zero or negative offsets have been observed",
                path=path_str,
            )

        tabpos = int(main_d["tabpos"])
        ntiles = int(main_d["ntiles"])
        rec = 4 + 4 * ndim + 8 + 4 + 4
        table_bytes = rec * ntiles
        table_end = tabpos + table_bytes
        _validate_ets_layout(
            path_str,
            fsize,
            main_end=64,
            addpos=addpos,
            addsize=addsize,
            tabpos=tabpos,
            table_end=table_end,
        )
        f.seek(tabpos)
        raw_tab = f.read(table_bytes)
        if len(raw_tab) < table_bytes:
            raise TruncatedFileError("truncated tile table", path=path_str)

        dtype = np.dtype(
            [
                ("u0", "<u4"),
                ("c", "<i4", (ndim,)),
                ("off", "<u8"),
                ("len", "<u4"),
                ("u1", "<u4"),
            ]
        )
        tab = np.frombuffer(raw_tab, dtype=dtype)

        header = EtsHeader(
            path=path_str,
            file_size=fsize,
            ndim=ndim,
            tile_table_offset=tabpos,
            ntiles=ntiles,
            pixel_type=parsed["ptype"],
            ncomp=parsed["ncomp"],
            colour_space=parsed["cspace"],
            compression=parsed["comp"],
            quality=parsed["quality"],
            tile_width=parsed["tw"],
            tile_height=parsed["th"],
            tile_depth=parsed["td"],
            use_pyramid=parsed["use_pyramid"],
            origin_x=parsed["origin_x"],
            origin_y=parsed["origin_y"],
            extent_width=extent_w,
            extent_height=extent_h,
            extent_slots=extent_slots,
            table_end=table_end,
            record_size=rec,
        )
        _validate_header_geometry(header)

        _validate_tile_payloads(path_str, header, tab, data_start, tabpos)

        coord_to_index: dict[tuple[int, ...], int] = {}
        for i, row in enumerate(tab["c"]):
            key = tuple(int(x) for x in row)
            if key in coord_to_index:
                raise DuplicateTileError(
                    f"duplicate tile at coordinates {key}",
                    path=path_str,
                )
            coord_to_index[key] = i

        index = EtsIndex(
            header=header,
            coordinates=tab["c"].copy(),
            offsets=tab["off"].copy(),
            lengths=tab["len"].copy(),
            coord_to_index=coord_to_index,
        )
        index.validate_grid()
        return index


def sizes_from_index(index: EtsIndex, n_declared: int) -> dict[int, int]:
    """Map ETS slot index (2..) to size from tile coordinate maxima."""
    cmax = index.coordinates.max(axis=0)
    out: dict[int, int] = {}
    for slot_idx in range(2, 2 + n_declared):
        out[slot_idx] = int(cmax[slot_idx]) + 1
    return out


def slot_sizes_from_index(index: EtsIndex) -> list[int]:
    """Per-slot sizes from tile coordinate maxima (slots 2 .. ndim-2)."""
    cmax = index.coordinates.max(axis=0)
    n_slots = index.header.ndim - 3
    return [int(cmax[2 + i]) + 1 for i in range(n_slots)]


def check_orphan_header_consistency(index: EtsIndex) -> None:
    """Cross-check orphan ETS header extents against the tile table."""
    h = index.header
    path = h.path
    from_table = slot_sizes_from_index(index)
    if list(h.extent_slots) != from_table:
        raise SizeMismatchError(
            f"ETS header slot sizes {list(h.extent_slots)} != tile table maxima {from_table}",
            path=path,
        )
    ly, lx = h.stored_height, h.stored_width
    th, tw = h.tile_height, h.tile_width
    level0 = index.coordinates[index.coordinates[:, -1] == 0]
    if level0.size == 0:
        raise IncompleteTileGridError("pyramid level 0 has no tiles", path=path)
    expected_tx = (lx + tw - 1) // tw
    expected_ty = (ly + th - 1) // th
    nd = h.ndim
    plane_groups: dict[tuple[int, ...], list[tuple[int, int]]] = defaultdict(list)
    for row in level0:
        plane_key = tuple(int(x) for x in row[2 : nd - 1])
        plane_groups[plane_key].append((int(row[0]), int(row[1])))
    for _plane_key, tiles in plane_groups.items():
        if not tiles:
            max_tx = max_ty = -1
        else:
            tiles_xy = np.asarray(tiles, dtype=np.int32)
            max_tx = max(int(x) for x in tiles_xy[:, 0])
            max_ty = max(int(x) for x in tiles_xy[:, 1])
        if max_tx + 1 != expected_tx or max_ty + 1 != expected_ty:
            raise SizeMismatchError(
                f"level-0 tile grid ({max_tx + 1}x{max_ty + 1}) does not cover "
                f"header extent {lx}x{ly}",
                path=path,
            )


def check_extents_match(
    path: str,
    index: EtsIndex,
    width: int,
    height: int,
    slot_sizes: list[int],
) -> None:
    h = index.header
    if h.extent_width != width or h.extent_height != height:
        raise SizeMismatchError(
            f"ETS extents {h.extent_width}x{h.extent_height} != VSI size {width}x{height}",
            path=path,
        )
    if list(h.extent_slots) != slot_sizes:
        raise SizeMismatchError(
            f"ETS slot sizes {list(h.extent_slots)} != VSI 2003 sizes {slot_sizes}",
            path=path,
        )
