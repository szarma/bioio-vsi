"""ETS header, tile table, pyramid grid, and extent validation."""

from __future__ import annotations

import struct
import time
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from bioio_vsi import describe, open_ets, open_vsi
from bioio_vsi._ets import inspect_ets_header, read_ets_index
from bioio_vsi.errors import (
    DuplicateTileError,
    IncompleteTileGridError,
    InvalidStructureError,
    SizeMismatchError,
    UnsupportedColourSpaceError,
    UnsupportedFeatureError,
    UnsupportedTileLayoutError,
    VsiError,
)
from bioio_vsi.reader import Reader
from bioio_vsi.tests.synthetic import make_stack, write_1025_pyramid, write_ets, write_minimal_vsi


def test_explicit_error_path_preserved() -> None:
    exc = InvalidStructureError("broken geometry", path="/tmp/context-file.ets")
    assert exc.path == "/tmp/context-file.ets"
    assert "context-file.ets" in str(exc)


def test_header_table_order_refused(tmp_path: Path) -> None:
    vsi, ets = make_stack(tmp_path, shape=(1, 1, 1, 16, 16), tile_size=16)
    data = bytearray(ets.read_bytes())
    struct.pack_into("<Q", data, 32, 64)
    ets.write_bytes(data)
    with pytest.raises(InvalidStructureError, match="overlaps second header"):
        read_ets_index(str(ets))


def test_grid_mutations_refused(tmp_path: Path) -> None:
    vsi, ets = make_stack(tmp_path, shape=(1, 1, 1, 1500, 1500), tile_size=512, n_levels=3)
    mutations = 0
    data0 = bytearray(ets.read_bytes())
    tabpos = struct.unpack_from("<Q", data0, 32)[0]
    ndim = struct.unpack_from("<I", data0, 12)[0]
    rec = 4 + 4 * ndim + 8 + 4 + 4
    ntiles = struct.unpack_from("<I", data0, 40)[0]
    for rec_i in range(ntiles):
        for coord_pos in range(ndim):
            data = bytearray(data0)
            coord_off = tabpos + rec_i * rec + 4 + 4 * coord_pos
            val = struct.unpack_from("<i", data, coord_off)[0]
            struct.pack_into("<i", data, coord_off, val + 1)
            ets.write_bytes(data)
            with pytest.raises((IncompleteTileGridError, DuplicateTileError)):
                open_vsi(str(vsi))
            with pytest.raises((IncompleteTileGridError, DuplicateTileError)):
                open_ets(str(ets))
            mutations += 1
        data = bytearray(data0)
        off = tabpos + rec_i * rec
        del data[off : off + rec]
        struct.pack_into("<I", data, 40, ntiles - 1)
        ets.write_bytes(data)
        with pytest.raises(IncompleteTileGridError):
            open_vsi(str(vsi))
        mutations += 1
        ets.write_bytes(data0)

    assert mutations == ntiles * (ndim + 1)


def test_pyramid_1025_floor_grid_accepted(tmp_path: Path) -> None:
    vsi, _ = write_1025_pyramid(tmp_path / "floor", stored_padding=False)
    img = open_vsi(str(vsi))
    assert img.read_plane(level=1).shape == (512, 512)


def test_pyramid_1025_stored_padding_accepted(tmp_path: Path) -> None:
    vsi, _ = write_1025_pyramid(tmp_path / "stored", stored_padding=True)
    img = open_vsi(str(vsi))
    assert img.read_plane(level=1).shape == (512, 512)


def test_pyramid_1025_invalid_grid_refused(tmp_path: Path) -> None:
    vsi, ets = write_1025_pyramid(tmp_path, stored_padding=True)
    data = bytearray(ets.read_bytes())
    tabpos = struct.unpack_from("<Q", data, 32)[0]
    ndim = struct.unpack_from("<I", data, 12)[0]
    rec = 4 + 4 * ndim + 8 + 4 + 4
    ntiles = struct.unpack_from("<I", data, 40)[0]
    off = tabpos
    del data[off : off + rec]
    struct.pack_into("<I", data, 40, ntiles - 1)
    ets.write_bytes(data)
    with pytest.raises(IncompleteTileGridError):
        open_vsi(str(vsi))


def test_pyramid_1025_one_level_too_many_refused(tmp_path: Path) -> None:
    vsi, _ = write_1025_pyramid(tmp_path, stored_padding=True, extra_level=3)
    with pytest.raises(IncompleteTileGridError):
        open_vsi(str(vsi))


def test_validation_timing_under_budget(tmp_path: Path) -> None:
    index_timings: dict[int, float] = {}
    orphan_timings: dict[int, float] = {}
    for n_planes in (2000, 4000, 8000):
        z = n_planes
        vsi, ets = make_stack(tmp_path / f"p{n_planes}", shape=(1, 1, z, 8, 8), tile_size=8)
        t0 = time.perf_counter()
        read_ets_index(str(ets))
        index_timings[n_planes] = time.perf_counter() - t0
        t1 = time.perf_counter()
        open_ets(str(ets))
        orphan_timings[n_planes] = time.perf_counter() - t1
    assert index_timings[8000] < 0.2
    assert orphan_timings[8000] < 0.2


def _fixture_paths(tmp_path: Path, name: str) -> tuple[Path, Path]:
    if name == "floor_1025":
        return write_1025_pyramid(tmp_path / "floor", stored_padding=False)
    if name == "stored_1025":
        return write_1025_pyramid(tmp_path / "stored", stored_padding=True)
    if name == "single_level":
        return make_stack(tmp_path / "single", shape=(1, 1, 1, 64, 64), tile_size=64, n_levels=1)
    if name == "pyramid_1500":
        return make_stack(
            tmp_path / "p1500",
            shape=(1, 1, 1, 1500, 1500),
            tile_size=512,
            n_levels=3,
        )
    raise ValueError(name)


@pytest.mark.parametrize(
    ("fixture_name", "expected_levels", "check_pixel_scale"),
    [
        ("floor_1025", (0, 1), False),
        ("stored_1025", (0, 1, 2), False),
        ("single_level", (0,), False),
        ("pyramid_1500", (0, 1, 2), True),
    ],
)
def test_pyramid_levels_advertised_and_readable(
    tmp_path: Path,
    fixture_name: str,
    expected_levels: tuple[int, ...],
    check_pixel_scale: bool,
) -> None:
    vsi, ets = _fixture_paths(tmp_path, fixture_name)
    levels = expected_levels

    vsi_img = open_vsi(str(vsi))
    ets_img = open_ets(str(ets))
    assert tuple(range(vsi_img.n_levels)) == levels
    assert tuple(range(ets_img.n_levels)) == levels

    reader = Reader(str(vsi))
    assert reader.resolution_levels == levels
    dims = reader.resolution_level_dims

    n_slots = len(ets_img.slot_sizes)
    slot0 = tuple(0 for _ in range(n_slots))

    for lv in levels:
        shape_vsi = vsi_img.level_shape(lv)
        shape_ets = ets_img.level_shape(lv)
        assert shape_vsi == shape_ets
        assert dims[lv] == shape_vsi

        vsi_img.read_plane(level=lv)
        ets_img.read_plane_slots(slot0, level=lv)
        reader.get_image_dask_data("YX", T=0, C=0, Z=0).compute()

        if check_pixel_scale:
            r = Reader(str(vsi))
            r.set_resolution_level(lv)
            px = r.physical_pixel_sizes
            scale = 2**lv
            assert pytest.approx(scale) == px.X
            assert pytest.approx(scale) == px.Y


def test_empty_tile_table_refused(tmp_path: Path) -> None:
    vsi, ets = make_stack(tmp_path, shape=(1, 1, 1, 16, 16), tile_size=16)
    data = bytearray(ets.read_bytes())
    struct.pack_into("<I", data, 40, 0)
    ets.write_bytes(data)
    with pytest.raises(InvalidStructureError, match="empty"):
        read_ets_index(str(ets))


def test_zero_tile_width_refused(tmp_path: Path) -> None:
    vsi, ets = make_stack(tmp_path, shape=(1, 1, 1, 16, 16), tile_size=16)
    data = bytearray(ets.read_bytes())
    struct.pack_into("<I", data, 64 + 28, 0)
    ets.write_bytes(data)
    with pytest.raises(InvalidStructureError, match="tile width"):
        read_ets_index(str(ets))


def test_negative_tile_coordinate_refused(tmp_path: Path) -> None:
    vsi, ets = make_stack(tmp_path, shape=(1, 1, 1, 16, 16), tile_size=16)
    data = bytearray(ets.read_bytes())
    tabpos = struct.unpack_from("<Q", data, 32)[0]
    struct.pack_into("<i", data, tabpos + 4, -1)
    ets.write_bytes(data)
    with pytest.raises(InvalidStructureError, match="non-negative"):
        read_ets_index(str(ets))


def test_orphan_header_width_mismatch_refused(tmp_path: Path) -> None:
    vsi, ets = make_stack(tmp_path, shape=(1, 1, 1, 8, 8), tile_size=8)
    data = bytearray(ets.read_bytes())
    struct.pack_into("<I", data, 64 + 188, 9)
    ets.write_bytes(data)
    with pytest.raises((SizeMismatchError, IncompleteTileGridError)):
        open_ets(str(ets))


def test_tile_offset_into_header_refused(tmp_path: Path) -> None:
    vsi, ets = make_stack(tmp_path, shape=(1, 1, 1, 32, 32), tile_size=32)
    data = bytearray(ets.read_bytes())
    tabpos = struct.unpack_from("<Q", data, 32)[0]
    ndim = struct.unpack_from("<I", data, 12)[0]
    struct.pack_into("<Q", data, tabpos + 4 + 4 * ndim, 0)
    ets.write_bytes(data)
    with pytest.raises(InvalidStructureError, match="before pixel data"):
        open_vsi(str(vsi))


def test_orphan_origin_offset_exposed_without_warning(tmp_path: Path) -> None:
    _, ets = make_stack(tmp_path, shape=(1, 1, 1, 32, 32), tile_size=32, origin=(-8, -8))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        orphan = open_ets(str(ets))
    assert orphan.origin_offset == (-8, -8)


def _stored_grid(shape_yx: tuple[int, int], origin: tuple[int, int]) -> np.ndarray:
    sy, sx = shape_yx[0] - origin[1], shape_yx[1] - origin[0]
    return np.arange(sy * sx, dtype=np.uint16).reshape(sy, sx)


def test_origin_offset_shifts_plane_vsi_and_orphan(tmp_path: Path) -> None:
    origin = (-3, -1)
    vsi, ets = make_stack(tmp_path, shape=(1, 1, 1, 40, 50), tile_size=16, origin=origin)
    want = _stored_grid((40, 50), origin)[1 : 1 + 40, 3 : 3 + 50]
    img = open_vsi(str(vsi))
    assert img.level_shape(0)[3:] == (40, 50)
    np.testing.assert_array_equal(img.read_plane(), want)
    np.testing.assert_array_equal(img.read_region(5, 7, 20, 30), want[5:25, 7:37])
    orphan = open_ets(str(ets))
    assert orphan.origin_offset == origin
    np.testing.assert_array_equal(orphan.read_plane_slots([0, 0, 0]), want)
    np.testing.assert_array_equal(
        orphan.read_region(5, 7, 20, 30, slot_coords=[0, 0, 0]), want[5:25, 7:37]
    )


def test_origin_offset_missing_shifted_tile_refused(tmp_path: Path) -> None:
    # A 32x32 image in 16-px tiles needs 2x2 tiles at origin 0, but 3x3 once shifted by 1.
    _, ets = make_stack(tmp_path, shape=(1, 1, 1, 32, 32), tile_size=16, origin=(-1, -1))
    data = bytearray(ets.read_bytes())
    assert read_ets_index(ets).header.ntiles == 9
    struct.pack_into("<I", data, 64 + 188, 48)  # declare a wider image than the grid covers
    ets.write_bytes(data)
    with pytest.raises(IncompleteTileGridError):
        read_ets_index(ets)


def test_origin_offset_exposes_only_level_zero(tmp_path: Path) -> None:
    vsi, ets = make_stack(
        tmp_path, shape=(1, 1, 1, 64, 64), tile_size=16, n_levels=3, origin=(-4, -2)
    )
    assert read_ets_index(ets).n_levels > 1
    img = open_vsi(str(vsi))
    assert img.n_levels == 1
    with pytest.raises(IndexError):
        img.read_plane(level=1)
    assert open_ets(str(ets)).n_levels == 1


def test_zero_origin_keeps_pyramid(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 64, 64), tile_size=16, n_levels=3)
    assert open_vsi(str(vsi)).n_levels == 3


def test_orphan_header_slot_mismatch_refused(tmp_path: Path) -> None:
    vsi, ets = make_stack(tmp_path, shape=(1, 1, 3, 16, 16), tile_size=16)
    data = bytearray(ets.read_bytes())
    struct.pack_into("<I", data, 64 + 188 + 4 * 3, 1)
    ets.write_bytes(data)
    with pytest.raises(
        (SizeMismatchError, IncompleteTileGridError),
        match="slot sizes|declared volume",
    ):
        open_ets(str(ets))


def test_ets_colour_space_and_depth_refused(tmp_path: Path) -> None:
    vsi, ets = make_stack(tmp_path, shape=(1, 1, 1, 16, 16), tile_size=16)
    data = bytearray(ets.read_bytes())
    struct.pack_into("<I", data, 64 + 16, 999)
    struct.pack_into("<I", data, 64 + 36, 2)
    ets.write_bytes(data)
    with pytest.raises((UnsupportedColourSpaceError, UnsupportedTileLayoutError)):
        open_vsi(str(vsi))


def test_thin_pyramid_refused_at_open(tmp_path: Path) -> None:
    vsi = tmp_path / "thin" / "syn.vsi"
    ets = vsi.parent / "_syn_" / "stack1" / "frame_t_0.ets"
    planes = {
        (0, lv): np.full(
            (max(1, (2 + (1 << lv) - 1) >> lv), (33 + (1 << lv) - 1) >> lv),
            lv + 1,
            dtype=np.uint16,
        )
        for lv in range(4)
    }
    write_ets(
        ets,
        axis_order=["C"],
        axis_sizes={"C": 1},
        planes=cast("dict[tuple[int, ...], np.ndarray[Any, Any]]", planes),
        tile_size=8,
    )
    write_minimal_vsi(
        vsi,
        stack_id=1,
        width=33,
        height=2,
        axis_order=["C"],
        axis_sizes={"C": 1},
        channel_names=["C0"],
    )
    with pytest.raises(UnsupportedFeatureError, match="zero floor size"):
        open_vsi(str(vsi))
    with pytest.raises(UnsupportedFeatureError, match="zero floor size"):
        open_ets(str(ets))


def test_oversized_extent_refused_before_enumeration(tmp_path: Path) -> None:
    _, ets = make_stack(tmp_path, shape=(1, 1, 1, 8, 8), tile_size=8)
    blob = bytearray(ets.read_bytes())
    struct.pack_into("<I", blob, 64 + 196, 1_000_000_000)
    ets.write_bytes(blob)
    with pytest.raises(VsiError):
        open_ets(str(ets))


def test_describe_truncated_table_notes_invalid(tmp_path: Path) -> None:
    vsi, ets = make_stack(tmp_path, shape=(1, 1, 1, 16, 16), tile_size=16)
    data = bytearray(ets.read_bytes())
    ets.write_bytes(data[:-20])
    text = describe(str(vsi))
    assert "malformed" in text or "truncated" in text.lower()
    with pytest.raises(VsiError):
        read_ets_index(str(ets))


def test_inspect_truncated_table_raises(tmp_path: Path) -> None:
    vsi, ets = make_stack(tmp_path, shape=(1, 1, 1, 16, 16), tile_size=16)
    data = bytearray(ets.read_bytes())
    ets.write_bytes(data[:-20])
    with pytest.raises(VsiError):
        inspect_ets_header(str(ets))


def test_describe_present_includes_index_not_validated(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 8, 8), tile_size=8)
    text = describe(str(vsi))
    assert "index not validated" in text
