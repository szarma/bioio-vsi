from __future__ import annotations

import pickle
import struct
from pathlib import Path

import numpy as np
import pytest
from bioio import BioImage

from bioio_vsi import describe, open_ets, open_vsi
from bioio_vsi.errors import (
    InvalidIdentifierError,
    UndeclaredCoordinateError,
    UnsupportedFeatureError,
)
from bioio_vsi.reader import Reader
from bioio_vsi.tests.conftest import public_samples_root
from bioio_vsi.tests.synthetic import _build_volume, _field_inline, _field_nested, _int_vector


def _legacy_volume(stack_id: int, role: int, width: int, height: int) -> bytes:
    boundary = _int_vector(2053, [0, 0, width, height])
    external = _field_nested(2018, [boundary, _field_inline(20005, 1)])
    return _field_nested(
        2001,
        [
            _field_nested(2002, [external], second=0),
            _field_nested(2005, [_field_inline(2074, role)]),
        ],
        second=stack_id,
    )


def _write_legacy_ets(path: Path, payload: bytes, *, slot: int = 0) -> None:
    table_offset = 64 + 228 + len(payload)
    main = struct.pack(
        "<4sIIIQIIQIIQII",
        b"SIS\0",
        64,
        2,
        4,
        64,
        228,
        0,
        table_offset,
        1,
        0,
        table_offset + 36,
        0,
        0,
    )
    secondary = struct.pack("<4s9I", b"ETS\0", 0x30003, 2, 3, 4, 3, 90, 512, 512, 1)
    entry = struct.pack("<I4iQII", 4, 0, 0, slot, 0, 64 + 228, len(payload), 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(main + secondary.ljust(228, b"\0") + payload + entry)


def test_legacy_scanner_uses_paired_vsi_geometry(tmp_path: Path) -> None:
    imagecodecs = pytest.importorskip("imagecodecs")
    vsi = tmp_path / "scanner.vsi"
    collection = _field_nested(
        2000,
        [
            _int_vector(2035, [0x30007, 10555]),
            _legacy_volume(3, 1, 13, 21),
            _legacy_volume(10, 0, 61, 34),
            _field_nested(2001, [_field_nested(2005, [_field_inline(2074, 4)])], second=11),
        ],
        second=0,
    )
    vsi.write_bytes(b"II*\0" + bytes(4) + _build_volume([collection]))

    tile = np.empty((512, 512, 3), dtype=np.uint8)
    tile[:] = (23, 51, 122)
    main_ets = tmp_path / "_scanner_" / "stack10" / "frame_t.ets"
    main_payload = bytes(imagecodecs.jpeg2k_encode(tile, level=0))
    _write_legacy_ets(main_ets, main_payload)

    tile[:] = (37, 14, 89)
    overview_ets = tmp_path / "_scanner_" / "stack3" / "frame_t.ets"
    _write_legacy_ets(overview_ets, bytes(imagecodecs.jpeg2k_encode(tile, level=0)))

    assert open_vsi(vsi).level_shape() == (1, 1, 1, 34, 61, 3)
    assert open_vsi(vsi, stack_id=3).level_shape() == (1, 1, 1, 21, 13, 3)
    assert describe(vsi, redact=True).count("pixel file: PRESENT") == 2
    with pytest.raises(InvalidIdentifierError, match="stack ID 11"):
        open_vsi(vsi, stack_id=11)
    with pytest.raises(UnsupportedFeatureError, match="companion VSI"):
        open_ets(main_ets)

    reader = Reader(vsi)
    assert reader.scenes == ("stack10", "stack3")
    main = reader.get_image_dask_data("YXS")
    reader.set_scene("stack3")
    overview = reader.get_image_dask_data("YXS")
    assert main.shape == (34, 61, 3)
    assert overview.shape == (21, 13, 3)
    np.testing.assert_array_equal(main[0, 0].compute(), [23, 51, 122])
    np.testing.assert_array_equal(overview[0, 0].compute(), [37, 14, 89])
    restored = pickle.loads(pickle.dumps(reader))
    assert restored.current_scene == "stack3"
    np.testing.assert_array_equal(restored.get_image_data("YXS"), overview.compute())

    _write_legacy_ets(main_ets, main_payload, slot=1)
    with pytest.raises(UndeclaredCoordinateError, match="coordinate without a declared extent"):
        open_vsi(vsi)


def test_os1_regions_match_slideio() -> None:
    vsi = public_samples_root() / "openslide-olympus" / "OS-1.vsi"
    if not vsi.is_file():
        pytest.skip("OS-1 public scanner sample not on disk")
    pytest.importorskip("imagecodecs")
    slideio = pytest.importorskip("slideio")
    oracle = slideio.open_slide(str(vsi), "VSI").get_scene(0)
    core = open_vsi(vsi, stack_id=10001)
    image = BioImage(str(vsi))
    assert image.scenes == ("stack10001", "stack1")
    assert image.current_scene == "stack10001"
    assert image.dims.order == "TCZYXS"
    assert image.shape == (1, 1, 1, 76963, 66982, 3)
    assert image.resolution_levels == tuple(range(9))
    rgb = image.get_image_dask_data("YXS")
    for x, y, w, h in [(500, 500, 30, 30), (66966, 76947, 16, 16)]:
        expected = oracle.read_block((x, y, w, h))
        np.testing.assert_array_equal(core.read_region(y, x, h, w), expected)
        np.testing.assert_array_equal(rgb[y : y + h, x : x + w].compute(), expected)

    image.set_resolution_level(8)
    assert image.get_image_data("YXS").shape == (300, 261, 3)
    image.set_scene("stack1")
    assert image.current_resolution_level == 0
    assert image.shape == (1, 1, 1, 13196, 6753, 3)
    assert image.resolution_levels == tuple(range(6))
    assert image.physical_pixel_sizes.X == 3.4887564563687983
