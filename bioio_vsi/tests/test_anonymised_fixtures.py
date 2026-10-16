"""Committed anonymised fixtures: materialise, then compare with values recorded
from the original acquisitions (see docs/samples.md)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bioio_vsi import open_ets, open_vsi
from bioio_vsi._fixture_pixels import synthetic_plane
from bioio_vsi._materialize import _axis_order_from_meta, materialize
from bioio_vsi._vsi import load_vsi
from bioio_vsi.tests.content_check import (
    measured_numbers,
    preview_pixels_left,
    varied_byte_arrays,
)

ROOT = Path(__file__).parent / "resources" / "anonymised"
EXPECTED = json.loads((ROOT / "expected.json").read_text())


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_no_specimen_preview_or_content_arrays(name: str) -> None:
    data = (ROOT / name / "anon.vsi").read_bytes()
    assert preview_pixels_left(data) == 0
    assert varied_byte_arrays(data) == []
    assert measured_numbers(data) == []


@pytest.fixture(params=sorted(EXPECTED), ids=str)
def fixture(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[str, Path]:
    name = request.param
    return name, materialize(ROOT / name, tmp_path)


def test_metadata_matches_original(fixture: tuple[str, Path]) -> None:
    name, vsi = fixture
    expected = EXPECTED[name]
    img = open_vsi(str(vsi))
    if "stitched" in expected:
        # Only level 0 is exposed when the origin is non-zero; the file stores more.
        st = expected["stitched"]
        assert dict(img.sizes) == st["sizes"]
        assert img.n_levels == st["n_levels"]
        assert img._index.n_levels == st["stored_levels"]
        assert str(img.dtype) == st["dtype"]
        assert list(img.level_shape(0)[3:]) == st["shape_yx"]
        return
    want = expected["vsi"]
    assert dict(img.sizes) == want["sizes"]
    assert img.n_levels == want["n_levels"]
    assert str(img.dtype) == want["dtype"]
    assert img.pixel_size_um is not None
    assert [img.pixel_size_um.x, img.pixel_size_um.y] == want["pixel_size_um"]
    assert img.z_step_um == want["z_step_um"]
    assert img.z_start_um == want["z_start_um"]
    assert list(img.emission_wavelengths_nm) == want["emission_wavelengths_nm"]
    assert img.camera_bit_depth == want["camera_bit_depth"]
    z = np.asarray(img.plane_z_positions_um, dtype=float)
    assert [z.flat[0], z.flat[-1]] == want["plane_z_um_first_last"]
    assert (
        float(np.nanmax(np.asarray(img.plane_times_ms, dtype=float))) == want["plane_time_ms_max"]
    )


def test_pyramid_matches_original(fixture: tuple[str, Path]) -> None:
    name, vsi = fixture
    ets = next(vsi.parent.rglob("frame_t*.ets"))
    img = open_ets(str(ets))
    assert img.n_levels == EXPECTED[name]["ets"]["n_levels"]
    assert list(img.origin_offset) == EXPECTED[name]["ets"]["origin_offset"]


@pytest.mark.parametrize(
    "name", sorted(n for n in EXPECTED if tuple(EXPECTED[n]["ets"]["origin_offset"]) == (0, 0))
)
def test_first_plane_is_the_synthetic_picture(name: str, tmp_path: Path) -> None:
    vsi = materialize(ROOT / name, tmp_path)
    img = open_vsi(str(vsi))
    meta = load_vsi(str(vsi))
    want = synthetic_plane(
        meta.height,
        meta.width,
        dtype=img.dtype,
        t=0,
        c=0,
        z=0,
        level=0,
        axis_order=_axis_order_from_meta(meta),
    )
    np.testing.assert_array_equal(img.read_plane(), want)
