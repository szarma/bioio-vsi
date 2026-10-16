from __future__ import annotations

import numpy as np
import pytest

from bioio_vsi import open_vsi
from bioio_vsi.tests.conftest import COMMITTED_VSI, has_imagecodecs
from bioio_vsi.tests.pixel_hash import all_planes_sha256

COMMITTED_SHA = "b4ae93610966b4abfb6287cfbdd50128824dd0ce719263c46441da7b1c4da632"


def test_committed_metadata() -> None:
    img = open_vsi(str(COMMITTED_VSI))
    assert img.shape_tcz == (1, 2, 11)
    assert img.sizes["t"] == 1 and img.sizes["z"] == 11 and img.sizes["c"] == 2
    ps = img.pixel_size_um
    assert ps is not None
    assert ps.x == pytest.approx(0.10833333333333334)
    assert ps.y == pytest.approx(0.10833333333333334)
    assert img.z_step_um == 1.0
    assert img.z_start_um == pytest.approx(5074.94)
    assert img.channel_names == ["C405", "C488"]
    assert img.emission_wavelengths_nm == [455.0, 518.0]
    assert img.dtype == np.uint16
    assert img.format_version == (3, 22)
    assert img.format_build == 21199
    zpos = img.plane_z_positions_um
    assert zpos is not None and zpos.shape == (1, 2, 11)
    along_z = zpos[0, 0, :]
    assert along_z[0] == pytest.approx(5074.94)
    assert along_z[1] == pytest.approx(5075.94)
    assert along_z[-1] == pytest.approx(5074.94 + 10.0)


@pytest.mark.skipif(not has_imagecodecs(), reason="JPEG 2000 needs imagecodecs")
def test_committed_all_planes_hash() -> None:
    img = open_vsi(str(COMMITTED_VSI))
    assert all_planes_sha256(img) == COMMITTED_SHA


@pytest.mark.skipif(not has_imagecodecs(), reason="JPEG 2000 needs imagecodecs")
def test_committed_slideio_all_planes() -> None:
    slideio = pytest.importorskip("slideio")
    img = open_vsi(str(COMMITTED_VSI))
    scene = slideio.open_slide(str(COMMITTED_VSI)).get_scene(0)
    _t, c, z = img.shape_tcz
    for ci in range(c):
        for zi in range(z):
            block = np.asarray(scene.read_block(slices=(zi, zi + 1)))
            assert np.array_equal(img.read_plane(c=ci, z=zi), block[:, :, ci])


def test_open_reads_headers_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opening indexes the ETS tile table only; no tile bytes are decoded until read."""
    import bioio_vsi._stack as stack_mod

    decode_calls = 0
    real_decode = stack_mod.VsiImage._decode_tile

    def counted_decode(self, coord):  # type: ignore[no-untyped-def]
        nonlocal decode_calls
        decode_calls += 1
        return real_decode(self, coord)

    monkeypatch.setattr(stack_mod.VsiImage, "_decode_tile", counted_decode)
    img = open_vsi(str(COMMITTED_VSI))
    assert img.n_levels == 1
    assert decode_calls == 0
