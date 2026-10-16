from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bioio_vsi import open_vsi
from bioio_vsi._tree import child, collection_root, kids, load_tree
from bioio_vsi.tests.conftest import COMMITTED_VSI, has_imagecodecs, public_samples_root
from bioio_vsi.tests.pixel_hash import all_planes_sha256

TIME_SERIES = Path("ome-cellsens/zenodo-17590655/metadataTest_01.vsi")
TIME_SERIES_SHA = "48f65504213995d7b0e5e12daca96ec020f580d2497717c87eaaf7b69a4bd934"


def _sample(path: Path) -> Path | None:
    root = public_samples_root()
    full = root / path
    return full if full.is_file() else None


def test_public_samples_have_dimension_identifiers() -> None:
    paths = [COMMITTED_VSI]
    ts = public_samples_root() / TIME_SERIES
    if ts.is_file():
        paths.append(ts)
    for path in paths:
        tree = load_tree(str(path))
        coll = collection_root(tree)
        for vol in kids(coll, 2001):
            for ord_i, dd in enumerate(kids(vol, 2007)):
                ps = kids(dd, 2008)
                if not ps:
                    continue
                assert dd.second is not None, f"{path}: dimension {ord_i} missing identifier"
                kinds = []
                for ch_i, p in enumerate(ps):
                    assert p.second is not None, f"{path}: channel {ch_i} missing identifier"
                    kc = child(p, 2023)
                    if kc is not None:
                        kinds.append(int(kc.value))
                if kinds:
                    assert len(set(kinds)) == 1


@pytest.mark.skipif(_sample(TIME_SERIES) is None, reason="no public sample")
def test_time_series_metadata() -> None:
    vsi = _sample(TIME_SERIES)
    assert vsi is not None
    img = open_vsi(str(vsi))
    assert img.shape_tcz == (5, 2, 18)
    ps = img.pixel_size_um
    assert ps is not None
    assert ps.x == pytest.approx(0.065)
    assert ps.y == pytest.approx(0.065)
    assert img.z_step_um == pytest.approx(0.2)
    assert img.camera_bit_depth == 12
    assert img.format_version == (3, 32)
    times = img.plane_times_ms
    assert times is not None and times.shape == (5, 2, 18)
    along_t = times[:, 0, 0]
    steps = np.diff(along_t)
    assert np.all(steps > 14_000) and np.all(steps < 16_000)


@pytest.mark.skipif(
    not has_imagecodecs() or _sample(TIME_SERIES) is None,
    reason="needs imagecodecs and public sample",
)
def test_time_series_all_planes_hash() -> None:
    vsi = _sample(TIME_SERIES)
    assert vsi is not None
    img = open_vsi(str(vsi))
    assert all_planes_sha256(img) == TIME_SERIES_SHA


@pytest.mark.skipif(not has_imagecodecs(), reason="needs imagecodecs")
def test_slideio_all_planes() -> None:
    slideio = pytest.importorskip("slideio")
    vsi = _sample(TIME_SERIES)
    if vsi is None:
        pytest.skip("no public sample")
    img = open_vsi(str(vsi))
    scene = slideio.open_slide(str(vsi)).get_scene(0)
    t, c, z = img.shape_tcz
    for ti in range(t):
        for ci in range(c):
            for zi in range(z):
                ours = img.read_plane(t=ti, c=ci, z=zi)
                block = np.asarray(scene.read_block(slices=(zi, zi + 1), frames=(ti, ti + 1)))
                assert np.array_equal(ours, block[:, :, ci])
