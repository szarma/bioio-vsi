from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from bioio import BioImage
from bioio_base.test_utilities import run_image_file_checks

from bioio_vsi import open_vsi
from bioio_vsi._ets import read_ets_index
from bioio_vsi._tree import VsiTree
from bioio_vsi.errors import MissingPixelFileError, MultiImageError
from bioio_vsi.reader import Reader
from bioio_vsi.tests.conftest import COMMITTED_VSI, has_imagecodecs, public_samples_root
from bioio_vsi.tests.synthetic import make_stack, write_minimal_vsi

TIME_SERIES = Path("ome-cellsens/zenodo-17590655/metadataTest_01.vsi")


def _public_vsi(rel: Path) -> Path | None:
    full = public_samples_root() / rel
    return full if full.is_file() else None


@pytest.mark.skipif(not has_imagecodecs(), reason="JPEG 2000 needs imagecodecs")
def test_committed_bioio_standard_checks() -> None:
    run_image_file_checks(
        Reader,
        COMMITTED_VSI,
        set_scene="001 C405, C488",
        expected_scenes=("001 C405, C488",),
        expected_current_scene="001 C405, C488",
        expected_metadata_type=VsiTree,
        reader_kwargs={},
        expected_shape=(1, 2, 11, 1682, 1645),
        expected_dtype=np.dtype(np.uint16),
        expected_dims_order="TCZYX",
        expected_channel_names=["C405", "C488"],
        expected_physical_pixel_sizes=(
            1.0,
            0.10833333333333334,
            0.10833333333333334,
        ),
    )


@pytest.mark.skipif(not has_imagecodecs(), reason="JPEG 2000 needs imagecodecs")
def test_bioimage_entry_point() -> None:
    img = BioImage(str(COMMITTED_VSI))
    assert img.reader.__class__.__name__ == "Reader"
    plane = img.get_image_dask_data("YX", T=0, C=0, Z=0).compute()
    assert plane.shape == (1682, 1645)


@pytest.mark.skipif(not has_imagecodecs(), reason="JPEG 2000 needs imagecodecs")
def test_bioio_pixels_match_core() -> None:
    core = open_vsi(str(COMMITTED_VSI))
    reader = Reader(str(COMMITTED_VSI))
    t, c, z = core.shape_tcz
    for ti in range(t):
        for ci in range(c):
            for zi in range(z):
                from_core = core.read_plane(t=ti, c=ci, z=zi)
                from_bioio = reader.get_image_dask_data("YX", T=ti, C=ci, Z=zi).compute()
                np.testing.assert_array_equal(from_core, from_bioio)


def test_dask_does_not_reparse_ets_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 64, 64), tile_size=32)
    calls = 0
    real = read_ets_index

    def counted(path, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return real(path, **kwargs)

    monkeypatch.setattr("bioio_vsi._stack.read_ets_index", counted)
    reader = Reader(str(vsi))
    plane = reader.get_image_dask_data("YX", C=0, Z=0)
    calls = 0
    plane.compute()
    assert calls == 0


@pytest.mark.skipif(not has_imagecodecs(), reason="JPEG 2000 needs imagecodecs")
def test_dask_plane_read_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    import bioio_vsi.reader as reader_mod

    nbytes = 0
    real = reader_mod.read_tile_from_spec

    def counted(spec):  # type: ignore[no-untyped-def]
        nonlocal nbytes
        nbytes += spec.length
        return real(spec)

    monkeypatch.setattr(reader_mod, "read_tile_from_spec", counted)
    reader = Reader(str(COMMITTED_VSI))
    dask_vol = reader.get_image_dask_data("ZYX")
    assert nbytes == 0
    dask_vol[0].compute()
    one_plane = nbytes
    dask_vol.compute()
    all_planes = nbytes
    assert one_plane > 0
    assert all_planes >= one_plane


def test_pyramid_resolution_levels(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 1500, 1500), tile_size=512, n_levels=3)
    reader = Reader(str(vsi))
    assert reader.resolution_levels == (0, 1, 2)
    dims = reader.resolution_level_dims
    assert dims[0][-2:] == (1500, 1500)
    assert dims[1][-2:] == (750, 750)
    assert dims[2][-2:] == (375, 375)
    reader.set_resolution_level(1)
    px = reader.physical_pixel_sizes
    assert pytest.approx(2.0) == px.Y
    assert pytest.approx(2.0) == px.X


def test_negative_z_step_reports_absolute_z(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 3, 32, 32), tile_size=32)
    write_minimal_vsi(
        vsi,
        stack_id=1,
        width=32,
        height=32,
        axis_order=["T", "Z", "C"],
        axis_sizes={"T": 1, "Z": 3, "C": 1},
        channel_names=["C0"],
        z_step=-0.5,
    )
    reader = Reader(str(vsi))
    assert pytest.approx(0.5) == reader.physical_pixel_sizes.Z
    core = open_vsi(str(vsi))
    assert core.z_step_um == pytest.approx(-0.5)


def test_time_interval_regular(tmp_path: Path) -> None:
    times = [0.0, 100.0, 200.0, 300.0]
    vsi, _ = make_stack(
        tmp_path,
        shape=(4, 1, 1, 16, 16),
        tile_size=16,
        plane_times_ms=times,
    )
    reader = Reader(str(vsi))
    assert reader.time_interval == timedelta(milliseconds=100.0)


def test_time_interval_empty_on_public_series() -> None:
    vsi = _public_vsi(TIME_SERIES)
    if vsi is None:
        pytest.skip("public time series not on disk")
    reader = Reader(str(vsi))
    assert reader.time_interval is None


def test_bioio_refusals(tmp_path: Path) -> None:
    vsi, ets = make_stack(tmp_path, shape=(1, 1, 1, 16, 16), tile_size=16)
    ets.unlink()
    with pytest.raises(MissingPixelFileError):
        _ = BioImage(str(vsi)).data

    vsi2, _ = make_stack(
        tmp_path / "m",
        shape=(1, 1, 1, 16, 16),
        tile_size=16,
        extra_images=[10000],
    )
    with pytest.raises(MultiImageError):
        _ = BioImage(str(vsi2)).data


def test_bioio_shifted_origin_levels_and_pixels(tmp_path: Path) -> None:
    origin = (-3, -1)
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 40, 50), tile_size=16, n_levels=3, origin=origin)
    stored = np.arange(41 * 53, dtype=np.uint16).reshape(41, 53)
    want = stored[1:41, 3:53]
    reader = Reader(str(vsi))
    assert reader.resolution_levels == (0,)
    np.testing.assert_array_equal(reader.get_image_data("YX"), want)
    np.testing.assert_array_equal(reader.get_image_dask_data("YX").compute(), want)


def test_non_local_fs_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fsspec.spec import AbstractFileSystem

    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 16, 16), tile_size=16)

    class RemoteFS(AbstractFileSystem):
        protocol = "remote"

    monkeypatch.setattr(
        "bioio_vsi.reader.pathlike_to_fs",
        lambda *args, **kwargs: (RemoteFS(), str(vsi)),
    )
    with pytest.raises(TypeError, match="local"):
        Reader(str(vsi))


@pytest.mark.skipif(
    not has_imagecodecs() or _public_vsi(TIME_SERIES) is None,
    reason="needs imagecodecs and public time series",
)
def test_public_time_series_bioio_checks() -> None:
    vsi = _public_vsi(TIME_SERIES)
    assert vsi is not None
    core = open_vsi(str(vsi))
    run_image_file_checks(
        Reader,
        vsi,
        set_scene=core.image_name or "stack1",
        expected_scenes=(core.image_name or "stack1",),
        expected_current_scene=core.image_name or "stack1",
        expected_shape=(5, 2, 18) + core.level_shape(0)[3:],
        expected_dtype=core.dtype,
        expected_dims_order="TCZYX",
        expected_channel_names=[n or f"C{i}" for i, n in enumerate(core.channel_names)],
        expected_physical_pixel_sizes=(
            abs(core.z_step_um) if core.z_step_um is not None else None,
            core.pixel_size_um.y if core.pixel_size_um else None,
            core.pixel_size_um.x if core.pixel_size_um else None,
        ),
        expected_metadata_type=VsiTree,
        reader_kwargs={},
    )
