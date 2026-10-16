"""Typed metadata accessors, units, z/time, and interpreted-path refusal."""

from __future__ import annotations

import struct
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from bioio_base import constants, dimensions
from bioio_base.types import PhysicalPixelSizes

from bioio_vsi import describe, open_vsi
from bioio_vsi._accessors import PixelSizeUm
from bioio_vsi._tree import FieldNode, load_tree, walk
from bioio_vsi._vsi import load_vsi
from bioio_vsi.errors import (
    AmbiguousMetadataError,
    InvalidIdentifierError,
    InvalidMetadataError,
    UnknownDimensionKindError,
    VsiError,
)
from bioio_vsi.reader import Reader
from bioio_vsi.tests import synthetic as syn
from bioio_vsi.tests.interpreted_paths import (
    CORRUPT,
    interpreted_cases,
    mutation_transform,
    prepare_vsi,
)
from bioio_vsi.tests.synthetic import make_stack, write_minimal_vsi
from bioio_vsi.tests.tree_rewrite import rewrite_vsi


def test_string_pixel_size_refused(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 8, 8), tile_size=8)
    rewrite_vsi(vsi, lambda n: syn._string(2019, "19") if n.tag == 2019 else None)
    with pytest.raises(InvalidMetadataError):
        open_vsi(str(vsi))


def test_string_dimension_sizes_refused(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(2, 2, 2, 8, 8), tile_size=8)
    rewrite_vsi(vsi, lambda n: syn._string(2003, "222") if n.tag == 2003 else None)
    with pytest.raises(InvalidMetadataError):
        open_vsi(str(vsi))


def test_fractional_boundary_rect_refused(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 8, 8), tile_size=8)
    rewrite_vsi(
        vsi,
        lambda n: syn._double_vector(2053, [0.9, 0, 8.9, 8.9]) if n.tag == 2053 else None,
    )
    with pytest.raises(InvalidMetadataError):
        open_vsi(str(vsi))


def test_fractional_dimension_kind_refused(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 2, 8, 8), tile_size=8)
    rewrite_vsi(
        vsi,
        lambda n: (
            syn._field_data(2023, 10, struct.pack("<d", 1.9))
            if n.tag == 2023 and n.value == 1
            else None
        ),
    )
    with pytest.raises(InvalidMetadataError):
        open_vsi(str(vsi))


def test_duplicate_pixel_size_is_ambiguous(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 8, 8), tile_size=8)

    def trans(n: FieldNode) -> bytes | None:
        if n.tag == 2005:
            return syn._field_nested(
                2005,
                [syn._double_vector(2019, [1.0, 1.0]), syn._double_vector(2019, [9.0, 9.0])],
            )
        return None

    rewrite_vsi(vsi, trans)
    with pytest.raises(AmbiguousMetadataError):
        open_vsi(str(vsi))


def _tree_contains_corrupt(tree: Any) -> bool:
    for node in walk(tree.root):
        if node.value == CORRUPT:
            return True
        if isinstance(node.value, str) and CORRUPT in node.value:
            return True
        if isinstance(node.value, (list, tuple)) and CORRUPT in node.value:
            return True
    return False


@pytest.mark.parametrize(("path_id", "mutation"), interpreted_cases())
def test_interpreted_metadata_refused(
    tmp_path: Path,
    path_id: str,
    mutation: str,
) -> None:
    vsi = prepare_vsi(tmp_path / path_id.replace("/", "_"), path_id)
    rewrite_vsi(vsi, mutation_transform(path_id, mutation), with_enc=True)
    tree = load_tree(str(vsi))
    if mutation == "string":
        assert _tree_contains_corrupt(tree), "string mutation not present in tree"
    with pytest.raises(VsiError):
        open_vsi(str(vsi))
    text = describe(str(vsi))
    assert CORRUPT not in text


def test_dimension_kind_conflict_refused(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 2, 1, 16, 16), tile_size=16)
    write_minimal_vsi(
        vsi,
        stack_id=1,
        width=16,
        height=16,
        axis_order=["T", "Z", "C"],
        axis_sizes={"T": 1, "Z": 1, "C": 2},
        channel_names=["C0", "C1"],
        kind_codes={"C": [4, 1]},
    )
    with pytest.raises(UnknownDimensionKindError, match="conflicting"):
        open_vsi(str(vsi))


def test_missing_dimension_identifiers_refused(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 2, 1, 16, 16), tile_size=16)
    write_minimal_vsi(
        vsi,
        stack_id=1,
        width=16,
        height=16,
        axis_order=["T", "Z", "C"],
        axis_sizes={"T": 1, "Z": 1, "C": 2},
        channel_names=["C0", "C1"],
        dimension_seconds=(None, None, None),
        channel_seconds=(None, None),
    )
    with pytest.raises((InvalidIdentifierError, AmbiguousMetadataError)):
        open_vsi(str(vsi))


def test_dimension_identifier_order_required(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 2, 3, 16, 16), tile_size=16, axis_order="CZT")
    write_minimal_vsi(
        vsi,
        stack_id=1,
        width=16,
        height=16,
        axis_order=list("CZT"),
        axis_sizes={"C": 2, "Z": 3, "T": 1},
        channel_names=["C0", "C1"],
        dimension_seconds=[0, 2, 1],
    )
    with pytest.raises(InvalidIdentifierError):
        open_vsi(str(vsi))


def test_duplicate_channel_identifier_refused(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 2, 1, 16, 16), tile_size=16)
    write_minimal_vsi(
        vsi,
        stack_id=1,
        width=16,
        height=16,
        axis_order=["T", "Z", "C"],
        axis_sizes={"T": 1, "Z": 1, "C": 2},
        channel_names=["C0", "C1"],
        channel_seconds=[0, 0],
    )
    with pytest.raises((InvalidIdentifierError, AmbiguousMetadataError)):
        open_vsi(str(vsi))


def test_anisotropic_physical_pixel_sizes_bioio(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 16, 16), tile_size=16)
    write_minimal_vsi(
        vsi,
        stack_id=1,
        width=16,
        height=16,
        axis_order=["T", "Z", "C"],
        axis_sizes={"T": 1, "Z": 1, "C": 1},
        channel_names=["C0"],
        pixel_size=(2.0, 5.0),
    )
    core = open_vsi(str(vsi))
    assert core.pixel_size_um == PixelSizeUm(x=2.0, y=5.0)
    reader = Reader(str(vsi))
    px: PhysicalPixelSizes = reader.physical_pixel_sizes
    assert pytest.approx(2.0) == px.X
    assert pytest.approx(5.0) == px.Y


def test_multi_z_positions_from_step(tmp_path: Path) -> None:
    vsi, _ = make_stack(
        tmp_path,
        shape=(1, 1, 4, 32, 32),
        axis_order="TZC",
        tile_size=32,
    )
    img = open_vsi(str(vsi))
    zpos = img.plane_z_positions_um
    assert zpos is not None
    along_z = zpos[0, 0, :]
    np.testing.assert_allclose(along_z, [0.0, 1.0, 2.0, 3.0])


def test_uniform_z_t_coords_multi_channel_time(tmp_path: Path) -> None:
    z_vals = [1.0, 2.0, 3.0]
    t_vals = [10.0, 20.0]
    times = [t_vals[ti] for _zi in range(3) for _ci in range(2) for ti in range(2)]
    vsi, _ = make_stack(
        tmp_path,
        shape=(2, 2, 3, 8, 8),
        tile_size=8,
        plane_times_ms=times,
    )
    write_minimal_vsi(
        vsi,
        stack_id=1,
        width=8,
        height=8,
        axis_order=["T", "Z", "C"],
        axis_sizes={"T": 2, "Z": 3, "C": 2},
        channel_names=["A", "B"],
        z_start=1.0,
        z_step=1.0,
        plane_times_ms=times,
    )
    reader = Reader(str(vsi))
    da = reader.xarray_dask_data
    assert dimensions.DimensionNames.SpatialZ in da.coords
    assert dimensions.DimensionNames.Time in da.coords
    np.testing.assert_allclose(da.coords["Z"].values, z_vals)
    np.testing.assert_allclose(da.coords["T"].values, t_vals)
    assert da.coords["Z"].attrs.get("units") == "micrometer"
    assert da.coords["T"].attrs.get("units") == "millisecond"


def test_varying_time_omits_coord(tmp_path: Path) -> None:
    vsi, _ = make_stack(
        tmp_path,
        shape=(2, 2, 1, 8, 8),
        tile_size=8,
        plane_times_ms=[0.0, 50.0, 200.0, 250.0],
    )
    reader = Reader(str(vsi))
    da = reader.xarray_dask_data
    assert "T" not in da.coords


def test_plane_time_nan_kept_as_unknown(tmp_path: Path) -> None:
    vsi, _ = make_stack(
        tmp_path,
        shape=(2, 1, 1, 8, 8),
        tile_size=8,
        plane_times_ms=[0.0, float("nan")],
    )
    image = open_vsi(str(vsi))
    times = image.plane_times_ms
    assert times is not None
    assert times.ravel()[0] == 0.0 and np.isnan(times.ravel()[1])
    assert Reader(str(vsi)).time_interval is None


def test_bioio_time_coord_only_when_uniform(tmp_path: Path) -> None:
    times = [0.0, 100.0, 200.0, 300.0]
    vsi, _ = make_stack(
        tmp_path,
        shape=(4, 1, 1, 8, 8),
        tile_size=8,
        plane_times_ms=times,
    )
    reader = Reader(str(vsi))
    da = reader.xarray_dask_data
    assert "T" in da.coords
    vsi2, _ = make_stack(
        tmp_path / "v",
        shape=(2, 1, 2, 8, 8),
        tile_size=8,
        plane_times_ms=[0.0, 50.0, 200.0, 250.0],
    )
    reader2 = Reader(str(vsi2))
    da2 = reader2.xarray_dask_data
    assert "T" not in da2.coords
    proc = da2.attrs.get(constants.METADATA_PROCESSED)
    assert proc is not None and proc.get("plane_time_ms") is not None


def test_time_interval_none_when_step_missing(tmp_path: Path) -> None:
    vsi, _ = make_stack(
        tmp_path,
        shape=(3, 1, 1, 8, 8),
        tile_size=8,
        plane_times_ms=[0.0, 100.0, 250.0],
    )
    reader = Reader(str(vsi))
    assert reader.time_interval is None


def test_partial_plane_metadata_keeps_pixels(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 3, 16, 16), tile_size=16)
    write_minimal_vsi(
        vsi,
        stack_id=1,
        width=16,
        height=16,
        axis_order=["T", "Z", "C"],
        axis_sizes={"T": 1, "Z": 3, "C": 1},
        channel_names=["C0"],
        plane_time_frames={0: 0.0, 2: 2.0},
    )
    meta = load_vsi(str(vsi))
    assert meta.plane_time_ms is not None
    assert np.isnan(meta.plane_time_ms[0, 0, 1])
    img = open_vsi(str(vsi))
    plane = img.read_plane(z=2)
    assert plane.shape == (16, 16)


def test_unseen_format_version_warns(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 16, 16), tile_size=16)
    write_minimal_vsi(
        vsi,
        stack_id=1,
        width=16,
        height=16,
        axis_order=["T", "Z", "C"],
        axis_sizes={"T": 1, "Z": 1, "C": 1},
        channel_names=["C0"],
        format_2035=(0x30063, 99999),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        img = open_vsi(str(vsi))
        assert img.format_version == (3, 99)
        assert any("not been tested" in str(w.message) for w in caught)


def test_open_vsi_import_is_lightweight() -> None:
    script = """
import sys
import bioio_vsi
bioio_vsi.open_vsi
for name in ("dask", "xarray", "bioio_base"):
    assert name not in sys.modules, name
"""
    subprocess.check_call([sys.executable, "-c", script])
