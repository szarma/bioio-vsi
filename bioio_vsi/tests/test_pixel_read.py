from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from bioio_vsi import open_ets, open_vsi
from bioio_vsi.tests.synthetic import make_stack


def test_raw_stack_roundtrip(tmp_path: Path) -> None:
    vsi, _ets = make_stack(
        tmp_path, shape=(1, 2, 3, 100, 120), axis_order="TZC", tile_size=64, n_levels=1
    )
    img = open_vsi(str(vsi))
    assert img.shape_tcz == (1, 2, 3)
    plane = img.read_plane(c=1, z=2)
    assert plane.shape == (100, 120)
    assert plane[0, 0] == 102  # t=0, c=1, z=2


def test_pyramid_and_tiles(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 1500, 1500), tile_size=512, n_levels=3)
    img = open_vsi(str(vsi))
    assert img.n_levels == 3
    assert img.level_shape(1)[3:] == (750, 750)
    region = img.read_region(400, 500, 200, 150, level=0)
    full = img.read_plane(level=0)
    assert np.array_equal(region, full[400:600, 500:650])


def test_axis_order_czt(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(2, 3, 4, 50, 60), axis_order="CZT", tile_size=50)
    img = open_vsi(str(vsi))
    assert img.shape_tcz == (2, 3, 4)
    p = img.read_plane(t=1, c=2, z=3)
    assert p[0, 0] == 1 * 1000 + 2 * 100 + 3


def test_orphan_ets(tmp_path: Path) -> None:
    _, ets = make_stack(tmp_path, shape=(1, 2, 1, 32, 32), tile_size=32)
    orphan = open_ets(str(ets))
    assert orphan.slot_sizes == [1, 1, 2]  # T, Z, C sizes in default TZC order
    plane = orphan.read_plane_slots((0, 0, 1))  # T=0, Z=0, C=1
    assert plane.shape == (32, 32)


def test_pickling(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 40, 40), tile_size=40)
    img = open_vsi(str(vsi))
    img2 = pickle.loads(pickle.dumps(img))
    assert np.array_equal(img.read_plane(), img2.read_plane())


def test_read_region_out_of_bounds(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 32, 32), tile_size=32)
    img = open_vsi(str(vsi))
    with pytest.raises(IndexError):
        img.read_region(0, 0, 32, 33)
    with pytest.raises(IndexError):
        img.read_region(-1, 0, 10, 10)


class Index:
    def __init__(self, value: int = 0) -> None:
        self._value = value

    def __index__(self) -> int:
        return self._value


def test_index_protocol_vsi_image(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 32, 32), tile_size=32)
    img = open_vsi(str(vsi))
    with pytest.raises(TypeError):
        img.read_plane(z=0.5)  # type: ignore[arg-type]
    with pytest.raises(IndexError):
        img.read_plane(z=-1)
    with pytest.raises(IndexError):
        img.read_plane(z=9)
    with pytest.raises(IndexError):
        img.read_plane(t=99)
    assert img.read_plane(z=Index(0)).shape == (32, 32)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        img.read_region(0.5, 0, 8, 8)  # type: ignore[arg-type]
    with pytest.raises(IndexError):
        img.read_region(0, 0, 8, 8, level=-1)
    assert img.read_region(Index(), Index(), 8, 8).shape == (8, 8)  # type: ignore[arg-type]


def test_index_protocol_orphan(tmp_path: Path) -> None:
    _, ets = make_stack(tmp_path, shape=(1, 1, 1, 32, 32), tile_size=32)
    orphan = open_ets(str(ets))
    with pytest.raises(IndexError):
        orphan.read_plane_slots([0, 0, 99])
    with pytest.raises(IndexError):
        orphan.read_region(0, 0, 8, 8, slot_coords=[0, 0, 0], level=-1)


def test_orphan_fractional_slot_refused(tmp_path: Path) -> None:
    _, ets = make_stack(tmp_path, shape=(1, 1, 1, 32, 32), tile_size=32)
    orphan = open_ets(str(ets))
    with pytest.raises(TypeError):
        orphan.read_plane_slots([0, 0.9, 0])  # type: ignore[list-item]


def test_vsi_fractional_plane_index_refused(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 16, 16), tile_size=16)
    img = open_vsi(str(vsi))
    with pytest.raises(TypeError):
        img.read_plane(z=0.9)  # type: ignore[arg-type]
