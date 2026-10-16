"""Absolute paths after chdir and pickling round-trips."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from bioio_vsi import open_ets, open_vsi
from bioio_vsi._stack import OrphanEtsImage
from bioio_vsi.reader import Reader
from bioio_vsi.tests.synthetic import make_stack


def test_paths_stable_after_chdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a, ae = make_stack(tmp_path / "a", shape=(1, 1, 1, 8, 8), tile_size=8)
    b, be = make_stack(tmp_path / "b", shape=(1, 1, 1, 8, 8), tile_size=8)
    blob = bytearray(be.read_bytes())
    blob[292:420] = np.full((8, 8), 60000, dtype="<u2").tobytes()
    be.write_bytes(blob)

    monkeypatch.chdir(a.parent)
    image = open_vsi("syn.vsi")
    orphan = open_ets("_syn_/stack1/frame_t_0.ets")
    reader = Reader("syn.vsi")
    before = int(image.read_plane()[0, 0])
    monkeypatch.chdir(b.parent)
    after_vsi = int(image.read_plane()[0, 0])
    after_ets = int(orphan.read_plane_slots([0, 0, 0])[0, 0])
    assert before == after_vsi == after_ets == 0
    assert Path(image.path).is_absolute()
    assert Path(image.ets_path).is_absolute()
    assert Path(orphan.path).is_absolute()

    restored = pickle.loads(pickle.dumps(image))
    assert int(restored.read_plane()[0, 0]) == 0
    da_plane = reader.get_image_dask_data("YX", T=0, C=0, Z=0).compute()
    assert int(da_plane[0, 0]) == 0


def test_orphan_constructor_resolves_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, ae = make_stack(tmp_path / "a", shape=(1, 1, 1, 8, 8), tile_size=8)
    _, be = make_stack(tmp_path / "b", shape=(1, 1, 1, 8, 8), tile_size=8)
    blob = bytearray(be.read_bytes())
    blob[292:420] = np.full((8, 8), 60000, dtype="<u2").tobytes()
    be.write_bytes(blob)
    rel = str(ae.relative_to(tmp_path / "a"))
    monkeypatch.chdir(tmp_path / "a")
    orphan = OrphanEtsImage(rel)
    monkeypatch.chdir(tmp_path / "b")
    assert Path(orphan.path).is_absolute()
    assert int(orphan.read_plane_slots([0, 0, 0])[0, 0]) == 0
    assert int(pickle.loads(pickle.dumps(orphan)).read_plane_slots([0, 0, 0])[0, 0]) == 0
