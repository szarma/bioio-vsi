"""Materialisation I/O behaviour (sparse files, JPEG 2000 layout)."""

from __future__ import annotations

import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from bioio_vsi._anonymize import anonymize, read_skel
from bioio_vsi._ets import read_ets_index
from bioio_vsi._fixture_pixels import render_tile
from bioio_vsi._materialize import (
    _axis_order_from_meta,
    _write_ets_shell,
    materialize,
    materialize_ets,
)
from bioio_vsi._stack import level_shape
from bioio_vsi._vsi import load_vsi
from bioio_vsi.tests import synthetic as syn
from bioio_vsi.tests.conftest import has_imagecodecs


def _sparse_allocation_bytes(path: Path) -> int:
    st = path.stat()
    return st.st_blocks * 512


def _filesystem_sparse_ok(tmp_path: Path) -> bool:
    probe = tmp_path / "sparse_probe"
    with open(probe, "wb") as f:
        f.seek(64 * 1024 * 1024 - 1)
        f.write(b"\0")
    return _sparse_allocation_bytes(probe) < 8 * 1024 * 1024


@pytest.mark.skipif(sys.platform == "win32", reason="sparse files")
def test_materialize_huge_sparse_ets(tmp_path: Path) -> None:
    if not _filesystem_sparse_ok(tmp_path):
        pytest.skip("filesystem does not allocate sparse files efficiently")
    vsi, _ets = syn.make_stack(tmp_path / "small", shape=(1, 1, 1, 64, 64), tile_size=64)
    fix = tmp_path / "fix"
    anonymize(vsi, fix)
    skel = next(fix.rglob("*.skel"))
    nominal, ranges = read_skel(skel)
    huge = 8 * 1024**3
    assert nominal < huge
    # Rebuild skel with inflated nominal size (headers unchanged).
    from bioio_vsi._anonymize import SKEL_MAGIC, SKEL_VERSION

    body = bytearray()
    for offset, chunk in ranges:
        body.extend(struct.pack("<QI", offset, len(chunk)))
        body.extend(chunk)
    header = SKEL_MAGIC + struct.pack("<IQI", SKEL_VERSION, huge, len(ranges))
    skel.write_bytes(header + bytes(body))

    out = tmp_path / "out"
    materialize(fix, out, planes="default")
    ets_path = next(out.rglob("*.ets"))
    assert ets_path.stat().st_size == huge
    assert _sparse_allocation_bytes(ets_path) < 64 * 1024 * 1024

    meta = load_vsi(str(out / "anon.vsi"))
    idx = read_ets_index(ets_path)
    h = idx.header
    ti = 0
    off = int(idx.offsets[ti])
    length = int(idx.lengths[ti])
    with open(ets_path, "rb") as f:
        f.seek(off)
        raw = f.read(length)
    tile = np.frombuffer(raw, dtype=h.dtype).reshape(h.tile_height, h.tile_width)
    ly, lx = level_shape(meta.height, meta.width, 0)
    order = _axis_order_from_meta(meta)
    want = render_tile(
        ly,
        lx,
        h.origin_x,
        h.origin_y,
        h.tile_height,
        h.tile_width,
        dtype=h.dtype,
        t=0,
        c=0,
        z=0,
        level=0,
        axis_order=order,
    )
    np.testing.assert_array_equal(tile, want)


@pytest.mark.skipif(not has_imagecodecs(), reason="needs imagecodecs")
def test_jpeg2000_headers_and_unfilled_tile(tmp_path: Path) -> None:
    vsi_path, _ets = syn.make_stack(
        tmp_path / "jp2k",
        shape=(2, 1, 1, 128, 128),
        tile_size=64,
        compression=3,
    )
    fix = tmp_path / "fix"
    out = tmp_path / "out"
    anonymize(vsi_path, fix)
    skel = next(fix.rglob("*.skel"))
    out.mkdir()
    shutil.copy2(fix / "anon.vsi", out / "anon.vsi")
    rel = skel.relative_to(fix)
    ets_dest = out / rel.with_suffix("")
    ets_dest.parent.mkdir(parents=True, exist_ok=True)
    _write_ets_shell(skel, ets_dest)
    with open(ets_dest, "rb") as f:
        main_before = f.read(64)
        addpos = struct.unpack_from("<Q", main_before, 16)[0]
        addsize = struct.unpack_from("<I", main_before, 24)[0]
        f.seek(addpos)
        second_before = f.read(addsize)
    meta = load_vsi(str(out / "anon.vsi"))
    materialize_ets(skel, ets_dest, meta, planes="default")
    with open(ets_dest, "rb") as f:
        main_after = f.read(64)
        f.seek(addpos)
        second_after = f.read(addsize)
    assert main_before[:32] == main_after[:32]
    assert main_before[36:64] == main_after[36:64]
    assert second_before == second_after

    idx = read_ets_index(ets_dest)
    h = idx.header
    unfilled = next(
        ti
        for ti in range(h.ntiles)
        if int(idx.lengths[ti]) > 0
        and int(idx.coordinates[ti][2]) == 1  # time index 1 not materialised by default
    )
    off = int(idx.offsets[unfilled])
    length = int(idx.lengths[unfilled])
    with open(ets_dest, "rb") as f:
        f.seek(off)
        blob = f.read(length)
    assert blob == b"\0" * length
