"""Anonymise / materialise fixture tests."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from bioio_vsi import describe, open_vsi
from bioio_vsi._anonymize import _UNIT_RE, anonymize, anonymize_vsi_bytes
from bioio_vsi._anonymize_dates import is_plausible_absolute_date_int
from bioio_vsi._describe_redact import describe_redacted
from bioio_vsi._fixture_pixels import synthetic_plane
from bioio_vsi._materialize import _axis_order_from_meta, materialize
from bioio_vsi._stack import level_shape
from bioio_vsi._synthetic_numbers import synthesize_numbers
from bioio_vsi._tree import SCALAR, VEC, load_tree, walk
from bioio_vsi._vsi import load_vsi
from bioio_vsi.errors import VsiError
from bioio_vsi.tests import synthetic as syn
from bioio_vsi.tests.conftest import COMMITTED_VSI, has_imagecodecs, public_samples_root
from bioio_vsi.tests.content_check import (
    measured_numbers,
    preview_pixels_left,
    varied_byte_arrays,
)

STRING_TYPES = frozenset({13, 8192, 8194})
MARKER = "UNIQUE_LEAK_MARKER_42"
DATE_TAGS = frozenset({11, 12, 14, 23, 2015})


def _iter_vsi_files() -> list[Path]:
    paths = [COMMITTED_VSI]
    root = public_samples_root()
    if root.is_dir():
        paths.extend(sorted(root.rglob("*.vsi")))
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        if not p.is_file():
            continue
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _collect_strings_and_dates(tree: object) -> tuple[list[bytes], list[bytes], list[int]]:
    latin: list[bytes] = []
    utf16: list[bytes] = []
    dates: list[int] = []

    def scan_value(value: object, value_type: int) -> None:
        # Unit strings are format vocabulary and are kept on purpose.
        if (
            value_type in STRING_TYPES
            and isinstance(value, str)
            and value
            and not _UNIT_RE.fullmatch(value)
        ):
            latin.append(value.encode("latin-1"))
            utf16.append(value.encode("utf-16-le"))
        if (
            value_type in SCALAR
            and isinstance(value, int)
            and is_plausible_absolute_date_int(value)
        ):
            dates.append(value)
        if value_type in VEC and isinstance(value, list):
            for item in value:
                if isinstance(item, int) and is_plausible_absolute_date_int(item):
                    dates.append(item)
                if isinstance(item, float) and item > 1e9:
                    dates.append(int(item))

    for node in walk(tree.root):  # type: ignore[attr-defined]
        if node.tag in DATE_TAGS and node.inline and isinstance(node.value, int):
            if is_plausible_absolute_date_int(node.value):
                dates.append(node.value)
        elif node.tag in DATE_TAGS and node.value is not None and not node.extended:
            vt = node.value_type
            if isinstance(node.value, dict):
                continue
            scan_value(node.value, vt)
        if node.children:
            for ch in node.children:
                if ch.value is not None and not ch.inline:
                    scan_value(ch.value, ch.value_type)

    return latin, utf16, dates


@pytest.mark.parametrize("vsi_path", _iter_vsi_files(), ids=lambda p: p.name)
def test_anonymize_coverage(vsi_path: Path) -> None:
    tree = load_tree(str(vsi_path))
    out, report = anonymize_vsi_bytes(tree)
    assert len(out) == len(tree.data)
    load_tree(out)
    non_zero = sum(1 for b in out if b != 0)
    classified = sum(
        report.byte_counts.get(k, 0) for k in ("structure", "numeric", "date", "placeholder")
    )
    assert classified >= non_zero * 0.95


@pytest.mark.parametrize("vsi_path", _iter_vsi_files(), ids=lambda p: p.name)
def test_anonymize_leak(vsi_path: Path, tmp_path: Path) -> None:
    tree = load_tree(str(vsi_path))
    latin, utf16, dates = _collect_strings_and_dates(tree)
    fixture = anonymize(vsi_path, tmp_path / "fix")
    anon = (fixture).read_bytes()
    skel_blob = b"".join(p.read_bytes() for p in (tmp_path / "fix").rglob("*.skel"))
    hay = anon + skel_blob
    for s in latin:
        if len(s) < 4:
            continue
        assert s not in hay
    for s in utf16:
        if len(s) < 8:
            continue
        assert s not in hay
    for d in dates:
        if -(2**31) <= d <= 2**31 - 1:
            assert struct.pack("<i", d) not in hay
        if -(2**63) <= d <= 2**63 - 1:
            assert struct.pack("<q", d) not in hay
    if len(vsi_path.name) >= 12:
        assert vsi_path.name.encode("latin-1") not in hay


@pytest.mark.parametrize("vsi_path", _iter_vsi_files(), ids=lambda p: p.name)
def test_anonymize_removes_pictures_and_text(vsi_path: Path) -> None:
    out, _report = anonymize_vsi_bytes(load_tree(str(vsi_path)))
    assert preview_pixels_left(out) == 0
    assert varied_byte_arrays(out) == []


@pytest.mark.parametrize("vsi_path", _iter_vsi_files(), ids=lambda p: p.name)
def test_synthetic_numbers_leave_no_measurement(vsi_path: Path) -> None:
    anon, _report = anonymize_vsi_bytes(load_tree(str(vsi_path)))
    assert measured_numbers(anon)  # real files carry measured values
    out = synthesize_numbers(anon)
    assert len(out) == len(anon)
    load_tree(out)
    assert measured_numbers(out) == []


def test_anonymize_leak_synthetic_markers(tmp_path: Path) -> None:
    vsi = tmp_path / "m.vsi"
    syn.write_minimal_vsi(
        vsi,
        stack_id=1,
        width=64,
        height=64,
        axis_order=["T", "Z", "C"],
        axis_sizes={"T": 1, "Z": 1, "C": 1},
        channel_names=[MARKER],
        document_children=[syn._string(11, MARKER), syn._string(12, MARKER)],
    )
    syn.write_ets(
        tmp_path / "_m_" / "stack1" / "frame_t_0.ets",
        axis_order=["T", "Z", "C"],
        axis_sizes={"T": 1, "Z": 1, "C": 1},
        planes={(0, 0, 0, 0): np.zeros((64, 64), dtype=np.uint16)},
        tile_size=64,
    )
    fix = anonymize(vsi, tmp_path / "fix")
    blob = fix.read_bytes() + b"".join(p.read_bytes() for p in (tmp_path / "fix").rglob("*.skel"))
    assert MARKER.encode("latin-1") not in blob
    assert MARKER.encode("utf-16-le") not in blob


@pytest.mark.skipif(not has_imagecodecs(), reason="needs imagecodecs")
@pytest.mark.parametrize("vsi_path", [COMMITTED_VSI], ids=["committed"])
def test_anonymize_equivalence_committed(vsi_path: Path, tmp_path: Path) -> None:
    orig_meta = load_vsi(str(vsi_path))
    orig = open_vsi(str(vsi_path))
    fix_dir = tmp_path / "fix"
    out_dir = tmp_path / "out"
    anonymize(vsi_path, fix_dir, synthetic_numbers=False)
    materialize(fix_dir, out_dir, planes="all")
    new = open_vsi(str(out_dir / "anon.vsi"))
    assert new.shape_tcz == orig.shape_tcz
    assert new.dtype == orig.dtype
    assert new.n_levels == orig.n_levels
    assert new.pixel_size_um == orig.pixel_size_um
    assert new.z_step_um == orig.z_step_um
    order = _axis_order_from_meta(orig_meta)
    t, c, z = orig.shape_tcz
    for ti in range(t):
        for ci in range(c):
            for zi in range(z):
                for level in range(orig.n_levels):
                    got = new.read_plane(t=ti, c=ci, z=zi, level=level)
                    ly, lx = level_shape(orig_meta.height, orig_meta.width, level)
                    want = synthetic_plane(
                        ly,
                        lx,
                        dtype=orig.dtype,
                        t=ti,
                        c=ci,
                        z=zi,
                        level=level,
                        axis_order=order,
                    )
                    assert got.shape == want.shape
                    np.testing.assert_array_equal(got, want)
    text, err = describe_redacted(str(out_dir / "anon.vsi"))
    assert not err
    describe(str(out_dir / "anon.vsi"))


@pytest.mark.parametrize(
    "axis_order",
    ["TZC", "ZCT", "CTZ"],
)
def test_anonymize_equivalence_synthetic_axis_orders(tmp_path: Path, axis_order: str) -> None:
    vsi, _ets = syn.make_stack(
        tmp_path / axis_order,
        shape=(2, 2, 2, 128, 128),
        axis_order=axis_order,
    )
    fix = tmp_path / "fix"
    out = tmp_path / "out"
    anonymize(vsi, fix)
    materialize(fix, out, planes="all")
    orig = open_vsi(str(vsi))
    new = open_vsi(str(out / "anon.vsi"))
    assert new.shape_tcz == orig.shape_tcz
    assert new.sizes == orig.sizes


def test_materialize_refusal_carries_over(tmp_path: Path) -> None:
    vsi, _ets = syn.make_stack(
        tmp_path,
        shape=(1, 1, 1, 64, 64),
        axis_order="TZC",
        origin=(3, 0),
    )
    fix = tmp_path / "fix"
    out = tmp_path / "out"
    anonymize(vsi, fix)
    with pytest.raises(VsiError):
        materialize(fix, out)
        open_vsi(str(out / "anon.vsi"))


def test_date_inventory_public_samples() -> None:
    """Document plausibly absolute date fields seen in public metadata."""
    found: set[str] = set()
    for vsi in _iter_vsi_files():
        tree = load_tree(str(vsi))
        _, report = anonymize_vsi_bytes(tree)
        for entry in report.date_fields:
            found.add(entry)
    assert found  # noqa: S101 — inventory must be non-empty on real samples
