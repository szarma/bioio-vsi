"""Describe redaction: golden output, taint-by-construction, and path leak checks."""

from __future__ import annotations

import stat
import struct
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from bioio_vsi import describe
from bioio_vsi._tree import child, collection_root, document_root, kids, load_tree
from bioio_vsi._vsi import document_properties, format_version_tuple
from bioio_vsi.tests.conftest import COMMITTED_VSI
from bioio_vsi.tests.synthetic import (
    _build_volume,
    _field_nested,
    _int_vector,
    _string,
    make_stack,
    write_ets,
    write_minimal_vsi,
)

MARKER = "TAINT_R4_UNIQUE_7f3a9c2e"
MARKER_DIR = f"dir_{MARKER}"
MARKER_STEM = f"stem_{MARKER}"


def _assert_no_taint(text: str, path: Path) -> None:
    assert MARKER not in text
    assert MARKER_DIR not in text
    assert MARKER_STEM not in text
    assert str(path) not in text
    assert "frame_t" not in text


def _describe_all(path: Path) -> str:
    api = describe(str(path), redact=True)
    proc = subprocess.run(
        [sys.executable, "-m", "bioio_vsi", "describe", "--redact", str(path)],
        capture_output=True,
        text=True,
    )
    combined = api + proc.stdout + proc.stderr
    _assert_no_taint(combined, path)
    return combined


def _write_all_string_markers(tmp: Path) -> tuple[Path, Path]:
    """VSI tree strings and byte payloads use MARKER; numeric report fields mistyped."""
    base = tmp / MARKER_DIR / MARKER_STEM
    vsi = base.with_suffix(".vsi")
    ets = base.parent / f"_{MARKER_STEM}_" / "stack1" / "frame_t_0.ets"
    write_ets(
        ets,
        axis_order=["T", "Z", "C"],
        axis_sizes={"T": 1, "Z": 1, "C": 1},
        planes={(0, 0, 0, 0): __import__("numpy").zeros((8, 8), dtype=__import__("numpy").uint16)},
        tile_size=8,
    )
    write_minimal_vsi(
        vsi,
        stack_id=1,
        width=16,
        height=16,
        axis_order=["T", "Z", "C"],
        axis_sizes={"T": 1, "Z": 1, "C": 1},
        channel_names=["C0"],
        document_children=[_string(21, MARKER), _string(34, MARKER), _string(11, MARKER)],
        collection_2035=_string(2035, MARKER),
        volume_2003=_string(2003, MARKER),
        volume_2053=_string(2053, MARKER),
        volume_2074=_string(2074, MARKER),
        volume_2023=_string(2023, MARKER),
    )
    return vsi, ets


def _write_all_string_markers_legacy_tree(tmp: Path) -> tuple[Path, Path]:
    """Older layout used for golden-style multi-string tree (non-numeric leaves)."""
    base = tmp / MARKER_DIR / MARKER_STEM
    vsi = base.with_suffix(".vsi")
    ets = base.parent / f"_{MARKER_STEM}_" / "stack1" / "frame_t_0.ets"
    write_ets(
        ets,
        axis_order=["T", "Z", "C"],
        axis_sizes={"T": 1, "Z": 1, "C": 1},
        planes={(0, 0, 0, 0): __import__("numpy").zeros((8, 8), dtype=__import__("numpy").uint16)},
        tile_size=8,
    )
    coll_children: list[bytes] = [
        _string(2035, MARKER),
        _field_nested(
            2001,
            [
                _string(2003, MARKER),
                _field_nested(
                    2005,
                    [
                        _string(2030, MARKER),
                        _string(2074, MARKER),
                    ],
                ),
                _field_nested(
                    2007,
                    [
                        _field_nested(
                            2008,
                            [_string(2023, MARKER), _string(2021, MARKER)],
                        )
                    ],
                    second=0,
                ),
                _field_nested(2007, [], second=1),
            ],
            second=1,
        ),
    ]
    root = _build_volume(
        [
            _field_nested(
                2,
                [
                    _string(34, MARKER),
                    _string(35, MARKER),
                    _string(11, MARKER),
                    _int_vector(21, [99999]),
                ],
            ),
            _field_nested(2000, coll_children, second=0),
        ]
    )
    vsi.parent.mkdir(parents=True, exist_ok=True)
    vsi.write_bytes(b"II*\x00" + struct.pack("<I", 0) + root)
    return vsi, ets


def test_redacted_committed_matches_golden() -> None:
    expected = (Path(__file__).parent / "resources" / "describe_redacted_committed.txt").read_text()
    assert describe(str(COMMITTED_VSI), redact=True) == expected


def test_taint_all_string_fields(tmp_path: Path) -> None:
    vsi, _ = _write_all_string_markers_legacy_tree(tmp_path)
    _describe_all(vsi)


_NUMERIC_CORRUPTIONS: dict[str, Callable[[Path, Path], None]] = {}
_NUMERIC_ASSERTIONS: dict[str, Callable[[Any], None]] = {}


def _register_numeric(
    name: str,
    corrupt: Callable[[Path, Path], None],
    assert_marker: Callable[[Any], None],
) -> None:
    _NUMERIC_CORRUPTIONS[name] = corrupt
    _NUMERIC_ASSERTIONS[name] = assert_marker


def _base_stack(tmp: Path) -> tuple[Path, Path]:
    base = tmp / MARKER_DIR
    return make_stack(base / MARKER_STEM, shape=(1, 1, 1, 16, 16), tile_size=16)


def _minimal(vsi: Path, **kwargs: Any) -> None:
    write_minimal_vsi(
        vsi,
        stack_id=1,
        width=16,
        height=16,
        axis_order=["T", "Z", "C"],
        axis_sizes={"T": 1, "Z": 1, "C": 1},
        channel_names=["C0"],
        **kwargs,
    )


def _vol(tree: Any) -> Any:
    return kids(collection_root(tree), 2001)[0]


def _assert_marker_in_tree(tree: Any, case: str) -> None:
    _NUMERIC_ASSERTIONS[case](tree)


def _corrupt_document_build(vsi: Path, _ets: Path) -> None:
    _minimal(vsi, document_children=[_string(21, MARKER), _string(34, "synthetic")])


def _assert_document_build(tree: Any) -> None:
    doc = document_root(tree)
    assert doc is not None
    build = child(doc, 21)
    assert build is not None
    assert build.value == MARKER


_register_numeric("document_build", _corrupt_document_build, _assert_document_build)


def _corrupt_format_2035(vsi: Path, _ets: Path) -> None:
    _minimal(vsi, collection_2035=_string(2035, MARKER))


def _assert_format_2035(tree: Any) -> None:
    coll = collection_root(tree)
    fmt = child(coll, 2035)
    assert fmt is not None
    assert fmt.value == MARKER


_register_numeric("format_version_2035", _corrupt_format_2035, _assert_format_2035)


def _corrupt_image_type_2074(vsi: Path, _ets: Path) -> None:
    _minimal(vsi, volume_2074=_string(2074, MARKER))


def _assert_image_type_2074(tree: Any) -> None:
    sp = child(_vol(tree), 2005)
    assert sp is not None
    itype = child(sp, 2074)
    assert itype is not None
    assert itype.value == MARKER


_register_numeric("image_type_2074", _corrupt_image_type_2074, _assert_image_type_2074)


def _corrupt_boundary_2053(vsi: Path, _ets: Path) -> None:
    _minimal(vsi, volume_2053=_string(2053, MARKER))


def _assert_boundary_2053(tree: Any) -> None:
    fr = kids(_vol(tree), 2002)[0]
    ext = child(fr, 2018)
    assert ext is not None
    rect = child(ext, 2053)
    assert rect is not None
    assert rect.value == MARKER


_register_numeric("boundary_rectangle_2053", _corrupt_boundary_2053, _assert_boundary_2053)


def _corrupt_dimension_sizes_2003(vsi: Path, _ets: Path) -> None:
    _minimal(vsi, volume_2003=_string(2003, MARKER))


def _assert_dimension_sizes_2003(tree: Any) -> None:
    sizes = child(_vol(tree), 2003)
    assert sizes is not None
    assert sizes.value == MARKER


_register_numeric(
    "dimension_sizes_2003",
    _corrupt_dimension_sizes_2003,
    _assert_dimension_sizes_2003,
)


def _corrupt_dimension_kind_2023(vsi: Path, _ets: Path) -> None:
    _minimal(vsi, volume_2023=_string(2023, MARKER))


def _assert_dimension_kind_2023(tree: Any) -> None:
    dim = kids(_vol(tree), 2007)[0]
    ps = kids(dim, 2008)[0]
    kind = child(ps, 2023)
    assert kind is not None
    assert kind.value == MARKER


_register_numeric("dimension_kind_2023", _corrupt_dimension_kind_2023, _assert_dimension_kind_2023)


@pytest.mark.parametrize("case", list(_NUMERIC_CORRUPTIONS.keys()))
def test_taint_numeric_field_string(tmp_path: Path, case: str) -> None:
    vsi, ets = _base_stack(tmp_path)
    _NUMERIC_CORRUPTIONS[case](vsi, ets)
    tree = load_tree(str(vsi))
    _assert_marker_in_tree(tree, case)
    _describe_all(vsi)


def test_taint_numeric_fields_all_at_once(tmp_path: Path) -> None:
    vsi, _ = _write_all_string_markers(tmp_path)
    tree = load_tree(str(vsi))
    doc = document_properties(tree)
    assert doc.get("build") == MARKER
    coll = collection_root(tree)
    fmt = child(coll, 2035)
    assert fmt is not None and fmt.value == MARKER
    vol = _vol(tree)
    sizes = child(vol, 2003)
    assert sizes is not None and sizes.value == MARKER
    sp = child(vol, 2005)
    assert sp is not None
    itype = child(sp, 2074)
    assert itype is not None and itype.value == MARKER
    fr = kids(vol, 2002)[0]
    ext = child(fr, 2018)
    assert ext is not None
    rect = child(ext, 2053)
    assert rect is not None and rect.value == MARKER
    dim = kids(vol, 2007)[0]
    ps = kids(dim, 2008)[0]
    kind = child(ps, 2023)
    assert kind is not None and kind.value == MARKER
    raw_2035 = fmt.value
    assert format_version_tuple(raw_2035 if isinstance(raw_2035, (list, tuple)) else None) is None
    _describe_all(vsi)


def test_taint_ets_permission(tmp_path: Path) -> None:
    vsi, ets = _base_stack(tmp_path)
    ets.chmod(0)
    try:
        _describe_all(vsi)
    finally:
        ets.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_taint_ets_missing(tmp_path: Path) -> None:
    vsi, ets = _base_stack(tmp_path)
    ets.unlink()
    _describe_all(vsi)


def test_taint_ets_is_directory(tmp_path: Path) -> None:
    vsi, ets = _base_stack(tmp_path)
    ets.unlink()
    ets.mkdir()
    _describe_all(vsi)


@pytest.mark.parametrize("truncate", [64, 128, 256, 512, 1024])
def test_taint_ets_truncated(tmp_path: Path, truncate: int) -> None:
    vsi, ets = _base_stack(tmp_path)
    data = ets.read_bytes()
    if truncate < len(data):
        ets.write_bytes(data[:truncate])
    _describe_all(vsi)


@pytest.mark.parametrize(
    "offset,payload",
    [
        (0, b"BAD!"),
        (64, b"BAD!"),
        (12, struct.pack("<I", 99)),
    ],
)
def test_taint_ets_header_corrupt(tmp_path: Path, offset: int, payload: bytes) -> None:
    vsi, ets = _base_stack(tmp_path)
    data = bytearray(ets.read_bytes())
    for i, b in enumerate(payload):
        if offset + i < len(data):
            data[offset + i] = b
    ets.write_bytes(data)
    _describe_all(vsi)


def test_taint_vsi_truncated(tmp_path: Path) -> None:
    vsi, _ = _base_stack(tmp_path)
    broken = vsi.parent / f"{MARKER_STEM}_trunc.vsi"
    broken.write_bytes(vsi.read_bytes()[:-48])
    _describe_all(broken)


def test_taint_vsi_unreadable(tmp_path: Path) -> None:
    vsi, _ = _base_stack(tmp_path)
    target = vsi.parent / f"{MARKER_STEM}_perm.vsi"
    target.write_bytes(vsi.read_bytes())
    target.chmod(0)
    try:
        _describe_all(target)
    finally:
        target.chmod(stat.S_IRUSR | stat.S_IWUSR)


TAINT_VARIANT_COUNT = (
    1  # all string markers (legacy tree)
    + len(_NUMERIC_CORRUPTIONS)  # numeric mistypes one at a time
    + 1  # all at once
    + 3  # ets permission, missing, directory
    + 5  # ets truncations
    + 3  # ets header corruptions
    + 2  # vsi truncated, unreadable
)

PATH_MARKER = "secret-experiment-ABC"

SECRET = "SECRET_EXPERIMENT"
SECRET_VARIANTS = [
    "a: SECRET",
    "SECRET (1)",
    "SECRET: b: c",
    "Sëcrét",
]


def _assert_no_path_marker(text: str) -> None:
    assert PATH_MARKER not in text


def _assert_no_secret(text: str) -> None:
    assert SECRET not in text
    assert "SECRET_EXPERIMENT" not in text
    for part in ("Sëcrét", "a: SECRET", "SECRET (1)", "SECRET: b: c"):
        assert part not in text


def test_redact_malformed_ets_no_path_leak(tmp_path: Path) -> None:
    base = tmp_path / PATH_MARKER
    vsi, ets = make_stack(base, shape=(1, 1, 1, 16, 16), tile_size=16)
    data = bytearray(ets.read_bytes())
    struct.pack_into("<4s", data, 64, b"BAD!")
    ets.write_bytes(data)
    text = describe(str(vsi), redact=True)
    _assert_no_path_marker(text)
    assert "MALFORMED" in text


def test_redact_truncated_vsi_no_path_leak(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path / PATH_MARKER, shape=(1, 1, 1, 8, 8), tile_size=8)
    broken = tmp_path / PATH_MARKER / "broken.vsi"
    broken.parent.mkdir(parents=True, exist_ok=True)
    data = bytearray(vsi.read_bytes())
    del data[-32:]
    broken.write_bytes(data)
    text = describe(str(broken), redact=True)
    _assert_no_path_marker(text)


def test_redact_cli_missing_file(tmp_path: Path) -> None:
    bogus = tmp_path / PATH_MARKER / "nope.vsi"
    bogus.parent.mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, "-m", "bioio_vsi", "describe", "--redact", str(bogus)],
        capture_output=True,
        text=True,
    )
    combined = proc.stdout + proc.stderr
    _assert_no_path_marker(combined)


def test_redact_permission_error(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path / PATH_MARKER, shape=(1, 1, 1, 8, 8), tile_size=8)
    target = tmp_path / PATH_MARKER / "perm.vsi"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(vsi.read_bytes())
    target.chmod(0)
    try:
        text = describe(str(target), redact=True)
    finally:
        target.chmod(stat.S_IRUSR | stat.S_IWUSR)
    _assert_no_path_marker(text)
    assert "filesystem" in text.lower() or "permission" in text.lower()


@pytest.mark.parametrize("dirname", SECRET_VARIANTS)
def test_redact_describe_colon_paths(tmp_path: Path, dirname: str) -> None:
    base = tmp_path / dirname
    vsi, ets = make_stack(base / "nested", shape=(1, 1, 1, 8, 8), tile_size=8)
    data = bytearray(ets.read_bytes())
    struct.pack_into("<4s", data, 64, b"BAD!")
    ets.write_bytes(data)
    text = describe(str(vsi), redact=True)
    _assert_no_secret(text)
    assert "MALFORMED" in text


@pytest.mark.parametrize("dirname", SECRET_VARIANTS)
def test_redact_describe_truncated_vsi(tmp_path: Path, dirname: str) -> None:
    base = tmp_path / dirname
    vsi, _ = make_stack(base, shape=(1, 1, 1, 8, 8), tile_size=8)
    broken = base / "broken.vsi"
    data = bytearray(vsi.read_bytes())
    del data[-32:]
    broken.write_bytes(data)
    text = describe(str(broken), redact=True)
    _assert_no_secret(text)


@pytest.mark.parametrize("dirname", SECRET_VARIANTS)
def test_redact_cli_errors(tmp_path: Path, dirname: str) -> None:
    base = tmp_path / dirname
    bogus = base / "missing.vsi"
    bogus.parent.mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, "-m", "bioio_vsi", "describe", "--redact", str(bogus)],
        capture_output=True,
        text=True,
    )
    _assert_no_secret(proc.stdout + proc.stderr)

    vsi, ets = make_stack(base / "pair", shape=(1, 1, 1, 8, 8), tile_size=8)
    ets.write_bytes(ets.read_bytes()[:-40])
    proc2 = subprocess.run(
        [sys.executable, "-m", "bioio_vsi", "describe", "--redact", str(vsi)],
        capture_output=True,
        text=True,
    )
    _assert_no_secret(proc2.stdout + proc2.stderr)

    vsi2, _ets2 = make_stack(base / "pair2", shape=(1, 1, 1, 8, 8), tile_size=8)
    stack2 = base / "pair2" / "_syn_" / "stack1"
    stack2.mkdir(parents=True, exist_ok=True)
    (stack2 / "frame_t_1.ets").write_bytes(b"\x00" * 128)
    proc3 = subprocess.run(
        [sys.executable, "-m", "bioio_vsi", "describe", "--redact", str(vsi2)],
        capture_output=True,
        text=True,
    )
    _assert_no_secret(proc3.stdout + proc3.stderr)
