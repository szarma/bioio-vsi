"""Data-driven interpreted metadata path corruption (review mutation table)."""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import pytest

from bioio_vsi._tree import FieldNode
from bioio_vsi.tests import synthetic as syn
from bioio_vsi.tests.synthetic import make_stack, write_minimal_vsi
from bioio_vsi.tests.tree_rewrite import CORRUPT_META_MARKER, rewrite_vsi, siblings_replace

CORRUPT = CORRUPT_META_MARKER


@dataclass(frozen=True)
class _InterpretedRow:
    path_id: str
    string: bool
    fractional: bool
    wrong_len: bool
    duplicate: bool


_INTERPRETED_ROWS: tuple[_InterpretedRow, ...] = (
    _InterpretedRow("2000/2035", True, True, True, True),
    _InterpretedRow("2/21", True, True, False, True),
    _InterpretedRow("2001/2002/2018/2053", True, True, True, True),
    _InterpretedRow("2001/2003", True, True, True, True),
    _InterpretedRow("2001/2005/2019", True, False, True, True),
    _InterpretedRow("2001/2005/2020", True, False, False, True),
    _InterpretedRow("2001/2005/2018", True, True, True, True),
    _InterpretedRow("2001/2005/2015", True, True, False, True),
    _InterpretedRow("2001/2005/2030", False, False, False, True),
    _InterpretedRow("2001/2005/2074", True, True, False, True),
    _InterpretedRow("2001/2007/2008/2023", True, True, False, True),
    _InterpretedRow("2001/2007/2008/2012_2013", True, False, False, True),
    _InterpretedRow("2001/2007/2008/2021_2419", False, False, False, True),
    _InterpretedRow("2001/2007/2008/2417", True, True, False, True),
    _InterpretedRow("2001/2002/2006/2014_2017", True, True, False, True),
    _InterpretedRow("2001/2002/2018/20005", True, True, False, True),
    _InterpretedRow("2005/2043/120114/100049", True, True, False, True),
)


def interpreted_cases() -> list[Any]:
    out: list[Any] = []
    for row in _INTERPRETED_ROWS:
        for mut, applies in (
            ("string", row.string),
            ("fractional", row.fractional),
            ("wrong_len", row.wrong_len),
            ("duplicate_first", row.duplicate),
            ("duplicate_second", row.duplicate),
        ):
            if applies:
                out.append(pytest.param(row.path_id, mut))
            else:
                out.append(
                    pytest.param(
                        row.path_id,
                        mut,
                        marks=pytest.mark.skip(reason="mutation does not apply"),
                    )
                )
    return out


def _camera_device_field() -> bytes:
    return syn._field_nested(
        2043,
        [syn._field_nested(120114, [syn._field_inline(100049, 16)], second=0)],
        second=0,
    )


def _prop_corrupt_string(tag: int) -> bytes:
    return syn._field_nested(tag, [syn._string(268435458, CORRUPT)])


def _prop_corrupt_fractional(tag: int) -> bytes:
    return syn._field_nested(tag, [syn._field_data(268435458, 10, struct.pack("<d", 1.5))])


def _rewrite_2005_extra(vsi: Path, extra_fields: list[bytes]) -> None:
    rewrite_vsi(
        vsi,
        lambda n: (
            syn._field_nested(
                2005,
                [
                    syn._double_vector(2019, [1.0, 1.0]),
                    syn._string(2030, "synthetic"),
                ]
                + extra_fields,
            )
            if n.tag == 2005
            else None
        ),
    )


_PREPARE: dict[str, Callable[[Path], Path]] = {}


def _register_prepare(path_id: str, fn: Callable[[Path], Path]) -> None:
    _PREPARE[path_id] = fn


def _default_stack(tmp_path: Path) -> Path:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 2, 8, 8), tile_size=8)
    return vsi


_register_prepare("2001/2003", _default_stack)
_register_prepare("2001/2005/2019", _default_stack)
_register_prepare("2001/2002/2018/2053", _default_stack)


def _prepare_2001_2002_2006_2014_2017(tmp_path: Path) -> Path:
    vsi, _ = make_stack(
        tmp_path,
        shape=(2, 1, 1, 8, 8),
        tile_size=8,
        plane_times_ms=[0.0, 1.0],
    )
    rewrite_vsi(
        vsi,
        lambda n: (
            syn._field_nested(2006, [syn._prop_double(2014, 1.0)])
            if n.tag == 2002 and n.second == 0
            else None
        ),
    )
    return vsi


_register_prepare("2001/2002/2006/2014_2017", _prepare_2001_2002_2006_2014_2017)


def _minimal_z2(tmp_path: Path) -> Path:
    vsi = tmp_path / "syn.vsi"
    write_minimal_vsi(
        vsi,
        stack_id=1,
        width=8,
        height=8,
        axis_order=["Z"],
        axis_sizes={"Z": 2},
        channel_names=["C0"],
    )
    return vsi


def _prepare_2_21(tmp_path: Path) -> Path:
    vsi = _minimal_z2(tmp_path)
    write_minimal_vsi(
        vsi,
        stack_id=1,
        width=8,
        height=8,
        axis_order=["Z"],
        axis_sizes={"Z": 2},
        channel_names=["C0"],
        document_children=[syn._string(34, "synthetic"), syn._field_inline(21, 99)],
    )
    return vsi


_register_prepare("2/21", _prepare_2_21)
_register_prepare("2000/2035", _default_stack)


def _prepare_2005_branch(tmp_path: Path, extra: list[bytes], **vol_kw: Any) -> Path:
    vsi = _minimal_z2(tmp_path)
    write_minimal_vsi(
        vsi,
        stack_id=1,
        width=8,
        height=8,
        axis_order=["Z"],
        axis_sizes={"Z": 2},
        channel_names=["C0"],
        **vol_kw,
    )
    _rewrite_2005_extra(vsi, extra)
    return vsi


_register_prepare(
    "2001/2005/2020",
    lambda p: _prepare_2005_branch(p, [syn._string(2020, "10^-6m^1", 13)]),
)
_register_prepare(
    "2001/2005/2018",
    lambda p: _prepare_2005_branch(p, [syn._double_vector(2018, [0.0, 0.0])]),
)
_register_prepare(
    "2001/2005/2015",
    lambda p: _prepare_2005_branch(
        p, [syn._field_data(2015, 10, struct.pack("<d", 1_700_000_000.0))]
    ),
)


def _prepare_2074(tmp_path: Path) -> Path:
    vsi = _minimal_z2(tmp_path)
    write_minimal_vsi(
        vsi,
        stack_id=1,
        width=8,
        height=8,
        axis_order=["Z"],
        axis_sizes={"Z": 2},
        channel_names=["C0"],
        volume_2074=syn._field_inline(2074, 0),
    )
    return vsi


_register_prepare("2001/2005/2074", _prepare_2074)
_register_prepare(
    "2001/2005/2030",
    _default_stack,
)


def _prepare_cz(tmp_path: Path) -> Path:
    vsi = tmp_path / "syn.vsi"
    write_minimal_vsi(
        vsi,
        stack_id=1,
        width=8,
        height=8,
        axis_order=["C", "Z"],
        axis_sizes={"C": 1, "Z": 2},
        channel_names=["C0"],
    )
    return vsi


def _prepare_2417(tmp_path: Path) -> Path:
    vsi = _prepare_cz(tmp_path)
    rewrite_vsi(
        vsi,
        lambda n, enc: (
            syn._field_nested(
                n.tag,
                [enc(c) for c in n.children] + [syn._prop_double(2417, 500.0)],
                n.second,
            )
            if n.tag == 2008 and n.second == 0
            else None
        ),
        with_enc=True,
    )
    return vsi


_register_prepare("2001/2007/2008/2417", _prepare_2417)
_register_prepare("2001/2007/2008/2023", _default_stack)
_register_prepare("2001/2007/2008/2012_2013", _default_stack)
_register_prepare("2001/2007/2008/2021_2419", _default_stack)
_register_prepare("2001/2002/2018/20005", _default_stack)


def _prepare_camera(tmp_path: Path) -> Path:
    vsi = _default_stack(tmp_path)
    rewrite_vsi(
        vsi,
        lambda n: (
            syn._field_nested(
                2005,
                [
                    syn._double_vector(2019, [1.0, 1.0]),
                    syn._string(2030, "synthetic"),
                    _camera_device_field(),
                ],
            )
            if n.tag == 2005
            else None
        ),
    )
    return vsi


_register_prepare("2005/2043/120114/100049", _prepare_camera)


def prepare_vsi(tmp_path: Path, path_id: str) -> Path:
    fn = _PREPARE.get(path_id, _default_stack)
    return fn(tmp_path)


_LeafFn: TypeAlias = Callable[[], bytes | None]
_NodeMatch: TypeAlias = Callable[[FieldNode], bool]


@dataclass(frozen=True)
class _DupRule:
    path_id: str
    parent_tag: int
    child_tag: int
    pair: Callable[[bool], list[bytes]]


@dataclass(frozen=True)
class _LeafRule:
    path_id: str
    tag: int
    mutations: dict[str, _LeafFn]
    match: _NodeMatch = lambda _n: True


def _dup(path_id: str, parent: int, child: int, pair: Callable[[bool], list[bytes]]) -> _DupRule:
    return _DupRule(path_id, parent, child, pair)


_DUP_RULES: tuple[_DupRule, ...] = (
    _dup(
        "2000/2035",
        2000,
        2035,
        lambda f: [
            syn._int_vector(2035, [0x30016, 21199]),
            syn._int_vector(2035, [1, 2]),
        ][:: -1 if not f else 1],
    ),
    _dup(
        "2/21",
        2,
        21,
        lambda f: [
            syn._field_inline(21, 1),
            syn._field_inline(21, 2),
        ][:: -1 if not f else 1],
    ),
    _dup(
        "2001/2002/2018/2053",
        2018,
        2053,
        lambda f: [
            syn._int_vector(2053, [0, 0, 8, 8]),
            syn._int_vector(2053, [0, 0, 9, 9]),
        ][:: -1 if not f else 1],
    ),
    _dup(
        "2001/2003",
        2001,
        2003,
        lambda f: [
            syn._int_vector(2003, [2, 2]),
            syn._int_vector(2003, [3, 3]),
        ][:: -1 if not f else 1],
    ),
    _dup(
        "2001/2005/2019",
        2005,
        2019,
        lambda f: [
            syn._double_vector(2019, [1.0, 1.0]),
            syn._double_vector(2019, [9.0, 9.0]),
        ][:: -1 if not f else 1],
    ),
    _dup(
        "2001/2005/2020",
        2005,
        2020,
        lambda f: [
            syn._string(2020, "10^-6m^1", 13),
            syn._string(2020, "2"),
        ][:: -1 if not f else 1],
    ),
    _dup(
        "2001/2005/2018",
        2005,
        2018,
        lambda f: [
            syn._double_vector(2018, [0.0, 0.0]),
            syn._double_vector(2018, [1.0, 1.0]),
        ][:: -1 if not f else 1],
    ),
    _dup(
        "2001/2005/2015",
        2005,
        2015,
        lambda f: [
            syn._field_data(2015, 10, struct.pack("<d", 1.0)),
            syn._field_data(2015, 10, struct.pack("<d", 2.0)),
        ][:: -1 if not f else 1],
    ),
    _dup(
        "2001/2005/2030",
        2005,
        2030,
        lambda f: [
            syn._string(2030, "a"),
            syn._string(2030, "b"),
        ][:: -1 if not f else 1],
    ),
    _dup(
        "2001/2005/2074",
        2005,
        2074,
        lambda f: [
            syn._field_inline(2074, 0),
            syn._field_inline(2074, 1),
        ][:: -1 if not f else 1],
    ),
    _dup(
        "2001/2007/2008/2023",
        2008,
        2023,
        lambda f: [
            syn._field_data(2023, 6, struct.pack("<i", 1)),
            syn._field_data(2023, 6, struct.pack("<i", 2)),
        ][:: -1 if not f else 1],
    ),
    _dup(
        "2001/2007/2008/2012_2013",
        2008,
        2013,
        lambda f: [
            syn._prop_double(2013, 1.0),
            syn._prop_double(2013, 9.0),
        ][:: -1 if not f else 1],
    ),
    _dup(
        "2001/2007/2008/2021_2419",
        2008,
        2021,
        lambda f: [
            syn._string(2021, "a"),
            syn._string(2021, "b"),
        ][:: -1 if not f else 1],
    ),
    _dup(
        "2001/2007/2008/2417",
        2008,
        2417,
        lambda f: [
            syn._prop_double(2417, 500.0),
            syn._prop_double(2417, 600.0),
        ][:: -1 if not f else 1],
    ),
    _dup(
        "2001/2002/2006/2014_2017",
        2006,
        2017,
        lambda f: [
            syn._prop_double(2017, 1.0),
            syn._prop_double(2017, 2.0),
        ][:: -1 if not f else 1],
    ),
    _dup(
        "2001/2002/2018/20005",
        2018,
        20005,
        lambda f: [
            syn._field_inline(20005, 1),
            syn._field_inline(20005, 0),
        ][:: -1 if not f else 1],
    ),
    _dup(
        "2005/2043/120114/100049",
        120114,
        100049,
        lambda f: [
            syn._field_inline(100049, 8),
            syn._field_inline(100049, 16),
        ][:: -1 if not f else 1],
    ),
)


def _leaf(
    path_id: str,
    tag: int,
    mutations: dict[str, _LeafFn],
    match: _NodeMatch = lambda _n: True,
) -> _LeafRule:
    return _LeafRule(path_id, tag, mutations, match)


_LEAF_RULES: tuple[_LeafRule, ...] = (
    _leaf(
        "2000/2035",
        2035,
        {
            "string": lambda: syn._string(2035, CORRUPT),
            "fractional": lambda: syn._double_vector(2035, [1.5, 2.0]),
            "wrong_len": lambda: syn._int_vector(2035, [1, 2, 3]),
        },
    ),
    _leaf(
        "2/21",
        21,
        {
            "string": lambda: syn._string(21, CORRUPT),
            "fractional": lambda: syn._field_data(21, 10, struct.pack("<d", 1.5)),
        },
    ),
    _leaf(
        "2001/2002/2018/2053",
        2053,
        {
            "string": lambda: syn._string(2053, CORRUPT),
            "fractional": lambda: syn._double_vector(2053, [0.9, 0, 8.9, 8.9]),
            "wrong_len": lambda: syn._int_vector(2053, [0, 0, 8]),
        },
    ),
    _leaf(
        "2001/2003",
        2003,
        {
            "string": lambda: syn._string(2003, CORRUPT),
            "fractional": lambda: syn._double_vector(2003, [1.5, 2.0]),
            "wrong_len": lambda: syn._int_vector(2003, [2]),
        },
    ),
    _leaf(
        "2001/2005/2019",
        2019,
        {
            "string": lambda: syn._string(2019, CORRUPT),
            "fractional": lambda: syn._double_vector(2019, [1.5, 1.0]),
            "wrong_len": lambda: syn._double_vector(2019, [1.0]),
        },
    ),
    _leaf("2001/2005/2020", 2020, {"string": lambda: syn._string(2020, CORRUPT)}),
    _leaf(
        "2001/2005/2018",
        2018,
        {
            "string": lambda: syn._string(2018, CORRUPT),
            "fractional": lambda: syn._double_vector(2018, [1.5, 0.0]),
            "wrong_len": lambda: syn._double_vector(2018, [0.0]),
        },
    ),
    _leaf(
        "2001/2005/2015",
        2015,
        {
            "string": lambda: syn._string(2015, CORRUPT),
            "fractional": lambda: syn._field_data(2015, 10, struct.pack("<d", 1.5)),
        },
    ),
    _leaf("2001/2005/2030", 2030, {"string": lambda: syn._string(2030, CORRUPT)}),
    _leaf(
        "2001/2005/2074",
        2074,
        {
            "string": lambda: syn._string(2074, CORRUPT),
            "fractional": lambda: syn._field_data(2074, 10, struct.pack("<d", 1.5)),
        },
    ),
    _leaf(
        "2001/2007/2008/2023",
        2023,
        {
            "string": lambda: syn._string(2023, CORRUPT),
            "fractional": lambda: syn._field_data(2023, 10, struct.pack("<d", 1.9)),
        },
        match=lambda n: n.value == 1,
    ),
    _leaf(
        "2001/2007/2008/2012_2013",
        2013,
        {
            "string": lambda: _prop_corrupt_string(2013),
            "fractional": lambda: _prop_corrupt_fractional(2013),
        },
    ),
    _leaf("2001/2007/2008/2021_2419", 2021, {"string": lambda: syn._string(2021, CORRUPT)}),
    _leaf(
        "2001/2007/2008/2417",
        2417,
        {
            "string": lambda: _prop_corrupt_string(2417),
            "fractional": lambda: _prop_corrupt_fractional(2417),
        },
    ),
    _leaf(
        "2001/2002/2006/2014_2017",
        2017,
        {
            "string": lambda: _prop_corrupt_string(2017),
            "fractional": lambda: _prop_corrupt_fractional(2017),
        },
    ),
    _leaf(
        "2001/2002/2018/20005",
        20005,
        {
            "string": lambda: syn._string(20005, CORRUPT),
            "fractional": lambda: syn._field_data(20005, 10, struct.pack("<d", 1.5)),
        },
    ),
    _leaf(
        "2005/2043/120114/100049",
        100049,
        {
            "string": lambda: syn._string(100049, CORRUPT),
            "fractional": lambda: syn._field_data(100049, 10, struct.pack("<d", 1.5)),
        },
    ),
)


def mutation_transform(
    path_id: str,
    mutation: str,
) -> Callable[[FieldNode, Callable[[FieldNode], bytes]], bytes | None]:
    dup_first = mutation == "duplicate_first"

    def t(n: FieldNode, enc: Callable[[FieldNode], bytes]) -> bytes | None:
        if mutation.startswith("duplicate"):
            for dup_rule in _DUP_RULES:
                if dup_rule.path_id != path_id:
                    continue
                if n.tag == dup_rule.parent_tag:
                    pair = dup_rule.pair(dup_first)
                    return siblings_replace(n, enc, child_tag=dup_rule.child_tag, pair=pair)
        for leaf_rule in _LEAF_RULES:
            if leaf_rule.path_id != path_id or n.tag != leaf_rule.tag or not leaf_rule.match(n):
                continue
            if mutation not in leaf_rule.mutations:
                return None
            return leaf_rule.mutations[mutation]()
        return None

    return t
