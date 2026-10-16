"""Typed reads of interpreted VSI metadata paths."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from bioio_vsi import errors
from bioio_vsi._tree import FieldNode, UninterpretedInline, children, kids

PathRef = str

UNIT_MICROMETRES = "10^-6m^1"
UNIT_MILLISECONDS = "10^-3s^1"
UNIT_NANOMETRES = "10^-9m^1"
UNIT_TAG = 268435456
VALUE_TAG = 268435458


@dataclass(frozen=True)
class PixelSizeUm:
    """Physical pixel size from VSI tag 2005/2019: x then y, in micrometres."""

    x: float
    y: float


def _type_label(value: Any) -> str:
    if isinstance(value, UninterpretedInline):
        return f"uninterpreted inline type {value.value_type}"
    return type(value).__name__


def _reject_uninterpreted(value: Any, *, path: PathRef, tag: int) -> None:
    if isinstance(value, UninterpretedInline):
        raise errors.UnsupportedFeatureError(
            f"{path}: tag {tag} is inline value type {value.value_type}, "
            "which this release does not interpret",
            path=path,
        )


def matching_children(
    node: FieldNode | None,
    tag: int,
    second: int | None = None,
) -> list[FieldNode]:
    if node is None:
        return []
    return [
        c
        for c in children(node)
        if c.tag == tag and (second is None or c.second == second)
    ]


def children_unique_by_second(
    node: FieldNode | None,
    *,
    path: PathRef,
    member: str,
) -> list[FieldNode]:
    """Return all children, refusing two with the same ``second`` identifier."""
    if node is None:
        return []
    found = children(node)
    seen: set[int | None] = set()
    for child_node in found:
        ident = child_node.second
        if ident in seen:
            raise errors.AmbiguousMetadataError(
                f"{path}: duplicate {member} identifier {ident!r}",
                path=path,
            )
        seen.add(ident)
    return found


def kids_unique_by_second(
    node: FieldNode | None,
    tag: int,
    *,
    path: PathRef,
    member: str,
) -> list[FieldNode]:
    """Return every child with ``tag``, refusing duplicate ``second`` identifiers."""
    if node is None:
        return []
    found = kids(node, tag)
    seen: set[int | None] = set()
    for child_node in found:
        ident = child_node.second
        if ident in seen:
            raise errors.AmbiguousMetadataError(
                f"{path}/{tag}: duplicate {member} identifier {ident!r}",
                path=path,
            )
        seen.add(ident)
    return found


def unique_child(
    node: FieldNode | None,
    tag: int,
    *,
    path: PathRef,
    second: int | None = None,
) -> FieldNode | None:
    matches = matching_children(node, tag, second)
    if len(matches) > 1:
        where = f"{path}/{tag}"
        if second is not None:
            where = f"{path}/{tag}[{second}]"
        raise errors.AmbiguousMetadataError(
            f"{where}: {len(matches)} sibling fields match, expected at most one",
            path=path,
        )
    return matches[0] if matches else None


def read_optional_int(value: Any, *, path: PathRef, field: str) -> int | None:
    if value is None:
        return None
    _reject_uninterpreted(value, path=path, tag=0)
    if isinstance(value, bool) or not isinstance(value, int):
        raise errors.InvalidMetadataError(
            f"{path}: {field} must be an integer, got {_type_label(value)}",
            path=path,
        )
    return value


def read_required_int(value: Any, *, path: PathRef, field: str) -> int:
    out = read_optional_int(value, path=path, field=field)
    if out is None:
        raise errors.InvalidMetadataError(
            f"{path}: {field} is missing",
            path=path,
        )
    return out


def read_optional_finite_float(
    value: Any,
    *,
    path: PathRef,
    field: str,
    allow_nan: bool = False,
) -> float | None:
    if value is None:
        return None
    _reject_uninterpreted(value, path=path, tag=0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise errors.InvalidMetadataError(
            f"{path}: {field} must be a number, got {_type_label(value)}",
            path=path,
        )
    fv = float(value)
    if math.isnan(fv) and allow_nan:
        return fv
    if not math.isfinite(fv):
        raise errors.InvalidMetadataError(
            f"{path}: {field} must be finite, got {value!r}",
            path=path,
        )
    return fv


def read_int_list(
    value: Any,
    *,
    path: PathRef,
    field: str,
    length: int | None = None,
    min_value: int | None = None,
) -> list[int]:
    if value is None:
        raise errors.InvalidMetadataError(
            f"{path}: {field} is missing",
            path=path,
        )
    if not isinstance(value, (list, tuple)):
        raise errors.InvalidMetadataError(
            f"{path}: {field} must be a vector, got {_type_label(value)}",
            path=path,
        )
    if length is not None and len(value) != length:
        raise errors.InvalidMetadataError(
            f"{path}: {field} must have length {length}, got {len(value)}",
            path=path,
        )
    out: list[int] = []
    for i, item in enumerate(value):
        if isinstance(item, UninterpretedInline):
            raise errors.UnsupportedFeatureError(
                f"{path}: {field}[{i}] is uninterpreted inline type {item.value_type}",
                path=path,
            )
        if isinstance(item, bool) or not isinstance(item, int):
            raise errors.InvalidMetadataError(
                f"{path}: {field}[{i}] must be an integer, got {_type_label(item)}",
                path=path,
            )
        if min_value is not None and item < min_value:
            raise errors.InvalidMetadataError(
                f"{path}: {field}[{i}] must be >= {min_value}, got {item}",
                path=path,
            )
        out.append(item)
    return out


def read_double_list(
    value: Any,
    *,
    path: PathRef,
    field: str,
    length: int,
    positive: bool = False,
) -> list[float]:
    if value is None:
        raise errors.InvalidMetadataError(
            f"{path}: {field} is missing",
            path=path,
        )
    if not isinstance(value, (list, tuple)):
        raise errors.InvalidMetadataError(
            f"{path}: {field} must be a vector, got {_type_label(value)}",
            path=path,
        )
    if len(value) != length:
        raise errors.InvalidMetadataError(
            f"{path}: {field} must have length {length}, got {len(value)}",
            path=path,
        )
    out: list[float] = []
    for i, item in enumerate(value):
        if isinstance(item, UninterpretedInline):
            raise errors.UnsupportedFeatureError(
                f"{path}: {field}[{i}] is uninterpreted inline type {item.value_type}",
                path=path,
            )
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise errors.InvalidMetadataError(
                f"{path}: {field}[{i}] must be a number, got {_type_label(item)}",
                path=path,
            )
        fv = float(item)
        if not math.isfinite(fv):
            raise errors.InvalidMetadataError(
                f"{path}: {field}[{i}] must be finite, got {item!r}",
                path=path,
            )
        if positive and fv <= 0:
            raise errors.InvalidMetadataError(
                f"{path}: {field}[{i}] must be positive, got {fv}",
                path=path,
            )
        out.append(fv)
    return out


def read_boundary_rect(value: Any, *, path: PathRef) -> tuple[int, int, int, int]:
    ints = read_int_list(value, path=path, field="2053", length=4, min_value=0)
    x0, y0, w, h = ints
    if (x0, y0) != (0, 0):
        raise errors.InvalidStructureError(
            f"{path}: boundary origin ({x0}, {y0}) is not (0, 0)",
            path=path,
        )
    if w <= 0 or h <= 0:
        raise errors.InvalidMetadataError(
            f"{path}: boundary width and height must be positive, got {w}x{h}",
            path=path,
        )
    return x0, y0, w, h


def read_dimension_sizes(value: Any, *, path: PathRef) -> list[int]:
    return read_int_list(value, path=path, field="2003", min_value=1)


def read_pixel_size_pair(value: Any, *, path: PathRef) -> PixelSizeUm:
    xs = read_double_list(value, path=path, field="2019", length=2, positive=True)
    return PixelSizeUm(x=xs[0], y=xs[1])


def read_enum_int(
    value: Any,
    *,
    path: PathRef,
    field: str,
    allowed: frozenset[int],
) -> int:
    iv = read_required_int(value, path=path, field=field)
    if iv not in allowed:
        raise errors.InvalidMetadataError(
            f"{path}: {field} value {iv} is not one of {sorted(allowed)}",
            path=path,
        )
    return iv


def read_dimension_kind_code(value: Any, *, path: PathRef, field: str) -> int:
    if value is None:
        raise errors.InvalidMetadataError(
            f"{path}: {field} is missing",
            path=path,
        )
    _reject_uninterpreted(value, path=path, tag=2023)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise errors.InvalidMetadataError(
            f"{path}: {field} must be a numeric kind code, got {_type_label(value)}",
            path=path,
        )
    fv = float(value)
    if not math.isfinite(fv) or fv != math.trunc(fv):
        raise errors.InvalidMetadataError(
            f"{path}: {field} must be an integer kind code, got {value!r}",
            path=path,
        )
    return int(fv)


def read_optional_string(value: Any, *, path: PathRef, field: str) -> str | None:
    if value is None:
        return None
    _reject_uninterpreted(value, path=path, tag=0)
    if not isinstance(value, str):
        raise errors.InvalidMetadataError(
            f"{path}: {field} must be a string, got {_type_label(value)}",
            path=path,
        )
    return value


def read_property_float(
    node: FieldNode | None,
    *,
    path: PathRef,
    expected_unit: str,
    allow_nan: bool = False,
) -> float | None:
    if node is None:
        return None
    unit_node = unique_child(node, 268435456, path=path)
    if unit_node is not None:
        unit_val = unit_node.value
        read_optional_string(unit_val, path=path, field="unit")
        if str(unit_val) != expected_unit:
            raise errors.UnknownUnitError(
                f"{path}: unit field {unit_val!r} != expected {expected_unit!r}",
                path=path,
            )
    val_node = unique_child(node, 268435458, path=path)
    if val_node is None:
        return None
    return read_optional_finite_float(
        val_node.value,
        path=path,
        field="268435458",
        allow_nan=allow_nan,
    )


def read_format_version_raw(value: Any, *, path: PathRef) -> list[int]:
    return read_int_list(value, path=path, field="2035", length=2, min_value=0)


def read_stage_origin(value: Any, *, path: PathRef) -> Sequence[float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise errors.InvalidMetadataError(
            f"{path}: 2018 must be a vector, got {_type_label(value)}",
            path=path,
        )
    out: list[float] = []
    for i, item in enumerate(value):
        fv = read_optional_finite_float(item, path=path, field=f"2018[{i}]")
        if fv is None:
            raise errors.InvalidMetadataError(
                f"{path}: 2018[{i}] is missing",
                path=path,
            )
        out.append(fv)
    return out


def read_optional_unix_time(value: Any, *, path: PathRef, field: str) -> float | None:
    return read_optional_finite_float(value, path=path, field=field)


def node_value(node: FieldNode | None, *, path: PathRef, tag: int) -> Any:
    child_node = unique_child(node, tag, path=path)
    return None if child_node is None else child_node.value


def prop_value(
    node: FieldNode | None,
    *,
    path: str,
    expected_unit: str,
    allow_nan: bool = False,
) -> float | None:
    return read_property_float(
        node,
        path=path,
        expected_unit=expected_unit,
        allow_nan=allow_nan,
    )


def camera_bit_depth_from_path(sp: FieldNode | None, vsi_path: str) -> int | None:
    """2005/2043/<device>/120114/100049 (format.md); never search by tag number alone."""
    if sp is None:
        return None
    sp_path = f"{vsi_path}/2005"
    mic = unique_child(sp, 2043, path=sp_path)
    if mic is None:
        return None
    mic_path = f"{sp_path}/2043"
    depths: list[int] = []
    for ord_i, dev in enumerate(children(mic)):
        row: FieldNode | None
        row_path: str
        if dev.tag == 120114:
            row = dev
            row_path = f"{mic_path}/120114"
        else:
            dev_path = f"{mic_path}/{ord_i}"
            row = unique_child(dev, 120114, path=dev_path)
            if row is None:
                continue
            row_path = f"{dev_path}/120114"
        bd = unique_child(row, 100049, path=row_path)
        if bd is not None and bd.value is not None:
            iv = read_optional_int(
                bd.value,
                path=f"{row_path}/100049",
                field="100049",
            )
            if iv is not None:
                depths.append(iv)
    if not depths:
        return None
    if len(set(depths)) > 1:
        return None
    return depths[0]
