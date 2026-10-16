"""Detect and replace absolute date/time values in numeric metadata."""

from __future__ import annotations

import struct
from typing import Any, cast

from bioio_vsi._tree import SCALAR, VEC

# Fixed replacements (same type width as detected family).
SENTINEL_UNIX_S = 1_600_000_000  # 2020-09-13 UTC
SENTINEL_UNIX_MS = SENTINEL_UNIX_S * 1000
# Windows FILETIME for 2020-09-13 (100-ns since 1601-01-01).
SENTINEL_FILETIME = 132_400_000_000_000_000

_UNIX_S_MIN = 631_152_000  # 1990-01-01
_UNIX_S_MAX = 4_102_444_800  # 2100-01-01
_UNIX_MS_MIN = _UNIX_S_MIN * 1000
_UNIX_MS_MAX = _UNIX_S_MAX * 1000
# FILETIME for 1990-01-01 .. 2100-01-01 (approximate window).
_FILETIME_MIN = 116_444_736_000_000_000
_FILETIME_MAX = 265_046_496_000_000_000


def is_plausible_unix_seconds(v: int) -> bool:
    return _UNIX_S_MIN <= v <= _UNIX_S_MAX


def is_plausible_unix_milliseconds(v: int) -> bool:
    return _UNIX_MS_MIN <= v <= _UNIX_MS_MAX


def is_plausible_filetime(v: int) -> bool:
    return _FILETIME_MIN <= v <= _FILETIME_MAX


def is_plausible_absolute_date_int(v: int) -> bool:
    if isinstance(v, bool):
        return False
    return (
        is_plausible_unix_seconds(v)
        or is_plausible_unix_milliseconds(v)
        or is_plausible_filetime(v)
    )


def is_plausible_absolute_date_float(v: float) -> bool:
    if v != v:  # NaN
        return False
    iv = int(v)
    if abs(v - iv) < 1e-3 and is_plausible_unix_seconds(iv):
        return True
    return _UNIX_MS_MIN <= v <= _UNIX_MS_MAX


def sanitize_scalar_int(v: int, *, value_type: int) -> tuple[int, bool]:
    if not is_plausible_absolute_date_int(v):
        return v, False
    if is_plausible_filetime(v):
        return SENTINEL_FILETIME, True
    if is_plausible_unix_milliseconds(v):
        return SENTINEL_UNIX_MS, True
    return SENTINEL_UNIX_S, True


def sanitize_scalar_float(v: float) -> tuple[float, bool]:
    if not is_plausible_absolute_date_float(v):
        return v, False
    return float(SENTINEL_UNIX_S), True


def _pack_scalar(value_type: int, value: int | float) -> bytes:
    fmt = SCALAR[value_type]
    return struct.pack(f"<{fmt}", value)


def _sanitize_numeric_raw(value_type: int, raw: bytes) -> tuple[bytes, bool]:
    if value_type in SCALAR:
        fmt = SCALAR[value_type]
        size = struct.calcsize(fmt)
        if len(raw) < size:
            return raw, False
        val = cast(int | float, struct.unpack(f"<{fmt}", raw[:size])[0])
        if fmt in ("b", "B", "h", "H", "i", "I", "q", "Q"):
            if isinstance(val, bool):
                return raw, False
            new_i, hit = sanitize_scalar_int(int(val), value_type=value_type)
            if hit:
                return _pack_scalar(value_type, new_i) + raw[size:], True
        elif fmt in ("f", "d"):
            new_f, hit = sanitize_scalar_float(float(val))
            if hit:
                return _pack_scalar(value_type, new_f) + raw[size:], True
        return raw, False
    if value_type in VEC:
        fmt = VEC[value_type]
        elsize = struct.calcsize(fmt)
        if elsize == 0 or len(raw) % elsize:
            return raw, False
        n = len(raw) // elsize
        vals = list(struct.unpack(f"<{n}{fmt}", raw))
        out_vals: list[Any] = []
        any_hit = False
        for v in vals:
            hit = False
            if fmt in ("i", "I", "q", "Q", "h", "H", "b", "B"):
                new_i, hit = sanitize_scalar_int(int(v), value_type=value_type)
                out_vals.append(new_i)
            elif fmt in ("f", "d"):
                new_f, hit = sanitize_scalar_float(float(v))
                out_vals.append(new_f)
            else:
                out_vals.append(v)
            any_hit = any_hit or hit
        if any_hit:
            return struct.pack(f"<{n}{fmt}", *out_vals), True
        return raw, False
    return raw, False


def sanitize_inline_bits(value_type: int, size: int) -> tuple[int, bool]:
    if value_type not in (5, 6, 14, 271, 4104):
        return size, False
    bits = size & 0xFFFFFFFF
    v = struct.unpack("<i", struct.pack("<I", bits))[0] if value_type == 5 else bits
    nv, hit = sanitize_scalar_int(int(v), value_type=value_type)
    if not hit:
        return size, False
    if value_type == 5:
        return struct.unpack("<I", struct.pack("<i", nv))[0], True
    return nv & 0xFFFFFFFF, True
