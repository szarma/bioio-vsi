"""Allowlist copy for the JPEG thumbnail TIFF reachable from the VSI file header."""

from __future__ import annotations

import struct
from collections.abc import Callable

_TIFF_TYPES = {
    1: 1,  # BYTE
    2: 1,  # ASCII
    3: 2,  # SHORT
    4: 4,  # LONG
    5: 8,  # RATIONAL
    6: 1,
    7: 1,  # UNDEFINED
    8: 2,
    9: 4,
    10: 8,
    11: 4,
    12: 8,
}

_STRIP_DATA_TAGS = frozenset({273, 279, 288, 289, 291, 513, 514, 347})
_COPY_VALUE_TAGS = frozenset(
    {
        256,
        257,
        258,
        259,
        262,
        263,
        264,
        265,
        266,
        269,
        270,
        271,
        272,
        277,
        278,
        280,
        281,
        282,
        283,
        284,
        296,
        305,
        306,
        315,
        316,
        317,
        318,
        319,
        320,
        321,
        322,
        323,
        324,
        325,
        330,
        332,
        334,
        336,
        337,
        338,
        339,
        340,
        341,
        342,
        343,
        344,
        345,
        346,
        512,
        530,
    }
)


def _copy_range(
    src: bytes,
    out: bytearray,
    start: int,
    length: int,
    mark: Callable[[int, int, str], None],
) -> None:
    if length <= 0:
        return
    end = start + length
    if start < 0 or end > len(src) or end > len(out):
        return
    out[start:end] = src[start:end]
    mark(start, length, "structure")


def _copy_value_bytes(
    src: bytes,
    out: bytearray,
    offset: int,
    length: int,
    mark: Callable[[int, int, str], None],
    *,
    cls: str,
) -> None:
    if length <= 0:
        return
    end = offset + length
    if offset < 0 or end > len(src) or end > len(out):
        return
    out[offset:end] = src[offset:end]
    mark(offset, length, cls)


def sanitize_vsi_thumbnail_tiff(
    src: bytes,
    out: bytearray,
    mark: Callable[[int, int, str], None],
) -> None:
    """Copy IFD structure and numeric TIFF values; leave strip/JPEG bytes zeroed."""
    if len(src) < 8 or src[:4] != b"II*\x00":
        return
    _copy_range(src, out, 0, 8, mark)
    ifd0 = struct.unpack_from("<I", src, 4)[0]
    if ifd0 == 0 or ifd0 >= len(src):
        return
    seen_ifds: set[int] = set()
    cur = ifd0
    while cur and cur not in seen_ifds and cur + 2 <= len(src):
        seen_ifds.add(cur)
        ntags = struct.unpack_from("<H", src, cur)[0]
        _copy_range(src, out, cur, 2, mark)
        entry_base = cur + 2
        need = entry_base + 12 * ntags + 4
        if need > len(src):
            break
        for i in range(ntags):
            ent = entry_base + 12 * i
            tag, typ, count = struct.unpack_from("<HHI", src, ent)
            _copy_range(src, out, ent, 12, mark)
            tsize = _TIFF_TYPES.get(typ, 0)
            if tsize == 0:
                continue
            total = tsize * count
            if total <= 4:
                if typ == 2 or typ == 7:
                    continue
                if tag in _COPY_VALUE_TAGS or tag not in _STRIP_DATA_TAGS:
                    _copy_value_bytes(src, out, ent + 8, 4, mark, cls="numeric")
                continue
            val_off = struct.unpack_from("<I", src, ent + 8)[0]
            if val_off + total > len(src):
                continue
            if typ == 2 or typ == 7:
                continue
            if tag in _STRIP_DATA_TAGS:
                continue
            if tag in _COPY_VALUE_TAGS or typ in (3, 4, 5, 8, 9, 10, 11, 12):
                _copy_value_bytes(src, out, val_off, total, mark, cls="numeric")
        next_ifd = struct.unpack_from("<I", src, entry_base + 12 * ntags)[0]
        _copy_range(src, out, entry_base + 12 * ntags, 4, mark)
        cur = next_ifd
