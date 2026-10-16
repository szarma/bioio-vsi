"""Replace the numbers of an anonymised .vsi with synthetic ones.

After anonymising, strings and dates are gone but numbers are as measured:
pixel size, stage position, z positions, plane times, wavelengths, device
settings. This pass keeps only the numbers that describe the file's structure,
gives the measurements the reader reports fixed synthetic values with the same
shape (z direction, X = Y, channel count), and zeroes every other number.

Rules match (parent tag, tag) pairs because a tag number means different
things under different parents (2003 is the dimension sizes under an image
and display limits under a channel).
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterator

from bioio_vsi._tree import (
    FLD,
    INLINE_VALUE_TYPES,
    SCALAR,
    VEC,
    FieldNode,
    load_tree,
)

VALUE_TAG = 268435458  # a property's value
UNIT_KIND_TAG = 268435459  # constant next to every property value

# Numbers that are format facts: kept as stored.
KEEP = frozenset(
    {
        (2, 21),  # writer build
        (2000, 2035),  # image count
        (2001, 2003),  # dimension sizes
        (2008, 2023),  # dimension kind of a coordinate slot
        (2018, 2053),  # frame indices
        (2018, 20005),
        (2005, 2074),  # image type
        (120114, 100049),  # camera bit depth (the reader's dtype check)
    }
)

PIXEL_SIZE_UM = 0.25
STAGE_ORIGIN_UM = 1000.0
Z_START_UM = 100.0
Z_STEP_UM = 1.0
PLANE_INTERVAL_MS = 1000.0
EMISSION_NM = (460.0, 525.0, 595.0, 670.0)
ACQUISITION_TIME_S = 1600000000.0  # the anonymiser's fixed date


def _walk(
    nodes: list[FieldNode], parents: tuple[FieldNode, ...] = ()
) -> Iterator[tuple[FieldNode, tuple[FieldNode, ...]]]:
    for node in nodes:
        yield node, parents
        if node.children:
            yield from _walk(node.children, (*parents, node))


def _pairs(node: FieldNode, parents: tuple[FieldNode, ...]) -> list[tuple[int, int]]:
    tags = [p.tag for p in parents] + [node.tag]
    return list(zip(tags, tags[1:], strict=False))


def _payload(node: FieldNode, data: bytes) -> tuple[int, str] | None:
    """Offset and struct element format of a numeric payload, if it is one."""
    if node.extended or node.inline:
        return None
    fmt = SCALAR.get(node.value_type) or VEC.get(node.value_type)
    if fmt is None:
        return None
    ft, tag, _nxt, size = FLD.unpack_from(data, node.file_offset)
    if (ft >> 24) & 0xFF == 0x28 and tag in (2023, 2024):
        return None  # data stored elsewhere; the anonymiser left it zeroed
    q = node.file_offset + 16 + (4 if ft & 0x08000000 else 0)
    if q + size > len(data):
        raise ValueError(f"tag {tag}: payload runs past the end of the file")
    return q, fmt


def _write(out: bytearray, data: bytes, node: FieldNode, values: list[float]) -> None:
    where = _payload(node, data)
    assert where is not None
    q, fmt = where
    count = FLD.unpack_from(data, node.file_offset)[3] // struct.calcsize(fmt)
    assert count == len(values), (node.tag, count, len(values))
    typed = [float(v) if fmt in "fd" else round(v) for v in values]
    struct.pack_into(f"<{count}{fmt}", out, q, *typed)


def _old(node: FieldNode) -> list[float]:
    v = node.value
    return [float(x) for x in v] if isinstance(v, list) else [float(v)]


def synthesize_numbers(data: bytes) -> bytes:
    """Return the anonymised .vsi bytes with synthetic numbers."""
    tree = load_tree(data)
    out = bytearray(data)
    nodes = list(_walk(tree.root))

    # z start and step of each image, to map stored z positions to plane index.
    z_axis: dict[int, tuple[float, float]] = {}
    for node, parents in nodes:
        pairs = _pairs(node, parents)
        if node.tag == VALUE_TAG and pairs[-2:-1] in ([(2008, 2012)], [(2008, 2013)]):
            image = next((p.file_offset for p in parents if p.tag == 2001), -1)
            start, step = z_axis.get(image, (math.nan, math.nan))
            value = _old(node)[0]
            z_axis[image] = (value, step) if pairs[-2][1] == 2012 else (start, value)

    frame_index: dict[int, int] = {}
    for node, parents in nodes:
        if not _payload(node, data) and not (node.inline and node.value_type in INLINE_VALUE_TYPES):
            continue
        pairs = _pairs(node, parents)
        if any(p in KEEP for p in pairs) or node.tag == UNIT_KIND_TAG:
            continue
        image = next((p.file_offset for p in parents if p.tag == 2001), -1)
        owner = (
            pairs[-2] if node.tag == VALUE_TAG and len(pairs) >= 2 else pairs[-1] if pairs else None
        )

        if owner == (2005, 2019):
            x, y = _old(node)[:2]
            _write(out, data, node, [PIXEL_SIZE_UM, PIXEL_SIZE_UM * (y / x if x else 1.0)])
        elif owner == (2005, 2018):
            _write(out, data, node, [STAGE_ORIGIN_UM * (i + 1) for i in range(len(_old(node)))])
        elif owner == (2005, 2015):
            _write(out, data, node, [ACQUISITION_TIME_S])
        elif owner == (2008, 2012):
            _write(out, data, node, [Z_START_UM])
        elif owner == (2008, 2013):
            step = _old(node)[0]
            _write(out, data, node, [math.copysign(Z_STEP_UM, step) if step else 0.0])
        elif owner == (2006, 2014):
            z = _old(node)[0]
            start, step = z_axis.get(image, (math.nan, math.nan))
            if math.isnan(z):
                continue
            k = round((z - start) / step) if step and not math.isnan(start + step) else 0
            new_step = math.copysign(Z_STEP_UM, step) if step and not math.isnan(step) else 0.0
            _write(out, data, node, [Z_START_UM + k * new_step])
        elif owner == (2006, 2017):
            if math.isnan(_old(node)[0]):
                continue
            k = frame_index.setdefault(node.file_offset, len(frame_index))
            _write(out, data, node, [k * PLANE_INTERVAL_MS])
        elif owner == (2008, 2417):
            channel = next(p for p in reversed(parents) if p.tag == 2008)
            siblings = next(p for p in reversed(parents) if p.tag == 2007).children or []
            ordinal = [c.file_offset for c in siblings if c.tag == 2008].index(channel.file_offset)
            _write(out, data, node, [EMISSION_NM[ordinal % len(EMISSION_NM)]])
        elif node.inline:
            struct.pack_into("<I", out, node.file_offset + 12, 0)
        else:
            where = _payload(node, data)
            assert where is not None
            q, _fmt = where
            size = FLD.unpack_from(data, node.file_offset)[3]
            out[q : q + size] = bytes(size)
    return bytes(out)


__all__ = ["synthesize_numbers"]
