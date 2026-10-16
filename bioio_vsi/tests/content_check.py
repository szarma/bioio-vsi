"""Find content (pictures, text) left in numeric arrays of an anonymised .vsi."""

from __future__ import annotations

import struct
from collections.abc import Callable

from bioio_vsi._tree import load_tree, walk

# A 96x96, 24-bit BITMAPINFOHEADER: the specimen preview cellSens embeds.
PREVIEW_HEADER = struct.pack("<IiiHH", 40, 96, 96, 1, 24)
PREVIEW_PIXELS = 96 * 96 * 3


def varied_byte_arrays(data: bytes, *, min_len: int = 256, max_distinct: int = 8) -> list[str]:
    """Tree byte arrays longer than min_len with more than max_distinct values."""
    found = []
    for node in walk(load_tree(data).root):
        value = node.value
        if node.inline or not isinstance(value, list) or len(value) <= min_len:
            continue
        if node.value_type in (1, 2, 12, 269) and len(set(value)) > max_distinct:
            found.append(f"tag {node.tag} type {node.value_type} len {len(value)}")
    return found


def preview_pixels_left(data: bytes) -> int:
    """Non-zero bytes after every preview header, found by scanning the raw file."""
    left = 0
    start = data.find(PREVIEW_HEADER)
    while start != -1:
        pixels = data[start + 40 : start + 40 + PREVIEW_PIXELS]
        left += len(pixels) - pixels.count(0)
        start = data.find(PREVIEW_HEADER, start + 1)
    return left


# Numbers allowed in a file with synthetic numbers, by (parent tag, tag) of the
# field or of the property that holds the value. Written out here rather than
# imported, so the test does not share the rules it checks.
_KEPT = {(2, 21), (2000, 2035), (2001, 2003), (2008, 2023), (2018, 2053), (2018, 20005)}
_KEPT |= {(2005, 2074), (120114, 100049)}
_SYNTHETIC: dict[tuple[int, ...], Callable[[list[float]], bool]] = {
    (2005, 2019): lambda v: v[0] == 0.25,
    (2005, 2018): lambda v: v == [1000.0 * (i + 1) for i in range(len(v))],
    (2005, 2015): lambda v: v == [1600000000.0],
    (2008, 2012): lambda v: v == [100.0],
    (2008, 2013): lambda v: v in ([1.0], [-1.0], [0.0]),
    (2006, 2014): lambda v: v[0] != v[0] or (v[0] - 100.0).is_integer(),
    (2006, 2017): lambda v: v[0] != v[0] or v[0] % 1000.0 == 0,
    (2008, 2417): lambda v: v[0] in (460.0, 525.0, 595.0, 670.0),
}


def measured_numbers(data: bytes) -> list[str]:
    """Numbers that are neither a kept format fact, a synthetic value, nor zero."""
    from bioio_vsi._tree import INLINE_VALUE_TYPES, SCALAR, VEC, FieldNode

    found: list[str] = []

    def visit(node: FieldNode, tags: tuple[int, ...]) -> None:
        tags = (*tags, node.tag)
        pairs = set(zip(tags, tags[1:], strict=False))
        numeric = (node.inline and node.value_type in INLINE_VALUE_TYPES) or (
            not node.inline
            and not node.extended
            and (node.value_type in SCALAR or node.value_type in VEC)
        )
        if numeric and not pairs & _KEPT and node.tag != 268435459:
            values = node.value if isinstance(node.value, list) else [node.value]
            values = [float(v) for v in values if v is not None]
            owner = tags[-3:-1] if node.tag == 268435458 else tags[-2:]
            check = _SYNTHETIC.get(owner)
            ok = check(values) if check else all(v == 0 for v in values)
            if not ok:
                found.append("/".join(map(str, tags)))
        for child in node.children or []:
            visit(child, tags)

    for root in load_tree(data).root:
        visit(root, ())
    return found
