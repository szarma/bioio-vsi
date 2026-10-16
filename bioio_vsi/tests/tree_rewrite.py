"""Rewrite parsed VSI trees on disk for corruption tests."""

from __future__ import annotations

import struct
from collections.abc import Callable
from typing import Any

from bioio_vsi._tree import FieldNode, load_tree
from bioio_vsi.tests import synthetic as syn

CORRUPT_META_MARKER = "CORRUPT_META_MARKER_XYZ"


def nested_volume_pos(data: bytes | bytearray, field: FieldNode) -> int:
    off = field.file_offset
    ft, tag, nxt, size = struct.unpack_from("<iiiI", data, off)
    return off + 16 + (4 if ft & 0x08000000 else 0)


def rewrite_vsi(
    vsi: Any,
    transform: Callable[..., bytes | None],
    *,
    with_enc: bool = False,
) -> None:
    tree = load_tree(str(vsi))

    def enc(n: FieldNode) -> bytes:
        replacement = transform(n, enc) if with_enc else transform(n)
        if replacement is not None:
            return replacement
        if n.children is not None:
            return syn._field_nested(n.tag, [enc(c) for c in n.children], n.second)
        off = n.file_offset
        ft, tag, nxt, size = struct.unpack_from("<iiiI", tree.data, off)
        hdrlen = 16 + (4 if n.second is not None else 0)
        return tree.data[off : off + hdrlen + (0 if n.inline else size)]

    vsi.write_bytes(
        b"II*\0" + struct.pack("<I", 0) + syn._build_volume([enc(n) for n in tree.root])
    )


def siblings_replace(
    n: FieldNode,
    enc: Callable[[FieldNode], bytes],
    *,
    child_tag: int,
    pair: list[bytes],
) -> bytes:
    rest = [enc(c) for c in n.children or [] if c.tag != child_tag]
    return syn._field_nested(n.tag, pair + rest, n.second)
