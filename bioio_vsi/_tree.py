"""Parse the proprietary tag tree in a cellSens .vsi file.

Volume and field layout follow slideio's BSD-3 VSI driver
(src/slideio/drivers/vsi), then verified against sample files.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from bioio_vsi._paths import resolve_path
from bioio_vsi.errors import (
    AmbiguousMetadataError,
    InvalidStructureError,
    TruncatedFileError,
)

# Value types whose inline payload is the four bytes of dataSize (observed in files).
INLINE_VALUE_TYPES: frozenset[int] = frozenset({5, 6, 12, 14, 271, 4104})


@dataclass(frozen=True)
class UninterpretedInline:
    """Inline bits for a value type this release does not decode."""

    value_type: int
    raw_bits: int


VOL = struct.Struct("<HHIqII")
FLD = struct.Struct("<iiiI")

SCALAR: dict[int, str] = {
    1: "b",
    2: "B",
    3: "h",
    4: "H",
    5: "i",
    6: "I",
    7: "q",
    8: "Q",
    9: "f",
    10: "d",
    11: "d",
    12: "B",
    14: "I",
    17: "q",
    18: "q",
    271: "I",
    272: "I",
    273: "I",
    8470: "I",
}
VEC: dict[int, str] = {
    256: "i",
    257: "i",
    258: "i",
    259: "i",
    260: "d",
    261: "d",
    262: "d",
    263: "d",
    264: "d",
    265: "d",
    266: "d",
    267: "i",
    268: "d",
    269: "B",
    270: "B",
    274: "i",
    275: "i",
    276: "i",
    277: "i",
    279: "d",
    280: "d",
    8195: "i",
    8199: "i",
    8200: "i",
}


@dataclass
class FieldNode:
    tag: int
    second: int | None
    value_type: int
    inline: bool
    extended: bool
    value: Any
    children: list[FieldNode] | None = None
    file_offset: int = 0


@dataclass
class VsiTree:
    """Parsed metadata tree and raw bytes (for pickling and bounds checks)."""

    path: str
    data: bytes
    root: list[FieldNode]
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.data)


def _check_bounds(buf: bytes, pos: int, need: int, context: str) -> None:
    if pos < 0 or pos + need > len(buf):
        raise TruncatedFileError(
            f"{context}: need {need} bytes at offset {pos}, file size is {len(buf)}"
        )


def _decode_inline(value_type: int, size: int) -> Any:
    bits = size & 0xFFFFFFFF
    if value_type == 5:
        return struct.unpack("<i", struct.pack("<I", bits))[0]
    if value_type in (6, 14, 271, 4104):
        return bits
    if value_type == 12:
        return 1 if bits else 0
    return UninterpretedInline(value_type=value_type, raw_bits=bits)


def _decode_value(value_type: int, raw: bytes) -> Any:
    # Type 13 holds UTF-16 too (unit strings such as "10^-6m^1"), unterminated.
    if value_type in (13, 8192, 8194):
        return raw.decode("utf-16-le", "replace").split("\0")[0]
    if value_type in SCALAR:
        fmt = SCALAR[value_type]
        n = len(raw) // struct.calcsize(fmt)
        vals = struct.unpack(f"<{n}{fmt}", raw[: n * struct.calcsize(fmt)])
        return vals[0] if n == 1 else list(vals)
    if value_type in VEC:
        fmt = VEC[value_type]
        n = len(raw) // struct.calcsize(fmt)
        return list(struct.unpack(f"<{n}{fmt}", raw[: n * struct.calcsize(fmt)]))
    return {"raw_type": value_type, "len": len(raw)}


def _parse_volume(
    buf: bytes,
    pos: int,
    stats: dict[str, Any],
    seen_offsets: set[int],
) -> tuple[list[FieldNode], int]:
    _check_bounds(buf, pos, 24, "volume header")
    hs, magic, ver, first, flags, _ = VOL.unpack_from(buf, pos)
    if hs != 24 or magic != 21321:
        raise InvalidStructureError(f"volume at {pos}: bad header (size={hs}, magic={magic})")
    if first < 0:
        raise InvalidStructureError(f"volume at {pos}: negative offsetFirstField")
    stats.setdefault("vol_versions", set()).add(ver)
    count = flags & 0x0FFFFFFF
    if count == 0:
        return [], pos + 24
    p = pos + first
    out: list[FieldNode] = []
    end = p
    vol_seen: set[int] = set()
    last_nxt = 0
    for field_i in range(count):
        if p in seen_offsets:
            raise InvalidStructureError(
                f"volume at {pos}: field offset {p} was already visited in this file"
            )
        if p in vol_seen:
            raise InvalidStructureError(
                f"volume at {pos}: field offset {p} was already visited in this volume"
            )
        seen_offsets.add(p)
        vol_seen.add(p)
        _check_bounds(buf, p, 16, "field header")
        ft, tag, nxt, size = FLD.unpack_from(buf, p)
        last_nxt = nxt
        node = FieldNode(
            tag=tag,
            second=None,
            value_type=ft & 0xFFFFFF,
            inline=bool(ft & 0x40000000),
            extended=bool(ft & 0x10000000),
            value=None,
            file_offset=p,
        )
        q = p + 16
        if ft & 0x08000000:
            _check_bounds(buf, q, 4, "second tag")
            node.second = struct.unpack_from("<i", buf, q)[0]
            q += 4
        if node.extended:
            if node.value_type in (0, 1, 2):
                children, r2 = _parse_volume(buf, q, stats, seen_offsets)
                node.children = children
                end = max(end, r2)
            else:
                # VS200 mask/annotation blobs (flag 0x28, tags 2023/2024) declare a
                # payload size past EOF; skip without reading (format.md section 2).
                flag_byte = (ft >> 24) & 0xFF
                avail = len(buf) - q
                skip_blob = flag_byte == 0x28 and tag in (2023, 2024)
                if skip_blob:
                    stats.setdefault("skipped_ext_blobs", 0)
                    stats["skipped_ext_blobs"] += 1
                    end = max(end, q)
                elif size > avail:
                    raise TruncatedFileError(
                        f"extended field tag {tag} at offset {q}: "
                        f"need {size} bytes, only {avail} available"
                    )
                else:
                    _check_bounds(buf, q, size, f"extended field tag {tag}")
                    end = max(end, q + size)
        elif node.inline:
            if node.value_type in INLINE_VALUE_TYPES:
                node.value = _decode_inline(node.value_type, size)
            else:
                node.value = UninterpretedInline(value_type=node.value_type, raw_bits=size)
            end = max(end, q)
        else:
            flag_byte = (ft >> 24) & 0xFF
            avail = len(buf) - q
            skip_blob = flag_byte == 0x28 and tag in (2023, 2024)
            if skip_blob:
                stats.setdefault("skipped_ext_blobs", 0)
                stats["skipped_ext_blobs"] += 1
                node.value = None
                end = max(end, q)
            elif size > avail:
                raise TruncatedFileError(
                    f"field tag {tag} at offset {q}: need {size} bytes, only {avail} available"
                )
            else:
                _check_bounds(buf, q, size, f"field tag {tag} payload")
                node.value = _decode_value(node.value_type, buf[q : q + size]) if size else None
                end = max(end, q + size)
        out.append(node)
        if nxt == 0:
            if field_i + 1 < count:
                raise InvalidStructureError(
                    f"volume at {pos}: field chain ended after {field_i + 1} fields, "
                    f"but header declares {count}"
                )
            break
        if nxt < 0:
            raise InvalidStructureError(f"field tag {tag}: negative nextField")
        next_p = pos + (nxt & 0xFFFFFFFF)
        if next_p <= p or next_p in vol_seen:
            raise InvalidStructureError(
                f"volume at {pos}: field tag {tag} nextField {nxt} points to offset "
                f"{next_p}, which is not after the current field at {p}"
            )
        p = next_p
        _check_bounds(buf, p, 16, "next field")
    else:
        if last_nxt != 0:
            raise InvalidStructureError(
                f"volume at {pos}: read {count} fields but the last nextField is "
                f"{last_nxt}, expected 0"
            )
    if len(out) != count:
        raise InvalidStructureError(
            f"volume at {pos}: read {len(out)} fields, header declares {count}"
        )
    return out, end


def load_tree(path: str | bytes) -> VsiTree:
    """Read a .vsi file and parse the metadata tree starting at byte 8."""
    if isinstance(path, bytes):
        data = path
        path_str = "<memory>"
    else:
        path_str = resolve_path(path)
        with open(path_str, "rb") as f:
            data = f.read()
    if len(data) < 8 or data[:4] != b"II*\x00":
        raise InvalidStructureError("not a little-endian TIFF/VSI header", path=path_str)
    stats: dict[str, Any] = {
        "vol_versions": set(),
        "tree_end": 0,
        "size": len(data),
    }
    seen_offsets: set[int] = set()
    root, end = _parse_volume(data, 8, stats, seen_offsets)
    stats["tree_end"] = end
    return VsiTree(path=path_str, data=data, root=root, stats=stats)


def children(node: FieldNode) -> list[FieldNode]:
    return node.children or []


def child(node: FieldNode, tag: int, second: int | None = None) -> FieldNode | None:
    for c in children(node):
        if c.tag == tag and (second is None or c.second == second):
            return c
    return None


def kids(node: FieldNode, tag: int) -> list[FieldNode]:
    return [c for c in children(node) if c.tag == tag]


def walk(nodes: Sequence[FieldNode]) -> Iterator[FieldNode]:
    for n in nodes:
        yield n
        if n.children:
            yield from walk(n.children)


def _unique_root(tree: VsiTree, tag: int) -> FieldNode | None:
    found = [n for n in tree.root if n.tag == tag]
    if len(found) > 1:
        raise AmbiguousMetadataError(
            f"{len(found)} root fields with tag {tag}, expected at most one", path=tree.path
        )
    return found[0] if found else None


def collection_root(tree: VsiTree) -> FieldNode:
    node = _unique_root(tree, 2000)
    if node is None:
        raise InvalidStructureError("no image collection (tag 2000)", path=tree.path)
    return node


def document_root(tree: VsiTree) -> FieldNode | None:
    return _unique_root(tree, 2)
