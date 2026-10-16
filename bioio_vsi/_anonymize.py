"""Build anonymised structure fixtures from a .vsi acquisition.

Kept: the TIFF IFD structure (not the thumbnail pixels), volume and field
headers, and in `.skel` sidecars the `.ets` main header, the documented
second-header fields and the tile table. Strings become same-length
placeholders (the tree's offsets depend on lengths); absolute times become one
fixed sentinel per encoding. Unit strings (`10^-6m^1` and the like, on tag 2020
and unit tag 268435456) are format vocabulary and are copied verbatim.

Removed: other text, the TIFF thumbnail, the 96 x 96 specimen preview in the
tree (tag 38; its 40-byte bitmap header is kept), tile pixels, mask blobs,
every other array of one-byte elements longer than 8 bytes (XML under tag
2038, unknown blobs) and any payload type the reader does not interpret.
Numbers are then replaced by `_synthetic_numbers` unless kept on request.
"""

from __future__ import annotations

import os
import re
import struct
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from bioio_vsi._accessors import UNIT_MICROMETRES, UNIT_TAG
from bioio_vsi._anonymize_dates import (
    _sanitize_numeric_raw,
    sanitize_inline_bits,
    sanitize_scalar_float,
)
from bioio_vsi._anonymize_tiff import sanitize_vsi_thumbnail_tiff
from bioio_vsi._ets import _read_main_header
from bioio_vsi._paths import resolve_path
from bioio_vsi._synthetic_numbers import synthesize_numbers
from bioio_vsi._tree import (
    FLD,
    INLINE_VALUE_TYPES,
    SCALAR,
    VEC,
    FieldNode,
    VsiTree,
    collection_root,
    kids,
    load_tree,
)
from bioio_vsi._vsi import _find_ets_path
from bioio_vsi.errors import VsiError

SKEL_MAGIC = b"BVSK"
SKEL_VERSION = 1

STRING_TYPES = frozenset({13, 8192, 8194})
PLACEHOLDER_UTF16 = "X".encode("utf-16-le")

# Documented second-header byte ranges (format.md); bytes outside are zeroed in .skel.
_SKEL_SECOND_KEEP = (
    (0, 40),
    (108, 112),
    (148, 156),
    (172, 180),
    (184, None),  # extents: filled from parsed n_extents
    (216, 228),
)


@dataclass
class AnonymizeReport:
    date_fields: list[str] = field(default_factory=list)
    format_constant_strings: list[str] = field(default_factory=list)
    byte_counts: dict[str, int] = field(default_factory=dict)


def _mark_factory(counts: dict[str, int]) -> Callable[[int, int, str], None]:
    def mark(_start: int, length: int, cls: str) -> None:
        counts[cls] = counts.get(cls, 0) + length

    return mark


# Arrays of one-byte elements longer than this are content, not numbers.
_MAX_KEPT_BYTE_ARRAY = 8
_BITMAP_HEADER_LEN = 40


def _is_byte_array(value_type: int, size: int) -> bool:
    fmt = SCALAR.get(value_type) or VEC.get(value_type)
    return fmt is not None and struct.calcsize(fmt) == 1 and size > _MAX_KEPT_BYTE_ARRAY


def _field_header_len(node: FieldNode, data: bytes) -> int:
    off = node.file_offset
    ft, _tag, _nxt, _size = FLD.unpack_from(data, off)
    return 16 + (4 if ft & 0x08000000 else 0)


def _payload_offset(node: FieldNode, data: bytes) -> int:
    return node.file_offset + _field_header_len(node, data)


# Unit strings are a fixed vocabulary of the format ("10^-6m^1", "10^0%^1").
_UNIT_RE = re.compile(r"10\^-?\d+[A-Za-z%]*\^-?\d+")


def _keep_format_string(tag: int, decoded: str, _path: str) -> bool:
    if tag == 2020 and decoded == UNIT_MICROMETRES:
        return True
    return tag == UNIT_TAG and _UNIT_RE.fullmatch(decoded) is not None


def _utf16_placeholder(raw: bytes) -> bytes:
    if not raw:
        return raw
    out = bytearray(len(raw))
    # Preserve UTF-16 NUL terminator if present.
    if len(raw) >= 2 and raw[-2:] == b"\x00\x00":
        n_chars = (len(raw) - 2) // 2
        for i in range(n_chars):
            out[2 * i : 2 * i + 2] = PLACEHOLDER_UTF16
        out[-2:] = b"\x00\x00"
    else:
        n_chars = len(raw) // 2
        for i in range(n_chars):
            out[2 * i : 2 * i + 2] = PLACEHOLDER_UTF16
    return bytes(out)


def _placeholder_raw(value_type: int, raw: bytes) -> bytes:
    if value_type in STRING_TYPES:
        return _utf16_placeholder(raw)
    return bytes(len(raw))


def _copy_span(
    src: bytes,
    out: bytearray,
    start: int,
    length: int,
    mark: Callable[[int, int, str], None],
    cls: str,
) -> None:
    if length <= 0:
        return
    out[start : start + length] = src[start : start + length]
    mark(start, length, cls)


def _sanitize_field_payload(
    node: FieldNode,
    src: bytes,
    out: bytearray,
    mark: Callable[[int, int, str], None],
    path: str,
    report: AnonymizeReport,
) -> None:
    off = node.file_offset
    ft, tag, _nxt, size = FLD.unpack_from(src, off)
    value_type = ft & 0xFFFFFF
    flag_byte = (ft >> 24) & 0xFF
    q = _payload_offset(node, src)

    if node.extended:
        if value_type in (0, 1, 2):
            return
        if value_type == 10:
            return
        skip_blob = flag_byte == 0x28 and tag in (2023, 2024)
        if skip_blob:
            return
        return

    if node.inline:
        if value_type in INLINE_VALUE_TYPES:
            new_size, hit = sanitize_inline_bits(value_type, size)
            if hit:
                struct.pack_into("<I", out, off + 12, new_size)
                mark(off + 12, 4, "date")
                report.date_fields.append(f"{path}/{tag} (inline)")
            else:
                mark(off + 12, 4, "numeric")
        return

    skip_blob = flag_byte == 0x28 and tag in (2023, 2024)
    if skip_blob:
        return

    if size <= 0:
        return

    raw = src[q : q + size]
    if value_type in STRING_TYPES:
        decoded = raw.decode("utf-16-le", "replace").split("\0")[0]
        if _keep_format_string(tag, decoded, path):
            _copy_span(src, out, q, size, mark, "numeric")
            report.format_constant_strings.append(f"{path}/{tag}={decoded!r}")
            return
        ph = _placeholder_raw(value_type, raw)
        out[q : q + size] = ph
        mark(q, size, "placeholder")
        return

    if _is_byte_array(value_type, size):
        # Byte arrays carry content, not numbers: the 96x96 specimen preview
        # (a BITMAPINFOHEADER and BGR pixels), XML (tag 2038) and unknown
        # blobs. Keep only a bitmap header so the structure stays valid.
        if size > _BITMAP_HEADER_LEN and struct.unpack_from("<I", raw)[0] == _BITMAP_HEADER_LEN:
            _copy_span(src, out, q, _BITMAP_HEADER_LEN, mark, "structure")
        return

    if value_type in SCALAR or value_type in VEC:
        new_raw, hit = _sanitize_numeric_raw(value_type, raw)
        out[q : q + size] = new_raw
        mark(q, size, "date" if hit else "numeric")
        if hit:
            report.date_fields.append(f"{path}/{tag}")
        return

    # Nested prop doubles (type 10 inside 2014/2017 paths) are handled when value_type==10 scalar
    if value_type == 10 and size == 8:
        val = struct.unpack("<d", raw)[0]
        nv, hit = sanitize_scalar_float(val)
        if hit:
            out[q : q + 8] = struct.pack("<d", nv)
            mark(q, 8, "date")
            report.date_fields.append(f"{path}/{tag}")
        else:
            _copy_span(src, out, q, size, mark, "numeric")
        return


def _volume_header_span(pos: int) -> int:
    return 24


def _field_path(node: FieldNode, parents: list[FieldNode]) -> str:
    parts = [str(p.tag) + (f"[{p.second}]" if p.second is not None else "") for p in parents]
    parts.append(str(node.tag))
    return "/".join(parts)


def anonymize_vsi_bytes(tree: VsiTree) -> tuple[bytes, AnonymizeReport]:
    src = tree.data
    out = bytearray(len(src))
    counts: dict[str, int] = defaultdict(int)
    mark = _mark_factory(counts)
    report = AnonymizeReport()

    sanitize_vsi_thumbnail_tiff(src, out, mark)
    _copy_span(src, out, 8, 24, mark, "structure")

    def visit(node: FieldNode, parents: list[FieldNode]) -> None:
        off = node.file_offset
        hdrlen = _field_header_len(node, src)
        _copy_span(src, out, off, hdrlen, mark, "structure")
        if node.extended and node.value_type in (0, 1, 2) and node.children is not None:
            q = _payload_offset(node, src)
            _copy_span(src, out, q, 24, mark, "structure")
        path = _field_path(node, parents)
        _sanitize_field_payload(node, src, out, mark, path, report)
        if node.children:
            for child in node.children:
                visit(child, parents + [node])

    for root_node in tree.root:
        visit(root_node, [])

    written = sum(counts.values())
    counts["zero"] = max(0, len(src) - written)
    report.byte_counts = dict(counts)
    return bytes(out), report


def _second_header_keep_mask(second: bytes) -> bytearray:
    mask = bytearray(len(second))
    n_ext = struct.unpack_from("<I", second, 184)[0] if len(second) >= 188 else 0
    extent_end = 188 + 4 * n_ext
    for start, end in _SKEL_SECOND_KEEP:
        if start is None:
            continue
        if end is None:
            end = min(extent_end, len(second))
        for i in range(start, min(end, len(second))):
            mask[i] = 1
    if extent_end > 188:
        for i in range(188, min(extent_end, len(second))):
            mask[i] = 1
    return mask


def _ets_skeleton_bytes(ets_path: str) -> tuple[bytes, int]:
    with open(ets_path, "rb") as f:
        data = f.read()
    fsize = len(data)
    main = _read_main_header(ets_path, fsize, data)
    addpos = int(main["addpos"])
    addsize = int(main["addsize"])
    tabpos = int(main["tabpos"])
    ntiles = int(main["ntiles"])
    ndim = int(main["ndim"])
    rec_size = 4 * (ndim + 5)
    table_len = ntiles * rec_size
    table_end = tabpos + table_len

    second_raw = data[addpos : addpos + addsize]
    mask = _second_header_keep_mask(second_raw)
    second = bytearray(second_raw)
    for i, keep in enumerate(mask):
        if not keep:
            second[i] = 0

    ranges: list[tuple[int, bytes]] = [
        (0, data[:64]),
        (addpos, bytes(second)),
        (tabpos, data[tabpos:table_end]),
    ]

    body = bytearray()
    for offset, chunk in ranges:
        body.extend(struct.pack("<QI", offset, len(chunk)))
        body.extend(chunk)

    header = SKEL_MAGIC + struct.pack("<IQI", SKEL_VERSION, fsize, len(ranges))
    return header + bytes(body), fsize


def _parse_skel(blob: bytes) -> tuple[int, list[tuple[int, bytes]]]:
    if len(blob) < 20 or blob[:4] != SKEL_MAGIC:
        raise VsiError("not a BVSK skeleton file")
    version, nominal, n_ranges = struct.unpack_from("<IQI", blob, 4)
    if version != SKEL_VERSION:
        raise VsiError(f"unsupported skeleton version {version}")
    pos = 20
    ranges: list[tuple[int, bytes]] = []
    for _ in range(n_ranges):
        if pos + 12 > len(blob):
            raise VsiError("truncated skeleton range header")
        offset, length = struct.unpack_from("<QI", blob, pos)
        pos += 12
        ranges.append((offset, blob[pos : pos + length]))
        pos += length
    return nominal, ranges


def read_skel(path: str | os.PathLike[str]) -> tuple[int, list[tuple[int, bytes]]]:
    with open(path, "rb") as f:
        return _parse_skel(f.read())


def anonymize(
    vsi_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    synthetic_numbers: bool = True,
) -> Path:
    """Write anon.vsi and .skel pixel skeletons under out_dir.

    With synthetic_numbers (the default) measured values are replaced too; see
    _synthetic_numbers. Without it, numbers are kept as stored.
    """
    vsi_path = resolve_path(str(vsi_path))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tree = load_tree(vsi_path)
    anon_bytes, _report = anonymize_vsi_bytes(tree)
    if synthetic_numbers:
        anon_bytes = synthesize_numbers(anon_bytes)
    anon_vsi = out / "anon.vsi"
    anon_vsi.write_bytes(anon_bytes)

    stack_dir = out / "_anon_"
    coll = collection_root(tree)
    for vol in kids(coll, 2001):
        sid = vol.second if vol.second is not None else 0
        ets = _find_ets_path(vsi_path, int(sid))
        if ets is None or not os.path.isfile(ets):
            continue
        frame_name = os.path.basename(ets)
        skel_bytes, _ = _ets_skeleton_bytes(ets)
        dest = stack_dir / f"stack{sid}" / f"{frame_name}.skel"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(skel_bytes)

    return anon_vsi


__all__ = [
    "AnonymizeReport",
    "anonymize",
    "anonymize_vsi_bytes",
    "read_skel",
]
