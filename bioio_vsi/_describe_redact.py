"""Redacted describe output built only from typed renderers (whitelist)."""

from __future__ import annotations

import glob
import os
import re
from enum import Enum
from typing import Any

import numpy as np

from bioio_vsi._codecs import SUPPORTED_COMPRESSION
from bioio_vsi._ets import PIXEL_TYPES, inspect_ets_header
from bioio_vsi._tree import child, collection_root, kids, load_tree
from bioio_vsi._vsi import (
    DIM_KIND,
    _external_frame,
    _find_ets_path,
    document_properties,
    format_version_tuple,
)
from bioio_vsi.errors import MultipleEtsCandidatesError, VsiError

# Products named in docs/format.md section 7 (tag 34 + space + tag 35).
KNOWN_PRODUCTS: frozenset[str] = frozenset(
    {
        "OLYMPUS cellSens Dimension 3.1",
        "OLYMPUS VS200 ASW 3.4.1",
        "OLYMPUS VS200 ASW 4.1",
        "OLYMPUS cellSens Dimension 4.3",
        "cellSens Dimension 3.1",
        "VS200 ASW 3.4.1",
        "VS200 ASW 4.1",
        "cellSens Dimension 4.3",
    }
)

_PRODUCT_VERSION_RE = re.compile(r"^\d+(\.\d+){0,3}$")


class ImageTypeCode(Enum):
    MAIN_SCAN = 0
    OVERVIEW = 1
    FOCUS_MAP = 4
    EFI_STACK = 64
    LABEL = 256
    FOCUS_POINTS = 1024
    SAMPLE_MASK = 8388608


_IMAGE_TYPE_BY_VALUE = {m.value: m for m in ImageTypeCode}


class DimensionKindName(Enum):
    Z = "Z"
    T = "T"
    C = "C"
    EMPTY_SLOT = "empty"


class CompressionName(Enum):
    RAW = "raw"
    JPEG = "JPEG"
    JPEG_2000 = "JPEG 2000"
    UNSUPPORTED = "unsupported"


class PixelTypeName(Enum):
    UINT8 = "uint8"
    UINT16 = "uint16"


class PixelFileState(Enum):
    ABSENT = "absent"
    PRESENT = "present"
    AMBIGUOUS = "ambiguous"
    MALFORMED = "malformed"


def _invalid(value: Any, tag: int) -> str:
    return f"<invalid: {type(value).__name__} for tag {tag}>"


def render_int(value: Any, tag: int) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        return _invalid(value, tag)
    return str(value)


def render_int_list(value: Any, tag: int) -> str:
    if not isinstance(value, (list, tuple)):
        return _invalid(value, tag)
    parts: list[str] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return _invalid(value, tag)
        parts.append(str(item))
    return "[" + ", ".join(parts) + "]"


def render_product_line(product: Any, version: Any) -> str:
    if not isinstance(product, str) or not isinstance(version, str):
        return "<unrecognised product>"
    combined = f"{product} {version}".strip()
    if combined not in KNOWN_PRODUCTS:
        return "<unrecognised product>"
    if not _PRODUCT_VERSION_RE.match(version):
        return "<unrecognised product>"
    return combined


def render_build(value: Any, tag: int = 21) -> str:
    return render_int(value, tag)


def render_format_version(raw: Any) -> str | None:
    fv = format_version_tuple(raw if isinstance(raw, (list, tuple)) else None)
    if fv is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) < 1:
        return None
    major, minor = fv
    build_val = raw[1] if len(raw) > 1 else None
    build_s = render_int(build_val, 2035) if build_val is not None else "?"
    if build_s.startswith("<invalid"):
        build_s = "?"
    return f"Format: {major}.{minor} (build {build_s})"


def render_image_type(value: Any, tag: int = 2074) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return f"  type code: {_invalid(value, tag)}"
    member = _IMAGE_TYPE_BY_VALUE.get(value)
    if member is None:
        return f"  type code: {_invalid(value, tag)}"
    return f"  type code: {member.name}"


def render_dimension_kind(code: Any, tag: int = 2023) -> str:
    if code is None:
        return DimensionKindName.EMPTY_SLOT.value
    if isinstance(code, bool) or not isinstance(code, int):
        return _invalid(code, tag)
    name = DIM_KIND.get(code)
    if name is None:
        return _invalid(code, tag)
    return name


def render_compression(code: Any) -> CompressionName:
    if isinstance(code, bool) or not isinstance(code, int):
        return CompressionName.UNSUPPORTED
    if code == 0:
        return CompressionName.RAW
    if code == 2:
        return CompressionName.JPEG
    if code == 3:
        return CompressionName.JPEG_2000
    return CompressionName.UNSUPPORTED


def render_pixel_type(code: Any) -> PixelTypeName | None:
    if isinstance(code, bool) or not isinstance(code, int):
        return None
    if code not in PIXEL_TYPES:
        return None
    dt = PIXEL_TYPES[code]
    if dt == np.dtype(np.uint8):
        return PixelTypeName.UINT8
    if dt == np.dtype(np.uint16):
        return PixelTypeName.UINT16
    return None


def _count_frame_ets_candidates(vsi_path: str, stack_id: int) -> int:
    base = os.path.join(
        os.path.dirname(vsi_path),
        "_" + os.path.splitext(os.path.basename(vsi_path))[0] + "_",
        f"stack{stack_id}",
    )
    return len(glob.glob(glob.escape(base) + "/frame_t*.ets"))


def _format_redacted_vsi_error(exc: VsiError, *, safe_detail: str | None = None) -> str:
    name = type(exc).__name__
    if safe_detail is not None:
        return f"Error: {name}: {safe_detail}"
    return f"Error: {name}"


def _pixel_file_section(vsi_path: str, stack_id: int) -> list[str]:
    try:
        ets_path = _find_ets_path(vsi_path, stack_id)
    except MultipleEtsCandidatesError:
        n = _count_frame_ets_candidates(vsi_path, stack_id)
        return [f"  pixel file: {PixelFileState.AMBIGUOUS.name} ({n} candidates)"]

    if ets_path is None or not os.path.isfile(ets_path):
        return [f"  pixel file: {PixelFileState.ABSENT.name}"]

    try:
        eh = inspect_ets_header(ets_path)
    except VsiError as exc:
        return [
            f"  pixel file: {PixelFileState.MALFORMED.name} "
            f"({type(exc).__name__})"
        ]
    except OSError as exc:
        return [
            f"  pixel file: {PixelFileState.MALFORMED.name} "
            f"({type(exc).__name__})"
        ]

    ptype = render_pixel_type(eh.pixel_type)
    comp = render_compression(eh.compression)
    ptype_s = _invalid(eh.pixel_type, 0) if ptype is None else ptype.value
    decode_note = ""
    if eh.compression not in SUPPORTED_COMPRESSION:
        decode_note = " (reader does not decode this compression)"
    tw = render_int(eh.tile_width, 0)
    th = render_int(eh.tile_height, 0)
    nt = render_int(eh.ntiles, 0)
    nd = render_int(eh.ndim, 0)
    lines = [
        f"  pixel file: {PixelFileState.PRESENT.name}, "
        f"{ptype_s}, {comp.value}{decode_note}, "
        f"{tw}x{th} tiles, {nt} records, {nd}D coords; index not validated"
    ]
    if eh.origin_x or eh.origin_y:
        ox = render_int(eh.origin_x, 0)
        oy = render_int(eh.origin_y, 0)
        lines.append(f"  ETS origin offset: ({ox}, {oy})")
    return lines


def _val(node: Any, tag: int) -> Any:
    if node is None:
        return None
    c = child(node, tag)
    return None if c is None else c.value


def describe_redacted(path: str | os.PathLike[str]) -> tuple[str, bool]:
    """Return (report text, has_error_line). Never raises."""
    path_str = str(path)
    lines: list[str] = ["File: (redacted)"]
    has_error = False

    try:
        tree = load_tree(path_str)
    except VsiError as exc:
        lines.append(_format_redacted_vsi_error(exc))
        return "\n".join(lines) + "\n", True
    except OSError as exc:
        lines.append(f"Error: <unexpected {type(exc).__name__}>")
        return "\n".join(lines) + "\n", True
    except Exception as exc:
        lines.append(f"Error: <unexpected {type(exc).__name__}>")
        return "\n".join(lines) + "\n", True

    try:
        doc = document_properties(tree)
        if doc:
            product = render_product_line(doc.get("product"), doc.get("product_version"))
            build = render_build(doc.get("build"))
            lines.append(f"Document: {product} (build {build})")

        coll = collection_root(tree)
        raw_2035 = _val(coll, 2035)
        fmt_line = render_format_version(raw_2035)
        if fmt_line:
            lines.append(fmt_line)

        for vol in kids(coll, 2001):
            sid = vol.second if vol.second is not None else 0
            sid_s = render_int(sid, 2001)
            lines.append("")
            lines.append(f"Image stack {sid_s}:")
            sp = child(vol, 2005)
            itype_line = render_image_type(_val(sp, 2074) if sp else None)
            if itype_line:
                lines.append(itype_line)

            fr = _external_frame(vol, vsi_path=path_str, stack_id=int(sid))
            w = h = None
            if fr is not None:
                ext = child(fr, 2018)
                rect = _val(ext, 2053) if ext else None
                if (
                    rect
                    and isinstance(rect, (list, tuple))
                    and len(rect) >= 4
                    and all(isinstance(x, int) and not isinstance(x, bool) for x in rect[2:4])
                ):
                    w, h = int(rect[2]), int(rect[3])
            sizes = _val(vol, 2003)
            slot_kinds: list[str] = []
            for dd in kids(vol, 2007):
                ps = kids(dd, 2008)
                if ps:
                    kc = _val(ps[0], 2023)
                    slot_kinds.append(render_dimension_kind(kc))
                else:
                    slot_kinds.append(DimensionKindName.EMPTY_SLOT.value)
            if w is not None and h is not None:
                lines.append(f"  size: {w} x {h}")
            if sizes is not None:
                lines.append(f"  dimension sizes (2003): {render_int_list(sizes, 2003)}")
            if slot_kinds:
                rendered = ", ".join(slot_kinds)
                lines.append(f"  slot order: [{rendered}]")

            lines.extend(_pixel_file_section(path_str, int(sid) if isinstance(sid, int) else 0))
    except VsiError as exc:
        lines.append(_format_redacted_vsi_error(exc))
        has_error = True
    except OSError as exc:
        lines.append(f"Error: <unexpected {type(exc).__name__}>")
        has_error = True
    except Exception as exc:
        lines.append(f"Error: <unexpected {type(exc).__name__}>")
        has_error = True

    return "\n".join(lines) + "\n", has_error
