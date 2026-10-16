"""Pixel-free summary of a VSI file (works even when open_vsi would refuse)."""

from __future__ import annotations

import glob
import os
from collections.abc import Callable
from pathlib import Path

from bioio_vsi._accessors import (
    kids_unique_by_second,
    node_value,
    read_boundary_rect,
    read_dimension_kind_code,
    read_dimension_sizes,
    read_format_version_raw,
    read_optional_int,
    read_optional_string,
    unique_child,
)
from bioio_vsi._codecs import SUPPORTED_COMPRESSION
from bioio_vsi._ets import PIXEL_TYPES, inspect_ets_header
from bioio_vsi._paths import resolve_path
from bioio_vsi._tree import FieldNode, collection_root, load_tree
from bioio_vsi._vsi import (
    DIM_KIND,
    _external_frame,
    _find_ets_path,
    document_properties,
    format_version_tuple,
    list_image_volumes,
)
from bioio_vsi.errors import MultipleEtsCandidatesError, VsiError

COMPRESSION_NAMES = {0: "raw", 2: "JPEG", 3: "JPEG 2000"}


def _format_vsi_error(exc: VsiError, *, redact: bool) -> str:
    return exc.redacted_str() if redact else str(exc)


def _append_or_unreadable(
    lines: list[str],
    label: str,
    fn: Callable[[], str | None],
) -> None:
    try:
        text = fn()
    except VsiError as exc:
        lines.append(f"  {label}: unreadable ({exc})")
        return
    if text is not None:
        lines.append(text)


def _pixel_file_line(
    path_str: str,
    stack_id: int,
    *,
    redact: bool,
) -> str:
    try:
        ets_path = _find_ets_path(path_str, stack_id)
    except MultipleEtsCandidatesError as exc:
        if redact:
            return (
                "  pixel file: ambiguous (several frame_t*.ets candidates; "
                "inspect locally without --redact)"
            )
        return f"  pixel file: ambiguous ({exc})"

    if ets_path is None or not os.path.isfile(ets_path):
        return "  pixel file: absent"

    try:
        eh = inspect_ets_header(ets_path)
    except VsiError as exc:
        label = os.path.basename(ets_path) if not redact else "<pixel file>"
        detail = _format_vsi_error(exc, redact=redact)
        return f"  pixel file: malformed ({label}: {detail})"

    ptype = PIXEL_TYPES.get(eh.pixel_type, eh.pixel_type)
    comp = COMPRESSION_NAMES.get(eh.compression, str(eh.compression))
    decode_note = ""
    if eh.compression not in SUPPORTED_COMPRESSION:
        decode_note = " (reader does not decode this compression)"
    name_part = ""
    if not redact:
        name_part = f" ({os.path.basename(ets_path)}),"
    lines = [
        f"  pixel file: present{name_part} "
        f"{ptype}, {comp}{decode_note}, {eh.tile_width}x{eh.tile_height} tiles, "
        f"{eh.ntiles} records, {eh.ndim}D coords; index not validated"
    ]
    if eh.origin_x or eh.origin_y:
        lines.append(f"  ETS origin offset: ({eh.origin_x}, {eh.origin_y})")
    return "\n".join(lines)


def describe_vsi(path: str | Path, *, redact: bool = False) -> str:
    if redact:
        from bioio_vsi._describe_redact import describe_redacted

        text, _has_error = describe_redacted(path)
        return text

    path_str = resolve_path(path)
    lines: list[str] = [f"File: {path_str}"]
    try:
        tree = load_tree(path_str)
    except VsiError as exc:
        lines.append(f"Error: {_format_vsi_error(exc, redact=False)}")
        return "\n".join(lines) + "\n"
    except OSError as exc:
        lines.append(f"Error: {exc}")
        return "\n".join(lines) + "\n"
    except Exception as exc:
        lines.append(f"Error: {exc}")
        return "\n".join(lines) + "\n"

    try:
        doc = document_properties(tree)
    except VsiError as exc:
        lines.append(f"Document: unreadable ({exc})")
        doc = {}
    if doc:
        doc_path = f"{path_str}/2"
        build_s = "?"
        try:
            from bioio_vsi._tree import document_root

            doc_node = document_root(tree)
            if doc_node is not None:
                build = read_optional_int(
                    node_value(doc_node, path=doc_path, tag=21),
                    path=f"{doc_path}/21",
                    field="build",
                )
                if build is not None:
                    build_s = str(build)
        except VsiError as exc:
            build_s = f"unreadable ({exc})"
        lines.append(
            f"Document: {doc.get('product')} {doc.get('product_version')} "
            f"(build {build_s})"
        )
    coll = collection_root(tree)
    coll_path = f"{path_str}/2000"

    try:
        raw_2035 = node_value(coll, path=coll_path, tag=2035)
        if raw_2035 is not None:
            parsed = read_format_version_raw(raw_2035, path=f"{coll_path}/2035")
            fv = format_version_tuple(parsed)
            if fv is not None:
                build = parsed[1]
                lines.append(f"Format: {fv[0]}.{fv[1]} (build {build})")
    except VsiError as exc:
        lines.append(f"Format: unreadable ({exc})")

    for vol in list_image_volumes(tree):
        sid = vol.second if vol.second is not None else 0
        vol_path = f"{path_str}/2001[{sid}]"
        lines.append("")
        lines.append(f"Image stack{sid}:")
        sp = unique_child(vol, 2005, path=vol_path)
        sp_path = f"{vol_path}/2005"

        def _name_line(
            sp_node: FieldNode | None = sp,
            stack_path: str = sp_path,
        ) -> str | None:
            if sp_node is None:
                return None
            name = read_optional_string(
                node_value(sp_node, path=stack_path, tag=2030),
                path=f"{stack_path}/2030",
                field="2030",
            )
            return f"  name: {name}" if name else None

        _append_or_unreadable(lines, "name", _name_line)

        def _itype_line(
            sp_node: FieldNode | None = sp,
            stack_path: str = sp_path,
        ) -> str | None:
            if sp_node is None:
                return None
            itype = read_optional_int(
                node_value(sp_node, path=stack_path, tag=2074),
                path=f"{stack_path}/2074",
                field="2074",
            )
            return f"  type code: {itype}" if itype is not None else None

        _append_or_unreadable(lines, "type code", _itype_line)

        def _size_line(
            volume: FieldNode = vol,
            image_path: str = vol_path,
            stack_id: int = sid,
            file_path: str = path_str,
        ) -> str | None:
            fr = _external_frame(volume, vsi_path=file_path, stack_id=stack_id)
            if fr is None:
                return None
            frame_path = f"{image_path}/2002[{fr.second}]"
            ext = unique_child(fr, 2018, path=frame_path)
            if ext is None:
                return None
            ext_path = f"{frame_path}/2018"
            rect = node_value(ext, path=ext_path, tag=2053)
            if rect is None:
                return None
            _x0, _y0, w, h = read_boundary_rect(
                rect,
                path=f"{ext_path}/2053",
            )
            return f"  size: {w} x {h}"

        _append_or_unreadable(lines, "size", _size_line)

        def _sizes_line(
            volume: FieldNode = vol,
            image_path: str = vol_path,
        ) -> str | None:
            raw = node_value(volume, path=image_path, tag=2003)
            if raw is None:
                return None
            sizes = read_dimension_sizes(raw, path=f"{image_path}/2003")
            return f"  dimension sizes (2003): {sizes}"

        _append_or_unreadable(lines, "dimension sizes (2003)", _sizes_line)

        def _slots_line(
            volume: FieldNode = vol,
            image_path: str = vol_path,
        ) -> str | None:
            slot_kinds: list[str] = []
            for ord_i, dd in enumerate(
                kids_unique_by_second(
                    volume,
                    2007,
                    path=image_path,
                    member="dimension slot",
                )
            ):
                slot_path = f"{image_path}/2007[{ord_i}]"
                ps = kids_unique_by_second(dd, 2008, path=slot_path, member="channel")
                if ps:
                    kind_node = unique_child(
                        ps[0],
                        2023,
                        path=f"{slot_path}/2008[0]",
                    )
                    if kind_node is None:
                        slot_kinds.append("?")
                    else:
                        kc = read_dimension_kind_code(
                            kind_node.value,
                            path=f"{slot_path}/2008[0]/2023",
                            field="2023",
                        )
                        slot_kinds.append(DIM_KIND.get(kc, str(kc)))
                else:
                    slot_kinds.append("(empty)")
            if not slot_kinds:
                return None
            return f"  slot order: {slot_kinds}"

        _append_or_unreadable(lines, "slot order", _slots_line)

        lines.append(_pixel_file_line(path_str, sid, redact=False))

    base = os.path.join(
        os.path.dirname(path_str),
        "_" + os.path.splitext(os.path.basename(path_str))[0] + "_",
    )
    if os.path.isdir(base):
        extras = sorted(glob.glob(glob.escape(base) + "/*/*.ets"))
        for e in extras:
            if "frame_t" not in os.path.basename(e):
                rel = os.path.relpath(e, base)
                lines.append(f"Other ETS: {rel}")

    return "\n".join(lines) + "\n"
