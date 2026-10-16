"""Build minimal valid .vsi / .ets pairs for tests.

Synthetic files prove self-consistency of the reader only; they are not oracles
for real cellSens output.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np

from bioio_vsi._ets import _pyramid_level_count

MAIN_HDR = struct.Struct("<4sIIIQIIQIIQII")
VOL = struct.Struct("<HHIqII")
FLD = struct.Struct("<iiiI")

DIM_CODE = {"Z": 1, "T": 2, "C": 4}


def _round_half_up_mean(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    s = a.astype(np.int32) + b.astype(np.int32) + c.astype(np.int32) + d.astype(np.int32)
    out: np.ndarray = ((s + 2) // 4).astype(a.dtype)
    return out


def build_pyramid(level0: np.ndarray, n_levels: int) -> list[np.ndarray]:
    levels = [level0]
    cur = level0
    for _ in range(1, n_levels):
        y, x = cur.shape
        ny, nx = y >> 1, x >> 1
        if ny == 0 or nx == 0:
            break
        nxt = np.zeros((ny, nx), dtype=cur.dtype)
        for j in range(ny):
            for i in range(nx):
                y0, x0 = j * 2, i * 2
                vals = []
                for dy in (0, 1):
                    for dx in (0, 1):
                        yy, xx = y0 + dy, x0 + dx
                        vals.append(cur[yy, xx] if yy < y and xx < x else np.uint16(0))
                quads = tuple(cast(np.ndarray, np.asarray(v, dtype=cur.dtype)) for v in vals)
                nxt[j, i] = _round_half_up_mean(quads[0], quads[1], quads[2], quads[3])
        levels.append(nxt)
        cur = nxt
    return levels


def _build_volume(fields: list[bytes]) -> bytes:
    if not fields:
        return bytes(VOL.pack(24, 21321, 1, 24, 0, 0))
    vol = bytearray(VOL.pack(24, 21321, 1, 24, len(fields), 0))
    pos = 24
    patched: list[bytearray] = []
    for fb in fields:
        patched.append(bytearray(fb))
    for i in range(len(patched) - 1):
        next_rel = pos + len(patched[i])
        struct.pack_into("<i", patched[i], 8, next_rel)
        pos += len(patched[i])
    struct.pack_into("<i", patched[-1], 8, 0)
    out = bytes(vol)
    for p in patched:
        out += bytes(p)
    return out


def _field_data(tag: int, value_type: int, payload: bytes, second: int | None = None) -> bytes:
    ft = value_type
    if second is not None:
        ft |= 0x08000000
    hdr = FLD.pack(ft, tag, 0, len(payload))
    sec = struct.pack("<i", second) if second is not None else b""
    return hdr + sec + payload


def _field_inline(tag: int, value: int, second: int | None = None) -> bytes:
    ft = 0x40000000 | 6
    if second is not None:
        ft |= 0x08000000
    hdr = FLD.pack(ft, tag, 0, value)
    sec = struct.pack("<i", second) if second is not None else b""
    return hdr + sec


def _field_nested(tag: int, child_fields: list[bytes], second: int | None = None) -> bytes:
    child = _build_volume(child_fields)
    ft = 0x10000000  # extended type 0
    if second is not None:
        ft |= 0x08000000
    hdr = FLD.pack(ft, tag, 0, len(child))
    sec = struct.pack("<i", second) if second is not None else b""
    return hdr + sec + child


def _int_vector(tag: int, values: Sequence[int], second: int | None = None) -> bytes:
    payload = struct.pack(f"<{len(values)}i", *values)
    return _field_data(tag, 257, payload, second)


def _double_vector(tag: int, values: Sequence[float]) -> bytes:
    payload = struct.pack(f"<{len(values)}d", *values)
    return _field_data(tag, 260, payload)


def _string(tag: int, text: str, value_type: int = 8192) -> bytes:
    """UTF-16 string; type 13 (unit strings) is UTF-16 as well, unterminated."""
    return _field_data(tag, value_type, text.encode("utf-16-le"))


def _prop_double(tag: int, value: float) -> bytes:
    inner = [_field_data(268435458, 10, struct.pack("<d", value))]
    return _field_nested(tag, inner)


def write_ets(
    path: Path,
    *,
    axis_order: Sequence[str],
    axis_sizes: dict[str, int],
    planes: dict[tuple[int, ...], np.ndarray],
    tile_size: int,
    compression: int = 0,
    pixel_type: int = 4,
    origin: tuple[int, int] = (0, 0),
) -> None:
    sample = planes[next(iter(planes))]
    height, width = sample.shape
    dims = [axis_sizes[a] for a in axis_order]
    ndim = 2 + len(dims) + 1
    n_extents = ndim - 1
    # The planes are the stored grid; the image is smaller by the (negative) origin.
    extent_vals = [width + origin[0], height + origin[1]] + dims

    records: list[bytes] = []
    payload = bytearray()
    th = tw = tile_size
    data_start = 64 + 228

    max_level = max(key[-1] for key in planes)
    use_pyramid = 1 if max_level > 0 else 0

    for key, plane in sorted(planes.items()):
        level = key[-1]
        slot_vals = key[:-1]
        y_size, x_size = plane.shape
        for ty in range((y_size + th - 1) // th):
            for tx in range((x_size + tw - 1) // tw):
                tile = np.zeros((th, tw), dtype=plane.dtype)
                ya, xa = ty * th, tx * tw
                yb, xb = min(ya + th, y_size), min(xa + tw, x_size)
                tile[: yb - ya, : xb - xa] = plane[ya:yb, xa:xb]
                raw = tile.tobytes()
                off = data_start + len(payload)
                payload.extend(raw)
                coord = [tx, ty, *slot_vals, level]
                records.append(
                    struct.pack("<I", ndim)
                    + struct.pack(f"<{ndim}i", *coord)
                    + struct.pack("<QII", off, len(raw), 0)
                )

    second = bytearray(228)
    struct.pack_into(
        "<4s9I",
        second,
        0,
        b"ETS\x00",
        0x30006,
        pixel_type,
        1,
        1,
        compression,
        100,
        tw,
        th,
        1,
    )
    struct.pack_into("<i", second, 172, origin[0])
    struct.pack_into("<i", second, 176, origin[1])
    struct.pack_into("<I", second, 184, n_extents)
    for i, v in enumerate(extent_vals):
        struct.pack_into("<I", second, 188 + 4 * i, int(v))
    struct.pack_into("<I", second, 152, use_pyramid)
    if origin != (0, 0):
        struct.pack_into("<i", second, 216, -origin[0])
        struct.pack_into("<i", second, 220, -origin[1])

    tabpos = 64 + 228 + len(payload)
    main = bytearray(64)
    struct.pack_into(
        "<4sIIIQIIQIIQII",
        main,
        0,
        b"SIS\x00",
        64,
        3,
        ndim,
        64,
        228,
        0,
        tabpos,
        len(records),
        0,
        0,
        0,
        0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(main)
        f.write(second)
        f.write(payload)
        for rec in records:
            f.write(rec)


def write_minimal_vsi(
    vsi_path: Path,
    *,
    stack_id: int,
    width: int,
    height: int,
    axis_order: Sequence[str],
    axis_sizes: dict[str, int],
    channel_names: Sequence[str],
    pixel_size: tuple[float, float] = (1.0, 1.0),
    z_start: float = 0.0,
    z_step: float = 1.0,
    extra_image_ids: Sequence[int] = (),
    kind_codes: dict[str, int | Sequence[int]] | None = None,
    format_2035: tuple[int, int] = (0x30016, 21199),
    plane_times_ms: Sequence[float] | None = None,
    plane_time_frames: dict[int, float] | None = None,
    dimension_seconds: Sequence[int | None] | None = None,
    channel_seconds: Sequence[int | None] | None = None,
    document_children: Sequence[bytes] | None = None,
    collection_2035: bytes | None = None,
    volume_2003: bytes | None = None,
    volume_2053: bytes | None = None,
    volume_2074: bytes | None = None,
    volume_2023: bytes | None = None,
) -> None:
    """Write a minimal single-stack .vsi (external pixels expected under _<stem>_/stackN/)."""
    kinds = kind_codes or {}
    if collection_2035 is not None:
        fmt_field = collection_2035
    else:
        fmt_field = _int_vector(2035, [format_2035[0], format_2035[1]])
    coll_children: list[bytes] = [fmt_field]

    img_children: list[bytes] = [
        volume_2003
        if volume_2003 is not None
        else _int_vector(2003, [axis_sizes[a] for a in axis_order]),
    ]

    boundary = volume_2053 if volume_2053 is not None else _int_vector(2053, [0, 0, width, height])
    ext_kids = [boundary, _field_inline(20005, 1)]
    n_frames = 1
    for ax in axis_order:
        n_frames *= axis_sizes[ax]
    if plane_times_ms is not None and len(plane_times_ms) != n_frames:
        raise ValueError("plane_times_ms length must match T*C*Z")
    for fi in range(n_frames):
        frame_kids: list[bytes] = []
        if fi == 0:
            frame_kids.append(_field_nested(2018, ext_kids))
        if plane_times_ms is not None:
            frame_kids.append(_field_nested(2006, [_prop_double(2017, plane_times_ms[fi])]))
        elif plane_time_frames is not None and fi in plane_time_frames:
            frame_kids.append(_field_nested(2006, [_prop_double(2017, plane_time_frames[fi])]))
        img_children.append(_field_nested(2002, frame_kids, second=fi))

    sp_kids = [_double_vector(2019, list(pixel_size)), _string(2030, "synthetic")]
    if volume_2074 is not None:
        sp_kids.append(volume_2074)
    img_children.append(_field_nested(2005, sp_kids))

    dim_seconds = list(dimension_seconds) if dimension_seconds is not None else None
    ch_seconds = list(channel_seconds) if channel_seconds is not None else None
    for slot_i, ax in enumerate(axis_order):
        dim_second = dim_seconds[slot_i] if dim_seconds is not None else slot_i
        if ax == "C":
            raw_codes = kinds.get(ax, DIM_CODE["C"])
            if isinstance(raw_codes, int):
                ch_codes = [raw_codes] * len(channel_names)
            else:
                ch_codes = list(raw_codes)
            dim_kids = []
            for ci, name in enumerate(channel_names):
                ch_second = ch_seconds[ci] if ch_seconds is not None else ci
                kind_field = (
                    volume_2023
                    if volume_2023 is not None and slot_i == 0 and ci == 0
                    else _field_inline(2023, ch_codes[ci])
                )
                ch = [kind_field, _string(2021, name)]
                dim_kids.append(_field_nested(2008, ch, second=ch_second))
        else:
            raw = kinds.get(ax, DIM_CODE.get(ax, 1))
            code = int(raw) if isinstance(raw, int) else int(raw[0])  # type: ignore[index]
            if volume_2023 is not None and slot_i == 0:
                kind_field = volume_2023
            else:
                kind_field = _field_inline(2023, code)
            p = [kind_field]
            if ax == "Z":
                p.extend([_prop_double(2012, z_start), _prop_double(2013, z_step)])
            dim_kids = [_field_nested(2008, p, second=0)]
        img_children.append(_field_nested(2007, dim_kids, second=dim_second))
    img_children.append(_field_nested(2007, [], second=len(axis_order)))

    coll_children.append(_field_nested(2001, img_children, second=stack_id))
    for eid in extra_image_ids:
        coll_children.append(_field_nested(2001, [_int_vector(2003, [1, 1, 1])], second=eid))

    if document_children is not None:
        doc_kids = list(document_children)
    else:
        doc_kids = [_string(34, "synthetic")]
    root = _build_volume(
        [
            _field_nested(2, doc_kids),
            _field_nested(2000, coll_children, second=0),
        ]
    )
    vsi_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vsi_path, "wb") as f:
        f.write(b"II*\x00" + struct.pack("<I", 0))
        f.write(root)


def make_stack(
    tmp: Path,
    *,
    shape: tuple[int, int, int, int, int],
    axis_order: str = "TZC",
    tile_size: int = 512,
    n_levels: int = 1,
    compression: int = 0,
    origin: tuple[int, int] = (0, 0),
    extra_images: Sequence[int] = (),
    plane_times_ms: Sequence[float] | None = None,
) -> tuple[Path, Path]:
    """Return (vsi_path, ets_path) for a raw uint16 stack. shape is (T,C,Z,Y,X).

    With a negative ``origin`` the image is Y x X and the stored grid is larger by
    ``-origin``.
    """
    t, c, z, y, x = shape
    axis_sizes = {"T": t, "C": c, "Z": z}
    stored_y, stored_x = y - origin[1], x - origin[0]
    tw = tile_size if max(stored_y, stored_x) > tile_size else max(stored_y, stored_x)
    if n_levels > 1:
        n_levels = _pyramid_level_count(stored_y, stored_x, tw, tw)
    planes: dict[tuple[int, ...], np.ndarray] = {}
    for ti in range(t):
        for ci in range(c):
            for zi in range(z):
                base = (
                    np.arange(stored_y * stored_x, dtype=np.uint16).reshape(stored_y, stored_x)
                    + ti * 1000
                    + ci * 100
                    + zi
                )
                for lv, plane in enumerate(build_pyramid(base, n_levels)):
                    key: tuple[int, ...]
                    idx = {a: v for a, v in zip("TCZ", (ti, ci, zi), strict=True)}
                    key = tuple(idx[a] for a in axis_order) + (lv,)
                    planes[key] = plane

    stem = "syn"
    ets = tmp / f"_{stem}_" / "stack1" / "frame_t_0.ets"
    write_ets(
        ets,
        axis_order=list(axis_order),
        axis_sizes=axis_sizes,
        planes=planes,
        tile_size=tw,
        compression=compression,
        origin=origin,
    )
    vsi = tmp / f"{stem}.vsi"
    write_minimal_vsi(
        vsi,
        stack_id=1,
        width=x,
        height=y,
        axis_order=list(axis_order),
        axis_sizes=axis_sizes,
        channel_names=[f"C{i}" for i in range(c)],
        extra_image_ids=extra_images,
        plane_times_ms=plane_times_ms,
    )
    return vsi, ets


def write_1025_pyramid(
    tmp: Path,
    *,
    stored_padding: bool,
    extra_level: int | None = None,
) -> tuple[Path, Path]:
    """Build a 1025×1025 two- or three-level pyramid fixture (synthetic only)."""
    y = x = 1025
    tw = 512
    planes: dict[tuple[int, ...], np.ndarray] = {}
    l0 = np.arange(y * x, dtype=np.uint16).reshape(y, x)
    planes[(0, 0, 0, 0)] = l0
    l1: np.ndarray
    if stored_padding:
        l1 = np.zeros((513, 513), dtype=np.uint16)
        l1[:512, :512] = l0[:512, :512]
    else:
        l1 = l0[:512, :512].copy()
    planes[(0, 0, 0, 1)] = l1
    if stored_padding:
        l2 = np.zeros((257, 257), dtype=np.uint16)
        l2[:256, :256] = l1[:256, :256]
        planes[(0, 0, 0, 2)] = l2
    if extra_level is not None:
        planes[(0, 0, 0, extra_level)] = np.zeros((128, 128), dtype=np.uint16)
    tmp.mkdir(parents=True, exist_ok=True)
    ets = tmp / "_syn_" / "stack1" / "frame_t_0.ets"
    write_ets(
        ets,
        axis_order=["T", "Z", "C"],
        axis_sizes={"T": 1, "Z": 1, "C": 1},
        planes=planes,
        tile_size=tw,
    )
    vsi = tmp / "syn.vsi"
    write_minimal_vsi(
        vsi,
        stack_id=1,
        width=x,
        height=y,
        axis_order=["T", "Z", "C"],
        axis_sizes={"T": 1, "Z": 1, "C": 1},
        channel_names=["C0"],
    )
    return vsi, ets
