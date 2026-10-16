"""Procedural plane and tile pixels for anonymised fixture materialisation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from bioio_vsi._stack import level_shape

# 5x7 bitmap font (rows as bit strings).
_GLYPHS: dict[str, tuple[str, ...]] = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00110", "01000", "10000", "11111"),
    "3": ("01110", "10001", "00001", "00110", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("01110", "10000", "11110", "10001", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "10001", "01110"),
    "a": ("00000", "00000", "01110", "00001", "01111", "10001", "01111"),
    "b": ("10000", "10000", "11110", "10001", "10001", "10001", "11110"),
    "c": ("00000", "00000", "01110", "10000", "10000", "10001", "01110"),
    "d": ("00001", "00001", "01111", "10001", "10001", "10001", "01111"),
    "e": ("00000", "00000", "01110", "10001", "11111", "10000", "01110"),
    "f": ("00110", "01001", "01000", "11100", "01000", "01000", "01000"),
    "g": ("00000", "00000", "01111", "10001", "01111", "00001", "01110"),
    "h": ("10000", "10000", "10110", "11001", "10001", "10001", "10001"),
    "i": ("00100", "00000", "01100", "00100", "00100", "00100", "01110"),
    "j": ("00010", "00000", "00110", "00010", "00010", "10010", "01100"),
    "k": ("10000", "10000", "10010", "10100", "11000", "10100", "10010"),
    "l": ("01100", "00100", "00100", "00100", "00100", "00100", "01110"),
    "m": ("00000", "00000", "11010", "10101", "10101", "10001", "10001"),
    "n": ("00000", "00000", "10110", "11001", "10001", "10001", "10001"),
    "o": ("00000", "00000", "01110", "10001", "10001", "10001", "01110"),
    "p": ("00000", "00000", "11110", "10001", "11110", "10000", "10000"),
    "q": ("00000", "00000", "01111", "10001", "01111", "00001", "00001"),
    "r": ("00000", "00000", "10110", "11001", "10000", "10000", "10000"),
    "s": ("00000", "00000", "01111", "10000", "01110", "00001", "11110"),
    "t": ("01000", "01000", "11110", "01000", "01000", "01001", "00110"),
    "u": ("00000", "00000", "10001", "10001", "10001", "10011", "01101"),
    "v": ("00000", "00000", "10001", "10001", "10001", "01010", "00100"),
    "w": ("00000", "00000", "10001", "10001", "10101", "10101", "01010"),
    "x": ("00000", "00000", "10001", "01010", "00100", "01010", "10001"),
    "y": ("00000", "00000", "10001", "10001", "01111", "00001", "01110"),
    "z": ("00000", "00000", "11111", "00010", "00100", "01000", "11111"),
    # Upper-case L for the level, so "L1" does not read as "11".
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
}

_F_MARKER_ROWS = 12
_F_MARKER_COLS = 10
_FONT_ROWS = len(_GLYPHS[" "])


def _text_bitmap(text: str) -> np.ndarray:
    cols = max(1, len(text)) * 6
    bmp = np.zeros((_FONT_ROWS, cols), dtype=bool)
    for i, ch in enumerate(text):
        glyph = _GLYPHS.get(ch, _GLYPHS.get(ch.lower(), _GLYPHS[" "]))
        for row_i, row in enumerate(glyph):
            for col_i, bit in enumerate(row):
                if bit == "1":
                    bmp[row_i, i * 6 + col_i] = True
    return bmp


def _f_marker_bitmap() -> np.ndarray:
    bmp = np.zeros((_F_MARKER_ROWS, _F_MARKER_COLS), dtype=bool)
    bmp[0:2, 0:10] = True
    bmp[0:12, 0:2] = True
    bmp[5:7, 0:7] = True
    return bmp


def _text_width_cells(text: str) -> int:
    return max(1, len(text)) * 6


@dataclass(frozen=True)
class _PictureLayout:
    scale: int
    f_box: tuple[int, int, int, int]
    title_box: tuple[int, int, int, int]
    label_box: tuple[int, int, int, int]
    smiley_box: tuple[int, int, int, int]
    cy: int
    cx: int
    radius: int
    eye_r: int
    left_eye: tuple[int, int]
    right_eye: tuple[int, int]
    mouth_center: tuple[int, int]
    mouth_radius: int
    mouth_thick: int


def _box_from_origin(
    y0: int, x0: int, rows: int, cols: int, scale: int
) -> tuple[int, int, int, int]:
    return (y0, x0, y0 + rows * scale, x0 + cols * scale)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _layout_for_size(
    height: int,
    width: int,
    *,
    title: str,
    label: str,
) -> _PictureLayout:
    margin = max(1, min(height, width) // 64)
    text_rows = _FONT_ROWS
    title_cells = _text_width_cells(title)
    label_cells = _text_width_cells(label)

    stem_w = 2
    scale = 1
    for candidate in range(max(1, min(height, width) // 8), 0, -1):
        f_w = _F_MARKER_COLS * candidate
        f_h = _F_MARKER_ROWS * candidate
        title_w = title_cells * candidate
        label_w = label_cells * candidate
        beside = margin + f_w + candidate + title_w <= width
        if beside:
            title_x = margin + f_w + candidate
            title_y = margin
        else:
            title_x = margin + stem_w * candidate + candidate
            title_y = margin + f_h + candidate
        label_x = title_x
        label_y = title_y + text_rows * candidate + candidate
        header_bottom = label_y + text_rows * candidate + margin
        smiley_y0 = header_bottom + margin
        smiley_y1 = height - margin
        smiley_x0 = margin
        smiley_x1 = width - margin
        if smiley_y1 - smiley_y0 < 16 or smiley_x1 - smiley_x0 < 16:
            continue
        if title_x + title_w > width or label_x + label_w > width:
            continue
        if header_bottom > height // 2:
            continue
        scale = candidate
        break

    f_box = _box_from_origin(margin, margin, _F_MARKER_ROWS, _F_MARKER_COLS, scale)
    f_w = _F_MARKER_COLS * scale
    f_h = _F_MARKER_ROWS * scale
    beside = margin + f_w + scale + title_cells * scale <= width
    if beside:
        title_box = _box_from_origin(margin, margin + f_w + scale, text_rows, title_cells, scale)
    else:
        title_box = _box_from_origin(
            margin + f_h + scale,
            margin + stem_w * scale + scale,
            text_rows,
            title_cells,
            scale,
        )
    label_y = title_box[2] + scale
    label_box = _box_from_origin(label_y, title_box[1], text_rows, label_cells, scale)
    smiley_y0 = label_box[2] + margin
    smiley_box = (smiley_y0, margin, height - margin, width - margin)
    cy = (smiley_box[0] + smiley_box[2]) // 2
    cx = (smiley_box[1] + smiley_box[3]) // 2
    radius = max(4, min(smiley_box[2] - smiley_box[0], smiley_box[3] - smiley_box[1]) // 2 - scale)
    radius = min(radius, max(4, min(height, width) // 6))
    eye_r = max(1, radius // 5)
    left_eye = (cy - radius // 3, cx - radius // 2)
    right_eye = (cy - radius // 3, cx + radius // 2)
    mouth_center = (cy + radius // 3, cx - radius // 6)
    mouth_radius = max(2, int(radius * 0.55))
    mouth_thick = max(2, scale)
    return _PictureLayout(
        scale=scale,
        f_box=f_box,
        title_box=title_box,
        label_box=label_box,
        smiley_box=smiley_box,
        cy=cy,
        cx=cx,
        radius=radius,
        eye_r=eye_r,
        left_eye=left_eye,
        right_eye=right_eye,
        mouth_center=mouth_center,
        mouth_radius=mouth_radius,
        mouth_thick=mouth_thick,
    )


def layout_element_boxes(
    height: int,
    width: int,
    *,
    t: int,
    c: int,
    z: int,
    level: int,
    axis_order: Sequence[str],
) -> dict[str, tuple[int, int, int, int]]:
    """Axis-aligned bounds of major drawn elements (for layout tests)."""
    label = plane_label(t=t, c=c, z=z, level=level, axis_order=axis_order)
    lay = _layout_for_size(height, width, title="bioio-vsi", label=label)
    ley, lex = lay.left_eye
    rey, rex = lay.right_eye
    wink_half = max(1, lay.eye_r)
    wink = (rey - wink_half, rex - lay.eye_r, rey + wink_half + 1, rex + lay.eye_r + 1)
    eye = (
        ley - lay.eye_r,
        lex - lay.eye_r,
        ley + lay.eye_r + 1,
        lex + lay.eye_r + 1,
    )
    mcy, mcx = lay.mouth_center
    mouth = (
        mcy - lay.mouth_radius - lay.mouth_thick,
        mcx - lay.mouth_radius - lay.mouth_thick,
        mcy + lay.mouth_radius + lay.mouth_thick,
        mcx + lay.mouth_radius + lay.mouth_thick,
    )
    face = (
        lay.cy - lay.radius,
        lay.cx - lay.radius,
        lay.cy + lay.radius + 1,
        lay.cx + lay.radius + 1,
    )
    return {
        "f": lay.f_box,
        "title": lay.title_box,
        "label": lay.label_box,
        "face": face,
        "left_eye": eye,
        "right_wink": wink,
        "mouth": mouth,
    }


def plane_label(
    *,
    t: int,
    c: int,
    z: int,
    level: int,
    axis_order: Sequence[str],
) -> str:
    parts: list[str] = []
    idx = {"T": t, "C": c, "Z": z}
    for ax in axis_order:
        if ax in idx:
            parts.append(f"{ax.lower()}{idx[ax]}")
    parts.append(f"L{level}")
    return " ".join(parts)


def _scaled_bitmap_mask(
    gy: np.ndarray,
    gx: np.ndarray,
    y0: int,
    x0: int,
    bmp: np.ndarray,
    scale: int,
) -> np.ndarray:
    row = (gy - y0) // scale
    col = (gx - x0) // scale
    h, w = bmp.shape
    valid = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    safe_row = np.clip(row, 0, h - 1)
    safe_col = np.clip(col, 0, w - 1)
    return np.asarray(valid & bmp[safe_row, safe_col])


def _padding_value(gy: np.ndarray, gx: np.ndarray, peak_pad: int) -> np.ndarray:
    return (gx.astype(np.int64) * 13 + gy.astype(np.int64) * 7) % peak_pad


def render_tile(
    level_h: int,
    level_w: int,
    x0: int,
    y0: int,
    tile_h: int,
    tile_w: int,
    *,
    dtype: np.dtype,
    t: int,
    c: int,
    z: int,
    level: int,
    axis_order: Sequence[str],
) -> np.ndarray:
    """Render one tile from level-local coordinates (vectorised)."""
    peak = 255 if dtype == np.dtype(np.uint8) else 65535
    peak_pad = 255 if dtype == np.dtype(np.uint8) else 4095
    gy = np.arange(tile_h, dtype=np.int64)[:, None] + y0
    gx = np.arange(tile_w, dtype=np.int64)[None, :] + x0
    ramp = (gx * 3 + gy * 5 + t * 997 + c * 131 + z * 17 + level * 4099) % peak
    tile = ramp.astype(dtype)

    inside = (gy >= 0) & (gy < level_h) & (gx >= 0) & (gx < level_w)
    pad = _padding_value(gy, gx, peak_pad).astype(dtype)
    tile = np.where(inside, tile, pad)
    label = plane_label(t=t, c=c, z=z, level=level, axis_order=axis_order)
    lay = _layout_for_size(level_h, level_w, title="bioio-vsi", label=label)
    face_val = np.int64((peak * 3) // 5)
    dark = np.int64(peak // 10)
    stroke = np.int64(peak // 4)

    dist_face = np.sqrt((gy - lay.cy) ** 2 + (gx - lay.cx) ** 2)
    face = inside & (dist_face <= lay.radius)
    tile[face] = face_val.astype(dtype)

    ley, lex = lay.left_eye
    left_eye = inside & face & ((gy - ley) ** 2 + (gx - lex) ** 2 <= lay.eye_r**2)
    tile[left_eye] = dark.astype(dtype)

    rey, rex = lay.right_eye
    wink = (
        inside
        & (np.abs(gy - rey) <= max(0, lay.eye_r // 2))
        & (gx >= rex - lay.eye_r)
        & (gx <= rex + lay.eye_r)
    )
    tile[wink] = dark.astype(dtype)

    mcy, mcx = lay.mouth_center
    dist_m = np.sqrt((gy - mcy) ** 2 + (gx - mcx) ** 2)
    angle = np.arctan2(gy - mcy, gx - mcx)
    mouth = (
        inside
        & face
        & (dist_m >= lay.mouth_radius - lay.mouth_thick)
        & (dist_m <= lay.mouth_radius + lay.mouth_thick)
        & (angle > 0)
        & (angle < np.pi)
    )
    tile[mouth] = stroke.astype(dtype)

    f_mask = _scaled_bitmap_mask(gy, gx, lay.f_box[0], lay.f_box[1], _f_marker_bitmap(), lay.scale)
    title_mask = _scaled_bitmap_mask(
        gy, gx, lay.title_box[0], lay.title_box[1], _text_bitmap("bioio-vsi"), lay.scale
    )
    label_mask = _scaled_bitmap_mask(
        gy, gx, lay.label_box[0], lay.label_box[1], _text_bitmap(label), lay.scale
    )
    ink = peak - 1
    for mask in (f_mask, title_mask, label_mask):
        tile[inside & mask] = np.int64(ink).astype(dtype)

    tile = np.clip(tile.astype(np.int64) + c * (peak // 8), 0, peak).astype(dtype)
    return tile


def synthetic_plane(
    height: int,
    width: int,
    *,
    dtype: np.dtype,
    t: int,
    c: int,
    z: int,
    level: int,
    axis_order: Sequence[str],
) -> np.ndarray:
    """Full plane picture at this pyramid level (geometry is level-local)."""
    return render_tile(
        height,
        width,
        0,
        0,
        height,
        width,
        dtype=dtype,
        t=t,
        c=c,
        z=z,
        level=level,
        axis_order=axis_order,
    )


def expected_tile_pixels(
    *,
    height: int,
    width: int,
    dtype: np.dtype,
    t: int,
    c: int,
    z: int,
    level: int,
    axis_order: Sequence[str],
    tile_x: int,
    tile_y: int,
    tile_w: int,
    tile_h: int,
    origin_x: int = 0,
    origin_y: int = 0,
) -> np.ndarray:
    level_h, level_w = level_shape(height, width, level)
    x0 = tile_x * tile_w + origin_x
    y0 = tile_y * tile_h + origin_y
    return render_tile(
        level_h,
        level_w,
        x0,
        y0,
        tile_h,
        tile_w,
        dtype=dtype,
        t=t,
        c=c,
        z=z,
        level=level,
        axis_order=axis_order,
    )


def assert_layout_disjoint(height: int, width: int, *, level: int = 2) -> None:
    """Raise AssertionError if header elements overlap or intrude on the smiley band."""
    boxes = layout_element_boxes(
        height, width, t=0, c=1, z=7, level=level, axis_order=["T", "C", "Z"]
    )
    header_keys = ("f", "title", "label")
    for i, a in enumerate(header_keys):
        for b in header_keys[i + 1 :]:
            assert not _boxes_overlap(boxes[a], boxes[b]), f"{a} overlaps {b}"
    smiley = boxes["face"]
    for key in header_keys:
        assert not _boxes_overlap(boxes[key], smiley), f"{key} overlaps smiley"


__all__ = [
    "expected_tile_pixels",
    "layout_element_boxes",
    "plane_label",
    "render_tile",
    "synthetic_plane",
]
