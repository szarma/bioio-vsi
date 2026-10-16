"""Layout and rendering tests for procedural fixture pixels."""

from __future__ import annotations

import numpy as np
import pytest

from bioio_vsi._fixture_pixels import (
    assert_layout_disjoint,
    layout_element_boxes,
    render_tile,
    synthetic_plane,
)


@pytest.mark.parametrize(
    ("height", "width"),
    [
        (64, 64),
        (200, 300),
        (2048, 2048),
        (4096, 512),
    ],
)
def test_fixture_layout_disjoint(height: int, width: int) -> None:
    assert_layout_disjoint(height, width, level=3)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def test_fixture_layout_boxes_inside_plane() -> None:
    h, w = 300, 200
    boxes = layout_element_boxes(h, w, t=0, c=1, z=7, level=2, axis_order=["T", "C", "Z"])
    for name, (y0, x0, y1, x1) in boxes.items():
        assert 0 <= y0 < y1 <= h, name
        assert 0 <= x0 < x1 <= w, name
    header = ("f", "title", "label")
    for i, a in enumerate(header):
        for b in header[i + 1 :]:
            assert not _boxes_overlap(boxes[a], boxes[b])


@pytest.mark.parametrize(
    ("height", "width"),
    [(64, 64), (200, 300), (2048, 2048), (4096, 512)],
)
def test_fixture_plane_asymmetry(height: int, width: int) -> None:
    plane = synthetic_plane(
        height,
        width,
        dtype=np.dtype(np.uint16),
        t=0,
        c=1,
        z=7,
        level=2,
        axis_order=["T", "C", "Z"],
    )
    assert not np.array_equal(plane, np.fliplr(plane))
    assert not np.array_equal(plane, np.flipud(plane))


def test_render_tile_matches_plane_crop() -> None:
    h, w = 128, 128
    plane = synthetic_plane(
        h,
        w,
        dtype=np.dtype(np.uint8),
        t=1,
        c=0,
        z=2,
        level=1,
        axis_order=["Z", "C", "T"],
    )
    tile = render_tile(
        h,
        w,
        32,
        16,
        64,
        48,
        dtype=np.dtype(np.uint8),
        t=1,
        c=0,
        z=2,
        level=1,
        axis_order=["Z", "C", "T"],
    )
    np.testing.assert_array_equal(tile, plane[16 : 16 + 64, 32 : 32 + 48])
