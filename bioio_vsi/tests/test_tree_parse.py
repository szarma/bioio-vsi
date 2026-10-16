"""VSI tag tree parsing: field chains, inline values, truncation, roots."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from bioio_vsi import open_vsi
from bioio_vsi._tree import FieldNode, collection_root, kids, load_tree
from bioio_vsi.errors import (
    AmbiguousMetadataError,
    InvalidStructureError,
    TruncatedFileError,
    UnsupportedFeatureError,
)
from bioio_vsi.tests import synthetic as syn
from bioio_vsi.tests.conftest import COMMITTED_VSI, public_samples_root
from bioio_vsi.tests.synthetic import make_stack
from bioio_vsi.tests.tree_rewrite import nested_volume_pos, rewrite_vsi


def test_truncated_scalar_field_raises(tmp_path: Path) -> None:
    data = bytearray(COMMITTED_VSI.read_bytes())
    del data[-32:]
    broken = tmp_path / "trunc.vsi"
    broken.write_bytes(data)
    with pytest.raises(TruncatedFileError, match="need .* only"):
        load_tree(str(broken))


def test_field_chain_early_stop_hides_second_image(tmp_path: Path) -> None:
    vsi, _ = make_stack(
        tmp_path,
        shape=(1, 1, 1, 8, 8),
        tile_size=8,
        extra_images=[10000],
    )
    tree = load_tree(str(vsi))
    first = kids(collection_root(tree), 2001)[0]
    data = bytearray(vsi.read_bytes())
    struct.pack_into("<i", data, first.file_offset + 8, 0)
    vsi.write_bytes(data)
    with pytest.raises(InvalidStructureError, match="field chain ended"):
        load_tree(str(vsi))


def test_field_chain_declared_count_exceeds_chain(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 8, 8), tile_size=8)
    tree = load_tree(str(vsi))
    vol = kids(collection_root(tree), 2001)[0]
    data = bytearray(vsi.read_bytes())
    vol_pos = nested_volume_pos(data, vol)
    flags = struct.unpack_from("<I", data, vol_pos + 16)[0]
    struct.pack_into("<I", data, vol_pos + 16, flags + 3)
    vsi.write_bytes(data)
    with pytest.raises(InvalidStructureError):
        load_tree(str(vsi))


def test_field_chain_revisits_field_offset(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 8, 8), tile_size=8)
    tree = load_tree(str(vsi))
    vol = kids(collection_root(tree), 2001)[0]
    data = bytearray(vsi.read_bytes())
    vol_pos = nested_volume_pos(data, vol)
    first_rel = struct.unpack_from("<I", data, vol_pos + 8)[0]
    p = vol_pos + first_rel
    ft, tag, nxt, size = struct.unpack_from("<iiiI", data, p)
    if nxt == 0:
        pytest.skip("image volume has only one field")
    struct.pack_into("<i", data, p + 8, first_rel)
    vsi.write_bytes(data)
    with pytest.raises(InvalidStructureError):
        load_tree(str(vsi))


@pytest.mark.parametrize("use_inline", [False, True])
def test_inline_signed_matches_out_of_line(tmp_path: Path, use_inline: bool) -> None:
    vsi, _ = make_stack(
        tmp_path / f"signed_{use_inline}",
        shape=(1, 1, 2, 8, 8),
        tile_size=8,
    )

    def trans(n: FieldNode) -> bytes | None:
        if n.tag != 2013:
            return None
        rawbits = 0xFFFFFFFF
        if use_inline:
            val = struct.pack("<iiiI", 0x40000000 | 5, 268435458, 0, rawbits)
        else:
            val = syn._field_data(268435458, 5, struct.pack("<i", -1))
        return syn._field_nested(2013, [val])

    rewrite_vsi(vsi, trans)
    img = open_vsi(str(vsi))
    assert img.z_step_um == -1.0


def test_inline_uninterpreted_on_z_step_raises(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 2, 8, 8), tile_size=8)

    def trans(n: FieldNode) -> bytes | None:
        if n.tag != 2013:
            return None
        val = struct.pack("<iiiI", 0x40000000 | 9, 268435458, 0, 0x3F000000)
        return syn._field_nested(2013, [val])

    rewrite_vsi(vsi, trans)
    with pytest.raises(UnsupportedFeatureError):
        open_vsi(str(vsi))


def test_uninterpreted_inline_on_unread_tag_still_opens(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 8, 8), tile_size=8)

    def trans(n: FieldNode) -> bytes | None:
        if n.tag != 34:
            return None
        return struct.pack("<iiiI", 0x40000000 | 9, 34, 0, 0x3F000000)

    rewrite_vsi(vsi, trans)
    open_vsi(str(vsi))


def test_duplicate_collection_root_is_ambiguous(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 8, 8), tile_size=8)
    tree = load_tree(str(vsi))
    tree.root.append(tree.root[[n.tag for n in tree.root].index(2000)])
    with pytest.raises(AmbiguousMetadataError):
        collection_root(tree)


def test_evident_tree_skips_oversized_blobs() -> None:
    evident = public_samples_root() / "ome-cellsens/evident/Image_V4.1_BF.vsi"
    if not evident.is_file():
        pytest.skip("no evident sample")
    tree = load_tree(str(evident))
    assert tree.stats.get("skipped_ext_blobs", 0) > 0
