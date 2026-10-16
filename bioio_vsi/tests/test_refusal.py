from __future__ import annotations

import shutil
import struct
from pathlib import Path

import pytest

from bioio_vsi import describe, open_vsi
from bioio_vsi.errors import (
    InvalidStructureError,
    MissingCodecsExtraError,
    MissingPixelFileError,
    MultiImageError,
    SizeMismatchError,
    TruncatedFileError,
    UndeclaredCoordinateError,
    UnknownDimensionKindError,
    UnknownUnitError,
    UnsupportedCompressionError,
    UnsupportedPixelTypeError,
)
from bioio_vsi.tests.conftest import COMMITTED_VSI, public_samples_root
from bioio_vsi.tests.synthetic import make_stack, write_ets, write_minimal_vsi

SLIDE_SCANNERS = [
    "ome-cellsens/evident/Image_V4.1_BF.vsi",
    "ome-cellsens/evident/Image_V4.1_FL_2D.vsi",
    "ome-cellsens/evident/Image_V4.1_FL_3D.vsi",
    "zenodo-8161864/VS200_Bioformats-issue/Image_L2410_Part3_1_H+E_20X.vsi",
]


def _sample(rel: str) -> Path | None:
    p = public_samples_root() / rel
    return p if p.is_file() else None


def test_missing_ets(tmp_path: Path) -> None:
    vsi, ets = make_stack(tmp_path, shape=(1, 1, 1, 16, 16), tile_size=16)
    ets.unlink()
    with pytest.raises(MissingPixelFileError):
        open_vsi(str(vsi))


def test_multi_image(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 16, 16), tile_size=16, extra_images=[10000])
    with pytest.raises(MultiImageError):
        open_vsi(str(vsi))


def test_positive_origin_offset(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 32, 32), tile_size=32, origin=(3, 0))
    with pytest.raises(InvalidStructureError, match="positive"):
        open_vsi(str(vsi))


def test_jpeg_compression(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 16, 16), tile_size=16, compression=2)
    with pytest.raises(UnsupportedCompressionError):
        open_vsi(str(vsi))


def test_unknown_compression_code(tmp_path: Path) -> None:
    vsi, ets = make_stack(tmp_path, shape=(1, 1, 1, 16, 16), tile_size=16)
    data = bytearray(ets.read_bytes())
    struct.pack_into("<I", data, 64 + 20, 99)
    ets.write_bytes(data)
    with pytest.raises(UnsupportedCompressionError):
        open_vsi(str(vsi))


def test_bad_pixel_type(tmp_path: Path) -> None:
    vsi, ets = make_stack(tmp_path, shape=(1, 1, 1, 16, 16), tile_size=16)
    data = bytearray(ets.read_bytes())
    struct.pack_into("<I", data, 64 + 8, 7)
    ets.write_bytes(data)
    with pytest.raises(UnsupportedPixelTypeError):
        open_vsi(str(vsi))


def test_truncated_table(tmp_path: Path) -> None:
    vsi, ets = make_stack(tmp_path, shape=(1, 1, 1, 16, 16), tile_size=16)
    data = ets.read_bytes()[:-10]
    ets.write_bytes(data)
    with pytest.raises(TruncatedFileError):
        open_vsi(str(vsi))


def test_size_mismatch_vsi_ets(tmp_path: Path) -> None:
    vsi, ets = make_stack(tmp_path, shape=(1, 1, 1, 16, 16), tile_size=16)
    write_minimal_vsi(
        vsi,
        stack_id=1,
        width=20,
        height=20,
        axis_order=["T", "Z", "C"],
        axis_sizes={"T": 1, "Z": 1, "C": 1},
        channel_names=["C0"],
    )
    with pytest.raises(SizeMismatchError):
        open_vsi(str(vsi))


def test_missing_tile(tmp_path: Path) -> None:
    vsi, ets = make_stack(tmp_path, shape=(1, 1, 1, 64, 64), tile_size=32)
    data = bytearray(ets.read_bytes())
    # Remove the last tile record from the table.
    rec = 4 + 6 * 4 + 8 + 4 + 4
    ets.write_bytes(bytes(data[:-rec]))
    with pytest.raises(TruncatedFileError):
        open_vsi(str(vsi))


def test_jpeg2000_without_codecs(monkeypatch: pytest.MonkeyPatch) -> None:
    import bioio_vsi._stack as stack_mod

    def fail_j2k(raw, compression, dtype, shape):  # type: ignore[no-untyped-def]
        if compression == 3:
            raise MissingCodecsExtraError(
                "JPEG 2000 tiles require the optional codecs extra: "
                "pip install bioio-vsi[codecs] (provides imagecodecs)"
            )
        raise AssertionError("unexpected compression")

    monkeypatch.setattr(stack_mod, "decode_tile", fail_j2k)
    img = open_vsi(str(COMMITTED_VSI))
    with pytest.raises(MissingCodecsExtraError):
        img.read_plane()


def test_undeclared_coordinate(tmp_path: Path) -> None:
    """ETS has an extra varying slot not declared in the VSI."""
    planes = {(0, 0, 0): __import__("numpy").zeros((16, 16), dtype=__import__("numpy").uint16)}
    ets = tmp_path / "x.ets"
    write_ets(
        ets,
        axis_order=["T", "Z", "C"],
        axis_sizes={"T": 1, "Z": 1, "C": 2},
        planes={(0, 0, 0, 0): planes[(0, 0, 0)], (0, 0, 1, 0): planes[(0, 0, 0)]},
        tile_size=16,
    )
    stack = tmp_path / "_x_" / "stack1"
    stack.mkdir(parents=True)
    (stack / "frame_t_0.ets").write_bytes(ets.read_bytes())
    vsi = tmp_path / "x.vsi"
    write_minimal_vsi(
        vsi,
        stack_id=1,
        width=16,
        height=16,
        axis_order=["T", "Z", "C"],
        axis_sizes={"T": 1, "Z": 1, "C": 1},
        channel_names=["C0"],
    )
    with pytest.raises((SizeMismatchError, UndeclaredCoordinateError)):
        open_vsi(str(vsi))


@pytest.mark.parametrize("rel", SLIDE_SCANNERS)
def test_slide_scanner_refused_and_described(rel: str) -> None:
    vsi = _sample(rel)
    if vsi is None:
        pytest.skip("slide scanner sample not on disk")
    with pytest.raises(MultiImageError):
        open_vsi(str(vsi))
    text = describe(str(vsi))
    assert text.count("Image stack") >= 2
    assert "pixel file" in text


def test_lambda_dimension_refused(tmp_path: Path) -> None:
    vsi, ets = make_stack(tmp_path, shape=(1, 1, 1, 16, 16), tile_size=16)
    stack = tmp_path / "_lam_" / "stack1"
    stack.mkdir(parents=True)
    (stack / "frame_t_0.ets").write_bytes(ets.read_bytes())
    lam_vsi = tmp_path / "lam.vsi"
    write_minimal_vsi(
        lam_vsi,
        stack_id=1,
        width=16,
        height=16,
        axis_order=["Z"],
        axis_sizes={"Z": 1},
        channel_names=[],
        kind_codes={"Z": 3},
    )
    with pytest.raises(UnknownDimensionKindError, match="lambda"):
        open_vsi(str(lam_vsi))


@pytest.mark.parametrize(
    ("stored", "replacement"),
    [
        ("10^-6m^1", "10^-9m^1"),  # micrometres read as nanometres
        ("10^-6m^1", "10^-3m^1"),  # ... as millimetres
        ("10^-3s^1", "10^-6s^1"),  # milliseconds as microseconds
        ("10^-9m^1", "10^-6m^1"),  # emission wavelength in micrometres
    ],
)
def test_unexpected_unit(tmp_path: Path, stored: str, replacement: str) -> None:
    """Unit strings are compared in full: units starting with "1" used to pass."""
    shutil.copytree(COMMITTED_VSI.parent, tmp_path, dirs_exist_ok=True)
    vsi = tmp_path / COMMITTED_VSI.name
    data = vsi.read_bytes()
    assert stored.encode("utf-16-le") in data
    vsi.write_bytes(data.replace(stored.encode("utf-16-le"), replacement.encode("utf-16-le")))
    with pytest.raises(UnknownUnitError):
        open_vsi(str(vsi))
