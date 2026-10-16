"""JVM-free reader for Olympus/Evident cellSens VSI microscopy images."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from bioio_vsi._describe import describe_vsi as describe
from bioio_vsi._paths import resolve_path
from bioio_vsi._stack import OrphanEtsImage, VsiImage
from bioio_vsi._vsi import load_vsi
from bioio_vsi.errors import VsiError

if TYPE_CHECKING:
    from bioio_vsi.reader import Reader
    from bioio_vsi.reader_metadata import ReaderMetadata

__all__ = [
    "VsiError",
    "Reader",
    "ReaderMetadata",
    "anonymize",
    "describe",
    "materialize",
    "expected_tile_pixels",
    "open_ets",
    "open_vsi",
    "OrphanEtsImage",
    "VsiImage",
]


def open_vsi(path: str) -> VsiImage:
    """Open a single-image cellSens VSI and its external ETS pixel file."""
    meta = load_vsi(path)
    return VsiImage(meta)


def anonymize(path: str, out_dir: str, *, synthetic_numbers: bool = True) -> Path:
    from bioio_vsi._anonymize import anonymize as _anonymize

    return _anonymize(path, out_dir, synthetic_numbers=synthetic_numbers)


def materialize(
    fixture_dir: str,
    out_dir: str,
    *,
    planes: str = "default",
) -> Path:
    from bioio_vsi._materialize import materialize as _materialize

    p: Literal["default", "all"] = cast(Literal["default", "all"], planes)
    return _materialize(fixture_dir, out_dir, planes=p)


def expected_tile_pixels(**kwargs: Any) -> Any:
    from bioio_vsi._fixture_pixels import expected_tile_pixels as _fn

    return _fn(**kwargs)


def open_ets(path: str) -> OrphanEtsImage:
    """Open an orphan ETS pixel file (geometry only, no axis labels)."""
    return OrphanEtsImage(resolve_path(path))


def __getattr__(name: str) -> Any:
    if name == "Reader":
        from bioio_vsi.reader import Reader as _Reader

        return _Reader
    if name == "ReaderMetadata":
        from bioio_vsi.reader_metadata import ReaderMetadata as _ReaderMetadata

        return _ReaderMetadata
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
