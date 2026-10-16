"""Errors raised when a VSI/ETS file cannot be read as understood."""

from __future__ import annotations

from pathlib import Path

PathLike = str | Path


class VsiError(Exception):
    """Base class for all reader errors."""

    path: str | None
    detail: str

    def __init__(
        self,
        message: str = "",
        *,
        path: PathLike | None = None,
        detail: str | None = None,
    ) -> None:
        if detail is not None:
            self.detail = detail
        else:
            self.detail = message
        self.path = str(path) if path is not None else None
        super().__init__(self.public_message(redact_paths=False))

    def public_message(self, *, redact_paths: bool) -> str:
        if redact_paths or self.path is None:
            return self.detail
        return f"{self.path}: {self.detail}"

    def redacted_str(self) -> str:
        return self.public_message(redact_paths=True)

    def __str__(self) -> str:
        return self.public_message(redact_paths=False)


class MissingPixelFileError(VsiError):
    """The image declares an external pixel file but none was found."""


class MultiImageError(VsiError):
    """The file contains more than one image (slide-scanner layout)."""


class UnsupportedCompressionError(VsiError):
    """Tile compression is not supported by this release."""


class UnsupportedPixelTypeError(VsiError):
    """Pixel type in the ETS header is not uint8 or uint16."""


class UndeclaredCoordinateError(VsiError):
    """A coordinate slot varies in the tile table but is not declared in the VSI."""


class UnknownDimensionKindError(VsiError):
    """A declared dimension kind is not Z, time, or channel."""


class UnknownUnitError(VsiError):
    """A metadata value uses a unit field this reader does not recognise."""


class SizeMismatchError(VsiError):
    """Sizes disagree between the VSI metadata and the ETS header or tile table."""


class IncompleteTileGridError(VsiError):
    """The tile table does not cover every coordinate on a pyramid level."""


class DuplicateTileError(VsiError):
    """The tile table lists the same coordinates twice."""


class TruncatedFileError(VsiError):
    """A read would extend past the end of the file."""


class InvalidStructureError(VsiError):
    """The file structure is invalid or an offset points outside the file."""


class MissingCodecsExtraError(VsiError):
    """A compressed tile needs the optional codecs extra (imagecodecs)."""


class MultipleEtsCandidatesError(VsiError):
    """More than one frame_t*.ets file was found for the image."""


class InvalidIdentifierError(VsiError):
    """A dimension or channel identifier is missing, duplicated, or out of order."""


class UnsupportedColourSpaceError(VsiError):
    """The ETS header declares an unsupported colour space."""


class UnsupportedTileLayoutError(VsiError):
    """Tile depth or sample count in the ETS header is not supported."""


class NoImageError(VsiError):
    """The VSI contains no image volume."""


class AmbiguousMetadataError(VsiError):
    """More than one tree node matches an interpreted metadata path."""


class InvalidMetadataError(VsiError):
    """A metadata value has the wrong type, shape, or range."""


class UnsupportedFeatureError(VsiError):
    """The file uses a construct this release does not interpret."""
