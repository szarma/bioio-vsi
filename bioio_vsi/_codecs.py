"""Tile decompression (optional imagecodecs)."""

from __future__ import annotations

import numpy as np

from bioio_vsi.errors import MissingCodecsExtraError, UnsupportedCompressionError

# Decision 7: raw and JPEG 2000 only for release 1.
SUPPORTED_COMPRESSION = frozenset({0, 3})


def decode_tile(
    raw: bytes,
    compression: int,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.ndarray:
    if compression == 0:
        file_dtype = np.dtype("<u1") if dtype == np.dtype(np.uint8) else np.dtype("<u2")
        arr = np.frombuffer(raw, dtype=file_dtype).reshape(shape)
        if arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
        return arr
    if compression == 2:
        raise UnsupportedCompressionError(
            "ETS compression code 2 (JPEG) is not supported in release 1"
        )
    if compression == 3:
        try:
            import imagecodecs
        except ImportError as exc:
            raise MissingCodecsExtraError(
                "JPEG 2000 tiles require the optional codecs extra: "
                "pip install bioio-vsi[codecs] (provides imagecodecs)"
            ) from exc
        arr = np.asarray(imagecodecs.jpeg2k_decode(raw))
        if arr.shape != shape or arr.dtype != dtype:
            raise ValueError(
                f"JPEG 2000 tile decoded to shape {arr.shape} dtype {arr.dtype}, "
                f"expected {shape} {dtype}"
            )
        return arr
    raise UnsupportedCompressionError(
        f"ETS compression code {compression} is not supported"
    )
