"""bioio plugin metadata for the VSI reader."""

from __future__ import annotations

import bioio_base.reader
import bioio_base.reader_metadata


class ReaderMetadata(bioio_base.reader_metadata.ReaderMetadata):
    @staticmethod
    def get_supported_extensions() -> list[str]:
        return [".vsi"]

    @staticmethod
    def get_reader() -> bioio_base.reader.Reader:
        from bioio_vsi.reader import Reader

        return Reader  # type: ignore[return-value]
