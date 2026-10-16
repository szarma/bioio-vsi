"""Resolve user-supplied paths once at API entry."""

from __future__ import annotations

from pathlib import Path

PathLike = str | Path


def resolve_path(path: PathLike) -> str:
    """Return an absolute, normalized path for a local file."""
    return str(Path(path).expanduser().resolve())
