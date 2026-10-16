from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from bioio_vsi import describe
from bioio_vsi.tests.conftest import COMMITTED_VSI
from bioio_vsi.tests.synthetic import make_stack


def test_describe_committed() -> None:
    text = describe(str(COMMITTED_VSI))
    assert "Format: 3.22" in text
    assert "JPEG 2000" in text
    assert "slot order" in text


def test_cli_describe() -> None:
    out = subprocess.check_output(
        [sys.executable, "-m", "bioio_vsi", "describe", str(COMMITTED_VSI)],
        text=True,
    )
    assert "vsi-ets-test-jpg2k" in out or "File:" in out


def test_describe_ambiguous_and_jpeg_header(tmp_path: Path) -> None:
    vsi, ets = make_stack(tmp_path, shape=(1, 1, 1, 16, 16), tile_size=16)
    extra = ets.parent / "frame_t_1.ets"
    extra.write_bytes(ets.read_bytes())
    text = describe(str(vsi))
    assert "ambiguous" in text
    vsi_jpeg, _ets_j = make_stack(
        tmp_path / "j",
        shape=(1, 1, 1, 16, 16),
        tile_size=16,
        compression=2,
    )
    text_j = describe(str(vsi_jpeg))
    assert "present" in text_j
    assert "JPEG" in text_j
    assert "unreadable" not in text_j


def test_describe_redact_omits_paths(tmp_path: Path) -> None:
    vsi, _ = make_stack(tmp_path, shape=(1, 1, 1, 8, 8), tile_size=8)
    text = describe(str(vsi), redact=True)
    assert str(vsi) not in text
    assert "(redacted)" in text
