from __future__ import annotations

import os
from pathlib import Path

import pytest

RESOURCES = Path(__file__).parent / "resources"
COMMITTED_VSI = RESOURCES / "vsi-ets-test-jpg2k.vsi"


def public_samples_root() -> Path:
    return Path(os.environ.get("VSI_PUBLIC_SAMPLES", "data"))


@pytest.fixture
def committed_vsi() -> Path:
    return COMMITTED_VSI


@pytest.fixture
def public_samples() -> Path:
    return public_samples_root()


def has_imagecodecs() -> bool:
    try:
        import imagecodecs  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture
def require_imagecodecs() -> None:
    pytest.importorskip("imagecodecs")


# Before a release every comparison must run: with VSI_RELEASE_CHECK=1 a test
# skipped for a missing sample, codec or oracle fails instead. Skips that mark a
# combination as not applicable are kept.
_NOT_APPLICABLE = "mutation does not apply"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):  # type: ignore[no-untyped-def]
    outcome = yield
    report = outcome.get_result()
    if os.environ.get("VSI_RELEASE_CHECK") != "1" or not report.skipped:
        return
    reason = str(report.longrepr[2]) if isinstance(report.longrepr, tuple) else ""
    if _NOT_APPLICABLE not in reason:
        report.outcome = "failed"
        report.longrepr = f"skipped during a release check: {reason}"
