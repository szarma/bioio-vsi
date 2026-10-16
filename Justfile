# Development commands (uv)

test:
    uv run pytest

lint:
    uv run ruff check bioio_vsi
    uv run mypy bioio_vsi

check: lint test

# Before a release: public samples, imagecodecs and slideio must all be present;
# any test that would skip for lack of them fails instead.
release-check:
    VSI_RELEASE_CHECK=1 uv run --extra test --extra codecs pytest
