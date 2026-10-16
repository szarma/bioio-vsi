# bioio-vsi

Read `README.md` first: what the reader does, what it refuses, and that the
format is reverse engineered. This file adds what is needed to change the code.
The package is a bioio-independent core (`open_vsi`, `open_ets`, `describe`)
with a bioio `Reader` on top.

## Rules

1. **No JVM and no Bio-Formats at runtime.**
2. **Do not copy or translate Bio-Formats code** (GPL), and do not read its
   `CellSensReader` source. Its output may be used as a test oracle. slideio's
   VSI driver is BSD-3 and may be consulted; credit it where it helped.
3. **Fail loudly, never guess.** Anything not understood that affects pixels,
   axes or calibration raises a named `VsiError`: a plausible wrong answer
   is worse than a refusal.
4. **Stay small.** Runtime dependencies are bioio-base and numpy; codecs are an
   optional extra; slideio is a test oracle only.
5. **Read lazily.** Opening reads headers and the tile index; one plane never
   loads the stack.
6. **Report metadata as found.** No repairing or rounding of calibration.
7. **Sample files:** only files whose licence allows redistribution may be
   committed or back tests; record each in `docs/samples.md`. The `.vsi` tree
   embeds the original file name and path, so treat metadata dumps like the
   file itself.

## Traps

- The `.vsi` is a valid TIFF holding only a thumbnail; TIFF readers "succeed".
- Which `.ets` coordinate slot is z, channel or time differs between files and
  is declared in the `.vsi`. Tile data is not in coordinate order: always go
  through the tile table.
- A tag number means different things under different parents: address values
  by path.
- Only `frame_t*.ets` files hold pixels.
- The oracles (Bio-Formats, slideio) disagree in places; agreement with one is
  not proof. `docs/format.md` records each disagreement.

## Development

Python 3.10+ with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra test --extra codecs
just check           # ruff, mypy, pytest
just release-check   # before a release: public samples, codecs and slideio must be present
uv run python -m bioio_vsi describe path/to/file.vsi
```

Larger public samples are found through `VSI_PUBLIC_SAMPLES` (default `data/`,
gitignored); their tests skip when absent.

## Documentation

- `docs/format.md` is the format specification, with how each fact was
  established. The reader is only as good as the files seen: a new fact about
  the format needs evidence from a file, and goes here when the code learns it.
- `docs/design.md` records the choices the format leaves open, with reasons.
  Change it together with the code.
- `docs/samples.md` lists public samples, sources and licences.
- Feature requests and open questions go to the issue tracker, not the docs.
