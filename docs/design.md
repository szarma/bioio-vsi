# Design choices

Where the format leaves a choice open, or where existing readers disagree, this
reader does the following. Each entry says why and what the alternative was,
so a change can be argued against the same reasons. How the underlying facts
were established is in `format.md`.

## What is refused

- **Files with more than one image** (slide scanners: scans, overview, label)
  are refused. Reading them as scenes is planned, not designed yet.
- **A positive origin offset** in the `.ets` header is refused. Only zero and
  negative offsets have been seen, and a positive one would mean something the
  placement rule (`format.md`, section 4) does not cover.
- **JPEG tiles** are refused; raw and JPEG 2000 are read. JPEG has only been
  seen in slide-scanner files, which are refused anyway, and decoded JPEG
  pixels depend on the decoder library.
- **Pyramids with a level of zero width or height** are refused rather than
  advertising levels that cannot be read.
- **Remote files.** Only local paths are read; a non-local fsspec filesystem
  raises an error instead of downloading.

Accepting more files later is compatible. Refusing files that used to open is
not, which is why the accepting choices below are made conservatively.

## What is accepted with a notice

- **Unseen format versions** are read with a warning when every structural
  check passes. Software updates that make readers refuse new files are a
  common complaint; the reader checks each file's structure rather than its
  version number, and lists the tested versions in one constant.
- **Stitched scans expose only level 0.** The image starts at stored-grid pixel
  (-origin_x, -origin_y), and lower levels are means of the stored grid from
  (0, 0), so their offset is origin / 2**level, usually not a whole pixel.
  Resampling or rounding would invent pixels, so an image with a
  non-zero origin reports one level, through `open_vsi`, `open_ets` and bioio.
  Exposing more levels later stays compatible. Orphan `.ets` files of stitched
  scans return the same shifted image; `origin_offset` is reported as stored.
- **Pyramid level storage.** A lower level may be stored as the grid of tiles
  for the floor-sized level or for the rounded-up size; both are accepted and
  the floor-sized region is returned. No file seen shows which cellSens writes
  when the two differ.

## How values are reported

- **Dimension order through bioio is TCZYX**, bioio's default. `BioImage`
  presents TCZYX whatever a reader returns; for direct `Reader` users this
  gives the same order for every file, where the native order differs between
  files from the same microscope.
- **Z planes stay in acquisition order and bioio's Z spacing is the absolute
  step.** In many cellSens stacks z decreases with plane index. Keeping stored
  order keeps plane indices identical to cellSens, Fiji, Bio-Formats and
  slideio; a negative spacing breaks downstream tools. The signed step and
  each plane's position are available from `open_vsi`.
- **Channel names come from the per-channel field (tag 2021)**, usually the
  dye. Bio-Formats and slideio use tag 2419, which in one known file gives
  both channels the same name, so channels could not be selected by name.
  Tag 2419 stays in the raw metadata.
- **`time_interval` is empty unless every time step is equal to within
  0.1 %.** A mean would present an irregular series as regular. Per-plane
  time stamps are always available. The tolerance is a choice, not a property
  of the format.
- **Pyramid level sizes are `floor(size / 2^level)`**, dropping a partly
  covered edge row or column (slideio's convention; Bio-Formats rounds up).
  At lower levels, bioio's X and Y pixel sizes are multiplied by `2^level`.
- **Units are micrometres, milliseconds and nanometres (emission) as stored.**
  Each value's unit string is compared in full with the one expected for that
  value, and any other unit is refused rather than converted: no file seen
  stores another, so a conversion would be untested.
- **Unknown values stay unknown.** Missing values are `None`, never a guess. A
  NaN stored for one plane's z or time stays NaN. When a z stack stores no
  per-plane positions, they are computed from the declared start and step.
- **Metadata is reported as stored.** Pixel size and z step are not repaired,
  rounded or judged.

## Anonymised structure fixtures

`anonymize` turns an acquisition that cannot be shared into a fixture that
reproduces a reading problem; `materialize` turns the fixture back into a
readable file with synthetic pixels.

- **Structure is kept, content is not.** Headers, the metadata tree's layout,
  dimension sizes, slot order and the tile table stay, because they decide how
  a file is read. Pixels, the thumbnail and specimen preview, text, dates and
  any payload the reader does not interpret are removed.
- **Anything not known to be safe is removed.** Keeping is by allowlist, so a
  field nobody has looked at is zeroed rather than passed on.
- **Numbers are synthetic by default.** Pixel size, positions, times and
  wavelengths could let someone recognise the acquisition, so the values the
  reader reports get fixed values of the same shape (z direction and X:Y ratio
  kept), and every other number is zeroed. `--keep-numbers` keeps them, for
  local work only.
- **`materialize` fills only the first plane by default**, at levels up to
  64 MiB, so that a large scan stays small on disk; `planes="all"` fills all.

The exact rules are in `_anonymize.py`, `_synthetic_numbers.py` and
`_materialize.py`.

## Packaging

- Required dependencies are `bioio-base` and `numpy`; JPEG 2000 decoding is
  the `codecs` extra (imagecodecs). The core (`open_vsi`, `open_ets`,
  `describe`) imports neither bioio nor dask.
- Names follow the bioio plugin convention: distribution `bioio-vsi`, import
  `bioio_vsi`, entry point `bioio-vsi`. Python 3.10 and newer. BSD-3-Clause,
  as bioio and slideio.
- One small CC BY sample and five anonymised fixtures (`samples.md`) are
  committed for the tests; neither is in the wheel. Larger public samples are
  found through `$VSI_PUBLIC_SAMPLES` and their tests skip when absent;
  `just release-check` makes those skips fail, so a release is never tested
  without them.
