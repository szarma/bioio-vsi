# bioio-vsi

A small, JVM-free reader for Olympus/Evident cellSens VSI microscopy images
(a `.vsi` metadata file plus sibling `.ets` pixel files).

The format has no public specification (the vendor's exists but is not
distributed). This reader is reverse engineered
from sample files, with the tree layout taken from slideio's BSD-3 VSI driver
(`src/slideio/drivers/vsi`) and checked against those files, and with
Bio-Formats and slideio output as test oracles. It reads what it has seen and
refuses the rest; [`docs/format.md`](https://github.com/szarma/bioio-vsi/blob/main/docs/format.md) records what is known and
how each fact was established.

## Install

```bash
pip install bioio-vsi
pip install "bioio-vsi[codecs]"   # JPEG 2000 tiles (imagecodecs)

# bioio plugin (registers on ``bioio.readers``)
pip install bioio bioio-vsi
pip install "bioio-vsi[codecs]"   # if your files use JPEG 2000
```

Developers use [uv](https://docs.astral.sh/uv/); see [AGENTS.md](https://github.com/szarma/bioio-vsi/blob/main/AGENTS.md).

## Supported and refused

**Reads** single-image cellSens files: uint8/uint16, raw or JPEG 2000 tiles,
optional pyramids, any declared order of time, z, and channel.

**Refuses** (with a named error): missing pixel file, multiple images per
``.vsi``, unsupported compression or pixel type, size mismatches, incomplete
tile grids, lambda/phase dimensions, truncated metadata fields, and unknown
metadata units.

Why each of these is refused, and how values such as z spacing and channel
names are reported, is in [docs/design.md](https://github.com/szarma/bioio-vsi/blob/main/docs/design.md).

## Usage

Standalone API:

```python
from bioio_vsi import open_vsi, open_ets, describe

img = open_vsi("path/to/file.vsi")
plane = img.read_plane(t=0, c=0, z=0)

orphan = open_ets("path/to/frame_t_0.ets")  # geometry only, no axis names
print(describe("path/to/file.vsi"))
```

Through bioio:

```python
from bioio import BioImage

image = BioImage("path/to/file.vsi")
stack = image.get_image_dask_data("TCZYX")  # dimension order TCZYX
plane = stack[0, 0, 0].compute()
print(image.physical_pixel_sizes, image.channel_names)
```

Z planes are in acquisition order, and in many cellSens stacks z decreases with
plane index. bioio reports the z spacing as a positive distance; the signed step
and each plane's position are in `open_vsi(...).z_step_um` and
`plane_z_positions_um`. Channel names are the per-channel names cellSens stores
(usually the dye), which can differ from the names Bio-Formats and slideio report.

```bash
python -m bioio_vsi describe path/to/file.vsi   # what the reader sees in a file
```

## Reporting a file that fails to open

Open an issue with the output of

```bash
python -m bioio_vsi describe --redact path/to/file.vsi
```

which leaves out paths, file names and image or channel names (also in error
lines).

If that is not enough and the acquisition cannot be shared, you can send a
**structure fixture** instead. It keeps the `.ets` headers, tile table and the
layout of the metadata, and removes pixels, text, dates and previews; measured
values such as pixel size, positions and times are replaced by synthetic ones:

```bash
python -m bioio_vsi anonymize path/to/your.vsi /tmp/fixture-out
```

Upload the resulting ``anon.vsi`` and ``_anon_/stack*/`` ``.skel`` files (not
your original ``.vsi`` tree). Maintainers can turn the fixture back into a
viewable file locally with ``python -m bioio_vsi materialize FIXTURE OUTDIR``.
See [docs/design.md](https://github.com/szarma/bioio-vsi/blob/main/docs/design.md) for what is kept and stripped.

## Sample files wanted

The reader only knows the files it has seen. Files under a licence that allows
redistribution help most, especially older cellSens versions, slide scanners
(VS120, VS200), stitched scans and long time series; see the
[issue tracker](https://github.com/szarma/bioio-vsi/issues).
