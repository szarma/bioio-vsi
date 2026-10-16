# The cellSens VSI / ETS format, as far as it is known here

The format is undocumented in public. A vendor specification exists (the
Bio-Formats format page mentions versions 1.3 and 1.6 held by OME, not
distributable); it was not seen. Everything below was worked out from files.

Each statement carries how it was established:

- **[obs]** observed directly in files: the public samples and anonymised
  fixtures in `samples.md`, and further unpublished acquisitions from the same
  microscope as the fixtures.
- **[sio]** layout or name taken from slideio's BSD-3 VSI driver
  (`src/slideio/drivers/vsi`, commit `18d7fdbe`), then
  checked against the files.
- **[=sio]**, **[=bf]** the resulting pixels or values agree with slideio 2.10.0
  or with Bio-Formats 8.3.0 output.
- **[inferred]** a reading that fits the files but is not confirmed by anything
  independent.

Bio-Formats source was not read. Tag names are used here only as labels for
numeric ids.

All integers are little-endian.

## 1. Files that make up one acquisition

- `name.vsi` plus a sibling directory `_name_/` holding one sub-directory
  `stackN/` per image, each with one pixel file. [obs]
- The pixel file is `frame_t_0.ets` in cellSens Dimension files and in VS200
  fluorescence stacks, and `frame_t.ets` in VS200 RGB stacks. [obs]
- `N` is the id of the image in the `.vsi` (the second tag of its 2001 node,
  section 3). Ids seen: 1 for the only or first image, 10000 and up for further
  images in slide-scanner files. Confirmed for ids 1, 10000, 10002, 10005 by
  matching the extent stored in each `.ets` against the `.vsi`. [obs]
- `stack10000/` in VS200 files also contains `blob_21_f.meta` (UTF-16 XML) and
  `blob_21_f_Frame#0.ets`. The latter has the ETS layout but 65536 x 1 "tiles"
  and absurd coordinates; it is not an image. Only `frame_t*.ets` holds pixels.
  [obs]
- Not every image in the `.vsi` has a `stackN/`: focus maps live in a TIFF
  directory inside the `.vsi`, and sample masks and focus points have no pixel
  file at all. [obs]
- Whether a long time series is split into `frame_t_1.ets`, ... is unknown; the
  name suggests it and no sample has it.

## 2. The `.vsi` container

- Bytes 0..7 are a classic little-endian TIFF header (`II`, 42, first IFD
  offset). The IFD describes a JPEG thumbnail, at most 512 px on the long side.
  This is why generic TIFF readers "succeed" on a `.vsi`. [obs]
- The metadata tree starts at byte 8 and, in cellSens Dimension files, ends
  exactly at end of file; the thumbnail and its IFD sit *inside* the tree as the
  payload of field 2016 (extended type 10). [obs]

### Tree encoding [sio, obs]

A *volume* is a 24-byte header followed by a chain of fields.

```
volume header  <HHIqII   headerSize (=24), magic (=21321), version,
                         offsetFirstField, flags, unused
field header   <iiiI     fieldType, tag, nextField, dataSize
               [<i       secondTag, present if fieldType bit 27 is set]
```

- The low 28 bits of `flags` give the number of fields. `offsetFirstField` and
  every `nextField` are relative to the start of the volume header; `nextField`
  0 ends the chain.
- `fieldType` bits: 30 = the value is stored in `dataSize` itself (inline
  scalar); 28 = extended field; 27 = a second tag follows; low 24 bits = value
  type or extended type. When bit 30 is set, the four bytes of `dataSize` are
  read as the declared value type for types 5 (signed int32), 6 and 14 and 271
  and 4104 (unsigned int32), and 12 (0 or 1). Other inline types are kept in
  the tree but not decoded; no other inline type has been seen. [obs]
- A volume's declared field count must match its `nextField` chain: the chain
  ends exactly on the last declared field, every offset is visited once in the
  file, and `nextField` always points forward within the volume. [obs]
- Extended types 0, 1 and 2 each hold one nested volume starting right after
  the field header. Type 10 is the embedded TIFF data. For type 0, `dataSize` is
  not a byte count from the payload start; walk by `nextField`.
- Volume versions seen: 1 and 402 everywhere, 403 only in VS200 files.
- Value types seen: 1, 2, 4, 5, 6, 7, 10, 12 (bool), 13 (unit string, UTF-16 without terminator), 14,
  256/257/259/260/261 (int and double vectors and rects), 268, 270, 271, 279,
  4104, 8192 and 8194 (UTF-16 strings), 8195, 8199, 8200 (int vectors).
- **The same tag number means different things under different parents.** Tag
  2003 is the dimension sizes under an image node and display limits under its
  properties; 2018 is the external-file node under a frame and the origin under
  properties. A reader must address values by path, never by a global search for
  a tag number.
- VS200 files contain fields with flag byte 0x28 (tags 2023 and 2024, large byte
  blobs) whose `dataSize` runs past the end of the file in two cases. They hold
  mask and annotation data, not image geometry, and are skipped safely by
  following `nextField`. What bit 29 means is unknown. [obs]

### Layout of the tree [obs]

```
2            document properties: 34 product, 35 version, 21 build, 11 name,
             12 original path, 14/23 times (Unix seconds)
2000[0]      image collection
  2035       [format version, build]         e.g. [196638, 31056]
  2016[0]    embedded TIFF (thumbnail)
  2001[id]   one per image (id = stackN)     see below
  2004, 2011, 2038   collection properties, display settings, XML blob
```

Inside each `2001[id]`:

| What | Path | Notes |
|---|---|---|
| image width, height | `2002[k]/2018/2053` = `[0, 0, W, H]` | on the one 2002 node that has a 2018 child |
| has external file | `2002[k]/2018/20005` | 1 in every image that has a `stackN/` |
| dimension sizes | `2003` | one entry per declared dimension, in slot order; absent for RGB slide images |
| meaning of each slot | `2007[slot]/2008[0]/2023` | 1 = Z, 2 = T, 3 = lambda, 4 = channel, 9 = phase [sio]; only 1, 2, 4 seen |
| channel names | `2007[slot]/2008[c]/2021` and `.../2419` | one 2008 node per channel; see section 5 |
| emission wavelength | `2007[slot]/2008[c]/2417/268435458` | nm |
| z start, z step | `2007[slot]/2008[0]/2012`, `/2013` -> `268435458` | absent when there is one z |
| per-plane z position, time | `2002[k]/2006/2014`, `/2017` -> `268435458` | one 2002 node per plane, ordered with slot 0 fastest |
| pixel size x, y | `2005/2019` | double pair; unit tag `2005/2020` is `10^-6m^1` (micrometres) |
| stage origin | `2005/2018` | |
| image name | `2005/2030` | |
| image type | `2005/2074` | absent in cellSens Dimension files; see section 6 |
| acquisition time | `2005/2015` | Unix seconds |
| camera bit depth | `2005/2043/<n>/120114/100049` | `<n>` is the device slot holding the camera; 8, 12 or 16 seen |
| objective magnification | `2005/2043/<n>/120114/120060` | |

The last `2007[slot]` node of every image is empty. In z/channel/time stacks it
lines up with the pyramid-level coordinate (section 4).

## 3. The `.ets` pixel file

### Main header, 64 bytes [sio, obs]

`<4sIIIQIIQIIQII`: magic `SIS\0`, header size (64), version (3), number of
dimensions `ndim`, offset of the second header (64), its size (228), unused,
offset of the tile table, number of tiles, unused, offset of a free-chunk table,
number of free chunks (0 in every file), unused.

### Second header, 228 bytes [sio for the first ten fields; the rest obs]

| Offset | Field | Values seen |
|---|---|---|
| 0 | magic `ETS\0` | |
| 4 | version | 0x30006 in every file |
| 8 | pixel type | 2 = uint8, 4 = uint16 |
| 12 | samples per tile | 1, or 3 for RGB |
| 16 | colour space | 1 grey, 4 for the RGB JPEG stacks |
| 20 | compression | 0 raw, 2 JPEG, 3 JPEG 2000 |
| 24 | quality | 100 raw, 90 JPEG, 98 or 100 JPEG 2000 |
| 28, 32, 36 | tile width, height, depth | 512 x 512, or the full frame; depth 1 |
| 108 | background value | 65535, 4095, 0, or 0xEEEEEE for RGB |
| 148 | component order | 0xFFFFFFFF grey, 1 for RGB |
| 152 | pyramid flag | 1 exactly in the files that have more than one level |
| 172, 176 | origin offset x, y | 0, or small negatives in stitched cellSens scans: the image starts at stored pixel (-x, -y); see section 4 |
| 184 | number of extents | `ndim - 1`, or 2 for RGB images |
| 188.. | extents | W, H, then the size of each further coordinate slot in order |
| 216, 220 | the negated offset again | |

The extents at 188 equal the `.vsi` values (`2053` and `2003`) in every file.
[obs] Two consequences:

- pairing a `stackN/` with its image can be *checked* rather than assumed;
- an `.ets` whose `.vsi` is lost still gives exact width, height and plane
  counts. Only the meaning of the slots (which is z, which is channel), names
  and calibration are gone.

### Tile table [sio, obs]

At the offset in the main header, `ntiles` records of `4 * (ndim + 5)` bytes:
`<I` (= `ndim`), `ndim` x `<i` coordinates, `<Q` byte offset, `<I` byte length,
`<I` (values 0..7, meaning unknown).

- The table is the last thing in the file in every sample. [obs]
- The tile data is not stored in coordinate order. The first tile on disk is
  z 45 of the fourth channel in `tzc-z-down` and z 176 of the fourth channel
  in `czt-z-up`. Reading planes in file order gives the wrong order. [obs]
- Every image has a complete grid: no missing and no duplicate coordinates on
  any level. [obs]

## 4. From coordinates to pixels

### Coordinate slots

`coordinate = (tile x, tile y, slot 0, slot 1, ..., pyramid level)`.

- Slots 0.. take their meaning from `2007[slot]` in the `.vsi`. **The order
  differs between files from the same microscope and software**: five unpublished
  stacks are (T, Z, C) and five are (C, Z, T) (fixtures `tzc-z-down` and
  `czt-z-up`). VS200 fluorescence is (C) or
  (Z, C). [obs] Reading with the declared order gives pixels identical to
  slideio on every file both could open without JPEG tiles, and to Bio-Formats on the planes
  compared (for stitched scans, after the shift described below). [=sio, =bf]
- The last coordinate is the pyramid level, also when the file has only level
  0. [obs]
- RGB slide images declare no dimensions and have `ndim` 4: one coordinate
  between y and level that is always 0. [obs] So "2 + declared dimensions +
  level" does not always account for every coordinate; an undeclared
  coordinate is accepted only if it is 0 in every tile.
- `ndim` seen: 4 (RGB; or channels only), 5 (z and channels), 6 (t, z, c).

### Tiles

- Stitched and slide-scanner images use 512 x 512 tiles, edge tiles padded to
  full size. Single-field stacks store one full-frame tile per plane. [obs]
- Raw tiles are `tile height x width x samples` in the pixel type, little-endian.
  [=sio, =bf]
- JPEG tiles decode to 8-bit, 3 samples, already in the order Bio-Formats
  returns; no channel reversal is needed although the colour-space code is 4.
  libjpeg-turbo (via imagecodecs) matches Bio-Formats exactly; slideio differs
  from both by up to 15 grey levels, mean 0.4. JPEG pixels depend on the
  decoder. [obs, =bf]
- JPEG 2000 tiles are bare codestreams, 16-bit, one sample. Quality 98 files are
  lossy. Decoding with OpenJPEG (via imagecodecs) matches slideio exactly on
  one quality-100 and one quality-98 file, and Bio-Formats to within 1 grey
  level on both. [=sio, =bf]

### Pyramid

- Level L+1 is the 2 x 2 mean of level L, rounded half up, exact for every
  pixel that has four parents. [obs, unpublished stitched scans]
- When a level has an odd width or height, the stored tile data has one more
  column or row of partially covered pixels. slideio reports level sizes as
  `floor(size / 2^L)` and drops them; Bio-Formats reports `ceil` and returns
  them. Those edge pixels are not the mean of real data and darken with each
  level. [obs; the two oracles disagree]
- Whether that extra column or row ever adds a tile is not known. In every
  lower level seen, the floor size and the rounded-up size need the same
  number of tiles. A file where they differ (a level whose floor size is an
  exact multiple of the tile size) has not been seen. [obs]
- The pyramid stops at the first level that fits in a single tile. [obs]

### Origin offset in stitched scans

All four unpublished stitched stacks have a non-zero offset in the `.ets`
header: (-3, -1), (-4, -1), (-1, -1) and (-15, -1); the last two are the
fixtures `stitched-origin-1` and `stitched-origin-15`. In the three whose edge tiles were
examined, the tile data holds `W + |dx| + 1` by
`H + |dy| + 1` real pixels, more than the declared `W x H`. [obs]

The image of declared size starts at stored level-0 pixel `(-origin_x,
-origin_y)`: `image[y, x] = stored[y + |origin_y|, x + |origin_x|]`. The tile
grid therefore has to cover `W + |origin_x|` by `H + |origin_y|` pixels.

- [obs] Two stitched acquisitions exported by cellSens itself to OME-TIFF
  (uint16, 4 channels; one 2D with origin (-15, -1), one z stack with origin
  (-4, -1)) equal the stored level-0 grid starting at (-origin_x, -origin_y),
  exactly, in every channel over the whole image.
- This is Bio-Formats' placement. slideio returns the window starting at
  tile-grid pixel (0, 0), which is off by the offset. [obs]
- Lower pyramid levels are 2x2 means of the stored grid from (0, 0), so their
  offset is origin / 2**L, generally not a whole pixel. The reader exposes
  level 0 only for these images (`design.md`). [obs]
- Positive offsets have never been seen and are refused.

Single-field stacks and all slide-scanner files have offset 0.

## 5. Metadata values

- **Pixel size**: `2005/2019`, micrometres. Equal to Bio-Formats' and slideio's
  values on every file. [=sio, =bf] The unit field says `10^-6m^1`.
- **Units** are stored as type-13 strings: the unit tag `268435456` next to a
  property's value, and `2005/2020` for the pixel size. They are UTF-16 like
  the other strings although the type differs, and read as a power of ten, a
  unit and an exponent: `10^-6m^1` micrometres, `10^-9m^1` nanometres,
  `10^-3s^1` milliseconds, `10^0s^-1` hertz, `10^0%^1` percent. Read as 8-bit
  text, every one of them is the string `'1'`. The values the reader uses carry
  one unit each in every file seen: z start, z step and per-plane z
  `10^-6m^1`, per-plane time `10^-3s^1`, emission wavelength `10^-9m^1`;
  any other unit is refused. [obs]
- **Z step**: `2013` value, micrometres, signed. Two unpublished stacks (one is
  `tzc-z-down`) and the four public deconvolution files step downwards (negative). slideio reports the negative value;
  Bio-Formats reports no z size at all for the two it could open. [obs; oracles
  disagree]
  The per-plane position (`2014`, or start + k x step) is consistent in both.
  No file seen stores NaN in a per-plane z or time value (`2014`, `2017` ->
  `268435458`); a NaN there is kept as an unknown value for that plane, and
  infinities are refused. [obs]
- **Time**: there is no time-step field in the one time series available. Each
  plane has a time stamp (`2002[k]/2006/2017`, milliseconds since start
  [inferred from magnitudes]). Steps between time points range from 14.84 to
  15.27 s. Neither oracle reports a time increment for this file. [obs]
- **Channel names**: two fields, `2021` on the channel node and `2419`. They are
  equal in every file but one. In the public time series `2021` is the fluorophore
  (`Cy5`, `GFP ...`) and `2419` is the same experiment label for both
  channels; both oracles return the `2419` value, so they report two channels
  with identical names. Which one cellSens shows is not known. [obs]
- **Bit depth**: the camera value (`100049`) is 8, 12 or 16. Pixels are stored
  in 16 bits either way. Bio-Formats reports 16 significant bits for the 12-bit
  files; slideio reports 12 on the one such file checked. [obs]
- **Exposure time** per channel exists (Bio-Formats reports it); its tag was not
  located here.

## 6. Images in one file, and their types

`2005/2074` values seen, with the name each carries in the file:

| Value | Name in file | Pixel data |
|---|---|---|
| absent | the only image in cellSens Dimension files | `stack1/` |
| 0 | main scan, e.g. `20x_BF_01` | `stackN/` |
| 1 | `Overview` | `stack10000/` |
| 4 | `20x FocusMap` | a TIFF directory inside the `.vsi`; field 2016 carries a second tag (4 or 6) that looks like its IFD number [inferred] |
| 64 | main scan with a z dimension (slideio calls it "EFI stack") | `stackN/` |
| 256 | `Label` (slideio: macro image) | `stack1/` |
| 1024 | `FocusPoints` | none |
| 8388608 | `Sample Mask` | none; not in slideio's list |

A slide may be scanned more than once: the brightfield and 2D fluorescence
samples each contain two main images of the same size (`..._01`, `..._02`).

## 7. Versions

The first value of `2035` reads as `major << 16 | minor`.

| Product (tag 34, 35) | Build | Format | Files | Differences from the others |
|---|---|---|---|---|
| cellSens Dimension 3.1 | 21199 | 3.22 | 1 public | none in structure; one extra tag (175122) |
| VS200 ASW 3.4.1 | 26606 | 3.26 | 1 public | several images per file; volume version 403; flag 0x28 blobs |
| VS200 ASW 4.1 | 27540, 27544 | 3.27 | 3 public | as 3.4.1, plus four new tags |
| cellSens Dimension 4.3 | 31056 | 3.30 | 10 unpublished, 2 public | a block of extra tags numbered 91 and up |
| not recorded | 34364 | 3.32 | 1 public | document properties absent; distinct `2021`/`2419` names; eight tags not seen elsewhere |
| cellSens Dimension 4.3 (resaved by a newer build) | 35701 | 3.33 | 2 public | none in structure |

- The `.ets` version fields are identical in every file (3 and 0x30006).
- Across six format versions, every value in the section 2 table sits at the
  same path. What changes between versions is which *other* tags exist and, for
  the slide scanner, how many images there are. [obs]
- The product named in tag 34/35 is not the writing software's version: two
  files say "4.3 build 31056" in the document properties while `2035` records
  build 35701. `2035` is the more specific stamp. [obs]

## 8. Not determined

- The meaning of `268435459` (268435466 on every value) next to the unit.
- The meaning of the last word of a tile record and of fieldType bit 29.
- Why RGB images carry an extra constant coordinate.
- Multi-file time series, lambda and phase dimensions, compression codes other
  than 0, 2, 3, and `.vsi` files with no `.ets`.
- Anything about files written before cellSens 3.1.
