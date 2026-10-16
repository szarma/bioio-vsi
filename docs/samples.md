# Sample files

Public VSI files found on 2026-10-13, where they came from, and what each may be
used for. Downloaded copies live in `data/` (gitignored). Anonymised fixtures, committed
for the tests, are listed at the end.

"CI" below means: the licence allows redistribution, so the file may back tests
that run in public CI.

## Summary

| Set | Source | Licence | Size | Writer | Content | CI |
|---|---|---|---|---|---|---|
| `zenodo-6094961` | [10.5281/zenodo.6094961](https://doi.org/10.5281/zenodo.6094961), N. Ehrenfeuchter | CC BY 4.0 | 0.2 MB | cellSens Dimension 3.1 (build 21199) | 2 channels x 11 z, 1645 x 1682, uint16, JPEG 2000, one full-frame tile per plane | yes, small enough to commit |
| `zenodo-17590655` | [10.5281/zenodo.17590655](https://doi.org/10.5281/zenodo.17590655), A. Gros; mirrored by OME | CC BY 4.0 | 20 MB | not recorded in the file (format 3.32, build 34364) | 5 time points x 18 z x 2 channels, 260 x 216, uint16 raw, 12 significant bits | yes |
| `zenodo-18163893` | [10.5281/zenodo.18163893](https://doi.org/10.5281/zenodo.18163893), B. Hyklova and I. Svecova; mirrored by OME | CC BY 4.0 | 1.09 GB (four stacks of 109 to 435 MB) | cellSens Dimension 4.3 (builds 31056 and 35701) | 41 z, one channel, 1152 or 2304 px square, uint16 raw; two are deconvolved copies | yes by licence, too large to fetch per run |
| `zenodo-8161864` | [10.5281/zenodo.8161864](https://doi.org/10.5281/zenodo.8161864), J. Ross | CC BY 4.0 | 1.41 GB zip | VS200 ASW 3.4.1 (build 26606) | one H&E slide at 20x, 67170 x 58321, plus label, overview, focus map | yes by licence, too large |
| `ome-cellsens/evident` | <https://downloads.openmicroscopy.org/images/CellSens/evident/> | none stated | 0.93 GB | VS200 ASW 4.1 (builds 27540, 27544) | three slides: brightfield RGB JPEG; 4-channel fluorescence JPEG 2000; the same with 5 z | no |

The OME directory `https://downloads.openmicroscopy.org/images/CellSens/` also
mirrors the second and third Zenodo records with a `COPYING` file each. The
`evident` folder has no licence file and the Bio-Formats format page states none,
so those three slides are for local study only until someone asks OME or Evident.

The pixels of `zenodo-18163893`, `zenodo-8161864` and `Image_V4.1_FL_3D` were
not read: only their `.vsi` and the `.ets` headers and tile tables, so they
back structural facts but no pixel comparison.

## Coverage, and what is missing

Covered by at least one public file: cellSens Dimension 3.1 and 4.3; VS200 ASW
3.4.1 and 4.1; raw, JPEG and JPEG 2000 tiles; uint8 RGB and uint16 grey; z
stacks; one short time series; multi-image slide-scanner files with label,
overview and focus map.

Not found anywhere public:

- An uncompressed, tiled, pyramidal fluorescence stack with real pixels. The
  `stitched-origin-*` fixtures below carry the structure of two such scans,
  with synthetic pixels (the origin rule was settled against OME-TIFF exports
  of other stitched scans, `format.md`, section 4).
- Any file older than cellSens 3.1 or from the VS120 scanner.
- Lossless JPEG, PNG or BMP tiles (compression codes 5, 8, 9 in slideio's enum).
- A `.vsi` that stores its pixels inside the TIFF with no `.ets` at all
  (reported in Bio-Formats issue 4015; its sample is not public).
- Lambda (spectral) or phase dimensions, and a time series longer than five
  points or split over several `.ets` files.
- Any redistributable VS200 file small enough for CI.

Also seen but not used: Zenodo study datasets that contain `.vsi` files inside
zips (for example records 10908572, 5222122 and 6523649). They would widen
version coverage but are 17 to 38 GB each. Only the first 25 Zenodo hits for
"olympus vsi" and for "cellSens vsi ets" were read. IDR and the BioImage Archive
were not searched.

## Anonymised fixtures

`bioio_vsi/tests/resources/anonymised/` holds five fixtures made with
`python -m bioio_vsi anonymize` from unpublished acquisitions of one cellSens
Dimension 4.3 microscope (build 31056, uncompressed uint16 fluorescence, four
channels). Each is the `.vsi` with every string replaced by a same-length
placeholder, every date by a fixed value, the thumbnail and specimen
preview zeroed, and an `.ets.skel` with the pixel
file's headers and tile table but no pixels. `python -m bioio_vsi materialize`
turns one into a readable `.vsi` and sparse `.ets` with a synthetic picture.
Numbers are synthetic (`design.md`): only format facts such as dimension
sizes, slot order, frame indices, tile table, pyramid levels and stitch
origin offsets are as in the originals. `expected.json` records the values
the reader gives on the synthetic fixtures; `test_anonymised_fixtures.py`
checks the materialised fixtures against them and fails if a measured number
comes back. Licence: BSD-3-Clause, as the
repository.

| Fixture | Image | Shows |
|---|---|---|
| `tzc-z-down` | 2304 x 2304, 90 z x 4 channels, slots (T, Z, C) | z decreasing with plane index; tiles stored out of coordinate order |
| `czt-z-up` | 2304 x 2304, 202 z x 4 channels, slots (C, Z, T) | the other slot order from the same microscope; z increasing |
| `single-plane` | 2304 x 2304, 4 channels | no z: start and step absent |
| `stitched-origin-1` | 8177 x 4260, 45 z x 4 channels, 5 levels | stitched scan, `.ets` origin offset (-1, -1): level 0 only |
| `stitched-origin-15` | 10122 x 6219, 4 channels, 6 levels | stitched scan, origin offset (-15, -1): level 0 only |
