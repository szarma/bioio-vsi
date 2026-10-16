"""bioio ``Reader`` plugin for cellSens ``.vsi`` files.

Dask chunking
-------------
* **Full-frame planes** (one tile covers the image at the current resolution
  level): one chunk per ``(T, C, Z)`` plane, shape ``(1, 1, 1, Y, X)``.
* **Tiled planes**: tile-aligned blocks of ``2 x 2`` ETS tiles per chunk within
  each plane (edge blocks may be smaller), concatenated to the full ``Y, X``
  extent.

Each dask task receives a precomputed :class:`~bioio_vsi._stack.TileReadSpec`
(or a list of specs for a tile block), opens the pixel file, reads the payload,
and closes the file. The VSI tree and ETS tile table are parsed once when the
dask graph is built, not per task.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, cast

import dask.array as da
import numpy as np
import xarray as xr
from bioio_base import constants, dimensions, exceptions, reader, types
from bioio_base.io import pathlike_to_fs
from dask import delayed
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from bioio_vsi import open_vsi
from bioio_vsi._paths import resolve_path
from bioio_vsi._stack import TileReadSpec, level_shape, read_tile_from_spec
from bioio_vsi._tree import VsiTree, collection_root, load_tree
from bioio_vsi._vsi import ImageVolumeMeta, load_vsi
from bioio_vsi.errors import InvalidStructureError, TruncatedFileError, VsiError

_DIMS = dimensions.DEFAULT_DIMENSION_ORDER_LIST
_TILE_BLOCK = 2


def _z_step_um(meta: ImageVolumeMeta) -> float | None:
    for slot in meta.slots:
        if slot.kind == "Z":
            return slot.z_step_um
    return None


def _read_tile_block(
    specs: tuple[TileReadSpec, ...],
    placements: tuple[tuple[int, int, int, int], ...],
    out_shape: tuple[int, int],
    dtype: np.dtype,
) -> np.ndarray:
    out = np.zeros(out_shape, dtype=dtype)
    for spec, (y0, x0, h, w) in zip(specs, placements, strict=True):
        tile = read_tile_from_spec(spec)
        out[y0 : y0 + h, x0 : x0 + w] = tile[:h, :w]
    return out


def _read_plane_from_specs(
    specs: tuple[TileReadSpec, ...],
    placements: tuple[tuple[int, int, int, int], ...],
    out_shape: tuple[int, int],
    dtype: np.dtype,
) -> np.ndarray:
    if len(specs) == 1:
        tile = read_tile_from_spec(specs[0])
        return tile[: out_shape[0], : out_shape[1]]
    return _read_tile_block(specs, placements, out_shape, dtype)


def _tile_coord(img: Any, level: int, t: int, c: int, z: int, ty: int, tx: int) -> tuple[int, ...]:
    nd = img._index.header.ndim
    layout = img._layout
    pos = {"t": t, "c": c, "z": z}
    full = [0] * nd
    full[0], full[1], full[-1] = tx, ty, level
    for axis, slot in layout.slot_by_axis.items():
        full[slot] = pos[axis]
    return tuple(full)


def _dask_plane_tiled(img: Any, level: int, t: int, c: int, z: int, dtype: np.dtype) -> da.Array:
    """One plane built from ``_TILE_BLOCK`` x ``_TILE_BLOCK`` tile-aligned dask chunks."""
    ly, lx = level_shape(img._meta.height, img._meta.width, level)
    th, tw = img._th, img._tw
    ny = (ly + th - 1) // th
    nx = (lx + tw - 1) // tw
    row_arrays: list[da.Array] = []
    for ty0 in range(0, ny, _TILE_BLOCK):
        row_blocks: list[da.Array] = []
        y_row_end = min((ty0 + _TILE_BLOCK) * th, ly)
        for tx0 in range(0, nx, _TILE_BLOCK):
            specs: list[TileReadSpec] = []
            placements: list[tuple[int, int, int, int]] = []
            x_block_end = min((tx0 + _TILE_BLOCK) * tw, lx)
            y_block_end = y_row_end
            y_base = ty0 * th
            x_base = tx0 * tw
            for ty in range(ty0, min(ty0 + _TILE_BLOCK, ny)):
                for tx in range(tx0, min(tx0 + _TILE_BLOCK, nx)):
                    y_off = ty * th
                    x_off = tx * tw
                    h = min(th, ly - y_off)
                    w = min(tw, lx - x_off)
                    specs.append(img.tile_read_spec(_tile_coord(img, level, t, c, z, ty, tx)))
                    placements.append((y_off - y_base, x_off - x_base, h, w))
            block_h = y_block_end - y_base
            block_w = x_block_end - x_base
            row_blocks.append(
                da.from_delayed(
                    delayed(_read_tile_block)(
                        tuple(specs),
                        tuple(placements),
                        (block_h, block_w),
                        dtype,
                    ),
                    shape=(block_h, block_w),
                    dtype=dtype,
                )
            )
        row_arrays.append(da.concatenate(row_blocks, axis=1))
    return cast(da.Array, da.concatenate(row_arrays, axis=0))


def _dask_plane_regions(img: Any, level: int, t: int, c: int, z: int, dtype: np.dtype) -> da.Array:
    """One plane of a shifted (stitched, non-zero origin) image, in region chunks.

    Image tiles do not line up with stored tiles, so each chunk is a ``read_region`` call.
    """
    ly, lx = level_shape(img._meta.height, img._meta.width, level)
    bh, bw = _TILE_BLOCK * img._th, _TILE_BLOCK * img._tw
    rows: list[da.Array] = []
    for y0 in range(0, ly, bh):
        h = min(bh, ly - y0)
        blocks = []
        for x0 in range(0, lx, bw):
            w = min(bw, lx - x0)
            blocks.append(
                da.from_delayed(
                    delayed(img.read_region)(y0, x0, h, w, t=t, c=c, z=z, level=level),
                    shape=(h, w),
                    dtype=dtype,
                )
            )
        rows.append(da.concatenate(blocks, axis=1))
    return cast(da.Array, da.concatenate(rows, axis=0))


def _build_dask_stack(
    vsi_path: str,
    level: int,
    shape_tcz: tuple[int, int, int],
    y: int,
    x: int,
    dtype: np.dtype,
) -> da.Array:
    img = open_vsi(vsi_path)
    t_size, c_size, z_size = shape_tcz
    shifted = img.is_shifted
    single = not shifted and img._th >= y and img._tw >= x

    planes: list[da.Array] = []
    for ti in range(t_size):
        for ci in range(c_size):
            for zi in range(z_size):
                if single:
                    coords = img.plane_tile_coords(ti, ci, zi, level)
                    spec = img.tile_read_spec(coords[0])
                    plane_arr = da.from_delayed(
                        delayed(_read_plane_from_specs)(
                            (spec,),
                            ((0, 0, y, x),),
                            (y, x),
                            dtype,
                        ),
                        shape=(y, x),
                        dtype=dtype,
                    )
                elif shifted:
                    plane_arr = _dask_plane_regions(img, level, ti, ci, zi, dtype)
                else:
                    plane_arr = _dask_plane_tiled(img, level, ti, ci, zi, dtype)
                planes.append(plane_arr)
    return cast(
        da.Array,
        da.stack(planes).reshape(t_size, c_size, z_size, y, x),
    )


class Reader(reader.Reader):
    """bioio reader for single-image cellSens VSI files."""

    NAME = "bioio-vsi"

    _meta: ImageVolumeMeta | None
    _scenes: tuple[str, ...] | None

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        if not isinstance(fs, LocalFileSystem):
            raise exceptions.UnsupportedFileFormatError(
                "bioio-vsi",
                path,
                "only local paths are supported",
            )
        with fs.open(path, "rb") as fh:
            head = fh.read(8)
        if len(head) < 8 or head[:4] != b"II*\x00":
            raise exceptions.UnsupportedFileFormatError(
                "bioio-vsi",
                path,
                "not a little-endian TIFF/VSI header",
            )
        try:
            tree = load_tree(path)
            collection_root(tree)
        except (InvalidStructureError, TruncatedFileError, VsiError) as exc:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-vsi",
                path,
                str(exc),
            ) from exc
        return True

    def __init__(self, image: types.PathLike, **kwargs: Any) -> None:
        fs_kwargs = kwargs.get("fs_kwargs", {})
        self._fs, self._path = pathlike_to_fs(
            image,
            enforce_exists=True,
            fs_kwargs=fs_kwargs,
        )
        if not isinstance(self._fs, LocalFileSystem):
            raise TypeError(
                f"bioio-vsi only supports local filesystem paths; got {type(self._fs).__name__}"
            )
        self._path = resolve_path(self._path)
        self._meta = None
        self._scenes = None
        self._physical_pixel_sizes = None
        self._is_supported_image(self._fs, self._path)

    def _ensure_meta(self) -> ImageVolumeMeta:
        if self._meta is None:
            self._meta = load_vsi(self._path)
        return self._meta

    @property
    def scenes(self) -> tuple[str, ...]:
        if self._scenes is None:
            meta = self._ensure_meta()
            label = meta.name or f"stack{meta.stack_id}"
            self._scenes = (label,)
        return self._scenes

    @property
    def resolution_levels(self) -> tuple[int, ...]:
        img = open_vsi(self._path)
        return tuple(range(img.n_levels))

    @property
    def metadata(self) -> Any:
        if self._metadata is None:
            self._metadata = self._ensure_meta().tree
        return self._metadata

    @property
    def physical_pixel_sizes(self) -> types.PhysicalPixelSizes:
        if self._physical_pixel_sizes is not None:
            return self._physical_pixel_sizes
        meta = self._ensure_meta()
        scale = meta.pixel_size_um
        z_step = _z_step_um(meta)
        x_um = scale.x if scale else None
        y_um = scale.y if scale else None
        z = abs(z_step) if z_step is not None else None
        level_scale = 2**self._current_resolution_level
        self._physical_pixel_sizes = types.PhysicalPixelSizes(
            Z=z,
            Y=y_um * level_scale if y_um is not None else None,
            X=x_um * level_scale if x_um is not None else None,
        )
        return self._physical_pixel_sizes

    @property
    def time_interval(self) -> types.TimeInterval:
        meta = self._ensure_meta()
        times = meta.plane_time_ms
        if times is None or times.shape[0] < 2:
            return None
        steps = times[1:, :, :] - times[:-1, :, :]
        if np.any(np.isnan(steps)):
            return None
        flat = steps.reshape(-1)
        if flat.size == 0:
            return None
        ref = float(np.median(flat))
        if ref == 0.0:
            return None
        if np.any(np.abs(flat - ref) / abs(ref) > 0.001):
            return None
        return timedelta(milliseconds=ref)

    def _level_shape_tcz_yx(self) -> tuple[tuple[int, int, int], int, int, np.dtype]:
        img = open_vsi(self._path)
        level = self._current_resolution_level
        t, c, z, y, x = img.level_shape(level)
        return (t, c, z), y, x, img.dtype

    @staticmethod
    def _values_uniform_along_axes(arr: np.ndarray, fixed_axes: tuple[int, ...]) -> bool:
        """True when ``arr`` is constant along every axis not listed in ``fixed_axes``."""
        for axis in range(arr.ndim):
            if axis in fixed_axes:
                continue
            if arr.shape[axis] <= 1:
                continue
            ref = np.take(arr, indices=0, axis=axis)
            for i in range(1, arr.shape[axis]):
                other = np.take(arr, indices=i, axis=axis)
                if not np.array_equal(other, ref, equal_nan=True):
                    return False
        return True

    def _coords(self, meta: ImageVolumeMeta, c_size: int) -> dict[str, Any]:
        channel_labels: list[str] = []
        for slot in meta.slots:
            if slot.kind == "C":
                channel_labels = [
                    ch.name or f"C{ch.index}"
                    for ch in sorted(slot.channels, key=lambda ch: ch.index)
                ]
                break
        if len(channel_labels) != c_size:
            channel_labels = [f"C{i}" for i in range(c_size)]
        coords: dict[str, Any] = {
            dimensions.DimensionNames.Channel: channel_labels,
        }
        zpos = meta.plane_z_um
        if (
            zpos is not None
            and zpos.shape[2] > 1
            and self._values_uniform_along_axes(zpos, fixed_axes=(2,))
        ):
            coords[dimensions.DimensionNames.SpatialZ] = xr.DataArray(
                zpos[0, 0, :],
                dims=("Z",),
                attrs={"units": "micrometer"},
            )
        times = meta.plane_time_ms
        if (
            times is not None
            and times.shape[0] > 1
            and self._values_uniform_along_axes(times, fixed_axes=(0,))
        ):
            coords[dimensions.DimensionNames.Time] = xr.DataArray(
                times[:, 0, 0],
                dims=("T",),
                attrs={"units": "millisecond"},
            )
        return coords

    def _read_delayed(self) -> xr.DataArray:
        meta = self._ensure_meta()
        shape_tcz, y, x, dtype = self._level_shape_tcz_yx()
        level = self._current_resolution_level
        stack = _build_dask_stack(self._path, level, shape_tcz, y, x, dtype)
        coords = self._coords(meta, shape_tcz[1])
        tree: VsiTree = meta.tree
        return xr.DataArray(
            stack,
            dims=_DIMS,
            coords=coords,
            attrs={
                constants.METADATA_UNPROCESSED: tree,
                constants.METADATA_PROCESSED: {
                    "stack_id": meta.stack_id,
                    "format_version": meta.format_version,
                    "pixel_size_um": meta.pixel_size_um,
                    "z_step_um": _z_step_um(meta),
                    "plane_time_ms": meta.plane_time_ms,
                    "plane_z_um": meta.plane_z_um,
                },
            },
        )

    def _read_immediate(self) -> xr.DataArray:
        meta = self._ensure_meta()
        img = open_vsi(self._path)
        level = self._current_resolution_level
        t, c, z, y, x = img.level_shape(level)
        volume = np.zeros((t, c, z, y, x), dtype=img.dtype)
        for ti in range(t):
            for ci in range(c):
                for zi in range(z):
                    volume[ti, ci, zi] = img.read_plane(t=ti, c=ci, z=zi, level=level)
        coords = self._coords(meta, c)
        tree = meta.tree
        return xr.DataArray(
            volume,
            dims=_DIMS,
            coords=coords,
            attrs={
                constants.METADATA_UNPROCESSED: tree,
                constants.METADATA_PROCESSED: {
                    "stack_id": meta.stack_id,
                    "format_version": meta.format_version,
                    "plane_time_ms": meta.plane_time_ms,
                    "plane_z_um": meta.plane_z_um,
                },
            },
        )

    def __getstate__(self) -> dict[str, Any]:
        return {
            "_fs": self._fs,
            "_path": self._path,
            "_current_scene_index": self._current_scene_index,
            "_current_resolution_level": self._current_resolution_level,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._fs = state["_fs"]
        self._path = state["_path"]
        self._current_scene_index = state["_current_scene_index"]
        self._current_resolution_level = state["_current_resolution_level"]
        self._meta = None
        self._scenes = None
        self._physical_pixel_sizes = None
        self._metadata = None
        self._xarray_dask_data = None
        self._xarray_data = None
        self._dims = None
