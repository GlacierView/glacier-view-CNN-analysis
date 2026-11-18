"""Read Landsat GeoTIFFs and DEMs off disk into numpy arrays.

There are two ways to read, one per preprocessing pipeline (see
`glacierview.preprocess`):

- **Current:** `read_scene` and `read_dem` resample onto one glacier's
  `reference_grid`, so every date and the DEM share a single pixel grid. They
  read only the bands asked for and return missing pixels as NaN.
- **Published:** `get_rasters` and `get_dem` read each file whole, on its own
  grid, with negative values clamped to zero. This is what the published
  checkpoint was trained and run on.

Either way, arrays come back in HWC layout (height, width, channel).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform_bounds

#: The resolution of the reference grid, matching Landsat's 30 m pixels so that
#: `config.KM2_PER_PIXEL` holds on it.
GRID_RESOLUTION_M = 30.0
#: Landsat Collection 2 pixel edges sit 15 m off a multiple of 30 m in UTM.
#: Putting the reference grid on the same phase means a scene already in the
#: grid's UTM zone is copied pixel for pixel, with no resampling.
LANDSAT_GRID_PHASE_M = 15.0


@dataclass(frozen=True)
class Grid:
    """A pixel grid: where each pixel of an array sits on the ground."""

    crs: CRS
    transform: Affine
    shape: tuple[int, int]  # (height, width)


def utm_crs(lon: float, lat: float) -> CRS:
    """The WGS 84 UTM zone containing a point."""
    lon = (lon + 180) % 360 - 180
    zone = int((lon + 180) // 6) % 60 + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def reference_grid(dem_path: Path | str, resolution: float = GRID_RESOLUTION_M) -> Grid:
    """The one grid that a glacier's scenes, DEM and mask are all put on.

    Earth Engine returns each scene in its own UTM zone, and it returns the DEM in
    EPSG:4326. The same glacier can therefore arrive on two or more different grids,
    which don't overlap exactly. Trient's two grids are 128 m apart once each is
    resized to 128x128, and its DEM is 270 m from either. Resampling everything
    onto one grid makes a pixel mean the same patch of ground on every date.

    The grid covers the DEM's extent, which is the glacier's padded bounding
    box (the region every scene and the DEM were requested for). It sits in the
    UTM zone of the box's centre, at 30 m, snapped outwards to Landsat's pixel
    phase.
    """
    with rasterio.open(dem_path) as src:
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        crs = utm_crs((west + east) / 2, (south + north) / 2)
        left, bottom, right, top = transform_bounds(src.crs, crs, *src.bounds, densify_pts=21)
    phase = LANDSAT_GRID_PHASE_M
    left = math.floor((left - phase) / resolution) * resolution + phase
    top = math.ceil((top - phase) / resolution) * resolution + phase
    width = math.ceil((right - left) / resolution)
    height = math.ceil((top - bottom) / resolution)
    return Grid(crs, Affine(resolution, 0, left, 0, -resolution, top), (height, width))


def list_scenes(
    directory: Path | str,
    filtered_file_path: Path | str | None = None,
    since: datetime | None = None,
) -> list[str]:
    """The scene filenames in a glacier's directory, sorted (so date-ordered).

    Args:
        filtered_file_path: CSV with a ``file_name`` column; only files listed
            there are kept. This is how the metadata quality filter is applied
            before anything touches memory.
        since: skip scenes acquired before this date (token 1 of the name).
    """
    names = sorted(p.name for p in Path(directory).glob("*.tif"))
    if filtered_file_path:
        allowed = set(pd.read_csv(filtered_file_path)["file_name"])
        names = [n for n in names if n in allowed]
    if since:
        names = [n for n in names if datetime.fromisoformat(n.split("_")[1]) >= since]
    return names


def read_scene(path: Path | str, band_indexes: list[int], grid: Grid) -> np.ndarray:
    """Read some of a scene's bands onto `grid` as float32 HWC, NaN where missing.

    Args:
        band_indexes: 0-based band positions in the file, in the order wanted.

    A pixel is missing if it holds the file's nodata value or DN <= 0 (0 is
    Collection 2 fill, and negatives only come from nodata), or if it falls
    outside the scene's footprint. Saturated pixels are kept: they are real,
    bright measurements. Resampling is nearest-neighbour, so DN codes keep their
    meaning.
    """
    with rasterio.open(path) as src:
        data = src.read([i + 1 for i in band_indexes])
        invalid = data <= 0
        if src.nodata is not None:
            invalid |= data == src.nodata
        data[invalid] = 0
        out = np.zeros((len(band_indexes), *grid.shape), dtype=data.dtype)
        reproject(
            source=data, destination=out,
            src_transform=src.transform, src_crs=src.crs, src_nodata=0,
            dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=0,
            resampling=Resampling.nearest,
        )
    out = out.astype(np.float32)
    out[out == 0] = np.nan
    return np.moveaxis(out, 0, -1)


def read_dem(path: Path | str, grid: Grid) -> np.ndarray:
    """Read a DEM's elevation band onto `grid` as float32 HW1, NaN where missing.

    The DEM is continuous, so it is resampled bilinearly. Its nodata value is kept
    out of the interpolation.
    """
    with rasterio.open(path) as src:
        out = np.full((1, *grid.shape), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1), destination=out,
            src_nodata=src.nodata,
            dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return np.moveaxis(out, 0, -1)


# --------------------------------------------------------------------------- #
# The published pipeline's readers
# --------------------------------------------------------------------------- #

def _clean(raster: np.ndarray) -> np.ndarray:
    """Clamp negative values to zero and move the band axis last.

    The scenes are int32, and Earth Engine's nodata value (-2**31) is negative,
    so this turns nodata into 0, the same value as Collection 2 fill.
    """
    raster[raster < 0] = 0
    return np.moveaxis(raster, 0, -1)


def get_rasters(
    directory: Path | str,
    file_name: str | None = None,
    filtered_file_path: Path | str | None = None,
    since: datetime | None = None,
) -> dict[str, np.ndarray]:
    """Read whole GeoTIFFs, each on its own grid, into ``{file_name: HWC array}``.

    Args:
        directory: one glacier's scene directory.
        file_name: read only this one file instead of the whole directory.
        filtered_file_path, since: as for `list_scenes`.
    """
    directory = Path(directory)
    names = list_scenes(directory, filtered_file_path=filtered_file_path, since=since)
    if file_name:
        names = [n for n in names if n == file_name]

    rasters = {}
    for name in names:
        with rasterio.open(directory / name) as src:
            rasters[name] = _clean(src.read())
    return rasters


def get_dem(path: Path | str) -> dict[str, np.ndarray]:
    """Read a DEM's first band, on its own grid, as ``{file_name: HWC array}``.

    It comes back as a single-entry dict so that it composes with `get_rasters`
    output. One DEM covers a glacier on every date, since terrain is static.
    """
    path = Path(path)
    with rasterio.open(path) as src:
        return {path.name: _clean(src.read(indexes=[1]))}
