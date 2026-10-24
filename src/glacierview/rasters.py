"""Read Landsat GeoTIFFs and DEMs off disk into numpy arrays.

Every raster comes back in HWC layout (height, width, channel) with negative
values and -inf clamped to zero, since those mark unmeasured pixels rather
than real reflectance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def _clean(raster: np.ndarray) -> np.ndarray:
    """Clamp negative values to zero and move the band axis last.

    The comparison also catches -inf. NaN and +inf are deliberately left
    alone: they mean "no usable measurement", and callers decide how to fill
    them (inference substitutes the stack mean) rather than silently treating
    them as zero reflectance.
    """
    raster[raster < 0] = 0
    return np.moveaxis(raster, 0, -1)


def get_rasters(
    directory: Path | str,
    file_name: str | None = None,
    filtered_file_path: Path | str | None = None,
) -> dict[str, np.ndarray]:
    """Read the GeoTIFFs in a directory into ``{file_name: HWC array}``.

    Args:
        directory: one glacier's scene directory.
        file_name: read only this one file instead of the whole directory.
        filtered_file_path: CSV with a ``file_name`` column; only files listed
            there are read. This is how the metadata quality filter is applied
            before anything touches memory.
    """
    directory = Path(directory)
    names = [file_name] if file_name else sorted(p.name for p in directory.glob("*.tif"))
    if filtered_file_path:
        allowed = set(pd.read_csv(filtered_file_path)["file_name"])
        names = [n for n in names if n in allowed]

    rasters = {}
    for name in names:
        with rasterio.open(directory / name) as src:
            rasters[name] = _clean(src.read())
    return rasters


def get_dem(path: Path | str) -> dict[str, np.ndarray]:
    """Read a DEM's first band as ``{file_name: HWC array}``.

    Returned as a single-entry dict so it composes with `get_rasters` output.
    One DEM covers a glacier for every date, since terrain is static.
    """
    path = Path(path)
    with rasterio.open(path) as src:
        return {path.name: _clean(src.read(indexes=[1]))}
