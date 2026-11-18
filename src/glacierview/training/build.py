"""Build the training set: GLIMS outlines -> masks, and scene + DEM + mask -> pairs.

Notebook 09 calls `rasterize_outline` once per glacier. Notebook 10 calls
`build_training_pair` once per chosen scene and writes the result to
`glacierview.config.training_set_dir`.

Scene, DEM and mask all go on the glacier's `rasters.reference_grid`, the same
grid inference uses. So a label pixel covers the same ground as the image
pixel under it, whichever UTM zone the training scene came in.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image
from rasterio.features import rasterize
from shapely.geometry.base import BaseGeometry

from glacierview import config
from glacierview import rasters as raster_io
from glacierview.preprocess import fill_missing, prepare_dem, prepare_scene, resize_rasters


def rasterize_outline(
    grid: raster_io.Grid, outline: BaseGeometry, outline_crs: str = "EPSG:4326"
) -> np.ndarray:
    """A binary int32 mask of ``outline`` on `grid` (1 = glacier).

    Pass the glacier's ``rasters.reference_grid(dem_path)``. A pixel is glacier
    if its centre falls inside the outline.
    """
    shape = gpd.GeoSeries([outline], crs=outline_crs).to_crs(grid.crs).iloc[0]
    return rasterize(
        [shape], out_shape=grid.shape, transform=grid.transform,
        fill=0, default_value=1, dtype="int32",
    )


def build_training_pair(
    scene_path: Path, dem_path: Path, mask_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """One model-ready image and its label, preprocessed exactly as at inference.

    The scene goes onto the reference grid, is clipped to the fixed cut-offs and
    scaled, and has its fill masked. The DEM goes through `prepare_dem`. Inference
    fills missing pixels from neighbouring dates, but a training pair is a single
    date, so whatever is missing takes the scene's mean (`fill_missing`).

    Returns:
        image: (1, 128, 128, 7) float32: the six `COMMON_BANDS`, then the DEM.
        label: (128, 128, 1).

    Raises:
        ValueError: if the mask isn't on the reference grid. Masks from before
            the grid existed were drawn on one scene's grid. Rebuild them with
            notebook 09.
    """
    grid = raster_io.reference_grid(dem_path)
    scene = prepare_scene(scene_path, grid)
    dem = prepare_dem(dem_path, grid)
    image = fill_missing(np.concatenate((scene, dem), axis=2)[np.newaxis])

    label = np.array(Image.open(mask_path))
    if label.shape != grid.shape:
        raise ValueError(
            f"{mask_path} is {label.shape}, but the reference grid is {grid.shape}; "
            f"it predates the grid. Rebuild the masks with notebook 09."
        )
    label = resize_rasters({"mask": label[..., np.newaxis]}, config.MODEL_INPUT_DIM)["mask"]
    return image, label
