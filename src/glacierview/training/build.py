"""Build the training set: GLIMS outlines -> masks, and scene + DEM + mask -> pairs.

Notebook 09 calls `rasterize_outline` once per glacier; notebook 10 calls
`build_training_pair` once per chosen scene and writes the result to
`glacierview.config.training_set_dir`.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.mask import mask
from shapely.geometry.base import BaseGeometry

from glacierview import config
from glacierview import rasters as raster_io
from glacierview.preprocess import get_common_bands, normalize_rasters, resize_rasters

_OUTSIDE = -999


def rasterize_outline(
    scene_path: Path, outline: BaseGeometry, outline_crs: str = "EPSG:4326"
) -> np.ndarray:
    """A binary mask of ``outline`` on the scene's pixel grid (1 = glacier).

    The outline is reprojected into the scene's CRS first: scenes arrive in
    UTM, and the same glacier can fall in different zones on different dates.
    """
    with rasterio.open(scene_path) as src:
        shape = gpd.GeoSeries([outline], crs=outline_crs).to_crs(src.crs).iloc[0]
        masked, _ = mask(src, [shape], nodata=_OUTSIDE)
    return np.where(masked[0] == _OUTSIDE, 0, 1).astype(masked.dtype)


def build_training_pair(
    scene_path: Path, dem_path: Path, mask_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """One model-ready image and its label.

    Returns:
        image: (1, 128, 128, 7) — the six `COMMON_BANDS` then the DEM.
        label: (128, 128, 1).

    The DEM is normalised *before* resizing here, the reverse of
    `preprocess.prepare_glacier_stack`; both orders are as the published
    model was trained and run, so they are kept.
    """
    scene_path = Path(scene_path)
    scenes = raster_io.get_rasters(scene_path.parent, file_name=scene_path.name)
    scenes = resize_rasters(normalize_rasters(get_common_bands(scenes, list(config.COMMON_BANDS))),
                            config.MODEL_INPUT_DIM)
    dem = resize_rasters(normalize_rasters(raster_io.get_dem(dem_path)), config.MODEL_INPUT_DIM)
    label = resize_rasters({"mask": np.array(Image.open(mask_path))[..., np.newaxis]},
                           config.MODEL_INPUT_DIM)["mask"]
    image = np.stack([np.concatenate((scene, *dem.values()), axis=2) for scene in scenes.values()])
    return image, label
