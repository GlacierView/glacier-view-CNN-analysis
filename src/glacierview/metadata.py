"""Turn a landing zone into the flat tables behind the Athena database.

Four tables, one row per scene, all keyed on ``file_name``:

* `parse_pull_log` — what the pull logged (CRS, cloud cover, quality).
* `scan_scenes` — file attributes (date, satellite, size) and image
  attributes (shape, bounds, EPSG, zero/no-data/negative pixel counts).
* `read_ee_metadata` — Earth Engine's own per-scene metadata, flattened.

The SQL in ``sql/`` filters on these to choose training and inference scenes.
Athena reads the CSVs by column position, so column order is part of the
contract: keep the ``*_COLUMNS`` lists stable.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioIOError
from tqdm import tqdm

from glacierview.log import get_logger

logger = get_logger(__name__)

FILE_ATTRIBUTE_COLUMNS = ["file_name", "src_date", "landsat", "file size in bytes"]
IMAGE_ATTRIBUTE_COLUMNS = [
    "file_name", "height_in_pixels", "width_in_pixels", "left_bound", "right_bound",
    "top_bound", "bottom_bound", "num_of_bands", "epsg_code", "no_data_val",
    "pixel_width_res", "pixel_height_res", "zero_pixel_count", "no_data_pixel_count",
    "negative_pixel_count",
]
#: Earth Engine properties, in table order. Each column is the lower-cased
#: property name; properties a satellite does not report are left blank.
EE_PROPERTIES = [
    "algorithm_source_surface_reflectance", "algorithm_source_surface_temperature",
    "cloud_cover", "cloud_cover_land", "collection_category", "data_source_air_temperature",
    "data_source_elevation", "data_source_ozone", "data_source_pressure",
    "data_source_reanalysis", "data_source_water_vapor", "date_product_generated",
    "earth_sun_distance", "geometric_rmse_model", "geometric_rmse_model_x",
    "geometric_rmse_model_y", "ground_control_points_model", "ground_control_points_version",
    "l1_date_product_generated", "l1_landsat_product_id", "l1_processing_level",
    "l1_processing_software_version", "landsat_product_id", "landsat_scene_id",
    "processing_level", "processing_software_version", "scene_center_time", "sensor_id",
    "spacecraft_id", "sun_azimuth", "sun_elevation", "wrs_path", "wrs_row", "image_quality",
    "image_quality_oli", "image_quality_tirs", "temperature_maximum_band_st_b6",
    "temperature_minimum_band_st_b6", "ephemeris_type",
    "data_source_tirs_stray_light_correction", "geometric_rmse_verify",
    "ground_control_points_verify", "roll_angle", "target_wrs_path", "target_wrs_row",
    "temperature_maximum_band_st_b10", "temperature_minimum_band_st_b10", "tirs_ssm_model",
]


def glacier_ids(landsat_dir: Path) -> list[str]:
    """The per-glacier directories under a landing zone's ``landsat/``."""
    dirs = (p for p in Path(landsat_dir).iterdir() if p.is_dir())
    return sorted(p.name for p in dirs if not p.name.startswith("."))


def parse_pull_log(path: Path) -> pd.DataFrame:
    """One row per logged scene, from a pull log written by `EePull`.

    Lines look like ``<date> <time>,<ms>:INFO:GLIMSID:<id>; CRS:EPSG:<code>;
    CLOUD_COVER:<x>; CLOUD_COVER_LAND:<x>; SCENE_CENTER_TIME:<hh:mm:ss>;
    IMAGE_QUALITY[_OLI]:<x>; ...`` — so the fields are found by position.
    """
    log = pd.read_csv(path, sep=";", header=None, dtype=str)

    def value(column: int) -> pd.Series:
        return log[column].str.split(":", n=1).str[1]

    stamp, rest = log[0].str.split(",", n=1).str[0], log[0].str.split(",", n=1).str[1]
    return pd.DataFrame({
        "ee_pull_date": stamp,
        "glims_id": rest.str.split(":").str[3],
        "espg_crs": log[1].str.split(":").str[2],
        "cloud_cover": value(2),
        "cloud_cover_land": value(3),
        "scene_center_time": value(4),
        "image_quality": value(5),
    })


def _image_attributes(path: Path) -> list | None:
    """One `IMAGE_ATTRIBUTE_COLUMNS` row, or None if the raster cannot be read."""
    with rasterio.open(path) as src:
        try:
            raster = src.read()
        except RasterioIOError:  # a few exported tifs are truncated
            return None
        return [
            path.name, src.height, src.width, src.bounds.left, src.bounds.right,
            src.bounds.top, src.bounds.bottom, src.count, src.crs.to_epsg(), src.nodata,
            src.res[0], src.res[1], np.sum(raster == 0), np.sum(raster == src.nodata),
            np.sum(raster < 0),
        ]


def scan_scenes(landsat_dir: Path, glims_ids: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """File and image attributes for every scene of the given glaciers.

    Every pixel is read, so over a full landing zone this is slow and
    disk-bound. Unreadable rasters keep their file row but get no image row.
    """
    files, images = [], []
    for glims_id in tqdm(list(glims_ids)):
        for path in sorted((Path(landsat_dir) / glims_id).glob("*.tif")):
            _, src_date, landsat, *_ = path.stem.split("_")
            files.append([path.name, src_date, landsat, path.stat().st_size])
            if (row := _image_attributes(path)) is not None:
                images.append(row)
    return (pd.DataFrame(files, columns=FILE_ATTRIBUTE_COLUMNS),
            pd.DataFrame(images, columns=IMAGE_ATTRIBUTE_COLUMNS))


def _metadata_records(path: Path) -> list[dict]:
    """The ``getInfo()`` dicts `EePull` saved as a one-column CSV of reprs."""
    saved = pd.read_csv(path, index_col=0)
    return [] if saved.empty else [ast.literal_eval(text) for text in saved.iloc[:, 0]]


def _flatten(record: dict, glims_id: str, include_version: bool) -> dict:
    # ids look like LANDSAT/LC08/C02/T1_L2/LC08_195028_20130411
    id_parts = record["id"].split("_")
    satellite, date = f"L{int(id_parts[1][-2:])}", id_parts[-1]
    src_date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    props = record["properties"]
    row = {"file_name": f"{glims_id}_{src_date}_{satellite}_C02_T1_L2_SR.tif"}
    if include_version:
        row["version"] = str(record["version"])
    row |= {"src_date": src_date, "ee_identifier": record["id"], "glims_id": glims_id}
    row |= {name: props.get(name.upper(), "") for name in EE_PROPERTIES}
    return row


def read_ee_metadata(
    landsat_dir: Path, glims_ids: Iterable[str], *, include_version: bool = False
) -> pd.DataFrame:
    """Earth Engine's metadata for every scene, one row per ``file_name``.

    Earth Engine sometimes returns more than one record for a scene; the first
    is kept. ``include_version`` adds the column the training table carries
    and the inference table does not.
    """
    rows = []
    for glims_id in tqdm(list(glims_ids)):
        for path in sorted((Path(landsat_dir) / glims_id / "meta_data").glob("metadata_list_*")):
            rows += [_flatten(r, glims_id, include_version) for r in _metadata_records(path)]
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    deduped = table.drop_duplicates("file_name").sort_values("file_name", ignore_index=True)
    logger.info("%d scenes (%d duplicate records dropped)", len(deduped), len(table) - len(deduped))
    return deduped
