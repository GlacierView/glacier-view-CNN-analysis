"""The GLIMS glacier inventory: which glaciers to study, and where they are.

`select_glaciers` turns the full inventory into the ~18k glaciers the project
pulls imagery for; `add_bounding_boxes` adds the region each Earth Engine
request is clipped to; `add_region_rollup` buckets GLIMS's many regional
centres into the five regions the analysis reports.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import affinity
from shapely.geometry.base import BaseGeometry

from glacierview import config
from glacierview.log import get_logger

logger = get_logger(__name__)

#: Outlines drawn before this are too old to pair with the Landsat archive's
#: well-covered years.
MIN_OUTLINE_DATE = pd.Timestamp("2005-01-01")
#: A judgement call to keep typical glaciers: nothing tiny, nothing unusually large.
AREA_RANGE_KM2 = (1, 50)
#: Catch-all regional centres whose outlines duplicate or overlap the others.
EXCLUDED_REGIONS = (
    "Svalbard, Jan Mayen, and Bouvet",
    "Various (GlobGlacier)",
    "Randolph Glacier Inventory; Umbrella RC for merging the RGI into GLIMS",
    "Antarctic Peninsula",
    "Various (NSIDC)",
)
#: How much to pad a glacier's envelope, so the scene shows its surroundings.
BBOX_SCALE = 1.1

#: GLIMS ``geog_area`` -> the analysis region. ("Caucausus" is the spelling the
#: Athena tables and SQL use; keep it.)
REGION_ROLLUP = {
    "Himalaya (India, Nepal, Bhutan)": "Asia",
    "China": "Asia",
    "Northern Cordillera": "North America",
    "Canada": "North America",
    "Argentina": "South America",
    "Asia": "Asia",
    "Alaska": "North America",
    "Caucausus Region": "Caucausus",
    "Norway": "Europe",
    "Swiss Alps": "Europe",
    "Cordillera Blanca, Peru": "South America",
    "Austrian Alps": "Europe",
    "Italy": "Europe",
    "Mongolia": "Asia",
    "Chile": "South America",
    "New Zealand": "Oceania",
    "Siberian Mountains": "Asia",
    "Central Asia": "Asia",
    "Prince William Sound": "North America",
    "Iran": "Caucausus",
    "Olympic Peninsula": "North America",
    "Russian Glaciers (exact boundaries of RC to be defined)": "Asia",
    "Chilean glaciers, between -18 and -56 degrees latitude": "South America",
    "Turkey": "Asia",
    "Coropuna, Peru": "South America",
    "Northern Siberia": "Asia",
}


def read_inventory(path: Path | None = None) -> gpd.GeoDataFrame:
    """The full GLIMS polygon inventory."""
    return gpd.read_file(path or config.glims_inventory_path())


def read_selection(path: Path | None = None) -> gpd.GeoDataFrame:
    """The selected glaciers written by `write_selection` (notebook 01)."""
    return gpd.read_file(path or config.glims_selection_path())


def select_glaciers(inventory: pd.DataFrame) -> pd.DataFrame:
    """Narrow the inventory to recent, typical, regionally-specific glaciers.

    Keeps outlines drawn after `MIN_OUTLINE_DATE`, drops `EXCLUDED_REGIONS`,
    keeps areas strictly inside `AREA_RANGE_KM2`, then keeps the first outline
    of any glacier GLIMS has outlined more than once.
    """
    outline_date = pd.to_datetime(inventory["src_date"].str.split("T").str[0])
    low, high = AREA_RANGE_KM2
    steps = {
        f"outlined after {MIN_OUTLINE_DATE:%Y-%m-%d}": outline_date > MIN_OUTLINE_DATE,
        "not a catch-all region": ~inventory["geog_area"].isin(EXCLUDED_REGIONS),
        f"{low}-{high} km^2": inventory["db_area"].between(low, high, inclusive="neither"),
    }
    logger.info("%d glaciers in the inventory", inventory["glac_id"].nunique())
    selected = inventory
    for label, keep in steps.items():
        selected = selected[keep.loc[selected.index]]
        logger.info("%d %s", selected["glac_id"].nunique(), label)
    return selected.drop_duplicates("glac_id", keep="first").reset_index(drop=True)


def bounding_box(geometry: BaseGeometry, scale: float = BBOX_SCALE) -> list[tuple[float, float]]:
    """The geometry's envelope, scaled about its centre, as (lon, lat) corners."""
    envelope = affinity.scale(geometry.envelope, xfact=scale, yfact=scale)
    return [(float(x), float(y)) for x, y in envelope.exterior.coords]


def add_bounding_boxes(glaciers: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add ``bboxes``: each glacier's padded envelope, stored as a string.

    A string because shapefiles cannot hold lists. Coordinates are plain
    floats: numpy 2's ``np.float64(...)`` repr makes the string overflow the
    shapefile's 254-character field limit, which truncates it.
    """
    return glaciers.assign(bboxes=[str(bounding_box(g)) for g in glaciers.geometry])


def parse_bounding_box(text: str) -> list[tuple[float, float]]:
    """Read a ``bboxes`` string back. Accepts the old ``np.float64(...)`` form."""
    return ast.literal_eval(re.sub(r"np\.float64\(([^)]*)\)", r"\1", text))


def add_region_rollup(glaciers: pd.DataFrame) -> pd.DataFrame:
    """Add ``geog_area_rollup``, the region each glacier is reported under."""
    return glaciers.assign(geog_area_rollup=glaciers["geog_area"].map(REGION_ROLLUP))


def write_selection(glaciers: gpd.GeoDataFrame, path: Path | None = None) -> Path:
    path = path or config.glims_selection_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    glaciers.to_file(path)
    logger.info("wrote %d glaciers to %s", len(glaciers), path)
    return path


def write_athena_table(glaciers: pd.DataFrame, path: Path | None = None) -> Path:
    """Write the selection as the Parquet file behind the Athena ``glims_18k`` table.

    Geometry is dropped (it does not survive the conversion), and integer
    columns are widened to int64 so the Parquet schema matches the table's,
    whatever width the shapefile reader picked.
    """
    path = path or config.GLIMS_PARQUET
    table = pd.DataFrame(glaciers.drop(columns="geometry", errors="ignore"))
    ints = table.select_dtypes("integer").columns
    table = table.astype(dict.fromkeys(ints, "int64"))
    table.to_parquet(path)
    logger.info("wrote %d glaciers to %s", len(table), path)
    return path
