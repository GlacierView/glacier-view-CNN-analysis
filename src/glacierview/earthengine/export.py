"""Download Landsat scenes and NASADEMs from Google Earth Engine.

Each scene is written as ``<glims_id>_<date>_<L5|L7|L8>_C02_T1_L2_SR.tif`` —
the name is load-bearing downstream — and its ``getInfo()`` metadata is
collected into ``meta_data/metadata_list_l{5,7,8}`` beside it. One line per
scene goes to the pull log, which `glacierview.metadata.parse_pull_log` reads
back; its format is positional, so change both together.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import ee
import geemap
import pandas as pd

from glacierview.log import get_logger

logger = get_logger(__name__)

#: Satellite token -> (Earth Engine collection, quality property). Landsat 8
#: reports image quality per sensor, so its OLI score stands in.
COLLECTIONS = {
    "L5": ("LANDSAT/LT05/C02/T1_L2", "IMAGE_QUALITY"),
    "L7": ("LANDSAT/LE07/C02/T1_L2", "IMAGE_QUALITY"),
    "L8": ("LANDSAT/LC08/C02/T1_L2", "IMAGE_QUALITY_OLI"),
}
DEM_IMAGE = "NASA/NASADEM_HGT/001"
#: Scene-level filters applied server-side. Stricter cut-offs happen later, in SQL.
MAX_CLOUD_COVER = 10
REQUIRED_QUALITY = 9
SCALE_M = 30
#: Pull-log line format; `parse_pull_log` depends on it.
LOG_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"

BoundingBox = Sequence[tuple[float, float]]


def scene_filename(glims_id: str, date: str, satellite: str) -> str:
    return f"{glims_id}_{date}_{satellite}_C02_T1_L2_SR.tif"


def _log_line(glims_id: str, info: dict, satellite: str) -> str:
    props = info["properties"]
    fields = [
        f"GLIMSID:{glims_id}",
        f"CRS:{info['bands'][0]['crs']}",
        f"CLOUD_COVER:{props['CLOUD_COVER']}",
        f"CLOUD_COVER_LAND:{props['CLOUD_COVER_LAND']}",
        f"SCENE_CENTER_TIME:{props['SCENE_CENTER_TIME']}",
    ]
    if satellite == "L8":
        fields += [f"IMAGE_QUALITY_OLI:{props['IMAGE_QUALITY_OLI']}",
                   f"IMAGE_QUALITY_TIRS:{props['IMAGE_QUALITY_TIRS']}"]
    else:
        # The trailing separator keeps L5/L7 lines the same width as L8's.
        fields += [f"IMAGE_QUALITY:{props['IMAGE_QUALITY']}", ""]
    return "; ".join(fields).rstrip(" ")


class EePull:
    """One Earth Engine pull, logging every scene to ``<log_dir>/log_<name>.log``.

    Call ``ee.Authenticate()`` and ``ee.Initialize(project=...)`` first.
    """

    def __init__(self, log_dir: Path | str, name: str):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = (log_dir / f"log_{name}.log").resolve()
        # A dedicated logger, so the pull log works whatever else has
        # configured logging (``basicConfig`` silently does nothing then).
        self.log = logging.getLogger(f"glacierview.earthengine.pull.{name}")
        self.log.setLevel(logging.INFO)
        self.log.propagate = False
        attached = {getattr(h, "baseFilename", None) for h in self.log.handlers}
        if str(self.log_path) not in attached:
            handler = logging.FileHandler(self.log_path)
            handler.setFormatter(logging.Formatter(LOG_FORMAT))
            self.log.addHandler(handler)

    def export_landsat(
        self,
        satellite: str,
        glims_id: str,
        bounding_box: BoundingBox,
        start: datetime,
        end: datetime,
        out_dir: Path | str,
    ) -> int:
        """Export every qualifying scene from one satellite. Returns the count."""
        collection_id, quality = COLLECTIONS[satellite]
        region = ee.Geometry.Polygon(list(bounding_box))
        scenes = (
            ee.ImageCollection(collection_id)
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.lte("CLOUD_COVER", MAX_CLOUD_COVER))
            .filter(ee.Filter.eq(quality, REQUIRED_QUALITY))
        )
        dates = geemap.image_dates(scenes).getInfo()
        scene_list = scenes.toList(scenes.size())

        out_dir = Path(out_dir)
        (out_dir / "meta_data").mkdir(parents=True, exist_ok=True)
        metadata = []
        for i, date in enumerate(dates):
            image = ee.Image(scene_list.get(i))
            info = image.getInfo()
            metadata.append(info)
            self.log.info(_log_line(glims_id, info, satellite))
            geemap.ee_export_image(
                image,
                filename=str(out_dir / scene_filename(glims_id, date, satellite)),
                scale=SCALE_M,
                region=region,
                file_per_band=False,
            )
        pd.Series(metadata).to_csv(out_dir / "meta_data" / f"metadata_list_{satellite.lower()}")
        return len(dates)

    def export_all_landsat(
        self,
        glims_id: str,
        bounding_box: BoundingBox,
        start: datetime,
        end: datetime,
        out_dir: Path | str,
    ) -> dict[str, int | None]:
        """Export from every satellite. A failing satellite is logged and skipped.

        Returns ``{satellite: scene count}``, with ``None`` for a failure.
        """
        counts: dict[str, int | None] = {}
        for satellite in COLLECTIONS:
            try:
                counts[satellite] = self.export_landsat(
                    satellite, glims_id, bounding_box, start, end, out_dir
                )
            except Exception as exc:
                counts[satellite] = None
                logger.warning("%s %s failed: %s", glims_id, satellite, exc)
        return counts

    def export_dem(self, glims_id: str, bounding_box: BoundingBox, out_dir: Path | str) -> Path:
        """Export the glacier's NASADEM, clipped to its bounding box."""
        region = ee.Geometry.Polygon(list(bounding_box))
        out_path = Path(out_dir) / f"{glims_id}_NASADEM.tif"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        geemap.ee_export_image(
            ee.Image(DEM_IMAGE).clip(region), filename=str(out_path), scale=SCALE_M, region=region
        )
        return out_path
