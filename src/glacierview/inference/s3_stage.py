"""Pull a batch of glaciers from S3 into the landing zone, then remove them.

The inference imagery is 486 GB, and even the quality-filtered 250-per-region
set is 57.5 GB, more than a laptop has free. So `glacierview areas-s3` works
in batches: `staged` downloads a few glaciers' listed scenes, the caller
segments them, and on leaving the ``with`` block the scenes are deleted.

Two rules keep this safe to run against a real data root:

* Only directories this module created are ever deleted. A glacier already
  present locally is used as it is and left in place.
* The batch's size is checked against free disk space before anything is
  downloaded.

DEMs are kept: all of them together are 89 MB, and a rerun would only fetch
them again.

The file list is the Athena output of ``sql/inference_data_query_250.sql``:
one row per scene, with ``glims_id`` and ``file_name`` columns.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import boto3
import pandas as pd
from botocore.config import Config

from glacierview import config
from glacierview.log import get_logger

logger = get_logger(__name__)

#: ``{glims_id}/{file_name}``. us-west-1.
IMAGES_BUCKET = ("full-time-series-images-t1-l2-sr", "us-west-1")
#: Flat ``{glims_id}_NASADEM.tif``. us-east-1 — the buckets are split.
DEMS_BUCKET = ("full-time-series-dems-t1-l2-sr", "us-east-1")
#: Free space to leave on the disk after a batch is downloaded.
DISK_MARGIN_BYTES = 2 * 1024**3


@dataclass
class Batch:
    """What `staged` managed to put on disk."""

    #: Glaciers whose scenes and DEM are on disk, ready to segment.
    ready: list[str] = field(default_factory=list)
    #: Glaciers that could not be downloaded; nothing of theirs is left behind.
    failed: list[str] = field(default_factory=list)


def read_file_list(path: Path | str) -> pd.DataFrame:
    """Read the filtered scene list: one row per ``(glims_id, file_name)``."""
    files = pd.read_csv(path, usecols=["glims_id", "file_name"])
    return files.drop_duplicates().sort_values(["glims_id", "file_name"])


@contextmanager
def staged(
    glims_ids: list[str],
    files: pd.DataFrame,
    data_label: str | None = None,
    workers: int = 16,
) -> Iterator[Batch]:
    """Download ``glims_ids``' listed scenes and DEMs; delete the scenes on exit.

    Raises:
        OSError: if the batch would leave less than `DISK_MARGIN_BYTES` free.
            Nothing has been downloaded at that point.
    """
    # Clients are thread-safe; size the pool to the workers sharing them.
    pool_config = Config(max_pool_connections=workers * 2)
    images = boto3.client("s3", region_name=IMAGES_BUCKET[1], config=pool_config)
    dems = boto3.client("s3", region_name=DEMS_BUCKET[1], config=pool_config)
    batch = Batch()
    created: list[Path] = []
    try:
        plan = _plan(images, glims_ids, files, data_label, batch)
        _check_disk(sum(size for scenes in plan.values() for *_, size in scenes), data_label)

        def fetch(glims_id: str) -> None:
            for key, local, _ in plan[glims_id]:
                images.download_file(IMAGES_BUCKET[0], key, str(local))
            dem = config.dem_path(glims_id, data_label)
            if not dem.exists():
                dem.parent.mkdir(parents=True, exist_ok=True)
                dems.download_file(DEMS_BUCKET[0], dem.name, str(dem))
            logger.info("staged %s: %d scenes", glims_id, len(plan[glims_id]))

        for glims_id in plan:
            dest = config.glacier_dir(glims_id, data_label)
            dest.mkdir(parents=True)
            created.append(dest)

        # One glacier per worker; boto3 already parallelises within a file.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {glims_id: pool.submit(fetch, glims_id) for glims_id in plan}
        for glims_id, future in futures.items():
            if future.exception() is None:
                batch.ready.append(glims_id)
                continue
            logger.error("download failed for %s: %s", glims_id, future.exception())
            batch.failed.append(glims_id)
            dest = config.glacier_dir(glims_id, data_label)
            shutil.rmtree(dest, ignore_errors=True)
            created.remove(dest)

        batch.ready.sort()
        yield batch
    finally:
        cleanup(created)


def cleanup(created: list[Path]) -> None:
    """Delete the glacier directories `staged` created, and nothing else."""
    for path in created:
        shutil.rmtree(path, ignore_errors=True)
    if created:
        logger.info("removed %d staged glacier directories", len(created))


def _plan(
    client,
    glims_ids: list[str],
    files: pd.DataFrame,
    data_label: str | None,
    batch: Batch,
) -> dict[str, list[tuple[str, Path, int]]]:
    """``{glims_id: [(s3 key, local path, bytes), ...]}`` for glaciers to download.

    Glaciers already on disk go straight to ``batch.ready``; ones with no
    listed scene in the bucket go to ``batch.failed``.
    """
    plan = {}
    for glims_id in glims_ids:
        dest = config.glacier_dir(glims_id, data_label)
        if dest.exists():
            logger.info("%s already on disk; using it and leaving it in place", glims_id)
            batch.ready.append(glims_id)
            continue

        sizes = _object_sizes(client, IMAGES_BUCKET[0], f"{glims_id}/")
        wanted = files.loc[files["glims_id"] == glims_id, "file_name"]
        keys = [f"{glims_id}/{name}" for name in wanted]
        missing = [k for k in keys if k not in sizes]
        if missing:
            logger.warning("%s: %d listed scenes not in the bucket", glims_id, len(missing))
        present = [(k, dest / k.split("/", 1)[1], sizes[k]) for k in keys if k in sizes]
        if not present:
            logger.error("%s: no listed scenes in the bucket", glims_id)
            batch.failed.append(glims_id)
            continue
        plan[glims_id] = present
    return plan


def _object_sizes(client, bucket: str, prefix: str) -> dict[str, int]:
    """``{key: bytes}`` for every object under ``prefix``."""
    sizes = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        sizes.update({o["Key"]: o["Size"] for o in page.get("Contents", [])})
    return sizes


def _check_disk(needed: int, data_label: str | None) -> None:
    """Refuse a batch that would leave less than `DISK_MARGIN_BYTES` free."""
    path = config.landsat_dir(data_label)
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    if needed + DISK_MARGIN_BYTES > free:
        raise OSError(
            f"batch needs {needed / 1e9:.1f} GB but {path} has {free / 1e9:.1f} GB free "
            f"(keeping {DISK_MARGIN_BYTES / 1e9:.1f} GB spare); lower --batch-size"
        )
    logger.info("batch is %.2f GB; %.1f GB free", needed / 1e9, free / 1e9)
