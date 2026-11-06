"""Segment every glacier in the landing zone and write area time series.

Each glacier's areas are written to ``glacier_areas/<glims_id>_areas.csv`` as
soon as it finishes, one column per area variant. Those files are the source
of truth: the aggregate CSVs are rebuilt from every one of them in the output
directory, so a run that dies partway still leaves usable output, and a later
run into the same directory adds to it rather than replacing it. With
``skip_done`` a rerun picks up where the last one stopped.

Failures are logged with a traceback and counted rather than aborting the run
— with hundreds of glaciers, one bad scene should not cost the whole batch.
The tally at the end is the thing to read: previously this loop could fail on
every glacier and still look successful.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from glacierview import config
from glacierview.inference import render
from glacierview.inference.predict import AREA_VARIANTS, segment_glacier
from glacierview.log import get_logger

logger = get_logger(__name__)

#: Monthly index the annual means are merged onto.
DATE_INDEX = pd.date_range(start="1979-01-01", end="2024-01-01", freq="MS")

CSV_NAMES = {
    "raw": "areas_no_threshold.csv",
    "thresholded": "areas_05_thresh.csv",
    "binary": "areas_binary_05.csv",
}

PER_GLACIER_DIR = "glacier_areas"


def per_glacier_path(out_dir: Path, glims_id: str) -> Path:
    """Where one glacier's per-scene areas are written."""
    return out_dir / PER_GLACIER_DIR / f"{glims_id}_areas.csv"


def done_glims_ids(out_dir: Path) -> set[str]:
    """Glaciers that already have a per-glacier CSV in ``out_dir``."""
    return {
        p.name.removesuffix("_areas.csv")
        for p in (out_dir / PER_GLACIER_DIR).glob("*_areas.csv")
    }


def to_annual_means(areas: list[float], dates, glims_id: str) -> pd.DataFrame:
    """Collapse a per-date series to one value per year, stamped mid-year.

    The mid-year offset makes each annual value plot at the centre of the
    period it summarises rather than at January.
    """
    series = pd.DataFrame({glims_id: areas}, index=pd.DatetimeIndex(dates))
    annual = series.groupby(pd.PeriodIndex(series.index, freq="Y"))[glims_id].mean()
    annual.index = annual.index.to_timestamp() + pd.DateOffset(months=6)
    return annual.to_frame().rename_axis("Dates").reset_index()


def write_aggregates(out_dir: Path) -> int:
    """Rebuild the aggregate CSVs from every per-glacier CSV in ``out_dir``.

    Annual means outside `DATE_INDEX` are dropped, as before; they remain in
    the per-glacier files. Returns the number of glaciers aggregated.
    """
    columns: dict[str, list[pd.Series]] = {name: [] for name in AREA_VARIANTS}
    paths = sorted((out_dir / PER_GLACIER_DIR).glob("*_areas.csv"))
    for path in paths:
        glims_id = path.name.removesuffix("_areas.csv")
        per_scene = pd.read_csv(path, index_col="Dates", parse_dates=["Dates"])
        missing = set(AREA_VARIANTS) - set(per_scene.columns)
        if missing:
            logger.warning("%s lacks %s; skipped in aggregates", path.name, sorted(missing))
            continue
        for name in AREA_VARIANTS:
            annual = to_annual_means(per_scene[name].tolist(), per_scene.index, glims_id)
            columns[name].append(annual.set_index("Dates")[glims_id])

    for name in AREA_VARIANTS:
        frame = pd.DataFrame(index=pd.DatetimeIndex(DATE_INDEX, name="Dates"))
        if columns[name]:
            frame = frame.join(pd.concat(columns[name], axis=1), how="left")
        frame.reset_index().to_csv(out_dir / CSV_NAMES[name], index=False)
    return len(columns["binary"])


def run(
    model,
    device,
    out_dir: Path,
    data_label: str | None = None,
    glims_ids: list[str] | None = None,
    write_gifs: bool = True,
    filtered_file_path: Path | str | None = None,
    skip_done: bool = False,
) -> dict:
    """Process the landing zone. Returns a summary dict.

    Args:
        filtered_file_path: CSV with a ``file_name`` column; only those scenes
            are segmented.
        skip_done: skip glaciers that already have a per-glacier CSV in
            ``out_dir``. They are counted in ``skipped``, not ``succeeded``.
    """
    landsat = config.landsat_dir(data_label)
    if not landsat.is_dir():
        raise FileNotFoundError(f"no landing zone at {landsat}")

    if glims_ids is None:
        glims_ids = sorted(
            d.name for d in landsat.iterdir() if d.is_dir() and not d.name.startswith(".")
        )
    if not glims_ids:
        raise FileNotFoundError(f"{landsat} contains no glacier directories")

    (out_dir / PER_GLACIER_DIR).mkdir(parents=True, exist_ok=True)

    skipped: list[str] = []
    if skip_done:
        done = done_glims_ids(out_dir)
        skipped = [g for g in glims_ids if g in done]
        glims_ids = [g for g in glims_ids if g not in done]

    failures: list[str] = []
    try:
        for i, glims_id in enumerate(glims_ids, start=1):
            try:
                result = segment_glacier(
                    model, glims_id, device, data_label,
                    variants=tuple(AREA_VARIANTS),
                    filtered_file_path=filtered_file_path,
                )
            except Exception:
                failures.append(glims_id)
                logger.exception("[%d/%d] failed: %s", i, len(glims_ids), glims_id)
                continue

            pd.DataFrame(
                result["areas"], index=pd.DatetimeIndex(result["dates"], name="Dates"),
            ).to_csv(per_glacier_path(out_dir, glims_id))

            if write_gifs:
                render.write_gif(
                    gif_path=out_dir / "gifs" / f"{glims_id}_final.gif",
                    scratch_dir=out_dir / "tmp" / glims_id,
                    images=result["smoothed"],
                    probabilities=result["probabilities"],
                    file_names=result["file_names"],
                    stride=10,
                    overlay=True,
                )
            logger.info("[%d/%d] %s", i, len(glims_ids), glims_id)
    finally:
        # Also on Ctrl-C, so whatever finished is reflected in the aggregates.
        write_aggregates(out_dir)

    succeeded = len(glims_ids) - len(failures)
    logger.info("%d/%d glaciers succeeded", succeeded, len(glims_ids))
    if skipped:
        logger.info("%d already done, skipped", len(skipped))
    if failures:
        logger.warning("%d failed: %s", len(failures), ", ".join(failures))
    return {
        "total": len(glims_ids),
        "succeeded": succeeded,
        "skipped": len(skipped),
        "failures": failures,
    }
