"""Animate three glaciers per region as one grid GIF.

Picks three glaciers in each region of the 250-per-region file list,
downloads their scenes from S3, segments them with the published model, and
writes one row per region, one column per glacier, one frame per year.
``--select`` chooses which three:

    largest    largest GLIMS ``db_area``
               -> figures/largest_glaciers_by_region.gif
    shrinkage  largest relative loss of area, from the 250-per-region run's
               annual means (see `relative_change`)
               -> figures/most_shrinking_glaciers_by_region.gif

Alongside, in src/segmentation/gifs/by_region/ (gitignored):

    <glims_id>.gif    every scene, RGB above and mask overlay below
    <glims_id>.npz    what the grid is drawn from: RGB, probabilities,
                      dates, native sizes, binary areas

A glacier's .npz is reused if present, so rerunning only redraws the figure;
delete the .npz files to segment again. The scenes are deleted after
segmentation, as in ``glacierview areas-s3``.

Usage, from the repo root (needs the same AWS and Hugging Face logins as
scripts/run_inference_250.sh; ``shrinkage`` also needs that run's output):

    uv run python scripts/region_gif_figure.py --select largest
    uv run python scripts/region_gif_figure.py --select shrinkage
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from glacierview.inference import render, s3_stage
from glacierview.inference.predict import segment_glacier
from glacierview.log import configure, get_logger

logger = get_logger("region_gif_figure")

REPO = Path(__file__).resolve().parent.parent
FILE_LIST = REPO / "src/glims/data/inference_samples/filtered_inference_data_250.csv"
#: Annual means from scripts/run_inference_250.sh; ranks glaciers for ``shrinkage``.
ANNUAL_AREAS = REPO / "src/segmentation/areas_250/areas_binary_05.csv"
OUT_DIR = REPO / "src/segmentation/gifs/by_region"
CHECKPOINT = "GlacierView/glacierview-unet"
PER_REGION = 3

#: --select -> (figure, title)
SELECTIONS = {
    "largest": (
        REPO / "figures/largest_glaciers_by_region.gif",
        "Three largest glaciers per region",
    ),
    "shrinkage": (
        REPO / "figures/most_shrinking_glaciers_by_region.gif",
        "Three glaciers per region with the largest relative area loss",
    ),
}

#: A glacier is ranked for ``shrinkage`` only with at least this record.
MIN_SPAN_YEARS = 25
MIN_YEARS = 10
#: Years averaged at each end of the record, so one bad year can't rank it.
END_YEARS = 3

#: Row order, and display names for the rollup values (the data misspells Caucasus).
REGIONS = {
    "Europe": "Europe",
    "Caucausus": "Caucasus",
    "Asia": "Asia",
    "North America": "North America",
    "South America": "South America",
}


def relative_change(annual_areas: Path) -> pd.DataFrame:
    """Each glacier's fractional area change over its observed record.

    The mean of its last `END_YEARS` annual means over the mean of its first
    `END_YEARS`, minus one: -0.4 is a 40% loss. ``ranked`` is False for
    glaciers observed over fewer than `MIN_SPAN_YEARS`, or in fewer than
    `MIN_YEARS` years; they are shown but never picked as most shrinking.
    """
    annual = pd.read_csv(annual_areas, index_col="Dates", parse_dates=["Dates"])
    rows = {}
    for glims_id, series in annual.items():
        series = series.dropna()
        if len(series) < 2:
            continue
        years = series.index.year
        start = series.iloc[:END_YEARS].mean()
        end = series.iloc[-END_YEARS:].mean()
        rows[glims_id] = {
            "change": end / start - 1,
            "from": years[0],
            "to": years[-1],
            "ranked": len(series) >= MIN_YEARS and years[-1] - years[0] >= MIN_SPAN_YEARS,
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("glims_id")


def pick_glaciers(file_list: Path, select: str) -> pd.DataFrame:
    """`PER_REGION` glaciers per region, most extreme first."""
    rows = pd.read_csv(
        file_list, usecols=["glims_id", "geog_area_rollup", "geog_area", "db_area"]
    )
    glaciers = rows.drop_duplicates("glims_id").rename(
        columns={"geog_area_rollup": "region", "geog_area": "place"}
    )
    glaciers = glaciers.join(relative_change(ANNUAL_AREAS), on="glims_id")
    if select == "shrinkage":
        glaciers = glaciers[glaciers["ranked"] == True]  # noqa: E712 (NaN-safe)
        key, ascending = "change", True
    else:
        key, ascending = "db_area", False
    picked = (
        glaciers.sort_values(key, ascending=ascending)
        .groupby("region")
        .head(PER_REGION)
    )
    picked = picked[picked["region"].isin(REGIONS)]
    picked["row"] = picked["region"].map(list(REGIONS).index)
    picked["place"] = picked["place"].str.replace("Caucausus", "Caucasus")
    return picked.sort_values(["row", key], ascending=[True, ascending])


def cache_path(glims_id: str) -> Path:
    return OUT_DIR / f"{glims_id}.npz"


def segment(glims_ids: list[str], device_name: str | None) -> None:
    """Download, segment and cache every glacier without an .npz yet."""
    todo = [g for g in glims_ids if not cache_path(g).exists()]
    if not todo:
        logger.info("all %d glaciers cached; skipping inference", len(glims_ids))
        return

    import torch

    from glacierview.models import load_checkpoint

    if device_name:
        device = torch.device(device_name)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(CHECKPOINT, device)

    files = s3_stage.read_file_list(FILE_LIST)
    with s3_stage.staged(todo, files[files["glims_id"].isin(todo)]) as staged:
        if staged.failed:
            logger.error("download failed: %s", ", ".join(staged.failed))
        for glims_id in staged.ready:
            result = segment_glacier(
                model, glims_id, device,
                variants=("binary",), filtered_file_path=FILE_LIST,
            )
            render.write_gif(
                gif_path=OUT_DIR / f"{glims_id}.gif",
                scratch_dir=OUT_DIR / "tmp" / glims_id,
                images=result["smoothed"],
                probabilities=result["probabilities"],
                file_names=result["file_names"],
                stride=1,
                overlay=True,
            )
            rgb = np.clip(result["smoothed"][..., [2, 1, 0]], 0, 1)
            np.savez_compressed(
                cache_path(glims_id),
                rgb=(rgb * 255).astype(np.uint8),
                probabilities=result["probabilities"].numpy()[:, 0].astype(np.float16),
                dates=np.array([d.strftime("%Y-%m-%d") for d in result["dates"]]),
                sizes=np.array(_native_sizes(glims_id, result["file_names"])),
                areas=np.array(result["areas"]["binary"]),
            )
            logger.info("cached %s", glims_id)


def _native_sizes(glims_id: str, file_names: list[str]) -> list[tuple[int, int]]:
    """Each scene's native (height, width), so the grid keeps true aspect."""
    import rasterio

    from glacierview import config

    sizes = []
    for name in file_names:
        with rasterio.open(config.glacier_dir(glims_id) / name) as src:
            sizes.append((src.height, src.width))
    return sizes


def compose(
    glaciers: pd.DataFrame, out_path: Path, title: str, frame_ms: int, dpi: int
) -> None:
    """One frame per year; each cell holds its glacier's latest scene so far."""
    series = {g: dict(np.load(cache_path(g))) for g in glaciers["glims_id"]}
    for data in series.values():
        data["years"] = np.array([int(d[:4]) for d in data["dates"]])
    first = min(int(d["years"].min()) for d in series.values())
    last = max(int(d["years"].max()) for d in series.values())

    n_rows = glaciers["row"].nunique()
    frames = []
    for year in range(first, last + 1):
        fig, axes = plt.subplots(
            n_rows, PER_REGION, figsize=(2.6 * PER_REGION + 0.6, 2.75 * n_rows),
            facecolor="white", dpi=dpi,
        )
        fig.suptitle(
            f"{title} — {year}",
            fontsize=13, fontweight="bold", y=0.995,
        )
        for (_, rows), ax_row in zip(glaciers.groupby("row", sort=True), axes):
            for col, ax in enumerate(ax_row):
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(
                        REGIONS[rows["region"].iloc[0]], fontsize=12, fontweight="bold"
                    )
                if col >= len(rows):
                    ax.set_axis_off()
                    continue
                _draw_cell(ax, rows.iloc[col], series[rows.iloc[col]["glims_id"]], year)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in frames]
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=[frame_ms] * (len(frames) - 1) + [frame_ms * 4], loop=0,
        optimize=True,
    )
    logger.info("wrote %s: %d frames, %.1f MB",
                out_path, len(frames), out_path.stat().st_size / 1e6)


def _draw_cell(ax, glacier: pd.Series, data: dict, year: int) -> None:
    title = f"{glacier['place']}\n{glacier['glims_id']}"
    if pd.notna(glacier["change"]):
        title += f"  {glacier['change']:+.0%} ({glacier['from']:.0f}–{glacier['to']:.0f})"
    ax.set_title(title, fontsize=8)
    shown = np.flatnonzero(data["years"] <= year)
    if not shown.size:
        ax.set_facecolor("#e6e6e6")
        ax.text(0.5, 0.5, "no scene yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#555555")
        return
    i = shown[-1]
    height, width = data["sizes"][i]
    extent = (0, width, height, 0)  # back to the scene's native aspect
    ax.imshow(data["rgb"][i], extent=extent, interpolation="bilinear")
    prob = data["probabilities"][i].astype(np.float32)
    ys = np.linspace(0, height, prob.shape[0])
    xs = np.linspace(0, width, prob.shape[1])
    ax.contourf(xs, ys, prob, levels=[0.5, 1.01], colors=["#00e5ff"], alpha=0.25)
    ax.contour(xs, ys, prob, levels=[0.5], colors=["#00e5ff"], linewidths=1.1)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    area = data["areas"][i]
    ax.set_xlabel(f"{data['dates'][i]}   {area:.{1 if area >= 10 else 2}f} km²", fontsize=8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", help="cuda, mps or cpu; autodetected by default")
    parser.add_argument("--frame-ms", type=int, default=500, help="ms per year")
    parser.add_argument("--dpi", type=int, default=75)
    parser.add_argument("--select", choices=SELECTIONS, default="largest")
    parser.add_argument("--out", type=Path, help="default: set by --select")
    args = parser.parse_args()
    configure()
    figure, title = SELECTIONS[args.select]

    glaciers = pick_glaciers(FILE_LIST, args.select)
    logger.info("glaciers:\n%s", glaciers.to_string(index=False))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    segment(list(glaciers["glims_id"]), args.device)

    missing = [g for g in glaciers["glims_id"] if not cache_path(g).exists()]
    if missing:
        logger.error("not segmented, left out of the figure: %s", ", ".join(missing))
        glaciers = glaciers[~glaciers["glims_id"].isin(missing)]
    compose(glaciers, args.out or figure, title, args.frame_ms, args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
