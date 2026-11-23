"""The paper's overview figure: every glacier studied, and one per region up close.

A world map (Equal Earth) shows the 7,174 glaciers the model was trained on,
which include all 1,024 inference glaciers, coloured by region. Each region's
highlighted glacier is the one with the largest relative area loss among those
whose predicted area at the start of their record is over `MIN_START_KM2`;
it gets a pair of panels, the earliest and latest predicted outline, joined
to its point on the map. The change is measured as in
``region_gif_figure.relative_change``.

Two layouts, each saved as PNG to figures/:

    global_overview_stacked   panels above and below the map
    global_overview_side      map on the left, panels stacked on the right

The panels are drawn from ``region_gif_figure``'s per-glacier .npz files;
glaciers without one are downloaded and segmented first, which needs the same
AWS and Hugging Face logins as scripts/run_inference_250.sh. The trained-on
glacier list is read from the training bucket once and kept in
data/analysis/trained_glims_ids.csv.

Usage, from the repo root:

    uv run python scripts/global_overview_figure.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from pyproj import Transformer

import region_gif_figure as rgf
from glacierview import config, glims
from glacierview.log import configure, get_logger

logger = get_logger("global_overview_figure")

TRAINED_IDS = rgf.REPO / "data/analysis/trained_glims_ids.csv"
TRAINING_BUCKET = ("training-images-t1-l2-sr", "images/")
LAND = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"
CRS = "EPSG:8857"  # Equal Earth
#: Map latitude range: no studied glacier lies outside it, and it drops Antarctica.
LAT_RANGE = (-58, 84)
MIN_START_KM2 = 10
DPI = 300

#: Region -> colour; the first five slots of the dataviz skill's validated
#: categorical palette, in `rgf.REGIONS` order.
COLOURS = dict(zip(rgf.REGIONS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
OUTLINE = "#00e5ff"  # the mask colour of the GIFs
LAND_COLOUR = "#e4e2dc"
TEXT = "#1f1f1e"
MUTED = "#6b6a63"


def trained_glims_ids() -> pd.Series:
    """Glaciers in the training set, from the training bucket's image names."""
    if not TRAINED_IDS.exists():
        import boto3

        bucket, prefix = TRAINING_BUCKET
        pages = boto3.client("s3", region_name="us-west-1").get_paginator(
            "list_objects_v2"
        ).paginate(Bucket=bucket, Prefix=prefix)
        names = [o["Key"].removeprefix(prefix) for p in pages for o in p.get("Contents", [])]
        ids = sorted({n.split("_")[0] for n in names if n.endswith(".tif")})
        pd.Series(ids, name="glims_id").to_csv(TRAINED_IDS, index=False)
        logger.info("wrote %s: %d glaciers", TRAINED_IDS, len(ids))
    return pd.read_csv(TRAINED_IDS)["glims_id"]


def studied_glaciers() -> gpd.GeoDataFrame:
    """Every trained-on glacier as a point, with its region."""
    selection = glims.add_region_rollup(glims.read_selection())
    studied = selection[selection["glac_id"].isin(trained_glims_ids())]
    points = studied.geometry.representative_point()
    return gpd.GeoDataFrame(
        {"glims_id": studied["glac_id"], "region": studied["geog_area_rollup"]},
        geometry=points, crs=selection.crs,
    ).to_crs(CRS)


def highlighted_glaciers() -> pd.DataFrame:
    """Per region, the largest relative loss among glaciers starting over `MIN_START_KM2`."""
    annual = pd.read_csv(rgf.ANNUAL_AREAS, index_col="Dates", parse_dates=["Dates"])
    change = rgf.relative_change(rgf.ANNUAL_AREAS)
    change["start_km2"] = [annual[g].dropna().iloc[: rgf.END_YEARS].mean() for g in change.index]
    change["end_km2"] = [annual[g].dropna().iloc[-rgf.END_YEARS:].mean() for g in change.index]

    info = pd.read_csv(rgf.FILE_LIST, usecols=["glims_id", "geog_area_rollup", "geog_area"])
    info = info.drop_duplicates("glims_id").set_index("glims_id").rename(
        columns={"geog_area_rollup": "region", "geog_area": "place"}
    )
    candidates = change.join(info)
    candidates = candidates[
        (candidates["ranked"] == True)  # noqa: E712 (NaN-safe)
        & (candidates["start_km2"] > MIN_START_KM2)
        & candidates["region"].isin(rgf.REGIONS)
    ]
    picked = candidates.sort_values("change").groupby("region").head(1).reset_index()
    picked["place"] = picked["place"].str.replace("Caucausus", "Caucasus")
    lon, lat = zip(*(_glims_lonlat(g) for g in picked["glims_id"]))
    picked["x"], picked["y"] = Transformer.from_crs(
        "EPSG:4326", CRS, always_xy=True
    ).transform(lon, lat)
    return picked.set_index("region").loc[[r for r in rgf.REGIONS if r in set(picked["region"])]]


def _glims_lonlat(glims_id: str) -> tuple[float, float]:
    """``G289972E33043S`` -> (-70.028, -33.043). Longitudes run 0-360 east."""
    lon = float(glims_id[1:7]) / 1000
    lat = float(glims_id[8:13]) / 1000 * (1 if glims_id[13] == "N" else -1)
    return (lon - 360 if lon > 180 else lon), lat


def _end_scene(data: dict, years: np.ndarray, target: float) -> int:
    """The scene in those years whose area is closest to their mean, ``target``."""
    candidates = np.flatnonzero(np.isin(data["years"], years))
    return int(candidates[np.argmin(np.abs(data["areas"][candidates] - target))])


def load_panels(picked: pd.DataFrame) -> dict[str, dict]:
    """Before and after scene indices for each highlighted glacier."""
    rgf.OUT_DIR.mkdir(parents=True, exist_ok=True)
    rgf.segment(list(picked["glims_id"]), device_name=None)
    panels = {}
    for region, row in picked.iterrows():
        data = dict(np.load(rgf.cache_path(row["glims_id"])))
        data["years"] = np.array([int(d[:4]) for d in data["dates"]])
        observed = np.unique(data["years"][data["years"] <= 2023])
        data["before"] = _end_scene(data, observed[: rgf.END_YEARS], row["start_km2"])
        data["after"] = _end_scene(data, observed[-rgf.END_YEARS:], row["end_km2"])
        panels[region] = data
    return panels


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #


def draw_map(ax, points: gpd.GeoDataFrame, picked: pd.DataFrame, land) -> list:
    """The map, without a legend; returns the legend handles."""
    land.plot(ax=ax, color=LAND_COLOUR, linewidth=0)
    for region, colour in COLOURS.items():
        sub = points[points["region"] == region]
        ax.scatter(sub.geometry.x, sub.geometry.y, s=1.2, color=colour,
                   linewidths=0, alpha=0.75, rasterized=True)
    ax.scatter(picked["x"], picked["y"], s=34, c=[COLOURS[r] for r in picked.index],
               edgecolors=TEXT, linewidths=0.9, zorder=5)

    to_xy = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    _, y0 = to_xy.transform(0, LAT_RANGE[0])
    _, y1 = to_xy.transform(0, LAT_RANGE[1])
    x_edge, _ = to_xy.transform(180, 0)
    ax.set_xlim(-x_edge, x_edge)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()

    counts = points["region"].value_counts()
    return [
        Line2D([], [], marker="o", linestyle="", markersize=5, color=COLOURS[r],
               label=f"{rgf.REGIONS[r]} ({counts.get(r, 0):,})")
        for r in rgf.REGIONS if counts.get(r, 0)
    ]


def add_legend(target, handles, n_points: int, **kwargs) -> None:
    target.legend(handles=handles, title=f"{n_points:,} glaciers studied",
                  fontsize=6.5, title_fontsize=7, frameon=False, borderaxespad=0.2,
                  handletextpad=0.2, labelcolor=TEXT, alignment="left", **kwargs)


#: Fixed sizes, in inches, inside a pair of panels.
TITLE_IN = 0.30
LABEL_IN = 0.27
GAP_IN = 0.10


def draw_pair(fig, rect, region: str, row: pd.Series, data: dict) -> dict:
    """Before and after panels, side by side and centred in ``rect``.

    The panels keep each scene's native aspect and sit directly under the
    title. Returns the group's edges in figure fraction: ``left``, ``right``,
    ``top`` (above the title), ``bottom`` (below the labels) and ``middle``
    (the panels' vertical centre).
    """
    left, bottom, width, height = rect
    fig_w, fig_h = fig.get_size_inches()
    before, after = data["before"], data["after"]
    aspects = [data["sizes"][i][1] / data["sizes"][i][0] for i in (before, after)]

    panel_h = height * fig_h - TITLE_IN - LABEL_IN
    panel_w = [panel_h * a for a in aspects]
    room = width * fig_w - GAP_IN
    if sum(panel_w) > room:  # too wide for the slot: shrink to fit
        panel_h *= room / sum(panel_w)
        panel_w = [panel_h * a for a in aspects]

    total_w = (sum(panel_w) + GAP_IN) / fig_w
    x = left + (width - total_w) / 2
    top = bottom + height
    panel_top = top - TITLE_IN / fig_h
    panel_bottom = panel_top - panel_h / fig_h
    for i, w in zip((before, after), panel_w):
        ax = fig.add_axes((x, panel_bottom, w / fig_w, panel_h / fig_h))
        _draw_panel(ax, region, data, i, outline_of=before if i == after else None)
        x += (w + GAP_IN) / fig_w

    centre = left + width / 2
    fig.text(centre, top, f"{rgf.REGIONS[region]} · {row['place']}",
             ha="center", va="top", fontsize=7.5, fontweight="bold", color=TEXT)
    fig.text(centre, top - 0.14 / fig_h,
             f"{row['glims_id']}   {row['change']:+.0%} ({row['from']:.0f}–{row['to']:.0f})",
             ha="center", va="top", fontsize=6.5, color=MUTED)
    return {
        "left": centre - total_w / 2,
        "right": centre + total_w / 2,
        "top": top,
        "bottom": panel_bottom - LABEL_IN / fig_h,
        "middle": panel_bottom + panel_h / fig_h / 2,
    }


def _draw_panel(ax, region: str, data: dict, i: int, outline_of: int | None) -> None:
    height_px, width_px = data["sizes"][i]
    ax.imshow(data["rgb"][i], extent=(0, width_px, height_px, 0),
              interpolation="bilinear", aspect="auto")
    xs = np.linspace(0, width_px, 128)
    ys = np.linspace(0, height_px, 128)
    prob = data["probabilities"][i].astype(np.float32)
    ax.contourf(xs, ys, prob, levels=[0.5, 1.01], colors=[OUTLINE], alpha=0.25)
    ax.contour(xs, ys, prob, levels=[0.5], colors=[OUTLINE], linewidths=0.8)
    # The earlier outline, only where both scenes cover the same footprint;
    # across a UTM zone change the pixel grids differ and it would misalign.
    if outline_of is not None and tuple(data["sizes"][i]) == tuple(data["sizes"][outline_of]):
        ax.contour(xs, ys, data["probabilities"][outline_of].astype(np.float32),
                   levels=[0.5], colors=["white"], linewidths=0.6,
                   linestyles=[(0, (2, 1.5))])
    ax.set_xlim(0, width_px)
    ax.set_ylim(height_px, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(COLOURS[region])
        spine.set_linewidth(1.6)
    ax.set_xlabel(f"{data['dates'][i]}\n{data['areas'][i]:.1f} km²",
                  fontsize=6, color=TEXT, labelpad=2)


def connect(fig, map_ax, row: pd.Series, region: str, xy_fig: tuple[float, float]) -> None:
    line = ConnectionPatch(
        xyA=xy_fig, coordsA="figure fraction",
        xyB=(row["x"], row["y"]), coordsB=map_ax.transData,
        color=COLOURS[region], linewidth=0.9, zorder=4,
    )
    fig.add_artist(line)


def _footnote(fig, y: float) -> None:
    fig.text(
        0.5, y,
        "Cyan: predicted glacier (probability > 0.5). Dashed white: the earliest outline, "
        "where both scenes share a footprint. Change: mean of the last three annual "
        "areas over the first three.",
        ha="center", va="bottom", fontsize=5.5, color=MUTED,
    )


def stacked(points, picked, panels, land, out_stem) -> None:
    """Map in the middle; northern glaciers above, the rest below, by longitude."""
    fig = plt.figure(figsize=(7.2, 6.3), facecolor="white")
    map_ax = fig.add_axes((0.02, 0.27, 0.96, 0.44))
    handles = draw_map(map_ax, points, picked, land)
    add_legend(map_ax, handles, len(points), loc="lower left", bbox_to_anchor=(0, 0.16))

    order = picked.sort_values("x")
    top = [r for r in order.index if r != "Caucausus" and order.loc[r, "y"] > 0]
    bottom = [r for r in order.index if r not in top]
    rows = (
        (top, 0.715, list(np.linspace(0.01, 0.68, len(top)))),
        (bottom, 0.02, [0.17, 0.55]),
    )
    for regions, y, lefts in rows:
        for region, left in zip(regions, lefts):
            row = picked.loc[region]
            box = draw_pair(fig, (left, y, 0.31, 0.27), region, row, panels[region])
            anchor_y = box["bottom"] - 0.005 if y > 0.5 else box["top"] + 0.005
            connect(fig, map_ax, row, region, (left + 0.155, anchor_y))
    _footnote(fig, 0.003)
    _save(fig, out_stem)


def side(points, picked, panels, land, out_stem) -> None:
    """Map on the left; one row per glacier on the right, north to south."""
    regions = list(picked.sort_values("y", ascending=False).index)
    fig = plt.figure(figsize=(7.2, 6.2), facecolor="white")
    map_ax = fig.add_axes((0.0, 0.30, 0.60, 0.42))
    handles = draw_map(map_ax, points, picked, land)
    # Bottom left: clear of the lines, which all run right from the map.
    add_legend(fig, handles, len(points), loc="lower left", bbox_to_anchor=(0.01, 0.05))

    top, bottom = 0.995, 0.035
    row_h = (top - bottom) / len(regions)
    for k, region in enumerate(regions):
        y = top - (k + 1) * row_h
        row = picked.loc[region]
        box = draw_pair(fig, (0.62, y + 0.004, 0.37, row_h - 0.008),
                        region, row, panels[region])
        connect(fig, map_ax, row, region, (box["left"] - 0.01, box["middle"]))
    _footnote(fig, 0.003)
    _save(fig, out_stem)


def _save(fig, out_stem) -> None:
    path = out_stem.with_suffix(".png")
    fig.savefig(path, dpi=DPI, facecolor="white")
    logger.info("wrote %s", path)
    plt.close(fig)


def main() -> int:
    configure()
    points = studied_glaciers()
    picked = highlighted_glaciers()
    logger.info("highlighted:\n%s", picked[["glims_id", "place", "change", "start_km2",
                                            "end_km2", "from", "to"]].to_string())
    panels = load_panels(picked)
    for region, data in panels.items():
        logger.info("%s: before %s, after %s", region,
                    data["dates"][data["before"]], data["dates"][data["after"]])
    land = gpd.read_file(LAND).to_crs(CRS)
    figures = rgf.REPO / "figures"
    stacked(points, picked, panels, land, figures / "global_overview_stacked")
    side(points, picked, panels, land, figures / "global_overview_side")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
