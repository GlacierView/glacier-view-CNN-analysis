"""Render predictions as GIFs and area plots."""

from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # headless: these run on servers and in batch
import matplotlib.pyplot as plt

from glacierview import config, display, glims
from glacierview import rasters as raster_io
from glacierview.log import get_logger
from glacierview.preprocess import add_spectral_indices, resize_rasters
from glacierview.training.build import rasterize_outline

logger = get_logger(__name__)

#: Keep every Nth frame; a 460-scene series is too long to animate whole.
DEFAULT_FRAME_STRIDE = 5

_SWIR = config.COMMON_BANDS.index("swir")
_THERMAL = config.COMMON_BANDS.index("thermal")


@lru_cache(maxsize=1)
def _outlines():
    """GLIMS outlines by ID, read once per process (a batch run renders hundreds)."""
    outlines = glims.read_selection().set_index("glac_id").geometry
    return outlines[~outlines.index.duplicated()]


def ground_truth_mask(glims_id: str, dem_path: Path | str) -> np.ndarray | None:
    """The glacier's GLIMS outline as a (128, 128) mask, built like a training label.

    It is rasterised on the glacier's reference grid, so it lines up with
    "current" preprocessing only. None if the GLIMS selection (notebook 01)
    isn't on disk or doesn't list the glacier.
    """
    if not config.glims_selection_path().exists():
        logger.warning("no GLIMS selection at %s; GIF has no ground truth",
                       config.glims_selection_path())
        return None
    outlines = _outlines()
    if glims_id not in outlines.index:
        logger.warning("%s is not in the GLIMS selection; GIF has no ground truth", glims_id)
        return None
    mask = rasterize_outline(raster_io.reference_grid(dem_path), outlines[glims_id])
    resized = resize_rasters({"mask": mask[..., np.newaxis].astype(np.float32)},
                             config.MODEL_INPUT_DIM)
    return resized["mask"][..., 0]


def write_gif(
    gif_path: Path,
    scratch_dir: Path,
    images,
    probabilities,
    file_names: list[str],
    stride: int = DEFAULT_FRAME_STRIDE,
    overlay: bool = False,
    ground_truth: np.ndarray | None = None,
    areas: list[float] | None = None,
    missing=None,
    scenes_dir: Path | None = None,
    grid: raster_io.Grid | None = None,
) -> Path:
    """Animate the prediction over the series.

    Each frame shows true colour, the prediction, the ground truth and the DEM,
    then NDSI, SWIR and thermal: the inputs the model leans on most.

    Args:
        images: (N, 128, 128, C) stack, `COMMON_BANDS` then the DEM.
        overlay: if True, draw the mask on top of the RGB composite (the batch
            style); otherwise show it on its own.
        ground_truth: a (128, 128) mask from `ground_truth_mask`; None leaves
            that panel blank.
        areas: each scene's area in km^2, shown in the frame's title.
        missing: (N, 128, 128) bool, pixels with no data before gaps were
            filled (`segment_glacier`'s ``missing``); drawn black in true colour.
        scenes_dir, grid: the glacier's scene directory and reference grid
            ("current" preprocessing only). With both, true colour is read from
            each scene file at 30 m, and every panel is drawn in the grid's
            proportions instead of the model's 128x128 square, which blurs the
            scene and distorts its shape. That true colour is the scene as the
            satellite saw it, before any time smoothing.

    True colour goes through `glacierview.display`, as in the scene viewer.
    """
    scratch_dir.mkdir(parents=True, exist_ok=True)
    for stale in scratch_dir.iterdir():
        stale.unlink()

    masks = probabilities.numpy()
    native = scenes_dir is not None and grid is not None
    # Stretch the 128x128 panels back over the grid's true shape.
    extent = (0, grid.shape[1], grid.shape[0], 0) if native else None
    # Every frame of one GIF must be the same size; size panels to the glacier.
    aspect = min(max(grid.shape[1] / grid.shape[0], 0.5), 2.0) if native else 1.0
    figsize = (4 * 4 * aspect, 2 * 4 + 0.5)
    for i in range(0, len(file_names), stride):
        frame = images[i]
        if native:
            rgb = display.file_true_colour(Path(scenes_dir) / file_names[i], grid)
        else:
            rgb = display.stack_true_colour(frame, None if missing is None else missing[i])
        ndsi = add_spectral_indices(torch.from_numpy(frame).permute(2, 0, 1)[None])[0, 1]
        fig, axs = plt.subplots(2, 4, figsize=figsize)
        title = file_names[i] if areas is None else f"{file_names[i]}: {areas[i]:.2f} km$^2$"
        fig.suptitle(title)
        axs[0, 0].imshow(rgb, extent=extent)
        axs[0, 0].set_title("true colour")
        if overlay:
            axs[0, 1].imshow(rgb, extent=extent)
            axs[0, 1].imshow(masks[i, 0], alpha=0.8, extent=extent)
        else:
            axs[0, 1].imshow(masks[i, 0], vmin=0, vmax=1, extent=extent)
        axs[0, 1].set_title("prediction")
        if ground_truth is not None:
            axs[0, 2].imshow(ground_truth, vmin=0, vmax=1, extent=extent)
        axs[0, 2].set_title("ground truth (GLIMS)" if ground_truth is not None
                            else "no ground truth")
        axs[0, 3].imshow(frame[:, :, -1], cmap="terrain", extent=extent)
        axs[0, 3].set_title("DEM")
        axs[1, 0].imshow(ndsi.numpy(), cmap="RdBu", vmin=-1, vmax=1, extent=extent)
        axs[1, 0].set_title("NDSI")
        axs[1, 1].imshow(frame[:, :, _SWIR], cmap="gray", vmin=0, vmax=1, extent=extent)
        axs[1, 1].set_title("SWIR")
        axs[1, 2].imshow(frame[:, :, _THERMAL], cmap="inferno", vmin=0, vmax=1, extent=extent)
        axs[1, 2].set_title("thermal")
        for ax in axs.flat:
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(scratch_dir / f"{file_names[i]}.png", dpi=100)
        plt.close(fig)

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(gif_path, mode="I") as writer:
        for frame in sorted(scratch_dir.iterdir()):
            writer.append_data(imageio.imread(frame))
    logger.info("wrote %s", gif_path)
    return gif_path


def plot_area_series(out_path: Path, dates, areas, title: str) -> Path:
    """Plot a km^2 time series using the project's shared style."""
    style = config.mpl_style()
    with plt.style.context(str(style)) if style.exists() else nullcontext():
        fig, ax = plt.subplots()
        ax.plot(dates, areas)
        ax.set_title(title)
        ax.set_ylabel("km$^2$")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path

