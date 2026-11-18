"""Run the model over a glacier's time series and turn masks into areas.

This is the shared core of both inference entry points. `infer` renders one
glacier; `areas` batches the whole landing zone. They used to be separate
scripts that had drifted into computing subtly different things — the
smoothing happened at a different point and one padded its input with a
duplicated channel — so the sequence lives here once.

Order of operations, as in the published method:

    sort by date -> preprocess to 128x128 -> smooth along time
    -> derive NDSI/NDWI -> model -> resize back -> count pixels

`glacierview.preprocess` describes the two preprocessing pipelines.
"current" is the default and needs a model trained on it. "published" feeds
the published checkpoint what it was trained on.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from skimage.filters import gaussian
from torch.utils.data import DataLoader, TensorDataset

from glacierview import config
from glacierview.log import get_logger
from glacierview.models import IN_CHANNELS
from glacierview.preprocess import (
    PREPROCESSING,
    add_spectral_indices,
    fill_missing,
    prepare_glacier_stack,
    prepare_glacier_stack_published,
    smooth_over_time,
)

logger = get_logger(__name__)

#: How an area is derived from the probability map. Three variants are kept
#: because the published analysis compared them.
AREA_VARIANTS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "raw": lambda p: p.clamp(min=0),
    "thresholded": lambda p: torch.where(
        p > config.PROB_THRESHOLD, p, torch.zeros_like(p)
    ),
    "binary": lambda p: (p > config.PROB_THRESHOLD).double(),
}


def dates_from_filenames(file_names: list[str]) -> list[datetime]:
    """Parse the acquisition date out of each filename.

    Filenames are load-bearing: ``{glims_id}_{date}_{L5|L7|L8}_...``, so token
    1 is the date.
    """
    return [datetime.strptime(f.split("_")[1], "%Y-%m-%d") for f in file_names]


def read_stack(
    glacier_dir: Path | str,
    dem_path: Path | str,
    filtered_file_path: Path | str | None = None,
    preprocessing: str = "current",
) -> tuple[np.ndarray, list[str], list[tuple[int, int]]]:
    """Read one glacier into its unsmoothed (N, 128, 128, C) stack.

    Returns the stack, the source filenames in date order, and the (height,
    width) each prediction is resized back to for area. For "current" that is
    the reference grid's shape; for "published" it is each scene's native size.
    "current" stacks keep missing pixels as NaN.
    """
    if preprocessing not in PREPROCESSING:
        raise ValueError(f"preprocessing must be one of {PREPROCESSING}, not {preprocessing!r}")
    if preprocessing == "published":
        return prepare_glacier_stack_published(
            glacier_dir, dem_path, filtered_file_path=filtered_file_path
        )
    stack, file_names, grid = prepare_glacier_stack(
        glacier_dir, dem_path, filtered_file_path=filtered_file_path
    )
    return stack, file_names, [grid.shape] * len(file_names)


def smooth_stack(
    stack: np.ndarray,
    preprocessing: str = "current",
    sigma: float | None = None,
    with_missing: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Smooth a `read_stack` stack along time, as the pipeline does, with nothing missing left.

    This is a time-only smooth: each pixel against itself on neighbouring dates,
    not a spatial blur.

    Args:
        sigma: in frames; 0 turns smoothing off. None, which is what the
            pipeline uses, means `config.TIME_SMOOTHING_SIGMA` for "current" and
            `config.PUBLISHED_TIME_SMOOTHING_SIGMA` (20) for "published".
        with_missing: also return an (N, H, W) bool mask of the pixels that had
            no band data even after smoothing, so were filled by `fill_missing`.
            The GIFs draw them black. Always all False for "published".
    """
    if sigma is None:
        sigma = (config.PUBLISHED_TIME_SMOOTHING_SIGMA if preprocessing == "published"
                 else config.TIME_SMOOTHING_SIGMA)
    if preprocessing == "published":
        smoothed = gaussian(stack, sigma=[sigma, 0, 0, 0], mode="reflect")
        return (smoothed, np.zeros(smoothed.shape[:3], dtype=bool)) if with_missing else smoothed
    smoothed = smooth_over_time(stack, sigma) if sigma > 0 else stack
    missing = np.isnan(smoothed[..., :-1]).any(axis=-1) if with_missing else None
    filled = fill_missing(smoothed, copy=sigma <= 0)
    return (filled, missing) if with_missing else filled


def model_inputs(stack: np.ndarray, preprocessing: str = "current") -> torch.Tensor:
    """An (N, H, W, C) stack as the model's (N, IN_CHANNELS, H, W) batch, NDSI/NDWI prepended."""
    inputs = add_spectral_indices(
        torch.from_numpy(stack).permute(0, 3, 1, 2), legacy=preprocessing == "published"
    )
    if inputs.shape[1] != IN_CHANNELS:
        raise ValueError(
            f"built {inputs.shape[1]} channels, model expects {IN_CHANNELS}"
        )
    return inputs


def build_inputs(
    glacier_dir: Path | str,
    dem_path: Path | str,
    filtered_file_path: Path | str | None = None,
    preprocessing: str = "current",
) -> tuple[torch.Tensor, np.ndarray, list[str], list[tuple[int, int]]]:
    """Read one glacier and return a model-ready batch.

    It runs `read_stack`, `smooth_stack` and `model_inputs` in order.

    Args:
        filtered_file_path: the quality-filter CSV; only scenes listed in it
            are used. None uses every scene on disk.
        preprocessing: "current" or "published"; see `glacierview.preprocess`.

    Returns:
        inputs: (N, IN_CHANNELS, 128, 128) float tensor.
        smoothed: the pre-index stack, kept for RGB rendering.
        file_names: source filenames in date order.
        sizes: the (height, width) each prediction is resized back to for
            area: the reference grid's for "current", and each scene's native
            size for "published".
    """
    stack, file_names, sizes = read_stack(
        glacier_dir, dem_path, filtered_file_path=filtered_file_path, preprocessing=preprocessing
    )
    smoothed = smooth_stack(stack, preprocessing)
    del stack  # gigabytes for a full series
    return model_inputs(smoothed, preprocessing), smoothed, file_names, sizes


def predict_probabilities(
    model, inputs: torch.Tensor, device: torch.device, batch_size: int = 64
) -> torch.Tensor:
    """Per-pixel glacier probability, as (N, 1, H, W) on the CPU."""
    batches = DataLoader(TensorDataset(inputs), batch_size=batch_size, shuffle=False)
    out = []
    with torch.no_grad():
        for (batch,) in batches:
            logits = model(batch.to(device=device, dtype=torch.float))
            out.append(torch.softmax(logits, dim=1)[:, 1].unsqueeze(1).cpu())
    return torch.cat(out, dim=0)


def areas_km2(
    probabilities: torch.Tensor, original_sizes: list[tuple[int, int]]
) -> list[float]:
    """Resize each prediction back to its 30 m grid and sum its area.

    Areas must be measured on that grid. The model works at 128x128, but only a
    30 m grid pixel represents 900 m^2.
    """
    out = []
    for prediction, size in zip(probabilities, original_sizes, strict=True):
        resized = torchvision.transforms.Resize(size, antialias=True)(prediction)
        out.append(float(resized.sum()) * config.KM2_PER_PIXEL)
    return out


def segment_glacier(
    model,
    glims_id: str,
    device: torch.device,
    data_label: str | None = None,
    variants: tuple[str, ...] = ("binary",),
    filtered_file_path: Path | str | None = None,
    preprocessing: str = "current",
) -> dict:
    """Segment one glacier end to end.

    `filtered_file_path` and `preprocessing` are as for `build_inputs`.

    Returns a dict with ``file_names``, ``dates``, ``probabilities``,
    ``smoothed``, ``missing`` (see `smooth_stack`) and an ``areas`` mapping of
    variant name to km^2 series.
    """
    stack, file_names, sizes = read_stack(
        config.glacier_dir(glims_id, data_label),
        config.dem_path(glims_id, data_label),
        filtered_file_path=filtered_file_path,
        preprocessing=preprocessing,
    )
    smoothed, missing = smooth_stack(stack, preprocessing, with_missing=True)
    del stack  # gigabytes for a full series
    inputs = model_inputs(smoothed, preprocessing)
    logger.info("%s: %d scenes", glims_id, len(file_names))

    probabilities = predict_probabilities(model, inputs, device)
    areas = {
        name: areas_km2(AREA_VARIANTS[name](probabilities), sizes) for name in variants
    }
    return {
        "glims_id": glims_id,
        "file_names": file_names,
        "dates": dates_from_filenames(file_names),
        "probabilities": probabilities,
        "smoothed": smoothed,
        "missing": missing,
        "areas": areas,
    }
