"""Turn raw Landsat rasters into model-ready input.

Both inference entry points (``glacierview infer`` and ``glacierview areas``)
and the training-set build share this module. A change here therefore affects
all three equally, and that is the point. There are two pipelines:

**current** (the default; a model must be retrained on it):

    per glacier: one reference grid, from the DEM's extent
    per scene:   read the six bands onto the grid, fill -> NaN
                 -> clip to fixed cut-offs and scale -> NaN-aware resize to 128x128
    DEM:         read onto the grid -> per-raster stretch -> resize
    stack:       NaN-aware smoothing along time -> fill what is still missing
                 -> prepend NDSI and NDWI

**published** (what the published checkpoint was trained and run on):

    read each scene whole, on its own grid -> select bands -> per-scene stretch
    -> resize to 128x128 -> stack the DEM on -> smooth along time
    -> prepend NDSI and NDWI (the old formula)

`prepare_glacier_stack` and `prepare_glacier_stack_published` cover everything
up to the DEM stack. `smooth_over_time`, `fill_missing` and
`add_spectral_indices` finish it, and `inference.predict.build_inputs` runs the
whole sequence.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
import torchvision
from scipy.ndimage import gaussian_filter1d

from glacierview import config
from glacierview import rasters as raster_io
from glacierview.bands import CUTOFFS_DN, landsat_bands
from glacierview.config import (
    COMMON_BANDS,
    INDEX_EPSILON,
    MODEL_INPUT_DIM,
)

#: The preprocessing pipelines `inference.predict.build_inputs` can run.
PREPROCESSING = ("current", "published")
#: Below this share of valid input under it, a resized or time-smoothed pixel
#: counts as missing. The threshold only guards the division: any real support
#: gives a usable average.
MIN_VALID_WEIGHT = 1e-3


# --------------------------------------------------------------------------- #
# Band selection
# --------------------------------------------------------------------------- #

def satellite_of(file_name: str) -> str:
    """``l5``, ``l7`` or ``l8``, from token 2 of a scene's filename."""
    return file_name.split("_")[2].lower()


def band_indexes(satellite: str, band_names: list[str] | tuple[str, ...]) -> list[int]:
    """Positions of `band_names` in one satellite's files, in the order given."""
    bands = landsat_bands[satellite]
    return [bands[b] for b in band_names]


def get_common_bands(
    rasters: dict[str, np.ndarray], common_bands: list[str]
) -> dict[str, np.ndarray]:
    """Keep only `common_bands`, in that order, so every satellite yields the same channels.

    Landsat 5/7 and Landsat 8 order their bands differently. Landsat 8 has an
    extra coastal-aerosol band first, so the band index for e.g. `swir` depends
    on the satellite. The satellite is read from the third underscore-delimited
    field of the filename.
    """
    selected = {}
    for file_name, raster in rasters.items():
        satellite = satellite_of(file_name)
        expected = len(landsat_bands[satellite])
        if raster.shape[2] != expected:
            raise ValueError(
                f"{file_name}: {raster.shape[2]} bands, expected {expected} for {satellite}"
            )
        selected[file_name] = raster[:, :, band_indexes(satellite, common_bands)]
    return selected


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #

def normalize_rasters(rasters: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Scale each band to [0, 1] between its second-smallest and second-largest values.

    This is deliberately not a plain min-max. A band's most extreme values are
    usually artifacts, such as fill, dead pixels or saturation. So the single
    smallest and single largest *distinct* values are dropped as bounds and
    clamped to their neighbours. That removes one artifact value at each end,
    however many pixels hold it (Landsat 5 blue saturates on 57% of pixels).
    It does not help if two different extreme values are both artifacts.

    NaN pixels are ignored and stay NaN.
    """
    normalized = {}
    for file_name, raster in rasters.items():
        out = raster.astype(np.double, copy=True)
        for i in range(out.shape[2]):
            band = out[:, :, i]
            valid = band[~np.isnan(band)]
            if valid.size == 0:
                continue
            low, high = valid.min(), valid.max()

            if low == high:
                # Uniform band: nothing to stretch. Scale to 1 if non-zero.
                if low != 0:
                    out[:, :, i] = band / low
                continue

            distinct = np.unique(valid)
            if distinct.size < 4:
                # Too few values for the second-extreme trick; fall back.
                second_low, second_high = low, high
            else:
                second_low, second_high = distinct[1], distinct[-2]

            clamped = np.clip(band, second_low, second_high)
            out[:, :, i] = (clamped - second_low) / (second_high - second_low)
        normalized[file_name] = out
    return normalized


def scale_to_cutoffs(
    rasters: dict[str, np.ndarray], band_names: list[str]
) -> dict[str, np.ndarray]:
    """Clip each band to its fixed `CUTOFFS_DN` range and scale it to [0, 1].

    Unlike `normalize_rasters`, this puts every scene on the same scale.
    `band_names` gives the bands on the last axis, in order. Reflectance bands
    come out equal to surface reflectance, clipped to 0-1. Saturated pixels land
    at 1, and NaN (missing) stays NaN. Raw DN 0 fill lands at 0, which is why the
    current pipeline turns fill into NaN before it gets here.
    """
    low, high = np.array([CUTOFFS_DN[b] for b in band_names], dtype=np.double).T
    return {
        file_name: (np.clip(raster, low, high) - low) / (high - low)
        for file_name, raster in rasters.items()
    }


# --------------------------------------------------------------------------- #
# Resizing
# --------------------------------------------------------------------------- #

def resize_rasters(rasters: dict[str, np.ndarray], dim: tuple[int, int]) -> dict[str, np.ndarray]:
    """Resize every raster to `dim`, preserving the HWC layout.

    Antialiasing is set explicitly. torchvision changed its default for tensors
    in 0.17, and the dependency floor is 0.16.
    """
    resizer = torchvision.transforms.Resize(dim, antialias=True)
    resized = {}
    for file_name, raster in rasters.items():
        chw = torch.from_numpy(np.ascontiguousarray(raster)).permute(2, 0, 1)
        resized[file_name] = resizer(chw).permute(1, 2, 0).numpy()
    return resized


def resize_masked(array: np.ndarray, dim: tuple[int, int]) -> np.ndarray:
    """Resize an HWC float array to `dim`, averaging only its non-NaN pixels.

    Each output pixel is the weighted mean of the valid pixels under it (a
    normalised convolution), so fill doesn't drag real values towards zero. An
    output pixel with no valid pixels under it is NaN.
    """
    valid = ~np.isnan(array)
    resized = resize_rasters(
        {"values": np.where(valid, array, 0).astype(array.dtype),
         "weights": valid.astype(array.dtype)},
        dim,
    )
    weights = resized["weights"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = resized["values"] / weights
    out[weights < MIN_VALID_WEIGHT] = np.nan
    return out


# --------------------------------------------------------------------------- #
# The current pipeline
# --------------------------------------------------------------------------- #

def prepare_scene(path: Path | str, grid: raster_io.Grid) -> np.ndarray:
    """One scene on `grid`, as (128, 128, len(COMMON_BANDS)) float32, NaN where missing."""
    path = Path(path)
    indexes = band_indexes(satellite_of(path.name), COMMON_BANDS)
    raw = raster_io.read_scene(path, indexes, grid)
    scaled = scale_to_cutoffs({path.name: raw}, list(COMMON_BANDS))[path.name]
    return resize_masked(scaled.astype(np.float32), MODEL_INPUT_DIM)


def prepare_dem(dem_path: Path | str, grid: raster_io.Grid) -> np.ndarray:
    """A glacier's DEM on `grid`, as (128, 128, 1) float32 with no NaN.

    The DEM has no fixed cut-offs, so it keeps the per-raster stretch. It is
    stretched at 30 m and then resized, like the scenes, in training and at
    inference alike.
    """
    dem = raster_io.read_dem(dem_path, grid)
    dem = normalize_rasters({"dem": dem})["dem"]
    dem = fill_missing(dem[np.newaxis])[0].astype(np.float32)
    return resize_rasters({"dem": dem}, MODEL_INPUT_DIM)["dem"]


def _no_scenes_message(glacier_dir: Path, filtered_file_path) -> str:
    total = len(list(glacier_dir.glob("*.tif")))
    if total == 0:
        return f"no .tif files in {glacier_dir}"
    reasons = [f"dated before {config.FIRST_VALID_DATE:%Y-%m-%d}"]
    if filtered_file_path:
        reasons.append(f"not listed in {filtered_file_path}")
    return f"all {total} scenes in {glacier_dir} were excluded ({' or '.join(reasons)})"


def prepare_glacier_stack(
    glacier_dir: Path | str,
    dem_path: Path | str,
    filtered_file_path: Path | str | None = None,
) -> tuple[np.ndarray, list[str], raster_io.Grid]:
    """Read one glacier's scenes onto a single grid and return the stack.

    Scenes before `config.FIRST_VALID_DATE` are skipped. Each scene is read,
    scaled and shrunk to 128x128 before the next is opened, so memory holds only
    the 128x128 stack. Nothing is read at full resolution all at once.

    Returns:
        stack: (N, 128, 128, len(COMMON_BANDS) + 1) float32, date-ordered, DEM
            last. Missing band pixels are NaN; the DEM has none.
        file_names: the source filenames in the same order.
        grid: the reference grid every scene was resampled onto. Predictions are
            resized back to ``grid.shape`` before areas are measured.
    """
    glacier_dir = Path(glacier_dir)
    names = raster_io.list_scenes(
        glacier_dir, filtered_file_path=filtered_file_path, since=config.FIRST_VALID_DATE
    )
    if not names:
        raise FileNotFoundError(_no_scenes_message(glacier_dir, filtered_file_path))

    grid = raster_io.reference_grid(dem_path)
    dem = prepare_dem(dem_path, grid)
    stack = np.empty((len(names), *MODEL_INPUT_DIM, len(COMMON_BANDS) + 1), dtype=np.float32)
    for i, name in enumerate(names):
        stack[i, :, :, :-1] = prepare_scene(glacier_dir / name, grid)
        stack[i, :, :, -1:] = dem
    return stack, names, grid


def smooth_over_time(stack: np.ndarray, sigma: float = config.TIME_SMOOTHING_SIGMA) -> np.ndarray:
    """Gaussian-smooth an (N, H, W, C) stack along time, ignoring NaN.

    Each pixel is smoothed against itself on neighbouring dates; this is not a
    spatial blur. Missing values are left out of the average rather than
    counted as zero. That way a Landsat 7 stripe or a thermal gap is filled from
    the dates around it. A pixel with no valid value on any nearby date stays
    NaN.

    The filter is the same as the published pipeline's: sigma in frames, reflect
    at the ends, truncated at 4 sigma. It runs one channel at a time to keep
    memory down.
    """
    out = np.empty_like(stack)
    for c in range(stack.shape[-1]):
        channel = stack[..., c]
        valid = ~np.isnan(channel)
        values = gaussian_filter1d(np.where(valid, channel, 0), sigma, axis=0, mode="reflect")
        weights = gaussian_filter1d(valid.astype(stack.dtype), sigma, axis=0, mode="reflect")
        with np.errstate(divide="ignore", invalid="ignore"):
            smoothed = values / weights
        smoothed[weights < MIN_VALID_WEIGHT] = np.nan
        out[..., c] = smoothed
    return out


def fill_missing(stack: np.ndarray, copy: bool = True) -> np.ndarray:
    """Replace NaN in an (N, H, W, C) stack so that nothing missing reaches the model.

    A NaN takes the mean of the valid pixels in its own frame and channel. If
    that whole frame-channel is missing, it takes the channel's mean over the
    stack, and 0 if the channel is missing everywhere. At inference this runs
    after `smooth_over_time`, so it only touches pixels that no nearby date
    covers. It works one channel at a time, to keep memory down.

    Args:
        copy: False fills `stack` in place and returns it. That saves a copy of a
            series that can run to gigabytes, when the caller no longer needs the
            NaN.
    """
    out = stack.copy() if copy else stack
    for c in range(out.shape[-1]):
        channel = out[..., c]
        missing = np.isnan(channel)
        if not missing.any():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # mean of an all-NaN slice
            frame_means = np.nanmean(channel, axis=(1, 2))
        empty = np.isnan(frame_means)
        if empty.any():
            channel_mean = 0.0 if missing.all() else np.nanmean(channel)
            frame_means = np.where(empty, channel_mean, frame_means)
        frames, _, _ = np.nonzero(missing)
        channel[missing] = frame_means[frames]
    return out


# --------------------------------------------------------------------------- #
# The published pipeline
# --------------------------------------------------------------------------- #

def prepare_glacier_stack_published(
    glacier_dir: Path | str,
    dem_path: Path | str,
    filtered_file_path: Path | str | None = None,
) -> tuple[np.ndarray, list[str], list[tuple[int, int]]]:
    """Read one glacier's scenes the way the published checkpoint saw them.

    Each scene stays on its own grid and gets a per-scene stretch
    (`normalize_rasters`). Scenes and DEM are resized independently. The DEM is
    resized and then stretched, while training stretched it and then resized
    it. Fill and saturation are not masked. Use this only to run the published
    checkpoint as it was run, or to reproduce published numbers. Scenes before
    `config.FIRST_VALID_DATE` are still skipped.

    Returns:
        stack: (N, 128, 128, len(COMMON_BANDS) + 1) float64, DEM last.
        file_names: the source filenames in the same order.
        original_sizes: each scene's native (height, width), so predictions can
            be resized back before areas are measured.
    """
    glacier_dir = Path(glacier_dir)
    images = raster_io.get_rasters(
        glacier_dir, filtered_file_path=filtered_file_path, since=config.FIRST_VALID_DATE
    )
    if not images:
        raise FileNotFoundError(_no_scenes_message(glacier_dir, filtered_file_path))

    file_names = sorted(images)
    original_sizes = [images[f].shape[:2] for f in file_names]

    images = get_common_bands(images, list(COMMON_BANDS))
    images = normalize_rasters(images)
    images = resize_rasters(images, MODEL_INPUT_DIM)

    dem = raster_io.get_dem(dem_path)
    dem_key = next(iter(dem))
    dem = normalize_rasters(resize_rasters(dem, MODEL_INPUT_DIM))

    stack = np.stack([
        np.concatenate((images[f], dem[dem_key]), axis=2) for f in file_names
    ])
    return stack, file_names, original_sizes


# --------------------------------------------------------------------------- #
# Spectral indices
# --------------------------------------------------------------------------- #

def add_spectral_indices(inputs: torch.Tensor, legacy: bool = False) -> torch.Tensor:
    """Prepend NDSI then NDWI to an (N, C, H, W) batch whose channels are COMMON_BANDS (+ DEM).

    Each index is prepended in turn, so NDWI ends up at channel 0 and NDSI at
    1, which is the order the model was trained on.

    NDSI (snow) is high for ice: bright in green, dark in shortwave infrared.
    NDWI (water) is high for meltwater, a common false positive.

    Args:
        legacy: the published formula, which adds epsilon to the numerator as
            well, ``(a - b + eps) / (a + b + eps)``. It turns 0/0 (fill, or
            reflectance clipped to 0 in both bands) into +1, the strongest snow
            and water signal possible. The current formula gives 0 there.
    """
    eps = INDEX_EPSILON
    green, nir, swir = (inputs[:, COMMON_BANDS.index(b)] for b in ("green", "nir", "swir"))
    top_eps = eps if legacy else 0.0

    # Filled in place rather than concatenated twice: a full time series is
    # gigabytes, and each concatenation would copy all of it.
    n, c, h, w = inputs.shape
    out = inputs.new_empty((n, c + 2, h, w))
    out[:, 2:] = inputs
    out[:, 1] = (green - swir + top_eps) / (green + swir + eps)  # NDSI
    out[:, 0] = (green - nir + top_eps) / (green + nir + eps)  # NDWI
    return out
