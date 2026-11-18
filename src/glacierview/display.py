"""True- and false-colour composites for looking at scenes. Display only, never model input.

The scene viewer (raw DNs) and the inference GIFs (the model's scaled stack)
both go through `composite`, so the same scene looks the same in each:

- Reflectance bands are clipped to `bands.SR_CUTOFF_DN` (reflectance 0 to 1)
  and scaled to [0, 1].
- A saturated visible band takes the brightest unsaturated visible band's value
  in that pixel, or full brightness if there is none. Snow is close to neutral
  in the visible, and Landsat 5/7's blue band saturates over snow while red and
  green often don't, which would otherwise tint snow blue or pink.
- Missing pixels are black.
- A gamma brightens the result.

Prefer `file_true_colour` over `stack_true_colour` where a glacier's reference
grid is known: the model's stack is squeezed into a 128x128 square, which
blurs it and distorts its shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from glacierview import bands, config, rasters

GAMMA = 0.7
VISIBLE = frozenset({"blue", "green", "red"})
TRUE_COLOUR = ("red", "green", "blue")
#: SWIR, NIR, green: snow and ice show cyan, clouds white, rock brown.
FALSE_COLOUR = ("swir", "nir", "green")
#: Landsat 5/7 write 65535 where a band saturates; Landsat 8 piles up at 65454.
SATURATED_DN = bands.SR_VALID_DN[1] - 1


def composite(
    channels: np.ndarray,
    saturated: np.ndarray,
    visible,
    missing: np.ndarray | None = None,
    gamma: float = GAMMA,
    flag_saturated: bool = False,
) -> np.ndarray:
    """An (H, W, 3) RGB image from three bands already scaled to [0, 1].

    Args:
        channels: (H, W, 3); NaN where missing.
        saturated: (H, W, 3) bool.
        visible: for each of the three bands, whether it is a visible band.
        missing: (H, W) bool, extra pixels to draw black.
        flag_saturated: paint any pixel with a saturated band magenta instead.
    """
    visible = np.asarray(visible, dtype=bool)
    donors = np.where(visible & ~saturated, channels, -np.inf).max(axis=-1, keepdims=True)
    replacement = np.where(np.isfinite(donors), donors, 1.0)
    out = np.where(saturated & visible, replacement, np.where(saturated, 1.0, channels))
    black = np.isnan(out).any(axis=-1)
    if missing is not None:
        black |= missing
    out = np.clip(np.nan_to_num(out), 0, 1) ** gamma
    out[black] = 0
    if flag_saturated:
        out[saturated.any(axis=-1)] = (1, 0, 1)
    return out


def scene_composite(
    scene: np.ndarray,
    satellite: str,
    names=TRUE_COLOUR,
    gamma: float = GAMMA,
    flag_saturated: bool = False,
) -> np.ndarray:
    """A composite from a whole exported scene: (19, H, W) DNs, as rasterio reads it.

    Fill (DN 0) is black; a band is saturated at DN `SATURATED_DN` and up.
    """
    dn = np.stack([scene[bands.landsat_bands[satellite][n]] for n in names], axis=-1)
    dn = dn.astype(float)
    low, high = bands.SR_CUTOFF_DN
    channels = np.clip((dn - low) / (high - low), 0, 1)
    channels[dn <= 0] = np.nan
    return composite(channels, dn >= SATURATED_DN, [n in VISIBLE for n in names],
                     gamma=gamma, flag_saturated=flag_saturated)


def stack_true_colour(
    frame: np.ndarray, missing: np.ndarray | None = None, gamma: float = GAMMA
) -> np.ndarray:
    """True colour from one frame of the model's stack: (H, W, C), `COMMON_BANDS` first.

    The stack is already clipped and scaled, so the original DNs are gone; a
    band at the top of its range (1) counts as saturated. That also catches
    reflectance above 1, which is clipped to 1 anyway.

    Args:
        missing: (H, W) bool, pixels with no data before gaps were filled.
    """
    channels = frame[..., [config.COMMON_BANDS.index(b) for b in TRUE_COLOUR]]
    return composite(channels, channels >= 1.0, [True] * 3, missing=missing, gamma=gamma)


def file_true_colour(path: Path | str, grid: rasters.Grid, gamma: float = GAMMA) -> np.ndarray:
    """True colour read straight from a scene file at 30 m on `grid`.

    Sharper and in proportion, unlike the model's 128x128 stack: this is what
    the scene viewer shows, put on the glacier's reference grid. Missing pixels
    (fill, or outside the scene's footprint) are black.
    """
    path = Path(path)
    satellite = path.name.split("_")[2].lower()
    indexes = [bands.landsat_bands[satellite][n] for n in TRUE_COLOUR]
    dn = rasters.read_scene(path, indexes, grid).astype(float)  # NaN where missing
    low, high = bands.SR_CUTOFF_DN
    channels = np.clip((dn - low) / (high - low), 0, 1)
    return composite(channels, dn >= SATURATED_DN, [True] * 3, gamma=gamma)
