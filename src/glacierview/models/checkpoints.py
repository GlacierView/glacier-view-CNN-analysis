"""Load and save model checkpoints.

Checkpoints in this project are ``torch.save(model)`` output — a pickled
module, not a state dict. That has one awkward consequence worth isolating in
one place: the pickle records where the class came from, and the released
checkpoints were written by a script, so they reference ``__main__.UNet`` and
``__main__.conv_block``. Loading one from anywhere that is not that original
script fails with::

    AttributeError: Can't get attribute 'UNet' on <module '__main__'>

`load_checkpoint` injects the names into ``__main__`` before unpickling, which
is why callers no longer need to import the classes for their side effect.

The published weights are a safetensors state dict on the Hugging Face Hub
(`HF_REPO_ID`), which carries no class references and no
arbitrary-code-execution risk. `load_checkpoint` takes either form;
`export_pretrained` turns a pickled checkpoint into the published one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from glacierview.log import get_logger
from glacierview.models.unet import HF_REPO_ID, IN_CHANNELS, UNet, conv_block

logger = get_logger(__name__)


def _inject_legacy_names() -> None:
    """Make ``__main__.UNet`` / ``__main__.conv_block`` resolvable."""
    main = sys.modules["__main__"]
    for name, obj in (("UNet", UNet), ("conv_block", conv_block)):
        if not hasattr(main, name):
            setattr(main, name, obj)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> UNet:
    """Load a checkpoint and put it in eval mode.

    ``path`` is one of:

    * a pickled-module checkpoint file (the original format);
    * a directory written by `export_pretrained` / ``save_pretrained``;
    * a Hugging Face repo id such as ``GlacierView/glacierview-unet``, optionally
      pinned to a tag or commit as ``GlacierView/glacierview-unet@v1-paper``.

    Raises:
        FileNotFoundError: with the searched path, rather than torch's less
            helpful error.
    """
    local = Path(path)
    if local.is_dir() or (not local.exists() and _looks_like_repo_id(str(path))):
        repo_id, _, revision = str(path).partition("@")
        model = UNet.from_pretrained(
            repo_id, revision=revision or None, map_location=str(device), strict=True
        )
        model.to(device).eval()
        logger.info("loaded %s onto %s", path, device)
        return model

    path = local
    if not path.exists():
        raise FileNotFoundError(
            f"no checkpoint at {path}. Pass a Hugging Face repo id such as "
            f"{HF_REPO_ID}, or see docs/MODEL_MANIFEST.md for the original files"
        )
    _inject_legacy_names()
    model = torch.load(path, map_location=device, weights_only=False)
    model.to(device).eval()

    channels = model.resnet[0].weight.shape[1]
    if channels != IN_CHANNELS:
        logger.warning(
            "checkpoint %s takes %d input channels but this code builds %d",
            path.name, channels, IN_CHANNELS,
        )
    logger.info("loaded %s (%d input channels) onto %s", path.name, channels, device)
    return model


def save_state_dict(model: UNet, path: str | Path) -> Path:
    """Write weights only — portable, and safe to share."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info("wrote state dict to %s", path)
    return path


def _looks_like_repo_id(text: str) -> bool:
    """``owner/name`` or ``owner/name@revision``, with no other path parts."""
    parts = text.partition("@")[0].split("/")
    return len(parts) == 2 and all(parts)


#: Facts about the inputs and outputs, written into the published config.json
#: next to the constructor arguments. `from_pretrained` ignores them.
MODEL_CONTRACT = {
    "in_channels": IN_CHANNELS,
    "channel_order": ["ndwi", "ndsi", "blue", "green", "red", "nir", "swir", "thermal", "dem"],
    "input_size": [128, 128],
    "glacier_class": 1,
    "threshold": 0.5,
    "km2_per_pixel": 0.0009,
}


def export_pretrained(checkpoint: str | Path, out_dir: str | Path) -> Path:
    """Convert a pickled checkpoint to the published safetensors format.

    Writes ``model.safetensors`` and ``config.json`` into ``out_dir``.
    """
    pickled = load_checkpoint(checkpoint)
    model = UNet(n_class=getattr(pickled, "n_class", 2), pretrained_encoder=False)
    model.load_state_dict(pickled.state_dict(), strict=True)
    config = {"n_class": model.n_class, "freeze_encoder": True,
              "pretrained_encoder": False, **MODEL_CONTRACT}
    model.save_pretrained(out_dir, config=config)
    logger.info("exported %s to %s", Path(checkpoint).name, out_dir)
    return Path(out_dir)
