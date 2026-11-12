"""Score a checkpoint on the frozen holdout, one evaluation choice at a time.

The published model is reported at Dice 0.92; measured in this repo it came out
at 0.868. This script changes nothing and retrains nothing. It scores the same
predictions under each evaluation choice, so the gap can be attributed:

    inputs   normalize=True   the 2023 train.py's per-image min-max with +1
                              on the span (inputs top out near 0.5), which
                              `GlacierDataset` used to apply by default
             normalize=False  the [0, 1] stacks as stored, which is what
                              inference feeds the model
    metric   glacier          `losses.dice_score`, glacier pixels only,
                              averaged per batch (as `trainer.score` does)
             pooled           the same, over every pixel of the slice at once
             torchmetrics     `torchmetrics.Dice(average='micro')` on int
                              masks, exactly as the original train.py and
                              notebook scored; needs torchmetrics < 1.9

The holdout is data/analysis/holdout_seed0.csv: `build_manifest(seed=0)` over
the full training set, halved into eval and test as `glacierview train` does.
Its "test" half is the 359-image slice in docs/MODEL_MANIFEST.md. The
checkpoint almost certainly trained on some of these images (the original
split was never saved), so every score here is an upper bound.

Missing images and masks are fetched from s3://training-images-t1-l2-sr
(about 770 MB for all 718). Usage, from the repo root, with the AWS and
Hugging Face logins of scripts/run_inference_250.sh:

    uv run --with "torchmetrics==1.2.1" python scripts/score_checkpoint.py \\
        --data-dir <dir> [--checkpoint <path or repo id>] [--slice test]

Without the `--with`, the torchmetrics rows are skipped.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from glacierview import config
from glacierview.models import load_checkpoint
from glacierview.training.dataset import GlacierDataset
from glacierview.training.losses import dice_score, jaccard_index

HOLDOUT = Path("data/analysis/holdout_seed0.csv")
TRAINING_BUCKET = ("training-images-t1-l2-sr", "us-west-1")
DEFAULT_CHECKPOINT = "GlacierView/glacierview-unet"
BATCH_SIZES = (8, 16, 32)


def fetch_missing(manifest: pd.DataFrame, data_dir: Path) -> None:
    """Download whichever listed images and masks are not already on disk."""
    keys = sorted(set(manifest["img"]) | set(manifest["mask"]))
    missing = [k for k in keys if not (data_dir / k).exists()]
    if not missing:
        return
    client = boto3.client("s3", region_name=TRAINING_BUCKET[1])

    def fetch(key: str) -> None:
        target = data_dir / key
        target.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(TRAINING_BUCKET[0], key, str(target))

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(tqdm(pool.map(fetch, missing), total=len(missing), desc="fetching"))


@torch.no_grad()
def predict(model, dataset, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Binary predictions and targets for the whole slice, (N, H, W) each."""
    model.eval()
    preds, targets = [], []
    for imgs, segs in tqdm(DataLoader(dataset, batch_size=8), leave=False):
        logits = model(imgs.to(device))
        preds.append((torch.softmax(logits, dim=1)[:, 1] > config.PROB_THRESHOLD).float().cpu())
        targets.append(segs.squeeze(1).float())
    return torch.cat(preds), torch.cat(targets)


def batch_mean(fn, preds, targets, batch_size: int) -> float:
    """`fn` per batch, then averaged over batches, as the training loop does."""
    scores = [
        float(fn(preds[i:i + batch_size], targets[i:i + batch_size]))
        for i in range(0, len(preds), batch_size)
    ]
    return sum(scores) / len(scores)


def legacy_metrics():
    """The original Dice and Jaccard, or None if torchmetrics has dropped Dice."""
    try:
        from torchmetrics import Dice, JaccardIndex
    except ImportError:
        return None
    dice = Dice(average="micro", threshold=config.PROB_THRESHOLD)
    jaccard = JaccardIndex(task="multiclass", num_classes=2, threshold=config.PROB_THRESHOLD)
    # Masks are squeezed to match the predictions' (B, H, W); with micro
    # averaging that does not change the count.
    return (
        lambda p, t: dice(p.int(), t.int()),
        lambda p, t: jaccard(p, t.int()),
    )


def score_rows(preds, targets, normalize: bool) -> list[dict]:
    inputs = "normalize=True" if normalize else "normalize=False"
    rows = []
    for bs in BATCH_SIZES:
        rows.append({
            "inputs": inputs, "metric": "glacier", "batch": bs,
            "dice": batch_mean(dice_score, preds, targets, bs),
            "jaccard": batch_mean(jaccard_index, preds, targets, bs),
        })
    rows.append({
        "inputs": inputs, "metric": "pooled", "batch": len(preds),
        "dice": dice_score(preds, targets), "jaccard": jaccard_index(preds, targets),
    })
    legacy = legacy_metrics()
    if legacy is not None:
        for bs in BATCH_SIZES:
            rows.append({
                "inputs": inputs, "metric": "torchmetrics", "batch": bs,
                "dice": batch_mean(legacy[0], preds, targets, bs),
                "jaccard": batch_mean(legacy[1], preds, targets, bs),
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="holds images/ and masks/; missing files are fetched here")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help=f"file path or Hugging Face repo id (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--slice", choices=("test", "eval", "all"), default="test",
                        help="test = the 359 images in MODEL_MANIFEST.md (default)")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    manifest = pd.read_csv(HOLDOUT)
    if args.slice != "all":
        manifest = manifest[manifest["split"] == args.slice].reset_index(drop=True)
    fetch_missing(manifest, args.data_dir)

    model = load_checkpoint(args.checkpoint, device=args.device)
    rows = []
    for normalize in (True, False):
        dataset = GlacierDataset(manifest, args.data_dir, normalize=normalize)
        rows += score_rows(*predict(model, dataset, args.device), normalize)
    if legacy_metrics() is None:
        print('torchmetrics rows skipped: rerun with uv run --with "torchmetrics==1.2.1"')

    table = pd.DataFrame(rows)
    print(f"\n{args.checkpoint} on {len(manifest)} images ({args.slice})\n")
    print(table.to_string(index=False, float_format="{:.4f}".format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
