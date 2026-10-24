# Train the model

## When to use

To retrain or fine-tune the segmentation U-Net.

## Workflow

```bash
uv run glacierview train --epochs 10 --out-dir experiments/run1
# smoke test on the committed 32-pair fixture
uv run glacierview train --data-dir data/sample/training --epochs 1 --batch-size 4 --device cpu \
  --out-dir /tmp/smoke
```

`--data-dir` defaults to `config.training_set_dir()` (the `images/` + `masks/` set notebook 10
builds). Each run writes `model.state_dict.pt` and `run.json` (config, per-epoch losses, best epoch,
test Dice and Jaccard) to `--out-dir`.

## Match the published recipe

The CLI defaults *are* the published recipe — learning rate 1e-5, batch
size 32, unfrozen encoder, cross-entropy, a 90/5/5 split, and the epoch that
minimises eval loss. The old script defaulted to batch size 2, had no eval
split, and kept whatever the last epoch produced.

Two bugs in the old loop are fixed, and both change training behaviour:
`optimizer.zero_grad()` was never called, so gradients accumulated across a
whole epoch; and only the last batch's loss was recorded. Reproducing the
paper's exact numbers therefore needs the old script from git history.

## Notes

- Training data on disk is 7-channel; NDSI and NDWI bring it to the 9 the
  model takes. The loader checks this per image and names the offending file.
- Two tif vintages exist: the current set reads back as `(1, 128, 128, 7)`,
  an older one as `(128, 128, 8)`. The loader normalises both to channel-first.
- The paper reports test Dice 0.92; the published checkpoint measures 0.868 on a re-drawn holdout
  (`docs/MODEL_MANIFEST.md`). Full ablation trail in `.agents/references/paper.md`.

## References

- `.agents/context/model.md` — architecture and the checkpoint contract
- `docs/MODEL_MANIFEST.md` — what each existing checkpoint is
