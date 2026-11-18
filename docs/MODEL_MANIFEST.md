# Model checkpoint manifest

Generated 2026-09-19. Records what each checkpoint is, because nothing else on disk does.

**None of these files is in git** (`.gitignore` excludes `src/segmentation/saved_models/`).
All of them were backed up on 2026-09-19 to `s3://segmentation-model-gv/checkpoints/` (us-east-1).
The production checkpoint is also published, converted to safetensors, at
<https://huggingface.co/GlacierView/glacierview-unet>. See "Published copy" below.

## PyTorch checkpoints

Both take **9 input channels**, verified from the serialized tensor sizes (stem conv
`Conv2d(9,64,7,7)` = 112,896 B; `c5` = `conv_block(41,16)` = 23,616 B). Channel order:

```
0: NDWI  1: NDSI  2: blue  3: green  4: red  5: nir  6: swir  7: thermal  8: DEM
```

`swir_2` was deliberately excluded for quality reasons (per the paper's channel figure).

Saved with `torch.save(model)` — a **pickled module referencing `__main__.UNet` and**
**`__main__.conv_block`**, not a state dict. Both names must exist in `__main__` before
`torch.load`. Convert to a state dict before sharing these anywhere.

### `unet_summer_model_unfrozen_100`

- modified: 2024-10-02
- size: 343,522,287 bytes
- sha256: `9c29c41515bde39c1722c5bd18f5e82711fb54c86cd5f53d5262d9348623c2b8`
- input channels: 9
- Production checkpoint and the paper's model, confirmed by the project lead and empirically
  (Dice 0.8681 vs the 2023 checkpoint's 0.8261 on the same 359 images, measured with the old `+1`
  rescale; see "Updated method" below, which supersedes these figures. The paper's 0.92 is pixel
  accuracy, not glacier Dice).

### `saved_models/model`

- modified: 2023-12-27
- size: 343,487,293 bytes
- sha256: `da5296e839fa4e7e6fc601735273757b329531abee01839abcf4b8f6c8f60237`
- input channels: 9
- Earlier PyTorch checkpoint. Distinct weights from the above. Relationship to the paper unknown.

### Training recipe (the paper's description, corrected against the code where noted)

| Parameter | Value |
|---|---|
| architecture | U-Net, ResNet-50 encoder (ImageNet weights), last two ResNet layers removed |
| input | (N, 9, 128, 128); channels 2-8 normalized to [0,1], channels 0-1 (NDWI, NDSI) span about [-1,1] |
| output | 2-class logits; glacier probability = softmax(...)[:,1], threshold 0.5 |
| training set | 7,174 images (one single scene per glacier, not a composite); the paper gives 6,456 train / 359 eval / 359 test, the code gives 7,171 pairs (3 blank masks) → 6,453 / 359 / 359 |
| augmentation | vflip, hflip, 3x3 Gaussian blur (sigma 0.8), each independently at p=0.5, on the fly, replacing the sample (the paper's 11,298 images does not exist; the set does not grow) |
| learning rate | 0.00001 |
| batch size | 32 |
| optimizer | Adam |
| loss | cross-entropy |
| encoder | unfrozen (fine-tuned) |
| model selection | epoch minimizing eval-set cross-entropy |
| reported test Dice | 0.92 |

Area conversion: `mask.sum() * 0.0009` km² (30 m x 30 m pixels), after resizing the
prediction back to the source image's original dimensions (for the binary variant the 0/1 mask is
resized bilinearly, so the sum includes interpolated edge values).

## Legacy Keras checkpoints (`saved_models/*.h5`)

The TensorFlow era, 2022–2023, superseded by the PyTorch model. Retained as the record of
the band-selection experiments — the filenames encode which bands each used
(`bl`=blue, `gr`=green, `re`=red, `ni`=nir, `sw`=swir, `th`=thermal, `de`=DEM).
`re_ni_sw_de_v1.h5` is the file `glacierview infer`'s commented-out Keras block loaded.
The Keras code that built these (`helpers/model.py`, `inference/model.py`) is dead.

| File | Modified | Bytes | sha256 |
|---|---|---|---|
| `8_16_21-500t-6b-a.h5` | 2022-10-03 | 25,709,336 | `04e053ede418c563…` |
| `8_25_21-1136t-7b-a.h5` | 2022-10-03 | 25,711,064 | `2a89154377b35269…` |
| `8_25_21-1908t-7b-a.h5` | 2022-10-03 | 25,711,064 | `b1b75ccdf1d45c92…` |
| `bl_gr_re_ni_sw_th_de_v0.h5` | 2022-11-21 | 25,711,080 | `7af82771b0ce74e7…` |
| `bl_gr_re_ni_sw_th_de_v1.h5` | 2022-11-21 | 25,711,080 | `c611b557dbf60940…` |
| `bl_gr_re_ni_sw_th_de_v2.h5` | 2023-01-22 | 25,711,080 | `2fc6dfc77299c860…` |
| `four_band_model.h5` | 2022-10-03 | 25,705,784 | `535908751301b92a…` |
| `largest_ee.h5` | 2022-10-03 | 25,709,336 | `e6ce600fbcb12ea6…` |
| `largest_model.h5` | 2022-10-03 | 25,705,784 | `0f7596aeb6d96f34…` |
| `model_large.h5` | 2022-10-03 | 25,705,784 | `8f78063469600468…` |
| `no_23.h5` | 2022-10-03 | 25,705,784 | `be03428b4417b12f…` |
| `no_post_trient.h5` | 2022-10-03 | 25,705,784 | `6896d69fb00ff71e…` |
| `no_trient.h5` | 2022-10-03 | 25,705,784 | `9e1a4d05b9ffdf30…` |
| `re_ni_sw_de_v1.h5` | 2023-02-13 | 25,705,800 | `43fb5b70bb88168a…` |
| `test.h5` | 2022-10-23 | 25,709,352 | `ec6c68eaacc61bf3…` |
| `unaugmented_model.h5` | 2022-10-03 | 25,706,864 | `418e4ad1152b9ef7…` |
| `unaugmented_model_fixed_masks.h5` | 2022-10-03 | 25,705,784 | `10fc51f3e706eeaa…` |
| `with_trient.h5` | 2022-10-03 | 25,705,784 | `eb9c17da2b27afd0…` |

Full hashes: run `shasum -a 256 saved_models/*.h5`.

## Measured scores

Both checkpoints scored on the same 359-image slice (`build_manifest` with
`seed=0`, the second half of the 10% holdout), threshold 0.5. These scores used the old `+1` rescale
(now off by default); see "Updated method" below for the current figures:

| Checkpoint | Dice | Jaccard |
|---|---|---|
| `unet_summer_model_unfrozen_100` | **0.8681** | 0.7679 |
| `unet_2023-12-27` | 0.8261 | 0.7055 |

`unet_summer_model_unfrozen_100` is the published model — confirmed by the
project lead and consistent with these scores and with its filename, which
encodes the recipe (summer images, unfrozen encoder, 100 epochs).

⚠️ **Neither reaches the paper's reported 0.92, and batching does not explain
it.** Averaging Dice different ways over the same predictions:

| Averaging | Dice |
|---|---|
| per image | 0.8612 |
| batches of 8 | 0.8681 |
| batches of 32 (the paper's batch size) | 0.8713 |
| pooled over all pixels | 0.8692 |

All land at 0.86–0.87, so the ~0.05 gap is real rather than a measurement
convention. Note the gap should if anything be *smaller* than reported here:
these 359 images were almost certainly in the checkpoint's training data,
since the original script shuffled unseeded with no persistent split, so this
slice has no relationship to whatever it held out. That biases the score up.

Three candidate explanations, not distinguished:

1. The 0.92 was measured on a different test split, which was never recorded.
   Dice varies considerably with which glaciers land in the holdout.
2. The 0.92 came from a third checkpoint that no longer exists.
3. **The 0.92 is a different quantity.** The original script used
   `torchmetrics.Dice(average='micro')` on class indices, which counts
   background agreement as well as ice overlap and scores much higher than the
   mask-overlap Dice computed here. This is the leading suspicion, and it
   would mean the published figure is not comparable to what the current code
   reports.

## Updated method (2026-11-11): the gap is in the evaluation, not the model

> **This section supersedes the "Measured scores" conclusion above.** The
> earlier text is kept as the record of how 0.8681 was measured.

The 0.8681 above was measured with two choices that differ from how the model
is actually used and how the paper scored it. `scripts/score_checkpoint.py`
scores the same predictions under each choice. The slice is unchanged (the
"test" half of `data/analysis/holdout_seed0.csv`, the 359 images above), and
the script reproduces 0.8681 and 0.8261 exactly before changing anything.

1. **Input scaling.** `GlacierDataset` applied the 2023 `train.py`'s
   per-image min-max with `+1` added to the span, which puts the stored [0, 1]
   stacks at roughly [0, 0.5]. Inference never does this. Both checkpoints
   score clearly better without it, so neither was trained on it. The training
   default is now `normalize=False`; the flag remains to reproduce old runs.
2. **Metric.** On binary masks, `torchmetrics.Dice(average='micro')`
   (torchmetrics < 1.9, as the original code used) is **pixel accuracy**: it
   counts background agreement as well as ice overlap. That was verified
   against a hand computation. Candidate explanation 3 above is confirmed.

Dice on the 359-image slice, threshold 0.5, batches of 8:

| Checkpoint | Inputs | Glacier Dice (this repo) | `torchmetrics` micro Dice (= pixel accuracy) |
|---|---|---|---|
| `unet_summer_model_unfrozen_100` | with the `+1` rescale (old measurement) | 0.8681 | 0.9023 |
| `unet_summer_model_unfrozen_100` | **as at inference** | **0.9057** | **0.9309** |
| `unet_2023-12-27` | with the `+1` rescale (old measurement) | 0.8261 | 0.8862 |
| `unet_2023-12-27` | as at inference | 0.8683 | 0.9107 |

Batch size moves any figure by at most 0.004 (full grid, including Jaccard
and pooled scores, in the script's output).

What this means:

- **The published model's glacier-overlap Dice is about 0.906**, not 0.868.
  The paper's 0.92 is most likely the pixel-accuracy figure (0.931 here).
  The two numbers are not comparable, and the paper should say which it
  reports.
- **The caveat above still holds.** These images were probably in the training
  data, so all four figures are upper bounds. The paper's own test split
  cannot be rebuilt.
- **Inference and published areas are unaffected.** Inference never applied
  the rescale. Nothing needs retraining or re-running.

## Remaining unknowns

1. What changed between the 2023-12-27 and 2024-10-02 checkpoints.
2. Neither has recorded per-region metrics, training duration, or the exact training-set snapshot.

Ask Somansh Budhwar or Armin Schwartzman (armins@ucsd.edu).

## Published copy

`unet_summer_model_unfrozen_100` converted with `glacierview.models.export_pretrained`:

- repo: `GlacierView/glacierview-unet` (**private**), tag `v1-paper` → commit `7a00d6d`. Loading it
  needs `hf auth login` (or `HF_TOKEN` on a server) with read access to the GlacierView org.
- files: `model.safetensors` (weights only, no pickle), `config.json`, `README.md` (model card)
- `model.safetensors` sha256: `d9a878aa8b028c90d2859f28ab5ca4a495cf4089ab81df15f694a534b54c9e4e`
- verified 2026-10-23: every tensor equal to the pickle's; outputs bit-identical on random and real
  inputs; `glacierview infer` on Trient gives byte-identical results from both, including when
  downloaded from the Hub as `GlacierView/glacierview-unet@v1-paper`.

Load with `load_checkpoint("GlacierView/glacierview-unet")` or `UNet.from_pretrained(...)`.
