# Inference sample

Two parts: Trient's full time series, and one scene each for 100 other glaciers.

## Trient glacier

Every Landsat scene Earth Engine returned for Trient glacier (`G007026E45991N`, Swiss Alps), plus
its DEM, committed so you can run inference without downloading anything from S3.

- **Scenes:** 1,748 GeoTIFFs, 1984-04-09 to 2023-11-15, from Landsat 5, 7 and 8 (Collection 2, Tier 1,
  Level 2 surface reflectance). About 708 MB.
- **DEM:** `G007026E45991N_NASADEM.tif`.
- **Source:** the Trient folder of the full time-series pull, copied unmodified. The same files are
  in `s3://full-time-series-images-t1-l2-sr/G007026E45991N/` (us-west-1). Landsat and NASADEM are
  US-government data in the public domain.

The tree mirrors the real data root, so point `GV_DATA_ROOT` here:

```bash
GV_DATA_ROOT=data/sample/inference uv run glacierview infer \
  --glims-id G007026E45991N --checkpoint GlacierView/glacierview-unet --out-dir /tmp/trient
```

Expected result: `G007026E45991N: 1748 scenes, 7.49 -> 6.31 km^2`.

Pass `--out-dir`. Without it, outputs go under `$GV_DATA_ROOT/segmentation`, which is gitignored.

These are all scenes, not the quality-filtered subset the paper's analysis used, so the area series
is noisier than the published one. The paper's Trient figure used 464 scenes chosen with its own
looser filters.

## 100 more glaciers, one scene each

`glaciers.csv` lists 100 more glaciers, 20 from each region (Asia, Caucausus, Europe, North America,
South America), drawn at random (seed 0) from the 245 in the paper's inference run. Each has:

- **The training scene:** `landsat/<glims_id>/` holds the summer scene the model was trained on for
  that glacier. It's copied unmodified from the training pull (`localized_time_series_for_training_c02_t1_l2`).
- **The DEM:** `dems/<glims_id>_NASADEM.tif`.
- **The label:** `masks/<glims_id>.tif`, the glacier's GLIMS outline rasterised at 128×128. It's the
  exact mask used in training, the same size as the model's output.

They cover 1.5–49 km² and Landsat 5, 7 and 8, and add about 175 MB. `glacierview infer` runs on any of
them:

```bash
GV_DATA_ROOT=data/sample/inference uv run glacierview infer \
  --glims-id G072276E36720N --checkpoint GlacierView/glacierview-unet --out-dir /tmp/one
```

Comparing the model's 128×128 output to the masks gives mean Dice 0.87, which matches the measured
holdout figure. **Don't read that as accuracy, though.** These are the images the model trained on,
so the score only shows the files and the pipeline work. No glacier the model has never seen is
included. The 23 such glaciers in the inference run exist only in S3.
