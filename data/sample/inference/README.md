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

Expected result: `G007026E45991N: 62 scenes, 6.26 -> 5.60 km^2`, using the quality filter (below) and
`config.TIME_SMOOTHING_SIGMA = 0`. `--no-filter` uses all 1,733 scenes. The published checkpoint was
trained on the published preprocessing, so until a model is retrained on the current one,
`--preprocessing published` is the setting that matches it. Inference
skips the 15 scenes before 1 November 1984 (`config.FIRST_VALID_DATE`), when Landsat 5's blue band
reads about 45% low.

Pass `--out-dir`. Without it, outputs go under `$GV_DATA_ROOT/segmentation`, which is gitignored.

**Quality filter.** `earth_engine/data/processed_metadata/full_time_series_c02_t1_l2/filtered_inference_data.csv`
holds the rows of the full quality filter (`sql/inference_data_query.sql`) for the scenes in this
sample: 62 of Trient's, and 97 of the other glaciers' 100. It sits where `glacierview infer` and
`areas` look for the filter, so with `GV_DATA_ROOT` pointed here they use it automatically; pass
`--no-filter` to use every scene. Unfiltered, the area series is much noisier than the published
one. The paper's Trient figure used 464 scenes chosen with its own looser filters.

## 100 more glaciers, one scene each

`glaciers.csv` lists 100 more glaciers, 20 from each region (Asia, Caucausus, Europe, North America,
South America), drawn at random (seed 0) from the 245 in the paper's inference run. Each has:

- **The training scene:** `landsat/<glims_id>/` holds the summer scene the model was trained on for
  that glacier. It's copied unmodified from the training pull (`localized_time_series_for_training_c02_t1_l2`).
- **The DEM:** `dems/<glims_id>_NASADEM.tif`.
- **The label:** `masks/<glims_id>.tif`, the glacier's GLIMS outline rasterised at 128×128. It's the
  exact mask the published checkpoint was trained with, on the published pipeline's grid. It matches
  `--preprocessing published` only; for `current`, rasterise the outline onto the reference grid
  (`training.rasterize_outline(rasters.reference_grid(dem_path), outline)`).

They cover 1.5–49 km² and Landsat 5, 7 and 8, and add about 175 MB. `glacierview infer` runs on any of
them:

```bash
GV_DATA_ROOT=data/sample/inference uv run glacierview infer \
  --glims-id G072276E36720N --checkpoint GlacierView/glacierview-unet --out-dir /tmp/one
```

Comparing the model's 128×128 output to the masks gives mean Dice 0.87 with `--preprocessing
published`, which matches the measured holdout figure (0.848 with `current` against reference-grid
masks). **Don't read that as accuracy, though.** These are the images the model trained on,
so the score only shows the files and the pipeline work. No glacier the model has never seen is
included. The 23 such glaciers in the inference run exist only in S3.
