# Inference sample: Trient glacier

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
