# Data model

Nothing under `data/` is in git. The code reads from a local "landing zone" whose layout is fixed by
`data_label` — a string hardcoded near the top of every script and notebook that selects which dataset
to operate on:

```
src/earth_engine/data/
  ee_landing_zone/<data_label>/
    landsat/<glims_id>/<glims_id>_<date>_<L5|L7|L8>_C02_T1_L2_SR.tif
    landsat/<glims_id>/meta_data/metadata_list_l{5,7,8}   # raw EE getInfo() blobs
    dems/<glims_id>_NASADEM.tif
    logs/log_<name>.log
  processed_metadata/<data_label>/*.csv                    # relationalized metadata
src/glims/data/
  glims_db_20210914/glims_polygons.shp                     # full GLIMS inventory
  training_samples/glims_18k_bb.shp                        # bounding boxes, 18k glaciers
  inference_samples/geog_area_rollup_{50,250}.csv          # which glaciers to infer on
```

Two `data_label`s are live: `localized_time_series_for_training_c02_t1_l2` (training, small crops) and
`full_time_series_c02_t1_l2` (inference, full time series). `full_time_series`,
`localized_time_series_for_segmentation_training`, and `..._large` are earlier, pre-Collection-2
iterations — don't populate them.

Bucket inventory and the Athena catalog: `.agents/references/aws.md`.

## Raster conventions

`glacierview.rasters` + `glacierview.preprocess` are the shared raster path used by every notebook and
script. Two pipelines exist (`--preprocessing current|published`); the module docstring of
`preprocess` lays out both.

- **current** (default; a model must be retrained on it). Each glacier gets one
  `rasters.reference_grid`: 30 m UTM over the DEM's extent, in the zone of its centre. Every scene,
  the DEM and the training mask are resampled onto it, because Earth Engine returns scenes in
  whichever UTM zone their path/row uses (Trient arrives on two grids 128 m apart) and the DEM in
  EPSG:4326. `rasters.read_scene` reads only the six bands, turns fill (DN <= 0 or nodata) into NaN,
  and resamples nearest-neighbour. Bands are clipped to `bands.CUTOFFS_DN` and scaled to [0, 1]
  (`scale_to_cutoffs`), resized with a NaN-aware average (`resize_masked`), smoothed along time with
  NaN left out (`smooth_over_time`), and only then filled (`fill_missing`). NDSI/NDWI use
  `(a - b) / (a + b + eps)`. Areas are measured on the reference grid.
- **published** (what the published checkpoint was trained and run on): `rasters.get_rasters`
  returns `{file_name: HWC array}` per file on its own grid, negatives clamped to 0;
  `normalize_rasters` stretches each band between its *second* smallest and largest distinct values;
  scenes and the DEM are resized independently; NDSI/NDWI use `(a - b + eps) / (a + b + eps)`, which
  turns 0/0 fill into +1.
- Band indices differ per satellite. `glacierview.bands` maps names to indices for `l5`/`l7`/`l8`,
  and `preprocess.band_indexes` picks the map from the **third underscore-delimited token of the
  filename** (`{glims_id}_{date}_{L5|L7|L8}_C02_T1_L2_SR.tif`), returning the bands in
  `COMMON_BANDS` order. Filenames are load-bearing; inference also parses the date from token 1.
- Everything is resized to 128×128 before the model and resized back afterwards for area.

Two steps the paper documents that are easy to miss in the code:

- **The paper says training images were temporally averaged**: all scenes within ±100 days of the
  GLIMS `src_date`, combined by arithmetic mean. The code does not do this. Notebook 03 pulls scenes
  within ±50 days, `sql/training_data_query.sql` picks one, and notebook 10 builds the pair from that
  single scene; the training set on disk is single scenes too.
- **Inference applies a Gaussian filter along the time axis**, not in space, smoothing each pixel
  across the glacier's time series to reduce noise. The published run used
  `gaussian(X, sigma=[20,0,0,0], mode='reflect')` (σ = 20 *time points*), and `--preprocessing
  published` still does (`config.PUBLISHED_TIME_SMOOTHING_SIGMA`). The current pipeline uses
  `config.TIME_SMOOTHING_SIGMA` (2) and leaves missing pixels out (`preprocess.smooth_over_time`).

Both `glacierview infer` and `glacierview areas` go through `inference.predict.build_inputs`, which
smooths first and computes NDSI/NDWI on the smoothed stack, as the paper does. (The two original
scripts disagreed on this order; that is why they were merged.)

⚠️ Reprojection direction differs from the paper: the paper says the *rasters* were reprojected to
WGS 84 to match the polygons. The published code reprojected the *polygons* into one scene's UTM CRS
(notebook 09), and the current code rasterises them onto the reference grid. Don't assume any of the
three produced a given artifact.
