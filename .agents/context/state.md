# Current state of the data

Checked by diffing local inventories against the buckets.

**Present and complete:**

- Published training data — `training/data/processed_training_data_summer_months/` holds 7,174
  images + 8,902 masks, an exact mirror of `training-images-t1-l2-sr`. It is
  `config.published_training_set_dir()`: what the published checkpoint was trained on, built by the
  published preprocessing. `glacierview train` no longer reads it by default (see "Preprocessing
  rebuilt" below); pass it as `--data-dir` with `--normalize` to reproduce published training.
- Raw training data — 18,093 glacier dirs of imagery and DEMs under
  `localized_time_series_for_training_c02_t1_l2/`, plus `masks_staging_2/`. That is 148 glaciers *more*
  than `raw-training-images-t1-l2-sr` holds (see "unbacked" below).
- Inference DEMs — all 1,122, byte-for-byte the same id set as `full-time-series-dems-t1-l2-sr`.
- Inference metadata — the four table CSVs plus `filtered_inference_data_250.csv`
  (`filtered_inference_data.csv`, the old 50-vintage list, is not on disk).

**Missing and blocking:**

- `full_time_series_c02_t1_l2/landsat/` is **empty** — every inference image is absent. That tree lived
  on an external SSD (`/Volumes/T7/GlacierView/ee_landing_zone/...`; set `GV_DATA_ROOT` to use it).
  Nothing needed is missing *upstream*: all 1,024 glaciers in `filtered_inference_data_250.csv` and
  `geog_area_rollup_250.csv` exist in S3. Trient and 100 single-scene glaciers are committed
  under `data/sample/inference`, so inference runs without it.
- ~~`model` checkpoint file~~ — fixed: `--checkpoint GlacierView/glacierview-unet` loads the
  published model from Hugging Face.
- ~~`glacier_areas/` never created~~ — fixed: `glacierview areas` creates it and writes the
  per-glacier CSVs.
- ~~Input-channel mismatch~~ — fixed. All entry points now derive their channel count from
  `IN_CHANNELS` in `glacierview.models.unet` and assert it before the first convolution.
- ~~Truncated bounding boxes~~ — fixed 2026-10-23. Running notebook 01 under numpy 2 had written
  `np.float64(...)` reprs that overflowed the shapefile's 254-character field, corrupting 17,354 of
  18,093 `bboxes`. `glims.add_bounding_boxes` writes plain floats, and the shapefile is regenerated
  and verified against `data/glims_18k.parquet`.

**Open issues** (found 2026-10-23; all affect what the results mean; fixes are marked):

- **Training and inference see inputs at different scales.** `GlacierDataset` (used by
  `glacierview train` and by the 0.868 Dice measurement) min-max normalises images that notebook 10
  already normalised, with `NORMALIZE_SMOOTHING = 1` added to each range, so bands land in ~[0, 0.5].
  Inference (`preprocess.prepare_glacier_stack`) feeds ~[0, 1]. The original `train.py` did the same,
  so the published checkpoint was trained at half scale, yet on the 100 committed sample glaciers it
  scores Dice **0.870 at inference scale vs 0.821 at training scale**. **Fixed in code 2026-11-02:**
  `TrainConfig.normalize` now defaults to off, so a retrained model sees [0, 1] like inference. The
  published checkpoint is unchanged.
- **The inference glaciers were in the training set.** All 1,024 glaciers of the 250-per-region
  selection, and 222 of the 245 in `filtered_inference_data.csv`, are among the 7,174 trained on (by
  construction: the `identify_inference_*` queries read the training tables). Accuracy on them is
  in-sample; only those 23 are unseen. The 0.868 holdout may also overlap training, since the
  original split was never saved.
- ~~`glacierview infer` / `areas` apply no quality filter~~ — fixed 2026-11-02. Both now use only the
  scenes listed in `config.filtered_inference_path()` (`filtered_inference_data.csv`) when it exists,
  and `areas` skips glaciers with no listed scene. `--filter-csv` picks another file and `--no-filter`
  uses everything on disk. The committed sample carries its own copy of the CSV's rows for its
  scenes, so it runs Trient on 62 scenes by default, or on all 1,733 with `--no-filter`.
- ~~The DEM is preprocessed in a different order in training and inference~~ — fixed 2026-11-02 for
  the current pipeline: `preprocess.prepare_dem` is shared. `--preprocessing published` keeps the
  old orders.
- **Some training pairs on disk come from older masks.** `masks_staging_2/` has 17,946 masks, the
  training set 8,902; e.g. `G079924E35890N`'s pair differs from a rebuild. Rerunning notebooks 09–10
  will change some pairs.
- **Time smoothing** is `config.TIME_SMOOTHING_SIGMA` = 2 frames for `current`, down from 20.
  `gif_smoothing.ipynb` shows that sigma 20 flattens the seasonal cycle. `published` always uses
  `config.PUBLISHED_TIME_SMOOTHING_SIGMA` = 20.
- **Preprocessing rebuilt (2026-11-02); the published checkpoint needs `--preprocessing published`
  until a model is retrained.** `glacierview.preprocess` now has two pipelines. `current` is the
  default for `infer`, `areas` and notebook 10's training pairs; `published` reproduces exactly what
  the published checkpoint saw (verified: identical Trient areas with `--no-filter`, Dice 0.870 on the 100 sample
  glaciers). What `current` fixes:
  - **Scenes and DEM weren't on one grid.** Earth Engine returns each scene in its path/row's UTM zone
    and the DEM in EPSG:4326, each resized to 128×128 on its own. Trient's scenes come on two grids
    (816 in zone 31 at 179×108, 932 in zone 32 at 177×103) that sit 128 m apart (median; up to
    287 m) at 128×128. The time smoothing averaged across both. Its DEM was off by a median 270 m
    (up to 690 m), in training too, and each training mask was drawn on the glacier's *first*
    scene's grid, not necessarily the grid of the scene it was paired with. Now everything goes onto
    `rasters.reference_grid` (30 m UTM over the DEM's extent, snapped to Landsat's pixel phase, so a
    scene in that zone is copied exactly), and areas are measured on that grid. The grid covers the
    glacier's padded box rather than each scene's larger UTM box, so neighbouring ice outside the
    box no longer counts.
  - **Fill read as ice.** DN 0 fill scaled to 0 in every band and to 200 K in thermal, and the old
    index formula `(a - b + eps) / (a + b + eps)` turned 0/0 into NDSI = NDWI = +1. Fill is common:
    from `band_distributions.ipynb` (200 training scenes per satellite), Landsat 7 is 27% fill
    (likely the post-2003 SLC-off stripes) and thermal 14–31% on every satellite. Fill is now NaN,
    resized and time-smoothed with NaN left out (so a stripe is filled from neighbouring dates),
    then filled with the frame mean. NDSI/NDWI use `(a - b) / (a + b + eps)`.
  - **Scaling.** Bands are clipped to `bands.CUTOFFS_DN` and scaled to [0, 1]: reflectance 0–1
    (DN 7,273–43,636; a 60,000 top was tried, and clipping at 1 gave better scaled values) and
    thermal 200–320 K (DN 14,921–50,029). Saturation (Landsat 5/7 visible bands at DN 65535 over
    snow: 57% of Landsat 5 blue pixels, 31–41% of Landsat 7 visible) lands at 1, the same as any
    reflectance ≥ 1. Landsat 8 doesn't saturate. `QA_RADSAT` is still unused.
  - **Memory.** Only the six bands are read, one scene at a time, so the peak no longer grows with
    glacier footprint. Trient's full series peaks at 3.7 GB, against 6.8 GB for `published`.
  - **Smaller:** bands are selected in `COMMON_BANDS` order (`preprocess.band_indexes`) rather than
    relying on the native order matching; index positions derive from `COMMON_BANDS`; resizes set
    `antialias=True` explicitly.
  
  **What it costs until retraining.** The published checkpoint on `current` inputs scores Dice 0.848
  on the 100 sample glaciers (against masks rasterised on the reference grid), the same as
  cut-offs-only scaling gave (L5 0.873→0.855, L7 0.827→0.811, L8 0.883→0.840 for that change), so
  the grid alignment didn't cost it anything. On all 1,733 Trient scenes, the series reads
  5.85 → 5.79 km² (`current`, sigma 2) instead of 7.41 → 6.31 km² (`published`, sigma 20). The
  table of settings is in `data/sample/inference/README.md`. To retrain:
  1. Rebuild masks with notebook 09. They now go to `config.masks_dir()` (`masks_reference_grid/`);
     the old ones in `masks_staging_2/` have no S3 copy and are untouched.
  2. Rebuild pairs with notebook 10, into `config.training_set_dir()`
     (`processed_training_data_reference_grid/`).
  3. Run `glacierview train`. `build_training_pair` refuses an old scene-grid mask.
- **Display:** `scene_viewer.ipynb` gives a saturated visible band the brightest unsaturated
  visible band's value, since snow is near-neutral in the visible; otherwise saturated blue tints
  snow pink.
- ~~Early Landsat 5 blue band~~ — handled 2026-11-02. From launch until November 1984, Landsat 5's
  blue band reads about 45% low over snow. The blue/green ratio over unsaturated snow is 0.52–0.63
  through August 1984, recovers through October, and is ~0.95–1.0 from early November, on all 16
  glaciers checked across five regions; every later period on every satellite is 0.94–1.03. USGS
  documents that the 1984–1991 calibration had the largest corrections (gains up to 15%, bands 1–2
  rescaled), but nothing describes an error this large. `config.FIRST_VALID_DATE` (1984-11-01) is
  now enforced by `EePull`, by inference (`rasters.list_scenes(since=...)`), by all four filtering
  queries in `sql/`, and by the analysis notebook (`START = None` reproduces the published numbers).
  Applying it moves the published weighted slopes: North America −0.0398 → −0.0277, Europe
  −0.1054 → −0.1092, South America −0.0836 → −0.0793, Asia −0.0552 → −0.0540, Caucausus
  −0.0654 → −0.0642. The data already on S3 and the Athena tables still include the early scenes.
- **Notebooks 03–05 hardcode a personal Earth Engine Cloud project** (`psyched-upgrade-377000`).
  Anyone else must substitute theirs.

**Missing, regenerable:** `areas_no_threshold.csv`, `areas_05_thresh.csv`, `areas_binary_05.csv` are
outputs of the old inference script and are not on disk. The current `areas` run writes its
per-glacier CSVs at the end of the run (`inference/batch.py`); the 250-glacier results are in
`src/segmentation/areas_250/`, made on 2026-11-03 with the pre-reference-grid pipeline at sigma 20
(today's `--preprocessing published`).

**Recovered (2026-09-19).** The four analysis inputs `areas_true_vs_predicted_final.ipynb`
live in **`data/analysis/`**, committed (the blanket `*.csv` ignore has a negation for that
directory). The notebook resolves them via an `ANALYSIS_DIR` constant derived from its own location,
so it no longer depends on where Jupyter was launched:

| File | Shape | What it is |
|---|---|---|
| `geo_areas.csv` | 18,093 × 32 | the `glims_18k` table itself — `glac_id`, `db_area`, `geog_area`, **`geog_area_rollup`**, `bboxes`, elevations, `src_date`, submitters. Rollups: Asia 12,625 / N. America 3,704 / S. America 872 / Europe 602 / Caucausus 255 / **Oceania 35** |
| `training_data_set.csv` | 10,447 × 12 | output of an older version of `training_data_query.sql` (it still has 660 Canada rows and 1,560 non-Canada rows outside the low-snow window) — one row per candidate training image with quality metadata + `rank_score`. This is the paper's stale "10,443". It is not the source of the trained set: only 4,078 of its scenes are in the training images, and the empty-mask variance filter then removes 3 pairs (7,174 → 7,171) |
| `training_areas.csv` | 3 × 260 | transposed lookup: ~259 glacier IDs → areas |
| `areas_binary_05_filtered_smooth_25.csv` | 541 monthly dates × **245** glaciers | smoothed output of the 2024 `final_areas.py` (predates the `glacierview` package), wide format (`pd.date_range('1979-01-01','2024-01-01',freq='MS')` = 541 months) |

**The 1,083-glacier output has been recovered** (2026-09-19):
`data/analysis/areas_binary_05_filtered_smooth_f.xlsx` — 1,085 glacier columns, 541 dated rows
(1979-01 to 2024-01), 39,241 observations, median 38 per glacier. Grouping to annual means and
regressing per region reproduces the paper's Table II, confirming this is the artifact behind the
published rates. The regional results use all 1,085 glaciers; the abstract's 1,083 does not describe
the analysed set. The 245-glacier CSV beside it is the older 50-per-region
vintage, kept for comparison.

⚠️ **Two quirks of the original analysis notebook** (`notebooks/analysis/regional_decline.ipynb`).
Neither shaped the paper's Table II or its histograms, which use observation-count weights and all
1,085 glaciers (`factcheck/wrong_docs/downstream_analysis.md`):

1. **The notebook's bootstrap weighting is inverted.** The original code computes
   `weight = (total_rows - n_observations) / total_rows` — the proportion of data *missing* — so a
   glacier with more observations gets less weight. This affected only the original bootstrap
   plots, not the paper's table or figure, and none of the inverted-weight numbers below is in
   the paper. Measured on the real data (total relative change over each glacier's observed span):

   | Region | n | Inverted weights | Corrected |
   |---|---|---|---|
   | Europe | 197 | −0.10537 | −0.09898 |
   | Caucausus | 74 | −0.06541 | −0.06468 |
   | Asia | 270 | −0.05515 | −0.05692 |
   | North America | 293 | **−0.03979** | **−0.06004** |
   | South America | 249 | −0.08361 | −0.09938 |
   | **All** | **1083** | **−0.06706** | **−0.07557** |

   North America is the one that matters: with inverted weights it is the slowest decline by a wide margin;
   corrected it is in line with Asia. The fix changes the regional *ordering*, not just magnitudes.

2. **Two glaciers are dropped by a positional slice** (`iloc[:, 2:]` on the annual table removed
   Trient and `G007045E45994N`), which is how 1,085 columns become 1,083. The paper's Europe value
   reproduces only with those two kept, so the drop did not shape the published regional results.

The notebook reproduces both columns above exactly, with each quirk behind its own flag. An earlier note here said a `rel_error_train < 8` filter gated the
results; it doesn't. The original computed that filter and never applied it, and the published
figures reproduce with no filter. Oceania's 35 glaciers never reach inference: none passes the SQL
filters, so five regions appear in the output.

**Present locally but unbacked — in S3 nowhere:**

- ~~Model checkpoints~~ — **backed up 2026-09-19** to
  `s3://segmentation-model-gv/checkpoints/` (us-east-1): both PyTorch checkpoints under
  `pytorch/`, the 18 legacy Keras files under `keras-legacy/`, and a `MANIFEST.md` recording
  channels, hashes and the training recipe. All 20 objects byte-verified. ⚠️ **Bucket versioning is
  still disabled** — an overwrite is unrecoverable, so enable it before re-uploading.
- 148 training glaciers that are local-only (`raw-training-images-t1-l2-sr` has 17,945 of the 18,093).
- All 18,093 training-landing-zone DEMs — there is no training DEM bucket.
- The derived metadata CSVs: `denormalized_metadata.csv`, `filtered_training_data.csv`,
  `filtered_training_data_summer_months.csv`, `filtered_inference_data.csv`,
  `net_new_glims_ids_for_segmentation.csv`. The buckets hold only the four crawled table CSVs.
- `src/glims/data/glims_db_20210914/` — the source GLIMS inventory. Only the derived
  `glims_18k.parquet` is in S3.
