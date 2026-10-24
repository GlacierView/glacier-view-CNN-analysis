# Current state of the data

Checked by diffing local inventories against the buckets.

**Present and complete:**

- Ready-to-train data — `training/data/processed_training_data_summer_months/` holds 7,174 images +
  8,902 masks, an exact mirror of `training-images-t1-l2-sr`. `glacierview train` reads it from
  `config.training_set_dir()` by default, so it trains on this with no flags.
- Raw training data — 18,093 glacier dirs of imagery and DEMs under
  `localized_time_series_for_training_c02_t1_l2/`, plus `masks_staging_2/`. That is 148 glaciers *more*
  than `raw-training-images-t1-l2-sr` holds (see "unbacked" below).
- Inference DEMs — all 1,122, byte-for-byte the same id set as `full-time-series-dems-t1-l2-sr`.
- Inference metadata — the four table CSVs plus `filtered_inference_data.csv`.

**Missing and blocking:**

- `full_time_series_c02_t1_l2/landsat/` is **empty** — every inference image is absent. That tree lived
  on an external SSD (`/Volumes/T7/GlacierView/ee_landing_zone/...`; set `GV_DATA_ROOT` to use it).
  Nothing needed is missing *upstream*: all 245 glaciers in `filtered_inference_data.csv`, and all
  1,024 in `geog_area_rollup_250.csv`, exist in S3. Trient and 100 single-scene glaciers are committed
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

**Missing, regenerable:** `areas_no_threshold.csv`, `areas_05_thresh.csv`, `areas_binary_05.csv` are
outputs `glacierview areas` creates on startup.

**Recovered (2026-09-19).** The four analysis inputs `areas_true_vs_predicted_final.ipynb`
live in **`data/analysis/`**, committed (the blanket `*.csv` ignore has a negation for that
directory). The notebook resolves them via an `ANALYSIS_DIR` constant derived from its own location,
so it no longer depends on where Jupyter was launched:

| File | Shape | What it is |
|---|---|---|
| `geo_areas.csv` | 18,093 × 32 | the `glims_18k` table itself — `glac_id`, `db_area`, `geog_area`, **`geog_area_rollup`**, `bboxes`, elevations, `src_date`, submitters. Rollups: Asia 12,625 / N. America 3,704 / S. America 872 / Europe 602 / Caucausus 255 / **Oceania 35** |
| `training_data_set.csv` | 10,447 × 12 | output of `training_data_query.sql` — one row per candidate training image with quality metadata + `rank_score`. This is the paper's stale "10,443". The empty-mask variance filter cuts it to the 7,174 actually trained on |
| `training_areas.csv` | 3 × 260 | transposed lookup: ~259 glacier IDs → areas |
| `areas_binary_05_filtered_smooth_25.csv` | 541 monthly dates × **245** glaciers | smoothed `glacierview areas` output, wide format (`pd.date_range('1979-01-01','2024-01-01',freq='MS')` = 541 months) |

**The 1,083-glacier output has been recovered** (2026-09-19):
`data/analysis/areas_binary_05_filtered_smooth_f.xlsx` — 1,085 glacier columns, 541 dated rows
(1979-01 to 2024-01), 39,241 observations, median 38 per glacier. Grouping to annual means and
regressing per region yields exactly the paper's **1,083** glaciers, confirming this is the
artifact behind the published rates. The 245-glacier CSV beside it is the older 50-per-region
vintage, kept for comparison.

⚠️ **Two methodological findings from the recovered analysis notebook**
(`notebooks/analysis/regional_decline.ipynb`):

1. **The bootstrap weighting is inverted.** The code computes
   `weight = (total_rows - n_observations) / total_rows` — the proportion of data *missing* — so a
   glacier with more observations gets less weight. The paper's text claims the opposite. Measured
   on the real data (total relative change over each glacier's observed span):

   | Region | n | As published | Corrected |
   |---|---|---|---|
   | Europe | 197 | −0.10537 | −0.09898 |
   | Caucausus | 74 | −0.06541 | −0.06468 |
   | Asia | 270 | −0.05515 | −0.05692 |
   | North America | 293 | **−0.03979** | **−0.06004** |
   | South America | 249 | −0.08361 | −0.09938 |
   | **All** | **1083** | **−0.06706** | **−0.07557** |

   North America is the one that matters: as published it is the slowest decline by a wide margin;
   corrected it is in line with Asia. The fix changes the regional *ordering*, not just magnitudes.

2. **Two glaciers are dropped by a positional slice** (`iloc[:, 2:]` on the annual table removed
   Trient and `G007045E45994N`), which is how 1,085 columns become the paper's 1,083.

The notebook now reproduces both the "as published" and "corrected" columns above exactly, with
each quirk behind its own flag. An earlier note here said a `rel_error_train < 8` filter gated the
results; it doesn't. The original computed that filter and never applied it, and the published
figures reproduce with no filter. Oceania's 35 glaciers drop out because the analysis only covers
the paper's five regions.

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
