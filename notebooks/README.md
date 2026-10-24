# Notebooks

Notebooks are thin drivers: each one sets a few parameters, calls the `glacierview` package, and
shows or writes the result. The logic lives in the package, where it can be imported, reused and
checked. Anything that runs the same way every time is a CLI command instead; see
`glacierview --help`.

They import the installed package, so run `uv sync` first and start Jupyter from anywhere:

```bash
uv run jupyter lab
```

Paths come from `glacierview.config` and are environment-overridable: start Jupyter with
`GV_DATA_ROOT` set to work off an external disk, or off the committed sample.

Each notebook opens with a header: its purpose, its inputs (files and non-code dependencies), its
outputs, and any manual step afterwards.

## `pipeline/`

Numbered because order matters: these produce the datasets everything else consumes. They stay
notebooks because each step runs rarely and is inspected as it goes.

| Notebook | What it does | Package code |
|---|---|---|
| `01_glims_select_glaciers.ipynb` | Filters GLIMS to 18,093 glaciers and adds the bounding boxes Earth Engine is queried with. | `glims` |
| `02_glims_to_parquet.ipynb` | Adds `geog_area_rollup`, drops geometry, writes `data/glims_18k.parquet` for Athena. | `glims` |
| `03_ee_pull_training.ipynb` | Downloads the scenes within ±50 days of each glacier's GLIMS outline, for training. | `earthengine` |
| `04_ee_pull_dems.ipynb` | Downloads one NASADEM per glacier, for inference or training. | `earthengine` |
| `05_ee_pull_time_series.ipynb` | Downloads the full 1984-present series for the inference set. | `earthengine` |
| `06_metadata_training.ipynb` | Turns the training pull into the CSVs behind the Athena tables. | `metadata` |
| `07_metadata_time_series.ipynb` | The same for the inference pull. | `metadata` |
| `08_inventory_downloads.ipynb` | Lists what actually landed on disk. | — |
| `09_build_masks.ipynb` | Rasterises GLIMS outlines into masks, reprojected into each scene's CRS. | `training.build` |
| `10_build_training_set.ipynb` | Pairs scenes, DEMs and masks into the `images/` + `masks/` training set. | `training.build` |

The Earth Engine notebooks (03–05) need an Earth Engine sign-in; the rest run offline on local data.
All the pulls resume: glaciers already on disk are skipped.

## `exploration/`

| Notebook | What it does |
|---|---|
| `glims_inventory.ipynb` | Region counts and maps for the selected, trained-on and inference glaciers. |
| `landsat_metadata.ipynb` | Where the cloud-cover, zero-pixel and pixel-count thresholds came from. |
| `band_distributions.ipynb` | Per-scene band minima and maxima across Landsat 8. |
| `preprocessing_sandbox.ipynb` | Scenes before and after preprocessing, band by band. |

`band_distributions` and `preprocessing_sandbox` take `USE_SAMPLE = True` to run on the committed
sample (`data/sample/inference`) instead of the full landing zone.

## `analysis/`

| Notebook | What it does |
|---|---|
| `regional_decline.ipynb` | The published analysis. Estimates each region's decline from the 1,085-glacier area series, as a bootstrapped log-linear trend and as a weighted mean of per-glacier slopes. Reads `data/analysis/`. |

It reproduces the published numbers exactly, including two quirks of the original code that are
kept behind flags:
- **Two glaciers were dropped** (Trient and `G007045E45994N`) by a positional slice, which is why the
  paper reports 1,083 glaciers.
- **The slope weights are inverted:** each glacier is weighted by the share of years *missing*, so
  glaciers with more data count less. Corrected, North America's decline goes from the slowest to in
  line with Asia; see `.agents/context/state.md`.

The original also computed each glacier's error against its GLIMS area but never used it to filter
anything; the rewrite keeps the error as a diagnostic. Oceania's 35 glaciers aren't in the paper's
five regions and are left out.

## Not here any more

- `inspect_training_data.ipynb`: its inspection cells are now the end of notebook 10.
- `reprojection.ipynb`: one-off reprojection of an old, non-existent data layout.
- `glacier_segmentation-updated.ipynb`, `test_saved_model_*.ipynb`: training is
  `glacierview train`, single-glacier inference is `glacierview infer`.
- `get_s3_data.ipynb`: did what `aws s3 cp` does.

See git history for any of them.
