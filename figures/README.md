# Figures

Diagrams and plots for the paper and the docs. Several filenames are opaque, so
here is what each one is.

## Pipeline and architecture

| File | What it shows |
|---|---|
| `cnn_architecture.png` | The U-Net: ResNet-50 encoder, decoder, skip connections. Used in `README.md`. |
| `data_collection.png` | The training and inference data-collection workflow, end to end. |
| `data_collection_and_preprocessing.drawio` | **Editable source** for the workflow diagram — open at [app.diagrams.net](https://app.diagrams.net). Edit this rather than the exported PNG. |
| `data_collection_and_preprocessing-Data collection.drawio.png` | Export of the above. |
| `training_data_processing.png` | How raw scenes and GLIMS polygons become image/mask training pairs. |

## Data distributions

| File | What it shows |
|---|---|
| `distribution_of_glacier_surface_area.jpg` | Glacier area across GLIMS, with the 1 km² line marked. Motivates the 1–50 km² selection filter. |
| `distribution_of_metadata.jpg` | Landsat metadata distributions under the quality filters. This is where the cloud-cover and zero-pixel thresholds came from. |
| `glims_subsets.png` | How the GLIMS snapshot narrows to the 18,093 selected glaciers. |

## The UTM problem

The same glacier can be delivered in different UTM zones on different dates,
which is why masks are reprojected per image rather than once globally. See the
"Files, projections, and why the same glacier moves" section of
`docs/ONBOARDING.md`.

| File | What it shows |
|---|---|
| `utm zone 31 trient landsat 7.png` | Trient in UTM zone 31 (Landsat 7). |
| `utm zone 32 trient landsat 5.png` | The same glacier in zone 32 (Landsat 5). |
| `utm zone 31-32 overlap trient landsat 7-5.png` | The two overlaid — the offset is the problem. |

The `.pgw` files are world files: the georeferencing for the matching `.png`.

## `inventory-examples/`

Screenshots of glacier outlines from two GLIMS contributors, kept as the
evidence behind excluding them from the training set:

- `rgi-1..6.png` — Randolph Glacier Inventory. Prioritised complete global
  coverage over per-outline accuracy.
- `globglacier-1..6.png` — GlobGlacier (ESA). A smaller, more carefully
  curated set.

## `regional-decline/`

Per-region regression figures from
`notebooks/analysis/regional_decline.ipynb` (these Python plots match only the paper's commented-out figure;
the paper's active figure uses base-R plots from code not in this repo): estimated area for individual
glaciers, the pointwise average, and the fitted log-linear trend.

`asia-reg.png`, `caucausus-reg.png`, `europe-reg.png`,
`north-america-reg.png`, `south-america-reg.png`.

## Model output

`largest_glaciers_by_region.gif` — the three largest glaciers (by GLIMS area)
in each region of the 250-per-region set, one row per region, one frame per
year 1984–2023. Each cell shows that glacier's latest scene up to the frame's
year, with the model's mask (probability > 0.5) outlined and the binary area
beneath. Each title gives the glacier's relative change, measured as for the
figure below, and the years it spans. The set has no Oceania glaciers, so
there are five rows.

`most_shrinking_glaciers_by_region.gif` — the same layout, for the three
glaciers per region that lost the largest share of their area: the mean of
their last three annual areas over the mean of their first three (from
`src/segmentation/areas_250/areas_binary_05.csv`), among glaciers observed
over at least 25 years and in at least 10. The change and the years it spans
are in each cell's title.

`global_overview_stacked.{png,pdf}` — the paper's introduction figure, with
panels above and below the map. An Equal Earth map of the 7,174 glaciers
trained on (which include all 1,024 inference glaciers), coloured by region,
and for each region the glacier with the largest relative area loss among
those whose predicted starting area is over 10 km²: its earliest and latest
outline, with the earliest dashed over the latest where the two scenes share a
footprint. Made by `scripts/global_overview_figure.py`; the trained-on list is
`data/analysis/trained_glims_ids.csv`, read from the training bucket.

The two GIFs are made by `scripts/region_gif_figure.py` (`--select largest` or
`--select shrinkage`), which also writes each glacier's own GIF to
`src/segmentation/gifs/by_region/`.

## Logos

`earth-engine-logo.png`, `glims_logo_smooth.png` — for slides and diagrams.
