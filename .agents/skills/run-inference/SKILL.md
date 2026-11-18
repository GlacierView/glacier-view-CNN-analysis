# Run inference

## When to use

To segment glaciers and produce surface-area time series, for one glacier or
the whole landing zone.

## One glacier

```bash
uv run glacierview infer --glims-id G007026E45991N \
  --checkpoint src/segmentation/unet_summer_model_unfrozen_100
```

Writes a GIF and an area plot under the output directory. Use
`glacierview config` first if a path looks wrong — it prints everything
resolved, and flags what is missing.

Two options matter for what the numbers mean:

- `--preprocessing current|published` (default `current`). The published checkpoint
  was trained on `published`; `current` puts every scene on one grid, masks fill and
  scales to fixed cut-offs, and needs a retrained model. See `glacierview.preprocess`.
- The quality filter: only scenes in `config.filtered_inference_path()` are used when
  that CSV exists. `--filter-csv <path>` picks another; `--no-filter` uses everything
  on disk. The log says which applied.

## Every glacier in the landing zone

```bash
uv run glacierview areas --checkpoint <path>
uv run glacierview areas --checkpoint <path> --glims-id G007026E45991N --no-gif
```

Writes three aggregate CSVs (raw, thresholded, binary), per-glacier CSVs under
`glacier_areas/` (one column per variant), and GIFs. Each per-glacier CSV is
written as soon as that glacier finishes, and the aggregates are rebuilt from
all of them when the run ends, even on Ctrl-C. So a run that dies partway still
leaves usable output, and `glacierview.inference.batch.write_aggregates(out_dir)`
rebuilds the aggregates by hand.

**Read the summary it prints.** Failures are logged with a traceback and
counted, not raised — with hundreds of glaciers one bad scene should not cost
the run. The exit code is non-zero if anything failed.

## The 250-per-region set, streamed from S3

The 1,024 glaciers chosen by `sql/inference_data_query_250.sql` are 68,880
scenes / 57.5 GB, more than a laptop holds. `areas-s3` downloads a batch of
glaciers' listed scenes, segments them, deletes them, and moves on:

```bash
uv run hf auth login     # once: the Hugging Face repo is private
aws s3 cp --region us-west-1 <Athena result of inference_data_query_250.sql> \
  src/glims/data/inference_samples/filtered_inference_data_250.csv
caffeinate -i uv run glacierview --logfile areas_s3.log areas-s3 \
  --file-list src/glims/data/inference_samples/filtered_inference_data_250.csv \
  --checkpoint GlacierView/glacierview-unet --out-dir <dir>
```

`--logfile` belongs to `glacierview` itself, so it goes before the subcommand.

```
```

`scripts/run_inference_250.sh` wraps this command with its prerequisites
documented; extra flags pass through to `areas-s3`. Try `--limit 4 --batch-size 2` first. It checks free space before each batch,
deletes only directories it downloaded (a glacier already on disk is used and
kept), and keeps the DEMs (89 MB in all). Rerun the same command to resume:
glaciers with a CSV in `<dir>/glacier_areas/` are skipped, and the aggregates
are rebuilt from all of them. No GIFs are written; use `infer` for those.

## Getting the imagery first

Trient needs no download: every scene and its DEM are committed under
`data/sample/inference`, laid out like the data root.

```bash
GV_DATA_ROOT=data/sample/inference uv run glacierview infer --glims-id G007026E45991N \
  --checkpoint GlacierView/glacierview-unet --out-dir /tmp/trient
```

Expect `62 scenes, 6.20 -> 5.73 km^2`: the sample carries its own quality-filter CSV, which is applied
automatically. With `--no-filter` the run uses all 1,733 scenes (15 before `config.FIRST_VALID_DATE`
are skipped). `--preprocessing published --no-filter` reproduces the published `7.41 -> 6.31 km^2`
exactly. The full table is in `data/sample/inference/README.md`. Always pass `--out-dir` here. The same root holds one
training scene, DEM and GLIMS mask for 100 more glaciers (`data/sample/inference/glaciers.csv`).
They're training images, so scoring against the masks checks plumbing, not accuracy.

For any other glacier, the landing zone's `landsat/` directory is empty locally. Pull one
glacier (Trient's is ~700 MB):

```bash
aws s3 cp s3://full-time-series-images-t1-l2-sr/G007026E45991N/ \
  "$(uv run glacierview config | awk '/landsat_dir/{print $2}')/G007026E45991N/" \
  --recursive --region us-west-1 --exclude "*/meta_data/*" --exclude "*.DS_Store"
```

The whole bucket is 486 GB; the filtered inference set is 1,024 glaciers /
68,880 files (`filtered_inference_data_250.csv`). Pull that subset. See `.agents/rules/data.md`.

## Verifying it worked

Trient (`G007026E45991N`) is the documented regression case. The paper reports 464
images (1984-04-18 to 2023-10-07), but the current code gives 1,748 scenes on the committed sample
and 62 in the filtered run; 464 is reproduced only if Landsat 8 is included, so do not use it as a
check.

## References

- `.agents/context/model.md` — what the model expects
- `.agents/references/aws.md` — bucket inventory
