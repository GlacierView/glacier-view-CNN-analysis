#!/usr/bin/env bash
# Segment the 250-per-region glaciers (1,024 glaciers, 68,880 scenes, 57.5 GB),
# pulling their imagery from S3 a batch at a time.
#
# Each batch of 25 glaciers is downloaded, segmented with the published model,
# and deleted before the next, so only ~1.5 GB of imagery is ever on disk.
# It uses `--preprocessing published`, the pipeline the published model was
# trained on; the stored run in src/segmentation/areas_250/ (Nov 3, 2026)
# predates the `current` pipeline and was made with this one.
# DEMs (89 MB in all) are kept. The full run took about 2 hours on a MacBook
# Air M4 (Nov 2026); the download sets the pace.
#
# Usage, from anywhere:
#     scripts/run_inference_250.sh
#
# ---------------------------------------------------------------------------
# Before the first run
# ---------------------------------------------------------------------------
# 1. Install the environment:            uv sync
#    (boto3[crt] is needed if your AWS profile signs in with `aws login`.)
# 2. Hugging Face: the model repo GlacierView/glacierview-unet is private.
#                                        uv run hf auth login
#    or export HF_TOKEN=... ; check with uv run hf auth whoami
# 3. AWS: credentials for account 579803789230. Check with
#                                        aws sts get-caller-identity
#    `aws login` sessions expire; if one does mid-run, the remaining glaciers
#    fail to download. Run `aws login` and rerun this script.
# 4. The file list: one row per scene to segment. It is the output of
#    sql/inference_data_query_250.sql, run in Athena against the `glacier-view`
#    database in us-west-1. The result from 2026-10-30 is still in S3:
#
#    aws s3 cp --region us-west-1 \
#      s3://glacier-view-athena-output-us-west-1/claude-runs/2026-10-30/78a3b66b-d040-45c5-bd8a-0796fc14f55e.csv \
#      src/glims/data/inference_samples/filtered_inference_data_250.csv
#
#    If that object is gone, rerun the query in the Athena console and copy
#    its result CSV to the same path.
#
# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
# src/segmentation/areas_250/   (gitignored: back it up somewhere)
#   glacier_areas/<glims_id>_areas.csv  every scene's area, km^2, in three
#                                       variants: raw, thresholded, binary
#   areas_binary_05.csv                 annual means, one column per glacier,
#   areas_05_thresh.csv                 1979-2024; binary is the one the
#   areas_no_threshold.csv              regional analysis uses
#   areas_250.log                       the run log
#
# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------
# - Resumable: rerun the same script and glaciers that already have a CSV in
#   glacier_areas/ are skipped. Failed glaciers have no CSV, so they retry.
# - It checks free disk space before each batch and deletes only directories
#   it downloaded.
# - caffeinate -i keeps the Mac from idle-sleeping (the display may still turn
#   off, which is fine). Keep it plugged in with the lid open.
# - Try a small run first by passing extra flags through, e.g.
#       scripts/run_inference_250.sh --limit 4 --batch-size 2
#   (use a different OUT_DIR for a trial so it doesn't mix with the real run).
# - --logfile belongs to `glacierview` itself, so it goes *before* areas-s3.

set -euo pipefail

cd "$(dirname "$0")/.."

FILE_LIST=src/glims/data/inference_samples/filtered_inference_data_250.csv
OUT_DIR=${OUT_DIR:-src/segmentation/areas_250}

if [[ ! -f "$FILE_LIST" ]]; then
    echo "missing $FILE_LIST; see step 4 at the top of $0" >&2
    exit 2
fi
mkdir -p "$OUT_DIR"

caffeinate -i uv run glacierview --logfile "$OUT_DIR/areas_250.log" areas-s3 \
    --file-list "$FILE_LIST" \
    --checkpoint GlacierView/glacierview-unet \
    --preprocessing published \
    --out-dir "$OUT_DIR" \
    "$@"
