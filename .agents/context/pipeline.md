# Pipeline

Data flows in one direction; each stage writes files that the next stage reads.

1. **GLIMS selection** — notebooks 01–02 (`glacierview.glims`) filter the GLIMS polygons to 18,093
   glaciers, add bounding boxes, roll `geog_area` up into continental buckets (`geog_area_rollup`),
   and write the shapefile plus `data/glims_18k.parquet` for Athena.
2. **Earth Engine pull** — notebooks 03–05 drive `glacierview.earthengine.EePull`, which downloads
   one GeoTIFF per image via `geemap.ee_export_image` at 30 m scale, plus a NASADEM per glacier.
   Server-side filters: `CLOUD_COVER <= 10`, `IMAGE_QUALITY == 9` (`IMAGE_QUALITY_OLI` for L8).
   Each image's `getInfo()` JSON is appended to `meta_data/metadata_list_l{5,7,8}` alongside the tifs,
   and a log line per image goes to the run log.
3. **Metadata relationalization** — notebooks 06 (training) and 07 (inference) drive
   `glacierview.metadata`: parse the pull log, dedupe the EE metadata (EE sometimes returns >1 record
   per image), open each tif for image/file attributes, and write CSVs → S3 → AWS Glue → Athena
   (database `glacier-view` in us-west-1 — see "Athena / Glue" below). Before 2026-10-23 the log
   parser kept only the hour of `scene_center_time`; CSVs written earlier carry that truncated value.
4. **SQL filtering** — `sql/` queries run against Athena to pick which GLIMS IDs and which individual
   files are eligible. `identify_inference_*` picks glaciers (N per `geog_area_rollup`);
   `inference_data_query.sql` / `training_data_query.sql` pick files. Criteria: cloud cover < 5%,
   summer months only (May–Oct northern hemisphere, Nov–Apr southern), > 50,000 pixels,
   0 no-data pixels, < 10% zero pixels.
5. **Training-set build** — notebook 09 rasterizes GLIMS outlines into masks, reprojecting each into
   its scene's CRS; notebook 10 pairs scenes, DEMs and masks into `config.training_set_dir()`
   (`images/` + `masks/`). Both call `glacierview.training.build`.
6. **Train** — `glacierview train`.
7. **Inference** — `glacierview infer` (single glacier, GIF + area plot) or
   `glacierview areas` (every glacier in the landing zone, area CSVs + GIFs).
8. **Analysis** — `notebooks/analysis/regional_decline.ipynb` estimates each region's decline from
   the area series in `data/analysis/`.
