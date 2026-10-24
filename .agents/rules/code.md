# Code rules

- The model is defined **once**, in `glacierview/models/unet.py`. Import `UNet` and
  `conv_block` from there; do not copy the classes into another file.
- Load models with `glacierview.models.load_checkpoint`: it takes the published Hugging Face repo id
  (safetensors) as well as the original pickled checkpoints, which reference `__main__.UNet` and
  `__main__.conv_block`. Renaming or reshaping a layer breaks both.
- Input channel count comes from `IN_CHANNELS` in `glacierview.models.unet`. Do not hardcode it.
- Filenames are load-bearing: `{glims_id}_{date}_{L5|L7|L8}_C02_T1_L2_SR.tif`. Token 0 is the GLIMS
  ID, token 1 the date, token 2 the satellite used to pick the band map. Renaming a file silently
  changes which bands get read.
- `glacierview.rasters` and `glacierview.preprocess` are shared by both inference entry points. A change
  there affects them equally — that is intentional; don't fork them.
- Notebooks are drivers: parameters, a call into the package, and output. Logic that a notebook
  needs goes in `glacierview` (`glims`, `earthengine`, `metadata`, `training.build`, ...), and paths
  come from `glacierview.config`, never `os.path.join` chains or a hardcoded root.
- Two formats are positional contracts: the Earth Engine pull-log line (`earthengine.export`, read
  by `metadata.parse_pull_log`) and the column order of the `metadata` CSVs (Athena reads them by
  position). Change either end only together with the other.
- Plots use the shared style `glacierview.config.mpl_style()`.
- `uv run ruff check .` must pass, notebooks included. The rule set is pinned in `pyproject.toml`;
  notebooks are exempt only from `B018` (a bare last expression displays a value). `ruff format`
  is deliberately not enforced.
