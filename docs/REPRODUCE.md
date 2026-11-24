# Reproducing the results

## Environment used for the reported numbers

- Windows 11, Python 3.12.3, a laptop with an NVIDIA RTX 3050 (4 GB). The GPU is used only for the small GRU.
- Package versions are pinned in [`requirements.txt`](../requirements.txt). The PyTorch build is `2.14.1+cu126`; install it with
  `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126`. For a CPU-only machine, install a CPU build of PyTorch instead; the GRU step will then be slower.
- Every random seed is fixed in the scripts (the seed is `20261117`).

## Steps

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
export MBZUAI_DATA=/path/to/data        # PowerShell: $env:MBZUAI_DATA="D:\data"
python src/download_datasets.py         # about 1.9 GB, checks checksums
python src/run_all.py                   # all steps, results in results/, logs in results/logs/
```

`run_all.py` options: `--quick` skips the three slow steps; `--only forecast_baselines plot_baselines` runs chosen steps. A step that fails stops the run and its log is in `results/logs/<step>.log`. Timings are written to `results/run_all_timings.json`.

## Expected run time (the reference machine)

A complete run took 72 minutes with no failures:

| Step group | Approximate time |
|---|---|
| Data loading, data overview, sensor checks | 2 minutes |
| Baselines, external cohorts | 4 minutes |
| Feature ladder | 5 minutes |
| Microbiome, richer meals | 10 minutes |
| Personalisation (glucose only) | 21 minutes |
| Personalisation on top of meals | 23 minutes |
| Small GRU (GPU) | 7 minutes |
| Tables and derived numbers | under 1 minute |

## What to expect

- On the reference machine, a second complete run reproduced all 41 result files exactly (one raw per-person file differed by 4e-15, floating-point noise), including the GPU-trained network. On other hardware, tiny differences in the GRU step are possible because GPU training is not guaranteed to be deterministic. The other steps use seeded CPU code.
- The figures are produced by the `plot_*.py` and `eda_cgmacros.py` scripts and are not stored in this repository; they are regenerated into `figures/`.

## Tests

```bash
pip install pytest ruff
pytest -q        # 10 unit tests on synthetic data, no dataset needed
ruff check .     # lint: syntax errors, undefined names, unused imports and variables
```

The tests cover the steps most likely to go wrong silently: recovering real readings from interpolated minute data, lag ordering and forecast targets, dropping windows with gaps, aligning irregular readings to the grid, the HbA1c groups and unit conversion, the rule that meal features use only meals logged at or before the forecast time, the shuffled control, the baselines, and the fold construction (nobody is split between training and testing).
