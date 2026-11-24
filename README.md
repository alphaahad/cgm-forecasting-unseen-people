# cgm-forecasting-unseen-people

[![CI](https://github.com/alphaahad/cgm-forecasting-unseen-people/actions/workflows/ci.yml/badge.svg)](https://github.com/alphaahad/cgm-forecasting-unseen-people/actions/workflows/ci.yml)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Results: CC BY-NC-SA 4.0](https://img.shields.io/badge/results-CC%20BY--NC--SA%204.0-lightgrey.svg)](results/LICENSE.md)

Code and results for **"What actually helps glucose forecasting on unseen people? A controlled study across three CGM cohorts"** (Ahad Nadeem).

The study asks which inputs actually improve forecasts of glucose 30, 60 and 120 minutes ahead when the model is tested on **people it has never seen**, and checks every added input against a **shuffled or swapped-data control**. It uses three public continuous glucose monitor (CGM) datasets, and it runs every analysis for both sensors worn at the same time in the main cohort.

> **Not medical advice.** These are forecasting experiments on small public cohorts. They do not support dosing, diet or any other clinical decision. Typical errors are 13 to 29 mg/dL.

## Findings at a glance

All figures are RMSE changes in mg/dL at 60 minutes on unseen people, with 95% bootstrap intervals over people (see [`results/`](results/) for every number).

| Question | Result |
|---|---|
| Does glucose history beat "same as now"? | Yes, modestly: 3.2 mg/dL (2.4 to 4.0) on Libre and 3.1 (2.5 to 3.7) on Dexcom |
| Do meals help? | Yes, a little, and the shuffled-meal control shows it is real: 1.05 (0.64 to 1.44) Libre, 1.69 (1.17 to 2.15) Dexcom |
| Does a few days of a person's own data help? | Yes, and it adds to meals: about 11% lower error in total with 5 days of personal data |
| Health group, simple activity features, ten richer meal features, a small GRU | No benefit |
| Six gut-microbiome summary values | Made forecasts worse (about 1.2 to 1.4 mg/dL), and no better than another person's values |
| Does a model transfer to other cohorts? | Partly: about two thirds of the possible gain on Shanghai T2DM and about a quarter on BUT CGM |
| Do the two CGMacros sensors agree? | No: Libre read about 32 mg/dL lower than Dexcom on average, and neither matched fingersticks. This is reported, not explained |

## Repository layout

```
src/         analysis scripts (one script per experiment) and a runner, run_all.py
tests/       unit tests on synthetic data (no dataset needed)
results/     every logged result as CSV/JSON, the paper tables in results/tables/  (CC BY-NC-SA 4.0)
docs/        DATA.md (datasets and licences) and REPRODUCE.md (how to rerun and what to expect)
.github/     continuous integration (lint and tests)
```

| Script | What it does |
|---|---|
| `download_datasets.py` | downloads the three datasets and checks their checksums |
| `cgmacros_loader.py`, `external_loaders.py` | load CGMacros, Shanghai T2DM and BUT CGM onto one 15-minute mg/dL grid |
| `eda_cgmacros.py`, `sensor_*.py` | data overview and the Libre versus Dexcom comparison |
| `forecast_baselines.py` | persistence, linear trend, ridge and LightGBM on unseen people |
| `forecast_external.py` | transfer to other cohorts and each cohort's own ceiling |
| `forecast_ladder.py` | adds health group, meals and activity with shuffled controls |
| `forecast_microbiome.py`, `forecast_meals_rich.py` | the microbiome and richer-meal experiments with controls |
| `forecast_personal.py`, `forecast_personal_meals.py` | personalisation with the first K days of a person's data |
| `forecast_gru.py` | a small GRU compared with LightGBM |
| `export_tables.py`, `derived_numbers.py` | build the paper tables and derived ratios from the results |
| `run_all.py` | runs everything in order and logs timings |

## Quick start

Python 3.12 (tested with 3.12.3). The pinned environment is in [`requirements.txt`](requirements.txt); PyTorch is the CUDA 12.6 build and is needed only for the small GRU step (a GPU is optional).

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; on Linux or macOS: source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126   # CPU-only: install a CPU build of PyTorch instead

export MBZUAI_DATA=/path/to/data  # where datasets go (default D:\MBZUAI_data). On Windows PowerShell: $env:MBZUAI_DATA="D:\data"
python src/download_datasets.py   # about 1.9 GB: CGMacros, BUT CGM (without raw ECG/accelerometer files), Shanghai
python src/run_all.py             # about 72 minutes on a laptop with an RTX 3050 (4 GB)
```

`python src/run_all.py --quick` skips the three slow steps; `--only <step> [<step> ...]` runs chosen steps. See [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

Run the tests (seconds, no data needed):

```bash
pip install pytest ruff
pytest -q
ruff check .
```

## Data

The data are public and **not included**; the downloader fetches them from their original sources. Licences and required citations are in [`docs/DATA.md`](docs/DATA.md).

| Dataset | People | Sensor | Licence |
|---|---|---|---|
| CGMacros (PhysioNet) | 45 (15 normal, 16 pre-diabetes, 14 T2D) | Libre Pro and Dexcom G6 Pro, worn together | CC BY-NC-SA 4.0 |
| Shanghai T2DM (figshare) | 100 | FreeStyle Libre H | CC BY 4.0 |
| BUT CGM (PhysioNet) | 58 with CGM data (60 enrolled) | Dexcom G7 | CC BY 4.0 |

## Method in brief

- **Real readings only.** The CGMacros minute-level glucose is linearly interpolated between the sensors' real 15-minute (Libre) and 5-minute (Dexcom) readings, so everything is rebuilt on a 15-minute grid from real readings.
- **Task.** Two hours of history (eight readings) predict glucose 30, 60 and 120 minutes ahead.
- **Unseen people.** Five-fold cross-validation over people, repeated three times; every person is tested once per repeat. The person is the unit of analysis, with bootstrap intervals over people and paired comparisons.
- **Controls.** Each added input is paired with a control that keeps the same columns but breaks their link to the glucose curve (features shuffled in time within a person, or another person's values swapped in).
- **Both sensors.** Because the two CGMacros sensors disagree strongly, every analysis is run for both and reported separately.
- **Design recorded before running.** Features and controls for each experiment were written to a dated project log before the experiment ran. That log is a file on the author's computer, not an independent registration.

## Results files

`results/` holds the output of `run_all.py`. The tables used in the paper are in `results/tables/` (`T1.csv` to `T5.csv`, plus `tables.md`).

| Topic | Main files |
|---|---|
| Baselines | `baselines_summary.csv`, `baselines_gain_vs_persistence.csv`, `baselines_by_group.csv` |
| Transfer to other cohorts | `external_summary.csv`, `external_cohort_summary.json`, `but_roster.json` |
| Feature ladder and controls | `ladder_summary.csv` |
| Microbiome, richer meals | `microbiome_summary.csv`, `microbiome_real_vs_control.csv`, `rich_meals_summary.csv` |
| Personalisation | `personal_summary.csv`, `personal_meals_summary.csv` |
| Neural network | `gru_summary.csv` |
| The two sensors | `sensor_offset_stats.json`, `sensor_offset_by_day.csv`, `sensor_interpolation.json`, `fingerstick_vs_sensors.csv` |
| Derived ratios quoted in the text | `derived_numbers.json` |

`*_per_participant.csv` and `*_per_person.csv` hold the per-person results the summaries are built from.

## Limitations

The main cohort has 45 people, so intervals are wide. The two sensors disagree and the gap is unexplained. Meal logs are estimates and only about a quarter of samples have a recent meal. The microbiome analysis used six summary values only. In personalisation, more days of data also means data closer in time to the test window. Many comparisons were run without correction for multiple testing. See the paper for the full list.

## AI assistance

The code, documentation and analysis in this repository were written with substantial assistance from an AI model (Claude, Anthropic) working under the author's direction. The author is responsible for the content.

## Citation

Please cite the datasets (see [`docs/DATA.md`](docs/DATA.md)) and this work (see [`CITATION.cff`](CITATION.cff)); a preprint reference will be added when it is posted.

## Licences

- **Code** (`src/`, `tests/`, `.github/`): [MIT](LICENSE).
- **Result files** (`results/`): [CC BY-NC-SA 4.0](results/LICENSE.md), because they are derived from CGMacros, which carries that licence.
- **Datasets** are not redistributed; each keeps its own licence (see `docs/DATA.md`).

## Author

Ahad Nadeem, independent researcher. ORCID [0009-0002-7224-6088](https://orcid.org/0009-0002-7224-6088).
