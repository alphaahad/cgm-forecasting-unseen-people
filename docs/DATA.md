# Data

The datasets are public and are **not** stored in this repository. `python src/download_datasets.py` downloads them from their original hosts into `$MBZUAI_DATA` (default `D:\MBZUAI_data`) and checks the checksums that the hosts publish. Please read each dataset's page and licence before use.

## The three datasets

| | CGMacros | Shanghai T1DM / T2DM | BUT CGM |
|---|---|---|---|
| Host | PhysioNet | figshare | PhysioNet |
| Version / id | v1.0.0, doi:10.13026/3z8q-x658 | collection doi:10.6084/m9.figshare.c.6310860 | v1.0.0, doi:10.13026/hv8f-3w60 |
| Licence | CC BY-NC-SA 4.0 | CC BY 4.0 | CC BY 4.0 |
| People used | 45 | 100 with T2D (the 12 with T1D are not used) | 58 with CGM data (60 subjects, 65 recordings, 63 with CGM) |
| Sensor | FreeStyle Libre Pro and Dexcom G6 Pro, worn together | FreeStyle Libre H, one reading every 15 minutes | Dexcom G7, one reading every 5 minutes |
| Glucose units | mg/dL | mg/dL | **mmol/L** (we multiply by 18.016) |
| What we download | the zip, the data dictionaries | one zip (3.7 MB) | everything except the raw ECG and accelerometer files (about 1.2 GB of CGM, meals, heart rate, steps, sleep, questionnaires) |
| Download size | about 660 MB | about 4 MB | about 1.2 GB |

## Required citations

- **CGMacros:** Gutierrez-Osuna R, Kerr D, Mortazavi B, Das A. *CGMacros: a scientific dataset for personalized nutrition and diet monitoring* (version 1.0.0). PhysioNet. 2025. doi:10.13026/3z8q-x658. The dataset page also asks for the Scientific Data paper (listed there as under review) and the standard PhysioNet citation (Pollard et al., *Nature Health*, 2026, doi:10.1038/s44360-026-00096-z). Use the exact text on the dataset page.
- **BUT CGM:** Saclova L, Nemcova A, Goldschmidt T, Kralik M, Chlibkova D, Smital L, Smisek R, Hubalek J. *Brno University of Technology Continuous Glycaemia, ECG, Activity and Food Intake Database (BUT CGM)* (version 1.0.0). PhysioNet. doi:10.13026/hv8f-3w60.
- **Shanghai T1DM/T2DM:** *Chinese diabetes datasets for data-driven machine learning*, Scientific Data, 2023, doi:10.1038/s41597-023-01940-7. Data: figshare collection doi:10.6084/m9.figshare.c.6310860.

## Known quirks in the data (and how the code handles them)

- **CGMacros minute-level glucose is interpolated.** About 94% of Libre minutes and 83% of Dexcom minutes lie on straight lines between the real readings (every 15 and 5 minutes). We rebuild each series from the real readings only (`native_grid`).
- **The two CGMacros sensors disagree.** Libre reads about 32 mg/dL lower than Dexcom on average, and neither matches the fingersticks. All CGMacros results are reported for both sensors.
- **No health-group label.** We derive groups from HbA1c (below 5.7% normal, 5.7 to 6.4% pre-diabetes, 6.5% or more T2D), which reproduces the published group sizes (15, 16, 14). The data dictionary labels HbA1c as mmol/mol, but the values are percentages.
- **Meal fields.** Meal-type labels differ in capitalisation (`dinner`, `Dinner`, `snack 1`). The "percent consumed" field holds values such as 1, 2, 200 and 300 and contradicts the dictionary's 0 to 100, so it is **not used**. Some nutrient values exceed the dictionary's ranges (carbohydrate up to 761 against a documented maximum of 176) and are capped at the documented ranges where the richer-meal features are built.
- **Microbiome table.** `microbes.csv` is presence or absence (0 or 1) of about 1,980 taxa; one person has no taxa data. Only 42 of 45 people have gut-health scores.
- **BUT CGM identifiers.** The first three digits of an ID are the subject and the last three the measurement. There are 60 subjects and 65 recordings; two recordings (150001 and 158001) have no CGM file.
- **Shanghai files.** File names are `patient_visit_date`; some patients have two recordings. We group by patient so nobody is split between training and testing.

## Dataset licence reminder

CGMacros is CC BY-NC-SA 4.0: attribution, non-commercial use, and share-alike. The files in `results/` are derived from it and carry the same licence (see `results/LICENSE.md`). Do not use this repository's results commercially.
