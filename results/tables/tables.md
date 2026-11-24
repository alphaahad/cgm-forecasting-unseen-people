# Paper tables (auto-generated from results/*.csv)


## T1: Cohorts used

| cohort        |   people |   recordings | sensor                    | notes                                                            |
|:--------------|---------:|-------------:|:--------------------------|:-----------------------------------------------------------------|
| CGMacros      |       45 |           45 | Libre Pro + Dexcom G6 Pro | 15 normal glycaemia, 16 pre-diabetes, 14 T2D; 1,706 logged meals |
| Shanghai T2DM |      100 |          109 | FreeStyle Libre H         | 112462 readings, mean 140 mg/dL                                  |
| BUT CGM       |       58 |           63 | Dexcom G7                 | 58836 readings, mean 107 mg/dL, converted from mmol/L            |

## T2: RMSE (mg/dL) on unseen people, mean over people with 95% bootstrap CI

| sensor   |   horizon_min | persistence         | linear_trend        | ridge               | lightgbm            | lightgbm +group     | lightgbm +meals     | lightgbm +meals+activity   |
|:---------|--------------:|:--------------------|:--------------------|:--------------------|:--------------------|:--------------------|:--------------------|:---------------------------|
| libre    |            30 | 16.6 (15.4 to 17.9) | 20.9 (19.9 to 21.9) | 14.0 (13.3 to 14.8) | 13.7 (12.9 to 14.4) | 13.5 (12.8 to 14.4) | 13.1 (12.3 to 13.9) | 13.1 (12.4 to 14.0)        |
| libre    |            60 | 25.1 (22.6 to 27.8) | 38.7 (36.4 to 41.2) | 23.1 (21.1 to 25.2) | 21.9 (20.0 to 24.0) | 21.6 (19.6 to 23.7) | 20.9 (19.0 to 22.8) | 21.0 (19.1 to 23.1)        |
| libre    |           120 | 34.1 (29.9 to 38.5) | 66.6 (62.3 to 71.2) | 30.1 (26.8 to 33.6) | 27.9 (24.9 to 31.4) | 27.4 (24.2 to 30.8) | 26.5 (23.3 to 29.8) | 26.6 (23.5 to 30.0)        |
| dexcom   |            30 | 19.9 (18.5 to 21.4) | 27.0 (25.4 to 28.6) | 19.0 (17.7 to 20.2) | 18.3 (17.1 to 19.5) | 18.1 (17.0 to 19.4) | 16.9 (15.7 to 18.1) | 17.0 (15.8 to 18.2)        |
| dexcom   |            60 | 28.5 (25.9 to 31.0) | 47.1 (44.5 to 49.7) | 26.5 (24.2 to 29.1) | 25.4 (23.2 to 27.6) | 25.1 (23.0 to 27.5) | 23.7 (21.6 to 25.9) | 23.9 (21.8 to 26.2)        |
| dexcom   |           120 | 37.5 (33.5 to 42.0) | 76.7 (72.2 to 81.1) | 32.4 (28.9 to 36.2) | 30.4 (27.2 to 33.8) | 30.0 (26.6 to 33.3) | 28.6 (25.4 to 31.6) | 29.0 (25.8 to 32.4)        |

## T3: Null and negative results: RMSE reduction in mg/dL (positive = better, 95% CI)

| sensor   |   horizon_min | microbiome (vs meals)   | microbiome control (other person)   | richer meals (vs basic meals)   | richer meals control (shuffled)   |
|:---------|--------------:|:------------------------|:------------------------------------|:--------------------------------|:----------------------------------|
| libre    |            30 | -0.46 (-0.69 to -0.25)  | -0.56 (-0.93 to -0.29)              | 0.01 (-0.03 to 0.05)            | -0.02 (-0.03 to 0.00)             |
| libre    |            60 | -1.43 (-2.05 to -0.89)  | -1.77 (-2.74 to -1.03)              | 0.04 (-0.08 to 0.16)            | -0.04 (-0.07 to -0.01)            |
| libre    |           120 | -2.72 (-3.89 to -1.66)  | -3.17 (-4.73 to -1.87)              | 0.16 (-0.05 to 0.34)            | -0.05 (-0.09 to -0.01)            |
| dexcom   |            30 | -0.50 (-0.79 to -0.22)  | -0.63 (-1.06 to -0.29)              | 0.03 (-0.02 to 0.08)            | -0.00 (-0.02 to 0.02)             |
| dexcom   |            60 | -1.18 (-1.97 to -0.43)  | -1.53 (-2.38 to -0.83)              | 0.02 (-0.07 to 0.11)            | -0.02 (-0.06 to 0.02)             |
| dexcom   |           120 | -1.86 (-3.30 to -0.62)  | -2.89 (-4.41 to -1.59)              | -0.04 (-0.24 to 0.15)           | -0.06 (-0.13 to 0.00)             |

## T4: GRU versus LightGBM with the same inputs (gain positive = GRU better)

| sensor   |   horizon_min | model          | rmse                | gain_vs_lightgbm       |   people_better |   n |
|:---------|--------------:|:---------------|:--------------------|:-----------------------|----------------:|----:|
| dexcom   |            30 | GRU_R1_glucose | 18.6 (17.4 to 19.9) | -0.33 (-0.45 to -0.20) |              10 |  45 |
| dexcom   |            30 | GRU_R3_+meals  | 17.7 (16.6 to 18.9) | -0.75 (-0.96 to -0.53) |               6 |  45 |
| dexcom   |            60 | GRU_R1_glucose | 25.7 (23.6 to 28.0) | -0.31 (-0.47 to -0.14) |               9 |  45 |
| dexcom   |            60 | GRU_R3_+meals  | 24.5 (22.2 to 26.7) | -0.74 (-1.11 to -0.38) |               9 |  45 |
| dexcom   |           120 | GRU_R1_glucose | 31.0 (27.7 to 34.6) | -0.64 (-0.88 to -0.40) |              11 |  45 |
| dexcom   |           120 | GRU_R3_+meals  | 29.6 (26.5 to 33.0) | -0.97 (-1.40 to -0.64) |               6 |  45 |
| libre    |            30 | GRU_R1_glucose | 13.6 (12.8 to 14.4) | 0.08 (-0.13 to 0.26)   |              27 |  45 |
| libre    |            30 | GRU_R3_+meals  | 13.2 (12.4 to 14.0) | -0.12 (-0.29 to 0.06)  |              21 |  45 |
| libre    |            60 | GRU_R1_glucose | 22.1 (20.2 to 24.0) | -0.14 (-0.31 to 0.03)  |              19 |  45 |
| libre    |            60 | GRU_R3_+meals  | 21.2 (19.3 to 23.3) | -0.39 (-0.65 to -0.14) |              16 |  45 |
| libre    |           120 | GRU_R1_glucose | 28.4 (25.3 to 32.0) | -0.44 (-0.73 to -0.18) |              15 |  45 |
| libre    |           120 | GRU_R3_+meals  | 27.0 (23.8 to 30.4) | -0.51 (-0.78 to -0.22) |               9 |  45 |

## T5: Personalisation with 5 days of own data: RMSE reduction in mg/dL vs the global model (positive = better, 95% CI)

| sensor   |   horizon_min | method            | glucose_only_gain    | on_top_of_meals_gain   |
|:---------|--------------:|:------------------|:---------------------|:-----------------------|
| libre    |            30 | M1_bias           | 0.22 (0.11 to 0.36)  | 0.28 (0.15 to 0.44)    |
| libre    |            30 | M2_recalib        | 0.37 (0.19 to 0.57)  | 0.41 (0.22 to 0.61)    |
| libre    |            30 | M3_weighted_refit | 0.42 (0.32 to 0.53)  | 0.60 (0.45 to 0.78)    |
| libre    |            30 | M4_baseline_feats | 0.40 (0.20 to 0.59)  | not run                |
| libre    |            60 | M1_bias           | 0.70 (0.36 to 1.16)  | 0.80 (0.45 to 1.23)    |
| libre    |            60 | M2_recalib        | 1.21 (0.80 to 1.67)  | 1.04 (0.57 to 1.56)    |
| libre    |            60 | M3_weighted_refit | 1.08 (0.78 to 1.47)  | 1.47 (1.10 to 1.89)    |
| libre    |            60 | M4_baseline_feats | 1.07 (0.61 to 1.53)  | not run                |
| libre    |           120 | M1_bias           | 1.49 (0.80 to 2.46)  | 1.67 (0.96 to 2.63)    |
| libre    |           120 | M2_recalib        | 2.89 (2.14 to 3.81)  | 2.31 (1.40 to 3.43)    |
| libre    |           120 | M3_weighted_refit | 2.06 (1.50 to 2.73)  | 2.42 (1.74 to 3.19)    |
| libre    |           120 | M4_baseline_feats | 2.32 (1.52 to 3.09)  | not run                |
| dexcom   |            30 | M1_bias           | 0.23 (0.10 to 0.37)  | 0.24 (0.10 to 0.39)    |
| dexcom   |            30 | M2_recalib        | 0.32 (-0.40 to 0.78) | 0.27 (-0.47 to 0.75)   |
| dexcom   |            30 | M3_weighted_refit | 0.41 (0.31 to 0.52)  | 0.68 (0.50 to 0.88)    |
| dexcom   |            30 | M4_baseline_feats | 0.47 (-0.06 to 0.86) | not run                |
| dexcom   |            60 | M1_bias           | 0.61 (0.30 to 0.95)  | 0.59 (0.25 to 0.95)    |
| dexcom   |            60 | M2_recalib        | 1.12 (0.28 to 1.77)  | 0.92 (0.04 to 1.62)    |
| dexcom   |            60 | M3_weighted_refit | 0.91 (0.71 to 1.14)  | 1.34 (1.00 to 1.70)    |
| dexcom   |            60 | M4_baseline_feats | 1.06 (0.24 to 1.71)  | not run                |
| dexcom   |           120 | M1_bias           | 1.24 (0.62 to 2.02)  | 1.27 (0.60 to 2.07)    |
| dexcom   |           120 | M2_recalib        | 2.24 (1.10 to 3.33)  | 1.87 (0.76 to 2.89)    |
| dexcom   |           120 | M3_weighted_refit | 1.54 (1.20 to 1.93)  | 1.96 (1.42 to 2.59)    |
| dexcom   |           120 | M4_baseline_feats | 1.69 (0.15 to 2.97)  | not run                |

_Glucose-only gains are versus the glucose-only global model; the on-top-of-meals gains are versus the meals-only global model. M4 (personal baseline features) is measured against a global model trained on the same restricted rows (after day K) without the personal values, and was not repeated with meals._
