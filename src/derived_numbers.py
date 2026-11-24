"""Numbers that the paper quotes but that are computed from other results (ratios and percentages).
Saved to results/derived_numbers.json so that every figure in the text has a logged source.
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
e = pd.read_csv(os.path.join(RES, "external_summary.csv"))
e = e[(e.horizon == 60) & (e.model == "lightgbm")].set_index(["cohort", "train_source"])
pm = pd.read_csv(os.path.join(RES, "personal_meals_summary.csv"))
pm = pm[(pm.horizon == 60)].set_index(["sensor", "method", "K"])
day = pd.read_csv(os.path.join(RES, "sensor_offset_by_day.csv"))
days = day[day.day.astype(str).isin([str(i) for i in range(10)])]
out = dict(
    transfer_fraction_shanghai=float(e.loc[("shanghai_t2dm", "cgmacros_libre"), "gain_vs_persistence"] / e.loc[("shanghai_t2dm", "own"), "gain_vs_persistence"]),
    transfer_fraction_but=float(e.loc[("but_cgm", "cgmacros_dexcom"), "gain_vs_persistence"] / e.loc[("but_cgm", "own"), "gain_vs_persistence"]),
    meals_plus_personal_pct_libre=float(100 * pm.loc[("libre", "M3_weighted_refit", 5), "gain_vs_R1"] / pm.loc[("libre", "M0_R1_global", 0), "rmse"]),
    meals_plus_personal_pct_dexcom=float(100 * pm.loc[("dexcom", "M3_weighted_refit", 5), "gain_vs_R1"] / pm.loc[("dexcom", "M0_R1_global", 0), "rmse"]),
    sensor_gap_relative_pct_mean_days0_9=float(days.mean_rel_pct.mean()),
)
json.dump(out, open(os.path.join(RES, "derived_numbers.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
