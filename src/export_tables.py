"""Build the paper tables T1-T5 straight from results/*.csv (never typed by hand).

Outputs results/tables/T1..T5 as CSV plus results/tables/tables.md for reading.
Run after all experiments (run_all.py does this last).
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "tables")
os.makedirs(OUT, exist_ok=True)
H = [30, 60, 120]


def rd(name):
    return pd.read_csv(os.path.join(RES, name))


def ci(m, lo, hi, d=1):
    return f"{m:.{d}f} ({lo:.{d}f} to {hi:.{d}f})"


md = ["# Paper tables (auto-generated from results/*.csv)\n"]


def emit(name, title, df, note=""):
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    md.append(f"\n## {name}: {title}\n")
    md.append(df.to_markdown(index=False))
    if note:
        md.append(f"\n_{note}_\n")


# ---- T1 cohorts
ext = json.load(open(os.path.join(RES, "external_cohort_summary.json")))
eda = json.load(open(os.path.join(RES, "cgmacros_eda_stats.json")))
t1 = pd.DataFrame([
    dict(cohort="CGMacros", people=eda["participants"], recordings=eda["participants"], sensor="Libre Pro + Dexcom G6 Pro",
         notes=f"{eda['group_counts']['healthy']} normal glycaemia, {eda['group_counts']['prediabetes']} pre-diabetes, {eda['group_counts']['t2d']} T2D; {eda['meals_total']:,} logged meals"),
    dict(cohort="Shanghai T2DM", people=ext[0]["people"], recordings=ext[0]["recordings"], sensor="FreeStyle Libre H",
         notes=f"{ext[0]['readings']} readings, mean {ext[0]['glucose_mean']:.0f} mg/dL"),
    dict(cohort="BUT CGM", people=ext[1]["people"], recordings=ext[1]["recordings"], sensor="Dexcom G7",
         notes=f"{ext[1]['readings']} readings, mean {ext[1]['glucose_mean']:.0f} mg/dL, converted from mmol/L"),
])
emit("T1", "Cohorts used", t1)

# ---- T2 main results, RMSE (95% CI over people), unseen people
b = rd("baselines_summary.csv")
b = b[b.metric == "rmse"]
lad = rd("ladder_summary.csv")
rows = []
for sensor in ["libre", "dexcom"]:
    for h in H:
        r = dict(sensor=sensor, horizon_min=h)
        for m in ["persistence", "linear_trend", "ridge", "lightgbm"]:
            x = b[(b.sensor == sensor) & (b.model == m) & (b.horizon == h)].iloc[0]
            r[m] = ci(x["mean"], x.ci_low, x.ci_high)
        for rung, lab in [("R2_+group", "+group"), ("R3_+meals", "+meals"), ("R4_+activity", "+meals+activity")]:
            x = lad[(lad.sensor == sensor) & (lad.rung == rung) & (lad.horizon == h)].iloc[0]
            r[f"lightgbm {lab}"] = ci(x.rmse, x.rmse_lo, x.rmse_hi)
        rows.append(r)
emit("T2", "RMSE (mg/dL) on unseen people, mean over people with 95% bootstrap CI", pd.DataFrame(rows))

# ---- T3 null and negative results (gain vs the rung they extend; positive = better)
mic = rd("microbiome_summary.csv")
mic = mic[mic.rung.isin(["R5m_meals+micro", "R5m_control_other_person"])]
rich = rd("rich_meals_summary.csv")
rich = rich[rich.rung.isin(["R3r_+rich_meals", "R3r_control_shuffled"])]
rows = []
for sensor in ["libre", "dexcom"]:
    for h in H:
        r = dict(sensor=sensor, horizon_min=h)
        for rung, lab in [("R5m_meals+micro", "microbiome (vs meals)"), ("R5m_control_other_person", "microbiome control (other person)")]:
            x = mic[(mic.sensor == sensor) & (mic.rung == rung) & (mic.horizon == h)].iloc[0]
            r[lab] = ci(x.gain_vs_parent, x.gain_vs_parent_lo, x.gain_vs_parent_hi, 2)
        for rung, lab in [("R3r_+rich_meals", "richer meals (vs basic meals)"), ("R3r_control_shuffled", "richer meals control (shuffled)")]:
            x = rich[(rich.sensor == sensor) & (rich.rung == rung) & (rich.horizon == h)].iloc[0]
            r[lab] = ci(x.gain_vs_R3, x.gain_vs_R3_lo, x.gain_vs_R3_hi, 2)
        rows.append(r)
emit("T3", "Null and negative results: RMSE reduction in mg/dL (positive = better, 95% CI)", pd.DataFrame(rows))

# ---- T4 GRU vs LightGBM
g = rd("gru_summary.csv")
g = g[g.rung.str.startswith("GRU")]
rows = []
for _, x in g.iterrows():
    rows.append(dict(sensor=x.sensor, horizon_min=int(x.horizon), model=x.rung, rmse=ci(x.rmse, x.rmse_lo, x.rmse_hi),
                     gain_vs_lightgbm=ci(x.gain_vs_lightgbm, x.gain_lo, x.gain_hi, 2),
                     people_better=int(x.people_better), n=int(x.n_people)))
emit("T4", "GRU versus LightGBM with the same inputs (gain positive = GRU better)", pd.DataFrame(rows))

# ---- T5 personalisation at K = 5 days
p = rd("personal_summary.csv")
pm = rd("personal_meals_summary.csv")
rows = []
for sensor in ["libre", "dexcom"]:
    for h in H:
        for meth in ["M1_bias", "M2_recalib", "M3_weighted_refit", "M4_baseline_feats"]:
            x = p[(p.sensor == sensor) & (p.horizon == h) & (p.method == meth) & (p.K == 5)].iloc[0]
            r = dict(sensor=sensor, horizon_min=h, method=meth, glucose_only_gain=ci(x.gain, x.gain_lo, x.gain_hi, 2))
            y = pm[(pm.sensor == sensor) & (pm.horizon == h) & (pm.method == meth) & (pm.K == 5)]
            r["on_top_of_meals_gain"] = ci(y.iloc[0].gain_vs_R3, y.iloc[0].gain_vs_R3_lo, y.iloc[0].gain_vs_R3_hi, 2) if len(y) else "not run"
            rows.append(r)
emit("T5", "Personalisation with 5 days of own data: RMSE reduction in mg/dL vs the global model (positive = better, 95% CI)",
     pd.DataFrame(rows), note="Glucose-only gains are versus the glucose-only global model; the on-top-of-meals gains are versus the meals-only global model. M4 (personal baseline features) is measured against a global model trained on the same restricted rows (after day K) without the personal values, and was not repeated with meals.")

open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8").write("\n".join(md))
print("wrote", sorted(os.listdir(OUT)))
