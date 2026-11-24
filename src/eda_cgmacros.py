"""First exploratory plots for CGMacros. Saves PNGs to ../figures and a stats JSON to ../results."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cgmacros_loader import DATA

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "..", "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

df = pd.read_parquet(os.path.join(DATA, "processed", "cgmacros_all.parquet"))
COL = {"healthy": "#2a9d8f", "prediabetes": "#e9a23b", "t2d": "#c8553d"}
stats = {}

# ---- 0. basic counts
per = df.groupby("subject").agg(
    group=("group", "first"), rows=("Timestamp", "size"),
    libre_obs=("libre", lambda s: s.notna().mean()), dexcom_obs=("dexcom", lambda s: s.notna().mean()),
    meals=("meal_type", lambda s: s.notna().sum()),
    days=("Timestamp", lambda s: (s.max() - s.min()).total_seconds() / 86400),
)
stats["participants"] = int(per.shape[0])
stats["group_counts"] = per["group"].value_counts().to_dict()
stats["meals_total"] = int(per["meals"].sum())
stats["meals_per_participant_median"] = float(per["meals"].median())
stats["days_median"] = float(per["days"].median())
stats["libre_missing_median"] = float(1 - per["libre_obs"].median())
stats["dexcom_missing_median"] = float(1 - per["dexcom_obs"].median())
stats["participants_without_dexcom"] = int((per["dexcom_obs"] == 0).sum())
stats["participants_without_libre"] = int((per["libre_obs"] == 0).sum())

# ---- 1. traces with meal markers: one participant per group (median-length)
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
for ax, g in zip(axes, ["healthy", "prediabetes", "t2d"]):
    cand = per[(per.group == g) & (per.libre_obs > 0.5)]
    sid = cand.index[len(cand) // 2]
    d = df[df.subject == sid].iloc[: 3 * 1440]
    ax.plot(d.Timestamp, d.libre, color=COL[g], lw=1.2, label="Libre")
    ax.plot(d.Timestamp, d.dexcom, color="k", lw=0.8, alpha=0.6, label="Dexcom")
    for _, r in d[d.meal_type.notna()].iterrows():
        ax.axvline(r.Timestamp, color="grey", ls=":", lw=0.8)
        ax.text(r.Timestamp, ax.get_ylim()[1], str(r.meal_type)[0], fontsize=8, va="top", ha="center")
    ax.set_title(f"{g}  (participant {sid}, first 3 days; dotted = meal start: B/L/D/S)", fontsize=10)
    ax.set_ylabel("glucose (mg/dL)")
    ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "01_traces_with_meals.png"), dpi=150)
plt.close(fig)

# ---- 2. missing data per participant
order = per.sort_values(["group", "libre_obs"]).index
fig, ax = plt.subplots(figsize=(12, 4.5))
x = np.arange(len(order))
ax.bar(x - 0.2, per.loc[order, "libre_obs"], 0.4, label="Libre", color="#4c78a8")
ax.bar(x + 0.2, per.loc[order, "dexcom_obs"], 0.4, label="Dexcom", color="#222")
ax.set_xticks(x)
ax.set_xticklabels(order, rotation=90, fontsize=7)
ax.set_ylabel("fraction of minutes with a reading")
ax.set_title("Sensor coverage per participant (sorted by group)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "02_sensor_coverage.png"), dpi=150)
plt.close(fig)

# ---- 3. Libre vs Dexcom agreement
both = df.dropna(subset=["libre", "dexcom"])
stats["minutes_with_both_sensors"] = int(len(both))
if len(both):
    diff = both.libre - both.dexcom
    stats["libre_minus_dexcom_mean"] = float(diff.mean())
    stats["libre_minus_dexcom_sd"] = float(diff.std())
    stats["libre_dexcom_corr"] = float(both.libre.corr(both.dexcom))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    s = both.sample(min(len(both), 40000), random_state=0)
    axs[0].scatter(s.dexcom, s.libre, s=2, alpha=0.2, color="#4c78a8")
    axs[0].plot([40, 400], [40, 400], "k--", lw=1)
    axs[0].set_xlabel("Dexcom (mg/dL)"); axs[0].set_ylabel("Libre (mg/dL)")
    axs[0].set_title(f"Agreement (r={stats['libre_dexcom_corr']:.2f})")
    m = (s.libre + s.dexcom) / 2
    axs[1].scatter(m, s.libre - s.dexcom, s=2, alpha=0.2, color="#4c78a8")
    axs[1].axhline(diff.mean(), color="k", lw=1)
    axs[1].axhline(diff.mean() + 1.96 * diff.std(), color="k", ls="--", lw=0.8)
    axs[1].axhline(diff.mean() - 1.96 * diff.std(), color="k", ls="--", lw=0.8)
    axs[1].set_xlabel("mean of sensors"); axs[1].set_ylabel("Libre - Dexcom")
    axs[1].set_title("Bland-Altman")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "03_sensor_agreement.png"), dpi=150)
    plt.close(fig)

# ---- 4. glucose distribution per group (Libre; one value per participant mean + all minutes)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
data = [df[(df.group == g)].libre.dropna().values for g in ["healthy", "prediabetes", "t2d"]]
parts = axs[0].violinplot(data, showmedians=True)
for pc, g in zip(parts["bodies"], ["healthy", "prediabetes", "t2d"]):
    pc.set_facecolor(COL[g]); pc.set_alpha(0.7)
axs[0].set_xticks([1, 2, 3]); axs[0].set_xticklabels(["healthy", "prediabetes", "t2d"])
axs[0].set_ylabel("Libre glucose (mg/dL), all minutes")
axs[0].set_title("Glucose distribution by HbA1c-derived group")
sd_by = df.groupby("subject").agg(group=("group", "first"), sd=("libre", "std"), a1c=("a1c", "first"))
for g in COL:
    q = sd_by[sd_by.group == g]
    axs[1].scatter(q.a1c, q.sd, color=COL[g], label=g, s=30)
axs[1].set_xlabel("HbA1c (%)"); axs[1].set_ylabel("within-person glucose SD (mg/dL)")
axs[1].set_title("More diabetic = more variable glucose?")
axs[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "04_group_distributions.png"), dpi=150)
plt.close(fig)

# ---- 5. mean daily profile per group
d = df.dropna(subset=["libre"]).copy()
d["hour"] = d.Timestamp.dt.hour + d.Timestamp.dt.minute / 60
d["hbin"] = (d.hour // 0.5) * 0.5
fig, ax = plt.subplots(figsize=(9, 4.5))
for g in COL:
    pp = d[d.group == g].groupby(["subject", "hbin"]).libre.mean().groupby("hbin").agg(["mean", "sem"])
    ax.plot(pp.index, pp["mean"], color=COL[g], label=g)
    ax.fill_between(pp.index, pp["mean"] - pp["sem"], pp["mean"] + pp["sem"], color=COL[g], alpha=0.2)
ax.set_xlabel("hour of day"); ax.set_ylabel("Libre glucose (mg/dL)")
ax.set_title("Average daily glucose profile by group (mean of participant means ± s.e.m.)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "05_daily_profiles.png"), dpi=150)
plt.close(fig)

per.to_csv(os.path.join(RES, "cgmacros_participant_summary.csv"))
with open(os.path.join(RES, "cgmacros_eda_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))
