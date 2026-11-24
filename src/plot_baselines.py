"""Figure + paired comparison (vs persistence) from results/baselines_per_participant.csv."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "..", "results"), os.path.join(HERE, "..", "figures")
r = pd.read_csv(os.path.join(RES, "baselines_per_participant.csv"))
per = r.groupby(["sensor", "model", "horizon", "subject"])[["rmse", "mae"]].mean().reset_index()
rng = np.random.default_rng(1)

# paired improvement over persistence (positive = better than persistence), participant is the unit
rows = []
for (sensor, h), d in per.groupby(["sensor", "horizon"]):
    base = d[d.model == "persistence"].set_index("subject").rmse
    for m in ["linear_trend", "ridge", "lightgbm"]:
        x = d[d.model == m].set_index("subject").rmse
        diff = (base - x).dropna().to_numpy()
        boots = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
        rows.append(dict(sensor=sensor, horizon=h, model=m, rmse_gain_vs_persistence=diff.mean(),
                         ci_low=np.percentile(boots, 2.5), ci_high=np.percentile(boots, 97.5),
                         participants_better=int((diff > 0).sum()), n=len(diff)))
gain = pd.DataFrame(rows)
gain.to_csv(os.path.join(RES, "baselines_gain_vs_persistence.csv"), index=False)
print(gain.round(2).to_string())

summ = pd.read_csv(os.path.join(RES, "baselines_summary.csv"))
s = summ[summ.metric == "rmse"]
colors = {"persistence": "#777", "linear_trend": "#c8553d", "ridge": "#e9a23b", "lightgbm": "#2a9d8f"}
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
for ax, sensor in zip(axs, ["libre", "dexcom"]):
    for m, c in colors.items():
        d = s[(s.sensor == sensor) & (s.model == m)].sort_values("horizon")
        ax.errorbar(d.horizon, d["mean"], yerr=[d["mean"] - d.ci_low, d.ci_high - d["mean"]], color=c, marker="o",
                    capsize=3, label=m.replace("_", " "))
    ax.set_title(f"{sensor.capitalize()}: error on unseen participants")
    ax.set_xlabel("forecast horizon (minutes)")
    ax.set_xticks([30, 60, 120])
    ax.grid(alpha=0.3)
axs[0].set_ylabel("RMSE (mg/dL), lower is better")
axs[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "06_baselines_rmse_vs_horizon.png"), dpi=150)
plt.close(fig)

# per-group RMSE of the best model at 60 min
g = r[(r.horizon == 60) & (r.model == "lightgbm")].groupby(["sensor", "group", "subject"]).rmse.mean().reset_index()
print(g.groupby(["sensor", "group"]).rmse.agg(["mean", "count"]).round(1))
g.groupby(["sensor", "group"]).rmse.agg(["mean", "count"]).reset_index().to_csv(os.path.join(RES, "baselines_by_group.csv"), index=False)
fig, ax = plt.subplots(figsize=(7, 4))
for i, sensor in enumerate(["libre", "dexcom"]):
    for j, grp in enumerate(["healthy", "prediabetes", "t2d"]):
        v = g[(g.sensor == sensor) & (g.group == grp)].rmse
        ax.scatter(np.full(len(v), j + (i - 0.5) * 0.3) + rng.normal(0, 0.03, len(v)), v,
                   color=["#4c78a8", "#222"][i], s=18, alpha=0.7, label=sensor if j == 0 else None)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["healthy", "prediabetes", "t2d"])
ax.set_ylabel("60-min RMSE (mg/dL), LightGBM, per participant")
ax.set_title("Is glucose harder to forecast in diabetes?")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "07_baseline_error_by_group.png"), dpi=150)
