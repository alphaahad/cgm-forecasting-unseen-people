"""Figure for the feature ladder (from results/ladder_summary.csv): error reduction vs glucose-only (R1)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
s = pd.read_csv(os.path.join(HERE, "..", "results", "ladder_summary.csv"))
rungs = ["R2_+group", "R3s_meals_shuffled", "R3_+meals", "R4s_activity_shuffled", "R4_+activity"]
labels = ["+group", "+meals\n(shuffled\ncontrol)", "+meals", "+meals +activity\n(activity\nshuffled)", "+meals\n+activity"]
cols = ["#999", "#d9d9d9", "#2a9d8f", "#d9d9d9", "#264653"]
fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
for r, sensor in enumerate(["libre", "dexcom"]):
    for c, h in enumerate([30, 60, 120]):
        ax = axs[r, c]
        d = s[(s.sensor == sensor) & (s.horizon == h)].set_index("rung")
        y = [d.loc[k, "gain_vs_R1"] for k in rungs]
        lo = [d.loc[k, "gain_vs_R1"] - d.loc[k, "gain_lo"] for k in rungs]
        hi = [d.loc[k, "gain_hi"] - d.loc[k, "gain_vs_R1"] for k in rungs]
        ax.bar(range(len(rungs)), y, yerr=[lo, hi], color=cols, capsize=3, edgecolor="k", linewidth=0.5)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(rungs)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(f"{sensor.capitalize()}, {h} min ahead", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axs[r, 0].set_ylabel("RMSE reduction vs glucose-only (mg/dL)\nhigher = better")
fig.suptitle("Feature ladder: does extra information help on unseen people? (grey bars = shuffled controls)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "..", "figures", "09_feature_ladder.png"), dpi=150)
print("saved")
