"""Learning-curve figure for personalisation (from results/personal_summary.csv)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
s = pd.read_csv(os.path.join(HERE, "..", "results", "personal_summary.csv"))
methods = [("M1_bias", "bias fix", "#e9a23b", "o"), ("M2_recalib", "linear recalibration", "#c8553d", "s"),
           ("M3_weighted_refit", "weighted refit", "#2a9d8f", "^"), ("M4_baseline_feats", "personal baseline features", "#264653", "D")]
fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
for r, sensor in enumerate(["libre", "dexcom"]):
    for c, h in enumerate([30, 60, 120]):
        ax = axs[r, c]
        for m, lab, col, mk in methods:
            d = s[(s.sensor == sensor) & (s.horizon == h) & (s.method == m)].sort_values("K")
            ax.errorbar(d.K, d.gain, yerr=[d.gain - d.gain_lo, d.gain_hi - d.gain], color=col, marker=mk, capsize=3,
                        label=lab, alpha=0.9)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks([1, 2, 3, 5])
        ax.set_title(f"{sensor.capitalize()}, {h} min ahead", fontsize=10)
        ax.set_xlabel("days of the person's own data used")
        ax.grid(alpha=0.3)
    axs[r, 0].set_ylabel("RMSE reduction vs global model (mg/dL)\nhigher = better")
axs[0, 0].legend(fontsize=7.5, loc="upper left")
fig.suptitle("Personalisation learning curves on unseen people (44 people, tested on day 5 onward). "
             "Controls using another person's data hurt (see knowledge base).", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "..", "figures", "10_personalisation.png"), dpi=150)
print("saved")
