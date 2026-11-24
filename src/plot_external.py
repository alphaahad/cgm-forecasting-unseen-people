"""Figure for the external-cohort baselines (from results/external_summary.csv)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
s = pd.read_csv(os.path.join(HERE, "..", "results", "external_summary.csv"))
FIG = os.path.join(HERE, "..", "figures")
matched = {"shanghai_t2dm": "cgmacros_libre", "but_cgm": "cgmacros_dexcom"}
mism = {"shanghai_t2dm": "cgmacros_dexcom", "but_cgm": "cgmacros_libre"}
names = {"shanghai_t2dm": "Shanghai T2DM (Libre H, 100 people)", "but_cgm": "BUT CGM (healthy, Dexcom G7, 58 people)"}

fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, c in zip(axs, ["shanghai_t2dm", "but_cgm"]):
    series = [
        ("persistence (no training)", s[(s.cohort == c) & (s.model == "persistence")], "#777", "o", "-"),
        ("LightGBM trained on CGMacros, same sensor family", s[(s.cohort == c) & (s.model == "lightgbm") & (s.train_source == matched[c])], "#2a9d8f", "s", "-"),
        ("LightGBM trained on CGMacros, other sensor", s[(s.cohort == c) & (s.model == "lightgbm") & (s.train_source == mism[c])], "#e9a23b", "^", "--"),
        ("LightGBM trained on this cohort (ceiling)", s[(s.cohort == c) & (s.model == "lightgbm") & (s.job == "own")], "#264653", "D", "-"),
    ]
    for lab, d, col, mk, ls in series:
        d = d.sort_values("horizon")
        ax.errorbar(d.horizon, d.rmse, yerr=[d.rmse - d.rmse_lo, d.rmse_hi - d.rmse], color=col, marker=mk, ls=ls,
                    capsize=3, label=lab)
    ax.set_title(names[c], fontsize=10)
    ax.set_xlabel("forecast horizon (minutes)")
    ax.set_xticks([30, 60, 120])
    ax.grid(alpha=0.3)
axs[0].set_ylabel("RMSE (mg/dL) on unseen people, lower is better")
axs[0].legend(fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "08_external_cohorts.png"), dpi=150)
print("saved")
