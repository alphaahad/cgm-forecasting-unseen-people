"""Paired comparison: real microbiome vs the other-person control (same people, same folds)."""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
r = pd.read_csv(os.path.join(RES, "microbiome_per_participant.csv"))
per = r.groupby(["sensor", "rung", "horizon", "subject"]).rmse.mean().reset_index()
rng = np.random.default_rng(7)
rows = []
for (sensor, h), d in per.groupby(["sensor", "horizon"]):
    p = d.pivot(index="subject", columns="rung", values="rmse")
    for real, ctrl in [("R5m_meals+micro", "R5m_control_other_person"), ("R5_act+micro", "R5_control_other_person")]:
        diff = (p[ctrl] - p[real]).dropna().to_numpy()      # positive = real microbiome better than control
        b = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
        rows.append(dict(sensor=sensor, horizon=h, variant=real, real_better_than_control=diff.mean(),
                         lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), people_better=int((diff > 0).sum()), n=len(diff)))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, "microbiome_real_vs_control.csv"), index=False)
print(out.round(2).to_string(index=False))
