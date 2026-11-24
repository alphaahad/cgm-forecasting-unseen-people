"""Exploratory: is the Libre-minus-Dexcom gap a first-day (warm-up) effect? (follow-up to Denham's first-day study)

Uses the real-reading 15-minute grids from forecast_baselines (not the interpolated minute data) and, for each person,
averages the Libre-minus-Dexcom difference within each day since the first CGMacros row. Person is the unit.
This is exploratory and is not part of the pre-set analyses; report it as such.
"""
import os

import numpy as np
import pandas as pd

from cgmacros_loader import DATA
from forecast_baselines import native_grid

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
df = pd.read_parquet(os.path.join(DATA, "processed", "cgmacros_all.parquet"))
rows = []
for sid, d in df.groupby("subject"):
    a = native_grid(d, "libre")
    if not isinstance(a, tuple):
        continue
    ga, _ = a                                     # Libre real readings, index = minutes since the first row
    d = d.sort_values("Timestamp")
    mins = ((d.Timestamp - d.Timestamp.min()).dt.total_seconds() // 60).astype(int).to_numpy()
    dm = pd.Series(d["dexcom"].to_numpy(float), index=mins)
    dm = dm[~dm.index.duplicated()]
    # Dexcom value at the same minute as each real Libre reading (Dexcom minute values are filled between its
    # 5-minute readings, a small error next to a gap of tens of mg/dL)
    j = pd.concat([ga.rename("libre"), dm.reindex(ga.index).rename("dexcom")], axis=1).dropna()
    j["diff"] = j.libre - j.dexcom
    j["rel"] = 100.0 * j["diff"] / j.dexcom
    j["day"] = (j.index // 1440).astype(int)
    j["first_day_block"] = np.where(j.index < 240, "0-4 h", np.where(j.index < 480, "4-8 h", np.where(j.index < 1440, "8-24 h", "later")))
    for day, g in j.groupby("day"):
        rows.append(dict(subject=sid, day=day, n=len(g), diff=g["diff"].mean(), rel=g["rel"].mean(), libre=g.libre.mean(), dexcom=g.dexcom.mean()))
    for blk, g in j[j.index < 1440].groupby("first_day_block"):
        rows.append(dict(subject=sid, day=f"first-day {blk}", n=len(g), diff=g["diff"].mean(), rel=g["rel"].mean(), libre=g.libre.mean(), dexcom=g.dexcom.mean()))
r = pd.DataFrame(rows)
r = r[r.n >= 8]                      # at least 2 hours of paired readings
r.to_csv(os.path.join(RES, "sensor_offset_by_day_per_person.csv"), index=False)
rng = np.random.default_rng(3)
out = []
for day, g in r.groupby("day"):
    v = g["diff"].to_numpy()
    b = [rng.choice(v, len(v)).mean() for _ in range(2000)]
    out.append(dict(day=day, people=len(v), mean_diff_mgdl=v.mean(), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5),
                    mean_rel_pct=g.rel.mean(), mean_libre=g.libre.mean(), mean_dexcom=g.dexcom.mean()))
o = pd.DataFrame(out)
o["day"] = o["day"].astype(str)
o = o.sort_values("day", key=lambda s: s.map(lambda x: (0 if x.startswith("first") else 1, x.zfill(3))))
o.to_csv(os.path.join(RES, "sensor_offset_by_day.csv"), index=False)
pd.set_option("display.width", 200)
print(o.round(1).to_string(index=False))
