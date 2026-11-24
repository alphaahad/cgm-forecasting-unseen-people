"""Log how much of the CGMacros minute-level glucose is linear interpolation, and the real reading intervals.

A minute is 'interpolated' when its second difference is exactly zero (it lies on a straight line between neighbours).
The real reading interval is the most common gap between slope changes. Output: results/sensor_interpolation.json
"""
import json
import os

import numpy as np
import pandas as pd

from cgmacros_loader import DATA

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
df = pd.read_parquet(os.path.join(DATA, "processed", "cgmacros_all.parquet"))
out = {}
for sensor in ["libre", "dexcom"]:
    shares, gaps = [], []
    for sid, d in df.groupby("subject"):
        x = d.sort_values("Timestamp")[sensor].to_numpy(float)
        d2 = np.abs(x[2:] - 2 * x[1:-1] + x[:-2])
        shares.append(float(np.nanmean(d2[~np.isnan(d2)] < 1e-6)))
        nz = np.where(d2 > 1e-6)[0]
        if len(nz) > 2:
            gaps.append(int(pd.Series(np.diff(nz)).mode().iloc[0]))
    out[sensor] = dict(mean_share_exactly_linear=float(np.mean(shares)),
                       most_common_reading_interval_min=int(pd.Series(gaps).mode().iloc[0]),
                       people=len(shares))
json.dump(out, open(os.path.join(RES, "sensor_interpolation.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
