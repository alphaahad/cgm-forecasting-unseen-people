"""Why does Libre read lower than Dexcom? Check offsets per participant and against fingersticks."""
import json
import os

import pandas as pd

from cgmacros_loader import DATA, ROOT

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
df = pd.read_parquet(os.path.join(DATA, "processed", "cgmacros_all.parquet"))
out = {}

# 1) per-participant mean offset (Libre - Dexcom)
both = df.dropna(subset=["libre", "dexcom"]).copy()
both["diff"] = both.libre - both.dexcom
pp = both.groupby("subject").agg(offset=("diff", "mean"), group=("group", "first"), n=("diff", "size"))
out["per_participant_offset_mean"] = float(pp.offset.mean())
out["per_participant_offset_sd"] = float(pp.offset.std())
out["per_participant_offset_min"] = float(pp.offset.min())
out["per_participant_offset_max"] = float(pp.offset.max())
out["participants_libre_lower"] = int((pp.offset < 0).sum())
out["participants_libre_higher"] = int((pp.offset > 0).sum())
out["offset_by_group"] = pp.groupby("group").offset.mean().round(1).to_dict()
pp.to_csv(os.path.join(RES, "sensor_offset_per_participant.csv"))

# 2) offset by glucose level (ratio vs additive)
both["bin"] = pd.cut(both.dexcom, [40, 80, 100, 120, 150, 200, 400])
out["offset_by_dexcom_level"] = both.groupby("bin", observed=True)["diff"].mean().round(1).astype(float).to_dict()
out["offset_by_dexcom_level"] = {str(k): v for k, v in out["offset_by_dexcom_level"].items()}

# 3) fingerstick ground truth: bio.csv has 3 readings with clock times; match to sensor +-10 min on day 1
bio = pd.read_csv(os.path.join(ROOT, "bio.csv"))
bio.columns = [c.strip() for c in bio.columns]
tcols = [c for c in bio.columns if c.startswith("Time (t)")]
gcols = [c for c in bio.columns if "Fingerstick" in c]
rows = []
for _, b in bio.iterrows():
    sid = int(b["subject"])
    d = df[df.subject == sid]
    if d.empty:
        continue
    day1 = d.Timestamp.min().normalize()
    for gcol, tcol in zip(gcols, tcols):
        try:
            t = pd.to_datetime(str(b[tcol]), format="%H:%M")
        except Exception:
            continue
        ts = day1 + pd.Timedelta(hours=t.hour, minutes=t.minute)
        w = d[(d.Timestamp >= ts - pd.Timedelta(minutes=10)) & (d.Timestamp <= ts + pd.Timedelta(minutes=10))]
        if w.empty:
            continue
        rows.append(dict(subject=sid, finger=b[gcol], libre=w.libre.mean(), dexcom=w.dexcom.mean()))
fs = pd.DataFrame(rows).dropna(subset=["finger"])
out["fingerstick_pairs_total"] = int(len(fs))
for s in ["libre", "dexcom"]:
    x = fs.dropna(subset=[s])
    out[f"fingerstick_pairs_{s}"] = int(len(x))
    if len(x):
        out[f"{s}_minus_finger_mean"] = float((x[s] - x.finger).mean())
        out[f"{s}_mae_vs_finger"] = float((x[s] - x.finger).abs().mean())
        out[f"{s}_mard_vs_finger_pct"] = float(((x[s] - x.finger).abs() / x.finger).mean() * 100)
fs.to_csv(os.path.join(RES, "fingerstick_vs_sensors.csv"), index=False)
with open(os.path.join(RES, "sensor_offset_stats.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(out, indent=2))
