"""Load the external CGM cohorts (Shanghai T2DM, BUT CGM) onto the same 15-minute mg/dL grid as CGMacros.

Each loader returns a list of recordings: dict(subject, recording, group, t0, series)
where series is indexed by minutes since t0 on an exact 15-minute grid (NaN = gap).
`subject` is the PERSON key used for grouped cross-validation (never split a person across train and test).
"""
import glob
import os
import warnings

import numpy as np
import pandas as pd

from cgmacros_loader import DATA

warnings.filterwarnings("ignore", message=".*xlrd engine.*")
MMOL_TO_MGDL = 18.016
GRID = 15


def to_grid(ts: pd.Series, vals: pd.Series, tol_min: float):
    """Put irregular native readings onto a 15-minute grid anchored at the first reading (nearest within tol)."""
    s = pd.Series(vals.to_numpy(float), index=pd.to_datetime(ts)).dropna().sort_index()
    s = s[~s.index.duplicated()]
    if len(s) < 20:
        return None
    t0 = s.index[0]
    idx = pd.date_range(t0, s.index[-1], freq=f"{GRID}min")
    g = s.reindex(idx, method="nearest", tolerance=pd.Timedelta(minutes=tol_min))
    return t0, pd.Series(g.to_numpy(), index=np.arange(len(g)) * GRID)


def load_shanghai_t2dm() -> list:
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, "shanghai", "extracted", "Shanghai_T2DM", "*.xls*"))):
        name = os.path.splitext(os.path.basename(f))[0]
        if name.startswith("."):
            continue
        d = pd.read_excel(f)
        cgm_col = [c for c in d.columns if "CGM" in str(c)][0]
        res = to_grid(d.iloc[:, 0], d[cgm_col], tol_min=3)
        if res is None:
            continue
        t0, g = res
        out.append(dict(subject=f"S{name.split('_')[0]}", recording=name, group="t2d", t0=t0, series=g,
                        cohort="shanghai_t2dm", sensor="libre_family"))
    return out


def load_but_cgm() -> list:
    out = []
    root = os.path.join(DATA, "but_cgm")
    for f in sorted(glob.glob(os.path.join(root, "*", "*_CGM.csv"))):
        rid = os.path.basename(f).split("_")[0]          # e.g. 101001 -> subject 101, measurement 001
        d = pd.read_csv(f, sep=None, engine="python", encoding="utf-8-sig")
        res = to_grid(d["Timestamp"], d["Glycemia"] * MMOL_TO_MGDL, tol_min=2.5)
        if res is None:
            continue
        t0, g = res
        out.append(dict(subject=f"B{rid[:3]}", recording=rid, group="healthy", t0=t0, series=g,
                        cohort="but_cgm", sensor="dexcom_g7"))
    return out


def summarise(recs: list, name: str) -> dict:
    allv = np.concatenate([r["series"].dropna().to_numpy() for r in recs])
    days = [len(r["series"]) * GRID / 1440 for r in recs]
    return dict(cohort=name, recordings=len(recs), people=len({r["subject"] for r in recs}),
                readings=int(len(allv)), glucose_mean=float(allv.mean()), glucose_sd=float(allv.std()),
                glucose_min=float(allv.min()), glucose_max=float(allv.max()),
                pct_at_or_below_40=float((allv <= 40).mean() * 100), median_days=float(np.median(days)))


if __name__ == "__main__":
    import json
    sh, bu = load_shanghai_t2dm(), load_but_cgm()
    s = [summarise(sh, "shanghai_t2dm"), summarise(bu, "but_cgm")]
    print(json.dumps(s, indent=2))
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(s, open(os.path.join(here, "..", "results", "external_cohort_summary.json"), "w"), indent=2)
    # BUT CGM roster: the first three digits of an ID are the subject, the last three the measurement
    c = pd.read_csv(os.path.join(DATA, "but_cgm", "data_completness.csv"), sep=";", encoding="utf-8-sig")
    c["subject"] = c.ID // 1000
    roster = dict(subjects=int(c.subject.nunique()), recordings=int(len(c)),
                  recordings_with_cgm=int((c.CGM == 1).sum()), subjects_with_cgm=int(c[c.CGM == 1].subject.nunique()))
    json.dump(roster, open(os.path.join(here, "..", "results", "but_roster.json"), "w"), indent=2)
    print(roster)
