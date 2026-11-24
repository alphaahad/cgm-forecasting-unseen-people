"""Feature ladder on CGMacros (D7a order): R1 glucose -> R2 +health group -> R3 +meals -> R4 +activity.

Same rows, same people, same folds for every rung, so differences come from the added information only.
Controls: R3s and R4s keep the same columns but SHUFFLE the meal / activity features across time within each
person, breaking the link to the glucose curve. If a rung "helps" only because of extra columns, the control
would help too. It should not.
Run for BOTH sensors (D5). LightGBM, grouped CV by participant, 5 folds x 3 repeats, bootstrap over participants.
"""
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from cgmacros_loader import DATA
from forecast_baselines import (HORIZONS, N_FOLDS, N_REPEATS, RES, SEED, feats, make_samples)

MEAL_COLS = ["mins_since_meal", "meal_kcal", "meal_carbs", "meal_protein", "meal_fat", "meal_fiber",
             "carbs_2h", "kcal_2h"]
ACT_COLS = ["hr_30", "mets_30", "calact_30"]
GRP = {"healthy": 0, "prediabetes": 1, "t2d": 2}
RUNGS = {
    "R1_glucose": [],
    "R2_+group": ["grp"],
    "R3_+meals": ["grp"] + MEAL_COLS,
    "R3s_meals_shuffled": ["grp"] + MEAL_COLS,
    "R4_+activity": ["grp"] + MEAL_COLS + ACT_COLS,
    "R4s_activity_shuffled": ["grp"] + MEAL_COLS + ACT_COLS,
}


def context_features(df: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    """Add meal and activity features known at time t (nothing from the future)."""
    S = S.copy()
    S["grp"] = S.group.map(GRP)
    for c in MEAL_COLS + ACT_COLS:
        S[c] = np.nan
    for sid, d in df.groupby("subject"):
        d = d.sort_values("Timestamp")
        t0 = d.Timestamp.min()
        mins = ((d.Timestamp - t0).dt.total_seconds() // 60).astype(int).to_numpy()
        n = mins.max() + 1
        hr, mets, cal = (np.full(n, np.nan) for _ in range(3))
        hr[mins], mets[mins], cal[mins] = d["HR"].to_numpy(float), d["METs"].to_numpy(float), d["cal_activity"].to_numpy(float)
        meals = d[d.meal_type.notna()]
        mm = ((meals.Timestamp - t0).dt.total_seconds() // 60).astype(int).to_numpy()
        order = np.argsort(mm)
        mm = mm[order]
        mv = {k: meals[k].to_numpy(float)[order] for k in ["meal_kcal", "carbs", "protein", "fat", "fiber"]}
        idx = S.index[S.subject == sid]
        t = S.loc[idx, "tmin"].to_numpy()
        # meals: last logged meal at or before t
        pos = np.searchsorted(mm, t, side="right") - 1
        have = pos >= 0
        since = np.where(have, t - mm[np.clip(pos, 0, None)], 360).astype(float) if len(mm) else np.full(len(t), 360.0)
        since = np.minimum(since, 360)
        S.loc[idx, "mins_since_meal"] = since
        for out, key in [("meal_kcal", "meal_kcal"), ("meal_carbs", "carbs"), ("meal_protein", "protein"),
                         ("meal_fat", "fat"), ("meal_fiber", "fiber")]:
            if len(mm):
                v = mv[key][np.clip(pos, 0, None)]
                S.loc[idx, out] = np.where(have & (since < 360), v, 0.0)
            else:
                S.loc[idx, out] = 0.0
        # sums over the last 2 hours
        for out, key in [("carbs_2h", "carbs"), ("kcal_2h", "meal_kcal")]:
            vals = np.nan_to_num(mv[key]) if len(mm) else np.array([])
            cs = np.concatenate([[0.0], np.cumsum(vals)])
            hi = np.searchsorted(mm, t, side="right")
            lo = np.searchsorted(mm, t - 120, side="right")
            S.loc[idx, out] = cs[hi] - cs[lo]
        # activity over the last 30 minutes (Fitbit), NaN if no data in the window
        def roll(a, fn):
            r = np.full(len(t), np.nan)
            for k, tt in enumerate(t):
                w = a[max(0, tt - 29): tt + 1]
                if np.isfinite(w).any():
                    r[k] = fn(w)
            return r
        S.loc[idx, "hr_30"] = roll(hr, np.nanmean)
        S.loc[idx, "mets_30"] = roll(mets, np.nanmean)
        S.loc[idx, "calact_30"] = roll(cal, np.nansum)
    return S


def shuffled(S: pd.DataFrame, cols, seed) -> pd.DataFrame:
    """Permute the rows of `cols` within each person (keeps each person's distribution, destroys timing)."""
    S = S.copy()
    rng = np.random.default_rng(seed)
    for sid, ix in S.groupby("subject").groups.items():
        ix = np.array(list(ix))
        S.loc[ix, cols] = S.loc[rng.permutation(ix), cols].to_numpy()
    return S


def X_for(S, extra):
    base = feats(S, True)
    return np.hstack([base, S[extra].to_numpy(float)]) if extra else base


def fit_pred(tr, te, h, extra):
    tr = tr.dropna(subset=[f"y{h}"])
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=4)
    m.fit(X_for(tr, extra), tr[f"y{h}"])
    return m.predict(X_for(te, extra))


def run(sensor, df):
    S = context_features(df, make_samples(df, sensor)).reset_index(drop=True)
    S_meal_sh = shuffled(S, MEAL_COLS, SEED + 1)
    S_act_sh = shuffled(S, ACT_COLS, SEED + 2)
    people = np.array(sorted(S.subject.unique()))
    print(f"[{sensor}] samples={len(S)} people={len(people)} "
          f"meal-in-last-2h={(S.carbs_2h > 0).mean():.1%} fitbit-available={S.hr_30.notna().mean():.1%}", flush=True)
    rows = []
    for rep in range(N_REPEATS):
        perm = np.random.default_rng(SEED + rep).permutation(people)
        for test_ids in np.array_split(perm, N_FOLDS):
            te_mask = S.subject.isin(test_ids).to_numpy()
            for h in HORIZONS:
                ok = S[f"y{h}"].notna().to_numpy()
                for rung, cols in RUNGS.items():
                    if rung == "R3s_meals_shuffled":
                        SS = S_meal_sh
                    elif rung == "R4s_activity_shuffled":
                        SS = S.copy()
                        SS[MEAL_COLS] = S[MEAL_COLS]
                        SS[ACT_COLS] = S_act_sh[ACT_COLS]
                    else:
                        SS = S
                    tr, te = SS[~te_mask], SS[te_mask & ok]
                    p = fit_pred(tr, te, h, cols)
                    e = p - te[f"y{h}"].to_numpy()
                    tmp = pd.DataFrame({"subject": te.subject.to_numpy(), "se": e ** 2, "ae": np.abs(e), "e": e})
                    pp = tmp.groupby("subject").agg(mse=("se", "mean"), mae=("ae", "mean"), bias=("e", "mean")).reset_index()
                    pp["rmse"] = np.sqrt(pp.mse)
                    pp["sensor"], pp["rung"], pp["horizon"], pp["repeat"] = sensor, rung, h, rep
                    rows.append(pp.drop(columns="mse"))
    return pd.concat(rows, ignore_index=True), S


if __name__ == "__main__":
    t0 = time.time()
    df = pd.read_parquet(os.path.join(DATA, "processed", "cgmacros_all.parquet"))
    out, stats = [], {}
    for sensor in ["libre", "dexcom"]:
        r, S = run(sensor, df)
        out.append(r)
        stats[sensor] = dict(samples=len(S), people=int(S.subject.nunique()),
                             meal_in_last_2h=float((S.carbs_2h > 0).mean()), fitbit_available=float(S.hr_30.notna().mean()))
        print(sensor, "done", round(time.time() - t0), "s", flush=True)
    res = pd.concat(out, ignore_index=True)
    res.to_csv(os.path.join(RES, "ladder_per_participant.csv"), index=False)
    per = res.groupby(["sensor", "rung", "horizon", "subject"])[["rmse", "mae", "bias"]].mean().reset_index()
    rng = np.random.default_rng(SEED)
    summ = []
    for (sensor, h), d in per.groupby(["sensor", "horizon"]):
        base = d[d.rung == "R1_glucose"].set_index("subject").rmse
        for rung, dd in d.groupby("rung"):
            v = dd.set_index("subject").rmse
            diff = (base - v).dropna().to_numpy()
            b = [rng.choice(v.to_numpy(), len(v)).mean() for _ in range(2000)]
            bd = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
            summ.append(dict(sensor=sensor, horizon=h, rung=rung, n_people=len(v), rmse=v.mean(),
                             rmse_lo=np.percentile(b, 2.5), rmse_hi=np.percentile(b, 97.5),
                             gain_vs_R1=diff.mean(), gain_lo=np.percentile(bd, 2.5), gain_hi=np.percentile(bd, 97.5),
                             people_better=int((diff > 0).sum())))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(RES, "ladder_summary.csv"), index=False)
    json.dump(dict(seed=SEED, stats=stats, runtime_sec=round(time.time() - t0, 1), rungs=RUNGS),
              open(os.path.join(RES, "ladder_config.json"), "w"), indent=2)
    pd.set_option("display.width", 250)
    print(summ.round(2)[["sensor", "horizon", "rung", "n_people", "rmse", "gain_vs_R1", "gain_lo", "gain_hi", "people_better"]].to_string(index=False))
