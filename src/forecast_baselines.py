"""Glucose forecasting baselines on CGMacros (rung 1 of the feature ladder: glucose history only).

Design (see DECISIONS.md):
- Run for BOTH sensors (Libre, Dexcom) with identical code and identical participant folds (D5).
- Use only the sensors' REAL readings. The minute-level file is linearly interpolated (Libre native = 15 min,
  Dexcom native = 5 min), so we rebuild a 15-minute grid from native sample times.
- Task: given glucose history up to time t, predict glucose at t + h, h in {30, 60, 120} minutes.
- Evaluation: grouped CV by participant (every test participant is unseen in training), 5 folds x 3 repeats.
  Participant is the statistical unit: we average error per participant, then bootstrap over participants.
- Models: persistence, linear trend, Ridge (global), LightGBM (global, with time-of-day).
Everything written to ../results is logged output of this script.
"""
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from cgmacros_loader import DATA

SEED = 20261117
HORIZONS = [30, 60, 120]   # minutes
N_LAGS = 8                 # 8 x 15 min = 2 h of history
GRID = 15                  # minutes
N_FOLDS, N_REPEATS = 5, 3
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)


def native_grid(d: pd.DataFrame, sensor: str) -> pd.Series:
    """15-minute series (index = minutes since participant start) built from native sensor readings only."""
    d = d.sort_values("Timestamp")
    t0 = d.Timestamp.min()
    mins = ((d.Timestamp - t0).dt.total_seconds() // 60).astype(int).to_numpy()
    s = pd.Series(d[sensor].to_numpy(float), index=mins)
    s = s[~s.index.duplicated()].reindex(np.arange(mins.min(), mins.max() + 1))
    x = s.to_numpy()
    d2 = np.abs(x[2:] - 2 * x[1:-1] + x[:-2])
    native = np.where(d2 > 1e-6)[0] + 1            # slope changes = real readings
    if len(native) < 20:
        return pd.Series(dtype=float)
    phase = pd.Series(native % GRID).mode().iloc[0]  # which minute-of-15 holds real readings
    idx = np.arange(len(x))
    keep = idx[(idx % GRID) == phase]
    return pd.Series(x[keep], index=keep + s.index[0]), t0


def rows_from_series(sid, group, g: pd.Series, t0: pd.Timestamp) -> list:
    """Turn one person's 15-minute glucose series (index = minutes since t0, NaN = gap) into forecasting samples.
    Shared by every cohort so that all cohorts go through exactly the same pipeline."""
    g = g.reindex(np.arange(g.index.min(), g.index.max() + 1, GRID))  # exact 15-min grid, NaN gaps
    v = g.to_numpy()
    n = len(v)
    rows = []
    for i in range(N_LAGS - 1, n):
        hist = v[i - N_LAGS + 1: i + 1][::-1]   # lag0 = now, lag1 = 15 min ago, ...
        if np.isnan(hist).any():
            continue
        ts = t0 + pd.Timedelta(minutes=int(g.index[i]))
        base = dict(subject=sid, group=group, tod=ts.hour * 60 + ts.minute, tmin=int(g.index[i]),
                    **{f"lag{k}": hist[k] for k in range(N_LAGS)})
        for h in HORIZONS:
            j = i + h // GRID
            if j < n and not np.isnan(v[j]):
                base[f"y{h}"] = v[j]
        rows.append(base)
    return rows


def make_samples(df: pd.DataFrame, sensor: str) -> pd.DataFrame:
    rows = []
    for sid, d in df.groupby("subject"):
        out = native_grid(d, sensor)
        if not isinstance(out, tuple):
            continue
        g, t0 = out
        rows += rows_from_series(sid, d.group.iloc[0], g, t0)
    return pd.DataFrame(rows)


def feats(X: pd.DataFrame, gbm: bool) -> np.ndarray:
    lags = X[[f"lag{k}" for k in range(N_LAGS)]].to_numpy()
    f = [lags, (lags[:, [0]] - lags[:, [1]]), (lags[:, [0]] - lags[:, [2]])]
    if gbm:
        ang = 2 * np.pi * X["tod"].to_numpy() / 1440
        f += [np.sin(ang)[:, None], np.cos(ang)[:, None]]
    return np.hstack(f)


def fit_predict(name, tr, te, h):
    y = f"y{h}"
    if name == "persistence":
        return te["lag0"].to_numpy()
    if name == "linear_trend":
        slope = (te["lag0"] - te["lag2"]) / 30.0   # mg/dL per minute over the last 30 min
        return np.clip(te["lag0"] + slope * h, 40, 400).to_numpy()
    tr = tr.dropna(subset=[y])
    if name == "ridge":
        m = Ridge(alpha=10.0).fit(feats(tr, False), tr[y])
        return m.predict(feats(te, False))
    if name == "lightgbm":
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=SEED, verbose=-1, n_jobs=4)
        m.fit(feats(tr, True), tr[y])
        return m.predict(feats(te, True))
    raise ValueError(name)


def run(sensor: str, df: pd.DataFrame):
    S = make_samples(df, sensor)
    subjects = np.array(sorted(S.subject.unique()))
    print(f"[{sensor}] samples={len(S)} participants={len(subjects)} "
          f"clipped_at_40={(S.lag0 <= 40).mean():.3%}", flush=True)
    models = ["persistence", "linear_trend", "ridge", "lightgbm"]
    rows = []
    for rep in range(N_REPEATS):
        rng = np.random.default_rng(SEED + rep)
        perm = rng.permutation(subjects)
        folds = np.array_split(perm, N_FOLDS)             # same folds for every model and both sensors' seeds
        for k, test_ids in enumerate(folds):
            te_mask = S.subject.isin(test_ids)
            tr, te = S[~te_mask], S[te_mask]
            for h in HORIZONS:
                te_h = te.dropna(subset=[f"y{h}"])
                for m in models:
                    p = fit_predict(m, tr, te_h, h)
                    e = p - te_h[f"y{h}"].to_numpy()
                    tmp = pd.DataFrame({"subject": te_h.subject.to_numpy(), "e": e, "ae": np.abs(e), "se": e ** 2})
                    pp = tmp.groupby("subject").agg(mae=("ae", "mean"), mse=("se", "mean"), bias=("e", "mean"), n=("e", "size"))
                    pp["rmse"] = np.sqrt(pp.mse)
                    pp = pp.drop(columns="mse").reset_index()
                    pp["sensor"], pp["model"], pp["horizon"], pp["repeat"] = sensor, m, h, rep
                    rows.append(pp)
    res = pd.concat(rows, ignore_index=True)
    grp = S.groupby("subject").group.first()
    res["group"] = res.subject.map(grp)
    return res, len(S), len(subjects)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    out = []
    for (sensor, model, h), d in res.groupby(["sensor", "model", "horizon"]):
        per = d.groupby("subject")[["rmse", "mae", "bias"]].mean()   # average over repeats, one value per participant
        for metric in ["rmse", "mae", "bias"]:
            v = per[metric].to_numpy()
            boots = [rng.choice(v, len(v)).mean() for _ in range(2000)]
            out.append(dict(sensor=sensor, model=model, horizon=h, metric=metric, mean=v.mean(),
                            ci_low=np.percentile(boots, 2.5), ci_high=np.percentile(boots, 97.5), n_participants=len(v)))
    return pd.DataFrame(out)


if __name__ == "__main__":
    t0 = time.time()
    df = pd.read_parquet(os.path.join(DATA, "processed", "cgmacros_all.parquet"))
    all_res, meta = [], {}
    for sensor in ["libre", "dexcom"]:
        r, n, ns = run(sensor, df)
        all_res.append(r)
        meta[sensor] = dict(samples=n, participants=ns)
    res = pd.concat(all_res, ignore_index=True)
    res.to_csv(os.path.join(RES, "baselines_per_participant.csv"), index=False)
    summ = summarise(res)
    summ.to_csv(os.path.join(RES, "baselines_summary.csv"), index=False)
    cfg = dict(seed=SEED, horizons=HORIZONS, n_lags=N_LAGS, grid_minutes=GRID, n_folds=N_FOLDS,
               n_repeats=N_REPEATS, meta=meta, runtime_sec=round(time.time() - t0, 1),
               note="rung 1: glucose history only; grouped CV by participant; participant = unit for CIs")
    json.dump(cfg, open(os.path.join(RES, "baselines_config.json"), "w"), indent=2)
    piv = summ[summ.metric == "rmse"].pivot_table(index=["sensor", "model"], columns="horizon", values="mean").round(1)
    print(piv.to_string())
    print(json.dumps(cfg, indent=2))
