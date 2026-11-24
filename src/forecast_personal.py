"""Personalisation (D11a): does a little personal data help? Design fixed before running, see DECISIONS.md.

Global model = LightGBM on glucose history (rung R1), trained only on OTHER people (grouped CV 5 x 3).
For each held-out person, the first K days (K in 1,2,3,5) are adaptation data; everyone is scored on the SAME later
window (day 5 onward). Adaptation rows are those whose target lies inside the first K days (no look-ahead).
Methods: M0 global | M1 bias fix | M2 linear recalibration | M3 weighted refit | M4 personal baseline features.
Controls: M1c, M2c (another person's correction), M4c (another person's baseline stats).
M4 is trained on rows after day K only, and compared with M0K (same restricted training, no personal stats).
"""
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from cgmacros_loader import DATA
from forecast_baselines import HORIZONS, N_FOLDS, N_REPEATS, RES, SEED, feats, make_samples

KS = [1, 2, 3, 5]
TEST_START = 5 * 1440
MIN_TEST = 50
W_REFIT = 10.0
DAY = 1440


def fit_lgbm(X, y, w=None):
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=4)
    m.fit(X, y, sample_weight=w)
    return m


def person_stats(part: pd.DataFrame, K: int) -> np.ndarray:
    v = part.loc[part.tmin < K * DAY, "lag0"].to_numpy()
    if len(v) < 20:
        return np.full(4, np.nan)
    return np.array([v.mean(), v.std(), np.percentile(v, 10), np.percentile(v, 90)])


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def run(sensor, df):
    S = make_samples(df, sensor).reset_index(drop=True)
    parts = {p: d for p, d in S.groupby("subject")}
    eligible = []
    for p, d in parts.items():
        ok = all(((d.tmin >= TEST_START) & d[f"y{h}"].notna()).sum() >= MIN_TEST for h in HORIZONS)
        has_adapt = all(((d.tmin + h < max(KS) * DAY) & d[f"y{h}"].notna()).sum() >= 30 for h in HORIZONS)
        if ok and has_adapt:
            eligible.append(p)
    eligible = np.array(sorted(eligible))
    print(f"[{sensor}] people total={len(parts)} eligible={len(eligible)}", flush=True)
    rows = []
    for rep in range(N_REPEATS):
        folds = np.array_split(np.random.default_rng(SEED + rep).permutation(np.array(sorted(parts))), N_FOLDS)
        for test_ids in folds:
            tid = [p for p in test_ids if p in set(eligible)]
            train = S[~S.subject.isin(test_ids)]
            rng = np.random.default_rng(SEED + 500 + rep + int(test_ids[0]))
            for h in HORIZONS:
                y = f"y{h}"
                tr = train.dropna(subset=[y])
                g = fit_lgbm(feats(tr, True), tr[y])
                per = {}
                for p in tid:
                    d = parts[p]
                    te = d[(d.tmin >= TEST_START) & d[y].notna()]
                    pred0 = g.predict(feats(te, True))
                    per[p] = dict(te=te, pred0=pred0, ytrue=te[y].to_numpy(), corr={})
                    for K in KS:
                        ad = d[(d.tmin + h < K * DAY) & d[y].notna()]
                        gad = g.predict(feats(ad, True))
                        res = ad[y].to_numpy() - gad
                        b = np.cov(gad, ad[y].to_numpy())[0, 1] / max(np.var(gad, ddof=1), 1e-9)
                        b = float(np.clip(b, 0.5, 1.5))
                        a = float(ad[y].mean() - b * gad.mean())
                        per[p]["corr"][K] = dict(bias=float(res.mean()), a=a, b=b, ad=ad)
                for p in tid:
                    P = per[p]
                    base = dict(sensor=sensor, repeat=rep, horizon=h, subject=p, n_test=len(P["ytrue"]))
                    rows.append(dict(base, method="persistence", K=0, rmse=rmse(P["te"]["lag0"].to_numpy() - P["ytrue"])))
                    rows.append(dict(base, method="M0_global", K=0, rmse=rmse(P["pred0"] - P["ytrue"])))
                    others = [q for q in tid if q != p]
                    for K in KS:
                        c = P["corr"][K]
                        rows.append(dict(base, method="M1_bias", K=K, rmse=rmse(P["pred0"] + c["bias"] - P["ytrue"])))
                        rows.append(dict(base, method="M2_recalib", K=K,
                                         rmse=rmse(c["a"] + c["b"] * P["pred0"] - P["ytrue"])))
                        if others:
                            o = per[others[rng.integers(len(others))]]["corr"][K]
                            rows.append(dict(base, method="M1c_control", K=K, rmse=rmse(P["pred0"] + o["bias"] - P["ytrue"])))
                            rows.append(dict(base, method="M2c_control", K=K,
                                             rmse=rmse(o["a"] + o["b"] * P["pred0"] - P["ytrue"])))
                        ad = c["ad"]
                        Xw = np.vstack([feats(tr, True), feats(ad, True)])
                        yw = np.concatenate([tr[y].to_numpy(), ad[y].to_numpy()])
                        w = np.concatenate([np.ones(len(tr)), np.full(len(ad), W_REFIT)])
                        m3 = fit_lgbm(Xw, yw, w)
                        rows.append(dict(base, method="M3_weighted_refit", K=K,
                                         rmse=rmse(m3.predict(feats(P["te"], True)) - P["ytrue"])))
                # M4: personal baseline features (per K); trained on rows after day K only
                for K in KS:
                    trK = tr[tr.tmin >= K * DAY]
                    stats_tr = np.vstack([person_stats(parts[s], K) for s in trK.subject.unique()])
                    sd = dict(zip(trK.subject.unique(), stats_tr))
                    Xs = np.hstack([feats(trK, True), np.vstack([sd[s] for s in trK.subject])])
                    mK = fit_lgbm(feats(trK, True), trK[y])          # M0K: same rows, no personal stats
                    m4 = fit_lgbm(Xs, trK[y])
                    for p in tid:
                        P = per[p]
                        st = person_stats(parts[p], K)
                        oth = [q for q in tid if q != p]
                        base = dict(sensor=sensor, repeat=rep, horizon=h, subject=p, n_test=len(P["ytrue"]))
                        Xte = feats(P["te"], True)
                        rows.append(dict(base, method="M0K_restricted_global", K=K, rmse=rmse(mK.predict(Xte) - P["ytrue"])))
                        rows.append(dict(base, method="M4_baseline_feats", K=K,
                                         rmse=rmse(m4.predict(np.hstack([Xte, np.tile(st, (len(Xte), 1))])) - P["ytrue"])))
                        if oth:
                            so = person_stats(parts[oth[rng.integers(len(oth))]], K)
                            rows.append(dict(base, method="M4c_control", K=K,
                                             rmse=rmse(m4.predict(np.hstack([Xte, np.tile(so, (len(Xte), 1))])) - P["ytrue"])))
            print(f"  [{sensor}] rep {rep} fold done", flush=True)
    return pd.DataFrame(rows), len(eligible)


if __name__ == "__main__":
    t0 = time.time()
    df = pd.read_parquet(os.path.join(DATA, "processed", "cgmacros_all.parquet"))
    outs, elig = [], {}
    for s in ["libre", "dexcom"]:
        r, n = run(s, df)
        outs.append(r)
        elig[s] = n
    res = pd.concat(outs, ignore_index=True)
    res.to_csv(os.path.join(RES, "personal_per_person.csv"), index=False)
    per = res.groupby(["sensor", "method", "K", "horizon", "subject"]).rmse.mean().reset_index()
    rng = np.random.default_rng(SEED)
    summ = []
    for (sensor, h), d in per.groupby(["sensor", "horizon"]):
        piv = d.pivot_table(index="subject", columns=["method", "K"], values="rmse")
        for (method, K) in piv.columns:
            v = piv[(method, K)].dropna()
            bm = [rng.choice(v.to_numpy(), len(v)).mean() for _ in range(2000)]
            row = dict(sensor=sensor, horizon=h, method=method, K=K, n_people=len(v), rmse=v.mean(),
                       rmse_lo=np.percentile(bm, 2.5), rmse_hi=np.percentile(bm, 97.5))
            ref = ("M0K_restricted_global", K) if method in ("M4_baseline_feats", "M4c_control") else ("M0_global", 0)
            if ref in piv.columns and (method, K) != ref:
                diff = (piv[ref] - piv[(method, K)]).dropna().to_numpy()
                bd = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
                row.update(gain=diff.mean(), gain_lo=np.percentile(bd, 2.5), gain_hi=np.percentile(bd, 97.5),
                           people_better=int((diff > 0).sum()), gain_ref=f"{ref[0]}")
            summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(RES, "personal_summary.csv"), index=False)
    json.dump(dict(seed=SEED, Ks=KS, test_start_day=5, eligible=elig, w_refit=W_REFIT, runtime_sec=round(time.time() - t0, 1)),
              open(os.path.join(RES, "personal_config.json"), "w"), indent=2)
    pd.set_option("display.width", 250)
    show = summ[summ.horizon == 60].copy()
    print(show[["sensor", "method", "K", "n_people", "rmse", "gain", "gain_lo", "gain_hi", "people_better"]].round(2).to_string(index=False))
    print(json.dumps(dict(eligible=elig, runtime_sec=round(time.time() - t0, 1))))
