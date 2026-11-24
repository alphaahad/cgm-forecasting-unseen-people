"""D11b: personalisation ON TOP OF meals. Design fixed before running, see DECISIONS.md.

Global model = LightGBM on glucose history + health group + basic meal features (R3), trained on other people only.
Same protocol as forecast_personal.py: K in 1,2,3,5 days of the person's own data, test window = day 5 onward,
grouped CV 5x3, both sensors. Methods: M1 bias fix, M2 linear recalibration, M3 weighted refit, controls M1c/M2c.
Reference rows: M0_R1 (glucose-only global) and M0_R3 (meals global) on the same test samples.
"""
import json
import os
import time

import numpy as np
import pandas as pd

from cgmacros_loader import DATA
from forecast_baselines import HORIZONS, N_FOLDS, N_REPEATS, RES, SEED, feats, make_samples
from forecast_ladder import MEAL_COLS, context_features
from forecast_personal import DAY, KS, MIN_TEST, TEST_START, W_REFIT, fit_lgbm, rmse

EXTRA = ["grp"] + MEAL_COLS


def XF(S):
    return np.hstack([feats(S, True), S[EXTRA].to_numpy(float)])


def run(sensor, df):
    S = context_features(df, make_samples(df, sensor)).reset_index(drop=True)
    parts = {p: d for p, d in S.groupby("subject")}
    eligible = set()
    for p, d in parts.items():
        ok = all(((d.tmin >= TEST_START) & d[f"y{h}"].notna()).sum() >= MIN_TEST for h in HORIZONS)
        ad = all(((d.tmin + h < max(KS) * DAY) & d[f"y{h}"].notna()).sum() >= 30 for h in HORIZONS)
        if ok and ad:
            eligible.add(p)
    print(f"[{sensor}] people={len(parts)} eligible={len(eligible)}", flush=True)
    rows = []
    for rep in range(N_REPEATS):
        folds = np.array_split(np.random.default_rng(SEED + rep).permutation(np.array(sorted(parts))), N_FOLDS)
        for test_ids in folds:
            tid = [p for p in test_ids if p in eligible]
            train = S[~S.subject.isin(test_ids)]
            rng = np.random.default_rng(SEED + 900 + rep + int(test_ids[0]))
            for h in HORIZONS:
                y = f"y{h}"
                tr = train.dropna(subset=[y])
                g1 = fit_lgbm(feats(tr, True), tr[y])
                g3 = fit_lgbm(XF(tr), tr[y])
                Xtr3 = XF(tr)
                per = {}
                for p in tid:
                    d = parts[p]
                    te = d[(d.tmin >= TEST_START) & d[y].notna()]
                    P = dict(te=te, ytrue=te[y].to_numpy(), p1=g1.predict(feats(te, True)), p3=g3.predict(XF(te)), corr={})
                    for K in KS:
                        ad = d[(d.tmin + h < K * DAY) & d[y].notna()]
                        gad = g3.predict(XF(ad))
                        res = ad[y].to_numpy() - gad
                        b = float(np.clip(np.cov(gad, ad[y].to_numpy())[0, 1] / max(np.var(gad, ddof=1), 1e-9), 0.5, 1.5))
                        P["corr"][K] = dict(bias=float(res.mean()), a=float(ad[y].mean() - b * gad.mean()), b=b, ad=ad)
                    per[p] = P
                for p in tid:
                    P = per[p]
                    base = dict(sensor=sensor, repeat=rep, horizon=h, subject=p, n_test=len(P["ytrue"]))
                    rows.append(dict(base, method="persistence", K=0, rmse=rmse(P["te"]["lag0"].to_numpy() - P["ytrue"])))
                    rows.append(dict(base, method="M0_R1_global", K=0, rmse=rmse(P["p1"] - P["ytrue"])))
                    rows.append(dict(base, method="M0_R3_global", K=0, rmse=rmse(P["p3"] - P["ytrue"])))
                    others = [q for q in tid if q != p]
                    for K in KS:
                        c = P["corr"][K]
                        rows.append(dict(base, method="M1_bias", K=K, rmse=rmse(P["p3"] + c["bias"] - P["ytrue"])))
                        rows.append(dict(base, method="M2_recalib", K=K, rmse=rmse(c["a"] + c["b"] * P["p3"] - P["ytrue"])))
                        if others:
                            o = per[others[rng.integers(len(others))]]["corr"][K]
                            rows.append(dict(base, method="M1c_control", K=K, rmse=rmse(P["p3"] + o["bias"] - P["ytrue"])))
                            rows.append(dict(base, method="M2c_control", K=K, rmse=rmse(o["a"] + o["b"] * P["p3"] - P["ytrue"])))
                        ad = c["ad"]
                        m3 = fit_lgbm(np.vstack([Xtr3, XF(ad)]), np.concatenate([tr[y].to_numpy(), ad[y].to_numpy()]),
                                      np.concatenate([np.ones(len(tr)), np.full(len(ad), W_REFIT)]))
                        rows.append(dict(base, method="M3_weighted_refit", K=K, rmse=rmse(m3.predict(XF(P["te"])) - P["ytrue"])))
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
    res.to_csv(os.path.join(RES, "personal_meals_per_person.csv"), index=False)
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
            for nm, ref in [("R3", ("M0_R3_global", 0)), ("R1", ("M0_R1_global", 0))]:
                if (method, K) != ref:
                    diff = (piv[ref] - piv[(method, K)]).dropna().to_numpy()
                    bd = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
                    row.update({f"gain_vs_{nm}": diff.mean(), f"gain_vs_{nm}_lo": np.percentile(bd, 2.5),
                                f"gain_vs_{nm}_hi": np.percentile(bd, 97.5), f"better_vs_{nm}": int((diff > 0).sum())})
            summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(RES, "personal_meals_summary.csv"), index=False)
    json.dump(dict(seed=SEED, Ks=KS, eligible=elig, runtime_sec=round(time.time() - t0, 1)),
              open(os.path.join(RES, "personal_meals_config.json"), "w"), indent=2)
    pd.set_option("display.width", 250)
    sh = summ[(summ.horizon == 60) & summ.method.isin(["M0_R1_global", "M0_R3_global", "M1_bias", "M2_recalib", "M3_weighted_refit", "M1c_control", "M2c_control"])]
    print(sh[["sensor", "method", "K", "n_people", "rmse", "gain_vs_R3", "gain_vs_R3_lo", "gain_vs_R3_hi", "gain_vs_R1"]].round(2).to_string(index=False))
    print(json.dumps(dict(eligible=elig, runtime_sec=round(time.time() - t0, 1))))
