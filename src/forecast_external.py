"""External-cohort baselines (D6d): both jobs.

Job 1 (transfer): train on ALL of CGMacros (per sensor), test on every person in an external cohort.
                  Persistence and linear-trend need no training and are scored on the same people.
                  Sensor-matched pairings are Libre -> Shanghai and Dexcom -> BUT; the other two are a control.
Job 2 (own ceiling): train and test inside each external cohort with grouped CV by person (5 folds x 3 repeats).
Everything is rung 1 (glucose history only). Person = statistical unit.
"""
import json
import os
import time

import numpy as np
import pandas as pd

from cgmacros_loader import DATA
from external_loaders import load_but_cgm, load_shanghai_t2dm
from forecast_baselines import (HORIZONS, N_FOLDS, N_REPEATS, SEED, RES, fit_predict, make_samples, rows_from_series)

MODELS = ["persistence", "linear_trend", "ridge", "lightgbm"]


def samples_for(recs):
    rows = []
    for r in recs:
        rows += rows_from_series(r["subject"], r["group"], r["series"], r["t0"])
    return pd.DataFrame(rows)


def per_person(te_h, p, **meta):
    e = p - te_h[f"y{meta['horizon']}"].to_numpy()
    tmp = pd.DataFrame({"subject": te_h.subject.to_numpy(), "e": e, "ae": np.abs(e), "se": e ** 2})
    pp = tmp.groupby("subject").agg(mae=("ae", "mean"), mse=("se", "mean"), bias=("e", "mean"), n=("e", "size"))
    pp["rmse"] = np.sqrt(pp.mse)
    pp = pp.drop(columns="mse").reset_index()
    for k, v in meta.items():
        pp[k] = v
    return pp


if __name__ == "__main__":
    t0 = time.time()
    cg = pd.read_parquet(os.path.join(DATA, "processed", "cgmacros_all.parquet"))
    train = {s: make_samples(cg, s) for s in ["libre", "dexcom"]}
    cohorts = {"shanghai_t2dm": samples_for(load_shanghai_t2dm()), "but_cgm": samples_for(load_but_cgm())}
    meta = {c: dict(samples=len(S), people=int(S.subject.nunique())) for c, S in cohorts.items()}
    print(meta, flush=True)
    rows = []

    # ---- Job 1: transfer from CGMacros (+ untrained baselines on the same people)
    for cname, S in cohorts.items():
        for h in HORIZONS:
            te = S.dropna(subset=[f"y{h}"])
            for m in ["persistence", "linear_trend"]:
                rows.append(per_person(te, fit_predict(m, None, te, h), cohort=cname, job="baseline",
                                       train_source="none", model=m, horizon=h, repeat=0))
            for tsens, tr in train.items():
                for m in ["ridge", "lightgbm"]:
                    rows.append(per_person(te, fit_predict(m, tr, te, h), cohort=cname, job="transfer",
                                           train_source=f"cgmacros_{tsens}", model=m, horizon=h, repeat=0))
        print("job1 done", cname, round(time.time() - t0), "s", flush=True)

    # ---- Job 2: own ceiling, grouped CV by person
    for cname, S in cohorts.items():
        people = np.array(sorted(S.subject.unique()))
        for rep in range(N_REPEATS):
            perm = np.random.default_rng(SEED + rep).permutation(people)
            for test_ids in np.array_split(perm, N_FOLDS):
                te_mask = S.subject.isin(test_ids)
                tr, te = S[~te_mask], S[te_mask]
                for h in HORIZONS:
                    te_h = te.dropna(subset=[f"y{h}"])
                    for m in ["ridge", "lightgbm"]:
                        rows.append(per_person(te_h, fit_predict(m, tr, te_h, h), cohort=cname, job="own",
                                               train_source="own", model=m, horizon=h, repeat=rep))
        print("job2 done", cname, round(time.time() - t0), "s", flush=True)

    res = pd.concat(rows, ignore_index=True)
    res.to_csv(os.path.join(RES, "external_per_person.csv"), index=False)

    # ---- summaries: mean over repeats per person, bootstrap over people; paired gain vs persistence
    rng = np.random.default_rng(SEED)
    per = res.groupby(["cohort", "job", "train_source", "model", "horizon", "subject"])[["rmse", "mae", "bias"]].mean().reset_index()
    summ = []
    for (cohort, job, src, model, h), d in per.groupby(["cohort", "job", "train_source", "model", "horizon"]):
        v = d.rmse.to_numpy()
        b = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        base = per[(per.cohort == cohort) & (per.model == "persistence") & (per.horizon == h)].set_index("subject").rmse
        diff = (base - d.set_index("subject").rmse).dropna().to_numpy()
        bd = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
        summ.append(dict(cohort=cohort, job=job, train_source=src, model=model, horizon=h, n_people=len(v),
                         rmse=v.mean(), rmse_lo=np.percentile(b, 2.5), rmse_hi=np.percentile(b, 97.5),
                         bias=d.bias.mean(), gain_vs_persistence=diff.mean(),
                         gain_lo=np.percentile(bd, 2.5), gain_hi=np.percentile(bd, 97.5),
                         people_better=int((diff > 0).sum())))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(RES, "external_summary.csv"), index=False)
    json.dump(dict(seed=SEED, meta=meta, runtime_sec=round(time.time() - t0, 1)),
              open(os.path.join(RES, "external_config.json"), "w"), indent=2)
    pd.set_option("display.width", 250)
    for h in HORIZONS:
        print(f"\n=== 60-min-style table, horizon {h} min: RMSE (gain vs persistence) ===" if h == 60 else f"\n--- horizon {h}")
        d = summ[summ.horizon == h].copy()
        d["rmse"] = d.rmse.round(1)
        d["gain"] = d.gain_vs_persistence.round(1)
        print(d[["cohort", "job", "train_source", "model", "n_people", "rmse", "gain", "bias"]].round(1).to_string(index=False))
