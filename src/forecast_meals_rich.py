"""R3r: richer meal features (D9e), added on top of the basic meal rung R3. Design fixed before running.

Extras (10): decayed meal loads carbs(30/90/180 min), protein(120), fat(180), fiber(120), kcal(120);
meals in last 6 h; last meal type; carb share of last meal's calories.
Only meals logged at or before t are used. Macros are capped at the ranges given in the CGMacros data dictionary.
Control: the 10 extras are shuffled across time within each person (new shuffle each repeat).
Gains are reported against R3 (parent) and R1.
"""
import json
import os
import time

import numpy as np
import pandas as pd

from cgmacros_loader import DATA
from forecast_baselines import HORIZONS, N_FOLDS, N_REPEATS, RES, SEED, make_samples
from forecast_ladder import MEAL_COLS, context_features, fit_pred

CAPS = dict(meal_kcal=1180.0, carbs=176.0, protein=176.0, fat=176.0, fiber=176.0)   # from DataDictionary_CGMacros-00X.csv
TYPE = {"b": 1, "l": 2, "d": 3, "s": 4}
EXTRA = ["carbs_d30", "carbs_d90", "carbs_d180", "protein_d120", "fat_d180", "fiber_d120", "kcal_d120",
         "n_meals_6h", "last_meal_type", "last_carb_share"]
RUNGS = {
    "R1_glucose": [],
    "R3_+meals": ["grp"] + MEAL_COLS,
    "R3r_+rich_meals": ["grp"] + MEAL_COLS + EXTRA,
    "R3r_control_shuffled": ["grp"] + MEAL_COLS + EXTRA,
}


def add_rich(df: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    S = S.copy()
    for c in EXTRA:
        S[c] = np.nan
    for sid, d in df.groupby("subject"):
        d = d.sort_values("Timestamp")
        t0 = d.Timestamp.min()
        meals = d[d.meal_type.notna()].copy()
        mm = ((meals.Timestamp - t0).dt.total_seconds() // 60).astype(int).to_numpy()
        o = np.argsort(mm)
        mm = mm[o]
        vals = {k: np.nan_to_num(np.minimum(meals[k].to_numpy(float), CAPS[k]))[o] for k in CAPS}
        mt = meals.meal_type.astype(str).str.strip().str.lower().str[0].map(TYPE).fillna(0).to_numpy()[o]
        idx = S.index[S.subject == sid]
        t = S.loc[idx, "tmin"].to_numpy().astype(float)
        dt = t[:, None] - mm[None, :]                         # minutes since each meal (negative = future)
        past = dt >= 0
        def decayed(key, tau):
            w = np.where(past, np.exp(-np.clip(dt, 0, None) / tau), 0.0)
            return w @ vals[key] if len(mm) else np.zeros(len(t))
        for name, key, tau in [("carbs_d30", "carbs", 30), ("carbs_d90", "carbs", 90), ("carbs_d180", "carbs", 180),
                               ("protein_d120", "protein", 120), ("fat_d180", "fat", 180),
                               ("fiber_d120", "fiber", 120), ("kcal_d120", "meal_kcal", 120)]:
            S.loc[idx, name] = decayed(key, tau)
        if len(mm):
            S.loc[idx, "n_meals_6h"] = (past & (dt < 360)).sum(axis=1)
            last = np.where(past.any(axis=1), (np.where(past, mm[None, :], -1)).argmax(axis=1), -1)
            # minutes since the last meal, via dt at the last index
            since = np.array([dt[i, last[i]] if last[i] >= 0 else np.inf for i in range(len(t))])
            within = since < 360
            S.loc[idx, "last_meal_type"] = np.where(within, [mt[l] if l >= 0 else 0 for l in last], 0)
            kc = np.array([vals["meal_kcal"][l] if l >= 0 else np.nan for l in last])
            cb = np.array([vals["carbs"][l] if l >= 0 else np.nan for l in last])
            share = np.where((kc > 0) & within, np.clip(cb * 4.0 / np.where(kc > 0, kc, 1), 0, 1.5), np.nan)
            S.loc[idx, "last_carb_share"] = share
        else:
            S.loc[idx, ["n_meals_6h", "last_meal_type"]] = 0
    return S


def shuffle_extras(S, seed):
    S = S.copy()
    rng = np.random.default_rng(seed)
    for sid, ix in S.groupby("subject").groups.items():
        ix = np.array(list(ix))
        S.loc[ix, EXTRA] = S.loc[rng.permutation(ix), EXTRA].to_numpy()
    return S


def run(sensor, df):
    S = add_rich(df, context_features(df, make_samples(df, sensor)).reset_index(drop=True))
    people = np.array(sorted(S.subject.unique()))
    print(f"[{sensor}] samples={len(S)} people={len(people)} extras NaN share="
          f"{S[EXTRA].isna().mean().round(3).to_dict()}", flush=True)
    rows = []
    for rep in range(N_REPEATS):
        Sc = shuffle_extras(S, SEED + 200 + rep)
        folds = np.array_split(np.random.default_rng(SEED + rep).permutation(people), N_FOLDS)
        for test_ids in folds:
            te_mask = S.subject.isin(test_ids).to_numpy()
            for h in HORIZONS:
                ok = S[f"y{h}"].notna().to_numpy()
                for rung, cols in RUNGS.items():
                    SS = Sc if "control" in rung else S
                    p = fit_pred(SS[~te_mask], SS[te_mask & ok], h, cols)
                    te = SS[te_mask & ok]
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
    outs, sample_stats = [], {}
    for s in ["libre", "dexcom"]:
        r, S = run(s, df)
        outs.append(r)
        sample_stats[s] = dict(samples=len(S), mean_decayed_carbs_90=float(S.carbs_d90.mean()),
                               share_with_meal_in_6h=float((S.n_meals_6h > 0).mean()))
    res = pd.concat(outs, ignore_index=True)
    res.to_csv(os.path.join(RES, "rich_meals_per_participant.csv"), index=False)
    per = res.groupby(["sensor", "rung", "horizon", "subject"]).rmse.mean().reset_index()
    rng = np.random.default_rng(SEED)
    summ = []
    for (sensor, h), d in per.groupby(["sensor", "horizon"]):
        piv = d.pivot(index="subject", columns="rung", values="rmse")
        for rung in RUNGS:
            v = piv[rung]
            b = [rng.choice(v.to_numpy(), len(v)).mean() for _ in range(2000)]
            row = dict(sensor=sensor, horizon=h, rung=rung, n_people=len(v), rmse=v.mean(),
                       rmse_lo=np.percentile(b, 2.5), rmse_hi=np.percentile(b, 97.5))
            for nm, ref in [("R1", "R1_glucose"), ("R3", "R3_+meals")]:
                if rung == ref:
                    continue
                diff = (piv[ref] - v).dropna().to_numpy()
                bd = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
                row.update({f"gain_vs_{nm}": diff.mean(), f"gain_vs_{nm}_lo": np.percentile(bd, 2.5),
                            f"gain_vs_{nm}_hi": np.percentile(bd, 97.5), f"better_vs_{nm}": int((diff > 0).sum())})
            summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(RES, "rich_meals_summary.csv"), index=False)
    # paired: real rich vs shuffled control
    chk = []
    for (sensor, h), d in per.groupby(["sensor", "horizon"]):
        piv = d.pivot(index="subject", columns="rung", values="rmse")
        diff = (piv["R3r_control_shuffled"] - piv["R3r_+rich_meals"]).dropna().to_numpy()
        bd = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
        chk.append(dict(sensor=sensor, horizon=h, real_better_than_control=diff.mean(), lo=np.percentile(bd, 2.5),
                        hi=np.percentile(bd, 97.5), people_better=int((diff > 0).sum())))
    pd.DataFrame(chk).to_csv(os.path.join(RES, "rich_meals_real_vs_control.csv"), index=False)
    json.dump(dict(seed=SEED, extras=EXTRA, caps=CAPS, sample_stats=sample_stats, runtime_sec=round(time.time() - t0, 1)),
              open(os.path.join(RES, "rich_meals_config.json"), "w"), indent=2)
    pd.set_option("display.width", 250)
    print(summ[["sensor", "horizon", "rung", "rmse", "gain_vs_R1", "gain_vs_R3", "gain_vs_R3_lo", "gain_vs_R3_hi", "better_vs_R3"]].round(2).to_string(index=False))
    print(pd.DataFrame(chk).round(2).to_string(index=False))
