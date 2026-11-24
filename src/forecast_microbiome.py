"""R5: microbiome rung (D9d). Six pre-specified person-level summary features.

Person-level features are constant for a person, so a flexible model can use them as an identity tag in training.
Grouped CV (test people never in training) guards against that. The control hands each person ANOTHER person's
microbiome values (new random assignment each repeat); a real effect must vanish under the control.
Gains are measured against the rung each variant extends (R3 for R5m, R4 for R5) and against R1.
"""
import json
import os
import time

import numpy as np
import pandas as pd

from cgmacros_loader import DATA, ROOT
from forecast_baselines import HORIZONS, N_FOLDS, N_REPEATS, RES, SEED, make_samples
from forecast_ladder import ACT_COLS, MEAL_COLS, context_features, fit_pred

MICRO_SCORES = ["Active Microbial Diversity", "Metabolic Fitness", "Gut Microbiome Health", "Inflammatory Activity",
                "Butyrate Production Pathways"]
MICRO_COLS = ["m_richness", "m_diversity", "m_metabolic", "m_health", "m_inflam", "m_butyrate"]
RUNGS = {
    "R1_glucose": [],
    "R3_+meals": ["grp"] + MEAL_COLS,
    "R5m_meals+micro": ["grp"] + MEAL_COLS + MICRO_COLS,
    "R5m_control_other_person": ["grp"] + MEAL_COLS + MICRO_COLS,
    "R4_+activity": ["grp"] + MEAL_COLS + ACT_COLS,
    "R5_act+micro": ["grp"] + MEAL_COLS + ACT_COLS + MICRO_COLS,
    "R5_control_other_person": ["grp"] + MEAL_COLS + ACT_COLS + MICRO_COLS,
}
PARENT = {"R5m_meals+micro": "R3_+meals", "R5m_control_other_person": "R3_+meals",
          "R5_act+micro": "R4_+activity", "R5_control_other_person": "R4_+activity"}


def micro_table() -> pd.DataFrame:
    m = pd.read_csv(os.path.join(ROOT, "microbes.csv")).set_index("subject")
    g = pd.read_csv(os.path.join(ROOT, "gut_health_test.csv")).set_index("subject")
    out = pd.DataFrame(index=m.index)
    rich = (m > 0).sum(axis=1).astype(float)
    out["m_richness"] = rich.where(rich > 0)            # all-zero row = no data, not "zero bacteria"
    for col, name in zip(["m_diversity", "m_metabolic", "m_health", "m_inflam", "m_butyrate"], MICRO_SCORES):
        out[col] = g[name].reindex(out.index)
    return out[MICRO_COLS]


def run(sensor, df, micro):
    S = context_features(df, make_samples(df, sensor)).reset_index(drop=True)
    for c in MICRO_COLS:
        S[c] = S.subject.map(micro[c])
    people = np.array(sorted(S.subject.unique()))
    print(f"[{sensor}] samples={len(S)} people={len(people)} people with taxa data="
          f"{int(micro.loc[people, 'm_richness'].notna().sum())} with scores={int(micro.loc[people, 'm_diversity'].notna().sum())}",
          flush=True)
    rows = []
    for rep in range(N_REPEATS):
        rng = np.random.default_rng(SEED + 100 + rep)
        perm = rng.permutation(people)
        other = dict(zip(people, perm))                     # person -> whose microbiome they receive
        Sc = S.copy()
        for c in MICRO_COLS:
            Sc[c] = S.subject.map(lambda s: micro[c].get(other[s], np.nan))
        folds = np.array_split(np.random.default_rng(SEED + rep).permutation(people), N_FOLDS)
        for test_ids in folds:
            te_mask = S.subject.isin(test_ids).to_numpy()
            for h in HORIZONS:
                ok = S[f"y{h}"].notna().to_numpy()
                for rung, cols in RUNGS.items():
                    SS = Sc if "control" in rung else S
                    tr, te = SS[~te_mask], SS[te_mask & ok]
                    p = fit_pred(tr, te, h, cols)
                    e = p - te[f"y{h}"].to_numpy()
                    tmp = pd.DataFrame({"subject": te.subject.to_numpy(), "se": e ** 2, "ae": np.abs(e), "e": e})
                    pp = tmp.groupby("subject").agg(mse=("se", "mean"), mae=("ae", "mean"), bias=("e", "mean")).reset_index()
                    pp["rmse"] = np.sqrt(pp.mse)
                    pp["sensor"], pp["rung"], pp["horizon"], pp["repeat"] = sensor, rung, h, rep
                    rows.append(pp.drop(columns="mse"))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    t0 = time.time()
    df = pd.read_parquet(os.path.join(DATA, "processed", "cgmacros_all.parquet"))
    micro = micro_table()
    res = pd.concat([run(s, df, micro) for s in ["libre", "dexcom"]], ignore_index=True)
    res.to_csv(os.path.join(RES, "microbiome_per_participant.csv"), index=False)
    per = res.groupby(["sensor", "rung", "horizon", "subject"])[["rmse"]].mean().reset_index()
    rng = np.random.default_rng(SEED)
    summ = []
    for (sensor, h), d in per.groupby(["sensor", "horizon"]):
        piv = d.pivot(index="subject", columns="rung", values="rmse")
        for rung in RUNGS:
            v = piv[rung]
            b = [rng.choice(v.to_numpy(), len(v)).mean() for _ in range(2000)]
            row = dict(sensor=sensor, horizon=h, rung=rung, n_people=len(v), rmse=v.mean(),
                       rmse_lo=np.percentile(b, 2.5), rmse_hi=np.percentile(b, 97.5))
            for ref_name, ref in [("R1", "R1_glucose"), ("parent", PARENT.get(rung))]:
                if ref is None or rung == ref:
                    continue
                diff = (piv[ref] - v).dropna().to_numpy()
                bd = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
                row[f"gain_vs_{ref_name}"] = diff.mean()
                row[f"gain_vs_{ref_name}_lo"] = np.percentile(bd, 2.5)
                row[f"gain_vs_{ref_name}_hi"] = np.percentile(bd, 97.5)
                row[f"people_better_vs_{ref_name}"] = int((diff > 0).sum())
            summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(RES, "microbiome_summary.csv"), index=False)
    json.dump(dict(seed=SEED, micro_cols=MICRO_COLS, scores=MICRO_SCORES, runtime_sec=round(time.time() - t0, 1),
                   people_with_taxa=int(micro.m_richness.notna().sum()), people_with_scores=int(micro.m_diversity.notna().sum())),
              open(os.path.join(RES, "microbiome_config.json"), "w"), indent=2)
    pd.set_option("display.width", 250)
    cols = ["sensor", "horizon", "rung", "rmse", "gain_vs_R1", "gain_vs_parent", "gain_vs_parent_lo", "gain_vs_parent_hi"]
    print(summ[cols].round(2).to_string(index=False))
