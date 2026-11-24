"""D12: one small GRU versus LightGBM. Design fixed before running, see DECISIONS.md.

GRU(64) over the last 8 glucose readings (15-min grid) with time-of-day sin/cos at every step; one head predicts the
30, 60 and 120 minute targets jointly (missing targets masked). Variants: GRU_R1 (history only) and GRU_R3
(+ health group + basic meal features through a small static branch). Adam 1e-3, batch 512, up to 40 epochs, early
stopping (patience 5) on 10% of the TRAINING PEOPLE; test people are never touched. Same grouped folds as the ladder.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cgmacros_loader import DATA
from forecast_baselines import HORIZONS, N_FOLDS, N_LAGS, N_REPEATS, RES, SEED, make_samples
from forecast_ladder import MEAL_COLS, context_features

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATIC = ["grp"] + MEAL_COLS
G0, GS = 120.0, 50.0
MAX_EPOCHS, PATIENCE, BATCH = 40, 5, 512


class Net(nn.Module):
    def __init__(self, static_dim):
        super().__init__()
        self.gru = nn.GRU(3, 64, batch_first=True)
        self.static = nn.Sequential(nn.Linear(static_dim, 16), nn.ReLU()) if static_dim else None
        self.head = nn.Sequential(nn.Linear(64 + (16 if static_dim else 0), 32), nn.ReLU(), nn.Linear(32, len(HORIZONS)))

    def forward(self, seq, stat=None):
        _, h = self.gru(seq)
        z = h[-1]
        if self.static is not None:
            z = torch.cat([z, self.static(stat)], dim=1)
        return self.head(z)


def tensors(S):
    lags = S[[f"lag{k}" for k in range(N_LAGS)]].to_numpy(float)[:, ::-1]          # oldest -> newest
    tod = S["tod"].to_numpy(float)[:, None] - 15.0 * np.arange(N_LAGS - 1, -1, -1)[None, :]
    ang = 2 * np.pi * tod / 1440.0
    seq = np.stack([(lags - G0) / GS, np.sin(ang), np.cos(ang)], axis=2).astype(np.float32)
    y = np.stack([S[f"y{h}"].to_numpy(float) for h in HORIZONS], axis=1)
    return seq, y


def masked_mse(pred, y):
    m = ~torch.isnan(y)
    return ((pred - torch.nan_to_num(y)) ** 2)[m].mean()


def train_predict(seq_tr, st_tr, y_tr, ppl_tr, seq_te, st_te, rng, seed):
    torch.manual_seed(seed)
    ids = np.unique(ppl_tr)
    val_ids = rng.choice(ids, max(1, int(round(0.1 * len(ids)))), replace=False)
    isval = np.isin(ppl_tr, val_ids)
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    ytr = T((y_tr - G0) / GS)
    A, B = np.where(~isval)[0], np.where(isval)[0]
    seq_t, st_t = T(seq_tr), (T(st_tr) if st_tr is not None else None)
    net = Net(st_tr.shape[1] if st_tr is not None else 0).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, best_state, bad = np.inf, None, 0
    for ep in range(MAX_EPOCHS):
        net.train()
        perm = np.random.default_rng(seed + ep).permutation(A)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            loss = masked_mse(net(seq_t[b], st_t[b] if st_t is not None else None), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = masked_mse(net(seq_t[B], st_t[B] if st_t is not None else None), ytr[B]).item()
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: x.clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        out = net(T(seq_te), T(st_te) if st_te is not None else None).cpu().numpy()
    return out * GS + G0, ep + 1


def run(sensor, df):
    S = context_features(df, make_samples(df, sensor)).reset_index(drop=True)
    seq, Y = tensors(S)
    stat = S[STATIC].to_numpy(float)
    people = np.array(sorted(S.subject.unique()))
    print(f"[{sensor}] samples={len(S)} people={len(people)} device={DEV}", flush=True)
    rows, epochs = [], []
    for rep in range(N_REPEATS):
        folds = np.array_split(np.random.default_rng(SEED + rep).permutation(people), N_FOLDS)
        for fi, test_ids in enumerate(folds):
            te = S.subject.isin(test_ids).to_numpy()
            rng = np.random.default_rng(SEED + 700 + rep * 10 + fi)
            mu, sd = np.nanmean(stat[~te], axis=0), np.nanstd(stat[~te], axis=0) + 1e-6
            st = np.nan_to_num((stat - mu) / sd).astype(np.float32)
            for variant, use_static in [("GRU_R1_glucose", False), ("GRU_R3_+meals", True)]:
                pred, ne = train_predict(seq[~te], st[~te] if use_static else None, Y[~te], S.subject.to_numpy()[~te],
                                         seq[te], st[te] if use_static else None, rng, SEED + rep * 100 + fi)
                epochs.append(ne)
                St = S[te]
                for j, h in enumerate(HORIZONS):
                    ok = ~np.isnan(Y[te][:, j])
                    e = pred[ok, j] - Y[te][ok, j]
                    tmp = pd.DataFrame({"subject": St.subject.to_numpy()[ok], "se": e ** 2, "ae": np.abs(e), "e": e})
                    pp = tmp.groupby("subject").agg(mse=("se", "mean"), mae=("ae", "mean"), bias=("e", "mean")).reset_index()
                    pp["rmse"] = np.sqrt(pp.mse)
                    pp["sensor"], pp["rung"], pp["horizon"], pp["repeat"] = sensor, variant, h, rep
                    rows.append(pp.drop(columns="mse"))
            print(f"  [{sensor}] rep {rep} fold {fi} done (epochs {epochs[-2]}/{epochs[-1]})", flush=True)
    return pd.concat(rows, ignore_index=True), epochs


if __name__ == "__main__":
    t0 = time.time()
    df = pd.read_parquet(os.path.join(DATA, "processed", "cgmacros_all.parquet"))
    outs, ep = [], {}
    for s in ["libre", "dexcom"]:
        r, e = run(s, df)
        outs.append(r)
        ep[s] = dict(mean_epochs=float(np.mean(e)), min=int(np.min(e)), max=int(np.max(e)))
    res = pd.concat(outs, ignore_index=True)
    res.to_csv(os.path.join(RES, "gru_per_participant.csv"), index=False)
    lg = pd.read_csv(os.path.join(RES, "ladder_per_participant.csv"))
    lg = lg[lg.rung.isin(["R1_glucose", "R3_+meals"])]
    allr = pd.concat([res[["sensor", "rung", "horizon", "subject", "repeat", "rmse"]],
                      lg[["sensor", "rung", "horizon", "subject", "repeat", "rmse"]]], ignore_index=True)
    per = allr.groupby(["sensor", "rung", "horizon", "subject"]).rmse.mean().reset_index()
    rng = np.random.default_rng(SEED)
    summ = []
    for (sensor, h), d in per.groupby(["sensor", "horizon"]):
        piv = d.pivot(index="subject", columns="rung", values="rmse")
        for rung in piv.columns:
            v = piv[rung]
            b = [rng.choice(v.to_numpy(), len(v)).mean() for _ in range(2000)]
            row = dict(sensor=sensor, horizon=h, rung=rung, n_people=len(v), rmse=v.mean(), rmse_lo=np.percentile(b, 2.5), rmse_hi=np.percentile(b, 97.5))
            for gru, lgb_ in [("GRU_R1_glucose", "R1_glucose"), ("GRU_R3_+meals", "R3_+meals")]:
                if rung == gru:
                    diff = (piv[lgb_] - v).dropna().to_numpy()         # positive = GRU better than LightGBM
                    bd = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
                    row.update(gain_vs_lightgbm=diff.mean(), gain_lo=np.percentile(bd, 2.5), gain_hi=np.percentile(bd, 97.5),
                               people_better=int((diff > 0).sum()), compared_with=lgb_)
            summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(RES, "gru_summary.csv"), index=False)
    json.dump(dict(seed=SEED, device=str(DEV), epochs=ep, runtime_sec=round(time.time() - t0, 1)),
              open(os.path.join(RES, "gru_config.json"), "w"), indent=2)
    pd.set_option("display.width", 250)
    print(summ[["sensor", "horizon", "rung", "rmse", "gain_vs_lightgbm", "gain_lo", "gain_hi", "people_better"]].round(2).to_string(index=False))
    print(json.dumps(dict(epochs=ep, runtime_sec=round(time.time() - t0, 1))))
