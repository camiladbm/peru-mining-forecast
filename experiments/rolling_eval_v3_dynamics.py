"""
Rolling-origin evaluation of the improved models on data/peru_mining_grid_panel_v3.csv.

Folds: for t = 2020..2024, train on year < t, test on year == t (label = new mining in t+1).
Models
  Baselines           distance to nearest mining; distance along rivers to mining
  Logistic v1         past-mining features only (previous best overall)
  Logistic v3         + dynamics (recent mining, rings, lags) + river network + geography + interactions
  XGBoost v3 (mono)   same features, monotone constraints (risk falls with distance, rises with nearby mining)
  Frontier models     logistic v3 and XGBoost v3 trained ONLY on frontier rows (scored on frontier cells)
Metrics: per-year PR-AUC averaged over folds, recall in the top-k cells per year,
bootstrap (over cells) 95% intervals and P(better than distance baseline).
"""
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HISTORY = ["mining_present", "new_mining_t", "new_mining_within_10km_t", "mined_within_10km",
           "dist_nearest_mining_km"]
DYNAMICS = ["dist_recent_mining_km", "mined_0_5km", "mined_5_10km", "mined_10_20km",
            "new_within_10km_lag1", "growth_20km_2yr"]
RIVER_NET = ["river_net_dist_mining_km", "dist_downstream_of_mining_km", "dist_upstream_of_mining_km",
             "river_net_dist_recent_km", "cell_to_reach_km"]
GEO = ["dist_river_km", "dist_river_10cms_km", "dist_river_100cms_km", "in_protected_area",
       "in_buffer_zone", "dist_protected_km", "poverty_rate", "pop_density"]
INTER = ["near_mining_x_river", "near_mining_x_gold"]
V3 = HISTORY + DYNAMICS + RIVER_NET + GEO + INTER

NEG = {"dist_nearest_mining_km", "dist_recent_mining_km", "river_net_dist_mining_km",
       "dist_downstream_of_mining_km", "dist_upstream_of_mining_km", "river_net_dist_recent_km",
       "dist_river_10cms_km"}
POS = {"mining_present", "new_mining_t", "new_mining_within_10km_t", "mined_within_10km", "mined_0_5km",
       "mined_5_10km", "mined_10_20km", "new_within_10km_lag1", "near_mining_x_river"}
K = {"all cells": 500, "frontier cells": 200}
N_BOOT = 200
RNG = np.random.default_rng(0)


def add_interactions(p):
    near = np.exp(-p.dist_nearest_mining_km / 10)                  # 1 at mining, ~0.37 at 10 km
    p["near_mining_x_river"] = near * np.exp(-p.dist_river_10cms_km / 5)
    p["near_mining_x_gold"] = near * p.gold_price / 1000
    return p


def prep(df, cols):
    X = df[cols].astype(float).copy()
    for c in cols:
        if c.endswith("_km") or c.startswith(("mined", "new_", "growth")) or c == "pop_density":
            X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))
    return X


def logit():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=4000, C=0.5, class_weight="balanced"))


def mono_xgb(cols, y, depth=3):
    cons = tuple(-1 if c in NEG else (1 if c in POS else 0) for c in cols)
    return xgb.XGBClassifier(n_estimators=300, max_depth=depth, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=10, reg_lambda=5,
                             scale_pos_weight=(y == 0).sum() / max(y.sum(), 1),
                             monotone_constraints=cons, random_state=0, n_jobs=4)


def recall_at_k(df, s, k):
    hit = sum(g.loc[g[s].nlargest(k).index, "label"].sum() for _, g in df.groupby("year"))
    return hit / max(df.label.sum(), 1)


def mean_ap(df, s):
    return np.mean([average_precision_score(g.label, g[s]) for _, g in df.groupby("year") if g.label.sum()])


def main(path):
    p = add_interactions(pd.read_csv(path))
    preds = []
    for t in range(2020, 2025):
        tr, te = p[p.year < t], p[p.year == t].copy()
        trf = tr[tr.frontier == 1]
        te["Baseline: distance"] = -te.dist_nearest_mining_km
        te["Baseline: river distance"] = -te.river_net_dist_mining_km - te.dist_nearest_mining_km * 1e-3
        m = logit().fit(prep(tr, HISTORY), tr.label)
        te["Logistic v1 (past mining)"] = m.predict_proba(prep(te, HISTORY))[:, 1]
        m = logit().fit(prep(tr, V3), tr.label)
        te["Logistic v3"] = m.predict_proba(prep(te, V3))[:, 1]
        m = mono_xgb(V3, tr.label).fit(tr[V3], tr.label)
        te["XGBoost v3 (monotone)"] = m.predict_proba(te[V3])[:, 1]
        m = logit().fit(prep(trf, V3), trf.label)
        te["Frontier logistic v3"] = m.predict_proba(prep(te, V3))[:, 1]
        m = mono_xgb(V3, trf.label, depth=2).fit(trf[V3], trf.label)
        te["Frontier XGBoost v3"] = m.predict_proba(te[V3])[:, 1]
        preds.append(te)
        print(f"fold t={t} done (train frontier positives: {trf.label.sum()})")
    pr = pd.concat(preds)
    models = [c for c in pr.columns if c.startswith(("Baseline", "Logistic", "XGBoost", "Frontier"))]
    pr[["cell_id", "year", "label", "frontier"] + models].to_csv("results/experiments_rolling_predictions_v3.csv", index=False)

    rows = []
    for subset, mask in [("all cells", np.ones(len(pr), bool)), ("frontier cells", pr.frontier.values == 1)]:
        d = pr[mask].reset_index(drop=True)
        ms = [m for m in models if subset == "frontier cells" or not m.startswith("Frontier")]
        k = K[subset]
        point = {m: (mean_ap(d, m), recall_at_k(d, m, k)) for m in ms}
        idx = d.groupby("cell_id").indices
        cells = np.array(list(idx))
        boot = {m: [] for m in ms}
        for _ in range(N_BOOT):
            ii = np.concatenate([idx[c] for c in RNG.choice(cells, len(cells))])
            bd = d.iloc[ii].reset_index(drop=True)
            for m in ms:
                boot[m].append((mean_ap(bd, m), recall_at_k(bd, m, k)))
        base = np.array(boot["Baseline: distance"])
        for m in ms:
            b = np.array(boot[m])
            rows.append({"subset": subset, "model": m, "PR-AUC": point[m][0],
                         "PR-AUC 95% low": np.percentile(b[:, 0], 2.5), "PR-AUC 95% high": np.percentile(b[:, 0], 97.5),
                         "recall@k": point[m][1], "recall 95% low": np.percentile(b[:, 1], 2.5),
                         "recall 95% high": np.percentile(b[:, 1], 97.5),
                         "P(PR-AUC > distance)": (b[:, 0] > base[:, 0]).mean(),
                         "P(recall > distance)": (b[:, 1] > base[:, 1]).mean(), "k": k})
    res = pd.DataFrame(rows)
    res.to_csv("results/experiments_rolling_results_v3.csv", index=False)
    pd.set_option("display.width", 230, "display.max_columns", 20)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/peru_mining_grid_panel_v3.csv")
