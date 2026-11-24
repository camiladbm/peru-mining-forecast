"""
Rolling-origin (expanding window) evaluation on peru_mining_grid_panel_v2.csv.

For each feature year t = 2020..2024: train on all rows with year < t, predict rows with year == t
(label = new mining in t+1). Predictions from all folds are pooled, which gives many more
frontier test cases than a single split. Uncertainty comes from a bootstrap over test cells
(resampling cells, keeping each fold's ranking), and every model is compared with the
distance-only baseline on the same bootstrap draws.

Outputs
  rolling_results.csv          pooled metrics with 95% intervals (all cells, frontier cells)
  rolling_results_by_year.csv  per-fold metrics
Usage: python rolling_eval.py data/peru_mining_grid_panel_v2.csv
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
GEO = ["dist_river_km", "dist_river_10cms_km", "dist_river_100cms_km", "in_protected_area",
       "in_buffer_zone", "dist_protected_km", "poverty_rate", "pop_density"]
LOG_COLS = ["new_mining_t", "new_mining_within_10km_t", "mined_within_10km", "dist_nearest_mining_km",
            "dist_river_km", "dist_river_10cms_km", "dist_river_100cms_km", "dist_protected_km", "pop_density"]
K = {"all cells": 500, "frontier cells": 200}   # cells inspected per year
N_BOOT = 200
RNG = np.random.default_rng(0)


def prep(df, cols):
    X = df[cols].astype(float).copy()
    for c in set(LOG_COLS) & set(cols):
        X[c] = np.log1p(X[c])
    return X


def logit(cols):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced")), cols


def booster(cols, y):
    spw = (y == 0).sum() / max(y.sum(), 1)
    return xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=5, scale_pos_weight=spw,
                             random_state=0, n_jobs=4), cols


def recall_at_k(df, score, k):
    """Share of positives captured when inspecting the top-k cells in EACH year."""
    hit = 0
    for _, g in df.groupby("year"):
        top = g[score].nlargest(k).index
        hit += g.loc[top, "label"].sum()
    return hit / max(df.label.sum(), 1)


def main(path):
    p = pd.read_csv(path)
    preds = []
    for t in range(2020, 2025):
        tr, te = p[p.year < t], p[p.year == t].copy()
        te["Baseline: distance only"] = -te.dist_nearest_mining_km
        te["Baseline: rule"] = (((te.new_mining_t > 0) | (te.new_mining_within_10km_t > 0)).astype(float)
                                - te.dist_nearest_mining_km * 1e-6)
        specs = {"Logistic (history)": logit(HISTORY), "Logistic (history + geography)": logit(HISTORY + GEO),
                 "XGBoost (history)": booster(HISTORY, tr.label), "XGBoost (history + geography)": booster(HISTORY + GEO, tr.label)}
        for name, (model, cols) in specs.items():
            Xtr = prep(tr, cols) if name.startswith("Logistic") else tr[cols]
            Xte = prep(te, cols) if name.startswith("Logistic") else te[cols]
            model.fit(Xtr, tr.label)
            te[name] = model.predict_proba(Xte)[:, 1]
        preds.append(te)
        print(f"fold t={t}: trained on {len(tr):,} rows, test positives {te.label.sum()} "
              f"(frontier {te.loc[te.frontier == 1, 'label'].sum()})")
    pr = pd.concat(preds)
    models = [c for c in pr.columns if c.startswith(("Baseline", "Logistic", "XGBoost"))]
    pr[["cell_id", "year", "label", "frontier", "label_is_provisional"] + models].to_csv(
        "results/experiments_rolling_predictions_v2.csv", index=False)

    rows, by_year = [], []
    for subset, mask in [("all cells", np.ones(len(pr), bool)), ("frontier cells", pr.frontier.values == 1)]:
        d = pr[mask].reset_index(drop=True)
        k = K[subset]
        # pooled point estimates (PR-AUC computed within each year, then averaged = fair across folds)
        def scores(df):
            out = {}
            for m in models:
                aps = [average_precision_score(g.label, g[m]) for _, g in df.groupby("year") if g.label.sum() > 0]
                out[m] = (np.mean(aps), recall_at_k(df, m, k))
            return out
        point = scores(d)
        # bootstrap over cells
        cells = d.cell_id.unique()
        idx_by_cell = d.groupby("cell_id").indices
        boot = {m: [] for m in models}
        for _ in range(N_BOOT):
            pick = RNG.choice(cells, size=len(cells), replace=True)
            ii = np.concatenate([idx_by_cell[c] for c in pick])
            bd = d.iloc[ii].reset_index(drop=True)
            s = scores(bd)
            for m in models:
                boot[m].append(s[m])
        base = np.array(boot["Baseline: distance only"])
        for m in models:
            b = np.array(boot[m])
            diff = b[:, 0] - base[:, 0]
            rows.append({"subset": subset, "model": m,
                         "PR-AUC": point[m][0], "PR-AUC 95% low": np.percentile(b[:, 0], 2.5),
                         "PR-AUC 95% high": np.percentile(b[:, 0], 97.5),
                         f"recall@k per year": point[m][1],
                         "recall 95% low": np.percentile(b[:, 1], 2.5), "recall 95% high": np.percentile(b[:, 1], 97.5),
                         "PR-AUC gain vs distance": point[m][0] - point["Baseline: distance only"][0],
                         "P(gain > 0)": (diff > 0).mean(), "k": k, "positives": int(d.label.sum())})
        for (yr), g in d.groupby("year"):
            for m in models:
                by_year.append({"subset": subset, "year": yr, "model": m,
                                "PR-AUC": average_precision_score(g.label, g[m]) if g.label.sum() else np.nan,
                                "recall@k": recall_at_k(g, m, k), "positives": int(g.label.sum())})

    res = pd.DataFrame(rows)
    res.to_csv("results/experiments_rolling_results.csv", index=False)
    pd.DataFrame(by_year).to_csv("results/experiments_rolling_results_by_year.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/peru_mining_grid_panel_v2.csv")
