"""
Step 7. Run from the repository root: python src/expansion_eval.py

Do models beat simple rules when the task is HOW MUCH mining expands, not WHETHER?

Targets (rolling folds t = 2020..2024, train on years < t)
  big5 / big10  : 1 if the cell gets >= 5 / >= 10 new mining patches in t+1  (classification)
  volume        : number of new patches in t+1                                  (count regression)
Baselines (no training)
  persistence   : new patches in the cell this year (new_mining_t)
  momentum      : new patches within 10 km this year
  distance      : -distance to nearest mining
  (ties broken by distance)
Models
  XGBoost classifier (monotone) for big5/big10; XGBoost Poisson regressor for volume;
  Poisson GLM (log-features) for volume.
Metrics (computed per year, then averaged)
  classification: PR-AUC, recall in top 100 cells
  volume: capture@k = share of all next-year new patches that fall in the top-k ranked cells (k=100, 500),
          and Spearman correlation over cells with any mining within 20 km
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import rolling_eval as R

FEATS = R.V4
P = R.add_interactions(pd.read_csv("data/peru_mining_grid_panel_v4.csv"))
P["big5"] = (P.new_mining_next_year >= 5).astype(int)
P["big10"] = (P.new_mining_next_year >= 10).astype(int)


def baselines(te):
    tb = -te.dist_nearest_mining_km * 1e-6
    return {"Baseline: persistence": te.new_mining_t + tb,
            "Baseline: momentum (10 km)": te.new_mining_within_10km_t + tb,
            "Baseline: distance": -te.dist_nearest_mining_km}


def cls_model(tr, target):
    m = R.mono_xgb(FEATS, tr[target])
    return m.fit(tr[FEATS], tr[target])


def vol_models(tr):
    cons = tuple(-1 if c in R.NEG else (1 if c in R.POS else 0) for c in FEATS)
    xg = xgb.XGBRegressor(objective="count:poisson", n_estimators=400, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_lambda=5,
                          monotone_constraints=cons, max_delta_step=0.7, random_state=0, n_jobs=4)
    xg.fit(tr[FEATS], tr.new_mining_next_year)
    glm = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=3000))
    glm.fit(R.prep(tr, FEATS), tr.new_mining_next_year)
    return xg, glm


def capture(te, s, k):
    top = te[s].nlargest(k).index
    return te.loc[top, "new_mining_next_year"].sum() / max(te.new_mining_next_year.sum(), 1)


def main():
    rows, preds = [], []
    for t in range(2020, 2025):
        tr, te = P[P.year < t], P[P.year == t].copy()
        base = baselines(te)
        for k, v in base.items():
            te[k] = v
        for target in ["big5", "big10"]:
            te[f"XGBoost ({target})"] = cls_model(tr, target).predict_proba(te[FEATS])[:, 1]
        xg, glm = vol_models(tr)
        te["XGBoost Poisson (volume)"] = xg.predict(te[FEATS])
        te["Poisson GLM (volume)"] = glm.predict(R.prep(te, FEATS))
        preds.append(te)
        near = te.dist_nearest_mining_km <= 20
        for target in ["big5", "big10"]:
            for s in list(base) + [f"XGBoost ({target})", "XGBoost Poisson (volume)"]:
                rows.append({"year": t, "task": target, "model": s,
                             "PR-AUC": average_precision_score(te[target], te[s]),
                             "recall@100": te.loc[te[s].nlargest(100).index, target].sum() / max(te[target].sum(), 1),
                             "positives": int(te[target].sum())})
        for s in list(base) + ["XGBoost Poisson (volume)", "Poisson GLM (volume)", "XGBoost (big5)"]:
            rows.append({"year": t, "task": "volume", "model": s,
                         "capture@100": capture(te, s, 100), "capture@500": capture(te, s, 500),
                         "spearman (near mining)": spearmanr(te.loc[near, s], te.loc[near, "new_mining_next_year"])[0]})
        print(f"fold {t} done")
    res = pd.DataFrame(rows)
    res.to_csv("results/expansion_results_by_year.csv", index=False)
    pd.concat(preds)[["cell_id", "year", "new_mining_next_year", "big5", "big10"] +
                     [c for c in preds[0].columns if c.startswith(("Baseline", "XGBoost", "Poisson"))]
                     ].to_csv("results/expansion_predictions.csv", index=False)
    pd.set_option("display.width", 200)
    summ = res.groupby(["task", "model"]).mean(numeric_only=True).drop(columns="year").round(3)
    print(summ.to_string())
    summ.to_csv("results/expansion_results.csv")


if __name__ == "__main__":
    main()
