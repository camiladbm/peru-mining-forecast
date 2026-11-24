"""
Train and evaluate models on peru_mining_grid_panel.csv.

Design
  - Time split: train on feature years 2019-2022 (labels 2020-2023),
    test on 2023 (label 2024, confirmed) and 2024 (label 2025, provisional).
  - Baselines: distance to nearest mining (alone) and a simple rule.
  - Models: logistic regression (log-transformed, standardized features) and XGBoost.
  - Metrics on all cells and on frontier cells (no mining yet): ROC-AUC, PR-AUC,
    recall and precision in the top 500 cells (about 1.3% of the area, a patrol budget).

Usage: python train_models.py data/peru_mining_grid_panel.csv
"""
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

TOP_K = 500
HISTORY = ["mining_present", "new_mining_t", "new_mining_within_10km_t", "mined_within_10km",
           "dist_nearest_mining_km"]
GEO = ["dist_river_km", "dist_river_10cms_km", "dist_river_100cms_km", "in_protected_area",
       "in_buffer_zone", "dist_protected_km", "poverty_rate", "pop_density"]
FEATURES = HISTORY + GEO  # gold price dropped: it only varies by year and added nothing in v1
LOG_COLS = ["new_mining_t", "new_mining_within_10km_t", "mined_within_10km", "dist_nearest_mining_km",
            "dist_river_km", "dist_river_10cms_km", "dist_river_100cms_km", "dist_protected_km", "pop_density"]


def logit_X(df):
    X = df[FEATURES].copy().astype(float)
    for c in LOG_COLS:
        X[c] = np.log1p(X[c])
    return X


def metrics(y, s, k=TOP_K):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    top = np.argsort(-s, kind="stable")[:k]
    return {"ROC-AUC": roc_auc_score(y, s), "PR-AUC": average_precision_score(y, s),
            f"recall@{k}": y[top].sum() / max(y.sum(), 1), f"precision@{k}": y[top].mean(),
            "positives": int(y.sum()), "base_rate": y.mean()}


def main(path):
    p = pd.read_csv(path)
    train = p[p.year <= 2022]
    tests = {"2024 (confirmed)": p[p.year == 2023], "2025 (provisional)": p[p.year == 2024]}

    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    lr.fit(logit_X(train), train.label)

    spw = (train.label == 0).sum() / train.label.sum()
    xg = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="aucpr",
                           random_state=0, n_jobs=4)
    xg.fit(train[FEATURES], train.label)
    from sklearn.base import clone
    lr_h = clone(lr).fit(logit_X(train)[HISTORY], train.label)
    xg_h = clone(xg).fit(train[HISTORY], train.label)

    rows = []
    for tname, te in tests.items():
        te_hist = te
        scores = {
            "Baseline: distance only": -te.dist_nearest_mining_km.values,
            "Baseline: rule (new mining in cell or within 10 km)":
                ((te.new_mining_t > 0) | (te.new_mining_within_10km_t > 0)).astype(float).values
                - te.dist_nearest_mining_km.values * 1e-6,  # tie-break by distance
            "Logistic regression": lr.predict_proba(logit_X(te))[:, 1],
            "Logistic regression (history only)": lr_h.predict_proba(logit_X(te)[HISTORY])[:, 1],
            "XGBoost (history only)": xg_h.predict_proba(te[HISTORY])[:, 1],
            "XGBoost": xg.predict_proba(te[FEATURES])[:, 1],
        }
        for subset, mask in [("all cells", np.ones(len(te), bool)), ("frontier cells", te.frontier.values == 1)]:
            for m, s in scores.items():
                k = TOP_K if subset == "all cells" else 200
                r = metrics(te.label.values[mask], s[mask], k)
                r = {"test year": tname, "subset": subset, "model": m,
                     **{kk.replace(str(k), "k"): v for kk, v in r.items()}, "k": k}
                rows.append(r)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(res.round(3).to_string(index=False))
    res.to_csv("results/experiments_model_results_v2.csv", index=False)

    # Interpretation
    coefs = pd.Series(lr[-1].coef_[0], index=FEATURES).sort_values()
    print("\nLogistic regression standardized coefficients:\n", coefs.round(3))
    te = tests["2024 (confirmed)"]
    contrib = xg.get_booster().predict(xgb.DMatrix(te[FEATURES]), pred_contribs=True)[:, :-1]
    shap_imp = pd.Series(np.abs(contrib).mean(0), index=FEATURES).sort_values(ascending=False)
    print("\nXGBoost mean |SHAP| (2024 test):\n", shap_imp.round(3))
    coefs.to_csv("results/experiments_logit_coefficients_v2.csv")
    shap_imp.to_csv("results/experiments_xgb_shap_importance_v2.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/peru_mining_grid_panel_v2.csv")
