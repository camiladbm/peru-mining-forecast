"""
Step 8. The two-stage system: rank places INSIDE the 10 km ring around existing mining.

Stage 1 (rule): keep only cells within RING_KM of mining that existed by year t.
Stage 2 (model): models trained only on ring cells rank where new mining appears in t+1.

Same rolling design as rolling_eval.py (feature years 2020..2024, train on years < t).
Two subsets:
  all ring cells          includes cells that already have mining (expansion + new ground)
  never-mined ring cells  cells inside the ring with no mining yet
Baselines: distance to mining; persistence (new patches this year) with distance tie-break.
Models: logistic regression and monotone XGBoost, trained on ring cells only.
Bootstrap over cells gives 95% intervals, P(beats distance), and a paired comparison with
persistence (PR-AUC gain vs persistence with 95% interval, P(beats persistence)).

Run from the repository root:  python src/ring_eval.py [--ring-km 10]
Output: results/ring_results.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import rolling_eval as R

N_BOOT = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring-km", type=float, default=10)
    a = ap.parse_args()

    P = R.add_interactions(pd.read_csv("data/peru_mining_grid_panel_v4.csv"))
    F = R.V4
    subsets = {
        "all ring cells": (lambda d: d.dist_nearest_mining_km <= a.ring_km, 100),
        "never-mined ring cells": (lambda d: (d.dist_nearest_mining_km <= a.ring_km) & (d.frontier == 1), 50),
    }
    rng = np.random.default_rng(0)
    rows = []
    for name, (cond, k) in subsets.items():
        parts = []
        for t in range(2020, 2025):
            tr, te = P[(P.year < t) & cond(P)], P[(P.year == t) & cond(P)].copy()
            te["Distance"] = -te.dist_nearest_mining_km
            te["Persistence + distance"] = (te.new_mining_t + te.new_mining_within_10km_t * 1e-3
                                            - te.dist_nearest_mining_km * 1e-6)
            te["Logistic (ring-trained)"] = R.logit().fit(R.prep(tr, F), tr.label).predict_proba(R.prep(te, F))[:, 1]
            te["XGBoost (ring-trained)"] = (R.mono_xgb(F, tr.label, depth=3).fit(tr[F], tr.label)
                                            .predict_proba(te[F])[:, 1])
            parts.append(te)
        d = pd.concat(parts).reset_index(drop=True)
        models = ["Distance", "Persistence + distance", "Logistic (ring-trained)", "XGBoost (ring-trained)"]

        def score(df, m):
            ap_ = np.mean([average_precision_score(g.label, g[m]) for _, g in df.groupby("year") if g.label.sum()])
            rec = sum(g.loc[g[m].nlargest(k).index, "label"].sum() for _, g in df.groupby("year")) / df.label.sum()
            return ap_, rec

        idx = d.groupby("cell_id").indices
        cells = np.array(list(idx))
        boot = {m: [] for m in models}
        for _ in range(N_BOOT):
            b = d.iloc[np.concatenate([idx[c] for c in rng.choice(cells, len(cells))])]
            for m in models:
                boot[m].append(score(b, m)[0])
        base = np.array(boot["Distance"])
        pers = np.array(boot["Persistence + distance"])
        print(f"\n{name}: {len(d):,} cell-years, {d.label.sum()} positives, base rate {d.label.mean():.3f}")
        for m in models:
            pr_auc, rec = score(d, m)
            bb = np.array(boot[m])
            roc = np.mean([roc_auc_score(g.label, g[m]) for _, g in d.groupby("year")])
            row = {"subset": name, "ring_km": a.ring_km, "model": m, "PR-AUC": pr_auc,
                   "PR-AUC 95% low": np.percentile(bb, 2.5), "PR-AUC 95% high": np.percentile(bb, 97.5),
                   "ROC-AUC": roc, f"recall@{k} per year": rec, "P(beats distance)": (bb > base).mean(),
                   "PR-AUC gain vs persistence": pr_auc - score(d, "Persistence + distance")[0],
                   "gain 95% low": np.percentile(bb - pers, 2.5), "gain 95% high": np.percentile(bb - pers, 97.5),
                   "P(beats persistence)": (bb > pers).mean(),
                   "cell-years": len(d), "positives": int(d.label.sum())}
            rows.append(row)
            print(f"  {m:26s} PR-AUC {pr_auc:.3f} [{row['PR-AUC 95% low']:.3f}-{row['PR-AUC 95% high']:.3f}]"
                  f"  ROC {roc:.3f}  recall@{k} {rec:.2f}  P(beats distance) {row['P(beats distance)']:.2f}"
                  f"  gain vs persistence {row['PR-AUC gain vs persistence']:+.3f}"
                  f" [{row['gain 95% low']:+.3f} to {row['gain 95% high']:+.3f}]")
    pd.DataFrame(rows).to_csv("results/ring_results.csv", index=False)


if __name__ == "__main__":
    main()
