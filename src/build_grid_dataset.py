"""
Build the 5 km grid-cell x year dataset for "Predicting new illegal gold mining in Peru".

One row = one 5 km x 5 km cell in Peru (inside the Amazon Mining Watch area) in year t.
Features use detections with onset <= t; the label says whether new mining was first
detected in the cell in year t+1. Each cell carries its district (ubigeo) so district-level
data (protected areas, poverty, REINFO...) can be joined later.

Step 2. Run from the repository root:  python src/build_grid_dataset.py
Output: data/peru_mining_grid_panel.csv
"""
import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

CRS_M = "ESRI:102033"
CELL_M = 5000
FIRST_T, LAST_T = 2019, 2024
PROVISIONAL_YEAR = 2025
BLOCK_KM = 50


def parse_year(v):
    s = str(v).strip().upper()
    if s.startswith("Q") and len(s) == 4:
        return 2000 + int(s[2:])
    return int(s[:4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amw", default="data/raw/amw.parquet")
    ap.add_argument("--districts", default="data/raw/peru_districts.geojson")
    ap.add_argument("--gold", default="data/raw/gold_monthly.csv")
    ap.add_argument("--coverage", default="data/raw/Amazon_ACA.geojson")
    ap.add_argument("--out", default="data/peru_mining_grid_panel.csv")
    a = ap.parse_args()

    # Detections (all of them, incl. neighbors' countries, for distance features)
    det = gpd.read_parquet(a.amw) if a.amw.endswith(".parquet") else gpd.read_file(a.amw)
    det["year"] = det["onset"].map(parse_year)
    det = det[(det.status == "confirmed") | (det.year == PROVISIONAL_YEAR)].to_crs(CRS_M)
    pts = det.geometry.representative_point()
    dxy, dyr = np.column_stack([pts.x, pts.y]), det.year.values.astype(int)

    # Study area: Peru x AMW coverage
    dist = gpd.read_file(a.districts).to_crs(CRS_M)
    dist = dist.rename(columns={"IDDIST": "ubigeo", "NOMBDIST": "district", "NOMBDEP": "department"})
    cov = gpd.read_file(a.coverage).to_crs(CRS_M).union_all()
    area = dist.union_all().intersection(cov)

    # Grid of cell centers inside the study area
    x0, y0, x1, y1 = area.bounds
    xs = np.arange(x0, x1, CELL_M) + CELL_M / 2
    ys = np.arange(y0, y1, CELL_M) + CELL_M / 2
    X, Y = np.meshgrid(xs, ys)
    cen = gpd.GeoDataFrame(geometry=gpd.points_from_xy(X.ravel(), Y.ravel()), crs=CRS_M)
    cen = cen[cen.within(area)].reset_index(drop=True)
    cen = gpd.sjoin(cen, dist[["ubigeo", "district", "department", "geometry"]],
                    predicate="within", how="left").drop(columns="index_right")
    cen = cen.drop_duplicates(subset="geometry").reset_index(drop=True)
    cen["cell_id"] = np.arange(len(cen))
    cxy = np.column_stack([cen.geometry.x, cen.geometry.y])
    ll = cen.to_crs("EPSG:4326")
    cen["lon"], cen["lat"] = ll.geometry.x.round(4), ll.geometry.y.round(4)
    b = BLOCK_KM * 1000
    cen["block_id"] = ((cxy[:, 0] // b).astype(int).astype(str) + "_"
                       + (cxy[:, 1] // b).astype(int).astype(str))
    print(f"cells: {len(cen):,}")

    # Count detections per cell and onset year (assign each detection to nearest center)
    tree_c = cKDTree(cxy)
    dd, idx = tree_c.query(dxy, k=1, distance_upper_bound=CELL_M * 0.75)
    inside = np.isfinite(dd)
    cnt = (pd.DataFrame({"cell_id": idx[inside], "year": dyr[inside]})
           .groupby(["cell_id", "year"]).size().unstack(fill_value=0)
           .reindex(index=cen.cell_id, columns=range(2018, PROVISIONAL_YEAR + 1), fill_value=0))
    cum = cnt.cumsum(axis=1)

    g = pd.read_csv(a.gold)
    gold = g.assign(year=g.Date.str[:4].astype(int)).groupby("year").Price.mean()

    base = cen[["cell_id", "ubigeo", "district", "department", "lon", "lat", "block_id"]]
    rows = []
    for t in range(FIRST_T, LAST_T + 1):
        upto, new = dyr <= t, dyr == t
        tree_all, tree_new = cKDTree(dxy[upto]), cKDTree(dxy[new])
        df = base.copy()
        df["year"] = t
        df["mining_present"] = (cum[t].values > 0).astype(int)
        df["new_mining_t"] = cnt[t].values
        df["new_mining_within_10km_t"] = tree_new.query_ball_point(cxy, r=10_000, return_length=True)
        df["mined_within_10km"] = tree_all.query_ball_point(cxy, r=10_000, return_length=True)
        df["dist_nearest_mining_km"] = (tree_all.query(cxy, k=1)[0] / 1000).round(2)
        df["gold_price"] = round(gold[t], 1)
        df["gold_x_mining_within_10km"] = df.gold_price * (df.mined_within_10km > 0)
        df["new_mining_next_year"] = cnt[t + 1].values
        df["label"] = (df.new_mining_next_year > 0).astype(int)
        df["frontier"] = (cum[t].values == 0).astype(int)  # cell never mined before
        df["label_is_provisional"] = int(t + 1 == PROVISIONAL_YEAR)
        rows.append(df)

    panel = pd.concat(rows, ignore_index=True)
    panel.to_csv(a.out, index=False)
    print(f"saved {a.out}: {len(panel):,} rows")
    print(panel.groupby("year").agg(positives=("label", "sum"), share=("label", "mean"),
                                    frontier_positives=("label", lambda s: s[panel.loc[s.index, 'frontier'] == 1].sum())))


if __name__ == "__main__":
    main()
