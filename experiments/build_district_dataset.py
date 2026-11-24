"""
Build the district x year dataset for "Predicting new illegal gold mining in Peru".

One row = one district in year t. Features use information up to the end of year t;
the label says whether new mining was first detected in the district in year t+1.

Required inputs
  --amw        amazon_basin_detections.geojson   (Amazon Mining Watch, data.source.coop)
  --districts  peru_districts.geojson            (district polygons with IDDIST ubigeo)
  --gold       gold_monthly.csv                  (columns Date=YYYY-MM, Price=US$/oz)
  --coverage   amazon_basin.geojson              (AMW coverage area; defines the sample)

Optional inputs (add features 7-12 when you have them)
  --protected    polygons of SERNANP protected areas        -> share_protected
  --concessions  polygons of GEOCATMIN mining concessions   -> share_concessions
  --extras       CSV with a 'ubigeo' column plus any district columns
                 (e.g. reinfo_miners, poverty_rate, population) -> merged as-is

Usage
  python build_district_dataset.py --amw amazon_basin_detections.geojson \
      --districts peru_districts.geojson --gold gold_monthly.csv \
      --coverage amazon_basin.geojson --out peru_mining_district_panel.csv
"""
import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

CRS_M = "ESRI:102033"      # South America Albers Equal Area (meters)
FIRST_T, LAST_T = 2019, 2024   # feature years; labels are t+1 = 2020..2025
LAST_DATA_YEAR = 2025          # 2018-2024 confirmed annual; 2025 = provisional quarterly (Q1-Q4 2025)
PROVISIONAL_YEAR = 2025


def parse_year(v):
    """AMW onset: '2018'..'2024' (annual, confirmed) or 'Q125' = Q1 2025 (provisional)."""
    s = str(v).strip().upper()
    if s.startswith("Q") and len(s) == 4:
        return 2000 + int(s[2:])
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits[:4]) if len(digits) >= 4 else np.nan


def load_detections(path):
    det = gpd.read_parquet(path) if path.endswith(".parquet") else gpd.read_file(path)
    det["year"] = det["onset"].map(parse_year)
    keep = (det.status == "confirmed") | ((det.status == "provisional") & (det.year == PROVISIONAL_YEAR))
    det = det[keep & (det.year <= LAST_DATA_YEAR)].copy()
    det["year"] = det["year"].astype(int)
    print("detections used, by onset year:")
    print(det.groupby(["year", "status"]).size().to_string())
    det = det.to_crs(CRS_M)
    det["geometry"] = det.geometry.representative_point()
    return det[["year", "geometry"]]


def share_of_area(districts, polys_path):
    polys = gpd.read_file(polys_path).to_crs(CRS_M)
    polys = gpd.GeoDataFrame(geometry=[polys.union_all()], crs=CRS_M)
    inter = gpd.overlay(districts[["ubigeo", "geometry"]], polys, how="intersection")
    inter["a"] = inter.area
    share = inter.groupby("ubigeo")["a"].sum() / districts.set_index("ubigeo").area
    return share.reindex(districts.ubigeo).fillna(0).clip(0, 1).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amw", required=True)
    ap.add_argument("--districts", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--protected")
    ap.add_argument("--concessions")
    ap.add_argument("--extras")
    ap.add_argument("--out", default="peru_mining_district_panel.csv")
    a = ap.parse_args()

    # Districts and sample ------------------------------------------------------------
    d = gpd.read_file(a.districts).to_crs(CRS_M)
    d = d.rename(columns={"IDDIST": "ubigeo", "NOMBDIST": "district",
                          "NOMBPROV": "province", "NOMBDEP": "department"})
    d["ubigeo"] = d["ubigeo"].astype(str).str.zfill(6)
    cov = gpd.read_file(a.coverage).to_crs(CRS_M).union_all()
    d = d[d.intersects(cov)].reset_index(drop=True)
    d["area_km2"] = d.area / 1e6
    print(f"districts in the AMW coverage area: {len(d):,}")

    # Detections -> districts ------------------------------------------------------------
    det = load_detections(a.amw)
    det_d = gpd.sjoin(det, d[["ubigeo", "geometry"]], how="inner", predicate="within")
    counts = det_d.groupby(["ubigeo", "year"]).size().unstack(fill_value=0)
    years = list(range(2018, LAST_DATA_YEAR + 1))
    counts = counts.reindex(index=d.ubigeo, columns=years, fill_value=0)
    print(f"detections in Peru's sample districts: {len(det_d):,}")

    # Neighbors (districts sharing a border; small buffer absorbs simplification gaps)
    buf = d[["ubigeo", "geometry"]].copy()
    buf["geometry"] = buf.buffer(200)
    nb = gpd.sjoin(buf, d[["ubigeo", "geometry"]], predicate="intersects")
    nb = nb[nb.ubigeo_left != nb.ubigeo_right][["ubigeo_left", "ubigeo_right"]]

    # Distances use ALL detections (incl. Bolivia/Brazil), which can sit next to Peru
    cx = np.column_stack([d.centroid.x, d.centroid.y])
    det_xy = np.column_stack([det.geometry.x, det.geometry.y])
    det_year = det.year.values

    # Gold price: annual average, US$/oz
    g = pd.read_csv(a.gold)
    g["year"] = g["Date"].str[:4].astype(int)
    gold = g.groupby("year")["Price"].mean()

    # Static optional features
    static = d[["ubigeo", "district", "province", "department", "area_km2"]].copy()
    if a.protected:
        static["share_protected"] = share_of_area(d, a.protected)
    if a.concessions:
        static["share_concessions"] = share_of_area(d, a.concessions)

    # Panel --------------------------------------------------------------------------------
    rows = []
    cum = counts.cumsum(axis=1)
    for t in range(FIRST_T, LAST_T + 1):
        new_t = counts[t]
        nb_new = (nb.assign(n=nb.ubigeo_right.map(new_t))
                    .groupby("ubigeo_left")["n"].sum()
                    .reindex(d.ubigeo).fillna(0))
        mask = det_year <= t
        dist = cKDTree(det_xy[mask]).query(cx, k=1)[0] / 1000 if mask.any() else np.nan
        df = static.copy()
        df["year"] = t
        df["mining_present"] = (cum[t].values > 0).astype(int)            # feature 1
        df["new_mining_t"] = new_t.values                                  # feature 2
        df["new_mining_neighbors_t"] = nb_new.values                       # feature 3
        df["dist_nearest_mining_km"] = np.round(dist, 2)                   # feature 4
        df["gold_price"] = round(gold.get(t, np.nan), 1)                   # feature 5
        df["gold_x_mining_present"] = df.gold_price * df.mining_present    # feature 6
        df["new_mining_next_year"] = counts[t + 1].values                  # count in t+1
        df["label"] = (df.new_mining_next_year > 0).astype(int)            # target
        df["label_is_provisional"] = int(t + 1 == PROVISIONAL_YEAR)
        rows.append(df)

    panel = pd.concat(rows, ignore_index=True)
    if a.extras:
        ex = pd.read_csv(a.extras, dtype={"ubigeo": str})
        ex["ubigeo"] = ex.ubigeo.str.zfill(6)
        panel = panel.merge(ex, on="ubigeo", how="left")

    panel.to_csv(a.out, index=False)
    print(f"\nsaved {a.out}: {len(panel):,} rows, {panel.ubigeo.nunique():,} districts")
    print(panel.groupby("year").label.agg(["mean", "sum"]).rename(
        columns={"mean": "share_positive", "sum": "positives"}))


if __name__ == "__main__":
    main()
