"""
Step 4. Run from the repository root: python src/add_dynamic_features.py

Add river-network and mining-dynamics features -> data/peru_mining_grid_panel_v3.csv

All features for row (cell, t) use detections with onset <= t only.

River network (HydroRIVERS reaches with mean flow >= 1 m3/s; cell and detections snapped to nearest reach)
  river_net_dist_mining_km     distance ALONG RIVERS from the cell's reach to the nearest reach with mining (<= t)
  dist_downstream_of_mining_km along-flow distance if the cell lies DOWNSTREAM of mining (999 if none within 500 km)
  dist_upstream_of_mining_km   along-flow distance if the cell lies UPSTREAM of mining (999 if none)
  river_net_dist_recent_km     same as the first, but only mining that started in t-1 or t
  cell_to_reach_km             straight-line distance from cell center to its reach
Mining dynamics (straight-line)
  dist_recent_mining_km        distance to mining with onset in t-1 or t
  mined_0_5km, mined_5_10km, mined_10_20km   detections up to t in distance rings
  new_within_10km_lag1         new detections within 10 km in t-1
  growth_20km_2yr              change in detections within 20 km from t-2 to t
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely import get_coordinates

CRS_M = "ESRI:102033"
BBOX = (-81.5, -18.5, -68.5, 0.1)
LIMIT_KM = 500
FAR = 999.0


def parse_year(v):
    s = str(v).strip().upper()
    return 2000 + int(s[2:]) if s.startswith("Q") else int(s[:4])


def main():
    p = pd.read_csv("data/peru_mining_grid_panel_v2.csv")
    cells = p.drop_duplicates("cell_id")[["cell_id", "lon", "lat"]].sort_values("cell_id").reset_index(drop=True)
    tf = Transformer.from_crs("EPSG:4326", CRS_M, always_xy=True)
    cxy = np.column_stack(tf.transform(cells.lon.values, cells.lat.values))

    det = gpd.read_parquet("data/raw/amw.parquet")
    det["year"] = det.onset.map(parse_year)
    det = det[(det.status == "confirmed") | (det.year == 2025)].to_crs(CRS_M)
    dpts = det.geometry.representative_point()
    dxy, dyr = np.column_stack([dpts.x, dpts.y]), det.year.values

    # ---- River graph --------------------------------------------------------------
    rv = gpd.read_file("data/raw/ext/HydroRIVERS_v10_sa_shp/HydroRIVERS_v10_sa.shp", bbox=BBOX)
    rv = rv[rv.DIS_AV_CMS >= 1].reset_index(drop=True).to_crs(CRS_M)
    n = len(rv)
    pos = pd.Series(np.arange(n), index=rv.HYRIV_ID.values)
    down = rv.NEXT_DOWN.map(pos)
    has = down.notna().values
    src, dst = np.arange(n)[has], down[has].astype(int).values
    # edge weight = half of each reach's length (center to center)
    w = (rv.LENGTH_KM.values[src] + rv.LENGTH_KM.values[dst]) / 2
    G_down = csr_matrix((w, (src, dst)), shape=(n, n))        # flow direction
    G_up = G_down.T.tocsr()                                    # against flow
    G_und = (G_down + G_up).tocsr()
    print(f"river reaches (>= 1 m3/s): {n:,}, links: {has.sum():,}")

    # snap cells and detections to nearest reach via densified vertices
    geo = rv.geometry.segmentize(500)
    verts = get_coordinates(geo.values)
    counts = np.array([len(get_coordinates(g)) for g in geo.values])
    vreach = np.repeat(np.arange(n), counts)
    vt = cKDTree(verts)
    c_off, c_v = vt.query(cxy)
    c_reach = vreach[c_v]
    d_off, d_v = vt.query(dxy)
    d_reach = vreach[d_v]
    near = d_off < 5000  # only detections within 5 km of a mapped river seed the network
    print(f"detections snapped to rivers (<5 km): {near.mean():.1%}")

    def net_dist(G, mask, directed=True):
        srcs = np.unique(d_reach[mask & near])
        if len(srcs) == 0:
            return np.full(len(cells), FAR)
        d = dijkstra(G, directed=directed, indices=srcs, min_only=True, limit=LIMIT_KM)
        out = d[c_reach]
        return np.where(np.isfinite(out), out, FAR)

    tree_all = {}
    rows = []
    for t in sorted(p.year.unique()):
        upto, recent = dyr <= t, (dyr >= t - 1) & (dyr <= t)
        f = pd.DataFrame({"cell_id": cells.cell_id, "year": t})
        f["river_net_dist_mining_km"] = np.minimum(net_dist(G_und, upto, directed=False) + c_off / 1000, FAR)
        # cell downstream of mining = reachable from mining following the flow
        f["dist_downstream_of_mining_km"] = net_dist(G_down, upto)
        f["dist_upstream_of_mining_km"] = net_dist(G_up, upto)
        f["river_net_dist_recent_km"] = np.minimum(net_dist(G_und, recent, directed=False) + c_off / 1000, FAR)
        f["cell_to_reach_km"] = (c_off / 1000).round(2)
        ta, tr = cKDTree(dxy[upto]), cKDTree(dxy[recent])
        f["dist_recent_mining_km"] = (tr.query(cxy)[0] / 1000).round(2)
        r5 = ta.query_ball_point(cxy, 5000, return_length=True)
        r10 = ta.query_ball_point(cxy, 10000, return_length=True)
        r20 = ta.query_ball_point(cxy, 20000, return_length=True)
        f["mined_0_5km"], f["mined_5_10km"], f["mined_10_20km"] = r5, r10 - r5, r20 - r10
        lag = dyr == t - 1
        f["new_within_10km_lag1"] = (cKDTree(dxy[lag]).query_ball_point(cxy, 10000, return_length=True)
                                     if lag.any() else 0)
        prev2 = dyr <= t - 2
        f["growth_20km_2yr"] = r20 - cKDTree(dxy[prev2]).query_ball_point(cxy, 20000, return_length=True)
        rows.append(f)
        print(f"t={t} done")

    feats = pd.concat(rows)
    # Onset 2018 means "already mined by 2018" (the starting stock), not new in 2018,
    # so "recent" and "lagged new" features are undefined for t = 2019.
    first = feats.year == 2019
    feats.loc[first, ["river_net_dist_recent_km", "dist_recent_mining_km",
                      "new_within_10km_lag1", "growth_20km_2yr"]] = np.nan
    out = p.merge(feats, on=["cell_id", "year"], how="left")
    out.to_csv("data/peru_mining_grid_panel_v3.csv", index=False)
    print("saved data/peru_mining_grid_panel_v3.csv", out.shape)
    print(feats.drop(columns=["cell_id"]).describe().T.round(2))


if __name__ == "__main__":
    main()
