"""
Step 5. Run from the repository root: python src/add_gold_features.py

Add "where is the gold" features (INGEMMET) -> data/peru_mining_grid_panel_v4.csv

Cell-level, time-invariant:
  in_gold_belt              cell center inside a metallogenic belt whose type includes Au
  dist_gold_occurrence_km   km to nearest INGEMMET mineral occurrence with Au
  gold_occ_within_20km      number of Au occurrences within 20 km
  share_alluvial            share of the cell (25 sample points on a 1 km lattice) on alluvial/fluvial deposits
  share_madre_de_dios_fm    share of the cell on the Madre de Dios Formation (gold-bearing terraces)
  share_quaternary          share of the cell on any Quaternary / Plio-Quaternary unit
Not used: INGEMMET small-mining sample points (taken at existing mines in 2023 -> would leak).
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

CRS_M = "ESRI:102033"
EXT = "data/raw/ext"


def inside_any(points, polys):
    """Boolean array: point inside any polygon (spatial index join, no union)."""
    pts = gpd.GeoDataFrame(geometry=points.reset_index(drop=True))
    j = gpd.sjoin(pts, polys[["geometry"]], predicate="within", how="inner")
    out = np.zeros(len(pts), bool)
    out[j.index.unique()] = True
    return out


def share_inside(polys, cx, cy, tf_inv):
    offs = np.arange(-2000, 2001, 1000)
    ox, oy = np.meshgrid(offs, offs)
    px = (cx[:, None] + ox.ravel()[None, :]).ravel()
    py = (cy[:, None] + oy.ravel()[None, :]).ravel()
    lon, lat = tf_inv.transform(px, py)
    pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs="EPSG:4326")
    return inside_any(pts, polys).reshape(len(cx), -1).mean(axis=1).round(2)


def main():
    p = pd.read_csv("data/peru_mining_grid_panel_v3.csv")
    cells = p.drop_duplicates("cell_id")[["cell_id", "lon", "lat"]].reset_index(drop=True)
    tf = Transformer.from_crs("EPSG:4326", CRS_M, always_xy=True)
    tf_inv = Transformer.from_crs(CRS_M, "EPSG:4326", always_xy=True)
    cx, cy = tf.transform(cells.lon.values, cells.lat.values)
    pts = gpd.GeoSeries(gpd.points_from_xy(cells.lon, cells.lat), crs="EPSG:4326")

    fr = gpd.read_file(f"{EXT}/franjas_metalogeneticas.geojson")
    fr["geometry"] = fr.make_valid()
    au_belts = fr[fr.tipo.str.contains(r"\bAu\b|Au-|-Au", regex=True, na=False)]
    print(f"belts with Au: {len(au_belts)} of {len(fr)}")
    cells["in_gold_belt"] = inside_any(pts, au_belts).astype(int)

    oc = gpd.read_file(f"{EXT}/ocurrencias_oro.geojson").to_crs(CRS_M)  # Au from layers 0 + 2
    t = cKDTree(np.column_stack([oc.geometry.x, oc.geometry.y]))
    cxy = np.column_stack([cx, cy])
    cells["dist_gold_occurrence_km"] = (t.query(cxy)[0] / 1000).round(2)
    cells["gold_occ_within_20km"] = t.query_ball_point(cxy, 20000, return_length=True)

    q = gpd.read_file(f"{EXT}/geologia_cuaternaria.geojson")
    q["geometry"] = q.make_valid()
    alluv = q[q.unidad.str.contains("aluvial|fluvial", case=False, na=False)]
    mdd = q[q.unidad.str.contains("Madre de Dios", case=False, na=False)]
    print(f"alluvial/fluvial polygons: {len(alluv):,}; Madre de Dios Fm: {len(mdd):,}")
    cells["share_alluvial"] = share_inside(alluv, cx, cy, tf_inv)
    cells["share_madre_de_dios_fm"] = share_inside(mdd, cx, cy, tf_inv)
    cells["share_quaternary"] = share_inside(q, cx, cy, tf_inv)

    out = p.merge(cells.drop(columns=["lon", "lat"]), on="cell_id", how="left")
    out.to_csv("data/peru_mining_grid_panel_v4.csv", index=False)
    print("saved data/peru_mining_grid_panel_v4.csv", out.shape)
    print(cells.drop(columns=["cell_id", "lon", "lat"]).describe().T.round(2))
    # quick look: do frontier positives sit on gold geology more than other frontier cells?
    f = out[(out.frontier == 1) & (out.year >= 2020)]
    print(f.groupby("label")[["in_gold_belt", "dist_gold_occurrence_km", "share_alluvial",
                             "share_madre_de_dios_fm"]].mean().round(3))


if __name__ == "__main__":
    main()
