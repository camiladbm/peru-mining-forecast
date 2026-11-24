"""
Step 3. Run from the repository root: python src/add_geo_features.py

Add geography features to peru_mining_grid_panel.csv -> peru_mining_grid_panel_v2.csv

New cell-level features (time-invariant)
  dist_river_km          distance to any river reach (HydroRIVERS, catchment >= 10 km2)
  dist_river_10cms_km    distance to a river with average flow >= 10 m3/s
  dist_river_100cms_km   distance to a river with average flow >= 100 m3/s (large rivers)
  in_protected_area      cell center inside a protected area (national definitive,
                         transitory, or regional conservation area; SERNANP)
  in_buffer_zone         cell center inside a protected-area buffer zone (SERNANP)
  dist_protected_km      distance to the nearest protected-area edge (0 if inside)
New district-level features (joined by ubigeo)
  poverty_rate           monetary poverty %, INEI district poverty map 2018
  pop_density            people per km2 (INEI 2020 population / district area)
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import get_coordinates

CRS_M = "ESRI:102033"
RAW = "data/raw"
EXT = f"{RAW}/ext"
PERU_BBOX = (-81.5, -18.5, -68.5, 0.1)


def vertex_tree(geoms):
    """KD-tree over all vertices of (densified) line/polygon boundaries, in meters."""
    geoms = geoms.to_crs(CRS_M).segmentize(500)
    return cKDTree(get_coordinates(geoms.values))


def main():
    p = pd.read_csv("data/peru_mining_grid_panel.csv", dtype={"ubigeo": str})
    cells = p.drop_duplicates("cell_id")[["cell_id", "ubigeo", "lon", "lat"]].reset_index(drop=True)
    pts = gpd.GeoDataFrame(cells, geometry=gpd.points_from_xy(cells.lon, cells.lat), crs="EPSG:4326")
    xy = get_coordinates(pts.to_crs(CRS_M).geometry.values)

    # Rivers
    rv = gpd.read_file(f"{EXT}/HydroRIVERS_v10_sa_shp/HydroRIVERS_v10_sa.shp", bbox=PERU_BBOX)
    print(f"river reaches in Peru bbox: {len(rv):,}")
    for name, sub in [("dist_river_km", rv), ("dist_river_10cms_km", rv[rv.DIS_AV_CMS >= 10]),
                      ("dist_river_100cms_km", rv[rv.DIS_AV_CMS >= 100])]:
        cells[name] = (vertex_tree(sub.geometry).query(xy)[0] / 1000).round(2)

    # Protected areas and buffer zones
    anp = pd.concat([gpd.read_file(f"{EXT}/anp_{i}.geojson")[["geometry"]] for i in (0, 1, 2)])
    anp = gpd.GeoDataFrame(anp, crs="EPSG:4326").make_valid()
    za = gpd.read_file(f"{EXT}/za.geojson")[["geometry"]].make_valid()
    cells["in_protected_area"] = pts.geometry.within(anp.union_all()).astype(int)
    cells["in_buffer_zone"] = pts.geometry.within(za.union_all()).astype(int)
    d_edge = vertex_tree(anp.boundary).query(xy)[0] / 1000
    cells["dist_protected_km"] = np.where(cells.in_protected_area == 1, 0, d_edge).round(2)

    # District poverty and population density
    pov = pd.read_csv(f"{EXT}/pobreza2018_district.csv", dtype={"ubigeo": str})
    pov["ubigeo"] = pov.ubigeo.str.zfill(6)
    dist = gpd.read_file(f"{RAW}/peru_districts.geojson").to_crs(CRS_M)
    area = pd.Series(dist.area.values / 1e6, index=dist.IDDIST.astype(str).str.zfill(6), name="area_km2")
    pov = pov.join(area, on="ubigeo")
    pov["pop_density"] = (pov.pob_2020 / pov.area_km2).round(2)
    cells = cells.merge(pov[["ubigeo", "pobreza_monetaria", "pop_density"]]
                        .rename(columns={"pobreza_monetaria": "poverty_rate"}), on="ubigeo", how="left")
    print("missing poverty for", cells.poverty_rate.isna().sum(), "cells (district code changes)")

    new_cols = [c for c in cells.columns if c not in ("ubigeo", "lon", "lat")]
    out = p.merge(cells[new_cols], on="cell_id", how="left")
    out.to_csv("data/peru_mining_grid_panel_v2.csv", index=False)
    print(f"saved data/peru_mining_grid_panel_v2.csv: {out.shape}")
    print(cells.describe().T.round(2))


if __name__ == "__main__":
    main()
