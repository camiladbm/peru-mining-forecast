"""
Step 1. Download every public data source into data/raw/.

Run from the repository root:  python src/download_data.py

Sources
  Amazon Mining Watch detections (Earth Genome / Pulitzer Center, CC-BY-4.0)
  Amazon Mining Watch coverage boundaries (earthrise-media/mining-detector on GitHub)
  Peru district boundaries (INEI, via github.com/juaneladio/peru-geojson)
  Monthly gold price (github.com/datasets/gold-prices)
  HydroRIVERS v1.0, South America (hydrosheds.org)
  SERNANP protected areas and buffer zones (geoservicios.sernanp.gob.pe)
  INEI district poverty map 2018 (github.com/calderonsamuel/perupobreza2018)
  INGEMMET geology: gold occurrences, metallogenic belts, Quaternary units (geocatminnube.ingemmet.gob.pe)

Total download: roughly 0.5 GB. Files already present are skipped.
"""
import json
import os
import shutil
import urllib.request
import zipfile

RAW = "data/raw"
EXT = f"{RAW}/ext"

FILES = {
    f"{RAW}/amazon_basin_detections.geojson":
        "https://data.source.coop/earthgenome/amazon-mining-watch/amazon_basin_detections.geojson",
    f"{RAW}/Amazon_ACA.geojson":
        "https://raw.githubusercontent.com/earthrise-media/mining-detector/main/data/boundaries/Amazon_ACA.geojson",
    f"{RAW}/peru_districts.geojson":
        "https://raw.githubusercontent.com/juaneladio/peru-geojson/master/peru_distrital_simple.geojson",
    f"{RAW}/gold_monthly.csv":
        "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv",
    f"{EXT}/hydrorivers_sa.zip":
        "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_sa_shp.zip",
    f"{EXT}/pobreza2018.rda":
        "https://raw.githubusercontent.com/calderonsamuel/perupobreza2018/master/data/pobreza2018.rda",
}
SERNANP = "https://geoservicios.sernanp.gob.pe/arcgis/rest/services"
QUERY = "query?where=1%3D1&outFields=*&outSR=4326&f=geojson"
FILES.update({
    f"{EXT}/anp_0.geojson": f"{SERNANP}/base_fisica/peru_sernanp_0102/MapServer/0/{QUERY}",  # national, definitive
    f"{EXT}/anp_1.geojson": f"{SERNANP}/base_fisica/peru_sernanp_0102/MapServer/1/{QUERY}",  # national, transitory
    f"{EXT}/anp_2.geojson": f"{SERNANP}/base_fisica/peru_sernanp_0102/MapServer/2/{QUERY}",  # regional
    f"{EXT}/za.geojson": f"{SERNANP}/gestion_de_anp/peru_sernanp_0214/MapServer/0/{QUERY}",  # buffer zones
})


def fetch(path, url):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"skip  {path}")
        return
    print(f"get   {path}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research data download)"})
    tmp = path + ".part"
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(tmp, path)  # only keep complete downloads


def main():
    os.makedirs(EXT, exist_ok=True)
    for path, url in FILES.items():
        fetch(path, url)

    # HydroRIVERS shapefile
    if not os.path.exists(f"{EXT}/HydroRIVERS_v10_sa_shp"):
        with zipfile.ZipFile(f"{EXT}/hydrorivers_sa.zip") as z:
            z.extractall(EXT)

    # Poverty table: R data file -> CSV
    if not os.path.exists(f"{EXT}/pobreza2018_district.csv"):
        import rdata
        r = rdata.read_rda(f"{EXT}/pobreza2018.rda")
        next(iter(r.values())).to_csv(f"{EXT}/pobreza2018_district.csv", index=False)

    # Mining detections: GeoJSON -> Parquet (much faster to read)
    if not os.path.exists(f"{RAW}/amw.parquet"):
        import geopandas as gpd
        gpd.read_file(f"{RAW}/amazon_basin_detections.geojson").to_parquet(f"{RAW}/amw.parquet")

    # INGEMMET geology (paged ArcGIS queries)
    import download_ingemmet as ing
    if not os.path.exists(f"{EXT}/geologia_cuaternaria.geojson"):
        ing.download("SERV_OCURRENCIA_MINERAL/MapServer/0", "ocurrencias_metalicas.geojson")
        ing.download("SERV_OCURRENCIA_MINERAL/MapServer/2", "ocurrencias_referenciales.geojson")
        ing.download("SERV_METALOGENETICO/MapServer/4", "franjas_metalogeneticas.geojson")
        ing.download("SERV_GEOLOGIA_100K_INTEGRADA/MapServer/6", "geologia_cuaternaria.geojson",
                     where="name LIKE '%Q%'", fields="name,unidad", bbox=True, offset=0.001)

    # Gold occurrences from both occurrence layers (element field contains Au)
    if not os.path.exists(f"{EXT}/ocurrencias_oro.geojson"):
        feats = []
        for f in ["ocurrencias_metalicas.geojson", "ocurrencias_referenciales.geojson"]:
            with open(f"{EXT}/{f}") as fh:
                for ft in json.load(fh)["features"]:
                    props = {k.lower(): v for k, v in ft["properties"].items()}
                    if "AU" in str(props.get("elemento", "")).upper():
                        feats.append({"type": "Feature", "geometry": ft["geometry"],
                                      "properties": {"elemento": props.get("elemento")}})
        with open(f"{EXT}/ocurrencias_oro.geojson", "w") as fh:
            json.dump({"type": "FeatureCollection", "features": feats}, fh)
        print(f"gold occurrences: {len(feats)}")
    print("done")


if __name__ == "__main__":
    main()
