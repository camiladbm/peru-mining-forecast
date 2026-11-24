"""
Helper for step 1. Run from the repository root: python src/download_ingemmet.py

Download INGEMMET geology layers from the GEOCATMIN mirror (geocatminnube.ingemmet.gob.pe) as GeoJSON.

  ocurrencias_metalicas.geojson   metallic mineral occurrences (SERV_OCURRENCIA_MINERAL/0)
  franjas_metalogeneticas.geojson metallogenic belts (SERV_METALOGENETICO/4)
  pequena_mineria.geojson         small-mining sampling points (SERV_PEQUENA_MINERIA/0)
  geologia_cuaternaria.geojson    Quaternary / Plio-Quaternary units (name contains 'Q') from the
                                  1:100k integrated geology map, clipped to the study area box
"""
import json
import os
import time
import urllib.parse
import urllib.request

H = "https://geocatminnube.ingemmet.gob.pe/arcgis/rest/services"
OUT = "data/raw/ext"
BBOX = "-79.5,-14.5,-68.6,-0.1"


def get(url, params, tries=4):
    q = url + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            req = urllib.request.Request(q, headers={"User-Agent": "Mozilla/5.0 (research data download)"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            print("  retry", i + 1, e)
            time.sleep(5 * (i + 1))
    raise RuntimeError(q)


def download(layer, fname, where="1=1", fields="*", bbox=False, offset=None, page=1000):
    url = f"{H}/{layer}/query"
    base = {"where": where, "outFields": fields, "outSR": 4326, "f": "geojson",
            "resultRecordCount": page, "orderByFields": "objectid"}
    if bbox:
        base.update({"geometry": BBOX, "geometryType": "esriGeometryEnvelope", "inSR": 4326,
                     "spatialRel": "esriSpatialRelIntersects"})
    if offset:
        base["maxAllowableOffset"] = offset
    feats, start = [], 0
    while True:
        d = get(url, {**base, "resultOffset": start})
        if "error" in d:
            raise RuntimeError(d)
        got = d.get("features", [])
        feats += got
        print(f"  {fname}: {len(feats):,}")
        if len(got) < page and not d.get("exceededTransferLimit") and not d.get("properties", {}).get("exceededTransferLimit"):
            break
        if not got:
            break
        start += len(got)
    with open(os.path.join(OUT, fname), "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    return len(feats)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    download("SERV_OCURRENCIA_MINERAL/MapServer/0", "ocurrencias_metalicas.geojson")
    download("SERV_METALOGENETICO/MapServer/4", "franjas_metalogeneticas.geojson")
    download("SERV_PEQUENA_MINERIA/MapServer/0", "pequena_mineria.geojson")
    download("SERV_GEOLOGIA_100K_INTEGRADA/MapServer/6", "geologia_cuaternaria.geojson",
             where="name LIKE '%Q%'", fields="name,unidad", bbox=True, offset=0.001)
