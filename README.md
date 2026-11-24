# Forecasting New Gold Mining in the Peruvian Amazon

Can free public data predict **where new gold mining will appear next year** in the Peruvian Amazon,
so that patrols go to the right places first?

This repository builds a 5 km grid over the Peruvian Amazon (38,493 cells, 2019–2024, 230,958 cell-years),
joins seven public data sources, and evaluates simple rules and machine-learning models with a
leakage-safe rolling evaluation on future years.

## Main findings

1. **New mining spreads outward from existing mining.** 90% of the mining that appears in never-mined
   cells (110 of 122 cases) lies within 10 km of existing mining; the median distance is 4.4 km.
2. **A simple rule finds the right zone.** The 10 km ring around active mining covers only 1.5% of
   never-mined cells but contains 90% of new mining in never-mined places.
3. **Inside the ring, a model ranks better than simple rules.** Models trained only on ring cells reach
   PR-AUC 0.90 (95% CI 0.88–0.92), against 0.82 for distance and 0.88 for last year's expansion.
   In a paired comparison the gain over last year's expansion is small but consistent:
   +0.024 PR-AUC (95% CI +0.017 to +0.031), with both models ahead in every bootstrap draw (and the logistic model ahead in each of the 5 test years).
4. **Exactly which never-mined cell is hit next is not predictable with public data.** No model beats
   distance on never-mined cells (PR-AUC 0.17).
5. **Miners go where the gold is, but existing mines already reveal it.** Gold geology (INGEMMET) differs
   sharply between cells that get mining and those that don't, yet adds no predictive power beyond
   distance to existing mining.

### Recommended two-stage system

| Stage | Tool | Role | Performance |
| --- | --- | --- | --- |
| 1 | Rule: 10 km ring around active mining | Where to look | 90% of new frontier mining in 1.5% of never-mined land |
| 2 | Model trained on ring cells | Which places in the ring first | PR-AUC 0.90 vs 0.82 for distance |

### Key results (rolling evaluation, 5 test years: new mining in 2021–2025)

| Model | PR-AUC, all cells | PR-AUC, never-mined cells |
| --- | --- | --- |
| Baseline: distance to nearest mining | 0.817 (0.788–0.850) | **0.166** (0.130–0.247) |
| Logistic regression, past-mining features | **0.884** (0.862–0.906) | 0.146 |
| XGBoost (monotone), all features incl. gold | **0.884** (0.863–0.906) | 0.145 |

| Inside the 10 km ring (4,527 cell-years, 26% positive) | PR-AUC (95% CI) | ROC-AUC | Gain vs persistence (95% CI) |
| --- | --- | --- | --- |
| Distance | 0.824 (0.788–0.854) | 0.934 | −0.054 (−0.076 to −0.033) |
| Persistence (last year's new patches) + distance | 0.878 (0.853–0.900) | 0.923 | — |
| Logistic regression, ring-trained | **0.901** (0.878–0.920) | 0.945 | +0.024 (+0.017 to +0.031) |
| XGBoost (monotone), ring-trained | **0.901** (0.879–0.921) | 0.946 | +0.024 (+0.017 to +0.031) |

The gain is a paired bootstrap difference over cells (both rankings scored on the same resampled cells).

Full tables are in `results/`.

## How to reproduce

Requires Python 3.11+ (results verified with Python 3.13) and about 1 GB of disk space.
`requirements.txt` pins exact package versions; XGBoost results can shift in the third decimal with other versions. Run every command from the repository root.

```bash
pip install -r requirements.txt

python src/download_data.py          # 1. download all public data into data/raw/ (~2 min, ~0.5 GB)
python src/build_grid_dataset.py     # 2. 5 km grid, labels, past-mining features
python src/add_geo_features.py       # 3. rivers, protected areas, poverty, population
python src/add_dynamic_features.py   # 4. river-network distances, recent mining, rings, lags
python src/add_gold_features.py      # 5. INGEMMET gold geology  -> data/peru_mining_grid_panel_v4.csv
python src/rolling_eval.py           # 6. main rolling evaluation (~6 min)
python src/expansion_eval.py         # 7. "how much will mining grow" tasks
python src/ring_eval.py              # 8. two-stage system: ranking inside the 10 km ring
```

Steps 2–5 take under a minute in total. Results are written to `results/`.

## Repository structure

```
src/
  download_data.py          step 1: all data downloads
  download_ingemmet.py      helper: paged downloads from INGEMMET's map server
  build_grid_dataset.py     step 2: grid, labels, past-mining features
  add_geo_features.py       step 3: geography features
  add_dynamic_features.py   step 4: river-network and mining-dynamics features
  add_gold_features.py      step 5: gold geology features
  rolling_eval.py           step 6: final rolling evaluation (shared model and feature definitions)
  expansion_eval.py         step 7: expansion-volume and big-expansion targets
  ring_eval.py              step 8: two-stage ring analysis
experiments/                earlier stages kept for transparency
  build_district_dataset.py       first attempt at district level (abandoned: too few positives)
  single_split_eval.py            first train/test split (train 2019-2022)
  rolling_eval_v2_geography.py    rolling evaluation after adding geography
  rolling_eval_v3_dynamics.py     rolling evaluation after adding dynamics and river networks
results/                    result tables (CSV)
data/                       created by the scripts; not tracked in git
```

## Method in brief

- **Unit:** 5 km × 5 km cell × year. **Label:** 1 if a new mining patch is first detected in the cell in year t+1.
- **Leakage control:** every feature uses only information available by the end of year t.
- **Evaluation:** expanding-window rolling evaluation (train on years < t, test on year t, for t = 2020–2024),
  with simple baselines (distance, persistence, momentum).
- **Metrics:** PR-AUC and recall in the top-k cells (rare positives: ~0.6% of rows); ROC-AUC for reference only.
- **Uncertainty:** bootstrap over cells (all years of a cell kept together), 200 draws.
- **Models:** logistic regression (log-transformed, standardized features, balanced class weights) and
  XGBoost with monotone constraints (risk can only fall with distance to mining and rise with nearby mining).

## Data sources

| Source | Use | License / terms |
| --- | --- | --- |
| [Amazon Mining Watch](https://source.coop/earthgenome/amazon-mining-watch) (Earth Genome, Pulitzer Center) | Mining detections (labels) | CC-BY-4.0 |
| [mining-detector boundaries](https://github.com/earthrise-media/mining-detector) | Study-area coverage | See repository |
| [Peru district boundaries](https://github.com/juaneladio/peru-geojson) (INEI) | District codes | See repository |
| [Gold prices](https://github.com/datasets/gold-prices) | Yearly gold price | See repository |
| [HydroRIVERS](https://hydrosheds.org/page/hydrorivers) | Rivers and river network | Free for scientific, educational and commercial use |
| SERNANP geoservices | Protected areas and buffer zones | Peruvian public data |
| INGEMMET GEOCATMIN | Gold occurrences, metallogenic belts, Quaternary geology | Peruvian public data |
| [INEI poverty map 2018](https://github.com/calderonsamuel/perupobreza2018) | District poverty and population | See repository |

## Limitations

- Labels come from a satellite model: they include legal mining and some false positives
  (61% of never-mined positives are a single ~20 ha patch).
- Only the Amazon side of Peru is covered; Andean mining is outside the labels.
- Legal concessions, the REINFO registry and roads could not be included.
- 2025 labels are provisional (quarterly detections).
- Hyperparameters were set by hand, not tuned.
