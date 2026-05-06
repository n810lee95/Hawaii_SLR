# Hawaii Sea Level Rise Risk Model

Predictive model estimating parcel-level sea level rise inundation risk for Oahu, Hawaii.
Combines LiDAR elevation data, FEMA flood zones, NOAA SLR scenarios, and Hawaii parcel data
to produce a per-parcel risk score and interactive map output.

---

## Project Structure

```
Hawaii Model/
├── audit_gdb_folder.py          # Inspects and catalogs the NOAA .gdb file
├── debug_atx.py                 # Utility for reading GDB index files
├── hawaii_slr_pipeline.py       # Main model pipeline
├── gdb_layer_catalog.csv        # Output from audit script
├── README.md
├── .gitignore
└── Hawaii_SLR_Project/
    ├── hawaii_slr_pipeline.py   # Pipeline (also at root)
    └── data/                    # Download data here (see below — not tracked by Git)
```

---

## Data Sources

Download each file and place it in `Hawaii_SLR_Project/data/` before running the pipeline.
All datasets are free and publicly available.

---

### 1. Oahu Digital Elevation Model (DEM)
**File:** `HI_Oahu_GCS_3m_LMSLm.tif`
**Description:** 3-meter resolution bare-earth DEM for Oahu. Elevations in meters relative
to Local Mean Sea Level (LMSL). CRS: NAD83 geographic (EPSG:4269).
**Source:** NOAA Digital Coast — Continuously Updated Digital Elevation Model (CUDEM)
**URL:** https://coast.noaa.gov/slrdata/DEMs/HI/index.html
**Size:** ~403 MB

---

### 2. Hawaii Statewide Parcels (TMK)
**File:** `hawtmk.shp` (+ associated `.dbf`, `.shx`, `.prj`, `.cpg` files)
**Description:** Hawaii statewide parcel boundaries with Tax Map Key (TMK) identifiers,
land values, building values, zoning, and acreage. 135,471 parcels statewide.
**Source:** Hawaii Statewide GIS Program — Geoportal
**URL:** https://geoportal.hawaii.gov/datasets/1eb5fa03038d49cba930096ea67194e0_5/explore?location=19.582700%2C-155.431750%2C8
**Direct download:** Click "Download" → Shapefile
**Size:** ~26 MB (zipped)

---

### 3. FEMA Base Flood Elevations (BFE)
**File:** `DFIRM_Base_Flood_Elevations_(BFE).shp` (+ associated files)
**Description:** FEMA Digital Flood Insurance Rate Map (DFIRM) base flood elevation lines
for Hawaii. Elevations in feet referenced to Local Tidal Datum. Updated December 2025.
**Source:** Hawaii Open Data Portal
**URL:** https://opendata.hawaii.gov/dataset/dfirm-base-flood-elevations-bfe
**Direct download:** Click "Export" → Shapefile
**Note:** Elevations are in feet — the pipeline converts to meters (× 0.3048)

---

### 4. FEMA DFIRM Water Lines
**File:** `S_WTR_LN.shp` (+ associated files)
**Description:** Stream and water body centerlines used in FEMA flood mapping for Hawaii.
2,470 features. Used in pipeline to compute parcel distance-to-water feature.
**Source:** Hawaii Open Data Portal — DFIRM Flood Hazard Data
**URL:** https://opendata.hawaii.gov/dataset/flood-hazard-areas-dfirm-hawaii-county
**Direct download:** Download the full DFIRM package — S_WTR_LN is included inside
**Size:** Small (~1 MB)

---

### 5. NOAA Sea Level Rise Inundation Zones
**File:** `HI_Oahu_slr_final_dist.gdb/` (ESRI File Geodatabase folder)
**Description:** NOAA SLR inundation scenario polygons for Oahu at 0–10 ft of sea level
rise (1 ft increments). Contains two layer series:
- `hi_oahu_slr_Xft` — ocean-connected inundation zones (used as model labels)
- `hi_oahu_low_Xft` — low-lying land below each threshold (used as model features)
gridcode field = SLR scenario in feet (1 = 1ft, 2 = 2ft ... 10 = 10ft)
CRS: NAD83 geographic (EPSG:4269)
**Source:** NOAA Office for Coastal Management — Sea Level Rise Viewer
**URL:** https://coast.noaa.gov/slrdata/Sea_Level_Rise_Vectors/HI/index.html
**Direct download:** Download the HI Oahu slr data dis.zip
**Size:** ~20 MB (zipped)

---


### 6. FEMA DFIRM Flood Hazard Areas (optional — additional feature)
**File:** Included in the DFIRM package from Source 4 above
**Description:** Polygon layer of FEMA flood zone classifications (AE, VE, X zones).
Useful as an additional categorical feature in the model.
**URL:** https://opendata.hawaii.gov/dataset/flood-hazard-areas-dfirm-hawaii-county

---

## Setup

### Requirements
- Python 3.9+
- ~5 GB disk space for all data files

### Install dependencies
```bash
pip install -r requirements.txt
```

### File paths
Update the `PATHS` dictionary at the top of `hawaii_slr_pipeline.py` to match
where you saved the data files on your machine:

```python
PATHS = {
    "dem":     "Hawaii_SLR_Project/data/HI_Oahu_GCS_3m_LMSLm.tif",
    "parcels": "Hawaii_SLR_Project/data/hawtmk.shp",
    "bfe":     "Hawaii_SLR_Project/data/DFIRM_Base_Flood_Elevations_(BFE).shp",
    "water":   "Hawaii_SLR_Project/data/S_WTR_LN.shp",
    "slr_gdb": "Hawaii_SLR_Project/data/HI_Oahu_slr_final_dist.gdb",
}
```

---

## Usage

### Step 1 — Audit the GDB file to confirm layer names
```bash
python audit_gdb_folder.py --gdb "Hawaii_SLR_Project/data/HI_Oahu_slr_final_dist.gdb"
```
This produces `gdb_layer_catalog.csv` confirming all available SLR scenario layers.

### Step 2 — Run the main pipeline
```bash
python hawaii_slr_pipeline.py
```

### Outputs
All outputs are saved to the `outputs/` folder (created automatically):

| File | Description |
|---|---|
| `oahu_slr_risk_scores.csv` | Risk score and tier for every Oahu parcel |
| `oahu_slr_risk_scores.geojson` | Same data in GeoJSON format for GIS |
| `oahu_slr_risk_map.html` | Interactive Folium map with parcel popups |
| `feature_importance.png` | Model feature importance chart |
| `risk_distribution.png` | Risk score and tier distribution plots |
| `elevation_vs_risk.png` | Elevation vs predicted risk scatter plot |

---

## Model Overview

The pipeline trains three candidate models and selects the best by AUC-ROC:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline — interpretable, fast |
| Random Forest | Handles non-linear relationships |
| XGBoost | Typically best on tabular geospatial data |

**Target variable:** Binary — parcel inundated (1) or not (0) at the chosen SLR scenario
(default: 3ft, Hawaii state planning standard). Can be changed via `SLR_TARGET_FEET`.

**Key features:**
- Minimum and mean parcel elevation from LiDAR DEM
- Distance to nearest water body
- Nearest FEMA Base Flood Elevation and distance to BFE line
- Elevation above/below BFE threshold
- Parcel latitude/longitude
- Land value, building value, parcel acreage

---

## SLR Scenario Reference

| Scenario | Approx. Year | Planning Context |
|---|---|---|
| 1 ft | ~2030s | Near-term planning horizon |
| 3 ft | ~2050s | Hawaii state minimum planning standard |
| 4 ft | ~2060s | Hawaii state minimum design standard |
| 6 ft | ~2070s | Mid-century high emissions scenario |
| 10 ft | 2100+ | Long-term worst case |

Source: 2022 Hawaii Sea Level Rise Vulnerability and Adaptation Report

---

## Author
Nathan
