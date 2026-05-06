"""
Hawaii Sea Level Rise - Predictive Model Data Pipeline
=======================================================
Full pipeline: raw files → feature matrix → trained model → risk predictions

Data inputs:
  - HI_Oahu_GCS_3m_LMSLm.tif        : Oahu DEM (3m, MSL meters, NAD83)
  - hawtmk.shp                      : Hawaii statewide parcels (TMK)
  - DFIRM_Base_Flood_Elevations.shp : FEMA Base Flood Elevations
  - S_WTR_LN.shp                    : FEMA water body lines
  - HI_Oahu_slr_final_dist.gdb      : NOAA SLR inundation zones (gridcode 1-10)

Author: Nathan
"""

# ─────────────────────────────────────────────────────────────
# 0. INSTALL DEPENDENCIES
# ─────────────────────────────────────────────────────────────
# Run once in terminal:
#   pip install geopandas rasterio rasterstats shapely pandas numpy
#   pip install scikit-learn xgboost matplotlib seaborn folium

# ─────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.mask import mask as _rasterio_mask
from shapely.geometry import Point
from shapely.ops import unary_union

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.inspection import permutation_importance
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

print("✓ All imports successful")

# ─────────────────────────────────────────────────────────────
# 2. FILE PATHS  ← Update these to your actual file locations
# ─────────────────────────────────────────────────────────────
PATHS = {
    # Elevation raster (your 403MB TIF)
    "dem": r"C:\Users\natha\OneDrive\Desktop\Nathan\Datacamp Python\Hawaii Model\Hawaii_SLR_Project\data\HI_Oahu_GCS_3m_LMSLm.tif",

    # Parcel shapefile (hawtmk)
    "parcels": r"C:\Users\natha\OneDrive\Desktop\Nathan\Datacamp Python\Hawaii Model\Hawaii_SLR_Project\data\tmk_state.shp",

    # FEMA layers
    "bfe": r"C:\Users\natha\OneDrive\Desktop\Nathan\Datacamp Python\Hawaii Model\Hawaii_SLR_Project\data\DFIRM_Base_Flood_Elevations_(BFE).shp",
    "water_lines": r"C:\Users\natha\OneDrive\Desktop\Nathan\Datacamp Python\Hawaii Model\Hawaii_SLR_Project\data\S_WTR_LN.shp",

    # NOAA SLR inundation zones
    # If you have a .gdb folder:
    "slr": r"C:\Users\natha\OneDrive\Desktop\Nathan\Datacamp Python\Hawaii Model\HI_Oahu_slr_final_dist.gdb"
    # If you extracted to shapefiles:
    #"slr_layer1": "noaa_slr_inundation_zones_layer1.shp",
    #"slr_layer2": "noaa_slr_inundation_zones_layer2.shp",
    #"slr_layer3": "noaa_slr_inundation_zones_layer3.shp",
}

# SLR scenario to use as prediction target (1-10 ft of sea level rise)
# 3 ft is Hawaii state recommended near-term planning scenario
SLR_TARGET_FEET = 3

# ─────────────────────────────────────────────────────────────
# 3. LOAD & VALIDATE DATA
# ─────────────────────────────────────────────────────────────
def load_and_validate():
    print("\n[1/6] Loading data layers...")

    # Define target CRS once at the top — used by all layers
    target_crs = CRS.from_epsg(32604)  # UTM Zone 4
  

    # --- Parcels ---
    parcels = gpd.read_file(PATHS["parcels"], on_invalid="ignore")
    parcels["geometry"] = parcels.geometry.make_valid()
    print(f"  Parcels loaded: {len(parcels):,} records, CRS={parcels.crs}")

    # Filter to Oahu only
    if "island" in parcels.columns:
        parcels = parcels[parcels["island"] == "OAH"].copy()
        print(f"  Oahu parcels (island=OAH): {len(parcels):,}")
    elif "county" in parcels.columns:
        parcels = parcels[parcels["county"] == "Honolulu"].copy()
        print(f"  Oahu parcels (county=Honolulu): {len(parcels):,}")

    # Reproject FIRST — must happen before centroid computation
    if parcels.crs != target_crs:
        parcels = parcels.to_crs(target_crs)
        print(f"  Parcels reprojected to UTM Zone 4N (EPSG:32604)")

    # Fix invalid geometries BEFORE computing centroids
    # make_valid() is safer than buffer(0) — never returns None
    parcels["geometry"] = parcels.geometry.make_valid()

    # Drop any rows where geometry is still null after repair
    before = len(parcels)
    parcels = parcels[parcels.geometry.notna() & ~parcels.geometry.is_empty].copy()
    dropped = before - len(parcels)
    if dropped > 0:
        print(f"  Dropped {dropped} parcels with unrecoverable geometry")

    # Compute centroids AFTER reprojection and geometry repair
    parcels["centroid"] = parcels.geometry.centroid
    parcels["centroid_lon"] = parcels["centroid"].x
    parcels["centroid_lat"] = parcels["centroid"].y
    print(f"  Centroid sample — x: {parcels['centroid_lon'].iloc[0]:.1f}m, y: {parcels['centroid_lat'].iloc[0]:.1f}m")

    # --- FEMA BFE ---
    bfe = gpd.read_file(PATHS["bfe"])
    print(f"  FEMA BFE loaded: {len(bfe):,} records")
    if "elev" in bfe.columns:
        bfe["elev_m"] = bfe["elev"] * 0.3048
        print(f"  BFE elevation range: {bfe['elev_m'].min():.1f} to {bfe['elev_m'].max():.1f} m")
    if bfe.crs != target_crs:
        bfe = bfe.to_crs(target_crs)

    # --- Water Lines ---
    water = gpd.read_file(PATHS["water_lines"])
    print(f"  Water lines loaded: {len(water):,} records")
    if water.crs != target_crs:
        water = water.to_crs(target_crs)

    # --- NOAA SLR Zones ---
    slr_frames = []
    for ft in range(0, 11):
        layer_name = f"hi_oahu_slr_{ft}ft"
        try:
            layer = gpd.read_file(PATHS["slr"], layer=layer_name)
            layer["gridcode"] = ft
            slr_frames.append(layer)
        except Exception:
            pass

    if slr_frames:
        slr = pd.concat(slr_frames, ignore_index=True)
        slr = gpd.GeoDataFrame(slr, geometry="geometry")
        print(f"  SLR native CRS: {slr.crs}")
        if slr.crs is None:
            slr = slr.set_crs(target_crs)
        elif slr.crs != target_crs:
            slr = slr.to_crs(target_crs)
        print(f"  SLR zones loaded: {len(slr):,} polygons, gridcodes: {sorted(slr['gridcode'].unique())}")
    else:
        print("  WARNING: No SLR layers found in .gdb — check PATHS['slr']")
        slr = None

    return parcels, bfe, water, slr


# ─────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def build_features(parcels, bfe, water, slr):
    print("\n[2/6] Extracting features...")

    gdf = parcels.copy()

    # ── Feature 1: Elevation from DEM (mean & min per parcel) ──────────
    print("  Sampling DEM elevation per parcel centroid...")
    with rasterio.open(PATHS["dem"]) as src:
        dem_crs = src.crs
        print(f"  DEM CRS: {dem_crs}, bounds: {src.bounds}")

        # Reproject parcel geometries to DEM's native CRS for accurate masking
        if gdf.crs != dem_crs:
            gdf_dem = gdf.to_crs(dem_crs)
        else:
            gdf_dem = gdf

        # Sample DEM at parcel centroids using DEM-CRS coordinates
        coords = list(zip(gdf_dem.geometry.centroid.x, gdf_dem.geometry.centroid.y))
        sampled = list(src.sample(coords))
        nodata_val = src.nodata
        gdf["elevation_m"] = [
            float(s[0]) if (nodata_val is None or s[0] != nodata_val) else np.nan
            for s in sampled
        ]

        print("  Computing zonal min elevation per parcel (may take a few minutes)...")
        first_exc = None
        elev_min, elev_mean = [], []
        for geom in gdf_dem.geometry:
            try:
                out, _ = _rasterio_mask(src, [geom], crop=True, all_touched=True)
                vals = out[0].astype(float)
                if nodata_val is not None:
                    vals[vals == nodata_val] = np.nan
                valid = vals[np.isfinite(vals)]
                if len(valid) == 0:
                    elev_min.append(np.nan)
                    elev_mean.append(np.nan)
                else:
                    elev_min.append(float(np.min(valid)))
                    elev_mean.append(float(np.mean(valid)))
            except Exception as e:
                if first_exc is None:
                    first_exc = e
                elev_min.append(np.nan)
                elev_mean.append(np.nan)
        if first_exc is not None:
            print(f"  WARNING: zonal mask exception (first): {first_exc}")
        gdf["elev_min_m"] = elev_min
        gdf["elev_mean_m"] = elev_mean

    print(f"  Elevation stats: min={gdf['elev_min_m'].min():.1f}m, max={gdf['elev_min_m'].max():.1f}m")

    # ── Feature 2: Distance to shoreline ──────────────────────────────
    # Approximate shoreline as boundary of land areas below 5m
    print("  Computing distance to coastline...")
    # Use water lines as proxy for coastal boundary
    # Reproject water to UTM to match parcels — distance now in true metres
    water_utm = water.to_crs(gdf.crs)
    water_union = water_utm.geometry.unary_union

    # Filter out null centroids before distance calculation
    valid_centroid_mask = gdf["centroid"].notna()
    gdf["dist_to_water_m"] = np.nan
    gdf.loc[valid_centroid_mask, "dist_to_water_m"] = gdf.loc[valid_centroid_mask, "centroid"].apply(
        lambda pt: pt.distance(water_union)  # UTM = metres, no conversion needed
                )

    # ── Feature 3: Nearest FEMA BFE elevation ─────────────────────────
    print("  Joining nearest FEMA Base Flood Elevation...")
    # For each parcel, find the nearest BFE line and its elevation
    bfe_union = bfe.copy()
    bfe_union["geometry_proj"] = bfe_union.geometry

    # Spatial join: nearest BFE line to each parcel centroid
    centroids_gdf = gpd.GeoDataFrame(
        gdf[["TMK", "centroid_lon", "centroid_lat"]],
        geometry=gpd.points_from_xy(gdf["centroid_lon"], gdf["centroid_lat"]),
        crs=gdf.crs
    )
    nearest_bfe = gpd.sjoin_nearest(
        centroids_gdf,
        bfe[["geometry", "elev_m"]],
        how="left",
        distance_col="dist_to_bfe_m"
    )
    gdf["nearest_bfe_elev_m"] = nearest_bfe["elev_m"].values
    gdf["dist_to_bfe_m"] = nearest_bfe["dist_to_bfe_m"].values  # already metres in UTM

    # ── Feature 4: FEMA flood zone (current risk classification) ──────
    # If you have DFIRM flood hazard areas shapefile (S_FLD_HAZ_AR), join here
    # gdf = gpd.sjoin(gdf, flood_zones[["geometry", "FLD_ZONE"]], how="left", predicate="within")
    # For now, use BFE elevation as proxy

    # ── Feature 5: Elevation relative to BFE ──────────────────────────
    gdf["elev_above_bfe_m"] = gdf["elev_min_m"] - gdf["nearest_bfe_elev_m"]

    # ── Feature 6: Terrain slope (derived from DEM) ───────────────────
    # Slope = elevation difference over distance (approximated from neighbors)
    # Full slope requires gdal/richdem; simplified proxy: use elev range within parcel
    # zonal_stats already gave us min and mean

    # ── Feature 7: Property value features ───────────────────────────
    # tmk_state.shp does not include LandValue/BldgValue
    # Use GISAcres as the available parcel size feature
    # land_value and bldg_value default to 0 — excluded from model automatically
    # via FEATURE_COLS filter since they carry no signal

    available_cols = gdf.columns.tolist()
    print(f"  Available property columns: {[c for c in available_cols if any(x in c.lower() for x in ['value','acre','area'])]}")

    gdf["land_value"] = pd.to_numeric(
        gdf["LandValue"] if "LandValue" in gdf.columns else 0,
        errors="coerce"
    ).fillna(0) if "LandValue" in gdf.columns else 0.0

    gdf["bldg_value"] = pd.to_numeric(
        gdf["BldgValue"] if "BldgValue" in gdf.columns else 0,
        errors="coerce"
    ).fillna(0) if "BldgValue" in gdf.columns else 0.0

    gdf["total_value"] = gdf["land_value"] + gdf["bldg_value"]

    gdf["parcel_acres"] = pd.to_numeric(
        gdf["GISAcres"] if "GISAcres" in gdf.columns else 0,
        errors="coerce"
    ).fillna(0)

    print(f"  Feature engineering complete. Shape: {gdf.shape}")
    return gdf


# ─────────────────────────────────────────────────────────────
# 5. CREATE TARGET LABELS FROM NOAA SLR ZONES
# ─────────────────────────────────────────────────────────────
def create_labels(gdf, slr, target_feet=SLR_TARGET_FEET):
    print(f"\n[3/6] Creating SLR labels (target = {target_feet} ft scenario)...")
    
    # ── Diagnostic — print all column names ──────────────────
    print(f"  Available columns: {gdf.columns.tolist()}")
    print(f"  TMK present: {'TMK' in gdf.columns}")
    
    print(f"\n[3/6] Creating SLR labels (target = {target_feet} ft scenario)...")

    if slr is None:
        print("  WARNING: SLR layer missing — labels will be elevation-threshold based")
        # Fallback: label parcels below target_feet * 0.3048 meters as at-risk
        threshold_m = target_feet * 0.3048
        gdf["slr_label"] = (gdf["elev_min_m"] <= threshold_m).astype(int)
        print(f"  Elevation threshold fallback: {threshold_m:.2f}m → {gdf['slr_label'].sum()} at-risk parcels")
        return gdf

    # Filter SLR zones to target scenario and below
    # gridcode 1 = 1ft, gridcode 3 = 3ft, etc.
    inundation_zones = slr[slr["gridcode"] <= target_feet].copy()
    inundated_union = inundation_zones.geometry.unary_union
    print(f"  Inundation zone polygons (gridcode ≤ {target_feet}): {len(inundation_zones):,}")

    # Label: 1 if parcel centroid falls within SLR inundation zone
    centroids_gdf = gpd.GeoDataFrame(
        gdf[["TMK"]],
        geometry=gpd.points_from_xy(gdf["centroid_lon"], gdf["centroid_lat"]),
        crs=gdf.crs
    )
    joined = gpd.sjoin(
        centroids_gdf,
        inundation_zones[["geometry", "gridcode"]],
        how="left",
        predicate="within"
    )
    # Mark as inundated if any match
    inundated_tmks = set(joined.dropna(subset=["gridcode"])["TMK"])
    gdf["slr_label"] = gdf["TMK"].isin(inundated_tmks).astype(int)

    pos = gdf["slr_label"].sum()
    neg = len(gdf) - pos
    print(f"  Labels: {pos:,} at-risk ({pos/len(gdf)*100:.1f}%), {neg:,} not at-risk")

    # Also store which SLR scenario first inundates each parcel
    for ft in range(1, 11):
        zone = slr[slr["gridcode"] <= ft]
        if len(zone) == 0:
            continue
        zone_union = zone.geometry.unary_union
        tmks_at_ft = set(
            gpd.sjoin(
                centroids_gdf,
                zone[["geometry", "gridcode"]],
                how="inner",
                predicate="within"
            )["TMK"]
        )
        col = f"slr_{ft}ft"
        gdf[col] = gdf["TMK"].isin(tmks_at_ft).astype(int)

    return gdf


# ─────────────────────────────────────────────────────────────
# 6. TRAIN MODELS & EVALUATE
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "elevation_m",
    "elev_min_m",
    "elev_mean_m",
    "dist_to_water_m",
    "nearest_bfe_elev_m",
    "dist_to_bfe_m",
    "elev_above_bfe_m",
    "centroid_lat",
    "centroid_lon",
    "parcel_acres",
    # land_value and bldg_value removed — not in tmk_state.shp
    # add back if you source a property value dataset later
]

def train_and_evaluate(gdf):
    print("\n[4/6] Training models...")

    # Prepare feature matrix
    feature_cols = [c for c in FEATURE_COLS if c in gdf.columns]
    X = gdf[feature_cols].copy()
    y = gdf["slr_label"].copy()

    # Drop rows with missing features
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"  Training set: {len(X):,} parcels, {y.mean()*100:.1f}% positive class")
    print(f"  Features used: {feature_cols}")

    # Handle class imbalance
    pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"  Class imbalance ratio: {pos_weight:.1f}:1 (neg:pos)")

    # ── Model 1: Logistic Regression (baseline) ──────────────────────
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])

    # ── Model 2: Random Forest ────────────────────────────────────────
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # ── Model 3: XGBoost ─────────────────────────────────────────────
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )

    models = {
        "Logistic Regression": lr_pipeline,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
    }

    results = {}
    for name, model in models.items():
        scores = cross_val_score(
            model, X, y,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1
        )
        results[name] = {
            "model": model,
            "auc_mean": scores.mean(),
            "auc_std": scores.std()
        }
        print(f"  {name}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    # Select best model
    best_name = max(results, key=lambda k: results[k]["auc_mean"])
    best_model = results[best_name]["model"]
    print(f"\n  Best model: {best_name} (AUC={results[best_name]['auc_mean']:.4f})")

    # Fit best model on full data
    best_model.fit(X, y)

    # Predict risk probabilities for all parcels
    valid_idx = gdf.index[valid_mask]
    gdf.loc[valid_idx, "risk_score"] = best_model.predict_proba(X)[:, 1]
    gdf.loc[valid_idx, "risk_predicted"] = best_model.predict(X)

    # ── Risk tiers ────────────────────────────────────────────────────
    gdf["risk_tier"] = pd.cut(
        gdf["risk_score"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["Very Low", "Low", "Moderate", "High", "Very High"]
    )

    print(f"\n  Risk tier distribution:")
    print(gdf["risk_tier"].value_counts().sort_index().to_string())

    return gdf, best_model, best_name, feature_cols, X, y


# ─────────────────────────────────────────────────────────────
# 7. VISUALIZE RESULTS
# ─────────────────────────────────────────────────────────────
def visualize(gdf, model, model_name, feature_cols, X, y):
    print("\n[5/6] Generating visualizations...")
    os.makedirs("outputs", exist_ok=True)

    # ── Plot 1: Feature Importance ────────────────────────────────────
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        # Pipeline — extract inner model
        inner = model.named_steps["model"]
        if hasattr(inner, "coef_"):
            importances = np.abs(inner.coef_[0])
        else:
            importances = inner.feature_importances_
    else:
        importances = None

    if importances is not None:
        fig, ax = plt.subplots(figsize=(9, 5))
        sorted_idx = np.argsort(importances)
        bars = ax.barh(
            [feature_cols[i] for i in sorted_idx],
            importances[sorted_idx],
            color="#1a6b8a"
        )
        ax.set_title(f"Feature Importance — {model_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig("outputs/feature_importance.png", dpi=150)
        plt.close()
        print("  Saved: outputs/feature_importance.png")

    # ── Plot 2: Risk score distribution ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    gdf["risk_score"].dropna().hist(bins=50, ax=axes[0], color="#1a6b8a", edgecolor="white")
    axes[0].set_title("Distribution of Risk Scores")
    axes[0].set_xlabel("Predicted Risk Probability")

    tier_counts = gdf["risk_tier"].value_counts().sort_index()
    colors = ["#2ecc71", "#a8e063", "#f39c12", "#e67e22", "#e74c3c"]
    tier_counts.plot(kind="bar", ax=axes[1], color=colors, edgecolor="white")
    axes[1].set_title("Parcels by Risk Tier")
    axes[1].set_ylabel("Number of Parcels")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig("outputs/risk_distribution.png", dpi=150)
    plt.close()
    print("  Saved: outputs/risk_distribution.png")

    # ── Plot 3: Elevation vs Risk Score ──────────────────────────────
    plot_df = gdf[["elev_min_m", "risk_score", "slr_label"]].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(
        plot_df["elev_min_m"],
        plot_df["risk_score"],
        c=plot_df["slr_label"],
        cmap="RdYlGn_r",
        alpha=0.3,
        s=5
    )
    ax.set_xlabel("Minimum Parcel Elevation (m)")
    ax.set_ylabel("Predicted SLR Risk Score")
    ax.set_title("Elevation vs SLR Risk Score\n(red=inundated by target scenario)")
    plt.colorbar(scatter, ax=ax, label="Actual SLR Label")
    plt.tight_layout()
    plt.savefig("outputs/elevation_vs_risk.png", dpi=150)
    plt.close()
    print("  Saved: outputs/elevation_vs_risk.png")

    # ── Map: Interactive Folium risk map ──────────────────────────────
    print("  Building interactive map (this may take a moment)...")
    map_data = gdf[["centroid_lat", "centroid_lon", "risk_score", "TMK",
                     "risk_tier", "elev_min_m", "total_value"]].dropna(subset=["risk_score"])

    # Sample for performance if very large
    if len(map_data) > 20000:
        map_data = map_data.sample(20000, random_state=42)

    m = folium.Map(
        location=[21.45, -157.97],
        zoom_start=11,
        tiles="CartoDB positron"
    )

    color_map = {
        "Very Low": "#2ecc71",
        "Low": "#a8e063",
        "Moderate": "#f39c12",
        "High": "#e67e22",
        "Very High": "#e74c3c"
    }

    for _, row in map_data.iterrows():
        color = color_map.get(str(row["risk_tier"]), "#aaaaaa")
        folium.CircleMarker(
            location=[row["centroid_lat"], row["centroid_lon"]],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>TMK:</b> {row['TMK']}<br>"
                f"<b>Risk Score:</b> {row['risk_score']:.3f}<br>"
                f"<b>Risk Tier:</b> {row['risk_tier']}<br>"
                f"<b>Min Elevation:</b> {row['elev_min_m']:.1f}m<br>"
                f"<b>Total Value:</b> ${row['total_value']:,.0f}",
                max_width=250
            )
        ).add_to(m)

    m.save("outputs/oahu_slr_risk_map.html")
    print("  Saved: outputs/oahu_slr_risk_map.html")


# ─────────────────────────────────────────────────────────────
# 8. EXPORT RESULTS
# ─────────────────────────────────────────────────────────────
def export_results(gdf):
    print("\n[6/6] Exporting results...")
    os.makedirs("outputs", exist_ok=True)

    output_cols = [
        "TMK", "TMK_txt", "centroid_lat", "centroid_lon",
        "elevation_m", "elev_min_m", "elev_mean_m",
        "dist_to_water_m", "nearest_bfe_elev_m", "elev_above_bfe_m",
        "land_value", "bldg_value", "total_value", "parcel_acres",
        "slr_label", "risk_score", "risk_predicted", "risk_tier",
    ] + [c for c in gdf.columns if c.startswith("slr_") and "ft" in c]

    export_cols = [c for c in output_cols if c in gdf.columns]

    # CSV for analysis
    gdf[export_cols].to_csv("outputs/oahu_slr_risk_scores.csv", index=False)
    print(f"  Saved: outputs/oahu_slr_risk_scores.csv ({len(gdf):,} rows)")

    # GeoJSON for GIS / web mapping
    geo_cols = export_cols + ["geometry"]
    geo_cols = [c for c in geo_cols if c in gdf.columns]
    gdf[geo_cols].to_file("outputs/oahu_slr_risk_scores.geojson", driver="GeoJSON")
    print("  Saved: outputs/oahu_slr_risk_scores.geojson")

    # Summary stats
    print("\n  ── Model Summary ──────────────────────────────────────")
    print(f"  Total parcels scored:  {gdf['risk_score'].notna().sum():,}")
    print(f"  High/Very High risk:   {gdf['risk_tier'].isin(['High','Very High']).sum():,}")
    high_risk_value = gdf[gdf["risk_tier"].isin(["High", "Very High"])]["total_value"].sum()
    print(f"  At-risk property value: ${high_risk_value:,.0f}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Hawaii SLR Predictive Model Pipeline")
    print("=" * 60)

    # Step 1: Load data
    parcels, bfe, water, slr = load_and_validate()

    # Step 2: Build feature matrix
    gdf = build_features(parcels, bfe, water, slr)

    # Step 3: Create labels
    gdf = create_labels(gdf, slr, target_feet=SLR_TARGET_FEET)

    # Step 4: Train & evaluate models
    gdf, best_model, best_name, feature_cols, X, y = train_and_evaluate(gdf)

    # Step 5: Visualize
    visualize(gdf, best_model, best_name, feature_cols, X, y)

    # Step 6: Export
    export_results(gdf)

    print("\n✓ Pipeline complete. Check the outputs/ folder.")
