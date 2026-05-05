"""
GDB Folder Audit Script
=======================
Run this locally against your full .gdb folder to identify every layer,
its contents, schema, and how it fits into the SLR pipeline.

Usage:
    python audit_gdb_folder.py --gdb "path/to/your_folder.gdb"

Output:
    gdb_audit_report.txt  — full human-readable inventory
    gdb_layer_catalog.csv — machine-readable layer catalog
"""

import os
import re
import sys
import struct
import argparse
import csv
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# HELPERS: binary string extraction
# ─────────────────────────────────────────────────────────────

def extract_ascii(data, min_len=4, scan_limit=20000):
    strings, current = [], []
    for b in data[:scan_limit]:
        if 32 <= b <= 126:
            current.append(chr(b))
        else:
            if len(current) >= min_len:
                strings.append(''.join(current))
            current = []
    if len(current) >= min_len:
        strings.append(''.join(current))
    return strings


def extract_utf16(data, min_len=4, scan_limit=20000):
    strings, i = [], 0
    while i < min(len(data) - 1, scan_limit):
        if data[i + 1] == 0 and 32 <= data[i] <= 126:
            chars, j = [], i
            while j < len(data) - 1 and data[j + 1] == 0 and 32 <= data[j] <= 126:
                chars.append(chr(data[j]))
                j += 2
            if len(chars) >= min_len:
                strings.append(''.join(chars))
            i = j
        else:
            i += 1
    return strings


def dedupe_strings(string_list, min_len=4):
    seen, result = set(), []
    for s in string_list:
        s = s.strip()
        if s and s not in seen and len(s) >= min_len and not all(c == s[0] for c in s):
            seen.add(s)
            result.append(s)
    return result


# ─────────────────────────────────────────────────────────────
# LAYER CLASSIFICATION
# ─────────────────────────────────────────────────────────────

# Known NOAA SLR layer patterns and their pipeline roles
LAYER_CLASSIFICATIONS = {
    # NOAA SLR inundation zones (ocean-connected flooding) — PRIMARY LABELS
    "hi_oahu_slr_0ft":   ("SLR Inundation",  "label_0ft",   "Current tidal inundation baseline"),
    "hi_oahu_slr_1ft":   ("SLR Inundation",  "label_1ft",   "1 ft SLR — near-term ~2030s"),
    "hi_oahu_slr_2ft":   ("SLR Inundation",  "label_2ft",   "2 ft SLR"),
    "hi_oahu_slr_3ft":   ("SLR Inundation",  "label_3ft",   "3 ft SLR — HI state planning standard ★"),
    "hi_oahu_slr_4ft":   ("SLR Inundation",  "label_4ft",   "4 ft SLR — HI state min design standard"),
    "hi_oahu_slr_5ft":   ("SLR Inundation",  "label_5ft",   "5 ft SLR"),
    "hi_oahu_slr_6ft":   ("SLR Inundation",  "label_6ft",   "6 ft SLR — mid-century high scenario"),
    "hi_oahu_slr_7ft":   ("SLR Inundation",  "label_7ft",   "7 ft SLR"),
    "hi_oahu_slr_8ft":   ("SLR Inundation",  "label_8ft",   "8 ft SLR"),
    "hi_oahu_slr_9ft":   ("SLR Inundation",  "label_9ft",   "9 ft SLR"),
    "hi_oahu_slr_10ft":  ("SLR Inundation",  "label_10ft",  "10 ft SLR — long-term worst case"),

    # Low-lying land areas (topographic, not hydrologically connected) — FEATURES
    "hi_oahu_low_0ft":   ("Low-Lying Land",  "feature_low", "Land at/below current MHHW"),
    "hi_oahu_low_1ft":   ("Low-Lying Land",  "feature_low", "Land below 1 ft above MHHW"),
    "hi_oahu_low_2ft":   ("Low-Lying Land",  "feature_low", "Land below 2 ft above MHHW"),
    "hi_oahu_low_3ft":   ("Low-Lying Land",  "feature_low", "Land below 3 ft above MHHW ★ alt label"),
    "hi_oahu_low_4ft":   ("Low-Lying Land",  "feature_low", "Land below 4 ft above MHHW"),
    "hi_oahu_low_5ft":   ("Low-Lying Land",  "feature_low", "Land below 5 ft above MHHW"),
    "hi_oahu_low_6ft":   ("Low-Lying Land",  "feature_low", "Land below 6 ft above MHHW"),
    "hi_oahu_low_7ft":   ("Low-Lying Land",  "feature_low", "Land below 7 ft above MHHW"),
    "hi_oahu_low_8ft":   ("Low-Lying Land",  "feature_low", "Land below 8 ft above MHHW"),
    "hi_oahu_low_9ft":   ("Low-Lying Land",  "feature_low", "Land below 9 ft above MHHW"),
    "hi_oahu_low_10ft":  ("Low-Lying Land",  "feature_low", "Land below 10 ft above MHHW"),

    # GDB system tables — skip
    "gdb_items":                ("System",  "skip", "ESRI GDB internal catalog"),
    "gdb_itemtypes":            ("System",  "skip", "ESRI GDB type registry"),
    "gdb_itemrelationships":    ("System",  "skip", "ESRI GDB relationship index"),
    "gdb_itemrelationshiptypes":("System",  "skip", "ESRI GDB relationship types"),
    "gdb_spatialrefs":          ("System",  "skip", "Spatial reference registry"),
    "gdb_systemcatalog":        ("System",  "skip", "GDB system catalog"),
    "gdb_dbtune":               ("System",  "skip", "GDB tuning parameters"),
    "gdb_replicalog":           ("System",  "skip", "Replication log"),
}

# File extension roles
EXT_ROLES = {
    ".gdbtable":   "Data table (records + geometry)",
    ".gdbtablx":   "Row offset index (maps row# → byte offset)",
    ".gdbindexes": "Field index definitions",
    ".atx":        "Attribute B-tree index",
    ".spx":        "Spatial index (R-tree)",
    ".freelist":   "Deleted row recycling list",
    ".horizon":    "Transaction horizon file",
}


# ─────────────────────────────────────────────────────────────
# GDBTABLE INSPECTION
# ─────────────────────────────────────────────────────────────

def inspect_gdbtable(path):
    """Extract record count, field names, CRS hint from a .gdbtable file."""
    info = {"record_count": None, "fields": [], "crs_hint": None, "layer_name": None}
    try:
        with open(path, "rb") as f:
            data = f.read()

        # Record count at byte offset 4
        if len(data) >= 8:
            info["record_count"] = struct.unpack_from("<I", data, 4)[0]

        # Extract strings to find field names and CRS
        all_strings = dedupe_strings(
            extract_ascii(data, min_len=4) + extract_utf16(data, min_len=4)
        )

        known_fields = ["OBJECTID", "Shape", "gridcode", "Shape_Length", "Shape_Area",
                        "FLD_ZONE", "BFE_LN_ID", "ELEV", "WTR_NM", "TMK", "LandValue",
                        "ZONE_", "SFHA_TF", "STUDY_TYP", "FIRM_PAN"]
        for s in all_strings:
            if s in known_fields:
                info["fields"].append(s)
            if "GCS_North_American" in s or "NAD83" in s or "NAD_1983" in s:
                info["crs_hint"] = "NAD83"
            if "WGS_1984" in s or "WGS84" in s:
                info["crs_hint"] = "WGS84"

    except Exception as e:
        info["error"] = str(e)
    return info


def inspect_atx(path):
    """Extract layer names from a TablesByName ATX index."""
    names = []
    try:
        with open(path, "rb") as f:
            data = f.read()
        all_strings = dedupe_strings(
            extract_ascii(data, min_len=5) + extract_utf16(data, min_len=5),
            min_len=5
        )
        for s in all_strings:
            # GDB ATX pads layer names to a fixed width with spaces;
            # split on 3+ consecutive spaces to recover individual names.
            parts = re.split(r' {3,}', s)
            for part in parts:
                clean = part.strip("_").strip().lower().replace(" ", "_")
                if not clean:
                    continue
                if any(clean.startswith(p) for p in ["hi_", "gdb_", "slr_", "low_"]):
                    names.append(clean)
                elif "_" in clean and len(clean) > 6 and clean.islower():
                    names.append(clean)
    except Exception:
        pass
    return names


# ─────────────────────────────────────────────────────────────
# MAIN AUDIT
# ─────────────────────────────────────────────────────────────

def audit_gdb(gdb_path):
    gdb_path = Path(gdb_path)
    if not gdb_path.exists():
        print(f"ERROR: Path not found: {gdb_path}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  GDB FOLDER AUDIT")
    print(f"  Path: {gdb_path}")
    print(f"  Run:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    all_files = sorted(gdb_path.iterdir())
    print(f"Total files in folder: {len(all_files)}\n")

    # ── Step 1: Catalog all files by stem ────────────────────────────
    stems = {}
    for f in all_files:
        stem = f.stem
        if stem not in stems:
            stems[stem] = []
        stems[stem].append(f)

    print(f"Unique table stems: {len(stems)}\n")

    # ── Step 2: Find layer name mapping from ATX indexes ─────────────
    layer_names_found = []
    print("── Scanning ATX indexes for layer names ──────────────────────")
    for f in all_files:
        if "TablesByName" in f.name or "ItemsByName" in f.name:
            names = inspect_atx(f)
            if names:
                layer_names_found.extend(names)
                print(f"  {f.name}: found {len(names)} names")
                for n in names:
                    print(f"    → {n}")

    layer_names_found = list(dict.fromkeys(layer_names_found))

    # ── Step 3: Classify each layer ───────────────────────────────────
    print(f"\n── Layer Classification ──────────────────────────────────────")

    data_layers = []     # slr_*, low_* 
    system_tables = []   # gdb_*
    unknown_tables = []  # anything else

    for name in layer_names_found:
        name_clean = name.strip().lower()
        if name_clean in LAYER_CLASSIFICATIONS:
            cat, role, desc = LAYER_CLASSIFICATIONS[name_clean]
        elif name_clean.startswith("gdb_"):
            cat, role, desc = "System", "skip", "ESRI internal"
        elif "slr" in name_clean:
            cat, role, desc = "SLR Inundation", "label_unknown", "SLR scenario — review gridcode"
        elif "low" in name_clean:
            cat, role, desc = "Low-Lying Land", "feature_low", "Low-lying land — use as feature"
        else:
            cat, role, desc = "Unknown", "review", "Manual review needed"

        entry = {"name": name_clean, "category": cat, "pipeline_role": role, "description": desc}

        if cat == "System":
            system_tables.append(entry)
        elif cat in ("SLR Inundation", "Low-Lying Land"):
            data_layers.append(entry)
        else:
            unknown_tables.append(entry)

    # ── Step 4: Inspect .gdbtable files ───────────────────────────────
    print(f"\n── Inspecting .gdbtable files ────────────────────────────────")
    table_details = {}
    gdbtables = [f for f in all_files if f.suffix == ".gdbtable"]
    print(f"  Found {len(gdbtables)} .gdbtable files\n")

    for f in sorted(gdbtables):
        info = inspect_gdbtable(f)
        table_details[f.stem] = info
        rec = info.get("record_count", "?")
        fields = ", ".join(info.get("fields", [])) or "—"
        crs = info.get("crs_hint", "?")
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}")
        print(f"    Size: {size_kb:,.0f} KB | Records: {rec:,} | CRS: {crs}")
        print(f"    Fields: {fields}")

    # ── Step 5: File extension inventory ──────────────────────────────
    print(f"\n── File Extension Summary ────────────────────────────────────")
    ext_counts = {}
    ext_sizes = {}
    for f in all_files:
        ext = f.suffix
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
        ext_sizes[ext]  = ext_sizes.get(ext, 0) + f.stat().st_size

    for ext, count in sorted(ext_counts.items()):
        role = EXT_ROLES.get(ext, "Unknown role")
        size_mb = ext_sizes[ext] / 1024 / 1024
        print(f"  {ext:<16} {count:>3} files   {size_mb:>8.1f} MB   {role}")

    total_mb = sum(f.stat().st_size for f in all_files) / 1024 / 1024
    print(f"\n  TOTAL: {len(all_files)} files, {total_mb:.1f} MB")

    # ── Step 6: Pipeline recommendations ──────────────────────────────
    slr_layers   = [d for d in data_layers if d["category"] == "SLR Inundation"]
    low_layers   = [d for d in data_layers if d["category"] == "Low-Lying Land"]

    print(f"\n{'='*70}")
    print(f"  PIPELINE RECOMMENDATIONS")
    print(f"{'='*70}")

    print(f"\n  ► PRIMARY LABEL LAYERS ({len(slr_layers)} found)")
    print(f"    Use hi_oahu_slr_Xft layers as your model target variable.")
    print(f"    Recommended: encode as ordinal (first_inundated_at_ft)")
    for d in sorted(slr_layers, key=lambda x: x["name"]):
        marker = " ★" if "3ft" in d["name"] or "4ft" in d["name"] else ""
        print(f"    • {d['name']:<25} {d['description']}{marker}")

    print(f"\n  ► FEATURE LAYERS ({len(low_layers)} found)")
    print(f"    Use hi_oahu_low_Xft to create binary features per parcel:")
    print(f"    e.g. parcel_below_1ft, parcel_below_3ft, parcel_below_5ft")
    for d in sorted(low_layers, key=lambda x: x["name"]):
        print(f"    • {d['name']:<25} {d['description']}")

    print(f"\n  ► SYSTEM TABLES ({len(system_tables)} found) — skip in pipeline")

    if unknown_tables:
        print(f"\n  ► UNKNOWN LAYERS ({len(unknown_tables)}) — review manually")
        for d in unknown_tables:
            print(f"    • {d['name']}")

    print(f"\n  ► RECOMMENDED MODEL TARGET VARIABLE")
    print(f"    Instead of binary 0/1, encode as:")
    print(f"    'first_slr_ft_inundated' = min(gridcode) across all slr layers")
    print(f"    This enables ordinal regression or survival analysis.")
    print(f"    Parcels never inundated by 10ft → label = 11 (censored)")

    print(f"\n  ► MULTI-LABEL APPROACH (best for portfolio)")
    print(f"    Train separate binary classifiers per threshold:")
    print(f"    slr_1ft_risk, slr_3ft_risk, slr_6ft_risk, slr_10ft_risk")
    print(f"    Visualize as a risk timeline per parcel.\n")

    # ── Step 7: Write CSV catalog ──────────────────────────────────────
    csv_path = "gdb_layer_catalog.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "category", "pipeline_role", "description"])
        writer.writeheader()
        for row in sorted(data_layers + system_tables + unknown_tables, key=lambda x: x["name"]):
            writer.writerow(row)
    print(f"  Saved: {csv_path}")

    # ── Step 8: Print the geopandas load snippet ───────────────────────
    print(f"\n{'='*70}")
    print(f"  LOADING LAYERS IN YOUR PIPELINE (copy-paste ready)")
    print(f"{'='*70}\n")
    print(f"  import geopandas as gpd\n")
    print(f"  GDB = \"{gdb_path}\"  # update this path\n")

    for ft in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        slr_name = f"hi_oahu_slr_{ft}ft"
        low_name = f"hi_oahu_low_{ft}ft"
        if any(d["name"] == slr_name for d in slr_layers):
            print(f"  slr_{ft}ft = gpd.read_file(GDB, layer=\"{slr_name}\")")
        if any(d["name"] == low_name for d in low_layers):
            print(f"  low_{ft}ft = gpd.read_file(GDB, layer=\"{low_name}\")")
        if ft in [1, 3, 6, 10]:
            print()

    print(f"\n  # Assign 'first inundated at' label to parcels:")
    print(f"  for ft in range(0, 11):")
    print(f"      zone = gpd.read_file(GDB, layer=f\"hi_oahu_slr_{{ft}}ft\")")
    print(f"      joined = gpd.sjoin(parcels, zone[[\"geometry\"]], how=\"left\", predicate=\"within\")")
    print(f"      hit_tmks = set(joined.dropna(subset=[\"index_right\"])[\"TMK\"])")
    print(f"      mask = parcels[\"TMK\"].isin(hit_tmks) & parcels[\"first_slr_ft\"].isna()")
    print(f"      parcels.loc[mask, \"first_slr_ft\"] = ft")
    print(f"  parcels[\"first_slr_ft\"].fillna(11, inplace=True)  # 11 = never inundated\n")

    print("  Audit complete.\n")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit a FileGDB folder for SLR pipeline use")
    parser.add_argument("--gdb", required=True, help="c:/Users/natha/OneDrive/Desktop/Nathan/Datacamp Python/Hawaii Model/HI_Oahu_slr_data_dist.zip")
    args = parser.parse_args()
    audit_gdb(args.gdb)
