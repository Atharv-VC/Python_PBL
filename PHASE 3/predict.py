#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
 ║          🌾  CROP CLASSIFICATION SYSTEM  🌾                     ║
║   ICRISAT District-Level Agricultural Data — Prediction Script   ║
╚══════════════════════════════════════════════════════════════════╝

Mode 1: Input environmental features → Predict the most suitable crop
Mode 2: Input a crop name → Get ideal growing conditions

Usage:2
    python predict.py
"""

import pickle
import numpy as np
import pandas as pd
import os
import sys

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')

# Preferred model for predictions
BEST_MODEL = 'random_forest'  # highest non-trivial generalization

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
 ║              🌾  CROP CLASSIFICATION SYSTEM  🌾                 ║
║         Indian Agricultural Data — ICRISAT DLD Dataset           ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────
def load_artifacts():
    """Load all saved preprocessors and models."""
    artifacts = {}
    files = {
    'scaler':      'scaler.pkl',
    'pca':         'pca.pkl',          
    'kmeans':      'kmeans.pkl',       
    'le_crop':     'label_encoder_crop.pkl',
    'le_state':    'label_encoder_state.pkl',
    'le_district': 'label_encoder_district.pkl',
    'feature_cols':'feature_cols.pkl',
    'rf':          'random_forest.pkl',
    'lr':          'logistic.pkl',
    'gb':          'xgboost.pkl',
    'svm':         'svm.pkl',
    'knn':         'knn.pkl',
    'ensemble':    'ensemble.pkl',
    }
    for key, fname in files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                artifacts[key] = pickle.load(f)
        else:
            artifacts[key] = None
    return artifacts


def load_raw_data():
    #Load dirty dataset for crop insight queries.
    path = os.path.join(DATA_DIR, 'dirty_dataset.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def separator(char='─', width=66):
    print(char * width)


def print_header(title):
    separator('═')
    print(f"  {title}")
    separator('═')


def get_float(prompt, lo=None, hi=None, default=None):
    """Prompt user for a float with optional range validation."""
    while True:
        raw = input(f"  {prompt}").strip()
        if raw == '' and default is not None:
            return float(default)
        try:
            val = float(raw)
            if lo is not None and val < lo:
                print(f"    ⚠  Value must be ≥ {lo}. Try again.")
                continue
            if hi is not None and val > hi:
                print(f"    ⚠  Value must be ≤ {hi}. Try again.")
                continue
            return val
        except ValueError:
            print("    ⚠  Please enter a valid number.")


def choose_from_list(prompt, options):
    """Present numbered list and return chosen item."""
    print(f"\n  {prompt}")
    for i, opt in enumerate(options, 1):
        print(f"    [{i}] {opt}")
    while True:
        raw = input("  Your choice (number): ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print("    ⚠  Invalid choice. Try again.")


# ─────────────────────────────────────────────────────────────────
# DERIVED FEATURES (must match feature engineering pipeline)
# ─────────────────────────────────────────────────────────────────
def build_derived(feat):
    """Compute derived features from raw inputs."""
    feat['NPK_Balance']       = feat['N_kg_per_ha'] + feat['P_kg_per_ha'] + feat['K_kg_per_ha']
    feat['Water_Stress_Index']= feat['Rainfall_mm'] / (max(feat['Irrigation_1000ha'], 0.01) + 1)
    feat['Fert_Efficiency']   = feat['Yield_kg_per_ha'] / (max(feat['Fertilizer_kg_per_ha'], 0.01) + 1)
    hum  = min(max(feat['Humidity_pct'], 0), 100) / 100
    temp = feat['Temperature_C']
    feat['THI'] = temp + 0.33 * (hum * 6.105 * np.exp(17.27 * temp / (237.7 + temp))) - 4
    feat['Year_norm'] = (feat.get('Year', 2000) - 1966) / (2017 - 1966)
    return feat


# ─────────────────────────────────────────────────────────────────
# MODE 1 — Feature Input → Predict Crop
# ─────────────────────────────────────────────────────────────────
def mode_predict_crop(artifacts):
    print_header("MODE 1 — Predict Crop from Environmental Features")
    print("\n  Enter the agricultural/environmental features below.")
    print("  Press [Enter] to use the default value shown in brackets.\n")
    separator()

    # ── Collect inputs ────────────────────────────────────────────
    feat = {}
    feat['Rainfall_mm']         = get_float("Rainfall (mm)                   [default=825]  : ", 0, 5000,  825)
    feat['Irrigation_1000ha']   = get_float("Irrigation area (1000 ha)       [default=3.0]  : ", 0, 1000,   3.0)
    feat['Fertilizer_kg_per_ha']= get_float("Fertilizer usage (kg/ha)        [default=75]   : ", 0, 2000,   75)
    feat['Yield_kg_per_ha']     = get_float("Expected/historical yield (kg/ha)[default=1200] : ", 0, 50000, 1200)
    feat['Temperature_C']       = get_float("Temperature (°C)                [default=24]   : ", 5,  50,    24)
    feat['Humidity_pct']        = get_float("Humidity (%)                    [default=65]   : ", 0, 100,    65)
    feat['Soil_Moisture_pct']   = get_float("Soil moisture (%)               [default=30]   : ", 0, 100,    30)
    feat['Pesticide_kg_per_ha'] = get_float("Pesticide usage (kg/ha)         [default=2.5]  : ", 0, 500,   2.5)
    feat['Groundwater_Level_m'] = get_float("Groundwater level (m)           [default=15]   : ", 0, 200,    15)
    feat['Mechanization_Score'] = get_float("Mechanization score (0–100)     [default=50]   : ", 0, 100,    50)
    feat['N_kg_per_ha']         = get_float("Nitrogen requirement (kg/ha)    [default=25]   : ", 0, 500,    25)
    feat['P_kg_per_ha']         = get_float("Phosphorus requirement (kg/ha)  [default=10]   : ", 0, 500,    10)
    feat['K_kg_per_ha']         = get_float("Potassium requirement (kg/ha)   [default=12]   : ", 0, 500,    12)
    feat['pH']                  = get_float("Soil pH                         [default=6.5]  : ", 3,   10,  6.5)
    feat['Altitude_m']          = get_float("Altitude (meters)               [default=200]  : ", 0, 5000,  200)
    feat['Wind_Speed_m_s']      = get_float("Wind speed (m/s)                [default=2.5]  : ", 0,   30,  2.5)

    # State / District (for encoding)
    le_state    = artifacts['le_state']
    le_district = artifacts['le_district']
    known_states = list(le_state.classes_)
    state = choose_from_list("Select State:", known_states)

    known_districts = list(le_district.classes_)
    # Show only relevant districts (first 20 alphabetically, manageable)
    print(f"\n  District (type name or press Enter for 'Durg'):")
    print(f"  Available: {', '.join(known_districts[:30])} ...")
    raw_district = input("  District: ").strip()
    if raw_district == '' or raw_district not in known_districts:
        raw_district = known_districts[0]
        print(f"  → Using '{raw_district}'")

    feat['State_encoded']    = int(le_state.transform([state])[0])
    feat['District_encoded'] = int(le_district.transform([raw_district])[0])

    # Build derived features
    feat = build_derived(feat)

    # ── Build feature vector ──────────────────────────────────────
    feature_cols = artifacts['feature_cols']
    X = np.array([[feat.get(c, 0) for c in feature_cols]], dtype=float)

    # Scale
    X_scaled = artifacts['scaler'].transform(X) 
    # Apply PCA (same as training) 
    if artifacts['pca'] is not None: 
        X_pca = artifacts['pca'].transform(X_scaled) 
    else: 
        X_pca = None

    # ── Choose model ──────────────────────────────────────────────
    separator()
    print("\n  Choose prediction model:")
    model_map = {
        '1': ('Random Forest (recommended)', artifacts['rf']),
        '2': ('Gradient Boosting',           artifacts['gb']),
        '3': ('Ensemble (Voting)',            artifacts['ensemble']),
        '4': ('SVM',                          artifacts['svm']),
        '5': ('KNN',                          artifacts['knn']),
        '6': ('Logistic Regression',          artifacts['lr']),
    }
    for k, (name, _) in model_map.items():
        print(f"    [{k}] {name}")
    choice = input("  Model choice [1]: ").strip() or '1'
    if choice not in model_map:
        choice = '1'
    model_name, model = model_map[choice]

    # ── Predict ───────────────────────────────────────────────────
    separator()
    pred_encoded = model.predict(X_scaled)[0]
    predicted_crop = artifacts['le_crop'].inverse_transform([pred_encoded])[0]
    # KMeans cluster prediction 
    cluster_info = "" 
    if artifacts['kmeans'] is not None and X_pca is not None: 
        cluster = artifacts['kmeans'].predict(X_pca)[0] 
        cluster_info = f"\n 📊 Cluster Group: {cluster}"

    # Probability (if model supports it)
    proba_str = ""
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X_scaled)[0]
        top3_idx  = np.argsort(probas)[::-1][:3]
        top3_crops = artifacts['le_crop'].inverse_transform(top3_idx)
        top3_probs = probas[top3_idx]
        proba_lines = "\n".join([f"    {c:<16}: {p*100:.1f}%" for c, p in zip(top3_crops, top3_probs)])
        proba_str = f"\n  Top-3 Predictions:\n{proba_lines}"

    print(f"""
  ╔═════════════════════════════════════════╗
  ║  🌱 PREDICTED CROP: {predicted_crop:<20} ║
  ╚═════════════════════════════════════════╝
  Model used : {model_name}{proba_str}
    {cluster_info}
    """)

    # ── Provide brief growing tips ────────────────────────────────
    tips = {
        'Rice':         "Requires flooded fields, warm/humid climate, heavy rainfall or irrigation.",
        'Wheat':        "Cool dry climate, well-drained loamy soil. Rabi (winter) crop.",
        'Maize':        "Versatile crop; needs moderate rainfall, warm days, well-drained soil.",
        'Sorghum':      "Drought-tolerant; suited for semi-arid tropics, black/red soils.",
        'Pearl Millet': "Extremely drought-resilient; sandy soils, high heat tolerance.",
        'Groundnut':    "Well-drained sandy loam, 500–1000mm rainfall, long warm growing season.",
        'Cotton':       "Black (regur) soil, warm climate, 600–1200mm rainfall, long season.",
        'Sugarcane':    "Tropical/subtropical, heavy irrigation, deep fertile soils, 12–18 month crop.",
        'Pigeonpea':    "Legume; semi-arid tropics, nitrogen-fixing, moderate rainfall.",
    }
    tip = tips.get(predicted_crop, "Consult local agricultural extension for growing advice.")
    print(f"  💡 Tip: {tip}\n")


# ─────────────────────────────────────────────────────────────────
# MODE 2 — Crop Input → Ideal Conditions
# ─────────────────────────────────────────────────────────────────
def mode_crop_insights(artifacts):
    print_header("MODE 2 — Crop Insights & Ideal Conditions")

    df = load_raw_data()
    if df is None:
        print("\n  ❌ Could not load dirty_dataset.csv. Check data/ directory.")
        return

    crops = sorted(df['Crop'].dropna().unique())
    selected_crop = choose_from_list("Select a crop to get insights:", crops)

    subset = df[df['Crop'] == selected_crop].copy()
    separator()
    print(f"\n  📊 INSIGHTS FOR: {selected_crop.upper()}")
    print(f"  Records in dataset: {len(subset)}")
    print(f"  States where grown: {', '.join(sorted(subset['State'].unique()))}\n")

    # Numeric feature stats
    stat_cols = {
        'Rainfall_mm':          'Rainfall (mm)',
        'Irrigation_1000ha':    'Irrigation (1000 ha)',
        'Fertilizer_kg_per_ha': 'Fertilizer (kg/ha)',
        'Yield_kg_per_ha':      'Yield (kg/ha)',
        'Temperature_C':        'Temperature (°C)',
        'Humidity_pct':         'Humidity (%)',
        'Soil_Moisture_pct':    'Soil Moisture (%)',
        'Groundwater_Level_m':  'Groundwater Level (m)',
        'pH':                   'Soil pH',
        'N_kg_per_ha':          'Nitrogen req. (kg/ha)',
        'P_kg_per_ha':          'Phosphorus req. (kg/ha)',
        'K_kg_per_ha':          'Potassium req. (kg/ha)',
    }

    print(f"  {'Feature':<30} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>8} {'Max':>8}")
    separator()
    for col, label in stat_cols.items():
        if col in subset.columns:
            s = subset[col].dropna()
            if len(s) > 0:
                print(f"  {label:<30} {s.mean():>10.2f} {s.median():>10.2f} "
                      f"{s.std():>10.2f} {s.min():>8.2f} {s.max():>8.2f}")

    separator()

    # Year trend
    if 'Year' in subset.columns:
        trend = subset.groupby('Year')['Yield_kg_per_ha'].mean()
        if len(trend) > 5:
            early = trend.iloc[:5].mean()
            late  = trend.iloc[-5:].mean()
            direction = "📈 improving" if late > early else "📉 declining"
            print(f"\n  Yield trend over years: {direction} "
                  f"({early:.0f} kg/ha early → {late:.0f} kg/ha recent)")

    # Optimal state
    state_yield = subset.groupby('State')['Yield_kg_per_ha'].mean().sort_values(ascending=False)
    print(f"\n  🏆 Highest yielding state: {state_yield.index[0]} "
          f"(avg {state_yield.iloc[0]:.0f} kg/ha)")
    print(f"  Top 3 states by yield: {', '.join(state_yield.index[:3])}\n")

    # Ideal conditions summary
    print(f"  ✅ IDEAL CONDITIONS FOR {selected_crop.upper()}:")
    ideal = {
        col: subset[col].median()
        for col in stat_cols if col in subset.columns
    }
    for col, label in stat_cols.items():
        if col in ideal:
            print(f"    • {label:<30}: {ideal[col]:.2f}")
    print()


# ─────────────────────────────────────────────────────────────────
# MAIN MENU
# ─────────────────────────────────────────────────────────────────
def main():
    print(BANNER)

    # Load artifacts once
    print("  Loading models and preprocessors...", end=' ', flush=True)
    artifacts = load_artifacts()
    missing = [k for k, v in artifacts.items() if v is None]
    if missing:
        print(f"\n  ⚠  Some artifacts not found: {missing}")
        print("  Run the notebooks first to generate all model files.\n")
    else:
        print("✅")

    while True:
        separator('═')
        print("\n  MAIN MENU")
        separator()
        print("  [1]  Predict crop from environmental features")
        print("  [2]  Get ideal conditions for a specific crop")
        print("  [3]  Exit")
        separator()
        choice = input("  Select option: ").strip()

        if choice == '1':
            mode_predict_crop(artifacts)
        elif choice == '2':
            mode_crop_insights(artifacts)
        elif choice == '3':
            print("\n  👋 Thank you for using the Crop Classification System!\n")
            sys.exit(0)
        else:
            print("  ⚠  Invalid option. Please enter 1, 2, or 3.\n")

        input("\n  Press [Enter] to return to main menu...")


if __name__ == '__main__':
    main()
