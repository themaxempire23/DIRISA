import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from datetime import datetime

ARTIFACT_PATH = "models/protest_best_model.joblib"

# ---------- Model ----------
@st.cache_resource
def load_model():
    if not os.path.exists(ARTIFACT_PATH):
        st.error("Model artifact not found. Train & save it first.")
        st.stop()
    return joblib.load(ARTIFACT_PATH)

def preprocess_for_inference(x, bundle):
    selector = bundle["feature_selector"]
    scaler   = bundle["scaler"]
    cols     = bundle["fitted_feature_columns"]
    df = pd.DataFrame([x]) if isinstance(x, dict) else x.copy()
    df = df.reindex(columns=cols, fill_value=0).replace([np.inf, -np.inf], 0).fillna(0)
    X  = selector.transform(df) if selector is not None else df.values
    return scaler.transform(X)

# ---------- Geo + helpers ----------
SA_PROVINCES = [
    {"province": "Eastern Cape",   "lat": -33.0460, "lon": 27.9064, "landmark": "Addo Elephant NP"},
    {"province": "Free State",     "lat": -29.1158, "lon": 26.2290, "landmark": "Golden Gate"},
    {"province": "Gauteng",        "lat": -26.2708, "lon": 28.1123, "landmark": "Union Buildings"},
    {"province": "KwaZulu-Natal",  "lat": -29.7980, "lon": 30.9990, "landmark": "uShaka / Drakensberg"},
    {"province": "Limpopo",        "lat": -23.4013, "lon": 29.4179, "landmark": "Mapungubwe"},
    {"province": "Mpumalanga",     "lat": -25.5653, "lon": 30.5277, "landmark": "Blyde Canyon"},
    {"province": "North West",     "lat": -26.6639, "lon": 25.2838, "landmark": "Pilanesberg"},
    {"province": "Northern Cape",  "lat": -28.7282, "lon": 23.0759, "landmark": "Augrabies"},
    {"province": "Western Cape",   "lat": -33.9249, "lon": 18.4241, "landmark": "Table Mountain"},
]

def norm01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    lo, hi = float(s.min()), float(s.max())
    return pd.Series(0.0, index=s.index) if hi == lo else (s - lo) / (hi - lo)

def one_hot_prov(row, province, cols):
    prov_cols = [c for c in cols if c.lower().startswith("province_")]
    if not prov_cols: return row
    for c in prov_cols: row[c] = 0
    t = f"province_{province}".lower()
    for c in prov_cols:
        if c.lower() == t: row[c] = 1; break
    return row

def predict_all_provinces(base_row: dict, bundle) -> pd.DataFrame:
    est = bundle["estimator"]
    cols = bundle["fitted_feature_columns"]
    out = []
    for p in SA_PROVINCES:
        r = base_row.copy()
        r = one_hot_prov(r, p["province"], cols)
        X = preprocess_for_inference(r, bundle)
        proba = float(est.predict_proba(X)[0, 1])
        out.append({
            "province": p["province"], "lat": p["lat"], "lon": p["lon"],
            "landmark": p["landmark"], "prob_high_risk": proba,
            "protest_events": float(r.get("protest_events", 0.0)),
            "violence_events": float(r.get("violence_events", 0.0)),
        })
    return pd.DataFrame(out)

# ---------- App ----------
st.set_page_config(page_title="Service Delivery Risk", page_icon="📊", layout="wide")
st.title("📊 Service Delivery Protest Risk — South Africa")

bundle = load_model()
fitted_cols = bundle["fitted_feature_columns"]

# Sidebar: simple inputs
st.sidebar.header("Inputs")
protest_events = st.sidebar.number_input("protest_events", min_value=0.0, value=3.0, step=1.0)
violence_events = st.sidebar.number_input("violence_events", min_value=0.0, value=1.0, step=1.0)
month = st.sidebar.number_input("month (1-12)", min_value=1, max_value=12, value=datetime.now().month, step=1)
year  = st.sidebar.number_input("year", min_value=2000, max_value=2100, value=datetime.now().year, step=1)

# Single, clear toggle for what the columns mean
metric = st.segmented_control("Map metric", options=["Risk", "Protest", "Violence"], default="Risk")

base_row = {"protest_events": protest_events, "violence_events": violence_events, "month": month, "year": year}
df = predict_all_provinces(base_row, bundle)

# Choose value
if metric == "Protest":
    df["value"] = df["protest_events"]
elif metric == "Violence":
    df["value"] = df["violence_events"]
else:
    df["value"] = df["prob_high_risk"]  # Risk

# Scale to height + color
df["v01"] = norm01(df["value"])
# red for high, green for low
df["color"] = df["v01"].apply(lambda t: [int(255*t), int(255*(1-t)), 60])
elev_scale = 60000 if metric == "Risk" else 5000
df["elev"] = (df["v01"] * elev_scale).astype(float)

# ColumnLayer map (only this layer)
view = pdk.ViewState(latitude=-29.0, longitude=25.0, zoom=4.2, pitch=45, bearing=15)
layer = pdk.Layer(
    "ColumnLayer",
    data=df,
    get_position='[lon, lat]',
    get_elevation='elev',
    get_fill_color='color',
    radius=10000,
    elevation_scale=1,
    pickable=True,
    auto_highlight=True,
)
tooltip = {
    "html": "<b>{province}</b><br/>Landmark: {landmark}<br/>Value: {value:.3f}<br/>P(high): {prob_high_risk:.3f}",
    "style": {"backgroundColor": "rgba(35,35,35,0.85)", "color": "white"}
}
deck = pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip, map_style="mapbox://styles/mapbox/dark-v10")
st.pydeck_chart(deck, use_container_width=True)

# Small dashboard row (keeps it real, not “sus”)
c1, c2, c3, c4 = st.columns(4)
c1.metric("High‑risk avg", f"{df['prob_high_risk'].mean():.2%}")
c2.metric("Max value", f"{df['value'].max():.2f}")
c3.metric("Min value", f"{df['value'].min():.2f}")
c4.metric("Provinces", len(df))

# Quick top list
st.markdown("### Top Provinces")
top = df.nlargest(5, "value")[["province", "value"]].reset_index(drop=True)
st.dataframe(top, use_container_width=True)
