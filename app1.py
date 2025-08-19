# app.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import joblib

import time

def color_scale_factory(theme: str):
    """Return a function mapping value in [0,1] -> RGB list."""
    if theme == "Viridis":
        # quick 3-point viridis-ish
        stops = [(0.0, (68, 1, 84)), (0.5, (33, 145, 140)), (1.0, (253, 231, 37))]
    elif theme == "Heat":
        stops = [(0.0, (0, 200, 0)), (0.5, (255, 200, 0)), (1.0, (220, 30, 30))]
    else:  # Classic
        stops = [(0.0, (40, 167, 69)), (0.5, (255, 193, 7)), (1.0, (220, 53, 69))]

    def interp_rgb(t):
        t = max(0.0, min(1.0, float(t)))
        # find segment
        for i in range(1, len(stops)):
            if t <= stops[i][0]:
                t0, c0 = stops[i-1]
                t1, c1 = stops[i]
                w = 0 if t1 == t0 else (t - t0)/(t1 - t0)
                return [
                    int(c0[j] + (c1[j] - c0[j])*w) for j in range(3)
                ]
        return list(stops[-1][1])
    return interp_rgb


def norm01(series: pd.Series) -> pd.Series:
    """Normalize to [0,1] safely."""
    s = series.astype(float).copy()
    lo, hi = s.min(), s.max()
    if hi - lo == 0:
        return pd.Series([0.0]*len(s), index=s.index)
    return (s - lo) / (hi - lo)



# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Service Delivery Protest Risk — South Africa",
                   page_icon="🛡️", layout="wide")

DATA_PATH = Path("crime_all_clean.csv")
ART_DIR   = Path("artifacts")
MODEL_PATH = ART_DIR / "best_model.joblib"
META_PATH  = ART_DIR / "model_meta.json"
FEATURES_PATH = ART_DIR / "features_crime.csv"   # optional; created in notebook

PRIMARY_CAT_COLS_DEFAULT = ["province", "crime_category"]
PRIMARY_NUM_COLS_DEFAULT = ["year", "yoy_delta", "roll3_mean"]


# ----------------------------
# Helpers (cached)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_crime() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # normalise column names
    df.columns = [c.strip() for c in df.columns]
    # ensure dtypes
    if "year" in df:
        df["year"] = df["year"].astype(int)
    if "count" in df:
        df["count"] = pd.to_numeric(df["count"], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_features_fallback(crime: pd.DataFrame) -> pd.DataFrame:
    """If artifacts/features_crime.csv is missing, compute minimal features."""
    df = crime.sort_values(["province", "crime_category", "year"]).copy()
    df["yoy_delta"] = (
        df.groupby(["province", "crime_category"])["count"].diff()
    )
    df["roll3_mean"] = (
        df.groupby(["province", "crime_category"])["count"]
          .rolling(3, min_periods=2).mean().reset_index(level=[0,1], drop=True)
    )
    return df

@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    crime = load_crime()
    if FEATURES_PATH.exists():
        f = pd.read_csv(FEATURES_PATH)
        # keep consistent dtypes
        if "year" in f:
            f["year"] = f["year"].astype(int)
        return f
    return load_features_fallback(crime)

@st.cache_resource(show_spinner=False)
def load_model_and_meta():
    pipe = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    return pipe, meta

def get_ohe_feature_names(prep: ColumnTransformer, cat_cols: list, num_cols: list):
    """Rebuild feature names after OneHot."""
    try:
        ohe = prep.named_transformers_["cat"].named_steps["onehot"]
        cat_names = list(ohe.get_feature_names_out(cat_cols))
    except Exception:
        cat_names = []  # if model doesn't have OHE
    return cat_names + list(num_cols)

def kfmt(x):
    return f"{int(x/1000)}k" if x >= 1_000 else f"{int(x)}"


# ----------------------------
# Load data & model
# ----------------------------
crime = load_crime()
features = load_features()
pipe, meta = load_model_and_meta()

cat_cols = meta.get("cat_cols", PRIMARY_CAT_COLS_DEFAULT)
num_cols = meta.get("num_cols", PRIMARY_NUM_COLS_DEFAULT)

st.title("🛡️ Service Delivery Protest Risk — South Africa")

# Sidebar
st.sidebar.header("Filters")
year_sel = st.sidebar.select_slider("Year", options=sorted(crime["year"].unique()), value=int(crime["year"].max()))
prov_sel = st.sidebar.selectbox("Province", sorted(crime["province"].unique()))
topN = st.sidebar.slider("Top N crimes", min_value=3, max_value=10, value=5, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("Artifacts")
st.sidebar.write(f"**Model:** `{MODEL_PATH.name}`" if MODEL_PATH.exists() else ":red[Model not found]")
st.sidebar.write(f"**Meta:** `{META_PATH.name}`" if META_PATH.exists() else ":red[Meta not found]")


# ----------------------------
# Section 1 — Data overview
# ----------------------------
with st.expander("📦 Data overview", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(crime):,}")
    c2.metric("Provinces", crime["province"].nunique())
    c3.metric("Crime categories", crime["crime_category"].nunique())
    c4.metric("Years", crime["year"].nunique())
    st.dataframe(crime.head(20), use_container_width=True)


# ----------------------------
# Section 2 — Dashboards (Plotly)
# ----------------------------
st.subheader("📊 Storytelling Dashboards")

# 2.1 Heatmap province x year
hm = crime.pivot_table(values="count", index="province", columns="year", aggfunc="sum").fillna(0)
fig_hm = px.imshow(
    hm, aspect="auto", color_continuous_scale="Reds",
    labels=dict(color="Total crimes"), title="Heatmap — Crimes by Province & Year"
)
st.plotly_chart(fig_hm, use_container_width=True)

# 2.2 Top N crimes (national)
top_df = (crime.groupby("crime_category")["count"].sum()
          .nlargest(topN).sort_values(ascending=True).reset_index())
fig_top = px.bar(
    top_df, x="count", y="crime_category", orientation="h",
    title=f"Top {topN} Crime Categories (National)", text="count", color="crime_category",
    color_discrete_sequence=px.colors.sequential.Mako
)
fig_top.update_traces(texttemplate="%{text:,}", textposition="outside")
fig_top.update_layout(showlegend=False, xaxis_title="Total cases", yaxis_title="")
st.plotly_chart(fig_top, use_container_width=True)

# 2.3 Province comparison for Top N crimes
pc = (crime[crime["crime_category"].isin(top_df["crime_category"])]
      .groupby(["province", "crime_category"])["count"].sum().reset_index())
fig_stack = px.bar(
    pc, x="province", y="count", color="crime_category", barmode="group",
    title="Province Comparison — Top Crimes"
)
fig_stack.update_layout(xaxis_tickangle=35, yaxis_title="Total cases")
st.plotly_chart(fig_stack, use_container_width=True)

# 2.4 National trend + 2.5 Gauteng trend (adjacent)
cA, cB = st.columns(2)
trend = crime.groupby("year")["count"].sum().reset_index()
fig_trend = px.line(trend, x="year", y="count", markers=True, title="National Crime Trend Over Time")
fig_trend.update_yaxes(tickformat=",")  # thousand separator
cA.plotly_chart(fig_trend, use_container_width=True)

gt = (crime[crime["province"].str.lower() == "gauteng"]
      .groupby("year")["count"].sum().reset_index())
fig_gt = px.line(gt, x="year", y="count", markers=True, title="Gauteng Crime Trends Over Years",
                 color_discrete_sequence=["#ff7f0e"])
fig_gt.update_yaxes(tickformat=",")
cB.plotly_chart(fig_gt, use_container_width=True)

# 2.6 Regression scatter (roll3_mean vs count) if available
if {"roll3_mean", "count"}.issubset(features.columns):
    reg = features.dropna(subset=["roll3_mean", "count"]).sample(min(5000, len(features)), random_state=42)
    fig_reg = px.scatter(reg, x="roll3_mean", y="count", trendline="ols",
                         opacity=0.35, title="Scatter + Trendline (roll3_mean vs count)")
    fig_reg.update_yaxes(tickformat=",")
    st.plotly_chart(fig_reg, use_container_width=True)


# ----------------------------
# Section 3 — Predict risk (High / Low)
# ----------------------------
st.subheader("🤖 Predict Risk")

if pipe is None:
    st.warning("Model not found. Train the notebook and ensure artifacts are in the **artifacts/** folder.")
else:
    # Controls
    left, right = st.columns([1, 2])
    with left:
        province = st.selectbox("Province", sorted(features["province"].unique()), index=sorted(features["province"].unique()).index(prov_sel) if prov_sel in features["province"].unique() else 0)
        category = st.selectbox("Crime category", sorted(features["crime_category"].unique()))
        year_inp = st.select_slider("Year for prediction", options=sorted(features["year"].unique()), value=year_sel)

    # Pull engineered row (compute if missing)
    row = (features[(features["province"] == province) &
                    (features["crime_category"] == category) &
                    (features["year"] == year_inp)]).copy()

    if row.empty:
        st.info("No engineered row found for that combination — computing features on-the-fly.")
        subset = crime[(crime["province"] == province) & (crime["crime_category"] == category)].sort_values("year")
        subset["yoy_delta"] = subset["count"].diff()
        subset["roll3_mean"] = subset["count"].rolling(3, min_periods=2).mean()
        row = subset[subset["year"] == year_inp].copy()

    # Build model input
    need_cols = cat_cols + [c for c in num_cols if c in row.columns]
    X_infer = row[need_cols].head(1)

    # Predict
    pred = pipe.predict(X_infer)[0]
    proba = None
    try:
        classes = list(getattr(pipe.named_steps["model"], "classes_", []))
        pos_idx = classes.index(1) if 1 in classes else 0
        p = pipe.predict_proba(X_infer)
        proba = float(p[0, pos_idx]) if p.shape[1] > pos_idx else None
    except Exception:
        pass

    # Display
    with right:
        risk_label = "High Risk" if int(pred) == 1 else "Low Risk"
        st.metric("Predicted class", risk_label)
        if proba is not None:
            st.progress(proba, text=f"Probability of HIGH risk: {proba:.2%}")

        st.caption("Model input (after feature engineering)")
        st.dataframe(X_infer, use_container_width=True)


# ----------------------------
# Section 4 — Model insights
# ----------------------------
st.subheader("🔎 Model Insights")

if pipe is not None and isinstance(pipe, Pipeline) and "prep" in pipe.named_steps:
    prep = pipe.named_steps["prep"]
    feature_names = get_ohe_feature_names(prep, cat_cols, num_cols)
    model = pipe.named_steps["model"]

    # Feature importances (tree models)
    if hasattr(model, "feature_importances_"):
        importances = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False).head(20)
        fig_imp = px.bar(importances.sort_values("importance"),
                         x="importance", y="feature", orientation="h",
                         title="Top Feature Importances",
                         color="importance", color_continuous_scale="Viridis")
        fig_imp.update_layout(coloraxis_showscale=False, xaxis_title="Importance", yaxis_title="")
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Current model does not expose `feature_importances_`.")
else:
    st.info("Model pipeline is missing the preprocessing step.")


# ----------------------------
# Section 5 — About / Metrics
# ----------------------------
st.subheader("📑 About & Metrics")
if (ART_DIR / "metrics.json").exists():
    m = json.loads((ART_DIR / "metrics.json").read_text())
    cols = st.columns(5)
    cols[0].metric("AUC", f"{m.get('AUC', '–')}")
    cols[1].metric("Accuracy", f"{m.get('Accuracy', 0):.3f}")
    cols[2].metric("Precision", f"{m.get('Precision', 0):.3f}")
    cols[3].metric("Recall", f"{m.get('Recall', 0):.3f}")
    cols[4].metric("F1-Score", f"{m.get('F1', 0):.3f}")
else:
    st.write("No metrics file found. Train/evaluate in the notebook to create `artifacts/metrics.json`.")

st.caption("Built from notebook artifacts. Data: SAPS + ACLED (processed).")
