import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


st.set_page_config(page_title="Deforestation Trend (5-year)", layout="wide")

st.title("🌳 Deforestation Trend Early-Warning (5-year window)")
st.caption(
    "Satellite-derived tree cover loss (GFW / World Bank Data360). "
    "Predicts whether loss is WORSENING vs IMPROVING/STABLE compared to the previous 5-year average."
)

# ----------------------------
# CONFIG: set your local CSV path here
# ----------------------------
DEFAULT_CSV_PATH = "WRI_GFW_TREE_COVER_LOSS.csv"

# Optional: let you type/edit the path in the sidebar
st.sidebar.header("Data (local)")
csv_path = st.sidebar.text_input("Local CSV path", value=DEFAULT_CSV_PATH)

import os

if st.sidebar.button("Clear cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

if not os.path.exists(csv_path):
    st.error(f"CSV file not found: {csv_path}")
    st.stop()

@st.cache_data
def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Expected Data360 columns. If you use OWID, adapt here.
    rename_map = {
        "REF_AREA_LABEL": "country",
        "REF_AREA": "iso",
        "TIME_PERIOD": "year",
        "OBS_VALUE": "tree_loss_ha",
    }
    missing = [c for c in rename_map.keys() if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}\n"
            f"Found columns: {df.columns.tolist()}\n\n"
            "If your file is OWID format, change rename_map accordingly."
        )

    df = df.rename(columns=rename_map)
    df = df[["country", "iso", "year", "tree_loss_ha"]].copy()

    df["year"] = df["year"].astype(int)
    df["tree_loss_ha"] = pd.to_numeric(df["tree_loss_ha"], errors="coerce")
    df = df.dropna(subset=["country", "iso", "year", "tree_loss_ha"])
    df["iso"] = df["iso"].astype(str).str.upper().str.strip()
    df = df[df["iso"].str.fullmatch(r"[A-Z]{3}")]

    df = df.sort_values(["country", "year"])

    # Build 5-year lags
    for i in range(1, 6):
        df[f"loss_{i}"] = df.groupby("country")["tree_loss_ha"].shift(i)

    lag_cols = [f"loss_{i}" for i in range(1, 6)]
    df = df.dropna(subset=lag_cols)

    # 5-year average + delta + label
    df["recent_avg_5"] = df[lag_cols].mean(axis=1)
    df["delta_5"] = df["tree_loss_ha"] - df["recent_avg_5"]
    EPS_HA = 100.0  # tolerance in hectares (tune if you want)
    df["trend_worsening"] = (df["delta_5"] > EPS_HA).astype(int)


    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    features = [f"loss_{i}" for i in range(1, 6)]
    X = df[features]
    y = df["trend_worsening"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=400, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, features, acc


def compute_year_map(df: pd.DataFrame, clf, features, year: int) -> pd.DataFrame:
    d = df[df["year"] == year].copy()
    if d.empty:
        return d
    probs = clf.predict_proba(d[features])[:, 1]
    d["prob_worsening"] = probs
    d["trend_pred"] = (d["prob_worsening"] >= 0.5).astype(int)
    d["trend_label"] = np.where(d["trend_pred"] == 1, "WORSENING", "IMPROVING / STABLE")
    return d


def last_years_for_country(df: pd.DataFrame, country: str, year: int):
    d = df[df["country"].str.lower() == country.lower()].copy()
    if d.empty:
        return None
    d = d[d["year"] <= year].sort_values("year")
    return d.tail(6)[["year", "tree_loss_ha"]].copy()  # last 5 + current


# ----------------------------
# Load data
# ----------------------------
try:
    df = load_and_prepare(csv_path)
except Exception as e:
    st.error(f"Failed to load data from: {csv_path}\n\n{e}")
    st.stop()

clf, features, acc = train_model(df)
st.sidebar.success(f"Model trained ✅ (test accuracy ≈ {acc:.3f})")

# Controls
years = sorted(df["year"].unique().tolist())
year = st.sidebar.selectbox("Select year", years, index=len(years) - 1)

# Build year_df first so the dropdown matches what the map can show
year_df = compute_year_map(df, clf, features, year)

if year_df.empty:
    st.warning("No data for this year after requiring 5-year history. Try a later year.")
    st.stop()

countries_in_year = sorted(year_df["country"].unique().tolist())
country = st.sidebar.selectbox("Select country", countries_in_year)


sel = year_df[year_df["country"] == country]
if sel.empty:
    st.warning("Selected country not available for that year (needs 5-year history). Try another year.")
    st.stop()
row = sel.iloc[0]

trend_label = "WORSENING" if row["prob_worsening"] >= 0.5 else "IMPROVING / STABLE"

# ----------------------------
# Layout
# ----------------------------
left, right = st.columns([1.2, 1])

with left:
    st.subheader(f"🗺️ Global map — Probability of Worsening ({year})")
    fig = px.choropleth(
        year_df,
        locations="iso",
        color="prob_worsening",
        hover_name="country",
        hover_data={
            "prob_worsening": ":.3f",
            "tree_loss_ha": ":.1f",
            "recent_avg_5": ":.1f",
            "delta_5": ":.1f",
            "iso": False,
        },
        color_continuous_scale="RdYlGn_r",
        range_color=(0, 1),
        labels={"prob_worsening": "P(worsening)"},
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("📌 Selected location summary")
    st.metric("Trend prediction", trend_label)
    st.metric("Probability worsening", f"{row['prob_worsening']:.3f}")

    st.write("**Current loss (ha):**", f"{row['tree_loss_ha']:.2f}")
    st.write("**5-year avg (ha):**", f"{row['recent_avg_5']:.2f}")
    st.write("**Delta vs 5-yr avg (ha):**", f"{row['delta_5']:.2f}")

    st.divider()
    st.subheader("📈 Last 5 years of loss (plus current year)")
    tail6 = last_years_for_country(df, country, year)
    if tail6 is None or tail6.empty:
        st.info("Not enough history to plot.")
    else:
        fig2 = px.line(
        tail6, x="year", y="tree_loss_ha", markers=True,
        labels={"tree_loss_ha": "Tree cover loss (ha)", "year": "Year"}
        )
        # Optional: log scale toggle (helps small countries)
        if st.checkbox("Log scale (plot)", value=False):
            fig2.update_yaxes(type="log")
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("🔎 Top countries at risk (this year)")
    topk = year_df.sort_values("prob_worsening", ascending=False).head(10)
    st.dataframe(topk[["country", "prob_worsening", "tree_loss_ha", "recent_avg_5", "delta_5"]].reset_index(drop=True))
