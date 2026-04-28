"""
Satellite Intelligence Platform — Streamlit Dashboard
Visualises damage-detection results across Mariupol, Kharkiv and Bakhmut.
Run from the project root:  streamlit run dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent / "satellite-damage-detection"
METRICS_DIR = ROOT / "results" / "metrics"
PREDS_PATH = ROOT / "results" / "predictions" / "test_predictions.csv"
SPECTRAL_PATH = ROOT / "data" / "processed" / "sentinel2_clean_2021_2024.csv"
SUMMARY_PATH = METRICS_DIR / "final_summary.json"

CITY_COLORS = {
    "Kharkiv": "#4C9BE8",
    "Mariupol": "#E8844C",
    "Bakhmut": "#6CC24A",
}

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Satellite Intelligence Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── light custom CSS ───────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        border-left: 4px solid #4C9BE8;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin-bottom: .5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── data loading (cached) ──────────────────────────────────────────────────────
@st.cache_data
def load_summary():
    with open(SUMMARY_PATH) as f:
        return json.load(f)

@st.cache_data
def load_model_comparison():
    return pd.read_csv(METRICS_DIR / "model_comparison.csv")

@st.cache_data
def load_per_city():
    return pd.read_csv(METRICS_DIR / "per_city_metrics.csv")

@st.cache_data
def load_spectral():
    df = pd.read_csv(SPECTRAL_PATH)
    df["year"] = df["year"].astype(int)
    df["period_label"] = df["year"].astype(str) + " " + df["quarter"].fillna("")
    df["period_label"] = df["period_label"].str.strip()
    df = df.sort_values(["city", "year", "quarter"])
    return df

@st.cache_data
def load_predictions():
    return pd.read_csv(PREDS_PATH)

summary = load_summary()
model_df = load_model_comparison()
city_df = load_per_city()
spectral_df = load_spectral()
pred_df = load_predictions()

# ── header ────────────────────────────────────────────────────────────────────
st.title("🛰️ Satellite Intelligence Platform")
st.markdown(
    "**Automated conflict-damage detection** in Ukrainian cities using Sentinel-2 imagery "
    "(Mariupol · Kharkiv · Bakhmut · 2021–2024)"
)
st.divider()

# ── tabs ───────────────────────────────────────────────────────────────────────
tab_overview, tab_models, tab_cities, tab_spectral, tab_predictions = st.tabs(
    ["📊 Overview", "🤖 Model Comparison", "🏙️ Per-City Analysis", "🌿 Spectral Trends", "🔍 Predictions"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    st.markdown("### Best model performance on held-out test set")

    cols = st.columns(5)
    metrics = [
        ("Accuracy", summary["test_accuracy"], "🎯"),
        ("Precision", summary["test_precision"], "🔬"),
        ("Recall", summary["test_recall"], "📡"),
        ("F1 Score", summary["test_f1"], "⚖️"),
        ("AUC-ROC", summary["test_auc_roc"], "📈"),
    ]
    for col, (label, val, icon) in zip(cols, metrics):
        col.metric(f"{icon} {label}", f"{val:.1%}")

    st.divider()

    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("#### Dataset split")
        split_data = pd.DataFrame(
            {"Split": ["Train", "Validation", "Test"],
             "Scenes": [summary["n_train"], summary["n_val"], summary["n_test"]]}
        )
        fig = px.bar(
            split_data, x="Split", y="Scenes", color="Split",
            color_discrete_sequence=["#4C9BE8", "#6CC24A", "#E8844C"],
            text="Scenes",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.markdown("#### Project at a glance")
        st.markdown(
            f"""
| Detail | Value |
|---|---|
| Best model | **{summary['best_model']}** |
| Features | {summary['n_features']} spectral indices |
| Cities | {', '.join(summary['cities_evaluated'])} |
| Training scenes | {summary['n_train']} |
| Test scenes | {summary['n_test']} |
| Decision threshold | {summary['optimal_threshold']} |
"""
        )
        st.info(
            "Labels: **0 = undamaged** (2021 baseline) · **1 = damaged** (2022–2024 conflict period)",
            icon="ℹ️",
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tab_models:
    st.markdown("### Model comparison on test set")

    metric_col = st.selectbox(
        "Metric to compare",
        ["test_acc", "test_f1", "test_auc", "test_precision", "test_recall"],
        format_func=lambda x: {
            "test_acc": "Accuracy",
            "test_f1": "F1 Score",
            "test_auc": "AUC-ROC",
            "test_precision": "Precision",
            "test_recall": "Recall",
        }[x],
    )

    fig = px.bar(
        model_df.sort_values(metric_col, ascending=True),
        x=metric_col, y="model",
        orientation="h",
        color=metric_col,
        color_continuous_scale="Blues",
        text=model_df.sort_values(metric_col, ascending=True)[metric_col].map(lambda v: f"{v:.3f}"),
        labels={"model": "Model", metric_col: metric_col.replace("_", " ").title()},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False, height=340,
    )
    st.plotly_chart(fig, width='stretch')

    st.divider()
    st.markdown("#### All metrics side-by-side")

    display_df = model_df.copy()
    for col in ["train_acc", "val_acc", "test_acc", "test_precision", "test_recall", "test_f1", "test_auc"]:
        display_df[col] = display_df[col].map(lambda v: f"{v:.4f}")
    display_df.columns = ["Model", "Train Acc", "Val Acc", "Test Acc", "Precision", "Recall", "F1", "AUC"]
    st.dataframe(display_df.set_index("Model"), width='stretch')

    st.markdown(
        "> **Note:** ResNet-50 was trained for fewer epochs due to compute constraints and "
        "shows degraded performance; the classical ML models generalise well on 6 spectral features."
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · PER-CITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_cities:
    st.markdown("### Per-city model performance")

    metric_city = st.radio(
        "Metric",
        ["accuracy", "f1", "auc_roc", "precision", "recall"],
        horizontal=True,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    city_df_sorted = city_df.sort_values(metric_city, ascending=False)
    fig = px.bar(
        city_df_sorted, x="city", y=metric_city,
        color="city",
        color_discrete_map=CITY_COLORS,
        text=city_df_sorted[metric_city].map(lambda v: f"{v:.3f}"),
        labels={"city": "City", metric_city: metric_city.replace("_", " ").title()},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)", height=360, yaxis_range=[0, 1.08],
    )
    st.plotly_chart(fig, width='stretch')

    st.divider()
    st.markdown("#### Damage scene counts by city")

    c1, c2 = st.columns(2)
    with c1:
        fig_pie = px.pie(
            city_df, values="n_scenes", names="city",
            color="city", color_discrete_map=CITY_COLORS,
            hole=0.4, title="Total scenes per city",
        )
        fig_pie.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=320,
        )
        st.plotly_chart(fig_pie, width='stretch')

    with c2:
        fig_dmg = px.bar(
            city_df, x="city", y=["n_damaged"],
            color_discrete_sequence=["#E8844C"],
            labels={"value": "Damaged scenes", "city": "City"},
            title="Damaged scenes per city",
        )
        fig_dmg.update_layout(
            showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", height=320,
        )
        st.plotly_chart(fig_dmg, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · SPECTRAL TRENDS
# ─────────────────────────────────────────────────────────────────────────────
with tab_spectral:
    st.markdown("### Spectral index trends over time")
    st.caption(
        "NDVI (vegetation), NDBI (built-up), BSI (bare soil) averaged across "
        "all valid pixels per city per quarter. Conflict onset: early 2022."
    )

    index_choice = st.selectbox(
        "Spectral index",
        ["ndvi_mean", "ndbi_mean", "bsi_mean"],
        format_func=lambda x: {
            "ndvi_mean": "NDVI — Normalised Difference Vegetation Index",
            "ndbi_mean": "NDBI — Normalised Difference Built-up Index",
            "bsi_mean": "BSI — Bare Soil Index",
        }[x],
    )

    city_filter = st.multiselect(
        "Cities", options=spectral_df["city"].unique().tolist(),
        default=spectral_df["city"].unique().tolist(),
    )

    plot_df = spectral_df[spectral_df["city"].isin(city_filter)].copy()
    plot_df["date_approx"] = pd.to_datetime(
        plot_df["year"].astype(str) + plot_df["quarter"].str.replace("Q", "-0", regex=False),
        format="%Y-%m",
        errors="coerce",
    )

    fig = px.line(
        plot_df, x="period_label", y=index_choice, color="city",
        color_discrete_map=CITY_COLORS,
        markers=True,
        labels={"period_label": "Period", index_choice: index_choice.replace("_", " ").upper()},
    )
    # add_vline doesn't support categorical string axes in this Plotly version;
    # use add_shape + add_annotation with xref="x" / yref="paper" instead.
    if "2022 Q1" in plot_df["period_label"].values:
        fig.add_shape(
            type="line",
            x0="2022 Q1", x1="2022 Q1",
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="red", dash="dash", width=2),
        )
        fig.add_annotation(
            x="2022 Q1", y=1.07,
            xref="x", yref="paper",
            text="⬇ Conflict onset",
            showarrow=False,
            font=dict(color="red", size=12),
        )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=420, xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, width='stretch')

    st.divider()
    st.markdown("#### All three indices — faceted by city")

    long_df = plot_df.melt(
        id_vars=["city", "period_label"],
        value_vars=["ndvi_mean", "ndbi_mean", "bsi_mean"],
        var_name="index", value_name="value",
    )
    long_df["index"] = long_df["index"].str.upper().str.replace("_MEAN", "")

    fig2 = px.line(
        long_df, x="period_label", y="value", color="index",
        facet_col="city", facet_col_wrap=3, markers=True,
        color_discrete_sequence=["#6CC24A", "#4C9BE8", "#E8844C"],
        labels={"period_label": "Period", "value": "Index value", "index": "Index"},
    )
    fig2.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=380, xaxis_tickangle=-45,
    )
    st.plotly_chart(fig2, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 · PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
with tab_predictions:
    st.markdown("### Test-set prediction explorer")

    c1, c2, c3 = st.columns(3)
    total = len(pred_df)
    correct = pred_df["correct_prediction"].sum()
    c1.metric("Total patches", total)
    c2.metric("Correctly classified", int(correct))
    c3.metric("Error rate", f"{(1 - correct/total):.1%}")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Damage probability distribution")
        fig_hist = px.histogram(
            pred_df, x="damage_probability", color="city",
            color_discrete_map=CITY_COLORS,
            nbins=30, barmode="overlay", opacity=0.75,
            labels={"damage_probability": "Damage probability", "city": "City"},
        )
        fig_hist.add_shape(
            type="line", x0=0.5, x1=0.5, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="white", dash="dash", width=2),
        )
        fig_hist.add_annotation(
            x=0.5, y=1.04, xref="x", yref="paper",
            text="Decision threshold (0.5)",
            showarrow=False, font=dict(color="white", size=11),
        )
        fig_hist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=360,
        )
        st.plotly_chart(fig_hist, width='stretch')

    with col_b:
        st.markdown("#### Damage score vs probability")
        fig_scatter = px.scatter(
            pred_df, x="damage_score", y="damage_probability",
            color="city", symbol="correct_prediction",
            color_discrete_map=CITY_COLORS,
            symbol_map={1: "circle", 0: "x"},
            labels={
                "damage_score": "Raw damage score",
                "damage_probability": "Predicted probability",
                "city": "City",
                "correct_prediction": "Correct?",
            },
            opacity=0.75,
        )
        fig_scatter.add_shape(
            type="line", x0=0, x1=1, y0=0.5, y1=0.5,
            xref="paper", yref="y",
            line=dict(color="white", dash="dash", width=2),
        )
        fig_scatter.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=360,
        )
        st.plotly_chart(fig_scatter, width='stretch')

    st.divider()
    st.markdown("#### NDVI vs NDBI coloured by prediction correctness")

    fig_ndvi = px.scatter(
        pred_df, x="ndvi_mean", y="ndbi_mean",
        color="correct_prediction",
        color_continuous_scale=["#E8844C", "#4C9BE8"],
        facet_col="city",
        opacity=0.8,
        labels={
            "ndvi_mean": "NDVI mean",
            "ndbi_mean": "NDBI mean",
            "correct_prediction": "Correct",
        },
        hover_data=["period", "damage_probability"],
    )
    fig_ndvi.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=360,
    )
    st.plotly_chart(fig_ndvi, width='stretch')

    with st.expander("Raw prediction data"):
        st.dataframe(pred_df, width='stretch', height=320)