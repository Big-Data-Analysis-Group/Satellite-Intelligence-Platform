"""
Satellite Intelligence Platform — Combined Dashboard
Run from the project root:  streamlit run app.py
"""

import io
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Satellite Intelligence Platform",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .section-title {
        font-size: 1.05rem; font-weight: 600; color: #a0aec0;
        text-transform: uppercase; letter-spacing: .08em; margin-bottom: .4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent / "satellite-damage-detection"
METRICS_DIR = ROOT / "results" / "metrics"
PREDS_PATH  = ROOT / "results" / "predictions" / "test_predictions.csv"
SPECTRAL_PATH = ROOT / "data" / "processed" / "sentinel2_clean_2021_2024.csv"
SUMMARY_PATH  = METRICS_DIR / "final_summary.json"
BASE_RAW      = str(ROOT / "data" / "raw")
BASE_SPLITS   = str(ROOT / "data" / "splits")

CITY_COLORS = {"Kharkiv": "#4C9BE8", "Mariupol": "#E8844C", "Bakhmut": "#6CC24A"}
CITIES    = ["Mariupol", "Kharkiv", "Bakhmut"]
QUARTERS  = ["Q1", "Q2", "Q3", "Q4"]
ALL_YEARS = [2021, 2022, 2023, 2024]

# ── Heatmap constants ─────────────────────────────────────────────────────────
CITY_BBOX = {
    "Mariupol": [37.40, 46.95, 37.68, 47.15],
    "Kharkiv":  [35.80, 49.85, 36.50, 50.10],
    "Bakhmut":  [37.70, 48.50, 37.90, 48.65],
}
CITY_CROP = {
    "Mariupol": (1024, 1024),
    "Kharkiv":  (1408, 2560),
    "Bakhmut":  (768,  640),
}
MAX_VALID_DIM = {"Mariupol": 3000, "Kharkiv": 5000, "Bakhmut": 9999}
DISPLAY_PS, MIN_VALID = 64, 0.10

_STOPS = [(0.00,"#2d6a4f"),(0.15,"#52b788"),(0.30,"#d4e157"),
          (0.45,"#ffca28"),(0.60,"#ff7043"),(0.75,"#e53935"),(1.00,"#4a1a00")]
DAMAGE_CMAP = LinearSegmentedColormap.from_list("damage", [(s[0],s[1]) for s in _STOPS])
NDVI_CMAP   = DAMAGE_CMAP.reversed()

# ── Dashboard data loaders ────────────────────────────────────────────────────
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
    df["period_label"] = (df["year"].astype(str) + " " + df["quarter"].fillna("")).str.strip()
    return df.sort_values(["city", "year", "quarter"])

@st.cache_data
def load_predictions():
    return pd.read_csv(PREDS_PATH)

# ── Heatmap helpers ───────────────────────────────────────────────────────────
def scene_path(city, year, quarter=None):
    name = f"S2_{year}_{quarter}_{city}_clean.npy" if quarter else f"S2_{year}_{city}_clean.npy"
    p = os.path.join(BASE_RAW, name)
    return p if os.path.exists(p) else None

def _crop(arr, city):
    mH, mW = CITY_CROP[city]
    H, W = arr.shape[:2]
    return arr[:min(H, mH), :min(W, mW)]

def _stretch(x):
    v = x[~np.isnan(x)]
    if v.size == 0:
        return np.full_like(x, 0.15)
    p2, p98 = np.percentile(v, [2, 98])
    return np.clip((x - p2) / max(p98 - p2, 1e-6), 0, 1)

@st.cache_data(show_spinner="Rendering satellite image…")
def get_rgb_thumbnail(city, year, quarter=None, max_dim=900):
    p = scene_path(city, year, quarter)
    if p is None:
        return None
    raw = np.load(p, mmap_mode="r")
    if max(raw.shape[:2]) > MAX_VALID_DIM[city]:
        return None
    arr = _crop(raw, city)
    H, W = arr.shape[:2]
    rgb = np.stack([_stretch(np.array(arr[:,:,i], np.float32)) for i in [2,1,0]], 2)
    rgb[np.isnan(rgb)] = 0.15
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    if max(H, W) > max_dim:
        scale = max_dim / max(H, W)
        img = img.resize((int(W*scale), int(H*scale)), Image.LANCZOS)
    return np.array(img), H, W

@st.cache_data(show_spinner="Computing patches…")
def get_patch_ndvi(city, year, quarter=None, ps=DISPLAY_PS):
    p = scene_path(city, year, quarter)
    if p is None:
        return {}, 0, 0
    raw = np.load(p, mmap_mode="r")
    if max(raw.shape[:2]) > MAX_VALID_DIM[city]:
        return {}, 0, 0
    arr = _crop(raw, city)
    H, W = arr.shape[:2]
    red = np.array(arr[:,:,2], np.float32)
    nir = np.array(arr[:,:,3], np.float32)
    out, threshold = {}, MIN_VALID * ps * ps
    for ri in range(H // ps):
        for ci in range(W // ps):
            rp = red[ri*ps:(ri+1)*ps, ci*ps:(ci+1)*ps]
            np_ = nir[ri*ps:(ri+1)*ps, ci*ps:(ci+1)*ps]
            d = np_ + rp
            ndvi_p = np.where(d != 0, (np_ - rp) / d, np.nan)
            v = ndvi_p[~np.isnan(ndvi_p)]
            if v.size >= threshold:
                out[(ri, ci)] = float(np.mean(v))
    return out, H, W

@st.cache_data(show_spinner="Computing 2021 baseline…")
def get_baseline(city, ps=DISPLAY_PS):
    from collections import defaultdict
    sums, cnts = defaultdict(float), defaultdict(int)
    H0 = W0 = 0
    for q in QUARTERS:
        patches, H, W = get_patch_ndvi(city, 2021, q, ps)
        if H: H0, W0 = H, W
        for k, v in patches.items():
            sums[k] += v; cnts[k] += 1
    return {k: sums[k]/cnts[k] for k in sums}, H0, W0

def get_damage(city, year, quarter, ps=DISPLAY_PS):
    current, H, W = get_patch_ndvi(city, year, quarter, ps)
    baseline, H0, W0 = get_baseline(city, ps)
    return {k: baseline[k] - v for k, v in current.items() if k in baseline}, H or H0, W or W0

@st.cache_data(show_spinner=False)
def find_best_quarter(city, year, ps=DISPLAY_PS):
    best_q, best_n = "Q2", -1
    for q in ["Q2", "Q1", "Q3", "Q4"]:
        if scene_path(city, year, q) is None:
            continue
        patches, _, _ = get_patch_ndvi(city, year, q, ps)
        n = len(patches)
        if n > best_n + 5:
            best_n, best_q = n, q
    return best_q

def _draw_panel(ax, thumb_info, patches, H, W, ps, cmap, norm, title):
    ax.set_facecolor("#0e1117")
    if thumb_info is not None:
        ax.imshow(thumb_info[0], origin="upper", extent=[0,W,H,0],
                  interpolation="bilinear", aspect="auto")
    if patches:
        n_rows, n_cols = H // ps, W // ps
        overlay = np.zeros((n_rows, n_cols, 4), np.float32)
        for (r, c), val in patches.items():
            if r < n_rows and c < n_cols:
                rgba = cmap(norm(val))
                overlay[r, c, :3] = rgba[:3]
                overlay[r, c, 3]  = 0.72
        ax.imshow(overlay, origin="upper", extent=[0,n_cols*ps,n_rows*ps,0],
                  interpolation="nearest", aspect="auto")
        for ri in range(n_rows + 1):
            ax.axhline(ri*ps, color="white", lw=0.3, alpha=0.3)
        for ci in range(n_cols + 1):
            ax.axvline(ci*ps, color="white", lw=0.3, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                color="gray", fontsize=9, transform=ax.transAxes)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_title(title, color="white", fontsize=10, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#555"); sp.set_linewidth(0.5)

def make_heatmap_fig(panels, cmap, norm, zmin, zmax):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5.5),
                             facecolor="#0e1117",
                             gridspec_kw={"wspace": 0.04})
    if n == 1: axes = [axes]
    for ax, p in zip(axes, panels):
        _draw_panel(ax, p["thumb"], p["patches"], p["H"], p["W"],
                    p["ps"], cmap, norm, p["title"])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.78, pad=0.02, aspect=22)
    cbar.ax.set_facecolor("#0e1117")
    cbar.set_ticks(np.linspace(zmin, zmax, 7))
    cbar.set_ticklabels(["No Change","Minimal","Patchy","Partial","Near Peak","Peak","Past Peak"])
    cbar.ax.tick_params(colors="white", labelsize=8)
    cbar.outline.set_edgecolor("#555")
    fig.patch.set_facecolor("#0e1117")
    return fig


@st.cache_data(show_spinner="Generating animation…")
def make_city_gif(city, use_intensity=True, ps=DISPLAY_PS):
    zmin, zmax = (-0.1, 0.5) if use_intensity else (-0.3, 0.7)
    cmap = DAMAGE_CMAP if use_intensity else NDVI_CMAP
    norm_g = Normalize(vmin=zmin, vmax=zmax)
    frames = []
    for yr in ALL_YEARS:
        q = find_best_quarter(city, yr)
        if scene_path(city, yr, q) is None:
            continue
        thumb = get_rgb_thumbnail(city, yr, q)
        if use_intensity:
            patches, H, W = get_damage(city, yr, q, ps)
        else:
            patches, H, W = get_patch_ndvi(city, yr, q, ps)
        if thumb:
            _, H, W = thumb
        fig, ax = plt.subplots(1, 1, figsize=(5, 5.4), facecolor="#0e1117")
        _draw_panel(ax, thumb, patches, H, W, ps, cmap, norm_g,
                    f"{city}  {yr}  ({q})")
        sm = ScalarMappable(cmap=cmap, norm=norm_g)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02, aspect=25)
        cbar.ax.tick_params(colors="white", labelsize=7)
        cbar.outline.set_edgecolor("#555")
        fig.tight_layout(pad=0.5)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, facecolor="#0e1117",
                    bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
    if not frames:
        return None
    w = max(f.size[0] for f in frames)
    h = max(f.size[1] for f in frames)
    frames = [f.resize((w, h), Image.LANCZOS) for f in frames]
    out = io.BytesIO()
    frames[0].save(out, format="GIF", save_all=True, append_images=frames[1:],
                   duration=1200, loop=0, optimize=False)
    out.seek(0)
    return out.read()

# ── Load dashboard data ───────────────────────────────────────────────────────
summary   = load_summary()
model_df  = load_model_comparison()
city_df   = load_per_city()
spectral_df = load_spectral()
pred_df   = load_predictions()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Satellite Intelligence Platform")
st.markdown(
    "Automated conflict-damage detection in Ukrainian cities using Sentinel-2 imagery "
    "(Mariupol · Kharkiv · Bakhmut · 2021–2024)"
)
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
(tab_overview, tab_models, tab_cities,
 tab_spectral, tab_preds, tab_heatmap, tab_anim) = st.tabs(
    ["Overview", "Model Comparison", "Per-City Analysis",
     "Spectral Trends", "Predictions", "Damage Heatmap", "City Animations"]
)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 · OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("### Best model performance on held-out test set")
    cols = st.columns(5)
    for col, (label, val) in zip(cols, [
        ("Accuracy",  summary["test_accuracy"]),
        ("Precision", summary["test_precision"]),
        ("Recall",    summary["test_recall"]),
        ("F1 Score",  summary["test_f1"]),
        ("AUC-ROC",   summary["test_auc_roc"]),
    ]):
        col.metric(label, f"{val:.1%}")
    st.divider()

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("#### Dataset split")
        split_data = pd.DataFrame({
            "Split": ["Train", "Validation", "Test"],
            "Patches": [summary["n_train"], summary["n_val"], summary["n_test"]],
        })
        fig = px.bar(split_data, x="Split", y="Patches", color="Split",
                     color_discrete_sequence=["#4C9BE8","#6CC24A","#E8844C"],
                     text="Patches")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Project at a glance")
        st.markdown(f"""
| Detail | Value |
|---|---|
| Best model | **{summary['best_model']}** |
| Features | {summary['n_features']} spectral indices |
| Cities | {', '.join(summary['cities_evaluated'])} |
| Training patches | {summary['n_train']} |
| Test patches | {summary['n_test']} |
| Decision threshold | {summary['optimal_threshold']} |
""")
        st.info("Labels: **0 = undamaged** (2021 baseline) · **1 = damaged** (2022–2024)")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 · MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
with tab_models:
    st.markdown("### Model comparison on test set")

    metric_col = st.selectbox(
        "Metric to compare",
        ["test_acc","test_f1","test_auc","test_precision","test_recall"],
        format_func=lambda x: {"test_acc":"Accuracy","test_f1":"F1 Score",
                               "test_auc":"AUC-ROC","test_precision":"Precision",
                               "test_recall":"Recall"}[x],
    )

    sorted_df = model_df.sort_values(metric_col, ascending=True)
    fig = px.bar(sorted_df, x=metric_col, y="model", orientation="h",
                 color=metric_col, color_continuous_scale="Blues",
                 text=sorted_df[metric_col].map(lambda v: f"{v:.3f}"),
                 labels={"model":"Model", metric_col: metric_col.replace("_"," ").title()})
    fig.update_traces(textposition="outside")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      coloraxis_showscale=False, height=340)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### All metrics side-by-side")

    models_filter = st.multiselect("Filter models", model_df["model"].tolist(),
                                   default=model_df["model"].tolist())
    disp = model_df[model_df["model"].isin(models_filter)].copy()
    for c in ["train_acc","val_acc","test_acc","test_precision","test_recall","test_f1","test_auc"]:
        disp[c] = disp[c].map(lambda v: f"{v:.4f}")
    disp.columns = ["Model","Train Acc","Val Acc","Test Acc","Precision","Recall","F1","AUC"]
    st.dataframe(disp.set_index("Model"), use_container_width=True)

    st.caption(
        "ResNet-50 was trained for fewer epochs due to compute constraints; "
        "classical ML models generalise well on 6 spectral features."
    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 · PER-CITY ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab_cities:
    st.markdown("### Per-city model performance")

    metric_city = st.radio(
        "Metric", ["accuracy","f1","auc_roc","precision","recall"],
        horizontal=True,
        format_func=lambda x: x.replace("_"," ").title(),
    )

    cities_sel = st.multiselect("Cities", CITIES, default=CITIES)
    city_filt = city_df[city_df["city"].isin(cities_sel)]

    fig = px.bar(city_filt.sort_values(metric_city, ascending=False),
                 x="city", y=metric_city, color="city",
                 color_discrete_map=CITY_COLORS,
                 text=city_filt.sort_values(metric_city, ascending=False)[metric_city].map(lambda v: f"{v:.3f}"),
                 labels={"city":"City", metric_city: metric_city.replace("_"," ").title()})
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", height=360, yaxis_range=[0,1.1])
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        fig_pie = px.pie(city_filt, values="n_scenes", names="city",
                         color="city", color_discrete_map=CITY_COLORS,
                         hole=0.4, title="Total scenes per city")
        fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        fig_dmg = px.bar(city_filt, x="city", y="n_damaged",
                         color="city", color_discrete_map=CITY_COLORS,
                         labels={"n_damaged":"Damaged scenes","city":"City"},
                         title="Damaged scenes per city")
        fig_dmg.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                               paper_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig_dmg, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 · SPECTRAL TRENDS
# ═════════════════════════════════════════════════════════════════════════════
with tab_spectral:
    st.markdown("### Spectral index trends over time")
    st.caption("NDVI (vegetation), NDBI (built-up), BSI (bare soil) averaged per city per quarter. Conflict onset: early 2022.")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 3])
    with col_ctrl1:
        index_choice = st.selectbox(
            "Spectral index",
            ["ndvi_mean","ndbi_mean","bsi_mean"],
            format_func=lambda x: {"ndvi_mean":"NDVI","ndbi_mean":"NDBI","bsi_mean":"BSI"}[x],
        )
    with col_ctrl2:
        year_range = st.slider("Year range", 2021, 2024, (2021, 2024))
    with col_ctrl3:
        city_filter = st.multiselect("Cities", spectral_df["city"].unique().tolist(),
                                     default=spectral_df["city"].unique().tolist())

    plot_df = spectral_df[
        spectral_df["city"].isin(city_filter) &
        spectral_df["year"].between(*year_range)
    ].copy()

    fig = px.line(plot_df, x="period_label", y=index_choice, color="city",
                  color_discrete_map=CITY_COLORS, markers=True,
                  labels={"period_label":"Period", index_choice: index_choice.replace("_"," ").upper()})
    if "2022 Q1" in plot_df["period_label"].values:
        fig.add_shape(type="line", x0="2022 Q1", x1="2022 Q1", y0=0, y1=1,
                      xref="x", yref="paper", line=dict(color="red", dash="dash", width=2))
        fig.add_annotation(x="2022 Q1", y=1.07, xref="x", yref="paper",
                           text="Conflict onset", showarrow=False,
                           font=dict(color="red", size=12))
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      height=420, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### All three indices — faceted by city")
    long_df = plot_df.melt(id_vars=["city","period_label"],
                           value_vars=["ndvi_mean","ndbi_mean","bsi_mean"],
                           var_name="index", value_name="value")
    long_df["index"] = long_df["index"].str.upper().str.replace("_MEAN","", regex=False)
    fig2 = px.line(long_df, x="period_label", y="value", color="index",
                   facet_col="city", facet_col_wrap=3, markers=True,
                   color_discrete_sequence=["#6CC24A","#4C9BE8","#E8844C"],
                   labels={"period_label":"Period","value":"Index value","index":"Index"})
    fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       height=380, xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 · PREDICTIONS
# ═════════════════════════════════════════════════════════════════════════════
with tab_preds:
    st.markdown("### Test-set prediction explorer")

    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 3])
    with ctrl1:
        threshold = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01)
    with ctrl2:
        city_pred_sel = st.multiselect("Cities", CITIES, default=CITIES, key="pred_cities")
    with ctrl3:
        show_only = st.radio("Show patches", ["All","Correct only","Errors only"],
                             horizontal=True)

    filt = pred_df[pred_df["city"].isin(city_pred_sel)].copy()
    filt["pred_dyn"] = (filt["damage_probability"] >= threshold).astype(int)
    filt["correct_dyn"] = (filt["pred_dyn"] == filt["damage_label"]).astype(int)
    if show_only == "Correct only":
        filt = filt[filt["correct_dyn"] == 1]
    elif show_only == "Errors only":
        filt = filt[filt["correct_dyn"] == 0]

    # Dynamic metrics at chosen threshold
    y_true = filt["damage_label"]
    y_pred = filt["pred_dyn"]
    tp = int(((y_pred==1)&(y_true==1)).sum())
    tn = int(((y_pred==0)&(y_true==0)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
    acc  = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn) else 0
    prec = tp/(tp+fp) if (tp+fp) else 0
    rec  = tp/(tp+fn) if (tp+fn) else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{acc:.1%}")
    m2.metric("Precision", f"{prec:.1%}")
    m3.metric("Recall",    f"{rec:.1%}")
    m4.metric("F1",        f"{f1:.3f}")
    m5.metric("Errors", f"{fp+fn} / {len(filt)}")

    st.divider()

    # Confusion matrix heatmap
    col_cm, col_hist = st.columns(2)
    with col_cm:
        st.markdown("#### Confusion matrix")
        cm_fig = go.Figure(go.Heatmap(
            z=[[tn,fp],[fn,tp]],
            x=["Pred: Undamaged","Pred: Damaged"],
            y=["True: Undamaged","True: Damaged"],
            text=[[str(tn),str(fp)],[str(fn),str(tp)]],
            texttemplate="%{text}", textfont={"size":18},
            colorscale="Blues", showscale=False,
        ))
        cm_fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              height=300, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(cm_fig, use_container_width=True)

    with col_hist:
        st.markdown("#### Damage probability distribution")
        fig_hist = px.histogram(filt, x="damage_probability", color="city",
                                color_discrete_map=CITY_COLORS, nbins=30,
                                barmode="overlay", opacity=0.75,
                                labels={"damage_probability":"Damage probability","city":"City"})
        fig_hist.add_vline(x=threshold, line_dash="dash", line_color="white",
                           annotation_text=f"Threshold {threshold:.2f}",
                           annotation_position="top right")
        fig_hist.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        st.markdown("#### Damage score vs probability")
        fig_sc = px.scatter(filt, x="damage_score", y="damage_probability",
                            color="city", symbol="correct_dyn",
                            color_discrete_map=CITY_COLORS,
                            symbol_map={1:"circle",0:"x"},
                            opacity=0.75,
                            labels={"damage_score":"Raw damage score",
                                    "damage_probability":"Predicted probability",
                                    "city":"City","correct_dyn":"Correct?"})
        fig_sc.add_hline(y=threshold, line_dash="dash", line_color="white")
        fig_sc.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=360)
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_sc2:
        st.markdown("#### NDVI vs NDBI")
        fig_ndvi = px.scatter(filt, x="ndvi_mean", y="ndbi_mean",
                              color="correct_dyn",
                              color_continuous_scale=["#E8844C","#4C9BE8"],
                              facet_col="city", opacity=0.8,
                              labels={"ndvi_mean":"NDVI mean","ndbi_mean":"NDBI mean",
                                      "correct_dyn":"Correct"},
                              hover_data=["period","damage_probability"])
        fig_ndvi.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=360)
        st.plotly_chart(fig_ndvi, use_container_width=True)

    with st.expander("Raw prediction data"):
        st.dataframe(filt[["city","period","damage_label","pred_dyn",
                            "damage_probability","damage_score",
                            "ndvi_mean","ndbi_mean","correct_dyn"]],
                     use_container_width=True, height=300)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 · DAMAGE HEATMAP
# ═════════════════════════════════════════════════════════════════════════════
with tab_heatmap:
    st.markdown("### Satellite damage heatmap")
    st.caption(
        "Sentinel-2 imagery as background. Coloured cells are 64×64 px patches (~1.28 km²). "
        "Green = healthy vegetation, dark brown = severe damage."
    )

    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        mode = st.radio("Temporal mode", ["Yearly (best quarter)", "Quarterly (raw scenes)"])
    with h_col2:
        view = st.radio("View", ["City Through Time", "Cities Compared"])
    with h_col3:
        metric_label = st.selectbox(
            "Metric",
            ["Damage Intensity (vs 2021 baseline)", "Raw NDVI"]
            if mode == "Quarterly (raw scenes)"
            else ["Damage Intensity (NDVI change)", "Raw NDVI"],
        )

    use_intensity = metric_label.startswith("Damage")
    cmap  = DAMAGE_CMAP if use_intensity else NDVI_CMAP
    zmin, zmax = (-0.1, 0.5) if use_intensity else (-0.3, 0.7)
    norm  = Normalize(vmin=zmin, vmax=zmax)

    st.markdown(
        "**Color scale:** "
        "Green = No change / healthy  |  Yellow = Partial  |  "
        "Red = Peak damage  |  Dark brown = Past peak / severe"
    )
    st.divider()

    if mode == "Yearly (best quarter)":
        if view == "City Through Time":
            city_sel = st.selectbox("City", CITIES, key="hm_city_yr")
            panels = []
            for yr in ALL_YEARS:
                q_bg = find_best_quarter(city_sel, yr)
                thumb = get_rgb_thumbnail(city_sel, yr, q_bg)
                patches, H, W = (get_damage(city_sel, yr, q_bg) if use_intensity
                                 else get_patch_ndvi(city_sel, yr, q_bg))
                if thumb: _, H, W = thumb
                panels.append(dict(thumb=thumb, patches=patches, H=H, W=W,
                                   ps=DISPLAY_PS, title=f"{yr} ({q_bg})"))
            fig = make_heatmap_fig(panels, cmap, norm, zmin, zmax)
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        else:
            yr_sel = st.selectbox("Year", ALL_YEARS, index=1, key="hm_yr_cmp")
            panels = []
            for city in CITIES:
                q_bg = find_best_quarter(city, yr_sel)
                thumb = get_rgb_thumbnail(city, yr_sel, q_bg)
                patches, H, W = (get_damage(city, yr_sel, q_bg) if use_intensity
                                 else get_patch_ndvi(city, yr_sel, q_bg))
                if thumb: _, H, W = thumb
                panels.append(dict(thumb=thumb, patches=patches, H=H, W=W,
                                   ps=DISPLAY_PS, title=f"{city} ({q_bg})"))
            fig = make_heatmap_fig(panels, cmap, norm, zmin, zmax)
            st.pyplot(fig, use_container_width=True); plt.close(fig)

    else:  # Quarterly
        if view == "City Through Time":
            hc1, hc2 = st.columns(2)
            city_sel = hc1.selectbox("City", CITIES, key="hm_city_q")
            yr_sel   = hc2.selectbox("Year", ALL_YEARS, index=1, key="hm_yr_q")
            st.caption("Q1 = Jan–Mar · Q2 = Apr–Jun · Q3 = Jul–Sep · Q4 = Oct–Dec")
            panels = []
            for q in QUARTERS:
                thumb = get_rgb_thumbnail(city_sel, yr_sel, q)
                patches, H, W = (get_damage(city_sel, yr_sel, q) if use_intensity
                                 else get_patch_ndvi(city_sel, yr_sel, q))
                if thumb: _, H, W = thumb
                panels.append(dict(thumb=thumb, patches=patches, H=H, W=W,
                                   ps=DISPLAY_PS, title=q))
            fig = make_heatmap_fig(panels, cmap, norm, zmin, zmax)
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        else:
            hc1, hc2 = st.columns(2)
            yr_sel = hc1.selectbox("Year", ALL_YEARS, index=1, key="hm_yr_qcmp")
            q_sel  = hc2.selectbox("Quarter", QUARTERS, index=1, key="hm_q_cmp")
            panels = []
            for city in CITIES:
                thumb = get_rgb_thumbnail(city, yr_sel, q_sel)
                patches, H, W = (get_damage(city, yr_sel, q_sel) if use_intensity
                                 else get_patch_ndvi(city, yr_sel, q_sel))
                if thumb: _, H, W = thumb
                panels.append(dict(thumb=thumb, patches=patches, H=H, W=W,
                                   ps=DISPLAY_PS, title=city))
            fig = make_heatmap_fig(panels, cmap, norm, zmin, zmax)
            st.pyplot(fig, use_container_width=True); plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 · CITY ANIMATIONS
# ═════════════════════════════════════════════════════════════════════════════
with tab_anim:
    st.markdown("### Temporal damage evolution — animated")
    st.caption(
        "Each frame shows the best-coverage quarter for that year (2021 → 2024), "
        "rendered at the same scale as the heatmap panels. "
        "First generation takes ~20–30 s per city; results are cached for the session."
    )

    anim_metric = st.radio(
        "Metric",
        ["Damage Intensity (vs 2021 baseline)", "Raw NDVI"],
        horizontal=True,
        key="anim_metric",
    )
    use_intensity_anim = anim_metric.startswith("Damage")
    st.divider()

    a_cols = st.columns(3)
    for col, city in zip(a_cols, CITIES):
        with col:
            st.markdown(f"**{city}**")
            gif_bytes = make_city_gif(city, use_intensity=use_intensity_anim)
            if gif_bytes:
                st.image(gif_bytes, use_container_width=True)
            else:
                st.warning("No scene data available.")
