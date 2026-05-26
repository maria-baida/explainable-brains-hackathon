"""
Challenge B — Brain Data Dashboard
Run with: streamlit run dashboard.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from bucket_access.bucket_utils import download_file

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Data Explorer",
    page_icon="🧠",
    layout="wide",
)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_data():
    os.makedirs("data_b", exist_ok=True)

    quant_path = "data_b/quantification.csv"
    stats_path = "data_b/statistics.csv"

    if not os.path.exists(quant_path):
        download_file(
            "challengeB/tabular_data_quantification/cfos_object_density_quantification.csv",
            quant_path,
        )
    if not os.path.exists(stats_path):
        download_file(
            "challengeB/tabular_data_quantification/cfos_object_density_statistics_G002_vs_G001.csv",
            stats_path,
        )

    quant = pd.read_csv(quant_path)
    stats = pd.read_csv(stats_path)
    return quant, stats


quant, stats = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")

sig_only = st.sidebar.checkbox("Show significant regions only (corrected p < 0.05)", value=False)
hier_range = st.sidebar.slider(
    "Hierarchy level (specificity)",
    int(stats["hierarchy_level"].min()),
    int(stats["hierarchy_level"].max()),
    (int(stats["hierarchy_level"].min()), int(stats["hierarchy_level"].max())),
)
top_n = st.sidebar.slider("Label top N hits on volcano", 0, 30, 10)
search_query = st.sidebar.text_input("Search region name / acronym", "")

# ── Apply filters ─────────────────────────────────────────────────────────────
df = stats.copy()
if sig_only:
    df = df[df["significant_corrected"]]
df = df[df["hierarchy_level"].between(*hier_range)]
if search_query:
    q = search_query.lower()
    df = df[
        df["region_name"].str.lower().str.contains(q, na=False)
        | df["acronym"].str.lower().str.contains(q, na=False)
    ]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 Brain Data Explorer")
st.caption(
    "c-Fos neuronal activation · Semaglutide (G002) vs Vehicle (G001) · Mouse brain"
)

# ── KPI row ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Brain regions", len(df))
col2.metric("Significant (corrected)", int(df["significant_corrected"].sum()))
col3.metric("Upregulated in Sema", int((df["log2_fold_change"] > 0).sum()))
col4.metric("Downregulated in Sema", int((df["log2_fold_change"] < 0).sum()))

st.divider()

# ── Volcano plot ──────────────────────────────────────────────────────────────
st.subheader("Volcano plot — fold change vs significance")

df["neg_log10_p"] = -np.log10(df["p_corrected"].clip(lower=1e-300))
df["sig_label"] = df["significant_corrected"].map(
    {True: "Significant (corrected)", False: "Not significant"}
)

color_map = {"Significant (corrected)": "#e05c5c", "Not significant": "#aaaaaa"}

fig_volcano = px.scatter(
    df,
    x="log2_fold_change",
    y="neg_log10_p",
    color="sig_label",
    color_discrete_map=color_map,
    hover_name="region_name",
    hover_data={
        "acronym": True,
        "log2_fold_change": ":.3f",
        "p_corrected": ":.4f",
        "sig_label": False,
        "neg_log10_p": False,
    },
    labels={
        "log2_fold_change": "log₂ Fold Change  (Semaglutide / Vehicle)",
        "neg_log10_p": "−log₁₀(p corrected)",
        "sig_label": "",
    },
    opacity=0.7,
)

# Add vertical zero line
fig_volcano.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)

# Label top N hits
if top_n > 0:
    top_up = df.nlargest(top_n // 2 + top_n % 2, "log2_fold_change")
    top_down = df.nsmallest(top_n // 2, "log2_fold_change")
    to_label = pd.concat([top_up, top_down])
    for _, row in to_label.iterrows():
        fig_volcano.add_annotation(
            x=row["log2_fold_change"],
            y=row["neg_log10_p"],
            text=row["acronym"],
            showarrow=False,
            font=dict(size=9),
            xshift=8,
        )

fig_volcano.update_layout(
    height=480,
    legend=dict(orientation="h", y=-0.15),
    margin=dict(t=20, b=40),
)
st.plotly_chart(fig_volcano, use_container_width=True)

st.divider()

# ── Top regions bar chart ─────────────────────────────────────────────────────
st.subheader("Top differentially activated regions")

tab_up, tab_down, tab_both = st.tabs(["⬆ Upregulated", "⬇ Downregulated", "Both (by |FC|)"])

def region_bar(subset_df, title_suffix):
    region_cols = [c for c in quant.columns if c not in ["scan_name", "animal_nr", "group_nr"]]
    available = [r for r in subset_df["acronym"].tolist() if r in region_cols]
    if not available:
        st.info("No matching regions found in quantification data.")
        return

    long = quant.melt(
        id_vars=["group_nr"],
        value_vars=available,
        var_name="region",
        value_name="density",
    )
    means = long.groupby(["region", "group_nr"])["density"].mean().reset_index()
    means["group"] = means["group_nr"].map({"G001": "Vehicle (G001)", "G002": "Semaglutide (G002)"})
    order = available  # preserve ranking order

    fig = px.bar(
        means,
        x="region",
        y="density",
        color="group",
        barmode="group",
        category_orders={"region": order},
        color_discrete_map={
            "Vehicle (G001)": "#4c8fc4",
            "Semaglutide (G002)": "#e05c5c",
        },
        labels={"density": "Mean c-Fos density (cells/mm³)", "region": "Brain region"},
    )
    fig.update_layout(height=380, margin=dict(t=10, b=40), legend_title="")
    st.plotly_chart(fig, use_container_width=True)

with tab_up:
    n_bar = st.slider("Show top N regions", 5, 30, 15, key="up_n")
    top = df.nlargest(n_bar, "log2_fold_change")
    region_bar(top, "upregulated")

with tab_down:
    n_bar2 = st.slider("Show top N regions", 5, 30, 15, key="down_n")
    top2 = df.nsmallest(n_bar2, "log2_fold_change")
    region_bar(top2, "downregulated")

with tab_both:
    n_bar3 = st.slider("Show top N regions", 5, 30, 20, key="both_n")
    top3 = df.reindex(df["log2_fold_change"].abs().nlargest(n_bar3).index)
    region_bar(top3, "most changed")

st.divider()

# ── Data table ────────────────────────────────────────────────────────────────
st.subheader("Region statistics table")

display_cols = [
    "acronym", "region_name", "hierarchy_level",
    "log2_fold_change", "fold_change", "p_corrected",
    "significant_corrected", "mean_A", "mean_B",
]
st.dataframe(
    df[display_cols]
    .sort_values("log2_fold_change", ascending=False)
    .rename(columns={
        "mean_A": "mean_Sema (G002)",
        "mean_B": "mean_Vehicle (G001)",
        "significant_corrected": "significant",
    })
    .style.background_gradient(subset=["log2_fold_change"], cmap="RdBu_r"),
    use_container_width=True,
    height=400,
)
