"""
Explainable Brains — Challenge B Dashboard
==========================================

A Streamlit dashboard for exploring c-Fos activity differences between
Semaglutide-treated and Vehicle (control) mice.

Three sections:
  1. Chat — ask plain-English questions, get answers grounded in the data
  2. Volcano plot — every brain region's effect size vs significance
  3. Top regions table — filterable, sortable

Run:
    streamlit run app.py

Requires:
    - data_b/statistics.csv and data_b/quantification.csv
      (auto-downloaded on first run via challenge_b_agent.load_data)
    - ANTHROPIC_API_KEY env var (only for the chat box; charts work without it)
"""

import asyncio
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
)
from claude_agent_sdk import query as claude_query

from bucket_access.bucket_utils import download_file
from challenge_b_agent import load_data, run_agent_turn


# ────────────────────────────────────────────────────────────────────────────
# Page configuration
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explainable Brains — Semaglutide",
    page_icon="🧠",
    layout="wide",
)


# ────────────────────────────────────────────────────────────────────────────
# Cached resources — data and Anthropic client load once per session
# ────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading brain data…")
def load_cached_data():
    """Download (if needed) and load the two CSVs from the hackathon bucket."""
    return load_data()


@st.cache_resource
def get_anthropic_client():
    """Return an Anthropic client if ANTHROPIC_API_KEY is set, else None.

    Kept for fallback; the main chat path now uses subscription auth via the
    Claude Agent SDK (see ask_via_subscription below), which avoids needing
    an API key entirely.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic
    return anthropic.Anthropic()


@st.cache_data
def build_chat_system_prompt(_stats_df: pd.DataFrame) -> str:
    """
    Build the chat system prompt with the top differentially activated regions
    embedded inline. The underscore on `_stats_df` tells Streamlit not to hash
    the DataFrame for cache invalidation.
    """
    top = (
        _stats_df
        .dropna(subset=["log2_fold_change", "p_value"])
        .sort_values("p_value")
        .head(80)
        .copy()
    )

    rows = []
    for _, r in top.iterrows():
        acronym = str(r.get("acronym", ""))[:10]
        name = str(r.get("region_name", ""))[:42]
        rows.append(
            f"  {acronym:10s} | {name:42s} | "
            f"log2FC={r['log2_fold_change']:+.2f} | p={r['p_value']:.4f}"
        )
    data_block = "\n".join(rows)

    return f"""You are a neuroscience research assistant for an Ozempic / Semaglutide brain-imaging study.

Experimental setup:
- G001 (Vehicle): control mice given saline
- G002 (Semaglutide): mice given Semaglutide (active ingredient in Ozempic / Wegovy)
- Marker: c-Fos, expressed by recently active neurons — used as a proxy for neuronal activation
- 1,356 brain regions analyzed total

Below are the 80 regions with the strongest evidence of differential activation
(sorted by uncorrected p-value, smallest first):

```
  acronym    | region_name                                | log2FC      | p
{data_block}
```

How to read this:
- Positive log2FC = MORE active in Semaglutide
- Negative log2FC = LESS active in Semaglutide
- With 1356 tests, multiple-testing correction wipes out everything, so we use
  uncorrected p<0.05 as the working threshold

When answering the user's question:
1. Use your neuroanatomy knowledge to identify which of the regions ABOVE are
   relevant (do not invent regions that aren't in the table).
2. Cite specific acronyms with their numbers — e.g., "the NTS (nucleus of the
   solitary tract) is upregulated, log2FC = +2.77, p = 0.006".
3. Connect findings to Semaglutide's mechanism (gut→brainstem satiety signaling,
   amygdalar food valuation, reward dampening).
4. Be concise — 3 to 5 sentences for most questions.
5. If a region the user asks about isn't in the table, say so honestly."""


async def _ask_subscription_async(prompt: str, system_prompt: str) -> str:
    """Run one chat turn through the Claude Agent SDK using subscription auth.

    We explicitly disable all built-in tools (Bash, FileRead, WebFetch, etc.)
    because this is a pure chat — Claude shouldn't be touching the filesystem
    or running shell commands while answering a question about brain regions.
    """
    opts = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=[],
    )
    collected: list[str] = []
    seen_types: list[str] = []
    async for msg in claude_query(prompt=prompt, options=opts):
        seen_types.append(type(msg).__name__)
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    collected.append(block.text)

    text = "".join(collected).strip()
    if not text:
        # Surface what came back so we can debug
        return f"_(No text response. SDK returned message types: {seen_types})_"
    return text


def ask_via_subscription(prompt: str, system_prompt: str) -> str:
    """
    Sync wrapper so Streamlit (which doesn't natively await) can call the
    subscription-auth chat without ceremony. Each call spins up a fresh event
    loop — fine because the response is one-shot per user turn.
    """
    return asyncio.run(_ask_subscription_async(prompt, system_prompt))


@st.cache_data(show_spinner="Loading brain maps (one-time download, ~30 sec)…")
def load_cached_spatial_data():
    """
    Download (first run only) and load the four spatial brain-map files into
    numpy arrays + a pandas DataFrame. Results are cached, so the second-and-
    onwards page renders are instant.

    Returns:
        anatomy:        (Z, Y, X) uint16 grayscale structural brain
        region_volume:  (Z, Y, X) int   atlas region label per voxel
        diff_map:       (Z, Y, X) float c-Fos signal: Semaglutide − Vehicle
        atlas:          DataFrame mapping integer label → acronym, region name
    """
    import SimpleITK as sitk  # imported lazily — heavy native dep

    spatial_dir = "data_b/spatial"
    os.makedirs(spatial_dir, exist_ok=True)

    file_map = [
        ("challengeB/spatial_brain_maps/brain_atlas_anatomy.nii.gz",
         f"{spatial_dir}/anatomy.nii.gz"),
        ("challengeB/spatial_brain_maps/brain_atlas_regions.nii.gz",
         f"{spatial_dir}/regions.nii.gz"),
        ("challengeB/spatial_brain_maps/cfos_group_median_difference_G002_vs_G001.nii.gz",
         f"{spatial_dir}/diff.nii.gz"),
        ("challengeB/spatial_brain_maps/atlas_regions.csv",
         f"{spatial_dir}/atlas_regions.csv"),
    ]
    for s3_key, local_path in file_map:
        if not os.path.exists(local_path):
            download_file(s3_key, local_path)

    anatomy       = sitk.GetArrayFromImage(sitk.ReadImage(f"{spatial_dir}/anatomy.nii.gz"))
    region_volume = sitk.GetArrayFromImage(sitk.ReadImage(f"{spatial_dir}/regions.nii.gz")).astype(int)
    diff_map      = sitk.GetArrayFromImage(sitk.ReadImage(f"{spatial_dir}/diff.nii.gz"))
    atlas         = pd.read_csv(f"{spatial_dir}/atlas_regions.csv")
    return anatomy, region_volume, diff_map, atlas


def render_brain_slice(
    anatomy: np.ndarray,
    diff_map: np.ndarray,
    region_volume: np.ndarray,
    axis: str,
    slice_idx: int,
    region_label: int | None = None,
):
    """
    Compose a brain slice figure with three layers:
      1. Anatomy in grayscale (base)
      2. Difference map (Semaglutide − Vehicle) as a diverging red/blue overlay
      3. Optional cyan outline around a selected anatomical region

    Args:
        anatomy:        (Z, Y, X) grayscale structural volume
        diff_map:       (Z, Y, X) signed difference volume
        region_volume:  (Z, Y, X) integer atlas labels per voxel
        axis:           'coronal' | 'sagittal' | 'axial'
        slice_idx:      index along the chosen axis
        region_label:   integer atlas label to outline, or None for no outline

    Returns:
        matplotlib Figure ready for st.pyplot()
    """
    # Slice along the chosen anatomical axis. SimpleITK delivers (Z, Y, X).
    if axis == "coronal":
        anat_slice    = anatomy[:, slice_idx, :]
        diff_slice    = diff_map[:, slice_idx, :]
        regions_slice = region_volume[:, slice_idx, :]
    elif axis == "sagittal":
        anat_slice    = anatomy[:, :, slice_idx]
        diff_slice    = diff_map[:, :, slice_idx]
        regions_slice = region_volume[:, :, slice_idx]
    else:  # axial
        anat_slice    = anatomy[slice_idx, :, :]
        diff_slice    = diff_map[slice_idx, :, :]
        regions_slice = region_volume[slice_idx, :, :]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # Layer 1 — anatomy (grayscale base)
    ax.imshow(anat_slice, cmap="gray", aspect="equal", origin="lower")

    # Layer 2 — difference map. Symmetric color range so 0 sits at the
    # center of the colormap. Mask tiny values so the anatomy shows through
    # where Semaglutide and Vehicle agree.
    if np.any(diff_map != 0):
        vmax = float(np.percentile(np.abs(diff_map[diff_map != 0]), 99))
    else:
        vmax = 1.0
    threshold = vmax * 0.05
    diff_masked = np.ma.masked_where(np.abs(diff_slice) < threshold, diff_slice)
    im = ax.imshow(
        diff_masked,
        cmap="RdBu_r",
        alpha=0.65,
        vmin=-vmax, vmax=vmax,
        aspect="equal",
        origin="lower",
    )

    # Layer 3 — region outline (optional)
    if region_label is not None:
        mask = regions_slice == region_label
        if mask.any():
            ax.contour(mask, levels=[0.5], colors="cyan", linewidths=1.8)

    # Colorbar, styled for the dark theme
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("c-Fos: Semaglutide − Vehicle", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.outline.set_edgecolor("white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ax.set_xticks([])
    ax.set_yticks([])
    return fig


# ────────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────────
st.title("🧠 Explainable Brains")
st.markdown(
    """
    **Where does Semaglutide activate the brain?**
    Semaglutide is the active ingredient in Ozempic and Wegovy. This dashboard
    explores how it changes neuronal activity (c-Fos) across hundreds of brain
    regions in mice, compared to a saline control. Ask questions in plain
    English, or browse the interactive charts below.
    """
)

# Load data — fail loudly if the CSVs aren't downloadable
try:
    quant_df, stats_df = load_cached_data()
except Exception as e:
    st.error(f"Could not load data. Did you run `python explore_challenge_b.py` first?\n\n{e}")
    st.stop()

# Some regions failed the statistical test (e.g. no animals had that region), so
# `significant_corrected` comes in as float64 with NaN values instead of clean
# True/False. Coerce to bool (treating NaN as False) so .map() and boolean
# indexing both work downstream.
stats_df = stats_df.copy()
for _col in ("significant_corrected", "significant_uncorrected"):
    if _col in stats_df.columns:
        stats_df[_col] = stats_df[_col].fillna(False).astype(bool)


# ────────────────────────────────────────────────────────────────────────────
# Section 1 — Chat
# ────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("💬 Ask the data")
st.caption(
    "Ask anything about which brain regions Semaglutide affects. "
    "Try: *Which regions handle hunger and satiety?* or *Top 5 most significant regions.* "
    "Powered by Claude through your Claude Code subscription — no API key needed."
)

# Build (and cache) the system prompt with embedded study data
system_prompt = build_chat_system_prompt(stats_df)

# Persist a simple [{role, content}] chat log across Streamlit reruns
if "display_history" not in st.session_state:
    st.session_state.display_history = []

# Render past turns
for entry in st.session_state.display_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

user_input = st.chat_input(
    "Ask about a brain region, a group difference, or what Semaglutide affects…"
)

if user_input:
    # Show the user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.display_history.append({"role": "user", "content": user_input})

    # Run the chat via subscription auth (no API key, no API credits used)
    with st.chat_message("assistant"):
        with st.spinner("Asking Claude…"):
            try:
                response_text = ask_via_subscription(user_input, system_prompt)
            except Exception as e:
                response_text = (
                    f"_Chat failed: `{e}`. Charts below still work. "
                    "If this persists, make sure Claude Code is logged in on this machine._"
                )
        st.markdown(response_text)
    st.session_state.display_history.append(
        {"role": "assistant", "content": response_text}
    )


# ────────────────────────────────────────────────────────────────────────────
# Section 2 — Volcano plot
# ────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("📊 Volcano plot — all brain regions")
st.caption(
    "Each dot is one of 1,356 brain regions. "
    "**X-axis** = effect size (positive → more active under Semaglutide). "
    "**Y-axis** = statistical significance. "
    "**Red** = p<0.05 (uncorrected). Hover to see region details. "
    "*Multiple-testing correction across 1,356 tests yields zero corrected-significant regions — "
    "we use uncorrected p-values to surface candidate hits.*"
)

# Build the plot data
# - Drop regions where the statistical test failed (NaN p-value or fold change)
# - Use UNCORRECTED p-value: with 1,356 tests the corrected threshold is so harsh
#   that no region clears it, so corrected significance is uninformative here.
# - Clamp p away from 0 so -log10 stays finite
df_plot = stats_df.dropna(subset=["p_value", "log2_fold_change"]).copy()
df_plot["neg_log_p"] = -np.log10(df_plot["p_value"].clip(lower=1e-300))
df_plot["category"] = df_plot["significant_uncorrected"].map(
    {True: "p < 0.05", False: "n.s."}
)

fig = px.scatter(
    df_plot,
    x="log2_fold_change",
    y="neg_log_p",
    color="category",
    color_discrete_map={"p < 0.05": "#e74c3c", "n.s.": "#95a5a6"},
    category_orders={"category": ["p < 0.05", "n.s."]},
    hover_data={
        "acronym": True,
        "region_name": True,
        "log2_fold_change": ":.2f",
        "p_value": ":.4f",
        "neg_log_p": False,
        "category": False,
    },
    labels={
        "log2_fold_change": "log₂ Fold Change (Semaglutide / Vehicle)",
        "neg_log_p": "-log₁₀(p uncorrected)",
        "category": "Significance",
    },
)
fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="DarkSlateGrey")))
fig.update_layout(height=520, hovermode="closest")
fig.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.5)
st.plotly_chart(fig, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# Section 3 — Top regions table
# ────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🔍 Top regions")

control_col, table_col = st.columns([1, 3])

with control_col:
    direction = st.radio(
        "Sort by",
        ["Most upregulated", "Most downregulated", "Most significant"],
        index=0,
    )
    n_rows = st.slider("Number of regions", min_value=5, max_value=50, value=20)
    only_sig = st.checkbox("Significant only (uncorrected p < 0.05)", value=True)

with table_col:
    df_table = stats_df.copy()
    if only_sig:
        df_table = df_table[df_table["significant_uncorrected"]]

    if direction == "Most upregulated":
        df_table = df_table.nlargest(n_rows, "log2_fold_change")
    elif direction == "Most downregulated":
        df_table = df_table.nsmallest(n_rows, "log2_fold_change")
    else:  # Most significant
        df_table = df_table.nsmallest(n_rows, "p_value")

    visible_cols = [
        "acronym", "region_name", "log2_fold_change", "fold_change",
        "p_value", "significant_uncorrected", "mean_A", "mean_B",
    ]
    st.dataframe(
        df_table[visible_cols].round(4),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")
st.caption(
    "Built for the Explainable Brains Hackathon · "
    "Data from Vibraint · "
    "Powered by Claude"
)
