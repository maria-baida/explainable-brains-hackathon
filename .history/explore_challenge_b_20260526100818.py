"""
Challenge B — Quick exploration starter
Run this to see what the data looks like before building your solution.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from bucket_access.bucket_utils import list_files, download_file

# ── 1. List available files ──────────────────────────────────────────────────
print("=== Files in challengeB/ ===")
for f in list_files("challengeB/"):
    print(" ", f)

# ── 2. Download tabular data ─────────────────────────────────────────────────
os.makedirs("data_b", exist_ok=True)
print("\nDownloading tabular data...")
download_file(
    "challengeB/tabular_data_quantification/cfos_object_density_quantification.csv",
    "data_b/quantification.csv"
)
download_file(
    "challengeB/tabular_data_quantification/cfos_object_density_statistics_G002_vs_G001.csv",
    "data_b/statistics.csv"
)

# ── 3. Load and explore ───────────────────────────────────────────────────────
quant = pd.read_csv("data_b/quantification.csv")
stats = pd.read_csv("data_b/statistics.csv")

print(f"\n=== Quantification data ===")
print(f"  Shape: {quant.shape}  ({quant['group_nr'].value_counts().to_dict()})")
print(quant[["scan_name", "animal_nr", "group_nr"]].head(6).to_string(index=False))

print(f"\n=== Statistics data ===")
print(f"  Shape: {stats.shape}")
print(f"  Columns: {list(stats.columns)}")
sig_corrected = stats[stats["significant_corrected"]].shape[0]
sig_uncorrected = stats[stats["significant_uncorrected"]].shape[0]
print(f"\n  Significant (corrected p<0.05):   {sig_corrected} regions")
print(f"  Significant (uncorrected p<0.05): {sig_uncorrected} regions")

# ── 4. Top differentially activated regions ───────────────────────────────────
print("\n=== Top 10 UPREGULATED in Semaglutide (G002) ===")
up = stats.nlargest(10, "log2_fold_change")[
    ["acronym", "region_name", "log2_fold_change", "p_corrected", "significant_corrected"]
]
print(up.to_string(index=False))

print("\n=== Top 10 DOWNREGULATED in Semaglutide (G002) ===")
down = stats.nsmallest(10, "log2_fold_change")[
    ["acronym", "region_name", "log2_fold_change", "p_corrected", "significant_corrected"]
]
print(down.to_string(index=False))

# ── 5. Volcano plot ───────────────────────────────────────────────────────────
print("\nGenerating volcano plot...")

fig, ax = plt.subplots(figsize=(10, 7))

# colour by significance
not_sig = stats[~stats["significant_corrected"]]
sig = stats[stats["significant_corrected"]]

ax.scatter(not_sig["log2_fold_change"], -np.log10(not_sig["p_corrected"]),
           s=10, alpha=0.4, color="grey", label="Not significant")
ax.scatter(sig["log2_fold_change"], -np.log10(sig["p_corrected"]),
           s=20, alpha=0.8, color="tomato", label=f"Significant (n={len(sig)})")

# label top hits
for _, row in stats.nlargest(8, "log2_fold_change").iterrows():
    ax.annotate(row["acronym"], (row["log2_fold_change"], -np.log10(row["p_corrected"])),
                fontsize=7, ha="left", xytext=(4, 0), textcoords="offset points")
for _, row in stats.nsmallest(5, "log2_fold_change").iterrows():
    ax.annotate(row["acronym"], (row["log2_fold_change"], -np.log10(row["p_corrected"])),
                fontsize=7, ha="right", xytext=(-4, 0), textcoords="offset points")

ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("log₂ Fold Change  (Semaglutide / Vehicle)", fontsize=11)
ax.set_ylabel("-log₁₀(p corrected)", fontsize=11)
ax.set_title("Volcano plot — c-Fos density: Semaglutide vs Vehicle\nacross all brain regions", fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("challenge_b_volcano.png", dpi=130)
print("  Saved: challenge_b_volcano.png")

# ── 6. Distribution of c-Fos by region (top 20 significant regions) ──────────
top_regions = stats[stats["significant_corrected"]].nlargest(20, "log2_fold_change")["acronym"].tolist()
region_cols = [c for c in quant.columns if c in top_regions]

if region_cols:
    long_df = quant.melt(
        id_vars=["group_nr"], value_vars=region_cols,
        var_name="region", value_name="density"
    )
    g001 = long_df[long_df["group_nr"] == "G001"].groupby("region")["density"].mean()
    g002 = long_df[long_df["group_nr"] == "G002"].groupby("region")["density"].mean()
    merged = pd.DataFrame({"Vehicle (G001)": g001, "Semaglutide (G002)": g002}).dropna()
    merged = merged.loc[merged.index.isin(region_cols)]

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    x = np.arange(len(merged))
    ax2.bar(x - 0.2, merged["Vehicle (G001)"], 0.35, label="Vehicle (G001)", color="steelblue", alpha=0.8)
    ax2.bar(x + 0.2, merged["Semaglutide (G002)"], 0.35, label="Semaglutide (G002)", color="tomato", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(merged.index, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Mean c-Fos density (cells/mm³)")
    ax2.set_title("Mean c-Fos density — top significantly upregulated regions")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("challenge_b_top_regions.png", dpi=130)
    print("  Saved: challenge_b_top_regions.png")

print("\nDone! Open challenge_b_volcano.png to see the full picture.")
print("\nNext steps for Challenge B:")
print("  1. Run `python challenge_b_agent.py` to chat with Claude about the data")
print("  2. Build a Streamlit/Dash dashboard using the volcano + bar plots above")
print("  3. Add NIfTI spatial brain maps for the coolest visual extension")
