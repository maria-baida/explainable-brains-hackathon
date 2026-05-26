"""
Challenge A — Quick exploration starter
Run this to see what the data looks like before building your solution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from bucket_access.bucket_utils import list_files, read_h5_patches, read_h5_embeddings

# ── 1. List available files ──────────────────────────────────────────────────
print("=== Files in challengeA/ ===")
files = list_files("challengeA/")
for f in files[:20]:
    print(" ", f)

patch_files = [f for f in files if f.endswith("_patches.h5")]
emb_files   = [f for f in files if f.endswith("_embeddings.h5")]
print(f"\n{len(patch_files)} patch files, {len(emb_files)} embedding files")

# ── 2. Load one brain (patches + embeddings) ─────────────────────────────────
sample_key = patch_files[0]
print(f"\nLoading: {sample_key.split('/')[-1]}")

patches, metadata, attrs = read_h5_patches(sample_key)
emb_key = sample_key.replace("patches", "embeddings").replace("_patches.h5", "_embeddings.h5")
embeddings, emb_attrs = read_h5_embeddings(emb_key)

print(f"  patches shape : {patches.shape}  dtype={patches.dtype}")
print(f"  embeddings    : {embeddings.shape}")
print(f"  condition     : {attrs['condition']}")
print(f"  metadata cols : {list(metadata.columns)}")
print(metadata[["mean_intensity", "sharpness", "snr", "local_contrast"]].describe().round(2))

# ── 3. Show a grid of sample patches ─────────────────────────────────────────
print("\nPlotting sample patches and embedding space...")

# pick 12 diverse patches by sharpness
sharp_idx = metadata["sharpness"].nlargest(6).index.tolist()
blurry_idx = metadata["sharpness"].nsmallest(6).index.tolist()
sample_idx = sharp_idx + blurry_idx

fig = plt.figure(figsize=(16, 10))
fig.suptitle(f"Challenge A — Sample patches ({attrs['condition']})", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.4, wspace=0.3)

for col, idx in enumerate(sharp_idx):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(patches[idx], cmap="magma", vmin=np.percentile(patches[idx], 2), vmax=np.percentile(patches[idx], 98))
    ax.set_title(f"sharp={metadata.loc[idx, 'sharpness']:.0f}", fontsize=7)
    ax.axis("off")
    if col == 0:
        ax.set_ylabel("High sharpness", fontsize=8)

for col, idx in enumerate(blurry_idx):
    ax = fig.add_subplot(gs[1, col])
    ax.imshow(patches[idx], cmap="magma", vmin=np.percentile(patches[idx], 2), vmax=np.percentile(patches[idx], 98))
    ax.set_title(f"sharp={metadata.loc[idx, 'sharpness']:.0f}", fontsize=7)
    ax.axis("off")
    if col == 0:
        ax.set_ylabel("Low sharpness", fontsize=8)

plt.savefig("challenge_a_patches.png", dpi=120, bbox_inches="tight")
print("  Saved: challenge_a_patches.png")

# ── 4. Quick embedding space visualisation (PCA, fast, no GPU needed) ────────
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(embeddings)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sc = ax2.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=metadata["sharpness"].values, cmap="viridis", s=8, alpha=0.7
    )
    plt.colorbar(sc, ax=ax2, label="sharpness")
    ax2.set_title(f"PCA of embeddings — {sample_key.split('/')[-1]}\n(colour = sharpness)", fontsize=10)
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.tight_layout()
    plt.savefig("challenge_a_pca.png", dpi=120)
    print("  Saved: challenge_a_pca.png")
except ImportError:
    print("  (scikit-learn not available — skipping PCA)")

print("\nDone! Open challenge_a_patches.png and challenge_a_pca.png to explore the data.")
print("\nNext steps for Challenge A:")
print("  1. Load all 12 brains and stack embeddings")
print("  2. Run UMAP (pip install umap-learn) for a richer 2D view")
print("  3. Cluster with k-means or DBSCAN to find signal-pattern groups")
print("  4. Select one representative patch per cluster → your curated subset")
