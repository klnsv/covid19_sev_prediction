import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

import common

N_PCS_COMPUTE = 50
N_PCS_USE = 30
N_NEIGHBORS = 15
LEIDEN_RESOLUTIONS = [0.5, 1.0, 1.5]
HEATMAP_RESOLUTION = 1.0

EMBEDDED_H5AD = common.PROCESSED / "nasal_embedded.h5ad"
QC_DIR = common.RESULTS / "qc"
FIG_DIR = QC_DIR / "figures"


GROUP_ORDER = ["control", "control_icu", "mild", "severe", "convalescent"]
GROUP_COLORS = {
    "control": "#2a78d6",
    "control_icu": "#eb6834",
    "mild": "#1baf7a",
    "severe": "#eda100",
    "convalescent": "#e87ba4",
}


LINEAGE_OF = {
    "Ciliated Cells": "ciliated",
    "Developing Ciliated Cells": "ciliated",
    "Deuterosomal Cells": "ciliated",
    "Secretory Cells": "secretory / goblet",
    "Goblet Cells": "secretory / goblet",
    "Developing Secretory and Goblet Cells": "secretory / goblet",
    "Basal Cells": "basal / squamous",
    "Mitotic Basal Cells": "basal / squamous",
    "Squamous Cells": "basal / squamous",
    "Ionocytes": "rare epithelial",
    "Enteroendocrine Cells": "rare epithelial",
    "T Cells": "immune",
    "B Cells": "immune",
    "Macrophages": "immune",
    "Dendritic Cells": "immune",
    "Plasmacytoid DCs": "immune",
    "Mast Cells": "immune",
    "Erythroblasts": "blood (contamination)",
}
LINEAGE_COLORS = {
    "ciliated": "#2a78d6",
    "secretory / goblet": "#eb6834",
    "basal / squamous": "#1baf7a",
    "rare epithelial": "#eda100",
    "immune": "#e87ba4",
    "blood (contamination)": "#008300",
}
MIN_CELLS_TO_LABEL = 100


MARKERS = {
    "Ciliated Cells": ["FOXJ1", "CAPS"],
    "Developing Ciliated Cells": ["CDC20B"],
    "Deuterosomal Cells": ["DEUP1"],
    "Secretory Cells": ["SCGB1A1", "BPIFA1"],
    "Goblet Cells": ["MUC5AC", "TFF3"],
    "Basal Cells": ["KRT5", "TP63"],
    "Mitotic Basal Cells": ["MKI67"],
    "Squamous Cells": ["KRT13", "SPRR3"],
    "Ionocytes": ["FOXI1", "CFTR"],
    "Enteroendocrine Cells": ["CHGA"],
    "T Cells": ["CD3E"],
    "B Cells": ["MS4A1"],
    "Macrophages": ["CD68", "C1QA"],
    "Dendritic Cells": ["CD1C"],
    "Plasmacytoid DCs": ["LILRA4"],
    "Mast Cells": ["TPSAB1"],
    "Erythroblasts": ["HBB"],
}


def run_pca_umap_leiden(adata):
    """PCA on scaled HVGs, then neighbour graph, UMAP and Leiden on the full object."""

    adata.obsm["X_pca_author"] = adata.obsm.pop("X_pca")
    adata.obsm["X_umap_author"] = adata.obsm.pop("X_umap")

    hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(hvg, max_value=10)
    sc.tl.pca(hvg, n_comps=N_PCS_COMPUTE, svd_solver="arpack", random_state=common.SEED)
    adata.obsm["X_pca"] = hvg.obsm["X_pca"]
    adata.uns["pca"] = hvg.uns["pca"]
    del hvg

    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS_USE,
                    use_rep="X_pca", random_state=common.SEED)
    sc.tl.umap(adata, random_state=common.SEED)
    for res in LEIDEN_RESOLUTIONS:
        sc.tl.leiden(adata, resolution=res, key_added=f"leiden_{res}",
                     flavor="igraph", n_iterations=2, directed=False,
                     random_state=common.SEED)
    return adata


def ari_table(obs):
    rows = []
    for res in LEIDEN_RESOLUTIONS:
        clusters = obs[f"leiden_{res}"]
        rows.append({
            "resolution": res,
            "n_clusters": clusters.nunique(),
            "ari_vs_ct_coarse": round(adjusted_rand_score(obs["ct_coarse"], clusters), 3),
            "ari_vs_ct_detailed": round(adjusted_rand_score(obs["ct_detailed"], clusters), 3),
            "n_cells": len(obs),
        })
    return pd.DataFrame(rows)


def style_umap_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def plot_pca_variance(adata, out_path):
    ratio = adata.uns["pca"]["variance_ratio"] * 100
    pcs = np.arange(1, len(ratio) + 1)
    used = ratio[:N_PCS_USE].sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(pcs, ratio, color="#2a78d6", width=0.8)
    ax.axvline(N_PCS_USE + 0.5, color="#666666", linestyle="--", linewidth=1)
    ax.text(N_PCS_USE + 1, ratio.max() * 0.9, f"first {N_PCS_USE} PCs used\n({used:.1f}% of variance)",
            fontsize=9, color="#444444", va="top")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("% variance explained")
    ax.set_title(f"PCA on {int(adata.var['highly_variable'].sum()):,} HVGs "
                 f"({adata.n_obs:,} cells, n = {adata.obs['donor_id'].nunique()} donors)")
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_umap_celltype_group(adata, out_path):
    """Slide 3: UMAP by cell type (coloured by lineage, labelled) and by group."""
    umap = adata.obsm["X_umap"]
    obs = adata.obs

    order = np.random.default_rng(common.SEED).permutation(adata.n_obs)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))


    ax = axes[0]
    lineage = obs["ct_coarse"].astype(str).map(LINEAGE_OF)
    colors = lineage.map(LINEAGE_COLORS).to_numpy()
    ax.scatter(umap[order, 0], umap[order, 1], c=colors[order], s=1, linewidths=0, rasterized=True)

    placed = []
    x_gap = 0.12 * np.ptp(umap[:, 0])
    y_gap = 0.035 * np.ptp(umap[:, 1])
    for ct, n in obs["ct_coarse"].value_counts().items():
        if n < MIN_CELLS_TO_LABEL:
            continue
        x, y = np.median(umap[(obs["ct_coarse"] == ct).to_numpy()], axis=0)
        while any(abs(x - px) < x_gap and abs(y - py) < y_gap for px, py in placed):
            y -= y_gap
        placed.append((x, y))
        label = ct.replace(" Cells", "").replace("Developing Secretory and Goblet", "Dev. Secretory/Goblet")
        text = ax.text(x, y, label, fontsize=8, ha="center", va="center", color="#1a1a1a")
        text.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground="white")])
    handles = []
    for name, color in LINEAGE_COLORS.items():
        n_cells = int((lineage == name).sum())
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="", markersize=7,
                                  markerfacecolor=color, markeredgewidth=0, label=f"{name} ({n_cells:,} cells)"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.04), ncol=2)
    ax.set_title(f"Author cell types ({obs['ct_coarse'].nunique()} types, labelled if ≥ {MIN_CELLS_TO_LABEL} cells)")
    style_umap_axis(ax)


    ax = axes[1]
    colors = obs["group"].astype(str).map(GROUP_COLORS).to_numpy()
    ax.scatter(umap[order, 0], umap[order, 1], c=colors[order], s=1, linewidths=0, rasterized=True)
    handles = []
    for g in GROUP_ORDER:
        cells = obs[obs["group"] == g]
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="", markersize=7,
                                  markerfacecolor=GROUP_COLORS[g], markeredgewidth=0,
                                  label=f"{g} (n = {cells['donor_id'].nunique()} donors, {len(cells):,} cells)"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.04), ncol=2)
    ax.set_title("Patient group")
    style_umap_axis(ax)

    fig.suptitle(f"Nasal swab cell landscape: {adata.n_obs:,} cells from "
                 f"n = {obs['donor_id'].nunique()} donors", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_umap_donor(adata, out_path):
    """Small multiples: the 3 largest donors of each main group highlighted on grey."""
    umap = adata.obsm["X_umap"]
    obs = adata.obs
    counts = obs.groupby(["group", "donor_id"], observed=True).size().reset_index(name="n")
    picks = []
    for g in ["control", "control_icu", "mild", "severe"]:
        top = counts[counts["group"] == g].sort_values("n", ascending=False).head(3)
        picks += list(zip(top["donor_id"], top["group"], top["n"]))

    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5))
    for ax, (donor, group, n) in zip(axes.T.ravel(), picks):
        is_donor = (obs["donor_id"] == donor).to_numpy()
        ax.scatter(umap[~is_donor, 0], umap[~is_donor, 1], c="#d9d9d9", s=0.5, linewidths=0, rasterized=True)
        ax.scatter(umap[is_donor, 0], umap[is_donor, 1], c=GROUP_COLORS[group], s=2, linewidths=0, rasterized=True)
        ax.set_title(f"{donor}\n{group}, {n:,} cells", fontsize=9)
        style_umap_axis(ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Where each donor's cells fall (3 largest donors per group; grey = all other cells)\n"
                 "Compact single-donor patches = cells group by patient (patient effect), "
                 "not only by cell type", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_leiden_vs_author(obs, ari, out_path):
    """Heatmap: for each Leiden cluster, the fraction of its cells in each author type."""
    key = f"leiden_{HEATMAP_RESOLUTION}"
    table = pd.crosstab(obs[key], obs["ct_coarse"])
    frac = table.div(table.sum(axis=1), axis=0)

    ct_order = obs["ct_coarse"].value_counts().index.tolist()
    frac = frac[ct_order]
    majority = frac.idxmax(axis=1).map({ct: i for i, ct in enumerate(ct_order)})
    frac = frac.loc[majority.sort_values(kind="stable").index]
    sizes = table.sum(axis=1).loc[frac.index]

    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(frac.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ct_order)))
    ax.set_xticklabels([ct.replace(" Cells", "") for ct in ct_order], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(frac)))
    ax.set_yticklabels([f"cluster {c} ({sizes[c]:,})" for c in frac.index], fontsize=7)
    ax.set_xlabel("Author cell type (Coarse_Cell_Annotations)")
    ax.set_ylabel("Leiden cluster (number of cells)")
    row = ari[ari["resolution"] == HEATMAP_RESOLUTION].iloc[0]
    ax.set_title(f"Leiden (resolution {HEATMAP_RESOLUTION}, {int(row['n_clusters'])} clusters) vs author labels\n"
                 f"ARI = {row['ari_vs_ct_coarse']:.2f} (coarse), {row['ari_vs_ct_detailed']:.2f} (detailed); "
                 f"{len(obs):,} cells, n = {obs['donor_id'].nunique()} donors")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.6)
    colorbar.set_label("Fraction of the cluster's cells")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_marker_dotplot(adata, out_path):
    ct_order = list(LINEAGE_OF)
    present = set(adata.var["gene_symbol"])
    markers = {ct: [g for g in genes if g in present] for ct, genes in MARKERS.items()}
    missing = [g for genes in MARKERS.values() for g in genes if g not in present]
    if missing:
        print(f"  markers not found in the data (skipped): {missing}")
    markers = {ct.replace(" Cells", ""): genes for ct, genes in markers.items() if genes}


    labels = pd.Categorical(adata.obs["ct_coarse"], categories=ct_order, ordered=True)
    adata.obs["ct_dotplot"] = labels.rename_categories(
        [ct.replace(" Cells", "").replace("Developing Secretory and Goblet", "Dev. Secretory/Goblet")
         for ct in ct_order])
    sc.pl.dotplot(adata, markers, groupby="ct_dotplot", gene_symbols="gene_symbol",
                  standard_scale="var", show=False)
    del adata.obs["ct_dotplot"]
    fig = plt.gcf()
    fig.suptitle(f"Marker genes per author cell type ({adata.n_obs:,} cells, "
                 f"n = {adata.obs['donor_id'].nunique()} donors)", y=1.12, fontsize=12)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading {common.QC_H5AD}")
    adata = sc.read_h5ad(common.QC_H5AD)
    print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes, "
          f"{int(adata.var['highly_variable'].sum())} HVGs")

    adata = run_pca_umap_leiden(adata)
    ratio = adata.uns["pca"]["variance_ratio"]
    print(f"  PCA: first {N_PCS_USE} of {N_PCS_COMPUTE} PCs explain {100 * ratio[:N_PCS_USE].sum():.1f}% "
          f"of HVG variance (all {N_PCS_COMPUTE}: {100 * ratio.sum():.1f}%)")

    ari = ari_table(adata.obs)
    ari_path = QC_DIR / "02_leiden_ari.tsv"
    ari.to_csv(ari_path, sep="\t", index=False)
    print(f"Wrote {ari_path}")
    print(ari.to_string(index=False))

    adata.write_h5ad(EMBEDDED_H5AD)
    print(f"Wrote {EMBEDDED_H5AD}")

    path = FIG_DIR / "02_pca_variance.png"
    plot_pca_variance(adata, path)
    print(f"Wrote {path}")

    path = FIG_DIR / "02_umap_celltype_group.png"
    plot_umap_celltype_group(adata, path)
    print(f"Wrote {path}")

    path = FIG_DIR / "02_umap_donor.png"
    plot_umap_donor(adata, path)
    print(f"Wrote {path}")

    path = FIG_DIR / "02_leiden_vs_author.png"
    plot_leiden_vs_author(adata.obs, ari, path)
    print(f"Wrote {path}")

    path = FIG_DIR / "02_marker_dotplot.png"
    plot_marker_dotplot(adata, path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
