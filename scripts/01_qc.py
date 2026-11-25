import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common

VIRAL_REFERENCE = "NCBITaxon:2697049"
MITO_PREFIX = "MT-"
N_TOP_HVG = 2000
HVG_FLAVOR = "seurat"


GROUP_ORDER = ["control", "control_icu", "mild", "severe", "convalescent"]
GROUP_COLORS = {
    "control": "#2a78d6",
    "control_icu": "#eb6834",
    "mild": "#1baf7a",
    "severe": "#eda100",
    "convalescent": "#e87ba4",
}

QC_DIR = common.RESULTS / "qc"
FIG_DIR = QC_DIR / "figures"


def add_counts_layer(adata):
    """Move the raw integer counts from .raw.X into layers['counts']."""
    assert list(adata.raw.var_names) == list(adata.var_names), "raw and var genes differ"
    counts = adata.raw.X.copy()

    assert np.all(counts.data == np.round(counts.data)), "raw.X is not integer-valued"
    adata.layers["counts"] = counts
    adata.raw = None
    return adata


def add_var_columns(adata):
    adata.var["gene_symbol"] = adata.var["feature_name"].astype(str)
    adata.var["is_viral"] = (adata.var["feature_reference"] == VIRAL_REFERENCE).to_numpy()
    return adata


def add_qc_metrics(adata):
    """n_counts, n_genes and pct_mito recomputed from the counts layer (all genes)."""
    counts = adata.layers["counts"]
    is_mito = adata.var["gene_symbol"].str.startswith(MITO_PREFIX).to_numpy()

    adata.obs["n_counts"] = np.asarray(counts.sum(axis=1)).ravel()
    adata.obs["n_genes"] = np.asarray((counts > 0).sum(axis=1)).ravel()
    mito_counts = np.asarray(counts[:, is_mito].sum(axis=1)).ravel()
    adata.obs["pct_mito"] = 100 * mito_counts / adata.obs["n_counts"]
    return adata, int(is_mito.sum())


def add_obs_columns(adata):
    """Standard obs columns from the C1 contract. Original columns are kept."""
    obs = adata.obs

    cohort = obs["Cohort_Disease_WHO_Score"].astype(str)
    unmapped = sorted(set(cohort) - set(common.GROUP_MAP))
    assert not unmapped, f"cohort values missing from GROUP_MAP: {unmapped}"
    obs["group"] = cohort.map(common.GROUP_MAP)


    obs["severe"] = obs["group"].map({"severe": 1.0, "mild": 0.0}).astype(float)

    obs["donor_id"] = obs["donor_id"].astype(str)
    obs["who_peak"] = obs["Peak_Respiratory_Support_WHO_Score"].astype(str).astype(int)
    obs["sex"] = obs["sex"].astype(str)

    obs["age_decade"] = obs["development_stage"].astype(str).str.replace(" stage", "", regex=False)
    obs["bloody_swab"] = (obs["Bloody_Swab"].astype(str) == "Yes").to_numpy()
    obs["ct_coarse"] = obs["Coarse_Cell_Annotations"].astype(str)
    obs["ct_detailed"] = obs["Detailed_Cell_Annotations"].astype(str)
    obs["viral_umis"] = obs["SARSCoV2_Total_Corrected"].astype(float)
    return adata


def normalise_and_hvg(adata):
    """X = log1p(CP10k) from counts; HVGs on host genes only."""
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    host = ~adata.var["is_viral"].to_numpy()
    hvg = sc.pp.highly_variable_genes(
        adata[:, host], flavor=HVG_FLAVOR, n_top_genes=N_TOP_HVG, inplace=False
    )
    highly_variable = np.zeros(adata.n_vars, dtype=bool)
    highly_variable[host] = hvg["highly_variable"].to_numpy()
    adata.var["highly_variable"] = highly_variable
    return adata


def donor_summary(obs):
    """One row per donor: group, cell count and median QC metrics."""
    rows = []
    for donor, cells in obs.groupby("donor_id", observed=True):
        rows.append({
            "donor_id": donor,
            "group": cells["group"].iloc[0],
            "who_peak": cells["who_peak"].iloc[0],
            "bloody_swab": cells["bloody_swab"].iloc[0],
            "n_cells": len(cells),
            "median_n_genes": cells["n_genes"].median(),
            "median_n_counts": cells["n_counts"].median(),
            "median_pct_mito": round(cells["pct_mito"].median(), 2),
        })
    table = pd.DataFrame(rows)
    table["group"] = pd.Categorical(table["group"], categories=GROUP_ORDER, ordered=True)
    table = table.sort_values(["group", "n_cells"], ascending=[True, False])
    return table


def plot_qc_violins(obs, out_path):
    """Genes, UMIs and % mito per cell, one violin per group, n on the axis labels."""
    groups = [g for g in GROUP_ORDER if g in set(obs["group"])]
    labels = []
    for g in groups:
        cells = obs[obs["group"] == g]
        labels.append(f"{g}\n{cells['donor_id'].nunique()} donors\n{len(cells):,} cells")

    panels = [
        ("n_genes", "Genes per cell", "log10 genes per cell"),
        ("n_counts", "UMIs per cell", "log10 UMIs per cell"),
        ("pct_mito", "% mitochondrial UMIs", "% mitochondrial UMIs"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (col, title, ylabel) in zip(axes, panels):
        values = [obs.loc[obs["group"] == g, col].to_numpy() for g in groups]
        if ylabel.startswith("log10"):
            values = [np.log10(v) for v in values]
        parts = ax.violinplot(values, showmedians=True, widths=0.8)
        for body, g in zip(parts["bodies"], groups):
            body.set_facecolor(GROUP_COLORS[g])
            body.set_edgecolor("white")
            body.set_alpha(0.85)
        for key in ["cmedians", "cmins", "cmaxes", "cbars"]:
            parts[key].set_color("#444444")
            parts[key].set_linewidth(1)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    fig.suptitle(f"Per-cell QC metrics by group (author-QC'd cells, n = {obs['donor_id'].nunique()} donors)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_donor_overview(table, out_path):
    """Cells per donor, sorted within group, coloured by group, n donors in the legend."""
    fig, ax = plt.subplots(figsize=(15, 5))
    x = np.arange(len(table))
    colors = [GROUP_COLORS[g] for g in table["group"]]
    ax.bar(x, table["n_cells"], color=colors, width=0.85, edgecolor="white", linewidth=0.5)
    ax.axhline(100, color="#666666", linestyle="--", linewidth=1)


    short = (table["donor_id"]
             .str.replace("COVID19_Participant", "C", regex=False)
             .str.replace("Control_Participant", "H", regex=False))
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_xlim(-0.6, len(table) - 0.4)
    ax.set_ylabel("Cells per donor (after QC)")
    ax.set_xlabel("Donor (C = COVID19_Participant, H = Control_Participant)")

    handles = []
    for g in GROUP_ORDER:
        n = int((table["group"] == g).sum())
        if n > 0:
            handles.append(plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g], label=f"{g} (n = {n})"))
    handles.append(plt.Line2D([0], [0], color="#666666", linestyle="--", linewidth=1, label="100 cells"))
    ax.legend(handles=handles, frameon=False, fontsize=9, ncol=len(handles),
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.set_title(f"Cells per donor (n = {len(table)} donors, {int(table['n_cells'].sum()):,} cells)", pad=28)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    common.PROCESSED.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading {common.RAW_H5AD}")
    adata = sc.read_h5ad(common.RAW_H5AD)
    print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    adata = add_counts_layer(adata)
    adata = add_var_columns(adata)
    adata, n_mito_genes = add_qc_metrics(adata)
    adata = add_obs_columns(adata)


    author_mito = adata.obs["Percent_Mitochondrial"].astype(float)
    diff = (adata.obs["pct_mito"] - author_mito).abs()
    corr = np.corrcoef(adata.obs["pct_mito"], author_mito)[0, 1]
    print(f"  pct_mito vs author Percent_Mitochondrial: r = {corr:.4f}, "
          f"median |diff| = {diff.median():.3f}, max |diff| = {diff.max():.3f}")

    adata.uns["qc_params"] = {
        "filtering_applied": False,
        "note": "author-QC'd cells kept; no extra filtering (CONTEXT.md 5.3)",
        "min_genes_per_cell": 0,
        "min_counts_per_cell": 0,
        "max_pct_mito": 100.0,
        "min_cells_per_donor": 0,
        "mito_prefix": MITO_PREFIX,
        "n_mito_genes": n_mito_genes,
        "normalisation": "normalize_total(target_sum=1e4) + log1p on layers['counts']",
        "hvg_flavor": HVG_FLAVOR,
        "hvg_n_top_genes": N_TOP_HVG,
        "hvg_host_genes_only": True,
    }

    adata = normalise_and_hvg(adata)

    adata.write_h5ad(common.QC_H5AD)
    print(f"Wrote {common.QC_H5AD}")

    table = donor_summary(adata.obs)
    table_path = QC_DIR / "01_donor_qc_summary.tsv"
    table.to_csv(table_path, sep="\t", index=False)
    print(f"Wrote {table_path}")

    violin_path = FIG_DIR / "01_qc_violins.png"
    plot_qc_violins(adata.obs, violin_path)
    print(f"Wrote {violin_path}")

    overview_path = FIG_DIR / "01_donor_overview.png"
    plot_donor_overview(table, overview_path)
    print(f"Wrote {overview_path}")

    # summary
    print("\nSummary")
    print(f"  cells x genes: {adata.n_obs:,} x {adata.n_vars:,} "
          f"({int(adata.var['is_viral'].sum())} viral, {n_mito_genes} mito)")
    print(f"  highly variable host genes: {int(adata.var['highly_variable'].sum())}")
    per_group = adata.obs.groupby("group", observed=True).agg(
        donors=("donor_id", "nunique"), cells=("donor_id", "size"))
    print(per_group.reindex(GROUP_ORDER).to_string())
    print(f"  severe label: {int((table['group'] == 'severe').sum())} severe, "
          f"{int((table['group'] == 'mild').sum())} mild donors")
    for col in ["n_genes", "n_counts", "pct_mito"]:
        v = adata.obs[col]
        print(f"  {col}: min {v.min():.1f}, median {v.median():.1f}, max {v.max():.1f}")


if __name__ == "__main__":
    main()
