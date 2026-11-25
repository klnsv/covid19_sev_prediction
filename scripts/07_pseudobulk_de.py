"""Step 07: pseudobulk differential expression with DESeq2.

Reads:  data/processed/pseudobulk_coarse.h5ad (summed raw counts per donor x cell type)
Writes: results/de/de_<ct>_<contrast>.tsv   one table per cell type x contrast
        results/de/de_all.tsv                stacked, rows with pvalue < 0.05 only (the full
                                             stack is 16 MB, over the 10 MB results/ limit)
        results/de/de_summary.tsv            n donors, n genes tested, n significant per test
        results/de/de_sex_check.tsv          severe_vs_mild refit with ~ sex + condition
        results/de/figures/07_de_n_significant.png
        results/de/figures/07_volcano_severe_vs_mild.png

The donor is the sample: each row of the pseudobulk file is one donor's
summed counts for one cell type, so the p-values are donor-level.
log2fc > 0 means higher in the first group of the contrast name.
BH-FDR (DESeq2's padj) is within each cell type x contrast; significant = padj < 0.05.

Run from the repo root: python scripts/07_pseudobulk_de.py
"""
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import spearmanr
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

import common

# contrast name -> (groups in "a", groups in "b"); log2fc = a vs b
CONTRASTS = {
    "severe_vs_mild": (["severe"], ["mild"]),
    "covid_vs_control": (["mild", "severe"], ["control"]),
    "severe_vs_control_icu": (["severe"], ["control_icu"]),
}
EXCLUDED_CTS = ["erythroblasts"]   # blood contamination
MIN_DONORS_PER_GROUP = 3
MIN_COUNT = 10        # gene kept if >= MIN_COUNT counts in >= (smaller group size) samples
PADJ_CUTOFF = 0.05
DE_ALL_PVALUE = 0.05  # de_all.tsv keeps rows below this nominal p; per-test files keep every gene
N_CPUS = 8
N_LABELS = 5          # genes labelled per volcano panel

DE_DIR = common.RESULTS / "de"
FIG_DIR = DE_DIR / "figures"
UP_COLOR = "#eb6834"     # up in the first group
DOWN_COLOR = "#2a78d6"   # down in the first group
NS_COLOR = "#c8c8c8"


def load_pseudobulk():
    pb = ad.read_h5ad(common.PSEUDOBULK_H5AD)
    counts = pd.DataFrame(pb.X.toarray().astype(int), index=pb.obs_names, columns=pb.var_names)
    obs = pb.obs.copy()
    obs["group"] = obs["group"].astype(str)
    obs["sex"] = obs["sex"].astype(str)
    obs["ct"] = obs["ct"].astype(str)
    symbols = pb.var["gene_symbol"].astype(str)
    return counts, obs, symbols


def samples_for_contrast(obs, ct, groups_a, groups_b):
    """Rows of one cell type in the two groups, with condition = 'a' or 'b'."""
    rows = obs[(obs["ct"] == ct) & obs["group"].isin(groups_a + groups_b)].copy()
    rows["condition"] = np.where(rows["group"].isin(groups_a), "a", "b")
    return rows


def filter_genes(counts, n_min):
    keep = (counts >= MIN_COUNT).sum(axis=0) >= n_min
    return counts.loc[:, keep]


def run_deseq(counts, meta, design):
    dds = DeseqDataSet(counts=counts, metadata=meta, design=design,
                       refit_cooks=True, n_cpus=N_CPUS, quiet=True)
    dds.deseq2()
    stats = DeseqStats(dds, contrast=["condition", "a", "b"], alpha=PADJ_CUTOFF,
                       n_cpus=N_CPUS, quiet=True)
    stats.summary()
    return stats.results_df


def tidy_results(res, symbols, ct, contrast, n_a, n_b):
    table = pd.DataFrame({
        "gene_id": res.index,
        "gene_symbol": symbols.loc[res.index].to_numpy(),
        "ct": ct,
        "contrast": contrast,
        "base_mean": res["baseMean"].to_numpy(),
        "log2fc": res["log2FoldChange"].to_numpy(),
        "lfcse": res["lfcSE"].to_numpy(),
        "stat": res["stat"].to_numpy(),
        "pvalue": res["pvalue"].to_numpy(),
        "padj": res["padj"].to_numpy(),
        "n_donors_a": n_a,
        "n_donors_b": n_b,
    })
    return table.sort_values("pvalue", na_position="last").reset_index(drop=True)


def count_significant(table):
    sig = table["padj"] < PADJ_CUTOFF
    return int(sig.sum()), int((sig & (table["log2fc"] > 0)).sum()), int((sig & (table["log2fc"] < 0)).sum())


def plot_n_significant(summary, out_path):
    tested = summary[summary["status"] == "tested"]
    cts = list(dict.fromkeys(tested["ct"]))
    contrasts = list(CONTRASTS)
    colors = {"severe_vs_mild": "#2a78d6", "covid_vs_control": "#eb6834", "severe_vs_control_icu": "#1baf7a"}
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.27
    x = np.arange(len(cts))
    for i, contrast in enumerate(contrasts):
        heights, labels = [], []
        for ct in cts:
            row = tested[(tested["ct"] == ct) & (tested["contrast"] == contrast)]
            if len(row):
                heights.append(row["n_sig"].iloc[0])
                labels.append(f"{int(row['n_donors_a'].iloc[0])}v{int(row['n_donors_b'].iloc[0])}")
            else:
                heights.append(0)
                labels.append("")
        positions = x + (i - 1) * width
        ax.bar(positions, heights, width=width * 0.92, color=colors[contrast], label=contrast)
        for pos, h, lab in zip(positions, heights, labels):
            if lab:
                ax.text(pos, h, f"{h}\n({lab})", ha="center", va="bottom", fontsize=6, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(cts, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"Genes with padj < {PADJ_CUTOFF}")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_title("Pseudobulk DE: significant genes per cell type and contrast\n"
                 "(label: count, then n donors first group v second group)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_volcanoes(tables, out_path):
    """Small multiples of severe_vs_mild, one panel per cell type."""
    panels = [t for t in tables if t["contrast"].iloc[0] == "severe_vs_mild"]
    n_cols = 4
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.6 * n_rows), squeeze=False)
    for ax, table in zip(axes.ravel(), panels):
        t = table.dropna(subset=["padj"])
        y = -np.log10(t["padj"].clip(lower=1e-300))
        sig = t["padj"] < PADJ_CUTOFF
        colors = np.where(~sig, NS_COLOR, np.where(t["log2fc"] > 0, UP_COLOR, DOWN_COLOR))
        ax.scatter(t["log2fc"], y, c=colors, s=4, linewidths=0, rasterized=True)
        ax.axhline(-np.log10(PADJ_CUTOFF), color="#888888", linestyle="--", linewidth=0.8)
        for _, row in t[sig].head(N_LABELS).iterrows():
            text = ax.text(row["log2fc"], -np.log10(max(row["padj"], 1e-300)), row["gene_symbol"],
                           fontsize=6.5, ha="center", va="bottom")
            text.set_path_effects([path_effects.withStroke(linewidth=2, foreground="white")])
        n_up = int((sig & (t["log2fc"] > 0)).sum())
        n_down = int((sig & (t["log2fc"] < 0)).sum())
        ax.set_title(f"{t['ct'].iloc[0]} (n = {int(t['n_donors_a'].iloc[0])} severe, "
                     f"{int(t['n_donors_b'].iloc[0])} mild)\n{n_up} up, {n_down} down in severe", fontsize=8)
        ax.set_xlabel("log2 fold change (severe / mild)", fontsize=8)
        ax.set_ylabel("-log10 padj", fontsize=8)
        ax.tick_params(labelsize=7)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle("Severe vs mild, pseudobulk DESeq2 per cell type "
                 f"(orange = up in severe, blue = down; dashed line padj = {PADJ_CUTOFF})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    counts, obs, symbols = load_pseudobulk()
    print(f"Read {common.PSEUDOBULK_H5AD}: {counts.shape[0]} samples x {counts.shape[1]} genes")

    cell_types = [ct for ct in obs["ct"].value_counts().index if ct not in EXCLUDED_CTS]
    cell_types = ["all"] + [ct for ct in cell_types if ct != "all"]

    tables, summary_rows, sex_rows = [], [], []
    for contrast, (groups_a, groups_b) in CONTRASTS.items():
        for ct in cell_types:
            meta = samples_for_contrast(obs, ct, groups_a, groups_b)
            n_a = int((meta["condition"] == "a").sum())
            n_b = int((meta["condition"] == "b").sum())
            row = {"ct": ct, "contrast": contrast, "n_donors_a": n_a, "n_donors_b": n_b}
            if min(n_a, n_b) < MIN_DONORS_PER_GROUP:
                row["status"] = f"skipped (< {MIN_DONORS_PER_GROUP} donors in a group)"
                summary_rows.append(row)
                continue

            ct_counts = filter_genes(counts.loc[meta.index], min(n_a, n_b))
            res = run_deseq(ct_counts, meta[["condition", "sex"]], "~condition")
            table = tidy_results(res, symbols, ct, contrast, n_a, n_b)
            table.to_csv(DE_DIR / f"de_{ct}_{contrast}.tsv", sep="\t", index=False, float_format="%.6g")
            tables.append(table)

            n_sig, n_up, n_down = count_significant(table)
            row.update({"status": "tested", "n_genes_tested": len(table),
                        "n_sig": n_sig, "n_up": n_up, "n_down": n_down})
            summary_rows.append(row)
            print(f"  {contrast:22s} {ct:40s} {n_a:2d} v {n_b:2d} donors, "
                  f"{len(table):6d} genes, {n_sig:5d} sig ({n_up} up, {n_down} down)")

            # check: does adjusting for sex change the severe-vs-mild result?
            if contrast == "severe_vs_mild" and meta["sex"].nunique() == 2:
                res_sex = run_deseq(ct_counts, meta[["condition", "sex"]], "~sex + condition")
                sig_primary = set(res.index[res["padj"] < PADJ_CUTOFF])
                sig_sex = set(res_sex.index[res_sex["padj"] < PADJ_CUTOFF])
                both = res[["log2FoldChange"]].merge(res_sex[["log2FoldChange"]], left_index=True,
                                                     right_index=True, suffixes=("", "_sex")).dropna()
                rho = spearmanr(both.iloc[:, 0], both.iloc[:, 1])[0]
                sex_rows.append({"ct": ct, "n_donors_severe": n_a, "n_donors_mild": n_b,
                                 "n_sig_primary": len(sig_primary), "n_sig_sex_adjusted": len(sig_sex),
                                 "n_sig_both": len(sig_primary & sig_sex),
                                 "spearman_log2fc": round(rho, 3)})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(DE_DIR / "de_summary.tsv", sep="\t", index=False)
    sex_check = pd.DataFrame(sex_rows)
    sex_check.to_csv(DE_DIR / "de_sex_check.tsv", sep="\t", index=False)
    de_all = pd.concat(tables, ignore_index=True)
    de_all = de_all[de_all["pvalue"] < DE_ALL_PVALUE]
    de_all.to_csv(DE_DIR / "de_all.tsv", sep="\t", index=False, float_format="%.6g")
    print(f"Wrote {len(tables)} tables de_<ct>_<contrast>.tsv, de_all.tsv ({len(de_all):,} rows), "
          f"de_summary.tsv, de_sex_check.tsv in {DE_DIR}")

    plot_n_significant(summary, FIG_DIR / "07_de_n_significant.png")
    plot_volcanoes(tables, FIG_DIR / "07_volcano_severe_vs_mild.png")
    print(f"Wrote figures in {FIG_DIR}")

    print("\nSex-adjustment check (severe_vs_mild):")
    print(sex_check.to_string(index=False))


if __name__ == "__main__":
    main()
