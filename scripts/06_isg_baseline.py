import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests

import common

GMT = common.EXTERNAL / "gene_sets" / "hallmark_interferon_v2024.1.Hs.gmt"
GENE_SETS = {
    "ifn_alpha": "HALLMARK_INTERFERON_ALPHA_RESPONSE",  # primary
    "ifn_gamma": "HALLMARK_INTERFERON_GAMMA_RESPONSE",
}
OUT = common.RESULTS / "isg"
FIG = OUT / "figures"

EPITHELIAL = [
    "Basal Cells", "Mitotic Basal Cells", "Ciliated Cells", "Developing Ciliated Cells",
    "Deuterosomal Cells", "Secretory Cells", "Developing Secretory and Goblet Cells",
    "Goblet Cells", "Squamous Cells", "Ionocytes", "Enteroendocrine Cells",
]
# the only types with >= 10 cells in >= 8 mild and >= 8 severe donors (CONTEXT.md 7.1)
SINGLE_TYPES = ["Ciliated Cells", "Secretory Cells", "Goblet Cells", "Squamous Cells", "T Cells"]

CV_SEEDS = range(10)
N_PERM = 1000

GROUP_ORDER = ["control", "mild", "severe", "control_icu"]
GROUP_COLORS = {"control": "#8a8984", "mild": "#2a78d6", "severe": "#eb6834", "control_icu": "#1baf7a"}


def slug(name):
    return name.lower().replace(" ", "_")


def load_cells():
    """Log-normalised host genes from C1, with gene symbols as var_names."""
    print("reading", common.QC_H5AD)
    adata = ad.read_h5ad(common.QC_H5AD)
    counts = adata.layers["counts"]
    is_viral = adata.var["is_viral"].to_numpy()
    symbols = adata.var["gene_symbol"].astype(str).to_numpy()

    obs = adata.obs[["donor_id", "group", "severe", "who_peak", "ct_coarse", "viral_umis"]].copy()
    obs["donor_id"] = obs["donor_id"].astype(str)
    obs["ct_coarse"] = obs["ct_coarse"].astype(str)
    obs["total_umis"] = np.asarray(counts.sum(axis=1)).ravel()

    host = ad.AnnData(X=counts[:, ~is_viral].astype(np.float32), obs=obs)
    host.var_names = symbols[~is_viral]
    host.var_names_make_unique()
    sc.pp.normalize_total(host, target_sum=1e4)
    sc.pp.log1p(host)
    return host


def read_gmt(path):
    gene_sets = {}
    with open(path) as f:
        for line in f:
            # rstrip() rather than rstrip("\n"): a Windows checkout adds "\r"
            fields = line.rstrip().split("\t")
            gene_sets[fields[0]] = fields[2:]
    return gene_sets


def score_cells(adata):
    """Per-cell score for each gene set (scanpy: mean of set minus matched control genes)."""
    gmt = read_gmt(GMT)
    for short, full in GENE_SETS.items():
        genes = [g for g in gmt[full] if g in adata.var_names]
        print(f"{short}: {len(genes)}/{len(gmt[full])} genes found")
        sc.tl.score_genes(adata, genes, score_name=short, random_state=common.SEED)


def donor_viral_load(obs):
    """log10 viral UMIs per million total UMIs, per donor (Erythroblasts left out)."""
    keep = obs[obs["ct_coarse"] != "Erythroblasts"]
    per_donor = keep.groupby("donor_id")[["viral_umis", "total_umis"]].sum()
    return np.log10(per_donor["viral_umis"] / per_donor["total_umis"] * 1e6 + 1).rename("log_viral_load")


def donor_scores(obs):
    """Mean cell score per donor for epithelium and each single cell type (>= 10 cells)."""
    donor_info = obs.groupby("donor_id")[["group", "severe", "who_peak"]].first()
    cell_groups = {"epithelial": obs["ct_coarse"].isin(EPITHELIAL)}
    for ct in SINGLE_TYPES:
        cell_groups[slug(ct)] = obs["ct_coarse"] == ct

    rows = []
    for ct_name, mask in cell_groups.items():
        sub = obs[mask]
        n_cells = sub.groupby("donor_id").size()
        for short in GENE_SETS:
            means = sub.groupby("donor_id")[short].mean()
            table = pd.DataFrame({"ct": ct_name, "gene_set": short, "n_cells": n_cells, "score": means})
            rows.append(table[table["n_cells"] >= common.MIN_CELLS_PSEUDOBULK])

    scores = pd.concat(rows).rename_axis("donor_id").reset_index()
    scores = scores.merge(donor_info, on="donor_id").merge(donor_viral_load(obs), on="donor_id")
    return scores[scores["group"] != "convalescent"]


def compare(a, b):
    """MWU p and AUROC for 'group b has the higher score'."""
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    auroc = roc_auc_score(y, np.r_[a, b])
    test = mannwhitneyu(b, a)
    return test.statistic, test.pvalue, auroc


def run_tests(scores):
    """Three comparisons per cell group and gene set. AUROC is for the second group named
    having higher ISG, so mild_vs_severe below 0.5 means lower ISG in severe."""
    comparisons = {
        "mild_vs_severe": (["mild"], ["severe"]),
        "control_vs_covid": (["control"], ["mild", "severe"]),
        "control_icu_vs_severe": (["control_icu"], ["severe"]),
    }
    rows = []
    for (ct, gene_set), table in scores.groupby(["ct", "gene_set"], sort=False):
        for comp, (first, second) in comparisons.items():
            a = table.loc[table["group"].isin(first), "score"].to_numpy()
            b = table.loc[table["group"].isin(second), "score"].to_numpy()
            if len(a) < 3 or len(b) < 3:
                continue
            stat, p, auroc = compare(a, b)
            rows.append({
                "ct": ct, "gene_set": gene_set, "comparison": comp, "test": "mannwhitneyu",
                "statistic": stat, "pvalue": p, "auroc_second_higher": auroc,
                "median_first": np.median(a), "median_second": np.median(b),
                "n_first": len(a), "n_second": len(b),
            })
    tests = pd.DataFrame(rows)
    # family = all cell groups for one comparison and one gene set
    tests["qvalue_bh"] = tests.groupby(["comparison", "gene_set"])["pvalue"].transform(
        lambda p: multipletests(p, method="fdr_bh")[1])
    return tests


def cv_isg(isg, viral_load, y, adjust_viral_load):
    """Treat -ISG as a severity score inside repeated stratified 5-fold CV.

    The only things fitted are the decision threshold (midpoint of the two
    training-class medians) and, if asked, a linear fit of ISG on viral load.
    Both are fitted on the training donors only.
    Returns AUROC and balanced accuracy per repeat.
    """
    aucs, bal_accs = [], []
    for seed in CV_SEEDS:
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        pred = np.zeros(len(y))
        for train, test in folds.split(isg, y):
            score = -isg
            if adjust_viral_load:
                slope, intercept = np.polyfit(viral_load[train], isg[train], 1)
                score = -(isg - (slope * viral_load + intercept))
            threshold = (np.median(score[train][y[train] == 0]) + np.median(score[train][y[train] == 1])) / 2
            oof[test] = score[test]
            pred[test] = score[test] > threshold
        aucs.append(roc_auc_score(y, oof))
        bal_accs.append(balanced_accuracy_score(y, pred))
    return np.array(aucs), np.array(bal_accs)


def run_cv(scores):
    epi = scores[(scores["ct"] == "epithelial") & scores["group"].isin(["mild", "severe"])]
    rng = np.random.default_rng(common.SEED)
    rows, nulls = [], []
    for gene_set in GENE_SETS:
        table = epi[epi["gene_set"] == gene_set]
        isg = table["score"].to_numpy()
        viral_load = table["log_viral_load"].to_numpy()
        y = table["severe"].to_numpy().astype(int)
        for model, adjust in [("isg_score", False), ("isg_vl_adjusted", True)]:
            aucs, bal_accs = cv_isg(isg, viral_load, y, adjust)
            null = np.array([np.median(cv_isg(isg, viral_load, rng.permutation(y), adjust)[0])
                             for _ in range(N_PERM)])
            observed = np.median(aucs)
            rows.append({
                "task": "mild_vs_severe", "feature_set": f"{gene_set}_epithelial", "model": model,
                "cv_scheme": "strat5fold_x10", "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
                "auroc_median": observed, "auroc_min": aucs.min(), "auroc_max": aucs.max(),
                "bal_acc_median": np.median(bal_accs),
                "perm_p": (1 + np.sum(null >= observed)) / (1 + N_PERM), "n_perm": N_PERM,
            })
            nulls.append(pd.DataFrame({"feature_set": f"{gene_set}_epithelial", "model": model,
                                       "perm_id": range(N_PERM), "auroc": null}))
    return pd.DataFrame(rows), pd.concat(nulls)


def box_with_points(ax, values, x, width, color, rng, size):
    """Grey box (no fliers) with every donor drawn as a jittered dot on top."""
    grey = {"color": "#52514e"}
    ax.boxplot(values, positions=[x], widths=width, showfliers=False, medianprops={"color": "black"},
               boxprops=grey, whiskerprops=grey, capprops=grey)
    ax.scatter(x + rng.uniform(-width / 4, width / 4, len(values)), values, s=size,
               color=color, edgecolor="white", linewidth=0.6, zorder=3)


def plot_by_group(scores, path):
    table = scores[(scores["ct"] == "epithelial") & (scores["gene_set"] == "ifn_alpha")]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    labels = []
    rng = np.random.default_rng(common.SEED)
    for i, group in enumerate(GROUP_ORDER):
        values = table.loc[table["group"] == group, "score"].to_numpy()
        box_with_points(ax, values, i, 0.5, GROUP_COLORS[group], rng, size=28)
        labels.append(f"{group}\n(n={len(values)})")
    ax.set_xticks(range(len(GROUP_ORDER)), labels)
    ax.set_ylabel("IFN-α response score\n(donor mean, epithelial cells)")
    ax.set_title("Interferon response by group (n = patients)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_vs_viral_load(scores, path):
    table = scores[(scores["ct"] == "epithelial") & (scores["gene_set"] == "ifn_alpha")
                   & scores["group"].isin(["mild", "severe"])]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for group in ["mild", "severe"]:
        sub = table[table["group"] == group]
        ax.scatter(sub["log_viral_load"], sub["score"], s=36, color=GROUP_COLORS[group],
                   edgecolor="white", linewidth=0.8, label=f"{group} (n={len(sub)})", zorder=3)
    # one fit over all COVID donors, no labels: the adjusted score is the distance from this line
    slope, intercept = np.polyfit(table["log_viral_load"], table["score"], 1)
    xs = np.linspace(table["log_viral_load"].min(), table["log_viral_load"].max(), 50)
    ax.plot(xs, slope * xs + intercept, color="#52514e", linewidth=1.5, linestyle="--",
            label="fit, all COVID donors")
    ax.set_xlabel("log10(viral UMIs per million + 1)")
    ax.set_ylabel("IFN-α response score\n(donor mean, epithelial cells)")
    ax.set_title("ISG response vs viral load")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_by_celltype(scores, tests, path):
    table = scores[(scores["gene_set"] == "ifn_alpha") & scores["group"].isin(["mild", "severe"])]
    cts = ["epithelial"] + [slug(ct) for ct in SINGLE_TYPES]
    fig, ax = plt.subplots(figsize=(8, 4))
    rng = np.random.default_rng(common.SEED)
    for i, ct in enumerate(cts):
        for offset, group in [(-0.18, "mild"), (0.18, "severe")]:
            values = table.loc[(table["ct"] == ct) & (table["group"] == group), "score"].to_numpy()
            box_with_points(ax, values, i + offset, 0.3, GROUP_COLORS[group], rng, size=14)
    labels = []
    for ct in cts:
        row = tests[(tests["ct"] == ct) & (tests["gene_set"] == "ifn_alpha")
                    & (tests["comparison"] == "mild_vs_severe")].iloc[0]
        labels.append(f"{ct.replace('_', ' ')}\n{row.n_first} vs {row.n_second}\nq={row.qvalue_bh:.2f}")
    ax.set_xticks(range(len(cts)), labels, fontsize=7)
    ax.set_ylabel("IFN-α response score (donor mean)")
    ax.set_title("Mild vs severe by cell type (n = mild vs severe patients, BH across cell groups)")
    for group in ["mild", "severe"]:
        ax.scatter([], [], color=GROUP_COLORS[group], label=group)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)

    adata = load_cells()
    score_cells(adata)
    scores = donor_scores(adata.obs)
    tests = run_tests(scores)
    cv_metrics, perm_null = run_cv(scores)

    scores.to_csv(OUT / "donor_isg_scores.tsv", sep="\t", index=False)
    tests.to_csv(OUT / "isg_tests.tsv", sep="\t", index=False)
    cv_metrics.to_csv(OUT / "isg_cv_metrics.tsv", sep="\t", index=False)
    perm_null.to_csv(OUT / "isg_perm_null.tsv", sep="\t", index=False)
    plot_by_group(scores, FIG / "06_isg_by_group.png")
    plot_vs_viral_load(scores, FIG / "06_isg_vs_viral_load.png")
    plot_by_celltype(scores, tests, FIG / "06_isg_by_celltype.png")

    pd.set_option("display.width", 200)
    print("\nepithelial tests:")
    print(tests[tests["ct"] == "epithelial"].round(4).to_string(index=False))
    print("\nCV metrics (mild vs severe):")
    print(cv_metrics.round(3).to_string(index=False))
    print(f"\nwrote 4 tables and 3 figures to {OUT}")


if __name__ == "__main__":
    main()
