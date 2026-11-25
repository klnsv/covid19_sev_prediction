"""Step 09: which cell types and genes carry the mild-vs-severe signal (contract C8).

Reads:  data/processed/patient_meta.tsv (C2), features_composition.tsv (C3),
        pseudobulk_coarse.h5ad (C4)
Writes: results/attribution/celltype_importance.tsv  (C8)
        results/attribution/gene_importance.tsv      (C8)
        results/attribution/single_type_null.tsv     permutation AUROCs per cell type x model
        results/attribution/ablation_repeats.tsv     per-repeat AUROCs of the full and dropped models
        results/attribution/figures/09_single_type_auroc.png
        results/attribution/figures/09_ablation.png
        results/attribution/figures/09_top_genes.png

Definitions are CONTEXT.md section 7.1:
- Single-cell-type models: for each type with >= 10 cells in >= 8 mild AND >= 8
  severe donors, the section 5.6 models on that type's pseudobulk logCPM (in-fold
  top-2000-variance genes -> scale -> PCA 10, via classify_utils), only donors that
  have the type. AUROC vs that type's own label-permutation null.
- Ablation: combined model = composition + pseudobulk of all eligible types.
  Drop one block at a time; delta_auroc = full - dropped (medians over 10 repeats).
  A donor missing a type gets that block imputed with the training-fold median.
- Gene importance: L2 logistic regression on standardised in-fold top-variance
  genes (mean |coef|) and random forest (mean |SHAP| on the test fold), over the
  10 x 5 CV fits; stability = fraction of fits where the gene is in the top 50.

Run from the repo root: python scripts/09_attribution.py
Quick test:             python scripts/09_attribution.py --quick
"""
import sys
import numpy as np
import pandas as pd
import anndata as ad
import shap
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
import classify_utils as cu

QUICK = "--quick" in sys.argv
N_REPEATS = 2 if QUICK else 10                     # CV seeds 0-9
N_PERM = {"logreg": 20, "linsvm": 20, "rf": 10} if QUICK else {"logreg": 1000, "linsvm": 1000, "rf": 200}
N_JOBS = -1                  # all cores; seeds are fixed per job, so results do not depend on it
MIN_DONORS_PER_CLASS = 8     # CONTEXT.md 7.1 eligibility (C4 already requires >= 10 cells)
N_TOP_GENES_FIT = 2000       # in-fold top-variance genes for the gene-importance models
TOP_K_STABILITY = 50

OUT_DIR = common.RESULTS / "attribution"
FIG_DIR = OUT_DIR / "figures"
MODEL_COLORS = {"logreg": "#2a78d6", "linsvm": "#eb6834", "rf": "#1baf7a"}


# ---------- data ----------

def log_cpm(counts):
    """log1p counts per million, per sample (same as step 05)."""
    cpm = counts / counts.sum(axis=1, keepdims=True) * 1e6
    return np.log1p(cpm)


def load_data():
    meta = pd.read_csv(common.PATIENT_META, sep="\t")
    meta = meta[meta["group"].isin(["mild", "severe"])].reset_index(drop=True)
    y = (meta["group"] == "severe").astype(int).to_numpy()

    comp = pd.read_csv(common.FEATURES_COMPOSITION, sep="\t").set_index("donor_id")
    clr_cols = [c for c in comp.columns if c.startswith("clr__")]
    comp = comp.loc[meta["donor_id"], clr_cols]

    pb = ad.read_h5ad(common.PSEUDOBULK_H5AD)
    pb_obs = pb.obs.copy()
    pb_obs["ct"] = pb_obs["ct"].astype(str)
    pb_obs["donor_id"] = pb_obs["donor_id"].astype(str)
    counts = pb.X.toarray().astype(float)
    gene_ids = np.asarray(pb.var_names)
    symbols = pb.var["gene_symbol"].astype(str).to_numpy()

    # one logCPM table per cell type, rows = donors that have the type
    blocks = {}
    for ct in pb_obs["ct"].unique():
        rows = np.where((pb_obs["ct"] == ct) & pb_obs["donor_id"].isin(meta["donor_id"]))[0]
        table = pd.DataFrame(log_cpm(counts[rows]), index=pb_obs["donor_id"].to_numpy()[rows])
        blocks[ct] = table
    return meta, y, comp, blocks, gene_ids, symbols


def eligible_types(meta, blocks):
    severe = set(meta.loc[meta["group"] == "severe", "donor_id"])
    mild = set(meta.loc[meta["group"] == "mild", "donor_id"])
    keep = []
    for ct, table in blocks.items():
        if ct == "all":
            continue
        donors = set(table.index)
        if len(donors & severe) >= MIN_DONORS_PER_CLASS and len(donors & mild) >= MIN_DONORS_PER_CLASS:
            keep.append(ct)
    return sorted(keep)


# ---------- single-cell-type models ----------

def one_perm_auroc(X, y_shuffled, model_name, seed):
    oof = cu.run_cv(X, y_shuffled, model_name, "rep5fold", seed, pca=True)
    return roc_auc_score(y_shuffled, oof)


def permutation_null_parallel(X, y, model_name, n_perm):
    """Same shuffles and seeds as classify_utils.permutation_null, run in parallel."""
    rng = np.random.RandomState(common.SEED)
    shuffles = [rng.permutation(y) for _ in range(n_perm)]
    return Parallel(n_jobs=N_JOBS)(
        delayed(one_perm_auroc)(X, ys, model_name, i % 10) for i, ys in enumerate(shuffles))


def single_type_models(meta, y, blocks, cell_types):
    rows, null_rows = [], []
    donor_index = {d: i for i, d in enumerate(meta["donor_id"])}
    for ct in ["all"] + cell_types:
        table = blocks[ct]
        idx = np.array([donor_index[d] for d in table.index])
        X, y_ct = table.to_numpy(), y[idx]
        for model_name in cu.MODEL_NAMES:
            aurocs = [cu.auroc_balacc(y_ct, cu.run_cv(X, y_ct, model_name, "rep5fold", rep, pca=True))[0]
                      for rep in range(N_REPEATS)]
            null = permutation_null_parallel(X, y_ct, model_name, N_PERM[model_name])
            observed = float(np.median(aurocs))
            rows.append({"ct": ct, "model": model_name, "n_donors": len(y_ct),
                         "n_pos": int(y_ct.sum()), "n_neg": int((1 - y_ct).sum()),
                         "auroc_single_median": observed,
                         "auroc_single_min": float(np.min(aurocs)), "auroc_single_max": float(np.max(aurocs)),
                         "perm_p": cu.perm_pvalue(observed, null), "n_perm": len(null)})
            null_rows += [{"ct": ct, "model": model_name, "perm_id": i, "auroc": a} for i, a in enumerate(null)]
            print(f"  single {ct:28s} {model_name:6s} n={len(y_ct):2d} "
                  f"AUROC {observed:.3f} [{np.min(aurocs):.3f}-{np.max(aurocs):.3f}] "
                  f"perm p {rows[-1]['perm_p']:.4f}")
    return pd.DataFrame(rows), pd.DataFrame(null_rows)


# ---------- ablation ----------

def build_combined(meta, comp, blocks, cell_types):
    """Composition + one pseudobulk block per eligible type; missing donors = NaN."""
    parts, block_columns = [comp.to_numpy()], {"composition": list(range(comp.shape[1]))}
    start = comp.shape[1]
    for ct in cell_types:
        block = blocks[ct].reindex(meta["donor_id"]).to_numpy()
        parts.append(block)
        block_columns[ct] = list(range(start, start + block.shape[1]))
        start += block.shape[1]
    return np.hstack(parts), block_columns


def make_combined_model(model_name, seed, block_columns):
    """Each pseudobulk block: median impute -> top-variance genes -> scale -> PCA 10,
    all fit on the training fold. Composition: scale only."""
    branches = []
    for name, cols in block_columns.items():
        if name == "composition":
            branches.append((name, StandardScaler(), cols))
        else:
            branches.append((name, make_pipeline(SimpleImputer(strategy="median"),
                                                 cu.TopVarianceGenes(2000), StandardScaler(),
                                                 PCA(n_components=10, random_state=seed)), cols))
    classifier = cu.make_model(model_name, seed).steps[-1][1]   # same fixed hyperparameters as step 05
    return make_pipeline(ColumnTransformer(branches), classifier)


def cv_combined(X, y, model_name, seed, block_columns):
    oof = np.zeros(len(y))
    for train_idx, test_idx in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        model = make_combined_model(model_name, seed, block_columns)
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = cu.get_score(model, X[test_idx])
    return roc_auc_score(y, oof)


def ablation(meta, y, comp, blocks, cell_types):
    X, block_columns = build_combined(meta, comp, blocks, cell_types)
    jobs, keys = [], []
    for model_name in cu.MODEL_NAMES:
        for dropped in ["none"] + list(block_columns):
            kept = {k: v for k, v in block_columns.items() if k != dropped}
            # re-index the kept columns so they point into the reduced matrix
            cols = [c for v in kept.values() for c in v]
            X_kept = X[:, cols]
            new_cols, start = {}, 0
            for k, v in kept.items():
                new_cols[k] = list(range(start, start + len(v)))
                start += len(v)
            for rep in range(N_REPEATS):
                jobs.append(delayed(cv_combined)(X_kept, y, model_name, rep, new_cols))
                keys.append((model_name, dropped, rep))
    aurocs = Parallel(n_jobs=N_JOBS)(jobs)
    repeats = pd.DataFrame(keys, columns=["model", "dropped", "repeat"])
    repeats["auroc"] = aurocs
    return repeats


# ---------- gene importance ----------

def importance_one_fit(X, y, train_idx, test_idx, seed):
    """Fit logreg and RF on in-fold top-variance genes; return (gene columns, |coef|, mean |SHAP|)."""
    selector = cu.TopVarianceGenes(N_TOP_GENES_FIT).fit(X[train_idx])
    cols = selector.keep_
    scaler = StandardScaler().fit(X[train_idx][:, cols])
    X_train = scaler.transform(X[train_idx][:, cols])
    X_test = scaler.transform(X[test_idx][:, cols])

    logreg = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=seed)
    logreg.fit(X_train, y[train_idx])
    coef = np.abs(logreg.coef_[0])

    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=1)
    rf.fit(X_train, y[train_idx])
    shap_values = shap.TreeExplainer(rf).shap_values(X_test)
    # binary classifier: SHAP for class 1 (severe); shape depends on the shap version
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    mean_shap = np.abs(shap_values).mean(axis=0)
    return cols, coef, mean_shap


def gene_importance(meta, y, blocks, cell_types, gene_ids, symbols):
    donor_index = {d: i for i, d in enumerate(meta["donor_id"])}
    rows = []
    for ct in ["all"] + cell_types:
        table = blocks[ct]
        idx = np.array([donor_index[d] for d in table.index])
        X, y_ct = table.to_numpy(), y[idx]
        jobs = []
        for rep in range(N_REPEATS):
            for train_idx, test_idx in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y_ct):
                jobs.append(delayed(importance_one_fit)(X, y_ct, train_idx, test_idx, rep))
        fits = Parallel(n_jobs=N_JOBS)(jobs)

        n_fits = len(fits)
        for model_name, which in [("logreg", 1), ("rf", 2)]:
            total = np.zeros(X.shape[1])
            in_top = np.zeros(X.shape[1])
            for fit in fits:
                cols, values = fit[0], fit[which]
                total[cols] += values      # genes not selected in a fit count as 0
                top = cols[np.argsort(values)[::-1][:TOP_K_STABILITY]]
                in_top[top] += 1
            mean_imp = total / n_fits
            keep = np.where(mean_imp > 0)[0]
            table_ct = pd.DataFrame({"gene_id": gene_ids[keep], "gene_symbol": symbols[keep], "ct": ct,
                                     "model": model_name, "importance": mean_imp[keep],
                                     "stability": in_top[keep] / n_fits})
            table_ct = table_ct.sort_values("importance", ascending=False).reset_index(drop=True)
            table_ct["rank"] = np.arange(1, len(table_ct) + 1)
            rows.append(table_ct)
            top5 = ", ".join(table_ct["gene_symbol"].head(5))
            print(f"  genes {ct:28s} {model_name:6s} {n_fits} fits, top 5: {top5}")
    columns = ["gene_id", "gene_symbol", "ct", "model", "importance", "rank", "stability"]
    return pd.concat(rows, ignore_index=True)[columns]


# ---------- figures ----------

def style(ax):
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def plot_single_type(single, null, out_path):
    cts = list(dict.fromkeys(single["ct"]))
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.27
    for i, model_name in enumerate(cu.MODEL_NAMES):
        sub = single[single["model"] == model_name].set_index("ct").loc[cts]
        x = np.arange(len(cts)) + (i - 1) * width
        err = [sub["auroc_single_median"] - sub["auroc_single_min"], sub["auroc_single_max"] - sub["auroc_single_median"]]
        ax.bar(x, sub["auroc_single_median"], width=width * 0.92, color=MODEL_COLORS[model_name],
               yerr=err, capsize=2, error_kw={"linewidth": 0.8}, label=model_name)
        # 95th percentile of each null as a black tick
        for xi, ct in zip(x, cts):
            q95 = null[(null["ct"] == ct) & (null["model"] == model_name)]["auroc"].quantile(0.95)
            ax.plot([xi - width * 0.4, xi + width * 0.4], [q95, q95], color="#1a1a1a", linewidth=1.2)
        for xi, (_, r) in zip(x, sub.iterrows()):
            ax.text(xi, r["auroc_single_max"] + 0.01, f"p={r['perm_p']:.3f}", ha="center", fontsize=6, rotation=90)
    labels = []
    for ct in cts:
        r = single[(single["ct"] == ct)].iloc[0]
        labels.append(f"{ct}\n{r['n_pos']} severe / {r['n_neg']} mild")
    ax.set_xticks(np.arange(len(cts)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("AUROC (median, bars = min-max over 10 CV repeats)")
    ax.set_title("Single-cell-type models, mild vs severe\nblack tick = 95th percentile of that type's permutation null")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_ablation(repeats, y, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, model_name in zip(axes, cu.MODEL_NAMES):
        sub = repeats[repeats["model"] == model_name]
        full = sub[sub["dropped"] == "none"]["auroc"].to_numpy()
        names = [d for d in dict.fromkeys(sub["dropped"]) if d != "none"]
        deltas = [full - sub[sub["dropped"] == d].sort_values("repeat")["auroc"].to_numpy() for d in names]
        ax.boxplot(deltas, widths=0.6, medianprops={"color": "#1a1a1a"})
        ax.axhline(0, color="#888888", linestyle="--", linewidth=0.8)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{model_name}: full model AUROC {np.median(full):.2f}", fontsize=10)
        style(ax)
    axes[0].set_ylabel("ΔAUROC = full − without this block\n(per CV repeat, > 0 = block helps)")
    n_blocks = len(names) - 1    # every dropped block except composition is a cell type
    fig.suptitle(f"Ablation of the combined model (composition + {n_blocks} cell-type pseudobulk blocks), "
                 f"n = {int(y.sum())} severe / {int((1 - y).sum())} mild", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_top_genes(genes, cell_types, out_path):
    panels = ["all"] + cell_types
    fig, axes = plt.subplots(2, len(panels), figsize=(3.2 * len(panels), 8))
    for row, model_name in enumerate(["logreg", "rf"]):
        for col, ct in enumerate(panels):
            ax = axes[row, col]
            sub = genes[(genes["ct"] == ct) & (genes["model"] == model_name)].head(15)[::-1]
            ax.barh(sub["gene_symbol"], sub["importance"], color=MODEL_COLORS[model_name])
            for yi, (_, r) in enumerate(sub.iterrows()):
                ax.text(r["importance"], yi, f" {r['stability']:.2f}", va="center", fontsize=6, color="#444444")
            ax.set_title(f"{ct}\n{model_name}", fontsize=8)
            ax.tick_params(labelsize=7)
            for side in ["top", "right"]:
                ax.spines[side].set_visible(False)
        axes[row, 0].set_xlabel("mean |coef|" if model_name == "logreg" else "mean |SHAP|", fontsize=8)
    fig.suptitle("Top 15 genes per cell type (number = stability: fraction of 50 CV fits in the top 50)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    meta, y, comp, blocks, gene_ids, symbols = load_data()
    cell_types = eligible_types(meta, blocks)
    print(f"{int(y.sum())} severe, {int((1 - y).sum())} mild patients; eligible cell types: {cell_types}")

    print("\nSingle-cell-type models")
    single, null = single_type_models(meta, y, blocks, cell_types)
    null.to_csv(OUT_DIR / "single_type_null.tsv", sep="\t", index=False)

    print("\nAblation")
    repeats = ablation(meta, y, comp, blocks, cell_types)
    repeats.to_csv(OUT_DIR / "ablation_repeats.tsv", sep="\t", index=False)
    drop_rows = []
    for model_name in cu.MODEL_NAMES:
        sub = repeats[repeats["model"] == model_name]
        full = sub[sub["dropped"] == "none"].sort_values("repeat")["auroc"].to_numpy()
        for dropped in [d for d in dict.fromkeys(sub["dropped"]) if d != "none"]:
            dropped_auc = sub[sub["dropped"] == dropped].sort_values("repeat")["auroc"].to_numpy()
            drop_rows.append({"ct": dropped, "model": model_name, "auroc_full_median": float(np.median(full)),
                              "auroc_full_min": float(np.min(full)), "auroc_full_max": float(np.max(full)),
                              "auroc_drop_median": float(np.median(dropped_auc)),
                              "delta_auroc": float(np.median(full - dropped_auc))})
        print(f"  {model_name}: full AUROC {np.median(full):.3f} [{np.min(full):.3f}-{np.max(full):.3f}]")
    drops = pd.DataFrame(drop_rows)

    # C8 celltype_importance: one row per cell type x model (composition appears only in the ablation)
    celltype = single.merge(drops, on=["ct", "model"], how="outer")
    columns = ["ct", "model", "n_donors", "n_pos", "n_neg", "auroc_single_median", "auroc_single_min",
               "auroc_single_max", "perm_p", "n_perm", "auroc_full_median", "auroc_full_min",
               "auroc_full_max", "auroc_drop_median", "delta_auroc"]
    celltype = celltype[columns]
    celltype.to_csv(OUT_DIR / "celltype_importance.tsv", sep="\t", index=False, float_format="%.4f")
    print(celltype.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nGene importance")
    genes = gene_importance(meta, y, blocks, cell_types, gene_ids, symbols)
    genes.to_csv(OUT_DIR / "gene_importance.tsv", sep="\t", index=False, float_format="%.6g")

    plot_single_type(single, null, FIG_DIR / "09_single_type_auroc.png")
    plot_ablation(repeats, y, FIG_DIR / "09_ablation.png")
    plot_top_genes(genes, cell_types, FIG_DIR / "09_top_genes.png")
    print(f"\nWrote tables in {OUT_DIR} and figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
