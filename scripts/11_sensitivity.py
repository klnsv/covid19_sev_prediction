
import sys
import numpy as np
import pandas as pd
import anndata as ad
from joblib import Parallel, delayed
from scipy import sparse
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
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
N_REPEATS = 2 if QUICK else 10
N_PERM = 10 if QUICK else 200
N_BOOT = 200 if QUICK else 2000
N_DRAWS = 2 if QUICK else 10
N_JOBS = 10
PAIRS = [("comp", "rf"), ("comp", "logreg"), ("pb_all", "logreg"), ("comb", "rf")]
ELIGIBLE_TYPES = ["Ciliated Cells", "Secretory Cells", "Goblet Cells", "Squamous Cells", "T Cells"]
AGE_ORDER = {"third decade": 2, "fourth decade": 3, "fifth decade": 4, "sixth decade": 5,
             "seventh decade": 6, "eighth decade": 7, "ninth decade": 8}
OUT_DIR = common.RESULTS / "sensitivity"
FIG_DIR = OUT_DIR / "figures"


def slug(cell_type):
    return cell_type.lower().replace(" ", "_")



def load_cells():
    adata = ad.read_h5ad(common.QC_H5AD)
    host = ~adata.var["is_viral"].to_numpy()
    counts = adata.layers["counts"][:, host].tocsr()
    obs = adata.obs
    cells = pd.DataFrame({
        "donor_id": obs["donor_id"].astype(str).to_numpy(),
        "ct": obs["ct_coarse"].astype(str).to_numpy(),
        "ct_detailed": obs["ct_detailed"].astype(str).to_numpy(),
        "pct_mito": obs["pct_mito"].astype(float).to_numpy(),
    })
    coarse_types = sorted(t for t in cells["ct"].unique() if t not in common.EXCLUDED_CELL_TYPES)
    detailed_types = sorted(cells.loc[~cells["ct"].isin(common.EXCLUDED_CELL_TYPES), "ct_detailed"].unique())
    del adata
    return cells, counts, coarse_types, detailed_types


def clr(table):
    log_counts = np.log(table + 0.5)
    return log_counts.sub(log_counts.mean(axis=1), axis=0)


def log_cpm(counts):
    total = counts.sum(axis=1, keepdims=True)
    total[total == 0] = 1
    return np.log1p(counts / total * 1e6)


def build_features(cells, counts, donors, keep, coarse_types, detailed_types=None):
    """Composition (CLR) and whole-sample pseudobulk logCPM from the kept cells."""
    use = keep & ~cells["ct"].isin(common.EXCLUDED_CELL_TYPES).to_numpy()
    kept = cells[use]
    comp_counts = pd.crosstab(kept["donor_id"], kept["ct"]).reindex(index=donors, columns=coarse_types, fill_value=0)
    features = {"comp": clr(comp_counts).to_numpy(), "n_cells": comp_counts.sum(axis=1).to_numpy()}
    if detailed_types is not None:
        det = pd.crosstab(kept["donor_id"], kept["ct_detailed"]).reindex(index=donors, columns=detailed_types, fill_value=0)
        features["comp_detailed"] = clr(det).to_numpy()
    donor_index = {d: i for i, d in enumerate(donors)}
    rows = np.array([donor_index.get(d, -1) for d in cells["donor_id"]])
    mask = use & (rows >= 0)
    indicator = sparse.csr_matrix((np.ones(mask.sum()), (rows[mask], np.where(mask)[0])),
                                  shape=(len(donors), len(cells)))
    features["pb_all"] = log_cpm(np.asarray((indicator @ counts).todense(), dtype=float))
    return features


def pseudobulk_type(cells, counts, donors, cell_type, min_cells):
    """logCPM of one cell type per donor; donors with < min_cells of it are dropped."""
    rows, keep_donors = [], []
    for d in donors:
        mask = (cells["donor_id"].to_numpy() == d) & (cells["ct"].to_numpy() == cell_type)
        if mask.sum() >= min_cells:
            rows.append(np.asarray(counts[np.where(mask)[0]].sum(axis=0)).ravel())
            keep_donors.append(d)
    return np.array(keep_donors), log_cpm(np.array(rows, dtype=float))




def model_with_n_genes(name, seed, n_genes):
    """classify_utils.make_model(pca=True) with a different number of in-fold top-variance genes."""
    classifier = cu.make_model(name, seed).steps[-1][1]
    return make_pipeline(cu.TopVarianceGenes(n_genes), StandardScaler(), PCA(n_components=10, random_state=seed),
                         classifier)


def run_cv_custom(X, y, build, seed):
    oof = np.zeros(len(y))
    for train_idx, test_idx in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        model = build(seed)
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = cu.get_score(model, X[test_idx])
    return oof


def cv_scores(X, y, model_name, pca, n_plain, seed, n_genes=None):
    if n_genes is not None:
        return run_cv_custom(X, y, lambda s: model_with_n_genes(model_name, s, n_genes), seed)
    return cu.run_cv(X, y, model_name, "rep5fold", seed, pca=pca, n_plain=n_plain)


def one_perm(X, y_shuffled, model_name, pca, n_plain, seed, n_genes):
    return roc_auc_score(y_shuffled, cv_scores(X, y_shuffled, model_name, pca, n_plain, seed, n_genes))


def evaluate(X, y, model_name, pca=False, n_plain=0, n_genes=None, n_perm=N_PERM):
    """AUROC median/min/max over repeats, balanced accuracy, permutation p, repeat-averaged OOF."""
    oofs = Parallel(n_jobs=N_JOBS)(delayed(cv_scores)(X, y, model_name, pca, n_plain, rep, n_genes)
                                   for rep in range(N_REPEATS))
    aurocs, balaccs = zip(*[cu.auroc_balacc(y, o) for o in oofs])
    result = {"auroc_median": float(np.median(aurocs)), "auroc_min": float(np.min(aurocs)),
              "auroc_max": float(np.max(aurocs)), "bal_acc_median": float(np.median(balaccs)),
              "perm_p": np.nan, "oof_mean": np.mean(oofs, axis=0), "aurocs": list(aurocs)}
    if n_perm > 0:
        rng = np.random.RandomState(common.SEED)
        shuffles = [rng.permutation(y) for _ in range(n_perm)]
        null = Parallel(n_jobs=N_JOBS)(delayed(one_perm)(X, ys, model_name, pca, n_plain, i % 10, n_genes)
                                       for i, ys in enumerate(shuffles))
        result["perm_p"] = cu.perm_pvalue(result["auroc_median"], null)
    return result


def lodo_auroc(X, y, i, model_name, pca, n_plain):
    """Median AUROC over CV repeats with donor i left out."""
    X_i, y_i = np.delete(X, i, axis=0), np.delete(y, i)
    return float(np.median([cu.auroc_balacc(y_i, cv_scores(X_i, y_i, model_name, pca, n_plain, rep))[0]
                            for rep in range(N_REPEATS)]))


def top_celltypes(comp, y, names):
    coefs = []
    for rep in range(N_REPEATS):
        for train_idx, _ in StratifiedKFold(5, shuffle=True, random_state=rep).split(comp, y):
            scaler = StandardScaler().fit(comp[train_idx])
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=rep)
            lr.fit(scaler.transform(comp[train_idx]), y[train_idx])
            coefs.append(lr.coef_[0])
    mean = np.mean(coefs, axis=0)
    order = np.argsort(np.abs(mean))[::-1][:3]
    return ";".join(f"{names[i]}({'+' if mean[i] > 0 else '-'})" for i in order)


def top_genes(pb, y, gene_ids, k=20):
    total = np.zeros(pb.shape[1])
    for rep in range(N_REPEATS):
        for train_idx, _ in StratifiedKFold(5, shuffle=True, random_state=rep).split(pb, y):
            selector = cu.TopVarianceGenes(2000).fit(pb[train_idx])
            cols = selector.keep_
            scaler = StandardScaler().fit(pb[train_idx][:, cols])
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=rep)
            lr.fit(scaler.transform(pb[train_idx][:, cols]), y[train_idx])
            total[cols] += np.abs(lr.coef_[0])
    return set(gene_ids[np.argsort(total)[::-1][:k]])




def feature_matrix(features, name, idx):
    if name == "comp":
        return features["comp"][idx], False, 0
    if name == "pb_all":
        return features["pb_all"][idx], True, 0
    if name == "comb":
        return np.hstack([features["comp"][idx], features["pb_all"][idx]]), True, features["comp"].shape[1]
    raise ValueError(name)


def run_setting(axis, setting, features, idx, y, ct_names, gene_ids, primary_genes, pairs=PAIRS, extra=None):
    rows = []
    top3 = top_celltypes(features["comp"][idx], y, ct_names)
    genes = top_genes(features["pb_all"][idx], y, gene_ids)
    jaccard = len(genes & primary_genes) / len(genes | primary_genes) if primary_genes else 1.0
    results = {}
    for fs, model_name in pairs:
        X, pca, n_plain = feature_matrix(features, fs, idx)
        res = evaluate(X, y, model_name, pca, n_plain)
        results[(fs, model_name)] = res
        row = {"axis": axis, "setting": setting, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
               "feature_set": fs, "model": model_name,
               **{k: res[k] for k in ["auroc_median", "auroc_min", "auroc_max", "bal_acc_median", "perm_p"]},
               "top3_celltypes": top3, "jaccard_top20_genes_vs_primary": round(jaccard, 3)}
        if extra:
            row.update(extra)
        rows.append(row)
        print(f"  {axis} {setting:34s} {fs:7s} {model_name:6s} n={int(y.sum())}v{int((1 - y).sum())} "
              f"AUROC {res['auroc_median']:.3f} [{res['auroc_min']:.3f}-{res['auroc_max']:.3f}] "
              f"perm p {res['perm_p']:.3f}")
    return rows, results, genes




def plot_forest(summary, out_path):
    pairs = [("comp", "rf"), ("pb_all", "logreg")]
    sub = summary[summary["auroc_median"].notna()]
    settings = list(dict.fromkeys(zip(sub["axis"], sub["setting"])))
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.28 * len(settings) + 2), sharey=True)
    for ax, (fs, model_name) in zip(axes, pairs):
        for i, (axis, setting) in enumerate(settings):
            r = sub[(sub["axis"] == axis) & (sub["setting"] == setting) & (sub["feature_set"] == fs)
                    & (sub["model"] == model_name)]
            if r.empty:
                continue
            r = r.iloc[0]
            color = "#2a78d6" if r["perm_p"] < 0.05 else "#b0b0b0"
            ax.plot([r["auroc_min"], r["auroc_max"]], [i, i], color=color, linewidth=2)
            ax.plot(r["auroc_median"], i, "o", color=color, markersize=6)
        ax.axvline(0.5, color="#888888", linestyle="--", linewidth=0.8)
        ax.set_title(f"{fs} / {model_name}", fontsize=10)
        ax.set_xlabel("AUROC, median and min-max over CV repeats")
        ax.set_xlim(0.2, 1.0)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    labels = []
    for axis, setting in settings:
        r = sub[(sub["axis"] == axis) & (sub["setting"] == setting)].iloc[0]
        labels.append(f"{axis}: {setting} ({int(r['n_pos'])}v{int(r['n_neg'])})")
    axes[0].set_yticks(range(len(settings)))
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()
    fig.suptitle("Sensitivity of mild-vs-severe classification (blue = permutation p < 0.05; n = severe v mild)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_downsampling(draws, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"comp/rf": "#2a78d6", "comp/logreg": "#eb6834", "pb_all/logreg": "#1baf7a", "comb/rf": "#eda100"}
    for pair, color in colors.items():
        sub = draws[draws["pair"] == pair]
        summary = sub.groupby("cells_per_donor")["auroc_median"].agg(["median", "min", "max"]).reset_index()
        ax.plot(summary["cells_per_donor"], summary["median"], marker="o", color=color, linewidth=2, label=pair)
        ax.fill_between(summary["cells_per_donor"], summary["min"], summary["max"], color=color, alpha=0.15)
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks(sorted(draws["cells_per_donor"].unique()))
    ax.set_xticklabels([str(int(v)) if v < 1e6 else "all" for v in sorted(draws["cells_per_donor"].unique())])
    ax.set_xlabel("cells per patient (random downsample; donors with fewer keep all)")
    ax.set_ylabel("AUROC (median over draws; band = min-max)")
    ax.set_title("How many cells per patient are needed? (n = 21 severe / 14 mild)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_specificity(scores, out_path):
    order = ["mild", "severe", "control", "control_icu"]
    colors = {"mild": "#1baf7a", "severe": "#eda100", "control": "#2a78d6", "control_icu": "#eb6834"}
    pairs = list(dict.fromkeys(scores["pair"]))
    fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 4), sharey=True)
    for ax, pair in zip(np.atleast_1d(axes), pairs):
        sub = scores[scores["pair"] == pair]
        for i, g in enumerate(order):
            values = sub.loc[sub["group"] == g, "score"].to_numpy()
            jitter = np.random.default_rng(common.SEED).uniform(-0.15, 0.15, len(values))
            ax.scatter(np.full(len(values), i) + jitter, values, color=colors[g], s=18)
            ax.plot([i - 0.25, i + 0.25], [np.median(values)] * 2, color="#1a1a1a", linewidth=2)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{g}\nn={int((sub['group'] == g).sum())}" for g in order], fontsize=8)
        ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8)
        ax.set_title(pair, fontsize=10)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("P(severe) from the mild-vs-severe model\n(mild/severe: out-of-fold)")
    fig.suptitle("Axis B, specificity: do non-COVID ICU patients score as 'severe COVID'?", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)




def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(common.PATIENT_META, sep="\t")
    meta["age"] = meta["age_decade"].map(AGE_ORDER)
    donors = meta["donor_id"].to_numpy()
    cells, counts, coarse_types, detailed_types = load_cells()
    gene_ids = np.asarray(ad.read_h5ad(common.PSEUDOBULK_H5AD, backed="r").var_names)
    ct_names = [slug(t) for t in coarse_types]
    print(f"Loaded {len(cells):,} cells, {counts.shape[1]:,} host genes, {len(donors)} donors")

    all_cells = np.ones(len(cells), dtype=bool)
    base = build_features(cells, counts, donors, all_cells, coarse_types, detailed_types)
    covid = meta["group"].isin(["mild", "severe"]).to_numpy()
    primary_idx = np.where(covid)[0]
    y_primary = (meta.loc[covid, "group"] == "severe").astype(int).to_numpy()
    rows = []


    print("\nA. severity definition")
    r, primary_results, primary_genes = run_setting("A", "primary (WHO 1-5 vs 6-8)", base, primary_idx, y_primary,
                                                    ct_names, gene_ids, None)
    rows += r
    who = meta["who_peak"].to_numpy()
    for setting, mild_max, severe_min in [("WHO <=5 vs >=7 (drop WHO 6)", 5, 7), ("WHO <=4 vs 8 (extremes)", 4, 8)]:
        idx = np.where(covid & ((who <= mild_max) | (who >= severe_min)))[0]
        y = (who[idx] >= severe_min).astype(int)
        rows += run_setting("A", setting, base, idx, y, ct_names, gene_ids, primary_genes)[0]
    for (fs, model_name), res in primary_results.items():
        rho, p = spearmanr(res["oof_mean"], who[primary_idx])
        rows.append({"axis": "A", "setting": "ordinal: Spearman(OOF score, WHO peak)", "n_pos": int(y_primary.sum()),
                     "n_neg": int((1 - y_primary).sum()), "feature_set": fs, "model": model_name,
                     "extra_metric": "spearman_rho", "extra_value": rho, "extra_p": p})
        print(f"  A ordinal {fs}/{model_name}: Spearman rho {rho:.3f} (p {p:.4f}) across 35 patients")


    print("\nB. specificity")
    score_rows = []
    others = np.where(meta["group"].isin(["control", "control_icu"]).to_numpy())[0]
    for fs, model_name in PAIRS:
        X, pca, n_plain = feature_matrix(base, fs, primary_idx)
        X_other, _, _ = feature_matrix(base, fs, others)
        preds = []
        for rep in range(N_REPEATS):
            model = cu.make_model(model_name, rep, pca, n_plain)
            model.fit(X, y_primary)
            preds.append(cu.get_score(model, X_other))
        other_scores = np.mean(preds, axis=0)
        pair = f"{fs}/{model_name}"
        for d, g, s in zip(meta.loc[covid, "donor_id"], meta.loc[covid, "group"], primary_results[(fs, model_name)]["oof_mean"]):
            score_rows.append({"pair": pair, "donor_id": d, "group": g, "score": s})
        for d, g, s in zip(donors[others], meta["group"].to_numpy()[others], other_scores):
            score_rows.append({"pair": pair, "donor_id": d, "group": g, "score": s})
        scores = pd.DataFrame([s for s in score_rows if s["pair"] == pair])
        icu = scores.loc[scores["group"] == "control_icu", "score"]
        healthy = scores.loc[scores["group"] == "control", "score"]
        severe = scores.loc[scores["group"] == "severe", "score"]
        threshold = 0.5 if model_name != "linsvm" else 0.0
        auc_sev_icu = roc_auc_score(np.r_[np.ones(len(severe)), np.zeros(len(icu))], np.r_[severe, icu])
        rows.append({"axis": "B", "setting": "ICU controls scored by the mild-vs-severe model",
                     "n_pos": len(severe), "n_neg": len(icu), "feature_set": fs, "model": model_name,
                     "auroc_median": auc_sev_icu,
                     "extra_metric": "frac_icu_called_severe", "extra_value": float((icu >= threshold).mean()),
                     "extra_p": mannwhitneyu(icu, healthy).pvalue})
        rows.append({"axis": "B", "setting": "healthy controls scored by the mild-vs-severe model",
                     "n_pos": len(severe), "n_neg": len(healthy), "feature_set": fs, "model": model_name,
                     "extra_metric": "frac_healthy_called_severe", "extra_value": float((healthy >= threshold).mean())})
        print(f"  B {pair}: ICU called severe {(icu >= threshold).mean():.2f} (median score {icu.median():.2f}), "
              f"healthy {(healthy >= threshold).mean():.2f} ({healthy.median():.2f}); "
              f"AUROC severe(OOF) vs ICU {auc_sev_icu:.3f}")
    pd.DataFrame(score_rows).to_csv(OUT_DIR / "specificity_scores.tsv", sep="\t", index=False, float_format="%.4f")


    print("\nC. feature set")
    for n_genes in [500, 1000]:
        res = evaluate(base["pb_all"][primary_idx], y_primary, "logreg", n_genes=n_genes)
        rows.append({"axis": "C", "setting": f"pb_all with {n_genes} in-fold HVGs (primary 2000)",
                     "n_pos": int(y_primary.sum()), "n_neg": int((1 - y_primary).sum()),
                     "feature_set": "pb_all", "model": "logreg",
                     **{k: res[k] for k in ["auroc_median", "auroc_min", "auroc_max", "bal_acc_median", "perm_p"]}})
        print(f"  C pb_all {n_genes} HVGs logreg AUROC {res['auroc_median']:.3f} perm p {res['perm_p']:.3f}")
    for model_name in ["rf", "logreg"]:
        res = evaluate(base["comp_detailed"][primary_idx], y_primary, model_name)
        rows.append({"axis": "C", "setting": "composition over detailed (39) cell types",
                     "n_pos": int(y_primary.sum()), "n_neg": int((1 - y_primary).sum()),
                     "feature_set": "comp_detailed", "model": model_name,
                     **{k: res[k] for k in ["auroc_median", "auroc_min", "auroc_max", "bal_acc_median", "perm_p"]}})
        print(f"  C detailed composition {model_name} AUROC {res['auroc_median']:.3f} perm p {res['perm_p']:.3f}")
    for cell_type in ELIGIBLE_TYPES:
        keep_donors, X = pseudobulk_type(cells, counts, donors[primary_idx], cell_type, common.MIN_CELLS_PSEUDOBULK)
        y = (meta.set_index("donor_id").loc[keep_donors, "group"] == "severe").astype(int).to_numpy()
        res = evaluate(X, y, "logreg", pca=True)
        rows.append({"axis": "C", "setting": f"pb_{slug(cell_type)} (single cell type)", "n_pos": int(y.sum()),
                     "n_neg": int((1 - y).sum()), "feature_set": f"pb_{slug(cell_type)}", "model": "logreg",
                     **{k: res[k] for k in ["auroc_median", "auroc_min", "auroc_max", "bal_acc_median", "perm_p"]}})
        print(f"  C pb_{slug(cell_type)} logreg n={int(y.sum())}v{int((1 - y).sum())} "
              f"AUROC {res['auroc_median']:.3f} perm p {res['perm_p']:.3f}")


    print("\nD. QC and inclusion")
    n_cells = base["n_cells"]
    for min_cells in [50, 100]:
        idx = np.where(covid & (n_cells >= min_cells))[0]
        y = (meta["group"].to_numpy()[idx] == "severe").astype(int)
        rows += run_setting("D", f"donors with >= {min_cells} cells", base, idx, y, ct_names, gene_ids, primary_genes)[0]
    idx = np.where(covid & ~meta["bloody_swab"].to_numpy())[0]
    y = (meta["group"].to_numpy()[idx] == "severe").astype(int)
    rows += run_setting("D", "exclude bloody-swab donors", base, idx, y, ct_names, gene_ids, primary_genes)[0]
    for cutoff in [30, 20]:
        keep = cells["pct_mito"].to_numpy() < cutoff
        features = build_features(cells, counts, donors, keep, coarse_types)
        print(f"  mito < {cutoff}%: {int(keep.sum()):,} of {len(keep):,} cells kept")
        rows += run_setting("D", f"cells with pct_mito < {cutoff}% (recomputed)", features, primary_idx, y_primary,
                            ct_names, gene_ids, primary_genes)[0]
    for min_cells in [5, 20]:
        keep_donors, X = pseudobulk_type(cells, counts, donors[primary_idx], "Ciliated Cells", min_cells)
        y = (meta.set_index("donor_id").loc[keep_donors, "group"] == "severe").astype(int).to_numpy()
        res = evaluate(X, y, "logreg", pca=True)
        rows.append({"axis": "D", "setting": f"pb_ciliated with min {min_cells} cells (primary 10)",
                     "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "feature_set": "pb_ciliated_cells",
                     "model": "logreg",
                     **{k: res[k] for k in ["auroc_median", "auroc_min", "auroc_max", "bal_acc_median", "perm_p"]}})
        print(f"  D pb_ciliated min {min_cells} cells n={int(y.sum())}v{int((1 - y).sum())} "
              f"AUROC {res['auroc_median']:.3f} perm p {res['perm_p']:.3f}")


    print("\nE. sampling depth")
    draw_rows = []
    rng = np.random.default_rng(common.SEED)
    donor_of_cell = cells["donor_id"].to_numpy()
    for depth in [50, 100, 200]:
        for draw in range(N_DRAWS):
            keep = np.zeros(len(cells), dtype=bool)
            for d in donors:
                where = np.where(donor_of_cell == d)[0]
                keep[rng.choice(where, size=min(depth, len(where)), replace=False)] = True
            features = build_features(cells, counts, donors, keep, coarse_types)
            for fs, model_name in PAIRS:
                X, pca, n_plain = feature_matrix(features, fs, primary_idx)
                res = evaluate(X, y_primary, model_name, pca, n_plain, n_perm=N_PERM if draw == 0 else 0)
                draw_rows.append({"cells_per_donor": depth, "draw": draw, "pair": f"{fs}/{model_name}",
                                  "auroc_median": res["auroc_median"], "perm_p": res["perm_p"]})
        sub = pd.DataFrame([d for d in draw_rows if d["cells_per_donor"] == depth])
        for pair in sub["pair"].unique():
            s = sub[sub["pair"] == pair]
            fs, model_name = pair.split("/")
            rows.append({"axis": "E", "setting": f"downsample to {depth} cells per donor ({N_DRAWS} draws)",
                         "n_pos": int(y_primary.sum()), "n_neg": int((1 - y_primary).sum()), "feature_set": fs,
                         "model": model_name, "auroc_median": s["auroc_median"].median(),
                         "auroc_min": s["auroc_median"].min(), "auroc_max": s["auroc_median"].max(),
                         "perm_p": s["perm_p"].iloc[0]})
            print(f"  E {depth} cells {pair}: AUROC over draws {s['auroc_median'].median():.3f} "
                  f"[{s['auroc_median'].min():.3f}-{s['auroc_median'].max():.3f}], perm p (draw 0) {s['perm_p'].iloc[0]:.3f}")
    for (fs, model_name), res in primary_results.items():
        draw_rows.append({"cells_per_donor": 1e7, "draw": 0, "pair": f"{fs}/{model_name}",
                          "auroc_median": res["auroc_median"], "perm_p": res["perm_p"]})
    draws = pd.DataFrame(draw_rows)
    draws.to_csv(OUT_DIR / "downsampling_draws.tsv", sep="\t", index=False, float_format="%.4f")


    print("\nF. influence and bootstrap")
    influence_rows, boot_rows = [], []
    boot_rng = np.random.default_rng(common.SEED)
    for fs, model_name in PAIRS:
        X, pca, n_plain = feature_matrix(base, fs, primary_idx)
        loo = Parallel(n_jobs=N_JOBS)(delayed(lodo_auroc)(X, y_primary, i, model_name, pca, n_plain)
                                      for i in range(len(y_primary)))
        for i, a in enumerate(loo):
            influence_rows.append({"pair": f"{fs}/{model_name}", "left_out_donor": donors[primary_idx][i],
                                   "group": meta["group"].to_numpy()[primary_idx][i], "auroc_median": a})
        oof = primary_results[(fs, model_name)]["oof_mean"]
        boots = []
        for _ in range(N_BOOT):
            sample = boot_rng.integers(0, len(y_primary), len(y_primary))
            if y_primary[sample].min() == y_primary[sample].max():
                continue
            boots.append(roc_auc_score(y_primary[sample], oof[sample]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        point = roc_auc_score(y_primary, oof)
        boot_rows.append({"pair": f"{fs}/{model_name}", "auroc_repeat_averaged_oof": point,
                          "ci95_low": lo, "ci95_high": hi, "n_boot": len(boots)})
        worst = int(np.argmin(loo))
        rows.append({"axis": "F", "setting": "leave-one-donor-out influence (min-max over 35)",
                     "n_pos": int(y_primary.sum()), "n_neg": int((1 - y_primary).sum()), "feature_set": fs,
                     "model": model_name, "auroc_median": float(np.median(loo)), "auroc_min": float(np.min(loo)),
                     "auroc_max": float(np.max(loo)), "extra_metric": "most_influential_donor",
                     "extra_value": np.nan, "extra_note": donors[primary_idx][worst]})
        rows.append({"axis": "F", "setting": "patient bootstrap 95% CI of AUROC", "n_pos": int(y_primary.sum()),
                     "n_neg": int((1 - y_primary).sum()), "feature_set": fs, "model": model_name,
                     "auroc_median": point, "auroc_min": lo, "auroc_max": hi})
        print(f"  F {fs}/{model_name}: LODO AUROC {np.min(loo):.3f}-{np.max(loo):.3f} "
              f"(lowest without {donors[primary_idx][worst]}); bootstrap AUROC {point:.3f} [95% CI {lo:.3f}-{hi:.3f}]")
    pd.DataFrame(influence_rows).to_csv(OUT_DIR / "influence.tsv", sep="\t", index=False, float_format="%.4f")
    pd.DataFrame(boot_rows).to_csv(OUT_DIR / "bootstrap_ci.tsv", sep="\t", index=False, float_format="%.4f")


    print("\nG. covariates")
    covars = np.c_[(meta["sex"] == "male").astype(float), meta["age"].astype(float)]
    for model_name in ["rf", "logreg"]:
        X = np.hstack([base["comp"][primary_idx], covars[primary_idx]])
        res = evaluate(X, y_primary, model_name)
        rows.append({"axis": "G", "setting": "composition + sex + age decade", "n_pos": int(y_primary.sum()),
                     "n_neg": int((1 - y_primary).sum()), "feature_set": "comp+covariates", "model": model_name,
                     **{k: res[k] for k in ["auroc_median", "auroc_min", "auroc_max", "bal_acc_median", "perm_p"]}})
        print(f"  G comp+sex+age {model_name} AUROC {res['auroc_median']:.3f} perm p {res['perm_p']:.3f}")
        res = evaluate(covars[primary_idx], y_primary, model_name)
        rows.append({"axis": "G", "setting": "sex + age decade only", "n_pos": int(y_primary.sum()),
                     "n_neg": int((1 - y_primary).sum()), "feature_set": "covariates", "model": model_name,
                     **{k: res[k] for k in ["auroc_median", "auroc_min", "auroc_max", "bal_acc_median", "perm_p"]}})
        print(f"  G sex+age only {model_name} AUROC {res['auroc_median']:.3f} perm p {res['perm_p']:.3f}")

    strata = (meta["age"].to_numpy()[primary_idx] >= 6).astype(int)
    for (fs, model_name), res in primary_results.items():
        X, pca, n_plain = feature_matrix(base, fs, primary_idx)
        rng = np.random.RandomState(common.SEED)
        shuffles = []
        for _ in range(N_PERM):
            y_shuffled = y_primary.copy()
            for s in [0, 1]:
                where = np.where(strata == s)[0]
                y_shuffled[where] = rng.permutation(y_primary[where])
            shuffles.append(y_shuffled)
        null = Parallel(n_jobs=N_JOBS)(delayed(one_perm)(X, ys, model_name, pca, n_plain, i % 10, None)
                                       for i, ys in enumerate(shuffles))
        p_strat = cu.perm_pvalue(res["auroc_median"], null)
        rows.append({"axis": "G", "setting": "age-stratified permutation (labels shuffled within <60 / >=60)",
                     "n_pos": int(y_primary.sum()), "n_neg": int((1 - y_primary).sum()), "feature_set": fs,
                     "model": model_name, "auroc_median": res["auroc_median"], "auroc_min": res["auroc_min"],
                     "auroc_max": res["auroc_max"], "perm_p": p_strat,
                     "extra_metric": "null_auroc_median", "extra_value": float(np.median(null))})
        print(f"  G {fs}/{model_name}: AUROC {res['auroc_median']:.3f}, age-stratified null median "
              f"{np.median(null):.3f}, p {p_strat:.3f}")

    for (fs, model_name), res in primary_results.items():
        sex = meta["sex"].to_numpy()[primary_idx]
        p_sex = mannwhitneyu(res["oof_mean"][sex == "male"], res["oof_mean"][sex == "female"]).pvalue
        rho, p_age = spearmanr(res["oof_mean"], meta["age"].to_numpy()[primary_idx])
        rows.append({"axis": "G", "setting": "OOF score vs sex (Mann-Whitney p)", "n_pos": int(y_primary.sum()),
                     "n_neg": int((1 - y_primary).sum()), "feature_set": fs, "model": model_name,
                     "extra_metric": "mwu_p_sex", "extra_value": p_sex})
        rows.append({"axis": "G", "setting": "OOF score vs age decade (Spearman)", "n_pos": int(y_primary.sum()),
                     "n_neg": int((1 - y_primary).sum()), "feature_set": fs, "model": model_name,
                     "extra_metric": "spearman_rho_age", "extra_value": rho, "extra_p": p_age})
        print(f"  G {fs}/{model_name}: score vs sex p {p_sex:.3f}; vs age rho {rho:.3f} (p {p_age:.3f})")

    summary = pd.DataFrame(rows)
    columns = ["axis", "setting", "n_pos", "n_neg", "feature_set", "model", "auroc_median", "auroc_min", "auroc_max",
               "bal_acc_median", "perm_p", "top3_celltypes", "jaccard_top20_genes_vs_primary",
               "extra_metric", "extra_value", "extra_p", "extra_note"]
    summary = summary.reindex(columns=columns)
    summary.to_csv(OUT_DIR / "sensitivity_summary.tsv", sep="\t", index=False, float_format="%.4f")
    plot_forest(summary[summary["axis"].isin(["A", "C", "D", "E", "F", "G"]) & summary["extra_metric"].isna()],
                FIG_DIR / "11_forest.png")
    plot_downsampling(draws, FIG_DIR / "11_downsampling.png")
    plot_specificity(pd.DataFrame(score_rows), FIG_DIR / "11_specificity.png")
    print(f"\nWrote {OUT_DIR / 'sensitivity_summary.tsv'} ({len(summary)} rows), specificity_scores.tsv, "
          f"influence.tsv, downsampling_draws.tsv, bootstrap_ci.tsv and figures")


if __name__ == "__main__":
    main()
