"""Step 05: patient-level classifiers, cross-validation and permutation tests (contract C5).

Reads:  data/processed/patient_meta.tsv (C2), features_composition.tsv (C3),
        pseudobulk_coarse.h5ad (C4), all from step 03.
Writes: results/classify/oof_predictions.tsv, cv_metrics.tsv, perm_null.tsv
        results/classify/figures/05_roc_comp_vs_pb.png, 05_perm_null.png

Run from the repo root:  python scripts/05_classify.py
Quick test:              python scripts/05_classify.py --quick
"""
import sys
import numpy as np
import pandas as pd
import anndata as ad
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

import common
import classify_utils as cu

OUT_DIR = common.RESULTS / "classify"
FIG_DIR = OUT_DIR / "figures"

QUICK = "--quick" in sys.argv
N_REPEATS = 2 if QUICK else 10          # CV repeats (seeds 0-9)
N_PERM = {"logreg": 1000, "linsvm": 1000, "rf": 200}
if QUICK:
    N_PERM = {"logreg": 20, "linsvm": 20, "rf": 10}

# task name -> (groups that are class 0, groups that are class 1)
TASKS = {
    "mild_vs_severe": (["mild"], ["severe"]),
    "control_vs_covid": (["control"], ["mild", "severe"]),
}


def load_features():
    """C2 metadata, C3 composition and the whole-sample ("all") rows of C4 pseudobulk counts."""
    meta = pd.read_csv(common.PATIENT_META, sep="\t")
    comp = pd.read_csv(common.FEATURES_COMPOSITION, sep="\t")
    pbad = ad.read_h5ad(common.PSEUDOBULK_H5AD)
    pbad = pbad[pbad.obs["ct"] == "all"]
    counts = pbad.X.toarray() if hasattr(pbad.X, "toarray") else pbad.X
    pb = pd.DataFrame(counts, index=pbad.obs["donor_id"].astype(str).values)
    pb = pb.reindex(meta["donor_id"].astype(str))
    return meta, comp, pb.values


def log_cpm(counts):
    """log1p counts per million, per donor (uses only that donor's own counts)."""
    cpm = counts / counts.sum(axis=1, keepdims=True) * 1e6
    return np.log1p(cpm)


def make_feature_sets(meta, comp, pb):
    comp = comp.set_index("donor_id").loc[meta["donor_id"]]
    clr_cols = [c for c in comp.columns if c.startswith("clr__")]
    return {
        # name: (matrix, needs in-fold PCA, number of leading plain columns)
        "comp": (comp[clr_cols].values, False, 0),
        "pb_all": (log_cpm(pb), True, 0),
        "comb": (np.hstack([comp[clr_cols].values, log_cpm(pb)]), True, len(clr_cols)),
    }


def run_one(task, fs_name, model_name, X, y, donors, use_pca, n_plain):
    """Both CV schemes + the permutation test for one task x feature set x model."""
    oof_rows, metric_rows, null_rows, roc_data = [], [], [], {}
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    for scheme in cu.CV_SCHEMES:
        repeats = range(N_REPEATS) if scheme == "rep5fold" else [0]
        aurocs, bal_accs = [], []
        for rep in repeats:
            oof = cu.run_cv(X, y, model_name, scheme, rep, use_pca, n_plain)
            auc, bal = cu.auroc_balacc(y, oof)
            aurocs.append(auc)
            bal_accs.append(bal)
            for d, yt, s in zip(donors, y, oof):
                oof_rows.append([d, task, fs_name, model_name, scheme, rep, yt, s])
            if task == "mild_vs_severe" and scheme == "rep5fold" and rep == 0:
                roc_data[(fs_name, model_name)] = (y, oof)

        # permutation test on the primary scheme only (LOPO: not run)
        perm_p, n_perm = np.nan, 0
        if scheme == "rep5fold":
            n_perm = N_PERM[model_name]
            null = cu.permutation_null(X, y, model_name, scheme, n_perm,
                                       common.SEED, use_pca, n_plain)
            perm_p = cu.perm_pvalue(np.median(aurocs), null)
            for i, a in enumerate(null):
                null_rows.append([task, fs_name, model_name, i, a])

        metric_rows.append([task, fs_name, model_name, scheme, n_pos, n_neg,
                            np.median(aurocs), np.min(aurocs), np.max(aurocs),
                            np.median(bal_accs), perm_p, n_perm])
        print(f"{task} {fs_name} {model_name} {scheme}: "
              f"AUROC {np.median(aurocs):.3f} [{np.min(aurocs):.3f}-{np.max(aurocs):.3f}] "
              f"bal_acc {np.median(bal_accs):.3f} perm_p {perm_p}", flush=True)
    return oof_rows, metric_rows, null_rows, roc_data


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    meta, comp, pb = load_features()
    feature_sets = make_feature_sets(meta, comp, pb)

    # one job per (task, feature set, model); the jobs are independent, so they run in
    # parallel. Seeds are fixed inside each job, so results match a serial run.
    jobs = []
    for task, (neg, pos) in TASKS.items():
        use = meta["group"].isin(neg + pos).values
        donors = meta.loc[use, "donor_id"].values
        y = meta.loc[use, "group"].isin(pos).astype(int).values
        print(f"{task}: {int(y.sum())} positive, {int((1 - y).sum())} negative patients")
        for fs_name, (X_all, use_pca, n_plain) in feature_sets.items():
            for model_name in cu.MODEL_NAMES:
                jobs.append((task, fs_name, model_name, X_all[use], y, donors, use_pca, n_plain))
    results = Parallel(n_jobs=-1)(delayed(run_one)(*job) for job in jobs)

    oof_rows, metric_rows, null_rows = [], [], []
    roc_data = {}   # for the figure
    for oof_part, metric_part, null_part, roc_part in results:
        oof_rows += oof_part
        metric_rows += metric_part
        null_rows += null_part
        roc_data.update(roc_part)

    oof_df = pd.DataFrame(oof_rows, columns=["donor_id", "task", "feature_set", "model",
                                             "cv_scheme", "repeat", "y_true", "score"])
    metrics_df = pd.DataFrame(metric_rows, columns=[
        "task", "feature_set", "model", "cv_scheme", "n_pos", "n_neg", "auroc_median",
        "auroc_min", "auroc_max", "bal_acc_median", "perm_p", "n_perm"])
    null_df = pd.DataFrame(null_rows, columns=["task", "feature_set", "model", "perm_id", "auroc"])

    oof_df.to_csv(OUT_DIR / "oof_predictions.tsv", sep="\t", index=False)
    metrics_df.to_csv(OUT_DIR / "cv_metrics.tsv", sep="\t", index=False)
    null_df.to_csv(OUT_DIR / "perm_null.tsv", sep="\t", index=False)
    print("wrote", OUT_DIR / "oof_predictions.tsv", OUT_DIR / "cv_metrics.tsv",
          OUT_DIR / "perm_null.tsv")

    make_figures(roc_data, metrics_df, null_df)


def make_figures(roc_data, metrics_df, null_df):
    """ROC curves (repeat 0) and permutation nulls for mild vs severe."""
    row = metrics_df[(metrics_df["task"] == "mild_vs_severe")].iloc[0]
    n_label = f"n = {row['n_neg']} mild, {row['n_pos']} severe patients"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, fs_name in zip(axes, ["comp", "pb_all", "comb"]):
        for model_name in cu.MODEL_NAMES:
            y, oof = roc_data[(fs_name, model_name)]
            fpr, tpr, _ = roc_curve(y, oof)
            m = metrics_df[(metrics_df["task"] == "mild_vs_severe") &
                           (metrics_df["feature_set"] == fs_name) &
                           (metrics_df["model"] == model_name) &
                           (metrics_df["cv_scheme"] == "rep5fold")].iloc[0]
            ax.plot(fpr, tpr, label=f"{model_name} (median AUROC {m['auroc_median']:.2f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_title(f"{fs_name}: mild vs severe\n{n_label}", fontsize=10)
        ax.set_xlabel("False positive rate")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("True positive rate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_roc_comp_vs_pb.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True)
    for i, fs_name in enumerate(["comp", "pb_all", "comb"]):
        for j, model_name in enumerate(cu.MODEL_NAMES):
            ax = axes[i, j]
            null = null_df[(null_df["task"] == "mild_vs_severe") &
                           (null_df["feature_set"] == fs_name) &
                           (null_df["model"] == model_name)]["auroc"]
            m = metrics_df[(metrics_df["task"] == "mild_vs_severe") &
                           (metrics_df["feature_set"] == fs_name) &
                           (metrics_df["model"] == model_name) &
                           (metrics_df["cv_scheme"] == "rep5fold")].iloc[0]
            ax.hist(null, bins=20, color="lightgray")
            ax.axvline(m["auroc_median"], color="red")
            ax.set_title(f"{fs_name} / {model_name}\nobserved {m['auroc_median']:.2f}, "
                         f"perm p = {m['perm_p']:.3f} ({int(m['n_perm'])} perms)", fontsize=9)
            ax.set_xlabel("AUROC under shuffled labels")
    fig.suptitle("Permutation null, mild vs severe, " + n_label)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_perm_null.png", dpi=300)
    plt.close(fig)
    print("wrote figures to", FIG_DIR)


if __name__ == "__main__":
    main()
