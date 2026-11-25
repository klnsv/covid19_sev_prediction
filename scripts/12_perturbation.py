"""Step 12: in-silico perturbation of key genes (contract C11, CONTEXT.md section 7.2).

Question: if gene g in cell type c were moved towards its level in mild (or healthy)
patients, how much would the model's predicted P(severe) fall?

Reads:  data/processed/pseudobulk_coarse.h5ad (C4)
        results/de/key_gene_shortlist.tsv (C9)
Writes: results/perturb/perturbation_effects.tsv  (C11)
        results/perturb/dose_response.tsv          mean delta P over the alpha sweep (logreg, single gene)
        results/perturb/model_cv_auroc.tsv         CV AUROC of the model being perturbed, per cell type
        results/perturb/figures/12_effects.png
        results/perturb/figures/12_dose_response.png

Model: L2 logistic regression (C = 1, balanced) on standardised pseudobulk logCPM of
cell type c, mild vs severe donors that have c. Genes = the N_TOP_GENES most variable
genes of the training fold (among genes with mean logCPM >= MIN_MEAN_LOGCPM, as in
steps 05 and 09), plus every shortlist gene of c. The model is fit per fold
(stratified 5-fold, seeds 0-9) and each patient is perturbed only with a model that
did not see them. Results are averaged over the repeats.

Perturbation: x' = x + alpha * (m_ref - x) for gene g only. m_ref is the median logCPM
of g in the training fold's mild donors (severe_to_mild), the training fold's severe
donors (mild_to_severe), or the healthy controls (severe_to_control; controls are never
in the training data). The model is linear in the standardised genes, so the change in
log-odds is exactly coef_g * (x' - x) / sd_g.

Module version: the same shift applied to g's co-expression module (genes with
Spearman rho > 0.7 to g across the training donors), each gene towards its own m_ref.
Null: 200 random genes matched to g on training-fold mean and variance (10 x 10 quantile
bins); null_p = (1 + #random with |mean delta P| >= |observed|) / (1 + 200).
RF check: random forest (500 trees) on the same genes, single gene only, no null.

frac_flipped: among patients whose P(severe) starts on the side of 0.5 the shift moves
away from, the fraction that crosses 0.5.
This measures the model's dependence on a gene, not biological causality.

Run: python scripts/12_perturbation.py
"""
import numpy as np
import pandas as pd
import anndata as ad
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common

N_REPEATS = 10
N_NULL = 200
N_JOBS = 10
MIN_MEAN_LOGCPM = 1.0
N_TOP_GENES = 2000
MODULE_RHO = 0.7
N_BINS = 10
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
ALPHA_SWEEP = [round(a, 2) for a in np.arange(-1.0, 1.51, 0.25)]
SHIFTS = ["severe_to_mild", "severe_to_control", "mild_to_severe"]
SHIFTED_GROUP = {"severe_to_mild": "severe", "severe_to_control": "severe", "mild_to_severe": "mild"}

OUT_DIR = common.RESULTS / "perturb"
FIG_DIR = OUT_DIR / "figures"


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_cpm(counts):
    cpm = counts / counts.sum(axis=1, keepdims=True) * 1e6
    return np.log1p(cpm)


def load_blocks():
    """logCPM per cell type: dict ct -> (DataFrame donors x genes, group per donor)."""
    pb = ad.read_h5ad(common.PSEUDOBULK_H5AD)
    obs = pb.obs.copy()
    obs["ct"] = obs["ct"].astype(str)
    obs["group"] = obs["group"].astype(str)
    obs["donor_id"] = obs["donor_id"].astype(str)
    logcpm = log_cpm(pb.X.toarray().astype(float))
    blocks = {}
    for ct in obs["ct"].unique():
        rows = np.where(obs["ct"] == ct)[0]
        table = pd.DataFrame(logcpm[rows], index=obs["donor_id"].to_numpy()[rows], columns=pb.var_names)
        blocks[ct] = (table, obs["group"].to_numpy()[rows])
    symbols = pb.var["gene_symbol"].astype(str)
    return blocks, symbols


def matched_null_genes(train_X, target_col, rng):
    """N_NULL random genes in the same mean x variance quantile bin as the target."""
    means = train_X.mean(axis=0)
    variances = train_X.var(axis=0)
    mean_bin = np.minimum((rankdata(means) - 1) * N_BINS // len(means), N_BINS - 1)
    var_bin = np.minimum((rankdata(variances) - 1) * N_BINS // len(variances), N_BINS - 1)
    same = np.where((mean_bin == mean_bin[target_col]) & (var_bin == var_bin[target_col]))[0]
    same = same[same != target_col]
    return rng.choice(same, size=N_NULL, replace=len(same) < N_NULL)


def standardised_ranks(train_X):
    """Per-gene ranks across training donors, standardised, so Spearman rho = mean product."""
    ranks = np.apply_along_axis(rankdata, 0, train_X)
    return (ranks - ranks.mean(axis=0)) / (ranks.std(axis=0) + 1e-12)


def module_columns(ranks, target_col):
    """Genes with Spearman rho > MODULE_RHO to the target across training donors (incl. target)."""
    rho = ranks.T @ ranks[:, target_col] / len(ranks)
    return np.where(rho > MODULE_RHO)[0]


def one_fold(X, groups, controls_X, targets, rep, train_idx, test_idx):
    """Fit the fold's models; return per-patient P and P' for every target, shift and alpha."""
    y = (groups == "severe").astype(int)
    train_X = X[train_idx]
    mu = train_X.mean(axis=0)
    sd = train_X.std(axis=0)
    sd[sd == 0] = 1.0
    logreg = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=rep)
    logreg.fit((train_X - mu) / sd, y[train_idx])
    coef, intercept = logreg.coef_[0], logreg.intercept_[0]
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=rep, n_jobs=1)
    rf.fit(train_X, y[train_idx])

    test_X = X[test_idx]
    logit = intercept + ((test_X - mu) / sd) @ coef
    p_base = sigmoid(logit)
    p_base_rf = rf.predict_proba(test_X)[:, 1]
    test_groups = groups[test_idx]
    oof_logreg = dict(zip(test_idx, p_base))

    references = {
        "severe_to_mild": np.median(train_X[groups[train_idx] == "mild"], axis=0),
        "mild_to_severe": np.median(train_X[groups[train_idx] == "severe"], axis=0),
        "severe_to_control": np.median(controls_X, axis=0) if len(controls_X) else None,
    }
    rng = np.random.RandomState(1000 * rep + int(test_idx[0]))
    ranks = standardised_ranks(train_X)
    step = coef / sd    # log-odds change per unit logCPM

    rows = []
    for target_col in targets:
        null_cols = matched_null_genes(train_X, target_col, rng)
        module = module_columns(ranks, target_col)
        for shift in SHIFTS:
            m_ref = references[shift]
            mask = test_groups == SHIFTED_GROUP[shift]
            if m_ref is None or not mask.any():
                continue
            x = test_X[mask]
            gap = m_ref - x    # patients x genes
            for alpha in ALPHA_SWEEP:
                single = sigmoid(logit[mask] + alpha * gap[:, target_col] * step[target_col]) - p_base[mask]
                record = {"rep": rep, "target_col": target_col, "shift": shift, "alpha": alpha,
                          "patients": test_idx[mask], "p_base": p_base[mask], "delta": single}
                if alpha in ALPHA_GRID:
                    module_shift = (alpha * gap[:, module] * step[module]).sum(axis=1)
                    record["delta_module"] = sigmoid(logit[mask] + module_shift) - p_base[mask]
                    record["null_delta"] = sigmoid(logit[mask][:, None] + alpha * gap[:, null_cols] * step[null_cols]) \
                        - p_base[mask][:, None]    # patients x N_NULL
                    record["module_size"] = len(module)
                    x_rf = x.copy()
                    x_rf[:, target_col] = x[:, target_col] + alpha * gap[:, target_col]
                    record["delta_rf"] = rf.predict_proba(x_rf)[:, 1] - p_base_rf[mask]
                    record["p_base_rf"] = p_base_rf[mask]
                rows.append(record)
    return rows, oof_logreg


def per_patient_mean(records, key, patients):
    """Concatenate one field over folds/repeats and average it per patient."""
    return pd.Series(np.concatenate([r[key] for r in records])).groupby(patients).mean()


def frac_flipped(p_base, delta, towards_mild):
    """Fraction of patients starting on the shifted-from side of 0.5 that cross it."""
    start = p_base >= 0.5 if towards_mild else p_base < 0.5
    after = (p_base + delta) < 0.5 if towards_mild else (p_base + delta) >= 0.5
    return float((start & after).sum() / start.sum()) if start.sum() else np.nan


def summarise_ct(ct, records, symbols, shortlist_ids):
    """Average per patient over repeats, then summarise per gene x shift x alpha."""
    effect_rows, sweep_rows = [], []
    for gene_id in shortlist_ids:
        for shift in SHIFTS:
            mine = [r for r in records if r["gene_id"] == gene_id and r["shift"] == shift]
            if not mine:
                continue
            towards_mild = shift != "mild_to_severe"
            for alpha in ALPHA_SWEEP:
                at = [r for r in mine if r["alpha"] == alpha]
                patients = np.concatenate([r["patients"] for r in at])
                delta = per_patient_mean(at, "delta", patients)
                sweep_rows.append({"gene_id": gene_id, "gene_symbol": symbols[gene_id], "ct": ct,
                                   "shift": shift, "alpha": alpha, "mean_delta_p": delta.mean()})
                if alpha not in ALPHA_GRID:
                    continue

                p_base = per_patient_mean(at, "p_base", patients)
                null = pd.DataFrame(np.vstack([r["null_delta"] for r in at])).groupby(patients).mean()
                null_means = null.mean(axis=0).to_numpy()    # one mean delta P per random gene
                observed = delta.mean()
                null_p = (1 + np.sum(np.abs(null_means) >= abs(observed))) / (1 + len(null_means))
                base = {"gene_id": gene_id, "gene_symbol": symbols[gene_id], "ct": ct, "shift": shift,
                        "alpha": alpha, "n_patients": len(delta)}
                effect_rows.append({**base, "model": "logreg", "mean_delta_p": observed,
                                    "median_delta_p": delta.median(),
                                    "frac_flipped": frac_flipped(p_base, delta, towards_mild),
                                    "null_p": null_p, "null_n": len(null_means),
                                    "null_mean_abs_q95": float(np.quantile(np.abs(null_means), 0.95)),
                                    "module_size": np.nan})

                module = per_patient_mean(at, "delta_module", patients)
                effect_rows.append({**base, "model": "logreg_module", "mean_delta_p": module.mean(),
                                    "median_delta_p": module.median(),
                                    "frac_flipped": frac_flipped(p_base, module, towards_mild),
                                    "null_p": np.nan, "null_n": 0, "null_mean_abs_q95": np.nan,
                                    "module_size": float(np.median([r["module_size"] for r in at]))})

                rf = per_patient_mean(at, "delta_rf", patients)
                p_rf = per_patient_mean(at, "p_base_rf", patients)
                effect_rows.append({**base, "model": "rf", "mean_delta_p": rf.mean(),
                                    "median_delta_p": rf.median(),
                                    "frac_flipped": frac_flipped(p_rf, rf, towards_mild),
                                    "null_p": np.nan, "null_n": 0, "null_mean_abs_q95": np.nan,
                                    "module_size": np.nan})
    return effect_rows, sweep_rows


def run_ct(ct, blocks, symbols, shortlist_ids):
    table, group = blocks[ct]
    keep = np.isin(group, ["mild", "severe"])
    X_df, groups = table[keep], group[keep]
    controls_df = table[group == "control"]

    # the expression filter is applied again inside each fold; here it only sets the
    # candidate pool, so no test-donor information reaches the model's gene choice
    pool = X_df.columns[(X_df.mean(axis=0) >= MIN_MEAN_LOGCPM) | X_df.columns.isin(shortlist_ids)]
    X_all = X_df[pool].to_numpy()
    controls_X = controls_df[pool].to_numpy()
    gene_ids = np.asarray(pool)
    y = (groups == "severe").astype(int)
    forced = np.where(np.isin(gene_ids, shortlist_ids))[0]

    jobs = []
    for rep in range(N_REPEATS):
        for train_idx, test_idx in StratifiedKFold(5, shuffle=True, random_state=rep).split(X_all, y):
            train = X_all[train_idx]
            variance = np.where(train.mean(axis=0) >= MIN_MEAN_LOGCPM, train.var(axis=0), -1.0)
            top = np.argsort(variance)[::-1][:N_TOP_GENES]
            cols = np.union1d(top, forced)
            targets = [int(np.where(gene_ids[cols] == g)[0][0]) for g in shortlist_ids]
            jobs.append((rep, train_idx, test_idx, cols, targets))

    results = Parallel(n_jobs=N_JOBS)(
        delayed(one_fold)(X_all[:, cols], groups, controls_X[:, cols], targets, rep, train_idx, test_idx)
        for rep, train_idx, test_idx, cols, targets in jobs)

    records, oof_by_rep = [], {}
    for (rep, _, _, cols, _), (rows, oof) in zip(jobs, results):
        for r in rows:
            r["gene_id"] = gene_ids[cols[r.pop("target_col")]]    # fold-local column -> gene id
        records += rows
        oof_by_rep.setdefault(rep, {}).update(oof)
    aurocs = []
    for oof in oof_by_rep.values():
        idx = np.array(sorted(oof))
        aurocs.append(roc_auc_score(y[idx], [oof[i] for i in idx]))

    effects, sweep = summarise_ct(ct, records, symbols, shortlist_ids)
    model_row = {"ct": ct, "n_severe": int(y.sum()), "n_mild": int((1 - y).sum()),
                 "n_controls": len(controls_df), "n_genes_median": int(np.median([len(j[3]) for j in jobs])),
                 "auroc_median": float(np.median(aurocs)), "auroc_min": float(np.min(aurocs)),
                 "auroc_max": float(np.max(aurocs))}
    return effects, sweep, model_row


def plot_effects(effects, out_path):
    sub = effects[(effects["shift"] == "severe_to_mild") & (effects["alpha"] == 1.0)
                  & (effects["model"] == "logreg")]
    sub = sub.sort_values("mean_delta_p")
    labels = [f"{s} ({c.replace('_cells', '')})" for s, c in zip(sub["gene_symbol"], sub["ct"])]
    colors = ["#eb6834" if p < 0.05 else "#b0b0b0" for p in sub["null_p"]]
    fig, ax = plt.subplots(figsize=(8, 0.22 * len(sub) + 1.8))
    y = np.arange(len(sub))
    ax.barh(y, sub["mean_delta_p"], color=colors)
    ax.scatter(-sub["null_mean_abs_q95"], y, marker="|", color="#1a1a1a", s=40,
               label="matched-null 95th pct (|mean ΔP|)")
    ax.scatter(sub["null_mean_abs_q95"], y, marker="|", color="#1a1a1a", s=40)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("mean ΔP(severe) for severe patients, gene moved fully to the mild median (α = 1)")
    ax.set_title("In-silico perturbation, severe → mild (logistic regression, out-of-fold)\n"
                 "orange = beyond matched random genes (null p < 0.05); n per cell type in model_cv_auroc.tsv",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_dose_response(sweep, effects, out_path):
    grid = effects[(effects["shift"] == "severe_to_mild") & (effects["alpha"] == 1.0)
                   & (effects["model"] == "logreg")]
    top = grid.reindex(grid["mean_delta_p"].abs().sort_values(ascending=False).index).head(6)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    colors = {"severe_to_mild": "#2a78d6", "severe_to_control": "#1baf7a", "mild_to_severe": "#eb6834"}
    for ax, (_, row) in zip(axes.ravel(), top.iterrows()):
        for shift, color in colors.items():
            curve = sweep[(sweep["gene_id"] == row["gene_id"]) & (sweep["ct"] == row["ct"])
                          & (sweep["shift"] == shift)]
            if len(curve):
                ax.plot(curve["alpha"], curve["mean_delta_p"], marker="o", markersize=3, linewidth=2,
                        color=color, label=shift)
        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.axvline(0, color="#888888", linewidth=0.8)
        ax.set_title(f"{row['gene_symbol']} ({row['ct'].replace('_cells', '')}), null p = {row['null_p']:.3f}",
                     fontsize=9)
        ax.set_xlabel("α (0 = unchanged, 1 = at reference median)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("mean ΔP(severe)")
    axes[1, 0].set_ylabel("mean ΔP(severe)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Dose-response of predicted severity (6 genes with the largest |ΔP| at α = 1, severe → mild)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    blocks, symbols = load_blocks()
    shortlist = pd.read_csv(common.RESULTS / "de" / "key_gene_shortlist.tsv", sep="\t")

    all_effects, all_sweep, model_rows = [], [], []
    for ct in shortlist["ct"].unique():
        ids = list(dict.fromkeys(shortlist.loc[shortlist["ct"] == ct, "gene_id"]))
        effects, sweep, model_row = run_ct(ct, blocks, symbols, ids)
        all_effects += effects
        all_sweep += sweep
        model_rows.append(model_row)
        print(f"  {ct:18s} {len(ids):2d} genes | model CV AUROC {model_row['auroc_median']:.3f} "
              f"[{model_row['auroc_min']:.3f}-{model_row['auroc_max']:.3f}], "
              f"{model_row['n_severe']} severe / {model_row['n_mild']} mild / {model_row['n_controls']} controls, "
              f"~{model_row['n_genes_median']} genes in the model")

    columns = ["gene_id", "gene_symbol", "ct", "shift", "alpha", "model", "n_patients", "mean_delta_p",
               "median_delta_p", "frac_flipped", "null_p", "null_n", "null_mean_abs_q95", "module_size"]
    effects = pd.DataFrame(all_effects)[columns]
    effects.to_csv(OUT_DIR / "perturbation_effects.tsv", sep="\t", index=False, float_format="%.6g")
    sweep = pd.DataFrame(all_sweep)
    sweep.to_csv(OUT_DIR / "dose_response.tsv", sep="\t", index=False, float_format="%.6g")
    pd.DataFrame(model_rows).to_csv(OUT_DIR / "model_cv_auroc.tsv", sep="\t", index=False, float_format="%.4f")

    plot_effects(effects, FIG_DIR / "12_effects.png")
    plot_dose_response(sweep, effects, FIG_DIR / "12_dose_response.png")

    main_rows = effects[(effects["shift"] == "severe_to_mild") & (effects["alpha"] == 1.0)]
    logreg_rows = main_rows[main_rows["model"] == "logreg"].set_index(["gene_symbol", "ct"])
    table = main_rows.pivot_table(index=["gene_symbol", "ct"], columns="model", values="mean_delta_p") \
        .join(logreg_rows[["null_p", "frac_flipped"]])
    print("\nsevere_to_mild, alpha = 1 (mean delta P per model; null_p and frac_flipped from logreg):")
    print(table.sort_values("logreg").to_string(float_format=lambda v: f"{v:+.4f}"))
    print(f"\nwrote perturbation_effects.tsv ({len(effects):,} rows), dose_response.tsv, model_cv_auroc.tsv "
          f"and figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
