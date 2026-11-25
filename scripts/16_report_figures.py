
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common

FIG_DIR = common.RESULTS / "report" / "figures"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#b0b0b0"
GROUP_COLORS = {"control": BLUE, "mild": AQUA, "severe": YELLOW, "control_icu": ORANGE}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def clean(ax):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def panel_label(ax, letter):
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def fig_classification(out_path):
    cv = pd.read_csv(common.RESULTS / "classify" / "cv_metrics.tsv", sep="\t")
    cv = cv[(cv["task"] == "mild_vs_severe") & (cv["cv_scheme"] == "rep5fold")]
    isg = pd.read_csv(common.RESULTS / "isg" / "isg_cv_metrics.tsv", sep="\t")
    sens = pd.read_csv(common.RESULTS / "sensitivity" / "sensitivity_summary.tsv", sep="\t")
    draws = pd.read_csv(common.RESULTS / "sensitivity" / "downsampling_draws.tsv", sep="\t")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.1, 1.3, 1]})


    ax = axes[0]
    labels, rows = [], []
    for _, r in cv.iterrows():
        labels.append(f"{r['feature_set']} / {r['model']}")
        rows.append((r["auroc_median"], r["auroc_min"], r["auroc_max"], r["perm_p"]))
    r = isg[(isg["feature_set"] == "ifn_alpha_epithelial") & (isg["model"] == "isg_score")].iloc[0]
    labels.append("ISG score (no training)")
    rows.append((r["auroc_median"], r["auroc_min"], r["auroc_max"], r["perm_p"]))
    for i, (med, lo, hi, p) in enumerate(rows):
        color = BLUE if p < 0.05 else GREY
        ax.plot([lo, hi], [i, i], color=color, linewidth=2)
        ax.plot(med, i, "o", color=color, markersize=5)
        ax.text(1.0, i, f"p={p:.3f}", va="center", fontsize=6.5, color="#444444")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0.5, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_xlim(0.4, 1.08)
    ax.set_xlabel("AUROC (median, min-max over 10 CV repeats)")
    ax.set_title("Mild vs severe (14 v 21 patients)", fontsize=9)
    clean(ax)
    panel_label(ax, "a")


    ax = axes[1]
    picks = [("A", "primary (WHO 1-5 vs 6-8)"), ("A", "WHO <=5 vs >=7 (drop WHO 6)"), ("A", "WHO <=4 vs 8 (extremes)"),
             ("D", "exclude bloody-swab donors"), ("D", "cells with pct_mito < 20% (recomputed)"),
             ("D", "donors with >= 100 cells"), ("F", "patient bootstrap 95% CI of AUROC"),
             ("G", "age-stratified permutation (labels shuffled within <60 / >=60)")]
    names = ["primary", "drop WHO-6 patient", "extremes WHO<=4 vs 8", "no bloody swabs", "mito < 20%",
             ">= 100 cells/patient", "bootstrap 95% CI", "age-stratified null"]
    for i, ((axis, setting), name) in enumerate(zip(picks, names)):
        r = sens[(sens["axis"] == axis) & (sens["setting"] == setting) & (sens["feature_set"] == "comp")
                 & (sens["model"] == "rf")]
        if r.empty:
            continue
        r = r.iloc[0]
        p = r["perm_p"]
        color = BLUE if (np.isnan(p) or p < 0.05) else GREY
        ax.plot([r["auroc_min"], r["auroc_max"]], [i, i], color=color, linewidth=2)
        ax.plot(r["auroc_median"], i, "o", color=color, markersize=5)
        note = f"{int(r['n_pos'])}v{int(r['n_neg'])}" + ("" if np.isnan(p) else f", p={p:.3f}")
        ax.text(1.0, i, note, va="center", fontsize=6.5, color="#444444")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0.5, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_xlim(0.4, 1.15)
    ax.set_xlabel("AUROC, composition + random forest")
    ax.set_title("Sensitivity (n = severe v mild)", fontsize=9)
    clean(ax)
    panel_label(ax, "b")


    ax = axes[2]
    colors = {"comp/rf": BLUE, "pb_all/logreg": AQUA, "comb/rf": YELLOW}
    depths = sorted(draws["cells_per_donor"].unique())
    for pair, color in colors.items():
        sub = draws[draws["pair"] == pair]
        s = sub.groupby("cells_per_donor")["auroc_median"].agg(["median", "min", "max"]).reindex(depths)
        x = np.arange(len(depths))
        ax.plot(x, s["median"], marker="o", markersize=4, color=color, linewidth=2, label=pair)
        ax.fill_between(x, s["min"], s["max"], color=color, alpha=0.15)
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([str(int(t)) if t < 1e6 else "all" for t in depths])
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("cells per patient (random downsample)")
    ax.set_ylabel("AUROC (median of 10 draws, band = range)")
    ax.set_title("How many cells are needed?", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    clean(ax)
    panel_label(ax, "c")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def fig_interferon(out_path):
    gsea = pd.read_csv(common.RESULTS / "de" / "enrichment_gsea.tsv", sep="\t")
    hall = gsea[gsea["library"] == "hallmark"]
    terms = ["HALLMARK_INTERFERON_ALPHA_RESPONSE", "HALLMARK_INTERFERON_GAMMA_RESPONSE",
             "HALLMARK_TNFA_SIGNALING_VIA_NFKB", "HALLMARK_OXIDATIVE_PHOSPHORYLATION"]
    short = ["IFN-α", "IFN-γ", "TNF-α/NF-κB", "OXPHOS"]
    cts = ["all", "ciliated_cells", "secretory_cells", "goblet_cells", "squamous_cells", "deuterosomal_cells", "t_cells"]
    contrasts = [("covid_vs_control", "COVID vs control"), ("severe_vs_mild", "severe vs mild")]

    fig = plt.figure(figsize=(11, 3.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.6, 0.05, 1], wspace=0.08)
    ax = fig.add_subplot(grid[0])
    nes = np.full((len(cts), len(terms) * 2), np.nan)
    fdr = np.full_like(nes, np.nan)
    for c_i, (contrast, _) in enumerate(contrasts):
        for t_i, term in enumerate(terms):
            for r_i, ct in enumerate(cts):
                hit = hall[(hall["contrast"] == contrast) & (hall["ct"] == ct) & (hall["term"] == term)]
                if len(hit):
                    nes[r_i, c_i * len(terms) + t_i] = hit["nes"].iloc[0]
                    fdr[r_i, c_i * len(terms) + t_i] = hit["fdr"].iloc[0]
    image = ax.imshow(nes, cmap="RdBu_r", vmin=-3.7, vmax=3.7, aspect="auto")
    for i in range(nes.shape[0]):
        for j in range(nes.shape[1]):
            if not np.isnan(nes[i, j]):
                ax.text(j, i, f"{nes[i, j]:+.1f}{'*' if fdr[i, j] < 0.05 else ''}", ha="center", va="center",
                        fontsize=6.5, color="white" if abs(nes[i, j]) > 2.2 else "#1a1a1a")
            else:
                ax.text(j, i, "n/t", ha="center", va="center", fontsize=6, color="#888888")
    ax.set_xticks(range(nes.shape[1]))
    ax.set_xticklabels(short * 2, fontsize=7)
    ax.set_yticks(range(len(cts)))
    ax.set_yticklabels([c.replace("_cells", "").replace("all", "whole sample") for c in cts], fontsize=7)
    ax.axvline(len(terms) - 0.5, color="white", linewidth=3)
    for c_i, (_, label) in enumerate(contrasts):
        ax.text(c_i * len(terms) + (len(terms) - 1) / 2, -0.9, label, ha="center", fontsize=8, fontweight="bold")
    cax = fig.add_subplot(grid[1])
    fig.colorbar(image, cax=cax).set_label("NES (> 0 = up in first group)", fontsize=7)
    cax.tick_params(labelsize=6)
    panel_label(ax, "a")


    ax = fig.add_subplot(grid[2])
    pos = ax.get_position()
    ax.set_position([pos.x0 + 0.08, pos.y0, pos.width - 0.08, pos.height])
    obs = ad.read_h5ad(common.QC_H5AD, backed="r").obs
    cells = pd.DataFrame({"donor_id": obs["donor_id"].astype(str).to_numpy(),
                          "ct": obs["ct_coarse"].astype(str).to_numpy(),
                          "ct_detailed": obs["ct_detailed"].astype(str).to_numpy(),
                          "group": obs["group"].astype(str).to_numpy()})
    immune = ["T Cells", "B Cells", "Macrophages", "Dendritic Cells", "Plasmacytoid DCs", "Mast Cells", "Erythroblasts"]
    epi = cells[~cells["ct"].isin(immune)]
    frac = (epi["ct_detailed"] == "Interferon Responsive Ciliated Cells").groupby(epi["donor_id"]).mean() * 100
    group = epi.groupby("donor_id")["group"].first()
    order = ["control", "mild", "severe", "control_icu"]
    rng = np.random.default_rng(common.SEED)
    for i, g in enumerate(order):
        values = frac[group == g]
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(values)), values, s=14, color=GROUP_COLORS[g])
        ax.plot([i - 0.25, i + 0.25], [values.median()] * 2, color="#1a1a1a", linewidth=2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{g}\nn={int((group == g).sum())}" for g in order], fontsize=7)
    ax.set_ylabel("IFN-responsive ciliated cells\n(% of epithelial cells per patient)")
    ax.set_title("Interferon-responsive ciliated cells", fontsize=9)
    clean(ax)
    panel_label(ax, "b")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_attribution_perturbation(out_path):
    celltype = pd.read_csv(common.RESULTS / "attribution" / "celltype_importance.tsv", sep="\t")
    null = pd.read_csv(common.RESULTS / "attribution" / "single_type_null.tsv", sep="\t")
    effects = pd.read_csv(common.RESULTS / "perturb" / "perturbation_effects.tsv", sep="\t")
    targets = pd.read_csv(common.RESULTS / "targets" / "target_ranking.tsv", sep="\t")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.3]})


    ax = axes[0]
    sub = celltype[(celltype["model"] == "logreg") & celltype["n_donors"].notna()]
    for i, (_, r) in enumerate(sub.iterrows()):
        q95 = null[(null["ct"] == r["ct"]) & (null["model"] == "logreg")]["auroc"].quantile(0.95)
        color = BLUE if r["perm_p"] < 0.05 else GREY
        ax.bar(i, r["auroc_single_median"], color=color, width=0.7)
        ax.plot([i - 0.35, i + 0.35], [q95, q95], color="#1a1a1a", linewidth=1.5)
        ax.text(i, r["auroc_single_median"] + 0.02, f"p={r['perm_p']:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels([f"{c.replace('_cells', '').replace('all', 'whole')}\n{int(p)}v{int(n)}"
                        for c, p, n in zip(sub["ct"], sub["n_pos"], sub["n_neg"])], fontsize=7)
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC (median of 10 CV repeats)")
    ax.set_title("Pseudobulk of one cell type at a time (logistic regression)\n"
                 "black line = 95th percentile of its permutation null", fontsize=8)
    clean(ax)
    panel_label(ax, "a")


    ax = axes[1]
    ranked = targets[targets["meaningful_perturbation"]].copy()
    ranked["abs"] = ranked["perturb_mean_delta_p"].abs()
    top = ranked.sort_values("abs", ascending=False).drop_duplicates(["gene_symbol"]).head(12).iloc[::-1]
    main = effects[(effects["shift"] == "severe_to_mild") & (effects["alpha"] == 1.0)]
    y = np.arange(len(top))
    for model, offset, color, label in [("logreg", 0.2, BLUE, "gene alone"),
                                        ("logreg_module", -0.2, ORANGE, "gene + co-expression module")]:
        values = [main[(main["gene_id"] == g) & (main["ct"] == c) & (main["model"] == model)]["mean_delta_p"].iloc[0] * 100
                  for g, c in zip(top["gene_id"], top["ct"])]
        ax.barh(y + offset, values, height=0.38, color=color, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g} ({c.replace('_cells', '').replace('all', 'whole')}, {a})"
                        for g, c, a in zip(top["gene_symbol"], top["ct"], top["desired_drug_action"])], fontsize=7)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("change in predicted P(severe), percentage points\n(severe patients, gene moved to the mild median)")
    ax.set_title("In-silico perturbation (out-of-fold, all null p < 0.05 vs 200 matched random genes)", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    clean(ax)
    panel_label(ax, "b")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_classification(FIG_DIR / "fig2_classification.png")
    fig_interferon(FIG_DIR / "fig3_interferon.png")
    fig_attribution_perturbation(FIG_DIR / "fig4_attribution_perturbation.png")
    print(f"Wrote 3 figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
