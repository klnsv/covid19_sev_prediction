"""Step 13: drug-target prioritisation of the key genes (contract C13, CONTEXT.md 7.4).

Reads:  results/de/key_gene_shortlist.tsv (C9)
        results/perturb/perturbation_effects.tsv (C11)
        results/perturb/model_cv_auroc.tsv (step 12)
        data/processed/external/target_annotations.tsv (C10, step 00)
        data/external/interactions.tsv (DGIdb, for interaction types)
Writes: results/targets/target_ranking.tsv (C13, every shortlist gene x cell type, ranked)
        results/targets/figures/13_target_ranking.png

Meaningful perturbation (CONTEXT.md 7.2), for severe_to_mild at alpha = 1:
  logreg null_p < 0.05, logreg mean delta P < 0 (moving the gene towards mild lowers
  predicted severity), the random-forest check has the same sign (delta P <= 0), and
  the cell type's perturbation model predicts severity (CV AUROC median >= MIN_MODEL_AUROC;
  perturbing a model at chance level says nothing).
Directional drugs: DGIdb interactions whose type matches the desired action
  (inhibit: inhibitor/antagonist/antibody/blocker/negative modulator/inverse agonist;
  activate: agonist/activator/positive modulator/inducer/potentiator). Untyped links
  (e.g. pharmacogenomic or text-mined) do not count; the undirected DGIdb counts are kept
  for transparency (dgidb_n_drugs_any, dgidb_n_approved_any).
Score (additive, weights below, each component scaled to 0-1 before weighting):
  druggable        HPA protein class or DGIdb category says druggable         x W_DRUGGABLE
  directional drug >= 1 DGIdb drug acting in the desired direction             x W_DIRECTIONAL
  approved         >= 1 such drug is approved                                  x W_APPROVED
  covid evidence   Open Targets COVID-19 score / max score among candidates    x W_COVID
  perturbation     |logreg mean delta P| / max among candidates (0 if not
                   meaningful)                                                 x W_PERTURB
  safety           broadly expressed (> 10 TPM in >= BROAD_TISSUES GTEx
                   tissues) subtracts                                          x W_BROAD
desired_drug_action: up_in_severe -> inhibit, down_in_severe -> activate.
Ranks are among genes with a meaningful perturbation first; the rest follow (rank NaN).

Run from the repo root: python scripts/13_targets.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common

W_DRUGGABLE = 1.0
W_DIRECTIONAL = 2.0
W_APPROVED = 1.0
W_COVID = 1.0
W_PERTURB = 2.0
W_BROAD = 1.0
BROAD_TISSUES = 45            # of 54 GTEx tissues
MIN_MODEL_AUROC = 0.6
INHIBIT_TYPES = ["inhibitor", "antagonist", "antibody", "blocker", "negative modulator", "inverse agonist"]
ACTIVATE_TYPES = ["agonist", "activator", "positive modulator", "inducer", "potentiator"]
DRUGGABLE_HPA = ["Enzymes", "Transporters", "G-protein coupled receptors", "Voltage-gated ion channels",
                 "Predicted secreted proteins", "CD markers", "Nuclear receptors", "Potential drug targets",
                 "FDA approved drug targets"]
DRUGGABLE_DGIDB = ["DRUGGABLE GENOME", "KINASE", "SERINE THREONINE KINASE", "TYROSINE KINASE", "ENZYME",
                   "TRANSPORTER", "ABC TRANSPORTER", "ION CHANNEL", "G PROTEIN COUPLED RECEPTOR",
                   "CELL SURFACE", "PROTEASE", "NUCLEAR HORMONE RECEPTOR", "EXTERNAL SIDE OF PLASMA MEMBRANE",
                   "GROWTH FACTOR", "LIPASE", "PHOSPHOLIPASE", "PROTEIN PHOSPHATASE"]
OUT_DIR = common.RESULTS / "targets"
FIG_DIR = OUT_DIR / "figures"


def is_druggable(row):
    hpa = str(row["hpa_protein_class"])
    dgidb = str(row["dgidb_categories"]).upper()
    return any(c in hpa for c in DRUGGABLE_HPA) or any(c in dgidb.split(";") for c in DRUGGABLE_DGIDB)


def directional_drugs(interactions, gene, action):
    """Drugs whose DGIdb interaction type matches the desired action."""
    wanted = INHIBIT_TYPES if action == "inhibit" else ACTIVATE_TYPES
    rows = interactions[interactions["gene_name"] == gene]
    if rows.empty:
        return 0, 0, ""
    types = rows["interaction_types"].fillna("").str.split("|")
    match = rows[types.apply(lambda t: any(x in wanted for x in t))].drop_duplicates("drug_name")
    is_approved = match["drug_is_approved"].astype(str).str.lower() == "true"
    # up to 5 example drugs, approved ones first
    examples = list(match.loc[is_approved, "drug_name"]) + list(match.loc[~is_approved, "drug_name"])
    return len(match), int(is_approved.sum()), ";".join(examples[:5])


def perturbation_summary(effects, model_auroc):
    main = effects[(effects["shift"] == "severe_to_mild") & (effects["alpha"] == 1.0)]
    logreg = main[main["model"] == "logreg"].set_index(["gene_id", "ct"])
    module = main[main["model"] == "logreg_module"].set_index(["gene_id", "ct"])
    rf = main[main["model"] == "rf"].set_index(["gene_id", "ct"])
    table = pd.DataFrame({
        "perturb_mean_delta_p": logreg["mean_delta_p"],
        "perturb_null_p": logreg["null_p"],
        "perturb_frac_flipped": logreg["frac_flipped"],
        "perturb_module_delta_p": module["mean_delta_p"],
        "perturb_rf_delta_p": rf["mean_delta_p"],
    }).reset_index()
    table["model_cv_auroc"] = table["ct"].map(model_auroc)
    table["meaningful_perturbation"] = (table["perturb_null_p"] < 0.05) & (table["perturb_mean_delta_p"] < 0) \
        & (table["perturb_rf_delta_p"] <= 0) & (table["model_cv_auroc"] >= MIN_MODEL_AUROC)
    return table


def rationale(row):
    parts = [f"{row['direction'].replace('_', ' ')} in {row['ct']}"]
    parts.append(f"perturbation dP {row['perturb_mean_delta_p']:+.3f} (null p {row['perturb_null_p']:.3f}, "
                 f"RF {row['perturb_rf_delta_p']:+.3f}, model AUROC {row['model_cv_auroc']:.2f})")
    parts.append("druggable class" if row["druggable"] else "not in a druggable class")
    if row["n_directional_drugs"] > 0:
        parts.append(f"{int(row['n_directional_drugs'])} drug(s) that {row['desired_drug_action']} it "
                     f"({int(row['n_approved_drugs'])} approved), e.g. {row['example_drugs']}")
    else:
        parts.append(f"no DGIdb drug that {row['desired_drug_action']}s it "
                     f"({int(row['dgidb_n_drugs_any'])} untyped/opposite links)")
    if not np.isnan(row["ot_covid19_score"]):
        parts.append(f"Open Targets COVID-19 score {row['ot_covid19_score']:.3f}")
    if row["tissue_specificity_flag"] == "broad":
        parts.append(f"broadly expressed ({int(row['gtex_n_tissues_tpm_gt10'])} GTEx tissues > 10 TPM)")
    return "; ".join(parts)


def plot_ranking(ranking, out_path):
    top = ranking.dropna(subset=["rank"]).head(15).iloc[::-1]
    if top.empty:
        return
    components = {
        "druggable class": W_DRUGGABLE * top["druggable"].astype(float),
        "drug in desired direction": W_DIRECTIONAL * (top["n_directional_drugs"] > 0).astype(float),
        "approved": W_APPROVED * (top["n_approved_drugs"] > 0).astype(float),
        "COVID-19 evidence": top["component_covid"],
        "perturbation": top["component_perturb"],
    }
    colors = {"druggable class": "#2a78d6", "drug in desired direction": "#eb6834", "approved": "#e87ba4",
              "COVID-19 evidence": "#1baf7a", "perturbation": "#eda100"}
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(top) + 1.8))
    left = np.zeros(len(top))
    y = np.arange(len(top))
    for name, values in components.items():
        ax.barh(y, values, left=left, color=colors[name], label=name, edgecolor="white", linewidth=1)
        left += values.to_numpy()
    penalty = W_BROAD * (top["tissue_specificity_flag"] == "broad").astype(float)
    ax.barh(y, -penalty, color="#b0b0b0", label="broad expression (penalty)", edgecolor="white", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g} ({c.replace('_cells', '')}, {a})" for g, c, a in
                        zip(top["gene_symbol"], top["ct"], top["desired_drug_action"])], fontsize=8)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("score (sum of weighted components)")
    ax.set_title("Drug-target prioritisation: genes with a meaningful perturbation effect\n"
                 "(n = 21 severe / 14 mild patients; per-cell-type n in C11/C13)", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    shortlist = pd.read_csv(common.RESULTS / "de" / "key_gene_shortlist.tsv", sep="\t")
    effects = pd.read_csv(common.RESULTS / "perturb" / "perturbation_effects.tsv", sep="\t")
    annotations = pd.read_csv(common.PROCESSED / "external" / "target_annotations.tsv", sep="\t")
    interactions = pd.read_csv(common.EXTERNAL / "interactions.tsv", sep="\t", comment="#")
    model_auroc = pd.read_csv(common.RESULTS / "perturb" / "model_cv_auroc.tsv", sep="\t") \
        .set_index("ct")["auroc_median"]

    table = shortlist[["gene_id", "gene_symbol", "ct", "direction"]].merge(
        perturbation_summary(effects, model_auroc), on=["gene_id", "ct"], how="left")
    table = table.merge(annotations, on="gene_symbol", how="left")
    table["druggable"] = table.apply(is_druggable, axis=1)
    table["dgidb_n_drugs_any"] = table["dgidb_n_drugs"].fillna(0).astype(int)
    table["dgidb_n_approved_any"] = table["dgidb_n_approved_drugs"].fillna(0).astype(int)
    n_tissues = table["gtex_n_tissues_tpm_gt10"]
    table["tissue_specificity_flag"] = np.where(n_tissues >= BROAD_TISSUES, "broad",
                                                np.where(n_tissues.isna(), "unknown", "restricted"))
    table["desired_drug_action"] = np.where(table["direction"] == "up_in_severe", "inhibit", "activate")
    drugs = [directional_drugs(interactions, gene, action)
             for gene, action in zip(table["gene_symbol"], table["desired_drug_action"])]
    table["n_directional_drugs"] = [d[0] for d in drugs]
    table["n_approved_drugs"] = [d[1] for d in drugs]
    table["example_drugs"] = [d[2] for d in drugs]

    meaningful = table["meaningful_perturbation"].fillna(False)
    max_ot = table.loc[meaningful, "ot_covid19_score"].max()
    max_dp = table.loc[meaningful, "perturb_mean_delta_p"].abs().max()
    table["component_covid"] = W_COVID * (table["ot_covid19_score"].fillna(0) / max_ot if max_ot > 0 else 0)
    table["component_perturb"] = np.where(meaningful, W_PERTURB * table["perturb_mean_delta_p"].abs() / max_dp, 0.0)
    table["score"] = (W_DRUGGABLE * table["druggable"] + W_DIRECTIONAL * (table["n_directional_drugs"] > 0)
                      + W_APPROVED * (table["n_approved_drugs"] > 0)
                      + table["component_covid"] + table["component_perturb"]
                      - W_BROAD * (table["tissue_specificity_flag"] == "broad"))
    table["meaningful_perturbation"] = meaningful
    table = table.sort_values(["meaningful_perturbation", "score"], ascending=False).reset_index(drop=True)
    table["rank"] = np.where(table["meaningful_perturbation"], np.arange(1, len(table) + 1), np.nan)
    table["rationale"] = table.apply(rationale, axis=1)

    columns = ["gene_symbol", "ct", "direction", "desired_drug_action", "perturb_mean_delta_p", "druggable",
               "n_approved_drugs", "example_drugs", "ot_covid19_score", "tissue_specificity_flag", "score", "rank",
               "rationale", "gene_id", "meaningful_perturbation", "perturb_null_p", "perturb_module_delta_p",
               "perturb_rf_delta_p", "model_cv_auroc", "n_directional_drugs", "dgidb_n_drugs_any",
               "dgidb_n_approved_any", "hpa_protein_class", "dgidb_categories", "gtex_n_tissues_tpm_gt10",
               "component_covid", "component_perturb"]
    out = table[columns]
    out.to_csv(OUT_DIR / "target_ranking.tsv", sep="\t", index=False, float_format="%.4g")
    plot_ranking(out, FIG_DIR / "13_target_ranking.png")

    print(f"{int(meaningful.sum())} of {len(table)} shortlist rows have a meaningful perturbation effect")
    shown = ["rank", "gene_symbol", "ct", "desired_drug_action", "score", "perturb_mean_delta_p", "druggable",
             "n_directional_drugs", "n_approved_drugs", "example_drugs", "ot_covid19_score",
             "tissue_specificity_flag"]
    print(out.dropna(subset=["rank"]).head(15)[shown].to_string(index=False))
    print(f"\nwrote {OUT_DIR / 'target_ranking.tsv'} and figures/13_target_ranking.png")


if __name__ == "__main__":
    main()
