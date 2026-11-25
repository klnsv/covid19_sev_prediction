"""Step 10: key severity-gene shortlist.

Reads:  results/de/de_<ct>_severe_vs_mild.tsv (step 07)
        data/processed/pseudobulk_coarse.h5ad (for the direction of genes not DE-tested)
        results/attribution/gene_importance.tsv (step 09)
        results/attribution/celltype_importance.tsv (step 09)
        data/external/gene_sets/hallmark_interferon_v2024.1.Hs.gmt (ISG sets, step 06)
Writes: results/de/key_gene_shortlist.tsv (<= 50 rows)
        results/de/key_gene_candidates.tsv (every candidate before the cap)
        results/de/figures/10_key_genes.png

A gene x cell type is a candidate if:
  de    : padj < 0.05 for severe_vs_mild in that cell type, or
  model : in the top 50 of that cell type's gene importance (logreg |coef| or RF |SHAP|)
          AND in the top 50 in >= 50% of the CV fits (stability), or
  both  : both of the above.
Ribosomal (RPL*/RPS*) and mitochondrial (MT-*) genes are left out: they track
cell size/quality and dominate some lists (e.g. every T-cell DE hit), not disease biology.
Selection (balanced, so neither evidence type crowds out the other):
every 'both' row, then per cell type up to PER_CT_EACH 'de'-only rows (lowest padj)
and PER_CT_EACH 'model'-only rows (best importance rank). The 'all' (whole-sample)
pseudobulk mostly reflects composition, so its genes are kept only when evidence = both.
role_summary is filled from ROLES below (one line per gene).
Genes picked by the model but too lowly expressed to enter the DESeq2 test have no
log2fc/padj; their direction comes from mean logCPM (severe - mild) in that cell type.

Run from the repo root: python scripts/10_key_genes.py
"""
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common

PADJ_CUTOFF = 0.05
TOP_K = 50
MIN_STABILITY = 0.5
MAX_ROWS = 50
PER_CT_EACH = 4
DE_DIR = common.RESULTS / "de"
ATTR_DIR = common.RESULTS / "attribution"
ISG_GMT = common.EXTERNAL / "gene_sets" / "hallmark_interferon_v2024.1.Hs.gmt"
EVIDENCE_ORDER = {"both": 0, "de": 1, "model": 2}

# One-line roles for shortlisted genes. Written from general biology knowledge (Claude);
# not individually literature-checked. The report cites sources for the genes it discusses.
ROLES = {
    "KRT16": "stress/wound-induced keratin of activated, injured epithelium",
    "NEK6": "NIMA-related serine/threonine kinase, mitotic progression",
    "S100A2": "calcium-binding S100 protein of basal/squamous epithelium, stress response",
    "ENSG00000251076": "gene without a symbol in the reference annotation",
    "SERPINB4": "serine protease inhibitor (SCCA2), induced in inflamed squamous epithelium",
    "SAA1": "serum amyloid A1, acute-phase protein induced by IL-6/IL-1",
    "SAA2": "serum amyloid A2, acute-phase protein (SAA1 paralog)",
    "CD38": "NAD+ glycohydrolase and immune-cell activation marker; interferon-inducible",
    "CAVIN2": "caveolae component (SDPR), membrane organisation",
    "SPINK5": "protease inhibitor LEKTI, maintains the epithelial barrier",
    "NR4A2": "orphan nuclear receptor, immediate-early gene that restrains inflammation",
    "CSTB": "cystatin B, inhibitor of cathepsin cysteine proteases",
    "ELOB": "elongin B, part of the elongin/VHL ubiquitin ligase and transcription elongation",
    "FTL": "ferritin light chain, iron storage; ferritin rises in inflammation",
    "CXCL10": "interferon-induced chemokine (IP-10) that recruits T and NK cells",
    "LINC01503": "long non-coding RNA, function not well defined",
    "EPM2A": "laforin, glycogen phosphatase",
    "PDZK1IP1": "MAP17, small membrane protein up in inflamed epithelium",
    "PROM1": "prominin-1 (CD133), membrane glycoprotein of cilia and progenitor cells",
    "SLC6A6": "taurine transporter",
    "SLC37A1": "endoplasmic-reticulum glycerol-3-phosphate/phosphate antiporter",
    "STK33": "serine/threonine kinase expressed in ciliated cells",
    "CWH43": "lipid remodelling of GPI-anchored proteins",
    "POU2AF3": "POU-domain transcriptional coactivator (COLCA2/OCA-T2)",
    "LINC00964": "long non-coding RNA, function unknown",
    "SLC39A9": "zinc transporter ZIP9",
    "WARS1": "tryptophanyl-tRNA synthetase, strongly interferon-gamma-inducible",
    "NDUFB2": "complex I subunit, oxidative phosphorylation",
    "UQCR11": "complex III subunit, oxidative phosphorylation",
    "UQCRQ": "complex III subunit, oxidative phosphorylation",
    "UCA1": "long non-coding RNA (urothelial cancer associated 1)",
    "ZNF473": "zinc-finger protein in histone mRNA 3' end processing",
    "CCL20": "epithelial chemokine attracting CCR6+ dendritic and Th17 cells",
    "TMSB4X": "thymosin beta-4, actin sequestering; abundant cytoskeletal gene",
    "EEF1A1": "translation elongation factor 1-alpha",
    "NPM1": "nucleophosmin, nucleolar chaperone in ribosome biogenesis",
    "FAM107B": "stress-responsive protein of unknown function",
    "ZNF200": "zinc-finger protein, function poorly characterised",
    "MORF4L1": "chromatin-remodelling (NuA4/Sin3B) complex component",
    "IRF8": "interferon regulatory factor 8, interferon responses and myeloid/DC development",
    "MAN1C1": "Golgi alpha-mannosidase, N-glycan processing",
    "RIOK1": "atypical kinase in 40S ribosome maturation",
    "SPRR1B": "small proline-rich protein of the squamous cornified envelope",
    "FASTKD1": "mitochondrial RNA-binding protein",
}


def is_excluded(symbol):
    return symbol.startswith(("RPL", "RPS", "MT-"))


def read_isg_genes():
    genes = set()
    with open(ISG_GMT) as handle:
        for line in handle:
            genes.update(line.rstrip("\n").split("\t")[2:])
    return genes


def mean_logcpm_difference():
    """Per cell type: mean logCPM in severe donors minus mild donors, per gene."""
    pb = ad.read_h5ad(common.PSEUDOBULK_H5AD)
    counts = pb.X.toarray().astype(float)
    logcpm = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e6)
    obs = pb.obs
    diffs = {}
    for ct in obs["ct"].astype(str).unique():
        severe = ((obs["ct"] == ct) & (obs["group"] == "severe")).to_numpy()
        mild = ((obs["ct"] == ct) & (obs["group"] == "mild")).to_numpy()
        diffs[ct] = pd.Series(logcpm[severe].mean(axis=0) - logcpm[mild].mean(axis=0), index=pb.var_names)
    return diffs


def candidates_for(ct, genes, diffs):
    de = pd.read_csv(DE_DIR / f"de_{ct}_severe_vs_mild.tsv", sep="\t")
    de = de[~de["gene_symbol"].apply(is_excluded)]
    de_sig = set(de.loc[de["padj"] < PADJ_CUTOFF, "gene_id"])

    imp = genes[(genes["ct"] == ct) & ~genes["gene_symbol"].apply(is_excluded)]
    best_rank = imp.groupby("gene_id")["rank"].min()
    best_stab = imp.groupby("gene_id")["stability"].max()
    model_hit = set(imp.loc[(imp["rank"] <= TOP_K) & (imp["stability"] >= MIN_STABILITY), "gene_id"])

    rows = []
    de_by_id = de.set_index("gene_id")
    for gene_id in sorted(de_sig | model_hit):
        evidence = "both" if (gene_id in de_sig and gene_id in model_hit) else ("de" if gene_id in de_sig else "model")
        if gene_id in de_by_id.index:
            d = de_by_id.loc[gene_id]
            symbol, log2fc, padj = d["gene_symbol"], d["log2fc"], d["padj"]
        else:
            # selected by the model but filtered out of the DE test (low counts)
            symbol = imp.loc[imp["gene_id"] == gene_id, "gene_symbol"].iloc[0]
            log2fc, padj = np.nan, np.nan
        sign = log2fc if not np.isnan(log2fc) else diffs[ct][gene_id]
        rows.append({"gene_id": gene_id, "gene_symbol": symbol, "ct": ct,
                     "direction": "up_in_severe" if sign > 0 else "down_in_severe",
                     "log2fc": log2fc, "padj": padj,
                     "importance_rank": best_rank.get(gene_id, np.nan),
                     "stability": best_stab.get(gene_id, np.nan), "evidence": evidence})
    return pd.DataFrame(rows)


def plot_shortlist(shortlist, out_path):
    table = shortlist.iloc[::-1]
    labels = [f"{r.gene_symbol} ({r.ct.replace('_cells', '')}, {r.evidence})" for r in table.itertuples()]
    colors = ["#eb6834" if d == "up_in_severe" else "#2a78d6" for d in table["direction"]]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.22 * len(table))))
    ax.barh(range(len(table)), table["log2fc"].fillna(0), color=colors)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("log2 fold change, severe / mild (pseudobulk DESeq2)")
    ax.set_title(f"Key-gene shortlist ({len(table)} genes; orange = up in severe, blue = down)\n"
                 "n = 21 severe / 14 mild patients overall; per-cell-type n in key_gene_shortlist.tsv")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    genes = pd.read_csv(ATTR_DIR / "gene_importance.tsv", sep="\t")
    cell_types = [ct for ct in genes["ct"].unique()]
    isg = read_isg_genes()

    diffs = mean_logcpm_difference()
    parts = [candidates_for(ct, genes, diffs) for ct in cell_types]
    candidates = pd.concat(parts, ignore_index=True)
    candidates = candidates[(candidates["ct"] != "all") | (candidates["evidence"] == "both")]
    candidates["in_isg_set"] = candidates["gene_symbol"].isin(isg)
    candidates["order"] = candidates["evidence"].map(EVIDENCE_ORDER)
    candidates = candidates.sort_values(["order", "padj", "importance_rank"], na_position="last")
    candidates = candidates.drop(columns="order").reset_index(drop=True)
    candidates.to_csv(DE_DIR / "key_gene_candidates.tsv", sep="\t", index=False, float_format="%.6g")

    picks = [candidates[candidates["evidence"] == "both"]]
    for ct in candidates["ct"].unique():
        de_only = candidates[(candidates["ct"] == ct) & (candidates["evidence"] == "de")]
        picks.append(de_only.sort_values("padj").head(PER_CT_EACH))
        model_only = candidates[(candidates["ct"] == ct) & (candidates["evidence"] == "model")]
        picks.append(model_only.sort_values(["importance_rank", "stability"], ascending=[True, False]).head(PER_CT_EACH))
    shortlist = pd.concat(picks).head(MAX_ROWS).copy()
    shortlist["role_summary"] = shortlist["gene_symbol"].map(ROLES).fillna("")
    columns = ["gene_id", "gene_symbol", "ct", "direction", "log2fc", "padj", "importance_rank",
               "evidence", "in_isg_set", "role_summary", "stability"]
    shortlist = shortlist[columns]
    shortlist.to_csv(DE_DIR / "key_gene_shortlist.tsv", sep="\t", index=False, float_format="%.6g")

    (DE_DIR / "figures").mkdir(parents=True, exist_ok=True)
    plot_shortlist(shortlist, DE_DIR / "figures" / "10_key_genes.png")

    print(f"Candidates: {len(candidates)} gene x cell type rows "
          f"({candidates['evidence'].value_counts().to_dict()})")
    print(f"Shortlist: {len(shortlist)} rows, evidence {shortlist['evidence'].value_counts().to_dict()}, "
          f"{int(shortlist['in_isg_set'].sum())} in the ISG sets")
    print(shortlist[["gene_symbol", "ct", "direction", "log2fc", "padj", "importance_rank",
                     "evidence", "in_isg_set"]].to_string(index=False))
    missing = shortlist.loc[shortlist["role_summary"] == "", "gene_symbol"].tolist()
    if missing:
        print(f"\nNo role_summary yet for: {missing}")
    print(f"\nWrote {DE_DIR / 'key_gene_shortlist.tsv'}, key_gene_candidates.tsv, figures/10_key_genes.png")


if __name__ == "__main__":
    main()
