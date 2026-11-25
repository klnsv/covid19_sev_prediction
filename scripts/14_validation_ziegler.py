
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import mannwhitneyu, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common

OUT_DIR = common.RESULTS / "validation"
FIG_DIR = OUT_DIR / "figures"
DE_DIR = common.RESULTS / "de"
IMMUNE = ["T Cells", "B Cells", "Macrophages", "Dendritic Cells", "Plasmacytoid DCs", "Mast Cells"]
EXCLUDED = IMMUNE + common.EXCLUDED_CELL_TYPES


def verdict(direction_ok, p):
    if direction_ok is None:
        return "descriptive"
    if not direction_ok:
        return "contradicted" if p < 0.05 else "not supported (opposite, n.s.)"
    return "supported" if p < 0.05 else "consistent, not significant"


def row(claim, check, cell_type, n, result, p, direction_ok, source, q=np.nan):
    test_p = q if not np.isnan(q) else p
    return {"claim": claim, "check": check, "cell_type": cell_type, "n": n, "our_result": result,
            "pvalue": p, "qvalue": q, "verdict": verdict(direction_ok, test_p), "source": source}


def donor_fractions(cells, donors, numerator_mask, denominator_mask):
    num = cells[numerator_mask].groupby("donor_id").size().reindex(donors, fill_value=0)
    den = cells[denominator_mask].groupby("donor_id").size().reindex(donors, fill_value=0)
    return (num / den.replace(0, np.nan))


def compare(values, groups, a, b):
    """Mann-Whitney U of group a vs b; returns medians, n and p."""
    va, vb = values[groups == a].dropna(), values[groups == b].dropna()
    p = mannwhitneyu(va, vb).pvalue
    return va.median(), vb.median(), len(va), len(vb), p


def de_gene(ct, contrast, gene):
    table = pd.read_csv(DE_DIR / f"de_{ct}_{contrast}.tsv", sep="\t")
    hit = table[table["gene_symbol"] == gene]
    return None if hit.empty else hit.iloc[0]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(common.PATIENT_META, sep="\t").set_index("donor_id")
    donors = meta.index.to_numpy()
    groups = meta["group"]
    adata = ad.read_h5ad(common.QC_H5AD)
    obs = adata.obs
    cells = pd.DataFrame({"donor_id": obs["donor_id"].astype(str).to_numpy(),
                          "ct": obs["ct_coarse"].astype(str).to_numpy(),
                          "ct_detailed": obs["ct_detailed"].astype(str).to_numpy(),
                          "viral_umis": obs["viral_umis"].astype(float).to_numpy(),
                          "n_counts": obs["n_counts"].astype(float).to_numpy(),
                          "rna_status": obs["SingleCell_SARSCoV2_RNA_Status"].astype(str).to_numpy()})
    epithelial = ~cells["ct"].isin(EXCLUDED)
    comp_tests = pd.read_csv(common.RESULTS / "composition" / "composition_tests.tsv", sep="\t")
    gsea = pd.read_csv(DE_DIR / "enrichment_gsea.tsv", sep="\t")
    rows = []


    for ct, expected in [("ciliated_cells", "higher_in_mild"), ("secretory_cells", "higher_in_severe")]:
        t = comp_tests[(comp_tests["ct"] == ct) & (comp_tests["comparison"] == "mild_vs_severe")].iloc[0]
        rows.append(row(1, f"{ct} proportion, mild vs severe (C6)", ct, f"{t['n_mild']} mild / {t['n_severe']} severe",
                        f"median {t['median_mild']:.3f} mild vs {t['median_severe']:.3f} severe ({t['direction']})",
                        t["pvalue"], t["direction"] == expected, "composition_tests.tsv", q=t["qvalue_bh"]))


    for ct in ["deuterosomal_cells", "developing_ciliated_cells"]:
        t = comp_tests[(comp_tests["ct"] == ct) & (comp_tests["comparison"] == "control_vs_covid")].iloc[0]
        rows.append(row(2, f"{ct} proportion, control vs COVID (C6)", ct,
                        f"{t['n_control']} control / {t['n_mild'] + t['n_severe']} COVID",
                        f"median control {t['median_control']:.3f}, mild {t['median_mild']:.3f}, severe "
                        f"{t['median_severe']:.3f} ({t['direction']})",
                        t["pvalue"], t["direction"] == "higher_in_covid", "composition_tests.tsv", q=t["qvalue_bh"]))


    epi = cells[epithelial]
    props = pd.crosstab(epi["donor_id"], epi["ct"]).reindex(donors, fill_value=0)
    props = props.div(props.sum(axis=1).replace(0, np.nan), axis=0)
    simpson = 1 - (props ** 2).sum(axis=1)
    covid_label = groups.replace({"mild": "covid", "severe": "covid"})
    m_c, m_v, n_c, n_v, p = compare(simpson, covid_label, "control", "covid")
    rows.append(row(3, "Gini-Simpson diversity of epithelial types, control vs COVID", "epithelial",
                    f"{n_c} control / {n_v} COVID", f"median {m_c:.3f} control vs {m_v:.3f} COVID", p, m_v > m_c,
                    "C1 (computed here)"))


    symbols = adata.var["gene_symbol"].astype(str)
    for gene in ["ACE2", "TMPRSS2", "CTSL", "FURIN"]:
        col = np.where(symbols.to_numpy() == gene)[0]
        if len(col) == 0:
            continue
        values = np.asarray(adata.X[:, col[0]].todense()).ravel()
        means = pd.Series(values).groupby(cells["ct"]).mean().drop(common.EXCLUDED_CELL_TYPES, errors="ignore")
        top = means.sort_values(ascending=False).head(3)
        text = ", ".join(f"{k.replace(' Cells', '')} {v:.3f}" for k, v in top.items())
        expected = None
        rows.append(row(4, f"{gene}: highest mean expression by coarse type", "all", f"{len(cells):,} cells",
                        text, np.nan, expected, "C1 (computed here)"))


    frac = donor_fractions(cells, donors, cells["ct_detailed"] == "Interferon Responsive Ciliated Cells", epithelial)
    for a, b in [("mild", "control"), ("mild", "severe")]:
        m_a, m_b, n_a, n_b, p = compare(frac, groups, a, b)
        rows.append(row(5, f"IFN-responsive ciliated cells, % of epithelial, {a} vs {b}", "ciliated (detailed)",
                        f"{n_a} {a} / {n_b} {b}", f"median {100 * m_a:.1f}% {a} vs {100 * m_b:.1f}% {b}",
                        p, m_a > m_b, "C1 (computed here)"))


    for contrast, expected_sign in [("severe_vs_mild", -1), ("covid_vs_control", 1)]:
        for ct in ["ciliated_cells", "all"]:
            t = gsea[(gsea["library"] == "hallmark") & (gsea["contrast"] == contrast) & (gsea["ct"] == ct)
                     & (gsea["term"] == "HALLMARK_INTERFERON_ALPHA_RESPONSE")]
            if t.empty:
                continue
            t = t.iloc[0]
            rows.append(row(6, f"GSEA Hallmark IFN-alpha response, {contrast}", ct, "see de_summary.tsv",
                            f"NES {t['nes']:+.2f}", t["pvalue"], np.sign(t["nes"]) == expected_sign,
                            "enrichment_gsea.tsv", q=t["fdr"]))
    for gene in ["IFI27", "IFIT1", "IFI6", "IFITM3", "GBP3", "STAT1", "MX1", "HLA-A", "HLA-C", "HLA-DRA"]:
        d = de_gene("ciliated_cells", "severe_vs_mild", gene)
        if d is None:
            continue
        rows.append(row(6, f"{gene} pseudobulk DE severe vs mild", "ciliated_cells",
                        f"{int(d['n_donors_a'])} severe / {int(d['n_donors_b'])} mild",
                        f"log2FC {d['log2fc']:+.2f}", d["pvalue"], d["log2fc"] < 0, "de_ciliated_cells_severe_vs_mild.tsv",
                        q=d["padj"]))
    isg = pd.read_csv(common.RESULTS / "isg" / "isg_tests.tsv", sep="\t")
    for _, t in isg[(isg["comparison"] == "mild_vs_severe") & isg["ct"].isin(["epithelial", "ciliated_cells"])].iterrows():
        rows.append(row(6, f"step 06 ISG score ({t['gene_set']}), mild vs severe", t["ct"], "see isg_tests.tsv",
                        f"Mann-Whitney U {t['statistic']:.0f}", t["pvalue"], True if t["statistic"] > 0 else None,
                        "isg_tests.tsv"))

    scores = pd.read_csv(common.RESULTS / "isg" / "donor_isg_scores.tsv", sep="\t")
    for ct in ["epithelial", "ciliated_cells"]:
        for gs in ["ifn_alpha", "ifn_gamma"]:
            s = scores[(scores["ct"] == ct) & (scores["gene_set"] == gs)].set_index("donor_id")["score"]
            m_m, m_s, n_m, n_s, p = compare(s, groups.reindex(s.index), "mild", "severe")
            for r in rows:
                if r["check"] == f"step 06 ISG score ({gs}), mild vs severe" and r["cell_type"] == ct:
                    r["our_result"] = f"median score {m_m:.3f} mild vs {m_s:.3f} severe"
                    r["n"] = f"{n_m} mild / {n_s} severe"
                    r["verdict"] = verdict(m_m > m_s, r["pvalue"])


    t = gsea[(gsea["library"] == "hallmark") & (gsea["contrast"] == "severe_vs_mild") & (gsea["ct"] == "goblet_cells")
             & (gsea["term"] == "HALLMARK_INTERFERON_ALPHA_RESPONSE")].iloc[0]
    rows.append(row(7, "GSEA Hallmark IFN-alpha response, severe vs mild (all goblet, not only MUC5AC-high)",
                    "goblet_cells", "12 severe / 9 mild", f"NES {t['nes']:+.2f}", t["pvalue"], t["nes"] < 0,
                    "enrichment_gsea.tsv", q=t["fdr"]))


    for gene in ["IFNAR1", "IFNAR2", "IFNGR1", "IFNGR2"]:
        d = de_gene("ciliated_cells", "severe_vs_mild", gene)
        if d is None:
            continue
        rows.append(row(8, f"{gene} pseudobulk DE severe vs mild (expect no significant loss)", "ciliated_cells",
                        f"{int(d['n_donors_a'])} severe / {int(d['n_donors_b'])} mild",
                        f"base mean {d['base_mean']:.0f}, log2FC {d['log2fc']:+.2f}, padj {d['padj']:.2f}",
                        d["pvalue"], None, "de_ciliated_cells_severe_vs_mild.tsv", q=d["padj"]))
        rows[-1]["verdict"] = "supported" if not (d["padj"] < 0.05 and d["log2fc"] < 0) else "not supported"


    for gene in ["S100A8", "S100A9", "HSPA8", "HSPA1A", "DUSP1"]:
        for ct in ["all", "ciliated_cells", "secretory_cells", "squamous_cells"]:
            d = de_gene(ct, "severe_vs_mild", gene)
            if d is None or np.isnan(d["pvalue"]):
                continue
            rows.append(row(9, f"{gene} pseudobulk DE severe vs mild", ct,
                            f"{int(d['n_donors_a'])} severe / {int(d['n_donors_b'])} mild",
                            f"log2FC {d['log2fc']:+.2f}", d["pvalue"], d["log2fc"] > 0, f"de_{ct}_severe_vs_mild.tsv",
                            q=d["padj"]))


    rows.append({"claim": 10, "check": "steroid effect on ISGs", "cell_type": "", "n": "", "pvalue": np.nan,
                 "qvalue": np.nan, "our_result": "no steroid field in the data", "verdict": "not testable",
                 "source": "CONTEXT.md section 2"})


    load = cells.groupby("donor_id")[["viral_umis", "n_counts"]].sum().reindex(donors)
    load["viral_per_million"] = load["viral_umis"] / load["n_counts"] * 1e6
    load["group"] = groups
    load.to_csv(OUT_DIR / "viral_load.tsv", sep="\t", float_format="%.4g")
    m_m, m_s, n_m, n_s, p = compare(load["viral_per_million"], groups, "mild", "severe")
    geo = {g: np.exp(np.log(load.loc[groups == g, "viral_per_million"] + 1).mean()) - 1 for g in ["mild", "severe"]}
    rows.append(row(11, "viral UMIs per million, mild vs severe", "whole sample", f"{n_m} mild / {n_s} severe",
                    f"median {m_m:.1f} mild vs {m_s:.1f} severe; geometric mean(+1) {geo['mild']:.1f} vs {geo['severe']:.1f}",
                    p, m_s > m_m, "C1 (computed here)"))


    covid = groups.isin(["mild", "severe"])
    log_load = np.log10(load.loc[covid, "viral_per_million"] + 1)
    for label, mask, expected_sign in [
            ("secretory cells", cells["ct"] == "Secretory Cells", 1),
            ("FOXJ1 high ciliated cells", cells["ct_detailed"] == "FOXJ1 high Ciliated Cells", -1)]:
        frac = donor_fractions(cells, donors, mask, epithelial)[covid.to_numpy()]
        rho, p = spearmanr(frac, log_load, nan_policy="omit")
        rows.append(row(12, f"{label} % of epithelial vs log10 viral load (Spearman)", label,
                        f"{int(frac.notna().sum())} COVID donors", f"rho {rho:+.2f}", p, np.sign(rho) == expected_sign,
                        "C1 (computed here)"))


    pos = cells[cells["rna_status"] == "pos"]
    by_group = pos.groupby(pos["donor_id"].map(groups)).size()
    rows.append(row(13, "SARS-CoV-2 RNA+ cells by group", "all", f"{len(pos)} RNA+ cells",
                    ", ".join(f"{g} {n}" for g, n in by_group.items()), np.nan, None, "C1 (computed here)"))
    rows[-1]["verdict"] = "supported" if len(pos) == 415 else "not supported"


    top = pos["ct_detailed"].value_counts().head(5)
    rows.append(row(14, "most frequent detailed types among RNA+ cells", "all", f"{len(pos)} RNA+ cells",
                    ", ".join(f"{k} {v}" for k, v in top.items()), np.nan, None, "C1 (computed here)"))


    genes15 = ["IFITM3", "IFITM1", "IFI44L", "MX1", "IRF1", "OAS1", "OAS2", "EIF2AK2"]
    cols = [int(np.where(symbols.to_numpy() == g)[0][0]) for g in genes15 if (symbols == g).any()]
    expr = np.asarray(adata.X[:, cols].todense())
    score = expr.mean(axis=1)
    cells["isg15_score"] = score
    diffs = []
    for (d, ct), sub in cells[cells["rna_status"].isin(["pos", "neg"])].groupby(["donor_id", "ct"]):
        n_pos = (sub["rna_status"] == "pos").sum()
        if n_pos >= 5 and (sub["rna_status"] == "neg").sum() >= 5:
            diffs.append({"donor_id": d, "ct": ct, "group": groups[d],
                          "diff": sub.loc[sub["rna_status"] == "pos", "isg15_score"].mean()
                          - sub.loc[sub["rna_status"] == "neg", "isg15_score"].mean()})
    diffs = pd.DataFrame(diffs)
    if len(diffs):
        text = (f"{(diffs['diff'] > 0).sum()} of {len(diffs)} donor x cell-type pairs higher in RNA+ cells; "
                f"median difference {diffs['diff'].median():+.3f} (mean log-norm of {len(cols)} antiviral genes)")
        rows.append(row(15, "antiviral genes, RNA+ vs bystander cells (exploratory, no p-value)", "pairs with >= 5 RNA+",
                        f"{len(diffs)} donor x cell-type pairs", text, np.nan, None, "C1 (computed here)"))


    t = comp_tests[(comp_tests["ct"] == "macrophages") & (comp_tests["comparison"] == "mild_vs_severe")].iloc[0]
    rows.append({"claim": 16, "check": "macrophage proportion, mild vs severe (C6)", "cell_type": "macrophages",
                 "n": f"{t['n_mild']} mild / {t['n_severe']} severe", "pvalue": t["pvalue"], "qvalue": t["qvalue_bh"],
                 "our_result": f"median {t['median_mild']:.4f} mild vs {t['median_severe']:.4f} severe ({t['direction']}); "
                               "only 1 mild / 6 severe donors have >= 10 macrophages",
                 "verdict": verdict(t["direction"] == "higher_in_severe", t["qvalue_bh"]) + " (proportion only)",
                 "source": "composition_tests.tsv"})

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "ziegler_checks.tsv", sep="\t", index=False, float_format="%.4g")
    print(table[["claim", "check", "cell_type", "n", "our_result", "pvalue", "qvalue", "verdict"]].to_string(
        index=False, max_colwidth=70))
    print("\nVerdicts:", table["verdict"].value_counts().to_dict())

    order = ["supported", "consistent, not significant", "not supported (opposite, n.s.)", "contradicted",
             "descriptive", "not testable"]
    colors = {"supported": "#1baf7a", "consistent, not significant": "#eda100",
              "not supported (opposite, n.s.)": "#e87ba4", "contradicted": "#eb6834",
              "descriptive": "#b0b0b0", "not testable": "#e6e6e6"}
    tested = table.copy()
    tested["v"] = tested["verdict"].str.replace(" (proportion only)", "", regex=False)
    counts = tested.groupby(["claim", "v"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    left = np.zeros(len(counts))
    for v in order:
        ax.barh(range(len(counts)), counts[v], left=left, color=colors[v], label=v, edgecolor="white")
        left += counts[v].to_numpy()
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels([f"claim {c}" for c in counts.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("number of checks")
    ax.set_title("Ziegler et al. 2021 claims checked in our data (donor-level; n per check in ziegler_checks.tsv)",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "14_claims.png", dpi=300)
    plt.close(fig)
    print(f"\nWrote {OUT_DIR / 'ziegler_checks.tsv'}, viral_load.tsv and figures/14_claims.png")


if __name__ == "__main__":
    main()
