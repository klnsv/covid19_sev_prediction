"""Step 08: pathway enrichment of the pseudobulk DE results.

Reads:  results/de/de_<ct>_<contrast>.tsv (step 07)
        data/external/gene_sets/h.all / c5.go.bp / c2.cp.reactome v2024.1.Hs GMTs (MSigDB)
Writes: results/de/enrichment_gsea.tsv  preranked GSEA: every Hallmark row, GO/Reactome rows with FDR < 0.25
        results/de/enrichment_ora.tsv   over-representation (Fisher): rows with qvalue < 0.25
        results/de/figures/08_hallmark_gsea.png

Two methods, for every cell type x contrast tested in step 07:
- GSEA: genes ranked by the DESeq2 Wald statistic (stat > 0 = higher in the first
  group), gseapy.prerank, 1,000 gene-set permutations, set size 15-500, seed 42.
  NES > 0 = the set is shifted towards the first group (e.g. up in severe).
- ORA: significant genes (padj < 0.05), up and down separately, vs the background
  of all genes tested in that DE (expressed genes); one-sided Fisher's exact test,
  BH within each test x direction x library. Run only when >= 5 genes are significant.

Run from the repo root: python scripts/08_enrichment.py
"""
import numpy as np
import pandas as pd
import gseapy
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common

GENE_SET_DIR = common.EXTERNAL / "gene_sets"
LIBRARIES = {
    "hallmark": GENE_SET_DIR / "h.all.v2024.1.Hs.symbols.gmt",
    "go_bp": GENE_SET_DIR / "c5.go.bp.v2024.1.Hs.symbols.gmt",
    "reactome": GENE_SET_DIR / "c2.cp.reactome.v2024.1.Hs.symbols.gmt",
}
MIN_SIZE, MAX_SIZE = 15, 500
N_PERM = 1000
N_THREADS = 10
PADJ_CUTOFF = 0.05
MIN_SIG_FOR_ORA = 5
KEEP_FDR = 0.25          # GO/Reactome GSEA rows and ORA rows kept below this (file size)
DE_DIR = common.RESULTS / "de"
FIG_DIR = DE_DIR / "figures"


def read_gmt(path):
    sets = {}
    with open(path) as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            sets[fields[0]] = set(fields[2:])
    return sets


def ranking(de):
    """Wald statistic per gene symbol; duplicate symbols keep the larger |stat|."""
    table = de.dropna(subset=["stat"])
    table = table[~table["gene_symbol"].str.startswith("ENSG")]
    table = table.reindex(table["stat"].abs().sort_values(ascending=False).index)
    table = table.drop_duplicates("gene_symbol")
    return table.set_index("gene_symbol")["stat"].sort_values(ascending=False)


def run_gsea(rank, library_name, sets):
    result = gseapy.prerank(rnk=rank, gene_sets={k: list(v) for k, v in sets.items()},
                            min_size=MIN_SIZE, max_size=MAX_SIZE, permutation_num=N_PERM,
                            threads=N_THREADS, seed=common.SEED, outdir=None, no_plot=True,
                            verbose=False)
    res = result.res2d.rename(columns={"Term": "term", "ES": "es", "NES": "nes", "NOM p-val": "pvalue",
                                       "FDR q-val": "fdr", "Lead_genes": "leading_edge"})
    res["library"] = library_name
    res["set_size"] = res["Tag %"].astype(str).str.split("/").str[1].astype(int)
    for col in ["es", "nes", "pvalue", "fdr"]:
        res[col] = res[col].astype(float)
    return res[["library", "term", "set_size", "es", "nes", "pvalue", "fdr", "leading_edge"]]


def run_ora(sig_genes, background, library_name, sets):
    rows = []
    n_sig, n_bg = len(sig_genes), len(background)
    for term, genes in sets.items():
        in_bg = genes & background
        if not (MIN_SIZE <= len(in_bg) <= MAX_SIZE):
            continue
        overlap = sig_genes & in_bg
        a = len(overlap)
        table = [[a, n_sig - a], [len(in_bg) - a, n_bg - n_sig - len(in_bg) + a]]
        pvalue = fisher_exact(table, alternative="greater")[1]
        rows.append({"library": library_name, "term": term, "set_size": len(in_bg), "n_overlap": a,
                     "n_sig": n_sig, "expected_overlap": round(len(in_bg) * n_sig / n_bg, 3),
                     "pvalue": pvalue, "genes": ";".join(sorted(overlap))})
    table = pd.DataFrame(rows)
    table["qvalue_bh"] = multipletests(table["pvalue"], method="fdr_bh")[1]
    return table


def plot_hallmark(gsea, out_path):
    hall = gsea[gsea["library"] == "hallmark"].copy()
    hall["column"] = hall["contrast"] + "\n" + hall["ct"]
    columns = list(dict.fromkeys(hall.sort_values(["contrast", "ct"])["column"]))
    nes = hall.pivot(index="term", columns="column", values="nes")[columns]
    fdr = hall.pivot(index="term", columns="column", values="fdr")[columns]
    keep = (fdr < KEEP_FDR).any(axis=1)
    nes, fdr = nes[keep], fdr[keep]
    nes = nes.loc[nes.mean(axis=1).sort_values().index]
    fdr = fdr.loc[nes.index]

    fig, ax = plt.subplots(figsize=(0.55 * len(columns) + 4, 0.3 * len(nes) + 2.5))
    limit = np.nanmax(np.abs(nes.to_numpy()))
    image = ax.imshow(nes.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(nes.shape[0]):
        for j in range(nes.shape[1]):
            q = fdr.iat[i, j]
            if q < 0.05:
                ax.text(j, i, "*", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([c.replace("_cells", "") for c in columns], rotation=90, fontsize=6)
    ax.set_yticks(range(len(nes)))
    ax.set_yticklabels([t.replace("HALLMARK_", "") for t in nes.index], fontsize=7)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.5)
    colorbar.set_label("NES (> 0 = up in first group of the contrast)")
    ax.set_title("Hallmark GSEA on pseudobulk DE (Wald statistic ranking)\n"
                 "* = FDR < 0.05; rows shown if FDR < 0.25 anywhere; donor n per test in de_summary.tsv",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    libraries = {name: read_gmt(path) for name, path in LIBRARIES.items()}
    summary = pd.read_csv(DE_DIR / "de_summary.tsv", sep="\t")
    tested = summary[summary["status"] == "tested"]

    gsea_parts, ora_parts = [], []
    for _, test in tested.iterrows():
        ct, contrast = test["ct"], test["contrast"]
        de = pd.read_csv(DE_DIR / f"de_{ct}_{contrast}.tsv", sep="\t")
        rank = ranking(de)
        background = set(rank.index)
        for library_name, sets in libraries.items():
            res = run_gsea(rank, library_name, sets)
            res.insert(0, "contrast", contrast)
            res.insert(0, "ct", ct)
            gsea_parts.append(res)

            sig = de[(de["padj"] < PADJ_CUTOFF) & de["gene_symbol"].isin(background)]
            for direction, genes in [("up", sig[sig["log2fc"] > 0]), ("down", sig[sig["log2fc"] < 0])]:
                if len(genes) < MIN_SIG_FOR_ORA:
                    continue
                ora = run_ora(set(genes["gene_symbol"]), background, library_name, sets)
                ora.insert(0, "direction", direction)
                ora.insert(0, "contrast", contrast)
                ora.insert(0, "ct", ct)
                ora_parts.append(ora)
        hall = gsea_parts[-3]
        top = hall.sort_values("fdr").head(3)
        top_text = ", ".join(f"{t.replace('HALLMARK_', '')} (NES {n:+.2f}, FDR {q:.3f})"
                             for t, n, q in zip(top["term"], top["nes"], top["fdr"]))
        print(f"  {contrast:22s} {ct:28s} {len(rank):6d} genes ranked | top Hallmark: {top_text}")

    gsea = pd.concat(gsea_parts, ignore_index=True)
    gsea_out = gsea[(gsea["library"] == "hallmark") | (gsea["fdr"] < KEEP_FDR)]
    gsea_out.to_csv(DE_DIR / "enrichment_gsea.tsv", sep="\t", index=False, float_format="%.6g")
    ora = pd.concat(ora_parts, ignore_index=True)
    ora_out = ora[ora["qvalue_bh"] < KEEP_FDR]
    ora_out.to_csv(DE_DIR / "enrichment_ora.tsv", sep="\t", index=False, float_format="%.6g")
    print(f"\nWrote enrichment_gsea.tsv ({len(gsea_out):,} rows) and enrichment_ora.tsv ({len(ora_out):,} rows)")

    n_sig = gsea[gsea["fdr"] < 0.05].groupby(["contrast", "ct", "library"]).size().unstack(fill_value=0)
    print("\nGSEA sets with FDR < 0.05:")
    print(n_sig.to_string())

    plot_hallmark(gsea, FIG_DIR / "08_hallmark_gsea.png")
    print(f"Wrote {FIG_DIR / '08_hallmark_gsea.png'}")


if __name__ == "__main__":
    main()
