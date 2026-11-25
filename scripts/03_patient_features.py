"""Step 03: patient-level tables.

Reads:  data/processed/nasal_qc.h5ad
Writes: data/processed/patient_meta.tsv
        data/processed/features_composition.tsv
        data/processed/pseudobulk_coarse.h5ad

Run from the repo root: python scripts/03_patient_features.py
"""
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

import common

CLR_PSEUDOCOUNT = 0.5
N_DONORS = 58
N_COMPOSITION_TYPES = 17


def slug(cell_type):
    return cell_type.lower().replace(" ", "_")


def build_patient_meta(obs):
    donor_columns = ["group", "severe", "who_peak", "sex", "age_decade", "bloody_swab", "pcr_status"]
    obs = obs.assign(pcr_status=obs["SARSCoV2_PCR_Status"].astype(str))

    per_donor = obs.groupby("donor_id", observed=True)[donor_columns]
    assert (per_donor.nunique(dropna=False) == 1).all(axis=None), "donor metadata varies across cells"

    meta = per_donor.first()
    meta["n_cells"] = obs.groupby("donor_id", observed=True).size()
    return meta.sort_index().reset_index()


def build_composition(obs, donors):
    kept = obs[~obs["ct_coarse"].isin(common.EXCLUDED_CELL_TYPES)]
    counts = pd.crosstab(kept["donor_id"], kept["ct_coarse"]).reindex(donors, fill_value=0)
    counts.columns = [slug(ct) for ct in counts.columns]
    assert counts.shape[1] == N_COMPOSITION_TYPES, f"expected {N_COMPOSITION_TYPES} types, got {counts.shape[1]}"

    n_cells = counts.sum(axis=1)
    proportions = counts.div(n_cells, axis=0)

    # pseudocount first: most donors have zero cells of some rare types
    log_counts = np.log(counts + CLR_PSEUDOCOUNT)
    clr = log_counts.sub(log_counts.mean(axis=1), axis=0)

    table = pd.concat([proportions.add_prefix("prop__"), clr.add_prefix("clr__")], axis=1)
    table.insert(0, "n_cells_composition", n_cells)
    table.index.name = "donor_id"
    return table.reset_index()


def sum_counts(counts, cell_mask):
    return sparse.csr_matrix(counts[cell_mask].sum(axis=0), dtype=np.int64)


def build_pseudobulk(adata, meta):
    host = ~adata.var["is_viral"].to_numpy()
    counts = adata.layers["counts"][:, host].tocsr()
    donor_of_cell = adata.obs["donor_id"].to_numpy()
    ct_of_cell = adata.obs["ct_coarse"].to_numpy()
    in_all = ~np.isin(ct_of_cell, common.EXCLUDED_CELL_TYPES)

    rows = []
    row_info = []
    for donor in meta["donor_id"]:
        is_donor = donor_of_cell == donor

        # no min_cells here: the classifier needs an "all" row for every donor
        all_mask = is_donor & in_all
        rows.append(sum_counts(counts, all_mask))
        row_info.append({"donor_id": donor, "ct": "all", "n_cells": int(all_mask.sum())})

        for cell_type in sorted(set(ct_of_cell[is_donor])):
            ct_mask = is_donor & (ct_of_cell == cell_type)
            if ct_mask.sum() >= common.MIN_CELLS_PSEUDOBULK:
                rows.append(sum_counts(counts, ct_mask))
                row_info.append({"donor_id": donor, "ct": slug(cell_type), "n_cells": int(ct_mask.sum())})

    pb_obs = pd.DataFrame(row_info).merge(
        meta[["donor_id", "group", "severe", "who_peak", "sex"]], on="donor_id", how="left"
    )
    pb_obs.index = pb_obs["donor_id"] + "__" + pb_obs["ct"]

    pseudobulk = ad.AnnData(
        X=sparse.vstack(rows, format="csr"),
        obs=pb_obs,
        var=adata.var.loc[host, ["gene_symbol"]].copy(),
    )
    pseudobulk.uns["min_cells"] = common.MIN_CELLS_PSEUDOBULK
    return pseudobulk


def main():
    print(f"Reading {common.QC_H5AD}")
    adata = sc.read_h5ad(common.QC_H5AD)
    adata.obs["donor_id"] = adata.obs["donor_id"].astype(str)
    adata.obs["ct_coarse"] = adata.obs["ct_coarse"].astype(str)

    meta = build_patient_meta(adata.obs)
    assert len(meta) == N_DONORS, f"expected {N_DONORS} donors, got {len(meta)}"
    meta.to_csv(common.PATIENT_META, sep="\t", index=False)
    print(f"Wrote {common.PATIENT_META}")

    composition = build_composition(adata.obs, meta["donor_id"])
    composition.to_csv(common.FEATURES_COMPOSITION, sep="\t", index=False)
    print(f"Wrote {common.FEATURES_COMPOSITION}")

    pseudobulk = build_pseudobulk(adata, meta)
    assert (pseudobulk.obs["ct"] == "all").sum() == N_DONORS, "missing 'all' rows"
    pseudobulk.write_h5ad(common.PSEUDOBULK_H5AD)
    print(f"Wrote {common.PSEUDOBULK_H5AD}")

    print("\nSummary")
    print(meta["group"].value_counts().to_string())
    print(f"composition: {composition.shape[0]} donors x {composition.shape[1]} columns")
    print(f"pseudobulk: {pseudobulk.n_obs} samples x {pseudobulk.n_vars} host genes")
    print(pseudobulk.obs["ct"].value_counts().to_string())


if __name__ == "__main__":
    main()
