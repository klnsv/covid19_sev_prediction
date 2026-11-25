"""Shared constants: paths, seed and label mapping.

Every script imports this so paths and labels are defined in one place.
This is a shared file: change it only in its own small PR (see CONVENTIONS.md).
Use from a script in scripts/:      import common
Use from a notebook in notebooks/:  import sys; sys.path.append("../scripts"); import common
"""
from pathlib import Path

# repo root = parent of scripts/, so paths work from any working directory
ROOT = Path(__file__).resolve().parent.parent

RAW_H5AD = ROOT / "data" / "raw" / "zeigler2021_nasal.h5ad"
RAW_H5AD_MD5 = "2fcf7a8a3a52ec91e824f2041679195f"

PROCESSED = ROOT / "data" / "processed"
EXTERNAL = ROOT / "data" / "external"
RESULTS = ROOT / "results"

# data contracts (see CONTEXT.md section 6)
QC_H5AD = PROCESSED / "nasal_qc.h5ad"                      # C1
PATIENT_META = PROCESSED / "patient_meta.tsv"              # C2
FEATURES_COMPOSITION = PROCESSED / "features_composition.tsv"  # C3
PSEUDOBULK_H5AD = PROCESSED / "pseudobulk_coarse.h5ad"     # C4

SEED = 42

# author column Cohort_Disease_WHO_Score -> our `group` label (CONTEXT.md section 4)
GROUP_MAP = {
    "Control_WHO_0": "control",                 # healthy, 15 donors
    "COVID19_WHO_1-5": "mild",                  # mild/moderate COVID-19, 14 donors
    "COVID19_WHO_6-8": "severe",                # severe COVID-19, 21 donors
    "Control_WHO_8": "control_icu",             # non-COVID respiratory failure, 6 donors
    "ConvalescentCOVID19_WHO_0": "convalescent",  # 2 donors, excluded from analyses
}

# minimum cells for a donor x cell-type pseudobulk sample
MIN_CELLS_PSEUDOBULK = 10

# blood contamination, left out of composition features
EXCLUDED_CELL_TYPES = ["Erythroblasts"]
