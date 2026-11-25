# Session log: 2026-11-21 18:50 IST · ruchir (workstream owner) · attribution

<!-- Filename uses the workstream owner's name (Ruchir, per TEAM_PLAN.md §3). The session itself was run
     by Vijay with Claude Code: the post-presentation steps were not on main on 2026-11-21, so Vijay
     carried them out. Start time approximate (system clock is UTC). Same session as
     logs/2026-11-21_1850_vijay_sensitivity.md; the full transcript (/export) covers all of it. -->

- **Workstream owner (TEAM_PLAN.md):** Ruchir (`Ruchir-r`)
- **Carried out by (human in the session):** Vijay (`klnsv`)
- **Agent:** Claude Code (Claude Opus 5.5, `claude-opus-5-5`)
- **Workstream:** attribution (step 09 → C8)
- **Branch:** `vijay/final-analysis`
- **Transcript:** `logs/transcripts/2026-11-21_1850_ruchir_attribution.txt` (copy of the session export)

## Goal
Find which cell types and genes carry the mild-vs-severe signal (CONTEXT.md §7.1), producing C8.

## Prompts given (by Vijay, relevant to this workstream)
1. "alright, we need to do those now. Forget about all the team mates thingy. ..."
2. "yes create one branch for all of it. regarding the scope we shouldn't cut it short. What did you cut off?"
3. "alright let's complete this step by step and updating me with the results after each step. ..."
4. "yeah start it up"
5. "so far the model is not beating random right? You said it is negative so far"

## What was done
- Wrote `scripts/09_attribution.py`, reusing `classify_utils.py` (not edited). Eligible types (≥ 10 cells in ≥ 8 mild and ≥ 8 severe donors, checked from C4): ciliated, secretory, goblet, squamous, T cells.
- Single-cell-type models (logreg, linear SVM, RF; in-fold top-2000 genes → scale → PCA 10) with their own label-permutation nulls (1,000 / 1,000 / 200), run in parallel with the same shuffles and seeds as `classify_utils.permutation_null`.
- Ablation of a combined model (composition + 5 pseudobulk blocks; missing blocks median-imputed inside the fold), all three models, 10 repeats.
- Gene importance: logreg |coef| and RF mean |SHAP| (test fold) over 50 fits, stability = fraction of fits in the top 50.
- Quick test, then full run (14 min). Later removed a deprecated `vert=` argument from the boxplot call (figure-only, same output).

## Files changed
| File | Change |
|---|---|
| `scripts/09_attribution.py` | created |
| `results/attribution/celltype_importance.tsv`, `gene_importance.tsv` (C8) | created |
| `results/attribution/single_type_null.tsv`, `ablation_repeats.tsv` | created |
| `results/attribution/figures/09_single_type_auroc.png`, `09_ablation.png`, `09_top_genes.png` | created |

## Results / numbers produced
From console output of the full run (n = 21 severe / 14 mild overall):
- Whole sample (`all`): AUROC logreg 0.760, SVM 0.736, RF 0.795; perm p 0.015 / 0.019 / 0.010.
- Single types (logreg / SVM / RF, median AUROC): ciliated 0.616 / 0.623 / 0.609 (p 0.12–0.17, n 16 v 14); secretory 0.578 / 0.596 / 0.646 (p 0.13–0.24, 15 v 9); squamous 0.542 / 0.481 / 0.588 (12 v 9); T 0.500 / 0.535 / 0.562 (8 v 9); goblet 0.338 / 0.347 / 0.458 (12 v 9). None beats its null.
- Combined model AUROC: logreg 0.537, SVM 0.571, RF 0.726. ΔAUROC when dropped: composition 0.151 (RF), T cells 0.112 / 0.128 (logreg / SVM), ciliated 0.051 / 0.104 / 0.024.
- Top genes, e.g. ciliated logreg: CWH43, LINC00964, OASL, RYR3, IDO1; squamous logreg: CXCL10 (rank 1, stability 0.86).

## How to regenerate
```bash
.venv/bin/python scripts/09_attribution.py      # ~15 min on 12 cores
```

## Decisions made
- C8 `celltype_importance.tsv` has one row per cell type × model plus `composition` ablation rows, and extra min/max columns · CONTEXT.md §6 C8 note · yes
- No cell type meets the §7.1 "meaningful result" bar; reported as a negative finding (signal distributed, composition-driven, per-type n 17–30) · no

## Open issues / next steps
- Ruchir (owner) to review and write the attribution report section; classifier v2 (`pb_<ct>`, `comb` per CONTEXT.md §5.6) results are within this script's outputs.

## Note added 2026-11-25 (Ruchir, move to covid19_sev_prediction)
- Rerun of the cleaned `09_attribution.py` on Ruchir's machine (macOS 27.2.1) shifted some random-forest and linear-SVM values by at most 0.003 AUROC and 0.01 in permutation p. Logistic regression results and its gene importances are unchanged. The numbers above that moved (old → new, from `celltype_importance.tsv`): combined RF AUROC 0.7262 → 0.7287; ciliated single-type SVM 0.6228 → 0.6205; ciliated perm p range 0.12–0.17 → 0.13–0.16; ciliated RF ΔAUROC 0.0238 → 0.0213; composition RF ΔAUROC 0.1505 → 0.1514. The top-15 genes per cell type and model are the same set as before. No cell type beats its null, so the conclusion is unchanged.
- Cause: not the cleanup (core count no longer hard-coded; ablation figure title computed from the data). In quick mode, the original and cleaned scripts give byte-identical output on the same machine. The shift comes from running on a different machine and system maths library, as for step 05 (see `logs/2026-11-01_1055_ruchir_classify.md`). The regenerated results are the ones committed.
