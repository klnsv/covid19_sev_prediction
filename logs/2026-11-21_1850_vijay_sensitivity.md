# Session log: 2026-11-21 18:50 IST · vijay · sensitivity

<!-- Start time approximate (system clock is UTC; session work began ~13:20 UTC).
     The same Claude Code session also produced steps 03 (rebuild), 07, 08, 09, 10, 12, 13, 16 and the
     report draft; those are not Vijay's workstreams and are not described in this log. The full
     transcript (/export) contains all of it. Validation (step 14) has its own log:
     logs/2026-11-21_1850_vijay_validation.md -->

- **Human:** Vijay (`klnsv`)
- **Agent:** Claude Code (Claude Opus 5.5, `claude-opus-5-5`)
- **Workstream:** sensitivity
- **Branch:** `vijay/sensitivity-validation`
- **Transcript:** `logs/transcripts/2026-11-21_1850_vijay_sensitivity.txt`

## Goal
Run the sensitivity analysis of CONTEXT.md §7.3 (axes A–G) for the final report, plus a patient-level bootstrap CI of the AUROC (agreed with the human instead of the Chua et al. external cohort).

## Prompts given (relevant to this workstream)
1. "as per team plan, what should vijay do now? I did qc and did PCA and umap so far. What else is left for me?"
2. "alright, understood. We need to do sensitivity analysis now. We obtained ROC of 0.83 which is much better than random but we need to do sensitivity analysis to ensure that the model is robust. How do we go about this sensitivity testing though? Can you give me an example for reference?"
3. "How about we do this? we use 7 axes. Severity definition(like the WHO score), Specificity(Is the model just predicting COVID severe for people in ICU), Features(try it out with fewer & more genes), QC(remove some contaminated datapoints), do bootstrapping & loocv, age & sex(check if we are just saying severe if patient is older) "
4. "I have created a new file called 11_sensitivity.py in the scripts directory so just help me code the sensitivity analysis there'
5. "The AUROC score still seems around the 0.83 so it is good. save the chat history also. It doesn't seem like the model just predicts severity based off age according to the results. Save all the results in the results directory so that i can write the report."

## What was done
- Rebuilt C2–C4 locally with `scripts/03_patient_features.py` (merged code, unchanged) because step 11 reads them.
- Wrote `scripts/11_sensitivity.py`. Each setting rebuilds patient features from C1 with step 03's definitions and reruns `classify_utils.run_cv` (repeated stratified 5-fold, seeds 0–9) with a 200-permutation null, for four feature/model pairs: comp/rf, comp/logreg, pb_all/logreg, comb/rf.
- Quick test (`--quick`) ran end to end (3 m 55 s). Full run: 43 m 24 s, wrote all outputs listed below.
- From the full run, found that severe patients are older and that sex + age alone predicts severity. Added an **age-stratified permutation test** to axis G (labels shuffled only within < 60 / ≥ 60 years) and started a full rerun; that rerun was cut off when the session ended (no outputs written).
- 2026-11-24 (same session, resumed): ran the full script again, unchanged (54 m 37 s). All earlier rows came out identical (fixed seeds; `git diff` shows only the 4 new age-stratified rows in `sensitivity_summary.tsv`). Rebuilt the report figures (`scripts/16_report_figures.py`); only `fig2_classification.png` changed (new row in panel b).
- Checked age by group directly from C2 (numbers below).

## Files changed
| File | Change |
|---|---|
| `scripts/11_sensitivity.py` | created: axes A–G + bootstrap + age-stratified test |
| `results/sensitivity/sensitivity_summary.tsv` | created (C12; 91 rows incl. 4 age-stratified rows) |
| `results/sensitivity/specificity_scores.tsv`, `influence.tsv`, `downsampling_draws.tsv`, `bootstrap_ci.tsv` | created |
| `results/sensitivity/figures/11_forest.png`, `11_downsampling.png`, `11_specificity.png` | created |
| `logs/2026-11-21_1850_vijay_sensitivity.md` | created: this log |

## Results / numbers produced
All from the console output of the first full run of `python scripts/11_sensitivity.py` (n = 21 severe / 14 mild unless stated; AUROC median [min–max] over 10 CV repeats; permutation p with 200 permutations):
- **A. Severity definition.** Primary comp/rf 0.832 [0.740–0.908] p 0.005; comb/rf 0.837 p 0.005; pb_all/logreg 0.760 p 0.020; comp/logreg 0.740 p 0.035. Drop the WHO-6 patient (20 v 14): comp/rf 0.822 p 0.010. Extremes WHO ≤ 4 vs 8 (16 v 9): 0.899–0.934, all p 0.005. Ordinal: Spearman ρ between OOF score and WHO peak 0.555–0.765 (p ≤ 0.0005).
- **B. Specificity.** Models trained on the 35 COVID patients called 33–50% of the 6 non-COVID ICU controls "severe" and 20–47% of the 15 healthy controls; AUROC severe (OOF) vs ICU 0.452–0.754.
- **C. Feature set.** pb_all with 500 / 1,000 in-fold HVGs: 0.757 (p 0.030) / 0.784 (p 0.015). Detailed (39-type) composition: rf 0.648 (p 0.159), logreg 0.702 (p 0.060). Single cell types (logreg): ciliated 0.616 (p 0.129), secretory 0.578, goblet 0.338, squamous 0.542, T 0.500; none p < 0.05.
- **D. QC and inclusion.** ≥ 50 cells (20 v 13): comp/rf 0.828 p 0.010; ≥ 100 cells (17 v 10): 0.776 p 0.035; no bloody swabs (19 v 14): 0.826 p 0.015; mito < 30%: 0.829 p 0.005 (29,039 of 32,588 cells kept); mito < 20%: 0.808 p 0.005 (25,222 kept). pb_ciliated with min 5 / 20 cells: 0.664 (p 0.055) / 0.589 (p 0.239).
- **E. Cells per patient** (median over 10 draws): 50 cells 0.642–0.722, 100 cells 0.684–0.743, 200 cells 0.696–0.764 across the four pairs (all cells: 0.740–0.837).
- **F. Influence and uncertainty.** Leave-one-donor-out AUROC comp/rf 0.788–0.883, comb/rf 0.796–0.891. Patient bootstrap (2,000 resamples) of repeat-averaged OOF AUROC: comp/rf 0.854 [95% CI 0.685–0.984], comb/rf 0.861 [0.708–0.980], pb_all/logreg 0.796 [0.627–0.939], comp/logreg 0.762 [0.581–0.923].
- **G. Covariates.** comp + sex + age: rf 0.829 p 0.005, logreg 0.723 p 0.050. Sex + age only: rf 0.763 p 0.025, logreg 0.709 p 0.065. OOF score vs sex: Mann–Whitney p 0.518–1.000; vs age decade: Spearman ρ 0.402–0.474 (p 0.004–0.017).
- Age by group (from C2): median age decade 4.5 mild vs 6 severe, Mann–Whitney p 0.0205.
- **Age-stratified permutation test** (200 permutations, labels shuffled within < 60 / ≥ 60): comp/rf AUROC 0.832 vs stratified-null median 0.515, p 0.005; comb/rf 0.837 vs 0.526, p 0.005; comp/logreg 0.740 vs 0.510, p 0.030; pb_all/logreg 0.760 vs 0.509, p 0.030. All four models beat a null that keeps the age–severity association, so the nasal signal is not explained by age alone (within this coarse two-stratum adjustment).

## How to regenerate
```bash
.venv/bin/python scripts/03_patient_features.py     # C2-C4, if not built locally
.venv/bin/python scripts/11_sensitivity.py          # ~45 min on 12 cores
```

## Decisions made
- Chua et al. external cohort (step 15) dropped by the human; replaced by a patient-level bootstrap CI and stated as a limitation · needs adding to CONTEXT.md decisions log? yes
- `top3_celltypes` and `jaccard_top20_genes_vs_primary` use cheap proxies (composition logreg coefficients; top-20 pb_all logreg genes), not a full step-09 rerun per setting · documented in the script docstring · yes
- C12 has extra columns `extra_metric`, `extra_value`, `extra_p`, `extra_note` for rows that are not AUROCs (ordinal ρ, specificity fractions, covariate tests) · CONTEXT.md §6 C12 should mention them · yes
- Mitochondrial cut-offs use our recomputed `pct_mito` (open issue from the 2026-11-01 QC log) · no
- Age strata for the stratified test: < 60 vs ≥ 60 years (decade ≥ "seventh") · no

## Open issues / next steps
- Report section 4.2 still says "p pending" (the `.docx` build script was in Claude's session scratch folder and was lost); the sentence is replaced by hand in the Google Doc (Eswar's review).
- Report text must state the age overlap and the ICU caveat (axis B) as limitations.
