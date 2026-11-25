# Session log: 2026-11-21 18:50 IST · vijay · validation

<!-- Start time approximate (system clock is UTC). Same Claude Code session as
     logs/2026-11-21_1850_vijay_sensitivity.md. That session also produced steps 07, 08, 09, 10, 12, 13,
     16 and the report draft; they are not Vijay's workstreams and are not described here, but step 14
     reads some of their outputs (listed below). The full transcript (/export) contains all of it. -->

- **Human:** Vijay (`klnsv`)
- **Agent:** Claude Code (Claude Opus 5.5, `claude-opus-5-5`)
- **Workstream:** validation
- **Branch:** `vijay/sensitivity-validation`
- **Transcript:** `logs/transcripts/2026-11-21_1850_vijay_validation.txt`

## Goal
Check the claims of Ziegler et al. 2021 (Eswar's table, `docs/literature/ziegler_findings.md`, 16 claims) one by one against our data at donor level (CONTEXT.md §7.5). Decide on the Chua et al. external cohort (step 15).

## Prompts given (relevant to this workstream)
1. "here we need to validate the claims made by Ziegler and Eswar all those 16 claims. Read through them"
2. "We need to employ hypothesis testing stuff like Mann whitney, Student t testing, Bonferroni, Benjamini hochberg corrections to say that those claims are significant enough or no"
3. "Help me code this logic in 14_validation_ziegler.py and always use alpha=0.05"
4. "Save the results in ziegler_checks.tsv, viral_load.tsv, 14_claims.png"
5. "Oh I forgot, save this chat history as well under the logs directory"

## What was done

- Read the 16 claims and the 39 detailed cell-type labels in C1 (to map "interferon-responsive ciliated", "FOXJ1 high ciliated", "MUC5AC high goblet").
- Wrote and ran `scripts/14_validation_ziegler.py` (≈ 5 s). Every check is donor level except claim 15 (cell level, exploratory, no p-value). Fixed the claim-2 direction check after reading the actual `direction` values in C6, and split the "not supported" verdict into "not supported (opposite, n.s.)" and "contradicted" after the first run.

## Inputs read by step 14
C1, C2, C3; C6 `results/composition/composition_tests.tsv` (Sahaj, merged); step 06 `results/isg/isg_tests.tsv`, `donor_isg_scores.tsv` (Sidharth, merged); C7 `results/de/de_<ct>_<contrast>.tsv` and `results/de/enrichment_gsea.tsv` (steps 07–08, produced earlier in this same session on this branch).

## Files changed
| File | Change |
|---|---|
| `scripts/14_validation_ziegler.py` | created |
| `results/validation/ziegler_checks.tsv` | created: one row per check (claim, check, cell type, n, our result, p, q, verdict, source) |
| `results/validation/viral_load.tsv` | created: per-donor viral UMIs per million |
| `results/validation/figures/14_claims.png` | created |
| `logs/2026-11-21_1850_vijay_validation.md` | created: this log |

## Results / numbers produced
From the console output of `python scripts/14_validation_ziegler.py` (59 checks of 16 claims):
- Verdicts: supported 13; consistent, not significant 32 (+1 "proportion only", claim 16); not supported (opposite, n.s.) 6; contradicted 0; descriptive 6; not testable 1.
- Claim 1: ciliated proportion median 0.382 mild vs 0.116 severe (p 0.019, q 0.142); secretory 0.037 vs 0.110 (p 0.072, q 0.174).
- Claim 2: deuterosomal higher in COVID (q 0.0063); developing ciliated higher in COVID (q 0.450).
- Claim 3: epithelial Gini-Simpson diversity median 0.540 control vs 0.625 COVID (p 0.017).
- Claim 5: interferon-responsive ciliated cells, median % of epithelial cells: 10.4% mild vs 0.4% control (p 0.0002) and vs 0.8% severe (p 0.0053).
- Claim 6: Hallmark IFN-α GSEA severe vs mild NES −2.83 (ciliated) and −2.52 (whole sample), COVID vs control +3.64 / +3.54 (all FDR < 0.001). Ciliated pseudobulk genes all lower in severe, none padj < 0.05 (e.g. STAT1 log2FC −1.09, padj 0.090; MX1 −1.01, padj 0.24). Step 06 ISG scores lower in severe (p 0.053–0.096).
- Claim 7: goblet IFN-α NES −2.17 (FDR 0.0005).
- Claim 8: IFNAR1, IFNAR2, IFNGR1 not lower in severe ciliated cells (no padj < 0.05).
- Claim 9: S100A8/S100A9 up in severe (e.g. S100A9 whole sample log2FC +1.93, q 0.076); HSPA8, HSPA1A, DUSP1 mixed (6 opposite, n.s.).
- Claim 11: viral UMIs per million, median 0.0 mild vs 5.0 severe; geometric mean(+1) 7.5 vs 34.8 (p 0.167).
- Claim 12: secretory % vs log10 viral load ρ +0.12 (p 0.48); FOXJ1-high ciliated ρ −0.25 (p 0.15).
- Claim 13: 415 RNA+ cells: severe 262, mild 150, control 1, convalescent 2 (matches the preprint).
- Claim 14: most frequent RNA+ types: Interferon Responsive Ciliated 82, AZGP1 high Goblet 52, … (differs from the preprint's order; descriptive).
- Claim 15 (exploratory): 7 of 12 donor × cell-type pairs with ≥ 5 RNA+ cells had higher antiviral-gene expression in RNA+ than bystander cells.
- Claim 16: macrophage proportion median 0.0010 mild vs 0.0152 severe (q 0.142); only 1 mild / 6 severe donors have ≥ 10 macrophages.

## How to regenerate
```bash
.venv/bin/python scripts/14_validation_ziegler.py   # needs C1-C3, C6, step 06 and steps 07-08 outputs
```

## Decisions made
- Step 15 (Chua et al. 2020) dropped by the human; stated as future work / limitation in the report · CONTEXT.md decisions log? yes
- Verdict scheme: supported / consistent, not significant / not supported (opposite, n.s.) / contradicted / descriptive / not testable · no
- Claim 7 uses all goblet cells (the per-subtype "MUC5AC-high" pseudobulk has too few donors) · no
- Claim 10 (steroids) marked not testable: no steroid field in the data · no

## Open issues / next steps
- Claim 14's cell-type order differs from the preprint; may reflect label differences between the preprint and the CELLxGENE release (not checked).
- CONTEXT.md §8 status rows and the decisions log need updating (shared file; separate small PR per CONVENTIONS.md §3).
