# Session log: 2026-11-01 10:55 IST · ruchir · classify

- **Human:** Ruchir (`Ruchir-r`)
- **Agent:** Claude Code (Claude Sonnet 5.5, `claude-sonnet-5-5`)
- **Workstream:** classify
- **Branch:** `ruchir/classifier-v1`
- **Transcript:** `logs/transcripts/2026-11-01_1055_ruchir_classify.txt` (to be saved with `/export`)

## Goal
Fork the repo, set up the environment, and build the sprint classifier (`classify_utils.py`, `05_classify.py`, contract C5) on composition, pseudobulk and combined features. Separately, explore (outside the repo) whether AUROC above 0.90 is reachable for mild vs severe.

## Prompts given
1. "in a new downloads subdir open a fork to this using my github creds and read through instructions" (repo URL)
2. "do the install and create the branch. List out tasks in order (and dependencies if any)."
3. "done, start doing all the tasks"
4. "Have none of the others done their tasks ?"
5. "Yes do that, and in a separate folder outside this explore ways to achieve max performance >90 ideally"
6. "is it done?" / "status ?"

## What was done
- Installed `gh` and Python 3.12 with brew, made the venv, installed `requirements.txt`, forked to `Ruchir-r/covid19-scrna`.
- Release `data-v1` was not reachable with `gh`; downloaded the CELLxGENE fallback copy instead. md5 matched `2fcf7a8a3a52ec91e824f2041679195f`.
- Checked upstream: only 3 scaffolding commits, no branches, no PRs. C1-C4 do not exist yet, so `05_classify.py` builds the same features from the raw h5ad when C2/C3/C4 are missing (that path is the only one tested; the C2/C3/C4 reading path is NOT run).
- Wrote `classify_utils.py` and `05_classify.py`; quick test, then full run (10 CV repeats; 1000 permutations for logreg/SVM, 200 for RF).
- Exploration in `~/Downloads/covid19-scrna-explore/` (outside the repo): feature/model screen, cell-level classifier, nested CV.

## Files changed
| File | Change |
|---|---|
| `scripts/classify_utils.py` | created: `make_model`, `run_cv`, permutation helpers (optional `pca`, `n_plain` args added to the contract signatures) |
| `scripts/05_classify.py` | created: C5 pipeline, feature sets `comp`, `pb_all`, `comb` |
| `results/classify/*.tsv`, `figures/*.png` | created by the script |
| `CONTEXT.md` | edited: my own row in section 8 only |

## Results / numbers produced
From `results/classify/cv_metrics.tsv`, mild vs severe (14 mild, 21 severe patients), repeated 5-fold, median AUROC [min-max], permutation p:

| Features | logreg | linsvm | rf |
|---|---|---|---|
| comp | 0.740 [0.684-0.786] p=0.033 | 0.752 [0.643-0.789] p=0.025 | 0.832 [0.741-0.908] p=0.005 |
| pb_all | 0.760 [0.646-0.827] p=0.015 | 0.736 [0.656-0.827] p=0.019 | 0.795 [0.599-0.830] p=0.010 |
| comb | 0.759 [0.670-0.823] p=0.016 | 0.757 [0.714-0.816] p=0.016 | 0.837 [0.757-0.888] p=0.005 |

Control vs COVID (15 vs 35) sanity task: AUROC 0.82-0.93, p <= 0.005; best comb linsvm 0.926.
These p-values are not corrected across the 9 model x feature combinations. Leave-one-patient-out has no permutation p-value (not run).
Exploration (`~/Downloads/covid19-scrna-explore/`, see its README): screen best 0.83 (optimistic), cell-level 0.80, age+sex only 0.72, nested CV 0.835. Nothing reached 0.90 without selection bias.
Results come from features built from the raw h5ad, not from C3/C4.

## How to regenerate
```bash
python scripts/05_classify.py
```

## Decisions made
- Fallback feature building from raw until C3/C4 merge · avoids blocking on teammates · no
- Added optional `pca` / `n_plain` args to `make_model` / `run_cv` · pseudobulk and combined features need in-fold PCA · yes (contract signature extended, backward compatible)
- Permutations only for the repeated 5-fold scheme · cost · no

## Open issues / next steps
- Rerun `05_classify.py` once Sahaj's C2/C3/C4 are merged and compare numbers; the reading path for those files is untested.
- Age and sex alone give AUROC 0.72 (exploration); worth telling the team as a confound.
- Nothing committed or pushed; the human reviews the diff, commits, pushes and opens the PR.
- Slides 7-8, classifier v2 and `09_attribution.py` not started.

## Note added 2026-11-25 (Ruchir, move to bt3041_proj)
- Rerun of the cleaned `05_classify.py` (C2/C3/C4 inputs, macOS 27.2.1, same package versions) changed 3 values in `cv_metrics.tsv`, all for the `comb` linear SVM: mild vs severe median AUROC 0.757 → 0.760 and max 0.816 → 0.820; control vs COVID min 0.878 → 0.876. All permutation p-values and all other rows are unchanged.
- Cause: not the cleanup. In quick mode, the original (`3c10d75`) and cleaned scripts give byte-identical output on the same machine, and the 2026-11-02 rerun reproduced the committed files exactly. The shift follows a macOS update (2026-11-20) that changed the system maths library: logistic regression moves by about 1e-9 and the linear SVM amplifies the difference. The regenerated results are the ones committed.
