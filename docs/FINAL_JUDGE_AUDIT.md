# Independent QFF 2026 Q1 judge audit — 2026-10-14 IST

## Verdict

**NO-GO for the stated October 14, 2026, 23:59 IST deadline as of the October 15 client date; CONDITIONAL technical GO only if the organizer accepts a late/extended submission.** The repository is reproducible, preserves the official 40-row benchmark, and gives explicit architecture and failure analysis. Its major unresolved compliance risk is the Master Guide's invalid literal generator call: `n_redundant=0` is an engineering repair without specific organizer confirmation. Its scientific competitiveness is limited by the official VQC's weak recall and the absence of clinical or hardware validation. Public GitHub publication and portal submission have not occurred. No extension or late-submission eligibility has been verified.

The review used the latest Master Guide (pp.1–3, 5, 7) before the Updated Challenge Kit (Q1 section), then the Problem Statement Details and other supplied PDFs. The Master Guide governs the Phase 1 synthetic task, 5/9-qubit A/B budgets, participant-defined maps, noise, basis, seed and 4096 shots. The lower-priority biomedical-data and fixed-map descriptions remain documented conflicts; no organizer approval was inferred.

The three review lenses reach distinct conclusions: the **judge** can audit a complete, clearly separated Phase 1 package but should penalize unsupported superiority claims; the **ML scientist** finds no demonstrated split or preprocessing leakage yet regards 20 positive test rows and post hoc comparisons as narrow evidence; the **Qiskit engineer** verifies the frozen A/B model and compiler/noise settings while limiting the routing result to these custom graphs, circuit and compiler version.

## Findings and dispositions

### J1 — BLOCKER, confirmed specification ambiguity; unresolved

- **Reference/evidence:** [synthetic fixture](../src/datasets.py:139) sets `n_redundant=0` at line 146. Master Guide p.5 asks for 200 rows, four features and three informative features but omits `n_redundant`; scikit-learn's default two redundant features makes that literal call invalid.
- **Judging impact:** a strict organizer interpretation could reject or re-score the provisional fixture even though it is deterministic and clearly labeled.
- **Exact correction:** obtain explicit organizer acceptance of `n_redundant=0`, or create a separately named new experiment with their corrected generator instruction; retain all current hashes and results. No approval is claimed.
- **Verification:** `test_corrected_fixture_is_distinct_and_training_fitted`; [model/data identity](../results/tables/q1_provisional_manifest.json). Organizer acceptance is **NOT VERIFIED**.

### J2 — HIGH, confirmed model limitation; transparently reported, not retrained

- **Reference/evidence:** [final-layer rotations](../src/vqc.py:96) include RZ gates at line 101 followed only by CX permutations and computational-basis measurement. [Exact replay audit](../results/tables/q1_judge_audit.json) finds zero score change after separately perturbing `w[9]`, `w[11]`, `w[13]`, `w[15]` by 0.37 radians. The official ideal VQC misses 13/20 positives and has recall 0.350, versus logistic recall 0.850.
- **Judging impact:** four of 16 trainable angles are ineffective for this readout, and the official model does not outperform a small classical baseline.
- **Exact correction:** keep this frozen official result; describe the mechanism and bounded optimizer evidence. A revised circuit would need a new architecture identity, training and separately labeled benchmark, so it is not silently substituted near submission.
- **Verification:** exact saved-score replay in `src.judge_diagnostics`; `test_original_artifact_hash_and_exact_score_replay`. Causal contribution of these angles to recall is **NOT ISOLATED**.

### J3 — HIGH, confirmed overinterpretation risk; corrected

- **Reference/evidence:** [statistical review](../results/tables/q1_statistical_review.json:344) shows noisy A versus B differs by one A-only and three B-only correct predictions, exact paired two-sided p=0.625; positive-row paired p=0.625. The 95% Wilson recall intervals are 0.112–0.469 (A) and 0.181–0.567 (B).
- **Judging impact:** the point difference 0.250 versus 0.350 recall is too small on 20 positive rows to claim B improves classification.
- **Exact correction:** README, report and judge summary now distinguish **measured compiler-cost advantage** from uncertain classification differences; [recall intervals](../results/figures/recall_uncertainty.svg) are displayed.
- **Verification:** `test_saved_statistics_and_paired_comparisons`, notebook uncertainty output. These tests are exploratory, with repeated test-set inspection and no multiplicity adjustment.

### J4 — MEDIUM, confirmed threshold trade-off; corrected

- **Reference/evidence:** [validation selection](../src/recall_calibration.py:131) precedes test scoring at lines 138–143. [Independent statistical replay](../src/judge_statistics.py:106) verifies the deterministic 128/32 inner split, 128-row scaler fit, saved validation labels/grid and threshold 0.45 without test labels. The separate model gets recall 0.800 but specificity 0.250, 15 false positives, accuracy 0.525 and precision 0.516.
- **Judging impact:** presenting recall alone would conceal many false alarms and confuse a separately fitted model with the official full-training VQC.
- **Exact correction:** all primary documents and notebook label it **separate** and show precision, F1, specificity and false positives. The official model and A/B results are unchanged.
- **Verification:** `test_inner_split_scaler_and_test_metrics`, `test_validation_label_tamper_is_rejected`, [saved statistical record](../results/tables/q1_statistical_review.json). Chronological intent is supported by source order and saved evidence; an external immutable run log is unavailable.

### J5 — MEDIUM, confirmed layout and graph confounding; partly mitigated

- **Reference/evidence:** [official compiler settings](../src/routing.py:14) use level 1, seed 42, SABRE layout/routing and `rz/sx/x/cx`; A is a 5-node line, B a 9-node grid. Their native-basis cost is 24 CX/depth 35 and 8 CX/depth 20, respectively. [Layout ablation](../results/tables/q1_layout_ablation.json) changes only initial placement: trivial gives A 22 CX/depth 34, B 14 CX/depth 26. [Three-seed check](../results/tables/q1_seed_robustness.json) records A 24 CX/depth 33–35 and B 8 CX/depth 20, with routing-stage SWAPs 6/0 at seeds 7, 42, 99.
- **Judging impact:** a one-seed, unequal-qubit comparison cannot establish universal topology superiority; unconstrained-depth deltas are compilation proxies, not pure SWAP depth.
- **Exact correction:** retain seed-42 official values, expose one controlled initial-layout ablation and one bounded compilation-only seed check; limit the conclusion to this logical circuit and settings. No alternate-seed noisy inference claim is made.
- **Verification:** `test_ablation_preserves_official_compilation`, `test_seed_probe_retains_frozen_seed_42_and_costs`, `test_processor_graphs_and_sabre_swap_trace`.

### J6 — MEDIUM, confirmed presentation ambiguity; corrected without changing evidence

- **Reference/evidence:** preserved [original figure](../results/figures/classification_metrics.svg:1410) labels synthetic recall “Abnormal recall”. The historical [A/B module](../src/processor_benchmark.py:88) defaults to old 0.001/0.01/0.02 noise, whereas the official [entry point](../src/q1_submission.py:35) uses 0.001/0.005/0.01.
- **Judging impact:** a judge could infer clinical labels or run the historical entry point as if it were official.
- **Exact correction:** the plot generator now uses “Synthetic label-1 recall”; the notebook presents a new [clarified comparison](../results/figures/judge_model_comparison.svg). The historical module explicitly announces its role, and README names the official CLI. Original SVG/JSON/weights remain byte-preserved.
- **Verification:** `test_notebook_primary_plot_uses_synthetic_label_language`, `test_saved_figures_expose_measured_labels`; official model and processor SHA256 identities unchanged.

### J7 — LOW, stale checklist evidence; corrected

- **Reference/evidence:** [submission checklist](SUBMISSION_CHECKLIST.md) still described 29 tests/four new SVGs/six notebook code outputs before this audit.
- **Judging impact:** inconsistent verification counts reduce confidence despite passing work.
- **Exact correction:** refresh the checklist and [final verification record](FINAL_VERIFICATION.md) with the actual final commands and counts.
- **Verification:** final package review and notebook schema/output check below.

### J8 — MEDIUM, confirmed publication prerequisite; unresolved by design

- **Reference/evidence:** local `.git/config:1-7` contains only `[core]`; `git config --show-origin --get user.name` and `user.email` both exited 1 with no value. The repository also has no commit or remote.
- **Judging impact:** the staged technical package cannot be committed or submitted through GitHub until a valid author identity and repository URL are supplied.
- **Exact correction:** the team supplies its preferred Git author name/email and creates a public GitHub repository; configure identity locally, then commit, add that remote and push only after approval and a timing eligibility check.
- **Verification:** `git config --get user.name`, `git config --get user.email`, `git status`, `git remote -v`, and public URL review after the team's action. No identity or URL has been invented.

## Model, statistics and architecture evidence

The **official before and after metrics are identical**; no official model, dataset, threshold or noisy benchmark was overwritten. Official model SHA256 is `9902dd5dce0d069633081b7787618f98797fc1442c5ceff5ac3679067c8332fd`, processor benchmark SHA256 is `3c6bd53295d4c22adfafb42a285240890c5df411a4bcb9fa4040fb0dfd3dc908`. The separate validation model SHA256 is `aa2f0f853f33ec139fbe855b9019b0dbaf7a0496d33408dbde2195122805289c`. Historical original-fixture results and the prior 0.001/0.01/0.02 noise experiment remain in `results/tables/historical/` and are not compared as if they shared this fixture.

| Corrected 40-row test case | Accuracy | Precision | Recall | F1 | Specificity | FP / FN | Confusion matrix `[0,1]` |
|---|---:|---:|---:|---:|---:|---:|---|
| Logistic | 0.875 | 0.895 | 0.850 | 0.872 | 0.900 | 2 / 3 | `[[18,2],[3,17]]` |
| Official ideal VQC | 0.600 | 0.700 | 0.350 | 0.467 | 0.850 | 3 / 13 | `[[17,3],[13,7]]` |
| Official noisy A | 0.550 | 0.625 | 0.250 | 0.357 | 0.850 | 3 / 15 | `[[17,3],[15,5]]` |
| Official noisy B | 0.600 | 0.700 | 0.350 | 0.467 | 0.850 | 3 / 13 | `[[17,3],[13,7]]` |
| Separate validation model | 0.525 | 0.516 | 0.800 | 0.627 | 0.250 | 15 / 4 | `[[5,15],[4,16]]` |

The logistic-versus-official-ideal accuracy comparison has exploratory exact paired p=0.0034 (12 logistic-only correct versus one VQC-only). This does not turn a synthetic 40-row benchmark into a population or clinical guarantee. Wilson intervals and all five paired contrasts are in [machine-readable statistics](../results/tables/q1_statistical_review.json). No held-out label was used to select the official model or separate threshold.

Qiskit checks: the logical circuit has four qubits, eight CX and gate depth 13; custom Processor A/B JSONs are connected bidirectional 5/9-node graphs. Both official mapped runs use the same frozen weights and 40 test rows, native basis, level-1 seed-42 SABRE settings, 4096 shots, Aer density-matrix simulation and identical synthetic noise 0.001/0.005/0.01. Unconstrained native depth/CX is 20/8. A has gate/full depth 35/36, 24 CX, +15/+16 overhead and six **explicit DAG SWAPs immediately after SabreLayout**; B has 20/21, 8 CX, zero overhead and zero trace-stage SWAPs. The count is not inferred from CX/3. Neither map is claimed as organizer-provided hardware.

## Verification and release risk

Final command results, notebook execution, staged-export replay, package/secret checks and any failures are recorded in [FINAL_VERIFICATION.md](FINAL_VERIFICATION.md). The specific residual risks are: expired stated deadline or unverified extension, unresolved generator approval (J1), missing publication identity/remote (J8), low official quantum recall (J2), only 20 positive test examples, post hoc comparisons on one synthetic split, and no real biomedical or physical-hardware validation. Processor C/D are outside Phase 1. There is no claim of guaranteed ranking or quantum advantage.
