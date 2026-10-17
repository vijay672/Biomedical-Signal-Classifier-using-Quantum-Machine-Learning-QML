# QFF 2026 Phase 1 submission checklist — updated 2026-10-17

## Verified package

- [x] Highest-priority Master Guide used for Phase 1 PS-Q1; only Processor A (5 qubits) and B (9 qubits) are benchmarked. Lower-priority biomedical and fixed-map conflicts are recorded in `docs/DECISIONS.md`.
- [x] Master Guide p.3 structure: `README.md`, `main.ipynb`, `requirements.txt`, `report.md`, `data/raw/`, `data/processed/`, `processors/processor_A.json`, `processors/processor_B.json`, `src/problem.py`, `src/processors.py`, `src/routing.py`, `src/metrics.py`, `src/visualization.py`, `results/tables/`, `results/figures/`. `.gitkeep` retains empty raw/processed directories. Optional equation-image assets are not needed.
- [x] Provisional corrected 200×4 data with three informative features, zero redundant features and seed 42 is explicitly labeled; original dataset artifacts are separate. Shared 160/40 stratified split and train-fitted scaler identity are validated.
- [x] New VQC was trained once on corrected data, with 25 real callback objective values. The saved schema-2 model includes architecture, parameter names/order/count, dataset/split/scaler hashes and package versions. Loading rejects mismatches.
- [x] Processor A/B share the same corrected model SHA256 `9902dd5dce0d069633081b7787618f98797fc1442c5ceff5ac3679067c8332fd`, dataset identity `5e4bebee74b379c199394c99c7deb6ee5418843d0b6380b1f1ef75c7b2bf7b9e`, basis, explicit SABRE settings, level 1, seed 42, 4096 shots and noise rates 0.001/0.005/0.01. Their maps are labeled custom.
- [x] `results/tables/q1_provisional_*.json` contains all predictions and accuracy, precision, label-1 recall, F1 and confusion matrices. Circuit costs separate gate-only/full depths and CX; a later Qiskit 2.5.2 SABRE-stage trace independently verified inserted SWAPs as described below.
- [x] Sixteen current SVG figures include coupling graphs, circuits, mapping, comparisons, confusion matrices, interval plots and training convergence. Text circuit dumps are also saved. `main.ipynb` executed successfully with seven output-bearing code cells and embedded visual evidence.
- [x] Original model/baseline SHA256 hashes remain `cf214628f7137e6760973b60a7d48c222d337f1eac2acdeff03902c6fb79644b` and `f0b0b192a1e0a3e34bb4c7bd00a55a79249cbca286eaae7924daa5cb8ab38895`; historical alternative-noise output is archived and labeled.
- [x] `.gitignore` allowlists final JSON, text circuits, notebook and SVG evidence while excluding local raw data, caches and temporary root results. `git add --all` in the initialized repository includes the complete package and excludes `.venv`, Python caches, generated local datasets, and legacy root outputs. No secret-pattern matches or files over 5 MB were found. Organizer source PDFs total about 3.1 MB and are included as relevant documentation; the executed notebook is 1,537,101 bytes.
- [x] The prior 23/23 suite passed at remediation; the final independent-review suite is **35/35 passed**. `src.q1_submission.smoke_test()` passed. Current `.venv/bin/python -m pip check`: **passed**. Fresh isolated Python 3.11 environment installation of `requirements.txt`, `pip check` and imports was checked during remediation. Saved-artifact identity, history hashes, figure files and report-value checks passed.
- [x] README's documented CLI target `python -m src.q1_submission` now loads and validates complete saved outputs in a fresh process without retraining or resimulation; it trains only if the manifest is absent. Notebook execution loaded and verified the outputs without retraining. Exact reproducibility across a second official training run was not tested, to avoid unnecessary retraining.
- [x] Final packaging validation: all Master Guide p.3 mandatory paths exist; final JSON parses; notebook schema validates; Git inclusion review covers the final manifest/benchmark/figures; `.venv`, caches and temporary outputs are ignored. At the initial review no remote was configured; `origin` was added during publication below.
- [x] Final organizer discrepancy check: Master Guide precedence confirms A=5, B=9 and participant-designed graphs. Processor B is a valid connected nine-node bidirectional 3×3 grid labeled custom. README, report, executed notebook and final JSON tables contain accuracy, label-1 recall and F1 for logistic, exact VQC and noisy A/B. Label 1 is explicitly non-clinical.
- [x] Deterministic transpilation-only SWAP trace with Qiskit 2.5.2: 6 explicit SWAP nodes after `SabreLayout` for A and 0 for B. Method and compiler settings are saved in `results/tables/q1_provisional_swap_diagnostic.json`; existing benchmark outputs and final +16 CX/+15 depth versus zero/zero overhead were preserved.
- [x] Focused discrepancy checks: 7/7 routing, processor configuration, dataset identity and artifact tests passed. Independent consistency checks passed for graph validity, repeatable 6/0 SWAP traces, +16/+15 versus 0/0 overhead, required metrics in JSON/notebook, and non-clinical label wording. Corrected and historical model hashes remain unchanged.
- [x] At the previous upgrade stage, original model and benchmark SHA256 values remained unchanged; `q1_judge_audit.json` recalculated six metrics cases and exactly replayed ideal QNN scores. One separately labeled 128-fit/32-validation model selected threshold 0.45 before one held-out test evaluation. `q1_layout_ablation.json` measured SABRE versus trivial initial placement without changing the official benchmark. That stage added four SVGs and six executed notebook outputs, with 29/29 tests passing. The newer counts appear above and in the final independent-audit item below.
- [x] Final Git dry-run includes the new JSON, figures, notebook, processor configurations, reports and `docs/JUDGE_SUMMARY.md`/`docs/FINAL_VERIFICATION.md`; ignored files are `.venv`, caches, temporary outputs and local datasets. No source/document secret-pattern hits or non-environment files over 5 MB were found.
- [x] Exported the staged package to a temporary clean directory with `git checkout-index` and replayed the documented official CLI, diagnostics, saved separate-experiment loader and figure generator there. All passed without ignored local files or retraining; official and separate model/benchmark SHA256 values matched the originals. Generated SVG/text circuit whitespace is exempted in `.gitattributes` so staged source lint passes without changing evidence bytes.
- [x] Independent final judge audit adds 95% Wilson intervals, specificity/FP/FN and five exploratory exact paired contrasts from saved aligned test predictions; one 7/42/99 compilation-only seed check; a clarified synthetic-label figure; and `docs/FINAL_JUDGE_AUDIT.md`. These are separate from frozen official results. New focused tests 6/6 and full suite 35/35 pass; the executed notebook has seven output-bearing code cells.
- [x] October 16 circuit-focused review verified the exact four-qubit gate order, all 16 parity words, complete routed distribution/measurement equivalence, original Qiskit drawing bytes, and six output-inert angles. Added a negative MIT-BIH CSV label-feature check and a conceptual D topology test. Full suite **42/42 passed**, `pip check` passed, and the executed notebook has eight output-bearing code cells. See `docs/FINAL_OFFICIAL_RUBRIC_AUDIT.md` and `docs/FINAL_VERIFICATION.md`.
- [x] The A/B/D topology schematic is included as `results/figures/architecture_problem_and_D_concept.svg`; it reads saved A/B maps and placements and labels D as **conceptual, uncompiled and unmeasured**. No official benchmark result or model weight changed.

## Required before an organizer-compliant public submission

- [x] **Timing:** The team reports that October 17, 2026 is the organizers' final submission day. This latest organizer confirmation supersedes the October 14 date printed in the locally available Master Guide.
- [ ] **Organizer confirmation:** Master Guide's literal `make_classification` call fails under scikit-learn defaults. Confirm that `n_redundant=0` is accepted, or rerun/relabel the benchmark under the organizer's corrected generator instruction. This provisional assumption remains a compliance blocker.
- [x] **New rubric confirmation:** the team reports organizer confirmation of the five categories and 70/30 weighting. The updated announcement file is not included in the local PDF directory; the Master Guide copy retains its earlier rubric. Processor D remains a conceptual illustration only, as the Master Guide reserves D execution for Phase 2.
- [x] **Team/identity information:** README identifies MOM'S MAGIC and its three members: MATTA VIJAY, GUDE YUVA NIKESH, and BOLLA DINESH KUMAR. The team reports organizer confirmation for Phase 1.
- [x] **Local Git preparation:** repository is on branch `main`; the package is committed as `460fb808c23d1d852381c89b214432f6f8a925e0` and remote `origin` points to the team-provided GitHub repository.
- [ ] **Git author:** `user.name` and `user.email` are not configured locally or globally. Supply the team's preferred commit author name and email before committing.
- [x] **Publication:** pushed branch `main` to [the public GitHub repository](https://github.com/vijay672/Biomedical-Signal-Classifier-using-Quantum-Machine-Learning-QML-). Verified the repository's public visibility through the GitHub API and confirmed the remote branch points to commit `460fb808c23d1d852381c89b214432f6f8a925e0`.
- [ ] **Portal submission:** submit the public repository URL through the organizer portal. No portal receipt is available in this repository.
- [ ] **Final review:** inspect the provisional assumption, report, notebook output size and repository visibility before publication. No clinical or quantum-advantage claim should be added without evidence.

## Reproduce locally

From a clean checkout with Python 3.11.15, run the commands in `README.md`. With complete committed artifacts, the corrected CLI validates and loads them; if the manifest is absent it trains once and writes `results/tables/q1_provisional_*` and `results/figures/*.svg`. Open/run `main.ipynb` using the environment's Python kernel; complete artifacts are validated and displayed without retraining. If the organizer provides a different dataset correction, save it as a new experiment and preserve this provisional evidence.

## GitHub publication commands — completed 2026-10-17

The branch and remote below match the published repository. These commands were run after explicit team authorization.

```sh
git status --short --ignored
git add --all
git diff --cached --check
git diff --cached --stat
git diff --cached
git config user.name "MOM'S MAGIC"
git config user.email "226694861+vijay672@users.noreply.github.com"
git commit -m "Finalize QFF 2026 Q1 Phase 1 submission"
git remote add origin https://github.com/vijay672/Biomedical-Signal-Classifier-using-Quantum-Machine-Learning-QML-.git
git push --set-upstream origin main
```

The push and public visibility checks passed. Submit the public URL through the portal listed in the organizer instructions to complete the external submission.
