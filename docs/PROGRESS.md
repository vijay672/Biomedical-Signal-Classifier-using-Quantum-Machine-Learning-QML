# Progress

## Phase 0

- Created the repository scaffold and pinned dependency manifest.
- Added an import smoke test for Qiskit, Qiskit Aer, scikit-learn, and Qiskit Machine Learning.
- No model training, dataset generation, or expensive simulation was started.
- Source inspection outcome: the repository was empty; no architecture or processor configuration files were available to inspect or preserve.
- Initial smoke-test result in the host interpreter: blocked because Qiskit was not installed (`ModuleNotFoundError: No module named 'qiskit'`). The declared dependencies were later installed into the ignored project `.venv` for QNN implementation and verification.
- Dependency resolver check: passed with `python3 -m pip install --dry-run --ignore-installed -r requirements.txt`; the pinned set resolves for the current Python 3.11 environment.

## Dataset and preprocessing implementation

- Added the reproducible 200-sample/four-feature `make_classification(random_state=42)` fixture with stratified 80/20 split.
- Added training-only MinMax scaling to quantum input range [0, pi], with clipping for held-out points outside training bounds.
- Added a configurable MIT-BIH-derived local feature CSV adapter. It requires a user-provided file and does not download data.
- Dataset tests: passed (5 tests), including determinism, stratified labels, shapes, range, train-only fit/leakage, configurable CSV mapping, and missing-file handling.
- At Phase 0, full unittest discovery passed all 5 dataset tests; the environment smoke test was blocked because Qiskit was absent from the host interpreter.
- No quantum model was trained.

## Logistic regression baseline

- Added deterministic `LogisticRegression(random_state=42, max_iter=1000)` evaluation using the existing prepared train and held-out test arrays. No re-splitting or independent test preprocessing is done. Output captures generator, split, and model seeds/settings.
- Saved metrics and run metadata to `results/logistic_regression_baseline.json`.
- Synthetic held-out results (40 examples): accuracy 0.875; abnormal-class recall 0.900; precision 0.8571; F1 0.8780; confusion matrix `[[17, 3], [2, 18]]` in label order `[0, 1]`.
- Baseline tests: passed (2 tests). Existing dataset tests: passed (5 tests).
- Result records scikit-learn 1.9.0, which is the version present in this environment; `requirements.txt` pins 1.9.1 for clean reproductions.
- No VQC was implemented or trained during this stage.

## Four-qubit VQC

- Implemented four RY input angles, two trainable RY/RZ rotation layers, and ring CNOT entanglers with parity readout through Qiskit Machine Learning `SamplerQNN` and `NeuralNetworkClassifier`.
- Trained with exact statevector `QMLSampler` and deterministic initialization on the existing 160-row training partition. COBYLA used its default capped budget of 25 evaluations; held-out predictions use the existing 40-row test partition.
- Saved 16 optimized angles and run metadata to `results/vqc_trained_parameters.json`.
- Initial held-out synthetic accuracy: 0.625 (40 rows); this is a model smoke result, not a processor comparison.
- Focused VQC tests passed (4 tests); full unittest discovery passed (12 tests) in `.venv`.
- No processor benchmarking was run.

## Simulated Processor A/B comparison

- No architecture specification, reference coupling map, or processor configuration file was present to inspect. Added explicitly custom map files for a bidirectional five-qubit line (A) and bidirectional nine-qubit 3×3 grid (B); reference-map authority remains unresolved in `docs/DECISIONS.md`.
- Reused the saved 16 VQC parameters and the same 40-row held-out synthetic partition. Aer ideal and noisy runs shared `rz/sx/x/cx` basis, optimization level 1, seed 42, 4,096 shots, and one uniform synthetic noise model. No retraining occurred.
- Measured unconstrained compile baseline: 8 CX, depth 21. Processor A: 24 CX, depth 36, routing proxy +16 CX/+15 depth, ideal/noisy recall 0.85/0.85. Processor B: 8 CX, depth 21, routing proxy +0 CX/+0 depth, ideal/noisy recall 0.85/0.80. The full predictions, noise settings, map provenance, and result hashes are saved in `results/processor_benchmark.json`.
- Processor validation tests passed (4 tests), covering explicit map provenance, parameter/dataset matching, topology-respecting transpilation, and deterministic seeded outputs. Full unittest discovery passed (16 tests) in `.venv`.

## Controlled VQC experiment and final artifacts

- Compared exactly three fixed VQC variants on a stratified 128/32 split inside the original 160 training rows. An inner scaler was fitted only on the 128 fit rows. All candidates used seed 42, exact QMLSampler statevector probabilities, and at most 25 COBYLA evaluations. The original baseline artifacts were preserved byte-for-byte.
- Validation label-1 recall/precision/F1: original ring/parity 0.875/0.824/0.848; line/parity 0.313/0.500/0.385; ring/first-qubit readout 0.938/0.682/0.789. The latter was selected by validation recall. Its full-training refit yielded held-out accuracy 0.675, label-1 recall 0.400, precision 0.889, F1 0.552, confusion `[[19, 1], [12, 8]]`. This validation-to-test drop is reported, not hidden.
- Saved full configurations, validation predictions, compilation costs, selected weights, and final held-out metrics to `results/vqc_controlled_experiments.json` and `results/vqc_selected_parameters.json`. The selected model was **not** used in the existing Processor A/B comparison; those Aer results continue to use the original ring/parity weights.
- Added `main.ipynb`, updated `README.md`, created `report.md`, and generated three SVG figures from JSON only. The notebook writes separate `results/notebook_*` outputs and completed all code cells successfully. Notebook metric summaries matched the recorded baseline, controlled VQC, and processor artifacts.
- Focused controlled-experiment/figure tests passed (4). Full unittest discovery passed (20 tests). Notebook JSON syntax passed. Original VQC and logistic baseline SHA256 hashes remained `cf214628f7137e6760973b60a7d48c222d337f1eac2acdeff03902c6fb79644b` and `f0b0b192a1e0a3e34bb4c7bd00a55a79249cbca286eaae7924daa5cb8ab38895` respectively.

## Next task after final report

Obtain authoritative organizer Q1 documents and any processor reference maps, then reconcile the unresolved dataset and coupling-map choices. If a real biomedical feature dataset is supplied, define its provenance and split protocol before running any new model selection or processor claim.

## 2026-10-14 — final compliance remediation

- Verified the supplied PDFs at `docs/drive-download-20261013T124849Z-1-001/` using Master Guide priority. Retained original outputs; copied historical JSON, figures, notebook and former report to `results/tables/historical/` and `results/figures/historical/` for package inclusion.
- Added a separately labeled provisional 200×4 fixture with three informative and zero redundant features, seed 42. The zero-redundant setting is an unconfirmed engineering assumption. Both models use one 160/40 stratified split, with scaler fit on training data only.
- Added strict versioned model/data/preprocessing signatures. Negative tests reject changed architecture, raw data, scaler and software version. Added explicit prescribed noise, SABRE compilation, layouts, gate-only/full depth, metrics and figures.
- Ran one corrected-fixture COBYLA training (25 recorded objective evaluations). Held-out label-1 accuracy/recall/F1: logistic 0.875/0.850/0.872; exact ideal VQC 0.600/0.350/0.467; noisy A 0.550/0.250/0.357; noisy B 0.600/0.350/0.467. These are not the historical experiment. A/B gate-only depths are 35/20, full depths 36/21, CX 24/8, gate-depth routing overhead +15/0 and CX overhead +16/0.
- Saved JSON manifests, strict trained weights, actual convergence trace, predictions, logical and transpiled circuits, coupling/mapping diagrams, confusion matrices and comparison figures under `results/tables/` and `results/figures/`. Updated README, report, executed notebook, dependencies and Git allowlist.
- Focused pre-run suite: 23/23 passed; prescribed-settings smoke passed. Final suite, notebook execution, pip check, fresh-install and staging-rule verification outcomes are recorded in `docs/SUBMISSION_CHECKLIST.md` after completion.

## Next action

Obtain organizer confirmation of `n_redundant=0`; the submitter must initialize a Git repository, supply team details, publish a public repository and perform portal submission only after review.

## 2026-10-14 — final organizer discrepancy check

- Confirmed that the latest Master Guide overrides the conflicting fixed-map and older seven-qubit Processor B descriptions: Phase 1 uses A=5, B=9, with participant-designed valid coupling maps.
- Validated `processor_B.json` as a participant-designed connected bidirectional nine-node 3×3 grid. No topology or benchmark output was changed.
- Verified accuracy, label-1 recall and F1 in README, report, executed notebook outputs, and final JSON for logistic regression, exact ideal VQC, and noisy A/B. Clarified that synthetic label 1 is not a clinically validated abnormal ECG diagnosis.
- Ran a deterministic transpilation-only Qiskit 2.5.2 callback diagnostic. The DAG immediately after `SabreLayout` contains 6 inserted SWAPs for A and 0 for B; the logical input contains none. Final native-basis routing overhead remains +16 CX/+15 gate-only depth for A and 0/0 for B. Saved the method and settings separately in `results/tables/q1_provisional_swap_diagnostic.json` without overwriting benchmark results.
- Focused tests passed 7/7. Additional consistency checks passed for graph validity, repeatable trace counts, final routing overhead, required metrics, executed notebook schema/output, and label wording. The corrected model SHA256 remains `9902dd5dce0d069633081b7787618f98797fc1442c5ceff5ac3679067c8332fd`; historical model/baseline hashes also remain unchanged.

## 2026-10-14 — competitive upgrade and final hardening

- Read-first audit recalculated six saved classification cases, verified original artifact hashes, dataset/split/scaler identity, prescribed noise and seed settings, coupling-map-respecting recompilation, and exact ideal QNN scores. Official files and weights remain unchanged.
- Diagnosed low official VQC recall from confusion matrices, positive prediction counts, score overlap, actual 25-point convergence and four experimentally verified ineffective final-layer RZ parameters. Evidence and causal limits are in `report.md` and `results/tables/q1_judge_audit.json`.
- Ran exactly one **separate** bounded 25-evaluation inner-fit VQC: 128 training-only fit rows, 32 stratified validation rows, inner scaler fit on 128 only. Validation-selected threshold 0.45 achieved held-out recall 0.800, precision 0.516, F1 0.627, accuracy 0.525; the official model and benchmark were not changed. No score improvement claim is made without its accuracy/precision tradeoff.
- Recorded a deterministic, transpilation-only initial-layout ablation. Official A/B SABRE gate depths/CX are 35/24 and 20/8; trivial-layout variants are 34/22 and 26/14. SWAP callback evidence and settings are saved separately. Added four measured-data SVGs, refreshed the executed notebook, README and technical report.
- Added five focused tests; final full suite **29/29 passed**. `pip check` passed; fresh-process official CLI loaded saved artifacts; notebook six code cells executed with visible outputs; all JSON parsed and Git dry-run included final artifacts. See `docs/FINAL_VERIFICATION.md`.
- Staged the reviewed package locally (no commit or remote), then replayed the official CLI, diagnostics, separate artifact loader and figure generation from an isolated export of exactly the staged files. All passed and the official and separate artifact hashes remained stable.

## Next action

Confirm the provisional `n_redundant=0` generator correction with organizers if possible. The team should review the package, create a public GitHub repository, stage/commit/push only with explicit approval, and submit the public URL by the announced deadline.

## 2026-10-14 — independent judge review

- Re-read the staged source, tests, notebook, tables/figures, processor JSONs and organizer PDFs before edits. Master Guide p.5 generator ambiguity remains unresolved; latest p.2 permits participant-defined 5/9-qubit maps.
- Added a saved-data statistical review for all seven corrected-fixture cases: specificity, false positives/negatives, 95% Wilson intervals for accuracy/recall/specificity, and five exploratory exact paired contrasts. Verified identical outer test identity, deterministic inner split, fit-only scaler and validation-only threshold selection.
- Ran one additional bounded compilation-only seed check at 7, 42 and 99. A stayed at 24 CX/6 routing-stage SWAPs (gate depth 33–35); B stayed at 8 CX/0 SWAPs (depth 20). Seed 42 official costs remain unchanged.
- Added a clarified official-model comparison and recall-interval figure, moved the notebook's main comparison to synthetic-label wording, and documented that the separate recall gain sacrifices specificity and accuracy. The historical alternative-noise CLI now declares itself historical.
- Added six focused tests; full suite **35/35 passed**, dependency check passed and seven notebook code cells executed with visible outputs. Remaining checks and release decision are in `docs/FINAL_JUDGE_AUDIT.md` and `docs/FINAL_VERIFICATION.md`.

## 2026-10-15 — final package closeout

- Exported exactly the 105 staged files to an isolated temporary directory and replayed the official loader, saved-data diagnostics, statistical review, compilation seed check, separate model loader and figure generation. All exited 0; frozen official and separate artifact hashes matched.
- Re-ran 35/35 tests, `pip check`, all seven notebook code cells, JSON/SVG validation, processor configuration tests, README/report/notebook metric comparisons, Git whitespace lint and an all-staged-file token-pattern scan. No test or replay failure remains. The first notebook-value check inspected the wrong output MIME type; a corrected Markdown-output check passed.
- The client date is now October 15, after the stated October 14 23:59 IST deadline. Technical packaging is complete, but submission eligibility requires organizer confirmation of an extension/late entry and the provisional dataset correction. No commit, remote, push or portal submission was performed.
- Final publication prerequisite check found no configured Git author name/email. The team must supply its preferred identity and GitHub repository URL; both are intentionally absent from the local checkout.

## 2026-10-16 — five-category review and conceptual D

- Added exact circuit/measurement tests: four input RY gates; two RY/RZ-plus-ring layers; 16 unique weights; eight logical CX; all 16 odd-parity outputs; and complete ideal outcome equivalence after A/B routing and measurement mapping. The tests also confirm six final-layer angles are output-inert under this parity readout, a disclosed model limitation rather than a change to the official circuit.
- Fixed a future-use MIT-BIH CSV adapter leakage route by rejecting a label column selected as a feature. No biomedical data or official synthetic results changed.
- Added a provenance-backed A/B topology figure and a separately marked, uncompiled five-qubit ring-compatible D concept. The local Master Guide reserves D execution for Phase 2; the newer five-category announcement remains independently unverified from the shared Drive link.
- Updated README, report, judge summary and executed notebook with the five task-supplied judging categories, explicit circuit limitation, D caveat and evidence links. Added the [rubric audit](FINAL_OFFICIAL_RUBRIC_AUDIT.md) and refreshed [verification evidence](FINAL_VERIFICATION.md).
- Full suite: **42/42 passed**; `pip check` passed; all eight notebook code cells executed with outputs. Staged-export replay ran seven CLIs plus notebook without retraining; official model, A/B benchmark and manifest SHA256 identities remained unchanged. All seven saved classification cases were independently recalculated; 22 JSON and 20 SVG files parsed.
- Remaining tasks are organizer clarification of `n_redundant=0`, access to the newer rubric announcement, and confirmation that submission is permitted after the October 14 deadline. No commit, remote, push or portal submission occurred.
