# Final verification — 2026-10-14 IST

## Commands and outcomes

| Check | Observed outcome |
|---|---|
| `.venv/bin/python -m unittest discover -s tests -v` | 35/35 passed: 29 pre-review tests and six new focused statistics/robustness/presentation tests. |
| `.venv/bin/python -m pip check` | `No broken requirements found.` |
| `.venv/bin/python -m src.q1_submission` in a fresh process | Loaded and identity-checked saved official outputs; no model retraining or noisy resimulation. |
| `.venv/bin/python -m src.judge_diagnostics` | Verified six saved classification cases; recompiled four deterministic layout cases. |
| `.venv/bin/python -m src.judge_statistics` | Verified seven saved classification cases, identical outer test identity, validation-only threshold, 95% Wilson intervals and five exact paired contrasts. |
| `.venv/bin/python -m src.judge_robustness` | Recorded six deterministic transpilation rows at seeds 7/42/99; seed 42 reproduced official costs/layout. No Aer inference. |
| `.venv/bin/python -m src.recall_calibration` | One separate 25-evaluation fit was performed once; subsequent invocation loads and checks its saved model/data SHA256. |
| `.venv/bin/python -m src.judge_figures` | Rendered six judge-facing SVGs from saved measured JSON, including clarified label and recall-interval figures. |
| Fresh-process replay from a `git checkout-index --prefix=… -a` export of staged files | `src.q1_submission`, `src.judge_diagnostics`, `src.judge_statistics`, `src.judge_robustness`, `src.recall_calibration` and `src.judge_figures` all exited 0 without ignored local files, retraining or noisy resimulation. Official model, benchmark and separate model SHA256 values matched the source package. |
| `NotebookClient` execution of `main.ipynb` | Seven code cells executed; all seven have visible output; schema validates. Complete official artifacts loaded without retraining. |
| JSON/package review | All 22 staged `results/tables/**/*.json` parsed; all 19 staged SVGs parsed (16 current, 3 historical); required Master Guide p.3 files/directories exist; both processor JSONs validate through tests. The staged package has 105 files; the largest is `main.ipynb` at 1,537,101 bytes. No staged cache, raw dataset, bytecode or temporary run was found. |
| Exact published-value comparison | All six official rows in `README.md` and `report.md` matched saved JSON for accuracy, precision, recall, F1 and confusion matrix to the displayed three decimals. The notebook's rendered Markdown outputs contain all six cases' saved accuracy, recall and F1 values. |

The [audit JSON](../results/tables/q1_judge_audit.json) records SHA256 checksums of seven unchanged original artifacts. Key identities: official model `9902dd5dce0d069633081b7787618f98797fc1442c5ceff5ac3679067c8332fd`; official processor benchmark `3c6bd53295d4c22adfafb42a285240890c5df411a4bcb9fa4040fb0dfd3dc908`; official manifest `fc03ac8785151980dc057b02408a6a85a21a7130a956950d9271922d7a4dbace`; separate recall model `aa2f0f853f33ec139fbe855b9019b0dbaf7a0496d33408dbde2195122805289c`. The exact ideal QNN score replay error is zero in this environment. The validation-selected threshold and its test metrics are recalculated in focused tests. Official and separate model artifacts remain distinct.

`pdftotext` was unavailable (`zsh: command not found`) during this pass. System Python `pypdf 6.19.0` extracted each of the seven organizer PDFs once for a read-only priority/requirements check; no dependency was added. The latest Master Guide p.1 gives the 23:59 IST deadline; pp.2,5,7 specify participant-designed 5/9-qubit A/B maps, the Q1 synthetic fixture, and the judging rubric. The Updated Challenge Kit's biomedical description is lower priority and remains an explicit discrepancy.

## Git/package review

`git add --all` staged `README.md`, `report.md`, executed `main.ipynb`, all source/tests/configurations, organizer PDFs, original and new final JSON, text circuits, and original/new SVG figures. `git diff --cached --check` exited 0. `.gitattributes` exempts generated Matplotlib SVGs and Qiskit text circuits from whitespace lint while preserving their measured bytes. `.gitignore` excludes `.venv`, caches, local raw/processed datasets, temporary root results and local repeat runs. No staged file exceeds 5 MB. A byte-pattern scan across every staged file for private-key blocks, common GitHub/OpenAI/AWS token forms and credential assignments found zero matches; this pattern search cannot rule out every possible secret. The Git repository is local, has no commit, remote, configured `user.name` or configured `user.email`; nothing has been pushed or submitted. A fresh isolated install of the pinned dependencies was checked during the earlier remediation; this final pass checked the current environment rather than repeating that installation.

The initial `pdftotext` read attempt failed because that command is absent; read-only extraction with existing `pypdf` succeeded. An initial ad hoc notebook-value assertion checked only `stream` output and failed to find the rendered Markdown table; the corrected check read `display_data['text/markdown']` and passed. No automated tests, notebook cells, staged-export commands or JSON/SVG validations failed in this final pass. Figure/table consistency is covered by `test_saved_figures_expose_measured_labels`, `test_figures_reflect_recorded_metrics`, `test_saved_statistics_and_paired_comparisons`, and the saved JSON/SVG parse check. The provisional generator parameter still requires organizer confirmation, and publication remains a manual step.

## Submission conditions

At the time of the October 15 verification, the `n_redundant=0` correction remained an unconfirmed engineering assumption, and the only locally available deadline was October 14. The team has since reported that organizers confirmed October 17 as the final submission day and confirmed the updated rubric. The dataset generator assumption remains unconfirmed. Publication still requires the team's GitHub URL and a commit/push; no clinical or quantum-advantage claim is made.

## 2026-10-16 — five-category circuit and package review

This is a new verification pass; the October 14–15 record above is retained as historical evidence. The installed interpreter is `.venv/bin/python` (Python 3.11.15), with Qiskit 2.5.2, Aer 0.17.2, Qiskit Machine Learning 0.9.1, NumPy 2.4.6, SciPy 1.17.1 and scikit-learn 1.9.1. A bare host `python` is a different interpreter and cannot import Qiskit; invoking tests with that interpreter failed with `ModuleNotFoundError`. All project checks below used the `.venv` interpreter.

| Executed check | Actual result |
|---|---|
| `.venv/bin/python -m unittest tests.test_circuit_semantics tests.test_datasets -v` | 11/11 passed: exact gate order, 16 parity classes, complete mapped 16-outcome distribution, six output-inert weights, saved Qiskit drawing match and dataset/adapter checks. |
| `.venv/bin/python -m unittest discover -s tests -v` | **42/42 passed** in 2.056 s; this includes the new conceptual-figure test and the preceding 35 tests. |
| `.venv/bin/python -m pip check` | `No broken requirements found.` |
| `.venv/bin/python -m src.q1_submission` | Exit 0; loaded and identity-checked saved official artifacts without training or Aer resimulation. |
| `.venv/bin/python -m src.judge_diagnostics`, `src.judge_statistics`, `src.judge_robustness`, `src.recall_calibration` | Each exited 0; verified six/seven saved classification cases, five paired contrasts, four layout cases, six compiler-seed rows and the separate saved threshold model. No new model fit was performed. |
| `NotebookClient` execution of `main.ipynb` | Exit 0, schema valid; 15 cells, eight code cells, all eight with visible output. Rendered output includes the official metric rows and the clearly hypothetical D panel. |
| `git checkout-index --prefix=<temporary directory>/ -a` staged-file export | From that isolated export, `src.q1_submission`, `src.judge_diagnostics`, `src.judge_statistics`, `src.judge_robustness`, `src.recall_calibration`, `src.judge_figures` and `src.architecture_concept` each exited 0. The exported notebook also executed all eight code cells with output. Four frozen artifact hashes matched the source package. |
| Artifact/package script | Official model, A/B benchmark, manifest and separate model SHA256 values matched the pre-edit identities; two processor JSONs parsed and validated as 5/9 qubits; all seven cases' metrics were independently recalculated from the source predictions; 22 JSON files and 20 SVG figures parsed; notebook schema and visible outputs passed. |
| Staged-file and common-token scan | 111 staged files at this point; largest `main.ipynb` was 1,616,978 bytes. No `.venv`, cache, bytecode, raw dataset or processed dataset was staged; required `data/raw/.gitkeep` and `data/processed/.gitkeep` were allowed. A byte-pattern scan found zero private-key blocks or common GitHub/OpenAI/AWS token forms. This is a limited pattern scan, not proof that no secret exists. |
| README/report/notebook value check | The official accuracy/label-1 recall/F1 strings for logistic, ideal VQC, noisy A and noisy B, plus the provisional `n_redundant=0` statement, were found; notebook's executed output displays those cases. [Saved-data tests](../tests/test_judge_statistics.py) provide the numeric verification. |
| Local Markdown link check | All local links in README, report, judge summary, D concept, rubric audit, verification record and submission checklist resolved. `git diff --cached --check` exited 0 after staging. |

Two ad hoc **verification-script** mistakes were corrected and rerun: the first tried to read predictions from the statistical summary rather than the source result JSON; the next excluded the mandatory `data/*/.gitkeep` placeholders as if they were raw data. Both corrected checks passed; neither indicated an experiment defect. Full tests, notebook execution and staged-export replay had no failures. A fresh dependency installation was **not repeated** in this pass; the earlier remediation's isolated-install result and current `pip check` are the available evidence.

At that October 16 pass, the five-category announcement and deadline change were not yet reported. The team has since confirmed both; this dated record is superseded by the October 17 freeze entry below. The updated announcement file is still absent locally, and the Master Guide's `n_redundant=0` ambiguity remains open. No commit, remote, push or portal submission occurred.

## 2026-10-17 — final Phase 1 freeze

- `.venv/bin/python -m unittest discover -s tests -v`: **42/42 passed** (2.053 s). `.venv/bin/python -m pip check`: `No broken requirements found.` No model training or noisy benchmark simulation was launched.
- Independently recalculated the seven saved classification cases; all metrics and confusion matrices match the statistics table. Confirmed A/B use the same frozen model hash, configured compiler/noise/shots, A=24 CX/depth35/+16 CX/+15 depth/six routing-stage SWAPs, B=8 CX/depth20/zero overhead/no routing-stage SWAPs.
- Confirmed all five organizer-confirmed categories are indexed in README, report, notebook and judge summary. The primary documents identify all six inactive parameters (`w[9]`, `w[10]`, `w[11]`, `w[13]`, `w[14]`, `w[15]`) with test evidence. The notebook remains valid with all eight code-cell outputs present.
- All 200 relative Markdown targets across the package resolve; no absolute local paths remain. The archived October 13 audit was labeled historical and machine-specific links removed. Staged scan: 111 files; no cache, local dataset, common credential pattern, or non-placeholder data file. Required organizer directories/configurations, code, report, notebook, results JSON and figures are tracked. `git diff --cached --check` passed.
- Git state: branch `main`; all package files staged as additions; no unstaged or untracked files; no commit or remote configured. No publication or submission occurred. The team reports October 17 as the final deadline; public push requires the GitHub URL and a first commit, to be performed promptly by the team after approval.
