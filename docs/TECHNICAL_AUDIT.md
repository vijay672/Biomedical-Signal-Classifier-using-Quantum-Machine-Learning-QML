# QFF Q1 technical and organizer-requirements audit

Date: 2026-10-13 (Asia/Kolkata). Archived read-first snapshot of the repository at that time. Later remediation superseded its implementation and result references; consult `FINAL_OFFICIAL_RUBRIC_AUDIT.md` for the current package. The audit document was the only file added during that original review.

**Conclusion:** the current synthetic experiment executes and its saved A/B results reproduce, but it is not ready to claim compliance with the supplied organizer benchmark. The highest-priority issues are the noise rates, the synthetic instance definition, and insufficient model/data validation when loading trained parameters. No synthetic preprocessing leakage or routed-circuit semantic error was found in the checks performed.

## Authority and conflicts

The user supplied the following precedence. The local filenames omit the `(1)` suffix from the first three originally named documents:

1. [Master Guide](drive-download-20261013T124849Z-1-001/QFF_Phase_1_Master_Guide.pdf), abbreviated **M** below.
2. [Updated Challenge Kit](drive-download-20261013T124849Z-1-001/Updated_Challenge_Kit.pdf), **K**.
3. [Problem Statement Details](drive-download-20261013T124849Z-1-001/Problem_Statment_Details.pdf), **D**.
4. The remaining four supplied PDFs were inspected as supporting material. They do not override M/K/D under the requested priority.

Relevant requirements pages were extracted once. M pages 2 and 5 and K pages 4 and 11 were also rendered and visually checked, including the noise table and dataset call.

| Topic | Conflict in the supplied documents | Applied interpretation / unresolved issue |
|---|---|---|
| Q1 dataset and title | M p.5 calls Q1 a four-qubit VQC using 200 synthetic rows, four features and `n_informative=3`. K p.11 and D pp.11–12 call Q1 a biomedical classifier using 200–500 ECG/EEG samples and 4–8 features. | Follow M's synthetic Phase 1 problem under the user-specified priority. Do not claim clinical validation. The exact redundant-feature setting is **unresolved**, because M's code example raises an error; see F2. |
| Coupling maps | M p.2 explicitly permits any justified valid 5-qubit/9-qubit graphs. K p.3 mandates A's chain and an eight-edge B graph. D p.2 instead mandates A's branching graph and a different twelve-edge B graph. | M overrides the fixed-map rules. The repository's custom A line and B grid are allowed; replacing them with a lower-priority reference is not required. Document their design rationale and the precedence. |
| Phase 1 scope | M p.1 reserves C and D for Phase 2. K pp.11,15–16 requests D for Q1/quantum submissions and discusses C as study-only. D p.12 mentions a Phase 2 C design and rerun. | Phase 1 is A/B only. The lack of C/D implementation is not a finding. |
| Noise | M p.2 and K p.4 agree on one-qubit 0.001, CX 0.005, symmetric readout 0.01. | These are the benchmark rates. The repository's 0.01 CX / 0.02 readout is a separate illustrative experiment, not the prescribed benchmark. |
| Compiler | M p.2 requires level 1, seed 42 and 4096 shots. K p.4 adds SABRE-based layout/routing. | Apply these compatible requirements together. Implicit compiler defaults do not establish an explicit SABRE baseline. |
| Rubric | M p.7 assigns 30 correctness, 35 A/B architecture/creativity, 20 reproducibility, 15 documentation. K pp.2–3 assigns 40/30/30 including D design. | Use M's rubric; do not add a Phase 1 D experiment to satisfy the older rubric. |
| Structure | M p.3 calls its named file/directory structure mandatory. K p.5 calls a similar structure recommended. | Treat M's structure as the submission target; see F7. |
| Deadline/submission | M p.1 specifies October 14, 2026, 23:59 IST and a public GitHub URL submitted through its portal. D p.2 specifies October 13 and permits a Space. | Use M's operational instructions. The audit did not publish or submit anything. |

## Findings, ranked by severity

P1 means fix or resolve before claiming benchmark compliance; P2 means a material correctness, reproducibility, or deliverable gap; P3 means a lower-priority implementation concern. A recommended change below is not a change performed by this audit.

### F1 — P1: recorded noisy benchmark uses the wrong prescribed rates

**Locations:** `src/processor_benchmark.py:84`, `results/processor_benchmark.json`, `report.md:46`.

`NoiseRates` specifies CX depolarizing probability 0.01 and symmetric readout flip 0.02. M p.2 and K p.4 require 0.005 and 0.01. The single-qubit value 0.001 is correct. Both processors receive the same implementation rates, so the comparison is internally controlled, but its noisy outcomes cannot stand in for the organizer benchmark.

**Suggested fix:** retain the existing outputs as an explicitly labeled alternative-noise experiment. Add a benchmark run using 0.001/0.005/0.01, with one fixed weight vector and the same dataset for both A and B. Update the main tables and figures from measured outputs; changing prose or metadata alone is insufficient.

### F2 — P1: current synthetic instance differs from M, whose example is itself invalid

**Locations:** `src/datasets.py:121`, `docs/DECISIONS.md:8`, `report.md:9`. Requirement: M p.5.

The repository omits `n_informative`, giving the installed scikit-learn default of 2. M explicitly specifies 3. However, running M's literal call raises `ValueError`: with default `n_redundant=2`, the informative/redundant feature count exceeds four. This is an organizer ambiguity, not permission to silently substitute a different instance.

**Suggested fix:** obtain clarification of `n_redundant` and explicitly record the complete generation configuration. If a provisional repair is approved, label it as an assumption. Preserve all existing data/results as the original fixture and reproduce all comparisons on a separately identified corrected instance; do not relabel current results as that instance. The 80/20 stratified split and seed 42 are otherwise implemented correctly.

### F3 — P1: trained-parameter loading does not enforce circuit or dataset identity

**Locations:** `src/processor_benchmark.py:121`, `src/processor_benchmark.py:281`, `src/vqc.py:216`.

The loader accepts any artifact with four qubits, matching parameter names and coarse dataset metadata. The benchmark then always builds the default ring/parity model. Targeted probes showed that changing saved entanglement to a line, readout to qubit zero, or layer count to seven was accepted. Changing the actual prepared features and labels while retaining metadata was also accepted. A supported variant saved through the general VQC writer can therefore be evaluated as a different circuit. The present original-model A/B results passed the replay and parameter-hash checks; this is a loader defect, not evidence that those results used mismatched weights.

**Suggested fix:** validate a versioned architecture/encoding/readout signature and weight schema, or reconstruct the circuit from that signature. Record and verify hashes of the split arrays and fitted preprocessing state. Add rejection checks for architecture and content mismatches.

### F4 — P2: required Q1 evidence is incomplete

**Locations:** `src/processor_benchmark.py:203`, `src/vqc.py:139`, `src/vqc.py:233`, `src/result_figures.py:110`, `report.md:41`, `main.ipynb`.

The A/B artifact and table expose recall but omit named accuracy and F1 metrics. M p.5 asks for accuracy and training convergence; K p.11 and D p.12 also require F1/recall. Training stores only the final objective/evaluation count, so a convergence history cannot be recovered from those artifacts. The figures contain no coupling maps, and no logical/transpiled circuit visuals or saved layouts are supplied, despite M p.7 and K p.4/p.11 requirements. All notebook code cells also have empty saved outputs; it prints SVG paths rather than displaying the figures.

**Suggested fix:** export accuracy/F1 from existing predictions without refitting or changing selection. Capture objective history during a separately identified reproducible training run. Add coupling diagrams, logical/compiled circuit views and logical-to-physical layouts; save an executed notebook with the key results. Explain shared ideal training versus processor-specific inference rather than suggesting independent A/B training occurred.

### F5 — P2: the compiler baseline is implicit and B does not use SABRE layout

**Locations:** `src/processor_benchmark.py:228`, `src/vqc_experiments.py:129`. Requirement: K p.4.

Both calls omit `layout_method` and `routing_method`. A pass trace in the installed environment showed A reaching `SabreLayout`, while B used `VF2Layout`/`ApplyLayout` without `SabreLayout`. Thus level 1 and seed 42 alone do not implement the requested SABRE-based layout/routing baseline. An explicit-SABRE probe happened to retain A's 24 CX/depth 36 and B's 8 CX/depth 21; that does not establish identical layouts or noisy sampled outputs.

**Suggested fix:** explicitly select and record the required layout/routing methods, retain default-compiler outputs as an additional experiment, and rerun the official comparison. Save initial/final layouts alongside the compiler version and settings.

### F6 — P2: gate-depth reporting includes measurement operations

**Locations:** `src/processor_benchmark.py:166`, `src/vqc_experiments.py:125`. Requirement: K p.4 excludes measurement/reset from gate-cost comparison.

`circuit.depth()` includes the terminal measurements. A targeted filter gave A gate-only depth 35 instead of reported 36, and B gate-only depth 20 instead of 21. This is a reporting-convention mismatch with the gate-cost rule; the existing values are valid full measured-circuit depths. CX counts are unaffected. The A/B depth overhead remains +15/+0 because the unconstrained baseline also includes measurement.

**Suggested fix:** export separately labeled gate-only depth and full execution-circuit depth, using the same filter for the unconstrained reference. Continue labeling CX/depth deltas as routing proxies. A CX delta must not be converted into an exact inserted SWAP count by dividing by three, because decomposition and optimization affect it.

### F7 — P2: submission packaging can omit results and does not match the required structure

**Locations:** `.gitignore:17`, `README.md:21`, `processors/processor_a.json`, `processors/processor_b.json`. Requirements: M pp.3,7; K p.6.

All JSON/SVG result files are ignored by Git, so normal repository staging will omit the evidence linked by the report. `data/raw`, `data/processed`, `results/tables`, and `results/figures` are absent; processor filenames use lowercase `a/b` rather than the prescribed `processor_A/B.json`. The expected source entry names also differ. README links to the report but contains no actual key-results table or team identification. The local checkout has no `.git`, and no public submission URL was available for verification.

**Suggested fix:** package an explicit allowlist of final result artifacts and their provenance while keeping temporary runs ignored. Match the mandatory directory/file names, using small compatibility entry points if appropriate rather than rewriting working internals. Add the selected official PS code/title, team details and key results to README. Verify public repository access and portal submission separately; neither was performed during this audit. Optional equation-image assets are not treated as mandatory.

### F8 — P2: numerical dependency versions are not fully reproducible

**Locations:** `requirements.txt:2`, `src/vqc.py:143`, `src/vqc_experiments.py:337`.

The four top-level packages are pinned, but directly used NumPy and the SciPy optimizer dependency are not. The recorded original model used NumPy 2.4.6 / SciPy 1.17.1; QML's installed constraints permit broader versions. Fresh resolution therefore does not fix the numerical/optimizer environment that produced the weights. The controlled-experiment metadata omits those versions. `pip check` passes in the current environment, so this is a clean-reproduction gap, not a current import incompatibility.

**Suggested fix:** record a validated Python version and lock the resolved numerical dependencies, including NumPy/SciPy. Include those versions in all experiment manifests. No organizer environment file was present among the supplied PDFs, so exact matching to such a file remains unverified.

### F9 — P2: current documentation still says organizer requirements are unavailable

**Locations:** `docs/DECISIONS.md:19`, `report.md:5`, `README.md:23`.

Historical statements about missing documents were accurate when written. The current report still presents the problem as a working statement and all map authority as unresolved even though M now explicitly grants design freedom. It also has no documented precedence or acknowledgement that current noise and informative-feature settings differ from M.

**Suggested fix:** retain the historical record, but add a current decision ledger citing document/page, priority, the implemented choice, and any remaining ambiguity. Distinguish resolved scope/map freedom from the unresolved dataset-call defect. Relabel existing outputs as prior-fixture/prior-noise results until compliant comparisons exist.

### F10 — P3: four optimized RZ angles are unobservable in the current readout

**Locations:** `src/vqc.py:96`, `src/vqc.py:143`.

Final-layer RZ gates `w[9]`, `w[11]`, `w[13]`, `w[15]` are followed only by CNOTs and computational-basis measurement. They change phases but not output probabilities under either supported readout. Perturbing each by 0.713 radians on 16 training inputs left probabilities unchanged within 1e-12 in all three variants. COBYLA nevertheless searches all 16 dimensions under a 25-evaluation budget. The circuit is mathematically valid, but this wastes optimization capacity; the audit does not attribute the poor recall to this alone.

**Suggested fix:** preserve the current baseline, then consider freezing/removing these angles or adding a justified measurement-basis rotation in a new controlled variant. Any model change requires a fresh training/validation protocol and clearly separate artifacts.

### F11 — P3: the unused MIT-BIH adapter has no target-column or group-leakage guard

**Locations:** `src/datasets.py:54`, `src/datasets.py:91`, `src/datasets.py:156`.

The CSV adapter accepts `label_column` inside `feature_columns`; a two-row numeric-label probe returned the target as a feature. It also discards subject/record identifiers and directs callers to a row-wise random split. If several segments originate from one patient/record, that API cannot enforce subject/record-disjoint evaluation. No real biomedical dataset was used, so neither issue contaminates the reported synthetic results. Under M's synthetic task, this is not a Phase 1 dataset-compliance blocker.

**Suggested fix:** reject target-column overlap and retain group identifiers or accept an authoritative split before using this adapter for biomedical performance claims.

## Targeted checks and observed results

- `.venv/bin/python -m unittest discover -s tests -v`: **20 passed**. The suite is small and covers imports, datasets, baseline, QNN shapes/parameters, processor configuration/routing, seeded simulation and controlled selection.
- `.venv/bin/python -m pip check`: **passed**, no broken requirements. Environment: Python 3.11.15, Qiskit 2.5.2, Aer 0.17.2, QML 0.9.1, scikit-learn 1.9.1, NumPy 2.4.6, SciPy 1.17.1.
- Additional inner-validation perturbation: adding 1e6 to validation-only raw features left the fitted scaler and inner-training features unchanged. Existing outer-test perturbation test also passed. Selection reads validation metrics only; no test-based selection was found.
- Recomputed exact logical-versus-routed probabilities over all 40 held-out rows, respecting physical-to-classical measurement mappings. Maximum absolute discrepancy was approximately **3.7e-12** on both processors. Current routing preserves model semantics.
- Replayed the complete recorded A/B benchmark into a temporary directory: **entire JSON matched**, including noisy/ideal predictions and routing metrics. The saved parameter-file SHA256 matched the actual original artifact. Both processors share one weight vector, dataset, basis, level, seeds, shots and noise model.
- Replayed selected-model predictions from its saved weights without retraining: **all predictions matched**. No new model was selected during this audit.
- Negative probes reproduced F2, F3 and F11. Compiler pass/depth probes reproduced F5/F6. Parameter-sensitivity probes reproduced F10. These cases are not covered by the current passing suite.
- Original VQC artifact SHA256 remained `cf214628f7137e6760973b60a7d48c222d337f1eac2acdeff03902c6fb79644b`; logistic artifact remained `f0b0b192a1e0a3e34bb4c7bd00a55a79249cbca286eaae7924daa5cb8ab38895`.

For illustration of F4, the following metrics were recomputed from existing saved confusion matrices, not from a new benchmark. They still describe the prior synthetic fixture and prior noise model:

| Processor/mode | Accuracy | F1 | Recall |
|---|---:|---:|---:|
| A ideal sampled | 0.650 | 0.7083 | 0.850 |
| A noisy sampled | 0.650 | 0.7083 | 0.850 |
| B ideal sampled | 0.625 | 0.6939 | 0.850 |
| B noisy sampled | 0.625 | 0.6809 | 0.800 |

The original exact-statevector VQC was also replayed from its saved weights: accuracy 0.625, precision 0.5862, F1 0.6939, recall 0.85, confusion matrix `[[8, 12], [3, 17]]`. These values were not written into the existing experiment files. The sampled A ideal result differs near the decision threshold, as expected from finite-shot execution; it is not proof of a routing error.

## Recommended repair order

1. Resolve and record M's exact synthetic generator settings; preserve the original instance/results.
2. Add model/data identity validation and an explicit official compiler/noise configuration.
3. Run the corrected benchmark once with shared trained weights and a frozen split, capturing convergence and required metrics.
4. Complete diagrams, layouts, gate-depth labels, dependency lock and submission packaging; reconcile the decision ledger against the supplied priority.

No code fixes, benchmark-setting changes, existing-report rewrites, Git initialization, publication or submission were performed. Organizer clarification remains needed for the invalid dataset call and any separately released environment or AI-disclosure policy; those missing details have not been inferred.
