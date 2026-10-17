# Decisions and open questions

The early sections below preserve the historical implementation record. The dated **Current Phase 1 decisions** and later sections supersede historical statements about absent PDFs, illustrative noise rates, and unverified SWAPs.

## Recorded decisions

- Phase 0 is setup only: no dataset processing, model training, or benchmarking.
- The intended classifier architecture has four qubits. Phase 1 processor targets are Processor A (5 qubits) and Processor B (9 qubits).
- Runtime versions are pinned in `requirements.txt` and resolved from stable package releases supporting Python 3.11.
- The requested synthetic engineering fixture uses `make_classification(n_samples=200, n_features=4, random_state=42)` and a stratified 80/20 split. This is implemented as requested, without asserting that it resolves the organizer's contested real biomedical dataset choice.
- Quantum input preprocessing uses `MinMaxScaler((0, pi), clip=True)`, fit on training rows only. Clipping keeps held-out transformed values in range without incorporating held-out statistics.
- The MIT-BIH adapter reads a configurable local CSV of numeric, pre-extracted features and maps configured source labels to binary classes. It does not download records or impose a feature schema.
- The classical reference baseline is scikit-learn `LogisticRegression(random_state=42, max_iter=1000)`, trained on the shared training partition and evaluated on the shared test partition. Label `1` is the positive class for recall/precision/F1; on the synthetic fixture this is a binary benchmark label, not a clinical interpretation.
- The initial Q1 VQC is a four-qubit RY angle encoder followed by two layers of per-qubit RY/RZ rotations and ring CNOT entanglement. `SamplerQNN` parity readout feeds a one-hot `NeuralNetworkClassifier` trained with COBYLA on the exact statevector simulator. The default evaluation budget is 25; supported budgets are bounded to 18–100 because COBYLA needs at least parameter-count + 2 evaluations for 16 weights.
- **Processor map choice for the current simulation:** no architecture specification or processor configuration was found in the repository when this benchmark was implemented. `processors/processor_a.json` and `processors/processor_b.json` therefore declare `coupling_map_source: custom`, not `reference`. A is an illustrative bidirectional five-qubit line; B is an illustrative bidirectional nine-qubit 3×3 grid. A future reference map must identify its source and be reviewed against organizer material before replacing these maps.
- Processor runs reuse one saved VQC weight vector and the same synthetic held-out data, `rz/sx/x/cx` basis, transpiler optimization level 1, seed 42, and 4,096 shots per circuit. Both receive the same illustrative noise rates: one-qubit depolarizing 0.001 on `rz/sx/x`, two-qubit depolarizing 0.01 on `cx`, and symmetric readout flip 0.02. These are project simulation assumptions, not device calibrations.
- "Ideal" processor predictions mean an Aer density-matrix run without injected noise, still sampled at 4,096 shots. Small differences between ideal predictions across maps can therefore come from finite-shot sampling near the class threshold.
- Routing overhead is reported as the CX-count and depth difference from an unconstrained four-qubit transpilation of the same circuit with the same basis, optimization level, and seed. It is a compilation proxy; the basis decomposition does not preserve an explicit SWAP count.
- The controlled VQC study fixes three candidates in advance: original ring/parity, line/parity, and ring/first-qubit readout. It selects by label-1 recall on a stratified inner validation split of the original training rows, with precision, F1, routing CX overhead, and fixed candidate order as tie-breakers. The held-out test set is used only after selection and full-training refit. The original baseline and original ring/parity processor results remain separate artifacts.

## Unresolved

- **Dataset selection:** organizer documents reportedly conflict. Select a dataset only after reviewing the authoritative organizer material and recording the rationale here.
- **Coupling-map authority:** organizer documents reportedly conflict about whether the supplied map or another source governs. Confirm with the organizer's authoritative source before treating any map as a reference or device model. The current custom maps are provisional simulations.
- **Source files:** no Q1 architecture specification, processor configuration, or organizer document was present in the repository at initialization. Reconcile any supplied files with organizer documents when available; retain originals unchanged.

## Current Phase 1 decisions (2026-10-14; supersedes historical statements above)

1. **Authority and scope.** The local PDFs are available in `docs/drive-download-20261013T124849Z-1-001/`. Apply Master Guide > Updated Challenge Kit > Problem Statement Details > supporting PDFs. Master Guide pp.2,5 governs Phase 1: PS-Q1 four-qubit VQC and Processor A/B only. Its synthetic dataset supersedes lower-priority biomedical descriptions for this Phase 1 package. The optional MIT-BIH adapter is not used for results.
2. **Provisional dataset.** Master Guide p.5 specifies 200 samples, four features, three informative features, seed 42, but its literal call fails with scikit-learn's default two redundant features. `n_redundant=0` is a *project engineering assumption*, not organizer-confirmed. The new fixture is separately identified and hashed; the original fixture and outcomes remain in `results/tables/historical/`. Organizer clarification remains open.
3. **Processor maps.** Master Guide p.2 permits justified custom 5/9-qubit graphs. A is a connected bidirectional five-qubit line to expose routing cost; B is a connected bidirectional nine-qubit 3×3 grid with additional placement choices. These are `custom`, not reference or measured hardware maps. This higher-priority permission resolves the conflicting lower-priority fixed-map descriptions for this Phase 1 implementation.
4. **Official simulation.** New benchmark default uses one-qubit depolarizing 0.001, CX depolarizing 0.005, symmetric readout flip 0.01, 4096 shots, level-1 transpilation, seed 42, explicit SABRE layout/routing and common `rz/sx/x/cx` basis. Prior 0.001/0.01/0.02 outputs are preserved as an *alternative-noise historical experiment*, not official results.
5. **Model and metrics.** The provisional run trains one fixed ring/parity model on the corrected training rows, checks its versioned architecture and dataset/preprocessing identity, then uses those saved weights for exact inference and both Aer processors. No corrected-test-set variant selection occurred. Gate-only and full measured depths are separate; CX/depth deltas are routing cost proxies, not verified inserted SWAP counts. Label 1 is the positive synthetic proxy.
6. **Remaining ambiguities.** The exact redundant-feature value needs organizer confirmation; a future biomedical data requirement, if separately imposed, would require sourced records, feature provenance and a subject/record-disjoint protocol. Any later organizer environment file or AI disclosure rule has not been supplied/verified. Team identity is now recorded; public URL and submission state depend on the submitter.

## Final organizer discrepancy resolution (2026-10-14)

- The latest Master Guide is authoritative for this submission. Its pp.1–2 limit Phase 1 to A/B, set A=5 qubits and B=9 qubits, call the published topologies reference-only, and permit any justified valid graph. This overrides the Updated Challenge Kit's fixed nine-qubit B map, the older IBM Hackathon Guideline's seven-qubit B, and the Participant Instructions' general wording about exact supplied definitions.
- `processors/processor_B.json` is a participant-designed nine-node bidirectional 3×3 grid. It uses nodes 0–8, has 12 undirected edges represented in both directions, contains no self-loops or duplicate directed edges, includes every node, and is connected. It is not organizer-provided.
- A deterministic Qiskit 2.5.2 transpilation-only trace records the DAG immediately after `SabreLayout`, before native-basis decomposition. The logical circuit has zero SWAPs; the trace has 6 explicit inserted SWAPs for A and 0 for B. Final measured routing costs remain +16 CX/+15 gate-only depth for A and zero/zero for B. The trace-stage count is recorded separately in `results/tables/q1_provisional_swap_diagnostic.json`; the benchmark output was not overwritten.

## Final competitive analysis (2026-10-14)

- The official full-training ring/parity model and A/B benchmark remain frozen and SHA256-identified. A separate 128-fit/32-validation, 25-evaluation VQC run was permitted solely to test a predeclared, validation-constrained threshold rule. Its threshold and held-out result are stored under `q1_recall_calibration*` and never replace official metrics. No validation set was extracted from rows already used to fit the official model.
- The official exact QNN scores replay exactly from saved weights. Four final-layer RZ perturbations leave scores unchanged, showing parameter redundancy for this computational-basis readout. The observed low recall, overlapping scores and short convergence trace are reported as evidence; their individual causal contributions remain unisolated.
- The one controlled architecture ablation changes initial layout from `sabre` to `trivial`, with SABRE routing and all other compilation settings fixed. It is a transpilation-only cost study, not a new noisy inference run. Processor B's seed-42 cost advantage is circuit/layout-specific.

## Independent judge review (2026-10-14)

- Report Wilson intervals for accuracy/recall/specificity and exact paired tests only as exploratory evidence on this single inspected synthetic split. No held-out model or threshold selection is performed. Noisy A/B classification differences are not established as a statistically persuasive advantage.
- Add one secondary compilation-only robustness check with seeds 7, 42 and 99. The official seed-42 row remains frozen. This check does not retrain, resimulate noise, or change processor maps.
- Present “synthetic label-1 recall” in newly generated judge-facing figures and future figure generation. Preserve the earlier SVG with its historical “Abnormal recall” wording; that wording never denotes a verified clinical diagnosis.

## Five-category review and D concept (2026-10-16)

- The five-category weights and 70% LLM/30% human assessment split came from the review request. The linked Drive folder was not independently readable through the available web tool, and the local highest-priority Master Guide still has a four-category rubric. Use the supplied categories only as an explicitly provisional evidence index/self-assessment until the organizer's dated announcement is obtained.
- The Master Guide reserves Processor D execution for Phase 2. The supplementary D drawing is a **hypothetical five-qubit ring-compatible graph**, not an official configuration, compiled result or Phase 1 benchmark. Only A and B are evaluated.
- A complete 16-outcome semantic test shows the last directed CX ring maps final odd parity to the pre-ring `q0 XOR q2` marginal. Consequently six second-layer angles (`w9,w10,w11,w13,w14,w15`) are output-inert at this readout. This extends the earlier four-RZ observation; keep the frozen official circuit and weights unchanged and disclose the limitation.
