# QFF 2026 Phase 1 — PS-Q1 technical report

## Problem, authority and provenance

**Provisional generator assumption:** `n_redundant=0` is not organizer-confirmed. The Master Guide's literal call is invalid with scikit-learn's default redundant count. Do not present this fixture as organizer-approved until clarified.

**Team:** MOM'S MAGIC — MATTA VIJAY, GUDE YUVA NIKESH, BOLLA DINESH KUMAR. The team reports these details were confirmed for Phase 1.

The latest and highest-priority [Phase 1 Master Guide](docs/drive-download-20261013T124849Z-1-001/QFF_Phase_1_Master_Guide.pdf) defines Q1 as a four-qubit VQC on 200 synthetic samples with four features and three informative features. It specifies Processor A as 5 qubits, Processor B as 9 qubits, and expressly permits participant-designed coupling maps. This supersedes the Updated Challenge Kit's fixed B map, the older IBM Hackathon Guideline's 7-qubit B, and the Participant Instructions' general instruction to use supplied definitions. The lower-priority Challenge Kit and Problem Statement Details describe a biomedical ECG/EEG dataset instead. This report follows the Master Guide for Phase 1 and records the conflicts in [DECISIONS.md](docs/DECISIONS.md). The Master Guide's literal `make_classification` example is invalid with the installed default `n_redundant=2`; the corrected fixture explicitly sets `n_redundant=0`. That value is an engineering assumption awaiting organizer confirmation. No clinical data or hardware were used.

`src/datasets.py` generates the provisional fixture using `make_classification(n_samples=200,n_features=4,n_informative=3,n_redundant=0,random_state=42)`. A seed-42 stratified 80/20 split yields 160 training and 40 held-out rows, with 20 of each label in the test set. A `MinMaxScaler((0,π),clip=True)` is fitted on training rows only. Its state, data arrays, generator settings and split are hashed in the model artifact. Logistic regression and VQC use the same prepared rows. Label 1 is a synthetic positive-class proxy, not a clinically validated abnormal ECG diagnosis.

### Five-category evidence index

The team reports that organizers confirmed this five-category rubric and the 70% LLM / 30% human assessment split. The local Master Guide p.7 PDF has an earlier four-category rubric, and the updated announcement itself is not included in the downloaded files. The score in [the independent audit](docs/FINAL_OFFICIAL_RUBRIC_AUDIT.md) is our evidence-based self-assessment against the confirmed categories, not an organizer score.

| Task-supplied category | Maximum | Primary evidence |
|---|---:|---|
| Technical Correctness | 30 | [Exact circuit semantics test](tests/test_circuit_semantics.py); [model and split identity](results/tables/q1_provisional_manifest.json) |
| Processor A/B Benchmarking | 25 | [Processor definitions](processors/); [measured comparison](results/tables/q1_provisional_processor_benchmark.json); [routing trace](results/tables/q1_provisional_swap_diagnostic.json) |
| Performance Analysis | 20 | [Saved prediction statistics](results/tables/q1_statistical_review.json); [confusion figures](results/figures/confusion_matrices.svg) |
| Code Quality & Reproducibility | 15 | [Pinned dependencies](requirements.txt); [executed notebook](main.ipynb); [verification log](docs/FINAL_VERIFICATION.md) |
| Technical Explanation | 10 | This report; [judge summary](docs/JUDGE_SUMMARY.md); [conceptual D](docs/PROCESSOR_D_CONCEPT.md) |

## Models and protocol

The classical model is `LogisticRegression(random_state=42,max_iter=1000)`. The quantum model uses four RY feature angles, two trainable layers of per-qubit RY/RZ rotations and ring CX entanglement, parity readout, `SamplerQNN`/`NeuralNetworkClassifier`, exact `QMLSampler` inference, and COBYLA with 25 evaluations and seed 42. The architecture was fixed before held-out evaluation; no model was selected using test performance. The 25-point [convergence history](results/figures/training_convergence.svg) comes from the actual training callback and is stored with the [trained weights](results/tables/q1_provisional_vqc_model.json).

The same saved corrected-model weights are strictly validated and reused for exact ideal inference and both processor runs. The custom Processor A map is a bidirectional five-qubit line, providing a constrained routing case. The custom Processor B map is a bidirectional nine-qubit 3×3 grid, giving more placement choices. These connected graphs are valid under the Master Guide's design freedom; they are not organizer reference maps or physical calibration data. See their [configurations](processors/), [coupling figures](results/figures/processor_A_coupling.svg) and [mapping figure](results/figures/logical_to_physical_mapping.svg).

The saved-artifact [judge audit](results/tables/q1_judge_audit.json) checks hashes of the original JSON, matches the corrected dataset identity and outer split across models, recomputes six classification cases from stored predictions, validates the uniform noise/readout channels, and recompiles both mapped circuits. An exact QNN forward pass from the frozen weights reproduced every stored ideal positive-class score with zero maximum absolute error in this environment. The [circuit semantics suite](tests/test_circuit_semantics.py) checks every one of the 16 measured bitstrings and independently compares the full 16-outcome ideal distribution before and after A/B transpilation, using each compiled circuit's actual physical-to-classical measurement map. It also verifies that saved Qiskit text drawings match current deterministic recompilation. No 4096-shot noisy benchmark was rerun; sampled outputs are verified by provenance, saved configuration, and metric recalculation.

Both maps use Aer density-matrix simulation with 4096 shots, `rz/sx/x/cx` basis, level-1 optimization, explicit SABRE layout/routing, transpilation and simulation seeds 42, and a shared synthetic noise model: single-qubit depolarizing 0.001, CX depolarizing 0.005, symmetric readout flip 0.01. Ideal processor results are still finite-shot samples. [Logical](results/figures/logical_circuit.svg) and [A](results/figures/processor_A_transpiled_circuit.svg)/[B](results/figures/processor_B_transpiled_circuit.svg) compiled circuit evidence is saved.

## Measured results on the provisional fixture

| Inference | Accuracy | Precision (label 1) | Recall (label 1) | F1 (label 1) | Confusion matrix, labels `[0,1]` |
|---|---:|---:|---:|---:|---|
| Logistic regression | 0.875 | 0.895 | 0.850 | 0.872 | `[[18,2],[3,17]]` |
| VQC, exact statevector | 0.600 | 0.700 | 0.350 | 0.467 | `[[17,3],[13,7]]` |
| A, ideal finite-shot | 0.600 | 0.700 | 0.350 | 0.467 | `[[17,3],[13,7]]` |
| A, noisy finite-shot | 0.550 | 0.625 | 0.250 | 0.357 | `[[17,3],[15,5]]` |
| B, ideal finite-shot | 0.625 | 0.727 | 0.400 | 0.516 | `[[17,3],[12,8]]` |
| B, noisy finite-shot | 0.600 | 0.700 | 0.350 | 0.467 | `[[17,3],[13,7]]` |

The [clarified classification plot](results/figures/judge_model_comparison.svg) and [confusion plots](results/figures/confusion_matrices.svg) derive from saved predictions. The classical baseline is stronger than all reported quantum modes on accuracy, recall and F1. The original `classification_metrics.svg` figure remains archived as generated, but its “Abnormal recall” legend refers only to the synthetic label 1; the judge-facing figure uses that precise label.

### Statistical validity of one small synthetic split

The [statistical review JSON](results/tables/q1_statistical_review.json) independently recomputes all official and separate cases from stored predictions on the **same 40 held-out rows**. It also verifies the separate model's stored raw test hash, deterministic 128/32 inner split, fit-only scaler, validation labels, threshold rule and final test predictions. There is no evidence of test-label use in threshold selection. The following two-sided 95% Wilson intervals describe binomial variation for this one synthetic split; they do not measure retraining variation, clinical performance or broad deployment accuracy.

| Model | Specificity | False positives | False negatives | Accuracy interval (n=40) | Label-1 recall interval (n=20) |
|---|---:|---:|---:|---:|---:|
| Logistic | 0.900 | 2 | 3 | 0.739–0.945 | 0.640–0.948 |
| Official ideal VQC | 0.850 | 3 | 13 | 0.446–0.737 | 0.181–0.567 |
| Noisy A | 0.850 | 3 | 15 | 0.398–0.693 | 0.112–0.469 |
| Noisy B | 0.850 | 3 | 13 | 0.446–0.737 | 0.181–0.567 |
| Separate validation model | 0.250 | 15 | 4 | 0.375–0.671 | 0.584–0.919 |

The [recall-interval figure](results/figures/recall_uncertainty.svg) exposes the uncertainty. A two-sided exact paired McNemar test on correctness gives exploratory p=0.0034 for logistic versus official ideal VQC, and p=0.625 for noisy A versus noisy B (one A-only and three B-only correct cases). The latter is too little paired evidence to claim a classification advantage for B. A recall-restricted paired test on the 20 positive rows also gives p=0.625 for noisy A versus B. These p-values are **post hoc descriptive comparisons** after repeated inspection of one test set, with no multiple-comparison adjustment; they are not confirmatory evidence.

## Recall diagnosis and one bounded improvement probe

Observed evidence: the outer training split has 80 examples of each synthetic label, and the untouched test split has 20 of each. The official logistic model predicts 19 positives, exact VQC 10, noisy A 8 and noisy B 10. Their respective false-negative counts are 3, 13, 15 and 13, as shown in the matrices above. The official exact positive-class scores have means 0.468 for true label 0 and 0.484 for true label 1; both distributions [overlap around 0.5](results/figures/ideal_score_distribution.svg). The actual 25-evaluation training objective moves from 1.091 to 0.970, with best recorded value 0.965. All 16 trainable symbols occur independently in the circuit, but [parameter-sensitivity tests](tests/test_circuit_semantics.py) show that perturbing `w[9]`, `w[10]`, `w[11]`, `w[13]`, `w[14]` or `w[15]` by 0.37 radians leaves the exact class probabilities unchanged on the tested rows. The final directed CX ring transforms measured four-bit parity into the pre-ring parity of qubits q0 and q2. Consequently, the final-layer RY gates on q1/q3 cannot change the readout, and the final-layer RZ gates commute with it. The effective classifier therefore has fewer observable degrees of freedom than its 16 named parameters suggest. A short optimization budget, class-score overlap and these ineffective parameters are plausible contributors; this experiment does not isolate their causal effects.

Because the official model used all 160 training rows, those rows cannot be repurposed as an independent validation set for its threshold. We therefore performed **one separate**, bounded 25-evaluation training run on a stratified 128-row inner fit split; its scaler was fitted only on those 128 rows. The other 32 outer-training rows were validation only. Threshold candidates were fixed to `{0.35, 0.40, 0.45, 0.50, 0.55, 0.60}`. The rule selected maximum validation recall subject to precision ≥0.55 and F1 ≥0.60, then used F1, precision and the higher threshold as tie breakers. Threshold 0.45 achieved validation accuracy/precision/recall/F1 **0.594/0.552/1.000/0.711**, confusion `[[3,13],[0,16]]`. Only after that choice was the outer test transformed and scored once. Its measured accuracy/precision/recall/F1 were **0.525/0.516/0.800/0.627**, confusion `[[5,15],[4,16]]`, with 31 positive predictions. The test precision did not retain the validation floor. This result is stored in [separate machine-readable artifacts](results/tables/q1_recall_calibration.json) and [a separate 25-point convergence history](results/figures/recall_calibration_convergence.svg); the official full-training weights and A/B benchmarks remain unchanged. The improved recall comes with worse accuracy and many false positives, and the separate fit prevents attributing the difference to threshold adjustment alone. No held-out result was used for threshold selection.

| Circuit | Gate-only depth | Full measured depth | CX | Verified SABRE-stage SWAPs | Gate-depth overhead | CX overhead |
|---|---:|---:|---:|---:|---:|---:|
| Unconstrained reference | 20 | 21 | 8 | — | — | — |
| Processor A | 35 | 36 | 24 | 6 | +15 | +16 |
| Processor B | 20 | 21 | 8 | 0 | 0 | 0 |

Overhead is relative to the unconstrained transpilation with the same basis, level and seed. It is a **compilation-cost proxy**, because a coupling constraint can also change layout and downstream optimization. The existing benchmark JSON remains the source for final native-basis CX/depth costs. A separate transpilation-only diagnostic with Qiskit 2.5.2 counts explicit `swap` nodes in the DAG immediately after `SabreLayout`, before basis translation: A has 6 inserted SWAPs and B has 0. The logical input contains zero SWAPs, and repeated seed-42 compilation is deterministic. This count is not derived by dividing CX overhead by three; later decomposition and optimization can alter final CX cost, so six routed SWAPs need not yield exactly +18 CX. Detailed layouts, settings, counts and predictions are in the [A/B result](results/tables/q1_provisional_processor_benchmark.json); the extraction record is in the [SWAP diagnostic](results/tables/q1_provisional_swap_diagnostic.json).

### Controlled initial-layout ablation

The [ablation JSON](results/tables/q1_layout_ablation.json) holds the logical four-qubit circuit, each custom map, `rz/sx/x/cx`, optimization level 1, seed 42 and `routing_method=sabre` constant, changing only `layout_method` from official `sabre` to `trivial`. No weights, shots, noise, predictions or processor benchmark outputs were changed. Explicit SWAP nodes were counted in the `SabreLayout` or `SabreSwap` DAG callback, before native-basis decomposition. The [cost figure](results/figures/layout_ablation.svg) is generated from these measured compilations.

| Processor / initial layout | Gate-only depth | Measured depth | CX | Routing-stage SWAP | Gate-depth / CX overhead vs reference |
|---|---:|---:|---:|---:|---:|
| A / SABRE, official | 35 | 36 | 24 | 6 | +15 / +16 |
| A / trivial | 34 | 35 | 22 | 6 | +14 / +14 |
| B / SABRE, official | 20 | 21 | 8 | 0 | 0 / 0 |
| B / trivial | 26 | 27 | 14 | 2 | +6 / +6 |

The logical circuit itself has gate-only depth 13 and 8 CX; the unconstrained native-basis reference has depth 20 and 8 CX. Four logical qubits occupy 4/5 of A's physical budget and 4/9 of B's. Under the specified seed-42 SABRE baseline, B's grid embeds this ring without additional routing; the ablation shows layout choice materially changes the cost and even gives A a slightly cheaper alternate compile. This supports the narrow conclusion that **this** B configuration compiles **this** circuit more cheaply under the official settings. It does not establish universal processor superiority, better classification, or hardware performance.

A single small [secondary seed robustness experiment](results/tables/q1_seed_robustness.json) compiled the same logical circuit on the same maps under seeds 7, 42 and 99, with the `rz/sx/x/cx` basis, level 1, SABRE layout/routing held fixed. The seed-42 row matches the frozen official benchmark exactly. A had 24 CX and six routing-stage SWAPs at all three seeds, with gate depth 33, 35 and 33; B had 8 CX, zero SWAPs and depth 20 at all three. This strengthens the narrow cost observation within these three seeds. The processors differ in both qubit budget and graph geometry, so the comparison does not isolate a single topology feature; no new noisy predictions were produced for alternate seeds or layouts.

### Conceptual Processor D illustration

The [architecture figure](results/figures/architecture_problem_and_D_concept.svg) shows the actual A/B graphs and their recorded seed-42 initial logical placements. A's line lacks the interaction between its initially placed logical q3 and q0; B's grid contains all four ring links. The separate [Processor D concept](docs/PROCESSOR_D_CONCEPT.md) proposes a five-node ring with one connected spare qubit. Directly embedding the logical ring could reduce routing on that proposed graph, but no D compilation, noise simulation or classification result was run. A new coupler could add calibration and cross-talk costs. The local Master Guide reserves D for Phase 2; this is a clearly labeled supplemental explanation for the newer task-supplied assessment request.

## Historical experiment and limits

The original generator used scikit-learn's default informative-feature count; its old A/B noise rates were `0.001/0.01/0.02`. Its results are preserved, unchanged, in [historical tables](results/tables/historical/) and must not be relabeled as the corrected fixture. In that original study logistic accuracy/recall/F1 were 0.875/0.900/0.878, original VQC accuracy was 0.625, and the validation-selected VQC accuracy/recall/F1 were 0.675/0.400/0.552. Those are historical comparisons only.

The provisional dataset setting remains unconfirmed. The 40-row test sample, shallow ansatz, short optimizer budget and synthetic uniform noise limit generalization. Wilson intervals describe only this fixed test split; no retraining uncertainty estimate, patient-level split, real biomedical validation, hardware execution, physical-device calibration, or quantum advantage is claimed. Exact package versions, hashes and experiment configuration are in the [manifest](results/tables/q1_provisional_manifest.json). Reproduction instructions are in [README.md](README.md) and the executed [notebook](main.ipynb).
