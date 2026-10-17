# QFF 2026 QML Q1

Reproducible four-qubit Qiskit classification study on a 200-row, four-feature **synthetic** benchmark. It includes a train-only preprocessing pipeline, logistic regression baseline, original and validation-selected VQCs, and Qiskit Aer comparisons on custom five- and nine-qubit processor maps. The synthetic positive class is not a clinical label. See [report.md](../../../report.md) for verified results and [main.ipynb](../../../main.ipynb) for the complete executable workflow.

## Reproduce

Use Python 3.11+ from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.baseline
python -m src.vqc
python -m src.vqc_experiments
python -m src.processor_benchmark
python -m src.result_figures
```

The bounded VQC selection evaluates exactly three variants using inner training/validation rows, then refits the selected architecture and evaluates the held-out split once. Run the original VQC before the processor benchmark; the A/B simulation loads its saved ring/parity weights. Generated JSON and SVG files go to `results/` and are ignored by Git. The notebook writes separate `results/notebook_*` files, preserving the recorded baseline artifacts; use a Jupyter-capable environment with this virtual environment as the kernel.

`src/datasets.py` also provides a configurable adapter for a user-supplied local MIT-BIH-derived feature CSV. It does not download records. No such data was used in the reported results. Processor maps and noise rates are illustrative simulations, not authoritative device specifications. Dataset and coupling-map questions are tracked in [docs/DECISIONS.md](../../../docs/DECISIONS.md); work status is in [docs/PROGRESS.md](../../../docs/PROGRESS.md).
