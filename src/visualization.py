"""Submission figures generated from measured Q1 outputs and compiled circuits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit

from src.processors import ProcessorSpec


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def coupling_figure(spec: ProcessorSpec, layout: dict[str, Any], path: Path) -> None:
    if spec.num_qubits == 5:
        positions = {i: (i, 0) for i in range(5)}
    else:
        positions = {i: (i % 3, -(i // 3)) for i in range(9)}
    fig, ax = plt.subplots(figsize=(7, 4))
    for start, end in sorted({tuple(sorted(edge)) for edge in spec.coupling_edges}):
        a, b = positions[start], positions[end]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#73899a", lw=2, zorder=1)
    initial = layout["logical_to_initial_physical"]
    for physical, (x, y) in positions.items():
        label = f"p{physical}"
        if physical in initial:
            label += f"\nq{initial.index(physical)}"
        ax.scatter([x], [y], s=1700, c="#beded5" if physical in initial else "#e6edf0",
                   edgecolors="#24465a", zorder=2)
        ax.text(x, y, label, ha="center", va="center", fontsize=10, zorder=3)
    ax.set_title(f"{spec.name}: custom {spec.num_qubits}-qubit coupling map\nInitial logical placement is shown inside nodes")
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, path)


def circuit_figure(circuit: QuantumCircuit, title: str, path: Path) -> None:
    drawing = str(circuit.draw(output="text", fold=105))
    rows = drawing.splitlines()
    fig, ax = plt.subplots(figsize=(15, max(3, 0.14 * len(rows) + 0.8)))
    ax.text(0.01, 0.99, drawing, va="top", ha="left", family="monospace", fontsize=7,
            transform=ax.transAxes)
    ax.set_title(title, loc="left", fontsize=12)
    ax.axis("off")
    _save(fig, path)


def mapping_figure(processors: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for processor in processors:
        layout = processor["layout"]
        for logical, (initial, final) in enumerate(zip(
            layout["logical_to_initial_physical"],
            layout["logical_to_final_physical"], strict=True
        )):
            rows.append([processor["name"], f"q{logical}", f"p{initial}", f"p{final}"])
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(rows) + 1.4))
    ax.table(cellText=rows, colLabels=["Processor", "Logical", "Initial physical", "Final physical"],
             loc="center", cellLoc="center")
    ax.set_title("Recorded SABRE logical-to-physical placement")
    ax.axis("off")
    _save(fig, path)


def results_figures(
    baseline: dict[str, Any], ideal: dict[str, Any], benchmark: dict[str, Any],
    convergence: list[dict[str, float | int]], output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    rows = [
        ("Logistic", baseline["metrics"]),
        ("Ideal VQC", ideal["metrics"]),
    ]
    for processor in benchmark["processors"]:
        rows.append((processor["name"] + " ideal", processor["ideal"]["metrics"]))
        rows.append((processor["name"] + " noisy", processor["noisy"]["metrics"]))
    names, metrics = zip(*rows)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(names))
    for offset, (key, title) in enumerate((
        ("accuracy", "Accuracy"), ("recall_label_1", "Synthetic label-1 recall"),
        ("precision_label_1", "Precision"), ("f1_label_1", "F1")
    )):
        ax.bar(x + (offset - 1.5) * 0.19, [row[key] for row in metrics], width=0.18, label=title)
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Held-out score")
    ax.legend(ncol=4)
    ax.set_title("Provisional Q1 fixture: measured classical and quantum metrics")
    path = output_dir / "classification_metrics.svg"
    _save(fig, path); generated.append(path)

    names = [item["name"] for item in benchmark["processors"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key, title in zip(axes, ("gate_only_depth", "full_measured_depth", "cx_count"),
                              ("Gate-only depth", "Measured depth", "CX count"), strict=True):
        values = [item["circuit_cost"][key] for item in benchmark["processors"]]
        ax.bar(names, values, color=["#3c7292", "#b05b58"])
        ax.set_title(title)
        ax.bar_label(ax.containers[0])
    fig.suptitle("Explicit SABRE compilation on custom Processor A/B maps")
    path = output_dir / "circuit_costs.svg"
    _save(fig, path); generated.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    cases = [("Logistic", baseline["metrics"]), ("Ideal VQC", ideal["metrics"])] + [
        (item["name"] + " noisy", item["noisy"]["metrics"])
        for item in benchmark["processors"]
    ]
    for ax, (title, result) in zip(axes.flat, cases, strict=True):
        matrix = np.asarray(result["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="#122637")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout()
    path = output_dir / "confusion_matrices.svg"
    _save(fig, path); generated.append(path)

    if not convergence:
        raise ValueError("training convergence must come from actual optimizer callbacks")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([point["evaluation"] for point in convergence],
            [point["objective"] for point in convergence], marker="o", ms=3)
    ax.set_xlabel("COBYLA callback evaluation")
    ax.set_ylabel("Cross-entropy objective")
    ax.set_title("Recorded ideal VQC training convergence")
    path = output_dir / "training_convergence.svg"
    _save(fig, path); generated.append(path)
    return generated
