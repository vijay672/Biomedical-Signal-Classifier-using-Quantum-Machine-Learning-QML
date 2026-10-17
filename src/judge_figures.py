"""Render judge-facing diagnostics from saved, measured JSON only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.q1_submission import ROOT


TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"


def _read(name: str) -> dict:
    return json.loads((TABLES / name).read_text(encoding="utf-8"))


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render() -> list[Path]:
    audit = _read("q1_judge_audit.json")
    ablation = _read("q1_layout_ablation.json")
    experiment = _read("q1_recall_calibration.json")
    model = _read("q1_recall_calibration_model.json")
    statistics = _read("q1_statistical_review.json")
    created = []

    official_names = ["Logistic", "Ideal VQC", "Noisy A", "Noisy B"]
    official_keys = ["logistic", "ideal_vqc", "processor_A_noisy", "processor_B_noisy"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(len(official_names))
    for index, (key, label, color) in enumerate((
        ("accuracy", "Accuracy", "#237a73"),
        ("precision_label_1", "Precision (label 1)", "#52789a"),
        ("recall_label_1", "Recall (label 1)", "#b66754"),
        ("f1_label_1", "F1 (label 1)", "#a68043"),
    )):
        bars = ax.bar(positions + (index - 1.5) * .19,
                      [statistics["cases"][name]["metrics"][key] for name in official_keys],
                      width=.18, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7, rotation=90)
    ax.set_xticks(positions, official_names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Held-out metric")
    ax.set_title("Official Q1 models: synthetic label-1 classification metrics")
    ax.legend(ncol=2)
    created.append(_save(fig, "judge_model_comparison.svg"))

    names = ["Logistic", "Ideal VQC", "Noisy A", "Noisy B", "Separate\nvalidation model"]
    records = [
        audit["cases"]["logistic_regression"]["metrics"],
        audit["cases"]["ideal_vqc_exact"]["metrics"],
        audit["cases"]["Processor A noisy sampled"]["metrics"],
        audit["cases"]["Processor B noisy sampled"]["metrics"],
        experiment["held_out_test"]["metrics"],
    ]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    bars = ax.bar(names, [row["recall_label_1"] for row in records],
                  color=["#237a73", "#52789a", "#b66754", "#a68043", "#674b91"])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylim(0, 1.07)
    ax.set_ylabel("Held-out synthetic label-1 recall")
    ax.set_title("Recall on the same 40 test rows; last bar is a separate 128-row fit")
    ax.text(.01, -.27, "Separate model: threshold selected on 32 validation rows; precision 0.516, accuracy 0.525.",
            transform=ax.transAxes, fontsize=9)
    created.append(_save(fig, "recall_analysis.svg"))

    interval_keys = official_keys + ["separate_validation_model"]
    intervals = [statistics["cases"][name]["wilson_95_percent"]["recall_label_1"]
                 for name in interval_keys]
    points = [statistics["cases"][name]["metrics"]["recall_label_1"]
              for name in interval_keys]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for index, (point, interval) in enumerate(zip(points, intervals, strict=True)):
        color = "#674b91" if index == 4 else "#52789a"
        ax.errorbar(index, point, yerr=[[point - interval[0]], [interval[1] - point]],
                    fmt="o", color=color, capsize=5, markersize=7)
        ax.text(index, min(1.03, interval[1] + .04), f"{point:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(5), names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Recall of synthetic label 1")
    ax.set_title("95% Wilson intervals for recall (20 positive test rows)")
    ax.text(.01, -.27, "Separate model was selected on inner validation; intervals describe this fixed synthetic test split.",
            transform=ax.transAxes, fontsize=9)
    created.append(_save(fig, "recall_uncertainty.svg"))

    rows = ablation["rows"]
    labels = [row["processor"].replace("Processor ", "") + " / " + row["layout_method"]
              for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(axes, ("cx_count", "gate_only_depth"),
                              ("Native CX count", "Gate-only depth"), strict=True):
        values = [row["cost"][key] for row in rows]
        bars = ax.bar(labels, values, color=["#52789a", "#9bb1c2", "#b66754", "#d5ad9f"])
        ax.bar_label(bars)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(0, max(values) * 1.2)
        ax.set_title(title)
    fig.suptitle("Transpilation-only layout ablation: same circuit, basis, seed and SABRE routing")
    fig.tight_layout()
    created.append(_save(fig, "layout_ablation.svg"))

    exact = _read("q1_provisional_ideal_vqc.json")
    # True labels are reconstructed from the audited, fixed dataset rather than predictions.
    from src.datasets import make_provisional_q1_benchmark
    true_labels = make_provisional_q1_benchmark().y_test
    scores = np.asarray(exact["positive_probabilities"])
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(float(scores.min()) - .01, float(scores.max()) + .01, 16)
    for label, color in ((0, "#52789a"), (1, "#b66754")):
        ax.hist(scores[true_labels == label], bins=bins, alpha=.65,
                label=f"True label {label} (n={sum(true_labels == label)})", color=color)
    ax.axvline(.5, color="#333333", linestyle="--", label="Official threshold 0.5")
    ax.set_xlabel("Exact ideal VQC positive-class score")
    ax.set_ylabel("Test-row count")
    ax.set_title("Official VQC scores overlap near the decision threshold")
    ax.legend()
    created.append(_save(fig, "ideal_score_distribution.svg"))

    trace = model["training"]["convergence"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([row["evaluation"] for row in trace],
            [row["objective"] for row in trace], marker="o", markersize=3)
    ax.set_xlabel("Actual COBYLA callback evaluation")
    ax.set_ylabel("Training objective")
    ax.set_title("Separate 128-row fit: recorded 25-evaluation convergence")
    created.append(_save(fig, "recall_calibration_convergence.svg"))
    return created


if __name__ == "__main__":
    for path in render():
        print(path)
