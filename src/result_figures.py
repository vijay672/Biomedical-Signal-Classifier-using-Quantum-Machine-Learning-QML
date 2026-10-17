"""Render small, dependency-free SVG charts from recorded JSON results."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _read(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _label(x: float, y: float, value: str, *, size: int = 13, anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" fill="#183044">{escape(value)}</text>'
    )


def _bar(x: float, y: float, width: float, height: float, color: str) -> str:
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{color}"/>'


def _svg(width: int, height: int, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        + "".join(body)
        + "</svg>\n"
    )


def _metric_chart(
    title: str, rows: list[tuple[str, list[float]]], metric_names: list[str],
    colors: list[str], footer: str,
) -> str:
    width, height = 800, 125 + len(rows) * 54 + 45
    body = [_label(28, 35, title, size=20), _label(28, 59, footer, size=11)]
    left, plot_width = 220, 440
    for tick in (0, 0.25, 0.5, 0.75, 1):
        x = left + plot_width * tick
        body.append(f'<line x1="{x}" y1="90" x2="{x}" y2="{height-40}" stroke="#e1e8ed"/>')
        body.append(_label(x, height - 20, f"{tick:.0%}", size=11, anchor="middle"))
    for row_index, (name, values) in enumerate(rows):
        y0 = 100 + row_index * 54
        body.append(_label(28, y0 + 15, name, size=13))
        for metric_index, value in enumerate(values):
            if not 0 <= value <= 1:
                raise ValueError("metric rates must be in [0, 1]")
            y = y0 + 19 + metric_index * 8
            body.append(_bar(left, y, plot_width * value, 7, colors[metric_index]))
            body.append(_label(left + plot_width * value + 8, y + 7, f"{value:.3f}", size=10))
    legend_x = 28
    for name, color in zip(metric_names, colors, strict=True):
        body.append(_bar(legend_x, height - 44, 12, 12, color))
        body.append(_label(legend_x + 17, height - 33, name, size=11))
        legend_x += 120
    return _svg(width, height, body)


def render_result_figures(
    *,
    baseline_path: Path | str = RESULTS / "logistic_regression_baseline.json",
    experiment_path: Path | str = RESULTS / "vqc_controlled_experiments.json",
    processor_path: Path | str = RESULTS / "processor_benchmark.json",
    output_dir: Path | str = RESULTS,
) -> tuple[Path, Path, Path]:
    """Create three charts solely from saved experiment records."""

    baseline, experiment, processors = map(
        _read, (baseline_path, experiment_path, processor_path)
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    selected = experiment["final_refit_and_held_out"]["metrics"]
    matrix = selected["confusion_matrix"]
    selected_accuracy = (matrix[0][0] + matrix[1][1]) / selected["sample_count"]
    performance = _metric_chart(
        "Held-out synthetic benchmark (40 rows)",
        [
            ("Logistic regression", [baseline["accuracy"], baseline["abnormal_recall"], baseline["precision"], baseline["f1"]]),
            ("Selected VQC", [selected_accuracy, selected["recall_label_1"], selected["precision_label_1"], selected["f1_label_1"]]),
        ],
        ["Accuracy", "Label-1 recall", "Precision", "F1"],
        ["#315c91", "#be4b49", "#309475", "#9a6ab1"],
        "Selected using inner validation only; label 1 is a synthetic positive class.",
    )
    validation = _metric_chart(
        "Three bounded VQC variants (inner validation, 32 rows)",
        [
            (entry["variant"]["name"].replace("_", " "), [
                entry["validation"]["recall_label_1"],
                entry["validation"]["precision_label_1"],
                entry["validation"]["f1_label_1"],
            ])
            for entry in experiment["candidates"]
        ],
        ["Label-1 recall", "Precision", "F1"],
        ["#be4b49", "#309475", "#9a6ab1"],
        "Selection prioritized recall. Circuit costs are tabulated in report.md.",
    )
    rows = processors["processors"]
    processor_recall = _metric_chart(
        "Aer recall on custom Processor A/B maps (original VQC)",
        [
            (f"{entry['name']} ({entry['num_qubits']} qubits)", [
                entry["ideal"]["recall_label_1"], entry["noisy"]["recall_label_1"]
            ])
            for entry in rows
        ],
        ["Ideal sampled", "Noisy sampled"],
        ["#315c91", "#be4b49"],
        "4,096 shots; same saved ring/parity VQC weights and synthetic noise rates.",
    )
    paths = (
        destination / "held_out_performance.svg",
        destination / "validation_variants.svg",
        destination / "processor_recall.svg",
    )
    for path, chart in zip(paths, (performance, validation, processor_recall), strict=True):
        path.write_text(chart, encoding="utf-8")
    return paths


if __name__ == "__main__":
    for figure in render_result_figures():
        print(figure)
