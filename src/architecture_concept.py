"""Evidence-backed A/B topology comparison and an unbenchmarked D concept."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from src.processors import ProcessorSpec
from src.q1_submission import BENCHMARK_PATH, ROOT
from src.routing_diagnostic import OUTPUT_PATH as SWAP_PATH


OUTPUT_PATH = ROOT / "results/figures/architecture_problem_and_D_concept.svg"
LOGICAL_RING = ((0, 1), (1, 2), (2, 3), (3, 0))
D_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 4))


def _positions(name: str) -> dict[int, tuple[float, float]]:
    if name == "A":
        return {index: (float(index), 0.5) for index in range(5)}
    if name == "B":
        return {index: (float(index % 3), float(2 - index // 3)) for index in range(9)}
    return {0: (0.0, 1.5), 1: (1.3, 1.5), 2: (1.3, 0.2),
            3: (0.0, 0.2), 4: (-0.9, 2.0)}


def _panel(ax, *, name: str, physical_count: int, edges: set[frozenset[int]],
           placement: list[int], subtitle: str) -> None:
    coordinates = _positions(name)
    direct = {
        frozenset((placement[start], placement[end]))
        for start, end in LOGICAL_RING
    }
    for edge in edges:
        first, second = sorted(edge)
        x1, y1 = coordinates[first]
        x2, y2 = coordinates[second]
        ax.plot((x1, x2), (y1, y2), color="#198568" if edge in direct else "#91a4af",
                lw=3.5 if edge in direct else 2, zorder=1)
    for missing in direct - edges:
        first, second = sorted(missing)
        x1, y1 = coordinates[first]
        x2, y2 = coordinates[second]
        ax.add_patch(FancyArrowPatch(
            (x1, y1 + .12), (x2, y2 + .12), arrowstyle="-",
            connectionstyle="arc3,rad=0.4", color="#c55743",
            linestyle="--", linewidth=2.5, zorder=1.5,
        ))
    for physical in range(physical_count):
        x, y = coordinates[physical]
        logical = placement.index(physical) if physical in placement else None
        label = f"p{physical}" + (f"\nq{logical}" if logical is not None else "\nspare")
        ax.scatter((x,), (y,), s=1100, color="#d7ede7" if logical is not None else "#e7edf0",
                   edgecolor="#16485a", linewidth=1.5, zorder=2)
        ax.text(x, y, label, va="center", ha="center", fontsize=9, zorder=3)
    ax.set_title(f"{name}: {subtitle}", fontsize=10, loc="left")
    ax.set_aspect("equal")
    ax.set_xlim(min(x for x, _ in coordinates.values()) - .5,
                max(x for x, _ in coordinates.values()) + .5)
    if name == "A":
        ax.set_ylim(-.9, 1.3)
    else:
        ax.set_ylim(min(y for _, y in coordinates.values()) - .5,
                    max(y for _, y in coordinates.values()) + .5)
    ax.axis("off")


def render(output_path: Path = OUTPUT_PATH) -> Path:
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    diagnostic = json.loads(SWAP_PATH.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    fig.suptitle("The four-qubit logical ring meets three physical graphs", fontsize=14)
    for index, suffix in enumerate(("A", "B")):
        spec = ProcessorSpec.from_json(ROOT / f"processors/processor_{suffix}.json")
        saved = benchmark["processors"][index]
        traced = diagnostic["processors"][index]
        if (saved["name"] != spec.name or traced["processor"] != spec.name
                or saved["coupling_edges"] != [list(edge) for edge in spec.coupling_edges]):
            raise ValueError("saved benchmark and physical graph differ")
        _panel(
            axes[index], name=suffix, physical_count=spec.num_qubits,
            edges={frozenset(edge) for edge in spec.coupling_edges},
            placement=saved["layout"]["logical_to_initial_physical"],
            subtitle=(f"measured {saved['circuit_cost']['cx_count']} CX, "
                      f"{traced['verified_inserted_swap_count']} routing SWAPs"),
        )
    _panel(axes[2], name="D", physical_count=5,
           edges={frozenset(edge) for edge in D_EDGES}, placement=[0, 1, 2, 3],
           subtitle="concept only; no compilation or noise result")
    fig.text(.02, .01,
             "Green: direct logical-ring interaction at shown placement. Red dashed: missing link on A. "
             "D adds a 0-3 coupler; all D effects remain hypothetical.", fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    print(render())
