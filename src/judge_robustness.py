"""One bounded, compilation-only seed robustness check of the official maps."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

from qiskit import transpile
from qiskit.transpiler import CouplingMap

from src.processors import ProcessorSpec
from src.q1_submission import BENCHMARK_PATH, PROCESSOR_PATHS, ROOT
from src.routing import COMPILER_SETTINGS, circuit_cost, layout_record
from src.vqc import build_vqc_circuit


OUTPUT_PATH = ROOT / "results" / "tables" / "q1_seed_robustness.json"
SEEDS = (7, 42, 99)


def run(output_path: Path = OUTPUT_PATH) -> dict[str, Any]:
    logical, _, _ = build_vqc_circuit()
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    rows = []
    for path, official in zip(PROCESSOR_PATHS, benchmark["processors"], strict=True):
        spec = ProcessorSpec.from_json(path)
        if official["name"] != spec.name:
            raise ValueError("processor configuration order differs from the official benchmark")
        for seed in SEEDS:
            events: list[int] = []

            def callback(**kwargs: Any) -> None:
                if kwargs["pass_"].__class__.__name__ == "SabreLayout":
                    events.append(int(kwargs["dag"].count_ops().get("swap", 0)))

            settings = {**COMPILER_SETTINGS, "seed_transpiler": seed}
            compiled = transpile(
                logical, coupling_map=CouplingMap(list(spec.coupling_edges)),
                callback=callback, **settings,
            )
            if len(events) != 1:
                raise RuntimeError("expected exactly one SabreLayout routing-stage trace")
            cost = circuit_cost(compiled)
            layout = layout_record(logical, compiled)
            if seed == 42 and (cost != official["circuit_cost"] or layout != official["layout"]):
                raise ValueError("seed-42 row no longer reproduces the frozen official benchmark")
            rows.append({
                "processor": spec.name,
                "seed_transpiler": seed,
                "gate_only_depth": cost["gate_only_depth"],
                "full_measured_depth": cost["full_measured_depth"],
                "cx_count": cost["cx_count"],
                "explicit_swap_count_after_sabre_layout": events[0],
                "layout": layout,
            })
    result = {
        "schema_version": 1,
        "scope": "secondary deterministic transpilation-only robustness check; no training or Aer inference",
        "qiskit_version": version("qiskit"),
        "seeds": list(SEEDS),
        "controlled_settings": {key: value for key, value in COMPILER_SETTINGS.items()
                                if key != "seed_transpiler"},
        "logical_qubits": logical.num_qubits,
        "logical_cx_count": int(logical.count_ops().get("cx", 0)),
        "seed_42_is_frozen_official_baseline": True,
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    output = run()
    print(OUTPUT_PATH, "recorded", len(output["rows"]), "compilation rows")
