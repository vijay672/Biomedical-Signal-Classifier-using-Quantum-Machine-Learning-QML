"""Write the lightweight, deterministic SABRE SWAP trace diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

from src.processor_benchmark import PROCESSOR_FILES
from src.processors import ProcessorSpec
from src.routing import sabre_swap_trace
from src.vqc import build_vqc_circuit


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "results" / "tables" / "q1_provisional_swap_diagnostic.json"


def run(output_path: Path = OUTPUT_PATH) -> dict[str, object]:
    logical, _, _ = build_vqc_circuit()
    processors = [
        sabre_swap_trace(logical, ProcessorSpec.from_json(path))
        for path in PROCESSOR_FILES
    ]
    result: dict[str, object] = {
        "schema_version": 1,
        "scope": "transpilation-only diagnostic; no training or simulation",
        "logical_input_swap_count": int(logical.count_ops().get("swap", 0)),
        "processors": processors,
        "interpretation": (
            "SWAP counts are observed at the SABRE routing stage. Final CX/depth "
            "overheads remain separately measured because later basis translation "
            "and optimization can decompose and cancel gates."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    payload = run()
    print(OUTPUT_PATH)
    for processor in payload["processors"]:
        print(processor["processor"], processor["verified_inserted_swap_count"])
