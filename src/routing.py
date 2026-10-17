"""Explicit SABRE compilation and comparable gate-cost measurements."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

from src.processors import BASIS_GATES, ProcessorSpec


COMPILER_SETTINGS = {
    "basis_gates": list(BASIS_GATES),
    "optimization_level": 1,
    "seed_transpiler": 42,
    "layout_method": "sabre",
    "routing_method": "sabre",
}


def compile_unconstrained(circuit: QuantumCircuit) -> QuantumCircuit:
    return transpile(
        circuit,
        basis_gates=list(BASIS_GATES),
        optimization_level=1,
        seed_transpiler=42,
    )


def compile_processor(circuit: QuantumCircuit, spec: ProcessorSpec) -> QuantumCircuit:
    return transpile(
        circuit,
        coupling_map=CouplingMap(list(spec.coupling_edges)),
        **COMPILER_SETTINGS,
    )


def sabre_swap_trace(circuit: QuantumCircuit, spec: ProcessorSpec) -> dict[str, Any]:
    """Count explicit SWAPs immediately after deterministic SABRE routing.

    The input must contain no SWAP operations. The transpiler callback observes the
    DAG emitted by ``SabreLayout`` before basis translation decomposes SWAPs into
    CX gates. This is a routing-stage count, not an estimate from final CX cost.
    """

    if circuit.count_ops().get("swap", 0):
        raise ValueError("SWAP trace requires a logical input with no SWAP operations")
    events: list[dict[str, Any]] = []

    def callback(**kwargs: Any) -> None:
        compiler_pass = kwargs.get("pass_")
        dag = kwargs.get("dag")
        if compiler_pass is None or dag is None:
            return
        if compiler_pass.__class__.__name__ == "SabreLayout":
            events.append({
                "pass_index": int(kwargs["count"]),
                "pass_name": "SabreLayout",
                "operation_counts": {
                    str(name): int(count) for name, count in dag.count_ops().items()
                },
            })

    compiled = transpile(
        circuit,
        coupling_map=CouplingMap(list(spec.coupling_edges)),
        callback=callback,
        **COMPILER_SETTINGS,
    )
    if len(events) != 1:
        raise RuntimeError(f"expected one SabreLayout trace event, observed {len(events)}")
    event = events[0]
    return {
        "processor": spec.name,
        "qiskit_version": version("qiskit"),
        "method": (
            "count explicit swap nodes in the DAG immediately after SabreLayout; "
            "logical input contains zero swap nodes"
        ),
        "compiler_settings": dict(COMPILER_SETTINGS),
        "trace_event": event,
        "verified_inserted_swap_count": int(event["operation_counts"].get("swap", 0)),
        "final_native_basis_operation_counts": {
            str(name): int(count) for name, count in compiled.count_ops().items()
        },
    }


def circuit_cost(circuit: QuantumCircuit) -> dict[str, int | None]:
    operations = circuit.count_ops()
    return {
        "gate_only_depth": int(circuit.depth(
            filter_function=lambda instruction: instruction.operation.name not in {"measure", "reset", "barrier"}
        )),
        "full_measured_depth": int(circuit.depth()),
        "cx_count": int(operations.get("cx", 0)),
        "explicit_swap_ops_after_transpile": int(operations.get("swap", 0)),
        # Native-basis decomposition prevents reconstructing exact inserted SWAPs.
        "verified_inserted_swap_count": None,
    }


def layout_record(logical: QuantumCircuit, compiled: QuantumCircuit) -> dict[str, Any]:
    layout = compiled.layout
    if layout is None:
        raise RuntimeError("compiled circuit has no layout")
    initial = layout.initial_layout.get_virtual_bits()
    return {
        "logical_to_initial_physical": [int(initial[qubit]) for qubit in logical.qubits],
        "logical_to_final_physical": [int(index) for index in layout.final_index_layout()],
        "physical_qubit_count": compiled.num_qubits,
        "logical_qubit_count": logical.num_qubits,
    }
