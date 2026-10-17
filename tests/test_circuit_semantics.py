"""Exact checks of the frozen Q1 circuit and its routed measurement semantics."""

import json
import unittest
from collections import Counter
from html import escape
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

from src.artifacts import load_verified_model
from src.datasets import make_provisional_q1_benchmark
from src.processors import ProcessorSpec
from src.q1_submission import BENCHMARK_PATH, CONFIG, MODEL_PATH, ROOT
from src.routing import compile_processor
from src.vqc import build_vqc_circuit, build_vqc_components, parity_class


def measured_distribution(circuit):
    """Return probabilities indexed by the measured classical four-bit word."""

    measured = [
        (circuit.find_bit(item.qubits[0]).index,
         circuit.find_bit(item.clbits[0]).index)
        for item in circuit.data if item.operation.name == "measure"
    ]
    if len(measured) != 4 or {bit for _, bit in measured} != set(range(4)):
        raise AssertionError("four distinct classical output bits are required")
    state = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
    probabilities = np.zeros(16)
    for physical_word, probability in enumerate(state.probabilities()):
        classical_word = sum(
            ((physical_word >> physical) & 1) << classical
            for physical, classical in measured
        )
        probabilities[classical_word] += probability
    return probabilities, dict((classical, physical) for physical, classical in measured)


class FrozenCircuitSemanticsTests(unittest.TestCase):
    def test_exact_gate_order_and_unique_symbols(self):
        circuit, inputs, weights = build_vqc_circuit()
        observed = [
            (item.operation.name,
             tuple(circuit.find_bit(qubit).index for qubit in item.qubits),
             tuple(str(parameter) for parameter in item.operation.params))
            for item in circuit.data if item.operation.name != "barrier"
        ]
        expected = [("ry", (qubit,), (f"x[{qubit}]",)) for qubit in range(4)]
        for layer in range(2):
            for qubit in range(4):
                index = 8 * layer + 2 * qubit
                expected.extend((
                    ("ry", (qubit,), (f"w[{index}]",)),
                    ("rz", (qubit,), (f"w[{index + 1}]",)),
                ))
            expected.extend(("cx", edge, ()) for edge in ((0, 1), (1, 2), (2, 3), (3, 0)))
        expected.extend(("measure", (qubit,), ()) for qubit in range(4))
        self.assertEqual(observed, expected)
        self.assertEqual((len(inputs), len(weights), circuit.num_parameters), (4, 16, 20))
        symbols = [str(parameter) for name, _, params in observed
                   if name in {"ry", "rz"} for parameter in params]
        self.assertEqual(Counter(symbols), Counter(
            [f"x[{i}]" for i in range(4)] + [f"w[{i}]" for i in range(16)]
        ))

    def test_all_basis_words_use_odd_parity_readout(self):
        qnn = build_vqc_components(CONFIG).qnn
        outputs = [parity_class(word) for word in range(16)]
        self.assertEqual(outputs, [word.bit_count() % 2 for word in range(16)])
        self.assertEqual(sum(outputs), 8)
        self.assertEqual([qnn.interpret(word) for word in range(16)], outputs)
        self.assertEqual(qnn.output_shape, (2,))
        for word in range(16):
            bits = [(word >> qubit) & 1 for qubit in range(4)]
            for control, target in ((0, 1), (1, 2), (2, 3), (3, 0)):
                bits[target] ^= bits[control]
            after_ring = sum(bit << qubit for qubit, bit in enumerate(bits))
            self.assertEqual(parity_class(after_ring), ((word >> 0) & 1) ^ ((word >> 2) & 1))

    def test_all_sixteen_outcome_probabilities_survive_routing(self):
        logical, inputs, weights = build_vqc_circuit()
        benchmark = json.loads(BENCHMARK_PATH.read_text())
        sample = make_provisional_q1_benchmark().X_test[:2]
        parameter_values = np.linspace(-0.7, 0.8, len(weights))
        for row in sample:
            bindings = dict(zip(inputs, row, strict=True))
            bindings.update(zip(weights, parameter_values, strict=True))
            reference, logical_map = measured_distribution(logical.assign_parameters(bindings))
            self.assertEqual(logical_map, {i: i for i in range(4)})
            self.assertAlmostEqual(float(reference.sum()), 1.0)
            self.assertTrue(np.all(reference > 0), "the test input must exercise all 16 outcomes")
            for suffix, record in zip(("A", "B"), benchmark["processors"], strict=True):
                spec = ProcessorSpec.from_json(ROOT / f"processors/processor_{suffix}.json")
                compiled = compile_processor(logical, spec).assign_parameters(bindings)
                routed, physical_map = measured_distribution(compiled)
                with self.subTest(processor=suffix):
                    self.assertEqual(
                        [physical_map[logical_bit] for logical_bit in range(4)],
                        record["layout"]["logical_to_final_physical"],
                    )
                    np.testing.assert_allclose(routed, reference, rtol=0, atol=1e-12)
                    self.assertAlmostEqual(float(sum(
                        routed[word] for word in range(16) if parity_class(word)
                    )), float(sum(
                        reference[word] for word in range(16) if parity_class(word)
                    )), places=12)

    def test_six_symbols_are_structurally_distinct_but_output_inert(self):
        dataset = make_provisional_q1_benchmark()
        parameters, _ = load_verified_model(MODEL_PATH, dataset, CONFIG)
        qnn = build_vqc_components(CONFIG).qnn
        rows = dataset.X_test[:8]
        reference = np.asarray(qnn.forward(rows, parameters))
        sensitivity = []
        for index in range(16):
            changed = parameters.copy()
            changed[index] += 0.37
            sensitivity.append(float(np.max(np.abs(qnn.forward(rows, changed) - reference))))
        inert = {index for index, change in enumerate(sensitivity) if change < 1e-9}
        self.assertEqual(inert, {9, 10, 11, 13, 14, 15})
        self.assertTrue(all(sensitivity[index] > 1e-5 for index in set(range(16)) - inert))

    def test_saved_qiskit_drawings_match_current_compilation(self):
        logical, _, _ = build_vqc_circuit()
        drawings = [("logical", logical, "logical_circuit")]
        for suffix in ("A", "B"):
            spec = ProcessorSpec.from_json(ROOT / f"processors/processor_{suffix}.json")
            drawings.append((f"processor_{suffix}_transpiled", compile_processor(logical, spec),
                             f"processor_{suffix}_transpiled_circuit"))
        for stem, circuit, figure_stem in drawings:
            text_path = ROOT / f"results/tables/q1_provisional_{stem}_circuit.txt"
            self.assertEqual(text_path.read_text(), str(circuit.draw(output="text", fold=100)) + "\n")
            svg = (ROOT / f"results/figures/{figure_stem}.svg").read_text()
            first_gate_row = next(row.strip() for row in str(circuit.draw(output="text", fold=105)).splitlines()
                                  if "x[0]" in row)
            self.assertIn(escape(first_gate_row), svg)


if __name__ == "__main__":
    unittest.main()
