"""Keep the supplemental D illustration tied to saved A/B evidence."""

import tempfile
import unittest
from pathlib import Path

from src.architecture_concept import D_EDGES, LOGICAL_RING, render


class ArchitectureConceptTests(unittest.TestCase):
    def test_ring_concept_is_valid_and_explicitly_unmeasured(self):
        self.assertEqual(len({node for edge in D_EDGES for node in edge}), 5)
        self.assertTrue(set(LOGICAL_RING).issubset(set(D_EDGES)))
        with tempfile.TemporaryDirectory() as directory:
            output = render(Path(directory) / "concept.svg")
            drawing = output.read_text()
            self.assertIn("concept only; no compilation or noise result", drawing)
            self.assertIn("24 CX", drawing)
            self.assertIn("8 CX", drawing)


if __name__ == "__main__":
    unittest.main()
