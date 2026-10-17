"""Lightweight checks for the required scientific Python environment."""

import unittest
from importlib.metadata import version


class EnvironmentSmokeTest(unittest.TestCase):
    def test_required_packages_import(self):
        import qiskit
        import qiskit_aer
        import qiskit_machine_learning
        import sklearn

        # Read package versions from distribution metadata instead of relying
        # on optional or deprecated module-level __version__ attributes.
        distributions = (
            "qiskit",
            "qiskit-aer",
            "qiskit-machine-learning",
            "scikit-learn",
        )
        for distribution in distributions:
            with self.subTest(distribution=distribution):
                self.assertTrue(version(distribution))


if __name__ == "__main__":
    unittest.main()
