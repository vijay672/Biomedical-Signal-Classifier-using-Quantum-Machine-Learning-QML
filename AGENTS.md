# Project instructions

- Keep implementation small and reproducible; pin runtime dependencies in `requirements.txt`.
- Apply the organizer PDF priority recorded in `docs/DECISIONS.md`; preserve unresolved specifications and never invent results.
- Keep final verified tables and figures versioned through the `.gitignore` allowlist; keep local datasets, caches, and temporary runs excluded.
- Phase 1 targets Processor A (5 qubits) and Processor B (9 qubits); the proposed model architecture is four qubits.
- Keep the original synthetic fixture and results separate from the provisional three-informative-feature fixture. Reuse only matching identity-checked saved VQC parameters for A/B; label maps as custom.
- Run the lightweight environment check with `python -m unittest discover -s tests -v` after environment changes.
