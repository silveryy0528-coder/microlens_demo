# Project context

- This repository is research software for characterizing spherical microlenses from 2D height maps.
- Prioritize scientific correctness, reproducibility, clarity, and maintainability over general-purpose architecture.
- The scientific requirements and unresolved design choices are recorded in [docs/methodology.md](../docs/methodology.md). Read it before designing or modifying scientific processing code; treat it as the scientific source of truth.

# Architecture and coding

- Keep the implementation focused on this one use case. Avoid frameworks, plugin systems, and abstractions for hypothetical future applications.
- Use `UseCaseMicrolens` to orchestrate preprocessing and processing. Keep numerical algorithms independently testable; separate analysis from file I/O and plotting.
- Represent height maps with the project's `Image` class, preserving pixel sizes and the lateral/data units. Do not silently convert units.
- Prefer NumPy and SciPy for numerical methods, and established Python libraries for plotting, tabular outputs, and testing. Add dependencies only when justified.
- Prefer straightforward implementations and small, coherent changes. Do not change unrelated files or public behavior without a reason.
- Do not invent unspecified algorithm choices (for example, fit masks, threshold percentiles, or profile sampling parameters). Ask for a decision or clearly flag a proposed default.

# Scientific correctness and tests

- Distinguish software correctness from scientific validation: use unit tests for local behavior and synthetic analytical ground truth for end-to-end accuracy.
- Keep coordinate conventions and physical units explicit. Use nm for the initial synthetic dataset; radius of curvature and lens height are reported in nm.
- Do not change equations, scientific definitions, or processing order without explaining the change and obtaining approval.
- Use pytest. Add or update focused tests for changed behavior, including important edge cases.
- Prefer test names such as `test_<function>_WHEN_<condition>_THEN_<expected_behavior>`.
- Run relevant tests after changes and report what was executed and whether it passed. Never claim an unrun test passed.

# Documentation

- Follow PEP 257 and use concise NumPy-style docstrings for public scientific interfaces.
- Document non-obvious assumptions, array/coordinate conventions, and numerical limitations, not line-by-line code behavior.
- Keep methodology specifications in `docs/methodology.md`; avoid copying the full methodology into these instructions.

# Collaboration with Copilot

- Inspect relevant files and existing tests before proposing changes.
- For new modules, architectural decisions, or scientific algorithms: outline a short plan and any assumptions first; wait for approval before editing.
- For small, approved changes: implement directly, keep the diff focused, and verify it.
- Summarize (1) what changed, (2) why, (3) test/validation results, and (4) any open decisions.
- When comparing designs, present the meaningful tradeoffs and recommend the simplest adequate approach.
