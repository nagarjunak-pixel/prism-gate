# Contributing to PrismGate

Contributions to **PrismGate** are welcome and appreciated.

## Local Development Workflow

1. Sync Python 3.12 dependencies via `uv`:
   ```bash
   uv sync --python 3.12 --all-extras
   ```
2. Run formatting, strict static typing, and the 95%+ coverage test gate:
   ```bash
   make lint typecheck test drift
   ```
3. Verify the TypeScript 5.6 + React 18 observability UI bundle:
   ```bash
   make ui-build
   ```

## Standards
- All domain contracts in `src/prism_gate/types.py` must remain frozen Pydantic v2 models (`frozen=True, extra="forbid"`).
- Statistical & concurrency primitives require deterministic `Hypothesis` property tests proving bounds, symmetry, and conservation invariants.
- Conventional Commits (`feat(...)`, `fix(...)`, `docs(...)`) are required on all pull requests.
