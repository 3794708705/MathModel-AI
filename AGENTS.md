# MathModel AI repository guide

## Goal

Build a traceable, reproducible mathematical-modeling competition system.
Priority: correctness, traceability, validation, competition value, reliability,
maintainability, performance, then complexity.

## Project map

- Architecture: `docs/ARCHITECTURE.md`
- Product scope: `docs/PRODUCT.md`
- State contracts: `docs/STATE_MODEL.md`
- Provider and routing rules: `docs/MODEL_ROUTING.md`
- Agent contracts: `docs/AGENT_CONTRACTS.md`
- Reasoning core: `docs/REASONING_CORE.md`
- Data and execution: `docs/DATA_EXECUTION.md`
- Mathematical core: `docs/MATHEMATICAL_CORE.md`
- Solver architecture: `docs/SOLVER_ARCHITECTURE.md`
- Verification and repair: `docs/VERIFICATION_REPAIR.md`
- Computational truth: `docs/COMPUTATIONAL_TRUTH.md`
- Security: `docs/SECURITY.md`
- Testing: `docs/TESTING.md`
- Execution plans: `docs/exec-plans/`

Current phase: Phase 5 — Verification, Sensitivity, Robustness, Red Team, and
Model Repair (implemented, pending human acceptance). Do not implement Phase 6
literature, citation, figure/table, Paper IR, LaTeX, or PDF behavior unless the
phase is explicitly advanced.

## Commands

```text
uv sync --dev
uv run ruff format .
uv run ruff check .
uv run mypy --strict src
uv run pytest --cov=mathmodel_ai
docker build -t mathmodel-ai-sandbox:phase3 sandbox
docker build -f sandbox/solver.Dockerfile -t mathmodel-ai-solver:phase4 sandbox
```

## Non-negotiable rules

- Never fabricate solver output, executions, evidence, or citations.
- Never allow an unverified number into a final paper artifact.
- Keep model SDK details behind `BaseModelProvider`.
- Keep shared facts in `ProblemState`; agents must not maintain private truth.
- Preserve raw inputs and use migrations for persisted schema changes.
- Python code may run only through `SandboxExecutor`; an `ExecutionRecord` is
  required before any execution claim.
- A numerical result must link to its exact `MathematicalModel` version,
  `SolverRun`, and non-Mock `ExecutionRecord` before it is verified.
- Sensitivity and robustness claims require independently recorded, non-Mock
  experiment executions; a Mock Red Team or repair can never pass its gate.
- Do not commit credentials or log secret values.
- Add tests and documentation for every completed module.
