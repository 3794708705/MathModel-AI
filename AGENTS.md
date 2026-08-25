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
- Computational truth: `docs/COMPUTATIONAL_TRUTH.md`
- Security: `docs/SECURITY.md`
- Testing: `docs/TESTING.md`
- Execution plans: `docs/exec-plans/`

Current phase: Phase 2 — Reasoning Core (complete). Do not implement Phase 3
data/file behavior unless the phase is explicitly advanced.

## Commands

```text
uv sync --dev
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest --cov=mathmodel_ai
```

## Non-negotiable rules

- Never fabricate solver output, executions, evidence, or citations.
- Never allow an unverified number into a final paper artifact.
- Keep model SDK details behind `BaseModelProvider`.
- Keep shared facts in `ProblemState`; agents must not maintain private truth.
- Preserve raw inputs and use migrations for persisted schema changes.
- Generated code may run only through the future sandbox boundary.
- Do not commit credentials or log secret values.
- Add tests and documentation for every completed module.
