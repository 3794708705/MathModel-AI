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
- Evidence and paper production: `docs/PAPER_PIPELINE.md`
- Competition profiles: `docs/COMPETITION_PROFILE.md`
- Final Jury: `docs/FINAL_JURY.md`
- Submission pipeline: `docs/SUBMISSION_PIPELINE.md`
- Computational truth: `docs/COMPUTATIONAL_TRUTH.md`
- Security: `docs/SECURITY.md`
- Testing: `docs/TESTING.md`
- Execution plans: `docs/exec-plans/`

Current phase: Phase 7 — Final Jury and submission freeze (implemented, pending
independent acceptance). Do not start the Phase 8 historical competition
benchmark unless explicitly advanced.

## Commands

```text
uv sync --dev
uv run ruff format .
uv run ruff check .
uv run mypy --strict src
uv run pytest --cov=mathmodel_ai
docker build -t mathmodel-ai-sandbox:phase3 sandbox
docker build -f sandbox/solver.Dockerfile -t mathmodel-ai-solver:phase4 sandbox
docker build -f sandbox/paper.Dockerfile -t mathmodel-ai-paper:phase6 sandbox
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
- Paper claims must resolve through the exact `verified_result_id` evidence
  snapshot; PaperAgent prose, unverified references, and Mock review are never
  factual authority.
- Submission readiness must bind an exact profile version, paper version,
  verified result, manifest, and package hash. Unknown blocking rules require
  human review; a persisted jury/check summary is never deterministic authority.
- Do not commit credentials or log secret values.
- Add tests and documentation for every completed module.
