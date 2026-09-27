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
- Provider setup: `docs/CUSTOM_PROVIDERS.md`
- Provider/model persistence: `docs/MODEL_REGISTRY.md`
- Web control center: `docs/WEB_UI.md`
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
- Final project/benchmark status: `docs/FINAL_PROJECT_ACCEPTANCE.md`
- Computational truth: `docs/COMPUTATIONAL_TRUTH.md`
- Security: `docs/SECURITY.md`
- Testing: `docs/TESTING.md`
- Execution plans: `docs/exec-plans/`

The current delivery target, case status, and authorized run scope are in
`GOAL.md`; historical Case A decisions remain in its execution plan. Preserve
every historical attempt, never substitute Mock for a claimed live result, and
do not rerun unsupported scientific requirements blindly or remove them to
manufacture verification.

## Commands

```text
uv sync --dev
.\scripts\dev-backend.ps1
.\scripts\dev-frontend.ps1
uv run ruff format .
uv run ruff check .
uv run mypy --strict src
uv run pytest --cov=mathmodel_ai
cd frontend && npm ci && npm run build && npm run lint && npm run test:coverage
docker build -t mathmodel-ai-sandbox:phase3 sandbox
docker build -f sandbox/solver.Dockerfile -t mathmodel-ai-solver:phase4 sandbox
docker build -f sandbox/paper.Dockerfile -t mathmodel-ai-paper:phase6 .
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
