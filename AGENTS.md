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

Current phase: Phase 8 live Case A acceptance. Real DeepSeek smoke is human
accepted; the provider blocker is cleared. Latest Case A rerun 13 reached a real
FEASIBLE solve but independent validation is NOT_EVALUABLE. See the Case A
execution plan for exact evidence. The generic versioned metric-recomputation
and scenario-replay extension is implemented. Case A now has a reviewed v2
modeling contract and production-eligible verification policy bound to model
digest `16c3f292...`; its normalized initial state/time contract, fixed-ratio
comparator, Jacobian spectrum, local sensitivity design, persistence boundary,
and parasite interpretation are explicit assumptions/derivations. This does not
retroactively verify rerun 13 and no fresh attempt has run. Phase 8 remains
NOT_READY until a fresh solve produces the exact reviewed model/evidence chain.
Preserve every historical attempt, do not substitute Mock, create a commit, or
start B/C before A is stable. Do not rerun unsupported scientific requirements
blindly or remove them to manufacture verification.

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
