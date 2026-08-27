# Phase 4.1 — Mathematical Core Hardening & Acceptance Fix

## Blockers

- Evidence verification proves row existence but not numerical, status, model-digest,
  program-hash, or execution-truth consistency.
- `CodeAgent` exists as an isolated agent but has no selectable, persisted,
  sandboxed workflow path.
- Solver routing follows static preferences; size, deadline pressure, and declared
  solver requirements do not materially affect selection or runtime options.
- The real-Solver E2E uses Mock reasoning/model construction, and the paid-provider
  check does not exercise MathModeler through evidence verification.
- Expression, selector, routing, generated execution, evidence, and unlicensed
  Gurobi logic need stronger focused coverage.

## Root Cause

- Phase 4 treated evidence primarily as relational linkage. It did not define one
  content-integrity verifier shared by the in-memory SOLVE gate and persisted
  evidence endpoint.
- Deterministic adapters directly construct and execute their own generated
  program, while the separately registered `CodeAgent` was intentionally left
  outside orchestration.
- `AlgorithmPlan` records size and risk metadata, but `SolverRouter` iterates the
  preference list without hard-filter/policy scoring or a resolved runtime budget.
- Live-provider coverage was limited to configuration because normal CI must not
  require paid credentials.

## Affected Modules

- Schemas: mathematical model, program, execution, solver, result, API, and state
  references.
- Mathematical services: canonical model digest, evidence verifier, quality gates,
  repository, workflow, algorithm selection, and execution-strategy selection.
- Solver services: routing decision/scoring, option propagation, deterministic and
  generated-program execution/result parsing, and Gurobi status mapping helpers.
- Persistence: one additive `0005` migration for digests, execution origin,
  routing trace, and executable hash metadata.
- Tests and documentation for computational truth, solver architecture, state,
  agents, data/sandbox contracts, and Phase 4 acceptance.

## Implementation Plan

1. Define canonical semantic `model_digest`, execution origin, executable hash,
   structured evidence error codes, and numeric tolerance settings.
2. Implement `EvidenceIntegrityVerifier` over domain objects and a persisted
   loader that checks ResultRecord, SolverRun/SolverResult, exact model revision,
   GeneratedProgram where applicable, and successful non-Mock ExecutionRecord.
3. Make SOLVE gate depend on that verifier before persistence, and make the API
   evidence endpoint use the same semantic checks after persistence.
4. Add `ExecutionStrategy` (`AUTO`, `DETERMINISTIC`, `GENERATED`) and a selector.
   Keep `AUTO` deterministic-first. Integrate CodeAgent, approved dependency/static
   checks, existing SandboxExecutor, strict `result.json` parsing, result/solver-run
   construction, agent-run persistence, and explicit requested-strategy failures.
5. Add size classification and configurable thresholds. Route with hard filters,
   deterministic policy scores, declared solver preferences/capabilities, deadline
   policy, and a resolved runtime budget propagated into `SolverOptions`.
6. Add an explicitly gated live MathModeler-to-real-SciPy E2E. It must never fall
   back to Mock and must report skip separately when credentials are absent.
7. Add focused tests for all requested tamper cases, generated code/hash behavior,
   routing changes, runtime propagation, expression branches, selector branches,
   and license-independent Gurobi mappings.

## Test Plan

- Unit: canonical digest stability; numeric tolerance; all structured evidence
  errors; expression translation/evaluation; selector; strategy selector; routing
  scores and hard rejections; generated result parser; Gurobi mappings.
- Tamper regression: objective, status, key outputs, result ref, model version,
  model digest, program hash, executed hash, exit code, and Mock execution. Restore
  each mutation and confirm validity returns.
- Integration: deterministic API path; Mock-CodeAgent generated path with real
  Docker/SciPy; explicit strategy errors; persisted evidence endpoint; PostgreSQL
  columns/FKs and Alembic upgrade/downgrade/re-upgrade.
- Optional live: non-Mock reasoning and MathModeler, real deterministic SciPy,
  objective approximately 30, non-Mock execution, valid evidence, and `SOLVE` state.
- Regression: all existing Phase 1–4 tests, Ruff format/check, strict mypy,
  coverage at least 85%, Alembic check, Docker/Sandbox/Solver tests, and
  `git diff --check`.

## Acceptance Criteria

- Persisted objective `30 -> 999` mutation returns invalid with
  `OBJECTIVE_MISMATCH`; every other requested mutation returns its exact code and
  restoration returns valid.
- SOLVE cannot pass without schema, status, feasibility, and full evidence
  integrity.
- Both deterministic and generated execution strategies produce real non-Mock
  ExecutionRecords and valid ResultRecords; `AUTO` remains deterministic-first.
- Generated source A cannot be accepted when ExecutionRecord proves source B.
- Size, deadline pressure, and solver requirements materially affect a recorded
  routing decision, and model maximum runtime reaches the executed solver options.
- Live provider E2E is `PASS`, `SKIPPED_NO_CREDENTIALS`, or `FAIL`; a skip is never
  represented as a pass.
- Targeted core coverage is materially strengthened without fake branches or a
  commercial license dependency; all existing tests remain green.
- Documentation is current, no Phase 5 feature is implemented, no baseline commit
  is created, and the final report states `READY` only if every non-credential
  acceptance condition passes.

## Implementation Result

- P0 content-integrity verifier and all ten restore-after-tamper regressions:
  implemented and passing against a real Docker/SciPy result.
- `AUTO`/`DETERMINISTIC`/`GENERATED` workflow integration: implemented;
  deterministic and Mock-CodeAgent/real-sandbox paths pass.
- Dynamic routing: implemented with structured hard rejections, configurable
  size thresholds, deadline-sensitive policy score/budget, and requirements.
- Live provider E2E: implemented and gated; current acceptance environment has
  no credentials, so its status is `SKIPPED_NO_CREDENTIALS`.
- Migration `20260827_0005`: PostgreSQL upgrade, downgrade to `0004`, re-upgrade,
  autogenerate check, and PostgreSQL integration test pass.
- Phase 5 features and baseline commit: intentionally not started.
