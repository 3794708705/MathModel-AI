# Phase 5 — verification and repair execution plan

## Goal

Establish a traceable, computation-backed reliability loop from an accepted
Phase 4 result through independent validation, sensitivity, robustness, red
team review, and bounded model repair. Phase 6 paper behavior is out of scope.

## Current state

Baseline `6b626074052c2b5655c5df313f8bec3230df04b7` provides immutable
mathematical models, real sandboxed solver executions, result evidence, typed
expressions, and minimum feasibility recomputation. It has no independent
validation record, perturbation experiment registry, red-team report, or
versioned repair workflow.

## Files affected

- `src/mathmodel_ai/schemas/`, `verification/`, `agents/`, `api/`, and `main.py`
- `src/mathmodel_ai/db/models.py` and one additive Alembic migration
- `src/mathmodel_ai/schemas/problem_state.py` and workflow transitions
- Phase 5 unit/integration/solver/PostgreSQL tests
- architecture, state, computational-truth, agent, security, and testing docs

## Design

- Independent validation uses a separate bounded expression evaluator and
  recomputes objective, variables, domains, bounds, every declared constraint,
  required outputs, and evidence integrity from persisted inputs.
- Sensitivity performs configurable one-at-a-time perturbations (defaults
  `±5%`, `±10%`, `±20%`) only on sourced scalar parameters. Every scenario is
  actually solved; failed scenarios remain failed records.
- Robustness supports scenario, worst-case, seeded noise/Monte Carlo, and
  data-backed bootstrap only when required inputs exist. The selected method
  and seed are explicit; advanced methods are never run by default merely for
  appearance.
- Red Team merges deterministic attacks with one structured reviewer output.
  Findings are `CRITICAL`, `MAJOR`, or `MINOR`; any critical finding blocks
  progression.
- Model repair is LLM-proposed but deterministically identity-bound. It reuses
  `model_id`, increments the immutable version, records addressed finding IDs,
  passes the MODEL gate, then requires real re-solve, validation, and Red Team.
  At most three automatic cycles are allowed before `HUMAN_REVIEW`.
- Database records are immutable JSON-backed registries with exact model/result/
  execution references. `ProblemState` stores compact typed references only.

## Implementation steps

1. Add strict Phase 5 schemas and settings-backed limits/tolerances.
2. Implement independent validation and experiment engines plus quality gates.
3. Add Red Team and Model Repair agents and version binding.
4. Add additive persistence tables/repository, workflow transitions, and API.
5. Exercise pass/fail/tamper/perturbation/critical/repair-loop behavior.
6. Run all static, migration, unit, integration, and real-solver checks.

## Tests

- Schema invariants and independent objective/constraint/domain recomputation.
- Sensitivity default ranges, configured ranges, zero baselines, failed solves,
  immutable provenance, and deterministic summaries.
- Robustness method selection, seeded repeatability, scenario/worst-case
  aggregation, and unsupported-data handling.
- Red Team severity gate and no progression with critical findings.
- Repair version increment, unchanged stable identity, digest change, maximum
  three cycles, and mandatory re-execution.
- API/state/persistence E2E and PostgreSQL migration/check round trip.
- Full `pytest`, Ruff format/check, strict mypy, Alembic check, and diff check.

## Risks

- Common-mode validation defects are reduced by a separate evaluator, but both
  paths still consume the same typed model contract.
- Perturbation cost grows with parameters; configured caps and deadline-aware
  solver limits prevent unbounded experiment fan-out.
- Mock reviewers can test orchestration only. They cannot prove model quality,
  robustness, or successful repair.

## Acceptance criteria

- A valid real result advances through VALIDATE, SENSITIVITY, ROBUSTNESS, and
  RED_TEAM only after deterministic gates pass.
- Every experiment has a non-Mock execution record and exact base/scenario
  model digest; no fabricated number is accepted.
- Critical findings trigger repair; accepted repairs create a new model version
  and must rerun the computational loop. Three unsuccessful cycles end in
  `HUMAN_REVIEW`.
- Phase 5 tests and repository quality gates pass; no Phase 6 behavior or
  baseline commit is created.

## Implementation outcome

Implemented on 2026-08-28 without entering Phase 6 or creating a commit.

- ProblemState v5 and the complete SOLVE -> VALIDATE -> SENSITIVITY ->
  ROBUSTNESS -> RED_TEAM -> MODEL_REPAIR -> SOLVE transition loop are active.
- Migration `20260828_0006` passed PostgreSQL downgrade/upgrade round-trip,
  `alembic check`, table, and foreign-key verification.
- A real Docker E2E passed the initial solve plus four verification experiments;
  a second E2E created model v2, re-solved it, and repeated every Phase 5 gate.
- Three recorded automatic repair cycles force `HUMAN_REVIEW`; failed reviewer
  runs are audited before the workflow raises an error.
- Final suite: 162 passed, 4 optional tests skipped, 88.18% coverage. Ruff
  format/check, strict mypy, Alembic, PostgreSQL, and diff checks passed.
