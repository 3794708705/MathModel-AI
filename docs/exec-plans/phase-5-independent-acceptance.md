# Phase 5 independent acceptance audit

## Goal

Attack Phase 5 final-verification semantics and prove that only one exact formal
result can become VERIFIED after the complete non-Mock validation, experiment,
and Red Team chain. Do not create a commit or enter Phase 6.

## Current state

The audit found and fixed premature SOLVE verification, the missing final-result
pointer, insufficient terminal re-audit of persisted validation/experiment
evidence, trusted Red Team count fields, and version-only no-op repair acceptance.
Targeted unit/schema/config tests and the real Docker verification/repair E2E pass.

## Files affected

- Phase 5 schemas, quality gates, experiment integrity, repository, and workflow
- ProblemState and Phase 4 result status handoff
- Verification API response contracts
- Adversarial unit and real-execution integration tests
- Phase 5 state, computational truth, testing, and verification documentation

## Design

- Reserve `VERIFIED` for a new final deterministic gate; SOLVE proves only
  computational provenance and leaves the result UNVERIFIED.
- Store `verified_result_id` in ProblemState and validate that exactly that
  persisted formal result has VERIFIED status and a passing VERIFIED gate.
- Reconstruct every scenario from baseline model plus perturbation metadata,
  parse the deterministic program payload without `eval`, and bind it to program,
  bundle, execution, solver result, and independent feasibility digests.
- Re-audit persisted experiment rows and execution evidence before final
  acceptance; never rely only on report JSON.
- Recompute persisted validation at the terminal gate and compare every
  deterministic check before accepting it.
- Recompute Red Team severity counts from findings and cross-check the complete
  report/model/result chain.
- Keep repair output model-only and enforce immutable source records, a new model
  version, new solve/run/execution/result, and the configured maximum of three
  automatic cycles.

## Tests

Adversarial cases A-J cover Critical findings, NOT_EVALUABLE, objective tamper,
sensitivity and robustness replay, severity-count tamper, immutable repair
versioning, iteration exhaustion, explicit final-result selection, and Mock/
BLOCKED rejection. Run real Docker E2E plus full coverage, Ruff, strict mypy,
Alembic, PostgreSQL, and diff checks.

## Risks

- Deterministic payload parsing intentionally supports only the trusted solver
  adapter template; an unfamiliar program is not accepted as experiment evidence.
- Common-mode defects remain possible because independent evaluation consumes the
  same typed mathematical model, though it does not consume solver feasibility or
  objective assertions as truth.

## Acceptance criteria

- A-J pass with no path from Mock, BLOCKED, NOT_EVALUABLE, tampered metadata, or
  Critical findings to `verified_result_id`.
- Repair creates immutable new computational identities and cannot mutate Result.
- Full quality suite passes; no commit and no Phase 6 code are created.

## Audit outcome

- A-J have explicit adversarial or real-integration coverage.
- The real repair path preserves v1 model/result/solver/execution payloads and
  creates distinct v2 records before re-verification.
- Full regression: 175 passed, 4 explicit environment skips, 87.47% coverage;
  dedicated PostgreSQL: 1 passed; Ruff, strict mypy, Alembic, and diff checks pass.
