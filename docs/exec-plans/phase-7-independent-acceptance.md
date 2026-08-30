# Phase 7 Independent Adversarial Acceptance

## Goal

Independently prove that no stale, tampered, Mock, incomplete, or physically
modified Phase 7 state can become or remain `FROZEN` without rebuilding the
full deterministic chain from requirements through the physical package.

## Current State

- Parent baseline: `12506d85c9cfd74a836320921bd395418f3a3826`.
- Phase 7 self-test: 306 passed, 6 environment-gated skips, 88.49% coverage.
- Alembic head: `20260829_0008`.
- The worktree contains only uncommitted Phase 7 implementation and tests.
- The built-in competition profile is a `TEST_FIXTURE`, not a verified real
  competition ruleset.

## Files Affected

- `src/mathmodel_ai/schemas/submission.py`
- `src/mathmodel_ai/submission/`
- `src/mathmodel_ai/api/routes/final_submission.py`
- `src/mathmodel_ai/db/models.py` and the uncommitted Phase 7 migration if a
  persisted integrity field is required
- `tests/submission/` and Phase 7 integration tests
- Phase 7 architecture, security, and testing documentation

## Design

Use immutable canonical digests for every approval input that can become stale:
competition profile, requirement coverage, jury input, candidate artifact set,
manifest, and package. Recompute rule, requirement, jury, submission, and
physical-package facts at the final boundary. Fail closed on unsupported rule
semantics, ambiguous provenance, unsafe filenames, hidden data, Mock/live policy
mismatch, concurrent conflicting freezes, and partial storage failures.

## Implementation Steps

1. Add adversarial tests for profile versions/digests/provenance and stale rule
   results.
2. Add semantic requirement-role/subproblem binding and stale requirement
   detection.
3. Bind Final Jury to paper, model/result, profile, requirement, and artifact
   digests; verify immutable findings and Mock/live policy.
4. Harden anonymity/secret scanning across PDF metadata, Unicode-normalized
   text, supported artifact contents, and path variants.
5. Harden archive canonical names, case/Unicode collisions, resource limits,
   manifest equality, final PDF/rule checks, and TOCTOU behavior.
6. Prove freeze idempotency/conflict policy and atomic persistence behavior.
7. Run the independent matrix, all earlier adversarial regressions, real E2E,
   PostgreSQL/Alembic, static checks, and full coverage.

## Tests

- Independent acceptance tests grouped by profile, rules, requirements, jury,
  invalidation, anonymity/secrets, archive/manifest, freeze/dirty semantics,
  deadline/AI disclosure, reproduction, concurrency, and failure injection.
- Existing Phase 7 A-Z and Phase 4-6 adversarial tests.
- Real Phase 5 → Phase 6 PDF → Phase 7 physical ZIP roundtrip E2E.
- `pytest --cov`, Ruff, strict mypy, PostgreSQL integration, Alembic check, and
  `git diff --check`.

## Risks

- PDF hidden-content extraction is format-dependent; unsupported structures
  must fail closed or require human review rather than claim complete scanning.
- Windows reparse-point behavior varies by host privileges and filesystem.
- Live Final Jury and real competition provenance remain environment/Phase 8
  gated and must not be reported as passing.
- Concurrency tests must not imply cross-process guarantees stronger than the
  database constraints and transaction boundaries actually provide.

## Acceptance Criteria

- Every audit matrix group is `PASS`, or the phase is `NOT_READY` with an
  explicit blocker.
- Any semantic bypass found is fixed with a regression test.
- A persisted PASS/FROZEN flag alone can never establish readiness.
- Overall coverage remains at least 85% and all required regression/static/
  migration/E2E checks pass.
- No commit is created and Phase 8 is not started.

## Audit Result

`READY` for human acceptance. The generic profile remains a `TEST_FIXTURE`;
this result proves the Phase 7 software invariants, not the rules of any real
competition.

### Verification

- Full regression: 329 passed, 6 environment-gated skips.
- Coverage: 87.87% (required minimum: 85%).
- Phase 4-7 focused adversarial regression: 81 passed.
- PostgreSQL integration: 1 passed; Alembic downgrade/upgrade roundtrip and
  `alembic check` passed at head `20260829_0008`.
- Real Phase 5 -> Phase 6 PDF -> Phase 7 ZIP/manifest roundtrip E2E: passed.
- Ruff format/check, strict mypy over 142 source files, and `git diff --check`:
  passed.
- Live provider and live Final Jury: `SKIPPED_NO_CREDENTIALS`.
- Live literature: `SKIPPED_NO_NETWORK`.
- Licensed Gurobi E2E: `SKIPPED_NO_LICENSE`.

### Findings and Resolution

- P0: Final Jury approval was not cryptographically bound to every semantic
  input. Jury reports now bind the Paper IR/manifest, official model/result,
  profile, requirements, rules, candidate, and artifact-set digests; all
  scores and finding counts are recomputed.
- P0: Freeze could consume caller-supplied requirement coverage. Freeze now
  rebuilds coverage from the current problem requirements, Paper IR, and
  candidate artifacts.
- P1: Persisted profile, rule, jury, check, and snapshot summaries could become
  stale independently of their payloads. Canonical digests and repository/API
  read-time verification now reject this state.
- P1: Package construction and roundtrip verification did not independently
  prove the complete approval chain and physical contents. Both boundaries now
  rerun deterministic gates and require exact manifest/file/hash bindings.
- P1: Scanner and archive coverage omitted PDF metadata/attachments, several
  Unicode/path/secret forms, canonical filename collisions, and nested archive
  limits. These inputs now fail closed with adversarial regressions.
- P2: Repeated/concurrent identical freezes and correction classification had
  ambiguous behavior. Same-candidate freeze is idempotent within the process,
  database state is revision-guarded, and unsafe format-only relabeling enters
  human review.

### Remaining Limits

- Code reproduction has no automatic trusted executor in Phase 7. A profile
  requiring reproduction fails closed instead of receiving a PASS.
- Generic fixture rules do not establish MCM, CUMCM, or any other real
  competition readiness; real rule provenance is a Phase 8 acceptance item.
- Unicode normalization and supported PDF/image metadata scanning reduce known
  bypasses but do not constitute complete OCR, steganography, or malware
  analysis.
- The in-process freeze lock does not claim distributed lock guarantees; the
  database revision/current-state constraints are the cross-process boundary.
