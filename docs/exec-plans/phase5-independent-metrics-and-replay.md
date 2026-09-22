# Phase 5 independent metrics and replay

## Goal

Add artifact-grounded, versioned metric recomputation and fresh scenario execution
inside the existing verification boundary. Preserve all historical Phase 8 data;
no commit, no B/C run, no inferred scientific thresholds or READY declaration.

## Current State / source inventory

- HEAD: `30a7ae72cb28ad3845decb1b6c61ae95efa1cc95`; dirty Phase 8 tree preserved.
- Backup: `C:/Users/zeon/phase5-backup/20260906-independent-extension` (binary
  tracked diff plus 115 non-ignored untracked source files).
- Phase 5: `verification/{validation,evaluator,experiments,sensitivity,robustness,
  experiment_integrity,quality_gates,workflow,repository}.py`.
- Existing objective/domain/constraint checks: EXISTS; generic artifact metric
  registry: MISSING; objective-based perturbations: EXISTS; no-objective replay
  and explicit stage-owned scientific obligations: PARTIAL/MISSING.
- Benchmark: `benchmark/{executor,evaluation,reporting,repository,workflow}.py`;
  case manifests are `benchmarks/case-00{1,2,3}-mcm-2024-{c,b,a}/manifest.json`.
- Acceptance requires three attempted cases, two successes, live provider and
  literature, actual paper/package evidence, and all existing hard gates.
  Current manifests have no executable metric/scenario specifications. Do not
  invent a Case A stability threshold or infer Gurobi is mandatory.
- Actual PostgreSQL service/user/database: postgres/mathmodel/mathmodel; 54 tables,
  migration head `20260831_0011`. Existing verification tables are validation_runs,
  sensitivity_runs, robustness_runs, verification_experiments, red_team_reports,
  repair_cycles; artifacts/execution_records/results already have exact links.

## Files Affected / design

Add schemas and deterministic components under `verification`, additive ORM and
migration, thin existing-style API, benchmark AND-gate integration, tests and a
small Benchmark UI evidence panel. Reuse FileStore, SandboxExecutor, independent
AST evaluator, solver routing, database sessions and existing cryptographic hashes.
Use a strict canonicalization boundary for new records without changing historical
Paper digests. Explicit frozen plans bind exact attempt/result/artifact identities;
commands cannot submit code, host paths, imports, or substitute a latest result.
Unknown metrics/scenarios or missing plans remain not-ready. Heavy executions
occur outside transactions; persisted logical requests are idempotent and immutable.

## Implementation Steps

1. Implement strict MetricSpec/ScenarioSpec/plan/result contracts and allowlisted
   calculators, with raw arrays separated from reported metrics.
2. Reuse optimization checks; support explicit no-objective metrics without
   fabricating an objective. Implement bounded, input-digest-bound replay.
3. Add immutable storage/read-time integrity checks and idempotent command APIs.
4. Wire an additional independent-verification AND gate; never clear Phase 5/8
   failures or mutate verified_result_id merely because metrics pass.
5. Add minimal evidence visibility, adversarial/unit/API/real Docker/PostgreSQL
   tests, migration roundtrip only on a dedicated database, full regression/docs.

## Tests / acceptance criteria

False reported metrics, NaN/Inf, tampered artifacts, unknown keys, missing required
metrics/scenarios, stale source/version, replay input mismatch, wrong execution,
duplicate commands and skipped mandatory work cannot pass. Test exact and tolerant
comparison, fixed-seed replay, physical artifact hashes, persisted readback, safe
timeouts, and unchanged Phase 1-7 semantics. Run pytest --cov, Ruff format/check,
strict mypy, Alembic check, diff check and frontend checks. Extension completion
does not itself make the historical Case A or Phase 8 READY.

## Risks

Scientific metric/trajectory bindings and acceptance thresholds are not present in
historical Case A manifests. Generic infrastructure cannot legitimately fill these
with case-specific guessed targets. Legacy generated programs lacking an explicit
runtime input contract must not be replayed by string-replacing embedded constants.

## Implementation status

The generic extension is implemented: closed/versioned metric schemas and
calculators, exact raw-artifact/result binding, fixed-template dynamic replay,
deterministic solver replay, read-time code/input/artifact audits, immutable and
idempotent persistence, command/read APIs, a UI evidence panel, automatic binding
for future reviewed sidecars, and a Benchmark logical-AND gate. Synthetic real
Docker and isolated PostgreSQL evidence pass. Real Case A remains `NOT_READY`
because no reviewed scientific sidecar exists and its current formal result is
not fully verified; this plan does not authorize inventing that scientific policy.
