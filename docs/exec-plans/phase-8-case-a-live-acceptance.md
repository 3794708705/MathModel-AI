# Phase 8 Case A Live Acceptance

## Goal

Run a new blind MCM 2024 Case A benchmark with the accepted real DeepSeek
provider, retain every prior run, and stop B/C whenever a generic P0/P1 defect
is observed.

## Current State

Three historical runs and nine provider-preflight-blocked attempts are retained.
Fourteen live Case A runs are also retained. Successive generic fixes moved the
live pipeline through reasoning, MathModeler, generated code, and real Docker
execution. Rerun 13 passed SOLVE after a new immutable CodeAgent attempt, then
stopped at independent validation with NOT_EVALUABLE. No verified result, paper,
PDF, or package exists. A read-only audit found a keyword-based false requirement
PASS and undercounted formal solver calls; both are corrected for new audits and
outcomes. A generic versioned metric-recomputation and fresh scenario-replay
extension is now implemented and accepted with real Docker and isolated
PostgreSQL evidence. Case A remains blocked because it has no reviewed metric or
scenario policy and rerun 13 has no verified result; the implementation does not
infer scientific thresholds. Historical failures and process interruptions below
remain unchanged, and no benchmark was resumed.

A source-level policy review then created a non-production DRAFT bound to the
official PDF digest, manifest digest, and rerun-13 model digest. The review found
blocking missing definitions for initial state/time scale, a formal fixed-ratio
control, stability/eigenvalue evaluation, parameter ranges, resilience and
persistence criteria, the parasite ambiguity, and binding to any newly generated
model. Consequently no reviewed sidecar or fresh attempt was created.

## Files Affected

- `src/mathmodel_ai/agents/base.py`
- `src/mathmodel_ai/benchmark/executor.py`
- `src/mathmodel_ai/benchmark/repository.py`
- `src/mathmodel_ai/benchmark/reporting.py`
- `src/mathmodel_ai/benchmark/workflow.py`
- `src/mathmodel_ai/agents/math_modeler.py`
- `src/mathmodel_ai/agents/code.py`
- `src/mathmodel_ai/verification/validation.py`
- `src/mathmodel_ai/prompt_templates/code_agent.prompt`
- `src/mathmodel_ai/prompt_templates/math_modeler.prompt`
- `src/mathmodel_ai/prompt_templates/model_explorer.prompt`
- `src/mathmodel_ai/prompt_templates/model_jury.prompt`
- focused agent and benchmark regression tests

## Design

Permit one same-tier retry before escalation, aggregate usage from every issued
provider request, and retain the last executed route when a later route requires
human review. A Phase 1-7 pipeline failure must return a project-bound outcome
with real partial usage. Benchmark persistence independently recomputes provider
activity and critical-agent coverage. Reports for an active run keep the run in
`RUNNING` instead of manufacturing a terminal status. Data availability prompts
distinguish a genuinely unavailable input from literature-, derivation-, or
assumption-mitigated calibration. MathModeler requires explicit source-typed
baseline parameters and receives a 65,536-token budget because reasoning tokens
count against the OpenAI-compatible output limit. Its deterministic state-aware
gate runs inside the bounded agent attempt, and safe validation diagnostics from a
failed attempt are supplied to the next attempt instead of blindly repeating the
same prompt. The prompt requires a compact symbol-closed model, explicit treatment
of every targeted ambiguity, and exact solver-enum values. New live attempts bind
their project immediately so a process interruption cannot erase that provenance.
Generated programs likewise receive a bounded retry only after the failed
program, execution, solver run, result, and exact parser/gate diagnostics are
persisted. A failed SOLVE gate cannot be followed by VERIFY.

## Implementation Steps

1. Harden BaseAgent retry accounting and failed-call provenance.
2. Convert stage exceptions into project-bound failed benchmark outcomes.
3. Distinguish real partial provider activity (`FAIL`) from no-call preflight
   blocking (`BLOCKED`).
4. Add adversarial regression tests and rerun Case A only.
5. Correct the model-exploration/Jury data-gap contract without weakening hard
   failures.
6. Correct the MathModeler executable-parameter contract and increase only that
   stage's output budget.
7. Bind the benchmark project before long-running agent stages and conservatively
   close the interrupted run without claiming completion.
8. Move the deterministic model gate into MathModeler's bounded attempt loop,
   provide safe gate diagnostics to the retry, and require a compact closed-symbol
   draft before the workflow accepts it.
9. Preserve invalid generated-result schema diagnostics, retry generated solves
   through new immutable program/execution/result records, and prohibit VERIFY
   from starting until the SOLVE gate passes.
10. Preserve safe JSON syntax locations from provider failures, feed the error
    into ModelExplorer's existing bounded retry, and request concise JSON with
    correctly escaped strings. Keep strict schema parsing and all gates intact.
    Test malformed JSON rejection, safe diagnostics, retry feedback, and input
    immutability before another live Case A attempt.
11. Clarify MathModeler's existing constant and equation-reference contracts in
    schema descriptions and prompt 4.3. A constant keeps `parameter_id`; an
    equation's `dependency_refs` points only to registered equations. Test that
    real constants resolve while invented constant fields and constants used as
    equation dependencies still fail. Do not weaken the registry.
12. Reject exploration sets with fewer than two candidates free of mandatory
    MISSING data, using the same predicate as the existing Jury selector. Run
    the EXPLORE gate inside the bounded Agent attempt so feedback reaches the
    component that owns the candidate designs. Prompt 2.3 distinguishes required
    inputs of the proposed implementation from optional richer-model data.
    Never relabel unavailable observations or relax the downstream Jury gate.
13. Require two end-to-end alternatives for the existing single-primary
    workflow. EXPLORE checks jointly for full subproblem coverage and usable
    inputs; deterministic scoring disqualifies partial candidates; SELECT
    independently rechecks the chosen primary and backup against original
    requirements. Update prompts to compose complementary modules inside one
    candidate chain. Test complementary partial sets, optimistic scores, and
    tampered eligibility before rerunning A. Mathematical/Solver semantics and
    historical records remain unchanged.
14. Audit rerun 13's persisted solve and independent validation separately. Replace
    keyword-based validation-requirement acceptance with exact supported contracts;
    compound contracts must check every component, and scientific prose remains
    UNCHECKED without an implemented evaluator. Add adversarial regressions for
    eigenvalue/variable, objective, constraint, and evidence keyword collisions and
    stale report acceptance. Count formal solver calls from persisted execution
    links instead of resetting failures to zero or assuming one successful solve.
    Clarify the existing flat generated-result metrics contract. Preserve every
    historical record and do not rerun an unsupported scientific pipeline blindly.

## Tests

- focused BaseAgent and benchmark tests
- relevant provider/reasoning regression
- full regression after JSON diagnostics/retry correction: 543 passed,
  10 environment-gated skips, coverage 86.96%
- Ruff format/check, strict mypy, Alembic check, and git diff check passed
- isolated migrated PostgreSQL integration: 2 passed
- after constant/equation contract clarification: 92 focused regression tests
  passed; Ruff, strict mypy, Alembic and diff checks passed. The prior full run
  remains 543 passed / 10 skipped; the added namespace regression also passed.
- complete regression after constant/equation clarification: 544 passed,
  10 skipped, coverage 87.03%, with temporary JUnit/coverage XML retained.
- after upstream data-feasibility gate: 57 focused tests passed, including
  bounded retry and unchanged missing-data disqualification; full regression:
  549 passed, 10 skipped, coverage 87.04%.
- after complete primary/backup coverage checks: 61 focused tests passed;
  full regression: 553 passed, 10 skipped, combined coverage 87.05%.
  Ruff, strict mypy, Alembic, and diff
  checks passed.
- after the independent-requirement and execution-count audit: 114 focused tests
  passed. A fresh-process read-only audit of the real rerun 13 changes the former
  false Jacobian PASS to UNCHECKED, detects the stale report, counts two formal
  executions, and confirms the stored report/run are unchanged. Final-tree full
  regression: 566 passed, 10 environment-gated skips, one non-blocking Starlette
  deprecation warning, 87.09% combined coverage (259.32 seconds). Ruff format/check,
  strict mypy (168 files), Alembic check, and git diff --check passed. Durable
  JUnit and coverage XML are in the temporary backend-log directory with the
  `regression-20260906-final-audit` prefix; they are not repository artifacts.
  Real Docker sandbox/solver/verification-repair/PDF regression ran. The paid
  provider, literature opt-in, Gurobi-license, and separately configured PostgreSQL
  tests remain honestly skipped in this process; live provider evidence from the
  benchmark and previous dedicated PostgreSQL acceptance are separate records.

## Risks

- Retrying large structured output consumes additional live tokens.
- A valid software fix may reveal a separate model-tier or capability blocker;
  that blocker must remain explicit rather than weakening routing gates.
- The generic Phase 5 extension can independently recompute declared metrics and
  run exact fresh dynamic or optimization scenarios. It cannot decide which
  metrics, tolerances, or scenarios are scientifically valid for Case A. A
  reviewed versioned case policy is still required; do not substitute an
  artificial objective, infer thresholds, or delete unfulfilled requirements.
- Some model obligations belong to later experiment stages, but the current
  VALIDATE gate requires every free-form obligation to pass before sensitivity
  can run. A future stage-owned obligation contract must distinguish permission
  to run an experiment from final scientific verification; pending work can
  never silently become a final PASS.
- Existing partial-failure experiment/repair counters still use conservative
  defaults; full success counts use completed reports/loop outcomes. This audit
  fixes formal solver retries specifically, not general partial experiment
  accounting. No experiments or repairs occurred in rerun 13.
- This audit used fresh Python processes without restarting the credentialed
  backend. Before a new live attempt, restart the backend normally to load the
  new validator/prompt; the existing process must not be mistaken for the updated
  implementation.

## Acceptance Criteria

- prior runs and attempts remain immutable
- failed live requests retain model identity, retries, usage, and project link
- Case A rerun is not mislabeled `BLOCKED_ENVIRONMENT` after a real call
- no Case B/C run starts
- no commit is created

## Current Outcome

- Case A has not reached a stable full-pipeline terminal result; Phase 8 remains
  `NOT_READY`.
- The completed live run `c98b0faa-e463-4dc7-9ebd-012b1977590d` reached SELECT,
  then failed because both MathModeler responses ended with `finish_reason=length`
  at exactly 32,768 output tokens.
- The next run `2f64b19b-4042-427f-9b98-bab546da56cd` lost its host process and is
  retained as `NOT_READY`; its attempt is `BLOCKED_ENVIRONMENT`, not PASS.
- The credentialed run `6c79803c-4046-4b0a-8336-b1a8ca1a6cc0` retained live
  provider/literature evidence and stopped at a generic P1 MathModeler gate
  failure. Case B/C remain unstarted while that issue is fixed and regressed.
- Run `2ba466c3-6c8c-4c26-9b22-03fd07ee76a5` reached a passing MathModeler after
  deterministic retry, then its host process ended before CodeAgent persisted;
  it is conservatively `BLOCKED_ENVIRONMENT` with no backfilled metrics.
- Run `e79da65b-dabb-4ee0-b5e2-e1ac537ef4d8` reached a real non-Mock CodeAgent
  and successful sandbox execution. Its generated `result.json` used informal
  solver/status strings and a nonnumeric variable value, so schema validation
  correctly produced `EXECUTION_ERROR`. The workflow then exposed a generic
  MODEL-to-VALIDATE transition bug; both defects are fixed before Case A is
  rerun. B/C remain unstarted.
- Run `a8fbe6b4-dccc-4a35-9017-8cf663c696b7` passed ProblemAgent but both
  ModelExplorer responses failed JSON parsing (42,986 and 48,927 characters,
  `finish_reason=stop`). It remains FAIL; raw content was not retained, so the
  exact syntax defect is unknown. Add syntax diagnostics and corrective retry
  guidance rather than accepting or guessing malformed output.
- Rerun 9 (`1d9bb26f-ba4e-48f2-b36b-0fab95697506`) passed ProblemAgent,
  ModelExplorer 2.2, and ModelJury without retry. MathModeler first used constant
  IDs as equation dependencies, then invented `constant_id` on retry. It reached
  HUMAN_REVIEW and the attempt is FAIL, with live literature PASS. Registry
  inspection confirmed the gate was correct; improve contract clarity only.
- Rerun 10 (`f9516b6a-8313-4ccc-9f7c-d46a7d83842f`) lost its local backend
  and PostgreSQL before any AgentRun was persisted. It is retained as
  BLOCKED_ENVIRONMENT / NOT_READY, with no backfilled metrics or claimed token
  usage. Services were restored on September 6; rerun 11 will use the same
  corrected contracts and a new project. Regression output is additionally
  written to timestamped temporary JUnit/coverage reports for recovery.
- Rerun 11 (`0d6de680-d02b-4e69-8a35-7fc834556370`) proved live JSON
  correction: ModelExplorer's first response failed at line 409, column 7;
  its second response parsed successfully. However, all five candidates required
  MISSING data. EXPLORE incorrectly let an unselectable set reach Jury, where
  both attempts necessarily failed the two-eligible-candidate rule. The run is
  retained as FAIL / NOT_READY, with literature PASS. Fix the upstream gate and
  feedback ownership, retaining the missing-data disqualification.
- Rerun 12 (`b38d6c0f-1979-40e2-b355-3a3bf8950a74`) passed the three
  reasoning Agents without retry, but independent inspection found a P1:
  the selected primary omitted Q3/Q4/Q6 and the backup omitted Q1/Q2/Q5.
  Jury explicitly called them complementary modules. The host was stopped
  during MathModeler; the run is retained as FAIL / NOT_READY with a
  MODEL_SELECTION_COVERAGE_AUDIT failure and no backfilled in-flight usage.
- Rerun 13 (`104df85e-8f04-46fe-a7ff-96c744fae9b5`) completed two real
  generated executions. The first emitted nested metrics rejected by the flat
  schema; bounded CodeAgent retry produced a FEASIBLE result and passing SOLVE
  gate. Independent validation then correctly blocked final verification as
  NOT_EVALUABLE: no scalar objective/metric recalculation and unsupported dynamic,
  scenario, and sensitivity requirements. An additional P1 was found: a Jacobian
  eigenvalue requirement was incorrectly marked PASS because its text contained
  "variable". The final gate still blocked the result. Formal solver_calls was
  also recorded as zero despite two persisted executions. Historical metrics and
  reports remain immutable; fixes apply only to new audits/runs. No verified
  result, paper, PDF, or package was created.
- The later independent-verification extension does not alter rerun 13. It adds
  deterministic metric recomputation, fresh scenario execution, immutable
  evidence binding, PostgreSQL persistence, an AND-only benchmark gate, and a
  read/command UI. All current case manifests still lack the reviewed sidecar,
  so Case A and Phase 8 remain NOT_READY without a new benchmark attempt.
- The subsequent Case A contract-completion task archived the original DRAFT,
  created a formal model v2, and promoted a reviewed source/model-bound policy
  after independent Red Team (`CRITICAL=0`) and Model Jury (`PASS`). The policy
  defines 19 future replay scenarios but does not execute them here. Rerun 13
  remains unverified and no new BenchmarkRun, attempt, AgentRun, execution, or
  result was made.
- Read-only history check after the audit: 17 runs and 23 attempts retained,
  zero RUNNING benchmarks. These include all three original blocked runs/nine
  attempts and fourteen live Case A attempts. No B/C live rerun or commit was
  created in this continuation.

## Fresh Case A Reviewed-Contract Rerun

### Goal

Create exactly one new Case A run from fresh reasoning while binding the
reviewed MathematicalModel v2 before CodeAgent/SOLVE, execute the reviewed
independent policy before any VERIFIED promotion, and continue to paper and
submission only when both independent and Phase 5 gates pass.

### Current State

The reviewed model and policy are valid, but the generic live executor still
generates and solves a new MathModeler model before it invokes independent
verification. Its independent runner is currently called only after an older
verification path has already produced `verified_result_id`. This ordering is a
P1 provenance defect for reviewed-contract benchmarks and must be corrected
before attempt 18 is created.

### Files Affected

- reviewed-policy/model registry and validation
- mathematical workflow deterministic reviewed-model binding
- benchmark executor ordering and live-agent coverage
- application dependency wiring
- focused registry, model-binding, executor-order, and provenance tests

### Design

Load a reviewed model only from the bounded case sidecar, validate its model,
contract, policy, Red Team, Jury, manifest, and official-problem digests, then
rebind only excluded identity fields to the fresh project. Persist that action
as a deterministic `reviewed_model_binder` audit run; it is neither an LLM call
nor a Mock. For this path, the live critical-agent set replaces MathModeler with
CodeAgent because mathematical meaning comes from the reviewed immutable
contract. Run all independent metrics and replays immediately after the new
formal solve. Enter the existing Phase 5 workflow only after the independent
report is PASS, and reject any later model-digest change as a stale policy.

### Implementation Steps

1. Add fail-closed reviewed model/contract loading and digest checks.
2. Add deterministic identity-only model binding and preserve the exact v2
   mathematical digest.
3. Execute independent verification before Phase 5 VERIFIED promotion.
4. Require independent result identity and exact required metric/scenario
   completion before downstream paper work.
5. Add adversarial regressions, run focused and full checks, back up/migrate the
   local PostgreSQL database, then create one fresh Case A run only.

### Tests

- reviewed contract tamper and digest mismatch rejection
- identity rebinding preserves mathematical digest
- reviewed binder cannot count as a live provider call
- CodeAgent replaces MathModeler only for an audited reviewed binding
- independent runner precedes Phase 5 verification and paper
- model digest changes invalidate the reviewed policy
- full backend/frontend, PostgreSQL, Docker, static, and migration checks

### Risks

- Live provider output can still fail a strict reasoning or paper contract.
- Independent replays may expose a model or implementation defect; failures
  remain terminal evidence and cannot be downgraded to warnings.
- A Phase 5 repair that changes mathematical meaning invalidates the reviewed
  policy and stops downstream work.

### Acceptance Criteria

- Historical 17 Case A attempts remain byte-semantically unchanged.
- One new attempt uses the official PDF and exact reviewed model/policy digests.
- Baseline solve and every replay have distinct non-Mock executions.
- No VERIFIED result exists before independent metrics and all 19 scenarios pass.
- B/C remain unstarted and no commit is created.

### Fresh Attempt 18 Outcome

- Run `a1c4c1a4-d39b-4588-aee2-81044567edca` and attempt
  `f3e82da8-fe74-485e-b078-a690aeb43f29` were created from the hash-pinned
  official Case A PDF. The historical Case A attempt count advanced from 17 to
  18; no historical row was deleted, overwritten, or promoted.
- ProblemAgent, ModelExplorer, and ModelJury completed through the real
  `deepseek-main` route with non-Mock AgentRuns. Live Crossref literature
  verification passed.
- The reviewed registry revalidated model digest
  `16c3f2922ab8612e6e7ecee855b173f94c321b100b5d45be57e3546510e4bf0d`,
  model-contract digest
  `72d611664d6584299642a33a9f59ae222af32145e9153dbd5b3d3ea69dbe9d2e`,
  and policy digest
  `ba03db8d10f54dcd89dee511cc19ae820c54f34a0bfe70ca98d7acafb3ae4b87`.
- Binding stopped fail-closed at the MODEL gate. The reviewed contract targets
  `Q1` through `Q5`, while the fresh accepted ProblemState uses the semantic
  identifiers `Q-SEXRATIO-FUNCTION`, `Q-POPULATION-DYNAMICS`,
  `Q-ECOSYSTEM-IMPACT`, `Q-ADAPTIVE-VALUE`, and `Q-OTHER-SPECIES`.
  `MODEL_GATE_FAIL:target_subproblems_known` prevents treating these identities
  as equivalent without a reviewed mapping or a new model/policy version.
- No MathematicalModel, generated program, execution, solver result,
  independent-verification plan, verified result, paper, PDF, submission,
  freeze, or package was created for this project. B/C were not run in this
  task. The terminal status is `CASE_A_FAIL`; Phase 8 remains `NOT_READY`.

## Fast-Path Case A Completion Plan

### Goal

Complete one new Case A pipeline end to end after a single full-chain preflight
and batch repair pass. Preserve all 18 prior attempts, use only the eligible
real provider, and never weaken a mathematical, evidence, or security gate.

### Current State

Attempt 18 proved that official input, live literature, real reasoning, and all
reviewed digests are available. It stopped before model persistence because the
reviewed model's ordinal target identifiers and the fresh ProblemState's
semantic identifiers lack an explicit versioned identity binding.

### Files Affected

- canonical subproblem identity schemas, normalization, and persistence
- ProblemAgent/ProblemState and reviewed MathematicalModel binding
- verification, paper, Final Jury, and submission coverage resolvers
- benchmark preflight/executor and focused adversarial tests
- Phase 8 execution and acceptance documentation

### Design

Introduce an explicit versioned canonical identity contract with exact source
bindings and deterministic aliases. Resolve identifiers only through that
contract; never infer similarity from labels. Preserve old IDs as audit aliases,
keep mathematical semantics/digests unchanged when only metadata is normalized,
and fail closed on missing, duplicate, or cross-namespace mappings. Preflight
every downstream dependency before creating the next attempt.

### Implementation Steps

1. Inventory P0-P4 blockers across problem, model, verification, paper, jury,
   submission, Docker, PDF compilation, and package creation.
2. Fix all known P0/P1 identity and contract blockers generically in one batch.
3. Add cross-stage adversarial regressions and run focused tests.
4. Re-run the complete static/runtime preflight, then create one fresh Case A
   attempt and continue through every eligible stage.
5. Batch-repair recoverable code/model/paper defects within existing limits,
   run full regression once, and publish one final acceptance report.

### Tests

- canonical identity resolution, versioning, ambiguity, and tamper rejection
- legacy ordinal to canonical binding without fuzzy authorization
- model/verification/paper/jury/submission coverage invariance
- reviewed model digest preservation and stale-policy rejection
- real PostgreSQL, Docker solver/replay, PDF render, package roundtrip
- complete backend/frontend/static/migration regression

### Risks

- A true equation, parameter, comparator, or scientific interpretation change
  requires a new reviewed model/policy rather than metadata normalization.
- Real provider output may expose additional contract failures; collect and
  repair same-class failures in batches without exceeding bounded retries.
- Required verification scenarios can legitimately fail and must remain
  blockers rather than warnings.

### Acceptance Criteria

- all 18 historical Case A attempts remain immutable
- a new attempt uses explicit canonical subproblem bindings and no Mock
- all required independent metrics and 19 scenario executions are complete
- only a fully verified result drives evidence, paper, PDF, jury, and package
- no B/C run and no commit is created
