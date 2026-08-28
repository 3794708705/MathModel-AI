# Verification and model repair

Phase 5 determines whether a truthful Phase 4 result is independently credible,
stable under declared perturbations, and defensible against adversarial review.
It does not generate paper content or perform Phase 6 work.

## Workflow and gates

```text
SOLVE
  -> Independent Validation -> VALIDATE
  -> Sensitivity            -> SENSITIVITY
  -> Robustness             -> ROBUSTNESS
  -> Red Team               -> RED_TEAM
       | PASS
       ` Critical -> MODEL_REPAIR -> model vN+1 -> SOLVE -> full verification
```

Each step reads persisted prerequisites, emits a typed immutable report, appends
a `ProblemState` v5 revision, and applies a deterministic quality gate. A gate
returns `PASS`, `RETRY`, or `HUMAN_REVIEW`; later stages cannot reinterpret a
failed predecessor as successful. `SOLVE` proves computational provenance but
leaves the result `UNVERIFIED`. Only the terminal `VERIFIED` gate may set
`ProblemState.verified_result_id` and mark that exact formal result `VERIFIED`.

## Independent validation

`IndependentValidator` receives the exact persisted model, result, solver run,
execution, and Phase 4 evidence report. Its evaluator is separate from the
solver feasibility evaluator. It recomputes:

- variable presence, finite values, bounds, and discrete domains;
- every declared constraint relation and maximum violation;
- objective value and result-objective agreement;
- expected key outputs and result metadata;
- the complete persisted evidence chain; and
- supported model validation requirements.

The absolute and relative tolerances are configured by
`MM_VALIDATION_ABS_TOLERANCE` and `MM_VALIDATION_REL_TOLERANCE`. Unsupported
free-form validation requirements are `NOT_EVALUABLE`, never assumed to pass.
At the terminal gate, the validator recomputes the report again and compares all
deterministic checks with the persisted payload so post-validation tampering
cannot clear the final gate.

## Sensitivity experiments

Sensitivity is one-at-a-time over sourced finite scalar parameters. Defaults are
signed `±5%`, `±10%`, and `±20%`, up to five parameters and 30 runs. Callers may
select symbols, fractions, caps, and solver options through `SensitivityConfig`.
Python computes objective ranges, relative changes, and mean elasticities only
from successful experiments.

Every scenario clones the immutable model, changes only named parameter values,
computes a new executable model digest, and enters the normal solver path. A
scenario is `PASS` only with a successful non-Mock `ExecutionRecord`, a canonical
solver result, and independent feasibility. Partial or failed reports do not pass
the SENSITIVITY gate. Before final acceptance, Python reconstructs each scenario
from its perturbation metadata and verifies the exact model embedded in the
deterministic program, program/bundle/code hashes, execution record, solver
result, persisted relational columns, objective changes, and summary statistics.

## Robustness experiments

`RobustnessConfig` makes the method explicit:

- `SCENARIO_ANALYSIS`: apply each configured fraction to selected parameters;
- `WORST_CASE`: execute declared joint extremes and retain the adverse objective;
- `NOISE_PERTURBATION`: seeded bounded uniform parameter noise;
- `MONTE_CARLO`: seeded clipped Gaussian parameter noise; or
- `BOOTSTRAP`: currently `BLOCKED` because row resampling is not yet bound to
  dataset rows and parameter recomputation.

Runs are capped at 200 by schema and 50 by default. Reports contain feasibility
rate, objective mean/standard deviation/range/quantiles, and the worst scenario
identity. The ROBUSTNESS gate requires every declared run to be independently
feasible and executed.

## Red Team

`RedTeamAgent` reviews problem interpretation, assumptions, data use, model
structure, objective, constraints, algorithm, parameters, extreme cases,
overfitting, leakage, sensitivity, robustness, and result interpretation. Python
merges its structured findings with deterministic attacks generated from failed
validation, incomplete experiments, extreme sensitivity, and unresolved units.

Findings are `CRITICAL`, `MAJOR`, or `MINOR` and cite evidence plus affected
references. Any unresolved Critical finding makes the report fail and enters
`MODEL_REPAIR`. Major and Minor findings remain visible warnings. A Mock reviewer
always produces `INCONCLUSIVE` and `HUMAN_REVIEW`, even when its fixture contains
no findings. Severity counts are recomputed from unresolved findings at both the
Red Team and terminal gates; persisted count fields are not trusted.

The terminal gate additionally requires one model identity/digest, one formal
result ID, exact prerequisite report IDs, matching baseline objectives, complete
experiment designs and summaries, successful persisted execution audits, and a
non-Mock Red Team pass. `BLOCKED`, `NOT_EVALUABLE`, `INCONCLUSIVE`, or Mock
evidence yields `HUMAN_REVIEW`, never `VERIFIED`.

## Model repair loop

`ModelRepairAgent` receives the exact current model, Red Team report, validation,
sensitivity, and robustness evidence. It must return explicit repair actions that
reference every unresolved Critical finding. Deterministic gates require:

- unchanged stable `model_id` and exactly `version + 1`;
- a changed mathematical digest rather than a version-only no-op;
- all Critical finding IDs covered by scoped actions;
- a complete valid `MathematicalModel` that passes the Phase 4 MODEL gate; and
- non-Mock agent output.

Accepted repairs are new database records and new state revisions; source models
and prior models, solver runs, executions, and results are never overwritten.
Repair output cannot contain or modify a Result. The repaired model re-enters the real
Phase 4 solver, then VALIDATE, SENSITIVITY, ROBUSTNESS, and RED_TEAM. The maximum
automatic cycle count is configurable from one to three and defaults to three.
Exhaustion records a `MODEL_REPAIR` gate and sets state to `HUMAN_REVIEW`.

## Persistence and API

Migration `20260828_0006` adds `validation_runs`, `sensitivity_runs`,
`robustness_runs`, `verification_experiments`, `red_team_reports`, and
`repair_cycles`. Foreign keys bind reports to the exact model record, result,
execution, prerequisite reports, agent runs, and source/target model versions.

Project-scoped endpoints are:

- `POST /api/v1/projects/{id}/verification/validate`
- `POST /api/v1/projects/{id}/verification/sensitivity`
- `POST /api/v1/projects/{id}/verification/robustness`
- `POST /api/v1/projects/{id}/verification/red-team`
- `POST /api/v1/projects/{id}/verification/run`
- `POST /api/v1/projects/{id}/model/repair`
- `POST /api/v1/projects/{id}/model/repair-loop`
- `GET /api/v1/projects/{id}/validation-runs/latest`
- `GET /api/v1/projects/{id}/sensitivity-runs/latest`
- `GET /api/v1/projects/{id}/robustness-runs/latest`
- `GET /api/v1/projects/{id}/red-team-reports/latest`
- `GET /api/v1/projects/{id}/repair-cycles`

The aggregate verification endpoint intentionally stops at Red Team. It never
starts repair implicitly; callers must choose the repair endpoint or bounded
repair loop after inspecting Critical findings.

## Deliberate boundaries

Phase 5 does not claim model correctness from one test suite, invent data-backed
Bootstrap samples, treat Mock review as acceptance, or permit repair to mutate
historical models. Literature, citation validation, paper generation, rendering,
and submission checks remain Phase 6 and later.
