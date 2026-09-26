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
Validator `5.1.0` resolves only exact supported legacy contracts (case/whitespace
normalization is allowed): `recompute variable bounds`, `recompute every
constraint`, `recompute variable bounds and constraints`, `recalculate objective
metric`, and `verify evidence trace`. Combined contracts require all components;
an empty component remains unchecked. Words such as "variable", "constraint",
"objective", or "evidence" inside a scientific request never prove it. Eigenvalue
stability, empirical prediction accuracy, literature support, and scenario or
resampling requirements need their own implemented evidence checks; relabeling
them as a basic bounds check is not a repair. Existing reports remain immutable;
a new audit recomputes requirements and detects old false-PASS reports.
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

For an objective-free model with non-indexed, uniquely defined scalar derived
outputs and no state variables, the program instead holds the formal result's
decision values fixed and perturbs sourced parameters. The versioned sandbox
evaluates each complete response equation, while a separate host evaluator
recomputes every output from the scenario model. Reports contain response ranges,
not a fabricated optimization objective or objective elasticity. The integrity
audit binds the fixed point, exact scenario payload, output values, code hash,
execution ID, and non-Mock status. Missing/cyclic equations, mismatched formal
baseline outputs, or uncheckable responses fail closed. This tests the formal
scalar response model; it does not establish holdout predictive superiority.

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
for optimization models. Objective-free scalar response models report response
ranges and feasibility only; `WORST_CASE` in that mode is a declared joint
scenario comparison, not an objective-ranked worst case.
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
For a benchmark with a causal holdout, each accepted repaired model must use a
new GENERATED solve and a new isolated predictor execution from that exact
program. Its fresh execution and formal result IDs are re-audited before
VALIDATE, SENSITIVITY, ROBUSTNESS, and RED_TEAM repeat. An unsuccessful solve,
missing predictor, stale trace, or rejected repair stops the chain; no previous
holdout evidence or verified result is transferred to the new revision.

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

## Independent metric recomputation and scenario replay

The Phase 5 extension does not trust Agent prose, solver aggregate metrics, or a
persisted PASS/count. `MetricSpec` version 1 selects a closed calculator registry.
MAE, RMSE, R2, maximum error, bounded series summaries, objective, feasibility,
maximum hard-constraint violation, and MIP gap are recomputed from immutable raw
artifacts and the exact MathematicalModel AST. Undefined R2, absent observations,
non-finite values, unsupported versions, unevaluable constraints, and missing
solver bounds fail closed.

The narrow `algebraic_scalar` primitive resolves one declared derived-variable
symbol by evaluating only the model's typed, allowlisted expression AST. It
starts from exact model parameters/constants plus raw decision/state values,
ignores raw or reported derived values, rejects duplicate/cyclic/incomplete
definitions, and never accepts source text, `eval`, dynamic imports, or
benchmark-specific branches.

`ScenarioSpec` version 1 binds explicit parameter/decision changes, optional
seeded noise, resource limits, required metrics, and—when needed—an explicit
state-to-derivative-equation map. Every scenario creates a fresh non-Mock Docker
execution with network disabled, non-root user, read-only root, CPU/RAM/PID/time
limits, stdout/stderr, image identity, code hash, bundle hash, output artifact,
and input/scenario digests. Reads reconstruct the expected source and parameters,
rehash physical artifacts, recompute all metrics, and reject reused executions.

Reviewed requirements live only in evaluation-side
`benchmarks/case-*/independent-verification.json` sidecars and bind the existing
solve-manifest digest, exact mathematical-model digest, metric model bindings,
and—when supplied—the official problem artifact digest and independent contract
review digests. Every metric carries calculation/input/unit/tolerance provenance;
every scenario declares its exact changed inputs and review criterion. The
Web/API cannot create or weaken these requirements.
When a sidecar exists, the formal benchmark binds it automatically to the exact
`verified_result_id` and its `result.json`, then runs every declared obligation.
If it is absent, stale, or lacks an unambiguous observation file, the additional
gate is `NOT_READY`; no case-specific threshold is inferred.

Migration `20260906_0012` adds immutable `independent_verification_plans` and
idempotent `independent_verification_jobs`. The API exposes only read and command
operations:

- `GET /api/v1/benchmarks/attempts/{id}/verification`
- `POST /api/v1/benchmarks/attempts/{id}/verification/recompute`
- `POST /api/v1/benchmarks/attempts/{id}/verification/scenarios/{scenario}/replay`

This is an additional logical-AND gate. It cannot set `verified_result_id`, clear
existing Phase 5 failures, repair a model, or override provider, literature,
paper, submission, security, and benchmark gates.

### Group-held-out binary prediction evidence

For a benchmark with `causal-holdout-v1.json`, the trusted host keeps the
complete official CSV. Modeling and solving see training groups only; the
generated predictor receives one pre-outcome feature at a time in an isolated
container. The auxiliary execution records the exact formal result, source
digest, and whole science-policy digest. Formal VALIDATE re-reads the training
file and stored artifacts, replays the official labels and group split, and
recalculates Brier loss and its pre-outcome baseline. Red Team audits the same
persisted evidence again; a repaired formal result cannot reuse the old trace.

The policy may also list host-owned scientific checks, bound to the official
problem-file digest and included in the solve input digest. These checks are
present even if an agent omits them from `validation_requirements`. The current
binary evaluator can independently assess held-out prediction and calibration.
Calibration uses ten fixed-width probability bins and descriptive expected
calibration error, `sum_b |sum_predicted_b - sum_observed_b| / N`; it asserts
neither good calibration nor an improvement over baseline. The host also
computes a training-only conditional-randomness reference: within each match
and condition, 499 source-seeded permutations preserve outcome counts and
compare lag-one adjusted residual products. Its source digest, statistic,
simulation count and descriptive two-sided probability enter validation
evidence. If—and only if—the exact persisted formal `result.json` reports the
named statistic, p-value and simulation count, the validator compares all
three against this training-only reference. A missing claim remains
`UNCHECKED`; a mismatched claim fails. Neither a matching test nor a small
p-value establishes psychological or causal momentum.
The seeded shuffles visit matches and then conditions in their first-seen CSV
order; sorting groups changes the random sequence and is not an equivalent
implementation. Causal code generation rejects Pandas `groupby` without an
explicit `sort=False` before execution. For these causal cases, the modeler
also rejects validation requirements lacking an exact current-stage contract
before solving; downstream sensitivity and scenario obligations remain in the
later experiment stages and limitations, not as fictitious VALIDATE passes.
A second training-only reference reconstructs a pre-outcome rolling mean of condition-adjusted
residuals for every row. Its `match_flow` check passes only when the same
row-aligned series is present in the exact formal `result.json` and agrees
pointwise. This does not prove that a paper visualizes or explains the flow.
The one-point imminent-swing protocol defines an event as a strict sign
reversal of that rolling flow after the next point, excluding positions without
a full prior window. Before revealing the point outcome, the trusted host
transforms the isolated model's point probability into an event probability by
evaluating both possible outcomes. It then scores those sealed forecasts and a
condition-adjusted baseline on held-out matches. The check requires the model's
exact formal `result.json` to declare the protocol; absence remains
`UNCHECKED`, and another protocol fails. This is a narrow, one-point event,
not evidence of all momentum swings or predictive improvement. Associated
factor analysis and paper visualizations remain separate obligations.

For causal cases without a fixed-model reviewed sidecar, the aggregate
verification may continue only if all five host-owned scientific checks, all
model-declared validation requirements, the source-bound policy reference,
and the formal validation gate pass. Sensitivity, robustness, and Red Team
retain their existing gates and independent re-audit; an incomplete host policy
still fails closed. Unknown agent-declared requirements
likewise remain `UNCHECKED`; no wording or keyword match upgrades them to PASS.

The aggregate verification endpoint intentionally stops at Red Team. It never
starts repair implicitly; callers must choose the repair endpoint or bounded
repair loop after inspecting Critical findings.

## Deliberate boundaries

Phase 5 does not claim model correctness from one test suite, invent data-backed
Bootstrap samples, treat Mock review as acceptance, or permit repair to mutate
historical models. Literature, citation validation, paper generation, rendering,
and submission checks remain Phase 6 and later.
