# Solver architecture

Phase 4.1 executes one of two explicit paths after the MODEL gate:

```text
MathematicalModel -> AlgorithmSelector -> ExecutionStrategySelector
  |-> DETERMINISTIC -> SolverRouter -> BaseSolver adapter ---------|
  `-> GENERATED -> CodeAgent -> GeneratedProgram ------------------|
                   -> SandboxExecutor -> strict result.json parser
                   -> SolverResult -> EvidenceIntegrityVerifier -> SOLVE gate
                   -> SolverRun + ResultRecord
```

## Selection and support

`AlgorithmSelector` uses declared family and variable domains—not problem-text
keywords. Preferences are settings-backed and produce a recordable
`AlgorithmPlan` with reason, requirements, alternatives, complexity notes, and
numerical risks.

| Model family | Default ordered path | Phase 4 scope |
|---|---|---|
| LP | SciPy HiGHS, Gurobi | Flattened continuous linear models |
| MILP | Gurobi, SciPy MILP, OR-Tools | Linear mixed/integer models; CP-SAT only when all variables and coefficients are integral |
| Integer/CP | OR-Tools, Gurobi, SciPy MILP | Finite integral bounds for CP-SAT |
| NLP | SciPy minimize | Basic bounded continuous SLSQP path |

Reserved statistical, time-series, graph, flow, simulation, multi-objective, and
other families remain representable, but unsupported adapters return an explicit
plan/unavailability rather than inventing code.

`BaseSolver` exposes `supports`, `solve`, `health_check`, and capabilities. The
router applies hard filters for adapter registration, model family/domain and
declared capabilities, installed module/image, commercial-license policy, and
translation support. It then scores configured/model preferences, configurable
`TINY`/`SMALL`/`MEDIUM`/`LARGE` size policy, and deadline pressure. The recorded
decision contains scores, hard rejections, alternatives, size metrics, reason,
and runtime budget. Model maximum runtime, caller limit, and deadline cap are
resolved to the minimum and propagated into the actual `SolverOptions`.

## Adapters and canonical status

SciPy uses `linprog(method="highs")`, `milp`, and `minimize(method="SLSQP")`.
OR-Tools uses single-worker CP-SAT for reproducibility and rejects non-integral
coefficients without a declared scaling plan. Gurobi supports linear continuous
and integer paths only when a configured image contains `gurobipy` and a
read-only runtime license exists.

Native outcomes map to project statuses: `OPTIMAL`, `FEASIBLE`, `INFEASIBLE`,
`UNBOUNDED`, `INFEASIBLE_OR_UNBOUNDED`, `TIME_LIMIT`, `ITERATION_LIMIT`,
`NUMERICAL_ERROR`, `MODEL_INVALID`, `SOLVER_UNAVAILABLE`, `EXECUTION_ERROR`, and
`UNKNOWN`. LP and SLSQP mappings are separate. A feasible incumbent under a
limit remains limited; `FEASIBLE` never becomes `OPTIMAL`.

## Direct adapter and CodeAgent paths

Supported standard models use a deterministic translator. It serializes the
typed model/options as JSON into a small hashed Python entrypoint that invokes a
trusted, versioned runtime package. No numerical answer or solver boilerplate is
authored by an LLM.

`CodeAgent` is the formal second path for custom preprocessing, simulation, or
logic outside deterministic coverage. It receives an accepted immutable model
and plan and returns typed source files, approved dependencies, entrypoint,
explanation, and aggregate hash. Dynamic installation is forbidden; unavailable
packages return `DEPENDENCY_UNAVAILABLE`. Its machine output is only
`/output/result.json`; stdout is never parsed as a numerical result. Host code
recomputes constraints, bounds, domains, and objective from returned variables.
An explicit `GENERATED` request cannot silently fall back to a deterministic
adapter, while `AUTO` prefers deterministic support.

## Sandbox and optional license

The default `mathmodel-ai-solver:phase4` image contains NumPy, SciPy, OR-Tools,
and the deterministic runtime. Execution uses an inspected immutable image ID,
no network, UID/GID 65532, read-only root/workspace, dropped capabilities,
`no-new-privileges`, CPU/RAM/swap/PID/wall-clock limits, bounded logs, and bounded
tmpfs artifacts. A completion marker lets the host validate/export tmpfs while
the container is alive; the container is then force-removed.

Gurobi is not in the default dependency path. A separately configured image and
license file are probed at runtime. The file is mounted read-only under
`/run/secrets`; only its count is audited. Contents and host path are never stored
in the execution record.

## Feasibility and SOLVE gate

Phase 4 recomputes declared decision-variable bounds, integrality, and all
implemented constraint relations from returned values. It records maximum
violation, violated constraints, bound violations, tolerance, and whether a
candidate could be checked.

The SOLVE gate calls the same `EvidenceIntegrityVerifier` as the persisted API.
It checks canonical status/result agreement, reciprocal references, solver
identity, tolerance-aware objective/key outputs, exact model version/digest,
non-Mock successful execution, entrypoint and complete source-bundle hashes,
required variables/objective, truthful flags, and recomputed feasibility.
Failures are persisted as attempts and return to MODEL with `RETRY`.
For a generated-program path, the combined mathematical run may make at most
three persisted solve attempts. Each retry receives bounded deterministic gate
and result-schema diagnostics, creates a new CodeAgent run/program/execution/
solver-result chain, and leaves every failed chain immutable. Verification is
never entered unless the final SOLVE gate is `PASS`; exhaustion raises an
explicit quality-gate error. Invalid `result.json` content is not normalized or
silently repaired: field-level schema errors are retained for audit and retry.

Phase 5 experiment runs reuse this exact router, adapter, sandbox, canonical
status, and feasibility machinery on perturbed immutable model copies. They do
not call a lighter in-process calculator. Each scenario has its own model digest,
solver result, non-Mock `ExecutionRecord`, and independent Phase 5 feasibility
check. Repair solves enter from `MODEL_REPAIR` and return there on failure.

## Persistence and API

Migration `20260826_0004` adds the Phase 4 registries. Additive migration
`20260827_0005` adds model digests, execution origin, generated-program agent
links, entrypoint/bundle execution hashes, and persisted routing decisions.
PostgreSQL foreign keys link each result to one model record, solver run, and
execution. The evidence endpoint verifies both relational and JSON payload
copies and returns structured error codes for any mismatch.

Phase 4 endpoints are:

- `POST /api/v1/projects/{id}/model/build`
- `GET /api/v1/projects/{id}/mathematical-model`
- `POST /api/v1/projects/{id}/solve`
- `POST /api/v1/projects/{id}/mathematical/run`
- `GET /api/v1/projects/{id}/generated-programs`
- `GET /api/v1/projects/{id}/solver-runs`
- `GET /api/v1/projects/{id}/results`
- `GET /api/v1/projects/{id}/results/{result_id}/evidence`
