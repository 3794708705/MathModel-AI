# Phase 4 — Mathematical and Solver Core

## Goal

Deliver the first truthful `SELECT -> MODEL -> SOLVE` loop: a selected model is
converted once into a strongly typed, solver-independent mathematical model,
checked deterministically, routed to a compatible real solver, executed through
the existing isolated sandbox, and persisted with an unbroken model-version,
solver-run, execution-record, artifact, and result evidence chain.

## Current State

- Baseline: `384f322a9b1b5fb39d494f3912d9c19c6bf4061d`.
- Phase 1–3 tests pass (`67 passed, 1 skipped`) with 89.66% coverage.
- `ProblemState` v3 is immutable and currently ends the main workflow at
  `SELECT`; the independent data workflow ends at `EXECUTION`.
- `BaseAgent`, `ModelRouter`, provider adapters, PostgreSQL repositories,
  immutable local artifact storage, and `SandboxExecutor` are established.
- Solver packages, mathematical registries, canonical solver results, and
  result provenance do not yet exist.

## Architecture Changes

- Add a versioned `MathematicalModel` domain contract with a machine-readable
  expression tree. Solver SDK objects and executable code are forbidden in this
  contract.
- Add deterministic symbol, parameter, equation, and unit registries plus MODEL
  and SOLVE quality gates.
- Add `MathModeler` and `CodeAgent` as normal `BaseAgent` implementations using
  one versioned structured-generation call and Pydantic validation.
- Add `AlgorithmSelector`, capability-based `SolverRouter`, and adapters for
  SciPy, optional Gurobi, and OR-Tools CP-SAT.
- Reuse `SandboxExecutor` with a separate versioned solver image; this is another
  configured instance of the existing execution boundary, not a second execution
  implementation.
- Add a mathematical repository and four coarse-grained relational registries:
  mathematical models, generated programs, solver runs, and results. Symbols,
  parameters, and equations remain versioned inside model JSONB to avoid an
  unjustified table explosion.
- Advance only `SELECT -> MODEL -> SOLVE`. Failed solve attempts remain
  traceable and cannot be represented as successful state transitions.

## Schemas

- Mathematical: model identity/version, sets/indices, variables and domains,
  parameters and provenance, objective, constraints, equations, assumptions,
  units, data/evidence bindings, requirements, outputs, limitations, confidence,
  and status.
- Registries: canonical symbol definitions, equation dependencies and versions,
  duplicate/conflict/undefined/unused diagnostics.
- Units: parsed `UnitExpression` with base-dimension exponents, scale, and unknown
  tokens; checking returns `PASS`, `FAIL`, or `UNKNOWN`.
- Programs: source files, entrypoint, dependencies, solver target, hashes,
  generator/prompt metadata, status, and mock flag.
- Solvers: capabilities, health, algorithm plan, options, canonical statuses,
  solver result, feasibility report, solver run, and result record/ref.
- State v4: exact mathematical-model, generated-program, solver-run, and result
  references; full large solver results remain in their independent registries.

## Solver Strategy

- LP: SciPy `linprog(method="highs")` by default; optional Gurobi alternative.
- MILP: configured preference with Gurobi fallback to SciPy `milp`; OR-Tools is
  eligible only for integer-compatible CP-SAT expressions.
- Integer/CP: OR-Tools CP-SAT then optional Gurobi/SciPy where compatible.
- Basic continuous NLP: SciPy `minimize`; only declared convex successful models
  may be canonicalized as `OPTIMAL`, otherwise successful feasible points remain
  `FEASIBLE`.
- Every default SciPy/OR-Tools computation runs a deterministic, non-LLM
  translator output in the sandbox and emits a real `ExecutionRecord`.
- Gurobi remains optional, checks image/module/license availability, accepts a
  runtime read-only license secret, and falls back without crashing.
- Solver-native statuses are mapped exactly; feasible incumbents at a time limit
  never become `OPTIMAL`.

## Code Generation Strategy

- Deterministic adapters are preferred for supported standard models and produce
  auditable `GeneratedProgram` artifacts without an LLM call.
- `CodeAgent` is reserved for custom processing, simulation, or unsupported
  solver logic. It may translate but may not alter the mathematical model.
- Generated paths are relative and traversal-safe; dependencies are explicit;
  source hashes are deterministic; obvious hard-coded result patterns are
  rejected before execution.
- Generated code executes only through `SandboxExecutor`, with network disabled,
  non-root identity, read-only root/workspace, bounded CPU/RAM/PIDs/time/output,
  and no inherited host credentials.

## Tests

- Schema, registry, symbol conflict, expression, unit, MODEL gate, algorithm
  selection, routing, status conversion, SOLVE gate, feasibility, and evidence
  chain unit tests.
- Real sandboxed SciPy LP, MILP, NLP, infeasible, and unbounded solver tests.
- OR-Tools CP-SAT compatibility/real execution tests when the versioned image is
  available; Gurobi availability and fallback tests without a commercial license.
- API/database integration tests for build, solve, listings, immutable versions,
  migrations, failed execution evidence, and exact result links.
- A real-SciPy E2E from problem reasoning through `SOLVE`, explicitly retaining
  mock metadata for LLM fixture stages while never mocking the numerical result.
- Preserve all Phase 1–3 tests and overall coverage of at least 85%.

## Security Risks

- Expression or source text injection: adapters consume a typed AST and serialize
  JSON; no `eval`, shell invocation, or source interpolation is used.
- Generated-code escape: preserve all Phase 3 Docker boundaries and artifact
  size/count/hash checks.
- Solver resource exhaustion: enforce sandbox limits and canonical timeout/error
  records.
- CP-SAT numeric corruption: reject non-integral coefficients unless an explicit,
  auditable scaling plan exists; Phase 4 does not invent one.
- Gurobi credential leakage: mount an explicitly configured license file
  read-only at runtime; never persist its contents/path in result or logs.
- Result fabrication or broken provenance: SOLVE gate requires a non-mock
  execution record and exact persisted model/run/result references.

## Acceptance Criteria

- All required Phase 4 schemas, agents, registries, units, gates, solver adapters,
  routing, workflow, API, migration, evidence links, tests, and documentation
  exist without core `pass`/TODO implementations.
- Real sandboxed SciPy LP, MILP, and NLP cases compute expected deterministic
  values; infeasible and unbounded cases retain the correct canonical statuses.
- Bounds and implemented linear constraints are recomputed, with maximum
  violation recorded and gate-enforced.
- `ResultRecord -> SolverRun -> ExecutionRecord -> MathematicalModel(version)` is
  queryable and rejects a broken chain.
- The E2E state reaches `SOLVE`, numerical evidence is non-mock, and no Phase 5 or
  paper functionality is implemented.
- Ruff format/check, strict mypy, full pytest with PostgreSQL and Docker/Sandbox,
  Alembic upgrade/check, coverage >=85%, and `git diff --check` all pass.

## Outcome

Completed on 2026-08-27 without entering Phase 5. Final verification:

- `ruff format --check .`: 149 files formatted.
- `ruff check .`: passed.
- `mypy --strict src`: 89 source files passed.
- `pytest --cov=mathmodel_ai`: 104 passed, 2 explicitly skipped, 87.10% coverage.
- Real Docker tests passed for Phase 3 isolation and the Phase 4 SciPy/OR-Tools image.
- PostgreSQL migration `20260826_0004` passed integration checks; a dedicated
  disposable database passed upgrade, downgrade to `20260826_0003`, re-upgrade,
  and `alembic check` before deletion.
- Real acceptance LP: SciPy/HiGHS `OPTIMAL`, objective `30.0`, `x=10.0`,
  `y=0.0`, maximum constraint violation `0.0`, with a non-Mock ExecutionRecord.
- `git diff --check`: passed. No Phase 4 commit was created.
