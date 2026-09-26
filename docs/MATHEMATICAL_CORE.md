# Mathematical core

Phase 4 introduces a solver-independent intermediate representation between a
selected modeling direction and executable code:

```text
SelectedModel -> MathModeler -> MathematicalModel vN -> MODEL gate
```

`MathematicalModel` is the source of truth. It contains stable identity/version,
target subproblems, family, assumptions and interpretation resolutions, sets and
indices, typed variables/domains, sourced parameters/constants, objective,
constraints, equations, conditions, units, data/evidence bindings, algorithm and
solver requirements, expected outputs, validation requirements, limitations,
confidence, and status. It cannot contain SciPy/Gurobi/OR-Tools objects or an
executed result.

## Machine-readable expressions

Objectives, constraints, and equations use a bounded recursive `MathExpression`
tree with constant, symbol, add, subtract, multiply, divide, power, and negate
nodes. Deterministic code walks this tree for reference discovery, evaluation,
linearization, unit composition, feasibility checking, and solver translation.
The deterministic SciPy nonlinear runtime resolves uniquely defined, acyclic
derived scalar equations at each candidate point before evaluating objectives
and constraints, then reports those derived values with the decision values for
independent verification. Undefined or non-finite values remain failures.
There is no `eval`, regex reconstruction from LaTeX, or general-purpose CAS.
LaTeX remains explanatory equation metadata, not executable truth.

## Registries

`SymbolRegistry` records canonical name, meaning, unit, domain, kind, owner model
and version, first definition, and references. It reports undefined, duplicate,
conflicting, and unused core symbols; indexed references resolve only through a
declared indexed base variable. A variable/parameter collision is critical.

`ParameterRegistry` provides version-scoped lookup and duplicate diagnostics.
Every parameter has an explicit `source_type` (`PROBLEM_FACT`, `DATA`,
`ASSUMPTION`, `DERIVATION`, `EXTERNAL`, or `ESTIMATED`) and a non-empty source
reference. Data-bound parameters refer to registered dataset IDs.

`EquationRegistry` keys every equation by `EQ-*` within one model version,
rejects duplicates, validates dependency references, and preserves both typed
lhs/rhs and human-readable derivation metadata. Objective and constraint
equation references must exist before solving.

## Units and dimensions

`UnitExpression` stores base-dimension exponents, a scale factor, display text,
and unresolved tokens. Supported base dimensions are mass, length, time,
currency, count, energy, power, and dimensionless. Multiplication/division/power
compose exponents; addition/subtraction and relation sides require equal
dimensions.

`UnitChecker` returns `PASS`, `FAIL`, or `UNKNOWN`. Unknown/custom units produce
`UNIT_CHECK_REQUIRED`; they never silently pass. An explicit mismatch causes
MODEL rejection. Phase 4 intentionally does not implement affine-temperature
units, arbitrary conversions, symbolic exponent inference, or a complete unit
ontology.

## MathModeler and identity binding

`MathModeler` receives accepted `ProblemAnalysis`, `SelectedModel`, optional data
understanding/profiles, and user guidance. It performs one
`structured_generate()` call for a `MathematicalModelDraft`; Pydantic validates
the complete machine contract. Deterministic code then binds project/problem,
the repository-assigned stable `model_id`, next version, selected candidate ID,
and `READY` status. The minimum route is `FLAGSHIP_XHIGH`, and its `AgentRun`
records the current prompt version, provider/model/reasoning, usage, latency, state
versions, retries, and Mock status.

When a DATA parameter declares an exact, supported binding to a deterministic
registered profile but leaves its scalar value empty, trusted construction fills
the value from that profile before the MODEL gate and records the materialization
in limitations. A provided but incorrect numeric value is never silently fixed:
the original bound-scalar gate rejects it. Unknown transforms, stale source-file
identity, and ambiguous profiles likewise remain unmaterialized and rejected.

Every accepted revision also receives a canonical SHA-256 `model_digest` over
its executable mathematical representation. Database identity, revision number,
timestamps, prose, and audit-only metadata do not affect it; objective,
constraints, domains, parameters, units, and solver requirements do. Programs,
executions, solver runs, and results bind the same digest, so an exact model
revision cannot be silently replaced behind stable IDs.

## MODEL quality gate

SOLVE is blocked unless deterministic checks confirm:

- the selected candidate exists and matches the model;
- target subproblems and data bindings resolve;
- decision variables and required optimization objective exist;
- a scalar optimization objective without time-indexed state is not mislabeled
  `TIME_SERIES`, which would leave later independent parameter experiments
  without a compatible deterministic solver;
- any decision-independent scalar hard constraint provable from sourced
  parameters and defining equations is feasible before code generation;
- symbol, parameter-source, and equation registries are valid;
- objective/constraint equation references exist;
- no explicit dimensional failure exists;
- targeted critical ambiguities are resolved from evidence or by a human; and
- critical assumptions are supported.

An unknown unit is a visible warning; a known conflict is a failure. Rejected
agent output is audited but cannot create a successful MODEL revision.

## Version and Phase 5 integration

Accepted models are immutable records. Phase 5 repair reuses `model_id`, creates
`version=2`, recomputes the executable digest, and leaves `v1` and its results
unchanged. A repaired model is accepted only when Red Team Critical findings are
mapped to explicit repair actions and both MODEL and MODEL_REPAIR gates pass.
It then re-enters the Phase 4 solve path; new results always reference the exact
version they solved. See `docs/VERIFICATION_REPAIR.md`.
