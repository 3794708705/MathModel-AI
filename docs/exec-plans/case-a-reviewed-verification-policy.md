# Case A modeling contract completion

## Goal

Complete the formal MCM 2024 Case A mathematical-model contract required by
independent verification, review it independently, and stop before a fresh
benchmark attempt.

## Current State

Rerun 13 remains a historical real FEASIBLE result with independent validation
`NOT_EVALUABLE` and no `verified_result_id`. The official problem supplies only
qualitative ecosystem requirements and approximate male-ratio anchors 0.78 and
0.56; it supplies no ecosystem initial values, physical time scale, ecological
coefficients, parameter ranges, stability threshold, or extinction threshold.

## Files Affected

- `benchmarks/case-003-mcm-2024-a/mathematical-model-v2.json`
- `benchmarks/case-003-mcm-2024-a/independent-verification.v1-draft.json`
- `benchmarks/case-003-mcm-2024-a/independent-verification.draft.json`
- `benchmarks/case-003-mcm-2024-a/independent-verification.json`
- `benchmarks/case-003-mcm-2024-a/model-contract-red-team.json`
- `benchmarks/case-003-mcm-2024-a/model-contract-jury.json`
- `benchmarks/case-003-mcm-2024-a/model-contract-review.json`
- `scripts/build_case_a_modeling_contract.py`
- `scripts/review_case_a_modeling_contract.py`
- independent-verification schemas, metric calculator, source/model binding,
  focused tests, and status documentation

No historical database row, benchmark attempt, solver result, paper, package,
or migration is changed.

## Design

- Preserve the official PDF and manifest digests and the rerun-13 v1 digest.
- Create MathematicalModel v2 with an ordered normalized state `[J,A,F,P]`, a
  common positive reference equilibrium at `E=0.5`, and nondimensional
  `tau=t/T_ref`; physical `T_ref` remains unavailable.
- Add `adaptation_weight`: 1 selects the resource-adaptive rule and 0 selects
  the fixed midpoint ratio 0.67. Paired scenarios change only this mechanism.
- Define local stability through the complete block-triangular Jacobian, four
  explicit real eigenvalues, dominant real part, and a numerical equilibrium
  residual. Verification does not require a favorable stability result.
- Cover the 14 provisional ecological coefficients with one-at-a-time +1%
  local finite changes. This is an analysis design assumption, not a biological
  uncertainty range.
- Interpret `P` as normalized abundance of a representative non-lamprey
  fish-parasite guild. Forbid prevalence, burden, infection-pressure, global
  resilience, physical-time, and practical-extinction claims without new
  evidence.
- Add only one generic verifier primitive: select and recompute a named derived
  algebraic scalar through the existing allowlisted AST. Raw/reported derived
  values are ignored; `eval`, Python expressions, imports, and case branches are
  impossible.

## Implementation Steps

1. Re-read the official problem, manifest, ProblemState, model, execution, and
   result evidence without using evaluation material.
2. Archive the original DRAFT with content digest
   `650560edde518156a05ed31b2b60357fd06826a60f60008b5c31ae174beb9384`.
3. Generate and validate the v2 formal model and DRAFT policy candidate.
4. Run independent deterministic Red Team and Model Jury review.
5. Promote only when every contract check passes, critical findings are zero,
   Jury score is at least 85, and unresolved scientific obligations are zero.
6. Do not start any benchmark after promotion.

## Tests

- Pydantic model/policy parsing and semantic digest checks.
- SymbolRegistry, EquationRegistry, UnitChecker, and model-quality gate.
- Exact adaptive anchors, fixed comparator, reference equilibrium, Jacobian
  spectrum, and dominant-real-part recomputation.
- Generic algebraic scalar tamper regression: a raw fake derived value is
  ignored and the model AST is recomputed.
- Policy metadata, model/problem digest binding, scenario change inventory,
  14-parameter sensitivity coverage, and prior-DRAFT preservation.
- Full backend regression, coverage, Ruff, strict mypy, Alembic, and diff check.

## Risks

- All 14 ecological coefficients remain provisional assumptions.
- The common normalized initial state controls comparator fairness but does not
  remove initial-condition dependence from transient magnitudes.
- Nondimensional time cannot support calendar-time claims.
- Local Jacobian stability is not global ecological resilience.
- A practical extinction threshold is intentionally absent.

## Acceptance Criteria

- Model version 2 and a new mathematical digest exist.
- All symbols have known units and all formula dependencies resolve.
- Metrics and scenarios are executable, reproducible, source-bound, and carry
  tolerance/criterion provenance.
- Unresolved scientific obligations equal zero.
- Independent Red Team has zero critical findings.
- Independent Model Jury returns PASS.
- Production policy is REVIEWED and production eligible.
- Fresh Case A and B/C remain not run; Phase 8 remains NOT_READY.
