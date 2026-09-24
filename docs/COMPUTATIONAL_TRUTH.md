# Computational truth

LLMs reason, plan, review, interpret, and write. Python and solvers calculate,
optimize, simulate, and validate. Database records state and provenance; file
storage preserves inputs and artifacts.

No run may be described as executed without an `ExecutionRecord`. Mock outputs
must carry `is_mock=true` and can validate orchestration only. Any important
number without a complete evidence path remains `UNVERIFIED` and is excluded
from final paper artifacts.

Phase 3 `ExecutionRecord` stores the code SHA-256 and code artifact, configured
image plus inspected image digest, start/end/runtime, limits, bounded stdout and
stderr, exit code, status, isolation flags, metrics, and output artifacts.
`UNAVAILABLE`, `FAILED`, `TIMEOUT`, and `REJECTED` are recorded outcomes, never
converted to success. The EXECUTION quality gate additionally requires exit code
zero, non-Mock execution, disabled network, non-root identity, read-only root,
and existence of every declared output artifact.
Code and output artifact hashes are recomputed by the gate; metadata presence
alone is insufficient.

Phase 4.1 numerical truth follows this exact persisted chain:

```text
ResultRecord
  -> SolverRun (canonical status, options, full SolverResult)
  -> ExecutionRecord (entrypoint/bundle/image hashes, real exit and artifacts)
  -> MathematicalModel record (stable model_id + exact version + digest)
  -> GeneratedProgram (when a program is recorded; exact source bundle)
```

The SOLVE gate requires matching identities, a non-Mock execution, legal status,
required variables/objective when feasible, truthful optimal/feasible flags, and
recomputed bounds/implemented constraints. Objective and key variable values are
compared with configured absolute/relative tolerances; IDs, status, enums, and
hashes require exact equality. `OPTIMAL` cannot be accepted unless
`is_optimal=true` and `is_feasible=true`. Time limits, execution failures,
missing artifacts, NaN/Inf, and violated constraints cannot be wrapped as
success. `EvidenceIntegrityVerifier` returns structured errors such as
`OBJECTIVE_MISMATCH`, `STATUS_MISMATCH`, `KEY_OUTPUT_MISMATCH`,
`MODEL_DIGEST_MISMATCH`, `CODE_HASH_MISMATCH`, `MOCK_EXECUTION`, and
`BROKEN_RESULT_CHAIN`. The in-memory SOLVE gate and persisted evidence endpoint
use the same verifier; content integrity is not deferred to Phase 5 validation.

Phase 5 asks a different question: whether an already truthful result remains
mathematically credible under independent recomputation and controlled changes.
`IndependentValidator` has a separate expression evaluator and does not trust
the solver's feasibility summary. It reconstructs variable/domain checks, every
declared constraint, the objective, expected key outputs, evidence integrity,
and machine-supported validation requirements. An unknown requirement is
`NOT_EVALUABLE`; it is not silently passed. The terminal gate repeats this
calculation and compares every deterministic report field with persisted state.

For generated data-bound programs, `result.json` may now carry bounded finite
`predictions` and named numeric `series` alongside scalar `variable_values`.
The execution-bound raw artifact parser preserves these arrays for the closed
independent metric calculators; flat reported `metrics` remain comparisons, not
calculator inputs. This is an artifact-contract bridge, not proof that an array
was computed from a particular input. Acceptance still requires an exact
reviewed metric/observation binding and independent recomputation. A generated
series or a successful solve alone cannot resolve an unchecked validation
requirement or make the 2024 MCM C benchmark pass.

Every accepted sensitivity or robustness scenario follows:

```text
Perturbation specification
  -> immutable MathematicalModel copy + scenario digest
  -> real solver execution
  -> non-Mock ExecutionRecord
  -> independently recomputed feasibility/objective
  -> VerificationExperiment record
  -> aggregate report
  -> persisted evidence re-audit
```

The re-audit reconstructs the scenario from perturbation metadata and parses the
constant deterministic solver payload without executing or evaluating source.
It requires the payload model, model digest, generated program, code and bundle
hashes, ExecutionRecord relational fields, SolverResult, objective, outputs, and
independent feasibility to agree. Metadata alone cannot prove a perturbation ran.

Default sensitivity is one-at-a-time signed `±5%`, `±10%`, and `±20%`, subject
to explicit experiment caps and eligible numeric parameters. Robustness selects
one declared method: scenario analysis, worst-case analysis, seeded noise, or
seeded Monte Carlo. Bootstrap returns `BLOCKED` until a real row-resampling and
data-to-parameter binding exists.

Red Team combines deterministic attacks with structured model review. A Mock
review is always `INCONCLUSIVE`/`HUMAN_REVIEW`; no absence of findings is inferred
from it. Unresolved Critical findings prevent progression. Model Repair must use
the same stable model ID, increment exactly one version, change the mathematical
digest, map actions to every Critical finding, pass the Phase 4 MODEL gate, and
then undergo a new real solve and the entire Phase 5 chain again. Three
unsuccessful automatic cycles end in `HUMAN_REVIEW`.

Final acceptance is explicit:

```text
one formal Result
  + independently recomputed Validation
  + re-audited Sensitivity executions
  + re-audited Robustness executions
  + deterministic non-Mock Red Team pass
  -> VERIFIED gate PASS
  -> ProblemState.verified_result_id
```

No other result may have `VERIFIED` trace status. A latest timestamp or a
successful SOLVE gate is never a substitute for this pointer and chain.

Phase 6 preserves that truth rather than copying it into prose:

```text
ProblemState.verified_result_id
  -> exact persisted Phase 5 verification chain
  -> immutable EvidenceRecord payloads and provenance hashes
  -> ClaimEvidenceLink structured source fields
  -> deterministic claim/citation/document gates
  -> Paper IR evidence snapshot
  -> rendered artifacts and manifest hashes
```

Numeric and comparison claims are checked against structured evidence; prose is
never parsed as the authoritative number. Solver/optimality assertions are
checked against the formal result. Formal assumptions must already be accepted
and supported. Retrieved bibliographic metadata and trusted source text are
separate from model-generated support review; both metadata and support must
pass for a critical literature claim.

Figure records bind canonical data, generation code, and image hashes. Table
records bind canonical rows and values. The renderer consumes registry IDs and
escapes all prose; it does not recover state from TeX. PDF compilation is a real
Docker execution with network disabled, a non-root user, read-only root,
bounded CPU/RAM/PIDs/time, and shell escape disabled. A successful compile alone
cannot satisfy the paper gate: deterministic evidence checks and a non-Mock
independent factual audit must also pass.
