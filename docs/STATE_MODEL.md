# State model

`ProblemState` v4 is the sole shared workflow state. It retains registered files,
artifacts, datasets, deterministic profiles, data understanding, execution
records, and reasoning fields, then adds compact references to the accepted
`MathematicalModel`, `AlgorithmPlan`, generated programs, solver runs, and result
records. Large full model/run/result payloads remain in independent registries;
state carries identities and exact versions. Typed `ProblemAnalysis`, evidence items,
subproblems, ambiguities, proposed assumptions, model candidates, score matrix,
primary/backup selection, decision evidence, quality gates, and stage history.

Every traceable statement is classified as `FACT`, `DATA`, `ASSUMPTION`,
`DERIVATION`, `RESULT`, or `EXTERNAL_EVIDENCE`. Verification status defaults to
`UNVERIFIED`. Persistence stores a schema version and immutable revision number;
application code validates JSON before use.

State JSON is never overwritten in place. Project creation stores version 0 at
`INGEST`; accepted Phase 2 stages create versions 1, 2, and 3 at `UNDERSTAND`,
`EXPLORE`, and `SELECT`. Phase 4 then creates version 4 at `MODEL` and version 5
at `SOLVE` for the first successful pass. Each revision records `updated_at`, `updated_by`, and
`update_reason`. The relational revision number and JSON `version` must agree.

```text
INGEST -> UNDERSTAND -> EXPLORE -> SELECT -> MODEL -> SOLVE
```

The MODEL revision stores a model record ID plus stable `model_id` and `version`.
A future repair creates `v2`; it does not overwrite `v1`. The reference also
stores the canonical `model_digest`. The SOLVE revision stores only
`GeneratedProgramRef`, `SolverRunRef`, and `ResultRecordRef`, each bound to that
digest, while
the independent records retain complete options, canonical status, variables,
feasibility, routing decision, execution origin, artifacts, and execution
evidence. A failed SOLVE gate creates a
traceable retry revision back at `MODEL`; it cannot advance as a successful solve.

The database now contains `projects`, `problems`, `problem_states`,
`agent_runs`, `model_decisions`, `files`, `artifacts`, `datasets`,
`data_profiles`, `execution_records`, `mathematical_models`,
`generated_programs`, `solver_runs`, and `results`. File and execution objects are stored
both in relational registries and referenced by immutable state revisions.

The Phase 3 sub-workflow is ordered independently of the main reasoning stage:

```text
PENDING -> FILES -> DATA -> EXECUTION
```

This prevents attachment work from rewinding a project already at `SELECT`.
Every attempt appends a deterministic quality gate and data-stage history entry.
A failed execution is persisted but leaves the stage at `DATA`. Adding a new
accepted file resets semantic data understanding and the data stage to `FILES`.
