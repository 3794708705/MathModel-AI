# State model

`ProblemState` v3 is the sole shared workflow state. It retains the Phase 2
reasoning fields and adds registered files, tracked artifacts, datasets,
deterministic data profiles, cross-dataset relationships, structured data
understanding, execution records, and a separate data-stage history. Typed
`ProblemAnalysis`, evidence items,
subproblems, ambiguities, proposed assumptions, model candidates, score matrix,
primary/backup selection, decision evidence, quality gates, and stage history.

Every traceable statement is classified as `FACT`, `DATA`, `ASSUMPTION`,
`DERIVATION`, `RESULT`, or `EXTERNAL_EVIDENCE`. Verification status defaults to
`UNVERIFIED`. Persistence stores a schema version and immutable revision number;
application code validates JSON before use.

State JSON is never overwritten in place. Project creation stores version 0 at
`INGEST`; accepted Phase 2 stages create versions 1, 2, and 3 at `UNDERSTAND`,
`EXPLORE`, and `SELECT`. Each revision records `updated_at`, `updated_by`, and
`update_reason`. The relational revision number and JSON `version` must agree.

The database now contains `projects`, `problems`, `problem_states`,
`agent_runs`, `model_decisions`, `files`, `artifacts`, `datasets`,
`data_profiles`, and `execution_records`. File and execution objects are stored
both in relational registries and referenced by immutable state revisions.

The Phase 3 sub-workflow is ordered independently of the main reasoning stage:

```text
PENDING -> FILES -> DATA -> EXECUTION
```

This prevents attachment work from rewinding a project already at `SELECT`.
Every attempt appends a deterministic quality gate and data-stage history entry.
A failed execution is persisted but leaves the stage at `DATA`. Adding a new
accepted file resets semantic data understanding and the data stage to `FILES`.
