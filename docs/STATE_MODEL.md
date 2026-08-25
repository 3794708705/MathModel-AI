# State model

`ProblemState` v2 is the sole shared workflow state. It retains the Phase 1
future-facing fields and adds typed `ProblemAnalysis`, evidence items,
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
`agent_runs`, and `model_decisions`. `AgentRun` captures model routing and call
metadata. `ModelDecisionEvidence` carries the exact weights, score matrix,
rationale, selected/backup IDs, timestamp, and originating agent run.
