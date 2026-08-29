# State model

`ProblemState` v6 is the sole shared workflow state. It retains registered files,
artifacts, datasets, deterministic profiles, data understanding, execution
records, and reasoning fields, then adds compact references to the accepted
`MathematicalModel`, `AlgorithmPlan`, generated programs, solver runs, and result
records. Phase 5 adds typed references to validation, sensitivity, robustness,
Red Team, and repair-cycle records. Phase 6 adds immutable `PaperVersionRef`
entries whose snapshots pin the exact verified model, formal result, validation,
sensitivity, robustness, Red Team, and citation set. Large full payloads remain
in independent registries; state carries identities and exact versions. Typed `ProblemAnalysis`, evidence items,
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
`update_reason`. Phase 5 creates subsequent revisions for each gate and each
repair/solve/reverification step. The relational revision number and JSON
`version` must agree.

```text
INGEST -> UNDERSTAND -> EXPLORE -> SELECT -> MODEL -> SOLVE
       -> VALIDATE -> SENSITIVITY -> ROBUSTNESS -> RED_TEAM
                                                   | PASS
                                                   ` MODEL_REPAIR -> SOLVE -> ...
RED_TEAM -> PAPER
```

The MODEL revision stores a model record ID plus stable `model_id` and `version`.
An accepted repair creates `v2`; it does not overwrite `v1`. The reference also
stores the canonical `model_digest`. The SOLVE revision stores only
`GeneratedProgramRef`, `SolverRunRef`, and `ResultRecordRef`, each bound to that
digest, while
the independent records retain complete options, canonical status, variables,
feasibility, routing decision, execution origin, artifacts, and execution
evidence. A failed SOLVE gate creates a traceable retry revision back to its
originating modeling stage (`MODEL` or `MODEL_REPAIR`); it cannot advance as a
successful solve.

`ValidationReportRef`, `SensitivityReportRef`, `RobustnessReportRef`, and
`RedTeamReportRef` bind their report to the stable model identity, exact version
and digest, result, and prerequisite reports. Experiment executions and artifacts
are appended to the existing state registries. `RepairCycleRef` records the source
and optional accepted target model version/digest. Failed gates do not permit a
later stage to claim success; Mock Red Team/repair output moves to
`HUMAN_REVIEW`.

`verified_result_id` is the sole final-result pointer. A v5+ state may set it only
when it references a persisted `ResultRecordRef`, the matching `RESULT-{id}` is
the only trace item marked `VERIFIED`, and a passing `VERIFIED` quality gate has
`subject_ref=result:{id}`. SOLVE and intermediate verification revisions clear
the pointer. This prevents selecting a sensitivity, robustness, stale, repaired,
or merely latest result as the accepted formal result.

A state can enter `PAPER` only through a matching persisted Phase 5 terminal
gate. In schema v6, any paper marked `READY_FOR_FINAL_JURY` must reference the
same `verified_result_id` held by the state. Paper versions append; they never
replace prior evidence snapshots or artifacts. Phase 6 does not set submission
state and cannot mark a paper `SUBMISSION_READY`.

The database now contains `projects`, `problems`, `problem_states`,
`agent_runs`, `model_decisions`, `files`, `artifacts`, `datasets`,
`data_profiles`, `execution_records`, `mathematical_models`,
`generated_programs`, `solver_runs`, `results`, `validation_runs`,
`sensitivity_runs`, `robustness_runs`, `verification_experiments`,
`red_team_reports`, and `repair_cycles`. File and execution objects are stored
both in relational registries and referenced by immutable state revisions.
The Phase 6 database registries preserve evidence/claim links, retrieved
literature and citation checks, paper versions, document IDs, figure/table
hashes, and paper artifact manifests.

The Phase 3 sub-workflow is ordered independently of the main reasoning stage:

```text
PENDING -> FILES -> DATA -> EXECUTION
```

This prevents attachment work from rewinding a project already at `SELECT`.
Every attempt appends a deterministic quality gate and data-stage history entry.
A failed execution is persisted but leaves the stage at `DATA`. Adding a new
accepted file resets semantic data understanding and the data stage to `FILES`.
