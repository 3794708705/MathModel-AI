# State model

`ProblemState` v7 is the sole shared workflow state. It retains registered files,
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
Phase 7 appends compact `SubmissionSummaryRef` values that pin the profile
version, approved paper version, formal verified result, manifest hash, package
hash, and status. Full jury/check/package records stay in their registries.
Phase 8 does not add private benchmark truth to a single project state. A
`BenchmarkRun` spans multiple projects and binds each terminal `ProblemState`
through its `BenchmarkAttempt.project_id`; benchmark identity, metrics, failures,
and interventions live in a separate immutable audit registry.
Phase 8.1 likewise keeps ProviderEndpoint, ModelProfile, capability probes, and
Agent route policies outside `ProblemState`. State revisions refer to AgentRun
evidence; each run snapshots the exact provider/model IDs and configuration
digests so a later registry change cannot rewrite project history.

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
PAPER -> FINAL_JURY -> SUBMISSION -> FINAL
                    | correction/recheck
                    `------------> PAPER
```

The complete product lifecycle is therefore logically:

```text
INGEST -> UNDERSTAND -> DATA -> EXPLORE -> SELECT -> MODEL -> SOLVE
       -> VALIDATE -> SENSITIVITY -> ROBUSTNESS -> RED_TEAM -> REPAIR
       -> PAPER -> FINAL_JURY -> SUBMISSION -> BENCHMARK
```

`BENCHMARK` is an outer audit stage, not a value written into every
`ProblemState.current_stage`: one run evaluates three or more independently
versioned projects and must not rewrite their frozen Phase 7 histories.

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

Schema v7 permits `FROZEN` only when a submission uses the state's exact
`verified_result_id`, references a `READY_FOR_FINAL_JURY` paper version, and has
both manifest and package hashes. A correction or rerun first marks current
readiness `DIRTY`/`RECHECK_REQUIRED`; it never mutates a historical snapshot.
`FINAL` means the local package is frozen and ready to submit, not that an
external competition platform accepted an upload.

The frozen snapshot additionally binds canonical digests for the profile,
current requirement coverage, candidate/Paper IR, Jury report, SubmissionCheck,
and exact source artifact set. The read path does not trust `status=FROZEN`: it
revalidates those records against the current official `verified_result_id`,
paper reference and problem requirements before reopening the physical package.

The database now contains `projects`, `problems`, `problem_states`,
`agent_runs`, `model_decisions`, `files`, `artifacts`, `datasets`,
`data_profiles`, `execution_records`, `mathematical_models`,
`generated_programs`, `solver_runs`, `results`, `validation_runs`,
`sensitivity_runs`, `robustness_runs`, `verification_experiments`,
`red_team_reports`, and `repair_cycles`. File and execution objects are stored
both in relational registries and referenced by immutable state revisions.
The Phase 6 database registries preserve evidence/claim links, retrieved
literature and citation checks, paper versions, document IDs, figure/table
hashes, and paper artifact manifests. Phase 7 registries preserve versioned rule
provenance, requirement mappings, jury findings, deterministic check inputs,
frozen snapshots, source artifact hashes, manifests, packages, and correction
scopes. Phase 8 registries preserve run configuration/source-tree identity,
every formal attempt, case result links, atomic metrics, failures, and human
interventions. A deterministic report rebuild verifies each digest and
recalculates scores/statuses; no `latest result` shortcut is used.

The Phase 3 sub-workflow is ordered independently of the main reasoning stage:

```text
PENDING -> FILES -> DATA -> EXECUTION
```

This prevents attachment work from rewinding a project already at `SELECT`.
Every attempt appends a deterministic quality gate and data-stage history entry.
A failed execution is persisted but leaves the stage at `DATA`. Adding a new
accepted file resets semantic data understanding and the data stage to `FILES`.
