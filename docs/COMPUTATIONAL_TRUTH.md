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
