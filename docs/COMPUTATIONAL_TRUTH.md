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
