# Security baseline

- API keys use `SecretStr`, environment variables, or a future secret manager.
- Health and system endpoints never serialize settings or credentials.
- Provider errors sanitize response bodies before logging or returning.
- Upload requests have transport and exact-byte limits. Generated UUID storage
  keys prevent path traversal; raw files are immutable and SHA-256-addressed in
  metadata. Allowed extensions are checked against declared MIME and actual
  bytes/package structure.
- XLSX is never extracted to the host filesystem. Entry count, expanded bytes,
  compression ratio, member paths, required package members, and checksums are
  validated. Images have a hard pixel limit; PDF pages and rendered previews are
  bounded.
- Uploaded instructions and media are untrusted evidence and never supersede
  the DataAgent system prompt or deterministic profile values.
- `SandboxExecutor` constructs Docker arguments without interpolating untrusted
  text into the trusted wrapper, uses an inspected image
  digest, network disabled, UID/GID 65532, read-only root/workspace, bounded
  output tmpfs, dropped capabilities,
  `no-new-privileges`, and CPU/RAM/PID/time/output limits. Host credentials and
  environment are not passed to the container.
- The wrapper keeps tmpfs alive only long enough to capture logs and export a
  bounded tar stream. Host extraction rejects absolute/traversal/Windows-drive,
  duplicate, symlink, device, and oversized members before artifact storage.
- Solver programs receive serialized typed AST data; neither host nor runtime
  uses `eval`, arbitrary shell fragments, or solver code embedded in the model.
- CodeAgent programs use a fixed dependency allowlist, cannot install packages
  or enable network access, and must emit a schema-validated `result.json`.
  Entrypoint and complete source-bundle SHA-256 values are both bound to the
  non-Mock execution record before evidence can pass.
- An optional Gurobi license is an existing validated file mounted read-only at
  runtime. Its contents and host path are absent from `ExecutionRecord` and
  application logs; the default solver image contains no license or `gurobipy`.

Phase 3–4 tests exercise traversal, MIME mismatch, oversized multipart input,
image decompression limits, network denial, non-root identity, read-only mounts,
timeout, unsafe output archives, and invalid secret mounts. Residual risk: local Docker is a security boundary dependency and
must be patched/hardened by deployment; antivirus/content disarm, S3 policies,
and a production reverse-proxy request cap remain deployment work.
