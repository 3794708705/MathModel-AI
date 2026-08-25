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
- `SandboxExecutor` uses Docker arguments without a shell, an inspected image
  digest, network disabled, UID/GID 65532, read-only root/workspace, bounded
  output tmpfs copied only after exit, dropped capabilities,
  `no-new-privileges`, and CPU/RAM/PID/time/output limits. Host credentials and
  environment are not passed to the container.
Phase 3 tests exercise traversal, MIME mismatch, oversized multipart input,
image decompression limits, network denial, non-root identity, read-only mounts,
and timeout. Residual risk: local Docker is a security boundary dependency and
must be patched/hardened by deployment; antivirus/content disarm, S3 policies,
and a production reverse-proxy request cap remain deployment work.
