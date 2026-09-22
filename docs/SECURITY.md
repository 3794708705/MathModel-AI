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
- Sensitivity and robustness never execute model-authored shell commands. They
  clone only validated mathematical fields, serialize through the existing
  deterministic solver path, retain the same network-disabled/non-root sandbox,
  cap scenario counts, and require a distinct `ExecutionRecord` per scenario.
- Independent replay accepts typed AST and reviewed sidecar fields only—never
  Python, imports, host paths, or shell fragments. Dynamic replay uses a fixed
  source template; optimization replay rebuilds the deterministic solver program.
  A read-time audit reconstructs code/input digests and rehashes every physical
  artifact before a persisted outcome can contribute to acceptance.
- Uploaded data or Red Team guidance remains untrusted context. Repair output
  must pass Pydantic, stable-identity/version checks, the deterministic MODEL
  gate, and another sandboxed solve before it can affect accepted results.

Phase 3–5 tests exercise traversal, MIME mismatch, oversized multipart input,
image decompression limits, network denial, non-root identity, read-only mounts,
timeout, unsafe output archives, and invalid secret mounts. Residual risk: local Docker is a security boundary dependency and
must be patched/hardened by deployment; antivirus/content disarm, S3 policies,
and a production reverse-proxy request cap remain deployment work.

Phase 7 package construction is allow-list based. Source symlinks and Windows
reparse points are rejected; ZIP verification rejects traversal/absolute paths,
symlink entries, duplicates, case/NFC collisions, nested archives, excessive
entry count, expanded bytes, and compression ratios. Final package bytes are
independently reopened and rescanned. Unicode-normalized text, PDF text/
metadata/attachments, and supported image metadata are checked for identity,
local path, credential, database URL, private-key, internal-audit, fixture, and
Mock markers. Unsupported OCR/steganographic content remains a human-review/
deployment scanning risk; Phase 7 does not claim complete malware or steganography
detection.

Phase 8 treats official webpages, PDFs, CSV cells, literature text, and
evaluation documents as untrusted data rather than instructions. Solve
materialization permits only HTTPS sources on the explicit official-host
allowlist, refuses redirects, enforces byte limits, verifies expected size and
SHA-256, uses atomic writes, and rejects symlinks/Windows reparse points.
Evaluation-phase resources (reference solutions, judge commentary, and human
notes) are excluded by role before a `BlindSolveBundle` is constructed. A
regression cell containing `IGNORE ALL PREVIOUS INSTRUCTIONS` remains ordinary
CSV data and cannot replace agent system instructions.

Benchmark reports reject secret-field names, credential-like text, and local
absolute paths. Provider keys remain only in existing `SecretStr` configuration;
run identity stores provider/model names, not key material. Official source
bytes, runtime caches, generated reports, PDFs, and database files stay under
ignored runtime storage; only manifests, hashes, provenance, and distribution
notes are versioned. Report/result/run digests and relational scalar-vs-JSON
checks detect score, status, cost, intervention-count, and artifact-link tamper.

Residual Phase 8 risk includes scanned-image prompt injection without OCR,
malicious content outside supported structural inspection, dependency on
official-host and Crossref availability, and the absence of an asynchronous
human-cancel/crash-resume coordinator. These limitations are fail-closed and
prevent a live acceptance claim; they are not converted to Mock success.

## Configurable model endpoints

Phase 8.1 stores only `credential_ref`; `EnvironmentSecretResolver` is the sole
first-version secret lookup boundary and is invoked per outbound request rather
than cached with the adapter. Provider API responses omit the reference and
expose only a configured boolean, never value-derived metadata. Additional
headers cannot override Authorization, Proxy-Authorization, Cookie, Host,
Content-Length, Connection, `X-Api-Key`, or other managed authentication
headers; nested provider/model configuration rejects secret-bearing fields and
credential-like values. Request/Pydantic/provider exceptions are normalized and
redacted before they reach HTTP responses, AgentRun errors, or logs.

Custom endpoint configuration requires credential-free HTTPS URLs without query
or fragment data. Encoded traversal, backslashes, scheme-relative paths, and
control characters are rejected. Registration and every outbound call resolve
DNS and reject numeric/encoded loopback, private, link-local, metadata, reserved,
unspecified, IPv4-mapped-private, and mixed public/private destinations.
The socket target is then pinned to one validated address while the original
HTTP Host and TLS SNI are preserved, closing the validation-to-connect DNS race.
Redirect following is disabled and every 3xx is rejected, preventing bearer
credentials from crossing hosts or schemes. TLS verification can be disabled
only through an explicit non-production server setting; production always
rejects it. Connect/read/total timeouts, strict token/usage bounds, finite
pricing, and decoded streamed-response byte limits are mandatory.

`MM_ALLOW_LOCAL_MODEL_ENDPOINTS=true` is an administrator-level escape hatch for
local development gateways. It is not represented in ProviderEndpoint and
cannot be enabled by the registration API. Official trust is accepted only for
a built-in official domain/protocol pair; an official-looking remote model name
behind a proxy remains `USER_MANAGED_PROXY` and
`NOT_INDEPENDENTLY_VERIFIED`.

Persisted provider/model/probe rows are untrusted input. Reads recompute config
and probe digests and reapply endpoint trust/network policy. Routing consumes
the latest atomic authenticated probe rather than the mutable capability summary,
and execution revalidates the exact task/config/probe binding before each call.
This detects row tampering and preview-to-execution races; a fully compromised
database remains outside the application's cryptographic trust boundary.

Provider management reads distinguish persistence integrity from current
runtime endpoint eligibility. They still recompute and require the exact stored
digest, but may return an integrity-valid endpoint as `UNAVAILABLE` when an
administrator has disabled its local/insecure destination class. This allows
the endpoint to be edited to a permitted destination without making it usable:
Router, probes, model discovery, connection tests, and every outbound request
continue through the strict policy-validated provider read and fail closed.

## Web credential bridge

Phase 8.2 keeps `EnvironmentSecretResolver` and composes it with an
`EncryptedDatabaseSecretStore`. The UI credential endpoint accepts a Pydantic
`SecretStr`, encrypts with AES-256-GCM under a URL-safe base64 32-byte
`MM_SECRET_MASTER_KEY`, and stores only an opaque secret ID, nonce, and
ciphertext. Provider rows still contain only `credential_ref`. Every rotation
uses a new opaque ID and invalidates provider health/probe evidence; deletion
clears the reference and encrypted row. Missing or malformed master-key state
fails closed.

Credential responses contain only `credential_configured`; public provider
views omit both the key and reference. Validation errors use sanitized handlers,
application logging never receives the plaintext, and database writes see only
ciphertext. The browser input is write-only, defaults to password display, is
never put in URL/storage/logs, and is cleared immediately after success. CORS is
an explicit origin allowlist and credentials are disabled.

The independent Phase 8.2 audit additionally treats encrypted rows, persisted
provider health, route preferences, and browser network outcomes as untrusted.
`credential_configured` is true only when the current resolver can decrypt the
stored value; ciphertext or master-key tampering therefore exposes the provider
as `UNCONFIGURED` even if an older health row said `READY`. Credential responses,
including errors and unsupported methods, carry `Cache-Control: no-store` and
`Pragma: no-cache`. The frontend normalizes transport/server errors and redacts
credential-like text instead of rendering backend exception details.

Route persistence repeats the canonical Router hard filters for the primary and
every fallback model. An earlier browser preview is never authority, and stale
probe, disabled provider/model, absent credential, insufficient tier/capability,
unknown Agent/model, and the built-in `mock` provider all fail closed. Mock
providers/models are also labelled `MOCK / TEST ONLY` in the control center;
this label is explanatory and does not replace the backend rejection.
