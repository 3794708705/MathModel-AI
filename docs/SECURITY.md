# Security baseline

- API keys use `SecretStr`, environment variables, or a future secret manager.
- Health and system endpoints never serialize settings or credentials.
- Provider errors sanitize response bodies before logging or returning.
- Uploaded paths, archive extraction, generated-code execution, and network
  access are outside Phase 1 and must enter through dedicated guarded modules.
- The future sandbox defaults to non-root, network disabled, bounded CPU/RAM,
  bounded processes, bounded time, and an isolated filesystem.
- Uploaded instructions are untrusted data and never supersede system policy.

Threats tracked for later phases include path traversal, zip bombs, command
injection, SSRF, credential leakage, sandbox escape, and resource exhaustion.

