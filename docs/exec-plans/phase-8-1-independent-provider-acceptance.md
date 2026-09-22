# Phase 8.1 independent provider acceptance

## Goal

Adversarially verify that malicious or stale provider configuration, forged
capability/trust data, secrets, SSRF destinations, preview races, Mock fallback,
and persistence tampering cannot enter an official live Agent route.

## Current state

- Parent baseline: `30a7ae72cb28ad3845decb1b6c61ae95efa1cc95`.
- Phase 8 and Phase 8.1 changes remain intentionally uncommitted.
- Phase 8.1 self-test entered this audit as `SELF_TEST_READY`; independent
  adversarial fixes and regression tests are intentionally uncommitted.
- Migration head: `20260830_0010`.
- Historical invariant: 3 complete BenchmarkRun records and 9 complete official
  attempts remain readable.
- No real provider credentials may be searched for or configured during this
  audit. Live-provider gates remain `SKIPPED_NO_CREDENTIALS`.

## Files affected

- Provider security, schemas, adapters, registry, probe, Router, Agent execution
  boundary, API and persistence code only when a concrete defect is found.
- Adversarial tests under `tests/providers`, `tests/routing`, `tests/integration`,
  and `tests/benchmark`.
- Provider/security/testing documentation and this audit record.

## Design

1. Treat persisted state, user declarations, endpoint/model names, remote model
   strings, preview decisions, provider responses, and error text as untrusted.
2. Recompute canonical provider/model integrity and current capability evidence
   at route and execution boundaries.
3. Keep secret resolution late-bound; persist and serialize only a reference or
   configured boolean.
4. Apply URL/DNS/TLS/redirect/timeout/decompressed-byte controls both at
   registration and immediately before requests.
5. Enforce hard requirements before cost/deadline/preferences and reject Mock in
   every registry/live fallback path.
6. Preserve immutable historical AgentRun/BenchmarkRun snapshots and all probe
   history while selecting current evidence deterministically.

## Implementation steps

1. Attack secret persistence/redaction and URL/header/auth boundaries.
2. Attack encoded/IPv6/mixed/rebound destinations, redirects, TLS, timeout,
   cancellation, decompression and response limits.
3. Attack declared/observed capability precedence, stale/tampered/partial probes,
   usage/pricing/context/quality metadata, and probe cost/tool safety.
4. Attack preview-to-execution races, exact route binding, Agent policies,
   fallback loops/duplicates, deterministic scoring, Mock and legacy paths.
5. Attack official trust/model identity, protocol confusion, custom mapping,
   deletion/collision behavior, persistence snapshots, and vendor leakage.
6. Add only regression tests and fixes justified by real findings.
7. Re-run migration roundtrip, PostgreSQL, complete coverage and static gates.

## Findings resolved

- Public provider views exposed `credential_ref`, and framework validation/error
  payloads could echo secret-shaped request input.
- Cached configured adapters captured a secret value instead of resolving it for
  every call, so rotation/removal was not an immediate execution boundary.
- Router and benchmark preflight trusted a mutable observed-capability summary
  and READY health without requiring the latest atomic probe and authentication
  PASS.
- Route decisions did not bind the exact TaskProfile or probe, leaving a
  preview-to-execution race across disable/config/probe changes.
- Persisted provider/model/probe rows were not all independently revalidated for
  canonical digest and endpoint-trust integrity on read.
- Tool probe success did not require the exact inert tool name; remote usage and
  pricing accepted values that could corrupt cost evidence.
- Encoded custom paths, decoded compression limits, timeout classes, and an
  attacker-controlled response header had incomplete adversarial boundaries.
- Request-stage DNS revalidation was followed by a second transport lookup,
  leaving a rebinding TOCTOU; the connection now pins the validated address and
  preserves the configured Host/TLS SNI.
- Historical AgentRun identity treated provider-reported model text as the
  configured model snapshot; both values are now retained separately.
- Benchmark registry availability repeated the capability-summary/authentication
  trust defect instead of evaluating the current atomic probe.
- One wall-budget regression relied on sub-tick clock movement and was made
  deterministic with an injected monotonic clock.

## Tests

- Deterministic fake transports only; no real provider network calls.
- Independent OpenAI-compatible, partial-compatible, and custom-JSON E2E.
- Phase 4–8 adversarial and Phase 8.1 security/routing regressions.
- PostgreSQL `0009 -> 0010 -> 0009 -> 0010`, integration, Alembic check, and
  historical 3/9 integrity query.
- Full pytest coverage, Ruff, strict mypy, `git diff --check`, secret/artifact
  scans, and repository vendor-conditional search.

## Risks

- A database administrator can rewrite arbitrary rows; application integrity can
  detect canonical config mismatch and stale evidence but cannot provide a
  cryptographic trust root against a fully compromised database.
- DNS validation pins the actual socket target while retaining the configured
  Host/TLS SNI. Production deployment should still enforce outbound firewall and
  resolver policy as defense in depth.
- Provider behavior is heterogeneous; unsupported downgrade must remain explicit
  and cannot silently change a native-schema or tool claim.

## Acceptance criteria

- Every required attack group is PASS, or the audit is `NOT_READY` with explicit
  blockers.
- No secret leak, SSRF/trust/capability/routing/Mock bypass, arbitrary custom
  execution, stale preview execution, or historical integrity regression remains.
- Full regression passes with coverage at least 85%.
- Final status is only `READY` or `NOT_READY`; no commit or benchmark run occurs.

## Verification results

- Independent acceptance: `READY`.
- Full regression: 482 passed, 8 truthful environment-gated skips.
- Combined statement/branch coverage: 86.87% (required minimum 85%).
- Provider protocol E2E: OpenAI-compatible native-schema, partial-compatible
  prompt-JSON, and constrained custom-JSON HTTP all passed through AgentRun
  persistence using deterministic fake transports.
- PostgreSQL: `0009 -> 0010 -> 0009 -> 0010`, schema/FK/index inspection,
  integration test, and Alembic check passed at head `20260830_0010`.
- Historical integrity: 3 BenchmarkRuns, 9 attempts, and 9 results remained
  readable; run and attempt row hashes were unchanged across migration.
- Ruff format/check, strict mypy, and `git diff --check` passed.
- DeepSeek, Qwen, and generic custom live tests remain
  `SKIPPED_NO_CREDENTIALS`; no real credential was searched for or configured.
