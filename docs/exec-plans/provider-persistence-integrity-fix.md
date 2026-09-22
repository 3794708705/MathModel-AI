# Provider Persistence Integrity Fix

## Goal

Make ProviderEndpoint create/update/persist/reload digest verification deterministic while
keeping endpoint-security enforcement fail-closed for routing and outbound calls.

## Current State

The real PostgreSQL row that reproduces the Web UI failure has a valid canonical digest.
Its local HTTP destination is blocked by the current runtime security policy, but the
repository catches that policy error together with digest validation and reports both as
`persisted provider configuration failed integrity checks`. One blocked endpoint therefore
also prevents the management registry list from loading.

## Files Affected

- `src/mathmodel_ai/schemas/provider_registry.py`
- `src/mathmodel_ai/providers/repository.py`
- `src/mathmodel_ai/api/routes/providers.py`
- provider registry/API/PostgreSQL regression tests
- `docs/MODEL_REGISTRY.md`, `docs/SECURITY.md`, and `docs/WEB_UI.md`

## Design

- Define one canonical ProviderEndpoint payload/digest implementation and use it for model
  construction and persisted-row verification.
- Verify persisted fields and digest before any management read succeeds.
- Treat current runtime endpoint policy as an execution-eligibility concern distinct from
  persistence integrity. Management reads may expose an integrity-valid blocked endpoint so
  it can be edited; routing, probes, discovery, and provider execution continue to use strict
  policy-validated reads.
- Keep credentials and all health/probe/discovery/catalog metadata out of the stable digest.

## Implementation Steps

1. Add canonical payload/digest helpers without changing the existing digest schema.
2. Refactor row reconstruction into integrity-only and execution-policy-validated paths.
3. Route management list/get/update/view operations through integrity-only reads.
4. Add default, enum, URL, header, metadata, update, runtime-state, restart, and tamper tests.
5. Reproduce and verify through real PostgreSQL API and browser flows.

## Tests

- Provider schema and repository unit/adversarial tests.
- Real FastAPI + PostgreSQL create/persist/reload/restart integration.
- Provider catalog/custom/update/connection/discovery/model API regressions.
- Browser Add Provider, refresh, connection/discovery, model-add flow.
- Full backend and frontend regression suites plus static and migration checks.

## Risks

- A management read must never become execution authorization.
- Unknown digest mismatch must continue to reject both management and execution reads.
- Error handling and diagnostics must not expose credential references or secret material.

## Acceptance Criteria

- Canonical create and reload digests match for every supported default/enum/URL/header shape.
- A policy-blocked but integrity-valid endpoint cannot poison the Provider registry UI.
- Such an endpoint remains unusable for routing, probe, discovery, and outbound execution.
- Persisted field tampering remains fail-closed.
- No credentials, runtime health, probe output, model discovery, or catalog hints enter the
  provider configuration digest.

## Result

Status: **READY**

- The stored and recomputed canonical digest inputs were identical in the real failing row.
  The failure was a runtime local-endpoint policy rejection that had been caught and
  mislabeled as persistence-integrity failure.
- Provider digest construction now has one public canonical payload/digest implementation.
- Management reads remain integrity-checked but can expose a currently policy-blocked row
  for repair; every routing, probe, discovery, connection-test, and outbound path continues
  to use the strict policy-validated read.
- Real PostgreSQL create/persist/reload/restart and direct-row tamper regression passed.
- Real browser Add Provider, refresh, connection test, discovery, Add Model, and refresh
  passed against an isolated controlled endpoint. Exact audit rows and processes were
  removed afterward.
- Full verification: 520 passed, 8 environment-gated skips, 86.87% coverage; Ruff, strict
  mypy, Alembic check, git diff check, and all frontend checks passed.
