# Frontend Connectivity and Multi-Model Routing UX

Status: completed and verified locally on 2026-09-01.

## Goal

Make backend connectivity failures actionable and expose the existing provider, model-profile, and per-Agent routing contracts without changing backend routing semantics.

## Current State

The frontend defaults to `http://127.0.0.1:8000`, but all failed fetches collapse to `HTTP_0`. Provider and model registries already support one-to-many relationships, and the backend already owns route preview and save-time hard-filter validation.

## Files Affected

- `frontend/src/api/client.ts` and frontend connectivity UI/tests
- `frontend/src/app/AppShell.tsx`
- `frontend/src/pages/ModelsApiPage.tsx`
- `frontend/src/pages/RoutingPage.tsx`
- `scripts/dev-frontend.ps1`
- relevant documentation and tests

## Design

- Classify timeout, offline/network, likely CORS, and HTTP failures without exposing internals.
- Derive the global connectivity indicator only from `/health/live`.
- Keep provider/model/routing writes on existing backend endpoints.
- Treat the default model as a Router fallback preference; preserve explicit Agent routes.
- Preview every selected primary/fallback candidate and rely on backend save-time revalidation.

## Implementation Steps

1. Add bounded fetch handling and actionable connection errors.
2. Add a live-only backend status indicator and frontend startup warning.
3. Clarify registry counts, credential availability, repeated model creation, and next actions.
4. Add human-readable Agent labels, fallback selection, and the bulk default shortcut.
5. Add adversarial frontend/backend tests and run real browser checks.

## Tests

- Frontend unit/integration tests for connectivity, registry, and route persistence.
- Existing backend Router/provider security suites.
- Typecheck, lint, formatting, Python regression, and startup tests.
- Real browser offline/online and controlled multi-model routing checks.

## Risks

- Browser fetch failures do not expose a standard CORS error object; classification must remain explicit about likely CORS.
- Credential source is intentionally hidden by the backend contract, so UI wording must not invent it.

## Acceptance Criteria

- [x] No `HTTP_0` is shown to users.
- [x] Offline guidance includes the exact backend command and URL.
- [x] Two providers/four models are visible and additional models remain easy to add.
- [x] Four Agent routes persist across refresh and backend rejection reasons remain visible.
- [x] Default model selection does not overwrite Agent-specific routes.

## Verification Result

- Real browser: offline classification, connected status, two Provider records, four
  ModelProfile records, and all backend-catalog Agent routing controls verified.
- Frontend: 31 tests passed; lint, strict TypeScript checking, and production build passed.
- Backend: 508 tests passed, 8 environment-gated tests skipped, 86.85% coverage.
- PostgreSQL integration, Docker sandbox, Ruff, strict mypy, Alembic, and diff checks passed.
- Controlled browser records were removed after verification; no server process was left running.
