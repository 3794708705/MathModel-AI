# Frontend reliability and diagnostics follow-up

## Scope

Improve engineering usability after the backend CI repair. Preserve all live
benchmark evidence and acceptance thresholds. Do not start paid-provider calls,
Case A reruns, or Cases B/C. No automatic Git commit or push in this follow-up.

## Changes

- Add a separate Linux frontend CI job with Node 22 / Python 3.12, locked
  dependency installation, regenerated OpenAPI type drift detection, ESLint,
  Vitest coverage reporting, TypeScript, and production build.
- Refresh stale generated types for independent verification plans, dynamic
  replay specifications, algebraic metrics, and subproblem identity.
- Return an explicit `DATABASE_UNAVAILABLE` code with HTTP 503 from readiness;
  liveness remains independent. The browser shows a fixed safe explanation,
  never connection strings or raw database errors.
- System displays unavailable configuration as unknown, invalidates cached
  healthy presentation after failed refreshes, and supports read-only recheck.
  Local PostgreSQL troubleshooting instructions are displayed, not executed.
- Correct the PDF Docker build context and documented Case A policy status;
  distinguish the confirmed historical green backend CI from new local work.

## Verification

- Frontend: 46 tests pass; ESLint, TypeScript, and production build pass.
- Backend health/startup subset: 8 tests pass. Existing upstream test-client
  deprecation warning remains, unrelated to this change.
- Ruff formatting/lint and strict mypy pass.
- OpenAPI regeneration is deterministic across two runs (identical SHA-256).
- A fresh Linux `node:22-bookworm-slim` container installed 342 locked packages
  with `npm ci` and passed ESLint, all 46 tests, TypeScript, and production build.
  The repository was mounted read-only; installs and outputs stayed in the
  temporary container filesystem. Frontend reported line coverage is 83.27%.
- Full live benchmark acceptance remains `NOT_READY`.

The new frontend GitHub job must still run after these changes are committed
and pushed; the previous green run is not evidence that this new job has run.
