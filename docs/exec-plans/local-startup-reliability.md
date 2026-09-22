# Local Startup Reliability Fix

## Goal

Provide one reliable Windows local-development startup path without changing
backend business semantics or application lifecycle isolation.

## Current State

- The ASGI application is constructed by `mathmodel_ai.main.create_app`.
- `mathmodel_ai.main:application` is invalid because `application` is only a
  local variable inside the factory.
- A module-level `app = create_app()` adds import-time construction and is used
  by current Docker/OpenAPI tooling, but tests already use the factory.
- `/health/live` and `/health/ready` already separate process liveness from
  database readiness.
- The project uses `uv`, a repository `.venv`, and a Python 3.12 test baseline.

## Files Affected

- `src/mathmodel_ai/main.py`, new CLI/preflight modules, and package entry point.
- System-status adapter/schema for non-secret SecretStore availability.
- Docker/OpenAPI startup adapters.
- `scripts/dev-backend.ps1` and `scripts/dev-frontend.ps1`.
- Startup/CLI tests and local-development documentation.

## Design

1. Keep `create_app` as the sole ASGI construction contract and use Uvicorn
   factory mode everywhere.
2. Add `python -m mathmodel_ai serve` using only `argparse` and existing Uvicorn.
3. Run a non-secret preflight that reports Python, redacted database readiness,
   SecretStore state, local directories, and live-provider configuration without
   requiring a provider credential.
4. Let a new environment start without `MM_SECRET_MASTER_KEY`; if encrypted rows
   already exist, warn explicitly while runtime resolution remains fail-closed.
5. Make PowerShell scripts resolve the repository and `.venv` directly so an
   unrelated Anaconda/base Python cannot be selected.

## Implementation Steps

1. Remove import-time ASGI construction and update internal consumers to factory
   mode.
2. Implement typed CLI parsing, preflight reporting, and `__main__` dispatch.
3. Expose only SecretStore configured/not-configured status through the existing
   system adapter and UI.
4. Add backend/frontend development scripts.
5. Add factory, OpenAPI, health, CLI, invalid-port, preflight, and missing-key
   regression tests.
6. Update README, Web UI, testing, and repository command documentation.
7. Run backend/frontend regression and real local startup smoke tests.

## Tests

- Factory import and FastAPI/OpenAPI/docs/liveness smoke.
- CLI help, invalid port, factory target, host/port/reload forwarding.
- Preflight with empty local SecretStore and with existing encrypted rows but no
  master key.
- Existing SecretStore tamper/missing-key regressions.
- Full pytest/coverage, Ruff, mypy strict, Alembic, diff check, frontend
  typecheck/build.
- Real PowerShell backend/frontend launch and HTTP reachability.

## Risks

- Uvicorn reload starts a child process; smoke cleanup must target the exact
  process tree and use temporary runtime configuration.
- Database inspection must never print credentials or prevent liveness-only
  startup when PostgreSQL is unavailable.
- OpenAPI generation must construct its own isolated factory instance after the
  module-level app is removed.

## Acceptance Criteria

- `scripts/dev-backend.ps1` is the documented canonical local command.
- `python -m mathmodel_ai serve --help` works and the runner uses
  `mathmodel_ai.main:create_app` with `factory=True`.
- `/docs`, `/openapi.json`, and `/health/live` load in a real process.
- Missing live provider credentials never block Web UI startup.
- Missing master key is explicit and stored encrypted credentials remain
  unavailable.
- Backend business semantic changes equal zero; no commit is created.

## Verification Outcome

- Follow-up manual audit reproduced a silent preflight wait longer than 60
  seconds when PostgreSQL was stopped. The developer-only preflight now caps
  PostgreSQL connection setup at three seconds, and both launchers print their
  URLs and foreground/`Ctrl+C` behavior before dependency checks.
- `pytest --cov`: 506 passed, 9 environment-gated skips; 86.85% coverage.
- PostgreSQL integration: 1 passed against the local Compose database.
- Ruff format/check and mypy strict: passed.
- Alembic check: no new upgrade operations detected.
- Frontend tests: 20 passed with 83.55% line coverage.
- Frontend lint, typecheck, and production build: passed.
- Real PowerShell startup: backend and frontend reached their documented ports.
- Browser smoke: dashboard, System, and Models & API loaded from the live backend;
  missing credentials remained explicit and did not block startup.
- Temporary processes and logs were removed after verification; no commit was
  created.
