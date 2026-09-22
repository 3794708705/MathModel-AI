# Phase 8.2 — Lightweight Web Control Center

## Goal

Add a small React control center that consumes the accepted FastAPI contracts
for provider/model configuration, routing, projects, and benchmark inspection.
Keep backend mathematical, verification, paper, submission, and benchmark
decisions authoritative.

## Current State at Start

- Git baseline remains `30a7ae72cb28ad3845decb1b6c61ae95efa1cc95`.
- Phase 8 and 8.1 changes are intentionally uncommitted and must be preserved.
- Provider/model registration, capability probes, route preview, project-stage
  workflows, and individual benchmark reports already have FastAPI contracts.
- No frontend exists.
- The API has no secret-write boundary, runtime default-model preference,
  project list/current-state read, benchmark-run list, or routable-agent catalog.

## Files Affected

- `frontend/` for the Vite application, generated OpenAPI types, tests, and
  frontend-only tooling.
- Provider secret abstractions, API schemas/routes, application wiring, and one
  additive encrypted-secret migration.
- Reasoning and benchmark repositories/routes for read-only list adapters.
- Router preference lookup only for a persisted default preference; all hard
  filters and route evidence remain unchanged.
- Root configuration, documentation, and tests.

## Design

1. Generate TypeScript contracts from the actual FastAPI OpenAPI document and
   use one typed fetch boundary.
2. Store UI-submitted credentials only as AES-GCM ciphertext protected by
   `MM_SECRET_MASTER_KEY`. Providers retain an opaque `secret:` reference;
   `env:` resolution remains supported.
3. Persist the default model in the existing route-policy store under a reserved
   policy key. Router treats it as a preference and still performs its existing
   health, trust, capability, probe, and Mock exclusions.
4. Add only read adapters for current project summaries/state, benchmark run
   history, benchmark case catalog, and routable Agent names. These adapters do
   not compute readiness or orchestrate stages.
5. Use TanStack Query polling and backend-returned statuses. The browser never
   persists credentials or infers system readiness.

## Implementation Steps

1. Add and test encrypted secret storage, composite resolution, credential
   PUT/DELETE endpoints, and credential invalidation behavior.
2. Add and test default-model preference plus project/benchmark/Agent read APIs.
3. Add configured-origin CORS and regenerate OpenAPI types.
4. Build the seven scoped pages and reusable status/error/form components.
5. Add frontend unit/component/API-client/security tests and local browser QA.
6. Update `README.md`, `docs/WEB_UI.md`, security, architecture, testing, and
   migration documentation.

## Tests

- Backend secret plaintext persistence/response/logging, rotation, deletion,
  missing-master-key, and environment resolver regressions.
- Router rejection and successful preference tests; project and benchmark list
  API tests; CORS allow-list tests.
- Frontend typecheck/build, ESLint, API-client tests, Models & API tests,
  routing preview/save tests, browser-storage/DOM/URL credential tests, project
  and benchmark state tests.
- Full `pytest --cov`, Ruff, strict mypy, Alembic check, PostgreSQL integration,
  and `git diff --check`.

## Risks

- Secret plaintext leakage through validation or logging: use `SecretStr`, safe
  errors, no request-body logging, and adversarial tests.
- UI status drift: render backend enums verbatim and never duplicate readiness
  gates.
- Default-model bypass: persist only a preference and route through the existing
  deterministic hard filters.
- Large OpenAPI drift: keep generated types reproducible with a checked script.

## Acceptance Criteria

- All seven pages load against real FastAPI contracts with explicit loading,
  empty, error, and success states.
- Provider/model create/edit/disable, encrypted credential lifecycle, probes,
  default selection, and Agent route preview-before-save work.
- Project workflows are invoked only through backend workflow endpoints.
- Benchmark failures and blocked attempts remain visible.
- No API key appears in browser persistence, URL, post-submit DOM, responses,
  logs, or plaintext database fields.
- Backend Phase 1–8.1 regressions and all frontend/static/migration gates pass.
- No commit is created and Phase 8 remains blocked until a real provider is
  configured and validated.

## Self-Test Outcome

- Seven real-FastAPI pages, the typed OpenAPI client, encrypted credential
  bridge, runtime default preference, project/benchmark reads, and Agent catalog
  are implemented.
- Frontend: 6 files / 9 tests passed; statement coverage 68.59%; TypeScript,
  ESLint, generated-contract sync, and production Vite build passed.
- Backend: 492 passed, 8 truthful environment-gated skips, 86.87% coverage;
  Ruff, strict mypy, PostgreSQL integration, migration roundtrip/check, Docker
  sandbox/Solver/PDF paths, and diff checks passed.
- Browser QA covered Provider, encrypted credential clearing/persistence,
  Model, default preference, real Probe control, Router rejection/save lock,
  Project/Workspace, Benchmark, System, refresh, laptop, and narrow viewport.
- The optional frontend container build was attempted twice but Docker Hub OAuth
  endpoints were unreachable. Local production build and Compose configuration
  validation passed; no source or architecture change was made to hide that
  environmental failure.
- No commit was created. Phase 8 benchmark remains blocked by missing real
  provider configuration and was not resumed.
