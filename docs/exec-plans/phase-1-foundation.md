# Phase 1 — Foundation execution plan

## Goal

Establish a tested foundation on which later reasoning and computation stages
can be built without coupling business logic to vendors or losing traceability.

## Current state

The authorized target was an empty directory. No existing code, tests,
documentation, technical debt, or compatibility constraints existed. The two
repositories found under the original `System32` working directory were
unrelated and were not modified.

## Files affected

- `src/mathmodel_ai/`: API, configuration, persistence, state, providers,
  routing, and base agent contracts.
- `migrations/`: first PostgreSQL migration.
- `tests/`: unit, schema, provider, database, and API integration tests.
- root configuration and `docs/`: runtime, container, quality, and design docs.

## Design

- A single typed `ProblemState` is the shared source of workflow truth.
- SQLAlchemy stores only Phase 1 entities: project, problem, and versioned state.
- Provider adapters use documented HTTP APIs behind one async contract.
- Routing is deterministic, configuration-driven, and records model/reasoning
  recommendations even when an environment cannot switch models.
- `BaseAgent` owns validation, retry, routing, and run metadata; concrete
  reasoning agents remain Phase 2 work.

## Implementation steps

1. Create repository and quality configuration.
2. Add settings, structured logging, FastAPI health/system endpoints.
3. Add SQLAlchemy models, sessions, and Alembic migration.
4. Add `ProblemState`, task profiles, routing, providers, and `BaseAgent`.
5. Add tests and architecture/security/testing documentation.
6. Lock dependencies and run format, lint, type, and test checks.

## Tests

- Schema validation and JSON round trips.
- Routing minimums and escalation behavior.
- Mock and HTTP provider request/response contracts.
- Agent validation/retry behavior.
- Database metadata and API readiness integration.
- Alembic upgrade against PostgreSQL when Docker is available.

## Risks

- Vendor APIs evolve: isolate payloads and test each adapter with captured
  contract shapes.
- PostgreSQL may be unavailable locally: keep unit tests independent and make
  the real migration test explicit.
- JSON state can drift: version state and validate through `ProblemState` at
  every boundary.

## Acceptance criteria

- Application starts with validated configuration.
- PostgreSQL schema can migrate from zero to head.
- All Phase 1 interfaces have real implementations or explicit abstract
  contracts; no core `pass`/TODO placeholders.
- Ruff, mypy, and pytest pass.
- No Phase 2 agent behavior is introduced.

## Outcome

Completed on 2026-08-25. The final gate passed Ruff, strict mypy, 27 tests
(including a real PostgreSQL JSONB integration test), 89% branch-aware coverage,
an Alembic upgrade/downgrade/upgrade cycle on PostgreSQL 17, a production image
build, and a non-root container readiness check against PostgreSQL.
