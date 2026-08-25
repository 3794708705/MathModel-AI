# Architecture

## Current assessment

The repository began empty. The implemented stack is Python 3.12+, FastAPI,
Pydantic v2, SQLAlchemy 2, Alembic, PostgreSQL, and HTTPX. Phase 1 established
the infrastructure; Phase 2 adds structured problem understanding, candidate
exploration, deterministic model ranking, versioned prompts, quality gates, and
state/agent audit persistence. SQLite is used for database-independent tests;
PostgreSQL remains the production contract and migration target.

## Boundaries

```text
API -> reasoning workflow -> agent -> model router -> provider adapter
                      |          \-> versioned structured prompt
                      \-> deterministic quality/scoring -> ProblemState revision
                                                   \-> PostgreSQL audit evidence
```

- `api`: transport concerns and health reporting only.
- `schemas`: durable domain and orchestration contracts.
- `routing`: deterministic task classification and model-level selection.
- `providers`: the only code allowed to know vendor HTTP shapes.
- `agents`: shared execution, validation, retry, and audit behavior.
- `reasoning`: state transitions, prompt registry, deduplication, gates,
  deterministic scoring, orchestration, and repository boundary.
- `db`: persistence mapping and sessions; no reasoning logic.

External model calls happen outside database transactions. A successful stage
then uses a short transaction to retire the current state, insert one immutable
revision, record its `AgentRun`, and optionally record model-decision evidence.
The partial unique index on current state remains the final concurrency guard.

Later data files and generated artifacts belong in a file-store abstraction,
not database JSON blobs. The database stores identities, state, evidence links,
versions, and execution metadata.

## Technical debt

Intentional debt is bounded: no attachment/data pipeline, sandbox, solver,
literature verification, or paper system exists yet. Candidate semantic
deduplication uses a conservative local alias/token method rather than an
embedding service. Live-provider output quality has not been benchmarked; Mock
tests prove orchestration, not mathematical correctness.
