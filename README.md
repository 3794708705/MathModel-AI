# MathModel AI

MathModel AI is an evidence-first automation system for mathematical modeling
competitions. Phases 1 and 2 are implemented: the foundation plus a persisted
reasoning chain from raw problem text to a primary and backup model candidate.

## Quick start

```powershell
Copy-Item .env.example .env
docker compose up -d postgres
uv sync --dev
uv run alembic upgrade head
uv run uvicorn mathmodel_ai.main:app --reload
```

The API exposes liveness at `/health/live`, database readiness at
`/health/ready`, and non-secret runtime metadata at `/api/v1/system`. Phase 2
reasoning is available under `/api/v1/projects/{project_id}`; create a fixture
project with `POST /api/v1/projects`, then call `POST
/api/v1/projects/{project_id}/reasoning/run`.

## Verification

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest --cov=mathmodel_ai --cov-report=term-missing
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for boundaries and
[docs/REASONING_CORE.md](docs/REASONING_CORE.md) for the Phase 2 contracts.

## Known limitations

- Mock reasoning verifies schemas, routing, workflow, and persistence only; it
  does not prove that a candidate is mathematically good.
- Live provider calls require user-supplied keys and an explicit paid-test opt-in.
- Sandbox, solver, attachment, evidence, and paper pipelines belong to later phases.
- FastAPI's current test client emits an upstream `httpx2` migration warning.
