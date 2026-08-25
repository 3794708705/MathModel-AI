# Testing

Run the local quality gate:

```text
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest --cov=mathmodel_ai --cov-report=term-missing
```

Unit tests must not require API keys or network access. HTTP provider tests use
in-memory transports and assert documented request/response shapes. Mock
providers always expose `is_mock=true`.

Phase 2 tests cover problem decomposition, ambiguity and assumption handling,
candidate/model-chain contracts, semantic deduplication, exact jury arithmetic,
hard failures, stage transitions, API behavior, immutable state revisions,
agent traces, model decisions, and a complete Mock reasoning E2E. The optional
paid-provider check runs only when `MM_RUN_LIVE_PROVIDER_TESTS=1` and an
appropriate non-Mock provider is configured.

PostgreSQL migration verification is an integration check:

```text
docker compose up -d postgres
uv run alembic upgrade head
$env:MM_TEST_DATABASE_URL = "postgresql+psycopg://mathmodel:mathmodel@localhost:5432/mathmodel"
uv run pytest -m postgres
uv run alembic downgrade base
uv run alembic upgrade head
```

Later phases add attachment, sandbox, solver, and real competition benchmark
tests; Mock tests cannot satisfy mathematical or benchmark acceptance.
