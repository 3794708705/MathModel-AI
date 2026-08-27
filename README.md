# MathModel AI

MathModel AI is an evidence-first automation system for mathematical modeling
competitions. Phases 1–4 are implemented: foundation, persisted reasoning,
guarded data ingestion, isolated execution, structured mathematical models,
deterministic solver routing, and traceable real numerical results.

## Quick start

```powershell
Copy-Item .env.example .env
docker compose up -d postgres
uv sync --dev
uv run alembic upgrade head
docker build -t mathmodel-ai-sandbox:phase3 sandbox
docker build -f sandbox/solver.Dockerfile -t mathmodel-ai-solver:phase4 sandbox
uv run uvicorn mathmodel_ai.main:app --reload
```

The API exposes liveness at `/health/live`, database readiness at
`/health/ready`, and non-secret runtime metadata at `/api/v1/system`. Phase 2
reasoning is available under `/api/v1/projects/{project_id}`; create a fixture
project with `POST /api/v1/projects`, then call `POST
/api/v1/projects/{project_id}/reasoning/run`.

Phase 3 endpoints under the same project prefix accept files, expose file/
dataset/profile/artifact registries, run structured data understanding, and
execute explicitly submitted Python only after the FILES and DATA gates pass.
See [docs/DATA_EXECUTION.md](docs/DATA_EXECUTION.md) for contracts and limits.

Phase 4 extends the main workflow from `SELECT` through `MODEL` and `SOLVE`.
`POST /api/v1/projects/{project_id}/mathematical/run` constructs one typed,
solver-independent model, applies the MODEL gate, and selects `AUTO`,
`DETERMINISTIC`, or `GENERATED` execution. `AUTO` remains deterministic-first;
custom generated programs pass through CodeAgent, approved dependency checks,
the same real Docker sandbox, and a strict `result.json` contract. The SOLVE gate
requires recomputed feasibility plus content-consistent model/run/program/code/
execution/result evidence before a result is verified. See
[docs/MATHEMATICAL_CORE.md](docs/MATHEMATICAL_CORE.md) and
[docs/SOLVER_ARCHITECTURE.md](docs/SOLVER_ARCHITECTURE.md).

## Verification

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy --strict src
uv run pytest --cov=mathmodel_ai --cov-report=term-missing
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for boundaries and
[docs/REASONING_CORE.md](docs/REASONING_CORE.md) for the Phase 2 contracts.

## Known limitations

- Mock reasoning verifies schemas, routing, workflow, and persistence only; it
  does not prove that a candidate is mathematically good.
- Live provider calls require user-supplied keys and an explicit paid-test opt-in.
- The local file store has no S3/MinIO adapter, scanned-PDF OCR, or orphan cleanup.
- Phase 4 adapters intentionally cover flattened scalar LP/MILP/integer models,
  basic continuous NLP, and integral CP-SAT input—not every model family in the
  extensible schema.
- Gurobi is optional and needs a separately configured image, `gurobipy`, and a
  runtime license file; SciPy/OR-Tools remain usable without it.
- Full validation, sensitivity, robustness, red team/model repair, literature,
  citation, and paper pipelines belong to later phases.
- FastAPI's current test client emits an upstream `httpx2` migration warning.
