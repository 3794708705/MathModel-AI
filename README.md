# MathModel AI

MathModel AI is an evidence-first automation system for mathematical modeling
competitions. Phases 1–7 are implemented: foundation, persisted reasoning,
guarded data ingestion, isolated execution, structured mathematical models,
deterministic solver routing, traceable real numerical results, independent
validation, executed perturbation experiments, adversarial review, and bounded
model repair, verified literature, evidence-linked Paper IR, reproducible
figures/tables, sandboxed LaTeX/PDF rendering, versioned competition rules,
deterministic final checks, and hash-frozen submission packages.

Phase 8 adds immutable real-case benchmark runs, blind solve/evaluation
isolation, hash-pinned official resources, a source-verified COMAP MCM 2024
profile, deterministic score/acceptance recomputation, failure and human-
intervention registries, budget gates, and JSON/Markdown reports. Its first
formal three-case run is intentionally reported as `NOT_READY`: official source
and live Crossref checks passed, but no live model-provider credential was
available, so no case entered reasoning, solving, paper production, or package
freeze. Mock was not used as a fallback.

Phase 8.1 adds a persisted provider/model registry, OpenAI-compatible and
constrained custom-JSON adapters, environment-only secret resolution, capability
probing, SSRF/TLS/redirect/response-size controls, capability-based and
Agent-specific routing, exact AgentRun/BenchmarkRun provider identity, and
non-destructive provider/model APIs. Native OpenAI, Google, Anthropic, and Mock
paths remain available; legacy provider/model settings remain deprecated but
compatible. See [docs/CUSTOM_PROVIDERS.md](docs/CUSTOM_PROVIDERS.md) and
[docs/MODEL_REGISTRY.md](docs/MODEL_REGISTRY.md).
The deterministic Phase 8.1 self-test is `SELF_TEST_READY`; no live credential
was present, so this status does not unblock the Phase 8 benchmark.

Phase 8.2 adds a lightweight React control center for provider/model setup,
Router-governed Agent preferences, projects, workflow entry points, benchmark
history, and system status. It consumes the FastAPI OpenAPI contract and never
reimplements readiness or workflow orchestration in the browser. See
[docs/WEB_UI.md](docs/WEB_UI.md).

## Local Development

The tested local baseline is **Python 3.12**. On Windows, use the repository
`.venv` directly; do not rely on an activated Anaconda base Python 3.13 or
another `python` found earlier on `PATH`.

Synchronize dependencies once, then migrate PostgreSQL:

```powershell
Copy-Item .env.example .env
docker compose up -d postgres
uv sync --dev
uv run alembic upgrade head
docker build -t mathmodel-ai-sandbox:phase3 sandbox
docker build -f sandbox/solver.Dockerfile -t mathmodel-ai-solver:phase4 sandbox
docker build -f sandbox/paper.Dockerfile -t mathmodel-ai-paper:phase6 sandbox
```

`uv sync` is intentionally not run on every startup. The single recommended
local backend launcher is:

```powershell
Set-Location C:\Users\zeon\Projects\MathModel-AI
.\scripts\dev-backend.ps1
```

It resolves the repository, calls `.\.venv\Scripts\python.exe` directly, runs a
non-secret startup preflight, and starts the `create_app` Uvicorn factory with
reload enabled. The command intentionally stays in the foreground; keep that
PowerShell window open and press `Ctrl+C` to stop the backend. The preflight
bounds a failed PostgreSQL connection attempt and reports database unavailability
instead of waiting silently. Its underlying stable CLI supports custom
development options:

```powershell
.\.venv\Scripts\python.exe -m mathmodel_ai serve --host 127.0.0.1 --port 8000 --reload
```

`mathmodel_ai.main:application` is not an ASGI target: `application` is local to
`create_app`. Do not use that obsolete command.

## Web UI

With the backend running on port 8000, use the frontend launcher:

```powershell
.\scripts\dev-frontend.ps1
```

Run it in a second PowerShell window; it also remains in the foreground until
`Ctrl+C`. The script does not install packages. If `frontend/node_modules` is
absent, run `npm install` in `frontend` first.

Open `http://127.0.0.1:5173`. Backend API and FastAPI documentation are at
`http://127.0.0.1:8000` and `http://127.0.0.1:8000/docs`. Use
`VITE_API_BASE_URL` when the backend is not at
`http://127.0.0.1:8000`. Browser-entered provider credentials require a random
URL-safe base64 32-byte `MM_SECRET_MASTER_KEY` on the backend; keys are encrypted
server-side, are never returned, and are cleared from component state after a
successful write. Environment-backed `credential_ref` values remain supported.
Without a master key, a new local environment still starts and displays the
encrypted credential store as not configured. If encrypted rows already exist,
they remain unavailable and providers fail closed until the original key is
restored. A missing live Provider credential never blocks the control center.
For an integrated local image, run `docker compose up --build` after creating
the untracked `.env`.

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

Phase 5 extends an accepted `SOLVE` result through `VALIDATE`, `SENSITIVITY`,
`ROBUSTNESS`, and `RED_TEAM`. Validation recomputes variables, constraints,
objective, expected outputs, and declared checks without trusting the Phase 4
feasibility record. Sensitivity and robustness perturb immutable model copies
and require a real `ExecutionRecord` for every accepted scenario. A non-Mock
Critical Red Team finding enters `MODEL_REPAIR`; an accepted repair creates the
next model version and must be solved and verified again. Automatic repair is
capped at three cycles. SOLVE leaves results `UNVERIFIED`; only the final
deterministic gate sets `verified_result_id` for the exact accepted formal result. See
[docs/VERIFICATION_REPAIR.md](docs/VERIFICATION_REPAIR.md).

Phase 6 consumes only the explicit Phase 5 `verified_result_id`. `POST
/api/v1/projects/{project_id}/paper/run` builds an immutable evidence snapshot,
retrieves and verifies reference metadata, links structured claims to evidence,
generates reproducible figures/tables, renders deterministic LaTeX/BibTeX, and
compiles a real PDF in a network-disabled non-root container. Paper records and
artifacts are available from the `/literature`, `/paper`, and
`/paper/artifacts` project routes. See
[docs/PAPER_PIPELINE.md](docs/PAPER_PIPELINE.md).

Phase 7 binds an explicit accepted paper and verified result to one immutable
`CompetitionProfile` version. `POST /api/v1/projects/{project_id}/final/run`
recomputes requirement coverage and competition rules, runs structured Final
Jury review behind deterministic gates, performs submission checks, and can
freeze a real manifest/ZIP when `freeze_on_pass=true`. Frozen artifacts are
rehash-verified on read; modification returns `DIRTY`, never silently ready.
See [docs/COMPETITION_PROFILE.md](docs/COMPETITION_PROFILE.md),
[docs/FINAL_JURY.md](docs/FINAL_JURY.md), and
[docs/SUBMISSION_PIPELINE.md](docs/SUBMISSION_PIPELINE.md).

Phase 8 endpoints are `POST /api/v1/benchmarks/runs`, `GET
/api/v1/benchmarks/runs/{id}`, `GET /api/v1/benchmarks/runs/{id}/cases`, and
`GET /api/v1/benchmarks/runs/{id}/report`. Formal runs bind the full Git commit
and source-tree digest, versioned configuration, exact case/profile digests,
every attempt, atomic metrics, failures, and interventions. See
[docs/FINAL_PROJECT_ACCEPTANCE.md](docs/FINAL_PROJECT_ACCEPTANCE.md).

Provider setup uses `/api/v1/providers`, `/api/v1/models`, and
`/api/v1/model-routing`. Configure secrets as environment variables referenced
by `credential_ref`, or submit them once through the encrypted Web UI bridge;
responses never contain raw keys or secret references. Run a model capability
probe and a route preview before a paid Agent smoke. A successful proxy call
proves protocol behavior, not official model identity.

## Capability status

### Implemented

- Phase 1-7 application workflows and Phase 8 benchmark persistence, gates,
  reports, API, official manifests, resource validation, and profile isolation.
- Phase 8.1 provider/model registries, capability probe, secure custom endpoint
  adapters, capability routing, migration, APIs, and audit trace fields.
- Deterministic rejection of Mock/non-live promotion, score/status/count tamper,
  hidden attempts, evaluation leakage, source tamper, and budget overruns.

### Validated

- Phase 1-7 unit, integration, adversarial, PostgreSQL, Docker solver, real PDF,
  and real package paths under their documented fixture/environment boundaries.
- Phase 8 manifest/profile integrity, blind loading, report recomputation,
  exception terminalization, retry-history retention, deadline/budget gates,
  source-tree identity, API, and live Crossref metadata resolution.

### Benchmark-tested

- Official COMAP MCM 2024 Problems A, B, and C were downloaded from hash-pinned
  official URLs and structurally inspected; Problem C includes its official CSV
  and data dictionary.
- The full live competition solve is not benchmark-validated. All three formal
  cases are `BLOCKED_ENVIRONMENT` at `LIVE_PROVIDER`, with zero provider calls,
  tokens, cost, solver runs, papers, and packages.

### Environment-dependent

- Live reasoning/paper/final-jury execution requires a user-supplied non-Mock
  provider credential. Gurobi needs its optional licensed runtime. Crossref and
  official benchmark retrieval need network access; Docker is required for
  isolated solving and PDF compilation.

### Not implemented

- A local scanned-PDF OCR engine, S3/MinIO storage adapter, automatic benchmark
  crash/resume coordinator, asynchronous human-cancel endpoint, independent
  benchmark LLM/human evaluator, and general submission-code reproduction
  executor.

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
- Bootstrap robustness is explicitly blocked until a dataset resampling
  contract binds sampled rows to model parameters; it is never approximated by
  parameter noise.
- Live literature and PaperAgent checks remain explicit opt-ins; their default
  skip statuses are not evidence of success.
- The normal submission API keeps its deterministic test profile isolated from
  the verified COMAP 2024 benchmark profile. The latter is a historical
  technical-benchmark ruleset, not a claim that current COMAP rules are identical.
- Phase 8 cannot reach `SELF_TEST_READY` until at least one configured live
  provider completes the critical agent chain and at least two cases produce
  verified papers/packages. A missing credential is a blocker, not an allowed
  acceptance skip.
- Benchmark cost enforcement is post-attempt for an in-flight provider chain;
  it prevents acceptance and blocks later cases but does not stream-cancel an
  already-issued provider request.
- Problem-understanding accuracy, model appropriateness, and Red Team usefulness
  remain zero/Human Review until an independent evaluator scores them; workflow
  stage success is not treated as rubric accuracy.
- FastAPI's current test client emits an upstream `httpx2` migration warning.
