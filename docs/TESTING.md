# Testing

Run the local quality gate:

```text
uv run ruff format --check .
uv run ruff check .
uv run mypy --strict src
docker build -t mathmodel-ai-sandbox:phase3 sandbox
docker build -f sandbox/solver.Dockerfile -t mathmodel-ai-solver:phase4 sandbox
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

Phase 3 tests use real parser libraries and Python calculations for CSV/XLSX/
PDF/image fixtures, profile missingness/duplicates/outliers/statistics, cross-
sheet relationships, MIME/path/size limits, Gemini inline-media payloads,
registry persistence, ProblemState v3, and FILES -> DATA -> EXECUTION ordering.
Sandbox unit tests verify exact Docker arguments and failure records; tests
marked `sandbox` require the image above and verify isolation plus timeout in a
real container. A Mock DataAgent is only an orchestration/schema fixture and is
never evidence that the data interpretation is correct.

Phase 4 tests cover typed expression/model schemas, symbol/parameter/equation
registries, unit PASS/FAIL/UNKNOWN behavior, MODEL and SOLVE gates, algorithm
selection, size/deadline/requirements-aware solver routing, status
canonicalization, feasibility/objective recomputation, immutable model digests,
and content-integrity evidence. Tamper regression covers objective, status, key
outputs, reciprocal result reference, model version/digest, program/executed
hashes, exit code, and Mock execution; every case must restore to valid. Tests
marked `solver` use the Phase 4 image and run real SciPy
LP, MILP, NLP, infeasible, unbounded, and OR-Tools CP-SAT cases. The complete E2E
uses Mock only for structured reasoning/MathModeler fixtures; its SciPy/HiGHS
numerical result and `ExecutionRecord` are real and `is_mock=false`.

The CodeAgent integration fixture may use a Mock provider only to return fixed
source code; that program then executes real SciPy in the real Docker sandbox,
emits the strict `result.json` contract, and must pass persisted evidence. The
paid full live chain is separately gated:

```text
MM_RUN_LIVE_PROVIDER_TESTS=1
MM_DEFAULT_PROVIDER=<non-mock provider with credential>
MM_DEFAULT_PROVIDER_MODEL=<real structured-output model>
uv run pytest tests/integration/test_live_mathematical_e2e.py
```

Without credentials the acceptance status is `SKIPPED_NO_CREDENTIALS`, never
`PASS`.

Gurobi integration is marked `gurobi`. It runs only with both
`MM_GUROBI_SANDBOX_IMAGE` and `MM_GUROBI_LICENSE_FILE`; otherwise it is skipped
with an explicit reason while unlicensed fallback remains mandatory and tested.

PostgreSQL migration verification is an integration check:

```text
docker compose up -d postgres
uv run alembic upgrade head
uv run alembic check
$env:MM_TEST_DATABASE_URL = "postgresql+psycopg://mathmodel:mathmodel@localhost:5432/mathmodel"
uv run pytest -m postgres
uv run alembic downgrade base
uv run alembic upgrade head
```

Run downgrade verification only against a dedicated disposable test database;
it intentionally removes migrated tables.

Phase 5 unit tests cover independent expression evaluation, variable/domain and
constraint rechecks, objective/key-output tampering, experiment provenance,
signed sensitivity, seeded robustness, blocked Bootstrap, deterministic Red
Team attacks, Mock review rejection, and scoped repair gates. The integration
E2E runs a real Phase 4 Docker solve followed by VALIDATE, real perturbation
executions, ROBUSTNESS, and Red Team persistence. A second E2E uses a non-Mock
Critical finding, creates model v2 under the same stable ID, performs another
real Docker solve, and repeats verification until Red Team passes.

The independent acceptance suite adds adversarial cases A-J: unresolved Critical
findings, `NOT_EVALUABLE`, SolverResult objective tampering, sensitivity baseline
replay, robustness input replay, persisted severity-count tampering, model-only
repair plus immutable v1/v2 evidence, three-cycle exhaustion, exact formal-result
selection, and Mock/`BLOCKED` rejection. The terminal gate also re-runs
independent validation and re-audits persisted experiment payloads and relational
execution fields before setting `verified_result_id`.

Migration `20260828_0006` is checked against PostgreSQL, including the six Phase
5 tables and critical foreign keys. The current complete suite has 175 passing,
4 explicitly skipped optional/environment-gated tests, and 87.47% statement/
branch coverage under the configured 85% floor. The dedicated PostgreSQL test
also passes when `MM_TEST_DATABASE_URL` is supplied. A real historical
competition benchmark remains Phase 8; Mock tests cannot satisfy that claim.
