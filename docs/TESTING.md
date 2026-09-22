# Testing

## Independent verification extension

Focused tests cover known metric vectors, false reported metrics, NaN/Inf and
undefined R2, objective/constraint recalculation, unknown metric/version input,
tampered plans/jobs/artifacts, missing obligations, exact-result binding,
concurrent idempotency, and Benchmark AND-gate behavior. Real Docker tests run a
seeded ODE replay and deterministic LP scenario, prove distinct execution/code
identity after changed input, and reject a real failed execution. PostgreSQL is
tested on a dedicated database for migration upgrade/downgrade, restart readback,
foreign keys, concurrent claims, and physical replay evidence persistence.

Case A has a reviewed `independent-verification.json` sidecar bound to the v2
modeling contract. This does not verify historical attempts: a fresh exact-model
solve and its independent evidence chain are still required. Cases B/C remain
unaccepted; missing reviewed policies are mandatory `NOT_READY`, not a test skip
or permission to invent thresholds.

Run the local quality gate:

```text
uv run ruff format --check .
uv run ruff check .
uv run mypy --strict src
docker build -t mathmodel-ai-sandbox:phase3 sandbox
docker build -f sandbox/solver.Dockerfile -t mathmodel-ai-solver:phase4 sandbox
docker build -f sandbox/paper.Dockerfile -t mathmodel-ai-paper:phase6 .
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

Phase 6 tests cover EvidenceBuilder selection of the exact
`verified_result_id`, verified-chain and Mock rejection, supported-assumption
filtering, structured numeric/comparison/unit claims, metadata and citation
support separation, Paper IR and registry identities, cross-references,
symbol/equation integration, figure/table hashes, LaTeX injection and path
traversal, manifest integrity, and deterministic paper quality gates. Acceptance
cases A-Z include numeric text/value drift, objective/solver/optimality/unit
tampering, stale or unsupported citation decisions, equation-render drift,
figure/table source swaps, Critical Red Team findings, missing subproblem and
abstract coverage, unsupported robustness/significance claims, LaTeX injection,
PDF/manifest byte replacement, immutable version snapshots, missing document
objects, and Mock evidence promotion.

The Paper E2E performs a real Phase 4 solve and Phase 5 verification chain,
builds Phase 6 evidence and Paper IR, generates real assets, compiles with
XeLaTeX/BibTeX in `mathmodel-ai-paper:phase6`, validates the produced PDF and
manifest, and persists the complete paper version. Live tests remain gated:

```text
MM_RUN_LIVE_LITERATURE_TESTS=1  # otherwise SKIPPED_NO_NETWORK
MM_RUN_LIVE_PROVIDER_TESTS=1    # otherwise SKIPPED_NO_CREDENTIALS
```

Migration `20260828_0007` must be upgraded, downgraded to `20260828_0006`,
re-upgraded, and checked against PostgreSQL. The dedicated PostgreSQL test
asserts Phase 6 tables and critical foreign keys. The independent acceptance run
completed with 230 passing tests, 5 explicitly skipped optional/environment-
gated tests, and 87.65% statement/branch coverage under the configured 85%
floor.

Phase 7 adds named adversarial cases A-Z for missing/fake requirement coverage,
page/filename/MIME/anonymity/secret/code-policy failures, unverified rules,
Critical and score tampering, freeze success, paper/figure/code/data/manifest/ZIP
tampering, package swap, forbidden files, absolute paths, safe roundtrip, exact
paper-version selection, correction invalidation, and a clean frozen package.
Additional tests reject ZIP slip, symlinks, compression bombs, persisted count
tampering, and deadline-mode bypass. The final E2E extends a real Phase 5 solve
and Phase 6 Docker-compiled PDF through deterministic Phase 7 gates, a Mock
structured FinalJury fixture, real filesystem manifest/ZIP, re-open validation,
PostgreSQL persistence, and post-freeze DIRTY detection.

Migration `20260829_0008` must be upgraded from `20260828_0007`, downgraded,
re-upgraded, and checked against PostgreSQL. The built-in profile remains
`TEST_FIXTURE`; live competition-rule conformance is not tested in Phase 7.
The Phase 7 self-test run completed with 306 passing tests, 6 explicitly skipped
environment-gated tests, and 88.49% statement/branch coverage under the
configured 85% floor. Dedicated Phase 7 coverage reached 100% for submission
checks, corrections, freeze, profiles, and package construction; 99% for Final
Jury, 98% for requirement coverage, 98% for package integrity, and 94% for the
competition RuleEngine. PostgreSQL migration check and integration also passed
against migration head `20260829_0008`.

The Phase 7 independent acceptance suite adds profile JSON/child-rule tampering,
fixture promotion, exact v1/v2 snapshots, required-output claim-role checks,
required `NOT_APPLICABLE` rejection, stale Paper/Jury/finding invalidation,
Freeze-time coverage recomputation, PDF metadata/attachment attacks,
Unicode-obfuscated identities, Windows/Unix/UNC/file-URL path variants, Bearer/
database/private-key patterns, internal fixture/Mock leaks, deterministic and
concurrent-idempotent package builds, snapshot/status tampering, NFC/case archive
collisions, nested archives, dynamic deadline rechecks, fail-closed reproduction,
and correction-classification tampering. The real E2E additionally reopens the
physical ZIP after source context, detects persisted Jury tampering, proves exact
record restoration policy, and detects final package byte replacement.

Phase 8 tests cover three distinct official manifests, verified rule provenance,
blind solve/evaluation separation, hash/size/source tamper, prompt injection as
data, immutable attempts and reruns, source-tree identity, deterministic score/
status/cost/intervention recomputation, fake live-provider/literature rejection,
budget and failure terminalization, report redaction, and the benchmark API:

```text
uv run pytest tests/benchmark tests/integration/test_benchmark_api.py
$env:MM_RUN_LIVE_LITERATURE_TESTS = "1"
uv run pytest tests/integration/test_live_literature_phase6.py
```

The live historical run uses the three `benchmarks/case-*` manifests and the
existing secure provider configuration. It must not run with Mock. Without a
non-Mock credential, the required result is `NOT_READY` with
`LIVE_PROVIDER=BLOCKED`; this is not an allowed acceptance skip. Live Crossref
search plus independent DOI resolution passed in the Phase 8 formal run.

Deadline behavior is covered by all five Phase 7 modes plus the Phase 8
versioned mode list; anonymity, secret, required-subproblem, validation, and
hard-fail gates never weaken under deadline pressure. Recovery regression proves
a thrown live-pipeline exception reaches an immutable terminal attempt and a
later run cannot remove the first history. Automatic process crash/resume and
asynchronous human cancellation are not implemented and must remain listed as
limitations.

Migration `20260830_0009` adds the six benchmark-history tables. Verify it by
upgrading from `20260829_0008`, downgrading back to that revision on a disposable
database, re-upgrading, running `alembic check`, and executing the PostgreSQL
integration marker. Formal report artifacts are rebuilt from database records;
changing a persisted summary cannot change acceptance.

## Phase 8.1 provider acceptance

The configurable-provider suite uses `httpx.MockTransport` and in-memory
registries; no real provider network is needed for deterministic compatibility/
security tests. The independent adversarial suite adds numeric/IPv6/mapped and
mixed-DNS SSRF, encoded traversal, all-3xx redirects, trust/lookalike domains,
secret rotation/removal, public API redaction, persisted config/probe tamper,
capability-summary spoofing, authentication/count tamper, exact task/probe route
binding, preview races, DNS address pinning with original Host/SNI,
decompression limits, cancellation/timeout classes,
usage/pricing bounds, tool-name validation, and immutable configured-vs-reported
model identity. It also executes three independent fake-transport E2E paths:
OpenAI-compatible native schema, partial-compatible prompt JSON, and constrained
custom JSON HTTP, each through Provider → Probe → Router → ProblemAgent →
AgentRun persistence.

```text
uv run pytest tests/providers tests/routing/test_registry_router.py
uv run pytest tests/integration/test_provider_registry_api.py
uv run pytest tests/integration/test_custom_provider_e2e.py
uv run pytest tests/integration/test_provider_protocol_e2e.py
uv run pytest tests/benchmark/test_provider_history.py
```

Paid/live compatibility gates are separate and truthful:

```text
uv run pytest tests/integration/test_live_custom_provider_phase81.py
```

They require the DeepSeek, Qwen, or generic custom environment configuration
documented in `CUSTOM_PROVIDERS.md`. Missing credentials produce
`SKIPPED_NO_CREDENTIALS`; a skip is not a provider PASS.

Migration `20260830_0010` adds provider/model/probe/Agent-route registries,
canonical probe digests, and nullable historical trace columns. On the
disposable PostgreSQL test database,
verify exactly:

```text
uv run alembic upgrade 20260830_0010
uv run alembic downgrade 20260830_0009
uv run alembic upgrade 20260830_0010
uv run alembic check
$env:MM_TEST_DATABASE_URL = "postgresql+psycopg://mathmodel:mathmodel@localhost:5432/mathmodel"
uv run pytest tests/integration/test_postgres.py
```

The PostgreSQL suite includes ProviderEndpoint canonical digest creation,
reload through a fresh registry instance, deterministic update, and direct-row
tamper rejection. Provider/API regressions also cover the distinction between
an integrity-valid configuration blocked by current endpoint policy and a true
digest mismatch.

The downgrade/upgrade cycle must preserve the existing Phase 8 baseline before
formal use. Never run it against an unreviewed production database.

The independent acceptance run completed with 482 passed, 8 truthful
environment-gated skips, and 86.87% combined statement/branch coverage. Real
PostgreSQL migration/integration, Docker sandbox, deterministic solver,
verification-repair, PDF, Phase 4-7 adversarial, Phase 8 benchmark, and Phase 8.1
security tests all ran. DeepSeek, Qwen, and generic custom live gates remain
`SKIPPED_NO_CREDENTIALS`; Gurobi remains separately license-gated.

## Phase 8.2 Web control center

Frontend verification is deterministic and uses mocked HTTP only at the API
boundary. Runtime code always calls FastAPI. Tests cover provider list/create/
validation/disable, write-only credential clearing and browser non-persistence,
model creation/probe/capability display/disable, Router preview rejection before
save, project create/workflow delegation, and visible failed/blocked benchmark
attempts.

```text
cd frontend
npm ci
npm run api:generate
npm run typecheck
npm run lint
npm run test:coverage
npm run build
```

Backend additions are covered by encrypted-secret plaintext, rotation,
deletion, missing-key, environment-resolver, API redaction, runtime-default hard
filter, project/benchmark read adapter, Agent catalog, and CORS regression tests.
Migration `20260831_0011` creates only `encrypted_secrets`; verify its PostgreSQL
downgrade/re-upgrade roundtrip from `20260830_0010` before self-test reporting.

The independent UI acceptance suite extends that baseline with encrypted-row
and stale-health tamper, direct route-write races, unknown Agent/model, stale or
missing Probe, Mock routing, mass-assignment/readback, proxy trust spoofing,
network/error secret redaction, mutation-cache exclusion, duplicate-submit and
ambiguous-commit reconciliation, XSS-safe rendering, polling termination, and
visible formal benchmark rejection tests. A controlled fake Provider is used
only to prove Provider → credential → model → Probe → Router UI wiring; it never
establishes live-provider or Phase 8 benchmark readiness.

```text
uv run pytest --cov
uv run ruff format --check .
uv run ruff check .
uv run mypy --strict src
uv run alembic check

cd frontend
npm run typecheck
npm run lint
npm run test:coverage
npm run build
```

After the frontend build, scan `frontend/dist` for synthetic credential markers,
secret environment names, source maps, and private-key/API-key patterns. The
PostgreSQL acceptance database must independently run
`20260830_0010 → 20260831_0011 → 20260830_0010 → 20260831_0011`; never perform
that destructive roundtrip on an unreviewed production database.

## Local startup acceptance

### Continuous frontend checks

The independent frontend CI job uses Python 3.12 and Node 22, installs locked
dependencies with `uv sync --locked --dev` / `npm ci`, regenerates the OpenAPI
TypeScript contract and fails if `frontend/src/api/schema.d.ts` differs from
the committed backend contract. It then runs lint, component tests with
coverage reporting, and the typechecked production build. No live Provider,
database, paid key, or scientific benchmark run is needed for that job.
After backend API changes, run `npm run api:generate` in `frontend` and include
the generated type changes with the implementation.

System-page regressions cover unavailable database vs live backend, redacted
database error messages, unknown system configuration on failure, stale-success
invalidation, and a manual recheck that recovers all three health/system queries.
The backend health test verifies HTTP 503 with `DATABASE_UNAVAILABLE`, continued
liveness, and subsequent database recovery without exposing connection details.

### Startup and health

The backend has one ASGI factory contract: `mathmodel_ai.main:create_app`.
Startup regressions construct the real FastAPI graph and request OpenAPI, docs,
liveness, and system status. CLI tests cover help, invalid ports, factory mode,
and host/port/reload forwarding without binding a permanent socket. Preflight
tests cover a fresh database without a master key, existing encrypted rows with
the key missing, redacted output, runtime directories, and absent live Provider
credentials.

```text
uv run pytest tests/test_cli.py tests/integration/test_startup.py
python -m mathmodel_ai --help
python -m mathmodel_ai serve --help
```

Manual acceptance uses `scripts/dev-backend.ps1` and
`scripts/dev-frontend.ps1`, then requests `/docs`, `/openapi.json`,
`/health/live`, `/health/ready`, and the frontend root. Do not use
`mathmodel_ai.main:application`; it is not a module attribute.
Run the backend once with PostgreSQL stopped to confirm that the preflight
reports the unavailable dependency within its bounded connection timeout, then
repeat with the Compose `postgres` service healthy and confirm readiness. Both
development scripts are foreground processes and must stop cleanly with
`Ctrl+C`.
