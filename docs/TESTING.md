# Testing

Run the local quality gate:

```text
uv run ruff format --check .
uv run ruff check .
uv run mypy --strict src
docker build -t mathmodel-ai-sandbox:phase3 sandbox
docker build -f sandbox/solver.Dockerfile -t mathmodel-ai-solver:phase4 sandbox
docker build -f sandbox/paper.Dockerfile -t mathmodel-ai-paper:phase6 sandbox
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
