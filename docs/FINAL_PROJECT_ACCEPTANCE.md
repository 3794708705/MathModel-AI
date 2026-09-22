# Final project acceptance status

## Decision

```text
PHASE_8_1_PROVIDER_ACCEPTANCE = READY
PHASE_8_BENCHMARK = NOT_READY
FULL_PROJECT_READINESS = NOT_READY
```

Phase 8.1's configurable provider architecture passed deterministic self-test,
independent adversarial acceptance, and subsequent human acceptance. The real
DeepSeek provider smoke is now accepted. The original preflight-blocked run
details below are historical snapshots, not the current provider status.

## Engineering regression update (2026-09-22)

GitHub Linux CI run `35696926010` at commit `854194e` passed with 639 tests,
8 explicit environment-gated skips, and 86.82% coverage. Ruff, strict mypy,
fresh PostgreSQL migrations, and all three sandbox builds passed. That run
covered the backend; it did not run frontend checks. The working-tree follow-up
adds an independent frontend CI job and refreshes generated API types, plus
database-outage diagnostics and unknown/stale-state regressions. None of these
engineering checks promote live Case A or overall project readiness.

## Current live acceptance update (2026-09-06)

Case A rerun 13 (`104df85e-8f04-46fe-a7ff-96c744fae9b5`) is FAIL / NOT_READY.
It retained eight real provider calls and two real generated Docker executions.
The first result failed schema validation; a new immutable program then produced
a FEASIBLE result and passing SOLVE gate. Independent validation remained
NOT_EVALUABLE: dynamic/scenario obligations were not independently evaluated and
the objective-based verification path does not support this no-objective model.
No verified result, paper, PDF, or package exists for this run. B/C have not been
resumed. No commit was created.

The generic Phase 5 extension now supports versioned independent metric
recomputation and fresh scenario replay with immutable source, calculator,
input, output, execution, and artifact digests. Its adversarial, real Docker,
and isolated PostgreSQL acceptance tests pass. This does not retroactively
verify Case A: rerun 13 still has no `verified_result_id`.

The follow-up Case A contract-completion review created a version-2 formal model
and reviewed `independent-verification.json`. The policy binds the official
problem SHA-256, manifest, exact model digest, model-contract digest, independent
Red Team report, and independent Model Jury report. The contract explicitly
labels the normalized reference state, nondimensional time, midpoint fixed-ratio
comparator, provisional coefficients, +1% local sensitivity design, local
Jacobian stability definition, mathematical persistence boundary, and
representative non-lamprey parasite-guild interpretation as ASSUMPTION or
DERIVATION. The prior version-1 DRAFT remains archived byte-semantically. No
fresh Case A attempt, B/C attempt, paper, PDF, or package was created by this
work, so Phase 8 remains NOT_READY pending a fresh exact-model run.

A read-only independent audit found a false requirement PASS caused by keyword
matching ("variable" in a Jacobian stability request). Validator 5.0.1 now keeps
that request UNCHECKED and detects the stale report. Benchmark failure metrics
also incorrectly recorded zero formal solver calls despite two persisted
execution links; new outcomes count the real links. Historical report/metric
bytes remain unchanged, with this correction documented rather than backfilled.
See `exec-plans/phase-8-case-a-live-acceptance.md` for regressions and limitations.
Final-tree regression after the independent-verification extension: 606 passed,
8 environment-gated skips, 87.10% backend coverage, and 41 frontend tests passed.
Ruff, strict mypy, Alembic check, frontend typecheck/lint/build, and diff check
passed. These are software regression results, not a live Case A acceptance PASS.

## Architecture summary

MathModel AI is an evidence-first FastAPI/Pydantic/SQLAlchemy system. Phase 1-7
projects advance through persisted immutable state revisions; provider SDKs are
behind adapters, calculations and optimization execute in bounded non-root
Docker sandboxes, Phase 5 independently verifies the exact formal result, Phase
6 builds a claim/evidence graph and real PDF, and Phase 7 rechecks an immutable
competition profile before freezing a digest-bound package.

Phase 8 is a separate cross-project audit plane. It binds a full Git commit and
source-tree digest, exact case/profile/config digests, blind official resources,
every formal attempt, atomic metrics, failures, interventions, result links, and
deterministically rebuilt JSON/Markdown reports. Persisted totals and latest-
result shortcuts are not acceptance authority.

## Phase status

| Phase | Status | Evidence boundary |
|---|---|---|
| 1 Foundation | ACCEPTED | configuration, API, database, provider/router, state, tests |
| 2 Reasoning Core | ACCEPTED | structured Mock E2E plus deterministic gates; Mock is not math truth |
| 3 Data/Execution | ACCEPTED | real parsers/profilers and isolated Python execution |
| 4 Mathematical/Solver | ACCEPTED | typed models, real SciPy/OR-Tools execution, exact evidence |
| 5 Verification/Repair | ACCEPTED | independent recalculation, real perturbations, immutable repair loop |
| 6 Paper | ACCEPTED | evidence snapshots, verified citation pipeline, real Docker PDF |
| 7 Submission | ACCEPTED | deterministic profile/jury/check/package integrity |
| 8 Real Benchmark | NOT_READY | live provider and generated solve reached; dynamic verification blocked |
| 8.1 Provider Registry | READY | independent secret, SSRF, trust, probe, routing, persistence, and protocol acceptance |

## Real competition profile

- Competition: COMAP Mathematical Contest in Modeling
- Year: 2024
- Profile version: 1
- Status: `VERIFIED` for the historical technical benchmark
- Profile digest: `426690724d9889bc16b126b7618bc0382486314ca9848211da285939851f3466`
- Official provenance: 2024 MCM problem instructions and MCM/ICM Tips PDFs,
  hash-pinned in the profile and case manifests
- Scope: historical technical benchmark only; current rules and registered team
  control-number semantics require a fresh human check before real submission

## Historical preflight benchmark cases

Formal run `0fabda14-058c-45c4-8245-25bddf58b668` used the Phase 7 baseline
commit `30a7ae72cb28ad3845decb1b6c61ae95efa1cc95` plus an explicit dirty-source-tree
digest. All attempts and their P1 failures remain persisted.

| Case | Category | Official input result | Literature | Status | Score |
|---|---|---|---|---|---:|
| MCM 2024 A | ecological simulation / multi-stage | hash/size/PDF structure PASS | live Crossref PASS | BLOCKED_ENVIRONMENT | 10 |
| MCM 2024 B | uncertainty/search optimization | hash/size/PDF structure PASS | live Crossref PASS | BLOCKED_ENVIRONMENT | 10 |
| MCM 2024 C | data/prediction | PDF + two CSV hash/size/shape PASS | live Crossref PASS | BLOCKED_ENVIRONMENT | 10 |

Problem C's official Wimbledon file was inspected as 7,284 rows by 46 columns
with no missing cells, and its official data dictionary as 46 rows by 3 columns.
The score of 10 reflects only deterministic data/structural handling; it is not
a partial model-quality claim.

## Historical preflight benchmark results

- Real cases attempted: 3
- Cases passing: 0
- Verified real profiles: 1
- Live provider: `BLOCKED`
- Live literature: `PASS`
- Hidden P0: 0
- Human interventions: 0
- Provider calls/tokens/cost: 0 / 0 / USD 0
- Solver calls/experiments/repairs: 0 / 0 / 0
- Verified results/papers/PDFs/packages: 0 / 0 / 0 / 0
- Report digest:
  `2bb1d9426b916aede00a42c19fa0831a20ad87598bdb6e8bab424dd446b7e306`

Each case has one explicit `LIVE_PROVIDER` P1 failure: configured provider
`openai` was unavailable and no Mock fallback was used. Because no live agent
chain ran, problem understanding, model choice, mathematical correctness,
solver, validation, sensitivity, robustness, Red Team/repair, paper, citation
support, and submission quality are not evaluated for these cases.

## Live literature

The existing Crossref adapter performed real search and independent DOI
resolution for each formal case. A generic defect discovered during this run
was fixed: valid retrieved book/report records without `container-title` now use
the actual Crossref publisher, and publication year resolves across the
documented date fields. Fixture sources still cannot satisfy the live gate.
Human citation support and source-quality spot checks remain required once a
paper exists.

## Validation and regression

The complete local suite reports:

```text
409 passed
9 explicitly skipped environment-gated tests
coverage = 86.73% (required >= 85%)
```

The nine skips remain truthful: three custom-provider live gates lack credentials;
live literature requires explicit network opt-in; two paid live-provider paths
require explicit opt-in; live PaperAgent lacks credentials; PostgreSQL uses its
separate configured run; and optional Gurobi lacks a licensed runtime. The
dedicated PostgreSQL integration passed against migration head `20260830_0010`.
Phase 8 regressions cover evaluation leakage, resource/profile/result tamper,
fake live status, hidden retries/interventions, prompt injection as data, budget
gates, exception terminalization, source-tree identity, report redaction, and
API read/rebuild stability. Phase 8.1 additionally covers provider compatibility,
capability spoofing/staleness, SSRF, secret/redirect protection, stable routing
identity, native-adapter registry binding, total timeout, and custom-provider E2E.

## Operational requirements

- Register one approved non-Mock ProviderEndpoint/ModelProfile, reference its
  environment secret, pass CapabilityProbe and live Agent smoke, and never put
  the secret in manifests, logs, reports, papers, or Git.
- Pin provider/model configuration digests, trust, reasoning tier, and pricing
  in the run config.
- Keep PostgreSQL migration head `20260830_0010`, official-source/Crossref
  network access, and the Phase 3/4/6 Docker images available.
- Repeat all three cases and at least one primary case twice; retain every prior
  failed run and compare model family, conclusion, verification, and package
  status rather than prose identity.
- Add independent evaluator and human rubric/citation/rule review before asking
  for Phase 8 independent acceptance.

## Security boundaries

Official/network content is untrusted data. Retrieval is HTTPS host-allowlisted,
non-redirecting, byte-bounded, hash/size checked, atomically cached, and rejects
symlinks/reparse points. Evaluation-only solution/commentary resources cannot
enter blind solve bundles. Reports reject credential-like material and local
paths. Generated computation remains network-disabled, non-root, read-only, and
resource-bounded; benchmark code never executes evaluation documents.

## Competition-policy limitations

The 2024 COMAP profile allows AI only with disclosure and has historical
control-number/submission semantics. The current benchmark is technical; it
does not prove compliance with present-day rules or external portal acceptance.
If a profile requires code reproduction, the package remains fail-closed because
the general reproduction executor is not implemented.

## Known limitations

- A live-provider Phase 8 FEASIBLE solve exists, but no fully verified result,
  paper, PDF, or package exists.
- Case A now supplies the reviewed, versioned independent metric and scenario
  policy described above, but no fresh exact-model live attempt has passed its
  evidence chain. Cases B/C remain unaccepted. The system will not infer case
  metrics, scenario obligations, or acceptance thresholds.
- No independent benchmark LLM/human evaluator; human-rubric dimensions remain
  zero/Human Review rather than inferred from workflow success.
- No automatic benchmark crash/resume coordinator or asynchronous human-cancel
  endpoint. Provider/agent retries are recorded, and exceptions reach terminal
  attempts, but process-level resume is not claimed.
- Budget gates prevent acceptance and later work, but an already-issued provider
  request is not stream-cancelled when its post-attempt estimate crosses a cap.
- No scanned-PDF OCR, S3/MinIO backend, arbitrary model-family coverage, general
  package reproduction, or licensed Gurobi validation in this environment.
- Live model reproducibility and overfitting checks cannot run without a live
  model chain; blocked-environment repeatability is not a substitute.

## Final acceptance decision

The system has implemented and regression-tested the Phase 8 benchmark control
plane and has honestly attempted three official cases. It has not demonstrated
the defining Phase 8 live end-to-end outcome. The only defensible decision is:

```text
NOT_READY
```

Do not claim autonomous competition-ready operation until the live-provider,
independent-evaluation, two-successful-case, real-PDF, and real-package gates all
pass in retained formal runs.
