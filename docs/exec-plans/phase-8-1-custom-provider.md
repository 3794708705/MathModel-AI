# Phase 8.1 — configurable provider architecture

## Goal

Replace brand-slotted routing with a secure, extensible provider/model registry:

```text
Agent -> TaskProfile -> capability requirements -> ModelRouter
      -> ModelProfile -> ProviderEndpoint -> ProtocolAdapter -> remote API
```

Keep the native OpenAI, Google, Anthropic, and Mock adapters working while
adding OpenAI-compatible and constrained custom-JSON endpoints. No Mock route
may satisfy a live-provider or benchmark gate.

## Current state

- Parent baseline: `30a7ae72cb28ad3845decb1b6c61ae95efa1cc95`
- Phase 1–7: accepted.
- Phase 8 infrastructure: present in the uncommitted working tree.
- Phase 8 benchmark: `NOT_READY`; live provider is blocked by missing credentials.
- Preserved history: 3 `BenchmarkRun` records and 9 official attempts.
- Existing runtime provider registry is keyed by the fixed `ProviderName` enum.
- Existing router maps logical tiers to brand/model pairs in `Settings.model_catalog`.
- Existing provider credentials use `SecretStr`, but arbitrary credential references,
  endpoint safety policy, capability probes, and persisted model routes do not exist.

### Baseline repository state

`git status --short` contains only the existing Phase 8 modifications and new
Phase 8 files. No reset, stash, clean, or checkout was performed.

Tracked diff before Phase 8.1:

```text
15 files changed, 516 insertions(+), 38 deletions(-)
```

Untracked Phase 8 paths include `benchmarks/`, the Phase 8 execution plan and
acceptance document, migration `20260830_0009`, benchmark API/package/schema,
and benchmark tests. They must remain intact.

Baseline test on 2026-08-30:

```text
353 passed, 6 skipped, 1 warning
coverage 87.15% (required floor 85%)
```

The skips were explicitly environment-gated: live literature, three live
provider paths, PostgreSQL, and licensed Gurobi. None is recorded as PASS.

## Files affected

- `src/mathmodel_ai/providers/`: endpoint/model contracts, secret resolver,
  security policy, protocol adapters, probe, registries, and presets.
- `src/mathmodel_ai/routing/`: capability requirements and registry-backed routing.
- `src/mathmodel_ai/db/models.py` and new migration `20260830_0010`.
- `src/mathmodel_ai/agents/base.py` and reasoning persistence for exact route trace.
- `src/mathmodel_ai/api/routes/` and API schemas for provider/model/route operations.
- `src/mathmodel_ai/core/config.py`, `src/mathmodel_ai/main.py`, and exports.
- Provider, routing, API, migration, security, E2E, and regression tests.
- `docs/CUSTOM_PROVIDERS.md`, `docs/MODEL_REGISTRY.md`, routing/architecture/testing,
  README, security documentation, and this plan.

## Design

1. Persist `ProviderEndpoint` separately from `ModelProfile`; both use stable
   string IDs and canonical configuration digests that exclude secret values,
   timestamps, and database IDs.
2. Store only `credential_ref` (`env:NAME`). `EnvironmentSecretResolver` is the
   sole environment access boundary. API responses expose only
   `credential_configured`.
3. Validate endpoint URLs at registration and immediately before requests.
   Public endpoints require HTTPS; private/local destinations require the
   server-level `MM_ALLOW_LOCAL_MODEL_ENDPOINTS=true`. DNS results are checked,
   redirects are disabled, TLS verification is server-controlled, timeouts are
   finite, response bodies are bounded, and reserved headers are rejected.
4. Select one protocol adapter from an enum. OpenAI-compatible vendors share
   one implementation. Custom JSON mapping is declarative and constrained to
   known placeholders and JSON field paths; no Python or executable template is
   accepted.
5. Persist observed probe results bound to both endpoint and model digests.
   Effective critical capabilities prefer `PROBED`/`BUILTIN_VERIFIED` evidence;
   user declarations alone cannot become verified support.
6. Route with hard filters before scoring: enabled/configured/healthy endpoint,
   enabled model, current capabilities, context, protocol, and trust policy.
   Agent route policy and explicit model preference are auditable; fallback is
   between eligible real models only for live operation.
7. Keep legacy `MM_DEFAULT_PROVIDER` and `MM_DEFAULT_PROVIDER_MODEL` as a
   compatibility alias. `MM_DEFAULT_MODEL_ID` takes precedence when set.
8. Add nullable exact provider/model/digest/protocol/trust fields to historical
   AgentRun and BenchmarkRun records so old rows remain readable.

## Implementation steps

1. Add typed registry schemas, canonical digest helpers, secret resolver, endpoint
   safety policy, and database rows/repository.
2. Add migration 0010 without editing migration 0009.
3. Add secure HTTP protocol adapters, parameter mapper, structured-output
   strategies, custom JSON mapping, presets, and provider factory integration.
4. Add low-cost capability probing, normalized health/errors/usage, and stale
   probe detection.
5. Add registry-backed capability routing, agent-specific policies, explicit
   preference, real-only fallback, and legacy mode.
6. Add provider/model/routing APIs and exact AgentRun/Benchmark trace fields.
7. Add adversarial A–Z tests, fake-transport E2E, live gated tests, PostgreSQL
   migration roundtrip, documentation, and complete regression verification.

## Tests

- Unit: schemas/digests, resolver/redaction, URL/DNS/header/redirect/TLS/size/time
  controls, adapters, parameter mapping, structured fallback, probe matrix/staleness.
- Routing: disabled/unconfigured/unhealthy, capabilities, context, preferred model,
  agent policies, real fallback, Mock exclusion, legacy/new-config priority.
- API/integration: CRUD without secret exposure, probes, preview without calls,
  immutable references, exact trace persistence, custom-provider fake E2E.
- PostgreSQL: `0009 -> 0010 -> 0009 -> 0010`, Alembic check, schema/FKs.
- Full: pytest with coverage, Ruff format/check, mypy strict, Alembic check,
  `git diff --check`, and credential-pattern scan.

## Risks

- DNS rebinding cannot be eliminated solely in application code; each connection
  is therefore preceded by resolution validation and redirects stay disabled.
- Provider-compatible APIs vary. Unknown parameters are omitted and capabilities
  remain UNKNOWN/UNSUPPORTED until observed.
- Existing Agents expect a remote model string. Route decisions retain that field
  while adding stable model/provider IDs and digests.
- Persisted custom configuration may be unavailable before migration; APIs fail
  closed rather than silently selecting Mock.

## Acceptance criteria

- All 37 Phase 8.1 Definition-of-Done items are implemented and tested.
- A–Z adversarial/provider tests pass without external network access.
- Existing Phase 1–8 tests remain green and coverage is at least 85%.
- PostgreSQL migration roundtrip and integration pass.
- No secret value appears in DB models, API responses, logs, AgentRun, reports,
  documentation, or Git diff.
- Existing 3 benchmark runs and 9 attempts remain unchanged.
- Final status is only `SELF_TEST_READY` or `NOT_READY`; no commit is created.
