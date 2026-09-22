# Real DeepSeek Provider Smoke

## Goal

Configure the official DeepSeek API through the existing provider registry,
verify real discovery and generation behavior, probe capabilities, and route a
small Agent set without resuming the Phase 8 benchmark.

## Current State

The backend inherited `DEEPSEEK_API_KEY` and the official endpoint passes the
existing endpoint-security policy. The first real connection test exposed an
HTTP client transport defect: Windows system proxy settings were inherited by
HTTPX after DNS pinning, causing TLS hostname verification to use the pinned IP
instead of the official hostname.

## Files Affected

- `src/mathmodel_ai/providers/http.py`
- `src/mathmodel_ai/providers/compatible.py`
- `src/mathmodel_ai/providers/discovery.py`
- `src/mathmodel_ai/providers/presets.py`
- provider security/discovery tests
- current provider documentation and this plan

No Agent, solver, verification, paper, submission, benchmark, migration, or
credential-storage semantics are in scope.

## Design

- Keep DNS destination validation and address pinning unchanged.
- Prevent secure provider clients from implicitly using operating-system or
  environment proxy settings. Explicit proxy support would require its own
  validated endpoint contract and is not inferred from ambient state.
- Keep TLS verification, redirect blocking, official-domain validation, secret
  references, and Router hard filters unchanged.
- Use the current official stable text model returned by discovery/documentation
  and record only probed capability evidence.

## Implementation Steps

1. Add a regression proving default secure clients set `trust_env=False`.
2. Reuse that client construction in discovery and provider-generation paths.
3. Refresh DeepSeek preset hints to current official stable text model IDs.
4. Re-run connection, discovery, model creation, probe, default selection,
   route preview/persistence, and low-cost Agent smoke.
5. Scan logs, persisted metadata, API responses, and repository files for
   credential leakage.

## Tests

- Focused provider security, discovery, integrity, and routing regressions.
- Real DeepSeek connection, discovery, capability probe, route preview, and
  Agent smoke.
- Full backend and frontend regression/static/build checks.

## Risks

- A transport fix must not weaken DNS-rebinding, SSRF, TLS, or redirect controls.
- Probe failures must remain truthful and may make the model ineligible.
- The API key must never enter source, logs, database plaintext, or reports.

## Acceptance Criteria

- Official DeepSeek connection and authentication pass.
- One current stable text model has a persisted real probe.
- Default model and three Agent routes select the verified DeepSeek profile.
- Provenance contains provider/model/route/probe/timing/usage without plaintext
  credentials.
- Phase 8 benchmark remains stopped and no commit is created.

## Outcome

- The official `https://api.deepseek.com` endpoint passed DNS, TLS, SSRF,
  authentication, and model-discovery checks through the existing
  OpenAI-compatible adapter.
- Discovery returned the current three-model catalog; only
  `deepseek-v4-flash` was registered as `deepseek-main`.
- The current probe proves text, JSON-mode structured output, streaming,
  reasoning control, system role, and usage reporting. Native JSON Schema and
  native tools remain truthfully unsupported by the probe.
- `deepseek-main` is the default model. `problem_agent`, `model_explorer`, and
  `model_jury` each preview as eligible and have explicit routes with no model
  fallback.
- A real three-stage Agent smoke completed through `SELECT`, with every final
  Agent run using the official provider, `deepseek-v4-flash`, JSON mode, high
  reasoning, non-Mock responses, current capability evidence, and recorded
  timing/token usage.
- Provider JSON mode now sends the target schema, requires all required fields,
  and reports only safe validation metadata. The probe can recover from stale
  unsupported observations and uses a small low-reasoning structured budget.
- Full verification: backend `523 passed, 10 skipped`, backend coverage
  `86.87%`; frontend `39 passed`; Ruff, mypy strict, Alembic check, frontend
  typecheck/lint/build, and `git diff --check` passed.
- Credential leakage scans found no plaintext key in provider/model/AgentRun
  records, runtime files, process command line, browser-visible text, or browser
  console output. Repository key-shaped matches are synthetic test fixtures.
- Phase 8 benchmark was not resumed and no commit was created.
