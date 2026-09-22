# Provider Frontend UX Correction

## Goal

Make provider/model setup and per-Agent routing understandable from the Web UI
without changing registry, probing, routing, Agent, solver, verification, paper,
submission, or benchmark semantics. DeepSeek Harness is a visual-flow reference
only and is not a runtime dependency.

## Current State

The backend already supports durable providers, multiple models per provider,
write-only credentials, independent connection tests and capability probes,
default-model preference, and persisted Agent route policies. The existing UI
exposes these capabilities but presents too many infrastructure fields in the
ordinary flow and does not foreground provider/model inventory clearly enough.

## Files Affected

- `frontend/src/pages/ModelsApiPage.tsx`
- `frontend/src/pages/ModelsApiPage.test.tsx`
- `frontend/src/pages/RoutingPage.tsx`
- `frontend/src/pages/RoutingPage.test.tsx`
- shared frontend styling or contracts only if the existing primitives require it
- `docs/WEB_UI.md`
- this execution plan

No backend registry, adapter, probe, Router, Agent, solver, verification, paper,
submission, benchmark, or migration semantics are in scope.

## Design

- Put Default Model and the provider inventory first.
- Make the catalog path API-key-first; place protocol, endpoint, timeout, and
  trust details behind explicit Advanced controls.
- Keep API-key inputs empty and write-only, with replacement language on edit.
- Treat providers and models as separate resources and make discovery a model
  import convenience, never capability evidence.
- Show connection health separately from model probe evidence.
- Use the backend Agent catalog as the only route-stage source. Show one primary
  model in the ordinary flow and ordered fallbacks in Advanced.
- Keep local/private endpoints a Custom Provider advanced concern and retain the
  server default `MM_ALLOW_LOCAL_MODEL_ENDPOINTS=false`.
- Add Zhipu GLM and Moonshot/Kimi as custom-provider entry points with no guessed
  endpoint.

## Implementation Steps

1. Refactor Models & API page information hierarchy and copy.
2. Simplify catalog/custom setup while preserving existing API contracts.
3. Refine model status, discovery, probe, and default-model explanations.
4. Refactor Agent Routing cards into primary and advanced fallback controls.
5. Add focused component regressions and update Web UI documentation.
6. Run frontend tests/build, backend regression/static checks, source scan, and
   browser E2E against isolated public-style provider fixtures.

## Tests

- Component tests for catalog coverage, write-only credential UX, Advanced-only
  endpoint settings, multi-model inventory, discovery/probe separation, default
  semantics, and routing persistence.
- Browser flow with Provider A/A1/A2 and Provider B/B1/B2, four Agent routes,
  refresh, and persisted assignment verification.
- Repository scan proving no DeepSeek Harness runtime/dependency/proxy/session
  integration or Harness localhost URL.
- Existing Python and frontend regression suites and static checks.

## Risks

- Existing ambiguous-write reconciliation and secret redaction must remain intact.
- UI readiness labels must not imply capability proof or route eligibility.
- Frontend labels must not invent Agents absent from the backend catalog.
- Public-style browser fixtures must remain isolated and must not alter user data.

## Acceptance Criteria

- Multiple providers, model counts, credential state, and Default Model are
  immediately visible.
- Catalog setup is API-key-first; custom endpoints remain explicit and advanced.
- One provider can visibly manage multiple model profiles.
- Test Connection and Probe remain separate operations with accurate copy.
- Routing uses the live backend Agent catalog, supports primary plus ordered
  fallbacks, previews eligibility, and persists across refresh.
- `DeepSeek Harness runtime dependency = NONE`.
- `MM_ALLOW_LOCAL_MODEL_ENDPOINTS remains false by default`.
- No backend core semantic changes and no commit.

## Outcome

- Browser acceptance passed with isolated public-style Provider A/A1/A2 and
  Provider B/B1/B2 fixtures. ProblemAgent, MathModeler, PaperAgent, and
  FinalJury routes remained assigned to A1, A2, B1, and B2 after refresh.
- Frontend regression passed: 39 tests, TypeScript, ESLint, and production build.
- Python regression passed: 518 tests, 10 environment-gated skips, 86.86%
  coverage, Ruff, and strict mypy.
- Runtime-source scan found no DeepSeek Harness dependency or Harness localhost
  integration. Local/private model endpoints remain disabled by default.
