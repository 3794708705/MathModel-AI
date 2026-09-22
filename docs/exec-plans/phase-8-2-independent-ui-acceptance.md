# Phase 8.2 Independent UI Acceptance Audit

## Goal

Attack the Web control center, encrypted credential bridge, provider/model
registry, Router preferences, project workspace, and benchmark presentation to
prove the browser is never business authority and cannot leak credentials.

## Current State

- Baseline remains `30a7ae72cb28ad3845decb1b6c61ae95efa1cc95`.
- Phase 8/8.1/8.2 changes remain intentionally uncommitted.
- Phase 8.2 self-test is ready; Phase 8 remains blocked by missing real provider
  configuration.
- Migration head is `20260831_0011`.

## Files Affected

- Frontend API client, credential/routing/provider/model/benchmark/workspace
  components, and their tests if an attack exposes a defect.
- Secret bridge, provider registry API, Router/repository, CORS, and tests only
  where a deterministic backend bypass or leakage exists.
- This audit plan and testing/security documentation.

## Design

1. Treat preview and UI disabled states as usability only; attack direct backend
   writes and execution-time route binding independently.
2. Inspect every browser persistence/error/cache surface for credential data.
3. Exercise controlled Provider transports for authenticated Probe/Router
   success without real credentials or a formal benchmark.
4. Inject stale/tampered/default/Mock/trust/capability/XSS/error states and
   require backend-owned status or fail-closed behavior.
5. Preserve immutable Phase 1–8.1 evidence and rerun historical adversarial,
   Docker, PDF, package, PostgreSQL, and migration gates.

## Implementation Steps

1. Build a requirement-to-test matrix from the 107-point audit.
2. Review semantic diffs and static browser/business-authority boundaries.
3. Add adversarial backend and frontend tests before changing behavior.
4. Apply minimal general fixes and document the security boundary.
5. Run real-build browser positive and negative flows against an isolated test
   database and controlled provider transport.
6. Run all final quality gates and classify every required attack.

## Tests

- Secret DOM/storage/URL/cookie/cache/React Query/error/log/readback/tamper/key
  lifecycle.
- Default model, direct Agent policy, stale preview, Mock, missing/tampered
  preference, capability, trust, and probe freshness.
- XSS/JavaScript URL, status authority, error mapping, mutation retry/double
  submit, polling cleanup, failed history visibility, and workflow delegation.
- Frontend full suite/build/lint/typecheck/security scan.
- Backend full coverage, Phase 4–8.1 adversarial subsets, PostgreSQL migration
  roundtrip/integration, Docker Solver/PDF/package, Ruff/mypy/Alembic/diff.

## Risks

- A browser fixture could accidentally be mistaken for live readiness: all fake
  transport evidence stays in an isolated test environment and is labelled.
- Credential test strings could contaminate artifacts: use synthetic markers,
  scan bundles/logs, and delete isolated runtime directories.
- Direct policy persistence may be intentionally preference-only: acceptance
  depends on execution-time hard-filter revalidation, not UI enforcement.
- Docker Hub availability may block optional image rebuilds; existing accepted
  images and local production builds remain separate evidence.

## Acceptance Criteria

- No secret remains in browser, logs, responses, persisted plaintext, query
  caches, or readable API surfaces after success or every tested failure.
- Defaults and Agent policies never bypass enabled/health/credential/trust/
  capability/fresh-probe/tier/Mock filters, including preview races.
- Backend is the sole source of READY/PASS/VERIFIED/FROZEN and benchmark score.
- Controlled Provider success and negative browser flows both pass.
- All mandatory regression and migration gates pass truthfully.
- No commit is created and Phase 8 benchmark is not resumed.

## Audit Progress

- Static contract/security review: complete.
- Backend and frontend adversarial regressions: complete.
- Controlled browser positive/negative flows: complete; the in-app browser
  intermittently lost local fetch completion, which exposed and led to fixes for
  ambiguous provider-create commits and credential-write refresh handling.
- Full backend/frontend regression and builds: complete.
- PostgreSQL migration roundtrip/integration and final repository checks:
  complete.

## Defects Found and Repaired

- Encrypted-row/master-key tamper could leave a stale public `READY`/configured
  view.
- Route PUT trusted earlier UI preview too much and did not independently apply
  every current Router hard filter.
- Browser error paths could display backend exception/credential-like detail.
- Credential mutation state could retain sensitive variables in a shared query
  cache, and post-write refresh failure could be mislabeled as write failure.
- An ambiguous provider POST response could invite duplicate creation.
- Router preview authority survived a rejected save race.
- Project/benchmark pages polled terminal records indefinitely.
- Unknown capabilities and Mock models lacked safe/explicit UI treatment.

## Final Verification

- Backend: 496 passed, 9 truthful environment-gated skips, 86.90% coverage.
- Frontend: 19 passed; 77.55% statements, 72.03% branches, 69.64%
  functions, and 82.46% lines.
- TypeScript, ESLint, production build, Ruff format/check, mypy strict,
  Alembic check, and `git diff --check`: pass.
- PostgreSQL: `0010 → 0011 → 0010 → 0011`, autogenerate check, and integration
  test pass on a disposable container; container and anonymous volume removed.
- Controlled browser: full positive route-save/reload flow, disabled-provider
  rejection, 1366×768 and 390×844 responsive smoke, and browser secret residue
  checks pass. No formal benchmark was started.
