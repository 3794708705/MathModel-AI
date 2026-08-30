# Phase 7 — final jury and submission execution plan

## Goal

Produce a deterministic, version-bound path from one accepted Phase 6 paper and
its exact verified result to a competition-rule decision, final jury report,
submission check, immutable freeze, manifest, and independently re-opened package.
Phase 7 stops at `FROZEN`/`READY_TO_SUBMIT`; it never claims external submission.

## Current State

Phase 6 baseline `12506d85c9cfd74a836320921bd395418f3a3826` is clean and its
Phase 1–6 regression passes. The current state pins `verified_result_id`; every
`READY_FOR_FINAL_JURY` paper pins its evidence snapshot and manifest. Phase 7
must consume an explicitly approved paper identity/version and may not substitute
the latest model, result, paper, or artifact.

## CompetitionProfile Design

Add versioned typed profiles with structured page, naming, anonymity, reference,
file, code, data, AI-disclosure, deadline, and custom rules. Every material rule
stores provenance and verification status. `TEST_FIXTURE` profiles are valid only
for deterministic tests; unknown/conflicting/unverified blocking rules require
human review.

## Rule Engine

Evaluate typed rules against a candidate made from the exact PaperVersion and an
explicit artifact registry. Re-read PDF page count, bytes, MIME signatures,
filenames, sizes, package policy, anonymity/security content, and deadline.
Return bottom-level `RuleResult` records; final decisions recompute counts and do
not trust persisted summaries.

## Final Jury Design

`FinalJuryAgent` receives only structured state, evidence, paper, rule results,
and artifacts. A deterministic jury gate recomputes the weighted scorecard,
findings by severity, requirement coverage, Phase 5/6 prerequisites, and blocking
rules. High scores cannot offset Critical findings. Mock jury output remains
explicit and cannot be represented as a live-provider pass.

## SubmissionCheck

Combine independently recomputed rule, requirement, paper, result, citation,
asset, code/data, security, deadline, and manifest checks. `READY` requires no
blocking failure, missing requirement, unresolved Critical finding, unknown hard
rule, or stale Phase 5/6 identity.

## Correction Workflow

Create typed correction plans and dependency-based invalidation. Format-only
changes restart Phase 6/7 checks; claim/equation/model/result/data changes map
back to their required upstream stages. Phase 7 never edits a verified model,
result, equation, data record, or critical claim.

## Submission Freeze

Freeze an explicit approved paper version and exact verified model/result into a
`SubmissionSnapshot`. Hash every protected file and bind profile identity/version,
paper manifest, artifact IDs, and package hash. Any byte, file-set, identity, or
manifest change yields `DIRTY` and `RECHECK_REQUIRED`.

## Manifest

Build canonical JSON from sorted relative paths, SHA-256 values, sizes, MIME
types, project/submission/profile/paper/model/result identities, and the Paper
manifest hash. The package hash excludes archive timestamps and the manifest
contains its own deterministic digest contract.

## Package Builder

Use an explicit allow-list; never archive the repository root. Reject absolute
paths, traversal, symlinks, hidden/cache/private files, forbidden extensions,
secrets, and local machine paths. Build a deterministic ZIP and verify it by
closing source context, reopening entries, and recomputing every manifest field.

## Security

Fail closed on ZIP slip, unsafe names, symlinks, wrong MIME/signature, hidden
files, local paths, credentials/tokens/passwords/license material, corrupt PDFs,
and artifact/project/version swaps. Security checks remain mandatory in every
deadline mode.

## Tests

- Unit/schema tests for profiles, rule provenance/evaluation, requirements,
  jury scoring/gates, correction/invalidation, deadline modes, and persistence.
- A–Z adversarial tests for rule/coverage/jury/freeze/package bypass attempts.
- Real filesystem + real PDF + manifest + deterministic ZIP roundtrip E2E using
  the accepted Phase 5 result and Phase 6 paper path.
- PostgreSQL migration upgrade/downgrade/re-upgrade and API integration.
- Full Phase 1–7 pytest/coverage, Ruff, strict mypy, Alembic, and diff checks.

## Acceptance Criteria

All Phase 7 definition-of-done items pass; overall coverage remains at least
85%; focused components meet their stated targets where measurable; package
roundtrip verifies independently; status is `FROZEN`, never `SUBMITTED`; live
provider/literature/Gurobi skips remain honest; no Phase 8 implementation or
Phase 7 baseline commit is created.
