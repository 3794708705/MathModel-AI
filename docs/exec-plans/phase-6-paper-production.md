# Phase 6 evidence-grounded paper production

## Goal

Build a fail-closed paper pipeline that consumes only the explicit Phase 5
`verified_result_id` chain and produces traceable Paper IR, references, figures,
tables, LaTeX, BibTeX, a real PDF, and deterministic readiness evidence. Phase 7
submission behavior and a Phase 6 baseline commit are out of scope.

## Current State

Phase 5 baseline `327593e237ffdb490c210cc2471ab5bdf1e4de8c` is clean and its
baseline suite passes with 175 tests, four explicit environment skips, and
87.47% coverage. `ProblemState` v5 exposes the sole accepted formal result as
`verified_result_id`; models, results, executions, validation, sensitivity,
robustness, Red Team, and repair history already have immutable registries.
There is no literature, claim, Paper IR, document registry, or PDF compiler.

## Evidence Architecture

Phase 6 will not copy computational truth. `EvidenceBuilder` resolves the exact
verified result and its matching model/report chain, then records immutable
`EvidenceRecord` snapshots with source IDs, versions, hashes, verification
status, Mock status, and structured source fields. A paper evidence snapshot
pins model version/digest, result, validation, sensitivity, robustness, Red Team,
repair set, and citation set.

## Claim Model

Claims and many-to-many evidence links are first-class schemas and rows.
Numeric and comparison values are calculated or checked from structured source
fields, never prose. Required-evidence claim types fail closed. Deterministic
validators cover numeric/unit values, cross-section consistency, model family,
solver, optimality, sensitivity, robustness, accepted assumptions, Mock evidence,
and unsupported claims.

## Literature Strategy

`LiteratureAgent` emits typed search needs, not bibliographic metadata.
`LiteratureSource` adapters own retrieved metadata; the first implementation
supports deterministic/manual fixtures and an HTTP-independent adapter contract,
with gated live tests. `LiteratureStore` persists source metadata and retrieval
provenance without treating an LLM response as a source.

## Citation Verification

Metadata verification and claim-support verification remain separate. Metadata
is compared against retrieved records and yields VERIFIED/PARTIAL/NOT_FOUND/
CONFLICT. Support requires trusted abstract/excerpt text and yields SUPPORTED,
PARTIALLY_SUPPORTED, NOT_SUPPORTED, INSUFFICIENT_TEXT, or CONTRADICTED. Critical
literature claims require both verified metadata and supported content.

## Paper IR

`PaperAgent` receives a verified evidence packet and registries and returns typed
Paper IR plus claims; it cannot emit TeX, bibliography metadata, equations, or
figure/table data. Sections contain typed blocks and registry references. The
abstract is validated under the same claim rules as every other section.

## Registry Design

One `DocumentRegistry` owns deterministic SEC/EQ/FIG/TAB/REF/CLAIM IDs and
duplicate detection. Equation objects are imported from the Phase 4 model.
Symbol tables are built from model variables, parameters, constants, and sets.
Figure and table records bind source evidence, canonical data bytes/hash,
generation code/hash where applicable, rendered artifact/hash, and status.
Citation records wrap only verified retrieved references.

## Renderer Design

`LaTeXRenderer` and `BibTeXRenderer` are deterministic and escape all prose and
metadata. IDs become labels and renderer-controlled numbers. Raw LaTeX commands,
`\input`, `\include`, `\write18`, file URLs, and unsafe artifact paths are
rejected. Renderers never reconstruct business state from TeX.

## PDF Pipeline

`PDFCompiler` runs XeLaTeX/BibTeX in a dedicated Docker image as non-root with no
network, read-only root, bounded CPU/RAM/PIDs/time, and no shell escape. It uses
an isolated per-run directory, captures stdout/stderr/exit code and hashes, and
publishes artifacts only after validating a non-empty PDF. The compiler is an
adapter and never invokes an interpolated shell command.

## Quality Gates

Paper readiness combines claim/evidence, citation, symbol, equation, figure,
table, cross-reference, hallucinated-number, LaTeX safety/compile, artifact, and
manifest-integrity gates. Deterministic failures dominate LLM review. Mock or
unverified computational evidence produces HUMAN_REVIEW/FAILED, never
READY_FOR_FINAL_JURY.

## Persistence

Add one Phase 6 migration without modifying history. Persist evidence, claims,
claim links, references/support checks, paper versions, document objects,
figures, tables, and paper artifacts. Paper sections remain in versioned Paper IR
JSON to avoid premature normalization. Transactions atomically create a paper
version and its immutable registry snapshot.

## Tests

- Schema and repository tests for identity, versioning, links, and snapshots.
- Acceptance cases A-T for numeric/unit/solver/optimality/citation/reference/
  symbol/cross-section/cross-reference/figure/table/Mock/assumption tampering.
- Real verified-result selection with multiple competing result records.
- Real Docker Paper IR -> LaTeX -> PDF E2E with non-Mock Phase 5 evidence.
- PostgreSQL upgrade/downgrade/re-upgrade and foreign-key checks.
- Gated live literature and PaperAgent tests that never fall back to Mock PASS.
- Full pytest coverage, Ruff, strict mypy, Alembic check, and diff check.

## Risks

- A citation abstract may be insufficient to establish a nuanced claim; the
  verifier must return INSUFFICIENT_TEXT rather than infer support.
- Deterministic number scanning cannot understand every prose context; only a
  conservative whitelist is accepted and unresolved numbers block or warn.
- TeX images and fonts enlarge the sandbox image; Phase 6 starts with a portable
  article template and PDF-safe generated assets.
- Independent paper review can share model interpretation errors with upstream
  agents; deterministic provenance cannot prove that the chosen model is ideal.

## Acceptance Criteria

- Only the exact `verified_result_id` and matching model/report chain are used.
- Required claims have verified direct support; all numeric/unit comparisons
  match structured evidence.
- Metadata existence and claim support are independently verified.
- EQ/FIG/TAB/REF/CLAIM references are complete and renderer-numbered.
- Figure/table source hashes detect mutation.
- Unsafe LaTeX and Mock/unverified evidence cannot become final-ready.
- At least one real sandboxed PDF compiles and has a traceable manifest.
- Migration, PostgreSQL, A-T, E2E, coverage >=85%, Ruff, mypy, Alembic, and diff
  checks pass.
- Final self-assessment is only SELF_TEST_READY or NOT_READY; no commit and no
  Phase 7 implementation are created.
