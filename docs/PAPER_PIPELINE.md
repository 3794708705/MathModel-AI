# Evidence-grounded paper pipeline

## Truth boundaries

Phase 6 consumes only `ProblemState.verified_result_id` and its exact persisted
Phase 5 chain. `VerifiedEvidenceBuilder` independently checks model/result/run/
execution identities and terminal validation, sensitivity, robustness, and Red
Team reports before creating immutable evidence. It never selects a latest
model or result. Unverified, blocked, or Mock computational evidence cannot
support a final-ready claim.

```text
verified_result_id -> EvidenceSnapshot -> ClaimEvidenceGraph -> Paper IR
                  -> registries/gates -> LaTeX/BibTeX -> PDF -> manifest
```

Paper IR is the document-structure source of truth. Evidence records are the
factual source of truth. LaTeX and PDF are rendered artifacts only.

## Claims and citations

PaperAgent emits typed blocks and Claim records together. Required claim types
must have declared evidence links. Numeric and unit values resolve through a
structured evidence field; comparisons resolve both operands and recompute the
percentage and direction. Displayed claim text is checked against the structured
value with deterministic rounding and an allow-listed unit-conversion policy.
Cross-section number conflicts, solver/optimality misstatements, unsupported
assumptions, hallucinated numbers, and Mock evidence fail closed.

LiteratureAgent emits search needs only. `LiteratureSource` adapters supply
bibliographic metadata and trusted excerpts. Metadata verification compares
title, authors, year, venue, and DOI independently from claim-support review.
A real reference with irrelevant or insufficient text cannot support a claim.
Citation support records bind exact claim and reference digests, so edited
claims, metadata, or trusted source text invalidate stale decisions.

## Registries and artifacts

`DocumentRegistry` assigns and verifies `SEC`, `EQ`, `FIG`, `TAB`, `REF`, and
`CLAIM` identities. Formal equations are reused from the Phase 4
`EquationRegistry`; symbol tables are generated from mathematical registries.
Figure records bind canonical data, rendering code, and image artifacts. Table
records bind canonical columns/rows. Figure and table source bindings also
identify the evidence type, experiment, parameter, and data source needed for
their stated semantics. Recomputed hashes detect mutation or source swaps.

Required competition subproblems are represented explicitly and must be covered
by referenced sections, claims, equations, figures, or tables. The abstract has
typed purpose, method, result, validation, and conclusion roles, which are
checked deterministically rather than inferred from section titles.

Each paper version stores its complete Paper IR, evidence snapshot, gate report,
compile record, manifest, parent version, and revision reason. Database rows are
immutable version records; bytes are stored through `FileStore`.

## Rendering and compilation

`LaTeXRenderer` escapes prose and metadata, accepts equations only from the
registry, renders labels/numbering, and rejects include/write commands, unsafe
citations, and traversal tokens. `BibTeXRenderer` accepts only references whose
metadata status is `VERIFIED`.

`PDFCompiler` passes an argv list to Docker—never an interpolated shell command.
The paper image runs as UID 65532 with no network, read-only root, dropped
capabilities, no-new-privileges, CPU/RAM/PID/time limits, and XeLaTeX shell
escape disabled. Success requires a real non-empty PDF and a matching artifact
record. PDF validation parses every page, checks the expected page count, and
treats unresolved final-pass references/citations and other configured LaTeX
warnings as fatal. The final manifest hashes Paper IR, claims, evidence snapshot,
registries, references, figures, tables, TeX, BibTeX, and PDF, then validates the
stored bytes again before final promotion.

## Quality status

The gate combines claim/evidence, numeric consistency, factual integrity,
citation metadata/support, cross-reference, symbol, hallucinated-number,
equation/rendered-content, requirement coverage, artifact/hash, compile,
manifest, and independent factual-audit results. It recomputes unresolved Red
Team severity from findings instead of trusting persisted counts. A paper
reaches `READY_FOR_FINAL_JURY` only when all deterministic errors are absent and
every Critical/Major claim is covered by a passing non-Mock audit. Phase 6 never
emits `SUBMISSION_READY`.

## API

Project-scoped routes are:

```text
POST /api/v1/projects/{project_id}/literature/search
GET  /api/v1/projects/{project_id}/literature
POST /api/v1/projects/{project_id}/paper/run
POST /api/v1/projects/{project_id}/paper/build
GET  /api/v1/projects/{project_id}/paper
POST /api/v1/projects/{project_id}/paper/validate
POST /api/v1/projects/{project_id}/paper/render
GET  /api/v1/projects/{project_id}/paper/artifacts
```

`paper/run` and `paper/build` invoke the same one-click workflow. Validate and
render endpoints expose the persisted immutable reports/artifacts for the
selected paper version; they do not silently create a new version.
