# Phase 3 — Data and execution foundation plan

Status: completed on 2026-08-26. Acceptance used real parser/statistics tests,
PostgreSQL migration plus `alembic check`, and a real Docker isolation smoke test.
The local sandbox image used the documented offline Python 3.13 build override
because Docker Hub access to the default Python 3.12 image was unavailable.

## Goal

Implement a secure, traceable foundation for competition attachments, deterministic
data profiling, structured data understanding, and isolated Python execution:

```text
upload -> validate -> immutable store -> parse -> profile -> DataAgent
                                                -> sandbox execution record
```

Phase 3 stops before mathematical-model construction, solver selection, generated
model code, sensitivity analysis, literature, and paper generation.

## Current state

Commit `b575275` is the clean Phase 2 baseline. It provides FastAPI, Pydantic,
SQLAlchemy/Alembic/PostgreSQL, versioned `ProblemState`, model providers/router,
agent traces, structured reasoning agents, and deterministic quality gates. There
is no file store, attachment parser, dataset registry, data agent, or code sandbox.

## Files affected

- `src/mathmodel_ai/files/`: immutable local storage, MIME/signature validation,
  upload limits, parser registry, CSV/XLSX/PDF/image/text pipelines.
- `src/mathmodel_ai/data/`: schema inference, deterministic statistics, missing
  values, duplicates, outliers, cross-sheet relations, feature candidates, and
  quality reports.
- `src/mathmodel_ai/sandbox/`: Docker command construction, bounded execution,
  artifact collection, and execution records.
- `src/mathmodel_ai/agents/`, `providers/`, and `routing/`: `DataAgent`, media
  request contract, Gemini inline-data path, and multimodal/long-context route.
- `schemas/`, `db/`, `api/`, `main.py`, migration `0003`, tests, and docs.

## Design

- Raw uploads are immutable. User filenames never become storage paths; UUIDs
  identify files and SHA-256 proves content identity.
- Validation combines an extension allowlist, declared MIME, byte signatures,
  parser verification, size limits, and OOXML archive bomb/traversal checks.
- Python computes all statistics. The DataAgent receives typed profiles and may
  interpret semantic roles, units, relationships, features, and risks, but cannot
  invent profile values.
- Parsed/profile artifacts are referenced in PostgreSQL; large bytes remain in a
  file-store abstraction. Local storage is the Phase 3 implementation and keeps
  an S3-compatible boundary possible.
- The data sub-workflow records `FILES`, `DATA`, and `EXECUTION` gates separately
  from the Phase 2 reasoning stage, avoiding an invalid rewind from `SELECT`.
- Generated Python runs only in Docker with `network=none`, non-root user, read-only
  root filesystem, bounded CPU/RAM/PIDs/time, dropped capabilities, and explicit
  read-only input plus writable output mounts. No shell command interpolation is
  used.
- External model calls and sandbox execution happen outside database transactions;
  resulting metadata/state revisions use short transactions.

## Implementation steps

1. Add dependencies, settings, Phase 3 schemas, and file/data/execution contracts.
2. Implement local file store, security validator, parser registry, and artifact
   output for CSV, XLSX, PDF, images, and text.
3. Implement deterministic data profiler and cross-sheet relationship discovery.
4. Add database tables/migration/repositories and ProblemState v3 data fields.
5. Add media request support, Gemini multimodal routing, versioned prompt, and
   structured `DataAgent`.
6. Implement Docker sandbox, execution repository, quality gates, and APIs.
7. Add unit, security, integration, PostgreSQL, real Docker sandbox, and Phase 3
   E2E tests; update CI and documentation.

## Tests

- Safe and malicious filenames, unsupported types, MIME mismatch, oversized input,
  corrupt files, OOXML traversal/bomb limits, and immutable raw-byte preservation.
- CSV schema/missing/outlier/duplicate profiling; multi-sheet XLSX relationships;
  PDF text/page extraction; image metadata and multimodal media construction.
- DataAgent structured Mock run, Gemini inline-data payload, and routing preference.
- Database reload of files, datasets, profiles, artifacts, execution records, and
  ProblemState v3 fields.
- Sandbox success, non-root identity, timeout, network denial, read-only filesystem,
  stdout/stderr capture, artifact limits, and code hash.
- `FILES -> DATA -> EXECUTION` gate order and an API E2E using real parsers plus
  explicit Mock/DataAgent and real local Docker sandbox where available.

## Risks

- Office/PDF/image parsers process adversarial bytes: pre-parse limits and archive
  inspection are mandatory, and parsing errors remain explicit failures.
- Docker may be unavailable: the executor reports an unavailable status and never
  claims execution. CI and local acceptance build a pinned sandbox image.
- Multimodal request size can expand through base64: inline media is capped below
  the provider request limit; larger assets require a future provider file API.
- Heuristic feature/relationship discovery can be wrong: outputs are candidates
  with evidence and confidence, never promoted to facts.

## Acceptance criteria

- Original files are byte-identical, hashed, immutable, validated, and registered.
- CSV/XLSX/PDF/image/text fixtures produce typed parse results and tracked artifacts.
- Deterministic profiles include schema, missingness, duplicates, outliers,
  statistics, relationships, feature/target candidates, and quality issues.
- DataAgent uses `structured_generate()`, versioned prompt, ModelRouter, and Gemini
  path for multimodal/large-context tasks; Mock remains explicit.
- Sandbox records truthful success/failure/timeout, limits, stdout/stderr, code hash,
  environment, and collected artifacts under the required isolation flags.
- PostgreSQL migration/reload, Phase 1/2 regression tests, Phase 3 tests, Ruff,
  strict mypy, coverage, Alembic check, and real sandbox smoke tests pass.
