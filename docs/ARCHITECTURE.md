# Architecture

## Current assessment

The repository began empty. The implemented stack is Python 3.12+, FastAPI,
Pydantic v2, SQLAlchemy 2, Alembic, PostgreSQL, and HTTPX. Phase 1 established
the infrastructure; Phase 2 adds the reasoning core. Phase 3 adds guarded file
ingestion, immutable local object storage, deterministic data profiling,
structured data interpretation, artifact registries, and isolated Python
execution. SQLite is used for database-independent tests; PostgreSQL remains the
production contract and migration target.

## Boundaries

```text
API -> reasoning workflow -> agent -> model router -> provider adapter
                      |          \-> versioned structured prompt
                      \-> deterministic quality/scoring -> ProblemState revision
                                                   \-> PostgreSQL audit evidence

upload API -> body limit -> validator -> immutable file store -> parser
                                                    |          \-> artifacts
                                                    \-> profiler -> DataAgent
                                                                  \-> state v3

execution API -> SandboxExecutor -> Docker (no network, non-root, bounded)
                              \-> ExecutionRecord + output artifacts -> state v3
```

- `api`: transport concerns and health reporting only.
- `schemas`: durable domain and orchestration contracts.
- `routing`: deterministic task classification and model-level selection.
- `providers`: the only code allowed to know vendor HTTP shapes.
- `agents`: shared execution, validation, retry, and audit behavior.
- `reasoning`: state transitions, prompt registry, deduplication, gates,
  deterministic scoring, orchestration, and repository boundary.
- `files`: generated storage keys, immutable bytes, upload validation, and parser
  adapters for CSV, XLSX, PDF, images, and text.
- `data`: deterministic profiles, registries, quality gates, DataAgent workflow,
  and persistence. LLM interpretation cannot replace computed statistics.
- `sandbox`: Docker invocation, isolation limits, output capture, artifact
  collection, and truthful execution records.
- `db`: persistence mapping and sessions; no reasoning logic.

External model calls happen outside database transactions. A successful stage
then uses a short transaction to retire the current state, insert one immutable
revision, record its `AgentRun`, and optionally record model-decision evidence.
The partial unique index on current state remains the final concurrency guard.

File bytes and generated artifacts live in the file-store abstraction, not
database JSON blobs. PostgreSQL stores identities, hashes, storage keys,
profiles, state, evidence links, versions, and execution metadata. Phase 3 uses
the local immutable implementation; its generated-key contract leaves room for
an S3-compatible implementation without changing domain schemas.

## Technical debt

Intentional debt is bounded: the local file store has no S3/MinIO adapter or
orphan-artifact garbage collector; cross-file relationship recomputation is not
implemented (cross-sheet discovery is); scanned-PDF OCR is represented through
the multimodal interface rather than a local OCR engine. The Phase 3 sandbox is
a minimal Python image; numerical/solver packages and model-derived code belong
to Phase 4. There is still no literature, verification, or paper pipeline.
