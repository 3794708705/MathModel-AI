# Architecture

## Current assessment

The repository began empty. The implemented stack is Python 3.12+, FastAPI,
Pydantic v2, SQLAlchemy 2, Alembic, PostgreSQL, and HTTPX. Phase 1 established
the infrastructure; Phase 2 adds the reasoning core. Phase 3 adds guarded file
ingestion, immutable local object storage, deterministic data profiling,
structured data interpretation, artifact registries, and isolated Python
execution. Phase 4 adds a typed mathematical intermediate representation,
registries and dimensional checks, deterministic algorithm/solver selection,
versioned generated programs, real SciPy and OR-Tools solving, optional Gurobi,
and exact result evidence. Phase 5 adds independent result validation, real
perturb-and-resolve experiments, configurable robustness methods, structured
adversarial review, and a three-cycle immutable model-repair loop. Phase 6 adds
verified-evidence snapshots, claim/evidence graphs, retrieved literature and
two-stage citation checks, Paper IR, document/asset registries, deterministic
LaTeX/BibTeX rendering, and sandboxed real PDF compilation. SQLite is used for database-independent tests;
PostgreSQL remains the production contract and migration target.

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

SELECT -> MathModeler -> MathematicalModel -> MODEL gate -> AlgorithmSelector
                                             -> ExecutionStrategySelector
                                                |-> SolverRouter -> adapters
                                                \-> CodeAgent -> GeneratedProgram
                                                            |
                 ResultRecord <- SOLVE gate <- SolverResult <- SandboxExecutor
                       \-> content-valid Model/Run/Program/Execution evidence

SOLVE -> IndependentValidator -> VALIDATE gate -> Sensitivity experiments
      -> SENSITIVITY gate -> Robustness experiments -> ROBUSTNESS gate
      -> RedTeamAgent + deterministic attacks -> RED_TEAM gate
          | PASS
          ` Critical -> ModelRepairAgent -> MODEL_REPAIR gate -> model vN+1
                       -> SOLVE -> VALIDATE -> SENSITIVITY -> ROBUSTNESS
                       -> RED_TEAM (maximum three automatic repair cycles)

verified_result_id -> EvidenceBuilder -> Claim/Evidence graph -> PaperAgent
                  -> Paper IR + registries -> deterministic factual gates
                  -> LaTeX/BibTeX -> restricted PDFCompiler -> manifest
                  -> READY_FOR_FINAL_JURY (never SUBMISSION_READY)
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
- `mathematical`: expression/unit/registry logic, deterministic algorithm
  selection, MODEL/SOLVE gates, versioned persistence, and orchestration.
- `solvers`: capability/health adapters, fallback routing, deterministic program
  translation, canonical status mapping, and minimum feasibility recomputation.
- `verification`: an independent expression evaluator, result/constraint/metric
  validation, executed sensitivity and robustness experiments, deterministic
  plus model-assisted Red Team analysis, quality gates, repair orchestration,
  and Phase 5 persistence.
- `paper`: exact verified-chain evidence resolution, claims, literature and
  citation verification, Paper IR registries, assets, deterministic validators,
  renderers, restricted compilation, manifest generation, orchestration, and
  Phase 6 persistence.
- `db`: persistence mapping and sessions; no reasoning logic.

External model calls happen outside database transactions. A successful stage
then uses a short transaction to retire the current state, insert one immutable
revision, record its `AgentRun`, and optionally record model-decision evidence.
The partial unique index on current state remains the final concurrency guard.

File bytes and generated artifacts live in the file-store abstraction, not
database JSON blobs. PostgreSQL stores identities, hashes, storage keys,
profiles, state, evidence links, versions, and execution metadata. Phase 4 adds
coarse `mathematical_models`, `generated_programs`, `solver_runs`, and `results`
registries. Phase 5 adds `validation_runs`, `sensitivity_runs`,
`robustness_runs`, `verification_experiments`, `red_team_reports`, and
`repair_cycles`. Symbols, parameters, constraints, and equations remain inside the
versioned model JSONB because Phase 4 has no independent query requirement for
them. Exact relational foreign keys bind every result to one model record, one
solver run, and one execution record. Phase 3 uses
the local immutable implementation; its generated-key contract leaves room for
an S3-compatible implementation without changing domain schemas.
Phase 6 adds `evidence_records`, `claims`, `claim_evidence_links`,
`literature_searches`, `literature`, `citation_metadata_checks`,
`citation_support_checks`, `paper_versions`, `document_registry`, `figures`,
`tables`, and `paper_artifacts`. Paper section structure stays in immutable
Paper IR JSON; artifact bytes remain in the file store.

## Technical debt

Intentional debt is bounded: the local file store has no S3/MinIO adapter or
orphan-artifact garbage collector; cross-file relationship recomputation is not
implemented (cross-sheet discovery is); scanned-PDF OCR is represented through
the multimodal interface rather than a local OCR engine. Phase 4 supports
flattened scalar deterministic adapters, not indexed expansion, general CAS,
nonlinear global optimization, or every reserved model family. Gurobi requires a
separate licensed runtime. Bootstrap remains blocked without row-level
data-to-parameter resampling. Crossref is the first live literature adapter;
additional scholarly sources and orphaned file-artifact cleanup remain future
work. Phase 6 does not implement Final Jury or submission packaging.
