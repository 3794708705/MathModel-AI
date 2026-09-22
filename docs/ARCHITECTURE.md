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
LaTeX/BibTeX rendering, and sandboxed real PDF compilation. Phase 7 adds
versioned competition profiles, deterministic Final Jury/submission checks, and
hash-frozen packages. Phase 8 adds a separate benchmark audit plane with blind
official inputs, immutable attempts, atomic metrics/failures/interventions,
deterministic score recomputation, and report generation. SQLite is used for database-independent tests;
PostgreSQL remains the production contract and migration target.

Phase 8.1 replaces brand-bound routing with a persisted provider/model registry.
An Agent selects a stable `ModelProfile`; runtime then binds its
`ProviderEndpoint` to a native, OpenAI-compatible, or constrained custom-JSON
adapter. Endpoint health, capability evidence, trust, and model identity remain
separate claims. A route binds the exact task, provider/model configurations, and
atomic capability probe; an execution guard revalidates those bindings before
every outbound call. Outbound HTTP connects to the validated resolved address
with the configured hostname retained for Host/SNI.

Phase 8.2 adds a separate React/Vite presentation boundary. The Web UI consumes
generated OpenAPI types and calls coarse backend workflow endpoints; it does not
call Agents in sequence or derive readiness. Additive read adapters expose the
existing project, Agent catalog, benchmark, and provider evidence. The only new
write boundary outside existing registry APIs is the encrypted credential
bridge. Runtime default-model persistence remains a Router preference and still
passes every existing hard filter.

## Boundaries

```text
API -> reasoning workflow -> agent -> model router -> provider adapter
                      |          \-> versioned structured prompt
                      \-> deterministic quality/scoring -> ProblemState revision
                                                   \-> PostgreSQL audit evidence

Agent -> TaskProfile -> ModelRouter -> ModelProfile -> ProviderEndpoint
                         |                   \-> protocol adapter -> remote API
                         \-> task/config/probe digests -> execution guard
                                                    \-> registry audit evidence

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

verified raw result artifact -> versioned MetricSpec calculators -> independent values
                            -> reviewed ScenarioSpec -> fresh isolated execution
                            -> physical artifact/code/input hash re-audit
                            -> independent verification AND gate -> benchmark evaluator
                            (missing reviewed case policy remains NOT_READY)

verified_result_id -> EvidenceBuilder -> Claim/Evidence graph -> PaperAgent
                  -> Paper IR + registries -> deterministic factual gates
                  -> LaTeX/BibTeX -> restricted PDFCompiler -> manifest
                  -> READY_FOR_FINAL_JURY (never SUBMISSION_READY)

explicit paper/profile versions -> RuleEngine + RequirementCoverageValidator
                  -> FinalJuryAgent -> deterministic FinalJuryGate
                  -> SubmissionCheck -> SubmissionFreeze
                  -> canonical manifest + deterministic ZIP -> FROZEN
                  -> rehash/re-open verification; changed bytes -> DIRTY

official manifest + VERIFIED profile -> blind solve bundle -> Phase 1-7 project
                  -> atomic benchmark records -> deterministic evaluator
                  -> JSON/Markdown report -> SELF_TEST_READY or NOT_READY
                  (evaluation resources never enter the solve bundle)

browser -> typed API client -> FastAPI workflow/registry adapters
                               |-> ModelRouter hard filters remain authoritative
                               `-> encrypted credential write -> credential_ref
```

- `api`: transport concerns and health reporting only.
- `schemas`: durable domain and orchestration contracts.
- `routing`: deterministic task classification and model-level selection.
- `providers`: native/compatible protocol adapters, secret resolution, endpoint
  security, capability probing, and provider/model registry repositories. This
  is the only code allowed to know vendor HTTP shapes.
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
- `submission`: versioned rule profiles, deterministic rule and requirement
  checks, structured Final Jury review, correction invalidation, freeze gating,
  artifact allow-listing, canonical digest-linked manifests, deterministic ZIP
  construction, archive/security rechecks, idempotent freeze coordination, and
  current-chain plus post-freeze integrity verification.
- `benchmark`: official manifest/profile binding, host-allowlisted and hash-
  pinned source materialization, blind solve/evaluation isolation, Phase 1-7
  orchestration, immutable run/attempt records, atomic metrics/failures/human
  interventions, budget gates, deterministic score/acceptance recomputation,
  report redaction/rendering, and the minimal benchmark API.
- `frontend`: React Router pages, TanStack Query server-state handling, generated
  OpenAPI types, and plain-text rendering. It owns no workflow, routing, probe,
  verification, benchmark, or readiness rule.
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
Phase 7 adds `competition_profiles`, `competition_rules`,
`requirement_coverage`, `final_jury_reports`, `jury_findings`,
`submission_checks`, `submission_snapshots`, `submission_artifacts`,
`submission_manifests`, and `correction_plans`. Exact version/hash bindings are
relationally retained while immutable package bytes remain in the file store.
Profile, Jury, check, artifact-set, and snapshot digests are also stored in
separate indexed columns so JSON/scalar disagreement is detected on read.
Phase 8 adds `benchmark_runs`, `benchmark_attempts`,
`benchmark_case_results`, `benchmark_metrics`, `benchmark_failures`, and
`benchmark_human_interventions`. Run identity binds the baseline commit plus a
digest of tracked differences and untracked non-ignored source files, so a
formal run cannot misattribute an uncommitted Phase 8 tree to the Phase 7
commit. Terminal attempts and runs are immutable; reports are rebuilt from
atomic records rather than persisted totals.
Phase 8.1 adds `provider_endpoints`, `model_profiles`,
`capability_probe_runs`, and `agent_route_policies`, plus exact nullable
provider/model trace columns on historical AgentRun and BenchmarkRun records.
Only credential references are persisted. Configuration digests exclude
runtime health, timestamps, observed probe output, and secret values; each
atomic probe has its own digest, binds both configuration digests, and becomes
stale when configuration changes. Credentials are late-bound per request, while
AgentRun stores configured and provider-reported remote identities separately.
Phase 8.2 adds `encrypted_secrets` only. AES-GCM ciphertext and nonce are stored
under an opaque secret ID; the master key remains server environment state and
ProviderEndpoint continues to persist only `credential_ref`.

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
work. The ordinary Phase 7 API profile remains test-only, while Phase 8 owns an
isolated verified historical COMAP 2024 profile. That profile does not establish
compliance with current rules. Phase 8 now has accepted real DeepSeek evidence
and generated Case A solver executions, but no independent benchmark evaluator,
automatic benchmark crash/resume or asynchronous human-cancel coordinator, or
live case paper/package artifacts; its formal status remains `NOT_READY`.
The latest Case A model is a dynamic, no-objective system. Phase 5 currently
summarizes scalar optimization objectives and routes perturbations through
deterministic adapters; it does not yet provide generated dynamic-program
scenario replay, typed trajectory/equilibrium metrics, or independent Jacobian
stability/empirical checks. Those gaps must not be disguised as bounds checks,
fixed by inventing an objective, or bypassed by deleting scientific requirements.
Phase 7 also intentionally lacks an automatic code-reproduction executor; a
profile requiring it is fail-closed to human review rather than accepted from
README claims or a Mock run.
