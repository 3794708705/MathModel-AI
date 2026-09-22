# Product scope

MathModel AI accepts competition problems and attachments and builds a
traceable modeling submission. Correctness and provenance outrank novelty,
agent count, presentation, cost, and speed.

## Implemented product path

The current code implements the Phase 1-7 path from persisted problem analysis
through candidate selection, typed mathematical modeling, real sandboxed
solving, independent validation, executed sensitivity/robustness experiments,
Red Team and bounded immutable repair, evidence-grounded Paper IR, real PDF
compilation, deterministic Final Jury/rule checks, and hash-frozen submission
packages. A final numerical claim must resolve through the exact model, equation,
solver run, execution, result, verification, paper snapshot, and package digest.

Phase 8 implements the benchmark control plane: official case manifests,
hash-pinned resource retrieval, blind solve/evaluation separation, a verified
historical competition profile, immutable run/attempt/metric/failure/
intervention records, deterministic score and acceptance gates, budget/status
handling, reports, and a minimal API.

Phase 8.1 implements configurable non-Mock model access without coupling Agent
code to vendor brands. It supports existing native adapters, OpenAI-compatible
endpoints, and a constrained JSON HTTP adapter; stores provider/model/probe/Agent
route registries; enforces endpoint and secret controls; and routes by current
capability evidence. A successful proxy test is a protocol-compatibility result,
not an independently verified model-identity claim.

## Supported computation

Accepted deterministic paths include continuous LP through SciPy/HiGHS, small
MILP through SciPy, integral CP-SAT through OR-Tools, and basic continuous NLP
through SciPy. Gurobi is optional and fail-closed when its image or license is
unavailable. Generated programs use the same network-disabled non-root Docker
sandbox and strict result/evidence contracts.

## Current benchmark claim

Three official COMAP MCM 2024 cases (A, B, C) and one source-verified historical
profile are registered. Official files and the Problem C CSV were retrieved and
structurally validated; live Crossref lookup and DOI resolution passed. The
formal benchmark status is `NOT_READY`, not accepted: no live model-provider
credential was available, all cases stopped at `LIVE_PROVIDER`, and no live
reasoning, solver, paper, or package result exists for those cases.

Mock output proves schemas, routing, workflow, error handling, and persistence
only. It never proves mathematical quality, numerical success, literature
truth, Red Team clearance, repair acceptance, paper correctness, competition
readiness, or Phase 8 live acceptance.

## Out of scope or blocked

- Winning guarantees, universal problem coverage, and replacement of human
  judgment.
- Scanned-document OCR, arbitrary nonlinear/global model support, mandatory
  licensed solvers, and unconstrained host code execution.
- Independent benchmark LLM/human scoring, automatic benchmark crash/resume,
  asynchronous human cancellation, and general package reproduction when a
  competition requires it.
- A real competition-ready claim until live-provider cases, independent review,
  papers, and packages meet the Phase 8 thresholds.
