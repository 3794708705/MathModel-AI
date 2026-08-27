# Reasoning core

Phase 2 implements this bounded chain:

```text
INGEST -> ProblemAgent -> UNDERSTAND gate -> state v1
       -> ModelExplorer -> EXPLORE gate    -> state v2
       -> ModelJury     -> SELECT gate     -> state v3
```

## Truth and decision contracts

`ProblemAnalysis` distinguishes `FACT`, `DATA`, `ASSUMPTION`, and
`DERIVATION`. Problem-agent assumptions must remain `proposed`; ambiguity keeps
at least two interpretations, a preferred interpretation and reason when one is
recommended, confidence, and a human-review marker. Subproblems identify task
types, required outputs, data needs, and directed dependencies.

`ModelExplorer` emits 2-5 structured candidates. Fewer than three requires a
reason. Each candidate declares its mathematical core, I/O, data, assumptions,
advantages, limitations, validation/robustness potential, implementation and
computational cost, risks, and model-chain stages.

`ModelJury` supplies qualitative 0-10 dimensions and rationale in one structured
call. Python applies configurable weights totaling 100. Required data marked
`missing` creates `DATA_NOT_AVAILABLE` even if the model assessment omitted the
failure. Any hard failure makes a candidate ineligible for primary and backup.

## Gates and trace

`UNDERSTAND`, `EXPLORE`, and `SELECT` gates use schemas and deterministic
checks; model confidence is never the only acceptance input. Invalid stages are
not persisted as successful revisions. Low-confidence ambiguity can pass with a
visible human-review warning because preserving uncertainty is preferable to
silently choosing an interpretation.

Every accepted stage atomically stores its state revision and `AgentRun` after
the external model call completes. The SELECT transaction also stores
`ModelDecisionEvidence`. Mock runs expose `is_mock=true` through state-adjacent
API responses and audit rows.

## API

All paths use the existing `/api/v1` prefix:

```text
POST /projects
POST /projects/{id}/problem/analyze
POST /projects/{id}/models/explore
POST /projects/{id}/models/select
POST /projects/{id}/reasoning/run
GET  /projects/{id}/problem-analysis
GET  /projects/{id}/models
GET  /projects/{id}/model-selection
```

The Phase 2 reasoning chain itself deliberately does not parse files, retrieve
literature, execute code, solve a model, or create a paper. Phase 4 consumes its
persisted `SelectedModel` downstream through `SELECT -> MODEL -> SOLVE`; it does
not weaken or reinterpret Phase 2 evidence contracts.
