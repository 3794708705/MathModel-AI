# Phase 2 — Reasoning Core execution plan

## Goal

Implement the validated and persisted reasoning chain:

```text
Problem input -> ProblemAgent -> ModelExplorer -> ModelJury
              -> selected model + backup model
```

Phase 2 stops at model selection. It does not parse attachments, generate code,
run solvers, or create papers.

## Current state

Phase 1 is green: FastAPI, settings, PostgreSQL/Alembic, `ProblemState` v1,
provider adapters, deterministic routing, `BaseAgent`, Docker, and CI are in
place. The repository has no commits yet, so all valid Phase 1 files remain
untracked and must be preserved. The state table already supports immutable
revisions and a single-current-revision constraint.

## Files affected

- `src/mathmodel_ai/schemas/`: problem analysis, candidate, score, selection,
  gate, trace, and `ProblemState` v2 contracts.
- `src/mathmodel_ai/agents/`: ProblemAgent, ModelExplorer, and ModelJury.
- `src/mathmodel_ai/reasoning/`: prompt registry, deterministic gates/scoring,
  persistence repository, state machine, and workflow orchestration.
- `src/mathmodel_ai/api/`: Phase 2 project and reasoning endpoints.
- `src/mathmodel_ai/db/` and `migrations/`: revision audit metadata, agent runs,
  and model-decision evidence.
- `src/mathmodel_ai/prompt_templates/`: versioned prompts outside Python code.
- `tests/`, `docs/`, and root configuration.

## Design

- `ProblemAnalysis` owns structured interpretation. Evidence items carry type,
  source, location, confidence, and proposal status; proposed assumptions never
  become accepted facts.
- `ProblemState` remains the only inter-agent handoff. Each successful stage
  creates an immutable database revision: v1 UNDERSTAND, v2 EXPLORE, v3 SELECT.
- Each reasoning agent makes one structured model call per attempt through the
  existing router/provider boundary. Large prompts are versioned resources.
- The LLM supplies qualitative dimension scores and rationale. Python validates
  hard failures, computes weighted totals, sorts candidates, and enforces that
  failed candidates cannot be selected.
- Quality gates block invalid stage advancement. Low-confidence ambiguity is
  retained as `HUMAN_REVIEW_RECOMMENDED` without crashing the workflow.
- Agent runs and selection evidence are relational audit records; full typed
  stage outputs remain in versioned `ProblemState` JSON.

## Implementation steps

1. Add Phase 2 schemas and configurable jury weights/retry threshold.
2. Add prompt registry and three versioned agent prompts.
3. Add deterministic candidate deduplication, scoring, gates, and transitions.
4. Implement the three agents and enrich base run trace metadata.
5. Implement state revision repository and reasoning workflow.
6. Add migration, API endpoints, exception mapping, and app wiring.
7. Add unit, API, persistence, optional live-provider, and mock E2E tests.
8. Run all Phase 1/2 gates and PostgreSQL migration cycles; update docs.

## Tests

- Optimization, prediction, and prediction/optimization/evaluation chain cases.
- Ambiguity preservation and proposed-assumption classification.
- Candidate semantic deduplication and minimum-candidate gate.
- Deterministic weighted totals, poor-data deep-model scoring, hard failures,
  ranking, and selected/backup invariants.
- State transition rejection for skipped stages.
- SQLite and PostgreSQL persistence of analysis, candidates, scores, selection,
  revisions, agent traces, and decision evidence.
- Full API and direct workflow E2E with explicit `is_mock=true`.
- Optional paid-provider integration test when a dedicated opt-in variable and
  key are present.

## Risks

- LLM outputs may be schema-valid but substantively weak: deterministic gates
  reject structural defects and preserve review flags, but mock tests cannot
  prove real modeling quality.
- State schema evolution can invalidate old JSON: v2 keeps Phase 1 fields and
  accepts empty new collections while incrementing `schema_version`.
- Long external calls must not hold database transactions: each state load/save
  and trace write uses a short transaction.
- Concurrent stage calls can race: current revisions are locked and the unique
  current-state index remains the final consistency guard.

## Acceptance criteria

- All three agents use `structured_generate()` and versioned prompts.
- All three deterministic quality gates pass the valid E2E case.
- State revisions 1/2/3 correspond to UNDERSTAND/EXPLORE/SELECT.
- Selected and backup candidates exist, differ, and have no hard failure.
- Agent run and model decision evidence survive database reload.
- Phase 1 tests, Phase 2 tests, Ruff, strict mypy, coverage, migrations, and mock
  E2E pass without implementing Phase 3 functionality.

