# Phase 8 Real Competition Benchmark

## Goal

Run MathModel AI against at least three official historical competition cases,
retain every formal attempt, and produce evidence-backed benchmark reports that
cannot turn Mock, stale, cherry-picked, or manually edited summaries into a
successful live benchmark.

## Benchmark Philosophy

Benchmark first, diagnose the observed failure, make the smallest generic fix,
add a regression, and rerun affected plus control cases. Correctness,
reproducibility, and failure transparency outrank score, speed, and presentation.

## Competition Selection

The first set uses the official COMAP 2024 MCM archive:

- `MCM-2024-C`: data/prediction, including the official Wimbledon CSV data.
- `MCM-2024-B`: search, uncertainty, and optimization.
- `MCM-2024-A`: ecological simulation and multi-stage impact evaluation.

This set covers materially different model families while sharing one historical
rule version. Official problem and attachment URLs are recorded in versioned
manifests; copyrighted source bytes remain in ignored benchmark storage.

## Rule Verification

The official problem PDFs contain the 25-page rule, required report components,
and the contemporaneous COMAP generative-AI policy. A proposed profile is built
only from page/section-provenanced official text. Duplicate/conflict, file,
anonymity, submission-format, and AI-disclosure semantics are independently
checked before the profile can be marked `VERIFIED`. Unknown historical rules
remain `HUMAN_REVIEW`.

## Benchmark Isolation

Solve resources contain only official problems, attachments, verified rules,
and legitimate literature. Evaluation resources use a separate manifest and
are inaccessible to the solve loader. Official solutions, winning papers,
result commentary, expected model families, and human notes never enter solve
inputs.

## Ground Truth Policy

Ground truth is layered into hard facts, required outputs, known constraints,
reference ranges, reasonable model families, invalid approaches, and historical
quality signals. Only hard facts/requirements/constraints participate in
deterministic solve-time scoring. Evaluation-only material is introduced after
an attempt is sealed.

## Live Provider Plan

Critical ProblemAgent, ModelExplorer, ModelJury, MathModeler, PaperAgent, and
FinalJury calls must use the configured non-Mock provider. Provider identity,
model, reasoning tier, usage, latency, retries, and sanitized failures are
retained. Missing credentials or provider failure creates
`BLOCKED_ENVIRONMENT`; it never falls back to Mock. At plan creation, no live
provider credential is present in the process environment, so this is an active
acceptance blocker until external configuration changes.

## Live Literature Plan

Use the existing Crossref adapter for at least one case, retaining retrieval
time, source identity, metadata, trusted excerpt, and payload digest. Network or
source failure produces `NOT_VERIFIED`; no reference may be invented.

## Execution Plan

1. Add typed manifests, run/attempt/results/metrics/failure/intervention records,
   canonical digests, deterministic score and acceptance gates, report output,
   and the minimal API. A CLI remains optional and was not added.
2. Persist all attempts before any solve work and forbid deletion through the
   business repository.
3. Verify official source bytes and the historical competition profile.
4. Inspect attachments and compare deterministic structure with manifest facts.
5. Run cases blind; seal attempts before evaluation; retain retries and failures.
6. Apply only generic P0/P1 and materially relevant P2 fixes, then rerun failed
   and control cases.
7. Recompute reports from immutable records, run full regression, and stop at
   `SELF_TEST_READY` or `NOT_READY` without committing.

## Metrics

Record problem/constraint/requirement recall, extraction accuracy, model and
mathematical ratings, solver/validation/experiment outcomes, Red Team/repair,
citation integrity, paper/submission completeness, runtime breakdown, tokens,
provider/solver calls, retries, cost estimates, seeds, and human interventions.
Pricing is supplied by versioned benchmark configuration rather than code.

## Failure Taxonomy

Use the Phase 8 categories from problem understanding through provider and
infrastructure, with P0-P3 severity. Every failure binds stage, evidence,
reproducibility, generic/case-specific classification, root cause, and proposed
fix. No formal attempt can be removed from the final report.

## Regression Strategy

Add adversarial tests for reference leakage, digest/status/score/cost/tokens/
human-count tampering, hidden failed attempts, fake live status, prompt
injection as data, budget cancellation, report recomputation, and API bypass.
Retain every Phase 1-7 test and overall coverage of at least 85%.

## Acceptance Thresholds

- At least three real cases attempted and all attempts reported.
- At least two cases are `PASS` or `PASS_WITH_WARNINGS`.
- At least one real CompetitionProfile is `VERIFIED`.
- Live provider is `PASS`; live literature passes on at least one case.
- No fabricated result/citation, hidden P0, or missing required subproblem.
- Every final numeric claim is evidence-backed and every package manifest valid.
- At least two real packages are frozen or technically frozen with an explicit
  rule-policy block.
- Full regression and coverage gates pass.

## Final Project Acceptance

Phase 8 may only report `SELF_TEST_READY` or `NOT_READY`; project acceptance is
reserved for the independent benchmark audit and human acceptance. The final
report distinguishes implemented, deterministically validated, live-tested,
environment-dependent, blocked, and not implemented capabilities.

## Infrastructure Gap Analysis Result

Phase 1-7 supplied the project workflows but had no immutable benchmark history,
blind materialization, real-profile registry, deterministic cross-case gate,
reporting, budget status, or benchmark API. Phase 8 added only those generic
control-plane capabilities and reused the existing provider, literature, solver,
verification, paper, and submission implementations. It did not add case-keyword
routing or model hints.

## Formal Run Result

Formal run `0fabda14-058c-45c4-8245-25bddf58b668` retained one official attempt
for each COMAP 2024 MCM A/B/C case. All official resources passed URL, size,
SHA-256, and structural validation; live Crossref search and independent DOI
resolution passed for every case. No configured OpenAI, Google, or Anthropic
credential existed, so each attempt stopped at `LIVE_PROVIDER` as
`BLOCKED_ENVIRONMENT`. No Mock fallback occurred and provider calls, tokens,
cost, solves, experiments, papers, and packages are all zero. Acceptance is
`NOT_READY`, with three explicit P1 provider blockers and no hidden P0.

## Generic Fixes From Self-test

- Crossref filtering incorrectly discarded valid book/report records without a
  container title. The adapter now accepts the retrieved publisher and checks
  multiple official date fields; regression uses real-shaped metadata.
- Registering the historical benchmark profile in the ordinary API registry
  changed Phase 7 profile selection. Benchmark and ordinary profiles now use
  isolated immutable registries; the Phase 7 real PDF/package E2E passes again.
- Benchmark execution counts now derive solver and perturbation runs from actual
  verification/repair outcomes. Human-rubric dimensions remain zero until an
  independent evaluator runs, and blocking-rule counts use rule severity rather
  than all non-pass statuses.
- Source identity now binds the base commit, tracked binary diff, and untracked
  non-ignored files. Pipeline exceptions become terminal recorded attempts;
  wall/case/total budgets cannot be reported as successful completion.

## Remaining Blockers

- A real provider credential and paid live critical-agent chain are absent.
- Consequently no real Phase 8 solver, validation, perturbation, repair, PDF,
  Final Jury, or package evidence exists and no case can pass.
- Independent LLM/human rubric scoring, automatic benchmark crash/resume,
  asynchronous human stop, and general package reproduction are not implemented.
- The formal run must be repeated after secure provider configuration; all prior
  failed attempts/runs remain part of benchmark history.
