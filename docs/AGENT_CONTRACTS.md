# Agent contracts

`BaseAgent` exposes name, role, typed input/output schemas, capabilities,
`run()`, `validate_output()`, and bounded retry behavior. `run()` records
attempts, routes, provider/model/reasoning, prompt version, token use, latency,
errors, state versions, and mock status. Concrete agents may not bypass state
validation or directly instantiate vendor clients.

Concrete reasoning/data agents are:

- `ProblemAgent`: produces typed analysis, keeps assumptions proposed, and
  preserves low-confidence interpretations for human review.
- `ModelExplorer`: emits 2-5 structured candidates/model chains in one call and
  removes semantic duplicates before acceptance.
- `ModelJury`: obtains one qualitative assessment, then delegates weighted
  arithmetic, hard-failure enforcement, ranking, and primary/backup selection
  to deterministic Python.
- `DataAgent`: interprets deterministic profiles and optional untrusted media;
  it identifies semantic roles, likely units, problem alignment, and quality
  priorities but cannot change computed counts/statistics or invent columns.

All agents use `structured_generate()` and package-resource prompts. Phase 2
prompts are version `2.0.0`; the DataAgent prompt is version `3.0.0`.
Free-text-to-JSON regex parsing is not part of the agent contract.
