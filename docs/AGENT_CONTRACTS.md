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
- `MathModeler`: converts the accepted analysis, selected candidate, and optional
  data state into one complete `MathematicalModelDraft` via
  `structured_generate()`. Python binds model/project/problem identity and
  version. It may not emit results, claim execution, turn assumptions into facts,
  or embed solver SDK code in the model.
- `CodeAgent`: translates an already accepted mathematical model into a hashed,
  auditable `GeneratedProgram` for the formal generated execution path. It may
  not alter mathematics; unsupported translation becomes
  `CODE_GENERATION_BLOCKED`. Its approved source bundle runs through the shared
  sandbox and strict result contract. Standard LP/MILP/NLP/CP paths remain
  deterministic-first instead of asking a model to write boilerplate.
- `RedTeamAgent`: receives the exact model plus persisted validation,
  sensitivity, and robustness reports and emits structured adversarial findings
  classified as Critical, Major, or Minor. Deterministic attacks are merged in
  Python. Mock output cannot clear the Red Team gate.
- `ModelRepairAgent`: receives one unresolved Critical report and the complete
  verification context. It proposes the repository-assigned next model version
  plus scoped actions that cite finding IDs. It cannot change stable model
  identity, claim a result, or bypass MODEL/SOLVE/verification gates. Mock
  output is retained for orchestration evidence but requires human review.

All agents use `structured_generate()` and package-resource prompts. Phase 2
prompts are version `2.0.0`; the DataAgent prompt is version `3.0.0`; MathModeler
and CodeAgent prompts are version `4.0.0`; Red Team and Model Repair prompts are
version `5.0.0`.
Free-text-to-JSON regex parsing is not part of the agent contract.

`AlgorithmSelector`, `SolverRouter`, solver adapters, registries, unit checking,
feasibility recomputation, and quality gates are deterministic services—not
agents and not LLM calls. One MathModeler call produces the whole draft; there is
no per-variable or per-constraint call pattern. Independent validation and all
perturbation statistics are also deterministic Python/solver work, not agent
claims.
