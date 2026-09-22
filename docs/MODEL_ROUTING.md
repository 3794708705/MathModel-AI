# Model routing

Agents submit an explicit `TaskProfile`. `ModelRouter` combines task minimum,
declared demand, deadline pressure, blast radius, retry count, capability and
context requirements, endpoint trust, health, cost sensitivity, explicit model
preference, and optional Agent policy. It returns a recordable `RouteDecision`;
it never calls a provider during route calculation.

Logical escalation is:

```text
FAST -> BALANCED -> FLAGSHIP_HIGH -> FLAGSHIP_XHIGH -> FLAGSHIP_MAX
     -> MULTI_MODEL_REVIEW -> HUMAN_REVIEW
```

Mathematical modeling, Model Repair, Red Team, and final acceptance start at
high critical tiers; documentation may start at FAST. Deterministic solver,
validation, perturbation, and quality-gate logic does not consume an LLM route.

## Registry route

The preferred configuration is the persisted runtime default selected through
`PUT /api/v1/model-routing/default`, followed by `MM_DEFAULT_MODEL_ID` when no
runtime preference exists. Router resolves that stable ID—or an explicit
`preferred_model_id`/Agent policy—to one
`ModelProfile` and its `ProviderEndpoint`:

```text
TaskProfile
  -> hard filters
  -> eligible ModelProfile scoring
  -> ProviderEndpoint/protocol binding
  -> RouteDecision with task/config/current-probe digests
  -> execution-time binding revalidation
```

Hard filters run before scoring:

- Provider and model enabled.
- Referenced credential configured.
- Provider health exactly READY and not in cooldown.
- Model is not Mock for a registry/live route.
- Minimum logical quality tier.
- Latest current probe has authentication `PASS`.
- Atomic `PROBED` evidence for every required capability.
- Structured output, JSON Schema, tools, reasoning, vision, context, and trust
  requirements.

User declarations and persisted capability summaries cannot bypass a hard
requirement. Prompt-JSON fallback is the only partial structured-output path and
still requires atomically probed text support. Probe evidence becomes stale when
either provider or model configuration digest changes. Price, quality labels,
deadline pressure, and retry count are preferences among already eligible
models; none can weaken correctness filters.

The runtime default is stored as a reserved route-policy row and is not returned
as an Agent policy. Selecting it in the Web UI never marks it eligible: preview
and execution continue through the same health, credential, trust, capability,
quality, and exact-probe hard filters.

`RouteDecision` binds `TaskProfile.requirements_digest`, provider/model config
digests, and the exact capability probe ID/digest. `BaseAgent` rejects a decision
for another task profile. `ProviderRegistry.get()` and its execution guard
re-read persisted configuration and the latest probe before every network call,
so a preview cannot survive a disable, credential removal, config change, or
new/failing probe. The configured remote model is the historical authority;
provider-reported model text is stored separately and cannot rewrite identity.

## Preference and fallback

An eligible explicit preference is honored. An ineligible preference is listed
in `rejected_models`; Router either selects an allowed real fallback and records
`fallback_reason`, or returns `NO_ELIGIBLE_MODEL`/HUMAN_REVIEW. An Agent route
restricts candidates to its primary plus de-duplicated ordered fallback list. It
never silently expands that list to Mock or to the legacy catalog.

Agent policies are managed through:

```text
GET /api/v1/model-routing/agents
PUT /api/v1/model-routing/agents/{agent_name}
POST /api/v1/model-routing/preview
```

Preview has `provider_called=false` by contract.

## Structured output and reasoning

Supported structured strategies are native JSON Schema, JSON mode, prompt JSON
fallback, and unsupported. Prompt fallback is permitted only when the task
policy allows it, remains explicitly traced as fallback, and still requires
parser/Pydantic validation and normal retries.

Logical reasoning is normalized to current `ReasoningEffort` values. A
ModelProfile maps each supported effort to its protocol value. If control is
only preferred and unsupported, the mapper sends no unknown parameter. If the
task requires reasoning control, the model is ineligible.

## Legacy route

If `MM_DEFAULT_MODEL_ID` is absent, `MM_DEFAULT_PROVIDER` and
`MM_DEFAULT_PROVIDER_MODEL` continue to use the native provider catalog and set
`legacy_config_used=true`. These variables are deprecated, not removed. If a
new default model ID is configured but missing, routing fails closed instead of
falling back to legacy Mock.

See [CUSTOM_PROVIDERS.md](CUSTOM_PROVIDERS.md) for setup and
[MODEL_REGISTRY.md](MODEL_REGISTRY.md) for persistence/evidence semantics.
