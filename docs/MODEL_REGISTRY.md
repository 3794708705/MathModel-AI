# Model registry

The registry is the durable boundary between business Agents and provider HTTP
details.

```text
Agent
  -> TaskProfile
  -> ModelRouter
  -> stable ModelProfile.model_id
  -> ProviderEndpoint
  -> protocol adapter
  -> remote_model
```

## Durable records

Migration `20260830_0010` adds:

- `provider_endpoints`: protocol, safe endpoint configuration, credential
  reference, health, trust, circuit metadata, and canonical digest.
- `model_profiles`: stable/remote IDs, quality tier and source, declared and
  observed capabilities, parameter strategies, pricing, trust, and digest.
- `capability_probe_runs`: immutable capability evidence bound to both digests.
- `agent_route_policies`: one primary and an ordered real-model fallback list.

It also snapshots exact provider/model identity on AgentRun and BenchmarkRun.
Secret values are absent from every table.

## Provider and model identity

`provider_id` and `model_id` are stable internal identifiers. `base_url` and
`remote_model` may change only by producing new configuration digests. Digests
exclude timestamps, database IDs, runtime health, observed probe output, and the
resolved secret value. They include fields that change request meaning, such as
protocol, endpoint, credential reference, remote model, declared capability,
reasoning mapping, and structured-output strategy.

Probe output is evidence about a configuration, not part of that configuration.
Changing a provider/model field invalidates the old digest-bound probe; Router
will reject stale probed capabilities until a new probe runs.

`canonical_provider_config()` is the single ProviderEndpoint digest source.
It materializes schema defaults, stable enum values, the canonical URL, safe
headers/metadata, and the credential reference before deterministic JSON hashing.
The resolved credential value is never available to this function. Runtime
health/circuit state, timestamps, catalog presets and model hints, connection
tests, discovery results, latency, usage, and probe evidence are excluded.

Persisted rows are reconstructed through this same schema and must match their
stored digest before either management or execution can use them. A valid row
whose endpoint is blocked by the current runtime network policy remains visible
to the management API so an operator can repair it, but `get_provider()` still
reapplies the runtime policy for Router, probe, discovery, and outbound use. A
digest mismatch remains a hard failure on both paths and is never rewritten on
read.

## Capability evidence

Capabilities include text, vision, structured output, JSON mode/schema, tools,
streaming, reasoning control, system/developer roles, long context, and usage
reporting. Each value has a status and source:

```text
status: SUPPORTED | PARTIAL | UNSUPPORTED | UNKNOWN
source: BUILTIN_VERIFIED | USER_DECLARED | PROBED | UNKNOWN
```

Observed probe evidence overrides declarations. A user-declared `SUPPORTED`
capability is conservatively effective as `PARTIAL` until verified; it cannot
bypass a hard requirement. Unknown optional protocol parameters are not sent.

## Routing

Router first applies hard filters:

```text
enabled provider/model
credential configured
provider health READY and outside cooldown
non-Mock live identity
current required capabilities
context and trust policy
```

Only eligible models are scored for quality fit, trust, cost sensitivity, and
preference. Quality tier is advisory, never a hard minimum; even critical
agents may use a lower-tier model if its live probe proves required capabilities.
Explicit model preference never overrides a hard filter. An Agent
policy restricts fallback to its configured ordered model list and records the
reason when a backup is used. Routing preview performs this calculation without
calling a provider.

Every successful AgentRun stores provider/model IDs and digests, protocol,
remote model, structured mode, requested/effective reasoning, endpoint trust,
usage, and Mock status. A response whose identity differs from the route is
rejected before persistence.

## Structured output and reasoning

Structured strategies are `NATIVE_JSON_SCHEMA`, `JSON_MODE`,
`PROMPT_JSON_FALLBACK`, and `UNSUPPORTED`. Prompt fallback is explicitly traced
and still requires parser/Pydantic validation and normal Agent retry behavior.
It is never described as native schema enforcement.

Reasoning remains normalized as `LOW`, `MEDIUM`, `HIGH`, `XHIGH`, and `MAX` in
current provider requests. A model-specific mapping controls the emitted value. If
reasoning is merely preferred and unverified, no parameter is sent; if a task
requires reasoning control, Router rejects the model.

Usage fields are normalized without estimation. Unreported token fields remain
`null`; they are not silently converted into provider-reported zero usage.

## Legacy configuration

`MM_DEFAULT_MODEL_ID` is the preferred configuration. If absent, the existing
`MM_DEFAULT_PROVIDER` and `MM_DEFAULT_PROVIDER_MODEL` path remains operational
and is marked `legacy_config_used`. The legacy variables are deprecated but not
removed in Phase 8.1. When a new default model is configured but missing or
ineligible, routing fails closed instead of falling back to legacy Mock.

## Benchmark binding

A benchmark that supplies `model_id` is resolved to the current registry before
the run is created. The run snapshots provider/model digests, protocol, endpoint
trust, and model-identity confidence. A later disable or configuration change
does not rewrite that history. `LIVE_PROVIDER=PASS` means only that a real
non-Mock API path succeeded; proxy model identity remains
`NOT_INDEPENDENTLY_VERIFIED`.
