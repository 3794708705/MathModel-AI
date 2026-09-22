# Custom providers

MathModel AI separates an API service from the models exposed by that service:

```text
ProviderEndpoint                         ModelProfile
where/how to call the API                which remote model to call
protocol and base URL                    stable internal model_id
credential reference                    remote_model API slug
network trust and health                 capabilities and quality tier
timeouts and response limit              reasoning/structured-output strategy
```

One endpoint can expose multiple model profiles. Agents and persisted routes use
the stable `model_id`; only the protocol adapter sees `remote_model`.

## Secrets

The registry stores a reference such as `env:DEEPSEEK_API_KEY`, never the secret
value. Set the referenced variable in the process environment or an untracked
local `.env` file. `EnvironmentSecretResolver` resolves it for every outbound
request, including requests made through a cached adapter, so rotation or
removal takes effect without rebuilding the process. Provider responses expose
only `credential_configured=true|false`; they do not expose the reference, a
value, prefix, suffix, hash, or length.

Do not put credentials in a base URL, query string, custom header, metadata,
model configuration, JSON mapping, repository file, or API request body.

## Add an OpenAI-compatible endpoint

Create the endpoint, then create at least one model profile. The following is a
configuration shape, not a file that is automatically imported:

```yaml
provider:
  provider_id: my-provider
  display_name: My Provider
  protocol: openai_chat_completions
  base_url: ${MY_PROVIDER_BASE_URL}
  credential_ref: env:MY_PROVIDER_API_KEY
  trust_level: USER_MANAGED_PROXY

model:
  model_id: my-main-model
  provider_id: my-provider
  display_name: My Main Model
  remote_model: ${MY_PROVIDER_MODEL}
  quality_tier: FLAGSHIP_HIGH
  quality_tier_source: USER_DECLARED
```

Use `POST /api/v1/providers` and `POST /api/v1/models`, then run `POST
/api/v1/models/{model_id}/probe`. A model is not routable merely because its API
accepted registration.

## DeepSeek

The `deepseek_official` preset supplies the official endpoint shape:

```yaml
provider:
  provider_id: deepseek-official
  protocol: openai_chat_completions
  base_url: https://api.deepseek.com
  credential_ref: env:DEEPSEEK_API_KEY
  trust_level: OFFICIAL_VENDOR

model:
  model_id: deepseek-main
  provider_id: deepseek-official
  remote_model: deepseek-v4-flash
```

The remote model name belongs in `ModelProfile`; it is not hard-coded in Router
or Agent code. The stable-text model hint must be refreshed against the official
API catalog before a live setup. Preset hints are not proof of runtime
capability. Run the probe.

## Qwen / Model Studio

Qwen endpoints can vary by account, region, and workspace, so the preset does
not invent a universal base URL:

```yaml
provider:
  provider_id: qwen-modelstudio
  protocol: openai_chat_completions
  base_url: ${QWEN_BASE_URL}
  credential_ref: env:DASHSCOPE_API_KEY

model:
  model_id: qwen-main
  provider_id: qwen-modelstudio
  remote_model: ${QWEN_MODEL}
```

Do not commit a workspace identifier. An endpoint that does not match a built-in
official domain must use `USER_MANAGED_PROXY`, even when `remote_model` contains
an official-looking product name.

## Arbitrary JSON HTTP APIs

`custom_json_http` is deliberately constrained. It accepts only `POST`, a
bounded relative endpoint path, static JSON fields, and these placeholders:

```text
{model} {system} {messages} {prompt} {max_tokens}
```

Response extraction uses dotted JSON field paths. Authentication is `BEARER` or
an API-key header whose value still comes from `SecretResolver`. Python, shell,
Jinja, filesystem, network, and `{secret}` templates are rejected.

```yaml
model:
  configuration:
    custom_json:
      request:
        method: POST
        endpoint_path: /generate
        body:
          model: "{model}"
          prompt: "{prompt}"
          max_tokens: "{max_tokens}"
      response:
        text_path: result.text
        input_tokens_path: usage.input
        output_tokens_path: usage.output
        total_tokens_path: usage.total
      auth_scheme: BEARER
```

Custom JSON starts with unknown capabilities and supports structured generation
only through an explicitly configured, Pydantic-validated prompt JSON fallback.

OpenAI-compatible structured output remains strict JSON plus Pydantic validation.
Malformed responses record parser-authored syntax diagnostics (line, column,
position), response length, and finish reason without retaining response content
in error messages. Markdown wrappers, extra prose, and partial JSON are rejected;
the Agent must issue a new bounded attempt to correct them.

## Capability probe

`ProviderCompatibilityProbe` uses short, bounded calls for authentication,
text, structured output, roles, a no-side-effect `dummy_test_tool`, reasoning,
streaming, usage, and optional one-pixel vision. It never executes the dummy
tool, and a tool probe passes only when the response names that exact inert tool.
Results have a canonical probe digest and bind the exact provider and model
configuration digests. Every probe is retained; the current probe is selected
deterministically by timestamp and probe ID.

Capability states are `SUPPORTED`, `PARTIAL`, `UNSUPPORTED`, or `UNKNOWN`.
User declarations and the mutable model capability summary do not silently
become verified support. Registry routing uses only atomic `PROBED` evidence from
the latest current probe and requires its authentication status to be `PASS`.
A provider/model configuration change makes old probe evidence stale.

A probe demonstrates observed protocol behavior. It does not independently
prove the identity of a model behind a proxy.

## Endpoint trust and network controls

- `OFFICIAL_VENDOR`: only a built-in official domain with its matching protocol.
- `USER_MANAGED_PROXY`: user or third-party gateway; model identity is not
  independently verified.
- `LOCAL_ENDPOINT`: allowed only when the server administrator sets
  `MM_ALLOW_LOCAL_MODEL_ENDPOINTS=true`.
- `UNKNOWN`: no stronger trust statement is available.

Custom endpoints require HTTPS by default. Configuration and every outbound
request re-resolve and validate all resolved addresses, rejecting loopback,
private, link-local, metadata, reserved, unspecified, IPv4-mapped-private,
mixed public/private, and unsupported schemes. The actual connection is pinned
to a validated address while retaining the original HTTP Host and TLS SNI, so a
second resolver lookup cannot redirect the socket. Redirects are disabled for
every 3xx response, TLS verification is on, connect/read/total timeouts are
bounded, and decoded responses are streamed only up to `max_response_bytes`. A
normal registration request cannot turn these server-level controls off.

## Test and route a model

1. `GET /api/v1/providers/{provider_id}` and verify
   `credential_configured=true`.
2. `POST /api/v1/models/{model_id}/probe` and inspect the capability matrix.
3. `POST /api/v1/model-routing/preview` to calculate a route without making a
   provider call.
4. Configure an Agent route with `PUT
   /api/v1/model-routing/agents/{agent_name}`.
5. Run a small live Agent smoke before resuming a paid benchmark.

A preview is advisory. The recorded decision binds the `TaskProfile`, provider,
model, and exact current probe digests. The Agent/provider execution boundary
revalidates all bindings, current health, credential availability, and probe
authentication immediately before each request; disabling, rotating, or
re-probing after preview therefore fails closed or uses the new secret safely.

Agent fallback lists may contain real registered models only. Mock remains a
test provider and can never satisfy a live fallback or benchmark gate.

## Disable and preserve history

There is intentionally no destructive provider/model delete API. Use the
provider enable/disable endpoints or patch `model.enabled=false`. Historical
AgentRun and BenchmarkRun records keep their exact provider/model IDs and
configuration digests after a model is disabled or superseded.

## Live gated tests

The following tests run only when their non-secret configuration and referenced
environment credential exist:

```text
MM_TEST_DEEPSEEK_MODEL_ID + DEEPSEEK_API_KEY
MM_TEST_QWEN_MODEL_ID + QWEN_BASE_URL + DASHSCOPE_API_KEY
MM_TEST_CUSTOM_PROVIDER_CONFIG + its credential_ref environment variable
```

Missing configuration is reported as `SKIPPED_NO_CREDENTIALS`, never PASS.
