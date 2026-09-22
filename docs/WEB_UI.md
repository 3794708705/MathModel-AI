# Web control center

Phase 8.2 is a lightweight local control plane over the accepted FastAPI
workflows. React renders backend state; it does not calculate provider,
benchmark, paper, submission, or system readiness and never orchestrates Agents.

## Local development

Python 3.12 is the tested backend baseline. On Windows, avoid Anaconda/base
Python 3.13 path ambiguity and use the repository `.venv`. Run `uv sync --dev`
once when dependencies change; startup scripts never install dependencies.

Run PostgreSQL and migrations from the repository root:

```powershell
docker compose up -d postgres
uv sync --dev
uv run alembic upgrade head
```

The canonical backend command is:

```powershell
.\scripts\dev-backend.ps1
```

This calls `.\.venv\Scripts\python.exe -m mathmodel_ai serve --reload`, which
uses the `mathmodel_ai.main:create_app` Uvicorn factory. The factory owns all
lifespan and dependency wiring; there is no module-level `application` ASGI
object. Use `--host`, `--port`, and `--reload` on the CLI when custom values are
needed. The launcher prints its URLs immediately and then intentionally occupies
that PowerShell window; keep it open and use `Ctrl+C` to stop it. A PostgreSQL
preflight attempt is capped so a stopped database produces an explicit warning
instead of an unbounded silent wait.

Start the UI in another terminal with `.\scripts\dev-frontend.ps1`; it also
remains in the foreground until `Ctrl+C`. If
`frontend/node_modules` does not exist, run `npm install` in `frontend` first;
the script intentionally does not install packages. The launcher probes
`/health/live` before Vite starts and prints a non-blocking warning with the
exact backend command when port 8000 is unavailable.

The default UI is `http://127.0.0.1:5173`; the default API is
`http://127.0.0.1:8000`. Set `VITE_API_BASE_URL` before build/start for another
API origin, and add that UI origin to the backend `MM_CORS_ORIGINS` JSON list.
Wildcards are rejected. Development ports 5173 and 5174 are allowed by default.
Vite uses `strictPort`, so an occupied requested port fails clearly instead of
silently moving the UI to an origin that the backend may reject. When an
alternate `-Port` is requested, the launcher verifies its exact origin against
the running backend before starting Vite.

The sidebar's Backend indicator is derived only from `/health/live`; database
readiness cannot turn process liveness into an offline state. Browser failures
are reported separately as `BACKEND_OFFLINE`, `CORS_BLOCKED`,
`REQUEST_TIMEOUT`, `NETWORK_ERROR`, or their actual HTTP status. The local
offline message includes `.\scripts\dev-backend.ps1` instead of the ambiguous
legacy `HTTP_0` label.

FastAPI docs are available at `http://127.0.0.1:8000/docs`. `/health/live`
reports only process liveness; `/health/ready` independently reports database
readiness. Startup does not require OpenAI, Google, Anthropic, DeepSeek, or Qwen
credentials.

For a local integrated build, create an untracked `.env` (optionally including
`MM_SECRET_MASTER_KEY`) and run `docker compose up --build`. Compose starts
PostgreSQL, applies migrations, serves FastAPI on port 8000, and serves the
static UI on port 5173. The frontend image accepts a build-time
`VITE_API_BASE_URL`; its default is `http://localhost:8000`.

## Contract and build

`npm run api:generate` asks the FastAPI application for its OpenAPI document and
regenerates `src/api/schema.d.ts`. All component traffic is centralized in
`src/api/client.ts`. Use `npm run typecheck`, `npm run lint`,
`npm run test:coverage`, and `npm run build` before delivery.

## Models and API

Models & API lists the public ProviderEndpoint and ModelProfile views. A
Provider is an endpoint/account configuration and can own any number of model
profiles. Default Model and the provider inventory appear first. Every provider
card shows model count, write-only credential status, backend health, Test
Connection, and Manage; protocol, endpoint, timeout, trust, replacement key,
and disable controls stay under Advanced provider settings.

The Provider Catalog contains DeepSeek, Qwen / Alibaba Model Studio, OpenAI,
Anthropic, and Google Gemini backend presets. Zhipu GLM and Moonshot / Kimi are
offered as Custom Provider entry points because the UI does not invent an
endpoint that may vary by account or region. Catalog provider IDs are generated
from the backend preset and are not editable. The ordinary catalog path is API
Key -> Save & Test; protocol, base URL, and timeouts are visible under Advanced.
Custom Provider requires an explicit Provider ID, display name, public Base URL,
protocol, and credential. Its Base URL starts empty and never defaults to
localhost.

Connection testing verifies only endpoint reachability and credential
acceptance. Fetch Models imports remote identifiers and creates explicit model
profiles, while manual entry remains available when discovery is unsupported.
Neither operation creates capability evidence. Each model card separately shows
its provider, remote model, status, current Probe timestamp, conservative
capability labels, and Probe Again/Edit/Disable controls. The symbols `✓`, `△`,
`✕`, and `?` mean `SUPPORTED`, `PARTIAL`, `UNSUPPORTED`, and `UNKNOWN`; declared
values are never displayed as probed evidence. A probe may issue a small paid
provider request. Provider URLs render as text, not links. Unknown future
capability/status values render neutrally rather than being silently
interpreted. The built-in Mock provider and all of its models carry a visible
`MOCK / TEST ONLY` label.

The runtime default applies to the next eligible request only when an Agent has
no explicit route. It is not a unique global model and never replaces
stage-specific assignments.

An existing Provider whose canonical persisted configuration is valid but whose
destination is blocked by the current server policy remains listed as
`UNAVAILABLE`. This does not authorize network use; Test Connection, model
discovery, probing, routing, and execution stay blocked until the endpoint is
edited to satisfy the current policy (or an administrator explicitly enables
the appropriate local development policy and restarts the backend).
`MM_ALLOW_LOCAL_MODEL_ENDPOINTS` remains `false` by default. Local endpoint
trust is shown only in Custom Provider Advanced settings, and a rejected
localhost/private address receives a direct server-policy explanation.

For the encrypted UI credential path, configure a random URL-safe base64
32-byte `MM_SECRET_MASTER_KEY` only in the backend environment. The API encrypts
the key with AES-GCM, rotates to a new opaque `secret:provider/...` reference,
and returns only configuration status. The browser does not use localStorage,
sessionStorage, IndexedDB, URLs, or logs for keys and clears the controlled input
after success. Existing `env:VARIABLE_NAME` references continue to resolve.
Removing a key requires confirmation and lets the backend invalidate health and
probe evidence.
If the master key is absent in a new database, the application still starts and
System displays `Encrypted credential store: Not configured`. If encrypted rows
already exist, startup warns without exposing any key or ciphertext, and all
stored credential resolution remains fail-closed. No temporary key is generated.
Models & API then disables direct secret entry before the user can type and
offers `env:VARIABLE_NAME` references for credentials supplied to the backend
environment. Existing provider views truthfully say only configured/not
configured because the API intentionally hides the exact credential source.

## Agent routing

The **Agent Routing / 阶段模型分配** page loads the actual backend Agent catalog
and adds human-readable labels without changing Agent IDs. Each stage supports
a primary model in the ordinary view and ordered fallback 1/2 controls under
Advanced. Save remains disabled until
`/model-routing/preview` returns `EXECUTE` for every selected profile; Router
hard-filter rejections remain visible and authoritative. The backend rechecks
every primary/fallback profile on save and again at execution time. Applying the
default through **Use Default Model for All Stages** is an explicit shortcut
that writes ordinary Agent route policies; it is not a second routing mechanism
and does not hard-code a vendor.

The visual flow borrows only general usability ideas from reference provider
control panels. DeepSeek Harness is not started, proxied, imported, depended on,
or connected at runtime. The unchanged runtime path is Web UI -> FastAPI ->
ProviderRegistry -> ModelProfile -> ProtocolAdapter -> provider API.

## Projects and workspace

Projects supports basic creation and opening. Workspace displays current
ProblemState, files, selected/verified evidence IDs, paper, and submission state.
Each action calls one existing coarse workflow endpoint and then refetches the
affected backend state. The browser does not call ProblemAgent, ModelExplorer,
or other Agents directly and does not poll terminal project/paper/submission
records indefinitely.

## Benchmark and system

Benchmark displays persisted run history, including `FAIL` and `BLOCKED`
attempts, and opens the backend-generated report. Starting a run posts the
selected real model, case IDs, limits, and pricing to the existing Phase 8 API;
the UI does not recompute acceptance. Dashboard and System show only fields
actually exposed by health/system APIs and label absent data as not exposed.
Benchmark reports poll only while their persisted run status is `RUNNING`.

## Independent acceptance behavior

Credential operations use component-local transient state rather than the
TanStack Query mutation cache. Inputs are cleared before the request begins,
double submission is locked, refresh failures cannot turn a completed secret
write into a displayed write failure, and error bodies are not trusted to be
secret-free. Provider creation reconciles an ambiguous network loss only when
the backend contains an exact matching configuration, preventing an automatic
duplicate POST.

Router save authority expires when the backend rejects a save after preview.
The backend independently revalidates every primary/fallback choice, so direct
API calls and preview-to-save races cannot bypass the Router. The UI renders
backend benchmark and readiness status verbatim and never promotes
`BLOCKED`, `NOT_READY`, `SKIPPED`, or Mock evidence.

## Limitations

- There is no authentication or multi-tenant boundary; deploy only as a trusted
  local/self-hosted control plane behind an appropriate network boundary.
- There is no WebSocket/SSE event stream. Polling is intentionally used.
- A live capability probe and Phase 8 benchmark still require user-supplied
  non-Mock credentials. Component fixtures do not establish live readiness.
- Solver, Docker, PDF compiler, migration, and Git details are shown only when
  the backend exposes them; the UI does not infer missing status.
