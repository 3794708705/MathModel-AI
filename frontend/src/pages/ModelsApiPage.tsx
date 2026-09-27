import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { ApiError, api } from "../api/client";
import {
  CapabilityMark,
  DefinitionList,
  EmptyState,
  ErrorBanner,
  LoadingState,
  PageHeader,
  Panel,
  StatusBadge,
  formatDate,
} from "../components/ui";
import {
  capabilityNames,
  type Model,
  type ModelDiscoveryCandidate,
  type ModelInput,
  type Provider,
  type ProviderInput,
  type ProviderPreset,
} from "../types/contracts";

export function ModelsApiPage() {
  const queryClient = useQueryClient();
  const providers = useQuery({ queryKey: ["providers"], queryFn: api.providers });
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const presets = useQuery({ queryKey: ["provider-presets"], queryFn: api.providerPresets });
  const routing = useQuery({ queryKey: ["routing"], queryFn: api.routingOverview });
  const system = useQuery({ queryKey: ["system"], queryFn: api.system });
  const [providerFlow, setProviderFlow] = useState<"catalog" | "custom" | null>(null);
  const [customTemplate, setCustomTemplate] = useState<{ provider_id: string; display_name: string } | null>(null);
  const [setupNotice, setSetupNotice] = useState<{ tone: "success" | "warning"; message: string } | null>(null);
  const [showModelForm, setShowModelForm] = useState(false);
  const [manualProviderId, setManualProviderId] = useState<string | null>(null);
  const [managedProviderId, setManagedProviderId] = useState<string | null>(null);
  const refresh = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["providers"] }),
      queryClient.invalidateQueries({ queryKey: ["models"] }),
      queryClient.invalidateQueries({ queryKey: ["routing"] }),
    ]);
  };
  const defaultModel = useMutation({ mutationFn: api.setDefaultModel, onSuccess: refresh });
  const error = providers.error || models.error || presets.error || routing.error || system.error || defaultModel.error;

  if (providers.isLoading || models.isLoading || presets.isLoading || routing.isLoading || system.isLoading) {
    return <LoadingState label="Loading provider and model registry…" />;
  }

  return (
    <>
      <PageHeader
        title="Models & API"
        description="Connect API providers, register one or many models for each provider, and choose which model each Agent stage should use."
        actions={
          <>
            <button className="button button-primary" onClick={() => { setCustomTemplate(null); setProviderFlow((value) => value === "catalog" ? null : "catalog"); }}>Add Provider</button>
            <button className="button" onClick={() => { setCustomTemplate(null); setProviderFlow((value) => value === "custom" ? null : "custom"); }}>Add Custom Provider</button>
          </>
        }
      />
      {error ? <div className="mb-5"><ErrorBanner error={error} /></div> : null}
      {setupNotice ? <p role="status" className={`mb-5 rounded-md border px-4 py-3 text-sm ${setupNotice.tone === "success" ? "border-emerald-200 bg-emerald-50 text-emerald-800" : "border-amber-200 bg-amber-50 text-amber-900"}`}>{setupNotice.message}</p> : null}

      <Panel title="Default Model / 默认模型" description="Used for the next request only when an Agent has no explicit stage route. It is not the only global model and never replaces stage-specific assignments. 默认模型只是不设单独路由时的候选，各阶段仍可使用不同模型。" className="mb-5">
        <div className="flex max-w-xl flex-col gap-2 sm:flex-row">
          <label className="field flex-1">
            <span className="sr-only">Default model</span>
            <select
              aria-label="Default model"
              className="input"
              value={routing.data?.default_model_id ?? ""}
              disabled={defaultModel.isPending || !models.data?.length}
              onChange={(event) => defaultModel.mutate(event.target.value)}
            >
              <option value="" disabled>Select a registered model</option>
              {models.data?.map((model) => {
                const provider = providers.data?.find((item) => item.provider_id === model.provider_id);
                return <option key={model.model_id} value={model.model_id} disabled={!model.enabled}>{provider?.display_name ?? model.provider_id} / {model.display_name}{model.enabled ? "" : " · unavailable"}</option>;
              })}
            </select>
          </label>
          <span className="self-center text-sm text-muted">{defaultModel.isPending ? "Saving…" : "Applies to the next eligible request"}</span>
        </div>
      </Panel>

      <div className="mb-5 space-y-4">
        <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
          <div>
            <h2 className="text-lg font-semibold">Providers ({providers.data?.length ?? 0})</h2>
            <p className="text-sm text-muted">Connection status is provider-level. Capability evidence is tested separately for each model.</p>
          </div>
          <button className="button" onClick={() => { setManualProviderId(null); setShowModelForm((value) => !value); }}>Add Model</button>
        </div>
        {providers.data?.length ? (
          <div className="grid gap-4 xl:grid-cols-2">
            {providers.data.map((provider) => (
              <ProviderCard key={provider.provider_id} provider={provider} modelCount={models.data?.filter((model) => model.provider_id === provider.provider_id).length ?? 0} secretStoreAvailable={system.data?.secret_store_configured ?? false} onManage={() => setManagedProviderId(provider.provider_id)} onChange={refresh} />
            ))}
          </div>
        ) : <EmptyState>No providers registered. Add a catalog provider or a custom endpoint to begin.</EmptyState>}
      </div>

      {providerFlow ? (
        <ProviderCreatePanel
          key={providerFlow}
          mode={providerFlow}
          initialCustom={customTemplate ?? undefined}
          presets={presets.data ?? []}
          secretStoreAvailable={system.data?.secret_store_configured ?? false}
          onChooseCustom={(template) => {
            setCustomTemplate(template);
            setProviderFlow("custom");
          }}
          onDone={async (providerId, notice) => {
            setProviderFlow(null);
            setCustomTemplate(null);
            setManagedProviderId(providerId);
            setSetupNotice(notice);
            await refresh();
          }}
        />
      ) : null}
      {managedProviderId ? (
        <ModelCatalogPanel
          provider={providers.data?.find((item) => item.provider_id === managedProviderId)}
          models={models.data ?? []}
          onClose={() => setManagedProviderId(null)}
          onCreated={refresh}
          onAddManual={() => {
            setManualProviderId(managedProviderId);
            setShowModelForm(true);
            setManagedProviderId(null);
          }}
        />
      ) : null}
      {showModelForm ? <ModelCreatePanel providers={providers.data ?? []} initialProviderId={manualProviderId ?? undefined} onCreated={refresh} /> : null}

      <div className="mt-5 space-y-4">
        <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
          <div>
            <h2 className="text-lg font-semibold">Models ({models.data?.length ?? 0})</h2>
            <p className="text-sm text-muted">Remote model identity, current probe evidence, and controls remain visible per model.</p>
          </div>
          <Link className="button" to="/routing">Agent Routing / 阶段模型分配</Link>
        </div>
        {models.data?.length ? (
          <div className="grid gap-4 xl:grid-cols-2">
            {models.data.map((model) => (
              <ModelCard key={model.model_id} model={model} provider={providers.data?.find((provider) => provider.provider_id === model.provider_id)} onChange={refresh} />
            ))}
          </div>
        ) : <EmptyState>No models registered. Manage a provider to fetch models or add one manually.</EmptyState>}
      </div>
    </>
  );
}

const protocolLabels: Record<ProviderInput["protocol"], string> = {
  openai_chat_completions: "OpenAI Chat Completions",
  openai_responses: "OpenAI Responses",
  anthropic_messages: "Anthropic Messages",
  google_generate_content: "Google Generate Content",
  custom_json_http: "Custom JSON HTTP",
};

const catalogDisplayNames: Record<string, string> = {
  deepseek_official: "DeepSeek",
  qwen_modelstudio: "Qwen / Alibaba Model Studio",
  openai_official: "OpenAI",
  anthropic_official: "Anthropic",
  google_ai: "Google Gemini",
};

const customCatalogEntries = [
  {
    provider_id: "zhipu-glm",
    display_name: "Zhipu GLM",
    description: "Configure as a custom provider; enter the endpoint supplied for your account.",
  },
  {
    provider_id: "moonshot-kimi",
    display_name: "Moonshot / Kimi",
    description: "Configure as a custom provider; enter the endpoint supplied for your account.",
  },
] as const;

const capabilityDisplayNames: Record<string, string> = {
  TEXT: "Text",
  VISION: "Vision",
  STRUCTURED_OUTPUT: "Structured output",
  JSON_MODE: "JSON mode",
  JSON_SCHEMA: "JSON Schema",
  TOOLS: "Tools",
  STREAMING: "Streaming",
  REASONING_CONTROL: "Reasoning control",
  SYSTEM_ROLE: "System role",
  DEVELOPER_ROLE: "Developer role",
  LONG_CONTEXT: "Long context",
  USAGE_REPORTING: "Usage reporting",
};

function providerIdForPreset(preset: ProviderPreset): string {
  return preset.preset_id.replaceAll("_", "-").replace(/-official$/, "-official");
}

function ProviderCreatePanel({ mode, initialCustom, presets, secretStoreAvailable, onChooseCustom, onDone }: { mode: "catalog" | "custom"; initialCustom?: { provider_id: string; display_name: string }; presets: ProviderPreset[]; secretStoreAvailable: boolean; onChooseCustom: (template: { provider_id: string; display_name: string } | null) => void; onDone: (providerId: string, notice: { tone: "success" | "warning"; message: string }) => Promise<void> }) {
  const effectiveMode = mode === "catalog" && presets.length === 0 ? "custom" : mode;
  const [selectedPresetId, setSelectedPresetId] = useState("");
  const [form, setForm] = useState({
    provider_id: initialCustom?.provider_id ?? "",
    display_name: initialCustom?.display_name ?? "",
    protocol: "openai_chat_completions" as ProviderInput["protocol"],
    base_url: "",
    trust_level: "USER_MANAGED_PROXY" as ProviderInput["trust_level"],
    credential_ref: "",
    apiKey: "",
    connect_timeout_seconds: 10,
    timeout_seconds: 60,
  });
  const create = useLocalOperation();
  const submit = async () => {
    const apiKey = form.apiKey;
    setForm((current) => ({ ...current, apiKey: "" }));
    const input: ProviderInput = {
      provider_id: form.provider_id,
      display_name: form.display_name,
      protocol: form.protocol,
      base_url: form.base_url,
      trust_level: form.trust_level,
      credential_ref: form.credential_ref || null,
      allow_redirects: false,
      connect_timeout_seconds: form.connect_timeout_seconds,
      enabled: true,
      max_response_bytes: 8_388_608,
      timeout_seconds: form.timeout_seconds,
      verify_tls: true,
    };
    let createdProviderId = "";
    let connectionPassed = false;
    let connectionAttempted = false;
    const succeeded = await create.run(async () => {
      let provider: Provider;
      try {
        provider = await createProviderWithReconciliation(input);
      } catch (caught) {
        throw providerSetupError(caught, input.base_url);
      }
      createdProviderId = provider.provider_id;
      if (apiKey) await api.putCredential(provider.provider_id, apiKey);
      if (apiKey || input.credential_ref) {
        connectionAttempted = true;
        try {
          const result = await api.testProviderConnection(provider.provider_id);
          connectionPassed = result.connected && result.credential_accepted;
        } catch {
          connectionPassed = false;
        }
      }
    });
    if (succeeded) {
      const notice = connectionPassed
        ? { tone: "success" as const, message: "Provider saved and connection test passed. Add or fetch model profiles next." }
        : connectionAttempted
          ? { tone: "warning" as const, message: "Provider saved, but the connection test did not pass. Review the credential or endpoint, then run Test Connection again." }
          : { tone: "warning" as const, message: "Provider saved without a credential. Add an API key before testing the connection or fetching models." };
      await onDone(createdProviderId, notice).catch(() => undefined);
    }
  };
  const applyPreset = (presetId: string) => {
    const preset = presets.find((item) => item.preset_id === presetId);
    if (!preset) return;
    setSelectedPresetId(presetId);
    setForm((current) => ({
      ...current,
      provider_id: providerIdForPreset(preset),
      display_name: preset.display_name,
      protocol: preset.protocol,
      base_url: preset.base_url ?? "",
      trust_level: preset.trust_level,
      credential_ref: preset.recommended_credential_ref,
    }));
  };
  const selectedPreset = presets.find((item) => item.preset_id === selectedPresetId);
  if (mode === "catalog" && !selectedPreset && presets.length) {
    return (
      <Panel title="Provider Catalog" description="Choose a provider, add its API key, then save and test. Catalog setup never proves model capabilities; each model still needs its own Probe." className="mb-5">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {presets.map((preset) => (
            <button key={preset.preset_id} type="button" className="rounded-lg border bg-white p-4 text-left transition hover:border-accent hover:shadow-sm" onClick={() => applyPreset(preset.preset_id)}>
              <span className="block font-semibold">{catalogDisplayNames[preset.preset_id] ?? preset.display_name}</span>
              <span className="mt-1 block text-xs text-muted">{protocolLabels[preset.protocol]}</span>
              <span className="mt-2 block text-xs text-muted">Known models: {preset.model_hints.length ? preset.model_hints.join(", ") : "fetch after save"}</span>
            </button>
          ))}
          {customCatalogEntries.map((entry) => (
            <button key={entry.provider_id} type="button" className="rounded-lg border bg-white p-4 text-left transition hover:border-accent hover:shadow-sm" onClick={() => onChooseCustom(entry)}>
              <span className="block font-semibold">{entry.display_name}</span>
              <span className="mt-1 block text-xs text-muted">Custom provider setup</span>
              <span className="mt-2 block text-xs text-muted">{entry.description}</span>
            </button>
          ))}
          <button type="button" className="rounded-lg border border-dashed bg-white p-4 text-left transition hover:border-accent hover:shadow-sm" onClick={() => onChooseCustom(null)}>
            <span className="block font-semibold">Custom Provider</span>
            <span className="mt-1 block text-xs text-muted">OpenAI-compatible or custom JSON HTTP</span>
            <span className="mt-2 block text-xs text-muted">Bring your own public HTTPS API endpoint.</span>
          </button>
        </div>
        <p className="mt-4 text-sm text-muted">Provider endpoints are contacted by FastAPI through the registry and protocol adapter. The browser never calls a model vendor directly.</p>
      </Panel>
    );
  }
  return (
    <Panel title={effectiveMode === "catalog" ? `Configure ${catalogDisplayNames[selectedPreset?.preset_id ?? ""] ?? selectedPreset?.display_name ?? "Provider"}` : "Add Custom Provider"} description={effectiveMode === "catalog" ? "For the normal setup path, enter the API key and choose Save & Test. Provider identity is generated from the catalog entry." : "Register a public API endpoint that MathModel AI will call through its existing backend protocol adapters."} className="mb-5">
      {create.error ? <div className="mb-4"><ErrorBanner error={create.error} /></div> : null}
      <form className="grid gap-4" onSubmit={(event) => { event.preventDefault(); void submit(); }}>
        {effectiveMode === "custom" ? (
          <div className="grid gap-4 md:grid-cols-2">
            <label className="field">Provider ID<input aria-label="Provider ID" className="input" required pattern="[a-z0-9][a-z0-9-]{1,99}" placeholder="my-company-gateway" value={form.provider_id} onChange={(event) => setForm({ ...form, provider_id: event.target.value })} /><span className="text-xs font-normal text-muted">Stable and immutable after creation. Use lowercase letters, numbers, and hyphens.</span></label>
            <label className="field">Display name<input aria-label="Display name" className="input" required value={form.display_name} onChange={(event) => setForm({ ...form, display_name: event.target.value })} /></label>
            <label className="field">Protocol<select aria-label="Protocol" className="input" value={form.protocol} onChange={(event) => setForm({ ...form, protocol: event.target.value as ProviderInput["protocol"] })}>{Object.entries(protocolLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
            <label className="field">Base URL<input aria-label="Base URL" className="input" required type="url" placeholder="https://api.example.com/v1" value={form.base_url} onChange={(event) => setForm({ ...form, base_url: event.target.value })} /></label>
          </div>
        ) : null}
        {effectiveMode === "catalog" ? (
          <div className="rounded-md border bg-slate-50 px-4 py-3 text-sm">
            <span className="font-semibold">{selectedPreset?.display_name}</span>
            <span className="ml-2 text-muted">{protocolLabels[form.protocol]} · official endpoint</span>
          </div>
        ) : null}
        {!secretStoreAvailable ? <div role="note" className="rounded-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">Encrypted UI credential storage is not configured. Set the secret in the backend environment and use an <code>env:VARIABLE_NAME</code> reference. Direct API-key entry is disabled.</div> : null}
        <label className="field">API Key<input aria-label="New provider API key" className="input" type="password" autoComplete="new-password" disabled={!secretStoreAvailable} value={form.apiKey} onChange={(event) => setForm({ ...form, apiKey: event.target.value })} /><span className="text-xs font-normal text-muted">Write only. After save, only credential status is returned.</span></label>
        <details className="rounded-md border bg-slate-50 p-3">
          <summary className="cursor-pointer text-sm font-semibold">Advanced</summary>
          <div className="mt-3 grid gap-4 md:grid-cols-2">
            {effectiveMode === "catalog" ? <label className="field">Protocol<input className="input" readOnly value={protocolLabels[form.protocol]} /></label> : null}
            {effectiveMode === "catalog" ? <label className="field">Base URL<input className="input" readOnly value={form.base_url} /></label> : null}
            <label className="field">Request timeout (seconds)<input aria-label="Request timeout" className="input" type="number" min="1" max="600" value={form.timeout_seconds} onChange={(event) => setForm({ ...form, timeout_seconds: Number(event.target.value) })} /></label>
            <label className="field">Connect timeout (seconds)<input aria-label="Connect timeout" className="input" type="number" min="1" max="60" value={form.connect_timeout_seconds} onChange={(event) => setForm({ ...form, connect_timeout_seconds: Number(event.target.value) })} /></label>
            {effectiveMode === "custom" ? <label className="field">Endpoint trust<select aria-label="Endpoint trust" className="input" value={form.trust_level} onChange={(event) => setForm({ ...form, trust_level: event.target.value as ProviderInput["trust_level"] })}><option>USER_MANAGED_PROXY</option><option>LOCAL_ENDPOINT</option><option>UNKNOWN</option></select><span className="text-xs font-normal text-muted">Local/private endpoints require server administrator opt-in and are disabled by default.</span></label> : null}
            <label className="field">Credential reference (optional)<input aria-label="Credential reference" className="input" placeholder="env:PROVIDER_API_KEY" value={form.credential_ref} onChange={(event) => setForm({ ...form, credential_ref: event.target.value })} /><span className="text-xs font-normal text-muted">Use for an externally managed environment credential.</span></label>
          </div>
        </details>
        <div className="flex items-center gap-3"><button className="button button-primary" disabled={create.isPending}>{create.isPending ? "Saving & testing…" : "Save & Test"}</button>{effectiveMode === "catalog" ? <button type="button" className="button" onClick={() => setSelectedPresetId("")}>Back to catalog</button> : null}</div>
        <p className="text-xs text-muted">Model discovery intentionally begins after Provider save, so draft credentials are never sent or persisted as a discovery side effect.</p>
      </form>
    </Panel>
  );
}

function generatedModelId(providerId: string, remoteModelId: string, used: Set<string>): string {
  const suffix = remoteModelId.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "model";
  const base = `${providerId}-${suffix}`.slice(0, 100).replace(/-$/, "");
  let candidate = base;
  let counter = 2;
  while (used.has(candidate)) {
    const addition = `-${counter}`;
    candidate = `${base.slice(0, 100 - addition.length)}${addition}`;
    counter += 1;
  }
  used.add(candidate);
  return candidate;
}

function ModelCatalogPanel({ provider, models, onClose, onCreated, onAddManual }: { provider?: Provider; models: Model[]; onClose: () => void; onCreated: () => Promise<void>; onAddManual: () => void }) {
  const [selected, setSelected] = useState<string[]>([]);
  const discovery = useMutation({
    mutationFn: () => api.discoverModels(provider!.provider_id),
    onSuccess: (result) => setSelected(result.models.filter((candidate) => !models.some((model) => model.provider_id === provider?.provider_id && model.remote_model === candidate.remote_model_id)).map((candidate) => candidate.remote_model_id)),
  });
  const create = useLocalOperation();
  if (!provider) return null;
  const providerModels = models.filter((model) => model.provider_id === provider.provider_id);
  const addCandidates = async (candidates: ModelDiscoveryCandidate[]) => {
    const used = new Set(models.map((model) => model.model_id));
    const succeeded = await create.run(async () => {
      for (const candidate of candidates) {
        await api.createModel({
          model_id: generatedModelId(provider.provider_id, candidate.remote_model_id, used),
          provider_id: provider.provider_id,
          display_name: candidate.display_name ?? candidate.remote_model_id,
          remote_model: candidate.remote_model_id,
          declared_capabilities: {},
        });
      }
    });
    if (succeeded) {
      setSelected([]);
      await onCreated().catch(() => undefined);
    }
  };
  const candidates = discovery.data?.models ?? [];
  const discoveryUnavailable = discovery.error instanceof ApiError && discovery.error.code === "MODEL_DISCOVERY_UNSUPPORTED";
  return (
    <Panel title={`Manage ${provider.display_name} Models`} description="One provider can own many models. Fetch Models imports identifiers only; it never counts as capability evidence." className="mb-5">
      {(discovery.error || create.error) ? <div className="mb-4"><ErrorBanner error={discovery.error || create.error} /></div> : null}
      {discoveryUnavailable ? <p role="status" className="mb-4 rounded-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">This provider does not expose model discovery. You can still add a remote model ID manually, then run a capability Probe.</p> : null}
      <div className="mb-4 flex flex-wrap gap-2">
        <button className="button button-primary" disabled={discovery.isPending || !provider.enabled || !provider.credential_configured} onClick={() => discovery.mutate()}>{discovery.isPending ? "Fetching…" : "Fetch Models"}</button>
        <button className="button" onClick={onAddManual}>Add Model</button>
        <button className="button" onClick={onClose}>Close</button>
      </div>
      {!provider.credential_configured ? <p className="mb-4 text-sm text-amber-800">Configure a credential before fetching the remote catalog. Manual model entry remains available.</p> : null}
      {candidates.length ? (
        <div className="space-y-2">
          {candidates.map((candidate) => {
            const existing = providerModels.some((model) => model.remote_model === candidate.remote_model_id);
            return (
              <label key={candidate.remote_model_id} className="flex items-center gap-3 rounded-md border px-3 py-2 text-sm">
                <input type="checkbox" disabled={existing} checked={!existing && selected.includes(candidate.remote_model_id)} onChange={(event) => setSelected((current) => event.target.checked ? [...current, candidate.remote_model_id] : current.filter((item) => item !== candidate.remote_model_id))} />
                <span className="flex-1"><span className="font-semibold">{candidate.display_name ?? candidate.remote_model_id}</span><span className="ml-2 text-muted">{candidate.remote_model_id}</span></span>
                {existing ? <StatusBadge value="ADDED" /> : null}
              </label>
            );
          })}
          <button className="button button-primary mt-2" disabled={!selected.length || create.isPending} onClick={() => void addCandidates(candidates.filter((candidate) => selected.includes(candidate.remote_model_id)))}>{create.isPending ? "Adding…" : `Add Selected Models (${selected.length})`}</button>
        </div>
      ) : discovery.isSuccess ? <EmptyState>No models were reported by this provider. Add the remote model ID manually; capability probing remains available.</EmptyState> : null}
      {providerModels.length ? <p className="mt-4 text-xs text-muted">Registered now: {providerModels.map((model) => model.display_name).join(", ")}</p> : null}
    </Panel>
  );
}

function ProviderCard({ provider, modelCount, secretStoreAvailable, onManage, onChange }: { provider: Provider; modelCount: number; secretStoreAvailable: boolean; onManage: () => void; onChange: () => Promise<void> }) {
  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState({ display_name: provider.display_name, base_url: provider.base_url, timeout_seconds: provider.timeout_seconds });
  const update = useMutation({ mutationFn: () => api.patchProvider(provider.provider_id, form), onSuccess: async () => { setEditing(false); await onChange(); } });
  const toggle = useMutation({ mutationFn: () => api.setProviderEnabled(provider.provider_id, !provider.enabled), onSuccess: onChange });
  const connection = useMutation({ mutationFn: () => api.testProviderConnection(provider.provider_id) });
  return (
    <Panel
      title={provider.display_name}
      description={`${modelCount} registered model${modelCount === 1 ? "" : "s"}`}
      actions={(
        <div className="flex flex-wrap items-center gap-2">
          {provider.provider_id === "mock" ? <StatusBadge value="MOCK / TEST ONLY" /> : null}
          <StatusBadge value={provider.health_status} />
        </div>
      )}
    >
      {(update.error || toggle.error || connection.error) ? <div className="mb-4"><ErrorBanner error={update.error || toggle.error || connection.error} /></div> : null}
      {editing ? (
        <form className="grid gap-3" onSubmit={(event) => { event.preventDefault(); update.mutate(); }}>
          <p className="text-sm text-muted">Provider ID <code>{provider.provider_id}</code> is immutable. Create a new Provider to use another identity.</p>
          <label className="field">Display name<input className="input" required value={form.display_name} onChange={(event) => setForm({ ...form, display_name: event.target.value })} /></label>
          <label className="field">Base URL<input className="input" type="url" required value={form.base_url} onChange={(event) => setForm({ ...form, base_url: event.target.value })} /></label>
          <label className="field">Timeout seconds<input className="input" type="number" min="1" max="600" value={form.timeout_seconds} onChange={(event) => setForm({ ...form, timeout_seconds: Number(event.target.value) })} /></label>
          <div className="flex gap-2"><button className="button button-primary">Save</button><button type="button" className="button" onClick={() => setEditing(false)}>Cancel</button></div>
        </form>
      ) : (
        <>
          <div className="grid gap-3 text-sm sm:grid-cols-2">
            <div className="rounded-md bg-slate-50 px-3 py-2">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">API credential</p>
              <p className={`mt-1 font-semibold ${provider.credential_configured ? "text-emerald-700" : "text-amber-800"}`}>{provider.credential_configured ? "Configured · write only" : "Not configured"}</p>
            </div>
            <div className="rounded-md bg-slate-50 px-3 py-2">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Models</p>
              <p className="mt-1 font-semibold">{modelCount} registered</p>
            </div>
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            <button className="button button-primary" onClick={onManage}>Manage</button>
            <button className="button" disabled={connection.isPending || !provider.enabled || !provider.credential_configured} onClick={() => connection.mutate()}>{connection.isPending ? "Testing…" : "Test Connection"}</button>
          </div>
          {connection.data ? <p role="status" className="mt-2 text-xs text-emerald-700">Connection successful. Credential accepted.{connection.data.model_discovery_supported ? " Model discovery is available." : " Model discovery is unavailable; manual models are supported."}</p> : null}
          <p className="mt-2 text-xs text-muted">Test Connection checks only the endpoint and credential. It does not prove any model capability.</p>
          <details className="mt-4 rounded-md border bg-slate-50 p-3">
            <summary className="cursor-pointer text-sm font-semibold">Advanced provider settings</summary>
            <div className="mt-3">
              <DefinitionList items={[
                ["Provider ID", provider.provider_id],
                ["Protocol", protocolLabels[provider.protocol]],
                ["Endpoint trust", provider.trust_level],
                ["Base URL", <span className="break-all">{provider.base_url}</span>],
                ["Timeout", `${provider.timeout_seconds} seconds`],
                ["Updated", formatDate(provider.updated_at)],
              ]} />
              <div className="mt-4 flex flex-wrap gap-2">
                <button className="button" onClick={() => setEditing(true)}>Edit Provider</button>
                <button className="button" disabled={toggle.isPending} onClick={() => toggle.mutate()}>{provider.enabled ? "Disable" : "Enable"}</button>
              </div>
              <CredentialEditor provider={provider} secretStoreAvailable={secretStoreAvailable} onChange={onChange} />
            </div>
          </details>
        </>
      )}
    </Panel>
  );
}

export function CredentialEditor({ provider, secretStoreAvailable, onChange }: { provider: Provider; secretStoreAvailable: boolean; onChange: () => Promise<void> }) {
  const [apiKey, setApiKey] = useState("");
  const [show, setShow] = useState(false);
  const save = useLocalOperation();
  const remove = useLocalOperation();
  const saveKey = async () => {
    const submittedKey = apiKey;
    setApiKey("");
    setShow(false);
    const succeeded = await save.run(async () => {
      await api.putCredential(provider.provider_id, submittedKey);
    });
    if (succeeded) await onChange().catch(() => undefined);
  };
  const removeKey = async () => {
    const succeeded = await remove.run(async () => {
      await api.deleteCredential(provider.provider_id);
    });
    if (succeeded) await onChange().catch(() => undefined);
  };
  return (
    <div className="mt-4 border-t pt-4">
      {(save.error || remove.error) ? <div className="mb-3"><ErrorBanner error={save.error || remove.error} /></div> : null}
      {!secretStoreAvailable ? <p className="mb-3 text-sm text-amber-800">Credential values are managed outside the Web UI because encrypted storage is unavailable.</p> : null}
      <div className="flex flex-col gap-2 sm:flex-row">
        <label className="field flex-1">
          <span>Replace API Key</span>
          <input
            aria-label={`API key for ${provider.display_name}`}
            className="input"
            type={show ? "text" : "password"}
            autoComplete="new-password"
            disabled={!secretStoreAvailable}
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
          />
        </label>
        <div className="flex items-end gap-2">
          <button type="button" className="button" disabled={!secretStoreAvailable} aria-label={show ? "Hide API key" : "Show API key"} onClick={() => setShow((value) => !value)}>{show ? "Hide" : "Show"}</button>
          <button type="button" className="button button-primary" disabled={!secretStoreAvailable || !apiKey || save.isPending} onClick={() => void saveKey()}>{save.isPending ? "Saving…" : "Replace API Key"}</button>
        </div>
      </div>
      {provider.credential_configured && secretStoreAvailable ? (
        <button
          type="button"
          className="mt-3 text-sm font-semibold text-red-700"
          disabled={remove.isPending}
          onClick={() => { if (window.confirm("Remove this provider credential? Existing probes will become stale.")) void removeKey(); }}
        >
          Remove credential
        </button>
      ) : null}
    </div>
  );
}

function useLocalOperation() {
  const lock = useRef(false);
  const [isPending, setIsPending] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const run = useCallback(async (operation: () => Promise<void>) => {
    if (lock.current) return false;
    lock.current = true;
    setIsPending(true);
    setError(null);
    try {
      await operation();
      return true;
    } catch (caught) {
      setError(caught instanceof Error ? caught : new Error("An unexpected error occurred"));
      return false;
    } finally {
      lock.current = false;
      setIsPending(false);
    }
  }, []);
  return { error, isPending, run };
}

async function createProviderWithReconciliation(input: ProviderInput): Promise<Provider> {
  try {
    return await api.createProvider(input);
  } catch (caught) {
    if (!(caught instanceof ApiError) || caught.status !== 0) throw caught;
    const existing = (await api.providers()).find(
      (provider) =>
        provider.provider_id === input.provider_id
        && provider.display_name === input.display_name
        && provider.protocol === input.protocol
        && provider.base_url === input.base_url
        && provider.trust_level === input.trust_level,
    );
    if (!existing) throw caught;
    return existing;
  }
}

function providerSetupError(caught: unknown, baseUrl: string): Error {
  const error = caught instanceof Error ? caught : new Error("An unexpected error occurred");
  const serverRejectedPrivateDestination = /private or local provider destinations are disabled/i.test(error.message);
  let localHostname = false;
  try {
    const hostname = new URL(baseUrl).hostname.toLowerCase();
    localHostname = hostname === "localhost" || hostname.endsWith(".localhost") || hostname === "::1" || hostname.startsWith("127.");
  } catch { /* Invalid URLs are reported by the backend without local-policy rewriting. */ }
  if (serverRejectedPrivateDestination || (localHostname && /require HTTPS by default/i.test(error.message))) {
    return new Error("Local/private endpoints are disabled by server policy.");
  }
  return error;
}

function ModelCreatePanel({ providers, initialProviderId, onCreated }: { providers: Provider[]; initialProviderId?: string; onCreated: () => Promise<void> }) {
  const [form, setForm] = useState({ model_id: "", provider_id: initialProviderId ?? providers[0]?.provider_id ?? "", display_name: "", remote_model: "", quality_tier: "BALANCED" as ModelInput["quality_tier"], structured_output_strategy: "UNSUPPORTED" as ModelInput["structured_output_strategy"], reasoning_low: "", reasoning_high: "" });
  const [capabilities, setCapabilities] = useState<string[]>(["TEXT"]);
  const create = useMutation({
    mutationFn: () => {
      const { reasoning_low: reasoningLow, reasoning_high: reasoningHigh, ...profile } = form;
      return api.createModel({
        ...profile,
        reasoning_mapping: {
          ...(reasoningLow ? { LOW: reasoningLow } : {}),
          ...(reasoningHigh ? { HIGH: reasoningHigh } : {}),
        },
        declared_capabilities: Object.fromEntries(capabilities.map((name) => [name, { status: "SUPPORTED", source: "USER_DECLARED", detail: "Declared in Web UI" }])),
      });
    },
    onSuccess: onCreated,
  });
  const probe = useMutation({ mutationFn: () => api.probeModel(create.data!.model_id) });
  const addAnother = () => {
    setForm((current) => ({ ...current, model_id: "", display_name: "", remote_model: "", reasoning_low: "", reasoning_high: "" }));
    setCapabilities(["TEXT"]);
    create.reset();
    probe.reset();
  };
  if (create.data) {
    return (
      <Panel title="Model added" description={`${create.data.display_name} is now registered under ${create.data.provider_id}. Add more profiles or continue to routing.`} className="mb-5">
        {probe.error ? <div className="mb-4"><ErrorBanner error={probe.error} /></div> : null}
        {probe.isSuccess ? <p role="status" className="mb-4 rounded-md border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-800">Capability Probe completed. Review the model card for observed evidence.</p> : null}
        <div className="flex flex-wrap gap-2">
          <button className="button" disabled={probe.isPending} onClick={() => probe.mutate()}>{probe.isPending ? "Probing…" : "Run Probe"}</button>
          <button className="button" onClick={addAnother}>Add Another Model</button>
          <Link className="button button-primary" to="/routing">Configure Stage Routing</Link>
        </div>
      </Panel>
    );
  }
  return (
    <Panel title="Add model manually" description="Declare only what you know. Names and provider metadata never become capability proof; Probe remains authoritative." className="mb-5">
      {create.error ? <div className="mb-4"><ErrorBanner error={create.error} /></div> : null}
      <form className="grid gap-4" onSubmit={(event) => { event.preventDefault(); create.mutate(); }}>
        <div className="grid gap-4 md:grid-cols-2">
          <label className="field">Model ID<input className="input" required pattern="[a-z0-9][a-z0-9._-]{1,99}" value={form.model_id} onChange={(event) => setForm({ ...form, model_id: event.target.value })} /></label>
          <label className="field">Provider<select className="input" required value={form.provider_id} onChange={(event) => setForm({ ...form, provider_id: event.target.value })}><option value="" disabled>Select provider</option>{providers.map((provider) => <option key={provider.provider_id} value={provider.provider_id}>{provider.display_name}</option>)}</select></label>
          <label className="field">Display name<input className="input" required value={form.display_name} onChange={(event) => setForm({ ...form, display_name: event.target.value })} /></label>
          <label className="field">Remote model<input className="input" required value={form.remote_model} onChange={(event) => setForm({ ...form, remote_model: event.target.value })} /></label>
          <label className="field">Quality tier (preference, not a minimum)<select className="input" value={form.quality_tier} onChange={(event) => setForm({ ...form, quality_tier: event.target.value as ModelInput["quality_tier"] })}>{["ROUTINE", "BALANCED", "FLAGSHIP_HIGH", "FLAGSHIP_XHIGH", "FLAGSHIP_MAX"].map((value) => <option key={value}>{value}</option>)}</select></label>
          <label className="field">Structured output<select className="input" value={form.structured_output_strategy} onChange={(event) => setForm({ ...form, structured_output_strategy: event.target.value as ModelInput["structured_output_strategy"] })}>{["UNSUPPORTED", "PROMPT_JSON_FALLBACK", "JSON_MODE", "NATIVE_JSON_SCHEMA"].map((value) => <option key={value}>{value}</option>)}</select></label>
          <label className="field">Input modality<select aria-label="Input modality" className="input" value={capabilities.includes("VISION") ? "text-image" : "text"} onChange={(event) => setCapabilities(event.target.value === "text-image" ? Array.from(new Set([...capabilities, "TEXT", "VISION"])) : capabilities.filter((item) => item !== "VISION"))}><option value="text">Text</option><option value="text-image">Text + image (user declared)</option></select></label>
        </div>
        <details className="rounded-md border bg-slate-50 p-3"><summary className="cursor-pointer text-sm font-semibold">Advanced capability declarations and reasoning mapping</summary><div className="mt-3 grid gap-4"><fieldset><legend className="text-sm font-semibold">Declared capabilities</legend><div className="mt-2 flex flex-wrap gap-3">{capabilityNames.map((name) => <label key={name} className="flex items-center gap-2 text-sm"><input type="checkbox" checked={capabilities.includes(name)} onChange={(event) => setCapabilities(event.target.checked ? Array.from(new Set([...capabilities, name])) : capabilities.filter((item) => item !== name))} />{name}</label>)}</div></fieldset><div className="grid gap-4 md:grid-cols-2"><label className="field">Remote value for LOW reasoning<input className="input" value={form.reasoning_low} onChange={(event) => setForm({ ...form, reasoning_low: event.target.value })} /></label><label className="field">Remote value for HIGH reasoning<input className="input" value={form.reasoning_high} onChange={(event) => setForm({ ...form, reasoning_high: event.target.value })} /></label></div></div></details>
        <div><button className="button button-primary" disabled={create.isPending || !providers.length}>{create.isPending ? "Saving…" : "Save model"}</button></div>
      </form>
    </Panel>
  );
}

function ModelCard({ model, provider, onChange }: { model: Model; provider?: Provider; onChange: () => Promise<void> }) {
  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState({ display_name: model.display_name, remote_model: model.remote_model, quality_tier: model.quality_tier, structured_output_strategy: model.structured_output_strategy });
  const [declareJsonSchema, setDeclareJsonSchema] = useState(Boolean(model.declared_capabilities.JSON_SCHEMA));
  const latestProbe = useQuery({ queryKey: ["model-probe", model.model_id, model.config_digest], queryFn: () => api.latestProbe(model.model_id) });
  const update = useMutation({
    mutationFn: () => {
      const declared = { ...model.declared_capabilities };
      if (declareJsonSchema) {
        const evidence = { status: "SUPPORTED" as const, source: "USER_DECLARED" as const, detail: "Declared in Web UI; Probe required" };
        declared.TEXT = evidence;
        declared.STRUCTURED_OUTPUT = evidence;
        declared.JSON_SCHEMA = evidence;
      } else {
        delete declared.STRUCTURED_OUTPUT;
        delete declared.JSON_SCHEMA;
      }
      return api.patchModel(model.model_id, { ...form, declared_capabilities: declared });
    },
    onSuccess: async () => { setEditing(false); await onChange(); },
  });
  const toggle = useMutation({ mutationFn: () => api.patchModel(model.model_id, { enabled: !model.enabled }), onSuccess: onChange });
  const probe = useMutation({ mutationFn: () => api.probeModel(model.model_id), onSuccess: async () => { await onChange(); await latestProbe.refetch(); } });
  const error = latestProbe.error || update.error || toggle.error || probe.error;
  const modelStatus = !model.enabled || provider?.enabled === false
    ? "DISABLED"
    : !provider?.credential_configured
      ? "CREDENTIAL_REQUIRED"
      : !latestProbe.data
        ? "PROBE_REQUIRED"
        : latestProbe.data.authentication_status !== "PASS"
          ? "AUTH_FAILED"
          : "PROBED";
  return (
    <Panel
      title={model.display_name}
      description={model.model_id}
      actions={(
        <div className="flex flex-wrap items-center gap-2">
          {model.provider_id === "mock" ? <StatusBadge value="MOCK / TEST ONLY" /> : null}
          <StatusBadge value={modelStatus} />
        </div>
      )}
    >
      {error ? <div className="mb-4"><ErrorBanner error={error} /></div> : null}
      {editing ? (
        <form className="grid gap-3" onSubmit={(event) => { event.preventDefault(); update.mutate(); }}>
          <label className="field">Display name<input className="input" value={form.display_name} onChange={(event) => setForm({ ...form, display_name: event.target.value })} /></label>
          <label className="field">Remote model<input className="input" value={form.remote_model} onChange={(event) => setForm({ ...form, remote_model: event.target.value })} /></label>
          <label className="field">Quality tier (preference, not a minimum)<select className="input" value={form.quality_tier} onChange={(event) => setForm({ ...form, quality_tier: event.target.value as Model["quality_tier"] })}>{["ROUTINE", "BALANCED", "FLAGSHIP_HIGH", "FLAGSHIP_XHIGH", "FLAGSHIP_MAX"].map((value) => <option key={value}>{value}</option>)}</select></label>
          <label className="field">Structured output strategy<select className="input" value={form.structured_output_strategy} onChange={(event) => setForm({ ...form, structured_output_strategy: event.target.value as Model["structured_output_strategy"] })}>{["UNSUPPORTED", "PROMPT_JSON_FALLBACK", "JSON_MODE", "NATIVE_JSON_SCHEMA"].map((value) => <option key={value}>{value}</option>)}</select></label>
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={declareJsonSchema} onChange={(event) => setDeclareJsonSchema(event.target.checked)} />Declare JSON Schema structured output (Probe required)</label>
          <div className="flex gap-2"><button className="button button-primary">Save</button><button type="button" className="button" onClick={() => setEditing(false)}>Cancel</button></div>
        </form>
      ) : (
        <>
          <DefinitionList items={[["Provider", provider?.display_name ?? model.provider_id], ["Remote model", model.remote_model], ["Tier", model.quality_tier], ["Probe evidence", latestProbe.data ? `Current · ${formatDate(latestProbe.data.performed_at)}` : "Missing or stale"]]} />
          <div className="mt-4 grid grid-cols-2 gap-x-5 gap-y-2 text-sm sm:grid-cols-3">
            {capabilityNames.map((name) => <div key={name} className="flex justify-between gap-2 border-b py-1"><span className="truncate text-xs text-muted">{capabilityDisplayNames[name] ?? name}</span><span className="flex items-center gap-1.5 text-xs"><CapabilityMark evidence={model.effective_capabilities[name]} /><span>{model.effective_capabilities[name]?.status ?? "UNKNOWN"}</span></span></div>)}
          </div>
          {latestProbe.data ? <p className="mt-3 text-xs text-muted">Probe auth: {latestProbe.data.authentication_status} · {latestProbe.data.latency_ms} ms · input {latestProbe.data.usage.input_tokens ?? "?"} / output {latestProbe.data.usage.output_tokens ?? "?"} tokens</p> : null}
          <div className="mt-4 flex flex-wrap gap-2"><button className="button" disabled={probe.isPending || !model.enabled || !provider?.credential_configured} onClick={() => probe.mutate()}>{probe.isPending ? "Probing…" : latestProbe.data ? "Probe Again" : "Run Capability Probe"}</button><button className="button" onClick={() => setEditing(true)}>Edit</button><button className="button" disabled={toggle.isPending} onClick={() => toggle.mutate()}>{model.enabled ? "Disable" : "Enable"}</button></div>
          <p className="mt-2 text-xs text-muted">Capability labels reflect current Probe evidence or conservative declarations. A connection test alone never changes them.</p>
        </>
      )}
    </Panel>
  );
}
