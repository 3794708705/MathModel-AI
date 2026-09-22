import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api } from "../api/client";
import { EmptyState, ErrorBanner, LoadingState, PageHeader, Panel, StatusBadge } from "../components/ui";
import type { AgentRoute, Model, Provider, RoutableAgent, RoutingPreview } from "../types/contracts";

const agentLabels: Record<string, string> = {
  problem_agent: "题目理解",
  data_agent: "数据理解",
  literature_agent: "文献规划",
  model_explorer: "模型探索",
  model_jury: "模型评审",
  math_modeler: "数学建模",
  code_agent: "代码生成",
  citation_agent: "引用验证",
  paper_agent: "论文写作",
  paper_factual_audit_agent: "论文事实审计",
  red_team_agent: "红队审查",
  model_repair_agent: "模型修复",
  final_jury_agent: "最终评审",
};

function routeIsEligible(preview: RoutingPreview, modelId: string): boolean {
  return preview.decision.action === "EXECUTE" && preview.decision.selected_model_id === modelId;
}

function modelOptionLabel(model: Model, providers: Provider[]): string {
  const provider = providers.find((item) => item.provider_id === model.provider_id);
  return `${provider?.display_name ?? model.provider_id} / ${model.display_name}${model.enabled ? "" : " · Model unavailable"}`;
}

export function RoutingPage() {
  const queryClient = useQueryClient();
  const agents = useQuery({ queryKey: ["routable-agents"], queryFn: api.routableAgents });
  const providers = useQuery({ queryKey: ["providers"], queryFn: api.providers });
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const routes = useQuery({ queryKey: ["agent-routes"], queryFn: api.agentRoutes });
  const routing = useQuery({ queryKey: ["routing"], queryFn: api.routingOverview });
  const applyDefault = useMutation({
    mutationFn: async () => {
      const defaultModelId = routing.data?.default_model_id;
      if (!defaultModelId) throw new Error("Choose a default model on Models & API first.");
      const previews = await Promise.all(
        (agents.data ?? []).map(async (agent) => ({
          agent,
          preview: await api.previewRoute(agent, defaultModelId),
        })),
      );
      const rejected = previews.filter(({ preview }) => !routeIsEligible(preview, defaultModelId));
      if (rejected.length) {
        throw new Error(
          `Default model failed backend hard filters for: ${rejected.map(({ agent }) => agent.name).join(", ")}. No routes were changed.`,
        );
      }
      await Promise.all(
        previews.map(({ agent }) => api.saveAgentRoute(agent.name, {
          primary_model_id: defaultModelId,
          fallback_model_ids: [],
        })),
      );
    },
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["agent-routes"] }),
        queryClient.invalidateQueries({ queryKey: ["routable-agents"] }),
        queryClient.invalidateQueries({ queryKey: ["routing"] }),
      ]);
    },
  });
  const error = agents.error || providers.error || models.error || routes.error || routing.error;

  if (agents.isLoading || providers.isLoading || models.isLoading || routes.isLoading || routing.isLoading) {
    return <LoadingState label="Loading Router policies…" />;
  }

  const defaultModel = models.data?.find((model) => model.model_id === routing.data?.default_model_id);
  return (
    <>
      <PageHeader
        title="Agent Routing / 阶段模型分配"
        description="Choose a primary model for every backend-defined Agent stage. Advanced fallbacks are optional; Preview explains eligibility and Save always revalidates on the backend."
        actions={<button className="button button-primary" disabled={!defaultModel || applyDefault.isPending || !agents.data?.length} onClick={() => applyDefault.mutate()}>{applyDefault.isPending ? "Applying…" : "Use Default Model for All Stages"}</button>}
      />
      {error || applyDefault.error ? <div className="mb-5"><ErrorBanner error={error || applyDefault.error} /></div> : null}
      <Panel title="Quick Setup" description="Use the default model for all stages as a starting point, then customize individual Agents below. Default is a preference for the next eligible request, not a one-model restriction." className="mb-5">
        <div className="grid gap-3 text-sm sm:grid-cols-3">
          <div><span className="font-semibold">Default fallback</span><p className="text-muted">{defaultModel ? modelOptionLabel(defaultModel, providers.data ?? []) : routing.data?.default_model_id ? "Model unavailable · Select another model" : "Not configured"}</p></div>
          <div><span className="font-semibold">Backend Agent catalog</span><p className="text-muted">{agents.data?.length ?? 0} routable stages</p></div>
          <div><span className="font-semibold">Explicit routes</span><p className="text-muted">{routes.data?.length ?? 0} configured policies</p></div>
        </div>
      </Panel>
      {agents.data?.length ? (
        <div className="space-y-4">
          {agents.data.map((agent) => (
            <AgentRouteEditor
              key={`${agent.name}:${routes.data?.find((route) => route.agent_name === agent.name)?.updated_at ?? "default"}`}
              agent={agent}
              models={models.data ?? []}
              providers={providers.data ?? []}
              route={routes.data?.find((route) => route.agent_name === agent.name)}
            />
          ))}
        </div>
      ) : <EmptyState>No routable Agents were returned by the backend.</EmptyState>}
    </>
  );
}

function AgentRouteEditor({ agent, models, providers, route }: { agent: RoutableAgent; models: Model[]; providers: Provider[]; route?: AgentRoute }) {
  const queryClient = useQueryClient();
  const initial = route?.primary_model_id ?? agent.configured_model_id ?? "";
  const [selected, setSelected] = useState(initial);
  const [fallbacks, setFallbacks] = useState<string[]>(() => {
    const configured = route?.fallback_model_ids ?? [];
    if (configured.length === 0) return ["", ""];
    if (configured.length === 1) return [configured[0] ?? "", ""];
    return configured;
  });
  const [previewResults, setPreviewResults] = useState<Record<string, RoutingPreview>>({});
  const selectedFallbacks = fallbacks.filter(Boolean);
  const candidates = [selected, ...selectedFallbacks].filter(Boolean);
  const preview = useMutation({
    mutationFn: async () => Object.fromEntries(
      await Promise.all(candidates.map(async (modelId) => [modelId, await api.previewRoute(agent, modelId)] as const)),
    ),
    onSuccess: setPreviewResults,
  });
  const eligible = candidates.length > 0
    && candidates.every((modelId) => previewResults[modelId] && routeIsEligible(previewResults[modelId], modelId));
  const save = useMutation({
    mutationFn: () => api.saveAgentRoute(agent.name, {
      primary_model_id: selected,
      fallback_model_ids: selectedFallbacks,
    }),
    onError: () => setPreviewResults({}),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["agent-routes"] }),
        queryClient.invalidateQueries({ queryKey: ["routable-agents"] }),
        queryClient.invalidateQueries({ queryKey: ["routing"] }),
      ]);
    },
  });
  const resetPreview = () => setPreviewResults({});
  const updateFallback = (index: number, value: string) => {
    setFallbacks((current) => current.map((item, itemIndex) => itemIndex === index ? value : item));
    resetPreview();
  };
  return (
    <Panel title={agentLabels[agent.name] ?? agent.name} description={`${agent.role} · Backend Agent: ${agent.name}`} actions={route ? <StatusBadge value="CONFIGURED" /> : <StatusBadge value="DEFAULT" />}>
      {(preview.error || save.error) ? <div className="mb-4"><ErrorBanner error={preview.error || save.error} /></div> : null}
      <div className="grid items-end gap-3 lg:grid-cols-[minmax(240px,1fr)_auto_auto]">
        <label className="field">Primary model<select aria-label={`Primary model for ${agent.name}`} className="input" value={selected} onChange={(event) => { setSelected(event.target.value); resetPreview(); }}><option value="" disabled>Select model</option>{models.map((model) => <option key={model.model_id} value={model.model_id} disabled={!model.enabled}>{modelOptionLabel(model, providers)}</option>)}</select></label>
        <button className="button" disabled={!selected || preview.isPending} onClick={() => preview.mutate()}>{preview.isPending ? "Checking…" : "Preview route"}</button>
        <button className="button button-primary" disabled={!eligible || save.isPending} onClick={() => save.mutate()}>{save.isPending ? "Saving…" : "Save route"}</button>
      </div>
      <details className="mt-4 rounded-md border bg-slate-50 p-3">
        <summary className="cursor-pointer text-sm font-semibold">Advanced · Fallback 1 / 2 ({selectedFallbacks.length} configured)</summary>
        <div className="mt-3 grid gap-3 md:grid-cols-2">
          {fallbacks.map((fallback, index) => (
            <label className="field" key={index}>Fallback model {index + 1}<select aria-label={`Fallback model ${index + 1} for ${agent.name}`} className="input" value={fallback} onChange={(event) => updateFallback(index, event.target.value)}><option value="">None</option>{models.map((model) => <option key={model.model_id} value={model.model_id} disabled={!model.enabled || model.model_id === selected || fallbacks.some((item, itemIndex) => itemIndex !== index && item === model.model_id)}>{modelOptionLabel(model, providers)}</option>)}</select></label>
          ))}
        </div>
        <p className="mt-3 text-xs text-muted">Fallback order is preserved. The backend rechecks every selected model when this route is saved and again before execution.</p>
      </details>
      {Object.keys(previewResults).length ? (
        <div className="mt-4 space-y-2">
          {candidates.map((modelId, index) => {
            const result = previewResults[modelId];
            const modelEligible = result && routeIsEligible(result, modelId);
            return result ? (
              <div key={modelId} className={`rounded-md border p-3 text-sm ${modelEligible ? "border-emerald-200 bg-emerald-50" : "border-amber-200 bg-amber-50"}`}>
                <p className="font-semibold">{index === 0 ? "Primary" : `Fallback ${index}`} · {modelId}: {modelEligible ? "Eligible ✓" : "Rejected by backend hard filters"}</p>
                <p className="mt-1">{result.decision.reason}</p>
                {Object.keys(result.decision.rejected_models ?? {}).length ? <ul className="mt-2 list-disc pl-5">{Object.entries(result.decision.rejected_models).map(([rejectedId, reasons]) => <li key={rejectedId}>{rejectedId}: {reasons.join("; ")}</li>)}</ul> : null}
                <p className="mt-2 text-xs">Provider called during preview: {result.provider_called ? "yes" : "no"}</p>
              </div>
            ) : null;
          })}
        </div>
      ) : null}
    </Panel>
  );
}
