import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "../api/client";
import { ErrorBanner, LoadingState, StatusBadge } from "./ui";

export function IndependentVerificationPanel({ attemptId }: { attemptId: string }) {
  const queryClient = useQueryClient();
  const key = ["independent-verification", attemptId];
  const query = useQuery({ queryKey: key, queryFn: () => api.independentVerification(attemptId) });
  const command = useMutation({
    mutationFn: (scenario?: string) => scenario ? api.replayScenario(attemptId, scenario) : api.recomputeMetrics(attemptId),
    onSuccess: async (data) => {
      queryClient.setQueryData(key, data);
      await queryClient.invalidateQueries({ queryKey: ["benchmark-report"] });
    },
  });
  if (query.isLoading) return <LoadingState label="Loading independent evidence…" />;
  if (query.error) return <ErrorBanner error={query.error} />;
  const view = query.data;
  if (!view) return null;
  return <section className="mt-4 space-y-3 border-t pt-3" aria-label="Independent verification">
    <div className="flex items-center justify-between gap-2"><h3 className="text-sm font-semibold">Independent metrics & replay</h3><StatusBadge value={view.status} /></div>
    <p className="text-xs text-muted">This evidence is an additional gate, not final Phase 5 or benchmark approval.</p>
    {view.blockers?.length ? <ul className="list-disc pl-5 text-sm text-red-800">{view.blockers.map((item) => <li key={item}>{item}</li>)}</ul> : null}
    {command.error ? <ErrorBanner error={command.error} /> : null}
    {view.report ? <>
      <p className="break-all text-xs">Result: {view.report.result_id} · Metrics: {view.report.passed_metrics}/{view.report.required_metrics} · Scenarios: {view.report.passed_scenarios}/{view.report.required_scenarios}</p>
      <div className="overflow-auto"><table className="w-full text-left text-xs"><thead><tr><th>Metric</th><th>Reported</th><th>Recomputed</th><th>Delta</th><th>Status</th></tr></thead><tbody>{view.report.metrics.map((metric) => <tr key={metric.metric_id}><td>{metric.metric_id} v{metric.version}</td><td>{metric.reported ?? "—"}</td><td>{metric.verified ?? "—"}</td><td>{metric.delta ?? "—"}</td><td><StatusBadge value={metric.status} /></td></tr>)}</tbody></table></div>
      {view.report.replays.map((replay) => <p className="break-all text-xs" key={replay.replay_id}>{replay.scenario_id}: {replay.status} · Execution {replay.execution.run_id} · Input SHA {replay.input_digest}</p>)}
    </> : null}
    {view.plan_id ? <div className="flex flex-wrap gap-2"><button className="btn-secondary" disabled={command.isPending} onClick={() => command.mutate(undefined)}>Recompute metrics</button>{view.scenario_ids?.map((scenario) => <button className="btn-secondary" key={scenario} disabled={command.isPending} onClick={() => command.mutate(scenario)}>Replay {scenario}</button>)}</div> : <p className="text-xs text-muted">A reviewed case policy and exact-result plan are required. No automatic thresholds are inferred.</p>}
  </section>;
}
