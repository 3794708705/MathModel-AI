import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { api } from "../api/client";
import { IndependentVerificationPanel } from "../components/IndependentVerificationPanel";
import { EmptyState, ErrorBanner, LoadingState, PageHeader, Panel, StatusBadge, formatDate } from "../components/ui";
import type { BenchmarkRunInput } from "../types/contracts";

export function BenchmarkPage() {
  const queryClient = useQueryClient();
  const runs = useQuery({ queryKey: ["benchmark-runs"], queryFn: api.benchmarkRuns });
  const cases = useQuery({ queryKey: ["benchmark-cases"], queryFn: api.benchmarkCases });
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [showRun, setShowRun] = useState(false);
  const activeRunId = selectedRun ?? runs.data?.[0]?.run_id ?? null;
  const report = useQuery({
    queryKey: ["benchmark-report", activeRunId],
    queryFn: () => api.benchmarkReport(activeRunId ?? ""),
    enabled: Boolean(activeRunId),
    refetchInterval: (query) => query.state.data?.run.status === "RUNNING" ? 5_000 : false,
  });
  const firstError = runs.error || cases.error || models.error || report.error;

  if (runs.isLoading || cases.isLoading || models.isLoading) return <LoadingState label="Loading benchmark history…" />;

  return (
    <>
      <PageHeader
        title="Benchmark"
        description="Formal Phase 8 records from the backend. Failed and blocked attempts are intentionally retained."
        actions={<button className="button button-primary" onClick={() => setShowRun((value) => !value)}>Run benchmark</button>}
      />
      {firstError ? <div className="mb-5"><ErrorBanner error={firstError} /></div> : null}
      {showRun ? <BenchmarkRunForm cases={cases.data ?? []} models={models.data ?? []} onDone={async (runId) => { setSelectedRun(runId); setShowRun(false); await queryClient.invalidateQueries({ queryKey: ["benchmark-runs"] }); }} /> : null}
      <div className="grid gap-5 xl:grid-cols-[360px_1fr]">
        <Panel title="Run history" description="No status is hidden.">
          {runs.data?.length ? (
            <div className="space-y-2">
              {runs.data.map((run) => (
                <button key={run.run_id} className={`w-full rounded-md border p-3 text-left ${activeRunId === run.run_id ? "border-accent bg-blue-50" : "bg-white hover:bg-slate-50"}`} onClick={() => setSelectedRun(run.run_id)}>
                  <div className="flex items-center justify-between gap-2"><span className="truncate text-sm font-semibold">{run.config.model}</span><StatusBadge value={run.status} /></div>
                  <p className="mt-2 truncate text-xs text-muted">{run.run_id}</p>
                  <p className="mt-1 text-xs text-muted">{formatDate(run.started_at)}</p>
                </button>
              ))}
            </div>
          ) : <EmptyState>No benchmark runs recorded.</EmptyState>}
        </Panel>
        <div>
          {report.isLoading ? <LoadingState label="Loading selected report…" /> : report.data ? <BenchmarkReportView report={report.data} /> : <EmptyState>Select a run to inspect its attempts.</EmptyState>}
        </div>
      </div>
    </>
  );
}

function BenchmarkRunForm({ cases, models, onDone }: { cases: Awaited<ReturnType<typeof api.benchmarkCases>>; models: Awaited<ReturnType<typeof api.models>>; onDone: (runId: string) => Promise<void> }) {
  const [selectedCases, setSelectedCases] = useState<string[]>(cases.map((item) => item.benchmark_id));
  const [modelId, setModelId] = useState(models[0]?.model_id ?? "");
  const [reasoningTier, setReasoningTier] = useState("high");
  const [pricingVersion, setPricingVersion] = useState("user-entered");
  const [inputPrice, setInputPrice] = useState("0");
  const [outputPrice, setOutputPrice] = useState("0");
  const model = models.find((item) => item.model_id === modelId);
  const mutation = useMutation({
    mutationFn: () => {
      if (!model) throw new Error("Select a model");
      const body: BenchmarkRunInput = {
        case_ids: selectedCases,
        config: {
          provider: model.provider_id,
          model: model.remote_model,
          model_id: model.model_id,
          reasoning_tier: reasoningTier,
          pricing: {
            version: pricingVersion,
            currency: "USD",
            input_per_million: Number(inputPrice),
            output_per_million: Number(outputPrice),
          },
        },
      };
      return api.createBenchmarkRun(body);
    },
    onSuccess: (result) => onDone(result.run.run_id),
  });
  return (
    <Panel title="Run formal benchmark" description="This calls the existing Phase 8 workflow. A missing real provider is recorded as BLOCKED/NOT_READY, never substituted with Mock." className="mb-5">
      {mutation.error ? <div className="mb-4"><ErrorBanner error={mutation.error} /></div> : null}
      <form className="grid gap-4" onSubmit={(event) => { event.preventDefault(); mutation.mutate(); }}>
        <fieldset><legend className="text-sm font-semibold">Cases</legend><div className="mt-2 grid gap-2">{cases.map((item) => <label key={item.benchmark_id} className="flex items-start gap-2 rounded-md border p-3 text-sm"><input className="mt-1" type="checkbox" checked={selectedCases.includes(item.benchmark_id)} onChange={(event) => setSelectedCases(event.target.checked ? [...selectedCases, item.benchmark_id] : selectedCases.filter((id) => id !== item.benchmark_id))} /><span><strong>{item.benchmark_id}</strong> · {item.title}<span className="mt-1 block text-xs text-muted">{item.competition} {item.year} · {item.modeling_category}</span></span></label>)}</div></fieldset>
        <div className="grid gap-4 md:grid-cols-2">
          <label className="field">Model<select className="input" value={modelId} onChange={(event) => setModelId(event.target.value)}><option value="" disabled>Select model</option>{models.map((item) => <option key={item.model_id} value={item.model_id}>{item.display_name}{item.provider_id === "mock" ? " · MOCK / TEST ONLY" : ""}</option>)}</select></label>
          <label className="field">Reasoning tier<input className="input" required value={reasoningTier} onChange={(event) => setReasoningTier(event.target.value)} /></label>
          <label className="field">Pricing version<input className="input" required value={pricingVersion} onChange={(event) => setPricingVersion(event.target.value)} /></label>
          <label className="field">Input USD / 1M tokens<input className="input" type="number" min="0" step="0.000001" required value={inputPrice} onChange={(event) => setInputPrice(event.target.value)} /></label>
          <label className="field">Output USD / 1M tokens<input className="input" type="number" min="0" step="0.000001" required value={outputPrice} onChange={(event) => setOutputPrice(event.target.value)} /></label>
        </div>
        {model?.provider_id === "mock" ? <p className="text-sm font-semibold text-red-700">MOCK / TEST ONLY — formal benchmark acceptance will reject this provider.</p> : null}
        <p className="text-xs text-muted">Pricing becomes benchmark evidence. Enter the real provider price; zero is only correct for a genuinely free model.</p>
        <div><button className="button button-primary" disabled={mutation.isPending || !selectedCases.length || !model}>{mutation.isPending ? "Running…" : "Start benchmark"}</button></div>
      </form>
    </Panel>
  );
}

function BenchmarkReportView({ report }: { report: Awaited<ReturnType<typeof api.benchmarkReport>> }) {
  const failuresByAttempt = useMemo(() => new Map(report.attempts.map((attempt) => [attempt.attempt_id, report.failures.filter((failure) => failure.attempt_id === attempt.attempt_id)])), [report]);
  return (
    <div className="space-y-5">
      <Panel title="Acceptance" actions={<StatusBadge value={report.acceptance.status} />}>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Summary label="Live provider" value={<StatusBadge value={report.acceptance.live_provider_status} />} />
          <Summary label="Live literature" value={<StatusBadge value={report.acceptance.live_literature_status} />} />
          <Summary label="Cases attempted" value={String(report.acceptance.real_cases_attempted)} />
          <Summary label="Successful" value={String(report.acceptance.successful_cases)} />
        </div>
        {report.acceptance.reasons.length ? <ul className="mt-4 list-disc pl-5 text-sm text-muted">{report.acceptance.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul> : null}
      </Panel>
      <Panel title="Attempts" description="Failed retries remain visible.">
        {report.attempts.length ? <div className="space-y-3">{report.attempts.map((attempt) => {
          const result = report.results.find((item) => item.attempt_id === attempt.attempt_id);
          const failures = failuresByAttempt.get(attempt.attempt_id) ?? [];
          return <div key={attempt.attempt_id} className="rounded-md border p-4"><div className="flex flex-wrap items-center justify-between gap-2"><p className="font-semibold">{attempt.benchmark_id} · attempt {attempt.attempt_number}</p><StatusBadge value={attempt.status} /></div><p className="mt-2 text-sm">Score: {result?.score ?? "Not evaluated"} · Runtime: {result?.wall_time_seconds ?? "—"} s · Cost: {result?.estimated_cost ?? "—"}</p>{failures.length ? <ul className="mt-3 space-y-2 text-sm text-red-800">{failures.map((failure) => <li key={failure.failure_id} className="rounded bg-red-50 p-2"><strong>{failure.severity} · {failure.stage}</strong><span className="block">{failure.root_cause}</span></li>)}</ul> : <p className="mt-2 text-sm text-muted">No recorded failures.</p>}<IndependentVerificationPanel attemptId={attempt.attempt_id} /></div>;
        })}</div> : <EmptyState>No attempts recorded for this run.</EmptyState>}
      </Panel>
    </div>
  );
}

function Summary({ label, value }: { label: string; value: React.ReactNode }) {
  return <div className="rounded-md bg-slate-50 p-3"><p className="text-xs font-semibold uppercase text-muted">{label}</p><div className="mt-2 font-semibold">{value}</div></div>;
}
