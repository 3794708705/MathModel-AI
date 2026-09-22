import { useQuery } from "@tanstack/react-query";

import { api } from "../api/client";
import {
  DefinitionList,
  ErrorBanner,
  LoadingState,
  PageHeader,
  Panel,
  StatusBadge,
  formatDate,
} from "../components/ui";

export function DashboardPage() {
  const live = useQuery({ queryKey: ["health", "live"], queryFn: api.live, refetchInterval: 15_000 });
  const ready = useQuery({ queryKey: ["health", "ready"], queryFn: api.ready, refetchInterval: 15_000 });
  const routing = useQuery({ queryKey: ["routing"], queryFn: api.routingOverview });
  const providers = useQuery({ queryKey: ["providers"], queryFn: api.providers });
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.projects });
  const runs = useQuery({ queryKey: ["benchmark-runs"], queryFn: api.benchmarkRuns });
  const firstError = [live, ready, routing, providers, models, projects, runs].find(
    (query) => query.error,
  )?.error;

  if ([live, ready, routing, providers, models, projects, runs].some((query) => query.isLoading)) {
    return <LoadingState label="Loading backend control data…" />;
  }

  return (
    <>
      <PageHeader
        title="Dashboard"
        description="A read-only view of status returned by the backend. This page does not calculate system readiness."
      />
      {firstError ? <div className="mb-5"><ErrorBanner error={firstError} /></div> : null}
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Metric label="Backend" value={<StatusBadge value={live.data?.status} />} />
        <Metric label="Database" value={<StatusBadge value={ready.data?.database} />} />
        <Metric label="Projects" value={String(projects.data?.length ?? 0)} />
        <Metric
          label="Ready providers"
          value={`${providers.data?.filter((provider) => provider.health_status === "READY").length ?? 0} / ${providers.data?.length ?? 0}`}
        />
      </div>

      <div className="mt-5 grid gap-5 xl:grid-cols-2">
        <Panel title="Runtime configuration">
          <DefinitionList
            items={[
              ["Default model", routing.data?.default_model_id ?? "Not configured"],
              ["Registered models", String(models.data?.length ?? 0)],
              ["Configured routes", String(routing.data?.agent_route_count ?? 0)],
              ["Solver availability", "Not exposed by backend API"],
            ]}
          />
        </Panel>
        <Panel title="Latest benchmark">
          {runs.data?.[0] ? (
            <DefinitionList
              items={[
                ["Status", <StatusBadge value={runs.data[0].status} />],
                ["Live provider", <StatusBadge value={runs.data[0].live_provider_status} />],
                ["Model", runs.data[0].config.model],
                ["Started", formatDate(runs.data[0].started_at)],
              ]}
            />
          ) : (
            <p className="text-sm text-muted">No benchmark history returned by the backend.</p>
          )}
        </Panel>
      </div>
    </>
  );
}

function Metric({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-lg border bg-white p-5 shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted">{label}</p>
      <div className="mt-3 text-2xl font-semibold">{value}</div>
    </div>
  );
}
