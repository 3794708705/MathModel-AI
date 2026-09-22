import { useQuery } from "@tanstack/react-query";

import { api, apiBaseUrl } from "../api/client";
import { DefinitionList, ErrorBanner, LoadingState, PageHeader, Panel, StatusBadge } from "../components/ui";

export function SystemPage() {
  const system = useQuery({ queryKey: ["system"], queryFn: api.system });
  const live = useQuery({ queryKey: ["health", "live"], queryFn: api.live, refetchInterval: 15_000 });
  const ready = useQuery({ queryKey: ["health", "ready"], queryFn: api.ready, refetchInterval: 15_000 });
  const error = system.error || live.error || ready.error;

  if (system.isLoading || live.isLoading || ready.isLoading) return <LoadingState />;

  return (
    <>
      <PageHeader
        title="System"
        description="Runtime facts exposed by FastAPI. Unexposed infrastructure fields remain explicitly unknown."
      />
      {error ? <div className="mb-5"><ErrorBanner error={error} /></div> : null}
      <Panel title="Backend status">
        <DefinitionList
          items={[
            ["Backend", <StatusBadge value={live.data?.status} />],
            ["Database", <StatusBadge value={ready.data?.database} />],
            ["Application", system.data?.name ?? "—"],
            ["Version", system.data?.version ?? "—"],
            ["Environment", system.data?.environment ?? "—"],
            ["API base", apiBaseUrl],
            ["Legacy provider", system.data?.default_provider ?? "—"],
            ["Configured SDK providers", system.data?.configured_providers.join(", ") || "None"],
            ["Encrypted credential store", system.data?.secret_store_configured ? "Configured" : "Not configured"],
            ["Docker / Sandbox", "Not exposed by backend API"],
            ["Solver", "Not exposed by backend API"],
            ["PDF compiler", "Not exposed by backend API"],
            ["Migration head", "Not exposed by backend API"],
          ]}
        />
      </Panel>
    </>
  );
}
