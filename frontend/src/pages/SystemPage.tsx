import { useQuery } from "@tanstack/react-query";

import { ApiError, api, apiBaseUrl } from "../api/client";
import { DefinitionList, ErrorBanner, PageHeader, Panel, StatusBadge } from "../components/ui";

export function SystemPage() {
  const system = useQuery({ queryKey: ["system"], queryFn: api.system });
  const live = useQuery({ queryKey: ["health", "live"], queryFn: api.live, refetchInterval: 15_000 });
  const ready = useQuery({ queryKey: ["health", "ready"], queryFn: api.ready, refetchInterval: 15_000 });
  const queries = [system, live, ready];
  const refreshing = queries.some((query) => query.isFetching);
  const currentSystem = system.isError ? undefined : system.data;
  const databaseUnavailable = ready.error instanceof ApiError
    && ready.error.code === "DATABASE_UNAVAILABLE";
  const databaseStatus = databaseUnavailable ? "UNAVAILABLE"
    : ready.isError ? "UNKNOWN" : ready.data?.database ?? "CHECKING";

  return (
    <>
      <PageHeader
        title="System"
        description="Runtime facts exposed by FastAPI. Unexposed infrastructure fields remain explicitly unknown."
        actions={
          <button
            className="button"
            disabled={refreshing}
            onClick={() => { void Promise.all(queries.map((query) => query.refetch())); }}
          >
            {refreshing ? "正在检查…" : "重新检查 / Recheck"}
          </button>
        }
      />
      {queries.map((query, index) => query.error ? (
        <div key={index} className="mb-5"><ErrorBanner error={query.error} /></div>
      ) : null)}
      <Panel title="Backend status">
        <DefinitionList
          items={[
            ["Backend", <StatusBadge value={live.isError ? "UNKNOWN" : live.data?.status ?? "CHECKING"} />],
            ["Database", <StatusBadge value={databaseStatus} />],
            ["Application", currentSystem?.name ?? "Unknown"],
            ["Version", currentSystem?.version ?? "Unknown"],
            ["Environment", currentSystem?.environment ?? "Unknown"],
            ["API base", apiBaseUrl],
            ["Legacy provider", currentSystem?.default_provider ?? "Unknown"],
            ["Configured SDK providers", currentSystem ? currentSystem.configured_providers.join(", ") || "None" : "Unknown"],
            ["Encrypted credential store", currentSystem ? currentSystem.secret_store_configured ? "Configured" : "Not configured" : "Unknown"],
            ["Docker / Sandbox", "Not exposed by backend API"],
            ["Solver", "Not exposed by backend API"],
            ["PDF compiler", "Not exposed by backend API"],
            ["Migration head", "Not exposed by backend API"],
          ]}
        />
      </Panel>
      {databaseUnavailable ? (
        <Panel title="数据库排查 / Database troubleshooting" className="mt-5">
          <p className="text-sm">若使用项目自带的本地 PostgreSQL，可在项目根目录执行：</p>
          <pre className="mt-3 overflow-x-auto rounded bg-slate-50 p-3 text-sm">docker compose up -d postgres</pre>
          <p className="mt-3 text-sm">远程或自定义数据库请检查服务、网络和后端连接配置，恢复后点击“重新检查”。不要在截图或反馈中暴露数据库密码。</p>
        </Panel>
      ) : null}
    </>
  );
}
