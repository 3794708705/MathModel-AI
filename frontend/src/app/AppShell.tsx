import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { NavLink, Outlet } from "react-router-dom";

import { api } from "../api/client";

const navigation = [
  ["Dashboard", "", "/"],
  ["Projects", "", "/projects"],
  ["Models & API", "Provider / model registry", "/models"],
  ["Agent Routing", "阶段模型分配", "/routing"],
  ["Benchmark", "", "/benchmark"],
  ["System", "", "/system"],
] as const;

export function AppShell() {
  const queryClient = useQueryClient();
  const backendWasOffline = useRef(false);
  const backendLive = useQuery({
    queryKey: ["backend-live"],
    queryFn: api.live,
    retry: false,
    refetchInterval: 10_000,
  });
  useEffect(() => {
    if (backendLive.isError) {
      backendWasOffline.current = true;
      return;
    }
    if (!backendLive.isSuccess || !backendWasOffline.current) return;
    backendWasOffline.current = false;
    void queryClient.refetchQueries({
      type: "active",
      predicate: (query) => query.queryKey[0] !== "backend-live",
    });
  }, [backendLive.isError, backendLive.isSuccess, queryClient]);
  const connected = backendLive.isSuccess;
  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[240px_1fr]">
      <aside className="border-b bg-slate-950 text-white lg:min-h-screen lg:border-b-0 lg:border-r lg:border-slate-800">
        <div className="flex h-16 items-center border-b border-slate-800 px-5">
          <div>
            <p className="text-sm font-semibold tracking-wide">MathModel AI</p>
            <p className="text-xs text-slate-400">Control Center</p>
          </div>
        </div>
        <nav aria-label="Main navigation" className="flex gap-1 overflow-x-auto p-3 lg:flex-col">
          {navigation.map(([label, description, path]) => (
            <NavLink
              key={path}
              to={path}
              end={path === "/"}
              className={({ isActive }) =>
                `whitespace-nowrap rounded-md px-3 py-2 text-sm font-medium ${
                  isActive ? "bg-slate-800 text-white" : "text-slate-300 hover:bg-slate-900"
                }`
              }
            >
              <span className="block">{label}</span>
              {description ? <span className="mt-0.5 block text-[11px] font-normal text-slate-400">{description}</span> : null}
            </NavLink>
          ))}
        </nav>
        <div className="mx-3 mb-3 mt-1 rounded-md border border-slate-800 bg-slate-900 px-3 py-2 text-xs">
          <span
            className={`mr-2 inline-block h-2 w-2 rounded-full ${
              backendLive.isPending ? "bg-slate-400" : connected ? "bg-emerald-400" : "bg-red-400"
            }`}
            aria-hidden="true"
          />
          <span>{backendLive.isPending ? "Backend checking" : connected ? "Backend connected" : "Backend offline"}</span>
        </div>
      </aside>
      <main className="min-w-0 p-4 sm:p-6 lg:p-8">
        <div className="mx-auto max-w-7xl">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
