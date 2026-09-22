import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it } from "vitest";

import { ProjectWorkspacePage } from "./ProjectWorkspacePage";
import { installFetch, jsonResponse, type FetchCall } from "../test/utils";

describe("Project workspace", () => {
  it("calls one backend workflow endpoint and never orchestrates Agents in the browser", async () => {
    const projectId = "11111111-1111-4111-8111-111111111111";
    const calls = installFetch((call: FetchCall) => {
      if (call.method === "GET" && call.path === `/api/v1/projects/${projectId}`) {
        return {
          project_id: projectId,
          problem_id: "22222222-2222-4222-8222-222222222222",
          title: "Workspace problem",
          raw_problem: "This problem asks for an auditable allocation model.",
          schema_version: 5,
          current_stage: "INGEST",
          status: "PENDING",
          data_stage: "FILES",
          version: 0,
          updated_by: "system",
          update_reason: "initial",
          registered_files: [],
          verified_result_id: null,
        };
      }
      if (call.method === "GET" && (call.path.endsWith("/paper") || call.path.endsWith("/submission"))) {
        return jsonResponse({ detail: "not available" }, 404);
      }
      if (call.method === "POST" && call.path.endsWith("/reasoning/run")) return { state: {}, agent_runs: [], is_mock: true };
      throw new Error(`Unexpected browser orchestration call: ${call.method} ${call.path}`);
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={[`/projects/${projectId}`]}>
          <Routes><Route path="/projects/:projectId" element={<ProjectWorkspacePage />} /></Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    );
    expect(await screen.findByText("Workspace problem")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Run reasoning" }));
    await waitFor(() => expect(calls.some((call) => call.method === "POST")).toBe(true));
    const postCalls = calls.filter((call) => call.method === "POST");
    expect(postCalls.map((call) => call.path)).toEqual([`/api/v1/projects/${projectId}/reasoning/run`]);
  });
});
