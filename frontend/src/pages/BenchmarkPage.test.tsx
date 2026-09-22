import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { BenchmarkPage } from "./BenchmarkPage";
import { model } from "../test/fixtures";
import { installFetch, renderPage, type FetchCall } from "../test/utils";

describe("Benchmark", () => {
  it("keeps FAIL and BLOCKED attempts and their failure evidence visible", async () => {
    const runId = "11111111-1111-4111-8111-111111111111";
    installFetch((call: FetchCall) => {
      if (call.path === "/api/v1/benchmarks/runs") return [{ run_id: runId, code_commit: "a".repeat(40), source_tree_digest: "b".repeat(64), working_tree_dirty: true, config: { provider: "deepseek", model: "deepseek-chat", reasoning_tier: "high", temperature: 0, random_seed: 1, allow_live_literature: true, enable_independent_evaluator: true, deadline_modes: ["NORMAL"], pricing: { version: "test", currency: "USD", input_per_million: 1, cached_input_per_million: 0, output_per_million: 2 }, budget: {} }, case_manifest_digests: { one: "c".repeat(64) }, competition_profile_digests: {}, status: "NOT_READY", live_provider_status: "BLOCKED", live_literature_status: "BLOCKED", started_at: "2026-08-31T00:00:00Z", finished_at: "2026-08-31T00:01:00Z", run_digest: "d".repeat(64) }];
      if (call.path === "/api/v1/benchmarks/cases") return [];
      if (call.path === "/api/v1/models") return [];
      if (call.path.endsWith("/report")) return benchmarkReport(runId);
      throw new Error(`Unhandled ${call.method} ${call.path}`);
    });
    renderPage(<BenchmarkPage />);
    expect(await screen.findByText("FAIL")).toBeInTheDocument();
    expect(screen.getByText("BLOCKED_ENVIRONMENT")).toBeInTheDocument();
    expect(screen.getByText("P0 · LIVE_PROVIDER")).toBeInTheDocument();
    expect(screen.getByText("P1 · SOLVER")).toBeInTheDocument();
  });

  it("renders a formal benchmark rejection as NOT_READY without hiding evidence", async () => {
    const runId = "11111111-1111-4111-8111-111111111111";
    const report = benchmarkReport(runId);
    const calls = installFetch((call: FetchCall) => {
      if (call.method === "GET" && call.path === "/api/v1/benchmarks/runs") return [];
      if (call.method === "GET" && call.path === "/api/v1/benchmarks/cases") {
        return [{ benchmark_id: "BENCH-case", title: "Controlled case", competition: "Audit", year: 2026, modeling_category: "optimization", difficulty: "medium", requires_literature: false, requires_solver: true }];
      }
      if (call.method === "GET" && call.path === "/api/v1/models") return [model];
      if (call.method === "POST" && call.path === "/api/v1/benchmarks/runs") return report;
      if (call.method === "GET" && call.path.endsWith(`/runs/${runId}/report`)) return report;
      throw new Error(`Unhandled ${call.method} ${call.path}`);
    });
    renderPage(<BenchmarkPage />);
    await screen.findByText("No benchmark runs recorded.");
    await userEvent.click(screen.getByRole("button", { name: "Run benchmark" }));
    await userEvent.selectOptions(screen.getByLabelText("Model"), model.model_id);
    await userEvent.click(screen.getByRole("button", { name: "Start benchmark" }));
    expect(await screen.findByText("NOT_READY")).toBeInTheDocument();
    expect(screen.getAllByText("BLOCKED").length).toBeGreaterThan(0);
    expect(screen.getByText("real provider not configured")).toBeInTheDocument();
    await waitFor(() => expect(calls.filter((call) => call.method === "POST")).toHaveLength(1));
  });

  it("marks Mock models as test-only before a formal benchmark can run", async () => {
    const mockModel = { ...model, model_id: "mock-audit", provider_id: "mock", display_name: "Mock audit model" };
    installFetch((call: FetchCall) => {
      if (call.method === "GET" && call.path === "/api/v1/benchmarks/runs") return [];
      if (call.method === "GET" && call.path === "/api/v1/benchmarks/cases") return [];
      if (call.method === "GET" && call.path === "/api/v1/models") return [mockModel];
      throw new Error(`Unhandled ${call.method} ${call.path}`);
    });
    renderPage(<BenchmarkPage />);
    await screen.findByText("No benchmark runs recorded.");
    await userEvent.click(screen.getByRole("button", { name: "Run benchmark" }));
    expect(screen.getByRole("option", { name: "Mock audit model · MOCK / TEST ONLY" })).toBeInTheDocument();
    expect(screen.getByText("MOCK / TEST ONLY — formal benchmark acceptance will reject this provider.")).toBeInTheDocument();
  });
});

function benchmarkReport(runId: string) {
  const attempt = (id: string, number: number, status: string) => ({ attempt_id: id, run_id: runId, benchmark_id: "BENCH-case", manifest_digest: "e".repeat(64), attempt_number: number, status, project_id: null, official: true, solve_input_digest: "f".repeat(64), provider_is_live: false, literature_is_live: false, started_at: "2026-08-31T00:00:00Z", finished_at: "2026-08-31T00:00:30Z", attempt_digest: "1".repeat(64) });
  const first = attempt("22222222-2222-4222-8222-222222222222", 1, "FAIL");
  const second = attempt("33333333-3333-4333-8333-333333333333", 2, "BLOCKED_ENVIRONMENT");
  const failure = (id: string, attemptId: string, severity: string, stage: string) => ({ failure_id: id, attempt_id: attemptId, stage, category: "PROVIDER", severity, root_cause: `${stage} failed with recorded evidence`, evidence_refs: ["evidence"], reproducible: true, generic_issue: true, proposed_fix: "configure environment", created_at: "2026-08-31T00:00:00Z", failure_digest: "2".repeat(64) });
  return {
    run: { run_id: runId, config: { model: "deepseek-chat" }, status: "NOT_READY" },
    attempts: [first, second],
    results: [],
    metrics: [],
    failures: [failure("44444444-4444-4444-8444-444444444444", first.attempt_id, "P0", "LIVE_PROVIDER"), failure("55555555-5555-4555-8555-555555555555", second.attempt_id, "P1", "SOLVER")],
    human_interventions: [],
    acceptance: { status: "NOT_READY", real_cases_attempted: 2, successful_cases: 0, verified_real_profiles: 0, live_provider_status: "BLOCKED", live_literature_status: "BLOCKED", hidden_p0_count: 0, reasons: ["real provider not configured"] },
    report_digest: "3".repeat(64),
  };
}
