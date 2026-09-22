import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { installFetch, renderPage } from "../test/utils";
import { IndependentVerificationPanel } from "./IndependentVerificationPanel";

describe("independent verification evidence", () => {
  it("keeps missing scientific definitions visibly NOT_READY without execution controls", async () => {
    installFetch(() => ({ attempt_id: "attempt", status: "NOT_READY", blockers: ["MISSING_REVIEWED_METRIC_AND_SCENARIO_DEFINITIONS"] }));
    renderPage(<IndependentVerificationPanel attemptId="attempt" />);
    expect(await screen.findByText("NOT_READY")).toBeInTheDocument();
    expect(screen.getByText("MISSING_REVIEWED_METRIC_AND_SCENARIO_DEFINITIONS")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Recompute metrics" })).not.toBeInTheDocument();
  });

  it("shows reported-versus-recomputed values and sends exact-attempt commands", async () => {
    const view = { attempt_id: "attempt", status: "NOT_READY", plan_id: "plan", scenario_ids: ["stress"], blockers: [], report: { result_id: "formal", passed_metrics: 0, required_metrics: 1, passed_scenarios: 0, required_scenarios: 1, replays: [], metrics: [{ metric_id: "rmse", version: "1", reported: 0.1, verified: 0.5, delta: -0.4, status: "FAIL" }] } };
    const calls = installFetch(() => view);
    renderPage(<IndependentVerificationPanel attemptId="attempt" />);
    expect(await screen.findByText("0.5")).toBeInTheDocument();
    expect(screen.getByText("FAIL")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Recompute metrics" }));
    await userEvent.click(screen.getByRole("button", { name: "Replay stress" }));
    expect(calls.filter(c => c.method === "POST").map(c => c.path)).toEqual([
      "/api/v1/benchmarks/attempts/attempt/verification/recompute",
      "/api/v1/benchmarks/attempts/attempt/verification/scenarios/stress/replay",
    ]);
  });
});
