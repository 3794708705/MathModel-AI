import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { SystemPage } from "./SystemPage";
import { installFetch, jsonResponse, renderPage, type FetchCall } from "../test/utils";

const systemInfo = {
  name: "MathModel AI",
  version: "0.1.0",
  environment: "local",
  default_provider: "mock",
  configured_providers: ["mock"],
  secret_store_configured: false,
};

function field(label: string) {
  return within(screen.getByText(label, { selector: "dt" }).parentElement!);
}

describe("System", () => {
  it("shows the backend-owned encrypted credential store state", async () => {
    installFetch((call: FetchCall) => {
      if (call.path === "/health/live") return { status: "ok" };
      if (call.path === "/health/ready") return { status: "ready", database: "ready" };
      if (call.path === "/api/v1/system") {
        return {
          name: "MathModel AI",
          version: "0.1.0",
          environment: "local",
          default_provider: "mock",
          configured_providers: ["mock"],
          secret_store_configured: false,
        };
      }
      throw new Error(`Unhandled ${call.method} ${call.path}`);
    });

    renderPage(<SystemPage />);

    expect(await screen.findByText("Encrypted credential store")).toBeInTheDocument();
    expect(await screen.findByText("Not configured")).toBeInTheDocument();
  });

  it("keeps unavailable system facts unknown instead of claiming no credentials", async () => {
    installFetch((call) => {
      if (call.path === "/health/live") return { status: "ok" };
      if (call.path === "/health/ready") return { status: "ready", database: "ready" };
      return jsonResponse({ detail: "internal error" }, 500);
    });
    renderPage(<SystemPage />);
    await screen.findByRole("alert");
    expect(field("Encrypted credential store").getByText("Unknown")).toBeInTheDocument();
    expect(field("Configured SDK providers").getByText("Unknown")).toBeInTheDocument();
    expect(screen.queryByText("Not configured")).not.toBeInTheDocument();
  });

  it("shows a database outage independently of liveness and can recheck recovery", async () => {
    let outage = true;
    const calls = installFetch((call) => {
      if (call.path === "/health/live") return { status: "ok" };
      if (call.path === "/api/v1/system") return systemInfo;
      return outage
        ? jsonResponse({ detail: { code: "DATABASE_UNAVAILABLE", message: "unavailable" } }, 503)
        : { status: "ready", database: "ready" };
    });
    renderPage(<SystemPage />);
    expect(await screen.findByText("UNAVAILABLE")).toBeInTheDocument();
    expect(field("Backend").getByText("ok")).toBeInTheDocument();
    expect(screen.getByText("docker compose up -d postgres")).toBeInTheDocument();
    outage = false;
    await userEvent.click(screen.getByRole("button", { name: "重新检查 / Recheck" }));
    await waitFor(() => expect(field("Database").getByText("ready")).toBeInTheDocument());
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.queryByText("docker compose up -d postgres")).not.toBeInTheDocument();
    for (const path of ["/health/live", "/health/ready", "/api/v1/system"]) {
      expect(calls.filter((call) => call.path === path)).toHaveLength(2);
    }
  });

  it("does not retain stale healthy facts after a failed refresh", async () => {
    let failed = false;
    installFetch((call) => {
      if (failed) return jsonResponse({ detail: "service unavailable" }, 503);
      if (call.path === "/health/live") return { status: "ok" };
      if (call.path === "/health/ready") return { status: "ready", database: "ready" };
      return systemInfo;
    });
    renderPage(<SystemPage />);
    await screen.findByText("Not configured");
    failed = true;
    await userEvent.click(screen.getByRole("button", { name: "重新检查 / Recheck" }));
    await waitFor(() => expect(screen.getAllByRole("alert")).toHaveLength(3));
    expect(field("Backend").getByText("UNKNOWN")).toBeInTheDocument();
    expect(field("Database").getByText("UNKNOWN")).toBeInTheDocument();
    expect(field("Encrypted credential store").getByText("Unknown")).toBeInTheDocument();
    expect(screen.queryByText("UNAVAILABLE")).not.toBeInTheDocument();
  });
});
