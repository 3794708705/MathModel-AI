import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SystemPage } from "./SystemPage";
import { installFetch, renderPage, type FetchCall } from "../test/utils";

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
    expect(screen.getByText("Not configured")).toBeInTheDocument();
  });
});
