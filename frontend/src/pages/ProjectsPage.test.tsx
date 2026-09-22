import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { ProjectsPage } from "./ProjectsPage";
import { installFetch, renderPage, type FetchCall } from "../test/utils";

describe("Projects", () => {
  it("renders backend project status and creates through the project endpoint", async () => {
    const calls = installFetch((call: FetchCall) => {
      if (call.method === "GET") return [{ project_id: "11111111-1111-4111-8111-111111111111", name: "Existing", title: "Existing problem", current_stage: "VALIDATE", status: "HUMAN_REVIEW", version: 7, created_at: "2026-08-31T00:00:00Z", updated_at: "2026-08-31T01:00:00Z" }];
      if (call.method === "POST") return { project_id: "22222222-2222-4222-8222-222222222222", problem_id: "33333333-3333-4333-8333-333333333333", title: "New problem", raw_problem: "A sufficiently long competition problem statement.", schema_version: 5, current_stage: "INGEST", status: "PENDING", data_stage: "FILES", version: 0, updated_by: "system", update_reason: "initial" };
      throw new Error("Unhandled request");
    });
    renderPage(<ProjectsPage />);
    expect(await screen.findByText("Existing")).toBeInTheDocument();
    expect(screen.getByText("HUMAN_REVIEW")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Create project" }));
    await userEvent.type(screen.getByLabelText("Name"), "New project");
    await userEvent.type(screen.getByLabelText("Problem title"), "New problem");
    await userEvent.type(screen.getByLabelText("Problem statement"), "A sufficiently long competition problem statement.");
    await userEvent.click(screen.getByRole("button", { name: "Create and open" }));
    expect(calls.some((call) => call.method === "POST" && call.path === "/api/v1/projects")).toBe(true);
  });
});
