import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { RoutingPage } from "./RoutingPage";
import { model, provider } from "../test/fixtures";
import { installFetch, renderPage, type FetchCall } from "../test/utils";

const alternate = { ...model, model_id: "eligible-model", display_name: "Eligible Model", config_digest: "d".repeat(64) };
const agent = {
  name: "problem_agent",
  role: "structured problem understanding",
  configured_model_id: null,
  task_profile: {
    task_type: "problem_understanding",
    complexity: 0,
    reasoning_requirement: 0,
    math_requirement: 0,
    coding_requirement: 0,
    multimodal_requirement: 0,
    long_context_requirement: 0,
    review_requirement: 0,
    security_risk: 0,
    blast_radius: 0,
    cost_sensitivity: 0,
    deadline_pressure: 0,
    retry_count: 0,
    required_capabilities: [],
    requires_json_schema: false,
    requires_native_tools: false,
    requires_reasoning_control: false,
    minimum_context_tokens: null,
    allowed_endpoint_trust: null,
    preferred_model_id: null,
    allow_model_fallback: true,
    allows_prompt_json_fallback: false,
    minimum_level: null,
  },
};

describe("Agent Routing", () => {
  it("cannot save a rejected selection and saves only after an eligible preview", async () => {
    const calls = installFetch((call: FetchCall) => {
      if (call.method === "GET" && call.path === "/api/v1/model-routing/agent-catalog") return [agent];
      if (call.method === "GET" && call.path === "/api/v1/models") return [model, alternate];
      if (call.method === "GET" && call.path === "/api/v1/providers") return [provider];
      if (call.method === "GET" && call.path === "/api/v1/model-routing/agents") return [];
      if (call.method === "GET" && call.path === "/api/v1/model-routing") return { default_model_id: null, legacy_default_provider: "mock", legacy_default_model: "mock-foundation", legacy_config_deprecated: true, registered_model_count: 2, agent_route_count: 0 };
      if (call.method === "POST" && call.path === "/api/v1/model-routing/preview") {
        const preferred = (call.body as { profile: { preferred_model_id: string } }).profile.preferred_model_id;
        if (preferred === model.model_id) {
          return {
            provider_called: false,
            decision: {
              level: 4,
              action: "HUMAN_REVIEW",
              recommended_model: model.model_id,
              selected_model_id: null,
              rejected_models: { [model.model_id]: ["required capability JSON_SCHEMA is not supported"] },
              fallback_used: false,
              reason: "NO_ELIGIBLE_MODEL",
              task_profile_digest: "e".repeat(64),
              legacy_config_used: false,
            },
          };
        }
        return {
          provider_called: false,
          decision: {
            level: 4,
            action: "EXECUTE",
            recommended_model: alternate.model_id,
            selected_provider: alternate.provider_id,
            selected_model: alternate.remote_model,
            selected_model_id: alternate.model_id,
            rejected_models: {},
            fallback_used: false,
            reason: "selected by capability hard filters",
            task_profile_digest: "f".repeat(64),
            legacy_config_used: false,
          },
        };
      }
      if (call.method === "PUT" && call.path === "/api/v1/model-routing/agents/problem_agent") {
        return { agent_name: "problem_agent", primary_model_id: alternate.model_id, fallback_model_ids: [], updated_at: "2026-08-31T00:00:00Z" };
      }
      throw new Error(`Unhandled ${call.method} ${call.path}`);
    });
    renderPage(<RoutingPage />);
    const selector = await screen.findByLabelText("Primary model for problem_agent");
    await userEvent.selectOptions(selector, model.model_id);
    await userEvent.click(screen.getByRole("button", { name: "Preview route" }));
    expect(await screen.findByText(/Rejected by backend hard filters/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save route" })).toBeDisabled();
    expect(calls.some((call) => call.method === "PUT")).toBe(false);

    await userEvent.selectOptions(selector, alternate.model_id);
    await userEvent.click(screen.getByRole("button", { name: "Preview route" }));
    expect(await screen.findByText(/Eligible ✓/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Save route" }));
    await waitFor(() => expect(calls.some((call) => call.method === "PUT")).toBe(true));
    expect(calls.find((call) => call.method === "PUT")?.body).toEqual({
      primary_model_id: alternate.model_id,
      fallback_model_ids: [],
    });
  });

  it("drops preview authority when backend revalidation rejects a raced save", async () => {
    const calls = installFetch((call: FetchCall) => {
      if (call.method === "GET" && call.path === "/api/v1/model-routing/agent-catalog") return [agent];
      if (call.method === "GET" && call.path === "/api/v1/models") return [alternate];
      if (call.method === "GET" && call.path === "/api/v1/providers") return [provider];
      if (call.method === "GET" && call.path === "/api/v1/model-routing/agents") return [];
      if (call.method === "GET" && call.path === "/api/v1/model-routing") return { default_model_id: null, legacy_default_provider: "mock", legacy_default_model: "mock-foundation", legacy_config_deprecated: true, registered_model_count: 1, agent_route_count: 0 };
      if (call.method === "POST" && call.path === "/api/v1/model-routing/preview") {
        return {
          provider_called: false,
          decision: {
            level: 4,
            action: "EXECUTE",
            recommended_model: alternate.model_id,
            selected_provider: alternate.provider_id,
            selected_model: alternate.remote_model,
            selected_model_id: alternate.model_id,
            rejected_models: {},
            fallback_used: false,
            reason: "selected by capability hard filters",
            task_profile_digest: "f".repeat(64),
            legacy_config_used: false,
          },
        };
      }
      if (call.method === "PUT") {
        return new Response(JSON.stringify({ detail: "route policy rejected model" }), {
          status: 400,
          headers: { "Content-Type": "application/json" },
        });
      }
      throw new Error(`Unhandled ${call.method} ${call.path}`);
    });
    renderPage(<RoutingPage />);
    await userEvent.selectOptions(
      await screen.findByLabelText("Primary model for problem_agent"),
      alternate.model_id,
    );
    await userEvent.click(await screen.findByRole("button", { name: "Preview route" }));
    expect(await screen.findByText(/Eligible ✓/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Save route" }));
    expect(await screen.findByText("route policy rejected model")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("button", { name: "Save route" })).toBeDisabled());
    expect(calls.filter((call) => call.method === "PUT")).toHaveLength(1);
  });

  it("previews and saves primary plus optional fallback models through standard route policy", async () => {
    const calls = installFetch((call: FetchCall) => {
      if (call.method === "GET" && call.path === "/api/v1/model-routing/agent-catalog") return [agent];
      if (call.method === "GET" && call.path === "/api/v1/models") return [model, alternate];
      if (call.method === "GET" && call.path === "/api/v1/providers") return [provider];
      if (call.method === "GET" && call.path === "/api/v1/model-routing/agents") return [];
      if (call.method === "GET" && call.path === "/api/v1/model-routing") return { default_model_id: model.model_id, legacy_default_provider: "mock", legacy_default_model: "mock-foundation", legacy_config_deprecated: true, registered_model_count: 2, agent_route_count: 0 };
      if (call.method === "POST" && call.path === "/api/v1/model-routing/preview") {
        const preferred = (call.body as { profile: { preferred_model_id: string } }).profile.preferred_model_id;
        return { provider_called: false, decision: { level: 4, action: "EXECUTE", recommended_model: preferred, selected_provider: model.provider_id, selected_model: preferred, selected_model_id: preferred, rejected_models: {}, fallback_used: false, reason: "selected by capability hard filters", task_profile_digest: "8".repeat(64), legacy_config_used: false } };
      }
      if (call.method === "PUT") return { agent_name: agent.name, ...(call.body as object), updated_at: "2026-08-31T00:00:00Z" };
      throw new Error(`Unhandled ${call.method} ${call.path}`);
    });

    renderPage(<RoutingPage />);
    await userEvent.selectOptions(await screen.findByLabelText("Primary model for problem_agent"), model.model_id);
    await userEvent.click(screen.getByText(/Advanced · Fallback 1 \/ 2/));
    await userEvent.selectOptions(screen.getByLabelText("Fallback model 1 for problem_agent"), alternate.model_id);
    await userEvent.click(screen.getByRole("button", { name: "Preview route" }));
    await waitFor(() => expect(screen.getAllByText(/Eligible ✓/)).toHaveLength(2));
    await userEvent.click(screen.getByRole("button", { name: "Save route" }));
    await waitFor(() => expect(calls.some((call) => call.method === "PUT")).toBe(true));
    expect(calls.find((call) => call.method === "PUT")?.body).toEqual({
      primary_model_id: model.model_id,
      fallback_model_ids: [alternate.model_id],
    });
  });

  it("persists four distinct Agent mappings across a fresh render while default remains separate", async () => {
    const agentNames = ["problem_agent", "data_agent", "math_modeler", "red_team_agent"];
    const agents = agentNames.map((name) => ({ ...agent, name, role: `${name} role` }));
    const models = agentNames.map((name, index) => ({
      ...model,
      model_id: `model-${index + 1}`,
      display_name: `Model ${index + 1}`,
      config_digest: String(index + 1).repeat(64),
    }));
    let routes = [{ agent_name: "problem_agent", primary_model_id: models[1].model_id, fallback_model_ids: [], updated_at: "2026-08-31T00:00:00Z" }];
    const calls = installFetch((call: FetchCall) => {
      if (call.method === "GET" && call.path === "/api/v1/model-routing/agent-catalog") return agents;
      if (call.method === "GET" && call.path === "/api/v1/models") return models;
      if (call.method === "GET" && call.path === "/api/v1/providers") return [provider];
      if (call.method === "GET" && call.path === "/api/v1/model-routing/agents") return routes;
      if (call.method === "GET" && call.path === "/api/v1/model-routing") return { default_model_id: models[0].model_id, legacy_default_provider: "mock", legacy_default_model: "mock-foundation", legacy_config_deprecated: true, registered_model_count: 4, agent_route_count: routes.length };
      if (call.method === "POST" && call.path === "/api/v1/model-routing/preview") {
        const body = call.body as { agent_name: string; profile: { preferred_model_id: string } };
        return { provider_called: false, decision: { level: 4, action: "EXECUTE", recommended_model: body.profile.preferred_model_id, selected_provider: model.provider_id, selected_model: body.profile.preferred_model_id, selected_model_id: body.profile.preferred_model_id, rejected_models: {}, fallback_used: false, reason: "selected by capability hard filters", task_profile_digest: "9".repeat(64), legacy_config_used: false } };
      }
      if (call.method === "PUT" && call.path.startsWith("/api/v1/model-routing/agents/")) {
        const name = call.path.split("/").at(-1)!;
        const body = call.body as { primary_model_id: string; fallback_model_ids: string[] };
        const saved = { agent_name: name, ...body, updated_at: `2026-08-31T00:00:0${routes.length}Z` };
        routes = [...routes.filter((route) => route.agent_name !== name), saved];
        return saved;
      }
      throw new Error(`Unhandled ${call.method} ${call.path}`);
    });

    const first = renderPage(<RoutingPage />);
    expect(await screen.findByRole("button", { name: "Use Default Model for All Stages" })).toBeEnabled();
    expect(await screen.findByLabelText("Primary model for problem_agent")).toHaveValue(models[1].model_id);
    expect(screen.getAllByText(/DeepSeek Official \/ Model 1/).length).toBeGreaterThan(0);

    for (const [index, name] of agentNames.entries()) {
      const selector = screen.getByLabelText(`Primary model for ${name}`);
      const card = selector.closest("section");
      expect(card).not.toBeNull();
      await userEvent.selectOptions(selector, models[index].model_id);
      await userEvent.click(within(card!).getByRole("button", { name: "Preview route" }));
      await waitFor(() => expect(within(card!).getByText(/Eligible ✓/)).toBeInTheDocument());
      await userEvent.click(within(card!).getByRole("button", { name: "Save route" }));
      await waitFor(() => expect(routes.find((route) => route.agent_name === name)?.primary_model_id).toBe(models[index].model_id));
    }
    expect(calls.filter((call) => call.method === "PUT")).toHaveLength(4);

    first.unmount();
    renderPage(<RoutingPage />);
    for (const [index, name] of agentNames.entries()) {
      expect(await screen.findByLabelText(`Primary model for ${name}`)).toHaveValue(models[index].model_id);
    }
  });
});
