import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ModelsApiPage } from "./ModelsApiPage";
import { capability, model, probe, provider } from "../test/fixtures";
import { installFetch, jsonResponse, renderPage, type FetchCall } from "../test/utils";

const catalogPresets = [
  {
    preset_id: "deepseek_official",
    display_name: "DeepSeek Official API",
    protocol: "openai_chat_completions",
    base_url: "https://api.deepseek.com",
    recommended_credential_ref: "env:DEEPSEEK_API_KEY",
    trust_level: "OFFICIAL_VENDOR",
    credential_type: "API_KEY",
    model_hints: ["deepseek-v4-flash", "deepseek-v4-pro"],
    capability_hints: {},
  },
  {
    preset_id: "qwen_modelstudio",
    display_name: "Qwen Model Studio",
    protocol: "openai_chat_completions",
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    recommended_credential_ref: "env:DASHSCOPE_API_KEY",
    trust_level: "OFFICIAL_VENDOR",
    credential_type: "API_KEY",
    model_hints: ["qwen-plus"],
    capability_hints: {},
  },
  {
    preset_id: "openai_official",
    display_name: "OpenAI",
    protocol: "openai_responses",
    base_url: "https://api.openai.com/v1",
    recommended_credential_ref: "env:OPENAI_API_KEY",
    trust_level: "OFFICIAL_VENDOR",
    credential_type: "API_KEY",
    model_hints: ["gpt-5"],
    capability_hints: {},
  },
  {
    preset_id: "anthropic_official",
    display_name: "Anthropic",
    protocol: "anthropic_messages",
    base_url: "https://api.anthropic.com/v1",
    recommended_credential_ref: "env:ANTHROPIC_API_KEY",
    trust_level: "OFFICIAL_VENDOR",
    credential_type: "API_KEY",
    model_hints: ["claude-sonnet"],
    capability_hints: {},
  },
  {
    preset_id: "google_ai",
    display_name: "Google AI",
    protocol: "google_generate_content",
    base_url: "https://generativelanguage.googleapis.com/v1beta",
    recommended_credential_ref: "env:GOOGLE_API_KEY",
    trust_level: "OFFICIAL_VENDOR",
    credential_type: "API_KEY",
    model_hints: ["gemini-2.5-pro"],
    capability_hints: {},
  },
] as const;

function registryHandler() {
  let providers = [{ ...provider }];
  let models = [{ ...model }];
  let latestProbe: typeof probe | null = null;
  let defaultModelId: string | null = null;
  return {
    get providers() { return providers; },
    get models() { return models; },
    handle(call: FetchCall) {
      const path = call.path.split("?")[0];
      if (call.method === "GET" && path === "/api/v1/providers") return providers;
      if (call.method === "GET" && path === "/api/v1/models") return models;
      if (call.method === "GET" && path === "/api/v1/providers/presets") return [];
      if (call.method === "GET" && path === "/api/v1/system") return { name: "MathModel AI", version: "0.1.0", environment: "local", default_provider: "mock", configured_providers: ["mock"], secret_store_configured: true };
      if (call.method === "GET" && path === "/api/v1/model-routing") return { default_model_id: defaultModelId, legacy_default_provider: "mock", legacy_default_model: "mock-foundation", legacy_config_deprecated: true, registered_model_count: models.length, agent_route_count: 0 };
      if (call.method === "PUT" && path === "/api/v1/model-routing/default") {
        defaultModelId = String((call.body as { model_id: string }).model_id);
        return { default_model_id: defaultModelId, legacy_default_provider: "mock", legacy_default_model: "mock-foundation", legacy_config_deprecated: true, registered_model_count: models.length, agent_route_count: 0 };
      }
      if (call.method === "GET" && path.endsWith("/probe")) return latestProbe;
      if (call.method === "PUT" && path.endsWith("/credential")) {
        providers = providers.map((item) => ({ ...item, credential_configured: true, health_status: "UNAVAILABLE" }));
        return { credential_configured: true };
      }
      if (call.method === "POST" && path === "/api/v1/providers") {
        const body = call.body as Record<string, unknown>;
        if (String(body.base_url).includes("invalid")) return jsonResponse({ detail: "Invalid Provider Configuration" }, 422);
        const created = { ...provider, ...body, credential_configured: false };
        providers = [...providers, created];
        return jsonResponse(created, 201);
      }
      if (call.method === "POST" && path === "/api/v1/models") {
        const created = { ...model, ...(call.body as object), effective_capabilities: model.effective_capabilities };
        models = [...models, created];
        return jsonResponse(created, 201);
      }
      if (call.method === "POST" && path.endsWith("/test-connection")) {
        return { provider_id: provider.provider_id, connected: true, credential_accepted: true, model_discovery_supported: true };
      }
      if (call.method === "POST" && path.endsWith("/probe")) {
        latestProbe = probe;
        models = models.map((item) => item.model_id === model.model_id ? { ...item, effective_capabilities: { ...item.effective_capabilities, TEXT: capability("SUPPORTED", "PROBED"), STRUCTURED_OUTPUT: capability("SUPPORTED", "PROBED") } } : item);
        return probe;
      }
      if (call.method === "PATCH" && path.startsWith("/api/v1/models/")) {
        models = models.map((item) => item.model_id === path.split("/").at(-1) ? { ...item, ...(call.body as object) } : item);
        return models.find((item) => item.model_id === path.split("/").at(-1));
      }
      if (call.method === "POST" && path.endsWith("/disable")) {
        providers = providers.map((item) => ({ ...item, enabled: false, health_status: "DISABLED" }));
        return providers[0];
      }
      throw new Error(`Unhandled ${call.method} ${call.path}`);
    },
  };
}

describe("Models & API", () => {
  it("persists the default model through the backend preference endpoint", async () => {
    const backend = registryHandler();
    const calls = installFetch((call) => backend.handle(call));
    renderPage(<ModelsApiPage />);
    const select = await screen.findByLabelText("Default model");
    await userEvent.selectOptions(select, model.model_id);
    await waitFor(() => expect(select).toHaveValue(model.model_id));
    expect(calls.some((call) => call.method === "PUT" && call.path === "/api/v1/model-routing/default")).toBe(true);
  });

  it("lists providers and clears a write-only API key after successful submit", async () => {
    const backend = registryHandler();
    const calls = installFetch((call) => backend.handle(call));
    const consoleSpy = vi.spyOn(console, "log");
    const storageSpy = vi.spyOn(Storage.prototype, "setItem");
    const { queryClient } = renderPage(<ModelsApiPage />);
    expect(await screen.findByRole("heading", { name: "DeepSeek Official" })).toBeInTheDocument();
    expect(screen.getByText("Not configured")).toBeInTheDocument();

    const secret = "sk-browser-secret-never-persist";
    const input = screen.getByLabelText("API key for DeepSeek Official");
    await userEvent.type(input, secret);
    await userEvent.click(screen.getByRole("button", { name: "Replace API Key" }));
    await waitFor(() => expect(input).toHaveValue(""));
    expect(await screen.findByText("Configured · write only")).toBeInTheDocument();
    expect(screen.queryByDisplayValue(secret)).not.toBeInTheDocument();
    expect(storageSpy).not.toHaveBeenCalled();
    expect(window.location.href).not.toContain(secret);
    expect(consoleSpy).not.toHaveBeenCalledWith(expect.stringContaining(secret));
    expect(calls.find((call) => call.path.endsWith("/credential"))?.body).toEqual({ api_key: secret });
    expect(
      JSON.stringify(queryClient.getMutationCache().getAll().map((entry) => entry.state)),
    ).not.toContain(secret);

    const providerCard = screen.getByRole("heading", { name: "DeepSeek Official" }).closest("section");
    expect(providerCard).not.toBeNull();
    await userEvent.click(within(providerCard!).getByRole("button", { name: "Test Connection" }));
    expect(await within(providerCard!).findByText(/Connection successful/)).toBeInTheDocument();
    const modelCard = screen.getByRole("heading", { name: "DeepSeek Main" }).closest("section");
    expect(modelCard).not.toBeNull();
    expect(within(modelCard!).getByText("PROBE_REQUIRED")).toBeInTheDocument();
    await userEvent.click(within(providerCard!).getByText("Advanced provider settings"));
    await userEvent.click(within(providerCard!).getByRole("button", { name: "Disable" }));
    await waitFor(() => expect(within(providerCard!).getByText("DISABLED")).toBeInTheDocument());
  });

  it("clears a rejected credential and never renders malicious backend detail", async () => {
    const backend = registryHandler();
    installFetch((call) => {
      if (call.method === "PUT" && call.path.endsWith("/credential")) {
        return jsonResponse({ detail: "credential=TEST_SECRET_MUST_NOT_ECHO" }, 422);
      }
      return backend.handle(call);
    });
    renderPage(<ModelsApiPage />);
    await screen.findByRole("heading", { name: "DeepSeek Official" });
    const input = screen.getByLabelText("API key for DeepSeek Official");
    await userEvent.type(input, "submitted-secret-value");
    await userEvent.click(screen.getByRole("button", { name: "Replace API Key" }));
    expect(await screen.findByText("Credential update failed")).toBeInTheDocument();
    expect(input).toHaveValue("");
    expect(screen.queryByText(/TEST_SECRET_MUST_NOT_ECHO/)).not.toBeInTheDocument();
    expect(screen.queryByText(/submitted-secret-value/)).not.toBeInTheDocument();
  });

  it("renders hostile registry text as inert text", async () => {
    const backend = registryHandler();
    const payload = '<img src=x onerror="window.__xss=1">';
    installFetch((call) => {
      if (call.method === "GET" && call.path === "/api/v1/providers") {
        return [{ ...provider, display_name: payload }];
      }
      return backend.handle(call);
    });
    const view = renderPage(<ModelsApiPage />);
    expect(await screen.findByRole("heading", { name: payload })).toBeInTheDocument();
    expect(view.container.querySelector("img")).toBeNull();
    expect((window as Window & { __xss?: number }).__xss).toBeUndefined();
  });

  it("marks Mock providers and models as test-only", async () => {
    const backend = registryHandler();
    installFetch((call) => {
      if (call.method === "GET" && call.path === "/api/v1/providers") {
        return [{ ...provider, provider_id: "mock", display_name: "Mock Provider" }];
      }
      if (call.method === "GET" && call.path === "/api/v1/models") {
        return [{ ...model, provider_id: "mock", model_id: "mock-model", display_name: "Mock Model" }];
      }
      return backend.handle(call);
    });
    renderPage(<ModelsApiPage />);
    expect(await screen.findByRole("heading", { name: "Mock Provider" })).toBeInTheDocument();
    expect(screen.getAllByText("MOCK / TEST ONLY")).toHaveLength(2);
  });

  it("shows two providers and four model profiles without hiding the add-model action", async () => {
    const backend = registryHandler();
    const providerB = { ...provider, provider_id: "qwen-official", display_name: "Qwen Official", config_digest: "1".repeat(64) };
    const models = [
      model,
      { ...model, model_id: "deepseek-v4-pro", display_name: "DeepSeek V4 Pro", remote_model: "deepseek-v4-pro", config_digest: "2".repeat(64) },
      { ...model, provider_id: providerB.provider_id, model_id: "qwen-fast", display_name: "Qwen Fast", config_digest: "3".repeat(64) },
      { ...model, provider_id: providerB.provider_id, model_id: "qwen-vision", display_name: "Qwen Vision", config_digest: "4".repeat(64) },
    ];
    installFetch((call) => {
      if (call.method === "GET" && call.path === "/api/v1/providers") return [provider, providerB];
      if (call.method === "GET" && call.path === "/api/v1/models") return models;
      return backend.handle(call);
    });

    renderPage(<ModelsApiPage />);

    expect(await screen.findByRole("heading", { name: "Providers (2)" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Models (4)" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add Model" })).toBeEnabled();
    for (const displayName of ["DeepSeek Main", "DeepSeek V4 Pro", "Qwen Fast", "Qwen Vision"]) {
      expect(screen.getByRole("heading", { name: displayName })).toBeInTheDocument();
    }
  });

  it("offers distinct catalog and custom provider flows with safe preset defaults", async () => {
    const backend = registryHandler();
    installFetch((call) => {
      if (call.method === "GET" && call.path === "/api/v1/providers/presets") return catalogPresets;
      return backend.handle(call);
    });
    renderPage(<ModelsApiPage />);
    await screen.findByRole("heading", { name: "DeepSeek Official" });

    await userEvent.click(screen.getByRole("button", { name: "Add Provider" }));
    expect(screen.getByRole("heading", { name: "Provider Catalog" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^DeepSeekOpenAI/ })).toBeEnabled();
    expect(screen.getByRole("button", { name: /Qwen \/ Alibaba Model Studio/ })).toBeEnabled();
    expect(screen.getByRole("button", { name: /^OpenAIOpenAI Responses/ })).toBeEnabled();
    expect(screen.getByRole("button", { name: /Anthropic/ })).toBeEnabled();
    expect(screen.getByRole("button", { name: /Google Gemini/ })).toBeEnabled();
    expect(screen.getByRole("button", { name: /Zhipu GLM/ })).toBeEnabled();
    expect(screen.getByRole("button", { name: /Moonshot \/ Kimi/ })).toBeEnabled();
    expect(screen.getByRole("button", { name: /^Custom Provider/ })).toBeEnabled();
    await userEvent.click(screen.getByRole("button", { name: /Qwen \/ Alibaba Model Studio/ }));
    expect(screen.getByDisplayValue("https://dashscope.aliyuncs.com/compatible-mode/v1")).toHaveAttribute("readonly");
    expect(screen.queryByLabelText("Provider ID")).not.toBeInTheDocument();
    expect(screen.getByLabelText("New provider API key")).toHaveAttribute("type", "password");
    expect(screen.getByLabelText("Request timeout")).toHaveValue(60);

    await userEvent.click(screen.getByRole("button", { name: "Add Custom Provider" }));
    expect(screen.getByRole("heading", { name: "Add Custom Provider" })).toBeInTheDocument();
    expect(screen.getByLabelText("Provider ID")).toHaveAttribute("placeholder", "my-company-gateway");
    expect(screen.getByText(/Stable and immutable after creation/)).toBeInTheDocument();
    expect(screen.getByLabelText("Credential reference")).toHaveValue("");
    expect(screen.getByLabelText("Base URL")).toHaveValue("");
    expect(screen.getByText(/Local\/private endpoints require server administrator opt-in/)).toBeInTheDocument();
  });

  it("saves a catalog provider, writes the API key once, and tests through FastAPI", async () => {
    const backend = registryHandler();
    const calls = installFetch((call) => {
      if (call.method === "GET" && call.path === "/api/v1/providers/presets") return catalogPresets;
      return backend.handle(call);
    });
    renderPage(<ModelsApiPage />);
    await screen.findByRole("heading", { name: "DeepSeek Official" });
    await userEvent.click(screen.getByRole("button", { name: "Add Provider" }));
    await userEvent.click(screen.getByRole("button", { name: /Qwen \/ Alibaba Model Studio/ }));
    await userEvent.type(screen.getByLabelText("New provider API key"), "catalog-secret");
    await userEvent.click(screen.getByRole("button", { name: "Save & Test" }));

    expect(await screen.findByText(/Provider saved and connection test passed/)).toBeInTheDocument();
    expect(calls.filter((call) => call.method === "POST" && call.path === "/api/v1/providers")).toHaveLength(1);
    expect(calls.filter((call) => call.method === "PUT" && call.path.endsWith("/credential"))).toHaveLength(1);
    expect(calls.filter((call) => call.method === "POST" && call.path.endsWith("/test-connection"))).toHaveLength(1);
    expect(screen.queryByDisplayValue("catalog-secret")).not.toBeInTheDocument();
  });

  it("opens GLM and Kimi as custom templates without inventing an endpoint", async () => {
    const backend = registryHandler();
    installFetch((call) => {
      if (call.method === "GET" && call.path === "/api/v1/providers/presets") return catalogPresets;
      return backend.handle(call);
    });
    renderPage(<ModelsApiPage />);
    await screen.findByRole("heading", { name: "DeepSeek Official" });
    await userEvent.click(screen.getByRole("button", { name: "Add Provider" }));
    await userEvent.click(screen.getByRole("button", { name: /Zhipu GLM/ }));
    expect(screen.getByLabelText("Provider ID")).toHaveValue("zhipu-glm");
    expect(screen.getByLabelText("Display name")).toHaveValue("Zhipu GLM");
    expect(screen.getByLabelText("Base URL")).toHaveValue("");
    expect(screen.queryByDisplayValue(/localhost/i)).not.toBeInTheDocument();
  });

  it("fetches two model candidates through the backend and explicitly creates profiles", async () => {
    const backend = registryHandler();
    const configuredProvider = { ...provider, credential_configured: true, health_status: "UNAVAILABLE" as const };
    const calls = installFetch((call) => {
      if (call.method === "GET" && call.path === "/api/v1/providers") return [configuredProvider];
      if (call.method === "POST" && call.path.endsWith("/discover-models")) {
        return { provider_id: provider.provider_id, capabilities_probed: false, models: [{ remote_model_id: "model-a", display_name: "Model A" }, { remote_model_id: "model-b", display_name: "Model B" }] };
      }
      return backend.handle(call);
    });
    renderPage(<ModelsApiPage />);
    await screen.findByRole("heading", { name: "DeepSeek Official" });
    await userEvent.click(screen.getByRole("button", { name: "Manage" }));
    expect(screen.getByText(/Fetch Models imports identifiers only/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Fetch Models" }));
    expect(await screen.findByText("Model A")).toBeInTheDocument();
    expect(screen.getByText("Model B")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Add Selected Models (2)" }));
    await waitFor(() => expect(calls.filter((call) => call.method === "POST" && call.path === "/api/v1/models")).toHaveLength(2));
    const created = calls.filter((call) => call.method === "POST" && call.path === "/api/v1/models");
    expect(created.map((call) => (call.body as { remote_model: string }).remote_model)).toEqual(["model-a", "model-b"]);
    expect(created.every((call) => Object.keys((call.body as { declared_capabilities: object }).declared_capabilities).length === 0)).toBe(true);
    expect(await screen.findByRole("heading", { name: "Models (3)" })).toBeInTheDocument();
  });

  it.each([
    [409, "MODEL_DISCOVERY_UNSUPPORTED", "Model discovery is unavailable. Enter model IDs manually.", /does not expose model discovery/],
    [401, "AUTHENTICATION_FAILED", "Credential rejected by the provider.", /Credential rejected/],
  ])("renders discovery failure semantics without invalidating the provider", async (status, code, message, expected) => {
    const backend = registryHandler();
    const configuredProvider = { ...provider, credential_configured: true };
    installFetch((call) => {
      if (call.method === "GET" && call.path === "/api/v1/providers") return [configuredProvider];
      if (call.method === "POST" && call.path.endsWith("/discover-models")) {
        return jsonResponse({ detail: { code, message } }, status);
      }
      return backend.handle(call);
    });
    renderPage(<ModelsApiPage />);
    await screen.findByRole("heading", { name: "DeepSeek Official" });
    await userEvent.click(screen.getByRole("button", { name: "Manage" }));
    await userEvent.click(screen.getByRole("button", { name: "Fetch Models" }));
    expect(await screen.findByText(expected)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "DeepSeek Official" })).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Add Model" }).length).toBeGreaterThanOrEqual(2);
  });

  it("disables direct credential entry when encrypted storage is unavailable", async () => {
    const backend = registryHandler();
    installFetch((call) => {
      if (call.method === "GET" && call.path === "/api/v1/system") {
        return { name: "MathModel AI", version: "0.1.0", environment: "local", default_provider: "mock", configured_providers: ["mock"], secret_store_configured: false };
      }
      return backend.handle(call);
    });

    renderPage(<ModelsApiPage />);

    const existingKeyInput = await screen.findByLabelText("API key for DeepSeek Official");
    expect(existingKeyInput).toBeDisabled();
    expect(screen.getByText(/Credential values are managed outside the Web UI/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Add Provider" }));
    expect(screen.getByLabelText("New provider API key")).toBeDisabled();
    expect(screen.getByText(/Encrypted UI credential storage is not configured/)).toBeInTheDocument();
  });

  it("reconciles an ambiguous provider create without submitting it twice", async () => {
    const backend = registryHandler();
    let committedProvider: typeof provider | null = null;
    const calls = installFetch((call) => {
      const path = call.path.split("?")[0];
      if (call.method === "GET" && path === "/api/v1/providers" && committedProvider) {
        return [provider, committedProvider];
      }
      if (call.method === "POST" && path === "/api/v1/providers") {
        committedProvider = {
          ...provider,
          ...(call.body as object),
          provider_id: "ambiguous-provider",
          display_name: "Ambiguous Provider",
          base_url: "https://ambiguous.example/v1",
          credential_configured: false,
        };
        throw new TypeError("response connection dropped after commit");
      }
      return backend.handle(call);
    });
    renderPage(<ModelsApiPage />);
    await screen.findByRole("heading", { name: "DeepSeek Official" });
    await userEvent.click(screen.getByRole("button", { name: "Add Provider" }));
    await userEvent.type(screen.getByLabelText("Provider ID"), "ambiguous-provider");
    await userEvent.type(screen.getByLabelText("Display name"), "Ambiguous Provider");
    await userEvent.type(screen.getByLabelText("Base URL"), "https://ambiguous.example/v1");
    await userEvent.click(screen.getByRole("button", { name: "Save & Test" }));
    expect(await screen.findByRole("heading", { name: "Ambiguous Provider" })).toBeInTheDocument();
    expect(calls.filter((call) => call.method === "POST")).toHaveLength(1);
    expect(screen.queryByText("Backend is unreachable")).not.toBeInTheDocument();
  });

  it("shows backend validation errors for an invalid provider URL", async () => {
    const backend = registryHandler();
    installFetch((call) => backend.handle(call));
    renderPage(<ModelsApiPage />);
    await screen.findByRole("heading", { name: "DeepSeek Official" });
    await userEvent.click(screen.getByRole("button", { name: "Add Provider" }));
    await userEvent.type(screen.getByLabelText("Provider ID"), "invalid-provider");
    await userEvent.type(screen.getByLabelText("Display name"), "Invalid Provider");
    await userEvent.type(screen.getByLabelText("Base URL"), "https://invalid.example/v1");
    await userEvent.click(screen.getByRole("button", { name: "Save & Test" }));
    expect(await screen.findByText("Invalid Provider Configuration")).toBeInTheDocument();
  });

  it("explains the default server policy when a custom localhost endpoint is rejected", async () => {
    const backend = registryHandler();
    installFetch((call) => {
      if (call.method === "POST" && call.path === "/api/v1/providers") {
        return jsonResponse({ detail: "provider endpoints require HTTPS by default" }, 400);
      }
      return backend.handle(call);
    });
    renderPage(<ModelsApiPage />);
    await screen.findByRole("heading", { name: "DeepSeek Official" });
    await userEvent.click(screen.getByRole("button", { name: "Add Custom Provider" }));
    await userEvent.type(screen.getByLabelText("Provider ID"), "local-provider");
    await userEvent.type(screen.getByLabelText("Display name"), "Local Provider");
    await userEvent.type(screen.getByLabelText("Base URL"), "http://localhost:9000/v1");
    await userEvent.click(screen.getByRole("button", { name: "Save & Test" }));
    expect(await screen.findByText("Local/private endpoints are disabled by server policy.")).toBeInTheDocument();
  });

  it("adds a model, runs a real backend probe, renders capabilities, and disables it", async () => {
    const backend = registryHandler();
    const calls = installFetch((call) => {
      if (call.method === "GET" && call.path === "/api/v1/providers") return [{ ...provider, credential_configured: true, health_status: "READY" }];
      return backend.handle(call);
    });
    renderPage(<ModelsApiPage />);
    await screen.findByText("DeepSeek Main");
    await userEvent.click(screen.getByRole("button", { name: "Add Model" }));
    await userEvent.type(screen.getByLabelText("Model ID"), "deepseek-v4-pro");
    await userEvent.selectOptions(screen.getByLabelText("Provider"), provider.provider_id);
    await userEvent.type(screen.getByLabelText("Display name"), "DeepSeek Reasoner");
    await userEvent.type(screen.getByLabelText("Remote model"), "deepseek-v4-pro");
    await userEvent.click(screen.getByRole("button", { name: "Save model" }));
    expect(await screen.findByText("DeepSeek Reasoner")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Model added" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run Probe" })).toBeEnabled();
    expect(screen.getByRole("link", { name: "Configure Stage Routing" })).toHaveAttribute("href", "/routing");
    await userEvent.click(screen.getByRole("button", { name: "Add Another Model" }));
    expect(screen.getByLabelText("Model ID")).toHaveValue("");

    const mainCard = screen.getByText("DeepSeek Main").closest("section");
    expect(mainCard).not.toBeNull();
    expect(within(mainCard!).getByText("PROBE_REQUIRED")).toBeInTheDocument();
    await userEvent.click(within(mainCard!).getByRole("button", { name: "Run Capability Probe" }));
    await waitFor(() => expect(within(mainCard!).getByText(/Probe auth: PASS/)).toBeInTheDocument());
    expect(within(mainCard!).getByText("PROBED")).toBeInTheDocument();
    expect(within(mainCard!).getByRole("button", { name: "Probe Again" })).toBeEnabled();
    expect(calls.some((call) => call.method === "POST" && call.path.endsWith("/models/deepseek-main/probe"))).toBe(true);
    await userEvent.click(within(mainCard!).getByRole("button", { name: "Disable" }));
    await waitFor(() => expect(within(mainCard!).getByText("DISABLED")).toBeInTheDocument());
  });
});
