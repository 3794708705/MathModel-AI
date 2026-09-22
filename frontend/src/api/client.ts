import createClient from "openapi-fetch";

import type { paths } from "./schema";
import type {
  AgentRoute,
  AgentRouteInput,
  BenchmarkCase,
  BenchmarkReport,
  BenchmarkRun,
  BenchmarkRunInput,
  IndependentVerification,
  CapabilityProbe,
  Model,
  ModelDiscovery,
  ModelInput,
  ModelPatch,
  Paper,
  ProblemState,
  ProjectInput,
  ProjectSummary,
  Provider,
  ProviderConnectionTest,
  ProviderInput,
  ProviderPatch,
  ProviderPreset,
  Readiness,
  RoutableAgent,
  RoutingOverview,
  RoutingPreview,
  Submission,
  SystemInfo,
} from "../types/contracts";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";
const REQUEST_TIMEOUT_MS = 15_000;
const REACHABILITY_TIMEOUT_MS = 2_000;

export function resolveApiBaseUrl(configured?: string): string {
  return (configured?.trim() || DEFAULT_API_BASE_URL).replace(/\/$/, "");
}

export const apiBaseUrl = resolveApiBaseUrl(import.meta.env.VITE_API_BASE_URL);

class RequestTimeoutError extends Error {
  constructor() {
    super("request timeout");
    this.name = "RequestTimeoutError";
  }
}

async function fetchWithTimeout(
  input: RequestInfo | URL,
  init: RequestInit = {},
  timeoutMs = REQUEST_TIMEOUT_MS,
): Promise<Response> {
  const controller = new AbortController();
  const requestSignal = init.signal ?? (input instanceof Request ? input.signal : undefined);
  const abortFromRequest = () => controller.abort(requestSignal?.reason);
  requestSignal?.addEventListener("abort", abortFromRequest, { once: true });
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await globalThis.fetch(input, { ...init, signal: controller.signal });
  } catch (error) {
    if (controller.signal.aborted && !requestSignal?.aborted) throw new RequestTimeoutError();
    throw error;
  } finally {
    clearTimeout(timeout);
    requestSignal?.removeEventListener("abort", abortFromRequest);
  }
}

const client = createClient<paths>({
  baseUrl: apiBaseUrl,
  credentials: "omit",
  fetch: (request: Request) => fetchWithTimeout(request),
});

type ApiResult<T> = {
  data?: T;
  error?: unknown;
  response: Response;
};

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, message: string, code = `HTTP_${status}`) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function backendRespondsWithoutCors(): Promise<boolean> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REACHABILITY_TIMEOUT_MS);
  try {
    await globalThis.fetch(`${apiBaseUrl}/health/live`, {
      cache: "no-store",
      credentials: "omit",
      mode: "no-cors",
      signal: controller.signal,
    });
    return true;
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}

function isLocalApi(): boolean {
  try {
    return ["127.0.0.1", "localhost", "::1"].includes(new URL(apiBaseUrl).hostname);
  } catch {
    return false;
  }
}

async function connectionError(error: unknown): Promise<ApiError> {
  if (
    error instanceof RequestTimeoutError
    || (error instanceof DOMException && error.name === "AbortError")
  ) {
    return new ApiError(
      0,
      `The backend did not respond within ${REQUEST_TIMEOUT_MS / 1_000} seconds.`,
      "REQUEST_TIMEOUT",
    );
  }
  if (await backendRespondsWithoutCors()) {
    return new ApiError(
      0,
      `The backend responded at ${apiBaseUrl}, but the browser blocked access. Allow this frontend origin in MM_CORS_ORIGINS.`,
      "CORS_BLOCKED",
    );
  }
  if (isLocalApi()) {
    return new ApiError(
      0,
      `Backend is not listening at ${apiBaseUrl}. From the repository root, run: .\\scripts\\dev-backend.ps1`,
      "BACKEND_OFFLINE",
    );
  }
  return new ApiError(
    0,
    `Cannot reach the configured backend at ${apiBaseUrl}. Check VITE_API_BASE_URL and the network connection.`,
    "NETWORK_ERROR",
  );
}

async function unwrap<T>(
  request: Promise<ApiResult<T>>,
  options: { safeErrorMessage?: string } = {},
): Promise<T> {
  let result: ApiResult<T>;
  try {
    result = await request;
  } catch (error) {
    throw await connectionError(error);
  }
  const { data, error, response } = result;
  if (!response.ok || data === undefined) {
    const details = errorDetails(error, response.status);
    throw new ApiError(
      response.status,
      options.safeErrorMessage ?? details.message,
      details.code,
    );
  }
  return data;
}

function errorMessage(error: unknown, status: number): string {
  return errorDetails(error, status).message;
}

function errorDetails(error: unknown, status: number): { message: string; code: string } {
  if (typeof error === "object" && error !== null && "detail" in error) {
    const detail = error.detail;
    if (status >= 500) {
      const structuredCode = typeof detail === "object" && detail !== null && "code" in detail
        ? String(detail.code)
        : "";
      if (!["MODEL_DISCOVERY_FAILED", "PROVIDER_UNREACHABLE"].includes(structuredCode)) {
        return { message: "The backend encountered an internal error", code: `HTTP_${status}` };
      }
    }
    if (typeof detail === "string") return { message: redactSensitiveText(detail).slice(0, 600), code: `HTTP_${status}` };
    if (typeof detail === "object" && detail !== null && "message" in detail) {
      const code = "code" in detail ? String(detail.code) : `HTTP_${status}`;
      return {
        message: redactSensitiveText(String(detail.message)).slice(0, 600),
        code,
      };
    }
    if (Array.isArray(detail)) {
      const messages = detail
        .map((item) => {
          if (typeof item === "object" && item !== null && "msg" in item) {
            return redactSensitiveText(String((item as { msg: unknown }).msg));
          }
          return "Invalid request";
        })
        .join("; ");
      return { message: messages.slice(0, 600), code: `HTTP_${status}` };
    }
  }
  if (status >= 500) {
    return { message: "The backend encountered an internal error", code: `HTTP_${status}` };
  }
  return {
    message: status === 0 ? "Backend is unreachable" : `Request failed (${status})`,
    code: `HTTP_${status}`,
  };
}

function redactSensitiveText(value: string): string {
  return value
    .replace(/Bearer\s+[A-Za-z0-9._~+/-]+/gi, "[REDACTED]")
    .replace(/(?:sk|key)-[A-Za-z0-9_-]{8,}/gi, "[REDACTED]")
    .replace(
      /(authorization|api[_-]?key|secret|password|cookie|credential)\s*[:=]\s*[^\s,;]+/gi,
      "[REDACTED]",
    );
}

async function optional<T>(request: Promise<T>): Promise<T | null> {
  try {
    return await request;
  } catch (error) {
    if (error instanceof ApiError && [400, 404, 409].includes(error.status)) return null;
    throw error;
  }
}

export const api = {
  live: () => unwrap(client.GET("/health/live")),
  ready: (): Promise<Readiness> => unwrap(client.GET("/health/ready")),
  system: (): Promise<SystemInfo> => unwrap(client.GET("/api/v1/system")),

  providers: (): Promise<Provider[]> => unwrap(client.GET("/api/v1/providers")),
  providerPresets: (): Promise<ProviderPreset[]> =>
    unwrap(client.GET("/api/v1/providers/presets")),
  createProvider: (body: ProviderInput): Promise<Provider> =>
    unwrap(client.POST("/api/v1/providers", { body })),
  patchProvider: (providerId: string, body: ProviderPatch): Promise<Provider> =>
    unwrap(
      client.PATCH("/api/v1/providers/{provider_id}", {
        params: { path: { provider_id: providerId } },
        body,
      }),
    ),
  setProviderEnabled: (providerId: string, enabled: boolean): Promise<Provider> =>
    unwrap(
      client.POST(
        enabled
          ? "/api/v1/providers/{provider_id}/enable"
          : "/api/v1/providers/{provider_id}/disable",
        { params: { path: { provider_id: providerId } } },
      ),
    ),
  putCredential: (providerId: string, apiKey: string): Promise<{ credential_configured: boolean }> =>
    unwrap(
      client.PUT("/api/v1/providers/{provider_id}/credential", {
        params: { path: { provider_id: providerId } },
        body: { api_key: apiKey },
        cache: "no-store",
      }),
      { safeErrorMessage: "Credential update failed" },
    ),
  deleteCredential: (providerId: string): Promise<{ credential_configured: boolean }> =>
    unwrap(
      client.DELETE("/api/v1/providers/{provider_id}/credential", {
        params: { path: { provider_id: providerId } },
        cache: "no-store",
      }),
      { safeErrorMessage: "Credential removal failed" },
    ),
  probeProvider: (providerId: string): Promise<CapabilityProbe[]> =>
    unwrap(
      client.POST("/api/v1/providers/{provider_id}/probe", {
        params: { path: { provider_id: providerId } },
      }),
    ),
  testProviderConnection: (providerId: string): Promise<ProviderConnectionTest> =>
    unwrap(
      client.POST("/api/v1/providers/{provider_id}/test-connection", {
        params: { path: { provider_id: providerId } },
      }),
    ),
  discoverModels: (providerId: string): Promise<ModelDiscovery> =>
    unwrap(
      client.POST("/api/v1/providers/{provider_id}/discover-models", {
        params: { path: { provider_id: providerId } },
      }),
    ),

  models: (): Promise<Model[]> => unwrap(client.GET("/api/v1/models")),
  createModel: (body: ModelInput): Promise<Model> =>
    unwrap(client.POST("/api/v1/models", { body })),
  patchModel: (modelId: string, body: ModelPatch): Promise<Model> =>
    unwrap(
      client.PATCH("/api/v1/models/{model_id}", {
        params: { path: { model_id: modelId } },
        body,
      }),
    ),
  probeModel: (modelId: string): Promise<CapabilityProbe> =>
    unwrap(
      client.POST("/api/v1/models/{model_id}/probe", {
        params: { path: { model_id: modelId } },
      }),
    ),
  latestProbe: (modelId: string): Promise<CapabilityProbe | null> =>
    unwrap(
      client.GET("/api/v1/models/{model_id}/probe", {
        params: { path: { model_id: modelId } },
      }),
    ),

  routingOverview: (): Promise<RoutingOverview> =>
    unwrap(client.GET("/api/v1/model-routing")),
  setDefaultModel: (modelId: string): Promise<RoutingOverview> =>
    unwrap(client.PUT("/api/v1/model-routing/default", { body: { model_id: modelId } })),
  routableAgents: (): Promise<RoutableAgent[]> =>
    unwrap(client.GET("/api/v1/model-routing/agent-catalog")),
  agentRoutes: (): Promise<AgentRoute[]> =>
    unwrap(client.GET("/api/v1/model-routing/agents")),
  previewRoute: (agent: RoutableAgent, modelId: string): Promise<RoutingPreview> =>
    unwrap(
      client.POST("/api/v1/model-routing/preview", {
        body: {
          agent_name: agent.name,
          profile: { ...agent.task_profile, preferred_model_id: modelId },
        },
      }),
    ),
  saveAgentRoute: (agentName: string, body: AgentRouteInput): Promise<AgentRoute> =>
    unwrap(
      client.PUT("/api/v1/model-routing/agents/{agent_name}", {
        params: { path: { agent_name: agentName } },
        body,
      }),
    ),

  projects: (): Promise<ProjectSummary[]> => unwrap(client.GET("/api/v1/projects")),
  createProject: (body: ProjectInput): Promise<ProblemState> =>
    unwrap(client.POST("/api/v1/projects", { body })),
  project: (projectId: string): Promise<ProblemState> =>
    unwrap(
      client.GET("/api/v1/projects/{project_id}", {
        params: { path: { project_id: projectId } },
      }),
    ),
  runReasoning: (projectId: string): Promise<unknown> =>
    unwrap(
      client.POST("/api/v1/projects/{project_id}/reasoning/run", {
        params: { path: { project_id: projectId } },
        body: {},
      }),
    ),
  analyzeData: (projectId: string): Promise<unknown> =>
    unwrap(
      client.POST("/api/v1/projects/{project_id}/data/analyze", {
        params: { path: { project_id: projectId } },
        body: {},
      }),
    ),
  runMathematical: (projectId: string): Promise<unknown> =>
    unwrap(
      client.POST("/api/v1/projects/{project_id}/mathematical/run", {
        params: { path: { project_id: projectId } },
        body: {},
      }),
    ),
  runVerification: (projectId: string): Promise<unknown> =>
    unwrap(
      client.POST("/api/v1/projects/{project_id}/verification/run", {
        params: { path: { project_id: projectId } },
        body: {},
      }),
    ),
  runPaper: (projectId: string): Promise<unknown> =>
    unwrap(
      client.POST("/api/v1/projects/{project_id}/paper/run", {
        params: { path: { project_id: projectId } },
        body: {},
      }),
    ),
  paper: (projectId: string): Promise<Paper | null> =>
    optional(
      unwrap(
        client.GET("/api/v1/projects/{project_id}/paper", {
          params: { path: { project_id: projectId } },
        }),
      ),
    ),
  submission: (projectId: string): Promise<Submission | null> =>
    optional(
      unwrap(
        client.GET("/api/v1/projects/{project_id}/submission", {
          params: { path: { project_id: projectId } },
        }),
      ),
    ),
  uploadProjectFile: async (projectId: string, file: File): Promise<unknown> => {
    const body = new FormData();
    body.append("upload", file);
    let response: Response;
    try {
      response = await fetchWithTimeout(`${apiBaseUrl}/api/v1/projects/${encodeURIComponent(projectId)}/files`, {
        method: "POST",
        credentials: "omit",
        body,
      });
    } catch (error) {
      throw await connectionError(error);
    }
    if (!response.ok) {
      const error: unknown = await response.json().catch(() => undefined);
      throw new ApiError(response.status, errorMessage(error, response.status));
    }
    return response.json() as Promise<unknown>;
  },

  benchmarkCases: (): Promise<BenchmarkCase[]> =>
    unwrap(client.GET("/api/v1/benchmarks/cases")),
  independentVerification: (attemptId: string): Promise<IndependentVerification> =>
    unwrap(client.GET("/api/v1/benchmarks/attempts/{attempt_id}/verification", {
      params: { path: { attempt_id: attemptId } },
    })),
  recomputeMetrics: (attemptId: string): Promise<IndependentVerification> =>
    unwrap(client.POST("/api/v1/benchmarks/attempts/{attempt_id}/verification/recompute", {
      params: { path: { attempt_id: attemptId } },
    })),
  replayScenario: (attemptId: string, scenarioId: string): Promise<IndependentVerification> =>
    unwrap(client.POST("/api/v1/benchmarks/attempts/{attempt_id}/verification/scenarios/{scenario_id}/replay", {
      params: { path: { attempt_id: attemptId, scenario_id: scenarioId } },
      fetch: (request: Request) => fetchWithTimeout(request, {}, 150_000),
    })),
  benchmarkRuns: (): Promise<BenchmarkRun[]> =>
    unwrap(client.GET("/api/v1/benchmarks/runs")),
  benchmarkReport: (runId: string): Promise<BenchmarkReport> =>
    unwrap(
      client.GET("/api/v1/benchmarks/runs/{run_id}/report", {
        params: { path: { run_id: runId } },
      }),
    ),
  createBenchmarkRun: (body: BenchmarkRunInput): Promise<BenchmarkReport> =>
    unwrap(client.POST("/api/v1/benchmarks/runs", { body })),
};
