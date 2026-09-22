import type { components } from "../api/schema";

export type AgentRoute = components["schemas"]["AgentRoutePolicy"];
export type AgentRouteInput = components["schemas"]["AgentRoutePutRequest"];
export type BenchmarkCase = components["schemas"]["BenchmarkCaseSummary"];
export type BenchmarkReport = components["schemas"]["BenchmarkReport"];
export type BenchmarkRun = components["schemas"]["BenchmarkRun"];
export type BenchmarkRunInput = components["schemas"]["BenchmarkRunRequest"];
export type IndependentVerification = components["schemas"]["IndependentVerificationView"];
export type CapabilityEvidence = components["schemas"]["CapabilityEvidence"];
export type CapabilityProbe = components["schemas"]["CapabilityProbeResult"];
export type CredentialInput = components["schemas"]["CredentialPutRequest"];
export type Health = components["schemas"]["HealthResponse"];
export type Model = components["schemas"]["ModelProfile"];
export type ModelDiscovery = components["schemas"]["ModelDiscoveryResponse"];
export type ModelDiscoveryCandidate = components["schemas"]["ModelDiscoveryCandidateView"];
export type ModelInput = components["schemas"]["ModelCreateRequest"];
export type ModelPatch = components["schemas"]["ModelPatchRequest"];
export type Paper = components["schemas"]["PaperVersion"];
export type ProblemState = components["schemas"]["ProblemState"];
export type ProjectInput = components["schemas"]["ProjectProblemCreateRequest"];
export type ProjectSummary = components["schemas"]["ProjectSummary"];
export type Provider = components["schemas"]["ProviderEndpointView"];
export type ProviderConnectionTest = components["schemas"]["ProviderConnectionTestResponse"];
export type ProviderInput = components["schemas"]["ProviderCreateRequest"];
export type ProviderPatch = components["schemas"]["ProviderPatchRequest"];
export type ProviderPreset = components["schemas"]["ProviderPresetView"];
export type Readiness = components["schemas"]["ReadinessResponse"];
export type RoutableAgent = components["schemas"]["RoutableAgentView"];
export type RoutingOverview = components["schemas"]["RoutingOverview"];
export type RoutingPreview = components["schemas"]["RoutingPreviewResponse"];
export type Submission = components["schemas"]["SubmissionSnapshot"];
export type SystemInfo = components["schemas"]["SystemInfoResponse"];

export const capabilityNames = [
  "TEXT",
  "VISION",
  "STRUCTURED_OUTPUT",
  "JSON_MODE",
  "JSON_SCHEMA",
  "TOOLS",
  "STREAMING",
  "REASONING_CONTROL",
  "SYSTEM_ROLE",
  "DEVELOPER_ROLE",
  "LONG_CONTEXT",
  "USAGE_REPORTING",
] as const;

export type CapabilityName = (typeof capabilityNames)[number];
