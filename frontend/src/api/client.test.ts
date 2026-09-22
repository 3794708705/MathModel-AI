import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api, resolveApiBaseUrl } from "./client";
import { provider } from "../test/fixtures";
import { installFetch, jsonResponse } from "../test/utils";

describe("api client", () => {
  afterEach(() => vi.unstubAllEnvs());
  it("uses the typed provider contract through the central client", async () => {
    const calls = installFetch((call) => {
      expect(call.path).toBe("/api/v1/providers");
      return [provider];
    });
    await expect(api.providers()).resolves.toEqual([provider]);
    expect(calls).toHaveLength(1);
  });

  it("maps backend errors without exposing arbitrary response internals", async () => {
    installFetch(() => jsonResponse({ detail: "Invalid provider configuration" }, 422));
    const error = await api.providers().catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 422, code: "HTTP_422" });
    expect(String(error)).toContain("Invalid provider configuration");
  });

  it("reports a local backend that is not listening without leaking network internals", async () => {
    installFetch(() => {
      throw new TypeError("socket path C:\\private\\backend and TEST_SECRET_MUST_NOT_ECHO");
    });
    const networkError = await api.providers().catch((caught: unknown) => caught);
    expect(networkError).toMatchObject({ status: 0, code: "BACKEND_OFFLINE" });
    expect(String(networkError)).toContain(".\\scripts\\dev-backend.ps1");
    expect(String(networkError)).not.toContain("TEST_SECRET_MUST_NOT_ECHO");
  });

  it("distinguishes likely CORS blocking from an offline backend", async () => {
    installFetch((call) => {
      if (call.mode === "no-cors" && call.path === "/health/live") {
        return { status: "ok" };
      }
      throw new TypeError("Failed to fetch");
    });

    await expect(api.providers()).rejects.toMatchObject({
      status: 0,
      code: "CORS_BLOCKED",
    });
  });

  it("distinguishes a request timeout", async () => {
    installFetch(() => {
      throw new DOMException("aborted", "AbortError");
    });

    await expect(api.providers()).rejects.toMatchObject({
      status: 0,
      code: "REQUEST_TIMEOUT",
    });
  });

  it("normalizes the default and configured API base URL", () => {
    expect(resolveApiBaseUrl()).toBe("http://127.0.0.1:8000");
    expect(resolveApiBaseUrl(" https://api.example.test/root/ ")).toBe(
      "https://api.example.test/root",
    );
  });

  it("reports a wrong remote API base separately from a local offline backend", async () => {
    vi.stubEnv("VITE_API_BASE_URL", "https://wrong-api.example.test");
    vi.resetModules();
    const configuredClient = await import("./client");
    installFetch(() => {
      throw new TypeError("network unreachable");
    });

    await expect(configuredClient.api.providers()).rejects.toMatchObject({
      code: "NETWORK_ERROR",
      message: expect.stringContaining("https://wrong-api.example.test"),
    });
  });

  it("suppresses server internals for HTTP 500 responses", async () => {

    installFetch(() =>
      jsonResponse({ detail: "traceback /srv/app.py api_key=TEST_SECRET_MUST_NOT_ECHO" }, 500),
    );
    const serverError = await api.providers().catch((caught: unknown) => caught);
    expect(serverError).toMatchObject({
      status: 500,
      message: "The backend encountered an internal error",
    });
    expect(String(serverError)).not.toContain("TEST_SECRET_MUST_NOT_ECHO");
  });

  it("uses a fixed safe error for write-only credential operations", async () => {
    installFetch(() =>
      jsonResponse({ detail: "credential=TEST_SECRET_MUST_NOT_ECHO" }, 422),
    );
    const error = await api
      .putCredential(provider.provider_id, "submitted-secret-value")
      .catch((caught: unknown) => caught);
    expect(error).toMatchObject({ status: 422, message: "Credential update failed" });
    expect(String(error)).not.toContain("TEST_SECRET_MUST_NOT_ECHO");
    expect(String(error)).not.toContain("submitted-secret-value");
  });

  it("recognizes database outages without echoing server-provided details", async () => {
    installFetch(() => jsonResponse({
      detail: { code: "DATABASE_UNAVAILABLE", message: "postgres://user:TEST_SECRET_MUST_NOT_ECHO@host/db" },
    }, 503));
    const error = await api.ready().catch((caught: unknown) => caught);
    expect(error).toMatchObject({ status: 503, code: "DATABASE_UNAVAILABLE" });
    expect(String(error)).toContain("PostgreSQL");
    expect(String(error)).not.toContain("TEST_SECRET_MUST_NOT_ECHO");
  });

  it("does not misclassify an arbitrary 503 as a database outage", async () => {
    installFetch(() => jsonResponse({ detail: "proxy unavailable" }, 503));
    await expect(api.ready()).rejects.toMatchObject({ status: 503, code: "HTTP_503" });
  });
});
