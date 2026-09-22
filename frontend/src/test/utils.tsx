import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";

export type FetchCall = {
  method: string;
  path: string;
  body: unknown;
  mode?: RequestMode;
  url: string;
};

export function renderPage(element: ReactElement, initialEntries = ["/"]) {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: 0 },
      mutations: { retry: false },
    },
  });
  const result = render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={initialEntries}>{element}</MemoryRouter>
    </QueryClientProvider>,
  );
  return Object.assign(result, { queryClient: client });
}

export function installFetch(
  handler: (call: FetchCall) => unknown,
): FetchCall[] {
  const calls: FetchCall[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const request = input instanceof Request ? input : null;
      const method = (init?.method ?? request?.method ?? "GET").toUpperCase();
      const url = new URL(typeof input === "string" || input instanceof URL ? input : input.url);
      const rawBody = init?.body ?? (request ? await request.clone().text() : undefined);
      let body: unknown;
      if (typeof rawBody === "string" && rawBody) {
        try {
          body = JSON.parse(rawBody);
        } catch {
          body = rawBody;
        }
      }
      const call = {
        method,
        path: `${url.pathname}${url.search}`,
        body,
        mode: init?.mode ?? request?.mode,
        url: url.toString(),
      };
      calls.push(call);
      const result = await handler(call);
      if (result instanceof Response) return result;
      return jsonResponse(result);
    }),
  );
  return calls;
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
