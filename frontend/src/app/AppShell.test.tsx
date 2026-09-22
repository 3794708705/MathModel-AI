import { useQuery } from "@tanstack/react-query";
import { act, screen } from "@testing-library/react";
import { Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AppShell } from "./AppShell";
import { installFetch, renderPage } from "../test/utils";

function renderShell(page = <div>Page content</div>) {
  return renderPage(
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={page} />
      </Route>
    </Routes>,
  );
}

function RecoverablePage({ load }: { load: () => Promise<string> }) {
  const query = useQuery({ queryKey: ["recoverable-page"], queryFn: load, retry: false });
  return <div>{query.isError ? "Page offline" : query.data ?? "Page loading"}</div>;
}

describe("AppShell backend connectivity", () => {
  afterEach(() => vi.useRealTimers());
  it("shows connected only when health live succeeds", async () => {
    installFetch((call) => {
      if (call.path === "/health/live") return { status: "ok" };
      throw new Error(`Unhandled ${call.method} ${call.path}`);
    });

    renderShell();

    expect(await screen.findByText("Backend connected")).toBeInTheDocument();
    expect(screen.getByText("阶段模型分配")).toBeInTheDocument();
  });

  it("shows offline when health live cannot be reached", async () => {
    installFetch(() => {
      throw new TypeError("connection refused");
    });

    renderShell();

    expect(await screen.findByText("Backend offline")).toBeInTheDocument();
  });

  it("refetches active page data when health polling detects backend recovery", async () => {
    vi.useFakeTimers();
    let online = false;
    installFetch((call) => {
      if (call.path === "/health/live" && online) return { status: "ok" };
      throw new TypeError("connection refused");
    });
    const load = vi.fn(() => (
      online ? Promise.resolve("Page recovered") : Promise.reject(new Error("offline"))
    ));

    renderShell(<RecoverablePage load={load} />);
    await act(async () => vi.advanceTimersByTimeAsync(2_100));
    expect(screen.getByText("Backend offline")).toBeInTheDocument();
    expect(screen.getByText("Page offline")).toBeInTheDocument();

    online = true;
    await act(async () => vi.advanceTimersByTimeAsync(10_000));
    expect(screen.getByText("Backend connected")).toBeInTheDocument();
    expect(screen.getByText("Page recovered")).toBeInTheDocument();
    expect(load).toHaveBeenCalledTimes(2);
  });
});
