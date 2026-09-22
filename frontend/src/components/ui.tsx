import type { ReactNode } from "react";

import { ApiError } from "../api/client";
import type { CapabilityEvidence } from "../types/contracts";

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <header className="mb-6 flex flex-col justify-between gap-4 border-b pb-5 sm:flex-row sm:items-start">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-muted">{description}</p>
      </div>
      {actions ? <div className="flex shrink-0 gap-2">{actions}</div> : null}
    </header>
  );
}

export function Panel({
  title,
  description,
  children,
  actions,
  className = "",
}: {
  title?: string;
  description?: string;
  children: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-lg border bg-panel shadow-sm ${className}`}>
      {title ? (
        <div className="flex items-start justify-between gap-4 border-b px-5 py-4">
          <div>
            <h2 className="font-semibold">{title}</h2>
            {description ? <p className="mt-1 text-sm text-muted">{description}</p> : null}
          </div>
          {actions}
        </div>
      ) : null}
      <div className="p-5">{children}</div>
    </section>
  );
}

export function StatusBadge({ value }: { value: string | null | undefined }) {
  const normalized = value || "NOT_EXPOSED";
  const positive = ["READY", "PASS", "PROBED", "SUPPORTED", "SUCCEEDED", "VERIFIED", "FROZEN"];
  const negative = ["FAIL", "FAILED", "AUTH_FAILED", "UNAVAILABLE", "DIRTY", "REJECTED"];
  const warning = [
    "PARTIAL",
    "BLOCKED",
    "BLOCKED_ENVIRONMENT",
    "HUMAN_REVIEW",
    "UNCONFIGURED",
    "NOT_READY",
  ];
  const tone = positive.includes(normalized)
    ? "bg-emerald-50 text-emerald-700 ring-emerald-200"
    : negative.includes(normalized)
      ? "bg-red-50 text-red-700 ring-red-200"
      : warning.includes(normalized)
        ? "bg-amber-50 text-amber-800 ring-amber-200"
        : "bg-slate-100 text-slate-600 ring-slate-200";
  return (
    <span className={`inline-flex rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ${tone}`}>
      {normalized}
    </span>
  );
}

export function CapabilityMark({ evidence }: { evidence?: CapabilityEvidence }) {
  const rawStatus = evidence?.status;
  const status = ["SUPPORTED", "PARTIAL", "UNSUPPORTED", "UNKNOWN"].includes(
    String(rawStatus),
  )
    ? (rawStatus as "SUPPORTED" | "PARTIAL" | "UNSUPPORTED" | "UNKNOWN")
    : "UNKNOWN";
  const mark = { SUPPORTED: "✓", PARTIAL: "△", UNSUPPORTED: "✕", UNKNOWN: "?" }[status];
  const style = {
    SUPPORTED: "text-emerald-700",
    PARTIAL: "text-amber-700",
    UNSUPPORTED: "text-red-700",
    UNKNOWN: "text-slate-500",
  }[status];
  return (
    <span className={`inline-flex items-center gap-1 font-semibold ${style}`} title={evidence?.source}>
      <span aria-hidden="true">{mark}</span>
      <span className="sr-only">{status}</span>
    </span>
  );
}

export function ErrorBanner({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : "An unexpected error occurred";
  const code = error instanceof ApiError ? error.code : "CLIENT_ERROR";
  const title = {
    BACKEND_OFFLINE: "Backend is offline",
    CORS_BLOCKED: "Browser blocked backend access",
    REQUEST_TIMEOUT: "Backend request timed out",
    NETWORK_ERROR: "Network request failed",
  }[code] ?? "Request failed";
  return (
    <div role="alert" className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
      <p className="font-semibold">{title}</p>
      <p className="mt-1">{message}</p>
      <p className="mt-1 text-xs text-red-600">{code}</p>
    </div>
  );
}

export function LoadingState({ label = "Loading backend data…" }: { label?: string }) {
  return (
    <div role="status" className="rounded-md border border-dashed p-6 text-sm text-muted">
      {label}
    </div>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return <div className="rounded-md border border-dashed p-6 text-sm text-muted">{children}</div>;
}

export function DefinitionList({ items }: { items: Array<[string, ReactNode]> }) {
  return (
    <dl className="grid gap-3 text-sm sm:grid-cols-2">
      {items.map(([label, value]) => (
        <div key={label} className="rounded-md bg-slate-50 px-3 py-2">
          <dt className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</dt>
          <dd className="mt-1 break-words font-medium">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function formatDate(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}
