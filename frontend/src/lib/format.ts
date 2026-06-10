/** Shared formatting helpers for the observability dashboard. */

export function formatTimestamp(value?: string): string {
  if (!value) return "—";
  return new Date(value).toLocaleString("en-HK", { timeZone: "Asia/Hong_Kong" });
}

export function httpStatusClass(code?: number | null): string {
  if (!code) return "status-badge error";
  if (code >= 200 && code < 300) return "status-badge success";
  if (code >= 400) return "status-badge error";
  return "status-badge";
}

export function tavilyStatusClass(status?: string): string {
  if (status === "success") return "status-badge success";
  if (status === "error") return "status-badge error";
  return "status-badge";
}
