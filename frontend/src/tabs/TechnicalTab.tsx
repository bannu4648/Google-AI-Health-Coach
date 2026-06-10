import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { GoogleApiTab } from "../GoogleApiTab";
import { TavilyApiTab } from "../TavilyApiTab";
import { Card } from "../components/Card";
import { JsonBlock } from "../components/JsonBlock";
import { Table } from "../components/Table";
import type { ApiRow } from "../lib/api";

type TechnicalSubtab =
  | "system"
  | "messages"
  | "llm"
  | "google"
  | "tavily"
  | "actions"
  | "jobs"
  | "raw";

export function TechnicalTab({
  summary,
  messages,
  llmCalls,
  googleCalls,
  tavilyCalls,
  actions,
  jobs,
  metrics,
}: {
  summary: ApiRow | null;
  messages: ApiRow[];
  llmCalls: ApiRow[];
  googleCalls: ApiRow[];
  tavilyCalls: ApiRow[];
  actions: ApiRow[];
  jobs: ApiRow[];
  metrics: ApiRow | null;
}) {
  const [subtab, setSubtab] = useState<TechnicalSubtab>("system");

  const latencySeries = useMemo(
    () =>
      [...googleCalls]
        .reverse()
        .map((row, index) => ({
          index,
          latency: row.latency_ms ?? 0,
          dataType: row.data_type ?? "unknown",
        })),
    [googleCalls],
  );

  const statusBars = useMemo(() => {
    const counts = summary?.google_status ?? {};
    return Object.entries(counts).map(([status, count]) => ({ status, count }));
  }, [summary]);

  return (
    <>
      <nav className="subtabs">
        {[
          ["system", "System"],
          ["messages", "Messages"],
          ["llm", "LLM"],
          ["google", "Google Health"],
          ["tavily", "Tavily"],
          ["actions", "Actions"],
          ["jobs", "Jobs"],
          ["raw", "Raw"],
        ].map(([id, label]) => (
          <button key={id} className={subtab === id ? "tab active" : "tab"} onClick={() => setSubtab(id as TechnicalSubtab)}>
            {label}
          </button>
        ))}
      </nav>

      {subtab === "system" && (
        <>
          <div className="grid stats">
            {Object.entries(summary?.counts ?? {}).map(([label, value]) => (
              <Card key={label} title={label.replaceAll("_", " ")}>
                <p className="stat">{String(value)}</p>
              </Card>
            ))}
          </div>
          <div className="grid">
            <Card title="Google Health Latency">
              <ResponsiveContainer width="100%" height={260}>
                <AreaChart data={latencySeries}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="index" />
                  <YAxis />
                  <Tooltip />
                  <Area dataKey="latency" stroke="#4f46e5" fill="#c7d2fe" />
                </AreaChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Google Status Codes">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={statusBars}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="status" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#059669" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </>
      )}

      {subtab === "messages" && (
        <Card title="Messages">
          <Table rows={messages} columns={["created_at", "direction", "phone", "status", "text"]} />
        </Card>
      )}
      {subtab === "llm" && (
        <Card title="Mistral Calls">
          <Table rows={llmCalls} columns={["created_at", "purpose", "model", "status", "latency_ms", "error"]} />
        </Card>
      )}
      {subtab === "google" && <GoogleApiTab calls={googleCalls} />}
      {subtab === "tavily" && <TavilyApiTab calls={tavilyCalls} />}
      {subtab === "actions" && (
        <Card title="Health Actions">
          <Table rows={actions} columns={["created_at", "intent", "status", "error"]} />
        </Card>
      )}
      {subtab === "jobs" && (
        <Card title="Scheduled Jobs">
          <Table rows={jobs} columns={["created_at", "job_name", "status", "error"]} />
        </Card>
      )}
      {subtab === "raw" && (
        <Card title="Raw Metric Series">
          <JsonBlock value={metrics ?? {}} />
        </Card>
      )}
    </>
  );
}
