import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card } from "../components/Card";
import { Table } from "../components/Table";
import type { ApiRow } from "../lib/api";

function formatMetric(value: unknown, suffix = ""): string {
  if (value == null || value === "") return "—";
  if (typeof value === "number") return `${Math.round(value).toLocaleString()}${suffix}`;
  return `${value}${suffix}`;
}

export function OverviewTab({
  health,
  trends,
}: {
  health: ApiRow | null;
  trends: ApiRow[];
}) {
  const metrics = health?.metrics ?? {};
  const readiness = health?.readiness ?? {};
  const cards = [
    ["Steps", formatMetric(metrics.steps)],
    ["Active Zone", formatMetric(metrics.active_zone_minutes, " min")],
    ["Workouts", formatMetric(metrics.workouts)],
    ["Sleep Sessions", formatMetric(metrics.sleep_sessions)],
    ["Meals", formatMetric(metrics.meals_logged)],
    ["Hydration", formatMetric(metrics.hydration_ml, " ml")],
  ];

  return (
    <>
      <div className="hero-grid">
        <Card title="Readiness">
          <p className="readiness-score">{readiness.score ?? "—"}</p>
          <p className="muted">{readiness.label ?? "No readiness snapshot yet"}</p>
          {(readiness.reasons ?? []).slice(0, 3).map((reason: string) => (
            <p className="insight" key={reason}>
              {reason}
            </p>
          ))}
        </Card>
        <Card title="Coach Message">
          <p className="coach-message">
            {health?.coach_message || "No daily summary yet. The scheduler will populate this after a summary job runs."}
          </p>
        </Card>
      </div>

      <div className="grid stats">
        {cards.map(([label, value]) => (
          <Card key={label} title={label}>
            <p className="stat">{value}</p>
          </Card>
        ))}
      </div>

      <div className="grid">
        <Card title="Steps Trend">
          {trends.length ? (
            <ResponsiveContainer width="100%" height={240}>
              <AreaChart data={trends}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date_hkt" />
                <YAxis />
                <Tooltip />
                <Area dataKey="steps" stroke="#4f46e5" fill="#c7d2fe" />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <p className="muted">No trend history yet.</p>
          )}
        </Card>
        <Card title="Recommendations">
          {(health?.recommendations ?? []).length ? (
            <ul className="clean-list">
              {health!.recommendations.map((item: string) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          ) : (
            <p className="muted">No recommendations yet.</p>
          )}
        </Card>
      </div>

      <Card title="Recent Activity">
        <Table rows={health?.recent_activity ?? []} columns={["created_at", "intent", "status"]} />
      </Card>

      <Card title="Coach Notes">
        <Table rows={health?.coach_notes ?? []} columns={["created_at", "category", "note"]} />
      </Card>
    </>
  );
}
