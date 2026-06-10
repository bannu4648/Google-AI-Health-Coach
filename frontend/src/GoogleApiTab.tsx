import React, { useMemo, useState } from "react";
import { JsonBlock } from "./components/JsonBlock";
import { formatTimestamp, httpStatusClass } from "./lib/format";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type ApiRow = Record<string, any>;

const STAGE_COLORS: Record<string, string> = {
  AWAKE: "#f59e0b",
  LIGHT: "#60a5fa",
  DEEP: "#4f46e5",
  REM: "#a78bfa",
};

function formatCivilTime(civil?: { date?: { year: number; month: number; day: number }; time?: { hours?: number; minutes?: number } }): string {
  if (!civil?.date) return "—";
  const { year, month, day } = civil.date;
  const hours = civil.time?.hours ?? 0;
  const minutes = civil.time?.minutes ?? 0;
  return `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")} ${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")} HKT`;
}

function formatInterval(interval?: { startTime?: string; endTime?: string }): string {
  if (!interval?.startTime) return "—";
  const start = formatTimestamp(interval.startTime);
  const end = interval.endTime ? formatTimestamp(interval.endTime) : "";
  return end ? `${start} → ${end}` : start;
}

function durationMinutes(start?: string, end?: string): number | null {
  if (!start || !end) return null;
  const ms = new Date(end).getTime() - new Date(start).getTime();
  return ms > 0 ? Math.round(ms / 60000) : null;
}

function getPayloadKey(dataType?: string): string {
  if (!dataType) return "";
  return dataType.replace(/-([a-z])/g, (_, c: string) => c.toUpperCase());
}

function getDataPointBody(dp: ApiRow, dataType?: string): ApiRow {
  const key = getPayloadKey(dataType);
  return dp[key] ?? dp[dataType ?? ""] ?? {};
}

function DataPointsTable({ rows, columns }: { rows: ApiRow[]; columns: { key: string; label: string }[] }) {
  if (!rows.length) {
    return <p className="muted">No data points in this response.</p>;
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key}>{column.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={row.id ?? index}>
              {columns.map((column) => (
                <td key={column.key}>{row[column.key] ?? "—"}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function NutritionVisualization({ dataPoints }: { dataPoints: ApiRow[] }) {
  const rows = dataPoints.map((dp, index) => {
    const body = getDataPointBody(dp, "nutrition-log");
    const interval = body.interval ?? {};
    const civil = interval.civilStartTime ?? interval.civilEndTime;
    return {
      id: dp.name ?? index,
      time: formatCivilTime(civil) || formatInterval(interval),
      food: body.foodDisplayName ?? "—",
      meal: body.mealType ?? "—",
      kcal: body.energy?.kcal ?? "—",
      carbs: body.totalCarbohydrate?.grams ?? "—",
      protein: body.nutrients?.find((n: ApiRow) => n.nutrient === "PROTEIN")?.quantity?.grams ?? "—",
      fat: body.totalFat?.grams ?? "—",
    };
  });

  const chartData = rows
    .filter((row) => typeof row.kcal === "number")
    .map((row) => ({ name: String(row.food).slice(0, 18), kcal: row.kcal }));

  return (
    <div className="viz-stack">
      <DataPointsTable
        rows={rows}
        columns={[
          { key: "time", label: "Time (HKT)" },
          { key: "food", label: "Food" },
          { key: "meal", label: "Meal" },
          { key: "kcal", label: "kcal" },
          { key: "carbs", label: "Carbs (g)" },
          { key: "protein", label: "Protein (g)" },
          { key: "fat", label: "Fat (g)" },
        ]}
      />
      {chartData.length > 0 && (
        <div className="viz-chart">
          <h4>Calories by entry</h4>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" interval={0} angle={-20} textAnchor="end" height={70} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="kcal" fill="#4f46e5" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function ExerciseVisualization({ dataPoints }: { dataPoints: ApiRow[] }) {
  const rows = dataPoints.map((dp, index) => {
    const body = getDataPointBody(dp, "exercise");
    const interval = body.interval ?? {};
    const minutes = durationMinutes(interval.startTime, interval.endTime);
    const metrics = body.metricsSummary ?? {};
    return {
      id: dp.name ?? index,
      type: body.exerciseType ?? "—",
      time: formatInterval(interval),
      duration: minutes != null ? `${minutes} min` : "—",
      calories: metrics.caloriesKcal ?? "—",
      steps: metrics.steps ?? "—",
      avgHr: metrics.averageHeartRateBeatsPerMinute ?? "—",
      azm: metrics.activeZoneMinutes ?? "—",
    };
  });

  const chartData = rows
    .filter((row) => typeof row.calories === "number")
    .map((row) => ({ name: String(row.type).slice(0, 12), calories: row.calories }));

  return (
    <div className="viz-stack">
      <DataPointsTable
        rows={rows}
        columns={[
          { key: "type", label: "Type" },
          { key: "time", label: "Interval" },
          { key: "duration", label: "Duration" },
          { key: "calories", label: "Calories" },
          { key: "steps", label: "Steps" },
          { key: "avgHr", label: "Avg HR" },
          { key: "azm", label: "AZM" },
        ]}
      />
      {chartData.length > 0 && (
        <div className="viz-chart">
          <h4>Calories by workout</h4>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="calories" fill="#059669" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function SleepVisualization({ dataPoints }: { dataPoints: ApiRow[] }) {
  return (
    <div className="viz-stack">
      {dataPoints.map((dp, index) => {
        const body = getDataPointBody(dp, "sleep");
        const interval = body.interval ?? {};
        const totalMinutes = durationMinutes(interval.startTime, interval.endTime);
        const stages = (body.stages ?? []) as ApiRow[];
        const stageTotals = stages.reduce<Record<string, number>>((acc, stage) => {
          const mins = durationMinutes(stage.startTime, stage.endTime) ?? 0;
          const type = String(stage.type ?? "UNKNOWN");
          acc[type] = (acc[type] ?? 0) + mins;
          return acc;
        }, {});
        const chartData = Object.entries(stageTotals).map(([name, value]) => ({ name, value }));

        return (
          <div className="sleep-card" key={dp.dataPointName ?? dp.name ?? index}>
            <div className="sleep-card-header">
              <strong>{body.type ?? "Sleep"}</strong>
              <span>{formatInterval(interval)}</span>
              {totalMinutes != null && <span className="pill">{totalMinutes} min total</span>}
            </div>
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={chartData} dataKey="value" nameKey="name" innerRadius={50} outerRadius={80} paddingAngle={2}>
                    {chartData.map((entry) => (
                      <Cell key={entry.name} fill={STAGE_COLORS[entry.name] ?? "#94a3b8"} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value: number) => [`${value} min`, "Duration"]} />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <p className="muted">No stage breakdown available.</p>
            )}
          </div>
        );
      })}
    </div>
  );
}

function RollupVisualization({ rollupDataPoints, dataType }: { rollupDataPoints: ApiRow[]; dataType?: string }) {
  const rows = rollupDataPoints.map((point, index) => {
    const label = formatCivilTime(point.civilStartTime);
    let value: number | string = "—";
    if (point.steps?.countSum != null) value = Number(point.steps.countSum);
    else if (point.activeZoneMinutes?.minutesSum != null) value = Number(point.activeZoneMinutes.minutesSum);
    else if (point.weight?.weightGrams != null) value = Number(point.weight.weightGrams) / 1000;
    else {
      const numeric = Object.values(point).find((v) => typeof v === "object" && v && "countSum" in (v as ApiRow));
      if (numeric && typeof numeric === "object") value = Number((numeric as ApiRow).countSum ?? (numeric as ApiRow).minutesSum ?? 0);
    }
    return { id: index, label, value: typeof value === "number" ? value : 0, display: value };
  });

  const valueLabel =
    dataType === "steps" ? "Steps" : dataType === "active-zone-minutes" ? "Active zone minutes" : dataType === "weight" ? "Weight (kg)" : "Value";

  return (
    <div className="viz-stack">
      <DataPointsTable
        rows={rows.map((row) => ({ id: row.id, period: row.label, value: row.display }))}
        columns={[
          { key: "period", label: "Period start (HKT)" },
          { key: "value", label: valueLabel },
        ]}
      />
      {rows.some((row) => row.value > 0) && (
        <div className="viz-chart">
          <h4>{valueLabel} by period</h4>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={rows}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#0ea5e9" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function GenericDataPointsVisualization({ dataPoints, dataType }: { dataPoints: ApiRow[]; dataType?: string }) {
  const rows = dataPoints.map((dp, index) => {
    const body = getDataPointBody(dp, dataType);
    const interval = body.interval ?? {};
    const scalar =
      body.value ??
      body.count ??
      body.weightGrams ??
      body.volumeMilliliters ??
      body.beatsPerMinute ??
      Object.values(body).find((v) => typeof v === "number" || typeof v === "string");
    return {
      id: dp.name ?? dp.dataPointName ?? index,
      time: formatInterval(interval) || formatCivilTime(body.civilStartTime),
      value: typeof scalar === "object" ? JSON.stringify(scalar) : String(scalar ?? "—"),
      source: dp.dataSource?.platform ?? "—",
    };
  });

  return (
    <DataPointsTable
      rows={rows}
      columns={[
        { key: "time", label: "Time" },
        { key: "value", label: "Value" },
        { key: "source", label: "Source" },
      ]}
    />
  );
}

function ResponseVisualization({ call }: { call: ApiRow }) {
  const response = call.response ?? {};
  const dataType = call.data_type as string | undefined;

  if (call.error) {
    return <div className="error-inline">{call.error}</div>;
  }

  if (response.dataPoints?.length) {
    if (dataType === "nutrition-log") return <NutritionVisualization dataPoints={response.dataPoints} />;
    if (dataType === "exercise") return <ExerciseVisualization dataPoints={response.dataPoints} />;
    if (dataType === "sleep") return <SleepVisualization dataPoints={response.dataPoints} />;
    return <GenericDataPointsVisualization dataPoints={response.dataPoints} dataType={dataType} />;
  }

  if (response.rollupDataPoints?.length) {
    return <RollupVisualization rollupDataPoints={response.rollupDataPoints} dataType={dataType} />;
  }

  if (response.done != null || response.response) {
    return (
      <div className="viz-stack">
        <p className="muted">Write operation response</p>
        <JsonBlock value={response.response ?? response} />
      </div>
    );
  }

  if (response.error) {
    return <div className="error-inline">{JSON.stringify(response.error, null, 2)}</div>;
  }

  if (!Object.keys(response).length) {
    return <p className="muted">Empty response body.</p>;
  }

  return <JsonBlock value={response} />;
}

function CallDetail({ call }: { call: ApiRow }) {
  const [showRaw, setShowRaw] = useState(false);
  const request = call.request ?? {};

  return (
    <div className="call-detail">
      <div className="call-detail-header">
        <div className="call-badges">
          <span className="method-badge">{call.method}</span>
          <span className={httpStatusClass(call.status_code)}>{call.status_code ?? "error"}</span>
          {call.data_type && <span className="pill">{call.data_type}</span>}
          {call.latency_ms != null && <span className="pill muted-pill">{call.latency_ms} ms</span>}
        </div>
        <p className="call-url">{call.url}</p>
        <p className="muted">{formatTimestamp(call.created_at)}</p>
      </div>

      <section className="detail-section">
        <h3>Request</h3>
        {request.params && (
          <div className="request-block">
            <h4>Query params</h4>
            <JsonBlock value={request.params} />
          </div>
        )}
        {request.json && (
          <div className="request-block">
            <h4>JSON body</h4>
            <JsonBlock value={request.json} />
          </div>
        )}
        {!request.params && !request.json && <p className="muted">No request payload recorded.</p>}
      </section>

      <section className="detail-section">
        <h3>Response visualization</h3>
        <ResponseVisualization call={call} />
      </section>

      <section className="detail-section">
        <button className="ghost-button" onClick={() => setShowRaw((value) => !value)}>
          {showRaw ? "Hide raw JSON" : "Show raw JSON"}
        </button>
        {showRaw && (
          <div className="raw-grid">
            <div>
              <h4>Raw request</h4>
              <JsonBlock value={request} />
            </div>
            <div>
              <h4>Raw response</h4>
              <JsonBlock value={call.response ?? {}} />
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

export function GoogleApiTab({ calls }: { calls: ApiRow[] }) {
  const [selectedId, setSelectedId] = useState<string | null>(calls[0]?.id ?? null);
  const [dataTypeFilter, setDataTypeFilter] = useState("all");
  const [methodFilter, setMethodFilter] = useState("all");

  const dataTypes = useMemo(
    () => ["all", ...Array.from(new Set(calls.map((call) => call.data_type).filter(Boolean))).sort()],
    [calls],
  );

  const filteredCalls = useMemo(
    () =>
      calls.filter((call) => {
        if (dataTypeFilter !== "all" && call.data_type !== dataTypeFilter) return false;
        if (methodFilter !== "all" && call.method !== methodFilter) return false;
        return true;
      }),
    [calls, dataTypeFilter, methodFilter],
  );

  const selectedCall = filteredCalls.find((call) => call.id === selectedId) ?? filteredCalls[0] ?? null;

  return (
    <div className="google-api-tab">
      <div className="google-api-layout">
        <aside className="call-list-panel">
          <div className="filters">
            <label>
              Data type
              <select value={dataTypeFilter} onChange={(event) => setDataTypeFilter(event.target.value)}>
                {dataTypes.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Method
              <select value={methodFilter} onChange={(event) => setMethodFilter(event.target.value)}>
                <option value="all">all</option>
                <option value="GET">GET</option>
                <option value="POST">POST</option>
                <option value="PATCH">PATCH</option>
              </select>
            </label>
          </div>

          <p className="muted call-count">{filteredCalls.length} calls</p>

          <div className="call-list">
            {filteredCalls.map((call) => (
              <button
                key={call.id}
                className={`call-item ${selectedCall?.id === call.id ? "active" : ""}`}
                onClick={() => setSelectedId(call.id)}
              >
                <div className="call-item-top">
                  <span className="method-badge small">{call.method}</span>
                  <span className={httpStatusClass(call.status_code)}>{call.status_code ?? "err"}</span>
                </div>
                <strong>{call.data_type ?? "unknown"}</strong>
                <span className="muted">{formatTimestamp(call.created_at)}</span>
                <span className="muted">{call.latency_ms ?? "—"} ms</span>
              </button>
            ))}
            {!filteredCalls.length && <p className="muted">No calls match these filters.</p>}
          </div>
        </aside>

        <div className="call-detail-panel">
          {selectedCall ? <CallDetail call={selectedCall} /> : <p className="muted">Select a Google API call to inspect request and response.</p>}
        </div>
      </div>
    </div>
  );
}
