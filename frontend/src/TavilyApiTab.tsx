import React, { useMemo, useState } from "react";
import { JsonBlock } from "./components/JsonBlock";
import { formatTimestamp, tavilyStatusClass } from "./lib/format";

type ApiRow = Record<string, any>;

function ResultsTable({ results }: { results: ApiRow[] }) {
  if (!results.length) {
    return <p className="muted">No search results returned.</p>;
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Title</th>
            <th>URL</th>
            <th>Score</th>
            <th>Snippet</th>
          </tr>
        </thead>
        <tbody>
          {results.map((result, index) => (
            <tr key={result.url ?? index}>
              <td>{result.title ?? "—"}</td>
              <td>
                {result.url ? (
                  <a href={result.url} target="_blank" rel="noreferrer">
                    {result.url}
                  </a>
                ) : (
                  "—"
                )}
              </td>
              <td>{result.score ?? "—"}</td>
              <td>{result.content ? String(result.content).slice(0, 280) : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CallDetail({ call }: { call: ApiRow }) {
  const [showRaw, setShowRaw] = useState(false);
  const response = call.response ?? {};
  const request = call.request ?? {};
  const results = (response.results ?? []) as ApiRow[];

  return (
    <div className="call-detail">
      <div className="call-detail-header">
        <div className="call-badges">
          <span className="method-badge">SEARCH</span>
          <span className={tavilyStatusClass(call.status)}>{call.status ?? "unknown"}</span>
          {call.result_count != null && <span className="pill">{call.result_count} results</span>}
          {call.latency_ms != null && <span className="pill muted-pill">{call.latency_ms} ms</span>}
        </div>
        <p className="call-url">{call.query}</p>
        <p className="muted">
          {call.food_display_name ?? "unknown food"}
          {call.portion_description ? ` · ${call.portion_description}` : ""}
        </p>
        <p className="muted">{formatTimestamp(call.created_at)}</p>
      </div>

      {call.error && <div className="error-inline">{call.error}</div>}

      <section className="detail-section">
        <h3>Request</h3>
        <JsonBlock
          value={{
            query: call.query,
            food_display_name: call.food_display_name,
            portion_description: call.portion_description,
            ...request,
          }}
        />
      </section>

      <section className="detail-section">
        <h3>Tavily answer</h3>
        {response.answer ? <p>{response.answer}</p> : <p className="muted">No synthesized answer returned.</p>}
      </section>

      <section className="detail-section">
        <h3>Search results</h3>
        <ResultsTable results={results} />
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
              <JsonBlock value={response} />
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

export function TavilyApiTab({ calls }: { calls: ApiRow[] }) {
  const [selectedId, setSelectedId] = useState<string | null>(calls[0]?.id ?? null);
  const [statusFilter, setStatusFilter] = useState("all");

  const statuses = useMemo(
    () => ["all", ...Array.from(new Set(calls.map((call) => call.status).filter(Boolean))).sort()],
    [calls],
  );

  const filteredCalls = useMemo(
    () =>
      calls.filter((call) => {
        if (statusFilter !== "all" && call.status !== statusFilter) return false;
        return true;
      }),
    [calls, statusFilter],
  );

  const selectedCall = filteredCalls.find((call) => call.id === selectedId) ?? filteredCalls[0] ?? null;

  return (
    <div className="google-api-tab">
      <div className="google-api-layout">
        <aside className="call-list-panel">
          <div className="filters">
            <label>
              Status
              <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
                {statuses.map((status) => (
                  <option key={status} value={status}>
                    {status}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <p className="muted call-count">{filteredCalls.length} Tavily searches</p>

          <div className="call-list">
            {filteredCalls.map((call) => (
              <button
                key={call.id}
                className={`call-item ${selectedCall?.id === call.id ? "active" : ""}`}
                onClick={() => setSelectedId(call.id)}
              >
                <div className="call-item-top">
                  <span className="method-badge small">SEARCH</span>
                  <span className={tavilyStatusClass(call.status)}>{call.status ?? "unknown"}</span>
                </div>
                <strong>{call.food_display_name ?? "nutrition lookup"}</strong>
                <span className="muted">{String(call.query ?? "").slice(0, 72)}</span>
                <span className="muted">{formatTimestamp(call.created_at)}</span>
                <span className="muted">
                  {call.result_count ?? 0} results · {call.latency_ms ?? "—"} ms
                </span>
              </button>
            ))}
            {!filteredCalls.length && <p className="muted">No Tavily calls match these filters.</p>}
          </div>
        </aside>

        <div className="call-detail-panel">
          {selectedCall ? (
            <CallDetail call={selectedCall} />
          ) : (
            <p className="muted">Select a Tavily search to inspect the query and results.</p>
          )}
        </div>
      </div>
    </div>
  );
}
