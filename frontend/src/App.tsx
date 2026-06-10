import { useEffect, useState } from "react";
import { OverviewTab } from "./tabs/OverviewTab";
import { TechnicalTab } from "./tabs/TechnicalTab";
import { getJson, type ApiRow } from "./lib/api";

type TabId = "overview" | "technical";

export function App() {
  const [activeTab, setActiveTab] = useState<TabId>("overview");
  const [health, setHealth] = useState<ApiRow | null>(null);
  const [trends, setTrends] = useState<ApiRow[]>([]);
  const [summary, setSummary] = useState<ApiRow | null>(null);
  const [messages, setMessages] = useState<ApiRow[]>([]);
  const [llmCalls, setLlmCalls] = useState<ApiRow[]>([]);
  const [googleCalls, setGoogleCalls] = useState<ApiRow[]>([]);
  const [tavilyCalls, setTavilyCalls] = useState<ApiRow[]>([]);
  const [actions, setActions] = useState<ApiRow[]>([]);
  const [jobs, setJobs] = useState<ApiRow[]>([]);
  const [metrics, setMetrics] = useState<ApiRow | null>(null);
  const [error, setError] = useState<string>("");

  async function loadHealth() {
    const [healthData, trendsData] = await Promise.all([
      getJson<ApiRow>("/api/health/overview"),
      getJson<{ items: ApiRow[] }>("/api/health/trends?days=14"),
    ]);
    setHealth(healthData);
    setTrends(trendsData.items);
  }

  async function loadTechnical() {
    const [
      summaryData,
      messagesData,
      llmData,
      googleData,
      tavilyData,
      actionData,
      jobData,
      metricData,
    ] = await Promise.all([
      getJson<ApiRow>("/api/technical/summary"),
      getJson<{ items: ApiRow[] }>("/api/messages?limit=100"),
      getJson<{ items: ApiRow[] }>("/api/llm-calls?limit=100"),
      getJson<{ items: ApiRow[] }>("/api/google-health-calls?limit=100"),
      getJson<{ items: ApiRow[] }>("/api/tavily-calls?limit=100"),
      getJson<{ items: ApiRow[] }>("/api/health-actions?limit=100"),
      getJson<{ items: ApiRow[] }>("/api/job-runs?limit=50"),
      getJson<ApiRow>("/api/metrics/ranges"),
    ]);
    setSummary(summaryData);
    setMessages(messagesData.items);
    setLlmCalls(llmData.items);
    setGoogleCalls(googleData.items);
    setTavilyCalls(tavilyData.items);
    setActions(actionData.items);
    setJobs(jobData.items);
    setMetrics(metricData);
  }

  async function load() {
    try {
      await loadHealth();
      if (activeTab === "technical") {
        await loadTechnical();
      }
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  useEffect(() => {
    load();
    const intervalMs = activeTab === "technical" ? 15000 : 60000;
    const timer = window.setInterval(load, intervalMs);
    return () => window.clearInterval(timer);
  }, [activeTab]);

  useEffect(() => {
    if (activeTab === "technical" && summary == null) {
      loadTechnical().catch((err) => setError(err instanceof Error ? err.message : String(err)));
    }
  }, [activeTab]);

  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">Local WhatsApp AI Health Coach</p>
          <h1>Health Coach Dashboard</h1>
        </div>
        <button onClick={load}>Refresh</button>
      </header>

      {error && <div className="error">Dashboard API error: {error}</div>}

      <nav className="tabs">
        <button className={activeTab === "overview" ? "tab active" : "tab"} onClick={() => setActiveTab("overview")}>
          Health Overview
        </button>
        <button className={activeTab === "technical" ? "tab active" : "tab"} onClick={() => setActiveTab("technical")}>
          Technical Details
        </button>
      </nav>

      {activeTab === "overview" ? (
        <OverviewTab health={health} trends={trends} />
      ) : (
        <TechnicalTab
          summary={summary}
          messages={messages}
          llmCalls={llmCalls}
          googleCalls={googleCalls}
          tavilyCalls={tavilyCalls}
          actions={actions}
          jobs={jobs}
          metrics={metrics}
        />
      )}
    </main>
  );
}
