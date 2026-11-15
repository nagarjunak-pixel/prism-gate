import { useCallback, useEffect, useState } from "react";
import { Activity, Layers, MessageSquareCode, Radio, ShieldCheck } from "lucide-react";
import { DriftLabTab } from "./components/DriftLabTab";
import { FaithfulnessTab } from "./components/FaithfulnessTab";
import { PlaygroundTab } from "./components/PlaygroundTab";
import { TelemetryTab } from "./components/TelemetryTab";
import { INITIAL_OVERVIEW, type GatewayOverview } from "./types";

type TabKey = "playground" | "telemetry" | "faithfulness" | "drift";

export function App() {
  const [activeTab, setActiveTab] = useState<TabKey>("playground");
  const [overview, setOverview] = useState<GatewayOverview>(INITIAL_OVERVIEW);
  const [connected, setConnected] = useState(false);

  const refreshOverview = useCallback(async () => {
    try {
      const resp = await fetch("/api/v1/overview");
      if (resp.ok) {
        const data = (await resp.json()) as GatewayOverview;
        setOverview(data);
        setConnected(true);
      }
    } catch {
      setConnected(false);
    }
  }, []);

  useEffect(() => {
    void refreshOverview();
    const timer = window.setInterval(() => {
      void refreshOverview();
    }, 5000);
    return () => window.clearInterval(timer);
  }, [refreshOverview]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-group">
          <div className="brand-badge">
            <Layers size={22} />
          </div>
          <div>
            <h1 className="brand-title">PrismGate Control Plane</h1>
            <p className="brand-subtitle">
              Multi-Tenant LLM Inference Gateway • TCP-Vegas Concurrency • Speculative Cascade •
              Claim Judge & Drift Lab
            </p>
          </div>
        </div>

        <div className="status-strip">
          <span className={connected ? "pill pill-emerald" : "pill pill-amber"}>
            ● {connected ? `Gateway Live v${overview.version}` : "Standalone Preview"}
          </span>
          <span className="pill pill-cyan">
            FinOps Saved: {overview.finops.savings_pct.toFixed(1)}%
          </span>
          <span className="pill">p99: {overview.latency_p99_ms.toFixed(1)} ms</span>
        </div>
      </header>

      <nav className="tab-nav" aria-label="Control Plane Views">
        <button
          type="button"
          className={activeTab === "playground" ? "tab-btn active" : "tab-btn"}
          onClick={() => setActiveTab("playground")}
        >
          <MessageSquareCode size={16} /> Live OpenAI Chat / SSE Playground
        </button>
        <button
          type="button"
          className={activeTab === "telemetry" ? "tab-btn active" : "tab-btn"}
          onClick={() => setActiveTab("telemetry")}
        >
          <Activity size={16} /> Vegas Concurrency & Cascade FinOps
        </button>
        <button
          type="button"
          className={activeTab === "faithfulness" ? "tab-btn active" : "tab-btn"}
          onClick={() => setActiveTab("faithfulness")}
        >
          <ShieldCheck size={16} /> Claim-Faithfulness Inspector
        </button>
        <button
          type="button"
          className={activeTab === "drift" ? "tab-btn active" : "tab-btn"}
          onClick={() => setActiveTab("drift")}
        >
          <Radio size={16} /> PSI & Jensen-Shannon Drift Lab
        </button>
      </nav>

      <main>
        {activeTab === "playground" && <PlaygroundTab onTelemetryRefresh={() => void refreshOverview()} />}
        {activeTab === "telemetry" && (
          <TelemetryTab overview={overview} onRefresh={() => void refreshOverview()} />
        )}
        {activeTab === "faithfulness" && <FaithfulnessTab />}
        {activeTab === "drift" && <DriftLabTab initialDrift={overview.latest_drift} />}
      </main>
    </div>
  );
}
