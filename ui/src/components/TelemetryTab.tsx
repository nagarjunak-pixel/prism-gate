import { useState } from "react";
import { Activity, Cpu, DollarSign, Gauge } from "lucide-react";
import type { BenchmarkSummary, GatewayOverview } from "../types";

interface Props {
  overview: GatewayOverview;
  onRefresh: () => void;
}

export function TelemetryTab({ overview, onRefresh }: Props) {
  const [runningLoad, setRunningLoad] = useState(false);
  const [lastBench, setLastBench] = useState<BenchmarkSummary | null>(null);

  async function runBurst() {
    setRunningLoad(true);
    try {
      const resp = await fetch("/api/v1/benchmarks/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          requests: 200,
          concurrency: 28,
          inject_congestion_spike: true,
        }),
      });
      if (resp.ok) {
        const summary = (await resp.json()) as BenchmarkSummary;
        setLastBench(summary);
        onRefresh();
      }
    } catch {
      // Standalone static build fallback
    } finally {
      setRunningLoad(false);
    }
  }

  const utilPct = Math.min(100, overview.gateway_utilization * 100);

  return (
    <div>
      <div className="grid-4">
        <div className="card">
          <div className="kpi-label">FinOps Net Cost Reduction</div>
          <div className="kpi-value" style={{ color: "#34d399" }}>
            {overview.finops.savings_pct.toFixed(1)}%
          </div>
          <div className="kpi-sub">
            ${overview.finops.actual_total_cost_usd.toFixed(4)} actual vs $
            {overview.finops.counterfactual_frontier_cost_usd.toFixed(4)} static frontier
          </div>
        </div>

        <div className="card">
          <div className="kpi-label">Signed-Hash Cache Hit Rate</div>
          <div className="kpi-value" style={{ color: "#22d3ee" }}>
            {(overview.finops.cache_hit_rate * 100).toFixed(1)}%
          </div>
          <div className="kpi-sub">
            {overview.finops.semantic_cache_hits} tenant-isolated hits /{" "}
            {overview.finops.total_routed_requests} routed
          </div>
        </div>

        <div className="card">
          <div className="kpi-label">Gateway Utilization (85% Shed Floor)</div>
          <div className="kpi-value">
            {utilPct.toFixed(1)}%{" "}
            <span style={{ fontSize: "0.9rem", color: "#94a3b8" }}>
              ({overview.total_inflight}/{overview.total_capacity})
            </span>
          </div>
          <div className="progress-track" style={{ marginTop: 8 }}>
            <div
              className="progress-fill"
              style={{
                width: `${utilPct}%`,
                background: utilPct >= 85 ? "#fb7185" : "#22d3ee",
              }}
            />
          </div>
        </div>

        <div className="card">
          <div className="kpi-label">End-to-End Tail Latency</div>
          <div className="kpi-value">{overview.latency_p99_ms.toFixed(1)} ms</div>
          <div className="kpi-sub">
            p50: {overview.latency_p50_ms.toFixed(1)} ms • p95:{" "}
            {overview.latency_p95_ms.toFixed(1)} ms
          </div>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 18 }}>
        <div className="card-title">
          <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
            <Gauge size={17} /> TCP-Vegas / Netflix Gradient2 Adaptive Concurrency Control
          </span>
          <button
            type="button"
            className="btn-primary"
            disabled={runningLoad}
            onClick={() => void runBurst()}
          >
            <Activity size={14} />{" "}
            {runningLoad ? "Running 200-Req Burst..." : "Trigger 200-Req Concurrent Burst"}
          </button>
        </div>

        {lastBench && (
          <div
            style={{
              marginBottom: 14,
              padding: "10px 14px",
              borderRadius: 8,
              background: "rgba(34, 211, 238, 0.08)",
              border: "1px solid rgba(34, 211, 238, 0.3)",
              fontSize: "0.83rem",
            }}
          >
            Latest Burst: <strong>{lastBench.throughput_rps.toFixed(0)} req/s</strong> • Overhead
            p50/p99:{" "}
            <strong>
              {lastBench.gateway_overhead_p50_ms.toFixed(3)} /{" "}
              {lastBench.gateway_overhead_p99_ms.toFixed(3)} ms
            </strong>{" "}
            • CRITICAL Preserved:{" "}
            <strong>
              {lastBench.critical_admitted}/{lastBench.critical_total}
            </strong>{" "}
            • FinOps Savings: <strong>{lastBench.finops.savings_pct.toFixed(1)}%</strong>
          </div>
        )}

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Upstream Endpoint</th>
                <th>Tier</th>
                <th>Inflight / Vegas Ceiling</th>
                <th>RTT Min / EWMA</th>
                <th>Gradient</th>
                <th>Queue Est.</th>
                <th>Requests / Shed</th>
                <th>Price (1K In/Out)</th>
              </tr>
            </thead>
            <tbody>
              {overview.endpoints.map((ep) => {
                const capRatio = Math.min(100, (ep.max_inflight / ep.max_limit) * 100);
                return (
                  <tr key={ep.provider_id}>
                    <td style={{ fontWeight: 600 }}>{ep.provider_id}</td>
                    <td>
                      <span className={ep.tier === "FAST" ? "pill pill-cyan" : "pill"}>
                        {ep.tier}
                      </span>
                    </td>
                    <td style={{ minWidth: 170 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                        <span>
                          {ep.inflight} / <strong>{ep.max_inflight}</strong>
                        </span>
                        <span style={{ color: "#64748b", fontSize: "0.75rem" }}>
                          max {ep.max_limit}
                        </span>
                      </div>
                      <div className="progress-track">
                        <div
                          className="progress-fill"
                          style={{ width: `${capRatio}%`, background: "#818cf8" }}
                        />
                      </div>
                    </td>
                    <td>
                      {ep.rtt_min_ms.toFixed(1)} / {ep.rtt_ewma_ms.toFixed(1)} ms
                    </td>
                    <td>{ep.gradient.toFixed(3)}</td>
                    <td>{ep.queue_estimate.toFixed(2)}</td>
                    <td>
                      {ep.total_requests} /{" "}
                      <span style={{ color: ep.total_shed > 0 ? "#fbbf24" : "#94a3b8" }}>
                        {ep.total_shed}
                      </span>
                    </td>
                    <td>
                      ${ep.cost_per_1k_input_usd.toFixed(5)} / ${ep.cost_per_1k_output_usd.toFixed(4)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-title">
            <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
              <DollarSign size={16} /> Speculative Cascade Routing Mix
            </span>
          </div>
          <table>
            <tbody>
              <tr>
                <td>Fast Tier Direct Completions (`gemini-2.5-flash` / `vllm-qwen-32b`)</td>
                <td>{overview.finops.fast_tier_hits}</td>
              </tr>
              <tr>
                <td>Semantic Cache Instant Completions (`$0.00` upstream spend)</td>
                <td>{overview.finops.semantic_cache_hits}</td>
              </tr>
              <tr>
                <td>Speculative Fast → Frontier Escalations</td>
                <td>{overview.finops.cascade_escalations}</td>
              </tr>
              <tr>
                <td>Direct High-Complexity Frontier Routes</td>
                <td>{overview.finops.frontier_direct_hits}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="card">
          <div className="card-title">
            <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
              <Cpu size={16} /> Hierarchical Admission & Priority Shedding
            </span>
          </div>
          <table>
            <tbody>
              <tr>
                <td>Admitted Requests</td>
                <td style={{ color: "#34d399", fontWeight: 600 }}>
                  {overview.admission.admitted_total}
                </td>
              </tr>
              <tr>
                <td>Priority Load Shed (`BATCH_BACKGROUND` @ ≥85% util)</td>
                <td>{overview.admission.shed_batch_total}</td>
              </tr>
              <tr>
                <td>Priority Load Shed (`INTERACTIVE` @ ≥95% util)</td>
                <td>{overview.admission.shed_interactive_total}</td>
              </tr>
              <tr>
                <td>Tenant Dual Token-Bucket Rejections (RPS / TPM)</td>
                <td>
                  {overview.admission.rejected_rps_total} / {overview.admission.rejected_tpm_total}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
