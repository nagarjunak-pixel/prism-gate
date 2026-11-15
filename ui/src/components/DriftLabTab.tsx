import { useState } from "react";
import { BarChart3, Radio } from "lucide-react";
import type { DriftReport, DriftSeverity } from "../types";

interface Props {
  initialDrift: DriftReport;
}

type PresetName = "normal" | "moderate" | "critical" | "live_window";

function severityBadge(sev: DriftSeverity): string {
  if (sev === "CRITICAL_DRIFT") return "pill pill-rose";
  if (sev === "MODERATE_DRIFT") return "pill pill-amber";
  return "pill pill-emerald";
}

export function DriftLabTab({ initialDrift }: Props) {
  const [report, setReport] = useState<DriftReport>(initialDrift);
  const [activePreset, setActivePreset] = useState<PresetName>("normal");
  const [loading, setLoading] = useState(false);

  async function selectPreset(preset: PresetName) {
    setActivePreset(preset);
    setLoading(true);
    try {
      const resp = await fetch("/api/v1/drift/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preset }),
      });
      if (resp.ok) {
        setReport((await resp.json()) as DriftReport);
      }
    } catch {
      // Offline preview fallback
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="card" style={{ marginBottom: 18 }}>
        <div className="card-title">
          <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
            <Radio size={17} /> Statistical Distribution-Shift Simulator (PSI + Base-2 Jensen-Shannon)
          </span>
          <span className={severityBadge(report.overall_severity)}>
            {loading ? "Evaluating..." : report.overall_severity}
          </span>
        </div>

        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 16 }}>
          {(
            [
              ["normal", "Baseline Healthy Traffic (NO_DRIFT)"],
              ["moderate", "28% Code Cohort Shift (MODERATE_DRIFT)"],
              ["critical", "Adversarial Jailbreak Surge (CRITICAL_DRIFT)"],
              ["live_window", "Live Gateway Sliding Window"],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              type="button"
              className={activePreset === key ? "btn-primary" : "btn-ghost"}
              onClick={() => void selectPreset(key)}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="grid-4" style={{ marginBottom: 0 }}>
          <div className="card" style={{ background: "#080e1c" }}>
            <div className="kpi-label">Max Feature PSI</div>
            <div className="kpi-value">{report.max_psi.toFixed(4)}</div>
            <div className="kpi-sub">Thresholds: 0.10 Moderate • 0.25 Critical</div>
          </div>

          <div className="card" style={{ background: "#080e1c" }}>
            <div className="kpi-label">Token Stream JSD (base-2)</div>
            <div className="kpi-value" style={{ color: "#22d3ee" }}>
              {report.token_jsd.toFixed(4)} bits
            </div>
            <div className="kpi-sub">Bounded strictly in [0.0000, 1.0000]</div>
          </div>

          <div className="card" style={{ background: "#080e1c" }}>
            <div className="kpi-label">Topic Mix JSD (base-2)</div>
            <div className="kpi-value" style={{ color: "#818cf8" }}>
              {report.topic_jsd.toFixed(4)} bits
            </div>
            <div className="kpi-sub">Symmetric: JSD(P || Q) == JSD(Q || P)</div>
          </div>

          <div className="card" style={{ background: "#080e1c" }}>
            <div className="kpi-label">Sample Windows</div>
            <div className="kpi-value">
              {report.baseline_sample_size} / {report.live_sample_size}
            </div>
            <div className="kpi-sub">Reference Baseline vs Live Window</div>
          </div>
        </div>

        {report.alert_reasons.length > 0 && (
          <div
            style={{
              marginTop: 14,
              padding: "10px 14px",
              borderRadius: 8,
              background: "rgba(251, 113, 133, 0.1)",
              border: "1px solid rgba(251, 113, 133, 0.35)",
              fontSize: "0.82rem",
              color: "#fecdd3",
            }}
          >
            <strong>Triggered Drift Signals:</strong> {report.alert_reasons.join(" • ")}
          </div>
        )}
      </div>

      <div className="grid-4" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))" }}>
        {report.feature_psi.map((feat) => {
          const maxProp = Math.max(
            0.25,
            ...feat.baseline_proportions,
            ...feat.live_proportions
          );
          return (
            <div className="card" key={feat.feature_name}>
              <div className="card-title">
                <span style={{ display: "inline-flex", alignItems: "center", gap: 7 }}>
                  <BarChart3 size={15} /> {feat.feature_name}
                </span>
                <span className={severityBadge(feat.severity)}>
                  PSI = {feat.psi_value.toFixed(4)}
                </span>
              </div>

              <svg
                viewBox="0 0 320 130"
                style={{ width: "100%", height: 130, display: "block" }}
                role="img"
                aria-label={`${feat.feature_name} baseline vs live histogram`}
              >
                {feat.baseline_proportions.map((baseP, idx) => {
                  const liveP = feat.live_proportions[idx] ?? 0;
                  const groupW = 300 / feat.baseline_proportions.length;
                  const x0 = 10 + idx * groupW;
                  const baseH = (baseP / maxProp) * 98;
                  const liveH = (liveP / maxProp) * 98;
                  return (
                    <g key={idx}>
                      <rect
                        x={x0 + 2}
                        y={110 - baseH}
                        width={Math.max(4, groupW * 0.38)}
                        height={baseH}
                        rx={2}
                        fill="#475569"
                      />
                      <rect
                        x={x0 + 3 + groupW * 0.38}
                        y={110 - liveH}
                        width={Math.max(4, groupW * 0.38)}
                        height={liveH}
                        rx={2}
                        fill={feat.severity === "CRITICAL_DRIFT" ? "#fb7185" : "#22d3ee"}
                      />
                    </g>
                  );
                })}
                <line x1="8" y1="111" x2="312" y2="111" stroke="#1e2d4a" strokeWidth="1" />
              </svg>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: "0.75rem",
                  color: "#64748b",
                  marginTop: 6,
                }}
              >
                <span>■ Slate: Reference Deciles</span>
                <span>■ Cyan/Rose: Live Sliding Window</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
