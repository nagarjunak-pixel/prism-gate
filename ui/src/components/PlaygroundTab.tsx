import { useState } from "react";
import { Play, Sparkles, Zap } from "lucide-react";
import type { GatewayTelemetryMeta, PriorityTier } from "../types";

interface Props {
  onTelemetryRefresh: () => void;
}

interface Preset {
  label: string;
  prompt: string;
  rag: string;
  priority: PriorityTier;
}

const PRESETS: Preset[] = [
  {
    label: "Fast Lookup (FAQ)",
    prompt: "What are the standard enterprise support hours for EMEA customers?",
    rag: "",
    priority: "INTERACTIVE",
  },
  {
    label: "Semantic Cache Paraphrase",
    prompt: "Could you tell me the standard EMEA customer enterprise support hours?",
    rag: "",
    priority: "INTERACTIVE",
  },
  {
    label: "Cascade Escalation (RAG)",
    prompt: "Reconcile ambiguous multi-quarter margin variance from the filing.",
    rag: "Acme Corp reported Q3 2025 revenue of $12.4M with operating margin of 18.5%.\nEnterprise cloud subscriptions grew 22.0% year-over-year under GAAP.",
    priority: "CRITICAL",
  },
  {
    label: "Frontier Theorem & Code",
    prompt:
      "Derive step by step the Little's Law queue stability bound, prove convergence invariants, and provide an async Python class with strict JSON schema.",
    rag: "",
    priority: "CRITICAL",
  },
];

export function PlaygroundTab({ onTelemetryRefresh }: Props) {
  const [tenantId, setTenantId] = useState("tenant-enterprise-01");
  const [priority, setPriority] = useState<PriorityTier>("INTERACTIVE");
  const [model, setModel] = useState("prism-auto");
  const [streamSse, setStreamSse] = useState(true);
  const [bypassCache, setBypassCache] = useState(false);
  const [prompt, setPrompt] = useState(PRESETS[0].prompt);
  const [ragText, setRagText] = useState("");
  const [output, setOutput] = useState(
    "Ready. Dispatch an OpenAI-compatible request via SSE streaming or synchronous JSON."
  );
  const [meta, setMeta] = useState<GatewayTelemetryMeta | null>(null);
  const [loading, setLoading] = useState(false);
  const [chunkCount, setChunkCount] = useState(0);

  async function handleSend() {
    if (!prompt.trim()) return;
    setLoading(true);
    setOutput("");
    setChunkCount(0);
    setMeta(null);

    const ragContext = ragText
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);

    const payload = {
      model,
      messages: [{ role: "user", content: prompt }],
      stream: streamSse,
      tenant_id: tenantId,
      priority,
      rag_context: ragContext,
      bypass_cache: bypassCache,
    };

    try {
      const resp = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        const errBody = await resp.text();
        setOutput(`HTTP ${resp.status} Rejected:\n${errBody}`);
        onTelemetryRefresh();
        return;
      }

      if (!streamSse) {
        const data = (await resp.json()) as {
          choices: Array<{ message: { content: string } }>;
          prism_meta: GatewayTelemetryMeta;
        };
        setOutput(data.choices[0]?.message?.content ?? "");
        setMeta(data.prism_meta);
        setChunkCount(1);
        onTelemetryRefresh();
        return;
      }

      const reader = resp.body?.getReader();
      if (!reader) {
        setOutput("ReadableStream unavailable in browser.");
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";
      let count = 0;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";

        for (const evt of events) {
          const line = evt.trim();
          if (!line.startsWith("data: ")) continue;
          const rawJson = line.slice(6).trim();
          if (rawJson === "[DONE]") continue;
          const parsed = JSON.parse(rawJson) as {
            choices?: Array<{ delta?: { content?: string } }>;
            prism_meta?: GatewayTelemetryMeta;
          };
          const piece = parsed.choices?.[0]?.delta?.content;
          if (piece) {
            count += 1;
            setChunkCount(count);
            setOutput((prev) => prev + piece);
          }
          if (parsed.prism_meta) {
            setMeta(parsed.prism_meta);
          }
        }
      }
      onTelemetryRefresh();
    } catch {
      // Offline preview fallback when standalone Vite bundle is viewed without backend
      setOutput(
        "[Offline Demo Mode] Start `prism serve` on :8000 for live SSE streaming.\n" +
          "Simulated Fast-Tier routing -> vllm-qwen-32b (14.2 ms RTT, $0.000198 saved vs static frontier)."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid-2">
      <div className="card">
        <div className="card-title">
          <span>OpenAI `/v1/chat/completions` Request Composer</span>
          <span className="pill pill-cyan">
            <Zap size={13} /> {streamSse ? "SSE text/event-stream" : "Sync JSON"}
          </span>
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
          {PRESETS.map((p) => (
            <button
              key={p.label}
              type="button"
              className="btn-ghost"
              onClick={() => {
                setPrompt(p.prompt);
                setRagText(p.rag);
                setPriority(p.priority);
              }}
            >
              <Sparkles size={12} /> {p.label}
            </button>
          ))}
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gap: 10,
            marginBottom: 12,
          }}
        >
          <div>
            <label className="input-label">Tenant Namespace</label>
            <select
              className="select-box"
              value={tenantId}
              onChange={(e) => setTenantId(e.target.value)}
            >
              <option value="tenant-enterprise-01">tenant-enterprise-01</option>
              <option value="tenant-saas-prod">tenant-saas-prod</option>
              <option value="tenant-batch-analytics">tenant-batch-analytics</option>
            </select>
          </div>
          <div>
            <label className="input-label">Priority Tier</label>
            <select
              className="select-box"
              value={priority}
              onChange={(e) => setPriority(e.target.value as PriorityTier)}
            >
              <option value="CRITICAL">CRITICAL</option>
              <option value="INTERACTIVE">INTERACTIVE</option>
              <option value="BATCH_BACKGROUND">BATCH_BACKGROUND</option>
            </select>
          </div>
          <div>
            <label className="input-label">Routing Policy</label>
            <select
              className="select-box"
              value={model}
              onChange={(e) => setModel(e.target.value)}
            >
              <option value="prism-auto">prism-auto (Cascade)</option>
              <option value="gemini-2.5-flash">gemini-2.5-flash</option>
              <option value="vllm-qwen-32b">vllm-qwen-32b</option>
              <option value="gemini-2.5-pro">gemini-2.5-pro</option>
              <option value="claude-sonnet-4">claude-sonnet-4</option>
            </select>
          </div>
        </div>

        <div style={{ marginBottom: 12 }}>
          <label className="input-label">User Prompt</label>
          <textarea
            className="textarea-box"
            rows={4}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </div>

        <div style={{ marginBottom: 14 }}>
          <label className="input-label">
            Optional Retrieved RAG Context Chunks (1 per line — triggers online claim judge)
          </label>
          <textarea
            className="textarea-box"
            rows={3}
            placeholder="Paste authoritative RAG filing / clinical guidelines..."
            value={ragText}
            onChange={(e) => setRagText(e.target.value)}
          />
        </div>

        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: 12,
          }}
        >
          <div style={{ display: "flex", gap: 16, fontSize: "0.82rem", color: "#94a3b8" }}>
            <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={streamSse}
                onChange={(e) => setStreamSse(e.target.checked)}
              />
              SSE Token Stream
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={bypassCache}
                onChange={(e) => setBypassCache(e.target.checked)}
              />
              Bypass Semantic Cache
            </label>
          </div>

          <button
            type="button"
            className="btn-primary"
            disabled={loading}
            onClick={() => void handleSend()}
          >
            <Play size={14} /> {loading ? "Streaming..." : "Dispatch Request"}
          </button>
        </div>
      </div>

      <div className="card">
        <div className="card-title">
          <span>Live Completion Stream & Control-Plane Decision</span>
          {chunkCount > 0 && <span className="pill">{chunkCount} SSE chunks</span>}
        </div>

        <div className="mono-output">{output}</div>

        {meta && (
          <div style={{ marginTop: 16 }}>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
              <span className="pill pill-cyan">Model: {meta.selected_model}</span>
              <span className="pill">Tier: {meta.selected_tier}</span>
              {meta.cache_hit && (
                <span className="pill pill-emerald">
                  SEMANTIC CACHE HIT ({(meta.cache_similarity * 100).toFixed(1)}%)
                </span>
              )}
              {meta.escalated_from_fast && (
                <span className="pill pill-amber">CASCADE ESCALATED</span>
              )}
              {meta.faithfulness_score !== null && (
                <span className={meta.hallucination_detected ? "pill pill-rose" : "pill pill-emerald"}>
                  Faithfulness: {(meta.faithfulness_score * 100).toFixed(0)}%
                </span>
              )}
            </div>

            <table>
              <tbody>
                <tr>
                  <td>Structural Complexity Score</td>
                  <td>{meta.complexity_score.toFixed(3)} / 1.000</td>
                </tr>
                <tr>
                  <td>Gateway Overhead / Upstream RTT</td>
                  <td>
                    {meta.gateway_overhead_ms.toFixed(3)} ms / {meta.upstream_rtt_ms.toFixed(2)} ms
                  </td>
                </tr>
                <tr>
                  <td>Actual Cost vs Static Frontier Pin</td>
                  <td>
                    ${meta.actual_cost_usd.toFixed(6)} vs $
                    {meta.counterfactual_frontier_cost_usd.toFixed(6)}
                  </td>
                </tr>
                <tr>
                  <td>FinOps Net Savings on Request</td>
                  <td style={{ color: "#34d399", fontWeight: 700 }}>
                    ${meta.cost_saved_usd.toFixed(6)}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
