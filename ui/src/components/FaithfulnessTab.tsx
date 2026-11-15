import { useState } from "react";
import { AlertTriangle, CheckCircle2, ShieldAlert } from "lucide-react";
import type { FaithfulnessReport } from "../types";

interface FaithPreset {
  name: string;
  chunks: string;
  completion: string;
}

const FAITH_PRESETS: FaithPreset[] = [
  {
    name: "Clinical Dosage Contradiction (500mg vs 5mg)",
    chunks:
      "Warfarin oral maintenance therapy is prescribed at 5mg daily for atrial fibrillation.\nTarget INR therapeutic range must be maintained between 2.0 and 3.0.",
    completion:
      "Warfarin oral maintenance therapy is prescribed at 500mg daily for atrial fibrillation. Target INR therapeutic range must be maintained between 2.0 and 3.0.",
  },
  {
    name: "Financial Filing Revenue Hallucination ($18.9M vs $12.4M)",
    chunks:
      "Acme Corp reported Q3 2025 revenue of $12.4M, representing 14.2% growth.\nOperating GAAP margin expanded to 18.5% during Q3.",
    completion:
      "Acme Corp reported Q3 2025 revenue of $18.9M, representing 14.2% growth. Operating GAAP margin expanded to 18.5% during Q3.",
  },
  {
    name: "100% Grounded Enterprise RAG Summary",
    chunks:
      "Acme Corp reported Q3 2025 revenue of $12.4M, representing 14.2% growth.\nOperating GAAP margin expanded to 18.5% during Q3.",
    completion:
      "Acme Corp reported Q3 2025 revenue of $12.4M with 14.2% growth. Operating GAAP margin expanded to 18.5% in Q3.",
  },
];

const INITIAL_REPORT: FaithfulnessReport = {
  faithfulness_score: 0.3938,
  total_claims: 2,
  supported_claims_count: 1,
  unsupported_claims: [
    "Warfarin oral maintenance therapy is prescribed at 500mg daily for atrial fibrillation",
  ],
  contradicted_claims: [
    "Warfarin oral maintenance therapy is prescribed at 500mg daily for atrial fibrillation",
  ],
  flagged_numbers: ["500mg"],
  flagged_entities: [],
  hallucination_detected: true,
  claim_verifications: [
    {
      claim_text:
        "Warfarin oral maintenance therapy is prescribed at 500mg daily for atrial fibrillation",
      verdict: "CONTRADICTED",
      entailment_score: 0.15,
      matched_numbers: [],
      unmatched_numbers: ["500mg"],
      unmatched_entities: [],
      supporting_chunk_index: null,
      reason: "Numeric contradiction against chunk #0: unverified value(s) 500mg.",
    },
    {
      claim_text: "Target INR therapeutic range must be maintained between 2.0 and 3.0",
      verdict: "SUPPORTED",
      entailment_score: 1.0,
      matched_numbers: ["2.0", "3.0"],
      unmatched_numbers: [],
      unmatched_entities: [],
      supporting_chunk_index: 1,
      reason: "Entailed by context chunk #1.",
    },
  ],
};

export function FaithfulnessTab() {
  const [chunksText, setChunksText] = useState(FAITH_PRESETS[0].chunks);
  const [completion, setCompletion] = useState(FAITH_PRESETS[0].completion);
  const [report, setReport] = useState<FaithfulnessReport>(INITIAL_REPORT);
  const [evaluating, setEvaluating] = useState(false);

  async function runJudge(customChunks?: string, customCompletion?: string) {
    const rawChunks = customChunks ?? chunksText;
    const rawComp = customCompletion ?? completion;
    setEvaluating(true);
    try {
      const resp = await fetch("/api/v1/evals/faithfulness", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          completion: rawComp,
          context_chunks: rawChunks
            .split("\n")
            .map((c) => c.trim())
            .filter(Boolean),
        }),
      });
      if (resp.ok) {
        setReport((await resp.json()) as FaithfulnessReport);
      }
    } catch {
      // Offline standalone fallback preserves current report
    } finally {
      setEvaluating(false);
    }
  }

  return (
    <div className="grid-2">
      <div className="card">
        <div className="card-title">
          <span>Online Atomic Claim-Faithfulness Judge</span>
          <span className="pill pill-cyan">Sub-ms Deterministic Entailment</span>
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
          {FAITH_PRESETS.map((preset) => (
            <button
              key={preset.name}
              type="button"
              className="btn-ghost"
              onClick={() => {
                setChunksText(preset.chunks);
                setCompletion(preset.completion);
                void runJudge(preset.chunks, preset.completion);
              }}
            >
              {preset.name}
            </button>
          ))}
        </div>

        <div style={{ marginBottom: 12 }}>
          <label className="input-label">Retrieved Authoritative RAG Context Chunks (1 per line)</label>
          <textarea
            className="textarea-box"
            rows={5}
            value={chunksText}
            onChange={(e) => setChunksText(e.target.value)}
          />
        </div>

        <div style={{ marginBottom: 14 }}>
          <label className="input-label">Candidate Model Completion to Inspect</label>
          <textarea
            className="textarea-box"
            rows={4}
            value={completion}
            onChange={(e) => setCompletion(e.target.value)}
          />
        </div>

        <button
          type="button"
          className="btn-primary"
          disabled={evaluating}
          onClick={() => void runJudge()}
        >
          <ShieldAlert size={15} /> {evaluating ? "Verifying Claims..." : "Verify Groundedness"}
        </button>
      </div>

      <div className="card">
        <div className="card-title">
          <span>Atomic Claim Verification Breakdown</span>
          <span className={report.hallucination_detected ? "pill pill-rose" : "pill pill-emerald"}>
            {report.hallucination_detected ? (
              <>
                <AlertTriangle size={13} /> HALLUCINATION DETECTED
              </>
            ) : (
              <>
                <CheckCircle2 size={13} /> 100% GROUNDED
              </>
            )}
          </span>
        </div>

        <div style={{ display: "flex", gap: 14, marginBottom: 16, flexWrap: "wrap" }}>
          <span className="pill">
            Faithfulness Score: <strong>{(report.faithfulness_score * 100).toFixed(1)}%</strong>
          </span>
          <span className="pill">
            Supported: {report.supported_claims_count} / {report.total_claims}
          </span>
          {report.flagged_numbers.length > 0 && (
            <span className="pill pill-rose">
              Flagged Numbers: {report.flagged_numbers.join(", ")}
            </span>
          )}
          {report.flagged_entities.length > 0 && (
            <span className="pill pill-amber">
              Flagged Entities: {report.flagged_entities.join(", ")}
            </span>
          )}
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {report.claim_verifications.map((cv, idx) => {
            const badgeClass =
              cv.verdict === "SUPPORTED"
                ? "pill pill-emerald"
                : cv.verdict === "CONTRADICTED"
                  ? "pill pill-rose"
                  : "pill pill-amber";
            return (
              <div
                key={idx}
                style={{
                  padding: 13,
                  borderRadius: 9,
                  background: "#080e1b",
                  border: "1px solid #1e2d4a",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    gap: 8,
                    marginBottom: 8,
                  }}
                >
                  <span className={badgeClass}>{cv.verdict}</span>
                  <span style={{ fontSize: "0.78rem", color: "#94a3b8" }}>
                    Entailment: {(cv.entailment_score * 100).toFixed(0)}%
                  </span>
                </div>
                <div style={{ fontSize: "0.87rem", marginBottom: 6 }}>{cv.claim_text}</div>
                <div style={{ fontSize: "0.78rem", color: "#64748b" }}>{cv.reason}</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
