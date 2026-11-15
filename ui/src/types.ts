export type PriorityTier = "CRITICAL" | "INTERACTIVE" | "BATCH_BACKGROUND";
export type ModelTier = "FAST" | "FRONTIER";
export type DriftSeverity = "NO_DRIFT" | "MODERATE_DRIFT" | "CRITICAL_DRIFT";
export type ClaimVerdict = "SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED";

export interface GatewayTelemetryMeta {
  tenant_id: string;
  priority: PriorityTier;
  selected_model: string;
  selected_tier: ModelTier;
  escalated_from_fast: boolean;
  cache_hit: boolean;
  cache_similarity: number;
  complexity_score: number;
  gateway_overhead_ms: number;
  upstream_rtt_ms: number;
  actual_cost_usd: number;
  counterfactual_frontier_cost_usd: number;
  cost_saved_usd: number;
  faithfulness_score: number | null;
  hallucination_detected: boolean;
}

export interface EndpointMetrics {
  provider_id: string;
  tier: ModelTier;
  inflight: number;
  max_inflight: number;
  min_limit: number;
  max_limit: number;
  rtt_min_ms: number;
  rtt_ewma_ms: number;
  last_rtt_ms: number;
  queue_estimate: number;
  gradient: number;
  total_requests: number;
  total_shed: number;
  total_errors_429_503: number;
  cost_per_1k_input_usd: number;
  cost_per_1k_output_usd: number;
}

export interface FinOpsSummary {
  total_routed_requests: number;
  fast_tier_hits: number;
  frontier_direct_hits: number;
  cascade_escalations: number;
  semantic_cache_hits: number;
  cache_hit_rate: number;
  actual_total_cost_usd: number;
  counterfactual_frontier_cost_usd: number;
  net_savings_usd: number;
  savings_pct: number;
}

export interface AdmissionSummary {
  admitted_total: number;
  rejected_rps_total: number;
  rejected_tpm_total: number;
  shed_batch_total: number;
  shed_interactive_total: number;
  shed_saturated_total: number;
  shed_vegas_total: number;
}

export interface FeaturePSIResult {
  feature_name: string;
  psi_value: number;
  severity: DriftSeverity;
  bin_edges: number[];
  baseline_proportions: number[];
  live_proportions: number[];
}

export interface DriftReport {
  overall_severity: DriftSeverity;
  max_psi: number;
  token_jsd: number;
  topic_jsd: number;
  feature_psi: FeaturePSIResult[];
  baseline_sample_size: number;
  live_sample_size: number;
  alert_reasons: string[];
}

export interface ClaimVerification {
  claim_text: string;
  verdict: ClaimVerdict;
  entailment_score: number;
  matched_numbers: string[];
  unmatched_numbers: string[];
  unmatched_entities: string[];
  supporting_chunk_index: number | null;
  reason: string;
}

export interface FaithfulnessReport {
  faithfulness_score: number;
  total_claims: number;
  supported_claims_count: number;
  unsupported_claims: string[];
  contradicted_claims: string[];
  flagged_numbers: string[];
  flagged_entities: string[];
  claim_verifications: ClaimVerification[];
  hallucination_detected: boolean;
}

export interface GatewayOverview {
  version: string;
  uptime_seconds: number;
  gateway_utilization: number;
  total_inflight: number;
  total_capacity: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
  latency_p99_ms: number;
  avg_faithfulness_score: number;
  hallucinations_flagged_total: number;
  finops: FinOpsSummary;
  admission: AdmissionSummary;
  endpoints: EndpointMetrics[];
  latest_drift: DriftReport;
}

export interface BenchmarkSummary {
  total_requests: number;
  concurrency: number;
  succeeded_requests: number;
  shed_or_rate_limited: number;
  critical_admitted: number;
  critical_total: number;
  batch_shed: number;
  elapsed_wall_ms: number;
  throughput_rps: number;
  gateway_overhead_p50_ms: number;
  gateway_overhead_p95_ms: number;
  gateway_overhead_p99_ms: number;
  e2e_latency_p50_ms: number;
  e2e_latency_p95_ms: number;
  e2e_latency_p99_ms: number;
  finops: FinOpsSummary;
  endpoints: EndpointMetrics[];
}

export const INITIAL_OVERVIEW: GatewayOverview = {
  version: "1.0.0",
  uptime_seconds: 128.4,
  gateway_utilization: 0.28,
  total_inflight: 10,
  total_capacity: 36,
  latency_p50_ms: 15.9,
  latency_p95_ms: 62.15,
  latency_p99_ms: 72.02,
  avg_faithfulness_score: 0.94,
  hallucinations_flagged_total: 2,
  finops: {
    total_routed_requests: 285,
    fast_tier_hits: 148,
    frontier_direct_hits: 57,
    cascade_escalations: 30,
    semantic_cache_hits: 50,
    cache_hit_rate: 0.1754,
    actual_total_cost_usd: 0.040807,
    counterfactual_frontier_cost_usd: 0.095247,
    net_savings_usd: 0.05444,
    savings_pct: 57.16,
  },
  admission: {
    admitted_total: 285,
    rejected_rps_total: 0,
    rejected_tpm_total: 0,
    shed_batch_total: 4,
    shed_interactive_total: 0,
    shed_saturated_total: 0,
    shed_vegas_total: 11,
  },
  endpoints: [
    {
      provider_id: "gemini-2.5-flash",
      tier: "FAST",
      inflight: 3,
      max_inflight: 34,
      min_limit: 2,
      max_limit: 48,
      rtt_min_ms: 18.01,
      rtt_ewma_ms: 19.15,
      last_rtt_ms: 18.8,
      queue_estimate: 0.24,
      gradient: 0.958,
      total_requests: 71,
      total_shed: 6,
      total_errors_429_503: 0,
      cost_per_1k_input_usd: 0.00015,
      cost_per_1k_output_usd: 0.0006,
    },
    {
      provider_id: "vllm-qwen-32b",
      tier: "FAST",
      inflight: 4,
      max_inflight: 38,
      min_limit: 2,
      max_limit: 48,
      rtt_min_ms: 14.0,
      rtt_ewma_ms: 14.89,
      last_rtt_ms: 14.5,
      queue_estimate: 0.18,
      gradient: 0.965,
      total_requests: 107,
      total_shed: 5,
      total_errors_429_503: 0,
      cost_per_1k_input_usd: 0.00012,
      cost_per_1k_output_usd: 0.00036,
    },
    {
      provider_id: "gemini-2.5-pro",
      tier: "FRONTIER",
      inflight: 2,
      max_inflight: 22,
      min_limit: 2,
      max_limit: 32,
      rtt_min_ms: 42.0,
      rtt_ewma_ms: 43.07,
      last_rtt_ms: 42.9,
      queue_estimate: 0.09,
      gradient: 0.979,
      total_requests: 52,
      total_shed: 0,
      total_errors_429_503: 0,
      cost_per_1k_input_usd: 0.00125,
      cost_per_1k_output_usd: 0.01,
    },
    {
      provider_id: "claude-sonnet-4",
      tier: "FRONTIER",
      inflight: 1,
      max_inflight: 18,
      min_limit: 2,
      max_limit: 32,
      rtt_min_ms: 52.05,
      rtt_ewma_ms: 53.31,
      last_rtt_ms: 53.1,
      queue_estimate: 0.04,
      gradient: 0.98,
      total_requests: 33,
      total_shed: 0,
      total_errors_429_503: 0,
      cost_per_1k_input_usd: 0.003,
      cost_per_1k_output_usd: 0.015,
    },
  ],
  latest_drift: {
    overall_severity: "NO_DRIFT",
    max_psi: 0.0142,
    token_jsd: 0.0002,
    topic_jsd: 0.0001,
    baseline_sample_size: 200,
    live_sample_size: 200,
    alert_reasons: [],
    feature_psi: [
      {
        feature_name: "prompt_length_tokens",
        psi_value: 0.0142,
        severity: "NO_DRIFT",
        bin_edges: [28, 42, 56, 70, 84, 98, 112, 126, 140, 155, 172],
        baseline_proportions: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        live_proportions: [0.095, 0.105, 0.1, 0.095, 0.105, 0.1, 0.1, 0.095, 0.105, 0.1],
      },
      {
        feature_name: "complexity_score",
        psi_value: 0.0085,
        severity: "NO_DRIFT",
        bin_edges: [0.12, 0.19, 0.26, 0.33, 0.4, 0.47, 0.54, 0.61, 0.68, 0.74, 0.8],
        baseline_proportions: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        live_proportions: [0.1, 0.095, 0.105, 0.1, 0.1, 0.105, 0.095, 0.1, 0.1, 0.1],
      },
      {
        feature_name: "upstream_latency_ms",
        psi_value: 0.011,
        severity: "NO_DRIFT",
        bin_edges: [16, 22, 28, 34, 40, 46, 52, 58, 64, 70, 77],
        baseline_proportions: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        live_proportions: [0.105, 0.095, 0.1, 0.1, 0.095, 0.105, 0.1, 0.1, 0.095, 0.105],
      },
    ],
  },
};
