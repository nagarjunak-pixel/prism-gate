"""Frozen Pydantic v2 domain & wire contracts for PrismGate."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PriorityTier(StrEnum):
    """Hierarchical request admission priority tiers."""

    CRITICAL = "CRITICAL"
    INTERACTIVE = "INTERACTIVE"
    BATCH_BACKGROUND = "BATCH_BACKGROUND"


class ModelTier(StrEnum):
    """Speculative cascade routing model tiers."""

    FAST = "FAST"
    FRONTIER = "FRONTIER"


class DriftSeverity(StrEnum):
    """Statistical distribution-shift alert levels."""

    NO_DRIFT = "NO_DRIFT"
    MODERATE_DRIFT = "MODERATE_DRIFT"
    CRITICAL_DRIFT = "CRITICAL_DRIFT"


class ClaimVerdict(StrEnum):
    """Atomic factual claim grounding verdicts."""

    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


class ChatMessage(BaseModel):
    """Immutable OpenAI-compatible chat message."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request with multi-tenant & RAG extensions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(default="prism-auto", min_length=1)
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    stream: bool = False
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=16384)
    user: str | None = None

    # PrismGate control-plane extensions
    tenant_id: str = Field(default="tenant-enterprise-01", min_length=1)
    priority: PriorityTier = PriorityTier.INTERACTIVE
    rag_context: tuple[str, ...] = ()
    bypass_cache: bool = False
    min_confidence: float = Field(default=0.72, ge=0.0, le=1.0)


class ComplexityBreakdown(BaseModel):
    """Interpretable structural prompt complexity decomposition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    reasoning_depth: float = Field(ge=0.0, le=1.0)
    code_math_density: float = Field(ge=0.0, le=1.0)
    constraint_density: float = Field(ge=0.0, le=1.0)
    multihop_markers: float = Field(ge=0.0, le=1.0)
    estimated_prompt_tokens: int = Field(ge=1)


class GatewayTelemetryMeta(BaseModel):
    """Per-request telemetry envelope attached to PrismGate responses."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    priority: PriorityTier
    selected_model: str
    selected_tier: ModelTier
    escalated_from_fast: bool
    cache_hit: bool
    cache_similarity: float = Field(ge=0.0, le=1.0)
    complexity_score: float = Field(ge=0.0, le=1.0)
    gateway_overhead_ms: float = Field(ge=0.0)
    upstream_rtt_ms: float = Field(ge=0.0)
    actual_cost_usd: float = Field(ge=0.0)
    counterfactual_frontier_cost_usd: float = Field(ge=0.0)
    cost_saved_usd: float
    faithfulness_score: float | None = Field(default=None, ge=0.0, le=1.0)
    hallucination_detected: bool = False


class StreamDelta(BaseModel):
    """OpenAI SSE token delta payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["assistant"] | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    """OpenAI SSE streaming choice item."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(default=0, ge=0)
    delta: StreamDelta
    finish_reason: Literal["stop", "length"] | None = None


class StreamChunk(BaseModel):
    """OpenAI-compatible `chat.completion.chunk` SSE frame model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: tuple[StreamChoice, ...]
    prism_meta: GatewayTelemetryMeta | None = None


class CompletionChoice(BaseModel):
    """OpenAI synchronous completion choice."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(default=0, ge=0)
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class CompletionUsage(BaseModel):
    """OpenAI token accounting block."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible `chat.completion` synchronous JSON response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: tuple[CompletionChoice, ...]
    usage: CompletionUsage
    prism_meta: GatewayTelemetryMeta
    faithfulness_report: FaithfulnessReport | None = None


class TenantSLOProfile(BaseModel):
    """Immutable per-tenant dual rate-limit and latency SLO profile."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    rps_limit: float = Field(gt=0.0)
    burst_rps: float = Field(gt=0.0)
    tpm_limit: float = Field(gt=0.0)
    burst_tokens: float = Field(gt=0.0)
    default_priority: PriorityTier = PriorityTier.INTERACTIVE
    latency_slo_ms: float = Field(default=850.0, gt=0.0)
    semantic_cache_enabled: bool = True


class EndpointMetrics(BaseModel):
    """Live TCP-Vegas / Gradient2 adaptive concurrency & cost snapshot per upstream."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_id: str
    tier: ModelTier
    inflight: int = Field(ge=0)
    max_inflight: int = Field(ge=1)
    min_limit: int = Field(ge=1)
    max_limit: int = Field(ge=1)
    rtt_min_ms: float = Field(ge=0.0)
    rtt_ewma_ms: float = Field(ge=0.0)
    last_rtt_ms: float = Field(ge=0.0)
    queue_estimate: float = Field(ge=0.0)
    gradient: float = Field(ge=0.0, le=1.0)
    total_requests: int = Field(ge=0)
    total_shed: int = Field(ge=0)
    total_errors_429_503: int = Field(ge=0)
    cost_per_1k_input_usd: float = Field(ge=0.0)
    cost_per_1k_output_usd: float = Field(ge=0.0)


class ClaimVerification(BaseModel):
    """Per-claim entailment and numeric/entity alignment verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_text: str
    verdict: ClaimVerdict
    entailment_score: float = Field(ge=0.0, le=1.0)
    matched_numbers: tuple[str, ...] = ()
    unmatched_numbers: tuple[str, ...] = ()
    unmatched_entities: tuple[str, ...] = ()
    supporting_chunk_index: int | None = None
    reason: str = ""


class FaithfulnessReport(BaseModel):
    """Claim-level groundedness report produced by the online hallucination judge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    faithfulness_score: float = Field(ge=0.0, le=1.0)
    total_claims: int = Field(ge=0)
    supported_claims_count: int = Field(ge=0)
    unsupported_claims: tuple[str, ...] = ()
    contradicted_claims: tuple[str, ...] = ()
    flagged_numbers: tuple[str, ...] = ()
    flagged_entities: tuple[str, ...] = ()
    claim_verifications: tuple[ClaimVerification, ...] = ()
    hallucination_detected: bool


class FaithfulnessRequest(BaseModel):
    """REST payload for `/api/v1/evals/faithfulness`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    completion: str = Field(min_length=1)
    context_chunks: tuple[str, ...] = ()


class FeaturePSIResult(BaseModel):
    """Population Stability Index result for a single continuous telemetry feature."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_name: str
    psi_value: float = Field(ge=0.0)
    severity: DriftSeverity
    bin_edges: tuple[float, ...]
    baseline_proportions: tuple[float, ...]
    live_proportions: tuple[float, ...]


class DriftReport(BaseModel):
    """Unified statistical distribution-shift report across PSI & Jensen-Shannon metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    overall_severity: DriftSeverity
    max_psi: float = Field(ge=0.0)
    token_jsd: float = Field(ge=0.0, le=1.0)
    topic_jsd: float = Field(ge=0.0, le=1.0)
    feature_psi: tuple[FeaturePSIResult, ...]
    baseline_sample_size: int = Field(ge=1)
    live_sample_size: int = Field(ge=1)
    alert_reasons: tuple[str, ...] = ()


class DriftAnalyzeRequest(BaseModel):
    """REST payload for `/api/v1/drift/analyze` supporting presets or explicit distributions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    preset: Literal["normal", "moderate", "critical", "live_window"] | None = None
    baseline_prompt_lengths: tuple[float, ...] = ()
    live_prompt_lengths: tuple[float, ...] = ()
    baseline_complexities: tuple[float, ...] = ()
    live_complexities: tuple[float, ...] = ()
    baseline_latencies_ms: tuple[float, ...] = ()
    live_latencies_ms: tuple[float, ...] = ()
    baseline_token_counts: dict[str, float] = Field(default_factory=dict)
    live_token_counts: dict[str, float] = Field(default_factory=dict)
    baseline_topic_counts: dict[str, float] = Field(default_factory=dict)
    live_topic_counts: dict[str, float] = Field(default_factory=dict)


class FinOpsSummary(BaseModel):
    """Cumulative FinOps cost & cascade efficiency ledger snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_routed_requests: int = Field(ge=0)
    fast_tier_hits: int = Field(ge=0)
    frontier_direct_hits: int = Field(ge=0)
    cascade_escalations: int = Field(ge=0)
    semantic_cache_hits: int = Field(ge=0)
    cache_hit_rate: float = Field(ge=0.0, le=1.0)
    actual_total_cost_usd: float = Field(ge=0.0)
    counterfactual_frontier_cost_usd: float = Field(ge=0.0)
    net_savings_usd: float
    savings_pct: float


class AdmissionSummary(BaseModel):
    """Admission controller & priority shedding telemetry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    admitted_total: int = Field(ge=0)
    rejected_rps_total: int = Field(ge=0)
    rejected_tpm_total: int = Field(ge=0)
    shed_batch_total: int = Field(ge=0)
    shed_interactive_total: int = Field(ge=0)
    shed_saturated_total: int = Field(ge=0)
    shed_vegas_total: int = Field(ge=0)


class BenchmarkRequest(BaseModel):
    """REST payload for `/api/v1/benchmarks/load`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    requests: int = Field(default=150, ge=10, le=2000)
    concurrency: int = Field(default=24, ge=1, le=256)
    inject_congestion_spike: bool = True


class BenchmarkSummary(BaseModel):
    """Structured load-benchmark output capturing p50/p95/p99, FinOps, and shedding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_requests: int = Field(ge=1)
    concurrency: int = Field(ge=1)
    succeeded_requests: int = Field(ge=0)
    shed_or_rate_limited: int = Field(ge=0)
    critical_admitted: int = Field(ge=0)
    critical_total: int = Field(ge=0)
    batch_shed: int = Field(ge=0)
    elapsed_wall_ms: float = Field(ge=0.0)
    throughput_rps: float = Field(ge=0.0)
    gateway_overhead_p50_ms: float = Field(ge=0.0)
    gateway_overhead_p95_ms: float = Field(ge=0.0)
    gateway_overhead_p99_ms: float = Field(ge=0.0)
    e2e_latency_p50_ms: float = Field(ge=0.0)
    e2e_latency_p95_ms: float = Field(ge=0.0)
    e2e_latency_p99_ms: float = Field(ge=0.0)
    finops: FinOpsSummary
    endpoints: tuple[EndpointMetrics, ...]


class GatewayOverview(BaseModel):
    """Aggregate control-plane state returned by `GET /api/v1/overview`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    uptime_seconds: float = Field(ge=0.0)
    gateway_utilization: float = Field(ge=0.0, le=1.0)
    total_inflight: int = Field(ge=0)
    total_capacity: int = Field(ge=1)
    latency_p50_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)
    latency_p99_ms: float = Field(ge=0.0)
    avg_faithfulness_score: float = Field(ge=0.0, le=1.0)
    hallucinations_flagged_total: int = Field(ge=0)
    finops: FinOpsSummary
    admission: AdmissionSummary
    endpoints: tuple[EndpointMetrics, ...]
    latest_drift: DriftReport
