"""Core PrismGate Inference Execution Engine.

Coordinates hierarchical admission control, signed-hash semantic caching,
complexity-scored speculative cascade routing, TCP-Vegas / Gradient2 adaptive
concurrency leases, online claim-faithfulness verification, and telemetry drift
collection across synchronous and OpenAI SSE streaming pipelines.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator

from prism_gate import __version__
from prism_gate.admission.token_bucket import AdmissionDecision, HierarchicalAdmissionController
from prism_gate.concur.vegas_limiter import VegasAdaptiveLimiter, VegasLease
from prism_gate.drift.detectors import DriftPlaneAnalyzer
from prism_gate.evals.faithfulness import ClaimFaithfulnessJudge
from prism_gate.providers.upstreams import SimulatedUpstreamPool, UpstreamProvider, UpstreamResult
from prism_gate.routing.cascade_router import MODEL_CATALOG, SpeculativeCascadeRouter
from prism_gate.types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionUsage,
    EndpointMetrics,
    FaithfulnessReport,
    GatewayOverview,
    GatewayTelemetryMeta,
    ModelTier,
    StreamChoice,
    StreamChunk,
    StreamDelta,
)


class GatewayRejectedError(Exception):
    """Raised when admission control or Vegas concurrency limits reject a request."""

    def __init__(
        self,
        *,
        status_code: int,
        reason: str,
        message: str,
        retry_after_seconds: float = 0.25,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.reason = reason
        self.message = message
        self.retry_after_seconds = retry_after_seconds


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = math.floor(pos)
    hi = min(len(sorted_vals) - 1, lo + 1)
    frac = pos - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def _tokenize_for_sse(text: str) -> list[str]:
    """Chunk completion string into realistic token-sized SSE deltas preserving spaces."""
    words = text.split(" ")
    chunks: list[str] = []
    for idx, word in enumerate(words):
        piece = word if idx == 0 else f" {word}"
        if piece:
            chunks.append(piece)
    return chunks or [text]


class PrismGateEngine:
    """Unified multi-tenant LLM gateway control & data plane."""

    def __init__(
        self,
        *,
        upstream: UpstreamProvider | None = None,
        admission: HierarchicalAdmissionController | None = None,
        router: SpeculativeCascadeRouter | None = None,
        faithfulness_judge: ClaimFaithfulnessJudge | None = None,
        drift_analyzer: DriftPlaneAnalyzer | None = None,
        limiters: dict[str, VegasAdaptiveLimiter] | None = None,
    ) -> None:
        self.started_at = time.monotonic()
        self.upstream: UpstreamProvider = upstream or SimulatedUpstreamPool()
        self.admission = admission or HierarchicalAdmissionController()
        self.router = router or SpeculativeCascadeRouter()
        self.faithfulness_judge = faithfulness_judge or ClaimFaithfulnessJudge()
        self.drift_analyzer = drift_analyzer or DriftPlaneAnalyzer()

        self.limiters: dict[str, VegasAdaptiveLimiter] = limiters or self._default_limiters()

        self._e2e_latencies_ms: deque[float] = deque(maxlen=1024)
        self._faithfulness_scores: deque[float] = deque(maxlen=512)
        self._hallucinations_flagged = 0

    @staticmethod
    def _default_limiters() -> dict[str, VegasAdaptiveLimiter]:
        result: dict[str, VegasAdaptiveLimiter] = {}
        for model_id, spec in MODEL_CATALOG.items():
            initial = 10 if spec.tier == ModelTier.FAST else 8
            max_cap = 48 if spec.tier == ModelTier.FAST else 32
            result[model_id] = VegasAdaptiveLimiter(
                provider_id=model_id,
                tier=spec.tier,
                initial_limit=initial,
                min_limit=2,
                max_limit=max_cap,
                alpha=2.5,
                beta=5.5,
                cost_per_1k_input_usd=spec.cost_per_1k_input_usd,
                cost_per_1k_output_usd=spec.cost_per_1k_output_usd,
            )
        return result

    def gateway_utilization(self) -> tuple[int, int, float]:
        """Return `(total_inflight, total_capacity, utilization_ratio)`."""
        inflight = sum(lim.inflight for lim in self.limiters.values())
        capacity = max(1, sum(lim.max_inflight for lim in self.limiters.values()))
        return inflight, capacity, min(1.0, inflight / capacity)

    def _acquire_tier_lease(self, preferred_model: str, tier: ModelTier) -> tuple[str, VegasLease]:
        preferred_lim = self.limiters.get(preferred_model)
        if preferred_lim is not None:
            lease = preferred_lim.try_acquire()
            if lease is not None:
                return preferred_model, lease

        # Failover to sibling model in the same tier if available
        for model_id, lim in self.limiters.items():
            if model_id != preferred_model and lim.tier == tier:
                lease = lim.try_acquire()
                if lease is not None:
                    return model_id, lease

        self.admission.record_vegas_shed()
        raise GatewayRejectedError(
            status_code=503,
            reason="VEGAS_CONCURRENCY_SHED",
            message=(
                f"Adaptive Vegas concurrency ceiling reached for tier {tier.value} "
                f"(preferred={preferred_model}). Shedding fail-fast to protect p99 SLO."
            ),
            retry_after_seconds=0.15,
        )

    async def _invoke_with_lease(
        self,
        *,
        model_id: str,
        lease: VegasLease,
        request: ChatCompletionRequest,
        prompt_tokens: int,
    ) -> UpstreamResult:
        try:
            res = await self.upstream.generate(
                model_id=model_id,
                request=request,
                prompt_tokens=prompt_tokens,
            )
        except Exception:
            lease.release(status_code=503)
            raise

        lease.release(
            status_code=res.status_code,
            override_rtt_ms=res.simulated_rtt_ms,
        )
        return res

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Execute full control-plane + inference lifecycle and return OpenAI response."""
        t_gate_start = time.perf_counter()
        prompt_text = "\n".join(m.content for m in request.messages)
        plan = self.router.plan_route(request, self.limiters)
        estimated_tokens = plan.complexity.estimated_prompt_tokens + min(request.max_tokens, 128)

        _, _, util = self.gateway_utilization()
        decision: AdmissionDecision = self.admission.evaluate(
            tenant_id=request.tenant_id,
            priority=request.priority,
            estimated_tokens=estimated_tokens,
            gateway_utilization=util,
        )
        if not decision.allowed:
            raise GatewayRejectedError(
                status_code=decision.status_code,
                reason=decision.reason,
                message=f"Request rejected by PrismGate admission plane: {decision.reason}",
                retry_after_seconds=decision.retry_after_seconds,
            )

        bucket = self.admission.get_or_create_bucket(request.tenant_id)
        cache_enabled = bucket.profile.semantic_cache_enabled and not request.bypass_cache

        if cache_enabled:
            cached = self.router.cache.lookup(
                tenant_id=request.tenant_id,
                prompt_text=prompt_text,
                rag_context=request.rag_context,
            )
            if cached.hit:
                overhead_ms = (time.perf_counter() - t_gate_start) * 1000.0
                prompt_toks = plan.complexity.estimated_prompt_tokens
                comp_toks = cached.completion_tokens
                self.admission.reconcile_usage(
                    tenant_id=request.tenant_id,
                    estimated_tokens=estimated_tokens,
                    actual_tokens=0,
                )
                counterfactual, saved = self.router.record_finops(
                    prompt_tokens=prompt_toks,
                    completion_tokens=comp_toks,
                    final_tier=cached.tier,
                    actual_cost_usd=0.0,
                    escalated=False,
                    cache_hit=True,
                )
                faith_report: FaithfulnessReport | None = None
                if request.rag_context:
                    faith_report = self.faithfulness_judge.evaluate(
                        completion=cached.response_text,
                        context_chunks=request.rag_context,
                    )
                self._e2e_latencies_ms.append(overhead_ms)
                meta = GatewayTelemetryMeta(
                    tenant_id=request.tenant_id,
                    priority=request.priority,
                    selected_model=cached.model_id,
                    selected_tier=cached.tier,
                    escalated_from_fast=False,
                    cache_hit=True,
                    cache_similarity=cached.similarity,
                    complexity_score=plan.complexity.score,
                    gateway_overhead_ms=round(overhead_ms, 3),
                    upstream_rtt_ms=0.0,
                    actual_cost_usd=0.0,
                    counterfactual_frontier_cost_usd=round(counterfactual, 6),
                    cost_saved_usd=round(saved, 6),
                    faithfulness_score=(
                        faith_report.faithfulness_score if faith_report is not None else None
                    ),
                    hallucination_detected=(
                        faith_report.hallucination_detected if faith_report is not None else False
                    ),
                )
                return ChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4().hex[:16]}",
                    created=int(time.time()),
                    model=cached.model_id,
                    choices=(
                        CompletionChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content=cached.response_text),
                            finish_reason="stop",
                        ),
                    ),
                    usage=CompletionUsage(
                        prompt_tokens=prompt_toks,
                        completion_tokens=comp_toks,
                        total_tokens=prompt_toks + comp_toks,
                    ),
                    prism_meta=meta,
                    faithfulness_report=faith_report,
                )

        # Acquire primary Vegas concurrency lease
        chosen_model, lease = self._acquire_tier_lease(plan.primary_model, plan.primary_tier)
        gateway_overhead_ms = (time.perf_counter() - t_gate_start) * 1000.0

        primary_res = await self._invoke_with_lease(
            model_id=chosen_model,
            lease=lease,
            request=request,
            prompt_tokens=plan.complexity.estimated_prompt_tokens,
        )

        if primary_res.status_code != 200:
            self.admission.reconcile_usage(
                tenant_id=request.tenant_id,
                estimated_tokens=estimated_tokens,
                actual_tokens=0,
            )
            raise GatewayRejectedError(
                status_code=503 if primary_res.status_code >= 500 else primary_res.status_code,
                reason=f"UPSTREAM_HTTP_{primary_res.status_code}",
                message=f"Upstream provider {chosen_model} returned HTTP {primary_res.status_code}",
                retry_after_seconds=0.30,
            )

        final_res = primary_res
        final_tier = plan.primary_tier
        escalated = False
        actual_cost = self.router.compute_model_cost(
            primary_res.model_id,
            prompt_tokens=primary_res.prompt_tokens,
            completion_tokens=primary_res.completion_tokens,
        )
        total_upstream_rtt_ms = primary_res.simulated_rtt_ms

        faith_report = (
            self.faithfulness_judge.evaluate(
                completion=primary_res.content,
                context_chunks=request.rag_context,
            )
            if request.rag_context
            else None
        )

        if plan.allow_escalation and self.router.should_escalate(
            confidence_score=primary_res.confidence_score,
            min_confidence=request.min_confidence,
            faithfulness_score=(
                faith_report.faithfulness_score if faith_report is not None else None
            ),
        ):
            frontier_model, frontier_lease = self._acquire_tier_lease(
                plan.fallback_frontier_model,
                ModelTier.FRONTIER,
            )
            frontier_res = await self._invoke_with_lease(
                model_id=frontier_model,
                lease=frontier_lease,
                request=request,
                prompt_tokens=plan.complexity.estimated_prompt_tokens,
            )
            if frontier_res.status_code == 200:
                escalated = True
                final_res = frontier_res
                final_tier = ModelTier.FRONTIER
                actual_cost += self.router.compute_model_cost(
                    frontier_res.model_id,
                    prompt_tokens=frontier_res.prompt_tokens,
                    completion_tokens=frontier_res.completion_tokens,
                )
                total_upstream_rtt_ms += frontier_res.simulated_rtt_ms
                if request.rag_context:
                    faith_report = self.faithfulness_judge.evaluate(
                        completion=frontier_res.content,
                        context_chunks=request.rag_context,
                    )

        actual_total_tokens = final_res.prompt_tokens + final_res.completion_tokens
        self.admission.reconcile_usage(
            tenant_id=request.tenant_id,
            estimated_tokens=estimated_tokens,
            actual_tokens=actual_total_tokens,
        )

        counterfactual, saved = self.router.record_finops(
            prompt_tokens=final_res.prompt_tokens,
            completion_tokens=final_res.completion_tokens,
            final_tier=final_tier,
            actual_cost_usd=actual_cost,
            escalated=escalated,
            cache_hit=False,
        )

        hallucinated = faith_report.hallucination_detected if faith_report is not None else False
        if faith_report is not None:
            self._faithfulness_scores.append(faith_report.faithfulness_score)
            if hallucinated:
                self._hallucinations_flagged += 1

        # Store in isolated tenant cache only when grounded / hallucination-free
        if cache_enabled and not hallucinated:
            self.router.cache.store(
                tenant_id=request.tenant_id,
                prompt_text=prompt_text,
                rag_context=request.rag_context,
                response_text=final_res.content,
                model_id=final_res.model_id,
                tier=final_tier,
                completion_tokens=final_res.completion_tokens,
            )

        e2e_ms = gateway_overhead_ms + total_upstream_rtt_ms
        self._e2e_latencies_ms.append(e2e_ms)
        self.drift_analyzer.record_observation(
            prompt_text=prompt_text,
            prompt_tokens=final_res.prompt_tokens,
            complexity_score=plan.complexity.score,
            latency_ms=e2e_ms,
        )

        meta = GatewayTelemetryMeta(
            tenant_id=request.tenant_id,
            priority=request.priority,
            selected_model=final_res.model_id,
            selected_tier=final_tier,
            escalated_from_fast=escalated,
            cache_hit=False,
            cache_similarity=0.0,
            complexity_score=plan.complexity.score,
            gateway_overhead_ms=round(gateway_overhead_ms, 3),
            upstream_rtt_ms=round(total_upstream_rtt_ms, 3),
            actual_cost_usd=round(actual_cost, 6),
            counterfactual_frontier_cost_usd=round(counterfactual, 6),
            cost_saved_usd=round(saved, 6),
            faithfulness_score=(
                faith_report.faithfulness_score if faith_report is not None else None
            ),
            hallucination_detected=hallucinated,
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:16]}",
            created=int(time.time()),
            model=final_res.model_id,
            choices=(
                CompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=final_res.content),
                    finish_reason="stop",
                ),
            ),
            usage=CompletionUsage(
                prompt_tokens=final_res.prompt_tokens,
                completion_tokens=final_res.completion_tokens,
                total_tokens=actual_total_tokens,
            ),
            prism_meta=meta,
            faithfulness_report=faith_report,
        )

    async def prepare_sse_stream(
        self,
        request: ChatCompletionRequest,
    ) -> tuple[ChatCompletionResponse, AsyncIterator[str]]:
        """Execute admission/routing/verification first (so 429/503 fail fast before HTTP 200),
        then stream OpenAI-compliant `text/event-stream` SSE token chunks ending in `[DONE]`.
        """
        response = await self.complete(request)

        async def _event_generator() -> AsyncIterator[str]:
            role_chunk = StreamChunk(
                id=response.id,
                created=response.created,
                model=response.model,
                choices=(
                    StreamChoice(
                        index=0,
                        delta=StreamDelta(role="assistant", content=""),
                        finish_reason=None,
                    ),
                ),
            )
            yield f"data: {role_chunk.model_dump_json(exclude_none=True)}\n\n"

            content_text = response.choices[0].message.content
            for piece in _tokenize_for_sse(content_text):
                token_chunk = StreamChunk(
                    id=response.id,
                    created=response.created,
                    model=response.model,
                    choices=(
                        StreamChoice(
                            index=0,
                            delta=StreamDelta(content=piece),
                            finish_reason=None,
                        ),
                    ),
                )
                yield f"data: {token_chunk.model_dump_json(exclude_none=True)}\n\n"
                await asyncio.sleep(0)

            final_chunk = StreamChunk(
                id=response.id,
                created=response.created,
                model=response.model,
                choices=(
                    StreamChoice(
                        index=0,
                        delta=StreamDelta(),
                        finish_reason="stop",
                    ),
                ),
                prism_meta=response.prism_meta,
            )
            yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
            yield "data: [DONE]\n\n"

        return response, _event_generator()

    def endpoint_snapshots(self) -> tuple[EndpointMetrics, ...]:
        """Return current `EndpointMetrics` across all upstream limiters."""
        return tuple(lim.snapshot() for lim in self.limiters.values())

    def overview(self) -> GatewayOverview:
        """Assemble full `GatewayOverview` snapshot for the observability plane."""
        inflight, capacity, util = self.gateway_utilization()
        sorted_lat = sorted(self._e2e_latencies_ms)
        avg_faith = (
            sum(self._faithfulness_scores) / len(self._faithfulness_scores)
            if self._faithfulness_scores
            else 1.0
        )
        return GatewayOverview(
            version=__version__,
            uptime_seconds=round(max(0.0, time.monotonic() - self.started_at), 2),
            gateway_utilization=round(util, 4),
            total_inflight=inflight,
            total_capacity=capacity,
            latency_p50_ms=round(_percentile(sorted_lat, 0.50), 3),
            latency_p95_ms=round(_percentile(sorted_lat, 0.95), 3),
            latency_p99_ms=round(_percentile(sorted_lat, 0.99), 3),
            avg_faithfulness_score=round(avg_faith, 4),
            hallucinations_flagged_total=self._hallucinations_flagged,
            finops=self.router.finops_summary(),
            admission=self.admission.summary(),
            endpoints=self.endpoint_snapshots(),
            latest_drift=self.drift_analyzer.latest_report,
        )

    def render_prometheus_metrics(self) -> str:
        """Render Prometheus text exposition format (`0.0.4`)."""
        snap = self.overview()
        lines: list[str] = [
            "# HELP prism_gate_uptime_seconds Gateway process uptime in seconds.",
            "# TYPE prism_gate_uptime_seconds gauge",
            f"prism_gate_uptime_seconds {snap.uptime_seconds}",
            "# HELP prism_gate_gateway_utilization Fraction of dynamic Vegas capacity in use.",
            "# TYPE prism_gate_gateway_utilization gauge",
            f"prism_gate_gateway_utilization {snap.gateway_utilization}",
            "# HELP prism_gate_latency_ms End-to-end request latency quantiles in milliseconds.",
            "# TYPE prism_gate_latency_ms gauge",
            f'prism_gate_latency_ms{{quantile="0.5"}} {snap.latency_p50_ms}',
            f'prism_gate_latency_ms{{quantile="0.95"}} {snap.latency_p95_ms}',
            f'prism_gate_latency_ms{{quantile="0.99"}} {snap.latency_p99_ms}',
            "# HELP prism_gate_vegas_inflight Current inflight requests per upstream provider.",
            "# TYPE prism_gate_vegas_inflight gauge",
        ]
        for ep in snap.endpoints:
            t_lbl = ep.tier.value
            lines.append(
                f'prism_gate_vegas_inflight{{provider="{ep.provider_id}",tier="{t_lbl}"}} '
                f"{ep.inflight}"
            )
        lines.extend(
            [
                "# HELP prism_gate_vegas_max_inflight Dynamic TCP-Vegas concurrency limit.",
                "# TYPE prism_gate_vegas_max_inflight gauge",
            ]
        )
        for ep in snap.endpoints:
            t_lbl = ep.tier.value
            lines.append(
                f'prism_gate_vegas_max_inflight{{provider="{ep.provider_id}",tier="{t_lbl}"}} '
                f"{ep.max_inflight}"
            )
        lines.extend(
            [
                "# HELP prism_gate_vegas_rtt_min_ms Observed baseline no-queue RTT per upstream.",
                "# TYPE prism_gate_vegas_rtt_min_ms gauge",
            ]
        )
        for ep in snap.endpoints:
            lines.append(
                f'prism_gate_vegas_rtt_min_ms{{provider="{ep.provider_id}"}} {ep.rtt_min_ms}'
            )
        lines.extend(
            [
                "# HELP prism_gate_vegas_queue_estimate Standing queue depth per upstream.",
                "# TYPE prism_gate_vegas_queue_estimate gauge",
            ]
        )
        for ep in snap.endpoints:
            lines.append(
                f'prism_gate_vegas_queue_estimate{{provider="{ep.provider_id}"}} '
                f"{ep.queue_estimate}"
            )

        fo = snap.finops
        ad = snap.admission
        lines.extend(
            [
                "# HELP prism_gate_finops_cost_usd_total Cumulative inference spend in USD.",
                "# TYPE prism_gate_finops_cost_usd_total counter",
                f'prism_gate_finops_cost_usd_total{{mode="actual"}} {fo.actual_total_cost_usd}',
                f'prism_gate_finops_cost_usd_total{{mode="counterfactual_frontier"}} '
                f"{fo.counterfactual_frontier_cost_usd}",
                "# HELP prism_gate_finops_requests_total Requests routed by tier outcome.",
                "# TYPE prism_gate_finops_requests_total counter",
                f'prism_gate_finops_requests_total{{outcome="fast"}} {fo.fast_tier_hits}',
                f'prism_gate_finops_requests_total{{outcome="frontier"}} {fo.frontier_direct_hits}',
                f'prism_gate_finops_requests_total{{outcome="escalated"}} {fo.cascade_escalations}',
                f'prism_gate_finops_requests_total{{outcome="cache_hit"}} {fo.semantic_cache_hits}',
                "# HELP prism_gate_admission_decisions_total Admission controller outcomes.",
                "# TYPE prism_gate_admission_decisions_total counter",
                f'prism_gate_admission_decisions_total{{outcome="admitted"}} {ad.admitted_total}',
                f'prism_gate_admission_decisions_total{{outcome="rejected_rps"}} '
                f"{ad.rejected_rps_total}",
                f'prism_gate_admission_decisions_total{{outcome="rejected_tpm"}} '
                f"{ad.rejected_tpm_total}",
                f'prism_gate_admission_decisions_total{{outcome="shed_batch"}} '
                f"{ad.shed_batch_total}",
                f'prism_gate_admission_decisions_total{{outcome="shed_interactive"}} '
                f"{ad.shed_interactive_total}",
                f'prism_gate_admission_decisions_total{{outcome="shed_saturated"}} '
                f"{ad.shed_saturated_total}",
                f'prism_gate_admission_decisions_total{{outcome="shed_vegas"}} '
                f"{ad.shed_vegas_total}",
                "# HELP prism_gate_faithfulness_score_avg Mean online RAG faithfulness score.",
                "# TYPE prism_gate_faithfulness_score_avg gauge",
                f"prism_gate_faithfulness_score_avg {snap.avg_faithfulness_score}",
                "# HELP prism_gate_hallucinations_flagged_total Completions with hallucinations.",
                "# TYPE prism_gate_hallucinations_flagged_total counter",
                f"prism_gate_hallucinations_flagged_total {snap.hallucinations_flagged_total}",
                "# HELP prism_gate_drift_max_psi Maximum feature Population Stability Index.",
                "# TYPE prism_gate_drift_max_psi gauge",
                f"prism_gate_drift_max_psi {snap.latest_drift.max_psi}",
                "# HELP prism_gate_drift_token_jsd_bits Base-2 Jensen-Shannon token divergence.",
                "# TYPE prism_gate_drift_token_jsd_bits gauge",
                f"prism_gate_drift_token_jsd_bits {snap.latest_drift.token_jsd}",
                "",
            ]
        )
        return "\n".join(lines)
