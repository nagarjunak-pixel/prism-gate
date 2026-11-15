"""Async Concurrent Load Benchmark Harness.

Generates realistic multi-tenant enterprise LLM traffic spanning:
- Simple fast-tier lookups & ticket triage
- Paraphrased repeat queries exercising the signed-hash semantic cache
- High-complexity code/math/architecture prompts routed to frontier models
- Ambiguous & RAG-grounded prompts exercising speculative cascade escalation
- Mixed priority tiers (`CRITICAL`, `INTERACTIVE`, `BATCH_BACKGROUND`) under burst concurrency
"""

from __future__ import annotations

import asyncio
import math
import time

from prism_gate.admission.token_bucket import HierarchicalAdmissionController
from prism_gate.gateway.engine import GatewayRejectedError, PrismGateEngine
from prism_gate.providers.upstreams import SimulatedUpstreamPool
from prism_gate.types import (
    BenchmarkSummary,
    ChatCompletionRequest,
    ChatMessage,
    PriorityTier,
    TenantSLOProfile,
)

_SIMPLE_PROMPTS: tuple[str, ...] = (
    "What are the standard enterprise support hours for EMEA customers?",
    "Give a one-sentence status greeting for the billing portal banner.",
    "Classify ticket urgency: customer asks how to update dark mode theme.",
    "Format Tuesday's standup reminder for the platform engineering channel.",
    "Extract the invoice currency code from: Invoice #9042 settled in USD.",
    "Translate 'Your deployment succeeded' into concise release-note text.",
)

_CACHE_PARAPHRASE_PAIRS: tuple[tuple[str, str], ...] = (
    (
        "What are the standard enterprise support hours for EMEA customers?",
        "Could you tell me the standard EMEA customer enterprise support hours?",
    ),
    (
        "Extract the invoice currency code from: Invoice #9042 settled in USD.",
        "Please extract invoice currency code from Invoice #9042 settled in USD.",
    ),
)

_FRONTIER_COMPLEX_PROMPTS: tuple[str, ...] = (
    (
        "Derive step by step the Little's Law queue stability theorem for an adaptive "
        "TCP-Vegas concurrency limiter, prove convergence invariants under M/M/c bursts, "
        "and provide a thread-safe Python asyncio class implementation."
    ),
    (
        "Compare and contrast Jensen-Shannon divergence vs Population Stability Index: "
        "first formalize their KL-divergence equations, second prove boundedness in [0, 1], "
        "and finally specify deterministic JSON schema alert rules without false positives."
    ),
)

_ESCALATION_RAG_PROMPT = (
    "Reconcile ambiguous multi-quarter margin variance from the retrieved filing chunks."
)
_RAG_CHUNKS: tuple[str, ...] = (
    "Acme Corp reported Q3 2025 revenue of $12.4M with operating margin of 18.5%.",
    "Enterprise cloud subscriptions grew 22.0% year-over-year under GAAP accounting.",
)


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = math.floor(pos)
    hi = min(len(sorted_vals) - 1, lo + 1)
    frac = pos - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


class LoadBenchmarkHarness:
    """High-concurrency load generator measuring gateway overhead, FinOps, and shedding."""

    def __init__(self, engine: PrismGateEngine | None = None) -> None:
        if engine is not None:
            self.engine = engine
            return

        bench_profiles = [
            TenantSLOProfile(
                tenant_id="tenant-enterprise-01",
                rps_limit=5000.0,
                burst_rps=2000.0,
                tpm_limit=12_000_000.0,
                burst_tokens=2_000_000.0,
                default_priority=PriorityTier.CRITICAL,
            ),
            TenantSLOProfile(
                tenant_id="tenant-saas-prod",
                rps_limit=4000.0,
                burst_rps=1500.0,
                tpm_limit=10_000_000.0,
                burst_tokens=1_500_000.0,
                default_priority=PriorityTier.INTERACTIVE,
            ),
            TenantSLOProfile(
                tenant_id="tenant-batch-analytics",
                rps_limit=3000.0,
                burst_rps=1000.0,
                tpm_limit=8_000_000.0,
                burst_tokens=1_000_000.0,
                default_priority=PriorityTier.BATCH_BACKGROUND,
            ),
        ]
        self.engine = PrismGateEngine(
            upstream=SimulatedUpstreamPool(sleep_scale=0.00022, physical_knee=12),
            admission=HierarchicalAdmissionController(profiles=bench_profiles),
        )

    @staticmethod
    def build_workload(total_requests: int) -> list[ChatCompletionRequest]:
        """Build a deterministic, diverse sequence of `total_requests` chat payloads."""
        workload: list[ChatCompletionRequest] = []
        for idx in range(total_requests):
            slot = idx % 20
            if slot < 4:
                pair = _CACHE_PARAPHRASE_PAIRS[idx % len(_CACHE_PARAPHRASE_PAIRS)]
                prompt = pair[0] if idx % 2 == 0 else pair[1]
                tenant = "tenant-enterprise-01"
                priority = PriorityTier.CRITICAL
                rag: tuple[str, ...] = ()
            elif slot < 13:
                prompt = f"{_SIMPLE_PROMPTS[idx % len(_SIMPLE_PROMPTS)]} (ticket #{1000 + idx})"
                if idx % 3 == 0:
                    tenant = "tenant-batch-analytics"
                    priority = PriorityTier.BATCH_BACKGROUND
                else:
                    tenant = "tenant-saas-prod"
                    priority = PriorityTier.INTERACTIVE
                rag = ()
            elif slot < 17:
                base_complex = _FRONTIER_COMPLEX_PROMPTS[idx % len(_FRONTIER_COMPLEX_PROMPTS)]
                prompt = f"{base_complex} Target SLO window #{2000 + idx}."
                tenant = "tenant-enterprise-01" if idx % 2 == 0 else "tenant-saas-prod"
                priority = PriorityTier.CRITICAL if idx % 2 == 0 else PriorityTier.INTERACTIVE
                rag = ()
            elif slot < 19:
                prompt = f"{_ESCALATION_RAG_PROMPT} Audit trace #{3000 + idx}."
                tenant = "tenant-enterprise-01"
                priority = PriorityTier.CRITICAL
                rag = _RAG_CHUNKS
            else:
                prompt = f"{_SIMPLE_PROMPTS[idx % len(_SIMPLE_PROMPTS)]} (batch #{4000 + idx})"
                tenant = "tenant-batch-analytics"
                priority = PriorityTier.BATCH_BACKGROUND
                rag = ()

            workload.append(
                ChatCompletionRequest(
                    model="prism-auto",
                    messages=(ChatMessage(role="user", content=prompt),),
                    tenant_id=tenant,
                    priority=priority,
                    rag_context=rag,
                    max_tokens=160,
                )
            )
        return workload

    async def run(
        self,
        *,
        requests: int = 300,
        concurrency: int = 32,
        inject_congestion_spike: bool = True,
    ) -> BenchmarkSummary:
        """Execute concurrent workload and compute latency/cost/shedding statistics."""
        workload = self.build_workload(requests)
        sem = asyncio.Semaphore(max(1, concurrency))

        overheads_ms: list[float] = []
        e2e_ms_list: list[float] = []
        succeeded = 0
        shed_or_limited = 0
        critical_total = 0
        critical_admitted = 0
        batch_shed = 0

        sim_pool = (
            self.engine.upstream
            if isinstance(self.engine.upstream, SimulatedUpstreamPool)
            else None
        )

        t_start = time.perf_counter()

        async def _worker(idx: int, req: ChatCompletionRequest) -> None:
            nonlocal succeeded, shed_or_limited, critical_total, critical_admitted, batch_shed
            if req.priority == PriorityTier.CRITICAL:
                critical_total += 1

            # Inject temporary congestion window across middle third of traffic
            if inject_congestion_spike and sim_pool is not None:
                if requests // 3 <= idx < (2 * requests) // 3:
                    sim_pool.set_overload_injection("vllm-qwen-32b")
                else:
                    sim_pool.set_overload_injection(None)

            async with sem:
                req_t0 = time.perf_counter()
                try:
                    resp = await self.engine.complete(req)
                    req_elapsed_ms = (time.perf_counter() - req_t0) * 1000.0
                    succeeded += 1
                    if req.priority == PriorityTier.CRITICAL:
                        critical_admitted += 1
                    overheads_ms.append(resp.prism_meta.gateway_overhead_ms)
                    e2e_ms_list.append(max(req_elapsed_ms, resp.prism_meta.upstream_rtt_ms))
                except GatewayRejectedError as exc:
                    shed_or_limited += 1
                    if req.priority == PriorityTier.BATCH_BACKGROUND and "SHED" in exc.reason:
                        batch_shed += 1
                    await asyncio.sleep(0.002)

        try:
            await asyncio.gather(*(_worker(i, r) for i, r in enumerate(workload)))
        finally:
            if sim_pool is not None:
                sim_pool.set_overload_injection(None)

        wall_ms = max(0.1, (time.perf_counter() - t_start) * 1000.0)
        sorted_overhead = sorted(overheads_ms)
        sorted_e2e = sorted(e2e_ms_list)
        throughput = (requests / wall_ms) * 1000.0

        return BenchmarkSummary(
            total_requests=requests,
            concurrency=concurrency,
            succeeded_requests=succeeded,
            shed_or_rate_limited=shed_or_limited,
            critical_admitted=critical_admitted,
            critical_total=critical_total,
            batch_shed=batch_shed,
            elapsed_wall_ms=round(wall_ms, 2),
            throughput_rps=round(throughput, 2),
            gateway_overhead_p50_ms=round(_quantile(sorted_overhead, 0.50), 3),
            gateway_overhead_p95_ms=round(_quantile(sorted_overhead, 0.95), 3),
            gateway_overhead_p99_ms=round(_quantile(sorted_overhead, 0.99), 3),
            e2e_latency_p50_ms=round(_quantile(sorted_e2e, 0.50), 3),
            e2e_latency_p95_ms=round(_quantile(sorted_e2e, 0.95), 3),
            e2e_latency_p99_ms=round(_quantile(sorted_e2e, 0.99), 3),
            finops=self.engine.router.finops_summary(),
            endpoints=self.engine.endpoint_snapshots(),
        )
