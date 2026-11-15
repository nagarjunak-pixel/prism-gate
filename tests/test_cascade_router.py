"""Tests for complexity scoring, tenant semantic cache, and speculative cascade routing."""

from __future__ import annotations

import pytest

from prism_gate.concur.vegas_limiter import VegasAdaptiveLimiter
from prism_gate.routing.cascade_router import (
    PromptComplexityScorer,
    SpeculativeCascadeRouter,
    TenantSemanticCache,
)
from prism_gate.types import ChatCompletionRequest, ChatMessage, ModelTier


def test_complexity_scorer_separates_simple_and_frontier_prompts() -> None:
    scorer = PromptComplexityScorer()
    simple = scorer.score_text("Hello, what are the support hours?")
    complex_q = scorer.score_text(
        "Derive step by step the convergence theorem for Jensen-Shannon divergence, "
        "compare and contrast asymptotic bounds, and implement an async def Python "
        "algorithm that must strictly emit valid JSON without using external libraries."
    )
    assert 0.0 <= simple.score < 0.25
    assert 0.55 <= complex_q.score <= 1.0
    assert complex_q.reasoning_depth > simple.reasoning_depth
    assert complex_q.code_math_density > simple.code_math_density


def test_semantic_cache_paraphrase_hit_tenant_isolation_and_guards() -> None:
    with pytest.raises(ValueError, match="dim"):
        TenantSemanticCache(dim=16)
    with pytest.raises(ValueError, match="similarity_threshold"):
        TenantSemanticCache(similarity_threshold=0.2)

    now = 10.0

    def clock() -> float:
        return now

    cache = TenantSemanticCache(
        dim=256,
        similarity_threshold=0.84,
        max_entries_per_tenant=2,
        ttl_seconds=60.0,
        clock=clock,
    )

    q1 = "What are the standard enterprise support hours for EMEA customers?"
    q1_paraphrase = "Could you tell me the standard EMEA customer enterprise support hours?"

    cache.store(
        tenant_id="tenant-A",
        prompt_text=q1,
        rag_context=(),
        response_text="EMEA enterprise support runs 24/7.",
        model_id="gemini-2.5-flash",
        tier=ModelTier.FAST,
        completion_tokens=9,
    )

    # Same tenant paraphrase -> HIT
    hit = cache.lookup(tenant_id="tenant-A", prompt_text=q1_paraphrase)
    assert hit.hit
    assert hit.similarity >= 0.84
    assert hit.response_text == "EMEA enterprise support runs 24/7."

    # Different tenant -> STRICT ISOLATION MISS
    cross_tenant = cache.lookup(tenant_id="tenant-B", prompt_text=q1)
    assert not cross_tenant.hit

    # Negation guard prevents opposite-polarity collision
    negated = cache.lookup(
        tenant_id="tenant-A",
        prompt_text="What are not the standard enterprise support hours for EMEA customers?",
    )
    assert not negated.hit

    # Numeric guard prevents distinct numeric collision
    cache.store(
        tenant_id="tenant-A",
        prompt_text="Summarize invoice #1042 settlement status.",
        rag_context=(),
        response_text="Invoice #1042 is paid.",
        model_id="gemini-2.5-flash",
        tier=ModelTier.FAST,
        completion_tokens=6,
    )
    wrong_num = cache.lookup(
        tenant_id="tenant-A",
        prompt_text="Summarize invoice #9999 settlement status.",
    )
    assert not wrong_num.hit

    # Adding 3rd entry evicts oldest LRU entry (max_entries_per_tenant=2)
    cache.store(
        tenant_id="tenant-A",
        prompt_text="Third distinct entry for cache capacity test.",
        rag_context=("chunk-1",),
        response_text="Chunk response.",
        model_id="gemini-2.5-pro",
        tier=ModelTier.FRONTIER,
        completion_tokens=5,
    )
    rag_hit = cache.lookup(
        tenant_id="tenant-A",
        prompt_text="Third distinct entry for cache capacity test.",
        rag_context=("chunk-1",),
    )
    assert rag_hit.hit

    # Advance past TTL -> expires cleanly
    now = 100.0
    expired = cache.lookup(tenant_id="tenant-A", prompt_text=q1_paraphrase)
    assert not expired.hit


def test_cascade_router_planning_escalation_and_finops_ledger() -> None:
    with pytest.raises(ValueError, match="complexity_threshold"):
        SpeculativeCascadeRouter(complexity_threshold=1.5)

    router = SpeculativeCascadeRouter(complexity_threshold=0.48)
    limiters = {
        "gemini-2.5-flash": VegasAdaptiveLimiter("gemini-2.5-flash", tier=ModelTier.FAST),
        "vllm-qwen-32b": VegasAdaptiveLimiter("vllm-qwen-32b", tier=ModelTier.FAST),
        "gemini-2.5-pro": VegasAdaptiveLimiter("gemini-2.5-pro", tier=ModelTier.FRONTIER),
        "claude-sonnet-4": VegasAdaptiveLimiter("claude-sonnet-4", tier=ModelTier.FRONTIER),
    }

    # Explicit model pin
    pinned = ChatCompletionRequest(
        model="claude-sonnet-4",
        messages=(ChatMessage(role="user", content="Quick ping"),),
    )
    plan_pin = router.plan_route(pinned, limiters)
    assert plan_pin.primary_model == "claude-sonnet-4"
    assert not plan_pin.allow_escalation

    # Escalation trigger on low confidence or low RAG faithfulness
    assert router.should_escalate(confidence_score=0.55, min_confidence=0.72)
    assert router.should_escalate(
        confidence_score=0.90, min_confidence=0.72, faithfulness_score=0.50
    )
    assert not router.should_escalate(
        confidence_score=0.90, min_confidence=0.72, faithfulness_score=0.95
    )

    fast_cost = router.compute_model_cost(
        "gemini-2.5-flash", prompt_tokens=500, completion_tokens=200
    )
    cf, saved = router.record_finops(
        prompt_tokens=500,
        completion_tokens=200,
        final_tier=ModelTier.FAST,
        actual_cost_usd=fast_cost,
        escalated=False,
        cache_hit=False,
    )
    assert cf > fast_cost
    assert saved > 0.0
    assert router.finops_summary().savings_pct > 80.0
