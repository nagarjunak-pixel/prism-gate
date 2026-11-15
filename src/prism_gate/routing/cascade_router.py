"""Complexity-Scored Speculative Cascade Router & Signed-Hash Semantic Cache.

1. `PromptComplexityScorer`: Deterministic structural classifier in `[0.0, 1.0]`
   scoring reasoning depth, code/math density, constraint density, and multi-hop markers.
2. `TenantSemanticCache`: Tenant-isolated signed feature-hashing (Weinberger et al.)
   unit-norm cosine similarity cache with strict numeric & negation safety guards.
3. `SpeculativeCascadeRouter`: Routes low-complexity traffic to fast/cheap endpoints
   (`gemini-2.5-flash`, `vllm-qwen-32b`), escalates low-confidence drafts or complex
   analytical queries to frontier endpoints (`gemini-2.5-pro`, `claude-sonnet-4`), and
   maintains a real-time FinOps ledger comparing actual spend vs static frontier pinning.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from prism_gate.concur.vegas_limiter import VegasAdaptiveLimiter
from prism_gate.types import (
    ChatCompletionRequest,
    ComplexityBreakdown,
    FinOpsSummary,
    ModelTier,
)

_WORD_RE = re.compile(r"[a-zA-Z0-9_']+")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "please",
        "could",
        "would",
        "can",
        "you",
        "tell",
        "me",
        "kindly",
        "hey",
        "hi",
        "hello",
        "for",
        "of",
        "in",
        "on",
        "to",
        "is",
        "are",
        "our",
        "my",
        "what",
        "which",
        "how",
        "do",
        "does",
        "from",
        "with",
    }
)

_NEGATION_TOKENS: frozenset[str] = frozenset(
    {"not", "never", "no", "without", "exclude", "except", "neither", "nor"}
)

_REASONING_MARKERS: tuple[str, ...] = (
    "prove",
    "derive",
    "theorem",
    "lemma",
    "analyze",
    "analyse",
    "trade-off",
    "tradeoff",
    "step by step",
    "step-by-step",
    "root cause",
    "counterexample",
    "formalize",
    "induction",
    "architecture",
    "asymptotic",
    "invariant",
    "converge",
    "convergence",
    "equilibrium",
    "synthesise",
    "synthesize",
    "critique",
    "deduce",
    "verify",
)

_CODE_MATH_MARKERS: tuple[str, ...] = (
    "```",
    "def ",
    "async def",
    "class ",
    "select ",
    "join ",
    "interface ",
    "function ",
    "import ",
    "return ",
    "\\sum",
    "\\int",
    "\\frac",
    "o(n",
    "eigenvalue",
    "jacobian",
    "gradient",
    "divergence",
    "probability",
    "distribution",
    "matrix",
    "equation",
    "algorithm",
    "asyncio",
    "typescript",
    "pydantic",
)

_CONSTRAINT_MARKERS: tuple[str, ...] = (
    "must",
    "strictly",
    "exactly",
    "json schema",
    "valid json",
    "at most",
    "at least",
    "without using",
    "do not",
    "never",
    "constraint",
    "guaranteed",
    "idempotent",
    "thread-safe",
    "deterministic",
)

_MULTIHOP_MARKERS: tuple[str, ...] = (
    "compare and contrast",
    "first",
    "second",
    "finally",
    "given that",
    "assuming",
    "on the other hand",
    "whereas",
    "consequently",
    "multi-tenant",
    "end-to-end",
    "across both",
    "implications",
)


class ModelSpec(BaseModel):
    """Pricing and tier metadata for an upstream model endpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    tier: ModelTier
    cost_per_1k_input_usd: float = Field(ge=0.0)
    cost_per_1k_output_usd: float = Field(ge=0.0)
    base_rtt_ms: float = Field(gt=0.0)


MODEL_CATALOG: dict[str, ModelSpec] = {
    "gemini-2.5-flash": ModelSpec(
        model_id="gemini-2.5-flash",
        tier=ModelTier.FAST,
        cost_per_1k_input_usd=0.00015,
        cost_per_1k_output_usd=0.00060,
        base_rtt_ms=18.0,
    ),
    "vllm-qwen-32b": ModelSpec(
        model_id="vllm-qwen-32b",
        tier=ModelTier.FAST,
        cost_per_1k_input_usd=0.00012,
        cost_per_1k_output_usd=0.00036,
        base_rtt_ms=14.0,
    ),
    "gemini-2.5-pro": ModelSpec(
        model_id="gemini-2.5-pro",
        tier=ModelTier.FRONTIER,
        cost_per_1k_input_usd=0.00125,
        cost_per_1k_output_usd=0.01000,
        base_rtt_ms=42.0,
    ),
    "claude-sonnet-4": ModelSpec(
        model_id="claude-sonnet-4",
        tier=ModelTier.FRONTIER,
        cost_per_1k_input_usd=0.00300,
        cost_per_1k_output_usd=0.01500,
        base_rtt_ms=52.0,
    ),
}

STATIC_FRONTIER_PIN_MODEL = "claude-sonnet-4"


class PromptComplexityScorer:
    """Deterministic structural prompt complexity scorer producing `[0.0, 1.0]`."""

    def score_messages(self, request: ChatCompletionRequest) -> ComplexityBreakdown:
        """Compute interpretable complexity breakdown across conversation turns."""
        combined = "\n".join(msg.content for msg in request.messages)
        return self.score_text(combined, rag_chunks=len(request.rag_context))

    def score_text(self, text: str, *, rag_chunks: int = 0) -> ComplexityBreakdown:
        """Score raw prompt text across 4 structural complexity dimensions."""
        lower = text.lower()
        words = _WORD_RE.findall(lower)
        estimated_tokens = max(1, math.ceil(len(words) * 1.32) + rag_chunks * 48)

        reason_hits = sum(1 for marker in _REASONING_MARKERS if marker in lower)
        reasoning_depth = min(1.0, reason_hits / 3.0)

        code_hits = sum(1 for marker in _CODE_MATH_MARKERS if marker in lower)
        symbol_bonus = min(
            0.35,
            sum(text.count(ch) for ch in ("{", "}", "=>", "==", "<=", ">=", "$$")) * 0.05,
        )
        code_math_density = min(1.0, code_hits / 3.0 + symbol_bonus)

        constraint_hits = sum(1 for marker in _CONSTRAINT_MARKERS if marker in lower)
        numbered_items = len(re.findall(r"(?:^|\n)\s*(?:\d+\.|[-*])\s+", text))
        constraint_density = min(1.0, constraint_hits / 3.0 + min(0.4, numbered_items * 0.10))

        hop_hits = sum(1 for marker in _MULTIHOP_MARKERS if marker in lower)
        question_bonus = min(0.30, max(0, text.count("?") - 1) * 0.12)
        length_factor = min(0.45, len(words) / 180.0)
        rag_factor = min(0.20, rag_chunks * 0.06)
        multihop_markers = min(1.0, hop_hits / 3.0 + question_bonus + length_factor + rag_factor)

        weighted = (
            0.34 * reasoning_depth
            + 0.28 * code_math_density
            + 0.20 * constraint_density
            + 0.18 * multihop_markers
        )
        score = min(1.0, max(0.0, round(weighted, 4)))

        return ComplexityBreakdown(
            score=score,
            reasoning_depth=round(reasoning_depth, 4),
            code_math_density=round(code_math_density, 4),
            constraint_density=round(constraint_density, 4),
            multihop_markers=round(multihop_markers, 4),
            estimated_prompt_tokens=estimated_tokens,
        )


@dataclass(frozen=True)
class _CacheEntry:
    vector: tuple[float, ...]
    numbers_sig: frozenset[str]
    negation_parity: int
    response_text: str
    model_id: str
    tier: ModelTier
    completion_tokens: int
    created_at: float


class CacheLookupResult(BaseModel):
    """Result of a tenant-isolated semantic cache lookup."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hit: bool
    similarity: float = Field(ge=0.0, le=1.0)
    response_text: str = ""
    model_id: str = ""
    tier: ModelTier = ModelTier.FAST
    completion_tokens: int = Field(default=0, ge=0)


class TenantSemanticCache:
    """Tenant-isolated signed feature-hashing cosine semantic cache.

    Embeds normalized content n-grams into an `R^dim` unit sphere via two universal
    BLAKE2b hash functions (`index`, `sign in {-1, +1}`), preserving inner products
    in `O(|tokens|)` without external neural weights while guaranteeing zero cross-tenant
    data leakage.
    """

    def __init__(
        self,
        *,
        dim: int = 256,
        similarity_threshold: float = 0.86,
        max_entries_per_tenant: int = 256,
        ttl_seconds: float = 900.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if dim < 32:
            raise ValueError("dim must be >= 32")
        if not (0.5 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be in [0.5, 1.0]")
        self.dim = dim
        self.similarity_threshold = similarity_threshold
        self.max_entries_per_tenant = max(1, max_entries_per_tenant)
        self.ttl_seconds = max(1.0, ttl_seconds)
        self._clock = clock
        self._partitions: dict[str, OrderedDict[str, _CacheEntry]] = {}

    def embed(self, text: str) -> tuple[float, ...]:
        """Compute deterministic L2-normalized signed random projection vector."""
        raw_tokens = [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]
        tokens = [
            (w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w)
            for w in raw_tokens
        ]
        if not tokens:
            tokens = _WORD_RE.findall(text.lower()) or [text.strip().lower() or "_empty_"]

        features: list[tuple[str, float]] = []
        for tok in tokens:
            features.append((f"u:{tok}", 1.0))
            if len(tok) >= 4:
                for idx in range(len(tok) - 3):
                    features.append((f"c4:{tok[idx : idx + 4]}", 0.35))
        for idx in range(len(tokens) - 1):
            pair = tuple(sorted((tokens[idx], tokens[idx + 1])))
            features.append((f"b:{pair[0]}_{pair[1]}", 0.65))

        vec = [0.0] * self.dim
        for feat, weight in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[bucket] += sign * weight

        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 1e-12:
            vec[0] = 1.0
            return tuple(vec)
        return tuple(v / norm for v in vec)

    @staticmethod
    def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        """Compute bounded `[0.0, 1.0]` positive cosine similarity between unit vectors."""
        dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
        return min(1.0, max(0.0, dot))

    @staticmethod
    def _partition_key(tenant_id: str, rag_context: tuple[str, ...]) -> str:
        if not rag_context:
            return f"{tenant_id}::no_rag"
        ctx_hash = hashlib.blake2b(
            "\x1f".join(rag_context).encode("utf-8"),
            digest_size=12,
        ).hexdigest()
        return f"{tenant_id}::rag:{ctx_hash}"

    @staticmethod
    def _extract_guards(text: str) -> tuple[frozenset[str], int]:
        lower = text.lower()
        nums = frozenset(_NUMBER_RE.findall(lower))
        words = _WORD_RE.findall(lower)
        neg_parity = sum(1 for w in words if w in _NEGATION_TOKENS) % 2
        return nums, neg_parity

    def lookup(
        self,
        *,
        tenant_id: str,
        prompt_text: str,
        rag_context: tuple[str, ...] = (),
    ) -> CacheLookupResult:
        """Look up semantically equivalent cached response strictly inside `tenant_id`."""
        part_key = self._partition_key(tenant_id, rag_context)
        partition = self._partitions.get(part_key)
        if not partition:
            return CacheLookupResult(hit=False, similarity=0.0)

        now = self._clock()
        expired = [k for k, entry in partition.items() if now - entry.created_at > self.ttl_seconds]
        for k in expired:
            del partition[k]
        if not partition:
            return CacheLookupResult(hit=False, similarity=0.0)

        query_vec = self.embed(prompt_text)
        query_nums, query_neg = self._extract_guards(prompt_text)

        best_sim = 0.0
        best_key: str | None = None
        best_entry: _CacheEntry | None = None

        for key, entry in partition.items():
            if entry.numbers_sig != query_nums or entry.negation_parity != query_neg:
                continue
            sim = self.cosine_similarity(query_vec, entry.vector)
            if sim > best_sim:
                best_sim = sim
                best_key = key
                best_entry = entry

        if (
            best_entry is not None
            and best_key is not None
            and best_sim >= self.similarity_threshold
        ):
            partition.move_to_end(best_key)
            return CacheLookupResult(
                hit=True,
                similarity=round(best_sim, 4),
                response_text=best_entry.response_text,
                model_id=best_entry.model_id,
                tier=best_entry.tier,
                completion_tokens=best_entry.completion_tokens,
            )

        return CacheLookupResult(hit=False, similarity=round(best_sim, 4))

    def store(
        self,
        *,
        tenant_id: str,
        prompt_text: str,
        rag_context: tuple[str, ...],
        response_text: str,
        model_id: str,
        tier: ModelTier,
        completion_tokens: int,
    ) -> None:
        """Store a verified completion inside the tenant's isolated semantic partition."""
        part_key = self._partition_key(tenant_id, rag_context)
        partition = self._partitions.setdefault(part_key, OrderedDict())
        exact_key = hashlib.blake2b(
            prompt_text.strip().lower().encode("utf-8"), digest_size=12
        ).hexdigest()
        nums, neg = self._extract_guards(prompt_text)

        partition[exact_key] = _CacheEntry(
            vector=self.embed(prompt_text),
            numbers_sig=nums,
            negation_parity=neg,
            response_text=response_text,
            model_id=model_id,
            tier=tier,
            completion_tokens=max(1, completion_tokens),
            created_at=self._clock(),
        )
        partition.move_to_end(exact_key)
        while len(partition) > self.max_entries_per_tenant:
            partition.popitem(last=False)


class RoutePlan(BaseModel):
    """Initial routing decision computed before upstream invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    primary_model: str
    primary_tier: ModelTier
    fallback_frontier_model: str
    complexity: ComplexityBreakdown
    allow_escalation: bool


class SpeculativeCascadeRouter:
    """Complexity-aware speculative cascade router with live FinOps cost accounting."""

    def __init__(
        self,
        *,
        complexity_threshold: float = 0.48,
        semantic_cache: TenantSemanticCache | None = None,
        scorer: PromptComplexityScorer | None = None,
    ) -> None:
        if not (0.0 < complexity_threshold < 1.0):
            raise ValueError("complexity_threshold must be in (0, 1)")
        self.complexity_threshold = complexity_threshold
        self.cache = semantic_cache or TenantSemanticCache()
        self.scorer = scorer or PromptComplexityScorer()

        self._total_routed = 0
        self._fast_hits = 0
        self._frontier_direct_hits = 0
        self._cascade_escalations = 0
        self._cache_hits = 0
        self._actual_cost_usd = 0.0
        self._counterfactual_frontier_cost_usd = 0.0

    @staticmethod
    def compute_model_cost(
        model_id: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Calculate exact USD inference cost for `(prompt_tokens, completion_tokens)`."""
        spec = MODEL_CATALOG.get(model_id, MODEL_CATALOG[STATIC_FRONTIER_PIN_MODEL])
        in_cost = (max(0, prompt_tokens) / 1000.0) * spec.cost_per_1k_input_usd
        out_cost = (max(0, completion_tokens) / 1000.0) * spec.cost_per_1k_output_usd
        return in_cost + out_cost

    def select_least_loaded(
        self,
        tier: ModelTier,
        limiters: dict[str, VegasAdaptiveLimiter],
    ) -> str:
        """Pick the upstream provider in `tier` maximizing Vegas headroom-to-RTT ratio."""
        candidates = [
            spec
            for spec in MODEL_CATALOG.values()
            if spec.tier == tier and spec.model_id in limiters
        ]
        if not candidates:
            return "gemini-2.5-flash" if tier == ModelTier.FAST else STATIC_FRONTIER_PIN_MODEL

        def _score(spec: ModelSpec) -> float:
            lim = limiters[spec.model_id]
            headroom_ratio = lim.available_headroom() / max(1, lim.max_inflight)
            effective_rtt = lim.snapshot().rtt_ewma_ms or spec.base_rtt_ms
            return headroom_ratio * 100.0 - (effective_rtt * 0.25)

        best = max(candidates, key=_score)
        return best.model_id

    def plan_route(
        self,
        request: ChatCompletionRequest,
        limiters: dict[str, VegasAdaptiveLimiter],
    ) -> RoutePlan:
        """Classify prompt complexity and construct speculative fast->frontier cascade plan."""
        complexity = self.scorer.score_messages(request)
        frontier_pick = self.select_least_loaded(ModelTier.FRONTIER, limiters)

        # Explicit model override if caller pinned a specific catalog model
        if request.model in MODEL_CATALOG:
            spec = MODEL_CATALOG[request.model]
            return RoutePlan(
                primary_model=spec.model_id,
                primary_tier=spec.tier,
                fallback_frontier_model=frontier_pick,
                complexity=complexity,
                allow_escalation=(spec.tier == ModelTier.FAST),
            )

        if complexity.score >= self.complexity_threshold:
            return RoutePlan(
                primary_model=frontier_pick,
                primary_tier=ModelTier.FRONTIER,
                fallback_frontier_model=frontier_pick,
                complexity=complexity,
                allow_escalation=False,
            )

        fast_pick = self.select_least_loaded(ModelTier.FAST, limiters)
        return RoutePlan(
            primary_model=fast_pick,
            primary_tier=ModelTier.FAST,
            fallback_frontier_model=frontier_pick,
            complexity=complexity,
            allow_escalation=True,
        )

    @staticmethod
    def should_escalate(
        *,
        confidence_score: float,
        min_confidence: float,
        faithfulness_score: float | None = None,
    ) -> bool:
        """Decide whether a speculative fast-tier draft must escalate to the frontier tier."""
        if confidence_score < min_confidence:
            return True
        return faithfulness_score is not None and faithfulness_score < 0.75

    def record_finops(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        final_tier: ModelTier,
        actual_cost_usd: float,
        escalated: bool,
        cache_hit: bool,
    ) -> tuple[float, float]:
        """Record request cost vs counterfactual static frontier pinning (`claude-sonnet-4`).

        Returns `(counterfactual_frontier_cost_usd, cost_saved_usd)`.
        """
        counterfactual = self.compute_model_cost(
            STATIC_FRONTIER_PIN_MODEL,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self._total_routed += 1
        if cache_hit:
            self._cache_hits += 1
        elif escalated:
            self._cascade_escalations += 1
        elif final_tier == ModelTier.FAST:
            self._fast_hits += 1
        else:
            self._frontier_direct_hits += 1

        self._actual_cost_usd += actual_cost_usd
        self._counterfactual_frontier_cost_usd += counterfactual
        return counterfactual, counterfactual - actual_cost_usd

    def finops_summary(self) -> FinOpsSummary:
        """Return immutable FinOps cost & savings telemetry."""
        net_saved = self._counterfactual_frontier_cost_usd - self._actual_cost_usd
        savings_pct = (
            (net_saved / self._counterfactual_frontier_cost_usd) * 100.0
            if self._counterfactual_frontier_cost_usd > 1e-12
            else 0.0
        )
        hit_rate = self._cache_hits / self._total_routed if self._total_routed > 0 else 0.0
        return FinOpsSummary(
            total_routed_requests=self._total_routed,
            fast_tier_hits=self._fast_hits,
            frontier_direct_hits=self._frontier_direct_hits,
            cascade_escalations=self._cascade_escalations,
            semantic_cache_hits=self._cache_hits,
            cache_hit_rate=round(hit_rate, 4),
            actual_total_cost_usd=round(self._actual_cost_usd, 6),
            counterfactual_frontier_cost_usd=round(self._counterfactual_frontier_cost_usd, 6),
            net_savings_usd=round(net_saved, 6),
            savings_pct=round(savings_pct, 2),
        )
