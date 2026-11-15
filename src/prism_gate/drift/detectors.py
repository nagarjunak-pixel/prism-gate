"""Statistical Distribution-Shift Detectors (`PSI` + `Jensen-Shannon Divergence`).

Implements exact information-theoretic metrics over LLM gateway telemetry:
1. **Jensen-Shannon Divergence (`JSD`)**:
   - Symmetric, smoothed mutual-information divergence over discrete token/topic
     distributions: `JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)` where
     `M = 0.5 * (P + Q)`.
   - Bounded strictly in `[0.0, 1.0]` in base-2 (`bits`) and `[0.0, ln 2]` in base-e (`nats`).
   - Exact zero on identical distributions `JSD(P, P) == 0.0` and bit-for-bit symmetry
     `JSD(P, Q) == JSD(Q, P)`.
2. **Population Stability Index (`PSI`)**:
   - Symmetric Jeffreys-divergence statistic over binned continuous features
     (prompt length, structural complexity, upstream RTT):
     `PSI = sum_k (q_k - p_k) * ln(q_k / p_k) >= 0`.
"""

from __future__ import annotations

import bisect
import math
import re
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from prism_gate.types import DriftAnalyzeRequest, DriftReport, DriftSeverity, FeaturePSIResult

_TOKEN_RE = re.compile(r"[a-zA-Z]{3,}")


def _normalize_weights(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("Distribution cannot be empty")
    cleaned: list[float] = []
    total = 0.0
    for val in values:
        if not math.isfinite(val) or val < 0.0:
            raise ValueError("Distribution weights must be finite and >= 0")
        cleaned.append(float(val))
        total += float(val)
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("Distribution must have strictly positive finite total mass")
    return [v / total for v in cleaned]


def jensen_shannon_divergence(
    p: Sequence[float] | Mapping[str, float],
    q: Sequence[float] | Mapping[str, float],
    *,
    base: Literal[2, "e"] = 2,
) -> float:
    """Compute exact Jensen-Shannon Divergence `JSD(P || Q)`.

    - In `base=2` (default), returns bits strictly in `[0.0, 1.0]`.
    - In `base="e"`, returns nats strictly in `[0.0, ln(2)]`.
    """
    if isinstance(p, Mapping) and isinstance(q, Mapping):
        if not p or not q:
            raise ValueError("Mapping distributions cannot be empty")
        keys = sorted(set(p.keys()) | set(q.keys()))
        vec_p = [float(p.get(k, 0.0)) for k in keys]
        vec_q = [float(q.get(k, 0.0)) for k in keys]
    elif not isinstance(p, Mapping) and not isinstance(q, Mapping):
        if len(p) != len(q):
            raise ValueError("Sequence distributions P and Q must have equal length")
        vec_p = list(p)
        vec_q = list(q)
    else:
        raise TypeError("Both P and Q must be either Sequences or Mappings")

    prob_p = _normalize_weights(vec_p)
    prob_q = _normalize_weights(vec_q)

    log_fn: Callable[[float], float] = math.log2 if base == 2 else math.log
    upper_bound = 1.0 if base == 2 else math.log(2.0)

    total = 0.0
    for pi, qi in zip(prob_p, prob_q, strict=True):
        if pi == qi:
            continue
        mi = 0.5 * (pi + qi)
        term_p = pi * log_fn(pi / mi) if pi > 0.0 else 0.0
        term_q = qi * log_fn(qi / mi) if qi > 0.0 else 0.0
        # Commutative per-coordinate sum ensures exact bit-for-bit symmetry JSD(P, Q) == JSD(Q, P)
        total += 0.5 * (term_p + term_q)

    return min(upper_bound, max(0.0, total))


def psi_from_distributions(
    baseline_weights: Sequence[float],
    live_weights: Sequence[float],
    *,
    epsilon: float = 1e-4,
) -> float:
    """Compute non-negative symmetric Population Stability Index over aligned bin weights."""
    if len(baseline_weights) != len(live_weights):
        raise ValueError("baseline_weights and live_weights must have equal length")
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be a positive finite float")

    prob_p = _normalize_weights(baseline_weights)
    prob_q = _normalize_weights(live_weights)
    k = len(prob_p)
    denom = 1.0 + k * epsilon

    psi = 0.0
    for pi, qi in zip(prob_p, prob_q, strict=True):
        p_smooth = (pi + epsilon) / denom
        q_smooth = (qi + epsilon) / denom
        if p_smooth == q_smooth:
            continue
        psi += (q_smooth - p_smooth) * math.log(q_smooth / p_smooth)

    return max(0.0, psi)


def classify_psi_severity(psi_value: float) -> DriftSeverity:
    """Classify PSI value using canonical credit/ML telemetry thresholds (`0.10`, `0.25`)."""
    if psi_value >= 0.25:
        return DriftSeverity.CRITICAL_DRIFT
    if psi_value >= 0.10:
        return DriftSeverity.MODERATE_DRIFT
    return DriftSeverity.NO_DRIFT


def classify_jsd_severity(jsd_bits: float) -> DriftSeverity:
    """Classify base-2 Jensen-Shannon Divergence (`[0, 1]` bits) into drift alert tiers."""
    if jsd_bits >= 0.20:
        return DriftSeverity.CRITICAL_DRIFT
    if jsd_bits >= 0.08:
        return DriftSeverity.MODERATE_DRIFT
    return DriftSeverity.NO_DRIFT


def _build_interior_edges(sorted_base: list[float], num_bins: int) -> list[float]:
    n = len(sorted_base)
    min_val = sorted_base[0]
    max_val = sorted_base[-1]
    if min_val == max_val:
        span = max(1e-3, abs(min_val) * 0.05)
        return [min_val - span, min_val + span]

    raw_edges: list[float] = []
    for b in range(1, num_bins):
        pos = (b / num_bins) * (n - 1)
        lo = math.floor(pos)
        hi = min(n - 1, lo + 1)
        frac = pos - lo
        edge = sorted_base[lo] + frac * (sorted_base[hi] - sorted_base[lo])
        if not raw_edges or edge > raw_edges[-1] + 1e-12:
            raw_edges.append(edge)

    if not raw_edges:
        mid = 0.5 * (min_val + max_val)
        return [mid]
    return raw_edges


def population_stability_index(
    baseline: Sequence[float],
    live: Sequence[float],
    *,
    feature_name: str = "feature",
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> FeaturePSIResult:
    """Compute binned Population Stability Index comparing `live` window vs `baseline`."""
    if not baseline or not live:
        raise ValueError("baseline and live sample arrays must be non-empty")
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2")

    clean_base: list[float] = []
    for x in baseline:
        if not math.isfinite(x):
            raise ValueError("baseline samples must be finite")
        clean_base.append(float(x))

    clean_live: list[float] = []
    for x in live:
        if not math.isfinite(x):
            raise ValueError("live samples must be finite")
        clean_live.append(float(x))

    sorted_base = sorted(clean_base)
    interior_edges = _build_interior_edges(sorted_base, num_bins)
    k = len(interior_edges) + 1

    base_counts = [0] * k
    for x in clean_base:
        idx = bisect.bisect_right(interior_edges, x)
        base_counts[idx] += 1

    live_counts = [0] * k
    for x in clean_live:
        idx = bisect.bisect_right(interior_edges, x)
        live_counts[idx] += 1

    base_props = [c / len(clean_base) for c in base_counts]
    live_props = [c / len(clean_live) for c in live_counts]
    psi_val = psi_from_distributions(base_props, live_props, epsilon=epsilon)
    severity = classify_psi_severity(psi_val)

    display_edges = (
        round(min(sorted_base[0], min(clean_live)), 4),
        *(round(e, 4) for e in interior_edges),
        round(max(sorted_base[-1], max(clean_live)), 4),
    )

    return FeaturePSIResult(
        feature_name=feature_name,
        psi_value=round(psi_val, 6),
        severity=severity,
        bin_edges=display_edges,
        baseline_proportions=tuple(round(p, 4) for p in base_props),
        live_proportions=tuple(round(q, 4) for q in live_props),
    )


_SEVERITY_RANK: dict[DriftSeverity, int] = {
    DriftSeverity.NO_DRIFT: 0,
    DriftSeverity.MODERATE_DRIFT: 1,
    DriftSeverity.CRITICAL_DRIFT: 2,
}


class DriftPlaneAnalyzer:
    """Continuous telemetry distribution-shift analyzer combining PSI and JSD."""

    def __init__(self, *, window_capacity: int = 400) -> None:
        self.window_capacity = max(32, window_capacity)
        self._baseline_lengths, self._baseline_complexities, self._baseline_latencies = (
            self._reference_continuous()
        )
        self._baseline_tokens, self._baseline_topics = self._reference_discrete()

        self._live_lengths: deque[float] = deque(
            self._baseline_lengths[:120], maxlen=self.window_capacity
        )
        self._live_complexities: deque[float] = deque(
            self._baseline_complexities[:120], maxlen=self.window_capacity
        )
        self._live_latencies: deque[float] = deque(
            self._baseline_latencies[:120], maxlen=self.window_capacity
        )
        self._live_tokens: Counter[str] = Counter(
            {k: int(v) for k, v in self._baseline_tokens.items()}
        )
        self._live_topics: Counter[str] = Counter(
            {k: int(v) for k, v in self._baseline_topics.items()}
        )
        self._latest_report: DriftReport = self.analyze_preset("normal")

    @staticmethod
    def _reference_continuous() -> tuple[list[float], list[float], list[float]]:
        lengths = [float(28 + (i * 17) % 130 + (15 if i % 5 == 0 else 0)) for i in range(200)]
        complexities = [round(0.12 + ((i * 23) % 68) / 100.0, 3) for i in range(200)]
        latencies = [float(16 + (i * 13) % 44 + (18 if i % 7 == 0 else 0)) for i in range(200)]
        return lengths, complexities, latencies

    @staticmethod
    def _reference_discrete() -> tuple[dict[str, float], dict[str, float]]:
        tokens: dict[str, float] = {
            "revenue": 140.0,
            "quarter": 125.0,
            "latency": 110.0,
            "customer": 135.0,
            "summary": 150.0,
            "invoice": 95.0,
            "architecture": 80.0,
            "python": 90.0,
            "database": 85.0,
            "policy": 100.0,
        }
        topics: dict[str, float] = {
            "enterprise_qa": 0.36,
            "financial_rag": 0.24,
            "code_assistant": 0.20,
            "summarization": 0.14,
            "compliance": 0.06,
        }
        return tokens, topics

    @staticmethod
    def infer_topic(prompt_text: str) -> str:
        lower = prompt_text.lower()
        if any(k in lower for k in ("revenue", "ebitda", "gaap", "margin", "quarter", "invoice")):
            return "financial_rag"
        if any(k in lower for k in ("def ", "python", "sql", "class ", "async", "code", "bug")):
            return "code_assistant"
        if any(k in lower for k in ("summarize", "summary", "tl;dr", "brief")):
            return "summarization"
        if any(k in lower for k in ("hipaa", "soc2", "policy", "dosage", "clinical", "warfarin")):
            return "compliance"
        return "enterprise_qa"

    def record_observation(
        self,
        *,
        prompt_text: str,
        prompt_tokens: int,
        complexity_score: float,
        latency_ms: float,
    ) -> None:
        """Ingest a live gateway request into the sliding telemetry window."""
        self._live_lengths.append(float(max(1, prompt_tokens)))
        self._live_complexities.append(float(min(1.0, max(0.0, complexity_score))))
        self._live_latencies.append(float(max(0.1, latency_ms)))

        for tok in _TOKEN_RE.findall(prompt_text.lower())[:24]:
            if tok in self._baseline_tokens or len(self._live_tokens) < 64:
                self._live_tokens[tok] += 1
        self._live_topics[self.infer_topic(prompt_text)] += 1

    def analyze_distributions(
        self,
        *,
        baseline_lengths: Sequence[float],
        live_lengths: Sequence[float],
        baseline_complexities: Sequence[float],
        live_complexities: Sequence[float],
        baseline_latencies_ms: Sequence[float],
        live_latencies_ms: Sequence[float],
        baseline_tokens: Mapping[str, float],
        live_tokens: Mapping[str, float],
        baseline_topics: Mapping[str, float],
        live_topics: Mapping[str, float],
    ) -> DriftReport:
        """Run full multi-variate PSI + Jensen-Shannon drift evaluation."""
        psi_len = population_stability_index(
            baseline_lengths,
            live_lengths,
            feature_name="prompt_length_tokens",
        )
        psi_comp = population_stability_index(
            baseline_complexities,
            live_complexities,
            feature_name="complexity_score",
        )
        psi_lat = population_stability_index(
            baseline_latencies_ms,
            live_latencies_ms,
            feature_name="upstream_latency_ms",
        )

        features = (psi_len, psi_comp, psi_lat)
        max_psi = max(f.psi_value for f in features)
        token_jsd = round(jensen_shannon_divergence(baseline_tokens, live_tokens, base=2), 6)
        topic_jsd = round(jensen_shannon_divergence(baseline_topics, live_topics, base=2), 6)

        token_sev = classify_jsd_severity(token_jsd)
        topic_sev = classify_jsd_severity(topic_jsd)

        all_severities = [*(f.severity for f in features), token_sev, topic_sev]
        overall = max(all_severities, key=lambda s: _SEVERITY_RANK[s])

        reasons: list[str] = []
        for feat in features:
            if feat.severity != DriftSeverity.NO_DRIFT:
                reasons.append(
                    f"{feat.feature_name} PSI={feat.psi_value:.4f} ({feat.severity.value})"
                )
        if token_sev != DriftSeverity.NO_DRIFT:
            reasons.append(f"token_distribution JSD={token_jsd:.4f} bits ({token_sev.value})")
        if topic_sev != DriftSeverity.NO_DRIFT:
            reasons.append(f"topic_distribution JSD={topic_jsd:.4f} bits ({topic_sev.value})")

        report = DriftReport(
            overall_severity=overall,
            max_psi=round(max_psi, 6),
            token_jsd=token_jsd,
            topic_jsd=topic_jsd,
            feature_psi=features,
            baseline_sample_size=len(baseline_lengths),
            live_sample_size=len(live_lengths),
            alert_reasons=tuple(reasons),
        )
        self._latest_report = report
        return report

    def analyze_preset(
        self,
        preset: Literal["normal", "moderate", "critical", "live_window"],
    ) -> DriftReport:
        """Evaluate one of three deterministic telemetry scenarios or the live sliding window."""
        if preset == "live_window":
            return self.analyze_distributions(
                baseline_lengths=self._baseline_lengths,
                live_lengths=list(self._live_lengths),
                baseline_complexities=self._baseline_complexities,
                live_complexities=list(self._live_complexities),
                baseline_latencies_ms=self._baseline_latencies,
                live_latencies_ms=list(self._live_latencies),
                baseline_tokens=self._baseline_tokens,
                live_tokens={k: float(v) for k, v in self._live_tokens.items()},
                baseline_topics=self._baseline_topics,
                live_topics={k: float(v) for k, v in self._live_topics.items()},
            )

        if preset == "normal":
            # Stable traffic with tiny organic sampling jitter -> NO_DRIFT
            live_len = [
                x + (2.0 if idx % 3 == 0 else -1.5) for idx, x in enumerate(self._baseline_lengths)
            ]
            live_comp = [
                min(1.0, max(0.0, x + (0.01 if idx % 2 == 0 else -0.01)))
                for idx, x in enumerate(self._baseline_complexities)
            ]
            live_lat = [
                x + (1.0 if idx % 4 == 0 else -0.8)
                for idx, x in enumerate(self._baseline_latencies)
            ]
            live_tok = {
                k: v * (1.04 if idx % 2 == 0 else 0.97)
                for idx, (k, v) in enumerate(self._baseline_tokens.items())
            }
            live_top = {
                "enterprise_qa": 0.35,
                "financial_rag": 0.25,
                "code_assistant": 0.20,
                "summarization": 0.14,
                "compliance": 0.06,
            }
            return self.analyze_distributions(
                baseline_lengths=self._baseline_lengths,
                live_lengths=live_len,
                baseline_complexities=self._baseline_complexities,
                live_complexities=live_comp,
                baseline_latencies_ms=self._baseline_latencies,
                live_latencies_ms=live_lat,
                baseline_tokens=self._baseline_tokens,
                live_tokens=live_tok,
                baseline_topics=self._baseline_topics,
                live_topics=live_top,
            )

        if preset == "moderate":
            # Partial cohort shift (~28% traffic shifting to code prompts) -> MODERATE_DRIFT
            live_len = [
                (x * 1.32 + 22.0 if idx % 4 == 0 else x + 3.0)
                for idx, x in enumerate(self._baseline_lengths)
            ]
            live_comp = [
                min(0.95, x + 0.16 if idx % 4 == 0 else x + 0.02)
                for idx, x in enumerate(self._baseline_complexities)
            ]
            live_lat = [
                (x * 1.25 + 8.0 if idx % 4 == 0 else x + 1.5)
                for idx, x in enumerate(self._baseline_latencies)
            ]
            live_tok = {
                "revenue": 75.0,
                "quarter": 68.0,
                "latency": 145.0,
                "customer": 72.0,
                "summary": 80.0,
                "invoice": 50.0,
                "architecture": 210.0,
                "python": 260.0,
                "database": 195.0,
                "policy": 55.0,
            }
            live_top = {
                "enterprise_qa": 0.19,
                "financial_rag": 0.12,
                "code_assistant": 0.51,
                "summarization": 0.10,
                "compliance": 0.08,
            }
            return self.analyze_distributions(
                baseline_lengths=self._baseline_lengths,
                live_lengths=live_len,
                baseline_complexities=self._baseline_complexities,
                live_complexities=live_comp,
                baseline_latencies_ms=self._baseline_latencies,
                live_latencies_ms=live_lat,
                baseline_tokens=self._baseline_tokens,
                live_tokens=live_tok,
                baseline_topics=self._baseline_topics,
                live_topics=live_top,
            )

        # preset == "critical": Adversarial prompt-dump / tail-latency surge -> CRITICAL_DRIFT
        live_len = [x * 3.4 + 280.0 for x in self._baseline_lengths]
        live_comp = [
            min(1.0, 0.78 + (idx % 20) * 0.01) for idx in range(len(self._baseline_complexities))
        ]
        live_lat = [x * 4.2 + 190.0 for x in self._baseline_latencies]
        live_tok = {
            "revenue": 12.0,
            "quarter": 10.0,
            "latency": 25.0,
            "customer": 15.0,
            "summary": 14.0,
            "override": 420.0,
            "jailbreak": 380.0,
            "exfiltrate": 310.0,
            "system_prompt": 350.0,
            "unrestricted": 290.0,
        }
        live_top = {
            "enterprise_qa": 0.04,
            "financial_rag": 0.03,
            "code_assistant": 0.08,
            "summarization": 0.02,
            "adversarial_anomaly": 0.83,
        }
        return self.analyze_distributions(
            baseline_lengths=self._baseline_lengths,
            live_lengths=live_len,
            baseline_complexities=self._baseline_complexities,
            live_complexities=live_comp,
            baseline_latencies_ms=self._baseline_latencies,
            live_latencies_ms=live_lat,
            baseline_tokens=self._baseline_tokens,
            live_tokens=live_tok,
            baseline_topics=self._baseline_topics,
            live_topics=live_top,
        )

    def analyze_request(self, req: DriftAnalyzeRequest) -> DriftReport:
        """Handle custom REST payload or named preset."""
        if req.preset is not None:
            return self.analyze_preset(req.preset)

        return self.analyze_distributions(
            baseline_lengths=req.baseline_prompt_lengths or self._baseline_lengths,
            live_lengths=req.live_prompt_lengths or list(self._live_lengths),
            baseline_complexities=req.baseline_complexities or self._baseline_complexities,
            live_complexities=req.live_complexities or list(self._live_complexities),
            baseline_latencies_ms=req.baseline_latencies_ms or self._baseline_latencies,
            live_latencies_ms=req.live_latencies_ms or list(self._live_latencies),
            baseline_tokens=req.baseline_token_counts or self._baseline_tokens,
            live_tokens=req.live_token_counts
            or {k: float(v) for k, v in self._live_tokens.items()},
            baseline_topics=req.baseline_topic_counts or self._baseline_topics,
            live_topics=req.live_topic_counts
            or {k: float(v) for k, v in self._live_topics.items()},
        )

    @property
    def latest_report(self) -> DriftReport:
        """Most recently computed `DriftReport`."""
        return self._latest_report
