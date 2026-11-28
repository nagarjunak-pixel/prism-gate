"""Bounded zero-text decision routing, PipeAxes prefill slicing, and switch-adapted drift.

Implements four inference-gateway latency, determinism, and FinOps primitives:
1. `BoundedDecisionRouter`: Separates closed-set `1-of-N` probability classification
   (intent routing, escalation gates, rerank selection) from open-ended generative prose,
   eliminating invalid schema outputs while flagging `bfloat16` near-tie rank-flip
   margins (arXiv:2611.34462) for `float32` tie-break or looped-depth escalation
   (arXiv:2611.34923).
2. `compute_pipeaxes_slice_plan` (arXiv:2611.34615): Computes latency-slack-bounded
   prefill slice sizes with Hadamard-INT4 activation transport so `64K`-token prefills
   do not head-of-line-block interactive `<2s` TTFT SLO traffic.
3. `SwitchAdaptedDriftTracker` (arXiv:2611.34905): Two-block regime-switching Bayes
   estimator that adapts faster than fixed sliding windows upon abrupt provider drift.
4. `SignedBillingReceipt` (arXiv:2611.34623): Cryptographic `HMAC-SHA256` token-metering
   receipt binding `(request_id, model_id, prompt_tokens, completion_tokens, stream_sha256)`.
"""

from __future__ import annotations

import hashlib
import hmac
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

_BF16_NEAR_TIE_LOGIT_EPS: Final[float] = 0.015625  # 2^-6 (~1 ULP near 2.0 in bfloat16)


@dataclass(frozen=True)
class BoundedDecisionVerdict:
    """Routing and closed-set classification outcome for a bounded `1-of-N` decision."""

    route: str
    selected_option: str
    selected_index: int
    probabilities: tuple[float, ...]
    top1_top2_margin: float
    normalized_entropy: float
    near_tie_bf16_risk: bool


@dataclass(frozen=True)
class PipeAxesSlicePlan:
    """Prefill micro-slice plan preventing long-context head-of-line TTFT stalls."""

    prompt_tokens: int
    slice_tokens: int
    num_slices: int
    transport_compression_ratio: float
    estimated_short_req_p95_ttft_ms: float


@dataclass(frozen=True)
class SignedBillingReceipt:
    """Verifiable Layer-4 token-accounting receipt for MaaS FinOps auditing."""

    request_id: str
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    stream_sha256: str
    signature_hex: str

    @classmethod
    def issue(
        cls,
        signing_key: bytes,
        request_id: str,
        model_id: str,
        prompt_tokens: int,
        completion_chunks: Sequence[str],
    ) -> SignedBillingReceipt:
        if len(signing_key) < 16:
            raise ValueError("signing_key must be at least 16 bytes")
        if prompt_tokens < 0:
            raise ValueError("prompt_tokens must be >= 0")
        joined = "".join(completion_chunks)
        comp_tokens = sum(len(chunk.split()) for chunk in completion_chunks if chunk.strip())
        stream_digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
        payload = f"{request_id}|{model_id}|{prompt_tokens}|{comp_tokens}|{stream_digest}"
        sig = hmac.new(signing_key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return cls(
            request_id=request_id,
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=comp_tokens,
            stream_sha256=stream_digest,
            signature_hex=sig,
        )

    def verify(self, signing_key: bytes, observed_chunks: Sequence[str]) -> bool:
        joined = "".join(observed_chunks)
        observed_digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(observed_digest, self.stream_sha256):
            return False
        observed_tokens = sum(len(chunk.split()) for chunk in observed_chunks if chunk.strip())
        if observed_tokens != self.completion_tokens:
            return False
        payload = (
            f"{self.request_id}|{self.model_id}|{self.prompt_tokens}|"
            f"{self.completion_tokens}|{self.stream_sha256}"
        )
        expected_sig = hmac.new(signing_key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected_sig, self.signature_hex)


class BoundedDecisionRouter:
    """Routes confident closed-set `1-of-N` decisions to zero-text structured execution."""

    def __init__(
        self,
        min_margin: float = 0.15,
        max_normalized_entropy: float = 0.65,
        temperature: float = 1.0,
    ) -> None:
        if not (0.0 <= min_margin <= 1.0):
            raise ValueError("min_margin must be in [0, 1]")
        if not (0.0 < max_normalized_entropy <= 1.0):
            raise ValueError("max_normalized_entropy must be in (0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        self.min_margin = min_margin
        self.max_normalized_entropy = max_normalized_entropy
        self.temperature = temperature

    def evaluate(self, options: Sequence[str], logits: Sequence[float]) -> BoundedDecisionVerdict:
        """Compute calibrated probabilities, near-tie risk, and bounded vs looped route."""
        if len(options) < 2 or len(options) != len(logits):
            raise ValueError("options and logits must have equal length >= 2")

        scaled = [float(x) / self.temperature for x in logits]
        max_logit = max(scaled)
        exps = [math.exp(x - max_logit) for x in scaled]
        z = sum(exps)
        probs = tuple(v / z for v in exps)

        ranked_indices = sorted(range(len(options)), key=lambda idx: (-scaled[idx], idx))
        best_idx = ranked_indices[0]
        second_idx = ranked_indices[1]

        margin = probs[best_idx] - probs[second_idx]
        logit_gap = abs(float(logits[best_idx]) - float(logits[second_idx]))
        near_tie = logit_gap < _BF16_NEAR_TIE_LOGIT_EPS

        entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
        norm_entropy = entropy / math.log(len(options))

        if (
            margin >= self.min_margin
            and norm_entropy <= self.max_normalized_entropy
            and not near_tie
        ):
            route = "bounded_zero_text"
        else:
            route = "looped_reasoning_tier"

        return BoundedDecisionVerdict(
            route=route,
            selected_option=options[best_idx],
            selected_index=best_idx,
            probabilities=probs,
            top1_top2_margin=margin,
            normalized_entropy=norm_entropy,
            near_tie_bf16_risk=near_tie,
        )


def compute_pipeaxes_slice_plan(
    prompt_tokens: int,
    ttft_slo_ms: float = 600.0,
    tokens_per_ms_throughput: float = 24.0,
    hadamard_int4: bool = True,
) -> PipeAxesSlicePlan:
    """Compute latency-slack-bounded prefill slice sizing (arXiv:2611.34615)."""
    if prompt_tokens < 1:
        raise ValueError("prompt_tokens must be >= 1")
    if ttft_slo_ms <= 20.0:
        raise ValueError("ttft_slo_ms must be > 20.0")
    if tokens_per_ms_throughput <= 0.0:
        raise ValueError("tokens_per_ms_throughput must be > 0")

    compression = 3.75 if hadamard_int4 else 1.0
    slack_budget_ms = 0.65 * ttft_slo_ms
    raw_slice = int(max(128.0, min(4096.0, slack_budget_ms * tokens_per_ms_throughput / 4.0)))
    slice_tokens = min(prompt_tokens, raw_slice)
    num_slices = math.ceil(prompt_tokens / slice_tokens)
    transport_ms = (slice_tokens / tokens_per_ms_throughput) * (1.0 + 0.85 / compression)
    p95_ttft_ms = round(min(ttft_slo_ms * 0.92, 48.0 + transport_ms), 2)
    return PipeAxesSlicePlan(
        prompt_tokens=prompt_tokens,
        slice_tokens=slice_tokens,
        num_slices=num_slices,
        transport_compression_ratio=compression,
        estimated_short_req_p95_ttft_ms=p95_ttft_ms,
    )


class SwitchAdaptedDriftTracker:
    """Two-block regime-switch Bayes posterior estimator for non-stationary error rates."""

    def __init__(self, hazard_rate: float = 0.08, prior_alpha: float = 1.0) -> None:
        if not (0.0 < hazard_rate < 1.0):
            raise ValueError("hazard_rate must be in (0, 1)")
        if prior_alpha <= 0.0:
            raise ValueError("prior_alpha must be > 0")
        self._hazard = hazard_rate
        self._prior = prior_alpha
        self._stable_alpha = prior_alpha
        self._stable_beta = prior_alpha
        self._recent_alpha = prior_alpha
        self._recent_beta = prior_alpha
        self._switch_prob = 0.0

    def update(self, error_observed: bool) -> tuple[float, float]:
        """Ingest one binary outcome and return `(posterior_error_rate, switch_probability)`."""
        err = 1.0 if error_observed else 0.0
        ok = 1.0 - err

        stable_pred = self._stable_alpha / (self._stable_alpha + self._stable_beta)
        recent_pred = self._recent_alpha / (self._recent_alpha + self._recent_beta)

        l_stable = stable_pred if error_observed else (1.0 - stable_pred)
        l_recent = recent_pred if error_observed else (1.0 - recent_pred)

        prior_switch = (1.0 - self._switch_prob) * self._hazard + self._switch_prob * (
            1.0 - self._hazard * 0.25
        )
        num = prior_switch * l_recent
        den = num + (1.0 - prior_switch) * l_stable
        self._switch_prob = num / den if den > 0.0 else 0.0

        self._stable_alpha = 0.96 * self._stable_alpha + err
        self._stable_beta = 0.96 * self._stable_beta + ok
        self._recent_alpha = 0.65 * self._recent_alpha + err + 0.1 * self._prior
        self._recent_beta = 0.65 * self._recent_beta + ok + 0.1 * self._prior

        post_stable = self._stable_alpha / (self._stable_alpha + self._stable_beta)
        post_recent = self._recent_alpha / (self._recent_alpha + self._recent_beta)
        blended = (1.0 - self._switch_prob) * post_stable + self._switch_prob * post_recent
        return blended, self._switch_prob
