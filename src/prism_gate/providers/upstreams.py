"""Upstream LLM Inference Providers (`SimulatedUpstreamPool` & `HTTPUpstreamProvider`).

`SimulatedUpstreamPool` models realistic M/M/c queueing physics per model endpoint:
- Below `physical_knee` concurrent requests, RTT stays near `base_rtt_ms`.
- Above `physical_knee`, standing queue delay grows quadratically, allowing the
  TCP-Vegas / Gradient2 controller to discover the optimal concurrency ceiling.
- Synthesizes deterministic, context-grounded completions and realistic confidence
  scores so speculative fast->frontier cascades and RAG faithfulness checks execute
  end-to-end without requiring paid external API credentials.

`HTTPUpstreamProvider` dispatches OpenAI-compatible requests over `httpx.AsyncClient`
for live vLLM / SGLang / OpenAI / Gemini gateway deployments.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from prism_gate.routing.cascade_router import MODEL_CATALOG
from prism_gate.types import ChatCompletionRequest, ModelTier


class UpstreamResult(BaseModel):
    """Normalized upstream completion payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    status_code: int = Field(ge=200, le=599)
    content: str
    prompt_tokens: int = Field(ge=1)
    completion_tokens: int = Field(ge=0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    simulated_rtt_ms: float = Field(gt=0.0)


class UpstreamProvider(Protocol):
    """Protocol implemented by both simulated and live HTTP upstream adapters."""

    async def generate(
        self,
        *,
        model_id: str,
        request: ChatCompletionRequest,
        prompt_tokens: int,
    ) -> UpstreamResult: ...


_WORD_RE = re.compile(r"[a-zA-Z0-9$%.,-]+")


class SimulatedUpstreamPool:
    """Physics-accurate simulated LLM provider pool for deterministic benchmarks & UI."""

    def __init__(
        self,
        *,
        sleep_scale: float = 0.00015,
        physical_knee: int = 12,
        inject_overload_for_model: str | None = None,
    ) -> None:
        self.sleep_scale = max(0.0, sleep_scale)
        self.physical_knee = max(2, physical_knee)
        self.inject_overload_for_model = inject_overload_for_model
        self._active_per_model: dict[str, int] = dict.fromkeys(MODEL_CATALOG, 0)

    def set_overload_injection(self, model_id: str | None) -> None:
        """Enable or clear synthetic congestion spike on a specific upstream model."""
        self.inject_overload_for_model = model_id

    async def generate(
        self,
        *,
        model_id: str,
        request: ChatCompletionRequest,
        prompt_tokens: int,
    ) -> UpstreamResult:
        spec = MODEL_CATALOG.get(model_id, MODEL_CATALOG["gemini-2.5-flash"])
        active = self._active_per_model.get(model_id, 0) + 1
        self._active_per_model[model_id] = active

        try:
            prompt_text = request.messages[-1].content
            digest = hashlib.blake2b(f"{model_id}:{prompt_text}".encode(), digest_size=4).digest()
            jitter = (int.from_bytes(digest, "little") % 200) / 100.0  # [0.0, 1.99] ms

            # Queueing physics: flat below knee, quadratic delay above knee
            knee = 4 if self.inject_overload_for_model == model_id else self.physical_knee
            overload = max(0, active - knee)
            queue_delay_ms = (
                spec.base_rtt_ms * (overload * overload) / float(knee) if overload > 0 else 0.0
            )
            simulated_rtt_ms = spec.base_rtt_ms + jitter + queue_delay_ms

            if self.sleep_scale > 0.0:
                await asyncio.sleep(min(0.025, simulated_rtt_ms * self.sleep_scale))
            else:
                await asyncio.sleep(0)

            # Severe overload triggers realistic upstream HTTP 429 throttling
            if overload > knee:
                return UpstreamResult(
                    model_id=model_id,
                    status_code=429,
                    content="",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    confidence_score=0.0,
                    simulated_rtt_ms=simulated_rtt_ms,
                )

            content, confidence = self._synthesize_completion(
                model_id=model_id,
                tier=spec.tier,
                prompt_text=prompt_text,
                rag_context=request.rag_context,
                digest_byte=digest[0],
            )
            comp_tokens = min(
                request.max_tokens,
                max(8, int(len(_WORD_RE.findall(content)) * 1.25)),
            )
            return UpstreamResult(
                model_id=model_id,
                status_code=200,
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=comp_tokens,
                confidence_score=confidence,
                simulated_rtt_ms=round(simulated_rtt_ms, 3),
            )
        finally:
            self._active_per_model[model_id] = max(0, self._active_per_model.get(model_id, 1) - 1)

    @staticmethod
    def _synthesize_completion(
        *,
        model_id: str,
        tier: ModelTier,
        prompt_text: str,
        rag_context: tuple[str, ...],
        digest_byte: int,
    ) -> tuple[str, float]:
        lower = prompt_text.lower()

        if rag_context:
            # Ground directly in retrieved context chunks so faithfulness checks pass
            grounded_sentences: list[str] = []
            for chunk in rag_context[:3]:
                stripped = chunk.strip().rstrip(".")
                if stripped:
                    grounded_sentences.append(f"{stripped}.")
            joined = " ".join(grounded_sentences)
            if tier == ModelTier.FRONTIER:
                return joined, 0.96
            # Fast tier occasionally hedges on multi-chunk RAG -> triggers cascade escalation
            if len(rag_context) >= 2 and (
                "ambiguous" in lower or "reconcile" in lower or digest_byte < 36
            ):
                return joined, 0.64
            return joined, 0.89

        if "ambiguous" in lower or "uncertain" in lower or "escalate" in lower:
            if tier == ModelTier.FAST:
                return (
                    f"[{model_id}] Preliminary draft requires frontier verification "
                    "due to multi-constraint ambiguity.",
                    0.58,
                )
            return (
                f"[{model_id}] Verified frontier synthesis resolving multi-constraint "
                "edge cases with formal bounds.",
                0.95,
            )

        if tier == ModelTier.FRONTIER:
            return (
                f"[{model_id}] Architectural analysis complete: Little's Law concurrency bounds, "
                "signed-hash tenant cache isolation, and base-2 Jensen-Shannon drift thresholds "
                "maintain p99 tail-latency stability.",
                0.96,
            )

        return (
            f"[{model_id}] Fast-tier response: request processed within low-latency SLO budget.",
            0.91,
        )


class HTTPUpstreamProvider:
    """Production async HTTP upstream adapter speaking OpenAI `/v1/chat/completions`."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._external_client = client is not None
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def aclose(self) -> None:
        """Close owned `httpx.AsyncClient` cleanly."""
        if not self._external_client:
            await self._client.aclose()

    async def generate(
        self,
        *,
        model_id: str,
        request: ChatCompletionRequest,
        prompt_tokens: int,
    ) -> UpstreamResult:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model_id,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stream": False,
        }

        loop = asyncio.get_running_loop()
        t0 = loop.time()
        resp = await self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        elapsed_ms = max(0.1, (loop.time() - t0) * 1000.0)

        if resp.status_code != 200:
            return UpstreamResult(
                model_id=model_id,
                status_code=resp.status_code,
                content="",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                confidence_score=0.0,
                simulated_rtt_ms=round(elapsed_ms, 3),
            )

        body = resp.json()
        choices = body.get("choices") or []
        content = ""
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            content = str(msg.get("content") or "")

        usage = body.get("usage") or {}
        in_toks = int(usage.get("prompt_tokens") or prompt_tokens)
        out_toks = int(usage.get("completion_tokens") or max(1, len(content.split())))
        confidence = float(body.get("confidence_score", 0.90))

        return UpstreamResult(
            model_id=model_id,
            status_code=200,
            content=content,
            prompt_tokens=max(1, in_toks),
            completion_tokens=max(1, out_toks),
            confidence_score=min(1.0, max(0.0, confidence)),
            simulated_rtt_ms=round(elapsed_ms, 3),
        )
