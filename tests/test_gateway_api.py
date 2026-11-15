"""End-to-end tests for OpenAI REST + SSE streaming gateway, Prometheus, and HTTPX upstream."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from prism_gate.admission.token_bucket import HierarchicalAdmissionController
from prism_gate.concur.vegas_limiter import VegasAdaptiveLimiter
from prism_gate.gateway.app import create_app
from prism_gate.gateway.engine import PrismGateEngine
from prism_gate.providers.upstreams import HTTPUpstreamProvider
from prism_gate.types import ChatCompletionRequest, ChatMessage, ModelTier, TenantSLOProfile


@pytest.mark.asyncio
async def test_gateway_sync_cache_sse_streaming_and_observability_endpoints(
    tmp_path: Path,
) -> None:
    # 1. Verify JSON root fallback when ui/dist is absent (mirrors CI Python job)
    missing_ui = tmp_path / "no_dist"
    app = create_app(ui_dist_dir=missing_ui)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        root_resp = await client.get("/")
        assert root_resp.status_code == 200
        assert root_resp.json()["ui_bundle"] == "not_built"

        health = await client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        models = await client.get("/v1/models")
        assert models.status_code == 200
        assert any(m["id"] == "prism-auto" for m in models.json()["data"])

        # 2. Synchronous OpenAI /v1/chat/completions + Semantic Cache Hit on paraphrase
        payload = {
            "model": "prism-auto",
            "messages": [
                {
                    "role": "user",
                    "content": "What are the standard enterprise support hours for EMEA customers?",
                }
            ],
            "tenant_id": "tenant-enterprise-01",
            "priority": "INTERACTIVE",
        }
        r1 = await client.post("/v1/chat/completions", json=payload)
        assert r1.status_code == 200
        body1 = r1.json()
        assert body1["object"] == "chat.completion"
        assert body1["prism_meta"]["cache_hit"] is False

        paraphrase_payload = {
            **payload,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Could you tell me the standard EMEA customer enterprise support hours?"
                    ),
                }
            ],
        }
        r2 = await client.post("/v1/chat/completions", json=paraphrase_payload)
        assert r2.status_code == 200
        assert r2.headers["x-prism-cache-hit"] == "true"
        assert r2.json()["prism_meta"]["actual_cost_usd"] == 0.0

        # 3. Real OpenAI Server-Sent Events (SSE) token streaming (`stream: true`)
        sse_payload = {
            "model": "prism-auto",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": "Reconcile ambiguous multi-quarter margin variance.",
                }
            ],
            "rag_context": [
                "Acme Corp reported Q3 2025 revenue of $12.4M.",
                "Operating margin reached 18.5% under GAAP.",
            ],
            "bypass_cache": True,
        }
        async with client.stream("POST", "/v1/chat/completions", json=sse_payload) as stream_resp:
            assert stream_resp.status_code == 200
            assert "text/event-stream" in stream_resp.headers["content-type"]
            raw_sse = (await stream_resp.aread()).decode("utf-8")

        frames = [f.strip() for f in raw_sse.split("\n\n") if f.strip()]
        assert frames[-1] == "data: [DONE]"
        assembled = ""
        final_meta = None
        for frame in frames[:-1]:
            assert frame.startswith("data: ")
            parsed = json.loads(frame[6:])
            assert parsed["object"] == "chat.completion.chunk"
            delta_text = parsed["choices"][0]["delta"].get("content")
            if delta_text:
                assembled += delta_text
            if parsed.get("prism_meta"):
                final_meta = parsed["prism_meta"]

        assert "Acme Corp" in assembled
        assert final_meta is not None
        assert final_meta["escalated_from_fast"] is True
        assert final_meta["faithfulness_score"] >= 0.80

        # 4. Control-plane overview, faithfulness judge, drift lab, load bench, and Prometheus
        overview = await client.get("/api/v1/overview")
        assert overview.status_code == 200
        assert overview.json()["finops"]["total_routed_requests"] >= 3

        faith = await client.post(
            "/api/v1/evals/faithfulness",
            json={
                "completion": "Warfarin maintenance therapy is prescribed at 500mg daily.",
                "context_chunks": ["Warfarin maintenance therapy is prescribed at 5mg daily."],
            },
        )
        assert faith.status_code == 200
        assert faith.json()["hallucination_detected"] is True

        drift = await client.post("/api/v1/drift/analyze", json={"preset": "critical"})
        assert drift.status_code == 200
        assert drift.json()["overall_severity"] == "CRITICAL_DRIFT"

        bench = await client.post(
            "/api/v1/benchmarks/load",
            json={"requests": 24, "concurrency": 6, "inject_congestion_spike": False},
        )
        assert bench.status_code == 200
        assert bench.json()["total_requests"] == 24

        prom = await client.get("/metrics")
        assert prom.status_code == 200
        assert "prism_gate_vegas_max_inflight" in prom.text
        assert "prism_gate_finops_cost_usd_total" in prom.text


@pytest.mark.asyncio
async def test_gateway_static_ui_serving_and_429_503_rejection_paths(tmp_path: Path) -> None:
    # Create mock built UI bundle
    dist_dir = tmp_path / "dist"
    (dist_dir / "assets").mkdir(parents=True)
    (dist_dir / "index.html").write_text("<!doctype html><title>PrismGate UI</title>")
    (dist_dir / "assets" / "app.js").write_text("console.log('ok');")

    # Configure strict 1-request burst + tiny Vegas limit to test 429 and 503 handlers
    strict_admission = HierarchicalAdmissionController(
        profiles=[
            TenantSLOProfile(
                tenant_id="tiny-tenant",
                rps_limit=0.1,
                burst_rps=1.0,
                tpm_limit=60_000.0,
                burst_tokens=10_000.0,
            )
        ]
    )
    saturated_limiter = VegasAdaptiveLimiter(
        "gemini-2.5-flash", tier=ModelTier.FAST, initial_limit=1, min_limit=1, max_limit=2
    )
    frontier_limiter = VegasAdaptiveLimiter(
        "gemini-2.5-pro", tier=ModelTier.FRONTIER, initial_limit=10, min_limit=2, max_limit=20
    )
    held_lease = saturated_limiter.try_acquire()
    assert held_lease is not None

    engine = PrismGateEngine(
        admission=strict_admission,
        limiters={
            "gemini-2.5-flash": saturated_limiter,
            "gemini-2.5-pro": frontier_limiter,
        },
    )
    app = create_app(engine=engine, ui_dist_dir=dist_dir)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        html_resp = await client.get("/")
        assert html_resp.status_code == 200
        assert "PrismGate UI" in html_resp.text

        # First call passes admission for tiny-tenant, then hits Vegas saturation (503)
        r_shed = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "Hello"}],
                "tenant_id": "tiny-tenant",
            },
        )
        assert r_shed.status_code == 503
        assert r_shed.json()["error"]["code"] == "VEGAS_CONCURRENCY_SHED"

        # Second immediate call from tiny-tenant hits 429 RPS limit with Retry-After header
        r_429 = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "Hello again"}],
                "tenant_id": "tiny-tenant",
            },
        )
        assert r_429.status_code == 429
        assert "retry-after" in r_429.headers


@pytest.mark.asyncio
async def test_http_upstream_provider_success_and_error_handling() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer sk-test"
        body = json.loads(request.content.decode("utf-8"))
        if body["model"] == "fail-model":
            return httpx.Response(status_code=503, json={"error": "overloaded"})
        return httpx.Response(
            status_code=200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "Live HTTP reply."}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
                "confidence_score": 0.93,
            },
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = HTTPUpstreamProvider(
        "https://upstream.example.com",
        api_key="sk-test",
        client=mock_client,
    )

    req_ok = ChatCompletionRequest(
        model="gemini-2.5-flash",
        messages=(ChatMessage(role="user", content="Ping"),),
    )
    res_ok = await provider.generate(model_id="gemini-2.5-flash", request=req_ok, prompt_tokens=8)
    assert res_ok.status_code == 200
    assert res_ok.content == "Live HTTP reply."
    assert res_ok.prompt_tokens == 12

    res_err = await provider.generate(model_id="fail-model", request=req_ok, prompt_tokens=8)
    assert res_err.status_code == 503

    # Also test engine handling of non-200 upstream status & sibling failover
    def always_503(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=503, json={"error": "upstream down"})

    err_client = httpx.AsyncClient(transport=httpx.MockTransport(always_503))
    err_provider = HTTPUpstreamProvider("https://upstream.example.com", client=err_client)
    err_engine = PrismGateEngine(upstream=err_provider)
    err_app = create_app(engine=err_engine, ui_dist_dir=Path("/nonexistent"))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=err_app), base_url="http://testserver"
    ) as c_err:
        up_503 = await c_err.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "Trigger upstream 503"}],
                "bypass_cache": True,
            },
        )
        assert up_503.status_code == 503
        assert up_503.json()["error"]["code"] == "UPSTREAM_HTTP_503"

    await err_client.aclose()
    await provider.aclose()
    await mock_client.aclose()

    # Owned client lifecycle test
    owned = HTTPUpstreamProvider("https://upstream.example.com")
    await owned.aclose()


@pytest.mark.asyncio
async def test_sibling_vegas_failover_cached_rag_and_upstream_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Saturated primary fast model fails over seamlessly to sibling fast model
    lim_flash = VegasAdaptiveLimiter(
        "gemini-2.5-flash", tier=ModelTier.FAST, initial_limit=1, min_limit=1, max_limit=2
    )
    held = lim_flash.try_acquire()
    assert held is not None
    lim_vllm = VegasAdaptiveLimiter(
        "vllm-qwen-32b", tier=ModelTier.FAST, initial_limit=6, min_limit=2, max_limit=12
    )
    lim_pro = VegasAdaptiveLimiter(
        "gemini-2.5-pro", tier=ModelTier.FRONTIER, initial_limit=6, min_limit=2, max_limit=12
    )

    eng = PrismGateEngine(
        limiters={
            "gemini-2.5-flash": lim_flash,
            "vllm-qwen-32b": lim_vllm,
            "gemini-2.5-pro": lim_pro,
        }
    )
    req = ChatCompletionRequest(
        model="gemini-2.5-flash",
        messages=(ChatMessage(role="user", content="Test sibling failover."),),
        rag_context=("Test sibling failover grounded chunk.",),
    )
    resp1 = await eng.complete(req)
    assert resp1.model == "vllm-qwen-32b"
    # Second identical RAG request hits semantic cache with RAG faithfulness re-evaluated
    resp2 = await eng.complete(req)
    assert resp2.prism_meta.cache_hit is True
    assert resp2.faithfulness_report is not None

    # Upstream transport exception releases Vegas lease cleanly without leaking inflight count
    class BoomUpstream:
        async def generate(
            self,
            *,
            model_id: str,
            request: ChatCompletionRequest,
            prompt_tokens: int,
        ) -> object:
            raise RuntimeError("socket reset")

    boom_lim = VegasAdaptiveLimiter("gemini-2.5-flash", tier=ModelTier.FAST)
    boom_eng = PrismGateEngine(
        upstream=BoomUpstream(),  # type: ignore[arg-type]
        limiters={"gemini-2.5-flash": boom_lim},
    )
    with pytest.raises(RuntimeError, match="socket reset"):
        await boom_eng.complete(
            ChatCompletionRequest(
                model="gemini-2.5-flash",
                messages=(ChatMessage(role="user", content="boom"),),
                bypass_cache=True,
            )
        )
    assert boom_lim.inflight == 0

    # Verify env overrides in create_app
    monkeypatch.setenv("PRISM_UI_DIST_DIR", str(tmp_path))
    monkeypatch.setenv("PRISM_UPSTREAM_BASE_URL", "https://api.example.com")
    env_app = create_app()
    assert env_app.state.engine is not None
