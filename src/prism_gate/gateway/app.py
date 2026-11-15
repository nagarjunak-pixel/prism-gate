"""OpenAI-Compatible Async REST + Server-Sent Events (SSE) Streaming Gateway.

Exposes:
- `POST /v1/chat/completions` (JSON completion & `text/event-stream` SSE streaming)
- `GET  /v1/models` (OpenAI model discovery)
- `GET  /api/v1/overview` (Control-plane telemetry, Vegas metrics, FinOps & Drift)
- `POST /api/v1/evals/faithfulness` (Online atomic claim-faithfulness judge)
- `POST /api/v1/drift/analyze` (PSI + Jensen-Shannon distribution shift lab)
- `POST /api/v1/benchmarks/load` (In-process mixed-priority load test harness)
- `GET  /metrics` (Prometheus text exposition v0.0.4)
- `GET  /` (Serves built `ui/dist/index.html` when present, clean JSON fallback otherwise)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from prism_gate import __version__
from prism_gate.benchmarks.load_harness import LoadBenchmarkHarness
from prism_gate.gateway.engine import GatewayRejectedError, PrismGateEngine
from prism_gate.providers.upstreams import HTTPUpstreamProvider, UpstreamProvider
from prism_gate.routing.cascade_router import MODEL_CATALOG
from prism_gate.types import (
    BenchmarkRequest,
    BenchmarkSummary,
    ChatCompletionRequest,
    DriftAnalyzeRequest,
    DriftReport,
    FaithfulnessReport,
    FaithfulnessRequest,
    GatewayOverview,
)


def _default_ui_dist() -> Path:
    env_override = os.environ.get("PRISM_UI_DIST_DIR")
    if env_override:
        return Path(env_override)
    # src/prism_gate/gateway/app.py -> repo_root/ui/dist
    return Path(__file__).resolve().parents[3] / "ui" / "dist"


def _build_default_engine() -> PrismGateEngine:
    upstream_url = os.environ.get("PRISM_UPSTREAM_BASE_URL", "").strip()
    upstream: UpstreamProvider | None = None
    if upstream_url:
        upstream = HTTPUpstreamProvider(
            upstream_url,
            api_key=os.environ.get("PRISM_UPSTREAM_API_KEY", ""),
        )
    return PrismGateEngine(upstream=upstream)


def create_app(
    *,
    engine: PrismGateEngine | None = None,
    ui_dist_dir: Path | None = None,
) -> FastAPI:
    """Construct configured FastAPI gateway instance."""
    gate_engine = engine or _build_default_engine()
    resolved_ui = ui_dist_dir if ui_dist_dir is not None else _default_ui_dist()

    application = FastAPI(
        title="PrismGate — Multi-Tenant LLM Inference Gateway",
        version=__version__,
        description=(
            "High-throughput OpenAI-compatible LLM inference gateway with TCP-Vegas "
            "adaptive concurrency, complexity cascade routing, online claim-faithfulness "
            "verification, and PSI/Jensen-Shannon drift detection."
        ),
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.state.engine = gate_engine

    @application.exception_handler(GatewayRejectedError)
    async def _handle_gateway_rejected(
        _req: Any,
        exc: GatewayRejectedError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            headers={"Retry-After": f"{exc.retry_after_seconds:.2f}"},
            content={
                "error": {
                    "message": exc.message,
                    "type": "prism_gate_admission_error",
                    "code": exc.reason,
                }
            },
        )

    @application.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "service": "prism-gate", "version": __version__}

    @application.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        data = [
            {
                "id": "prism-auto",
                "object": "model",
                "owned_by": "prism-gate",
                "tier": "AUTO_CASCADE",
            },
            *(
                {
                    "id": spec.model_id,
                    "object": "model",
                    "owned_by": "prism-gate",
                    "tier": spec.tier.value,
                    "cost_per_1k_input_usd": spec.cost_per_1k_input_usd,
                    "cost_per_1k_output_usd": spec.cost_per_1k_output_usd,
                }
                for spec in MODEL_CATALOG.values()
            ),
        ]
        return {"object": "list", "data": data}

    @application.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: ChatCompletionRequest) -> Response:
        if request.stream:
            initial_resp, sse_iterator = await gate_engine.prepare_sse_stream(request)
            return StreamingResponse(
                sse_iterator,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Prism-Selected-Model": initial_resp.prism_meta.selected_model,
                    "X-Prism-Selected-Tier": initial_resp.prism_meta.selected_tier.value,
                    "X-Prism-Cache-Hit": str(initial_resp.prism_meta.cache_hit).lower(),
                },
            )

        sync_resp = await gate_engine.complete(request)
        return JSONResponse(
            status_code=200,
            headers={
                "X-Prism-Selected-Model": sync_resp.prism_meta.selected_model,
                "X-Prism-Selected-Tier": sync_resp.prism_meta.selected_tier.value,
                "X-Prism-Cache-Hit": str(sync_resp.prism_meta.cache_hit).lower(),
            },
            content=sync_resp.model_dump(mode="json"),
        )

    @application.get("/api/v1/overview", response_model=GatewayOverview)
    async def api_overview() -> GatewayOverview:
        return gate_engine.overview()

    @application.post("/api/v1/evals/faithfulness", response_model=FaithfulnessReport)
    async def api_eval_faithfulness(payload: FaithfulnessRequest) -> FaithfulnessReport:
        return gate_engine.faithfulness_judge.evaluate(
            completion=payload.completion,
            context_chunks=payload.context_chunks,
        )

    @application.post("/api/v1/drift/analyze", response_model=DriftReport)
    async def api_drift_analyze(payload: DriftAnalyzeRequest) -> DriftReport:
        return gate_engine.drift_analyzer.analyze_request(payload)

    @application.post("/api/v1/benchmarks/load", response_model=BenchmarkSummary)
    async def api_run_benchmark(payload: BenchmarkRequest) -> BenchmarkSummary:
        harness = LoadBenchmarkHarness(engine=gate_engine)
        return await harness.run(
            requests=payload.requests,
            concurrency=payload.concurrency,
            inject_congestion_spike=payload.inject_congestion_spike,
        )

    @application.get("/metrics")
    async def prometheus_metrics() -> PlainTextResponse:
        return PlainTextResponse(
            content=gate_engine.render_prometheus_metrics(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    index_file = resolved_ui / "index.html"
    assets_dir = resolved_ui / "assets"
    if index_file.is_file() and assets_dir.is_dir():
        application.mount(
            "/assets",
            StaticFiles(directory=str(assets_dir)),
            name="ui-assets",
        )

    @application.get("/", response_model=None)
    async def root() -> Response:
        if index_file.is_file():
            return FileResponse(str(index_file))
        return JSONResponse(
            status_code=200,
            content={
                "service": "prism-gate",
                "version": __version__,
                "status": "healthy",
                "ui_bundle": "not_built",
                "docs_url": "/docs",
                "openapi_url": "/openapi.json",
            },
        )

    return application


app = create_app()
