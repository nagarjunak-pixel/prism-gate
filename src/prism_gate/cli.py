"""`prism` Command-Line Interface: Demo, Load Benchmark, Drift Verification, and Server."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from prism_gate import __version__
from prism_gate.benchmarks.load_harness import LoadBenchmarkHarness
from prism_gate.concur.vegas_limiter import VegasAdaptiveLimiter
from prism_gate.drift.detectors import (
    DriftPlaneAnalyzer,
    jensen_shannon_divergence,
    psi_from_distributions,
)
from prism_gate.evals.faithfulness import ClaimFaithfulnessJudge
from prism_gate.gateway.engine import PrismGateEngine
from prism_gate.types import (
    ChatCompletionRequest,
    ChatMessage,
    DriftSeverity,
    ModelTier,
    PriorityTier,
)

app = typer.Typer(
    name="prism",
    help="PrismGate: Multi-Tenant LLM Inference Gateway, Adaptive Concurrency & Drift Plane.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"prism-gate v{__version__}")
        raise typer.Exit(code=0)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show PrismGate version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """PrismGate CLI control plane."""


@app.command("demo")
def demo_cmd() -> None:
    """Run an interactive end-to-end showcase of routing, Vegas control, faithfulness, and drift."""
    asyncio.run(_run_demo())


async def _run_demo() -> None:
    console.print(
        Panel.fit(
            f"[bold cyan]PrismGate v{__version__}[/bold cyan] — Control-Plane Showcase\n"
            "[dim]Cascade Routing • Semantic Cache • TCP-Vegas • Claim Judge • PSI/JSD[/dim]",
            border_style="cyan",
        )
    )

    engine = PrismGateEngine()
    q_emea = "What are the standard enterprise support hours for EMEA customers?"
    q_emea_paraphrase = "Could you tell me the standard EMEA customer enterprise support hours?"
    q_ambig = "Resolve this ambiguous multi-constraint SLA reconciliation edge case."
    q_proof = (
        "Derive step by step the Little's Law queue stability theorem, prove "
        "invariants, and implement an async Python class with strict JSON schema."
    )
    scenarios = [
        (
            "Simple Lookup (Fast Tier)",
            ChatCompletionRequest(
                messages=(ChatMessage(role="user", content=q_emea),),
                tenant_id="tenant-enterprise-01",
                priority=PriorityTier.INTERACTIVE,
            ),
        ),
        (
            "Paraphrased Repeat (Semantic Cache Hit)",
            ChatCompletionRequest(
                messages=(ChatMessage(role="user", content=q_emea_paraphrase),),
                tenant_id="tenant-enterprise-01",
                priority=PriorityTier.INTERACTIVE,
            ),
        ),
        (
            "Cross-Tenant Isolation Proof (Cache Miss)",
            ChatCompletionRequest(
                messages=(ChatMessage(role="user", content=q_emea),),
                tenant_id="tenant-saas-prod",
                priority=PriorityTier.INTERACTIVE,
            ),
        ),
        (
            "Speculative Cascade Escalation (Fast -> Frontier)",
            ChatCompletionRequest(
                messages=(ChatMessage(role="user", content=q_ambig),),
                tenant_id="tenant-enterprise-01",
                priority=PriorityTier.CRITICAL,
            ),
        ),
        (
            "High-Complexity Proof & Code (Direct Frontier)",
            ChatCompletionRequest(
                messages=(ChatMessage(role="user", content=q_proof),),
                tenant_id="tenant-enterprise-01",
                priority=PriorityTier.CRITICAL,
            ),
        ),
    ]

    route_table = Table(title="1. Speculative Cascade Routing & Tenant-Isolated Cache")
    route_table.add_column("Scenario", style="bold")
    route_table.add_column("Tenant")
    route_table.add_column("Complexity", justify="right")
    route_table.add_column("Selected Model", style="cyan")
    route_table.add_column("Outcome")
    route_table.add_column("Cost Saved", justify="right", style="green")

    for label, req in scenarios:
        resp = await engine.complete(req)
        meta = resp.prism_meta
        outcome = (
            "CACHE HIT"
            if meta.cache_hit
            else ("ESCALATED" if meta.escalated_from_fast else meta.selected_tier.value)
        )
        route_table.add_row(
            label,
            meta.tenant_id,
            f"{meta.complexity_score:.2f}",
            meta.selected_model,
            outcome,
            f"${meta.cost_saved_usd:.6f}",
        )
    console.print(route_table)

    # 2. Vegas adaptive concurrency walkthrough
    limiter = VegasAdaptiveLimiter(
        "gemini-2.5-flash",
        tier=ModelTier.FAST,
        initial_limit=10,
        min_limit=2,
        max_limit=32,
    )
    vegas_table = Table(title="2. TCP-Vegas / Gradient2 Concurrency Adaptation")
    vegas_table.add_column("Phase")
    vegas_table.add_column("RTT Sample", justify="right")
    vegas_table.add_column("Inflight", justify="right")
    vegas_table.add_column("Gradient", justify="right")
    vegas_table.add_column("Queue Est.", justify="right")
    vegas_table.add_column("Max Inflight", justify="right", style="bold yellow")

    for phase, rtt, inf, code in [
        ("Baseline Warmup", 18.0, 8, 200),
        ("Healthy Low-Queue Growth #1", 18.4, 9, 200),
        ("Healthy Low-Queue Growth #2", 18.6, 10, 200),
        ("Standing Queue Buildup", 52.0, 12, 200),
        ("Upstream HTTP 429 Backoff", 65.0, 10, 429),
    ]:
        new_lim = limiter.record_sample(rtt_sample_ms=rtt, inflight_sample=inf, status_code=code)
        vegas_table.add_row(
            phase,
            f"{rtt:.1f} ms",
            str(inf),
            f"{limiter.gradient:.3f}",
            f"{limiter.queue_estimate:.2f}",
            str(new_lim),
        )
    console.print(vegas_table)

    # 3. Online Claim-Faithfulness Hallucination Judge
    judge = ClaimFaithfulnessJudge()
    rag_chunks = [
        "Acme Corp reported Q3 2025 revenue of $12.4M with operating margin of 18.5%.",
        "Warfarin maintenance therapy is prescribed at 5mg daily oral dosage.",
    ]
    hallucinated = (
        "Acme Corp reported Q3 2025 revenue of $12.4M. "
        "Warfarin maintenance therapy is prescribed at 500mg daily oral dosage."
    )
    f_report = judge.evaluate(completion=hallucinated, context_chunks=rag_chunks)
    faith_table = Table(
        title=(
            f"3. Online Claim-Faithfulness Judge "
            f"(Score={f_report.faithfulness_score:.2f}, "
            f"Hallucination={f_report.hallucination_detected})"
        )
    )
    faith_table.add_column("Atomic Claim")
    faith_table.add_column("Verdict", style="bold")
    faith_table.add_column("Flagged Numbers", style="red")
    faith_table.add_column("Reason")
    for cv in f_report.claim_verifications:
        faith_table.add_row(
            cv.claim_text,
            cv.verdict.value,
            ", ".join(cv.unmatched_numbers) or "-",
            cv.reason,
        )
    console.print(faith_table)


@app.command("benchmark")
def benchmark_cmd(
    requests: Annotated[
        int, typer.Option("--requests", "-n", min=10, max=5000, help="Total requests to dispatch.")
    ] = 300,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", "-c", min=1, max=256, help="Concurrent worker coroutines."),
    ] = 32,
    congestion_spike: Annotated[
        bool,
        typer.Option(
            "--congestion-spike/--no-congestion-spike",
            help="Inject mid-run upstream congestion burst to exercise Vegas & priority shedding.",
        ),
    ] = True,
    json_out: Annotated[
        Path | None,
        typer.Option("--json-out", help="Optional path to write BenchmarkSummary JSON."),
    ] = None,
) -> None:
    """Execute concurrent multi-tenant load benchmark measuring p50/p95/p99 & FinOps savings."""
    summary = asyncio.run(
        LoadBenchmarkHarness().run(
            requests=requests,
            concurrency=concurrency,
            inject_congestion_spike=congestion_spike,
        )
    )

    lat_table = Table(
        title=(
            f"PrismGate Benchmark Summary ({summary.total_requests} reqs @ "
            f"c={summary.concurrency}, {summary.throughput_rps:.1f} req/s)"
        )
    )
    lat_table.add_column("Metric", style="bold")
    lat_table.add_column("Value", justify="right", style="cyan")
    lat_table.add_row(
        "Succeeded / Shed-or-Limited",
        f"{summary.succeeded_requests} / {summary.shed_or_rate_limited}",
    )
    lat_table.add_row(
        "CRITICAL Tier Preserved", f"{summary.critical_admitted} / {summary.critical_total}"
    )
    lat_table.add_row("BATCH_BACKGROUND Shed", str(summary.batch_shed))
    lat_table.add_row(
        "Gateway Control Overhead (p50 / p95 / p99)",
        (
            f"{summary.gateway_overhead_p50_ms:.3f} / "
            f"{summary.gateway_overhead_p95_ms:.3f} / "
            f"{summary.gateway_overhead_p99_ms:.3f} ms"
        ),
    )
    lat_table.add_row(
        "End-to-End Latency (p50 / p95 / p99)",
        (
            f"{summary.e2e_latency_p50_ms:.2f} / "
            f"{summary.e2e_latency_p95_ms:.2f} / "
            f"{summary.e2e_latency_p99_ms:.2f} ms"
        ),
    )
    lat_table.add_row(
        "Actual Cascade + Cache Spend", f"${summary.finops.actual_total_cost_usd:.6f}"
    )
    lat_table.add_row(
        "Static Frontier Baseline Spend", f"${summary.finops.counterfactual_frontier_cost_usd:.6f}"
    )
    lat_table.add_row(
        "FinOps Net Cost Reduction",
        f"${summary.finops.net_savings_usd:.6f} ({summary.finops.savings_pct:.2f}%)",
    )
    lat_table.add_row("Semantic Cache Hit Rate", f"{summary.finops.cache_hit_rate * 100.0:.1f}%")
    console.print(lat_table)

    ep_table = Table(title="Per-Provider TCP-Vegas Concurrency State")
    ep_table.add_column("Provider", style="bold")
    ep_table.add_column("Tier")
    ep_table.add_column("Max Inflight", justify="right")
    ep_table.add_column("RTT Min (ms)", justify="right")
    ep_table.add_column("RTT EWMA (ms)", justify="right")
    ep_table.add_column("Requests", justify="right")
    ep_table.add_column("Shed", justify="right")
    for ep in summary.endpoints:
        ep_table.add_row(
            ep.provider_id,
            ep.tier.value,
            str(ep.max_inflight),
            f"{ep.rtt_min_ms:.2f}",
            f"{ep.rtt_ewma_ms:.2f}",
            str(ep.total_requests),
            str(ep.total_shed),
        )
    console.print(ep_table)

    if json_out is not None:
        json_out.write_text(json.dumps(summary.model_dump(mode="json"), indent=2) + "\n")
        console.print(f"[green]Saved structured benchmark metrics to {json_out}[/green]")


@app.command("drift-check")
def drift_check_cmd() -> None:
    """Verify PSI & Jensen-Shannon invariants and evaluate reference traffic regimes."""
    dist_a = {"alpha": 0.5, "beta": 0.3, "gamma": 0.2}
    dist_b = {"alpha": 0.1, "beta": 0.4, "gamma": 0.5}
    disjoint_x = {"left": 1.0}
    disjoint_y = {"right": 1.0}

    jsd_self = jensen_shannon_divergence(dist_a, dist_a)
    jsd_ab = jensen_shannon_divergence(dist_a, dist_b)
    jsd_ba = jensen_shannon_divergence(dist_b, dist_a)
    jsd_disjoint = jensen_shannon_divergence(disjoint_x, disjoint_y)
    psi_self = psi_from_distributions([0.2, 0.5, 0.3], [0.2, 0.5, 0.3])

    if jsd_self != 0.0 or jsd_ab != jsd_ba or abs(jsd_disjoint - 1.0) > 1e-12 or psi_self != 0.0:
        console.print("[bold red]Mathematical drift invariant check FAILED![/bold red]")
        raise typer.Exit(code=1)

    analyzer = DriftPlaneAnalyzer()
    regimes: list[tuple[str, DriftSeverity]] = [
        ("normal", DriftSeverity.NO_DRIFT),
        ("moderate", DriftSeverity.MODERATE_DRIFT),
        ("critical", DriftSeverity.CRITICAL_DRIFT),
    ]

    table = Table(title="PrismGate Drift Plane Verification (PSI + Base-2 JSD)")
    table.add_column("Regime", style="bold")
    table.add_column("Expected")
    table.add_column("Observed", style="cyan")
    table.add_column("Max PSI", justify="right")
    table.add_column("Token JSD (bits)", justify="right")
    table.add_column("Topic JSD (bits)", justify="right")
    table.add_column("Status", style="green")

    for preset_name, expected_sev in regimes:
        rep = analyzer.analyze_preset(preset_name)  # type: ignore[arg-type]
        passed = rep.overall_severity == expected_sev
        table.add_row(
            preset_name,
            expected_sev.value,
            rep.overall_severity.value,
            f"{rep.max_psi:.4f}",
            f"{rep.token_jsd:.4f}",
            f"{rep.topic_jsd:.4f}",
            "PASS" if passed else "FAIL",
        )
        if not passed:
            console.print(table)
            raise typer.Exit(code=1)

    console.print(table)
    console.print(
        "[bold green]All PSI & Jensen-Shannon Divergence invariants "
        "and alert thresholds verified.[/bold green]"
    )


@app.command("serve")
def serve_cmd(
    host: Annotated[str, typer.Option("--host", help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", min=1, max=65535, help="Bind port.")] = 8000,
    log_level: Annotated[str, typer.Option("--log-level", help="Uvicorn log level.")] = "info",
) -> None:
    """Start the OpenAI-compatible PrismGate HTTP + SSE gateway server."""
    uvicorn.run("prism_gate.gateway.app:app", host=host, port=port, log_level=log_level)
