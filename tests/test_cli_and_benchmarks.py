"""Tests for the `prism` Typer CLI and `LoadBenchmarkHarness`."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from prism_gate import __version__
from prism_gate.cli import app

runner = CliRunner()


def test_cli_version_demo_drift_check_and_benchmark(tmp_path: Path) -> None:
    ver = runner.invoke(app, ["--version"])
    assert ver.exit_code == 0
    assert __version__ in ver.stdout

    drift = runner.invoke(app, ["drift-check"])
    assert drift.exit_code == 0
    assert "PASS" in drift.stdout

    demo = runner.invoke(app, ["demo"])
    assert demo.exit_code == 0
    assert "CACHE HIT" in demo.stdout
    assert "CONTRADICTED" in demo.stdout

    out_file = tmp_path / "bench.json"
    bench = runner.invoke(
        app,
        ["benchmark", "--requests", "40", "--concurrency", "8", "--json-out", str(out_file)],
    )
    assert bench.exit_code == 0
    assert out_file.is_file()
    parsed = json.loads(out_file.read_text())
    assert parsed["total_requests"] == 40
    assert parsed["finops"]["savings_pct"] > 0.0

    # High-concurrency spike benchmark exercising BATCH_BACKGROUND priority shedding
    burst = runner.invoke(
        app,
        ["benchmark", "--requests", "120", "--concurrency", "42", "--congestion-spike"],
    )
    assert burst.exit_code == 0


def test_cli_serve_invokes_uvicorn() -> None:
    with patch("prism_gate.cli.uvicorn.run") as mock_run:
        res = runner.invoke(app, ["serve", "--host", "0.0.0.0", "--port", "9090"])
        assert res.exit_code == 0
        mock_run.assert_called_once_with(
            "prism_gate.gateway.app:app",
            host="0.0.0.0",
            port=9090,
            log_level="info",
        )
