"""Unit & Hypothesis property-based tests for PSI & Jensen-Shannon Divergence detectors."""

from __future__ import annotations

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from prism_gate.drift.detectors import (
    DriftPlaneAnalyzer,
    classify_jsd_severity,
    classify_psi_severity,
    jensen_shannon_divergence,
    population_stability_index,
    psi_from_distributions,
)
from prism_gate.types import DriftAnalyzeRequest, DriftSeverity


def test_jsd_and_psi_validation_errors() -> None:
    with pytest.raises(ValueError, match="empty"):
        jensen_shannon_divergence([], [])
    with pytest.raises(ValueError, match="empty"):
        jensen_shannon_divergence({}, {"a": 1.0})
    with pytest.raises(ValueError, match="equal length"):
        jensen_shannon_divergence([1.0, 2.0], [1.0])
    with pytest.raises(TypeError, match="Sequences or Mappings"):
        jensen_shannon_divergence([1.0], {"a": 1.0})
    with pytest.raises(ValueError, match="finite and >= 0"):
        jensen_shannon_divergence([-0.1, 1.0], [0.5, 0.5])
    with pytest.raises(ValueError, match="positive finite total mass"):
        jensen_shannon_divergence([0.0, 0.0], [0.5, 0.5])

    with pytest.raises(ValueError, match="equal length"):
        psi_from_distributions([0.5, 0.5], [1.0])
    with pytest.raises(ValueError, match="epsilon"):
        psi_from_distributions([0.5, 0.5], [0.5, 0.5], epsilon=0.0)
    with pytest.raises(ValueError, match="non-empty"):
        population_stability_index([], [1.0])
    with pytest.raises(ValueError, match="num_bins"):
        population_stability_index([1.0, 2.0], [1.0, 2.0], num_bins=1)
    with pytest.raises(ValueError, match="finite"):
        population_stability_index([math.nan, 1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        population_stability_index([1.0, 2.0], [math.inf, 2.0])


def test_disjoint_jsd_bounds_and_constant_baseline_psi() -> None:
    # Disjoint support -> exact 1.0 bit in base 2, exact ln(2) nats in base e
    assert jensen_shannon_divergence({"a": 10.0}, {"b": 5.0}, base=2) == pytest.approx(1.0)
    assert jensen_shannon_divergence({"a": 10.0}, {"b": 5.0}, base="e") == pytest.approx(
        math.log(2.0)
    )

    # Degenerate constant baseline array handles both identical and shifted live windows
    same_const = population_stability_index([42.0] * 30, [42.0] * 30)
    assert same_const.psi_value == 0.0
    assert same_const.severity == DriftSeverity.NO_DRIFT

    shifted_const = population_stability_index([42.0] * 30, [100.0] * 30)
    assert shifted_const.psi_value > 0.25
    assert shifted_const.severity == DriftSeverity.CRITICAL_DRIFT

    # Degenerate narrow 2-element baseline
    two_elem = population_stability_index([1.0, 1.0 + 1e-15], [1.0, 2.0], num_bins=4)
    assert two_elem.psi_value >= 0.0

    assert classify_psi_severity(0.05) == DriftSeverity.NO_DRIFT
    assert classify_psi_severity(0.15) == DriftSeverity.MODERATE_DRIFT
    assert classify_psi_severity(0.30) == DriftSeverity.CRITICAL_DRIFT
    assert classify_jsd_severity(0.02) == DriftSeverity.NO_DRIFT
    assert classify_jsd_severity(0.12) == DriftSeverity.MODERATE_DRIFT
    assert classify_jsd_severity(0.35) == DriftSeverity.CRITICAL_DRIFT


@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=2, max_value=24),
    data=st.data(),
)
def test_hypothesis_jensen_shannon_divergence_axioms(n: int, data: st.DataObject) -> None:
    weight_strat = st.floats(
        min_value=0.0,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
    )
    p = data.draw(st.lists(weight_strat, min_size=n, max_size=n))
    q = data.draw(st.lists(weight_strat, min_size=n, max_size=n))
    assume(sum(p) > 1e-9 and sum(q) > 1e-9)

    # 1. Exact identity of indiscernibles: JSD(P, P) == 0.0
    assert jensen_shannon_divergence(p, p, base=2) == 0.0
    assert jensen_shannon_divergence(p, p, base="e") == 0.0

    # 2. Exact symmetry: JSD(P, Q) == JSD(Q, P)
    jsd_pq = jensen_shannon_divergence(p, q, base=2)
    jsd_qp = jensen_shannon_divergence(q, p, base=2)
    assert jsd_pq == jsd_qp

    # 3. Strict base-2 boundedness: 0.0 <= JSD(P, Q) <= 1.0
    assert 0.0 <= jsd_pq <= 1.0

    # 4. Base-e boundedness: 0.0 <= JSD_e(P, Q) <= ln(2)
    jsd_e = jensen_shannon_divergence(p, q, base="e")
    assert 0.0 <= jsd_e <= math.log(2.0) + 1e-15


@settings(max_examples=80, deadline=None)
@given(
    baseline=st.lists(
        st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=60,
    ),
    live=st.lists(
        st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=60,
    ),
)
def test_hypothesis_psi_non_negativity_and_self_zero(
    baseline: list[float],
    live: list[float],
) -> None:
    self_res = population_stability_index(baseline, baseline)
    assert self_res.psi_value == 0.0
    assert self_res.severity == DriftSeverity.NO_DRIFT

    cross_res = population_stability_index(baseline, live)
    assert cross_res.psi_value >= 0.0


def test_drift_plane_analyzer_presets_live_window_and_topics() -> None:
    analyzer = DriftPlaneAnalyzer()
    assert analyzer.analyze_preset("normal").overall_severity == DriftSeverity.NO_DRIFT
    assert analyzer.analyze_preset("moderate").overall_severity == DriftSeverity.MODERATE_DRIFT
    assert analyzer.analyze_preset("critical").overall_severity == DriftSeverity.CRITICAL_DRIFT

    # Exercise all topic classifiers & live sliding window
    for prompt in [
        "Calculate Q3 GAAP revenue margin",
        "Write async python SQL query",
        "Summarize executive brief",
        "Check HIPAA Warfarin clinical policy",
        "General enterprise onboarding question",
    ]:
        analyzer.record_observation(
            prompt_text=prompt,
            prompt_tokens=45,
            complexity_score=0.30,
            latency_ms=24.0,
        )

    live_rep = analyzer.analyze_preset("live_window")
    assert live_rep.live_sample_size >= 125

    custom_rep = analyzer.analyze_request(DriftAnalyzeRequest())
    assert custom_rep.overall_severity in {
        DriftSeverity.NO_DRIFT,
        DriftSeverity.MODERATE_DRIFT,
        DriftSeverity.CRITICAL_DRIFT,
    }
