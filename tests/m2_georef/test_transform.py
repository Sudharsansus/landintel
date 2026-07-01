"""Unit tests for the Umeyama similarity transform and cadastral adjustment.

These run on synthetic point sets where the answer is known exactly, so they
validate the transform math independently of matching, OCR, or real fixtures.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from landintel.pipeline.m2_georef.transform import cadastral_adjust, umeyama


def _R(theta_deg: float) -> np.ndarray:
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


def test_umeyama_recovers_known_transform():
    """Umeyama must recover scale, rotation, and translation exactly."""
    rng = np.random.default_rng(0)
    src = rng.uniform(-50, 50, size=(8, 2))

    s_true, theta_true, t_true = 1.7, 33.0, np.array([1000.0, -2000.0])
    dst = s_true * (src @ _R(theta_true).T) + t_true

    R, s, t, residuals = umeyama(src, dst)

    assert s == pytest.approx(s_true, rel=1e-6)
    angle = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    assert angle == pytest.approx(theta_true, abs=1e-6)
    assert t == pytest.approx(t_true, abs=1e-4)
    assert residuals.max() < 1e-6
    # Proper rotation, no reflection.
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


def test_umeyama_rejects_reflection():
    """A reflected point set must still yield a proper rotation (det = +1)."""
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    dst = src.copy()
    dst[:, 0] *= -1.0  # mirror across the y-axis
    R, s, t, _ = umeyama(src, dst)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


def test_umeyama_identity():
    """src == dst yields identity scale/rotation and zero translation."""
    src = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [0.0, 8.0]])
    R, s, t, residuals = umeyama(src, src)
    assert s == pytest.approx(1.0, abs=1e-9)
    assert t == pytest.approx(np.zeros(2), abs=1e-9)
    assert residuals.max() < 1e-9


def test_cadastral_adjust_holds_field_positions():
    """With a high field weight, matched stones land on surveyor positions."""
    m1 = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 35.0], [0.0, 22.0]])
    s_true, theta_true, t_true = 1.0, 20.0, np.array([783000.0, 1241000.0])
    surveyor = s_true * (m1 @ _R(theta_true).T) + t_true

    matched_pairs = [(0, 0), (1, 1), (2, 2), (3, 3)]
    # Outer edges with their FMB-measured lengths.
    edge_pairs = []
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        edge_pairs.append((a, b, float(np.linalg.norm(m1[b] - m1[a]))))

    R, s, t, _ = umeyama(m1, surveyor)
    adjusted = cadastral_adjust(
        m1_positions=m1,
        surveyor_positions=surveyor,
        matched_pairs=matched_pairs,
        edge_pairs=edge_pairs,
        field_weight=1000.0,
        dist_weight=1.0,
        umeyama_result=(R, s, t),
    )

    # Field residual: adjusted matched stones essentially coincide with surveyor.
    for m1_idx, surv_idx in matched_pairs:
        assert np.linalg.norm(adjusted[m1_idx] - surveyor[surv_idx]) < 0.05

    # Edge lengths preserved (consistent FMB + field data here).
    for a, b, target in edge_pairs:
        assert np.linalg.norm(adjusted[b] - adjusted[a]) == pytest.approx(target, abs=0.05)


def test_cadastral_adjust_distributes_discrepancy():
    """When FMB edges disagree with field by a constant, error spreads, not spikes."""
    m1 = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 35.0], [0.0, 22.0]])
    surveyor = m1 + np.array([700000.0, 1240000.0])  # pure translation, scale 1

    matched_pairs = [(0, 0), (1, 1), (2, 2), (3, 3)]
    # FMB edge lengths inflated by 0.4 m each -> conflicts with field positions.
    edge_pairs = []
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        edge_pairs.append((a, b, float(np.linalg.norm(m1[b] - m1[a])) + 0.4))

    R, s, t, _ = umeyama(m1, surveyor)
    adjusted = cadastral_adjust(
        m1_positions=m1, surveyor_positions=surveyor,
        matched_pairs=matched_pairs, edge_pairs=edge_pairs,
        field_weight=1000.0, dist_weight=1.0,
        umeyama_result=(R, s, t),
    )
    # Field weight dominates: stones stay within a few cm of surveyor positions.
    for m1_idx, surv_idx in matched_pairs:
        assert np.linalg.norm(adjusted[m1_idx] - surveyor[surv_idx]) < 0.1


def test_umeyama_degenerate_input_returns_clean_failure():
    """All-equal source points (or a single point, n<2) must NOT emit a RuntimeWarning or
    propagate NaN. The degenerate result (s=0, t=0, residuals=+inf) lets every downstream
    caller reject with a clean 'no fit' via the existing 0.5 < s < 2.0 / finite-residual gates."""
    import warnings

    # 1) all-equal source points -- zero variance, previously RuntimeWarning + NaN
    src = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
    dst = np.array([[100.0, 100.0], [110.0, 100.0], [110.0, 110.0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        R, s, t, r = umeyama(src, dst)
    assert not caught, f"RuntimeWarning emitted: {[str(x.message) for x in caught]}"
    assert s == 0.0
    assert np.all(np.isfinite(t))
    assert np.all(np.isinf(r))

    # 2) n < 2 -- no defined transform; returns clean instead of a degenerate SVD
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        R, s, t, r = umeyama(np.array([[1.0, 1.0]]), np.array([[2.0, 2.0]]))
    assert not caught
    assert s == 0.0 and np.all(np.isfinite(t)) and np.all(np.isinf(r))

    # 3) Real input still works (regression check)
    src = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    dst = np.array([[100.0, 100.0], [110.0, 100.0], [110.0, 110.0], [100.0, 110.0]])
    R, s, t, r = umeyama(src, dst)
    assert s == pytest.approx(1.0, abs=1e-9)
    assert r.max() < 1e-9
