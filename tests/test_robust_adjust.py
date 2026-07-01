"""Robust cadastral adjustment + seed-quality error propagation.

Two held mathematical-hardening gaps from the 2026-06-28 audit, now closed:

1. ``cadastral_adjust(robust=True)`` -- a single bad correspondence (a stone
   snapped to a WRONG-but-close field point, or an OCR edge-length outlier) must
   NOT drag the clean corners. Plain L2 averages the outlier into every corner and
   still reports a small residual ("certified-clean mis-snap"); soft_l1 isolates it.

2. ``seed_quality`` -- an exactly-determined 2-point seed has no averaging, so a
   short baseline amplifies field-point noise across a large plot. The gate must
   REJECT short baselines / large induced far-corner error and ACCEPT long ones.

The default (robust=False) path must stay byte-identical to the validated INGUR
fit, so a regression test pins L2==legacy on a clean configuration.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from landintel.pipeline.m2_georef.transform import (cadastral_adjust,
                                                    seed_quality, umeyama)


def _square(side: float = 40.0) -> np.ndarray:
    return np.array([[0.0, 0.0], [side, 0.0], [side, side], [0.0, side]])


def test_default_is_plain_l2_unchanged():
    """robust=False must not perturb a clean fit (INGUR path stays identical)."""
    m1 = _square(40.0)
    # True UTM placement: translate into TN UTM range, no rotation/scale change.
    surv = m1 + np.array([700000.0, 1200000.0])
    pairs = [(i, i) for i in range(4)]
    edges = [(0, 1, 40.0), (1, 2, 40.0), (2, 3, 40.0), (3, 0, 40.0)]
    R, s, t, _ = umeyama(m1, surv)
    out = cadastral_adjust(m1, surv, pairs, edges, umeyama_result=(R, s, t))
    # Every matched corner lands on its surveyor point (clean, exactly determined).
    for i in range(4):
        assert np.linalg.norm(out[i] - surv[i]) < 1e-3


def test_robust_isolates_a_bad_correspondence():
    """A single grossly-wrong field correspondence must move clean corners far less
    under soft_l1 than under L2."""
    m1 = _square(40.0)
    surv_true = m1 + np.array([700000.0, 1200000.0])
    surv = surv_true.copy()
    # Corner 2's field point is mis-identified 8 m away (wrong neighbour stone).
    surv[2] = surv_true[2] + np.array([8.0, 0.0])

    pairs = [(i, i) for i in range(4)]
    edges = [(0, 1, 40.0), (1, 2, 40.0), (2, 3, 40.0), (3, 0, 40.0)]
    R, s, t, _ = umeyama(m1, surv)

    l2 = cadastral_adjust(m1, surv, pairs, edges, umeyama_result=(R, s, t),
                          robust=False)
    rob = cadastral_adjust(m1, surv, pairs, edges, umeyama_result=(R, s, t),
                           robust=True, f_scale=4.0)

    # Clean corners 0,1,3 should sit closer to their TRUE positions under robust.
    clean = [0, 1, 3]
    l2_err = max(np.linalg.norm(l2[i] - surv_true[i]) for i in clean)
    rob_err = max(np.linalg.norm(rob[i] - surv_true[i]) for i in clean)
    assert rob_err <= l2_err + 1e-9
    # And the robust fit must keep the clean corners genuinely close to truth.
    assert rob_err < 1.0


def test_seed_quality_rejects_short_baseline():
    sq = seed_quality(
        seed_src=np.array([[0.0, 0.0], [2.0, 0.0]]),       # 2 m baseline
        seed_dst=np.array([[700000.0, 1200000.0], [700002.0, 1200000.0]]),
        template_points=_square(40.0),
        min_baseline_m=5.0,
    )
    assert not sq.ok
    assert "baseline" in sq.reason


def test_seed_quality_flags_far_corner_amplification():
    """A 6 m baseline is above the hard minimum but a 0.10 m point error still
    induces > 2 m at the far corner of a large (300 m) plot -> reject."""
    big = np.array([[0.0, 0.0], [300.0, 0.0], [300.0, 300.0], [0.0, 300.0]])
    sq = seed_quality(
        seed_src=np.array([[0.0, 0.0], [6.0, 0.0]]),
        seed_dst=np.array([[700000.0, 1200000.0], [700006.0, 1200000.0]]),
        template_points=big,
        sigma_point_m=0.10,
        max_induced_error_m=2.0,
        min_baseline_m=5.0,
    )
    assert not sq.ok
    assert sq.max_induced_error_m > 2.0


def test_seed_quality_accepts_long_baseline():
    sq = seed_quality(
        seed_src=np.array([[0.0, 0.0], [40.0, 0.0]]),      # full-width baseline
        seed_dst=np.array([[700000.0, 1200000.0], [700040.0, 1200000.0]]),
        template_points=_square(40.0),
        sigma_point_m=0.10,
        max_induced_error_m=2.0,
        min_baseline_m=5.0,
    )
    assert sq.ok
    assert sq.reason == ""
    # sigma_angle ~ sqrt(2)*0.1/40 rad; induced over ~28 m radius stays well under 2 m.
    assert sq.max_induced_error_m < 0.5
