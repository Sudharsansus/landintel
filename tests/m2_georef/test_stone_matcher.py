"""Unified rigid stone-matcher -- the client's base-stone + rotate approach, locked.

Every test is synthetic general geometry (no village data). The contract under test:
  * scale is 1 BY CONSTRUCTION (edge lengths exactly preserved -- client rule #2)
  * base-stone search finds the pose regardless of ordering/labels
  * the full-confidence bar is min(5, n_corners) -- data-keyed, never impossible
  * partial/no evidence is reported honestly, never fabricated
"""
from __future__ import annotations

import math

import numpy as np

from landintel.pipeline.m2_georef.stone_matcher import (
    RigidStoneMatch,
    rigid_procrustes,
    rigid_stone_match,
)


def _rot(pts, angle_deg, t=(0.0, 0.0)):
    a = math.radians(angle_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return np.asarray(pts, float) @ R.T + np.asarray(t, float)


SQUARE = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 30.0], [0.0, 30.0]])
PENT = np.array([[0.0, 0.0], [60.0, 0.0], [75.0, 40.0], [30.0, 65.0], [-10.0, 35.0]])


def _edge_lens(r):
    n = len(r)
    return np.array([np.linalg.norm(r[(i + 1) % n] - r[i]) for i in range(n)])


# ------------------------------------------------------------------ core matching
def test_recovers_exact_pose_scale_locked():
    tgt = _rot(PENT, 37.0, (781000.0, 1253000.0))
    m = rigid_stone_match(PENT, tgt, tol_m=1.0)
    assert m is not None and m.full
    assert m.n_matched == 5 and m.mean_residual < 1e-6
    assert m.s == 1.0                                     # scale locked, not fitted
    placed = m.apply(PENT)
    # edge lengths EXACTLY preserved (rule #2: never change FMB lengths)
    assert np.allclose(_edge_lens(placed), _edge_lens(PENT), atol=1e-9)
    assert np.allclose(placed, tgt, atol=1e-6)


def test_base_stone_search_survives_decoys_and_shuffle():
    # Target cloud = the true seat + decoy stones; src rows shuffled (labels
    # meaningless). The base-stone search must still find the true pose.
    rng = np.random.default_rng(7)
    tgt_true = _rot(PENT, -71.0, (500000.0, 1200000.0))
    decoys = rng.uniform(-400, 400, size=(12, 2)) + (500000.0, 1200000.0)
    tgt = np.vstack([decoys[:6], tgt_true, decoys[6:]])
    src = PENT[rng.permutation(5)]
    m = rigid_stone_match(src, tgt, tol_m=1.0)
    assert m is not None and m.full and m.n_matched == 5
    assert np.allclose(np.sort(m.apply(src), axis=0),
                       np.sort(tgt_true, axis=0), atol=1e-6)


def test_tolerates_field_jitter():
    # ~0.5 m of per-stone jitter (FMB-vs-cadastre reality) still yields a full match.
    rng = np.random.default_rng(3)
    tgt = _rot(PENT, 12.0, (600000.0, 1100000.0)) + rng.normal(0, 0.5, size=(5, 2))
    m = rigid_stone_match(PENT, tgt, tol_m=3.0)
    assert m is not None and m.full
    assert m.mean_residual < 2.0
    # geometry still rigid: edge lengths of the PLACED plot equal the FMB's exactly
    assert np.allclose(_edge_lens(m.apply(PENT)), _edge_lens(PENT), atol=1e-9)


# ------------------------------------------------------- conditional 5-stone bar
def test_full_bar_is_conditional_on_plot_corner_count():
    # 4-corner plot, bar=5 -> required is 4 (all its corners), so a perfect
    # 4/4 match IS full confidence. The flat "5" would have made this impossible.
    tgt = _rot(SQUARE, 20.0, (700000.0, 1150000.0))
    m = rigid_stone_match(SQUARE, tgt, tol_m=1.0, full_match_bar=5)
    assert m is not None
    assert m.required == 4 and m.n_matched == 4 and m.full


def test_five_corner_plot_needs_five():
    # 5-corner plot with only 4 stones present in the target -> NOT full (4 < 5),
    # reported honestly as partial evidence.
    tgt = _rot(PENT[:4], 20.0, (700000.0, 1150000.0))    # 5th stone missing
    m = rigid_stone_match(PENT, tgt, tol_m=1.0, full_match_bar=5)
    assert m is not None
    assert m.required == 5 and m.n_matched == 4 and not m.full


# ---------------------------------------------------------------- honest failure
def test_unrelated_target_is_not_full():
    # A completely different constellation: no congruent pose exists, so any
    # 2-point coincidence must be reported as partial, never full.
    rng = np.random.default_rng(11)
    tgt = rng.uniform(0, 1000, size=(8, 2))
    m = rigid_stone_match(PENT, tgt, tol_m=1.0)
    assert m is None or not m.full


def test_degenerate_input_returns_none():
    assert rigid_stone_match(PENT[:1], PENT, tol_m=1.0) is None      # 1 src stone
    assert rigid_stone_match(PENT, PENT[:1], tol_m=1.0) is None      # 1 target


# ------------------------------------------------------------- rigid procrustes
def test_rigid_procrustes_never_scales():
    # src and dst differ by scale 1.3: the rotation-only fit must NOT absorb it --
    # applying (R, t) preserves src edge lengths exactly (residual shows the truth).
    src = SQUARE
    dst = 1.3 * _rot(SQUARE, 45.0, (100.0, 200.0))
    R, t, res = rigid_procrustes(src, dst)
    placed = src @ R.T + t
    assert np.allclose(_edge_lens(placed), _edge_lens(src), atol=1e-9)
    assert float(res.mean()) > 1.0                       # honest misfit, not hidden


def test_rigid_procrustes_rejects_reflection():
    src = PENT
    dst = _rot(PENT, 30.0, (10.0, 20.0))
    R, _t, res = rigid_procrustes(src, dst)
    assert np.linalg.det(R) > 0.99                       # proper rotation, no flip
    assert float(res.max()) < 1e-6
