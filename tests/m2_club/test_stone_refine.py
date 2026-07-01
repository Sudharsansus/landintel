"""stone_refine: a confident congruent fit snaps the FMB onto true stones; weak fits keep seat."""
from __future__ import annotations

import numpy as np

from landintel.pipeline.m2_club.placement import CandidatePlacement, ClubResult
from landintel.pipeline.m2_club.stone_refine import refine_to_stones


def _ring(cx, cy, r=50.0):
    # An irregular (non-symmetric) octagon so the rigid fit has a UNIQUE orientation
    # (a regular polygon's rotational symmetry admits several equal-inlier fits).
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    rad = r * np.array([1.0, 0.7, 1.2, 0.6, 1.1, 0.8, 1.3, 0.65])
    return np.column_stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)])


def _result(sn, ring):
    pl = CandidatePlacement(method="cadastral", R=np.eye(2), s=1.0, t=np.zeros(2),
                            adjusted=ring.copy(), corner_ring=list(range(len(ring))),
                            passes_gate=True)
    return ClubResult(m1_file=f"{sn}.dxf", survey_number=sn, recommendation="ACCEPT",
                      method="cadastral", placement=pl)


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def test_confident_fit_snaps_to_stones():
    truth = _ring(0, 0)                       # true stones
    # our placed FMB = same ring rotated 20 deg + shifted 12 m (cadastre error).
    placed = (_rot(np.radians(20)) @ _ring(0, 0).T).T + np.array([12.0, -8.0])
    r = _result("A", placed)
    stats = refine_to_stones([r], truth)
    assert stats.n_refined == 1 and "A" in stats.anchored
    # after refine, corners land on the true stones.
    d = np.hypot(*(r.placement.corner_points() - truth).T)
    assert d.max() < 1.0
    # rigid: edge lengths preserved.
    ring = r.placement.corner_points()
    assert abs(np.hypot(*(ring[1] - ring[0])) - np.hypot(*(_ring(0, 0)[1] - _ring(0, 0)[0]))) < 1e-6


def test_weak_fit_keeps_cadastre_seat():
    truth = _ring(0, 0)
    rng = np.random.default_rng(1)
    noise = _ring(500, 500) + rng.normal(0, 30, size=(8, 2))   # far away + noisy
    r = _result("B", noise)
    before = r.placement.adjusted.copy()
    stats = refine_to_stones([r], truth)
    assert stats.n_refined == 0
    assert np.allclose(r.placement.adjusted, before)          # untouched


def test_far_jump_rejected():
    # a look-alike ring far from the cadastre seat must NOT be snapped across (max_shift).
    truth = _ring(0, 0)
    placed = _ring(300, 0)                    # identical shape, 300 m away
    r = _result("C", placed)
    stats = refine_to_stones([r], truth)
    assert stats.n_refined == 0


def _sq(x0, y0, s=40.0):
    return np.array([(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s)], float)


def _res_sq(sn, ring):
    pl = CandidatePlacement(method="cadastral", R=np.eye(2), s=1.0, t=np.zeros(2),
                            adjusted=ring.copy(), corner_ring=[0, 1, 2, 3], passes_gate=True)
    return ClubResult(m1_file=f"{sn}.dxf", survey_number=sn, recommendation="ACCEPT",
                      method="cadastral", placement=pl)


def test_resolve_overlaps_reverts_then_keeps_tiling():
    from landintel.pipeline.m2_club.stone_refine import resolve_overlaps
    # A anchored; B (unanchored) currently stacked on A but its cadastre original is clear.
    a = _res_sq("A", _sq(0, 0))
    b = _res_sq("B", _sq(5, 0))                       # 88% overlap with A now
    originals = {"A": (a.placement.R.copy(), a.placement.t.copy(), a.placement.adjusted.copy()),
                 "B": (b.placement.R.copy(), b.placement.t.copy(), _sq(100, 0))}  # clear seat
    acts = resolve_overlaps([a, b], originals, anchored={"A"})
    assert ("B", "A") in acts
    # B reverted to its clear original -> no overlap, both still ACCEPT.
    assert b.placement.adjusted[0][0] == 100 and b.recommendation == "ACCEPT"


def test_resolve_overlaps_demotes_when_revert_cannot_separate():
    from landintel.pipeline.m2_club.stone_refine import resolve_overlaps
    a = _res_sq("A", _sq(0, 0))
    b = _res_sq("B", _sq(5, 0))
    # B's cadastre original ALSO overlaps A (a genuine data conflict) -> demote to REVIEW.
    originals = {"A": (a.placement.R.copy(), a.placement.t.copy(), a.placement.adjusted.copy()),
                 "B": (b.placement.R.copy(), b.placement.t.copy(), _sq(5, 0))}
    resolve_overlaps([a, b], originals, anchored={"A"})
    assert b.recommendation == "REVIEW"
