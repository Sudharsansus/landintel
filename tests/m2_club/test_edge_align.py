"""edge_align: corroborated shared edges merge (translation only); non-shared don't move."""
from __future__ import annotations

import numpy as np

from landintel.pipeline.m2_club.edge_align import align_shared_edges
from landintel.pipeline.m2_club.placement import CandidatePlacement, ClubResult


def _square(x0, y0, side=100.0):
    return np.array([(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)],
                    float)


def _result(sn, ring, method="cadastral", rec="ACCEPT"):
    pl = CandidatePlacement(method=method, R=np.eye(2), s=1.0, t=np.zeros(2),
                            adjusted=ring.copy(), corner_ring=[0, 1, 2, 3],
                            passes_gate=True)
    return ClubResult(m1_file=f"{sn}.dxf", survey_number=sn, recommendation=rec,
                      method=method, placement=pl)


def test_offset_neighbours_merge_to_shared_edge():
    # A: x in [0,100]; B: x in [110,210] -> a 10 m gap on the shared vertical edge.
    a = _result("A", _square(0, 0))
    b = _result("B", _square(110, 0))
    stats = align_shared_edges([a, b])

    assert stats.n_constraints == 1
    assert stats.n_moved == 2
    # A's right edge and B's left edge should now coincide (gap ~0, was 10).
    a_right = a.placement.adjusted[1][0]      # x of (100,0) corner after move
    b_left = b.placement.adjusted[0][0]       # x of (110,0) corner after move
    assert abs(a_right - b_left) < 0.5
    # translation only: shape (edge lengths) preserved exactly.
    ring = a.placement.adjusted
    assert abs(np.hypot(*(ring[1] - ring[0])) - 100.0) < 1e-6
    # t and adjusted moved together by the same vector.
    assert np.allclose(a.placement.t, a.placement.adjusted[0] - np.array([0.0, 0.0]))


def test_perpendicular_neighbours_do_not_move():
    # A square, and B placed so its nearest edge is perpendicular (corner-only touch):
    # a diamond offset to the side. No near-parallel overlapping edge -> no constraint.
    a = _result("A", _square(0, 0))
    diamond = np.array([(160, 50), (210, 0), (260, 50), (210, 100)], float)
    b = _result("B", diamond)
    t_before = b.placement.t.copy()
    stats = align_shared_edges([a, b])
    assert stats.n_constraints == 0
    assert stats.n_moved == 0
    assert np.allclose(b.placement.t, t_before)


def test_review_plots_are_not_moved():
    a = _result("A", _square(0, 0))
    b = _result("B", _square(110, 0), rec="REVIEW")   # REVIEW -> not placed
    stats = align_shared_edges([a, b])
    assert stats.n_moved == 0                          # only one ACCEPT -> nothing to align
