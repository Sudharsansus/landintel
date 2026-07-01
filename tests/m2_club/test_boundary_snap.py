"""Tests for the inter-plot boundary snap (m2_club.boundary_snap).

The snap is a QUALITY pass: adjacent ACCEPT plots seated independently land NEAR a
common edge but not exactly on it. ``snap_shared_boundaries`` clusters coincident
corners across neighbours and snaps them to one position so the shared edge is exactly
coincident -- with 0-FP guards that revert any unsafe plot.
"""
from __future__ import annotations

import numpy as np

from landintel.pipeline.m2_club.boundary_snap import snap_shared_boundaries
from landintel.pipeline.m2_club.placement import CandidatePlacement, ClubResult


def _square(corners, *, method="cadastral", rec="ACCEPT") -> ClubResult:
    """Build a placed ClubResult from absolute-UTM corner coords (list of (x, y))."""
    adj = np.asarray(corners, float)
    pl = CandidatePlacement(
        method=method,
        R=np.eye(2), s=1.0, t=np.zeros(2),
        adjusted=adj,
        corner_ring=list(range(len(corners))),
        passes_gate=True,
    )
    r = ClubResult(m1_file=f"m1_{id(adj)}.dxf", survey_number=str(id(adj) % 100000),
                   recommendation=rec, method=method, placement=pl)
    return r


def _shared_edge_gap(ra: ClubResult, rb: ClubResult, tol=5.0):
    """Min distance between each of rb's near corners and ra's nearest corner (m)."""
    A = ra.placement.corner_points()
    B = rb.placement.corner_points()
    gaps = []
    for b in B:
        d = np.min(np.hypot(A[:, 0] - b[0], A[:, 1] - b[1]))
        if d <= tol + 1.0:
            gaps.append(d)
    return gaps


def test_two_adjacent_squares_snapped_to_shared_edge():
    """Plot A occupies x in [0,50]; plot B is placed 2 m to the right (x in [52,102]),
    so their shared edge is 2 m apart. After the snap the shared corners coincide."""
    a = _square([(0, 0), (50, 0), (50, 30), (0, 30)])
    b = _square([(52, 0), (102, 0), (102, 30), (52, 30)])

    # Pre: A's east corners (50,0)/(50,30) vs B's west corners (52,0)/(52,30) -> 2 m apart.
    pre = _shared_edge_gap(a, b)
    assert pre and max(pre) >= 1.5

    stats = snap_shared_boundaries([a, b], tol=5.0)
    assert stats.n_clusters == 2          # two shared corners -> two clusters
    assert stats.max_corner_move <= 5.0

    # Post: the shared corners now coincide (within a tight epsilon).
    A = a.placement.corner_points()
    B = b.placement.corner_points()
    for bx, by in B:
        d = np.min(np.hypot(A[:, 0] - bx, A[:, 1] - by))
        if d < 5.0:                       # only the two shared corners qualify
            assert d < 1e-6, f"shared corner not coincident: gap {d}"

    # Each shared corner snapped to the midpoint (x=51) of the 2 m gap.
    east_a = sorted([p for p in A if p[0] > 25], key=lambda p: p[1])
    west_b = sorted([p for p in B if p[0] < 77], key=lambda p: p[1])
    assert np.allclose([p[0] for p in east_a], 51.0, atol=1e-6)
    assert np.allclose([p[0] for p in west_b], 51.0, atol=1e-6)

    # Centroids barely moved (rigid preservation: 1 m each, half the 2 m gap).
    assert stats.max_centroid_move <= 1.5
    assert not stats.skipped


def test_snap_exceeding_tol_is_skipped():
    """A neighbour 12 m away exceeds the 5 m tol -> the corners are NOT clustered, so
    nothing snaps and the original placement is preserved (no false re-placement)."""
    a = _square([(0, 0), (50, 0), (50, 30), (0, 30)])
    b = _square([(62, 0), (112, 0), (112, 30), (62, 30)])   # 12 m gap

    before_b = b.placement.adjusted.copy()
    stats = snap_shared_boundaries([a, b], tol=5.0)

    assert stats.n_clusters == 0
    assert stats.n_corners_snapped == 0
    # Untouched: corners further than tol are never dragged across.
    assert np.allclose(b.placement.adjusted, before_b)
    assert np.allclose(a.placement.corner_points()[:, 0].max(), 50.0)


def test_review_and_no_coverage_plots_are_not_snapped():
    """Only placed (ACCEPT/ACCEPT_SEEDED) plots participate -- a REVIEW neighbour is
    ignored even if its corners are within tol."""
    a = _square([(0, 0), (50, 0), (50, 30), (0, 30)], rec="ACCEPT")
    b = _square([(52, 0), (102, 0), (102, 30), (52, 30)], rec="REVIEW")
    before_a = a.placement.adjusted.copy()
    before_b = b.placement.adjusted.copy()

    stats = snap_shared_boundaries([a, b], tol=5.0)
    assert stats.n_clusters == 0
    assert np.allclose(a.placement.adjusted, before_a)
    assert np.allclose(b.placement.adjusted, before_b)


def test_three_plots_meeting_at_a_corner_snap_to_one_point():
    """Three plots whose corners cluster near one shared stone snap to a single point."""
    # Common stone near (50,30); each plot offsets its copy by ~1.5 m.
    a = _square([(0, 0), (50, 0), (51, 31), (0, 30)])
    b = _square([(50, 0), (100, 0), (100, 30), (49, 29)])
    c = _square([(49, 31), (100, 31), (100, 61), (49, 61)])

    stats = snap_shared_boundaries([a, b, c], tol=5.0)
    assert stats.n_corners_snapped >= 3
    # The three near-(50,30) corners now coincide.
    pa = a.placement.corner_points()[2]
    pb = b.placement.corner_points()[3]
    pc = c.placement.corner_points()[0]
    assert np.allclose(pa, pb, atol=1e-6) and np.allclose(pb, pc, atol=1e-6)
    assert not stats.skipped


def test_snap_creating_new_overlap_is_reverted():
    """If snapping would shove a plot's interior into a neighbour beyond the overlap
    cap, the snap on the more-moved plot is reverted (0-FP: never create overlaps).

    A is fixed (x in [0,50]). B is a tiny 3 m square placed just OUTSIDE A's right edge
    (no pre-overlap), with BOTH its left AND right corners within tol of A's right-edge
    corners -- so snapping pulls B's whole body across A's edge, deep inside A. The
    overlap guard must detect that NEW overlap and revert B."""
    a = _square([(0, 0), (50, 0), (50, 30), (0, 30)])
    # 3 m square hugging A's right edge: corners near A's (50,0) and (50,30) so they
    # cluster, but the body lies just outside -> snapping drags it inside A.
    b = _square([(51, 0), (54, 0), (54, 30), (51, 30)])

    # No pre-existing overlap (B is entirely right of x=50).
    fa0, fb0 = a.placement.footprint(), b.placement.footprint()
    assert not fa0.intersects(fb0) or fa0.intersection(fb0).area < 1e-6

    snap_shared_boundaries([a, b], tol=5.0)

    # After the snap+guard, no interior overlap above the tiling cap may remain.
    fa, fb = a.placement.footprint(), b.placement.footprint()
    ov = (fa.intersection(fb).area / min(fa.area, fb.area)) if fa.intersects(fb) else 0.0
    assert ov <= 0.20
