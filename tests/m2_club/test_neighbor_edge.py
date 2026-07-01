"""Diamond 3 (audit): neighbour-edge matching uses point-to-segment, not midpoint.

A neighbour label sits somewhere ALONG the shared edge -- often near a corner. Matching
the label to the edge whose MIDPOINT is nearest mis-picks a perpendicular edge whose
midpoint happens to be closer to the corner-positioned label. Point-to-segment distance
is the correct measure. This test builds exactly that corner case and asserts the correct
(long, shared) edge is chosen. GENERAL geometry -- no survey numbers, no fixtures.
"""
from __future__ import annotations

import math

from landintel.pipeline.m2_georef.extract_m1 import M1PlotData, M1Stone
from landintel.pipeline.m2_club.relative_club import _neighbor_edge, _point_seg_dist


def test_point_seg_dist_basic():
    # point on the segment -> 0; point off the end -> distance to nearest endpoint.
    assert _point_seg_dist(5, 0, 0, 0, 10, 0) == 0.0            # midpoint, on segment
    assert _point_seg_dist(5, 3, 0, 0, 10, 0) == 3.0            # perpendicular offset
    assert _point_seg_dist(-4, 0, 0, 0, 10, 0) == 4.0          # beyond endpoint A
    assert _point_seg_dist(0, 0, 5, 5, 5, 5) == math.hypot(5, 5)  # degenerate segment


def _tall_plot():
    # Tall rectangle: stones 0..3 at the corners; outer ring = [0,1,2,3].
    ring = [(0.0, 0.0), (40.0, 0.0), (40.0, 90.0), (0.0, 90.0)]
    stones = [M1Stone(x=x, y=y, label=str(i), index=i) for i, (x, y) in enumerate(ring)]
    m1 = M1PlotData(stones=stones, survey_number="100")
    m1.outer_stone_indices = [0, 1, 2, 3]
    return m1


def test_neighbor_edge_picks_correct_edge_for_corner_label():
    """Neighbour label near the TOP of the long right edge (x=40) must match that right
    edge (stones 1->2), NOT the top edge (2->3) whose midpoint is nearer the label."""
    m1 = _tall_plot()
    # label just outside the right edge, near its top corner (40, 90)
    m1.neighbor_label_texts = [{"x": 41.0, "y": 85.0, "text": "999"}]

    result = _neighbor_edge(m1, "999")
    assert result is not None
    a, b, length = result
    assert {a, b} == {1, 2}, f"expected the right edge (stones 1,2), got ({a},{b})"
    assert length == 90.0  # the long shared edge

    # sanity: the OLD midpoint rule would have mis-picked the top edge (2,3)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    pts = [(0.0, 0.0), (40.0, 0.0), (40.0, 90.0), (0.0, 90.0)]
    lbl = (41.0, 85.0)
    mid_pick = min(edges, key=lambda e: math.hypot(
        (pts[e[0]][0] + pts[e[1]][0]) / 2 - lbl[0],
        (pts[e[0]][1] + pts[e[1]][1]) / 2 - lbl[1]))
    assert set(mid_pick) == {2, 3}, "test premise: midpoint rule mis-picks the top edge"
