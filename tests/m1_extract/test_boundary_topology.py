"""Boundary topology USE CASES -- general synthetic geometry, NO village data.

Each real village surfaced a boundary-topology bug (mid-edge stone shown as one line,
duplicate edge from a shared corner, subdivision line short of the boundary). Rather than
patch per village, these tests codify the SCENARIOS on synthetic geometry so the behaviour
is locked in and provably general:

  * a stone sitting mid-edge becomes a vertex (edge is DIVIDED at the point)
  * a node at/near a corner does NOT create a duplicate/overlapping edge
  * two nodes closer than the dedup distance collapse to one split point
  * a node off the edge (or at an endpoint) is ignored
  * subdivision dangling ends attach to the boundary; T-junctions split it
  * verify_dxf's geometric checks flag exactly these conditions
"""
from __future__ import annotations

import math

from landintel.core.models import Boundary, CornerPoint, Plot
from landintel.pipeline.m1_extract.to_dxf import _split_edge_at_nodes, build_document


# --------------------------------------------------------------------------- #
# 1. _split_edge_at_nodes -- the core geometry (pure function, no fixtures)
# --------------------------------------------------------------------------- #
def _seglen(seg):
    (ax, ay), (bx, by) = seg
    return math.hypot(bx - ax, by - ay)


def test_mid_edge_node_splits_into_two():
    segs = _split_edge_at_nodes((0.0, 0.0), (10.0, 0.0), [(4.0, 0.1)])
    assert len(segs) == 2
    # the shared vertex is the node itself (the surveyor's true point)
    assert segs[0][1] == (4.0, 0.1) and segs[1][0] == (4.0, 0.1)


def test_node_near_endpoint_is_ignored_no_duplicate():
    # a node 0.5 m from A must NOT split (it IS that corner) -> single edge, no duplicate
    assert _split_edge_at_nodes((0.0, 0.0), (10.0, 0.0), [(0.5, 0.0)]) == [((0.0, 0.0), (10.0, 0.0))]
    assert _split_edge_at_nodes((0.0, 0.0), (10.0, 0.0), [(9.6, 0.0)]) == [((0.0, 0.0), (10.0, 0.0))]


def test_node_off_the_edge_is_ignored():
    # 5 m perpendicular off a 10 m edge -> not on it -> no split
    assert _split_edge_at_nodes((0.0, 0.0), (10.0, 0.0), [(5.0, 5.0)]) == [((0.0, 0.0), (10.0, 0.0))]


def test_two_close_nodes_collapse_to_one_split():
    # two nodes 0.2 m apart (< 0.3 m dedup) -> a single interior vertex, never a 0.2 m sliver
    segs = _split_edge_at_nodes((0.0, 0.0), (10.0, 0.0), [(5.0, 0.0), (5.2, 0.0)])
    assert len(segs) == 2
    assert all(_seglen(s) > 0.3 for s in segs)


def test_multiple_spread_nodes_ordered():
    segs = _split_edge_at_nodes((0.0, 0.0), (10.0, 0.0), [(7.0, 0.0), (3.0, 0.0)])
    assert len(segs) == 3
    xs = [s[0][0] for s in segs]
    assert xs == sorted(xs)                      # segments emitted left-to-right


def test_short_edge_with_two_nodes_no_duplicate_segment():
    # regression for the MOOLAKARAI _18 EXACT-duplicate: a 1.15 m edge whose endpoints are
    # both nodes must not emit a reversed/zero-length duplicate.
    segs = _split_edge_at_nodes((177.71, 126.12), (178.86, 126.12),
                                [(177.71, 126.12), (178.86, 126.12)])
    keys = [frozenset({(round(a[0], 2), round(a[1], 2)), (round(b[0], 2), round(b[1], 2))})
            for a, b in segs]
    assert len(keys) == len(set(keys))           # no duplicate edge
    assert all(len(k) == 2 for k in keys)        # no zero-length edge


# --------------------------------------------------------------------------- #
# 2. build_document integration -- the real writer, incl. the dedup pass
# --------------------------------------------------------------------------- #
def _boundary_edges(doc):
    out = []
    for e in doc.modelspace().query("LWPOLYLINE"):
        if e.dxf.layer == "BOUNDARY":
            pts = [(round(p[0], 2), round(p[1], 2)) for p in e.get_points()]
            for i in range(len(pts) - 1):
                out.append((pts[i], pts[i + 1]))
    return out


def _square(side=20.0):
    return [(0.0, 0.0), (side, 0.0), (side, side), (0.0, side), (0.0, 0.0)]


def _plot(survey_no, corner_points, subdivision_segments=None) -> Plot:
    return Plot(client_id="c", survey_no=survey_no, district="D", taluk="T", village="V",
                scale=2000, stated_area=0.04,
                boundary=Boundary(points=_square(20.0)),
                corner_points=corner_points,
                subdivision_segments=subdivision_segments or [])


def test_stone_mid_edge_becomes_boundary_vertex():
    """A stone 0.1 m off the middle of an edge must end up a boundary VERTEX (the client's
    'divide the line at the point'), not have the edge run straight through it."""
    plot = _plot("1", [CornerPoint(label="A", x=0.0, y=0.0),
                       CornerPoint(label="B", x=20.0, y=0.0),
                       CornerPoint(label="C", x=20.0, y=20.0),
                       CornerPoint(label="D", x=0.0, y=20.0),
                       CornerPoint(label="M", x=10.0, y=0.1)])  # mid bottom edge
    edges = _boundary_edges(build_document(plot))
    verts = {p for e in edges for p in e}
    assert any(math.hypot(vx - 10.0, vy - 0.1) < 0.2 for vx, vy in verts), \
        "mid-edge stone did not become a boundary vertex"


def test_stone_at_corner_creates_no_duplicate_edge():
    """A stone sitting AT a shared corner must not spawn a duplicate/overlapping edge."""
    plot = _plot("2", [CornerPoint(label="A", x=0.0, y=0.0),
                       CornerPoint(label="B", x=20.0, y=0.0),
                       CornerPoint(label="C", x=20.0, y=20.0),
                       CornerPoint(label="D", x=0.0, y=20.0)])
    edges = _boundary_edges(build_document(plot))
    keys = [frozenset(e) for e in edges]
    assert len(keys) == len(set(keys)), "duplicate boundary edge emitted"
    assert all(len(k) == 2 for k in keys), "zero-length boundary edge emitted"


def test_subdivision_tjunction_splits_boundary():
    """A subdivision line meeting a boundary edge mid-span splits the boundary there."""
    plot = _plot("3", [CornerPoint(label="A", x=0.0, y=0.0),
                       CornerPoint(label="B", x=20.0, y=0.0),
                       CornerPoint(label="C", x=20.0, y=20.0),
                       CornerPoint(label="D", x=0.0, y=20.0)],
                 subdivision_segments=[((10.0, 10.0), (10.0, 0.0))])  # ends on bottom edge
    edges = _boundary_edges(build_document(plot))
    verts = {p for e in edges for p in e}
    assert any(math.hypot(vx - 10.0, vy - 0.0) < 0.2 for vx, vy in verts), \
        "boundary not split at the subdivision T-junction"
