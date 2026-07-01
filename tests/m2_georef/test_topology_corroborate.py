"""Tests for the FP-critical shared-edge primitive behind topology corroboration.

Topology upgrades a located REVIEW plot to ACCEPT only when it shares a REAL property
line with a confident neighbour. A corner-graze (short tangent) must NOT count -- that
is the coincidental case where a false positive could hide.
"""
from __future__ import annotations

from landintel.pipeline.m2_georef.pipeline import _shared_edge_length
from shapely.geometry import Polygon


def test_adjacent_parcels_share_full_edge():
    # Two 100x100 squares sharing the x=100 edge.
    a = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    b = Polygon([(100, 0), (200, 0), (200, 100), (100, 100)])
    assert _shared_edge_length(a, b) > 90  # ~100 m shared boundary


def test_separated_parcels_share_nothing():
    a = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    b = Polygon([(150, 0), (250, 0), (250, 100), (150, 100)])  # 50 m gap
    assert _shared_edge_length(a, b) == 0.0


def test_corner_graze_is_short():
    # Squares meeting only near one corner -> small shared length, below the 20 m bar.
    a = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    b = Polygon([(100, 95), (200, 95), (200, 195), (100, 195)])  # overlap only y95-100
    assert _shared_edge_length(a, b) < 20.0
