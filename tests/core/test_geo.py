"""core.geo: concave hull is tighter than convex for a band, still contains all points."""
from __future__ import annotations

import numpy as np
from shapely.geometry import Point, MultiPoint

from landintel.core.geo import concave_hull, village_fence


def _band(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([np.linspace(0, 1000, n), np.linspace(0, 1000, n)]) \
        + rng.normal(0, 12, (n, 2))


def test_concave_tighter_than_convex_and_contains_all():
    pts = _band()
    conv = MultiPoint([tuple(p) for p in pts]).convex_hull
    cc = concave_hull(pts, buffer=0.0)
    assert cc.area < 0.8 * conv.area                      # meaningfully tighter for a band
    assert all(cc.buffer(1e-6).contains(Point(*p)) for p in pts)   # loses no point


def test_village_fence_contains_all_stones_and_is_tighter():
    pts = _band()
    convex = village_fence(pts, buffer=100.0, concave=False)
    concave = village_fence(pts, buffer=100.0, concave=True)
    assert concave.area <= convex.area
    assert all(concave.contains(Point(*p)) for p in pts)  # every stone stays in-fence (no recall loss)


def test_degenerate_inputs_fall_back_to_convex():
    # < 4 points -> convex hull (buffered), never crashes.
    tri = np.array([(0, 0), (10, 0), (5, 8)], float)
    f = village_fence(tri, buffer=5.0)
    assert not f.is_empty and all(f.contains(Point(*p)) for p in tri)


def test_alpha_shape_never_drops_below_guard():
    # A blob (not a band): concave ~ convex; guard keeps it from collapsing.
    rng = np.random.default_rng(1)
    blob = rng.normal(0, 50, (120, 2))
    cc = concave_hull(blob, buffer=0.0)
    conv = MultiPoint([tuple(p) for p in blob]).convex_hull
    assert cc.area >= 0.2 * conv.area
