"""Geometry helpers built ONLY on the existing stack (numpy / scipy / shapely) -- no new deps.

``concave_hull`` / ``village_fence`` give a snug alpha-shape outline instead of a convex hull.
For an elongated, band-shaped village (INGUR's plots run in a diagonal ribbon), the convex hull
is a big triangle full of empty space, so the fence lets in far cross-village survey labels; a
concave hull hugs the band, so the fence drops those labels -- strengthening the 0-FP lock at no
recall cost (a valid label sits inside its plot, i.e. inside the band + buffer).
"""
from __future__ import annotations

import numpy as np


def concave_hull(points, alpha: float | None = None, buffer: float = 0.0):
    """Alpha-shape concave hull of 2-D points via Delaunay (scipy) + triangle union (shapely).

    Keeps Delaunay triangles whose circumradius is below a threshold (small radius = tight,
    band-hugging) and unions them. ``alpha`` sets the threshold as ``1/alpha``; when None it
    auto-derives from the median triangle-edge length (robust to scale). Falls back to the convex
    hull if the alpha shape degenerates (too few points, empty, or < 20 % of convex-hull area).
    """
    from scipy.spatial import Delaunay
    from shapely.geometry import MultiPoint, Polygon
    from shapely.ops import unary_union

    pts = np.asarray(points, float)
    conv = MultiPoint([tuple(p) for p in pts]).convex_hull
    if len(pts) < 4:
        return conv.buffer(buffer) if buffer else conv

    tri = Delaunay(pts)
    edge_lens: list[float] = []
    tris: list[tuple[float, tuple[int, int, int]]] = []
    for ia, ib, ic in tri.simplices:
        a, b, c = pts[ia], pts[ib], pts[ic]
        ab, bc, ca = np.hypot(*(a - b)), np.hypot(*(b - c)), np.hypot(*(c - a))
        s = (ab + bc + ca) / 2.0
        area = float(max(s * (s - ab) * (s - bc) * (s - ca), 1e-12)) ** 0.5
        circ_r = (ab * bc * ca) / (4.0 * area)
        tris.append((circ_r, (ia, ib, ic)))
        edge_lens.extend((ab, bc, ca))

    thr = (2.5 * float(np.median(edge_lens))) if alpha is None else (1.0 / alpha)
    keep = [Polygon([pts[i] for i in ijk]) for r, ijk in tris if r <= thr]
    if not keep:
        return conv.buffer(buffer) if buffer else conv

    shape = unary_union(keep)
    if buffer:
        shape = shape.buffer(buffer)
    ref = conv.buffer(buffer) if buffer else conv
    # Guard: a degenerate alpha shape (empty or far smaller than the convex hull) -> convex.
    if shape.is_empty or shape.area < 0.2 * conv.area:
        return ref
    return shape


def village_fence(points, buffer: float = 300.0, concave: bool = True,
                  alpha: float | None = None):
    """Village fence polygon from a point cloud (surveyor stones).

    Concave (alpha-shape) by default so the fence is snug around a band-shaped village -> better
    cross-village label rejection -- with a convex-hull fallback (``concave=False`` or < 4 points).
    """
    from shapely.geometry import MultiPoint

    pts = [tuple(p) for p in points]
    if not concave or len(pts) < 4:
        return MultiPoint(pts).convex_hull.buffer(buffer)
    return concave_hull(pts, alpha=alpha, buffer=buffer)
