"""Better village fence than the surveyor-stone convex hull.

Two implementations:

  build_village_fence_alphashape  -- concave hull of S3 cadastral line vertices.
                                       Follows the actual cadastral boundary
                                       instead of the corridor shape. Much tighter.

  build_village_fence_pyclipper   -- same but using pyclipper for a robust
                                       integer-arithmetic offset at the end
                                       (handles sharp corners shapely.buffer
                                       can mangle).
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                      # for the "Polygon | None" return annotations
    from shapely.geometry import Polygon

_log = logging.getLogger(__name__)


def build_village_fence_alphashape(
    cadastral_lines_gdf,     # geopandas.GeoDataFrame of cadastral linework
    alpha: float = 0.001,    # smaller = tighter curve; 0.001 = village-scale
    buffer_m: float = 50.0,  # post-buffer for tolerance (50m ~ 1 tile pixel)
    fallback_convex_hull_pts=None,
) -> "Polygon | None":
    """Concave-hull village boundary from cadastral line vertices.

    Parameters
    ----------
    cadastral_lines_gdf : geopandas.GeoDataFrame
        One row per cadastral line; geometry column has LineStrings.
    alpha : float
        alphashape parameter; smaller = tighter curve. 0.001 is good for
        village-scale (1-50 ha) cadastre.
    buffer_m : float
        Buffer around the result in metres (since vertices are in degrees,
        buffer is computed as a degree offset via the centroid latitude).
    fallback_convex_hull_pts : iterable
        If alphashape fails or returns nothing, fall back to a convex hull
        of these points (e.g. surveyor stones).

    Returns
    -------
    shapely.geometry.Polygon or None.
    """
    try:
        import alphashape
    except ImportError:
        _log.warning("alphashape not installed; pip install alphashape")
        return _convex_hull_fallback(fallback_convex_hull_pts, buffer_m)

    pts = []
    for geom in cadastral_lines_gdf.geometry.values:
        if geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            pts.extend(list(geom.coords))
        elif geom.geom_type == "MultiLineString":
            for ls in geom.geoms:
                pts.extend(list(ls.coords))
    if len(pts) < 4:
        return _convex_hull_fallback(fallback_convex_hull_pts, buffer_m)

    pts_arr = np.array(pts)
    # Subsample for performance: alphashape is O(n^2) on Delaunay triangulation,
    # so 240k cadastral vertices can take 30+ minutes. Take every Nth until the
    # total is manageable (<10k for ~1s alphashape).
    max_pts = 8000
    if len(pts_arr) > max_pts:
        step = len(pts_arr) // max_pts
        pts_arr = pts_arr[::step]
    _log.info("Village fence: alphashape on %d cadastral vertices", len(pts_arr))
    fence = alphashape.alphashape(pts_arr, alpha=alpha)
    if fence is None or fence.is_empty:
        return _convex_hull_fallback(fallback_convex_hull_pts, buffer_m)

    # Buffer by buffer_m metres (converted to degrees via centroid lat).
    if buffer_m > 0:
        lat = np.mean(pts_arr[:, 1])
        deg_per_m = 1.0 / (111000.0 * np.cos(np.radians(lat)))
        fence = fence.buffer(buffer_m * deg_per_m)
    return fence


def build_village_fence_pyclipper(
    cadastral_lines_gdf,
    alpha: float = 0.001,
    offset_m: float = 50.0,
    fallback_convex_hull_pts=None,
) -> "Polygon | None":
    """Same as build_village_fence_alphashape but offset via pyclipper.

    pyclipper uses the Vatti algorithm which is robust at sharp corners
    (shapely.buffer can produce self-intersecting artefacts on tightly
    concave boundaries). Worth using if the alphashape + buffer result
    has self-intersections.
    """
    fence = build_village_fence_alphashape(
        cadastral_lines_gdf, alpha=alpha, buffer_m=0,
        fallback_convex_hull_pts=fallback_convex_hull_pts)
    if fence is None:
        return None

    try:
        import pyclipper
    except ImportError:
        _log.warning("pyclipper not installed; pip install pyclipper")
        if offset_m > 0:
            lat = (fence.bounds[1] + fence.bounds[3]) / 2.0
            deg_per_m = 1.0 / (111000.0 * np.cos(np.radians(lat)))
            return fence.buffer(offset_m * deg_per_m)
        return fence

    # Scale to integer arithmetic.
    scale = 1e8
    polys = []
    if fence.geom_type == "Polygon":
        polys.append(fence)
    elif fence.geom_type == "MultiPolygon":
        polys.extend(fence.geoms)
    if not polys:
        return None

    lat = (fence.bounds[1] + fence.bounds[3]) / 2.0
    deg_per_m = 1.0 / (111000.0 * np.cos(np.radians(lat)))
    offset_units = offset_m * deg_per_m * scale

    pco = pyclipper.PyclipperOffset()
    for poly in polys:
        path = pyclipper.scale_to_clipper(list(poly.exterior.coords), scale)
        pco.AddPath(path, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)

    solutions = pco.Execute(offset_units) if offset_units > 0 else pco.Execute(-abs(offset_units))
    if not solutions:
        return fence  # offset failed; return un-offset fence

    from shapely.geometry import Polygon as ShapelyPoly
    out_polys = []
    for sol in solutions:
        coords = pyclipper.scale_from_clipper(sol, scale)
        out_polys.append(ShapelyPoly(coords))
    if len(out_polys) == 1:
        return out_polys[0]
    from shapely.geometry import MultiPolygon
    return MultiPolygon(out_polys)


def _convex_hull_fallback(pts, buffer_m: float) -> "Polygon | None":
    """Fallback fence = convex hull of given points (e.g. surveyor stones)."""
    if not pts or len(pts) < 3:
        return None
    from shapely.geometry import MultiPoint
    fence = MultiPoint(pts).convex_hull
    if buffer_m > 0:
        lat = sum(p[1] for p in pts) / len(pts)
        deg_per_m = 1.0 / (111000.0 * np.cos(np.radians(lat)))
        fence = fence.buffer(buffer_m * deg_per_m)
    return fence
