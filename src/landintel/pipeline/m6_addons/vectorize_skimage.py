"""Skeleton-based parcel vectorisation using scikit-image (vs cv2.connectedComponentsWithStats).

Replaces vectorize_parcels() in m5_cadastral/s3_tiles.py. scikit-image gives:
  - skeletonize() that handles small gaps better than cv2.ximgproc.thinning
  - remove_small_objects() to drop noise components
  - regionprops() with orientation/centroid/bbox info

The workflow:
  1. Read all tile PNGs into a single canvas (same as before).
  2. Extract yellow line mask (HSV inRange) and INVERT to get parcel interiors.
  3. Skeletonize the yellow lines (so they're 1-pixel wide) -- bridges small gaps.
  4. Connected-component label the FREE space (parcel interiors).
  5. For each component above min_area: extract the contour via marching squares
     (skimage.measure.find_contours) instead of cv2.findContours.

Cleaner parcels than the cv2 approach -- the contour is on the SKeleton of the
boundary, not on the rasterised edge itself, so it's less noisy.
"""
from __future__ import annotations
import logging
from pathlib import Path

import cv2
import numpy as np

_log = logging.getLogger(__name__)

_YELLOW = (np.array([20, 80, 90]), np.array([45, 255, 255]))
_TILE = 256


def skeletonize_parcels_skimage(
    tiles: dict[tuple[int, int], Path],
    grid,                          # TileGrid (utm_to_px, px_to_utm)
    min_area_m2: float = 150.0,
    max_area_m2: float = 200000.0,
    closing_kernel: int = 3,
    simplify_tol_m: float = 1.5,
):
    """Vectorise cadastral parcels using skimage skeleton + regionprops.

    Returns a list[Polygon] in the same UTM CRS as the TileGrid.
    """
    try:
        from skimage.morphology import skeletonize, remove_small_objects, closing as binary_closing
        from skimage.measure import label, regionprops, find_contours
    except ImportError:
        _log.warning("skimage not installed; pip install scikit-image")
        return []

    if not tiles:
        return []
    txs = [k[0] for k in tiles]; tys = [k[1] for k in tiles]
    tx0, tx1, ty0, ty1 = min(txs), max(txs), min(tys), max(tys)
    W = (tx1 - tx0 + 1) * _TILE; H = (ty1 - ty0 + 1) * _TILE
    canvas = np.zeros((H, W, 3), np.uint8)
    for (tx, ty), path in tiles.items():
        im = cv2.imread(str(path))
        if im is not None:
            canvas[(ty - ty0) * _TILE:(ty - ty0 + 1) * _TILE,
                   (tx - tx0) * _TILE:(tx - tx0 + 1) * _TILE] = im

    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, *_YELLOW)
    # Close gaps in the boundary lines.
    yellow = binary_closing(yellow > 0,
                            footprint=np.ones((closing_kernel, closing_kernel)))
    yellow_skel = skeletonize(yellow)             # 1-pixel-wide skeleton

    # Free space = parcels + unbounded background. Invert the skeleton.
    free = ~yellow_skel
    labels = label(free, connectivity=2)
    n = labels.max()

    m2_per_px = _approx_m2_per_px(grid, tx0, ty0)
    polys = []
    from shapely.geometry import Polygon
    for region in regionprops(labels):
        x, y, w, h, area_px = (
            region.bbox[1], region.bbox[0],
            region.bbox[3] - region.bbox[1], region.bbox[2] - region.bbox[0],
            region.area)
        # Skip the unbounded background (touches canvas edges).
        if x <= 0 or y <= 0 or x + w >= W or y + h >= H:
            continue
        area_m2 = area_px * m2_per_px
        if not (min_area_m2 <= area_m2 <= max_area_m2):
            continue
        # marching-squares contour on the binary mask of this region.
        comp = (labels == region.label).astype(np.uint8)
        contours = find_contours(comp, 0.5)
        if not contours:
            continue
        contour = max(contours, key=len)            # (row, col) pairs
        # Convert pixel coords to tile-coord (then UTM).
        ring_px = [(c + tx0 * _TILE, r + ty0 * _TILE) for r, c in contour]
        ring_utm = [grid.px_to_utm(px, py) for px, py in ring_px]
        if len(ring_utm) < 4:
            continue
        # Optional Visvalingam simplification -- fewer points, same shape.
        if simplify_tol_m > 0:
            try:
                from simplification.cutil import simplify_coords_vw
                # ring_utm is in metres; VW epsilon is m²
                arr = simplify_coords_vw(list(ring_utm), simplify_tol_m ** 2)
                if len(arr) >= 4:
                    ring_utm = [(float(x), float(y)) for x, y in arr]
            except ImportError:
                pass
        poly = Polygon(ring_utm)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if (poly.geom_type == "Polygon"
                and min_area_m2 <= poly.area <= max_area_m2):
            polys.append(poly)
    _log.info("skimage vectorise: %d parcel polygons", len(polys))
    return polys


def _approx_m2_per_px(grid, tx0, ty0) -> float:
    import math
    a = grid.px_to_utm(tx0 * _TILE, ty0 * _TILE)
    bx = grid.px_to_utm(tx0 * _TILE + 1, ty0 * _TILE)
    by = grid.px_to_utm(tx0 * _TILE, ty0 * _TILE + 1)
    return max(math.dist(a, bx) * math.dist(a, by), 1e-6)
