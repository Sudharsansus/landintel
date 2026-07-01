"""Boundary vertex warping -- proportional interpolation.

After the corner stones are adjusted by the cadastral adjustment, the
boundary polylines may have intermediate vertices (where two edges meet
at a non-stone point, or subdivision line intersections). These
intermediate vertices need to follow the adjusted stone positions smoothly.

Strategy: For each intermediate vertex on a boundary edge (stone_a -> stone_b),
compute its fractional position along the original M1 edge, then place it
at the same fraction along the adjusted edge.

For subdivision lines, chain lines, and other geometry, each vertex is
interpolated between its two nearest adjusted stones proportionally.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import cKDTree

_log = logging.getLogger(__name__)


def warp_point(
    pt: tuple[float, float] | np.ndarray,
    stone_a: tuple[float, float] | np.ndarray,
    stone_b: tuple[float, float] | np.ndarray,
    adj_a: tuple[float, float] | np.ndarray,
    adj_b: tuple[float, float] | np.ndarray,
) -> np.ndarray:
    """Warp a single point that lies on the line segment stone_a -> stone_b.

    Computes the fractional position of pt between stone_a and stone_b,
    then places it at the same fraction between adj_a and adj_b.

    Works even when pt is not exactly on the segment -- projects to nearest
    point on the segment first.
    """
    pt = np.asarray(pt, dtype=float)
    sa = np.asarray(stone_a, dtype=float)
    sb = np.asarray(stone_b, dtype=float)
    aa = np.asarray(adj_a, dtype=float)
    ab = np.asarray(adj_b, dtype=float)

    edge = sb - sa
    edge_len_sq = edge @ edge

    if edge_len_sq < 1e-10:
        return 0.5 * (aa + ab)

    t = ((pt - sa) @ edge) / edge_len_sq
    t = max(0.0, min(1.0, t))
    return aa + t * (ab - aa)


def warp_boundary_vertices(
    vertices: list[tuple[float, float]],
    stone_indices: list[int],
    original_positions: np.ndarray,
    adjusted_positions: np.ndarray,
) -> list[np.ndarray]:
    """Warp a list of boundary vertices using proportional interpolation.

    Each vertex is warped relative to its nearest stone neighbors.

    Parameters
    ----------
    vertices : list of (x, y) from the original M1 boundary polyline
    stone_indices : list of stone indices that these vertices belong to
                    (one per vertex, or -1 for non-stone vertices)
    original_positions : (N, 2) original M1 stone positions
    adjusted_positions : (N, 2) adjusted UTM stone positions

    Returns
    -------
    List of warped (x, y) as numpy arrays.
    """
    if not vertices:
        return []

    # Global stone tree for the fallback: a boundary segment whose BOTH endpoints
    # are intermediate (non-stone) vertices has no stone neighbour within its own
    # 2-point polyline. Such a vertex must still be carried into UTM via the
    # nearest stones' displacement -- never left at raw M1 coordinates (that was
    # the origin-blowup: BOUNDARY vertices stranded at X~270 while the rest of the
    # plot sat correctly at X~783000).
    tree = (cKDTree(original_positions)
            if original_positions is not None and len(original_positions) else None)

    def _global_offset(px: float, py: float) -> np.ndarray:
        if tree is None:
            return np.array([px, py])
        k = min(2, len(original_positions))
        dists, idxs = tree.query([px, py], k=k)
        idxs = np.atleast_1d(idxs)
        dists = np.atleast_1d(dists)
        w = 1.0 / np.maximum(dists, 1e-6)
        w /= w.sum()
        offset = np.zeros(2)
        for wi, ii in zip(w, idxs):
            offset += wi * (adjusted_positions[ii] - original_positions[ii])
        return np.array([px, py]) + offset

    warped = []
    n_verts = len(vertices)

    for i, (vx, vy) in enumerate(vertices):
        stone_idx = stone_indices[i] if i < len(stone_indices) else -1

        if stone_idx >= 0:
            warped.append(adjusted_positions[stone_idx].copy())
        else:
            prev_stone = None
            next_stone = None

            for j in range(i - 1, -1, -1):
                if j < len(stone_indices) and stone_indices[j] >= 0:
                    prev_stone = stone_indices[j]
                    break
            for j in range(i + 1, n_verts):
                if j < len(stone_indices) and stone_indices[j] >= 0:
                    next_stone = stone_indices[j]
                    break

            if prev_stone is not None and next_stone is not None:
                warped.append(warp_point(
                    (vx, vy),
                    original_positions[prev_stone],
                    original_positions[next_stone],
                    adjusted_positions[prev_stone],
                    adjusted_positions[next_stone],
                ))
            elif prev_stone is not None:
                offset = np.array([vx, vy]) - original_positions[prev_stone]
                warped.append(adjusted_positions[prev_stone] + offset)
            elif next_stone is not None:
                offset = np.array([vx, vy]) - original_positions[next_stone]
                warped.append(adjusted_positions[next_stone] + offset)
            else:
                warped.append(_global_offset(vx, vy))

    return warped


def warp_generic_vertices(
    vertices: list[tuple[float, float]],
    original_stone_positions: np.ndarray,
    adjusted_stone_positions: np.ndarray,
    k_neighbors: int = 2,
) -> list[np.ndarray]:
    """Warp arbitrary vertices (subdivision, chain, etc.) using nearest stones.

    For each vertex, find the k nearest stones, compute weighted average
    of their offsets, and apply.

    Parameters
    ----------
    vertices : list of (x, y) points
    original_stone_positions : (N, 2) original M1 stone positions
    adjusted_stone_positions : (N, 2) adjusted UTM stone positions
    k_neighbors : number of nearest stones to use for interpolation

    Returns
    -------
    List of warped (x, y) as numpy arrays.
    """
    if not vertices or original_stone_positions.shape[0] == 0:
        return [np.array(v) for v in vertices]

    tree = cKDTree(original_stone_positions)
    pts_arr = np.array(vertices)

    warped = []
    for pt in pts_arr:
        dists, idxs = tree.query(pt, k=min(k_neighbors, len(original_stone_positions)))
        if k_neighbors == 1 or np.isscalar(idxs):
            idxs = np.atleast_1d(idxs)
            dists = np.atleast_1d(dists)

        # Inverse-distance weighting
        weights = 1.0 / np.maximum(dists, 1e-10)
        weights /= weights.sum()

        offset = np.zeros(2)
        for w, idx in zip(weights, idxs):
            offset += w * (adjusted_stone_positions[idx] - original_stone_positions[idx])

        warped.append(np.array(pt) + offset)

    return warped
