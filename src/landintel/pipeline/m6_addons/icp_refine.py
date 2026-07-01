"""ICP (Iterative Closest Point) refinement for ACCEPT_CADASTRAL placements.

Instead of a GLOBAL (dx, dy) shift (which assumes the entire cadastral tile
area has uniform registration error and can mask real mislocations), this does
a PER-PLOT local ICP alignment of the placed corners to nearby surveyor stones.

Solves Bug #5 properly:
- Each plot gets its own (R, t) refinement.
- ICP is robust to partial overlap (some corners may not have a nearby stone).
- If a placed plot has no nearby surveyor stones, ICP returns None and the
  cadastral placement is unchanged.
"""
from __future__ import annotations
import logging
import numpy as np

_log = logging.getLogger(__name__)


def icp_refine_placement(
    placed_corners: np.ndarray,    # (N, 2) placed corners in UTM
    surveyor_stones_xy: np.ndarray, # (M, 2) surveyor stone UTM positions
    init_R: np.ndarray | None = None,
    init_t: np.ndarray | None = None,
    search_radius_m: float = 8.0,
    min_correspondences: int = 3,
    max_correspondence_m: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Refine a placement by aligning its corners to nearby surveyor stones.

    Returns (R, t, fitness) where fitness is fraction of corners that found
    a nearby stone, OR None if refinement should be skipped.
    """
    if len(placed_corners) < min_correspondences:
        return None
    if len(surveyor_stones_xy) < min_correspondences:
        return None

    try:
        from scipy.spatial import cKDTree
        import open3d as o3d
    except ImportError as e:
        _log.warning("icp_refine needs scipy + open3d: %s", e)
        return None

    tree = cKDTree(surveyor_stones_xy)
    # Find stones within search_radius_m of any placed corner.
    nearby_idx = set()
    for c in placed_corners:
        d, idx = tree.query(c, k=5, distance_upper_bound=search_radius_m)
        for di, ii in zip(d, idx):
            if di <= search_radius_m:
                nearby_idx.add(int(ii))
    if len(nearby_idx) < min_correspondences:
        return None

    tgt_pts = surveyor_stones_xy[sorted(nearby_idx)]
    src_pts = np.asarray(placed_corners, dtype=float)

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(np.column_stack([src_pts, np.zeros(len(src_pts))]))
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(np.column_stack([tgt_pts, np.zeros(len(tgt_pts))]))

    if init_R is None:
        init_R = np.eye(3)
    if init_t is None:
        init_t = np.zeros(3)
    init_T = np.block([[init_R, init_t.reshape(3, 1)], [np.zeros((1, 3)), np.ones((1, 1))]])

    try:
        result = o3d.pipelines.registration.registration_icp(
            src_pcd, tgt_pcd, max_correspondence_m, init_T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    except Exception as e:
        _log.debug("ICP failed: %s", e)
        return None

    R2 = result.transformation[:2, :2]
    t2 = result.transformation[:2, 3]
    return R2, t2, float(result.fitness)
