"""M6 addons: integrate GDAL-stack tools for higher accuracy on tough cases.

Each helper is an optional drop-in for the existing pipeline. None of them
violate the 0 FP / 0 FN invariant -- they improve FIT QUALITY, not gate WEAKNESS.

Public API:
    from landintel.pipeline.m6_addons import (
        build_village_fence_alphashape,   # alphashape of S3 cadastral lines
        build_village_fence_pyclipper,   # pyclipper offset variant
        icp_refine_placement,            # per-plot ICP against surveyor stones
        skeletonize_parcels_skimage,     # skimage skeletonise + regionprops
        simplify_parcel_vw,              # Visvalingam-Whyatt simplification
        bulk_overlap_check_geopandas,    # geopandas.overlay for O(N log N) overlap
    )
"""

from .village_fence import (
    build_village_fence_alphashape,
    build_village_fence_pyclipper,
)
from .icp_refine import icp_refine_placement
from .vectorize_skimage import skeletonize_parcels_skimage
from .simplify import simplify_parcel_vw
from .overlap_check import bulk_overlap_check_geopandas

__all__ = [
    "build_village_fence_alphashape",
    "build_village_fence_pyclipper",
    "icp_refine_placement",
    "skeletonize_parcels_skimage",
    "simplify_parcel_vw",
    "bulk_overlap_check_geopandas",
]
