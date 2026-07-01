"""M3 assemble -- turn a span's CONFIDENT placed plots into a polished village deliverable.

M2 already CLUBS every FMB into one combined DXF (placed plots at their UTM seat, the rest
staged). M3 adds the cadastral POLISH on top, NON-DESTRUCTIVELY (geometry from the rigid M2
placement is never warped):
  - area.py     -- the real ground area of each placed plot (from its verified boundary).
  - annotate.py -- write "survey# + area(ha)" labels at each parcel centroid.
(`merge.py`/`snap.py` reserved: any shared-boundary work must stay annotation/identification
only -- snapping that moves boundaries would violate the no-warp rule, so it is intentionally
not implemented as geometry mutation.)
"""

from __future__ import annotations

from .annotate import ANNOTATION_LAYER, annotate_combined
from .area import plot_area_hectares, plot_area_m2, plot_footprint

__all__ = [
    "annotate_combined", "ANNOTATION_LAYER",
    "plot_area_m2", "plot_area_hectares", "plot_footprint",
]
