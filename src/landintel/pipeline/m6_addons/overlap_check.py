"""Bulk overlap check using geopandas.overlay (O(N log N) vs O(N^2) loop)."""
from __future__ import annotations
import logging

_log = logging.getLogger(__name__)


def bulk_overlap_check_geopandas(placed_gdf, review_gdf, max_overlap_frac: float = 0.20):
    """Detect which review plots overlap any placed plot above max_overlap_frac.

    Parameters
    ----------
    placed_gdf : geopandas.GeoDataFrame
        One row per ACCEPT/ACCEPT_SEEDED/ACCEPT_CADASTRAL plot.
    review_gdf : geopandas.GeoDataFrame
        One row per REVIEW plot to check.
    max_overlap_frac : float
        Demote a review plot if its interior overlap with ANY placed plot
        is more than this fraction of the smaller footprint.

    Returns
    -------
    set[str] -- survey numbers of review plots that overlap too much.
    """
    try:
        import geopandas as gpd
    except ImportError:
        _log.warning("geopandas not installed; pip install geopandas")
        return set()

    if placed_gdf.empty or review_gdf.empty:
        return set()

    overlay = gpd.overlay(review_gdf, placed_gdf, how="intersection", keep_geom_type=True)
    if overlay.empty:
        return set()
    overlay["inter_area"] = overlay.geometry.area
    overlay["smaller_area"] = overlay[["__review_area", "__placed_area"]].min(axis=1)
    overlay["overlap_frac"] = overlay["inter_area"] / overlay["smaller_area"].clip(lower=1e-9)
    bad = overlay[overlay["overlap_frac"] > max_overlap_frac]
    return set(bad["survey_number"].unique())
