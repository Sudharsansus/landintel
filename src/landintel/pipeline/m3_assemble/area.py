"""M3 area -- the real-world georeferenced area of an assembled (placed) plot.

A cadastral deliverable must carry each parcel's AREA. After M2 places a plot rigidly in UTM
metres, its enclosed BOUNDARY polygon gives the true ground area directly (no scale factor --
the geometry is already real-world). Reuses the M2 footprint extractor so area is computed from
the SAME boundary ring the pipeline verified, not a re-derivation.
"""

from __future__ import annotations

from pathlib import Path


def plot_footprint(georef_dxf_path: str | Path):
    """Largest enclosed BOUNDARY polygon of a georeferenced DXF (UTM), or None."""
    from ..m2_georef.pipeline import _footprint_polygon
    fp = _footprint_polygon(str(georef_dxf_path))
    return fp if (fp is not None and fp.is_valid and fp.area > 0) else None


def plot_area_m2(georef_dxf_path: str | Path) -> float | None:
    """Ground area of a placed plot in SQUARE METRES (None if no closed boundary)."""
    fp = plot_footprint(georef_dxf_path)
    return float(fp.area) if fp is not None else None


def plot_area_hectares(georef_dxf_path: str | Path) -> float | None:
    a = plot_area_m2(georef_dxf_path)
    return None if a is None else a / 10000.0
