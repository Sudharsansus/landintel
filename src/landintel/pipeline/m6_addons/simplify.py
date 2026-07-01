"""Visvalingam-Whyatt polygon simplification (alternative to Douglas-Peucker)."""
from __future__ import annotations
import logging
import math

_log = logging.getLogger(__name__)


def simplify_parcel_vw(coords, threshold_m: float = 1.5, lat_for_scale: float = 11.2):
    """Visvalingam-Whyatt simplification of a polygon ring.

    Parameters
    ----------
    coords : list[(x, y)]
        Polygon coordinates (in degrees for WGS84 or UTM metres).
    threshold_m : float
        Side length threshold in metres. VW removes points whose removal
        loses less than (threshold_m)^2 in m² area.
    lat_for_scale : float
        Latitude for converting metres -> degrees (only used if coords look
        like WGS84 lat/lon).

    Returns
    -------
    list[(x, y)] -- simplified ring.
    """
    try:
        from simplification.cutil import simplify_coords_vw
    except ImportError:
        _log.warning("simplification not installed; pip install simplification")
        return list(coords)

    # Detect CRS: if values are in [-180, 180], it's lat/lon -> convert threshold.
    # VW's epsilon is the MINIMUM TRIANGLE AREA in coordinate-square units.
    # For a 1.5m simplification in lat/lon, the equivalent area is 1.5² m² ->
    #     deg²_per_m² = 1 / (111000² * cos(lat))
    if coords and max(abs(c[0]) for c in coords) <= 180 and max(abs(c[1]) for c in coords) <= 90:
        deg2_per_m2 = 1.0 / (111000.0**2 * math.cos(math.radians(lat_for_scale)))
        threshold = (threshold_m ** 2) * deg2_per_m2
    else:
        threshold = threshold_m ** 2  # UTM metres -> m² directly

    arr = simplify_coords_vw(list(coords), threshold)
    out = [(float(x), float(y)) for x, y in arr]
    if len(out) < 3:
        return list(coords)
    # Re-close the ring if the input was closed.
    if coords and coords[0] == coords[-1] and out[-1] != out[0]:
        out.append(out[0])
    return out
