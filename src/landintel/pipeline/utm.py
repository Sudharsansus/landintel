"""Automatic UTM-zone / CRS detection -- so the engine adapts to ANY land, not just INGUR.

Tamil Nadu straddles two UTM zones (43N west of 78E, 44N east), and India spans 42-47N.
Hardcoding EPSG:32643 silently corrupts georeferencing anywhere east of 78E. These helpers
pick the correct northern-hemisphere UTM CRS from a longitude or from a WGS84 vector/point, so
``--crs auto`` Just Works across the whole state (and country).
"""

from __future__ import annotations

import logging
from pathlib import Path

_log = logging.getLogger(__name__)


def utm_crs_for_lon(lon: float, northern: bool = True) -> str:
    """EPSG code of the UTM zone containing ``lon`` (degrees). 32601-32660 N, 32701-32760 S.
    e.g. 77.6E -> EPSG:32643 (43N), 78.5E -> EPSG:32644 (44N)."""
    zone = int((lon + 180.0) / 6.0) + 1
    zone = min(max(zone, 1), 60)
    return f"EPSG:32{'6' if northern else '7'}{zone:02d}"


def utm_crs_for_wgs84_point(lon: float, lat: float) -> str:
    return utm_crs_for_lon(lon, northern=(lat >= 0))


def detect_crs_from_cadastral(path: str | Path) -> str | None:
    """Sniff a WGS84 vector (GeoJSON/KML/KMZ) for its longitude and return the right UTM CRS.

    Only meaningful when the source is in lon/lat; a file already in UTM gives no zone hint
    (returns None -> caller keeps the explicit/default CRS). Best-effort, never raises."""
    p = Path(path)
    try:
        ext = p.suffix.lower()
        lon = lat = None
        if ext in (".geojson", ".json"):
            import json
            data = json.loads(p.read_text(encoding="utf-8"))
            feats = data.get("features", [data]) if isinstance(data, dict) else []
            for f in feats:
                c = _first_coord(f.get("geometry") if isinstance(f, dict) else None)
                if c:
                    lon, lat = c
                    break
        elif ext in (".kml", ".kmz", ".xml"):
            from .m5_cadastral.source import _kml_root_from, _KML_NS, _parse_kml_coords
            if ext == ".xml":
                return None  # LandXML is projected coords, not lon/lat -> no sniff
            root = _kml_root_from(p)
            for coords in root.iter(f"{_KML_NS}coordinates"):
                pts = _parse_kml_coords(coords.text or "")
                if pts:
                    lon, lat = pts[0]
                    break
        if lon is None or not (-180 <= lon <= 180 and -90 <= lat <= 90):
            return None
        crs = utm_crs_for_wgs84_point(lon, lat)
        _log.info("auto-CRS: cadastral at lon=%.3f -> %s", lon, crs)
        return crs
    except Exception as exc:  # noqa: BLE001
        _log.warning("auto-CRS sniff failed for %s: %s", p.name, exc)
        return None


def _first_coord(geom) -> tuple[float, float] | None:
    """First (lon, lat) of any GeoJSON geometry."""
    if not isinstance(geom, dict):
        return None
    c = geom.get("coordinates")
    while isinstance(c, (list, tuple)) and c and isinstance(c[0], (list, tuple)):
        c = c[0]
    if isinstance(c, (list, tuple)) and len(c) >= 2:
        return float(c[0]), float(c[1])
    return None
