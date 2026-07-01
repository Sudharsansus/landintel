"""LandXML loader -- client survey XML -> cadastral parcels + survey points.

LandXML (landxml.org schema, used by every survey/CAD package) is the richest client input:
it carries EXACT coordinates, so it bypasses OCR entirely. We pull two things from it:
  * PARCELS  -> authoritative polygons keyed by survey number (a CadastralSource, like the
               TNGIS vector export) for M2 placement + the M1 label-noise ground truth.
  * POINTS   -> the surveyed CgPoints (name/code + UTM) usable as direct georef anchors / seeds.

Parser notes (kept robust + dependency-free, stdlib xml.etree only):
  * NAMESPACE-AGNOSTIC: matches by local tag name, so LandXML 1.0/1.1/1.2/2.0 all parse.
  * LandXML point text is "NORTHING EASTING [elev]" (survey convention) -> we emit (E, N) so
    x=easting, y=northing for a UTM CRS.
  * Parcel boundary is walked from CoordGeom (Line/Curve Start/End, inline coords or pntRef to
    CgPoints); curves are chord-approximated. CRS from <CoordinateSystem epsgCode|...Name>.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from xml.etree import ElementTree as ET

from shapely.geometry import MultiPolygon, Polygon

_log = logging.getLogger(__name__)
_SURVEY_RE = re.compile(r"(\d{1,5})")


def _local(tag: str) -> str:
    """Strip any XML namespace: '{ns}Parcel' -> 'parcel' (lowercased)."""
    return tag.rsplit("}", 1)[-1].lower()


def _find_all(elem, name: str):
    return [e for e in elem.iter() if _local(e.tag) == name.lower()]


def _coords_from_text(text: str | None) -> tuple[float, float] | None:
    """LandXML point text 'northing easting [elev]' -> (easting, northing)."""
    if not text:
        return None
    parts = text.replace(",", " ").split()
    if len(parts) < 2:
        return None
    try:
        north, east = float(parts[0]), float(parts[1])
        return (east, north)                       # (x=E, y=N)
    except ValueError:
        return None


def _crs_of(root, fallback: str | None) -> str | None:
    for cs in _find_all(root, "coordinatesystem"):
        epsg = cs.get("epsgCode") or cs.get("epsgcode")
        if epsg and epsg.isdigit():
            return f"EPSG:{epsg}"
        name = (cs.get("horizontalCoordinateSystemName") or cs.get("desc")
                or cs.get("name") or "")
        m = re.search(r"EPSG[:/]+(\d+)", name)
        if m:
            return f"EPSG:{m.group(1)}"
    return fallback


def _point_map(root) -> dict[str, tuple[float, float]]:
    """name -> (E, N) for every CgPoint."""
    pts: dict[str, tuple[float, float]] = {}
    for cg in _find_all(root, "cgpoint"):
        name = cg.get("name") or cg.get("oID") or cg.get("desc")
        xy = _coords_from_text(cg.text)
        if name and xy:
            pts[str(name)] = xy
    return pts


def _ring_of_parcel(parcel, points: dict[str, tuple[float, float]]) -> list[tuple[float, float]]:
    """Walk a Parcel's CoordGeom into an ordered boundary ring (E, N)."""
    ring: list[tuple[float, float]] = []

    def push(xy):
        if xy and (not ring or (abs(xy[0] - ring[-1][0]) + abs(xy[1] - ring[-1][1]) > 1e-6)):
            ring.append(xy)

    for geom in _find_all(parcel, "coordgeom"):
        for seg in list(geom):
            if _local(seg.tag) not in ("line", "curve", "irregularline"):
                continue
            for endpoint in seg:
                tag = _local(endpoint.tag)
                if tag not in ("start", "end", "center", "pntlist2d", "pntlist3d"):
                    continue
                ref = endpoint.get("pntRef") or endpoint.get("pntref")
                if ref and ref in points:
                    push(points[ref])
                else:
                    push(_coords_from_text(endpoint.text))
    return ring


def parse_landxml(path: str | Path, src_crs: str | None = None):
    """Parse a LandXML file. Returns (parcels, points, crs):
      parcels : list[(survey_number, village_or_None, shapely_Polygon_in_src_crs)]
      points  : dict[name -> (easting, northing)]
      crs     : detected EPSG (or src_crs/None)
    Raises ValueError if the file is not recognisable LandXML."""
    root = ET.parse(str(path)).getroot()
    if _local(root.tag) != "landxml" and not _find_all(root, "cgpoint") \
            and not _find_all(root, "parcel"):
        raise ValueError(f"{Path(path).name}: not recognisable LandXML "
                         "(no <LandXML>/<CgPoint>/<Parcel>). Send a sample to add its schema.")
    crs = _crs_of(root, src_crs)
    points = _point_map(root)

    parcels = []
    for parcel in _find_all(root, "parcel"):
        name = parcel.get("name") or parcel.get("desc") or parcel.get("parcelType")
        m = _SURVEY_RE.search(str(name)) if name else None
        sn = m.group(1) if m else None
        if not sn:
            continue
        village = parcel.get("class") or parcel.get("parcelType")
        ring = _ring_of_parcel(parcel, points)
        if len(ring) >= 3:
            geom = Polygon(ring)
            if not geom.is_valid:
                geom = geom.buffer(0)              # may yield a MultiPolygon -> flatten
            parts = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
            for poly in parts:
                if isinstance(poly, Polygon) and poly.area > 0:
                    parcels.append((sn, village, poly))
    _log.info("LandXML %s: %d parcels, %d points (CRS %s)",
              Path(path).name, len(parcels), len(points), crs)
    return parcels, points, crs


def load_landxml_for_loaders(path, survey_field, village_field, src_crs):
    """Adapter to plug into m5_cadastral.source._LOADERS (parcels only)."""
    parcels, _points, crs = parse_landxml(path, src_crs)
    return parcels, crs


def landxml_points(path: str | Path) -> dict[str, tuple[float, float]]:
    """Just the surveyed CgPoints (name -> easting,northing) -- direct georef anchors/seeds."""
    return parse_landxml(path)[1]
