"""M5 cadastral reference: ingest authoritative parcel polygons keyed by survey no.

This is the layer that makes an EXTERNAL cadastral vector source useful to M2.
M2 georeferences FMB plots against the surveyor's corridor (the only field UTM
reference), which covers only the plots the corridor crosses. A cadastral vector
layer -- the government's georeferenced parcel boundaries WITH survey numbers --
covers the WHOLE village, so every FMB can be placed onto its authoritative parcel
by survey number, including the off-corridor plots the corridor survey never traced.

SOURCE-AGNOSTIC by design. A ``CadastralSource`` answers "give me the UTM polygon
for survey number N (in village V)". Implementations:
  - ``VectorFileCadastralSource`` -- a file the client provides. Reads GeoJSON and
    KML/KMZ with the Python STANDARD LIBRARY only (json / zipfile / xml.etree), and
    ESRI Shapefile if ``pyshp`` is installed. Reprojects to UTM 44N via pyproj.
  - ``TngisCadastralSource`` -- the auto-fetch-by-village slot. INERT: the TN GIS
    server (ArcGIS at 117.239.110.245, proxied by tngis.tn.gov.in) exposes no
    reachable public vector endpoint (REST/WFS 404, the IP is firewalled, only
    authenticated WMS image tiles are served), so it cannot fetch vectors today.
    Kept as a ready adapter to light up IF an authorised endpoint is provided.

Survey numbers are normalised to their base integer string ("82/1" -> "82") so a
cadastral key matches an FMB ``survey_no`` regardless of subdivision suffix.
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from shapely.geometry import shape as shapely_shape
from shapely.geometry import Polygon, MultiPolygon

_log = logging.getLogger(__name__)

# Default target CRS = the surveyor's UTM zone. INGUR/Erode (~77.6E) is Zone 43N.
TARGET_CRS = "EPSG:32643"
_WGS84 = "EPSG:4326"

# Normalise "82", "82/1", "82/2A", "82A" -> base integer "82".
_SURVEY_RE = re.compile(r"(\d{1,5})")

# Property/field names that commonly hold the survey number in TN cadastral data.
_SURVEY_FIELD_ALIASES = (
    "survey_no", "surveyno", "survey_number", "sno", "s_no", "sy_no", "syno",
    "fmb_no", "fmbno", "survey", "lgd_survey", "kide", "field_no", "fieldno",
)
_VILLAGE_FIELD_ALIASES = ("village", "village_na", "vil_name", "villname", "revenue_vi")


def _norm_survey(raw: str | None) -> str | None:
    if raw is None:
        return None
    m = _SURVEY_RE.search(str(raw))
    return m.group(1) if m else None


@dataclass
class CadastralParcel:
    """One cadastral parcel polygon in UTM Zone 44N (EPSG:32644)."""
    survey_number: str
    polygon: Polygon
    village: str | None = None
    source_crs: str = TARGET_CRS

    def exterior_coords(self) -> list[tuple[float, float]]:
        return [(float(x), float(y)) for x, y in self.polygon.exterior.coords]


class CadastralSource:
    """Interface: resolve a survey number to its UTM cadastral parcel."""

    def get(self, survey_number: str, village: str | None = None) -> CadastralParcel | None:
        raise NotImplementedError

    def __contains__(self, survey_number: str) -> bool:
        return self.get(_norm_survey(survey_number) or survey_number) is not None

    def survey_numbers(self) -> set[str]:
        return set()


def _reproject_to_utm(geom, src_crs: str, target_crs: str = TARGET_CRS):
    """Reproject a shapely geometry from src_crs to the surveyor UTM zone."""
    norm = lambda c: c.upper().replace(" ", "")
    if norm(src_crs) == norm(target_crs):
        return geom
    try:
        from pyproj import Transformer
        from shapely.ops import transform as shp_transform
    except ImportError:
        _log.warning("pyproj/shapely transform unavailable; assuming coords already UTM")
        return geom
    tr = Transformer.from_crs(src_crs, target_crs, always_xy=True)
    return shp_transform(lambda xs, ys, *a: tr.transform(xs, ys), geom)


def _polygons(geom):
    """Yield Polygon parts from a (Multi)Polygon; ignore non-areal geometry."""
    if isinstance(geom, Polygon):
        if geom.area > 0:
            yield geom
    elif isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            if p.area > 0:
                yield p


# ---------------------------------------------------------------------------
# Format loaders -> list[(survey_number, village, shapely_polygon_in_src_crs)]
# ---------------------------------------------------------------------------

def _detect_field(keys, aliases) -> str | None:
    low = {k.lower(): k for k in keys}
    for a in aliases:
        if a in low:
            return low[a]
    return None


def _load_geojson(path: Path, survey_field, village_field, src_crs):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # GeoJSON CRS member (legacy) overrides the default WGS84 if present.
    crs = src_crs
    if crs is None:
        crs = _WGS84
        cm = data.get("crs", {}).get("properties", {}).get("name") if isinstance(data.get("crs"), dict) else None
        if cm:
            m = re.search(r"EPSG[:/]+(\d+)", cm)
            if m:
                crs = f"EPSG:{m.group(1)}"
    feats = data.get("features", []) if data.get("type") == "FeatureCollection" else [data]
    out = []
    sf = survey_field
    vf = village_field
    for f in feats:
        props = f.get("properties", {}) or {}
        if sf is None:
            sf = _detect_field(props.keys(), _SURVEY_FIELD_ALIASES)
        if vf is None:
            vf = _detect_field(props.keys(), _VILLAGE_FIELD_ALIASES)
        sn = _norm_survey(props.get(sf)) if sf else None
        vil = props.get(vf) if vf else None
        geom = f.get("geometry")
        if not sn or not geom:
            continue
        for poly in _polygons(shapely_shape(geom)):
            out.append((sn, vil, poly))
    return out, crs


_KML_NS = "{http://www.opengis.net/kml/2.2}"


def _kml_root_from(path: Path):
    p = Path(path)
    if p.suffix.lower() == ".kmz":
        with zipfile.ZipFile(p) as z:
            name = next((n for n in z.namelist() if n.lower().endswith(".kml")), None)
            if name is None:
                raise ValueError(f"No .kml inside KMZ {p}")
            return ET.fromstring(z.read(name))
    return ET.parse(str(p)).getroot()


def _kml_text(elem, tag):
    e = elem.find(f"{_KML_NS}{tag}")
    return e.text.strip() if e is not None and e.text else None


def _parse_kml_coords(text: str) -> list[tuple[float, float]]:
    pts = []
    for tok in text.replace("\n", " ").split():
        parts = tok.split(",")
        if len(parts) >= 2:
            pts.append((float(parts[0]), float(parts[1])))  # lon, lat
    return pts


def _load_kml(path: Path, survey_field, village_field, src_crs):
    """KML/KMZ -> parcels. KML geometry is always WGS84 lon/lat per the spec."""
    root = _kml_root_from(path)
    out = []
    for pm in root.iter(f"{_KML_NS}Placemark"):
        name = _kml_text(pm, "name")
        # Survey number from a chosen ExtendedData field, else the placemark name.
        sn = None
        vil = None
        ext = {}
        for sd in pm.iter(f"{_KML_NS}SimpleData"):
            ext[(sd.get("name") or "").lower()] = (sd.text or "").strip()
        for d in pm.iter(f"{_KML_NS}Data"):
            v = d.find(f"{_KML_NS}value")
            ext[(d.get("name") or "").lower()] = (v.text or "").strip() if v is not None else ""
        sf = (survey_field or _detect_field(ext.keys(), _SURVEY_FIELD_ALIASES))
        vf = (village_field or _detect_field(ext.keys(), _VILLAGE_FIELD_ALIASES))
        if sf and sf.lower() in ext:
            sn = _norm_survey(ext[sf.lower()])
        if vf and vf.lower() in ext:
            vil = ext[vf.lower()]
        if sn is None:
            sn = _norm_survey(name)
        if sn is None:
            continue
        for poly_el in pm.iter(f"{_KML_NS}Polygon"):
            outer = poly_el.find(
                f"{_KML_NS}outerBoundaryIs/{_KML_NS}LinearRing/{_KML_NS}coordinates")
            if outer is None or not outer.text:
                continue
            ring = _parse_kml_coords(outer.text)
            if len(ring) >= 4:
                out.append((sn, vil, Polygon(ring)))
    return out, (src_crs or _WGS84)


def _load_shapefile(path: Path, survey_field, village_field, src_crs):
    try:
        import shapefile  # pyshp
    except ImportError as e:  # noqa: BLE001
        raise ImportError(
            "Reading ESRI Shapefile (.shp) needs pyshp: `pip install pyshp`. "
            "Or ask the client for GeoJSON / KMZ (read with no extra deps).") from e
    r = shapefile.Reader(str(path))
    fields = [f[0] for f in r.fields[1:]]
    sf = survey_field or _detect_field(fields, _SURVEY_FIELD_ALIASES)
    vf = village_field or _detect_field(fields, _VILLAGE_FIELD_ALIASES)
    crs = src_crs
    if crs is None:
        prj = Path(path).with_suffix(".prj")
        crs = prj.read_text(encoding="utf-8").strip() if prj.exists() else _WGS84
    out = []
    for sr in r.shapeRecords():
        rec = sr.record.as_dict()
        sn = _norm_survey(rec.get(sf)) if sf else None
        vil = rec.get(vf) if vf else None
        if not sn:
            continue
        geom = sr.shape.__geo_interface__
        for poly in _polygons(shapely_shape(geom)):
            out.append((sn, vil, poly))
    return out, crs


def _load_landxml(path, survey_field, village_field, src_crs):
    """LandXML survey export -> parcels (delegates to the dedicated landxml parser)."""
    from .landxml import load_landxml_for_loaders
    return load_landxml_for_loaders(path, survey_field, village_field, src_crs)


_CSV_X_ALIASES = ("easting", "east", "utm_e", "x", "e", "lon", "long", "longitude")
_CSV_Y_ALIASES = ("northing", "north", "utm_n", "y", "n", "lat", "latitude")
_CSV_SEQ_ALIASES = ("seq", "order", "vertex", "vno", "pno", "point_no", "idx", "sort")
_CSV_WKT_ALIASES = ("wkt", "geometry", "geom", "the_geom", "wkt_geom")


def _load_csv(path, survey_field, village_field, src_crs):
    """CSV cadastral/points -> parcels. Two shapes, auto-detected from the header:
      * a WKT column        -> one polygon per row;
      * x/y point columns   -> rows are boundary vertices grouped by survey number,
                               ordered by a sequence column if present (else file order).
    Coordinate columns are matched from common aliases (easting/northing, x/y, lon/lat).
    CRS: ``src_crs`` if given; else WGS84 when values look like lon/lat, else the target UTM."""
    import csv as _csv

    rows = list(_csv.DictReader(Path(path).open(encoding="utf-8-sig")))
    if not rows:
        return [], (src_crs or TARGET_CRS)
    keys = list(rows[0].keys())
    sf = survey_field or _detect_field(keys, _SURVEY_FIELD_ALIASES)
    vf = village_field or _detect_field(keys, _VILLAGE_FIELD_ALIASES)
    wktf = _detect_field(keys, _CSV_WKT_ALIASES)
    xf = _detect_field(keys, _CSV_X_ALIASES)
    yf = _detect_field(keys, _CSV_Y_ALIASES)
    seqf = _detect_field(keys, _CSV_SEQ_ALIASES)
    if sf is None:
        raise ValueError(f"CSV {Path(path).name}: no survey-number column "
                         f"(looked for {_SURVEY_FIELD_ALIASES[:4]}…). Columns: {keys}")

    out = []
    crs = src_crs
    if wktf:                                              # ---- WKT-per-row ----
        from shapely.wkt import loads as _wkt
        for r in rows:
            sn = _norm_survey(r.get(sf))
            if not sn or not r.get(wktf):
                continue
            try:
                for poly in _polygons(_wkt(r[wktf])):
                    out.append((sn, r.get(vf) if vf else None, poly))
            except Exception:  # noqa: BLE001
                continue
        return out, (crs or TARGET_CRS)

    if not (xf and yf):
        raise ValueError(f"CSV {Path(path).name}: need a WKT column or x/y point columns "
                         f"(easting/northing | x/y | lon/lat). Columns: {keys}")
    # ---- points grouped by survey number into a boundary ring ----
    groups: dict[str, list] = {}
    sample_xy = None
    for i, r in enumerate(rows):
        sn = _norm_survey(r.get(sf))
        try:
            x, y = float(r[xf]), float(r[yf])
        except (TypeError, ValueError):
            continue
        if sn is None:
            continue
        order = None
        if seqf and r.get(seqf):
            try:
                order = float(r[seqf])
            except ValueError:
                order = None
        groups.setdefault(sn, []).append((order if order is not None else i, x, y,
                                          r.get(vf) if vf else None))
        sample_xy = sample_xy or (x, y)
    if crs is None:                                       # lon/lat heuristic
        crs = _WGS84 if (sample_xy and abs(sample_xy[0]) <= 180 and abs(sample_xy[1]) <= 90) \
            else TARGET_CRS
    for sn, pts in groups.items():
        pts.sort(key=lambda t: t[0])
        ring = [(x, y) for _o, x, y, _v in pts]
        vil = next((v for *_r, v in pts if v), None)
        if len(ring) >= 3:
            geom = Polygon(ring)
            if not geom.is_valid:
                geom = geom.buffer(0)
            for poly in _polygons(geom):           # flatten Multi->single, area>0 (like other loaders)
                out.append((sn, vil, poly))
    return out, crs


_LOADERS = {
    ".geojson": _load_geojson, ".json": _load_geojson,
    ".kml": _load_kml, ".kmz": _load_kml,
    ".shp": _load_shapefile,
    ".xml": _load_landxml, ".landxml": _load_landxml,
    ".csv": _load_csv,
}


class VectorFileCadastralSource(CadastralSource):
    """Cadastral parcels loaded from a client-provided vector file.

    Supported: ``.geojson`` / ``.json``, ``.kml``, ``.kmz`` (stdlib only) and
    ``.shp`` (needs pyshp). All geometry is reprojected to UTM Zone 44N.

    survey_field / village_field : attribute names holding the survey number /
        village. Auto-detected from common aliases when omitted.
    source_crs : override the input CRS (e.g. "EPSG:32644"). Auto: KML=WGS84,
        GeoJSON=its crs member or WGS84, Shapefile=its .prj.
    """

    def __init__(self, path: str | Path, survey_field: str | None = None,
                 village_field: str | None = None, source_crs: str | None = None,
                 target_crs: str = TARGET_CRS):
        self.path = Path(path)
        ext = self.path.suffix.lower()
        loader = _LOADERS.get(ext)
        if loader is None:
            raise ValueError(
                f"Unsupported cadastral format {ext!r}. Use GeoJSON, KML, KMZ, or SHP.")
        raw, crs = loader(self.path, survey_field, village_field, source_crs)
        self._by_survey: dict[str, CadastralParcel] = {}
        # Retain EVERY ring per survey (largest-first) so the multi-part / merged /
        # buffer(0)-split rings that `get()` discards are available to
        # `recovered_candidates`. The pipeline runs each through its rigid shape gate,
        # so retaining them adds recall without adding false positives.
        self._rings_by_survey: dict[str, list[CadastralParcel]] = {}
        for sn, vil, poly in raw:
            utm = _reproject_to_utm(poly, crs, target_crs)
            if not utm.is_valid:
                utm = utm.buffer(0)
            for ring in _polygons(utm):                       # flatten Multi -> single
                parcel = CadastralParcel(
                    survey_number=sn, polygon=ring, village=vil, source_crs=target_crs)
                self._rings_by_survey.setdefault(sn, []).append(parcel)
        for sn, parcels in self._rings_by_survey.items():
            parcels.sort(key=lambda p: p.polygon.area, reverse=True)
            self._by_survey[sn] = parcels[0]                  # primary = largest ring
        n_multi = sum(1 for v in self._rings_by_survey.values() if len(v) > 1)
        _log.info("Cadastral source %s: %d parcels (%d multi-ring) (CRS %s -> %s)",
                  self.path.name, len(self._by_survey), n_multi, crs, target_crs)

    def get(self, survey_number: str, village: str | None = None) -> CadastralParcel | None:
        return self._by_survey.get(_norm_survey(survey_number) or survey_number)

    def recovered_candidates(self, survey_number: str) -> list[CadastralParcel]:
        """Alternative closed rings for a survey whose primary ring failed the gate.

        Mirrors ``S3CadastralSource.recovered_candidates``: when a survey number maps
        to several rings (a multi-part parcel, a merged super-parcel split by
        ``buffer(0)``, or duplicate features), ``get()`` returns only the largest.
        This returns the REMAINING rings (largest-first) so the pipeline can try each
        through its rigid shape gate and keep the first that fits. Empty for a clean
        single-ring parcel (``get`` already gave the one right ring), so this can only
        ADD recall -- every candidate still has to pass the same gate, never an FP.
        """
        sn = _norm_survey(survey_number) or survey_number
        rings = self._rings_by_survey.get(sn, [])
        return rings[1:] if len(rings) > 1 else []

    def label_point(self, survey_number: str) -> tuple[float, float] | None:
        """Parcel centroid in UTM -- the independent expected seat that activates M2's
        seat-locality false-positive guard (expected_xy) for this vector source too."""
        p = self.get(survey_number)
        if p is None:
            return None
        c = p.polygon.centroid
        return (float(c.x), float(c.y))

    def survey_numbers(self) -> set[str]:
        return set(self._by_survey)


class TngisCadastralSource(CadastralSource):
    """Auto-fetch-by-village slot for the TN GIS portal -- currently INERT.

    Verified (2026-06-26): no reachable public vector endpoint exists. The portal
    (tngis.tn.gov.in) is ArcGIS-backed but `/geoserver` and `/arcgis/rest/services`
    return 404, and the actual ArcGIS host (117.239.110.245) is firewalled
    (all requests time out); only authenticated WMS image tiles are proxied. So
    this raises with guidance rather than silently returning nothing. Provide a
    real endpoint + credentials to activate, or use ``VectorFileCadastralSource``.
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = base_url
        self.api_key = api_key

    def get(self, survey_number: str, village: str | None = None) -> CadastralParcel | None:
        raise RuntimeError(
            "TNGIS auto-fetch is not available: no reachable public vector endpoint "
            "(REST/WFS 404; ArcGIS host firewalled; only authenticated WMS images). "
            "Use VectorFileCadastralSource with a client-exported GeoJSON/KMZ/Shapefile.")


def load_cadastral(path: str | Path, **kw) -> VectorFileCadastralSource:
    """Convenience: open any supported cadastral file as a source."""
    return VectorFileCadastralSource(path, **kw)
