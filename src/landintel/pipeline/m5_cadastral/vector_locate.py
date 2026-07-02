"""Locate a village's EXACT cadastral parcels in the TNGIS statewide vector cadastre.

Same disambiguation principle as the tile path (TN villages all number parcels 1..N, so a
survey number recurs in every neighbour), but on exact vector geometry instead of z18-OCR:

  1. Read parcels near the web anchor from the GeoParquet (bbox-filtered, reprojected to UTM).
  2. Group by ``lgd_village_code`` -- the cadastre's own village key.
  3. Candidate villages = those near the anchor that contain enough of the FMB survey numbers.
  4. Return a ``VectorCadastralSource`` per candidate; run_m2_cad's existing eval loop picks the
     one whose FMB SHAPES actually fit (mean IoU), so the village code is never hardcoded and the
     0-FP shape gate still decides. General: no per-village constants.
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

from shapely import wkb as _wkb
from shapely.geometry import Polygon

from .source import TARGET_CRS, _norm_survey
from .vector_source import VectorCadastralSource

_log = logging.getLogger(__name__)

# default location of the downloaded statewide GeoParquet (override via env)
PARQUET_PATH = os.environ.get(
    "LANDINTEL_TNGIS_PARQUET", "data/tngis/TNGIS_TN_Cadastrals.parquet")


def _bbox_overlaps(b, lon0, lat0, lon1, lat1) -> bool:
    return (b["xmin"] <= lon1 and b["xmax"] >= lon0
            and b["ymin"] <= lat1 and b["ymax"] >= lat0)


def load_area_parcels(anchor_latlon: tuple[float, float], *,
                      parquet_path: str | None = None, pad_deg: float = 0.06,
                      target_crs: str = TARGET_CRS) -> list[dict]:
    """Read parcels within ``pad_deg`` of the anchor from the GeoParquet, reprojected to UTM.

    Returns a list of ``{sn, vc, poly}`` (poly = shapely Polygon in target UTM). Uses only the
    GeoParquet ``bbox`` struct column to prefilter, so it touches only the row-group(s) covering
    the anchor. ~0.06 deg (~6.6 km) padding comfortably includes a village + its neighbours."""
    import pyarrow.parquet as pq
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform

    path = parquet_path or PARQUET_PATH
    lat, lon = float(anchor_latlon[0]), float(anchor_latlon[1])
    lon0, lon1 = lon - pad_deg, lon + pad_deg
    lat0, lat1 = lat - pad_deg, lat + pad_deg
    to_utm = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True).transform

    pf = pq.ParquetFile(path)
    out: list[dict] = []
    for rg in range(pf.num_row_groups):
        # cheap pre-check: skip a row group whose bbox range can't reach the window
        t = pf.read_row_group(rg, columns=["bbox"])
        bb = t.column("bbox").to_pylist()
        idx = [i for i, b in enumerate(bb) if b is not None
               and _bbox_overlaps(b, lon0, lat0, lon1, lat1)]
        if not idx:
            continue
        cols = pf.read_row_group(rg, columns=["survey_number", "lgd_village_code", "geometry"])
        sn = cols.column("survey_number").to_pylist()
        vc = cols.column("lgd_village_code").to_pylist()
        geo = cols.column("geometry").to_pylist()
        for i in idx:
            if geo[i] is None or sn[i] is None:
                continue
            try:
                g = shp_transform(to_utm, _wkb.loads(geo[i]))
            except Exception:  # noqa: BLE001
                continue
            if g.is_empty:
                continue
            if g.geom_type == "MultiPolygon":
                g = max(g.geoms, key=lambda p: p.area)
            if g.geom_type != "Polygon":
                continue
            out.append({"sn": str(sn[i]), "vc": (vc[i] or ""), "poly": g})
    _log.info("vector cadastre: %d parcels within %.3f deg of anchor", len(out), pad_deg)
    return out


def load_area_parcels_cached(anchor_latlon, *, cache_json: str | None = None, **kw) -> list[dict]:
    """``load_area_parcels`` with an optional on-disk hex-WKB cache (fast repeat runs)."""
    if cache_json and Path(cache_json).exists():
        raw = json.loads(Path(cache_json).read_text())
        return [{"sn": str(r["sn"]), "vc": r.get("vc", ""),
                 "poly": _wkb.loads(bytes.fromhex(r["wkb_utm"]))} for r in raw]
    parcels = load_area_parcels(anchor_latlon, **kw)
    if cache_json:
        Path(cache_json).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_json).write_text(json.dumps(
            [{"sn": p["sn"], "vc": p["vc"], "wkb_utm": p["poly"].wkb.hex()} for p in parcels]))
    return parcels


def _one_parcel_per_survey(rows: list[dict]) -> dict[str, Polygon]:
    """Within a single village, collapse to one polygon per survey number (largest area wins
    if a survey appears more than once, e.g. an un-split subdivision)."""
    best: dict[str, Polygon] = {}
    for r in rows:
        key = _norm_survey(r["sn"]) or r["sn"]
        p = r["poly"]
        if key not in best or p.area > best[key].area:
            best[key] = p
    return best


def village_candidates(parcels: list[dict], fmb_surveys: set[str], anchor_utm: tuple[float, float],
                       *, radius_m: float = 5000.0, min_overlap: int = 3, max_cand: int = 6,
                       crs: str = TARGET_CRS) -> list[dict]:
    """Rank candidate villages (by ``lgd_village_code``) near the anchor that carry the FMB
    survey numbers. Returns dicts ``{vc, source, center, n_overlap, dist_m}`` best-first
    (most survey overlap, then nearest). Shape-IoU disambiguation across these is left to the
    caller's eval loop -- this only proposes the plausible villages, it never picks one."""
    fmb = {_norm_survey(s) or s for s in fmb_surveys}
    by_vc: dict[str, list[dict]] = {}
    for r in parcels:
        if r["vc"]:
            by_vc.setdefault(r["vc"], []).append(r)

    ax, ay = anchor_utm
    ranked: list[dict] = []
    for vc, rows in by_vc.items():
        surveys = {_norm_survey(r["sn"]) or r["sn"] for r in rows}
        overlap = fmb & surveys
        if len(overlap) < min_overlap:
            continue
        cx = sum(r["poly"].centroid.x for r in rows) / len(rows)
        cy = sum(r["poly"].centroid.y for r in rows) / len(rows)
        dist = math.hypot(cx - ax, cy - ay)
        if dist > radius_m:
            continue
        ranked.append({"vc": vc, "rows": rows, "center": (round(cx), round(cy)),
                       "n_overlap": len(overlap), "dist_m": round(dist)})
    # most FMB-survey coverage first, then nearest to the anchor
    ranked.sort(key=lambda d: (-d["n_overlap"], d["dist_m"]))
    ranked = ranked[:max_cand]
    for d in ranked:
        parcel_map = _one_parcel_per_survey([r for r in d["rows"]
                                             if (_norm_survey(r["sn"]) or r["sn"]) in fmb])
        d["source"] = VectorCadastralSource(parcel_map, crs=crs, village=d["vc"])
        d.pop("rows", None)
    return ranked
