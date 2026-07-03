"""General village geolocator for the cadastral fetch -- NO per-village hardcoding.

Finds the UTM bbox of a village's parcels on the public cadastral tileset using the
village's OWN survey numbers as the fingerprint, so nothing village-specific is baked in
(overfit-proof). Three layers, each a fallback for the last:

  1. ROUGH ANCHOR (``rough_center``): geocode "<village>, <taluk>, <district>, Tamil Nadu"
     via Nominatim. District/taluk almost always resolve even when a tiny revenue village
     does not, so we always get at least a coarse anchor.
  2. QWEN + WEB (``qwen_center``): when geocoding is ambiguous (e.g. Moolakarai vs
     Moolapalayam) the local Qwen brain, given a web-search tool, disambiguates and returns
     coordinates. Opt-in / best-effort -- skipped silently if Ollama is down.
  3. SURVEY-NUMBER FINGERPRINT + LOCALITY CLUSTER (``refine_by_surveys``): fetch a wide tile
     window around the anchor, OCR the orange survey labels, then SINGLE-LINKAGE CLUSTER the
     readings and keep the ONE component holding the most of THIS village's survey numbers.
     That cluster is the village; its convex hull (buffered) is the VILLAGE FENCE and its
     bounds are the tight bbox. This is the authoritative + anti-scatter step: the same
     survey number read in a neighbouring village lands in a different cluster and is dropped,
     so parcels never scatter across the district. The anchor only has to be close enough to
     land the survey numbers somewhere in the window.

``locate_village`` runs 1 (+2 if needed) then 3 and returns (bbox_utm, fence, info).
"""
from __future__ import annotations

import json
import logging
import math
import urllib.parse
import urllib.request

_log = logging.getLogger(__name__)

_NOMINATIM = "https://nominatim.openstreetmap.org/search"
_DDG = "https://api.duckduckgo.com/"
_UA = {"User-Agent": "landintel-geolocate/1.0 (cadastral survey mapping)"}


# --------------------------------------------------------------------------- #
# Layer 1: plain geocoding (internet, no key)
# --------------------------------------------------------------------------- #
def geocode(query: str) -> tuple[float, float] | None:
    """Return (lat, lon) for a free-text place query via Nominatim, or None."""
    c = geocode_candidates(query, limit=1)
    return c[0] if c else None


def geocode_candidates(query: str, limit: int = 5) -> list[tuple[float, float]]:
    """ALL (lat, lon) candidates for a place query, best-first.

    Provider is GENERAL and swappable (no village-specific logic):
      * Google Maps Geocoding API when ``GOOGLE_MAPS_API_KEY`` (or
        ``LANDINTEL_GOOGLE_MAPS_KEY``) is set -- far better on small Indian revenue
        villages, which Nominatim frequently misplaces onto a same-named homonym
        (measured: MOOLAKARAI landed 5.7 km off). Online + keyed.
      * Nominatim otherwise (offline-default deployment, no key).

    TN village names recur, so even a good geocoder returns homonyms: callers that
    hold an independent fingerprint (the FMB survey numbers) MUST score every
    candidate and let the fingerprint decide. The pin is only a hint.
    """
    g = _google_geocode_candidates(query, limit)
    if g is not None:
        return g
    try:
        url = _NOMINATIM + "?" + urllib.parse.urlencode(
            {"q": query, "format": "json", "limit": int(limit)})
        req = urllib.request.Request(url, headers=_UA)
        r = json.load(urllib.request.urlopen(req, timeout=20))
        return [(float(e["lat"]), float(e["lon"])) for e in r]
    except Exception as exc:  # noqa: BLE001
        _log.debug("geocode_candidates(%r) failed: %s", query, exc)
    return []


def _google_geocode_candidates(query: str, limit: int) -> list[tuple[float, float]] | None:
    """Google Maps geocode candidates, or None when no API key is configured (so the
    caller falls back to Nominatim). Uses the REST endpoint directly -- no extra
    dependency; if the optional ``googlemaps`` client is installed it is honoured too."""
    import os
    key = os.environ.get("GOOGLE_MAPS_API_KEY") or os.environ.get("LANDINTEL_GOOGLE_MAPS_KEY")
    if not key:
        return None
    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json?" + urllib.parse.urlencode(
            {"address": query, "key": key})
        req = urllib.request.Request(url, headers=_UA)
        data = json.load(urllib.request.urlopen(req, timeout=20))
        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            _log.warning("Google geocode status=%s for %r", data.get("status"), query)
        out = []
        for r in (data.get("results") or [])[:int(limit)]:
            loc = r.get("geometry", {}).get("location", {})
            if "lat" in loc and "lng" in loc:
                out.append((float(loc["lat"]), float(loc["lng"])))
        return out
    except Exception as exc:  # noqa: BLE001
        _log.warning("Google geocode failed for %r (%s); falling back to Nominatim", query, exc)
        return None


def rough_center(village: str, taluk: str = "", district: str = "",
                 state: str = "Tamil Nadu") -> tuple[tuple[float, float], str] | None:
    """Coarse (lat, lon) anchor: try village, then taluk, then district. Returns
    ((lat, lon), level) where level says how precise the hit was."""
    for q, level in (
        (", ".join(x for x in (village, taluk, district, state, "India") if x), "village"),
        (", ".join(x for x in (taluk, district, state, "India") if x), "taluk"),
        (", ".join(x for x in (district, state, "India") if x), "district"),
    ):
        ll = geocode(q)
        if ll is not None:
            return ll, level
    return None


# --------------------------------------------------------------------------- #
# Layer 2: Qwen + web search (disambiguation fallback)
# --------------------------------------------------------------------------- #
def web_search(query: str, max_chars: int = 800) -> str:
    """Best-effort keyless web lookup (DuckDuckGo Instant Answer). Returns a text blob
    the LLM can read; empty string on failure. This is the 'internet' the brain gets."""
    try:
        url = _DDG + "?" + urllib.parse.urlencode(
            {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1})
        req = urllib.request.Request(url, headers=_UA)
        d = json.load(urllib.request.urlopen(req, timeout=20))
        parts = [d.get("AbstractText", "")]
        for t in d.get("RelatedTopics", [])[:5]:
            if isinstance(t, dict) and t.get("Text"):
                parts.append(t["Text"])
        return " | ".join(p for p in parts if p)[:max_chars]
    except Exception as exc:  # noqa: BLE001
        _log.debug("web_search(%r) failed: %s", query, exc)
        return ""


def qwen_center(village: str, taluk: str = "", district: str = "",
                state: str = "Tamil Nadu") -> tuple[float, float] | None:
    """Ask the local Qwen brain (with a web-search observation) to disambiguate the
    village and return coordinates. Returns (lat, lon) or None if unreachable/unsure.

    Qwen READS a web-search blob + geocode hints and REASONS; the actual coordinate is
    validated downstream by the survey-number fingerprint, so a wrong guess cannot create
    a false placement -- it only fails to land the survey numbers and we widen/retry."""
    try:
        from ...llm.providers import llm_call
    except Exception:  # noqa: BLE001
        return None
    hint = web_search(f"{village} village {taluk} {district} {state} latitude longitude")
    geo = geocode(f"{village}, {taluk}, {district}, {state}, India") or geocode(
        f"{taluk}, {district}, {state}, India")
    system = ("You are a Tamil Nadu geography assistant. Given a revenue village, its "
              "taluk and district, return the village's approximate WGS84 centre as STRICT "
              "JSON {\"lat\": <deg>, \"lon\": <deg>}. Use the web snippet + geocode hint. "
              "If unsure, give the taluk centre. Reply with JSON only.")
    prompt = (f"Village: {village}\nTaluk: {taluk}\nDistrict: {district}\nState: {state}\n"
              f"Web snippet: {hint or '(none)'}\n"
              f"Geocode hint (lat,lon): {geo if geo else '(none)'}\n"
              "JSON centre:")
    out = llm_call(prompt, max_tokens=120, system=system)
    if not out:
        return None
    try:
        txt = out[0]
        s = txt[txt.index("{"): txt.rindex("}") + 1]
        j = json.loads(s)
        lat, lon = float(j["lat"]), float(j["lon"])
        # sanity: Tamil Nadu roughly lat 8-14, lon 76-81
        if 7.0 <= lat <= 14.5 and 75.5 <= lon <= 81.0:
            return lat, lon
    except Exception as exc:  # noqa: BLE001
        _log.debug("qwen_center parse failed: %s", exc)
    return None


# --------------------------------------------------------------------------- #
# Layer 3: survey-number fingerprint (authoritative) -- with LOCALITY CLUSTERING
# --------------------------------------------------------------------------- #
def _cluster_points(pts: list[tuple[float, float]], link_m: float) -> list[list[int]]:
    """Single-linkage clusters: two points join if within ``link_m`` (union-find).

    A village's survey labels are spatially contiguous (adjacent parcels, label
    centres tens-to-hundreds of metres apart), while the same survey NUMBER in a
    neighbouring village sits across a road/gap typically > link_m away -> lands in a
    different cluster. Returns lists of indices into ``pts``."""
    n = len(pts)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            if math.hypot(xi - pts[j][0], yi - pts[j][1]) <= link_m:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return list(comps.values())


def _select_in_village(found: dict[str, list[tuple[float, float, float]]],
                       support_r: float) -> dict[str, tuple[float, float, int]]:
    """Pick, for EACH survey, the reading that sits among the most village-mates.

    A survey number is unique only within a village, so the same number is OCR-read in
    several villages. The correct (in-village) reading is the one SURROUNDED BY OTHER of
    THIS village's survey numbers; an isolated cross-village duplicate has no village-mate
    nearby. So for each candidate reading we count the DISTINCT OTHER surveys with any
    reading within ``support_r`` and keep, per survey, the reading with the most support
    (tie-break: OCR confidence). Returns {sn: (x, y, support)} -- one clean point per
    survey, each at its in-village position. This removes the duplicate-bridging that made
    single-linkage over-connect neighbouring villages into one blob."""
    cand = [(sn, x, y, c) for sn, reads in found.items() for (x, y, c) in reads]
    chosen: dict[str, tuple[float, float, int, float]] = {}   # sn -> (x,y,support,conf)
    for i, (sn, x, y, c) in enumerate(cand):
        support = len({cand[j][0] for j in range(len(cand))
                       if cand[j][0] != sn
                       and math.hypot(x - cand[j][1], y - cand[j][2]) <= support_r})
        if sn not in chosen or (support, c) > (chosen[sn][2], chosen[sn][3]):
            chosen[sn] = (x, y, support, c)
    return {sn: (v[0], v[1], v[2]) for sn, v in chosen.items()}


def _village_blocks(found: dict[str, list[tuple[float, float, float]]],
                    village_r: float = 900.0, support_r: float = 450.0,
                    min_support: int = 2, min_surveys: int = 3,
                    anchor: tuple[float, float] | None = None,
                    anchor_radius_m: float = 6000.0
                    ) -> list[dict[str, tuple[float, float]]]:
    """Find the CANDIDATE VILLAGE BLOCKS among the raw multi-reads by GREEDY DENSITY-PEAK
    peeling.

    In TN cadastre every village numbers its parcels from 1, so the SAME survey numbers
    (1..N) recur in every neighbouring village. Our FMB numbers therefore appear as several
    complete, dense BLOCKS -- one per nearby village -- plus scattered single-label noise.
    Single-linkage FAILS to separate them: a super-common number like "2" is read hundreds
    of times and BRIDGES the villages into one blob. So instead:

      1. score each reading by SUPPORT = # distinct other surveys within ``support_r``; drop
         readings below ``min_support`` (isolated label noise).
      2. GREEDILY PEEL villages: repeatedly find the densest spot (the reading with the most
         DISTINCT surveys within ``village_r``), form a village from the nearest reading of
         each of those surveys, record it, then REMOVE every reading within ``village_r`` of
         that centre and repeat. Capping each village to ``village_r`` makes it impossible
         for one common number to bridge two villages; each block stays compact.
      3. keep blocks with >= ``min_surveys`` distinct surveys within ``anchor_radius_m`` of
         the geocoded centre. Returns blocks, most-complete (most distinct surveys) first.
         Which block is really OUR village is decided downstream by FMB-shape agreement."""
    cand = [(sn, x, y, c) for sn, reads in found.items() for (x, y, c) in reads]
    dense: list[tuple[str, float, float]] = []
    for i, (sn, x, y, _c) in enumerate(cand):
        support = len({cand[j][0] for j in range(len(cand))
                       if cand[j][0] != sn
                       and math.hypot(x - cand[j][1], y - cand[j][2]) <= support_r})
        if support >= min_support:
            dense.append((sn, x, y))
    remaining = list(dense)
    blocks: list[tuple[int, dict[str, tuple[float, float]]]] = []
    while remaining:
        # densest peak = reading whose village_r neighbourhood holds the most distinct surveys
        best_i, best_set = -1, set()
        for i, (sn, x, y) in enumerate(remaining):
            s = {p[0] for p in remaining
                 if math.hypot(x - p[1], y - p[2]) <= village_r}
            if len(s) > len(best_set):
                best_set, best_i = s, i
        if best_i < 0 or len(best_set) < min_surveys:
            break
        cx, cy = remaining[best_i][1], remaining[best_i][2]
        village: dict[str, tuple[float, float]] = {}
        for sn in best_set:                             # nearest reading of each survey
            reads = [p for p in remaining
                     if p[0] == sn and math.hypot(cx - p[1], cy - p[2]) <= village_r]
            if reads:
                p = min(reads, key=lambda q: math.hypot(cx - q[1], cy - q[2]))
                village[sn] = (p[1], p[2])
        if anchor is None or math.hypot(cx - anchor[0], cy - anchor[1]) <= anchor_radius_m:
            blocks.append((len(village), village))
        # consume this village's neighbourhood so the next peak is a DIFFERENT village
        remaining = [p for p in remaining
                     if math.hypot(cx - p[1], cy - p[2]) > village_r]
    blocks.sort(key=lambda b: -b[0])
    return [b[1] for b in blocks]


def _village_cluster(found: dict[str, list[tuple[float, float, float]]],
                     link_m: float, support_r: float = 450.0,
                     anchor: tuple[float, float] | None = None
                     ) -> dict[str, tuple[float, float]] | None:
    """The single most-complete candidate village block (largest distinct-survey block),
    or None. Thin wrapper over ``_village_blocks`` for callers wanting one fence."""
    blocks = _village_blocks(found, link_m, support_r, anchor=anchor)
    return blocks[0] if blocks else None


def _block_bbox_fence(block: dict[str, tuple[float, float]]):
    """(tight_bbox, fence_polygon) for one candidate village block."""
    from shapely.geometry import MultiPoint
    fence = MultiPoint(list(block.values())).convex_hull.buffer(300.0)
    b = fence.bounds
    return (b[0] - 50.0, b[1] - 50.0, b[2] + 50.0, b[3] + 50.0), fence


def refine_by_surveys(center_utm: tuple[float, float], surveys: set[str], crs: str,
                      cache_dir: str, radius_m: float = 2500.0, min_hits: int = 2,
                      village_r: float = 900.0):
    """Fetch tiles around ``center_utm``, OCR-locate which of ``surveys`` appear, and return
    the CANDIDATE VILLAGE BLOCKS (each a dense village that shares our survey numbers).

    Returns ``(candidates, info)`` where ``candidates`` is a list (largest block first) of
    ``{bbox, fence, surveys, n, center, span_m}``; ``fence`` (a shapely polygon) is passed
    to ``S3CadastralSource(village_fence=...)``. Empty list if nothing clusters. Which
    candidate is really our village is decided by the caller via FMB-shape agreement."""
    from .s3_tiles import TileGrid, download_tiles

    cx, cy = center_utm
    wide = (cx - radius_m, cy - radius_m, cx + radius_m, cy + radius_m)
    grid = TileGrid(crs)
    tiles = download_tiles(wide, grid, cache_dir)
    # LIGHT locate pass only (first-pass OCR over the wide window) -- just enough to find
    # the candidate label points; the expensive multi-pass recovery runs later, FENCED.
    found = _cached_locate(tiles, grid, surveys, cache_dir, wide)

    # anchor = the geocoded village centre; candidate BLOCKS = each dense village that shares
    # our survey numbers. Which one is really ours is decided downstream by FMB-shape fit.
    blocks = _village_blocks(found, village_r, anchor=(cx, cy))
    blocks = [b for b in blocks if len(b) >= min_hits]
    if not blocks:
        return [], {"hits": [], "n_hits": 0, "tight": False}
    candidates = []
    for blk in blocks:
        bbox, fence = _block_bbox_fence(blk)
        b = fence.bounds
        candidates.append({
            "bbox": bbox, "fence": fence,
            "surveys": sorted(blk, key=lambda s: int(s) if s.isdigit() else 0),
            "n": len(blk), "center": (round((b[0] + b[2]) / 2), round((b[1] + b[3]) / 2)),
            "span_m": (round(b[2] - b[0]), round(b[3] - b[1]))})
    return candidates, {
        "n_candidates": len(candidates), "tight": True, "village_r": village_r,
        "n_hits": candidates[0]["n"], "hits": candidates[0]["surveys"],
        "span_m": candidates[0]["span_m"],
        "candidate_centers": [c["center"] for c in candidates]}


def _cached_locate(tiles, grid, surveys: set[str], cache_dir: str,
                   wide: tuple[float, float, float, float]
                   ) -> dict[str, list[tuple[float, float, float]]]:
    """First-pass OCR locate over the wide window, cached by (survey set + window) so a
    re-run is instant. Returns the raw multi-reads ``{sn: [(x,y,conf),...]}``."""
    from pathlib import Path

    from .s3_tiles import ocr_locate_labels

    key = "_".join(str(round(v / 50.0) * 50) for v in wide)
    cf = Path(cache_dir) / f"locate_reads_{key}.json"
    if cf.exists():
        try:
            d = json.loads(cf.read_text())
            if set(d.get("known", [])) == set(surveys):
                return {k: [tuple(t) for t in v] for k, v in d["reads"].items()}
        except Exception:  # noqa: BLE001
            pass
    _best, found = ocr_locate_labels(tiles, grid, surveys)
    try:
        cf.write_text(json.dumps({"known": sorted(surveys),
                                  "reads": {k: [list(t) for t in v]
                                            for k, v in found.items()}}))
    except Exception:  # noqa: BLE001
        pass
    return found


def locate_village(village: str, surveys: set[str], crs: str, cache_dir: str,
                   taluk: str = "", district: str = "", state: str = "Tamil Nadu",
                   use_qwen: bool = True, anchor_latlon: tuple[float, float] | None = None):
    """Full locator: rough anchor (geocode, Qwen+web fallback, or an EXPLICIT web-provided
    anchor) -> survey-number fingerprint -> CANDIDATE VILLAGE BLOCKS.

    ``anchor_latlon`` (lat, lon), when given, OVERRIDES geocoding -- this is the web-anchor
    path (a Google/pincode/Nominatim lookup outside the pipeline pins the village centre,
    since a tiny revenue village often does not geocode from its name alone). Density-peak
    clustering + FMB-shape IoU still refine WHICH block near that anchor is really ours, so a
    coarse (centroid-precision) web anchor is enough.

    Returns ``(candidates, info)``; ``candidates`` is a list of dense village blocks that
    share our survey numbers (largest first, each with ``bbox``/``fence``). Empty if nothing
    locked."""
    from pyproj import Transformer
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    info: dict = {"village": village, "taluk": taluk, "district": district}
    if anchor_latlon is not None:                       # web-provided anchor wins
        latlon, level = (float(anchor_latlon[0]), float(anchor_latlon[1])), "web"
    else:
        anchor = rough_center(village, taluk, district, state)
        level = anchor[1] if anchor else None
        latlon = anchor[0] if anchor else None
        # Qwen disambiguation when the village itself did not geocode (only taluk/district).
        if use_qwen and (latlon is None or level != "village"):
            q = qwen_center(village, taluk, district, state)
            if q is not None:
                latlon, level = q, "qwen"
    if latlon is None:
        raise RuntimeError(f"could not locate {village} ({taluk}, {district}) at all")
    cx, cy = to_utm.transform(latlon[1], latlon[0])
    info.update(anchor_latlon=latlon, anchor_level=level, anchor_utm=(round(cx), round(cy)))

    # widen the search window until the survey-number fingerprint clusters onto a village.
    candidates: list = []
    for radius in (2500.0, 5000.0, 9000.0):
        candidates, fp = refine_by_surveys((cx, cy), surveys, crs, cache_dir,
                                           radius_m=radius)
        info.update(radius_m=radius, **fp)
        if fp["tight"]:
            return candidates, info
    return candidates, info      # possibly empty if the fingerprint never locked
