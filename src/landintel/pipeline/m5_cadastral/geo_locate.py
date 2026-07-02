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
  3. SURVEY-NUMBER FINGERPRINT (``refine_by_surveys``): fetch a wide tile window around the
     anchor, OCR the orange survey labels, and keep the tiles where THIS village's survey
     numbers actually appear -> a tight bbox. This is the authoritative step; the anchor
     only has to be close enough to land the survey numbers inside the window.

``locate_village`` runs 1 (+2 if needed) then 3 and returns (bbox_utm, info).
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
    try:
        url = _NOMINATIM + "?" + urllib.parse.urlencode(
            {"q": query, "format": "json", "limit": 1})
        req = urllib.request.Request(url, headers=_UA)
        r = json.load(urllib.request.urlopen(req, timeout=20))
        if r:
            return float(r[0]["lat"]), float(r[0]["lon"])
    except Exception as exc:  # noqa: BLE001
        _log.debug("geocode(%r) failed: %s", query, exc)
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
# Layer 3: survey-number fingerprint (authoritative)
# --------------------------------------------------------------------------- #
def refine_by_surveys(center_utm: tuple[float, float], surveys: set[str], crs: str,
                      cache_dir: str, radius_m: float = 2500.0,
                      min_hits: int = 2) -> tuple[tuple[float, float, float, float], dict]:
    """Fetch tiles around ``center_utm``, locate which of ``surveys`` appear, and return
    a TIGHT bbox around the found parcels (+ info). Falls back to the wide window when
    fewer than ``min_hits`` survey numbers are found (so the caller can widen/retry)."""
    from .s3_tiles import S3CadastralSource

    cx, cy = center_utm
    wide = (cx - radius_m, cy - radius_m, cx + radius_m, cy + radius_m)
    src = S3CadastralSource(wide, surveys, cache_dir=cache_dir, crs=crs)
    found: dict[str, tuple[float, float, float, float]] = {}
    for sn in surveys:
        p = src.get(sn)
        if p is not None and p.polygon is not None:
            found[sn] = p.polygon.bounds
    if len(found) < min_hits:
        return wide, {"hits": list(found), "n_hits": len(found), "tight": False}
    xs0 = min(b[0] for b in found.values()); ys0 = min(b[1] for b in found.values())
    xs1 = max(b[2] for b in found.values()); ys1 = max(b[3] for b in found.values())
    pad = 300.0
    tight = (xs0 - pad, ys0 - pad, xs1 + pad, ys1 + pad)
    return tight, {"hits": sorted(found, key=lambda s: int(s) if s.isdigit() else 0),
                   "n_hits": len(found), "tight": True}


def locate_village(village: str, surveys: set[str], crs: str, cache_dir: str,
                   taluk: str = "", district: str = "", state: str = "Tamil Nadu",
                   use_qwen: bool = True) -> tuple[tuple[float, float, float, float], dict]:
    """Full locator: rough anchor (geocode, Qwen+web fallback) -> survey-number fingerprint
    -> tight UTM bbox. Returns (bbox_utm, info)."""
    from pyproj import Transformer
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    info: dict = {"village": village, "taluk": taluk, "district": district}
    anchor = rough_center(village, taluk, district, state)
    level = anchor[1] if anchor else None
    latlon = anchor[0] if anchor else None
    # Qwen disambiguation when the village itself did not geocode (only taluk/district hit).
    if use_qwen and (latlon is None or level != "village"):
        q = qwen_center(village, taluk, district, state)
        if q is not None:
            latlon, level = q, "qwen"
    if latlon is None:
        raise RuntimeError(f"could not locate {village} ({taluk}, {district}) at all")
    cx, cy = to_utm.transform(latlon[1], latlon[0])
    info.update(anchor_latlon=latlon, anchor_level=level, anchor_utm=(round(cx), round(cy)))

    # widen the search window until the survey-number fingerprint locks on.
    for radius in (2500.0, 5000.0, 9000.0):
        bbox, fp = refine_by_surveys((cx, cy), surveys, crs, cache_dir, radius_m=radius)
        info.update(radius_m=radius, **fp)
        if fp["tight"]:
            return bbox, info
    return bbox, info      # best-effort wide bbox if the fingerprint never locked
