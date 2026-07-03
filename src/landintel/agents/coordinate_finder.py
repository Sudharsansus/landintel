"""CoordinateFinderAgent -- ONE agent that finds a village's EXACT location automatically.

Replaces the manual ``--lat/--lon`` anchor. GENERAL (no per-village constants):

  1. ROUGH web geocode of "<village>, <taluk>, <district>, Tamil Nadu" (Nominatim), falling
     back to taluk, then district. ALL geocode candidates are considered, not just the first
     hit -- TN village names recur, and the first hit can be a same-named village elsewhere
     (measured: MOOLAKARAI's first hit was a homonym 15+ km east of the real village).
  2. EXACT refine against the TNGIS VECTOR CADASTRE (the authoritative source): around EVERY
     rough pin, bbox-filter the parquet, group parcels by ``lgd_village_code``, and score each
     candidate village by COVERAGE = |its surveys ∩ FMB surveys| / |FMB surveys|. Coverage is
     the decisive fingerprint: small survey numbers (2, 3, 14...) recur in every TN village, so
     a raw >=3 overlap proves nothing, but carrying MOST of a 20-survey FMB set does. The pin
     only proposes where to look; the fingerprint decides.
  3. If no pin yields a DECISIVE village, the parcel search WIDENS (bigger bbox pad + radius)
     around the best pins before falling back to geocode-only.

Confidence: ``high`` only when the best village's coverage is decisive (>= 0.5 of the FMB
surveys AND >= 1.5x the runner-up village) -- both criteria keyed on the FMB's own survey
list, never absolute counts. Otherwise ``medium`` (best-effort anchor) / ``none``.

0-FP philosophy preserved: this agent only MEASURES/PROPOSES a coordinate; the final
which-block decision is still the shape-IoU disambiguation in run_m2_cad.
"""
from __future__ import annotations

import logging

from .base import Agent

_log = logging.getLogger(__name__)

# Decisiveness of the survey-number fingerprint (fractions of the FMB's OWN survey list).
COVERAGE_DECISIVE = 0.5     # best village must carry >= this fraction of the FMB surveys
COVERAGE_MARGIN = 1.5       # ... and beat the runner-up village by this factor
# Progressive parcel-search widening: (bbox pad deg, candidate radius m). The second pass
# only runs when the first finds nothing decisive anywhere.
_SEARCH_PASSES = ((0.06, 6000.0), (0.20, 20000.0))
_MAX_PINS = 6               # distinct geocode pins to refine (dedup within ~1 km)


class CoordinateFinderAgent(Agent):
    name = "CoordinateFinderAgent"

    def find(self, village: str, surveys: set[str], *, district: str = "", taluk: str = "",
             parquet_path: str | None = None, crs: str = "EPSG:32643",
             cache_json: str | None = None) -> dict:
        """Return ``{lat, lon, utm, confidence, method, vc, n_surveys, coverage}``.

        confidence: ``high`` (decisive TNGIS survey-number coverage) / ``medium``
        (best-effort: non-decisive cadastre hit or web geocode only) / ``none``."""
        from pyproj import Transformer

        from ..pipeline.m5_cadastral.geo_locate import geocode_candidates

        # --- 1. rough web geocode: collect DISTINCT pins across the query tiers ----------
        pins: list[tuple[float, float]] = []
        pin_q: dict[int, str] = {}
        for q in (f"{village}, {taluk}, {district}, Tamil Nadu, India",
                  f"{village}, {district}, Tamil Nadu, India",
                  f"{taluk}, {district}, Tamil Nadu, India",
                  f"{district}, Tamil Nadu, India"):
            if not q.replace(",", "").replace("Tamil Nadu", "").replace("India", "").strip():
                continue
            for lat, lon in geocode_candidates(q, limit=5):
                if any(abs(lat - a) < 0.01 and abs(lon - b) < 0.01 for a, b in pins):
                    continue                        # ~1 km dedup
                pin_q[len(pins)] = q
                pins.append((lat, lon))
                if len(pins) >= _MAX_PINS:
                    break
            if len(pins) >= _MAX_PINS:
                break
        if not pins:
            return {"method": "none", "confidence": "none", "lat": None, "lon": None}

        # --- 2. EXACT refine: score EVERY pin's cadastre villages by survey coverage ------
        n_surv = max(len(surveys), 1)
        best_by_vc: dict[str, dict] = {}
        try:
            from ..pipeline.m5_cadastral.vector_locate import (
                load_area_parcels_cached, village_candidates)
            fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            for pad_deg, radius_m in _SEARCH_PASSES:
                for pi, rough in enumerate(pins):
                    ax, ay = fwd.transform(rough[1], rough[0])
                    kw = {"parquet_path": parquet_path} if parquet_path else {}
                    if pad_deg != _SEARCH_PASSES[0][0]:
                        kw["pad_deg"] = pad_deg
                        cache = None                # widened pass: don't poison the cache
                    else:
                        cache = cache_json if pi == 0 else None
                    parcels = load_area_parcels_cached(rough, cache_json=cache, **kw)
                    for c in village_candidates(parcels, surveys, (ax, ay),
                                                radius_m=radius_m, min_overlap=3,
                                                max_cand=6, crs=crs):
                        cov = c["n_overlap"] / n_surv
                        prev = best_by_vc.get(str(c["vc"]))
                        if prev is None or cov > prev["coverage"]:
                            best_by_vc[str(c["vc"])] = {**c, "coverage": cov,
                                                        "pin": rough, "q": pin_q.get(pi, "")}
                ranked = sorted(best_by_vc.values(),
                                key=lambda c: (-c["coverage"], c.get("dist_m", 0.0)))
                if ranked:
                    top = ranked[0]
                    runner = ranked[1]["coverage"] if len(ranked) > 1 else 0.0
                    decisive = (top["coverage"] >= COVERAGE_DECISIVE
                                and top["coverage"] >= COVERAGE_MARGIN * max(runner, 1e-9))
                    if decisive or pad_deg == _SEARCH_PASSES[-1][0]:
                        cx, cy = top["center"]
                        inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                        lon, lat = inv.transform(cx, cy)
                        conf = "high" if decisive else "medium"
                        _log.info(
                            "CoordinateFinder: %s -> TNGIS vc=%s @ (%.5f, %.5f) coverage="
                            "%.0f%% (runner-up %.0f%%) -> %s", village, top["vc"], lat, lon,
                            100 * top["coverage"], 100 * runner, conf)
                        return {"lat": round(lat, 6), "lon": round(lon, 6),
                                "utm": (cx, cy), "confidence": conf,
                                "method": "tngis-cadastre", "vc": top["vc"],
                                "n_surveys": top["n_overlap"],
                                "coverage": round(top["coverage"], 3),
                                "n_candidates": len(ranked)}
                    # not decisive yet -> widen and keep accumulating candidates
        except Exception as exc:  # noqa: BLE001
            _log.warning("CoordinateFinder cadastre refine failed (%s); using geocode", exc)

        # --- 3. fallback: best rough geocode pin only --------------------------------------
        lat, lon = pins[0]
        return {"lat": round(lat, 6), "lon": round(lon, 6), "utm": None,
                "confidence": "medium", "method": f"geocode:{pin_q.get(0, '')}"}

    # Agent.run adapter: pull village/surveys/district/taluk from context, report as a note.
    def run(self, results, context: dict):  # noqa: D401
        from .base import AgentReport
        r = self.find(context.get("village", ""), set(context.get("surveys", [])),
                      district=context.get("district", ""), taluk=context.get("taluk", ""),
                      parquet_path=context.get("parquet_path"), crs=context.get("crs", "EPSG:32643"))
        note = (f"village anchor ({r['confidence']}, {r['method']}): "
                f"lat={r.get('lat')} lon={r.get('lon')}"
                + (f" vc={r['vc']} surveys={r['n_surveys']}" if r.get("vc") else ""))
        return AgentReport(agent=self.name, notes=[note])
