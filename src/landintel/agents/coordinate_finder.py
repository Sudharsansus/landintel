"""CoordinateFinderAgent -- ONE agent that finds a village's EXACT location automatically.

Replaces the manual ``--lat/--lon`` anchor. Two stages, coarse -> exact, GENERAL (no per-village
constants):

  1. ROUGH web geocode of "<village>, <taluk>, <district>, Tamil Nadu" (Nominatim), falling back
     to taluk, then district, so we always get a coarse pin even for a tiny revenue village that
     does not geocode by name.
  2. EXACT refine against the TNGIS VECTOR CADASTRE (the authoritative source): bbox-filter the
     parquet around the rough pin, group parcels by ``lgd_village_code``, and pick the village that
     actually CARRIES the FMB's survey numbers nearest the pin. Its parcel centroid is the exact
     anchor -- no web guessing, no Qwen hallucination. (The final which-block decision is still the
     shape-IoU disambiguation in run_m2_cad; this agent only proposes the anchor.)

0-FP philosophy of the agent layer is preserved: this agent only MEASURES/PROPOSES a coordinate;
it never decides a placement.
"""
from __future__ import annotations

import logging

from .base import Agent

_log = logging.getLogger(__name__)


class CoordinateFinderAgent(Agent):
    name = "CoordinateFinderAgent"

    def find(self, village: str, surveys: set[str], *, district: str = "", taluk: str = "",
             parquet_path: str | None = None, crs: str = "EPSG:32643",
             cache_json: str | None = None) -> dict:
        """Return ``{lat, lon, utm, confidence, method, vc, n_surveys}`` for the village.

        confidence: ``high`` (matched in the TNGIS cadastre by survey number) / ``medium``
        (web geocode only) / ``none`` (nothing found)."""
        from pyproj import Transformer

        from ..pipeline.m5_cadastral.geo_locate import geocode

        # --- 1. rough web geocode (coarse pin) --------------------------------------------
        rough = rough_q = None
        for q in (f"{village}, {taluk}, {district}, Tamil Nadu, India",
                  f"{village}, {district}, Tamil Nadu, India",
                  f"{taluk}, {district}, Tamil Nadu, India",
                  f"{district}, Tamil Nadu, India"):
            if not q.replace(",", "").replace("Tamil Nadu", "").replace("India", "").strip():
                continue
            g = geocode(q)
            if g:
                rough, rough_q = g, q
                break
        if rough is None:
            return {"method": "none", "confidence": "none", "lat": None, "lon": None}

        # --- 2. EXACT refine against the TNGIS vector cadastre ------------------------------
        try:
            from ..pipeline.m5_cadastral.vector_locate import (
                load_area_parcels_cached, village_candidates)
            fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            ax, ay = fwd.transform(rough[1], rough[0])
            parcels = load_area_parcels_cached(
                rough, cache_json=cache_json,
                **({"parquet_path": parquet_path} if parquet_path else {}))
            cands = village_candidates(parcels, surveys, (ax, ay),
                                       radius_m=6000.0, min_overlap=3, max_cand=6, crs=crs)
            if cands:
                best = cands[0]                     # most FMB-survey overlap, then nearest the pin
                cx, cy = best["center"]
                inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                lon, lat = inv.transform(cx, cy)
                _log.info("CoordinateFinder: %s -> TNGIS village vc=%s @ (%.5f, %.5f)",
                          village, best["vc"], lat, lon)
                return {"lat": round(lat, 6), "lon": round(lon, 6), "utm": (cx, cy),
                        "confidence": "high", "method": "tngis-cadastre", "vc": best["vc"],
                        "n_surveys": best["n_overlap"], "n_candidates": len(cands)}
        except Exception as exc:  # noqa: BLE001
            _log.warning("CoordinateFinder cadastre refine failed (%s); using geocode", exc)

        # --- 3. fallback: rough geocode only -----------------------------------------------
        return {"lat": round(rough[0], 6), "lon": round(rough[1], 6), "utm": None,
                "confidence": "medium", "method": f"geocode:{rough_q}"}

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
