"""QGIS review-layer export -- the FP-safe way to plug into the QGIS ecosystem.

Emits the job as an RFC-7946 GeoJSON (WGS84) that QGIS / any GIS reads natively: one
feature per plot, styled by status, with the InputRequest reason + instruction in the
attribute table. The surveyor opens it, sees which plots need input and exactly what,
and seeds them. NOTE: geometry here is produced DETERMINISTICALLY by Shapely + pyproj
(reproject UTM -> WGS84) -- no LLM/agent decides any coordinate; the agent layer only
attaches the human-facing review attributes. So this gives the QGIS-agent ecosystem's
ergonomics with none of its false-positive risk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .dispositions import CONFIDENT, normalize

_log = logging.getLogger(__name__)
_CONF = CONFIDENT


def write_review_geojson(results, requests_by_sn: dict, output_dir, crs: str) -> Path | None:
    """Write review_map.geojson: confident plots (status=confident) + the rest with their
    minimal-input request attached. Returns the path, or None if nothing geocodable.

    STAGE-AGNOSTIC: normalizes ClubResult / GeorefResult to one disposition so the export
    works for both stages; the footprint resolves from the in-memory placement (M2) or the
    output DXF (M3) automatically."""
    from pyproj import Transformer
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    output_dir = Path(output_dir)

    feats = []
    for d in normalize(results):
        if d.recommendation in _CONF:
            status, color = "confident", "#2ca02c"            # green
        elif d.recommendation == "NO_COVERAGE":
            status, color = "staged", "#d62728"               # red
        else:
            status, color = "review", "#ff7f0e"               # orange

        geom = None
        if d.has_geometry:
            fp = d.footprint
            ring = [list(to_wgs.transform(x, y)) for x, y in fp.exterior.coords]
            geom = {"type": "Polygon", "coordinates": [ring]}

        props = {"survey": d.survey, "status": status,
                 "disposition": d.recommendation, "method": d.method or "",
                 "qgis_color": color}
        q = requests_by_sn.get(d.survey)
        if q:
            props.update({"input_type": q["input_type"], "reason": q["reason"],
                          "instruction": q["instruction"]})
            if geom is None and q.get("known_utm"):
                lon, lat = to_wgs.transform(*q["known_utm"])
                geom = {"type": "Point", "coordinates": [lon, lat]}
        if geom is None:
            continue
        feats.append({"type": "Feature", "geometry": geom, "properties": props})

    if not feats:
        return None
    path = output_dir / "review_map.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    _log.info("QGIS review layer: %s (%d features; open in QGIS, orange=needs-input)",
              path, len(feats))
    return path
