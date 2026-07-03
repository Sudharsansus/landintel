"""Shared placement type for the new M2 (FMB-only georeference + club).

Every M2 method (cadastral seat, GPS seat, relative-club propagation) returns a
``CandidatePlacement``: a RIGID similarity (rotation + uniform scale + translation)
that maps the FMB's relative-metre geometry into absolute UTM, plus the inputs the
0-FP cross-check needs (whether the method's own strict gate passed, area ratio,
residual, scale). The pipeline collects candidates per FMB and lets the math gates
decide ACCEPT -- no method ACCEPTs on its own say-so.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CandidatePlacement:
    """One method's proposed absolute placement of an FMB.

    ``adjusted`` holds UTM positions for ALL M1 stones (so the corner ring and any
    interior stones are placed); the FMB shape is preserved exactly (rigid only).
    """
    method: str                     # "cadastral" | "gps_seed" | "propagated"
    R: np.ndarray
    s: float
    t: np.ndarray
    adjusted: np.ndarray            # (N, 2) UTM positions for ALL stones
    corner_ring: list[int]          # outer_stone_indices (ordered)
    passes_gate: bool               # the method's OWN strict gate passed
    area_ratio: float = float("nan")
    rot_residual: float = float("inf")
    scale: float = 1.0
    seed_ok: bool = True
    note: str = ""
    # Stone-match confidence (client's conditional "5 stones" bar): how many placed
    # corners coincide with target stones, out of min(FULL_MATCH_STONES, n_corners).
    # A CONFIDENCE LABEL, never a placement blocker (club-all M2 places everything).
    n_stone_matched: int = 0
    stones_required: int = 0
    full_stone_match: bool = False

    def corner_points(self) -> np.ndarray:
        """(M, 2) placed corner-ring positions in UTM."""
        ring = [i for i in self.corner_ring if 0 <= i < len(self.adjusted)]
        if len(ring) < 3:
            return self.adjusted
        return self.adjusted[np.array(ring)]

    def footprint(self):
        """Closed shapely Polygon of the placed corner ring (or None)."""
        from shapely.geometry import Polygon
        pts = self.corner_points()
        if len(pts) < 3:
            return None
        poly = Polygon([(float(x), float(y)) for x, y in pts])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            return None
        return poly

    def centroid(self) -> tuple[float, float]:
        pts = self.corner_points()
        c = pts.mean(axis=0)
        return (float(c[0]), float(c[1]))


@dataclass
class ClubResult:
    """Per-FMB outcome of the new M2 (georeference + club)."""
    m1_file: str
    survey_number: str
    recommendation: str = "NO_COVERAGE"   # ACCEPT | ACCEPT_SEEDED | REVIEW | NO_COVERAGE
    method: str = ""                       # winning method
    output_file: str = ""
    placement: CandidatePlacement | None = None
    candidates: dict[str, CandidatePlacement] = field(default_factory=dict)
    corroborated_by: list[str] = field(default_factory=list)
    confidence: float = 0.0
    note: str = ""
    error: str = ""

    @property
    def placed(self) -> bool:
        return self.recommendation in ("ACCEPT", "ACCEPT_SEEDED")
