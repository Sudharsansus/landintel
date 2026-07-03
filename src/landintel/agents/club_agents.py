"""Agentized M2 club -- each agent owns ONE job and VERIFIES its own output before it
hands off, so an error is caught at the stage it happens instead of surfacing as scatter
at the end. This is the decomposition of the old monolithic club into focused workers.

0-FP is preserved exactly: the agents only SEQUENCE, MEASURE and NARRATE; the math gates
inside the focused modules (``cadastral_seat`` rigid+seat gate, ``verify``) remain the sole
arbiters of ACCEPT/REVIEW. No agent promotes a placement.

The goal the agents optimise for is the client's: **the clubbed FMB village must overlay
the TNGIS/cadastre parcels**. So the load-bearing agent is ``TngisOverlayAgent`` -- it
MEASURES that overlay (per-plot IoU of the placed FMB footprint against its own cadastral
parcel) and the plot-vs-plot overlaps, giving the deliverable an objective score instead of
a subjective eyeball.

Agents here are plain objects (not Qwen) so they are deterministic and testable on synthetic
geometry; the Qwen ``PipelineOrchestrator`` sequences the village-level stages above them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..pipeline.m2_club.disposition_thresholds import TILING_OVERLAP_THRESHOLD


def overlay_gate(recommendation: str, iou: float | None, *,
                 seated: bool, has_placement: bool, in_demote: bool,
                 is_vector_parcel: bool, contested: bool,
                 containment: float | None = None, area_factor: float | None = None,
                 decisive_lock: bool = False,
                 iou_accept: float = 0.5, iou_strong: float = 0.6,
                 iou_contradict: float = 0.15,
                 contain_min: float = 0.90, factor_max: float = 2.5) -> tuple[str, str | None]:
    """0-FP TNGIS-overlay disposition for ONE plot. Pure + testable; returns
    ``(recommendation, reason)``. This is the single arbiter used by run_m2_cad so the
    false-positive discipline is locked by tests rather than living inline in a script.

    Rules (a plot's overlay IoU is against its OWN survey#'s parcel):
      * DEMOTE  ACCEPT -> REVIEW  when the parcel is a REAL vector ring but IoU < iou_contradict
        (self-contradiction: placed away from the parcel carrying its number, e.g. a mislabel).
      * PROMOTE non-ACCEPT -> ACCEPT (only if it has a placement and was not demoted for overlap)
        when ANY of:
          STRONG:     IoU >= iou_strong AND real vector ring AND label uncontested, OR
          SEATED:     IoU >= iou_accept AND not flagged off-seat, OR
          SAME-PARCEL: the village lock is DECISIVE and the plot's exact same-numbered vector
                       parcel is near-totally CONTAINED in the placed footprint (or vice versa,
                       ``containment >= contain_min``) within a ``<= factor_max`` area factor.
                       This is the FMB-parent / cadastre-subdivision case: one parcel nested in
                       the other of the same number = same land, correct LOCATION, only the
                       EXTENT diverged (flagged in the divergence report). IoU alone under-counts
                       it because the size difference caps IoU at 1/factor.
      * otherwise unchanged.

    0-FP: every promotion needs genuine overlap with the plot's OWN parcel. STRONG needs an
    uncontested vector parcel; SAME-PARCEL needs a DECISIVE village lock plus near-total mutual
    containment within a bounded size factor -- a wrong/renumbered parcel is neither contained nor
    within-factor (e.g. a 10x or 0.03x size parcel fails ``factor_max``). Demotion only removes."""
    if recommendation == "ACCEPT":
        if is_vector_parcel and iou is not None and iou < iou_contradict:
            return "REVIEW", f"IoU-contradiction {iou:.2f} vs own parcel"
        return "ACCEPT", None
    if in_demote or not has_placement:
        return recommendation, None
    v = iou or 0.0
    strong = (v >= iou_strong and is_vector_parcel and not contested)
    seated_ok = (v >= iou_accept and seated)
    same_parcel = (decisive_lock and is_vector_parcel and not contested
                   and containment is not None and containment >= contain_min
                   and area_factor is not None and area_factor <= factor_max)
    if seated_ok:
        return "ACCEPT", f"IoU-gate: overlays TNGIS {v:.2f}"
    if strong:
        return "ACCEPT", f"IoU-gate: strong/uncontested overlay {v:.2f}"
    if same_parcel:
        return "ACCEPT", (f"IoU-gate: same parcel: cadastre {containment:.0%} contained in "
                          f"footprint (extent diverged x{area_factor:.2f}, IoU {v:.2f})")
    return recommendation, None


@dataclass
class AgentResult:
    """Uniform hand-off: what the agent produced, whether its own check passed, a 0-1
    confidence, and any issues it wants the next agent / the operator to see."""
    name: str
    ok: bool
    data: Any = None
    confidence: float = 0.0
    issues: list[str] = field(default_factory=list)


def _ring(poly_pts) -> Polygon | None:
    if poly_pts is None or len(poly_pts) < 3:
        return None
    p = Polygon([(float(x), float(y)) for x, y in poly_pts])
    if not p.is_valid:
        p = p.buffer(0)
    return p if (p.geom_type == "Polygon" and p.area > 0) else None


class ParcelAgent:
    """ONE job: turn the located cadastral source into a clean survey# -> parcel-polygon
    map, and FLAG parcels that are implausible (sliver / merged super-parcel) so the match
    stage is not fed garbage. It never edits geometry -- only reports a per-survey health
    score -- so it cannot introduce a false placement."""

    name = "ParcelAgent"

    def run(self, cadastral_source, surveys: set[str]) -> AgentResult:
        parcels: dict[str, Polygon] = {}
        areas: list[float] = []
        for sv in surveys:
            p = cadastral_source.get(sv) if cadastral_source else None
            poly = _ring(list(p.polygon.exterior.coords)) if (p and p.polygon) else None
            if poly is not None:
                parcels[sv] = poly
                areas.append(poly.area)
        if not areas:
            return AgentResult(self.name, ok=False, issues=["no cadastral parcels located"])
        areas.sort()
        med = areas[len(areas) // 2]
        # flag slivers (<0.1x median) and merged super-parcels (>4x median) as low-trust.
        flagged = [sv for sv, poly in parcels.items()
                   if poly.area < 0.1 * med or poly.area > 4.0 * med]
        conf = 1.0 - len(flagged) / max(1, len(parcels))
        issues = [f"{len(flagged)} parcel(s) implausibly sized (sliver/merged): "
                  f"{sorted(flagged, key=lambda s: int(s) if s.isdigit() else 0)}"] if flagged else []
        return AgentResult(self.name, ok=True, data={"parcels": parcels, "median_area": med,
                                                     "flagged": set(flagged)},
                           confidence=conf, issues=issues)


class TngisOverlayAgent:
    """ONE job: MEASURE how well the clubbed FMB overlays the TNGIS/cadastre -- the client's
    actual acceptance criterion. For each placed plot it computes IoU(placed FMB footprint,
    its own cadastral parcel); it also measures plot-vs-plot overlap (real parcels tile, they
    must not overlap). Returns the per-plot IoU, the mean, and the overlap area. Pure
    measurement -> 0-FP by construction."""

    name = "TngisOverlayAgent"

    def run(self, placed_footprints: dict[str, Polygon],
            parcels: dict[str, Polygon]) -> AgentResult:
        ious: dict[str, float] = {}
        for sv, fp in placed_footprints.items():
            par = parcels.get(sv)
            if fp is None or par is None:
                continue
            inter = fp.intersection(par).area
            union = fp.union(par).area
            ious[sv] = (inter / union) if union > 0 else 0.0
        mean_iou = sum(ious.values()) / len(ious) if ious else 0.0
        # plot-vs-plot overlap: sum of pairwise intersection area (should be ~0 for a tiling)
        fps = [f for f in placed_footprints.values() if f is not None]
        overlap = 0.0
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                overlap += fps[i].intersection(fps[j]).area
        total = sum(f.area for f in fps) or 1.0
        overlap_frac = overlap / total
        issues = []
        if mean_iou < 0.4:
            issues.append(f"low TNGIS overlay: mean IoU {mean_iou:.2f}")
        if overlap_frac > 0.1:
            issues.append(f"plots overlap {overlap_frac*100:.0f}% (should tile, not stack)")
        return AgentResult(self.name, ok=mean_iou >= 0.4,
                           data={"iou": ious, "mean_iou": mean_iou,
                                 "overlap_frac": overlap_frac},
                           confidence=mean_iou, issues=issues)


class AssemblyAgent:
    """ONE job: keep the club a COHERENT TILING. Real parcels share edges, never interiors,
    so where two placed plots overlap by more than ``max_overlap_frac`` of the smaller, the
    LOWER-confidence plot of the pair is demoted from ACCEPT to REVIEW (never moved -- moving
    would fight the cadastre fit). This mirrors ``_resolve_footprint_conflicts`` but as an
    explicit, testable agent. It only DEMOTES, so it cannot create a false ACCEPT."""

    name = "AssemblyAgent"

    def run(self, footprints: dict[str, Polygon], accepted: set[str],
            confidence: dict[str, float] | None = None,
            max_overlap_frac: float = TILING_OVERLAP_THRESHOLD) -> AgentResult:
        confidence = confidence or {}
        demote: set[str] = set()
        acc = sorted(accepted)
        for i in range(len(acc)):
            for j in range(i + 1, len(acc)):
                a, b = acc[i], acc[j]
                fa, fb = footprints.get(a), footprints.get(b)
                if fa is None or fb is None:
                    continue
                inter = fa.intersection(fb).area
                if inter <= 0:
                    continue
                if inter > max_overlap_frac * min(fa.area, fb.area):
                    lo = a if confidence.get(a, 0.0) <= confidence.get(b, 0.0) else b
                    demote.add(lo)
        issues = [f"demoted {sorted(demote)} to REVIEW (footprint overlap -> not a tiling)"] \
            if demote else []
        return AgentResult(self.name, ok=True,
                           data={"demote": demote, "accept": accepted - demote},
                           confidence=1.0 - len(demote) / max(1, len(accepted)), issues=issues)
