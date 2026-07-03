"""Stage-agnostic disposition adapter for the runtime agent layer.

The agent layer must verify / build worklists for BOTH pipeline stages that emit a
per-plot disposition:

  * the NEW M2 ``m2_club`` -> ``list[ClubResult]`` (FMB DXFs only, no surveyor; the
    footprint is an in-memory shapely Polygon on ``placement``), and
  * the EXISTING surveyor-matching M3 ``m2_georef`` -> ``list[GeorefResult]`` (matched
    against the surveyor RAW DATA FILE; the footprint is read back from the output DXF).

Rather than fork every agent per stage, both result types are normalized to ONE
``PlotDisposition``. An agent then reasons over a uniform list and never needs to know
which stage produced it. THE HARD RULE is unchanged: a disposition is READ-ONLY here --
the adapter never promotes a recommendation, it only mirrors what the math gates already
decided (with one EXCEPTION wired explicitly for VerificationAgent: a DEMOTE-only mark,
ACCEPT -> REVIEW, never the reverse).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)

# The recommendations that count as a confident (math-gate ACCEPTed) placement, across
# both stages. ACCEPT_CADASTRAL is the legacy m2_georef cadastral-accept label.
CONFIDENT = ("ACCEPT", "ACCEPT_SEEDED", "ACCEPT_CADASTRAL")

# The four valid disposition states every plot must land in (nothing lost).
VALID_STATES = ("ACCEPT", "ACCEPT_SEEDED", "ACCEPT_CADASTRAL", "REVIEW", "NO_COVERAGE")


@dataclass
class PlotDisposition:
    """A uniform, stage-agnostic view of one plot's placement outcome.

    Built by ``from_club_result`` / ``from_georef_result``. ``footprint`` is resolved
    EAGERLY (from the in-memory placement for M2, or by reading the output DXF for M3)
    so downstream agents never branch on the source stage. ``source`` records which
    raw result it came from; ``raw`` keeps a handle to it so a DEMOTE can be reflected
    back onto the original object the pipeline/orchestrator holds.
    """
    survey: str
    recommendation: str
    method: str = ""
    footprint: object | None = None        # shapely Polygon (UTM metres) or None
    confidence: float = 0.0
    output_file: str = ""
    note: str = ""
    # --- context the existing agents already rely on (kept so nothing regresses) ---
    m1_file: str = ""
    error: str = ""
    scale: float = float("nan")
    area_ratio: float = float("nan")       # cadastral size match (M1 area / parcel area)
    chain_coverage: float = float("nan")   # surveyor-trace coverage (geometric path)
    cad_residual: float = float("inf")     # cadastral rigid-fit corner residual (m)
    n_inliers: int = 0
    n_corners: int = 0
    seed_ok: bool = True
    verify_passed: bool | None = None      # 7-gate verify result (geometric), if any
    source: str = ""                       # "club" | "georef"
    raw: object | None = field(default=None, repr=False)

    @property
    def is_confident(self) -> bool:
        return self.recommendation in CONFIDENT

    @property
    def has_geometry(self) -> bool:
        fp = self.footprint
        try:
            return fp is not None and fp.is_valid and fp.area > 0
        except Exception:  # noqa: BLE001
            return False

    def demote(self, note: str) -> None:
        """DEMOTE-ONLY: a confident plot -> REVIEW. Reflects onto the raw result so the
        orchestrator's deliverable set updates too. NEVER promotes (asserted)."""
        if self.recommendation not in CONFIDENT:
            return
        self.recommendation = "REVIEW"
        self.note = (self.note + "; " if self.note else "") + note
        if self.raw is not None:
            try:
                self.raw.recommendation = "REVIEW"
                prev = getattr(self.raw, "note", "")
                self.raw.note = (prev + "; " if prev else "") + note
            except Exception as exc:  # noqa: BLE001
                # FP-SAFETY: if the mirror-write fails, the disposition says REVIEW
                # while the raw result still says ACCEPT -- the orchestrator would
                # ship a plot an invariant just demoted. That divergence must be
                # LOUD, never swallowed.
                _log.warning(
                    "PlotDisposition.demote mirror-write FAILED for survey=%s "
                    "(disposition is REVIEW; raw still says %s). The deliverable "
                    "set would silently diverge from the verified set: %r",
                    self.survey, getattr(self.raw, "recommendation", "?"), exc)
                raise


def is_cross_village(d, village: str | None) -> bool:
    """Single source of truth for the cross-village check (drift between the three
    per-agent copies was an FP loophole -- one copy could silently diverge).

    True only on a clear FMB-village vs cadastre-village mismatch. Conservative: if
    the village can't be parsed (synthetic fixtures, no surveyor context) or the
    pipeline parser is unavailable, returns False so nothing is spuriously rejected.
    Delegates to the m2_georef parser, which owns the filename convention.
    """
    if not village or not getattr(d, "m1_file", ""):
        return False
    try:
        from ..pipeline.m2_georef.pipeline import _is_cross_village as _xv
    except Exception:  # noqa: BLE001
        return False
    try:
        return bool(_xv(d.m1_file, village))
    except Exception:  # noqa: BLE001
        return False


# ----------------------------------------------------------------- adapters ------
def from_club_result(r) -> PlotDisposition:
    """Normalize a new-M2 ``ClubResult``. Footprint comes from the in-memory placement."""
    fp = None
    scale = float("nan")
    area_ratio = float("nan")
    cad_resid = float("inf")
    seed_ok = True
    p = getattr(r, "placement", None)
    if p is not None:
        try:
            fp = p.footprint()
        except Exception:  # noqa: BLE001
            fp = None
        scale = float(getattr(p, "scale", float("nan")))
        area_ratio = float(getattr(p, "area_ratio", float("nan")))
        cad_resid = float(getattr(p, "rot_residual", float("inf")))
        seed_ok = bool(getattr(p, "seed_ok", True))
    return PlotDisposition(
        survey=r.survey_number or "?",
        recommendation=r.recommendation,
        method=r.method or "",
        footprint=fp,
        confidence=float(getattr(r, "confidence", 0.0) or 0.0),
        output_file=getattr(r, "output_file", "") or "",
        note=getattr(r, "note", "") or "",
        m1_file=getattr(r, "m1_file", "") or "",
        error=getattr(r, "error", "") or "",
        scale=scale,
        area_ratio=area_ratio,
        cad_residual=cad_resid,
        seed_ok=seed_ok,
        source="club",
        raw=r,
    )


def from_georef_result(r) -> PlotDisposition:
    """Normalize an M3 ``GeorefResult``. Footprint is read back from the output DXF.

    NOTE on the m2_georef convention: ``chain_coverage`` holds the surveyor-trace
    coverage for the geometric path but stores area_ratio for the cadastral path; we
    surface BOTH views so a stage-agnostic agent can read whichever applies by method.
    """
    method = getattr(r, "match_method", "") or ""
    cov = float(getattr(r, "chain_coverage", 0.0) or 0.0)
    fp = _georef_footprint(getattr(r, "output_file", "") or "")
    vr = getattr(r, "verify_result", None)
    verify_passed = None if vr is None else bool(getattr(vr, "all_passed", False))
    # cadastral path overloads chain_coverage as area_ratio; geometric uses it as coverage.
    if method.startswith("cadastral"):
        area_ratio, chain_cov = cov, float("nan")
    else:
        area_ratio, chain_cov = float("nan"), cov
    return PlotDisposition(
        survey=r.survey_number or "?",
        recommendation=r.recommendation,
        method=method,
        footprint=fp,
        confidence=float(getattr(r, "confidence", 0.0) or 0.0),
        output_file=getattr(r, "output_file", "") or "",
        note=getattr(r, "error", "") or "",
        m1_file=getattr(r, "m1_file", "") or "",
        error=getattr(r, "error", "") or "",
        area_ratio=area_ratio,
        chain_coverage=chain_cov,
        cad_residual=float(getattr(r, "cad_residual", float("inf"))),
        n_inliers=int(getattr(r, "n_inliers", 0) or 0),
        n_corners=int(getattr(r, "n_corners", 0) or 0),
        verify_passed=verify_passed,
        source="georef",
        raw=r,
    )


def _georef_footprint(output_file: str):
    if not output_file:
        return None
    try:
        from ..pipeline.m2_georef.pipeline import _footprint_polygon
    except Exception:  # noqa: BLE001
        return None
    try:
        fp = _footprint_polygon(output_file)
    except Exception:  # noqa: BLE001
        return None
    if fp is not None and fp.is_valid and fp.area > 0:
        return fp
    return None


def _looks_like_club(r) -> bool:
    # ClubResult carries `.placement` and `.method`; GeorefResult carries `.match_method`.
    return hasattr(r, "placement") and hasattr(r, "method") and not hasattr(r, "match_method")


def normalize(results) -> list[PlotDisposition]:
    """Coerce a heterogeneous results list into ``list[PlotDisposition]``.

    Accepts a list already of ``PlotDisposition`` (returned unchanged), or of
    ``ClubResult`` / ``GeorefResult`` (each adapted), so an agent can call this once at
    the top of ``run`` and treat both stages identically. Unknown/odd objects are
    skipped with a warning rather than crashing a job.
    """
    out: list[PlotDisposition] = []
    for r in results or []:
        if isinstance(r, PlotDisposition):
            out.append(r)
        elif _looks_like_club(r):
            out.append(from_club_result(r))
        elif hasattr(r, "match_method"):
            out.append(from_georef_result(r))
        else:
            _log.warning("normalize: skipping unrecognized result %r", type(r))
    return out
