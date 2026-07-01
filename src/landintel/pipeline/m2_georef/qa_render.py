"""VISUAL QA overlay -- the human-auditable false-positive backstop for a georef job.

The deterministic gates verify each placement NUMERICALLY. But a plot can pass every gate and
still be sitting in the wrong place -- the math cannot see that, only an eye (or an independent
authoritative reference) can. "Numerically verified -> done" is exactly how a false positive
ships. This renders ONE image per raw-data-file job so a person confirms placement in seconds.

WHAT IT DRAWS (everything in UTM metres, equal aspect):
  * the REFERENCE FRAME (authoritative, independent of the matcher): the surveyor's traced
    SITE DATA LINE (from the raw data DXF) AND each plot's S3/cadastral parcel pulled by survey
    number. These are where each FMB SHOULD be.
  * each placed FMB footprint, coloured by disposition (green ACCEPT / amber REVIEW / grey
    not-placed), filled faintly.
  * THE FP FLAG: for every placed plot we draw its own authoritative parcel; if the placed
    footprint sits far from that parcel (centroid gap > ``misplace_flag_m``), a RED connector +
    "!" is drawn -- a misplacement the gates rubber-stamped jumps straight out.
  * footprint OVERLAPS between placed plots are hatched red (real parcels tile, never overlap).
  * each plot labelled survey# + coverage%/residual.

Offline + matplotlib (lazy import; degrades to a logged no-op if matplotlib is absent). Geometry
is read from the already-written georef DXFs, so this never changes a placement.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

_log = logging.getLogger(__name__)

_ACCEPT = ("ACCEPT", "ACCEPT_SEEDED", "ACCEPT_CADASTRAL")


def disposition_color(rec: str) -> str:
    if rec in _ACCEPT:
        return "#2e7d32"          # green = georeferenced/confident
    if rec == "REVIEW":
        return "#ef6c00"          # amber = human confirms
    return "#9e9e9e"              # grey = not placed (NO_COVERAGE / staged)


def misplaced(placed_centroid, parcel_centroid, thresh_m: float) -> tuple[bool, float]:
    """Is a placed footprint too far from its authoritative parcel? -> (flag, distance_m)."""
    d = math.hypot(placed_centroid[0] - parcel_centroid[0],
                   placed_centroid[1] - parcel_centroid[1])
    return d > thresh_m, d


def footprint_overlaps(footprints: dict, min_frac: float = 0.05) -> list[tuple]:
    """Pairs of placed footprints whose interiors overlap (real parcels only share edges).

    ``footprints`` = {survey_number: shapely Polygon}. Returns
    [(sn_a, sn_b, intersection_polygon, overlap_fraction), ...] above ``min_frac``."""
    out = []
    items = [(sn, p) for sn, p in footprints.items() if p is not None and p.is_valid and p.area > 0]
    for i in range(len(items)):
        sn_a, pa = items[i]
        for j in range(i + 1, len(items)):
            sn_b, pb = items[j]
            if not pa.intersects(pb):
                continue
            inter = pa.intersection(pb)
            if inter.area <= 0:
                continue
            frac = inter.area / max(min(pa.area, pb.area), 1e-9)
            if frac > min_frac:
                out.append((sn_a, sn_b, inter, frac))
    return out


def _ring_xy(poly):
    x, y = poly.exterior.xy
    return list(x), list(y)


def render_qa_overlay(
    results,
    output_path: str | Path,
    *,
    surveyor=None,
    cadastral_source=None,
    title: str = "",
    misplace_flag_m: float = 100.0,
    overlap_frac: float = 0.05,
) -> Path | None:
    """Render the visual QA overlay for one georef job; returns the PNG path (or None).

    ``results`` = the job's GeorefResult list. ``surveyor`` (SurveyorData) supplies the traced
    reference lines; ``cadastral_source`` (S3 / vector) supplies each plot's authoritative parcel
    by survey number. Both optional -- whatever is available is drawn.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception as exc:  # noqa: BLE001 - QA image is best-effort
        _log.warning("matplotlib unavailable (%s); skipping QA overlay", exc)
        return None

    from .pipeline import _footprint_polygon

    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(15, 15))

    # 1) surveyor traced reference lines (independent ground truth).
    if surveyor is not None and getattr(surveyor, "polylines", None):
        for pl in surveyor.polylines:
            pts = getattr(pl, "raw_points", None)
            if pts and len(pts) >= 2:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color="#bdbdbd", lw=0.8, zorder=1)

    placed_footprints: dict = {}
    n_flagged = 0
    by_disp: dict[str, int] = {}

    # 2) per-plot: authoritative parcel + placed footprint + the misplacement flag.
    for r in results:
        sn = r.survey_number
        rec = r.recommendation
        by_disp[rec] = by_disp.get(rec, 0) + 1
        color = disposition_color(rec)

        parcel = None
        if cadastral_source is not None:
            try:
                cp = cadastral_source.get(sn)
                parcel = cp.polygon if cp is not None else None
            except Exception:  # noqa: BLE001
                parcel = None
        if parcel is not None and parcel.is_valid and parcel.area > 0:
            px, py = _ring_xy(parcel)
            ax.plot(px, py, color="#1565c0", lw=1.0, ls="--", zorder=2)  # expected location

        fp = None
        if rec in _ACCEPT + ("REVIEW",) and getattr(r, "output_file", ""):
            try:
                fp = _footprint_polygon(r.output_file)
            except Exception:  # noqa: BLE001
                fp = None
        if fp is not None and fp.is_valid and fp.area > 0:
            placed_footprints[sn] = fp
            fx, fy = _ring_xy(fp)
            ax.fill(fx, fy, color=color, alpha=0.18, zorder=3)
            ax.plot(fx, fy, color=color, lw=1.8, zorder=4)
            c = fp.centroid
            ax.text(c.x, c.y, f"{sn}\n{100 * r.chain_coverage:.0f}% / "
                    f"{_fmt(r.field_residual_max)}", ha="center", va="center",
                    fontsize=7, zorder=6, color="#000")

            # THE FP FLAG: placed footprint far from its authoritative parcel.
            if parcel is not None and parcel.area > 0:
                pc = parcel.centroid
                flag, d = misplaced((c.x, c.y), (pc.x, pc.y), misplace_flag_m)
                if flag:
                    n_flagged += 1
                    ax.plot([c.x, pc.x], [c.y, pc.y], color="#d50000", lw=1.6, zorder=5)
                    ax.scatter([c.x], [c.y], marker="X", s=90, color="#d50000", zorder=7)
                    ax.text(c.x, c.y, "  !MISPLACED?", color="#d50000", fontsize=8,
                            fontweight="bold", zorder=7)
        elif parcel is not None and parcel.area > 0:
            # not placed (NO_COVERAGE) -> show WHERE it should be, faintly.
            px, py = _ring_xy(parcel)
            ax.fill(px, py, color="#9e9e9e", alpha=0.10, zorder=2)

    # 3) footprint overlaps (placed parcels must tile, not overlap).
    overlaps = footprint_overlaps(placed_footprints, overlap_frac)
    for sn_a, sn_b, inter, frac in overlaps:
        try:
            polys = inter.geoms if hasattr(inter, "geoms") else [inter]
            for g in polys:
                if g.area > 0 and hasattr(g, "exterior"):
                    gx, gy = _ring_xy(g)
                    ax.fill(gx, gy, facecolor="none", edgecolor="#d50000",
                            hatch="xxx", lw=1.0, zorder=6)
        except Exception:  # noqa: BLE001
            pass

    n_placed = sum(by_disp.get(k, 0) for k in _ACCEPT)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(
        (title or "QA overlay") +
        f"   |   placed {n_placed}/{len(results)}   "
        f"misplaced-flags {n_flagged}   overlaps {len(overlaps)}", fontsize=13)
    ax.set_xlabel("UTM Easting (m)"); ax.set_ylabel("UTM Northing (m)")
    ax.grid(True, ls=":", alpha=0.4)

    legend = [
        Patch(fc="#2e7d32", alpha=0.4, label="ACCEPT (georeferenced)"),
        Patch(fc="#ef6c00", alpha=0.4, label="REVIEW (confirm)"),
        Patch(fc="#9e9e9e", alpha=0.4, label="not placed (NO_COVERAGE)"),
        plt.Line2D([0], [0], color="#1565c0", ls="--", label="authoritative parcel (S3/cadastral)"),
        plt.Line2D([0], [0], color="#bdbdbd", label="surveyor traced line"),
        plt.Line2D([0], [0], color="#d50000", lw=2, label="MISPLACED flag / overlap"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _log.info("QA overlay: %s (placed=%d, misplaced-flags=%d, overlaps=%d)",
              output_path.name, n_placed, n_flagged, len(overlaps))
    return output_path


def _fmt(v: float) -> str:
    return "-" if v is None or v != v or v == float("inf") else f"{v:.1f}m"
