"""VISUAL QA overlay for the CLUBBED M2 output -- the human-auditable FP backstop.

The deterministic gates (verify.py) verify each clubbed placement NUMERICALLY, but a
plot can pass every gate and still sit in the wrong place; only an eye (or an
independent authoritative reference) sees that. This renders ONE image of the whole
clubbed set so a person confirms placement in seconds.

WHAT IT DRAWS (UTM metres, equal aspect):
  * each PLACED (ACCEPT/ACCEPT_SEEDED) FMB footprint GREEN, filled faintly, labelled
    survey#.
  * each REVIEW footprint ORANGE (located, awaiting human confirmation).
  * when a ``cadastral_source`` is given, each plot's authoritative parcel PINK
    (dashed) -- where the FMB SHOULD be -- with a RED "MISPLACED" connector + marker
    when a placed centroid is > ``misplace_flag_m`` from its parcel centroid.

Offline + matplotlib Agg (no display); geometry comes from the in-memory ClubResult
placements, so it runs without any real data and never changes a placement. Mirrors
``m2_georef.qa_render.render_qa_overlay``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

_log = logging.getLogger(__name__)

_PLACED = ("ACCEPT", "ACCEPT_SEEDED")

_GREEN = "#2e7d32"      # placed / confident
_ORANGE = "#ef6c00"     # review
_PINK = "#e91e63"       # authoritative cadastral parcel
_RED = "#d50000"        # misplacement flag


def disposition_color(rec: str) -> str:
    if rec in _PLACED:
        return _GREEN
    if rec == "REVIEW":
        return _ORANGE
    return "#9e9e9e"       # NO_COVERAGE / staged -- not drawn as a footprint


def misplaced(placed_centroid, parcel_centroid, thresh_m: float) -> tuple[bool, float]:
    """Is a placed footprint too far from its authoritative parcel? -> (flag, dist_m)."""
    d = math.hypot(placed_centroid[0] - parcel_centroid[0],
                   placed_centroid[1] - parcel_centroid[1])
    return d > thresh_m, d


def _ring_xy(poly):
    x, y = poly.exterior.xy
    return list(x), list(y)


def render_club_qa(
    results,
    output_path: str | Path,
    cadastral_source=None,
    crs: str = "EPSG:32643",
    *,
    title: str = "",
    misplace_flag_m: float = 100.0,
) -> Path | None:
    """Render the clubbed-output QA overlay; returns the PNG path (or None).

    ``results`` = the job's ``ClubResult`` list. ``cadastral_source`` (optional) supplies
    each plot's authoritative parcel by survey number via ``.get(survey).polygon``.
    Runs headless (Agg) and on synthetic data.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception as exc:  # noqa: BLE001 - QA image is best-effort
        _log.warning("matplotlib unavailable (%s); skipping club QA overlay", exc)
        return None

    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(15, 15))

    placed_count = 0
    review_count = 0
    n_flagged = 0
    any_geom = False

    for r in results:
        sn = r.survey_number
        rec = r.recommendation

        # 1) authoritative parcel (where the FMB SHOULD be), if a cadastre is given.
        parcel = None
        if cadastral_source is not None:
            try:
                cp = cadastral_source.get(sn)
                parcel = cp.polygon if cp is not None else None
            except Exception:  # noqa: BLE001
                parcel = None
        if parcel is not None and getattr(parcel, "is_valid", False) and parcel.area > 0:
            any_geom = True
            px, py = _ring_xy(parcel)
            ax.plot(px, py, color=_PINK, lw=1.0, ls="--", zorder=2)
            ax.fill(px, py, color=_PINK, alpha=0.06, zorder=1)

        # 2) the placed/review footprint.
        if rec not in _PLACED + ("REVIEW",) or r.placement is None:
            continue
        fp = r.placement.footprint()
        if fp is None or not fp.is_valid or fp.area <= 0:
            continue
        any_geom = True
        color = disposition_color(rec)
        if rec in _PLACED:
            placed_count += 1
        else:
            review_count += 1

        fx, fy = _ring_xy(fp)
        ax.fill(fx, fy, color=color, alpha=0.18, zorder=3)
        ax.plot(fx, fy, color=color, lw=1.8, zorder=4)
        c = fp.centroid
        ax.text(c.x, c.y, f"{sn}", ha="center", va="center",
                fontsize=8, zorder=6, color="#000")

        # 3) THE FP FLAG: placed footprint far from its authoritative parcel.
        if parcel is not None and getattr(parcel, "is_valid", False) and parcel.area > 0:
            pc = parcel.centroid
            flag, d = misplaced((c.x, c.y), (pc.x, pc.y), misplace_flag_m)
            if flag:
                n_flagged += 1
                ax.plot([c.x, pc.x], [c.y, pc.y], color=_RED, lw=1.6, zorder=5)
                ax.scatter([c.x], [c.y], marker="X", s=90, color=_RED, zorder=7)
                ax.text(c.x, c.y, "  MISPLACED", color=_RED, fontsize=8,
                        fontweight="bold", zorder=7)

    ax.set_aspect("equal", adjustable="datalim")
    if not any_geom:
        # Nothing to draw (no placements, no cadastre) -- keep a valid, non-empty image.
        ax.text(0.5, 0.5, "no placed plots", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="#9e9e9e")
    ax.set_title(
        (title or "Clubbed M2 QA overlay")
        + f"   |   placed {placed_count}   review {review_count}   "
        f"misplaced-flags {n_flagged}", fontsize=13)
    ax.set_xlabel("UTM Easting (m)")
    ax.set_ylabel("UTM Northing (m)")
    ax.grid(True, ls=":", alpha=0.4)

    legend = [
        Patch(fc=_GREEN, alpha=0.4, label="ACCEPT (placed)"),
        Patch(fc=_ORANGE, alpha=0.4, label="REVIEW (confirm)"),
        plt.Line2D([0], [0], color=_PINK, ls="--", label="cadastral parcel"),
        plt.Line2D([0], [0], color=_RED, lw=2, label="MISPLACED flag"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _log.info("Club QA overlay: %s (placed=%d, review=%d, misplaced-flags=%d)",
              output_path.name, placed_count, review_count, n_flagged)
    return output_path
