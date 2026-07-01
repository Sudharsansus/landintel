"""Second pass: place REVIEW plots by schedule-anchored corridor propagation.

WHY a second pass: the first pass auto-places only the few corridor plots with
strong chain coverage. The rest find a >=4-inlier congruent subset but at the
WRONG corridor location -- verified on INGUR: the already-placed anchors are
MONOTONIC in schedule order along the corridor axis, while the REVIEW plots'
first-pass positions are SCRAMBLED (rank-vs-position correlation 0.21). A
near-congruent plot shape grabbed a foreign seat because nothing tied it to its
own neighbourhood.

THE FIX (sound, no false positives): the land schedule gives the ORDER of plots
along the corridor. Anchored on the already-placed plots, we fit corridor-position
<- schedule-rank, predict each REVIEW plot's window, and re-match it with the
geometric matcher RESTRICTED to that window (``allowed_stones``). A re-placement
is accepted only if it (a) matches with >=4 inliers at low residual INSIDE the
window, (b) lands at its predicted corridor position, and (c) does not overlap any
placed plot. Newly placed plots become anchors, so placement propagates outward
from the high-confidence core. Anything that still cannot be placed this way stays
REVIEW for the human -- never force-placed.

This does NOT use the working DXF for positions (it is a strip chart, not a
true-UTM cadastre) and does NOT use edge-length fingerprinting (discredited:
causes false positives). Identity comes only from the schedule + corridor order.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from .extract_m1 import M1PlotData
from .extract_surveyor import SurveyorData
from .match import geometric_match
from .output_dxf import write_georef_dxf
from .transform import cadastral_adjust, umeyama
from .verify import chain_coverage

_log = logging.getLogger(__name__)

# Half-width (metres, along the corridor axis) of the schedule-predicted window a
# REVIEW plot is re-matched within. Wide enough to absorb the linear-fit error,
# narrow enough that a foreign plot's seat is excluded.
_WINDOW = 450.0
# A re-placed plot must land within this of its predicted corridor position.
_POS_TOL = 450.0
# Interior overlap (of the smaller) above this = the two plots collide.
_OVERLAP = 0.20
_FIELD_WEIGHT = 1000.0
_DIST_WEIGHT = 1.0


def _axis(surveyor: SurveyorData):
    """Corridor centre + unit principal axis (PCA of the surveyor stones)."""
    P = surveyor.stone_positions
    c = P.mean(axis=0)
    axis = np.linalg.svd(P - c)[2][0]
    return c, axis


def _corner_polygon(m1: M1PlotData, adjusted: np.ndarray):
    ring = m1.outer_stone_indices
    if len(ring) < 3:
        return None
    try:
        return Polygon([(adjusted[i][0], adjusted[i][1]) for i in ring])
    except Exception:
        return None


def propagate_review_plots(
    surveyor: SurveyorData,
    results: list,                       # list[GeorefResult] (mutated)
    m1_data_map: dict[str, M1PlotData],
    corridor_order: list[str],           # survey numbers in corridor/schedule order
    output_dir: str | Path,
    crs: str = "EPSG:32643",
) -> list[str]:
    """Place REVIEW plots in their schedule-predicted corridor windows.

    Mutates ``results`` in place (upgrades placed plots to ACCEPT with
    match_method "propagated"). Returns the list of survey numbers upgraded.
    """
    output_dir = Path(output_dir)
    rank = {sn: i for i, sn in enumerate(corridor_order)}
    c, axis = _axis(surveyor)
    stone_proj = (surveyor.stone_positions - c) @ axis  # 1-D corridor coord/stone

    def proj_xy(x, y):
        return float((np.array([x, y]) - c) @ axis)

    # Already-placed anchors (first-pass ACCEPT) with corridor position + footprint.
    placed_pos: dict[str, float] = {}
    placed_poly: dict[str, Polygon] = {}
    for r in results:
        if r.recommendation == "ACCEPT" and r.output_file and r.survey_number in rank:
            m1 = m1_data_map.get(r.survey_number)
            if m1 is None:
                continue
            # Footprint from the already-written georef DXF boundary.
            try:
                import ezdxf
                msp = ezdxf.readfile(r.output_file).modelspace()
                pts = []
                for e in msp.query("LWPOLYLINE"):
                    if e.dxf.layer == "BOUNDARY":
                        pts = [(p[0], p[1]) for p in e.get_points()]
                        break
                if len(pts) >= 3:
                    poly = Polygon(pts)
                    placed_pos[r.survey_number] = proj_xy(poly.centroid.x, poly.centroid.y)
                    placed_poly[r.survey_number] = poly
            except Exception:
                continue

    review = [r for r in results
              if r.recommendation == "REVIEW" and r.survey_number in rank
              and r.survey_number in m1_data_map]
    if len(placed_pos) < 2 or not review:
        _log.info("Propagation: need >=2 anchors and >=1 review plot "
                  "(have %d, %d)", len(placed_pos), len(review))
        return []

    upgraded: list[str] = []
    surv_positions = np.array([[s.x, s.y] for s in surveyor.stones])

    progress = True
    while progress:
        progress = False
        # Linear corridor-position <- schedule-rank from the current anchor set.
        ranks = np.array([rank[sn] for sn in placed_pos])
        poss = np.array([placed_pos[sn] for sn in placed_pos])
        slope, intercept = np.polyfit(ranks, poss, 1)

        def rank_gap(r):
            return min(abs(rank[r.survey_number] - rank[p]) for p in placed_pos)

        for r in sorted(review, key=rank_gap):
            sn = r.survey_number
            if sn in placed_pos:
                continue
            m1 = m1_data_map[sn]
            pred = slope * rank[sn] + intercept
            mask = np.abs(stone_proj - pred) <= _WINDOW
            if int(mask.sum()) < 4:
                continue

            match = geometric_match(m1, surveyor, allowed_stones=mask)
            if not match.matched:
                continue
            matched_pairs = [(i, j) for i, j in enumerate(match.stone_map) if j >= 0]
            if len(matched_pairs) < 2:
                continue

            src = m1.stone_positions()[np.array([p[0] for p in matched_pairs])]
            dst = np.array([surveyor.stone_coords(p[1]) for p in matched_pairs])
            R, s, t = umeyama(src, dst)[:3]
            edge_pairs = [(e.stone_a, e.stone_b, e.length_m) for e in m1.outer_edges]
            adjusted = cadastral_adjust(
                m1_positions=m1.stone_positions(),
                surveyor_positions=surv_positions,
                matched_pairs=matched_pairs,
                edge_pairs=edge_pairs,
                field_weight=_FIELD_WEIGHT, dist_weight=_DIST_WEIGHT,
                umeyama_result=(R, s, t),
            )
            poly = _corner_polygon(m1, adjusted)
            if poly is None or not poly.is_valid or poly.area <= 0:
                continue
            newpos = proj_xy(poly.centroid.x, poly.centroid.y)

            # VALIDATION (the no-false-positive gate for the second pass):
            #  (a) lands at its schedule-predicted corridor position, and
            #  (b) does not overlap any already-placed plot.
            if abs(newpos - pred) > _POS_TOL:
                continue
            collide = False
            for psn, ppoly in placed_poly.items():
                if poly.intersects(ppoly):
                    ov = poly.intersection(ppoly).area / max(min(poly.area, ppoly.area), 1e-9)
                    if ov > _OVERLAP:
                        collide = True
                        break
            if collide:
                continue

            # Accept -> write, upgrade, become an anchor for the next round.
            out = output_dir / f"georef_{Path(r.m1_file).stem}.dxf"
            write_georef_dxf(
                m1_dxf_path=r.m1_file, output_path=out,
                adjusted_stone_positions=adjusted,
                original_stone_positions=m1.stone_positions(),
                stone_label_to_index={st.label: st.index for st in m1.stones},
                R=R, s=s, t=t, crs=crs, corner_ring=m1.outer_stone_indices,
            )
            ring_pts = [(float(adjusted[i][0]), float(adjusted[i][1]))
                        for i in m1.outer_stone_indices]
            segs = [(ring_pts[k], ring_pts[(k + 1) % len(ring_pts)])
                    for k in range(len(ring_pts))]
            r.matched = True
            r.output_file = str(out)
            r.n_inliers = match.n_matched_stones
            r.n_corners = len(m1.outer_stone_indices)
            r.chain_coverage = chain_coverage(
                segs, [pl.raw_points for pl in surveyor.polylines])
            r.recommendation = "ACCEPT"
            r.match_method = "propagated"
            placed_pos[sn] = newpos
            placed_poly[sn] = poly
            upgraded.append(sn)
            progress = True
            _log.info("Propagated %s -> ACCEPT (window @%.0f, pos %.0f, "
                      "inliers=%d, cov=%.0f%%)",
                      sn, pred, newpos, r.n_inliers, 100 * r.chain_coverage)

        review = [r for r in review if r.survey_number not in placed_pos]

    _log.info("Propagation complete: %d/%d REVIEW plots placed",
              len(upgraded), len(upgraded) + len(review))
    return upgraded
