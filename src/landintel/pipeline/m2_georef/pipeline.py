"""M2 Pipeline orchestrator -- end-to-end georeferencing.

Usage:
    python -m landintel.pipeline.m2_georef.pipeline \
        --surveyor /path/to/INGUR_RAW_DATA_FILE.dxf \
        --m1-dir /path/to/m1_outputs/ \
        --output-dir /path/to/georef_outputs/ \
        --crs EPSG:32643

Or programmatically:
    from landintel.pipeline.m2_georef import georef_pipeline
    results = georef_pipeline(surveyor_dxf, m1_dxf_paths, output_dir)
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .extract_m1 import extract_m1_dxf
from .extract_surveyor import SurveyorData, extract_surveyor
from .match import match_plot
from .output_dxf import build_full_combined_dxf, write_georef_dxf
from .self_calibrate import apply_calibrated_gate
from .transform import cadastral_adjust, seed_quality, umeyama
from .verify import (
    VerifyResult,
    chain_coverage,
    print_verify_result,
    verify_georef_dxf,
)

_log = logging.getLogger(__name__)

# Weight parameters for cadastral adjustment
FIELD_WEIGHT = 1000.0   # Hold surveyor positions nearly fixed
DIST_WEIGHT = 1.0       # Soft-pull FMB edge lengths

# Chain-coverage disposition gates: fraction of the georeferenced boundary lying
# on the surveyor's traced SITE DATA LINE. This is the INDEPENDENT ground-truth
# false-positive gate -- a stone-congruence match can be coincidental in a dense
# cloud, but only a truly-surveyed plot's boundary lies on the traced lines.
# INGUR separation is clean and wide: true matches 58-100%, coincidental 12-43%.
CHAIN_COVER_ACCEPT = 0.50   # >= this AND geometry sane -> ACCEPT (georeferenced)
CHAIN_COVER_REVIEW = 0.35   # [REVIEW, ACCEPT) -> human confirms; below -> NO_COVERAGE

# Two ACCEPT plots whose footprints overlap interiors by more than this are
# mutually exclusive (real parcels tile, sharing edges not interiors); the
# lower-coverage one is demoted. Safety net behind the chain-coverage gate.
FOOTPRINT_CONFLICT = 0.20

# Cadastral placement gates. S3 is a POSITION+ROTATION reference only (the M1 FMB
# geometry is placed RIGIDLY, never deformed to the S3 pixel boundary), so quality is
# judged by AREA RATIO (M1 plot area vs S3 parcel area) -- a size/identity match that
# never touches geometry -- NOT by IoU-against-the-raster-outline. ACCEPT_CADASTRAL
# requires ALL of: area ratio in band, the rigid corner-alignment residual is sane,
# near the corridor (rejects cross-village duplicate-survey mislocations), and no
# overlap with a placed plot. Otherwise REVIEW (located, human confirms).
CAD_AREA_LO = 0.65          # M1 area / parcel area lower bound (right-sized parcel)
CAD_AREA_HI = 1.55          # upper bound
CAD_ROT_RESID_MAX = 12.0    # rigid corner-alignment residual (m); gross misfit -> REVIEW
CAD_CORRIDOR_MAX = 500.0    # placement farther than this from any corridor stone = cross-village
# A1 scale gate: M1 is already real-world metres and the parcel is UTM metres, so a
# CORRECT placement aligns at scale ~1. Because ICP scales M1 to fit, area_ratio is
# ~1 post-scaling and cannot see a wrong-size parcel -- the leftover scale CAN. A
# wrong-parcel match (different aspect/size) needs s<0.85 or s>1.2 to fit. Independent
# of area_ratio, so it is the strongest off-corridor identity cross-check we have.
CAD_SCALE_LO = 0.80
CAD_SCALE_HI = 1.25
# Cross-corroboration: a corridor plot that matched REAL surveyor stones (authoritative
# geometry) but fell short of the chain-coverage ACCEPT gate (corridor-clipping: its
# boundary was only partly field-traced) is upgraded to ACCEPT if its placement lands
# near its INDEPENDENT S3 label point. Two unrelated sources (field stones + cadastral
# OCR) agreeing on position is FP-safe -- it substitutes for chain coverage, it does
# NOT loosen it. Label points sit ~25 m off the centroid, so the tol is generous but
# still well inside one parcel (neighbours are ~100 m apart).
CORROB_TOL = 50.0           # geometric-placement centroid <-> S3 label point (m)
CORROB_FIELD_RESID = 3.0    # matched corners must hit real stones this tightly (m)
CORROB_MIN_INLIERS = 5      # and there must be this many real-stone inliers
TOPO_MAX_RESID = 25.0       # topology upgrade requires the plot's own rigid fit to be
                            # this good (corner residual, m) -- a garbage fit that only
                            # grazes a neighbour by chance is a false positive, not a tile.


@dataclass
class GeorefResult:
    """Result of georeferencing a single M1 DXF."""
    m1_file: str
    survey_number: str
    matched: bool
    output_file: str = ""
    fingerprint_score: float = float("inf")
    neighborhood_score: float = float("inf")
    field_residual_mean: float = float("inf")
    field_residual_max: float = float("inf")
    edge_residual_mean: float = float("inf")
    error: str = ""
    verify_result: VerifyResult | None = None
    # Geometric-congruence confidence (the false-positive gate).
    n_corners: int = 0
    n_inliers: int = 0
    confidence: float = 0.0
    # Fraction of the georeferenced boundary lying on surveyor traced lines --
    # the INDEPENDENT ground-truth false-positive gate (see verify.chain_coverage).
    chain_coverage: float = 0.0
    recommendation: str = "REJECT"   # ACCEPT | REVIEW | NO_COVERAGE | REJECT
    match_method: str = ""
    cad_residual: float = float("inf")   # cadastral rigid-fit corner residual (m)


def georef_single(
    m1_dxf_path: str | Path,
    surveyor: SurveyorData,
    output_dir: str | Path,
    crs: str = "EPSG:32643",
    corridor_surveys: set[str] | None = None,
    cadastral_source: object | None = None,
) -> GeorefResult:
    """Georeference a single M1 DXF against the surveyor data.

    corridor_surveys : optional identity gate -- the set of survey numbers the
        corridor actually crosses (from the land schedule, see
        ``extract_corridor_surveys``). When given, a plot whose survey number is
        NOT in the set is rejected as NO_COVERAGE WITHOUT matching: the surveyor
        never crossed it, so any geometric congruence would be a false positive
        (a different plot's seat). This is the fix for geometry's inability to
        resolve identity on a dense corridor (measured: removes all 13 INGUR FPs).

    cadastral_source : optional cadastral reference (S3 tiles / client vector). Its
        ``label_point(survey)`` gives the plot's INDEPENDENT expected position, fed
        to the seat-locality FP guard in ``match_plot``/``geometric_match`` so a
        chance-congruent subset far from the true seat (measured 1.2-2.6 km on
        INGUR for 667/668/669/670/698/699) is rejected instead of becoming a
        wrong-seat placement. No-op when the source has no label for this plot.
    """
    m1_dxf_path = Path(m1_dxf_path)
    output_dir = Path(output_dir)
    result = GeorefResult(m1_file=str(m1_dxf_path), survey_number="", matched=False)

    try:
        _log.info("=" * 60)
        _log.info("Processing: %s", m1_dxf_path.name)
        t0 = time.time()

        # Step 1: Extract M1 data
        m1 = extract_m1_dxf(m1_dxf_path)
        result.survey_number = m1.survey_number

        # Identity gate: skip plots the corridor schedule says are NOT crossed.
        if corridor_surveys is not None and m1.survey_number not in corridor_surveys:
            result.error = (f"Survey {m1.survey_number} is not on the corridor "
                            f"land schedule -- not surveyed (no geometric match "
                            f"attempted, avoids a false positive)")
            result.recommendation = "NO_COVERAGE"
            _log.info(result.error)
            return result

        # A plot that PASSED the gate is a confirmed-corridor plot: the schedule
        # says the surveyor DID cross it, so identity is certain. It must never be
        # dropped as NO_COVERAGE -- if the auto-placement is weak it becomes
        # REVIEW (human confirms/seeds the position, per the spec workflow), never
        # silently discarded.
        confirmed_corridor = corridor_surveys is not None

        if m1.n_stones < 3 or m1.n_edges < 3:
            result.error = f"Too few stones ({m1.n_stones}) or edges ({m1.n_edges})"
            result.recommendation = "REVIEW" if confirmed_corridor else "NO_COVERAGE"
            _log.warning(result.error)
            return result

        # Step 2: Match against surveyor. Pass the plot's INDEPENDENT expected
        # position (cadastral label) so the seat-locality FP guard rejects a
        # chance-congruent subset far from the true seat (see match.geometric_match
        # / _GEOM_MAX_SEAT_DIST). No-op when no label is available for this plot.
        expected_xy = None
        if cadastral_source is not None:
            lp = getattr(cadastral_source, "label_point", lambda s: None)(
                m1.survey_number)
            if lp is not None:
                expected_xy = lp
        match = match_plot(m1, surveyor, expected_xy=expected_xy)
        result.fingerprint_score = match.fingerprint_score
        result.neighborhood_score = match.neighborhood_score
        result.match_method = match.match_method

        if not match.matched:
            # No congruent >=4-stone subset auto-found. If the schedule confirms
            # this plot IS on the corridor, it still needs placing -> REVIEW (the
            # corridor clips plots, so a heavily-clipped one may expose too few
            # corner stones to auto-match; a human seeds the 2-point fit). Without
            # a schedule, no congruent subset means OFF the corridor -> NO_COVERAGE.
            result.error = (f"No auto-match "
                            f"(best congruent subset < {4} stones; "
                            f"fp={match.fingerprint_score:.2f})")
            result.recommendation = "REVIEW" if confirmed_corridor else "NO_COVERAGE"
            _log.warning(result.error)
            return result

        matched_pairs = [(i, j) for i, j in enumerate(match.stone_map) if j >= 0]
        _log.info("Matched %d/%d stones", len(matched_pairs), m1.n_stones)

        # Geometric-congruence confidence = the false-positive gate. A genuine
        # corridor plot has many corners landing on surveyor stones at low
        # residual; a collision does not. Inlier count carries most of the
        # statistical weight (N congruent stones are exponentially unlikely by
        # chance), with fraction and residual refining it.
        result.n_corners = len(m1.outer_stone_indices)
        result.n_inliers = match.n_matched_stones
        frac = result.n_inliers / max(result.n_corners, 1)
        resid = match.fingerprint_score if match.fingerprint_score < 1e6 else 99.0
        inlier_term = min(1.0, max(0.0, (result.n_inliers - 3) / 5.0))
        frac_term = max(0.0, (frac - 0.5) / 0.5)
        resid_term = max(0.0, 1.0 - resid / 5.0)
        result.confidence = round(
            0.5 * inlier_term + 0.3 * frac_term + 0.2 * resid_term, 3)

        if len(matched_pairs) < 2:
            result.error = f"Too few matched stones ({len(matched_pairs)})"
            result.recommendation = "REVIEW" if confirmed_corridor else "NO_COVERAGE"
            _log.warning(result.error)
            return result

        # Step 3: Umeyama initial transform
        src_pts = m1.stone_positions()[np.array([p[0] for p in matched_pairs])]
        dst_pts = np.array([surveyor.stone_coords(p[1]) for p in matched_pairs])

        R, scale, translation, residuals = umeyama(src_pts, dst_pts)

        # Step 4: Least-squares cadastral adjustment
        edge_pairs = [(e.stone_a, e.stone_b, e.length_m) for e in m1.outer_edges]
        adjusted = cadastral_adjust(
            m1_positions=m1.stone_positions(),
            surveyor_positions=np.array([[s.x, s.y] for s in surveyor.stones]),
            matched_pairs=matched_pairs,
            edge_pairs=edge_pairs,
            field_weight=FIELD_WEIGHT,
            dist_weight=DIST_WEIGHT,
            umeyama_result=(R, scale, translation),
        )

        # Compute field residuals
        field_res = [np.linalg.norm(adjusted[m1_idx] -
                                     np.array(surveyor.stone_coords(surv_idx)))
                     for m1_idx, surv_idx in matched_pairs]
        result.field_residual_mean = float(np.mean(field_res))
        result.field_residual_max = float(np.max(field_res))

        # Step 5: Write georeferenced DXF
        stone_label_map = {}
        for stone in m1.stones:
            stone_label_map[stone.label] = stone.index

        output_path = output_dir / f"georef_{m1_dxf_path.stem}.dxf"
        write_georef_dxf(
            m1_dxf_path=m1_dxf_path,
            output_path=output_path,
            adjusted_stone_positions=adjusted,
            original_stone_positions=m1.stone_positions(),
            stone_label_to_index=stone_label_map,
            R=R, s=scale, t=translation,
            crs=crs,
            corner_ring=m1.outer_stone_indices,
        )

        result.matched = True
        result.output_file = str(output_path)

        # Chain coverage: fraction of the georeferenced boundary lying on the
        # surveyor's traced SITE DATA LINE. Built from the ADJUSTED corner ring
        # (what output_dxf writes as the BOUNDARY). This is the independent
        # ground-truth false-positive gate (see verify.chain_coverage).
        ring = m1.outer_stone_indices
        ring_pts = [(float(adjusted[i][0]), float(adjusted[i][1])) for i in ring]
        boundary_segs = [(ring_pts[k], ring_pts[(k + 1) % len(ring_pts)])
                         for k in range(len(ring_pts))] if len(ring_pts) >= 3 else []
        surv_polys = [pl.raw_points for pl in surveyor.polylines]
        result.chain_coverage = chain_coverage(boundary_segs, surv_polys)

        # Step 6: Verify output
        vr = verify_georef_dxf(
            output_path,
            m1_dxf_path=m1_dxf_path,
            match_result=match,
            m1_data=m1,
            field_residual_max=result.field_residual_max,
        )
        result.verify_result = vr
        print_verify_result(vr)

        # Final disposition. The DECISIVE false-positive gate is CHAIN COVERAGE:
        # in a dense stone cloud a similarity transform can drop a rigid plot
        # shape onto SOME congruent subset by chance (per-plot inlier count alone
        # over-matches -- 35/35 with 92 footprint overlaps), but only a TRULY
        # surveyed plot's boundary lies on the surveyor's traced lines. INGUR
        # separation is clean and wide (true 58-100%, coincidental 12-43%).
        # Inlier count + residual + the 7 verify gates remain as conjunctive
        # geometry-sanity requirements; a global footprint non-overlap pass
        # (georef_pipeline) is the final tiling safety net.
        sane = vr.all_passed if vr else False
        cov = result.chain_coverage
        if cov >= CHAIN_COVER_ACCEPT and result.n_inliers >= 6 and resid < 3.0 and sane:
            result.recommendation = "ACCEPT"
        elif cov >= CHAIN_COVER_REVIEW and result.n_inliers >= 4:
            result.recommendation = "REVIEW"
        elif confirmed_corridor:
            # Schedule confirms this plot IS on the corridor, so it is NOT a false
            # positive -- the weak coverage is corridor-clipping (only part of the
            # plot was field-traced). Surface it for human confirmation/seeding
            # rather than discarding a real plot.
            result.recommendation = "REVIEW"
        else:
            # No schedule + boundary NOT on traced lines -> coincidental match on a
            # plot the surveyor never crossed.
            result.recommendation = "NO_COVERAGE"
        _log.info("Disposition: %s (cov=%.0f%%, inliers=%d/%d, resid=%.2fm)",
                  result.recommendation, 100 * cov,
                  result.n_inliers, result.n_corners, resid)

        elapsed = time.time() - t0
        _log.info("SUCCESS: %s -> %s (%.1fs) field_res mean=%.4fm max=%.4fm",
                  m1_dxf_path.name, output_path.name, elapsed,
                  result.field_residual_mean, result.field_residual_max)

    except Exception as e:
        result.error = str(e)
        _log.exception("Failed to georeference %s", m1_dxf_path)

    return result


def seed_place(
    m1_dxf_path: str | Path,
    surveyor: SurveyorData | None,
    corner_a: str,
    corner_b: str,
    utm_a: tuple[float, float],
    utm_b: tuple[float, float],
    output_dir: str | Path,
    crs: str = "EPSG:32643",
    snap_tol: float = 3.0,
) -> GeorefResult:
    """Place ONE plot from TWO human-given corner->UTM correspondences (Stage-3).

    The automated matcher cannot safely place heavily corridor-clipped plots
    (too few stones at their true seat). This is the documented manual fallback,
    automated: the operator names two M1 corner stones and their two surveyed UTM
    coordinates; two correspondences fully determine the 2D similarity (rotation,
    uniform scale, translation), and the FMB polygon's full shape carries the
    rest. There is NO geometric guessing -- the operator supplies the identity --
    so this cannot produce a false positive; the 7 verify gates still run on the
    output. After the rigid 2-point fit, the placed corners are snapped to nearby
    surveyor stones (tight ``snap_tol``) and a cadastral least-squares adjustment
    refines the fit against those real stones, removing the ~0.2 m FMB-vs-field
    error. Disposition is ACCEPT_SEEDED (operator-confirmed) when verify passes.
    """
    m1_dxf_path = Path(m1_dxf_path)
    output_dir = Path(output_dir)
    result = GeorefResult(m1_file=str(m1_dxf_path), survey_number="", matched=False)

    try:
        m1 = extract_m1_dxf(m1_dxf_path)
        result.survey_number = m1.survey_number

        label_idx: dict[str, int] = {}
        for st in m1.stones:
            label_idx.setdefault(st.label, st.index)
        if corner_a not in label_idx or corner_b not in label_idx:
            result.error = (f"seed corner label(s) not on STONES layer: "
                            f"{corner_a!r}/{corner_b!r} "
                            f"(available e.g. {list(label_idx)[:6]})")
            result.recommendation = "REVIEW"
            return result

        ia, ib = label_idx[corner_a], label_idx[corner_b]
        src = np.array([[m1.stones[ia].x, m1.stones[ia].y],
                        [m1.stones[ib].x, m1.stones[ib].y]])
        dst = np.array([list(utm_a), list(utm_b)], dtype=float)
        R, scale, translation = umeyama(src, dst)[:3]

        # Seed quality: a 2-corner seed is exactly determined (no averaging), so a
        # SHORT baseline amplifies the operator's point error across a large plot.
        # Reject below 2 m worst-case far-corner induced error (== the field-residual
        # reject bound). Tighten-only: a poor seed is forced to REVIEW, never ACCEPT.
        sq = seed_quality(src, dst, template_points=m1.stone_positions(),
                          max_induced_error_m=2.0, min_baseline_m=5.0)
        if not sq.ok:
            _log.warning("Seed quality weak for %s: %s", m1.survey_number, sq.reason)

        placed = scale * (m1.stone_positions() @ R.T) + translation

        # Snap placed corners to real surveyor stones for cadastral refinement.
        # PURE 2-corner seed (no corridor reference): surveyor is None -> rigid placement
        # straight from the two operator correspondences (still 0-FP: human-supplied identity).
        matched_pairs: list[tuple[int, int]] = []
        if surveyor is not None and surveyor.stones:
            if surveyor._stone_tree is None:
                surveyor.build_index()
            for i, p in enumerate(placed):
                d, j = surveyor._stone_tree.query(p)
                if d <= snap_tol:
                    matched_pairs.append((i, int(j)))
            surv_pos = np.array([[s.x, s.y] for s in surveyor.stones])
        else:
            surv_pos = np.zeros((0, 2))

        if len(matched_pairs) >= 2:
            edge_pairs = [(e.stone_a, e.stone_b, e.length_m) for e in m1.outer_edges]
            adjusted = cadastral_adjust(
                m1_positions=m1.stone_positions(), surveyor_positions=surv_pos,
                matched_pairs=matched_pairs, edge_pairs=edge_pairs,
                field_weight=FIELD_WEIGHT, dist_weight=DIST_WEIGHT,
                umeyama_result=(R, scale, translation))
        else:
            adjusted = placed  # pure rigid 2-point placement

        output_path = output_dir / f"georef_{m1_dxf_path.stem}.dxf"
        write_georef_dxf(
            m1_dxf_path=m1_dxf_path, output_path=output_path,
            adjusted_stone_positions=adjusted,
            original_stone_positions=m1.stone_positions(),
            stone_label_to_index={s.label: s.index for s in m1.stones},
            R=R, s=scale, t=translation, crs=crs,
            corner_ring=m1.outer_stone_indices)

        from .match import MatchResult
        smap = [-1] * m1.n_stones
        for mi, sj in matched_pairs:
            smap[mi] = sj
        seed_match = MatchResult(
            stone_map=smap, fingerprint_score=0.0, neighborhood_score=0.0,
            combined_score=0.0, matched=True, match_method="seed_2point",
            n_matched_stones=len(matched_pairs))

        field_res = [float(np.linalg.norm(adjusted[mi] - surv_pos[sj]))
                     for mi, sj in matched_pairs]
        result.field_residual_max = float(max(field_res)) if field_res else 0.0
        vr = verify_georef_dxf(output_path, m1_dxf_path=m1_dxf_path,
                               match_result=seed_match, m1_data=m1,
                               field_residual_max=result.field_residual_max)
        result.verify_result = vr
        result.matched = True
        result.output_file = str(output_path)
        result.n_inliers = len(matched_pairs)
        result.n_corners = len(m1.outer_stone_indices)
        result.match_method = "seed_2point"

        ring = m1.outer_stone_indices
        ring_pts = [(float(adjusted[i][0]), float(adjusted[i][1])) for i in ring]
        segs = [(ring_pts[k], ring_pts[(k + 1) % len(ring_pts)])
                for k in range(len(ring_pts))] if len(ring_pts) >= 3 else []
        result.chain_coverage = (chain_coverage(
            segs, [pl.raw_points for pl in surveyor.polylines])
            if surveyor is not None else 0.0)

        # Operator supplied identity -> ACCEPT_SEEDED when the output verifies AND the
        # seed baseline is geometrically adequate (sq.ok). A weak/short baseline is
        # surfaced for human confirmation (REVIEW), never auto-accepted.
        if not sq.ok:
            result.error = (result.error + "; " if result.error else "") + sq.reason
        result.recommendation = (
            "ACCEPT_SEEDED" if (vr and vr.all_passed and sq.ok) else "REVIEW")
        _log.info("Seed-placed %s: %s (snapped %d stones, field_res<=%.3fm, "
                  "baseline=%.1fm induced=%.2fm)",
                  m1.survey_number, result.recommendation,
                  len(matched_pairs), result.field_residual_max,
                  sq.baseline_m, sq.max_induced_error_m)
    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)
        result.recommendation = "REVIEW"
        _log.exception("seed_place failed for %s", m1_dxf_path)

    return result


def _footprint_polygon(georef_dxf_path: str):
    """Largest enclosed BOUNDARY polygon of a georef DXF (the plot footprint)."""
    try:
        import ezdxf as _ezdxf
        from shapely.geometry import MultiLineString
        from shapely.ops import polygonize, unary_union
    except ImportError:
        return None
    try:
        msp = _ezdxf.readfile(georef_dxf_path).modelspace()
    except Exception:
        return None
    from ...core.enums import LayerType
    segs = []
    for e in msp.query("LWPOLYLINE"):
        if e.dxf.layer != LayerType.BOUNDARY.value:
            continue
        pts = [(p[0], p[1]) for p in e.get_points()]
        for i in range(len(pts) - 1):
            if pts[i] != pts[i + 1]:
                segs.append((pts[i], pts[i + 1]))
    if len(segs) < 3:
        return None
    faces = list(polygonize(unary_union(MultiLineString(segs))))
    return max(faces, key=lambda f: f.area) if faces else None


def _placement_confidence(r: GeorefResult) -> float:
    """Trust score for a confident placement, used to resolve footprint conflicts.

    Real parcels tile, so two placements that overlap are mutually exclusive and
    the lower-trust one is wrong. Ranking:
      * operator 2-point seed                       -> 0.99 (human identity)
      * strong corridor match (geometric + cov>=.5) -> 0.95 (real field stones)
      * cadastral polygon fit                       -> IoU (0.5-0.99; shape confirms id)
      * weak corridor (propagated / edge-registered)-> 0.40 (heuristic guess)
    So a high-IoU cadastral placement correctly OVERRIDES a propagated/edge-
    registered corridor guess that landed on the wrong parcel (measured on INGUR:
    721/722 were sitting on 667/669's parcels).
    """
    m = r.match_method or ""
    if r.recommendation == "ACCEPT_SEEDED" or m == "seed_2point":
        return 0.99
    if r.recommendation == "ACCEPT_CADASTRAL":
        return float(r.chain_coverage)            # IoU stored here
    if r.recommendation == "ACCEPT":
        if m.startswith("geometric") and r.chain_coverage >= CHAIN_COVER_ACCEPT:
            return 0.95
        if m == "geometric_corroborated":
            return 0.93        # real field stones + independent S3 label agreement
        return 0.40                                # propagated / edge_registered
    return 0.0


def _resolve_footprint_conflicts(results: list[GeorefResult]) -> None:
    """Demote overlapping confident placements so the placed set is a NON-OVERLAPPING
    tiling. Real parcels tile (shared edges, never interiors), so two confident
    placements whose footprints overlap are mutually exclusive: keep the higher-
    trust one (see _placement_confidence), demote the other to REVIEW. Considers
    ACCEPT, ACCEPT_SEEDED and ACCEPT_CADASTRAL together, so the cadastre can correct
    a wrong corridor guess and vice versa. Mutates ``results`` in place.
    """
    conf_states = ("ACCEPT", "ACCEPT_SEEDED", "ACCEPT_CADASTRAL")
    placed = [r for r in results if r.recommendation in conf_states and r.output_file]
    polys = {id(r): _footprint_polygon(r.output_file) for r in placed}
    kept: list[GeorefResult] = []
    for r in sorted(placed, key=_placement_confidence, reverse=True):
        pr = polys[id(r)]
        if pr is None:
            continue
        conflict = None
        for k in kept:
            pk = polys[id(k)]
            if pk is None or not pr.intersects(pk):
                continue
            ov = pr.intersection(pk).area / max(min(pr.area, pk.area), 1e-9)
            if ov > FOOTPRINT_CONFLICT:
                conflict = k
                break
        if conflict is not None:
            r.recommendation = "REVIEW"
            r.error = (r.error + "; " if r.error else "") + (
                f"footprint overlaps higher-trust {conflict.survey_number} "
                f"({conflict.match_method}); demoted")
        else:
            kept.append(r)


# Edge-registration pass tuning (corridor-window placement onto traced lines).
_ER_WINDOW = 450.0     # stone-window half-width (m) along the corridor axis
_ER_POS_TOL = 450.0    # placed centroid must land within this of its predicted pos
_ER_OVERLAP = 0.20     # interior overlap (of smaller) above this = collision


def _edge_register_reviews(
    surveyor: SurveyorData,
    results: list[GeorefResult],
    m1_data_map: dict,
    corridor_order: list[str],
    output_dir: Path,
    crs: str,
) -> list[str]:
    """Upgrade clipped REVIEW corridor plots by registering their boundary onto
    the surveyor's traced SITE DATA LINE (see edge_register.register_plot_on_traced).

    Runs AFTER corner-stone propagation, on whatever stays REVIEW. A plot is
    upgraded to ACCEPT only if its boundary, fit onto the traced lines within its
    schedule window, reaches the ACCEPT chain-coverage bar AND lands at its
    predicted corridor position AND does not overlap a placed plot -- so it cannot
    introduce a false positive (it is identity-confirmed by the schedule and its
    boundary measurably lies on the traced cadastral line). Mutates results;
    returns the upgraded survey numbers.
    """
    from .edge_register import register_plot_on_traced, traced_segments
    from .propagate import _axis, _corner_polygon

    rank = {sn: i for i, sn in enumerate(corridor_order)}
    placed_pos: dict[str, float] = {}
    placed_poly: dict[str, object] = {}
    c, axis = _axis(surveyor)
    stone_proj = (surveyor.stone_positions - c) @ axis

    def proj_xy(x, y):
        return float((np.array([x, y]) - c) @ axis)

    # Anchor positions from the already-placed plots (ACCEPT/ACCEPT_SEEDED).
    for r in results:
        if r.recommendation in ("ACCEPT", "ACCEPT_SEEDED") and r.output_file \
                and r.survey_number in rank:
            poly = _footprint_polygon(r.output_file)
            if poly is not None and poly.is_valid and poly.area > 0:
                placed_pos[r.survey_number] = proj_xy(poly.centroid.x, poly.centroid.y)
                placed_poly[r.survey_number] = poly

    reviews = [r for r in results
               if r.recommendation == "REVIEW" and r.survey_number in rank
               and r.survey_number in m1_data_map]
    if len(placed_pos) < 2 or not reviews:
        return []

    ranks = np.array([rank[sn] for sn in placed_pos])
    poss = np.array([placed_pos[sn] for sn in placed_pos])
    slope, intercept = np.polyfit(ranks, poss, 1)

    upgraded: list[str] = []
    for r in reviews:
        sn = r.survey_number
        m1 = m1_data_map[sn]
        pred = slope * rank[sn] + intercept
        mask = np.abs(stone_proj - pred) <= _ER_WINDOW
        if int(mask.sum()) < 2:
            continue
        segs = traced_segments(surveyor, keep=mask)
        if len(segs) < 1:
            continue
        reg = register_plot_on_traced(m1, surveyor, segments=segs)
        if reg is None or reg.coverage < CHAIN_COVER_ACCEPT:
            continue
        poly = _corner_polygon(m1, reg.adjusted)
        if poly is None or not poly.is_valid or poly.area <= 0:
            continue
        if abs(proj_xy(poly.centroid.x, poly.centroid.y) - pred) > _ER_POS_TOL:
            continue
        collide = False
        for ppoly in placed_poly.values():
            if poly.intersects(ppoly):
                ov = poly.intersection(ppoly).area / max(min(poly.area, ppoly.area), 1e-9)
                if ov > _ER_OVERLAP:
                    collide = True
                    break
        if collide:
            continue

        out = output_dir / f"georef_{Path(r.m1_file).stem}.dxf"
        write_georef_dxf(
            m1_dxf_path=r.m1_file, output_path=out,
            adjusted_stone_positions=reg.adjusted,
            original_stone_positions=m1.stone_positions(),
            stone_label_to_index={st.label: st.index for st in m1.stones},
            R=reg.R, s=reg.s, t=reg.t, crs=crs,
            corner_ring=m1.outer_stone_indices,
        )
        r.matched = True
        r.output_file = str(out)
        r.chain_coverage = reg.coverage
        r.n_inliers = len(reg.matched_pairs)
        r.n_corners = len(m1.outer_stone_indices)
        r.recommendation = "ACCEPT"
        r.match_method = "edge_registered"
        placed_pos[sn] = proj_xy(poly.centroid.x, poly.centroid.y)
        placed_poly[sn] = poly
        upgraded.append(sn)
        _log.info("Edge-registered %s -> ACCEPT (coverage=%.0f%%, %d anchor stones)",
                  sn, 100 * reg.coverage, len(reg.matched_pairs))

    return upgraded


def _is_cross_village(m1_file: str, parcel_village: str | None) -> bool:
    """True if the FMB's own village (from its filename) differs from the cadastral
    parcel's village. FMB filenames are FMB_<DISTRICT>_<TALUK>_<VILLAGE>_<survey>.dxf
    (e.g. ..._INGUR_763, ..._KANDAMPALAYAM _9). Conservative: returns False when the
    village can't be parsed (no spurious rejections), True only on a clear mismatch.
    """
    if not parcel_village:
        return False
    stem = Path(m1_file).stem.upper().replace(" ", "")
    pv = str(parcel_village).upper().replace(" ", "")
    if pv in stem:
        return False
    # Only trust the village token when the name follows the government FMB pattern
    # FMB_<DISTRICT>_<TALUK>_<VILLAGE>_<survey>. Other names (e.g. synthetic test
    # fixtures "m1_synth_784") carry no village -> stay conservative, don't reject.
    m = re.match(r"^FMB_[A-Z.]+_[A-Z.]+_([A-Z.]+)_?\d+$", stem)
    if not m:
        return False
    return m.group(1) != pv


def _build_village_fence(surveyor: SurveyorData, buffer_m: float = 300.0):
    """CONCAVE (alpha-shape) hull of all surveyor stones, buffered -- the target village extent.

    Used to reject OCR labels (and placements) from ADJACENT villages that share a
    survey number (e.g. KANDAMPALAYAM survey 9 vs INGUR). A concave hull hugs a band-shaped
    village far more snugly than a convex hull (which is a big empty triangle), so it rejects
    more far cross-village labels at no recall cost -- every stone stays inside. Convex fallback
    for < 4 stones. Always available -- it is the true field extent of this job.
    """
    from ...core.geo import village_fence
    pts = [(s.x, s.y) for s in surveyor.stones]
    if len(pts) < 3:
        return None
    return village_fence(pts, buffer=buffer_m, concave=True)


def _place_by_cadastral(results, cadastral_source, surveyor: SurveyorData,
                        output_dir: Path, crs: str) -> list[str]:
    """Place plots the corridor could NOT confidently place onto their authoritative
    cadastral parcel (from a client vector file or the S3 cadastral tiles).

    Identity is keyed by survey number (no shape-search). FP gates: (1) the placed
    centroid must be within CAD_CORRIDOR_MAX of a surveyor stone (rejects cross-
    village mislocations like survey 9); (2) it must not overlap an already-placed
    plot (real parcels tile); (3) ACCEPT_CADASTRAL requires shape agreement (IoU)
    AND centroid proximity -- otherwise REVIEW (located, human confirms).
    """
    from ..m5_cadastral.fit import fit_plot_to_parcel
    from .extract_m1 import extract_m1_dxf
    if surveyor._stone_tree is None:
        surveyor.build_index()
    label_point = getattr(cadastral_source, "label_point", lambda s: None)

    # Footprints already placed with confidence (for the overlap gate, Bug #2).
    placed_polys: list[tuple[str, object]] = []
    for r in results:
        if r.recommendation in ("ACCEPT", "ACCEPT_SEEDED", "ACCEPT_CADASTRAL") \
                and r.output_file:
            pp = _footprint_polygon(r.output_file)
            if pp is not None and pp.is_valid and pp.area > 0:
                placed_polys.append((r.survey_number, pp))

    placed: list[str] = []
    for r in results:
        strong = r.recommendation == "ACCEPT_SEEDED" or (
            r.recommendation == "ACCEPT"
            and (r.match_method or "").startswith("geometric")
            and r.chain_coverage >= CHAIN_COVER_ACCEPT)
        if strong:
            continue

        # --- Cross-corroboration (FP-safe recall, runs BEFORE any cadastral re-fit):
        # a REVIEW plot that matched REAL surveyor stones (authoritative geometry, low
        # field residual, enough inliers) and whose corridor placement lands near its
        # INDEPENDENT S3 label point is confirmed by two unrelated sources -> ACCEPT.
        # Keep the corridor placement (do NOT overwrite it with a cadastral parcel
        # fit, which for these corridor-end plots is the weaker signal).
        if (r.recommendation == "REVIEW"
                and (r.match_method or "").startswith("geometric")
                and r.output_file
                and r.n_inliers >= CORROB_MIN_INLIERS
                and r.field_residual_max <= CORROB_FIELD_RESID):
            lp = label_point(r.survey_number)
            fp = _footprint_polygon(r.output_file) if lp is not None else None
            if lp is not None and fp is not None and fp.is_valid and fp.area > 0:
                c = fp.centroid
                d = float(np.hypot(c.x - lp[0], c.y - lp[1]))
                conflict = any(
                    fp.intersects(pp) and fp.intersection(pp).area
                    / max(min(fp.area, pp.area), 1e-9) > FOOTPRINT_CONFLICT
                    for _, pp in placed_polys)
                if d <= CORROB_TOL and not conflict:
                    r.recommendation = "ACCEPT"
                    r.match_method = "geometric_corroborated"
                    placed.append(r.survey_number)
                    placed_polys.append((r.survey_number, fp))
                    _log.info("Corroborated %s -> ACCEPT (real stones %d inliers, "
                              "field_resid=%.2fm, S3 label %.0fm away)",
                              r.survey_number, r.n_inliers, r.field_residual_max, d)
                    continue

        parcel = cadastral_source.get(r.survey_number)
        if parcel is None:
            continue
        # Cross-village guard: a plot from a DIFFERENT village (e.g. KANDAMPALAYAM
        # survey 9) must never be placed onto this village's cadastre, even if a
        # same-numbered parcel here happens to fit. The corridor schedule cannot tell
        # it apart (legitimate off-corridor INGUR plots are off-schedule too), so use
        # the FMB's own village identity from its filename. Skip entirely (it stays
        # NO_COVERAGE -> staged to-scale in the combined file, not georeferenced here).
        if _is_cross_village(r.m1_file, getattr(parcel, "village", None)):
            _log.warning("Cadastral: %s is from a different village than the cadastre "
                         "-> skipped (not placed on this village's map)", r.survey_number)
            continue
        try:
            m1 = extract_m1_dxf(r.m1_file)
        except Exception:  # noqa: BLE001
            continue
        fit = fit_plot_to_parcel(m1, parcel, anchor=label_point(r.survey_number))
        if fit is None:
            continue
        # OPEN-PARCEL RECOVERY: for a label that had no clean cadastral face, the source
        # may offer several locally-reconstructed candidate rings (yellow-gap bridge).
        # If the primary fit does not clear the rigid SHAPE gate, try each candidate and
        # adopt the first whose fit DOES -- the gate (run here, unchanged) is the sole
        # arbiter, so a recovered ring can never become a false ACCEPT (verified: INGUR
        # cross-village survey 9 yields no gate-passing candidate). Picking the
        # gate-passing candidate rather than a fixed geometric heuristic is what lets the
        # right closure (e.g. 768) win over an over-bridged one.
        def _passes_shape_gate(f):
            return (f is not None and f.method == "rigid"
                    and CAD_AREA_LO <= f.area_ratio <= CAD_AREA_HI
                    and CAD_SCALE_LO <= f.s <= CAD_SCALE_HI
                    and f.orientation_ok
                    and f.rot_residual <= CAD_ROT_RESID_MAX)
        if not _passes_shape_gate(fit):
            get_cands = getattr(cadastral_source, "recovered_candidates",
                                lambda s: [])
            best_cand = None
            for cand in get_cands(r.survey_number):
                cf = fit_plot_to_parcel(m1, cand, anchor=label_point(r.survey_number))
                if _passes_shape_gate(cf):
                    if (best_cand is None
                            or abs(cf.area_ratio - 1.0) < abs(best_cand.area_ratio - 1.0)):
                        best_cand = cf
            if best_cand is not None:
                fit = best_cand
        # BUG #5 (corrected): per-plot local refinement against real field stones.
        fit = _refine_against_stones(fit, m1, surveyor, parcel)
        ring = fit.adjusted[np.array(m1.outer_stone_indices)]
        centroid = ring.mean(axis=0)

        # BUG #1 gate: a placement far from every corridor stone is a cross-village
        # mislocation (a duplicate survey number in an adjacent village). HARD reject.
        dist_corridor = float(surveyor._stone_tree.query(centroid)[0])
        if dist_corridor > CAD_CORRIDOR_MAX:
            _log.warning("Cadastral %s is %.0fm from corridor -> cross-village "
                         "mislocation, skipped", r.survey_number, dist_corridor)
            continue

        # BUG #2: an overlap with an already-placed plot does NOT delete the plot --
        # it just blocks ACCEPT (real parcels tile, so an overlap means this fit is
        # off). It stays REVIEW (visible on its own layer for human seeding).
        # REFINED: the veto only counts overlap with a CONFIDENT (ACCEPT) footprint.
        # A REVIEW plot's fit is unconfirmed -- a mis-placed REVIEW that sprawls over
        # a neighbour must NOT veto that neighbour's otherwise-clean ACCEPT (that was
        # demoting 721 behind 698 and 1025 behind 1023, both bad REVIEW sprawls).
        # `placed_polys` holds only confident footprints; the global
        # _resolve_footprint_conflicts pass is the safety net if two ACCEPTs collide.
        from shapely.geometry import Polygon as _Poly
        new_poly = _Poly([(p[0], p[1]) for p in ring])
        if not new_poly.is_valid:
            new_poly = new_poly.buffer(0)
        overlaps = None
        for psn, ppoly in placed_polys:
            if new_poly.intersects(ppoly):
                ov = new_poly.intersection(ppoly).area / max(min(new_poly.area, ppoly.area), 1e-9)
                if ov > FOOTPRINT_CONFLICT:
                    overlaps = psn
                    break

        out = output_dir / f"georef_{Path(r.m1_file).stem}.dxf"
        try:
            write_georef_dxf(
                m1_dxf_path=r.m1_file, output_path=out,
                adjusted_stone_positions=fit.adjusted,
                original_stone_positions=m1.stone_positions(),
                stone_label_to_index={s.label: s.index for s in m1.stones},
                R=fit.R, s=fit.s, t=fit.t, crs=crs,
                corner_ring=m1.outer_stone_indices)
        except Exception as exc:  # noqa: BLE001
            _log.warning("cadastral write failed for %s: %s", r.survey_number, exc)
            continue
        r.matched = True
        r.output_file = str(out)
        r.n_corners = len(m1.outer_stone_indices)
        r.n_inliers = fit.n_inliers
        r.chain_coverage = fit.area_ratio   # store the area-ratio quality here
        r.cad_residual = fit.rot_residual   # rigid-fit corner residual (m)

        # FP gate (NO IoU-against-pixel-boundary): ACCEPT_CADASTRAL requires the M1
        # plot to be the right SIZE for the parcel (area ratio in band), the rigid
        # corner alignment to be sane (residual small => correct rotation/parcel),
        # and no overlap with a placed plot. The corridor-distance gate above already
        # rejected cross-village mislocations. Otherwise REVIEW.
        accept = (fit.method == "rigid"
                  and CAD_AREA_LO <= fit.area_ratio <= CAD_AREA_HI
                  and CAD_SCALE_LO <= fit.s <= CAD_SCALE_HI      # A1 scale gate
                  and fit.orientation_ok                          # A2 flip gate
                  and fit.rot_residual <= CAD_ROT_RESID_MAX
                  and overlaps is None)
        if accept:
            r.recommendation = "ACCEPT_CADASTRAL"
            r.match_method = "cadastral_rigid"
            placed.append(r.survey_number)
            placed_polys.append((r.survey_number, new_poly))
            _log.info("Cadastral-placed %s -> ACCEPT_CADASTRAL (area_ratio=%.2f, "
                      "scale=%.3f, rot_resid=%.1fm)", r.survey_number,
                      fit.area_ratio, fit.s, fit.rot_residual)
        else:
            r.recommendation = "REVIEW"
            r.match_method = f"cadastral_{fit.method}_located"
            # NOTE: a REVIEW footprint is NOT added to placed_polys -- an unconfirmed
            # fit must not veto a later plot's ACCEPT (see overlap comment above).
            why = []
            if fit.method != "rigid":
                why.append(f"method={fit.method}")
            if not (CAD_AREA_LO <= fit.area_ratio <= CAD_AREA_HI):
                why.append(f"area_ratio={fit.area_ratio:.2f}")
            if not (CAD_SCALE_LO <= fit.s <= CAD_SCALE_HI):
                why.append(f"scale={fit.s:.3f}")
            if not fit.orientation_ok:
                why.append("flipped-orientation")
            if fit.rot_residual > CAD_ROT_RESID_MAX:
                why.append(f"rot_resid={fit.rot_residual:.1f}m")
            if overlaps is not None:
                why.append(f"overlaps={overlaps}")
            _log.info("Cadastral-located %s -> REVIEW (%s)", r.survey_number,
                      ", ".join(why) or "below gate")
    return placed


def _refine_against_stones(fit, m1, surveyor, parcel):
    """Per-plot LOCAL refinement: snap placed corners to nearby surveyor stones
    (<=5 m) and re-solve. Replaces a global datum shift (which masks FPs + breaks
    per-plot IoU). Off-corridor plots have no nearby stones -> no-op; corridor-
    adjacent cadastral plots get pinned to real field control. Returns a (possibly
    refined) CadastralFit.
    """
    from ..m5_cadastral.fit import CadastralFit, _poly_area
    corners = np.array(m1.outer_stone_indices)
    m1_pos = m1.stone_positions()
    placed = fit.adjusted
    pairs = []
    for i in range(len(placed)):
        d, j = surveyor._stone_tree.query(placed[i])
        if d <= 5.0:
            pairs.append((i, int(j)))
    if len(pairs) < 2:
        return fit
    # RIGID re-fit to the real field stones (rotation+scale+translation only) so the
    # M1 FMB geometry is preserved -- we only nudge its position/orientation to the
    # authoritative field control, never deform its shape.
    src = m1_pos[np.array([i for i, _ in pairs])]
    dst = np.array([surveyor.stone_coords(j) for _, j in pairs])
    R2, s2, t2 = umeyama(src, dst)[:3]
    if not (0.5 < s2 < 2.0):
        return fit
    adjusted = s2 * (m1_pos @ R2.T) + t2
    ring = adjusted[corners]
    ar = _poly_area(ring) / max(parcel.polygon.area, 1e-9) if parcel.polygon else fit.area_ratio
    return CadastralFit(adjusted=adjusted, R=R2, s=s2, t=t2,
                        method=fit.method, n_inliers=len(pairs),
                        area_ratio=ar, rot_residual=fit.rot_residual,
                        orientation_ok=fit.orientation_ok)


def georef_pipeline(
    surveyor_dxf: str | Path,
    m1_dxf_paths: list[str | Path],
    output_dir: str | Path,
    crs: str = "EPSG:32643",
    corridor_surveys: set[str] | None = None,
    cadastral_source: object | None = None,
    village: str = "INGUR",
) -> list[GeorefResult]:
    """Run the full M2 georeferencing pipeline on multiple M1 DXFs.

    Parameters
    ----------
    surveyor_dxf : path to the surveyor's field-surveyed DXF
    m1_dxf_paths : list of M1-produced DXF paths
    output_dir : directory for georeferenced output DXFs
    crs : coordinate reference system
    corridor_surveys : optional identity gate (survey numbers the corridor
        crosses, from the land schedule). Plots outside it are NO_COVERAGE without
        matching -- removes geometric false positives. See ``georef_single``.

    Returns
    -------
    List of GeorefResult, one per M1 DXF.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log.info("Extracting surveyor data from %s", surveyor_dxf)
    surveyor = extract_surveyor(surveyor_dxf)
    surveyor.build_index()

    if not surveyor.stones:
        _log.error("No boundary stones found in surveyor DXF")
        return []

    _log.info("Surveyor: %d boundary stones, %d chains, extent=%s",
              len(surveyor.stones), len(surveyor.chains), surveyor.extent)
    if corridor_surveys:
        _log.info("Corridor identity gate active: %d survey numbers",
                  len(corridor_surveys))

    results = []
    for m1_path in m1_dxf_paths:
        result = georef_single(m1_path, surveyor, output_dir, crs, corridor_surveys,
                               cadastral_source=cadastral_source)
        results.append(result)

    # Span self-calibration (tighten-only): if THIS span's coverage distribution
    # shows the coincidental cluster creeping above the static floor, raise the
    # ACCEPT bar to the clean gap and demote ACCEPTs below it to REVIEW. Never
    # loosens (0-FP); a no-op on INGUR and any span without a stronger-than-floor
    # coincidental cluster. See self_calibrate.apply_calibrated_gate.
    def _demote_review(r, bar: float) -> None:
        r.recommendation = "REVIEW"
        r.error = (r.error + "; " if r.error else "") + (
            f"below span-calibrated coverage bar {bar:.2f}; demoted")
    apply_calibrated_gate(
        results,
        get_recommendation=lambda r: r.recommendation,
        set_review=_demote_review,
        get_coverage=lambda r: float(r.chain_coverage),
        floor=CHAIN_COVER_ACCEPT,
    )

    # Cadastral cross-validation (INDEPENDENT accuracy/FP check). When an authoritative
    # cadastral source is provided, score every ACCEPT plot against its parcel: CONFIRM
    # good placements (trust signal) and DEMOTE gross misses the geometry gates passed.
    # Tighten-only (ACCEPT->REVIEW), never promotes -> can only improve accuracy. No-op
    # when no cadastral source. See cadastral_check.apply_cadastral_gate.
    if cadastral_source is not None:
        from .cadastral_check import apply_cadastral_gate

        def _cad_demote(r, ag) -> None:
            r.recommendation = "REVIEW"
            r.error = (r.error + "; " if r.error else "") + ag.note() + " -> demoted"

        def _cad_confirm(r, ag) -> None:
            r.error = (r.error + "; " if r.error else "") + ag.note()

        try:
            cad_stats = apply_cadastral_gate(
                results, cadastral_source,
                footprint_fn=lambda r: (_footprint_polygon(r.output_file)
                                        if getattr(r, "output_file", "") else None),
                on_demote=_cad_demote, on_confirm=_cad_confirm)
            if cad_stats.get("checked"):
                _log.info("Cadastral accuracy: %d/%d ACCEPT confirmed on parcel "
                          "(mean offset %sm), %d demoted as gross-miss",
                          cad_stats["confirmed"], cad_stats["checked"],
                          cad_stats["mean_confirmed_offset_m"],
                          cad_stats["gross_miss_demoted"])
        except Exception as exc:  # noqa: BLE001 - cross-check must never break a job
            _log.error("Cadastral cross-check failed: %s", exc)

    # Global safety net: the ACCEPT set must be a non-overlapping tiling.
    _resolve_footprint_conflicts(results)

    # Second pass: when the corridor schedule is known, place REVIEW plots in
    # their schedule-predicted corridor windows (propagation from the placed
    # anchors). Strictly gated (window match + predicted position + non-overlap),
    # so it upgrades only correctly-seated plots and never force-places a
    # heavily-clipped one. See propagate.py.
    if corridor_surveys:
        from .extract_m1 import extract_m1_dxf
        from .propagate import propagate_review_plots
        m1_data_map = {}
        for r in results:
            if r.survey_number and r.recommendation in ("ACCEPT", "REVIEW"):
                try:
                    m1_data_map[r.survey_number] = extract_m1_dxf(r.m1_file)
                except Exception:
                    pass
        corridor_order = sorted(
            (s for s in corridor_surveys if s.isdigit()), key=int)
        try:
            upgraded = propagate_review_plots(
                surveyor, results, m1_data_map, corridor_order, output_dir, crs)
            if upgraded:
                _resolve_footprint_conflicts(results)
                _log.info("Propagation upgraded %d REVIEW->ACCEPT: %s",
                          len(upgraded), ", ".join(sorted(upgraded, key=int)))
        except Exception as exc:  # noqa: BLE001
            _log.error("Propagation pass failed: %s", exc)

        # Third pass: register remaining REVIEW plots' boundaries onto the traced
        # SITE DATA LINEs (clipped plots that expose a traceable EDGE but too few
        # corner stones). Gated by ACCEPT chain coverage + window + non-overlap.
        try:
            er = _edge_register_reviews(
                surveyor, results, m1_data_map, corridor_order, output_dir, crs)
            if er:
                _resolve_footprint_conflicts(results)
                _log.info("Edge-registration upgraded %d REVIEW->ACCEPT: %s",
                          len(er), ", ".join(sorted(er, key=int)))
        except Exception as exc:  # noqa: BLE001
            _log.error("Edge-registration pass failed: %s", exc)

    # Cadastral pass: place whatever the corridor could not, onto the authoritative
    # cadastral parcel by survey number (client vector file or S3 tiles). This is
    # what georeferences the OFF-CORRIDOR plots -- the surveyor never traced them,
    # but the cadastre covers the whole village. Keyed by identity, so no FP.
    if cadastral_source is not None:
        try:
            cad = _place_by_cadastral(results, cadastral_source, surveyor, output_dir, crs)
            if cad:
                _resolve_footprint_conflicts(results)
                _log.info("Cadastral pass placed %d plots -> ACCEPT_CADASTRAL: %s",
                          len(cad), ", ".join(sorted(cad, key=lambda s: (len(s), s))))
        except Exception as exc:  # noqa: BLE001
            _log.error("Cadastral pass failed: %s", exc)

    # Topology CORROBORATION (a CHECK, never a placement source): a located REVIEW
    # plot whose footprint correctly ABUTS a confident neighbour (shared edge, no
    # overlap) is confirmed by an independent signal -> ACCEPT. This is the FP-safe
    # half of "topology-constrained placement": it never propagates a position
    # (which would inherit a wrong-seat error undetectably), it only upgrades a plot
    # that already tiles correctly against trusted geometry.
    try:
        topo = _corroborate_by_topology(results)
        if topo:
            _resolve_footprint_conflicts(results)
            _log.info("Topology corroboration upgraded %d REVIEW->ACCEPT: %s",
                      len(topo), ", ".join(sorted(topo, key=lambda s: (len(s), s))))
    except Exception as exc:  # noqa: BLE001
        _log.error("Topology corroboration failed: %s", exc)

    # Cadastral reconciliation (a final CHECK, never a new placement source): re-judge
    # each REVIEW plot the cadastral pass LOCATED (cadastral_*_located) against the
    # EXACT cadastral ACCEPT gates, now that the confident tiling is FINAL and stable.
    # A plot the cadastral pass held at REVIEW only because of a transient mid-pass
    # overlap (against a footprint a later pass demoted/moved) is recovered iff its
    # fresh rigid fit independently clears ALL the same gates the existing
    # ACCEPT_CADASTRAL plots cleared. Reuses only already-trusted gates (no new
    # tolerance), so it cannot create a false positive. No-op without a cadastral src.
    if cadastral_source is not None:
        try:
            rec = _corroborate_geometric_cadastral(
                results, cadastral_source, surveyor, output_dir, crs)
            if rec:
                _resolve_footprint_conflicts(results)
                _log.info("Cadastral reconciliation upgraded %d REVIEW->ACCEPT: %s",
                          len(rec), ", ".join(sorted(rec, key=lambda s: (len(s), s))))
        except Exception as exc:  # noqa: BLE001
            _log.error("Cadastral reconciliation failed: %s", exc)

    # THE M2 DELIVERABLE: ONE combined file with ALL FMBs clubbed into the raw
    # data file (the surveyor DXF is the base canvas). Plots WITH field control
    # (ACCEPT/ACCEPT_SEEDED/REVIEW) or a cadastral parcel (ACCEPT_CADASTRAL) are
    # merged at their georeferenced UTM position. Only plots with NO placement at
    # all are STAGED to-scale in a labelled band east of the survey (each on its
    # own STAGED_FMB_<survey> layer), so every plot is present and seedable.
    _ACCEPT = ("ACCEPT", "ACCEPT_SEEDED", "ACCEPT_CADASTRAL")
    combined_village_path: Path | None = None
    placed = [r.output_file for r in results
              if r.recommendation in _ACCEPT and r.output_file]
    review = [(r.output_file, r.survey_number) for r in results
              if r.recommendation == "REVIEW" and r.output_file]
    staged = [(r.m1_file, r.survey_number) for r in results
              if r.recommendation not in _ACCEPT and r.recommendation != "REVIEW"
              and r.m1_file]
    if placed or review or staged:
        combined = output_dir / "combined_village.dxf"
        # BUG #7: strip the dense surveyor markers from the base canvas (the yellow
        # Point_Code stone labels + the stone POINTs on "0" + feature labels) so only
        # the FMB plots read clearly. The corridor SITE DATA LINE and towers stay as
        # light spatial reference. BUG #2: REVIEW plots go on their own layers.
        def _write_combined(path):
            build_full_combined_dxf(surveyor_dxf, placed, staged, path, crs,
                                    base_layers_hide=["Point_Code", "0", "FEATURE_LABEL"],
                                    review_specs=review)
        try:
            _write_combined(combined)
        except PermissionError:
            # D2: target is open in AutoCAD/QGIS -> write a timestamped sibling
            # instead of silently leaving a stale file in place.
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback = output_dir / f"combined_village_{ts}.dxf"
            _log.warning("Cannot write %s (open in another app?) -> writing %s",
                         combined, fallback)
            try:
                _write_combined(fallback)
                combined = fallback
            except Exception as exc:  # noqa: BLE001
                _log.error("Fallback combined-DXF write also failed: %s", exc)
                combined = None
        except Exception as exc:  # noqa: BLE001
            _log.error("Failed to build combined DXF: %s", exc)
            combined = None
        if combined is not None:
            _log.info("Combined village DXF: %s (%d placed + %d review + %d staged "
                      "= %d FMBs clubbed into raw data)", combined,
                      len(placed), len(review), len(staged),
                      len(placed) + len(review) + len(staged))
            # M3 POLISH: label each placed parcel with its survey number + ground area
            # (non-destructive -- a new PARCEL_ANNOTATION layer; geometry never warped).
            try:
                from ..m3_assemble import annotate_combined
                conf_specs = [(r.output_file, r.survey_number) for r in results
                              if r.recommendation in _ACCEPT and r.output_file]
                annotate_combined(combined, conf_specs)
            except Exception as exc:  # noqa: BLE001 - annotation must never break the job
                _log.error("M3 annotation failed: %s", exc)
            combined_village_path = combined

    # Summary by disposition (every plot is accounted for: ACCEPT georeferenced,
    # REVIEW needs a human, NO_COVERAGE = surveyor never traced this plot).
    disp = {}
    for r in results:
        disp[r.recommendation] = disp.get(r.recommendation, 0) + 1

    _log.info("=" * 60)
    _log.info("M2 PIPELINE SUMMARY (%d plots): %s", len(results),
              "  ".join(f"{k}={v}" for k, v in sorted(disp.items())))
    for r in results:
        if r.recommendation in ("ACCEPT", "REVIEW"):
            _log.info("  %-11s %s  cov=%.0f%% inliers=%d/%d",
                      r.recommendation, r.survey_number,
                      100 * r.chain_coverage, r.n_inliers, r.n_corners)

    # Operator handoff: a per-plot CSV worklist (the client's real Stage-3 process
    # is manual seeding, so this report is the deliverable that makes the remainder
    # actionable in minutes).
    try:
        write_quality_report(results, output_dir)
    except Exception as exc:  # noqa: BLE001
        _log.error("Quality report failed: %s", exc)

    # Runtime AGENT LAYER: self-verify the job (FP-safety + module invariants), emit the
    # minimal-extra-input worklist that closes the gap to 100%, and narrate -- so the
    # shipped product catches its own errors without a human in the loop. Agents only
    # verify / request input / narrate; ACCEPT stays decided by the math gates above.
    try:
        from ...agents import run_agent_layer
        run_agent_layer(results, output_dir, context={
            "crs": crs, "village": village,
            # passed through so the reasoning agent's AUTO proposals can be re-gated
            # (propose -> re-run deterministic fix -> gate decides). Opt-in via
            # enable_auto_regate so a batch run doesn't recompute OCR on an unchanged source.
            "cadastral_source": cadastral_source, "surveyor": surveyor,
            "enable_auto_regate": False})
    except Exception as exc:  # noqa: BLE001 - the agent layer must never break a job
        _log.error("Agent layer failed: %s", exc)

    # C3: per-plot diagnosis of everything NOT confidently placed, so a reviewer
    # sees WHY each REVIEW/NO_COVERAGE plot landed there (gate that failed / no S3
    # label / off corridor), not just a bucket count.
    diag = _diagnose_unplaced(results)
    if diag:
        _log.warning("Plots needing attention (REVIEW / NO_COVERAGE):\n%s", diag)

    # M4 DELIVERABLE: the village area-statement PDF + Excel + a delivery zip that
    # bundles the combined DXF. Built straight from the dispositioned results (areas
    # from each placed plot's verified boundary), so a corridor run ships a client
    # package without going through the API/Job path. Best-effort -- never breaks a job.
    try:
        from ..m4_report.village import build_village_delivery
        build_village_delivery(results, combined_village_path, output_dir,
                               village=village, crs=crs)
    except Exception as exc:  # noqa: BLE001
        _log.error("M4 village delivery failed: %s", exc)

    # VISUAL QA: the human-auditable false-positive backstop. Draw every placed FMB
    # against its authoritative S3/cadastral parcel + the surveyor traced lines, so a
    # plot the gates passed but that sits in the WRONG place is visible at a glance
    # (red MISPLACED connector). One image per raw-data-file job. Best-effort.
    try:
        from .qa_render import render_qa_overlay
        src_name = (Path(surveyor.source_file).name
                    if getattr(surveyor, "source_file", "") else village)
        render_qa_overlay(results, output_dir / "qa_overlay.png",
                          surveyor=surveyor, cadastral_source=cadastral_source,
                          title=f"LandIntel QA - {src_name}")
    except Exception as exc:  # noqa: BLE001
        _log.error("QA overlay render failed: %s", exc)

    return results


def _shared_edge_length(a, b, tol: float = 3.0) -> float:
    """Length of boundary shared by two placed footprints: the part of polygon a's
    boundary lying within ``tol`` of polygon b's boundary (and vice-versa). Real
    adjacent parcels share an EDGE (a line), coincidental ones touch at a point or
    not at all. Returns the shared length in metres (0 if not genuinely adjacent)."""
    try:
        ab = a.boundary.intersection(b.buffer(tol))
        ba = b.boundary.intersection(a.buffer(tol))
        return min(ab.length, ba.length)
    except Exception:  # noqa: BLE001
        return 0.0


def _corroborate_by_topology(results: list[GeorefResult],
                             min_shared_m: float = 20.0) -> list[str]:
    """Upgrade a located REVIEW plot to ACCEPT iff its footprint genuinely ABUTS a
    CONFIDENT plot (shares >= min_shared_m of edge) AND does not overlap any confident
    footprint. Independent corroboration (it tiles correctly against trusted geometry),
    never a propagated position -- so it cannot inherit a wrong-seat error. Mutates
    ``results``; returns the upgraded survey numbers.

    min_shared_m is deliberately a REAL shared property line (these parcels have
    ~100 m+ sides), not a corner-graze: a short tangent overlap is the coincidental
    case and would be the place an FP could hide, so the bar is conservative.
    """
    conf_states = ("ACCEPT", "ACCEPT_SEEDED", "ACCEPT_CADASTRAL")
    conf = []
    for r in results:
        if r.recommendation in conf_states and r.output_file:
            fp = _footprint_polygon(r.output_file)
            if fp is not None and fp.is_valid and fp.area > 0:
                conf.append((r.survey_number, fp))
    upgraded = []
    for r in results:
        if r.recommendation != "REVIEW" or not r.output_file:
            continue
        # The plot's OWN fit must be sane first: a garbage rigid fit (no real parcel
        # -> huge residual / oversize) can still graze a neighbour's edge by chance,
        # which would be a false positive. A correct placement has corners landing
        # near its parcel (residual small). 1019 (resid 106 m, area_ratio 16.7) is
        # exactly this trap and must NOT be upgraded; 1027 (resid 17 m) is genuine.
        if r.cad_residual > TOPO_MAX_RESID:
            continue
        fp = _footprint_polygon(r.output_file)
        if fp is None or not fp.is_valid or fp.area <= 0:
            continue
        # Must not overlap any confident footprint (real parcels tile, never stack).
        overlap = any(
            fp.intersection(cp).area / max(min(fp.area, cp.area), 1e-9) > FOOTPRINT_CONFLICT
            for _, cp in conf if fp.intersects(cp))
        if overlap:
            continue
        # Must share a real EDGE (not a point) with at least one confident neighbour.
        best = max((_shared_edge_length(fp, cp) for _, cp in conf), default=0.0)
        if best >= min_shared_m:
            r.recommendation = "ACCEPT"
            r.match_method = (r.match_method or "located") + "+topology"
            upgraded.append(r.survey_number)
            conf.append((r.survey_number, fp))
            _log.info("Topology %s -> ACCEPT (shares %.0fm edge with a confident "
                      "neighbour, no overlap)", r.survey_number, best)
    return upgraded


def _corroborate_geometric_cadastral(
    results: list[GeorefResult],
    cadastral_source: object,
    surveyor: SurveyorData,
    output_dir: Path,
    crs: str,
) -> list[str]:
    """Final reconciliation (a CHECK, never a new placement source): re-judge each
    REVIEW plot the cadastral pass LOCATED (``cadastral_*_located``) against the EXACT
    cadastral ACCEPT gates, now that the confident tiling is FINAL and stable.

    Why this is needed: ``_place_by_cadastral`` decides ACCEPT/REVIEW against the
    placed set AS IT STANDS mid-pass. A plot can be held at REVIEW by a transient
    overlap against a footprint that a LATER pass (propagation / edge-register /
    topology) demotes or moves -- so an otherwise-perfect cadastral fit is stranded.
    This pass re-runs the identical fit and the identical gates once the tiling no
    longer changes, and upgrades the plot iff its FRESH rigid fit independently clears
    ALL of: rigid method, area-ratio band, scale band, flip gate, corner residual,
    on-corridor, AND non-overlap with the FINAL confident footprints.

    FP-safety: it reuses ONLY the already-trusted gates that the existing
    ACCEPT_CADASTRAL plots passed -- it introduces NO new tolerance and no geometric
    guessing (identity is keyed by survey number, geometry is the rigid FMB). It can
    only turn a LOCATED REVIEW into ACCEPT when that plot meets the full conjunctive
    cadastral bar, so it cannot create a false positive. Mutates ``results``; returns
    the upgraded survey numbers.

    NOTE on the name: the validated INGUR signal is NOT field-stone-geometry vs
    cadastral cross-corroboration. Measured on the 6 stuck off-corridor plots, no
    true-seat geometric congruence exists -- their windowed geometric matches are
    chance subsets 60-270 m off the seat, scale != 1, disagreeing with the cadastral
    pose by 20-110 deg -- so corroborating them would be corroborating two independent
    WRONG answers. The recoverable signal is the cadastral fit itself clearing the full
    ACCEPT bar against the final, stable tiling.
    """
    from ..m5_cadastral.fit import fit_plot_to_parcel
    if surveyor._stone_tree is None:
        surveyor.build_index()
    label_point = getattr(cadastral_source, "label_point", lambda s: None)
    conf_states = ("ACCEPT", "ACCEPT_SEEDED", "ACCEPT_CADASTRAL")

    # Confident footprints in the FINAL tiling (the stable overlap reference).
    conf: list[tuple[str, object]] = []
    for r in results:
        if r.recommendation in conf_states and r.output_file:
            fp = _footprint_polygon(r.output_file)
            if fp is not None and fp.is_valid and fp.area > 0:
                conf.append((r.survey_number, fp))

    from shapely.geometry import Polygon as _Poly
    upgraded: list[str] = []
    for r in results:
        # Only plots the cadastral pass already LOCATED at a rigid pose (identity keyed
        # by survey number; parcel/scale/orientation already computed) -- never a
        # corridor REVIEW with no parcel, never a NO_COVERAGE plot.
        if r.recommendation != "REVIEW" or not (r.match_method or "").startswith("cadastral"):
            continue
        parcel = cadastral_source.get(r.survey_number)
        if parcel is None:
            continue
        try:
            m1 = extract_m1_dxf(r.m1_file)
        except Exception:  # noqa: BLE001
            continue
        fit = fit_plot_to_parcel(m1, parcel, anchor=label_point(r.survey_number))
        if fit is None:
            continue
        fit = _refine_against_stones(fit, m1, surveyor, parcel)
        ring = fit.adjusted[np.array(m1.outer_stone_indices)]
        centroid = ring.mean(axis=0)
        if float(surveyor._stone_tree.query(centroid)[0]) > CAD_CORRIDOR_MAX:
            continue
        poly = _Poly([(float(x), float(y)) for x, y in ring])
        if not poly.is_valid:
            poly = poly.buffer(0)
        overlap = any(
            poly.intersects(cp) and poly.intersection(cp).area
            / max(min(poly.area, cp.area), 1e-9) > FOOTPRINT_CONFLICT
            for sn, cp in conf if sn != r.survey_number)
        passes = (fit.method == "rigid"
                  and CAD_AREA_LO <= fit.area_ratio <= CAD_AREA_HI
                  and CAD_SCALE_LO <= fit.s <= CAD_SCALE_HI
                  and fit.orientation_ok
                  and fit.rot_residual <= CAD_ROT_RESID_MAX
                  and not overlap)
        if not passes:
            continue
        out = output_dir / f"georef_{Path(r.m1_file).stem}.dxf"
        try:
            write_georef_dxf(
                m1_dxf_path=r.m1_file, output_path=out,
                adjusted_stone_positions=fit.adjusted,
                original_stone_positions=m1.stone_positions(),
                stone_label_to_index={s.label: s.index for s in m1.stones},
                R=fit.R, s=fit.s, t=fit.t, crs=crs,
                corner_ring=m1.outer_stone_indices)
        except Exception as exc:  # noqa: BLE001
            _log.warning("reconcile write failed for %s: %s", r.survey_number, exc)
            continue
        r.matched = True
        r.output_file = str(out)
        r.n_corners = len(m1.outer_stone_indices)
        r.n_inliers = fit.n_inliers
        r.chain_coverage = fit.area_ratio
        r.recommendation = "ACCEPT_CADASTRAL"
        r.match_method = "cadastral_rigid_corroborated"
        conf.append((r.survey_number, poly))
        upgraded.append(r.survey_number)
        _log.info("Reconciled %s -> ACCEPT_CADASTRAL (area_ratio=%.2f, scale=%.3f, "
                  "rot_resid=%.1fm; passes all cadastral gates on the final tiling)",
                  r.survey_number, fit.area_ratio, fit.s, fit.rot_residual)
    return upgraded


def write_quality_report(results: list[GeorefResult], output_dir: Path) -> Path:
    """Per-plot operator report (CSV): disposition, method, confidence signals, the
    located UTM centroid, and the action a human needs to take. This turns the
    REVIEW/staged remainder into a short, seedable worklist -- the client's real
    Stage-3 workflow is manual stone-seeding, so the report IS the handoff.
    """
    import csv
    conf_states = ("ACCEPT", "ACCEPT_SEEDED", "ACCEPT_CADASTRAL")
    path = output_dir / "quality_report.csv"
    rows = []
    for r in results:
        cx = cy = ""
        if r.output_file:
            fp = _footprint_polygon(r.output_file)
            if fp is not None and fp.is_valid and fp.area > 0:
                cx, cy = f"{fp.centroid.x:.1f}", f"{fp.centroid.y:.1f}"
        cadastral = (r.match_method or "").startswith("cadastral")
        signal = (f"area_ratio={r.chain_coverage:.2f}" if cadastral
                  else f"chain_cov={r.chain_coverage:.0%},inliers={r.n_inliers}/{r.n_corners}")
        if r.recommendation in conf_states:
            action = "none (georeferenced)"
        elif r.recommendation == "REVIEW":
            action = "operator: confirm placement (located, on REVIEW_FMB layer)"
        else:
            action = "operator: seed 2 corner->UTM points (no auto position)"
        rows.append({
            "survey": r.survey_number, "disposition": r.recommendation,
            "method": r.match_method or "", "signal": signal if r.matched else "",
            "utm_x": cx, "utm_y": cy, "action": action,
            "note": r.error or "",
        })
    rows.sort(key=lambda d: (d["disposition"], len(d["survey"]), d["survey"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["survey"])
        w.writeheader()
        w.writerows(rows)
    n_conf = sum(1 for r in results if r.recommendation in conf_states)
    _log.info("Quality report: %s (%d/%d georeferenced, %d need operator action)",
              path, n_conf, len(results), len(results) - n_conf)
    return path


def _diagnose_unplaced(results: list[GeorefResult]) -> str:
    """One line per REVIEW/NO_COVERAGE plot explaining the disposition."""
    lines = []
    for r in results:
        if r.recommendation not in ("REVIEW", "NO_COVERAGE"):
            continue
        bits = [f"  {(r.survey_number or '?'):>6s}: {r.recommendation:<11s}"]
        if r.match_method:
            bits.append(f"via {r.match_method}")
        if r.recommendation == "REVIEW" and r.chain_coverage:
            # cadastral path stores area_ratio here; corridor path stores coverage.
            if (r.match_method or "").startswith("cadastral"):
                bits.append(f"area_ratio={r.chain_coverage:.2f}")
            else:
                bits.append(f"cov={r.chain_coverage:.0%}")
        if r.error:
            bits.append(f"({r.error})")
        lines.append(" ".join(bits))
    return "\n".join(lines)


def _main_seed_place(argv: list[str]) -> None:
    """CLI: `seed-place` -- place ONE plot from 2 operator-given corner->UTM pairs."""
    p = argparse.ArgumentParser(
        prog="pipeline seed-place",
        description="Place one corridor-clipped plot from 2 known corner stones "
                    "(Stage-3 manual fallback, automated).")
    p.add_argument("--m1", required=True, help="M1 DXF of the plot to place")
    p.add_argument("--surveyor", required=True, help="Surveyor (raw data) DXF")
    p.add_argument("--corner-a", required=True, help="STONES label of corner A")
    p.add_argument("--corner-b", required=True, help="STONES label of corner B")
    p.add_argument("--utm-a", required=True, help='UTM of corner A, "x,y"')
    p.add_argument("--utm-b", required=True, help='UTM of corner B, "x,y"')
    p.add_argument("--output-dir", required=True)
    p.add_argument("--crs", default="EPSG:32643")
    p.add_argument("--verbose", "-v", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    def _xy(s):
        x, y = s.split(",")
        return (float(x), float(y))

    surveyor = extract_surveyor(a.surveyor)
    surveyor.build_index()
    r = seed_place(a.m1, surveyor, a.corner_a, a.corner_b,
                   _xy(a.utm_a), _xy(a.utm_b), a.output_dir, a.crs)
    if r.verify_result:
        print_verify_result(r.verify_result)
    print(f"\nseed-place: survey {r.survey_number} -> {r.recommendation} "
          f"(snapped {r.n_inliers} stones, field_res<={r.field_residual_max:.3f}m, "
          f"chain_cov={100*r.chain_coverage:.0f}%)\n  {r.output_file or r.error}")


def main():
    # Subcommand dispatch kept backward-compatible: the original flag-only form
    # (`--surveyor ... --m1-dir ...`) still works; `seed-place` is an opt-in verb.
    if len(sys.argv) > 1 and sys.argv[1] == "seed-place":
        _main_seed_place(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description="M2 Georeferencing Pipeline")
    parser.add_argument("--surveyor", required=True, help="Surveyor DXF file")
    parser.add_argument("--m1-dir", required=True, help="Directory with M1 DXF files")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--crs", default="EPSG:32643", help="CRS (default: EPSG:32643)")
    parser.add_argument("--schedule", default=None,
                        help="Corridor land-schedule DXF (its SURVEY NUMBER layer "
                             "is the identity gate -- only those plots are matched, "
                             "removing geometric false positives)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    m1_dir = Path(args.m1_dir)
    m1_files = sorted(m1_dir.glob("*.dxf"))

    if not m1_files:
        _log.error("No DXF files found in %s", args.m1_dir)
        sys.exit(1)

    _log.info("Found %d M1 DXF files in %s", len(m1_files), args.m1_dir)

    corridor_surveys = None
    if args.schedule:
        from .extract_surveyor import extract_corridor_surveys
        corridor_surveys = extract_corridor_surveys(args.schedule)
        _log.info("Corridor identity gate: %d survey numbers from %s",
                  len(corridor_surveys), args.schedule)

    results = georef_pipeline(args.surveyor, m1_files, args.output_dir, args.crs,
                              corridor_surveys=corridor_surveys)

    # Print summary table
    print(f"\n{'Survey#':<12} {'Match':<7} {'Verify':<7} {'FP':<8} {'NB':<8} "
          f"{'Field_m':<10} {'File'}")
    print("-" * 85)
    for r in results:
        sn = r.survey_number or Path(r.m1_file).stem[-10:]
        m = "YES" if r.matched else "NO"
        v = "YES" if (r.verify_result and r.verify_result.all_passed) else "NO"
        fp = f"{r.fingerprint_score:.2f}" if r.fingerprint_score < 100 else "-"
        nb = f"{r.neighborhood_score:.2f}" if r.neighborhood_score < 100 else "-"
        fm = f"{r.field_residual_mean:.4f}" if r.field_residual_mean < 100 else "-"
        print(f"{sn:<12} {m:<7} {v:<7} {fp:<8} {nb:<8} {fm:<10} "
              f"{Path(r.m1_file).name}")


if __name__ == "__main__":
    main()
