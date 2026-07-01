"""M2 (new) -- georeference FMB DXFs WITHOUT a surveyor file, then club them.

Input  : M1 FMB DXFs (relative metres) + optional cadastral source (TNGIS tiles /
         client vector) + optional GPS control points.
Output : one CLUBBED georeferenced DXF (+ GeoJSON + points CSV) and a per-plot
         disposition list.

ALL methods, layered, cross-checking each other (the user's directive -- never bet
on one source):
  * cadastral_seat   survey# -> official UTM parcel, rigid-gated   (absolute)
  * gps_seat         operator control points, 2-corner similarity  (absolute)
  * relative_club    FMB-to-FMB shared-edge corroboration + gated propagation

0-FP discipline (the math gates decide ACCEPT; no method self-promotes):
  ACCEPT_SEEDED  GPS seat with an adequate baseline (human-supplied identity).
  ACCEPT         cadastral seat passing the strict rigid shape gate; OR a cadastral
                 placement (even below the size gate) whose absolute boundary
                 COINCIDES with an already-ACCEPTed neighbour's edge (two unrelated
                 sources agreeing -> confirmed); OR a propagated placement that tiles
                 a seated neighbour without overlapping any placed plot.
  REVIEW         located but unconfirmed (cadastral below gate, no corroboration;
                 weak GPS baseline; demoted footprint conflict).
  NO_COVERAGE    no method could place it -> staged to-scale, never guessed.
A global footprint pass keeps the ACCEPT set a non-overlapping tiling.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..m2_georef.extract_m1 import M1PlotData, extract_m1_dxf
from . import relative_club as RC
from .cadastral_seat import cadastral_seat
from .club_output import (
    club_dxf,
    write_geojson,
    write_plot_dxf,
    write_points_csv,
)
from .gps_seat import gps_seat
from .placement import CandidatePlacement, ClubResult
from .verify import format_club_verify, gate_results, write_club_verify_sidecar

_log = logging.getLogger(__name__)

DEFAULT_CRS = "EPSG:32643"

# Centroid agreement between two independent absolute methods (cadastral vs GPS) (m).
METHOD_AGREE_TOL = 25.0
# Interior overlap (fraction of smaller) above which two ACCEPTs cannot both stand.
FOOTPRINT_CONFLICT = 0.20


def _confidence(r: ClubResult) -> float:
    rec, method = r.recommendation, r.method
    corro = bool(r.corroborated_by)
    if rec == "ACCEPT_SEEDED":
        return 0.97
    if rec == "ACCEPT":
        if method == "cadastral":
            return 0.95 if corro else 0.85
        if method == "cadastral_corroborated":
            return 0.90
        if method == "propagated":
            return 0.60
        return 0.80
    if rec == "REVIEW":
        return 0.30
    return 0.0


def _extract_all(m1_dxf_paths) -> dict[str, tuple[M1PlotData, str]]:
    plots: dict[str, tuple[M1PlotData, str]] = {}
    for p in m1_dxf_paths:
        p = str(p)
        try:
            m1 = extract_m1_dxf(p)
        except Exception as exc:  # noqa: BLE001
            _log.warning("M2 club: cannot extract %s: %s", p, exc)
            continue
        sn = m1.survey_number or Path(p).stem
        if sn in plots:
            _log.warning("M2 club: duplicate survey %s (%s) -- keeping first", sn, p)
            continue
        plots[sn] = (m1, p)
    return plots


def _gather_candidates(m1, cadastral_source, gps_control) -> dict[str, CandidatePlacement]:
    cands: dict[str, CandidatePlacement] = {}
    c = cadastral_seat(m1, cadastral_source)
    if c is not None:
        cands["cadastral"] = c
    ctrl = (gps_control or {}).get(m1.survey_number) or (gps_control or {}).get(
        str(m1.survey_number))
    if ctrl:
        g = gps_seat(m1, ctrl)
        if g is not None:
            cands["gps_seed"] = g
    return cands


def _initial_disposition(r: ClubResult) -> None:
    """Pick the best ABSOLUTE candidate and set a provisional disposition."""
    cands = r.candidates
    gps = cands.get("gps_seed")
    cad = cands.get("cadastral")

    # GPS = human-supplied identity -> strongest. ACCEPT_SEEDED if baseline is sound.
    if gps is not None:
        r.placement = gps
        r.method = "gps_seed"
        r.recommendation = "ACCEPT_SEEDED" if gps.seed_ok else "REVIEW"
        if not gps.seed_ok:
            r.note = gps.note
        # Cross-check: a cadastral seat that AGREES corroborates; gross disagreement
        # is noted but GPS (human identity) still wins.
        if cad is not None:
            d = _centroid_dist(gps, cad)
            if d <= METHOD_AGREE_TOL:
                r.corroborated_by.append("cadastral")
            else:
                r.note = (r.note + "; " if r.note else "") + (
                    f"cadastral disagrees by {d:.0f} m")
        return

    if cad is not None:
        r.placement = cad
        r.method = "cadastral"
        if cad.passes_gate:
            r.recommendation = "ACCEPT"
        else:
            r.recommendation = "REVIEW"
            r.note = cad.note
        return

    r.recommendation = "NO_COVERAGE"


def _centroid_dist(a: CandidatePlacement, b: CandidatePlacement) -> float:
    ca, cb = a.centroid(), b.centroid()
    return float(((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5)


def _resolve_conflicts(results: list[ClubResult]) -> None:
    """Demote the lower-confidence plot of any overlapping ACCEPT pair to REVIEW so
    the ACCEPT set is a non-overlapping tiling (real parcels tile)."""
    placed = [r for r in results if r.placed and r.placement is not None]
    polys = {id(r): r.placement.footprint() for r in placed}
    kept: list[ClubResult] = []
    for r in sorted(placed, key=_confidence, reverse=True):
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
            r.note = (r.note + "; " if r.note else "") + (
                f"footprint overlaps higher-confidence {conflict.survey_number}; demoted")
        else:
            kept.append(r)


def club_pipeline(
    m1_dxf_paths: list[str | Path],
    output_dir: str | Path,
    crs: str = DEFAULT_CRS,
    cadastral_source: object | None = None,
    gps_control: dict[str, list[tuple[str, tuple[float, float]]]] | None = None,
    village: str | None = None,
) -> list[ClubResult]:
    """Run the new M2: georeference FMB DXFs (no surveyor) and club them.

    Returns one ``ClubResult`` per input FMB. Side effects: per-plot georef DXFs,
    a clubbed ``clubbed_village.dxf``, ``clubbed.geojson`` and ``clubbed_points.csv``
    in ``output_dir``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots = _extract_all(m1_dxf_paths)
    m1s = {sn: m1 for sn, (m1, _p) in plots.items()}
    paths = {sn: p for sn, (_m1, p) in plots.items()}

    results: dict[str, ClubResult] = {}
    for sn, (m1, p) in plots.items():
        r = ClubResult(m1_file=p, survey_number=sn)
        r.candidates = _gather_candidates(m1, cadastral_source, gps_control)
        _initial_disposition(r)
        results[sn] = r

    # --- Corroboration: a below-gate cadastral REVIEW whose absolute boundary
    # coincides with an ACCEPTed neighbour's edge is confirmed by two unrelated
    # sources -> upgrade to ACCEPT (label-free geometry, so never a false upgrade).
    accept_now = {sn: r for sn, r in results.items()
                  if r.placed and r.placement is not None}
    for sn, r in results.items():
        if r.recommendation != "REVIEW" or r.method != "cadastral" or r.placement is None:
            continue
        for asn, ar in accept_now.items():
            if asn == sn or ar.placement is None:
                continue
            if RC.shares_edge(r.placement, ar.placement):
                r.recommendation = "ACCEPT"
                r.method = "cadastral_corroborated"
                r.corroborated_by.append(asn)
                _log.info("Corroborated %s -> ACCEPT (shares edge with %s)", sn, asn)
                break

    # --- Propagation rounds: seat un-placed plots from seated neighbours by their
    # shared edge (gated: scale ~1 + tiling non-overlap). Iterate so a newly
    # propagated plot can in turn seat its own neighbours.
    for _round in range(4):
        seated = {sn: r for sn, r in results.items()
                  if r.placed and r.placement is not None}
        placed_fps = [r.placement.footprint() for r in seated.values()]
        changed = False
        for sn, r in results.items():
            if r.recommendation != "NO_COVERAGE":
                continue
            m1_b = m1s[sn]
            for asn, ar in seated.items():
                prop = RC.propagate_from_seated(
                    m1_b, ar.placement, m1s[asn],
                    [fp for fp in placed_fps if fp is not None])
                if prop is not None:
                    r.placement = prop
                    r.method = "propagated"
                    r.recommendation = "ACCEPT"
                    r.corroborated_by.append(asn)
                    r.note = prop.note
                    _log.info("Propagated %s -> ACCEPT (from %s)", sn, asn)
                    changed = True
                    break
        if not changed:
            break

    # --- Record label-free corroboration graph among all seated plots (confidence).
    seated = {sn: r.placement for sn, r in results.items()
              if r.placed and r.placement is not None}
    corro = RC.corroborate_seated(seated, m1s)
    for sn, r in results.items():
        for nb in corro.get(sn, []):
            if nb not in r.corroborated_by:
                r.corroborated_by.append(nb)

    result_list = list(results.values())

    # --- Global tiling: ACCEPT set must not self-overlap.
    _resolve_conflicts(result_list)

    # --- Verification gate (DEMOTE-ONLY): any ACCEPT failing a HARD geometry check
    # (closure/area, UTM range, rigid scale, stone count) drops to REVIEW. Never
    # promotes -- math finds the failure, the human re-confirms. The returned
    # ClubVerifyResult reflects the post-demotion ACCEPT set.
    verify_result = gate_results(result_list, crs)

    for r in result_list:
        r.confidence = _confidence(r)

    # --- Write per-plot DXFs + club into one file.
    placed_specs, review_specs, staged_specs = [], [], []
    for r in result_list:
        if r.placement is not None and r.recommendation in (
                "ACCEPT", "ACCEPT_SEEDED", "REVIEW"):
            try:
                out = output_dir / f"georef_{Path(r.m1_file).stem}.dxf"
                write_plot_dxf(r.m1_file, r.placement, out, crs)
                r.output_file = str(out)
                if r.recommendation == "REVIEW":
                    review_specs.append((str(out), r.survey_number))
                else:
                    placed_specs.append((str(out), r.survey_number))
            except Exception as exc:  # noqa: BLE001
                _log.warning("club: write failed for %s: %s", r.survey_number, exc)
                staged_specs.append((r.m1_file, r.survey_number))
        else:
            staged_specs.append((r.m1_file, r.survey_number))

    club_dxf(placed_specs, staged_specs, output_dir / "clubbed_village.dxf",
             crs=crs, review_specs=review_specs)
    write_geojson(result_list, output_dir / "clubbed.geojson", crs=crs)
    write_points_csv(result_list, output_dir / "clubbed_points.csv", crs=crs)

    # --- Verification sidecar + summary (the gate already ran above; this records it).
    try:
        write_club_verify_sidecar(verify_result, output_dir / "clubbed.verify.txt")
    except Exception as exc:  # noqa: BLE001 - sidecar is best-effort
        _log.warning("club: could not write verify sidecar: %s", exc)
    _log.info("M2 club verify: %s",
              "PASS" if verify_result.all_passed else "FAIL ("
              + ", ".join(verify_result.failed_names()) + ")")
    _log.debug("M2 club verify detail:\n%s", format_club_verify(verify_result))

    n_acc = sum(1 for r in result_list if r.placed)
    _log.info("M2 club done: %d FMBs, %d placed (ACCEPT), %d review, %d no-coverage",
              len(result_list), n_acc,
              sum(1 for r in result_list if r.recommendation == "REVIEW"),
              sum(1 for r in result_list if r.recommendation == "NO_COVERAGE"))
    return result_list
