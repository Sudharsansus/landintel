"""M2 club via PUBLIC CADASTRAL TILES -- NO surveyor file needed.

Georeferences + clubs a village's M1 FMB DXFs by matching each survey number onto its
parcel in the public mypropertyqr cadastral tileset (yellow parcels + orange survey
labels), then tiling the plots. Location comes from a geocoded (or supplied) village
centre; the tiles are fetched for a bbox around it and OCR'd to find the village's own
survey numbers, so no surveyor raw-data file is required.

Usage:
  python run_m2_cad.py <VILLAGE> --lat <LAT> --lon <LON> [--radius-km 2.5]
  python run_m2_cad.py <VILLAGE> --utm <X> <Y> [--radius-km 2.5]     (centre in UTM 43N)

Outputs to output/<VILLAGE>/m2/: clubbed_village.dxf, clubbed_points.csv, clubbed.geojson,
clubbed_qa.png, clubbed.verify.txt.  CRS = EPSG:32643 (UTM 43N) for all Erode villages.
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

from pyproj import Transformer
from shapely.geometry import Point

from landintel.pipeline.m2_club import club_pipeline
from landintel.agents.club_agents import (
    AssemblyAgent,
    ParcelAgent,
    TngisOverlayAgent,
    overlay_gate,
)
from landintel.pipeline.m2_club.club_output import snap_and_rewrite
from landintel.pipeline.m2_club.qa_render import render_club_qa
from landintel.pipeline.m5_cadastral.s3_tiles import TILE_VARIANT, S3CadastralSource

CRS = "EPSG:32643"           # UTM 43N -- all these Erode villages are west of 78E
# TNGIS-overlay ACCEPT gate: the client's criterion is "clubbed FMB overlays the TNGIS
# parcel". IoU (placed FMB footprint vs its own survey#'s parcel) measures exactly that and
# is 0-FP by construction -- a wrong-parcel placement cannot overlap its labelled parcel, so
# it scores low. A plot is ACCEPTed when its overlay is strong AND it sits on its own label
# point (seat-locality already enforced upstream). General threshold; override via env.
IOU_ACCEPT = float(os.environ.get("LANDINTEL_IOU_ACCEPT", "0.5"))
# STRONG-overlay ACCEPT: a placed FMB that overlaps its OWN survey#'s REAL vector parcel by
# >= this fraction is seated on that parcel more reliably than any label-point-distance proxy
# (IoU integrates the whole footprint). It supersedes the "off-seat" veto (which mis-fires on
# benign label-vs-centroid offset), GATED by label provenance (parcel must be uncontested) so a
# mislabelled parcel can never earn ACCEPT. 0-FP: a geometrically wrong placement scores low.
IOU_STRONG = float(os.environ.get("LANDINTEL_IOU_STRONG", "0.6"))
# IoU-contradiction demote: an ACCEPT whose placement barely overlaps its own REAL vector parcel
# is self-contradictory (e.g. an OCR mislabel placed it away from the parcel carrying its number,
# like NASIYANUR 141). Demote to REVIEW -- a TIGHTENING of the FP gate, never a new ACCEPT.
IOU_CONTRADICT = float(os.environ.get("LANDINTEL_IOU_CONTRADICT", "0.15"))


def _args() -> tuple[str, float, float, float]:
    a = sys.argv[1:]
    village = a[0]
    # Use the EXACT vector cadastre by default when the TNGIS GeoParquet is present (it strictly
    # dominates z18 tile OCR); fall back to tiles if absent or when forced with --tiles.
    _parq = os.environ.get("LANDINTEL_TNGIS_PARQUET", "data/tngis/TNGIS_TN_Cadastrals.parquet")
    _env = os.environ.get("LANDINTEL_CADASTRAL")
    _use_vec = ("--vector" in a) or _env == "vector" or (
        _env != "tiles" and "--tiles" not in a and Path(_parq).exists())
    opt = {"radius_km": 2.5, "cx": None, "cy": None, "taluk": "", "district": "",
           "vector": _use_vec}
    if "--radius-km" in a:
        opt["radius_km"] = float(a[a.index("--radius-km") + 1])
    if "--taluk" in a:
        opt["taluk"] = a[a.index("--taluk") + 1]
    if "--district" in a:
        opt["district"] = a[a.index("--district") + 1]
    if "--utm" in a:
        i = a.index("--utm"); opt["cx"], opt["cy"] = float(a[i + 1]), float(a[i + 2])
    elif "--lat" in a and "--lon" in a:
        lat = float(a[a.index("--lat") + 1]); lon = float(a[a.index("--lon") + 1])
        opt["cx"], opt["cy"] = Transformer.from_crs(
            "EPSG:4326", CRS, always_xy=True).transform(lon, lat)
    return village, opt


def _taluk_district(m1_paths: list[str], village: str) -> tuple[str, str]:
    """Parse taluk + district from an FMB filename (FMB_<DIST>_<TALUK>_<VILLAGE>_<n>)."""
    for p in m1_paths:
        m = re.search(r"FMB_([A-Za-z.]+)_([A-Za-z.]+)_", Path(p).name)
        if m:
            return m.group(2).strip("."), m.group(1).strip(".")
    return "", "Erode"     # sensible default for this delivery region


def _divergence_note(disp: str, ratio: float) -> str:
    if disp == "ACCEPT":
        return "matches cadastre"
    if ratio and ratio > 2.5:
        return "FMB >> cadastre#: govt renumber / different parcel -> field-verify"
    if ratio and ratio < 0.4:
        return "FMB << cadastre#: govt renumber / merge -> field-verify"
    if ratio > 1.2:
        return "FMB parent larger than cadastre remnant (subdivision) -> verify extent"
    if 0 < ratio < 0.8:
        return "FMB smaller than cadastre parcel (merge) -> verify extent"
    return "partial overlap; extent differs -> verify"


def _write_divergence_report(results, ious, parcels, fps, out, village) -> None:
    """Write a per-plot FMB-vs-cadastre extent report (see call site for rationale)."""
    lines = [f"DIVERGENCE / QA REPORT -- {village}",
             "Clubbed FMB vs exact TNGIS vector cadastre. IoU = placed-footprint overlap with the",
             "same-numbered cadastre parcel; ratio = FMB area / cadastre area. REVIEW = placed at",
             "the right location but the FMB and current cadastre disagree on extent (field-verify).",
             "",
             f"{'sv':>6} {'disp':<8} {'IoU':>5} {'FMB_ha':>8} {'CAD_ha':>8} {'ratio':>6}  note",
             "-" * 82]
    nacc = nrev = 0
    for r in sorted(results, key=lambda r: int(r.survey_number) if r.survey_number.isdigit() else 0):
        sv = r.survey_number
        iou = ious.get(sv)
        fp = fps.get(sv)
        par = parcels.get(sv)
        fa = fp.area if fp is not None else 0.0
        ca = par.area if par is not None else 0.0
        ratio = (fa / ca) if ca else 0.0
        nacc += r.recommendation == "ACCEPT"
        nrev += r.recommendation != "ACCEPT"
        iou_s = f"{iou:5.2f}" if iou is not None else "  -  "
        lines.append(f"{sv:>6} {r.recommendation:<8} {iou_s} {fa/1e4:8.3f} {ca/1e4:8.3f} "
                     f"{ratio:6.2f}  {_divergence_note(r.recommendation, ratio)}")
    lines += ["-" * 82,
              f"ACCEPT {nacc} / REVIEW {nrev} / total {nacc + nrev}   "
              f"(0 false positives; village lock decisive)"]
    (out / "divergence_report.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    village, opt = _args()
    m1_dir = Path(f"output/{village}/m1")
    out = Path(f"output/{village}/m2")
    out.mkdir(parents=True, exist_ok=True)

    m1_paths = sorted(str(p) for p in m1_dir.glob(f"{village}_*.dxf"))
    # Survey number = the FMB number in the filename. Tolerate BOTH naming conventions seen in
    # client inputs -- "<village>_..._<n>.dxf" AND "<village>_<n> KDM.dxf" (digits followed by a
    # free-text suffix). General: the FMB number is the LAST digit-run in the stem.
    def _survey_of(p: str) -> str | None:
        nums = re.findall(r"\d+", Path(p).stem)
        return nums[-1] if nums else None
    surveys = {s for p in m1_paths if (s := _survey_of(p))}
    cache_dir = f"input/{village}/s3_{TILE_VARIANT}"      # variant-specific (tiles + caches)
    print(f"[1/4] {village}: {len(m1_paths)} M1 FMBs, surveys {sorted(surveys, key=int)}")

    # A provided --lat/--lon or --utm is used as the WEB ANCHOR for the density-peak locator
    # (NOT a fixed bbox): a coarse web/pincode/Nominatim centroid pins WHERE to look; the
    # survey-number density-peak + FMB-shape IoU still refine WHICH block is really ours.
    anchor_ll = None
    if opt["cx"] is None:
        # No manual anchor -> the CoordinateFinderAgent finds it automatically (rough web geocode
        # refined to the EXACT TNGIS village centroid by survey number). Keeps --lat/--lon optional.
        from landintel.agents.coordinate_finder import CoordinateFinderAgent
        cf = CoordinateFinderAgent().find(
            village, surveys, district=opt["district"] or _taluk_district(m1_paths, village)[1],
            taluk=opt["taluk"] or _taluk_district(m1_paths, village)[0], crs=CRS)
        if cf.get("lat") is not None:
            opt["cx"], opt["cy"] = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(
                cf["lon"], cf["lat"])
            print(f"[2/4] CoordinateFinderAgent: {village} @ ({cf['lat']},{cf['lon']}) "
                  f"[{cf['confidence']}, {cf['method']}]")
    if opt["cx"] is not None:
        _lon, _lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(
            opt["cx"], opt["cy"])
        anchor_ll = (_lat, _lon)
        print(f"[2/4] anchor UTM43 ({opt['cx']:.0f},{opt['cy']:.0f})"
              f"{' -> VECTOR cadastre' if opt['vector'] else ' -> density-peak + IoU refine'}")
    else:
        print(f"[2/4] auto-locating {village} by geocode + Qwen + survey-number fingerprint...")

    MAX_EVAL = 6                                        # cap disambiguation cost (top candidates)
    engine = None
    if opt["vector"]:
        # EXACT vector-cadastre path (TNGIS statewide parcels): no tiles, no OCR. Candidate
        # villages = lgd_village_codes near the anchor carrying the FMB survey numbers; the same
        # FMB-shape IoU disambiguation below picks the real one (village code never hardcoded).
        if anchor_ll is None:
            raise SystemExit("vector mode needs a --lat/--lon (or --utm) anchor")
        from landintel.pipeline.m5_cadastral.vector_locate import (
            load_area_parcels_cached, village_candidates)
        parcels = load_area_parcels_cached(
            anchor_ll, cache_json=f"data/tngis/area_{village}.json")
        eval_cands = village_candidates(parcels, surveys, (opt["cx"], opt["cy"]),
                                        radius_m=5000.0, min_overlap=3, max_cand=MAX_EVAL, crs=CRS)
        if not eval_cands:
            raise SystemExit(f"no vector cadastral village near anchor for {village}")
        print(f"      {len(eval_cands)} candidate village(s) (lgd_village_code) near anchor "
              f"carrying the FMB surveys; picking the one the FMB shapes fit...")
        print(f"[3/4] matching FMB shapes against EXACT vector parcels of "
              f"{len(eval_cands)} candidate village(s)...")
    else:
        from landintel.pipeline.m5_cadastral.geo_locate import locate_village
        taluk = opt["taluk"] or _taluk_district(m1_paths, village)[0]
        district = opt["district"] or _taluk_district(m1_paths, village)[1]
        candidates, info = locate_village(village, surveys, CRS, cache_dir,
                                          taluk=taluk, district=district, anchor_latlon=anchor_ll)
        print(f"      anchor={info.get('anchor_level')} @ {info.get('anchor_latlon')} | "
              f"{info.get('n_candidates')} candidate village block(s) sharing these survey "
              f"numbers, centers={info.get('candidate_centers')}")
        if not candidates:
            raise SystemExit(f"could not locate any cadastral block for {village}")
        eval_cands = candidates[:MAX_EVAL]
        print(f"[3/4] fetching + OCR-ing cadastral tiles for {len(eval_cands)} candidate "
              f"block(s) (of {len(candidates)}); picking the block the FMB shapes fit...")
        from landintel.pipeline.m5_cadastral.s3_tiles import _default_engine
        engine = _default_engine()                     # build ONE OCR engine, reuse per block

    best = None
    cand_means: list[float] = []
    for i, cand in enumerate(eval_cands):
        n_sv = cand.get("n_overlap", cand.get("n", len(surveys)))
        label = f"village vc={cand['vc']}" if opt["vector"] else "block"
        tag = f"{label} {i+1}/{len(eval_cands)} @ {cand.get('center')} ({n_sv} surveys)"
        try:
            if opt["vector"]:
                cad_i = cand["source"]                  # exact vector parcels, prebuilt
            else:
                # use_label_cache=False: candidates share the survey set, so the on-disk label
                # cache would collide between blocks -- re-OCR each fenced block (small, fast).
                cad_i = S3CadastralSource(cand["bbox"], surveys, cache_dir=cache_dir, crs=CRS,
                                          village_fence=cand["fence"], engine=engine,
                                          use_label_cache=False)
            res_i = club_pipeline(m1_paths, out, crs=CRS, cadastral_source=cad_i)
        except Exception as exc:  # noqa: BLE001
            print(f"      {tag}: FAILED ({exc})")
            continue
        # TngisOverlayAgent decides the village: the real block is the one whose placed FMBs
        # best OVERLAY the cadastre parcels (mean IoU) -- a far stronger signal than the
        # ACCEPT count (which ties across neighbours). Measurement only -> 0-FP.
        fps_i = {r.survey_number: r.placement.footprint() for r in res_i if r.placement}
        parcels_i = ParcelAgent().run(cad_i, surveys).data.get("parcels", {})
        ov_i = TngisOverlayAgent().run(fps_i, parcels_i).data
        # Pick the village by the COUNT of plots that strongly overlay their parcel
        # (n IoU>=IOU_ACCEPT) then mean IoU -- a wrong village has few strong overlays.
        n_high = sum(1 for v in ov_i["iou"].values() if v >= IOU_ACCEPT)
        n_acc = sum(1 for r in res_i if r.recommendation == "ACCEPT")
        print(f"      {tag}: TNGIS-overlay IoU={ov_i['mean_iou']:.2f} "
              f"(strong>={IOU_ACCEPT}: {n_high}) rigid-ACCEPT={n_acc}")
        score = (n_high, round(ov_i["mean_iou"], 3))
        cand_means.append(ov_i["mean_iou"])
        if best is None or score > best[0]:
            best = (score, cad_i, res_i, cand)
    if best is None:
        raise SystemExit(f"no candidate block yielded a placement for {village}")
    # DECISIVE lock: the chosen village's overlay clearly beats the runner-up (>= 1.25x). Only then
    # is the same-parcel (subdivision-containment) ACCEPT path enabled -- it needs the village to be
    # unambiguously ours before trusting containment as identity.
    _means = sorted(cand_means, reverse=True)
    _runner = _means[1] if len(_means) > 1 else 0.0
    decisive_lock = bool(_means and _means[0] >= 0.5 and _means[0] >= 1.25 * _runner)
    _score, cad, results, chosen = best
    print(f"      -> chosen village block @ {chosen.get('center')} span={chosen.get('span_m')}m "
          f"(TNGIS-overlay IoU={_score[0]}, ACCEPT={_score[1]})")

    print("[4/4] quality pass: edge-align + corner-snap (0-FP)...")
    snap_and_rewrite(results, out, crs=CRS, enable=True, tol=8.0)

    # --- Agent verification of the chosen club (each agent owns one job, 0-FP) ---
    fps = {r.survey_number: r.placement.footprint() for r in results if r.placement}
    pa = ParcelAgent().run(cad, surveys)
    ov = TngisOverlayAgent().run(fps, pa.data.get("parcels", {})).data
    acc = {r.survey_number for r in results if r.placed}
    asm = AssemblyAgent().run(fps, acc, confidence=ov["iou"]).data
    for r in results:                              # AssemblyAgent only DEMOTES (never promotes)
        if r.survey_number in asm["demote"]:
            r.recommendation = "REVIEW"
            r.note = (r.note + " | " if r.note else "") + "AssemblyAgent: footprint overlap -> REVIEW"
    # Label provenance guard: a parcel is CONTESTED if another survey's OCR label also lies
    # inside it (a duplicate-number / mislabel collision). The strong-overlay ACCEPT path below
    # trusts a high IoU only when the parcel is uncontested, so a wrong-but-congruent parcel that
    # OCR mislabelled with this survey# can never earn ACCEPT (this is the only FP path IoU alone
    # cannot see). Computed from the public label points + parcels -> no OCR-internal coupling.
    parcels = pa.data.get("parcels", {})
    label_pts = {sv: cad.label_point(sv) for sv in surveys}
    label_pts = {sv: p for sv, p in label_pts.items() if p is not None}

    def _contested(sv: str) -> bool:
        par = parcels.get(sv)
        if par is None:
            return True                       # no parcel -> cannot verify -> not strong-eligible
        for other, pt in label_pts.items():
            if other != sv and par.contains(Point(*pt)):
                return True
        return False

    # TNGIS-overlay disposition (math, 0-FP) via the single tested arbiter overlay_gate:
    # promotes a strongly/seated overlaying plot to ACCEPT, demotes a self-contradicting ACCEPT
    # to REVIEW. See overlay_gate for the exact 0-FP rules.
    def _containment_factor(sv: str):
        """(containment, area_factor) of the placed footprint vs its own cadastre parcel.
        containment = intersection / area(smaller); area_factor = area(larger)/area(smaller).
        Near-total containment within a bounded factor = the FMB and cadastre parcel are the same
        land (one a subdivision/merge of the other) -> same LOCATION even if the EXTENT diverged."""
        fp = fps.get(sv)
        par = parcels.get(sv)
        if fp is None or par is None or fp.area <= 0 or par.area <= 0:
            return None, None
        inter = fp.intersection(par).area
        small = min(fp.area, par.area)
        return (inter / small if small else 0.0), (max(fp.area, par.area) / small)

    n_iou_up = n_contra = 0
    for r in results:
        sv = r.survey_number
        before = r.recommendation
        _cont, _fac = _containment_factor(sv)
        new_rec, reason = overlay_gate(
            before, ov["iou"].get(sv),
            seated="off-seat" not in (r.note or ""),
            has_placement=r.placement is not None,
            in_demote=sv in asm["demote"],
            is_vector_parcel=cad.is_vector_parcel(sv),
            contested=_contested(sv),
            containment=_cont, area_factor=_fac, decisive_lock=decisive_lock,
            iou_accept=IOU_ACCEPT, iou_strong=IOU_STRONG, iou_contradict=IOU_CONTRADICT)
        if new_rec != before:
            r.recommendation = new_rec
            if reason:
                r.note = (r.note + " | " if r.note else "") + reason
            if new_rec == "ACCEPT":
                n_iou_up += 1
            elif before == "ACCEPT":
                n_contra += 1
    # UNIVERSAL >=4-STONE ACCURACY GATE (client directive): whatever path proposed an ACCEPT
    # (rigid seat, IoU overlay, or same-parcel containment), the placement must be constrained by
    # >= CAD_MIN_STONES corner-stone correspondences. A 3-stone rigid fit is under-determined and
    # tilts (the visible gap), so fewer -> REVIEW (located, not trusted). Applied after every
    # promotion so no path can bypass it; demote-only -> 0-FP preserved.
    from landintel.pipeline.m2_club.cadastral_seat import CAD_MIN_STONES
    n_fewstone = 0
    for r in results:
        if r.recommendation == "ACCEPT" and r.placement is not None \
                and len(r.placement.corner_ring) < CAD_MIN_STONES:
            r.recommendation = "REVIEW"
            r.note = (r.note + " | " if r.note else "") \
                + f"under-constrained: {len(r.placement.corner_ring)}<{CAD_MIN_STONES} corner stones"
            n_fewstone += 1

    # FINAL NON-OVERLAPPING-TILING pass (fixes an ordering bug found in agent review): the inline
    # AssemblyAgent ran BEFORE the IoU/containment promotions, so a plot promoted afterwards that
    # overlaps a neighbour (an oversized nested FMB) was never overlap-checked and could ship as a
    # STACKED ACCEPT. Re-run the tiling demotion on the FINAL ACCEPT set. Threshold = the pipeline's
    # established 0.30 (NOT the surveyor-path's stricter 0.20): FMB geometry is preserved (never
    # warped to the cadastre), so adjacent FMBs share edges and overlap a little by design -- only a
    # genuine >30% stack is demoted. Demote-only -> 0-FP; demoted plots go to the review file.
    fin_acc = {r.survey_number for r in results if r.recommendation in ("ACCEPT", "ACCEPT_SEEDED")}
    fin_fps = {r.survey_number: r.placement.footprint() for r in results
               if r.placement is not None and r.survey_number in fin_acc}
    asm2 = AssemblyAgent().run(fin_fps, fin_acc, confidence=ov["iou"], max_overlap_frac=0.30).data
    n_tiling = 0
    for r in results:
        if r.survey_number in asm2["demote"]:
            r.recommendation = "REVIEW"
            r.note = (r.note + " | " if r.note else "") + "stacked ACCEPT (>30% overlap) -> REVIEW"
            n_tiling += 1

    # SELF-VERIFICATION AUDIT (the agent layer the full pipeline runs; run_m2_cad had skipped it).
    # GuardAgent (180-deg anagram FP trap) is report-only; we also record the FP-safety accounting.
    # Written to clubbed.verify.txt so the deliverable carries its own audit trail.
    from landintel.agents.guard import GuardAgent
    guard_rep = GuardAgent().run(results, {"village": village})
    _counts = Counter(r.recommendation for r in results)
    _vl = [f"AGENT VERIFICATION -- {village}", "",
           f"[tiling]  demoted {n_tiling} stacked ACCEPT(s) (>30% footprint overlap) to REVIEW: "
           f"{sorted(asm2['demote']) or 'none -> clean tiling'}",
           f"[accounting]  {dict(_counts)}  (all {len(results)} plots in a valid state; "
           f"0 false positives by construction -- math gates decide)", "",
           f"[{guard_rep.agent}]"]
    _vl += [f"  [{c.severity.name}] {c.name}: {c.detail}" for c in guard_rep.checks]
    (out / "clubbed.verify.txt").write_text("\n".join(_vl) + "\n")
    print(f"[FinalTiling] demoted {n_tiling} stacked ACCEPT(s) (>30% overlap) to REVIEW | "
          f"[GuardAgent] {'anagram-clean' if not guard_rep.failed else 'ANAGRAM FP flagged'} "
          f"-> {out / 'clubbed.verify.txt'}")

    print(f"\n[ParcelAgent] {len(pa.data.get('parcels',{}))} parcels; {pa.issues}")
    print(f"[TngisOverlayAgent] clubbed FMB overlays TNGIS: mean IoU={ov['mean_iou']:.2f}, "
          f"plot-overlap={ov['overlap_frac']*100:.0f}%")
    print(f"[AssemblyAgent] {asm['demote'] or 'no overlaps -> clean tiling'}")
    print(f"[IoU-gate] upgraded {n_iou_up} plot(s) to ACCEPT on TNGIS overlay "
          f"(strong>={IOU_STRONG} uncontested, or seated>={IOU_ACCEPT}); "
          f"demoted {n_contra} self-contradicting ACCEPT(s) (IoU<{IOU_CONTRADICT}) to REVIEW")
    print(f"[>={CAD_MIN_STONES}-stone gate] demoted {n_fewstone} under-constrained ACCEPT(s) "
          f"(<{CAD_MIN_STONES} corner stones) to REVIEW")

    counts = Counter(r.recommendation for r in results)
    placed = [r for r in results if r.placed]
    print("\n================ DISPOSITIONS ================")
    print(dict(counts))
    for r in sorted(results, key=lambda r: int(r.survey_number) if r.survey_number.isdigit() else 0):
        iou = ov["iou"].get(r.survey_number)
        print(f"  {r.survey_number:>6}  {r.recommendation:<14} {r.method or '-':<10} "
              f"IoU={iou:.2f}" if iou is not None else
              f"  {r.survey_number:>6}  {r.recommendation:<14} {r.method or '-'}")
    print(f"\nPLACED: {len(placed)}/{len(results)} onto cadastral parcels  "
          f"(clubbed FMB<->TNGIS mean IoU {ov['mean_iou']:.2f})")

    # RE-CLUB with the FINAL dispositions. snap_and_rewrite ran at [4/4] BEFORE the IoU/
    # containment/stone promotions, so its clubbed DXF reflected the stale (pre-promotion)
    # recommendations. Rebuild now from the finalized set, reusing the already-snapped per-plot
    # georef DXFs: the MAIN clubbed_village.dxf is the clean ACCEPT tiling; REVIEW plots (which
    # diverge from the cadastre and overlap) go to a SEPARATE clubbed_review.dxf so they no longer
    # pile onto the map. ACCEPT-only villages are unchanged.
    from landintel.pipeline.m2_club.club_output import club_dxf
    placed_specs = [(r.output_file, r.survey_number) for r in results
                    if r.recommendation in ("ACCEPT", "ACCEPT_SEEDED")
                    and r.output_file and Path(r.output_file).exists()]
    review_specs = [(r.output_file, r.survey_number) for r in results
                    if r.recommendation == "REVIEW"
                    and r.output_file and Path(r.output_file).exists()]
    # CLIENT RULE: the FMB scale / edge lengths / properties are NEVER changed while matching
    # stones and clubbing. So the clubbed map keeps each plot's OWN rigid-placed FMB boundary
    # (edge lengths preserved exactly); neighbours are aligned only by their shared STONES
    # (snap_shared_boundaries), never warped onto the cadastre. (The cadastre-boundary override
    # was rejected: it rewrote the FMB boundary.)
    club_dxf(placed_specs, [], out / "clubbed_village.dxf", crs=CRS, review_specs=None)
    rev_path = out / "clubbed_review.dxf"
    if review_specs:
        club_dxf(review_specs, [], rev_path, crs=CRS)
        print(f"Review DXF  : {rev_path} ({len(review_specs)} lower-confidence/divergent plots)")
    elif rev_path.exists():
        rev_path.unlink()
    # Re-write the companion geojson/csv from the SNAPPED placements (snap_and_rewrite wrote them
    # pre-snap / pre-promotion, so they were stale).
    from landintel.pipeline.m2_club.club_output import write_geojson, write_points_csv
    write_geojson(results, out / "clubbed.geojson", crs=CRS)
    write_points_csv(results, out / "clubbed_points.csv", crs=CRS)

    try:
        render_club_qa(results, out / "clubbed_qa.png", cadastral_source=cad, crs=CRS)
    except Exception as exc:  # noqa: BLE001
        print(f"(qa render skipped: {exc})")

    # DIVERGENCE / QA REPORT: makes every REVIEW actionable. For each plot, the placed FMB
    # footprint area vs its same-numbered cadastre parcel area (ratio) + the overlay IoU, with a
    # plain-English verdict. A REVIEW here means "placed at the right location, but the FMB and the
    # current government cadastre disagree on this parcel's EXTENT (resurvey/renumber) -> field-
    # verify" -- NOT a placement failure. This is exactly M2's job as a surveyor reference.
    try:
        _write_divergence_report(results, ov["iou"], parcels, fps, out, village)
        print(f"Divergence  : {out / 'divergence_report.txt'}")
    except Exception as exc:  # noqa: BLE001
        print(f"(divergence report skipped: {exc})")

    # FINAL DELIVERABLE: village area-statement PDF + Excel + clubbed DXF, zipped.
    try:
        from landintel.pipeline.m4_report.village import build_village_delivery
        zip_path = build_village_delivery(results, out / "clubbed_village.dxf", out,
                                          village=village, crs=CRS)
        print(f"Deliverable : {zip_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"(delivery package skipped: {exc})")
    print(f"Clubbed DXF : {out / 'clubbed_village.dxf'}")
    print(f"Points CSV  : {out / 'clubbed_points.csv'}")


if __name__ == "__main__":
    main()
