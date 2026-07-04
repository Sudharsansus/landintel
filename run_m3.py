"""M3 -- georeference M1 FMBs against the SURVEYOR raw-data stones (survey-grade).

PARALLEL to M2, never downstream of it. Inputs: M1 FMB DXFs + surveyor DXF (+ Google
for the village LOCATION only). NO M2 output, NO cadastre matching -- so an M2 error
can never cascade into M3, and M3 can place the very plots M2 gets wrong.

Two phases, both cadastre/M2-free:
  1. ANCHOR  -- each M1 plot's corner shape is matched geometrically to the cropped
     surveyor stones; plots that match strongly + uniquely self-locate (a distinctive
     corner N-gon needs no seed).
  2. GROW    -- unplaced plots are matched against surveyor stones NEAR the already-
     placed plots (seed = M3's OWN anchored plots, not M2), and kept only if the new
     placement tiles (does not overlap a placed plot). Iterates outward until stable.

Final coordinates come 100% from the surveyor stones -> survey-grade.

Usage:  python run_m3.py <VILLAGE> --surveyor "<RAW DATA.dxf>" [--district Erode --taluk ..]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")


def _load_dotenv(path: str = ".env") -> None:
    try:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


_load_dotenv()

import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon

from landintel.pipeline.m2_georef.extract_m1 import extract_m1_dxf
from landintel.pipeline.m2_georef.extract_surveyor import extract_surveyor
from landintel.pipeline.m2_georef.match import geometric_match
from landintel.pipeline.m2_georef.m3_deliverables import (
    M3Placement, M3_CORROB_TOL_M, classify, place_scale_locked,
    write_dxf, write_overlay, write_report)
from landintel.pipeline.m5_cadastral.geo_locate import _google_geocode_candidates

CRS = "EPSG:32643"
# Confident self-locating ANCHOR bar (data-keyed, general): a distinctive corner subset
# that cannot chance-match -- >= ANCHOR_MIN_INLIERS stones AND >= ANCHOR_MIN_FRAC of the
# plot's own corners at sub-ANCHOR_MAX_RESID residual.
ANCHOR_MIN_INLIERS = 6
ANCHOR_MIN_FRAC = 0.45
ANCHOR_MAX_RESID = 3.0
# GROW acceptance: a neighbour placed by locality needs a real match too, but the tiling
# (non-overlap) constraint is the extra evidence, so the inlier bar is min(4, n_corners).
GROW_MIN_INLIERS = 4
GROW_MAX_RESID = 3.5
GROW_OVERLAP_MAX = 0.30          # placed plots tile; >30% interior overlap = wrong seat
GROW_REGION_PAD = 60.0           # metres beyond a placed plot's radius to admit neighbours
# Cadastre-SEEDED crop margin: metres added beyond a plot's own radius to size the local
# surveyor-stone window around its cadastre seat. Sized to the cadastre seat's uncertainty
# (~15-20 m: raster/vector registration + the FMB-vs-field 3-5 m discrepancy), while the
# chance-congruent decoys the whole-village search hits are ~700 m away -- so this window sits
# in a ~100x gap and captures the RIGHT local stones without a per-village constant.
SEED_CROP_MARGIN = 40.0


def _cadastre_seeds(village, anchor_latlon, ax, ay, surveys, m1_paths, crs):
    """Per-survey cadastre seat = where the FMB SITS on its own parcel, a LOCATION PRIOR only
    (for cropping + a corroboration cross-check). M3 computes it ITSELF by running the shared M2
    CLUB ALGORITHM (`club_pipeline`) on the M1 DXFs + an independently-built TNGIS vector source
    -- this reuses M2's ALGORITHM, never M2's OUTPUT files, so an M2 run's mistakes can never
    cascade here (a single-parcel `cadastral_seat` fit alone mis-places large plots by >200 m;
    the full club's fit+snap+propagation is what lands them correctly). Fail-safe: returns {}
    when the parquet/anchor is unavailable, so M3 falls back to blind anchor+grow with no
    regression. The seat only CROPS + CROSS-CHECKS; the surveyor stones set every coordinate
    (rule 2, cadastre-crop-only)."""
    import shutil
    import tempfile
    try:
        parq = os.environ.get("LANDINTEL_TNGIS_PARQUET",
                              "data/tngis/TNGIS_TN_Cadastrals.parquet")
        if not Path(parq).exists():
            return {}
        from landintel.pipeline.m2_club import club_pipeline
        from landintel.pipeline.m5_cadastral.vector_locate import (
            load_area_parcels_cached, village_candidates)
        ck = f"data/tngis/area_{village}_{anchor_latlon[0]:.2f}_{anchor_latlon[1]:.2f}.json"
        parcels = load_area_parcels_cached(anchor_latlon, cache_json=ck)
        cands = village_candidates(parcels, set(surveys), (ax, ay), radius_m=5000.0,
                                   min_overlap=3, max_cand=6, crs=crs)
        if not cands:
            return {}
        # Top candidate = most FMB-survey overlap, then nearest to the anchor. No IoU pick is
        # needed here (unlike M2): the surveyor stones + corroboration are the real gate, so a
        # wrong-village seed only yields empty crops / no corroboration -> falls back, 0-FP.
        src = cands[0]["source"]
        tmp = Path(tempfile.mkdtemp(prefix="m3seed_"))         # discarded; M2 output untouched
        try:
            results = club_pipeline(m1_paths, tmp, crs=crs, cadastral_source=src, village=village)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        seeds = {}
        for r in results:
            fp = r.placement.footprint() if r.placement is not None else None
            if fp is None or fp.is_empty:
                continue
            c = fp.centroid
            seeds[str(r.survey_number)] = (float(c.x), float(c.y))
        return seeds
    except Exception as exc:  # noqa: BLE001 - seeding is best-effort; never break M3
        print(f"[seed] cadastre seeding unavailable ({exc}); blind anchor+grow only")
        return {}


def _place(m1, surveyor, result):
    """SCALE-LOCKED rigid placement (rule 2) from a MatchResult's stone_map.

    Builds the matched (M1 corner -> surveyor stone) pairs and fits ROTATION + TRANSLATION
    only (scale == 1, via rigid_procrustes) -- FMB edge lengths are preserved exactly. The
    similarity scale umeyama WOULD have used is kept as a DIAGNOSTIC (``s_fitted``) and is
    never applied; a value far from 1 flags an upstream M1 unit bug. Returns a placement
    dict or None (fewer than 2 pairs, or the diagnostic scale is insane -> not a real match).
    """
    pos = m1.stone_positions()
    surv = surveyor.stone_positions
    src, dst = [], []
    for i, j in enumerate(result.stone_map):
        if j is not None and j >= 0:
            src.append(pos[i]); dst.append(surv[j])
    if len(src) < 2:
        return None
    R, t, s_fitted, residuals = place_scale_locked(np.array(src, float), np.array(dst, float))
    if not (0.5 < s_fitted < 2.0):        # diagnostic sanity: a true rigid match has s ~ 1
        return None
    ring = (pos @ R.T + t)[np.array(m1.outer_stone_indices)]   # scale-1 (rigid) placement
    return {"R": R, "t": t, "s_fitted": s_fitted, "ring": ring,
            "residuals": residuals, "n_matched": len(src)}


def _fp(ring):
    if ring is None or len(ring) < 3:
        return None
    p = Polygon([(float(x), float(y)) for x, y in ring])
    if not p.is_valid:
        p = p.buffer(0)
    return p if (not p.is_empty and p.area > 0) else None


def _overlaps(fp, placed):
    for pf in placed.values():
        if pf is None or not fp.intersects(pf):
            continue
        if fp.intersection(pf).area / max(min(fp.area, pf.area), 1e-9) > GROW_OVERLAP_MAX:
            return True
    return False


def main() -> None:
    a = sys.argv[1:]
    village = a[0]
    surveyor = a[a.index("--surveyor") + 1] if "--surveyor" in a else None
    district = a[a.index("--district") + 1] if "--district" in a else "Erode"
    taluk = a[a.index("--taluk") + 1] if "--taluk" in a else ""
    if not surveyor:
        raise SystemExit("need --surveyor <RAW DATA.dxf>")

    g = _google_geocode_candidates(f"{village}, {taluk}, {district}, Tamil Nadu, India", 1)
    if not g:
        raise SystemExit(f"Google geocode failed for {village}")
    ax, ay = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(g[0][1], g[0][0])
    print(f"[1/4] Google anchor {village} @ ({g[0][0]:.5f},{g[0][1]:.5f})")

    m1s = {}
    m1_paths = sorted(str(p) for p in Path(f"output/{village}/m1").glob("*.dxf"))
    for p in m1_paths:
        m1 = extract_m1_dxf(p)
        if len(m1.outer_stone_indices) >= 3:
            m1s[str(m1.survey_number)] = m1

    # --- Cadastre seeds FIRST (surveyor-independent): per-survey parcel positions from the
    # shared M2 club algorithm on an independently-built TNGIS source (never M2's output). ---
    seeds = _cadastre_seeds(village, (g[0][0], g[0][1]), ax, ay, set(m1s), m1_paths, CRS)

    # Surveyor window: PIN it on the seeds' own extent when available (+ pad for plot radius),
    # not the coarse village-name geocode. The village-name point can be ~2 km off the actual
    # parcels (measured: NASIYANUR's Google pin is 1.8 km from its cadastre block, outside a
    # 1500 m anchor window) -- the cadastre gives the true per-survey positions, so a seed-pinned
    # window reliably covers the real stones. Falls back to the anchor window with no seeds.
    if seeds:
        sx = [p[0] for p in seeds.values()]; sy = [p[1] for p in seeds.values()]
        pad = 700.0
        bbox = (min(sx) - pad, min(sy) - pad, max(sx) + pad, max(sy) + pad)
    else:
        pad = 1500.0
        bbox = (ax - pad, ay - pad, ax + pad, ay + pad)
    sd = extract_surveyor(surveyor, bbox=bbox); sd.build_index()
    spos = sd.stone_positions
    print(f"[2/4] {len(sd.stones)} surveyor stones in the {village} window "
          f"({'seed-pinned on the cadastre block' if seeds else 'anchor-pinned'})")

    placed, placed_fp = {}, {}

    # --- Phase 0: cadastre-SEEDED crop (cadastre CROPS, stones SCORE -- rule-2 clean) ---
    # For each plot, crop the surveyor stones to its own cadastre-seat neighbourhood and match
    # the FMB ring against ONLY those local stones -- de-saturating the ~500-stone window so
    # chance-congruent seats 700 m away cannot compete. A placement is kept only if it
    # INDEPENDENTLY agrees with the seat (<= M3_CORROB_TOL_M): two independent sources (field
    # stones + government cadastre) confirming each other. Stones set the coordinates; the
    # cadastre only crops/cross-checks and never warps anything (rule 2).
    n_seeded = 0
    if seeds:
        for sv, m1 in m1s.items():
            seat = seeds.get(sv)
            if seat is None:
                continue
            ring0 = m1.stone_positions()[np.array(m1.outer_stone_indices)]
            prad = float(np.hypot(np.ptp(ring0[:, 0]), np.ptp(ring0[:, 1]))) * 0.6
            d = np.hypot(spos[:, 0] - seat[0], spos[:, 1] - seat[1])
            allowed = d <= (prad + SEED_CROP_MARGIN)
            if allowed.sum() < 4:
                continue                             # surveyor set no stones here -> not seeded
            gm = geometric_match(m1, sd, allowed_stones=allowed)
            if not gm.matched:
                continue
            pl = _place(m1, sd, gm)
            if pl is None:
                continue
            pc = pl["ring"].mean(axis=0)
            corrob = float(np.hypot(pc[0] - seat[0], pc[1] - seat[1]))
            pl["corrob"] = corrob
            if corrob > M3_CORROB_TOL_M:              # placement disagrees w/ seat -> not seeded
                continue
            fp = _fp(pl["ring"])
            if fp is None or _overlaps(fp, placed_fp):
                continue
            placed[sv], placed_fp[sv] = pl, fp
            n_seeded += 1
        print(f"[2b/4] cadastre-seeded phase: {n_seeded}/{len(m1s)} plot(s) placed AND "
              f"corroborated by the cadastre (2 independent sources)")
    else:
        print("[2b/4] no cadastre seeds available -> blind anchor+grow only")

    # --- Phase 1: anchor (no seed) ---
    for sv, m1 in m1s.items():
        gm = geometric_match(m1, sd)
        n = len(m1.outer_stone_indices)
        if (gm.matched and gm.n_matched_stones >= ANCHOR_MIN_INLIERS
                and gm.n_matched_stones / n >= ANCHOR_MIN_FRAC
                and gm.fingerprint_score <= ANCHOR_MAX_RESID):
            pl = _place(m1, sd, gm)
            if pl is None:
                continue
            fp = _fp(pl["ring"])
            if fp is None or _overlaps(fp, placed_fp):
                continue
            placed[sv], placed_fp[sv] = pl, fp
    print(f"[3/4] anchor phase: {len(placed)}/{len(m1s)} plots self-located")

    # --- Phase 2: grow from placed plots (seed = OUR anchored plots, not M2) ---
    # Each unplaced plot is seated against EACH placed neighbour INDIVIDUALLY: crop the
    # surveyor stones tightly around that one neighbour (neighbour radius + plot radius),
    # which keeps the candidate set small so a distinctive plot cannot chance-match. Accept
    # the first neighbour that yields a strong, non-overlapping (tiling) placement. The seed
    # is purely M3's own placed plots -- no M2, no cadastre.
    for _round in range(12):
        changed = False
        for sv, m1 in m1s.items():
            if sv in placed:
                continue
            ring = m1.stone_positions()[np.array(m1.outer_stone_indices)]
            prad = float(np.hypot(np.ptp(ring[:, 0]), np.ptp(ring[:, 1]))) * 0.6 + GROW_REGION_PAD
            n = len(m1.outer_stone_indices)
            best = None
            for pf in placed_fp.values():
                if pf is None:
                    continue
                cx, cy = pf.centroid.x, pf.centroid.y
                nrad = np.hypot(*(np.array(pf.bounds[2:]) - np.array(pf.bounds[:2]))) * 0.6
                d = np.hypot(spos[:, 0] - cx, spos[:, 1] - cy)
                allowed = d <= (nrad + prad)
                if allowed.sum() < 4:
                    continue
                gm = geometric_match(m1, sd, allowed_stones=allowed)
                if not (gm.matched and gm.n_matched_stones >= min(GROW_MIN_INLIERS, n)
                        and gm.fingerprint_score <= GROW_MAX_RESID):
                    continue
                pl = _place(m1, sd, gm)
                if pl is None:
                    continue
                fp = _fp(pl["ring"])
                if fp is None or _overlaps(fp, placed_fp):
                    continue
                med = float(np.median(pl["residuals"])) if len(pl["residuals"]) else float("inf")
                if best is None or med < best[2]:
                    best = (pl, fp, med)
            if best is not None:
                placed[sv], placed_fp[sv] = best[0], best[1]
                changed = True
        if not changed:
            break

    # --- Honest first-class dispositions + the three deliverables (rule 2: scale-locked) ---
    placements = []
    for sv, m1 in m1s.items():
        n = len(m1.outer_stone_indices)
        pl = placed.get(sv)
        if pl is not None:
            res = pl["residuals"]
            med = float(np.median(res)) if len(res) else float("nan")
            mx = float(np.max(res)) if len(res) else float("nan")
            # Cadastre corroboration for ANY placed plot that has a seat (whether the seeded
            # phase or the blind anchor+grow placed it): distance from the placed ring centroid
            # to the independent cadastre seat. Feeds classify's 0-FP cross-check -- confirms a
            # seat-agreeing ACCEPT, demotes a survey-grade fit that lands away from the seat.
            seat = seeds.get(sv)
            corrob = pl.get("corrob")
            if corrob is None and seat is not None:
                pc = pl["ring"].mean(axis=0)
                corrob = float(np.hypot(pc[0] - seat[0], pc[1] - seat[1]))
            disp, note = classify(pl["n_matched"], n, med, mx, pl["s_fitted"],
                                  tiles=True, window_has_stones=True,
                                  cadastre_corrob_m=corrob)
            placements.append(M3Placement(
                survey_number=sv, disposition=disp, R=pl["R"], t=pl["t"],
                s_fitted=pl["s_fitted"], ring_utm=pl["ring"], n_matched=pl["n_matched"],
                n_corners=n, median_residual_m=med, max_residual_m=mx,
                cadastre_corrob_m=(float("nan") if corrob is None else corrob), note=note))
        else:
            # Not placed: stones exist in the village window but no confident fit -> NEEDS_GPS
            # (an operator GPS/seed disambiguates). run_m3 crops to the whole village window,
            # so it cannot yet prove a per-plot data gap (UNMEASURED); that needs a per-plot
            # seat and is left honestly as NEEDS_GPS rather than faked.
            disp, note = classify(0, n, float("nan"), float("nan"), float("nan"),
                                  tiles=True, window_has_stones=True)
            placements.append(M3Placement(survey_number=sv, disposition=disp,
                                          n_corners=n, note=note))

    outdir = Path(f"output/{village}/m3")
    write_dxf(placements, outdir / "clubbed_village.dxf", crs=CRS)
    write_overlay(placements, spos, outdir / "qa_overlay.png", village=village)
    write_report(placements, outdir / "m3_report.json", village=village)

    counts: dict[str, int] = {}
    for p in placements:
        counts[p.disposition] = counts.get(p.disposition, 0) + 1
    acc = [p.survey_number for p in placements if p.disposition == "ACCEPT"]
    meds = [p.median_residual_m for p in placements
            if p.disposition == "ACCEPT" and p.median_residual_m == p.median_residual_m]
    mr = float(np.median(meds)) if meds else 0.0
    print(f"[4/4] M3 complete: {counts.get('ACCEPT', 0)}/{len(placements)} ACCEPT "
          f"(survey-grade, scale-locked s=1) | median residual {mr:.2f} m")
    print(f"      dispositions: {counts}")
    print(f"      ACCEPT surveys: {sorted(acc, key=lambda s: int(s) if s.isdigit() else 0)}")
    print(f"      deliverables -> {outdir}/clubbed_village.dxf, qa_overlay.png, m3_report.json")


if __name__ == "__main__":
    main()
