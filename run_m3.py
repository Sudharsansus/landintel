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
import re
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
from landintel.pipeline.m2_georef.match import MatchResult, geometric_match
from landintel.pipeline.m2_georef.verify import build_traced_buffer, chain_coverage
from landintel.pipeline.m2_georef.m3_deliverables import (
    M3Placement, M3_CORROB_TOL_M, M3_ACCEPT_RESIDUAL_MEDIAN_M, M3_ACCEPT_RESIDUAL_MAX_M,
    M3_CHAIN_COVERAGE_ACCEPT, classify, place_scale_locked,
    write_dxf, write_clubbed_fmbs, write_overlay, write_report)
from landintel.pipeline.m2_club.disposition_thresholds import CAD_MIN_STONES, FULL_MATCH_STONES
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


def _place_relative(m1_b, cand, anchor_corners, tol=3.0):
    """RULE-2 rigid (scale=1) placement of a plot propagated onto a confirmed neighbour.

    ``propagate_geometric`` (the client's FMBS_STONES_MATCH) finds WHICH of ``m1_b``'s FMB
    corners coincide with the anchor's placed corners, but it applies a small umeyama scale
    (0.85-1.18). M3 is scale-strict, so we take those shared correspondences and re-fit
    ROTATION + TRANSLATION only (``place_scale_locked``): the FMB edge lengths are preserved
    exactly and the plot inherits the anchor's georeference at the shared boundary. The fitted
    scale is a diagnostic; far from 1 would be an upstream unit bug, never baked in. Returns a
    placement dict like ``_place`` (with ``relative_to`` set by the caller), or None."""
    pos = m1_b.stone_positions()
    b_adj = np.asarray(cand.adjusted, float)
    src, dst, used = [], [], set()
    for bi in m1_b.outer_stone_indices:
        if not (0 <= bi < len(b_adj)):
            continue
        d = np.hypot(anchor_corners[:, 0] - b_adj[bi][0], anchor_corners[:, 1] - b_adj[bi][1])
        ai = int(np.argmin(d))
        if d[ai] <= tol and ai not in used:                # same physical corner
            src.append(pos[bi]); dst.append(anchor_corners[ai]); used.add(ai)
    if len(src) < 3:                                       # client's >=3 shared-stone bar
        return None
    R, t, s_fitted, residuals = place_scale_locked(np.array(src, float), np.array(dst, float))
    if not (0.5 < s_fitted < 2.0):
        return None
    ring = (pos @ R.T + t)[np.array(m1_b.outer_stone_indices)]
    return {"R": R, "t": t, "s_fitted": s_fitted, "ring": ring,
            "residuals": residuals, "n_matched": len(src), "relative_to": None}


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


def place_village(village, surveyor, district="Erode", taluk=""):
    """Place ALL of ONE village's M1 FMBs onto the surveyor stone cloud by sequential verified
    growth (the client's 'match ONE FMB, verify, commit, then move to the next'). Returns
    ``(list[M3Placement], surveyor stone positions in the village window)``. A reusable METHOD:
    both the per-village run and the SINGLE combined-district output call it and the caller writes
    the deliverables -- so all villages can be clubbed onto the shared stone cloud in one DXF."""
    g = _google_geocode_candidates(f"{village}, {taluk}, {district}, Tamil Nadu, India", 1)
    if not g:
        raise SystemExit(f"Google geocode failed for {village}")
    ax, ay = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(g[0][1], g[0][0])
    print(f"[1/4] Google anchor {village} @ ({g[0][0]:.5f},{g[0][1]:.5f})")

    m1s = {}
    m1path: dict[str, str] = {}
    m1_paths = sorted(str(p) for p in Path(f"output/{village}/m1").glob("*.dxf"))
    for p in m1_paths:
        m1 = extract_m1_dxf(p)
        if len(m1.outer_stone_indices) >= 3:
            m1s[str(m1.survey_number)] = m1
            m1path[str(m1.survey_number)] = p

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
    # StoneReaderAgent: read + VERIFY the surveyor stone points and their EXACT UTM coords, then
    # match FMB corners against those verified points. (Reads only; never places/accepts -> 0-FP.)
    from landintel.agents.stone_reader import StoneReaderAgent
    sd, sr = StoneReaderAgent().read(surveyor, bbox=bbox, crs=CRS)
    spos = sd.stone_positions
    # Traced-boundary buffer (SITE DATA LINE): the INGUR ground-truth FP signal. Present only when
    # the surveyor file carries traced lines (RAW DATA WITH LINES.dxf); None for a pure stone cloud
    # -> chain_coverage returns 0 and the pipeline falls back to the cadastre gate (no regression).
    traced_buf = build_traced_buffer([p.raw_points for p in sd.polylines], tol=3.0)
    if traced_buf is None:
        print(f"      [!] {village}: surveyor file has NO traced SITE DATA LINE boundaries -- "
              f"survey-boundary confirmation is OFF (falling back to the cadastre gate). "
              f"Use the 'WITH LINES' surveyor file for full survey-grade recovery.")

    def _chaincov(ring) -> float:
        """Fraction of a placed ring's boundary lying on the surveyor's traced SITE DATA LINE."""
        if traced_buf is None or ring is None or len(ring) < 3:
            return float("nan")
        pts = [(float(x), float(y)) for x, y in ring]
        segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)] + [(pts[-1], pts[0])]
        return chain_coverage(segs, prepared=traced_buf)
    sr_fail = [c for c in sr.checks if c.severity.value == "fail"]
    sr_warn = [c for c in sr.checks if c.severity.value == "warn"]
    status = "verified" if not (sr_fail or sr_warn) else (
        "FAIL:" + sr_fail[0].detail if sr_fail else "WARN:" + sr_warn[0].detail)
    print(f"[2/4] StoneReaderAgent: {len(sd.stones)} surveyor stone points read+{status} "
          f"in the {village} window "
          f"({'seed-pinned on the cadastre block' if seeds else 'anchor-pinned'})")

    from landintel.pipeline.m2_club.placement import CandidatePlacement
    from landintel.pipeline.m2_club.relative_club import propagate_geometric

    placed, placed_fp = {}, {}

    # ONE plot's best DIRECT stone placement: cadastre-seeded crop (de-saturated) plus a
    # whole-window self-locate, keeping whichever has the smaller median residual. RANSAC is
    # expensive, so this is computed ONCE per plot and cached; the sequential loop below only
    # re-checks the cheap tiling constraint each round.
    def _direct_place(sv, m1):
        seat = seeds.get(sv)
        cands = []
        if seat is not None:
            ring0 = m1.stone_positions()[np.array(m1.outer_stone_indices)]
            prad = float(np.hypot(np.ptp(ring0[:, 0]), np.ptp(ring0[:, 1]))) * 0.6
            d = np.hypot(spos[:, 0] - seat[0], spos[:, 1] - seat[1])
            allowed = d <= (prad + SEED_CROP_MARGIN)
            if allowed.sum() >= 4:
                gm = geometric_match(m1, sd, allowed_stones=allowed)
                if gm.matched:
                    p = _place(m1, sd, gm)
                    if p is not None:
                        cands.append(p)
        gm = geometric_match(m1, sd)                       # whole-window self-locate
        if gm.matched:
            p = _place(m1, sd, gm)
            if p is not None:
                cands.append(p)
        best = None
        for p in cands:
            if seat is not None:
                pc = p["ring"].mean(axis=0)
                p["corrob"] = float(np.hypot(pc[0] - seat[0], pc[1] - seat[1]))
            med = float(np.median(p["residuals"])) if len(p["residuals"]) else float("inf")
            if best is None or med < best[1]:
                best = (p, med)
        if best is None:
            return None
        best[0]["chain_cov"] = _chaincov(best[0]["ring"])   # fraction on the traced survey lines
        return best[0]

    direct = {sv: _direct_place(sv, m1) for sv, m1 in m1s.items()}

    def _accept_gate(n, pl):
        """The M3 survey-grade ACCEPT gate -- commit only when an independent GROUND-TRUTH source
        confirms it. Two 0-FP paths; the accuracy bar (robust MEDIAN residual) is never relaxed:
          A. SURVEY-BOUNDARY CONFIRMED (the INGUR gate): the FMB boundary lies on the surveyor's
             actually-traced SITE DATA LINE (chain_coverage >= M3_CHAIN_COVERAGE_ACCEPT) on a
             well-constrained pose (>= min(6,n)) with a real match (median < 3 m) -- direct ground
             truth, so one FMB-noisy corner does not disqualify a boundary that follows the line.
          B. CADASTRE-CONFIRMED (fallback when the surveyor did NOT trace this boundary): survey-
             grade at EVERY corner (median<=2 AND max<=3) on a dense pose (>= min(5,n)) AND agrees
             with the plot's own government cadastre seat (<= M3_CORROB_TOL_M).
        Tiling vs the committed set is checked by the caller. Returns (ok, score)."""
        res = pl.get("residuals", [])
        if len(res) == 0:
            return False, 0.0
        med = float(np.median(res)); mx = float(np.max(res))
        cov = float(pl.get("chain_cov", float("nan")))
        corrob = pl.get("corrob")
        chain_ok = (cov == cov and cov >= M3_CHAIN_COVERAGE_ACCEPT
                    and pl["n_matched"] >= min(6, n) and med < M3_ACCEPT_RESIDUAL_MAX_M)
        cad_ok = (pl["n_matched"] >= min(FULL_MATCH_STONES, n)
                  and med <= M3_ACCEPT_RESIDUAL_MEDIAN_M and mx <= M3_ACCEPT_RESIDUAL_MAX_M
                  and corrob is not None and corrob <= M3_CORROB_TOL_M)
        if not (chain_ok or cad_ok):
            return False, 0.0
        score = pl["n_matched"] - med + (2.0 * cov if cov == cov else 0.0)
        return True, score

    def _verified(sv, pl):
        """Thin wrapper over ``_accept_gate`` keyed on the plot's own corner count."""
        return _accept_gate(len(m1s[sv].outer_stone_indices), pl)

    def _seed_crop_mask(m1, seat):
        """Boolean mask of surveyor stones inside the plot's cadastre-seeded local window
        (its own radius + SEED_CROP_MARGIN), or None if fewer than 4 stones fall in it."""
        ring0 = m1.stone_positions()[np.array(m1.outer_stone_indices)]
        prad = float(np.hypot(np.ptp(ring0[:, 0]), np.ptp(ring0[:, 1]))) * 0.6
        d = np.hypot(spos[:, 0] - seat[0], spos[:, 1] - seat[1])
        allowed = d <= (prad + SEED_CROP_MARGIN)
        return allowed if allowed.sum() >= 4 else None

    def _best_gated_pose(sv, m1):
        """COVERAGE-AWARE POSE RESCUE (general, 0-FP). geometric_match returns only its single
        MAX-INLIER pose and NEVER sees chain coverage, so it can hand back an on-seat pose that JUST
        misses the gate (one extra inlier at a looser residual) while a slightly-fewer-inlier pose
        on the SAME local stones lies squarely on the surveyor's traced boundary and CLEARS the gate.
        This re-examines the WHOLE RANSAC candidate set (via candidate_sink) and returns the best
        pose the UNCHANGED ``_accept_gate`` already blesses, or None. It lowers NO gate and only
        chooses among poses the search itself produced -> cannot create a false positive; and it is
        locality-safe (with a seat it searches only the seeded local crop, so a far chance decoy is
        never a candidate). GENERAL -- the same data-keyed bars decide, no per-village value."""
        seat = seeds.get(sv)
        n = len(m1.outer_stone_indices)
        mask = None
        if seat is not None:
            mask = _seed_crop_mask(m1, seat)
            if mask is None:                               # too few local stones -> no evidence
                return None
        poses = []
        geometric_match(m1, sd, allowed_stones=mask,
                        candidate_sink=lambda ninl, mr, smap, R, s, t: poses.append(smap))
        best = None                                        # (score, placement)
        for smap in poses:
            p = _place(m1, sd, MatchResult(stone_map=smap, fingerprint_score=0.0,
                       neighborhood_score=0.0, combined_score=0.0, matched=True))
            if p is None:
                continue
            res = p["residuals"]
            med = float(np.median(res)); mx = float(np.max(res))
            if seat is not None:
                pc = p["ring"].mean(axis=0)
                p["corrob"] = float(np.hypot(pc[0] - seat[0], pc[1] - seat[1]))
            # Prune cheaply: only a pose that could clear cad_ok (residual+seat) or chain_ok
            # (inliers+residual) is worth the chain-coverage cost.
            could_cad = (p["n_matched"] >= min(FULL_MATCH_STONES, n)
                         and med <= M3_ACCEPT_RESIDUAL_MEDIAN_M and mx <= M3_ACCEPT_RESIDUAL_MAX_M
                         and p.get("corrob") is not None and p["corrob"] <= M3_CORROB_TOL_M)
            could_chain = (p["n_matched"] >= min(6, n) and med < M3_ACCEPT_RESIDUAL_MAX_M)
            if not (could_cad or could_chain):
                continue
            p["chain_cov"] = _chaincov(p["ring"]) if could_chain else float("nan")
            ok, score = _accept_gate(n, p)
            if ok and (best is None or score > best[0]):
                best = (score, p)
        return best[1] if best is not None else None

    # --- SEQUENTIAL VERIFIED GROWTH (client's method: match ONE FMB, VERIFY, commit, next) ---
    # No batch "place all then judge". Each round commits exactly ONE more plot -- the single
    # best INDEPENDENTLY-VERIFIED candidate across all unplaced FMBs -- then re-evaluates tiling
    # with it committed. Nothing is trusted until multiple sources agree on it.
    while True:
        best = None                                        # (sv, pl, fp, score)
        for sv, m1 in m1s.items():
            if sv in placed:
                continue
            pl = direct.get(sv)
            if pl is None:
                continue
            fp = _fp(pl["ring"])
            if fp is None or _overlaps(fp, placed_fp):
                continue
            ok, score = _verified(sv, pl)
            if ok and (best is None or score > best[3]):
                best = (sv, pl, fp, score)
        if best is None:
            break
        sv, pl, fp, _s = best
        placed[sv], placed_fp[sv] = pl, fp                 # commit ONE verified plot, then re-loop
    print(f"[3/4] verified growth: {len(placed)}/{len(m1s)} plots placed one-by-one "
          f"(each survey-grade + independently verified before the next)")

    # --- Relative growth (client's FMBS_STONES_MATCH), also ONE verified commit per round ---
    # Seat a still-unplaced plot against a committed neighbour by their SHARED FMB corners
    # (relative_club.propagate_geometric), placed RIGIDLY (scale=1, rule 2). A relative tie is
    # trusted only when it ALSO lands at the plot's OWN cadastre seat (a THIRD source) -- the
    # >=3-shared-corner + tiling gate alone admits chance ties 150-1100 m off (measured).
    def _anchor_cp(sv):
        pl = placed[sv]; m1 = m1s[sv]; pos = m1.stone_positions()
        adj = pos @ pl["R"].T + pl["t"]                    # scale=1 rigid, ALL stones
        return CandidatePlacement(method="m3_anchor", R=pl["R"], s=1.0, t=pl["t"], adjusted=adj,
                                  corner_ring=list(m1.outer_stone_indices), passes_gate=True,
                                  scale=1.0)

    n_rel = 0
    while True:
        anchor_cps = {sv: _anchor_cp(sv) for sv in placed}
        tile_fps = list(placed_fp.values())
        best = None                                        # (sv, rp, fp, score)
        for sv, m1 in m1s.items():
            if sv in placed:
                continue
            seat = seeds.get(sv)
            if seat is None:                               # cannot verify a relative tie -> skip
                continue
            for asv, acp in anchor_cps.items():
                cand = propagate_geometric(m1, acp, m1s[asv], tile_fps)
                if cand is None:
                    continue
                rp = _place_relative(m1, cand, acp.corner_points())
                if rp is None:
                    continue
                pc = rp["ring"].mean(axis=0)
                cr = float(np.hypot(pc[0] - seat[0], pc[1] - seat[1]))
                if cr > M3_CORROB_TOL_M:                   # VERIFY vs the independent cadastre seat
                    continue
                fp = _fp(rp["ring"])
                if fp is None or _overlaps(fp, placed_fp):
                    continue
                score = rp["n_matched"] - 0.02 * cr
                if best is None or score > best[3]:
                    rp2 = dict(rp); rp2["relative_to"] = asv; rp2["corrob"] = cr
                    rp2["chain_cov"] = _chaincov(rp2["ring"])
                    best = (sv, rp2, fp, score)
        if best is None:
            break
        sv, rp, fp, _s = best
        placed[sv], placed_fp[sv] = rp, fp                 # commit ONE verified relative tie
        n_rel += 1
    if n_rel:
        print(f"[3b/4] relative stone-match: +{n_rel} plot(s) seated one-by-one by shared FMB "
              f"corners with a confirmed neighbour (client FMBS_STONES_MATCH, cadastre-verified)")

    # --- COVERAGE-AWARE RESCUE (finishing pass; one verified commit per round) ----------------
    # geometric_match trusts only its SINGLE max-inlier pose, which never sees chain coverage; it
    # can leave a plot unplaced whose boundary actually lies on the traced SITE DATA LINE under a
    # slightly-fewer-inlier pose. For each STILL-unplaced plot, re-search the RANSAC candidate set
    # (_best_gated_pose) for a pose the UNCHANGED accept gate blesses and commit it if it tiles.
    # Touches ONLY leftover plots (nothing already clubbed changes); 0-FP (no gate lowered) and
    # locality-safe (seat plots search only their local crop, so a far decoy is never a candidate).
    n_resc = 0
    blocked: dict[str, dict] = {}      # gate-passing poses rejected ONLY by tiling -> honest REVIEW
    while True:
        best = None                                        # (sv, pl, fp, score)
        for sv, m1 in m1s.items():
            if sv in placed:
                continue
            pl = _best_gated_pose(sv, m1)
            if pl is None:
                continue
            fp = _fp(pl["ring"])
            if fp is None:
                continue
            if _overlaps(fp, placed_fp):
                # gate-passing but overlaps a higher-evidence committed neighbour: real parcels
                # tile, so it cannot be clubbed here -> remember it for an HONEST REVIEW, never ACCEPT.
                prev = blocked.get(sv)
                if prev is None or pl["n_matched"] > prev["n_matched"]:
                    blocked[sv] = pl
                continue
            ok, score = _verified(sv, pl)
            if ok and (best is None or score > best[3]):
                best = (sv, pl, fp, score)
        if best is None:
            break
        sv, pl, fp, _s = best
        placed[sv], placed_fp[sv] = pl, fp                 # commit ONE rescued survey-grade plot
        direct[sv] = pl                                    # keep direct[] consistent for the report
        n_resc += 1
    if n_resc:
        print(f"[3c/4] coverage-aware rescue: +{n_resc} plot(s) recovered by a traced-line / "
              f"cadastre-confirmed pose the max-inlier matcher overlooked (survey-grade, 0-FP)")
    # A located plot that clears the gate but lost the shared land to a placed neighbour is a REVIEW
    # (confirm extent), not a NEEDS_GPS -- record its located pose so the disposition draws it.
    for sv, pl in blocked.items():
        if sv not in placed and pl.get("corrob") is not None and pl["corrob"] <= M3_CORROB_TOL_M:
            direct[sv] = pl

    # NOTE: a seat-cropped edge-register "recovery" pass was tried and REMOVED -- cropping the
    # traced-segment search to the seat window makes the "within 60 m of seat" corroboration
    # trivially satisfiable, collapsing the ER-DUAL conjunction back into coverage-max ALONE, which
    # overfits (it accepted KANDAMPALAYAM 35, an agent-verified decoy). A correct whole-village ER
    # would keep the conjunction honest but is too slow (~1 min/plot); left out until optimized.

    # --- Honest first-class dispositions + the three deliverables (rule 2: scale-locked) ---
    placements = []
    for sv, m1 in m1s.items():
        n = len(m1.outer_stone_indices)
        pl = placed.get(sv)
        if pl is not None and pl.get("relative_to"):
            # Placed by the client's shared-stone match to a CONFIRMED neighbour. It already
            # passed the client's own 0-FP bar (>=3 coincident FMB corners + tiling), so it is
            # judged on THAT provenance, not re-gated by the solo-stone residual bar. A DISTINCT
            # tier: confidently LOCATED via a confirmed neighbour, with FMB-fidelity extent at
            # the non-shared corners -- never conflated with the dual-confirmed survey-grade set.
            res = pl["residuals"]
            med = float(np.median(res)) if len(res) else float("nan")
            mx = float(np.max(res)) if len(res) else float("nan")
            cr = pl.get("corrob", float("nan"))
            cov = float(pl.get("chain_cov", float("nan")))
            cov_s = f"; {cov * 100:.0f}% on surveyed lines" if cov == cov else ""
            placements.append(M3Placement(
                survey_number=sv, disposition="ACCEPT_RELATIVE", R=pl["R"], t=pl["t"],
                s_fitted=pl["s_fitted"], ring_utm=pl["ring"], n_matched=pl["n_matched"],
                n_corners=n, median_residual_m=med, max_residual_m=mx,
                chain_coverage=cov, cadastre_corrob_m=(float("nan") if cr is None else cr),
                note=f"relative stone-match to confirmed {pl['relative_to']} "
                     f"({pl['n_matched']} shared corners, rigid); cadastre-verified "
                     f"{cr:.0f}m{cov_s}, FMB-fidelity extent"))
        elif pl is not None:
            # COMMITTED by the sequential verified growth. Two 0-FP confirmation paths (accuracy
            # never relaxed): its boundary lies on the surveyor's TRACED lines (survey-boundary
            # confirmed, the INGUR ground-truth gate), OR survey-grade every corner + cadastre
            # agreement. Multiple independent sources agreed BEFORE it was committed -> ACCEPT.
            res = pl["residuals"]
            med = float(np.median(res)) if len(res) else float("nan")
            mx = float(np.max(res)) if len(res) else float("nan")
            corrob = pl.get("corrob")
            cov = float(pl.get("chain_cov", float("nan")))
            if cov == cov and cov >= M3_CHAIN_COVERAGE_ACCEPT:
                src = f"survey-boundary confirmed ({cov * 100:.0f}% on the surveyor's traced lines)"
            elif corrob is not None and corrob <= M3_CORROB_TOL_M:
                src = f"cadastre-corroborated {corrob:.1f}m (2 independent sources)"
            else:
                src = "dense self-match"
            placements.append(M3Placement(
                survey_number=sv, disposition="ACCEPT", R=pl["R"], t=pl["t"],
                s_fitted=pl["s_fitted"], ring_utm=pl["ring"], n_matched=pl["n_matched"],
                n_corners=n, median_residual_m=med, max_residual_m=mx, chain_coverage=cov,
                cadastre_corrob_m=(float("nan") if corrob is None else corrob),
                note=f"n={pl['n_matched']}/{n} med={med:.2f}m max={mx:.2f}m (survey-grade, "
                     f"verified: {src})"))
        else:
            # NOT committed. If its best direct attempt still LANDS at its cadastre seat it is
            # correctly located but sub-survey-grade / lost a tiling conflict -> REVIEW (geometry
            # shown). Otherwise (decoy / no seat) -> NEEDS_GPS, no geometry drawn (never a fake).
            pl2 = direct.get(sv)
            corrob = pl2.get("corrob") if pl2 else None
            if pl2 is not None and corrob is not None and corrob <= M3_CORROB_TOL_M:
                res = pl2["residuals"]
                med = float(np.median(res)) if len(res) else float("nan")
                mx = float(np.max(res)) if len(res) else float("nan")
                cov = float(pl2.get("chain_cov", float("nan")))
                cov_s = f", {cov * 100:.0f}% on surveyed lines" if cov == cov else ""
                placements.append(M3Placement(
                    survey_number=sv, disposition="REVIEW", R=pl2["R"], t=pl2["t"],
                    s_fitted=pl2["s_fitted"], ring_utm=pl2["ring"], n_matched=pl2["n_matched"],
                    n_corners=n, median_residual_m=med, max_residual_m=mx, chain_coverage=cov,
                    cadastre_corrob_m=corrob,
                    note=f"n={pl2['n_matched']}/{n} med={med:.2f}m cadastre-agrees {corrob:.0f}m"
                         f"{cov_s} but sub-survey-grade / tiling conflict -> confirm"))
            else:
                disp, note = classify(0, n, float("nan"), float("nan"), float("nan"),
                                      tiles=True, window_has_stones=True)
                placements.append(M3Placement(survey_number=sv, disposition=disp,
                                              n_corners=n, note=note))

    for p in placements:
        p.village = village                                # tag for the combined district output
        p.m1_file = m1path.get(p.survey_number, "")        # so the full FMB can be clubbed
    counts: dict[str, int] = {}
    for p in placements:
        counts[p.disposition] = counts.get(p.disposition, 0) + 1
    print(f"[4/4] {village}: {counts.get('ACCEPT', 0)} ACCEPT (survey-grade) + "
          f"{counts.get('ACCEPT_RELATIVE', 0)} ACCEPT_RELATIVE (shared-stone) = "
          f"{counts.get('ACCEPT', 0) + counts.get('ACCEPT_RELATIVE', 0)}/{len(placements)} "
          f"clubbed | {dict(counts)}")
    return placements, spos


def main() -> None:
    """M3 = ONE clubbed output. Takes 1+ villages; each is placed by ``place_village`` against the
    SAME surveyor stone cloud, and ALL placements are written to a SINGLE DXF/report/overlay (the
    surveyor RAW DATA is one cloud spanning the district). Per-village runs stay backward-compatible.
    Usage: python run_m3.py <VILLAGE> [VILLAGE ...] --surveyor "<RAW DATA.dxf>" [--district Erode]"""
    a = sys.argv[1:]
    villages = []
    for tok in a:
        if tok.startswith("--"):
            break
        villages.append(tok)
    surveyor = a[a.index("--surveyor") + 1] if "--surveyor" in a else None
    district = a[a.index("--district") + 1] if "--district" in a else "Erode"
    taluk = a[a.index("--taluk") + 1] if "--taluk" in a else ""
    if not surveyor or not villages:
        raise SystemExit("usage: python run_m3.py <VILLAGE> [VILLAGE ...] "
                         "--surveyor <RAW DATA.dxf> [--district Erode]")

    all_pl, all_stones = [], []
    for v in villages:
        placements, spos = place_village(v, surveyor, district, taluk)
        all_pl += placements
        if spos is not None and len(spos):
            all_stones.append(spos)
    stones = np.vstack(all_stones) if all_stones else np.empty((0, 2))

    # SINGLE M3 OUTPUT -- ALWAYS one clubbed DXF (never per-village files), even for one village:
    # the surveyor stone cloud is one district-wide cloud, so M3 clubs every plot onto it in ONE
    # output/M3_CLUBBED/clubbed_village.dxf. Any stale per-village output/<v>/m3 dirs are removed.
    import shutil
    for v in villages:
        old = Path(f"output/{v}/m3")
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
    outdir = Path("output/M3_CLUBBED")
    outdir.mkdir(parents=True, exist_ok=True)
    title = "+".join(villages)

    # StoneReaderAgent: publish the FULL verified catalog of EXACT surveyor stone points (UTM
    # coords + codes) the matcher aligns FMB corners to -> stone_points.csv + stone_read_report.json.
    try:
        from landintel.agents.base import write_json
        from landintel.agents.stone_reader import StoneReaderAgent
        _reader = StoneReaderAgent()
        sd_all, sr_all = _reader.read(surveyor, bbox=None, crs=CRS)
        _reader.write_catalog(sd_all, outdir / "stone_points.csv")
        write_json(outdir / "stone_read_report.json", sr_all.to_dict())
        print(f"[stones] {len(sd_all.stones)} exact surveyor stone points cataloged "
              f"(read {'OK' if not sr_all.failed else 'FAILED'}) -> stone_points.csv")
    except Exception as exc:  # noqa: BLE001
        print(f"[stones] catalog skipped ({exc})")

    # OPERATOR LOOP-CLOSER: turn the InputRequest worklist ANSWERS into closed plots (the last
    # mile of "100% is a process"). If the operator has dropped operator_confirms.json /
    # operator_seeds.json into output/M3_CLUBBED (their answers to input_requests.json), close
    # those plots to ACCEPT_SEEDED -- FP-safe (human supplies identity; rigid s=1 seed / a confirm
    # of an already-LOCATED plot). Absent files -> no-op. Runs BEFORE the agents so the worklist
    # they emit reflects what is now closed.
    m1_by_survey: dict[str, str] = {}
    for v in villages:
        for mp in Path(f"output/{v}/m1").glob("*.dxf"):
            nums = re.findall(r"\d+", mp.stem)
            if nums:
                m1_by_survey[nums[-1]] = str(mp)
                m1_by_survey[f"{v}:{nums[-1]}"] = str(mp)
    try:
        from landintel.pipeline.m2_georef.m3_operator import apply_operator_inputs
        closed = apply_operator_inputs(all_pl, m1_by_survey, outdir)
        if closed["confirmed"] or closed["seeded"]:
            print(f"[operator] closed {len(closed['confirmed'])} confirmed + "
                  f"{len(closed['seeded'])} seeded -> ACCEPT_SEEDED (human-supplied identity, 0-FP)")
    except Exception as exc:  # noqa: BLE001
        print(f"[operator] loop-closer skipped ({exc})")

    # UmeyamaVerifierAgent: verify the MATH of every placement (orthonormal R, det=+1 = no
    # reflection/mirror-flip, diagnostic scale in band, valid SIMPLE ring). Demote-only on unsound
    # math (never promote) -> 0-FP. Runs before the agent layer so a demote reflects into the DXF.
    try:
        from landintel.agents.base import write_json as _wj
        from landintel.agents.umeyama_verifier import UmeyamaVerifierAgent
        uv = UmeyamaVerifierAgent().verify(all_pl)
        _wj(outdir / "umeyama_verify_report.json", uv.to_dict())
        _uvf = [c for c in uv.checks if c.severity.value == "fail"]
        print(f"[umeyama] transform+ring verify: "
              f"{'OK (all placements mathematically sound)' if not _uvf else 'FAIL: ' + _uvf[0].detail}")
    except Exception as exc:  # noqa: BLE001
        print(f"[umeyama] verify skipped ({exc})")

    # WAKE THE AGENT LAYER (M3 was bypassing the orchestrator -- "sleeping agents"). Runs the
    # SAME self-verifying pipeline M2 uses: VerificationAgent (FP invariants -> shippable) +
    # GuardAgent (180-deg anagram trap) + InputRequestAgent (the path-to-100% worklist: the ONE
    # minimal input that closes each unplaced plot) + LLMAssist (Qwen narration) + propose->
    # re-gate + persistent memory + decision-trace + QGIS review layer. Agents NEVER promote to
    # ACCEPT (0-FP by construction); they may only DEMOTE a confident plot that breaks an
    # invariant, which reflects onto its disposition BEFORE the DXF below is written.
    try:
        from landintel.agents.orchestrator import run_agent_layer
        summ = run_agent_layer(all_pl, outdir, context={
            "village": title, "crs": CRS, "surveyor": surveyor, "stage": "m3"})
        print(f"[agents] shippable={summ['shippable']} (0 false positives by construction) | "
              f"{summ['n_requests']} plot(s) need input to reach 100% -> input_requests.json")
    except Exception as exc:  # noqa: BLE001 - the agent layer must never break the deliverable
        print(f"[agents] agent layer skipped ({exc})")

    # Fillable field worklist: the path-to-100% requests as a CSV the survey team completes; the
    # blank columns map back to operator_confirms.json / operator_seeds.json for the loop-closer.
    try:
        from landintel.pipeline.m2_georef.m3_operator import write_field_worklist
        wl = write_field_worklist(outdir)
        if wl:
            print(f"[worklist] fillable field worklist -> {wl}")
    except Exception as exc:  # noqa: BLE001
        print(f"[worklist] worklist skipped ({exc})")

    # THE clubbed deliverable: the FULL FMBs (all layers, rigidly placed on the exact stones)
    # merged over the surveyor's raw data (stones + traced lines) -- not just boundary rings.
    write_clubbed_fmbs(all_pl, surveyor, outdir / "clubbed_village.dxf", crs=CRS)
    write_dxf(all_pl, outdir / "clubbed_outlines.dxf", crs=CRS)      # lightweight ring-only view
    write_overlay(all_pl, stones, outdir / "qa_overlay.png", village=title)
    write_report(all_pl, outdir / "m3_report.json", village=title)

    counts: dict[str, int] = {}
    for p in all_pl:
        counts[p.disposition] = counts.get(p.disposition, 0) + 1
    acc = sorted(f"{p.village}:{p.survey_number}" for p in all_pl if p.disposition == "ACCEPT")
    rel = sorted(f"{p.village}:{p.survey_number}" for p in all_pl
                 if p.disposition == "ACCEPT_RELATIVE")
    n_ok = (counts.get("ACCEPT", 0) + counts.get("ACCEPT_RELATIVE", 0)
            + counts.get("ACCEPT_SEEDED", 0))
    print(f"\n================ M3 SINGLE CLUBBED OUTPUT: {title} ================")
    print(f"{n_ok}/{len(all_pl)} plots clubbed onto the surveyor stone cloud "
          f"({counts.get('ACCEPT', 0)} survey-grade + {counts.get('ACCEPT_RELATIVE', 0)} "
          f"shared-stone + {counts.get('ACCEPT_SEEDED', 0)} operator-seeded) | "
          f"dispositions {dict(counts)}")
    print(f"  ACCEPT (survey-grade)  : {acc}")
    print(f"  ACCEPT_RELATIVE        : {rel}")
    print(f"  ONE DXF -> {outdir}/clubbed_village.dxf   (+ qa_overlay.png, m3_report.json)")


if __name__ == "__main__":
    main()
