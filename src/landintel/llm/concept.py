"""THE CONCEPT -- the whole LandIntel work loaded into the LLM's head.

This is what makes the LLM reason about a plot the way Claude does in dev, instead of just
narrating. The LLM gets (a) SYSTEM_CONCEPT -- the entire pipeline, every gate + threshold,
every known failure mode, and THE HARD RULE -- and (b) a per-plot evidence bundle, and is
asked to DIAGNOSE the root cause and PROPOSE one fix.

THE HARD RULE (stated to the model and enforced in code): the model may only propose an
action from SAFE_ACTIONS. It can NEVER emit geometry, a coordinate, or an "accept" verdict.
Every proposed fix is re-run through the unchanged deterministic gate, and only the gate can
accept. So the model reasons like Claude but cannot create a false positive -- worst case its
proposal is re-gated and rejected, or clamped to "no_action".
"""

from __future__ import annotations

import json

# --- the bounded fix vocabulary -------------------------------------------------
# AUTO_ACTIONS: a re-runnable DETERMINISTIC re-attempt. The pipeline re-runs it and the
#   UNCHANGED gate decides -- the model only picks WHICH safe re-attempt to try.
# INPUT_ACTIONS: the plot genuinely lacks data -> ask the operator for the minimal input.
AUTO_ACTIONS = {
    "rot180_relabel":      "re-query the cadastre with the 180deg-rotated survey number "
                           "(OCR read it upside-down, e.g. 669<->699); re-gate.",
    "road_closure_recover": "run recover_open_parcel: locally bridge yellow-wall gaps + "
                           "seal with local orange + flood-fill from the label; re-gate.",
    "multi_angle_ocr":     "re-OCR this tile at more rotations / warm-core mask to recover "
                           "the parcel label or boundary; re-gate.",
    "topology_corroborate": "try to upgrade via a shared real edge (>=20m) with a confident "
                           "neighbour, if the plot's own fit residual is sane; re-gate.",
}
INPUT_ACTIONS = {
    "request_clearer_parcel":   "parcel is merged/open in the source tiles -> ask for a "
                                "clearer/closed polygon (shapefile/KML/sharper image).",
    "request_two_corner_seed":  "no usable position -> ask for 2 corner stone -> UTM points.",
    "request_village_reference": "plot is from a different village -> ask for that village's "
                                "cadastral/surveyor reference.",
    "confirm_placement":        "located but just under the auto-accept bar -> ask a human "
                                "yes/no (or 2 corner points).",
    "no_action":                "already confident, or nothing safe to try.",
}
SAFE_ACTIONS = {**AUTO_ACTIONS, **INPUT_ACTIONS}


SYSTEM_CONCEPT = """\
You are the reasoning agent inside LandIntel, a pipeline that georeferences Tamil Nadu
government FMB (Field Measurement Book) land-survey plots into a single UTM village map.

WHAT THE PIPELINE DOES, end to end (the corrected M1 -> M2(m2_club) -> M3(m2_georef) split):
- M1 extract (pipeline/m1_extract):  each FMB PDF -> a per-plot FMB DXF in relative metres
               (corner STONES + boundary), via PyMuPDF vector parse + PaddleOCR + ezdxf. OCR
               recall on the small rotated dimension numbers is only ~24%, so labels are NOISY
               and NOT load-bearing; stone POSITIONS are solid.
- M2 club (pipeline/m2_club, NEW): takes the M1 FMB DXFs ONLY -- NO surveyor raw-data file --
               finds each plot's real-world UTM coordinates and CLUBS the plots into ONE
               georeferenced DXF. Entry point `club_pipeline(m1_dxf_paths, output_dir, crs,
               cadastral_source=, gps_control=) -> list[ClubResult]`; outputs clubbed_village.dxf
               + clubbed.geojson + clubbed_points.csv. It uses ALL available coordinate methods,
               cross-checked (never bets on one source):
               * cadastral_seat -- survey# -> authoritative UTM parcel (TNGIS cadastral tiles or
                 client vector), placed rigidly, gated by a strict shape check (area ratio +
                 scale ~= 1 + orientation + corner residual).
               * gps_seat -- operator GPS / control-point correspondences (2-corner similarity +
                 a seed-quality baseline gate).
               * relative_club -- FMB-to-FMB shared-edge clubbing (the client's
                 "FMBS_STONES_MATCH"): LABEL-FREE geometric corroboration of seated plots + gated
                 propagation to seat un-parcelled neighbours.
               Dispositions: ACCEPT / ACCEPT_SEEDED / REVIEW / NO_COVERAGE. 0 false positives --
               deterministic math gates decide every ACCEPT.
- M3 georef (pipeline/m2_georef, the EXISTING surveyor-matching code -- it was formerly
               mis-labelled "M2"; m2_club is now M2 and this is M3): takes M2's clubbed
               georeferenced FMBs PLUS the surveyor RAW DATA FILE.dxf and assembles/matches them
               via RANSAC stone congruence, gated by chain_coverage against the surveyor's traced
               SITE DATA LINE. Output geometry is ENTIRELY the FMB DXF, placed rigidly (rotation +
               uniform scale ~= 1 + translation); it is NEVER warped to surveyor pixels, and
               surveyor geometry is NEVER copied in. Entry point `georef_pipeline(...) ->
               list[GeorefResult]`.
- M4 report:   club the accepted plots into one village DWG + a PDF/Excel/zip deliverable.

MENTAL MODEL: M1 gives the FMB DXF -> M2 (m2_club) georeferences + clubs the FMBs WITHOUT the
surveyor raw-data file -> M3 (m2_georef) takes the clubbed DXF and matches it against the
surveyor raw-data file. 0-FP throughout; the LLM/brain only reads, proposes, narrates, and
remembers -- the deterministic math gate decides every placement.

HOW A PLACEMENT IS DECIDED (deterministic gates -- the ONLY thing allowed to ACCEPT):
- M2 (m2_club) cadastral_seat gates: area_ratio in [0.65,1.55], scale in [0.80,1.25],
  rot_residual <= 12 m, corridor distance <= 500 m, orientation not ~90deg flipped.
  fit.scale is the strongest identity signal (the fit scales M1 to the parcel, so a
  wrong-size parcel shows up as a bad scale).
- M2 (m2_club) relative_club: LABEL-FREE shared-edge corroboration -- two independent
  sources agreeing on a boundary confirms a placement; gated propagation tiles a seated
  neighbour without overlapping any placed plot.
- M3 (m2_georef) geometric fit: RANSAC stone congruence; the false-positive GATE is chain
  coverage (fraction of the boundary lying on a surveyor-traced SITE DATA LINE): ACCEPT needs
  coverage >= 0.50 AND >= 6 inliers AND residual < 3 m AND 7 verify gates pass.
- Seat-locality gate: reject if the placed centroid is > 600 m from the cadastral label seat.
- The confident set must be a NON-OVERLAPPING tiling (real parcels tile, sharing edges
  not interiors). No cross-village plot may be confident.
- Self-calibrating coverage gate: per span, the ACCEPT chain-coverage bar is auto-raised
  to the gap above the coincidental cluster (TIGHTEN-ONLY; never below the 0.50 floor).
- Seed quality gate: a 2-corner operator seed on too-short a baseline (worst-case far-corner
  error > 2 m) is forced to REVIEW, not auto-accepted -- a short baseline amplifies point error.

INPUTS THE ENGINE ADAPTS TO (any land, not just INGUR; UTM zone auto-detected 43N/44N at 78E):
- the FMB PDF (always, M1); a cadastral vector by survey number -- GeoJSON / KML/KMZ /
  Shapefile / LandXML / CSV / TNGIS export (POSITION+ROTATION+IDENTITY reference) and operator
  2-corner -> UTM GPS seeds, both consumed by M2 (m2_club) to club WITHOUT a surveyor file; and
  an external surveyor field DXF (UTM, the corridor ground truth) consumed by M3 (m2_georef) to
  match the clubbed FMBs against the surveyor's traced lines.
- When the client supplies EXACT edge lengths (LandXML/CSV/surveyor), label verification
  confirms/corrects each FMB dimension label (label_confidence) -- a DISPLAY/correction signal,
  never a gate, so noisy OCR labels never move geometry.
- Open/merged parcels: recovered_candidates offers alternative closed rings, each re-gated.
- Deliverable: M4 assembles the confident plots into one village DWG + a PDF/Excel/zip package.

KNOWN FAILURE MODES (diagnose against these):
- merged/open parcel: the parcel outline is fused with a neighbour or open toward a road,
  so area_ratio is far off and the fit residual is large -> road_closure_recover, else ask
  for a clearer parcel.
- 180deg OCR flip: a label read upside-down (6<->9), e.g. "699" is really "669" -> the
  rot180 anagram. If both anagrams look confident < 20 m apart that is the rotation-phantom
  trap; only one is real.
- undersized / partial parcel face: only part of the parcel is traced -> low coverage / bad
  area_ratio.
- cross-village: the FMB belongs to a different village than this cadastre.
- no parcel found: the cadastral source has no label/parcel for this survey number.

THE HARD RULE: you may ONLY choose one action from the provided SAFE_ACTIONS list. You may
NEVER output a coordinate, geometry, or an "accept"/"place it" verdict. Whatever you propose
is re-run through the unchanged deterministic gate, and only the gate can accept a plot. If
nothing safe applies, choose "no_action". Never invent numbers; reason only from the evidence.
"""


def plot_evidence(r) -> dict:
    """The compact evidence bundle the model reasons over for one plot (numbers only)."""
    method = r.match_method or ""
    ev = {
        "survey_number": r.survey_number,
        "disposition": r.recommendation,
        "match_method": method,
        "cad_residual_m": (None if r.cad_residual in (None, float("inf"))
                           else round(r.cad_residual, 1)),
        "error": r.error or "",
        "m1_file": r.m1_file,
    }
    # the cadastral path stores area_ratio in chain_coverage; geometric stores coverage there
    if method.startswith("cadastral"):
        ev["area_ratio"] = round(r.chain_coverage, 2)
    elif method.startswith("geometric"):
        ev["chain_coverage"] = round(r.chain_coverage, 2)
    return ev


def propose_prompt(ev: dict) -> str:
    """Build the per-plot diagnose+propose prompt (concept is sent as the system message)."""
    actions = "\n".join(f"  - {k}: {v}" for k, v in SAFE_ACTIONS.items())
    return (
        "A plot could not be auto-placed with confidence. Diagnose the most likely root "
        "cause from the evidence, then choose exactly ONE action from SAFE_ACTIONS.\n\n"
        f"EVIDENCE:\n{json.dumps(ev, indent=2)}\n\n"
        f"SAFE_ACTIONS (choose the key of exactly one):\n{actions}\n\n"
        "Reply with STRICT JSON only, no prose:\n"
        '{"survey_number": "...", "hypothesis": "<root cause in one sentence>", '
        '"action": "<one SAFE_ACTIONS key>", "rationale": "<why this action, one sentence>"}'
    )


def rule_based_proposal(ev: dict, cross_village: bool) -> dict:
    """Deterministic diagnoser -- the fallback when no LLM is reachable.

    Maps the evidence to the same {hypothesis, action, rationale} a model would produce, so
    the reasoning loop works fully offline. Mirrors the known failure modes in SYSTEM_CONCEPT.
    """
    method = ev.get("match_method", "")
    disp = ev.get("disposition", "")
    area = ev.get("area_ratio")
    resid = ev.get("cad_residual_m")
    cov = ev.get("chain_coverage")

    if cross_village:
        return dict(hypothesis="FMB belongs to a different village than this cadastre.",
                    action="request_village_reference",
                    rationale="cannot place without that village's reference frame.")
    if disp == "NO_COVERAGE":
        return dict(hypothesis="off the surveyed corridor and no cadastral label found.",
                    action="request_two_corner_seed",
                    rationale="no auto position exists; 2 corner->UTM points place it exactly.")
    if method.startswith("cadastral"):
        bad_area = area is not None and not (0.65 <= area <= 1.55)
        bad_resid = resid is not None and resid > 12.0
        if bad_area or bad_resid:
            # merged/open parcel: try the deterministic road-closure recovery first.
            return dict(
                hypothesis=f"parcel merged/open in the tiles (area_ratio={area}, "
                           f"fit_resid={resid}m) -- boundary not cleanly resolved.",
                action="road_closure_recover",
                rationale="locally bridge the road-side gap + reseal, then re-gate; "
                          "if still bad the operator supplies a clearer parcel.")
        return dict(hypothesis=f"cadastral fit borderline (area_ratio={area}).",
                    action="confirm_placement",
                    rationale="just under the auto-accept bar; a human confirm finalizes it.")
    if method.startswith("geometric"):
        return dict(
            hypothesis=f"matched real stones but boundary only partly traced "
                       f"(coverage={cov}).",
            action="topology_corroborate",
            rationale="if it shares a real edge with a confident neighbour at sane "
                      "residual, the gate can upgrade it; else a human confirm.")
    return dict(hypothesis=ev.get("error") or "located but unconfirmed.",
                action="request_two_corner_seed",
                rationale="needs 2 corner->UTM points to place exactly.")


def parse_proposal(text: str, ev: dict) -> dict | None:
    """Defensively parse the model's JSON and CLAMP the action to SAFE_ACTIONS.

    Any action not in SAFE_ACTIONS (a hallucination) is dropped -> None, so an off-vocab
    reply can do nothing. This is half the safety; the re-gate is the other half.
    """
    try:
        s = text[text.index("{"): text.rindex("}") + 1]
        d = json.loads(s)
    except Exception:  # noqa: BLE001
        return None
    action = str(d.get("action", "")).strip()
    if action not in SAFE_ACTIONS:
        return None
    return {
        "survey_number": ev["survey_number"],
        "hypothesis": str(d.get("hypothesis", ""))[:240],
        "action": action,
        "rationale": str(d.get("rationale", ""))[:240],
    }
