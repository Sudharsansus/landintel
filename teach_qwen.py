"""Teach the local Qwen brain WHAT WE ARE DOING, then verify it understood.

Run:  python teach_qwen.py

Three steps:
  1. Seed the persistent MEMORY GRAPH with project-level knowledge facts (architecture, the
     gates, the HARD RULE, the current hardening, the TN-FMB domain backlog). These persist
     across every session, so the brain carries the concept forward (remembers all sessions).
  2. Check whether the local Qwen (Ollama) is reachable.
  3. If reachable: send Qwen the SYSTEM_CONCEPT + the seeded knowledge and ask it to (a) explain
     LandIntel back in its own words and state the HARD RULE, and (b) diagnose one sample plot
     and pick exactly one SAFE_ACTION -- proving it understands and stays inside the safe vocab.

Nothing here changes any placement: the brain only reads/explains/proposes; the math gate
still decides. Safe to run any time; idempotent (knowledge nodes are upserted by topic).
"""
from __future__ import annotations

import json

from landintel.llm.concept import (SAFE_ACTIONS, SYSTEM_CONCEPT, parse_proposal,
                                    plot_evidence, propose_prompt)
from landintel.llm.memory_graph import default_graph
from landintel.llm.providers import llm_call, local_llm_status

# --- 1) the knowledge we want Qwen to carry forever --------------------------
KNOWLEDGE = [
    ("mission",
     "LandIntel georeferences Tamil Nadu FMB land-survey plots into one UTM village DWG, "
     "fully automated, with an ABSOLUTE prime directive of ZERO false positives.",
     ["core"]),
    ("architecture",
     "M1 extract (FMB PDF -> metric DXF). Then TWO PARALLEL branches off M1 -- M3 does NOT take "
     "M2's output (clarified 2026-06-30). M2 = m2_club: M1 FMB DXFs ONLY, NO surveyor file -> "
     "find UTM coords + CLUB into one georeferenced DXF (STANDALONE output), via cadastral_seat + "
     "gps_seat + relative_club, all cross-checked. M3 = m2_georef (the existing surveyor-matching "
     "code, formerly mis-labelled M2): the SAME M1 FMB DXFs + surveyor RAW DATA FILE -> RANSAC "
     "stone congruence gated by chain_coverage (separate output). M2's clubbed DXF feeds nothing "
     "downstream. M4 report (village DWG + PDF/Excel/zip). Rigid rotation+scale~1+translation "
     "throughout; geometry is NEVER warped to pixels.",
     ["core"]),
    ("hard_rule",
     "Deterministic math gates are the ONLY thing that can ACCEPT a placement. The LLM/agents "
     "may only diagnose, pick one SAFE_ACTION, request input, narrate, or remember -- never "
     "emit geometry, a coordinate, or an accept. Every proposal is re-gated; worst case it is "
     "rejected. So the brain cannot create a false positive.",
     ["core", "safety"]),
    ("fp_gates",
     "FP gates: chain-coverage >=0.50 (self-calibrating, tighten-only), seat-locality 600m, "
     "non-overlapping tiling, cadastral area_ratio/scale/rot-residual bands, seed-quality "
     "short-baseline gate, zone-agnostic verify range across UTM 43N+44N.",
     ["safety"]),
    ("input_adaptive",
     "Inputs adapt to any land: FMB PDF + surveyor UTM DXF + cadastral vector by survey number "
     "(GeoJSON/KML/Shapefile/LandXML/CSV/TNGIS) + operator 2-corner seeds. UTM zone auto-detected "
     "at 78E (43N west / 44N east; INGUR is 43N).",
     ["core"]),
    ("hardening_2026_06_28",
     "Built (opt-in/default-off until real-data validation): robust-loss + seed-covariance, "
     "multi-angle OCR augment, TensorRT, label_confidence verification, recovered_candidates, "
     "M4 village deliverable, acre/cent + decimal area parsing, API test 30s->2s.",
     ["status"]),
    ("domain_backlog",
     "Top real-data backlog: model FMB G-line/F-line/ladder (likely the real fix for ~24% OCR "
     "recall + label noise); subdivisions as first-class parcels; chain/link dimension units; "
     "curved boundaries; Tamil-script headers. Validate on real client files, do not build blind.",
     ["backlog"]),
    ("biggest_risk",
     "The engine has been validated on 3 districts but has NEVER run on real client data; "
     "the new input loaders are synthetic-tested only. That is the #1 risk.",
     ["status"]),
    ("m2_club_built_2026_06_29",
     "NEW M2 'm2_club' is BUILT + green: FMB DXFs only (no surveyor) -> club_pipeline -> one "
     "clubbed georeferenced DXF + GeoJSON + points CSV. Methods: cadastral_seat (survey#->UTM "
     "parcel, rigid shape gate), gps_seat (operator control points + seed-quality), relative_club "
     "(label-free shared-edge corroboration + gated propagation). verify.py (6 gates: closure/area, "
     "UTM range, rigid-scale, stone-count, non-overlap tiling, all-accounted; demote-only) + "
     "qa_render.py overlay. The runtime agent layer (Verification/InputRequest/Guard/LLMAssist) now "
     "works over BOTH ClubResult (M2) and GeorefResult (M3) via one PlotDisposition adapter; agents "
     "NEVER ACCEPT. Orchestrator runs M1->M2(club, opt-in via cadastral/gps)->M3(surveyor). Full "
     "suite 474 passed.",
     ["status", "core"]),
    ("cadastre_solved_validated_2026_06_29",
     "INGUR M2 club SOLVED + validated: 24 ACCEPT / 5 REVIEW / 6 NO_COVERAGE (from 0 ACCEPT), "
     "0 false positives by the pipeline's own gates (on-parcel + on-label + clean tiling + shape). "
     "KEY: reconstruction from the YELLOW boundary net + green-net recovered_candidates + a "
     "PARCEL-SIZE-RELATIVE seat gate (max(60, 1.6*sqrt(area/pi)) -- de-overfit vs the absolute 90m). "
     "LESSON (the hard way): the agents' combined yellow+magenta net + over-merge reject + "
     "metre-ladder kernels REGRESSED INGUR 23->1 (magenta is INGUR's INTERNAL subdivision, "
     "unioning it fragments the parcel) -- REVERTED to yellow-only; those belong behind per-village "
     "adaptation, not blanket defaults. BIG FINDING: the '699 false positive' (2313m from M3) is "
     "actually an M3 GROUND-TRUTH ERROR -- M3 placed 699 overlapping 1024 by 68% (impossible for "
     "tiling parcels), while the pipeline placed 699 on the real correctly-OCR'd SW '699' parcel "
     "with 0% neighbour overlap. So the pipeline is 0-FP AND more correct than the surveyor ref on "
     "699. Validate against M3 with that caveat: M3 itself has errors; the pipeline's internal "
     "tiling+on-label gates are the real 0-FP guarantee. Remaining NO_COVERAGE = parcels the "
     "cadastre couldn't reconstruct (honest recall gap), recoverable to REVIEW via aggressive "
     "recovery (follow-on).",
     ["status", "cadastre", "solution"]),
    ("m2_club_ingur_34accept_2026_06_30",
     "INGUR M2 club reached **34/34 ACCEPT, 0 FP** (only survey 9 = KANDAMPALAYAM, a different "
     "village, correctly REVIEW). Survey 698 -- a large 491x184 m two-lobed PARENT that TNGIS draws "
     "only as fragments -- was the last holdout; 4 hired agents proved the cadastre route: A "
     "(boundary-chamfer) false-locks on roads, B (cell-union) showed TNGIS's 698 parent boundary is "
     "INCOMPLETE, C (neighbour-tiling) fails on the sparse corridor; D CRACKED it -- the S3 "
     "(mypropertyqr) tiles draw 698's parent WHOLE in YELLOW (subdivisions are magenta there). "
     "Production fix (GENERAL, no 698 hardcoding): `recover_parent_yellow` extracts ALL bounded "
     "faces of the S3-yellow net (a yellow face = a whole survey since subdivisions are magenta) "
     "over low close-kernels in a ~1km window (so a large parcel whose label sits at its edge is "
     "not clipped/border-rejected); the rigid area_ratio+scale+rot+seat+tiling gate is the sole "
     "ACCEPT arbiter so it only ADDS recall (0-FP). Also removed a redundant 3x-median area cap "
     "that discarded large parents, and FIXED CompositeCadastralSource.is_aggressive: a plot is "
     "aggressive only if NEITHER source offers a non-aggressive parcel -- so a clean S3 parent is "
     "not demoted just because the TNGIS primary was an aggressive sub-cell (that bug kept 698 at "
     "REVIEW even after the parent was recovered). 698 ACCEPT verified: area 52697 m^2 (FMB 54817, "
     "ratio 0.96), 0.0% overlap with all 33 neighbours, clean tiling, scale 0.98, rot 3m. Full "
     "suite 480 passed. LESSON: don't overfit to one plot -- the all-faces+wide-window method is "
     "generic (compact, magenta-subdivision S3 cadastre). [Earlier 33-accept state superseded.]"),
    ("m2_club_ingur_33accept_2026_06_30_superseded",
     "[SUPERSEDED by m2_club_ingur_34accept] INGUR M2 club pushed 24 -> 33 ACCEPT / 1 genuine REVIEW (698) of the 34 INGUR plots "
     "(survey 9 = KANDAMPALAYAM, a different village, correctly held REVIEW not ACCEPT), 0 FALSE "
     "POSITIVES -- every ACCEPT matches the independent M3 surveyor ref within ~37m EXCEPT three "
     "explained outliers (699=M3's OWN error 2313m off; 776=M3 bad ref area_ratio 0.10; 723=64m "
     "between two imperfect cadastral estimates of the same parcel). FOUR fixes, each 0-FP "
     "(seat-locality "
     "+ area + scale + orientation remain the FP lock; rot_residual never stopped an FP): "
     "(1) SCALE-AWARE rot_residual gate -- the rigid corner residual is measured vs a z18 RASTER "
     "polygon (coarse approxPolyDP), so its tolerance must scale with parcel size: "
     "max(12, 0.30*equiv_radius), not flat 12m. Recovered 668/1024/1025 (failed ONLY rot_residual "
     "at 14-24m while area/scale/orient/seat all passed). (2) MULTI-DETECTION LABEL DISAMBIGUATION "
     "-- THE big one: a survey number is OCR'd at SEVERAL spots; the single highest-confidence "
     "reading is often NOT inside the true parcel (667/723/730 had the right label discarded by "
     "~100m, e.g. 730's correct label was conf 0.994 vs a wrong conf-1.000 one 111m off). Keep ALL "
     "in-fence detections; try the best first (existing ACCEPTs unchanged) then fall back to "
     "alternates ONLY when the net leaks -- the label whose flood-fill CLOSES self-selects the "
     "true position. Recovered 667/723/724/730 (all 5-22m from M3). (3) 16-COMPASS DIRECTIONAL "
     "SEED PROBE in reconstruct_parcel(directional=True) -- seed the flood-fill offset in the 16 "
     "cardinal/intermediate/sub-directions so a label in a parcel CORNER/sub-cell still floods the "
     "full face; added as recovered_candidates (gate decides). 0-FP, INGUR ACCEPT unchanged "
     "(its last 3 are SHAPE-MISMATCH reconstructions, not direction-missable) -- it is the "
     "generalization lever for DENSE 120-FMB villages. (4) MULTI-SOURCE COMPOSITE CADASTRE "
     "(`CompositeCadastralSource`, TNGIS primary + mypropertyqr S3 fallback) -- different public "
     "cadastres disagree on vintage/subdivision for a few parcels and one raster renders a small "
     "parcel cleaner than another; the composite OFFERS the S3 parcel as extra gated candidates "
     "where TNGIS fails. The rigid+seat gate stays the sole ACCEPT arbiter (additive, 0-FP). "
     "Recovered 670 (5.6m from M3) + 1022 (0.0m, M3-ACCEPT) -> 33 ACCEPT. 698 stays REVIEW "
     "(BOTH cadastres fail it: M3 also REVIEW area_ratio 6.97) = genuine VINTAGE/SUBDIVISION "
     "mismatch (FMB = parent survey, cadastre = a child sub-parcel; same family as the 699 "
     "namespace divergence). The remaining un-ACCEPTable plots are a cadastre-vs-FMB DATA problem "
     "(survey #s subdivided/merged/re-surveyed between FMB and cadastre vintages), not an algo bug "
     "-- the surveyor RAW DATA file (M3) is the resolver. Root-cause method: reconstruct at BOTH "
     "my-OCR-label and M3-true position -- closes clean at true but leaks at mine => bug is OCR "
     "POSITION, not raster closing. GDAL only tidies tile-fetching (tiles already cached); it does "
     "NOT fix label-to-parcel association.",
     ["status", "cadastre", "solution"]),
    ("cadastre_solution_2026_06_29",
     "4-agent analysis SOLVED the INGUR '0 ACCEPT' cadastre problem -> one layered solution, all "
     "0-FP, projected ~20-30 ACCEPT (from 0): (1) COVERAGE: harvest the full-village TNGIS tiles "
     "from the surveyor-DXF extent (~350 tiles, ~2 min, offline) + village_fence in ocr_labels to "
     "kill cross-village repeated-survey FPs -> 14/14 missing surveys recovered. (2) GREEN-NET "
     "parcels: the magenta net is SUBDIVISION lines; the SURVEY-PARCEL boundary is GREEN (hue "
     "~28-55). Label-seeded flood-fill on the green net -> 13/13 clean parcels (was 2/13) -> pass "
     "the rigid shape gate -> 12 ACCEPT. (3) SEAT-LOCALITY GATE is the decisive 0-FP lock (shape "
     "gate alone admits 27/156 wrong pairs): placed centroid must be <=90m (raw label <=30m) from "
     "the OCR label point (true 10-50m, wrong 170m+). (4) BLOCK-ASSEMBLY: neighbour-labels are "
     "DEAD on real data; use GEOMETRY-only coin>=4 coincident-corner adjacency (0.95 precision) -> "
     "8 blocks cover 27/35; ONE green-net anchor snaps a whole block (reaches polygon-less plots). "
     "Build order: green-net+seat-locality (validated core) -> coverage harvest+fence -> coin>=4 "
     "block-assembly. Geometry placed RIGIDLY, gates only tighten, 0 FP throughout.",
     ["status", "cadastre", "solution"]),
    ("tngis_cadastre_debug_2026_06_29",
     "TNGIS tile cadastre debugged on real INGUR: (1) FIXED tile-seam clipping -- vectorise_parcels "
     "now STITCHES all tiles into one mosaic before vectorising, so seam-crossing parcels stay "
     "whole; (2) FIXED runner bbox to the FULL cached-tile extent (was the 13-label cluster) -> "
     "coverage 13->21 of 35 surveys. KNOWN LIMITS: the cached tiles cover only the 727-784 cluster "
     "so 14 plots (667-670,695-699,721,1019,1022-1025) are NO_COVERAGE (un-harvested village area, "
     "needs a wider tile harvest); and the z18 raster vectorises into FRAGMENTED parcels, so the "
     "label POINT (~10m, load-bearing) is reliable but the POLYGON is supplementary and fails the "
     "rigid shape gate. INGUR club result: 21 LOCATED (REVIEW, positioned on cadastre), 0 ACCEPT "
     "(fragmented polygons can't confirm shape; with 0 ACCEPT anchor, relative-club corroboration "
     "can't upgrade), 14 NO_COVERAGE. Next: wider harvest + a position-based ACCEPT path (label "
     "point + neighbour corroboration) since polygons are unreliable.",
     ["status", "cadastre"]),
]


def seed_knowledge():
    g = default_graph()
    for topic, fact, tags in KNOWLEDGE:
        g.record_knowledge(topic, fact, tags)
    return g


def main():
    g = seed_knowledge()
    facts = g.recall_knowledge()
    print(f"[1] Seeded {len(KNOWLEDGE)} knowledge facts into the memory graph.")
    print(f"    graph stats: {json.dumps(g.stats(), indent=0)}")

    status = local_llm_status()
    print(f"\n[2] Local Qwen status: reachable={status['reachable']} "
          f"host={status['host']} model={status.get('wanted_model')} "
          f"present={status.get('model_present')}")

    if not status["reachable"]:
        print("\n[3] Qwen (Ollama) not reachable right now -- SKIPPING the live check.")
        print("    The concept + knowledge are persisted; the brain will use them as soon as")
        print("    Ollama is up (start it, `ollama pull qwen2.5:7b`, re-run this script).")
        return

    # Feed the concept + the seeded knowledge as the system message.
    knowledge_block = "\n".join(f"- {f['topic']}: {f['fact']}" for f in facts)
    system = SYSTEM_CONCEPT + "\n\nPROJECT KNOWLEDGE (carry this forward):\n" + knowledge_block

    print("\n[3a] Asking Qwen to explain LandIntel back + state the HARD RULE...")
    out = llm_call(
        "In 4 sentences and your own words: what does LandIntel do, and what is the HARD RULE "
        "about who is allowed to ACCEPT a placement?",
        max_tokens=300, system=system)
    if out:
        text, provider = out
        print(f"    [{provider}] says:\n    " + text.replace("\n", "\n    "))
    else:
        print("    No provider answered (Qwen unreachable mid-call).")
        return

    print("\n[3b] Giving Qwen a sample unplaced plot -- it must pick ONE SAFE_ACTION...")

    class _R:  # a minimal evidence row
        survey_number = "668"
        recommendation = "NO_COVERAGE"
        match_method = "cadastral_rigid"
        cad_residual = 18.0
        chain_coverage = 1.9   # area_ratio for the cadastral path
        error = "parcel merged/open in tiles"
        m1_file = "668.dxf"

    ev = plot_evidence(_R())
    out2 = llm_call(propose_prompt(ev), max_tokens=250, system=system)
    if out2:
        text2, provider2 = out2
        prop = parse_proposal(text2, ev)
        if prop and prop["action"] in SAFE_ACTIONS:
            print(f"    [{provider2}] VALID proposal -> action={prop['action']!r}")
            print(f"      hypothesis: {prop['hypothesis']}")
            print("    UNDERSTOOD: it diagnosed + chose a safe action inside the vocab. "
                  "(The gate still decides; this is only a proposal.)")
        else:
            print(f"    [{provider2}] replied but the action was off-vocab -> clamped to "
                  "no_action by parse_proposal (safety held).")
    print("\nDone. Qwen has the concept + persistent knowledge and stays inside the safe vocab.")


if __name__ == "__main__":
    main()
