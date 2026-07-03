# M3 Investigation — Session Handoff (portable across machines)

Pull this repo on any machine and open Claude Code in it to continue with full context.
Everything below is the state as of the latest commit on `main`.

## Where M3 stands (committed)
- **M3 finishing done** (commit `589c446`): scale-locked placement (rule-2 fix — places rigid
  `s=1`, keeps `s_fitted` as a diagnostic), honest dispositions (ACCEPT / REVIEW / NEEDS_GPS /
  UNMEASURED), and the three deliverables per village
  (`output/<V>/m3/clubbed_village.dxf` + `qa_overlay.png` + `m3_report.json`).
- **End-to-end run (3 Erode villages):** MOOLAKARAI **3/20** ACCEPT, KANDAMPALAYAM **1/19**,
  NASIYANUR **0/11** — honest, **0 false positives**.
- Run it: `python run_m3.py <VILLAGE> --surveyor "C:\Users\Admin\Desktop\M3 INPUT\RAW DATA.dxf" --district Erode`
  (needs `GOOGLE_MAPS_API_KEY` in the gitignored `.env`; surveyor DXF is a local file, not in the repo).

## First 10-agent investigation — CONCLUSION (8/10 delivered, unanimous)
**The honest ACCEPT ceiling for these three villages is 4. You cannot get more without faking it.**
- The surveyor RAW DATA is a **corridor trace** — it only set field stones on part of each plot's
  ring, so most corners have no stone to match (data gap, not a bug). *(A2, A7)*
- The ~500-stone village window is **saturated with chance congruences**: matching the 4
  known-good ACCEPTs against the whole window relocated 3 of 4 to wrong seats **700–1300 m away
  at equal quality**; even the genuine KAND 15 has a tie-quality decoy seat. *(A3)*
- The `classify` gate leaks **~5% false ACCEPTs** on off-seat plots; production safety rests on the
  **locality crop + tiling**, not the gate. Any recall boost that widens the window / reaches
  off-seat pushes FP toward ~5% (unsafe); staying inside GROW locality keeps FP ≈ 0 (safe). *(A10)*
- **M1 is not the constraint**; rotation is not ambiguous (self-congruence 0 everywhere). *(A8)*

### Rejected (all measured, not guessed)
- **A4 "trim-to-gate" (13/19 → ACCEPT)** — RETRACTED. It drops real corners until the number
  passes = gate-gaming = false positives. A6 measured the honest version: **0 new ACCEPTs**;
  A3 confirmed 0 recall failures a perfect matcher lifts to ACCEPT.
- **Per-plot TNGIS seat crop** — net **−3** ACCEPTs (divergent cadastre seats strip correct
  stones). *(A1)*
- **Window widening / whole-window global search** — chance seats at 300–2200 m. *(A2, A3, A10)*
- **Uniqueness-gap gate** — true and chance both at gap 0, no separation. *(A10)*
- **Tightening the gate (max 3.0→2.5 or med 2.0→1.5)** — A10 called it "free" but it demotes real
  ACCEPTs (KAND 15 max 2.81; MOOL 18 med 1.84). Do NOT tighten.
- **`m2_m3_lifts/` modules** (symmetry-axis seeding, PROSAC, residual_distribution, gate_evaluator)
  — clean, general, tests pass, but wrong target; `residual_distribution` would regress our median
  gate. Keep as a reference shelf; do not integrate.

### The two SAFE wins to integrate (0 new ACCEPTs, pure honesty gains, A10-cleared)
1. **Pairing fix** *(A2 + A6, same algorithm)*: replace `geometric_match`'s nearest-only inlier
   count + whole-pose collision-discard with a **distinct one-to-one corner→stone assignment**,
   then a **local scale-locked (s=1) rigid refit** of the *accepted* pose + one re-pair pass,
   **MAD-pruned**, with a **drift guard** (refine-in-place; if centroid moves > tolerance, keep the
   original pose — NEVER a global re-search, which FPs at 300 m). Reuses `STONE_TOL_FLOOR_M=3.0`,
   `STONE_TOL_K=0.08`, MAD k=3. Effect: tighter rings, +1 corner on many plots, **6 honest
   NEEDS_GPS→REVIEW recoveries + 1 REVIEW→NEEDS_GPS correctness fix**. Pseudocode in A6's report
   (scratchpad `a6/`).
2. **Disposition honesty** *(A7 + A3)*: stop surfacing >gate-residual placements (KAND 36 @27 m,
   45 @51 m) as "located"; route located-but-`s_fitted≠1` → **REVIEW not NEEDS_GPS** (never ACCEPT).

### The only honest path to MORE ACCEPTs
Upstream, not matcher tricks: better M1 corner fidelity (multi-angle OCR) + real **GPS seeds** for
the saturated plots. (Second agent batch B6 is quantifying the minimal-GPS-input worklist.)

## Second 10-agent investigation — NEW ideas (running now)
B1 global village jigsaw · B2 exploit the ignored `M3.xml` · B3 stone-code topology filter ·
B4 spatial clustering pre-segmentation · B5 boundary reconstruction → synthetic coverage gate ·
B6 minimal GPS-seed worklist · B7 placement covariance/uncertainty · B8 adjacent-FMB shared-stone
corroboration · B9 QGIS/GDAL external validation · B10 rarity-weighted correspondence.
Findings will be appended here as they land.

## Next action (pending user go)
Integrate the two safe fixes → re-run the 3 villages → run `pytest tests/m2_georef/ tests/m2_club/`
→ commit + push. Result stays 4 ACCEPT with cleaner REVIEW/NEEDS_GPS accounting and tighter rings.

## Key files
- `run_m3.py` — M3 entry (anchor + grow).
- `src/landintel/pipeline/m2_georef/m3_deliverables.py` — scale-locked placement, `classify()`, deliverables.
- `src/landintel/pipeline/m2_georef/match.py` — `geometric_match` (the pairing fix goes here).
- `src/landintel/pipeline/m2_georef/stone_matcher.py` — `rigid_procrustes` (scale-locked), `rigid_stone_match`.
- `src/landintel/pipeline/m2_club/disposition_thresholds.py` — the shared validated constants.
- `M3_ISSUE.md` — the expert brief on the stone-cloud problem.
