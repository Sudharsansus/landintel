# LandIntel — FMB to DWG Agentic Automation

Production system that converts Tamil Nadu government FMB (Field Measurement Book)
land survey PDFs into georeferenced DWG village maps, fully automated.

## Pipeline (4 modules, run in sequence by orchestrator.py)
- M1 extract:   FMB PDF -> structured DXF  (PyMuPDF vectors + PaddleOCR PP-OCRv5 + ezdxf)
- M2 georef:    DXF -> georeferenced DWG    (affine transform + pyproj + ODA File Converter)
- M3 assemble:  N plots -> village DWG      (ezdxf merge + Shapely boundary snap + annotate)
- M4 report:    reports + delivery          (ReportLab PDF + openpyxl Excel + S3 package)

## ARCHITECTURE CORRECTION (2026-06-29) — M2 vs M3 split, per client's real workflow
The client clarified the stage boundary, which RELABELS the modules. The driving rule:
**M2 georeferences + clubs the FMBs using NO surveyor raw-data file; M3 is the stage that
assembles the clubbed result AGAINST the surveyor raw-data file.** So:
- **M1** (`m1_extract/`): FMB PDF -> per-plot FMB DXF (relative metres). Unchanged.
- **M2 — NEW, `pipeline/m2_club/`** (built 2026-06-29, 11 tests green): takes M1 FMB DXFs
  ONLY (no surveyor file), finds each plot's real UTM coordinates and CLUBS them into ONE
  georeferenced DXF. Uses ALL methods, cross-checked (user's directive "can't rely on one"):
  `cadastral_seat` (survey# -> TNGIS/vector UTM parcel, rigid shape-gated),
  `gps_seat` (operator control points, 2-corner + seed-quality), `relative_club` (FMB-to-FMB
  shared-edge corroboration [label-free] + gated propagation = the client's FMBS_STONES_MATCH).
  Entry: `club_pipeline(m1_dxf_paths, output_dir, crs, cadastral_source=, gps_control=)
  -> list[ClubResult]`. Outputs clubbed_village.dxf + clubbed.geojson + clubbed_points.csv.
  Dispositions ACCEPT / ACCEPT_SEEDED / REVIEW / NO_COVERAGE; 0 FP (math gates decide).
- **M3 — the EXISTING `pipeline/m2_georef/` code** (formerly mis-labelled "M2"): takes **M1 FMB
  DXFs + the surveyor RAW DATA FILE.dxf** -> matched/assembled final. `georef_pipeline(...) ->
  list[GeorefResult]`, RANSAC stone congruence gated by chain_coverage on the traced SITE DATA
  LINE. Everything written below under "M2 georef — BUILT" is really **M3** now; the
  package will be physically renamed `m2_georef -> m3` in a later atomic pass (deferred to keep
  the validated 425-test suite green mid-build). Shared primitives (extract_m1, transform,
  output_dxf, verify, fit) currently live in m2_georef and are imported by m2_club.
- **ARCH CLARIFIED 2026-06-30: M2 and M3 are PARALLEL branches off M1 — M3 does NOT take M2's
  output as input.** Both consume M1 FMB DXFs directly: **M2 = M1 WITHOUT surveyor data** (clubs
  by cadastre/GPS/relative, standalone clubbed DXF output); **M3 = M1 WITH surveyor data**
  (RANSAC match against the raw-data file, standalone output). M2's clubbed DXF is a SEPARATE
  deliverable used nowhere downstream. (This SUPERSEDES the earlier "M2's clubbed DXF feeds M3"
  framing.) The code already matches this — `m2_georef`/M3 reads M1 DXFs + surveyor, never the
  m2_club output.
- Mental model: "M1 gives FMB DXF; then TWO independent branches — M2 georeferences+clubs the
  M1 FMBs WITHOUT the raw data file, and M3 (separately) matches the same M1 FMBs against the
  surveyor raw data file. Two separate outputs; neither feeds the other."

## Agent layer (Claude orchestrates, does not replace modules)
- validator.py  OCR sanity: "44,2"->44.2, reject noise like "Y8t6"
- anomaly.py    detect non-closing boundaries, area mismatch vs FMB stated area
- resolver.py   resolve shared-boundary conflicts between adjacent plots
- audit.py      natural-language audit trail per job
- memory.py     log corrections from day 1; retrieval logic deferred until real data exists

## Known limitation: measurement-label noise (validator vs anomaly boundaries)
- validator.py accepts numeric-but-non-measurement tokens (mis-anchored neighbour survey
  numbers, subdivision plot numbers like "7"/"21") as values — BY DESIGN, not a defect. OCR
  validation cannot tell a plausible number from a real edge measurement.
- This is NOT caught by the area cross-check: computed area comes from the vector boundary ring
  (polygonize), independent of measurement values, so a wrong measurement never moves it.
- The signal that DOES detect it is per-measurement value-vs-anchored-edge-length consistency
  (Measurement.line_length_m vs value), which anomaly.py REPORTS as a diagnostic. But measured
  on real fixtures, ~50% of validator-accepted measurements are inconsistent (≈half of anchored
  numeric tokens are non-measurements), so it is REPORTED, NOT GATED — gating would flood review
  with false positives.
- What the anomaly layer GUARANTEES is output GEOMETRY correctness (closure, area-vs-stated,
  stone count) — independent of measurement-label noise. Note this corrupts dimension LABELS on
  the map, never the boundary geometry (which is from vectors).
- Real fix is upstream (multi-angle OCR + better measurement/label discrimination in anchoring),
  tied to the ~24% OCR recall limitation. Not anomaly.py's job.

## Tech debt / known issues
- API test suite slowness FIXED (2026-06-28): the lifespan blocked ~30s per TestClient on the
  Mongo server-selection timeout. `api/main.py` lifespan now skips index creation when
  `LANDINTEL_SKIP_DB_INIT` is set, and the root `tests/conftest.py` sets it — so every TestClient
  fires instantly (test_jobs.py went to ~2s). The DB dep is overridden per-route in tests, so
  skipping index creation is safe. Never set `LANDINTEL_SKIP_DB_INIT` in production.
- API key (ANTHROPIC_API_KEY) containment is OK — `.env.example` holds only the placeholder
  `sk-ant-replace-me`, `.gitignore` line 3 ignores `.env`, and the real key lives only in the
  gitignored `.env` (no real key in `.env.example` history as checked). STILL PENDING (user
  action, cannot be automated): rotate the key at console.anthropic.com because it was briefly
  exposed earlier; the code reads it from env and never logs it.

## Stack
Python 3.11, FastAPI, Celery+Redis, MongoDB (motor), AWS S3, React+Vite+TS.
Single Docker image (api + worker) -> runs LOCALLY (docker-compose: api+worker+redis+mongo).

## Deployment: LOCAL-HOST, FULLY OFFLINE (decided 2026-06-28)
NOT deployed to any cloud. Everything runs on the local machine, offline:
- LLM brain: local Qwen via Ollama only. `LANDINTEL_LLM_ORDER` defaults to "local" (no cloud
  call ever); claude/manus are strictly opt-in via that env var. Front-ends: `qwen_chat.py`
  (talk), `qwen_code.py` (Claude-Code-style local coding agent), `teach_qwen.py` (teach/verify).
- OCR: PaddleOCR PP-OCRv5 local (mobile-det CPU / server-det via ONNX-CUDA, optional TensorRT).
  Google Vision is an OPTIONAL cloud engine, OFF by default.
- Cadastral: client-provided vector files (GeoJSON/KML/SHP/LandXML/CSV) are local; the S3/TNGIS
  network fetch paths are optional and not on the default path.
- The EC2/nginx/systemd cloud-deploy framing from earlier is SUPERSEDED -- local-host only.

## Tenancy
Single client now, tenant-ready: every Mongo doc has `client_id`,
deps.current_client() returns the hardcoded client for now.

## Two datasets exist — don't confuse them
- Vellore/Gudiyatham/Kallapadi set: analyzed early (standalone PDFs + KALLAPADI village DWG,
  surveys 405/611). This is where the old "boundary width ≈ 3.0" and the 405/611 counts came from.
- Sivagangai/Manamadurai/T.Pudukkottai set: the ACTUAL fixtures on disk (surveys 12–252,
  46 PDF/DXF pairs). All facts below are verified against THIS set. When a fact differs between
  datasets, the Sivagangai set wins because it is what tests run against.

## Key facts learned from sample data (verified against the Sivagangai fixtures)
- FMB PDFs: all numbers are rasterized images (OCR mandatory), geometry is vector.
- Lines by color/width: black strokes = boundary (width 2.0, occasionally 3.0) vs
  internal/subdivision (width 1.0); blue strokes = chain/traverse (absent in ~19 fixtures);
  blue fills = chain arrows/markers; red fills = corner stones.
  Classify with `BOUNDARY_MIN_WIDTH = 1.5` (thick >= 1.5, thin below) — robust across both
  datasets regardless of whether a district draws boundaries at 2.0 or 3.0.
- Existing DXF layers: BOUNDARY, SUBDIVISION_LINES, SUBDIVISION, CHAIN_LINES, CHAINLINE_DIMENSIONS,
  BOUNDARY_DIMENSIONS, DIMENSIONS, SEPARATION_LINE, STONES, SURVEY_NUMBER, BLUE_STROKES,
  Defpoints, 0. (DIMENSIONS appears in 31 of 46 fixtures.)
- Canonical M1 correctness check: PDF red-fill count == DXF STONES layer count. Verified EXACT
  on 8 fixtures (e.g. survey 100: 27, survey 32: 82, survey 31: 4). Use this as the ground-truth
  pairing — it is a real input/expected relationship, stronger than raw line-count snapshots.
  Note: PDF strokes are single segments, so PDF boundary count exceeds the DXF's consolidated
  BOUNDARY polylines (e.g. 30 vs 20) — segment→polyline merge is a to_dxf/assembly concern.
- Page frame removal: the sheet has a border/separator drawn in black that the colour/width
  rules would file as boundary/internal. pdf_vectors drops it with an AXIS-AWARE test
  (FRAME_PAGE_SPAN_RATIO=0.85): a stroke is frame if its x-extent ≥ 0.85·page_width OR its
  y-extent ≥ 0.85·page_height. A plain length cutoff was a false-positive trap — an elongated
  plot's real long edge (survey 31, ~480pt on a 57×370 plot) exceeded it and got deleted,
  breaking closure. Leaving the frame in would silently corrupt M2/M3 georeferencing.
- Boundary closure: build_plot recovers the perimeter via Shapely polygonize on the BOUNDARY
  (thick) segments only (bnd+internal over-extends, e.g. survey 199 → 130%). After the
  axis-aware frame fix, ALL 46 fixtures close and area lands within ~5% of stated
  (survey 100: 1.9%, 199: 5.1%, 31: 2.0%). No fixture is genuinely non-closing; the open-boundary
  path (honest is_closed=False, never force-closed) is exercised by a unit test, not a fixture.
- Corner-label borrowing (build_plot) is best-effort: it takes the nearest short OCR token to
  each stone, so labels tolerate OCR noise and may be empty/garbled. Positions are solid (what
  M2 needs as georef anchors); label strings are NOT load-bearing. CONSEQUENCE FOR M3: match
  shared boundaries between adjacent plots on geometry/proximity, never on corner-label equality.
- Regression-locked extraction counts (bnd/int/chain/blue/stone): survey 100 = 30/25/0/61/27;
  survey 199 = 18/7/303/12/9 (chain-heavy); survey 31 = 13/5/0/5/4 (smallest plot).
- OCR decimal uses comma in source ("44,2" means 44.2). Normalization is the agent's job
  (validator.py), NOT ocr.py — ocr.py returns raw strings untouched so the corrections loop
  can see what OCR actually read vs what got cleaned. (Note: the English PP-OCRv5 rec model
  often reads the comma glyph as a dot anyway, emitting "44.2" directly.)
- Header block reads cleanly and is parsed by ocr.parse_header() -> FmbHeader:
  Survey No, District, Taluk, Village, Scale, Area. e.g. "Scale : 1 : 2021",
  "Area : Hect 01 Ares 66.50" (= 1 ha + 66.50 ares = 1.665 ha; 1 ha = 100 ares).
- Scale (e.g. 1:1079, 1:2021) is per-PDF, read from the header in M1 (ocr.parse_header),
  stored on Plot.scale, and applied in build_plot.py to convert pixel geometry to real-world
  metres. Rule: drawing length = pixel length × scale ÷ DPI factor
  (PDF points → cm via 2.54/72, × scale denominator, ÷ 100 for cm → m).
- AREA CROSS-CHECK (anomaly layer): after applying scale, computed polygon area must match
  the header stated area (Plot.stated_area) within tolerance. A scale read wrong shows up as
  an area mismatch — a free correctness gate. Wire this into anomaly.py.
- OCR recall reality: single-pass mobile-model recall on the small, rotated dimension numbers
  is ~24% (near-horizontal numbers read cleanly; steeply-rotated ones poorly). Header reads
  near-perfectly. Highest-leverage future improvement: multi-angle OCR (OCR the page at a few
  rotations and merge detections) + server det in production — the path from ~24% to usable.
  Not built yet; the anomaly/review layer currently carries that load.
- PaddleOCR: use PP-OCRv5; mobile det locally, server det in production.

## Fixture layout
- tests/fixtures/FMB/  → the raw government FMB PDFs (OCR + vector input), surveys 12–252.
  Filename pattern: FMB_SIVAGANGAI_Manamadurai_T.Pudukkottai_<survey>.pdf
- tests/fixtures/DXF/  → the matching converted DXFs (expected-output reference).
  Filename pattern: SIVAGANGAI_Manamadurai_TPudukkottai_<survey>.dxf
- Same survey number appears in both folders → use them as input/expected pairs in M1 tests.

## M2 georef — BUILT and validated (2026-06-23, INGUR surveyor DXF)

Code in `src/landintel/pipeline/m2_georef/`. Run:
```
python -m landintel.pipeline.m2_georef.pipeline \
  --surveyor "test2/INGUR/INGUR RAW DATA FILE.dxf" \
  --m1-dir test2/INGUR/m1_outputs --output-dir test2/INGUR/georef_final
```

**What it does:** georeferences M1 DXF plots (relative metres) into the parcel's UTM zone by
matching their corner stones against an EXTERNAL surveyor field DXF (`INGUR RAW DATA FILE.dxf`:
522 boundary stones + SITE DATA LINE traced boundaries). NOTE on zone: TN straddles 78E — west is
Zone 43N (EPSG:32643), east is Zone 44N (EPSG:32644). INGUR/Erode is ~77.73E → **43N (32643)**, and
the code default + `process_village --crs auto` (utm.py detect_crs_from_cadastral) reflect this;
do NOT assume 44N. This is the real input, NOT just adjacent
M1 plots — see "what changed" below.

**Matching = geometric congruence, not label/RBS pairing** (`match.py` `geometric_match`). RANSAC
over 2-point similarity transforms (Umeyama: rotation + uniform scale + translation): take the M1
corner ring as a rigid template, fit the exact transform from each candidate stone pair, apply it
to ALL corners, and keep the transform whose congruent inlier subset (≥4 stones within 5 m, distinct
surveyor stones, residual ≤ tol) is largest. No reliance on clean OCR stone labels (recall is ~24%).
Edge-length fingerprinting is retained only as a fallback that may CONFIRM, never override, geometry.

**THE FALSE-POSITIVE GATE IS CHAIN COVERAGE** (`verify.chain_coverage`). KEY DISCOVERY: per-plot
geometric matching OVER-MATCHES in a dense stone cloud — every plot finds SOME congruent subset by
chance (35/35 matched, 92 footprint-overlap conflicts), so inlier count alone has no per-plot signal
separating true from false placement. The decisive signal is the fraction of the georeferenced
boundary lying within 3 m of a surveyor SITE DATA LINE (the actually-traced boundary). On INGUR it
cleanly separates true matches (58–100%) from coincidental ones (12–43%), a wide ~15-point gap. The
matcher never sees this signal, which is why it is independent ground truth.

**Disposition** (`pipeline.py`), every plot accounted for:
- ACCEPT (georeferenced): chain_coverage ≥ 0.50 AND n_inliers ≥ 6 AND residual < 3 m AND all 7
  `verify_georef_dxf` gates pass.
- REVIEW (human confirms): chain_coverage ≥ 0.35 AND n_inliers ≥ 4.
- NO_COVERAGE: below that — the surveyor never traced this plot (off the corridor). M1 output is
  RETAINED but NOT georeferenced. This is a data gap, distinct from a quality REJECT (a match found
  but failed the gates).
- A global `_resolve_footprint_conflicts` pass demotes the lower-coverage plot of any overlapping
  ACCEPT pair to REVIEW, so the ACCEPT set is a non-overlapping tiling (real parcels tile, sharing
  edges not interiors). Safety net behind the chain-coverage gate; changes nothing on INGUR.

**FINAL INGUR RESULT** (35 regenerated M1 plots): ACCEPT=12, REVIEW=6, NO_COVERAGE=17, no false
positives. The 12 ACCEPT surveys: 724,727,729,730,763,765,767,768,771,773,1023,1024 — all on traced
lines, non-overlapping, spatially coherent.

**M2 scale band** (`match._similarity_from_pair`): a 2-point fit is rejected unless `0.5 < s < 2.0`.
This band is CORRECT and protective: M1 already converts pixel geometry to real-world metres (see
Scale facts above) and the surveyor DXF is UTM metres, so a true match has scale ≈ 1 — all 18 INGUR
ACCEPT+REVIEW matches passed it. Do NOT widen it: a non-unity scale would mean an upstream M1 unit
bug to be fixed in M1, not masked in M2; widening only ADDS candidate transforms (false-positive
risk) with zero recall benefit. Don't re-litigate.

**M1-1 prerequisite:** `to_dxf.py` now writes ALL corner stones (synthetic `?N` label when OCR
can't read one) instead of dropping unlabeled ones — restores the red-fill == STONES invariant the
corner template depends on. Regenerate M1 first: `python run_m1_ingur.py --force`.

**Superseded from the 2026-06-05 pre-build spec:** "Base file = M1's own DXF outputs of ADJACENT
plots" and "blocked on adjacent-plot DXF fixtures" — the validated build matches against the external
surveyor field DXF, not adjacent M1 plots. "2 RBS stone points fitted by label/seed" — replaced by
RANSAC geometric congruence (no label dependence). "Validation = residual" was necessary but NOT
sufficient (residual is per-plot and over-matches); chain coverage is the actual gate. The employee
workflow context below remains historically accurate.

**TN government GIS portal** may expose cadastral data — research thread only (potential cleaner
input + authoritative stone-position validation reference). NOT used by the current build.

## Client workflow (real, from employee files)
The client's actual process is a 3-stage manual workflow, confirmed from employee working
files (Kallapadi village, surveys 405/419/611, saved by users "santh"/"INTEL" in AutoCAD 2024):
- Stage 2 — FIELD_BASE_FILE: the existing cadastral base map, survey numbers already positioned.
  In the automated pipeline, this is replaced by M1's own DXF outputs — no separate file needed.
- Stage 3 — FMBS_STONES_MATCH: a new FMB plot is aligned by matching its corner stones onto
  known stone positions from adjacent plots. Stone-to-stone matching, NOT GPS georeferencing.
  Corner-stone POSITIONS (not labels) are load-bearing for this reason.
- Stage 4 — LPS_WORKING: the final land-parcel-system output, built on the matched plots.

Design consequence (UPDATED to match the built M2 — see "M2 georef — BUILT" above):
- M2 = RANSAC geometric-congruence stone-matching of each M1 plot against an external surveyor field
  DXF (UTM metres), gated by chain coverage; output is georeferenced into the parcel's UTM zone
  (43N=EPSG:32643 west of 78E, 44N=EPSG:32644 east; INGUR is 43N). This IS the
  "Stage 3 FMBS_STONES_MATCH" idea (stone POSITIONS load-bearing, labels not) realized against
  authoritative field data rather than against adjacent M1 plots — the earlier "2 anchor stones
  between adjacent M1 outputs" framing is superseded.
- M3 = assemble all aligned plots into a single village DWG.
- GIS config keys are now live: M2 emits absolute UTM coordinates (zone per longitude), not just
  relative. verify.py Check 2 accepts the TN extent across BOTH zones (43N+44N).

## Build order
M1 first (pdf_vectors.py is the foundation), then M4+dashboard, then M2, then M3.
M1 and M2 are DONE (M2 built + validated against the INGUR surveyor DXF, 2026-06-23 — was never
actually blocked on adjacent-plot fixtures once the external surveyor DXF arrived). M3 (village
assembly of the ACCEPT plots) is next. Docker (step 20) is unblocked and on the critical path.
