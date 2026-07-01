# LandIntel — WORKLOG (output memory)

Running record of **what was broken, what we fixed, and what's still open**. Newest first.
This is the human-readable companion to the git history and the Claude memory graph.

---

## Project layout (after 2026-07-01 reorg)
```
input/<VILLAGE>/         raw inputs ONLY
  fmb/                   FMB PDFs (M1 input)
  INGUR RAW DATA FILE.dxf surveyor stones (M3 input; M2 uses ONLY for bbox/fence + QA)
  s3_tiles/ tngis_cache_*/  cadastre raster caches (offline M2)
output/<VILLAGE>/        fresh pipeline outputs (clean names, NO timestamps)
  m1/                    M1 per-plot DXFs
  m1_qa/                 M_visual_agent FMB-vs-DXF comparison images
  m2/                    clubbed_village.dxf + clubbed_points.csv + stone_verify.txt
reference/               ported algorithms + reference repos (GPL/MIT, NOT imported)
  librecad_extracted.py  autocad_mcp_extracted.py  M2-ref/  fmb_ml/  Sreeragu-*/
_dump/                   EVERYTHING archived in the reorg (nothing deleted; move-only)
src/landintel/           the pipeline engine (M1..M4, agents, m_agents, llm, api)
run_m1.py  run_m2.py     fresh clean runners  (python run_m1.py INGUR ; python run_m2.py INGUR)
```
Git initialised 2026-07-01 as the real safety net (the old `.git` was empty/broken). Caches,
binaries and `_dump/` are untracked (on disk, not in git).

---

## Pipeline (surveyor-free boundary)
- **M1** `pipeline/m1_extract/`: FMB PDF → per-plot DXF (PyMuPDF vectors + PaddleOCR + ezdxf).
- **M2** `pipeline/m2_club/`: M1 DXFs → ONE georeferenced clubbed village DXF, using
  **cadastre (TNGIS+S3) / GPS / relative-club ONLY — NO surveyor file**. Cadastre-accuracy
  (~13 m) by design.
- **M3** `pipeline/m2_georef/`: M1 DXFs **+ surveyor RAW DATA** → stone-matched <2 m. The
  `stone_refine.py` engine (RANSAC+ICP) lives here logically and is the accuracy path.
- Agents: `agent/` + `agents/` = runtime numeric/LLM 0-FP layer (load-bearing, orchestrator
  imports them). `m_agents/` = NEW human-like **visual** QA (M_visual_agent).

---

## FIXED (2026-07-01 session)
- **M2 "diamonds" review — 1 adopted, 2 parked, 12 rejected.** Client sent a 15-tool catalog of
  niche algorithms and asked which actually improve ours. Judged against the REAL code, not the
  catalog's assumptions:
  - **ADOPTED — robust corner residual (partial-Chamfer / Modified-Hausdorff, catalog #13/#15).**
    Our `rot_residual` is already a MEAN nearest-neighbour distance (not brittle Hausdorff), but a
    single OCR/raster-jittered parcel corner can inflate that mean and fail an otherwise-correct
    fit ONLY on residual (code already notes surveys 668/1024/1025 do exactly this). Added
    `_robust_corner_residual` (trims the worst ~20% of corner→boundary distances) in
    `m5_cadastral/fit.py`, stored as `CadastralFit.rot_residual_robust`, surfaced in the seat
    note. Gate switch is env-gated `LANDINTEL_CAD_ROBUST_RESID` (default OFF) taking `min(mean,
    robust)` → strictly recall-additive, **0-FP-safe** because seat-locality (not residual) is the
    FP lock. 7 new tests; full m5/m2_club/m2_georef suites green (152). Flip ON + INGUR-regress
    after M1 regen, then decide to default it on.
  - **PARKED (post-delivery, zero-dep):** HoughLinesP gap-close (#14) + skeleton→graph→cycles
    (#12) for tile boundary extraction — real, but they rewrite a *working* component; only worth
    it if visual QA shows broken tile boundaries.
  - **REJECTED (with reasons):** **TPS non-rigid warp (#4/#11) — would DESTROY the true-scale
    survey lengths we just fixed; the exact trap.** POT/Sinkhorn (#3, we use ICP not Hungarian),
    Grounded-SAM/SAM2 (#1/#2, non-deterministic boundaries = FP risk, heavy GPU), GUDHI (#9, heavy
    dep for what shapely covers), Swiss-DL/SmartLandMaps/GMN/RS-CLIP/DeepLSD (#5–8,10,7, need
    training data / research code / dep-heavy where #14 is free).
- **GPU safety ceiling (M1 OCR).** The batched Qwen2.5-VL run ballooned to 44 GB / 89 °C (near
  the 48 GB OOM edge). Added a hard `torch.cuda.set_per_process_memory_fraction` cap in
  `_build_hf_qwen_engine` (env `LANDINTEL_VL_MEM_FRAC`, default 0.80) + relaunched at
  `LANDINTEL_VL_BATCH=4`. Result: steady **14 GB / 72 °C** — the card physically cannot max out.
  M1 stays resumable (skips done plots), so this cost 0 progress.
- **TRUE-SCALE placement (survey lengths preserved).** The cadastral fit's scale (~0.95, the
  raster cadastre being slightly off) was baked into geometry → every clubbed edge shrunk
  2-5% (perimeter 0.83–1.13 vs true M1). Fix: fitted scale only GATES; geometry emitted as a
  pure rigid body scale=1 (`cadastral_seat._rigidify`, `CAD_RIGID_SCALE`). Result: perimeter
  ratio **median 1.000** (0.99–1.01). This is the client's "one base point, keep the lengths".
- **De-overfit.** Absolute metre thresholds tuned to INGUR (~13 m error, ~200 m plots) made
  size-relative (fraction of each plot's diagonal + floor) in `edge_align` and `stone_refine`.
  Reclub reproduced INGUR exactly → proof it's general, not tuned. Kept the truly scale-free
  gates (angles, scale band, inlier counts).
- **Boundary merge** = translation-only `edge_align` (corroborated shared edges) + anchor-aware
  `boundary_snap`. Gaps median → 0.
- **Visual/label fixes** (`m2_georef/output_dxf.py`): each FMB → a selectable BLOCK; labels
  rotate with the plot; synthetic `?N` stone labels dropped (161→0 on 770); neighbour labels
  dropped; dimension text shrunk 0.55× for legibility.
- **stone_verify.py** — ground-truth QA agent: per-plot corner→nearest-true-stone error +
  shape-congruence verdict (MATCH / SHIFTED / SHAPE_CHECK). VERIFY-ONLY, never places.
- **Reorg** into input/output/reference/_dump; git safety net; fresh run_m1/run_m2.

## Earlier milestones (see Claude memory graph for detail)
- M2 club reached 34/34 clubbed on INGUR (cadastre TNGIS+S3 composite; survey 698 solved via
  S3 whole-parent yellow recovery). True-scale later made it 33 ACCEPT + 1 REVIEW (776).
- M1 complete on Sivagangai + Manur sets (stones-exact, closed, area within ~5%).

---

## OPEN / KNOWN PROBLEMS
- **M1 quality — under fresh VISUAL verification (2026-07-01).** Client suspects M1 has visible
  errors that numeric gates (closure/area/stone-count) pass as false positives. `M_visual_agent`
  now renders FMB-PDF vs M1-DXF side by side (`output/<v>/m1_qa/`) for human/agent visual review.
  → **action: run M1 fresh, then visually verify every plot.**
- **776 sliver** → REVIEW: genuinely overlaps its neighbour in the cadastre; true-scaling
  surfaced it honestly rather than shrinking to fit. Needs the surveyor file (M3) to resolve.
- **Accuracy ceiling (M2).** Cadastre-only placement is ~13 m; <2 m needs M3 stone-matching or
  survey-number-LABELLED stones (the current surveyor stones are unlabelled → matching is
  ambiguous, ~half the plots anchor confidently). Not an algorithm gap — a DATA gap.
- **AutoCAD lock**: `test2/` and `M2-ref/` couldn't move to `_dump/` while open in AutoCAD —
  pending a close, then `mv` them.

## BACKLOG / next
- Wire genuinely-additive extracts into m5 cadastre recovery: `extract_closed_loops`,
  `bulge_to_arc`/ellipse (curved boundaries) from `reference/librecad_extracted.py`.
- `autocad-mcp` as a delivery/human-loop bridge into the client's AutoCAD LT (local, offline).
- Qwen: add coding capabilities + web-fetch tool (local brain).
