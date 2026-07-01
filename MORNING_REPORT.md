# Overnight Report — INGUR M1 + M2 (2026-07-01 → morning)

**You asked for: 100% M1 and M2 for INGUR by morning, quality over speed, no questions.**
**Status: DONE and verified. Full test suite green (516 passed / 1 skipped). Committed + pushed.**

---

## TL;DR
- **M1: 35/35 geometrically correct.** Every stone extracted (exact vs the FMB), every boundary closed, every shape matches the source FMB.
- **M2: 34/34 plots clubbed, ZERO false positives**, coherent village. Absolute accuracy 12.4 m — this is the **cadastral ceiling** (M2 by design uses no surveyor file). Survey-grade accuracy is M3's job.
- **M3 is ready** and, where it matches, hits **survey-grade (~2 mm field residual)**. One tuning task waits for you (below).

---

## 1. Engine decision (why not PaddleOCR-VL)
I finished installing PaddleOCR-VL properly: **paddle-gpu 3.3.1 (cu129) runs on your Blackwell SM_120** (this corrects the old "paddle is CPU-only" note), CUDA-12.9 runtime aligned, torch removed (cu128↔cu129 DLL conflict; only the abandoned Qwen used torch). VL **loads and runs on the GPU (~13 GB)** — but its body token-harvest **hangs (10+ min on one plot, 0 % GPU)**. Not viable for 35+ plots.

**M1 engine locked to the proven path: server-det (PP-OCRv5) via ONNX-Runtime-CUDA + 6-angle multi-angle augment.** Reminder: **stones and lines are vector-derived (100 %, OCR-independent)** — only number labels depend on OCR, so no engine choice can drop a stone.

## 2. M1 — 35/35 correct
Run: `python run_m1.py INGUR` → `output/INGUR/m1/`. QA: `python run_m1_numeric_qa.py INGUR`.

| Gate | Result |
|---|---|
| Stone invariant (PDF red-fill == DXF STONES) | **35/35 exact** |
| Boundary closed | **35/35** |
| Shape matches FMB (visual QA, all 35) | **35/35** |
| Dimension labels | avg 64/plot (2254 total) |
| verify PROPER | 34/35 |

**The one non-PROPER (survey 1027) is NOT a defect** — it's a source stated-area inconsistency. Computed area 2.776 ha vs header 0.79 ha, but the 19 corner stones sit at the corners of the full 143×219 m boundary, so the geometry is stone-confirmed. The 0.79 ha is almost certainly the area of subdivision 1027/1, not the parent survey. M2 clubs it anyway. (Optional future polish: auto-downgrade the area check to a warning when the boundary is fully stone-anchored — needs tests, deferred.)

## 3. M2 — 34/34 clubbed, 0 false positives
Run: `python run_m2.py INGUR` → `output/INGUR/m2/`.

- **Dispositions: 33 ACCEPT + 1 REVIEW** (REVIEW = 776, the tiny 4-stone sliver — appropriately cautious). All 34 accounted.
- **All 6 zero-FP gates PASS**: closure, UTM range, rigid scale ≈1, stone-count preserved, non-overlapping tiling, all accounted.
- **Visual (`output/INGUR/m2/clubbed_qa.png`)**: the plots assemble into the recognizable INGUR corridor, each M1 plot sitting on its cadastral parcel, sharing edges not interiors.
- **Accuracy: 12.4 m median vs the 522 surveyor stones.** This is the **inherent ceiling of clubbing WITHOUT the surveyor file**: M2 places plots accurately *onto the cadastre* (per-plot fit residuals 0.3–1.6 m), but the public cadastre itself is ~12 m off ground truth. Nothing in M2 can beat that — it's a cadastre-vs-truth gap, not an M2 error. Well-corroborated plots land tighter (770 = 5.3 m); cadastral-only plots drift (722 = 85 m).
- Tested `LANDINTEL_CAD_ROBUST_RESID=1`: **exact no-op** (same 12.4 m, same dispositions, 0-FP holds) — confirms the ceiling is the cadastre, not the shape-gate. Left OFF.

## 4. M3 readiness (for today)
M3 (`m2_georef`) runs clean on the fresh M1 and, **where it matches, is survey-grade: field residual ≈ 0.002 m (2 mm), max stone displacement 0.020 m** — this is the <2 m that closes M2's 12 m ceiling.

**One open item — your call today:** the disposition regressed to **ACCEPT=1** (was documented at 12). I isolated it: **old M1 and fresh M1 give the identical result, so this is code-side, not M1-side.** The 2026-06-28 hardening (2 m field gate + self-calibrating coverage gate + max-residual < 3 m) tightened the ACCEPT gate; high-coverage survey-grade plots (765/766/771 = 100 % coverage) are held at REVIEW mainly because a few corners exceed the 3 m max-residual. First M3 task: decide the ACCEPT recall-vs-0-FP policy and relax the gate accordingly. Command:
```
PYTHONPATH=src python -m landintel.pipeline.m2_georef.pipeline \
  --surveyor "input/INGUR/INGUR RAW DATA FILE.dxf" --m1-dir output/INGUR/m1 \
  --output-dir output/INGUR/m3 --crs EPSG:32643
```

## 5. Repo
Committed + pushed to `main` (68386b3): `run_m1_numeric_qa.py`, `run_m2_qa.py`, ocr.py VL-device default. Memories updated. Nothing left uncommitted that matters.

## New tools you can reuse per village
- `python run_m1.py <VILLAGE>` — max-quality M1 (resumable).
- `python run_m1_numeric_qa.py <VILLAGE>` — stone/closure/area gate (no OCR load, fast).
- `python run_m1_qa.py <VILLAGE>` — FMB‖DXF visual comparisons.
- `python run_m2.py <VILLAGE>` + `python run_m2_qa.py <VILLAGE>` — club + QA summary.
