"""M1 extract -- MAXIMUM QUALITY (quality over speed; time does not matter).

Every layer at its best setting so stones, lines, and number labels are extracted as
completely as possible:
  * STONES  = red vector fills  -> geometric, OCR-independent, exact (100%).
  * LINES   = vector strokes by colour/width -> exact (100%).
  * LABELS  = the only OCR-dependent layer. Read with the HIGHEST-quality config:
      - server-det on GPU (PP-OCRv5_server_det via onnxruntime-CUDA) -- best detector,
      - vector-guided glyph pass (OCRs each number where a vector fill says one is),
      - + MULTI-ANGLE augment (page OCR'd at 0/30/60/90/120/150 deg, tokens merged) --
        the documented path from ~24% single-pass recall toward complete.
    No OCR is physically guaranteed 100% on tiny rotated numbers, so run_m1_qa.py then
    VISUALLY verifies every plot (FMB vs DXF) and flags any gap for a targeted re-read.

Serial + one warm engine (time is not a constraint). RESUMABLE: re-run to continue
(skips finished DXFs); --force to redo.
    input/<VILLAGE>/fmb/*.pdf  ->  output/<VILLAGE>/m1/<VILLAGE>_<stem>.dxf  (+ _manifest.csv)
Usage:  python run_m1.py [VILLAGE] [--force]        (default INGUR)
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

# --- MAX-QUALITY OCR configuration (all set before importing the pipeline) ---
os.environ["LANDINTEL_OCR_TRT"] = "0"                 # CUDA EP (TRT first-build hangs on SM_120)
os.environ["LANDINTEL_BODY_GPU"] = "1"                # body/glyph OCR on the server-det GPU engine
os.environ["LANDINTEL_OCR_MULTIANGLE_AUGMENT"] = "1"  # 6-angle page merge -> max number recall
logging.basicConfig(level=logging.INFO, format="%(message)s")

from landintel.pipeline.m1_extract.batch import run_m1_batch

args = [a for a in sys.argv[1:] if not a.startswith("-")]
VILLAGE = args[0] if args else "INGUR"
force = "--force" in sys.argv

IN_DIR = Path(f"input/{VILLAGE}/fmb")
OUT_DIR = Path(f"output/{VILLAGE}/m1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[M1-QUALITY] {VILLAGE}: {len(list(IN_DIR.glob('*.pdf')))} FMB PDFs -> {OUT_DIR}"
      f"  (force={force})", flush=True)
print("[M1-QUALITY] server-det GPU + vector-guided glyphs + 6-angle multi-angle augment",
      flush=True)

rows = run_m1_batch(IN_DIR, OUT_DIR, VILLAGE, force=force)

ok = sum(1 for r in rows if r["ok"])
proper = sum(1 for r in rows if r.get("proper"))
errs = [(r["pdf"], r["error"]) for r in rows if not r["ok"]]
print(f"\n[M1-QUALITY] this run: {ok}/{len(rows)} ok, {proper} PROPER  ->  {OUT_DIR}")
for pdf, err in errs:
    print(f"   ERROR  {pdf}: {err}")
print(f"[M1-QUALITY] manifest: {OUT_DIR / '_manifest.csv'}  |  total DXF: "
      f"{len(list(OUT_DIR.glob('*.dxf')))}/{len(list(IN_DIR.glob('*.pdf')))}")
print("[M1-QUALITY] next: python run_m1_qa.py INGUR  (visual verify every plot)")
