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

PARALLEL by default: N GPU workers (VRAM-bounded), each a warm engine -> ~N x throughput
on the 64-thread box (this is the 200-FMB-village speed lever). RESUMABLE: re-run to
continue (skips finished DXFs); --force to redo; --serial for the single-engine path.
    input/<VILLAGE>/fmb/*.pdf  ->  output/<VILLAGE>/m1/<VILLAGE>_<stem>.dxf  (+ _manifest.csv)
Usage:  python run_m1.py [VILLAGE] [--force] [--serial]        (default INGUR)
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

# --- MAX-QUALITY OCR configuration (set at import so spawned workers inherit it too) ---
os.environ["LANDINTEL_OCR_TRT"] = "0"                 # CUDA EP (TRT first-build hangs on SM_120)
os.environ["LANDINTEL_BODY_GPU"] = "1"                # body/glyph OCR on the server-det GPU engine
os.environ["LANDINTEL_OCR_MULTIANGLE_AUGMENT"] = "1"  # 6-angle page merge -> max number recall
# rec_batch=256: more text crops per GPU inference call -> higher GPU utilisation and a
# little more VRAM on dense pages, on the 48 GB card. SAFE (recognition batching only).
# NOTE (measured 2026-07-02, 400 GPU samples): this is a text DETECTOR workload -- it runs
# on GPU in ms bursts (~8.5 GB) and is CPU-bound (6-angle warps + vector geometry), so it
# does NOT reach 30 GB / 80%. det_side is kept at the default 1920: 3600 stalls the CUDA
# graph with NO VRAM gain. Sustained high GPU is a generative-VLM profile (PaddleOCR-VL).
# THROUGHPUT LEVER is process parallelism (run_m1_batch_parallel), NOT bigger single-plot GPU.
os.environ.setdefault("LANDINTEL_REC_BATCH", "256")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    # Import inside main so a spawned pool worker importing this module does NOT re-run the
    # batch (Windows spawn re-imports __main__; only __name__=="__main__" executes main()).
    from landintel.pipeline.m1_extract.batch import run_m1_batch, run_m1_batch_parallel

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    village = args[0] if args else "INGUR"
    force = "--force" in sys.argv
    serial = "--serial" in sys.argv          # force the single-engine path (debug/CPU host)

    in_dir = Path(f"input/{village}/fmb")
    out_dir = Path(f"output/{village}/m1")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[M1-QUALITY] {village}: {len(list(in_dir.glob('*.pdf')))} FMB PDFs -> {out_dir}"
          f"  (force={force})", flush=True)
    print("[M1-QUALITY] server-det GPU + vector-guided glyphs + 6-angle augment "
          f"({'SERIAL' if serial else 'PARALLEL'})", flush=True)

    if serial or os.environ.get("LANDINTEL_M1_WORKERS") == "1":
        rows = run_m1_batch(in_dir, out_dir, village, force=force)
    else:
        rows = run_m1_batch_parallel(in_dir, out_dir, village, force=force)

    ok = sum(1 for r in rows if r["ok"])
    proper = sum(1 for r in rows if r.get("proper"))
    errs = [(r["pdf"], r["error"]) for r in rows if not r["ok"]]
    print(f"\n[M1-QUALITY] this run: {ok}/{len(rows)} ok, {proper} PROPER  ->  {out_dir}")
    for pdf, err in errs:
        print(f"   ERROR  {pdf}: {err}")
    print(f"[M1-QUALITY] manifest: {out_dir / '_manifest.csv'}  |  total DXF: "
          f"{len(list(out_dir.glob('*.dxf')))}/{len(list(in_dir.glob('*.pdf')))}")
    print(f"[M1-QUALITY] next: python run_m1_qa.py {village}  (visual verify every plot)")


if __name__ == "__main__":
    main()
