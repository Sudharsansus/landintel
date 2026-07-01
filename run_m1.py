"""M1 extract: batch-run every FMB PDF in a village input folder -> per-plot DXF.

Fresh, clean I/O:  input/<VILLAGE>/fmb/*.pdf  ->  output/<VILLAGE>/m1/<stem>.dxf
Usage:  python run_m1.py [VILLAGE]        (default INGUR)
"""
from __future__ import annotations

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# GPU OCR (Blackwell/SM_120): the working GPU path is Qwen2.5-VL on torch-CUDA (OCR_ENGINE=hf_vl)
# -- Paddle can't use SM_120 and the onnxruntime server-det balloons/stalls, but PyTorch runs
# natively. hf_vl reads the header with PP-OCRv5 server-det on the small header CROP (fine) and
# the body dimension numbers with Qwen2.5-VL-3B on the GPU (far better recall than the mobile
# detector's ~24%). Set OCR_ENGINE=paddle to fall back to the CPU mobile detector (~20 s/plot).
os.environ.setdefault("OCR_ENGINE", "hf_vl")

from landintel.pipeline.m1_extract.ocr import extract_text, parse_header
from landintel.pipeline.m1_extract.pdf_vectors import extract_vectors
from landintel.pipeline.m1_extract.anchor import anchor_measurements
from landintel.pipeline.m1_extract.build_plot import build_plot
from landintel.pipeline.m1_extract.to_dxf import write_dxf

_ENGINE = os.environ["OCR_ENGINE"]
print(f"[M1] OCR engine: {_ENGINE}  "
      f"({'GPU Qwen2.5-VL (torch-CUDA)' if _ENGINE == 'hf_vl' else 'CPU mobile'})", flush=True)

VILLAGE = sys.argv[1] if len(sys.argv) > 1 else "INGUR"
IN_DIR = Path(f"input/{VILLAGE}/fmb")
OUT_DIR = Path(f"output/{VILLAGE}/m1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

pdfs = sorted(IN_DIR.glob("*.pdf"))
# RESUMABLE: skip plots already extracted so a long (GPU) run survives interruption -- just
# relaunch and it continues. Set LANDINTEL_FORCE=1 to re-extract everything.
_force = os.environ.get("LANDINTEL_FORCE", "0") == "1"
todo = [p for p in pdfs if _force or not (OUT_DIR / f"{p.stem}.dxf").exists()]
done_already = len(pdfs) - len(todo)
print(f"[M1] {VILLAGE}: {len(pdfs)} FMB PDFs -> {OUT_DIR}  "
      f"({done_already} already done, {len(todo)} to do)", flush=True)
print(f"{'survey':>28} {'stones':>6} {'closed':>6} {'area%':>7}  status")

ok = 0
rows = []
for pdf in todo:
    t0 = time.time()
    try:
        vectors = extract_vectors(pdf)
        dets = extract_text(pdf)                 # engine chosen by OCR_ENGINE (hf_vl = GPU)
        header = parse_header(dets)
        result = anchor_measurements(vectors, dets)
        plot = build_plot(vectors, result, header, pdf_path=pdf)
        out_path = write_dxf(plot, OUT_DIR / f"{pdf.stem}.dxf")
        # area agreement (a stated area of 0/None means the header didn't parse)
        area_pct = float("nan")
        if plot.stated_area and plot.stated_area > 0 and plot.computed_area:
            area_pct = 100.0 * abs(plot.computed_area - plot.stated_area) / plot.stated_area
        status = "PROPER" if (plot.is_closed and (area_pct != area_pct or area_pct <= 5)) else "CHECK"
        rows.append((pdf.stem, len(plot.stones), plot.is_closed, area_pct, status))
        print(f"{pdf.stem:>28} {len(plot.stones):>6} {str(plot.is_closed):>6} "
              f"{area_pct:>7.1f}  {status}  ({time.time()-t0:.0f}s)")
        ok += 1
    except Exception as exc:  # noqa: BLE001
        rows.append((pdf.stem, 0, False, float("nan"), f"ERROR: {exc}"))
        print(f"{pdf.stem:>28} {'-':>6} {'-':>6} {'-':>7}  ERROR: {exc}")

total_dxf = len(list(OUT_DIR.glob("*.dxf")))
print(f"\n[M1] this run: {ok}/{len(todo)} extracted; total on disk: {total_dxf}/{len(pdfs)} -> {OUT_DIR}")
n_proper = sum(1 for r in rows if r[4] == "PROPER")
print(f"[M1] this run PROPER (closed + area<=5%): {n_proper}/{len(todo) or 1}")
