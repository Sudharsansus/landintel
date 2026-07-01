"""M1 batch driver -- SINGLE shared OCR engine (safe on RAM and VRAM).

HARD LESSON (measured 2026-06-28): fanning out N worker processes that each load the full
OCR stack (paddle + torch-cu128 + onnxruntime-gpu) + its own model copy + high-DPI page
buffers consumed ~40 GB RAM AND ~16 GB VRAM *per process* -> 3 workers exhausted 128 GB RAM
and maxed the 48 GB GPU. So we do NOT fan out heavy OCR processes.

THE DESIGN: ONE process holds ONE warm OCR engine (~15 GB VRAM, a few GB RAM) and runs the
PDFs through it. This is exactly what the original serial runners did and survived (120 Manur
PDFs in 77 min). Because OCR is ~85-90% of per-PDF time and a single GPU is the throughput
ceiling, process fan-out buys at most ~2x at catastrophic memory cost -- not worth it.

WHERE THROUGHPUT ACTUALLY COMES FROM (no extra processes, no RAM blow-up):
  * ``LANDINTEL_REC_BATCH`` -- more text crops per GPU inference call (default 32; raise on the
    48 GB card for dense pages -- feeds the ONE session bigger batches).
  * TensorRT execution provider (ORT exposes TensorrtExecutionProvider here) -- ~2-5x per
    inference on the same single session; the real speed lever (opt-in, builds an engine once).
A single OCR session (~15 GB) also leaves the card free for the Qwen brain (~6.6 GB) -> the two
engines share one GPU at ~22 GB, comfortably.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from pathlib import Path

_log = logging.getLogger(__name__)


def _process_one(pdf: Path, out_dir: Path, prefix: str, det_model: str) -> dict:
    """Extract one PDF -> DXF (+verify sidecar) using the process-wide warm OCR engine."""
    from landintel.pipeline.m1_extract.anchor import anchor_measurements
    from landintel.pipeline.m1_extract.build_plot import build_plot
    from landintel.pipeline.m1_extract.ocr import extract_text, parse_header
    from landintel.pipeline.m1_extract.pdf_vectors import extract_vectors
    from landintel.pipeline.m1_extract.to_dxf import write_dxf
    from landintel.pipeline.m1_extract.verify_dxf import verify_m1_dxf, write_verify_sidecar

    t0 = time.time()
    rec = {"pdf": pdf.name, "ok": False, "error": ""}
    try:
        vectors = extract_vectors(pdf)
        dets = extract_text(pdf, det_model=det_model)       # reuses the warm engine
        header = parse_header(dets)
        anchor = anchor_measurements(vectors, dets)
        plot = build_plot(client_id=prefix.lower(), vectors=vectors, detections=dets,
                          anchor_result=anchor, header=header)
        out = write_dxf(plot, out_dir / f"{prefix}_{pdf.stem}.dxf")
        vr = verify_m1_dxf(out, stated_area_ha=plot.stated_area)
        write_verify_sidecar(vr)
        rec.update(ok=True, out=out.name, survey=plot.survey_no,
                   stones=len(plot.corner_points),
                   proper=vr.proper, fails=[c.name for c in vr.failures],
                   secs=round(time.time() - t0, 1))
    except Exception as e:  # noqa: BLE001 - one bad PDF must not stop the batch
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["secs"] = round(time.time() - t0, 1)
    return rec


def run_m1_batch(input_dir, out_dir, prefix: str, *, det_model: str | None = None,
                 force: bool = False, only: set[str] | None = None) -> list[dict]:
    """Run M1 over every PDF in ``input_dir`` through ONE shared OCR engine; write DXFs +
    ``_manifest.csv`` to ``out_dir``. Single-process by design (memory-safe)."""
    from landintel.pipeline.m1_extract import ocr as _ocr
    det_model = det_model or _ocr.SERVER_DET_MODEL          # GPU OCR (one shared session)
    input_dir, out_dir = Path(input_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_dir.glob("*.pdf"),
                  key=lambda p: int(p.stem) if p.stem.isdigit() else 1 << 30)
    if only:
        pdfs = [p for p in pdfs if p.stem in only]
    if not force:
        before = len(pdfs)
        pdfs = [p for p in pdfs if not (out_dir / f"{prefix}_{p.stem}.dxf").exists()]
        if before != len(pdfs):
            _log.info("Skipping %d already-generated DXF(s) (force=True to regenerate)",
                      before - len(pdfs))

    _log.info("M1 batch: %d PDF(s), ONE shared OCR engine [det=%s rec_batch=%s] -> %s",
              len(pdfs), det_model, os.environ.get("LANDINTEL_REC_BATCH", "32"), out_dir)

    rows: list[dict] = []
    t0 = time.time()
    for i, pdf in enumerate(pdfs, 1):
        rec = _process_one(pdf, out_dir, prefix, det_model)
        rows.append(rec)
        _log.info("[%d/%d] %s %s (%ss)", i, len(pdfs),
                  "OK " if rec["ok"] else "ERR", rec["pdf"], rec.get("secs"))
        _write_manifest(out_dir, rows)                      # checkpoint every PDF (resumable)

    ok = sum(1 for r in rows if r["ok"])
    proper = sum(1 for r in rows if r.get("proper"))
    elapsed = time.time() - t0
    _log.info("M1 BATCH DONE: %d/%d ok, %d PROPER in %.0fs (%.1f s/PDF)",
              ok, len(rows), proper, elapsed, (elapsed / len(rows) if rows else 0))
    return rows


def _write_manifest(out_dir: Path, rows: list[dict]) -> None:
    cols = ["pdf", "ok", "out", "survey", "stones", "proper", "fails", "error", "secs"]
    with (out_dir / "_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
