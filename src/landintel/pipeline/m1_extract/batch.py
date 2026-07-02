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


def _worker_init(env_snapshot: dict, threads_per_worker: int) -> None:
    """Pool-worker startup: inherit the parent's OCR env, CAP the per-worker thread pools so
    N workers share the 64 cores instead of each grabbing all of them (the oversubscription
    that made naive fan-out ~1.3x instead of Nx: 4 workers x 64-thread OpenCV/OpenMP/BLAS
    pools = 256 threads thrashing 64 cores), and stagger the first engine build so the ONNX
    model load + CUDA allocation don't all fire at once (thundering herd)."""
    import os as _os
    import random
    import time as _t
    _os.environ.update(env_snapshot)
    try:
        import cv2
        cv2.setNumThreads(int(threads_per_worker))
    except Exception:  # noqa: BLE001
        pass
    _t.sleep(random.uniform(0.0, 8.0))


def run_m1_batch_parallel(input_dir, out_dir, prefix: str, *, det_model: str | None = None,
                          force: bool = False, only: set[str] | None = None,
                          workers: int | None = None) -> list[dict]:
    """Parallel M1: ``workers`` processes, EACH holding ONE warm OCR engine, load-balanced
    over the PDFs (one PDF per task; the pool reuses processes so each worker builds its
    engine ONCE and reuses it). This is the throughput lever for large villages -- the
    per-plot pipeline is CPU-bound (6-angle warp + vector geometry), so N processes give
    ~N x on the 64-thread box, while N engines share the one GPU at ~8.5 GB each.

    VRAM-bounded: default workers = min(free_VRAM // ~9.5 GB, cpu-based cap, n_pdfs), so it
    never oversubscribes the card. Same outputs + resumable ``_manifest.csv`` as the serial
    path; a worker crash on one PDF cannot stop the batch. Override with LANDINTEL_M1_WORKERS.
    """
    import concurrent.futures as _cf

    from landintel.pipeline.m1_extract import ocr as _ocr
    det_model = det_model or _ocr.SERVER_DET_MODEL
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
    if not pdfs:
        _log.info("M1 parallel: nothing to do (all DXFs present).")
        return []

    workers = _resolve_workers(workers, len(pdfs))
    _log.info("M1 PARALLEL: %d PDF(s) across %d worker(s) [det=%s rec_batch=%s] -> %s",
              len(pdfs), workers, det_model, os.environ.get("LANDINTEL_REC_BATCH", "32"),
              out_dir)
    if workers <= 1:
        return run_m1_batch(input_dir, out_dir, prefix, det_model=det_model,
                            force=force, only=only)

    # Cap per-worker CPU thread pools so N workers share the cores (no oversubscription).
    # Set in the PARENT env BEFORE spawn so children inherit it at import time (when OpenMP/
    # BLAS/ONNX size their pools), then also applied at runtime in _worker_init for OpenCV.
    n_cores = os.cpu_count() or 16
    tpw = max(2, n_cores // max(1, workers))
    for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_k] = str(tpw)
    _log.info("M1 parallel: %d threads/worker (%d cores / %d workers)", tpw, n_cores, workers)

    env_snapshot = {k: v for k, v in os.environ.items()
                    if k.startswith("LANDINTEL_") or k.endswith("_NUM_THREADS")
                    or k == "VECLIB_MAXIMUM_THREADS"}
    rows: list[dict] = []
    t0 = time.time()
    ctx = _mp_context()
    with _cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=_worker_init,
                                 initargs=(env_snapshot, tpw)) as ex:
        futs = {ex.submit(_process_one, pdf, out_dir, prefix, det_model): pdf
                for pdf in pdfs}
        done = 0
        for fut in _cf.as_completed(futs):
            done += 1
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001 - a hard worker crash on one PDF
                rec = {"pdf": futs[fut].name, "ok": False,
                       "error": f"worker crashed: {type(e).__name__}: {e}"}
            rows.append(rec)
            _log.info("[%d/%d] %s %s (%ss)", done, len(pdfs),
                      "OK " if rec["ok"] else "ERR", rec["pdf"], rec.get("secs"))
            _write_manifest(out_dir, rows)                  # checkpoint as each finishes

    ok = sum(1 for r in rows if r["ok"])
    proper = sum(1 for r in rows if r.get("proper"))
    elapsed = time.time() - t0
    _log.info("M1 PARALLEL DONE: %d/%d ok, %d PROPER in %.0fs (%.1f s/PDF wall, %d workers)",
              ok, len(rows), proper, elapsed,
              (elapsed / len(rows) if rows else 0), workers)
    return rows


def _mp_context():
    import multiprocessing as _mp
    # spawn everywhere: the CUDA/ONNX engine cannot survive a fork.
    try:
        return _mp.get_context("spawn")
    except ValueError:  # pragma: no cover
        return _mp.get_context()


def _resolve_workers(workers: int | None, n_pdfs: int) -> int:
    """Pick a worker count that fits VRAM (each engine ~8.5 GB) and the CPU, capped by the
    number of PDFs. Explicit arg / LANDINTEL_M1_WORKERS wins (still clamped to >=1)."""
    env = os.environ.get("LANDINTEL_M1_WORKERS")
    if workers is None and env:
        try:
            workers = int(env)
        except ValueError:
            workers = None
    if workers is not None:
        return max(1, min(workers, n_pdfs))
    # auto: bound by free VRAM (leave ~4 GB headroom, ~9.5 GB per engine) and CPU.
    vram_cap = 4
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        free_mb = int(out.stdout.strip().splitlines()[0])
        vram_cap = max(1, int((free_mb - 4000) // 9500))
    except Exception:  # noqa: BLE001 - no GPU / nvidia-smi -> conservative
        vram_cap = 2
    cpu_cap = max(1, (os.cpu_count() or 8) // 4)   # ~4 threads/worker for the augment warps
    return max(1, min(vram_cap, cpu_cap, n_pdfs, 6))


def _write_manifest(out_dir: Path, rows: list[dict]) -> None:
    cols = ["pdf", "ok", "out", "survey", "stones", "proper", "fails", "error", "secs"]
    with (out_dir / "_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
