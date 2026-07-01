"""process_village.py -- the tower-to-tower / village span ENGINE (input-adaptive).

One command processes ONE land span end to end and FUSES whatever inputs the client provides:
  FMB PDFs        -> M1 extract (the parcel geometry)
  raw data DXF    -> M2 corridor reference (--surveyor; the tower-to-tower SITE DATA LINE)
  TNGIS / vector  -> authoritative cadastral source (--cadastral .geojson/.kml/.shp)
  LandXML / XML   -> exact parcels + points          (--xml)
  CSV             -> points-or-WKT cadastral          (--csv)
  2-corner seeds  -> operator placement for stragglers (--seed seeds.json)

Every plot is placed by the RICHEST available signal, the agent layer self-verifies at 0 FP,
and the whole run is recorded into the memory graph (so each span TRAINS the next). Adaptable
to any land: village/CRS are parameters, the cadastral loader auto-detects fields/format.

Usage:
  python process_village.py --village INGUR --pdfs test2/INGUR --out deliveries/INGUR \
      --surveyor "test2/INGUR/INGUR RAW DATA FILE.dxf" --cadastral tngis_ingur.geojson \
      --crs EPSG:32643
  python process_village.py --village Manur --pdfs test2/Manur --out deliveries/Manur   # M1 only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")
os.environ.setdefault("OCR_ENGINE", "paddle")


def _load_cadastral(args, log):
    """Build a CadastralSource from whichever of --cadastral/--xml/--csv was given."""
    path = args.cadastral or args.xml or args.csv
    if not path:
        return None
    from landintel.pipeline.m5_cadastral.source import load_cadastral
    try:
        src = load_cadastral(path, target_crs=args.crs)
        log.info("cadastral reference: %s (%d parcels)", path, len(src.survey_numbers()))
        return src
    except Exception as exc:  # noqa: BLE001
        log.error("cadastral load failed (%s): %s", path, exc)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Process ONE land span end to end (input-adaptive).")
    ap.add_argument("--village", required=True, help="span/village name (keys the memory graph)")
    ap.add_argument("--pdfs", required=True, help="folder of this span's FMB PDFs")
    ap.add_argument("--out", required=True, help="delivery folder")
    ap.add_argument("--surveyor", default=None, help="raw data DXF (tower corridor) -> run M2")
    ap.add_argument("--cadastral", default=None, help="TNGIS/vector file (.geojson/.kml/.shp)")
    ap.add_argument("--xml", default=None, help="LandXML survey export (.xml)")
    ap.add_argument("--csv", default=None, help="CSV points/WKT cadastral (.csv)")
    ap.add_argument("--seed", default=None, help="seeds.json: [{survey,corner_a,corner_b,utm_a,utm_b}]")
    ap.add_argument("--crs", default="EPSG:32643", help="UTM CRS (43N=32643 west of 78E, 44N=32644)")
    ap.add_argument("--cpu", action="store_true", help="force CPU mobile-det OCR")
    ap.add_argument("--force", action="store_true", help="regenerate existing M1 DXFs")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("process_village")

    # CRS: auto-detect the UTM zone from the cadastral data (TN spans 43N & 44N) so the
    # engine adapts to ANY land; fall back to 43N if it can't be sniffed.
    if args.crs == "auto":
        from landintel.pipeline.utm import detect_crs_from_cadastral
        src = args.cadastral or args.xml or args.csv
        detected = detect_crs_from_cadastral(src) if src else None
        args.crs = detected or "EPSG:32643"
        log.info("CRS: %s (%s)", args.crs, "auto-detected" if detected else "default 43N")

    out = Path(args.out)
    m1_dir = out / "m1"
    m1_dir.mkdir(parents=True, exist_ok=True)

    from landintel.pipeline.m1_extract import ocr as _ocr
    from landintel.pipeline.m1_extract.batch import run_m1_batch
    from landintel.llm.memory_graph import default_graph

    # ---- STEP 1: M1 extract (safe single shared OCR engine) ----
    log.info("=" * 70)
    log.info("SPAN %s -- STEP 1/4: M1 extract (%s)", args.village,
             "CPU mobile-det" if args.cpu else "GPU server-det")
    det = _ocr.DEFAULT_DET_MODEL if args.cpu else _ocr.SERVER_DET_MODEL
    rows = run_m1_batch(args.pdfs, m1_dir, args.village, det_model=det, force=args.force)
    ok = sum(1 for r in rows if r["ok"])
    proper = sum(1 for r in rows if r.get("proper"))
    log.info("M1: %d/%d extracted, %d PROPER", ok, len(rows), proper)

    # ---- STEP 2: record M1 to memory (per-span TRAINING) ----
    log.info("SPAN %s -- STEP 2/4: record M1 outcomes to memory", args.village)
    default_graph().record_m1(rows, args.village)

    # ---- STEP 3: M2 georef + agent layer (corridor + cadastral fused) ----
    results = None
    if args.surveyor:
        from landintel.pipeline.m2_georef.pipeline import georef_pipeline
        cad = _load_cadastral(args, log)
        m1_paths = sorted(m1_dir.glob(f"{args.village}_*.dxf"))
        georef_out = out / "georef"
        log.info("SPAN %s -- STEP 3/4: M2 georef (%d M1 plots, corridor=%s, cadastral=%s)",
                 args.village, len(m1_paths), Path(args.surveyor).name, bool(cad))
        results = georef_pipeline(args.surveyor, m1_paths, georef_out,
                                  crs=args.crs, cadastral_source=cad, village=args.village)
    else:
        log.info("SPAN %s -- STEP 3/4 SKIPPED: no --surveyor. M1 retained + memory updated.",
                 args.village)

    # ---- STEP 4: seed-place the stragglers the operator supplied ----
    if args.seed and results is not None:
        _apply_seeds(args, out / "georef", results, log)

    _summary(args, m1_dir, ok, len(rows), proper, results)


def _apply_seeds(args, georef_out, results, log):
    """Operator-supplied 2-corner seeds close any plot the corridor/cadastre missed."""
    from landintel.pipeline.m2_georef.pipeline import seed_place
    from landintel.pipeline.m2_georef.extract_surveyor import extract_surveyor
    seeds = json.loads(Path(args.seed).read_text())
    surveyor = extract_surveyor(args.surveyor) if args.surveyor else None
    by_sn = {r.survey_number: r for r in results}
    n = 0
    for s in seeds:
        sn = str(s["survey"])
        r = by_sn.get(sn)
        if r is None or r.recommendation in ("ACCEPT", "ACCEPT_CADASTRAL", "ACCEPT_SEEDED"):
            continue
        sr = seed_place(r.m1_file, surveyor, s["corner_a"], s["corner_b"],
                        tuple(s["utm_a"]), tuple(s["utm_b"]), georef_out, crs=args.crs)
        if sr.recommendation == "ACCEPT_SEEDED":
            by_sn[sn].recommendation = "ACCEPT_SEEDED"
            by_sn[sn].output_file = sr.output_file
            n += 1
        log.info("seed %s -> %s", sn, sr.recommendation)
    log.info("seeds: %d plot(s) placed by operator (ACCEPT_SEEDED, 0-FP)", n)


def _summary(args, m1_dir, ok, total, proper, results):
    from landintel.llm.memory_graph import default_graph
    print(f"\nSPAN {args.village} DONE: M1 {ok}/{total} ok ({proper} PROPER) -> {m1_dir}")
    print(f"  memory: {default_graph().stats()['path']}")
    if results is not None:
        disp: dict[str, int] = {}
        for r in results:
            disp[r.recommendation] = disp.get(r.recommendation, 0) + 1
        conf = sum(v for k, v in disp.items() if k.startswith("ACCEPT"))
        print(f"  M2: {conf}/{len(results)} confident (0 FP) {disp} -> {args.out}/georef")


if __name__ == "__main__":
    main()
