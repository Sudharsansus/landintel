"""M2 club via PUBLIC CADASTRAL TILES -- NO surveyor file needed.

Georeferences + clubs a village's M1 FMB DXFs by matching each survey number onto its
parcel in the public mypropertyqr cadastral tileset (yellow parcels + orange survey
labels), then tiling the plots. Location comes from a geocoded (or supplied) village
centre; the tiles are fetched for a bbox around it and OCR'd to find the village's own
survey numbers, so no surveyor raw-data file is required.

Usage:
  python run_m2_cad.py <VILLAGE> --lat <LAT> --lon <LON> [--radius-km 2.5]
  python run_m2_cad.py <VILLAGE> --utm <X> <Y> [--radius-km 2.5]     (centre in UTM 43N)

Outputs to output/<VILLAGE>/m2/: clubbed_village.dxf, clubbed_points.csv, clubbed.geojson,
clubbed_qa.png, clubbed.verify.txt.  CRS = EPSG:32643 (UTM 43N) for all Erode villages.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

from pyproj import Transformer

from landintel.pipeline.m2_club import club_pipeline
from landintel.pipeline.m2_club.club_output import snap_and_rewrite
from landintel.pipeline.m2_club.qa_render import render_club_qa
from landintel.pipeline.m5_cadastral.s3_tiles import S3CadastralSource

CRS = "EPSG:32643"           # UTM 43N -- all these Erode villages are west of 78E


def _args() -> tuple[str, float, float, float]:
    a = sys.argv[1:]
    village = a[0]
    radius_km = 2.5
    if "--radius-km" in a:
        radius_km = float(a[a.index("--radius-km") + 1])
    if "--utm" in a:
        i = a.index("--utm")
        cx, cy = float(a[i + 1]), float(a[i + 2])
    elif "--lat" in a and "--lon" in a:
        lat = float(a[a.index("--lat") + 1]); lon = float(a[a.index("--lon") + 1])
        cx, cy = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(lon, lat)
    else:
        raise SystemExit("need --lat/--lon or --utm centre")
    return village, cx, cy, radius_km * 1000.0


def main() -> None:
    village, cx, cy, R = _args()
    m1_dir = Path(f"output/{village}/m1")
    out = Path(f"output/{village}/m2")
    out.mkdir(parents=True, exist_ok=True)

    m1_paths = sorted(str(p) for p in m1_dir.glob(f"{village}_*.dxf"))
    surveys = {m.group(1) for p in m1_paths if (m := re.search(r"_(\d+)\.dxf$", p))}
    bbox = (cx - R, cy - R, cx + R, cy + R)
    print(f"[1/4] {village}: {len(m1_paths)} M1 FMBs, surveys {sorted(surveys, key=int)}")
    print(f"[2/4] centre UTM43 ({cx:.0f},{cy:.0f}) +/- {R/1000:.1f} km -> bbox "
          f"{tuple(round(b) for b in bbox)}")

    print("[3/4] fetching + OCR-ing public cadastral tiles (mypropertyqr, no auth)...")
    cad = S3CadastralSource(bbox, surveys, cache_dir=f"input/{village}/s3_tiles", crs=CRS)

    results = club_pipeline(m1_paths, out, crs=CRS, cadastral_source=cad)
    print("[4/4] quality pass: edge-align + corner-snap (0-FP)...")
    snap_and_rewrite(results, out, crs=CRS, enable=True, tol=8.0)

    counts = Counter(r.recommendation for r in results)
    placed = [r for r in results if r.placed]
    print("\n================ DISPOSITIONS ================")
    print(dict(counts))
    for r in sorted(results, key=lambda r: int(r.survey_number) if r.survey_number.isdigit() else 0):
        print(f"  {r.survey_number:>6}  {r.recommendation:<14} {r.method or '-'}")
    print(f"\nPLACED: {len(placed)}/{len(results)} onto cadastral parcels")

    try:
        render_club_qa(results, out / "clubbed_qa.png", cadastral_source=cad, crs=CRS)
    except Exception as exc:  # noqa: BLE001
        print(f"(qa render skipped: {exc})")
    print(f"Clubbed DXF : {out / 'clubbed_village.dxf'}")
    print(f"Points CSV  : {out / 'clubbed_points.csv'}")


if __name__ == "__main__":
    main()
