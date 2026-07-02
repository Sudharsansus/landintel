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
    opt = {"radius_km": 2.5, "cx": None, "cy": None, "taluk": "", "district": ""}
    if "--radius-km" in a:
        opt["radius_km"] = float(a[a.index("--radius-km") + 1])
    if "--taluk" in a:
        opt["taluk"] = a[a.index("--taluk") + 1]
    if "--district" in a:
        opt["district"] = a[a.index("--district") + 1]
    if "--utm" in a:
        i = a.index("--utm"); opt["cx"], opt["cy"] = float(a[i + 1]), float(a[i + 2])
    elif "--lat" in a and "--lon" in a:
        lat = float(a[a.index("--lat") + 1]); lon = float(a[a.index("--lon") + 1])
        opt["cx"], opt["cy"] = Transformer.from_crs(
            "EPSG:4326", CRS, always_xy=True).transform(lon, lat)
    return village, opt


def _taluk_district(m1_paths: list[str], village: str) -> tuple[str, str]:
    """Parse taluk + district from an FMB filename (FMB_<DIST>_<TALUK>_<VILLAGE>_<n>)."""
    for p in m1_paths:
        m = re.search(r"FMB_([A-Za-z.]+)_([A-Za-z.]+)_", Path(p).name)
        if m:
            return m.group(2).strip("."), m.group(1).strip(".")
    return "", "Erode"     # sensible default for this delivery region


def main() -> None:
    village, opt = _args()
    m1_dir = Path(f"output/{village}/m1")
    out = Path(f"output/{village}/m2")
    out.mkdir(parents=True, exist_ok=True)

    m1_paths = sorted(str(p) for p in m1_dir.glob(f"{village}_*.dxf"))
    surveys = {m.group(1) for p in m1_paths if (m := re.search(r"_(\d+)\.dxf$", p))}
    cache_dir = f"input/{village}/s3_tiles"
    print(f"[1/4] {village}: {len(m1_paths)} M1 FMBs, surveys {sorted(surveys, key=int)}")

    if opt["cx"] is not None:                          # explicit centre supplied
        R = opt["radius_km"] * 1000.0
        bbox = (opt["cx"] - R, opt["cy"] - R, opt["cx"] + R, opt["cy"] + R)
        print(f"[2/4] centre UTM43 ({opt['cx']:.0f},{opt['cy']:.0f}) +/- {R/1000:.1f} km")
    else:                                              # AUTO-LOCATE (general, no hardcoding)
        from landintel.pipeline.m5_cadastral.geo_locate import locate_village
        taluk = opt["taluk"] or _taluk_district(m1_paths, village)[0]
        district = opt["district"] or _taluk_district(m1_paths, village)[1]
        print(f"[2/4] auto-locating {village} (taluk={taluk}, district={district}) by "
              f"geocode + Qwen + survey-number fingerprint...")
        bbox, info = locate_village(village, surveys, CRS, cache_dir,
                                    taluk=taluk, district=district)
        print(f"      anchor={info.get('anchor_level')} @ {info.get('anchor_latlon')} | "
              f"survey-hits={info.get('n_hits')} ({'tight' if info.get('tight') else 'wide'}) "
              f"bbox={tuple(round(b) for b in bbox)}")

    print("[3/4] fetching + OCR-ing public cadastral tiles (mypropertyqr, no auth)...")
    cad = S3CadastralSource(bbox, surveys, cache_dir=cache_dir, crs=CRS)

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

    # FINAL DELIVERABLE: village area-statement PDF + Excel + clubbed DXF, zipped.
    try:
        from landintel.pipeline.m4_report.village import build_village_delivery
        zip_path = build_village_delivery(results, out / "clubbed_village.dxf", out,
                                          village=village, crs=CRS)
        print(f"Deliverable : {zip_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"(delivery package skipped: {exc})")
    print(f"Clubbed DXF : {out / 'clubbed_village.dxf'}")
    print(f"Points CSV  : {out / 'clubbed_points.csv'}")


if __name__ == "__main__":
    main()
