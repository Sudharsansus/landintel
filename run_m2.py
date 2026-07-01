"""M2 club: georeference + club all M1 FMB DXFs into ONE village DXF -- NO surveyor file.

Fresh, clean I/O:
    output/<VILLAGE>/m1/*.dxf   (M1 output)  ->  output/<VILLAGE>/m2/clubbed_village.dxf
Placement uses cadastre (TNGIS + S3) / GPS / relative-club ONLY. The surveyor RAW DATA file
is read ONLY for (a) the tile-harvest bbox + village fence and (b) VERIFY-ONLY stone-error QA
-- never to place a plot (that is M3). Usage:  python run_m2.py [VILLAGE]   (default INGUR)
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from landintel.core.geo import village_fence
from landintel.pipeline.m2_club import club_pipeline
from landintel.pipeline.m2_club.club_output import snap_and_rewrite
from landintel.pipeline.m2_club.qa_render import render_club_qa
from landintel.pipeline.m2_club.stone_verify import verify_stones
from landintel.pipeline.m2_georef.extract_surveyor import extract_surveyor
from landintel.pipeline.m5_cadastral.composite import CompositeCadastralSource
from landintel.pipeline.m5_cadastral.s3_tiles import S3CadastralSource
from landintel.pipeline.m5_cadastral.tngis_tiles import TNGIS_ZOOM, TngisTileCadastralSource

VILLAGE = sys.argv[1] if len(sys.argv) > 1 else "INGUR"
IN = Path(f"input/{VILLAGE}")
M1_DIR = Path(f"output/{VILLAGE}/m1")
OUT = Path(f"output/{VILLAGE}/m2")
OUT.mkdir(parents=True, exist_ok=True)
CRS = "EPSG:32643"
SURVEYOR = IN / "INGUR RAW DATA FILE.dxf"
TNGIS_CACHE = IN / "tngis_cache_ingur_wide"
S3_CACHE = IN / "s3_tiles"

# village FMBs only (a stray cross-village survey would just clutter; the fence drops it too)
m1_paths = sorted(str(p) for p in M1_DIR.glob("*.dxf") if "KANDAMPALAYAM" not in p.name.upper())
surveys = [m.group(1) for p in m1_paths if (m := re.search(r"[_ ](\d+)\.dxf$", p))]
print(f"[1/5] {len(m1_paths)} {VILLAGE} M1 FMBs; surveys {sorted(set(surveys), key=int)}")

# bbox + fence from the surveyor extent (harvest bounds / label fence -- NOT placement)
surv = extract_surveyor(str(SURVEYOR))
xs = [s.x for s in surv.stones]; ys = [s.y for s in surv.stones]
bbox = (min(xs), min(ys), max(xs), max(ys))
# Concave (alpha-shape) fence: snug around a band-shaped village -> drops far cross-village
# labels better than a convex hull, at no recall cost (every stone stays inside). scipy+shapely.
fence = village_fence([(s.x, s.y) for s in surv.stones], buffer=300.0, concave=True)
print(f"[2/5] bbox {tuple(round(b,1) for b in bbox)}  ({len(list(TNGIS_CACHE.glob(f'{TNGIS_ZOOM}_*.png')))} tngis tiles)")

print("[3/5] cadastre: TNGIS primary + S3 fallback (yellow-net reconstruction + fenced OCR)...")
cad_tngis = TngisTileCadastralSource(surveys, bbox, crs=CRS, cache_dir=str(TNGIS_CACHE),
                                     ocr=True, village_fence=fence)
cad_s3 = S3CadastralSource(bbox, set(surveys), cache_dir=str(S3_CACHE), crs=CRS,
                           village_fence=fence)
cad = CompositeCadastralSource(cad_tngis, cad_s3)

print("[4/5] clubbing (cadastral seat @ TRUE scale + relative-club) -- NO surveyor placement...")
results = club_pipeline(m1_paths, OUT, crs=CRS, cadastral_source=cad)

print("[5/5] quality pass: edge-align + corner-snap (size-relative, 0-FP)...")
snap = snap_and_rewrite(results, OUT, crs=CRS, enable=True, tol=5.0)   # NO truth_stones -> surveyor-free
print(f"      edge-align {snap.n_edge_constraints} constraints / {snap.n_edge_moved} moved; "
      f"corner-snap {snap.n_corners_snapped} corners")

counts = Counter(r.recommendation for r in results)
print("\n================ DISPOSITIONS ================")
print(dict(counts))
for r in sorted(results, key=lambda r: int(r.survey_number) if r.survey_number.isdigit() else 0):
    corro = f" corro={r.corroborated_by}" if r.corroborated_by else ""
    print(f"  {r.survey_number:>5}  {r.recommendation:<12} {r.method or '-':<20}{corro}")

# VERIFY-ONLY stone-error QA vs surveyor stones (measures accuracy; never used to place)
corners = defaultdict(list)
with open(OUT / "clubbed_points.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        corners[row["survey_number"]].append((float(row["x_utm"]), float(row["y_utm"])))
rep = verify_stones({s: np.array(v) for s, v in corners.items()},
                    np.array([(s.x, s.y) for s in surv.stones], float), tol=2.5)
(OUT / "stone_verify.txt").write_text(rep.to_text(), encoding="utf-8")
print("\n" + rep.to_text().split(chr(10) + "  --")[0])

render_club_qa(results, OUT / "clubbed_qa.png", cadastral_source=cad, crs=CRS)
print(f"\nClubbed DXF : {OUT / 'clubbed_village.dxf'}")
print(f"Points CSV  : {OUT / 'clubbed_points.csv'}")
print(f"Stone QA    : {OUT / 'stone_verify.txt'}")
