"""List every deliverable file per village (run after the pipeline).  python list_deliverables.py [V ...]"""
import sys
from pathlib import Path

villages = sys.argv[1:] or ["MOOLAKARAI", "NASIYANUR", "KANDAMPALAYAM"]
for v in villages:
    m1, qa, m2 = Path(f"output/{v}/m1"), Path(f"output/{v}/m1_qa"), Path(f"output/{v}/m2")
    dxf = sorted(m1.glob(f"{v}_*.dxf"))
    proper = sum(1 for f in dxf if f.with_suffix(".verify.txt").exists()
                 and "STATUS: PROPER" in f.with_suffix(".verify.txt").read_text(
                     encoding="utf-8", errors="replace"))
    print(f"\n{'='*70}\n{v}\n{'='*70}")
    print(f"M1  ({m1}):")
    print(f"   {len(dxf)} DXF + {len(dxf)} .verify.txt  |  PROPER {proper}/{len(dxf)}")
    print(f"   _manifest.csv, numeric_qa.csv")
    print(f"M1 visual QA ({qa}):  {len(list(qa.glob('*.png')))} FMB-vs-DXF PNGs")
    print(f"M2 deliverables ({m2}):")
    for f in ["clubbed_village.dxf", "clubbed.geojson", "clubbed_points.csv",
              "clubbed_qa.png", "clubbed.verify.txt", "village_area_statement.pdf",
              "village_area_breakdown.xlsx", "village_delivery.zip"]:
        p = m2 / f
        print(f"   [{'x' if p.exists() else ' '}] {f}"
              + (f"  ({p.stat().st_size//1024} KB)" if p.exists() else ""))
    print(f"   + {len(list(m2.glob('georef_*.dxf')))} per-plot georef DXFs")
