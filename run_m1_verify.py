"""M1 VERIFICATION -- run the agents across a whole village and surface every flag.

This is the standing verification the pipeline runs so problems are caught AUTOMATICALLY
(not by eyeballing plots one at a time):
  * NUMERIC agent (verify_dxf): stones/closure/area + NEW topology checks
       - stones_are_boundary_vertices  (the "single line through a stone" bug)
       - subdivisions_attached         (subdivision lines stopping short of the boundary)
       - stone_numbers_read            (OCR '?' recall on stone numbers)
  * VISUAL agent (M_visual_agent, ezdxf + matplotlib): FMB-vs-DXF images for the eye.

Prints one line per plot with any WARN/FAIL, a village summary, and writes the render
images to output/<VILLAGE>/m1_qa/. Meant to be read by a human (or Claude) LAST -- the
agents flag, the reviewer decides. Usage: python run_m1_verify.py [VILLAGE] [--no-visual]
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

from landintel.pipeline.m1_extract.verify_dxf import verify_m1_dxf

VILLAGE = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "INGUR"
DO_VISUAL = "--no-visual" not in sys.argv
M1_DIR = Path(f"output/{VILLAGE}/m1")

dxfs = sorted(M1_DIR.glob(f"{VILLAGE}_*.dxf"))
print(f"=== M1 VERIFICATION: {VILLAGE} ({len(dxfs)} plots) ===\n")

flag_counts: Counter = Counter()
improper = []
per_plot = []
for dxf in dxfs:
    try:
        r = verify_m1_dxf(dxf)
    except Exception as exc:  # noqa: BLE001
        print(f"  {dxf.name}: VERIFY ERROR {exc}")
        continue
    warns = [c for c in r.checks if not c.passed and c.severity == "warn"]
    fails = [c for c in r.checks if not c.passed and c.severity == "fail"]
    survey = dxf.stem.split("_")[-1]
    for c in warns + fails:
        flag_counts[c.name] += 1
    if not r.proper:
        improper.append(survey)
    per_plot.append((survey, r.proper, [c.name for c in fails], [c.name for c in warns]))

# per-plot lines: only show plots with a FAIL or a topology/attach WARN (the actionable ones)
ACTIONABLE = {"stones_are_boundary_vertices", "subdivisions_attached",
              "boundary_closed", "no_duplicate_boundary", "area_vs_stated"}
print(f"{'survey':>8}  {'PROPER':>7}  flags")
print("-" * 60)
for survey, proper, fails, warns in per_plot:
    hot = [f for f in fails] + [w for w in warns if w in ACTIONABLE]
    if hot:
        print(f"{survey:>8}  {str(proper):>7}  {', '.join(hot)}")

print("\n=== FLAG TOTALS (plots affected) ===")
for name, n in flag_counts.most_common():
    print(f"  {name:32} {n:>3}/{len(dxfs)}")
print(f"\nPROPER: {len(dxfs) - len(improper)}/{len(dxfs)}"
      + (f"  |  IMPROPER: {improper}" if improper else ""))

if DO_VISUAL:
    try:
        from landintel.m_agents import M_visual_agent
        agent = M_visual_agent(VILLAGE)
        rep = agent.qa_village(fmb_dir=f"input/{VILLAGE}/fmb", m1_dir=str(M1_DIR),
                               out_dir=f"output/{VILLAGE}/m1_qa")
        print(f"\n[VISUAL agent] {len(rep.items)} FMB-vs-DXF images -> output/{VILLAGE}/m1_qa/")
    except Exception as exc:  # noqa: BLE001
        print(f"\n[VISUAL agent] skipped: {exc}")
