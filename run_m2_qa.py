"""M2 QA summary -- one shot. Reads the M2 club outputs and prints the quality verdict:
  * dispositions (ACCEPT / ACCEPT_SEEDED / REVIEW / NO_COVERAGE) -- coverage
  * clubbed.verify.txt 0-FP gates (all_passed?)  -- false-positive discipline
  * stone_verify.txt accuracy vs the 522 surveyor stones (median-of-medians, MATCH, SHAPE_OK)
Usage:  python run_m2_qa.py [VILLAGE]   (default INGUR)
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

VILLAGE = sys.argv[1] if len(sys.argv) > 1 else "INGUR"
M2 = Path(f"output/{VILLAGE}/m2")

print(f"=== M2 QA: {VILLAGE}  ({M2}) ===\n")

# 1) coverage from clubbed_points.csv / a dispositions file if present
pts = M2 / "clubbed_points.csv"
if pts.exists():
    surveys = set()
    with pts.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            surveys.add(row.get("survey_number", "?"))
    print(f"[COVERAGE] {len(surveys)} surveys have clubbed points: {sorted(surveys, key=lambda s: int(s) if s.isdigit() else 0)}")

# 2) 0-FP gates
vf = M2 / "clubbed.verify.txt"
if vf.exists():
    txt = vf.read_text(encoding="utf-8", errors="replace")
    passed = "PASS" if ("all_passed=True" in txt or "ALL PASS" in txt or "PASS" in txt.splitlines()[0]) else "?"
    print(f"\n[0-FP GATES] clubbed.verify.txt:")
    for line in txt.splitlines()[:25]:
        print("   " + line)
else:
    print("\n[0-FP GATES] no clubbed.verify.txt")

# 3) accuracy vs surveyor
sv = M2 / "stone_verify.txt"
if sv.exists():
    txt = sv.read_text(encoding="utf-8", errors="replace")
    print(f"\n[ACCURACY vs SURVEYOR] stone_verify.txt (head):")
    for line in txt.splitlines()[:6]:
        print("   " + line)
else:
    print("\n[ACCURACY] no stone_verify.txt")

print(f"\nArtifacts: {sorted(p.name for p in M2.glob('*'))}")
