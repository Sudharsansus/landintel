"""Human-like VISUAL QA of M1 output: render FMB PDF vs M1 DXF for every plot.

Produces output/<VILLAGE>/m1_qa/<stem>.png (FMB drawing | extracted DXF) for visual review
-- the check the numeric gates cannot make. Usage: python run_m1_qa.py [VILLAGE]  (default INGUR)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

from landintel.m_agents import M_visual_agent

VILLAGE = sys.argv[1] if len(sys.argv) > 1 else "INGUR"
agent = M_visual_agent(VILLAGE)
report = agent.qa_village(
    fmb_dir=f"input/{VILLAGE}/fmb",
    m1_dir=f"output/{VILLAGE}/m1",
    out_dir=f"output/{VILLAGE}/m1_qa",
)
print(f"[M1 visual QA] {len(report.items)} comparison images -> output/{VILLAGE}/m1_qa/")
for it in report.items:
    print(f"  {it.survey:>6}  stones={it.n_stones:<3} closed={it.is_closed}  {Path(it.compare_png).name}")
