"""m_agents -- the LandIntel agent layer for HUMAN-LIKE verification.

These agents check the pipeline the way a human reviewer would, not just numerically:
they RENDER the artefact and compare it to the source. The headline member is
``M_visual_agent`` (visual QA of M1: the extracted DXF vs the original FMB drawing),
built because numeric checks (closure, area) can pass while the drawing is visibly wrong
(a false positive a person would catch at a glance).
"""
from .visual_agent import M_visual_agent, VisualQAItem, render_m1_vs_fmb

__all__ = ["M_visual_agent", "VisualQAItem", "render_m1_vs_fmb"]
