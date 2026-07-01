"""FP-safe TOOL SURFACE -- the gateway any agent harness drives our engines through.

This is the "another Claude Code wired to the OCR engine + maths engine" surface. It exposes
LandIntel's real engines (OCR = m1_extract.ocr; maths/identity = the cadastral source + the
rigid-fit gates + regate) as a small set of harness-agnostic tools. ANY agent loop -- our own
harness, the LLMAssistAgent, opengeos/GeoAgent (Strands), or a Claude Code session -- can call
these tools.

THE HARD RULE, enforced here so no harness can break it: every tool is either
  * READ-ONLY (ocr_read / diagnose / cadastral_identity / list_requests / recall), or
  * PROPOSE-ONLY (propose_fix -> runs the deterministic re-gate; THE GATE decides).
NO tool sets a placement to ACCEPT, emits geometry, or runs arbitrary code. So an autonomous
LLM driving these tools can raise recall and explain itself, but can NEVER create a false
positive -- the worst it can do is propose a fix the gate then rejects.

Tools are bound to one job's ``results`` + ``context`` via a closure (the live objects stay
out of the LLM-visible parameters), so the only thing the model passes is a survey number and
a whitelisted action string.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from .concept import AUTO_ACTIONS, SAFE_ACTIONS, plot_evidence, rule_based_proposal

_log = logging.getLogger(__name__)
_CONF = ("ACCEPT", "ACCEPT_SEEDED", "ACCEPT_CADASTRAL")


@dataclass
class Tool:
    """A harness-agnostic tool description (name + doc + safety class + callable)."""
    name: str
    description: str
    safety: str            # "read_only" | "proposes" (never "decides" -- no such tool exists)
    fn: Callable[..., dict]


def build_tools(results, context: dict | None = None) -> list[Tool]:
    """Return the FP-safe tool surface bound to one job's results + context."""
    context = context or {}
    by_sn = {r.survey_number: r for r in results}
    village = context.get("village", "INGUR")
    cad = context.get("cadastral_source")
    graph = context.get("memory_graph")

    # ---------------------------------------------------------------- read-only ----
    def diagnose(survey_number: str) -> dict:
        """Return the MATH evidence + a root-cause hypothesis for one plot (read-only).
        Use this to understand WHY a plot is not confidently placed."""
        r = by_sn.get(survey_number)
        if r is None:
            return {"error": f"unknown survey_number {survey_number}"}
        from ..pipeline.m2_georef.pipeline import _is_cross_village
        ev = plot_evidence(r)
        ev["confident"] = r.recommendation in _CONF
        if not ev["confident"]:
            ev["hypothesis"] = rule_based_proposal(
                ev, cross_village=_is_cross_village(r.m1_file, village))
        return ev

    def cadastral_identity(survey_number: str) -> dict:
        """Read the cadastral reference (position + rough size + identity) for a survey
        number from the S3 tiles. READ-ONLY reference -- it does NOT place the plot."""
        if cad is None:
            return {"available": False, "note": "no cadastral source in this context"}
        try:
            pt = getattr(cad, "label_point", lambda s: None)(survey_number)
            parcel = getattr(cad, "get", lambda s: None)(survey_number)
            area = getattr(parcel, "area_m2", None) if parcel is not None else None
            return {"available": parcel is not None, "label_point_utm": list(pt) if pt else None,
                    "parcel_area_m2": area}
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "error": str(exc)}

    def ocr_read(survey_number: str) -> dict:
        """Re-read this plot's label/dimensions with the OCR engine (m1_extract.ocr),
        READ-ONLY. Returns raw OCR strings (normalization is downstream). Only available
        if the OCR resources for this plot are wired into the context."""
        reader = context.get("ocr_read")          # callable(sn) -> dict, supplied by pipeline
        if reader is None:
            return {"available": False,
                    "note": "OCR engine not wired into this context (no page image bound). "
                            "Pipeline binds context['ocr_read'] when a source PDF is present."}
        try:
            return {"available": True, **reader(survey_number)}
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "error": str(exc)}

    def recall(survey_number: str) -> dict:
        """Recall what PAST sessions learned about this survey number / village from the
        memory graph (prior dispositions, proposals, operator inputs). READ-ONLY."""
        if graph is None:
            return {"available": False, "note": "no memory graph in this context"}
        try:
            return {"available": True, **graph.recall(survey_number, village=village)}
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "error": str(exc)}

    def list_requests() -> dict:
        """List every plot not yet confidently placed and the minimal input each needs."""
        from ..agents.input_request import InputRequestAgent
        rep = InputRequestAgent().run(results, context)
        return {"n_requests": len(rep.requests),
                "requests": [q.to_dict() for q in rep.requests]}

    # ---------------------------------------------------------------- propose-only --
    def propose_fix(survey_number: str, action: str) -> dict:
        """Propose ONE bounded fix for a plot and RE-GATE it. ``action`` MUST be a key of
        SAFE_ACTIONS. AUTO actions re-run the deterministic fix and THE GATE decides (a
        global FP guard reverts anything that would break the tiling); INPUT actions just
        record the operator ask. This tool can NEVER place a plot itself."""
        if action not in SAFE_ACTIONS:
            return {"ok": False, "error": f"action must be one of {sorted(SAFE_ACTIONS)}"}
        r = by_sn.get(survey_number)
        if r is None:
            return {"ok": False, "error": f"unknown survey_number {survey_number}"}
        from ..agents.base import Proposal
        from ..agents.regate import regate_proposals
        prop = Proposal(survey_number, hypothesis="(agent-proposed)", action=action,
                        source="tool", is_auto=action in AUTO_ACTIONS)
        summary = regate_proposals(results, [prop], context)
        return {"ok": True, "action": action, "is_auto": prop.is_auto,
                "accepted_by_gate": prop.accepted_by_gate, "regated": prop.regated,
                "disposition_now": r.recommendation, "note": prop.note, "regate": summary}

    tools = [
        Tool("diagnose", diagnose.__doc__, "read_only", diagnose),
        Tool("cadastral_identity", cadastral_identity.__doc__, "read_only", cadastral_identity),
        Tool("ocr_read", ocr_read.__doc__, "read_only", ocr_read),
        Tool("recall", recall.__doc__, "read_only", recall),
        Tool("list_requests", list_requests.__doc__, "read_only", list_requests),
        Tool("propose_fix", propose_fix.__doc__, "proposes", propose_fix),
    ]
    return tools
