"""Agent-layer orchestrator -- runs the 4 runtime agents on every job.

Order matters: the deterministic FP agents (Verification, Guard) run FIRST and decide
whether the job is shippable; InputRequest builds the path-to-100% worklist; LLMAssist
narrates last (and never affects correctness). Writes verification_report.json and
input_requests.json next to the deliverable, and returns an overall pass/fail so the
product can refuse to ship a job whose invariants failed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .base import write_json
from .guard import GuardAgent
from .input_request import InputRequestAgent
from .llm_assist import LLMAssistAgent
from .verification import VerificationAgent

_log = logging.getLogger(__name__)


def run_agent_layer(results, output_dir, context: dict | None = None) -> dict:
    """Run Verification + Guard + InputRequest + LLMAssist; write reports; return summary.

    ``shippable`` is False iff a deterministic FP invariant FAILED -- the product should
    then NOT deliver the confident set as-is (a wrong plot may be present), but route the
    whole job to human review. The agents never change a placement; they only verify,
    request input, and narrate.
    """
    output_dir = Path(output_dir)
    context = context or {}

    # MEMORY GRAPH: the pipeline remembers every session. Bind it into the context BEFORE the
    # agents run, so the reasoning agent can RECALL what past sessions learned about a plot.
    if context.get("memory_graph") is None and context.get("enable_memory", True):
        try:
            from ..llm.memory_graph import default_graph
            context = {**context, "memory_graph": default_graph()}
        except Exception as exc:  # noqa: BLE001 - memory must never break a job
            _log.error("Memory graph unavailable: %s", exc)
    graph = context.get("memory_graph")

    # One grep-able line per job so an operator can follow a plot through every agent.
    _log.info(
        "AGENTS BEGIN: village=%s n_plots=%d surveys=%s",
        context.get("village", "?"),
        len(results),
        sorted({str(getattr(r, "survey_number", getattr(r, "survey", "?")))
                for r in results})[:25]
        + (["..."] if len(results) > 25 else []))

    agents = [VerificationAgent(), GuardAgent(), InputRequestAgent(), LLMAssistAgent()]
    reports = []
    for a in agents:
        try:
            reports.append(a.run(results, context))
        except Exception as exc:  # noqa: BLE001 - an agent must never crash the job
            _log.error("Agent %s failed: %s", a.name, exc)

    # PROPOSE -> RE-GATE: the reasoning agent diagnosed each unplaced plot and proposed a
    # bounded automation to retry. Re-run the AUTO ones through the deterministic gate (the
    # LLM never decides); a global FP guard reverts anything that would break the tiling.
    proposals = [p for r in reports for p in r.proposals]
    regate_summary = {"accepted": 0, "rejected": 0, "deferred": 0}
    if proposals:
        try:
            from .regate import build_cadastral_reattempt, regate_proposals
            # Build the production re-attempt hook only when explicitly enabled AND a cadastral
            # source is available (re-running OCR placement is wasteful when the source is
            # unchanged within a batch; the operator-refresh loop sets enable_auto_regate).
            if (context.get("reattempt") is None and context.get("enable_auto_regate")
                    and context.get("cadastral_source") is not None):
                context = {**context, "reattempt": build_cadastral_reattempt(
                    context["cadastral_source"], context.get("surveyor"),
                    output_dir, context.get("crs", "EPSG:32643"))}
            regate_summary = regate_proposals(results, proposals, context)
        except Exception as exc:  # noqa: BLE001
            _log.error("Re-gate step failed: %s", exc)

    # If an auto-fix was gate-accepted, the confident set changed -> re-verify FP-safety on
    # the updated results so 'shippable' reflects the final state.
    if regate_summary.get("accepted"):
        try:
            reports = [r for r in reports if r.agent not in ("verification", "guard")]
            reports = [VerificationAgent().run(results, context),
                       GuardAgent().run(results, context)] + reports
        except Exception as exc:  # noqa: BLE001
            _log.error("Post-regate re-verify failed: %s", exc)

    fp_failed = any(r.failed for r in reports
                    if r.agent in ("verification", "guard"))
    requests = [q.to_dict() for r in reports for q in r.requests]

    write_json(output_dir / "verification_report.json", {
        "shippable": not fp_failed,
        "false_positive_safe": not fp_failed,
        "reports": [r.to_dict() for r in reports],
    })
    write_json(output_dir / "input_requests.json", {
        "summary": f"{len(requests)} plot(s) need input to reach 100%",
        "requests": requests,
    })
    write_json(output_dir / "proposals.json", {
        "summary": (f"reasoning diagnosed {len(proposals)} unplaced plot(s); re-gate: "
                    f"{regate_summary['accepted']} auto-fixed (gate-accepted), "
                    f"{regate_summary['rejected']} still need input, "
                    f"{regate_summary['deferred']} deferred (no re-attempt hook this run)."),
        "regate": regate_summary,
        "proposals": [p.to_dict() for p in proposals],
    })

    # Record this whole job into the persistent memory graph (every plot disposition, the
    # proposals tried + gate verdicts, and the input requests) so future sessions remember it.
    mem_stats = {}
    if graph is not None:
        try:
            sid = graph.record_job(results, context.get("village", "INGUR"),
                                   proposals=proposals, requests=requests)
            mem_stats = graph.stats()
            write_json(output_dir / "memory_report.json",
                       {"session": sid, "graph": mem_stats})
            _log.info("Memory graph updated (%s): %s", sid, mem_stats)
        except Exception as exc:  # noqa: BLE001
            _log.error("Memory record failed: %s", exc)

    # QGIS review layer: the operator opens this in QGIS (or any GIS) and seeds the
    # flagged plots. Geometry is deterministic (Shapely+pyproj) -- no LLM touches coords.
    try:
        from .geojson import write_review_geojson
        write_review_geojson(results, {q["survey_number"]: q for q in requests},
                             output_dir, context.get("crs", "EPSG:32643"))
    except Exception as exc:  # noqa: BLE001
        _log.error("QGIS review-layer export failed: %s", exc)

    _log.info("=" * 60)
    _log.info("AGENT LAYER: shippable=%s, %d input request(s) to reach 100%%",
              not fp_failed, len(requests))
    if proposals:
        _log.info("  REASON->RE-GATE: %d auto-fixed by gate, %d still need input, %d deferred",
                  regate_summary["accepted"], regate_summary["rejected"],
                  regate_summary["deferred"])
        for p in proposals:
            tag = ("GATE-ACCEPTED" if p.accepted_by_gate else
                   "deferred" if not p.regated else "needs-input")
            _log.info("    [%s] %s: %s -> %s", tag, p.survey_number, p.action, p.hypothesis)
    for r in reports:
        for c in r.checks:
            if c.severity.value in ("fail", "warn"):
                _log.warning("  [%s] %s/%s: %s", c.severity.value.upper(),
                             r.agent, c.name, c.detail)
        for note in r.notes:
            _log.info("  (%s) %s", r.agent, note)
    return {"shippable": not fp_failed, "n_requests": len(requests),
            "regate": regate_summary, "n_proposals": len(proposals),
            "memory": mem_stats, "reports": [r.to_dict() for r in reports]}
