"""Consolidated per-plot DECISION TRACE -- one auditable record per plot.

The agent layer already writes verification_report.json, proposals.json and
input_requests.json, but an expert auditing a village has to cross-reference all three
by survey number to answer a single question: "why did plot X land where it did?" This
module merges them into ONE per-plot view -- the final disposition, the method + the
numeric evidence the math gate actually saw (residual / coverage / area ratio / inliers),
whether it is a confident deliverable, any DEMOTE reason, any automation that was
re-gated (with the gate's verdict), and any operator input requested.

It is PURE AGGREGATION of data the agents already computed. It makes NO decision and
touches NO placement, so it cannot affect the 0-FP guarantee -- it only makes the
existing, gate-made decisions auditable at a glance. Stage-agnostic: built from the
normalized ``PlotDisposition``, so it reads M2 (``ClubResult``) and M3 (``GeorefResult``)
identically.
"""

from __future__ import annotations

import math

from .dispositions import normalize


def _finite(x):
    """JSON-safe scalar: NaN/inf mean 'not applicable to this plot' -> drop to None."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def build_decision_trace(results, proposals=None, requests=None, village=None) -> dict:
    """One consolidated audit record per plot, keyed by survey number.

    ``results`` is the FINAL results list (after any DEMOTE / re-gate), so the trace
    reflects the shipped state. ``proposals`` / ``requests`` are the same objects the
    orchestrator already gathered. Returns a dict with a per-plot list ordered
    confident-first, then by survey number. Read-only.
    """
    disps = normalize(results)

    prop_by_sn: dict[str, list] = {}
    for p in proposals or []:
        prop_by_sn.setdefault(str(p.survey_number), []).append({
            "action": getattr(p, "action", ""),
            "hypothesis": getattr(p, "hypothesis", ""),
            "accepted_by_gate": getattr(p, "accepted_by_gate", None),
            "note": getattr(p, "note", ""),
        })

    req_by_sn: dict[str, dict] = {}
    for q in requests or []:
        qd = q if isinstance(q, dict) else q.to_dict()
        req_by_sn[str(qd.get("survey_number", "?"))] = {
            "input_type": qd.get("input_type"),
            "reason": qd.get("reason"),
        }

    rows = []
    for d in disps:
        sn = str(d.survey)
        # The numeric evidence the gate saw -- keep only the fields that APPLY (finite),
        # so a cadastral plot shows area_ratio and a surveyor plot shows chain_coverage
        # without either carrying meaningless placeholders.
        evidence = {k: _finite(v) for k, v in {
            "confidence": d.confidence,
            "cad_residual_m": d.cad_residual,
            "chain_coverage": d.chain_coverage,
            "area_ratio": d.area_ratio,
            "scale": d.scale,
            "n_inliers": d.n_inliers,
            "n_corners": d.n_corners,
            "verify_passed": d.verify_passed,
        }.items()}
        evidence = {k: v for k, v in evidence.items() if v is not None}

        rows.append({
            "survey_number": sn,
            "village": village,
            "disposition": d.recommendation,
            "confident": d.is_confident,
            "method": d.method,
            "stage": d.source,
            "has_geometry": d.has_geometry,
            "output_file": d.output_file,
            "evidence": evidence,
            "note": d.note,
            "proposals": prop_by_sn.get(sn, []),
            "input_request": req_by_sn.get(sn),
        })

    rows.sort(key=lambda r: (
        not r["confident"],
        int(r["survey_number"]) if r["survey_number"].isdigit() else 1_000_000_000,
        r["survey_number"]))

    n_conf = sum(1 for r in rows if r["confident"])
    return {
        "summary": (f"{n_conf}/{len(rows)} confident (ACCEPT); per-plot audit trail below "
                    f"(evidence = what the math gate saw; agents never place a plot)"),
        "village": village,
        "n_confident": n_conf,
        "n_plots": len(rows),
        "plots": rows,
    }
