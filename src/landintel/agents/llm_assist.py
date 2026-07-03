"""LLMAssistAgent -- LLM for the FUZZY tasks only, provider-agnostic, fail-safe.

THE COMBO (per product design): a LOCAL open-source LLM handles the cheap/offline bulk
(audit narration, OCR-disambiguation hints), and Anthropic Claude ESCALATES for the hard
cases or when no local model is deployed. The deterministic core decides everything that
affects correctness; this agent's output is NON-load-bearing (a plain-language audit note
+ optional hints re-checked by the math gates). THE HARD RULE: nothing here can promote a
plot to ACCEPT, so no model (local or Claude, however wrong) can create a false positive.

STAGE-AGNOSTIC: it normalizes the input to ``PlotDisposition`` first, so it DIAGNOSES +
PROPOSES + RE-GATES the REVIEW/NO_COVERAGE plots of the new M2 (``list[ClubResult]``) and
the surveyor-matching M3 (``list[GeorefResult]``) identically. It only ever touches
non-confident plots; a confident plot is never re-reasoned (and so never demoted/promoted).

Provider chain (env ``LANDINTEL_LLM_ORDER``, default "local" = offline-only; cloud is opt-in):
  * local  -- Qwen (default) or any open-source model (vLLM/LM Studio/Ollama).
  * claude -- Anthropic API (``ANTHROPIC_API_KEY``), model ``claude-opus-4-8``.
  * manus  -- a Manus (or any) OpenAI-compatible API.
If every provider is unavailable/errors -> deterministic fallback narration; the product
never breaks (or changes a placement) when the LLM is down.
"""

from __future__ import annotations

from .base import Agent, AgentReport, Check, Proposal, Severity
from .dispositions import normalize
# The LLM engine now lives in landintel.llm (Qwen-first provider chain) -- imported here so
# the agent layer uses the one shared brain. llm_call re-exported for back-compat.
from ..llm.concept import (AUTO_ACTIONS, SYSTEM_CONCEPT, parse_proposal,
                           propose_prompt, rule_based_proposal)
from ..llm.providers import llm_call  # noqa: F401


def _evidence(d) -> dict:
    """Stage-agnostic evidence bundle (numbers only) the model reasons over for one plot.

    Built from the unified ``PlotDisposition`` so the same diagnoser serves both M2 and
    M3. Mirrors ``llm.concept.plot_evidence`` but does not require a stage-specific result
    object -- the cadastral path surfaces area_ratio, the geometric path chain_coverage.
    """
    method = d.method or ""
    ev = {
        "survey_number": d.survey,
        "disposition": d.recommendation,
        "match_method": method,
        "cad_residual_m": (None if d.cad_residual in (None, float("inf"))
                           else round(d.cad_residual, 1)),
        "error": d.error or d.note or "",
        "m1_file": d.m1_file,
    }
    if d.area_ratio == d.area_ratio:               # not NaN
        ev["area_ratio"] = round(d.area_ratio, 2)
    if d.chain_coverage == d.chain_coverage:       # not NaN
        ev["chain_coverage"] = round(d.chain_coverage, 2)
    return ev


class LLMAssistAgent(Agent):
    name = "llm_assist"

    def run(self, results, context: dict) -> AgentReport:
        rep = AgentReport(agent=self.name)
        disps = normalize(results)
        village = (context or {}).get("village")
        graph = (context or {}).get("memory_graph")

        disp_counts: dict[str, int] = {}
        for d in disps:
            disp_counts[d.recommendation] = disp_counts.get(d.recommendation, 0) + 1
        facts = "; ".join(f"{k}={v}" for k, v in sorted(disp_counts.items()))
        review = [(d.survey, d.method, d.error or d.note or "")
                  for d in disps if not d.is_confident]

        # ---- 1) REASON: diagnose every unplaced plot + propose ONE bounded fix ----
        # The model carries the whole concept (SYSTEM_CONCEPT) and reasons over the math
        # evidence; its action is clamped to SAFE_ACTIONS and re-gated downstream, so it
        # can raise recall but never create a false positive. CONFIDENT plots are skipped
        # entirely -> the agent can never turn a non-ACCEPT into ACCEPT here.
        llm_used = n_recalled = 0
        for d in disps:
            if d.is_confident:
                continue
            ev = _evidence(d)
            if graph is not None:
                try:
                    past = graph.recall(d.survey, village=village or "INGUR")
                    if past.get("known"):
                        ev["memory"] = past
                        n_recalled += 1
                except Exception:  # noqa: BLE001
                    pass
            xv = _is_cross_village(d, village)
            prop, source = None, "rule"
            if not xv:  # cross-village needs no reasoning; it is a pure data ask
                out = llm_call(SYSTEM_CONCEPT + "\n\n" + propose_prompt(ev), max_tokens=300)
                if out is not None:
                    parsed = parse_proposal(out[0], ev)
                    if parsed is not None:
                        prop, source, llm_used = parsed, f"llm:{out[1]}", True
            if prop is None:
                prop = rule_based_proposal(ev, cross_village=xv)
            rep.proposals.append(Proposal(
                survey_number=ev["survey_number"],
                hypothesis=prop["hypothesis"], action=prop["action"],
                rationale=prop.get("rationale", ""), source=source,
                is_auto=prop["action"] in AUTO_ACTIONS))

        n_auto = sum(1 for p in rep.proposals if p.is_auto)
        rep.checks.append(Check(
            "reasoning", Severity.INFO,
            f"diagnosed {len(rep.proposals)} unplaced plot(s); {n_auto} have a re-runnable "
            f"automation to try (re-gated), the rest need operator input. "
            f"recalled {n_recalled} from memory. "
            f"source={'llm+rule' if llm_used else 'rule (no LLM reachable)'}."))

        # ---- 2) NARRATE: plain-language audit note for the client ----
        lines = "\n".join(f"- {sn}: {m} {e}" for sn, m, e in review[:40])
        prompt = ("You are the audit-narration step of a land-survey georeferencing "
                  "pipeline. In 4-6 plain sentences for a non-technical client, summarise "
                  f"this job. Dispositions: {facts}. Plots needing review/input:\n{lines}\n"
                  "State clearly that no plot was placed incorrectly (the confident set is "
                  "gate-verified, 0 false positives) and that the review plots need the "
                  "listed extra input to finish. Do not invent numbers.")
        out = llm_call(prompt)
        if out is None:
            rep.checks.append(Check("llm_available", Severity.WARN,
                "no LLM provider reachable -> deterministic fallback; placements and "
                "proposals UNAFFECTED (LLM never decides; re-gate is math)."))
            rep.notes.append(self._fallback(facts, review))
        else:
            text, provider = out
            rep.checks.append(Check("llm_available", Severity.OK, f"narration via {provider}"))
            rep.notes.append(text)
        return rep

    @staticmethod
    def _fallback(facts: str, review) -> str:
        return (f"Job summary: {facts}. The confident plots are gate-verified (0 false "
                f"positives by construction). {len(review)} plot(s) need a small extra "
                f"input to finish (see input_requests.json); none were placed incorrectly.")


# Single-source cross-village check (see dispositions.is_cross_village).
from .dispositions import is_cross_village as _is_cross_village  # noqa: E402
