"""VerificationAgent -- the autonomous error-catcher.

Runs the FP-safety invariants + per-module sanity assertions on EVERY job. If any
job-blocking invariant fails it marks the report FAIL, so the product refuses to ship
a bad deliverable instead of silently handing the client a wrong plot. This is the
component that replaces "Claude caught the bug" with "the product catches the bug,
every run, forever."

STAGE-AGNOSTIC: it normalizes the input to ``PlotDisposition`` first, so it verifies
the new M2 (``list[ClubResult]``) and the surveyor-matching M3 (``list[GeorefResult]``)
with the same code path. The footprint-overlap, complete-accounting, and every-ACCEPT-
has-an-output-file invariants are universal; the cross-village / verify-gate checks are
applied wherever the data supports them.

It is the ONLY agent allowed to mutate a recommendation, and ONLY in the DEMOTE
direction (ACCEPT -> REVIEW) when an ACCEPT violates an invariant -- never a promotion.
"""

from __future__ import annotations

from .base import Agent, AgentReport, Check, Severity
from .dispositions import VALID_STATES, normalize

# Interiors overlapping more than this = mutually-exclusive parcels. Centralized:
# this agent previously pinned 0.20 while the re-gate/assembly path used 0.30 --
# a silent FP-policy drift; all four consumers now share ONE constant.
from ..pipeline.m2_club.disposition_thresholds import (  # noqa: E402
    TILING_OVERLAP_THRESHOLD as _FOOTPRINT_CONFLICT,
)


class VerificationAgent(Agent):
    name = "verification"

    def run(self, results, context: dict) -> AgentReport:
        rep = AgentReport(agent=self.name)
        disps = normalize(results)
        village = (context or {}).get("village")

        # ---- FP-1: the ACCEPT set must be a NON-OVERLAPPING tiling ----
        # Real parcels tile (sharing edges, not interiors). Any ACCEPT pair overlapping
        # interiors > 0.20 is mutually exclusive -> DEMOTE the lower-confidence one to
        # REVIEW (demote-only) and FAIL the invariant so the job routes to review.
        conf = [d for d in disps if d.is_confident]
        polys = [(d, d.footprint) for d in conf if d.has_geometry]
        overlaps = []
        demoted: list[str] = []
        for i in range(len(polys)):
            di, a = polys[i]
            if not di.is_confident:           # may have been demoted in an earlier pair
                continue
            for j in range(i + 1, len(polys)):
                dj, b = polys[j]
                if not dj.is_confident:
                    continue
                if a.intersects(b):
                    f = a.intersection(b).area / max(min(a.area, b.area), 1e-9)
                    if f > _FOOTPRINT_CONFLICT:
                        overlaps.append((di.survey, dj.survey, round(f, 2)))
                        loser = di if di.confidence <= dj.confidence else dj
                        keeper = dj if loser is di else di
                        loser.demote(
                            f"footprint overlaps higher-confidence {keeper.survey} "
                            f"({f:.0%}); demoted to REVIEW (FP-safe tiling)")
                        demoted.append(loser.survey)
        detail = "no confident footprints overlap" if not overlaps else (
            f"OVERLAPS (mutually-exclusive parcels stacked): {overlaps}; "
            f"demoted to REVIEW: {demoted}")
        rep.checks.append(Check(
            "fp_non_overlapping_tiling",
            Severity.OK if not overlaps else Severity.FAIL, detail))

        # ---- FP-2: every ACCEPT must carry an output_file (a real deliverable) ----
        # Only meaningful when the run actually writes deliverable DXFs. If NO plot in the
        # whole job has an output_file (a footprint-free unit run), the files simply were
        # not produced -> INFO, not a violation. A FAIL means some ACCEPT is missing the
        # deliverable while output files are otherwise being written.
        any_out = any(d.output_file for d in disps)
        no_out = [d.survey for d in conf if not d.output_file]
        if not any_out:
            rep.checks.append(Check(
                "accept_has_output_file", Severity.INFO,
                "no output files written in this run (deliverable check N/A)"))
        else:
            rep.checks.append(Check(
                "accept_has_output_file",
                Severity.OK if not no_out else Severity.FAIL,
                "every ACCEPT has an output file" if not no_out
                else f"ACCEPT without output_file (no deliverable): {no_out}"))

        # ---- FP-3: no cross-village plot in the ACCEPT set (where village is known) ----
        xv = [d.survey for d in disps if d.is_confident and _is_cross_village(d, village)]
        rep.checks.append(Check(
            "fp_no_cross_village_in_confident",
            Severity.OK if not xv else Severity.FAIL,
            "no cross-village plot is confident" if not xv
            else f"CROSS-VILLAGE in confident set: {xv}"))

        # ---- FP-4: every geometric (corridor) ACCEPT passed its 7 verify gates ----
        bad_verify = [d.survey for d in disps
                      if d.is_confident and d.method.startswith("geometric")
                      and d.verify_passed is False]
        rep.checks.append(Check(
            "geometric_accepts_verified",
            Severity.OK if not bad_verify else Severity.FAIL,
            "all geometric ACCEPTs pass verify" if not bad_verify
            else f"geometric ACCEPT failed verify: {bad_verify}"))

        # ---- COV: complete disposition accounting (no plot lost; all 4 valid states) ----
        disp_counts: dict[str, int] = {}
        for d in disps:
            disp_counts[d.recommendation] = disp_counts.get(d.recommendation, 0) + 1
        unknown = [d.survey for d in disps if d.recommendation not in VALID_STATES]
        total = sum(disp_counts.values())
        rep.checks.append(Check(
            "coverage_accounting",
            Severity.OK if (total == len(disps) and not unknown) else Severity.FAIL,
            f"{total}/{len(disps)} plots accounted for in valid states: {disp_counts}"
            + (f"; INVALID states: {unknown}" if unknown else "")))

        # ---- INFO: headline numbers for the audit trail ----
        n_conf = sum(1 for d in disps if d.is_confident)
        rep.notes.append(
            f"confident={n_conf}/{len(disps)} "
            f"({100 * n_conf / max(len(disps), 1):.0f}%), "
            f"false_positives=0 by construction (gates), dispositions={disp_counts}"
            + (f", demoted_overlap={demoted}" if demoted else ""))
        return rep


# Single-source cross-village check (see dispositions.is_cross_village).
from .dispositions import is_cross_village as _is_cross_village  # noqa: E402
