"""InputRequestAgent -- turns "100% accuracy" into a process.

For every plot the math gates could NOT confidently place, this emits the ONE minimal
extra input that would close it -- never a guess. The client (who has no Claude) gets a
precise, ORDERED worklist (most-impactful first): "plot 668 needs a clearer parcel
polygon OR 2 corner GPS points;" feeding that input through the already-built,
identity-supplied paths (seed_place / a client cadastral file) makes the plot
ACCEPT_SEEDED -- FP-safe because the human supplies the identity. So the product reaches
100% by GATHERING the missing data, not by forcing a placement on data that cannot
support it.

STAGE-AGNOSTIC: works on normalized ``PlotDisposition`` so it builds the same worklist
for the new M2 (``list[ClubResult]``) and the surveyor-matching M3 (``list[GeorefResult]``).
"""

from __future__ import annotations

from .base import Agent, AgentReport, InputRequest, InputType, Severity, Check
from .dispositions import normalize

_SEED = "seed_place (operator gives 2 corner stone -> UTM points; identity is human-supplied, so 0-FP)"

# Cadastral fit bands (mirror the m2_georef gates; used only to phrase the right ask).
_CAD_AREA_LO, _CAD_AREA_HI, _CAD_ROT_RESID_MAX = 0.65, 1.55, 12.0

# Worklist ordering: most-impactful / easiest-to-resolve first. A plot that is LOCATED
# and just needs a human yes/no is one click from 100%; a cross-village plot needs a
# whole other reference and is the heaviest lift -> last.
_ORDER = {
    InputType.CONFIRM_PLACEMENT: 0,    # located, below the bar -> 1 human confirm
    InputType.CLEARER_PARCEL: 1,       # located, parcel outline unresolved
    InputType.TWO_CORNER_SEED: 2,      # no auto position -> 2 GPS points
    InputType.VILLAGE_REFERENCE: 3,    # different village -> a whole new reference
    InputType.NONE: 9,
}


class InputRequestAgent(Agent):
    name = "input_request"

    def run(self, results, context: dict) -> AgentReport:
        village = (context or {}).get("village")
        rep = AgentReport(agent=self.name)
        disps = normalize(results)

        for d in disps:
            if d.is_confident:
                continue
            sn = d.survey or "?"
            known = None
            if d.has_geometry:
                c = d.footprint.centroid
                known = (round(c.x, 1), round(c.y, 1))

            method = d.method
            cov = d.chain_coverage
            area = d.area_ratio
            note = d.note or d.error or ""

            if _is_cross_village(d, village):
                req = InputRequest(
                    sn, d.recommendation,
                    "plot belongs to a DIFFERENT village than this cadastre",
                    InputType.VILLAGE_REFERENCE,
                    f"Survey {sn} is not in {village}. Provide that village's cadastral "
                    f"reference (surveyor DXF / shapefile) OR 2 corner GPS points to seed it.",
                    _SEED, known)
            elif d.recommendation == "NO_COVERAGE":
                req = InputRequest(
                    sn, d.recommendation,
                    "no auto position (off the surveyed corridor / no cadastral label / "
                    "no seated neighbour to propagate from)",
                    InputType.TWO_CORNER_SEED,
                    f"Survey {sn} has no auto position. Provide 2 corner stone -> UTM "
                    f"points (GPS/total-station) to place it exactly.",
                    _SEED, known)
            elif method.startswith("cadastral"):
                bad_area = area == area and not (_CAD_AREA_LO <= area <= _CAD_AREA_HI)
                bad_resid = d.cad_residual > _CAD_ROT_RESID_MAX
                if bad_area or bad_resid:
                    area_s = f"{area:.2f}" if area == area else "n/a"
                    req = InputRequest(
                        sn, d.recommendation,
                        f"cadastral parcel boundary unresolved in the source "
                        f"(area_ratio={area_s}, fit_resid={d.cad_residual:.0f}m)",
                        InputType.CLEARER_PARCEL,
                        f"Survey {sn} is located but its parcel outline is merged/open in "
                        f"the source. Provide a clearer/closed parcel polygon "
                        f"(shapefile/KML/sharper image) OR 2 corner GPS points.",
                        _SEED, known)
                else:
                    area_s = f"{area:.2f}" if area == area else "n/a"
                    req = InputRequest(
                        sn, d.recommendation,
                        f"cadastral fit borderline (area_ratio={area_s})"
                        + (f": {note}" if note else ""),
                        InputType.CONFIRM_PLACEMENT,
                        f"Survey {sn} is placed but just under the auto-accept bar. A human "
                        f"confirm (or 2 corner points) finalizes it.",
                        _SEED, known)
            elif method.startswith("geometric"):
                cov_s = f"{cov:.0%}" if cov == cov else "n/a"
                req = InputRequest(
                    sn, d.recommendation,
                    f"matched field stones but boundary only partly traced "
                    f"(chain_cov={cov_s})",
                    InputType.CONFIRM_PLACEMENT,
                    f"Survey {sn} matched real stones but the surveyor traced too little of "
                    f"its boundary to auto-accept. A human confirm (or 2 corner points) "
                    f"finalizes it.",
                    _SEED, known)
            elif d.recommendation == "REVIEW" and d.has_geometry:
                # New-M2 located-REVIEW (e.g. GPS seat on too-short a baseline, propagated
                # plot demoted by tiling, or a method-disagreement) -> a human confirm.
                req = InputRequest(
                    sn, d.recommendation,
                    f"located but below the auto-accept gate"
                    + (f": {note}" if note else ""),
                    InputType.CONFIRM_PLACEMENT,
                    f"Survey {sn} is placed but did not clear the auto-accept gate. A human "
                    f"confirm (or 2 corner points) finalizes it.",
                    _SEED, known)
            else:
                req = InputRequest(
                    sn, d.recommendation, note or "located but unconfirmed",
                    InputType.TWO_CORNER_SEED,
                    f"Survey {sn} needs 2 corner stone -> UTM points to place it exactly.",
                    _SEED, known)
            rep.requests.append(req)

        # ORDER the worklist: most-impactful / cheapest-to-close first.
        rep.requests.sort(key=lambda q: (_ORDER.get(q.input_type, 5), q.survey_number))

        by_type: dict[str, int] = {}
        for q in rep.requests:
            by_type[q.input_type.value] = by_type.get(q.input_type.value, 0) + 1
        n_conf = sum(1 for d in disps if d.is_confident)
        rep.checks.append(Check(
            "path_to_100pct", Severity.INFO,
            f"{n_conf}/{len(disps)} auto-confident; {len(rep.requests)} plots need input "
            f"to reach 100%: {by_type}. Worklist ordered most-impactful first. "
            f"Every request is closeable via {_SEED}."))
        rep.notes.append("Providing the requested inputs makes each plot ACCEPT_SEEDED "
                         "(human-supplied identity -> still 0 false positives).")
        return rep


# Single-source cross-village check (see dispositions.is_cross_village).
from .dispositions import is_cross_village as _is_cross_village  # noqa: E402
