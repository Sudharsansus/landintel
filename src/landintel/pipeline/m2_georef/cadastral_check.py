"""Cadastral cross-validation -- an INDEPENDENT accuracy/false-positive check on M2 placements.

The math gates decide WHERE a plot is placed (corridor congruence / cadastral fit / seed). This
layer then asks the authoritative cadastral parcel for that survey number: "does the placed plot
actually land on the right parcel?" For a corridor-placed plot the cadastral parcel is a fully
INDEPENDENT reference, so this is genuine ground truth -- it CONFIRMS correct placements (raising
trust) and CATCHES gross misplacements the geometry gates rubber-stamped.

0-FP INVARIANT: this can only TIGHTEN. On a clear cadastral disagreement it DEMOTES ACCEPT->REVIEW
(a human confirms); it NEVER promotes anything. A confirmed agreement is recorded as a trust
signal but does not by itself create an ACCEPT. So adding cadastral data can only improve accuracy,
never manufacture a false positive. Geometry is never warped to the cadastral -- this is a check,
not a placement.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

_log = logging.getLogger(__name__)

_ACCEPT = ("ACCEPT", "ACCEPT_SEEDED", "ACCEPT_CADASTRAL")

# Confirmation: the placed plot clearly sits on its parcel (any one suffices). FMB plots can be
# subdivisions (smaller than the full survey parcel), so containment of the placed plot inside the
# parcel is the most reliable signal; IoU and centroid-inside corroborate.
CONFIRM_CONTAIN = 0.30      # >=30% of the placed footprint lies inside the parcel
CONFIRM_IOU = 0.15
# Gross miss (demote): the placed plot is essentially NOT on its parcel and far from it.
GROSS_MISS_IOU = 0.02
GROSS_MISS_OFFSET_FACTOR = 2.0   # centroid offset > factor * parcel effective radius


@dataclass
class CadastralAgreement:
    survey_number: str
    status: str               # "confirmed" | "weak" | "gross_miss" | "no_parcel"
    centroid_offset_m: float = float("nan")
    iou: float = float("nan")
    contained_frac: float = float("nan")
    centroid_inside: bool = False

    @property
    def confirmed(self) -> bool:
        return self.status == "confirmed"

    @property
    def gross_miss(self) -> bool:
        return self.status == "gross_miss"

    def note(self) -> str:
        if self.status == "no_parcel":
            return "no cadastral parcel"
        return (f"cadastral {self.status}: offset={self.centroid_offset_m:.1f}m "
                f"IoU={self.iou:.2f} contained={self.contained_frac:.2f}")


def cadastral_agreement(placed, parcel, survey_number: str = "") -> CadastralAgreement:
    """Agreement between a placed plot footprint and its authoritative cadastral parcel (both
    shapely polygons in the same UTM frame)."""
    if placed is None or parcel is None or not placed.is_valid or not parcel.is_valid \
            or placed.area <= 0 or parcel.area <= 0:
        return CadastralAgreement(survey_number, "no_parcel")
    inter = placed.intersection(parcel).area
    union = placed.union(parcel).area
    iou = inter / union if union > 0 else 0.0
    contained = inter / placed.area if placed.area > 0 else 0.0
    offset = placed.centroid.distance(parcel.centroid)
    centroid_in = bool(parcel.contains(placed.centroid))
    radius = math.sqrt(parcel.area / math.pi)

    if centroid_in or contained >= CONFIRM_CONTAIN or iou >= CONFIRM_IOU:
        status = "confirmed"
    elif iou < GROSS_MISS_IOU and not centroid_in and offset > GROSS_MISS_OFFSET_FACTOR * radius:
        status = "gross_miss"
    else:
        status = "weak"
    return CadastralAgreement(survey_number, status, round(offset, 2), round(iou, 4),
                              round(contained, 4), centroid_in)


def apply_cadastral_gate(results, cadastral_source, footprint_fn,
                         on_demote=None, on_confirm=None) -> dict:
    """Cross-check every ACCEPT plot against its authoritative cadastral parcel.

    ``footprint_fn(result) -> shapely Polygon | None`` extracts the placed footprint.
    ``cadastral_source.get(survey) -> parcel`` (with a .polygon) supplies the reference.
    Demotes gross misses (calls ``on_demote(result, agreement)``), tags confirmations
    (``on_confirm(result, agreement)``). Returns counts. No-op if no cadastral source.
    """
    out = {"checked": 0, "confirmed": 0, "weak": 0, "gross_miss_demoted": 0, "no_parcel": 0}
    if cadastral_source is None:
        return out
    get = getattr(cadastral_source, "get", None)
    if get is None:
        return out

    offsets = []
    for r in results:
        if r.recommendation not in _ACCEPT:
            continue
        try:
            cp = get(r.survey_number)
        except Exception:  # noqa: BLE001
            cp = None
        parcel = getattr(cp, "polygon", None) if cp is not None else None
        if parcel is None:
            out["no_parcel"] += 1
            continue
        placed = footprint_fn(r)
        ag = cadastral_agreement(placed, parcel, r.survey_number)
        out["checked"] += 1
        if ag.status == "confirmed":
            out["confirmed"] += 1
            offsets.append(ag.centroid_offset_m)
            if on_confirm:
                on_confirm(r, ag)
        elif ag.status == "gross_miss":
            out["gross_miss_demoted"] += 1
            if on_demote:
                on_demote(r, ag)
        else:
            out["weak"] += 1

    out["mean_confirmed_offset_m"] = round(sum(offsets) / len(offsets), 2) if offsets else None
    if out["checked"]:
        _log.info("Cadastral cross-check: %d checked -> %d confirmed, %d weak, %d gross-miss "
                  "demoted (mean confirmed offset %sm)", out["checked"], out["confirmed"],
                  out["weak"], out["gross_miss_demoted"], out["mean_confirmed_offset_m"])
    return out
