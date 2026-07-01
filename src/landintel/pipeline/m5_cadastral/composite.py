"""Composite cadastral source: a PRIMARY cadastre with a SECONDARY fallback.

Different public cadastres (TNGIS tiles, the mypropertyqr S3 tiles, client vector files)
disagree for a minority of parcels -- a survey number gets subdivided / merged / re-surveyed,
or one raster renders a small parcel cleaner than another. Relying on a single source leaves
those few as REVIEW even though another source draws them correctly.

This source prefers the ``primary`` for every survey and OFFERS the ``secondary``'s parcel +
candidates wherever the primary either has no parcel or its parcel fails the downstream gate.
The rigid shape + seat-locality gate in ``cadastral_seat`` remains the SOLE arbiter of ACCEPT,
so a second source can only ADD recall, never a false ACCEPT (a wrong-vintage / wrong-village
parcel that does not match the FMB shape+position is rejected exactly as before).
"""
from __future__ import annotations

from .source import CadastralParcel, CadastralSource


class CompositeCadastralSource(CadastralSource):
    """Try ``primary`` first; fall back to ``secondary`` per survey number.

    Every method delegates to whichever source owns the survey. ``recovered_candidates``
    UNIONS both sources' candidates (plus the secondary's primary parcel when the primary
    source already placed one) so the gate can pick the best-fitting ring across sources.
    """

    def __init__(self, primary: CadastralSource, secondary: CadastralSource):
        self.primary = primary
        self.secondary = secondary
        self.crs = getattr(primary, "crs", getattr(secondary, "crs", "EPSG:32643"))

    def get(self, survey_number: str, village: str | None = None) -> CadastralParcel | None:
        p = self.primary.get(survey_number)
        return p if p is not None else self.secondary.get(survey_number)

    def recovered_candidates(self, survey_number: str) -> list[CadastralParcel]:
        out = list(self.primary.recovered_candidates(survey_number) or [])
        # When the primary already supplies get(), the secondary's parcel is an extra candidate
        # (when it does not, that parcel surfaces through get() above instead -- avoid duplication).
        if self.primary.get(survey_number) is not None:
            sp = self.secondary.get(survey_number)
            if sp is not None:
                out.append(sp)
        out.extend(self.secondary.recovered_candidates(survey_number) or [])
        return out

    def label_point(self, survey_number: str) -> tuple[float, float] | None:
        lp = _call(self.primary, "label_point", survey_number)
        return lp if lp is not None else _call(self.secondary, "label_point", survey_number)

    def is_aggressive(self, survey_number: str) -> bool:
        """Aggressive only if NEITHER source offers a non-aggressive parcel for this survey.

        The gate may adopt a recovered candidate from EITHER source, so a clean (non-aggressive)
        parcel from one source must NOT be demoted to REVIEW just because the OTHER source's
        primary was an aggressive sub-cell recovery (e.g. survey 698: TNGIS only has an aggressive
        sub-cell, but S3 draws the full parent cleanly -> not aggressive)."""
        has_clean = has_aggr = False
        for src in (self.primary, self.secondary):
            if src.get(survey_number) is not None:
                if _aggr(src, survey_number):
                    has_aggr = True
                else:
                    has_clean = True
        return has_aggr and not has_clean

    def survey_numbers(self) -> set[str]:
        return set(self.primary.survey_numbers()) | set(self.secondary.survey_numbers())


def _call(source, method: str, *args):
    f = getattr(source, method, None)
    return f(*args) if callable(f) else None


def _aggr(source, survey_number: str) -> bool:
    f = getattr(source, "is_aggressive", None)
    return bool(f(survey_number)) if callable(f) else False
