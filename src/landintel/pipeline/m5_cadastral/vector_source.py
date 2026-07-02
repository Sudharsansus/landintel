"""EXACT vector cadastral source -- parcels straight from the TNGIS statewide cadastre
(ramSeraph/indian_cadastrals, CC0), NOT rasterized S3 tiles + OCR.

Why this exists: the S3 ``survey_border`` tiles are a pre-rendered z18 raster (z19+ is 403),
so parcel boundaries carry a few-metre raster registration error and small/elongated parcels'
labels drop below tile resolution. Both hurt the M2 IoU acceptance signal exactly on thin
parcels (see the KANDAMPALAYAM road-strip surveys). The TNGIS release is the SAME cadastre the
tiles are rasterized from, but as exact vector polygons keyed by survey number -- so it removes
the OCR label step and the raster coarseness in one move.

This source plugs into ``club_pipeline`` through the same duck-typed interface as
``S3CadastralSource`` (``get`` / ``label_point`` / ``recovered_candidates`` / ``is_vector_parcel``
/ ``survey_numbers``). It only supplies geometry; the same math gates decide ACCEPT/REVIEW, so
0-FP is unchanged.
"""
from __future__ import annotations

import logging

from shapely.geometry import Polygon

from .source import TARGET_CRS, CadastralParcel, _norm_survey

_log = logging.getLogger(__name__)


class VectorCadastralSource:
    """Resolve a survey number to its EXACT cadastral parcel polygon (target UTM zone).

    Construct from a ``{survey_number: Polygon}`` map that has ALREADY been disambiguated to the
    one target village (see ``select_village_parcels``). Every parcel here is a real vector ring,
    so ``is_vector_parcel`` is always True and ``label_point`` is the parcel's own centroid (an
    exact seat anchor, unlike the OCR label-pixel centroid the tile source must use)."""

    def __init__(self, parcels: dict[str, Polygon], crs: str = TARGET_CRS,
                 village: str | None = None):
        self._by_survey: dict[str, CadastralParcel] = {}
        for sv, poly in parcels.items():
            key = _norm_survey(sv) or sv
            if poly is None or poly.is_empty:
                continue
            p = poly if poly.is_valid else poly.buffer(0)
            if p.geom_type != "Polygon" or p.area <= 0:
                # keep only a clean single ring; a MultiPolygon -> largest part
                if p.geom_type == "MultiPolygon" and len(p.geoms):
                    p = max(p.geoms, key=lambda g: g.area)
                else:
                    continue
            self._by_survey[key] = CadastralParcel(survey_number=key, polygon=p,
                                                   village=village, source_crs=crs)

    # -- duck-typed CadastralSource interface used by club_pipeline / cadastral_seat --
    def get(self, survey_number: str, village: str | None = None) -> CadastralParcel | None:
        return self._by_survey.get(_norm_survey(survey_number) or survey_number)

    def label_point(self, survey_number: str) -> tuple[float, float] | None:
        p = self.get(survey_number)
        if p is None:
            return None
        c = p.polygon.centroid
        return (float(c.x), float(c.y))

    def recovered_candidates(self, survey_number: str) -> list[CadastralParcel]:
        return []                       # exact rings -- no open/merged recovery needed

    def is_vector_parcel(self, survey_number: str) -> bool:
        return (_norm_survey(survey_number) or survey_number) in self._by_survey

    def is_aggressive(self, survey_number: str) -> bool:
        return False

    def survey_numbers(self) -> set[str]:
        return set(self._by_survey)

    def __contains__(self, survey_number: str) -> bool:
        return self.get(survey_number) is not None

    def __len__(self) -> int:
        return len(self._by_survey)
