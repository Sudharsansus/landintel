"""VectorCadastralSource + village_candidates -- synthetic geometry, NO parquet/village data.

Locks the exact-vector-cadastre path as GENERAL (not tuned to any village): the source
duck-types the S3 interface, and the candidate finder disambiguates purely by anchor proximity
and FMB-survey overlap -- it PROPOSES villages, never picks one (shape-IoU does that upstream).
"""
from __future__ import annotations

from shapely.geometry import Polygon

from landintel.pipeline.m5_cadastral.vector_source import VectorCadastralSource
from landintel.pipeline.m5_cadastral.vector_locate import village_candidates


def _sq(cx, cy, s):
    h = s / 2.0
    return Polygon([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)])


# ------------------------------------------------------------------- VectorCadastralSource
def test_source_interface_and_exact_parcels():
    src = VectorCadastralSource({"12": _sq(0, 0, 100), "13": _sq(200, 0, 60)})
    p = src.get("12")
    assert p is not None and abs(p.polygon.area - 10000) < 1e-6
    assert src.is_vector_parcel("12") and not src.is_vector_parcel("99")
    assert src.label_point("12") == (0.0, 0.0)            # centroid = exact seat anchor
    assert src.recovered_candidates("12") == [] and src.is_aggressive("12") is False
    assert src.survey_numbers() == {"12", "13"} and "13" in src and len(src) == 2


def test_source_normalises_and_drops_degenerate():
    src = VectorCadastralSource({"5": _sq(0, 0, 40),
                                 "6": Polygon([(0, 0), (1, 0), (1, 0)])})  # zero-area -> dropped
    assert src.get("5") is not None and src.get("6") is None


# ------------------------------------------------------------------------- village_candidates
def _parcels_for(vc, base, surveys, span=40):
    # a compact village: each survey a small square near `base`
    return [{"sn": s, "vc": vc, "poly": _sq(base[0] + i * 60, base[1], span)}
            for i, s in enumerate(surveys)]


def test_candidates_ranked_by_overlap_then_proximity():
    fmb = {"1", "2", "3", "4", "5"}
    # village A near the anchor with all 5; village B further with all 5; village C near but few
    parcels = (_parcels_for("A", (0, 0), ["1", "2", "3", "4", "5"])
               + _parcels_for("B", (3000, 0), ["1", "2", "3", "4", "5"])
               + _parcels_for("C", (200, 200), ["1", "9"]))          # only 1 overlap -> dropped
    cands = village_candidates(parcels, fmb, (0, 0), radius_m=5000,
                               min_overlap=3, max_cand=6)
    vcs = [c["vc"] for c in cands]
    assert "C" not in vcs                                            # below min_overlap
    assert vcs[0] == "A"                                            # same overlap, nearest first
    assert set(vcs) == {"A", "B"}
    # each candidate exposes a ready VectorCadastralSource of just the FMB surveys
    src = cands[0]["source"]
    assert src.survey_numbers() == fmb and cands[0]["n_overlap"] == 5


def test_candidates_respects_radius():
    fmb = {"1", "2", "3"}
    parcels = _parcels_for("FAR", (9000, 0), ["1", "2", "3"])
    assert village_candidates(parcels, fmb, (0, 0), radius_m=5000, min_overlap=3) == []


def test_candidates_one_parcel_per_survey_takes_largest():
    fmb = {"7"}
    parcels = [{"sn": "7", "vc": "V", "poly": _sq(0, 0, 30)},
               {"sn": "7", "vc": "V", "poly": _sq(0, 0, 90)},        # larger dup
               {"sn": "2", "vc": "V", "poly": _sq(60, 0, 40)},
               {"sn": "3", "vc": "V", "poly": _sq(120, 0, 40)}]
    cands = village_candidates(parcels, fmb | {"2", "3"}, (0, 0),
                               radius_m=5000, min_overlap=1)
    src = cands[0]["source"]
    assert abs(src.get("7").polygon.area - 8100) < 1e-6             # 90x90, the larger dup
