"""Club-all M2 (reference-for-surveyor mode) -- every plot clubbed, tiers not exclusions.

Locks the 2026-07-03 client directive: M2 clubs ALL plots into ONE file with 0 review
(confidence carried as labels), while the low-confidence reseat REUSES the validated
edge_align machinery (translation-only toward fixed confident anchors). Synthetic only.
"""
from __future__ import annotations

import numpy as np

from landintel.pipeline.m2_club.edge_align import align_shared_edges
from landintel.pipeline.m2_club.placement import CandidatePlacement, ClubResult


def _placement(ring, rec_pts=None):
    ring = np.asarray(ring, float)
    return CandidatePlacement(
        method="cadastral", R=np.eye(2), s=1.0, t=np.zeros(2),
        adjusted=ring, corner_ring=list(range(len(ring))), passes_gate=True)


def _result(sn, rec, ring):
    return ClubResult(m1_file=f"{sn}.dxf", survey_number=sn, recommendation=rec,
                      method="cadastral", placement=_placement(ring))


def _sq(x0, y0, w, h):
    return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]


# ------------------------------------------------- low-conf reseat via edge_align
def test_lowconf_plot_takes_full_gap_toward_fixed_accept():
    # ACCEPT anchor at x=[0,100]; REVIEW neighbour at x=[108,208] -- an 8 m white
    # gap on a genuinely shared edge. With recs including REVIEW and the ACCEPT
    # fixed, the REVIEW plot must close (most of) the gap; the anchor must not move.
    acc = _result("1", "ACCEPT", _sq(0, 0, 100, 80))
    low = _result("2", "REVIEW", _sq(108, 0, 100, 80))
    a0 = acc.placement.adjusted.copy()
    l0 = low.placement.adjusted.copy()

    st = align_shared_edges([acc, low], recs=("ACCEPT", "ACCEPT_SEEDED", "REVIEW"),
                            fixed={"1"})
    assert np.allclose(acc.placement.adjusted, a0)            # anchor pinned
    moved = float(np.abs(low.placement.adjusted - l0).max())
    assert st.n_constraints >= 1 and st.n_moved == 1
    assert moved > 4.0                                        # gap substantially closed
    # translation-only: edge lengths of the moved plot unchanged (rule #2)
    def _lens(r):
        n = len(r)
        return np.array([np.linalg.norm(r[(i + 1) % n] - r[i]) for i in range(n)])
    assert np.allclose(_lens(low.placement.adjusted), _lens(l0), atol=1e-9)


def test_default_recs_still_ignores_review():
    # Without recs, the validated default stands: REVIEW plots never move.
    acc = _result("1", "ACCEPT", _sq(0, 0, 100, 80))
    low = _result("2", "REVIEW", _sq(108, 0, 100, 80))
    l0 = low.placement.adjusted.copy()
    align_shared_edges([acc, low])
    assert np.allclose(low.placement.adjusted, l0)


# ---------------------------------------------------------- club_dxf lowconf tier
def test_club_dxf_lowconf_blocks_in_main_club(tmp_path):
    import ezdxf

    from landintel.pipeline.m2_club.club_output import club_dxf

    def _mini_dxf(path, x0):
        doc = ezdxf.new()
        msp = doc.modelspace()
        msp.add_lwpolyline([(x0, 0), (x0 + 10, 0), (x0 + 10, 10), (x0, 10)],
                           close=True, dxfattribs={"layer": "BOUNDARY"})
        doc.saveas(path)
        return str(path)

    a = _mini_dxf(tmp_path / "a.dxf", 0)
    b = _mini_dxf(tmp_path / "b.dxf", 50)
    out = club_dxf([(a, "10")], [], tmp_path / "club.dxf",
                   review_specs=None, lowconf_specs=[(b, "20")])
    doc = ezdxf.readfile(str(out))
    blocks = {bl.name for bl in doc.blocks}
    assert any(n.startswith("FMB_10") for n in blocks)              # confident tier
    assert any(n.startswith("FMB_20_LOWCONF") for n in blocks)      # low tier, SAME file
    assert not any(n.startswith("REVIEW_FMB") for n in blocks)      # no review framing


# --------------------------------------------------- conditional stone-match label
def test_stone_match_stats_conditional_bar():
    from shapely.geometry import Polygon

    from landintel.pipeline.m2_club.cadastral_seat import _stone_match_stats

    class _P:
        polygon = Polygon([(0, 0), (100, 0), (100, 80), (0, 80)])

    # 4-corner plot placed exactly on the 4-corner parcel -> 4/4, full (bar min(5,4)=4)
    ring = np.array([(0, 0), (100, 0), (100, 80), (0, 80)], float)
    n, req, full = _stone_match_stats(ring, _P())
    assert (n, req, full) == (4, 4, True)

    # same plot displaced far -> 0 matched, honestly not full
    n2, req2, full2 = _stone_match_stats(ring + 500.0, _P())
    assert n2 == 0 and req2 == 4 and not full2
