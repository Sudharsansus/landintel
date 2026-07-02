"""Locality-cluster USE CASES -- general synthetic geometry, NO village data.

The cadastral fetch parses the FMB survey numbers, OCR-locates them on the public
tiles, and must extract only the ONE village's tiles. Survey numbers are unique only
WITHIN a village, so the same number ("2", "14", ...) is read in neighbouring villages
too. Without clustering, every FMB grabbed a same-numbered parcel anywhere in the window
and the parcels scattered across the whole district (the MOOLAKARAI QA scatter).

These tests codify the SCENARIO on synthetic point clouds so the anti-scatter behaviour
is provably general (not tuned to any village):

  * two dense groups separated by a gap > link distance -> two clusters
  * points chained within the link distance -> one cluster (single-linkage)
  * the village = the cluster with the MOST DISTINCT survey numbers, even if a
    neighbouring cluster is physically denser
  * a survey number read in BOTH the village and a neighbour keeps the IN-VILLAGE read
  * cross-village duplicates far from the village are dropped
"""
from __future__ import annotations

import math

from landintel.pipeline.m5_cadastral.geo_locate import (
    _cluster_points,
    _select_in_village,
    _village_blocks,
    _village_cluster,
)


# --------------------------------------------------------------------------- #
# 1. _cluster_points -- single-linkage on a point cloud (pure geometry)
# --------------------------------------------------------------------------- #
def test_two_groups_beyond_link_split():
    # group A near origin, group B ~3 km east; link 600 m -> two clusters
    pts = [(0.0, 0.0), (100.0, 50.0), (200.0, -30.0),        # A
           (3000.0, 0.0), (3100.0, 40.0)]                     # B
    comps = _cluster_points(pts, link_m=600.0)
    assert len(comps) == 2
    assert sorted(len(c) for c in comps) == [2, 3]


def test_chain_within_link_is_one_cluster():
    # a chain of points each < link from the next single-links into ONE cluster,
    # even though the ends are > link apart (500+500 > 600)
    pts = [(0.0, 0.0), (500.0, 0.0), (1000.0, 0.0), (1500.0, 0.0)]
    comps = _cluster_points(pts, link_m=600.0)
    assert len(comps) == 1
    assert len(comps[0]) == 4


def test_single_point_is_its_own_cluster():
    assert _cluster_points([(10.0, 10.0)], link_m=600.0) == [[0]]


# --------------------------------------------------------------------------- #
# 2. _village_cluster -- pick the village + dedup cross-village duplicates
# --------------------------------------------------------------------------- #
def _reads(*items):
    """Build {sn: [(x,y,conf), ...]} from (sn, x, y[, conf]) tuples."""
    out: dict[str, list[tuple[float, float, float]]] = {}
    for sn, x, y, *conf in items:
        out.setdefault(sn, []).append((float(x), float(y), float(conf[0]) if conf else 1.0))
    return out


def test_village_is_densest_distinct_cluster():
    """5 survey numbers tight near origin (the village) + 2 stray duplicates 3 km away.
    The village cluster wins and only its 5 numbers survive."""
    reads = _reads(
        ("2", 0, 0), ("3", 120, 40), ("14", 60, 150), ("16", 200, 90), ("26", 150, 210),
        # cross-village duplicates of 2 and 14, far away (different village, same numbers)
        ("2", 3000, 20), ("14", 3080, 120),
    )
    core = _village_cluster(reads, link_m=600.0)
    assert core is not None
    assert set(core) == {"2", "3", "14", "16", "26"}
    # the kept "2"/"14" are the IN-VILLAGE reads, not the 3 km duplicates
    assert math.hypot(*core["2"]) < 600.0
    assert math.hypot(*core["14"]) < 600.0


def test_more_distinct_surveys_beats_denser_neighbour():
    """A neighbouring cluster that is physically TIGHTER but holds FEWER distinct
    surveys must NOT win over the true village (more distinct numbers)."""
    reads = _reads(
        # village: 4 distinct numbers spread ~200 m
        ("5", 0, 0), ("6", 180, 20), ("7", 40, 190), ("8", 210, 200),
        # neighbour: only 2 distinct numbers but jammed within 10 m
        ("5", 5000, 0), ("6", 5005, 3),
    )
    core = _village_cluster(reads, link_m=600.0)
    assert set(core) == {"5", "6", "7", "8"}
    assert core["5"] == (0.0, 0.0)          # kept the village's 5, not the neighbour's


def test_isolated_duplicate_rejected_for_in_village_read():
    """A survey read once IN the village (among village-mates) and once as an isolated
    cross-village duplicate keeps the IN-VILLAGE read -- mutual support, not proximity to
    a (duplicate-polluted) centroid."""
    reads = _reads(
        ("9", 0, 0), ("10", 100, 0), ("11", 50, 100),
        ("9", 4000, 4000),                  # isolated cross-village duplicate of 9
    )
    core = _village_cluster(reads, link_m=600.0)
    assert set(core) == {"9", "10", "11"}
    assert core["9"] == (0.0, 0.0)          # the supported (in-village) read, not (4000,4000)


def test_select_in_village_scores_by_neighbours():
    """_select_in_village picks each survey's reading with the most distinct village-mates
    within support_r; an isolated read scores 0 and loses to a supported one."""
    reads = _reads(
        ("7", 0, 0), ("8", 200, 0), ("9", 0, 200),   # a tight triangle (mutual support)
        ("7", 9000, 9000),                            # lone duplicate of 7
    )
    chosen = _select_in_village(reads, support_r=450.0)
    assert set(chosen) == {"7", "8", "9"}
    x, y, support = chosen["7"]
    assert (x, y) == (0.0, 0.0) and support == 2      # sees 8 and 9, not the lone dup


def test_two_neighbour_villages_are_separate_candidate_blocks():
    """The same survey numbers 1..N recur in every TN village, so two neighbouring villages
    both containing our numbers must surface as TWO candidate blocks (never unioned into one
    3-km blob) -- and a distant third village is dropped by the anchor radius. Which block is
    really ours is decided later by FMB-shape fit, not here."""
    anchor = (1000.0, 1000.0)
    reads = _reads(
        # village A (4 of our numbers) near anchor
        ("5", 900, 900), ("6", 1050, 960), ("7", 980, 1150), ("8", 1120, 1040),
        # village B (SAME 4 numbers) ~2.2 km north -- a neighbour, still near anchor
        ("5", 950, 3100), ("6", 1080, 3150), ("7", 1010, 2950), ("8", 1150, 3050),
        # village C (same numbers) ~9 km east -> beyond anchor radius, excluded
        ("5", 9000, 1000), ("6", 9080, 1050), ("7", 9010, 900), ("8", 9100, 1100),
    )
    blocks = _village_blocks(reads, village_r=600.0, anchor=anchor, anchor_radius_m=4000.0)
    assert len(blocks) == 2                                  # A and B, NOT unioned, C dropped
    for blk in blocks:
        assert set(blk) == {"5", "6", "7", "8"}             # each block is a full village
        xs = [p[0] for p in blk.values()]
        assert max(xs) < 4000.0                             # neither is the 9 km village C
    # _village_cluster returns the single most-complete block (tie -> first)
    assert set(_village_cluster(reads, link_m=600.0, anchor=anchor)) == {"5", "6", "7", "8"}


def test_singleton_duplicate_never_forms_a_block():
    """A survey read once in isolation (no village-mates within support_r) is label noise:
    it is dropped before clustering and never becomes its own candidate block."""
    reads = _reads(
        ("40", 0, 0), ("41", 150, 0), ("42", 60, 140), ("43", 200, 90), ("44", 90, 210),
        ("99", 8000, 8000),                         # lone survey, no support -> noise
    )
    blocks = _village_blocks(reads, village_r=600.0)
    assert len(blocks) == 1
    assert set(blocks[0]) == {"40", "41", "42", "43", "44"}   # 99 is dropped


def test_no_reads_returns_none():
    assert _village_cluster({}, link_m=600.0) is None
    assert _select_in_village({}, support_r=450.0) == {}
