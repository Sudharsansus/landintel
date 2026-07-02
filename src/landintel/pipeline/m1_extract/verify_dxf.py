"""Deterministic verification gate for M1-produced DXFs (the M1 -> M2 barrier).

A DXF is only "proper" -- fit to hand to M2 georeferencing -- once it passes the
load-bearing structural and geometric invariants checked here. This runs ON THE
WRITTEN ARTIFACT (read back with ezdxf), so it also catches serialization bugs
and is exactly the file M2 will consume.

Design (see the agent layer for the judgment side):
  * These are PASS/FAIL invariants -- pure geometry/structure -- so they are
    deterministic CODE, not an LLM agent. An agent is the wrong tool for "does
    this polygon close to within tolerance".
  * Severity "fail" = structural/geometric breakage that makes the file unfit
    for M2 (open boundary, duplicated boundary lines, degenerate sliver, area
    far from the stated FMB area, non-finite coordinates). A failing file must
    NOT be promoted to M2.
  * Severity "warn" = label-level noise (a stray non-numeric dimension token, an
    odd stone label). Per the project's measurement-label-noise policy these are
    REPORTED, never gated -- OCR cannot reliably tell a plausible number from a
    real edge measurement, so gating would flood review with false positives.

The canonical failure this gate exists to stop: the INGUR survey-667 case, where
a parcel's left edge was drawn DASHDOT, filed as SEPARATION, dropped from the
boundary, and the perimeter silently collapsed to a top strip + chord ("multiple
boundary lines"). That is an open/duplicated/degenerate boundary -- all caught
below.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import ezdxf

from ...agent.anomaly import (  # area tolerances the anomaly layer calibrated
    AREA_FAIL_TOLERANCE,
    AREA_FLAG_TOLERANCE,
    MIN_CORNER_STONES,
)
from ...core.enums import LayerType
from .build_plot import _is_valid_stone_label  # reuse the canonical stone-label rule

__all__ = [
    "VerifyCheck",
    "M1VerifyReport",
    "verify_m1_dxf",
    "write_verify_sidecar",
]

# Two boundary endpoints within this many metres are treated as the same node
# (digitisation slack). A closed ring has every node shared by exactly 2 edges.
_NODE_TOL = 0.5

# A genuine plot fills a healthy fraction of its bounding box. A collapsed
# "top strip + chord" artifact fills almost none -- this ratio separates them.
_MIN_FILL_RATIO = 0.10

# UTM/relative coordinates this large are certainly corrupt (M1 is relative
# metres, ~hundreds; M2 is UTM, <1e7). Anything past this is a blow-up.
_MAX_ABS_COORD = 1.0e7

_NUMERIC_LABEL_RE = re.compile(r"^\d{1,4}([.,]\d{1,3})?$")


@dataclass
class VerifyCheck:
    """One verification check outcome."""
    name: str
    passed: bool
    severity: str          # "fail" (gates) | "warn" (reports only)
    detail: str = ""


@dataclass
class M1VerifyReport:
    """Verification outcome for one M1 DXF."""
    file: str
    checks: list[VerifyCheck] = field(default_factory=list)

    @property
    def proper(self) -> bool:
        """True when no hard (severity='fail') check failed -> fit for M2."""
        return not any((not c.passed) and c.severity == "fail" for c in self.checks)

    @property
    def warnings(self) -> list[VerifyCheck]:
        return [c for c in self.checks if (not c.passed) and c.severity == "warn"]

    @property
    def failures(self) -> list[VerifyCheck]:
        return [c for c in self.checks if (not c.passed) and c.severity == "fail"]

    def to_text(self) -> str:
        status = "PROPER (fit for M2)" if self.proper else "IMPROPER (blocked from M2)"
        lines = [
            f"M1 DXF VERIFICATION: {Path(self.file).name}",
            f"STATUS: {status}",
            "-" * 60,
        ]
        for c in self.checks:
            icon = "OK  " if c.passed else ("FAIL" if c.severity == "fail" else "WARN")
            lines.append(f"[{icon}] {c.name:<26} {c.detail}")
        return "\n".join(lines) + "\n"


def _boundary_segments(msp) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
    """Return BOUNDARY polylines as (start, end, length) using their endpoints."""
    out = []
    for e in msp.query("LWPOLYLINE"):
        if e.dxf.layer != LayerType.BOUNDARY.value:
            continue
        pts = [(p[0], p[1]) for p in e.get_points()]
        if len(pts) < 2:
            continue
        length = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
        out.append((pts[0], pts[-1], length))
    return out


def _cluster_id(pt, nodes):
    """Find an existing node within _NODE_TOL of pt, else register a new one."""
    for i, n in enumerate(nodes):
        if math.dist(pt, n) <= _NODE_TOL:
            return i
    nodes.append(pt)
    return len(nodes) - 1


def _drawing_self_consistency(msp) -> tuple[float, int, int]:
    """Agreement between boundary DIMENSION labels and drawn boundary edge lengths.

    Returns ``(median_ratio, n_consistent, n_total)``. Each boundary dimension
    value is matched to its nearest boundary edge (by midpoint proximity) and the
    ratio ``value / edge_length`` taken. A median ~1.0 means the plot's printed
    dimensions agree with its drawn geometry -- the drawing is internally
    self-consistent, INDEPENDENT of the header's stated area.

    This is the signal that tells two very different situations apart:
      * dimensions do NOT match the edges  -> the geometry/scale is wrong (a real
        extraction error; dimension VALUES are absolute metres, so a scale misread
        moves the edge lengths but not the values, driving the ratio off 1.0);
      * dimensions DO match the edges but the header stated area disagrees -> the
        geometry is faithful to the drawing and the header value is the outlier (a
        government source-data inconsistency, not something M1 can or should
        "fix" by deforming correct geometry).
    """
    edges = []
    for e in msp.query("LWPOLYLINE"):
        if e.dxf.layer != LayerType.BOUNDARY.value:
            continue
        pts = [(p[0], p[1]) for p in e.get_points()]
        if len(pts) < 2:
            continue
        a, b = pts[0], pts[-1]
        length = math.dist(a, b)
        if length > 0:
            edges.append((((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0), length))
    if not edges:
        return float("nan"), 0, 0

    ratios = []
    for e in msp.query("TEXT"):
        if e.dxf.layer != LayerType.BOUNDARY_DIMENSIONS.value:
            continue
        raw = str(e.dxf.text).strip()
        if not _NUMERIC_LABEL_RE.match(raw):
            continue
        try:
            val = float(raw.replace(",", "."))
        except ValueError:
            continue
        if val <= 0:
            continue
        pos = (e.dxf.insert[0], e.dxf.insert[1])
        _mid, length = min(edges, key=lambda ed: math.dist(pos, ed[0]))
        if length > 1.0:
            ratios.append(val / length)
    if not ratios:
        return float("nan"), 0, 0
    ratios.sort()
    n = len(ratios)
    median = ratios[n // 2] if n % 2 else (ratios[n // 2 - 1] + ratios[n // 2]) / 2.0
    n_consistent = sum(1 for r in ratios if 0.80 <= r <= 1.25)
    return median, n_consistent, n


def verify_m1_dxf(
    dxf_path: str | Path,
    *,
    stated_area_ha: float | None = None,
    expected_stones: int | None = None,
) -> M1VerifyReport:
    """Verify a written M1 DXF against the load-bearing invariants.

    Parameters
    ----------
    dxf_path : path to the M1 DXF to verify
    stated_area_ha : the FMB header stated area (hectares), if known, for the
        area cross-check. When None the area check is skipped (reported as warn).
    expected_stones : the expected corner-stone count (e.g. the PDF red-fill
        count), if known. When None only a >= MIN_CORNER_STONES sanity check runs.
    """
    dxf_path = Path(dxf_path)
    report = M1VerifyReport(file=str(dxf_path))
    add = report.checks.append

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    # --- 1. Canonical layers present -------------------------------------
    present = {layer.dxf.name for layer in doc.layers}
    missing = [lt.value for lt in LayerType if lt.value not in present]
    add(VerifyCheck(
        "layers_present", not missing, "fail",
        "all canonical layers present" if not missing else f"missing: {missing}",
    ))

    # --- 2. Stones present ------------------------------------------------
    stones = [e for e in msp.query("TEXT") if e.dxf.layer == LayerType.STONES.value]
    n_stones = len(stones)
    if expected_stones is not None:
        ok = n_stones == expected_stones
        add(VerifyCheck("stone_count", ok, "fail",
                        f"stones={n_stones} expected={expected_stones}"))
    else:
        ok = n_stones >= MIN_CORNER_STONES
        add(VerifyCheck("stone_count", ok, "fail",
                        f"stones={n_stones} (min {MIN_CORNER_STONES})"))

    # --- 3. Boundary present ---------------------------------------------
    segs = _boundary_segments(msp)
    add(VerifyCheck("boundary_present", len(segs) >= 3, "fail",
                    f"{len(segs)} boundary segments"))

    # --- 4. No duplicate boundary lines (the "multiple boundary lines" bug) -
    seen: set[tuple] = set()
    dups = 0
    for a, b, _ in segs:
        key = tuple(sorted([(round(a[0], 1), round(a[1], 1)),
                            (round(b[0], 1), round(b[1], 1))]))
        if key in seen:
            dups += 1
        seen.add(key)
    add(VerifyCheck("no_duplicate_boundary", dups == 0, "fail",
                    "no duplicate boundary edges" if dups == 0
                    else f"{dups} duplicate/overlapping boundary edge(s)"))

    # --- 5. Boundary closes into a single ring (every node degree 2) ------
    nodes: list[tuple[float, float]] = []
    degree: dict[int, int] = {}
    for a, b, _ in segs:
        ia, ib = _cluster_id(a, nodes), _cluster_id(b, nodes)
        degree[ia] = degree.get(ia, 0) + 1
        degree[ib] = degree.get(ib, 0) + 1
    # A closed ring traversal leaves every node at EVEN degree (entered as often
    # as it is left): degree 2 normally, degree 4 where the parcel legitimately
    # pinches/self-touches at a single shared corner (common on subdivision-dense
    # blocks, e.g. Manur survey 67). The collapse this gate guards against (the
    # INGUR-667 top-strip+chord) instead leaves ODD-degree nodes -- a dangling end
    # (degree 1) or a T-branch (degree 3) -- so we fail on odd degree, not on a
    # clean self-touch.
    odd_nodes = [d for d in degree.values() if d % 2 == 1]
    closed = len(segs) >= 3 and not odd_nodes and len(nodes) >= 3
    detail = ("ring closed (all nodes even degree)" if closed
              else f"open/branched: {len(odd_nodes)} odd-degree (dangling/branch) node(s)")
    add(VerifyCheck("boundary_closed", closed, "fail", detail))

    # Compute the boundary face area once (reused by checks 6 and 7).
    all_pts = [p for a, b, _ in segs for p in (a, b)]
    boundary_area_m2 = 0.0
    if segs:
        try:
            from shapely.geometry import MultiLineString
            from shapely.ops import polygonize, unary_union
            polys = list(polygonize(unary_union(
                MultiLineString([[a, b] for a, b, _ in segs]))))
            # Union of faces -> the full enclosed parcel (dissolves any internal
            # divisions back into one outer region), matching build_plot.
            if polys:
                merged = unary_union(polys)
                boundary_area_m2 = merged.area
        except Exception:
            boundary_area_m2 = 0.0

    # --- 6. Boundary not a degenerate sliver -----------------------------
    fill_ok = True
    detail = "no boundary"
    if all_pts:
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        bbox = (max(xs) - min(xs)) * (max(ys) - min(ys))
        ratio = (boundary_area_m2 / bbox) if bbox > 0 else 0.0
        fill_ok = ratio >= _MIN_FILL_RATIO
        detail = f"face/bbox fill ratio={ratio:.2f} (min {_MIN_FILL_RATIO})"
    add(VerifyCheck("boundary_not_degenerate", fill_ok, "fail", detail))

    # --- 7. Area vs stated FMB area (the canonical collapse gate) ---------
    # A large area error is only a HARD failure when the geometry itself is
    # suspect. If the drawing's own boundary dimensions confirm the drawn edge
    # lengths (self-consistent), the geometry is faithful and a disagreeing header
    # stated area is a SOURCE-DATA inconsistency (verified on Manur: many FMB
    # headers state an area that contradicts their own drawing) -- flag it, but do
    # not block an otherwise-correct parcel from M2.
    if stated_area_ha and stated_area_ha > 0:
        computed_ha = boundary_area_m2 / 10_000.0
        err = abs(computed_ha - stated_area_ha) / stated_area_ha
        if err <= AREA_FLAG_TOLERANCE:
            add(VerifyCheck("area_vs_stated", True, "warn",
                            f"computed {computed_ha:.3f} ha vs stated {stated_area_ha:.3f} ha "
                            f"({err * 100:.1f}% off)"))
        else:
            med, nc, nt = _drawing_self_consistency(msp)
            # The MEDIAN is the dispositive scale signal: dimension values are
            # absolute metres, so a scale misread shifts EVERY value/edge ratio and
            # moves the median off 1.0. A median ~1.0 over enough samples means the
            # scale/geometry is faithful. The fraction-consistent is a secondary
            # floor against a coincidental median (proximity dim->edge matching is
            # noisier than build-time anchoring, so it is kept lenient at 0.40).
            coherent = (nt >= 5 and not math.isnan(med)
                        and 0.85 <= med <= 1.18 and nc / nt >= 0.40)
            if err > AREA_FAIL_TOLERANCE and not coherent:
                add(VerifyCheck(
                    "area_vs_stated", False, "fail",
                    f"computed {computed_ha:.3f} ha vs stated {stated_area_ha:.3f} ha "
                    f"({err * 100:.1f}% off); boundary dimensions do NOT confirm the "
                    f"geometry (median dim/edge={med:.2f}, {nc}/{nt} consistent) "
                    f"-> geometry/scale suspect"))
            elif coherent:
                add(VerifyCheck(
                    "area_vs_stated", False, "warn",
                    f"stated {stated_area_ha:.3f} ha disagrees with drawing "
                    f"({computed_ha:.3f} ha, {err * 100:.1f}% off) BUT the drawing's own "
                    f"boundary dimensions confirm the geometry (median dim/edge={med:.2f}, "
                    f"{nc}/{nt} consistent) -> SOURCE-DATA inconsistency, geometry faithful"))
            else:
                add(VerifyCheck(
                    "area_vs_stated", False, "warn",
                    f"computed {computed_ha:.3f} ha vs stated {stated_area_ha:.3f} ha "
                    f"({err * 100:.1f}% off)"))
    else:
        add(VerifyCheck("area_vs_stated", True, "warn", "no stated area to verify against"))

    # --- 8. Coordinates finite & in range --------------------------------
    bad_coord = None
    for e in msp.query("LWPOLYLINE"):
        for p in e.get_points():
            if not (math.isfinite(p[0]) and math.isfinite(p[1])) or \
               abs(p[0]) > _MAX_ABS_COORD or abs(p[1]) > _MAX_ABS_COORD:
                bad_coord = (e.dxf.layer, p[0], p[1])
                break
        if bad_coord:
            break
    add(VerifyCheck("coords_finite", bad_coord is None, "fail",
                    "all coordinates finite & in range" if bad_coord is None
                    else f"bad coord on {bad_coord[0]}: ({bad_coord[1]:.1f},{bad_coord[2]:.1f})"))

    # --- 9. Labels well-formed (WARN only -- noise is reported, not gated) -
    bad_stone = [str(s.dxf.text).strip() for s in stones
                 if not _is_valid_stone_label(str(s.dxf.text).strip())]
    add(VerifyCheck("stone_labels_wellformed", not bad_stone, "warn",
                    "all stone labels valid" if not bad_stone
                    else f"{len(bad_stone)} odd stone label(s): {bad_stone[:5]}"))

    dim_layers = {LayerType.BOUNDARY_DIMENSIONS.value,
                  LayerType.CHAINLINE_DIMENSIONS.value, LayerType.DIMENSIONS.value}
    bad_dims = [str(e.dxf.text).strip() for e in msp.query("TEXT")
                if e.dxf.layer in dim_layers
                and not _NUMERIC_LABEL_RE.match(str(e.dxf.text).strip())]
    add(VerifyCheck("dimension_labels_numeric", not bad_dims, "warn",
                    "all dimension labels numeric" if not bad_dims
                    else f"{len(bad_dims)} non-numeric dim token(s): {bad_dims[:5]}"))

    # --- 10. Topology: every stone lying ON a boundary edge must be a VERTEX -----
    # Else the edge runs straight THROUGH the stone as one line (client: "the line is
    # divided but shown as a single line -- join it at the point"). to_dxf splits edges
    # at on-edge stones, so this should now pass; the check keeps it from regressing.
    bnd_pls = [e for e in msp.query("LWPOLYLINE") if e.dxf.layer == LayerType.BOUNDARY.value]
    bnd_edges, bnd_verts = [], set()
    for e in bnd_pls:
        pts = [(p[0], p[1]) for p in e.get_points()]
        for p in pts:
            bnd_verts.add((round(p[0], 1), round(p[1], 1)))
        for i in range(len(pts) - 1):
            bnd_edges.append((pts[i], pts[i + 1]))

    def _pt_seg(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-9:
            return math.hypot(px - ax, py - ay), 0.0
        t = ((px - ax) * dx + (py - ay) * dy) / L2
        tc = max(0.0, min(1.0, t))
        return math.hypot(px - (ax + tc * dx), py - (ay + tc * dy)), t

    thru = []
    for s in stones:
        sx, sy = s.dxf.insert.x, s.dxf.insert.y
        if (round(sx, 1), round(sy, 1)) in bnd_verts:
            continue
        for (ax, ay), (bx, by) in bnd_edges:
            d, t = _pt_seg(sx, sy, ax, ay, bx, by)
            if d < 1.0 and 0.05 < t < 0.95:
                thru.append(str(s.dxf.text).strip())
                break
    add(VerifyCheck("stones_are_boundary_vertices", not thru, "warn",
                    "every on-edge stone is a boundary vertex" if not thru
                    else f"{len(thru)} stone(s) sit mid-edge with no vertex (line runs "
                         f"through): {thru[:5]}"))

    # --- 11. Subdivision lines that stop just short of the boundary (attach gap) --
    sub_pls = [e for e in msp.query("LWPOLYLINE")
               if e.dxf.layer == LayerType.SUBDIVISION_LINES.value]
    sub_eps = []
    for e in sub_pls:
        pts = [(p[0], p[1]) for p in e.get_points()]
        if pts:
            sub_eps.append(pts[0])
            sub_eps.append(pts[-1])

    def _deg(pt):
        return sum(1 for q in sub_eps if math.hypot(pt[0] - q[0], pt[1] - q[1]) < 0.3)

    detached = 0
    for ep in sub_eps:
        if _deg(ep) > 1:                       # shared with another subdivision -> interior node
            continue
        d = min((_pt_seg(ep[0], ep[1], a[0], a[1], b[0], b[1])[0]
                 for a, b in bnd_edges), default=999.0)
        if 0.1 < d < 8.0:                      # dangling end, short of the boundary
            detached += 1
    add(VerifyCheck("subdivisions_attached", detached == 0, "warn",
                    "subdivision lines attach to boundary" if detached == 0
                    else f"{detached} subdivision end(s) stop 0.1-8 m short of the boundary"))

    # --- 12. Stone-number OCR recall (INFO/WARN): fraction of '?' synthetic labels --
    n_syn = sum(1 for s in stones if str(s.dxf.text).strip().startswith("?"))
    frac = n_syn / max(len(stones), 1)
    add(VerifyCheck("stone_numbers_read", frac <= 0.5, "warn",
                    f"{len(stones) - n_syn}/{len(stones)} stone numbers read by OCR "
                    f"({frac * 100:.0f}% unread '?'); positions exact regardless"))

    return report


def write_verify_sidecar(report: M1VerifyReport, *, as_json: bool = False) -> Path:
    """Write the verification report next to its DXF and return the sidecar path.

    Default is a human-readable ``<name>.verify.txt``; ``as_json=True`` writes
    ``<name>.verify.json`` for machine consumption.
    """
    dxf = Path(report.file)
    if as_json:
        out = dxf.with_suffix(".verify.json")
        payload = {
            "file": report.file,
            "proper": report.proper,
            "checks": [asdict(c) for c in report.checks],
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        out = dxf.with_suffix(".verify.txt")
        out.write_text(report.to_text(), encoding="utf-8")
    return out
