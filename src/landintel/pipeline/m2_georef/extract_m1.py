"""Extract structured data from an M1-produced FMB DXF.

M1 DXF layer convention (from to_dxf.py / core.enums.LayerType):
  BOUNDARY              -- LWPOLYLINE, one per boundary edge, red (ACI 1)
  STONES                -- TEXT, stone labels (e.g. "A", "B", "1", "2"), white (ACI 7)
  SURVEY_NUMBER         -- TEXT, survey number, red (ACI 1)
  CHAIN_LINES           -- LWPOLYLINE, CHAINLINE linetype, dark gray (ACI 251)
  SUBDIVISION_LINES     -- LWPOLYLINE, green (ACI 3)
  SEPARATION_LINE       -- LWPOLYLINE, DASHDOT linetype
  BOUNDARY_DIMENSIONS   -- TEXT, edge measurements for boundary (dot->comma)
  CHAINLINE_DIMENSIONS  -- TEXT, edge measurements for chain lines
  DIMENSIONS            -- TEXT, subdivision measurements
  SUBDIVISION           -- TEXT, sub-plot labels (e.g. "2A", "3B")
  neighbor label        -- TEXT, neighboring survey numbers (note: lowercase + space)
  DASHED_REF            -- LWPOLYLINE, DASHED linetype

Layer names are sourced from ``landintel.core.enums.LayerType`` (the single
source of truth that ``to_dxf.py`` also writes from), NOT hardcoded strings --
so this extractor cannot drift from M1's actual output. (The ``NEIGHBOR_LABEL``
layer value is literally ``"neighbor label"`` -- lowercase with a space -- which
a hardcoded ``"NEIGHBOR_LABEL"`` query would silently miss.)

This module produces:
  - M1PlotData.stones:  list of (x, y, label) for corner stones
  - M1PlotData.boundary_edges: list of (stone_a, stone_b, length_m)
  - M1PlotData.outer_edges: outer boundary cycle (after leaf peeling)
  - M1PlotData.survey_number: the plot's survey number string
  - M1PlotData.raw_boundary_polys: raw BOUNDARY polylines for warping
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import ezdxf
import numpy as np

from ...core.enums import LayerType

_log = logging.getLogger(__name__)

# Canonical layer names (single source of truth = LayerType, same as to_dxf.py).
_L_BOUNDARY = LayerType.BOUNDARY.value
_L_STONES = LayerType.STONES.value
_L_SURVEY = LayerType.SURVEY_NUMBER.value
_L_CHAIN = LayerType.CHAIN_LINES.value
_L_SUBDIV_LINES = LayerType.SUBDIVISION_LINES.value
_L_SEPARATION = LayerType.SEPARATION_LINE.value
_L_DASHED_REF = LayerType.DASHED_REF.value
_L_BOUNDARY_DIM = LayerType.BOUNDARY_DIMENSIONS.value
_L_CHAIN_DIM = LayerType.CHAINLINE_DIMENSIONS.value
_L_DIM = LayerType.DIMENSIONS.value
_L_SUBDIV = LayerType.SUBDIVISION.value
_L_NEIGHBOR = LayerType.NEIGHBOR_LABEL.value  # "neighbor label"

# Maximum distance (metres) to snap a BOUNDARY polyline endpoint to a stone.
_STONE_SNAP_DIST = 2.0

# Regex to parse survey number from SURVEY_NUMBER text.
# Handles: "784", "784/1", "784/2A", "784A", etc.
_SURVEY_NUM_RE = re.compile(r'^(\d{1,5}(?:/\d+)?(?:[A-Za-z])?)$')


def _entities_on(msp, dxftype: str, layer: str) -> list:
    """Return entities of ``dxftype`` on ``layer``.

    Filters in Python rather than via the ezdxf query DSL so layer names that
    contain spaces (e.g. ``"neighbor label"``) are matched exactly without any
    query-string quoting ambiguity.
    """
    return [e for e in msp.query(dxftype) if e.dxf.layer == layer]


# Ring-node clustering tolerance (metres): boundary segment endpoints closer
# than this are the same ring vertex.
_RING_NODE_TOL = 1.0

# A ring vertex within this distance (metres) of a stone IS that corner stone.
# Stones are written at the line junction, but OCR-borrowed placement can leave
# a small offset, so this is a touch generous.
_CORNER_SNAP = 3.0


def _chain_ring(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    tol: float = _RING_NODE_TOL,
) -> list[tuple[float, float]]:
    """Chain boundary SEGMENTS into one ordered ring of vertices.

    Works on geometry (coordinate adjacency), NOT on stones -- so a boundary
    edge that was drawn as several short segments with intermediate (non-stone)
    vertices still chains into the correct ordered perimeter. Degree-1 spurs are
    pruned; the largest cycle is walked. Returns ordered vertex coordinates
    (open list; the ring closes back to the first).
    """
    if len(segments) < 3:
        return []

    nodes: list[tuple[float, float]] = []

    def node_id(p: tuple[float, float]) -> int:
        for i, n in enumerate(nodes):
            if abs(p[0] - n[0]) <= tol and abs(p[1] - n[1]) <= tol:
                return i
        nodes.append((float(p[0]), float(p[1])))
        return len(nodes) - 1

    adj: dict[int, set[int]] = {}
    for a, b in segments:
        ia, ib = node_id(a), node_id(b)
        if ia == ib:
            continue
        adj.setdefault(ia, set()).add(ib)
        adj.setdefault(ib, set()).add(ia)

    if len(adj) < 3:
        return []

    # Prune dangling degree-1 spurs.
    active = set(adj)
    changed = True
    while changed:
        changed = False
        for n in list(active):
            if len(adj[n] & active) <= 1:
                active.discard(n)
                changed = True
    if len(active) < 3:
        return []

    # Walk the cycle, preferring an unvisited neighbour.
    start = min(active)
    order = [start]
    visited = {start}
    cur, prev = start, -1
    while True:
        nbrs = adj[cur] & active
        nxt = None
        for m in sorted(nbrs):
            if m != prev and m not in visited:
                nxt = m
                break
        if nxt is None:
            break
        order.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt

    return [nodes[i] for i in order]


@dataclass
class M1Stone:
    """A corner stone from the M1 DXF (relative coordinates in metres)."""
    x: float
    y: float
    label: str
    index: int


@dataclass
class M1Edge:
    """A boundary edge connecting two stones."""
    stone_a: int    # index into stones list
    stone_b: int
    length_m: float


@dataclass
class RawBoundaryPoly:
    """Raw BOUNDARY polyline with stone associations for warping."""
    vertices: list[tuple[float, float]]      # all vertex coordinates
    stone_indices: list[int | None]          # stone index per vertex (-1 if none)
    stone_a: int                             # start stone index
    stone_b: int                             # end stone index


@dataclass
class M1PlotData:
    """All extracted data from an M1 DXF."""
    stones: list[M1Stone] = field(default_factory=list)
    boundary_edges: list[M1Edge] = field(default_factory=list)
    raw_boundary_polys: list[RawBoundaryPoly] = field(default_factory=list)
    # Raw boundary segments (geometry, in DXF order) used to chain the ring.
    boundary_segments: list[tuple[tuple[float, float], tuple[float, float]]] = field(
        default_factory=list
    )
    survey_number: str = ""
    source_file: str = ""

    # Outer boundary (populated by extract_outer_boundary)
    outer_edges: list[M1Edge] = field(default_factory=list)
    outer_stone_indices: list[int] = field(default_factory=list)

    # Subdivision / chain / separation data (for warping in output)
    subdivision_verts: list[list[tuple[float, float]]] = field(default_factory=list)
    chain_verts: list[list[tuple[float, float]]] = field(default_factory=list)
    separation_verts: list[list[tuple[float, float]]] = field(default_factory=list)
    dashed_ref_verts: list[list[tuple[float, float]]] = field(default_factory=list)
    dimension_texts: list[dict] = field(default_factory=list)
    neighbor_label_texts: list[dict] = field(default_factory=list)
    sub_plot_label_texts: list[dict] = field(default_factory=list)

    @property
    def n_stones(self) -> int:
        return len(self.stones)

    @property
    def n_edges(self) -> int:
        return len(self.boundary_edges)

    @property
    def edge_lengths(self) -> list[float]:
        """Sorted list of ALL boundary edge lengths."""
        return sorted(e.length_m for e in self.boundary_edges)

    @property
    def outer_edge_lengths(self) -> list[float]:
        """Sorted list of outer boundary edge lengths."""
        return sorted(e.length_m for e in self.outer_edges)

    @property
    def total_perimeter(self) -> float:
        return sum(e.length_m for e in self.boundary_edges)

    @property
    def outer_perimeter(self) -> float:
        return sum(e.length_m for e in self.outer_edges)

    def stone_positions(self) -> np.ndarray:
        """Nx2 array of stone (x, y) positions."""
        return np.array([[s.x, s.y] for s in self.stones])

    def stone_labels(self) -> list[str]:
        return [s.label for s in self.stones]

    def extract_outer_boundary(self):
        """Recover the outer boundary as ORDERED corner-stone edges from geometry.

        The boundary is chained from its raw segments into one ordered ring of
        vertices (``_chain_ring``), independent of stones. Each ring vertex is
        then tested for a nearby stone; consecutive stone-vertices define a
        CORNER-TO-CORNER edge whose length is the summed segment path between
        them (so a multi-segment / slightly-curved edge measures correctly).
        This is what makes matching robust on INGUR boundaries, where many ring
        vertices are intermediate non-stone points.

        Populates ``outer_stone_indices`` (corner stones in ring order),
        ``outer_edges`` and ``boundary_edges`` (corner-to-corner edges). Falls
        back to the legacy leaf-peel on ``boundary_edges`` only if no ring can
        be chained (e.g. a degenerate/open boundary).
        """
        ring = _chain_ring(self.boundary_segments)
        if len(ring) >= 3 and self.stones:
            ring_arr = np.array(ring)

            # STONE-CENTRIC corner detection: assign each stone to its nearest
            # ring vertex (its position ALONG the ring). Every stone that lies on
            # the ring becomes a corner -- a vertex-centric argmin (ring vertex ->
            # nearest stone) silently dropped stones when two stones shared a
            # vertex's neighbourhood, merging edges and inflating their length.
            corner_at: dict[int, tuple[int, float]] = {}  # ring_idx -> (stone, dist)
            for si, st in enumerate(self.stones):
                d = np.sqrt(((ring_arr - np.array([st.x, st.y])) ** 2).sum(axis=1))
                ri = int(np.argmin(d))
                if d[ri] <= _CORNER_SNAP and (
                    ri not in corner_at or d[ri] < corner_at[ri][1]
                ):
                    corner_at[ri] = (si, float(d[ri]))

            # Corner stones in ring order.
            corners = [stone for _ri, (stone, _d) in sorted(corner_at.items())]
            # Collapse accidental consecutive repeats (incl. wrap-around).
            corners = [c for idx, c in enumerate(corners)
                       if idx == 0 or c != corners[idx - 1]]
            if len(corners) >= 2 and corners[0] == corners[-1]:
                corners = corners[:-1]

            if len(corners) >= 3:
                # Edge length = STRAIGHT-LINE chord between consecutive corner
                # stones -- matches how the surveyor measures stone-to-stone
                # (a curved/multi-segment boundary must not inflate the edge).
                self.outer_stone_indices = corners
                self.outer_edges = []
                self.boundary_edges = []
                for idx in range(len(corners)):
                    a = corners[idx]
                    b = corners[(idx + 1) % len(corners)]
                    length = float(math.dist(
                        (self.stones[a].x, self.stones[a].y),
                        (self.stones[b].x, self.stones[b].y)))
                    edge = M1Edge(stone_a=min(a, b), stone_b=max(a, b),
                                  length_m=length)
                    self.outer_edges.append(edge)
                    self.boundary_edges.append(edge)
                _log.info(
                    "Outer boundary (geometry ring): %d corners, %d edges, "
                    "perimeter=%.2fm",
                    len(corners), len(self.outer_edges), self.outer_perimeter)
                return

        # Fallback: legacy leaf-peel on whatever boundary_edges exist.
        self._extract_outer_boundary_legacy()

    def _extract_outer_boundary_legacy(self):
        """Recover the outer boundary as an ORDERED stone ring.

        M1's ``to_dxf`` writes the BOUNDARY layer as the plot's perimeter ring,
        one segment per edge, so the deduped boundary edges form (ideally) one
        simple cycle. The job here is to return that cycle as an *ordered*
        sequence of stones -- order is load-bearing for M2, because matching
        aligns the M1 ring against a surveyor sub-chain by trying cyclic
        rotations, and rotations of an UNORDERED set never recover the ring.

        Strategy:
          1. Build adjacency from the deduped boundary edges.
          2. Iteratively prune dangling degree-1 nodes (spurs from a stray
             segment that snapped to a stone) AND their edges.
          3. Chain the surviving edges into a single ordered walk, preferring at
             each junction an unvisited neighbour -- this recovers the ring for
             clean degree-2 graphs and a sensible longest-path otherwise.

        Populates ``self.outer_edges`` and ``self.outer_stone_indices``. Falls
        back to the raw edge list only when there are too few edges to form a
        ring (an honestly open/degenerate boundary).
        """
        edges = self.boundary_edges
        edge_lookup: dict[tuple[int, int], float] = {
            (min(e.stone_a, e.stone_b), max(e.stone_a, e.stone_b)): e.length_m
            for e in edges
        }

        def _finalize(cycle: list[int]) -> None:
            self.outer_stone_indices = cycle
            self.outer_edges = []
            perim = 0.0
            for i in range(len(cycle)):
                a = cycle[i]
                b = cycle[(i + 1) % len(cycle)]
                key = (min(a, b), max(a, b))
                if key not in edge_lookup:
                    continue
                self.outer_edges.append(
                    M1Edge(stone_a=key[0], stone_b=key[1], length_m=edge_lookup[key])
                )
                perim += edge_lookup[key]
            _log.info("Outer boundary: %d stones, %d edges, perimeter=%.2fm",
                      len(cycle), len(self.outer_edges), perim)

        if len(edges) < 3:
            self.outer_edges = list(edges)
            self.outer_stone_indices = list(
                {e.stone_a for e in edges} | {e.stone_b for e in edges}
            )
            return

        # Adjacency over the deduped edge set.
        adj: dict[int, set[int]] = {}
        for e in edges:
            adj.setdefault(e.stone_a, set()).add(e.stone_b)
            adj.setdefault(e.stone_b, set()).add(e.stone_a)

        # Prune dangling spurs: repeatedly drop degree-1 nodes and detach them.
        deg1 = [n for n, nb in adj.items() if len(nb) == 1]
        while deg1:
            n = deg1.pop()
            if n not in adj:
                continue
            for m in list(adj[n]):
                adj[m].discard(n)
                if len(adj[m]) == 1:
                    deg1.append(m)
            del adj[n]

        ring_nodes = {n for n, nb in adj.items() if nb}
        if len(ring_nodes) < 3:
            # No usable cycle survived -- honestly open/branched boundary.
            _log.warning("Boundary edges form no clean ring; using all edges unordered")
            self.outer_edges = list(edges)
            self.outer_stone_indices = list(
                {e.stone_a for e in edges} | {e.stone_b for e in edges}
            )
            return

        # Chain the surviving edges into one ordered walk.
        # Start at a node of minimal degree (a true ring node has degree 2).
        start = min(ring_nodes, key=lambda n: len(adj[n]))
        cycle = [start]
        visited = {start}
        current = start
        prev = -1
        while True:
            nbrs = [m for m in adj[current] if m in ring_nodes]
            # Prefer an unvisited neighbour; never immediately backtrack.
            candidates = [m for m in nbrs if m not in visited]
            if not candidates:
                # Close the ring if we can get back to the start; else stop.
                break
            # Deterministic, stable choice among candidates.
            nxt = min(candidates, key=lambda m: (m == prev, m))
            cycle.append(nxt)
            visited.add(nxt)
            prev = current
            current = nxt

        _finalize(cycle)


def extract_m1_dxf(dxf_path: str | Path) -> M1PlotData:
    """Parse an M1-produced DXF and return structured plot data.

    Parameters
    ----------
    dxf_path : path to the M1 DXF file

    Returns
    -------
    M1PlotData with stones, boundary edges, survey number, and raw geometry.
    """
    dxf_path = Path(dxf_path)
    _log.info("Extracting M1 data from %s", dxf_path)

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    data = M1PlotData(source_file=str(dxf_path))

    # --- Step 1: Extract survey number ---
    for e in _entities_on(msp, "TEXT", _L_SURVEY):
        txt = str(e.dxf.text).strip()
        if _SURVEY_NUM_RE.match(txt):
            data.survey_number = txt
            _log.info("Survey number: %s", data.survey_number)
            break

    # Fallback: parse survey number from filename
    if not data.survey_number:
        fname = dxf_path.stem
        nums = re.findall(r'(\d{2,5}(?:/\d+)?)', fname)
        if nums:
            data.survey_number = nums[-1]
            _log.info("Survey number (from filename): %s", data.survey_number)

    # --- Step 2: Extract stone positions from STONES layer ---
    for e in _entities_on(msp, "TEXT", _L_STONES):
        label = str(e.dxf.text).strip()
        x, y = e.dxf.insert.x, e.dxf.insert.y
        data.stones.append(M1Stone(x=x, y=y, label=label,
                                   index=len(data.stones)))
    _log.info("Extracted %d stones", len(data.stones))

    if not data.stones:
        _log.warning("No stones found on STONES layer")
        return data

    # --- Step 3: Collect boundary SEGMENTS (geometry) from BOUNDARY layer ---
    # We deliberately do NOT snap per-segment to stones here. INGUR boundaries
    # have curved/multi-vertex edges where many vertices are intermediate
    # (non-stone) points; snapping each segment to the nearest stone shatters
    # those edges. Instead we keep the raw segment geometry and let
    # ``extract_outer_boundary`` chain it into an ordered ring, then identify
    # corner stones and measure corner-to-corner path lengths.
    stone_arr = np.array([[s.x, s.y] for s in data.stones])

    boundary_polys = _entities_on(msp, "LWPOLYLINE", _L_BOUNDARY)
    _log.info("Processing %d BOUNDARY polylines", len(boundary_polys))

    for poly in boundary_polys:
        pts = [(p[0], p[1]) for p in poly.get_points()]
        if len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            if pts[i] != pts[i + 1]:
                data.boundary_segments.append((pts[i], pts[i + 1]))
        # Per-vertex stone association kept for warping.
        stone_idx_per_vert = []
        for pt in pts:
            d = np.sqrt(np.sum((stone_arr - np.array(pt)) ** 2, axis=1))
            si = int(np.argmin(d))
            stone_idx_per_vert.append(si if d[si] < _STONE_SNAP_DIST else -1)
        data.raw_boundary_polys.append(RawBoundaryPoly(
            vertices=pts,
            stone_indices=stone_idx_per_vert,
            stone_a=-1,
            stone_b=-1,
        ))

    _log.info("Collected %d boundary segments", len(data.boundary_segments))

    # --- Step 4: Extract other geometry layers for warping ---
    for e in _entities_on(msp, "LWPOLYLINE", _L_SUBDIV_LINES):
        pts = [(p[0], p[1]) for p in e.get_points()]
        if len(pts) >= 2:
            data.subdivision_verts.append(pts)

    for e in _entities_on(msp, "LWPOLYLINE", _L_CHAIN):
        pts = [(p[0], p[1]) for p in e.get_points()]
        if len(pts) >= 2:
            data.chain_verts.append(pts)

    for e in _entities_on(msp, "LWPOLYLINE", _L_SEPARATION):
        pts = [(p[0], p[1]) for p in e.get_points()]
        if len(pts) >= 2:
            data.separation_verts.append(pts)

    for e in _entities_on(msp, "LWPOLYLINE", _L_DASHED_REF):
        pts = [(p[0], p[1]) for p in e.get_points()]
        if len(pts) >= 2:
            data.dashed_ref_verts.append(pts)

    # Extract text entities for transformation
    for layer_name in (_L_BOUNDARY_DIM, _L_CHAIN_DIM, _L_DIM):
        for e in _entities_on(msp, "TEXT", layer_name):
            data.dimension_texts.append({
                "text": str(e.dxf.text).strip(),
                "x": e.dxf.insert.x,
                "y": e.dxf.insert.y,
                "layer": layer_name,
                "rotation": e.dxf.get("rotation", 0.0),
            })

    for e in _entities_on(msp, "TEXT", _L_NEIGHBOR):
        data.neighbor_label_texts.append({
            "text": str(e.dxf.text).strip(),
            "x": e.dxf.insert.x,
            "y": e.dxf.insert.y,
        })

    for e in _entities_on(msp, "TEXT", _L_SUBDIV):
        data.sub_plot_label_texts.append({
            "text": str(e.dxf.text).strip(),
            "x": e.dxf.insert.x,
            "y": e.dxf.insert.y,
        })

    # --- Step 5: Identify the outer boundary cycle ---
    data.extract_outer_boundary()

    return data
