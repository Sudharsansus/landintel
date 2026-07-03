"""Extract structured data from the surveyor's field-surveyed DXF.

Surveyor DXF structure (INGUR / Tamil Nadu tower-corridor convention):
  - POINT entities (624) on layer '0' with real UTM Zone 44N coordinates
  - TEXT entities (624) on 'Point_Code' layer (co-located labels)
  - Point codes: B (268), BS (185), RBS (61), VBS (5), RB (4), RS (1), T (41), W (6), FNC (41)
  - LWPOLYLINE (130) on 'SITE DATA LINE' -- measured chains connecting boundary stones
  - LWPOLYLINE (37) on 'TOWER AS PER DESIGN' -- tower footprint polygons (NOT plot boundaries)
  - TEXT (11) on 'FEATURE_LABEL' -- corridor approach points (AP80..AP84)
  - MTEXT (3) on layer '0' -- village name labels

CRITICAL INSIGHT: The surveyor DXF contains NO closed plot polygons. Instead,
property boundaries are traced as OPEN polyline chains (SITE DATA LINE) connecting
boundary stones (B/BS/RBS/VBS/RB/RS). Each chain follows the tower corridor and
may span multiple property boundaries consecutively. The M2 matcher must extract
consecutive sub-sequences from these open chains and match them against M1's
closed polygon boundaries.

This module produces:
  - SurveyorData.stones:  list of (x, y, code) for boundary-relevant points
  - SurveyorData.chains:  list of (stone_idx_a, stone_idx_b, length_m) edges
  - SurveyorData.polylines: raw polyline point sequences for topology analysis
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ezdxf
import numpy as np
from scipy.spatial import cKDTree

_log = logging.getLogger(__name__)

# Survey-number text on a corridor land schedule looks like "667", "773", "82/1".
_SCHED_NUM_RE = re.compile(r"^(\d{1,5})(?:/\d+)?[A-Za-z]?$")


def extract_corridor_surveys(
    schedule_dxf_path: str | Path,
    layer: str = "SURVEY NUMBER",
) -> set[str]:
    """Read the set of survey numbers a corridor crosses from a land schedule DXF.

    The raw surveyor DXF carries stones + traced lines but NO survey-number
    identity, so geometry alone cannot tell which FMB belongs on which corridor
    segment -- congruent plot shapes match the wrong seat (measured: 13 false
    positives on INGUR). The transmission-line LAND-COMPENSATION SCHEDULE (the
    client's working file) lists every survey number the line crosses on its
    ``SURVEY NUMBER`` layer. Returning that set lets M2 attempt ONLY the plots
    that are genuinely on the corridor -- the identity gate that removes the false
    positives. Survey numbers are normalised to their base integer string
    ("82/1" -> "82") to match FMB survey_no.

    Returns an empty set if the layer/file is absent (caller then runs ungated).
    """
    schedule_dxf_path = Path(schedule_dxf_path)
    try:
        msp = ezdxf.readfile(str(schedule_dxf_path)).modelspace()
    except Exception as exc:  # noqa: BLE001
        _log.warning("Could not read schedule DXF %s: %s", schedule_dxf_path, exc)
        return set()

    surveys: set[str] = set()
    for e in msp.query("TEXT MTEXT"):
        if e.dxf.layer != layer:
            continue
        raw = e.dxf.text if e.dxftype() == "TEXT" else e.text
        # Strip MTEXT formatting codes ({\fArial;...}) and braces.
        t = re.sub(r"\\[A-Za-z][^;]*;", "", str(raw)).replace("{", "").replace("}", "").strip()
        m = _SCHED_NUM_RE.match(t)
        if m:
            surveys.add(m.group(1))
    _log.info("Corridor schedule: %d distinct survey numbers from %s",
              len(surveys), schedule_dxf_path.name)
    return surveys

# Point codes that represent property boundary markers.
BOUNDARY_CODES = {"B", "BS", "RBS", "VBS", "RB", "RS"}


def _is_boundary_code(code: str) -> bool:
    """True if a stone code denotes a boundary marker -- GENERAL, no per-file list.

    Matches an exact boundary code OR any composite whose hyphen/space-separated tokens
    include one (e.g. 'VBS-RBS', 'B-X', 'BS-x'), so surveyor variants are covered while
    auxiliary points (T, FNC, CW, AP-104, GPS-2, ...) are excluded."""
    if not code:
        return False
    parts = [p.strip().upper() for p in code.replace(" ", "-").split("-")]
    return any(p in BOUNDARY_CODES for p in parts)

# Maximum distance (metres) to associate a TEXT label with a POINT entity.
_LABEL_ASSOC_DIST = 2.0


@dataclass
class SurveyorStone:
    """A single boundary stone with real UTM coordinates."""
    x: float
    y: float
    code: str          # B, BS, RBS, VBS, RB, RS
    index: int         # 0-based index in the stones list


@dataclass
class SurveyorChain:
    """An edge (measured chain segment) between two boundary stones."""
    stone_a: int       # index into stones list
    stone_b: int       # index into stones list
    length_m: float    # Euclidean distance in metres


@dataclass
class SurveyorPolyline:
    """Raw polyline from SITE DATA LINE with snapped stone indices."""
    raw_points: list[tuple[float, float]]   # original vertex coordinates
    stone_indices: list[int]                 # snapped stone index per vertex
    edge_lengths: list[float]                # length of each consecutive edge


@dataclass
class SurveyorData:
    """All extracted data from the surveyor's DXF."""
    stones: list[SurveyorStone] = field(default_factory=list)
    chains: list[SurveyorChain] = field(default_factory=list)
    polylines: list[SurveyorPolyline] = field(default_factory=list)
    crs: str = "EPSG:32643"   # UTM Zone 43N (INGUR / Erode, ~77.6E)
    source_file: str = ""

    # Fast lookup structures (built by build_index)
    _stone_tree: Optional[cKDTree] = field(default=None, repr=False)
    _stone_arr: Optional[np.ndarray] = field(default=None, repr=False)

    def build_index(self):
        """Build KD-tree for fast nearest-stone queries."""
        if not self.stones:
            return
        self._stone_arr = np.array([[s.x, s.y] for s in self.stones])
        self._stone_tree = cKDTree(self._stone_arr)

    def nearest_stone(self, x: float, y: float,
                      max_dist: float = 5.0) -> Optional[tuple[int, float]]:
        """Return (stone_index, distance) for the nearest stone, or None."""
        if self._stone_tree is None:
            self.build_index()
        dist, idx = self._stone_tree.query([x, y])
        if dist > max_dist:
            return None
        return int(idx), float(dist)

    def stone_coords(self, idx: int) -> tuple[float, float]:
        return self.stones[idx].x, self.stones[idx].y

    @property
    def stone_positions(self) -> np.ndarray:
        """(N, 2) array of all stone UTM positions."""
        return self._stone_arr if self._stone_arr is not None else np.empty((0, 2))

    @property
    def code_distribution(self) -> dict[str, int]:
        """Count of each boundary code."""
        dist = defaultdict(int)
        for s in self.stones:
            dist[s.code] += 1
        return dict(dist)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) of all stones."""
        if self._stone_arr is None or len(self._stone_arr) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        xs, ys = self._stone_arr[:, 0], self._stone_arr[:, 1]
        return (float(xs.min()), float(ys.min()),
                float(xs.max()), float(ys.max()))


def extract_surveyor(
    dxf_path: str | Path,
    bbox: tuple[float, float, float, float] | None = None,
) -> SurveyorData:
    """Parse the surveyor's DXF and return structured boundary data.

    Parameters
    ----------
    dxf_path : path to the surveyor DXF file
    bbox : optional (xmin, ymin, xmax, ymax) in the surveyor's own UTM metres. When
        given, ONLY stones inside it are kept -- the caller crops to the target
        village's window so a small plot is matched against its own village's stones,
        not a whole taluk's cloud (both a speed and a false-match reduction). General:
        the window is a parameter, never a per-village constant.

    Returns
    -------
    SurveyorData with all boundary stones, chain edges, and polylines populated.
    """
    dxf_path = Path(dxf_path)
    _log.info("Extracting surveyor data from %s", dxf_path)

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    data = SurveyorData(source_file=str(dxf_path))

    def _in_bbox(x: float, y: float) -> bool:
        return bbox is None or (bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3])

    # --- Step 1: Collect POINT entities and Point_Code TEXT labels ---
    raw_points: list[tuple[float, float]] = []
    for e in msp.query("POINT"):
        raw_points.append((e.dxf.location.x, e.dxf.location.y))

    raw_labels: list[tuple[float, float, str]] = []
    for e in msp.query('TEXT[layer=="Point_Code"]'):
        raw_labels.append((e.dxf.insert.x, e.dxf.insert.y,
                           str(e.dxf.text).strip()))

    if not raw_labels:
        _log.warning("No Point_Code TEXT entities found in surveyor DXF")
        return data

    code_counts = defaultdict(int)
    if raw_points:
        # --- POINT + nearest-label path (INGUR convention; unchanged) ---
        _log.info("Surveyor DXF: %d POINT entities, %d Point_Code labels",
                  len(raw_points), len(raw_labels))
        pt_arr = np.array(raw_points)
        label_arr = np.array([(lx, ly) for lx, ly, _ in raw_labels])
        label_tree = cKDTree(label_arr)
        dists, label_idxs = label_tree.query(pt_arr)
        for i, (px, py) in enumerate(raw_points):
            label = raw_labels[label_idxs[i]][2]
            if dists[i] > _LABEL_ASSOC_DIST or not _in_bbox(px, py):
                continue
            code_counts[label] += 1
            if _is_boundary_code(label):
                data.stones.append(
                    SurveyorStone(x=px, y=py, code=label, index=len(data.stones)))
    else:
        # --- TEXT-only path (no POINT entities): the Point_Code TEXT IS the stone,
        # positioned at its insert point with the code as its text. General fallback
        # for surveyor exports that omit POINT entities. ---
        _log.info("Surveyor DXF: no POINT entities; using %d Point_Code TEXT as stones",
                  len(raw_labels))
        for lx, ly, code in raw_labels:
            if not _in_bbox(lx, ly):
                continue
            code_counts[code] += 1
            if _is_boundary_code(code):
                data.stones.append(
                    SurveyorStone(x=lx, y=ly, code=code, index=len(data.stones)))

    _log.info("Point code distribution (top): %s",
              dict(sorted(code_counts.items(), key=lambda kv: -kv[1])[:10]))
    _log.info("Extracted %d boundary stones (B/BS/RBS/VBS/RB/RS)%s",
              len(data.stones), f" within bbox {bbox}" if bbox else "")

    if not data.stones:
        _log.warning("No boundary stones found after filtering")
        return data

    # --- Step 3: Build spatial index ---
    data.build_index()

    # --- Step 4: Extract SITE DATA LINE polylines ---
    seen_edges: set[tuple[int, int]] = set()
    site_polys = list(msp.query('LWPOLYLINE[layer=="SITE DATA LINE"]'))
    _log.info("Processing %d SITE DATA LINE polylines", len(site_polys))

    for poly in site_polys:
        pts = list(poly.get_points())
        if len(pts) < 2:
            continue

        # Snap each polyline vertex to nearest boundary stone
        pts_arr = np.array([(p[0], p[1]) for p in pts])
        snap_dists, snap_idxs = data._stone_tree.query(pts_arr)

        # Build chain edges and store polyline data
        raw_pts = [(p[0], p[1]) for p in pts]
        stone_idx_list = [int(i) for i in snap_idxs]
        edge_lengths = []

        prev_stone = None
        for i, stone_idx in enumerate(stone_idx_list):
            if prev_stone is not None and stone_idx != prev_stone:
                edge_key = (min(prev_stone, stone_idx),
                            max(prev_stone, stone_idx))
                if edge_key not in seen_edges:
                    sa = data.stones[prev_stone]
                    sb = data.stones[stone_idx]
                    length = math.sqrt((sa.x - sb.x) ** 2 +
                                       (sa.y - sb.y) ** 2)
                    data.chains.append(SurveyorChain(
                        stone_a=edge_key[0],
                        stone_b=edge_key[1],
                        length_m=length,
                    ))
                    seen_edges.add(edge_key)
                # Compute edge length for polyline record
                sa = data.stones[prev_stone]
                sb = data.stones[stone_idx]
                edge_lengths.append(
                    math.sqrt((sa.x - sb.x) ** 2 + (sa.y - sb.y) ** 2)
                )
            prev_stone = stone_idx

        data.polylines.append(SurveyorPolyline(
            raw_points=raw_pts,
            stone_indices=stone_idx_list,
            edge_lengths=edge_lengths,
        ))

    _log.info("Extracted %d unique chain edges from %d polylines",
              len(data.chains), len(data.polylines))
    _log.info("Stone extent: %s", data.extent)

    return data
