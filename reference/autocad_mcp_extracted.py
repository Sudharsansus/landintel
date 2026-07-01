"""
autocad_mcp_extracted.py — Useful patterns & algorithms extracted from autocad-mcp (MIT) for M2 pipeline.

Extracted and adapted from: https://github.com/puran-water/autocad-mcp/
License: MIT (inherited from autocad-mcp)

Original source files extracted and ported:
  - src/autocad_mcp/backends/ezdxf_backend.py   (749 lines — DXF entity I/O + transforms)
  - src/autocad_mcp/backends/file_ipc.py         (498 lines — IPC protocol + LISP dispatch)
  - src/autocad_mcp/backends/base.py             (283 lines — abstract backend + capabilities)
  - src/autocad_mcp/server.py                    (551 lines — MCP tool definitions)
  - src/autocad_mcp/screenshot.py                (207 lines — DXF→PNG rendering)
  - src/autocad_mcp/config.py                    (114 lines — backend config)
  - lisp-code/mcp_dispatch.lsp                   (1379 lines — AutoLISP command whitelist)
  - tests/test_ezdxf_backend.py                  (638 lines — test patterns)
  - tests/test_ipc_protocol.py                   (539 lines — protocol test patterns)

Categories of utilities extracted:
  1. DXF ENTITY I/O — Programmatic read/create/modify DXF entities (ezdxf patterns)
  2. GEOMETRIC TRANSFORM MATRICES — 4x4 matrix rotation, scale, mirror around arbitrary points
  3. 2D MIRROR REFLECTION — Custom reflection matrix across arbitrary line (pure math)
  4. ENTITY PROPERTY EXTRACTION — Extract coords, radii, angles from DXF entities
  5. DIMENSION ENTITY HANDLING — Linear, aligned, angular, radius dimension parsing/creation
  6. HATCH BOUNDARY PATTERNS — Hatch creation with boundary polylines
  7. BLOCK INSERTION WITH ATTRIBUTES — Block references with tag-value attribute data
  8. LAYER MANAGEMENT — Color/linetype/lineweight system and ACI color mapping
  9. ORTHOGONAL ROUTING — Manhattan-style path generation for pipe/process lines
 10. DXF→PNG RENDERING — Headless DXF visualization via ezdxf + matplotlib
 11. IPC PROTOCOL PATTERN — File-based inter-process communication for AutoCAD
 12. COMMAND DISPATCH PATTERN — Operation-based entity management architecture
 13. CAPABILITIES SYSTEM — Feature-detection for multi-backend abstraction
 14. ENTITY QUERY PATTERNS — List, count, filter entities by layer/type
 15. ERROR HANDLING PATTERNS — Safe decorator + CommandResult envelope

What is NOT here (delegated to ezdxf/numpy/shapely in M2):
  - No DWG binary reading (autocad-mcp delegates to AutoCAD via IPC)
  - No vector math library (uses only math.atan2/cos/sin/hypot — see librecad_extracted.py)
  - No polygon boolean ops (see librecad_extracted.py or shapely)
  - No heavy geometric computation (all delegates to ezdxf Matrix44 or AutoCAD commands)

Relationship to librecad_extracted.py:
  - librecad_extracted.py = low-level math (vectors, angles, intersections, convex hull,
    polynomial solvers, loop topology, DXF group codes)
  - THIS file = high-level DXF I/O patterns (entity creation, transforms, rendering,
    dimensions, hatches, blocks, attributes, IPC protocol, capabilities)
  - Together they provide complete DXF handling for the M2 FMB pipeline
"""

from __future__ import annotations

import math
import os
import json
import uuid
import time
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import (
    List, Optional, Tuple, Dict, Any, Union,
    Callable, Sequence, NamedTuple
)
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 1. ACI COLOR INDEX MAPPING (from ezdxf_backend.py)
# ============================================================
# AutoCAD Color Index — maps human-readable names to integer ACI codes.
# Used in DXF entity creation and layer color assignment.
# M2 usage: FMB DXF files use ACI colors; this map helps interpret them.

ACI_COLOR_MAP: Dict[str, int] = {
    "red": 1, "yellow": 2, "green": 3, "cyan": 4,
    "blue": 5, "magenta": 6, "white": 7,
    "grey": 8, "gray": 8,
    "light_grey": 9, "light_gray": 9,
}
ACI_COLOR_MAP_REVERSE: Dict[int, str] = {v: k for k, v in ACI_COLOR_MAP.items()}


def aci_to_rgb(aci: int) -> Tuple[int, int, int]:
    """Convert AutoCAD Color Index to approximate RGB tuple.
    Covers the standard 9 ACI colors (0-8). For ACI > 8, returns grey.
    M2 usage: interpreting entity colors from FMB DXF for boundary detection."""
    palette = {
        0: (0, 0, 0),         # BYBLOCK
        1: (255, 0, 0),       # Red
        2: (255, 255, 0),     # Yellow
        3: (0, 255, 0),       # Green
        4: (0, 255, 255),     # Cyan
        5: (0, 0, 255),       # Blue
        6: (255, 0, 255),     # Magenta
        7: (255, 255, 255),   # White/Black (depends on background)
        8: (128, 128, 128),   # Grey
        9: (192, 192, 192),   # Light Grey
    }
    return palette.get(aci, (128, 128, 128))


def color_name_to_aci(name: str) -> Optional[int]:
    """Map human-readable color name to ACI code. Returns None if unknown.
    M2 usage: setting entity/layer colors programmatically."""
    return ACI_COLOR_MAP.get(name.lower().strip())


# ============================================================
# 2. COMMAND RESULT ENVELOPE (from base.py)
# ============================================================
# Uniform result wrapper for all operations. Used by the backend system
# to provide consistent success/error handling across different backends.
# M2 usage: wrapping FMB parsing results with ok/error metadata.

@dataclass
class CommandResult:
    """Uniform operation result envelope (from autocad-mcp backends/base.py).
    Every operation returns this, making error handling deterministic."""
    ok: bool
    payload: Any = None
    error: Optional[str] = None

    @staticmethod
    def success(payload: Any = None) -> "CommandResult":
        return CommandResult(ok=True, payload=payload)

    @staticmethod
    def failure(error: str) -> "CommandResult":
        return CommandResult(ok=False, error=error)

    def unwrap(self) -> Any:
        """Get payload or raise RuntimeError. M2 usage: assert on critical ops."""
        if not self.ok:
            raise RuntimeError(f"CommandResult failure: {self.error}")
        return self.payload


# ============================================================
# 3. SAFE DECORATOR PATTERN (from server.py @_safe)
# ============================================================
# Wraps every tool/handler with try/except, converting unhandled exceptions
# into actionable error messages. Prevents MCP server crashes.
# M2 usage: wrapping FMB DXF parsing and georeferencing steps.

def safe_handler(func: Callable) -> Callable:
    """Error-safe decorator (from autocad-mcp server.py @_safe).
    Catches all exceptions and returns CommandResult.failure with hints."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            # Add actionable hints for known error patterns
            hints = {
                "Entity not found": "Check entity_id exists in the drawing",
                "Layer not found": "Verify layer name spelling",
                "Block not found": "Ensure block is defined before insertion",
                "Invalid handle": "Entity handle may be corrupted in DXF",
            }
            for pattern, hint in hints.items():
                if pattern.lower() in error_msg.lower():
                    error_msg += f" — Hint: {hint}"
                    break
            logger.error("safe_handler caught", func=func.__name__, error=error_msg)
            return CommandResult.failure(error_msg)
    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    return wrapper


# ============================================================
# 4. BACKEND CAPABILITIES SYSTEM (from base.py)
# ============================================================
# Feature-detection system for multi-backend abstraction.
# M2 usage: detecting whether ezdxf or other DXF backends support
# specific operations (e.g., can the backend modify entities? save files?).

@dataclass
class BackendCapabilities:
    """Feature-detection flags for DXF backends (from autocad-mcp).
    M2: use this to abstract over different DXF processing backends
    (ezdxf headless, libredxf, OGR, etc.)"""
    can_read_drawing: bool = False
    can_modify_entities: bool = False
    can_create_entities: bool = True
    can_screenshot: bool = False
    can_save: bool = False
    can_plot_pdf: bool = False
    can_zoom: bool = False
    can_query_entities: bool = False
    can_file_operations: bool = False
    can_undo: bool = False

    def supports(self, operation: str) -> bool:
        """Check if a named operation is supported."""
        return getattr(self, f"can_{operation}", False)


# ============================================================
# 5. 4x4 TRANSFORMATION MATRIX (from ezdxf_backend.py patterns)
# ============================================================
# Implements 4x4 homogeneous transformation matrices for 2D/3D geometric
# operations. Ported from ezdxf's Matrix44 usage patterns.
# M2 usage: rigid transformation (rotation + translation + uniform scale)
# for fitting FMB polygons to georeferenced tile coordinates.

class Mat4:
    """4x4 homogeneous transformation matrix (ported from ezdxf Matrix44 patterns).

    Used in autocad-mcp for all entity transforms (rotate, scale, mirror).
    Reimplemented in pure numpy for M2 — no ezdxf dependency for core math.

    M2 critical usage: RIGID TRANSFORMATION for FMB→geospatial fitting.
    Given: FMB corner points (relative meters) + target UTM points.
    Compute: optimal (R, s, t) that minimizes corner residuals.
    Apply: transform all FMB boundary points to geospatial coordinates.
    """

    def __init__(self, matrix: Optional[np.ndarray] = None):
        if matrix is not None:
            self.m = np.array(matrix, dtype=np.float64).reshape(4, 4)
        else:
            self.m = np.eye(4, dtype=np.float64)

    @staticmethod
    def identity() -> "Mat4":
        return Mat4(np.eye(4))

    @staticmethod
    def translate(dx: float, dy: float, dz: float = 0.0) -> "Mat4":
        """Translation matrix. M2: shift FMB to UTM origin."""
        m = np.eye(4)
        m[0, 3] = dx
        m[1, 3] = dy
        m[2, 3] = dz
        return Mat4(m)

    @staticmethod
    def scale(sx: float, sy: float, sz: float = 1.0) -> "Mat4":
        """Scaling matrix. M2: uniform scale for FMB meter→UTM meter."""
        m = np.eye(4)
        m[0, 0] = sx
        m[1, 1] = sy
        m[2, 2] = sz
        return Mat4(m)

    @staticmethod
    def z_rotate(angle_rad: float) -> "Mat4":
        """Z-axis rotation matrix. M2: rotate FMB to match tile orientation."""
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        m = np.eye(4)
        m[0, 0] = c;  m[0, 1] = -s
        m[1, 0] = s;  m[1, 1] = c
        return Mat4(m)

    @staticmethod
    def y_rotate(angle_rad: float) -> "Mat4":
        """Y-axis rotation matrix."""
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        m = np.eye(4)
        m[0, 0] = c;  m[0, 2] = s
        m[2, 0] = -s; m[2, 2] = c
        return Mat4(m)

    @staticmethod
    def x_rotate(angle_rad: float) -> "Mat4":
        """X-axis rotation matrix."""
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        m = np.eye(4)
        m[1, 1] = c;  m[1, 2] = -s
        m[2, 1] = s;  m[2, 2] = c
        return Mat4(m)

    @staticmethod
    def rotate_around_point(cx: float, cy: float, angle_rad: float) -> "Mat4":
        """Rotation around arbitrary 2D point (from ezdxf_backend.py entity_rotate pattern).
        Pattern: translate(-center) → rotate → translate(+center).
        M2 usage: rotate FMB polygon around its centroid before fitting."""
        return (
            Mat4.translate(cx, cy)
            * Mat4.z_rotate(angle_rad)
            * Mat4.translate(-cx, -cy)
        )

    @staticmethod
    def scale_around_point(cx: float, cy: float, factor: float) -> "Mat4":
        """Uniform scaling around arbitrary 2D point (from ezdxf_backend.py entity_scale).
        Pattern: translate(-center) → scale → translate(+center).
        M2 usage: scale FMB polygon around centroid."""
        return (
            Mat4.translate(cx, cy)
            * Mat4.scale(factor, factor, factor)
            * Mat4.translate(-cx, -cy)
        )

    @staticmethod
    def mirror_across_line(
        x1: float, y1: float, x2: float, y2: float
    ) -> "Mat4":
        """2D mirror reflection across a line defined by two points.

        From autocad-mcp ezdxf_backend.py entity_mirror — custom 2D reflection
        matrix. Formula: M = [[cos2α, sin2α], [sin2α, -cos2α]] where
        α = atan2(dy, dx) of the mirror line. Then translate to line origin
        and back.

        M2 usage: validating FMB symmetry, detecting flipped survey drawings,
        generating mirrored parcel boundaries."""
        dx, dy = x2 - x1, y2 - y1
        a = math.atan2(dy, dx)
        cos2a = math.cos(2 * a)
        sin2a = math.sin(2 * a)
        # 2D reflection embedded in 4x4 homogeneous matrix
        m = np.eye(4)
        m[0, 0] = cos2a;  m[0, 1] = sin2a
        m[1, 0] = sin2a;  m[1, 1] = -cos2a
        return (
            Mat4(m)
            * Mat4.translate(x1, y1)
            * Mat4.translate(-x1, -y1)
        )

    def __mul__(self, other: "Mat4") -> "Mat4":
        """Matrix composition (self @ other in application order)."""
        return Mat4(self.m @ other.m)

    def transform_point(self, x: float, y: float, z: float = 0.0) -> Tuple[float, float, float]:
        """Transform a single point through this matrix."""
        p = np.array([x, y, z, 1.0])
        r = self.m @ p
        return (r[0], r[1], r[2])

    def transform_points(
        self, points: np.ndarray
    ) -> np.ndarray:
        """Transform Nx2 or Nx3 array of points.

        M2 usage: batch-transform all FMB corner/boundary points through
        the fitted rigid transformation to get georeferenced coordinates.
        Input shape: (N, 2) or (N, 3). Output shape matches input."""
        pts = np.atleast_2d(points)
        n_dims = pts.shape[1]
        if n_dims == 2:
            ones = np.ones((pts.shape[0], 1))
            pts_h = np.hstack([pts, ones, ones])  # (N, 4) — z=0, w=1
        else:
            ones = np.ones((pts.shape[0], 1))
            pts_h = np.hstack([pts, ones])  # (N, 4)
        result = (self.m @ pts_h.T).T
        if n_dims == 2:
            return result[:, :2]
        return result[:, :3]

    def to_numpy(self) -> np.ndarray:
        return self.m.copy()

    def __repr__(self) -> str:
        return f"Mat4(\n{self.m})"


# ============================================================
# 6. RIGID + SIMILARITY TRANSFORMATION SOLVER (M2-custom, inspired by patterns)
# ============================================================
# Solves the optimal rigid (R+t) or similarity (sR+t) transformation that
# maps source points to target points using SVD.
# This is the CORE operation for M2 FMB→geospatial fitting.
#
# autocad-mcp uses ezdxf's Matrix44 for individual entity transforms.
# M2 needs to SOLVE for the optimal transform parameters given
# corresponding point pairs (FMB corners ↔ UTM coordinates from tiles).

def solve_rigid_transform(
    src: np.ndarray, tgt: np.ndarray
) -> Tuple[Mat4, float]:
    """Solve optimal rigid transformation (rotation + translation, no scale).

    Minimizes sum of ||R @ src[i] + t - tgt[i]||^2 using SVD.
    Uses the Kabsch/Umeyama algorithm (orthogonal Procrustes).

    Args:
        src: (N, 2) or (N, 3) source points (FMB corners, relative coords)
        tgt: (N, 2) or (N, 3) target points (UTM coords from tile OCR)

    Returns:
        (transform_matrix, residual_rms) — 4x4 homogeneous matrix + fit quality

    M2 usage: Fitting FMB polygon to UTM tile coordinates.
    The 'scale' between FMB (meters) and UTM (meters) should be ~1.0.
    If scale differs significantly, use solve_similarity_transform instead.
    """
    src = np.atleast_2d(src).astype(np.float64)
    tgt = np.atleast_2d(tgt).astype(np.float64)
    assert src.shape == tgt.shape and src.shape[0] >= 2

    # Center both point sets
    src_centroid = src.mean(axis=0)
    tgt_centroid = tgt.mean(axis=0)
    src_centered = src - src_centroid
    tgt_centered = tgt - tgt_centroid

    # Cross-covariance matrix
    H = src_centered.T @ tgt_centered  # (2,2) or (3,3)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation (det = +1, not reflection)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.eye(len(src_centroid))
    sign_matrix[-1, -1] = np.sign(d) if d != 0 else 1.0

    # Rotation matrix
    R = Vt.T @ sign_matrix @ U.T

    # Translation
    t = tgt_centroid - R @ src_centroid

    # Build 4x4 homogeneous matrix
    m = np.eye(4)
    dims = len(src_centroid)
    m[:dims, :dims] = R
    m[:dims, 3] = t

    # Compute residual RMS
    transformed = (R @ src.T).T + t
    residuals = np.linalg.norm(transformed - tgt, axis=1)
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    return Mat4(m), rms


def solve_similarity_transform(
    src: np.ndarray, tgt: np.ndarray
) -> Tuple[Mat4, float, float]:
    """Solve optimal similarity transformation (rotation + uniform scale + translation).

    Minimizes sum of ||s * R @ src[i] + t - tgt[i]||^2 using Umeyama (1991).

    Args:
        src: (N, 2) source points (FMB corners)
        tgt: (N, 2) target points (UTM coords)

    Returns:
        (transform_matrix, scale_factor, residual_rms)

    M2 usage: Primary FMB fitting method. The scale factor should be close
    to 1.0 (both FMB and UTM are in meters). If scale is outside [0.8, 1.25],
    the fit is likely wrong (M2 quality gate rejects it).
    """
    src = np.atleast_2d(src).astype(np.float64)
    tgt = np.atleast_2d(tgt).astype(np.float64)
    assert src.shape == tgt.shape and src.shape[0] >= 2

    n = src.shape[0]

    # Centroids
    src_centroid = src.mean(axis=0)
    tgt_centroid = tgt.mean(axis=0)
    src_c = src - src_centroid
    tgt_c = tgt - tgt_centroid

    # Variance of source points
    src_var = np.sum(src_c ** 2) / n

    # Cross-covariance
    H = src_c.T @ tgt_c

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Proper rotation guard
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.eye(src.shape[1])
    sign_matrix[-1, -1] = np.sign(d) if d != 0 else 1.0

    # Rotation
    R = Vt.T @ sign_matrix @ U.T

    # Scale (Umeyama formula)
    scale = np.trace(np.diag(S) @ sign_matrix) / src_var if src_var > 1e-12 else 1.0

    # Translation
    t = tgt_centroid - scale * (R @ src_centroid)

    # Build matrix
    m = np.eye(4)
    m[:2, :2] = scale * R
    m[:2, 3] = t

    # Residual RMS
    transformed = scale * (R @ src.T).T + t
    residuals = np.linalg.norm(transformed - tgt, axis=1)
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    return Mat4(m), float(scale), rms


# ============================================================
# 7. DXF ENTITY PROPERTY EXTRACTION (from ezdxf_backend.py entity_get patterns)
# ============================================================
# Patterns for extracting geometric properties from DXF entities.
# In autocad-mcp, entity_get extracts type-specific data for LINE and CIRCLE.
# M2 extends this for all entity types found in FMB DXF files.

class DXFEntityType(Enum):
    """Entity types relevant to M2 FMB DXF parsing."""
    LINE = "LINE"
    LWPOLYLINE = "LWPOLYLINE"
    POLYLINE_2D = "POLYLINE"
    POINT = "POINT"
    CIRCLE = "CIRCLE"
    ARC = "ARC"
    ELLIPSE = "ELLIPSE"
    TEXT = "TEXT"
    MTEXT = "MTEXT"
    DIMENSION = "DIMENSION"
    INSERT = "INSERT"
    HATCH = "HATCH"
    SOLID = "SOLID"
    SPLINE = "SPLINE"


@dataclass
class EntityInfo:
    """Extracted entity properties (from autocad-mcp entity_get pattern).
    M2: returned by DXF parsing to represent a single FMB entity."""
    handle: str
    entity_type: str
    layer: str
    color: Optional[int] = None  # ACI color index
    linetype: Optional[str] = None
    # Type-specific geometry
    start: Optional[Tuple[float, float]] = None   # LINE start point
    end: Optional[Tuple[float, float]] = None     # LINE end point
    center: Optional[Tuple[float, float]] = None  # CIRCLE/ARC center
    radius: Optional[float] = None                # CIRCLE/ARC radius
    points: Optional[List[Tuple[float, float]]] = None  # LWPOLYLINE vertices
    closed: Optional[bool] = None                 # LWPOLYLINE closed flag
    bulges: Optional[List[float]] = None          # LWPOLYLINE bulge values
    text: Optional[str] = None                    # TEXT/MTEXT content
    text_height: Optional[float] = None           # TEXT height
    text_position: Optional[Tuple[float, float]] = None  # TEXT insertion point
    # ARC-specific
    start_angle: Optional[float] = None
    end_angle: Optional[float] = None
    # DIMENSION-specific
    dim_type: Optional[int] = None
    dim_def_point: Optional[Tuple[float, float]] = None
    dim_text_mid: Optional[Tuple[float, float]] = None
    # INSERT-specific
    block_name: Optional[str] = None
    insert_point: Optional[Tuple[float, float]] = None
    scale_x: Optional[float] = None
    scale_y: Optional[float] = None
    rotation: Optional[float] = None
    attributes: Optional[Dict[str, str]] = None


def extract_entity_info(entity: Any) -> EntityInfo:
    """Extract geometric properties from an ezdxf entity object.

    From autocad-mcp ezdxf_backend.py entity_get pattern, extended for M2.
    Handles LINE, CIRCLE, ARC, LWPOLYLINE, TEXT, MTEXT, DIMENSION, INSERT.

    M2 usage: parsing FMB DXF files to extract parcel boundary polylines,
    survey number labels (TEXT/MTEXT), and measurement annotations (DIMENSION).
    """
    info = EntityInfo(
        handle=str(entity.dxf.handle) if hasattr(entity.dxf, 'handle') else "",
        entity_type=entity.dxftype(),
        layer=entity.dxf.layer,
    )

    # Color (0 = BYBLOCK, 256 = BYLAYER)
    if hasattr(entity.dxf, 'color'):
        info.color = entity.dxf.color

    # Entity type-specific extraction
    etype = entity.dxftype()

    if etype == "LINE":
        info.start = (float(entity.dxf.start.x), float(entity.dxf.start.y))
        info.end = (float(entity.dxf.end.x), float(entity.dxf.end.y))

    elif etype == "CIRCLE":
        info.center = (float(entity.dxf.center.x), float(entity.dxf.center.y))
        info.radius = float(entity.dxf.radius)

    elif etype == "ARC":
        info.center = (float(entity.dxf.center.x), float(entity.dxf.center.y))
        info.radius = float(entity.dxf.radius)
        info.start_angle = float(entity.dxf.start_angle)
        info.end_angle = float(entity.dxf.end_angle)

    elif etype == "LWPOLYLINE":
        pts = []
        bulges = []
        with entity.points() as iter_pts:
            for pt_data in iter_pts:
                pts.append((float(pt_data[0]), float(pt_data[1])))
                # Bulge is stored per-vertex in LWPOLYLINE
                if len(pt_data) > 2:
                    bulges.append(float(pt_data[2]))
                else:
                    bulges.append(0.0)
        info.points = pts
        info.bulges = bulges
        info.closed = bool(entity.dxf.get('closed', False))

    elif etype == "TEXT":
        info.text = entity.dxf.text
        info.text_height = float(entity.dxf.height) if hasattr(entity.dxf, 'height') else None
        info.text_position = (float(entity.dxf.insert.x), float(entity.dxf.insert.y))
        if hasattr(entity.dxf, 'rotation'):
            info.rotation = float(entity.dxf.rotation)

    elif etype == "MTEXT":
        info.text = entity.text
        info.text_height = float(entity.dxf.char_height) if hasattr(entity.dxf, 'char_height') else None
        info.text_position = (float(entity.dxf.insert.x), float(entity.dxf.insert.y))

    elif etype == "DIMENSION":
        info.dim_type = entity.dxf.dimtype
        if hasattr(entity.dxf, 'defpoint'):
            pt = entity.dxf.defpoint
            info.dim_def_point = (float(pt.x), float(pt.y))
        if hasattr(entity.dxf, 'text_midpoint'):
            pt = entity.dxf.text_midpoint
            info.dim_text_mid = (float(pt.x), float(pt.y))

    elif etype == "INSERT":
        info.block_name = entity.dxf.name
        info.insert_point = (float(entity.dxf.insert.x), float(entity.dxf.insert.y))
        if hasattr(entity.dxf, 'xscale'):
            info.scale_x = float(entity.dxf.xscale)
        if hasattr(entity.dxf, 'yscale'):
            info.scale_y = float(entity.dxf.yscale)
        if hasattr(entity.dxf, 'rotation'):
            info.rotation = float(entity.dxf.rotation)
        # Extract attributes (block tag-value pairs)
        attrs = {}
        for attrib in entity.attribs if hasattr(entity, 'attribs') else []:
            tag = attrib.dxf.tag if hasattr(attrib.dxf, 'tag') else str(attrib)
            val = attrib.dxf.text if hasattr(attrib.dxf, 'text') else ""
            attrs[tag] = val
        if attrs:
            info.attributes = attrs

    return info


# ============================================================
# 8. DXF ENTITY QUERY PATTERNS (from ezdxf_backend.py entity_list/entity_count)
# ============================================================
# Patterns for querying entities by layer, type, bounding box.
# M2 usage: filtering FMB DXF entities to find boundary polylines,
# survey labels, dimension annotations on specific layers.

def query_entities(
    msp: Any,
    layer: Optional[str] = None,
    entity_type: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> List[Any]:
    """Query entities from model space with optional filters.

    From autocad-mcp ezdxf_backend.py entity_list pattern.
    Filters by layer name, entity type, and/or bounding box.

    Args:
        msp: ezdxf model space object (doc.modelspace())
        layer: optional layer name filter
        entity_type: optional DXF type filter (e.g., "LWPOLYLINE", "TEXT")
        bbox: optional (x_min, y_min, x_max, y_max) bounding box

    Returns:
        List of matching entities

    M2 usage: Extracting boundary polylines from FMB DXF:
        boundaries = query_entities(msp, entity_type="LWPOLYLINE")
        labels = query_entities(msp, entity_type="TEXT")
    """
    results = []
    for e in msp:
        if layer and e.dxf.layer != layer:
            continue
        if entity_type and e.dxftype() != entity_type:
            continue
        if bbox:
            # Check if entity has a bounding box that intersects
            try:
                eb = e.bbox()
                if eb.extmin.x > bbox[2] or eb.extmax.x < bbox[0]:
                    continue
                if eb.extmin.y > bbox[3] or eb.extmax.y < bbox[1]:
                    continue
            except (AttributeError, Exception):
                # Entity has no bbox method, include it
                pass
        results.append(e)
    return results


def count_entities(msp: Any, layer: Optional[str] = None) -> Dict[str, int]:
    """Count entities by type, optionally filtered by layer.
    From autocad-mcp ezdxf_backend.py entity_count pattern.
    M2 usage: quick survey of FMB DXF content (how many boundaries, labels, etc.)"""
    counts: Dict[str, int] = {}
    for e in msp:
        if layer and e.dxf.layer != layer:
            continue
        etype = e.dxftype()
        counts[etype] = counts.get(etype, 0) + 1
    return counts


# ============================================================
# 9. DIMENSION ENTITY HANDLING (from ezdxf_backend.py create_dimension_*)
# ============================================================
# Dimension entities in DXF encode survey measurements (distances, angles, radii).
# M2 usage: extracting measurement annotations from FMB DXF to validate
# polygon side lengths against recorded survey dimensions.

@dataclass
class DimensionData:
    """Parsed dimension entity data (from autocad-mcp dimension patterns).
    M2: represents a measurement annotation from FMB DXF."""
    dim_type: str  # "linear", "aligned", "angular", "radius"
    # For linear/aligned: two defining points + dimension line position
    p1: Optional[Tuple[float, float]] = None
    p2: Optional[Tuple[float, float]] = None
    dim_line_point: Optional[Tuple[float, float]] = None
    # For angular: center + two points
    center: Optional[Tuple[float, float]] = None
    # For radius: center + radius + angle
    radius: Optional[float] = None
    angle: Optional[float] = None
    # Computed measurement value
    measured_value: Optional[float] = None


def parse_dimension(entity: Any) -> DimensionData:
    """Parse a DXF DIMENSION entity into structured data.

    From autocad-mcp ezdxf_backend.py dimension creation patterns (reversed).
    Handles linear, aligned, angular, and radius dimensions.

    M2 usage: extracting survey measurements from FMB dimension annotations
    to validate polygon fitting quality.
    """
    data = DimensionData(dim_type="unknown")
    dimtype = entity.dxf.dimtype

    if dimtype in (0, 1):  # Linear or Aligned
        data.dim_type = "aligned" if dimtype == 1 else "linear"
        # First extension line point
        x2, y2, _ = entity.dxf.xline2point  # (x, y, z)
        x1, y1, _ = entity.dxf.xline1point
        data.p1 = (float(x1), float(y1))
        data.p2 = (float(x2), float(y2))
        # Dimension line midpoint
        dm = entity.dxf.dimtext_midpoint
        data.dim_line_point = (float(dm.x), float(dm.y))
        # Compute measured distance
        dx, dy = data.p2[0] - data.p1[0], data.p2[1] - data.p1[1]
        data.measured_value = math.hypot(dx, dy)

    elif dimtype == 2:  # Angular
        data.dim_type = "angular"
        # Angular dimensions store center and two defining points
        dc = entity.dxf.defpoint
        data.center = (float(dc.x), float(dc.y))

    elif dimtype == 3:  # Radius
        data.dim_type = "radius"
        dc = entity.dxf.defpoint
        data.center = (float(dc.x), float(dc.y))
        # Radius dimension stores the leader endpoint
        if hasattr(entity.dxf, 'text_midpoint'):
            tm = entity.dxf.text_midpoint
            dx, dy = float(tm.x) - data.center[0], float(tm.y) - data.center[1]
            data.radius = math.hypot(dx, dy)
            data.angle = math.atan2(dy, dx)
            data.measured_value = data.radius

    return data


def create_linear_dimension_points(
    x1: float, y1: float, x2: float, y2: float, offset: float
) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
    """Compute placement point for an aligned dimension.

    From autocad-mcp ezdxf_backend.py create_dimension_aligned pattern.
    Given two points and an offset distance, computes the dimension line position.

    Args:
        x1, y1: First extension line origin
        x2, y2: Second extension line origin
        offset: Perpendicular offset from the measured line

    Returns:
        (dim_x, dim_y, p1, p2) — dimension line midpoint and original points

    M2 usage: placing dimension annotations on georeferenced FMB output DXF.
    """
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-12:
        return ((x1 + x2) / 2, (y1 + y2) / 2, (x1, y1), (x2, y2))

    # Perpendicular direction (rotated 90° CCW)
    nx, ny = -dy / length, dx / length
    # Dimension line midpoint
    mx, my = (x1 + x2) / 2 + nx * offset, (y1 + y2) / 2 + ny * offset
    return (mx, my, (x1, y1), (x2, y2))


def create_angular_dimension_points(
    cx: float, cy: float, x1: float, y1: float, x2: float, y2: float
) -> Tuple[float, float, float, float]:
    """Compute arc parameters for an angular dimension.

    From autocad-mcp ezdxf_backend.py create_dimension_angular pattern.
    Uses atan2 to find start/end angles and computes placement radius.

    M2 usage: annotating angle measurements in FMB output DXF.
    """
    a1 = math.atan2(y1 - cy, x1 - cx)
    a2 = math.atan2(y2 - cy, x2 - cx)
    amid = (a1 + a2) / 2
    r = max(
        math.hypot(x1 - cx, y1 - cy),
        math.hypot(x2 - cx, y2 - cy)
    ) * 0.7
    return (a1, a2, amid, r)


# ============================================================
# 10. HATCH BOUNDARY PATTERNS (from ezdxf_backend.py create_hatch)
# ============================================================
# Hatch entities in DXF represent filled areas. In FMB, hatches may indicate
# land use, survey classification, or parcel highlighting.
# M2 usage: creating visual overlays on georeferenced output DXF.

def create_hatch_with_boundary(
    msp: Any,
    boundary_points: List[Tuple[float, float]],
    pattern: str = "ANSI31",
    scale: float = 1.0,
    layer: str = "0",
    color: int = 7,
) -> Any:
    """Create a hatch entity with a polyline boundary.

    From autocad-mcp ezdxf_backend.py create_hatch pattern.
    Uses ezdxf's hatch API with path + edge construction.

    Args:
        msp: ezdxf model space
        boundary_points: List of (x, y) vertices defining the boundary polygon
        pattern: Hatch pattern name (ANSI31 = diagonal lines, SOLID = solid fill)
        scale: Pattern scale factor
        layer: Target layer name
        color: ACI color index

    Returns:
        The created hatch entity

    M2 usage: hatching accepted/reviewed parcels in the output village map DXF
    to visually distinguish georeferencing quality.
    """
    try:
        import ezdxf
        hatch = msp.add_hatch(color=color)
        hatch.dxf.layer = layer
        if pattern != "SOLID":
            hatch.set_pattern_fill(pattern, scale=scale)
        else:
            hatch.set_solid_fill(color=color)

        # Build boundary path
        path = hatch.paths.add_polyline_path(
            [tuple(p) for p in boundary_points],
            is_closed=True
        )
        return hatch
    except ImportError:
        logger.warning("ezdxf not available for hatch creation")
        return None


# ============================================================
# 11. BLOCK INSERTION WITH ATTRIBUTES (from ezdxf_backend.py block_*)
# ============================================================
# Block INSERT entities with ATTRIB sub-entities are used in FMB DXF for
# repeated symbols (north arrows, survey stamps, title blocks).
# M2 usage: placing standardized symbols on the output village map.

@dataclass
class BlockInsertData:
    """Data for creating a block insertion with attributes.
    From autocad-mcp ezdxf_backend.py block_insert_with_attributes pattern."""
    block_name: str
    x: float
    y: float
    scale: float = 1.0
    rotation: float = 0.0  # degrees
    attributes: Dict[str, str] = field(default_factory=dict)


def insert_block_with_attributes(
    msp: Any,
    data: BlockInsertData,
    layer: str = "0",
) -> Any:
    """Insert a block reference with attribute values.

    From autocad-mcp ezdxf_backend.py block_insert_with_attributes.
    Creates an INSERT entity and attaches ATTRIB sub-entities for each tag-value pair.

    M2 usage: placing survey stamps, title blocks, or legend entries on
    the georeferenced village map DXF with plot-specific attributes.
    """
    try:
        import ezdxf
        # Check if block exists in the document
        doc = msp.doc
        if data.block_name not in doc.blocks:
            # Auto-define a simple point block if it doesn't exist
            block = doc.blocks.new(data.block_name)
            block.add_point((0, 0))

        # Create insertion
        insert = msp.add_blockref(
            data.block_name,
            (data.x, data.y),
            dxfattribs={
                "xscale": data.scale,
                "yscale": data.scale,
                "rotation": data.rotation,
                "layer": layer,
            }
        )

        # Add attributes
        for tag, value in data.attributes.items():
            insert.add_attrib(tag, value, (data.x, data.y))

        return insert
    except ImportError:
        logger.warning("ezdxf not available for block insertion")
        return None


# ============================================================
# 12. LAYER MANAGEMENT (from ezdxf_backend.py layer_* methods)
# ============================================================
# Layer management patterns for organizing DXF content.
# M2 usage: organizing output DXF into layers (boundaries, labels, dimensions,
# hatch_overlays, grid_lines, etc.)

@dataclass
class LayerProperties:
    """Layer properties for creation/modification.
    From autocad-mcp ezdxf_backend.py layer patterns."""
    name: str
    color: Optional[int] = None      # ACI color index
    linetype: Optional[str] = None   # "CONTINUOUS", "DASHED", etc.
    lineweight: Optional[float] = None  # in mm (e.g., 0.25, 0.50)


def setup_cadastral_layers(
    msp: Any,
) -> Dict[str, str]:
    """Create standard cadstral/FMB layer structure.

    Inspired by autocad-mcp pid_setup_layers pattern (which creates 7 P&ID layers).
    This creates layers appropriate for M2 georeferenced cadastral output:

    Returns:
        Dict mapping layer name → description

    M2 usage: organizing the output village map DXF into logical layers
    for GIS import and visual clarity.
    """
    layer_defs = {
        "PARCEL_BOUNDARY": {"color": 1, "lineweight": 0.50},      # Red, thick — parcel outlines
        "PARCEL_FILL_ACCEPT": {"color": 3, "lineweight": 0.25},    # Green hatch — accepted parcels
        "PARCEL_FILL_REVIEW": {"color": 2, "lineweight": 0.25},    # Yellow hatch — reviewed parcels
        "SURVEY_LABELS": {"color": 7, "lineweight": 0.18},         # White — survey numbers
        "DIMENSIONS": {"color": 4, "lineweight": 0.18},            # Cyan — measurement annotations
        "TILE_BOUNDARY": {"color": 8, "lineweight": 0.13},         # Grey — tile grid lines
        "ROADS": {"color": 6, "lineweight": 0.35},                 # Magenta — road boundaries
        "SUBDIVISION": {"color": 30, "lineweight": 0.13},          # Orange — sub-parcel lines
    }

    try:
        import ezdxf
        doc = msp.doc
        for layer_name, props in layer_defs.items():
            if layer_name not in doc.layers:
                doc.layers.new(
                    layer_name,
                    dxfattribs={
                        "color": props["color"],
                        "linetype": "CONTINUOUS",
                    }
                )
    except ImportError:
        pass

    return {name: f"color={d['color']}" for name, d in layer_defs.items()}


# ============================================================
# 13. ORTHOGONAL ROUTING (from ezdxf_backend.py pid_connect_equipment)
# ============================================================
# Manhattan-style path routing between two points.
# M2 usage: drawing right-angle connection lines in output DXF,
# or routing annotation leaders around obstacles.

def orthogonal_route(
    x1: float, y1: float, x2: float, y2: float,
    strategy: str = "midpoint"
) -> List[Tuple[float, float]]:
    """Generate an orthogonal (Manhattan-style) route between two points.

    From autocad-mcp ezdxf_backend.py pid_connect_equipment pattern.
    Creates an L-shaped or Z-shaped path using only horizontal and vertical segments.

    Args:
        x1, y1: Start point
        x2, y2: End point
        strategy: "midpoint" (default) = horizontal-then-vertical L-shape
                  "z_route" = Z-shape through midpoint

    Returns:
        List of (x, y) waypoints forming the orthogonal path

    M2 usage: drawing process/connection lines in output DXF, or routing
    dimension leaders around parcel boundaries.
    """
    if strategy == "midpoint":
        mid_x = (x1 + x2) / 2
        return [(x1, y1), (mid_x, y1), (mid_x, y2), (x2, y2)]
    elif strategy == "z_route":
        mid_y = (y1 + y2) / 2
        return [(x1, y1), (x1, mid_y), (x2, mid_y), (x2, y2)]
    elif strategy == "direct_l":
        return [(x1, y1), (x2, y1), (x2, y2)]
    else:
        return [(x1, y1), (x2, y2)]


# ============================================================
# 14. DXF → PNG RENDERING (from screenshot.py MatplotlibScreenshotProvider)
# ============================================================
# Headless DXF rendering to PNG using ezdxf + matplotlib.
# M2 usage: generating preview images of FMB DXF files and
# georeferenced output for validation/visualization.

def render_dxf_to_png(
    dxf_path: str,
    output_path: str,
    width: int = 1200,
    height: int = 900,
    dpi: int = 150,
    bg_color: str = "white",
    fg_color: str = "black",
) -> Optional[str]:
    """Render a DXF file to PNG using ezdxf + matplotlib (headless).

    From autocad-mcp screenshot.py MatplotlibScreenshotProvider pattern.
    Uses matplotlib Agg backend for headless rendering.

    Args:
        dxf_path: Path to input DXF file
        output_path: Path for output PNG file
        width: Image width in pixels
        height: Image height in pixels
        dpi: DPI for matplotlib figure
        bg_color: Background color name
        fg_color: Foreground (entity) color name

    Returns:
        Output PNG path on success, None on failure

    M2 usage: generating preview images of:
      1. Raw FMB DXF files (before georeferencing)
      2. Georeferenced output DXF (after fitting)
      3. Comparison views (overlaid on tile imagery)
    """
    try:
        import ezdxf
        from ezdxf.addons.drawing import RenderContext, Frontend
        from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()

        # Create matplotlib figure
        fig, ax = plt.subplots(
            figsize=(width / dpi, height / dpi),
            dpi=dpi,
            facecolor=bg_color
        )

        # Setup rendering context
        ctx = RenderContext(doc)
        ctx.set_current_layout(msp)
        out = MatplotlibBackend(ax)

        # Render
        frontend = Frontend(ctx, out)
        frontend.draw_layout(msp, finalize=True)

        # Save
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                    facecolor=bg_color, edgecolor="none")
        plt.close(fig)

        return output_path
    except ImportError as e:
        logger.warning(f"Cannot render DXF: missing dependency ({e})")
        return None
    except Exception as e:
        logger.error(f"DXF rendering failed: {e}")
        return None


def render_dxf_to_base64(
    dxf_path: str,
    width: int = 800,
    height: int = 600,
    dpi: int = 100,
) -> Optional[str]:
    """Render DXF to base64-encoded PNG (for MCP image content).
    From autocad-mcp screenshot.py pattern for returning ImageContent.
    M2 usage: embedding FMB previews in pipeline status reports."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    result = render_dxf_to_png(dxf_path, tmp_path, width, height, dpi)
    if result and os.path.exists(result):
        with open(result, "rb") as f:
            import base64
            return base64.b64encode(f.read()).decode("utf-8")
    return None


# ============================================================
# 15. IPC PROTOCOL PATTERN (from file_ipc.py)
# ============================================================
# File-based inter-process communication for AutoCAD integration.
# Pattern: Python writes JSON command file → signals AutoCAD via Win32 →
# AutoCAD LISP reads, executes, writes result JSON → Python polls and reads.
# M2 usage: IF the M2 pipeline needs to interact with a running AutoCAD
# instance (e.g., for DWG→DXF conversion, advanced entity editing),
# this pattern provides the foundation.

class IPCCommand(NamedTuple):
    """IPC command structure (from autocad-mcp file_ipc.py)."""
    command: str
    params: Dict[str, Any]
    request_id: str = ""


class IPCProtocol:
    """File-based IPC protocol for AutoCAD communication.
    From autocad-mcp file_ipc.py — simplified and adapted for M2.

    Protocol flow:
    1. write_command(cmd) → atomically writes JSON file
    2. signal_autocad() → sends keystroke to trigger LISP dispatch
    3. poll_result(request_id) → reads result JSON file

    M2 usage: Optional integration with AutoCAD for operations that
    ezdxf cannot handle (DWG reading, advanced plotting, etc.)
    """

    def __init__(self, temp_dir: str = "C:/temp", timeout: float = 10.0):
        self.temp_dir = Path(temp_dir)
        self.timeout = timeout
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _cmd_path(self, request_id: str) -> Path:
        return self.temp_dir / f"mcp_cmd_{request_id}.json"

    def _result_path(self, request_id: str) -> Path:
        return self.temp_dir / f"mcp_result_{request_id}.json"

    def write_command(self, command: str, params: Dict[str, Any]) -> str:
        """Write command JSON atomically (.tmp → rename).
        From autocad-mcp file_ipc.py _send_command pattern."""
        request_id = str(uuid.uuid4())[:8]
        cmd = {"command": command, "params": params, "id": request_id}
        path = self._cmd_path(request_id)

        # Atomic write via temp file + rename
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(cmd, f, ensure_ascii=False)
        tmp_path.rename(path)

        return request_id

    def read_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Read result JSON if available.
        From autocad-mcp file_ipc.py _poll_result pattern."""
        path = self._result_path(request_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fallback: try cp1252 encoding (AutoCAD default on Windows)
            try:
                with open(path, "r", encoding="cp1252") as f:
                    return json.load(f)
            except Exception:
                return None
        finally:
            # Cleanup
            try:
                path.unlink()
                self._cmd_path(request_id).unlink(missing_ok=True)
            except OSError:
                pass

    def poll_result(self, request_id: str, interval: float = 0.1) -> Optional[Dict[str, Any]]:
        """Poll for result with timeout.
        From autocad-mcp file_ipc.py _poll_for_result pattern."""
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            result = self.read_result(request_id)
            if result is not None:
                return result
            time.sleep(interval)
        return None  # Timeout


# ============================================================
# 16. COMMAND DISPATCH PATTERN (from server.py + mcp_dispatch.lsp)
# ============================================================
# Operation-based entity management architecture.
# Maps string operation names to handler functions via a whitelist dispatch table.
# M2 usage: structuring the FMB processing pipeline as a dispatch table
# for extensibility and testability.

class CommandDispatcher:
    """Operation dispatch table (from autocad-mcp server.py + mcp_dispatch.lsp).
    Maps string operation names to handler functions with type-safe dispatch.

    M2 usage: structuring FMB pipeline operations (parse, fit, validate, export)
    as a dispatch table for easy testing, logging, and extension.
    """

    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        self._whitelist: set = set()

    def register(self, operation: str, handler: Callable) -> None:
        """Register a handler for an operation name."""
        self._handlers[operation] = handler
        self._whitelist.add(operation)

    def dispatch(self, operation: str, **kwargs) -> CommandResult:
        """Execute an operation by name. Returns CommandResult envelope.
        Only whitelisted operations can be dispatched (security pattern
        from mcp_dispatch.lsp which uses a command whitelist map)."""
        if operation not in self._whitelist:
            return CommandResult.failure(
                f"Unknown operation: '{operation}'. "
                f"Available: {sorted(self._whitelist)}"
            )
        handler = self._handlers[operation]
        try:
            result = handler(**kwargs)
            if isinstance(result, CommandResult):
                return result
            return CommandResult.success(result)
        except Exception as e:
            return CommandResult.failure(f"{operation} failed: {e}")

    def list_operations(self) -> List[str]:
        """Return sorted list of registered operations."""
        return sorted(self._whitelist)


# ============================================================
# 17. DXF DRAWING INFO (from ezdxf_backend.py drawing_info)
# ============================================================
# Extract summary information from a DXF file.
# M2 usage: quick inspection of FMB DXF files before processing.

@dataclass
class DrawingInfo:
    """Summary information about a DXF drawing.
    From autocad-mcp ezdxf_backend.py drawing_info pattern.
    M2 usage: inspecting FMB DXF files to understand their structure."""
    entity_count: int = 0
    layers: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)
    dxf_version: str = ""
    entity_counts_by_type: Dict[str, int] = field(default_factory=dict)
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x_min, y_min, x_max, y_max)
    extents: Optional[Tuple[float, float]] = None  # (width, height)


def get_drawing_info(doc: Any) -> DrawingInfo:
    """Extract drawing summary from ezdxf document.
    From autocad-mcp ezdxf_backend.py drawing_info pattern.
    M2 usage: inspecting FMB DXF structure before pipeline processing."""
    info = DrawingInfo()
    msp = doc.modelspace()

    # Count entities by type
    type_counts: Dict[str, int] = {}
    for e in msp:
        etype = e.dxftype()
        type_counts[etype] = type_counts.get(etype, 0) + 1
    info.entity_counts_by_type = type_counts
    info.entity_count = sum(type_counts.values())

    # Layers
    info.layers = sorted([l.dxf.name for l in doc.layers])

    # Blocks
    info.blocks = sorted([b.dxf.name for b in doc.blocks if not b.is_anonymous])

    # DXF version
    info.dxf_version = doc.dxfversion

    # Bounding box
    try:
        bbox = msp.bbox()
        info.bbox = (
            float(bbox.extmin.x), float(bbox.extmin.y),
            float(bbox.extmax.x), float(bbox.extmax.y)
        )
        info.extents = (
            float(bbox.extmax.x - bbox.extmin.x),
            float(bbox.extmax.y - bbox.extmin.y)
        )
    except Exception:
        pass

    return info


# ============================================================
# 18. ENTITY TRANSFORM HELPERS (from ezdxf_backend.py entity_* patterns)
# ============================================================
# High-level entity transformation functions using the Mat4 class.
# These wrap the translate→transform→translate-back pattern used throughout
# autocad-mcp's ezdxf_backend.py.

def transform_points_around_center(
    points: np.ndarray,
    cx: float, cy: float,
    angle_deg: float = 0.0,
    scale_factor: float = 1.0,
) -> np.ndarray:
    """Apply rotation and/or scaling around a center point to an array of points.

    Combines the autocad-mcp patterns from entity_rotate and entity_scale:
    1. Translate points so center is at origin
    2. Apply rotation (if angle != 0)
    3. Apply scaling (if factor != 1)
    4. Translate back

    Args:
        points: (N, 2) array of points
        cx, cy: Center of rotation/scaling
        angle_deg: Rotation angle in degrees
        scale_factor: Uniform scale factor

    Returns:
        (N, 2) transformed points

    M2 usage: Applying the solved rigid transformation to all FMB boundary
    points in a single call.
    """
    pts = np.atleast_2d(points).astype(np.float64).copy()
    center = np.array([cx, cy])

    # Translate to origin
    pts -= center

    # Rotate
    if abs(angle_deg) > 1e-10:
        angle_rad = math.radians(angle_deg)
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        rot = np.array([[c, -s], [s, c]])
        pts = pts @ rot.T

    # Scale
    if abs(scale_factor - 1.0) > 1e-10:
        pts *= scale_factor

    # Translate back
    pts += center
    return pts


def compute_polygon_centroid(
    points: Union[np.ndarray, List[Tuple[float, float]]]
) -> Tuple[float, float]:
    """Compute centroid of a polygon (simple mean of vertices).
    For area-weighted centroid, use shapely or the shoelace formula in librecad_extracted.py.
    M2 usage: finding FMB polygon center for rotation/scaling transforms."""
    pts = np.atleast_2d(points)
    return (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))


def compute_bounding_box(
    points: Union[np.ndarray, List[Tuple[float, float]]]
) -> Tuple[float, float, float, float]:
    """Compute axis-aligned bounding box of points.
    Returns (x_min, y_min, x_max, y_max).
    M2 usage: bounding box comparison for shape quality gating."""
    pts = np.atleast_2d(points)
    return (
        float(np.min(pts[:, 0])),
        float(np.min(pts[:, 1])),
        float(np.max(pts[:, 0])),
        float(np.max(pts[:, 1]))
    )


def polygon_area_shoelace(
    points: Union[np.ndarray, List[Tuple[float, float]]]
) -> float:
    """Compute signed area using the shoelace formula.
    Positive = counter-clockwise, Negative = clockwise.
    M2 usage: comparing FMB polygon area to recovered tile polygon area
    (quality gate: area_ratio must be 0.65-1.55)."""
    pts = np.atleast_2d(points).astype(np.float64)
    n = len(pts)
    if n < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    # Shoelace: 0.5 * sum(x_i * y_{i+1} - x_{i+1} * y_i)
    return float(0.5 * abs(
        np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)
    ))


# ============================================================
# 19. DXF CREATION HELPERS (from ezdxf_backend.py create_* patterns)
# ============================================================
# Programmatic DXF entity creation for building output files.
# M2 usage: writing georeferenced village map DXF with all parcels.

def create_output_dxf(
    output_path: str,
    parcels: List[Dict[str, Any]],
    metadata: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Create a georeferenced village map DXF from M2 pipeline results.

    From autocad-mcp ezdxf_backend.py entity creation patterns, composed into
    a full DXF generation function for M2 output.

    Args:
        output_path: Path for output DXF file
        parcels: List of parcel dicts, each containing:
            - "survey_number": str
            - "status": str ("ACCEPT", "REVIEW", "NO_COVERAGE")
            - "boundary_points": List[(x, y)] — georeferenced UTM coordinates
            - "label_point": (x, y) — survey label placement
            - "fmb_source": str — source FMB filename
            - "residual_rms": float — fitting residual
        metadata: Optional dict with "village", "district", "date", etc.

    Returns:
        Output DXF path on success, None on failure
    """
    try:
        import ezdxf
        from ezdxf import units

        doc = ezdxf.new(dxfversion="AC1015")  # R2000 for max compatibility
        doc.units = units.M  # Meters (UTM)
        msp = doc.modelspace()

        # Setup layers
        setup_cadastral_layers(msp)

        # Set metadata in header variables
        if metadata:
            for key, value in metadata.items():
                try:
                    doc.header[f"$CUSTOM_{key.upper()}"] = value
                except Exception:
                    pass

        # Write parcels
        for parcel in parcels:
            status = parcel.get("status", "REVIEW")
            points = parcel.get("boundary_points", [])
            survey_num = str(parcel.get("survey_number", ""))

            if not points or len(points) < 3:
                continue

            # Determine layer by status
            if status == "ACCEPT":
                boundary_layer = "PARCEL_BOUNDARY"
                hatch_layer = "PARCEL_FILL_ACCEPT"
            elif status == "REVIEW":
                boundary_layer = "PARCEL_BOUNDARY"
                hatch_layer = "PARCEL_FILL_REVIEW"
            else:
                boundary_layer = "TILE_BOUNDARY"
                hatch_layer = None

            # Create boundary polyline (closed)
            msp.add_lwpolyline(
                points, close=True,
                dxfattribs={"layer": boundary_layer}
            )

            # Create hatch fill for accepted/reviewed
            if hatch_layer and len(points) >= 3:
                create_hatch_with_boundary(
                    msp, points,
                    pattern="SOLID" if status == "ACCEPT" else "ANSI31",
                    layer=hatch_layer,
                    color=3 if status == "ACCEPT" else 2,
                )

            # Place survey number label
            label_pt = parcel.get("label_point")
            if label_pt is None and points:
                label_pt = compute_polygon_centroid(points)

            if label_pt:
                msp.add_text(
                    survey_num,
                    dxfattribs={
                        "insert": label_pt,
                        "height": 2.0,  # 2m text height in UTM
                        "layer": "SURVEY_LABELS",
                    }
                )

        doc.saveas(output_path)
        return output_path

    except ImportError:
        logger.warning("ezdxf not available for DXF creation")
        return None
    except Exception as e:
        logger.error(f"DXF creation failed: {e}")
        return None


# ============================================================
# 20. LISP COMMAND WHITELIST (from mcp_dispatch.lsp)
# ============================================================
# The autocad-mcp AutoLISP dispatcher uses a whitelist map for security —
# only pre-registered commands can be dispatched. This prevents arbitrary
# code execution via the IPC channel.
# M2 usage: IF using AutoCAD IPC, maintain a similar whitelist for safety.

# Command categories from autocad-mcp mcp_dispatch.lsp (49 total commands):
LISP_COMMAND_CATEGORIES = {
    "drawing": [
        "ping", "drawing-info", "drawing-save", "drawing-save-as-dxf",
        "drawing-create", "drawing-purge", "drawing-open",
        "drawing-get-variables", "drawing-plot-pdf",
    ],
    "entity_create": [
        "create-line", "create-circle", "create-polyline",
        "create-rectangle", "create-text", "create-arc",
        "create-ellipse", "create-mtext", "create-hatch",
    ],
    "entity_modify": [
        "entity-erase", "entity-move", "entity-copy",
        "entity-rotate", "entity-scale", "entity-mirror",
        "entity-offset", "entity-array", "entity-fillet",
        "entity-chamfer",
    ],
    "entity_query": ["entity-list", "entity-count", "entity-get"],
    "layer": [
        "layer-list", "layer-create", "layer-set-current",
        "layer-set-properties", "layer-freeze", "layer-thaw",
        "layer-lock", "layer-unlock",
    ],
    "block": [
        "block-list", "block-insert", "block-insert-with-attributes",
        "block-get-attributes", "block-update-attribute", "block-define",
    ],
    "annotation": [
        "create-dimension-linear", "create-dimension-aligned",
        "create-dimension-angular", "create-dimension-radius",
        "create-leader",
    ],
    "view": ["zoom-extents", "zoom-window"],
    "system": ["execute-lisp", "undo", "redo"],
}

# Flatten to full whitelist
ALL_LISP_COMMANDS = sorted(
    cmd for cmds in LISP_COMMAND_CATEGORIES.values() for cmd in cmds
)


# ============================================================
# 21. FLOW ARROW / SYMBOL GEOMETRY (from ezdxf_backend.py pid_* patterns)
# ============================================================
# Geometric construction of simple CAD symbols using basic primitives.
# M2 usage: drawing north arrows, scale bars, and directional indicators
# on the output village map DXF.

def flow_arrow_points(
    x: float, y: float, rotation_deg: float, size: float = 2.0
) -> List[Tuple[float, float]]:
    """Compute triangle vertices for a directional flow arrow.
    From autocad-mcp ezdxf_backend.py pid_add_flow_arrow pattern.
    Triangle with ±2.4 radian spread for base vertices.

    M2 usage: drawing direction indicators (e.g., north arrow)
    on the output village map DXF."""
    rad = math.radians(rotation_deg)
    tip = (x + size * math.cos(rad), y + size * math.sin(rad))
    p2 = (x + size * 0.5 * math.cos(rad + 2.4),
          y + size * 0.5 * math.sin(rad + 2.4))
    p3 = (x + size * 0.5 * math.cos(rad - 2.4),
          y + size * 0.5 * math.sin(rad - 2.4))
    return [tip, p2, p3]


def pump_symbol_points(
    x: float, y: float, rotation_deg: float, radius: float = 6.0
) -> Dict[str, Any]:
    """Compute geometry for a pump symbol (circle + directional triangle).
    From autocad-mcp ezdxf_backend.py pid_insert_pump pattern.
    Circle + triangle with ±0.5 radian offset for directional indicator.

    M2 usage: example of how to construct compound CAD symbols
    from primitive shapes."""
    rad = math.radians(rotation_deg)
    # Circle center
    center = (x, y)
    # Triangle tip (outside circle)
    tip_x = x + (radius + 2) * math.cos(rad)
    tip_y = y + (radius + 2) * math.sin(rad)
    # Triangle base points (on circle, ±0.5 rad offset)
    p2 = (x + radius * math.cos(rad + 0.5),
          y + radius * math.sin(rad + 0.5))
    p3 = (x + radius * math.cos(rad - 0.5),
          y + radius * math.sin(rad - 0.5))
    return {
        "circle_center": center,
        "circle_radius": radius,
        "triangle": [(tip_x, tip_y), p2, p3],
    }


def north_arrow_points(
    x: float, y: float, size: float = 10.0
) -> Dict[str, List[Tuple[float, float]]]:
    """Compute geometry for a north arrow symbol.
    M2 usage: placing north arrow on the georeferenced village map DXF."""
    # Arrow pointing up (north = +Y in UTM)
    # Filled triangle (north half)
    north_triangle = [
        (x, y + size),           # Tip (north)
        (x - size * 0.3, y),     # Base left
        (x + size * 0.3, y),     # Base right
    ]
    # Open triangle (south half)
    south_triangle = [
        (x, y - size * 0.6),     # Bottom tip
        (x - size * 0.3, y),     # Base left
        (x + size * 0.3, y),     # Base right
    ]
    return {
        "north_fill": north_triangle,
        "south_open": south_triangle,
    }


# ============================================================
# 22. RECTANGULAR ARRAY PATTERN (from ezdxf_backend.py entity_array)
# ============================================================
# Generate a grid of transformed copies of a point set.
# M2 usage: not directly used in FMB fitting, but useful for
# generating test data or tile grid overlays.

def rectangular_array(
    points: np.ndarray,
    rows: int, cols: int,
    row_spacing: float, col_spacing: float,
) -> np.ndarray:
    """Generate a rectangular array of point sets.

    From autocad-mcp ezdxf_backend.py entity_array pattern.
    Creates (rows × cols) copies, skipping the original at (0,0).

    Args:
        points: (N, 2) template points
        rows: Number of rows
        cols: Number of columns
        row_spacing: Vertical spacing between rows
        col_spacing: Horizontal spacing between columns

    Returns:
        (rows × cols × N, 2) array of all arrayed points

    M2 usage: generating tile grid overlays for the village map,
    or creating test parcel layouts for pipeline validation.
    """
    all_points = []
    pts = np.atleast_2d(points)
    for r in range(rows):
        for c in range(cols):
            if r == 0 and c == 0:
                all_points.append(pts.copy())
            else:
                offset = np.array([c * col_spacing, r * row_spacing])
                all_points.append(pts + offset)
    return np.vstack(all_points)


# ============================================================
# 23. DXF FILE I/O HELPERS (from ezdxf_backend.py file operations)
# ============================================================
# Safe DXF file reading/writing with error handling.
# M2 usage: reading FMB DXF input files and writing georeferenced output.

def safe_read_dxf(dxf_path: str) -> Optional[Any]:
    """Safely read a DXF file with comprehensive error handling.
    From autocad-mcp file_ipc.py and ezdxf_backend.py patterns.
    Handles encoding issues, corrupted files, and missing dependencies.

    M2 usage: reading FMB DXF files which may have encoding issues
    (Tamil text in survey labels, non-standard encodings)."""
    try:
        import ezdxf
        # Try UTF-8 first, then fallback encodings
        for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
            try:
                doc = ezdxf.readfile(dxf_path, encoding=encoding)
                logger.info("DXF read success", path=dxf_path, encoding=encoding)
                return doc
            except UnicodeDecodeError:
                continue
            except ezdxf.DXFError:
                raise
        logger.error("DXF encoding failed for all encodings", path=dxf_path)
        return None
    except ImportError:
        logger.error("ezdxf not installed — cannot read DXF")
        return None
    except ezdxf.DXFStructureError as e:
        logger.error("DXF structure error", path=dxf_path, error=str(e))
        return None
    except Exception as e:
        logger.error("DXF read failed", path=dxf_path, error=str(e))
        return None


def safe_save_dxf(doc: Any, output_path: str, version: str = "AC1015") -> bool:
    """Safely save a DXF file. Returns True on success.
    From autocad-mcp ezdxf_backend.py drawing_save patterns.
    M2 usage: saving georeferenced output DXF files."""
    try:
        doc.saveas(output_path)
        logger.info("DXF saved", path=output_path)
        return True
    except Exception as e:
        logger.error("DXF save failed", path=output_path, error=str(e))
        return False


# ============================================================
# 24. ANGLE/DISTANCE COMPUTATION HELPERS (scattered across ezdxf_backend.py)
# ============================================================
# Basic geometric computations used throughout autocad-mcp.
# M2 usage: computing bearing angles, distances, and directions
# for FMB boundary validation and neighbor analysis.

def angle_between_points(
    x1: float, y1: float, x2: float, y2: float
) -> float:
    """Compute angle from point 1 to point 2 in radians.
    From autocad-mcp's math.atan2 usage pattern throughout ezdxf_backend.py.
    M2 usage: computing bearing from centroid to survey label position
    (for 'seat gating' in the M2 quality gate)."""
    return math.atan2(y2 - y1, x2 - x1)


def distance_between_points(
    x1: float, y1: float, x2: float, y2: float
) -> float:
    """Euclidean distance between two points.
    M2 usage: computing centroid-to-label distance for seat gating
    (M2 quality gate: distance ≤ max(60, 1.6 * sqrt(area/pi)) meters)."""
    return math.hypot(x2 - x1, y2 - y1)


def bearing_degrees(
    x1: float, y1: float, x2: float, y2: float
) -> float:
    """Compute bearing from point 1 to point 2 in degrees (0=N, 90=E).
    Converts standard math angle (0=E, CCW positive) to compass bearing (0=N, CW positive).
    M2 usage: computing bearing from parcel centroid to survey label
    for neighbor azimuth analysis in cadastral_seat_v2.py."""
    angle_rad = math.atan2(y2 - y1, x2 - x1)
    angle_deg = math.degrees(angle_rad)
    # Convert from math convention (0=E, CCW+) to bearing (0=N, CW+)
    bearing = (90.0 - angle_deg) % 360.0
    return bearing


def point_on_circle(
    cx: float, cy: float, radius: float, angle_deg: float
) -> Tuple[float, float]:
    """Compute point on circle at given angle.
    From autocad-mcp ezdxf_backend.py radius dimension pattern.
    M2 usage: computing dimension leader endpoints on circular parcels."""
    rad = math.radians(angle_deg)
    return (cx + radius * math.cos(rad), cy + radius * math.sin(rad))


def midpoint(
    x1: float, y1: float, x2: float, y2: float
) -> Tuple[float, float]:
    """Midpoint of two points."""
    return ((x1 + x2) / 2, (y1 + y2) / 2)


# ============================================================
# 25. INTEGRATION: M2 QUALITY GATE HELPERS
# ============================================================
# Functions that directly support the M2 pipeline quality gating.
# These combine patterns from autocad-mcp with M2-specific thresholds.

@dataclass
class QualityGateResult:
    """Result of M2 quality gate check.
    M2 usage: each fitted parcel must pass ALL gates to get ACCEPT status."""
    shape_gate: bool = False      # Area ratio 0.65-1.55
    scale_gate: bool = False      # Scale factor 0.80-1.25
    corner_gate: bool = False     # Corner residual ≤ 12m
    seat_gate: bool = False       # Centroid-to-label ≤ max(60, 1.6*sqrt(area/pi))
    overlap_gate: bool = True     # No overlap with accepted parcels (default pass)
    status: str = "NO_COVERAGE"   # ACCEPT, REVIEW, NO_COVERAGE

    @property
    def is_accept(self) -> bool:
        return all([
            self.shape_gate, self.scale_gate,
            self.corner_gate, self.seat_gate, self.overlap_gate
        ])


def check_shape_gate(
    fmb_area: float, tile_area: float,
    ratio_min: float = 0.65, ratio_max: float = 1.55
) -> bool:
    """M2 Shape Gate: area ratio must be within [ratio_min, ratio_max].
    Uses polygon_area_shoelace for area computation.
    Rejects trivially small or merged-parcel polygons."""
    if tile_area < 1e-6:
        return False
    ratio = fmb_area / tile_area
    return ratio_min <= ratio <= ratio_max


def check_scale_gate(
    scale_factor: float,
    scale_min: float = 0.80, scale_max: float = 1.25
) -> bool:
    """M2 Scale Gate: similarity transform scale must be near 1.0.
    Both FMB and UTM are in meters, so ideal scale = 1.0.
    Large deviations indicate wrong correspondence."""
    return scale_min <= scale_factor <= scale_max


def check_corner_gate(
    residuals: np.ndarray, max_residual: float = 12.0
) -> bool:
    """M2 Corner Gate: all corner residuals must be ≤ max_residual meters.
    High residuals indicate poor point correspondence."""
    return bool(np.all(residuals <= max_residual))


def check_seat_gate(
    centroid: Tuple[float, float],
    label_point: Tuple[float, float],
    parcel_area: float,
    base_tolerance: float = 60.0,
    area_factor: float = 1.6
) -> bool:
    """M2 Seat Gate: centroid-to-label distance must be reasonable.

    Tolerance = max(base_tolerance, area_factor * sqrt(area / pi))
    This scales the allowed distance with parcel size — larger parcels
    can have more distant labels.

    M2 usage: the survey label should be near the parcel centroid.
    Large offsets indicate OCR error or wrong label association.
    """
    dist = distance_between_points(
        centroid[0], centroid[1],
        label_point[0], label_point[1]
    )
    # Scale tolerance with parcel area (radius of equivalent circle × factor)
    if parcel_area > 0:
        area_tolerance = area_factor * math.sqrt(parcel_area / math.pi)
    else:
        area_tolerance = base_tolerance
    tolerance = max(base_tolerance, area_tolerance)
    return dist <= tolerance


def run_all_quality_gates(
    fmb_area: float,
    tile_area: float,
    scale_factor: float,
    corner_residuals: np.ndarray,
    centroid: Tuple[float, float],
    label_point: Tuple[float, float],
    has_overlap: bool = False,
) -> QualityGateResult:
    """Run all M2 quality gates and determine parcel status.

    M2 acceptance rule: ALL gates must pass → ACCEPT
    Any gate failure → REVIEW or NO_COVERAGE depending on severity.

    0-FP PRINCIPLE: The system can only place parcels with independent evidence.
    These mathematical gates enforce that principle — a parcel must have
    correct shape, scale, corner fit, and label placement to be accepted.
    """
    result = QualityGateResult()

    result.shape_gate = check_shape_gate(fmb_area, tile_area)
    result.scale_gate = check_scale_gate(scale_factor)
    result.corner_gate = check_corner_gate(corner_residuals)
    result.seat_gate = check_seat_gate(centroid, label_point, fmb_area)
    result.overlap_gate = not has_overlap

    # Determine final status
    if result.is_accept:
        result.status = "ACCEPT"
    elif result.shape_gate and result.scale_gate:
        # Shape and scale OK but corner/seat/overlap failed
        result.status = "REVIEW"
    else:
        result.status = "NO_COVERAGE"

    return result


# ============================================================
# 26. MODULE SELF-TEST
# ============================================================

def _self_test() -> None:
    """Run basic self-tests for all extracted functions."""
    print("=" * 60)
    print("autocad_mcp_extracted.py — Self-Test")
    print("=" * 60)

    # Test Mat4
    m = Mat4.rotate_around_point(5, 5, math.radians(90))
    pts = np.array([[5, 6], [5, 4]])
    transformed = m.transform_points(pts)
    print(f"Mat4 rotate_around_point: {pts.tolist()} → {transformed.tolist()}")

    # Test mirror
    mm = Mat4.mirror_across_line(0, 0, 1, 0)  # Mirror across x-axis
    mirrored = mm.transform_points(np.array([[1, 2], [3, 4]]))
    print(f"Mat4 mirror_across_line (x-axis): {(1,2),(3,4)} → {mirrored.tolist()}")

    # Test rigid transform
    src = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    tgt = np.array([[10, 10], [10, 11], [9, 10]], dtype=float)
    M, rms = solve_rigid_transform(src, tgt)
    fitted = M.transform_points(src)
    print(f"Rigid transform RMS: {rms:.6f}")
    print(f"  Source: {src.tolist()}")
    print(f"  Target: {tgt.tolist()}")
    print(f"  Fitted: {fitted.tolist()}")

    # Test similarity transform
    src2 = np.array([[0, 0], [10, 0], [0, 5]], dtype=float)
    tgt2 = np.array([[100, 200], [110, 200], [100, 205]], dtype=float)
    M2_mat, scale, rms2 = solve_similarity_transform(src2, tgt2)
    print(f"Similarity transform: scale={scale:.4f}, rms={rms2:.6f}")

    # Test quality gates
    qg = run_all_quality_gates(
        fmb_area=1000, tile_area=950,
        scale_factor=1.02,
        corner_residuals=np.array([2.1, 3.5, 1.8, 4.2]),
        centroid=(50, 50), label_point=(52, 51),
        has_overlap=False
    )
    print(f"Quality gates: shape={qg.shape_gate}, scale={qg.scale_gate}, "
          f"corner={qg.corner_gate}, seat={qg.seat_gate} → {qg.status}")

    # Test polygon area
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    area = polygon_area_shoelace(square)
    print(f"Polygon area (10x10 square): {area:.1f} (expected 100.0)")

    # Test orthogonal routing
    route = orthogonal_route(0, 0, 10, 10, "midpoint")
    print(f"Orthogonal route (0,0)→(10,10): {route}")

    # Test ACI colors
    print(f"ACI 'red' → {aci_to_rgb(color_name_to_aci('red'))}")
    print(f"ACI 5 (blue) → {aci_to_rgb(5)}")

    # Test CommandResult
    cr = CommandResult.success({"key": "value"})
    print(f"CommandResult: ok={cr.ok}, unwrap={cr.unwrap()}")

    # Test CommandDispatcher
    disp = CommandDispatcher()
    disp.register("test_add", lambda a, b: a + b)
    result = disp.dispatch("test_add", a=3, b=4)
    print(f"Dispatcher 'test_add(3,4)': {result}")

    # Test bearing
    brg = bearing_degrees(0, 0, 1, 0)
    print(f"Bearing (0,0)→(1,0): {brg:.1f}° (expected 90° East)")

    # Test flow arrow
    arrow = flow_arrow_points(0, 0, 90, 5.0)
    print(f"Flow arrow (north): {arrow}")

    # Test IPC protocol (no AutoCAD, just test file I/O)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        ipc = IPCProtocol(temp_dir=tmpdir, timeout=1.0)
        rid = ipc.write_command("test", {"key": "value"})
        print(f"IPC command written: request_id={rid}")
        # Simulate result file
        result_path = ipc._result_path(rid)
        with open(result_path, "w") as f:
            json.dump({"ok": True, "data": "test"}, f)
        result = ipc.poll_result(rid)
        print(f"IPC result: {result}")

    print("=" * 60)
    print("All self-tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()