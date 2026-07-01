"""Write georeferenced DXF output.

Reads the original M1 DXF, transforms ALL geometry (stones, boundaries,
chain lines, subdivisions, dimensions, labels) from relative coordinates
to UTM using the adjusted stone positions, and writes a new DXF.

Transformation method:
  - Stone positions: use the cadastral-adjusted UTM coordinates directly.
  - Boundary polylines: warp vertices proportionally between adjusted stones.
  - Other geometry (subdivision, chain, separation): warp with nearest-stone
    offset field for cadastral consistency.
  - TEXT positions: transformed with the similarity transform.
  - Measurements: kept as-is (they are labels, not coordinates).

CRS metadata: EPSG:32644 (UTM Zone 44N) is embedded via $GDALWKT in the
DXF header and also as a TEXT entity in modelspace for GDAL/OGR readability.

Layer names are sourced from ``landintel.core.enums.LayerType`` (the same
single source of truth ``to_dxf.py`` writes from) so transforms cannot miss a
layer due to a name typo -- e.g. the neighbor-label layer is literally
``"neighbor label"`` (lowercase, with a space).
"""

from __future__ import annotations

import logging
from pathlib import Path

import ezdxf
import numpy as np
from ezdxf.addons import Importer
from scipy.spatial import cKDTree

from ...core.enums import LayerType
from .warp import warp_boundary_vertices, warp_generic_vertices

_log = logging.getLogger(__name__)

DEFAULT_CRS = "EPSG:32643"  # UTM Zone 43N (INGUR / Erode ~77.6E is west of 78E)

# Canonical layer names (single source of truth = LayerType).
_L_STONES = LayerType.STONES.value
_L_SURVEY = LayerType.SURVEY_NUMBER.value
_L_BOUNDARY = LayerType.BOUNDARY.value
_L_SUBDIV = LayerType.SUBDIVISION.value
_L_NEIGHBOR = LayerType.NEIGHBOR_LABEL.value  # "neighbor label"
_DIM_LAYERS = (
    LayerType.BOUNDARY_DIMENSIONS.value,
    LayerType.CHAINLINE_DIMENSIONS.value,
    LayerType.DIMENSIONS.value,
)
_GENERIC_POLY_LAYERS = (
    LayerType.CHAIN_LINES.value,
    LayerType.SUBDIVISION_LINES.value,
    LayerType.SEPARATION_LINE.value,
    LayerType.DASHED_REF.value,
)


def _entities_on(msp, dxftype: str, layer: str) -> list:
    """Return entities of ``dxftype`` on ``layer`` (Python-side filter).

    Avoids ezdxf query-DSL quoting ambiguity for layer names with spaces.
    """
    return [e for e in msp.query(dxftype) if e.dxf.layer == layer]


def _apply_transform(point: tuple[float, float],
                     R: np.ndarray, s: float, t: np.ndarray) -> tuple[float, float]:
    """Apply similarity transform: p' = s * R @ p + t"""
    x, y = point
    rx = R[0, 0] * x + R[0, 1] * y
    ry = R[1, 0] * x + R[1, 1] * y
    return (s * rx + t[0], s * ry + t[1])


def _set_text_pos(e, x: float, y: float) -> None:
    """Move a TEXT to (x, y), keeping insert AND align_point in sync.

    Aligned TEXT (non-default h/valign) renders at `align_point`; leaving it stale
    strands the label at its original coordinates. We set both to the same point
    (correct for a point-anchored label like a stone/sub-plot tag).
    """
    z = e.dxf.insert.z if e.dxf.hasattr("insert") else 0.0
    e.dxf.insert = (x, y, z)
    if e.dxf.hasattr("align_point"):
        e.dxf.align_point = (x, y, e.dxf.align_point.z)


def _rotation_deg(R: np.ndarray) -> float:
    """Plot rotation in degrees, or 0 for a reflected fit (a mirror is not a text angle)."""
    if float(np.linalg.det(R)) <= 0:
        return 0.0
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def _xform_text(e, R: np.ndarray, s: float, t: np.ndarray) -> None:
    """Similarity-transform a TEXT entity's insert, align_point AND glyph rotation.

    The plot is placed as a RIGID body: geometry and text turn by the same angle, so a
    dimension label that ran parallel to its edge in M1 stays parallel after georef
    (the M1 text-to-geometry relationship is preserved exactly). Height is left as M1
    set it -- the rigid scale is ~1, so it already matches the plot size.
    """
    ip = _apply_transform((e.dxf.insert.x, e.dxf.insert.y), R, s, t)
    e.dxf.insert = (ip[0], ip[1], e.dxf.insert.z)
    if e.dxf.hasattr("align_point"):
        ap = _apply_transform((e.dxf.align_point.x, e.dxf.align_point.y), R, s, t)
        e.dxf.align_point = (ap[0], ap[1], e.dxf.align_point.z)
    theta = _rotation_deg(R)
    if theta:
        cur = float(e.dxf.rotation) if e.dxf.hasattr("rotation") else 0.0
        e.dxf.rotation = (cur + theta) % 360.0


def _set_crs_in_header(doc: ezdxf.document.Drawing, crs: str) -> None:
    """Write CRS as extended data in the DXF header for GDAL/OGR to read.

    The WKT is derived from ``crs`` via pyproj so the zone is always correct
    (INGUR is UTM 43N / central meridian 75, NOT the old hardcoded 44N / 81).
    """
    wkt = None
    try:
        from pyproj import CRS
        wkt = CRS.from_user_input(crs).to_wkt()
    except Exception:  # noqa: BLE001 - pyproj missing / bad code -> generic fallback
        wkt = (
            'PROJCS["WGS 84 / UTM zone 43N",GEOGCS["WGS 84",DATUM["WGS_1984",'
            'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
            'UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
            'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",75],'
            'PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],'
            'PARAMETER["false_northing",0],UNIT["metre",1]]'
        )
    try:
        doc.header['$GDALWKT'] = wkt
    except Exception:
        _log.debug("Could not set $GDALWKT in header (version may not support it)")


def write_georef_dxf(
    m1_dxf_path: str | Path,
    output_path: str | Path,
    adjusted_stone_positions: np.ndarray,
    original_stone_positions: np.ndarray,
    stone_label_to_index: dict[str, int],
    R: np.ndarray,
    s: float,
    t: np.ndarray,
    crs: str = DEFAULT_CRS,
    corner_ring: list[int] | None = None,
    drop_synthetic_stone_labels: bool = True,
    drop_neighbor_labels: bool = True,
    dim_text_scale: float = 0.55,
) -> Path:
    """Transform an M1 DXF to georeferenced UTM coordinates and write output.

    Parameters
    ----------
    m1_dxf_path : path to the original M1 DXF
    output_path : path for the georeferenced output DXF
    adjusted_stone_positions : (N, 2) adjusted UTM positions for N stones
    original_stone_positions : (N, 2) original M1 positions for warping
    stone_label_to_index : mapping from stone label string -> stone index
    R, s, t : Umeyama similarity transform parameters
    crs : coordinate reference system string
    corner_ring : ordered corner-stone indices (the outer boundary ring). When
        given, the BOUNDARY layer is REBUILT as one closed polyline through the
        adjusted corner positions -- the surveyor's own straight stone-to-stone
        representation. This guarantees a closed, correctly-areaed boundary and
        avoids warp artifacts from the FMB sketch's intermediate vertices.

    Returns
    -------
    Path to the written DXF file.
    """
    m1_dxf_path = Path(m1_dxf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _log.info("Writing georeferenced DXF: %s", output_path)

    doc = ezdxf.readfile(str(m1_dxf_path))
    msp = doc.modelspace()

    # Set CRS metadata
    _set_crs_in_header(doc, crs)

    # --- Transform STONES layer TEXT (use adjusted positions directly) ---
    # NOTE: TEXT with a non-default alignment (h/valign) stores its REAL position
    # in `align_point`, not `insert` -- AutoCAD renders aligned text at the
    # align_point. So `insert` alone is not enough: a stale align_point leaves the
    # label at its original relative-metre coords (near origin). _set_text_pos and
    # _xform_text below keep BOTH in sync.
    for e in _entities_on(msp, "TEXT", _L_STONES):
        label = str(e.dxf.text).strip()
        # Synthetic '?N' tags mark stones OCR could not read. In a dense plot they
        # are hundreds of overlapping labels that BURY the real numbers, and the
        # manual clubbed sheets carry none of them -- stone POSITIONS are the
        # load-bearing datum, not these labels. Drop the TEXT only; the stone
        # marker geometry on this layer is untouched, so georef anchors stay intact.
        if drop_synthetic_stone_labels and label.startswith("?"):
            msp.delete_entity(e)
            continue
        if label in stone_label_to_index:
            idx = stone_label_to_index[label]
            _set_text_pos(e, float(adjusted_stone_positions[idx, 0]),
                          float(adjusted_stone_positions[idx, 1]))
        else:
            _xform_text(e, R, s, t)

    # --- Neighbour labels: the survey numbers of ADJACENT plots, borrowed by best-effort
    # OCR into THIS plot. They are not this parcel's data, frequently wrong/mis-anchored,
    # and the manual clubbed sheet does not carry them -- drop them (client request). ---
    if drop_neighbor_labels:
        for e in _entities_on(msp, "TEXT", _L_NEIGHBOR):
            msp.delete_entity(e)

    # --- Transform single-position TEXT layers via similarity transform ---
    text_layers = (_L_SURVEY, _L_SUBDIV, *_DIM_LAYERS) if drop_neighbor_labels \
        else (_L_SURVEY, _L_SUBDIV, _L_NEIGHBOR, *_DIM_LAYERS)
    for layer_name in text_layers:
        is_dim = layer_name in _DIM_LAYERS
        for e in _entities_on(msp, "TEXT", layer_name):
            _xform_text(e, R, s, t)
            # Dimension numbers are packed many-per-cell in subdivided plots, so M1's
            # readable-at-print height overlaps badly on-screen. Shrink ONLY the dimension
            # labels (survey number + subdivision tags stay full size = still legible).
            if is_dim and dim_text_scale != 1.0 and e.dxf.hasattr("height"):
                e.dxf.height = float(e.dxf.height) * dim_text_scale

    # --- Transform BOUNDARY LWPOLYLINE (warp vertices proportionally) ---
    if original_stone_positions.shape[0] > 0:
        tree = cKDTree(original_stone_positions)
    else:
        tree = None
    valid_ring = (
        corner_ring is not None
        and len(corner_ring) >= 3
        and all(0 <= i < len(adjusted_stone_positions) for i in corner_ring)
    )
    if valid_ring:
        # Rebuild the boundary as ONE closed polyline through adjusted corners:
        # the surveyor's straight stone-to-stone representation. Guarantees
        # closure + correct area, free of FMB intermediate-vertex warp artifacts.
        ring_pts = [(float(adjusted_stone_positions[i][0]),
                     float(adjusted_stone_positions[i][1])) for i in corner_ring]
        ring_pts.append(ring_pts[0])
        existing = _entities_on(msp, "LWPOLYLINE", _L_BOUNDARY)
        for e in existing[1:]:
            msp.delete_entity(e)
        if existing:
            existing[0].set_points(ring_pts)
        else:
            msp.add_lwpolyline(ring_pts, dxfattribs={"layer": _L_BOUNDARY})
    else:
        for e in _entities_on(msp, "LWPOLYLINE", _L_BOUNDARY):
            pts = list(e.get_points())
            if tree is not None:
                pts_arr = np.array([(p[0], p[1]) for p in pts])
                dists, idxs = tree.query(pts_arr)
                stone_idx_per_vert = [int(i) if d < 2.0 else -1
                                      for i, d in zip(idxs, dists)]
                warped = warp_boundary_vertices(
                    [(p[0], p[1]) for p in pts],
                    stone_idx_per_vert,
                    original_stone_positions,
                    adjusted_stone_positions,
                )
                new_pts = [(float(w[0]), float(w[1])) for w in warped]
            else:
                new_pts = [_apply_transform((p[0], p[1]), R, s, t) for p in pts]
            e.set_points(new_pts)

    # --- Transform other LWPOLYLINE layers (nearest-stone offset warp) ---
    for layer_name in _GENERIC_POLY_LAYERS:
        for e in _entities_on(msp, "LWPOLYLINE", layer_name):
            pts = list(e.get_points())
            if original_stone_positions.shape[0] > 0:
                verts = [(p[0], p[1]) for p in pts]
                warped = warp_generic_vertices(
                    verts, original_stone_positions, adjusted_stone_positions
                )
                new_pts = [(float(w[0]), float(w[1])) for w in warped]
            else:
                new_pts = [_apply_transform((p[0], p[1]), R, s, t) for p in pts]
            e.set_points(new_pts)

    # --- Add CRS marker text ---
    # Place the marker at the geometry centroid (in UTM range), NOT at the
    # origin (0,0): the verification suite scans every TEXT insert for the
    # coordinate-range check, and a marker at the origin would drag min(x)/min(y)
    # to 0 and fail an otherwise-valid georeferenced file. GDAL reads the marker
    # regardless of where it sits.
    if adjusted_stone_positions is not None and len(adjusted_stone_positions) > 0:
        cx = float(np.mean(adjusted_stone_positions[:, 0]))
        cy = float(np.mean(adjusted_stone_positions[:, 1]))
    else:
        cx, cy = 0.0, 0.0
    msp.add_text(
        f"CRS:{crs}",
        dxfattribs={"layer": "0", "insert": (cx, cy), "height": 1.0}
    )

    doc.saveas(str(output_path))
    _log.info("Georeferenced DXF written: %s", output_path)
    return output_path


def build_combined_dxf(
    surveyor_dxf_path: str | Path,
    plot_dxf_paths: list[str | Path],
    output_path: str | Path,
    crs: str = DEFAULT_CRS,
) -> Path:
    """Club every georeferenced FMB into ONE file, seated in the raw data file.

    This is the real M2 deliverable: not N separate per-plot DXFs, but a SINGLE
    DXF where the surveyor's raw data file is the base canvas (its stones, traced
    SITE DATA LINEs, towers) and every georeferenced FMB plot is merged in at its
    UTM position -- so all the FMBs "sit" inside the raw survey, clubbed together
    into one village drawing.

    The base is the surveyor (raw data) DXF. Each plot DXF in ``plot_dxf_paths``
    is already in UTM (written by ``write_georef_dxf``); its modelspace entities
    are imported onto their own semantic layers (BOUNDARY, STONES, SURVEY_NUMBER,
    ...), which do NOT collide with the surveyor's layers (0, Point_Code, SITE
    DATA LINE, TOWER), so the result overlays the FMB parcels on the field survey.

    Parameters
    ----------
    surveyor_dxf_path : the raw data file (becomes the base canvas)
    plot_dxf_paths : georeferenced per-plot DXFs to club in (UTM already)
    output_path : path for the single combined DXF
    crs : coordinate reference system embedded in the header

    Returns
    -------
    Path to the written combined DXF.
    """
    surveyor_dxf_path = Path(surveyor_dxf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = ezdxf.readfile(str(surveyor_dxf_path))
    _set_crs_in_header(base, crs)

    n_ok = 0
    for gp in plot_dxf_paths:
        gp = Path(gp)
        try:
            src = ezdxf.readfile(str(gp))
        except Exception as exc:  # noqa: BLE001
            _log.warning("Skipping unreadable plot DXF %s: %s", gp, exc)
            continue
        importer = Importer(src, base)
        importer.import_modelspace()      # copies entities + required layers
        importer.finalize()
        n_ok += 1

    base.saveas(str(output_path))
    _log.info("Combined village DXF written: %s (%d plots clubbed into raw data)",
              output_path, n_ok)
    return output_path


def _msp_extent(doc):
    """(min_x, min_y, max_x, max_y) of a modelspace, or None if empty."""
    import ezdxf.bbox
    try:
        bb = ezdxf.bbox.extents(doc.modelspace(), fast=True)
    except Exception:  # noqa: BLE001
        return None
    if not bb.has_data:
        return None
    return (float(bb.extmin.x), float(bb.extmin.y),
            float(bb.extmax.x), float(bb.extmax.y))


def build_full_combined_dxf(
    surveyor_dxf_path: str | Path,
    placed_plot_dxf_paths: list[str | Path],
    staged_specs: list[tuple[str | Path, str]],
    output_path: str | Path,
    crs: str = DEFAULT_CRS,
    base_layers_hide: list[str] | None = None,
    base_layers_keep: list[str] | None = None,
    review_specs: list[tuple[str | Path, str]] | None = None,
) -> Path:
    """Club EVERY FMB into one file: placed ones at their true UTM position, the
    rest STAGED in a labelled band beside the survey so all are present.

    This is the "all FMBs in the raw data file" deliverable. Plots with field
    control (``placed_plot_dxf_paths``, written by ``write_georef_dxf``) are
    merged at their georeferenced UTM position exactly like ``build_combined_dxf``.
    Plots the surveyor never traced (``staged_specs`` = ``(m1_dxf_path, survey_no)``)
    have NO UTM control, so placing them at a guessed position would be a false
    positive. Instead each is copied -- to scale, in real metres -- into a tidy
    grid in an empty band to the EAST of the survey extent, every plot on its own
    ``STAGED_FMB_<survey>`` layer with a bold survey-number label. An operator (or
    ``seed_place``) then drags/seeds each staged plot onto its two known corners.

    The result: a single drawing containing all FMBs -- the controlled ones seated
    on the field survey, the uncontrolled ones staged and individually selectable.
    """
    from ezdxf.math import Matrix44

    surveyor_dxf_path = Path(surveyor_dxf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = ezdxf.readfile(str(surveyor_dxf_path))
    _set_crs_in_header(base, crs)
    base_msp = base.modelspace()

    # BUG #7: curate the base canvas -- drop noisy layers (e.g. the 624 stone labels)
    # so the clubbed FMB tiling reads cleanly, like the manual working file.
    if base_layers_hide or base_layers_keep:
        hide = set(base_layers_hide or [])
        keep = set(base_layers_keep) if base_layers_keep is not None else None
        for e in list(base_msp):
            lyr = getattr(e.dxf, "layer", None)
            if lyr is None:
                continue
            if lyr in hide or (keep is not None and lyr not in keep):
                base_msp.delete_entity(e)

    # 1) Merge the georeferenced (controlled) plots at their UTM positions.
    n_placed = 0
    for gp in placed_plot_dxf_paths:
        gp = Path(gp)
        try:
            src = ezdxf.readfile(str(gp))
        except Exception as exc:  # noqa: BLE001
            _log.warning("Skipping unreadable placed DXF %s: %s", gp, exc)
            continue
        importer = Importer(src, base)
        importer.import_modelspace()
        importer.finalize()
        n_placed += 1

    # 1b) REVIEW plots: merge at their located UTM position but on their OWN
    # REVIEW_FMB_<survey> layer, so they are VISIBLE for human seeding yet do not
    # stack on the confident tiling (Bug #2). review_specs = [(georef_dxf, survey)].
    n_review = 0
    for gp, sn in (review_specs or []):
        gp = Path(gp)
        try:
            src = ezdxf.readfile(str(gp))
        except Exception:  # noqa: BLE001
            continue
        layer = f"REVIEW_FMB_{sn}"
        if layer not in src.layers:
            # ORANGE (30), NOT yellow -- yellow (2) is the S3 cadastral parcel colour,
            # so a yellow review layer looks (wrongly) like leaked S3 tiles. These are
            # the FMB's OWN geometry; orange reads as "FMB needs human confirmation".
            src.layers.add(layer, color=30)  # in SRC so the Importer copies it
        for e in src.modelspace():
            try:
                e.dxf.layer = layer
                e.dxf.color = 256            # BYLAYER -> inherit the orange layer colour
            except Exception:  # noqa: BLE001
                pass
        importer = Importer(src, base)
        importer.import_modelspace()
        importer.finalize()
        n_review += 1
    if n_review:
        _log.info("Combined: %d REVIEW plots on REVIEW_FMB_* layers", n_review)

    # 2) Stage the uncontrolled plots in a labelled grid east of the survey.
    n_staged = 0
    if staged_specs:
        ext = _msp_extent(base) or (0.0, 0.0, 100.0, 100.0)
        bx0, by0, bx1, by1 = ext
        survey_w = max(bx1 - bx0, 1.0)
        survey_h = max(by1 - by0, 1.0)

        # First pass: read each staged plot + its bbox, to size the grid cell.
        loaded = []
        max_w = max_h = 1.0
        for m1_path, sn in staged_specs:
            try:
                doc = ezdxf.readfile(str(m1_path))
            except Exception as exc:  # noqa: BLE001
                _log.warning("Skipping unreadable staged M1 %s: %s", m1_path, exc)
                continue
            pe = _msp_extent(doc)
            if pe is None:
                continue
            w, h = pe[2] - pe[0], pe[3] - pe[1]
            max_w, max_h = max(max_w, w), max(max_h, h)
            loaded.append((doc, str(sn), pe))

        if loaded:
            gap = 0.25
            cell_w = max_w * (1.0 + gap)
            cell_h = max_h * (1.0 + gap)
            ncols = max(1, int(round(survey_w / cell_w))) if survey_w > cell_w else 4
            ncols = min(ncols, max(1, len(loaded)))
            # Band starts one survey-width-margin east of the survey, top-aligned.
            band_x0 = bx1 + 0.10 * survey_w + cell_w * 0.5
            band_y1 = by1

            for k, (doc, sn, pe) in enumerate(loaded):
                row, col = divmod(k, ncols)
                # Target lower-left of this cell.
                tx = band_x0 + col * cell_w
                ty = band_y1 - (row + 1) * cell_h
                dx = tx - pe[0]
                dy = ty - pe[1]
                layer = f"STAGED_FMB_{sn}"
                if layer not in base.layers:
                    base.layers.add(layer, color=1)   # RED = parked, not georeferenced
                # Translate + relayer every modelspace entity, then import.
                m = Matrix44.translate(dx, dy, 0.0)
                src_msp = doc.modelspace()
                for e in src_msp:
                    try:
                        e.transform(m)
                    except Exception:  # noqa: BLE001 - skip non-transformable
                        continue
                    try:
                        e.dxf.layer = layer
                        e.dxf.color = 256             # BYLAYER -> inherit red
                    except Exception:  # noqa: BLE001
                        pass
                importer = Importer(doc, base)
                importer.import_modelspace()
                importer.finalize()
                # Bold survey-number tag above the staged plot.
                base_msp.add_text(
                    f"FMB {sn} (STAGED - needs 2-pt seed)",
                    dxfattribs={"layer": layer,
                                "insert": (tx, ty + (pe[3] - pe[1]) + cell_h * 0.05),
                                "height": max(max_h * 0.04, 1.0)},
                )
                n_staged += 1

    base.saveas(str(output_path))
    _log.info("FULL combined village DXF: %s (%d placed at UTM + %d staged = %d FMBs)",
              output_path, n_placed, n_staged, n_placed + n_staged)
    return output_path
