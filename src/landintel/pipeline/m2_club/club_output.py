"""Write the new M2 deliverable: every FMB CLUBBED into ONE georeferenced file.

Unlike the M3 combined file, there is NO surveyor base canvas here -- the clubbed
drawing is built from the FMB plots alone. Placed plots are merged at their true
UTM position; plots M2 could not georeference are STAGED in a labelled grid beside
the placed cluster (never dropped, never guessed onto a position). Companion
GeoJSON (WGS84, for any viewer) and a corner-points CSV/GeoJSON are written too.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import ezdxf
import numpy as np
from ezdxf.addons import Importer
from ezdxf.math import Matrix44

from ..m2_georef.output_dxf import write_georef_dxf
from .boundary_snap import DEFAULT_TOL, SnapStats, snap_shared_boundaries
from .edge_align import align_shared_edges
from .placement import CandidatePlacement, ClubResult

_log = logging.getLogger(__name__)

DEFAULT_CRS = "EPSG:32643"


def _safe_saveas(doc, output_path) -> Path:
    """Save a DXF, but if the target is LOCKED (open in AutoCAD -> PermissionError) write to a
    timestamped sibling instead of crashing the whole run. Returns the path actually written."""
    output_path = Path(output_path)
    try:
        doc.saveas(str(output_path))
        return output_path
    except PermissionError:
        import time
        alt = output_path.with_name(f"{output_path.stem}_{time.strftime('%H%M%S')}{output_path.suffix}")
        doc.saveas(str(alt))
        _log.warning("%s was locked (open in CAD?); wrote %s instead", output_path.name, alt.name)
        return alt


def write_plot_dxf(m1_dxf_path, placement: CandidatePlacement, output_path, crs: str):
    """Write one georeferenced FMB DXF from a CandidatePlacement (rigid -> UTM)."""
    return write_georef_dxf(
        m1_dxf_path=m1_dxf_path,
        output_path=output_path,
        adjusted_stone_positions=placement.adjusted,
        original_stone_positions=_orig_positions(m1_dxf_path, placement),
        stone_label_to_index=_label_index(m1_dxf_path),
        R=placement.R, s=placement.s, t=placement.t, crs=crs,
        corner_ring=placement.corner_ring,
    )


def _orig_positions(m1_dxf_path, placement: CandidatePlacement) -> np.ndarray:
    from ..m2_georef.extract_m1 import extract_m1_dxf
    try:
        return extract_m1_dxf(m1_dxf_path).stone_positions()
    except Exception:  # noqa: BLE001
        # adjusted is already UTM; identity original keeps write_georef_dxf safe.
        return placement.adjusted


def _label_index(m1_dxf_path) -> dict[str, int]:
    from ..m2_georef.extract_m1 import extract_m1_dxf
    try:
        m1 = extract_m1_dxf(m1_dxf_path)
        return {s.label: s.index for s in m1.stones}
    except Exception:  # noqa: BLE001
        return {}


def _import_as_block(base, src_doc, block_name: str) -> str:
    """Import a source DXF's modelspace into a NEW block of ``base`` and drop ONE block
    reference at the origin.

    The FMB's entities keep their true UTM coordinates (block base point (0,0), insert
    (0,0)), but the whole plot now selects as a SINGLE object in CAD instead of loose
    lines -- the client's "every FMB should be a block, not separate lines". Returns the
    actual (deduplicated) block name used.
    """
    name, k = block_name, 1
    while name in base.blocks:
        k += 1
        name = f"{block_name}_{k}"
    blk = base.blocks.new(name=name)
    imp = Importer(src_doc, base)
    imp.import_entities(list(src_doc.modelspace()), target_layout=blk)
    imp.finalize()
    base.modelspace().add_blockref(name, insert=(0, 0, 0))
    return name


def _msp_extent(doc):
    import ezdxf.bbox
    try:
        bb = ezdxf.bbox.extents(doc.modelspace(), fast=True)
    except Exception:  # noqa: BLE001
        return None
    if not bb.has_data:
        return None
    return (float(bb.extmin.x), float(bb.extmin.y),
            float(bb.extmax.x), float(bb.extmax.y))


def club_dxf(
    placed_specs: list[tuple[str, str]],
    staged_specs: list[tuple[str, str]],
    output_path,
    crs: str = DEFAULT_CRS,
    review_specs: list[tuple[str, str]] | None = None,
) -> Path:
    """Club FMBs into one DXF (no surveyor base).

    placed_specs  : [(georef_dxf_path, survey)]  merged at their UTM position.
    review_specs  : [(georef_dxf_path, survey)]  merged at UTM, own REVIEW_FMB_<s> layer.
    staged_specs  : [(m1_dxf_path, survey)]      no UTM control -> grid band, RED layer.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = ezdxf.new()
    from ..m2_georef.output_dxf import _set_crs_in_header
    _set_crs_in_header(base, crs)
    base_msp = base.modelspace()

    n_placed = 0
    for gp, sn in placed_specs:
        try:
            src = ezdxf.readfile(str(gp))
        except Exception as exc:  # noqa: BLE001
            _log.warning("club: skipping unreadable placed DXF %s: %s", gp, exc)
            continue
        _import_as_block(base, src, f"FMB_{sn}")
        n_placed += 1

    n_review = 0
    for gp, sn in (review_specs or []):
        try:
            src = ezdxf.readfile(str(gp))
        except Exception:  # noqa: BLE001
            continue
        # REVIEW plots ARE georeferenced at their cadastral position (they are only
        # lower-confidence, and are tracked as REVIEW in clubbed_points.csv + the
        # dispositions/verify). Render them with their NORMAL layers/colours -- the
        # old behaviour flattened every entity onto one orange layer, which made a
        # thin sliver plot (e.g. 776) read as a stray orange line across the map.
        # The block name still encodes REVIEW_FMB_<sn> for traceability in CAD.
        _import_as_block(base, src, f"REVIEW_FMB_{sn}")
        n_review += 1

    # Stage the uncontrolled plots in a labelled grid EAST of the placed cluster.
    n_staged = 0
    if staged_specs:
        ext = _msp_extent(base) or (0.0, 0.0, 100.0, 100.0)
        bx0, by0, bx1, by1 = ext
        survey_w = max(bx1 - bx0, 1.0)

        loaded, max_w, max_h = [], 1.0, 1.0
        for m1_path, sn in staged_specs:
            try:
                doc = ezdxf.readfile(str(m1_path))
            except Exception as exc:  # noqa: BLE001
                _log.warning("club: skipping unreadable staged M1 %s: %s", m1_path, exc)
                continue
            pe = _msp_extent(doc)
            if pe is None:
                continue
            max_w = max(max_w, pe[2] - pe[0])
            max_h = max(max_h, pe[3] - pe[1])
            loaded.append((doc, str(sn), pe))

        if loaded:
            gap = 0.25
            cell_w = max_w * (1.0 + gap)
            cell_h = max_h * (1.0 + gap)
            ncols = max(1, int(round(survey_w / cell_w))) if survey_w > cell_w else 4
            ncols = min(ncols, max(1, len(loaded)))
            band_x0 = bx1 + 0.10 * survey_w + cell_w * 0.5
            band_y1 = by1
            for k, (doc, sn, pe) in enumerate(loaded):
                row, col = divmod(k, ncols)
                tx = band_x0 + col * cell_w
                ty = band_y1 - (row + 1) * cell_h
                m = Matrix44.translate(tx - pe[0], ty - pe[1], 0.0)
                layer = f"STAGED_FMB_{sn}"
                if layer not in base.layers:
                    base.layers.add(layer, color=1)   # RED = parked, not georeferenced
                for e in doc.modelspace():
                    try:
                        e.transform(m)
                    except Exception:  # noqa: BLE001
                        continue
                    try:
                        e.dxf.layer = layer
                        e.dxf.color = 256
                    except Exception:  # noqa: BLE001
                        pass
                _import_as_block(base, doc, f"STAGED_FMB_{sn}")
                base_msp.add_text(
                    f"FMB {sn} (STAGED - needs seed)",
                    dxfattribs={"layer": layer,
                                "insert": (tx, ty + (pe[3] - pe[1]) + cell_h * 0.05),
                                "height": max(max_h * 0.04, 1.0)})
                n_staged += 1

    output_path = _safe_saveas(base, output_path)
    _log.info("Clubbed M2 DXF: %s (%d placed + %d review + %d staged)",
              output_path, n_placed, n_review, n_staged)
    return output_path


def _to_wgs84(coords: list[tuple[float, float]], crs: str):
    try:
        from pyproj import Transformer
    except Exception:  # noqa: BLE001
        return [(x, y) for x, y in coords]   # best-effort: leave as-is
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return [tuple(tr.transform(x, y)) for x, y in coords]


def write_geojson(results: list[ClubResult], output_path, crs: str = DEFAULT_CRS) -> Path:
    """Clubbed parcels as WGS84 GeoJSON (placed plots only), for any viewer."""
    output_path = Path(output_path)
    feats = []
    for r in results:
        if not r.placed or r.placement is None:
            continue
        ring = [(float(x), float(y)) for x, y in r.placement.corner_points()]
        if len(ring) < 3:
            continue
        ring_wgs = _to_wgs84(ring, crs)
        if ring_wgs and ring_wgs[0] != ring_wgs[-1]:
            ring_wgs.append(ring_wgs[0])
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in ring_wgs]]},
            "properties": {
                "survey_number": r.survey_number,
                "method": r.method,
                "recommendation": r.recommendation,
                "corroborated_by": r.corroborated_by,
            },
        })
    output_path.write_text(json.dumps(
        {"type": "FeatureCollection", "crs_source": crs, "features": feats},
        indent=2, ensure_ascii=False))
    return output_path


def write_points_csv(results: list[ClubResult], output_path, crs: str = DEFAULT_CRS) -> Path:
    """Corner-stone coordinates (UTM + WGS84) of every placed plot."""
    output_path = Path(output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["survey_number", "corner_index", "x_utm", "y_utm",
                    "lon", "lat", "method", "recommendation"])
        for r in results:
            if not r.placed or r.placement is None:
                continue
            ring = [(float(x), float(y)) for x, y in r.placement.corner_points()]
            ring_wgs = _to_wgs84(ring, crs)
            for i, ((xu, yu), (lon, lat)) in enumerate(zip(ring, ring_wgs)):
                w.writerow([r.survey_number, i, f"{xu:.3f}", f"{yu:.3f}",
                            f"{lon:.7f}", f"{lat:.7f}", r.method, r.recommendation])
    return output_path


def snap_and_rewrite(
    results: list[ClubResult],
    output_dir,
    crs: str = DEFAULT_CRS,
    *,
    enable: bool = True,
    tol: float = DEFAULT_TOL,
    truth_stones=None,
) -> SnapStats:
    """Snap shared boundaries of adjacent ACCEPT plots, then RE-WRITE every clubbed
    deliverable so the DXF / GeoJSON / CSV all reflect the edge-sharing geometry.

    This is the inter-plot QUALITY pass: ``club_pipeline`` seats each FMB rigidly on its
    own cadastral parcel, so neighbours land near -- but not exactly on -- a common edge.
    :func:`snap_shared_boundaries` clusters coincident corners across neighbours and snaps
    them to one position (0-FP guards revert any unsafe plot), then this re-emits:

      * ``georef_<m1>.dxf``    one per placed/review plot (from the snapped placement)
      * ``clubbed_village.dxf`` clubbed from those per-plot DXFs
      * ``clubbed.geojson`` / ``clubbed_points.csv``  snapped corner rings

    ``enable=False`` makes it a no-op (returns empty stats) so the snap is opt-in-able.
    Call AFTER ``club_pipeline`` and BEFORE consuming the outputs. Returns the SnapStats.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not enable:
        return SnapStats(n_plots=sum(1 for r in results if r.placed))

    # QUALITY pass, rigid + 0-FP-gated:
    #  0) stone_refine (ONLY if surveyor ground-truth stones supplied) -- snap the plots
    #     that confidently match the real stones to <2 m, then PROPAGATE that accuracy to
    #     their neighbours (anchored edge-align), re-refining each pass. This is the
    #     manual FMBS_STONES_MATCH and the only route to sub-2 m placement.
    #  1) edge_align  -- translate whole plots onto corroborated shared boundaries.
    #  2) snap_shared_boundaries -- snap the now-adjacent corners to exact coincidence.
    import numpy as _np
    anchored: set[str] = set()
    if truth_stones is not None and len(_np.asarray(truth_stones)) >= 2:
        from .stone_refine import refine_to_stones, resolve_overlaps
        truth = _np.asarray(truth_stones, float)
        # Snapshot the clean cadastre tiling so a bad stone-snap can be un-tiled later.
        originals = {r.survey_number: (_np.asarray(r.placement.R, float).copy(),
                                       _np.asarray(r.placement.t, float).copy(),
                                       r.placement.adjusted.copy())
                     for r in results if r.placed and r.placement is not None}
        for _pass in range(4):
            rf = refine_to_stones(results, truth, skip=anchored)
            newly = rf.anchored - anchored
            anchored |= rf.anchored
            align_shared_edges(results, fixed=anchored)   # size-relative caps (not tuned)
            if not newly and _pass > 0:
                break
        _log.info("stone_refine+propagate: %d plots anchored to surveyor stones", len(anchored))

    fixed = anchored or None
    align = align_shared_edges(results, fixed=fixed)
    stats = snap_shared_boundaries(results, tol=tol, fixed=fixed)
    if anchored:
        # Final 0-FP guard: revert any plot that ended up stacked on another to its clean
        # cadastre seat, so the delivered set is a non-overlapping tiling.
        resolve_overlaps(results, originals, anchored)
    stats.n_edge_constraints = align.n_constraints
    stats.n_edge_moved = align.n_moved
    stats.max_edge_move = align.max_move
    stats.n_anchored = len(anchored)

    # Re-write per-plot DXFs from the (now snapped) placements, then re-club them. Mirror
    # the pipeline's placed / review / staged routing so the deliverable is consistent.
    placed_specs, review_specs, staged_specs = [], [], []
    for r in results:
        if r.placement is not None and r.recommendation in (
                "ACCEPT", "ACCEPT_SEEDED", "REVIEW"):
            try:
                out = output_dir / f"georef_{Path(r.m1_file).stem}.dxf"
                write_plot_dxf(r.m1_file, r.placement, out, crs)
                r.output_file = str(out)
                if r.recommendation == "REVIEW":
                    review_specs.append((str(out), r.survey_number))
                else:
                    placed_specs.append((str(out), r.survey_number))
            except Exception as exc:  # noqa: BLE001
                _log.warning("snap_and_rewrite: write failed for %s: %s",
                             r.survey_number, exc)
                staged_specs.append((r.m1_file, r.survey_number))
        else:
            staged_specs.append((r.m1_file, r.survey_number))

    club_dxf(placed_specs, staged_specs, output_dir / "clubbed_village.dxf",
             crs=crs, review_specs=review_specs)
    write_geojson(results, output_dir / "clubbed.geojson", crs=crs)
    write_points_csv(results, output_dir / "clubbed_points.csv", crs=crs)
    return stats
