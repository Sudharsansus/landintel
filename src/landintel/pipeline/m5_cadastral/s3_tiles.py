"""S3 cadastral tile source: locate parcels by survey number from public tiles.

The TN cadastral web tiles at
``https://s3.../prod-assets.mypropertyqr.in/village_border/{x}/{y}.png`` (z=18,
public, no auth) render yellow parcel boundaries with orange survey-number labels.
They are slippy-map tiles, so every pixel has a known WGS84 position -> reproject
to the surveyor's UTM frame (EPSG:32643 for INGUR). VALIDATED: our surveyor-frame
placement of plot 773 lands exactly on the S3 parcel labelled 773, so this cadastre
is co-registered with the raw data DXF and supplies the off-corridor footprints the
corridor survey never traced.

This module:
  1. downloads + caches the tiles covering a UTM bbox,
  2. reads the orange survey-number labels with PaddleOCR, keeping only the KNOWN
     plot numbers we are placing (targeted -> robust to OCR noise), and records each
     label's UTM position,
  3. vectorises the yellow boundary lines into parcel polygons and assigns each
     located label to the polygon enclosing it,
  4. exposes ``S3CadastralSource.get(survey_no) -> CadastralParcel`` for M2.

Identity (which parcel is which survey number) comes from the label OCR; geometry
(the parcel outline) comes from the vectorised yellow lines; both are in UTM.
"""

from __future__ import annotations

import logging
import math
import re
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from .source import CadastralParcel, CadastralSource, _norm_survey

_log = logging.getLogger(__name__)

S3_BASE = "https://s3.ap-south-2.amazonaws.com/prod-assets.mypropertyqr.in/village_border/"
TILE = 256
ZOOM = 18

# Orange/red survey-label HSV ranges (two bands around the hue wheel's red end).
_ORANGE = [(np.array([0, 60, 90]), np.array([35, 255, 255])),
           (np.array([150, 60, 90]), np.array([180, 255, 255]))]
# Yellow boundary-line HSV range.
_YELLOW = (np.array([20, 80, 90]), np.array([45, 255, 255]))


class TileGrid:
    """Web-mercator z18 tile grid <-> a UTM CRS (default EPSG:32643)."""

    def __init__(self, crs: str = "EPSG:32643", z: int = ZOOM):
        from pyproj import Transformer
        self.crs = crs
        self.z = z
        self.world_px = (2.0 ** z) * TILE
        self._to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        self._to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    def utm_to_px(self, x: float, y: float) -> tuple[float, float]:
        lon, lat = self._to_ll.transform(x, y)
        r = math.radians(lat)
        gx = (lon + 180.0) / 360.0 * self.world_px
        gy = (1.0 - math.asinh(math.tan(r)) / math.pi) / 2.0 * self.world_px
        return gx, gy

    def px_to_utm(self, gx: float, gy: float) -> tuple[float, float]:
        lon = gx / self.world_px * 360.0 - 180.0
        lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * gy / self.world_px))))
        return self._to_utm.transform(lon, lat)


def download_tiles(
    bbox_utm: tuple[float, float, float, float],
    grid: TileGrid,
    cache_dir: str | Path,
    buffer_m: float = 150.0,
) -> dict[tuple[int, int], Path]:
    """Download + cache all z18 tiles covering ``bbox_utm`` (+buffer). Returns
    {(tx,ty): png_path} for tiles that exist (skips 404/blank)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = bbox_utm
    corners = [grid.utm_to_px(x0 - buffer_m, y0 - buffer_m),
               grid.utm_to_px(x1 + buffer_m, y1 + buffer_m)]
    gxs = [c[0] for c in corners]
    gys = [c[1] for c in corners]
    tx0, tx1 = int(min(gxs) // TILE), int(max(gxs) // TILE)
    ty0, ty1 = int(min(gys) // TILE), int(max(gys) // TILE)
    out: dict[tuple[int, int], Path] = {}
    n_dl = 0
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            p = cache_dir / f"{tx}_{ty}.png"
            if not p.exists():
                try:
                    req = urllib.request.Request(f"{S3_BASE}{tx}/{ty}.png",
                                                 headers={"User-Agent": "curl/8"})
                    data = urllib.request.urlopen(req, timeout=30).read()
                    if len(data) < 800:        # blank/placeholder tile
                        continue
                    p.write_bytes(data)
                    n_dl += 1
                except Exception:              # noqa: BLE001 - 404 / network
                    continue
            if p.exists():
                out[(tx, ty)] = p
    _log.info("S3 tiles: %d covering bbox (%d newly downloaded)", len(out), n_dl)
    return out


def _orange_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = np.zeros(bgr.shape[:2], np.uint8)
    for lo, hi in _ORANGE:
        m = cv2.bitwise_or(m, cv2.inRange(hsv, lo, hi))
    return m


def _fuzzy_survey_match(ocr_text: str, known_surveys: set[str]) -> str | None:
    """Match OCR text to a KNOWN survey number, tolerating common OCR errors.

    Only ever returns a number already in ``known_surveys`` (closed set), so a
    fuzzy read can never invent a survey -- it can only recover a known one that
    OCR garbled (e.g. "10/19" or "10 19" -> 1019, "O"->0, "I"->1).
    """
    clean = str(ocr_text).strip()
    cands: set[str] = set()
    sn = _norm_survey(clean)
    if sn and sn in known_surveys and clean == sn:
        cands.add(sn)
    fixed = clean.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")
    for cand in (clean, fixed):
        c2 = cand.replace(" ", "").replace("/", "").replace("\\", "")
        if c2 in known_surveys:
            cands.add(c2)
        parts = re.findall(r"\d+", cand)
        if len(parts) >= 2 and "".join(parts) in known_surveys:
            cands.add("".join(parts))
    # Prefer the LONGEST valid candidate so "10" can't shadow "1019".
    return max(cands, key=len) if cands else None


def ocr_locate_labels(
    tiles: dict[tuple[int, int], Path],
    grid: TileGrid,
    known_surveys: set[str],
    engine=None,
    upscale: int = 4,
) -> dict[str, tuple[float, float]]:
    """OCR the orange survey labels, keeping only KNOWN survey numbers, and return
    {survey_no: (utm_x, utm_y)} at each label's centre. Targeting the known set
    makes this robust to OCR noise and to the 2-digit in-parcel subdivision labels.
    """
    if engine is None:
        engine = _default_engine()
    found: dict[str, list[tuple[float, float, float]]] = {}   # sn -> [(x,y,conf)]
    for (tx, ty), path in tiles.items():
        img = cv2.imread(str(path))
        if img is None:
            continue
        boost = img.copy()
        boost[_orange_mask(img) > 0] = (255, 255, 255)
        up = cv2.resize(boost, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        for r in engine.predict(cv2.cvtColor(cv2.cvtColor(up, cv2.COLOR_BGR2GRAY),
                                             cv2.COLOR_GRAY2BGR)):
            d = r if isinstance(r, dict) else r.json.get("res", r)
            texts = d.get("rec_texts", [])
            polys = d.get("rec_polys", d.get("dt_polys", []))
            scores = d.get("rec_scores", [1.0] * len(texts))
            for t, poly, sc in zip(texts, polys, scores):
                sn = _fuzzy_survey_match(str(t).strip(), known_surveys)
                if sn is None:
                    continue
                pts = np.array(poly, dtype=float)
                cxp = pts[:, 0].mean() / upscale + tx * TILE
                cyp = pts[:, 1].mean() / upscale + ty * TILE
                ux, uy = grid.px_to_utm(cxp, cyp)
                found.setdefault(sn, []).append((ux, uy, float(sc)))
    # Keep the highest-confidence reading per survey number.
    best = {sn: max(v, key=lambda z: z[2])[:2] for sn, v in found.items()}
    _log.info("S3 OCR: located %d/%d known survey numbers", len(best), len(known_surveys))
    return best, found


def _select_in_fence(found: dict, village_fence) -> dict[str, tuple[float, float]]:
    """For each survey, pick the highest-confidence reading INSIDE the village fence.

    Kills cross-village duplicate-number placements (e.g. a "9" label in an adjacent
    village): a reading is kept only if it lies within the target village extent.
    """
    out: dict[str, tuple[float, float]] = {}
    for sn, reads in found.items():
        cands = reads
        if village_fence is not None:
            cands = [r for r in reads if village_fence.contains(Point(r[0], r[1]))]
        if cands:
            x, y, _ = max(cands, key=lambda z: z[2])
            out[sn] = (x, y)
    return out


def ocr_second_pass(
    tiles: dict[tuple[int, int], Path],
    grid: TileGrid,
    missing: set[str],
    labels: dict[str, tuple[float, float]],
    engine=None,
    upscale: int = 5,
    village_fence=None,
) -> dict[str, tuple[float, float]]:
    """Aggressive second OCR pass for surveys missed in the first pass.

    Higher upscale + adaptive threshold + morphological close to recover small or
    broken labels. Fuzzy-matched against the MISSING set only, and fenced to the
    village, so it cannot introduce a cross-village or wrong-survey placement.
    """
    if not missing:
        return labels
    _log.info("S3 OCR second pass: %d missing surveys: %s", len(missing),
              sorted(missing, key=lambda s: (len(s), s)))
    if engine is None:
        engine = _default_engine()
    for (tx, ty), path in tiles.items():
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        binary[_orange_mask(img) > 0] = 255
        up = cv2.resize(binary, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        for r in engine.predict(cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)):
            d = r if isinstance(r, dict) else r.json.get("res", r)
            texts = d.get("rec_texts", [])
            polys = d.get("rec_polys", d.get("dt_polys", []))
            for t, poly in zip(texts, polys):
                sn = _fuzzy_survey_match(str(t).strip(), missing)
                if sn is None or sn in labels:
                    continue
                pts = np.array(poly, dtype=float)
                ux, uy = grid.px_to_utm(pts[:, 0].mean() / upscale + tx * TILE,
                                        pts[:, 1].mean() / upscale + ty * TILE)
                if village_fence is not None and not village_fence.contains(Point(ux, uy)):
                    continue
                labels[sn] = (ux, uy)
                _log.info("  second pass found %s at (%.0f, %.0f)", sn, ux, uy)
    _log.info("S3 OCR after second pass: %d known surveys located", len(labels))
    return labels


def ocr_multi_angle_pass(
    tiles: dict[tuple[int, int], Path],
    grid: TileGrid,
    missing: set[str],
    labels: dict[str, tuple[float, float]],
    engine=None,
    angles=(0, 90, 180, 270, 45, 135),
    upscale: int = 5,
    village_fence=None,
) -> dict[str, tuple[float, float]]:
    """Multi-angle OCR recovery for surveys missed by the upright passes.

    PP-OCRv5 reads near-horizontal numbers well and steeply rotated ones poorly
    (~24% recall). Cadastral labels sit at parcel-specific angles, so OCR each tile
    at several rotations and map detections back to UTM. Targeted to the MISSING set
    and fenced to the village, so it can neither invent a survey nor place one
    cross-village. VALIDATED on INGUR: recovered 6 missed labels and re-found the
    known corridor surveys (773/724/730) at their true positions (<30 m).
    """
    if not missing:
        return labels
    if engine is None:
        engine = _default_engine()
    # Restrict to tiles that actually fall in the village (huge speed-up).
    if village_fence is not None:
        fb = village_fence.buffer(80)
        work = {k: p for k, p in tiles.items()
                if fb.contains(Point(*grid.px_to_utm((k[0] + 0.5) * TILE,
                                                      (k[1] + 0.5) * TILE)))}
    else:
        work = tiles
    _log.info("S3 OCR multi-angle pass: %d missing surveys over %d fenced tiles "
              "at angles %s", len(missing), len(work), list(angles))
    found: dict[str, list[tuple[float, float, float]]] = {}
    for (tx, ty), path in work.items():
        img = cv2.imread(str(path))
        if img is None:
            continue
        boost = img.copy()
        boost[_orange_mask(img) > 0] = (255, 255, 255)
        up = cv2.resize(boost, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(cv2.cvtColor(up, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        H, W = gray.shape[:2]
        center = (W / 2.0, H / 2.0)
        for ang in angles:
            if ang % 360 == 0:
                rot, minv = gray, None
            else:
                m = cv2.getRotationMatrix2D(center, ang, 1.0)
                rot = cv2.warpAffine(gray, m, (W, H), borderValue=(255, 255, 255))
                minv = cv2.invertAffineTransform(m)
            for r in engine.predict(rot):
                d = r if isinstance(r, dict) else r.json.get("res", r)
                texts = d.get("rec_texts", [])
                polys = d.get("rec_polys", d.get("dt_polys", []))
                scores = d.get("rec_scores", [1.0] * len(texts))
                for t, poly, sc in zip(texts, polys, scores):
                    sn = _fuzzy_survey_match(str(t).strip(), missing)
                    if sn is None or sn in labels:
                        continue
                    pts = np.array(poly, float)
                    cxr, cyr = pts[:, 0].mean(), pts[:, 1].mean()
                    if minv is not None:
                        v = minv @ np.array([cxr, cyr, 1.0])
                        cxr, cyr = float(v[0]), float(v[1])
                    ux, uy = grid.px_to_utm(cxr / upscale + tx * TILE,
                                            cyr / upscale + ty * TILE)
                    if village_fence is not None and not village_fence.contains(Point(ux, uy)):
                        continue
                    found.setdefault(sn, []).append((ux, uy, float(sc)))
    for sn, reads in found.items():
        x, y, _ = max(reads, key=lambda z: z[2])
        labels[sn] = (x, y)
        _log.info("  multi-angle recovered %s at (%.0f, %.0f) from %d reads",
                  sn, x, y, len(reads))
    _log.info("S3 OCR after multi-angle pass: %d known surveys located", len(labels))
    return labels


def _warm_core_mask(bgr: np.ndarray) -> np.ndarray:
    """Yellow-green glyph CORE (hue 25-50, saturated, bright). Some survey labels are
    drawn with a yellow-green digit fill + orange halo (not solid orange); boosting
    only the orange halo merges the digits into one blob. Isolating the digit core
    keeps them separable for OCR. (Agent C finding for INGUR survey 723.)"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    core = ((h >= 25) & (h <= 50) & (s > 120) & (v > 150)).astype(np.uint8) * 255
    return core


def _stitch_neighbourhood(tiles: dict, tx: int, ty: int, rad: int = 1):
    """Stitch a (2*rad+1)^2 tile canvas centred at (tx,ty). A label straddling a tile
    edge is split by per-tile OCR; stitching keeps it whole. Returns (canvas_bgr,
    ox, oy) where (ox,oy) is the GLOBAL tile-pixel coordinate of canvas[0,0], or
    (None,0,0) if the centre tile is missing."""
    if (tx, ty) not in tiles:
        return None, 0, 0
    size = (2 * rad + 1) * TILE
    canvas = np.full((size, size, 3), 255, np.uint8)
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            p = tiles.get((tx + dx, ty + dy))
            if p is None:
                continue
            im = cv2.imread(str(p))
            if im is not None:
                canvas[(dy + rad) * TILE:(dy + rad + 1) * TILE,
                       (dx + rad) * TILE:(dx + rad + 1) * TILE] = im
    return canvas, (tx - rad) * TILE, (ty - rad) * TILE


def _aggressive_scan(engine, base_bgr, req_up, offx, offy, angles, grid, missing,
                     labels, village_fence, found, pass_tag):
    """OCR ``base_bgr`` at every angle, fuzzy-match each read to a still-MISSING
    survey, fence it, and append (ux, uy, score, pass_tag) to ``found[sn]``.

    ``pass_tag`` records WHICH pass/mode produced the read (e.g. "coarseA",
    "fineB") so the acceptor can require corroboration from two INDEPENDENT passes
    -- a stronger zero-FP guard than two reads from one pass. ``req_up`` is adapted
    down so the upscaled long side stays <= ~3800 px (a 3x3 stitch * 6-8 would
    otherwise be skipped)."""
    h0, w0 = base_bgr.shape[:2]
    up = max(2.0, min(float(req_up), 3800.0 / max(h0, w0)))
    big = cv2.resize(base_bgr, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    H, W = big.shape[:2]
    center = (W / 2.0, H / 2.0)
    for ang in angles:
        if ang % 360 == 0:
            rot, minv = big, None
        else:
            m = cv2.getRotationMatrix2D(center, ang, 1.0)
            rot = cv2.warpAffine(big, m, (W, H), borderValue=(255, 255, 255))
            minv = cv2.invertAffineTransform(m)
        for r in engine.predict(rot):
            d = r if isinstance(r, dict) else r.json.get("res", r)
            texts = d.get("rec_texts", [])
            polys = d.get("rec_polys", d.get("dt_polys", []))
            scores = d.get("rec_scores", [1.0] * len(texts))
            for t, poly, sc in zip(texts, polys, scores):
                sn = _fuzzy_survey_match(str(t).strip(), missing)
                if sn is None or sn in labels:
                    continue
                pts = np.array(poly, float)
                cxr, cyr = pts[:, 0].mean(), pts[:, 1].mean()
                if minv is not None:
                    vv = minv @ np.array([cxr, cyr, 1.0])
                    cxr, cyr = float(vv[0]), float(vv[1])
                ux, uy = grid.px_to_utm(offx + cxr / up, offy + cyr / up)
                if village_fence is not None and not village_fence.contains(Point(ux, uy)):
                    continue
                found.setdefault(sn, []).append((ux, uy, float(sc), pass_tag))


def _stitch_modeA(tiles, tx, ty):
    """3x3 STITCHED + orange boost (edge-straddling labels, e.g. 1019/1022).
    Returns (gray_bgr, ox, oy) or (None, 0, 0)."""
    canvas, ox, oy = _stitch_neighbourhood(tiles, tx, ty, rad=1)
    if canvas is None:
        return None, 0, 0
    b = canvas.copy()
    b[_orange_mask(canvas) > 0] = (255, 255, 255)
    return cv2.cvtColor(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), ox, oy


def _warm_modeB(img):
    """PER-TILE warm-core glyph mask (yellow-green fill, e.g. 723)."""
    return cv2.cvtColor(255 - _warm_core_mask(img), cv2.COLOR_GRAY2BGR)


def _build_mosaics(work: dict[tuple[int, int], Path]):
    """Stitch the focused tiles into ONE big canvas ONCE and derive the two coarse
    boost mosaics (orange-boosted + warm-core). Returns (orangeA, warmB, ox, oy) in
    GLOBAL tile-pixel coordinates, or (None, None, 0, 0) if empty.

    This replaces per-tile 3x3 stitching in the coarse pre-scan: a label straddling a
    tile edge is whole in the mosaic, and each source tile is read/decoded ONCE
    (instead of ~9x by repeated `_stitch_neighbourhood`), which is the dominant cost."""
    if not work:
        return None, None, 0, 0
    txs = [k[0] for k in work]
    tys = [k[1] for k in work]
    tx0, tx1, ty0, ty1 = min(txs), max(txs), min(tys), max(tys)
    W = (tx1 - tx0 + 1) * TILE
    H = (ty1 - ty0 + 1) * TILE
    canvas = np.full((H, W, 3), 255, np.uint8)
    for (tx, ty), path in work.items():
        im = cv2.imread(str(path))
        if im is not None:
            canvas[(ty - ty0) * TILE:(ty - ty0 + 1) * TILE,
                   (tx - tx0) * TILE:(tx - tx0 + 1) * TILE] = im
    orange = canvas.copy()
    orange[_orange_mask(canvas) > 0] = (255, 255, 255)
    orangeA = cv2.cvtColor(cv2.cvtColor(orange, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    warmB = cv2.cvtColor(255 - _warm_core_mask(canvas), cv2.COLOR_GRAY2BGR)
    return orangeA, warmB, tx0 * TILE, ty0 * TILE


def _coarse_window_scan(engine, mosaic, ox, oy, up, angles, grid, missing, labels,
                        village_fence, found, pass_tag, win=512, step=448):
    """Coarse-OCR a big ``mosaic`` in OVERLAPPING ``win`` px windows stepping ``step``
    px, at ``angles``. Survey labels are small (~30 px), so a window/step with >=128 px
    overlap keeps every label whole in some window. Feeds the same ``found`` reservoir
    via ``_aggressive_scan`` (closed-set + fenced). Much cheaper than upscaling the
    whole mosaic: each window is a small ``up``x crop."""
    H, W = mosaic.shape[:2]
    ys = list(range(0, max(1, H - win + 1), step)) or [0]
    xs = list(range(0, max(1, W - win + 1), step)) or [0]
    if ys[-1] + win < H:
        ys.append(H - win)
    if xs[-1] + win < W:
        xs.append(W - win)
    for y0 in ys:
        for x0 in xs:
            crop = mosaic[y0:y0 + win, x0:x0 + win]
            _aggressive_scan(engine, crop, up, ox + x0, oy + y0, angles, grid,
                             missing, labels, village_fence, found, pass_tag)


def ocr_aggressive_pass(
    tiles: dict[tuple[int, int], Path],
    grid: TileGrid,
    missing: set[str],
    labels: dict[str, tuple[float, float]],
    engine=None,
    angles=range(0, 360, 20),
    village_fence=None,
    focus_radius_m: float = 450.0,
    coarse_angles=range(0, 360, 90),
) -> dict[str, tuple[float, float]]:
    """Last-resort OCR for the few labels the upright + multi-angle passes still miss.

    Two Agent-C fixes layered on the multi-angle recipe: (a) OCR a 3x3 STITCHED
    neighbourhood so an edge-straddling label is not split, and (b) try a SECOND
    colour boost (yellow-green glyph core, not just orange halo). Closed-set + fenced
    -> cannot invent a survey or place one cross-village.

    SPEED (the full fine recipe over every focused tile is ~25 s/tile -> >1 h):
    run it in TWO stages. (1) A CHEAP COARSE pre-scan stitches the focus area into
    ONE mosaic (each tile decoded once, not ~9x by per-tile stitching) and OCRs it
    in overlapping ``win``-px windows at only ``coarse_angles`` -- just to LOCALIZE
    a candidate tile per missing survey. (2) The EXPENSIVE fine recipe (full
    ``angles``, high upscale) runs ONLY on each localized tile's neighbourhood -- a
    handful of tiles, not hundreds. Both stages feed the same reservoir, tagged by
    pass, so cross-pass corroboration can accept a single CLEAN read (the coarse and
    fine passes are independent detectors). ~3 min on INGUR vs ~1 h.

    ZERO-FP acceptance: a survey is recovered only if its closed-set + fenced reads
    are CORROBORATED -- either >=2 reads within 5 m of the best, OR a high-confidence
    (>=0.95) best read seen by two DIFFERENT passes within 10 m. A lone, unconfirmed
    read never creates a label.

    VALIDATED on INGUR: recovers 1019, 1022 (stitched-orange) and 723 (warm-core,
    a single clean read corroborated across the coarse+fine passes) at score ~1.0,
    each <1 m from its seeded position, in a few minutes instead of an hour."""
    if not missing:
        return labels
    if engine is None:
        engine = _default_engine()
    located_pts = [np.array(v, float) for v in labels.values()]
    # Focus on tiles whose centre is within focus_radius of an already-located label.
    work = {}
    for k, p in tiles.items():
        c = np.array(grid.px_to_utm((k[0] + 0.5) * TILE, (k[1] + 0.5) * TILE))
        if village_fence is not None and not village_fence.buffer(80).contains(Point(*c)):
            continue
        if located_pts and min(float(np.hypot(*(c - lp))) for lp in located_pts) > focus_radius_m:
            continue
        work[k] = p
    _log.info("S3 OCR aggressive pass: %d still-missing over %d focused tiles "
              "(coarse pre-scan -> fine on localized neighbourhoods)",
              len(missing), len(work))

    found: dict[str, list[tuple[float, float, float, str]]] = {}

    # --- Stage 1: CHEAP coarse pre-scan to LOCALIZE a candidate tile per survey. ---
    # Build the focus area into ONE mosaic (each tile decoded once) and OCR it in
    # overlapping coarse windows. The orange mosaic catches edge-straddling labels
    # (1019/1022); the warm-core mosaic catches yellow-green-fill labels (723).
    orangeA, warmB, mox, moy = _build_mosaics(work)
    if orangeA is not None:
        _coarse_window_scan(engine, orangeA, mox, moy, 3, coarse_angles, grid,
                            missing, labels, village_fence, found, "coarseA")
        _coarse_window_scan(engine, warmB, mox, moy, 3, coarse_angles, grid,
                            missing, labels, village_fence, found, "coarseB")
    # A coarse read localizes the candidate TILE for each missing survey.
    cand_tile: dict[str, tuple[int, int]] = {}
    cand_score: dict[str, float] = {}
    for sn, reads in found.items():
        if sn not in missing or sn in labels:
            continue
        bx, by, bsc, _ = max(reads, key=lambda z: z[2])
        gx, gy = grid.utm_to_px(bx, by)
        cand_tile[sn] = (int(gx // TILE), int(gy // TILE))
        cand_score[sn] = bsc
    _log.info("S3 aggressive coarse pre-scan localized %d/%d missing: %s",
              len(cand_tile), len(missing), sorted(cand_tile))

    # --- Stage 2: EXPENSIVE fine recipe ONLY on localized neighbourhoods. ---
    # One fine pass per candidate tile (dedup), full angle sweep + high upscale.
    fine_tiles = sorted(set(cand_tile.values()))
    for (tx, ty) in fine_tiles:
        img = cv2.imread(str(tiles[(tx, ty)]))
        if img is None:
            continue
        grayA, ox, oy = _stitch_modeA(tiles, tx, ty)
        if grayA is not None:
            _aggressive_scan(engine, grayA, 6, ox, oy, angles, grid, missing,
                             labels, village_fence, found, "fineA")
        _aggressive_scan(engine, _warm_modeB(img), 8, tx * TILE, ty * TILE,
                         angles, grid, missing, labels, village_fence, found, "fineB")

    # --- Acceptance: corroborated reads only (closed-set + fenced already enforced). ---
    for sn, reads in found.items():
        bx, by, bsc, _ = max(reads, key=lambda z: z[2])
        near = [r for r in reads if np.hypot(r[0] - bx, r[1] - by) <= 5.0]
        near10 = [r for r in reads if np.hypot(r[0] - bx, r[1] - by) <= 10.0]
        # (a) classic cluster agreement: >=2 reads within 5 m.
        cluster_ok = len(near) >= 2
        # (b) a clean read confirmed by a SECOND, independent pass within 10 m.
        passes = {r[3] for r in near10}
        cross_pass_ok = bsc >= 0.95 and len(passes) >= 2
        if cluster_ok or cross_pass_ok:
            labels[sn] = (bx, by)
            _log.info("  aggressive recovered %s at (%.0f, %.0f) sc=%.2f from %d reads "
                      "(%d within 5m, passes=%s)", sn, bx, by, bsc, len(reads),
                      len(near), sorted(passes))
    _log.info("S3 OCR after aggressive pass: %d known surveys located", len(labels))
    return labels


def vectorize_parcels(
    tiles: dict[tuple[int, int], Path],
    grid: TileGrid,
    min_area_m2: float = 150.0,
    max_area_m2: float = 200000.0,
    close_iter: int = 1,
    dilate_iter: int = 1,
) -> list[Polygon]:
    """Vectorise the cadastral PARCELS as the FACES enclosed by the yellow lines.

    A parcel is a region BOUNDED by the boundary lines, not the lines themselves.
    So: build the yellow lines as a 'wall' mask (dilated/closed to bridge small
    gaps), then take connected components of the FREE space -- each interior
    component is one parcel -- and return its (simplified) outer contour as a UTM
    polygon. The unbounded background component and noise are dropped by area.
    """
    if not tiles:
        return []
    txs = [k[0] for k in tiles]
    tys = [k[1] for k in tiles]
    tx0, tx1, ty0, ty1 = min(txs), max(txs), min(tys), max(tys)
    W = (tx1 - tx0 + 1) * TILE
    H = (ty1 - ty0 + 1) * TILE
    canvas = np.zeros((H, W, 3), np.uint8)
    for (tx, ty), path in tiles.items():
        im = cv2.imread(str(path))
        if im is not None:
            canvas[(ty - ty0) * TILE:(ty - ty0 + 1) * TILE,
                   (tx - tx0) * TILE:(tx - tx0 + 1) * TILE] = im
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    walls = cv2.inRange(hsv, *_YELLOW)
    # Close small gaps in the boundary lines so parcels don't bleed together.
    # CLOSE (dilate+erode) bridges gaps WITHOUT net-thickening; explicit DILATE adds
    # thickness that bridges bigger gaps but can pinch a real parcel into fragments
    # (over-segmentation -> area_ratio 4-7x). Tunable so the split/merge tradeoff can
    # be measured; merges that result are caught downstream by the A3 size reject.
    if close_iter:
        walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8),
                                 iterations=close_iter)
    if dilate_iter:
        walls = cv2.dilate(walls, np.ones((3, 3), np.uint8), iterations=dilate_iter)
    free = cv2.bitwise_not(walls)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    # Approx pixel area bounds (z18 ~ 0.6 m/px near this latitude -> ~0.37 m2/px).
    m2_per_px = _approx_m2_per_px(grid, tx0, ty0)
    polys: list[Polygon] = []
    for i in range(1, n):
        x, y, w, h, area_px = stats[i]
        # Skip the unbounded background (spans the whole canvas) + tiny noise.
        if w >= W - 2 or h >= H - 2:
            continue
        if not (min_area_m2 <= area_px * m2_per_px <= max_area_m2):
            continue
        comp = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, epsilon=3.0, closed=True).reshape(-1, 2)
        if len(c) < 4:
            continue
        ring = [grid.px_to_utm(p[0] + tx0 * TILE, p[1] + ty0 * TILE) for p in c]
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        # Simplify away pixel-quantisation noise so FMB-vs-parcel IoU isn't depressed
        # by jagged sub-metre wobble on the cadastral outline.
        poly = poly.simplify(2.0, preserve_topology=True)
        if poly.geom_type == "Polygon" and min_area_m2 <= poly.area <= max_area_m2:
            polys.append(poly)
    _log.info("S3 vectorize: %d parcel polygons (face extraction)", len(polys))
    return polys


def recover_open_parcel(
    tiles: dict[tuple[int, int], Path],
    grid: TileGrid,
    label_utm: tuple[float, float],
    crop_px: int = 420,
    close_kernels: tuple[int, ...] = (5, 9, 15, 21, 27),
    orange_radii_px: tuple[int, ...] = (0, 80, 120, 160),
    min_area_m2: float = 150.0,
) -> list[Polygon]:
    """LOCAL, label-seeded recovery of a parcel whose yellow boundary is OPEN/gapped.

    A minority of cadastral parcels (e.g. INGUR 768/1019/1022/1023) are sub-cells of a
    yellow-bounded super-region whose dividing line has a RASTER GAP at z18 (it bleeds
    open toward a neighbour or an orange road), so the global face vectoriser files the
    label in the unbounded-background blob and yields no right-sized ring -> forced
    REVIEW. This rebuilds JUST that one parcel WITHOUT touching the global segmentation
    of the clean parcels: it works in a small crop around the label, bridges the gaps in
    the parcel's OWN yellow wall with a local morphological close (and, if needed, seals
    the open side with the orange road pixels within a radius of the label), then
    flood-fills the free space from the label point. Only basins that DON'T leak to the
    crop border (i.e. are actually closed) are returned, as UTM polygons.

    Returns ALL closed candidate polygons over the parameter sweep (de-duplicated). The
    caller is expected to keep only one whose rigid fit PASSES the area/scale/residual
    gate -- the bridge can never create a false ACCEPT the gate would not already reject
    (verified on INGUR that cross-village survey 9 yields NO gate-passing closure here),
    so emitting several candidates only ADDS recall, never FP risk.
    """
    if not tiles:
        return []
    txs = [k[0] for k in tiles]
    tys = [k[1] for k in tiles]
    tx0, ty0 = min(txs), min(tys)
    wt = (max(txs) - tx0 + 1) * TILE
    ht = (max(tys) - ty0 + 1) * TILE
    gx, gy = grid.utm_to_px(label_utm[0], label_utm[1])
    cx = int(round(gx - tx0 * TILE))
    cy = int(round(gy - ty0 * TILE))
    x0 = max(0, cx - crop_px)
    y0 = max(0, cy - crop_px)
    x1 = min(wt, cx + crop_px)
    y1 = min(ht, cy + crop_px)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return []
    canvas = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
    for (tx, ty), path in tiles.items():
        gx0 = (tx - tx0) * TILE
        gy0 = (ty - ty0) * TILE
        if gx0 + TILE < x0 or gx0 > x1 or gy0 + TILE < y0 or gy0 > y1:
            continue
        im = cv2.imread(str(path))
        if im is None:
            continue
        sx0, sy0 = max(x0, gx0), max(y0, gy0)
        sx1, sy1 = min(x1, gx0 + TILE), min(y1, gy0 + TILE)
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
            im[sy0 - gy0:sy1 - gy0, sx0 - gx0:sx1 - gx0]
    ox, oy = x0 + tx0 * TILE, y0 + ty0 * TILE      # crop[0,0] global tile-px
    lx, ly = cx - x0, cy - y0
    h, w = canvas.shape[:2]

    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, *_YELLOW)
    base = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    base = cv2.dilate(base, np.ones((3, 3), np.uint8), iterations=1)
    orange = cv2.dilate(_orange_mask(canvas), np.ones((3, 3), np.uint8), iterations=1)

    def _seed(free):
        if free[ly, lx] > 0:
            return (lx, ly)
        for rr in range(1, 60):
            for dy in range(-rr, rr + 1):
                for dx in range(-rr, rr + 1):
                    yy, xx = ly + dy, lx + dx
                    if 0 <= yy < h and 0 <= xx < w and free[yy, xx] > 0:
                        return (xx, yy)
        return None

    def _seed_candidates(free):
        """The label seed PLUS 16-compass directional probes. A survey-number label can sit
        in an open sub-cell or right next to a boundary gap, so flooding from the label LEAKS;
        offsetting the seed in the 16 directions finds the one INSIDE the closed parent. Each
        basin still passes the rigid+seat gate downstream, so extra seeds only ADD recall, 0-FP.
        (Recovered INGUR 698: its true ~52900 m^2 parent closes from a seed ~67 px N of the
        leaking label, not from the label itself.)"""
        seeds = []
        s0 = _seed(free)
        if s0 is not None:
            seeds.append(s0)
        for i in range(16):
            ang = 2.0 * np.pi * i / 16.0
            for rr in (40, 70, 100):
                xx = int(round(lx + np.cos(ang) * rr))
                yy = int(round(ly + np.sin(ang) * rr))
                if 0 <= xx < w and 0 <= yy < h and free[yy, xx] > 0:
                    seeds.append((xx, yy))
        return seeds

    out: list[Polygon] = []
    seen: set[int] = set()
    for close_k in close_kernels:
        for orad in orange_radii_px:
            walls = base
            if close_k > 0:
                walls = cv2.morphologyEx(
                    walls, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
            if orad > 0:
                rm = np.zeros((h, w), np.uint8)
                cv2.circle(rm, (lx, ly), orad, 255, -1)
                walls = cv2.bitwise_or(walls, cv2.bitwise_and(orange, rm))
            free = cv2.bitwise_not(walls)
            for seed in _seed_candidates(free):
                mask = np.zeros((h + 2, w + 2), np.uint8)
                cv2.floodFill(free.copy(), mask, seed, 255, flags=4 | (255 << 8))
                basin = mask[1:-1, 1:-1]
                # A basin touching the crop border is still open (leaked to background).
                if (basin[0, :].any() or basin[-1, :].any()
                        or basin[:, 0].any() or basin[:, -1].any()):
                    continue
                comp = (basin > 0).astype(np.uint8) * 255
                cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                c = max(cnts, key=cv2.contourArea)
                c = cv2.approxPolyDP(c, 3.0, True).reshape(-1, 2)
                if len(c) < 4:
                    continue
                ring = [grid.px_to_utm(p[0] + ox, p[1] + oy) for p in c]
                poly = Polygon(ring)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                poly = poly.simplify(2.0, preserve_topology=True)
                if poly.geom_type != "Polygon":
                    poly = max(poly.geoms, key=lambda g: g.area)
                if poly.area < min_area_m2:
                    continue
                # De-duplicate near-identical closures (area collapsed to ~25 m2 bins).
                key = int(round(poly.area / 25.0))
                if key in seen:
                    continue
                seen.add(key)
                out.append(poly)
    if out:
        _log.info("S3 open-parcel recovery: %d closed candidate(s) near (%.0f, %.0f)",
                  len(out), label_utm[0], label_utm[1])
    return out


def recover_parent_yellow(
    tiles: dict[tuple[int, int], Path],
    grid: TileGrid,
    label_utm: tuple[float, float],
    crop_px: int = 900,
    close_kernels: tuple[int, ...] = (3, 5, 7, 9, 11, 15, 21),
    min_area_m2: float = 2000.0,
) -> list[Polygon]:
    """Recover a LARGE elongated PARENT survey that the S3 tiles draw WHOLE in yellow but
    TNGIS draws only as sub-cells (e.g. INGUR 698 -- a 491x184 m two-lobed survey).

    Differs from ``recover_open_parcel`` on purpose: PURE yellow (no wall-dilate, no
    orange-road sealing -- both OVER-MERGE a long parent into a wrong shape), only LOW
    close-kernels (5/7/9/11 -- high kernels bridge into neighbours), a WIDE crop (so the
    ~491 m parent does not touch the crop border and get mis-flagged as leaking), and a
    16-compass seed probe (the survey-number label often sits in one sub-cell and leaks;
    the parent floods from an offset seed). Returns closed UTM polygons, largest first.
    The rigid area_ratio+scale+rot + seat-locality gate remains the sole ACCEPT arbiter,
    so emitting these candidates only ADDS recall, never a false ACCEPT.
    """
    if not tiles:
        return []
    txs = [k[0] for k in tiles]
    tys = [k[1] for k in tiles]
    tx0, ty0 = min(txs), min(tys)
    wt = (max(txs) - tx0 + 1) * TILE
    ht = (max(tys) - ty0 + 1) * TILE
    gx, gy = grid.utm_to_px(label_utm[0], label_utm[1])
    cx = int(round(gx - tx0 * TILE))
    cy = int(round(gy - ty0 * TILE))
    x0, y0 = max(0, cx - crop_px), max(0, cy - crop_px)
    x1, y1 = min(wt, cx + crop_px), min(ht, cy + crop_px)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return []
    canvas = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
    for (tx, ty), path in tiles.items():
        gx0, gy0 = (tx - tx0) * TILE, (ty - ty0) * TILE
        if gx0 + TILE < x0 or gx0 > x1 or gy0 + TILE < y0 or gy0 > y1:
            continue
        im = cv2.imread(str(path))
        if im is None:
            continue
        sx0, sy0 = max(x0, gx0), max(y0, gy0)
        sx1, sy1 = min(x1, gx0 + TILE), min(y1, gy0 + TILE)
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
            im[sy0 - gy0:sy1 - gy0, sx0 - gx0:sx1 - gx0]
    ox, oy = x0 + tx0 * TILE, y0 + ty0 * TILE
    lx, ly = cx - x0, cy - y0
    h, w = canvas.shape[:2]
    yellow = cv2.inRange(cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV), *_YELLOW)

    m2px = _approx_m2_per_px(grid, tx0, ty0)
    out: list[Polygon] = []
    seen: set[int] = set()
    for ck in close_kernels:
        walls = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, np.ones((ck, ck), np.uint8))
        free = cv2.bitwise_not(walls)
        # Every BOUNDED face of the yellow net is a whole parcel (in S3 the subdivisions are
        # MAGENTA, so a yellow-bounded free region is a full survey, not a sub-cell). A face
        # touching the crop border is still OPEN at this kernel -> skipped; at the kernel that
        # bridges a parcel's boundary gap its full ring closes and the parcel face appears.
        # Taking ALL closed faces (not one label-seeded basin) removes seed sensitivity and is
        # generic -- the rigid + seat-locality gate then selects the face the FMB actually fits.
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
        for i in range(1, n):
            x, y, ww, hh, area = (stats[i, 0], stats[i, 1], stats[i, 2],
                                  stats[i, 3], stats[i, 4])
            if x <= 0 or y <= 0 or x + ww >= w or y + hh >= h:
                continue                                    # touches border -> still open
            if area * m2px < min_area_m2:
                continue
            cnts, _ = cv2.findContours((lbl == i).astype(np.uint8) * 255,
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c0 = max(cnts, key=cv2.contourArea)
            # relative epsilon (1% of perimeter) -> a clean straight-edged ring whose corners
            # the rigid fit can align; a fixed epsilon over-vertices a large parent and inflates
            # the corner residual.
            c = cv2.approxPolyDP(c0, 0.01 * cv2.arcLength(c0, True), True).reshape(-1, 2)
            if len(c) < 4:
                continue
            poly = Polygon([grid.px_to_utm(p[0] + ox, p[1] + oy) for p in c])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.geom_type != "Polygon" or poly.area < min_area_m2:
                continue
            key = int(round(poly.area / 25.0))
            if key in seen:
                continue
            seen.add(key)
            out.append(poly)
    out.sort(key=lambda p: p.area, reverse=True)
    return out


def _approx_m2_per_px(grid: TileGrid, tx0: int, ty0: int) -> float:
    """Ground area per pixel near the tile origin (for px-area filtering)."""
    a = grid.px_to_utm(tx0 * TILE, ty0 * TILE)
    bx = grid.px_to_utm(tx0 * TILE + 1, ty0 * TILE)
    by = grid.px_to_utm(tx0 * TILE, ty0 * TILE + 1)
    dx = math.dist(a, bx)
    dy = math.dist(a, by)
    return max(dx * dy, 1e-6)


def _default_engine():
    """PaddleOCR engine, on GPU when available.

    Paddle is CPU-only on this Blackwell GPU (SM_120), so the GPU route is the
    project's server-detector-via-ONNX-Runtime-CUDA path. Use server_det (which has
    an ONNX model -> CUDAExecutionProvider) + mobile_rec (its ONNX also runs on
    CUDA). Falls back to mobile-det CPU only if ONNX-GPU is unavailable.
    """
    from ..m1_extract.ocr import (
        DEFAULT_DET_MODEL,
        DEFAULT_REC_MODEL,
        SERVER_DET_MODEL,
        _build_engine,
        _onnx_gpu_ready,
    )
    if _onnx_gpu_ready():
        _log.info("S3 OCR: using server_det on GPU (ONNX Runtime CUDA)")
        return _build_engine(SERVER_DET_MODEL, DEFAULT_REC_MODEL, False)
    _log.warning("S3 OCR: ONNX-GPU unavailable, falling back to mobile-det CPU")
    return _build_engine(DEFAULT_DET_MODEL, DEFAULT_REC_MODEL, False)


class S3CadastralSource(CadastralSource):
    """Cadastral parcels located from the public S3 cadastral tiles.

    Builds, for each KNOWN survey number present in the tiles, a CadastralParcel
    whose polygon is the vectorised parcel enclosing its OCR'd label (falling back
    to a small box around the label point if no clean polygon encloses it). All in
    the surveyor UTM frame, so M2 can place each FMB onto its real footprint.
    """

    def __init__(self, bbox_utm, known_surveys: set[str], cache_dir: str | Path,
                 crs: str = "EPSG:32643", engine=None, buffer_m: float = 150.0,
                 use_label_cache: bool = True, village_fence=None):
        import json
        self.crs = crs
        self.grid = TileGrid(crs)
        tiles = download_tiles(bbox_utm, self.grid, cache_dir, buffer_m)
        # Cache the (slow) OCR label-locate keyed by the known-survey set + fence.
        fkey = "fenced" if village_fence is not None else "open"
        # v3 = adds the aggressive stitched/warm-core pass; older caches are ignored.
        cache_f = Path(cache_dir) / f"labels_{fkey}_v3.json"
        self._labels = {}
        if use_label_cache and cache_f.exists():
            try:
                cached = json.loads(cache_f.read_text())
                if set(cached.get("known", [])) == set(known_surveys):
                    self._labels = {k: tuple(v) for k, v in cached["labels"].items()}
                    _log.info("S3 OCR: loaded %d cached labels", len(self._labels))
            except Exception:  # noqa: BLE001
                self._labels = {}
        if not self._labels:
            best, found = ocr_locate_labels(tiles, self.grid, known_surveys, engine)
            # BUG #1: keep only readings inside the village fence (kills cross-village
            # duplicate-survey-number placements like KANDAMPALAYAM's survey 9).
            self._labels = _select_in_fence(found, village_fence)
            if village_fence is not None:
                _log.warning("S3 village fence: kept %d/%d surveys inside the village",
                             len(self._labels), len(best))
            # BUG #6: second-pass OCR for any still-missing known surveys.
            missing = set(known_surveys) - set(self._labels)
            if missing:
                self._labels = ocr_second_pass(
                    tiles, self.grid, missing, self._labels, engine,
                    village_fence=village_fence)
            # Third pass: MULTI-ANGLE OCR for whatever the upright passes still miss
            # (rotated parcel labels). Recovered 6 INGUR labels; validated accurate.
            missing = set(known_surveys) - set(self._labels)
            if missing:
                self._labels = ocr_multi_angle_pass(
                    tiles, self.grid, missing, self._labels, engine,
                    village_fence=village_fence)
            # Fourth pass: AGGRESSIVE (stitched 3x3 + warm-core colour) for the few
            # the multi-angle pass still misses -- edge-straddling or non-orange-fill
            # labels (recovered INGUR 1019/1022/723).
            missing = set(known_surveys) - set(self._labels)
            if missing:
                self._labels = ocr_aggressive_pass(
                    tiles, self.grid, missing, self._labels, engine,
                    village_fence=village_fence)
            # B1 diagnostics: surface WHY any known survey was never located.
            still_missing = set(known_surveys) - set(self._labels)
            if still_missing:
                _log.warning("S3: %d survey(s) NOT located after both OCR passes: %s",
                             len(still_missing),
                             sorted(still_missing, key=lambda s: (len(s), s)))
                if tiles:
                    txs = [k[0] for k in tiles]
                    tys = [k[1] for k in tiles]
                    _log.info("S3: tile coverage x=[%d,%d] y=[%d,%d] (%d tiles) -- a "
                              "still-missing survey is either outside this tile bbox, "
                              "illegible at z18, or fenced out", min(txs), max(txs),
                              min(tys), max(tys), len(tiles))
            try:
                cache_f.write_text(json.dumps(
                    {"known": sorted(known_surveys),
                     "labels": {k: list(v) for k, v in self._labels.items()}}))
            except Exception:  # noqa: BLE001
                pass
        # cv2 face-extraction vectoriser (dilated walls correctly separate parcels;
        # the skimage-skeleton addon was tested and yielded 0 usable parcels here --
        # 1px skeletons don't separate the free-space regions).
        polys = vectorize_parcels(tiles, self.grid)
        # A3: a merged super-parcel (boundary lines had raster gaps -> adjacent
        # parcels fused into one huge polygon) gives a wrong, oversized footprint
        # and a 5-7x area_ratio. Reject any enclosing polygon larger than 3x the
        # median parcel size; fall back to the label point (rigid fit then has no
        # parcel ring -> anchor placement -> REVIEW, never a false ACCEPT).
        if polys:
            areas = sorted(p.area for p in polys)
            median_area = areas[len(areas) // 2]
            max_single_area = median_area * 3.0
            _log.info("S3 parcel areas: median=%.0f m2, merged-reject above %.0f m2 "
                      "(%d raw parcels)", median_area, max_single_area, len(polys))
        else:
            max_single_area = 200000.0
        self._by_survey: dict[str, CadastralParcel] = {}
        # For OPEN/merged labels we may recover several closed candidate rings; the
        # pipeline tries each against its rigid gate and keeps the first that passes
        # (so a recovered ring can never become a false ACCEPT the gate would reject).
        self._candidates: dict[str, list[CadastralParcel]] = {}
        n_merged = 0
        n_recovered = 0
        for sn, (ux, uy) in self._labels.items():
            pt = Point(ux, uy)
            enclosing = [p for p in polys
                         if p.contains(pt) and p.area <= max_single_area]
            if enclosing:
                poly = min(enclosing, key=lambda p: p.area)
            else:
                # No clean right-sized ring (merged/missing). Try a LOCAL, label-seeded
                # recovery of the parcel's own yellow ring (bridges its raster gaps
                # WITHOUT touching the global segmentation of the clean parcels). The
                # pipeline gate decides ACCEPT vs REVIEW per candidate.
                # LOW close-kernels (5/7/9/11) are added because a large SUBDIVIDED PARENT
                # (e.g. INGUR 698, an elongated 491x184 m two-lobed survey drawn whole in S3
                # yellow but only as sub-cells in TNGIS) closes cleanly at a low kernel; the
                # higher kernels over-merge it. The upper area cap is intentionally NOT applied
                # here: the rigid area_ratio+scale+seat gate downstream already rejects an
                # over-merged super-parcel, so capping only DISCARDS legitimate large parents
                # (it had dropped 698's ~52900 m^2 parent -> a 25 m fallback box -> wrong REVIEW).
                # crop_px=640 (~750 m crop) so a LARGE elongated parent (698 is ~491 m wide)
                # does not touch the crop border and get mis-flagged as leaking.
                cands = [c for c in recover_open_parcel(
                             tiles, self.grid, (ux, uy),
                             crop_px=640, close_kernels=(5, 7, 9, 11, 15, 21, 27))
                         if c.area >= 150.0]
                # Also try the dedicated LARGE-elongated-PARENT recovery (pure-yellow, low-k,
                # wide crop). It recovers a big two-lobed survey (e.g. 698) that the open-parcel
                # recovery over-merges; the gate still decides, so this only adds recall (0-FP).
                cands += [c for c in recover_parent_yellow(tiles, self.grid, (ux, uy))
                          if c.area >= 2000.0]
                if cands:
                    self._candidates[sn] = [
                        CadastralParcel(survey_number=sn, polygon=c, village="INGUR",
                                        source_crs=crs)
                        for c in sorted(cands, key=lambda p: p.area, reverse=True)]
                    poly = max(cands, key=lambda p: p.area)
                    n_recovered += 1
                    _log.info("S3: recovered open parcel %s by local yellow-gap bridge "
                              "(%d candidate ring(s), best area=%.0f m2)",
                              sn, len(cands), poly.area)
                else:
                    # Genuinely open (no closure exists) -> label point only.
                    if any(p.contains(pt) for p in polys):
                        n_merged += 1
                    poly = pt.buffer(25.0).envelope
            self._by_survey[sn] = CadastralParcel(
                survey_number=sn, polygon=poly, village="INGUR", source_crs=crs)
        if n_recovered:
            _log.warning("S3: recovered %d open/merged parcel(s) by local yellow-gap "
                         "bridge (gate decides ACCEPT vs REVIEW)", n_recovered)
        if n_merged:
            _log.warning("S3: %d survey(s) fell in a merged super-parcel -> using "
                         "label point only (REVIEW, not ACCEPT)", n_merged)
        _log.info("S3CadastralSource: %d parcels (%d with vector polygon)",
                  len(self._by_survey),
                  sum(1 for s in self._by_survey if any(
                      p.contains(Point(*self._labels[s])) for p in polys)))

    def get(self, survey_number: str, village: str | None = None) -> CadastralParcel | None:
        return self._by_survey.get(_norm_survey(survey_number) or survey_number)

    def recovered_candidates(self, survey_number: str) -> list[CadastralParcel]:
        """Closed candidate rings recovered for an OPEN/merged parcel (largest first).

        Empty for a clean parcel (it already has one right-sized ring via ``get``). The
        pipeline tries these against its rigid gate and keeps the first that passes, so
        none can become a false ACCEPT the gate would reject.
        """
        return self._candidates.get(_norm_survey(survey_number) or survey_number, [])

    def label_point(self, survey_number: str) -> tuple[float, float] | None:
        return self._labels.get(_norm_survey(survey_number) or survey_number)

    def survey_numbers(self) -> set[str]:
        return set(self._by_survey)
