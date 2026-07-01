"""TNGIS cadastral TILE source -- block-proof, offline, authoritative.

THE solution for M2's cadastral reference, settled after the API/offline/block analysis:

  * The TNGIS *parcel-lookup* POST (`rate_limit_land_details`) is IP rate-limited -> bulk use
    risks an IP block. DO NOT use it at volume.
  * The TNGIS *cadastral XYZ tiles* (`/data/xyz_tiles/cadastral_xyz/{z}/{x}/{y}.png`) are a
    raster basemap with NO rate limit (only a Referer header) -> they can be bulk-downloaded for
    a whole region and CACHED LOCALLY. After the first harvest, M2 runs FULLY OFFLINE from the
    cache and survives a dead/blocked API.
  * Each z18 tile carries BOTH the parcel boundary lines AND the survey/sub-division-number
    labels inside each parcel (verified on Vilankurichy/INGUR) -> boundary geometry + identity,
    georeferenced (slippy-map: every pixel has a known WGS84 position).

So this module: harvest tiles for a bbox -> cache -> deterministically VECTORISE the boundaries
into parcel polygons and OCR the survey-number labels (closed-set, fuzzy) -> survey-numbered
parcels in UTM, exposed as a ``CadastralSource`` for M2. 0-FP: code (not the LLM) produces every
coordinate; OCR labels are matched ONLY against the known survey set, so a misread can never
invent a survey.

NOTE: the label-OCR HSV/threshold tuning is finalised against real tiles WITH ground-truth
survey numbers (so we can verify 0 wrong labels). The tile harvest + caching + georeferencing +
boundary vectorisation below are deterministic and validated without ground truth.
"""

from __future__ import annotations

import logging
import math
import urllib.request
from pathlib import Path

import numpy as np

from .s3_tiles import TILE, TileGrid, _fuzzy_survey_match
from .source import CadastralParcel, CadastralSource, _norm_survey

_log = logging.getLogger(__name__)

# TNGIS cadastral basemap tiles. Max zoom 18 (z19 -> 404). Referer required; a browser UA
# avoids the 403 python-requests path. NOT rate limited (unlike the parcel-lookup POST).
TNGIS_TILE_BASE = "https://tngis.tn.gov.in/data/xyz_tiles/cadastral_xyz"
TNGIS_ZOOM = 18
_TNGIS_HEADERS = {
    "Referer": "https://tngis.tn.gov.in/",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
                  "(KHTML, like Gecko) Version/16.6 Safari/605.1.15",
}

# TNGIS cadastral tiles use TWO distinct ink colours (verified by hue histogram on a real INGUR
# z18 tile): the parcel BOUNDARY lines are MAGENTA (hue ~150-170) and the survey-number LABELS are
# RED/ORANGE (hue 0-30, plus the red wrap 170-180). OpenCV hue is 0-179. Separating them is what
# makes the label OCR work -- OCRing the magenta mask reads the boundary lines, not the numbers.
_BOUND_LO = np.array([142, 60, 60], np.uint8)          # magenta boundary lines
_BOUND_HI = np.array([170, 255, 255], np.uint8)
# red/orange survey-number labels (two hue bands around the red end)
_LABEL_RANGES = [(np.array([0, 70, 70], np.uint8),   np.array([30, 255, 255], np.uint8)),
                 (np.array([171, 70, 70], np.uint8), np.array([180, 255, 255], np.uint8))]


def pink_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask of the magenta BOUNDARY ink (parcel lines)."""
    import cv2
    return cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), _BOUND_LO, _BOUND_HI)


def label_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask of the RED/ORANGE survey-number LABEL ink (the text to OCR)."""
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = np.zeros(bgr.shape[:2], np.uint8)
    for lo, hi in _LABEL_RANGES:
        m = cv2.bitwise_or(m, cv2.inRange(hsv, lo, hi))
    return m


def download_tngis_tiles(
    bbox_utm: tuple[float, float, float, float],
    grid: TileGrid,
    cache_dir: str | Path,
    *,
    zoom: int = TNGIS_ZOOM,
    buffer_m: float = 120.0,
    base_url: str = TNGIS_TILE_BASE,
) -> dict[tuple[int, int], Path]:
    """Download + cache the cadastral tiles covering ``bbox_utm`` (+buffer). Block-proof
    (tiles are not rate limited). Returns {(tx,ty): png_path}. Cache hits never hit the net.

    Only the tiles covering the SPAN's bbox are fetched -- never the whole map, so this is
    'pull the required region', not a whole-state scrape."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = bbox_utm
    # corners -> tile-pixel -> tile indices
    cps = [grid.utm_to_px(x0 - buffer_m, y0 - buffer_m),
           grid.utm_to_px(x1 + buffer_m, y1 + buffer_m)]
    gxs, gys = [c[0] for c in cps], [c[1] for c in cps]
    tx0, tx1 = int(min(gxs) // TILE), int(max(gxs) // TILE)
    ty0, ty1 = int(min(gys) // TILE), int(max(gys) // TILE)
    out: dict[tuple[int, int], Path] = {}
    n_dl = 0
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            p = cache_dir / f"{zoom}_{tx}_{ty}.png"
            if not p.exists():
                try:
                    req = urllib.request.Request(f"{base_url}/{zoom}/{tx}/{ty}.png",
                                                 headers=_TNGIS_HEADERS)
                    data = urllib.request.urlopen(req, timeout=30).read()
                    if len(data) < 500 or data[:8] != b"\x89PNG\r\n\x1a\n":
                        continue                       # blank / non-tile
                    p.write_bytes(data)
                    n_dl += 1
                except Exception:                      # noqa: BLE001 - 404 / network
                    continue
            if p.exists():
                out[(tx, ty)] = p
    _log.info("TNGIS tiles: %d covering bbox (%d newly downloaded) in %s",
              len(out), n_dl, cache_dir)
    return out


def _boundary_only(ink: np.ndarray, min_len_px: int = 40) -> np.ndarray:
    """Keep the pink BOUNDARY lines, drop the small/compact pink LABEL blobs.

    Boundary strokes are long/elongated; survey-number labels are small compact blobs. Filtering
    the connected components by bounding-box max side keeps the parcel net intact and removes the
    in-parcel number text (which would otherwise split a parcel face)."""
    import cv2
    n, _lbl, stats, _c = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros_like(ink)
    for i in range(1, n):
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if max(w, h) >= min_len_px:                    # elongated -> boundary
            keep[_lbl == i] = 255
    return keep


def _stitch_boundary_mosaic(tiles: dict[tuple[int, int], Path]):
    """Place every tile's boundary mask into ONE mosaic at its global pixel offset.

    Returns ``(mosaic, min_tx, min_ty)`` or ``(None, 0, 0)``. Stitching BEFORE vectorising
    is what keeps a parcel that straddles a tile seam whole -- the old per-tile pass clipped
    it at the seam, leaving a 'hole' between plots and stranding the parcel's label outside
    any single-tile face (every INGUR label fell back to nearest-polygon)."""
    import cv2
    if not tiles:
        return None, 0, 0
    txs = [tx for tx, _ in tiles]
    tys = [ty for _, ty in tiles]
    min_tx, max_tx = min(txs), max(txs)
    min_ty, max_ty = min(tys), max(tys)
    w = (max_tx - min_tx + 1) * TILE
    h = (max_ty - min_ty + 1) * TILE
    mosaic = np.zeros((h, w), np.uint8)
    for (tx, ty), path in tiles.items():
        img = cv2.imread(str(path))
        if img is None:
            continue
        b = _boundary_only(pink_mask(img))
        ox, oy = (tx - min_tx) * TILE, (ty - min_ty) * TILE
        mosaic[oy:oy + TILE, ox:ox + TILE] = b
    return mosaic, min_tx, min_ty


def vectorise_parcels(tiles: dict[tuple[int, int], Path], grid: TileGrid,
                      min_area_m2: float = 25.0):
    """Extract parcel FACES (the regions enclosed by the pink boundary net) into UTM polygons.

    Deterministic, no OCR: STITCH every tile's boundary mask into one mosaic (so parcels are
    not clipped at tile seams), dilate to close anti-alias gaps, then take the WHITE regions
    between the lines as parcels (``RETR_CCOMP`` interior contours). Each pixel ring -> UTM via
    the slippy-map grid. Parcels below ``min_area_m2`` (label holes, anti-alias specks) dropped."""
    import cv2
    from shapely.geometry import Polygon

    mosaic, min_tx, min_ty = _stitch_boundary_mosaic(tiles)
    if mosaic is None:
        return []
    mosaic = cv2.dilate(mosaic, np.ones((3, 3), np.uint8), iterations=1)
    faces = cv2.bitwise_not(mosaic)                    # white regions = parcels
    contours, hierarchy = cv2.findContours(faces, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    parcels = []
    for c, h in zip(contours, hierarchy[0]):
        if h[3] == -1:                                 # skip the outer canvas region
            continue
        pts = c.reshape(-1, 2)
        if len(pts) < 4:
            continue
        ring = [grid.px_to_utm(px + min_tx * TILE, py + min_ty * TILE) for px, py in pts]
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if isinstance(poly, Polygon) and poly.is_valid and poly.area >= min_area_m2:
            parcels.append(poly)
    return parcels


# --- Decoding the TNGIS tile pixels (our convention is unchanged) ------------------
# OUR REFERENCE CONVENTION (the client's, used everywhere in our DXFs) is unchanged:
#   PARCEL BOUNDARY = RED,  SUBDIVISION = GREEN.
# The constants below are NOT a colour convention -- they ONLY decode the TNGIS *raster*
# tile, which renders those same two roles in its own ink: the PARCEL BOUNDARY as yellow,
# the SUBDIVISION as pink. So to read which pixels are the boundary in a downloaded PNG we
# match the tile's boundary ink (a tile-decoding detail); the recovered geometry then goes
# onto our normal RED boundary layer. The old code matched the tile's SUBDIVISION ink by
# mistake (enclosed ~2/13 labels); matching the boundary ink + label-seeded flood-fill
# enclosed 13/13 and seated 11-12.
_TILE_BOUNDARY_LO = np.array([20, 60, 80], np.uint8)   # the tile's parcel-boundary ink
_TILE_BOUNDARY_HI = np.array([48, 255, 255], np.uint8)

# Local crop radius (px) around a label, and the morphological-close kernels swept to seal
# anti-alias gaps in the boundary net (small k = a subdivision sub-cell, large k = the whole
# survey parcel; all closed candidates are returned largest-first).
PARCEL_RADIUS_PX = 360
_CLOSE_KERNELS = (3, 5, 7, 9, 11, 13)


def parcel_boundary_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask of the PARCEL-BOUNDARY ink in a TNGIS tile (the subdivision ink is the
    separate pink_mask). Decodes the raster only -- recovered parcels go onto our normal
    RED boundary layer; our RED/GREEN convention is unchanged."""
    import cv2
    return cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), _TILE_BOUNDARY_LO, _TILE_BOUNDARY_HI)


def _stitch_color_mosaic(tiles: dict[tuple[int, int], Path]):
    """Stitch the raw BGR tiles into one mosaic. Returns (bgr, min_tx, min_ty)."""
    import cv2
    if not tiles:
        return None, 0, 0
    txs = [tx for tx, _ in tiles]
    tys = [ty for _, ty in tiles]
    min_tx, max_tx = min(txs), max(txs)
    min_ty, max_ty = min(tys), max(tys)
    w = (max_tx - min_tx + 1) * TILE
    h = (max_ty - min_ty + 1) * TILE
    mosaic = np.full((h, w, 3), 255, np.uint8)
    for (tx, ty), path in tiles.items():
        img = cv2.imread(str(path))
        if img is None:
            continue
        oy, ox = (ty - min_ty) * TILE, (tx - min_tx) * TILE
        mosaic[oy:oy + TILE, ox:ox + TILE] = img
    return mosaic, min_tx, min_ty


# 16-point compass directions (the cardinal/intermediate/sub-direction set), as unit
# (dx, dy) in image pixels (dy down). Used to probe flood-fill seeds OFFSET from the label
# so a label in a parcel corner / sub-cell / next to a leak still seeds the full enclosing face.
_COMPASS_16 = tuple(
    (math.cos(2 * math.pi * i / 16), math.sin(2 * math.pi * i / 16)) for i in range(16)
)
_DIR_RADII_FRAC = (0.30, 0.55, 0.80)   # fractions of the crop radius to step out per direction


def reconstruct_parcel(label_utm, mosaic_bgr, min_tx, min_ty, grid,
                       min_area_m2: float = 50.0, aggressive: bool = False,
                       directional: bool = False):
    """Reconstruct the survey parcel ENCLOSING a label point, from the boundary net.

    Label-seeded flood-fill bounded by the (kernel-swept, gap-sealed) boundary net.
    A basin that LEAKS to the crop border is an OPEN parcel at that kernel -> rejected.
    Returns closed candidate rings (UTM Polygons), de-duped by area, LARGEST-FIRST.
    A wrong/over-grown reconstruction is gated downstream by the rigid shape gate AND the
    seat-locality gate, so this can only ADD recall, never a false ACCEPT.

    ``aggressive`` (default False keeps the validated path byte-identical): a SECOND pass for
    plots whose boundary net LEAKS on the normal pass -- a wider crop + larger gap-seal kernels
    (a metre ladder up to ~12 m, resolution-derived). It recovers a parcel where the ring has a
    road/anti-alias break, but the result is less reliable, so the caller routes it to REVIEW
    (located), NEVER ACCEPT -- strictly additive, 0-FP.

    ``directional``: ALSO seed the flood-fill at points offset from the label in the 16 compass
    directions (N, NE, ..., NNW) at a few radii. A survey-number label often sits in a parcel
    CORNER, inside one SUB-CELL of its parcel, or right next to a ring break -- seeding only at the
    label then floods a sub-region or leaks. Probing the 16 directions finds the seed that floods
    the full enclosing parcel face. Every basin still passes the rigid + seat gate, so the extra
    candidates only ADD recall (recovering missed plots) at 0-FP.
    """
    import cv2
    from shapely.geometry import Polygon

    if mosaic_bgr is None:
        return []
    h, w = mosaic_bgr.shape[:2]
    gx, gy = grid.utm_to_px(label_utm[0], label_utm[1])
    cx = int(round(gx - min_tx * TILE))
    cy = int(round(gy - min_ty * TILE))
    r = PARCEL_RADIUS_PX * (2 if aggressive else 1)
    x0, y0 = max(0, cx - r), max(0, cy - r)
    x1, y1 = min(w, cx + r), min(h, cy + r)
    crop = mosaic_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    gm = parcel_boundary_mask(crop)
    ch, cw = gm.shape
    sx, sy = cx - x0, cy - y0
    if not (0 <= sx < cw and 0 <= sy < ch):
        return []

    if aggressive:
        m_per_px = abs(grid.px_to_utm(1000, 0)[0] - grid.px_to_utm(0, 0)[0]) / 1000.0 or 0.586
        kernels = sorted({max(3, int(round(m / m_per_px)) | 1) for m in (3, 5, 7, 9, 12)})
    else:
        kernels = _CLOSE_KERNELS

    # directional probe seed points (offset from the label in the 16 compass directions)
    probe = []
    if directional:
        for dx, dy in _COMPASS_16:
            for frac in _DIR_RADII_FRAC:
                px, py = int(round(sx + dx * r * frac)), int(round(sy + dy * r * frac))
                if 0 <= px < cw and 0 <= py < ch:
                    probe.append((px, py))

    cands, seen = [], set()
    for k in kernels:
        sealed = cv2.morphologyEx(gm, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        free = cv2.bitwise_not(sealed)
        # label seed (nudged to free space if it lands on a boundary line)
        seeds = []
        if free[sy, sx] == 0:
            ys, xs = np.where(free > 0)
            if len(xs):
                j = int(np.argmin((xs - sx) ** 2 + (ys - sy) ** 2))
                seeds.append((int(xs[j]), int(ys[j])))
        else:
            seeds.append((sx, sy))
        # directional probe seeds (only those landing in free space; no nudge -> cheap)
        for px, py in probe:
            if free[py, px] != 0:
                seeds.append((px, py))
        for seed in seeds:
            mask = np.zeros((ch + 2, cw + 2), np.uint8)
            cv2.floodFill(free.copy(), mask, seed, 255, flags=8)
            basin = mask[1:-1, 1:-1]
            if (basin[0, :].any() or basin[-1, :].any()
                    or basin[:, 0].any() or basin[:, -1].any()):
                continue                            # leaked to border -> open parcel
            cnts, _ = cv2.findContours(basin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True).reshape(-1, 2)
            if len(approx) < 4:
                continue
            ring = [grid.px_to_utm(px + x0 + min_tx * TILE, py + y0 + min_ty * TILE)
                    for px, py in approx]
            poly = Polygon(ring)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not (isinstance(poly, Polygon) and poly.is_valid and poly.area >= min_area_m2):
                continue
            key = round(poly.area / 50.0)
            if key in seen:
                continue
            seen.add(key)
            cands.append(poly)
    cands.sort(key=lambda p: p.area, reverse=True)
    return cands


def _ocr_raw_detections(tiles, grid, known_surveys, engine=None, upscale: int = 12,
                        margin: int = 8) -> dict[str, list[tuple[float, float, float]]]:
    """Raw OCR pass -> {survey_no: [(utm_x, utm_y, conf), ...]} (ALL detections, no fence).

    KEY: the red/orange ``label_mask`` only LOCATES the label blobs; the OCR runs on a crop of the
    ORIGINAL GRAYSCALE tile (which preserves the anti-aliased digit detail) upscaled ~12x -- NOT on
    the colour-masked binary, which merges the tiny z18 digits into an unreadable blob.

    Closed-set + fuzzy: a read is kept ONLY if it fuzzy-matches a KNOWN survey number, so a misread
    can never invent a survey (0-FP on identity). The OCR engine is created LAZILY on the first
    readable tile so a tile-less / mocked call never loads PaddleOCR."""
    import cv2
    found: dict[str, list[tuple[float, float, float]]] = {}
    for (tx, ty), path in tiles.items():
        img = cv2.imread(str(path))
        if img is None:
            continue
        if engine is None:                                 # lazy: only when there is a real tile
            from .s3_tiles import _default_engine
            engine = _default_engine()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lm = label_mask(img)
        # locate candidate label blobs; merge nearby ones (a number can fragment into digits)
        merged = cv2.dilate(lm, np.ones((3, 7), np.uint8), iterations=1)   # join digits L-R
        n, _l, stats, _c = cv2.connectedComponentsWithStats(merged, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                                stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                                stats[i, cv2.CC_STAT_AREA])
            if area < 10 or max(w, h) > 70 or min(w, h) < 3:     # skip noise + long boundary bits
                continue
            x0, y0 = max(0, x - margin), max(0, y - margin)
            x1, y1 = min(img.shape[1], x + w + margin), min(img.shape[0], y + h + margin)
            crop = gray[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            big = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
            big = cv2.threshold(big, 180, 255, cv2.THRESH_BINARY)[1]
            try:
                results = engine.predict(cv2.cvtColor(big, cv2.COLOR_GRAY2BGR))
            except Exception:                              # noqa: BLE001
                continue
            cx, cy = x + w / 2.0, y + h / 2.0
            ux, uy = grid.px_to_utm(cx + tx * TILE, cy + ty * TILE)
            for r in results:
                d = r if isinstance(r, dict) else getattr(r, "json", {}).get("res", r)
                for t, sc in zip(d.get("rec_texts", []),
                                 d.get("rec_scores", [1.0] * len(d.get("rec_texts", [])))):
                    sn = _fuzzy_survey_match(str(t).strip(), known_surveys)
                    if sn is not None:
                        found.setdefault(sn, []).append((ux, uy, float(sc)))
    return found


def ocr_labels(tiles, grid, known_surveys, engine=None, upscale: int = 12, margin: int = 8,
               fence=None, return_all: bool = False):
    """OCR the survey-number labels -> {survey_no: (utm_x, utm_y)} (best in-fence reading).

    ``return_all=True`` -> ``(best, candidates)`` where ``candidates`` is
    ``{survey_no: [(utm_x, utm_y, conf), ...]}`` -- EVERY in-fence detection, confidence-sorted.
    The survey number is often printed at several spots (subdivision labels, repeats); the single
    best-confidence reading is frequently NOT inside the true parcel (measured: surveys 667/723/
    730 had the right label discarded). Keeping all detections lets the reconstruction pick the
    position whose boundary net actually CLOSES -- a leak-free self-selection of the true label."""
    found = _ocr_raw_detections(tiles, grid, known_surveys, engine, upscale, margin)
    best = _select_in_fence(found, fence)
    if return_all:
        return best, _select_all_in_fence(found, fence)
    return best


def _select_in_fence(found: dict, fence):
    """Best (highest-confidence) reading per survey, restricted to the village fence.

    A survey number can repeat in an ADJACENT village. Without a fence -> the global best
    reading per survey. With a fence (the village/corridor extent) -> only readings INSIDE
    it; a survey whose readings are ALL outside (the wrong village's parcel, ~1.7-2.6 km
    away on INGUR) is DROPPED. This is what keeps the wider full-village harvest 0-FP.
    """
    if fence is None:
        return {sn: max(v, key=lambda z: z[2])[:2] for sn, v in found.items()}
    from shapely.geometry import Point
    out = {}
    for sn, v in found.items():
        inside = [z for z in v if fence.contains(Point(z[0], z[1]))]
        if inside:
            out[sn] = max(inside, key=lambda z: z[2])[:2]
    return out


def _select_all_in_fence(found: dict, fence) -> dict[str, list[tuple[float, float, float]]]:
    """ALL in-fence detections per survey, highest-confidence first (best == element 0,
    matching ``_select_in_fence``). With no fence, every detection is kept."""
    from shapely.geometry import Point
    out: dict[str, list[tuple[float, float, float]]] = {}
    for sn, v in found.items():
        dets = v if fence is None else [z for z in v if fence.contains(Point(z[0], z[1]))]
        if dets:
            out[sn] = sorted(dets, key=lambda z: z[2], reverse=True)
    return out


class TngisTileCadastralSource(CadastralSource):
    """Survey-numbered cadastral parcels harvested from the TNGIS cadastral tiles, cached
    locally. Block-proof (tiles not rate limited), offline after harvest, 0-FP (code derives
    every coordinate; OCR labels matched only against the known survey set).

    Pass the span's ``survey_numbers`` (from the M1 plots) so the label OCR is a CLOSED set and
    only the span's bbox tiles are fetched. ``bbox_utm`` bounds the harvest; ``crs`` is the
    surveyor UTM zone (43N/44N)."""

    def __init__(self, survey_numbers, bbox_utm, *, crs: str = "EPSG:32643",
                 cache_dir: str | Path = ".tngis_tiles", engine=None,
                 ocr: bool = True, village_fence=None):
        self.crs = crs
        grid = TileGrid(crs=crs, z=TNGIS_ZOOM)
        known = {_norm_survey(s) or str(s) for s in survey_numbers}
        known.discard(None)
        tiles = download_tngis_tiles(bbox_utm, grid, cache_dir)
        if ocr and known:
            res = ocr_labels(tiles, grid, known, engine=engine, fence=village_fence,
                             return_all=True)
            labels, cand_map = res if isinstance(res, tuple) else (res, {})
        else:
            labels, cand_map = {}, {}

        # Candidate label POSITIONS per survey, best-confidence first. The same survey number is
        # often printed at several spots (subdivision labels, repeats), and the single best-
        # confidence reading is frequently NOT inside the true parcel (measured: surveys 667/723/
        # 730 had their right label discarded by ~100 m). We try the best label first -- so every
        # parcel that already closed is unchanged -- and fall back to the alternates ONLY when the
        # best label's net leaks. The alternate whose net actually CLOSES self-selects the label
        # inside the true parcel (a label on a road/edge leaks; one inside the parcel floods clean).
        positions: dict[str, list[tuple[float, float]]] = {}
        for sn in set(labels) | set(cand_map):
            pos = [(x, y) for (x, y, _c) in cand_map.get(sn, [])]
            if sn in labels and labels[sn] not in pos:
                pos.insert(0, labels[sn])         # guarantee the back-compat best is tried first
            positions[sn] = pos

        # Each label -> the survey PARCEL enclosing it (label-seeded flood-fill). Primary = largest
        # sealed ring; the rest are recovered_candidates. The rigid+seat-locality gate downstream is
        # the sole arbiter, so trying multiple label positions only ADDS recall, never a false ACCEPT.
        mosaic, min_tx, min_ty = _stitch_color_mosaic(tiles)
        self._by_survey: dict[str, CadastralParcel] = {}
        self._candidates: dict[str, list[CadastralParcel]] = {}
        self._aggressive: set[str] = set()       # parcels recovered by the wider 2nd pass
        self._labels: dict[str, tuple[float, float]] = {}
        n_green = n_aggr = n_alt = n_dir = 0
        for sn, poslist in positions.items():
            chosen = poslist[0] if poslist else None
            rings = []
            if mosaic is not None and chosen is not None:
                rings = reconstruct_parcel(chosen, mosaic, min_tx, min_ty, grid)
            # best label's net leaked -> try the OTHER in-fence detections; first that closes wins
            if not rings and mosaic is not None:
                for alt in poslist[1:]:
                    r = reconstruct_parcel(alt, mosaic, min_tx, min_ty, grid)
                    if r:
                        rings, chosen = r, alt
                        n_alt += 1
                        break
            # still nothing -> AGGRESSIVE (wider crop + larger kernels) at each detection. Flagged
            # so cadastral_seat routes it to REVIEW, never ACCEPT (strictly additive recall, 0-FP).
            if not rings and mosaic is not None:
                for alt in poslist:
                    r = reconstruct_parcel(alt, mosaic, min_tx, min_ty, grid, aggressive=True)
                    if r:
                        rings, chosen = r, alt
                        self._aggressive.add(sn)
                        n_aggr += 1
                        break
            if rings:
                n_green += 1
                primary = rings[0]
                cand_rings = list(rings[1:])
                # Enrich with DIRECTIONAL candidates (16-compass seed probe) so a sub-cell or
                # partial primary can be upgraded to the full enclosing parcel by the gate. The
                # primary stays the label-seed ring (existing ACCEPTs unchanged); directional rings
                # are only tried if the primary fails the gate -> additive recall, 0-FP.
                if mosaic is not None and chosen is not None and not self._aggressive.intersection({sn}):
                    dir_rings = reconstruct_parcel(chosen, mosaic, min_tx, min_ty, grid,
                                                   directional=True)
                    seen_a = {round(primary.area / 50.0)} | {round(p.area / 50.0) for p in cand_rings}
                    for p in dir_rings:
                        ka = round(p.area / 50.0)
                        if ka not in seen_a:
                            seen_a.add(ka)
                            cand_rings.append(p)
                    n_dir += 1
                self._by_survey[sn] = CadastralParcel(survey_number=sn, polygon=primary,
                                                      source_crs=crs)
                self._candidates[sn] = [CadastralParcel(survey_number=sn, polygon=r,
                                                        source_crs=crs) for r in cand_rings]
            if chosen is not None:
                self._labels[sn] = chosen        # the label that produced the parcel = seat anchor
        _log.info("TNGIS tile source: %d parcels (%d via alternate label, %d aggressive) / "
                  "%d labels in %s", n_green, n_alt, n_aggr, len(self._labels), crs)

    def is_aggressive(self, survey_number: str) -> bool:
        """True if this survey's parcel was recovered by the wider 2nd pass (less reliable ->
        the caller must route it to REVIEW, never ACCEPT)."""
        return (_norm_survey(survey_number) or survey_number) in self._aggressive

    def get(self, survey_number: str, village: str | None = None) -> CadastralParcel | None:
        return self._by_survey.get(_norm_survey(survey_number) or survey_number)

    def recovered_candidates(self, survey_number: str) -> list[CadastralParcel]:
        """Alternative reconstructed rings (smaller sealed sub-cells) for a survey whose
        primary ring failed the gate. Mirrors VectorFileCadastralSource; the gate is the
        sole arbiter, so a candidate can only ADD recall, never a false ACCEPT."""
        return self._candidates.get(_norm_survey(survey_number) or survey_number, [])

    def label_point(self, survey_number: str) -> tuple[float, float] | None:
        p = self._labels.get(_norm_survey(survey_number) or survey_number)
        return (float(p[0]), float(p[1])) if p else None

    def survey_numbers(self) -> set[str]:
        return set(self._by_survey)
