"""TNGIS cadastral TILE source -- block-proof, offline cadastral for M2.

Settled solution: harvest the TNGIS cadastral XYZ tiles (NOT rate-limited -> no IP block),
cache locally (offline + survives a dead API), and deterministically extract georeferenced
parcels. These tests pin the STABLE, deterministic contract (tile harvest + caching, magenta-ink
detection, boundary/label component split, source interface). The parcel-face polygon refinement
and label-OCR thresholds are tuned against real span tiles (with ground-truth survey numbers to
verify 0-FP) and are intentionally not asserted here.
"""
from __future__ import annotations

import numpy as np
import pytest

from landintel.pipeline.m5_cadastral import tngis_tiles as T


def _png_bytes(magenta: bool = True) -> bytes:
    import cv2
    img = np.full((256, 256, 3), 255, np.uint8)
    if magenta:
        img[120:130, :] = (200, 0, 200)        # a horizontal magenta line (BGR)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_pink_mask_detects_magenta():
    import cv2
    img = np.full((20, 20, 3), 255, np.uint8)
    img[:, 5:15] = (200, 0, 200)               # magenta band
    m = T.pink_mask(img)
    assert (m > 0).sum() > 0
    assert (T.pink_mask(np.full((20, 20, 3), 255, np.uint8)) > 0).sum() == 0  # white -> none


def test_label_and_boundary_masks_are_separable():
    """The KEY fix: TNGIS labels are RED/ORANGE, boundaries MAGENTA -- they must not cross-detect,
    or the label OCR ends up reading the boundary lines (the original bug)."""
    red = np.full((20, 20, 3), 255, np.uint8); red[:, :] = (0, 60, 220)      # BGR red/orange
    magenta = np.full((20, 20, 3), 255, np.uint8); magenta[:, :] = (200, 0, 200)
    assert (T.label_mask(red) > 0).sum() > 0        # labels detected by label_mask
    assert (T.label_mask(magenta) > 0).sum() == 0   # boundaries NOT in the label mask
    assert (T.pink_mask(magenta) > 0).sum() > 0     # boundaries detected by pink_mask
    assert (T.pink_mask(red) > 0).sum() == 0        # labels NOT in the boundary mask


def test_boundary_only_keeps_lines_drops_label_blobs():
    import cv2
    ink = np.zeros((256, 256), np.uint8)
    ink[100:103, 10:200] = 255                 # long boundary line (len 190)
    ink[50:60, 50:62] = 255                    # small label blob (12x10)
    out = T._boundary_only(ink, min_len_px=40)
    assert out[101, 100] == 255                # line kept
    assert out[55, 56] == 0                    # label blob dropped


def test_download_caches_and_skips_network_on_hit(tmp_path, monkeypatch):
    from landintel.pipeline.m5_cadastral.s3_tiles import TileGrid
    calls = {"n": 0}

    class _Resp:
        def __init__(self, data): self._d = data
        def read(self): return self._d
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=0):
        calls["n"] += 1
        return _Resp(_png_bytes())

    monkeypatch.setattr(T.urllib.request, "urlopen", fake_urlopen)
    grid = TileGrid(crs="EPSG:32643", z=18)
    bbox = (719600.0, 1224500.0, 719700.0, 1224600.0)   # tiny INGUR-area bbox (UTM 43N)

    tiles1 = T.download_tngis_tiles(bbox, grid, tmp_path, buffer_m=0.0)
    assert tiles1 and all(p.exists() for p in tiles1.values())
    first_calls = calls["n"]
    assert first_calls > 0

    tiles2 = T.download_tngis_tiles(bbox, grid, tmp_path, buffer_m=0.0)  # all cached now
    assert set(tiles2) == set(tiles1)
    assert calls["n"] == first_calls            # cache hit -> NO new network calls


def test_blank_response_not_cached(tmp_path, monkeypatch):
    from landintel.pipeline.m5_cadastral.s3_tiles import TileGrid

    class _Resp:
        def read(self): return b"x" * 50         # too small / not a PNG
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(T.urllib.request, "urlopen", lambda req, timeout=0: _Resp())
    grid = TileGrid(crs="EPSG:32643", z=18)
    tiles = T.download_tngis_tiles((719600.0, 1224500.0, 719700.0, 1224600.0),
                                   grid, tmp_path, buffer_m=0.0)
    assert tiles == {}                          # nothing cached from blank responses


def test_source_get_by_survey_number(tmp_path, monkeypatch):
    """End-to-end source contract with the network + OCR + green-net reconstruction mocked:
    each label -> its reconstructed survey parcel; get()/label_point()/recovered_candidates."""
    from shapely.geometry import Polygon

    parcel = Polygon([(719600, 1224500), (719660, 1224500),
                      (719660, 1224560), (719600, 1224560)])
    subcell = Polygon([(719600, 1224500), (719630, 1224500),
                       (719630, 1224530), (719600, 1224530)])   # a smaller candidate
    label_xy = (719630.0, 1224530.0)

    monkeypatch.setattr(T, "download_tngis_tiles", lambda *a, **k: {(0, 0): tmp_path / "t.png"})
    monkeypatch.setattr(T, "_stitch_color_mosaic", lambda *a, **k: ("mosaic", 0, 0))
    # reconstruct returns candidate rings largest-first (primary parcel, then sub-cell)
    monkeypatch.setattr(T, "reconstruct_parcel", lambda *a, **k: [parcel, subcell])
    monkeypatch.setattr(T, "ocr_labels", lambda *a, **k: {"82": label_xy})

    src = T.TngisTileCadastralSource(["82/1", "99"], (719600, 1224500, 719660, 1224560),
                                     crs="EPSG:32643", cache_dir=tmp_path)
    assert src.survey_numbers() == {"82"}
    p = src.get("82/1")                          # subdivision normalised to base "82"
    assert p is not None and abs(p.polygon.area - parcel.area) < 1.0   # primary = largest
    cands = src.recovered_candidates("82")
    assert len(cands) == 1 and abs(cands[0].polygon.area - subcell.area) < 1.0
    assert src.label_point("82") == label_xy
    assert src.get("123") is None                # unknown -> None, no crash


def test_parcel_boundary_mask_decodes_boundary_not_subdivision():
    """The KEY fix: in a TNGIS tile the parcel-BOUNDARY ink and the SUBDIVISION ink are
    different colours. parcel_boundary_mask must fire on the boundary ink (hue ~30-40) and
    NOT on the subdivision/pink ink (or it re-decodes the wrong lines, the original bug).
    (This decodes the raster only; our own boundary layer stays RED.)"""
    import cv2
    import numpy as np
    boundary_ink = cv2.cvtColor(
        np.full((20, 20, 3), (33, 200, 200), np.uint8), cv2.COLOR_HSV2BGR)  # tile boundary
    subdivision_ink = np.full((20, 20, 3), 255, np.uint8)
    subdivision_ink[:, :] = (200, 0, 200)                                   # tile pink
    assert (T.parcel_boundary_mask(boundary_ink) > 0).sum() > 0
    assert (T.parcel_boundary_mask(subdivision_ink) > 0).sum() == 0
    assert (T.pink_mask(subdivision_ink) > 0).sum() > 0                     # pink = subdivision


def test_select_in_fence_drops_cross_village_readings():
    """village_fence keeps the best reading INSIDE the extent and DROPS a survey whose
    only readings are outside (a same-numbered parcel in an adjacent village) -- the
    cross-village 0-FP hardening for the wider harvest."""
    from shapely.geometry import box

    # survey 82: one reading inside the fence (conf 0.7) + one far outside (conf 0.99);
    # survey 99: only an out-of-fence reading -> must be dropped entirely.
    found = {
        "82": [(719630.0, 1224530.0, 0.7), (700050.0, 1200050.0, 0.99)],
        "99": [(700060.0, 1200060.0, 0.95)],
    }
    fence = box(719500, 1224400, 719800, 1224700)

    out = T._select_in_fence(found, fence)
    assert set(out) == {"82"}                       # 99 dropped (no in-fence reading)
    assert out["82"] == (719630.0, 1224530.0)       # in-fence kept despite lower conf

    # No fence -> global best per survey (back-compat).
    nofence = T._select_in_fence(found, None)
    assert nofence["82"] == (700050.0, 1200050.0) and nofence["99"] == (700060.0, 1200060.0)
