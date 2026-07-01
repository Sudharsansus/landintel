"""Builders for the new-M2 (m2_club) tests: synthetic FMB DXFs + a mock cadastre.

No surveyor file is involved (that is M3). We build real M1 DXFs via ``to_dxf`` and
a cadastral source that maps survey# -> a UTM polygon, then assert the FMB-only
georeference + club behaves with 0 false positives.
"""
from __future__ import annotations

import re

import numpy as np
from shapely.geometry import Polygon

from landintel.core.models import Boundary, CornerPoint, NeighborLabel, Plot
from landintel.pipeline.m1_extract.to_dxf import write_dxf
from landintel.pipeline.m5_cadastral.source import CadastralParcel

# A patch of UTM 43N near INGUR/Erode -- realistic for the verify ranges.
BASE_X = 719600.0
BASE_Y = 1224500.0


def build_fmb(path, survey_no, corners, neighbors=None):
    """corners: [(label, x, y)] relative metres. neighbors: [(text, (x, y))]."""
    verts = [(float(x), float(y)) for _l, x, y in corners]
    plot = Plot(
        client_id="c", survey_no=str(survey_no), district="D", taluk="T",
        village="INGUR", scale=1000, stated_area=0.01,
        boundary=Boundary(points=verts + [verts[0]]),
        corner_points=[CornerPoint(label=str(l), x=float(x), y=float(y))
                       for l, x, y in corners],
        neighbor_labels=[NeighborLabel(label=str(t), position=(float(px), float(py)))
                         for t, (px, py) in (neighbors or [])],
    )
    return write_dxf(plot, path)


def utm_polygon(corners, angle_deg=0.0, tx=BASE_X, ty=BASE_Y, scale=1.0):
    """The UTM parcel = the relative corners rigidly placed (rot+scale+translate)."""
    th = np.radians(angle_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = [scale * (R @ np.array([float(x), float(y)])) + np.array([tx, ty])
           for _l, x, y in corners]
    return Polygon([(float(p[0]), float(p[1])) for p in pts])


class MockCadastral:
    """survey# -> UTM polygon, with the CadastralSource interface m2_club uses."""

    def __init__(self, parcels):
        self._p = dict(parcels)

    @staticmethod
    def _base(sn):
        m = re.search(r"\d+", str(sn))
        return m.group(0) if m else str(sn)

    def get(self, sn, village=None):
        poly = self._p.get(self._base(sn))
        return CadastralParcel(survey_number=self._base(sn), polygon=poly) \
            if poly is not None else None

    def label_point(self, sn):
        p = self.get(sn)
        if p is None:
            return None
        c = p.polygon.centroid
        return (float(c.x), float(c.y))

    def recovered_candidates(self, sn):
        return []

    def survey_numbers(self):
        return set(self._p)


# A standard 50x30 rectangle (unique principal axis -> stable orientation gate).
RECT = [("A", 0.0, 0.0), ("B", 50.0, 0.0), ("C", 50.0, 30.0), ("D", 0.0, 30.0)]
