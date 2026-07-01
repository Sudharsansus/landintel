"""M3 assemble -- area computation + non-destructive parcel annotation."""
from __future__ import annotations

import ezdxf

from landintel.pipeline.m3_assemble import (ANNOTATION_LAYER, annotate_combined,
                                           plot_area_hectares, plot_area_m2)


def _make_plot_dxf(path, side=100.0):
    from landintel.core.enums import LayerType
    lyr = LayerType.BOUNDARY.value
    doc = ezdxf.new()
    doc.layers.add(lyr, color=7)
    msp = doc.modelspace()
    # explicit closing vertex: _footprint_polygon builds segments from consecutive
    # vertices and does NOT honour the LWPOLYLINE close flag.
    msp.add_lwpolyline([(0, 0), (side, 0), (side, side), (0, side), (0, 0)],
                       dxfattribs={"layer": lyr})
    doc.saveas(str(path))


def test_plot_area_from_boundary(tmp_path):
    p = tmp_path / "plot.dxf"
    _make_plot_dxf(p, 100.0)                                # 100 x 100 m = 1 ha
    assert abs(plot_area_m2(p) - 10000.0) < 1.0
    assert abs(plot_area_hectares(p) - 1.0) < 1e-4


def test_plot_area_none_without_boundary(tmp_path):
    p = tmp_path / "empty.dxf"
    ezdxf.new().saveas(str(p))
    assert plot_area_m2(p) is None                          # no closed boundary -> None, no crash


def test_annotate_combined_adds_nondestructive_labels(tmp_path):
    plot = tmp_path / "724.dxf"
    _make_plot_dxf(plot, 100.0)
    combined = tmp_path / "combined.dxf"
    ezdxf.new().saveas(str(combined))

    n = annotate_combined(combined, [(str(plot), "724")])
    assert n == 1
    doc = ezdxf.readfile(str(combined))
    labels = [e for e in doc.modelspace()
              if e.dxftype() == "TEXT" and e.dxf.layer == ANNOTATION_LAYER]
    assert len(labels) == 1
    assert "724" in labels[0].dxf.text and "ha" in labels[0].dxf.text
