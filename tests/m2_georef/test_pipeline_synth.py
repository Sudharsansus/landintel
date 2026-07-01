"""End-to-end synthetic test of the full M2 pipeline.

Builds a real M1 DXF (via to_dxf) + a synthetic surveyor DXF related by a known
similarity transform, runs ``georef_pipeline``, and asserts the plot is matched,
transformed into the correct UTM region with sub-metre field residual, and that
all seven verification checks pass. This exercises match -> umeyama ->
cadastral_adjust -> warp -> output_dxf -> verify with no OCR dependency.
"""

from __future__ import annotations

import ezdxf
import numpy as np

from landintel.pipeline.m2_georef import georef_pipeline
from landintel.pipeline.m2_georef.extract_m1 import extract_m1_dxf

from conftest import TRUE_T


def test_pipeline_end_to_end(m1_dxf, surveyor_dxf, tmp_path):
    out_dir = tmp_path / "georef"
    results = georef_pipeline(surveyor_dxf, [m1_dxf], out_dir)

    assert len(results) == 1
    r = results[0]

    # --- Matched ---
    assert r.matched, f"plot did not match: {r.error}"
    assert r.survey_number == "784"
    assert r.fingerprint_score < 1.0
    assert r.neighborhood_score < 1.0

    # --- Sub-metre field residual (exact synthetic transform) ---
    assert r.field_residual_max < 0.5

    # --- Output written + all 7 verification checks pass ---
    assert r.output_file
    assert r.verify_result is not None
    failed = [c.name for c in r.verify_result.checks if not c.passed]
    assert r.verify_result.all_passed, f"failed checks: {failed}"

    # --- Output geometry sits in the surveyor's UTM region ---
    doc = ezdxf.readfile(r.output_file)
    msp = doc.modelspace()
    xs, ys = [], []
    for e in msp.query("LWPOLYLINE"):
        for p in e.get_points():
            xs.append(p[0]); ys.append(p[1])
    assert min(xs) > 600000 and max(xs) < 900000
    assert min(ys) > 1100000 and max(ys) < 1400000
    # Centred near the known translation target.
    assert abs(np.mean(xs) - TRUE_T[0]) < 200
    assert abs(np.mean(ys) - TRUE_T[1]) < 200


def test_pipeline_produces_single_combined_file(m1_dxf, surveyor_dxf, tmp_path):
    """The M2 deliverable is ONE file: every placed FMB clubbed into the raw data
    file. The combined DXF must contain BOTH the surveyor base layers (SITE DATA
    LINE) AND the FMB layers (BOUNDARY, SURVEY_NUMBER), all in the surveyor UTM
    frame."""
    out_dir = tmp_path / "georef"
    results = georef_pipeline(surveyor_dxf, [m1_dxf], out_dir)
    assert results[0].matched

    combined = out_dir / "combined_village.dxf"
    assert combined.exists(), "combined village DXF not written"

    doc = ezdxf.readfile(str(combined))
    layers = {e.dxf.layer for e in doc.modelspace()}
    assert "SITE DATA LINE" in layers   # surveyor raw-data base is present
    # The FMB plot is clubbed in -- on its semantic layers if ACCEPT, or on its own
    # REVIEW_FMB_<survey> layer if REVIEW (the 4-corner synthetic plot is REVIEW).
    clubbed = "BOUNDARY" in layers or any(l.startswith("REVIEW_FMB_") for l in layers)
    assert clubbed, f"FMB plot not clubbed into combined file; layers={layers}"
    # The clubbed FMB sits in the surveyor UTM frame; its survey number is present.
    sn = [str(e.dxf.text).strip() for e in doc.modelspace().query("TEXT")]
    assert "784" in sn
    xs = [p[0] for e in doc.modelspace().query("LWPOLYLINE")
          for p in e.get_points()]
    assert min(xs) > 600000 and max(xs) < 900000  # UTM Zone 44N easting range


def test_pipeline_no_decoy_still_matches(m1_dxf, tmp_path):
    """Sanity: matching works without a decoy chain present too."""
    from conftest import build_surveyor_dxf

    surveyor = build_surveyor_dxf(tmp_path / "surv_nodecoy.dxf", include_decoy=False)
    results = georef_pipeline(surveyor, [m1_dxf], tmp_path / "out2")
    assert results[0].matched


def test_pipeline_unmatchable_plot_reports_no_match(surveyor_dxf, tmp_path):
    """A plot whose fingerprint exists nowhere in the surveyor data is unmatched,
    and the pipeline records it without raising."""
    # Build an M1 DXF for a triangle with edge lengths unlike anything present.
    from landintel.core.models import Boundary, CornerPoint, Plot
    from landintel.pipeline.m1_extract.to_dxf import write_dxf

    verts = [(0.0, 0.0), (7.3, 0.0), (3.1, 6.7)]
    plot = Plot(
        client_id="c", survey_no="999", district="E", taluk="P", village="INGUR",
        scale=1000, stated_area=0.001,
        boundary=Boundary(points=verts + [verts[0]]),
        corner_points=[CornerPoint(label=str(i), x=x, y=y)
                       for i, (x, y) in enumerate(verts)],
    )
    odd = write_dxf(plot, tmp_path / "m1_odd_999.dxf")

    # Confirm it extracts but should not match the surveyor corridor.
    m1 = extract_m1_dxf(odd)
    assert m1.n_stones == 3

    results = georef_pipeline(surveyor_dxf, [odd], tmp_path / "out3")
    assert len(results) == 1
    assert not results[0].matched
    assert results[0].error  # a reason is recorded
