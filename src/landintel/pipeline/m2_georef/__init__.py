"""M2 -- Georeferencing: FMB DXF (relative metres) -> georeferenced UTM DXF.

M2 takes M1-produced FMB DXFs (relative coordinates extracted from Tamil Nadu
Field Measurement Book PDFs) and matches them against a surveyor's field-surveyed
DXF containing real UTM Zone 44N coordinates. The output is georeferenced DXFs
with all plot geometry transformed to real-world coordinates, EPSG:32644 CRS
metadata embedded, and validated through a 7-check verification suite.

Pipeline stages (see the module-level docstrings for detail):
  extract_surveyor -> SurveyorData     parse surveyor DXF (boundary stones + chains)
  extract_m1       -> M1PlotData       parse M1 DXF (stones, outer boundary, edges)
  match            -> MatchResult       3-layer fingerprint + neighborhood matching
  transform        -> umeyama + cadastral_adjust   similarity fit + LSQ refinement
  warp             -> warp boundary/generic vertices to follow adjusted stones
  output_dxf       -> write_georef_dxf  write the georeferenced UTM DXF
  verify           -> verify_georef_dxf 7-check quality gate
  pipeline         -> georef_pipeline   end-to-end orchestration

Public API:
    from landintel.pipeline.m2_georef import georef_pipeline, GeorefResult
    results = georef_pipeline(surveyor_dxf, m1_dxf_paths, output_dir, crs)

Layer-name binding: extraction/output/verify reference ``core.enums.LayerType``
(the same source ``m1_extract.to_dxf`` writes from), so M2 cannot drift from
M1's actual layer names.
"""

from __future__ import annotations

from .extract_m1 import M1Edge, M1PlotData, M1Stone, extract_m1_dxf
from .extract_surveyor import (
    SurveyorChain,
    SurveyorData,
    SurveyorStone,
    extract_surveyor,
)
from .match import MatchResult, match_plot
from .pipeline import GeorefResult, georef_pipeline, georef_single
from .transform import cadastral_adjust, umeyama
from .verify import VerifyCheck, VerifyResult, verify_georef_dxf

__all__ = [
    # Pipeline entry points
    "georef_pipeline",
    "georef_single",
    "GeorefResult",
    # Extraction
    "extract_surveyor",
    "SurveyorData",
    "SurveyorStone",
    "SurveyorChain",
    "extract_m1_dxf",
    "M1PlotData",
    "M1Stone",
    "M1Edge",
    # Matching
    "match_plot",
    "MatchResult",
    # Transform
    "umeyama",
    "cadastral_adjust",
    # Verify
    "verify_georef_dxf",
    "VerifyResult",
    "VerifyCheck",
]
