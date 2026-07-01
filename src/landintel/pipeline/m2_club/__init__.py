"""M2 (new) -- georeference FMB DXFs WITHOUT a surveyor file, then club them.

This is the FMB-only half of the pipeline. M1 produces per-plot FMB DXFs (relative
metres); M2 finds each plot's real-world coordinates and clubs the plots into ONE
georeferenced DXF -- using EVERY available coordinate source, cross-checking them,
never relying on one:

  * cadastral_seat   survey# -> authoritative UTM parcel (TNGIS tiles / client
                     vector), placed rigidly and gated by a strict shape check.
  * gps_seat         operator GPS / control-point correspondences (2-corner fit).
  * relative_club    FMB-to-FMB shared-edge corroboration + gated propagation, so
                     plots tie to each other (the client's "FMBS_STONES_MATCH").

Deterministic math gates decide every ACCEPT (0 false positives); the clubbed
output then feeds M3 (assembly against the surveyor raw-data file).

Public API:
    from landintel.pipeline.m2_club import club_pipeline, ClubResult
    results = club_pipeline(m1_dxf_paths, output_dir, crs,
                            cadastral_source=..., gps_control=...)
"""
from __future__ import annotations

from .cadastral_seat import cadastral_seat
from .gps_seat import gps_seat
from .pipeline import club_pipeline
from .placement import CandidatePlacement, ClubResult

__all__ = [
    "club_pipeline",
    "ClubResult",
    "CandidatePlacement",
    "cadastral_seat",
    "gps_seat",
]
