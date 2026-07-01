"""M5 cadastral reference layer: make external cadastral parcels usable in M2.

A ``CadastralSource`` resolves a survey number to its authoritative UTM parcel
polygon, so M2 can georeference an FMB onto its real footprint -- including the
off-corridor plots the surveyor corridor never traced.

Sources:
  - ``VectorFileCadastralSource`` -- a client GeoJSON / KML / KMZ / Shapefile.
  - ``S3CadastralSource`` -- the public TN cadastral web tiles (boundaries + survey
    labels), co-registered with the surveyor frame.
  - ``TngisCadastralSource`` -- inert (no reachable public vector endpoint).
"""

from .source import (
    CadastralParcel,
    CadastralSource,
    TngisCadastralSource,
    VectorFileCadastralSource,
    load_cadastral,
)

__all__ = [
    "CadastralParcel",
    "CadastralSource",
    "VectorFileCadastralSource",
    "TngisCadastralSource",
    "load_cadastral",
    "S3CadastralSource",
]


def __getattr__(name):
    # Lazy import so the (cv2/paddle-heavy) S3 path loads only when used.
    if name == "S3CadastralSource":
        from .s3_tiles import S3CadastralSource
        return S3CadastralSource
    raise AttributeError(name)
