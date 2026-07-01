"""M3 annotate -- label each placed parcel with its survey number + ground area.

Adds a text tag at each CONFIDENT plot's centroid in the combined village DXF, on a dedicated
PARCEL_ANNOTATION layer, so the deliverable reads like a cadastral map ("724  0.3891 ha").
Annotation is ADDITIVE and NON-DESTRUCTIVE: it never moves a boundary or stone (the rigid M2
placement is load-bearing and must not be warped) -- it only writes labels on a new layer.
"""

from __future__ import annotations

import logging
from pathlib import Path

_log = logging.getLogger(__name__)

ANNOTATION_LAYER = "PARCEL_ANNOTATION"


def annotate_combined(combined_dxf_path: str | Path,
                      confident_specs: list[tuple[str, str]],
                      layer: str = ANNOTATION_LAYER,
                      text_height: float = 2.0) -> int:
    """Write a "survey# + area(ha)" label at each confident plot's centroid.

    ``confident_specs`` = list of (georef_dxf_path, survey_no). Areas are read from each plot's
    own georeferenced boundary (so the label matches the verified geometry). Returns the number
    of plots annotated. Best-effort per plot; saves the combined file in place.
    """
    import ezdxf
    from .area import plot_footprint

    combined_dxf_path = Path(combined_dxf_path)
    doc = ezdxf.readfile(str(combined_dxf_path))
    msp = doc.modelspace()
    if layer not in doc.layers:
        doc.layers.add(layer, color=3)                    # green annotation layer

    n = 0
    for gp, sn in confident_specs:
        try:
            fp = plot_footprint(gp)
            if fp is None:
                continue
            c = fp.centroid
            ha = fp.area / 10000.0
            msp.add_text(
                f"{sn}  {ha:.4f} ha",
                dxfattribs={"layer": layer, "height": text_height,
                            "insert": (float(c.x), float(c.y))},
            )
            n += 1
        except Exception as exc:  # noqa: BLE001 - one bad plot must not stop annotation
            _log.warning("annotate %s failed: %s", sn, exc)

    doc.saveas(str(combined_dxf_path))
    _log.info("M3 annotate: labelled %d parcel(s) on layer %s in %s",
              n, layer, combined_dxf_path.name)
    return n
