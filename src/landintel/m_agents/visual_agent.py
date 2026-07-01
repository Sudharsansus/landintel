"""M_visual_agent -- human-like VISUAL QA of M1 output.

WHY
---
M1's numeric gates (boundary closes, computed area within ~5% of the stated area, red-fill
== STONES count) can all PASS while the extracted drawing is visibly WRONG to a human --
a missing edge that still closes into a plausible polygon, a stone cloud in the wrong place,
garbled/absent labels, a shape that does not resemble the FMB. Those are false positives a
reviewer catches at a glance. This agent reproduces that glance: it puts the ORIGINAL FMB
drawing next to the M1-extracted DXF as one image, so a multimodal agent (or a person) can
judge "does the extraction match the drawing?" -- the check the numbers cannot make.

WHAT IT PRODUCES
----------------
For each plot, ``output/<village>/m1_qa/<stem>.png``: LEFT = the FMB PDF page (rasterised),
RIGHT = the M1 DXF rendered in CAD style, with a caption carrying the numeric facts
(stones / closed / area%). ``M_visual_agent.qa_village`` builds them all + an index. The
images are the input to the visual verification pass (the Agent tool reads them and reports
mismatches); this module does the rendering + fact extraction, never the ACCEPT decision.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)


@dataclass
class VisualQAItem:
    """One plot's visual-QA artefact + the numeric facts shown beside it."""
    survey: str
    fmb_pdf: str
    m1_dxf: str
    compare_png: str
    n_stones: int = 0
    is_closed: bool | None = None
    area_pct: float | None = None
    note: str = ""


@dataclass
class VisualQAReport:
    village: str
    items: list[VisualQAItem] = field(default_factory=list)
    index_png: str = ""


def _render_pdf_page(pdf_path, dpi: int = 150):
    """Rasterise the first page of an FMB PDF to an RGB numpy array."""
    import fitz  # PyMuPDF
    import numpy as np
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    doc.close()
    return img[:, :, :3]


def _render_dxf(dxf_path, ax):
    """Render a DXF into a matplotlib axis in CAD style (black background)."""
    import ezdxf
    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
    try:
        from ezdxf.addons.drawing.config import Configuration
        cfg = Configuration(background_policy=None)
    except Exception:  # noqa: BLE001
        cfg = None
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    ctx = RenderContext(doc)
    be = MatplotlibBackend(ax)
    fe = Frontend(ctx, be) if cfg is None else Frontend(ctx, be, config=cfg)
    fe.draw_layout(msp, finalize=True)
    ax.set_aspect("equal")


def render_m1_vs_fmb(fmb_pdf, m1_dxf, out_png, caption: str = "", dpi: int = 150) -> str:
    """Write a side-by-side FMB-PDF vs M1-DXF comparison image. Returns the PNG path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(18, 10))
    fig.patch.set_facecolor("white")

    try:
        axl.imshow(_render_pdf_page(fmb_pdf, dpi=dpi))
    except Exception as exc:  # noqa: BLE001
        axl.text(0.5, 0.5, f"FMB render failed:\n{exc}", ha="center", va="center")
    axl.set_title("FMB drawing (source)", fontsize=13)
    axl.axis("off")

    axr.set_facecolor("black")
    try:
        _render_dxf(m1_dxf, axr)
    except Exception as exc:  # noqa: BLE001
        axr.text(0.5, 0.5, f"DXF render failed:\n{exc}", ha="center", va="center", color="w")
    axr.set_title("M1 extracted DXF", fontsize=13)

    if caption:
        fig.suptitle(caption, fontsize=14, y=0.995)
    fig.savefig(str(out_png), dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out_png)


class M_visual_agent:
    """Human-like visual QA agent for M1 extraction output.

    Usage::

        agent = M_visual_agent("INGUR")
        report = agent.qa_village(fmb_dir="input/INGUR/fmb",
                                  m1_dir="output/INGUR/m1",
                                  out_dir="output/INGUR/m1_qa")

    Each ``report.items[i].compare_png`` is the FMB-vs-DXF image to inspect. The agent does
    NOT decide pass/fail numerically -- it prepares the visual evidence and surfaces the
    numeric facts; the ACCEPT/REJECT judgement is made by the visual reviewer (the Agent
    tool or a person), so a numeric false-positive can be overruled by what the eye sees.
    """

    def __init__(self, village: str):
        self.village = village

    def _facts(self, m1_dxf: Path) -> tuple[int, bool | None]:
        """Cheap numeric facts read straight off the DXF (stone count, boundary closure)."""
        import ezdxf
        try:
            doc = ezdxf.readfile(str(m1_dxf))
            msp = doc.modelspace()
            stones = sum(1 for e in msp if e.dxf.layer == "STONES"
                         and e.dxftype() in ("LWPOLYLINE", "POINT", "CIRCLE"))
            bnd = [e for e in msp if e.dxf.layer == "BOUNDARY" and e.dxftype() == "LWPOLYLINE"]
            closed = None
            if bnd:
                closed = any(bool(getattr(e, "closed", False) or e.dxf.get("flags", 0) & 1)
                             for e in bnd)
            return stones, closed
        except Exception:  # noqa: BLE001
            return 0, None

    def qa_village(self, fmb_dir, m1_dir, out_dir) -> VisualQAReport:
        """Render an FMB-vs-M1 comparison for every extracted plot; return the report."""
        fmb_dir, m1_dir, out_dir = Path(fmb_dir), Path(m1_dir), Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report = VisualQAReport(village=self.village)

        for dxf in sorted(m1_dir.glob("*.dxf")):
            stem = dxf.stem
            pdf = fmb_dir / f"{stem}.pdf"
            if not pdf.exists():
                # tolerate a differing suffix/case
                cand = list(fmb_dir.glob(f"*{stem.split('_')[-1]}.pdf"))
                pdf = cand[0] if cand else pdf
            stones, closed = self._facts(dxf)
            survey = stem.split("_")[-1]
            caption = f"survey {survey}   |   stones={stones}   closed={closed}"
            png = out_dir / f"{stem}.png"
            try:
                render_m1_vs_fmb(pdf, dxf, png, caption=caption)
            except Exception as exc:  # noqa: BLE001
                _log.warning("visual QA render failed for %s: %s", stem, exc)
                continue
            report.items.append(VisualQAItem(
                survey=survey, fmb_pdf=str(pdf), m1_dxf=str(dxf),
                compare_png=str(png), n_stones=stones, is_closed=closed))
        _log.info("M_visual_agent: %d comparison images -> %s", len(report.items), out_dir)
        return report
