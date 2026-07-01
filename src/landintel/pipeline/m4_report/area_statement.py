"""Generate the PDF area statement for a job.

Single responsibility: produce a clean, government-document-style PDF per job
showing each survey's identity, stated vs computed area, closure status, and
review flags. This is what the client reads.

Layout mirrors Tamil Nadu survey document conventions: header with job metadata,
then a simple table (one row per plot), then a flag/notes section for any flagged
plots. No graphics — plain Helvetica, tabular, A4.

Generated entirely in memory (returns bytes); the caller writes or uploads.
PaddleOCR's comma-decimal source values are already normalized at this point
(validator.py ran first), so all numbers here are clean floats.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

from ...core.enums import PlotStatus
from ...core.models import Job, Plot

__all__ = ["generate_area_statement"]

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm

# Tamil Nadu survey document palette: black text, light grey header rows.
_HEADER_BG = colors.HexColor("#D9D9D9")
_FLAGGED_BG = colors.HexColor("#FFF2CC")
_FAILED_BG = colors.HexColor("#FFE0E0")
_OK_BG = colors.white


def _area_error(plot: Plot) -> float | None:
    if plot.boundary is None or plot.stated_area is None or plot.stated_area <= 0:
        return None
    computed_ha = plot.boundary.computed_area / 10_000.0
    return abs(computed_ha - plot.stated_area) / plot.stated_area


def _row_bg(plot: Plot) -> Any:
    if plot.status is PlotStatus.FAILED:
        return _FAILED_BG
    if plot.status is PlotStatus.FLAGGED:
        return _FLAGGED_BG
    return _OK_BG


def generate_area_statement(job: Job) -> bytes:
    """Generate the area statement PDF for ``job`` and return its bytes.

    Args:
        job: The completed (or partial) job. Plots need not all be validated —
            whatever state they are in is rendered honestly.

    Returns:
        Raw PDF bytes ready to write to disk or upload to S3.
    """
    buf = io.BytesIO()
    styles = getSampleStyleSheet()

    # --- Document ---
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
    )

    elements: list[Any] = []

    # --- Title block ---
    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"], fontSize=14, alignment=TA_CENTER,
        spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "Sub", parent=styles["Normal"], fontSize=9, alignment=TA_CENTER,
        spaceAfter=2, textColor=colors.grey,
    )
    elements.append(Paragraph("Survey and Settlement Department", sub_style))
    elements.append(Paragraph("Government of Tamil Nadu", sub_style))
    elements.append(Spacer(1, 3 * mm))
    elements.append(Paragraph("AREA STATEMENT", title_style))
    elements.append(Spacer(1, 2 * mm))

    # --- Job metadata ---
    if job.plots:
        p0 = job.plots[0]
        meta_rows = [
            ["District", p0.district, "Taluk", p0.taluk],
            ["Village", p0.village, "Job ID", job.id[:16] + "…"],
            ["Generated", datetime.now(timezone.utc).strftime("%d-%m-%Y %H:%M UTC"), "", ""],
        ]
        meta_table = Table(meta_rows, colWidths=[30*mm, 60*mm, 30*mm, 60*mm])
        meta_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F5F5")),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 5 * mm))

    # --- Main area table ---
    header = [
        "Survey No.", "Status", "Stated Area\n(Ha)", "Computed Area\n(Ha)", "Error %",
        "Boundary", "Stones",
    ]
    rows: list[list[Any]] = [header]
    row_styles: list[tuple] = []

    for idx, plot in enumerate(job.plots):
        row_idx = idx + 1  # +1 for header row
        computed_ha = (
            round(plot.boundary.computed_area / 10_000.0, 3)
            if plot.boundary is not None else "—"
        )
        stated = round(plot.stated_area, 3) if plot.stated_area is not None else "—"
        error = _area_error(plot)
        error_str = f"{error:.1%}" if error is not None else "—"
        closed = (
            "Closed" if (plot.boundary and plot.boundary.is_closed)
            else "Open" if plot.boundary
            else "—"
        )
        rows.append([
            plot.survey_no,
            plot.status.value.replace("_", " ").title(),
            str(stated),
            str(computed_ha),
            error_str,
            closed,
            str(len(plot.corner_points)),
        ])
        bg = _row_bg(plot)
        row_styles.append(("BACKGROUND", (0, row_idx), (-1, row_idx), bg))

    col_widths = [22*mm, 24*mm, 28*mm, 28*mm, 18*mm, 20*mm, 16*mm]
    area_table = Table(rows, colWidths=col_widths, repeatRows=1)
    base_style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BG),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9F9F9")]),
    ]
    area_table.setStyle(TableStyle(base_style + row_styles))
    elements.append(area_table)
    elements.append(Spacer(1, 5 * mm))

    # --- Summary row ---
    total = len(job.plots)
    flagged = sum(1 for p in job.plots if p.status is PlotStatus.FLAGGED)
    failed = sum(1 for p in job.plots if p.status is PlotStatus.FAILED)
    ok = total - flagged - failed
    summary_style = ParagraphStyle("Summary", parent=styles["Normal"], fontSize=9)
    elements.append(Paragraph(
        f"<b>Summary:</b> {total} plot(s) — {ok} OK, {flagged} flagged for review, "
        f"{failed} failed.",
        summary_style,
    ))
    elements.append(Spacer(1, 4 * mm))

    # --- Flags section (only for flagged/failed plots) ---
    flagged_plots = [p for p in job.plots if p.flags]
    if flagged_plots:
        elements.append(Paragraph("<b>Review Notes</b>", styles["Heading3"]))
        elements.append(Spacer(1, 2 * mm))
        note_style = ParagraphStyle(
            "Note", parent=styles["Normal"], fontSize=8,
            leftIndent=5 * mm, spaceAfter=2,
        )
        for plot in flagged_plots:
            elements.append(Paragraph(
                f"<b>Survey {plot.survey_no}:</b> "
                + "; ".join(plot.flags[:5])  # cap at 5 flags to keep it readable
                + ("…" if len(plot.flags) > 5 else ""),
                note_style,
            ))

    doc.build(elements)
    return buf.getvalue()
