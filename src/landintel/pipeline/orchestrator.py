"""Run one job through the pipeline.

The orchestrator knows the full sequence and owns stage transitions. The modules
themselves are pure logic — they take domain objects and return domain objects;
only this file knows how they chain together.

Current state: M1 (extract) + agent layer (validate, anomaly, audit) are fully
built and wired. M2/M3/M4 are explicit ``NotImplementedError`` stubs — they
raise clearly with a message naming the stage, so a half-built pipeline cannot
pretend to succeed. When a module is ready, replace its stub with the real call.

Job lifecycle managed here:
  INTAKE -> EXTRACT (M1 + agent) -> GEOREF* -> ASSEMBLE* -> REPORT* -> DELIVERED*
  (* = NotImplemented stub, raises clearly)

One flagged plot does NOT block the job — the job advances, the plot is marked
FLAGGED, and the audit trail records why. A hard-failed plot (GeometryError /
unrecoverable) marks the plot FAILED and the job status derives to FAILED via the
Job.status property. The orchestrator records the error and returns without
crashing the worker.

A missing scale on any PDF is a hard GeometryError per that plot — the rest of
the job continues.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..agent.anomaly import check_plot
from ..agent.audit import audit_plot
from ..agent.client import AgentClient
from ..agent.validator import validate_plot
from ..core.enums import PlotStatus, Stage
from ..core.exceptions import GeometryError, LandIntelError
from ..core.models import Job, Plot
from ..logging import log_context
from ..pipeline.m1_extract.anchor import anchor_measurements
from ..pipeline.m1_extract.build_plot import build_plot
from ..pipeline.m1_extract.ocr import extract_text, parse_header
from ..pipeline.m1_extract.pdf_vectors import extract_vectors
from ..pipeline.m1_extract.to_dxf import write_dxf
from ..pipeline.m1_extract.verify_dxf import M1VerifyReport, verify_m1_dxf, write_verify_sidecar
from ..pipeline.m2_club import ClubResult, club_pipeline
from ..pipeline.m2_georef import GeorefResult, georef_pipeline
from ..storage.s3 import download_file, upload_file

__all__ = ["run_job"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage helpers. Naming note (2026-06-29 architecture correction): the NEW M2 is
# the FMB-only club (``_m2_club`` -> m2_club.club_pipeline); the surveyor-matching
# code historically called "M2 georef" (``_m2_georef`` -> m2_georef.georef_pipeline)
# is actually M3. ``_m3_assemble`` / ``_m4_report`` remain explicit stubs; their
# names and messages are unchanged so the orchestrator tests still pin them.
# ---------------------------------------------------------------------------

def _m2_georef(
    m1_dxf_paths: list[Path],
    surveyor_dxf: Path,
    output_dir: Path,
    crs: str = "EPSG:32644",
    schedule_dxf: Path | None = None,
) -> list[GeorefResult]:
    """Georeference a batch of M1 DXFs against the surveyor's field DXF.

    Delegates to :func:`landintel.pipeline.m2_georef.georef_pipeline`, which
    matches each M1 plot's boundary onto the surveyor's stone network, fits a
    2-stage similarity + cadastral transform, and clubs the placed plots into one
    combined village DXF. Per-plot failures are isolated inside the pipeline (each
    plot returns its own ``GeorefResult`` with an ``error`` string) so one bad
    plot never aborts the batch.

    ``schedule_dxf`` (optional): a corridor land-schedule DXF. Its ``SURVEY
    NUMBER`` layer is the IDENTITY GATE -- only plots the corridor actually
    crosses are matched, which removes geometric false positives (a congruent but
    wrong-identity seat). Strongly recommended for tower-corridor surveys, whose
    raw DXF carries no survey numbers.
    """
    if not m1_dxf_paths:
        return []
    corridor_surveys = None
    if schedule_dxf is not None:
        from .m2_georef.extract_surveyor import extract_corridor_surveys
        corridor_surveys = extract_corridor_surveys(schedule_dxf) or None
    return georef_pipeline(surveyor_dxf, m1_dxf_paths, output_dir, crs,
                           corridor_surveys=corridor_surveys)


def _m2_club(
    m1_dxf_paths: list[Path],
    output_dir: Path,
    crs: str = "EPSG:32643",
    cadastral_source: object | None = None,
    gps_control: dict | None = None,
) -> list[ClubResult]:
    """NEW M2 (the FMB-only "club"): georeference M1 FMB DXFs WITHOUT a surveyor
    file and club them into ONE georeferenced DXF.

    Per the 2026-06-29 architecture correction: M2 finds each plot's UTM
    coordinates and clubs the plots using only the FMBs (cadastral seat + GPS seat
    + relative FMB-to-FMB clubbing, cross-checked), and M3 (``_m2_georef`` above,
    the surveyor-matching code) then assembles that clubbed result against the
    surveyor RAW DATA FILE. 0-FP: deterministic gates decide every ACCEPT. See
    :func:`landintel.pipeline.m2_club.club_pipeline`.
    """
    if not m1_dxf_paths:
        return []
    return club_pipeline(m1_dxf_paths, output_dir, crs,
                         cadastral_source=cadastral_source, gps_control=gps_control)


def _m3_assemble(plots: list[Plot], base_file: Path) -> Path:
    raise NotImplementedError(
        "M3 assemble not yet built — village DWG assembly onto the base-file "
        "frame is deferred. Replace this stub once m3_assemble/ is implemented."
    )


def _m4_report(job: Job, village_dwg: Path) -> str:
    raise NotImplementedError(
        "M4 report not yet built — area statement, Excel sheet, and S3 delivery "
        "are deferred. Replace this stub once m4_report/ is implemented."
    )


# ---------------------------------------------------------------------------
# M1 + agent: the built part of the pipeline
# ---------------------------------------------------------------------------

def _run_m1_and_agent(
    pdf_path: Path,
    client_id: str,
    agent_client: AgentClient | None,
    output_dir: Path,
) -> tuple[Plot, Path, M1VerifyReport]:
    """Extract, validate, check, and VERIFY one FMB PDF.

    Returns ``(plot, dxf_path, verify_report)``. The verification gate runs on
    the WRITTEN DXF (the exact artifact M2 consumes): a file failing a hard
    geometry/structure check is marked here so the caller can withhold it from
    M2. A ``.verify.txt`` sidecar is written next to every DXF.
    """
    vectors = extract_vectors(pdf_path)
    detections = extract_text(pdf_path)
    header = parse_header(detections)

    plot = build_plot(
        client_id=client_id,
        vectors=vectors,
        detections=detections,
        anchor_result=anchor_measurements(vectors, detections),
        header=header,
    )

    validate_plot(plot, client=agent_client)
    report = check_plot(plot)
    audit_line = audit_plot(plot, report=report)
    logger.info("plot audited", extra={"survey_no": plot.survey_no, "audit": audit_line})

    dxf_path = write_dxf(plot, output_dir / f"survey_{plot.survey_no}.dxf")

    # M1 -> M2 verification gate on the written artifact.
    verify_report = verify_m1_dxf(dxf_path, stated_area_ha=plot.stated_area)
    write_verify_sidecar(verify_report)
    if not verify_report.proper:
        reasons = ", ".join(f"{c.name}: {c.detail}" for c in verify_report.failures)
        plot.status = PlotStatus.FLAGGED
        plot.flags.append(f"[improper_dxf] {reasons}")
        logger.warning("DXF failed M1->M2 verification",
                       extra={"survey_no": plot.survey_no, "reasons": reasons})

    return plot, dxf_path, verify_report


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_job(
    job: Job,
    *,
    output_dir: Path,
    agent_client: AgentClient | None = None,
    surveyor_dxf: Path | None = None,
    schedule_dxf: Path | None = None,
    cadastral_source: object | None = None,
    gps_control: dict | None = None,
    crs: str = "EPSG:32643",
) -> Job:
    """Run ``job`` through the pipeline, mutating and returning it.

    Args:
        job: The job to run. Mutated in place (stage, audit, plot statuses).
        output_dir: Where intermediate files (DXFs) are written.
        agent_client: Agent client for OCR validation escalation. When ``None``,
            the validator operates in deterministic-only mode (no API calls).
        surveyor_dxf: Optional path to the surveyor's field-surveyed reference
            DXF (real UTM coordinates). When provided, M2 georeferencing runs
            after M1 on all extracted plots; matched plots are marked
            GEOREFERENCED and their georeferenced DXFs are added to the job
            outputs. When ``None`` (default), M2 is skipped — the job advances
            through M1+agent only, exactly as before.
        schedule_dxf: Optional corridor land-schedule DXF. Its ``SURVEY NUMBER``
            layer is the M3 identity gate — only plots the corridor crosses are
            matched, eliminating geometric false positives. Recommended whenever
            the surveyor DXF is a tower-corridor survey (no embedded survey
            numbers). Ignored when ``surveyor_dxf`` is ``None``.
        cadastral_source: Optional cadastral reference (TNGIS tiles / client vector)
            mapping survey# → UTM parcel. When given (or ``gps_control``), the NEW
            M2 club stage runs after M1: it georeferences + clubs the FMBs WITHOUT a
            surveyor file and the agent layer self-verifies the result. Default
            ``None`` → the M2 club stage is skipped (M1+agent only, as before).
        gps_control: Optional ``{survey_no: [(corner_label, (utm_x, utm_y)), ...]}``
            operator control points, also triggering the M2 club stage.
        crs: UTM CRS for the M2 club stage (default ``EPSG:32643``, INGUR/Erode 43N).

    Returns:
        The job after the built stages have run (M1 + agent, plus M2 when a
        surveyor DXF is supplied). M3/M4 remain stubs and are not called here.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    m1_dxf_paths: list[Path] = []

    with log_context(job_id=job.id, client_id=job.client_id):
        logger.info("job started", extra={"stage": Stage.EXTRACT.value})
        job.stage = Stage.EXTRACT

        for pdf_path_str in job.input_files:
            # In production input_files holds S3 keys (relative: client_id/jobs/…).
            # In tests and dev they are absolute filesystem paths (/tmp/…, C:\…).
            # is_absolute() cleanly separates the two: relative → S3, absolute → local.
            raw = Path(pdf_path_str)
            if raw.is_absolute():
                pdf_path = raw
            else:
                local = output_dir / raw.name
                pdf_path = download_file(pdf_path_str, local)
            with log_context(pdf=pdf_path.name):
                try:
                    plot, dxf_path, verify_report = _run_m1_and_agent(
                        pdf_path, job.client_id, agent_client, output_dir
                    )
                    job.plots.append(plot)
                    # Only PROPER DXFs are shifted to M2; an improper one is kept
                    # for review but withheld from georeferencing (its geometry
                    # would corrupt the match/transform).
                    if verify_report.proper:
                        m1_dxf_paths.append(dxf_path)
                    else:
                        job.audit.append(
                            f"Survey {plot.survey_no}: DXF IMPROPER — withheld from M2 "
                            f"({', '.join(c.name for c in verify_report.failures)})"
                        )
                    job.audit.append(audit_plot(plot))
                    # Upload the DXF to S3 so it survives past the worker's /tmp.
                    # In production the API and worker are separate containers —
                    # S3 is the only shared storage. If S3 is not configured,
                    # store the local path (useful in dev/tests) but log clearly
                    # so the missing credentials are visible in the worker logs
                    # and the job's audit trail.
                    try:
                        s3_key = upload_file(
                            job.client_id, job.id, dxf_path,
                            filename=f"m1_extract_survey_{plot.survey_no}.dxf",
                        )
                        job.output_files.append(s3_key)
                        logger.info("DXF uploaded to S3", extra={"key": s3_key})
                    except Exception as upload_exc:
                        job.output_files.append(str(dxf_path))
                        msg = (
                            f"Survey {plot.survey_no}: S3 upload failed — "
                            f"set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / "
                            f"S3_BUCKET in Render env vars for production downloads. "
                            f"({type(upload_exc).__name__}: {upload_exc})"
                        )
                        logger.warning(msg)
                        job.audit.append(msg)
                except GeometryError as exc:
                    # Hard geometry failure on this PDF — mark it but keep going.
                    logger.error("plot failed", exc_info=exc)
                    failed_plot = Plot(
                        client_id=job.client_id,
                        survey_no=pdf_path.stem,
                        district="", taluk="", village="",
                        status=PlotStatus.FAILED,
                        flags=[f"[geometry_error] {exc.message}"],
                    )
                    job.plots.append(failed_plot)
                    job.audit.append(
                        f"Survey {pdf_path.stem}: FAILED — {exc.message}"
                    )
                except LandIntelError as exc:
                    logger.error("plot error", exc_info=exc)
                    job.audit.append(
                        f"Survey {pdf_path.stem}: ERROR — {exc.message}"
                    )
                except Exception as exc:  # noqa: BLE001
                    # Catch-all for unexpected errors at the PDF boundary, e.g.
                    # pymupdf.FileNotFoundError (inherits RuntimeError, not
                    # built-in FileNotFoundError), corrupt files, permission errors.
                    # Log and continue rather than crashing the whole job.
                    logger.error("unexpected plot error", exc_info=exc)
                    job.audit.append(
                        f"Survey {pdf_path.stem}: ERROR — {type(exc).__name__}: {exc}"
                    )

        # --- M2 club (NEW M2): georeference + club the FMBs WITHOUT a surveyor --
        # file. Opt-in: runs when a cadastral source and/or GPS control points are
        # given. Each FMB is seated by survey# (cadastral), operator control points
        # (GPS), or shared-edge propagation, then clubbed into one georeferenced DXF.
        # The unified agent layer self-verifies the result (0-FP invariants + the
        # path-to-100% worklist). A whole-stage failure is caught, never crashes a job.
        if (cadastral_source is not None or gps_control) and m1_dxf_paths:
            job.stage = Stage.GEOREF
            logger.info("job M2 club", extra={"stage": Stage.GEOREF.value})
            try:
                club_dir = output_dir / "m2_club"
                club_results = _m2_club(m1_dxf_paths, club_dir, crs,
                                        cadastral_source=cadastral_source,
                                        gps_control=gps_control)
                for cr in club_results:
                    if cr.output_file:
                        job.output_files.append(cr.output_file)
                    is_accept = cr.recommendation in ("ACCEPT", "ACCEPT_SEEDED")
                    new_status = (
                        PlotStatus.GEOREFERENCED if is_accept
                        else PlotStatus.FLAGGED if cr.recommendation == "REVIEW"
                        else None)
                    if new_status is not None:
                        for p in job.plots:
                            if (p.survey_no == cr.survey_number
                                    and p.status is not PlotStatus.FAILED):
                                p.status = new_status
                    job.audit.append(
                        f"Survey {cr.survey_number}: M2 club {cr.recommendation} "
                        f"({cr.method or 'no method'})")
                combined = club_dir / "clubbed_village.dxf"
                if combined.exists():
                    job.output_files.append(str(combined))
                # Agent layer self-verifies the clubbed M2 (FP-safe; demote-only).
                try:
                    from ..agents import run_agent_layer
                    summary = run_agent_layer(
                        club_results, club_dir,
                        context={"village": job.client_id, "crs": crs,
                                 "cadastral_source": cadastral_source})
                    job.audit.append(
                        f"M2 club agent layer: shippable={summary['shippable']}, "
                        f"{summary['n_requests']} input request(s) to reach 100%")
                except Exception as exc:  # noqa: BLE001 - agent layer never fails a job
                    logger.error("M2 club agent layer failed", exc_info=exc)
            except Exception as exc:  # noqa: BLE001
                logger.error("M2 club stage failed", exc_info=exc)
                job.audit.append(
                    f"M2 club stage failed — {type(exc).__name__}: {exc}")

        # --- M3 georef (only when a surveyor reference DXF is supplied) ------
        # M3 matches each (clubbed) M1 plot's boundary onto the surveyor's stone
        # network and writes a georeferenced UTM DXF. (This is the code historically
        # named "M2 georef"; per the 2026-06-29 correction it is M3 -- assembly
        # against the surveyor RAW DATA FILE.) Batch (one surveyor file, many plots),
        # runs once after M1. A whole-stage failure is caught and recorded.
        if surveyor_dxf is not None and m1_dxf_paths:
            job.stage = Stage.GEOREF
            logger.info("job georeferencing", extra={"stage": Stage.GEOREF.value})
            try:
                georef_dir = output_dir / "georef"
                results = _m2_georef(m1_dxf_paths, surveyor_dxf, georef_dir,
                                     schedule_dxf=schedule_dxf)
                accepted = 0
                for gr in results:
                    # Disposition drives plot status (every plot is accounted for):
                    #   ACCEPT      -> GEOREFERENCED, georef DXF delivered.
                    #   REVIEW      -> FLAGGED, georef DXF delivered for a human to
                    #                  confirm (no uncertain match silently trusted).
                    #   NO_COVERAGE -> the surveyor never traced this plot (its
                    #                  boundary is NOT on the traced lines, so any
                    #                  congruent-stone match is coincidental). The
                    #                  M1 output stands; we do NOT deliver a
                    #                  misleading georef DXF and do NOT flag it
                    #                  (nothing to review) -- it stays EXTRACTED.
                    if gr.recommendation == "NO_COVERAGE":
                        job.audit.append(
                            f"Survey {gr.survey_number or Path(gr.m1_file).stem}: "
                            f"M2 no surveyor coverage "
                            f"(chain coverage {gr.chain_coverage:.0%}) — "
                            f"M1 output retained, not georeferenced"
                        )
                        continue

                    if gr.output_file:
                        job.output_files.append(gr.output_file)
                    is_accept = gr.recommendation in ("ACCEPT", "ACCEPT_SEEDED")
                    if is_accept:
                        accepted += 1
                    new_status = (PlotStatus.GEOREFERENCED if is_accept
                                  else PlotStatus.FLAGGED)
                    for p in job.plots:
                        if (p.survey_no == gr.survey_number
                                and p.status is not PlotStatus.FAILED):
                            p.status = new_status
                            if not is_accept:
                                p.flags.append(
                                    f"[m2_review] {gr.n_inliers}/{gr.n_corners} corners, "
                                    f"chain coverage {gr.chain_coverage:.0%}")
                    job.audit.append(
                        f"Survey {gr.survey_number}: M2 {gr.recommendation} — "
                        f"{gr.n_inliers}/{gr.n_corners} corner inliers, "
                        f"chain coverage {gr.chain_coverage:.0%}, "
                        f"residual={gr.fingerprint_score:.2f}m"
                    )
                logger.info("M2 georef complete",
                            extra={"accepted": accepted, "matched":
                                   sum(1 for r in results if r.matched),
                                   "total": len(results)})
                # Agent layer self-verifies M3 too (same FP-safe, demote-only gates).
                try:
                    from ..agents import run_agent_layer
                    m3_summary = run_agent_layer(
                        results, georef_dir,
                        context={"village": job.client_id, "crs": crs,
                                 "surveyor": surveyor_dxf})
                    job.audit.append(
                        f"M3 agent layer: shippable={m3_summary['shippable']}, "
                        f"{m3_summary['n_requests']} input request(s) to reach 100%")
                except Exception as exc:  # noqa: BLE001 - agent layer never fails a job
                    logger.error("M3 agent layer failed", exc_info=exc)
            except Exception as exc:  # noqa: BLE001
                # A whole-stage M3 failure (e.g. unreadable surveyor DXF) is
                # recorded but does not fail the job — M1 output still stands.
                logger.error("M3 georef stage failed", exc_info=exc)
                job.audit.append(
                    f"M3 georef stage failed — {type(exc).__name__}: {exc}"
                )

        # Advance to DELIVERED so Job.status resolves to COMPLETED/NEEDS_REVIEW.
        # M3/M4 remain stubs; when built, advance through ASSEMBLE -> REPORT here.
        job.stage = Stage.DELIVERED

        logger.info(
            "pipeline complete",
            extra={
                "plots": len(job.plots),
                "flagged": sum(1 for p in job.plots if p.status is PlotStatus.FLAGGED),
                "failed": sum(1 for p in job.plots if p.status is PlotStatus.FAILED),
                "georeferenced": sum(
                    1 for p in job.plots if p.status is PlotStatus.GEOREFERENCED
                ),
            },
        )

    return job
