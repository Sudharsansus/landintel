"""StoneReaderAgent -- reads the surveyor's field STONE POINTS + their EXACT UTM COORDINATES,
verifies the read, and publishes the stone catalog the matcher aligns FMB corners against.

ONE read, not three: every surveyor stone in the RAW DATA DXF *is* a UTM (x, y) point carrying a
type code (B / BS / RBS / VBS / RB / RS). "Stone-point reader", "coordinate reader" and "UTM
reader" are the same thing -- so this is a single agent, not three redundant ones. It does NOT
re-implement the extractor; it wraps ``extract_surveyor`` with self-checks so a bad / empty /
wrong-CRS surveyor file is caught BEFORE matching, and so the matcher aligns to the EXACT,
verified points -- never a silently-dropped or out-of-range coordinate.

FP-safe + no overfit: it only READS and VERIFIES (it never places or accepts anything), and the
UTM sanity envelope is the whole-Tamil-Nadu span across BOTH zones (43N + 44N) -- a general
wrong-CRS trap, not a per-village constant.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..pipeline.m2_georef.extract_surveyor import extract_surveyor
from .base import Agent, AgentReport, Check, Severity

# Tamil Nadu UTM sanity envelope (EPSG:32643 43N + 32644 44N), deliberately generous: it catches
# a wrong CRS (lat/lon, a different projection, a shifted datum) without being village-specific.
_TN_X = (100_000.0, 900_000.0)
_TN_Y = (800_000.0, 1_800_000.0)


class StoneReaderAgent(Agent):
    name = "stone_reader"

    def read(self, surveyor_path, bbox=None, crs: str = "EPSG:32643"):
        """Read + VERIFY the surveyor boundary stone points. Returns ``(SurveyorData, AgentReport)``.
        The returned data's ``stone_positions`` are the EXACT UTM coordinates the matcher uses."""
        rep = AgentReport(agent=self.name)
        data = extract_surveyor(surveyor_path, bbox=bbox)
        data.crs = crs
        data.build_index()
        pos = data.stone_positions
        n = len(pos)

        rep.checks.append(Check(
            "stones_found", Severity.OK if n > 0 else Severity.FAIL,
            f"{n} boundary stone points read"
            + ("" if n else " -- matching impossible (empty / unreadable / no boundary codes)")))
        if n == 0:
            return data, rep

        finite = bool(np.isfinite(pos).all())
        rep.checks.append(Check(
            "coords_finite", Severity.OK if finite else Severity.FAIL,
            "all stone coordinates finite" if finite else "NaN/inf coordinate(s) present"))

        x, y = pos[:, 0], pos[:, 1]
        in_env = (x >= _TN_X[0]) & (x <= _TN_X[1]) & (y >= _TN_Y[0]) & (y <= _TN_Y[1])
        n_out = int((~in_env).sum())
        rep.checks.append(Check(
            "utm_in_tn_envelope", Severity.OK if n_out == 0 else Severity.WARN,
            "all UTM within the Tamil Nadu 43N/44N envelope" if n_out == 0
            else f"{n_out}/{n} stones outside the TN UTM envelope -- check the CRS of the DXF"))

        uniq = len({(round(float(a), 3), round(float(b), 3)) for a, b in pos})
        ndup = n - uniq
        rep.checks.append(Check(
            "duplicate_positions", Severity.OK if ndup == 0 else Severity.INFO,
            "no duplicate stone coordinates" if ndup == 0
            else f"{ndup} co-located marker(s) at identical coordinates"))

        ext = data.extent
        area_km2 = max((ext[2] - ext[0]) * (ext[3] - ext[1]) / 1e6, 1e-9)
        rep.notes.append(
            f"stone catalog: {n} exact UTM points | codes={data.code_distribution} | "
            f"extent={tuple(round(v) for v in ext)} m | density={n / area_km2:.0f}/km^2 | crs={crs}")
        return data, rep

    def write_catalog(self, data, path) -> Path:
        """Publish the EXACT stone points the matcher aligns to (index, x_utm, y_utm, code, crs)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["index", "x_utm", "y_utm", "code", "crs"])
            for s in data.stones:
                w.writerow([s.index, f"{s.x:.4f}", f"{s.y:.4f}", s.code, data.crs])
        return path
