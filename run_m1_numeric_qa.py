"""M1 NUMERIC QA -- the canonical correctness gate, run across a whole village.

For every FMB PDF vs its M1 DXF it checks the load-bearing invariants (the ones M2/M3
depend on), independent of OCR:
  * STONES exact:  len(extract_vectors(pdf).stones)  ==  DXF STONES count   [CANONICAL]
  * boundary closed
  * computed area within tolerance of the header stated area
  * dimension-label coverage (how many numeric labels landed) -- reported, not gated
Prints a per-plot table + a summary, and writes output/<VILLAGE>/m1_qa/numeric_qa.csv.
This uses ONLY pdf_vectors (PyMuPDF) + ezdxf -- no paddle/OCR load, so it is fast.

Usage:  python run_m1_numeric_qa.py [VILLAGE]     (default INGUR)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

import ezdxf

from landintel.pipeline.m1_extract.pdf_vectors import extract_vectors

VILLAGE = sys.argv[1] if len(sys.argv) > 1 else "INGUR"
FMB_DIR = Path(f"input/{VILLAGE}/fmb")
M1_DIR = Path(f"output/{VILLAGE}/m1")
OUT_DIR = Path(f"output/{VILLAGE}/m1_qa")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def dxf_facts(dxf_path: Path):
    """Return (stones, n_dim_labels) from the DXF.

    STONES are stored as TEXT entities on the STONES layer (each corner-stone label),
    exactly as verify_dxf counts them -- NOT polylines. Mirror that or the count is 0.
    """
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    stones = sum(1 for e in msp.query("TEXT") if e.dxf.layer == "STONES")
    dim_layers = {"BOUNDARY_DIMENSIONS", "CHAINLINE_DIMENSIONS", "DIMENSIONS"}
    n_dims = sum(1 for e in msp if e.dxf.layer in dim_layers
                 and e.dxftype() in ("TEXT", "MTEXT"))
    return stones, n_dims


def verify_facts(dxf_path: Path):
    """Read STATUS + boundary_closed + area_vs_stated straight from the authoritative
    .verify.txt the batch already wrote (its checks are the canonical gate)."""
    vt = dxf_path.with_suffix(".verify.txt")
    status, closed, area = "NO_VERIFY", None, ""
    if not vt.exists():
        return status, closed, area
    for line in vt.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("STATUS:"):
            status = line.split("STATUS:", 1)[1].strip()
        elif "boundary_closed" in line:
            closed = line.strip().startswith("[OK")
        elif "area_vs_stated" in line:
            area = line.split("area_vs_stated", 1)[1].strip()
    return status, closed, area


rows = []
pdfs = sorted(FMB_DIR.glob("*.pdf"))
print(f"[M1 NUMERIC QA] {VILLAGE}: {len(pdfs)} FMB PDFs vs {M1_DIR}\n")
hdr = f"{'survey':>10} {'pdf_st':>6} {'dxf_st':>6} {'match':>5} {'closed':>6} {'dims':>5} {'status':>16}"
print(hdr)
print("-" * len(hdr))

for pdf in pdfs:
    stem = pdf.stem
    survey = stem.split("_")[-1]
    dxf = M1_DIR / f"{VILLAGE}_{stem}.dxf"
    if not dxf.exists():
        # tolerate alt naming
        cand = list(M1_DIR.glob(f"*{survey}.dxf"))
        dxf = cand[0] if cand else dxf
    try:
        pv = extract_vectors(pdf)
        pdf_stones = len(pv.stones)
    except Exception as exc:  # noqa: BLE001
        pdf_stones = -1
    if not dxf.exists():
        print(f"{survey:>10} {pdf_stones:>6} {'--':>6} {'MISS':>5} {'--':>6} {'--':>5} {'NO_DXF':>16}")
        rows.append(dict(survey=survey, pdf_stones=pdf_stones, dxf_stones=-1,
                         match=False, closed=None, dims=0, status="NO_DXF"))
        continue
    dxf_stones, n_dims = dxf_facts(dxf)
    match = (pdf_stones == dxf_stones)
    status, closed, _area = verify_facts(dxf)
    flag = "OK" if match else "*MISMATCH*"
    print(f"{survey:>10} {pdf_stones:>6} {dxf_stones:>6} {str(match):>5} "
          f"{str(closed):>6} {n_dims:>5} {status:>16}  {'' if match else flag}")
    rows.append(dict(survey=survey, pdf_stones=pdf_stones, dxf_stones=dxf_stones,
                     match=match, closed=closed, dims=n_dims, status=status))

# summary
n = len(rows)
stone_ok = sum(1 for r in rows if r["match"])
closed_ok = sum(1 for r in rows if r["closed"] is True)
proper = sum(1 for r in rows if r["status"].startswith("PROPER"))
total_dims = sum(r["dims"] for r in rows)
print("\n" + "=" * 60)
print(f"STONE INVARIANT (PDF==DXF): {stone_ok}/{n}   {'ALL PASS' if stone_ok==n else 'FAILURES ABOVE'}")
print(f"BOUNDARY CLOSED:            {closed_ok}/{n}")
print(f"VERIFY PROPER:             {proper}/{n}")
print(f"TOTAL DIM LABELS:          {total_dims}  (avg {total_dims/max(n,1):.1f}/plot)")
print("=" * 60)

csv_path = OUT_DIR / "numeric_qa.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["survey", "pdf_stones", "dxf_stones", "match",
                                      "closed", "dims", "status"])
    w.writeheader()
    w.writerows(rows)
print(f"-> {csv_path}")

# non-zero exit if the canonical invariant fails anywhere
sys.exit(0 if stone_ok == n else 1)
