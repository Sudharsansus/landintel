# M3 — Issue Brief for Experts (Erode: NASIYANUR / MOOLAKARAI / KANDAMPALAYAM)

**Read with the code on GitHub** (branch `main`, commit `596b9b5`):
`run_m3.py`, `src/landintel/pipeline/m2_georef/extract_surveyor.py`,
`src/landintel/pipeline/m2_georef/match.py` (`geometric_match`),
`src/landintel/pipeline/m2_georef/transform.py` (`umeyama`),
`src/landintel/pipeline/m2_georef/stone_matcher.py` (scale-locked rigid matcher).

---

## 0. The three non-negotiable rules (M3 must obey all)
1. **DON'T OVERFIT** — every gate/threshold is a general function of the data, never a
   per-village or per-file constant.
2. **RIGID ONLY** — placement is rotation + translation, **scale locked to 1**. We align an
   FMB onto surveyor stones by its SHARED STONES; we never stretch/warp FMB edge lengths.
3. **M1 QUALITY FIRST** — M1 corner rings are the template. They are already validated
   (red-fill == STONES exact, boundary closes, area within ~5% of the FMB header).

## 1. What M3 IS and what we WANT
M3 is the **final, survey-grade deliverable**: every M1 FMB placed so it **exactly sits on
the surveyor's field-measured stones**.

- **Inputs allowed:** M1 FMB DXFs + surveyor `RAW DATA.dxf` + Google Maps (village LOCATION
  only) + TNGIS/XML (per-plot SEAT hint only, to crop candidate stones).
- **Explicitly forbidden input:** M2's output. M3 must be 100% independent of M2, so an M2
  error can NEVER cascade into M3 — and M3 can place the very plots M2 gets wrong.
- **Want:** georeferenced DXF per placed plot + merged village DXF, on surveyor UTM
  coordinates, at survey-grade residual (~1.5–2 m), with an HONEST per-plot disposition and
  a QA overlay an expert can eyeball.

## 2. THE decisive issue — the surveyor file is a pure STONE CLOUD (no traced boundary)
Measured on the real file `C:\...\M3 INPUT\RAW DATA.dxf`:

```
ENTITY TYPES : {'TEXT': 8809, 'LWPOLYLINE': 2}     <- only 2 polylines in the whole file
LAYERS       : {'Point_Code': 8811}                <- everything on one layer
BOUNDARY CODES: B 5158, BS 1495, RBS 495, RS 58, VBS 43, RB 4   -> 7273 boundary stones
AUX (excluded): FNC 482, T 400, CW 179, TW 108, HD 76, TX 58, FN 40, HRCE 28, AP-xxx, GPS...
COORD RANGE  : X 782696..793220 (10.5 km) , Y 1241089..1279576 (38.5 km)  [UTM 43N metres]
```

**Why this is the #1 issue:** On INGUR the false-positive gate was `chain_coverage` — the
fraction of a georeferenced FMB boundary lying within 3 m of a **traced SITE DATA LINE**. It
cleanly separated true matches (58–100%) from coincidental ones (12–43%). **Erode's file has
no traced boundary** (2 polylines total), so **that proven gate does not exist here.** We
must replace it with an equally-strict discriminator built from the stone cloud alone.

## 3. Issue list

### I1 — Dense cloud ⇒ chance congruence (the over-match problem)
7,273 stones over 10.5×38.5 km (≈14× denser than INGUR, over a far bigger area). INGUR proved
that in a dense cloud **every plot finds SOME congruent subset by chance**, so inlier-count
alone has no true/false signal. Without `chain_coverage`, our discriminators are:
- **(a) Seat-locality crop** — Google village anchor + TNGIS cadastre centroid per survey# →
  crop 7,273 stones to a few-hundred-stone window around where the plot actually is. This is
  now the **primary FP control** (a plot physically cannot match a far-away look-alike). It is
  a legit M3 input (TNGIS is allowed), used as a **hint only** — it crops candidates, it never
  moves a stone or changes a length, so rule #2 is safe.
- **(b) Geometric uniqueness** — a distinctive corner N-gon at low residual (ANCHOR phase).
- **(c) Tiling** — real parcels tile; a placement overlapping a placed plot >30% is rejected
  (GROW phase).
**Open question for experts (Q1):** is (a)+(b)+(c) together as strong as INGUR's coverage
gate, or do we need an extra signal — e.g. **stone-code consistency** (do the matched surveyor
stones' codes — RBS/BS/B — agree with the FMB's corner types)?

### I2 — ACCEPT / REVIEW / NO_COVERAGE must be redefined without coverage
INGUR used coverage thresholds. Proposed replacement, all general:
- **ACCEPT** — placed at residual < ~2 m with ≥ min(5, n_corners) inlier stones, AND it tiles,
  AND fitted scale ≈ 1.
- **REVIEW** — a match exists but is sparse/borderline (fewer inliers or 2–3 m residual).
- **NO_COVERAGE** — no surveyor boundary stones fall in the plot's seat window ⇒ the surveyor
  never field-measured that plot. This is a **DATA GAP, not a quality reject**; M3 recall is
  bounded by surveyor field coverage, exactly like INGUR (12 ACCEPT / 17 NO_COVERAGE of 35).

### I3 — M3 produces no deliverable yet
`run_m3.py` currently only PRINTS placement counts. It does not write georeferenced DXFs, a
merged village DXF, a per-plot residual/disposition report, or a QA overlay PNG. This is the
biggest gap between the working matcher and a shippable M3.

### I4 — Scale-lock discipline (rule #2)
`run_m3.py` fits with `umeyama` (similarity: allows scale, gated 0.5 < s < 2.0). Rule #2 says
never change edge lengths ⇒ placement should use the **scale-locked rigid** fit (scale = 1,
rotation+translation only) that `stone_matcher.py` already implements. Keep the fitted scale
only as a **diagnostic** — a scale far from 1 signals an upstream M1 unit bug to fix in M1,
never something to absorb in M3.

### I5 — Honor the client's "base-stone rotate, 4 → 5 stones" directive
Client's method: take one stone as the base point, rotate, try to match ALL FMB stones; if it
fails, try another base; and **make the minimum-match bar 5, not 4**. M3's ANCHOR bar is 6
(already ≥5). But the GROW bar is `min(4, n)` — below 5 for large plots. Raise GROW to
`min(5, n)` so any plot with ≥5 corners must land ≥5 surveyor stones (smaller plots match all
n). Tighter FP control at a small recall cost on tiny plots — consistent with rule #2.

### I6 — GROW self-cascade risk
GROW seeds from M3's OWN anchored plots (correctly M2-free). A wrong anchor could grow a wrong
cluster. Defense: the ANCHOR bar stays high (6 inliers + resid < 3 m + uniqueness), and every
GROW placement must independently clear the same fit + tiling + scale gates — a grown plot is
never trusted just because its seed was placed.

### I7 — Does M3 actually rescue the KANDAMPALAYAM divergent plots? (yes, if surveyed)
M2's wall on KANDAMPALAYAM is FMB↔cadastre **extent divergence** (govt resurvey/renumber: same
survey# = different-sized parcel; ratios 0.03×–10.5×). The surveyor stones reflect the
**actual field boundary = the FMB's real extent**, not the stale cadastre. So M3 places a
divergent plot **correctly if the surveyor field-measured it**; if not, it is honest
NO_COVERAGE. M3 thus converts M2's "cadastre disagrees" into "surveyor measured it or didn't"
— the correct, honest framing — at survey-grade (~1.5–2 m) vs M2's cadastral ~12 m ceiling.

## 4. Proposed approach (for expert sign-off before we finalize)
1. Keep M3 strictly M2-free: Google location + TNGIS seat hint + surveyor stones only.
2. **Primary FP gate = seat-locality crop** (replaces `chain_coverage`). Validate on the real
   residual distribution that it separates true from chance matches.
3. Switch placement to **scale-locked rigid** (rule #2); keep fitted scale as a diagnostic.
4. Raise GROW bar to `min(5, n)` (client directive).
5. Build the deliverable: georeferenced DXFs + merged village DXF + per-plot
   residual/disposition report + overlay PNG (M1 rings on the stone cloud).
6. **Honest dispositions**, never inflated: ACCEPT (survey-grade fit + tiling), REVIEW
   (borderline), NO_COVERAGE (surveyor did not measure it).

## 5. Questions we want the experts to answer
- **Q1** With no traced boundary, is locality + geometric-uniqueness + tiling a strong enough
  false-positive gate, or should we add **stone-code consistency** (RBS/BS/B pattern agreement)
  or another cloud-only signal?
- **Q2** What residual threshold defines "survey-grade" for ACCEPT here — 2 m (the INGUR field
  gate), or tighter/looser given ~cm-level surveyor stones?
- **Q3** Is the **TNGIS cadastre seat hint** acceptable as an M3 input (crop only, never
  overrides geometry), or must M3 self-locate purely from Google + stone geometry with **no
  cadastre at all**? (The rule allows TNGIS; we want the experts' preference on purity.)
