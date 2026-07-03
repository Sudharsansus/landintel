# KANDAMPALAYAM M2 — issue brief for expert review

_Generated 2026-07-03. Villages NASIYANUR (11/11 clean) and MOOLAKARAI (20/20 clean) are
fully solved. This document is ONLY about KANDAMPALAYAM, which lands at **11/19 confidently
placed, 8 divergent**._

## 1. The exact issue

At the **correct** village location (Google Maps anchor `11.33205, 77.62354`, verified),
**11 of 19** FMBs genuinely overlay their same-survey-number TNGIS cadastre parcel
(IoU 0.43–0.95). **8 of 19 do NOT** — the same survey number denotes a **different-extent
parcel** in the current TNGIS cadastre than in the FMB.

Divergent plots and their FMB-area ÷ cadastre-area ratio (from `divergence_report.txt`):

| survey | IoU | FMB ha | CAD ha | ratio | pattern |
|---|---|---|---|---|---|
| 9  | 0.56 | 3.89 | 2.16 | **1.80** | FMB parent > cadastre remnant (subdivision) |
| 15 | 0.37 | 0.58 | 0.41 | 1.43 | subdivision |
| 37 | 0.34 | 2.00 | 1.33 | 1.51 | subdivision |
| 45 | 0.58 | 4.61 | 2.74 | 1.68 | subdivision |
| 48 | 0.49 | 4.71 | 2.86 | 1.65 | subdivision |
| 56 | 0.25 | 0.27 | 0.33 | 0.80 | partial / extent differs |
| 35 | 0.03 | 0.10 | 3.32 | **0.03** | full renumber (number now a different parcel) |
| 55 | 0.10 | 4.16 | 0.40 | **10.48** | full renumber (number now a different parcel) |

**Root cause:** KANDAMPALAYAM's FMB survey numbering is from an **older survey generation**
than the current TNGIS cadastre. ~5 parcels were **subdivided** (FMB shows the parent's full
extent ≈1.4–1.8× the current remnant) and ~2 were fully **renumbered** (35, 55). So
survey-number → parcel is no longer 1:1 for those 8. This is a **data mismatch**, not a
software bug.

### What we RULED OUT (with evidence)
- **Anchor / location** — was the cause for MOOLAKARAI (Nominatim put it on a homonym 8 km
  away); FIXED with Google Maps. KANDAMPALAYAM's Google anchor is correct (the 11 good plots
  form one contiguous ~500 m × 1.4 km village — see `clubbed_qa.png`).
- **M1 extraction** — CORRECT. Every M1 area matches its own FMB header to ~5%
  (see any `output/KANDAMPALAYAM/m1/*_<sv>.verify.txt`, `area_vs_stated` line).
- **Matching / village-block bug** — not the cause. Best single cadastre village block =
  0.63 mean IoU / 9-of-19. A "free" best-match ignoring the number gets 0.77 / 14-of-19 BUT
  those matches **scatter across a 6 km × 6 km area** → they are coincidental same-numbered
  parcels in OTHER villages (homonyms), NOT a real village. So 11/19 is the honest cadastre
  ceiling; the 8 cannot be honestly placed on the public cadastre.

## 2. What we USED (method)

1. **Anchor** — Google Maps Geocoding API (authoritative village location).
   → `src/landintel/pipeline/m5_cadastral/geo_locate.py` (`geocode_candidates`,
   `_google_geocode_candidates`)
2. **Cadastre** — EXACT TNGIS statewide **vector** parcels (GeoParquet, EPSG:4326→UTM43),
   matched by **survey number**.
   → `src/landintel/pipeline/m5_cadastral/vector_locate.py` (`load_area_parcels`,
   `village_candidates`)
3. **Placement** — RIGID fit only: rotation + translation, **scale locked ≈1**, FMB edge
   lengths/geometry preserved exactly (client rule: never warp the FMB onto the cadastre).
   → `src/landintel/pipeline/m5_cadastral/fit.py` (`fit_plot_to_parcel`, `_rigid_from_parcel`)
   → `src/landintel/pipeline/m2_club/cadastral_seat.py` (survey#→parcel rigid seat + gates)
4. **Gates (0 false positive)** — footprint **IoU** overlay vs own parcel, seat-locality,
   ≥4 corner-stones, non-overlapping tiling.
   → `src/landintel/agents/club_agents.py` (`overlay_gate`, `TngisOverlayAgent`, `ParcelAgent`)
5. **Disposition (honest)** — ACCEPT only genuine overlay → `clubbed_village.dxf`; divergent
   → separate `clubbed_needs_survey.dxf` (surveyor worklist), never inflated as "placed".
   → `run_m2_cad.py` (disposition loop + `HONEST RESULT` line + `_write_divergence_report`)

## 3. Files to refer

| File | Purpose |
|---|---|
| `run_m2_cad.py` | M2 driver: anchor, block pick, disposition, honest reporting |
| `src/landintel/pipeline/m5_cadastral/geo_locate.py` | Geocoder (Google + Nominatim) |
| `src/landintel/pipeline/m5_cadastral/vector_locate.py` | TNGIS parquet load + candidate villages |
| `src/landintel/pipeline/m5_cadastral/fit.py` | Rigid FMB→parcel fit (scale-locked) |
| `src/landintel/pipeline/m2_club/cadastral_seat.py` | survey#→parcel seat + shape/seat gates |
| `src/landintel/agents/club_agents.py` | `overlay_gate` IoU disposition (0-FP arbiter) |
| `src/landintel/pipeline/m2_club/pipeline.py` | `club_pipeline`: corroboration + propagation |
| `src/landintel/pipeline/m2_club/relative_club.py` | shared-edge/stone corroboration (recovery path) |
| `output/KANDAMPALAYAM/m2/divergence_report.txt` | per-plot IoU + area ratio + verdict |
| `output/KANDAMPALAYAM/m2/clubbed_qa.png` | visual overlay (green=placed, orange=divergent) |

## 4. The open question for the expert

How to place the **8 divergent plots** without surveyor field data. Candidate approaches:

- **(A) Relative stone-match to neighbours** (the client's FMBS_STONES_MATCH): place each
  divergent plot at its true RELATIVE position by the corner stones it SHARES with the 11
  confidently-placed neighbours — rigid, geometry preserved, cadastre-independent. Best fit
  for subdivision cases (9, 15, 37, 45, 48) which sit adjacent to placed plots.
  Existing machinery: `relative_club.py` (propagate_from_seated / shares_edge).
- **(B) Subdivision-lineage table** (old survey# → new sub-numbers): if the client can supply
  the resurvey mapping, the renumbered ones (35, 55) resolve directly. External data.
- **(C) M3 / surveyor raw-data file** — the authoritative fix; the 8 become exact.

Our recommendation: **(A)** for the 5 subdivision-adjacent plots (biggest honest win with no
new data), leave 35 & 55 for (B)/(C). Not yet implemented — awaiting direction (do not overfit).
