# Open problem: place EVERY FMB into one georeferenced village map, 0 false positives

## Context
We convert Tamil Nadu **FMB** (Field Measurement Book) land-survey plots into one georeferenced
village map. Each FMB is a rigid polygon (corner "stones" + boundary) in **relative metres** with
**no real-world coordinates** and an unreliable survey-number label (OCR recall on the FMB itself
is ~24%). We georeference + "club" them using the public **TNGIS cadastre** delivered as **z18
raster map tiles** (slippy-map; every pixel has a known UTM coordinate). On a tile:
- the **survey-parcel boundary** is drawn in **yellow** ink,
- the **subdivision** lines in **pink/magenta**,
- the **survey-number labels** in **red/orange** (we OCR these as a CLOSED set = the known FMBs).

Pipeline today (per plot): OCR the red label -> get its UTM point; **label-seeded flood-fill** on
the yellow boundary net -> reconstruct the parcel polygon; **rigidly** fit the FMB to that polygon
(rotation + uniform scale ~1 + translation, never warped); ACCEPT only if it passes a **shape gate**
(area-ratio 0.65-1.55, scale 0.80-1.25, corner-residual <=12 m) **and** a **seat-locality gate**
(the placed centroid must sit within `max(60, 1.6*sqrt(area/pi))` m of its own label). Real-parcels
must tile -> a global non-overlap pass demotes any overlapping ACCEPT.

**Prime directive: ZERO false positives.** A wrong placement is far worse than an un-placed plot.
Math gates decide ACCEPT; the system may only place a plot when it has independent evidence.

## Result so far (real village INGUR, 35 FMBs)
**24 ACCEPT (0 false positives, validated against an independent surveyor reference), 5 REVIEW,
6 NO_COVERAGE.** We want **ALL** FMBs placed (ACCEPT or at least a correct REVIEW), with 0 FP.

## The unplaced plots and WHY (this is the problem to solve)
1. **NO_COVERAGE — reconstruction fails (5 plots: 667, 698, 723, 724, 730).** The label IS OCR'd
   correctly, but the **yellow boundary net LEAKS** around that parcel (anti-alias gaps / a road or
   the label glyph breaking the ring) -> the flood-fill escapes to the crop border -> no closed
   parcel -> nothing to fit. How do you reliably CLOSE a fragmented raster parcel boundary, or place
   the plot WITHOUT a clean polygon, at 0 FP?
2. **REVIEW — fragmented or off-seat (5 plots: 668, 670, 1022, 1024, 1025).** A parcel reconstructs
   but the rigid fit is below the shape gate (a sub-cell, area-ratio too small) or the placement
   lands just outside the seat tolerance (the OCR label is ~50-140 m off the true centroid).
3. **Ambiguous / wrong label (the hardest, e.g. survey 699).** The cadastre's only "699" label is
   the WRONG instance: TNGIS draws survey 699's true ground as a *different* number (a survey-number
   **namespace divergence** between the FMB set and the cadastre). A wrong-but-self-consistent parcel
   passes EVERY single-parcel gate. (Here the surveyor reference itself was the error, but in general
   a duplicate/space-shifted survey number can place a plot 2 km off and look perfect.)
4. **Sparse layout defeats FMB-to-FMB jigsaw.** Our fallback is to club FMBs to EACH OTHER by shared
   edges (corner stones) and propagate a placement from a confident neighbour. But on a tower-corridor
   village the plots are **sparse** (measured: 0 plot pairs share >=4 corner stones; median nearest-
   neighbour gap 132 m) -> no shared edges to walk -> the jigsaw has nothing to anchor on.

## The question
Given (a) rigid FMB polygons with no coordinates + noisy labels, and (b) an incomplete, noisy raster
cadastre (some parcels un-closable, some labels missing/ambiguous), and (c) optionally a sparse layout
with no FMB-to-FMB adjacency: **what additional method(s) place the remaining plots with 0 false
positives?** Ideas welcome on: robust raster parcel-boundary closing; placing a plot from a reliable
label POINT + orientation prior without a clean polygon (and how to gate it 0-FP); detecting
ambiguous/duplicate survey numbers; survey-number spatial-locality interpolation; using a second
independent source (e.g. the surveyor's field DXF, GPS control points, or the rate-limited TNGIS
parcel API) as corroboration; and the right disposition policy (place-as-REVIEW vs stage-for-seed)
so that "all FMBs end up in the file" without ever asserting a wrong ACCEPT.
