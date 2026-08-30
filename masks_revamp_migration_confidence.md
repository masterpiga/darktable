# Mask migration reliability

<!-- GENERATED FILE -- do not edit by hand.
     Regenerate with:
       tools/masks_migration_confidence.py --record HARVEST.json[.gz]
     which merges a newly checked corpus into
     masks_revamp_migration_ledger.json and rewrites this file. -->

_Last updated 2026-08-30, over 13 contributed libraries._

## Where we stand

**0 migration failures in 7511 distinct configuration shapes &rarr; the failure rate is below 0.040% (1 in 2,507) at 95% confidence.**

| | |
|---|---:|
| contributed libraries | 13 |
| harvested edits | 58811 |
| distinct configuration shapes | 7511 |
| migration failures | 0 |
| classic-GPU outliers | 290 |
| shapes proving nothing (inert/skipped) | 32 |

Classic-GPU outliers are counted separately on purpose: there the CPU
renders classic and migrated identically and only the *classic* GPU
render disagrees, which is a pre-existing OpenCL bug in classic
blending that migration exposes rather than causes.

## What was measured

The unit is a distinct configuration **shape** -- operation, mask mode,
combine flags, form-type multiset and group structure, i.e. everything
migration branches on -- not an edit. One preset applied across
hundreds of images is one thing tested, not hundreds, and counting
edits would claim several times the evidence actually gathered.
Geometry is deliberately not part of the shape: it rides along without
selecting a different code path, and including it would leave the
correlated case uncollapsed.

Intervals are one-sided Clopper-Pearson. With zero observed failures
that degenerates to the rule of three, a bound of about 3/n; with
failures observed it widens accordingly, which is the interval doing
its job rather than the measurement regressing.

## Contributed corpora

| corpus | recorded | edits | shapes | darktable |
|---|---|---:|---:|---|
| `christian_pfister` | 2026-08-29 | 7765 | 1601 | 5.7.0+683~g2f70917066-dirty |
| `dudo` | 2026-08-29 | 2466 | 824 | 5.7.0+683~g2f70917066-dirty |
| `finestructure` | 2026-08-29 | 890 | 295 | 5.7.0+687~g33bf75f33c-dirty |
| `gwbarn` | 2026-08-29 | 1803 | 926 | 5.7.0+683~g2f70917066-dirty |
| `kofa_1` | 2026-08-30 | 4754 | 574 | 5.7.0+708~g9d5dcb8dcd-dirty |
| `kofa_2` | 2026-08-30 | 419 | 135 | 5.7.0+687~g33bf75f33c-dirty |
| `leonidas` | 2026-08-29 | 1722 | 546 | 5.7.0+683~g2f70917066-dirty |
| `macchiato17` | 2026-08-29 | 545 | 121 | 5.7.0+683~g2f70917066-dirty |
| `mino` | 2026-08-29 | 934 | 246 | 5.7.0+687~g33bf75f33c-dirty |
| `pascal` | 2026-08-29 | 9341 | 1721 | 5.7.0+687~g33bf75f33c-dirty |
| `phemisters` | 2026-08-29 | 198 | 106 | 5.7.0+687~g33bf75f33c-dirty |
| `thad` | 2026-08-29 | 27693 | 1417 | 5.7.0+683~g2f70917066-dirty |
| `zisoft` | 2026-08-29 | 281 | 86 | 5.7.0+683~g2f70917066-dirty |

## By mask mode

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `uniform\|drawn` | 3103 | 14254 | 0 | 13 | 0.096% (1 in 1,036) |
| `uniform\|drawn\|parametric` | 2934 | 32109 | 0 | 13 | 0.102% (1 in 979) |
| `uniform\|parametric` | 1063 | 10464 | 0 | 13 | 0.281% (1 in 355) |
| `uniform\|raster` | 396 | 1832 | 0 | 12 | 0.754% (1 in 132) |
| `drawn\|parametric\|raster` | 14 | 19 | 0 | 1 | _too few_ |
| `uniform\|drawn\|raster` | 1 | 1 | 0 | 1 | _too few_ |
| `uniform\|drawn\|flexi` | 0 | 47 | 0 | 2 | _too few_ |
| `uniform\|flexi` | 0 | 85 | 0 | 3 | _too few_ |

## By form type

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `group` | 6750 | 25427 | 0 | 13 | 0.044% (1 in 2,253) |
| `path` | 3848 | 13654 | 0 | 13 | 0.078% (1 in 1,284) |
| `brush` | 2599 | 7388 | 0 | 12 | 0.115% (1 in 868) |
| `group\|clone` | 2415 | 5697 | 0 | 12 | 0.124% (1 in 806) |
| `gradient` | 1778 | 4752 | 0 | 12 | 0.168% (1 in 594) |
| `ellipse` | 1591 | 5499 | 0 | 13 | 0.188% (1 in 531) |
| `circle\|clone` | 1398 | 3228 | 0 | 11 | 0.214% (1 in 467) |
| `path\|clone` | 833 | 2306 | 0 | 10 | 0.359% (1 in 278) |
| `circle` | 708 | 1392 | 0 | 13 | 0.422% (1 in 236) |
| `clone\|brush` | 535 | 820 | 0 | 11 | 0.558% (1 in 179) |
| `clone\|ellipse` | 401 | 694 | 0 | 9 | 0.744% (1 in 134) |
| `path\|non-clone` | 276 | 522 | 0 | 6 | 1.080% (1 in 92) |
| `brush\|non-clone` | 108 | 176 | 0 | 6 | 2.736% (1 in 36) |
| `circle\|non-clone` | 16 | 42 | 0 | 4 | _too few_ |
| `512` | 1 | 83 | 0 | 1 | _too few_ |
| `1024` | 0 | 41 | 0 | 1 | _too few_ |

## By mask combine

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `norm\|excl` | 6456 | 55189 | 0 | 13 | 0.046% (1 in 2,155) |
| `norm\|excl\|masks_pos` | 975 | 3439 | 0 | 13 | 0.307% (1 in 325) |
| `norm\|incl\|masks_pos` | 41 | 101 | 0 | 9 | 7.046% (1 in 14) |
| `norm\|incl` | 27 | 54 | 0 | 7 | _too few_ |
| `inv\|excl` | 7 | 23 | 0 | 4 | _too few_ |
| `inv\|incl` | 4 | 4 | 0 | 4 | _too few_ |
| `inv\|excl\|masks_pos` | 1 | 1 | 0 | 1 | _too few_ |

## By instance

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `first instance` | 4049 | 24801 | 0 | 13 | 0.074% (1 in 1,352) |
| `second instance` | 3462 | 34010 | 0 | 13 | 0.086% (1 in 1,156) |

## Coverage gaps

These strata have fewer than 30 shapes, so no bound is quoted for
them -- at n=5 a zero-failure bound is still ~45%, which would read
as reassurance it has not earned. **This is the list to ask
contributors for.**

| stratum | shapes | contributors |
|---|---:|---:|
| `1024` | 0 | 1 |
| `uniform\|drawn\|flexi` | 0 | 2 |
| `uniform\|flexi` | 0 | 3 |
| `512` | 1 | 1 |
| `inv\|excl\|masks_pos` | 1 | 1 |
| `uniform\|drawn\|raster` | 1 | 1 |
| `inv\|incl` | 4 | 4 |
| `inv\|excl` | 7 | 4 |
| `drawn\|parametric\|raster` | 14 | 1 |
| `circle\|non-clone` | 16 | 4 |
| `norm\|incl` | 27 | 7 |

## What this still cannot tell you

Contributors are the real sampling unit and there are 13 of them.
Shapes within one library stay correlated even after collapsing, so
the headline bound is optimistic as a statement about darktable users
at large. More *libraries* widen coverage far faster than more edits
from the same one.

