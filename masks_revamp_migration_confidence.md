# Mask migration reliability

<!-- GENERATED FILE -- do not edit by hand.
     Regenerate with:
       tools/masks_migration_confidence.py --record HARVEST.json[.gz]
     which merges a newly checked corpus into
     masks_revamp_migration_ledger.json and rewrites this file. -->

_Last updated 2026-08-31, over 14 contributed libraries._

## Where we stand

**0 migration failures in 7932 distinct configuration shapes &rarr; the failure rate is below 0.038% (1 in 2,648) at 95% confidence.**

| | |
|---|---:|
| contributed libraries | 14 |
| harvested edits | 61332 |
| distinct configuration shapes | 7932 |
| migration failures | 0 |
| classic-GPU outliers | 330 |
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
| `akgt94` | 2026-08-31 | 2521 | 626 | 5.7.0+715~g7fde21688a-dirty |
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
| `uniform\|drawn\|parametric` | 3259 | 34297 | 0 | 14 | 0.092% (1 in 1,088) |
| `uniform\|drawn` | 3135 | 14339 | 0 | 14 | 0.096% (1 in 1,046) |
| `uniform\|parametric` | 1076 | 10528 | 0 | 14 | 0.278% (1 in 359) |
| `uniform\|raster` | 447 | 2016 | 0 | 13 | 0.668% (1 in 149) |
| `drawn\|parametric\|raster` | 14 | 19 | 0 | 1 | _too few_ |
| `uniform\|drawn\|raster` | 1 | 1 | 0 | 1 | _too few_ |
| `uniform\|drawn\|flexi` | 0 | 47 | 0 | 2 | _too few_ |
| `uniform\|flexi` | 0 | 85 | 0 | 3 | _too few_ |

## By form type

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `group` | 7150 | 27306 | 0 | 14 | 0.042% (1 in 2,387) |
| `path` | 4078 | 14678 | 0 | 14 | 0.073% (1 in 1,361) |
| `brush` | 2697 | 7594 | 0 | 13 | 0.111% (1 in 900) |
| `group\|clone` | 2474 | 5828 | 0 | 13 | 0.121% (1 in 826) |
| `gradient` | 1994 | 5539 | 0 | 13 | 0.150% (1 in 666) |
| `ellipse` | 1704 | 5856 | 0 | 14 | 0.176% (1 in 569) |
| `circle\|clone` | 1414 | 3266 | 0 | 12 | 0.212% (1 in 472) |
| `path\|clone` | 839 | 2317 | 0 | 11 | 0.356% (1 in 280) |
| `circle` | 756 | 1513 | 0 | 14 | 0.395% (1 in 252) |
| `clone\|brush` | 537 | 823 | 0 | 12 | 0.556% (1 in 179) |
| `clone\|ellipse` | 442 | 785 | 0 | 10 | 0.675% (1 in 148) |
| `path\|non-clone` | 277 | 524 | 0 | 7 | 1.076% (1 in 92) |
| `brush\|non-clone` | 108 | 176 | 0 | 6 | 2.736% (1 in 36) |
| `circle\|non-clone` | 16 | 42 | 0 | 4 | _too few_ |
| `512` | 1 | 83 | 0 | 1 | _too few_ |
| `1024` | 0 | 41 | 0 | 1 | _too few_ |

## By mask combine

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `norm\|excl` | 6813 | 57323 | 0 | 14 | 0.044% (1 in 2,274) |
| `norm\|excl\|masks_pos` | 1035 | 3819 | 0 | 14 | 0.289% (1 in 345) |
| `norm\|incl\|masks_pos` | 42 | 104 | 0 | 10 | 6.884% (1 in 14) |
| `norm\|incl` | 29 | 57 | 0 | 8 | _too few_ |
| `inv\|excl` | 8 | 24 | 0 | 5 | _too few_ |
| `inv\|incl` | 4 | 4 | 0 | 4 | _too few_ |
| `inv\|excl\|masks_pos` | 1 | 1 | 0 | 1 | _too few_ |

## By instance

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `first instance` | 4161 | 25309 | 0 | 14 | 0.072% (1 in 1,389) |
| `second instance` | 3771 | 36023 | 0 | 14 | 0.079% (1 in 1,259) |

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
| `inv\|excl` | 8 | 5 |
| `drawn\|parametric\|raster` | 14 | 1 |
| `circle\|non-clone` | 16 | 4 |
| `norm\|incl` | 29 | 8 |

## What this still cannot tell you

Contributors are the real sampling unit and there are 14 of them.
Shapes within one library stay correlated even after collapsing, so
the headline bound is optimistic as a statement about darktable users
at large. More *libraries* widen coverage far faster than more edits
from the same one.

