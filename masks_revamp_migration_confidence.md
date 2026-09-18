# Mask migration reliability

<!-- GENERATED FILE -- do not edit by hand.
     Regenerate with:
       tools/masks_migration_confidence.py --record HARVEST.json[.gz]
     which merges a newly checked corpus into
     masks_revamp_migration_ledger.json and rewrites this file. -->

_Last updated 2026-09-14, over 15 contributed libraries._

## Where we stand

**0 migration failures in 8203 distinct configuration shapes &rarr; the failure rate is below 0.037% (1 in 2,738) at 95% confidence.**

| | |
|---|---:|
| contributed libraries | 15 |
| harvested edits | 63157 |
| distinct configuration shapes | 8203 |
| migration failures | 0 |
| classic-GPU outliers | 66 |
| shapes proving nothing (inert/skipped) | 30 |

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
| `akgt94` | 2026-09-14 | 2521 | 626 | 5.7.0+952~g24a00c080c-dirty |
| `benp` | 2026-09-14 | 1746 | 358 | 5.7.0+952~g24a00c080c-dirty |
| `christian_pfister` | 2026-09-14 | 7765 | 1601 | 5.7.0+952~g24a00c080c-dirty |
| `dudo` | 2026-09-14 | 2545 | 851 | 5.7.0+952~g24a00c080c-dirty |
| `finestructure` | 2026-09-14 | 890 | 295 | 5.7.0+952~g24a00c080c-dirty |
| `gwbarn` | 2026-09-14 | 1803 | 926 | 5.7.0+952~g24a00c080c-dirty |
| `kofa_1` | 2026-09-14 | 4754 | 574 | 5.7.0+952~g24a00c080c-dirty |
| `kofa_2` | 2026-09-14 | 419 | 135 | 5.7.0+952~g24a00c080c-dirty |
| `leonidas` | 2026-09-14 | 1722 | 546 | 5.7.0+952~g24a00c080c-dirty |
| `macchiato17` | 2026-09-14 | 545 | 121 | 5.7.0+952~g24a00c080c-dirty |
| `mino` | 2026-09-14 | 934 | 246 | 5.7.0+952~g24a00c080c-dirty |
| `pascal` | 2026-09-14 | 9341 | 1721 | 5.7.0+952~g24a00c080c-dirty |
| `phemisters` | 2026-09-14 | 198 | 106 | 5.7.0+952~g24a00c080c-dirty |
| `thad` | 2026-09-14 | 27693 | 1417 | 5.7.0+952~g24a00c080c-dirty |
| `zisoft` | 2026-09-14 | 281 | 86 | 5.7.0+952~g24a00c080c-dirty |

## By mask mode

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `uniform\|drawn\|parametric` | 3335 | 34528 | 0 | 15 | 0.090% (1 in 1,113) |
| `uniform\|drawn` | 3282 | 15189 | 0 | 15 | 0.091% (1 in 1,096) |
| `uniform\|parametric` | 1120 | 11303 | 0 | 15 | 0.267% (1 in 374) |
| `uniform\|raster` | 451 | 2022 | 0 | 13 | 0.662% (1 in 151) |
| `drawn\|parametric\|raster` | 14 | 19 | 0 | 1 | _too few_ |
| `uniform\|drawn\|raster` | 1 | 1 | 0 | 1 | _too few_ |
| `uniform\|drawn\|flexi` | 0 | 15 | 0 | 1 | _too few_ |
| `uniform\|flexi` | 0 | 80 | 0 | 2 | _too few_ |

## By form type

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `group` | 7402 | 28392 | 0 | 15 | 0.040% (1 in 2,471) |
| `path` | 4137 | 14943 | 0 | 15 | 0.072% (1 in 1,381) |
| `brush` | 2714 | 7649 | 0 | 14 | 0.110% (1 in 906) |
| `group\|clone` | 2523 | 5980 | 0 | 14 | 0.119% (1 in 842) |
| `gradient` | 2189 | 6346 | 0 | 14 | 0.137% (1 in 731) |
| `ellipse` | 1751 | 5969 | 0 | 15 | 0.171% (1 in 584) |
| `circle\|clone` | 1445 | 3345 | 0 | 13 | 0.207% (1 in 482) |
| `path\|clone` | 853 | 2379 | 0 | 11 | 0.351% (1 in 285) |
| `circle` | 794 | 1625 | 0 | 15 | 0.377% (1 in 265) |
| `clone\|brush` | 537 | 826 | 0 | 12 | 0.556% (1 in 179) |
| `clone\|ellipse` | 447 | 797 | 0 | 11 | 0.668% (1 in 149) |
| `path\|non-clone` | 295 | 582 | 0 | 7 | 1.010% (1 in 98) |
| `brush\|non-clone` | 112 | 184 | 0 | 6 | 2.639% (1 in 37) |
| `circle\|non-clone` | 16 | 42 | 0 | 4 | _too few_ |
| `512` | 1 | 83 | 0 | 1 | _too few_ |
| `1024` | 0 | 41 | 0 | 1 | _too few_ |

## By mask combine

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `norm\|excl` | 7015 | 58992 | 0 | 15 | 0.043% (1 in 2,342) |
| `norm\|excl\|masks_pos` | 1094 | 3956 | 0 | 15 | 0.273% (1 in 365) |
| `norm\|incl\|masks_pos` | 45 | 109 | 0 | 11 | 6.440% (1 in 15) |
| `norm\|incl` | 31 | 59 | 0 | 9 | 9.211% (1 in 10) |
| `inv\|excl` | 10 | 28 | 0 | 6 | _too few_ |
| `inv\|excl\|masks_pos` | 4 | 9 | 0 | 2 | _too few_ |
| `inv\|incl` | 4 | 4 | 0 | 4 | _too few_ |

## By instance

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `first instance` | 4315 | 26480 | 0 | 15 | 0.069% (1 in 1,440) |
| `second instance` | 3888 | 36677 | 0 | 15 | 0.077% (1 in 1,298) |

## Coverage gaps

These strata have fewer than 30 shapes, so no bound is quoted for
them -- at n=5 a zero-failure bound is still ~45%, which would read
as reassurance it has not earned. **This is the list to ask
contributors for.**

| stratum | shapes | contributors |
|---|---:|---:|
| `1024` | 0 | 1 |
| `uniform\|drawn\|flexi` | 0 | 1 |
| `uniform\|flexi` | 0 | 2 |
| `512` | 1 | 1 |
| `uniform\|drawn\|raster` | 1 | 1 |
| `inv\|excl\|masks_pos` | 4 | 2 |
| `inv\|incl` | 4 | 4 |
| `inv\|excl` | 10 | 6 |
| `drawn\|parametric\|raster` | 14 | 1 |
| `circle\|non-clone` | 16 | 4 |

## What this still cannot tell you

Contributors are the real sampling unit and there are 15 of them.
Shapes within one library stay correlated even after collapsing, so
the headline bound is optimistic as a statement about darktable users
at large. More *libraries* widen coverage far faster than more edits
from the same one.

