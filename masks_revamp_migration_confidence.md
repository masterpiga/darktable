# Mask migration reliability

<!-- GENERATED FILE -- do not edit by hand.
     Regenerate with:
       tools/masks_migration_confidence.py --record HARVEST.json[.gz]
     which merges a newly checked corpus into
     masks_revamp_migration_ledger.json and rewrites this file. -->

_Last updated 2026-08-29, over 7 contributed libraries._

## Where we stand

**0 migration failures in 4975 distinct configuration shapes &rarr; the failure rate is below 0.060% (1 in 1,661) at 95% confidence.**

| | |
|---|---:|
| contributed libraries | 7 |
| harvested edits | 42275 |
| distinct configuration shapes | 4975 |
| migration failures | 0 |
| classic-GPU outliers | 253 |
| shapes proving nothing (inert/skipped) | 3 |

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
| `gwbarn` | 2026-08-29 | 1803 | 926 | 5.7.0+683~g2f70917066-dirty |
| `leonidas` | 2026-08-29 | 1722 | 546 | 5.7.0+683~g2f70917066-dirty |
| `macchiato17` | 2026-08-29 | 545 | 121 | 5.7.0+683~g2f70917066-dirty |
| `thad` | 2026-08-29 | 27693 | 1417 | 5.7.0+683~g2f70917066-dirty |
| `zisoft` | 2026-08-29 | 281 | 86 | 5.7.0+683~g2f70917066-dirty |

## By mask mode

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `uniform\|drawn\|parametric` | 2529 | 30779 | 0 | 7 | 0.118% (1 in 844) |
| `uniform\|drawn` | 1916 | 9988 | 0 | 7 | 0.156% (1 in 640) |
| `uniform\|raster` | 285 | 675 | 0 | 7 | 1.046% (1 in 95) |
| `uniform\|parametric` | 245 | 795 | 0 | 7 | 1.215% (1 in 82) |
| `uniform\|drawn\|flexi` | 0 | 32 | 0 | 1 | _too few_ |
| `uniform\|flexi` | 0 | 6 | 0 | 2 | _too few_ |

## By form type

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `group` | 4643 | 18942 | 0 | 7 | 0.065% (1 in 1,550) |
| `path` | 3022 | 11380 | 0 | 7 | 0.099% (1 in 1,009) |
| `brush` | 1421 | 4397 | 0 | 6 | 0.211% (1 in 474) |
| `gradient` | 1388 | 3724 | 0 | 6 | 0.216% (1 in 463) |
| `ellipse` | 1302 | 4504 | 0 | 7 | 0.230% (1 in 435) |
| `group\|clone` | 1196 | 2536 | 0 | 7 | 0.250% (1 in 399) |
| `circle\|clone` | 523 | 912 | 0 | 6 | 0.571% (1 in 175) |
| `path\|clone` | 443 | 1234 | 0 | 7 | 0.674% (1 in 148) |
| `circle` | 435 | 734 | 0 | 7 | 0.686% (1 in 145) |
| `clone\|brush` | 362 | 559 | 0 | 7 | 0.824% (1 in 121) |
| `path\|non-clone` | 269 | 502 | 0 | 4 | 1.107% (1 in 90) |
| `clone\|ellipse` | 165 | 299 | 0 | 5 | 1.799% (1 in 55) |
| `brush\|non-clone` | 42 | 75 | 0 | 4 | 6.884% (1 in 14) |
| `circle\|non-clone` | 15 | 41 | 0 | 3 | _too few_ |

## By mask combine

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `norm\|excl` | 4126 | 39145 | 0 | 7 | 0.073% (1 in 1,377) |
| `norm\|excl\|masks_pos` | 806 | 3031 | 0 | 7 | 0.371% (1 in 269) |
| `norm\|incl` | 23 | 49 | 0 | 4 | _too few_ |
| `norm\|incl\|masks_pos` | 16 | 46 | 0 | 5 | _too few_ |
| `inv\|incl` | 2 | 2 | 0 | 2 | _too few_ |
| `inv\|excl` | 1 | 1 | 0 | 1 | _too few_ |
| `inv\|excl\|masks_pos` | 1 | 1 | 0 | 1 | _too few_ |

## By instance

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `second instance` | 2875 | 30123 | 0 | 7 | 0.104% (1 in 960) |
| `first instance` | 2100 | 12152 | 0 | 7 | 0.143% (1 in 701) |

## Coverage gaps

These strata have fewer than 30 shapes, so no bound is quoted for
them -- at n=5 a zero-failure bound is still ~45%, which would read
as reassurance it has not earned. **This is the list to ask
contributors for.**

| stratum | shapes | contributors |
|---|---:|---:|
| `uniform\|drawn\|flexi` | 0 | 1 |
| `uniform\|flexi` | 0 | 2 |
| `inv\|excl` | 1 | 1 |
| `inv\|excl\|masks_pos` | 1 | 1 |
| `inv\|incl` | 2 | 2 |
| `circle\|non-clone` | 15 | 3 |
| `norm\|incl\|masks_pos` | 16 | 5 |
| `norm\|incl` | 23 | 4 |

## What this still cannot tell you

Contributors are the real sampling unit and there are 7 of them.
Shapes within one library stay correlated even after collapsing, so
the headline bound is optimistic as a statement about darktable users
at large. More *libraries* widen coverage far faster than more edits
from the same one.

