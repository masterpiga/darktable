# Mask migration reliability

<!-- GENERATED FILE -- do not edit by hand.
     Regenerate with:
       tools/masks_migration_confidence.py --record HARVEST.json[.gz]
     which merges a newly checked corpus into
     masks_revamp_migration_ledger.json and rewrites this file. -->

_Last updated 2026-08-29, over 11 contributed libraries._

## Where we stand

**0 migration failures in 7012 distinct configuration shapes &rarr; the failure rate is below 0.043% (1 in 2,341) at 95% confidence.**

| | |
|---|---:|
| contributed libraries | 11 |
| harvested edits | 53638 |
| distinct configuration shapes | 7012 |
| migration failures | 0 |
| classic-GPU outliers | 271 |
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
| `uniform\|drawn` | 2883 | 12734 | 0 | 11 | 0.104% (1 in 962) |
| `uniform\|drawn\|parametric` | 2850 | 31780 | 0 | 11 | 0.105% (1 in 951) |
| `uniform\|parametric` | 955 | 8152 | 0 | 11 | 0.313% (1 in 319) |
| `uniform\|raster` | 324 | 840 | 0 | 10 | 0.920% (1 in 108) |
| `uniform\|drawn\|flexi` | 0 | 47 | 0 | 2 | _too few_ |
| `uniform\|flexi` | 0 | 85 | 0 | 3 | _too few_ |

## By form type

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `group` | 6338 | 24051 | 0 | 11 | 0.047% (1 in 2,116) |
| `path` | 3497 | 12459 | 0 | 11 | 0.086% (1 in 1,167) |
| `brush` | 2544 | 7286 | 0 | 10 | 0.118% (1 in 849) |
| `group\|clone` | 2323 | 5513 | 0 | 10 | 0.129% (1 in 775) |
| `gradient` | 1758 | 4698 | 0 | 10 | 0.170% (1 in 587) |
| `ellipse` | 1549 | 5398 | 0 | 11 | 0.193% (1 in 517) |
| `circle\|clone` | 1352 | 3129 | 0 | 9 | 0.221% (1 in 451) |
| `path\|clone` | 787 | 2222 | 0 | 8 | 0.380% (1 in 263) |
| `circle` | 664 | 1313 | 0 | 11 | 0.450% (1 in 222) |
| `clone\|brush` | 534 | 819 | 0 | 10 | 0.559% (1 in 178) |
| `clone\|ellipse` | 399 | 676 | 0 | 7 | 0.748% (1 in 133) |
| `path\|non-clone` | 275 | 521 | 0 | 5 | 1.083% (1 in 92) |
| `brush\|non-clone` | 107 | 160 | 0 | 5 | 2.761% (1 in 36) |
| `circle\|non-clone` | 16 | 42 | 0 | 4 | _too few_ |
| `512` | 1 | 83 | 0 | 1 | _too few_ |
| `1024` | 0 | 41 | 0 | 1 | _too few_ |

## By mask combine

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `norm\|excl` | 6013 | 50179 | 0 | 11 | 0.050% (1 in 2,007) |
| `norm\|excl\|masks_pos` | 924 | 3284 | 0 | 11 | 0.324% (1 in 308) |
| `norm\|incl\|masks_pos` | 39 | 99 | 0 | 8 | 7.394% (1 in 13) |
| `norm\|incl` | 26 | 53 | 0 | 6 | _too few_ |
| `inv\|excl` | 5 | 18 | 0 | 3 | _too few_ |
| `inv\|incl` | 4 | 4 | 0 | 4 | _too few_ |
| `inv\|excl\|masks_pos` | 1 | 1 | 0 | 1 | _too few_ |

## By instance

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `first instance` | 3710 | 21798 | 0 | 11 | 0.081% (1 in 1,238) |
| `second instance` | 3302 | 31840 | 0 | 11 | 0.091% (1 in 1,102) |

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
| `inv\|incl` | 4 | 4 |
| `inv\|excl` | 5 | 3 |
| `circle\|non-clone` | 16 | 4 |
| `norm\|incl` | 26 | 6 |

## What this still cannot tell you

Contributors are the real sampling unit and there are 11 of them.
Shapes within one library stay correlated even after collapsing, so
the headline bound is optimistic as a statement about darktable users
at large. More *libraries* widen coverage far faster than more edits
from the same one.

