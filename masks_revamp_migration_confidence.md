# Mask migration reliability

<!-- GENERATED FILE -- do not edit by hand.
     Regenerate with:
       tools/masks_migration_confidence.py --record HARVEST.json[.gz]
     which merges a newly checked corpus into
     masks_revamp_migration_ledger.json and rewrites this file. -->

_Last updated 2026-08-29, over 9 contributed libraries._

## Where we stand

**0 migration failures in 6785 distinct configuration shapes &rarr; the failure rate is below 0.044% (1 in 2,265) at 95% confidence.**

| | |
|---|---:|
| contributed libraries | 9 |
| harvested edits | 52550 |
| distinct configuration shapes | 6785 |
| migration failures | 0 |
| classic-GPU outliers | 263 |
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
| `mino` | 2026-08-29 | 934 | 246 | 5.7.0+687~g33bf75f33c-dirty |
| `pascal` | 2026-08-29 | 9341 | 1721 | 5.7.0+687~g33bf75f33c-dirty |
| `thad` | 2026-08-29 | 27693 | 1417 | 5.7.0+683~g2f70917066-dirty |
| `zisoft` | 2026-08-29 | 281 | 86 | 5.7.0+683~g2f70917066-dirty |

## By mask mode

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `uniform\|drawn\|parametric` | 2815 | 31615 | 0 | 9 | 0.106% (1 in 940) |
| `uniform\|drawn` | 2765 | 12372 | 0 | 9 | 0.108% (1 in 923) |
| `uniform\|parametric` | 914 | 7838 | 0 | 9 | 0.327% (1 in 305) |
| `uniform\|raster` | 291 | 687 | 0 | 8 | 1.024% (1 in 97) |
| `uniform\|drawn\|flexi` | 0 | 32 | 0 | 1 | _too few_ |
| `uniform\|flexi` | 0 | 6 | 0 | 2 | _too few_ |

## By form type

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `group` | 6141 | 23346 | 0 | 9 | 0.049% (1 in 2,050) |
| `path` | 3361 | 12006 | 0 | 9 | 0.089% (1 in 1,122) |
| `brush` | 2520 | 7256 | 0 | 8 | 0.119% (1 in 841) |
| `group\|clone` | 2301 | 5467 | 0 | 9 | 0.130% (1 in 768) |
| `gradient` | 1685 | 4529 | 0 | 8 | 0.178% (1 in 562) |
| `ellipse` | 1494 | 5245 | 0 | 9 | 0.200% (1 in 499) |
| `circle\|clone` | 1331 | 3085 | 0 | 8 | 0.225% (1 in 444) |
| `path\|clone` | 787 | 2222 | 0 | 8 | 0.380% (1 in 263) |
| `circle` | 632 | 1222 | 0 | 9 | 0.473% (1 in 211) |
| `clone\|brush` | 516 | 783 | 0 | 9 | 0.579% (1 in 172) |
| `clone\|ellipse` | 379 | 636 | 0 | 6 | 0.787% (1 in 127) |
| `path\|non-clone` | 275 | 521 | 0 | 5 | 1.083% (1 in 92) |
| `brush\|non-clone` | 107 | 160 | 0 | 5 | 2.761% (1 in 36) |
| `circle\|non-clone` | 16 | 42 | 0 | 4 | _too few_ |

## By mask combine

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `norm\|excl` | 5846 | 49243 | 0 | 9 | 0.051% (1 in 1,951) |
| `norm\|excl\|masks_pos` | 884 | 3177 | 0 | 9 | 0.338% (1 in 295) |
| `norm\|incl` | 25 | 52 | 0 | 5 | _too few_ |
| `norm\|incl\|masks_pos` | 21 | 56 | 0 | 6 | _too few_ |
| `inv\|excl` | 5 | 18 | 0 | 3 | _too few_ |
| `inv\|incl` | 3 | 3 | 0 | 3 | _too few_ |
| `inv\|excl\|masks_pos` | 1 | 1 | 0 | 1 | _too few_ |

## By instance

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `first instance` | 3569 | 21168 | 0 | 9 | 0.084% (1 in 1,191) |
| `second instance` | 3216 | 31382 | 0 | 9 | 0.093% (1 in 1,074) |

## Coverage gaps

These strata have fewer than 30 shapes, so no bound is quoted for
them -- at n=5 a zero-failure bound is still ~45%, which would read
as reassurance it has not earned. **This is the list to ask
contributors for.**

| stratum | shapes | contributors |
|---|---:|---:|
| `uniform\|drawn\|flexi` | 0 | 1 |
| `uniform\|flexi` | 0 | 2 |
| `inv\|excl\|masks_pos` | 1 | 1 |
| `inv\|incl` | 3 | 3 |
| `inv\|excl` | 5 | 3 |
| `circle\|non-clone` | 16 | 4 |
| `norm\|incl\|masks_pos` | 21 | 6 |
| `norm\|incl` | 25 | 5 |

## What this still cannot tell you

Contributors are the real sampling unit and there are 9 of them.
Shapes within one library stay correlated even after collapsing, so
the headline bound is optimistic as a statement about darktable users
at large. More *libraries* widen coverage far faster than more edits
from the same one.

