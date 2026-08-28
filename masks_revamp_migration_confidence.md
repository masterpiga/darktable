# Mask migration reliability

<!-- GENERATED FILE -- do not edit by hand.
     Regenerate with:
       tools/masks_migration_confidence.py --record HARVEST.json[.gz]
     which merges a newly checked corpus into
     masks_revamp_migration_ledger.json and rewrites this file. -->

_Last updated 2026-08-28, over 6 contributed libraries._

## Where we stand

**8 migration failures in 4905 distinct configuration shapes &rarr; the failure rate is below 0.294% (1 in 340) at 95% confidence.**

| | |
|---|---:|
| contributed libraries | 6 |
| harvested edits | 41730 |
| distinct configuration shapes | 4905 |
| migration failures | 8 |
| classic-GPU outliers | 246 |
| shapes proving nothing (inert/skipped) | 2 |

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
that is the rule of three: the bound is about 3/n.

## Contributed corpora

| corpus | recorded | edits | shapes | darktable |
|---|---|---:|---:|---|
| `christian_pfister` | 2026-08-28 | 7765 | 1601 | 5.7.0+675~g2a031d9525-dirty |
| `dudo` | 2026-08-28 | 2466 | 824 | - |
| `gwbarn` | 2026-08-28 | 1803 | 926 | 5.7.0+675~g2a031d9525-dirty |
| `leonidas` | 2026-08-28 | 1722 | 546 | 5.7.0+672~gdedea60e1d-dirty |
| `thad` | 2026-08-28 | 27693 | 1417 | 5.7.0+675~g2a031d9525-dirty |
| `zisoft` | 2026-08-28 | 281 | 86 | 5.7.0+675~g2a031d9525-dirty |

## By mask mode

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `uniform\|drawn\|parametric` | 2487 | 30410 | 0 | 6 | 0.120% (1 in 830) |
| `uniform\|drawn` | 1908 | 9919 | 7 | 6 | 0.688% (1 in 145) |
| `uniform\|raster` | 266 | 582 | 1 | 6 | 1.771% (1 in 56) |
| `uniform\|parametric` | 243 | 781 | 0 | 6 | 1.225% (1 in 81) |
| `uniform\|flexi` | 1 | 6 | 0 | 2 | _too few_ |
| `uniform\|drawn\|flexi` | 0 | 32 | 0 | 1 | _too few_ |

## By form type

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `group` | 4581 | 18632 | 8 | 6 | 0.315% (1 in 317) |
| `path` | 2966 | 11093 | 8 | 6 | 0.486% (1 in 205) |
| `brush` | 1416 | 4385 | 0 | 5 | 0.211% (1 in 473) |
| `gradient` | 1388 | 3724 | 1 | 6 | 0.341% (1 in 292) |
| `ellipse` | 1288 | 4477 | 4 | 6 | 0.709% (1 in 140) |
| `group\|clone` | 1183 | 2508 | 0 | 6 | 0.253% (1 in 395) |
| `circle\|clone` | 515 | 896 | 0 | 5 | 0.580% (1 in 172) |
| `path\|clone` | 442 | 1230 | 0 | 6 | 0.675% (1 in 148) |
| `circle` | 424 | 714 | 0 | 6 | 0.704% (1 in 142) |
| `clone\|brush` | 360 | 556 | 0 | 6 | 0.829% (1 in 120) |
| `path\|non-clone` | 269 | 502 | 0 | 4 | 1.107% (1 in 90) |
| `clone\|ellipse` | 163 | 294 | 0 | 4 | 1.821% (1 in 54) |
| `brush\|non-clone` | 42 | 75 | 0 | 4 | 6.884% (1 in 14) |
| `circle\|non-clone` | 15 | 41 | 0 | 3 | _too few_ |

## By mask combine

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `norm\|excl` | 4077 | 38710 | 6 | 6 | 0.290% (1 in 344) |
| `norm\|excl\|masks_pos` | 786 | 2928 | 2 | 6 | 0.799% (1 in 125) |
| `norm\|incl` | 23 | 49 | 0 | 4 | _too few_ |
| `norm\|incl\|masks_pos` | 15 | 39 | 0 | 4 | _too few_ |
| `inv\|incl` | 2 | 2 | 0 | 2 | _too few_ |
| `inv\|excl\|masks_pos` | 1 | 1 | 0 | 1 | _too few_ |
| `inv\|excl` | 1 | 1 | 0 | 1 | _too few_ |

## By instance

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `second instance` | 2854 | 29887 | 4 | 6 | 0.320% (1 in 312) |
| `first instance` | 2051 | 11843 | 4 | 6 | 0.446% (1 in 224) |

## Coverage gaps

These strata have fewer than 30 shapes, so no bound is quoted for
them -- at n=5 a zero-failure bound is still ~45%, which would read
as reassurance it has not earned. **This is the list to ask
contributors for.**

| stratum | shapes | contributors |
|---|---:|---:|
| `uniform\|drawn\|flexi` | 0 | 1 |
| `uniform\|flexi` | 1 | 2 |
| `inv\|excl\|masks_pos` | 1 | 1 |
| `inv\|excl` | 1 | 1 |
| `inv\|incl` | 2 | 2 |
| `circle\|non-clone` | 15 | 3 |
| `norm\|incl\|masks_pos` | 15 | 4 |
| `norm\|incl` | 23 | 4 |

## What this still cannot tell you

Contributors are the real sampling unit and there are 6 of them.
Shapes within one library stay correlated even after collapsing, so
the headline bound is optimistic as a statement about darktable users
at large. More *libraries* widen coverage far faster than more edits
from the same one.

