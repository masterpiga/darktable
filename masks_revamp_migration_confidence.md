# Mask migration reliability

<!-- GENERATED FILE -- do not edit by hand.
     Regenerate with:
       tools/masks_migration_confidence.py --record HARVEST.json[.gz]
     which merges a newly checked corpus into
     masks_revamp_migration_ledger.json and rewrites this file. -->

_Last updated 2026-08-28, over 2 contributed libraries._

## Where we stand

**0 migration failures in 1322 distinct configuration shapes &rarr; the failure rate is below 0.226% (1 in 441) at 95% confidence.**

| | |
|---|---:|
| contributed libraries | 2 |
| harvested edits | 4188 |
| distinct configuration shapes | 1322 |
| migration failures | 0 |
| classic-GPU outliers | 70 |
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
| `dudo` | 2026-08-28 | 2466 | 824 | - |
| `leonidas` | 2026-08-28 | 1722 | 546 | 5.7.0+672~gdedea60e1d-dirty |

## By mask mode

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `uniform\|drawn\|parametric` | 1019 | 3199 | 0 | 2 | 0.294% (1 in 340) |
| `uniform\|drawn` | 140 | 624 | 0 | 2 | 2.117% (1 in 47) |
| `uniform\|raster` | 136 | 271 | 0 | 2 | 2.179% (1 in 45) |
| `uniform\|parametric` | 27 | 57 | 0 | 2 | _too few_ |
| `uniform\|flexi` | 0 | 5 | 0 | 1 | _too few_ |
| `uniform\|drawn\|flexi` | 0 | 32 | 0 | 1 | _too few_ |

## By form type

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `group` | 1225 | 3635 | 0 | 2 | 0.244% (1 in 409) |
| `path` | 993 | 2511 | 0 | 2 | 0.301% (1 in 331) |
| `gradient` | 690 | 1978 | 0 | 2 | 0.433% (1 in 230) |
| `group\|clone` | 472 | 870 | 0 | 2 | 0.633% (1 in 158) |
| `ellipse` | 336 | 840 | 0 | 2 | 0.888% (1 in 112) |
| `path\|non-clone` | 264 | 482 | 0 | 2 | 1.128% (1 in 88) |
| `path\|clone` | 228 | 442 | 0 | 2 | 1.305% (1 in 76) |
| `circle\|clone` | 194 | 348 | 0 | 2 | 1.532% (1 in 65) |
| `brush` | 144 | 285 | 0 | 2 | 2.059% (1 in 48) |
| `clone\|brush` | 69 | 112 | 0 | 2 | 4.249% (1 in 23) |
| `brush\|non-clone` | 32 | 49 | 0 | 1 | 8.937% (1 in 11) |
| `circle` | 30 | 48 | 0 | 2 | 9.503% (1 in 10) |
| `clone\|ellipse` | 13 | 22 | 0 | 1 | _too few_ |
| `circle\|non-clone` | 12 | 25 | 0 | 2 | _too few_ |

## By mask combine

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `norm\|excl` | 1117 | 3596 | 0 | 2 | 0.268% (1 in 373) |
| `norm\|excl\|masks_pos` | 196 | 576 | 0 | 2 | 1.517% (1 in 65) |
| `norm\|incl` | 6 | 12 | 0 | 1 | _too few_ |
| `norm\|incl\|masks_pos` | 2 | 3 | 0 | 1 | _too few_ |
| `inv\|excl\|masks_pos` | 1 | 1 | 0 | 1 | _too few_ |

## By instance

| | shapes | edits | failures | contributors | 95% upper bound |
|---|---:|---:|---:|---:|---|
| `second instance` | 791 | 2809 | 0 | 2 | 0.378% (1 in 264) |
| `first instance` | 531 | 1379 | 0 | 2 | 0.563% (1 in 177) |

## Coverage gaps

These strata have fewer than 30 shapes, so no bound is quoted for
them -- at n=5 a zero-failure bound is still ~45%, which would read
as reassurance it has not earned. **This is the list to ask
contributors for.**

| stratum | shapes | contributors |
|---|---:|---:|
| `uniform\|flexi` | 0 | 1 |
| `uniform\|drawn\|flexi` | 0 | 1 |
| `inv\|excl\|masks_pos` | 1 | 1 |
| `norm\|incl\|masks_pos` | 2 | 1 |
| `norm\|incl` | 6 | 1 |
| `circle\|non-clone` | 12 | 2 |
| `clone\|ellipse` | 13 | 1 |
| `uniform\|parametric` | 27 | 2 |

## What this still cannot tell you

Contributors are the real sampling unit and there are 2 of them.
Shapes within one library stay correlated even after collapsing, so
the headline bound is optimistic as a statement about darktable users
at large. More *libraries* widen coverage far faster than more edits
from the same one.

