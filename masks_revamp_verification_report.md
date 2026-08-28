# Classic → flexi mask migration: verification report

**Branch:** `masks_revamp`  **Date:** 2026-08-28
**Purpose of this document:** hand to a reviewer to audit the verification for
correctness and for gaps. It states what was built, how it was run, what it
found, and — deliberately — where it was wrong along the way and what it still
cannot see.

---

## 1. What is being verified

`dt_masks_migrate_classic_to_flexi()` (`src/develop/masks/migrate_legacy.c`)
rewrites every module's classic mask configuration (`DEVELOP_MASK_MASK` /
`_CONDITIONAL` / `_RASTER` / `_MASK_CONDITIONAL`) into the flexi representation
(`DEVELOP_MASK_FLEXI`). `DEVELOP_BLEND_VERSION` goes 14 → 15; the migration runs
at the tail of every version branch of `dt_develop_blend_legacy_params_ext()`.

The claim under test: **a migrated edit renders the same mask as the classic
edit it came from, and that result survives a save/load.**

### Why the existing unit tests were not sufficient

There are 205 cmocka tests across 9 mask suites. Every assertion in them is
*structural* — that a parametric form was synthesized, that no classic mode bit
survived, that `MASKS_POS == INV ^ INCL`. **None of them looks at a pixel.** A
migration can satisfy all 205 and still shift a feathering guide, composite in
the wrong order, or apply opacity at the wrong level. That failure is silent:
the edit loads, the mask looks plausible, the module applies in the wrong place.

Two real product bugs found here were invisible to all 205 tests.

---

## 2. The mechanism

Four tools, ~3700 lines. All ship in the normal binary behind CLI flags.

### 2.1 `--harvest-masks FILE` — `src/develop/masks/harvest.c`

Reads a real library and writes every mask configuration to human-readable
JSON. Design constraints, all deliberate:

- Opens the library **read-only** twice over: `file:...?mode=ro` in the URI
  *and* `SQLITE_OPEN_READONLY` in the flags. Runs before `dt_database_init()`.
- Emits **no** file paths, filenames, or user-authored text (group names are
  skipped) — so a contributor can read the file before sending it.
- Hand-rolled JSON emitter rather than a serializer, so a field can only appear
  if it was typed out explicitly.
- `_group_point_stride()` mirrors `dt_masks_read_masks_history()`'s per-version
  strides (v7 refinement, v8 name, v9 group_opacity, v10 group_start).

**Auditable subtlety:** `masks_history` only writes rows when forms *change*,
and each row set is a full cumulative snapshot. The first version harvested
forms per history entry and produced 11,955 forms and 22 spurious differences.
Correct query:

```sql
SELECT formid, form, version, points, points_count, source, MAX(num)
FROM masks_history WHERE imgid = ?1 AND num <= ?2
GROUP BY formid ORDER BY formid
```

→ 27,803 forms. **A reviewer should check this reasoning holds.**

### 2.2 `--verify-masks FILE` — `src/develop/masks/verify.c`

Per edit: build a dev + real module instance + pipe/piece, render the mask,
migrate, render again, compare.

- **Both renders go through the production blend** (`dt_develop_blend_process`
  and `dt_develop_blend_process_cl`), unmodified. Nothing here computes "what
  the mask should be" — that would encode the same beliefs as the migration and
  cancel out.
- The mask is recovered by setting `pipe->store_all_raster_masks`, which makes
  the blend publish its finished mask into `piece->raster_masks`.
- Thresholds: `identical` ≤ 1e-6, `equivalent` ≤ 1/255 (invisible), else
  `different`. Rendered at ≤512px, harvested aspect preserved.
- `omp_set_num_threads(1)` — not for safety, for **reproducibility**. Across two
  full runs, 4 edits changed verdict from float reassociation alone; one by 0.1,
  which looked exactly like a real bug.

**Four renders per edit, not two.** CPU-classic, GPU-classic, CPU-migrated,
GPU-migrated. The GPU comparison is judged against the **classic CPU/GPU gap as
a baseline**, because the two blend implementations never agree bit-for-bit; the
defect condition is migration *widening* that gap, not the gap existing.

### 2.3 The probe image — `src/develop/masks/probe_image.c`

Masks are stored normalised and parametric masks are evaluated against whatever
pixels are present, so the user's photo is replaced by a generated probe (never
collected). **This is where the whole exercise can become vacuous** — two
all-zero masks compare equal however wrong the migration was.

So the probe's adequacy is *measured*, not assumed:
`src/tests/unittests/masks/test_probe_image.c`, 7 tests — full-channel coverage,
local-window coverage in the diffuse cube, hard edges with an orientation
distribution over 16 bins, multiscale texture via median HH² wavelet
coefficients over 4 octaves, determinism, scene-referred range, small-probe
coverage.

**Coverage bounds are derived from an RGB-cube sweep through darktable's own
colour maths, never from the corpus** — a threshold fitted to one person's
library would encode their habits.

The report also splits results into **live** (classic mask genuinely varies) and
**inert** (uniform): an inert comparison proves nothing and is counted
separately rather than padding the pass rate.

### 2.4 `--roundtrip-masks FILE` — `src/develop/masks/roundtrip.c`

`--verify-masks` replays entirely in memory (`history_num = -1`) and never
touches the database, so it is structurally blind to state that is right in
memory and lost on the way to disk.

Per edit: seed a scratch image with the harvested **classic** history and forms
→ read through the real `dt_dev_read_history_ext()` → simulate a mask edit →
write through the real `dt_dev_write_history_ext()` → read again → compare.

Compares **state, not pixels**, deliberately: §2.2 already establishes that a
given (blend_params, form tree) renders correctly, so the open question is only
whether that tuple survives a save — and a state diff names the field that
broke.

Plus a **positive invariant check on both loads**, because comparing load #1 to
load #2 would pass if both were wrong, and the two take genuinely different
paths (the first migrates, the second finds flexi and no-ops).

### 2.5 `gen_raster_matrix.py`

Classic raster mode is exclusive, so a raster edit is fully described by source
× invert × opacity × global refinements — a closed, enumerable space. Generates
288 edits in harvest format (no user data). Justified by mutation, not taste:
see §5.

---

## 3. How it was run

```bash
# harvest (read-only against the real library; checksum-verified unchanged after)
darktable --harvest-masks masks.json --library ~/Documents/Photos/darktable/library.db

# verification — private configdir + in-memory DB so no lock is taken
darktable --no-flexi-test-mode --configdir /tmp/verify_cfg --library :memory: \
          --verify-masks    real_masks_harvest.json
darktable --no-flexi-test-mode --configdir /tmp/verify_cfg --library :memory: \
          --roundtrip-masks real_masks_harvest.json
darktable --no-flexi-test-mode --configdir /tmp/verify_cfg --library :memory: \
          --verify-masks    raster_matrix.json

# unit suite
cd build-test && cmake --build . -j8 && ctest
```

Corpus: **2466 edits / 27,803 forms** from a real library.
Hardware: Apple M4 Pro, OpenCL available.

---

## 4. Results

| check | result |
|---|---|
| `--verify-masks`, 2466 real edits | 2246 identical, 133 equivalent (<1/255), **50 different, 0 CPU differences**, 37 skipped, 0 errors |
| worst CPU difference | **1.09603998e-05** (~350× below visibility) |
| GPU: CPU/GPU gap, migrated | **0.0058** |
| GPU: edits where migration *widened* the gap | **0** |
| `--roundtrip-masks`, 2429 edits | **2429 unchanged, 0 different, 0 errors** |
| generated raster matrix, 288 edits | 24 identical, 12 equivalent, **252 different — 0 of them CPU-driven** (worst CPU diff **exactly 0**); all 252 are classic-GPU outliers, see below |
| unit suite | **205 tests, 9 mask suites, all pass** |

The 37 skips are exactly the already-flexi edits (migration is a documented
no-op for them).

**All 50 "different" are classic-GPU being the outlier** — `dev_diff_before` >
1/255 and `dev_diff_after` ≤ 1/255, i.e. migration makes the GPU *agree* with
the CPU. `mask_mode` 9 (raster, 22), 7 (drawn+parametric, 18), 5 (10). These are
pre-existing OpenCL bugs in *classic* rendering, not regressions.

The same holds for the raster matrix, and more sharply: **all 252 "different"
satisfy that same predicate** (`dev_diff_before` > 1/255, `dev_diff_after` ≤
1/255; worst `dev_diff_after` across all 252 is `1.55e-05`), and the CPU
`max_diff` is **exactly 0 on every one of the 288**. The matrix is dense in
raster edits by construction, so it concentrates the known pre-existing OpenCL
raster defect — the CL raster branch publishes the host mask without
device-side post-processing — which is why the ratio is so much higher here
than in the real corpus. Aggregate: classic CPU-vs-GPU gap **0.385**, migrated
**1.55e-05**; edits where migration *widened* the gap: **0**.

Note this figure only became measurable after the pipe **input**-profile fix
(§6); before it, the OpenCL kernel `blendop_mask_rgb_jzczhz` returned early on
`use_work_profile == 0` without writing the mask, and the GPU comparison was
vacuous. An earlier draft of this table reported "288 identical, all live" from
a pre-fix run; that number was measuring nothing and is superseded by the above.

---

## 5. Product bugs found and fixed

| # | bug | file | how found |
|---|---|---|---|
| 1 | `IOP_FLAGS_NO_MASKS` gate blocked a **flexi** group render, collapsing a migrated parametric mask on retouch/spots to flat opacity (24 edits) | `blend.c` — new `dt_blend_may_render_group()` | CPU replay |
| 2 | An **unresolvable raster** rendered 1.0 instead of 0.0 — module applied to the whole image instead of nothing (5 edits) | `raster.c` — `_raster_unresolved()` | raster replay |
| 3 | OpenCL branch tested `mode_parametric` where the CPU tests `mode_drawn`; migration always clears `CONDITIONAL`, so the branch was **dead after migration** | `blend.c` | code audit; **reasoned, not measured** |
| 4 | `piece->drawn_mask_cache` key omitted `mask_mode` — which selects *which renderer runs* — so a migrated edit was served the classic renderer's output | `blend.c` | GPU replay |
| 5 | **Classic combine operators applied per-run instead of per-element**: `SUM`/`DIFFERENCE`/`INTERSECTION`/`EXCLUSION` under-composited (355 edits carry it, 27 mis-rendered) | `migrate_legacy.c`, `develop.c/.h`, `masks.h` | GPU replay, after #4 |

### Bug 5 in detail (the significant one)

`DT_MASKS_STATE_OP_COMBINE` **is** `UNION | INTERSECTION | DIFFERENCE | SUM |
EXCLUSION | MULTIPLY | OP_SCREEN` — the classic per-element bits *are* the flexi
between-group operators. Nothing is missing from the model. The difference is
*where* the operator is applied:

- **classic** folds sequentially, applying each member's operator once per member
- **flexi** partitions `grp->points` into maximal same-operator runs, folds each
  run by its *within-group* mode (none set = union/max), then applies the
  between-group operator **once per run**

A real 48-brush mask at 0.1 opacity: classic 0.6202, migrated 0.1723. Union is
unaffected because `max` is idempotent as well as associative.

**Fix (Option B):** `_split_nonunion_runs()` marks every non-union member as a
run start; `dt_masks_normalize_flexi_groups()` runs from `develop.c` **after**
`dt_masks_read_masks_history()` (which replaces `dev->forms` wholesale — doing
it before is discarded). **Nothing is written back**: the stored group keeps its
classic shape list, markers re-derive on every load, and reach the database only
via a mask-touching edit. This keeps the conversion reversible and the
classic-restore path open.

Three call sites are required, not one — drawn-only in `_dispatch`,
`_migrate_drawn_and_parametric`, and its `DT_COND_PASSTHROUGH` early return —
plus recursion into nested groups. **A reviewer should check no fourth path
reuses a classic `mask_id`.**

### Mutation testing

Every claim below was verified by breaking the code and watching the test fail.

| mutation | result |
|---|---|
| drop `DT_MASKS_STATE_INVERSE` in `_migrate_raster` | 144 different — exactly the inverted half of the matrix |
| skip post-processing for raster (`!uniform` → `!uniform && !raster`) | **264 of 288** generated vs **22 of 118** real — the argument for generating |
| `dt_masks_normalize_flexi_groups()` made a no-op | round-trip invariant fires |
| move that call *before* `dt_masks_read_masks_history()` | round-trip invariant fires |
| revert the `NO_MASKS` gate | 2 unit tests fire |

---

## 6. Harness bugs — read this section first

**Seven times a headline number turned out to be measuring nothing.** These are
listed because they are the most likely place a *remaining* error hides, and
because the pattern (a passing test that tests nothing) recurred in three
different disguises.

1. **Forms harvested per history entry** → 22 spurious differences. Fixed by the
   `GROUP BY formid` query (§2.1).
2. **OpenMP float reassociation** → 4 edits changed verdict between identical
   runs, one by 0.1. Fixed by single-threading.
3. **No colour profile on the replay pipe** → `make_mask` returns *leaving the
   mask untouched*, and the two sides failed **asymmetrically** (classic still
   carries `CONDITIONAL` and bails; migrated takes the early path and applies
   opacity). Manufactured a clean "migration changed this mask" on every
   parametric edit.
4. **`drawn_mask_cache` served the classic result to the migrated render**
   (product bug #4, but its *effect* here was that `before == after` was
   guaranteed by the cache). For drawn-only edits the CPU comparison was the
   classic renderer measured against itself. This invalidated an earlier
   "2466 edits, 0 differences" claim.
5. **`dev->iop` has no blend_params after a read** (they are only written when
   the stack is *popped*) → every round-trip snapshot had zero module lines, and
   two empty lists compare equal.
6. **`multi_priority > 0` history rows silently dropped**:
   `dt_ioppr_get_iop_order()` returns `INT_MAX` for a second instance absent
   from the default order and the reader `continue`s.
7. **Missing pipe *input* profile** → the OpenCL kernel's first line is
   `if(... || use_work_profile == 0) return;` — it returns **without writing the
   mask**, leaving zeros. Every scene-referred edit rendered zeros on the GPU on
   *both* sides of the migration, so they agreed and passed. This invalidated an
   earlier "91 GPU differences, 89 benign" claim; after the fix, 50 differences
   and the migrated CPU/GPU gap fell from 1.0 to 0.0058.

Items 6 and 7 both trace to the `cannot get iop-order for colorin instance 0`
log line, which was filtered out as noise for most of the work and was
load-bearing three times.

---

## 7. Known gaps — what this does NOT establish

1. **GUI behaviour is unverified.** Selection, solo, drag-and-drop, panel
   interaction. Not covered by any tool here.
2. **Bug #3 (the OpenCL `mode_drawn` fix) is reasoned, not measured.** The
   corpus cannot reach that branch. It is marked as such in the code comment and
   is deliberately excluded from the "0 differences" claim.
3. **The `identical` vs `equivalent` boundary at 1e-6 is not stable.** 13 edits
   moved from exactly 0 to ~4e-6 between two builds with the changes env-gated
   out; this is build-level float codegen. Only the **≤ 1/255** verdict is a
   supportable claim.
4. **Raster plumbing is synthesized, not replayed.** The verifier stands up a
   stand-in source module with a synthetic mask. The fetch
   (`dt_dev_get_raster_mask`) is shared code exercised identically by both
   sides, so what is compared is composition and refinement — *not* the pipe's
   raster distortion through a geometric module (crop/rotate/liquify).
5. **The round trip compares state, not pixels** (justified in §2.4, but it is
   an assumption worth checking).
6. **289 of the compared edits are "inert"** — the classic mask is uniform, so
   the comparison proves nothing. Counted separately, never folded into the pass
   rate. Live count: 2140.
7. **The 50 classic-GPU outliers are diagnosed but not fixed.** Pre-existing
   OpenCL bugs in classic rendering (the CL raster branch publishes a host mask
   that never received device-side post-processing; plus the profile asymmetry
   in §6.7). Migration incidentally improves them. **A reviewer may reasonably
   question whether they should be fixed rather than recorded.**
8. **`--roundtrip-masks` simulates the user's edit** via
   `dt_dev_add_masks_history_item_ext()`. If a real GUI edit path differs from
   that call, the round trip is not exactly what a user does.
9. **Single corpus, single machine.** One person's library, one GPU
   (Apple M4 Pro), one OpenCL implementation.

---

## 8. Files

| file | purpose |
|---|---|
| `src/develop/masks/harvest.{c,h}` | `--harvest-masks` |
| `src/develop/masks/verify.{c,h}` | `--verify-masks` (CPU + OpenCL replay) |
| `src/develop/masks/roundtrip.{c,h}` | `--roundtrip-masks` |
| `src/develop/masks/probe_image.{c,h}` | generated probe image |
| `src/develop/masks/harvest_read.h` | shared JSON→structs reconstruction |
| `src/tests/unittests/masks/test_probe_image.c` | 7 tests measuring probe adequacy |
| `src/tests/unittests/masks/gen_raster_matrix.py` | 288-edit raster coverage matrix |
| `branch_analysis_worklog.md` §34–§42 | full chronological record incl. wrong turns |

Product changes: `blend.c`, `blend.h`, `raster.c`, `migrate_legacy.c`,
`develop.c`, `develop.h`, `masks.h`, `test_flexi_migrate.c`.
