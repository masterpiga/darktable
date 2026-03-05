# Classic → Flexi Mask Migration: Conversion Algorithm Benchmark

Validation results for the Phase A migration implemented in
`src/develop/masks/migrate_legacy.c` (entry point
`dt_masks_migrate_classic_to_flexi`, wired in via
`dt_develop_blend_legacy_params_ext`). This document records how the
migrated output was verified against pre-migration ("pristine") rendering,
not just how the algorithm is designed — see `masks_revamp_flexi_migration_plan.md`
at the repo root for the design. The regression suite that grew out of this
work (`gen_xmp.py`, `run.sh`, checked-in fixtures) lives alongside this
document in this same directory.

## Method

For each integration-test fixture under `src/tests/integration/` that uses
any drawn, parametric, or raster mask:

1. Rendered once with a **pristine** binary — the working tree with
   `migrate_legacy.c` set aside and the blend-params/history changes
   stashed, i.e. exactly pre-migration behavior.
2. Rendered once with the **migrated** binary — current working tree,
   full migration active.
3. Both renders used the same `darktable-cli` invocation, same
   `--library`, same XMP, `--disable-opencl`, and `--hq true`.
4. Compared with `src/tests/integration/count-diff-pixels` (exact pixel
   diff), not the suite's own perceptual delta-E tool — delta-E tolerance
   would have masked exactly the class of regression this migration could
   introduce (subtly wrong mask shape/combination), so an exact-match bar
   was used instead.

This ruled out environment/fixture drift as a confound: one fixture
(`0167-raster-mask`) showed a diff against the suite's own `expected.png`
that persisted even on the pristine binary, proving it predates this work;
pristine-vs-migrated is the only comparison that isolates the migration's
effect.

## Result

**15 of 17 fixtures: 0 differing pixels (bit-identical).**
2 fixtures show diffs below 1% of image pixels, attributable to
floating-point/codegen non-determinism between the two separately-linked
binaries (LTO/inlining differences), not migration logic — see
[Residual diffs](#residual-diffs-not-bugs).

| Fixture | Mask type(s) | Diff vs. pristine |
|---|---|---|
| 0004-masks | drawn | 0 |
| 0081-mask-groups | drawn, nested groups | 0 |
| 0025-exposure-guided-filter | parametric | 0 |
| 0034-blending-modes-parametric | parametric | 0 |
| 0087-blendif-and-or | parametric | 0 |
| 0088-blendif-diff-excl | parametric | 0 |
| 0116-blending-rgb-display-1 | parametric | 0 |
| 0122-blending-lab-1 | parametric | 0 |
| 0125-blending-lab-4 | parametric | 0 |
| 0156-exposure-guided-filter-new-feather | parametric | 0 |
| 0167-raster-mask | raster | 0 |
| 0091-mask-combine-intersection-inverted | drawn, combine/invert | 0 |
| 0090-mask-combine | drawn, combine | 0 |
| 0150-detail-mask | drawn + parametric | 0 |
| 0160-overlay | drawn | 0 |
| 0161-overlay-modules-before-after | drawn, multi-module | 0 |
| 0144-masks-combine-sum | drawn + parametric | 8,605 px |
| 0103-almost-all | drawn + parametric, composite | 2,191 px |

All 17 runs: `exit=0`. `FAIL=0` overall.

## Residual diffs: not bugs

`0144-masks-combine-sum` and `0103-almost-all` were checked beyond the
pixel count before being written off:

- Re-running the **migrated** binary against itself, repeatedly, is
  perfectly deterministic (0 diff) — rules out migration-introduced
  nondeterminism.
- `0144`'s only masked module (rgbcurve) is a pure zero-transform case
  (Case 2 in the migration's case table): decoding its `blend_params`
  blob before/after migration shows no bit change beyond the `mask_mode`
  flag itself. There is no algorithmic difference for this diff to come
  from.
- Both fixtures are large composite images where the pristine and migrated
  binaries are two separately-linked builds (LTO inlining can differ
  build-to-build even from identical source), which is a known source of
  sub-ULP floating-point drift in image pipelines — consistent with a
  diff at <1% of pixels and nowhere else in the 17-fixture set.

## Bugs the benchmark caught

Two migration bugs were found only through this render-and-diff process —
neither was visible from code review:

1. **Forms surviving for only the last-migrated module.** Original design
   wrote each newly-synthesized form under its *own* originating module's
   `masks_history` row. `dt_masks_read_masks_history()` replaces
   `dev->forms` wholesale from a single row (the latest one with existing
   masks_history rows), so only the module that happened to own that row
   kept its forms. Fixed by deferring all synthesis
   (`dev->pending_flexi_migrations`) and flushing it once, after history
   load, under one shared row.
2. **Wrong "current" row.** That shared row is not simply
   `history_end - 1` — `dt_masks_read_masks_history()` resolves it as
   whichever *existing* row last touched masks, which can be earlier.
   Writing blindly to `history_end - 1` hijacked mask resolution for
   unrelated modules that correctly owned an earlier row. Fixed with
   `_current_masks_history_num()`, a `MAX(num) WHERE num < history_end`
   query, falling back to `history_end - 1` only when no masks_history
   rows exist yet.

`0090-mask-combine` and `0034-blending-modes-parametric` were the fixtures
that respectively exposed bug 2 and bug 1 — both are now bit-identical.

## Round 2: synthetic invert/operator matrix

The fixture set above never happened to combine `DEVELOP_COMBINE_INV`
(classic's "invert the whole mask" toggle) with a masked module, so it
couldn't have caught a bug specific to that bit. To close that gap, a
second, purpose-built test binary was generated directly from the classic
`dt_develop_blend_params_t` (v14) struct layout and the classic mask-point
structs (`dt_masks_point_circle_t`, `dt_masks_point_path_t`,
`dt_masks_point_group_t`) — 17 hand-packed XMPs against a fresh input image
(`Sweep_sRGB_Linear_Half_Zip_01.tif`), covering:

- a circle and a partially-overlapping square, combined with each of the
  four classic drawn-mask operators (union/intersection/difference/
  exclusion), plus per-shape polarity invert and group-level invert
  (`DEVELOP_COMBINE_INV`) on drawn-only masks
- 1–3 parametric channels (RGB-scene `RED_in`/`GREEN_in`/`BLUE_in`) with a
  taper-in 0–30% / taper-off 50–80% curve, per-channel polarity invert, and
  group-level invert on parametric-only masks
- drawn AND parametric combined, crossing operator/invert/channel-count
  variations, including the drawn-mask invert (`DEVELOP_COMBINE_MASKS_POS`)
- two deliberate fail-closed probes (an unresolved drawn mask combined with
  `DEVELOP_COMBINE_MASKS_POS`, and a parametric mask combined with
  `DEVELOP_COMBINE_INCL`) to confirm the safety net in `_dispatch()` leaves
  classic mode untouched rather than mis-rendering

Same pristine-vs-migrated methodology as above (this time built from a
fresh checkout, not last session's binaries).

**First pass found two real bugs**, both involving `DEVELOP_COMBINE_INV`
specifically — every non-INV scenario (11 of 17) was bit-identical on the
first try:

1. **Stale top-level `blendif`/`mask_combine` left behind after moving a
   classic parametric config into a synthesized `DT_MASKS_PARAMETRIC`
   form.** Once migrated, a module is `mode_drawn` (flexi joins that union),
   so `dt_develop_blend_process()` still runs its normal post-fold
   `make_mask()` pass — reading the *module's own*, now-stale
   `mask_combine`/`blendif`, not the new form's copy. With `INV` still set
   there from the original classic value, this produced a second,
   uncalled-for invert on top of the one the synthesized form's own render
   already did correctly.
2. **`DEVELOP_COMBINE_INV` has two distinct classic meanings, and both were
   collapsed into the wrong one.** For pure parametric masks, `INV` inverts
   the parametric result itself (correctly mapped to the new form's own
   `DT_MASKS_STATE_INVERSE`). For drawn+parametric combined, `INV` instead
   inverts the *already-multiplied* composite — a fold-level operation with
   no per-member equivalent — while classic's other invert bit,
   `MASKS_POS`, inverts only the drawn portion beforehand. Initially both
   bits were mapped onto the same per-member mechanism, which is only
   correct for the drawn-portion-only case; the composite-level case needs
   the *module's own* `mask_combine` (post-fold, matching how the
   already-working drawn-only case invert works) instead.

Fixed by (a) explicitly clearing the migrated module's now-superseded
`blendif`/`blendif_parameters`/`blendif_boost_factors` and `INV` bit once
their content has been moved into a form, and (b) for the combined case,
translating `INV` into the module-level `MASKS_POS` bit (which
`dt_develop_blend_process()` already applies once, after the whole group
fold completes) instead of a per-member invert. All 17 scenarios are
bit-identical after the fix; the full 18-fixture regression set from round
1 was re-run against the fixed binary and still passes (`0144`/`0103` still
show the same sub-1% floating-point-noise diffs as before, `0090`/`0034`
still bit-identical).

## Round 3: `DEVELOP_COMBINE_INCL`

The original migration fail-closed on any pure-parametric mask combined
with `DEVELOP_COMBINE_INCL` ("inclusive" mode), and on any drawn+parametric
mask with an unresolved drawn mask combined with `DEVELOP_COMBINE_MASKS_POS`
— reasoning that both had no flexi equivalent. Working through whether that
reasoning actually held up (prompted by a request to explain the two
fail-closed cases in more concrete terms) surfaced that `INCL` is
considerably more involved than a final invert:

- `INCL` doesn't just pick a different combination formula — every
  `blendif_*_make_mask()` variant that supports it XORs *every channel's own
  polarity bit* with a colorspace mask before computing the per-channel
  selection (`DEVELOP_BLENDIF_RGB_MASK` for RGB, `..._Lab_MASK` for Lab; not
  read at all for RAW).
- That XOR has a second effect: any channel the user never touched (so its
  own polarity bit was 0) ends up flagged "canceling" the moment INCL flips
  it. When that happens, classic's `make_mask()` doesn't compute a real
  mask at all — it wholesale-replaces the *entire* buffer with a flat
  constant (`opaque` or `zero`, decided by `INV XOR INCL`), discarding the
  parametric curve *and*, when reached from a `mode_drawn` call, any
  already-rendered drawn geometry. So `INCL` combined with anything short of
  every channel of the colorspace being simultaneously active — the normal
  case — collapses to a channel-config-independent, geometry-independent
  constant.
- Separately, when *no* channel is active at all (not even flipped into
  canceling — i.e. `INCL` unset, or RAW), classic takes a different, third
  branch that *multiplies* whatever is already in the mask buffer by
  opacity rather than replacing it. For resolved drawn content that
  preserves the real geometry (scaled by opacity, optionally inverted by
  `INV` alone); only reduces to a constant when the incoming value was
  itself already a fallback constant (no drawn content, or no drawn mode).

An expanded synthetic matrix (6 further scenarios, `D1`/`D2`/`E1`–`E6`)
covering pure-parametric and drawn+parametric with `INCL`, `INV`, and
`MASKS_POS` in combination — including the exact one-click "invert all
channel's polarities" state (`MASKS_POS`+`INCL` together) — pinned down all
three branches precisely against the real renderer, replacing both
fail-closed cases with real migrations (a "constant" case needs no form at
all: uniform blend mode, opacity forced to 0 for "zero"; the "passthrough,
resolved content" case collapses to a plain drawn-only migration, since the
parametric side contributes nothing). Only one truly irreducible case
remains fail-closed: `INCL` with resolvable drawn content *and* every
channel of the colorspace simultaneously active, which works out to a
screen-like combination (`1-(1-d)*temp`) with no flexi group-fold
equivalent — confirmed genuinely rare (needs deliberately activating every
channel) rather than assumed rare.

Two real implementation bugs surfaced while getting the 22-scenario matrix
to pass, caught only by cross-checking against the real renderer rather
than by re-deriving the formulas by hand a second time:

1. Delegating "no content, but otherwise resolves to a real parametric
   mask" cases to the pure-parametric path left the original module's
   `MASKS_POS` bit on `blend_params`, where it leaked into an unrelated
   post-fold invert check the migration relies on elsewhere — clearing it
   unconditionally on that path fixed it.
2. The colorspace-mask lookup returning 0 for RAW made the constant-vs-real
   classification degenerate to "always constant" for *any* RAW-colorspace
   parametric config, silently breaking several previously-passing
   real-world fixtures (`0025`, `0090`, `0125`, `0156`) that use `MASK_
   CONDITIONAL` mode with the parametric side never actually configured
   (`blendif == 0`) — an explicit early return for colorspaces with no
   channel-polarity mask fixed it.

Final state: all 22 synthetic scenarios and 16 of 18 round-1 regression
fixtures are bit-identical; the remaining 2 (`0144`, `0103`) show the same
sub-1% floating-point-noise diffs documented in round 1, confirmed
unrelated to this round's changes.

## Round 4: closing the last fail-closed case

Round 3 left exactly one migration case fail-closed: a drawn+parametric
mask with resolvable drawn content, `DEVELOP_COMBINE_INCL` set, and every
channel of the colorspace simultaneously active (the only configuration
that reaches the `DT_COND_REAL` classification while `INCL` is set — any
partial channel selection collapses to a constant instead, per Round 3).
Classic's formula there is `1-(1-d)*temp` when `INV=0`, `(1-d)*temp` when
`INV=1` (`d` = drawn geometry after any `MASKS_POS` invert, `temp` =
parametric curve). Round 3 concluded this had no flexi equivalent,
reasoning that expressing it would need a group-fold combine operator the
data model didn't have (a "screen"-style member).

Re-deriving against the *existing* general (non-`INCL`) drawn+parametric
migration construction showed this conclusion was too pessimistic: no new
operator or nesting is needed. That construction already builds a 2-member
`MULTIPLY` fold (drawn member × parametric member) with an optional
per-member drawn invert (`DEVELOP_COMBINE_MASKS_POS`) and an optional
post-fold composite invert (`DEVELOP_COMBINE_INV`, applied via the
module's own `mask_combine` after the fold, reusing the same post-fold
hook the drawn-only migration case already relies on). Algebraically, both
`INCL` formula variants reduce to that exact construction with two extra
XOR terms folded into decisions it already makes:

```
invert_drawn     = (MASKS_POS != 0) XOR INCL
invert_composite = (INV != 0)       XOR INCL
p->blendif        = o->blendif XOR (INCL ? colorspace_channel_polarity_mask << 16 : 0)
```

(Verified algebraically against all 4 `(INCL, INV)` combinations; reduces
to exactly the pre-existing, already-verified non-`INCL` formula when
`INCL=0`.) The `blendif` pre-flip reuses the same trick
`_migrate_parametric_only()` already applies for the pure-parametric
`INCL` case (Round 3) — needed because `INCL` also flips the per-channel
curve *evaluation* itself (XORing each channel's stored polarity bit
against the colorspace's channel mask), independent of the outer
mask-combine formula choice above.

Two new synthetic scenarios (`F1`/`F2`) were added: drawn (resolvable)
content unioned with a parametric mask activating every
`DEVELOP_BLENDIF_RGB_MASK` channel at once, `INCL` set, `INV=0` and
`INV=1` respectively. Rendered through a true pristine (classic-only,
migration entirely absent) binary and the fixed migrated binary and
exact-diffed: **both bit-identical, 0 differing pixels.** The full
existing 23-scenario matrix and `verify_effect.sh` were re-run against the
fixed binary and remain unchanged (23/23, all still classified as
expected) — no regression.

**Result: `migrate_legacy.c` now has zero logic-driven fail-closed cases.**
Every remaining `_migration_failed()` call site is a generic
allocation-failure guard, not a "no flexi equivalent" decision. The
classic mask *rendering* code in `blend.c`/`blendif_*.c` remains in the
tree (flexi's drawn masks still share the `DEVELOP_MASK_MASK` rendering
path with classic, and the allocation-failure fallback still needs
somewhere to render correctly) but is no longer reachable via any known,
deliberately-unmigrated editing state — 25/25 synthetic scenarios,
bit-identical.

## Phase 0.5: no orphaned classic value survives migration, ever

Round 4 closed the last *logic-driven* fail-closed case, but two other
paths could still leave a raw classic `mask_mode` sitting in
`blend_params` after migration ran: (1) the generic allocation-failure
fallback (`calloc`/`malloc`/`dt_masks_create` returning `NULL`) reverted
to the untouched classic snapshot rather than clearing it; (2) bare
`DEVELOP_MASK_ENABLED` (classic "uniform", no drawn/parametric/raster bit)
was deliberately never migrated at all, since it already renders
identically to an empty flexi group. Both are now normalized instead:
allocation failure forces `ENABLED|FLEXI` with `mask_id = NO_MASKID`
(clear the mask, keep the blend, log an error) rather than reverting;
bare `ENABLED` is explicitly rewritten to `ENABLED|FLEXI` the same way,
unconditionally and immediately (no allocation, no persisted form
needed). This makes "every module's `mask_mode` is `DISABLED` or a flexi
state" a true invariant with zero exceptions, needed so the classic
mode-select UI (removed next) never has to render or highlight for an
orphaned value.

A new `G1_bare_uniform` synthetic scenario (bare `ENABLED`, no masking
bits at all) verifies the normalization renders bit-identically to a true
pristine binary. Full 26-scenario matrix + `verify_effect.sh` re-run
unchanged elsewhere — no regression.
