# Staging masks_revamp for upstream (darktable master)

## Context

This plan covers *staging the flexi mask diff into a sequence of manageable PRs*.

Classic masks stay fully intact and untouched through every PR
below, and we can decide to remove them later on as soon as flexi is sufficiently battle-tested.

**To be clear:** classic is subsumed by flexi, and it's not
needed anymore if and when flexi lands. However, the two can coexist for a period of time to give us a softer transition, e.g.:

* Stage 1: Default to flexi only for new masks
* Stage 2: Offer "migrate-to-flexi" UI
* Stage 3: Auto-migrate to Flexi, classic kept as a (possibly hidden) fallback
* Stage 4: Remove classic mask UI


## Staging principles

- **Foundation before feature.** Generic infra/widget/perf fixes that
  don't depend on flexi at all go first, in their own tiny PRs — fastest
  to review, lowest risk, and they de-risk the rest of the stack.
- **Engine before UI.** The new mask data model and its pixelpipe
  rendering must land, and be verifiable headlessly via
  `darktable-cli` + the existing XMP fixtures, before any GTK code shows
  up. A reviewer should be able to `cd src/tests/masking/flexi && ./run.sh`
  and see it work without a UI PR in sight.
- **Additive and dark by default at every step.** Every PR in the engine/UI
  chain is purely additive — classic mask code paths, rendering, and the
  existing `libs/masks.c` manager stay untouched and reachable until the
  UI PR that finally exposes flexi's own mode button. No PR before that
  point changes behavior for an existing user.
- **One logical capability per PR, not one file per PR.** `blend_gui.c`
  can't be split by line ranges (it's one panel's event/draw logic with
  heavy internal reuse); split it by the same chronological
  capability-increments it was originally built in (per-module list →
  group operators → parametric-as-form inline editor → refinement panel →
  clustering → polish), since each of those was independently coherent and
  testable on its own.

## Recommended PR sequence

### Batch 1: upstream fixes (in progress)

These are all latent bugs / areas for improvement found during development, that will be upstreamed independently of flexi.

**PR 1 — `exif.cc`: O(n²) → O(n) XMP tag deletion on write-out. MERGED.**
`_deleteXmpTag`'s per-element erase on a vector-backed `Exiv2::XmpData` was
quadratic in the number of `history_params[n]`/similar array entries; this
was discovered while stress-testing flexi's longer history stacks but is a
correctness-neutral, feature-neutral perf fix applicable to current
upstream master as-is. Already merged on upstream `master`; nothing left
to do here.

**PR 2 — `history.c`: include `main.masks_history` in the edit-history
hash. IN FLIGHT: [#21896](https://github.com/darktable-org/darktable/pull/21896).**
`_history_hash_compute_from_db` hashed `history` and `module_order` but
not `masks_history`, so a mask-only edit (move/resize a shape, flip a
group operator, change a form's invert bit) never changed the hash
`dt_history_hash_is_mipmap_synced` checks — lighttable thumbnails went
stale after mask-only edits *today*, on master, independent of flexi (and
the same stale hash also skipped the sidecar resync on leaving an image
in darkroom, since that's gated by the same check). Standalone, a real bug
fix on its own merits, and needed before any masks_revamp work lands
anyway since flexi edits are mask-history edits too. Open, awaiting
review.

**PR 3a — `bauhaus.c`: fix broken dragging in the popup's precise-entry
mode on GTK3. IN FLIGHT: [#21894](https://github.com/darktable-org/darktable/pull/21894).**
`GtkEventControllerMotion` never fires on `GTK_WINDOW_POPUP` windows under
GTK3 — a real, independent-of-flexi regression (`b91b228482` silently
reverted an earlier, deliberate `f098290bdf` workaround for it) discovered
while working on flexi's own whisker-widget controls, but the fix is
entirely inside `bauhaus.c`'s existing `#if GTK_CHECK_VERSION(4, 0, 0)`
split and has nothing flexi-specific about it. Open, awaiting review.
Standalone from PR 3b below — lands independently, in either order.

This may be desirable, regardless of flexi's landing:

**PR 3b — widget-layer groundwork: `dtgtk/gradientslider.{c,h}`,
`dtgtk/paint.{c,h}`, `bauhaus/bauhaus.{c,h}`.** The gradient-slider marker
redesign (circles + polarity-aware feather-wedge overlay) and the
accompanying bauhaus slider-indicator/draw tweaks are generically useful
widget polish, visible and testable on any existing slider in the app
without flexi enabled. Add the new `dtgtk_cairo_paint_masks_*`/
`eye_toggle` icon glyphs here too (pure additions, unreferenced until PR
6, harmless to land early and keep the icon set in one place). No
dependency on the mask engine or on PR 3a.

### Batch 2: engine-only PRs

**PR 4 — mask engine, headless: `develop/masks.h`, `develop/masks/{masks,group}.c`,
new `develop/masks/{parametric,raster}.c`, `develop/blend.{c,h}`,
`develop/pixelpipe_hb.{c,h}`, `develop/develop.{c,h}`, `develop/imageop.c`,
`data/darktableconfig.xml.in`'s new `default_operator` key.** This is the
actual new data model: `DEVELOP_MASK_FLEXI` as an additive mode bit,
groups-as-inferred-runs over the existing flat `points` list (no DB schema
change — `_MULTIPLY`/`_SCREEN`/etc. reuse previously-unused bits in the
existing `state` field, round-tripping as neutral on old blobs; the one
genuine group-boundary marker, `group_start`, is a dedicated appended field
rather than a borrowed bit, carried forward from older data by its own
masks-format struct-version bump — see `masks_revamp_data_model.md` for
why that field earned dedicated storage instead of another borrowed bit),
parametric-forms-as-group-members (`DT_MASKS_PARAMETRIC`), raster-as-form
(`DT_MASKS_RASTER` in the new sense), and the flexi group-fold renderer
(`_group_get_mask_roi_flexi` in `group.c`). Entirely reachable only when
`DEVELOP_MASK_FLEXI` is set on `mask_mode`, which nothing sets yet at this
point in the sequence (no UI exists to set it, and migration isn't wired
in until PR 5) — so this PR changes zero rendering for any existing user,
provably, and can be verified by running the full existing regression
suite plus a new one.

**PR 5 — migration + test suite: `develop/masks/migrate_legacy.c` (new),
`src/tests/masking/flexi/**`.** Automatic classic→flexi conversion of old
XMPs on blend-params version upgrade, now that PR 4 gives it something to
convert *to*. Land together with the `darktable-cli`-driven
regression suite (`gen_xmp.py`, `run.sh`, `verify_effect.sh`,
`expected/*.png`, `xmps/*.xmp`) that was built specifically to prove
migration is bit-identical to classic rendering — this is the PR's own
verification story, and it's the first point in the sequence where a
reviewer can see flexi actually produce pixels, still with zero UI. The
sample TIFF (`Sweep_sRGB_Linear_Half_Zip_01.tif`) has already been
downscaled from its original resolution — done, nothing left to shrink here
before proposing this upstream.

### Batch 3: UI PRs (still grouped by functional chunk)

**PR 6 — panel host infra: `gui/gtk.{c,h}` (the
`_ui_init_panel_right` addition and panel-position plumbing),
`libs/masks_flexi_host.c` (new), `libs/CMakeLists.txt` swap.** The
embedded/utility/left/right panel-placement mechanism and the new lib
module that hosts it, *without* the panel's actual content yet (an empty
or stub panel is fine here) — isolates the "can flexi live in a
repositionable panel" infrastructure question from the "what's actually in
it" question. `libs/masks.c` (the classic mask manager) is untouched here;
the CMake swap only adds `masks_flexi_host` alongside it — do **not**
delete `libs/masks.c` in this sequence at all; that deletion belongs to
the later classic-UI-removal phase, not to upstreaming flexi.

**PR 7a..7f — `develop/blend_gui.c`, split by the same chronological
capability increments it was built in**, each adding one coherent,
demoable slice of the panel built in PR 6's host, all still gated so the
existing classic mode buttons remain the default and flexi is reached via
its own new mode button (the *only* new classic-facing UI surface in this
entire sequence — everything else is additive/invisible until this
button exists):
  - **7a**: mode button + per-module mask list skeleton (rows: operator |
    invert | name | mute | solo), no DnD, no groups yet — just flexi mode
    existing and showing shapes.
  - **7b**: group composite operators (union/intersection/difference/
    sum/exclusion, "screen"), add-group button, group headers, DnD
    reorder within/across groups, merge-confirm dialog.
  - **7c**: parametric forms as first-class group members — inline
    per-form blendif editor docked under its row (this is where the
    `masks_combine_combo` and single-channel parametric UI lands).
  - **7d**: mask refinement panel, selection-driven per-group targeting,
    bypass toggle.
  - **7e**: shape clustering (same-kind runs fold into expand/collapse
    groups) + the crash-class fix (`g_idle_add`-deferred rebuilds instead
    of synchronous `_build_masks_list` from inside GTK event handlers) —
    land the deferral fix as its own reviewable safety property, not
    buried inside a feature commit.
  - **7f**: terminology/CSS/spacing polish (`data/themes/darktable.css`
    group-block spacing, mute/solo naming pass), rebuild-suppression guard
    (`bd->masks_rebuild_suppressed`) for batched-delete flash.
  Each of 7a-7f should be buildable and demoable on its own on top of the
  previous one; none of them change classic's behavior or default-visible
  UI for a user who never clicks the new flexi mode button.

## Verification per PR

- PRs 1-3b: existing test suite + manual smoke (thumbnail regen — and
  sidecar resync on leaving an image — after a mask-only edit for PR 2;
  drag inside a right-click precise-entry popover under GTK3 for PR 3a;
  visual check of any slider for PR 3b).
- PR 4-5: `src/tests/masking/flexi/run.sh` + `verify_effect.sh`, the full
  suite, must be bit-identical/PARTIAL-classified as already
  established; also re-run whatever the existing masking regression tests
  are on `master` to confirm zero classic regression.
- PR 6-7: manual GUI walkthrough per increment (open a mask-capable
  module, exercise exactly the capability that increment added), plus
  confirming classic's own mode buttons/panel are pixel-for-pixel
  unchanged in behavior.