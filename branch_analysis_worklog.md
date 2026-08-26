# branch_analysis worklog

Running log of work addressing the findings in
[branch_analysis_merged.md](branch_analysis_merged.md). Newest entry last.

Ground rules for this log: record what was changed and *why*, what was verified
and how, and what was deliberately not done. One entry per finding worked.

---

## §1 — Pixelpipe worker threads reading GTK-owned state (merged §2.1, Phase 1)

**Status**: done, verified

### What the code actually does today

The transient "refinement bypass" preview is read from `module->blend_data`
(GUI-owned, GTK-thread-mutated) by pixelpipe worker threads:

- `_group_get_mask_roi_flexi` — `group.c:1167-1171`, `1261`, `1302`
- `_flexi_global_refine_bypassed` — `blend.c:313-318`
- drawn-mask cache key via `dt_masks_refine_bypass_hash(bd)` — `blend.c:795-800`,
  `blend.h:792-805`

Writers are on the GTK thread: `_refine_bypass_toggled` (`blend_gui.c:3515-3535`)
inserts into `bd->masks_refine_bypassed`, and `dt_iop_gui_cleanup_blending`
destroys the table (`blend_gui.c:15721`).

### Findings that narrow the fix

Two of the three pieces of state read by the pipe are **dead**:

- `bd->refine_bypass_all` and `bd->refine_bypass_group` are never assigned
  anywhere in `src/` (verified by tree-wide grep). `blend_data` is
  `g_malloc0`-allocated, so both are permanently `FALSE`.
- Consequently `bd->panel_selected_group_cid` is read by the renderer only
  inside a branch gated on `refine_bypass_group`, i.e. never.
- `bd->masks_refine_bypass_all_btn` is only ever assigned `NULL`
  (`blend_gui.c:16689`, "retired").

So the only live channel is the `masks_refine_bypassed` hash table.

One subtlety: `_refine_scope_key` (`blend_gui.c:3450-3461`) can return a raw
`dt_masks_empty_group_t *` as a key for `REFINE_SCOPE_EMPTY_GROUP`, so the table
mixes small integer keys with heap pointers. An empty group has no members and
can never affect rendering, so those entries are irrelevant to the pipe — but it
does mean the snapshot must be built by *querying* the table for the keys the
renderer cares about, not by iterating it.

### Approach

Snapshot the bypass set into pipe-local data at `commit_params` time, on the
thread that owns it:

1. `dt_dev_refine_bypass_t` on `dt_dev_pixelpipe_iop_t` (a sorted `guint32`
   key array plus its count).
2. `dt_masks_refine_bypass_commit()` fills it from `blend_data` during
   `dt_iop_commit_params`, by walking the module's mask group and testing the
   global/element/group keys. `blend_data == NULL` (export, CLI, thumbnails)
   yields an empty snapshot, which is exactly today's behavior there.
3. The renderer and the cache hash read only the snapshot.
4. Named key helpers replace the open-coded `formid | 0x80000000U`.

Invalidation is already correct: `_refine_bypass_toggled` calls
`dt_dev_reprocess_all` -> `dt_dev_pipe_synch_all` -> `DT_DEV_PIPE_SYNCH` ->
`dt_dev_pixelpipe_synch_all` -> `dt_iop_commit_params`, so every toggle
re-commits the snapshot before the next render.

### What changed

- `pixelpipe_hb.h`: new `dt_dev_refine_bypass_t` (sorted `guint32` key array +
  count) and a `refine_bypass` field on `dt_dev_pixelpipe_iop_t`. Pieces are
  `calloc`ed (`pixelpipe_hb.c:528`), so it starts empty.
- `blend.h`: named key helpers `dt_masks_refine_key_element/_group` and
  `DT_MASKS_REFINE_KEY_GLOBAL` replace the open-coded `| 0x80000000U`. Removed
  the three dead fields (`refine_bypass_all`, `refine_bypass_group`,
  `masks_refine_bypass_all_btn`) and the old `dt_masks_refine_bypass_hash(bd)`
  inline, which iterated the live table.
- `blend.c`: `dt_masks_refine_bypass_commit/_cleanup/_lookup/_hash`. The commit
  builds the snapshot by *querying* the table for global + per-member element
  and group keys, so staged-group pointer keys can never be mistaken for a mask
  id. Lookup bisects the sorted array.
- `imageop.c`: `dt_iop_commit_params` takes the snapshot right after the
  `blendop_data` memcpy, and the piece hash now mixes in the snapshot's hash
  instead of reaching into `blend_data`.
- `pixelpipe_hb.c`: free the key array in `dt_dev_pixelpipe_cleanup_nodes`.
- `group.c`: `_group_get_mask_roi_flexi` reads `piece->refine_bypass` only. The
  two hash-table lookups became `dt_masks_refine_bypass_lookup` calls, which
  also fixed the two >90-column lines there.
- `blend_gui.c`: `_refine_scope_key` uses the new helpers.

Net: no code outside `blend_gui.c` touches `masks_refine_bypassed` any more.

### Verification

- `./build_dudo.sh` (Release, installs): builds, no new compiler warnings
  (checked by forcing recompilation of every touched file; only the pre-existing
  "Building without SSE2" and duplicate-library linker notes appear).
- **Flexi mask suite**: `src/tests/masking/flexi/run.sh` -> **29 / 29 OK**,
  exact pixel match against the checked-in `expected/*.png`.
- **Mask/blend integration tests, before vs after**: built the pre-change
  binary (`git stash` + `build_dudo.sh`), rendered nine cases, restored and
  rebuilt, rendered again, compared with `count-diff-pixels`:
  `0004-masks`, `0081-mask-groups`, `0087-blendif-and-or`,
  `0088-blendif-diff-excl`, `0090-mask-combine`,
  `0091-mask-combine-intersection-inverted`, `0144-masks-combine-sum`,
  `0150-detail-mask`, `0167-raster-mask` -> **0 differing pixels on all nine**.
  Rendering is unchanged.

### Things learned along the way (not defects I introduced)

- `bd->refine_bypass_all` / `refine_bypass_group` were dead on arrival: the
  merged report's §2.1 described a mechanism that was only half-wired. The
  live surface was smaller than the report assumed.
- The in-tree `build/` binaries are **not runnable** (`@rpath/libdarktable.dylib`
  resolves against an install layout), so any test run needs `./build_dudo.sh`
  first. `cmake --build build` is a compile check only, exactly as AGENTS.md says.
- The **installed binary at the start of this session was stale** relative to
  the committed source: it failed all 29 flexi scenarios by 2-308 pixels, while
  a fresh build of the same commit passes 29/29. Anything measured against that
  install would have been misattributed. Worth re-installing before trusting a
  render comparison.
- The nine integration tests differ from their checked-in `expected.png` by
  1.7k-2.1M pixels, both before and after this change. That is a pre-existing
  branch state, not a regression from this work, and `count-diff-pixels` counts
  any nonzero delta while the real harness allows dE < 2 -- so the raw counts
  overstate it. Not investigated further here.
- `src/tests/integration/run` does not work on this machine: it needs bash 4
  (`local -n`), a `/bin/python3`, and the `packaging` module. Rendering the
  cases directly is the workaround used above.
- Running `./run` once dirtied `src/tests/integration/logs/perfs*.log` (a
  submodule); reverted.

---

## §2 — Dead classic-tab editor code (merged §2.5B, Phase 1)

**Status**: done, verified

### The report was wrong about the severity, in both directions

The merged report predicted "GTK criticals on every parametric color pick" from
`blend_gui.c:1670` and `:1344`, and flagged it as unverified. Tracing the call
graph shows something different:

- `bd->channel` is **never assigned** anywhere in the tree. So
  `_blendop_blendif_update_tab`'s `&data->channel[tab]` -> `channel->param_channels[in_out]`
  and `_blendop_blendif_highlight_changed_tabs`'s `bd->channel[tab].label` are
  **NULL dereferences that segfault**, not warnings.
- But they are unreachable: their only callers were
  `blend_color_picker_apply`'s first two branches, which test
  `picker == data->colorpicker` / `== data->colorpicker_set_values`. Neither
  field is ever assigned either (only the per-row editor's `ed->` versions are),
  so both are NULL, and the sole caller
  (`color_picker_proxy.c:461`) always passes a real widget. Every pick fell
  through to `_param_row_picker_apply`.

So: not a live crash, but a segfault sitting behind a dead branch. Two *other*
sites in the same cluster turned out to be genuinely live, though, and the
report missed both:

1. **`_blendop_blendif_disp_alternative_worker` (was `blend_gui.c:1426-1435`)**
   is reached from the live per-row parametric editor: pressing `a` on a row's
   gradient slider (`_blendop_blendif_key_pressed` -> `channels[ch].altdisplay`).
   It computed `in_out` from `data->filter[1].slider` (NULL) and wrote the head
   label via `gtk_label_set_text(data->filter[in_out].head, ...)` (also NULL) --
   a GTK critical, and the "(log)" / "(zoom)" suffix that is the entire point of
   the function silently never appeared. Now resolves the row editor from the
   widget and labels `ed->filter[in_out].head` plus its compact twin.
2. **`_blendif_change_blend_colorspace`** tested
   `gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(bd->colorpicker))` on the
   same permanently-NULL fields when the blend colorspace changes. Replaced with
   `dt_iop_color_picker_get_active_cst(module) != IOP_CS_NONE`, which asks the
   picker proxy instead of naming widgets that no longer exist.

### What was removed

Functions (all provably unreachable after the two fixes above):
`_blendop_blendif_update_tab`, `_blendop_blendif_highlight_changed_tabs`,
`_update_gradient_slider_pickers`, `_blendif_commit`,
`_blendop_blendif_disp_alternative_reset`, `_blendif_colorpicker_cst`,
`_get_boost_factor`, `_blendif_scale`; plus the two dead branches of
`blend_color_picker_apply`, which is now a one-line forward to
`_param_row_picker_apply`.

Fields from `dt_iop_gui_blend_data_t`: `colorpicker`, `colorpicker_set_values`,
`channel_tabs`, `blendif_section`, `blendif_header`,
`channel_boost_factor_slider`, `boost_box`, `masks_param_op`,
`masks_param_op_box`, `masks_elements_header`, `masks_elements_box`,
`masks_elements_title`. The "kept for struct-layout stability" rationale did not
apply: `blend_data` is `g_malloc0`ed by core and every consumer is rebuilt from
the same tree, so there is no ABI to preserve.

Kept deliberately: `_blendop_blendif_disp_alternative_mag/_log` and
`_blendif_scale_ex`/`_get_boost_factor_ex` (referenced by the channel tables and
the per-row editor), `_picker_colorspace_for_channel` (live at three per-row
sites), and `bd->filter[2]` / `bd->channel` / `bd->tab`, which still have live
readers elsewhere and are a separate cleanup.

`blend_gui.c`: 16,752 -> 16,317 lines (-435). Removing the fields made the
compiler find one more dead function (`_blendif_scale`) on its own, which is a
good sign the cluster is now actually closed.

### Verification

- `./build_dudo.sh`: clean, no new warnings.
- Flexi mask suite: **29 / 29 OK**.
- Nine mask/blend integration cases re-rendered and compared against the
  pre-change baseline captured for §1: **0 differing pixels on all nine**.
- Not verified by running the GUI: the two live-path fixes (the `a`-key label
  and the colorspace-change picker re-arm) are reasoned from the call graph and
  compile clean, but nobody has pressed `a` on a parametric row in a running
  darktable to watch the label change. Worth one manual check.

---

## §1b — REGRESSION from §1: bypass toggle had no effect (found by user)

**Status**: fixed, confirmed working in the running app

Reported: disabling element refinements changed neither the mask overlay nor the
rendered image. This was my §1 regression, and it disabled the bypass feature
completely (element, group and global alike), not just element scope.

### Cause

`dt_masks_refine_bypass_commit` gated on
`module->blend_params->mask_mode & DEVELOP_MASK_FLEXI` and read
`module->blend_params->mask_id`. Both are wrong at that point in the call:

`dt_dev_pixelpipe_synch_all` (`pixelpipe_hb.c:850-859`) commits every piece's
**defaults** first, and `dt_iop_commit_blend_params` memcpys those into
`module->blend_params`. History is replayed only afterwards. Within
`dt_iop_commit_params` itself, my snapshot ran at line 2474 while
`dt_iop_commit_blend_params` -- the thing that writes `module->blend_params` --
is not called until line 2497. So on every history-replay commit,
`module->blend_params` still held the defaults: no FLEXI bit, `mask_id`
`NO_MASKID`. The gate rejected every module, the snapshot was always empty, and
the renderer therefore never bypassed anything.

The branch already knew about this trap: `_build_masks_list` carries a long
comment about `blend_params` being transiently reset by `synch_all`, and the
piece-hash block six lines below my call deliberately uses the passed-in
`blendop_params` rather than `module->blend_params`. I read neither closely
enough.

### Fix

Read `piece->blendop_data`, which the caller memcpy'd from `blendop_params` one
line earlier and which is the pipe-local truth for this piece. Comment added at
the site naming the trap, since the correct source is not obvious and the
failure is silent.

Also added one `dt_print(DT_DEBUG_MASKS, ...)` at the end of the snapshot,
reporting how many table entries applied. An empty snapshot is invisible
otherwise -- which is exactly why this survived a full session and two test
suites.

### Why the tests did not catch it

They cannot: the bypass set lives in `blend_data`, which only exists when there
is a GUI. `darktable-cli` always snapshots empty, both before and after the bug.
Every render check in §1 was necessarily blind to this. The regression was only
observable by clicking the toggle in the running app.

Re-verified after the fix (guards against a *different* regression, not this
one): flexi suite 29/29, nine integration cases 0 differing pixels vs the
pre-change baseline.

Confirmed by the user in the running app: the toggle now changes the overlay
and the rendered image.

---

## §3 — `flexi_test_mode` (merged §2.4D) — FINDING WITHDRAWN, do not act on it

**Status**: closed as invalid. No code change, and none should be made.

The review called `darktable.flexi_test_mode = TRUE` "branch-only scaffolding in
core init" and put it in Phase 1 as remove-or-invert. **That was a misreading and
the finding is withdrawn.** Confirmed with the branch author: defaulting it to
TRUE is the entire point. It is a safeguard for people testing an experimental
masks branch, most of whom launch the app by double-clicking it and never pass a
flag; without it, a branch under active development could write to a tester's
real `library.db` and overwrite XMP sidecars next to their original images, which
nothing in the app can undo. `--no-flexi-test-mode` exists for developers who
want the real database.

Markers left so this does not get "cleaned up" by a future pass:

- a `DO NOT` comment on the assignment itself in `common/darktable.c`, saying
  why it looks like scaffolding and why it is not
- `branch_analysis_merged.md` §2.4D and `branch_analysis_claude.md` §7 both
  struck through, with the roadmap and action matrix rows marked withdrawn
- a project memory, so it is known before the code is next read

The general lesson: "development-only default" is an inference about intent, not
an observation about code. This one looked exactly like leftover debug
scaffolding and was in fact a user-protection feature. Ask before filing that
class of finding as a defect.

Phase 1 is therefore complete: §1 and §2 done and verified, §3 withdrawn.

---

## Cumulative diff after §1 and §2

```
 src/develop/blend.c        | 123 +++++++++--
 src/develop/blend.h        |  92 ++++-----
 src/develop/blend_gui.c    | 494 +++++----------------------------------------
 src/develop/imageop.c      |  15 +-
 src/develop/masks/group.c  |  29 ++-
 src/develop/pixelpipe_hb.c |   1 +
 src/develop/pixelpipe_hb.h |  18 ++
 7 files changed, 239 insertions(+), 533 deletions(-)
```

`blend_gui.c` 16,752 -> 16,366 lines (plus a comment-only edit to
`common/darktable.c`, see §3). Nothing is committed; the work is left
staged in the working tree for review. Style checked against AGENTS.md: no added
line over 90 columns, no non-ASCII punctuation, no trailing whitespace.



---

## §4 — Tests for element / group / global refinement

**Status**: done, mutation-validated

Prompted by §1b shipping green: the refinement *bypass* is GUI-only and cannot
be covered headlessly, but the refinements themselves are serialized, so all
three scopes are reachable from an XMP and were previously almost untested (only
`H1`/`H2`, per-shape refinement carried through migration).

Added eight scenarios to `src/tests/masking/flexi/`, `J1`-`J8`, driven from
`gen_xmp.py` like the rest of the suite. Blur is the probe: purely spatial, no
guide image or scharr buffer, and order-sensitive, so blur-then-union and
union-then-blur are distinguishable. `pack_blend_params` gained the global
refinement fields, which it previously hardcoded to zero.

### Two things worth recording, both found by checking rather than assuming

- **`J1` (global) and `J5` (group) render pixel-identical.** Correct, not a
  bug: with one group the base group seeds the accumulator directly, so the
  group's finished sub-mask *is* the whole mask. It also means `J5` alone would
  pass even if group scope were wrongly routed through the global pass. `J7`/`J8`
  were added for that: two groups, refinement on the first only, group scope vs
  global scope, differing by ~17k pixels.
- **The first `J6` tested nothing.** It marked the head `ELEMENT` and the tail
  `GROUP`; since group scope is read off the run head, the tail's marking was
  inert. It is also not a GUI-reachable state, because setting group scope
  broadcasts one refinement over every member. Caught only because it failed to
  fail under mutation A. Redefined as group + global stacked.

### Mutation validation

A reference image generated from current behaviour proves nothing until it has
been seen to go red, so both scopes were deliberately broken:

| Mutation to `_group_get_mask_roi_flexi` | Caught by |
|---|---|
| group-scope refinement never applied | `J5`, `J6`, `J7` |
| run head's refinement read unconditionally (the historical leak) | `J2`, `J4`, `H1`, `H2` |

Each mutation left the other 30+ scenarios green, so the failures are
attributable rather than blanket. `group.c` was restored from a copy afterwards
and re-verified against the suite.

### Result

`run.sh` 37/37 (was 29), `verify_effect.sh` all classified `PARTIAL` — none of
the new cases collapses to a constant mask. README documents the scope model,
the two traps above, and the mutation table.

**Still not covered, and not coverable this way**: the bypass toggle itself
(`bd->masks_refine_bypassed` is GUI-only, so `darktable-cli` always snapshots
empty). That gap is what §1b slipped through and it needs a GUI-level test or a
manual check.

---

## §5 — Phase 2, step 1: split `_build_masks_list` (merged §2.2)

**Status**: done, code-motion-verified; needs a GUI check

`_build_masks_list` was 844 lines doing three unrelated jobs. Split into:

| | lines | role |
|---|---|---|
| `_masks_panel_reconcile` | 164 | model + panel scratch state only, no widgets |
| `_masks_panel_pack` | 597 | builds the row tree, reads only |
| `_build_masks_list` | 100 | guards, signature skip, wipe, snapshot, orchestration |

Reconcile owns what used to be buried mid-build: realizing a just-drawn shape
into its staged group, seeding the foundation group, renumbering groups, pruning
stale solo state, seeding the initial selection, and the insert hint. It returns
FALSE for "nothing to render", which is how the old `have_content` early return
is now expressed.

This is the change the merged report calls the precondition for everything else:
the panel's mutations no longer live on its own render path, so "drawing a shape
takes effect" stops meaning "a rebuild happened to run".

### Verified as pure code motion

Not by reading the diff -- by comparing the statement multiset of the original
function against the three new ones (comments, blanks and braces stripped):

```
statements in original _build_masks_list : 419
statements across the three new functions: 430
LOST   : 1   -> if(!have_content)      (restructured into the two returns)
GAINED : 0   beyond the 12 expected scaffolding lines
```

`grp` is still the single pointer captured once under `history_mutex` before
reconcile and passed to both halves, exactly as the original captured it once
and used it throughout, so no re-read semantics changed.

### Verification

- builds clean, no new warnings
- flexi suite 37/37; nine integration cases 0 differing pixels vs the §1 baseline

**But note what that does not cover.** Both suites are `darktable-cli`, which
never builds the panel, so they prove rendering is intact and say nothing about
whether the panel still works. Same blind spot that let §1b ship. The statement
-multiset check above is the real evidence here, and it is static. A GUI pass
over add/delete/reorder/rename/solo/group-drag is still wanted before trusting
this.

---

## §6 — "reset mask" left the whole-mask refinement behind (user-reported)

**Status**: fixed, needs a GUI check

Reported: resetting the whole mask does not reset all refinements; the
whole-mask one survives.

**Pre-existing, not from this work** -- `_masks_reset_mask_core` is byte-identical
in HEAD. Confirmed before touching anything.

### Cause

Refinement lives in two different places, and reset only cleared one of them:

- **element and group scope** live in `dt_masks_point_group_t.refinement`, so
  they are destroyed along with the shapes -- reset handles them for free
- **global (whole-mask) scope** lives in `blend_params.{details,
  feathering_radius, feathering_guide, blur_radius, contrast, brightness}`,
  which `_masks_reset_mask_core` never touched

So after a reset the mask was gone while its whole-mask refinement stayed
applied, with nothing left in the panel pointing at it.

### Fix

The correct clear sequence already existed inside `_refine_reset_clicked`,
including the non-obvious bit: `details` crossing to zero needs a full
`dt_dev_reprocess_all`, because rebuilding the scharr-derived detail mask is not
forced by an ordinary history item. Rather than copy that into the reset path --
the exact duplication-by-comment pattern the review flagged -- it is extracted:

- `_refine_clear_global(module)` -- clears the six fields, returns whether
  `details` was non-zero so the caller knows it owes a reprocess
- `_refine_global_is_set(module)` -- so reset skips committing a history item
  when there was no global refinement to clear

`_refine_reset_clicked` now calls the helper; `_masks_reset_mask_core` calls it
too, and additionally drops the per-formid scratch that the wipe orphans
(`masks_refine_bypassed`, `masks_refine_expanded`, `masks_props_expanded`, all
keyed by ids that no longer exist) and resets the scope to global.

`_masks_reset_mask` then calls `_flexi_refine_follow_selection`, the existing
canonical resync, so the six sliders repopulate instead of continuing to display
values that were just cleared out from under them.

Note this also reaches `_flexi_layout_apply` (applying a group-layout preset),
which goes through the same core and replaces the whole mask. That is
deliberate: a preset that discards every shape leaving a stale whole-mask
refinement behind is the same inconsistency.

### Verification

Builds clean; flexi suite 37/37. As with §5 this is a GUI path the CLI never
executes, so the suite only proves rendering is unaffected. **Needs a manual
check**: set a whole-mask refinement, hit reset mask, confirm both the image and
the six sliders come back neutral.

---

## §7 — Phase 2, step 2: first file extraction (merged §2.2)

**Status**: done, verified as a pure move

### Measuring the seams first

The merged report proposes splitting `blend_gui.c` into seven files. Before
committing to those boundaries I measured what each would actually cost, since
splitting a C translation unit means every cross-boundary `static` function has
to be published in a shared internal header. Cost = symbols that must enter that
header:

| candidate block | lines | needs from rest | exports to rest | header symbols |
|---|---|---|---|---|
| group-layout presets (as bounded by its section comment) | 779 | 15 | 10 | 25 |
| **group-layout presets (its real cluster)** | **389** | **8** | **1** | **9** |
| per-row parametric editor | 2847 | 71 | 6 | 77 |
| shortcuts + init/cleanup | 1525 | 44 | 3 | 47 |
| classic blendif + blend modes | 2693 | 35 | 31 | 66, bidirectional |

Two things this changes about the plan:

- **The section comments lie about cohesion.** The block under
  `// ---- group-layout presets` runs 779 lines, but only its first 389 are
  presets; the rest are unrelated panel helpers (`_group_count`,
  `_recompute_insert_hint`, `_update_refine_sensitivity`, ...) that happen to
  sit beneath the heading. Splitting on the comments would have dragged them
  along and doubled the seam.
- **`classic blendif` is the worst first candidate, not the obvious one.** It is
  the most self-evidently "separate" subsystem by name, but its coupling is
  bidirectional (35 in, 31 out), so it should be done last, not first.

### What was extracted

The real presets cluster -> `src/develop/masks_gui_presets.c` (430 lines): group
layout capture/apply, preset load/save/delete against `data.presets`, and the
preset menu. Registered in `src/CMakeLists.txt`.

New `src/develop/blend_gui_internal.h` (106 lines) holds the seam: the nine
shared symbols plus `dt_masks_empty_group_t`, whose layout the presets code
needs. Its header comment states the rule -- a symbol lands there only because a
split separated its definition from a caller, and goes back to `static` if that
stops being true -- so the file does not silently become a dumping ground.

Nine functions lost `static` as a result. That is the unavoidable price of
splitting a TU in C and is why measuring the seam first matters: at 9 symbols
this is cheap, at 77 it would not have been.

### Verified as a pure move

Compared the extracted block against `HEAD`'s copy line by line: **366 lines
identical, 1 differing** -- the `static` removed from the single exported
function. Nothing was rewritten in transit, including eight pre-existing
over-90-column lines and one non-ASCII ellipsis in a translatable string
(`"save current layout as preset…"`, the §8 finding). Those were deliberately
left alone: reflowing them would destroy the property that makes this diff
checkable as a move. They remain filed, not fixed.

### Verification

- full build clean, no new warnings
- flexi suite 37/37
- integration renders not re-run: the extracted code is GUI-only and touches no
  pixel path, and the move is byte-verified above

`blend_gui.c`: 16,452 -> 16,029 lines. A small dent, deliberately -- the point of
this step was to establish the internal-header pattern and the seam measurements
on the cheapest possible case. The next extraction (shortcuts + init/cleanup, 47
symbols, or the parametric row editor, 77) is now a mechanical repeat, and the
table above says what each costs.

---

## §9 -- Phase 2, extraction 2: flexi panel hosting

The §7 entry ended by naming "shortcuts + init/cleanup, 47 symbols" as the next
candidate. Re-measuring against the post-§7 file said that was wrong, so it was
not done.

### The suffix split does not work here

Splitting the file's tail (line 14505 to EOF, ~1,526 lines) costs **51 needs + 3
exports = 54 symbols**, not the 47 estimated in §7. The cause is structural, not
an estimation error: that tail is mostly `dt_iop_gui_init_masks` and
`dt_iop_gui_init_blending`, which `g_signal_connect` essentially *every* callback
in the file. Widget construction is the hub of a GTK panel, so any block
containing it depends on almost everything else by definition. No amount of
boundary-nudging fixes that -- the construction functions must be extracted last,
after the things they wire up have already left, or not at all.

Measured alternatives (`needs` = defined outside, used inside; `exports` =
defined inside, used outside):

| block | lines | needs | exports | total |
|---|---|---|---|---|
| shortcuts only (14505-14644) | 140 | 12 | 1 | 13 |
| **flexi panel hosting (two ranges)** | **307** | **1** | **5** | **6** |
| empty (staged) groups | 583 | 23 | 16 | 39 |
| blend-modes region | 888 | 27 | 23 | 50 |
| per-row parametric editor | 2,847 | 92 | 10 | 102 |

### What was extracted

Flexi panel hosting -> `src/develop/masks_gui_panel_host.c` (356 lines). The
cluster was **not contiguous**: 15,347-15,552 (`_masks_flexi_host_reconfigure`,
`_mask_mode_label`, `_flexi_inline_collapse_clicked`, `_masks_flexi_release`,
`_masks_flexi_relocate`) plus 1,951-2,048 (`_masks_panel_position_activate`,
`_add_masks_panel_position_menu`), some 13,000 lines apart. They are one concern
regardless of where they sat: *which of the three homes* --- embedded in the
module's expander, the `masks_flexi_host` utility lib, or the separate grid panel
--- `bd->relocatable_box` currently occupies, and the menu the user picks that
from. The new file builds no panel content of its own.

Seam: **1 in** (`_reparent_into`), **5 out**. Six symbols for 307 lines is the
best ratio of anything measured so far, better even than the presets cluster.

Also removed the three stale forward declarations at the old lines 1,000-1,002,
which existed only because the definitions sat 14,000 lines below their callers.
That distance was itself the evidence the block did not belong there.

`_masks_flexi_host_reconfigure` **went back to `static`**. Its only out-of-block
caller (`_masks_panel_position_activate`) moved along with it, so it no longer
crosses a file boundary. This is the internal header's own stated rule applied
the first time it came up, rather than letting the header accumulate symbols
that no longer need to be in it.

### Verified as a pure move

Diffed the moved code against `HEAD`'s copy: **271 of 275 non-blank lines
identical, 4 differing** -- exactly the four `static` removals. Nothing was
rewritten in transit.

### Verification

- full build clean
- flexi suite 37/37
- integration renders not re-run, for the same reason as §7: GUI-only code, no
  pixel path touched, move byte-verified

`blend_gui.c`: 16,029 -> 15,720 lines.

### Where this leaves Phase 2

Both remaining file-split candidates are now expensive (39, 50, 102), and the
init/construction functions cannot move until later. Further splitting has
reached diminishing returns; the remaining Phase 2 item -- moving localized row
updates off the full-rebuild path -- is a genuine behavior change with real
regression risk and **no headless test coverage**, since `darktable-cli` never
builds the panel. That one needs the app exercised by hand between steps.

---

## §10 -- Phase 2, final item: localized row updates off the rebuild path

The roadmap item read "move localized row updates off the full-rebuild path
(in-place widget updates using the existing `masks_row_map`)". Auditing the
actual state of the branch first changed what that meant: **most of it was
already done**, and the finding as written is largely stale.

Already in place before this entry:

- `_build_masks_list` opens with a **reconcile-by-skip hash**
  (`_masks_list_signature`), so the ~25 defensive rebuild requests already
  collapse to a cheap compare when nothing the tree is built from moved.
- **Opacity** (`_props_row_apply`) never rebuilt: it refreshes the low-opacity
  badges in place on every drag tick, with a comment saying exactly that.
- **Selection** (`_update_row_selection`) never rebuilt, from either the canvas
  or a panel click.
- **Element solo** and **group solo** both used `_refresh_all_shape_rows`.

So the remaining work was not "build an in-place path" -- it was **finishing an
in-place path that had been applied unevenly**. Three of the four things fixed
below are inconsistencies, not missing machinery.

### 1. `_toggle_element_disable` rebuilt the whole list for one bit

The clearest case. Compare two sibling gestures:

| gesture | scope of change | path taken |
|---|---|---|
| `_toggle_solo_form` | `HIDDEN` on **every** point | in-place refresh |
| `_invert_element` | `INVERSE` on **one** point | in-place refresh |
| `_toggle_element_disable` | `DISABLE` on **one** point | **full rebuild** |

Backwards: the broadest gesture took the cheap path and the narrowest took the
expensive one. `_invert_element` even carries a comment explaining why it does
not rebuild ("visibly flashes the panel for what is just one bit", plus the
re-dock of an open parametric editor) -- `_toggle_element_disable` is the same
gesture and got neither the treatment nor the reasoning.

Verified before changing it that the in-place updater is a **complete**
substitute for this state: every `DT_MASKS_STATE_DISABLE` read in the file is
either inside `_update_shape_row_state` (badge, dimmed handle/name/opacity/
action icon, insensitive editors), inside `_make_shape_row`'s construction-time
block -- which is a strict *subset* of it -- or in a menu built fresh on open.
Nothing in the group headers or the pack pass branches on it.

Incidental find, not fixed: `_make_shape_row`'s construction block dims five
widgets, `_update_shape_row_state` dims six (it also dims `expand_toggle`). So a
freshly built row of a disabled element leaves its expand toggle undimmed until
something refreshes it. The **constructor** is the one that is wrong; filed for
Phase 3, where the two blocks should collapse into one call.

### 2. The in-place solo path had a hole the rebuild used to cover

`_refresh_all_shape_rows` repaints solo classes and badges but **not** the
selection. Its callers run it right after `_sync_hidden_to_form_visible`, which
drops `panel_selected_formid` when the selected element is the one that just
got hidden. So: solo element A while element B is selected -> B is deselected in
the model, but B's row keeps its selected border until some unrelated rebuild
happens to fire.

This is the characteristic failure of a *partial* move off the rebuild path: the
rebuild repainted everything, so every state an in-place path takes over has to
be re-derived explicitly, and one was missed. Fixed by settling the selection at
the end of `_refresh_all_shape_rows`, with a comment saying why it belongs there.

### 3. `_clear_soloedit_if_hidden` rebuilt on a stale premise

Its comment justified the rebuild: *"because the solo-edit toggle button's
checked state is only set at row-construction time"*. **There is no such button
any more** -- solo-edit became a check menu item built fresh each time a row's
actions menu opens, and another comment 500 lines up already says so. The
justification outlived the widget it was about. Both callers already run
`_refresh_all_shape_rows` immediately afterwards, so the rebuild was pure
duplicate work; dropped, and the stale comment replaced with what is actually
true now.

### 4. `_toggle_soloedit` deferred a rebuild it did not need

Same stale premise. Its rebuild was `g_idle_add`-deferred because the gesture is
reachable mid-dispatch from a row's own actions menu, and a synchronous rebuild
would destroy that menu's row underneath it. Correct -- *for a rebuild*.
Switching to `_refresh_all_shape_rows` removes the hazard rather than deferring
it: it mutates existing widgets and destroys nothing, which is the exact
argument `_toggle_solo_group` already makes for calling it synchronously from
the same kind of menu dispatch.

`_queue_masks_list_rebuild` call sites: 25 -> 22. The 22 that remain are all
genuine structural changes (add / delete / reorder / regroup / rename / history
reload), where a rebuild is the right answer.

### Verification -- and its limit

- full build clean, no new warnings
- flexi suite 37/37

**The suite does not cover any of this, and cannot.** All four changes are
GUI-only: `darktable-cli` never builds the masks panel, so the tests would stay
green whether these paths worked or not. Their value here is only as a
regression check that the edits did not disturb the render path -- which is a
real thing to confirm, since `_toggle_element_disable` mutates `pt->state` and
`_clear_soloedit_if_hidden` touches canvas edit mode, but it is **not** evidence
the panel behaves correctly.

What these need is hands on the running app:

1. toggle an element's disable state -- badge, dimming, and mask render follow,
   with no panel flash and no open parametric editor collapsing
2. select element B, then solo element A -- B's row must lose its selected
   border, not just its highlight
3. solo-edit from a row's actions menu -- badge moves to the right row, the menu
   does not misbehave mid-dispatch, canvas editability follows
4. solo an element whose solo-edit is active -- solo-edit clears, badge updates

Items 2 and 3 are the ones most likely to expose a mistake: 2 is the bug being
fixed, and 3 changed a deferred call into a synchronous one.

---

## §11 -- Phase 3: explicit blend params for the make_mask evaluators

First Phase 3 item. Picked deliberately for **where it sits**: the masks engine
(`parametric.c`, `blends/blendif_*.c`) is the one area of this branch with real
headless test coverage, and it is the upstream Batch 2 payload. After §10 -- four
GUI changes the suite structurally could not check -- the right next move was
work that can actually be verified.

### Two other Phase 3 items were checked first and set aside

- **"Move metadata property tables to `develop/masks/masks.c`" (the DRY item).**
  The comment above `_blend_masks_properties` says it "mirrors
  `src/libs/masks.c`'s file-local `_masks_properties` exactly ... keep the two
  in sync if either changes". **`src/libs/masks.c` does not exist on this
  branch** -- it was deleted in `d01ec1199a` ("Flexi mask panel POC"), and the
  table was copied out of it. So there is no live duplication to eliminate here;
  master still has the file, so the duplication is a *merge*-time concern, not a
  branch-time one. Also verified the surviving table covers all ten
  `DT_MASKS_PROPERTY_*` enumerators, so it is not silently short. Reduced to a
  stale comment plus a layering preference -- low value, deferred.
- **"Pre-resolve raster source module".** Real, but GUI/pipe-lifecycle shaped
  and not covered by the suite. Deferred behind the testable work.

### The change

`_parametric_get_mask_roi` evaluated a form's blendif config by **mocking the
pipe struct**: cast away the piece's `const`, point `piece->blendop_data` at a
stack-local `dt_develop_blend_params_t`, call `make_mask`, then restore the
pointer. It did that because the four `dt_develop_blendif_*_make_mask()`
functions each opened with `d = piece->blendop_data` -- their config was an
implicit, non-negotiable read off the piece.

Audited before changing it: the mock is **balanced and correct today** -- there
is no early return between the install and the restore, and nothing reads
`blendop_data` concurrently on this path. The problem is not a live bug, it is
that a shared pipe struct momentarily points into a stack frame, and correctness
rests on nobody ever adding a `return` between two statements 30 lines apart.

Fix: make the config an **explicit parameter** of all four `make_mask`
functions. Classic callers in `blend.c` pass their own `d` (which is literally
`piece->blendop_data`, so identical semantics); `parametric.c` passes `&tmp` and
the cast-plus-restore dance is deleted outright.

Chose an added parameter over the obvious alternative of `_with_params` twin
functions plus thin wrappers. Twins would have kept a second API surface whose
entire purpose is to read one field off a struct -- and the wrapper form leaves
the implicit read available to the next caller. Making every caller name its
params is what actually retires the bug class.

The const cast in `parametric.c` still stands, but only to match the non-const
`piece` parameter these functions share with the classic path; nothing mutates
the piece any more. Constifying that would have to go through
`dt_develop_blendif_init_masking_profile` as well -- shared blend infrastructure,
wider blast radius, no test benefit. Stopped there and said so in the comment
rather than half-doing it.

Also corrected the file-header comment, which still described the mocking.

### Verification -- this one is real

Behavior-preserving by construction, so it was checked that way rather than by
classification:

1. rendered all 37 scenarios + both baselines with the **pre-change** binary
   (`git stash` -> `./build_dudo.sh` -> render)
2. restored, rebuilt, rendered again
3. compared

**39/39 with 0 differing pixels.** Note the files are *not* byte-identical --
all 39 differ, including `ZBASE_module_off.png`, which this change cannot touch.
That is PNG metadata, and it is why `count-diff-pixels` is the right instrument
and `cmp` is not; a `cmp`-based check here would have reported a 39/39 "failure"
that means nothing.

This covers both affected paths: the classic parametric blend (`blend.c`, A/H/I
series) and parametric mask forms (`parametric.c`, B/C/D/E/F series).

Full build clean, flexi suite 37/37.

---

## §12 -- Bug: parametric elements survive a blend-colorspace switch

Found by hand-testing §11 (user report): start in Lab, add an "a" parametric
element, switch to RGB (display) -- the add-buttons correctly become
`g R G B H S L`, but the element stays, still labeled `a`, still drawing the Lab
a-channel green->magenta gradient.

**Pre-existing, not caused by §11.** `git show HEAD` confirms the dispatch line
at fault is identical before the refactor; §11 only changed how the params
reach `make_mask`.

### Root cause

Two sides disagree about which colorspace a parametric form is in:

- `dt_masks_point_parametric_t.colorspace` records "the colorspace the form was
  made in". It is written **once at creation** (blend_gui.c) and on legacy
  migration, and never reconciled afterwards. The whole panel renders the row
  from it -- hence the stale "a" label and Lab gradient.
- `_parametric_get_mask_roi` evaluates the form with
  `switch(saved->blend_cst)` -- the module's *current* colorspace.

The same function contradicts itself: it uses `channels_for_csp(p->colorspace)`
(the form's own) to mask off disabled sub-channels, then dispatches on the
module's `blend_cst`. Those two disagree by construction after any switch, and
the stored channel bits get reinterpreted under a different channel table.

### Why there is no local fix

Evaluating the form in its own `p->colorspace` instead would be **equally
wrong, in the other direction**: the pixel data handed to `make_mask` is
whatever `dt_develop_blend_colorspace()` says the module's current `blend_cst`
requires (RGB for both RGB modes, Lab for Lab). Reading RGB pixels through the
Lab evaluator is no better than the reverse. The renderer is not the bug -- it
has no choice. The form is genuinely stale once the colorspace changes.

Three honest options, all design calls rather than cleanups:

1. migrate forms on switch (remap where a counterpart exists, drop where not)
2. refuse the switch while parametric forms exist
3. switch, but clear the parametric forms (what the classic path already does
   to `blendif_parameters`)

Remapping is quietly lossy exactly where it matters -- Lab `a`/`b` have no
RGB-display counterpart, and those are the channels this was found with. Chose
**2** (user's call: "seems honest enough").

### Implementation

- `_module_parametric_form_count()` -- one predicate, carrying the explanation
  of *why* in its comment.
- `_blendif_change_blend_colorspace()` refuses and logs when the count is
  non-zero. This is the **authority**: the menu entries are disabled anyway, but
  a shortcut or a future caller cannot invalidate the forms either. Same
  one-underlying-function rule as the panel/canvas gesture pairs.
- The three colorspace entries and "reset to default blend colorspace" are
  disabled while locked.

**GTK constraint worth recording:** insensitive widgets receive no motion
events, so a tooltip on a disabled menu item is never shown. The explanation
therefore lives on the *section header*, which is left sensitive (and relabeled
"blend colorspace (locked)") purely so it can carry it; it has no activate
handler, so clicking it just dismisses the menu. This is the reason the obvious
"disable it and put the reason in its tooltip" does not work as written, and it
applies to the pre-existing insensitive-menu-item-plus-tooltip pattern elsewhere
in this file (e.g. the group delete entry), whose tooltips are equally invisible.

### Verification

Build clean, flexi suite 37/37 -- but that is a regression check only. The guard
is GUI-only and the suite cannot reach it. Needs hand-testing:

1. Lab module, no parametric elements -> colorspace entries enabled, switching
   works as before
2. add a parametric element -> header reads "blend colorspace (locked)", carries
   the explanation on hover, all four entries greyed
3. delete the element -> entries live again

### Still open

Whether an **already-saved edit** whose stored `blend_cst` disagrees with its
forms hits the same path on load. The guard added here only blocks the
interactive gesture; it does not detect or repair an XMP already in that state.
If such edits exist, this is a data-integrity issue rather than a UI one and
needs a migration. Not investigated yet.
