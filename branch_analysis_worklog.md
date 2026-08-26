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
