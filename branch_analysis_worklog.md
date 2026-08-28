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

---

## §13 -- UX: a sole group needs no selection to add elements

User request. Adding an element required a group to be selected even when the
mask had exactly one group -- the only place an element could possibly go. Now
the add controls are disabled only on a real ambiguity: several groups, none
selected.

### One resolver, two consumers

`_update_add_target_sensitivity` (button enabled state + tooltips) and
`_recompute_insert_hint` (where the element actually lands) each derived the
target independently, from the same two fields. Teaching both about the new
sole-group case would have left two copies of the rule -- and the failure mode
is nasty and silent: a button that says it can add while the insertion path
disagrees about the destination, or vice versa.

Added `_resolve_add_target()` returning a small struct (staged group / real
group cid / valid / implicit), and made both consume it. Resolution order:
selected empty group, else selected real group, else -- when `_group_count() ==
1` -- the sole group, flagged `implicit`. The sole real run's cid is its first
point in `grp->points` order, the same convention `_build_masks_list` uses for
group headers.

This is the same rule as the panel-slider/canvas-gesture pairs: when one
operation has two triggers, both call the identical underlying function rather
than reconciling parallel formulas after the fact.

### Tooltips

Previously the hint only appeared when adding was impossible ("(select a group
first)"). Now the buttons always say where the element will go:

- selection: "(added to the selected group)"
- sole group, nothing selected: "(added to the only group)"
- disabled: "(select a group first: there is more than one, so where the new
  element goes is ambiguous)" -- says *why* it is ambiguous, not just that a
  selection is missing

Applies uniformly to the shape buttons, the parametric channel buttons, raster
add and import/reuse -- the raster and import tooltips were built inline with
their own copy of the old two-state logic and now take the same hint string.

Factored `_append_tooltip_hint()` out of `_restate_tooltip_hint()` rather than
changing the latter's signature: it has an unrelated second caller (the refine
section label) whose two-state semantics are still correct.

### Verification

Build clean, flexi suite 37/37 (regression check only -- GUI-only change).
User-tested in the app.

---

## §14 -- The colorspace mismatch can also arrive from disk

Follow-up to §12's open question: can a *saved edit* already be in the bad
state, where a parametric form's `p->colorspace` disagrees with the module's
`blend_cst`? §12 blocked the interactive gesture; that says nothing about load.

### Yes, by one path -- and no gesture is involved

`dt_iop_commit_blend_params` (imageop.c) resolves the colorspace at **load**
time when the stored value is `DEVELOP_BLEND_CS_NONE`:

```c
if(blendop_params->blend_cst == DEVELOP_BLEND_CS_NONE)
  module->blend_params->blend_cst =
    dt_develop_blend_default_module_blend_colorspace(module);
```

and that default runs through `dt_is_scene_referred()` -- **a user preference**.
For an `IOP_CS_RGB` module it yields RGB_SCENE or RGB_DISPLAY accordingly. So an
edit carrying `blend_cst == NONE` plus parametric forms resolves to a
*preference-dependent* colorspace: the same XMP renders differently on a
display-referred and a scene-referred machine, and the forms' stored channel
bits are read through whichever table the preference picked.

This block is **upstream** -- verified present in `master` at the same place, so
the resolution rule is not a branch bug. What is new is that this branch put
colorspace-bound *forms* behind it.

### How reachable, honestly

Probably not very, and I could not construct it. Forms are only created after
`blend_cst` has been resolved to a concrete value, `p->colorspace` is written
from that resolved value, and history items copy the resolved
`module->blend_params` -- so a freshly written XMP should carry a concrete
`blend_cst`. `migrate_legacy.c` likewise sets `p->colorspace` from the resolved
`blend_cst`.

The residual gap is a history item recorded *before* commit resolved the value:
that item keeps `NONE`, and the resolution then happens again -- against the
current preference -- on every subsequent load. Whether that ordering actually
occurs is unproven either way.

### What was done, and what was deliberately not

Added a `DT_DEBUG_MASKS` diagnostic in `_parametric_get_mask_roi` that fires
when `p->colorspace != saved->blend_cst`, naming both values.

**The render is deliberately unchanged.** The tempting "fix" -- skip the form,
or return a neutral mask -- would alter existing edits on a path that has not
been shown to be reachable, and "neutral" is not even well defined here: a mask
of 1.0 unioned into a group selects the whole frame, which is a large visible
change, not a safe fallback. A log line that makes the state diagnosable is the
honest trade until someone demonstrates a real edit in this condition.

### If it ever needs fixing properly

The durable fix is upstream-shaped and out of scope for this pass: stop
resolving `blend_cst` from a preference at load, and instead migrate `NONE` to a
concrete value once, at the point the edit is first read, so the file rather
than the machine decides. That is a change to shared blend behavior affecting
every module, not just masked ones.

Build clean, flexi suite 37/37 (the diagnostic is inert in the tests -- they
never produce a mismatch, which is itself the expected result).

---

## §15 -- CORRECTION to §14: the from-disk mismatch was never demonstrated

§14 claimed a stored `blend_cst == DEVELOP_BLEND_CS_NONE` could reach a masked
module and make one XMP render two ways. An upstream fix was written against
`master` on the strength of that. **Two of the supporting claims were wrong**,
and both erred in the same direction, so the conclusion has to be withdrawn.

### Wrong claim 1: "all 37 scenario XMPs carry blend_cst = NONE"

A parsing error. The script used `re.search`, taking the **first**
`blendop_params` in each file rather than the entry belonging to the masked
module. Parsed per operation:

```
colorin    mask_mode=0x0  blend_cst=0
colorout   mask_mode=0x0  blend_cst=0
gamma      mask_mode=0x0  blend_cst=0
flip       mask_mode=0x0  blend_cst=0
exposure   mask_mode=0x5  blend_cst=4   <- the only masked module
```

The masked module always carries an explicit colorspace. The zeros belong to
pipeline scaffolding with `mask_mode == 0`, where `blend_cst` is never read.

### Wrong claim 2: "proven -- the same XMP resolves to 4 or 3 by preference"

The probe lines showing 4 vs 3 were **default-params** commits
(`blendop_params == module->default_blendop_params`), which follow the workflow
preference *by design* and are not stored data. Re-probing with the source
distinguished, the stored commit is `blend_cst=4` under **both** workflows.

The earlier "0 changed pixels across both workflows" result was therefore not
evidence that the channel tables coincide -- it was simply nothing differing.

### What actually stands

Only the code fact: `dt_iop_commit_blend_params` does resolve a stored `NONE`
against `dt_is_scene_referred()`, which is inconsistent with the pre-v10 rule
(`_blend_default_module_blend_colorspace(module, 0)`) a few hundred lines away
in the same subsystem. **No case was found where that is reachable with a mask
attached**, and with no mask the value is never read.

### Consequence for the patch

The fix builds clean, passes 37/37, and changes 0 pixels -- but it would alter
rendering for stored-NONE-with-blending edits, which is exactly the category
§14 itself refused to touch on the grounds that changing unproven-reachable
cases is the worse trade. Shipping it would contradict that reasoning, and
upstream would rightly ask for a repro that does not exist.

So: **reverted from this branch** (it was only applied here to build and test
it). It survives uncommitted on a `master` worktree for reference. The right
upstream move is an issue/question -- "commit resolves against a live
preference, legacy conversion pins display-referred; is stored NONE with
blending reachable?" -- not a behavioral patch.

The §14 diagnostic in `parametric.c` **stays**: it costs nothing, changes no
rendering, and if the state ever does occur it turns a silent wrong mask into a
log line. Its comment is accurate about the mechanism even though the state
remains hypothetical.

### Method note

Both errors came from trusting a quick extraction instead of checking it: a
regex that silently matched the wrong record, and a log filter that silently
mixed two different call sites. In both cases the result *looked* like the
hypothesis. Grep and probe output are evidence only once you have confirmed
they are measuring the thing you named.

---

## §16 -- Phase 3: one builder for both group-header event boxes

The report's DRY item read "Cloned headers (empty vs real) & shape rows (pending
vs real)". Measured both halves before touching either; **only one of them was
real**.

### The "cloned rows" half does not hold

`_make_pending_shape_row` (284 lines) vs `_make_shape_row` (456 lines):
**10 substantial identical lines** out of 206/272 non-comment lines -- roughly
5%, and all of it trivial (label ellipsize/xalign, the outer container's CSS
class, `return row_vbox`). The two build genuinely different things. Unifying
them would mean inventing a shared abstraction over code that is not shared,
which costs more than it saves. **Not done**, and the finding is withdrawn.

### The "cloned headers" half is real, and worth it

`_pack_empty_group_header` vs the real-group header block inside
`_masks_panel_pack`: **20 substantial identical lines** out of 148/154 -- and
unlike the rows, they form one coherent cluster rather than scattered
boilerplate: the header event box, its two identity tags, and the whole
drag-and-drop skeleton (drop-target list, drag action, drag-begin, the
conditional drag-source arm).

That is the part where duplication actually bites. A drop-target list one entry
short, or a missing `drag-begin`, is invisible until someone drags the right
thing onto the wrong header kind -- exactly the class of bug that survives
review and testing.

### What was extracted

`_make_group_header_evbox(module, hdr, lbl_box, press, release, drag_received,
source_targets, drag_get)`. It owns the invariant core: event box creation and
visible window, `gtk_container_add`, the `title-label-box` (ctrl+click rename)
and `group-header-widget` (solo dimming) tags, press/release connects,
`gtk_drag_dest_set` with `_mask_hdr_dnd`, `drag-data-received`, and the optional
drag-source arm (`gtk_drag_source_set` + `drag-data-get` + `drag-begin`).
`source_targets == NULL` means "not a drag source", which is how both kinds
already expressed "a lone group has nowhere to reorder to" (`group_movable` /
`_group_count(module) >= 2`).

Needed `g_signal_connect_data` rather than `g_signal_connect` for the
caller-supplied handlers -- the checked macro only accepts a literal
`G_CALLBACK(func)`, never a variable. The file already had that same note on
`_make_op_combo`.

### One difference deliberately left at the call sites

`drag-motion`/`drag-leave` are **not** in the helper. The two kinds highlight
different widgets on purpose: a real group highlights its whole `group_block`,
so the group-reorder insertion line spans the group's full body rather than just
its header row (its own comment says so); a staged group has only its header.
For a real group that widget does not even exist yet at the point the event box
is built. Folding this into the helper would have meant either reordering the
real header's construction or passing a widget that is sometimes NULL -- both
worse than leaving a two-line connect visible at each call site, where the
difference is the point.

Noted, not changed: a staged group's block *does* contain more than its header
(the pending-shape placeholder row), so by the real header's own reasoning its
hover highlight arguably belongs on the block too. That is a behavior change
with no test coverage, so it is filed rather than folded in.

### Verification

- full build clean
- flexi suite 37/37 (regression check only -- this is GUI-only)
- confirmed no `gtk_drag_dest_set`/`gtk_drag_source_set` on a header event box
  survives outside the helper

Needs hand-testing, since the suite cannot reach any of it: drag a group header
to reorder; drag a shape onto another group's header; drag a staged (empty)
group header; drop a group onto a staged group; ctrl+click both header kinds to
rename; right-click both for their menus. The lone-group case matters too -- with
exactly one group, neither header kind should be draggable.

---

## §17 -- Bug: group drop indicator flips at every internal boundary

User report: dragging a group upward past another group, the landing indicator
switched between "above" and "below" several times within that one group --
below over the body's lower half, above over its upper half, below again over
the header's lower half, above over its top half. Expected: one group, one
switch, at its middle.

### Cause

`_group_drop_motion` decided above/below with

```c
const int h = gtk_widget_get_allocated_height(w);
const gboolean above = (h > 0 && y < h / 2);
```

where `w` is the widget that *received* the motion event. A group is covered by
several drop targets stacked vertically -- its header event box, its block, each
element row, each cluster header -- all wired to this same handler, and each
reporting `y` relative to itself. So each one contributed its own independent
flip point. The user was not seeing one target misbehave; they were seeing four
targets each behaving correctly in isolation.

The three group-reorder *receive* handlers repeated the identical expression, so
the drop landed wherever the sub-widget under the pointer said -- the move agreed
with the flickering line, which is why this never looked like a mismatch between
preview and result.

### Fix

Two helpers, and the decision made exactly once:

- `_group_frame_of(w)` -- the frame a group-level drop target belongs to,
  resolving `"group-frame"` (element rows, cluster headers) then
  `"header-widget"` (both header kinds). Every sub-widget of a group resolves to
  the same group block, which is what makes the group *one* target instead of a
  stack of them.
- `_group_drop_above(w, y)` -- translates `y` into that frame
  (`gtk_widget_translate_coordinates`, falling back to the receiving widget if
  it cannot answer) and compares against the frame's midpoint.

`_group_drop_motion` and all three receive handlers now call
`_group_drop_above`, so the insertion line and the move cannot disagree by
construction. This is the same rule as the panel/canvas gesture pairs: one
operation, one underlying function, never parallel formulas.

### Also fixed: the staged group drew and measured on different rectangles

A staged (empty) group's header connected its hover highlight with `hdr` while
`_group_drop_above` resolves `"header-widget"` -> the block. Drawing the line on
one rectangle and deciding above/below from another would have reintroduced the
same class of inconsistency for that header kind. Its motion/leave connects
moved below the block's creation and now pass `block`, matching a real group.

That is the item §16 filed and deliberately did not fold in ("a staged group's
block does contain more than its header"). It stopped being a speculative
tidy-up once the frame became load-bearing for the drop decision, so it was done
here with a reason rather than on taste.

### Left alone

Three `allocated_height` midpoints remain and are correct: dropping a shape or a
cluster onto an element *row*, and an element hovering a row. Those are genuinely
row-level decisions ("above or below this element"), where the receiving widget
is the right rectangle.

### Verification

Build clean, flexi suite 37/37 -- regression check only; DnD is entirely
GUI-only and unreachable from `darktable-cli`.

Needs hand-testing: drag a group slowly up and down past another group and
confirm the indicator switches exactly once, at that group's middle, whether the
pointer is over its header, its body, or one of its element rows; the same past a
staged (empty) group; and that the drop lands where the line said. Worth checking
over a collapsed group too, where the block is barely taller than the header.

---

## §18 -- One insertion slot, one indicator

Follow-up to §17. With the flip-flopping fixed, the gap between two adjacent
groups still read as *two* drop targets: hovering the upper group's lower half
drew a line on its bottom edge, hovering the lower group's upper half drew one
on the lower group's top edge, ~4px away across the block margin. One slot, two
visuals, and the line appeared to jump as the pointer crossed the boundary.

### The gap has two names

"Below the upper group" and "above the lower group" are the same insertion slot.
Nothing was wrong with either -- they simply each rendered on their own block.

`_canonical_drop_frame(f, &above)` collapses the two names to one: a slot is
always drawn as the **top edge of the group below it**. Crossing between two
groups now changes nothing on screen, because both sides resolve to the same
widget and the same class. Only the bottom-most slot, which has no group below
it, stays a "below" on the last group's own bottom edge.

The list packs blocks with `gtk_box_pack_end`, so `gtk_container_get_children`
returns them bottom-first and the block visually below `f` is the one
immediately *before* it -- noted in the code, since it reads backwards.

**Presentational only.** The drop still acts on the group actually under the
pointer with its own above/below; the two describe the same gap, so the group
lands in the same place either way. `_group_drop_above` and the receive handlers
are untouched, so §17's guarantee (line and move cannot disagree) still holds
and the model logic carries no new risk.

### Consequence: clearing had to widen

The line can now be worn by a *neighbouring* block, so clearing only the frame
the event arrived on would strand it. `_clear_group_drop_classes()` clears the
frame and its siblings; both the motion and leave handlers use it.

### Also fixed: the indicator shifted the layout

`.mask-group-block` had `border: none`, and the drop rules applied
`border-top: 2px solid` -- so showing the indicator *grew* the block by 2px and
pushed every group below it down. Every change of drop target reflowed the list,
which added to the "two targets fighting" feel.

`.mask-panel-row` already avoids exactly this, and its comment says so ("The
base row already carries a 2px transparent border, so colouring one edge does
not shift the layout"). The group block simply never got the same treatment.
It now reserves transparent 2px top/bottom borders and the drop rules set only
`border-*-color`. The 4px gap between groups is now those two borders instead of
`margin-top: 4px`, so overall spacing is unchanged.

Dropped the now-inert `.mask-group-block:first-child { margin-top: 0 }`. Worth
noting it was probably not doing what it looked like anyway: with `pack_end`,
`:first-child` is the *bottom* block, not the top one.

### Correction: the first attempt had the sibling order backwards

The first version found the neighbour by position in the child list, assuming
`gtk_box_pack_end` meant `gtk_container_get_children` returns blocks
bottom-first. It returns them top-first, so the lookup ran the wrong way: the
*top* block found no neighbour and fell back to its own bottom edge, while the
lower block stayed on its top edge -- reproducing the very two-line effect the
change was meant to remove. Two user screenshots of the two pointer positions
pinned it exactly.

Fixed by dropping the assumption instead of inverting it: the neighbour is now
resolved from **widget allocations** -- the visible sibling whose vertical
midpoint is nearest below `f`'s. Same length, and it states where things are
rather than inferring it from packing order, which is easy to get backwards and
invisible in review.

### Verification

Build clean, flexi suite 37/37 (regression check only). Confirmed the rebuilt
CSS is installed to the app bundle, not just edited in the source tree -- a
theme change that is not installed silently does nothing.

Needs hand-testing: between two groups, one line only, not moving as the pointer
crosses the boundary; the slot above the topmost group and below the bottom-most
still reachable; no leftover line on a neighbour after moving away or dropping;
group spacing unchanged at rest; no vertical jitter when the indicator appears.

## §19 -- GLib/GTK criticals in the panel rebuild path

Three assertions were flooding the terminal (hundreds per session). Two are now
fixed; both were pre-existing, not introduced by this session's work.

### `g_signal_connect_data: assertion 'c_handler != NULL' failed`

`_make_op_combo` connected its `press` handler unconditionally, but every caller
passes `is_base ? NULL : _handler` -- the base group has no operator, so it has
no press handler. Fires once per base group per rebuild. Guarded with
`if(press) g_signal_connect_data(...)` (the `g_signal_connect` macro only accepts
a literal `G_CALLBACK(func)`, hence the `_data` variant for a variable).

### `gtk_box_pack: assertion '_gtk_widget_get_parent (child) == NULL' failed`

`_make_shape_row` packed `row_evbox` into `row_vbox` **twice** -- once right
after creating `row_vbox`, and again ~35 lines later at the end of the
`g_object_set_data` tagging block. GTK refused the second pack, so nothing was
visibly wrong; it just complained once per element row per rebuild. Removed the
second call.

### Method note

Reading did not find either one; the `lldb` backtrace did, in one shot. Two
things had to be right for it to work:

- `G_DEBUG=fatal-criticals`, **not** `fatal-warnings`. The latter aborts inside
  `gtk_parse_args` on GTK's own benign startup warning, before any darktable
  panel code runs -- and continuing just re-traps there forever, which is what
  made the first attempt look like a dead end.
- Remembering the build is Release + LTO, so frame #5 (`_masks_panel_pack`) is
  really *everything inlined into it* -- `_make_shape_row`, `_pack_group_elements`,
  `_pack_empty_group_header`, and so on. Searching only the literal body of
  `_masks_panel_pack` found nothing, correctly, and was the wrong search.

### Still open: `gdk_atom_intern: assertion 'atom_name != NULL' failed`

Not diagnosed. Ruled out: every `gtk_drag_dest_set` / `gtk_drag_source_set` in
the panel uses a literal, fully-populated `GtkTargetEntry[]`; darktable's only
direct `gdk_atom_intern` call site (`common/colorspaces.c`) always passes a
valid string. Its timestamp is also ~6s away from the other two, so it belongs
to a different action. Most likely GDK-quartz-internal (macOS pasteboard type
names). Left alone -- harmless, and not ours.

## §22 -- Phase 3: DnD consolidation -- mostly already done, one real defect found

### The roadmap's numbers do not hold

It calls for unifying "~23 handlers / ~650 lines". Measured: **15 handlers, 291
lines**. Sizes, largest first: `_masks_row_drag_received` 67,
`_group_drop_motion` 48, `_masks_group_drag_received` 31,
`_element_drop_motion` 26, `_masks_header_drag_received` 16, then nine handlers
of 7-14 lines each.

### The payload dispatcher already exists

The item asks for a "unified payload/dispatcher". That is exactly what the three
`*_drag_received` entry points already are -- a 3x4 matrix keyed on the
negotiated payload type:

| target \ payload | ROW | GROUP | EMPTY | CLUSTER |
|---|---|---|---|---|
| group header | `_shape_to_group_drop` | `_group_drag_received` | `_empty_reorder_drop` | `_cluster_to_group_drop` |
| element row | `_row_drag_received` | `_group_drag_received` | `_empty_reorder_drop` | `_cluster_row_drop` |
| empty header | `_shape_to_empty_drop` | `_group_to_empty_drop` | `_empty_reorder_drop` | `_cluster_to_empty_drop` |

Turning that into a literal table was considered and rejected: the eight drop
functions take four different signatures (some need x and y, some only y, some
neither, one also needs `info`). A table forces one uniform signature with
mostly-ignored parameters -- more code, and it hides which coordinates each drop
actually depends on. The if/else chains stay.

### What was actually wrong: the classifier ran twice per motion event

Both motion handlers hand-rolled the same "what is hovering me" test:

```c
const GdkAtom target = gtk_drag_dest_find_target(w, dc, NULL);
gchar *name = (target != GDK_NONE) ? gdk_atom_name(target) : NULL;
const gboolean is_reorder = name && (!strcmp(name, "dt-mask-group") || ...);
```

and `_element_drop_motion` did it, then delegated to `_group_drop_motion`, which
**did it again for the same event**. `gtk_drag_dest_find_target` negotiates
against the drag pasteboard -- on quartz a full type-list round trip -- and this
runs on every pointer motion during a drag. So hovering an element row cost two
pasteboard negotiations per motion event instead of one.

This is also the source of the `gdk_atom_intern: assertion 'atom_name != NULL'`
critical from §19: `gtk_drag_dest_find_target` is the frame directly below
`_gtk_quartz_pasteboard_types_to_atom_list` in that backtrace. The assertion is
still GTK's bug and still harmless, but element-row hovers now emit half as
much of it.

Fixed by extracting the classification once:

```c
typedef enum { DND_HOVER_OTHER, DND_HOVER_REORDER, DND_HOVER_ELEMENT } dt_masks_dnd_hover_t;
static dt_masks_dnd_hover_t _dnd_hover_kind(GtkWidget *w, GdkDragContext *dc);
static gboolean _group_drop_motion_kind(GtkWidget *w, gint y, GtkWidget *f,
                                        dt_masks_dnd_hover_t kind);
```

`_group_drop_motion` keeps its GTK callback signature and is now a two-line
wrapper that classifies and delegates; `_element_drop_motion` classifies once
and calls `_group_drop_motion_kind` directly with the kind it already has.
Behaviour is identical -- the callee was classifying the *same* widget, so it
always reached the same answer.

### Target names are now named constants

`DND_TARGET_{ROW,GROUP,EMPTY,CLUSTER}`. Each string was written twice: once in a
`GtkTargetEntry` table, once in a `strcmp` in the classifier. A typo in either
copy fails silently -- as a drag that simply never matches -- which is the worst
possible failure mode for a literal. (This overlaps Phase 4's named-constants
item; done here because the classifier refactor touched every one of them.)

### Verification

Build clean, no diagnostics in `blend_gui.c`. Flexi suite 37/37 -- again
meaningless for this change, which is entirely GUI. Needs hand testing: drag an
element over a collapsed group (must still auto-expand), drag a group over
another group's element rows (must still show the reorder line, not a
highlight), drag a cluster header, and drag an empty group -- i.e. one pass over
each of the three rows of the matrix above.

## §23 -- User-reported: an element dropped on a group lands in a NEW group

Reported once, not reproducible: "started with 2 groups, moved one element from
A to B, the element ended up in C". Target was the bottom (base) group.

### What it is not

- **Not the empty-group placeholder.** `_capture_emptied_group` only fires when
  the dragged element was the *sole* member of its run, and it leaves a
  placeholder for the *source* group, not a new group around the moved element.
  (It does have its own wart -- an unanchored placeholder, i.e. one whose run was
  bottom-most, renders in `_masks_panel_pack`'s `bottom_empties` pass at the very
  bottom of the list rather than staying put. Separate issue, noted below.)
- **Not an operator mismatch.** Both element-drop paths
  (`_masks_row_drag_received`, `_masks_shape_to_group_drop`) do
  `sp->state = (sp->state & ~DT_MASKS_STATE_OP) | (dp->state & DT_MASKS_STATE_OP)`,
  so `_eff_group_op(src) == _eff_group_op(dst)` exactly. A run boundary cannot
  come from the operator.
- **Not a stray drop on the list background.** There are exactly five
  `gtk_drag_dest_set` targets in the panel (name evbox, row evbox, group block,
  cluster box, group header). `masks_list_box` itself is not one, so a drop that
  misses them all is simply refused.

### The one mechanism left

`_starts_group` splits a run either on an operator change or on `group_start`.
Operator is ruled out, so the split must be a `group_start` stamped by
`_group_keys_apply` -- which happens iff `keys[src]` differs from the key of the
point below it.

Both paths set `keys[src] = keys[dst]`. So the failure requires **`dst` not to be
in the run the user was pointing at**. `dst` is read from the widget's
`"group-formids"`, which is refreshed only by a full panel rebuild -- and rebuilds
are `g_idle`-deferred (`_queue_masks_list_rebuild`), including the one each drop
itself queues. A stale `dst` hands `src` a foreign key, `_group_keys_apply` stamps
`group_start` on it, and it lands in a group of its own.

That is a hypothesis, not a demonstration. I could not construct the timing by
reading, and inventing a fix for an unproven cause is how §14 went wrong.

### Instrumented instead of guessed

`_verify_element_joined(grp, src, dst, where)` runs after
`_normalize_group_operators` in both drop paths, compares
`_group_cid_of_form(src)` against `_group_cid_of_form(dst)`, and on mismatch
prints at `DT_DEBUG_ALWAYS` (so no `-d masks` needed) the whole `grp->points`
layout -- index, formid, operator bits, `group_start`, and which entries are the
moved element and the target. Cost is two run lookups per drop.

Next occurrence gives the exact state, and the hypothesis above is then either
confirmed or killed in one shot.

### Separate finding, left unfixed pending a decision

The two element-drop paths disagree about the base shape.
`_masks_shape_to_group_drop` guards its insertion index with
`if(at < 1) at = 1;  // never displace the base shape from the bottom`.
`_masks_row_drag_received` has no such guard: dropping on the bottom half of the
bottom-most row inserts at index 0 and makes the dropped element the new
structural base. That may be intended -- a row drop is positional ("land exactly
here") while a header drop only means "join this group" -- so it was not
"fixed" on a guess.

## §24 -- Phase 3: DT_UI_PANEL_FLEXI joins the panel width arbitration

Two halves of one bug, both fixed.

**Left/right drags ignored the flexi column.** `_panel_set_side_panel_width`
(gtk.c) subtracts every *other* visible panel's width from what a drag may
claim, so the centre never drops below `min_center_width` -- but its
`side_panels[]` listed only `DT_UI_PANEL_LEFT` and `DT_UI_PANEL_RIGHT`. The
flexi panel is packed inside `centerrow` next to `centergrid`
(`_init_main_table`), so it takes width from the same centre. Added it to the
array (plus a NULL guard on the widget lookup, which the L/R entries did not
need).

**The flexi handle did no arbitration at all.** `_flexi_handle_motion_callback`
clamped to `min_panel_width`..`max_panel_width` and nothing else -- no window
width, no other panels, no `min_center_width` -- so that one handle could
squeeze the canvas to nothing. It now calls `_panel_set_side_panel_width`
directly. The two already shared the `panel_drag_start_size` convention (set on
press by `_flexi_handle_button_callback`), so it dropped straight in.

The block comment above the flexi handlers explicitly claimed this exclusion was
deliberate ("Deliberately not touching ... L/R-only width-arbitration logic").
Corrected: everything else about the flexi panel is self-contained, but width
arbitration cannot be, because all three columns take width from one centre.

Needs hand testing: widen flexi until the canvas hits its floor (must stop, not
keep going); widen left/right with flexi open (must account for it); flexi on
the right side (`flexi_panel_right`, the sign flip); and collapsed flexi (must
not reserve width).

## §25 -- Phase 3: the property tables -- the fork is gone, the citations were not

The roadmap wants `_blend_masks_properties` moved to `src/develop/masks/masks.c`
because it is a "fork of the deleted src/libs/masks.c" carrying "9 stale
citations".

**The citations were real: 9 of them, and `src/libs/masks.c` does not exist.**
This branch deleted that lib when the flexi panel replaced the mask manager. So
the table's own instruction -- "keep the two in sync if either changes" -- named
a file that cannot be consulted or changed. Eight references in `blend_gui.c`
and one in `masks/object.c` all pointed at it as if it were live code.

Reworded rather than deleted: the provenance ("this reproduces the old manager's
delta protocol / resize slider / paint function") is exactly what an upstream
reviewer needs, and losing it would make several non-obvious behaviours look
arbitrary. They now read "the removed mask manager's ...", with one anchor
comment at the table stating what that was and that there is nothing left to
sync against. The table is now described as authoritative, not a mirror.

**The move to masks.c was NOT done.** It is slider presentation metadata (label,
format string, display min/max) with exactly one consumer, in the file being
split. `masks/masks.c` is the data-model side; putting GUI slider metadata there
would export it for nobody. When Phase 2 eventually moves the props-row editor
into its own file, the table travels with it -- that is the move worth making,
and doing it now would land it in the wrong place twice.

### Verification

Build clean, flexi suite 37/37 (regression check only -- §24 is window layout
and §25 is comments, neither reachable from the export pipe).

## §26 -- REGRESSION from §24: negative panel width (user-reported)

"Make the window much smaller while the flexi panel is on" produced a flood of
`Gtk-CRITICAL: gtk_widget_set_size_request: assertion 'width >= -1' failed`.

**This one was mine.** §24 routed the flexi resize handle through
`_panel_set_side_panel_width`, whose ceiling is
`max_w = app_window_w - used_w`. That expression goes **negative** once the
window is narrower than the panels it already holds. `CLAMP(x, low, high)` with
`high < low` returns garbage, which `dt_ui_panel_set_size` then applies (the
assertion) once per motion event.

Not a window-resize path, despite how it looks: the flexi panel sits flush
against the window edge, so grabbing the window border to resize can land the
press on `panel-handle-flexi` instead. `panel_handle_dragging` goes TRUE and
every motion of the "window resize" is really a panel drag.

I had flagged exactly this ceiling-below-floor case when handing §24 over for
testing, and chose not to guard it pre-emptively because it also changes
left/right behaviour. That was the wrong call: the guard *is* the correct
behaviour for all three, and the flexi panel's position at the window edge makes
it trivially reachable. Pre-existing for left/right, newly easy for flexi.

Two fixes:

1. **Root cause** -- `max_w = MAX(max_w, min_w)` in `_panel_set_side_panel_width`.
   When the panels genuinely cannot all fit, refusing to shrink past
   `min_panel_width` is the honest answer; a negative width is not.
2. **Choke point** -- `dt_ui_panel_set_size` now returns early on `s < 0`. It is
   the one place that both applies a size and **persists it to conf**, so a bad
   value survives a restart. It also takes an unvalidated int straight from Lua
   (`dt_lua_gui_panel_set_size`), which no amount of fixing callers covers.

Checked the user's `darktablerc`: no panel `*_size` key holds a negative (none
are stored at all), so nothing was persisted. `_ui_init_flexi_panel_size` CLAMPs
on restore anyway, so flexi self-heals even if one had been. Note
`dt_ui_panel_get_size` does *not* clamp, and `_init_main_table` uses it before
`dt_ui_restore_panels` runs -- a stored negative would have fired one assertion
at startup. Moot now that nothing can write one.

Retest: repeat §24's four cases, then grab the window edge next to the flexi
panel and shrink the window hard. Expect zero criticals and the panel pinned at
min_panel_width rather than collapsing.

## §27 -- The canvas can still vanish: arbitration never ran on window resize

User-reported, with a screenshot: (1) start with a large window, (2) widen the
flexi panel as far as it goes, (3) shrink the window -> the canvas disappears
completely.

§24/§26 both worked on `_panel_set_side_panel_width`, which only ever runs
**while a resize handle is being dragged**. Nothing re-checked the invariant when
the *window* changed size. And a panel's width is a
`gtk_widget_set_size_request` -- a hard minimum GTK honours at the centre's
expense -- so the panels simply kept their widths and the centre was allocated
whatever was left, down to zero.

So §24 was only ever half the fix. It made the drag path honest and left the
resize path exactly as broken as it found it; the report is the same underlying
defect arriving through the other door.

`_enforce_center_width()` now runs from `_window_configure` (already connected to
the main window's configure-event for screen-change detection). It sums the three
visible side columns, and while the centre would be under `min_center_width`
shrinks the widest column -- never below `min_panel_width` -- until it fits or
every column is at its floor.

Deliberate trade, matching `_handle_panel_widths` (which makes the same one when
a panel is shown that would not fit): **the shrink is permanent**. Panels do not
grow back when the window is enlarged again. Panel width lost this way is
recoverable by dragging a handle; a canvas of zero width is not.

Note `dt_ui_panel_set_size` persists to conf on each step, so a slow window drag
writes the key repeatedly -- `dt_conf_set_int` is an in-memory hash table until
shutdown, so this costs nothing per event.

Retest: the §24 cases, §26's "grab the window edge next to the flexi panel", and
now the reported sequence -- large window, flexi as wide as possible, then shrink
the window hard. The canvas must survive at `min_center_width`, and no
Gtk-CRITICAL should appear.

## §28 -- Phase 4, part 1: ASCII punctuation, and what the key audit found

### Non-ASCII in translatable strings

Six occurrences. Only three were punctuation:

- `—` (em dash) x2 in `blend_gui.c` -> ` - `
- `…` (ellipsis) x1 in `masks_gui_presets.c` -> `...`

Checked against the rest of the tree first rather than converting on sight: the
house convention in `_()` strings is ASCII -- `_("data pending - please repeat")`,
`_("preferences...")` -- and ` -- ` appears in no translatable string anywhere.

**The three `°` (degree sign) occurrences were deliberately left.** That is a
unit symbol, not punctuation, and it is used in `_()` strings across
`channelmixerrgb.c`, `ashift.c`, `colorharmonizer.c`. Converting it would be a
regression, not compliance.

(Method note: the first attempt used `perl -i -pe 's/\x{2014}/-/'` and silently
matched nothing -- no `-CSD`, so the pattern was compared against raw bytes.
"No output" from a rewrite tool is not "no occurrences"; it was verified with a
re-grep, which is what caught it.)

### The widget-key audit found three things worth fixing

Before mass-renaming 389 `g_object_set/get_data` sites onto constants, it was
cheaper to first diff the key sets -- which is the actual failure this item
guards against:

```
distinct keys written: 83     distinct keys read: 79
```

- **`"skip-auto-expand"` -- read, never written anywhere in the tree.** The guard
  in `_shape_menu_closed` could never fire. Its comment credited
  `_shape_menu_toggle_props` with setting it; that function does not exist. The
  shape actions menu is now disable/solo/solo-edit/invert/rename/break-apart/
  delete -- the "toggle expanded controls" item was removed (shift+click on the
  handle or title replaced it), and the setter went with it. Dead code, not a
  bug. Removed, along with a second stale comment naming the same removed item.
- **`"disabled-ops"` -- written, never read.** A leftover from when the operator
  chooser disabled neighbouring operators; `disabled_ops` was a hard-coded 0.
  Removed with its local.
- **`"props-editor"` -- written, never read, and CORRECT.** It is a
  `g_object_set_data_full(..., ed, _props_row_editor_free)`: the point is the
  destroy-notify, tying the editor's lifetime to the widget. A "written but never
  read" heuristic flags every ownership handle like this. Left alone.

### The 389-site rename was NOT done

The constants' real benefit is turning a typo into a compile error -- worth
having. But the defect class they catch is exactly what the audit above found in
one command, and that audit can be re-run any time. Converting 389 call sites in
a file under active debugging is high churn for a benefit already banked, and it
would bury the last few rounds' behavioural fixes in mechanical noise.

Recommended as its own isolated commit containing nothing else, so it can be
reviewed by inspection and reverted cleanly. Not started.

### Verification

Build clean, flexi suite 37/37. The two removals are dead code (one unreachable
guard, one unread key); the string edits are user-visible text only.

## §29 -- Phase 4, part 2: dt_masks_state_t bit order and invariant asserts

### Declaration order

The enum declared its bits 0-12, then **15, 13, 14, 17, 16**. Reordered to run
monotonically 0..17.

**Values were not touched, and could not be** -- every bit is serialized into
masks blobs and XMP, so a changed value silently reinterprets existing edits.
Only the order of the declarations moved. Verified mechanically rather than by
eye: the `NAME = value` pairs were extracted before and after and diffed, and
came back identical.

### Invariant asserts

One `state` word carries three independent roles at once -- the point's own
between-group operator plus modifiers (`DT_MASKS_STATE_OP`), its group's
within-group combine mode (`DT_MASKS_STATE_WITHIN`), and the per-element flags.
Nothing enforced that those bit sets stay disjoint. Three `_Static_assert`s now
do:

1. `DT_MASKS_STATE_OP & DT_MASKS_STATE_WITHIN == 0` -- the two roles are carried
   simultaneously by a group's own state and must stay independently settable.
2. `DT_MASKS_STATE_OP_COMBINE & (OP_DISABLE | OP_INVERT) == 0` -- otherwise
   `OP_COMBINE` stops isolating the combining operator from its modifiers.
3. `DT_MASKS_STATE_GROUP_BREAK & (OP | WITHIN) == 0` -- bit 11 is historic and
   migration-only, which makes it *look* like a free bit to reclaim. It is not:
   pre-v10 blobs still carry the marker there and
   `dt_masks_legacy_params_v9_to_v10()` still reads it.

Compile-time is the right place: a collision fails silently at runtime (it reads
as an unrelated feature switching itself on), and because the bits are
serialized a clash can never be fixed by reassigning the value -- it would need
a migration.

### The asserts were verified to actually fire

`gui/gtk.h` contains a bare `#undef _Static_assert` (part of its
`g_signal_connect` signal-name checking), and `masks.h` includes it *before*
these asserts -- so "the build passed" was not evidence the asserts existed.
Proved it by temporarily moving `GROUP_BREAK` from bit 11 to bit 13 (colliding
with `OP_SCREEN`) and rebuilding:

```
error: static assertion failed due to requirement
'(DT_MASKS_STATE_GROUP_BREAK & (DT_MASKS_STATE_OP | DT_MASKS_STATE_WITHIN)) == 0':
the historic GROUP_BREAK bit has been reused by a live flag; ...
```

Then restored. An assertion nobody has seen fail is indistinguishable from a
comment.

### Not done: "group-key macros"

The roadmap's third sub-item. The group key is a `dt_mask_id_t` (a run's head
formid) passed as `guint` through `g_object_set_data`; the macros would wrap
`GUINT_TO_POINTER`/`GPOINTER_TO_UINT` at those sites. That is the same 389-site
widget-key churn deferred in §28 and belongs in the same isolated commit, not
scattered here.

### Verification

Build clean, flexi suite 37/37. The reorder is declaration-only (values diffed
identical); the asserts are compile-time and were demonstrated to fire.

## §30 -- Calibrating .clang-format instead of abandoning it

The 90-column item was initially dropped as not worth hand-reflowing ~485 lines.
`git clang-format` is the right tool for it, but running it as configured would
have made things worse. Three facts found before touching anything:

1. **Upstream deliberately deleted `.clang-format`** in `46b054cf15` (Pascal
   Obry, 2025-12-05): *"We do not want to use clang-format until a proper set of
   rules are setup which follows the dt style."* That commit is an ancestor of
   HEAD.
2. **The file here is untracked**, hidden from `git status` via
   `.git/info/exclude`. It is a local editor aid, not a project file.
3. **It contradicted the documented rules.** `ColumnLimit: 100` vs AGENTS.md's
   "lines under 90". A dry run on `masks/parametric.c` showed it *joining* a
   correctly-wrapped `snprintf` into a line of exactly **100 characters** -- it
   would have created violations while appearing to enforce style -- and
   exploding the pervasive `if(inside) *inside = FALSE;` idiom into two lines.

Rather than delete it (it is in daily use), it was calibrated.

### Method

Six pristine upstream files (`exposure.c`, `darktable.c`, `blend.c`,
`masks/circle.c`, `image.c`, `imageop.c`, ~16k lines) were taken **from the merge
base**, so the reference is real dt house style and not this branch's own code.
Each candidate option was scored by total changed lines, individually and then
cumulatively.

```
current config (ColumnLimit 100) : 5390
same at ColumnLimit 90           : 5592
calibrated                       : 4489
```

Adopted, each both measurable and a real idiom:

| option | effect |
|---|---|
| `ColumnLimit: 90` | matches AGENTS.md (correctness, not churn) |
| `AllowShortIfStatementsOnASingleLine: WithoutElse` | -281 |
| `BinPackArguments: true` | -620 |
| `BreakBeforeBinaryOperators: NonAssignment` | -166 (`&&` starts the line) |
| `AllowShortCaseLabelsOnASingleLine: true` | -36 |
| `AllowShortLoopsOnASingleLine: true` | 0 on corpus, but fixes the exact churn seen in the dry run |

Rejected: `BinPackParameters: true` (+484 -- calls pack, *declarations do not*,
a distinction worth having measured rather than assumed), `ContinuationIndentWidth: 4`
(+222), `IndentCaseLabels: true` (+208), `AlignAfterOpenBracket: DontAlign` (+1261),
`AlignTrailingComments: false` (+2).

Also rejected **despite scoring better**, because the gain was statistical noise
against a real convention: `BreakBeforeTernaryOperators: false` (-58 = 0.4%, but
puts `?`/`:` at line end when dt starts lines with them), `AlignOperands: DontAlign`
(-26), `MaxEmptyLinesToKeep: 2` (-19, and 1 is the actual convention).

### The residual ~4.5k is not tunable away, and that is the real lesson

What is left is upstream code that is simply not consistently formatted:
`*inside_border= TRUE`, `masks_size*100.0f`, `dt_masks_change_size\n  (up, ...)`.
clang-format is *right* about those -- they just are not what the surrounding
file does. No config reproduces inconsistency. This is precisely why upstream
dropped the file, and why the rule is: **format the lines you touch, never whole
files.** `git clang-format` does exactly that.

### Verified

- Config parses as `Language: C` (`--assume-filename=x.c`).
- `BinPackArguments` is genuinely live, not a silently-ignored deprecated key in
  clang-format 23 -- checked behaviourally by flipping it and diffing the output,
  since it no longer appears in `--dump-config`.
- Formatting `masks/parametric.c` now yields **0 lines over 90** (excluding
  modelines) and preserves the short-if idiom.
- **The trailing modeline block** (`// modelines:` / `// kate:` / `// vim:`,
  maintained by `tools/update_modelines.py`) is left byte-identical across the
  whole corpus. Those lines exceed 90 columns by necessity; reflowing them would
  corrupt editor configuration. Explicitly checked, not assumed.
- No file in the repo was reformatted; only `.clang-format` changed, and it
  remains untracked.

---

## §31 - Behavioural test suite for the flexi panel

Goal: pin the panel's behaviour (DnD, selection, grouping, cache invalidation)
so the classic-restore refactor and the file split cannot silently break it.

**Approach: unit-test the model, do not simulate GTK.** The panel's group model
turned out to be almost entirely pure -- `_group_keys_snapshot/apply`,
`_selected_group_formids`, `_group_cid_of_form`, `_group_partition_heads` touch
no widget and no global except `darktable.develop`. So the mock is a
`dt_develop_t` with a forms list, a module, and a `blend_data`. No `gtk_init`,
no display, no DB, no pixelpipe. Runs in 0.2s.

GTK event simulation was considered and rejected: needs a display in CI, and
injection tests are flaky enough that they get disabled rather than fixed. The
GTK-only failure modes (event bubbling, CSS, packing) are a documented manual
checklist instead -- see the suite README.

**The seam.** Gesture handlers split into a GTK half (decode the event, commit
afterwards) and a model half (perform the gesture). Both the handler and the
test call the same model function, so there is no second implementation to
drift. Worked example: `_masks_row_drag_received` ->
`_model_drop_element_onto_element`. Same principle as the panel-slider /
canvas-gesture unification.

**Layout strings** (`"u:1,2 | i:3"`) make scenarios readable and serialise
through `_starts_group`, so an assertion tests the partition the user sees
rather than the raw `group_start` flags.

**Files:** `src/tests/unittests/masks/{README.md,CMakeLists.txt,
flexi_fixture.{h,c},test_flexi_model.c,test_flexi_cache.c}`; five model
functions de-static'd into `blend_gui_internal.h`.

### Three real bugs found while building it

1. **Non-OpenCL builds were broken.** `_group_needs_host_guides` is called from
   the CPU blend path (`blend.c:895`) but its definition sat inside
   `#ifdef HAVE_OPENCL`. Any `-DUSE_OPENCL=OFF` build failed to compile.
   Surfaced immediately by configuring the test build without OpenCL. Fixed by
   moving the definition out of the OpenCL block, where the forward declaration
   already was.

2. **`group_opacity` was missing from `dt_masks_group_hash`.** It multiplies the
   group's finished sub-mask in `_group_get_mask_roi_flexi` (`group.c:1329`),
   so it is a rendering input -- but it never entered the pixelpipe cache key.
   Dragging the group-opacity slider produced no visible change until an
   unrelated edit forced a reprocess. Caught by `test_flexi_cache` on its first
   run; fixed. All three callers are runtime cache keys, nothing persisted, so
   extending the hash is safe.

3. **`unittests/` does not link against cmocka 2.x.** cmocka 1.x's config
   exports a bare `cmocka` target; 2.x exports only `cmocka::cmocka`. Every
   existing `add_cmocka_test(... LINK_LIBRARIES cmocka)` fails to link on 2.x.
   The new CMakeLists picks whichever exists; the pre-existing ones were left
   alone (own fix, arguably upstream).

### Mutation-tested, not just green

Removing the `_group_keys_apply` call from the extracted drop function
reproduces the exact reported bug -- moving an element from group A to group B
yields `u:2 | i:1 | i:3,4`, a **third** group. The suite catches it.

Notably only the *below*-drop direction caught it: dropping above the target
appends to the run's end and survives the same fault unnoticed. The group-count
test now exercises both directions.

**Status:** 65 tests across three suites, all passing, wired into `ctest -R flexi`.

### Second pass - breadth

- **Selection state machine** (7 tests). Extracted the *decision* from the
  effects: `_model_click_element` / `_model_click_group` return the selection a
  click produces; `_select_form` / `_select_group` apply it. Effect functions
  untouched, so this is a faithful split, not a rewrite. Pins the one-click
  contract including the recently-changed element-deselect-falls-back-to-group
  behaviour.
- **Operator normalisation** (4) - base-point break clearing, union default for
  operator-less points, operator preserved under bypass, partition preserved
  (the loop reads neighbour state, so mutating operators in it can misdetect a
  run boundary).
- **Solo/mute primitives** (4) - including the NULL-keep-list inversion:
  `formids == NULL` means "solo off, clear everywhere", not "keep nothing".
- **Negative cache cases** (6) - solo-EDIT, selection, cluster collapse/expand,
  canvas edit mode, solo bookkeeping and empty-group placeholders must NOT
  invalidate. Solo-edit vs solo/mute is the pair worth pinning: one is an
  editing scope, the other a rendering input.
- **`test_flexi_persistence`** (10) - mask blob version migration. Highest-risk
  surface found: runs against data nobody can regenerate, fails silently, and is
  the one place a zero-filled field is not automatically safe (`group_opacity`
  is multiplicative, so the read-time zero-fill would blank every pre-v9 group).
  Mutation-verified: removing the v8->v9 fixup fires two tests.

### Third pass - all known gaps closed

**160 tests across seven suites**, all passing, `ctest -R flexi` in ~2s.

New seams extracted (handler = decode + commit; model = the gesture):
`_model_drop_element_onto_group`, `_model_drop_element_onto_empty`,
`_model_click_element`/`_model_click_group`, `_model_toggle_solo_form`/
`_model_toggle_solo_group`/`_model_toggle_soloedit`, `_model_badge_kind`,
`_model_param_row_visibility`. Plus de-static'd: `_masks_cluster_move`,
`_masks_reorder_groups`, `_masks_visual_group_order`, `_capture_emptied_group`,
`_run_extent`, the ordinal/prune family, `_parametric_form_is_noop`,
`_param_channel_is_used`, `_normalize_group_operators`.

New suites:
- **test_flexi_dnd** (27) - group-header drops, empty-group realisation
  (ordinal/name/refinement carry-over), cluster moves, whole-group reorder
  including staged groups and re-anchoring.
- **test_flexi_groups** (29) - solo/group-solo/solo-edit mutual exclusivity and
  single-solo-at-a-time, DISABLE independence, group numbering, stale-solo and
  ordinal pruning, refinement scope + disjoint bypass key spaces.
- **test_flexi_panel** (21) - low-opacity and no-op badges, adaptive parametric
  row display, panel preferences (needs a scratch `darktable.conf`; see
  `flexi_conf_init`).
- **test_flexi_migrate** (18) - the classic->flexi case table, all 16 bit
  combinations swept, including the GUI-unreachable RASTER+MASK/CONDITIONAL
  cases and the degenerate-parametric collapse.

**Seam violation fixed:** `_masks_cluster_move` committed history inside the
model (it took dev->history_mutex), unlike every other extraction. Moved the
commit out to its two GTK callers.

### §32 - masks_revamp_flexi_migration_plan.md corrected against the code

Writing the structural migration tests surfaced that the plan doc was written
ahead of the implementation and never reconciled. Six divergences, all fixed in
the doc (the code was right in every case):

1. **Status** said "proposed, unimplemented". It has shipped.
2. **Where it hooks in.** The doc attaches the migration to the
   `old_version == 14` branch. It actually runs at the *tail of every* version
   branch, so a v9 edit migrates as surely as a v14 one.
3. **Idempotency rationale was wrong** as a consequence of (2): the doc's
   "only fires for old_version == 14 data" argument does not hold. What
   actually guarantees it is case 8's FLEXI guard.
4. **Case 1** (bare ENABLED): doc said leave it as plain ENABLED, "setting
   FLEXI would add a bit with nothing to point at". Code sets ENABLED|FLEXI
   with mask_id = NO_MASKID, so "every mask_mode is DISABLED or a flexi state"
   is exception-free -- which the mode-select UI relies on.
5. **Case 3, twice over.** (a) The doc says an all-default blendif "is still
   synthesized, not optimized away"; the code collapses it, correctly -- these
   configs never reach the per-channel curve in classic at all, so synthesizing
   a form would fabricate a mask where classic had none, and invert it in the
   INCL != INV case. (b) The doc says one `single = 0` multi-channel form; the
   code builds one single-channel form *per active channel*, joined by
   WITHIN_MULTIPLY, so each channel is separately editable in the panel.
6. **Case 4** was a one-liner in the doc and is a three-branch decision tree in
   the code. Added §3.1 documenting `_classify_conditional`'s REAL / CONSTANT /
   PASSTHROUGH split, which both cases 3 and 4 route through.

Also corrected: §5's persistence design (the plan's per-row
`_migrate_persist_form` does not survive a reload -- hence
`dev->pending_flexi_migrations` and `dt_masks_finish_flexi_migrations()`),
the proposed-but-never-written `dt_masks_tree_uses_raster()` helper (the
concern was real and is handled by `_reconcile_raster_form_users()` instead),
the fail-closed rule's one genuine exception (deferral-record allocation
failure clears the mask rather than staying classic), and §6, which now
describes the two suites that exist and what each reaches that the other
cannot.

Every corrected claim was re-verified against the source afterwards.

**Left alone deliberately:** the degenerate-parametric collapse leaves
`mask_mode == ENABLED` without the FLEXI bit, so migrating such an edit twice
gives 0x1 then 0x11. No classic bit survives either way, and the doc now
describes the behaviour accurately -- but whether ENABLED and ENABLED|FLEXI
render identically with no form is a renderer question, so the code was not
changed to "fix" the asymmetry.

### §33 - the operators themselves, and the bypass snapshot

Second sweep for testable-but-untested surfaces. Two found, both significant.

**test_flexi_compose (21).** `_combine_masks_*` -- the arithmetic behind every
operator the panel exposes -- had never been tested directly, only inferred
from rendered images. Exported via a new `masks/group_internal.h` (mirrors
`blend_gui_internal.h`). Beyond per-operator value checks, this pins the
properties the design *states as fact*: the group-fold operators
(union/screen) are commutative and associative, which is the whole
justification for a group being an unordered bag of shapes; difference is
asserted NOT to be; each operator's identity element, which is what makes
skipping an empty group correct -- notably that intersection's identity is an
all-ONE mask, so an empty intersect group must never composite as all-zero;
and that every operator keeps the mask in [0,1] across opacity and invert.
Mutation-verified (intersection MIN -> MAX fires 2 tests).

**Refinement-bypass snapshot (8, in test_flexi_groups).**
`dt_masks_refine_bypass_lookup` is a binary search over an array
`dt_masks_refine_bypass_commit` qsorts -- an unguarded sorted-array
precondition. Now covered: sortedness at scale (24 keys straddling the
top-bit boundary between the element and group key spaces), staged-group keys
(pointer-keyed) never leaking into the snapshot, flexi-only gating, empty
lookups, and hash canonicality (same set, different insertion order -> same
hash). Mutation-verified (removing the qsort fires the at-scale test).

**Process note.** The first qsort mutation run reported PASS -- but the build
had not actually rerun (output was redirected to /dev/null and the restore
raced it). Re-run with visible build output it failed correctly. A mutation
that "doesn't fire" must be re-checked before being believed.

**Also fixed:** the cmocka 1.x/2.x target-name split, centrally this time.
`unittests/ai` already used `cmocka::cmocka` while `unittests/` and
`unittests/iop` used bare `cmocka`, so the tree was internally inconsistent
and a full `cmake --build` failed on cmocka 2.x. The top-level CMakeLists now
aliases whichever name exists to the other, and the masks CMakeLists dropped
its local workaround.

**Pre-existing, not fixed:** `test_filmicrgb` cannot link on macOS -- it uses
`AddCMockaMockTest`'s `--wrap` symbol interposition, which is GNU ld only
(`ld: unknown options: --wrap=...`). Unrelated to this work; it means a bare
`cmake --build build-test` still fails at that target on macOS while every
other test target builds and passes.

**Status: 189 tests across eight suites**, all passing.

---

## §34 — The probe image, and three tests that measured the wrong thing

Groundwork for the migration verifier: the harvested masks have to be replayed
against *some* image, and it cannot be the user's own photo (we deliberately
never collect those). Hence `develop/masks/probe_image.c` — a deterministic,
programmatically generated image, plus `test_probe_image.c` (7 tests) that
measures whether it is actually fit for the job.

**Why the probe is the weak point.** A parametric mask selecting a range the
probe never produces renders all-zero, and all-zero compares equal to all-zero
however badly the migration mangled it. Every such case is a verification that
runs, reports success, and proves nothing.

**Where the bar comes from.** Not from any library of real edits. Profiling a
sample of real masks and covering exactly what it uses bakes one person's
habits into everyone's verification — ranges nobody in the sample happened to
touch look like ranges nobody needs. The test instead sweeps the linear-RGB
cube through darktable's own colour conversions to discover what each blendif
channel can physically take, and holds the probe to that. (Two intermediate
bounds *were* briefly taken from a real library and then removed; the whole
framing was wrong, not just the numbers.)

Two alternatives were considered and rejected on the way: a flat [0,1] per
channel is false for Jz/Cz, which the slider reaches only through a boost
factor (default offset log2(1/100), so slider 1.0 means Jz = 0.01); and bounds
from the boost slider's own 0..18 EV range put the largest addressable Jz near
2600, which nothing can cover and no image can reach.

**Global vs local coverage are not the same property** and must not share a
standard. Globally the probe should span everything the colour space expresses,
highlights included. Locally it should not: requiring a patch a sixty-fourth of
the image across to span four stops uniformly is not something a photograph
does either, and the probe could only satisfy it by becoming unphysical. The
local test is therefore taken over the diffuse cube ([0,1] linear).

**Three tests measured the wrong thing, and mutation testing is the only
reason that is known.** All three passed on probes that were plainly
inadequate:

- *Hard edges, by count.* Defeated by the tile lattice, which puts a sharp
  step every 16px regardless of whether anything else in the probe has
  structure. Replaced by a test on the *distribution of edge orientations* — a
  pure grid occupies 2 of 16 bins, real structure occupies all of them.
- *Texture, as total per-octave energy.* Same cause: a periodic square lattice
  has harmonics in every band.
- *Texture, restricted to edge-free quads.* Defeated by circularity — the
  noise is what makes a quad non-flat, so selecting flat quads selected
  against the signal. This version agreed with a noise-free probe to five
  decimal places.

The version that works takes the diagonal (HH) wavelet coefficient over every
quad with no selection at all, and reads its **median**. HH is identically zero
for any function linear in x and y, so the tile's own ramp contributes nothing
however steep; hard steps are sparse and cannot move a median. With the noise
switched off the median is exactly 0.0.

**Two real generator bugs surfaced the same way.** The "irregular" second edge
family used `cell = tile * (2 << level)`, so every cell boundary landed exactly
on a tile boundary and the whole family added no edge the lattice did not
already have. And the orientation bar was first expressed as a share of the
edge population — a denominator dominated by the axis-aligned lattice, so every
off-axis edge added also raised the bar it was judged against, penalising the
probe for having more structure. It is now an absolute fraction of pixels.

**Also found and fixed by the coverage tests, before any of the above:**
the probe had no genuinely dark pixels (LAB L bottom bin unreachable — red and
green sweep independently, so a dark pixel needs all three channels near zero
at once and the noise lifted even those), and never reached the gamut boundary
(top Cz bins unreachable — a rectangular walk of the cube hits its corners only
at exact tile corners). Fixed by extending the exposure ladder down to -6 EV
and adding a saturation ladder.

**Process note, second occurrence.** A mutation reported as not firing was
re-checked rather than believed — and this time the mutation *was* correctly
applied and rebuilt (verified by grepping the source and watching the object
file recompile). The test was simply blind. Same rule, opposite conclusion: it
resolved to a test defect, not a harness defect.

**Status: 196 tests across nine suites**, all passing.

---

## §35 — `--harvest-masks`: collecting real edits without collecting anything else

`develop/masks/harvest.c` + a CLI flag. `darktable --harvest-masks out.json`,
honouring `--library` / `--configdir`, writes every mask configuration in a
library to readable JSON and exits.

**Read-only is a structural property here, not a promise.** An ordinary
darktable startup opens the library read-write, locks it, and will upgrade its
schema in place if an older version wrote it — so merely pointing a normal run
at a helper's real library modifies it. The harvest therefore does not use
darktable's database handle at all: it is dispatched from `dt_init` *before*
the flexi-test-mode block and long before `dt_database_init()`, opens its own
connection with both `mode=ro` in the URI and `SQLITE_OPEN_READONLY` in the
flags, and exits without reaching the rest of startup.

Position also matters for correctness, not just safety: flexi test mode
rewrites `--library` to a scratch copy that is seeded once and then diverges,
so harvesting after it would have read a stale snapshot of the user's edits.

Verified empirically on the real 161MB library: identical SHA-256 before and
after, and no `.lock` / `-wal` / `-shm` files created.

**Privacy is why the format decodes blobs instead of shipping them.** We are
asking strangers for this file, so it is plain JSON with every value in named
fields — anyone can open it and check. Free text hides in four places and all
four are stripped: `images.filename`/film-roll paths (never queried),
`masks_history.name` (user-renameable), `history.multi_name` (there is even a
`multi_name_hand_edited` column), and — the one that forced the design —
`dt_masks_point_group_t.name[128]`, a user-typed group name that lives *inside
the points blob*. A base64 dump of a group's points would have carried it, and
nobody auditing the file would have seen it.

Audited on the real output: 2468 edits contain **42 distinct strings in total**,
every one a module operation name, form type, colourspace or mask-mode label.
Image identity is reduced to width/height plus a sequential index.

Group point blobs are decoded with the same per-version stride rules as
`dt_masks_read_masks_history()` (v7 refinement, v8 name, v9 group_opacity, v10
group_start), and the tail is zero-filled the same way, so an old edit is
reported with the defaults the loader would actually give it. A blob whose size
disagrees with stride × count is reported as an error rather than guessed at.

**Real library:** 144,242 history entries scanned, **2,468 edits with masks
across 544 images, 11,969 forms**. 42MB of JSON, 3.8MB gzipped. The 2,468
matches the earlier survey exactly (2,431 migratable + 37 already carrying the
FLEXI bit).

**Still to build: the verifier.** It replays each harvested edit against the
generated probe, before and after migration, and compares mask buffers. The
pieces are in place — parametric forms read their input from
`piece->blend_refine_guide_in/out`, so the probe can be handed straight to
them, and `dt_masks_group_render_roi()` needs only module/piece/form/roi.
json-glib is already a dependency, so shared harvest files can be read back.

---

## §36 — Generating the rare cases instead of waiting for them

Correction to §35's closing claim that `INV`/`INCL` "stay synthetic-fixture
territory permanently, and no corpus size fixes this". True as far as it went,
and then wrong about what follows from it: the right response to a branch no
real edit reaches is to *generate* the input, not to wait for a contributor who
happens to have one. `mask_combine` is three bits — eight values — so the space
is enumerable outright, and enumerating beats sampling.

Four new tests in `test_flexi_migrate.c` (suite now **200 tests across nine
suites**), crossing all 8 combine values with the mask modes:

- INV/INCL never survive migration where they mean something.
- Parametric-only, reaching `DT_COND_REAL`: `MASKS_POS_out == (incl != inv)`.
- Drawn+parametric: `MASKS_POS_out == (INV ^ INCL)`.
- Every combine value lands in a valid end state (crossing the existing
  exhaustive `mask_mode` enumeration with the combine bits it held fixed).

Both XOR tests are mutation-verified (flipping `incl != inv` to `incl == inv`
fires one; dropping the `^ incl` from `invert_composite` fires the other).

**Three test defects found on the way, all of them mine, none a product bug:**

1. *Asserting against inputs the real path cannot produce.*
   `_fix_masks_combine()` runs in **every** version branch of
   `dt_develop_blend_legacy_params_ext()` (lines 2314–2627) and the flexi
   migration runs after it (2698), so a **drawn** mask can never reach
   migration with `INV` set — it has already been rewritten to `MASKS_POS`.
   Calling the migration directly bypasses that. The tests now reproduce the
   precondition explicitly (`_apply_legacy_combine_fix`) and compute their
   expectations from the post-fix value.

2. *Degenerate test data.* All-zero `blendif_parameters` is not a partial
   range but an empty one, and darktable's default `[0,0,1,1]` is the full
   range; both classify as degenerate and collapse, so the test would never
   reach the branch it named. Ranges are now genuinely partial.

3. *Wrong expectation, right code* — the same doc-vs-code pattern as §33's
   deleted test. With `INCL` set, `_classify_conditional()` flips the polarity
   of *every* channel in the colourspace mask, so any channel left inactive
   becomes a canceling channel and the config collapses to
   `DT_COND_CONSTANT`. Reaching `DT_COND_REAL` with `INCL` therefore needs
   **all** channels active, not merely some. "Set INCL and two channels" — the
   obvious way to build this test, and what a user would do by hand in the GUI
   — never reaches the INCL algebra at all. That case is now pinned by its own
   test (`..._collapses_to_a_constant`) rather than being a silent hole.

Point 3 is also the reason to prefer generated cases here on the merits, not
just on convenience: a hand-made GUI example of "inclusive with a couple of
channels" would have looked like coverage of the INCL branch while testing the
constant-collapse path instead.

**Where real data is still irreplaceable** (i.e. the harvest is not made
redundant by this): old `blendop_version` / masks-version blobs. Synthesizing
those would encode *our belief* about what old darktable wrote — the same
belief the migration code encodes — so a wrong belief would cancel out and the
test would pass vacuously. Only blobs actually written by old versions break
that circularity. Likewise real path/brush geometry at scale, and combinations
nobody would think to enumerate.

**Also this round:** the harvest now tallies its own rare cases into a
`coverage` section (combine bits, the five mask-mode cases, refinement usage,
form types, blendop/masks version histograms) and calls out INV/INCL on stdout,
so a submitted file announces its own value instead of needing to be mined for
it.

---

## §37 — The verifier: rendering equivalence on 2466 real edits

`develop/masks/verify.c` + `--verify-masks FILE`. Replays every edit in a
harvest file, renders its mask before and after migration, compares.

**It does not compute what the mask "should" be.** That would encode this
file's beliefs about classic blending — the same beliefs migrate_legacy.c
encodes — so a wrong belief would cancel and the comparison would pass. Both
renders go through the real `dt_develop_blend_process()`, unmodified; the mask
is recovered by setting `pipe->store_all_raster_masks`, which makes the blend
publish its finished mask into `piece->raster_masks`.

**Mutation-verified before trusting any result**: flipping `invert_composite`
in the migration turns 4 of 9 live subset edits DIFFERENT with max diff 1.0.

### Five bugs in the harness, found by making it work

1. `self->flags()` — a hand-built `dt_iop_module_t` crashes. blend_process
   calls through the module's function pointers and its blend colourspace is
   per-module, so a stub would crash or (worse) replay everything in the wrong
   colour space. Now loads the real module the edit names.
2. `dt_iop_load_module_by_so()` returns **TRUE on failure**. My check was
   inverted, rejecting every successful load.
3. `dev->history_mutex` uninitialised → abort. It must also be **RECURSIVE**,
   as `dt_dev_init()` makes it: a default mutex does not abort here, it
   deadlocks, which reads as the verifier hanging rather than as a bug.
4. `piece->iscale` left at 0. Radii are converted as
   `roi_out->scale / piece->iscale`, so zero divides by zero and asks the
   guided filter for an effectively infinite window. Does not crash — runs
   forever. Also needs full-image dimensions on the pipe (masks are normalised
   against those) with `roi.scale` carrying the downscale.
5. **The big one: forms were harvested per history entry.** masks_history
   writes a row only when a form is *created or changed*; an entry that merely
   references an existing mask writes none. Selecting `num = ?` therefore
   returned nothing for such entries, and they replayed with a dangling
   mask_id and no geometry. Fixed to match `dt_masks_read_masks_history()`:
   every row with `num <= ?`, latest per formid. **Forms went from 11,955 to
   27,803** — more than half the geometry had been missing, and it was being
   attributed to the migration.

### Result on the real library

2466 replayed: **2274 identical, 2 equivalent, 30 different, 160 skipped, 0
errors.** Live (non-uniform classic mask) 1913 → 1897 identical, 2 equivalent,
14 different. Inert 393. The form fix moved 495 edits from inert to live, i.e.
that many comparisons went from proving nothing to actually testing something.

### The 30, NOT yet established as migration bugs

Two clusters:

- **15 × retouch**, mask_mode 5 (parametric only), on a module flagged
  `IOP_FLAGS_NO_MASKS`. Classic renders a uniform 1.0, flexi a uniform
  `opacity`. Systematic and identical across every instance. Note the
  guide-image assignment in blend.c sits behind
  `form && form->points && mode_drawn && !(flags & IOP_FLAGS_NO_MASKS)`, and
  parametric forms read their input from `piece->blend_refine_guide_in/out` —
  a plausible mechanism, unconfirmed.
- **14 × drawn+parametric**, mostly with feathering. Small systematic mean
  shift (e.g. 0.4948 → 0.4913) with min and max unchanged, but per-pixel max
  differences up to 0.37.
- 1 × overlay with a genuinely dangling mask_id.

**Caveat that must not be dropped: at least one edit (idx 1169) reported
DIFFERENT in the full run and identical when replayed in a small subset.** So
state still leaks between replays, and until that is found the 30 cannot be
called migration bugs. Triage order: fix the cross-edit state leak, re-run,
then investigate whatever survives.

---

## §38 — Every difference accounted for: 2466 real edits, 0 differences

Iterating on §37's 30 unexplained results until each had a cause. Final state
of `--verify-masks` on the real library:

**2301 identical, 5 equivalent (max 3.8e-5), 0 DIFFERENT, 160 skipped, 0 errors.**
Live (mask genuinely varies) 2006 → 2001 identical, 5 equivalent, 0 different.

### The five causes

1. **Forms harvested per history entry** (§37). Fixed; forms 11,955 → 27,803.

2. **OpenMP float reassociation.** Two identical full runs disagreed on 4 of
   2466 edits — three by <0.004 but one by **0.1**, which looked exactly like a
   real bug. Parallel reductions over pixels make the last bits depend on
   thread scheduling. Forced single-threaded in the verifier: all four become
   identical and stable. A verifier whose answer moves between runs cannot be
   used to investigate anything, so the slower pass buys the only property
   that makes the output actionable. It also removed the "equivalent" bucket
   entirely at that stage, proving the noise was all threading.

3. **No colour profile on the replay pipe** — the subtlest, and it made the
   *pass* count dishonest too. The per-channel branch of every
   `blendif_*_make_mask()` calls `dt_develop_blendif_init_masking_profile()`
   and **returns leaving the mask untouched** if there is none. So parametric
   masks were never evaluated at all — and the two sides failed
   *asymmetrically*: classic still carries `DEVELOP_MASK_CONDITIONAL`, enters
   that branch and bails; a migrated edit has had CONDITIONAL folded away,
   takes the early "not conditional" path, and applies global opacity. Result:
   a clean spurious "the migration changed this mask" on parametric edits.
   Fixed by giving the dev an iop-order list and the pipe a linear Rec2020
   work profile. Live edits rose 1913 → 2006.

4. **A real product bug** — see below.

5. **5 × float reassociation, accounted not fixed.** All exposure, all with
   both feathering and a detail threshold, max 3.8e-5 over ~200 of 175,000
   pixels. Classic computes `(drawn × parametric)` then refines; flexi
   composites the parametric as a group element, so the multiplications land
   in a different order. Two orders of magnitude below 1/255.

### The product bug: NO_MASKS modules lost their parametric mask

24 edits, every one `retouch` in parametric-only mode. `IOP_FLAGS_NO_MASKS`
(retouch, spots) means the module consumes drawn forms itself in `process()`,
so the blend must not also render the forms behind `mask_id`. That reasoning is
about **drawn** masks — such a module can still carry a parametric blend mask,
which classic evaluates in `make_mask()` with no group involved.

Migration moves that parametric config into a form inside a flexi group — and
the group is exactly what the gate refused to render. The mask collapsed to a
flat opacity.

Nothing structural was wrong: the migration produced a correct group with a
correct parametric form, and **all 200 structural tests passed**. It took
replaying real edits to see it. This is precisely the silent-divergence class
§35 argued the structural suite could not reach, and the first hard evidence
that the verifier earns its cost.

Fixed in `blend.c` via `dt_blend_may_render_group()`: the flag blocks a classic
drawn group, never a flexi one. A flexi group is never the module's own shapes
— it is created by the panel or by migration under a new formid, and for a
NO_MASKS module can only hold non-drawn elements anyway (`bd->masks_support`
refuses drawn shapes there, and migration only synthesizes parametric/raster
forms for a module that was never allowed a drawn mask).

Pinned by 4 new tests (suite now **204 across nine suites**), mutation-verified:
reverting the gate fires two of them.

### The 160 skips, accounted

Exactly `already_flexi (37) + raster (123)`. Already-flexi is a no-op by design
(case 8's guard). Raster reads its mask from another module's pipe piece, so a
standalone replay would render empty on both sides and report a meaningless
pass — skipped explicitly rather than counted. **123 edits remain unverified**;
closing that needs a two-module pipe and is the obvious next gap.

### Process note

Two darktable runs must never overlap — the library lock makes the second fail
(one background probe job died exit 1 that way). Verification runs are
serialized now.
## §39 — Closing the raster gap: 123 real edits + 288 generated, and a second product bug

§38 closed with the one coverage hole it could not close: 123 harvested raster
edits skipped, because "a raster mask reads its mask from another module's pipe
piece, so a standalone replay would render empty on both sides and report a
meaningless pass". That is now closed, and closing it found a second real bug.

**The scope was smaller than "a two-module pipe" made it sound.** Classic raster
mode is *exclusive* — it cannot be combined with a drawn or parametric mask — so
a classic raster edit is fully described by its source, an invert flag, the
module opacity, and the global refinements applied downstream of the mask. There
is no compositing algebra to verify, only inversion and refinement.

### The harness: a real upstream piece, wired by production code

`_attach_raster_source()` in `verify.c` stands up the source the edit names: a
real module instance pinned one step earlier in the iop order, an enabled pipe
piece holding a synthetic mask, and both `pipe->nodes` and `dev->iop` populated
so `dt_dev_get_raster_mask()` resolves exactly as it does live.

Two details worth recording, because getting either wrong makes the run lie:

- **The sink pointer is not `blend_params->raster_mask_source`.** The classic
  raster branch follows `module->raster_mask.sink.source`, a *resolved module
  pointer* that only `dt_iop_commit_blend_params()` ever sets. Setting it by
  hand would have been the harness deciding what the pipe should have resolved.
  The replay now calls the real function, for every edit rather than only raster
  ones, so there is one wiring path instead of a special case — and the flexi
  side gets its raster *form* elements registered by the same call
  (`_reconcile_raster_form_users`).
- **The synthetic mask is deliberately not derived from the probe.** A raster
  mask is an *input* to everything under test, so it needs shape of its own:
  a smooth radial falloff reaching exactly 0 and exactly 1, with a diagonal term
  so a transpose-style error cannot pass, and disagreeing with the probe's own
  edges so the guided filter has something real to work on.

Result: **123 replayed, 123 identical, 118 of them live.** The skip category is
gone; the full corpus now skips only the 37 already-flexi edits, and live
comparisons rose from 2006 to 2133.

### The bug: an unresolvable raster flipped from "off" to "fully on"

5 of the 123 carry mask mode RASTER with an **empty source** — a source module
removed at some point. Classic: `dt_dev_get_raster_mask()` returns NULL and the
raster branch fills the mask with **0.0**, so the module contributes nothing.
Flexi: the raster form failed to resolve and returned 0 meaning "did not
render", `_group_get_mask_roi_flexi()` did not count it, `nb_members == 0` tripped
the deliberate *"no active mask element → fully opaque"* fallback, and the mask
filled with **1.0**. The module went from doing nothing to applying at full
strength over the whole image. Max possible difference, on real edits.

That fallback is right for what it was written for — a group the user is still
building, where a yellow wall would hide the image they are placing shapes on.
It is wrong for a reference that cannot be resolved. Fixed in `raster.c`:
`_raster_unresolved()` fills zero and reports success, so the member is counted
and the group renders "nothing", matching classic.

The fix is deliberately at render time rather than in migration, because the
failure is not static. Migration can see an *empty* source; it cannot see a
source module the user deletes next week. Both cases now behave the same.

### Generating the corners the corpus does not have

The harvest proves the migration works on edits someone actually made. It does
not cover the corners: the global blur refinement appears in **1** of the 123
edits, and inversion never co-occurs with a details threshold at all.

Since the space is small and closed, it is enumerated —
`gen_raster_matrix.py` emits an ordinary harvest file (same schema, no new
machinery, no user data, regenerable by anyone) crossing invert × opacity ×
feathering × blur × tone curve × signed details × colour space = **288 edits,
all live, all identical**.

**Mutation-verified, and the comparison is the point:**

| mutation | generated matrix | real corpus |
|---|---|---|
| drop `DT_MASKS_STATE_INVERSE` in `_migrate_raster` | 144 different (exactly the inverted half) | — |
| skip post-processing for raster (`!uniform` → `!uniform && !raster`) | **264 of 288** caught | **22 of 118** caught |

The refinement mutation is the honest argument for generating: the real corpus
catches it 12× less often, because the combinations that would expose it are the
ones the corpus barely contains. Neither source replaces the other — real blobs
remain irreplaceable for old on-disk formats (§36), and generated cases remain
irreplaceable for the corners.

### A caveat on the "identical" count that I should have caught earlier

13 edits moved from `identical` (several at *exactly* 0) to `equivalent` at
~4e-6 between this run and §38's, and the previous worst case *shrank* from
3.8e-5 to 1.1e-5. I env-gated every change I had made and rebuilt: the new
numbers reproduce with all of them reverted, so **this is not caused by these
changes** — the binary's floating-point codegen differs between builds. I did
not isolate the mechanism further (a comment-only perturbation is not a codegen
perturbation, so that probe proved nothing).

What follows matters more than the cause: **the identical/equivalent boundary at
`VERIFY_EPS_IDENTICAL = 1e-6` is not a stable property of the migration**, it
sits inside build-level float noise. The claim the verifier actually supports is
the one at the other threshold — **0 differences above 1/255**, with the worst
observed deviation anywhere in 2466 edits at 1.1e-5, roughly 350× below
visibility. Quoting the "identical" count as if it were exact would be
overclaiming; it is quoted here as a diagnostic, not a guarantee.

### Standing

- Full corpus: **2466 replayed, 2411 identical, 18 equivalent, 0 DIFFERENT,
  37 skipped (already-flexi only), 0 errors.** Live: 2133.
- Generated raster matrix: **288/288 identical, all live.**
- Unit suite: **205 tests across 9 mask suites, all passing** (`test_filmicrgb`
  does not link on macOS — `--wrap` is a GNU ld option — pre-existing and
  unrelated).
- Two real product bugs found by replaying real edits, neither reachable by any
  structural test: the `IOP_FLAGS_NO_MASKS` group gate (§38) and this one.

## §40 — The GPU path, which none of the above tested

Reviewing readiness for testers surfaced the gap the whole verification effort
structurally cannot see: **`--verify-masks` replays `dt_develop_blend_process`,
the CPU blend. `dt_develop_blend_process_cl` is a separate hand-maintained
implementation of the same branch structure, and every number in §34-§39 says
nothing about it.** Most testers run OpenCL.

Auditing it against the CPU found one live divergence, in the branch
immediately next to the `IOP_FLAGS_NO_MASKS` fix of §38:

```c
else if(mode_parametric && dt_blend_may_render_group(self, mask_mode))
{
  // no form defined but drawn mask active          <- comment says drawn
  const float fill = inverted ? 0.0f : 1.0f;
```

The CPU tests `mode_drawn`; the comment says drawn; the code tested
`mode_parametric`. Pre-existing, and harmless while classic edits kept
`DEVELOP_MASK_CONDITIONAL` set — but **migration always clears CONDITIONAL**, so
`mode_parametric` is FALSE for every migrated edit and the branch became
unreachable. A flexi group that renders nothing (an empty group, or one whose
only member is a still-full-range parametric channel — the `is_uniform_noop`
rule in `_group_get_mask_roi_flexi`) then fell through to the final `else` and
its `INCL ? 0 : 1` fill.

Net effect: a migrated edit carrying `DEVELOP_COMBINE_MASKS_POS` with a group
that renders nothing applies to **nothing on the CPU and to the whole image on
the GPU**. Migration is what makes it reachable, so it counts as introduced by
this branch even though the typo is older.

Fixed to `mode_drawn`. **Reasoned, not measured** — the corpus cannot exercise
it, and the fix is recorded as such in the code comment rather than being
folded into the "0 differences" claim.

**Also checked and found *not* reachable:** the same final `else` reads
`DEVELOP_COMBINE_INCL`, which `_clear_toplevel_blendif()` does not clear on the
drawn-only or raster migration paths. It would need INCL set on a drawn-only
edit; across all 2466 harvested edits, drawn-only carries `(INV, INCL,
MASKS_POS)` = `(0,0,0)` 627 times and `(0,0,1)` twice, and never INCL. Left
alone rather than "fixed" speculatively.

**Standing gap for testers:** the OpenCL blend path has no equivalent of the
CPU corpus replay. Building one means running both pipes over the same edits and
diffing — the natural next step, and the honest caveat until then.
## §41 — The GPU replay, and the correction it forced

§40 flagged that `--verify-masks` only ever exercised `dt_develop_blend_process`
and said building a GPU equivalent was "the natural next step". Built. It did
not confirm the previous results — it invalidated a substantial part of them.

### The harness: four renders, and a baseline

`_render_mask_cl()` uploads the probe to the device, calls
`dt_develop_blend_process_cl`, and recovers the mask the same way as the CPU
side (the CL tail already copies it back and publishes it through
`dt_iop_piece_set_raster`). Each edit is now rendered four times: classic and
migrated, on CPU and on GPU.

Four rather than two because a raw CPU-vs-GPU number would be unreadable. The
two implementations are separately maintained and never agree to the last bit,
so the run records the CPU/GPU gap on the **classic** edit as a baseline and
judges the migrated gap against it. The question is whether migration *widens*
the gap, not whether a gap exists.

### The correction: "0 differences" was partly measuring nothing

`dt_masks_group_get_mask_roi()` dispatches on `bp->mask_mode & DEVELOP_MASK_FLEXI`
— flexi groups take `_group_get_mask_roi_flexi()`, classic groups the sequential
fold. **Two different algorithms for the same form.**

`piece->drawn_mask_cache` was keyed on the form hash, the refine-bypass hash and
the roi. **Not on `mask_mode`.** So in the replay: the classic render populated
the cache; migration flipped `mask_mode` to FLEXI without touching the form; the
"after" render computed the same key, hit the cache, and returned *the classic
renderer's output*. For drawn-only edits `cpu_before == cpu_after` was
guaranteed by the cache, not by the migration being right.

This is the vacuous-comparison failure this file's own header warns about ("two
all-zero masks compare equal however wrong the migration was") in a disguise the
probe-coverage tests and the inert/live split cannot see: the masks are live and
varied, they are just the same buffer twice.

**The GPU found it because the CL path has no such cache.** For one exposure
edit:

```
cpu_classic max=0.6202   cpu_flexi max=0.6202   <- cache hit
gpu_classic max=0.6202   gpu_flexi max=0.1723   <- rendered for real
```

Cache key fixed to include `mask_mode` (a genuine latent bug in its own right —
a key must cover everything the result depends on, and it was omitting *which
algorithm runs*). The CPU then agreed with the GPU: `cpu_flexi max=0.1723`.

Corrected headline, same corpus: **2187 identical, 124 equivalent, 118
DIFFERENT, 37 skipped.** Worst CPU difference **1.0**, previously reported as
1.1e-5.

### The real bug: the operator is applied per *run*, not per element

**Corrected from the first write-up of this section, which claimed flexi "has no
DIFFERENCE, SUM or EXCLUSION at all" and that migration dropped the operators.
Both were wrong.** `DT_MASKS_STATE_OP_COMBINE` is precisely `UNION |
INTERSECTION | DIFFERENCE | SUM | EXCLUSION | MULTIPLY | OP_SCREEN` -- the
classic bits *are* the flexi **between-group** operators. The model expresses
all of them. (Thanks to DP for catching this; the wrong diagnosis would have
sent the fix into extending the model, which is not what is needed.)

What actually differs is *where* the operator is applied:

- **Classic** folds sequentially, applying each element's own operator onto the
  accumulator, once per element.
- **Flexi** partitions `grp->points` into maximal same-operator runs, folds each
  run's members together by the run's *within-group* mode
  (`SCREEN`/`ISECT`/`WITHIN_MULTIPLY`, none set = union/max), then composites the
  finished sub-mask onto the accumulator with the run's between-group operator
  **once per run**.

Migration reuses the classic point list verbatim, so a run of N same-operator
elements collapses from N applications to one. The 48-brush exposure edit splits
into element 0 (`op=0`) and elements 1-47 (all `op=SUM`); classic sums 47 times
and reaches 0.6202, flexi max's them to ~0.1 and sums once, reaching 0.1723.

This only bites operators that are not idempotent-under-union. `max` is
associative *and* idempotent, so a union run folded then union'd once equals
each element union'd individually -- which is why classic union and flexi union
agree and no union-only edit differs. `SUM` (a+b), `DIFFERENCE`, `EXCLUSION`,
and `INTERSECTION` (min(acc, max(e1,e2)) != min(acc,e1,e2)) all differ.

**Fix shape, verified**: force every non-union element to start its own run
(`dt_masks_point_group_t.group_start = 1`), which restores per-element
application exactly. Probed in the verifier over the 27 failing edits:

| | result |
|---|---|
| current migration | 27 DIFFERENT, worst 1.0 |
| non-union elements split into own runs | **27 identical, worst 2.98e-08** |

Consecutive union elements can stay merged (max is idempotent), so the panel
does not degenerate into one group per stroke for the common case.

**Open, and the reason this was not landed here**: persistence.
`dt_masks_write_masks_history_item()` is a plain `INSERT`, not an upsert, so the
modified group cannot be written in place at migration time. The drawn-only path
currently takes the immediate route (`needs_new_form` is false for it precisely
because it changes no forms); making it modify the group means routing it
through the deferred writer (`dt_masks_finish_flexi_migrations`), which knows the
final `history_end` and runs before `dt_masks_read_masks_history()`. That is a
behavioural change to every drawn-only edit and needs a load/save round-trip
test the current harness does not have -- the verifier replays with
`history_num = -1` and never persists, so it can only confirm the render half.

The migrate_legacy.c comment on the drawn-only path ("flexi renders a drawn group
through the exact same code path as classic ... already correct with no form
changes at all") is wrong and should go with the fix.

**Corpus incidence** (2292 edits with a classic drawn group):

| operator | element occurrences | edits containing one |
|---|---|---|
| UNION | 3708 | 1235 |
| SUM | 2037 | 153 |
| DIFFERENCE | 341 | 158 |
| INTERSECTION | 289 | 125 |
| EXCLUSION | 3 | 2 |

355 compared edits carry a non-union operator. **27 mis-render today**; the other
328 escape only because their shapes do not overlap -- latent, not safe.

### Two pre-existing GPU issues, separated out

- **89 edits where migration *fixed* the GPU.** All mask_mode 9 (raster) or 5,
  all with refinements: the CL raster branch publishes the host-side `mask`,
  which never received the device-side post-processing, so the published raster
  mask is missing its refinements. Migration clears the RASTER bit, the tail
  copies back from the device, and the result becomes correct. Pre-existing,
  classic-only, and not caused by this branch — recorded, not fixed here.
- **2 colorequal raster edits** where the migrated GPU mask diverges by 1.0 with
  the CPU clean. Not yet diagnosed.
- **206 edits with a large CPU/GPU gap on the classic edit too** (colorequal
  dominating). The verdict logic correctly ignores these — the gap does not
  widen — but the GPU comparison is effectively vacuous for them, so they should
  not be counted as GPU coverage.

### Standing

**Not ready for testers.** The blocking item is the per-element operator gap:
355 edits carry it, 27 mis-render outright, and the failure is silent — the edit
loads, the mask looks plausible, and the module applies in the wrong places.

Process note, and the point of the whole exercise: this was found only by
rendering the same edits through a *second independent implementation*. A single
implementation plus a cache validated itself for 2466 edits and reported zero
differences.
## §42 — The round trip: does a migrated mask survive being saved?

§41 closed with the gap `--verify-masks` cannot reach by construction: it
replays in memory with `history_num = -1` and never touches the database, so
state that is right in memory and then lost on the way to disk is invisible to
it. `--roundtrip-masks` closes that.

Per edit: seed a scratch image with the harvested **classic** history and forms,
read it through the real `dt_dev_read_history_ext()`, simulate a mask edit,
write through the real `dt_dev_write_history_ext()`, read again, compare.

It compares *state*, not pixels, and that is not a weaker test: --verify-masks
already establishes that a given (blend_params, form tree) renders the same mask
as its classic original across the corpus. The only open question is whether
that tuple survives a save -- so a state diff, which names the field that moved,
is the right instrument.

**Result: 2429 round-tripped, 2429 unchanged, 0 different, 0 errors, 37 skipped
(already-flexi).**

### Three defects in the test, each of which made it pass while testing nothing

1. **`dev->iop` carries no blend_params after a read.** They are only written
   onto modules when the history stack is *popped*. Every snapshot came out with
   zero module lines, so the comparison was between two empty lists. Now
   snapshots `dev->history` -- which is also the object
   `dt_dev_write_history_ext()` actually persists, so it was the right thing to
   compare regardless.
2. **`multi_priority > 0` rows were silently dropped.**
   `dt_ioppr_get_iop_order()` returns `INT_MAX` for a second instance absent
   from the default order, and `dt_dev_read_history_ext()` `continue`s past the
   row. This is the "cannot get iop-order for ... instance N" line that had been
   filtered out of every log this session as noise; it was load-bearing.
   Instance is irrelevant to what this measures, so it is normalized to 0.
3. **`dt_dev_write_history_ext()` writes the history item's OWN forms snapshot**
   (`dt_dev_history_item_t.forms`), not `dev->forms` -- and a freshly-read stack
   has none. Calling it straight after a load wipes `masks_history` and writes
   nothing, so the test reported the normalization as lost when it had never
   been offered for saving.

Point 3 also corrects the standing description of Option B. The markers reach
the database only through a **mask-touching** edit: `_dev_add_history_item_ext()`
snapshots `dev->forms` only when `include_masks` is set, and of the public
wrappers only `dt_dev_add_masks_history_item_ext()` passes it. An ordinary
parameter edit appends an item with `forms == NULL` and persists no masks. This
is still correct -- an unsaved edit re-derives the markers from the classic
blend_params on every load -- but it is narrower than "the user edits
something".

### The invariant check, and why comparing the two loads was not enough

Load #1 against load #2 catches state that *changes* across a save and nothing
else: a migration producing the same wrong tree twice passes it. That is a real
blind spot, because the two loads take different paths -- the first migrates
from classic blend_params, the second finds them already flexi and no-ops -- and
the interesting failure is one of them silently doing nothing.

So both loads are also checked against the invariant the §41 fix establishes:
every non-union member heads its own run. Scoped to groups reachable from each
module's own `mask_id`, after a first version checking all of `dev->forms`
reported a violation on a group no module references -- `dev->forms` is
per-image and every masks_history row is a cumulative snapshot, so it routinely
holds other modules' groups and ones orphaned by earlier edits.

**Mutation-verified**, both halves:

| mutation | result |
|---|---|
| `dt_masks_normalize_flexi_groups()` made a no-op | fires, "violated after load" |
| the call moved to *before* `dt_masks_read_masks_history()` | fires, "violated after load" |

The second pins the ordering that Option B rests on -- the mistake it is
designed to avoid, caught.

### Standing

- CPU + GPU replay: 2466 edits, **0 CPU differences**, 91 GPU-only (89 = migration
  fixing a pre-existing CL raster bug, 2 undiagnosed colorequal raster).
- Round trip: **2429/2429 unchanged**.
- Generated raster matrix: 288/288.
- Unit suite: 205 tests, 9 mask suites, all passing.

Remaining before testers: the 2 undiagnosed colorequal raster edits, and GUI
testing (selection/solo/DnD).
