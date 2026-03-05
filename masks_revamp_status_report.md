# masks_revamp — status report (basis for efficiency/optimization phase)

Written 2026-07-09 from the current state of the `masks_revamp` branch (code, not just
`WORKLOG_masks_revamp.md`, which is stale past its §40 entry — this report folds in the
unlogged fixes from the most recent working session too).

## What this branch is

A new **Flexi mask** editing mode (`DEVELOP_MASK_FLEXI`, additive mask-mode bit) lives
alongside darktable's existing drawn/parametric/raster mask modes. It replaces the flat,
order-dependent list of shapes-with-their-own-operator with an **ordered list of groups**
(each group owns one compose operator: union/intersect/difference/sum/exclusion), rendered
as a dedicated panel inside each mask-capable module's blend UI
(`src/develop/blend_gui.c`, ~11,300 lines, the overwhelming majority of this branch's diff).
Classic drawn/parametric/raster editing is untouched — every new code path is gated on the
`DEVELOP_MASK_FLEXI` bit or on `bd->masks_list_box` existing, so legacy edits render
byte-identically (no masks-version bump beyond the pre-existing v7 per-shape refinement
block, which this branch also reuses/extends).

## Core data model (all additive, no schema bump)

- **Groups are inferred, not stored.** A "group" is a maximal run of adjacent
  same-operator points in the existing flat `grp->points` list. No new persisted group
  entity.
- **First-class group identity (§35)** was later bolted on top via a stolen bit:
  `DT_MASKS_STATE_GROUP_BREAK = 1<<11` on a point's `state` (serialized as a raw blob, so
  old edits round-trip as 0 = neutral) forces a group boundary even when the operator
  matches the run below — needed once two adjacent same-operator groups became allowed
  (e.g. two independent single-channel parametric masks both set to "union"). All group
  boundary detection funnels through one predicate, `_starts_group()`.
- **Empty groups are UI-only state**, never serialized: `bd->empty_groups` (a `GList` of
  `dt_masks_empty_group_t {op, screen, below_fid}`), anchored to a real member id so they
  keep their position as shapes are added/removed. This is explicitly called out in the
  worklog as the one piece that doesn't fit the "no schema" constraint if the feature set
  grows further (persistent multi-slot scaffolding).
- **`DT_MASKS_STATE_MULTIPLY`** (`1<<10`) and **`DT_MASKS_STATE_SCREEN`** (`1<<9`, the
  optional smooth "screen" within-group combiner, `a+b−ab`) are new compose primitives.
  New flexi renderer branch: `_group_get_mask_roi_flexi` in
  `src/develop/masks/group.c` — unions (or screens) a group's members once, refines once
  (broadcast per-shape refinement, reusing masks-v7 storage), then composites into the
  result once via the group's operator. Classic sequential fold is the unchanged original
  code, selected when the flexi bit is absent.
- **Single-channel parametric masks (§36, §39):** a parametric form can now be pinned to
  exactly one blendif channel (`dt_masks_point_parametric_t.single/channel/in_out/invert`,
  repurposed "reserved" struct words → same size, neutral for legacy blobs). Several
  single-channel forms intersect-grouped is mathematically identical to one classic
  multi-channel AND. Inversion is coupled to the shared per-shape invert bit rather than a
  second polarity flag.

## UI surface built (chronological, condensed from the 40-section worklog)

1. Per-module mask list (rows: operator | invert | name | mute | solo), DnD reorder,
   rename, right-click delete.
2. Composite operators incl. new "multiply"; first-visible-as-add rule.
3. New-shape default-operator selector.
4. Canvas↔list hover sync, then reworked twice more (§13, §32) into a persistent
   selection + transient hover model, bidirectional.
5. Flexi as a distinct mode/tab, implicit convert from classic (shared group), panel
   split so classic stays the vanilla two-row toolbar.
6. Parametric forms as first-class group members: per-form inline blendif editor docked
   under its row, independent config per form, several "resync stomps edits" bugs fixed
   (form is the single source of truth; scratch `bp` re-derived from it, not vice versa).
7. Shape clustering (same-kind runs ≥3 fold into an expand/collapse group to tame
   many-brush-stroke lists) — reworked at least three times (adjacent-only → whole-list
   count-based, expanded-by-default → collapsed-by-default this session).
8. Mask refinement panel (§15) made **scope-aware**: global / all-shapes / per-parametric,
   later replaced by **selection-driven per-group** targeting (§22, §31) with a bypass
   toggle (transient, non-serialized, never enters history).
9. Full group-composition UI: add-group button, group headers (operator chip, screen
   toggle, mute/solo/solo-edit, drag handle), interior insertion, realize-on-draw for
   empty groups, merge-confirm dialog on same-op drag, group renumbering
   (`"<op>-<id> · <count>"`), permanent foundation (bottom) group.
10. Explicit drag-handle column (§33/§34) after discovering child event-boxes were
    swallowing press events before the row's own drag source could arm — a real GTK DnD
    correctness fix, not cosmetic.
11. Terminology pass: "hide/show" → mute/solo (mixer metaphor), since the toggle removes
    an element from the composite rather than just hiding an overlay.
12. §40 (most recent logged): collapsed the separate master/detail "elements" panel back
    into a nested tree — each group's own element rows now sit indented directly under its
    header in one flat `masks_list_box`, dropping a whole extra section + selection
    indirection.

## Unlogged fixes from the most recent session (not yet in WORKLOG, now verified in code)

- **CSS group-block spacing** (`data/themes/darktable.css:792-801`): settled on
  `margin-top: 4px` only (no `padding-top`) + `:first-child { margin-top: 0 }`. Earlier
  iterations either produced inconsistent gaps (sibling-selector approach) or made the
  border visibly stretch past the header (padding sits inside the border, margin sits
  outside it — was conflated).
- **Crash class fixed at 8 call sites**: synchronously calling `_build_masks_list(module)`
  (destroys + rebuilds the whole widget tree) from inside a GTK event handler firing *on*
  the widget being destroyed caused a confirmed SIGSEGV in `dt_shortcut_dispatcher` /
  `gtk_widget_event`. Fixed by deferring via `g_idle_add(_rebuild_masks_list_idle, module)`
  instead of a direct call. **There are now 18 `g_idle_add(_rebuild_masks_list_idle, ...)`
  call sites** and only 8 remaining direct `_build_masks_list(module)` calls in the file
  (grep-verified) — the direct ones are presumably reached from contexts proven safe
  (top-level callbacks, not mid-dispatch on the rebuilt widget), but this split is worth
  auditing wholesale rather than case-by-case in the next phase.
- **Double/triple-rebuild "flash" on delete**: `dt_masks_form_remove()` (masks.c) already
  triggers its own full-panel rebuild internally via `dt_masks_iop_update` →
  `dt_iop_gui_update_blending` → `_build_masks_list`, once *per shape removed*. Callers in
  `blend_gui.c` that looped over several removals and then rebuilt again caused N+1
  rebuilds for one user action. Fixed with a new suppression guard,
  `bd->masks_rebuild_suppressed` (`blend.h:619`), set around such loops in
  `_group_delete_shapes`, `_group_reset_members`, `_name_button_press`, checked at the top
  of `_build_masks_list`.
- **Cluster (kind-fold) behavior**: default-collapsed (was default-expanded — inverted
  hash-table-lookup default), chevron click no longer self-cancels (press consumed by the
  arrow but its matching release wasn't, so it bubbled to the header and toggled a second
  time — fixed with a dedicated release-consuming handler,
  `_element_cluster_arrow_release`), and right-click now deletes the whole cluster
  (reuses `_group_delete_shapes` with the cluster's own `"hover-formids"`).

## Architectural facts most relevant to the upcoming efficiency/optimization phase

- **`_build_masks_list` is a full destroy-and-rebuild of the entire panel's widget tree**,
  not an incremental diff. It is the single choke point almost every mutation (add,
  delete, reorder, mute/solo, invert, operator change, refinement scope, cluster
  toggle...) eventually routes through — currently invoked from ~26 call sites (8 direct +
  18 deferred via `g_idle_add`). For lists with many shapes/groups this means every click
  anywhere in the panel tears down and reconstructs dozens-to-hundreds of GTK widgets
  (buttons, event boxes, revealers, drag sources/targets, CSS-classed boxes) plus
  re-walks `grp->points` multiple times per rebuild (once for the groups pass, once per
  group for the elements pass, once more for cluster-folding within each group).
- **`blend_gui.c` is ~11,300 lines**, up from a small fraction of that pre-branch — nearly
  all of it is this one file. `_build_masks_list` and its helpers
  (`_pack_group_elements`, `_make_shape_row`, `_pack_empty_group_header`, the DnD
  handlers) are the hot path for literally every panel interaction.
- **Selection/hover sync is push-based and runs on every rebuild**: `_update_row_selection`,
  `dt_iop_gui_masks_hover_form`, `_dock_editor_under` each linearly scan the just-rebuilt
  `masks_list_box` tree (recursively, since §32/§40's nesting) to find the row matching a
  formid, rather than keeping a formid→widget map.
- **`g_idle_add`-deferred rebuilds** are correctness fixes (avoid use-after-free on the
  widget mid-dispatch) but stack an extra main-loop round-trip on top of an already
  expensive full rebuild; worth checking whether any of the 18 sites fire multiple times
  per single user gesture (e.g. drag-and-drop) and could coalesce.
- **23 `dt_dev_add_masks_history_item` call sites** — each is a full history-item commit
  that (per existing darktable architecture) triggers a pipe reprocess; several of the
  worklog's own bug fixes (§26 interior insertion, §37 group-drag merge fix, §38 emptied
  group capture) were about *correctness* of what gets committed, not about reducing how
  often a commit fires. Worth checking for any remaining redundant multi-commit path
  analogous to the delete-flash bug that was just fixed.
- **Runtime CSS**: the inline `gtk_css_provider_load_from_data` block
  (`_ensure_mask_row_css`) was moved out to `data/themes/darktable.css` in §32 — CSS is no
  longer rebuilt/reloaded at runtime, which is good, but confirm no residual per-rebuild
  style-context churn (`gtk_widget_get_style_context` calls, class add/remove) inside the
  hot rebuild path.
- **Recursive tree walks post-§40 nesting**: since elements now nest under group headers
  (rather than a flat list), every "find row by formid" helper is a tree walk instead of a
  flat list scan — worth profiling with a realistically large mask (many groups × many
  shapes each) rather than assuming it's fine at small N.

## Suggested starting points for the optimization phase

1. Profile `_build_masks_list` under a mask with many groups/shapes (it's the single
   highest-traffic function in the new code); decide whether incremental widget updates
   for common cases (mute/solo toggle, single-row rename) are worth the complexity versus
   keeping full-rebuild-but-cheaper (e.g. widget pooling/reuse instead of destroy+recreate).
2. Audit the 8 remaining direct `_build_masks_list(module)` calls versus the 18 deferred
   ones for consistency — either all mid-dispatch-reachable calls should defer, or
   document why the remaining direct ones are safe, so the next contributor doesn't have
   to re-derive it from a crash report.
3. Look for more `dt_masks_form_remove`-style "hidden internal rebuild" traps before they
   cause another N+1 flash — anything in `masks.c`/`group.c` called in a loop from
   `blend_gui.c` is a candidate.
4. Replace the formid→widget linear/recursive scans (`_find_row_by_formid` and friends)
   with a hash map maintained alongside the rebuild, if profiling shows it matters.
5. Decide the empty-groups persistence question flagged since §24/§26 (session-scoped
   UI-only vs. real schema field) before building more on top of it — it's the one part of
   the model that isn't "free" neutrality-wise.

## Neutrality guarantee (unchanged, worth re-verifying before optimizing)

Every mechanism above is gated on `DEVELOP_MASK_FLEXI` and/or on fields that are zero for
every pre-existing edit (masks-v7 refinement block, `GROUP_BREAK`/`SCREEN`/`MULTIPLY`
state bits, `single`/`channel`/`in_out` parametric fields). Classic drawn/parametric/raster
masks take the original, untouched code paths. Any efficiency work should preserve this —
in particular, don't let a "shared helper" refactor accidentally pull classic mode through
new flexi-only branches.
