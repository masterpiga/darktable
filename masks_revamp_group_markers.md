# Flexi masks: groups as marker entries

Status: steps 1 to 4 of the sequencing below are implemented (steps 1 to 3
on 2026-09-11, step 4 on 2026-09-12, a follow-up on 2026-09-13). The classic
to flexi migration writes markers, the panel and the renderer read groups
only from them, and `group_start` is read only by the migration, from old
data. The GUI has not been tested on markers yet: see
`masks_revamp_gui_test_protocol.md`.

## The problem

A flexi group has no record of its own. It is a run of consecutive members
of a module's `DT_MASKS_GROUP` list, and everything that belongs to the group
is copied onto every member:

- the between-group operator and the within-group combine mode, as bits of
  `state` (`DT_MASKS_STATE_OP`, `DT_MASKS_STATE_WITHIN`)
- `name` (masks v8)
- `group_opacity` (v9)
- a group-scoped `refinement` (`DT_MASKS_REFINE_GROUP`), which shares the
  member's single refinement slot with the member's own element refinement

Where one group ends is inferred: at an operator change, or at a member with
`group_start` set (v10).

Readers then pick one of the copies. The flexi renderer reads the first
visible, resolvable member of the run (`group.c`, `_group_get_mask_roi_flexi`:
the head is chosen at lines 1286-1293, its settings read at 1301-1316 and
1447), so hiding or soloing a member changes which copy is read. The panel
reads the run's bottom member, its cid (`_group_custom_name` in
`blend_gui.c`).

Every mutation has to keep the copies and the inferred partition consistent.
It does not, by construction, and the bugs found on 2026-09-11 are all of
this one class:

- moving an element renamed the group it moved into, and could hand that
  group its old group's opacity and refinement, whenever the element became
  the member the group is read from (patched by `_join_group`)
- moving a group's last element put the group's empty placeholder on the
  wrong side of the element, or swapped two groups (patched by
  `_keep_emptied_group` and `_moves_up`)
- dropping onto an empty group anchored on the element being moved lost the
  landing position, and the element jumped past other groups
- members whose form had gone hid their group, and every empty group
  anchored on them (patched by `_model_prune_dangling_members`)

Two more consequences of the same design:

- empty groups exist only in the panel (`bd->empty_groups`, positioned by an
  anchor, `below_fid`). They are not saved, and undo, a history jump or a
  reload drops them.
- element and group refinements share one slot, so setting a group
  refinement overwrites every member's own (`_refine_commit_nonglobal`).

## The proposal

A group is a marker record in the same list, followed by its members up to
the next marker.

- A marker is a `dt_masks_point_group_t` with a new state bit,
  `DT_MASKS_STATE_GROUP_MARKER`. It refers to no form: its `formid` is an id
  of its own, taken from the same allocator as form ids (`dt_masks_create`:
  `time(NULL) + form_id++`), so it is unique and stable. It is the group's
  identity: panel selection, the group's number and its expanded state key on
  it.
- The marker's existing fields hold the group's settings, once: `state`
  (operator with its disable and invert modifiers, within-group mode),
  `name`, `group_opacity`, `refinement` (group scope). No struct change: a v11
  record is the same size as a v10 one.
- Members keep only what is theirs: the use, show, inverse, hidden and
  disable bits, `opacity`, and their own element refinement, in a slot the
  group no longer shares.
- A module's list, bottom-up: `[M1] a b [M2] c [M3] [M4] d`. `M1` is the
  foundation group and always comes first; `M3` is an empty group.
- The marker is the boundary, so `group_start` and boundary inference go. The
  field stays in the struct only to read old data.

Why not a form of a new type per group, which markers would then refer to: the
v10 to v11 migration would have to create those forms, and the per-version
steps rewrite one form at a time (`dt_masks_legacy_params_v9_to_v10` in
`masks/masks.c` receives a single `dt_masks_form_t` and edits its
`points`). A self-describing record can be inserted by such a step alone.

Why this answers the objections `masks_revamp_data_model.md` raised against a
groups table: it needs no database change (a marker is one more record in the
same blob), and "what happens to a group when its last member is deleted" has
a plain answer: the marker stays. A group goes away only when it is deleted
explicitly, which is also the panel rule.

## What it removes

- The copy discipline: no broadcast on rename, opacity or refinement, no
  reading a group off its first member, no `_join_group`.
- Boundary inference: `_starts_group`, `group_start` and the partition
  snapshot and re-stamp (`_group_keys_snapshot`, `_group_keys_apply`) that
  every reorder and move goes through.
- Panel-only empty groups: `bd->empty_groups`, `below_fid` anchoring,
  `_capture_emptied_group` and its multi variant, `_keep_emptied_group`,
  `_moves_up`, the realize-on-draw path (`insert_realize_empty`,
  `insert_realized_fid`), and `_model_ensure_a_group`, replaced by the rule
  that the list starts with a marker.
- Carrying a group's number across the empty and populated states, since the
  marker id is the group's identity.

For scale, on 2026-09-11: 53 references to `group_start`, 38 to
`_starts_group`, about 200 to empty groups and their anchors, and 92 to
`group_opacity`, most of which become a single read.

## What changes

### Format and migration

- Only the classic to flexi migration matters; edits already stored in a
  flexi format are not migrated (decided 2026-09-11). So there is no
  `dt_masks_legacy_params_v10_to_v11` step.
- Markers are inserted where the migration sets `group_start` today:
  `_split_nonunion_runs` (`masks/migrate_legacy.c:122`, the bit is set at
  141). A marker goes before each run, holding the run's operator, within
  mode, name, opacity and group refinement.
- That runs only on groups queued by the migration, which are flexi by
  construction, so classic groups never carry markers. It covers the live
  tree and every stored history snapshot
  (`dt_masks_normalize_flexi_groups`, `migrate_legacy.c:1350-1371`).
- Flexi edits made before markers (test fixtures, testers' edits on this
  branch) are not supported. They render through the step 2 fallback until
  step 4 removes it, then they need rebuilding.

### Renderer

- In the flexi fold, a marker starts a group and supplies its settings;
  members fold until the next marker, and an empty group contributes nothing.
  Simpler than the run detection it replaces.
- Every other loop over a group's records must skip markers: mask hashing,
  the classic fold, form duplication and copy, `dt_masks_cleanup_unused`, the
  `dt_masks_form_remove` cascades, `dt_masks_group_insert_point`, linking and
  import, persistence, postedit and `migrate_legacy.c`, plus about 30 loops in
  `blend_gui.c`. One predicate (`dt_masks_point_is_marker`) and a member
  iterator keep that in one place. `_model_prune_dangling_members` must spare
  markers.

### Panel

- One header per marker, with the marker id as the group's cid.
- Moving an element is a list move between markers. Reordering a group moves
  a slice (a marker and its members), deleting one removes the slice, and
  reset leaves `[M1]`.
- Empty groups become real: saved, undoable, kept across reload and export.
  That is a behavior change, and the intended one.
- A group-layout preset is a list of markers.

### Tests

- The fixture's layout strings already name groups and empty groups
  (`"u:1,2 | [d] | i:3"`); `flexi_build` would emit markers, so most tests
  read the same.
- The migration ledger: classic edits must render bit-identically through
  the marker migration, as they do through today's.
- The pixel suite's XMPs are classic edits that migrate on load, so they go
  through the marker migration unchanged and must match the current PNGs.
  (The flexi-native `I1_two_adjacent_intersect_groups` this once named was
  already gone; I1 is now a classic chain. J7/J8 separate two union groups
  with the pre-v10 `GROUP_BREAK` bit, which v9 to v10 carries into
  `group_start`, so the migration keeps reading that field from old data.)

## Sequencing

Each step builds and renders identically:

1. The marker bit, `dt_masks_point_is_marker`, and every loop skipping
   markers. No marker exists yet, so this changes nothing. Done. Most loops
   needed nothing: they look a member's form up and skip what does not
   resolve. The ones that did: the mask hash (hashes a marker's state, group
   opacity and refinement, not its name), group copy and duplication (keep
   markers, under a fresh id from `dt_masks_new_marker_id`), "use same
   mask as" (`dt_masks_group_add_members_of`), and the two used-form
   collections (`_cleanup_unused_recurs` in `masks.c`, `_fill_used_forms` in
   `common/history.c`), whose table has one slot per form, so a stored marker
   id crowded out a member, which was then deleted. Form ids are checked
   against marker ids too (`_check_id`), and `_model_prune_dangling_members`
   spares markers. No member iterator was needed.
2. The renderer reads a group's settings from its marker when there is one,
   and from the run's head as today otherwise. Done
   (`_group_get_mask_roi_flexi`: a marker starts a group and the next one
   ends it; a list may mix both forms).
3. The classic to flexi migration inserting markers, the panel reading
   groups from them, and empty groups as markers. Done.
   - `dt_masks_group_mark_runs` gives an unmarked list one marker per run,
     taking the settings from the run's first visible member, and an empty
     list a union marker. The migration calls it from `_normalize_group`
     (repair, split, mark) on the live tree and every history snapshot, with
     ids derived from the group and run head (`_run_marker_id`) so the same
     edit gets the same ids every time. The panel calls it on reconcile for
     lists that reach it unmarked.
   - Members kept their copies of the group bits, so a migration that fails
     closed still rendered classic over the marked tree. Dropped with step 4,
     see there.
   - Group settings (operator, within, opacity, name, refinement, bypass) are
     written to the marker only, keyed by the marker id. Moves, reorders,
     deletes and merges are list and slice moves; the empty-group
     placeholders, group keys, captures and join helpers are gone from
     `blend_gui.c` and `blend.h`.
   - A module with no mask group shows one phantom union group; the group
     form and its marker are created on the first write
     (`dt_masks_module_group_create`). Adding a group now records a history
     item.
   - Group-layout presets store and apply markers; their per-group
     "opacity" now means group opacity.
   - postedit's canon normalization is gone, so its "already normalized"
     check holds for every edit and no longer tells anything. The check was
     removed in the step 4 follow-up.
   - The harnesses addressed members by list position. On the benp harvest
     that silently emptied `--persist-masks` coverage: element steps landed
     on the marker (`elem-opacity` 872 live to 0, `geom:size` 158 to 0) and
     `structural:remove` deleted the marker instead of an element. The
     shared step code (`_resolve_scope`, `postedit.c`) now skips markers for
     element scopes, and a group break in a marked group inserts a copy of
     the enclosing marker (`_break_before`). The round-trip run invariant
     (`_check_group_runs`, `roundtrip.c`) skips marked groups, whose
     operator is on the marker. After the fix, benp's element and geometry
     live counts match the baseline exactly in both persist and undo.
   - Two structural counts rose, legitimately. The harnesses skip a removal
     when a group has fewer than two records, and a one-shape group now has
     two (marker and shape), so removing its only shape now runs and leaves
     an empty group (`remove`: 124 live to 894 in undo). A reorder past the
     list's first marker would put a member in no group, which the panel
     cannot produce, so that step is skipped.
   - Verified with `--check-masks` on seven harvests (benp, mino, zisoft,
     kofa_2, phemisters, macchiato17, gwbarn): round-trip, style-apply,
     persist and undo pass on all. Verify's DIFFERENT rows are the classic
     CPU vs OpenCL divergence: migration widened the GPU gap on no edit, the
     classic fold over the marked tree never changed, and the worst CPU
     difference is 1.4e-4. benp and zisoft flag the same edits as their
     committed reports.
   - The flexi pixel suite is 38/46 both with and without step 3; the eight
     refinement scenarios drift by at most 1/255 from references generated
     2026-09-09, so that predates this work.
4. Removing the copy, boundary and placeholder machinery, and `group_start`
   use. Done.
   - The flexi fold has no unmarked path: a marker starts a group and the
     next one ends it. Members a list holds before its first marker (only a
     flexi edit stored before markers has them) fold as one plain union
     group. Such edits are not supported and render differently than they
     did.
   - `dt_masks_group_mark_runs` split in two. `dt_masks_group_ensure_marker`
     gives a list that does not start with a marker a union one, and is what
     the panel, adding a shape and group-layout presets call; it infers no
     boundary. `dt_masks_group_mark_classic_runs` is the migration's run
     partition, at operator changes, at `group_start` from old data, and
     with `split_nonunion` at every non-union member. That absorbed
     `_split_nonunion_runs`, which wrote `group_start`, so nothing writes the
     field any more. `dt_masks_point_breaks_run` is gone.
   - `group_start` left the mask hash, the panel's list signature, and the
     `dt_masks_form_remove` cascade, which moved a break to the next member.
   - Migration is one way (decided 2026-09-12): its result has to render
     like classic, not to stay readable by classic. So the marking moves a
     group's settings off its members, which become plain union elements
     (operator, within mode, name, group opacity, group refinement and
     `group_start` cleared). The one migration step that could still fail
     after normalizing, `_migrate_drawn_and_parametric`, now normalizes past
     its last failure point, so a failed migration never leaves classic
     params over a stripped tree. The harness checks that re-rendered the
     migrated tree through the classic fold (`--verify-masks`
     `classic_restore_*`, `--persist-masks` `classic_over_stored_*`) are
     gone, since that is no longer a property.
   - postedit's group break only inserts a marker (`_break_before`); the
     group-start poke and its snapshot field are gone. The round-trip run
     invariant now asks that every group start with a marker and every
     non-union member sit right above its marker.
   - Verified: the 11 flexi unit suites pass; the pixel suite is 38/46 as
     before (the same eight refinement drifts, J7 off by 9 pixels, not by
     the 16,877 a merged group moves); `--check-masks` on benp passes
     round-trip (1740 unchanged), style-apply, persist and undo, and verify
     still flags one edit, as the earlier report did.
   - With members stripped, verified again: the 11 flexi suites pass, the
     pixel suite is unchanged (38/46, same pixel counts), and
     `--check-masks` on the seven harvests (benp, mino, zisoft, kofa_2,
     phemisters, macchiato17, gwbarn) passes round-trip, style-apply,
     persist and undo on all. Verify flags 1, 0, 1, 3, 0, 17 and 27 edits;
     every one has a CPU difference of 0 (at most 3.7e-6), so all are the
     classic CPU vs OpenCL divergence. benp (1695) and zisoft (131) flag the
     same edits as their earlier reports.
   - Follow-up, 2026-09-13. Migration has no failure path: the allocation
     checks, the fail-closed returns and the "mask kept in classic mode"
     message are gone, and `dt_masks_migrate_classic_to_flexi` returns void.
     `--postedit-masks` is removed: once migration writes the markers, its
     from-scratch side was the migrated group itself, so it skipped every
     edit. `postedit.c` keeps only the panel steps that `--persist-masks`
     and `--undo-masks` share. The flexi pixel README and `gen_xmp.py` no
     longer describe I1 as a flexi-native `group_start` fixture.

## Costs and risks

- **Loops that miss a marker fail silently.** 51 loops walk a group's
  `points`, and several places test or count them: removing a group when
  `points == NULL` (`masks.c:2063`, `masks.c:2126`, `blend_gui.c:10568`),
  and member counts (`blend.c:381`, `postedit.c:428`, `postedit.c:478`,
  `masks.c:2254`). A marker-only list is never NULL, and markers would count
  as members. `_model_prune_dangling_members` would delete every marker.
  Step 1 has to audit all of these.
- **The struct is overloaded.** A marker's `formid` is not a form id, and its
  `opacity` and `parentid` mean nothing. Only the state bit says which kind
  of record it is.
- **Empty groups change the hash.** Being saved, they appear in undo steps and
  invalidate the mask cache. A reset module keeps a one-marker list, so "has
  a mask" cannot be read off an empty list.
- **Stored flexi edits break** after step 4 (see Format and migration): an
  edit with no markers renders as one union group, and the panel shows it so.

## Open questions

- **The now-unused group fields on members.** Resolved 2026-09-12: the
  migration clears them (see step 4).
- **Marker ids and form ids.** A marker id must never be resolved as a form.
  Within one image nothing hands out a clash: new forms skip marker ids
  (`_check_id`), and new markers skip form ids (`dt_masks_new_marker_id`,
  `_run_marker_id`). The way in is another image: pasting history or
  applying a style copies the source's forms under their own ids
  (`dt_history_merge_module_into_history`, `common/history.c:540-561`), and
  nothing checks them against the target's markers. Ids are seconds since
  1970 plus a per-session counter, so both images would have to have issued
  the same number. Not observed, and what a clash breaks is not traced.
- **Linked forms.** Links share forms between modules, not member records, so
  markers should not affect them. To verify.
