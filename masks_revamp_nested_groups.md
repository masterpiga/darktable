# Nested groups in flexi masks

Status: decided 2026-09-13. Phases 0 to 1 done; phase 2 (panel display)
implemented 2026-09-14, not yet tried in the GUI.

## Decision

Flexi masks support nested groups. A group's member can be another group,
and that group can hold anything a top-level group can: shapes, parametric
channels, raster elements, AI objects, other groups. Nesting is a feature,
not a migration artifact.

This replaces the "never nest" flattening experiment. Flattening could not be
exact in general: a flat fold has one accumulator, and a classic mask such as
`(sky u cloud) - tree  u  (face u hair) - glasses` needs the left side stored
while the right side is built. Nesting is that storage. The experiment's
numbers, for the record: 7,182 identical, 2,763 equivalent and 8 visibly
different out of 10,123 affected edits, where "identical" only means the
shapes happened not to overlap in a structurally wrong place. The patch is
kept outside the tree.

In the harvest corpus (13 libraries, classic edits with a mask group), nesting
depth below the top group is 0 in 18,190 edits, 1 in 1,578, 2 in 98 and 3 in
5. Nothing is deeper.

## Stored format: no change

A nested group is already representable, the same way classic stores it: a
member whose form is a `DT_MASKS_GROUP`. That group's own `points` list is a
flexi list with its own markers. The member point's opacity, inversion and
element refinement apply to the subgroup's finished result.

Nothing new is persisted, so no masks version bump.

## What already handles nesting

| Piece | Where | Notes |
|---|---|---|
| Renderer | group.c:1355, group.c:1489 | a group member renders through `dt_masks_get_mask_roi` into the flexi fold, recursively |
| Member invert and opacity on a subgroup | group.c:1367-1387 | inversion, then opacity, as classic does |
| Pixelpipe hash | masks.c `dt_masks_group_hash_ext` | recurses into member groups |
| Marker ids | masks.c `_run_marker_id` | checked against every form of the image, so unique across levels |
| Refinement and bypass keys | group.c:1361, group.c:1416 | keyed by marker or form id, so unambiguous across levels |
| Marking classic runs | masks.c `_mark_runs` | recursive, depth guard 8 |
| History paste | history.c `_fill_used_forms` | copies nested forms recursively |
| Parametric colorspace lock | blend_gui.c:2397 | recursive, depth guard 16 |
| Canvas edit overlay and hidden state | blend_gui.c:5639 | the overlay is flattened through nested groups |

## Gaps

1. **Raster consumer walks are one level deep.** A raster element inside a
   subgroup would not keep its source publishing:
   `_pipe_has_raster_form_consumer` (imageop.c:4230) and
   `_raster_form_consumes` (pixelpipe_hb.c:751).
2. **Other one-level walks.** There are 62 `for(... ->points; ...)` loops over
   group members: masks.c 24, blend_gui.c 17, group.c 11, imageop.c 3, and 2
   each in blend.c, pixelpipe_hb.c and migrate_legacy.c, plus history.c 1.
   Each is either right to stay one level (it edits one list) or has to
   recurse. This is an audit, not a rewrite.
3. **Depth guards disagree** (8 in `_mark_runs`, 16 in the colorspace lock).
   Use one constant everywhere, together with a cycle guard.
4. **Sharing.** Forms belong to the image, not to a module, so two parents
   could reference the same group. The panel keys rows by form id
   (`bd->masks_row_map`), so a shared subgroup would also break the panel.
5. **The panel assumes one flat list.** There are 76 `_module_mask_group()`
   calls, 41 `_group_point(top, cid)` lookups, 23 `_starts_group()` walks
   and 46 reads of `panel_selected_group_cid`. `_masks_panel_pack`
   (blend_gui.c:14846) packs one block per group of the top list only.
6. **The roundtrip invariant.** `roundtrip.c` must accept a member that is a
   group, as long as every list at every level starts with a marker.

## Plan

Each phase builds and passes the suites on its own.

**Phase 0: migration keeps nesting. Done 2026-09-13.** The flattening is out
of the tree; migration marks runs at every level. There were no code changes
to stage, since the experiment was never staged. Results:
- 11/11 flexi suites pass.
- `--verify-masks` over the 127 hard edits: 59 identical, 31 equivalent, 0
  different, 37 skipped (already flexi). The worst CPU difference is 1.5e-6.
- `--roundtrip-masks` over the hard edits: 90 unchanged, 0 different.
- `--verify-masks` over the 10,123 affected edits: 7,191 identical, 2,762
  equivalent, 31 different, 139 skipped (already flexi). All 31 are identical
  on the CPU, within 3.7e-6, and differ only on the GPU. Classic already
  diverged there, and none of the gaps widened. That is the pre-existing
  classic OpenCL divergence.

`--check-masks` over all harvests has not been re-run for this phase.

**Phase 0b: migration dissolves exactly. Done 2026-09-14.** Marking a
classic group (`dt_masks_group_mark_classic_runs`, masks.c `_mark_runs`)
now replaces each nested group it converts by its own groups wherever the
three exact rules in Q2 apply (`_dissolve_member`), innermost first. Only
groups that had no markers when the pass began count as converted, so a
group nested on purpose in flexi stays nested, and a classic group nested in
two places dissolves in both (Q1). The drawn + parametric wrapper is
normalized from its top group, so the drawn group dissolves into it.
`--verify-masks` reports each edit's `nesting` after migration. Results:
- 11/11 flexi suites pass, with 7 new migration tests.
- Hard set: 57 identical, 33 equivalent, 0 different, 37 skipped; worst
  CPU 4.7e-6. `--roundtrip-masks`: 90 unchanged.
- Affected set (10,123): 7,189 identical, 2,764 equivalent, 31 different
  (GPU-only, CPU within 3.7e-6: the classic OpenCL divergence), 139
  skipped. The worst CPU difference, 1.4e-4 at edit 6930 (filmic rgb,
  drawn + parametric), is the same as in phase 0 and does not come from
  dissolving: with dissolving switched off it is bit-for-bit the same. With
  the mask refinement removed, classic and migrated differ by 6e-8 (float
  rounding); the feathering guided filter amplifies that to 1.4e-4, still
  invisible and within the equivalent tolerance.
- Depth, measured: nothing deeper than one level. 82 of 9,980 edits keep a
  subgroup: the 63 predicted, plus 19 whose nested member is disabled by
  the base-case repair (migrate_legacy.c `_repair_base_case_overwrite`),
  which is kept as it is. The measured depth matches the prediction on every
  other edit. The 19 predate 2026-09-16, when the repair changed from
  disabling the discarded members to deleting them: those subgroup references
  are now gone, so the figure needs re-measuring (the 63 are unaffected).

`--check-masks` over all 13 harvests was re-run after phase 1, which
includes this phase; see the phase 1 results below.

**Phase 1: model. Done 2026-09-14.** Results, with phase 0b included:
- `--check-masks` over all 15 harvests (63,157 edits; leonidas and dudo
  re-harvested and added to the corpus the same day): every library passes
  all five checks. The ledger's headline is 0 migration failures in 8,203
  shapes, below 0.037% at 95% confidence, with 66 classic-GPU outliers
  counted apart.
- Two check rules changed to get there, both because dissolving is exact
  and the old rule was not. The round-trip invariant no longer requires a
  non-union group to hold a single member (roundtrip.c
  `_check_group_runs`): dissolving legitimately makes multi-member
  difference groups, and verify covers what the rule was for. Verify judges
  the GPU only by whether migration widened the CPU/GPU gap;
  classic-GPU against migrated-GPU is reported but no longer fails an edit,
  since a classic OpenCL outlier is not migration's (akgt94: 32 different
  before, 0 after).
- The suite now runs in parallel from the corpus database:
  `tools/masks_corpus.py check data/masks_corpus.db OUTDIR --darktable BIN`
  cuts every library into chunks and runs one `--check-masks` process per
  chunk, then merges the per-library reports the ledger reads. All 13
  libraries took 64 minutes on 12 processes. Its merged reports match
  whole-library runs field for field on the 12 libraries that have one,
  except style-apply's mask ids, which differ between any two runs.
- Done: `dt_masks_group_find_marker(forms, root, cid, &owner)` returns the
  marker of a group at any depth and the group form holding it. The panel's
  `_group_point(top, cid)` calls move to it in phase 2, with the rest of the
  panel.
- Done: one constant, `DT_MASKS_NESTING_MAX` (8), in masks.h, and every
  recursive walk stops there. Before this, none of these had a limit, so a
  cyclic tree recursed until the stack ran out: the renderer
  (`dt_masks_group_get_mask` and `_get_mask_roi`, guarded by a per-thread
  count since it recurses through the functions table), the hash, ungroup,
  cleanup, history paste, the canvas hidden-state walk, the raster-sources
  hash, the host-guides test, the drawn-shape test, the mouse-action types,
  the mask counts, `dt_masks_is_in_module` and `_find_in_group`. The walks
  that had a limit (8 or 16) use the constant now.
- Done: raster consumers are found at any depth, through
  `dt_masks_group_find_raster_of`: the pipe's consumer test (imageop.c), the
  prune (pixelpipe_hb.c) and the users registration
  (`_reconcile_raster_form_users`). The detail-buffer test
  (`_blend_group_wants_details`) recurses too; one level deep, a subgroup's
  details refinement would have rendered without its detail mask.
- Done: the audit. Of the 68 member loops, the rest either edit one list (the
  panel's editing operations, phases 2 and 3), run on the flattened canvas
  copy (group.c events), belong to AI objects (Q6), or visit every form
  already (`_id_taken`, the legacy upgraders).
- Done: model tests for a marker in a subgroup, a raster element in a
  subgroup, and a cyclic tree that every walk leaves.
- Done: the ownership rule (Q1), one parent within a mask, applied by
  migration; tested.

**Phase 2: panel display. Implemented 2026-09-14; not yet tried in the GUI.**
- One lookup for the whole panel: `_group_point` and `_point_node` find a
  point at any depth, through `dt_masks_group_find_node` (masks.c), which
  also returns the group form whose list holds it. Every read keyed by id
  (selection, opacity, names, bypass, the group of an element) works at any
  depth without touching its call site. Every edit of a list acts on that
  holder: adding, deleting, emptying and merging groups, detaching members,
  and moving elements, clusters and groups.
- A move stays within one list. A drop across nesting levels is refused, so a
  group can never land inside itself; moving between levels is phase 3.
- Display: a member that is a nested group gets its own row (group icon,
  opacity, a chevron), with its groups packed under it by `_pack_group`, as
  the top list's are (`_pack_subgroup`, stopped at `DT_MASKS_NESTING_MAX`).
  The chevron is open unless closed by hand, and always while the selection
  is inside. A group holding the selection or the auto-expand anchor in a
  nested group opens too, and "auto-expand selected" never closes it
  (`_reveal_nesting`, `_collapse_auto_expanded_group`). Nested group rows
  never fold into same-kind clusters.
- Solo: `dt_masks_group_isolate_state` recurses. Soloing inside a nested
  group keeps the path to it, and soloing a whole nested group clears what an
  earlier solo hid inside it. A nested group another module's mask shares
  (Q1 keeps those links) is isolated for that mask too.
- New shapes land in the selected group at any depth
  (`dt_masks_group_insert_point`).
- Groups are numbered across the whole mask, nested ones included, and "the
  last group stays" counts the groups of one list (`_level_group_count`).
- The all-shapes refinement reaches nested shapes and skips the nested
  group's own member; the rebuild signature and the import menu's shape lists
  include nested members.
- Rows reach nested members everywhere the panel walks the mask: a shape's
  expanded properties and its shrink/grow (they walked the top list only,
  so a nested shape showed nothing but shrink/grow, and a nested row's
  opacity could not change), the in-place row refresh after solo, the
  low-opacity badge (with the gain of every enclosing group), raster names,
  dropping members whose form is gone, and canvas selection of an AI object.
  A row edits its form in the canvas's flattened copy at the index
  `_canvas_points` gives, the order `dt_masks_group_ungroup` flattens in:
  a nested group or an AI object takes no index of its own. The top-list
  count it replaces was off by one for each such member below, flat masks
  with an AI object included.
- Tests: 10 new model tests (test_flexi_model.c, "nested groups in the
  panel"); 11/11 flexi suites pass.
- `--check-masks` over all 15 harvests (`tools/masks_corpus.py check`, 12
  processes, 71 minutes): every library passes all five checks, and every
  report is identical to phase 1's in round-trip, verify, persistence and
  undo. The run used the build before the row fixes above, which none of
  the checks call.
- Not verified: anything seen in the GUI (layout, indentation, expanders,
  canvas hover on a nested group row).

**Migration: a run of differences is one group. Implemented 2026-09-14.**
Classic applies a difference once per shape, so marking gave each
difference shape a group of its own: an upstream AI object with 11 holes
became 12 groups. `_merge_difference_runs` (masks.c) folds a run of
one-member difference groups into one group whose members fold by screen.
That's exact: difference composites acc * (1 - g), and a screen fold makes
1 - g = (1 - x1)(1 - x2)..., which is what classic computes shape after
shape, inverted shapes and opacities included. A union fold would not be:
max is exact for hard edges only. Plain groups only (no fade, invert,
disable, refinement, within bits or name), and never onto the bottom group
of a list, whose operator is not applied. Classic only (`split_nonunion`);
flexi lists stored before markers are untouched. 3 model tests.
`--check-masks` over all 15 harvests: every library passes all five checks.
8 edits render differently from before, all still identical to classic:
float rounding of the screen fold against the sequential product, at most
2.9e-6 (thad 21165, 37 pixels), the rest at most 1.2e-7. Consecutive
differences are rare in the corpus; upstream AI objects, where they are the
rule, are not in it.

**Phase 3: panel editing. Started 2026-09-14; not yet tried in the GUI.**
Done:
- Shift+click on the add-group menu adds a nested group on top of the
  selected group's members, holding one empty group of the chosen operator,
  which is selected (`_model_nest_new_group`). Refused, with a message,
  inside a nested group.
- Dropping a group with shift held puts it inside the group under the
  pointer, as the one group of a new nested group (`_model_nest_group`); the
  target lights up whole while shift is held. A plain drop reorders.
- Elements and groups move across nesting levels. A move is refused where a
  nested group would land inside itself, where it would nest deeper than
  `PANEL_NESTING_MAX` (1: mask > group > nested group, Q2), or where a group
  would leave its list without one (`_may_move_into`).
- 5 model tests.
Still to do:
- "Group selected": wrap the selected elements or groups into a new subgroup.
- "Ungroup" (see Q3).
- Delete a subgroup: the nested group row's own "delete" detaches it like any
  element, which should already cover it; not checked in the GUI.

**Phase 4: checks and docs.**
- A nesting axis in the UI coverage check.
- Nested fixtures for `--persist-masks` and `--undo-masks`.
- `--check-masks` over all harvests: done after phase 1; re-run after each
  later phase with `tools/masks_corpus.py check`.
- A nesting section in masks_revamp_gui_test_protocol.md.
- Update masks_revamp_group_markers.md.

## Decisions and open questions

- **Q1. Sharing: decided.** A group has exactly one parent. Nesting a
  group that is already referenced elsewhere duplicates it. This is also
  what keeps the panel's row map valid.

  Classic data shares more than that, measured over the 13 harvests (19,963
  mask roots, 1,740 of them with nesting):
  - within one mask: 11 masks hold the same nested group twice (12 groups;
    christian_pfister, kofa_1, thad). Phase 0b copies such a group wherever
    it dissolves; one kept nested stays shared.
  - across modules: in 514 images two module instances share a group, as
    their mask or nested in it (592 groups). Classic lets a module use
    another's shapes, so editing the group in one changes the other.

  Decided 2026-09-14: one parent within a mask. Linking shapes from other
  modules stays: a group shared between two modules' masks is not copied,
  and editing it still changes both. Migration copies a nested group one
  mask still holds twice after dissolving (masks.c `_unshare_nested`); the
  panel copies a group nested where the mask already holds it (phase 3).
- **Q2. Depth: proposed, one level: mask > group > subgroup.** A subgroup
  cannot hold subgroups. Measured on the 13 harvests (19,731 classic drawn
  edits, 5,170 of them drawn + parametric, where migration wraps the drawn
  group as the first member of a new top group):

  | Levels below the mask | 0 | 1 | 2 | 3 |
  |---|---|---|---|---|
  | as migrated today, edits | 12,962 | 6,590 | 172 | 7 |
  | after exact dissolution, edits | 19,668 | 63 | 0 | 0 |
  | after exact dissolution, images | 9,120 | 32 | 0 | 0 |

  Exact dissolution replaces a nested group by its own groups when that
  provably leaves the mask unchanged:
  - it forms a single run: it becomes one group, its fade the group opacity
    and its inversion the group's invert output;
  - it is first in its parent: its runs are spliced in when its fade and
    inversion push down exactly (fade through union, intersection,
    difference and multiply; inversion through union, intersection,
    difference, multiply and screen, by De Morgan);
  - all its runs share its own associative operator (intersection, sum or
    multiply) and it is not inverted.

  The 63 remaining edits (26 distinct structures) are of three kinds:
  - an inverted drawn part holding a sum;
  - a faded nested group holding a sum or an exclusion;
  - two multi-step operands joined together, such as `[o] u[o do]`.

  Each of them needs exactly one subgroup. The analysis is algebraic, done
  by a static script over the stored structures, not by a render. Its
  wrapping test approximates migration's parametric branch by "any blendif
  channel active". To reach one level, migration has to do this dissolution:
  phase 0b. Deeper, malformed trees stop at `DT_MASKS_NESTING_MAX`
  (phase 1).
- **Q3. Ungroup: decided, always offered, with a warning** when dissolving
  the subgroup into its parent would change the mask. The flattening
  experiment's algebra computes whether it does.
- **Q4. Solo on a nested group: decided.** The mask becomes that
  subgroup's result alone, as soloing a top-level group does today.
- **Q5. Layout: decided, indentation.** A subgroup expands in place.
- **Q6. AI objects as groups** (raised with Q5): whether `DT_MASKS_OBJECT`
  should become an ordinary subgroup, so it needs no special handling.
  Proposal: yes.

  **It already is a group in all but name.** A committed multi-path object's
  points are group points. It renders through the group renderer
  (object.c:2297-2298). Hashing, paste, ungroup, the colorspace lock and
  raster-source hashing already treat it like a group (masks.c:106, 613,
  2563, 2620, 2682; blend.c:474; blend_gui.c:2415).

  **Upstream stores AI results as plain groups.** Upstream's object tool
  commits a `DT_MASKS_GROUP` of paths, with holes as difference members
  (master object.c:1039-1060), and never persists `DT_MASKS_OBJECT`. So
  upstream AI masks already arrive as nested groups, which phase 0 handles.
  The bundle type exists only in edits made on this branch; converting
  those at load needs no upstream format change.

  **Two likely bugs go away, neither verified by a render:**
  - Holes probably render filled in flexi. Objects never get markers, and
    the flexi fold reads an unmarked list as one union group and ignores the
    members' own difference state (group.c:1285, group.c:1380-1387).
  - A build without AI renders an object as nothing. Its function table is
    set only under `HAVE_AI` (masks.c:1285), and `dt_masks_get_mask_roi`
    returns 0 without one (masks.h:834).

  **Where each object-only behavior goes:**

  | Behavior | Today | As a group |
  |---|---|---|
  | creation session (clicks, preview, smoothing, cleanup, threshold) | object.c | stays a tool; it commits a group, as upstream does |
  | a single path is committed bare | object.c:1043 | unchanged |
  | coordinated feather, size, rotation, grow/shrink | object.c:1872-2115 | "transform the group as a unit", for any group. Grow/shrink flips the sign for difference members, as now. Size scales every member uniformly about the shared center, replacing the object's inverse scaling of holes |
  | enter on double-click to edit single paths | `entered_object` | entering any subgroup on the canvas. The panel's paths-only-when-entered rows (blend_gui.c:13985) go away: indentation shows members when expanded |
  | "AI object #n" name and icon | object.c:938, blend_gui.c:11444 | the name stays as the group's name; the row icon is lost unless kept as a flag |
  | shared-object tooltip | blend_gui.c:8246 | moot under Q1 |
  | shapes cannot be added to it | `dt_masks_group_add_form` gate | lifted: a brush can patch a segmentation |

  About 40 `DT_MASKS_OBJECT` special cases go (blend_gui.c 17, masks.c 16,
  others 7). This fits after phase 2, since it needs the nested display and
  the group transform. Open: keep an AI icon on such groups, or not.
- **Q7. What a nested group is: decided 2026-09-14, order-free half
  implemented 2026-09-16.**
  Phases 0 to 3 built a nested group as a list of groups, a small mask of
  its own. That shows levels that are not groups: a migrated AI object
  appeared as "union-1 > ai object > difference-1, union-2", where the inner
  two exist only to encode the object's signs.

  Decided model:
  - the top level stays a list of groups, applied in order with their
    between-group operators;
  - a nested group is an element with a within-group operator and no
    between-group operator. The containing group's within-group operator
    combines it with its siblings, as for any element;
  - every within-group operator is order-free (union, screen, intersection,
    multiply, plus a new sum), so no element order matters below the top.

  Classic's per-member operators map onto that exactly (group.c:1001-1140):

  | Classic | Nested form |
  |---|---|
  | union, intersection | union, intersection group |
  | difference `A - B` | multiply { A, inverted screen group { B... } } |
  | sum | a new within-group sum: `min(1, a + b)` in any order equals classic's per-step clamp |
  | exclusion `A x P` | union { multiply { A, inverted P }, multiply { P, inverted A } } |

  The difference form is exact for faded holes: the holes' opacities apply
  inside the screen group, before its invert (group.c:1427-1438), giving
  classic's `acc * (1 - o * x)` (group.c:1081). An element's own invert
  applies opacity after inverting (group.c:1012), so a faded operand is
  wrapped in a group rather than inverted directly. Exclusion was chosen
  over a two-element exclusion mode to keep one rule for every within-group
  operator; its cost is that both operands appear twice, a nested operand as
  a copy.

  An AI object becomes `multiply { outer paths, inverted screen { holes } }`.

  Corpus (26,283 distinct edits): nested groups in 1,393. Inside a nested
  group: difference 225, union 251, intersection 10, sum 5, exclusion 1
  (kofa_2 exposure, edit 9094: a nested group at 0.35 whose chain ends in an
  exclusion; the 0.35 cannot be folded into the blend opacity, since that
  edit's brightness 0.76 makes the tone curve nonlinear, blend.c:662-682).
  Top level, handled by the ordered stack: exclusion 40, sum 556.

  Implemented 2026-09-16, the order-free half: a NESTED classic group whose
  members all carry one operator is marked as a single group carrying the
  matching within-group operator, instead of one group per run
  (`_uniform_within` / `_mark_runs` in masks.c). Intersection maps onto
  `ISECT`, sum onto the new `DT_MASKS_STATE_WITHIN_SUM` (1 << 19, folded as
  `min(1, a + b)` in group.c); union needed nothing, since an all-union list
  is already one run. The top level still stays a list of groups applied in
  order with their own operators. A group collapsed this way usually then
  dissolves away entirely through `_dissolve_member`'s one-group rule, so
  edit 5667's "grp 6981 > union-2, sum-1" ends up one flat group holding its
  two shapes, the member's inversion becoming the group's invert-output.

  Measured over the 26 picked corpus edits: every one still migrates
  render-identical, and the three sum cases (5667, 5673, 17473) drop from one
  nesting level to none. The panel gained a "sum" entry in the within-group
  combine selector.

  Difference and exclusion followed on 2026-09-17, in `_rewrite_nested_runs`
  (masks.c). A nested group's runs are folded left into within-group forms:
  union/intersection/sum/multiply/screen extend the accumulator (wrapping it
  into a group of its own when the mode changes), difference becomes
  `multiply { acc, inverted screen { holes } }`, and exclusion
  `union { multiply { acc, inverted p }, multiply { p, inverted acc } }`.

  Two things made it smaller than the model above suggests. A member reference
  at opacity 1.0 carrying `DT_MASKS_STATE_INVERSE` *is* the referenced group's
  inverted output, since the combiners compute `opacity * (1 - newmask)` -- so
  one form is referred to twice, plain and inverted, instead of needing an
  inverted copy. And the arithmetic is pinned directly in test_flexi_compose
  (`test_a_faded_hole_is_a_multiply_of_an_inverted_screen_group`, the run-of-
  holes case, the exclusion identity, and a negative control proving that
  inverting the *element* is a different mask at fractional opacity).

  The wrapper rule. A drawn+parametric migration wraps the drawn group in a
  top group of its own, so the user's real top level sits one level down.
  Rewriting it there turned flat classic masks two levels deep (edit 781's
  `o xo` went from nesting 0 to 2), so `_is_synth_wrapper` recognizes that
  group -- synthesized parametric channels plus one group reference -- and
  does not count it as a level.

  Measured over the 26 picked corpus edits, all migrate render-identical
  except edit 9094, the corpus's only nested exclusion, which is `equivalent`:
  max deviation 5.7e-06 over 80 pixels (mean 9.3e-09) against a threshold of
  one 8-bit step, i.e. float reassociation from expressing exclusion as a max
  of two products rather than classic's branchy form. `--roundtrip-masks`
  passes on all seven libraries (26 edits, 0 different, 0 errors), which is
  what shows the synthesized forms survive a save and reload.

  The cost is depth: the wrapper runs go, but a difference adds a screen group
  and an exclusion four, so edit 6400 goes from nesting 1 to 2 and edits 20583
  and 9094 to 4.

  Still open: the depth cap. `PANEL_NESTING_MAX` is 1, so those deeper trees
  display but the panel's own nest/move operations refuse targets below it.

  Panel display, 2026-09-17: a nested group shows as a group, with the same
  header as a top-level one, except its lead icon is the nested-group icon
  and fixed (no between-group operator: its holder's within-group operator
  combines it). Before, it showed as an element row with its own group header
  under it, which read as an extra level. The header's "delete group" removes
  the nested group from its holder, as the element row's delete did.
  (`_nested_as_group`, `_sole_nested_group` in blend_gui.c.)

  For the header to show everything, migration moves a nested group
  reference's opacity and inversion onto the group's marker
  (`_fold_nested_refs`, masks.c). Exact because a member contributes
  `o * (inverted ? 1 - m : m)` to every within-group combine, and a group's
  sub-mask is inverted then scaled by group_opacity: uninverted the opacities
  multiply, inverted it holds over a group at full opacity
  (test_flexi_compose `test_a_nested_reference_folds_onto_its_group`). Left
  alone, and shown with the old element row: an inverted reference over a
  faded group (`o * (1 - g * x)` has no single-marker form), a refined or
  hidden reference, a nested group holding several groups, and a group
  another reference shares. References from a classic group migration just
  dissolved do not count as sharing: the emptied form stays in `forms`
  holding its old list. The 26 picked edits still migrate as before
  (25 identical, 9094 equivalent).

  Only groups and elements, 2026-09-17. The panel must show nothing but
  groups and elements, so a nested group is always one group, referred to by
  a plain member. The leftovers above were not a limit of the model: in
  classic a group has no settings of its own, its reference holds them, so
  they always map onto the converted group's own settings. They came from
  migration's order. Its final fold ran after dissolving and the rewrites had
  already given the group settings of its own, and a drawn group under the
  parametric wrapper was kept a list of groups even where it could not be
  spliced into the wrapper. Now (masks.c):
  - a classic group referenced more than once, in this mask or by another
    form, gets a copy per reference before anything is converted
    (`_unshare_nested` with `classic_elsewhere`);
  - a reference's opacity, inversion and refinement move onto its group as
    soon as that group is converted, while its marker is still plain
    (`_move_ref_settings`, called from `_mark_runs`);
  - the difference rewrite inverts the holes group itself, and the
    exclusion rewrite refers to inverted copies of its operands, so neither
    creates an inverted reference (`_synth_ref` takes no inversion any more);
  - dissolving into a nested list only replaces a member by one group
    (`_dissolve_member`'s `nested`), so a nested group stays one group;
  - a drawn group under the parametric wrapper that cannot be spliced in is
    rewritten into one group like any nested group, takes its reference's
    settings, and is spliced in as that group where exact.
  The panel's "add group" above or below a nested group now adds a nested
  sibling beside it instead of a second group inside it
  (`_model_add_group`).

  Checked: `--roundtrip-masks` now also fails an edit whose nested group
  holds more than one group or whose reference carries settings
  (roundtrip.c `_check_group_runs`). Over all 15 libraries (62,893 edits) it
  found 22 such edits with the old order, all nested groups of 2 to 5 groups
  and all drawn groups under the parametric wrapper, and 0 with the new one.
  Tests: test_flexi_migrate `test_nested_settings_become_their_groups`
  (fails on the old order) and `test_a_nested_exclusion_has_no_inverted_reference`;
  test_flexi_model `test_a_nested_group_gets_a_nested_sibling`. The 26
  picked edits render as before (25 identical, 9094 equivalent); 22361 keeps
  one more level, since a nested list no longer splices several groups in.
  Full `masks_corpus.py check` (all 15 libraries, 63,157 edits) passes:
  verify 62,571 identical, 322 equivalent, 0 different, 0 errors; round-trip
  62,893 unchanged, 0 different, including the new rule. Every summary count
  matches the previous full run.

  One-element groups and names, 2026-09-18 (reported on nest_11, kofa_2
  e9094, the corpus's nested exclusion). Two groups each held only 6683.
  Migration now replaces a plain reference to a nested group holding one
  element by that element where exact (masks.c `_collapse_single_members`):
  every within-group combine of one member is that member, so the group
  contributes `g * (inv ? 1 - m : m)`; uninverted that is the element at
  `g * o`, inverted (element at full opacity) the element with its inversion
  flipped. It is refused for a refined, bypassed or named group, a hidden
  member, and a parametric or raster member, whose no-op and unresolved
  cases the fold treats differently from a group. A faded hole keeps its
  inverted screen group: `1 - o * x` is not `o * (1 - x)`. Of the 26 picked
  edits, all render as before and seven keep fewer levels (6399, 6400,
  20583, 22361 one fewer; 6859, 21491, 22357 none left). The duplicated
  operand in an exclusion stays: `a x b` is
  `union { multiply { a, inverted b }, multiply { b, inverted a } }`.

  Group names follow the within-group mode, at every level: "multiply-2",
  "screen-1", "intersect-1" instead of the between-group operator, which a
  nested group does not have (`_within_index_for_state`,
  `_group_ord_max_for_within` in blend_gui.c). Numbers are per mode, and a
  group keeps its number when its mode changes, as it did when its operator
  changed.
  Full `masks_corpus.py check` passes with both: verify 62,571 identical,
  322 equivalent, 0 different; round-trip 62,893 unchanged; every summary
  count the same as the run before.

  Dragging a nested group by its header, 2026-09-17: its sole group has no
  sibling to reorder against, so the header was no drag source and the
  nested group could not move (reported in the GUI). A first fix gave the
  header the element row's payload; group-level targets then read it as
  "move into this group", so a nested group could not become a top-level
  group, and the drop could land somewhere other than the line drawn.

  Now the header of a nested group shown as its group is a group drag
  source, and every group drop goes through `_model_move_group`:
  - with shift, into the target group: a nested group as an element, any
    other group wrapped in a new nested group (`_model_nest_group`)
  - beside a nested group shown as its group: a nested group reorders as
    an element of the holder; any other group is wrapped in a new nested
    group beside it (`_wrap_group`). Before, it was reordered into the
    nested group's own list, which then held two groups and fell back to
    the unnamed element row (the "placeholder")
  - beside a top-level group: a nested group becomes that group
    (`_unnest_group`). The group keeps its own opacity, inversion and
    refinement; its between-group operator, unused while nested, becomes
    union. Only for a plain reference, which is what a nested group shown
    as its group has: other references show as element rows, whose
    settings apply on top of the group's own
  - otherwise the old `_masks_reorder_groups`
  Over an element row, a nested group's drag draws the element line and
  lands beside that row (`_drags_as_element`). Model tests: test_flexi_model
  `test_a_nested_group_moves_out_as_its_group`,
  `test_a_faded_nested_group_stays_nested`,
  `test_a_group_beside_a_nested_group_is_nested`,
  `test_a_nested_group_moves_as_an_element`.

  Union runs, 2026-09-17: dissolving a faded nested group puts its fade on a
  group of its own, one per member, so edit 6803 (panel case link_04) showed
  five one-shape unions at 36%, one shape listed in two of them. After
  marking, migration now merges consecutive groups that each fold as a plain
  maximum and join by union into one, multiplying each group's opacity into
  its members (`g * max(o * x) = max(g * o * x)`, `_merge_union_groups`), and
  then drops, within a union, a repeated shape's weaker reference
  (`max(o1 * x, o2 * x)`, `_drop_dominated_refs`; same inversion, no
  refinement). Not merged: a group with invert-output, bypass, a within-group
  mode, a group refinement, a name, or a parametric or raster member (a
  channel left at full range is not counted by the fold, so its group is
  skipped where, merged, it would add 1.0). Only on a tree converted by that
  migration, and not for the drawn+parametric wrapper.

  Then, across a run of groups joined by union, a shape in a plain maximum
  group goes when another group of the run holds it at least as strongly
  (group opacity times member opacity, same inversion, no refinement) and
  folds by union, screen or sum with nothing but its opacity applied to the
  result, since each of those is never below any member
  (`_drop_absorbed_refs`). Edit 17473 (link_06), `x u {x, sum y}`, renders
  `min(1, x + y)`: its lone x goes, with its group. A run starts at the bottom
  group or after a group joined by anything but union; that group is not part
  of it. Between two plain maximum groups a tie keeps the earlier copy.

  Settled 2026-09-16: one shape may appear twice in a mask, and the panel
  treats each appearance as its own row. 57 classic edits do it (39 across two
  groups of a mask, 18 twice in one group). The panel used to key rows by form
  id alone, which showed as a row painted from another reference's state, rows
  that did not follow one another's edits, and a geometry edit applied once per
  reference (a relative property such as size then compounded). Rows are now
  indexed per reference and tagged with the member point they were built from
  (blend_gui.c `_masks_row_for_point`); a geometry edit applies once per shape
  and refreshes every row showing it.

  Links: a shape shared with another module and a shape held twice by one mask
  are the same thing seen twice, so both carry the chain icon
  (`_model_form_uses_in_mask`). There is no owning reference -- all N are
  equal, as cross-module links are. Unlink acts on the row it was opened on
  (`_model_unlink_form_point`): that reference gets an independent copy and
  every other reference keeps the original, including when another module is
  also linked. Form-id-keyed panel state (selection, solo, expanded) moves to
  the copy only when no other reference to the original is left.
