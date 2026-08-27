# Flexi masks panel — behavioural tests

Headless regression tests for the flexi mask panel's *behaviour*: grouping,
drag-and-drop, selection, and cache invalidation. They run in ~0.2s, need no
display, and are the safety net for refactoring the panel.

```sh
cmake -B build-test -S . -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-test -j8
ctest --test-dir build-test -R flexi --output-on-failure
```

| Suite | Covers |
|---|---|
| `test_flexi_model` | grouping and partitioning, element drag-and-drop, the selection state machine, operator normalisation, solo/mute primitives |
| `test_flexi_cache` | which edits must — and must not — invalidate the pixelpipe's mask cache |
| `test_flexi_persistence` | mask blob version migration, i.e. carrying already-saved edits forward |
| `test_flexi_dnd` | the drop paths other than element-onto-element: group headers, staged groups, clusters, whole-group reorder |
| `test_flexi_groups` | the solo family's mutual exclusivity, group numbering, refinement scope |
| `test_flexi_panel` | what the panel shows: warning badges, adaptive parametric rows, preferences |
| `test_flexi_migrate` | the classic → flexi migration case table (structure, not pixels) |
| `test_flexi_compose` | the mask operators themselves: what union/intersection/… compute, and the algebraic properties the design relies on |

## Why these are unit tests and not simulated GTK

The panel looks like it needs a GUI harness to test. It mostly does not.

Its group model is a plain structure — a `dt_masks_form_t` of type
`DT_MASKS_GROUP` whose `points` list holds one `dt_masks_point_group_t` per
element, bottom-up, partitioned into groups by `group_start` (see
`_starts_group`). Every gesture the panel offers is ultimately a mutation of
that list. The functions that perform those mutations take a mask group and
plain values; the only global they touch is `darktable.develop`, and only to
resolve a formid to a form.

So the mock is a `dt_develop_t` holding a forms list, a module pointing at it,
and a `blend_data` for the panel's scratch state. No `gtk_init`, no display, no
database, no pixelpipe.

Simulating real GTK events against a real widget tree was considered and
rejected: it needs a display in CI, and event-injection tests are flaky enough
that they tend to get disabled rather than fixed — which is worse than not
having them. The seam below buys most of the coverage at none of that cost.

## The seam

A gesture handler is split in two:

- **the GTK handler** decodes the event into plain values (which element, which
  target, above or below) and commits the result afterwards — history, pipe,
  widget rebuild;
- **the model function** performs the gesture, and is what the tests call.

`_masks_row_drag_received` and `_model_drop_element_onto_element` in
`blend_gui.c` are the worked example. The handler owns nothing but decode and
commit, so the tests and the real panel run *identical* logic — there is no
second implementation to drift.

Model functions are declared in `develop/blend_gui_internal.h`. Adding a
gesture to the suite means extracting its handler the same way first.

## Layout strings

Scenarios read as what the panel shows, bottom group first:

```
"u:1,2 | i:3"
```

A union group holding elements 1 and 2, with an intersection group holding
element 3 above it. Operators: `u`nion, `i`ntersection, `d`ifference,
e`x`clusion, `s`um.

```c
dt_masks_form_t *grp = flexi_build("u:1,2 | i:3,4");
_model_drop_element_onto_element(&flexi_module, grp, 1, 3, TRUE);
assert_layout("u:2 | i:3,1,4");
```

`flexi_layout()` serialises through `_starts_group`, the same predicate the
panel and the renderer use — so a layout assertion tests what the user will see,
not what the flags happen to say.

## test_flexi_cache — the invalidation contract

`dt_masks_group_hash()` tells the pixelpipe whether a mask still renders the
same. Two opposite failure modes, both invisible until someone notices the
wrong thing happening:

- a rendering input **missing** from the hash → the edit does not appear (stale
  cached mask; "I moved the slider and nothing happened");
- a non-rendering value **included** → everything recomputes on cosmetic
  changes (renaming a group should not re-render the image).

Neither shows up in a pixel-comparison suite: the rendering is correct, it is
the decision to *re*-render that is wrong. Hence a dedicated suite — hash,
mutate one field, hash again, assert whether the two differ.

This caught `group_opacity` missing from the hash on its first run.

The negative cases matter as much as the positive ones. Solo-*edit* narrows
which shapes are editable on canvas and must **not** invalidate; solo/mute
(`DT_MASKS_STATE_HIDDEN`) changes what the mask renders to and **must**. Those
two are easy to conflate, so they are pinned apart explicitly — as are
selection, cluster collapse/expand, canvas edit mode, group renaming, and empty
group placeholders.

Some of those negative tests look tautological today, because the state they
poke lives in `blend_data` rather than in the group. That is what makes them
worth keeping: they are the tripwire for someone later storing presentation
state inside `dt_masks_point_group_t`, where it would silently start dragging a
full mask recompute behind every cosmetic click.

**When you add a value the group renderer reads, add a test here — and when you
add panel state that it doesn't, add one too.** That is the whole contract.

## test_flexi_persistence — version migration

`dt_masks_legacy_params` carries every already-saved edit forward when the group
point struct gains a field. It deserves its own suite because it runs against
data nobody can regenerate (a user's existing library), it fails silently, and
it is the one place where a zero-filled field is not automatically safe:
appended fields are read at the historic stride and zero-filled, which is
neutral for most of them, but `group_opacity` is multiplicative — a zero-fill
would blank out every pre-v9 group's mask.

The read-time stride selection itself is SQLite-coupled and out of reach here;
these tests cover the migration chain that runs after it.

## test_flexi_compose — what the operators mean

`_combine_masks_*` in `masks/group.c` is the arithmetic behind every operator
name the panel shows. Until this suite it was only ever checked end-to-end, by
rendering an image and comparing pixels — which proves the pipeline agrees with
itself on the fixtures it has, but does not pin what an operator *means*, and
cannot state the properties the rest of the design leans on:

- **the group-fold operators are commutative and associative.** This is the
  entire justification for treating a group as an unordered bag of shapes, and
  for letting the panel reorder members freely within a group. Difference is
  asserted *not* to be — it is a between-group operator, where order is the
  user's choice.
- **each operator's identity element.** An empty group contributes nothing,
  which the compositor implements by skipping it. These tests pin the
  arithmetic that makes skipping the right choice — in particular that
  intersection's identity is an all-*one* mask, so an empty intersect group
  could never be allowed to composite as all-zero and blank the whole mask.
- **every operator keeps the mask in [0,1]**, across opacity and invert, for
  every input in range.
- **operator dispatch**: `_flexi_apply_group_op` routes each state bit to the
  right operator, and an operator-less state falls back to union.

## Remaining gaps

The six gaps this suite started with are closed. What is left needs machinery
the fixture does not have:

- **Undo/redo interaction.** A stale `bd->empty_groups` placeholder previously
  duplicated a group on undo. Reproducing it needs a real history stack, so it
  belongs in an integration test rather than here.
- **Anything requiring the database.** The read-time stride selection in
  `dt_masks_read_forms_ext` (which picks how many bytes of each stored point to
  read, per masks version) is SQLite-coupled; only the migration chain that runs
  *after* it is covered. The deferred migration path
  (`dev->pending_flexi_migrations`, taken when `history_num >= 0`) is likewise
  out of reach -- the tests drive the inline path with `history_num = -1`.

## Not covered — manual checklist

These are properties of GTK rather than of the panel's logic, and need a real
widget tree and real event delivery:

- **event propagation between nested widgets.** A handler returning `FALSE` on
  a child bubbles to an ancestor carrying the same handler, firing it twice.
  This caused a real bug (group-header clicks toggling back on release, fixed
  by the `_event_on_own_window` filter). Check: click a group header, an
  element row, an element's editor body, and empty space — each selects or
  deselects exactly once.
- **CSS rendering.** `.mask-group-block` borders and selected-group shading,
  and the single drop-indicator line between two groups.
- **widget packing**, tooltips, drag icons, panel relocation between
  embedded / utility / left / right positions.


## Adding a test

1. If it is a gesture, extract its handler into a model function first
   (see **The seam**) and declare it in `blend_gui_internal.h`. Keep history,
   pipe and widget work in the handler — the model half must not commit.
2. Write the scenario as a layout string.
3. **Prove the test can fail** — break the code deliberately and watch it go
   red before committing it. A test that has never failed has not been shown to
   test anything. Two mutations worth knowing:
   - removing the `_group_keys_apply` call in
     `_model_drop_element_onto_element` reproduces the "moving an element
     between two groups creates a third group" bug exactly;
   - removing the solo-edit clear in `_model_toggle_solo_form` breaks the
     solo / solo-edit mutual exclusivity.
4. Check the invariant after **every** step of a sequence, not just at the end.
   The isolation-mode test originally checked only after all three toggles,
   which hid violations whenever the last toggle happened to be the one that
   cleaned up — it now checks after each, across all six orderings.
