
## §20 -- Phase 3: run/broadcast primitives for group member state

The roadmap's "~12 hand-rolled attribute broadcasts" resolve to two recurring
shapes, now one function each:

```c
void dt_masks_group_set_state(dt_masks_form_t *grp, GList *formids,
                              dt_masks_state_t bits, gboolean set);
void dt_masks_group_isolate_state(dt_masks_form_t *grp, GList *formids,
                                  dt_masks_state_t bits);
```

`set_state` is the plain broadcast (set/clear one bit across a run's members);
`isolate_state` is solo (clear on the named members, set on everyone else).

Five call sites converted, -32 lines net in `blend_gui.c`:

| site | was | now |
|---|---|---|
| `_toggle_solo_form` (off) | loop clearing HIDDEN on all | `isolate(grp, NULL, HIDDEN)` |
| `_toggle_solo_form` (on) | loop, `formid == id ? clear : set` | `isolate(grp, one, HIDDEN)` |
| `_toggle_solo_group` (off) | loop clearing HIDDEN on all | `isolate(grp, NULL, HIDDEN)` |
| `_toggle_solo_group` (on) | nested O(n*m) membership loop | `isolate(grp, members, HIDDEN)` |
| `_toggle_soloedit` (drop solo) | loop clearing HIDDEN on all | `isolate(grp, NULL, HIDDEN)` |
| `_group_op_apply` (bypass) | loop, set/clear OP_BYPASS | `set_state(..., OP_BYPASS, set_bypass)` |
| `_group_toggle_output_invert` | loop, set/clear OP_INVERT | `set_state(..., OP_INVERT, set_invert)` |

### What was deliberately NOT converted

- **`_toggle_element_disable`** -- one bit on one point, four lines. Routing it
  through a list primitive would mean allocating a GList to toggle a flag. Left
  alone.
- **`_group_op_apply`'s non-bypass branch** -- `state = (state & ~OP) | op` is a
  *field replacement*, not a bit broadcast: the operator is a value living in
  the `DT_MASKS_STATE_OP` bits. Wrapping both under one "set state" name would
  have hidden that difference. It stays hand-rolled, with a comment saying why.

### Two things worth recording

**The roadmap says `group.c`; the right file is `masks/masks.c`.** `group.c` is
the render/vtable side (`dt_masks_group_get_mask_roi`, `_render_roi`); the
structural group API (`dt_masks_group_add_form`, `dt_masks_group_ungroup`) lives
in `masks/masks.c`. The new primitives went next to their actual siblings.

**A NULL member list is not "no members".** The first version of
`isolate_state` looped `member ? clear : set` with no special case, so
`formids == NULL` made *every* point a non-member and would have set HIDDEN on
the entire group -- precisely inverting "solo off", the most common call. Caught
before building, but it is the kind of edge a "pure refactor" is expected not to
have, and the asymmetry is now stated in both the header doc and the body.

### One deliberate behaviour change

The old broadcasts iterated `formids` and resolved each id via
`_group_point(grp, id)`; the primitives iterate `grp->points` and test
membership. These agree unless a formid appears twice in `grp->points` (which
should not happen), where the old code updated only the first occurrence and the
new code updates both. That is the more correct of the two.

### Verification

Build clean -- zero diagnostics in `blend_gui.c`, `masks.c`, `masks.h`,
`raster.c` (the 349 warnings in the log are pre-existing, mostly libxcf).
Flexi suite 37/37.

**The suite proves nothing about this change.** All five converted sites are GUI
gestures, and HIDDEN is a canvas-display bit that does not reach the export
pipe at all. Green here means only "mask evaluation still works". Needs hand
testing: solo an element, solo a group, solo-edit while a solo is active (must
drop the solo AND restore every element's visibility), group bypass toggle,
group output-invert toggle -- and in particular that turning any solo *off*
restores visibility rather than hiding everything.

## §21 -- Phase 3: raster source pre-resolution -- ITEM WITHDRAWN

The roadmap asks to replace `raster.c`'s per-evaluation source lookup with a
pre-resolved pointer. Reading the code, there is nowhere correct to put it and
nothing to gain:

- The obvious cache, `module->raster_mask.sink.source`, is **not this form's
  source**. It is set by `dt_iop_commit_blend_params` from
  `blend_params.raster_mask_*`, which stays reserved for the exclusive
  whole-mask RASTER mode. Nothing anywhere writes those fields on behalf of a
  raster *form* -- I grepped the tree; the only other reader is
  `migrate_legacy.c`.
- Per-form dependency registration already happens at commit time, in
  `_reconcile_raster_form_users` (`imageop.c:2104`), which supports **several**
  raster elements per module each naming a different source.
- The only per-form place to cache a pointer would be
  `dt_masks_point_raster_t`, which is serialized to XMP. Not an option.
- The cost being optimised is one `dev->iop` walk per raster form per pipe
  *evaluation* (not per pixel) -- negligible against the mask render it guards.

So the lookup stays. What was actually wrong here was the **file's own header
comment**, which claimed the dependency is "wired through the module's
blend_params raster_mask_* fields, kept in sync with the (single, first-cut)
raster element by the mask-list UI". All three claims are false: not those
fields, not the mask-list UI, and not single. Rewritten to describe
`_reconcile_raster_form_users` and to say explicitly that
`module->raster_mask.sink.source` is *not* the form's source, since that is the
trap the roadmap item itself fell into.
