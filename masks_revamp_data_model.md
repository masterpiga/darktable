# Flexi masks: data model

## Summary

A module's mask is a tree of groups. Each group folds its members in list
order with one operator, then applies its own refinement, invert and
opacity. A member is a drawn shape, a single parametric channel, a raster
mask reference, an AI object or another group. Every kind of member yields
a 0-1 value per pixel and combines the same way.

Everything is stored in structures master already has. There is no database
schema change.

## Storage, against master

**Blend params (v14 → v15, same layout).**
- `mask_mode` gains `DEVELOP_MASK_FLEXI` (`1 << 4`). With it, `mask_id`
  names the mask group and the classic blendif fields are cleared.
- `raster_mask_*` stay on the params: the pipe's raster dependency wiring
  reads them.
- `mask_lock` takes over the first reserved field. Every legacy conversion
  clears it, since the reserved field was never guaranteed zero.

**Form types.** Two are new, each with its own point struct in `blend.h`:
- `DT_MASKS_PARAMETRIC`: one blendif channel (`single`, `channel`), with its
  ranges and the colorspace it was made in.
- `DT_MASKS_RASTER`: a source module op, an instance and a mask id.

`DT_MASKS_OBJECT` (AI segmentation) is stored as a group, and its list
starts with a marker like any other group's.

**Group point (masks v6 → v7).** `dt_masks_point_group_t` grows from 16 to
176 bytes. The new fields are appended to the struct:

| Field | Meaning | Old data |
|---|---|---|
| `refinement` | feathering, blur, contrast, brightness, details; `enabled` is off, element or group scope | zero-fill: off |
| `name[128]` | group name | zero-fill: none |
| `group_opacity` | multiplies the group's finished mask | the v6 → v7 step sets 1.0 |

The reader reads each stored point at its version's stride and zero-fills
the rest (`dt_masks_point_stride`).

**State bits.** All new bits were unused on master:

| Role | Bits |
|---|---|
| marker: this point is a group record, not a member | `GROUP_MARKER` 18 |
| within-group operator, on the marker (none = union) | `ISECT` 12, `SCREEN` 9, `WITHIN_MULTIPLY` 15, `WITHIN_SUM` 19, `WITHIN_DIFFERENCE` 20, `WITHIN_EXCLUSION` 21 |
| group modifiers, on the marker | `OP_INVERT` 16, `OP_DISABLE` (bypass) 14 |
| member flags | `HIDDEN` 8 (solo), `DISABLE` 17 |
| classic member operator, read only by migration | `MULTIPLY` 10, beside classic's `UNION`...`SUM` |

Static asserts in `masks.h` keep the three roles (classic operator and group
modifiers, group operator, member flags) from overlapping.

## The tree

- **Group.** A `DT_MASKS_GROUP` form whose point list starts with a
  **marker**. The marker has an id of its own (`dt_masks_new_marker_id`) and
  refers to no form. It holds the group's settings once: operator, invert,
  bypass, opacity, refinement and name. The points after it are the members.
  A list holds one marker; the fold ignores any other.
- **Mask group.** The group `mask_id` names. The panel labels it "whole
  mask". It cannot be deleted or moved. Classic's whole-mask invert
  (`MASKS_POS`) becomes its `OP_INVERT`.
- **Member.** A point that refers to a form, and carries the member's own
  opacity, invert (`INVERSE`), element refinement, `HIDDEN` and `DISABLE`.
  A member that refers to a group carries no settings: that group's marker
  holds them.
- **Nesting.** A group nested in a group has one parent within a mask:
  nesting a group the mask already holds copies it. A shape or group linked
  to another module's mask stays shared. Every walk stops at
  `DT_MASKS_NESTING_MAX` (8).
- **Refinement scopes.** Element scope applies to one member's mask before
  it is combined, group scope to a group's finished mask, and the module's
  own refinement (blend params) to the whole mask after the fold, as on
  master.

## Rendering

`_group_get_mask_roi_flexi` (`group.c`) is selected by `DEVELOP_MASK_FLEXI`.
The classic fold is unchanged. For each group:

1. Fold the visible members in list order with the within-group operator.
   - Union, screen and sum start from 0; intersection and multiply from 1.
   - Difference takes the first visible member as its base and subtracts
     the rest.
   - Exclusion is classic's combiner, applied member by member. It is not
     associative, so order matters.
   - The other operators are order-free.
2. Refine the result (group scope), invert it (`OP_INVERT`), and scale it by
   `group_opacity`.

A group with nothing to fold contributes nothing, so an empty intersection
group does not blank the mask. A parametric channel at full range does not
count as something to fold. If nothing contributes anywhere, the mask is 1.0
and the module applies everywhere, as on master when no form is present.

A nested group renders through the same function, as a member of its
parent.

## How classic maps onto it

Migration (`migrate_legacy.c`, `dt_masks_group_mark_classic_runs`) writes
this tree:
- **Drawn list.** One group per run of a classic operator; where the
  operator changes, the group so far becomes the first member of the next
  one. This is a left fold, so classic's arithmetic carries over unchanged.
  A nested group whose settings are all neutral is spliced into its parent
  when it uses the same operator (difference and exclusion only as the
  base).
- **Parametric.** One element per active channel, in a within-multiply
  group. This is classic's per-channel `mask *= factor`.
- **Drawn + parametric.** A multiply group over the drawn group and the
  channels.
- **Raster.** One raster element.
- **Polarity flags** (`MASKS_POS`, `INV`, `INCL`) resolve to member and
  group inversion; `MASKS_POS` becomes the mask group's `OP_INVERT`.

## Alternatives considered

**A group table in the database.** This is the relational design, with
groups as rows referencing member forms. It needs a schema change and a
rule for a group whose last member is deleted. Markers need neither, and an
empty group is simply a marker with no members.

**Group settings on the `DT_MASKS_GROUP` form.** The form row has no field
for them, so they would need a schema change too. Classic's convention puts
them on the reference to the group, but the mask group has no reference.

**Groups as inferred runs of one flat list.** A group would be a stretch of
members sharing an operator, with its settings copied onto every member.
This needs no marker, but has three problems:
- the copies have to be kept in sync by hand;
- two adjacent groups with the same operator need a boundary field;
- an empty group cannot be stored.

**A flat two-level model.** An ordered list of groups, each with a
within-group and a between-group operator, and no nesting. It cannot hold
every classic tree exactly: two multi-step operands joined by an operator
need storage for one of them. An attempt to flatten classic trees into it
left 8 of 10,123 affected corpus edits visibly different.

**Nested groups with order-free operators only.** A nested group would be
an element with a within-group operator, and every operator order-free.
Difference becomes `multiply { A, inverted screen { B } }`, and exclusion a
union of two such products. This is exact, but:
- it duplicates operands;
- it adds levels, up to four for one exclusion;
- the panel shows structure the user never built.

**Serialized blobs (JSON) instead of fixed-size points.** These would be
easier to extend, but slower to parse, harder to bulk-copy, and a large
change to the masks I/O code for flexibility nothing needs yet.

**A multi-channel parametric element.** One element would hold classic's
whole blendif config. Channels would then combine only by the fixed AND,
and the same channel could not appear twice with different ranges.

## Why this design

- **No schema change.** Markers, new types and new fields ride in the
  points blob, which is versioned per form type. Old blobs are a prefix of
  the new struct.
- **One place per setting.** A group's settings exist once, on its marker.
  Nothing broadcasts or re-derives a partition.
- **Classic arithmetic carries over.** One ordered operator per group is a
  left fold, which is what classic does. Difference and exclusion keep
  their classic meaning instead of being rewritten.
- **Shallow trees.** On a static conversion of 16,037 distinct corpus
  edits, 15,139 need no nested group and 50 need two or more levels. The
  29 that get deeper than the flat model all switch operators back and
  forth, and any one-operator-per-group model nests once per switch.
- **One vocabulary.** Shapes, channels, raster masks and groups all combine
  with the same seven operators. Parametric and raster masks lose their
  special cases: parametric channels are no longer ANDed, and a raster mask
  no longer replaces the whole mask.
- **Classic stays readable.** The classic fold reads only the member flags
  and per-shape refinement, which are zero in every classic edit. It stays
  as the reference the verification tools compare against.
