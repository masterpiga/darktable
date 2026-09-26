# Flexi masks

Flexi replaces the mask manager and the per-type mask modes (drawn,
parametric, raster, drawn + parametric) with one mask model and one panel.

## The model

- **A mask is a tree of groups.** Each group folds its members in order
  with one operator: union, intersection, difference, exclusion, sum,
  multiply or screen. It then applies its own refinement, invert and
  opacity. Groups nest.
- **Every member is an element.** Drawn shapes, single parametric channels,
  raster masks, AI objects and groups each yield a 0-1 value per pixel and
  combine the same way.
  - Parametric channels are no longer ANDed.
  - A raster mask no longer replaces the whole mask.
  - The same channel can appear twice with different ranges.
  - Several raster sources can be combined.
- **Settings at every level.**
  - Elements: opacity, invert and refinement.
  - Groups: opacity, invert, refinement, bypass and a name.
  - The whole mask is a group, with the same controls.
- **Existing edits convert on load and render the same.** Over 63,157
  harvested edits, there were 0 failures in 8,203 distinct configuration
  shapes. Conversion is one-way.

Storage, rendering and the alternatives considered:
[masks_revamp_data_model.md](../../masks_revamp_data_model.md).

## The panel

- **One place.** Operators, order, grouping and element settings are all
  edited next to the shapes. There is no separate mask manager.
- **Position.** Embedded in the module, in the utility panel, as a side
  panel on either edge, or floating on the canvas.
- **Building.**
  - Add shapes, parametric channels and raster masks straight into the
    selected group. The selection persists, so successive strokes land in
    the same group.
  - "Add group" nests a group into the selected one.
  - Drag and drop reorders elements and groups, and moves them between
    groups.
- **Restructuring.** "Compose" wraps an element or group in a new group with
  another operator. "Simplify" removes structure that does not change the
  mask.
- **Solo** isolates an element or group in the rendered mask. **Solo edit**
  limits canvas handles to one shape.
- **Clustering.** Three or more adjacent shapes of one kind fold into a
  collapsible row.
- **Canvas and list stay in sync.** Hovering or selecting in one highlights
  the same thing in the other.
- **Group layout presets** are shared by all modules. The built-ins are
  drawn, parametric, and drawn + parametric.
- **Linked shapes.** A shape shared with another module, or held twice by
  one mask, shows a chain icon. "Unlink" gives that row its own copy.
- **Mask lock.** A locked mask survives module reset, presets, styles and
  paste.

## More

- [User documentation](user_docs.md)
- Upstreaming: [masks_revamp_upstream_plan.md](../../masks_revamp_upstream_plan.md)
