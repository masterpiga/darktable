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
  defined in `data/masks_group_presets.json` (see below). The panel options
  pick the one a mask starts with when it is first switched on.
- **Linked shapes.** A shape shared with another module, or held twice by
  one mask, shows a chain icon. "Unlink" gives that row its own copy.
- **Mask lock.** A locked mask survives module reset, presets, styles and
  paste.

## Built-in group layout presets

`data/masks_group_presets.json` is read at runtime, and read again whenever
it changes. A copy in the config directory (`--configdir`, by default
`~/.config/darktable`) takes precedence over the installed one, so presets
can be edited without rebuilding.

```json
{
  "presets": [
    {
      "id": "drawn_parametric",
      "name": "drawn + parametric (classic)",
      "description": "menu tooltip",
      "mask": {
        "operator": "multiply",
        "notes": ["page 1, shown under the mask's own header", "page 2"],
        "groups": [
          { "id": "parametric", "name": "parametric", "operator": "multiply",
            "opacity": 1.0, "notes": ["..."], "groups": [] }
        ]
      }
    },
    {
      "id": "subtract",
      "name": "drawn + parametric, minus an area",
      "mask": {
        "operator": "difference",
        "groups": [
          { "id": "subtract", "name": "to subtract" },
          { "preset": "drawn_parametric", "name": "drawn + parametric" }
        ]
      }
    }
  ]
}
```

- `mask` is the mask's own group; `groups` lists nested groups top-first, as
  the panel shows them.
- `operator` is one of union, screen, intersect, multiply, sum, difference,
  exclusion; default union. `opacity` defaults to 1.
- A group can be another preset, inserted whole: `{ "preset": "<id>" }`.
  Whatever the group sets itself (`name`, `operator`, `opacity`, `notes`,
  `groups`) replaces that member of the preset's `mask`; everything else,
  nested groups included, comes from the preset. Presets can reference each
  other in turn, but not in a loop.
- Each string in `notes` is a page; with more than one, dots and arrows under
  the note move between them. Pages are
  [Pango markup](https://docs.gtk.org/Pango/pango_markup.html) (`<b>`, `<i>`,
  `<span>`...), so write `&` as `&amp;` and `<` as `&lt;`. A page that does
  not parse is reported on the console and shown as plain text.
- All notes are open when the preset is applied; after that only the
  selected group's, unless its info icon has switched it on or off.
- A group made by a preset remembers the key of its notes,
  `<preset id>/<group id>`: the group's `id`, else its `name`, else its
  position, within the preset that writes the notes. A group taken from a
  referenced preset keeps that preset's key, so its notes are written once.
  The notes are looked up by that key each time the group is shown, so edits
  show on groups made earlier; changing an `id` detaches them.
- Names, descriptions and notes are extracted for translation at build time
  by `tools/generate_masks_presets_strings.py`.
- Problems in the file are reported on the console, and the preset is
  skipped.

## More

- [User documentation](user_docs.md)
- Upstreaming: [masks_revamp_upstream_plan.md](../../masks_revamp_upstream_plan.md)
