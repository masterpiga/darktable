# Flexi masks: GUI test protocol for group markers

What to click through before calling the marker transition done. The headless
checks (`--check-masks`, the unit suites, the pixel suite) cover migration,
storage and rendering; nothing covers the panel, and steps 3 and 4 of
`masks_revamp_group_markers.md` rewrote most of it.

Each check says what to do and what should happen. Anything else is a bug:
note the step number, what you saw, and whether it survives a reload.

## Setup

- A fresh build of `masks_revamp`, started with `-d masks` into a log, so a
  failure has something to read afterwards.
- A test library, not your real one: a group that goes wrong gets written
  back to the database.
- Three images:
  - **A**, never edited: for building masks from scratch.
  - **B**, a copy of an image with classic drawn masks made in a released
    darktable (5.x): a module with several shapes combined by different
    operators (union, intersection, difference), ideally one with shapes
    inside a nested group.
  - **C**, a copy of an image with a classic parametric mask, a raster mask
    and a drawn + parametric mask, also from a released darktable.
- Exposure at +2 EV is the easiest module to judge masks with. Keep the mask
  overlay on (the mask display button) for every render check below.

## 1. Classic edits migrate once, and look the same

1.1 Before opening B in the branch, export it from the released darktable.
    Open B in the branch and export it again. The two exports should match.

1.2 The mask panel of each masked module shows one group per run of
    same-operator shapes. A shape with intersection, difference, sum or
    exclusion sits alone in its own group, with that operator on the group
    header, and its elements show no operator of their own.

1.3 Change nothing, switch to another image and back. The groups and the
    render are unchanged.

1.4 Restart darktable and open B again. Same groups, same render.

1.5 Drag the history slider back one step and forward again. Same groups and
    render at each position.

1.6 Repeat 1.1 to 1.4 with C. The parametric, raster and drawn + parametric
    masks each appear as groups with the matching elements, and render as
    before.

## 2. Building groups from scratch

2.1 On A, enable exposure and open its mask panel. It shows one group before
    any shape exists.

2.2 Add a circle. It lands in that group. History gains an item.

2.3 Add a group (the add-group control, or the shortcut "add group above
    selected group"). An empty group appears above the selected one. History
    gains an item.

2.4 With the new group selected, add an ellipse. It lands in the new group,
    not in the bottom one.

2.5 Add a third group and leave it empty. Switch images and come back: the
    empty group is still there. Undo once: it goes. Redo: it comes back.
    (Before markers, empty groups were lost on reload. Keeping them is the
    intended change.)

2.6 Add a parametric element (an L or g channel button) and a raster
    element to a group. Each renders, and each can be moved like a shape
    (section 4).

## 3. Group controls

Use a group holding two overlapping shapes, above another group holding one.
After each check, switch images and back: the setting and the render stay.

3.1 Change the group's mode through each operator (union, intersection,
    difference, exclusion, multiply, screen). The render changes each time,
    and only this group's header changes.

3.2 Change the within-group mode (union, intersection, multiply, screen, sum).
    The group's two shapes combine accordingly; the other group is untouched.

3.3 Group opacity: lower it to about 30%. The group's contribution fades; the
    element opacities in the rows do not move.

3.4 Group refinement: blur or feather the group. It applies to the group's
    combined shape once, not to each element. Set an element refinement on
    one of its shapes too; both stay set, neither overwrites the other.

3.5 Group menu, visibility: disable, then enable. Solo, then unsolo.

3.6 Group menu, mask operations: invert output, then invert all elements.
    Each inverts what it says, and undo restores it.

3.7 Rename the group. The name shows on the header, survives a reload, and
    does not appear on any other group.

3.8 Merge elements into group below. The shapes move into the lower group
    and take its settings; the upper group is gone (not left empty).

3.9 Empty group. Its shapes are deleted, the group stays, and it survives a
    reload.

3.10 Delete group. The group and its shapes go. With only one group left,
     delete is disabled.

3.11 Shortcuts: "change mode for current group", "bypass/resume current
     group" and "change within-group mode for current group" act on the
     selected group only.

## 4. Moving things

4.1 Drag a shape from one group into another. It takes the target group's
    settings, and neither group is renamed or gains the other's opacity or
    refinement.

4.2 Drag a shape within its group, to the top and to the bottom. The group
    keeps its name and settings.

4.3 Drag a group's last shape out. The group stays, empty, in its place;
    groups above and below keep their order.

4.4 Drop a shape onto an empty group, including one directly next to the
    shape's own group. The shape lands there and nothing else moves.

4.5 Drag a whole group above and below others. The group moves with all its
    shapes and settings. The bottom group cannot be moved below nothing.

4.6 Undo each move, then redo it. Each is one history step, and the panel
    matches the render after every step.

## 5. Element controls

5.1 Shape menu: disable, solo, invert, rename, delete. Each acts on that
    shape only; the group header does not change.

5.2 Element opacity and element refinement on a shape in a multi-shape
    group: only that shape changes.

5.3 Delete a group's only shape from the canvas (select it, press delete).
    The group stays, empty, as with the menu.

5.4 Link a shape into a second module ("link shapes" in the import menu),
    then unlink it from the shape menu. Both modules keep their groups.

## 6. Presets and reset

6.1 Apply each built-in group layout preset ("add + subtract + intersect",
    "drawn mask", "parametric", "drawn + parametric"). The groups match the
    preset's description, with its operators on the headers.

6.2 Build a layout with three groups, one of them empty, with different
    opacities. Save it as a preset, reset the mask, apply the preset: same
    groups, same modes, same group opacities, the empty one included.

6.3 Reset the mask ("reset the mask: remove every shape"). One empty group
    remains, and the module renders unmasked.

## 7. History, copy and styles

7.1 With a three-group mask on A, compress history. Groups and render stay.

7.2 Copy A's history and paste it onto another unedited image, in append and
    in overwrite mode. The target gets the same groups.

7.3 Create a style from A's masked module and apply it to another image.
    Same groups.

7.4 Duplicate the masked module (new instance with the same settings). The
    copy has its own groups; changing a group in one does not change the
    other.

7.5 Export A. The export matches the darkroom view.

## 8. Canvas

8.1 Select a shape in the panel: it highlights on the canvas, and the other
    way round.

8.2 Move, resize and feather a circle, a path and a brush on the canvas.
    The mask follows; the panel row stays in its group.

8.3 Nudge a shape of a group whose raster element comes from another module,
    and switch that group's mode back and forth. The overlay never goes
    empty. (Fixed 2026-09-12; this checks it stays fixed.)

8.4 Dock and undock the panel. The toggle switches sit in place from the
    start, not only after hovering. (Fixed 2026-09-12.)

## What to report

For each failure: the step, a screenshot of the panel, the `-d masks` log,
and whether the image still shows the problem after a restart. A problem
that survives a restart was written to the database, which matters more than
one that only lives in the panel.
