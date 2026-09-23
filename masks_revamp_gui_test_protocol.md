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

5.5 Give B a raster element reading A's mask, and a later module C a
    whole-mask raster from A. A's panel shows "mask consumers" below the
    refinements, listing B and C with their instance names. Click B: B gets
    the focus, expanded, with its mask panel open (also with the panel
    folded, and in the canvas and utility positions). Switch B off, or its
    mask off, or delete the raster element: B leaves A's list, and the
    section goes once nothing reads A. Rename C: A's row follows.

5.6 Link icons: on B's raster element row (left of the opacity), on each
    "mask consumers" row, and on a linked shape (left of the opacity, where
    a parametric row has its picker). All look alike (a plain chain, no
    light chip, whatever the row's state) and brighten on hover. The raster
    one focuses A with its mask on the canvas; a consumer one goes to that
    module's mask panel; a linked shape's goes to the next module using the
    shape, with its row selected there, and clicking it there goes on to the
    next, back to the first. Tooltips name modules with "&" shown plainly.
    Hovering a "mask consumers" row highlights the whole row, name and chain
    together, wherever the pointer is on it.

5.7 A parametric element never shows a chain, and its menu has no "link" or
    "unlink": it can only be copied ("copy parametric channel").

5.8 Fold the refinements, shape properties and mask consumers sections, then
    select another element, another group, and another module's mask. Each
    section stays folded everywhere, including in a second module showing its
    mask panel at the same time. Unfold one: it unfolds everywhere. Restart:
    the states are kept. The masks options menu has no "collapse refinements
    by default". With "element properties in subpanel" on and that section
    folded, start drawing a circle: the section opens with the creation
    controls, and folds again once the circle is placed, or the drawing
    canceled.

5.9 Turn "auto-expand selected" off and "use sliders for opacity" on. The
    sliders option stays clickable, and opacity leaves every row and group
    header: expanding a shape, raster or parametric row, or a group, by hand
    shows its opacity slider at the top.

5.10 Turn "element properties in subpanel" on as well. Shape and raster rows
     have no chevron; a parametric row keeps its chevron, which shows only
     its input and output sliders; groups show no opacity slider in the list.
     Below the list, an expander's header shows the selection's icon and
     name, and "properties" and "refinement" sit inside it; with only the
     mask's own group selected it shows the mask panel icon (as on the
     darkroom toolbar) and "whole mask". Fold it: it stays folded for other
     selections, other modules, and after a restart. With it folded, start
     drawing a circle: it opens with the creation controls, and folds again
     once the circle is placed or the drawing canceled. On the refinement header, "refinement" is
     centered, reset is on the left and the eye left of the arrow; both work
     with the section folded, and both are greyed out while the target has no
     refinement, turning active as soon as a slider moves off zero.
     Select in turn: a shape (geometry and opacity), a raster element
     (opacity), a parametric L or g channel (opacity and boost factor; moving
     the boost factor rescales the row's sliders), a hue channel (opacity
     only, no boost factor), a nested group row (opacity), a group header
     alone ("selected group": the group's opacity, greyed out while the group
     is disabled). Each slider
     changes the render, and undo restores it. Turn the sliders option off:
     only a shape and a parametric channel with a boost factor still fill the
     section; anything else hides it.

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

## 9. Showing the panel, and switching the mask on

The rule: the mask's on/off state is the image, the panel's fold is the view.
A view action never writes to the image, and editing anything switches the
mask on. Run 9.1 to 9.5 once per panel position (blending options >
right-click the mask on/off toggle, or the toolbar mask icon's right-click):
embedded in the module, the utility panel, and out on the canvas.

9.1 With the mask off, show the panel. The mask stays off: the module's
    header gains no mask indicator, and no history item appears.

9.2 With the mask off and the panel showing, every control in it is live, not
    greyed out: the shape buttons, add group, import, the group list, blend
    mode and opacity.

9.3 Touch any one of them (move opacity, pick a blend mode, add a shape). The
    mask switches on, and the change is kept: the value you set is the value
    left in the control. The header's on/off switch and the toolbar icon both
    report it straight away.

9.4 The same for an edit of the mask itself, with shapes already there and the
    mask switched off: change an element's opacity or operator, rename or
    delete one, drag a shape on the canvas. Each switches the mask back on.

9.5 Switch the mask off from the header toggle. The panel stays exactly where
    it was, still showing, and does not fold away.

9.6 Switch the mask on from the header toggle with the panel folded. The panel
    unfolds.

9.7 In the toolbar, the mask icon's box is highlighted only when the panel is
    really on screen. Collapse the module with the panel in the canvas
    position: the panel goes, and the box stops being highlighted with it.
    Clicking then shows the panel rather than hiding an already hidden one.

9.8 The exception, which must NOT switch the mask on: reset the module (its
    header reset button) while the mask is off. The mask stays off. Same for
    applying a preset that carries no mask.

## What to report

For each failure: the step, a screenshot of the panel, the `-d masks` log,
and whether the image still shows the problem after a restart. A problem
that survives a restart was written to the database, which matters more than
one that only lives in the panel.
