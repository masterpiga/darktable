# darktable's New Mask Panel (a.k.a. Flexi)

darktable's new mask panel (a.k.a., Flexi) completely subsumes both the classic mask panel and the mask manager, which has been retired. All mask operations--creating shapes, setting parametric ranges, grouping, combining with boolean and arithmetic operators, inverting, soloing, refining, and reusing masks across modules--now happen in one place, directly accessible within each processing module or in a dockable side panel beside the canvas.

The mask panel organizes masks into a tree structure based on these entities:
- **Element**: A single mask component--a drawn shape (path, brush, circle, ellipse, gradient), a single-channel parametric threshold, an external raster mask, or an AI-segmented object. An element is always an individual leaf and never a group.
- **Group**: A container that holds one or more members and combines them using a designated combination operator (such as Maximum, Product, Screen, Difference, or Sum). The top-level group representing the module's complete mask is named **whole mask**.
- **Member**: Anything held directly inside a group. A member can be a single **element** or another nested **group**.
- **Shape**: A drawn vector shape (path, brush, circle, ellipse, gradient).
- **Row** and **Header**: The visual lines in the panel list representing members (an element row or a group header). The list itself is the tree of the mask, at the top of the panel under the blending settings.
- **Selection**: What the panel currently acts on. It is one member, or several members when you select more than one (see *Selecting Several Members*).
- **Controls**: The sliders and options of the selected member. They are always in the same place, under the creation toolbar, and not in the list (see *The Controls of the Selected Member*).

---

## What Changed Since PR #22786

This section is for readers who know the previous version of the panel (PR #22786). This version is PR #22833.

### Main Changes

- **The controls of the selected member are in one place**:
  Before, the controls of a member could be in its row, in a *properties* section under the toolbar, or both, depending on three options and on what you had opened before. Now the list only shows the structure of the mask. The opacity and the other controls of whatever is selected are under the creation toolbar, and they stay where they are as the selection changes. Rows no longer grow when you select them.
- **Only groups expand**:
  Element rows have no chevron any more. Groups have one, at the left of their header, and start expanded.
- **Two options are gone**:
  *Auto-expand selected* and *member properties in subpanel* (and the *auto-expand selected* shortcut) are removed, since nothing in the list expands on selection and the properties are always in the same place.
- **The properties section is now the controls section**:
  It holds the shape controls (size, shrink or grow, hardness, fade-out border, rotation, curvature, compression) or the ranges of a parametric element. The opacity is above it, with no header. The section is folded or open independently for shapes and for parametric elements, and the choice is remembered.
- **Parametric elements show the input slider only**:
  The *select visible sliders* menu on the channel icon is replaced by a toggle at the left of the *controls* header. It shows the output slider and, for channels that have one, the boost factor. Each slider has a small icon telling input from output. In the list, a parametric row shows a strip of the colors the element selects behind its name, with a second strip when the output slider is shown.
- **The link button moved to the row**:
  For a raster mask or a linked shape, the link is now in the row's header, at the left of the invert button.
- **Group headers reorganized**:
  The operator button moved from the left of the header to the right, next to invert and the eye. The chevron is now the first thing on the header.
- **New look for selection and hover**:
  A selected or picked header is framed in the accent color. A header under the pointer gets a bar along its right edge. While <kbd>Shift</kbd> is held, or while several members are selected, the frame of a selected group takes in its members.- **<kbd>Shift</kbd>+click with the whole mask selected**:
  The whole mask now behaves like any other selected group: <kbd>Shift</kbd>+click on a member picks everything else in the mask.
- **Options popover rearranged**:
  The panel position and the default layout are combo boxes, not lists of radio buttons, so the popover is shorter. The options are grouped under *instance settings*, *interface options*, *shape element options*, *parametric element options* and *editing options*. *Show mask indicator in module headers* and *show preset help notes* moved to *interface options*. *Sticky opacity* is now *remember last shape's opacity* and is under *shape element options*.
- **Always-on panel (new option)**:
  With the panel beside the canvas, *always-on panel* keeps it open for every module, with a mask or without, as a column that takes its room from the canvas instead of floating over it.
- **A built-in layout was renamed**:
  `drawn + parametric, minus an area (advanced)` is now `drawn + parametric, minus an area`.

### Where Things Moved

| Control | In #22786 | Now |
|---|---|---|
| Opacity of the selected member | in the *properties* section, or in the row | under the toolbar, with no header |
| Size, hardness, fade-out border, rotation, curvature, compression, shrink or grow | *properties* section, or in the row | *controls* section |
| Output slider and boost factor of a parametric element | *select visible sliders* menu on the channel icon | toggle in the *controls* header |
| Chevron of an element row | at the right end of the row | none |
| Chevron of a group | at the right end of the header | left of the header |
| Group operator | left of the header | right of the header, before invert |
| Link to the other module | selection caption | row header |
| *Auto-expand selected*, *member properties in subpanel* | mask options popover | removed |

### Workflow Effects

- Selecting a member does not move the rows below it. The controls you are working with stay at the same place on the screen.
- To see or change the opacity of a member, select it. There is no need to open the row.
- To close a group, click its chevron. A group that holds the selection stays open.
- To use the link of a raster mask or a linked shape, click the chain icon on its row. <kbd>Shift</kbd>+click gives this module its own copy.
- The panel can stay on screen next to the canvas while you move from module to module.

---

## 1. Panel Header Controls

At the top of every module's mask panel sits the header bar. The domino is at the left. The other controls are at the right, in the order below, followed by the show/hide arrow.

- <img src="icons/masks_panel.svg" width="18" height="18" align="middle" alt="domino" /> **Mask On/Off (domino icon)**
  - **Left-click**: Toggles masking on or off for the current module. A locked mask cannot be switched on or off ("the mask is locked").
  - **Right-click**: Opens the **Mask Options popover** to configure panel behavior, color spaces, and layout presets.
- <img src="icons/showmask.svg" width="18" height="18" align="middle" alt="mask overlay" /> **Display Mask (yellow overlay icon)**
  - **Left-click**: Toggles the mask overlay on the image: gray, tinted yellow where the module applies (the stronger the yellow, the stronger the effect). An empty mask shows no overlay, since the module then applies everywhere. Switches the module and the mask on if they are off.
- <img src="icons/eye_toggle.svg" width="18" height="18" align="middle" alt="suppress mask" /> **Temporarily Switch Off Mask (eye toggle icon)**
  - **Left-click**: Compares the image with and without the mask: the module then applies everywhere, at the blend opacity. Only the center view is affected, not exports, and the mask switches back on when you leave the module. Greyed out while the module or its mask is off.
- <img src="icons/masks_eye.svg" width="18" height="18" align="middle" alt="edit on canvas" /> **Edit Mask on Canvas (eye icon)**
  - **Left-click**: Shows or hides the handles of the drawn mask elements on the image canvas.
  - **<kbd>Ctrl</kbd>+Left-click**: Activates restricted edit mode (shapes cannot be moved or resized).
  - Greyed out while the mask holds no drawn shape or is locked.
- <img src="icons/soloedit.svg" width="18" height="18" align="middle" alt="solo edit" /> **Solo Edit (spotlight icon)**
  - **Left-click**: While enabled, only the selected shape, or the shapes of the selected group, are shown and editable on the canvas. The other members still contribute to the mask.
  - Greyed out like the edit button.
- <img src="icons/mask_lock.svg" width="18" height="18" align="middle" alt="lock mask" /> **Lock Mask (padlock icon)**
  - **Left-click**: Locks or unlocks the mask. A locked mask cannot be edited and is not affected by module reset, presets, styles or pasted module parameters; for instance, switching a *color calibration* instance from no preset to a preset keeps the mask it had. Discarding the history still removes it. Only shown on modules that support masks.
- **Show/Hide Arrow**
  - **Left-click**: Hides or shows the mask panel's body. Clicking an empty part of the header does the same. Not shown in the utility module position, where the module's own header folds the panel.

### Panel Visibility

When the mask panel is a **separate panel beside the canvas**, clicking its header makes it disappear. It comes back by clicking the domino button in the darkroom bottom toolbar, or the domino icon and arrow of the halo that shows when the pointer rests at the edge of the canvas where the panel docks. The handle on the panel's edge also hides it on a click, and resizes it when dragged.

### Always-On Panel

With **always-on panel** enabled (see *Mask Options Popover*), the panel beside the canvas behaves differently:
- It stays open for every module, with a mask or without, until you fold it with its header or its handle.
- It is a column beside the canvas, not an overlay. The canvas is laid out next to it, and nothing is hidden under it.
- It shows the mask of the focused module. With no module focused it says *no module is focused: click a module's header to focus it*. With a module that has no masks it says *the focused module does not support masks*.
- Hiding the left or right panel it is docked next to hides it too. It comes back with that panel, as you left it. Collapse-all hides it with the other panels, and a second collapse-all gives it back as it was.

---

## 2. Mask Options Popover

The Mask Options popover configures interface settings, default layouts, and color spaces specific to the mask panel.

### Accessing the Popover
- **Right-click the On/Off domino button** (<img src="icons/masks_panel.svg" width="18" height="18" align="middle" alt="domino" />) in any module's mask panel header.
- **Right-click the mask panel toggle button** in the darkroom bottom toolbar. With no module focused, the popover only shows the settings that belong to the panel and not to a module.

### Popover Settings

The popover has these sections, in this order.

#### 1. Instance Settings
- **Blend colorspace** (combo box): Selects the color space used to evaluate the module's mask and its parametric channels:
  - **default**: Uses the module's native working color space.
  - **RGB (scene)**: Evaluates the mask in unbounded linear RGB. Preserves high dynamic range values in scene-referred pipeline modules.
  - **RGB (display)**: Evaluates the mask in display-referred RGB.
  - **Lab**: Perceptual Lab color space. **Only offered on modules that natively process in Lab**; it is withheld on RGB modules to avoid invalid color space conversions before the input color profile or after the output color profile.

This section is only shown for modules that support parametric masks, and the combo box is greyed out while the mask is locked. A change that would invalidate parametric elements already in the mask asks for confirmation first and deletes them on a yes.

#### 2. Interface Options
- **Panel position** (combo box): Controls where the mask panel is hosted in the interface:
  - **Embedded within each module (default)**: The mask panel lives directly inside each processing module's expandable body.
  - **Utility module, left panel**: Moves the entire panel into a standalone module labeled **masking** in the left sidebar. Focusing any processing module directs this utility module to display and edit that module's mask.
  - **Separate panel, beside the canvas**: Detaches the mask panel into a dockable, resizable panel positioned directly beside the image canvas.
    - It can be summoned to either side of the canvas: approaching either the left or right canvas edge with the pointer reveals a halo overlay spanning the whole height of the canvas with the domino icon (<img src="icons/masks_panel.svg" width="18" height="18" align="middle" alt="domino" />) and an inward arrow. Clicking the halo opens the panel on that side.
    - When the panel is already open, approaching the opposite canvas edge reveals the halo on that side; clicking it relocates the panel across to the opposite edge of the canvas.

  Changing the position applies at once and unfolds the panel. Three options appear under the combo box when they apply to the chosen position:
  - **Always-on panel** (disabled by default; beside-canvas position only): Keeps the panel open for every module, as a column that takes its room from the canvas. See *Always-On Panel*.
  - **Show the resize handle** (enabled by default; beside-canvas position only): Displays a visible grip handle with an arrow indicating the collapse direction on the panel's edge. When disabled, the handle is hidden but the edge can still be dragged to resize and clicked to hide.
  - **Show focused module under header** (enabled by default; utility and beside-canvas positions): Shows a row under the panel's header naming the module whose mask the panel displays, and its instance name if it has one.
- **Default layout for new masks** (combo box): Selects which layout preset a module's mask starts with when it is switched on for the first time. The built-in choices are `drawn (classic)` (the default), `parametric (classic)`, `drawn + parametric (classic)` and `drawn + parametric, minus an area`, followed by your own saved presets. The tooltip describes the selected preset.
- **Show preset help notes (beginners)** (enabled by default): When enabled, groups generated from built-in presets display a help note in the selection panel explaining their purpose and suggested usage. When disabled, the help section is not shown.
- **Show mask indicator in module headers** (enabled by default): Shows an icon in the header of every processing module that has a mask applied.

#### 3. Shape Element Options
- **Remember last shape's opacity** (enabled by default): When enabled, newly created shapes inherit the opacity value last used by any shape. When disabled, every new shape starts at 100% opacity.

#### 4. Parametric Element Options
- **Reuse the last picked area** (enabled by default): When sampling values on the canvas with the color picker, the sampled bounding box is remembered. Clicking the color picker on subsequent parametric channels immediately applies that same sample area without having to redraw a selection on the image.
- **Preview channel under cursor** (disabled by default): When enabled, hovering the pointer over any add-parametric-channel button displays a live center-view preview of that channel after a brief pause, allowing you to preview the channel's mask before adding it.

#### 5. Editing Options
- **Scroll down to increase mask parameters** (disabled by default): Sets which way the mouse wheel changes mask size, feather size, opacity, brush hardness and gradient curvature on the canvas. By default, scrolling up increases them. When enabled, scrolling down does.

---

## 3. Blending Settings

When the mask panel is expanded, the top section provides module-level blending controls:

- **Blend Mode dropdown**: Blends the module's processed image output with the input image (Normal, Multiply, Screen, Difference, etc.). This controls image layer blending only; it does not affect how mask elements combine.
- **Blend Order (input/output icon)**: Toggles the blending order between the input and the output of the module. By default the output is blended on top of the input; clicking reverses this (input on top of output).
- **Blend Parameter slider (labeled "fulcrum")**: Appears only for blend modes that accept an adjustable parameter; adjusts the fulcrum used by those blending operations.
- **Opacity slider (under the blend controls)**: Controls how strongly the module's blended effect applies to the image. Double-clicking resets it to 100%. A warning badge appears next to its label while the opacity is below 10%.

### Blend Opacity vs. "Whole Mask" Group Opacity
- **Blend opacity** scales the final mix of the module's processed image over the input image. If no mask is active, the module effect applies uniformly at this percentage. Blend opacity attenuates the final image blend without altering the mask itself.
- **"Whole mask" group opacity** (the opacity under the toolbar while the whole mask is selected) scales the values of the mask itself (from 0 to 1). Changing the "whole mask" opacity directly modifies the resulting mask data, which affects the yellow mask overlay preview and scales the raster mask exported to downstream modules.

---

## 4. The Mask Layout (The Member Tree)

The list under the blending settings is a tree: it represents the hierarchy of groups and elements that compose the mask. Its top entry is the whole mask's own group. The list only shows structure; the controls of a member are not in it (see *The Controls of the Selected Member*).

### Tree Rail and Nesting
Groups can nest inside other groups up to 8 levels deep. A continuous vertical tree rail runs down the left edge of each group, ticking across to each child member's header, visually communicating the group's boundaries and hierarchy. Every rail has the color of a group header at rest.

### Selection Colors
- **Selected**: A frame in the theme's accent color around the header (in the color of the text if the theme has no accent color).
- **Picked** (part of a multiple selection): Looks like selected.
- **Held by a selected or picked group**: A member of a group that is selected or picked as a whole. It is not framed. While <kbd>Shift</kbd> is held, or while several members are selected, the frame of a selected group takes in all its members.
- **Group holding the selected element**: A lighter shade on the group's header.
- **Under the pointer**: A bar along the right edge of the row or header. It also shows when the pointer is on the matching shape on the canvas. The shade of the header does not change.
- **Soloed** and **drop target while dragging**: Their own shades.
- **Linked shape**: A dashed frame (the shape is linked with other modules, or shown more than once in this mask).

### Group Combination Operators
A group combines its visible members using one chosen operator, starting from its bottom member and working up the list (a left fold):
- <img src="icons/masks_maximum.svg" width="18" height="18" align="middle" alt="maximum" /> **Maximum (Union)**: Takes the highest opacity value between overlapping members: `max(A, B)`.
- <img src="icons/masks_screen.svg" width="18" height="18" align="middle" alt="screen" /> **Screen**: Inverted product; soft additive combination without harsh clipping: `A + B - AB`.
- <img src="icons/masks_minimum.svg" width="18" height="18" align="middle" alt="minimum" /> **Minimum (Intersection)**: Takes the lowest opacity value: `min(A, B)`.
- <img src="icons/masks_product.svg" width="18" height="18" align="middle" alt="product" /> **Product**: Multiplies opacity values: `A * B`.
- <img src="icons/masks_sum.svg" width="18" height="18" align="middle" alt="sum" /> **Sum**: Clamped addition: `min(1, A + B)`.
- <img src="icons/masks_difference.svg" width="18" height="18" align="middle" alt="difference" /> **Difference**: Takes the first member (the bottom one) and subtracts all the following ones (those above it): `A - B - C...`.
- <img src="icons/masks_exclusion.svg" width="18" height="18" align="middle" alt="exclusion" /> **Exclusion**: Combines areas without overlap (exclusive OR).

### Group Headers
Each group displays a header bar. From left to right:
- **Expander Chevron**:
  - **Left-click**: Collapses or expands the group's members in the list, and selects the group. Greyed out on an empty group. Groups start expanded, and a group that holds the selection stays open.
  - Only groups have a chevron.
- **Group Title**:
  - Displays the group name (defaults to `<operator>-<ordinal>`, e.g., `max-1`, or `whole mask` for root). A warning badge follows it when the group holds a member with an opacity below 10%.
  - **Left-click**: Selects the group; clicking it again selects the whole mask.
  - **<kbd>Ctrl</kbd>+Left-click**: Opens an inline text field to rename the group (the whole mask cannot be renamed).
  - **<kbd>Shift</kbd>+Left-click**: Adds the group to the selection, or takes it off (see *Selecting Several Members*). The whole mask cannot be picked this way.
  - **<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+Left-click**: Deletes the group and every member in it (resets the mask on the whole mask).
  - **Drag**: Drag by the title or header to reorder the group or nest it inside another group.
- **Operator Button (first icon of the right-hand group of icons)**:
  - Displays the current combination operator icon (e.g., <img src="icons/masks_maximum.svg" width="18" height="18" align="middle" alt="maximum" />).
  - **Left-click**: Opens the operator menu to switch how members in this group combine. While several members are selected and this group is one of them, the choice is applied to every selected group.
  - **<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+Left-click**: Deletes the group and every member in it.
- <img src="icons/invert.svg" width="18" height="18" align="middle" alt="invert" /> **Invert Button**:
  - **Left-click**: Inverts the group's composite output. When inverted, the button and operator icon display an inverted highlight style (<img src="icons/masks_maximum_inv.svg" width="18" height="18" align="middle" alt="inverted operator" />).
  - **<kbd>Shift</kbd>+Left-click**: Inverts each direct member inside the group in one action.
- <img src="icons/eye_toggle.svg" width="18" height="18" align="middle" alt="visibility" /> **Visibility (eye icon, rightmost)**:
  - **Left-click**: Disables (bypasses) the group.
  - **<kbd>Shift</kbd>+Left-click**: Solos the group (evaluates only this group, temporarily hiding all other members in the mask). Not available on the whole mask.

### Member Rows
Every element inside a group is displayed as a row. From left to right:
- **Type Icon / Drag Handle**:
  - Displays the element type icon: <img src="icons/masks_path.svg" width="18" height="18" align="middle" alt="path" /> Path, <img src="icons/masks_brush.svg" width="18" height="18" align="middle" alt="brush" /> Brush, <img src="icons/masks_circle.svg" width="18" height="18" align="middle" alt="circle" /> Circle, <img src="icons/masks_ellipse.svg" width="18" height="18" align="middle" alt="ellipse" /> Ellipse, <img src="icons/masks_gradient.svg" width="18" height="18" align="middle" alt="gradient" /> Gradient, <img src="icons/masks_raster.svg" width="18" height="18" align="middle" alt="raster" /> Raster, or <img src="icons/masks_object.svg" width="18" height="18" align="middle" alt="object" /> AI Object. A parametric channel shows its channel code instead (e.g., `L`, `Cz`, `hz`).
  - **Drag**: Drag to reorder within the group or drop into another group. An insertion line indicates the target position. Dragging is off while several members are selected.
- **Member Name**:
  - Displays the member name (e.g., `circle #1`, `L in`, or a custom label). A parametric row shows a strip of the colors the element selects behind its name, and a second strip under it while the output slider is shown. A warning badge follows the name if the element's opacity is below 10%, warning that its contribution is practically invisible; its tooltip says so.
  - **Left-click**: Selects the member (click again to deselect).
  - **<kbd>Ctrl</kbd>+Left-click**: Opens an inline text field to rename the member.
  - **<kbd>Shift</kbd>+Left-click**: Adds the member to the selection, or takes it off.
  - **<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+Left-click**: Deletes the element.
  - The same clicks work on the drag handle and on the empty part of the row.
- <img src="icons/link.svg" width="18" height="18" align="middle" alt="link" /> **Link Button (raster masks and linked shapes only)**:
  - **Left-click**: On a raster mask, focuses the module that writes the mask, to edit it on the canvas. On a linked shape, jumps to the next module using the shape. The tooltip names the target.
  - **<kbd>Shift</kbd>+Left-click**: Unlinks a shape, giving this module its own independent copy so later edits do not affect other modules.
- <img src="icons/invert.svg" width="18" height="18" align="middle" alt="invert" /> **Invert Button**:
  - **Left-click**: Inverts the element. When inverted, the icon plate displays an inverted highlight (<img src="icons/masks_circle_inv.svg" width="18" height="18" align="middle" alt="inverted circle" />).
- <img src="icons/eye_toggle.svg" width="18" height="18" align="middle" alt="visibility" /> **Visibility (eye icon, rightmost)**:
  - **Left-click**: Disables (bypasses) the element.
  - **<kbd>Shift</kbd>+Left-click**: Solos the element (evaluates only this single element).

The invert buttons and the eyes line up in columns down the list. A row that has no link gives its room to the name.

### Armed Deletion (<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+Click)
Holding <kbd>Ctrl</kbd>+<kbd>Shift</kbd> over the list activates armed deletion:
- The pointer transforms into a **trash bin icon**.
- The hovered row or group header turns **red**.
- For a group, its **vertical rail turns red**, clearly highlighting all descendant members that will be deleted.
- Clicking immediately deletes the targeted element or group (or resets the mask if clicked on the root "whole mask").

---

## 5. Selecting Several Members

### Adding to the Selection
- **<kbd>Shift</kbd>+click** a row or a group header to add it to the selection. <kbd>Shift</kbd>+click it again to take it off.
- The member that was already selected counts as the first one. A selected member and one <kbd>Shift</kbd>+click make two.
- When the whole mask is the selection (nothing else is), it counts as a selected group: <kbd>Shift</kbd>+click on a member picks everything else in the mask. The whole mask itself can never be picked, and a <kbd>Shift</kbd>+click on its header does nothing special.
- While <kbd>Shift</kbd> is held over the list and there is something to add to, the pointer shows a plus sign over a row you can add, and a minus sign over a row that is already selected.
- A locked mask cannot be selected from.
- The selection ends when you click a row without <kbd>Shift</kbd> (that row is then the only selection), when the module loses focus, and after delete, move or group.

### Groups Are Selected Whole
- Selecting a group selects everything it holds, at any depth.
- Taking one member off a selected group takes the group off as well, since it is no longer whole. The other members stay selected.
- This also applies when the group was the one selected before you pressed <kbd>Shift</kbd>.
- Actions act on the group as one thing. The members of a selected group are not acted on a second time.

### The Multiple Selection Panel
From two selected members on, this panel replaces the selection panel (caption, creation toolbar, controls, help and refinements). While it is shown, you cannot add shapes or edit controls. Click a single row to get the usual panel back.

The caption reads **n items selected**. The count includes the members of selected groups.

- **delete**: Deletes everything selected in one step. A group goes with its members.
- **move into...**: Opens a menu of the places the selection can go: the whole mask and the groups in it, indented by depth, each with its operator icon. The selection lands on top of the chosen group and keeps its order. The menu leaves out groups where nothing would move, a group inside itself or inside its own members, and anything that would nest deeper than 8 levels. When nothing is left, a message says the selection has nowhere else to go.
- **group together...**: Puts the selection in a new group, at the place of the topmost selected member. The menu asks for the new group's operator. The members keep their order, and the new group becomes the selection. All selected members must be in the same group, or the button is greyed out and its tooltip says why. When the selection comes down to a single thing (for instance one whole group), the button reads **wrap in group...**.

### Buttons on Selected Rows
While several are selected, the **eye**, **invert** and **operator** buttons of a selected row or header apply to every selected member. All of them take the state the click gives the one you clicked: click the eye on a visible member and every selected member is disabled. A member held by a selected group follows its group. The chevron of a group still opens or closes only its own group.

### On the Canvas
The shapes of the selection are highlighted on the canvas as a selected shape is. Dragging rows is off while several are selected; a message says so if you try.

### Undo
Undoing a delete, move or group brings back the mask and the same selection.

### Shortcuts
While several are selected, *delete selection*, *invert selected element*, *invert output of selected group*, *disable/enable selected group* and *change operator of selected group* act on the whole selection. *Solo edit the selection* and *add group in selected group* do nothing. *Move selection into group* and *group selection* open the menus of the two buttons above.

---

## 6. The Selection Area & Action Toolbar

Beneath the tree view sits the **selection area**, which holds, from top to bottom: the selection caption, the creation toolbar, the controls of the selected member, the help note of a preset group, and the refinements. It is hidden while several members are selected (see above).

### The Selection Caption
The top row of the selection area names the currently active item. Delete and the wand sit on its left, presets on its right. A button greyed out has nothing to act on; while a shape is still being drawn or the mask is locked, delete, the wand and presets all are.
- <img src="icons/trash.svg" width="18" height="18" align="middle" alt="delete" /> **Delete Button (trash icon)**:
  - **Left-click**: Deletes the selected member (or resets the mask to the default group layout if "whole mask" is selected). On a linked element, only deletes it here; the other rows and modules it is linked with keep it. The tooltip also names the shortcut keys currently bound to *delete selection*.
  - **<kbd>Shift</kbd>+Left-click**: When a group is selected, **empties the group** (removes all its members while keeping the group container in place, with its settings and name).
- <img src="icons/wand.svg" width="18" height="18" align="middle" alt="restructure" /> **Wand Button (wand icon)**:
  - **Left-click**: Opens the **restructure** menu:
    - **wrap in group** (submenu of operators): Puts the selection in a new group of the chosen operator, where it is, to combine it with what you add to that group later. Not offered for the whole mask.
    - **compose** (submenu of operators): Uses the selection as a building block for a larger mask, for example to subtract from it or to intersect it with something else. It goes into a new group of the chosen operator, with an empty group on top to draw into.
    - **simplify**: Only for a group. Removes the groups inside it that change nothing. Groups with a name of their own are kept.
  - An operator that would change nothing is left out of the submenus.
  - Greyed out when the selection cannot be restructured where it is.
- **Selection Name Box (center)**:
  - Shows the name of the selected member (or `whole mask`).
  - **<kbd>Ctrl</kbd>+Left-click** or **double-click**: Renames the selected member (the whole mask cannot be renamed). A plain click does nothing.
  - **Canvas Hover Peek**: When hovering the mouse over a shape on the image canvas, the caption displays that shape's name, highlighted to show that it is a peek, not the selection.
- <img src="icons/presets.svg" width="18" height="18" align="middle" alt="presets" /> **Group Layout Presets Button (presets icon)**:
  - **Left-click**: Opens the **Group Layout Presets Popover** targeting the selected group. Greyed out when no group is selected, and for raster masks, which have no group layouts.

### The Action Toolbar
The creation toolbar sits directly under the caption:
- **Left side: Group & Import**:
  - <img src="icons/plus.svg" width="18" height="18" align="middle" alt="add group" /> **Add Group (`+`)**: Adds a new, empty group; click to pick its operator.
  - <img src="icons/import.svg" width="18" height="18" align="middle" alt="import" /> **Import**: Links or copies shapes from other modules, copies their parametric channels, or adds or uses another module's whole mask. Hovering an import entry previews it on canvas.
- **Right side: Drawn Shapes & Parametric Channels**:
  - **Drawn Shapes**: <img src="icons/masks_path.svg" width="18" height="18" align="middle" alt="path" /> Path, <img src="icons/masks_brush.svg" width="18" height="18" align="middle" alt="brush" /> Brush, <img src="icons/masks_circle.svg" width="18" height="18" align="middle" alt="circle" /> Circle, <img src="icons/masks_ellipse.svg" width="18" height="18" align="middle" alt="ellipse" /> Ellipse, <img src="icons/masks_gradient.svg" width="18" height="18" align="middle" alt="gradient" /> Gradient, <img src="icons/masks_object.svg" width="18" height="18" align="middle" alt="object" /> AI Object.
    - **Left-click**: Starts drawing a single shape on canvas.
    - **<kbd>Ctrl</kbd>+Left-click**: Activates continuous shape creation mode.
    - **<kbd>Escape</kbd> or Right-click**: Cancels drawing the pending shape.
  - **Parametric Channels**:
    - Flat buttons, one per channel of the mask's blend color space, labeled with the channel code (e.g., `g`, `R`, `G`, `B` for RGB; `L`, `a`, `b`, `C`, `h` for Lab).
    - **Hover**: Displays a live preview of that channel's mask on the canvas (only when "preview channel under cursor" is enabled in options).
    - **Left-click**: Adds that channel into the active group.
    - **<kbd>Shift</kbd>+Left-click**: Pins a persistent channel preview overlay on canvas.

---

## 7. The Controls of the Selected Member

The controls of the selected member are under the toolbar. They never move into the list, and they do not move when the selection changes. While you hover a shape on the canvas, they show that shape's controls instead. With several members selected, there are none.

They come in two parts.

### Opacity
Directly under the toolbar, with no header, for any selected member: an element, a group, or the whole mask. Double-click resets it to 100%.

### The Controls Section
Under the opacity is a collapsible section named **controls**. Click its header to fold or unfold it. Shapes and parametric elements fold independently, and each remembers its state. The section is greyed out and folded when the selection has nothing to put in it, and it opens by itself while you draw a shape.

What it holds depends on the selection:
- **Drawn Shapes**: Sliders for size and the ones that apply to the shape: hardness (brushes), fade-out border, rotation, and curvature and compression (gradients). Paths also have a *shrink or grow* slider (see below).
- **Parametric Channels**: The sliders of the channel (see below).
- **Raster masks, AI objects, groups, the whole mask**: Nothing besides the opacity.
- **While drawing a shape**: The creation controls of the shape before you commit it. For an AI object, these are *refine mask boundary*, *apply as paths* and, with paths, *smoothing* and *cleanup*. For a brush, they are the pen *pressure* (with a pen) and *smoothing*.

With a raster mask row hovered in the list and the masks shown on the canvas, the area the raster mask covers is tinted on the image.

### Shrink or Grow (Paths)
The slider moves the outline of a path in or out, in pixels or in percent of the path's size. The path keeps its curves and its nodes; extra nodes are added only where a curve needs them. Setting it to 0 restores the outline the path grew or shrank from.

The **gear** at the right end of the slider opens the **shrink or grow settings** popover:
- **Step size**: How much a path grows or shrinks per scroll tick when you scroll over it on the canvas (1 to 200).
- **Unit**: *pixels* (image pixels) or *% of path size* (percent of the largest dimension of the path before it is resized). It sets the unit of the step and of the slider.
- **Tracing resolution**: *512*, *1024* or *2048*. The size, in pixels along the longest side, of the raster a path is traced on. Higher values follow the outline of a large path more closely and are slower. A small path is never traced finer than the image it is on.
- **Curve smoothing**: *smooth* follows the resized outline with few nodes and round curves. *sharp* traces it as a polygon, which takes many more nodes.

The settings apply from the next resize on.

### Parametric Elements
The opacity is under the toolbar, as for any element. The sliders are in the controls section:
- **Input slider**: Always shown, with an input icon at its left.
- **Output slider and boost factor**: Shown with the toggle at the left of the *controls* header, which only appears while a parametric element is selected. The toggle also shows the boost factor, for channels that have one. The output slider has an output icon at its left. The state is remembered per element.
- **Dual Gradient Sliders**: Each slider features 4 threshold markers (black point, feather start, feather end, white point).
- **Right-click on any marker**: Opens a precise numeric input popover for that threshold.
- **Dedicated Color Pickers (<img src="icons/colorpicker.svg" width="18" height="18" align="middle" alt="color picker" />)**: Located directly at the right edge of each slider. Left-click sets the slider's range from an area dragged on the image; <kbd>Ctrl</kbd>+click picks a GUI color (point) shown on the sliders; <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+click does the same for an area.
- **Boost Factor**: Increases the range of values the channel can match over 100%, in EV.

### Property Sliders
The sliders are darktable's usual ones:
- **Scroll wheel**: Adjusts the value, with the usual fine and coarse modifiers, unless *mouse wheel scrolls modules side panel by default* is enabled in the preferences, in which case the wheel scrolls the panel and <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+wheel adjusts the value.
- **Right-click**: Opens the fine-tuning popup.
- **Double-click**: Resets the value: opacity to 100%, the others to what they were when the controls were shown, or to 0.
- **Typed values**: A value below the lowest the shape accepts (0 for instance, on size, hardness, fade-out border or compression) sets the lowest value, and a message names it.
- **Range**: The *fade-out border* slider stops at 100%. Scrolling with <kbd>Shift</kbd> over the shape on the canvas can set a wider border for some shapes.

---

## 8. Help and Mask Refinements

### Help (Preset Guidance Notes)
When a group created from a built-in preset is selected (and "show preset help notes (beginners)" is enabled in the options popover), a help section appears under the controls explaining the intended purpose, typical workflow, and tips for working with that preset layout. A note of several pages has arrows to turn them.

### Mask Refinements
Refinements provide edge-aware smoothing, feathering, and tonal adjustments. The refinement automatically applies to the current selection scope:
- **Element Scope**: Refines that specific element before it is combined by its group's operator.
- **Group Scope**: Refines the group's composite output after its operator combines its members.
- **Whole Mask Scope**: Refines the final combined mask of the module.

Click the section header to fold it; the state is remembered for every module.

#### Refinement Controls:
- <img src="icons/eye_toggle.svg" width="18" height="18" align="middle" alt="bypass" /> **Bypass Toggle**: Temporarily disables refinements for the current scope without resetting sliders.
- <img src="icons/reset.svg" width="18" height="18" align="middle" alt="reset" /> **Reset Button**: Resets refinement sliders for the current scope to default values.
- **Details Threshold**: Positive values select areas with strong details, negative values flat areas (computed from raw data).
- **Feathering Guide**: Selects whether edge feathering is guided by the input or output image, and whether feathering applies before or after mask blur.
- **Feathering Radius**: Spatial radius for guided feathering (in pixels).
- **Blurring Radius**: Standard Gaussian blur applied to the mask (in pixels).
- **Mask Brightness**: Shifts the tone curve of the mask to brighten or darken semi-transparent areas without altering fully transparent or opaque regions.
- **Mask Contrast**: Applies an S-curve to mask opacities to steepen or flatten transitions.

---

## 9. Group Layout Presets Popover

Clicking the Presets button (<img src="icons/presets.svg" width="18" height="18" align="middle" alt="presets" />) in the selection caption opens the **Group Layout Presets Popover**.

### Scope of Application
Presets apply within the **currently selected group** (or to the whole mask if no group is selected or if root is selected). A mask with no group yet automatically adopts the default layout.

### Popover Layout and Application Modes
The popover organizes presets into **built-in presets** and **user-defined presets**. Each preset row provides:
- **Preset Name Button**: Clicking the name adds the preset layout as a subgroup inside the active group (the safest method, which loses no existing elements).
- **Layout Outline Tooltip**: Hovering over the preset name displays an outline showing all nested groups, their combination operators, and initial opacities.
- **Three Mode Action Buttons (right edge)**:
  1. **Replace** (circular arrows icon): The group drops every member it holds and becomes the preset, keeping its name. If existing elements would be discarded, darktable prompts for confirmation first.
  2. **Merge** (union icon): The preset's groups go into the group, next to what it holds; the group keeps its own operator and opacity. Only offered for a preset with groups inside.
  3. **Add as Subgroup** (plus-in-a-square icon): The preset goes into the group whole, as a new group next to what it holds.
- **Delete Button** (<img src="icons/remove.svg" width="18" height="18" align="middle" alt="delete" />, left of the name): Available on user presets to delete custom layouts.

### Saving Custom Layouts
At the bottom of the popover, click **"save this group's layout as preset..."** (or **"save the whole mask's layout as preset..."**). The save dialog takes immediate keyboard focus, confirms with <kbd>Enter</kbd>, and cancels with <kbd>Escape</kbd>. Saved layouts are also offered in the *default layout for new masks* combo box of the options popover.

---

## 10. Mask Consumers

The **Mask Consumers** section appears at the bottom of the panel only when one or more modules further down the pixelpipe are using this module's mask as a raster mask.
- Lists every downstream module reading this mask.
- **Left-click on any entry**: Immediately focuses that consumer module and opens its mask panel.

---

## 11. Quick Reference: Mouse and Keyboard Shortcuts

| Target | Icon | Action | Result |
|---|:---:|---|---|
| **Header Domino** | <img src="icons/masks_panel.svg" width="18" height="18" /> | Left-click | Toggle mask on / off (refused while locked) |
| **Header Domino** | <img src="icons/masks_panel.svg" width="18" height="18" /> | Right-click | Open mask options popover |
| **Header Overlay** | <img src="icons/showmask.svg" width="18" height="18" /> | Left-click | Toggle mask overlay on image |
| **Header Suppress** | <img src="icons/eye_toggle.svg" width="18" height="18" /> | Left-click | Temporarily switch off mask in preview |
| **Header Edit on Canvas** | <img src="icons/masks_eye.svg" width="18" height="18" /> | Left-click | Toggle canvas shape controls on / off |
| **Header Edit on Canvas** | <img src="icons/masks_eye.svg" width="18" height="18" /> | <kbd>Ctrl</kbd> + Left-click | Toggle restricted canvas edit mode |
| **Header Solo** | <img src="icons/soloedit.svg" width="18" height="18" /> | Left-click | Toggle solo edit mode (only the selection is shown and editable on canvas) |
| **Header Lock** | <img src="icons/mask_lock.svg" width="18" height="18" /> | Left-click | Toggle lock mask against edits, reset, presets, styles and paste |
| **Header (beside-canvas panel)** | | Left-click | Hide / fold the panel |
| **Canvas Edge Halo** | <img src="icons/masks_panel.svg" width="18" height="18" /> | Left-click | Summon / move beside-canvas panel to that side |
| **Member Row or Header** | | Left-click | Select member (click again to deselect) |
| **Member Row or Header** | | <kbd>Ctrl</kbd> + Left-click | Inline rename |
| **Member Row or Header** | | <kbd>Shift</kbd> + Left-click | Add to / take off the selection |
| **Member Row or Header** | | <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + Hover | Arm deletion (shows trash pointer & turns row red) |
| **Member Row or Header** | | <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + Click | Delete member (or reset mask on "whole mask") |
| **Group Chevron** | | Left-click | Show / hide the group's members (and select the group) |
| **Member Drag Handle** | | Drag & Drop | Rearrange or nest in tree (off while several are selected) |
| **Row Link Button** | <img src="icons/link.svg" width="18" height="18" /> | Left-click | Jump to linked shape usage or raster source module |
| **Row Link Button** | <img src="icons/link.svg" width="18" height="18" /> | <kbd>Shift</kbd> + Left-click | Unlink shape (create independent local copy) |
| **Element Invert Button** | <img src="icons/invert.svg" width="18" height="18" /> | Left-click | Invert element (every selected member when several are) |
| **Group Invert Button** | <img src="icons/invert.svg" width="18" height="18" /> | Left-click | Invert group output |
| **Group Invert Button** | <img src="icons/invert.svg" width="18" height="18" /> | <kbd>Shift</kbd> + Left-click | Invert each direct member in group |
| **Visibility Eye** | <img src="icons/eye_toggle.svg" width="18" height="18" /> | Left-click | Bypass / enable member (every selected member when several are) |
| **Visibility Eye** | <img src="icons/eye_toggle.svg" width="18" height="18" /> | <kbd>Shift</kbd> + Left-click | Solo member (hides all other members) |
| **Operator Button** | <img src="icons/masks_maximum.svg" width="18" height="18" /> | Left-click | Open combination operator menu |
| **Operator Button** | <img src="icons/masks_maximum.svg" width="18" height="18" /> | <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + Click | Delete group and all its members (reset the mask on the whole mask) |
| **Selection Delete** | <img src="icons/trash.svg" width="18" height="18" /> | Left-click / <kbd>Delete</kbd> | Delete selected member |
| **Selection Delete** | <img src="icons/trash.svg" width="18" height="18" /> | <kbd>Shift</kbd> + Left-click | Empty selected group (keeps container) |
| **Selection Wand** | <img src="icons/wand.svg" width="18" height="18" /> | Left-click | Open the menu: wrap in group, compose, simplify |
| **Selection Presets** | <img src="icons/presets.svg" width="18" height="18" /> | Left-click | Open group layout presets popover |
| **Multiple Selection: delete** | | Left-click / <kbd>Delete</kbd> | Delete everything selected |
| **Multiple Selection: move into...** | | Left-click | Choose a group to move the selection into |
| **Multiple Selection: group together...** | | Left-click | Choose an operator and put the selection in a new group |
| **Shape Add Button** | <img src="icons/masks_circle.svg" width="18" height="18" /> | Left-click | Draw a single shape |
| **Shape Add Button** | <img src="icons/masks_circle.svg" width="18" height="18" /> | <kbd>Ctrl</kbd> + Left-click | Continuous shape drawing mode |
| **Drawing Shape** | | <kbd>Escape</kbd> or Right-click | Cancel drawing current shape |
| **Channel Add Button** | | Hover | Preview channel on image (if option enabled) |
| **Channel Add Button** | | Left-click | Add channel into active group |
| **Channel Add Button** | | <kbd>Shift</kbd> + Left-click | Pin channel preview overlay on canvas |
| **Import Menu Entry** | <img src="icons/import.svg" width="18" height="18" /> | Hover | Preview external shape/mask on canvas |
| **Raster Row** | <img src="icons/masks_raster.svg" width="18" height="18" /> | Hover | Tint the area the raster mask covers on the canvas |
| **Controls Header** | | Left-click | Fold / unfold the controls section |
| **Controls Header Toggle** | | Left-click | Show / hide the output slider and boost factor of a parametric element |
| **Parametric Marker** | | Right-click | Open precise numeric entry popover |
| **Parametric Picker** | <img src="icons/colorpicker.svg" width="18" height="18" /> | Left-click | Sample canvas region for threshold |
| **Shrink or Grow Gear** | | Left-click | Open the shrink or grow settings popover |
| **Property Slider** | | Scroll wheel | Adjust the value (see *Property Sliders*) |
| **Property Slider** | | Right-click | Open the fine-tuning popup |
| **Property Slider** | | Double-click | Reset the value (opacity: 100%) |

### Shortcuts

The panel registers these actions in the *masks* section of the blending shortcuts. *Delete selection* is bound to <kbd>Delete</kbd> by default, the others have no key until you assign one:

- show/hide mask panel
- add group in selected group
- invert all members of selected group
- invert output of selected group
- invert selected element
- delete selection
- move selection into group
- group selection
- solo edit the selection
- change operator of selected group
- disable/enable selected group
- preview channel under cursor
- sticky opacity

A second section, *channel overlay*, has one action per parametric channel that pins or unpins that channel's overlay on the canvas.

The *auto-expand selected* action of the previous version is gone.
