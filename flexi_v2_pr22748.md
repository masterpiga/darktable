# darktable's New Mask Panel (a.k.a. Flexi)

darktable's new mask panel (a.k.a., Flexi) completely subsumes both the classic mask panel and the mask manager, which has been retired. All mask operations--creating shapes, setting parametric ranges, grouping, combining with boolean and arithmetic operators, inverting, soloing, refining, and reusing masks across modules--now happen in one place, directly accessible within each processing module or in a dockable side panel beside the canvas.

The mask panel organizes masks into a tree structure based on three main entities:
- **Element**: A single mask component--a drawn shape (path, brush, circle, ellipse, gradient), a single-channel parametric threshold, an external raster mask, or an AI-segmented object. An element is always an individual leaf and never a group.
- **Group**: A container that holds one or more members and combines them using a designated combination operator (such as Maximum, Product, Screen, Difference, or Sum). The top-level group representing the module's complete mask is named **whole mask**.
- **Member**: Anything held directly inside a group. A member can be a single **element** or another nested **group**.
- **Shape**: A drawn vector shape (path, brush, circle, ellipse, gradient).
- **Row** and **Header**: The visual lines in the panel list representing members (an element row or a group header). The list itself sits under the **mask layout** caption.

---

## Transitioning from Flexi UI v0.1 (PR #22707): What Changed and How Your Workflow Evolves

This section summarizes the changes in Flexi UI v0.2 (PR #22748) for users familiar with Flexi UI v0.1 (PR #22707).

### 1. Main User-Facing Changes

- **Context Menus Removed**:
  Right-click context menus on rows and group headers have been removed. All actions (such as rename, invert, compose, simplify, empty, unlink, and delete) are now directly accessible on the rows, in the selection caption, or through keyboard modifiers.
- **Clusters Removed**:
  The automatic folding of three or more adjacent shapes of the same type into cluster rows (e.g., `3 circles`) has been removed. Every member now has its own distinct row in the tree list.
- **Invert Buttons on Rows and Headers**:
  Every element row and group header now includes an invert button (<img src="icons/invert.svg" width="18" height="18" align="middle" alt="invert" />) in its header bar:
  - Element row: Clicking inverts the element.
  - Group header: Clicking inverts the group's combined output; **<kbd>Shift</kbd>+clicking inverts each direct member** inside the group.
- **Armed Deletion (<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+Click)**:
  Holding <kbd>Ctrl</kbd>+<kbd>Shift</kbd> over the list arms deletion. The pointer changes to a trash bin icon, and the hovered row or group header is highlighted in red. For a group, its tree rail also turns red to indicate every child member that will be deleted. Clicking deletes the item (or resets the mask if clicked on the root "whole mask").
- **Selection Caption Actions**:
  The selection area beneath the tree features an action caption that displays the selected item's name and four action buttons:
  - <img src="icons/trash.svg" width="18" height="18" align="middle" alt="delete" /> **Delete**: Deletes the selected item; **<kbd>Shift</kbd>+click on a group empties it** while keeping the group container.
  - <img src="icons/wand.svg" width="18" height="18" align="middle" alt="compose" /> **Compose**: Opens the operator menu to wrap the selected item into a new parent group; **<kbd>Shift</kbd>+click on a group simplifies it** by removing redundant nesting levels.
  - <img src="icons/link.svg" width="18" height="18" align="middle" alt="link" /> **Link**: Visible when a linked shape or raster mask is selected. Click to jump to the other module or instance; **<kbd>Shift</kbd>+click unlinks** the shape to create an independent local copy.
  - <img src="icons/presets.svg" width="18" height="18" align="middle" alt="presets" /> **Presets**: Opens the group layout presets popover for the selected group.
- **Toolbar Location**:
  The toolbar of creation buttons has moved from above the tree to the selection panel beneath the caption. It contains:
  - Left side: Add Group (`+`) and Import (<img src="icons/import.svg" width="18" height="18" align="middle" alt="import" />).
  - Right side: Drawn shape buttons (Path, Brush, Circle, Ellipse, Gradient, AI Object) and parametric channel buttons.
- **Group Layout Presets Popover**:
  The presets menu is now an interactive popover that targets the currently selected group instead of replacing the whole mask. Each preset provides three application buttons: **Replace**, **Merge**, and **Add as subgroup**. Tooltips show an outline of the group tree structure, and the current group or mask layout can be saved as a preset from within the popover.
- **Parametric Sliders**:
  Per-range bypass eye toggles have been removed. Clicking the channel icon opens a menu to choose which sliders the row shows: input, output, or both. Each slider now has its own color picker button (<img src="icons/colorpicker.svg" width="18" height="18" align="middle" alt="color picker" />) directly at its right edge.
- **Preset Guidance Notes**:
  Preset notes have been moved out of the tree into a collapsible **help** section in the selection panel.
- **Modifier Key Conventions**:
  The <kbd>Alt</kbd> key is no longer used in the mask panel. Modifiers follow four rules across all rows and buttons:
  - <kbd>Ctrl</kbd>: Rename.
  - <kbd>Shift</kbd>: Toggle disclosure, or execute an action variant (Solo on eye, Invert All on group invert, Empty on delete, Simplify on compose, Unlink on link).
  - <kbd>Ctrl</kbd>+<kbd>Shift</kbd>: Armed deletion with red visual indicator.
  - Plain click: Select or primary action.

---

### 2. How the UX Changes

- **Direct Controls Instead of Menus**:
  Actions previously located in right-click context menus are now placed directly on member rows (such as the invert button, eye toggle, and expander chevron) or in the selection caption (compose, delete, link, and presets). Right-clicking a row no longer displays a menu.
- **One Row Per Member**:
  Because cluster rows have been removed, the list is a direct one-to-one representation of the shapes and channels in the mask. Selecting, reordering, and dragging individual shapes does not require expanding a cluster first.
- **Deletion Scope Preview**:
  Holding <kbd>Ctrl</kbd>+<kbd>Shift</kbd> turns the cursor into a trash bin and highlights the target in red before any click is registered. If the target is a group, the group's vertical tree rail turns red as well, indicating that all nested child members will be deleted along with it.

---

### 3. Effects on Workflow

- **Targeted Preset Application**:
  In v0.1, choosing a layout preset replaced the entire mask structure from the root. In v0.2, presets apply to the currently selected group. You can insert a preset layout as a subgroup or merge it into an existing group, allowing presets to be combined within a single mask.
- **One-Click Inversions**:
  Inverting does not require opening a menu or using keyboard modifiers on the operator icon. Click the invert button (<img src="icons/invert.svg" width="18" height="18" align="middle" alt="invert" />) on any element row or group header. To invert every member in a group, <kbd>Shift</kbd>+click the group's invert button.
- **Modifier Consistency and OS Compatibility**:
  The removal of <kbd>Alt</kbd> avoids shortcuts being intercepted by window managers (such as on Linux desktop environments where <kbd>Alt</kbd>+click moves windows). The four-modifier convention (<kbd>Ctrl</kbd> for rename, <kbd>Shift</kbd> for disclosures or secondary actions, <kbd>Ctrl</kbd>+<kbd>Shift</kbd> for delete, plain click for primary actions) is shared across all list items and buttons.

---

## 1. Panel Header Controls

At the top of every module's mask panel sits the header bar:

- <img src="icons/masks_panel.svg" width="18" height="18" align="middle" alt="domino" /> **Mask On/Off (domino icon)**
  - **Left-click**: Toggles masking on or off for the current module. A locked mask cannot be switched on or off ("the mask is locked").
  - **Right-click**: Opens the **Mask Options popover** to configure panel behavior, color spaces, and layout presets.
- <img src="icons/masks_eye.svg" width="18" height="18" align="middle" alt="edit on canvas" /> **Edit Mask on Canvas (eye icon)**
  - **Left-click**: Shows or hides the handles of the drawn mask elements on the image canvas.
  - **<kbd>Ctrl</kbd>+Left-click**: Activates restricted edit mode (shapes cannot be moved or resized).
  - Greyed out while the mask holds no drawn shape or is locked.
- <img src="icons/soloedit.svg" width="18" height="18" align="middle" alt="solo edit" /> **Solo Edit (spotlight icon)**
  - **Left-click**: While enabled, only the selected shape, or the shapes of the selected group, are shown and editable on the canvas. The other members still contribute to the mask.
  - Greyed out like the edit button.
- <img src="icons/showmask.svg" width="18" height="18" align="middle" alt="mask overlay" /> **Display Mask (yellow overlay icon)**
  - **Left-click**: Toggles the mask overlay on the image: gray, tinted yellow where the module applies (the stronger the yellow, the stronger the effect). An empty mask shows no overlay, since the module then applies everywhere. Switches the module and the mask on if they are off.
- <img src="icons/eye_toggle.svg" width="18" height="18" align="middle" alt="suppress mask" /> **Temporarily Switch Off Mask (eye toggle icon)**
  - **Left-click**: Compares the image with and without the mask: the module then applies everywhere, at the blend opacity. Only the center view is affected, not exports, and the mask switches back on when you leave the module. Greyed out while the module or its mask is off.
- <img src="icons/mask_lock.svg" width="18" height="18" align="middle" alt="lock mask" /> **Lock Mask (padlock icon)**
  - **Left-click**: Locks or unlocks the mask. A locked mask cannot be edited and is not affected by module reset, presets, styles or pasted module parameters; for instance, switching a *color calibration* instance from no preset to a preset keeps the mask it had. Discarding the history still removes it. Only shown on modules that support masks.
- **Show/Hide Arrow**
  - **Left-click**: Hides or shows the mask panel's body. Clicking an empty part of the header does the same. Not shown in the utility module position, where the module's own header folds the panel.

### Panel Visibility

When the mask panel is a **separate panel beside the canvas**, clicking its header makes it disappear. It comes back by clicking the domino button in the darkroom bottom toolbar, or the domino icon and arrow of the halo that shows when the pointer rests at the edge of the canvas where the panel docks. The handle on the panel's edge also hides it on a click, and resizes it when dragged.

---

## 2. Mask Options Popover

The Mask Options popover configures interface settings, default layouts, and color spaces specific to the mask panel.

### Accessing the Popover
- **Right-click the On/Off domino button** (<img src="icons/masks_panel.svg" width="18" height="18" align="middle" alt="domino" />) in any module's mask panel header.
- **Right-click the mask panel toggle button** in the darkroom bottom toolbar.

### Popover Settings

#### 1. Blend Color Space
Selects the color space used to evaluate the module's mask and its parametric channels:
- **default**: Uses the module's native working color space.
- **RGB (scene)**: Evaluates the mask in unbounded linear RGB. Preserves high dynamic range values in scene-referred pipeline modules.
- **RGB (display)**: Evaluates the mask in display-referred RGB.
- **Lab**: Perceptual Lab color space. **Only offered on modules that natively process in Lab**; it is withheld on RGB modules to avoid invalid color space conversions before the input color profile or after the output color profile.

This section is only shown for modules that support parametric masks, and is greyed out while the mask is locked. A change that would invalidate parametric elements already in the mask asks for confirmation first and deletes them on a yes.

#### 2. Blend Mask Panel Position
Controls where the mask panel is hosted in the interface:
- **Embedded within each module (default)**: The mask panel lives directly inside each processing module's expandable body.
- **Utility module, left panel**: Moves the entire panel into a standalone module labeled **masking** in the left sidebar. Focusing any processing module directs this utility module to display and edit that module's mask.
- **Separate panel, beside the canvas**: Detaches the mask panel into a dockable, resizable panel positioned directly beside the image canvas.
  - It can be summoned to either side of the canvas: approaching either the left or right canvas edge with the pointer reveals a halo overlay spanning the whole height of the canvas with the domino icon (<img src="icons/masks_panel.svg" width="18" height="18" align="middle" alt="domino" />) and an inward arrow. Clicking the halo opens the panel on that side.
  - When the panel is already open, approaching the opposite canvas edge reveals the halo on that side; clicking it relocates the panel across to the opposite edge of the canvas.
- **Show focused module under header** (enabled by default; only for the utility and beside-canvas positions): Shows a row under the panel's header naming the module whose mask the panel displays, and its instance name if it has one.

#### 3. Interface Options
- **Sticky opacity (enabled by default)**: When enabled, newly created shapes inherit the opacity value last used by any shape. When disabled, every new shape starts at 100% opacity.
- **Auto-expand selected (disabled by default)**: When enabled, selecting a member in the tree automatically expands it while collapsing the previously active member. When disabled, rows only expand or collapse when clicking their disclosure chevrons.
- **Member properties in subpanel (enabled by default)**: When enabled, parameters for the selected member (e.g., shape radius, feathering, parametric sliders) are displayed in a dedicated card below the tree. When disabled, parameters expand directly inside each row in the tree.
- **Show the mask panel's resize handle (enabled by default)**: When using the beside-canvas panel, displays a visible grip handle with an arrow indicating the collapse direction. When disabled, the handle is hidden but the edge can still be dragged to resize.

#### 4. Parametric Element Options
- **Reuse the last picked area (enabled by default)**: When sampling values on the canvas with the color picker, the sampled bounding box is remembered. Clicking the color picker on subsequent parametric channels immediately applies that same sample area without having to redraw a selection on the image.
- **Preview channel under cursor (disabled by default)**: When enabled, hovering the pointer over any add-parametric-channel button displays a live center-view preview of that channel after a brief pause, allowing you to preview the channel's mask before adding it.

#### 5. Default Group Layout and Notes
- **Default group layout**: Selects which layout preset a module's mask starts with when it is switched on for the first time. The built-in choices are `drawn (classic)` (the default), `parametric (classic)`, `drawn + parametric (classic)` and `drawn + parametric, minus an area (advanced)`, followed by your own saved presets.
- **Show preset help notes (beginners)** (enabled by default): When enabled, groups generated from built-in presets display a help note in the selection panel explaining their purpose and suggested usage. When disabled, the help section is not shown. This switch is the last entry of the popover, closing the *default group layout* section.

---

## 3. Blending Settings

When the mask panel is expanded, the top section provides module-level blending controls:

- **Blend Mode dropdown**: Blends the module's processed image output with the input image (Normal, Multiply, Screen, Difference, etc.). This controls image layer blending only; it does not affect how mask elements combine.
- **Blend Order (input/output icon)**: Toggles the blending order between the input and the output of the module. By default the output is blended on top of the input; clicking reverses this (input on top of output).
- **Blend Parameter slider (labeled "fulcrum")**: Appears only for blend modes that accept an adjustable parameter; adjusts the fulcrum used by those blending operations.
- **Opacity slider (under the blend controls)**: Controls how strongly the module's blended effect applies to the image. Double-clicking resets it to 100%. A warning badge appears next to its label while the opacity is below 10%.

### Blend Opacity vs. "Whole Mask" Group Opacity
- **Blend opacity** scales the final mix of the module's processed image over the input image. If no mask is active, the module effect applies uniformly at this percentage. Blend opacity attenuates the final image blend without altering the mask itself.
- **"Whole mask" group opacity** (in the properties section while the whole mask is selected) scales the values of the mask itself (from 0 to 1). Changing the "whole mask" opacity directly modifies the resulting mask data, which affects the yellow mask overlay preview and scales the raster mask exported to downstream modules.

---

## 4. The Mask Layout (The Member Tree)

The list under the **mask layout** caption is a tree: it represents the hierarchy of groups and elements that compose the mask. Its top entry is the whole mask's own group.

### Tree Rail and Nesting
Groups can nest inside other groups up to 8 levels deep. A continuous vertical tree rail runs down the left edge of each group, ticking across to each child member's header, visually communicating the group's boundaries and hierarchy.

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
Each group displays a header bar with the following controls:
- **Operator Button (left icon and caret)**:
  - Displays the current combination operator icon (e.g., <img src="icons/masks_maximum.svg" width="18" height="18" align="middle" alt="maximum" />).
  - **Left-click**: Opens the operator menu to switch how members in this group combine.
- **Group Title**:
  - Displays the group name (defaults to `<operator>-<ordinal>`, e.g., `max-1`, or `whole mask` for root).
  - **Left-click**: Selects the group; clicking it again selects the whole mask.
  - **<kbd>Ctrl</kbd>+Left-click**: Opens an inline text field to rename the group (the whole mask cannot be renamed).
  - **<kbd>Shift</kbd>+Left-click**: Shows or hides the group's members.
  - **<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+Left-click**: Deletes the group and every member in it (resets the mask on the whole mask).
  - **Drag**: Drag by the title or header to reorder the group or nest it inside another group.
- <img src="icons/invert.svg" width="18" height="18" align="middle" alt="invert" /> **Invert Button**:
  - **Left-click**: Inverts the group's composite output. When inverted, the button and operator icon display an inverted highlight style (<img src="icons/masks_maximum_inv.svg" width="18" height="18" align="middle" alt="inverted operator" />).
  - **<kbd>Shift</kbd>+Left-click**: Inverts each direct member inside the group in one action.
- <img src="icons/eye_toggle.svg" width="18" height="18" align="middle" alt="visibility" /> **Visibility (eye icon)**:
  - **Left-click**: Disables (bypasses) the group.
  - **<kbd>Shift</kbd>+Left-click**: Solos the group (evaluates only this group, temporarily hiding all other members in the mask). Not available on the whole mask.
- **Expander Chevron**:
  - **Left-click**: Collapses or expands the group's members in the list. Greyed out on an empty group.

### Member Rows
Every element inside a group is displayed as a row:
- **Type Icon / Drag Handle (left column)**:
  - Displays the element type icon: <img src="icons/masks_path.svg" width="18" height="18" align="middle" alt="path" /> Path, <img src="icons/masks_brush.svg" width="18" height="18" align="middle" alt="brush" /> Brush, <img src="icons/masks_circle.svg" width="18" height="18" align="middle" alt="circle" /> Circle, <img src="icons/masks_ellipse.svg" width="18" height="18" align="middle" alt="ellipse" /> Ellipse, <img src="icons/masks_gradient.svg" width="18" height="18" align="middle" alt="gradient" /> Gradient, <img src="icons/masks_raster.svg" width="18" height="18" align="middle" alt="raster" /> Raster, or <img src="icons/masks_object.svg" width="18" height="18" align="middle" alt="object" /> AI Object. A parametric channel shows its channel code instead (e.g., `L`, `Cz`, `hz`).
  - **Drag**: Drag to reorder within the group or drop into another group. An insertion line indicates the target position.
- **Member Name**:
  - Displays the member name (e.g., `circle #1`, `L in`, or a custom label).
  - **Left-click**: Selects the member (click again to deselect).
  - **<kbd>Ctrl</kbd>+Left-click**: Opens an inline text field to rename the member.
  - **<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+Left-click**: Deletes the element.
  - **<kbd>Shift</kbd>+Left-click**: Shows or hides the row's expanded controls. Shape and raster rows have them only while **member properties in subpanel** is switched off in the options popover (their properties then show in the properties section instead); a parametric row always has them (opacity and boost factor).
- **Indicators and Links**:
  - <img src="icons/link.svg" width="18" height="18" align="middle" alt="link" /> **Linked Shape / Raster Link**: On linked shapes or raster masks, clicking jumps to the other module/instance.
  - **Low-Opacity Warning Badge**: Appears at the end of the name if an element's opacity is below 10%, warning that its contribution is practically invisible. A group holding such a member at any depth shows it too; its tooltip says which.
- <img src="icons/invert.svg" width="18" height="18" align="middle" alt="invert" /> **Invert Button**:
  - **Left-click**: Inverts the element. When inverted, the icon plate displays an inverted highlight (<img src="icons/masks_circle_inv.svg" width="18" height="18" align="middle" alt="inverted circle" />).
- <img src="icons/eye_toggle.svg" width="18" height="18" align="middle" alt="visibility" /> **Visibility (eye icon)**:
  - **Left-click**: Disables (bypasses) the element.
  - **<kbd>Shift</kbd>+Left-click**: Solos the element (evaluates only this single element).

### Armed Deletion (<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+Click)
Holding <kbd>Ctrl</kbd>+<kbd>Shift</kbd> over the list activates armed deletion:
- The pointer transforms into a **trash bin icon**.
- The hovered row or group header turns **red**.
- For a group, its **vertical rail turns red**, clearly highlighting all descendant members that will be deleted.
- Clicking immediately deletes the targeted element or group (or resets the mask if clicked on the root "whole mask").

---

## 5. The Selection Area & Action Toolbar

Beneath the tree view sits the **Selection Area** (`#masks-selection-area`), which consolidates selection management, actions, the creation toolbar, properties, and refinements.

### The Selection Caption
The top row of the selection area names the currently active item. Delete and compose sit on its left, link and presets on its right. A button greyed out has nothing to act on; while a shape is still being drawn or the mask is locked, delete, compose and presets all are.
- <img src="icons/trash.svg" width="18" height="18" align="middle" alt="delete" /> **Delete Button (trash icon)**:
  - **Left-click**: Deletes the selected member (or resets the mask to the default group layout if "whole mask" is selected). On a linked element, only deletes it here; the other rows and modules it is linked with keep it. The tooltip also names the shortcut keys currently bound to *delete selection*.
  - **<kbd>Shift</kbd>+Left-click**: When a group is selected, **empties the group** (removes all its members while keeping the group container in place, with its settings and name).
- <img src="icons/wand.svg" width="18" height="18" align="middle" alt="compose" /> **Compose Button (wand icon)**:
  - **Left-click**: Opens the operator chooser menu to use the selection as a building block for a larger mask, for example to subtract from it or to intersect it with something else: it goes into a new group of the chosen operator, with an empty group on top to draw into.
  - **<kbd>Shift</kbd>+Left-click**: When a group is selected, **simplifies the group** (removes redundant nesting levels where collapsing them produces the exact same mask output).
- **Selection Name Box (center)**:
  - Shows the name of the selected member (or `whole mask`).
  - **<kbd>Ctrl</kbd>+Left-click** or **double-click**: Renames the selected member (the whole mask cannot be renamed). A plain click does nothing.
  - **Canvas Hover Peek**: When hovering the mouse over a shape on the image canvas, the caption displays that shape's name, highlighted to show that it is a peek, not the selection.
- <img src="icons/link.svg" width="18" height="18" align="middle" alt="link" /> **Link Button (chain icon)**:
  - Appears only when the selected element is a shared shape or a raster mask.
  - **Left-click**: Jumps to the linked instance, or focuses the raster mask's source module to edit its mask on the canvas.
  - **<kbd>Shift</kbd>+Left-click**: **Unlinks the shape**, giving this module its own independent local copy so subsequent edits do not affect other modules.
- <img src="icons/presets.svg" width="18" height="18" align="middle" alt="presets" /> **Group Layout Presets Button (presets icon)**:
  - **Left-click**: Opens the interactive **Group Layout Presets Popover** targeting the selected group. Greyed out when no group is selected, and for raster masks, which have no group layouts.

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

## 6. Member Properties & Refinements Sub-Panels

Beneath the toolbar, the selection area hosts collapsible cards that adapt dynamically to what is selected:

### 1. Help (Preset Guidance Notes)
When a group created from a built-in preset is selected (and "show preset help notes (beginners)" is enabled in the options popover), a help section appears here explaining the intended purpose, typical workflow, and tips for working with that preset layout. A note of several pages has arrows to turn them.

### 2. Member Properties
Shown while **member properties in subpanel** is enabled (the default); otherwise the same controls expand inside each row. Displays the parameters of the selected member, or of the one hovered on the canvas:
- **Drawn Shapes**: Sliders for opacity and size, plus those that apply to the shape: hardness (brushes), fade-out border, rotation, and curvature and compression (gradients). Paths also have a *shrink or grow* slider, whose unit (px or % of path size) is toggled by the button at its right end.
- **AI Objects**: Sliders and options for opacity, cleanup, smoothing, *refine mask boundary* and *apply as paths*.
- **Raster masks**: Opacity only.
- **Parametric Channels**: The sliders of a channel are always shown in its row. The properties section adds opacity and, for channels that have one, the boost factor.
- **Groups**: Group opacity slider.
- **Active Drawing**: While drawing a shape on canvas, displays creation controls before committing the shape.

A parametric row carries its own controls:
- **Channel icon (left of the sliders)**: Opens a *select visible sliders* menu: input, output, or input and output.
- **Dual Gradient Sliders**: Each slider features 4 threshold markers (black point, feather start, feather end, white point).
- **Right-click on any marker**: Opens a precise numeric input popover for that threshold.
- **Dedicated Color Pickers (<img src="icons/colorpicker.svg" width="18" height="18" align="middle" alt="color picker" />)**: Located directly at the right edge of each slider. Left-click sets the slider's range from an area dragged on the image; <kbd>Ctrl</kbd>+click picks a GUI color (point) shown on the sliders; <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+click does the same for an area.
- **Boost Factor**: Increases the range of values the channel can match over 100%, in EV.

#### Property Sliders
The sliders are darktable's usual ones:
- **Scroll wheel**: Adjusts the value, with the usual fine and coarse modifiers, unless *mouse wheel scrolls modules side panel by default* is enabled in the preferences, in which case the wheel scrolls the panel and <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+wheel adjusts the value.
- **Right-click**: Opens the fine-tuning popup.
- **Double-click**: Resets the value: opacity to 100%, the others to what they were when the properties were shown, or to 0.
- **Range**: The *fade-out border* slider stops at 100%. Scrolling with <kbd>Shift</kbd> over the shape on the canvas can set a wider border for some shapes.

### 3. Mask Refinements (`masks_refinement`)
Refinements provide edge-aware smoothing, feathering, and tonal adjustments. The refinement automatically applies to the current selection scope:
- **Element Scope**: Refines that specific element before it is combined by its group's operator.
- **Group Scope**: Refines the group's composite output after its operator combines its members.
- **Whole Mask Scope**: Refines the final combined mask of the module.

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

## 7. Group Layout Presets Popover

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
At the bottom of the popover, click **"save this group's layout as preset..."** (or **"save the whole mask's layout as preset..."**). The save dialog takes immediate keyboard focus, confirms with <kbd>Enter</kbd>, and cancels with <kbd>Escape</kbd>.

---

## 8. Mask Consumers

The **Mask Consumers** section appears at the bottom of the panel only when one or more modules further down the pixelpipe are using this module's mask as a raster mask.
- Lists every downstream module reading this mask.
- **Left-click on any entry**: Immediately focuses that consumer module and opens its mask panel.

---

## 9. Quick Reference: Mouse and Keyboard Shortcuts

| Target | Icon | Action | Result |
|---|:---:|---|---|
| **Header Domino** | <img src="icons/masks_panel.svg" width="18" height="18" /> | Left-click | Toggle mask on / off (refused while locked) |
| **Header Domino** | <img src="icons/masks_panel.svg" width="18" height="18" /> | Right-click | Open mask options popover |
| **Header Edit on Canvas** | <img src="icons/masks_eye.svg" width="18" height="18" /> | Left-click | Toggle canvas shape controls on / off |
| **Header Edit on Canvas** | <img src="icons/masks_eye.svg" width="18" height="18" /> | <kbd>Ctrl</kbd> + Left-click | Toggle restricted canvas edit mode |
| **Header Solo** | <img src="icons/soloedit.svg" width="18" height="18" /> | Left-click | Toggle solo edit mode (only the selection is shown and editable on canvas) |
| **Header Overlay** | <img src="icons/showmask.svg" width="18" height="18" /> | Left-click | Toggle mask overlay on image |
| **Header Suppress** | <img src="icons/eye_toggle.svg" width="18" height="18" /> | Left-click | Temporarily switch off mask in preview |
| **Header Lock** | <img src="icons/mask_lock.svg" width="18" height="18" /> | Left-click | Toggle lock mask against edits, reset, presets, styles and paste |
| **Header (beside-canvas panel)** | | Left-click | Hide the panel |
| **Canvas Edge Halo** | <img src="icons/masks_panel.svg" width="18" height="18" /> | Left-click | Summon / move beside-canvas panel to that side |
| **Member Row or Header** | | Left-click | Select member (click again to deselect) |
| **Member Row or Header** | | <kbd>Ctrl</kbd> + Left-click | Inline rename |
| **Member Row or Header** | | <kbd>Shift</kbd> + Left-click | Toggle disclosure (expanded controls, or child members) |
| **Member Row or Header** | | <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + Hover | Arm deletion (shows trash pointer & turns row red) |
| **Member Row or Header** | | <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + Click | Delete member (or reset mask on "whole mask") |
| **Member Drag Handle** | | Drag & Drop | Rearrange or nest in tree |
| **Element Invert Button** | <img src="icons/invert.svg" width="18" height="18" /> | Left-click | Invert element |
| **Group Invert Button** | <img src="icons/invert.svg" width="18" height="18" /> | Left-click | Invert group output |
| **Group Invert Button** | <img src="icons/invert.svg" width="18" height="18" /> | <kbd>Shift</kbd> + Left-click | Invert each direct member in group |
| **Visibility Eye** | <img src="icons/eye_toggle.svg" width="18" height="18" /> | Left-click | Bypass / enable member |
| **Visibility Eye** | <img src="icons/eye_toggle.svg" width="18" height="18" /> | <kbd>Shift</kbd> + Left-click | Solo member (hides all other members) |
| **Operator Button** | <img src="icons/masks_maximum.svg" width="18" height="18" /> | Left-click | Open combination operator menu |
| **Operator Button** | <img src="icons/masks_maximum.svg" width="18" height="18" /> | <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + Click | Delete group and all its members (reset the mask on the whole mask) |
| **Selection Delete** | <img src="icons/trash.svg" width="18" height="18" /> | Left-click / <kbd>Delete</kbd> | Delete selected member |
| **Selection Delete** | <img src="icons/trash.svg" width="18" height="18" /> | <kbd>Shift</kbd> + Left-click | Empty selected group (keeps container) |
| **Selection Compose** | <img src="icons/wand.svg" width="18" height="18" /> | Left-click | Wrap selection into new parent group |
| **Selection Compose** | <img src="icons/wand.svg" width="18" height="18" /> | <kbd>Shift</kbd> + Left-click | Simplify selected group |
| **Selection Link** | <img src="icons/link.svg" width="18" height="18" /> | Left-click | Jump to linked shape usage or raster source module |
| **Selection Link** | <img src="icons/link.svg" width="18" height="18" /> | <kbd>Shift</kbd> + Left-click | Unlink shape (create independent local copy) |
| **Selection Presets** | <img src="icons/presets.svg" width="18" height="18" /> | Left-click | Open group layout presets popover |
| **Shape Add Button** | <img src="icons/masks_circle.svg" width="18" height="18" /> | Left-click | Draw a single shape |
| **Shape Add Button** | <img src="icons/masks_circle.svg" width="18" height="18" /> | <kbd>Ctrl</kbd> + Left-click | Continuous shape drawing mode |
| **Drawing Shape** | | <kbd>Escape</kbd> or Right-click | Cancel drawing current shape |
| **Channel Add Button** | | Hover | Preview channel on image (if option enabled) |
| **Channel Add Button** | | Left-click | Add channel into active group |
| **Channel Add Button** | | <kbd>Shift</kbd> + Left-click | Pin channel preview overlay on canvas |
| **Import Menu Entry** | <img src="icons/import.svg" width="18" height="18" /> | Hover | Preview external shape/mask on canvas |
| **Channel Icon (Props)** | | Left-click | Choose input, output or both sliders |
| **Parametric Marker** | | Right-click | Open precise numeric entry popover |
| **Parametric Picker** | <img src="icons/colorpicker.svg" width="18" height="18" /> | Left-click | Sample canvas region for threshold |
| **Property Slider** | | Scroll wheel | Adjust the value (see *Property Sliders*) |
| **Property Slider** | | Right-click | Open the fine-tuning popup |
| **Property Slider** | | Double-click | Reset the value (opacity: 100%) |
