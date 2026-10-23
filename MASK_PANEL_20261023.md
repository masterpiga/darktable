# darktable's New Mask Panel (a.k.a. Flexi)

darktable's new mask panel (a.k.a., Flexi) completely subsumes both the classic mask panel and the mask manager, which has been retired. All mask operations--creating shapes, setting parametric ranges, grouping, combining with boolean and arithmetic operators, inverting, soloing, refining, and reusing masks across modules--now happen in one place, directly accessible within each processing module or in a dockable side panel beside the canvas.

Before diving into the interface, it helps to understand how a mask is structured in Flexi. A mask is a tree built from three basic concepts:
- **Element**: A single mask component--a drawn shape (path, brush, circle, ellipse, gradient), a single-channel parametric threshold, an external raster mask, or an AI-segmented object. An element is always an individual leaf and never a group.
- **Group**: A container that holds one or more members and combines them using a designated combination operator (such as Maximum, Product, Screen, Difference, or Sum). The top-level group representing the module's complete mask is named **whole mask**.
- **Member**: Anything held directly inside a group. A member can be a single **element** or another nested **group**.
- **Shape**: A drawn vector shape (path, brush, circle, ellipse, gradient).
- **Row** and **Header**: The visual lines in the panel list representing members (an element row or a group header).

---

## Transitioning from master: What Changed and How Your Workflow Evolves

For users already familiar with the `master` version of Flexi masks, the `flexi-ui-revamp` branch brings major refinements to usability, predictability, and speed.

### 1. Main User-Facing Changes

- **Right-Click Context Menus Removed**:
  In `master`, secondary actions (such as rename, invert, compose, simplify, empty, unlink, and delete) were tucked away inside right-click context menus. In the revamp, right-click menus have been completely eliminated. Every action is now directly exposed via dedicated on-row buttons, the selection action caption, or clean keyboard modifiers.
- **Shape Clusters Eliminated**:
  In `master`, three or more adjacent shapes of the same type automatically collapsed into a cluster row (e.g., `3 circles`). While this saved vertical space, it obscured individual shapes and added an extra click to access or move them. In the revamp, clusters are gone: every member always has its own explicit, distinct row in the tree.
- **Dedicated Invert Buttons on Every Row and Header**:
  Every element row and group header now features a dedicated invert button (<img src="icons/invert.svg" width="18" height="18" align="middle" alt="invert" />) directly in its drawer:
  - Element row: Clicking inverts the element.
  - Group header: Clicking inverts the group's composite output; **<kbd>Shift</kbd>+clicking inverts each direct member** inside the group.
- **Armed Deletion with Visual Hover Feedback (<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+Click)**:
  Deleting a member is armed by holding <kbd>Ctrl</kbd>+<kbd>Shift</kbd> over the list. While held, the cursor morphs into a trash bin, and hovering over any member row or group header shades it bright red. For a group, its entire vertical tree rail turns red as well, visually indicating every member that will be deleted. Clicking immediately deletes that member (or resets the mask if clicked on the root "whole mask").
- **Unified Selection Caption with Direct Actions**:
  The selection area beneath the tree features an action caption that displays the selected member's name and hosts direct action buttons:
  - <img src="icons/trash.svg" width="18" height="18" align="middle" alt="delete" /> **Delete button**: Click to delete the selection; **<kbd>Shift</kbd>+click on a group to empty it** (removes its members while keeping the group container).
  - <img src="icons/wand.svg" width="18" height="18" align="middle" alt="compose" /> **Compose button**: Click to wrap the selection into a new parent group with a chosen operator; **<kbd>Shift</kbd>+click on a group to simplify it** (removes redundant nesting levels).
  - <img src="icons/link.svg" width="18" height="18" align="middle" alt="link" /> **Link button**: Visible when a linked shape or raster mask is selected. Click to jump to the other module/instance; **<kbd>Shift</kbd>+click to unlink** (creates an independent local copy).
  - <img src="icons/presets.svg" width="18" height="18" align="middle" alt="presets" /> **Presets button**: Opens the group layout presets popover targeting the selected group.
- **Relocated Creation Toolbar**:
  The creation toolbar has moved from the top of the tree to the selection panel beneath the caption. It groups the creation tools logically into:
  - Left side: Add Group (`+`) and Import (<img src="icons/import.svg" width="18" height="18" align="middle" alt="import" />).
  - Right side: Drawn Shapes (Path, Brush, Circle, Ellipse, Gradient, AI Object) and Parametric Channels.
- **Overhauled Group Layout Presets Popover**:
  Replacing the former flat menu that unconditionally overwrote the whole mask, presets now open in an interactive popover targeting the **currently selected group**. Each preset provides three distinct application modes: **Replace**, **Merge**, and **Add as subgroup**. Tooltips show a formatted tree outline of the group structure, and new custom layouts can be saved with one click.
- **Streamlined Parametric Channel Sliders**:
  Per-range bypass eye toggles have been removed. Clicking the channel abbreviation icon toggles between showing the input or output slider, and each slider now features its own dedicated color picker button (<img src="icons/colorpicker.svg" width="18" height="18" align="middle" alt="color picker" />) directly at its right edge.
- **Dedicated Preset Guidance Notes**:
  Preset notes have been moved out of the tree into a collapsible `#masks-help-section` card in the selection panel, keeping the member tree uncluttered.
- **Streamlined Modifier Standard (Retirement of <kbd>Alt</kbd>)**:
  The <kbd>Alt</kbd> key is no longer used in the mask panel, avoiding conflicts with operating system and desktop window managers. Modifiers follow four clear rules:
  - <kbd>Ctrl</kbd>: Rename.
  - <kbd>Shift</kbd>: Toggle disclosure/expanded controls, or execute an action variant (Solo on eye, Invert All on group invert, Empty on delete, Simplify on compose, Unlink on link).
  - <kbd>Ctrl</kbd>+<kbd>Shift</kbd>: Armed deletion with red visual indicator.
  - Plain click: Select or default action.

---

### 2. How the UX Changes

- **Total Discoverability**:
  You no longer need to right-click rows or guess what actions exist. All operations--inverting, unlinking, composing, deleting, and applying presets--are immediately visible as buttons with explanatory tooltips.
- **Visual Predictability and Determinism**:
  Removing shape clusters prevents the layout from collapsing unexpectedly when adding multiple shapes. Dragging and dropping members in the tree is now completely deterministic, with an explicit insertion line showing exactly where an element or group will land.
- **Error-Free Deletions**:
  The red armed-deletion feedback gives total clarity before any destructive action takes place. You see exactly which subgroup and descendant branches will be discarded before releasing the click.
- **Clear Canvas-to-List Synchronization**:
  Hovering over an element on the canvas instantly peeks its name in the selection caption (`.dt_masks_peek`), making it effortless to identify shapes in dense compositions.

---

### 3. Effects on Workflow

- **Accelerated Multi-Shape Grouping**:
  Wrapping shapes into groups or simplifying redundant layers is now instant: select a member, click the Compose wand (<img src="icons/wand.svg" width="18" height="18" align="middle" alt="compose" />), and pick an operator. To flatten unnecessary nesting, simply <kbd>Shift</kbd>+click the wand.
- **Modular, Non-Destructive Preset Application**:
  Previously, loading a layout preset reset the entire mask. Now you can select any subgroup within an existing mask and apply a preset as an addition or merge, enabling modular mask building (e.g., adding an exclusion or highlights-subgroup inside a complex drawn mask).
- **Faster Parametric Adjustments**:
  Having independent color pickers beside each slider allows rapid sampling of shadows and highlights without having to switch modes or dig into submenus.
- **Uncompromised Keyboard Ergonomics**:
  With <kbd>Alt</kbd> eliminated, multi-modifier shortcuts (<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+Click to delete, <kbd>Shift</kbd>+Click on eye to solo, <kbd>Ctrl</kbd>+Click to rename) function reliably across Linux, macOS, and Windows without window manager interference.

---

## 1. Panel Header Controls

At the top of every module's mask panel sits the header bar:

- <img src="icons/masks_panel.svg" width="18" height="18" align="middle" alt="domino" /> **Mask On/Off (domino icon)**
  - **Left-click**: Toggles masking on or off for the current module.
  - **Right-click**: Opens the **Mask Options popover** to configure panel behavior, color spaces, and layout presets.
- <img src="icons/masks_eye.svg" width="18" height="18" align="middle" alt="edit on canvas" /> **Edit Mask on Canvas (eye icon)**
  - **Left-click**: Shows or hides shape control handles on the image canvas.
  - **<kbd>Ctrl</kbd>+Left-click**: Activates restricted edit mode (allows adjusting existing shape handles while preventing accidental creation of new shapes).
- <img src="icons/soloedit.svg" width="18" height="18" align="middle" alt="solo edit" /> **Solo Edit (spotlight icon)**
  - **Left-click**: Displays only the module's own mask shapes on canvas, hiding shape outlines from all other modules.
- <img src="icons/showmask.svg" width="18" height="18" align="middle" alt="mask overlay" /> **Show Mask (yellow overlay icon)**
  - **Left-click**: Toggles the semi-transparent mask preview overlay on the image canvas (yellow indicates masked areas).
- <img src="icons/eye_toggle.svg" width="18" height="18" align="middle" alt="suppress mask" /> **Temporarily Switch Off Mask (eye toggle icon)**
  - **Left-click**: Temporarily disables the mask in the center view to compare the image with and without masking. The mask switches back on when you navigate away from the module.
- <img src="icons/mask_lock.svg" width="18" height="18" align="middle" alt="lock mask" /> **Lock Mask (padlock icon)**
  - **Left-click**: Locks the mask against changes. A locked mask will not be overwritten when applying module presets or styles, pasting history, or resetting the module.

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
- **RGB (scene-referred)**: Evaluates the mask in unbounded linear RGB. Preserves high dynamic range values in scene-referred pipeline modules.
- **RGB (display-referred)**: Evaluates the mask in display-referred RGB.
- **Lab**: Perceptual Lab color space. **Only offered on modules that natively process in Lab**; it is withheld on RGB modules to avoid invalid color space conversions before the input color profile or after the output color profile.

#### 2. Blend Mask Panel Position
Controls where the mask panel is hosted in the interface:
- **Embedded within each module (default)**: The mask panel lives directly inside each processing module's expandable body.
- **Utility module, left panel**: Moves the entire panel into a standalone module labeled **masking** in the left sidebar. Focusing any processing module directs this utility module to display and edit that module's mask.
- **Separate panel, beside the canvas**: Detaches the mask panel into a dockable, resizable panel positioned directly beside the image canvas.
  - It can be summoned to either side of the canvas: approaching either the left or right canvas edge with the pointer reveals a halo overlay spanning the whole height of the canvas with the domino icon (<img src="icons/masks_panel.svg" width="18" height="18" align="middle" alt="domino" />) and an inward arrow. Clicking the halo opens the panel on that side.
  - When the panel is already open, approaching the opposite canvas edge reveals the halo on that side; clicking it relocates the panel across to the opposite edge of the canvas.

#### 3. Interface Options
- **Sticky opacity (enabled by default)**: When enabled, newly created shapes inherit the opacity value last used by any shape. When disabled, every new shape starts at 100% opacity.
- **Auto-expand selected (disabled by default)**: When enabled, selecting a member in the tree automatically expands it while collapsing the previously active member. When disabled, rows only expand or collapse when clicking their disclosure chevrons.
- **Member properties in subpanel (enabled by default)**: When enabled, parameters for the selected member (e.g., shape radius, feathering, parametric sliders) are displayed in a dedicated card below the tree. When disabled, parameters expand directly inside each row in the tree.
- **Show the mask panel's resize handle (enabled by default)**: When using the beside-canvas panel, displays a visible grip handle with an arrow indicating the collapse direction. When disabled, the handle is hidden but the edge can still be dragged to resize.

#### 4. Parametric Element Options
- **Reuse the last picked area (enabled by default)**: When sampling values on the canvas with the color picker, the sampled bounding box is remembered. Clicking the color picker on subsequent parametric channels immediately applies that same sample area without having to redraw a selection on the image.
- **Preview channel under cursor (disabled by default)**: When enabled, hovering the pointer over any add-parametric-channel button displays a live center-view preview of that channel after a brief pause, allowing you to preview the channel's mask before adding it.

#### 5. Default Group Layout and Notes
- **Default group layout**: Selects which template preset is automatically loaded when turning on a mask for the first time in a module (e.g., `drawn`, `parametric`, or custom saved presets).
- **Show preset notes (enabled by default)**: When enabled, groups generated from built-in presets display instructional cards in the selection panel explaining their purpose and suggested usage.

---

## 3. Blending Settings

When the mask panel is expanded, the top section provides module-level blending controls:

- **Blend Mode dropdown**: Blends the module's processed image output with the input image (Normal, Multiply, Screen, Difference, etc.). This controls image layer blending only; it does not affect how mask elements combine.
- **Blend Order**: Toggles whether blending takes place before or after module processing.
- **Blend Parameter slider**: Contextual slider for blend modes that accept an adjustable parameter.
- **Blend Opacity slider (labeled "blend opacity")**: Controls how strongly the module's blended effect applies to the image. Double-clicking resets it to 100%.

### Blend Opacity vs. "Whole Mask" Group Opacity
- **Blend opacity** scales the final mix of the module's processed image over the input image. If no mask is active, the module effect applies uniformly at this percentage. Blend opacity attenuates the final image blend without altering the mask itself.
- **"Whole mask" group opacity** (inside the mask tree) scales the values of the mask itself (from 0 to 1). Changing the "whole mask" opacity directly modifies the resulting mask data, which affects the yellow mask overlay preview and scales the raster mask exported to downstream modules.

---

## 4. The Member Tree

The tree view represents the hierarchy of groups and elements that compose the mask.

### Tree Rail and Nesting
Groups can nest inside other groups up to 8 levels deep. A continuous vertical tree rail runs down the left edge of each group, ticking across to each child member's header, visually communicating the group's boundaries and hierarchy.

### Group Combination Operators
A group combines its visible members in list order from top to bottom (a left fold) using one chosen operator:
- <img src="icons/masks_maximum.svg" width="18" height="18" align="middle" alt="maximum" /> **Maximum (Union)**: Takes the highest opacity value between overlapping members: `max(A, B)`.
- <img src="icons/masks_screen.svg" width="18" height="18" align="middle" alt="screen" /> **Screen**: Inverted product; soft additive combination without harsh clipping: `A + B - AB`.
- <img src="icons/masks_minimum.svg" width="18" height="18" align="middle" alt="minimum" /> **Minimum (Intersection)**: Takes the lowest opacity value: `min(A, B)`.
- <img src="icons/masks_product.svg" width="18" height="18" align="middle" alt="product" /> **Product**: Multiplies opacity values: `A * B`.
- <img src="icons/masks_sum.svg" width="18" height="18" align="middle" alt="sum" /> **Sum**: Clamped addition: `min(1, A + B)`.
- <img src="icons/masks_difference.svg" width="18" height="18" align="middle" alt="difference" /> **Difference**: Takes the first member and subtracts all subsequent members: `A - B - C...`.
- <img src="icons/masks_exclusion.svg" width="18" height="18" align="middle" alt="exclusion" /> **Exclusion**: Combines areas without overlap (exclusive OR).

### Group Headers
Each group displays a header bar with the following controls:
- **Operator Button (left icon and caret)**:
  - Displays the current combination operator icon (e.g., <img src="icons/masks_maximum.svg" width="18" height="18" align="middle" alt="maximum" />).
  - **Left-click**: Opens the operator menu to switch how members in this group combine.
- **Group Title**:
  - Displays the group name (defaults to `<operator>-<ordinal>`, e.g., `max-1`, or `whole mask` for root).
  - **Left-click**: Selects the group (click again to deselect).
  - **<kbd>Ctrl</kbd>+Left-click**: Opens an inline text field to rename the group.
  - **Drag**: Drag by the title or header to reorder the group or nest it inside another group.
- <img src="icons/invert.svg" width="18" height="18" align="middle" alt="invert" /> **Invert Button**:
  - **Left-click**: Inverts the group's composite output. When inverted, the button and operator icon display an inverted highlight style (<img src="icons/masks_maximum_inv.svg" width="18" height="18" align="middle" alt="inverted operator" />).
  - **<kbd>Shift</kbd>+Left-click**: Inverts each direct member inside the group in one action.
- <img src="icons/eye_toggle.svg" width="18" height="18" align="middle" alt="visibility" /> **Visibility (eye icon)**:
  - **Left-click**: Disables (bypasses) the group.
  - **<kbd>Shift</kbd>+Left-click**: Solos the group (evaluates only this group, temporarily hiding all other members in the mask).
- **Expander Chevron**:
  - **Left-click**: Collapses or expands the group's members in the list.

### Member Rows
Every element inside a group is displayed as a row:
- **Type Icon / Drag Handle (left column)**:
  - Displays the element type icon: <img src="icons/masks_path.svg" width="18" height="18" align="middle" alt="path" /> Path, <img src="icons/masks_brush.svg" width="18" height="18" align="middle" alt="brush" /> Brush, <img src="icons/masks_circle.svg" width="18" height="18" align="middle" alt="circle" /> Circle, <img src="icons/masks_ellipse.svg" width="18" height="18" align="middle" alt="ellipse" /> Ellipse, <img src="icons/masks_gradient.svg" width="18" height="18" align="middle" alt="gradient" /> Gradient, <img src="icons/masks_parametric.svg" width="18" height="18" align="middle" alt="parametric" /> Parametric, <img src="icons/masks_raster.svg" width="18" height="18" align="middle" alt="raster" /> Raster, or <img src="icons/masks_object.svg" width="18" height="18" align="middle" alt="object" /> AI Object.
  - **Drag**: Drag to reorder within the group or drop into another group. An insertion line indicates the target position.
- **Member Name**:
  - Displays the member name (e.g., `circle #1`, `L in`, or a custom label).
  - **Left-click**: Selects the member (click again to deselect).
  - **<kbd>Ctrl</kbd>+Left-click**: Opens an inline text field to rename the member.
  - **<kbd>Shift</kbd>+Left-click**: Toggles the row's inline controls (when inline properties are enabled).
- **Indicators and Links**:
  - <img src="icons/link.svg" width="18" height="18" align="middle" alt="link" /> **Linked Shape / Raster Link**: On linked shapes or raster masks, clicking jumps to the other module/instance.
  - **Low-Opacity Warning Badge**: Appears if an element's opacity is below 5%, warning that its contribution is practically invisible.
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
The top row of the selection area names the currently active item and hosts four core actions:
- <img src="icons/trash.svg" width="18" height="18" align="middle" alt="delete" /> **Delete Button (trash icon)**:
  - **Left-click**: Deletes the selected member (or resets the mask if "whole mask" is selected). Keyboard shortcut: <kbd>Delete</kbd> or <kbd>Backspace</kbd>.
  - **<kbd>Shift</kbd>+Left-click**: When a group is selected, **empties the group** (removes all its members while keeping the group container in place).
- <img src="icons/wand.svg" width="18" height="18" align="middle" alt="compose" /> **Compose Button (wand icon)**:
  - **Left-click**: Opens the operator chooser menu to wrap the selected member into a new parent group using the chosen operator.
  - **<kbd>Shift</kbd>+Left-click**: When a group is selected, **simplifies the group** (removes redundant nesting levels where collapsing them produces the exact same mask output).
- **Selection Name Box (center)**:
  - Shows the name of the selected member (or `whole mask`).
  - **Left-click**: Selects or deselects.
  - **<kbd>Ctrl</kbd>+Left-click**: Renames the selected member.
  - **Canvas Hover Peek**: When hovering the mouse over a shape on the image canvas, the caption displays that shape's name with `.dt_masks_peek` highlighting.
- <img src="icons/link.svg" width="18" height="18" align="middle" alt="link" /> **Link Button (chain icon)**:
  - Appears only when the selected element is a shared shape or raster mask.
  - **Left-click**: Jumps to the linked instance or source module.
  - **<kbd>Shift</kbd>+Left-click**: **Unlinks the shape**, giving this module its own independent local copy so subsequent edits do not affect other modules.
- <img src="icons/presets.svg" width="18" height="18" align="middle" alt="presets" /> **Group Layout Presets Button (presets icon)**:
  - **Left-click**: Opens the interactive **Group Layout Presets Popover** targeting the selected group.

### The Action Toolbar
The creation toolbar sits directly under the caption:
- **Left side: Group & Import**:
  - <img src="icons/plus.svg" width="18" height="18" align="middle" alt="add group" /> **Add Group (`+`)**: Opens the operator chooser menu to create a new group.
  - <img src="icons/import.svg" width="18" height="18" align="middle" alt="import" /> **Import**: Imports existing shapes or masks from other modules. Hovering an import entry previews it on canvas.
- **Right side: Drawn Shapes & Parametric Channels**:
  - **Drawn Shapes**: <img src="icons/masks_path.svg" width="18" height="18" align="middle" alt="path" /> Path, <img src="icons/masks_brush.svg" width="18" height="18" align="middle" alt="brush" /> Brush, <img src="icons/masks_circle.svg" width="18" height="18" align="middle" alt="circle" /> Circle, <img src="icons/masks_ellipse.svg" width="18" height="18" align="middle" alt="ellipse" /> Ellipse, <img src="icons/masks_gradient.svg" width="18" height="18" align="middle" alt="gradient" /> Gradient, <img src="icons/masks_object.svg" width="18" height="18" align="middle" alt="object" /> AI Object.
    - **Left-click**: Starts drawing a single shape on canvas.
    - **<kbd>Ctrl</kbd>+Left-click**: Activates continuous shape creation mode.
    - **<kbd>Escape</kbd> or Right-click**: Cancels drawing the pending shape.
  - **Parametric Channels**:
    - Flat buttons representing each channel in the module's working color space (e.g., `g`, `R`, `G`, `B` for RGB; `L`, `a`, `b`, `C`, `h` for Lab).
    - **Hover**: Displays a live preview of that channel's mask on the canvas (only when "preview channel under cursor" is enabled in options).
    - **Left-click**: Adds that channel into the active group.
    - **<kbd>Shift</kbd>+Left-click**: Pins a persistent channel preview overlay on canvas.

---

## 6. Member Properties & Refinements Sub-Panels

Beneath the toolbar, the selection area hosts collapsible cards that adapt dynamically to what is selected:

### 1. Preset Guidance Notes (`#masks-help-section`)
When a group created from a built-in preset is selected (and "show preset notes" is enabled in options), an instructional card appears here explaining the intended purpose, typical workflow, and tips for working with that preset layout.

### 2. Member Properties (`#masks-props-section`)
Displays the parameters of the selected member:
- **Drawn Shapes**: Sliders for opacity, size, hardness, feather radius, curvature, rotation, and feathering guide.
- **AI Objects**: Sliders for opacity, boundary smoothing, and cleanup.
- **Parametric Channels**:
  - **Input / Output Toggle**: Clicking the channel abbreviation icon at the top of the card toggles between showing the input slider and output slider.
  - **Dual Gradient Sliders**: Each slider features 4 threshold markers (black point, feather start, feather end, white point).
  - **Right-click on any marker**: Opens a precise numeric input popover for that threshold.
  - **Dedicated Color Pickers (<img src="icons/colorpicker.svg" width="18" height="18" align="middle" alt="color picker" />)**: Located directly at the right edge of each slider. Left-clicking activates the canvas picker to set thresholds from sampled image regions (<kbd>Ctrl</kbd>+click samples GUI color values).
  - **Boost Factor**: Adjusts the dynamic range for scene-referred channels in EV stops.
- **Groups**: Group opacity slider.
- **Active Drawing**: While drawing a shape on canvas, displays creation controls before committing the shape.

#### Opacity Value Adjustments
- **Scroll wheel**: Adjusts opacity in 5% steps.
- **<kbd>Ctrl</kbd> + Scroll wheel**: Fine adjustment in 1% steps.
- **<kbd>Shift</kbd> + Scroll wheel**: Coarse adjustment.
- **Right-click**: Opens a Bauhaus whisker slider popup for quick adjustments.
- **Double-click**: Resets opacity to 100%.

### 3. Mask Refinements (`masks_refinement`)
Refinements provide edge-aware smoothing, feathering, and tonal adjustments. The refinement automatically applies to the current selection scope:
- **Element Scope**: Refines that specific element before it is combined by its group's operator.
- **Group Scope**: Refines the group's composite output after its operator combines its members.
- **Whole Mask Scope**: Refines the final combined mask of the module.

#### Refinement Controls:
- <img src="icons/eye_toggle.svg" width="18" height="18" align="middle" alt="bypass" /> **Bypass Toggle**: Temporarily disables refinements for the current scope without resetting sliders.
- <img src="icons/reset.svg" width="18" height="18" align="middle" alt="reset" /> **Reset Button**: Resets refinement sliders for the current scope to default values.
- **Details**: Uses an edge-preserving guided filter to enhance or soften fine mask edges against the image details.
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
- **Layout Outline Tooltip**: Hovering over the preset name displays a formatted tree outline showing all nested groups, their combination operators, and initial opacities.
- **Three Mode Action Buttons (right edge)**:
  1. <img src="icons/reset.svg" width="18" height="18" align="middle" alt="replace" /> **Replace**: Replaces the selected group's structure with the preset. If existing elements would be discarded, darktable prompts for confirmation first.
  2. <img src="icons/plus.svg" width="18" height="18" align="middle" alt="merge" /> **Merge**: Merges the preset structure into the selected group (available when the preset contains multiple groups).
  3. <img src="icons/plus.svg" width="18" height="18" align="middle" alt="add" /> **Add as Subgroup**: Inserts the preset structure as a nested child group.
- **Delete Button** (<img src="icons/trash.svg" width="18" height="18" align="middle" alt="delete" />): Available on user presets to delete custom layouts.

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
| **Header Domino** | <img src="icons/masks_panel.svg" width="18" height="18" /> | Left-click | Toggle mask on / off |
| **Header Domino** | <img src="icons/masks_panel.svg" width="18" height="18" /> | Right-click | Open mask options popover |
| **Header Edit on Canvas** | <img src="icons/masks_eye.svg" width="18" height="18" /> | Left-click | Toggle canvas shape controls on / off |
| **Header Edit on Canvas** | <img src="icons/masks_eye.svg" width="18" height="18" /> | <kbd>Ctrl</kbd> + Left-click | Toggle restricted canvas edit mode |
| **Header Solo** | <img src="icons/soloedit.svg" width="18" height="18" /> | Left-click | Toggle solo edit mode |
| **Header Overlay** | <img src="icons/showmask.svg" width="18" height="18" /> | Left-click | Toggle mask overlay on image |
| **Header Suppress** | <img src="icons/eye_toggle.svg" width="18" height="18" /> | Left-click | Temporarily switch off mask in preview |
| **Header Lock** | <img src="icons/mask_lock.svg" width="18" height="18" /> | Left-click | Toggle lock mask against edits/presets |
| **Canvas Edge Halo** | <img src="icons/masks_panel.svg" width="18" height="18" /> | Left-click | Summon / move beside-canvas panel to that side |
| **Member Row or Header** | | Left-click | Select member (click again to deselect) |
| **Member Row or Header** | | <kbd>Ctrl</kbd> + Left-click | Inline rename |
| **Member Row or Header** | | <kbd>Shift</kbd> + Left-click | Toggle disclosure (controls or child members) |
| **Member Row or Header** | | <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + Hover | Arm deletion (shows trash pointer & turns row red) |
| **Member Row or Header** | | <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + Click | Delete member (or reset mask on "whole mask") |
| **Member Drag Handle** | | Drag & Drop | Rearrange or nest in tree |
| **Element Invert Button** | <img src="icons/invert.svg" width="18" height="18" /> | Left-click | Invert element |
| **Group Invert Button** | <img src="icons/invert.svg" width="18" height="18" /> | Left-click | Invert group output |
| **Group Invert Button** | <img src="icons/invert.svg" width="18" height="18" /> | <kbd>Shift</kbd> + Left-click | Invert each direct member in group |
| **Visibility Eye** | <img src="icons/eye_toggle.svg" width="18" height="18" /> | Left-click | Bypass / enable member |
| **Visibility Eye** | <img src="icons/eye_toggle.svg" width="18" height="18" /> | <kbd>Shift</kbd> + Left-click | Solo member (hides all other members) |
| **Operator Button** | <img src="icons/masks_maximum.svg" width="18" height="18" /> | Left-click | Open combination operator menu |
| **Operator Button** | <img src="icons/masks_maximum.svg" width="18" height="18" /> | <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + Click | Delete group and all its members |
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
| **Channel Add Button** | <img src="icons/masks_parametric.svg" width="18" height="18" /> | Hover | Preview channel on image (if option enabled) |
| **Channel Add Button** | <img src="icons/masks_parametric.svg" width="18" height="18" /> | Left-click | Add channel into active group |
| **Channel Add Button** | <img src="icons/masks_parametric.svg" width="18" height="18" /> | <kbd>Shift</kbd> + Left-click | Pin channel preview overlay on canvas |
| **Import Menu Entry** | <img src="icons/import.svg" width="18" height="18" /> | Hover | Preview external shape/mask on canvas |
| **Channel Icon (Props)** | | Left-click | Toggle input slider vs output slider |
| **Parametric Marker** | | Right-click | Open precise numeric entry popover |
| **Parametric Picker** | <img src="icons/colorpicker.svg" width="18" height="18" /> | Left-click | Sample canvas region for threshold |
| **Opacity Value** | | Scroll wheel | Adjust opacity (<kbd>Ctrl</kbd>: 1%, <kbd>Shift</kbd>: coarse) |
| **Opacity Value** | | Right-click | Open Bauhaus whisker popup slider |
| **Opacity Value** | | Double-click | Reset opacity to 100% |
