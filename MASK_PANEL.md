# The Mask Panel

darktable's new mask panel (a.k.a., Flexi) unifies all masking tools into a single hierarchical tree. It subsumes both the previous in-module mask controls and the standalone **mask manager**, which has been removed. All mask creation, shape grouping, combination logic, shape reuse across modules, and unused shape cleanup are now handled directly within this unified panel.

Instead of choosing between rigid drawn, parametric, or raster mask modes, a module's mask is built by combining any number of elements inside groups using combination operators. Every element—drawn shapes, parametric channels, raster masks from other modules, and AI objects—produces an opacity value between 0 and 1 at each pixel and can be freely combined, reordered, inverted, and refined.

---

## 1. Panel Layout and Header

### Where the Panel Lives
The mask panel can be hosted in three locations (configured via the Mask Options popover):
- **Embedded within each module (default)**: Located inside each processing module under the "blend mask" section. Clicking the header bar folds or unfolds the panel.
- **Utility module, left panel**: Hosted in the darkroom's left sidebar as a dedicated module labeled **masking** (not "blend mask"), occupying the panel position where the old mask manager once lived. It automatically switches to show and edit the mask of whichever processing module currently has focus.
- **Separate panel, beside the canvas**: Detached as a dockable, resizable panel beside the image canvas.

### Header Controls
The header bar of the mask panel contains the primary toggles and status indicators:

- <img src="icons/masks_panel.svg" width="18" height="18" align="middle" alt="domino" /> **On/Off Toggle (domino icon)**
  - **Left-click**: Switches the module's mask on (flexi mask) or off (disabled).
  - **Right-click**: Opens the **Mask Options** popover.
- <img src="icons/masks_eye.svg" width="18" height="18" align="middle" alt="edit on canvas" /> **Edit on Canvas (pointer on shape icon)**
  - **Left-click**: Toggles on-canvas shape outlines, control points, and manipulation handles.
  - **Ctrl+click**: Toggles *restricted edit mode*, allowing control points to be adjusted while preventing shapes from being moved or resized by mistake.
- <img src="icons/soloedit.svg" width="18" height="18" align="middle" alt="solo edit" /> **Solo Edit (target icon)**
  - **Left-click**: Isolates the selected shape or group on the canvas. Only the selected shape's handles remain editable on canvas, while all other shapes continue to be computed in the mask. This prevents accidental selection of neighboring shapes.
- <img src="icons/showmask.svg" width="18" height="18" align="middle" alt="show mask" /> **Show Mask Overlay (mask preview icon)**
  - **Left-click**: Toggles the mask visualization overlay on the image (e.g. yellow overlay), showing exactly where the mask is active.
- <img src="icons/eye_toggle.svg" width="18" height="18" align="middle" alt="suppress mask" /> **Temporarily Switch Off Mask (eye toggle icon)**
  - **Left-click**: Temporarily disables the mask in the center view to compare the image with and without masking. The mask switches back on when you navigate away from the module.
- <img src="icons/mask_lock.svg" width="18" height="18" align="middle" alt="lock mask" /> **Lock Mask (padlock icon)**
  - **Left-click**: Locks the mask against changes. A locked mask will not be overwritten when applying presets or styles, pasting history, or resetting the module.

---

## 2. Mask Options Popover

The Mask Options popover configures interface settings, default layouts, and color spaces specific to the mask panel.

### Accessing the Popover
The options popover can be opened in two ways:
- **Right-click the On/Off domino button** (<img src="icons/masks_panel.svg" width="18" height="18" align="middle" alt="domino" />) in any module's mask panel header.
- **Right-click the mask panel toggle button** in the darkroom bottom toolbar.

### Popover Settings

#### 1. Blend Color Space
Selects the color space used to evaluate the module's mask and its parametric channels.

The available color spaces depend on the currently focused module and where it is located in the pixelpipe. Not all color spaces are always available:
- **default**: Uses the module's native working color space.
- **RGB (scene-referred)**: Evaluates the mask in unbounded linear RGB. Preserves high dynamic range values in scene-referred pipeline modules.
- **RGB (display-referred)**: Evaluates the mask in display-referred RGB.
- **Lab**: Perceptual Lab color space. **Only offered on modules that natively process in Lab**; it is withheld on RGB modules to avoid invalid color space conversions before the input color profile or after the output color profile.

*Notes on color space selection*:
- Modules that do not support parametric masks (or modules without blend support) do not display the blend color space section in the popover.
- If switching color spaces when parametric channels already exist that cannot be converted to the new space, darktable prompts for confirmation before removing the incompatible channels.
- When the mask is locked, the color space options are disabled.

#### 2. Blend Mask Panel Position
Controls where the mask panel is hosted in the interface:
- **Embedded within each module (default)**: The mask panel lives directly inside each processing module's expandable body.
- **Utility module, left panel**: Moves the entire panel into a standalone module labeled **masking** in the left sidebar. Focusing any processing module directs this utility module to display and edit that module's mask.
- **Separate panel, beside the canvas**: Detaches the mask panel into a dockable, resizable panel positioned directly beside the image canvas.
  - It can be summoned to either side of the canvas: approaching either the left or right canvas edge with the pointer reveals a halo overlay spanning the whole height of the canvas with the domino icon (<img src="icons/masks_panel.svg" width="18" height="18" align="middle" alt="domino" />) and an inward arrow. Clicking the halo opens the panel on that side.
  - When the panel is already open, approaching the opposite canvas edge reveals the halo on that side; clicking it relocates the panel across to the opposite edge of the canvas.

#### 3. Interface Options
- **Sticky opacity (enabled by default)**: When enabled, newly created shapes inherit the opacity value last used by any shape. When disabled, every new shape always starts at 100% opacity.
- **Auto-expand selected (disabled by default)**: When enabled, selecting an element or group in the tree automatically expands it while collapsing the previously active item. When disabled, rows only expand or collapse when clicking their disclosure chevrons.
- **Element properties in subpanel (enabled by default)**: When enabled, parameters for the selected element (e.g., shape radius, feathering, boost factor) are displayed in a clean, dedicated card below the tree. When disabled, parameters expand directly inside each element row in the list.
- **Show the mask panel's resize handle (enabled by default)**: When using the floating or beside-canvas panel, displays a visible grip handle with an arrow indicating the collapse direction. When disabled, the handle is hidden but the edge can still be clicked or dragged to resize.

#### 4. Parametric Element Options
- **Reuse the last picked area (enabled by default)**: When sampling values on the canvas with the color picker, the sampled bounding box is remembered. Clicking the color picker on subsequent parametric channels immediately applies that same sample area without having to redraw a selection on the image.
- **Preview channel under cursor (disabled by default)**: When enabled, hovering the pointer over any add-parametric-channel button displays a live center-view preview of that channel after a brief pause, allowing you to preview the channel's mask before adding it.

#### 5. Default Group Layout and Notes
- **Default group layout**: Selects which template preset is automatically loaded when turning on a mask for the first time in a module (e.g., `drawn`, `parametric`, or custom saved presets).
- **Show preset notes (enabled by default)**: When enabled, groups generated from built-in presets display instructional cards under their headers explaining their purpose and how to use them.

---

## 3. Blending Settings

When the mask panel is expanded, the top section provides module-level blending controls:

- **Blend Mode dropdown**: Blends the module's processed image output with the input image (Normal, Multiply, Screen, Difference, etc.). This controls image layer blending only; it does not affect how mask elements combine (for example, "Difference" here blends image layers and does not create a subtractive mask).
- **Blend Order**: Toggles whether blending takes place before or after module processing.
- **Blend Parameter slider**: Contextual slider for blend modes that accept an adjustable parameter.
- **Blend Opacity slider (labeled "blend opacity")**: Controls how strongly the module's blended effect applies to the image. Double-clicking resets it to 100%.

### Blend Opacity vs. "Whole Mask" Group Opacity
It is important to distinguish between these two top-level opacity controls:
- **Blend opacity** scales the final mix of the module's processed image over the input image. If no mask is active, the module effect simply applies uniformly at this percentage. Blend opacity attenuates the final image blend without altering the mask itself.
- **"Whole mask" group opacity** (inside the mask tree) scales the values of the mask itself (from 0 to 1). Changing the "whole mask" opacity directly modifies the resulting mask data, which affects the yellow mask overlay preview and scales the raster mask exported to downstream modules.

---

## 4. Adding Elements: The Toolbar

The toolbar sits between the blending controls and the mask layout list. It provides quick access to create groups, add shapes, add parametric channels, or import elements.

### Add Group (<img src="icons/plus.svg" width="18" height="18" align="middle" alt="add group" /> `+` button)
- **Left-click**: Opens the operator chooser menu to create a new group using a selected combination operator (Maximum, Product, Screen, Sum, Difference, Exclusion, Minimum).

### Drawn Shapes
Buttons for drawn shapes:
- <img src="icons/masks_path.svg" width="18" height="18" align="middle" alt="path" /> **Path**: Add bezier path.
- <img src="icons/masks_brush.svg" width="18" height="18" align="middle" alt="brush" /> **Brush**: Add freehand brush stroke.
- <img src="icons/masks_circle.svg" width="18" height="18" align="middle" alt="circle" /> **Circle**: Add circular mask.
- <img src="icons/masks_ellipse.svg" width="18" height="18" align="middle" alt="ellipse" /> **Ellipse**: Add elliptical mask.
- <img src="icons/masks_gradient.svg" width="18" height="18" align="middle" alt="gradient" /> **Gradient**: Add linear gradient mask.
- <img src="icons/masks_object.svg" width="18" height="18" align="middle" alt="AI object" /> **AI Object** (when compiled with AI support): Segment objects using neural network.

**Interactions for shapes**:
- **Left-click**: Arms the tool to draw a single shape on the canvas.
- **Ctrl+click**: Continuous creation mode-keeps the tool active to add multiple shapes in succession until canceled.
- **Escape or Right-click on canvas**: Cancels drawing the current shape.

### Parametric Channels (<img src="icons/masks_parametric.svg" width="18" height="18" align="middle" alt="parametric" />)
Displays flat buttons representing each channel available in the active blend color space (e.g., `L`, `a`, `b` in Lab; `Jz`, `Cz`, `hz` or `R`, `G`, `B` in RGB):
- **Hover**: Previews the channel mask on the image (only when "preview channel under cursor" is enabled in the Mask Options popover).
- **Left-click**: Adds that channel as a parametric mask element into the active group.
- **Shift+click**: Pins the channel overlay preview on the canvas so it remains visible while working.

### Import Menu (<img src="icons/import.svg" width="18" height="18" align="middle" alt="import" /> import icon)
Opens a popover menu allowing you to reuse masks from other modules:
- **Link shapes**: Reuses a shape drawn in another module. Changes made to the shape will update across all modules sharing it.
- **Copy shapes**: Duplicates a shape from another module as an independent copy.
- **Copy parametric channels**: Imports parametric channel settings from another module.
- **Add raster mask**: Uses another module's mask output as a raster mask element in this mask.
- **Clean up unused shapes / compress history**: Removes orphaned shapes that are not referenced by any module.


**Live Canvas Preview**: Hovering over any shape or raster mask in the import menu previews it directly on the canvas before importing.

---

## 5. The Mask Layout (Groups and Elements Tree)

The central area displays the hierarchical mask structure.

### Tree Concept and Fold Order
- The root of the hierarchy is **"whole mask"**. Every element and group lives inside it.
- Each group combines its members in order **from the bottom up** using a single combination operator.
- Groups can be nested inside other groups up to 8 levels deep.

### Available Operators
- <img src="icons/masks_maximum.svg" width="18" height="18" align="middle" alt="maximum" /> **Maximum**: Takes the highest opacity value at each pixel (additive union, similar to logical OR).
- <img src="icons/masks_product.svg" width="18" height="18" align="middle" alt="product" /> **Product**: Multiplies opacities (pixel-by-pixel intersection, similar to logical AND).
- <img src="icons/masks_screen.svg" width="18" height="18" align="middle" alt="screen" /> **Screen**: Inverted product; soft additive combination without harsh clipping.
- <img src="icons/masks_sum.svg" width="18" height="18" align="middle" alt="sum" /> **Sum**: Clamped addition: `min(1, A + B)`.
- <img src="icons/masks_difference.svg" width="18" height="18" align="middle" alt="difference" /> **Difference**: Takes the first member and subtracts all subsequent members: `A - B - C...`.
- <img src="icons/masks_exclusion.svg" width="18" height="18" align="middle" alt="exclusion" /> **Exclusion**: Combines areas without overlap (similar to XOR).
- <img src="icons/masks_minimum.svg" width="18" height="18" align="middle" alt="minimum" /> **Minimum**: Takes the lowest opacity value (hard intersection).

### Group Headers
Each group displays a header bar with the following controls:

- **Operator Handle (left column icon and caret)**
  - Shows the icon of the operator combining this group's members (e.g. <img src="icons/masks_maximum.svg" width="18" height="18" align="middle" alt="maximum" />).
  - **Left-click**: Opens the operator chooser to switch the combination operator.
  - **Alt+click**: Inverts the group's composite output. The icon swaps to an inverted highlight style (<img src="icons/masks_maximum_inv.svg" width="18" height="18" align="middle" alt="inverted operator" />).
- **Group Title**
  - Shows the group name (defaults to `<operator>-<ordinal>`, e.g., `max-1`, or `whole mask` for root).
  - **Left-click**: Selects the group (refinements and properties then target this group). Click again to return selection to the whole mask.
  - **Ctrl+click**: Opens an inline text field to rename the group.
  - **Drag**: Drag by the title to move the group, nest it inside another group, or reposition it in the tree.
  - **Alt+click**: Inverts the group output.
- <img src="icons/info.svg" width="18" height="18" align="middle" alt="info" /> **Info Note Icon (`i`)**
  - Appears on groups created from built-in presets. Left-clicking toggles an instructional card explaining how to use the group.
- <img src="icons/eye_toggle.svg" width="18" height="18" align="middle" alt="visibility" /> **Visibility (eye icon)**
  - **Left-click**: Disables (bypasses) the group. The group remains in the tree, but contributes nothing to the mask.
  - **Shift+click**: Solos the group. Only this group is evaluated; all other elements in the mask are temporarily disabled.
- **Expander Chevron**
  - Collapses or expands the group's members.
- **Right-click on Group Header**: Opens the **Group Actions Context Menu**:
  - **Visibility Section**:
    - **disable** (checkbox): Switches the group off while keeping all settings. Clicking the eye icon on the header does the same.
    - **solo** (checkbox): Evaluates only this group to inspect its isolated contribution. Shift+clicking the eye icon on the header does the same.
  - **Mask Operations Section**:
    - **invert output** (checkbox): Inverts the group's folded mask output. Alt+clicking the operator icon does the same.
    - **invert all elements**: Inverts every direct member element inside the group in one action.
    - **compose**: Submenu offering combination operators (Maximum, Screen, Minimum, Product, Sum, Difference, Exclusion) to wrap this group inside a new parent group.
    - **simplify**: Makes the tree under this group shallower wherever collapsing redundant nesting levels renders the exact same mask.
  - **Edit Section**:
    - **rename**: Opens inline rename field for the group (disabled on root "whole mask"). Ctrl+clicking the title does the same.
    - **empty**: Removes all members from the group while preserving the empty group container.
    - **delete**: Deletes the group and all its members from the mask (disabled on root "whole mask").

### Element Rows
Every element inside a group (shape, parametric channel, raster mask, or AI object) is shown as a row:

- **Kind / Drag Handle (left column)**
  - Displays the element type icon (e.g. <img src="icons/masks_circle.svg" width="18" height="18" align="middle" alt="circle" /> Circle, <img src="icons/masks_path.svg" width="18" height="18" align="middle" alt="path" /> Path, <img src="icons/masks_brush.svg" width="18" height="18" align="middle" alt="brush" /> Brush, <img src="icons/masks_parametric.svg" width="18" height="18" align="middle" alt="parametric" /> Parametric, <img src="icons/masks_raster.svg" width="18" height="18" align="middle" alt="raster" /> Raster, or <img src="icons/masks_multi.svg" width="18" height="18" align="middle" alt="nested group" /> Nested Group).
  - When the element is inverted, the icon plate is filled with a contrasting highlight (<img src="icons/masks_circle_inv.svg" width="18" height="18" align="middle" alt="inverted circle" />).
  - **Drag**: Drag by the handle to reorder the element within its group or drop it into another group.
  - **Alt+click**: Inverts the element.
- **Element Name**
  - Displays the element name (e.g., `path #1`, `L in`, or a custom name).
  - **Left-click**: Selects the element. Click again to deselect.
  - **Ctrl+click**: Opens an inline text field to rename the element.
  - **Shift+click**: Toggles the element's expanded controls (when inline properties are enabled).
  - **Drag**: The element can also be dragged by its name label.
- **Indicators and Links (middle column)**
  - <img src="icons/link.svg" width="18" height="18" align="middle" alt="link" /> **Linked shape indicator (chain icon)**: Indicates the shape is shared across modules or used multiple times in this mask. Clicking displays usage details.
  - <img src="icons/link.svg" width="18" height="18" align="middle" alt="raster link" /> **Raster source link (chain icon)**: On raster mask rows, left-clicking jumps directly to the source module and opens its mask on canvas.
  - <img src="icons/colorpicker.svg" width="18" height="18" align="middle" alt="color picker" /> **Color picker button**: On parametric channel rows, left-clicking activates the color picker on canvas to set slider thresholds from sampled image regions (Ctrl+click samples GUI color values).
- **Low-Opacity Warning Badge**
  - Appears if an element's opacity is set below 5%, warning that its contribution is nearly invisible.
- <img src="icons/eye_toggle.svg" width="18" height="18" align="middle" alt="visibility" /> **Visibility (eye icon)**
  - **Left-click**: Toggles the element disabled/enabled (bypass).
  - **Shift+click**: Solos the element (evaluates only this single element).
- **Right-click on Element Row**: Opens the **Element Actions Context Menu**:
  - **Visibility Section**:
    - **disable** (checkbox): Switches this element off. Clicking the eye icon on the row does the same.
    - **solo** (checkbox): Evaluates only this element. Shift+clicking the eye icon on the row does the same.
  - **Mask Operations Section**:
    - **invert** (checkbox): Inverts the element so it selects what it previously left out. Alt+clicking its icon does the same.
    - **compose**: Submenu offering combination operators (Maximum, Screen, Minimum, Product, Sum, Difference, Exclusion) to wrap this element into a new group.
  - **Edit Section**:
    - **rename**: Opens inline rename field for the element. Ctrl+clicking the row does the same.
    - **unlink**: (Enabled on linked shapes) Creates an independent local copy of the shape so modifications no longer alter other rows or modules.
    - **delete**: Removes the element from the mask.

### Element Clusters
When 3 or more adjacent elements of the same kind exist in a group, they automatically fold into an expandable cluster row (e.g. `3 circles`) to reduce vertical clutter:
- Shows the kind icon, member count, kind name, and an expand/collapse arrow.
- **Left-click anywhere on row**: Expands or collapses the cluster.
- **Drag**: Grabbing anywhere in the row moves all clustered elements together as one contiguous block.
- **Right-click on cluster row**: Opens the cluster context menu:
  - **delete all**: Removes every element folded into the cluster in one step.

---

## 6. Selection Area

Beneath the mask tree sits the Selection Area (`#masks-selection-area`). It houses the selection header row, the element properties card, and the mask refinements card:

### The Selection Header Row
Displays the target currently being acted upon:
- **Selection Icon**: Displays the kind icon of the selected element or group.
- **Selection Name Label**: Names the active item (e.g., `path #1`, `group max-1`, or `whole mask`).
- <img src="icons/remove.svg" width="18" height="18" align="middle" alt="delete" /> **Delete Button**:
  - Located on the right edge of the selection row.
  - **Left-click**: Deletes the selected element or group (if "whole mask" is selected, resets the mask).
  - Keyboard shortcut: `Delete` or `Backspace`.

---

## 7. Element Properties Sub-Panel

When "element properties in sub-panel" is enabled (default), the properties of the currently selected element appear in a collapsible card directly inside the selection area:

- **Drawn Shapes**: Sliders for opacity, size, hardness, feather radius, curvature, rotation, and feathering guide.
- **AI Objects**: Sliders for opacity, boundary smoothing, and cleanup.
- **Parametric Channels**:
  - Input and output gradient sliders with 4 adjustable range markers (black point, feather start, feather end, white point).
  - **Right-click on any marker**: Opens a precise numeric input popover for that exact threshold.
  - **Boost Factor slider**: Expands the range for high dynamic range, scene-referred channels in EV stops.
- **Groups**: Group opacity slider.
- **Shape Drawing**: While actively drawing a shape on canvas, this card displays creation controls (e.g., initial radius and feathering) before the shape is committed.

### Quick Adjustments for Inline Opacity Values
Where opacity sliders or inline numbers appear:
- **Scroll wheel**: Adjusts opacity in 5% increments.
- **Ctrl + Scroll wheel**: Fine adjustments in 1% increments.
- **Shift + Scroll wheel**: Coarse adjustments.
- **Right-click**: Opens a Bauhaus whisker slider popup for quick adjustments without opening the full properties panel.
- **Double-click**: Resets opacity to 100%.

---

## 8. Refinements

The refinements section applies edge-aware smoothing, blurring, and tonal adjustments. It automatically adapts its target based on the current selection:

### Refinement Scope
- **Element Selected**: Refines that specific element before it is combined by its group's operator.
- **Group Selected**: Refines the group's composite output after its operator combines its members.
- **Nothing / Whole Mask Selected**: Refines the final combined mask of the module.

### Refinement Controls
- <img src="icons/eye_toggle.svg" width="18" height="18" align="middle" alt="bypass" /> **Bypass Toggle (eye icon on header)**: Temporarily disables the refinement without losing slider settings.
- <img src="icons/reset.svg" width="18" height="18" align="middle" alt="reset" /> **Reset Button (reset icon on header)**: Resets refinement sliders for the current scope to default values.
- **Details**: Uses an edge-preserving guided filter to enhance or soften fine mask edges against the image details.
- **Feathering Guide**: Selects whether edge feathering is guided by the input or output image, and whether feathering applies before or after mask blur.
- **Feathering Radius**: Spatial radius for guided feathering (in pixels).
- **Blurring Radius**: Standard Gaussian blur applied to the mask (in pixels).
- **Mask Brightness**: Shifts and tilts the tone curve of the mask to brighten or darken semi-transparent areas without altering fully transparent or opaque regions.
- **Mask Contrast**: Applies an S-curve to mask opacities to steepen or flatten transitions.

---

## 9. Mask Consumers

The **Mask Consumers** section appears at the bottom of the panel only when one or more modules further down the pixelpipe are using this module's mask as a raster mask.

- Lists every downstream module reading this mask.
- **Left-click on any entry**: Immediately focuses that consumer module and opens its mask panel.

---

## 10. Quick Reference: Mouse and Keyboard Shortcuts

| Target | Icon | Action | Result |
|---|:---:|---|---|
| **Header Domino** | <img src="icons/masks_panel.svg" width="18" height="18" /> | Left-click | Toggle mask on / off |
| **Header Domino** | <img src="icons/masks_panel.svg" width="18" height="18" /> | Right-click | Open mask options popover |
| **Header Edit on Canvas** | <img src="icons/masks_eye.svg" width="18" height="18" /> | Left-click | Toggle canvas shape controls on / off |
| **Header Edit on Canvas** | <img src="icons/masks_eye.svg" width="18" height="18" /> | Ctrl + Left-click | Toggle restricted canvas edit mode |
| **Header Solo** | <img src="icons/soloedit.svg" width="18" height="18" /> | Left-click | Toggle solo edit mode |
| **Header Overlay** | <img src="icons/showmask.svg" width="18" height="18" /> | Left-click | Toggle mask overlay on image |
| **Header Suppress** | <img src="icons/eye_toggle.svg" width="18" height="18" /> | Left-click | Temporarily switch off mask in preview |
| **Header Lock** | <img src="icons/mask_lock.svg" width="18" height="18" /> | Left-click | Toggle lock mask |
| **Canvas Edge Halo** | <img src="icons/masks_panel.svg" width="18" height="18" /> | Left-click | Summon / move beside-canvas panel to that side |
| **Shape Add Button** | <img src="icons/masks_circle.svg" width="18" height="18" /> | Left-click | Draw a single shape |
| **Shape Add Button** | <img src="icons/masks_circle.svg" width="18" height="18" /> | Ctrl + Left-click | Continuous shape drawing mode |
| **Drawing Shape** | | Escape or Right-click | Cancel drawing the current shape |
| **Channel Add Button** | <img src="icons/masks_parametric.svg" width="18" height="18" /> | Hover | Preview channel mask on image (if option enabled) |
| **Channel Add Button** | <img src="icons/masks_parametric.svg" width="18" height="18" /> | Left-click | Add channel into active group |
| **Channel Add Button** | <img src="icons/masks_parametric.svg" width="18" height="18" /> | Shift + Left-click | Pin channel preview overlay on canvas |
| **Import Menu Item** | <img src="icons/import.svg" width="18" height="18" /> | Hover | Preview external shape/mask on canvas |
| **Group / Element Row** | | Left-click | Select item (click again to deselect) |
| **Group / Element Row** | | Ctrl + Left-click | Inline rename |
| **Group / Element Row** | | Alt + Left-click | Invert group output / element |
| **Group / Element Handle** | | Drag & Drop | Rearrange or nest in tree |
| **Visibility Eye** | <img src="icons/eye_toggle.svg" width="18" height="18" /> | Left-click | Bypass / enable item |
| **Visibility Eye** | <img src="icons/eye_toggle.svg" width="18" height="18" /> | Shift + Left-click | Solo item (hides all other elements) |
| **Operator Icon** | <img src="icons/masks_maximum.svg" width="18" height="18" /> | Left-click | Open operator chooser menu |
| **Operator Icon** | <img src="icons/masks_maximum_inv.svg" width="18" height="18" /> | Alt + Left-click | Invert group output |
| **Element Row** | | Right-click | Open Element Actions context menu |
| **Group Header** | | Right-click | Open Group Actions context menu |
| **Cluster Row** | | Left-click | Expand / collapse cluster |
| **Cluster Row** | | Drag & Drop | Move all clustered elements together |
| **Cluster Row** | | Right-click | Open Cluster Actions menu ("delete all") |
| **Parametric Marker** | | Right-click | Open precise numeric entry popover |
| **Selection Delete** | <img src="icons/remove.svg" width="18" height="18" /> | Left-click / `Delete` / `Backspace` | Delete selected element or group |
| **Opacity Value** | | Right-click | Open whisker popup slider |
| **Opacity Value** | | Double-click | Reset opacity to 100% |
| **Opacity Value** | | Scroll wheel | Adjust opacity (Ctrl: fine, Shift: coarse) |
