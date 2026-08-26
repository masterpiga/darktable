/*
    This file is part of darktable,
    Copyright (C) 2011-2025 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "common/iop_profile.h"
#include "common/opencl.h"
#include "develop/pixelpipe.h"
#include "develop/masks.h"
#include "dtgtk/button.h"
#include "dtgtk/gradientslider.h"
#include "gui/color_picker_proxy.h"
#include "common/imagebuf.h"
#include "common/gaussian.h"

#define DEVELOP_BLEND_VERSION (15)

// masks_panel_position conf values ("plugins/darkroom/blend/masks_panel_position")
// -- where the flexi masks panel content lives. Shared between blend_gui.c
// (relocation logic + hamburger menu) and libs/masks_flexi_host.c
// (container()/collapsible setup for the shared host lib).
#define MASKS_PANEL_POS_EMBEDDED 0
#define MASKS_PANEL_POS_UTILITY  1
#define MASKS_PANEL_POS_LEFT     2
#define MASKS_PANEL_POS_RIGHT    3

G_BEGIN_DECLS

typedef enum dt_develop_blend_colorspace_t
{
  DEVELOP_BLEND_CS_NONE = 0,
  DEVELOP_BLEND_CS_RAW = 1,
  DEVELOP_BLEND_CS_LAB = 2,
  DEVELOP_BLEND_CS_RGB_DISPLAY = 3,
  DEVELOP_BLEND_CS_RGB_SCENE = 4,
} dt_develop_blend_colorspace_t;

typedef enum dt_develop_blend_mode_t
{
  DEVELOP_BLEND_DISABLED_OBSOLETE = 0x00, /* same as the new normal */
  DEVELOP_BLEND_NORMAL_OBSOLETE = 0x01, /* obsolete as it did clamping */
  DEVELOP_BLEND_LIGHTEN = 0x02,
  DEVELOP_BLEND_DARKEN = 0x03,
  DEVELOP_BLEND_MULTIPLY = 0x04,
  DEVELOP_BLEND_AVERAGE = 0x05,
  DEVELOP_BLEND_ADD = 0x06,
  DEVELOP_BLEND_SUBTRACT = 0x07,
  DEVELOP_BLEND_DIFFERENCE = 0x08, /* deprecated */
  DEVELOP_BLEND_SCREEN = 0x09,
  DEVELOP_BLEND_OVERLAY = 0x0A,
  DEVELOP_BLEND_SOFTLIGHT = 0x0B,
  DEVELOP_BLEND_HARDLIGHT = 0x0C,
  DEVELOP_BLEND_VIVIDLIGHT = 0x0D,
  DEVELOP_BLEND_LINEARLIGHT = 0x0E,
  DEVELOP_BLEND_PINLIGHT = 0x0F,
  DEVELOP_BLEND_LIGHTNESS = 0x10,
  DEVELOP_BLEND_CHROMATICITY = 0x11,
  DEVELOP_BLEND_HUE = 0x12,
  DEVELOP_BLEND_COLOR = 0x13,
  DEVELOP_BLEND_INVERSE_OBSOLETE = 0x14, /* obsolete */
  DEVELOP_BLEND_UNBOUNDED_OBSOLETE = 0x15, /* obsolete as new normal takes over */
  DEVELOP_BLEND_COLORADJUST = 0x16,
  DEVELOP_BLEND_DIFFERENCE2 = 0x17,
  DEVELOP_BLEND_NORMAL2 = 0x18,
  DEVELOP_BLEND_BOUNDED = 0x19,
  DEVELOP_BLEND_LAB_LIGHTNESS = 0x1A,
  DEVELOP_BLEND_LAB_COLOR = 0x1B,
  DEVELOP_BLEND_HSV_VALUE = 0x1C,
  DEVELOP_BLEND_HSV_COLOR = 0x1D,
  DEVELOP_BLEND_LAB_L = 0x1E,
  DEVELOP_BLEND_LAB_A = 0x1F,
  DEVELOP_BLEND_LAB_B = 0x20,
  DEVELOP_BLEND_RGB_R = 0x21,
  DEVELOP_BLEND_RGB_G = 0x22,
  DEVELOP_BLEND_RGB_B = 0x23,
  DEVELOP_BLEND_MULTIPLY_REVERSE_OBSOLETE = 0x24, /* obsoleted by MULTIPLY + REVERSE */
  DEVELOP_BLEND_SUBTRACT_INVERSE = 0x25,
  DEVELOP_BLEND_DIVIDE = 0x26,
  DEVELOP_BLEND_DIVIDE_INVERSE = 0x27,
  DEVELOP_BLEND_GEOMETRIC_MEAN = 0x28,
  DEVELOP_BLEND_HARMONIC_MEAN = 0x29,

  DEVELOP_BLEND_REVERSE = 0x80000000,
  DEVELOP_BLEND_MODE_MASK = 0xFF,
} dt_develop_blend_mode_t;

typedef enum dt_develop_mask_mode_t
{
  DEVELOP_MASK_DISABLED = 0,                                                         // off
  DEVELOP_MASK_ENABLED = 1,                                                          // uniformly
  DEVELOP_MASK_MASK = 1 << 1,                                                        // drawn mask
  DEVELOP_MASK_CONDITIONAL = 1 << 2,                                                 // parametric mask
  DEVELOP_MASK_RASTER = 1 << 3,                                                      // raster mask
  DEVELOP_MASK_MASK_CONDITIONAL = (DEVELOP_MASK_MASK | DEVELOP_MASK_CONDITIONAL),    // drawn & parametric
  DEVELOP_MASK_FLEXI = 1 << 4                                                        // flexi mask (revamp)
} dt_develop_mask_mode_t;

typedef enum dt_develop_mask_combine_mode_t
{
  DEVELOP_COMBINE_NORM = 0x00,
  DEVELOP_COMBINE_INV = 0x01,
  DEVELOP_COMBINE_EXCL = 0x00,
  DEVELOP_COMBINE_INCL = 0x02,
  DEVELOP_COMBINE_MASKS_POS = 0x04,
  DEVELOP_COMBINE_NORM_EXCL = (DEVELOP_COMBINE_NORM | DEVELOP_COMBINE_EXCL),
  DEVELOP_COMBINE_NORM_INCL = (DEVELOP_COMBINE_NORM | DEVELOP_COMBINE_INCL),
  DEVELOP_COMBINE_INV_EXCL = (DEVELOP_COMBINE_INV | DEVELOP_COMBINE_EXCL),
  DEVELOP_COMBINE_INV_INCL = (DEVELOP_COMBINE_INV | DEVELOP_COMBINE_INCL)
} dt_develop_mask_combine_mode_t;

typedef enum dt_develop_mask_feathering_guide_t
{
  DEVELOP_MASK_GUIDE_IN_BEFORE_BLUR = 0x01,
  DEVELOP_MASK_GUIDE_OUT_BEFORE_BLUR = 0x02,
  DEVELOP_MASK_GUIDE_IN_AFTER_BLUR = 0x05,
  DEVELOP_MASK_GUIDE_OUT_AFTER_BLUR = 0x06,
} dt_develop_mask_feathering_guide_t;

typedef enum dt_develop_blendif_channels_t
{
  DEVELOP_BLENDIF_L_in = 0,
  DEVELOP_BLENDIF_A_in = 1,
  DEVELOP_BLENDIF_B_in = 2,

  DEVELOP_BLENDIF_L_out = 4,
  DEVELOP_BLENDIF_A_out = 5,
  DEVELOP_BLENDIF_B_out = 6,

  DEVELOP_BLENDIF_GRAY_in = 0,
  DEVELOP_BLENDIF_RED_in = 1,
  DEVELOP_BLENDIF_GREEN_in = 2,
  DEVELOP_BLENDIF_BLUE_in = 3,

  DEVELOP_BLENDIF_GRAY_out = 4,
  DEVELOP_BLENDIF_RED_out = 5,
  DEVELOP_BLENDIF_GREEN_out = 6,
  DEVELOP_BLENDIF_BLUE_out = 7,

  DEVELOP_BLENDIF_C_in = 8,
  DEVELOP_BLENDIF_h_in = 9,

  DEVELOP_BLENDIF_C_out = 12,
  DEVELOP_BLENDIF_h_out = 13,

  DEVELOP_BLENDIF_H_in = 8,
  DEVELOP_BLENDIF_S_in = 9,
  DEVELOP_BLENDIF_l_in = 10,

  DEVELOP_BLENDIF_H_out = 12,
  DEVELOP_BLENDIF_S_out = 13,
  DEVELOP_BLENDIF_l_out = 14,

  DEVELOP_BLENDIF_Jz_in = 8,
  DEVELOP_BLENDIF_Cz_in = 9,
  DEVELOP_BLENDIF_hz_in = 10,

  DEVELOP_BLENDIF_Jz_out = 12,
  DEVELOP_BLENDIF_Cz_out = 13,
  DEVELOP_BLENDIF_hz_out = 14,

  DEVELOP_BLENDIF_MAX = 14,
  DEVELOP_BLENDIF_unused = 15,

  DEVELOP_BLENDIF_active = 31,

  DEVELOP_BLENDIF_SIZE = 16,

  DEVELOP_BLENDIF_Lab_MASK = 0x3377,
  DEVELOP_BLENDIF_RGB_MASK = 0x77FF,
  DEVELOP_BLENDIF_OUTPUT_MASK = 0xF0F0
} dt_develop_blendif_channels_t;


/** blend parameters current version */
typedef struct dt_develop_blend_params_t
{
  /** what kind of masking to use: off, non-mask (uniformly),
   *  hand-drawn mask and/or conditional mask or raster mask */
  uint32_t mask_mode;
  /** blending color space type */
  int32_t blend_cst;
  /** blending mode */
  uint32_t blend_mode;
  /** parameter for the blending */
  float blend_parameter;
  /** mixing opacity */
  float opacity;
  /** how masks are combined */
  uint32_t mask_combine;
  /** id of mask in current pipeline */
  dt_mask_id_t mask_id;
  /** blendif mask */
  uint32_t blendif;
  /** feathering radius */
  float feathering_radius;
  /** feathering guide */
  uint32_t feathering_guide;
  /** blur radius */
  float blur_radius;
  /** mask contrast enhancement */
  float contrast;
  /** mask brightness adjustment */
  float brightness;
  /** details threshold */
  float details;
  /** feathering parameters version */
  uint32_t feather_version;
  /** some reserved fields for future use */
  uint32_t reserved[2];
  /** blendif parameters */
  float blendif_parameters[4 * DEVELOP_BLENDIF_SIZE];
  float blendif_boost_factors[DEVELOP_BLENDIF_SIZE];
  dt_dev_operation_t raster_mask_source;
  int raster_mask_instance;
  dt_mask_id_t raster_mask_id;
  gboolean raster_mask_invert;
} dt_develop_blend_params_t;

/** point struct for a DT_MASKS_PARAMETRIC form: a self-contained copy of the
 * blendif channel configuration so multiple, independent parametric masks can
 * coexist in a module's mask group, each combined with the usual operators.
 * Mirrors the blendif fields of dt_develop_blend_params_t. */
typedef struct dt_masks_point_parametric_t
{
  uint32_t blendif;                            // active channel flags (+ polarity)
  float blendif_parameters[4 * DEVELOP_BLENDIF_SIZE];
  float blendif_boost_factors[DEVELOP_BLENDIF_SIZE];
  uint32_t colorspace;                         // dt_develop_blend_colorspace_t the form was made in
  // single-channel parametric form (flexi): one parametric form edits exactly one
  // blendif channel, so several can be combined with the usual operators. `single`
  // marks a form authored this way; `channel` indexes the colorspace's channel[]
  // array (Lab_channels/rgb_channels/rgbj_channels). `in_out` is GUI-only: it
  // controls whether the output sub-channel's slider is shown next to the input
  // one in the editor (same as legacy blendif, input and output are independent,
  // additive refinements on the same channel, not alternatives -- a non-empty
  // output range still refines the mask even while its slider is hidden).
  // `invert` is the polarity, coupled to the form's invert. Older
  // (pre-single-channel) parametric forms have these all 0, so single==0 ⇒
  // legacy multi-channel form, edited with the full tabbed editor.
  uint32_t single;                             // 1 = single-channel form
  uint32_t channel;                            // index into the colorspace's channel[] array
  uint32_t in_out;                             // GUI only: 0 = show input slider only, 1 = show output slider too
  uint32_t invert;                             // polarity, coupled to the form's invert
  uint32_t compact;                            // GUI only: 1 = compact display (see _apply_param_row_filter_layout)
  uint32_t disabled;                           // bit 0: input channel disabled, bit 1: output channel disabled
} dt_masks_point_parametric_t;

/** point struct for a DT_MASKS_RASTER form: references another module's output
 * (raster) mask so it can be composited as a first-class element inside a
 * module's mask group, combined with shapes and parametric channels by the
 * usual operators. Self-describing (source module op + instance + mask id) so
 * the reference survives serialization; the module-level blend_params
 * raster_mask_* fields are kept in sync with the (single, first-cut) raster
 * element so the existing raster dependency/distortion machinery in the pipe
 * wires up unchanged (see dt_iop_commit_blend_params / dt_dev_get_raster_mask). */
typedef struct dt_masks_point_raster_t
{
  dt_dev_operation_t source;   // op of the module that produces the raster mask
  int instance;                // multi_priority of that module instance
  dt_mask_id_t id;             // which mask within the source module
} dt_masks_point_raster_t;


typedef struct dt_blendop_cl_global_t
{
  int kernel_blendop_mask_Lab;
  int kernel_blendop_mask_RAW;
  int kernel_blendop_mask_rgb_hsl;
  int kernel_blendop_mask_rgb_jzczhz;
  int kernel_blendop_Lab;
  int kernel_blendop_RAW;
  int kernel_blendop_RAW4;
  int kernel_blendop_rgb_hsl;
  int kernel_blendop_rgb_jzczhz;
  int kernel_blendop_mask_tone_curve;
  int kernel_blendop_set_mask;
  int kernel_blendop_display_channel;
  int kernel_calc_Y0_mask;
  int kernel_calc_scharr_mask;
  int kernel_calc_blend;
} dt_blendop_cl_global_t;


typedef struct dt_iop_gui_blendif_colorstop_t
{
  float stoppoint;
  GdkRGBA color;
} dt_iop_gui_blendif_colorstop_t;

typedef struct dt_iop_gui_blendif_channel_t
{
  char *label;
  char *tooltip;
  float increment;
  int numberstops;
  const dt_iop_gui_blendif_colorstop_t *colorstops;
  gboolean boost_factor_enabled;
  float boost_factor_offset;
  dt_develop_blendif_channels_t param_channels[2];
  dt_dev_pixelpipe_display_mask_t display_channel;
  void (*scale_print)(float value, float boost_factor, char *string, int n);
  int (*altdisplay)(GtkWidget *, dt_iop_module_t *, int);
  char *name;
} dt_iop_gui_blendif_channel_t;

// per-colorspace channel[] array lookup (defined in blend_gui.c); used by
// parametric.c to label a single-channel form's name after its channel
const dt_iop_gui_blendif_channel_t *dt_develop_blendif_channels_for_csp(const int csp);

// localized type-label prefix for a parametric form (channel name, or a
// generic fallback for the legacy multi-channel form); defined in
// masks/parametric.c. Used by blend_gui.c to keep a stable "what is this"
// prefix on a form's name across renames.
const char *dt_masks_parametric_type_label(const dt_masks_form_t *const form);

// group-composition mask renderers (defined in masks/group.c, non-static so
// masks/object.c can reuse them by direct reference for a committed
// DT_MASKS_OBJECT bundle -- its ->points list is structurally identical to a
// DT_MASKS_GROUP's, a GList of dt_masks_point_group_t referencing real,
// independently-registered child forms).
int dt_masks_group_get_mask(const dt_iop_module_t *const module,
                            const dt_dev_pixelpipe_iop_t *const piece,
                            struct dt_masks_form_t *const form,
                            float **buffer,
                            int *width,
                            int *height,
                            int *posx,
                            int *posy);
int dt_masks_group_get_mask_roi(const dt_iop_module_t *const module,
                                const dt_dev_pixelpipe_iop_t *const piece,
                                struct dt_masks_form_t *const form,
                                const dt_iop_roi_t *const roi,
                                float *const buffer);
void dt_masks_group_duplicate_points(struct dt_develop_t *const dev,
                                     struct dt_masks_form_t *const base,
                                     struct dt_masks_form_t *const dest);

// drop a path form's cached shrink/grow baseline+results (defined in
// masks/path.c). Used by masks/object.c to invalidate a bundle child's
// resize cache after a bundle-wide SIZE/ROTATION edit mutates its points
// directly (bypassing path.c's own property-change cases, which do this
// invalidation themselves).
void dt_masks_path_resize_invalidate(const dt_mask_id_t formid);

typedef struct dt_iop_gui_blendif_filter_t
{
  GtkDarktableGradientSlider *slider;
  GtkLabel *head;
  GtkLabel *label[4];
  GtkLabel *picker_label;
  GtkWidget *polarity;
  GtkBox *box;
  // compact-mode extras, used only by the per-row parametric editor (see
  // _build_param_row_filter / _apply_param_row_filter_layout in blend_gui.c);
  // the classic shared editor leaves these NULL. values_box is the numeric
  // values overlay (hidden in compact mode); label_box is the grid pairing
  // head + values_box for the normal layout (hidden in compact mode, when
  // the slider moves into compact_row instead); head_compact is a second
  // label instance (same text as head) shown beside the slider in compact
  // mode; compact_row holds head_compact and (when compact) the slider.
  GtkWidget *values_box;
  GtkWidget *label_box;
  GtkLabel *head_compact;
  GtkBox *compact_row;
} dt_iop_gui_blendif_filter_t;

extern const dt_introspection_type_enum_tuple_t dt_develop_blend_colorspace_names[];
extern const dt_introspection_type_enum_tuple_t dt_develop_blend_mode_names[];
extern const dt_introspection_type_enum_tuple_t dt_develop_blend_mode_flag_names[];
extern const dt_introspection_type_enum_tuple_t dt_develop_mask_mode_names[];
extern const dt_introspection_type_enum_tuple_t dt_develop_combine_masks_names[];
extern const dt_introspection_type_enum_tuple_t dt_develop_feathering_guide_names[];
extern const dt_introspection_type_enum_tuple_t dt_develop_invert_mask_names[];

#ifdef HAVE_AI
#define DEVELOP_MASKS_NB_SHAPES 6
#else
#define DEVELOP_MASKS_NB_SHAPES 5
#endif

/** blend gui data */
typedef struct dt_iop_gui_blend_data_t
{
  gboolean blendif_support;
  gboolean blend_inited;
  gboolean blendif_inited;
  gboolean masks_support;
  gboolean masks_inited;
  gboolean raster_inited;

  dt_develop_blend_colorspace_t csp;
  dt_iop_module_t *module;

  GtkWidget *iopw;
  GtkBox *blend_box;
  GtkBox *refine_box;
  // on/off toggle for the whole blend mask (DEVELOP_MASK_DISABLED vs
  // DEVELOP_MASK_ENABLED|DEVELOP_MASK_FLEXI) -- lives in the "blend mask"
  // header (masks_blend_header) since there is only one mask type left to
  // pick, see _blendop_mask_enable_toggled
  GtkWidget *mask_enable_toggle;
  // collapse button prepended to the blend-mask header, shown only while
  // this module's masking content is hosted in the separate flexi masks
  // panel (left/right) -- lets the user collapse it without a dedicated
  // panel header (see _masks_flexi_relocate in blend_gui.c)
  GtkWidget *flexi_inline_collapse_btn;
  // hamburger options button in the blend-mask header (blend colorspace,
  // masking panel position, ...) -- hidden when hosted in the utility lib,
  // since that lib's own header hamburger is repurposed to the same menu
  // there instead of showing two redundant ones (see _masks_flexi_relocate
  // and dt_iop_gui_blend_masks_options_popup)
  GtkWidget *masks_options_btn;
  // holds the blend-mask header (masks_blend_header) plus everything below
  // it (blend/opacity, masks, raster, blendif, refinement). This box is the
  // unit that gets reparented when the flexi masks panel is relocated to a
  // side panel (see _masks_flexi_relocate in blend_gui.c and
  // "plugins/darkroom/blend/masks_panel_position" conf key).
  GtkBox *relocatable_box;
  GtkBox *blendif_box;
  GtkBox *masks_box;
  GtkBox *raster_box;

  dt_iop_gui_blendif_filter_t filter[2];
  GtkWidget *showmask;
  GtkWidget *suppress;
  GtkWidget *masks_combine_combo;
  GtkWidget *blend_modes_combo;
  GtkWidget *blend_modes_blend_order;
  GtkWidget *blend_mode_parameter_slider;
  GtkWidget *opacity_slider;
  GtkWidget *masks_feathering_guide_combo;
  GtkWidget *feathering_radius_slider;
  GtkWidget *blur_radius_slider;
  GtkWidget *contrast_slider;
  GtkWidget *brightness_slider;

  dt_develop_blend_colorspace_t blend_modes_csp;
  dt_develop_blend_colorspace_t channel_tabs_csp;

  const dt_iop_gui_blendif_channel_t *channel;
  int tab;
  int altmode[8][2];
  dt_dev_pixelpipe_display_mask_t save_for_leave;
  guint timeout_handle;
  // single-channel parametric editing chrome (flexi): blendif_invert = the
  // "invert all channels" header button (hidden when a single-channel form is
  // bound, where it is meaningless). The in/out toggle itself lives on the
  // shape row (see _make_shape_row / _masks_param_inout_toggled), not in
  // this editor chrome.
  GtkWidget *blendif_invert;
  int param_output_saved;
  GtkWidget *details_slider;

  GtkWidget *masks_combo;
  // flexi-only: compact button standing in for masks_combo (which, while
  // collapsed, only ever shows the fixed "import shape" label) so the shared
  // elements row doesn't permanently reserve a full expanding combo's worth of
  // width for it. See _masks_apply_layout / _masks_import_btn_clicked.
  GtkWidget *masks_import_btn;
  GtkWidget *masks_shapes[DEVELOP_MASKS_NB_SHAPES];
  int masks_type[DEVELOP_MASKS_NB_SHAPES];
  GtkWidget *masks_edit;
  GtkWidget *masks_polarity;
  int *masks_combo_ids;
  dt_masks_edit_mode_t masks_shown;

  // in-module per-shape composition list + parametric forms (Phase 3 UI).
  // masks_list_box: one row per form in this module's mask group, each with an
  // operator chooser + inverse toggle + reorder. Each parametric row owns its
  // own permanently-visible blendif editor (see _build_param_row_editor in
  // blend_gui.c) -- there is no single shared/docked editor for flexi anymore.
  // masks_param_channels_box: flexi-only cluster of one flat button per channel
  // of the module's blend colorspace, one of masks_toolbar_row2's children (see
  // masks_toolbar below). Clicking a button adds a single-channel parametric
  // form for that channel; hovering one previews that channel's mask.
  // param_channels_csp: the csp the buttons were last built for, so the
  // cluster is rebuilt only when the csp changes.
  GtkWidget *masks_param_channels_box;
  // masks_param_channels_inner: the sub-box that actually holds the flat channel
  // buttons (rebuilt per csp); the only child of masks_param_channels_box.
  GtkWidget *masks_param_channels_inner;
  GtkWidget *masks_raster_add_btn;
  int param_channels_csp;
  // masks_new_op: the "add group" button (flexi-only). Clicking it opens an
  // operator chooser; picking an operator stages a new (empty) group of that
  // operator on top of the list, which the next drawn shape joins.
  // masks_new_op_box: the combo-box-like wrapper (icon + border) holding it --
  // one of masks_toolbar's children, a fixed, always-visible position
  // (placing it dynamically above whichever group is selected turned out to
  // rely on the panel being tall enough to scroll, which is not always the
  // case).
  GtkWidget *masks_new_op;
  GtkWidget *masks_new_op_box;
  // masks_new_op_label: the "new group" caption next to the add-group button.
  // masks_new_group_op: the operator state the next added group will use. It is
  // driven ONLY by the user picking an operator from the add-group menu, never by
  // the current selection, so the add-group icon stays put until explicitly changed.
  GtkWidget *masks_new_op_label;
  int masks_new_group_op;
  // masks_reset_mask_btn: "reset mask" action on the import-shape row (clears all
  // shapes + re-seeds the scaffold). masks_import_label is unused (the combo label).
  GtkWidget *masks_reset_mask_btn;
  // flexi-only "groups" section divider (above the toolbar/list; holds reset).
  GtkWidget *masks_groups_header;
  // shared rows re-homed between the classic two-row toolbar and the compact flexi
  // layout (see _masks_apply_layout): masks_combo_row = classic combo header
  // ([combo][invert]); masks_shapes_row = classic shapes row ([edit][shapes_box]).
  // In flexi, "edit" and "invert" move onto masks_groups_header, and
  // masks_shapes_box itself moves into masks_toolbar_row1.
  GtkWidget *masks_combo_row;
  GtkWidget *masks_shapes_row;
  // masks_toolbar: flexi's single toolbar for every "add an element to the
  // mask" action, directly below masks_groups_header and above
  // masks_list_box. A plain vertical GtkBox with exactly two fixed,
  // non-wrapping rows (masks_toolbar_row1/row2) -- no dynamic reflow. Several
  // dynamic wrapping schemes (GtkFlowBox; destroy-and-rebuild rows driven by
  // "size-allocate"; a careful in-place reflow of individually-flowing
  // widgets) were each tried and rejected in turn -- see the git history on
  // this branch -- for looking broken, racing GTK's own layout pass, or
  // leaving icon-drawn buttons invisible until an unrelated redraw, for
  // reasons that didn't resolve after substantial debugging. A fixed,
  // possibly-clipping-if-the-panel-is-extremely-narrow two-row layout is far
  // more reliable. masks_toolbar_row1: add-group (masks_new_op_box) | shape
  // buttons (masks_shapes_box) | add-raster (masks_raster_add_btn).
  // masks_toolbar_row2: parametric channel buttons
  // (masks_param_channels_box) | import/reuse (masks_import_btn). Of these,
  // only masks_shapes_box is shared with classic mode (via masks_shapes_row)
  // and needs re-homing on every layout pass; the rest are flexi-only and
  // are inserted here once, at construction (parametric buttons lazily,
  // once the csp is known), and never moved again -- switching to classic
  // just hides the whole toolbar (and everything in it) as a unit.
  GtkWidget *masks_toolbar;
  GtkWidget *masks_toolbar_row1;
  GtkWidget *masks_toolbar_row2;
  // masks_shapes_box: the shape buttons' shared home, re-homed as a unit
  // between masks_shapes_row (classic) and masks_toolbar_row1 (flexi) by
  // _masks_toolbar_place_shapes_box / _masks_apply_layout in blend_gui.c.
  GtkWidget *masks_shapes_box;
  // the "blend mask" section header: label, on/off toggle, hamburger menu,
  // showmask/suppress, and (when hosted in a side panel) the collapse
  // button -- unrelated to the "mask elements" header above.
  GtkWidget *masks_blend_header;
  GtkWidget *masks_right_cluster;
  // the flexi group list: each group's header followed by that group's element rows,
  // nested (indented) directly under it (built by _pack_group_elements).
  GtkBox *masks_list_box;
  // formid -> shape-row widget (the "mask-row" row_vbox) index, rebuilt alongside
  // masks_list_box so the per-formid lookups (hover sync, selection, in-place row
  // refresh) are O(1) instead of a recursive walk of the whole (nested) widget
  // tree on every canvas-hover motion. Populated in _make_shape_row, cleared at
  // the top of _build_masks_list; values are borrowed (owned by the widget tree).
  GHashTable *masks_row_map;
  // signature of the inputs the last _build_masks_list pass built the tree from
  // (mask model via dt_masks_group_hash + the UI-state bits the build reads).
  // When a rebuild is requested but this signature is unchanged, the tree would
  // come out byte-identical, so the teardown/rebuild is skipped. DT_INVALID_HASH
  // means "never built" (always rebuild). See _masks_list_signature.
  dt_hash_t masks_list_sig;
  // the two AI-object creation-time sliders (smoothing/cleanup), live for the
  // whole duration of an active DT_MASKS_OBJECT creation session -- built once
  // by the pending-row synthesis in _build_masks_list, NOT torn down/rebuilt on
  // every value change (see dt_masks_object_creation_apply_property's own
  // caller), so an in-progress slider drag is never interrupted. NULL outside
  // an active AI-object creation session. Canvas scroll-wheel adjustments sync
  // into these via dt_iop_gui_blend_sync_pending_ai_sliders.
  GtkWidget *pending_ai_smoothing_slider;
  GtkWidget *pending_ai_cleanup_slider;
  float pending_ai_smoothing_last;
  float pending_ai_cleanup_last;
  // blendif_home: the container the shared classic (legacy multi-channel)
  // parametric editor lives in. Flexi never reparents it anymore -- each
  // parametric row owns its own editor instead (see _build_param_row_editor).
  GtkWidget *blendif_home;
  // panel_selected_formid: the mask-list row currently selected (highlighted
  // with a border). Drawn shape or parametric form being edited; INVALID = none.
  dt_mask_id_t panel_selected_formid;
  // panel_selected_group_cid: the group (operator run) currently selected by
  // clicking its header. Identified by its first member's formid (the group
  // id). A selected group is where the next drawn shape lands and what the
  // refinement controls target. INVALID = no group selected.
  dt_mask_id_t panel_selected_group_cid;
  // empty (staged) groups: groups with an operator but no members yet, shown as
  // headers in the list until a shape is drawn into them. They are UI-side state
  // (an empty group carries nothing to serialize) kept in render order. Each entry
  // is a dt_masks_empty_group_t (private to blend_gui.c). selected_empty points at
  // the one that is the active draw target (or NULL). scaffold_seeded guards the
  // one-shot "virgin mask shows add/intersect/subtract" seeding.
  GList *empty_groups;
  void *selected_empty;
  gboolean scaffold_seeded;
  // one-shot: default-select the sole group so the panel opens ready to add
  // elements without an extra click, without forcing reselection on every
  // rebuild (which would make the sole group impossible to deselect). Reset
  // alongside scaffold_seeded wherever the mask/selection state is wiped.
  gboolean masks_selection_seeded;
  // insertion hint read by dt_masks_gui_form_save_creation (flexi only): when a
  // group is the active target, the next drawn shape is inserted right above the
  // member insert_after_fid (INVALID = on top) with operator insert_op (0 = use
  // the default_operator pref) and the insert_within in-group combine bits
  // (DT_MASKS_STATE_WITHIN subset: SCREEN/ISECT/0=union). insert_active gates the
  // whole thing; classic drawing never sets it, so it stays untouched.
  gboolean insert_active;
  dt_mask_id_t insert_after_fid;
  int insert_op;
  int insert_within;
  // opacity a shape realizing an empty group should start at (see
  // dt_masks_empty_group_t::opacity in blend_gui.c); meaningless unless
  // insert_realize_empty is set. 1.0 for every ordinary new group; a group
  // restored from a saved layout preset carries its own remembered value.
  float insert_opacity;
  // insert_realize_empty: the active target is an empty group, so the next drawn
  // shape realizes it. save_creation writes the new form id into insert_realized_fid
  // so the panel can drop the empty group and select the new run on the next rebuild.
  gboolean insert_realize_empty;
  dt_mask_id_t insert_realized_fid;
  // insert_empty: the empty group an insertion realizes when there is no explicit
  // selection (the bottom/foundation group is the default target). It drives the
  // same realize cleanup as selected_empty but without changing the selection.
  // Type is dt_masks_empty_group_t* (private to blend_gui.c).
  void *insert_empty;
  // masks_cluster_expanded: remembers the expanded/collapsed state of each
  // kind-cluster expander (runs of adjacent same-kind shapes within a single
  // group, a purely visual sub-grouping distinct from the group/operator-run
  // itself) across list rebuilds, keyed by cluster key. Lazily created.
  // value = gboolean.
  GHashTable *masks_cluster_expanded;
  // solo state: solo_formid is the form being soloed (others hidden); un-soloing
  // just clears every hidden bit, since solo is the only thing that ever sets
  // DT_MASKS_STATE_HIDDEN (real mute has been removed). INVALID when inactive.
  dt_mask_id_t solo_formid;
  // group solo: the group key currently soloed. 0 = no group soloed
  // (real keys are always >= 16).
  guint solo_group_key;
  // solo-edit: restrict which shape outlines are editable in the canvas overlay
  // (form_visible) without touching the mask computation, so the other shapes'
  // effect still shows in the mask overlay. Per-element only -- groups have no
  // solo-edit. INVALID = no solo-edit.
  dt_mask_id_t soloedit_formid;

  // Phase 2: scoped mask refinement. The "mask refinement" sliders follow the
  // current list selection (see _flexi_refine_follow_selection): global (the
  // final group mask -- the legacy behavior), a whole group, or a single
  // element. masks_refine_scope_kind/_formid track the active target;
  // masks_refine_updating guards slider repopulation against re-entrant commits.
  // Flexi-only: in classic/raster modes the scope is forced to global, so the
  // sliders read/write blend_params exactly as before.
  GtkWidget *masks_refine_reset_btn;  // resets the refinement of the active scope
  int masks_refine_scope_kind;
  dt_mask_id_t masks_refine_scope_formid;
  gboolean masks_refine_updating;
  // one-shot guard: a group header's interactive children (operator chip, expander
  // arrow) are plain buttons whose press returns TRUE (no grab), so their release
  // bubbles up to the header event box and would toggle the group selection. Those
  // handlers set this when they act so the next header release is ignored instead
  // of (de)selecting the group. Cleared on a genuine header-background press.
  gboolean masks_skip_group_select_release;
  // event time the flag above was last set at (see _group_op_press): the operator
  // handle's own plain-click branch sets the flag then returns FALSE so its own
  // and the header's drag sources can still arm, which means the very same press
  // event goes on to bubble into _group_header_press -- whose stale-flag cleanup
  // must not clobber a flag that press itself just set. Comparing event times
  // tells the two cases apart without needing a widget-identity check.
  guint32 masks_skip_group_select_release_time;
  // separate one-shot guard, set only by _group_drag_begin and consumed only by
  // _group_op_release: a plain click on the operator handle must still open the
  // operator chooser on release, but not if the press turned into a drag instead
  // (the release then just ends the drag). Kept apart from
  // masks_skip_group_select_release above -- that flag now stays TRUE for the
  // whole plain-click press/release pair (see its own comment), so it can no
  // longer double as "did a drag happen in between" without also suppressing
  // the menu on every ordinary click.
  gboolean masks_group_op_drag_started;
  // set around _auto_expand_selected_row's own programmatic
  // gtk_toggle_button_set_active calls (see blend_gui.c): its own
  // "toggling this row's expander also selects it" side effect is meant for
  // a real user click, not a toggle flipped by code to enforce "auto-expand
  // selected shape"'s single-expansion invariant -- without this guard,
  // collapsing another row's toggle re-selects it, which recurses back into
  // _auto_expand_selected_row without end. Deliberately a dedicated flag
  // rather than DT_ENTER/LEAVE_GUI_UPDATE: that one already makes
  // _props_row_toggled bail out entirely (see its own top-of-function
  // guard), which would also suppress the hash/visibility update this
  // programmatic toggle still needs to take effect.
  gboolean masks_suppress_toggle_select;
  // "auto-expand selected shape" option: the most recently selected shape
  // that actually has its own props row (see _make_props_row_toggle /
  // _auto_expand_selected_row, blend_gui.c) -- kept separate from
  // panel_selected_formid so selecting a parametric channel row or a group
  // (neither of which has a props toggle) does not collapse whichever shape
  // was expanded before. NO_MASKID (0, the zero-init value) means none yet.
  dt_mask_id_t masks_last_expanded_shape;
  // set by _row_drag_begin (element rows' handle/name), consumed by
  // _row_click_release: a plain click that turns into a real drag still gets
  // its row selected (see _row_drag_begin), so the eventual release -- drop or
  // cancel alike -- must not also run the plain-click select/toggle a second
  // time, which would flip an already-selected row straight back off. Also
  // covers a drag source spuriously arming for what was, from the user's
  // perspective, an ordinary click with no real movement (observed on macOS):
  // either way, by the time release fires the row is already correctly
  // selected, so this flag is enough to make the release a no-op.
  gboolean masks_row_click_handled;
  // suppresses _build_masks_list while set: dt_masks_form_remove() (masks.c)
  // already triggers a full flexi list rebuild via dt_masks_iop_update() on
  // every single shape it removes, so a caller removing several shapes in a
  // loop (e.g. deleting a whole group) would otherwise rebuild the whole
  // panel once per shape, then again itself -- visible as a multi-flash of
  // the list. Set this around such a loop and do exactly one explicit
  // rebuild afterwards instead.
  gboolean masks_rebuild_suppressed;
  // set while a deferred (_rebuild_masks_list_idle) rebuild is already queued on
  // the main loop, so that several rebuild requests raised by one user gesture
  // (e.g. a drag-and-drop that both reorders and reselects) collapse into a
  // single teardown/rebuild instead of running it N times. Cleared when the
  // idle fires. See _queue_masks_list_rebuild().
  gboolean masks_rebuild_pending;
  // the g_idle_add() source id for that pending rebuild, so
  // dt_iop_gui_cleanup_blending can cancel it -- an idle callback captures the
  // dt_iop_module_t* by pointer, so one left running past module teardown (at
  // darkroom exit/app quit) dereferences already-destroyed widgets (observed
  // live as a burst of GTK_IS_WIDGET/GTK_IS_BOX critical warnings right at
  // quit). 0 when nothing is pending (g_idle_add never returns 0).
  guint masks_rebuild_idle_id;
  // masks_refine_header_label: section caption, updated to name the refinement
  // target ("mask refinement — <group>" or "— whole mask").
  GtkWidget *masks_refine_section_label;
  GtkWidget *masks_refine_expander;
  GtkWidget *masks_refine_icon_box;
  GtkWidget *masks_refine_name_label;
  GtkWidget *masks_refine_indicator_icon;
  GtkWidget *masks_refine_bypass_btn;
  GtkWidget *masks_refine_toggle_btn;
  GtkBox *masks_refine_sliders_box;
  GHashTable *masks_refine_expanded;
  // transient (non-serialized, flexi-only) refinement bypass set: which
  // refinement passes the user is previewing "off". Keyed by
  // dt_masks_refine_key_*() below, except for a staged (member-less) group,
  // which is keyed by its own dt_masks_empty_group_t pointer -- it has no
  // members, so it never reaches the renderer. Owned and mutated on the GTK
  // thread only; the pixelpipe reads the snapshot taken at commit time
  // (dt_dev_refine_bypass_t, see dt_masks_refine_bypass_commit) instead.
  GHashTable *masks_refine_bypassed;

  GHashTable *masks_props_expanded;
  GHashTable *group_ordinals;

  GtkWidget *raster_combo;
  GtkWidget *raster_polarity;

  int control_button_pressed;
  dt_pthread_mutex_t lock;
} dt_iop_gui_blend_data_t;

// keys into the refinement-bypass set (bd->masks_refine_bypassed above, and the
// pipe-local snapshot built from it). Three disjoint spaces share one table: the
// whole-mask (global) pass, one element's own refinement, and one group's. The
// group space is distinguished by the top bit, which no mask id ever uses --
// dt_mask_id_t is an int32_t and ids are small positive numbers.
#define DT_MASKS_REFINE_KEY_GROUP_FLAG (0x80000000U)
#define DT_MASKS_REFINE_KEY_GLOBAL     (0U)

static inline guint32 dt_masks_refine_key_element(const dt_mask_id_t id)
{
  return (guint32)id;
}

static inline guint32 dt_masks_refine_key_group(const dt_mask_id_t cid)
{
  return (guint32)cid | DT_MASKS_REFINE_KEY_GROUP_FLAG;
}

/** copy the module's refinement-bypass set into the piece, on the thread that
    owns the GUI state. Must be called from commit_params (GTK/history thread);
    a module with no GUI (export, CLI, thumbnails) snapshots as empty, which is
    the correct "nothing bypassed" state for a preview-only feature. */
void dt_masks_refine_bypass_commit(const dt_iop_module_t *const module,
                                   dt_dev_pixelpipe_iop_t *const piece);

/** release a snapshot's key array */
void dt_masks_refine_bypass_cleanup(dt_dev_refine_bypass_t *const bypass);

/** is `key` (see dt_masks_refine_key_*) bypassed in this snapshot? */
gboolean dt_masks_refine_bypass_lookup(const dt_dev_refine_bypass_t *const bypass,
                                       const guint32 key);

/** hash of a snapshot, for mask cache invalidation */
dt_hash_t dt_masks_refine_bypass_hash(const dt_dev_refine_bypass_t *const bypass);


/** global init of blendops */
dt_blendop_cl_global_t *dt_develop_blend_init_cl_global(void);
/** global cleanup of blendops */
void dt_develop_blend_free_cl_global(dt_blendop_cl_global_t *b);

/** apply blend */
void dt_develop_blend_process(dt_iop_module_t *self,
                              dt_dev_pixelpipe_iop_t *piece,
                              const void *const i,
                              void *const o,
                              const dt_iop_roi_t *const roi_in,
                              const dt_iop_roi_t *const roi_out);

/** get blend version */
int dt_develop_blend_version(void);

/** returns the default blend color space for the given module */
dt_develop_blend_colorspace_t
dt_develop_blend_default_module_blend_colorspace(dt_iop_module_t *module);

/** initializes the default blend parameters for the given color space in blend_params */
void dt_develop_blend_init_blend_parameters(dt_develop_blend_params_t *blend_params,
                                            const dt_develop_blend_colorspace_t cst);

/** initializes the default blendif parameters for the given color space in blend_params */
void dt_develop_blend_init_blendif_parameters(dt_develop_blend_params_t *blend_params,
                                              const dt_develop_blend_colorspace_t cst);

/** returns the color space for the given module */
dt_iop_colorspace_type_t
dt_develop_blend_colorspace(const dt_dev_pixelpipe_iop_t *const piece,
                            const dt_iop_colorspace_type_t cst);

/** update blendop params to current version */
gboolean dt_develop_blend_legacy_params(dt_iop_module_t *module,
                                        const void *const old_params,
                                        const int old_version,
                                        void *new_params,
                                        const int new_version,
                                        const int length);
/** same as dt_develop_blend_legacy_params(), plus the real history-stack `num`
    this row will be written back under (or a negative value when there is no
    such row, e.g. converting a style item or a preset) -- passed through to
    dt_masks_migrate_classic_to_flexi() so a synthesized mask can be persisted
    directly under the correct main.masks_history row when one exists, since
    dt_masks_read_masks_history() replaces dev->forms wholesale from the DB
    right after history load and would otherwise discard it. See
    dt_dev_read_history_ext() for the one caller that has a real `num`. */
gboolean dt_develop_blend_legacy_params_ext(dt_iop_module_t *module,
                                            const void *const old_params,
                                            const int old_version,
                                            void *new_params,
                                            const int new_version,
                                            const int length,
                                            const int history_num);
gboolean dt_develop_blend_legacy_params_from_so(dt_iop_module_so_t *module_so,
                                                const void *const old_params,
                                                const int old_version,
                                                void *new_params,
                                                const int new_version,
                                                const int length);

/** color blending utility functions */

#define DEVELOP_BLENDIF_PARAMETER_ITEMS 6

/** initializes the parameter array (of size
 * DEVELOP_BLENDIF_PARAMETER_ITEMS * DEVELOP_BLENDIF_SIZE) */
void dt_develop_blendif_process_parameters(float *const parameters,
                                           const dt_develop_blend_params_t *const params);

/**
 * Set up a profile adapted to the blending.
 *
 * darktable built-in color profiles are chroma-adjusted such that
 * they define a [D65 RGB -> D50 XYZ] transform, which is expected by
 * CIE Lab and the ICC pipeline. Since JzAzBz expects an XYZ vector
 * adjusted for D65, we apply a Bradford transform on the profile
 * primaries to output D65 XYZ. The updated primaries are stored in
 * matrix_out. This is valid only in the context of blending with
 * JzAzBz color space. The resulting XYZ is used only to define masks
 * and not re-injected into the pipeline.
 *
 * The initialized profile may only be used to convert from RGB to XYZ.
 */
gboolean dt_develop_blendif_init_masking_profile(dt_dev_pixelpipe_iop_t *piece,
                                                 dt_iop_order_iccprofile_info_t *blending_profile,
                                                 const dt_develop_blend_colorspace_t cst);

/** apply optional per-shape refinement (details/feathering/blur/contrast/
 * brightness) to a single form's mask buffer inside the group renderer, before
 * the shape is composited. No-op when refinement is disabled. The feathering
 * guide is taken from the transient blend_refine_* context on piece. */
void dt_develop_blend_refine_form_mask(struct dt_iop_module_t *self,
                                       dt_dev_pixelpipe_iop_t *piece,
                                       float *const mask,
                                       const dt_iop_roi_t *const roi,
                                       const dt_masks_refinement_t *const r);

/** color blending mask generation functions.

    `d` is the blend configuration to evaluate; it is passed explicitly rather
    than read off `piece->blendop_data` because it is not always the piece's
    own. A parametric mask *form* (see masks/parametric.c) carries its own
    blendif config and evaluates it against the same channel machinery while
    the piece's params describe the module's drawn/flexi mask instead. The
    piece is still needed for what genuinely belongs to it (channel count,
    pipe/mask_display state). Classic callers pass piece->blendop_data. */

void dt_develop_blendif_raw_make_mask(dt_dev_pixelpipe_iop_t *piece,
                                      const dt_develop_blend_params_t *const d,
                                      const float *const a,
                                      const float *const b,
                                      const dt_iop_roi_t *const roi_in,
                                      const dt_iop_roi_t *const roi_out,
                                      float *const mask);

void dt_develop_blendif_lab_make_mask(dt_dev_pixelpipe_iop_t *piece,
                                      const dt_develop_blend_params_t *const d,
                                      const float *const a,
                                      const float *const b,
                                      const dt_iop_roi_t *const roi_in,
                                      const dt_iop_roi_t *const roi_out,
                                      float *const mask);

void dt_develop_blendif_rgb_hsl_make_mask(dt_dev_pixelpipe_iop_t *piece,
                                          const dt_develop_blend_params_t *const d,
                                          const float *const a,
                                          const float *const b,
                                          const dt_iop_roi_t *const roi_in,
                                          const dt_iop_roi_t *const roi_out,
                                          float *const mask);

void dt_develop_blendif_rgb_jzczhz_make_mask(dt_dev_pixelpipe_iop_t *piece,
                                             const dt_develop_blend_params_t *const d,
                                             const float *const a,
                                             const float *const b,
                                             const dt_iop_roi_t *const roi_in,
                                             const dt_iop_roi_t *const roi_out,
                                             float *const mask);

/** color blending operators */

void dt_develop_blendif_raw_blend(dt_dev_pixelpipe_iop_t *piece,
                                  const float *const a,
                                  float *const b,
                                  const dt_iop_roi_t *const roi_in,
                                  const dt_iop_roi_t *const roi_out,
                                  const float *const mask,
                                  const dt_dev_pixelpipe_display_mask_t request_mask_display);

void dt_develop_blendif_lab_blend(dt_dev_pixelpipe_iop_t *piece,
                                  const float *const a,
                                  float *const b,
                                  const dt_iop_roi_t *const roi_in,
                                  const dt_iop_roi_t *const roi_out,
                                  const float *const mask,
                                  const dt_dev_pixelpipe_display_mask_t request_mask_display);

void dt_develop_blendif_rgb_hsl_blend(dt_dev_pixelpipe_iop_t *piece,
                                      const float *const a,
                                      float *const b,
                                      const dt_iop_roi_t *const roi_in,
                                      const dt_iop_roi_t *const roi_out,
                                      const float *const mask,
                                      const dt_dev_pixelpipe_display_mask_t request_mask_display);

void dt_develop_blendif_rgb_jzczhz_blend(dt_dev_pixelpipe_iop_t *piece,
                                         const float *const a,
                                         float *const b,
                                         const dt_iop_roi_t *const roi_in,
                                         const dt_iop_roi_t *const roi_out,
                                         const float *const mask,
                                         const dt_dev_pixelpipe_display_mask_t request_mask_display);


/** gui related stuff */
void dt_iop_gui_init_blending(GtkWidget *iopw, dt_iop_module_t *module);
void dt_iop_gui_update_blending(dt_iop_module_t *module);
void dt_iop_gui_update_masks(dt_iop_module_t *module);
// mirror the shape currently selected/edited on the canvas into the flexi
// mask list, so its row is highlighted. No-op outside flexi / without a list.
void dt_iop_gui_masks_select_form(dt_iop_module_t *module, dt_mask_id_t formid);
// mirror the shape currently *hovered* on the canvas into the flexi mask list:
// transiently highlight its row, or its cluster's header if that cluster is
// collapsed. INVALID_MASKID clears the transient highlight. No-op outside flexi.
void dt_iop_gui_masks_hover_form(dt_iop_module_t *module, dt_mask_id_t formid);
void dt_iop_gui_cleanup_blending(dt_iop_module_t *module);
void dt_iop_gui_blending_lose_focus(dt_iop_module_t *module);
// symmetric counterpart, called when a module gains focus: relocates the
// flexi masks panel content into whichever host the user picked (see
// _masks_flexi_relocate in blend_gui.c). No-op outside flexi / embedded mode.
void dt_iop_gui_blending_gain_focus(dt_iop_module_t *module);
void dt_iop_gui_blending_reload_defaults(dt_iop_module_t *module);
// dev->forms/history was just rewritten wholesale (undo/redo, jump to a
// history step, style paste, snapshot restore, compress history): drop any
// GUI-only empty-group placeholders, since they have no counterpart in what
// was just reloaded and would otherwise duplicate a group the reload restored.
void dt_iop_gui_blend_forms_reloaded(dt_iop_module_t *module);
// opens the masking options menu (blend colorspace, masking panel
// position, ...) for whichever module is currently hosted in
// the flexi masks panel utility lib -- used by masks_flexi_host.c to
// repurpose that lib's own header hamburger instead of keeping a second,
// redundant one in the "blend mask" header (see bd->masks_options_btn)
void dt_iop_gui_blend_masks_options_popup(GtkButton *button, gpointer user_data);
// enable module's blend mask (flexi, empty if nothing was ever added yet) if
// it is currently off; no-op if it is already on (mask/flexi/raster bit set)
// or the module doesn't support blending. Used by gtk.c's flexi collapsed-
// panel corner icon: clicking it while the hosted module's mask is off
// should turn the mask on, not just re-expand the panel to an inert editor.
void dt_iop_gui_blend_mask_enable(dt_iop_module_t *module);
// re-reads the active AI-object creation session's smoothing/cleanup (via
// dt_masks_object_creation_get_preview_params) and pushes the values into
// the flexi panel's pending-row sliders (bd->pending_ai_smoothing_slider/
// pending_ai_cleanup_slider), if that module currently owns one. No-op
// otherwise. Called from object.c's canvas scroll-wheel handler so the panel
// stays in sync without a full masks-list rebuild (which would interrupt an
// in-progress slider drag).
void dt_iop_gui_blend_sync_pending_ai_sliders(dt_iop_module_t *module);

gboolean blend_color_picker_apply(dt_iop_module_t *module,
                                  GtkWidget *picker,
                                  dt_dev_pixelpipe_t *pipe);

#ifdef HAVE_OPENCL
/** apply blend for opencl modules*/
gboolean dt_develop_blend_process_cl(dt_iop_module_t *self,
                                     dt_dev_pixelpipe_iop_t *piece,
                                     cl_mem dev_in,
                                     cl_mem dev_out,
                                     const dt_iop_roi_t *roi_in,
                                     const dt_iop_roi_t *roi_out);
#endif

#define _BLEND_FUNC_PROTO(align, uni) DT_OMP_DECLARE_SIMD(aligned align uniform uni) static void

G_END_DECLS

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
