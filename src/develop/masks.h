/*
    This file is part of darktable,
    Copyright (C) 2013-2026 darktable developers.

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

#include "common/darktable.h"
#include "common/opencl.h"
#include "develop/pixelpipe.h"
#include "dtgtk/button.h"
#include "dtgtk/gradientslider.h"
#include "gui/gtk.h"

#include <assert.h>

#ifdef __cplusplus
#ifndef _Static_assert
#define _Static_assert static_assert
#endif
#endif

#define DEVELOP_MASKS_VERSION (10)

G_BEGIN_DECLS

/**forms types */
typedef enum dt_masks_type_t
{
  DT_MASKS_NONE = 0, // keep first
  DT_MASKS_CIRCLE = 1 << 0,
  DT_MASKS_PATH = 1 << 1,
  DT_MASKS_GROUP = 1 << 2,
  DT_MASKS_CLONE = 1 << 3,
  DT_MASKS_GRADIENT = 1 << 4,
  DT_MASKS_ELLIPSE = 1 << 5,
  DT_MASKS_BRUSH = 1 << 6,
  DT_MASKS_NON_CLONE = 1 << 7,
  // always defined, even without HAVE_AI: generic type-bitmask code (switch
  // statements over every dt_masks_type_t, `form->type & (DT_MASKS_GROUP |
  // DT_MASKS_OBJECT)` checks in masks.c, ...) references this unconditionally
  // -- only object.c (the code that can ever actually create a form of this
  // type) is itself gated on HAVE_AI, so without AI support this bit simply
  // never appears on any real form.
  DT_MASKS_OBJECT = 1 << 8,
  DT_MASKS_PARAMETRIC = 1 << 9,  // a parametric (blendif) mask as a first-class form
  DT_MASKS_RASTER = 1 << 10,     // a raster mask (another module's output) as a first-class form
} dt_masks_type_t;

/**masts states */
typedef enum dt_masks_state_t
{
  DT_MASKS_STATE_NONE = 0,
  DT_MASKS_STATE_USE = 1 << 0,
  DT_MASKS_STATE_SHOW = 1 << 1,
  DT_MASKS_STATE_INVERSE = 1 << 2,
  DT_MASKS_STATE_UNION = 1 << 3,
  DT_MASKS_STATE_INTERSECTION = 1 << 4,
  DT_MASKS_STATE_DIFFERENCE = 1 << 5,
  DT_MASKS_STATE_EXCLUSION = 1 << 6,
  DT_MASKS_STATE_SUM = 1 << 7,
  // a hidden form is skipped by the group renderer. Defaults off, so legacy
  // edits (which never set it) render identically. Used by the in-module
  // per-shape list's hide/solo controls.
  DT_MASKS_STATE_HIDDEN = 1 << 8,
  // screen (flexi group composition): within-group members combine by the
  // soft union a+b-ab instead of max, smoothing feathered overlaps. Broadcast
  // across a group's members. Additive new flag, so legacy edits (which never
  // set it) render identically; only consulted on the flexi group-fold path.
  DT_MASKS_STATE_SCREEN = 1 << 9,
  // multiply: composite by multiplying into the accumulator (dest *= mask),
  // mirroring how legacy parametric masks combine. Additive new operator, so
  // legacy edits (which never set it) render identically.
  DT_MASKS_STATE_MULTIPLY = 1 << 10,
  // HISTORIC, migration-only: pre-v10 forms borrowed this spare `state` bit
  // to mark a group's bottom-most member ("head") as forcing a NEW group
  // even when its operator matches the group below it -- what lets two
  // same-operator groups sit adjacent, something the implicit "groups ==
  // maximal same-operator runs" model cannot express on its own. Since
  // masks v10 this is a real per-point field,
  // dt_masks_point_group_t.group_start (masks.h below); nothing sets or
  // reads this bit via `state` anymore. It stays defined only so
  // dt_masks_legacy_params_v9_to_v10() (masks.c) can still interpret
  // pre-v10 blobs, which carry the marker here.
  DT_MASKS_STATE_GROUP_BREAK = 1 << 11,
  // intersect (flexi group composition): within-group members combine by the
  // intersection (min) instead of max, so a group can express the product of
  // its members -- e.g. reproducing a legacy multi-channel parametric mask as a
  // group of single-channel parametric elements. Mutually exclusive with SCREEN
  // (see DT_MASKS_STATE_WITHIN); neither set = union (default). Broadcast across
  // a group's members like SCREEN. Additive new flag, so legacy edits (which
  // never set it) render identically; only consulted on the flexi group fold.
  DT_MASKS_STATE_ISECT = 1 << 12,
  // between-group operator counterpart to DT_MASKS_STATE_SCREEN: composites a
  // finished group sub-mask onto the accumulator with the soft union a+b-ab
  // instead of max, so two groups with feathered edges blend across their
  // overlap instead of showing a crease. Distinct bit from the within-group
  // SCREEN flag, since a group's state carries both roles at once (its own
  // operator and its members' within-group combine mode) and the two must
  // stay independently settable. Additive new operator, so legacy edits
  // (which never set it) render identically.
  DT_MASKS_STATE_OP_SCREEN = 1 << 13,
  // disabled (group-level modifier): the group is skipped entirely by the group fold,
  // contributing nothing to the accumulated mask -- the "temporarily disable this group"
  // switch. Unlike the operators above it is a MODIFIER, not an alternative:
  // it is set alongside the group's real between-group operator (which stays
  // in the state untouched), so re-enabling restores exactly the operator the
  // group had. It is part of DT_MASKS_STATE_OP so that a disabled group and an
  // adjacent same-operator live one still read as two distinct runs (see
  // _eff_group_op / _starts_group in blend_gui.c). Use
  // DT_MASKS_STATE_OP_COMBINE wherever the *combining* operator alone is
  // wanted. Additive new bit, 0 in every pre-existing edit.
  DT_MASKS_STATE_OP_DISABLE = 1 << 14,
  DT_MASKS_STATE_OP_BYPASS = DT_MASKS_STATE_OP_DISABLE,
  // within-group counterpart to the between-group DT_MASKS_STATE_MULTIPLY:
  // members fold together by true per-pixel multiplication (dest *= member)
  // instead of ISECT's min() -- the two agree only for hard 0/1 membership,
  // not for feathered/fractional values, so this is what exactly reproduces
  // classic's own multi-channel parametric combination (`mask *= factor` per
  // channel) as a group of single-channel elements (see migrate_legacy.c).
  // Mutually exclusive with SCREEN/ISECT (see DT_MASKS_STATE_WITHIN).
  // Additive new flag, so legacy edits (which never set it) render
  // identically; only consulted on the flexi group-fold path.
  DT_MASKS_STATE_WITHIN_MULTIPLY = 1 << 15,
  // invert-output (per-run, "true" group invert): flips this run's own finished
  // sub-mask (1-grp) after its members have folded together and any per-group
  // refinement has been applied, but before it composites onto the accumulator
  // -- distinct from DT_MASKS_STATE_INVERSE (flips one member's raw mask before
  // it folds into the run) and from DEVELOP_COMBINE_MASKS_POS (a per-MODULE bit
  // that flips the whole mask after every run/group has already combined).
  // Inverting a run's output is NOT the same as inverting each of its members:
  // for anything but a single-member run the two differ (e.g. two disjoint
  // shapes unioned then inverted is 0 only on their union; each shape inverted
  // then unioned is 1 almost everywhere). This is what lets one specific
  // first-class group's own contribution be inverted without touching its
  // members' own state and without affecting any other group in the module.
  // A MODIFIER like disable, not an operator of its own: broadcast across every
  // member of the run (same reason disable is), so it is part of
  // DT_MASKS_STATE_OP and participates in run-boundary detection like disable
  // does. Additive new bit, 0 in every pre-existing edit.
  DT_MASKS_STATE_OP_INVERT = 1 << 16,
  // disabled (element-level): this element is skipped by the group fold,
  // contributing nothing to the group's mask. Defaults off.
  DT_MASKS_STATE_DISABLE = 1 << 17,
  // the between-group combining operators: exactly one of these is set on a
  // group's members (disable/invert are modifiers on top, not one of these)
  DT_MASKS_STATE_OP_COMBINE = DT_MASKS_STATE_UNION
                            | DT_MASKS_STATE_INTERSECTION
                            | DT_MASKS_STATE_DIFFERENCE
                            | DT_MASKS_STATE_SUM
                            | DT_MASKS_STATE_EXCLUSION
                            | DT_MASKS_STATE_MULTIPLY
                            | DT_MASKS_STATE_OP_SCREEN,
  DT_MASKS_STATE_OP = DT_MASKS_STATE_OP_COMBINE
                     | DT_MASKS_STATE_OP_DISABLE
                     | DT_MASKS_STATE_OP_INVERT,
  // within-group combine mode: how a group's own members fold together (before
  // the finished sub-mask is composited onto the stack by the group's OP).
  // The three bits are mutually exclusive; none set = union (max, the default).
  DT_MASKS_STATE_WITHIN = DT_MASKS_STATE_SCREEN
                        | DT_MASKS_STATE_ISECT
                        | DT_MASKS_STATE_WITHIN_MULTIPLY
} dt_masks_state_t;

// One `state` word carries three INDEPENDENT roles at once, so their bit sets
// must never overlap:
//   - the point's own between-group operator + modifiers (DT_MASKS_STATE_OP)
//   - its group's within-group combine mode      (DT_MASKS_STATE_WITHIN)
//   - per-element flags (USE/SHOW/INVERSE/HIDDEN/DISABLE)
// A collision would not fail loudly; it would read as some unrelated feature
// silently switching itself on. And every one of these bits is SERIALIZED (in
// masks blobs and XMP), so a clashing value can never simply be reassigned to
// fix it -- the migration would have to be written instead. Hence compile-time.
_Static_assert((DT_MASKS_STATE_OP & DT_MASKS_STATE_WITHIN) == 0,
               "between-group operator bits overlap the within-group combine bits");
_Static_assert((DT_MASKS_STATE_OP_COMBINE
                & (DT_MASKS_STATE_OP_DISABLE | DT_MASKS_STATE_OP_INVERT)) == 0,
               "the combining operators overlap the disable/invert modifiers;"
               " DT_MASKS_STATE_OP_COMBINE would stop isolating the operator");
// DT_MASKS_STATE_GROUP_BREAK (bit 11) is historic and migration-only, which
// makes it look like a free bit to reuse. It is not: pre-v10 blobs still carry
// the marker there and dt_masks_legacy_params_v9_to_v10() still reads it.
_Static_assert((DT_MASKS_STATE_GROUP_BREAK
                & (DT_MASKS_STATE_OP | DT_MASKS_STATE_WITHIN)) == 0,
               "the historic GROUP_BREAK bit has been reused by a live flag;"
               " pre-v10 edits would be misread by the v9->v10 migration");

// A group member's effective between-group operator. A member carrying no
// combine bit at all is what classic's dt_masks_group_add_form() gives a
// group's *first* shape, so every group inherited from a classic edit has one
// at the bottom; in that position it means "union onto what is not there yet".
//
// The fold and the panel must resolve it the same way. They partition the same
// point list into the same runs -- the panel to draw a group's rows and its
// controls, the fold to render it -- and a member whose operator one of them
// reads as union while the other reads as "no operator" lands in a different
// run on each side. The panel then shows one group whose within-group mode,
// group opacity, refinement and invert-output all read from a head the fold
// never treats as one, and every one of those controls silently does nothing.
// See _group_get_mask_roi_flexi() in masks/group.c and _starts_group() in
// blend_gui.c.
static inline dt_masks_state_t dt_masks_eff_group_op(const int state)
{
  // cast: masks.h is included from C++ too (common/exif.cc), where the masked
  // int does not convert back to the enum on its own
  const dt_masks_state_t op = (dt_masks_state_t)(state & DT_MASKS_STATE_OP);
  // what is missing is a *combining* operator, so that is what decides. Bypass
  // and invert-output are modifiers layered on one, never a substitute for it
  // (the same reading _normalize_group_operators() in blend_gui.c applies when
  // it writes the default out): testing the whole of DT_MASKS_STATE_OP instead
  // would let an operator-less head that carries one of them keep reading as
  // "no union needed", and the two partitions would part company again the
  // moment the user bypassed or inverted such a group.
  return (op & DT_MASKS_STATE_OP_COMBINE)
             ? op
             : (dt_masks_state_t)(op | DT_MASKS_STATE_UNION);
}

typedef enum dt_masks_property_t
{
  DT_MASKS_PROPERTY_OPACITY,
  DT_MASKS_PROPERTY_SIZE,
  DT_MASKS_PROPERTY_HARDNESS,
  DT_MASKS_PROPERTY_FEATHER,
  DT_MASKS_PROPERTY_ROTATION,
  DT_MASKS_PROPERTY_CURVATURE,
  DT_MASKS_PROPERTY_COMPRESSION,
  DT_MASKS_PROPERTY_CLEANUP,
  DT_MASKS_PROPERTY_SMOOTHING,
  DT_MASKS_PROPERTY_REFINE,
  DT_MASKS_PROPERTY_LAST
} dt_masks_property_t;

typedef enum dt_masks_points_states_t
{
  DT_MASKS_POINT_STATE_NORMAL = 1,
  DT_MASKS_POINT_STATE_USER = 2
} dt_masks_points_states_t;

typedef enum dt_masks_gradient_states_t
{
  DT_MASKS_GRADIENT_STATE_LINEAR = 1,
  DT_MASKS_GRADIENT_STATE_SIGMOIDAL = 2
} dt_masks_gradient_states_t;

typedef enum dt_masks_edit_mode_t
{
  DT_MASKS_EDIT_OFF = 0,
  DT_MASKS_EDIT_FULL = 1,
  DT_MASKS_EDIT_RESTRICTED = 2
} dt_masks_edit_mode_t;

typedef enum dt_masks_pressure_sensitivity_t
{
  DT_MASKS_PRESSURE_OFF = 0,
  DT_MASKS_PRESSURE_HARDNESS_REL = 1,
  DT_MASKS_PRESSURE_HARDNESS_ABS = 2,
  DT_MASKS_PRESSURE_OPACITY_REL = 3,
  DT_MASKS_PRESSURE_OPACITY_ABS = 4,
  DT_MASKS_PRESSURE_BRUSHSIZE_REL = 5
} dt_masks_pressure_sensitivity_t;

typedef enum dt_masks_ellipse_flags_t
{
  DT_MASKS_ELLIPSE_EQUIDISTANT = 0,
  DT_MASKS_ELLIPSE_PROPORTIONAL = 1
} dt_masks_ellipse_flags_t;

typedef enum dt_masks_source_pos_type_t
{
  DT_MASKS_SOURCE_POS_RELATIVE = 0,
  DT_MASKS_SOURCE_POS_RELATIVE_TEMP = 1,
  DT_MASKS_SOURCE_POS_ABSOLUTE = 2
} dt_masks_source_pos_type_t;

/* selected Bézier control point for path*/
typedef enum dt_masks_path_ctrl_t
{
  DT_MASKS_PATH_CRTL_NONE = 0,
  DT_MASKS_PATH_CTRL1 = 1,
  DT_MASKS_PATH_CTRL2 = 2

} dt_masks_path_ctrl_t;

/* restrictions on moving Bézier control points */
typedef enum dt_masks_path_edit_mode_t
{
  DT_MASKS_BEZIER_NONE = 0,        // preserve angle & scale
  DT_MASKS_BEZIER_SINGLE = 1,      // no restriction
  DT_MASKS_BEZIER_SYMMETRIC = 2,   // force full symmetry
  DT_MASKS_BEZIER_SING_SYMM = 3    // SINGLE && SYMMETRIC => force angle symmetry only
} dt_masks_path_edit_mode_t;

/** structure used to store 1 point for a circle */
typedef struct dt_masks_point_circle_t
{
  float center[2];
  float radius;
  float border;
} dt_masks_point_circle_t;

/** structure used to store 1 point for an ellipse */
typedef struct dt_masks_point_ellipse_t
{
  float center[2];
  float radius[2];
  float rotation;
  float border;
  dt_masks_ellipse_flags_t flags;
} dt_masks_point_ellipse_t;

#ifdef HAVE_AI
/** structure used to store 1 point for an object (AI segmentation) form */
typedef struct dt_masks_point_object_t
{
  float anchor[2]; // click position (normalized image coords)
  int label;       // 1 = foreground, 0 = background
} dt_masks_point_object_t;
#endif

/** structure used to store 1 point for a path form */
typedef struct dt_masks_point_path_t
{
  float corner[2];
  float ctrl1[2];
  float ctrl2[2];
  float border[2];
  dt_masks_points_states_t state;
} dt_masks_point_path_t;

/** structure used to store 1 point for a brush form */
typedef struct dt_masks_point_brush_t
{
  float corner[2];
  float ctrl1[2];
  float ctrl2[2];
  float border[2];
  float density;
  float hardness;
  dt_masks_points_states_t state;
} dt_masks_point_brush_t;

/** structure used to store anchor for a gradient */
typedef struct dt_masks_point_gradient_t
{
  float anchor[2];
  float rotation;
  float compression;
  float steepness;
  float curvature;
  dt_masks_gradient_states_t state;
} dt_masks_point_gradient_t;

/** optional per-shape mask refinements (since masks v7).
 *
 * When 'enabled' is 0 (the default, and what every pre-v7 edit migrates to)
 * the shape composites exactly as before: this struct is laid out at the tail
 * of dt_masks_point_group_t and is zero-filled when loading older edits, so
 * rendering of existing masks is bit-identical.
 *
 * The fields mirror the global refinement controls in
 * dt_develop_blend_params_t; when enabled they are applied to this single
 * shape's mask buffer (inside the group renderer) before it is composited,
 * while the global refinements still run once on the final group mask. */

/** what 'enabled' below selects: which mask the refinement is applied to.
 *
 * A per-group refinement is broadcast onto every member of the run (there is no
 * per-group storage of its own), so the value alone cannot say whether a member
 * carries its own element refinement or a copy of the group's. Without that
 * distinction the flexi renderer could only guess -- it read the run head's copy
 * and applied it to the whole group, which silently dropped every non-head
 * element's own refinement and made the head's look group-wide. */
typedef enum dt_masks_refine_scope_t
{
  DT_MASKS_REFINE_OFF = 0,      // no refinement (default; what pre-v7 edits migrate to)
  DT_MASKS_REFINE_ELEMENT = 1,  // applies to this member's own mask, before compositing
  DT_MASKS_REFINE_GROUP = 2,    // broadcast copy; applied once to the composited group mask
} dt_masks_refine_scope_t;

typedef struct dt_masks_refinement_t
{
  int32_t enabled;            // dt_masks_refine_scope_t; 0 = none (default)
  float details;              // detail-mask threshold, [-1..1]
  float feathering_radius;    // guided-filter radius, [0..]
  uint32_t feathering_guide;  // dt_develop_mask_feathering_guide_t
  float blur_radius;          // gaussian blur radius, [0..]
  float contrast;             // mask contrast, [-1..1]
  float brightness;           // mask brightness, [-1..1]
} dt_masks_refinement_t;

/** structure used to store all forms's id for a group */
typedef struct dt_masks_point_group_t
{
  dt_mask_id_t formid;
  dt_mask_id_t parentid;
  int state;
  float opacity;
  dt_masks_refinement_t refinement;  // since masks v7; zero-filled = disabled
  // since masks v8: a user-given group name (flexi first-class groups only),
  // broadcast to every member of the run so any one of them reflects the
  // whole group -- same convention as refinement above. Empty = no custom
  // name, the group shows its auto "<operator>-<ordinal>" label alone.
  char name[128];
  // since masks v9: a persistent, multiplicative group-level opacity
  // (flexi first-class groups only), broadcast to every member of the run
  // the same way refinement/name above are. Applied to the group's own
  // finished sub-mask at render time (see _group_get_mask_roi_flexi in
  // group.c), on top of -- not instead of -- each member's own independent
  // opacity; the two multiply together. Unlike refinement/name, 0.0 is NOT
  // a neutral zero-fill value here (it would silently zero out the whole
  // group), so pre-v9 edits are explicitly set to the identity value (1.0)
  // by the version migration instead of relying on zero-fill (see
  // dt_masks_legacy_params_v8_to_v9 in masks/masks.c).
  float group_opacity;
  // since masks v10: first-class group-boundary marker, replacing the
  // DT_MASKS_STATE_GROUP_BREAK bit historically borrowed from `state` (see
  // that enum value's own comment). 1 = this point starts a new group even
  // if its effective operator (_eff_group_op) matches the point below it;
  // 0 = continues that run (or is the bottom/foundation point, where a
  // break would be meaningless). Unlike group_opacity above, zero-fill IS
  // neutral here: "field absent" and "bit not set" both mean "no explicit
  // break," so pre-v10 edits need no non-zero backfill -- the version
  // migration (dt_masks_legacy_params_v9_to_v10 in masks/masks.c) only has
  // to carry forward the 1s that already existed in the old bit.
  int group_start;
} dt_masks_point_group_t;

// Does `pt` end the run whose head's effective operator is `run_op`, and start
// the next one? The single place a group boundary is decided: the fold
// (_group_get_mask_roi_flexi() in masks/group.c) and the panel (_starts_group()
// in blend_gui.c) both go through it, so they cannot partition the same point
// list differently. They once could, and a group inherited from a classic edit
// -- whose bottom member carries no combine bit -- landed on the wrong side of
// exactly that disagreement (#21905).
static inline gboolean dt_masks_point_breaks_run(const dt_masks_point_group_t *pt,
                                                 const dt_masks_state_t run_op)
{
  return pt->group_start || dt_masks_eff_group_op(pt->state) != run_op;
}

/** structure used to store pointers to the functions implementing operations on a mask shape */
/** plus a few per-class descriptive data items */
typedef struct dt_masks_functions_t
{
  int point_struct_size;   // sizeof(struct dt_masks_point_*_t)
  void (*sanitize_config)(dt_masks_type_t type_flags);
  GSList *(*setup_mouse_actions)(const struct dt_masks_form_t *const form);
  void (*set_form_name)(struct dt_masks_form_t *const form, const size_t nb);
  void (*set_hint_message)(const struct dt_masks_form_gui_t *const gui,
                           const struct dt_masks_form_t *const form,
                           const int opacity,
                           char *const __restrict__ msgbuf,
                           const size_t msgbuf_len);
  void (*modify_property)(struct dt_masks_form_t *const form,
                          dt_masks_property_t prop,
                          const float old_val,
                          const float new_val,
                          float *sum,
                          int *count,
                          float *min,
                          float *max);
  // grow/shrink (outset/inset) a shape to a signed absolute amount in the given
  // unit (use_percent: TRUE = % of shape size, FALSE = image pixels), measured
  // from a baseline captured the first time the shape is resized. Positive grows,
  // negative shrinks, 0 restores the baseline. Results are cached per offset, so
  // re-requesting a value is lossless. Returns TRUE if a usable shape resulted.
  // Currently only implemented by path masks.
  gboolean (*resize)(struct dt_masks_form_t *const form,
                     const int amount,
                     const gboolean use_percent);
  // report the resize offset currently applied to the shape, in the requested
  // unit, so a UI control can mirror it. Returns FALSE (amount 0) if no resize is
  // active. Currently only implemented by path masks.
  gboolean (*resize_get)(struct dt_masks_form_t *const form,
                         const gboolean use_percent,
                         float *amount);
  void (*duplicate_points)(dt_develop_t *const dev,
                           struct dt_masks_form_t *base,
                           struct dt_masks_form_t *dest);
  void (*initial_source_pos)(const float iwd,
                             const float iht,
                             float *x,
                             float *y);
  void (*get_distance)(const float x,
                       const float y,
                       const float as,
                       struct dt_masks_form_gui_t *gui,
                       const int index,
                       const int num_points,
                       gboolean *inside,
                       gboolean *inside_border,
                       int *near,
                       gboolean *inside_source,
                       float *dist);
  int (*get_points)(dt_develop_t *dev,
                    const float x,
                    const float y,
                    const float radius_a,
                    const float radius_b,
                    const float rotation,
                    float **points,
                    int *points_count);
  int (*get_points_border)(dt_develop_t *dev,
                           struct dt_masks_form_t *form,
                           float **points,
                           int *points_count,
                           float **border,
                           int *border_count,
                           const int source,
                           const dt_iop_module_t *const module);
  int (*get_mask)(const dt_iop_module_t *const module,
                  const dt_dev_pixelpipe_iop_t *const piece,
                  struct dt_masks_form_t *const form,
                  float **buffer,
                  int *width,
                  int *height,
                  int *posx,
                  int *posy);
  int (*get_mask_roi)(const dt_iop_module_t *const fmodule,
                      const dt_dev_pixelpipe_iop_t *const piece,
                      struct dt_masks_form_t *const form,
                      const dt_iop_roi_t *roi,
                      float *buffer);
  int (*get_area)(const dt_iop_module_t *const module,
                  const dt_dev_pixelpipe_iop_t *const piece,
                  struct dt_masks_form_t *const form,
                  int *width,
                  int *height,
                  int *posx,
                  int *posy);
  int (*get_source_area)(dt_iop_module_t *module,
                         dt_dev_pixelpipe_iop_t *piece,
                         struct dt_masks_form_t *form,
                         int *width,
                         int *height,
                         int *posx,
                         int *posy);
  int (*mouse_moved)(dt_iop_module_t *module,
                     float pzx,
                     float pzy,
                     const double pressure,
                     const int which,
                     const float zoom_scale,
                     struct dt_masks_form_t *form,
                     const dt_imgid_t parentid,
                     struct dt_masks_form_gui_t *gui,
                     const int index);
  int (*mouse_scrolled)(dt_iop_module_t *module,
                        float pzx,
                        float pzy,
                        const gboolean up,
                        uint32_t state,
                        struct dt_masks_form_t *form,
                        const dt_imgid_t parentid,
                        struct dt_masks_form_gui_t *gui,
                        const int index);
  int (*button_pressed)(dt_iop_module_t *module,
                        float pzx,
                        float pzy,
                        const double pressure,
                        const int which,
                        const int type,
                        const uint32_t state,
                        struct dt_masks_form_t *form,
                        const dt_imgid_t parentid,
                        struct dt_masks_form_gui_t *gui,
                        const int index);
  int (*button_released)(dt_iop_module_t *module,
                         float pzx,
                         float pzy,
                         const int which,
                         const uint32_t state,
                         struct dt_masks_form_t *form,
                         const dt_imgid_t parentid,
                         struct dt_masks_form_gui_t *gui,
                         const int index);
  void (*post_expose)(cairo_t *cr,
                      const float zoom_scale,
                      struct dt_masks_form_gui_t *gui,
                      const int index,
                      const int num_points);
} dt_masks_functions_t;

/** structure used to define a form */
typedef struct dt_masks_form_t
{
  GList *points; // list of point structures
  dt_masks_type_t type;
  const dt_masks_functions_t *functions;

  // position of the source (used only for clone). [0]=dx, [1]=dy, [2]=angle
  float source[3];
  // name of the form
  char name[128];
  // id used to store the form
  dt_mask_id_t formid;
  // version of the form
  int version;
} dt_masks_form_t;

typedef struct dt_masks_form_gui_points_t
{
  float *points;
  int points_count;
  float *border;
  int border_count;
  float *source;
  int source_count;
  gboolean clockwise;
} dt_masks_form_gui_points_t;

/** structure for dynamic buffers */
typedef struct dt_masks_dynbuf_t
{
  float *buffer;
  char tag[128];
  size_t pos;
  size_t size;
} dt_masks_dynbuf_t;

typedef struct dt_masks_intbuf_t
{
  int *buffer;
  char tag[128];
  size_t pos;
  size_t size;
} dt_masks_intbuf_t;


/** structure used to display a form */
typedef struct dt_masks_form_gui_t
{
  // points used to draw the form
  GList *points; // list of dt_masks_form_gui_points_t

  // points used to sample mouse moves
  dt_masks_dynbuf_t *guipoints, *guipoints_payload;
  int guipoints_count;

  // values for mouse positions, etc...
  float posx, posy, dx, dy, scrollx, scrolly, posx_source, posy_source;
  // TRUE if mouse has leaved the center window
  gboolean form_selected;
  gboolean border_selected;
  gboolean source_selected;
  gboolean source_rotating;
  gboolean counter_rotate_source;
  // joint rotation grabbed from the source shape: the mouse circles the source,
  // so its angular sweep must be measured about the source centroid (not the
  // destination centroid) to keep the rotation gain symmetric with grabbing the
  // target. The applied angle is identical either way; only the pivot used to
  // read the mouse motion differs.
  gboolean rotate_about_source;
  gboolean pivot_selected;
  gboolean select_only_border;
  dt_masks_edit_mode_t edit_mode;
  int point_selected;
  int point_edited;
  int feather_selected;
  dt_masks_path_ctrl_t bezier_ctrl; // For paths, this selects a Bézier control point.
  int seg_selected;
  int point_border_selected;
  int source_pos_type;

  gboolean form_dragging;
  gboolean source_dragging;
  gboolean form_rotating;
  gboolean border_toggling;
  gboolean gradient_toggling;
  int point_dragging;
  int feather_dragging;
  int seg_dragging;
  int point_border_dragging;

  dt_masks_path_edit_mode_t bezier_mode;  // Bézier editing with shift or ctrl
  float bezier_ctrl_angle;  // angle between ctrl1 and ctrl2
  float bezier_ctrl_scale;  // length of ctrl2 relative to ctrl1

  int group_edited;
  int group_selected;

  guint show_all_feathers;

  gboolean creation;
  gboolean creation_continuous;
  gboolean creation_closing_form;
  dt_iop_module_t *creation_module;
  dt_iop_module_t *creation_continuous_module;

  dt_masks_pressure_sensitivity_t pressure_sensitivity;

  // ids
  dt_mask_id_t formid;
  dt_hash_t pipe_hash;

  // in-module mask panel <-> canvas feedback (flexi mode; set by blend_gui.c).
  // panel_hover_formids: shapes to highlight on the canvas because their list row
  //   (a single shape) or cluster header (every member) is hovered. NULL = none.
  // panel_selected_formid: the persistent click selection mirrored from the panel.
  //   Drawn highlighted on the canvas when nothing is being hovered.
  // canvas_hover_formid: the shape currently under the cursor on the canvas; used
  //   to drive (and de-dup) the reverse "highlight the matching list row" sync.
  // solo_formids: shapes to keep highlighted on the canvas regardless of hover,
  //   because they are soloed or solo-edited in the panel (see
  //   _sync_solo_canvas_highlight in blend_gui.c). NULL = none.
  GList *panel_hover_formids;
  dt_mask_id_t panel_selected_formid;
  dt_mask_id_t canvas_hover_formid;
  GList *solo_formids;

  // opaque per-type data (e.g. segmentation context for object masks)
  void *scratchpad;
  void (*scratchpad_cleanup)(struct dt_masks_form_gui_t *gui);
} dt_masks_form_gui_t;

/** special value to indicate an invalid or uninitialized coordinate */
/** (replaces former use of NAN and isnan() by the most negative float) **/
#define DT_INVALID_COORDINATE (-FLT_MAX)

/** the shape-specific function tables */
extern const dt_masks_functions_t dt_masks_functions_circle;
extern const dt_masks_functions_t dt_masks_functions_ellipse;
extern const dt_masks_functions_t dt_masks_functions_brush;
extern const dt_masks_functions_t dt_masks_functions_path;
extern const dt_masks_functions_t dt_masks_functions_gradient;
extern const dt_masks_functions_t dt_masks_functions_group;
extern const dt_masks_functions_t dt_masks_functions_parametric;
extern const dt_masks_functions_t dt_masks_functions_raster;
/** can this raster form still obtain a mask? TRUE (unresolved) when the source
    module was removed, was never named, is switched off, or writes no raster
    mask at all -- every structural reason dt_dev_get_raster_mask() hands back
    NULL and _raster_get_mask_roi() renders all-zero. Callers outside the
    renderer use it to keep a member that can contribute nothing from being
    treated as if it could: the group fold skips its inversion, and the panel
    badges the row. Returns FALSE for anything that is not a raster form.

    `piece` may be NULL outside a pipe (the panel). Inside one, pass it: the
    source's *piece* owns the enabled state, module->enabled is not maintained
    in an export pipe.

    Deliberately not included: a mask that is merely absent from the source's
    hash table this pass. That is transient -- it resolves on the next render --
    and badging it would make the marker flicker. */
gboolean dt_masks_raster_is_unresolved(const dt_iop_module_t *module,
                                       const dt_dev_pixelpipe_iop_t *piece,
                                       const dt_masks_form_t *form);
#ifdef HAVE_AI
extern const dt_masks_functions_t dt_masks_functions_object;
/** check if AI object mask model is downloaded and AI is enabled */
gboolean dt_masks_object_available(void);
/** apply a smoothing/cleanup delta to the active AI-object creation session
    (the pending, not-yet-committed preview) -- called from the flexi panel's
    pending-row sliders. No-op if no such session is active. */
void dt_masks_object_creation_apply_property(const dt_masks_property_t prop,
                                              const float old_val,
                                              const float new_val);
/** read the active AI-object creation session's current smoothing/cleanup,
    for initial slider population and re-sync after a canvas scroll-wheel
    change. Returns FALSE (leaving outputs untouched) if no session is active. */
gboolean dt_masks_object_creation_get_preview_params(float *smoothing, int *cleanup);
#endif

/** init dt_masks_form_gui_t struct with default values */
void dt_masks_init_form_gui(dt_masks_form_gui_t *gui);

/** get points in real space with respect of distortion dx and dy are
 * used to eventually move the center of the circle */
int dt_masks_get_points_border(dt_develop_t *dev,
                               dt_masks_form_t *form,
                               float **points,
                               int *points_count,
                               float **border,
                               int *border_count,
                               const int source,
                               const dt_iop_module_t *module);

/** get the rectangle which include the form and his border */
int dt_masks_get_area(const dt_iop_module_t *module,
                      const dt_dev_pixelpipe_iop_t *piece,
                      dt_masks_form_t *form,
                      int *width,
                      int *height,
                      int *posx,
                      int *posy);
int dt_masks_get_source_area(dt_iop_module_t *module,
                             dt_dev_pixelpipe_iop_t *piece,
                             dt_masks_form_t *form,
                             int *width,
                             int *height,
                             int *posx,
                             int *posy);
/** get the transparency mask of the form and his border */
static inline int dt_masks_get_mask(const dt_iop_module_t *const module,
                                    const dt_dev_pixelpipe_iop_t *const piece,
                                    dt_masks_form_t *const form,
                                    float **buffer,
                                    int *width,
                                    int *height,
                                    int *posx,
                                    int *posy)
{
  return (form->functions && form->functions->get_mask)
    ? form->functions->get_mask(module, piece, form, buffer, width, height, posx, posy)
    : 0;
}

static inline int dt_masks_get_mask_roi(const dt_iop_module_t *const module,
                                        const dt_dev_pixelpipe_iop_t *const piece,
                                        dt_masks_form_t *const form,
                                        const dt_iop_roi_t *roi,
                                        float *buffer)
{
  return (form->functions && form->functions->get_mask_roi)
    ? form->functions->get_mask_roi(module, piece, form, roi, buffer)
    : 0;
}

int dt_masks_group_render(dt_iop_module_t *module,
                          dt_dev_pixelpipe_iop_t *piece,
                          dt_masks_form_t *form,
                          float **buffer,
                          int *roi,
                          const float scale);
int dt_masks_group_render_roi(dt_iop_module_t *module,
                              dt_dev_pixelpipe_iop_t *piece,
                              dt_masks_form_t *form,
                              const dt_iop_roi_t *roi,
                              float *buffer);

// returns current masks version
int dt_masks_version(void);

// update masks from older versions
int dt_masks_legacy_params(dt_develop_t *dev,
                           void *params,
                           const int old_version,
                           const int new_version);
/*
 * TODO:
 *
 * int
 * dt_masks_legacy_params(
 *   dt_develop_t *dev,
 *   const void *const old_params, const int old_version,
 *   void *new_params,             const int new_version);
 */

/** convert a pre-flexi module's classic mask_mode (drawn/parametric/raster,
    including the drawn+parametric combination) in-place into the flexi
    representation: DEVELOP_MASK_MASK is reused verbatim (same mask_id, no
    form changes); DEVELOP_MASK_CONDITIONAL/RASTER synthesize a new
    DT_MASKS_PARAMETRIC/DT_MASKS_RASTER form; DEVELOP_MASK_MASK_CONDITIONAL
    synthesizes a wrapper group stacking the existing drawn group under a new
    parametric element via DT_MASKS_STATE_MULTIPLY. Called from
    dt_develop_blend_legacy_params_ext() as the old_version==14 step (see
    src/develop/masks/migrate_legacy.c for the full case-by-case recipe).

    `history_num`: the exact main.history row this bp will be written back
    under, or a negative value when there is none (style/preset conversion) --
    see the comment on dt_develop_blend_legacy_params_ext() in blend.h for why
    this matters. When >= 0 and a new form actually needs synthesizing
    (CONDITIONAL/RASTER/combined -- plain DEVELOP_MASK_MASK never does), the
    work is deferred onto dev->pending_flexi_migrations rather than done here:
    mask_mode is flipped to FLEXI immediately, but mask_id is only assigned
    once dt_masks_finish_flexi_migrations() actually creates the form, since
    only that later point knows the *final* history_end each new
    main.masks_history row must be written under (see that function's own
    comment, and the field's comment in develop.h).

    Returns FALSE only on a real synthesis failure, in which case bp is left
    with its original classic mask_mode untouched (never silently drops the
    mask) and the caller should treat the whole legacy upgrade as failed. */
gboolean dt_masks_migrate_classic_to_flexi(struct dt_iop_module_t *module,
                                           struct dt_develop_blend_params_t *bp,
                                           const int history_num);

/** synthesizes the forms for every migration
    dt_masks_migrate_classic_to_flexi() deferred (see dev->pending_flexi_migrations
    in develop.h), writing each one into main.masks_history under
    dev->history_end - 1 -- the row dt_masks_read_masks_history() (called right
    after this, see dt_dev_read_history_ext() in develop.c) will treat as
    "current", i.e. the one that ends up in dev->forms/pipe->forms. Must be
    called after dev->history_end has been corrected from the DB and before
    dt_masks_read_masks_history() runs; a no-op if nothing is pending. */
void dt_masks_finish_flexi_migrations(dt_develop_t *dev);

/** Apply the flexi run-boundary normalization to every classic drawn group a
    migration reused (dev->pending_flexi_group_splits), and drain the list.

    Must be called AFTER dt_masks_read_masks_history(), which replaces
    dev->forms wholesale -- unlike dt_masks_finish_flexi_migrations(), which
    must run before it. Writes nothing to the database. */
void dt_masks_normalize_flexi_groups(dt_develop_t *dev);

/** we create a completely new form. */
dt_masks_form_t *dt_masks_create(dt_masks_type_t type);
/** we create a completely new form and add it to darktable.develop->allforms. */
dt_masks_form_t *dt_masks_create_ext(dt_masks_type_t type);
/** replace dev->forms with forms */
void dt_masks_replace_current_forms(dt_develop_t *dev, GList *forms);
/** returns a form with formid == id from a list of forms */
dt_masks_form_t *dt_masks_get_from_id_ext(GList *forms, dt_mask_id_t id);
/** returns a form with formid == id from dev->forms */
dt_masks_form_t *dt_masks_get_from_id(const dt_develop_t *dev, dt_mask_id_t id);
/** check if a form is used by a given module (directly or as a child of its group) */
gboolean dt_masks_is_in_module(dt_mask_id_t maskid, const struct dt_iop_module_t *module);
/** register forms into the mask manager */
void dt_masks_register_forms(dt_develop_t *dev,
                             GList *forms);

/** read the forms from the db */
void dt_masks_read_masks_history(dt_develop_t *dev, const dt_imgid_t imgid);
/** write the forms into the db */
void dt_masks_write_masks_history_item(const dt_imgid_t imgid,
                                       const int num,
                                       const dt_masks_form_t *form);
void dt_masks_free_form(dt_masks_form_t *form);
void dt_masks_cleanup_unused(dt_develop_t *dev);

/** function used to manipulate forms for masks */
void dt_masks_change_form_gui(dt_masks_form_t *newform);
void dt_masks_clear_form_gui(const dt_develop_t *dev);
void dt_masks_reset_form_gui(void);
void dt_masks_reset_show_masks_icons(void);

gboolean dt_masks_events_mouse_moved(struct dt_iop_module_t *module,
                                     const float x,
                                     const float y,
                                     const double pressure,
                                     const int which,
                                     const float zoom_scale);
gboolean dt_masks_events_button_released(struct dt_iop_module_t *module,
                                         const float x,
                                         const float y,
                                         const int which,
                                         const uint32_t state,
                                         const float zoom_scale);
gboolean dt_masks_events_button_pressed(struct dt_iop_module_t *module,
                                        const float x,
                                        const float y,
                                        const double pressure,
                                        const int which,
                                        const int type,
                                        const uint32_t state);
gboolean dt_masks_events_mouse_scrolled(struct dt_iop_module_t *module,
                                        const float x,
                                        const float y,
                                        const gboolean up,
                                        const uint32_t state);
// Return TRUE if scrolling over the center view should adjust the visible
// mask (size/border/opacity) instead of zoom/pan. Returns FALSE while drawing
// a path, since path creation has no scroll-adjustable parameter.
gboolean dt_masks_scroll_over_mask(void);
void dt_masks_events_post_expose(const struct dt_iop_module_t *module,
                                 cairo_t *cr,
                                 const int32_t width,
                                 const int32_t height,
                                 const float pointerx,
                                 const float pointery,
                                 const float zoom_scale);
gboolean dt_masks_events_mouse_leave(struct dt_iop_module_t *module);
gboolean dt_masks_events_mouse_enter(struct dt_iop_module_t *module);

/** functions used to manipulate gui data */
void dt_masks_gui_form_create(dt_masks_form_t *form,
                              dt_masks_form_gui_t *gui,
                              const int index,
                              const struct dt_iop_module_t *module);
void dt_masks_gui_form_remove(dt_masks_form_t *form,
                              dt_masks_form_gui_t *gui,
                              const int index);
// Constrain a drag target (in preview/processed-pipe pixel coords, wd/ht =
// processed image size) so it stays within the image expanded by
// DT_MASKS_MOVE_MARGIN. Used when translating a whole form / its anchor / clone
// source so the dragged control point stays within the image or reasonably
// close, instead of being movable to an arbitrary distance where the shape
// would be lost.
void dt_masks_clamp_move_pts(float *pts, const float wd, const float ht);
void dt_masks_gui_form_test_create(dt_masks_form_t *form,
                                   dt_masks_form_gui_t *gui,
                                   const struct dt_iop_module_t *module);
void dt_masks_gui_form_save_creation(dt_develop_t *dev,
                                     struct dt_iop_module_t *module,
                                     dt_masks_form_t *form,
                                     dt_masks_form_gui_t *gui);
// the "attach an already-registered form to the module's mask group" half of
// dt_masks_gui_form_save_creation, factored out so any finalize path that
// builds/names/registers its own form (e.g. the AI-mask object.c finalizers)
// can still land it exactly where the flexi panel's insert-hint machinery
// (bd->insert_op/insert_after_fid/insert_realize_empty, see
// _recompute_insert_hint in blend_gui.c) says the next element should go,
// instead of always appending to the module's group -- one code path for
// "where does a new element land", regardless of what created the element.
void dt_masks_group_insert_member(dt_develop_t *dev,
                                  struct dt_iop_module_t *module,
                                  dt_masks_form_t *form,
                                  dt_masks_form_gui_t *gui);
// assigns `form` the next free "<type label> #<n>" name, exactly like a
// freshly-created shape/channel gets from dt_masks_gui_form_save_creation
// (which now calls this too) -- used directly by callers that build forms
// without going through the rest of that function's GUI-creation-state and
// history-item side effects (see migrate_legacy.c)
void dt_masks_assign_unique_name(dt_develop_t *dev, dt_masks_form_t *form);
/** Set (`set`) or clear (`!set`) `bits` on every member of `grp` whose formid
 * appears in `formids` (a GList of GINT_TO_POINTER ids). Members not named are
 * left untouched; ids naming no member are ignored. This is the "broadcast one
 * attribute across a run" primitive -- a group is a maximal same-operator run
 * of `grp->points`, so its callers pass that run's member ids. */
void dt_masks_group_set_state(dt_masks_form_t *grp,
                              GList *formids,
                              const dt_masks_state_t bits,
                              const gboolean set);
/** Solo: clear `bits` on the members named by `formids` and set them on every
 * other member of `grp`. Passing formids == NULL clears `bits` on every member
 * (i.e. "solo off"), which is why this is not just the negation of
 * dt_masks_group_set_state. */
void dt_masks_group_isolate_state(dt_masks_form_t *grp,
                                  GList *formids,
                                  const dt_masks_state_t bits);
void dt_masks_group_ungroup(dt_masks_form_t *dest_grp, dt_masks_form_t *grp);
void dt_masks_group_update_name(dt_iop_module_t *module);
dt_masks_point_group_t *dt_masks_group_add_form(dt_masks_form_t *grp,
                                                const dt_masks_form_t *form);
/** returns the composition operator state to assign to a newly added form,
 * honoring the user's "default operator" preference (or the historic
 * brush=sum / else=union behavior when unset). */
dt_masks_state_t dt_masks_get_default_operator(const dt_masks_form_t *form);

void dt_masks_iop_value_changed_callback(GtkWidget *widget,
                                         struct dt_iop_module_t *module);
dt_masks_edit_mode_t dt_masks_get_edit_mode(void);
void dt_masks_set_edit_mode(struct dt_iop_module_t *module,
                            const dt_masks_edit_mode_t value);
void dt_masks_set_edit_mode_single_form(struct dt_iop_module_t *module,
                                        const dt_mask_id_t formid,
                                        const dt_masks_edit_mode_t value);
// restrict canvas editing to the given set of forms (solo-edit): only their
// outlines/handles are editable, while the full mask still computes/composites.
void dt_masks_set_edit_mode_forms(struct dt_iop_module_t *module,
                                  GList *formids,
                                  const dt_masks_edit_mode_t value);
void dt_masks_iop_update(struct dt_iop_module_t *module);
void dt_masks_iop_combo_populate(GtkWidget *w,
                                 struct dt_iop_module_t **m);
void dt_masks_iop_use_same_as(struct dt_iop_module_t *module,
                              struct dt_iop_module_t *src);
dt_hash_t dt_masks_group_hash(dt_hash_t hash, dt_masks_form_t *form);

void dt_masks_form_remove(struct dt_iop_module_t *module,
                          dt_masks_form_t *grp,
                          dt_masks_form_t *form);
float dt_masks_form_change_opacity(dt_masks_form_t *form,
                                   const dt_imgid_t parentid,
                                   const float amount);
void dt_masks_form_move(dt_masks_form_t *grp,
                        const dt_mask_id_t formid,
                        const gboolean up);
int dt_masks_form_duplicate(dt_develop_t *dev,
                            const dt_mask_id_t formid);
/* returns a duplicate tof form, including the formid */
dt_masks_form_t *dt_masks_dup_masks_form(const dt_masks_form_t *form);
/* duplicate the list of forms, replace item in the list with form with the same formid */
GList *dt_masks_dup_forms_deep(GList *forms, dt_masks_form_t *form);

/** utils functions */
gboolean dt_masks_point_in_form_exact(const float x,
                                      const float y,
                                      const float *points,
                                      const int points_start,
                                      const int points_count);
gboolean dt_masks_point_in_form_near(const float x,
                                     const float y,
                                     const float *points,
                                     const int points_start,
                                     const int points_count,
                                     const float distance,
                                     int *near);
float dt_masks_drag_factor(dt_masks_form_gui_t *gui,
                           const int index,
                           const int k,
                           const gboolean border);

float dt_masks_change_size(const gboolean up,
                           const float value,
                           const float min,
                           const float max);

float dt_masks_change_rotation(const gboolean up,
                               const float value,
                               const gboolean is_degree);

/** allow to select a shape inside an iop */
void dt_masks_select_form(struct dt_iop_module_t *module,
                          const dt_masks_form_t *sel);

/** utils for selecting the source of a clone mask while creating it */
void dt_masks_draw_clone_source_pos(cairo_t *cr,
                                    const float zoom_scale,
                                    const float x,
                                    const float y);
void dt_masks_set_source_pos_initial_state(dt_masks_form_gui_t *gui,
                                           const uint32_t state,
                                           const float pzx,
                                           const float pzy);
void dt_masks_set_source_pos_initial_value(dt_masks_form_gui_t *gui,
                                           const int mask_type,
                                           dt_masks_form_t *form,
                                           const float pzx,
                                           const float pzy);
void dt_masks_calculate_source_pos_value(const dt_masks_form_gui_t *gui,
                                         const int mask_type,
                                         const float initial_xpos,
                                         const float initial_ypos,
                                         const float xpos,
                                         const float ypos,
                                         float *px,
                                         float *py,
                                         const int adding);

/** detail mask support */
float *dt_masks_calc_scharr_mask(struct dt_dev_pixelpipe_t *pipe,
                                 float *src,
                                 const int width,
                                 const int height,
                                 const gboolean rawmode);
float *dt_masks_calc_detail_mask(struct dt_dev_pixelpipe_iop_t *piece,
                                 const float threshold,
                                 const gboolean detail);
void dt_masks_calc_detail_blend(float *const src,
                                float *out,
                                const size_t msize,
                                const float threshold,
                                const gboolean detail);


/** return the list of possible mouse actions */
GSList *dt_masks_mouse_actions(const dt_masks_form_t *form);

void dt_group_events_post_expose(cairo_t *cr,
                                 const float zoom_scale,
                                 dt_masks_form_t *form,
                                 dt_masks_form_gui_t *gui);


/******************************************************
 * code for dynamic handling of intermediate buffers
 * buffer for floats
 */
static inline gboolean _dt_masks_dynbuf_growto(dt_masks_dynbuf_t *a,
                                               const size_t newsize)
{
  float *newbuf = dt_alloc_align_float(newsize);
  if (!newbuf)
  {
    // not much we can do here except emit an error message
    dt_print(DT_DEBUG_ALWAYS,
             "critical: out of memory for dynbuf '%s' with size request %zu!",
             a->tag, newsize);
    return FALSE;
  }
  if (a->buffer)
  {
    memcpy(newbuf, a->buffer, a->size * sizeof(float));
    dt_print(DT_DEBUG_MASKS, "[masks dynbuf '%s'] grows to size %lu (is %p, was %p)",
             a->tag,
             (unsigned long)a->size, newbuf, a->buffer);
    dt_free_align(a->buffer);
  }
  a->size = newsize;
  a->buffer = newbuf;
  return TRUE;
}

static inline
dt_masks_dynbuf_t *dt_masks_dynbuf_init(const size_t size, const char *tag)
{
  assert(size > 0);
  dt_masks_dynbuf_t *a = (dt_masks_dynbuf_t *)calloc(1, sizeof(dt_masks_dynbuf_t));

  if(a != NULL)
  {
    g_strlcpy(a->tag, tag, sizeof(a->tag)); //only for debugging purposes
    a->pos = 0;
    if(_dt_masks_dynbuf_growto(a, size))
      dt_print(DT_DEBUG_MASKS, "[masks dynbuf '%s'] with initial size %lu (is %p)",
               a->tag,
               (unsigned long)a->size, a->buffer);
    if(a->buffer == NULL)
    {
      free(a);
      a = NULL;
    }
  }
  return a;
}

static inline
void dt_masks_dynbuf_add(dt_masks_dynbuf_t *a, const float value)
{
  assert(a != NULL);
  assert(a->pos <= a->size);
  if(__builtin_expect(a->pos == a->size, 0))
  {
    if (a->size == 0 || !_dt_masks_dynbuf_growto(a, 2 * a->size))
      return;
  }
  a->buffer[a->pos++] = value;
}

static inline
void dt_masks_dynbuf_add_2(dt_masks_dynbuf_t *a, const float value1, const float value2)
{
  assert(a != NULL);
  assert(a->pos <= a->size);
  if(__builtin_expect(a->pos + 2 >= a->size, 0))
  {
    if (a->size == 0 || !_dt_masks_dynbuf_growto(a, 2 * (a->size+1)))
      return;
  }
  a->buffer[a->pos++] = value1;
  a->buffer[a->pos++] = value2;
}

// Return a pointer to N floats past the current end of the dynbuf's
// contents, marking them as already in use.  The caller should then
// fill in the reserved elements using the returned pointer.
static inline
float *dt_masks_dynbuf_reserve_n(dt_masks_dynbuf_t *a, const int n)
{
  assert(a != NULL);
  assert(a->pos <= a->size);
  if(__builtin_expect(a->pos + n >= a->size, 0))
  {
    if(a->size == 0) return NULL;
    size_t newsize = a->size;
    while(a->pos + n >= newsize) newsize *= 2;
    if (!_dt_masks_dynbuf_growto(a, newsize))
    {
      return NULL;
    }
  }
  // get the current end of the (possibly reallocated) buffer, then
  // mark the next N items as in-use
  float *reserved = a->buffer + a->pos;
  a->pos += n;
  return reserved;
}

static inline
void dt_masks_dynbuf_add_zeros(dt_masks_dynbuf_t *a, const int n)
{
  assert(a != NULL);
  assert(a->pos <= a->size);
  if(__builtin_expect(a->pos + n >= a->size, 0))
  {
    if(a->size == 0) return;
    size_t newsize = a->size;
    while(a->pos + n >= newsize) newsize *= 2;
    if (!_dt_masks_dynbuf_growto(a, newsize))
    {
      return;
    }
  }
  // now that we've ensured a sufficiently large buffer add N zeros to
  // the end of the existing data
  memset(a->buffer + a->pos, 0, n * sizeof(float));
  a->pos += n;
}


static inline
float dt_masks_dynbuf_get(dt_masks_dynbuf_t *a, const int offset)
{
  assert(a != NULL);
  // offset: must be negative distance relative to end of buffer
  assert(offset < 0);
  assert((long)a->pos + offset >= 0);
  return (a->buffer[a->pos + offset]);
}

static inline
float dt_masks_dynbuf_get_absolute(dt_masks_dynbuf_t *a, const int position)
{
  assert(a != NULL);
  assert(position >= 0);
  assert((long)a->pos > position);
  return (a->buffer[position]);
}

static inline
void dt_masks_dynbuf_set(dt_masks_dynbuf_t *a, const int offset, const float value)
{
  assert(a != NULL);
  // offset: must be negative distance relative to end of buffer
  assert(offset < 0);
  assert((long)a->pos + offset >= 0);
  a->buffer[a->pos + offset] = value;
}

static inline
void dt_masks_dynbuf_set_absolute(dt_masks_dynbuf_t *a, const int position, const float value)
{
  assert(a != NULL);
  assert(position >= 0);
  assert((long)a->pos > position);
  a->buffer[position] = value;
}

static inline
float *dt_masks_dynbuf_buffer(dt_masks_dynbuf_t *a)
{
  assert(a != NULL);
  return a->buffer;
}

static inline
size_t dt_masks_dynbuf_position(dt_masks_dynbuf_t *a)
{
  assert(a != NULL);
  return a->pos;
}

static inline
void dt_masks_dynbuf_reset_position(dt_masks_dynbuf_t *a, const size_t newpos)
{
  assert(a != NULL);
  assert(newpos <= a->pos);
  a->pos = newpos;
}

static inline
void dt_masks_dynbuf_reset(dt_masks_dynbuf_t *a)
{
  assert(a != NULL);
  a->pos = 0;
}

static inline
float *dt_masks_dynbuf_harvest(dt_masks_dynbuf_t *a)
{
  // take out data buffer and make dynamic buffer obsolete
  if(a == NULL) return NULL;
  float *r = a->buffer;
  a->buffer = NULL;
  a->pos = a->size = 0;
  return r;
}

static inline
void dt_masks_dynbuf_free(dt_masks_dynbuf_t *a)
{
  if(a == NULL) return;
  dt_print(DT_DEBUG_MASKS, "[masks dynbuf '%s'] freed (was %p)", a->tag,
          a->buffer);
  dt_free_align(a->buffer);
  free(a);
}

// Dump buffer to file for debugging.
static inline
void dt_masks_dynbuf_debug_print(dt_masks_dynbuf_t *a, gboolean to_stdout)
{
  if(a == NULL) return;
  if (to_stdout)
  {
    printf("'%s' buffer: ", a->tag);
    for (size_t i = 0; i < a->pos; i += 2)
    {
      printf("(%f %f), ", a->buffer[i], a->buffer[i+1]);
    }
    printf("\n");
  }
  else
  {
    FILE *f;
    char filename[255] = { 0 };
    sprintf(filename, "debug-%ld-%s", time(NULL), a->tag);
    f = g_fopen(filename, "w");
    for (size_t i = 0; i < a->pos; i += 2)
    {
      fprintf(f, "%f %f\n", a->buffer[i], a->buffer[i+1]);
    }
    fclose(f);
  }
}

/******************************************************
 * code for dynamic handling of intermediate buffers
 * buffer for ints
 */
static inline gboolean _dt_masks_intbuf_growto(dt_masks_intbuf_t *a,
                                               const size_t newsize)
{
  int *newbuf = dt_alloc_align_int(newsize);
  if (!newbuf)
  {
    // not much we can do here except emit an error message
    dt_print(DT_DEBUG_ALWAYS,
             "critical: out of memory for intbuf '%s' with size request %zu!",
             a->tag, newsize);
    return FALSE;
  }
  if (a->buffer)
  {
    memcpy(newbuf, a->buffer, a->size * sizeof(int));
    dt_print(DT_DEBUG_MASKS, "[masks intbuf '%s'] grows to size %lu (is %p, was %p)",
             a->tag,
             (unsigned long)a->size, newbuf, a->buffer);
    dt_free_align(a->buffer);
  }
  a->size = newsize;
  a->buffer = newbuf;
  return TRUE;
}


static inline
dt_masks_intbuf_t *dt_masks_intbuf_init(const size_t size, const char *tag)
{
  assert(size > 0);
  dt_masks_intbuf_t *a = (dt_masks_intbuf_t *)calloc(1, sizeof(dt_masks_intbuf_t));

  if(a != NULL)
  {
    g_strlcpy(a->tag, tag, sizeof(a->tag)); //only for debugging purposes
    a->pos = 0;
    if(_dt_masks_intbuf_growto(a, size))
      dt_print(DT_DEBUG_MASKS, "[masks intbuf '%s'] with initial size %lu (is %p)",
               a->tag,
               (unsigned long)a->size, a->buffer);
    if(a->buffer == NULL)
    {
      free(a);
      a = NULL;
    }
  }
  return a;
}


static inline
void dt_masks_intbuf_add2(dt_masks_intbuf_t *a, const float value1, const float value2)
{
  assert(a != NULL);
  assert(a->pos <= a->size);
  if(__builtin_expect(a->pos + 2 >= a->size, 0))
  {
    if (a->size == 0 || !_dt_masks_intbuf_growto(a, 2 * (a->size+1)))
      return;
  }
  a->buffer[a->pos++] = value1;
  a->buffer[a->pos++] = value2;
}

static inline
size_t dt_masks_intbuf_position(dt_masks_intbuf_t *a)
{
  assert(a != NULL);
  return a->pos;
}

static inline
void dt_masks_intbuf_free(dt_masks_intbuf_t *a)
{
  if(a == NULL) return;
  dt_print(DT_DEBUG_MASKS, "[masks intbuf '%s'] freed (was %p)", a->tag,
          a->buffer);
  dt_free_align(a->buffer);
  free(a);
}

// Dump buffer to file for debugging.
/*
static inline
void dt_masks_intnbuf_debug_print(dt_masks_intbuf_t *a)
{
  if(a == NULL) return;
  FILE *f;
  char filename[255] = { 0 };
  sprintf(filename, "debug-%ld-%s", time(NULL), a->tag);
  f = g_fopen(filename, "w");
  for (size_t i = 0; i < a->pos; i += 2)
  {
    fprintf(f, "%d %d\n", a->buffer[i], a->buffer[i+1]);
  }
  fclose(f);
}
*/

/* End of dynamic buffer code
 ******************************************************/

static inline
int dt_masks_roundup(const int num, const int mult)
{
  const int rem = num % mult;

  return (rem == 0) ? num : num + mult - rem;
}

#define DT_MASKS_CONF(type, shape, param) \
  (type & (DT_MASKS_CLONE | DT_MASKS_NON_CLONE) \
   ? "plugins/darkroom/spots/" #shape "_" #param \
   : "plugins/darkroom/masks/" #shape "/" #param)

void dt_masks_draw_anchor(cairo_t *cr,
                          const gboolean selected,
                          const float zoom_scale,
                          const float x,
                          const float y);

/* draw the small control point for selected anchor in path & brush */
void dt_masks_draw_ctrl(cairo_t *cr,
                        const float x,
                        const float y,
                        const float zoom_scale,
                        const gboolean selected);

/* find the closest to point (px, py) in points array.
   nb_ctrl is the number of points (control points) to
   skip at the start of points.
*/
void dt_masks_closest_point(const int count,
                            const int nb_ctrl,
                            const float *points,
                            const float px,
                            const float py,
                            float *x,
                            float *y);

/* Rotate the control points of a path/brush outline in screen space and project
   them back to normalized image coordinates. `gpt_points` is the gui display
   buffer (interleaved x,y) whose first `nb*3` pairs are the control points,
   stored per node as ctrl1, corner, ctrl2; `points_count` is its number of
   (x,y) pairs. Each control point is rotated by (cos_a, sin_a) around the screen
   pivot (cx, cy), back-transformed through the pipe in a single batch, and
   written to `out` (normalized, same interleaving, nb*6 floats). Shared by the
   path and brush rotate gestures. */
void dt_masks_rotate_ctrl_points(dt_develop_t *dev,
                                 const float *const gpt_points,
                                 const int points_count,
                                 const int nb,
                                 const float cx,
                                 const float cy,
                                 const float cos_a,
                                 const float sin_a,
                                 const float iwidth,
                                 const float iheight,
                                 float *const out);

/* draw a line from -> to with an arrow at the end.
   if touch_dest is true then the arrow will be at the
   (to_x, to_y) location, otherwise a small space will
   be left.
*/
void dt_masks_draw_arrow(cairo_t *cr,
                         const float from_x,
                         const float from_y,
                         const float to_x,
                         const float to_y,
                         const float zoom_scale,
                         const gboolean touch_dest);

/* stroke the arrow on cr depending on selection */
void dt_masks_stroke_arrow(cairo_t *cr,
                           const dt_masks_form_gui_t *gui,
                           const int group,
                           const float zoom_scale);

/* set line width for the mask drawing depending on the status
   border, source & selected
*/
void dt_masks_line_stroke(cairo_t *cr,
                          const gboolean border,
                          const gboolean source,
                          const gboolean selected,
                          const float zoom_scale);

static inline float dt_masks_sensitive_dist(const float zoom_scale)
{
  return DT_PIXEL_APPLY_DPI(7) / zoom_scale;
}

static inline void dt_masks_get_image_size(float *width,
                                           float *height,
                                           float *iwidth,
                                           float *iheight)
{
  // iwidth/iheight must match preview->iwidth/iheight (= pipe->iwidth/iheight used
  // by _path_get_pts_border to scale corner coordinates before distort_transform).
  // width/height must match preview->processed_width/height, which is what both
  // dt_dev_get_preview_size() and dt_view_paint_surface FALLBACK use as canvas size.
  const dt_develop_t *dev = darktable.develop;
  const dt_dev_pixelpipe_t *preview = dev->preview_pipe;
  const float iscale = preview->iscale > 0.f ? preview->iscale : 1.f;

  // Use preview pipe's actual processed dimensions, not full.pipe/iscale.
  // The two differ by up to 1 pixel due to independent integer truncations
  // in each pipeline (e.g. after crop), causing a systematic mask overlay shift.
  // dt_dev_get_preview_size() uses the same value, so both are consistent.
  if(preview->processed_width > 0)
  {
    if(width  ) *width   = preview->processed_width;
    if(height ) *height  = preview->processed_height;
  }
  else if(dev->full.pipe && dev->full.pipe->processed_width > 0)
  {
    if(width  ) *width   = dev->full.pipe->processed_width  / iscale;
    if(height ) *height  = dev->full.pipe->processed_height / iscale;
  }
  else
  {
    if(width  ) *width   = preview->backbuf_width;
    if(height ) *height  = preview->backbuf_height;
  }

  // iwidth/iheight must equal pipe->iwidth/iheight (the pipeline input dimensions
  // used to scale corners in _path_get_pts_border / other mask get_points_border
  // functions), so that backtransform(corner * pipe->iwidth) / iwidth = corner.
  if(iwidth ) *iwidth  = preview->iwidth;
  if(iheight) *iheight = preview->iheight;

}

G_END_DECLS

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
