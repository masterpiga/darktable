/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

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

// Internal seam between the flexi masks panel's translation units.
//
// The panel was built as one 16k-line file (blend_gui.c) and is being split
// into cohesive pieces. Everything declared here was file-static before the
// split and is shared only between those pieces -- it is not public API, and
// blend.h remains the place for anything an IOP or the pipe may call.
//
// Keep this small. A symbol lands here only because its definition and one of
// its callers ended up in different files -- either from a split, or because
// the caller is the panel's model test suite; if that stops being true, it
// goes back to static.

#include "develop/blend.h"
#include "develop/masks.h"

G_BEGIN_DECLS

typedef struct dt_masks_empty_group_t
{
  dt_masks_state_t op;
  dt_masks_state_t within;  // within-group combine bits (DT_MASKS_STATE_WITHIN subset)
  dt_mask_id_t below_fid;
  // opacity a shape realizing this empty group should start at. Normally 1.0
  // (see dt_masks_gui_form_save_creation); a group restored from a saved
  // layout preset carries the preset's own remembered opacity instead.
  float opacity;
  // group refinement staged before the group has any members. Per-group
  // refinement normally lives in each member's dt_masks_point_group_t, so an
  // empty group has nowhere to put it -- without this, selecting the sole
  // (empty) group of a fresh or just-reset mask silently fell back to global
  // scope, making "group" and "whole mask" refinement indistinguishable. Held
  // here and adopted by the run when the group is realized (see the realize
  // block in _build_masks_list and _masks_shape_to_empty_drop).
  dt_masks_refinement_t refinement;
  // the group's displayed number, held here for the same reason real groups
  // hold theirs in bd->group_ordinals: it is an identity, not a position. 0 =
  // not assigned yet. Carried across the empty <-> real transitions so a group
  // that is emptied and refilled keeps the number it had.
  int ordinal;
  // custom name (ctrl+click the title, mirrors dt_masks_point_group_t.name on
  // a real group's members) -- NULL until set. Carried across the empty <->
  // real transitions the same way refinement/ordinal already are: adopted
  // onto every member's own pt->name when the group is realized (see
  // _masks_shape_to_empty_drop and the realize block in _build_masks_list),
  // and stashed back here from the run's head member when a group empties
  // out (see _group_reset_members and friends), instead of being silently
  // dropped as it was before.
  gchar *name;
} dt_masks_empty_group_t;

// ---------------------------------------------------------------------------
// blend_gui.c -> masks_gui_presets.c
// ---------------------------------------------------------------------------

/** the module's mask group, or NULL if it has none / it is not a group */
dt_masks_form_t *_module_mask_group(dt_iop_module_t *module);
/** the group point for `id` within `grp`, or NULL */
dt_masks_point_group_t *_group_point(dt_masks_form_t *grp, const dt_mask_id_t id);
/** a member's effective between-group operator bits */
dt_masks_state_t _eff_group_op(const int state);
/** does this points-list node start a new group run? */
gboolean _starts_group(GList *l);
/** allocate a staged (member-less) group */
dt_masks_empty_group_t *_empty_group_new(const dt_masks_state_t op,
                                         const dt_masks_state_t within,
                                         const dt_mask_id_t below_fid);
/** remove every shape and reset the panel's scratch state (no confirmation) */
void _masks_reset_mask_core(dt_iop_module_t *module);
/** destroy and rebuild the panel's row tree */
void _build_masks_list(dt_iop_module_t *module);
/** re-arm on-canvas editing after the mask changed underneath it */
void _refresh_canvas_edit(dt_iop_module_t *module);

// ---------------------------------------------------------------------------
// blend_gui.c -> the flexi panel model tests
// (src/tests/unittests/masks/test_flexi_model.c)
// ---------------------------------------------------------------------------
//
// These five are the flexi panel's *group model*: pure functions over a
// group's points list, with no GTK, no widget state and no darktable globals
// between them. Everything the panel does to the mask structure -- every
// drag/drop, every group split or merge, every operator change -- is
// ultimately expressed as a call into these, which is what makes the panel's
// behaviour testable at all without a display.
//
// They are declared here for the same reason as everything else in this file:
// a caller lives in a different translation unit. That the caller is a test
// rather than another panel file makes no difference to the seam -- but it
// does mean these five carry a stability expectation the rest of this header
// does not, since the tests are the regression net for the panel's behaviour.

/** every group run's head formid, bottom-up. Caller frees the list. */
GList *_group_partition_heads(dt_masks_form_t *grp);
/** formids of the contiguous run containing `sel`. Caller frees the list. */
GList *_selected_group_formids(dt_masks_form_t *grp, const dt_mask_id_t sel);
/** the group id (run head formid) that `fid` belongs to, or INVALID_MASKID */
dt_mask_id_t _group_cid_of_form(dt_masks_form_t *grp, const dt_mask_id_t fid);
/** map every formid -> its run's key, so the partition survives a reorder.
    Caller destroys the table. */
GHashTable *_group_keys_snapshot(dt_masks_form_t *grp);
/** re-stamp every point's group_start from a key map (see _group_keys_snapshot) */
void _group_keys_apply(dt_masks_form_t *grp, GHashTable *keys);

// Gesture semantics, split out from their GTK handlers so the handler and the
// test drive identical code. These mutate the mask structure and the panel's
// selection state; they never touch history, the pipe or the widget tree --
// the handler commits afterwards. See _model_drop_element_onto_element's own
// comment in blend_gui.c for the pattern each of these follows.

/** move element `src` into `dst`'s group, landing above or below it */
gboolean _model_drop_element_onto_element(dt_iop_module_t *module,
                                          dt_masks_form_t *grp,
                                          const dt_mask_id_t src,
                                          const dt_mask_id_t dst,
                                          const gboolean above);

/** a panel selection: an element, and the group it sits in. Either may be
    INVALID_MASKID -- an element is never selected without its group, but a
    group is routinely selected on its own. */
typedef struct dt_masks_panel_sel_t
{
  dt_mask_id_t formid;
  dt_mask_id_t group_cid;
} dt_masks_panel_sel_t;

/** the selection a click on element `id` produces */
dt_masks_panel_sel_t _model_click_element(const dt_iop_gui_blend_data_t *bd,
                                          dt_masks_form_t *grp,
                                          const dt_mask_id_t id);
/** the selection a click on group `cid` produces */
dt_masks_panel_sel_t _model_click_group(const dt_iop_gui_blend_data_t *bd,
                                        const dt_mask_id_t cid);

/** clear DT_MASKS_STATE_SHOW on the base point, ensure it elsewhere, and give
    any operator-less point the union default */
void _normalize_group_operators(dt_masks_form_t *grp);

/** move element `src` into group `dst`'s run, landing on top of it */
gboolean _model_drop_element_onto_group(dt_iop_module_t *module,
                                        dt_masks_form_t *grp,
                                        const dt_mask_id_t src,
                                        const dt_mask_id_t dst);
/** move element `src` into staged group `eg`, realizing it */
gboolean _model_drop_element_onto_empty(dt_iop_module_t *module,
                                        dt_masks_form_t *grp,
                                        const dt_mask_id_t src,
                                        dt_masks_empty_group_t *eg);
/** move a whole same-kind cluster onto an element row or a group header */
gboolean _masks_cluster_move(dt_iop_module_t *module,
                             GList *member_ids,
                             const dt_mask_id_t dst,
                             const gboolean dst_is_group,
                             const gboolean above);
/** reorder one whole group (real or staged) above/below another */
gboolean _masks_reorder_groups(dt_iop_module_t *module,
                               const gboolean src_is_empty,
                               const dt_mask_id_t src_cid,
                               dt_masks_empty_group_t *src_eg,
                               const gboolean dst_is_empty,
                               const dt_mask_id_t dst_cid,
                               dt_masks_empty_group_t *dst_eg,
                               const gboolean above);
/** one group -- real run or staged empty -- in the unified bottom-up order */
typedef struct _dt_masks_order_item_t
{
  gboolean is_empty;
  dt_mask_id_t cid;           // real: the run's head formid (ignored if is_empty)
  dt_masks_empty_group_t *eg; // empty: the group itself (ignored otherwise)
} _dt_masks_order_item_t;

/** every group (real run or staged) in bottom-up visual order.
    Caller frees with g_list_free_full(..., g_free). */
GList *_masks_visual_group_order(dt_iop_module_t *module);
/** if removing `fid` would empty its run, a placeholder preserving that
    group's operator/ordinal/name/refinement; NULL otherwise */
struct dt_masks_empty_group_t *_capture_emptied_group(dt_masks_form_t *grp,
                                                      const dt_mask_id_t fid);
/** index of the run `ids` within grp->points, and its last index; -1 if absent */
int _run_extent(dt_masks_form_t *grp, GList *ids, int *last);

/** what a solo-family toggle leaves for its caller to do to the canvas edit
    scope. The model half never touches the canvas itself. */
typedef enum dt_masks_solo_canvas_t
{
  DT_MASKS_SOLO_CANVAS_NONE = 0, // nothing to do
  DT_MASKS_SOLO_CANVAS_FULL,     // restore whole-group editing
  DT_MASKS_SOLO_CANVAS_ONE,      // narrow editing to bd->soloedit_formid
} dt_masks_solo_canvas_t;

/* Solo, group-solo and solo-edit are mutually exclusive, and at most one
   element OR one group is ever soloed. These three enforce that between them:
   each cancels the other two on the way in. */
dt_masks_solo_canvas_t _model_toggle_solo_form(dt_iop_module_t *module,
                                               dt_masks_form_t *grp,
                                               const dt_mask_id_t id);
dt_masks_solo_canvas_t _model_toggle_solo_group(dt_iop_module_t *module,
                                                dt_masks_form_t *grp,
                                                const guint key,
                                                GList *members);
dt_masks_solo_canvas_t _model_toggle_soloedit(dt_iop_module_t *module,
                                              dt_masks_form_t *grp,
                                              const dt_mask_id_t id);

/** the warning badge a row shows, if any. NOOP outranks LOW_OPACITY. */
typedef enum dt_masks_badge_kind_t
{
  DT_MASKS_BADGE_NONE = 0,
  DT_MASKS_BADGE_LOW_OPACITY,
  DT_MASKS_BADGE_NOOP,
} dt_masks_badge_kind_t;

dt_masks_badge_kind_t _model_badge_kind(const float opacity, const gboolean is_noop);

/** does this parametric form still cover its channel's whole span, i.e.
    restrict the mask not at all? */
gboolean _parametric_form_is_noop(const dt_masks_form_t *const sel);
/** has the user touched this channel's input (0) / output (1) sub-range? */
gboolean _param_channel_is_used(const dt_masks_point_parametric_t *p,
                                const dt_iop_gui_blendif_channel_t *channel,
                                const int in_out);

/** which controls a parametric row shows */
typedef struct dt_masks_param_vis_t
{
  gboolean input;
  gboolean output;
  gboolean boost;
  gboolean bypass;
} dt_masks_param_vis_t;

dt_masks_param_vis_t _model_param_row_visibility(const gboolean expanded,
                                                 const gboolean in_used,
                                                 const gboolean out_used,
                                                 const gboolean boost_enabled);

/** group numbering: identity that must survive a group emptying and refilling */
int _op_index_for_state(const int state);
int _group_ord_max_for_op(dt_iop_module_t *module, const int opidx);
int _group_ordinal_of_cid(dt_iop_module_t *module, const dt_mask_id_t cid);
void _prune_group_ordinals(dt_iop_module_t *module);
void _prune_stale_solo(dt_iop_module_t *module);

// ---------------------------------------------------------------------------
// masks_gui_presets.c -> blend_gui.c
// ---------------------------------------------------------------------------

/** append the "group layout presets" section to an existing menu */
void _add_flexi_presets_menu(GtkMenu *menu, dt_iop_module_t *module);

// ---------------------------------------------------------------------------
// blend_gui.c -> masks_gui_panel_host.c
// ---------------------------------------------------------------------------

/** re-home a widget into a new parent (no-op if already there), keeping its
    shown state */
void _reparent_into(GtkWidget *w, GtkWidget *parent,
                    const gboolean at_end, const gboolean expand);

// ---------------------------------------------------------------------------
// masks_gui_panel_host.c -> blend_gui.c
// ---------------------------------------------------------------------------

/** (re)decide where this module's panel content should live, and move it */
void _masks_flexi_relocate(dt_iop_module_t *module);
/** move this module's panel content back into its own expander */
void _masks_flexi_release(dt_iop_module_t *module);
/** "collapse" button shown in the "blend mask" header */
void _flexi_inline_collapse_clicked(GtkWidget *w, gpointer user_data);
/** record whether the masking panel should be folded away. Shared by all three
    positions, and only ever *applied* by _masks_flexi_relocate/-release, so a
    caller states the intent and every position carries it out the same way */
void _masks_panel_set_collapsed_pref(const gboolean collapsed);
/** append the "blend mask panel position" section to an existing menu */
void _add_masks_panel_position_menu(GtkMenu *menu, dt_iop_module_t *module);

G_END_DECLS

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
