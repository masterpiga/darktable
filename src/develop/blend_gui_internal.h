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
// Keep this small. A symbol lands here only because a split put its definition
// and one of its callers in different files; if that stops being true, it goes
// back to static.

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
/** "collapse" button shown while hosted in a separate left/right panel */
void _flexi_inline_collapse_clicked(GtkWidget *w, gpointer user_data);
/** append the "blend mask panel position" section to an existing menu */
void _add_masks_panel_position_menu(GtkMenu *menu, dt_iop_module_t *module);

G_END_DECLS

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
