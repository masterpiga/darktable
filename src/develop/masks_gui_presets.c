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

// Group-layout presets for the flexi masks panel: capture a mask's group
// skeleton, store it in the presets database under a fake operation name, and
// apply it back onto a module. Split out of blend_gui.c, where it sat between
// unrelated panel helpers; it shares only the nine symbols in
// blend_gui_internal.h with the rest of the panel.

#include "develop/blend_gui_internal.h"

#include "common/darktable.h"
#include "common/debug.h"
#include "control/conf.h"
#include "control/control.h"
#include "develop/develop.h"
#include "gui/gtk.h"

#include <sqlite3.h>

// ---- group-layout presets --------------------------------------------------
// A "layout" is just the skeleton of a flexi mask: for each group (real or
// still-empty), its between-group operator, within-group combine mode, and
// opacity -- nothing else -- no shapes, no channel/raster elements. Captured/
// applied as a plain array of _flexi_group_entry_t, one entry per group,
// bottom-to-top (index 0 is the permanent foundation group), matching the
// bottom-up convention already used by grp->points and bd->empty_groups
// everywhere else in this file.
//
// Presets are stored in the regular presets database table (reusing its schema
// and INSERT/DELETE machinery directly) under a fixed, fake operation name that
// no real image operation will ever register -- so they are global, shared by
// every module's flexi panel, rather than scoped to one iop like ordinary
// module presets. The higher-level preset GUI/apply helpers in gui/presets.c
// are not reusable here: they apply a preset by overwriting a module's whole
// params blob, which is not what a group-layout preset means (it never touches
// mask elements, let alone the rest of a module's parameters).
#define FLEXI_GROUP_PRESET_OP "flexi_mask_groups"
// v1 stored a plain dt_masks_state_t per group (no opacity); v2 added opacity,
// see _flexi_preset_list_load's version-gated blob decoding.
#define FLEXI_GROUP_PRESET_VERSION 2

// one captured/restored group: its between-group + within-group state bits,
// and its opacity (the run's member average when captured from real shapes,
// or the empty group's own remembered opacity).
typedef struct _flexi_group_entry_t
{
  dt_masks_state_t state;
  float opacity;
} _flexi_group_entry_t;

// bottom-to-top ordered snapshot of the module's current group skeleton,
// mirroring exactly how _build_masks_list merges grp->points runs with
// bd->empty_groups (unanchored empties at the very bottom, each anchored empty
// directly above the run it anchors to). Caller frees the returned array.
static _flexi_group_entry_t *_flexi_layout_capture(dt_iop_module_t *module, int *n_out)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  GArray *out = g_array_new(FALSE, FALSE, sizeof(_flexi_group_entry_t));

  for(GList *e = bd->empty_groups; e; e = g_list_next(e))
  {
    dt_masks_empty_group_t *eg = e->data;
    const gboolean anchored =
      grp && dt_is_valid_maskid(eg->below_fid) && _group_point(grp, eg->below_fid);
    if(!anchored)
    {
      const _flexi_group_entry_t ent = { eg->op | eg->within, eg->opacity };
      g_array_append_val(out, ent);
    }
  }

  for(GList *l = grp ? grp->points : NULL; l;)
  {
    const dt_masks_state_t op = _eff_group_op(((dt_masks_point_group_t *)l->data)->state);
    GList *formids = NULL;
    gboolean all_screen = TRUE, all_isect = TRUE;
    float opacity_sum = 0.0f;
    int opacity_n = 0;
    GList *m = l;
    for(; m; m = g_list_next(m))
    {
      dt_masks_point_group_t *pm = m->data;
      if(m != l && _starts_group(m))
        break;
      if(!dt_masks_get_from_id(darktable.develop, pm->formid))
        continue;
      formids = g_list_prepend(formids, GINT_TO_POINTER(pm->formid));
      if(!(pm->state & DT_MASKS_STATE_SCREEN))
        all_screen = FALSE;
      if(!(pm->state & DT_MASKS_STATE_ISECT))
        all_isect = FALSE;
      opacity_sum += pm->opacity;
      opacity_n++;
    }
    if(formids)
    {
      const dt_masks_state_t within =
        all_isect ? DT_MASKS_STATE_ISECT : (all_screen ? DT_MASKS_STATE_SCREEN : 0);
      // the group's own opacity control has no single absolute value of its
      // own (it is a delta/ratio control, see _props_row_populate) -- the
      // member average is the representative value a preset can meaningfully
      // restore.
      const _flexi_group_entry_t ent =
        { op | within, opacity_n ? opacity_sum / opacity_n : 1.0f };
      g_array_append_val(out, ent);

      for(GList *e = bd->empty_groups; e; e = g_list_next(e))
      {
        dt_masks_empty_group_t *eg = e->data;
        gboolean match = FALSE;
        for(GList *mm = formids; mm; mm = g_list_next(mm))
          if(GPOINTER_TO_INT(mm->data) == eg->below_fid)
          {
            match = TRUE;
            break;
          }
        if(match)
        {
          const _flexi_group_entry_t eent = { eg->op | eg->within, eg->opacity };
          g_array_append_val(out, eent);
        }
      }
    }
    g_list_free(formids);
    l = m;
  }

  *n_out = out->len;
  return (_flexi_group_entry_t *)g_array_free(out, FALSE);
}

// replaces the module's whole mask -- shapes AND empty-group skeleton alike --
// with a fresh skeleton of empty groups matching `entries` (same bottom-to-top
// encoding as capture). Never asks for confirmation itself; callers that might
// be discarding real shapes confirm first (see _flexi_preset_item_activate).
static void _flexi_layout_apply(dt_iop_module_t *module, const _flexi_group_entry_t *entries, int n)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  _masks_reset_mask_core(module);
  dt_masks_empty_group_t *base = NULL;
  for(int i = 0; i < n; i++)
  {
    dt_masks_empty_group_t *eg =
      _empty_group_new(entries[i].state, entries[i].state, INVALID_MASKID);
    eg->opacity = entries[i].opacity;
    bd->empty_groups = g_list_append(bd->empty_groups, eg);
    if(i == 0)
      base = eg; // index 0 is the bottom (foundation) group, see capture/apply's
                 // shared bottom-to-top convention
  }
  bd->scaffold_seeded = TRUE;
  // give the panel an immediate, unambiguous starting point -- with more than
  // one group, nothing would otherwise be selected until the user clicks one
  bd->selected_empty = base;
  _build_masks_list(module);
  _refresh_canvas_edit(module);
}

static gboolean _flexi_layout_has_content(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  return (grp && grp->points) || bd->empty_groups;
}

// reads back every user-saved layout preset's name + entry array. Caller frees
// with _flexi_preset_list_free.
typedef struct _flexi_preset_t
{
  gchar *name;
  _flexi_group_entry_t *entries;
  int n;
} _flexi_preset_t;

static GList *_flexi_preset_list_load(void)
{
  GList *out = NULL;
  sqlite3_stmt *stmt;
  DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                              "SELECT name, op_params, op_version FROM data.presets"
                              " WHERE operation = ?1 AND writeprotect = 0 ORDER BY name",
                              -1,
                              &stmt,
                              NULL);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, FLEXI_GROUP_PRESET_OP, -1, SQLITE_TRANSIENT);
  while(sqlite3_step(stmt) == SQLITE_ROW)
  {
    const int blob_size = sqlite3_column_bytes(stmt, 1);
    const int version = sqlite3_column_int(stmt, 2);
    _flexi_group_entry_t *entries = NULL;
    int n = 0;
    if(version >= 2)
    {
      // current format: a plain array of _flexi_group_entry_t
      n = blob_size / (int)sizeof(_flexi_group_entry_t);
      if(n > 0)
      {
        entries = malloc(n * sizeof(_flexi_group_entry_t));
        memcpy(entries, sqlite3_column_blob(stmt, 1), n * sizeof(_flexi_group_entry_t));
      }
    }
    else
    {
      // v1: a plain array of dt_masks_state_t, no opacity -- default to fully opaque
      n = blob_size / (int)sizeof(dt_masks_state_t);
      if(n > 0)
      {
        const dt_masks_state_t *old = sqlite3_column_blob(stmt, 1);
        entries = malloc(n * sizeof(_flexi_group_entry_t));
        for(int i = 0; i < n; i++)
        {
          entries[i].state = old[i];
          entries[i].opacity = 1.0f;
        }
      }
    }
    if(!entries)
      continue;
    _flexi_preset_t *p = malloc(sizeof(_flexi_preset_t));
    p->name = g_strdup((const gchar *)sqlite3_column_text(stmt, 0));
    p->entries = entries;
    p->n = n;
    out = g_list_append(out, p);
  }
  sqlite3_finalize(stmt);
  return out;
}

static void _flexi_preset_free(gpointer data)
{
  _flexi_preset_t *p = data;
  g_free(p->name);
  free(p->entries);
  free(p);
}

static void _flexi_preset_list_free(GList *presets)
{
  g_list_free_full(presets, _flexi_preset_free);
}

static void _flexi_preset_save_to_db(const gchar *name, const _flexi_group_entry_t *entries, int n)
{
  sqlite3_stmt *stmt;
  // clang-format off
  DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
      "INSERT OR REPLACE INTO data.presets"
      " (name, description, operation, op_version, op_params, enabled,"
      "  blendop_params, blendop_version, multi_priority, multi_name,"
      "  model, maker, lens, iso_min, iso_max, exposure_min, exposure_max,"
      "  aperture_min, aperture_max, focal_length_min, focal_length_max,"
      "  writeprotect, autoapply, filter, def, format, multi_name_hand_edited)"
      " VALUES (?1, '', ?2, ?3, ?4, 1, NULL, 0, 0, '', '%', '%', '%', 0, 0, 0, 0,"
      "         0, 0, 0, 0, 0, 0, 0, 0, 0, 0)",
      -1, &stmt, NULL);
  // clang-format on
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, name, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 2, FLEXI_GROUP_PRESET_OP, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 3, FLEXI_GROUP_PRESET_VERSION);
  DT_DEBUG_SQLITE3_BIND_BLOB(
    stmt, 4, entries, (int)(n * sizeof(_flexi_group_entry_t)), SQLITE_TRANSIENT);
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);
}

static void _flexi_preset_delete_from_db(const gchar *name)
{
  sqlite3_stmt *stmt;
  DT_DEBUG_SQLITE3_PREPARE_V2(
    dt_database_get(darktable.db),
    "DELETE FROM data.presets WHERE operation = ?1 AND name = ?2 AND writeprotect = 0",
    -1,
    &stmt,
    NULL);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, FLEXI_GROUP_PRESET_OP, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 2, name, -1, SQLITE_TRANSIENT);
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);
}

// applying a preset discards any real shapes currently in the mask (it only
// ever restores the group skeleton) -- confirm first, same as the plain reset
// button, whenever there is anything to lose.
static void
_flexi_preset_apply_confirmed(dt_iop_module_t *module, const _flexi_group_entry_t *entries, int n)
{
  if(_flexi_layout_has_content(module) &&
     !dt_gui_show_yes_no_dialog(_("apply mask layout preset?"),
                                "",
                                _("this replaces the group layout and removes every shape "
                                  "currently in this mask. continue?")))
    return;
  _flexi_layout_apply(module, entries, n);
}

static void _flexi_preset_item_activate(GtkWidget *item, dt_iop_module_t *module)
{
  const _flexi_group_entry_t *entries = g_object_get_data(G_OBJECT(item), "entries");
  const int n = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(item), "n"));
  if(entries && n > 0)
    _flexi_preset_apply_confirmed(module, entries, n);
}

static gboolean
_flexi_preset_item_press(GtkWidget *item, GdkEventButton *ev, dt_iop_module_t *module)
{
  if(ev->button != GDK_BUTTON_SECONDARY)
    return FALSE;
  const gchar *name = g_object_get_data(G_OBJECT(item), "preset-name");
  if(!name)
    return FALSE;
  if(dt_gui_show_yes_no_dialog(_("delete preset?"),
                               "",
                               _("do you really want to delete the mask layout "
                                 "preset `%s'?"),
                               name))
    _flexi_preset_delete_from_db(name);
  return TRUE;
}

static void _flexi_preset_save_clicked(GtkWidget *item, dt_iop_module_t *module)
{
  if(!_flexi_layout_has_content(module))
  {
    dt_control_log(_("the mask has no groups yet, nothing to save as a preset"));
    return;
  }
  char *name = dt_gui_show_standalone_string_dialog(
    _("save mask layout preset"),
    _("enter a name for this preset\n"
      "(only the group layout is saved, not the shapes/channels inside it):"),
    _("preset name"),
    _("cancel"),
    _("save"));
  if(!name)
    return;
  if(!*name)
    dt_control_log(_("please give the preset a name"));
  else if(!strcmp(name, _("basic")) || !strcmp(name, _("add + subtract + intersect")))
    dt_control_log(_("`%s' is a reserved preset name, please pick another one"), name);
  else
  {
    int n = 0;
    _flexi_group_entry_t *entries = _flexi_layout_capture(module, &n);
    if(n > 0)
      _flexi_preset_save_to_db(name, entries, n);
    free(entries);
  }
  g_free(name);
}

// appends a "presets" section (group-layout presets) directly to `menu` --
// used by _blendif_options_callback, the "blend mask" header's hamburger
// (formerly its own separate hamburger on the "mask elements" header)
void _add_flexi_presets_menu(GtkMenu *menu, dt_iop_module_t *module)
{
  static const _flexi_group_entry_t _preset_basic[] = { { DT_MASKS_STATE_UNION, 1.0f } };
  static const _flexi_group_entry_t _preset_ops3[] =
    { { DT_MASKS_STATE_UNION, 1.0f },
      { DT_MASKS_STATE_DIFFERENCE, 1.0f },
      { DT_MASKS_STATE_INTERSECTION, 1.0f } };

  GtkWidget *header = gtk_menu_item_new_with_label(_("group layout presets"));
  gtk_widget_set_sensitive(header, FALSE);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), header);

  struct
  {
    const char *name;
    const _flexi_group_entry_t *entries;
    int n;
  } builtins[] = { { N_("basic"), _preset_basic, 1 },
                   { N_("add + subtract + intersect"), _preset_ops3, 3 } };
  for(size_t i = 0; i < G_N_ELEMENTS(builtins); i++)
  {
    GtkWidget *item = gtk_menu_item_new_with_label(_(builtins[i].name));
    g_object_set_data(G_OBJECT(item), "entries", (gpointer)builtins[i].entries);
    g_object_set_data(G_OBJECT(item), "n", GINT_TO_POINTER(builtins[i].n));
    g_signal_connect(G_OBJECT(item), "activate", G_CALLBACK(_flexi_preset_item_activate), module);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), item);
  }

  GList *user_presets = _flexi_preset_list_load();
  if(user_presets)
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), gtk_separator_menu_item_new());
  for(GList *p = user_presets; p; p = g_list_next(p))
  {
    _flexi_preset_t *preset = p->data;
    GtkWidget *item = gtk_menu_item_new_with_label(preset->name);
    gtk_widget_set_tooltip_text(item, _("click to apply, right-click to delete"));
    _flexi_group_entry_t *entries_copy = malloc(preset->n * sizeof(_flexi_group_entry_t));
    memcpy(entries_copy, preset->entries, preset->n * sizeof(_flexi_group_entry_t));
    g_object_set_data_full(G_OBJECT(item), "entries", entries_copy, free);
    g_object_set_data(G_OBJECT(item), "n", GINT_TO_POINTER(preset->n));
    g_object_set_data_full(G_OBJECT(item), "preset-name", g_strdup(preset->name), g_free);
    g_signal_connect(G_OBJECT(item), "activate", G_CALLBACK(_flexi_preset_item_activate), module);
    g_signal_connect(
      G_OBJECT(item), "button-press-event", G_CALLBACK(_flexi_preset_item_press), module);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), item);
  }
  _flexi_preset_list_free(user_presets);

  gtk_menu_shell_append(GTK_MENU_SHELL(menu), gtk_separator_menu_item_new());
  GtkWidget *save_item = gtk_menu_item_new_with_label(_("save current layout as preset..."));
  g_signal_connect(G_OBJECT(save_item), "activate", G_CALLBACK(_flexi_preset_save_clicked), module);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), save_item);
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
