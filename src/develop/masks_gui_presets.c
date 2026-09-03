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
// still-empty), its between-group operator, within-group combine mode, name and
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
// v1 stored a plain dt_masks_state_t per group (no opacity); v2 added opacity;
// v3 added the group name. See _flexi_preset_list_load's version-gated blob
// decoding.
#define FLEXI_GROUP_PRESET_VERSION 3

// one captured/restored group: its between-group + within-group state bits, its
// name, and its opacity (the run's member average when captured from real
// shapes, or the empty group's own remembered opacity).
typedef struct _flexi_group_entry_t
{
  dt_masks_state_t state;
  float opacity;
  // same width as dt_masks_point_group_t.name, which is where it ends up once a
  // restored group gets its first member. Inline rather than a pointer because
  // the entry array is written to the database verbatim as one blob.
  char name[128];
} _flexi_group_entry_t;

// the v2 blob's element, kept for reading presets saved before names existed
typedef struct _flexi_group_entry_v2_t
{
  dt_masks_state_t state;
  float opacity;
} _flexi_group_entry_v2_t;

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
      _flexi_group_entry_t ent = { .state = eg->op | eg->within, .opacity = eg->opacity };
      if(eg->name) g_strlcpy(ent.name, eg->name, sizeof(ent.name));
      g_array_append_val(out, ent);
    }
  }

  for(GList *l = grp ? grp->points : NULL; l;)
  {
    const dt_masks_state_t op = _eff_group_op(((dt_masks_point_group_t *)l->data)->state);
    GList *formids = NULL;
    gboolean all_screen = TRUE, all_isect = TRUE, all_multiply = TRUE;
    float opacity_sum = 0.0f;
    int opacity_n = 0;
    GList *m = l;
    for(; m; m = g_list_next(m))
    {
      dt_masks_point_group_t *pm = m->data;
      if(m != l && _starts_group(m)) break;
      if(!dt_masks_get_from_id(darktable.develop, pm->formid)) continue;
      formids = g_list_prepend(formids, GINT_TO_POINTER(pm->formid));
      if(!(pm->state & DT_MASKS_STATE_SCREEN)) all_screen = FALSE;
      if(!(pm->state & DT_MASKS_STATE_ISECT)) all_isect = FALSE;
      if(!(pm->state & DT_MASKS_STATE_WITHIN_MULTIPLY)) all_multiply = FALSE;
      opacity_sum += pm->opacity;
      opacity_n++;
    }
    if(formids)
    {
      const dt_masks_state_t within =
        all_isect ? DT_MASKS_STATE_ISECT
                  : all_screen ? DT_MASKS_STATE_SCREEN
                               : all_multiply ? DT_MASKS_STATE_WITHIN_MULTIPLY : 0;
      // the group's own opacity control has no single absolute value of its
      // own (it is a delta/ratio control, see _props_row_populate) -- the
      // member average is the representative value a preset can meaningfully
      // restore.
      _flexi_group_entry_t ent = { .state = op | within,
                                   .opacity =
                                     opacity_n ? opacity_sum / opacity_n : 1.0f };
      // every member of a run carries the group's name (see the realize block in
      // _build_masks_list), so the head's copy is the group's
      g_strlcpy(ent.name, ((dt_masks_point_group_t *)l->data)->name, sizeof(ent.name));
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
          _flexi_group_entry_t eent = { .state = eg->op | eg->within,
                                        .opacity = eg->opacity };
          if(eg->name) g_strlcpy(eent.name, eg->name, sizeof(eent.name));
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
static void
_flexi_layout_apply(dt_iop_module_t *module, const _flexi_group_entry_t *entries, int n)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  _masks_reset_mask_core(module);
  dt_masks_empty_group_t *base = NULL;
  for(int i = 0; i < n; i++)
  {
    dt_masks_empty_group_t *eg =
      _empty_group_new(entries[i].state, entries[i].state, INVALID_MASKID);
    eg->opacity = entries[i].opacity;
    if(entries[i].name[0]) eg->name = g_strdup(entries[i].name);
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
                              -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, FLEXI_GROUP_PRESET_OP, -1, SQLITE_TRANSIENT);
  while(sqlite3_step(stmt) == SQLITE_ROW)
  {
    const int blob_size = sqlite3_column_bytes(stmt, 1);
    const int version = sqlite3_column_int(stmt, 2);
    _flexi_group_entry_t *entries = NULL;
    int n = 0;
    if(version >= 3)
    {
      // current format: a plain array of _flexi_group_entry_t
      n = blob_size / (int)sizeof(_flexi_group_entry_t);
      if(n > 0)
      {
        entries = malloc(n * sizeof(_flexi_group_entry_t));
        memcpy(entries, sqlite3_column_blob(stmt, 1), n * sizeof(_flexi_group_entry_t));
      }
    }
    else if(version == 2)
    {
      // v2: state + opacity, no name -- the groups come back unnamed
      n = blob_size / (int)sizeof(_flexi_group_entry_v2_t);
      if(n > 0)
      {
        const _flexi_group_entry_v2_t *old = sqlite3_column_blob(stmt, 1);
        entries = calloc(n, sizeof(_flexi_group_entry_t));
        for(int i = 0; i < n; i++)
        {
          entries[i].state = old[i].state;
          entries[i].opacity = old[i].opacity;
        }
      }
    }
    else
    {
      // v1: a plain array of dt_masks_state_t, no opacity -- default to fully opaque
      n = blob_size / (int)sizeof(dt_masks_state_t);
      if(n > 0)
      {
        const dt_masks_state_t *old = sqlite3_column_blob(stmt, 1);
        entries = calloc(n, sizeof(_flexi_group_entry_t));
        for(int i = 0; i < n; i++)
        {
          entries[i].state = old[i];
          entries[i].opacity = 1.0f;
        }
      }
    }
    if(!entries) continue;
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

static void
_flexi_preset_save_to_db(const gchar *name, const _flexi_group_entry_t *entries, int n)
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
  DT_DEBUG_SQLITE3_BIND_BLOB(stmt, 4, entries, (int)(n * sizeof(_flexi_group_entry_t)),
                             SQLITE_TRANSIENT);
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);
}

static void _flexi_preset_delete_from_db(const gchar *name)
{
  sqlite3_stmt *stmt;
  DT_DEBUG_SQLITE3_PREPARE_V2(
    dt_database_get(darktable.db),
    "DELETE FROM data.presets WHERE operation = ?1 AND name = ?2 AND writeprotect = 0",
    -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, FLEXI_GROUP_PRESET_OP, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 2, name, -1, SQLITE_TRANSIENT);
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);
}

// applying a preset discards any real shapes currently in the mask (it only
// ever restores the group skeleton) -- confirm first, same as the plain reset
// button, whenever there is anything to lose.
static void _flexi_preset_apply_confirmed(dt_iop_module_t *module,
                                          const _flexi_group_entry_t *entries,
                                          int n)
{
  if(_flexi_layout_has_content(module)
     && !dt_gui_show_yes_no_dialog(
       _("apply mask layout preset?"), "",
       _("this replaces the group layout and removes every shape "
         "currently in this mask. continue?")))
    return;
  _flexi_layout_apply(module, entries, n);
}

static void _flexi_preset_item_activate(GtkWidget *item, dt_iop_module_t *module)
{
  const _flexi_group_entry_t *entries = g_object_get_data(G_OBJECT(item), "entries");
  const int n = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(item), "n"));
  if(entries && n > 0) _flexi_preset_apply_confirmed(module, entries, n);
}

static gboolean
_flexi_preset_item_press(GtkWidget *item, GdkEventButton *ev, dt_iop_module_t *module)
{
  if(ev->button != GDK_BUTTON_SECONDARY) return FALSE;
  const gchar *name = g_object_get_data(G_OBJECT(item), "preset-name");
  if(!name) return FALSE;
  if(dt_gui_show_yes_no_dialog(_("delete preset?"), "",
                               _("do you really want to delete the mask layout "
                                 "preset `%s'?"),
                               name))
    _flexi_preset_delete_from_db(name);
  return TRUE;
}

// ---- built-in layouts ------------------------------------------------------
// Listed above the user's own presets, and their names are reserved so a user
// preset cannot shadow one. The last three exist to make classic masks
// translate onto the panel in one click: classic composed its drawn shapes with
// each other, its parametric channels with each other, and then multiplied the
// two results together -- which here is a screened group of shapes intersected
// with a multiplied group of channels.
typedef struct _flexi_group_spec_t
{
  dt_masks_state_t state; // between-group operator | within-group combine mode
  const char *name;       // untranslated group name, NULL to leave it unnamed
} _flexi_group_spec_t;

typedef struct _flexi_builtin_t
{
  const char *name;                  // untranslated preset name
  const char *tooltip;               // untranslated, NULL for none
  const _flexi_group_spec_t *groups; // bottom-to-top, as everywhere else here
  int n;
} _flexi_builtin_t;

static const _flexi_group_spec_t _spec_basic[] = { { DT_MASKS_STATE_UNION, NULL } };

static const _flexi_group_spec_t _spec_ops3[] = { { DT_MASKS_STATE_UNION, NULL },
                                                  { DT_MASKS_STATE_DIFFERENCE, NULL },
                                                  { DT_MASKS_STATE_INTERSECTION, NULL } };

static const _flexi_group_spec_t _spec_drawn[] = {
  { DT_MASKS_STATE_UNION | DT_MASKS_STATE_SCREEN, N_("shapes") }
};

static const _flexi_group_spec_t _spec_parametric[] = {
  { DT_MASKS_STATE_UNION | DT_MASKS_STATE_WITHIN_MULTIPLY, N_("parametric") }
};

// bottom-to-top: parametric is the foundation, shapes sits on top of it and
// carries the between-group operator that joins the two
static const _flexi_group_spec_t _spec_drawn_parametric[] = {
  { DT_MASKS_STATE_UNION | DT_MASKS_STATE_WITHIN_MULTIPLY, N_("parametric") },
  { DT_MASKS_STATE_INTERSECTION | DT_MASKS_STATE_SCREEN, N_("shapes") }
};

static const _flexi_builtin_t _flexi_builtins[] = {
  { N_("basic"), NULL, _spec_basic, G_N_ELEMENTS(_spec_basic) },
  { N_("add + subtract + intersect"), NULL, _spec_ops3, G_N_ELEMENTS(_spec_ops3) },
  { N_("drawn mask"), N_("one group of shapes, combined with each other by screen"),
    _spec_drawn, G_N_ELEMENTS(_spec_drawn) },
  { N_("parametric"),
    N_("one group of parametric channels, combined with each other by multiply"),
    _spec_parametric, G_N_ELEMENTS(_spec_parametric) },
  { N_("drawn + parametric"),
    N_("shapes above parametric channels, the two intersected -- how classic drawn"
       " and parametric masks combined"),
    _spec_drawn_parametric, G_N_ELEMENTS(_spec_drawn_parametric) }
};

// materializes one built-in into the entry array the apply path takes. Caller
// frees. Group names are translated here rather than stored translated, since
// the built-ins are static and the entries are not.
static _flexi_group_entry_t *_flexi_builtin_entries(const _flexi_builtin_t *b)
{
  _flexi_group_entry_t *entries = calloc(b->n, sizeof(_flexi_group_entry_t));
  for(int i = 0; i < b->n; i++)
  {
    entries[i].state = b->groups[i].state;
    entries[i].opacity = 1.0f;
    if(b->groups[i].name)
      g_strlcpy(entries[i].name, _(b->groups[i].name), sizeof(entries[i].name));
  }
  return entries;
}

// appends one applicable preset row. Takes ownership of `entries`.
static GtkWidget *_flexi_preset_menu_item(GtkMenu *menu,
                                          dt_iop_module_t *module,
                                          const gchar *label,
                                          _flexi_group_entry_t *entries,
                                          const int n)
{
  GtkWidget *item = gtk_menu_item_new_with_label(label);
  g_object_set_data_full(G_OBJECT(item), "entries", entries, free);
  g_object_set_data(G_OBJECT(item), "n", GINT_TO_POINTER(n));
  g_signal_connect(G_OBJECT(item), "activate", G_CALLBACK(_flexi_preset_item_activate),
                   module);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), item);
  return item;
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
    _("preset name"), _("cancel"), _("save"));
  if(!name) return;
  gboolean reserved = FALSE;
  for(size_t i = 0; i < G_N_ELEMENTS(_flexi_builtins); i++)
    if(!strcmp(name, _(_flexi_builtins[i].name))) reserved = TRUE;
  if(!*name)
    dt_control_log(_("please give the preset a name"));
  else if(reserved)
    dt_control_log(_("`%s' is a reserved preset name, please pick another one"), name);
  else
  {
    int n = 0;
    _flexi_group_entry_t *entries = _flexi_layout_capture(module, &n);
    if(n > 0) _flexi_preset_save_to_db(name, entries, n);
    free(entries);
  }
  g_free(name);
}

// appends a "presets" section (group-layout presets) directly to `menu` --
// used by _blendif_options_callback, the "blend mask" header's hamburger
// (formerly its own separate hamburger on the "mask elements" header)
void _add_flexi_presets_menu(GtkMenu *menu, dt_iop_module_t *module)
{
  GtkWidget *header = gtk_menu_item_new_with_label(_("group layout presets"));
  gtk_widget_set_sensitive(header, FALSE);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), header);

  for(size_t i = 0; i < G_N_ELEMENTS(_flexi_builtins); i++)
  {
    const _flexi_builtin_t *b = &_flexi_builtins[i];
    GtkWidget *item = _flexi_preset_menu_item(menu, module, _(b->name),
                                              _flexi_builtin_entries(b), b->n);
    if(b->tooltip) gtk_widget_set_tooltip_text(item, _(b->tooltip));
  }

  GList *user_presets = _flexi_preset_list_load();
  if(user_presets)
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), gtk_separator_menu_item_new());
  for(GList *p = user_presets; p; p = g_list_next(p))
  {
    _flexi_preset_t *preset = p->data;
    _flexi_group_entry_t *entries_copy = malloc(preset->n * sizeof(_flexi_group_entry_t));
    memcpy(entries_copy, preset->entries, preset->n * sizeof(_flexi_group_entry_t));
    GtkWidget *item =
      _flexi_preset_menu_item(menu, module, preset->name, entries_copy, preset->n);
    gtk_widget_set_tooltip_text(item, _("click to apply, right-click to delete"));
    g_object_set_data_full(G_OBJECT(item), "preset-name", g_strdup(preset->name), g_free);
    g_signal_connect(G_OBJECT(item), "button-press-event",
                     G_CALLBACK(_flexi_preset_item_press), module);
  }
  _flexi_preset_list_free(user_presets);

  gtk_menu_shell_append(GTK_MENU_SHELL(menu), gtk_separator_menu_item_new());
  GtkWidget *save_item =
    gtk_menu_item_new_with_label(_("save current layout as preset..."));
  g_signal_connect(G_OBJECT(save_item), "activate",
                   G_CALLBACK(_flexi_preset_save_clicked), module);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), save_item);
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
