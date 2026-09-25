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
// A "layout" is the skeleton of a flexi mask: the mask's own group and the
// groups nested in it, each with its within-group operator, name and opacity
// -- nothing else: no shapes, no channel or raster elements. Captured and
// applied as an array of _flexi_layout_node_t in pre-order, the mask's own
// group first and every group's nested groups bottom-up after it, each naming
// its holder by index.
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
// v1-v3 stored a flat list of groups joined by between-group operators, which
// masks no longer have (masks_revamp_nested_groups.md, Q8): those are not
// listed any more. v4 stores the tree.
#define FLEXI_GROUP_PRESET_VERSION 4

// one group of a layout. Written to the database verbatim, as one blob of
// these, so every field is fixed-size
typedef struct _flexi_layout_node_t
{
  // within-group operator bits (DT_MASKS_STATE_WITHIN)
  dt_masks_state_t within;
  float opacity;
  // same width as dt_masks_point_group_t.name, the group's marker's name.
  // Unused for the mask's own group, which is always "whole mask"
  char name[128];
  // index of the holding group's node, -1 for the mask's own group
  int32_t parent;
} _flexi_layout_node_t;

// the groups nested in `list`'s groups, as nodes under `parent`, bottom-up
static void _flexi_layout_capture_nested(GArray *out,
                                         const dt_masks_form_t *list,
                                         const int32_t parent,
                                         const int depth)
{
  if(!list || depth > DT_MASKS_NESTING_MAX) return;
  for(const GList *l = list->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    if(dt_masks_point_is_marker(pt)) continue;
    const dt_masks_form_t *sub = dt_masks_get_from_id(darktable.develop, pt->formid);
    if(!sub || sub == list || !(sub->type & DT_MASKS_GROUP)) continue;
    // a nested group is one group: its marker heads its list
    const dt_masks_point_group_t *marker = sub->points ? sub->points->data : NULL;
    if(!marker || !dt_masks_point_is_marker(marker)) continue;
    _flexi_layout_node_t node = { .within = marker->state & DT_MASKS_STATE_WITHIN,
                                  .opacity = marker->group_opacity,
                                  .parent = parent };
    g_strlcpy(node.name, marker->name, sizeof(node.name));
    g_array_append_val(out, node);
    _flexi_layout_capture_nested(out, sub, (int32_t)out->len - 1, depth + 1);
  }
}

// the module's group skeleton, in pre-order (see _flexi_layout_node_t).
// Caller frees the returned array
static _flexi_layout_node_t *_flexi_layout_capture(dt_iop_module_t *module, int *n_out)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  GArray *out = g_array_new(FALSE, FALSE, sizeof(_flexi_layout_node_t));

  // a mask with no group form yet has the one group the panel shows for it
  const dt_masks_point_group_t *root =
    grp && grp->points && dt_masks_point_is_marker(grp->points->data) ? grp->points->data
                                                                       : NULL;
  const _flexi_layout_node_t node = {
    .within = root ? root->state & DT_MASKS_STATE_WITHIN : 0,
    .opacity = root ? root->group_opacity : 1.0f,
    .parent = -1
  };
  g_array_append_val(out, node);
  _flexi_layout_capture_nested(out, grp, 0, 1);

  *n_out = out->len;
  return (_flexi_layout_node_t *)g_array_free(out, FALSE);
}

// replaces the module's whole mask -- elements and groups alike -- with the
// empty groups of the layout `nodes`. Never asks for confirmation itself;
// callers that might be discarding elements confirm first (see
// _flexi_preset_apply_confirmed).
static void
_flexi_layout_apply(dt_iop_module_t *module, const _flexi_layout_node_t *nodes, int n)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  _masks_reset_mask_core(module);
  dt_mask_id_t root_cid = INVALID_MASKID;
  dt_masks_form_t *grp = _module_flexi_group(module, &root_cid);
  if(!grp || n <= 0 || nodes[0].parent != -1) return;
  // the reset left the mask's own group, empty: the layout's first node
  dt_masks_point_group_t *root = grp->points ? grp->points->data : NULL;
  if(!root || !dt_masks_point_is_marker(root)) return;
  root->state = (root->state & ~DT_MASKS_STATE_WITHIN)
                | (nodes[0].within & DT_MASKS_STATE_WITHIN);
  root->group_opacity = nodes[0].opacity;

  dt_mask_id_t *cids = g_new(dt_mask_id_t, n);
  cids[0] = root->formid;
  dt_mask_id_t first_nested = INVALID_MASKID;
  for(int i = 1; i < n; i++)
  {
    cids[i] = INVALID_MASKID;
    // pre-order: a holder always comes before what it holds
    const int32_t parent = nodes[i].parent;
    if(parent < 0 || parent >= i || !dt_is_valid_maskid(cids[parent])) continue;
    cids[i] = _model_nest_new_group(grp, nodes[i].within, cids[parent]);
    dt_masks_point_group_t *marker = _group_point(grp, cids[i]);
    if(!marker) continue;
    marker->group_opacity = nodes[i].opacity;
    g_strlcpy(marker->name, nodes[i].name, sizeof(marker->name));
    if(!dt_is_valid_maskid(first_nested)) first_nested = cids[i];
  }
  g_free(cids);

  // start where the next element most likely goes: the bottom nested group,
  // or the mask's own group when the layout nests none
  bd->panel_selected_formid = INVALID_MASKID;
  bd->panel_selected_group_cid =
    dt_is_valid_maskid(first_nested) ? first_nested : root->formid;
  dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
  _build_masks_list(module);
  _refresh_canvas_edit(module);
}

// does `list` hold, at any depth, an element an applied layout would discard?
// Its nested groups themselves are not: the layout replaces those anyway
static gboolean _flexi_list_has_content(const dt_masks_form_t *list, const int depth)
{
  if(!list || depth > DT_MASKS_NESTING_MAX) return FALSE;
  for(const GList *l = list->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    if(dt_masks_point_is_marker(pt)) continue;
    const dt_masks_form_t *f = dt_masks_get_from_id(darktable.develop, pt->formid);
    if(!f || f == list || !(f->type & DT_MASKS_GROUP)) return TRUE;
    if(_flexi_list_has_content(f, depth + 1)) return TRUE;
  }
  return FALSE;
}

static gboolean _flexi_layout_has_content(dt_iop_module_t *module)
{
  return _flexi_list_has_content(_module_mask_group(module), 0);
}

// reads back every user-saved layout preset's name + node array. Caller frees
// with _flexi_preset_list_free.
typedef struct _flexi_preset_t
{
  gchar *name;
  _flexi_layout_node_t *nodes;
  int n;
} _flexi_preset_t;

static GList *_flexi_preset_list_load(void)
{
  GList *out = NULL;
  sqlite3_stmt *stmt;
  // only the tree format: older presets describe a mask shape masks no longer
  // have, and stay in the database unlisted
  DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                              "SELECT name, op_params FROM data.presets"
                              " WHERE operation = ?1 AND writeprotect = 0"
                              "   AND op_version = ?2"
                              " ORDER BY name",
                              -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, FLEXI_GROUP_PRESET_OP, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, FLEXI_GROUP_PRESET_VERSION);
  while(sqlite3_step(stmt) == SQLITE_ROW)
  {
    const int n = sqlite3_column_bytes(stmt, 1) / (int)sizeof(_flexi_layout_node_t);
    if(n <= 0) continue;
    _flexi_preset_t *p = malloc(sizeof(_flexi_preset_t));
    p->name = g_strdup((const gchar *)sqlite3_column_text(stmt, 0));
    p->nodes = malloc(n * sizeof(_flexi_layout_node_t));
    memcpy(p->nodes, sqlite3_column_blob(stmt, 1), n * sizeof(_flexi_layout_node_t));
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
  free(p->nodes);
  free(p);
}

static void _flexi_preset_list_free(GList *presets)
{
  g_list_free_full(presets, _flexi_preset_free);
}

static void
_flexi_preset_save_to_db(const gchar *name, const _flexi_layout_node_t *nodes, int n)
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
  DT_DEBUG_SQLITE3_BIND_BLOB(stmt, 4, nodes, (int)(n * sizeof(_flexi_layout_node_t)),
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
                                          const _flexi_layout_node_t *nodes,
                                          int n)
{
  if(_flexi_layout_has_content(module)
     && !dt_gui_show_yes_no_dialog(
       _("apply mask layout preset?"), "",
       _("this replaces the group layout and removes every element "
         "currently in this mask. continue?")))
    return;
  _flexi_layout_apply(module, nodes, n);
}


// ---- built-in layouts ------------------------------------------------------
// Listed above the user's own presets, and their names are reserved so a user
// preset cannot shadow one. One for each classic mask type, as the mask it
// migrates to: classic combined its drawn shapes by union, multiplied its
// parametric channels together, and multiplied the two results.
typedef struct _flexi_group_spec_t
{
  dt_masks_state_t within; // within-group operator
  const char *name;        // untranslated group name, NULL to leave it unnamed
  int parent;              // holder's index, -1 for the mask's own group
} _flexi_group_spec_t;

typedef struct _flexi_builtin_t
{
  const char *name;                  // untranslated preset name
  const char *tooltip;               // untranslated
  const _flexi_group_spec_t *groups; // pre-order, as _flexi_layout_node_t
  int n;
} _flexi_builtin_t;

static const _flexi_group_spec_t _spec_drawn[] = {
  { 0, NULL, -1 }
};

static const _flexi_group_spec_t _spec_parametric[] = {
  { DT_MASKS_STATE_WITHIN_MULTIPLY, NULL, -1 }
};

// the shapes are the base the channels multiply into, as migration builds it
static const _flexi_group_spec_t _spec_drawn_parametric[] = {
  { DT_MASKS_STATE_WITHIN_MULTIPLY, NULL, -1 },
  { 0, N_("shapes"), 0 },
  { DT_MASKS_STATE_WITHIN_MULTIPLY, N_("parametric"), 0 }
};

static const _flexi_builtin_t _flexi_builtins[] = {
  { N_("drawn (classic)"),
    N_("one group combining its shapes by union, as a classic drawn mask did"),
    _spec_drawn, G_N_ELEMENTS(_spec_drawn) },
  { N_("parametric (classic)"),
    N_("one group multiplying its parametric channels together, as a classic\n"
       "parametric mask did"),
    _spec_parametric, G_N_ELEMENTS(_spec_parametric) },
  { N_("drawn + parametric (classic)"),
    N_("a \"shapes\" group (union) and a \"parametric\" group (multiply),\n"
       "multiplied together, as a classic drawn & parametric mask did"),
    _spec_drawn_parametric, G_N_ELEMENTS(_spec_drawn_parametric) }
};

// materializes one built-in into the node array the apply path takes. Caller
// frees. Group names are translated here rather than stored translated, since
// the built-ins are static and the nodes are not.
static _flexi_layout_node_t *_flexi_builtin_nodes(const _flexi_builtin_t *b)
{
  _flexi_layout_node_t *nodes = calloc(b->n, sizeof(_flexi_layout_node_t));
  for(int i = 0; i < b->n; i++)
  {
    nodes[i].within = b->groups[i].within;
    nodes[i].opacity = 1.0f;
    nodes[i].parent = b->groups[i].parent;
    if(b->groups[i].name)
      g_strlcpy(nodes[i].name, _(b->groups[i].name), sizeof(nodes[i].name));
  }
  return nodes;
}

static void _flexi_preset_save_clicked(dt_iop_module_t *module)
{
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
    _flexi_layout_node_t *nodes = _flexi_layout_capture(module, &n);
    if(n > 0) _flexi_preset_save_to_db(name, nodes, n);
    free(nodes);
  }
  g_free(name);
}

static void _flexi_preset_builtin_action(GSimpleAction *action,
                                         GVariant *parameter,
                                         gpointer user_data)
{
  dt_iop_module_t *module = (dt_iop_module_t *)user_data;
  const int idx = g_variant_get_int32(parameter);
  if(darktable.gui->active_popover_menu)
    gtk_popover_popdown(GTK_POPOVER(darktable.gui->active_popover_menu));
  if(idx >= 0 && idx < (int)G_N_ELEMENTS(_flexi_builtins))
  {
    const _flexi_builtin_t *b = &_flexi_builtins[idx];
    _flexi_layout_node_t *nodes = _flexi_builtin_nodes(b);
    _flexi_preset_apply_confirmed(module, nodes, b->n);
    free(nodes);
  }
}

static void _flexi_preset_user_action(GSimpleAction *action,
                                      GVariant *parameter,
                                      gpointer user_data)
{
  dt_iop_module_t *module = (dt_iop_module_t *)user_data;
  const gchar *name = g_variant_get_string(parameter, NULL);
  if(darktable.gui->active_popover_menu)
    gtk_popover_popdown(GTK_POPOVER(darktable.gui->active_popover_menu));
  if(name)
  {
    GList *user_presets = _flexi_preset_list_load();
    for(GList *p = user_presets; p; p = g_list_next(p))
    {
      _flexi_preset_t *preset = p->data;
      if(!g_strcmp0(preset->name, name))
      {
        _flexi_preset_apply_confirmed(module, preset->nodes, preset->n);
        break;
      }
    }
    _flexi_preset_list_free(user_presets);
  }
}

static void _flexi_preset_delete_action(GSimpleAction *action,
                                        GVariant *parameter,
                                        gpointer user_data)
{
  const gchar *name = g_variant_get_string(parameter, NULL);
  if(darktable.gui->active_popover_menu)
    gtk_popover_popdown(GTK_POPOVER(darktable.gui->active_popover_menu));
  if(name && dt_gui_show_yes_no_dialog(_("delete preset?"), "",
                                       _("do you really want to delete the mask layout "
                                         "preset `%s'?"),
                                       name))
  {
    _flexi_preset_delete_from_db(name);
  }
}

static void _flexi_preset_save_action(GSimpleAction *action,
                                      GVariant *parameter,
                                      gpointer user_data)
{
  dt_iop_module_t *module = (dt_iop_module_t *)user_data;
  if(darktable.gui->active_popover_menu)
    gtk_popover_popdown(GTK_POPOVER(darktable.gui->active_popover_menu));
  _flexi_preset_save_clicked(module);
}

// appends a "presets" section (group-layout presets) directly to `menu` --
// the right-click menu of the "add group" button (see _new_shape_op_press)
void _add_flexi_presets_menu(GMenu *menu, GtkWidget *anchor, dt_iop_module_t *module)
{
  GActionGroup *action_group = gtk_widget_get_action_group(anchor, "masks_presets");
  if(action_group == NULL)
  {
    GActionEntry action_entries[] =
    {
      { "builtin", _flexi_preset_builtin_action, "i", NULL },
      { "user",    _flexi_preset_user_action,    "s", NULL },
      { "delete",  _flexi_preset_delete_action,  "s", NULL },
      { "save",    _flexi_preset_save_action,    NULL, NULL },
    };
    action_group = G_ACTION_GROUP(g_simple_action_group_new());
    g_action_map_add_action_entries(G_ACTION_MAP(action_group), action_entries,
                                    G_N_ELEMENTS(action_entries), module);
    gtk_widget_insert_action_group(anchor, "masks_presets", action_group);
  }

  GMenu *sec_builtins = g_menu_new();
  for(size_t i = 0; i < G_N_ELEMENTS(_flexi_builtins); i++)
  {
    const _flexi_builtin_t *b = &_flexi_builtins[i];
    GMenuItem *item = g_menu_item_new(_(b->name), NULL);
    g_menu_item_set_action_and_target_value(item, "masks_presets.builtin", g_variant_new_int32(i));
    g_menu_item_set_attribute(item, "tooltip", "s", _(b->tooltip));
    g_menu_append_item(sec_builtins, item);
    g_object_unref(item);
  }
  g_menu_append_section(menu, _("group layout presets"), G_MENU_MODEL(sec_builtins));
  g_object_unref(sec_builtins);

  GList *user_presets = _flexi_preset_list_load();
  if(user_presets)
  {
    GMenu *sec_user = g_menu_new();
    GMenu *sub_delete = g_menu_new();
    for(GList *p = user_presets; p; p = g_list_next(p))
    {
      _flexi_preset_t *preset = p->data;
      GMenuItem *item = g_menu_item_new(preset->name, NULL);
      g_menu_item_set_action_and_target_value(item, "masks_presets.user", g_variant_new_string(preset->name));
      g_menu_append_item(sec_user, item);
      g_object_unref(item);

      GMenuItem *del = g_menu_item_new(preset->name, NULL);
      g_menu_item_set_action_and_target_value(del, "masks_presets.delete", g_variant_new_string(preset->name));
      g_menu_append_item(sub_delete, del);
      g_object_unref(del);
    }
    g_menu_append_submenu(sec_user, _("delete preset"), G_MENU_MODEL(sub_delete));
    g_object_unref(sub_delete);

    g_menu_append_section(menu, NULL, G_MENU_MODEL(sec_user));
    g_object_unref(sec_user);
    _flexi_preset_list_free(user_presets);
  }

  GMenu *sec_save = g_menu_new();
  g_menu_append(sec_save, _("save current layout as preset..."), "masks_presets.save");
  g_menu_append_section(menu, NULL, G_MENU_MODEL(sec_save));
  g_object_unref(sec_save);
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
