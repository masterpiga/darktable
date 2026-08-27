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

// Where the flexi masks panel's content lives.
//
// bd->relocatable_box holds the whole blend/mask panel and can sit in one of
// three homes, per "plugins/darkroom/blend/masks_panel_position": embedded in
// the module's own expander (the default), inside the masks_flexi_host utility
// lib, or inside the separate grid panel owned by gui/gtk.c
// (dt_ui_flexi_panel_*). This file owns moving it between them -- the
// relocate/release pair, the host lib's re-configure poke, and the menu
// section the user picks a position from -- and nothing else. It builds no
// panel content of its own; blend_gui.c does that.
//
// Split out of blend_gui.c. The seam is small on purpose: it needs one helper
// from there (_reparent_into) and exports five entry points back, all declared
// in blend_gui_internal.h.

#include "develop/blend_gui_internal.h"

#include "common/darktable.h"
#include "control/conf.h"
#include "develop/develop.h"
#include "dtgtk/button.h"
#include "dtgtk/expander.h"
#include "gui/gtk.h"
#include "libs/lib.h"

// let the utility-mode host lib re-apply its live visibility (see
// _reconfigure in masks_flexi_host.c) -- only relevant for
// MASKS_PANEL_POS_UTILITY, a no-op otherwise (its expander just stays
// hidden)
static void _masks_flexi_host_reconfigure(void)
{
  dt_lib_module_t *host = darktable.develop->proxy.masks_flexi_host.module;
  if(host && darktable.develop->proxy.masks_flexi_host.reconfigure)
    darktable.develop->proxy.masks_flexi_host.reconfigure(host);
}

// human-readable mask type name for the collapsed corner icon's tooltip
static const char *_mask_mode_label(const uint32_t mask_mode)
{
  switch(mask_mode)
  {
  case DEVELOP_MASK_DISABLED: return _("no mask");
  case DEVELOP_MASK_ENABLED: return _("uniformly");
  case DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK: return _("drawn mask");
  case DEVELOP_MASK_ENABLED | DEVELOP_MASK_CONDITIONAL: return _("parametric mask");
  case DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK_CONDITIONAL:
    return _("drawn & parametric mask");
  case DEVELOP_MASK_ENABLED | DEVELOP_MASK_RASTER: return _("raster mask");
  case DEVELOP_MASK_ENABLED | DEVELOP_MASK_FLEXI: return _("flexi mask");
  default: return _("mask");
  }
}

void _flexi_inline_collapse_clicked(GtkWidget *w, gpointer user_data)
{
  dt_ui_flexi_panel_set_collapsed(darktable.gui->ui, TRUE, TRUE, TRUE);
}

// move bd->relocatable_box back into this module's own expander (the
// "embedded" home), clearing the host's bookkeeping if this module was the
// one occupying it. No-op if the box is already home. This only happens
// when the module stops being eligible to be hosted at all (lost focus,
// position switched away from utility/left/right, or module torn down) --
// having no mask does NOT release it anymore, see _masks_flexi_relocate.
// Whether the box is actually revealed once back home always follows real
// focus (darktable.develop->gui_module == module), for every position
// preference: an expanded-but-unfocused module must never show its full
// blend/mask panel inline, whether it fell back here because the position
// preference IS "embedded", or because it used to be hosted elsewhere
// (utility/left/right) and just lost focus -- both are the same case from
// the user's point of view (BUG, previously only the first was covered).
void _masks_flexi_release(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd || !bd->relocatable_box) return;
  const gboolean show = darktable.develop->gui_module == module;

  const gboolean was_hosted =
    darktable.develop->proxy.masks_flexi_host.hosted_module == module;
  if(was_hosted) darktable.develop->proxy.masks_flexi_host.hosted_module = NULL;

  if(bd->masks_right_cluster)
  {
    if(gtk_widget_get_parent(bd->suppress) != bd->masks_right_cluster)
      _reparent_into(bd->suppress, bd->masks_right_cluster, FALSE, FALSE);
    if(gtk_widget_get_parent(bd->showmask) != bd->masks_right_cluster)
      _reparent_into(bd->showmask, bd->masks_right_cluster, FALSE, FALSE);
  }
  gtk_widget_set_visible(bd->masks_blend_header, TRUE);

  _reparent_into(GTK_WIDGET(bd->relocatable_box), bd->iopw, FALSE, FALSE);
  gtk_widget_set_visible(GTK_WIDGET(bd->relocatable_box), show);
  gtk_widget_set_visible(bd->flexi_inline_collapse_btn, FALSE);
  gtk_widget_set_visible(bd->masks_options_btn, TRUE);
  // back in the module's own content -- restore the embedded inset (see
  // darktable.css's "#blending-tabs.blending-tabs-embedded")
  dt_gui_add_class(bd->masks_blend_header, "blending-tabs-embedded");

  if(was_hosted)
  {
    _masks_flexi_host_reconfigure();
    // dev->gui_module is already updated to the new focus target (or NULL)
    // by the time this runs -- see dt_iop_gui_set_focus in imageop.c, which
    // sets it before calling lose_focus on the outgoing module
    if(darktable.develop->gui_module)
    {
      // another module is about to (or already did) take over hosting --
      // just re-apply whatever visibility the panel already had, not a new
      // user choice, so don't persist it
      dt_ui_flexi_panel_set_collapsed(darktable.gui->ui,
                                      dt_ui_flexi_panel_is_collapsed(darktable.gui->ui),
                                      FALSE, FALSE);
    }
    else
    {
      // nothing is focused anymore: there's no content left to host, so
      // hide the panel and its corner icon entirely instead of leaving an
      // empty panel visible
      dt_ui_flexi_panel_set_collapsed(darktable.gui->ui, TRUE, FALSE, FALSE);
    }
  }
}

// (re)decide where this module's masking panel content should live, per
// the current "plugins/darkroom/blend/masks_panel_position" preference:
// embedded (default) keeps it inline; utility uses the masks_flexi_host lib
// (LEFT_CENTER); left/right use the genuine extra grid panel owned by
// gui/gtk.c (dt_ui_flexi_panel_*). Hosting only depends on the module being
// focused and masking-capable -- NOT on the current mask mode, so the
// mode-select row stays reachable via the panel/corner-icon even with the
// mask off (see request: "overlay button ... should show" with no mask).
// With no mask the panel auto-collapses to just that icon; the collapse is
// visual only (persist=FALSE) so it doesn't clobber the user's own
// expand/collapse preference for when a mask *is* active.
void _masks_flexi_relocate(dt_iop_module_t *module)
{
  if(!module || !module->blend_data) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd->relocatable_box) return;

  const int pos = dt_conf_get_int("plugins/darkroom/blend/masks_panel_position");
  const uint32_t mask_mode = module->blend_params->mask_mode;
  const gboolean is_focused = darktable.develop->gui_module == module;
  const gboolean want_hosted = pos != MASKS_PANEL_POS_EMBEDDED && is_focused;

  GtkWidget *target = NULL;
  if(want_hosted && pos == MASKS_PANEL_POS_UTILITY)
  {
    dt_lib_module_t *host = darktable.develop->proxy.masks_flexi_host.module;
    GtkBox *content_box = darktable.develop->proxy.masks_flexi_host.content_box;
    if(host && content_box) target = GTK_WIDGET(content_box);
  }
  else if(want_hosted) // LEFT / RIGHT
  {
    target = dt_ui_flexi_panel_content(darktable.gui->ui);
    dt_ui_flexi_panel_set_side(darktable.gui->ui, pos == MASKS_PANEL_POS_RIGHT);
  }

  if(!target)
  {
    // an expanded-but-unfocused module must not show its full blend/mask
    // panel inline, whatever the position preference (see
    // _masks_flexi_release, which gates visibility on real focus)
    _masks_flexi_release(module);
    return;
  }

  dt_iop_module_t *prev = darktable.develop->proxy.masks_flexi_host.hosted_module;
  if(prev && prev != module) _masks_flexi_release(prev);

  darktable.develop->proxy.masks_flexi_host.hosted_module = module;
  _reparent_into(GTK_WIDGET(bd->relocatable_box), target, FALSE, FALSE);
  // being hosted implies focused (see want_hosted above); make sure the box
  // is visible in case an earlier embedded-and-unfocused state left it hidden.
  gtk_widget_show(GTK_WIDGET(bd->relocatable_box));
  _masks_flexi_host_reconfigure();

  if(pos == MASKS_PANEL_POS_UTILITY)
  {
    GtkBox *actions_box = darktable.develop->proxy.masks_flexi_host.actions_box;
    if(actions_box)
    {
      _reparent_into(bd->suppress, GTK_WIDGET(actions_box), FALSE, FALSE);
      _reparent_into(bd->showmask, GTK_WIDGET(actions_box), FALSE, FALSE);
    }
    gtk_widget_set_visible(bd->masks_blend_header, FALSE);
  }
  else
  {
    if(bd->masks_right_cluster)
    {
      if(gtk_widget_get_parent(bd->suppress) != bd->masks_right_cluster)
        _reparent_into(bd->suppress, bd->masks_right_cluster, FALSE, FALSE);
      if(gtk_widget_get_parent(bd->showmask) != bd->masks_right_cluster)
        _reparent_into(bd->showmask, bd->masks_right_cluster, FALSE, FALSE);
    }
    gtk_widget_set_visible(bd->masks_blend_header, TRUE);
    // hosted elsewhere now -- drop the embedded inset, the host already
    // provides its own (see darktable.css's "#blending-tabs.blending-tabs-embedded")
    dt_gui_remove_class(bd->masks_blend_header, "blending-tabs-embedded");
  }

  // in the utility lib, that lib's own header hamburger is repurposed to
  // this same options menu (see masks_flexi_host.c's view_enter and
  // dt_iop_gui_blend_masks_options_popup) -- don't show a second, redundant
  // one in the mode-select row too
  gtk_widget_set_visible(bd->masks_options_btn, pos != MASKS_PANEL_POS_UTILITY);

  if(pos == MASKS_PANEL_POS_LEFT || pos == MASKS_PANEL_POS_RIGHT)
  {
    dt_ui_flexi_panel_set_icon(darktable.gui->ui, mask_mode != DEVELOP_MASK_DISABLED,
                               _mask_mode_label(mask_mode));
    // with no mask there's nothing to show but the mode picker, so collapse
    // to the corner icon; otherwise apply the user's stored preference --
    // neither is a fresh user choice, so persist=FALSE either way
    const gboolean want_collapsed =
      mask_mode == DEVELOP_MASK_DISABLED
        ? TRUE
        : dt_conf_get_bool("plugins/darkroom/blend/masks_panel_collapsed");
    dt_ui_flexi_panel_set_collapsed(darktable.gui->ui, want_collapsed, TRUE, FALSE);

    // arrow points the direction the panel collapses toward (its docked side)
    dtgtk_button_set_paint(
      DTGTK_BUTTON(bd->flexi_inline_collapse_btn), dtgtk_cairo_paint_solid_arrow,
      pos == MASKS_PANEL_POS_RIGHT ? CPF_DIRECTION_RIGHT : CPF_DIRECTION_LEFT, NULL);
    gtk_widget_set_visible(bd->flexi_inline_collapse_btn, TRUE);
  }
  else
  {
    gtk_widget_set_visible(bd->flexi_inline_collapse_btn, FALSE);
  }
}

// ---- position preference ---------------------------------------------------

static void _masks_panel_position_activate(GtkCheckMenuItem *mi, dt_iop_module_t *module)
{
  // these are plain check items (not radio items) also fire "toggled" for
  // the item being deactivated -- only act on the one becoming active.
  // Plain gtk_check_menu_item, not gtk_radio_menu_item: this theme has no
  // visible styling for GTK's "radio" indicator CSS node (only "check" is
  // styled, see darktable.css), so a radio group's active item silently
  // showed no checkmark at all. Mutual exclusion is enforced manually below.
  if(!gtk_check_menu_item_get_active(mi)) return;

  GtkWidget *submenu = gtk_widget_get_parent(GTK_WIDGET(mi));
  GList *siblings = gtk_container_get_children(GTK_CONTAINER(submenu));
  for(GList *l = siblings; l; l = g_list_next(l))
    if(l->data != mi && GTK_IS_CHECK_MENU_ITEM(l->data))
      gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(l->data), FALSE);
  g_list_free(siblings);

  const int pos = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(mi), "dt-panel-pos"));
  dt_conf_set_int("plugins/darkroom/blend/masks_panel_position", pos);

  // update the utility-mode host lib's own visibility for the new position
  _masks_flexi_host_reconfigure();

  // leaving the separate-panel (left/right) mechanism entirely: force it
  // fully hidden (not just emptied) rather than leaving an empty panel
  // visible -- _masks_flexi_relocate()'s own release path only re-applies
  // whatever visibility it already had, which isn't enough here
  if(pos != MASKS_PANEL_POS_LEFT && pos != MASKS_PANEL_POS_RIGHT)
    dt_ui_flexi_panel_set_collapsed(darktable.gui->ui, TRUE, FALSE, FALSE);

  // decide where this (focused) module's content should live now
  _masks_flexi_relocate(module);

  // repositioning is a deliberate user action -- make sure the result is
  // actually visible, regardless of collapse state or whether there's a
  // mask to show right now
  switch(pos)
  {
  case MASKS_PANEL_POS_LEFT:
  case MASKS_PANEL_POS_RIGHT:
    // force-expand and persist: unlike _masks_flexi_relocate's normal
    // "no mask -> auto-collapse to the corner icon" behavior, explicitly
    // picking a separate panel should show it open every time
    dt_ui_flexi_panel_set_collapsed(darktable.gui->ui, FALSE, TRUE, TRUE);
    break;
  case MASKS_PANEL_POS_UTILITY:
  {
    dt_lib_module_t *host = darktable.develop->proxy.masks_flexi_host.module;
    if(host && host->expander)
      dtgtk_expander_set_expanded(DTGTK_EXPANDER(host->expander), TRUE);
    break;
  }
  case MASKS_PANEL_POS_EMBEDDED:
  default:
    // scrolls the already-expanded, focused module's own panel into
    // view -- dtgtk_expander_set_expanded(..., TRUE) re-triggers the
    // scroll-to-view animation even when already expanded (see its
    // "Quick Access Panel" comment in dtgtk/expander.c)
    if(module->expander)
      dtgtk_expander_set_expanded(DTGTK_EXPANDER(module->expander), TRUE);
    break;
  }
}

// appends a "blend mask panel position" section directly to `menu` -- inline check
// items, not a submenu, so the choice is visible at a glance
void _add_masks_panel_position_menu(GtkMenu *menu, dt_iop_module_t *module)
{
  GtkWidget *header = gtk_menu_item_new_with_label(_("blend mask panel position"));
  gtk_widget_set_sensitive(header, FALSE);
  gtk_widget_set_tooltip_text(
    header, _("where the flexi masks panel content (groups, elements, refinements)"
              " is shown. the on/off toggle and hamburger above always stay here.\n"
              "moving to/from the utility module or a separate panel takes effect"
              " the next time the panel is rebuilt (e.g. after reopening darkroom)."));
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), header);

  static const struct
  {
    int pos;
    const char *label;
  } items[] = {
    { MASKS_PANEL_POS_EMBEDDED, N_("embedded within each module (default)") },
    { MASKS_PANEL_POS_UTILITY, N_("utility module, left panel") },
    { MASKS_PANEL_POS_LEFT, N_("separate panel, left") },
    { MASKS_PANEL_POS_RIGHT, N_("separate panel, right") },
  };

  const int cur_pos = dt_conf_get_int("plugins/darkroom/blend/masks_panel_position");
  for(size_t i = 0; i < G_N_ELEMENTS(items); i++)
  {
    GtkWidget *ci = gtk_check_menu_item_new_with_label(_(items[i].label));
    dt_gui_add_class(ci, "dt_transparent_background");
    if(items[i].pos == cur_pos)
      gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(ci), TRUE);
    g_object_set_data(G_OBJECT(ci), "dt-panel-pos", GINT_TO_POINTER(items[i].pos));
    g_signal_connect(G_OBJECT(ci), "toggled", G_CALLBACK(_masks_panel_position_activate),
                     module);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), ci);
  }
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
