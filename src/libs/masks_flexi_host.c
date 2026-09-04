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

// Host lib for the flexi masks panel's relocatable content, "utility
// module" position only (masks_panel_position == MASKS_PANEL_POS_UTILITY,
// see develop/blend_gui.c). Docked at DT_UI_CONTAINER_PANEL_LEFT_CENTER
// like any ordinary lib (mask manager is the sibling precedent) -- always
// visible while registered, and collapsed through its own expander like any
// other lib (see expanded_state below, which ties on-canvas mask editing to
// that, as the other two positions do to their own collapse controls).
//
// The "separate panel, left/right" positions do NOT use this lib at all --
// those are a genuine extra grid column owned by src/gui/gtk.c
// (dt_ui_flexi_panel_*), since the user explicitly wants a real independent
// panel, not more content stacked inside the existing left/right panels.
// See _masks_flexi_relocate in develop/blend_gui.c for how the two
// mechanisms are picked between.
//
// This lib is deliberately kept dt_lib-visible (dt_lib_is_visible) at all
// times, for the same reason as before: src/views/view.c only calls
// dt_lib_gui_get_expander() (which builds and packs self->expander) for
// libs that are visible *at view-enter time* -- a lib hidden via
// dt_lib_set_visible() never gets its expander built/packed at all until
// the view is re-entered, which would make switching away from "embedded"
// live impossible. So "embedded" (this lib unused) is a plain
// gtk_widget_hide() of self->expander instead.

#include "control/conf.h"
#include "control/signal.h"
#include "develop/blend.h"
#include "develop/blend_gui_internal.h"
#include "develop/develop.h"
#include "dtgtk/expander.h"
#include "gui/gtk.h"
#include "libs/lib.h"
#include "libs/lib_api.h"

DT_MODULE(1)

typedef struct dt_lib_masks_flexi_host_t
{
  GtkBox *content_box;
  GtkBox *actions_box;
  GtkBox *toggle_box;
} dt_lib_masks_flexi_host_t;

static void _reconfigure(dt_lib_module_t *self);

const char *name(dt_lib_module_t *self)
{
  return _("blend mask");
}

const char *description(dt_lib_module_t *self)
{
  return _("blend mask panel for the focused module\n"
           "(see the panel options menu on the module\n"
           "whose masking is being edited)");
}

dt_view_type_flags_t views(dt_lib_module_t *self)
{
  return DT_VIEW_DARKROOM;
}

uint32_t container(dt_lib_module_t *self)
{
  return DT_UI_CONTAINER_PANEL_LEFT_CENTER;
}

int position(const dt_lib_module_t *self)
{
  // sorts right under navigation/histogram (LEFT_TOP), above the mask
  // manager lib's position of 10
  return 2;
}

GtkWidget *gui_tool_box(dt_lib_module_t *self)
{
  dt_lib_masks_flexi_host_t *d = (dt_lib_masks_flexi_host_t *)self->data;
  if(!d->actions_box)
  {
    d->actions_box = GTK_BOX(gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0));
    darktable.develop->proxy.masks_flexi_host.actions_box = d->actions_box;
  }
  return GTK_WIDGET(d->actions_box);
}

// shows/hides self->expander depending on whether masks_panel_position is
// currently "utility" -- see file comment for why this is a plain
// gtk_widget_hide/show rather than dt_lib_set_visible. Must only run once
// self->expander exists (see view_enter below); cheap to call repeatedly.
static void _reconfigure(dt_lib_module_t *self)
{
  if(!self->expander) return;

  const int pos = _masks_panel_position();
  gtk_widget_set_visible(self->expander, pos == MASKS_PANEL_POS_UTILITY);
}

// this lib's expander is the collapse control for the masking panel in the
// "utility module" position -- the counterpart of the grid panel's corner icon
// and of the embedded position's in-header arrow. Folding it hides the shapes
// list, so the on-canvas editing it drives goes with it (and comes back on
// expand), and a fold the user asked for is recorded as the panel's shared
// collapse preference, the same as in the other two positions. Both are
// masks_gui_panel_host.c's business, not this lib's; forward and let it decide.
void expanded_state(dt_lib_module_t *self, const gboolean expanded)
{
  if(!darktable.develop || !darktable.develop->proxy.masks_flexi_host.hosted_module)
  {
    if(expanded && self->expander)
      dt_lib_gui_set_expanded(self, FALSE);
    return;
  }
  dt_iop_gui_blend_masks_panel_host_expanded(expanded);
}

/* A mask row's warning badge can depend on state that lives outside the mask
 * list showing it: a raster element goes inert the moment its source module is
 * switched off, no longer carries a mask, or is removed. None of that touches
 * this module's own forms, so nothing in the panel's usual update paths fires
 * and the badge would keep saying whatever it said when the list was built.
 *
 * A history change is the signal every one of those arrives as (toggling a
 * module off adds a history item, as does deleting one), so re-evaluate the
 * badges then. The sweep is over the whole pipeline rather than the focused
 * module: the badge is a property of the *reader* of the raster mask, which is
 * a different module from the one the user just touched. Refreshing a module
 * with no mask list built returns immediately, so this costs a null check per
 * module.
 *
 * This lib is the hook's home because it is the panel's host and has a
 * darkroom-scoped lifecycle to hang connect/disconnect on -- the sweep itself
 * is not specific to the utility position and runs whatever position the panel
 * is in. */
static void _history_change_callback(gpointer instance, gpointer user_data)
{
  if(!darktable.develop) return;
  for(GList *m = darktable.develop->iop; m; m = g_list_next(m))
    dt_iop_gui_blend_refresh_mask_badges((dt_iop_module_t *)m->data);
}

void gui_init(dt_lib_module_t *self)
{
  dt_lib_masks_flexi_host_t *d = g_malloc0(sizeof(dt_lib_masks_flexi_host_t));
  self->data = (void *)d;

  d->content_box = GTK_BOX(dt_gui_vbox());
  gtk_widget_set_name(GTK_WIDGET(d->content_box), "masks-flexi-host-content");
  d->actions_box = GTK_BOX(gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0));
  d->toggle_box = GTK_BOX(gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0));
  gtk_widget_set_valign(GTK_WIDGET(d->actions_box), GTK_ALIGN_CENTER);
  gtk_widget_set_valign(GTK_WIDGET(d->toggle_box), GTK_ALIGN_CENTER);
  gtk_widget_show(GTK_WIDGET(d->actions_box));
  gtk_widget_show(GTK_WIDGET(d->toggle_box));
  self->widget = GTK_WIDGET(d->content_box);
  gtk_widget_show_all(self->widget);

  darktable.develop->proxy.masks_flexi_host.module = self;
  darktable.develop->proxy.masks_flexi_host.content_box = d->content_box;
  darktable.develop->proxy.masks_flexi_host.actions_box = d->actions_box;
  darktable.develop->proxy.masks_flexi_host.toggle_box = d->toggle_box;
  darktable.develop->proxy.masks_flexi_host.hosted_module = NULL;
  darktable.develop->proxy.masks_flexi_host.reconfigure = _reconfigure;

  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_HISTORY_CHANGE, _history_change_callback);

  // deliberately NOT calling dt_lib_set_visible(self, FALSE) here even
  // when the current position isn't utility -- see file comment.
  // self->expander doesn't exist yet at this point anyway (view.c builds
  // it after gui_init returns); the initial visual state is applied from
  // view_enter() instead.
}

void view_enter(dt_lib_module_t *self,
                struct dt_view_t *old_view,
                struct dt_view_t *new_view)
{
  // self->expander now exists (view.c just built it) -- apply the initial
  // visual state for the current masks_panel_position
  _reconfigure(self);

  dt_lib_masks_flexi_host_t *d = (dt_lib_masks_flexi_host_t *)self->data;
  if(self->expander && d && d->toggle_box && !gtk_widget_get_parent(GTK_WIDGET(d->toggle_box)))
  {
    GtkWidget *header = DTGTK_EXPANDER(self->expander)->header;
    dt_gui_add_class(header, "masks-flexi-host-header");
    gtk_box_pack_end(GTK_BOX(header), GTK_WIDGET(d->toggle_box), FALSE, FALSE, 0);
    // visual reading order from left to right: overlay | toggle
    // For GTK_PACK_END, the earlier child in the list is placed further to the right.
    // child 2 = toggle (rightmost), child 3 = overlay
    gtk_box_reorder_child(GTK_BOX(header), GTK_WIDGET(d->toggle_box), 2);
    if(d->actions_box)
      gtk_box_reorder_child(GTK_BOX(header), GTK_WIDGET(d->actions_box), 3);
    if(self->arrow)
      gtk_widget_set_valign(self->arrow, GTK_ALIGN_CENTER);
    gtk_widget_show(GTK_WIDGET(d->toggle_box));
  }

  if(self->expander)
  {
    GtkWidget *header = DTGTK_EXPANDER(self->expander)->header;
    dt_gui_add_class(header, "masks-flexi-host-header");
    if(self->arrow)
      gtk_widget_set_valign(self->arrow, GTK_ALIGN_CENTER);
    if(self->presets_button)
      gtk_widget_set_valign(self->presets_button, GTK_ALIGN_CENTER);
    GList *children = gtk_container_get_children(GTK_CONTAINER(header));
    for(GList *c = children; c; c = g_list_next(c))
    {
      if(GTK_IS_EVENT_BOX(c->data))
      {
        GtkWidget *child = gtk_bin_get_child(GTK_BIN(c->data));
        if(GTK_IS_LABEL(child))
        {
          darktable.develop->proxy.masks_flexi_host.label_evb = GTK_WIDGET(c->data);
          darktable.develop->proxy.masks_flexi_host.header_label = child;
          break;
        }
      }
    }
    g_list_free(children);
  }

  if(self->reset_button)
    gtk_widget_set_visible(self->reset_button, FALSE);

  // this lib has no presets or preferences of its own, so lib.c's default
  // header hamburger has nothing to open. It used to be repurposed to the
  // masking options menu, but no other panel position shows an icon for those
  // any more -- they open on a right-click of the mask on/off toggle, which is
  // in this header too (see _blendop_mask_enable_toggled). Hide it rather than
  // leave one position with a control the others dropped. no_show_all because
  // the expander header is shown with gtk_widget_show_all on every view enter.
  if(self->presets_button)
  {
    gtk_widget_set_no_show_all(self->presets_button, TRUE);
    gtk_widget_hide(self->presets_button);
  }

  dt_iop_module_t *module = darktable.develop ? darktable.develop->gui_module : NULL;
  if(module)
  {
    dt_iop_gui_blend_masks_panel_relocate(module);
  }
  else
  {
    if(self->expander)
      dt_lib_gui_set_expanded(self, FALSE);

    GtkWidget *lbl = darktable.develop->proxy.masks_flexi_host.header_label;
    GtkWidget *levb = darktable.develop->proxy.masks_flexi_host.label_evb;
    if(lbl && GTK_IS_LABEL(lbl))
    {
      gchar *markup = _model_masks_panel_header_markup(NULL, NULL, TRUE);
      gtk_label_set_markup(GTK_LABEL(lbl), markup);
      g_free(markup);
    }

    if(self->arrow)
    {
      gtk_widget_set_sensitive(self->arrow, FALSE);
      gtk_widget_set_tooltip_text(self->arrow, _("disabled because no module is selected"));
    }
    if(levb)
    {
      gtk_widget_set_sensitive(levb, FALSE);
      gtk_widget_set_tooltip_text(levb, _("disabled because no module is selected"));
    }
  }
}

void gui_cleanup(dt_lib_module_t *self)
{
  DT_CONTROL_SIGNAL_DISCONNECT(_history_change_callback, self);

  darktable.develop->proxy.masks_flexi_host.module = NULL;
  darktable.develop->proxy.masks_flexi_host.content_box = NULL;
  darktable.develop->proxy.masks_flexi_host.actions_box = NULL;
  darktable.develop->proxy.masks_flexi_host.toggle_box = NULL;
  darktable.develop->proxy.masks_flexi_host.header_label = NULL;
  darktable.develop->proxy.masks_flexi_host.label_evb = NULL;
  darktable.develop->proxy.masks_flexi_host.hosted_module = NULL;
  darktable.develop->proxy.masks_flexi_host.reconfigure = NULL;

  g_free(self->data);
  self->data = NULL;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
