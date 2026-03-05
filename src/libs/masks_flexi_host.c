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
// visible while registered, no collapse behaviour.
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
#include "develop/blend.h"
#include "develop/develop.h"
#include "gui/gtk.h"
#include "libs/lib.h"
#include "libs/lib_api.h"

DT_MODULE(1)

typedef struct dt_lib_masks_flexi_host_t
{
  GtkBox *content_box;
} dt_lib_masks_flexi_host_t;

static void _reconfigure(dt_lib_module_t *self);

const char *name(dt_lib_module_t *self)
{
  return _("mask editor");
}

const char *description(dt_lib_module_t *self)
{
  return _("relocated content of the flexi masks panel\n"
           "(see the panel's hamburger menu on the module\n"
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

// shows/hides self->expander depending on whether masks_panel_position is
// currently "utility" -- see file comment for why this is a plain
// gtk_widget_hide/show rather than dt_lib_set_visible. Must only run once
// self->expander exists (see view_enter below); cheap to call repeatedly.
static void _reconfigure(dt_lib_module_t *self)
{
  if(!self->expander) return;

  const int pos = dt_conf_get_int("plugins/darkroom/blend/masks_panel_position");
  gtk_widget_set_visible(self->expander, pos == MASKS_PANEL_POS_UTILITY);
}

void gui_init(dt_lib_module_t *self)
{
  dt_lib_masks_flexi_host_t *d = g_malloc0(sizeof(dt_lib_masks_flexi_host_t));
  self->data = (void *)d;

  d->content_box = GTK_BOX(dt_gui_vbox());
  self->widget = GTK_WIDGET(d->content_box);
  gtk_widget_show_all(self->widget);

  darktable.develop->proxy.masks_flexi_host.module = self;
  darktable.develop->proxy.masks_flexi_host.content_box = d->content_box;
  darktable.develop->proxy.masks_flexi_host.hosted_module = NULL;
  darktable.develop->proxy.masks_flexi_host.reconfigure = _reconfigure;

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

  // this lib has no presets/preferences of its own, so lib.c's default
  // header hamburger would otherwise sit there permanently disabled and
  // useless -- repurpose it instead to the same "masking options" menu the
  // mode-select row's own hamburger opens elsewhere (see
  // dt_iop_gui_blend_masks_options_popup and bd->masks_options_btn, which
  // is hidden in this position so there's only ever one such button)
  if(self->presets_button)
  {
    // view_enter can run more than once (leaving/re-entering darkroom) --
    // both disconnects are no-ops on repeat calls, so this stays idempotent
    // rather than stacking duplicate "clicked" handlers
    g_signal_handlers_disconnect_matched(self->presets_button, G_SIGNAL_MATCH_DATA,
                                         0, 0, NULL, NULL, self);
    g_signal_handlers_disconnect_by_func(self->presets_button,
                                         G_CALLBACK(dt_iop_gui_blend_masks_options_popup), NULL);
    g_signal_connect(G_OBJECT(self->presets_button), "clicked",
                     G_CALLBACK(dt_iop_gui_blend_masks_options_popup), NULL);
    gtk_widget_set_sensitive(self->presets_button, TRUE);
    gtk_widget_set_tooltip_text(self->presets_button, _("masking options"));
  }
}

void gui_cleanup(dt_lib_module_t *self)
{
  darktable.develop->proxy.masks_flexi_host.module = NULL;
  darktable.develop->proxy.masks_flexi_host.content_box = NULL;
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
