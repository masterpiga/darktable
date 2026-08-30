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

// One collapse state shared by all three positions ("the mask panel is folded
// away"), so it carries over when the panel is moved between them, and so all
// three draw the same line: a fold the *user* asked for is a preference and is
// stored here, a fold the panel does to itself (no mask to show) is visual
// only and never touches this key.
//
// The utility position needs the flag below to keep that line, because its
// collapse goes through dt_lib_gui_set_expanded(), the generic lib API, which
// persists to the lib's own "plugins/<view>/masks_flexi_host/expanded" key
// unconditionally -- it has no persist=FALSE the way the separate panel's
// dt_ui_flexi_panel_set_collapsed() does. So this file drives that expander
// itself from the shared preference (see _masks_utility_apply_collapsed and
// _masks_flexi_relocate), which makes the lib's own key a mirror rather than a
// source of truth in this position.
static gboolean _masks_panel_collapsed_pref(void)
{
  return dt_conf_get_bool("plugins/darkroom/blend/masks_panel_collapsed");
}

void _masks_panel_set_collapsed_pref(const gboolean collapsed)
{
  dt_conf_set_bool("plugins/darkroom/blend/masks_panel_collapsed", collapsed);
}

// set while we drive the utility lib's expander ourselves, so the
// expanded_state callback it fires back is not mistaken for the user folding
// the panel by hand. The utility position's counterpart of the persist
// argument the other two positions' collapse calls take.
static gboolean _driving_host_expander = FALSE;

static void _masks_utility_apply_collapsed(dt_lib_module_t *host,
                                           const gboolean collapsed)
{
  _driving_host_expander = TRUE;
  dt_lib_gui_set_expanded(host, !collapsed);
  _driving_host_expander = FALSE;
}

// apply the embedded position's collapse state to `module`'s panel: fold the
// body away below the header (or bring it back), and point the header arrow
// the way the next click will take it -- DOWN when open, RIGHT when folded,
// as everywhere else in the UI.
static void _masks_embedded_apply_collapsed(dt_iop_module_t *module,
                                            const gboolean collapsed)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd || !bd->masks_panel_body || !bd->flexi_inline_collapse_btn) return;

  gtk_widget_set_visible(GTK_WIDGET(bd->masks_panel_body), !collapsed);
  dtgtk_button_set_paint(DTGTK_BUTTON(bd->flexi_inline_collapse_btn),
                         dtgtk_cairo_paint_solid_arrow,
                         collapsed ? CPF_DIRECTION_RIGHT : CPF_DIRECTION_DOWN, NULL);
  gtk_widget_set_tooltip_text(bd->flexi_inline_collapse_btn,
                              collapsed
                                ? _("show the blend mask panel")
                                : _("hide the blend mask panel"));
}

// The header's fixed reading order is
//
//   expander | title | toggle | <space> | actions | hamburger
//
// which is the utility lib's own header layout (that lib supplies the first
// three from its expander, its label and the toggle lent to it), adopted by
// every position so the panel reads the same wherever it lives.
//
// The single exception, applied here: docked in the separate *right* panel, the
// expander and the hamburger trade ends, so the arrow sits against the edge it
// points at and folds toward. They are the only two header widgets that ever
// move -- the toggle and the eyes stay put in both docks, being reached by
// muscle memory and meaning the same thing whichever side the panel is on.
//
// Within masks_right_cluster every child is packed END, so the *first* child in
// the list is the rightmost one -- hence reorder to 0 for "outermost right".
static void _masks_header_apply_side(dt_iop_gui_blend_data_t *bd,
                                     const gboolean mirrored)
{
  GtkWidget *pin = bd->flexi_inline_collapse_btn;
  GtkWidget *burger = bd->masks_options_btn;
  if(!pin || !burger || !bd->masks_blend_header || !bd->masks_right_cluster) return;

  // the expander's slot is the header's own first child, ahead of the title --
  // not inside masks_left_cluster, which is just the toggle's home box
  GtkWidget *leading = mirrored ? burger : pin;
  GtkWidget *trailing = mirrored ? pin : burger;

  _reparent_into(leading, bd->masks_blend_header, FALSE, FALSE);
  gtk_box_set_child_packing(GTK_BOX(bd->masks_blend_header), leading,
                            FALSE, FALSE, 0, GTK_PACK_START);
  gtk_box_reorder_child(GTK_BOX(bd->masks_blend_header), leading, 0);

  _reparent_into(trailing, bd->masks_right_cluster, TRUE, FALSE);
  gtk_box_set_child_packing(GTK_BOX(bd->masks_right_cluster), trailing,
                            FALSE, FALSE, 0, GTK_PACK_END);
  gtk_box_reorder_child(GTK_BOX(bd->masks_right_cluster), trailing, 0);
}

void _flexi_inline_collapse_clicked(GtkWidget *w, gpointer user_data)
{
  dt_iop_module_t *module = (dt_iop_module_t *)user_data;
  const int pos = dt_conf_get_int("plugins/darkroom/blend/masks_panel_position");

  if(pos == MASKS_PANEL_POS_LEFT || pos == MASKS_PANEL_POS_RIGHT)
  {
    // while the panel is only being peeked at there is nothing to collapse --
    // it is not open, it is being looked at, and it would fold by itself the
    // moment the pointer left. So the arrow pins it open instead: this is the
    // one control in reach that can turn a look into a state.
    const gboolean peeking = dt_ui_flexi_panel_is_peeking(darktable.gui->ui);
    // ...and otherwise it folds the panel away to the canvas corner icon,
    // which is then the way back -- one-way, unlike the embedded arrow below.
    // Either way dt_ui_flexi_panel_set_collapsed handles the edit-mode stash.
    dt_ui_flexi_panel_set_collapsed(darktable.gui->ui, !peeking, TRUE, TRUE);
    return;
  }

  // embedded: the header this arrow sits in stays put, so it toggles both ways.
  // What it toggles is what is actually on screen, not the stored state: if the
  // two ever disagree, deriving the click from the stored one "folds" an
  // already-folded panel and the user has to click twice to open it.
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  const gboolean collapsed =
    bd && bd->masks_panel_body && gtk_widget_get_visible(GTK_WIDGET(bd->masks_panel_body));
  _masks_panel_set_collapsed_pref(collapsed);
  _masks_embedded_apply_collapsed(module, collapsed);
  dt_iop_gui_blend_masks_panel_collapsed(collapsed);
}

// While the panel is only being peeked at, its collapse arrow pins it open
// instead of folding it (see _flexi_inline_collapse_clicked) -- so make it look
// like what it now does, rather than leaving a collapse arrow that collapses
// nothing. Called from gtk.c as a peek starts and ends.
void dt_iop_gui_blend_masks_panel_set_peek(const gboolean peeking)
{
  if(!darktable.develop) return;
  dt_iop_module_t *module = darktable.develop->proxy.masks_flexi_host.hosted_module;
  if(!module) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd || !bd->flexi_inline_collapse_btn) return;

  const int pos = dt_conf_get_int("plugins/darkroom/blend/masks_panel_position");
  if(pos != MASKS_PANEL_POS_LEFT && pos != MASKS_PANEL_POS_RIGHT) return;

  if(peeking)
  {
    dtgtk_button_set_paint(DTGTK_BUTTON(bd->flexi_inline_collapse_btn),
                           dtgtk_cairo_paint_pin, 0, NULL);
    gtk_widget_set_tooltip_text(bd->flexi_inline_collapse_btn,
                                _("keep this panel open"));
  }
  else
  {
    dtgtk_button_set_paint(
      DTGTK_BUTTON(bd->flexi_inline_collapse_btn), dtgtk_cairo_paint_solid_arrow,
      pos == MASKS_PANEL_POS_RIGHT ? CPF_DIRECTION_RIGHT : CPF_DIRECTION_LEFT, NULL);
    gtk_widget_set_tooltip_text(bd->flexi_inline_collapse_btn,
                                _("collapse this panel; click the icon it leaves behind\n"
                                  "on the canvas to bring it back"));
  }
}

// the utility lib's expander was toggled -- by the user from its header, or by
// _masks_utility_apply_collapsed just above. Only the former is a preference:
// it goes into the shared key, exactly as the separate panel's collapse button
// and the embedded arrow do for their positions, so the panel folds and
// unfolds the same way wherever it lives.
void dt_iop_gui_blend_masks_panel_host_expanded(const gboolean expanded)
{
  if(dt_conf_get_int("plugins/darkroom/blend/masks_panel_position")
     != MASKS_PANEL_POS_UTILITY)
    return;

  if(!_driving_host_expander) _masks_panel_set_collapsed_pref(!expanded);
  dt_iop_gui_blend_masks_panel_collapsed(!expanded);
}

// The masking panel just folded away, or came back -- see the header comment
// on this function in blend.h for the contract; this is called from all three
// positions' own collapse mechanisms.
//
// The point: "edit on canvas" and the shape-add tools are driven from this
// panel, so leaving them armed once it is gone strands a live editing overlay
// on canvas with no visible control over it. Turning them off is not enough
// on its own, though -- collapsing the panel to get a clear look at the image
// and then re-opening it should not silently cost the user their editing
// mode, so stash it and put it back.
void dt_iop_gui_blend_masks_panel_collapsed(const gboolean collapsed)
{
  if(!darktable.develop) return;
  // whatever is hosted in a flexi panel; embedded, the focused module (only
  // the focused module's panel is on screen there -- see _masks_flexi_release)
  dt_iop_module_t *module = darktable.develop->proxy.masks_flexi_host.hosted_module;
  if(!module) module = darktable.develop->gui_module;
  if(!module) return;

  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd || !bd->masks_support || !bd->masks_inited) return;

  if(collapsed)
  {
    // an armed shape-add tool is just as unusable with the panel gone; it has
    // no state worth restoring, unlike the edit mode below
    for(int n = 0; n < DEVELOP_MASKS_NB_SHAPES; n++)
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_shapes[n]), FALSE);

    // unconditionally, so a collapse with editing already off also clears any
    // stale stash rather than leaving an older one to be restored later
    bd->masks_shown_stash = bd->masks_shown;
    if(bd->masks_shown == DT_MASKS_EDIT_OFF) return;
    // this untoggles bd->masks_edit itself, and drops any solo-edit
    dt_masks_set_edit_mode(module, DT_MASKS_EDIT_OFF);
  }
  else
  {
    const dt_masks_edit_mode_t stash = bd->masks_shown_stash;
    bd->masks_shown_stash = DT_MASKS_EDIT_OFF;
    // nothing was interrupted, or the user turned editing back on by hand
    // while the panel was away (via a shortcut) -- don't second-guess either
    if(stash == DT_MASKS_EDIT_OFF || bd->masks_shown != DT_MASKS_EDIT_OFF) return;

    // don't restore an overlay onto a mask that lost its shapes in the
    // meantime (module reset, history jump, undo): same guard the "edit on
    // canvas" toggle itself applies before entering edit mode
    dt_masks_form_t *grp =
      dt_masks_get_from_id(darktable.develop, module->blend_params->mask_id);
    if(!grp || !(grp->type & DT_MASKS_GROUP) || !grp->points) return;

    dt_masks_set_edit_mode(module, stash);
  }
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
  }
  if(bd->masks_left_cluster)
  {
    if(gtk_widget_get_parent(bd->mask_enable_toggle) != bd->masks_left_cluster)
      _reparent_into(bd->mask_enable_toggle, bd->masks_left_cluster, FALSE, FALSE);
  }
  gtk_widget_set_visible(bd->masks_blend_header, TRUE);

  _reparent_into(GTK_WIDGET(bd->relocatable_box), bd->iopw, FALSE, FALSE);
  gtk_widget_set_visible(GTK_WIDGET(bd->relocatable_box), show);
  // back in the module's own expander. When that is where the panel actually
  // lives (embedded), the in-header arrow keeps working, now folding the panel
  // body away in place; when the box only landed here because this module lost
  // focus, its real home is a host and the arrow has nothing to act on.
  const gboolean embedded =
    dt_conf_get_int("plugins/darkroom/blend/masks_panel_position") == MASKS_PANEL_POS_EMBEDDED;
  gtk_widget_set_visible(bd->flexi_inline_collapse_btn, embedded);
  // the right-dock mirroring is that dock's alone -- back home, the header
  // reads left-to-right like every other module's
  _masks_header_apply_side(bd, FALSE);
  _masks_embedded_apply_collapsed(module, embedded && _masks_panel_collapsed_pref());
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
  // hosted: the host itself collapses (grid panel to its corner icon, utility
  // lib to its expander header), so the body is never folded here -- undo any
  // embedded fold the box is carrying over
  if(bd->masks_panel_body)
    gtk_widget_set_visible(GTK_WIDGET(bd->masks_panel_body), TRUE);
  // being hosted implies focused (see want_hosted above); make sure the box
  // is visible in case an earlier embedded-and-unfocused state left it hidden.
  gtk_widget_show(GTK_WIDGET(bd->relocatable_box));
  _masks_flexi_host_reconfigure();

  if(pos == MASKS_PANEL_POS_UTILITY)
  {
    GtkBox *toggle_box = darktable.develop->proxy.masks_flexi_host.toggle_box;
    if(toggle_box)
      _reparent_into(bd->mask_enable_toggle, GTK_WIDGET(toggle_box), FALSE, FALSE);
    GtkBox *actions_box = darktable.develop->proxy.masks_flexi_host.actions_box;
    if(actions_box)
    {
      _reparent_into(bd->suppress, GTK_WIDGET(actions_box), FALSE, FALSE);
    }
    gtk_widget_set_visible(bd->masks_blend_header, FALSE);

    dt_lib_module_t *host = darktable.develop->proxy.masks_flexi_host.module;
    // the shared state, like the other two positions. Previously this derived
    // expansion from mask_mode alone, so a relocate (a focus change, a mode
    // change) re-expanded a lib the user had just folded -- and, since
    // dt_lib_gui_set_expanded persists, overwrote the folded state as it went.
    if(host) _masks_utility_apply_collapsed(host, _masks_panel_collapsed_pref());
  }
  else
  {
    if(bd->masks_right_cluster)
    {
      if(gtk_widget_get_parent(bd->suppress) != bd->masks_right_cluster)
        _reparent_into(bd->suppress, bd->masks_right_cluster, FALSE, FALSE);
    }
    if(bd->masks_left_cluster)
    {
      if(gtk_widget_get_parent(bd->mask_enable_toggle) != bd->masks_left_cluster)
        _reparent_into(bd->mask_enable_toggle, bd->masks_left_cluster, FALSE, FALSE);
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
    // the shared state again -- applying it, not deciding it, so persist=FALSE
    dt_ui_flexi_panel_set_collapsed(darktable.gui->ui, _masks_panel_collapsed_pref(),
                                    TRUE, FALSE);

    // arrow points the direction the panel collapses toward (its docked side)
    dtgtk_button_set_paint(
      DTGTK_BUTTON(bd->flexi_inline_collapse_btn), dtgtk_cairo_paint_solid_arrow,
      pos == MASKS_PANEL_POS_RIGHT ? CPF_DIRECTION_RIGHT : CPF_DIRECTION_LEFT, NULL);
    gtk_widget_set_tooltip_text(bd->flexi_inline_collapse_btn,
                                _("collapse this panel; click the icon it leaves behind\n"
                                  "on the canvas to bring it back"));
    gtk_widget_set_visible(bd->flexi_inline_collapse_btn, TRUE);
    _masks_header_apply_side(bd, pos == MASKS_PANEL_POS_RIGHT);
  }
  else
  {
    gtk_widget_set_visible(bd->flexi_inline_collapse_btn, FALSE);
    _masks_header_apply_side(bd, FALSE);
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

  // repositioning is a deliberate user action -- make sure the result is
  // actually visible, in every position: unfold the panel and store that,
  // before the relocate below applies it. Overriding _masks_flexi_relocate's
  // "no mask -> fold to the corner icon / the collapsed header" is the point:
  // explicitly picking a position should show what was picked.
  _masks_panel_set_collapsed_pref(FALSE);

  // decide where this (focused) module's content should live now
  _masks_flexi_relocate(module);

  switch(pos)
  {
  case MASKS_PANEL_POS_LEFT:
  case MASKS_PANEL_POS_RIGHT:
    // relocate folds the panel away when there is no mask; force it open
    dt_ui_flexi_panel_set_collapsed(darktable.gui->ui, FALSE, TRUE, TRUE);
    break;
  case MASKS_PANEL_POS_UTILITY:
  {
    dt_lib_module_t *host = darktable.develop->proxy.masks_flexi_host.module;
    if(host)
      _masks_utility_apply_collapsed(host, FALSE);
    break;
  }
  case MASKS_PANEL_POS_EMBEDDED:
  default:
    // ...and the same override of the no-mask fold as the two hosted cases
    _masks_embedded_apply_collapsed(module, FALSE);
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
              " is shown.\n"
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
