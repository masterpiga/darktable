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
#include "dtgtk/togglebutton.h"
#include "gui/gtk.h"
#include "libs/lib.h"

// The one place the position preference is read, so that the two retired
// "separate panel, left/right" values are migrated in exactly one place rather
// than understood in fifteen. Their side survives the migration as the panel's
// current side, which is now a separate setting the panel updates itself
// whenever the user pins it somewhere (see _masks_panel_set_side_right).
int _masks_panel_position(void)
{
  const int pos = dt_conf_get_int("plugins/darkroom/blend/masks_panel_position");
  if(pos != MASKS_PANEL_POS_LEGACY_LEFT && pos != MASKS_PANEL_POS_LEGACY_RIGHT)
    return pos;

  dt_conf_set_bool("plugins/darkroom/blend/masks_panel_side_right",
                   pos == MASKS_PANEL_POS_LEGACY_RIGHT);
  dt_conf_set_int("plugins/darkroom/blend/masks_panel_position",
                  MASKS_PANEL_POS_CANVAS);
  return MASKS_PANEL_POS_CANVAS;
}

// which edge the panel is docked against. Not part of the position choice: a
// the panel opens on whichever edge was clicked, and stays there.
gboolean _masks_panel_side_right(void)
{
  return dt_conf_get_bool("plugins/darkroom/blend/masks_panel_side_right");
}

void _masks_panel_set_side_right(const gboolean right)
{
  dt_conf_set_bool("plugins/darkroom/blend/masks_panel_side_right", right);
}

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

void dt_iop_gui_blend_masks_panel_toggle(void)
{
  dt_iop_module_t *module = dt_dev_gui_module();
  if(!module || !module->blend_data) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd->masks_support || !bd->masks_inited) return;
  // the click handler ignores its widget argument and dispatches on the
  // position itself, utility included
  _flexi_inline_collapse_clicked(NULL, module);
}

void dt_iop_gui_blend_masks_panel_sync_toolbox(void)
{
  GtkWidget *btn = darktable.develop ? darktable.develop->masks_panel_button : NULL;
  if(!btn || !GTK_IS_TOGGLE_BUTTON(btn)) return;

  const dt_iop_module_t *module = dt_dev_gui_module();
  const dt_iop_gui_blend_data_t *bd = module ? module->blend_data : NULL;
  const gboolean usable = bd && bd->masks_support && bd->masks_inited;

  // deliberately not gtk_widget_set_sensitive: an insensitive widget receives no
  // motion events, so its tooltip never shows -- and "why can I not use this?"
  // is exactly what has to be explained here. The button is left sensitive and
  // inert instead: dt_iop_gui_blend_masks_panel_toggle refuses without a masking
  // module, and _masks_panel_quickbutton_clicked re-syncs from the panel's real
  // state afterwards, so a click on it cannot leave the toggle showing a change
  // that did not happen. The unavailable look is carried by the same dimming the
  // "no mask" state uses, since both are reached with mask_active FALSE.
  gtk_widget_set_sensitive(btn, TRUE);

  // Four states, on two independent channels, because the button answers two
  // separate questions and the user needs both at a glance:
  //
  //   does the module have a mask?  -> the icon itself: filled and at full
  //                                    strength when it does, outline-only and
  //                                    dimmed when it does not. The fill is the
  //                                    icon's own designed meaning (see
  //                                    dtgtk_cairo_paint_masks_panel), carried
  //                                    on CPF_SPECIAL_FLAG because the toggle
  //                                    overwrites CPF_ACTIVE with its checked
  //                                    state
  //   is the panel showing?         -> the button's box: a highlighted
  //                                    background with a border when it is,
  //                                    nothing at all when it is not
  //
  // so the icon answers "is there a mask" and the box around it answers "is the
  // panel on screen". The dimming must not be dt_dimmed: that class restores
  // full opacity on :checked, which would tie the two answers back together
  // whenever the panel is out.
  const gboolean mask_active =
    usable && module->blend_params
    && module->blend_params->mask_mode != DEVELOP_MASK_DISABLED;
  const gboolean showing = usable && !_masks_panel_collapsed_pref();

  dtgtk_togglebutton_set_paint(DTGTK_TOGGLEBUTTON(btn), dtgtk_cairo_paint_masks_panel,
                               mask_active ? CPF_SPECIAL_FLAG : CPF_NONE, NULL);

  if(mask_active) dt_gui_remove_class(btn, "flexi-toolbar-masks-off");
  else dt_gui_add_class(btn, "flexi-toolbar-masks-off");

  // the shared fold preference rather than each position's own widget: it is
  // what all three positions already agree on, so this reads the same however
  // the panel is hosted.
  ++darktable.gui->reset;
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(btn), showing);
  --darktable.gui->reset;

  // the panel always belongs to one module, so with no module focused there is
  // nothing for it to show: say that, rather than leaving a button that looks
  // unavailable for no stated reason
  if(!usable)
    gtk_widget_set_tooltip_text(
      btn,
      module
        ? _("unavailable: the focused module does not support masks")
        : _("unavailable: the blend mask panel shows the mask of the focused"
            " module, and no module is focused.\n"
            "click a module's header to focus it."));
  else
  {
    gchar *tt = g_strdup_printf(
      _("%s the blend mask panel of the focused module\nmask: %s"),
      showing ? _("hide") : _("show"), mask_active ? _("on") : _("off"));
    gtk_widget_set_tooltip_text(btn, tt);
    g_free(tt);
  }

  gtk_widget_queue_draw(btn);
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

// The header's reading order is:
//
// Embedded (left):
//   expander | caption | <space> | overlay | toggle
//
// Embedded (mirrored):
//   caption | <space> | overlay | toggle | expander
//
// Utility and canvas positions:
//   caption | <space> | overlay | toggle
//
// with two controls on the right in every case: the mask overlay, and the mask
// on/off toggle at the very end. The old preferences button is gone from all of
// them -- the blending options open on a right-click of the on/off toggle now
// (see _blendop_mask_enable_right_click), the way guide settings hang off the
// guides icon. The expander is embedded-only: the utility position already has
// the lib's own expander to its left, and out on the canvas the panel is opened
// and closed from the darkroom toolbar button, which is a bigger and more
// findable target than an icon on a floating panel.
static void _masks_header_apply_side(dt_iop_gui_blend_data_t *bd,
                                     const gboolean mirrored)
{
  GtkWidget *pin = bd->flexi_inline_collapse_btn;
  GtkWidget *pref = bd->masks_options_btn;
  GtkWidget *toggle = bd->mask_enable_toggle;
  GtkWidget *showmask = bd->showmask;
  if(!pin || !pref || !toggle || !showmask || !bd->masks_blend_header || !bd->masks_right_cluster) return;
  if(!GTK_IS_WIDGET(pin) || !GTK_IS_WIDGET(pref) || !GTK_IS_WIDGET(toggle) || !GTK_IS_WIDGET(showmask)
     || !GTK_IS_BOX(bd->masks_blend_header) || !GTK_IS_BOX(bd->masks_right_cluster)) return;

  _reparent_into(showmask, bd->masks_right_cluster, FALSE, FALSE);
  _reparent_into(toggle, bd->masks_right_cluster, FALSE, FALSE);

  gtk_box_reorder_child(GTK_BOX(bd->masks_right_cluster), showmask, 0);
  gtk_box_reorder_child(GTK_BOX(bd->masks_right_cluster), toggle, 1);

  // the expander belongs to the embedded position alone
  const gboolean show_pin = _masks_panel_position() == MASKS_PANEL_POS_EMBEDDED;

  if(mirrored)
  {
    // separate panel right: no icon left of caption; expand/collapse arrow is at the far right
    dt_gui_remove_class(pin, "flexi-pin-left");
    dt_gui_add_class(pin, "flexi-pin-right");
    _reparent_into(pin, bd->masks_right_cluster, FALSE, FALSE);
    gtk_box_reorder_child(GTK_BOX(bd->masks_right_cluster), pin, 2);
  }
  else
  {
    // standard / embedded / separate panel left: expander arrow is ahead of caption
    dt_gui_remove_class(pin, "flexi-pin-right");
    dt_gui_add_class(pin, "flexi-pin-left");
    _reparent_into(pin, bd->masks_blend_header, FALSE, FALSE);
    gtk_box_reorder_child(GTK_BOX(bd->masks_blend_header), pin, 0);
  }

  const gboolean is_mask_enabled = (bd->module->blend_params->mask_mode != DEVELOP_MASK_DISABLED);
  gtk_widget_set_visible(pin, show_pin);
  gtk_widget_set_visible(showmask, is_mask_enabled && !bd->module->hide_enable_button);
  gtk_widget_hide(pref);
  gtk_widget_show(toggle);
}

static gboolean _scroll_widget_into_view_idle(gpointer user_data)
{
  GtkWidget *widget = GTK_WIDGET(user_data);
  if(!widget || !GTK_IS_WIDGET(widget) || !gtk_widget_get_realized(widget))
    return G_SOURCE_REMOVE;

  GtkWidget *sw = gtk_widget_get_ancestor(widget, GTK_TYPE_SCROLLED_WINDOW);
  if(!sw) return G_SOURCE_REMOVE;

  GtkAdjustment *adj = gtk_scrolled_window_get_vadjustment(GTK_SCROLLED_WINDOW(sw));
  if(!adj) return G_SOURCE_REMOVE;

  GtkWidget *child = gtk_bin_get_child(GTK_BIN(sw));
  if(GTK_IS_VIEWPORT(child))
    child = gtk_bin_get_child(GTK_BIN(child));
  if(!child) return G_SOURCE_REMOVE;

  gint wx = 0, wy = 0;
  if(!gtk_widget_translate_coordinates(widget, child, 0, 0, &wx, &wy))
    return G_SOURCE_REMOVE;

  gint total_height = gtk_widget_get_allocated_height(widget);
  GtkWidget *extra = g_object_get_data(G_OBJECT(widget), "scroll-extra-child");
  if(extra && GTK_IS_WIDGET(extra) && gtk_widget_get_visible(extra))
    total_height += gtk_widget_get_allocated_height(extra);

  const gdouble cur_val = gtk_adjustment_get_value(adj);
  const gdouble page_size = gtk_adjustment_get_page_size(adj);
  const gdouble lower = gtk_adjustment_get_lower(adj);
  const gdouble upper = gtk_adjustment_get_upper(adj);

  gdouble target_val = cur_val;

  if(wy < cur_val)
    target_val = wy;
  else if(wy + total_height > cur_val + page_size)
  {
    if(total_height <= page_size)
      target_val = wy + total_height - page_size;
    else
      target_val = wy;
  }

  target_val = CLAMP(target_val, lower, MAX(lower, upper - page_size));
  if(target_val != cur_val)
    gtk_adjustment_set_value(adj, target_val);

  return G_SOURCE_REMOVE;
}

void _flexi_inline_collapse_clicked(GtkWidget *w, gpointer user_data)
{
  dt_iop_module_t *module = (dt_iop_module_t *)user_data;
  const int pos = _masks_panel_position();

  if(pos == MASKS_PANEL_POS_CANVAS)
  {
    const gboolean collapsed = dt_ui_flexi_panel_is_collapsed(darktable.gui->ui);
    if(module && collapsed)
    {
      // opening onto an inert "off" editor the user would then have to turn on
      // separately is not what they asked for
      if(_model_masks_pin_should_enable_mask(module->blend_params->mask_mode))
        dt_iop_gui_blend_mask_enable(module);

      if(_model_masks_pin_should_expand_iop(module->expanded, collapsed))
      {
        const gboolean collapse_others = dt_conf_get_bool("darkroom/ui/single_module");
        dt_iop_gui_set_expanded(module, TRUE, collapse_others);
      }
    }

    dt_ui_flexi_panel_set_collapsed(darktable.gui->ui, !collapsed, TRUE, TRUE);
    return;
  }

  if(pos == MASKS_PANEL_POS_UTILITY)
  {
    dt_lib_module_t *host = darktable.develop->proxy.masks_flexi_host.module;
    if(host)
    {
      const gboolean exp =
        host->expander && dtgtk_expander_get_expanded(DTGTK_EXPANDER(host->expander));
      if(!exp && module && _model_masks_pin_should_enable_mask(module->blend_params->mask_mode))
        dt_iop_gui_blend_mask_enable(module);
      dt_lib_gui_set_expanded(host, !exp);
      if(!exp && host->expander)
        g_idle_add(_scroll_widget_into_view_idle, host->expander);
    }
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
  dt_iop_gui_blend_masks_panel_sync_toolbox();
  if(!collapsed && bd && bd->masks_blend_header)
  {
    g_object_set_data(G_OBJECT(bd->masks_blend_header), "scroll-extra-child",
                      bd->masks_panel_body);
    g_idle_add(_scroll_widget_into_view_idle, bd->masks_blend_header);
  }
}

// the utility lib's expander was toggled by the user, which is what its own
// collapse control means there -- the counterpart of what the canvas position's
// edge strip and the embedded arrow do for their positions, so the panel folds
// and unfolds the same way wherever it lives.
void dt_iop_gui_blend_masks_panel_host_expanded(const gboolean expanded)
{
  if(_masks_panel_position() != MASKS_PANEL_POS_UTILITY) return;

  if(!_driving_host_expander) _masks_panel_set_collapsed_pref(!expanded);
  dt_iop_gui_blend_masks_panel_collapsed(!expanded);
  dt_iop_gui_blend_masks_panel_sync_toolbox();
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
void dt_iop_gui_blend_masks_panel_relocate(dt_iop_module_t *module)
{
  _masks_flexi_relocate(module);
}

static void _masks_flexi_release_full(dt_iop_module_t *module, const gboolean handoff);

// does any of this module's panel content currently sit in a host? Asked of
// the widget tree, so it stays true even when hosted_module has lost track.
static gboolean _widgets_are_hosted(dt_iop_gui_blend_data_t *bd)
{
  if(!bd) return FALSE;
  GtkWidget *hosts[3] = { dt_ui_flexi_panel_header(darktable.gui->ui),
                          dt_ui_flexi_panel_content(darktable.gui->ui),
                          GTK_WIDGET(darktable.develop->proxy.masks_flexi_host.content_box) };
  // the three widgets a host can end up owning: header and body separately
  // (LEFT/RIGHT), or the whole relocatable box at once (UTILITY)
  GtkWidget *owned[3] = { bd->masks_blend_header, GTK_WIDGET(bd->masks_panel_body),
                          GTK_WIDGET(bd->relocatable_box) };

  for(int h = 0; h < 3; h++)
    for(int w = 0; w < 3; w++)
      if(hosts[h] && owned[w] && gtk_widget_get_parent(owned[w]) == hosts[h]) return TRUE;
  return FALSE;
}

// Hand the panel back unconditionally, whatever this module's focus or
// position says -- unlike relocate above, which would re-host a module that is
// still focused. dt_iop_gui_cleanup_module calls this immediately before it
// destroys the module's widget tree, and the ordering is the whole point: while
// the panel is hosted, masks_blend_header and masks_panel_body are children of
// the host, NOT of the module's expander, so destroying the expander leaves
// them behind. dt_iop_gui_cleanup_blending runs after that destroy and cannot
// undo it -- by then relocatable_box is freed and the release it would do is
// skipped. Every darkroom leave then stranded one header/body pair in the
// panel, with their signal handlers still bound to the module struct that is
// freed moments later: hovering one of those rows called _row_crossing with a
// dangling dt_iop_module_t and segfaulted.
void dt_iop_gui_blend_masks_panel_release(dt_iop_module_t *module)
{
  if(!module || !module->blend_data) return;
  // only when there is something to hand back. This runs for every module on
  // teardown, most of which never hosted anything, and release touches header
  // widgets that a module without an inited masks GUI does not have.
  if(darktable.develop->proxy.masks_flexi_host.hosted_module != module
     && !_widgets_are_hosted(module->blend_data))
    return;
  _masks_flexi_release_full(module, FALSE);
}

dt_masks_panel_state_t _model_masks_panel_state(const int pos,
                                                const gboolean is_focused,
                                                const gboolean has_masking,
                                                const gboolean is_expanded,
                                                const gboolean mask_active,
                                                const gboolean panel_pref_collapsed)
{
  dt_masks_panel_state_t s;
  s.want_hosted = (pos == MASKS_PANEL_POS_CANVAS
                   || pos == MASKS_PANEL_POS_UTILITY) && is_focused && has_masking;

  if(pos == MASKS_PANEL_POS_CANVAS)
  {
    if(!is_focused || !has_masking)
    {
      s.panel_collapsed = TRUE;
      s.corner_icon_visible = FALSE;
      s.corner_icon_active = FALSE;
    }
    else if(!is_expanded)
    {
      // module is collapsed: hide the separate panel so its controls aren't
      // stranded alone on screen, but keep the corner icon visible so the user
      // can still see mask status and click the canvas edge to show it
      s.panel_collapsed = TRUE;
      s.corner_icon_visible = TRUE;
      s.corner_icon_active = mask_active;
    }
    else
    {
      // module is expanded: follow the user's preference
      s.panel_collapsed = panel_pref_collapsed;
      s.corner_icon_visible = panel_pref_collapsed;
      s.corner_icon_active = mask_active;
    }
  }
  else if(pos == MASKS_PANEL_POS_UTILITY && is_focused && has_masking)
  {
    s.panel_collapsed = panel_pref_collapsed;
    s.corner_icon_visible = FALSE;
    s.corner_icon_active = mask_active;
  }
  else if(pos == MASKS_PANEL_POS_EMBEDDED && is_focused && has_masking)
  {
    s.panel_collapsed = !is_expanded || panel_pref_collapsed;
    s.corner_icon_visible = FALSE;
    s.corner_icon_active = mask_active;
  }
  else
  {
    s.panel_collapsed = TRUE;
    s.corner_icon_visible = FALSE;
    s.corner_icon_active = FALSE;
  }
  return s;
}

gboolean _model_masks_pin_should_expand_iop(const gboolean is_expanded,
                                            const gboolean is_collapsed)
{
  return !is_expanded && is_collapsed;
}

gboolean _model_masks_pin_should_enable_mask(const uint32_t mask_mode)
{
  return mask_mode == DEVELOP_MASK_DISABLED;
}

char *_model_masks_corner_icon_tooltip(const char *module_name,
                                       const char *instance_name,
                                       const gboolean is_active,
                                       const char *mask_label)
{
  const char *mname = module_name ? module_name : _("blend mask");
  gchar *mod_name = (instance_name && strlen(instance_name) > 0)
    ? g_strdup_printf("%s (%s)", mname, instance_name)
    : g_strdup(mname);

  gchar *tooltip = is_active
    ? g_strdup_printf(_("%s: blend mask - %s\nclick to expand"),
        mod_name, mask_label ? mask_label : _("active"))
    : g_strdup_printf(_("%s: blend mask - off\nclick to enable mask and pin"),
        mod_name);
  g_free(mod_name);
  return tooltip;
}

char *_model_masks_panel_header_markup(const char *module_name,
                                       const char *instance_name,
                                       const gboolean is_hosted)
{
  if(!is_hosted)
  {
    return g_strdup(_("blend mask"));
  }

  const char *mname = module_name ? module_name : "";
  gchar *esc_mname = g_markup_escape_text(mname, -1);
  gchar *esc_iname = (instance_name && strlen(instance_name) > 0)
    ? g_markup_escape_text(instance_name, -1)
    : NULL;

  gchar *markup;
  if(esc_iname && strlen(esc_iname) > 0)
  {
    markup = g_strdup_printf("<span size=\"smaller\" alpha=\"70%%\">%s</span>\n%s <span size=\"smaller\" weight=\"light\" alpha=\"80%%\">• %s</span>",
                             _("blend mask"), esc_mname, esc_iname);
  }
  else if(strlen(esc_mname) > 0)
  {
    markup = g_strdup_printf("<span size=\"smaller\" alpha=\"70%%\">%s</span>\n%s",
                             _("blend mask"), esc_mname);
  }
  else
  {
    markup = g_strdup_printf("<span size=\"smaller\" alpha=\"70%%\">%s</span>\n<span weight=\"light\" alpha=\"70%%\">%s</span>",
                             _("blend mask"), _("no focused module"));
  }

  g_free(esc_mname);
  g_free(esc_iname);
  return markup;
}

// `handoff`: whether the module giving the panel up is doing so because
// another one is taking focus, in which case the panel is passed straight on
// to whatever dev->gui_module now names. FALSE on teardown, where there is no
// successor to hand it to (see dt_iop_gui_blend_masks_panel_release).
static void _masks_flexi_release_full(dt_iop_module_t *module, const gboolean handoff)
{
  if(!module || !module->blend_data) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd->relocatable_box || !GTK_IS_WIDGET(bd->relocatable_box)) return;

  const gboolean is_focused = darktable.develop && darktable.develop->gui_module == module;
  const gboolean show = is_focused && module->expanded;

  const gboolean was_hosted =
    darktable.develop->proxy.masks_flexi_host.hosted_module == module;
  if(was_hosted) darktable.develop->proxy.masks_flexi_host.hosted_module = NULL;

  if(bd->masks_right_cluster && GTK_IS_WIDGET(bd->masks_right_cluster))
  {
    if(bd->showmask && GTK_IS_WIDGET(bd->showmask) && gtk_widget_get_parent(bd->showmask) != bd->masks_right_cluster)
      _reparent_into(bd->showmask, bd->masks_right_cluster, FALSE, FALSE);
    if(bd->mask_enable_toggle && GTK_IS_WIDGET(bd->mask_enable_toggle) && gtk_widget_get_parent(bd->mask_enable_toggle) != bd->masks_right_cluster)
      _reparent_into(bd->mask_enable_toggle, bd->masks_right_cluster, FALSE, FALSE);
  }
  if(bd->masks_blend_header && GTK_IS_WIDGET(bd->masks_blend_header)
     && gtk_widget_get_parent(bd->masks_blend_header) != GTK_WIDGET(bd->relocatable_box))
  {
    _reparent_into(bd->masks_blend_header, GTK_WIDGET(bd->relocatable_box), FALSE, FALSE);
    gtk_box_reorder_child(bd->relocatable_box, bd->masks_blend_header, 0);
  }
  if(bd->masks_panel_body && GTK_IS_WIDGET(bd->masks_panel_body)
     && gtk_widget_get_parent(GTK_WIDGET(bd->masks_panel_body)) != GTK_WIDGET(bd->relocatable_box))
  {
    _reparent_into(GTK_WIDGET(bd->masks_panel_body), GTK_WIDGET(bd->relocatable_box), FALSE, FALSE);
    gtk_box_reorder_child(bd->relocatable_box, GTK_WIDGET(bd->masks_panel_body), 1);
  }
  if(bd->masks_blend_header && GTK_IS_WIDGET(bd->masks_blend_header))
    gtk_widget_set_visible(bd->masks_blend_header, TRUE);

  if(bd->iopw && GTK_IS_WIDGET(bd->iopw))
  {
    _reparent_into(GTK_WIDGET(bd->relocatable_box), bd->iopw, FALSE, FALSE);
    gtk_widget_set_visible(GTK_WIDGET(bd->relocatable_box), show);
  }
  // back in the module's own expander. When that is where the panel actually
  // lives (embedded), the in-header arrow keeps working, now folding the panel
  // body away in place; when the box only landed here because this module lost
  // focus, its real home is a host and the arrow has nothing to act on.
  const gboolean embedded =
    _masks_panel_position() == MASKS_PANEL_POS_EMBEDDED;
  gtk_widget_set_visible(bd->flexi_inline_collapse_btn, embedded);
  // the right-dock mirroring is that dock's alone -- back home, the header
  // reads left-to-right like every other module's
  _masks_header_apply_side(bd, FALSE);
  _masks_embedded_apply_collapsed(module, embedded && (!module->expanded || _masks_panel_collapsed_pref()));
  gtk_widget_hide(bd->masks_options_btn);  // options open on the toggle's right-click now
  // back in the module's own content -- restore the embedded inset (see
  // darktable.css's "#blending-tabs.blending-tabs-embedded")
  dt_gui_add_class(bd->masks_blend_header, "blending-tabs-embedded");
  if(bd->masks_blend_header_label && GTK_IS_LABEL(bd->masks_blend_header_label))
    gtk_label_set_text(GTK_LABEL(bd->masks_blend_header_label), _("blend mask"));

  if(was_hosted)
  {
    dt_lib_module_t *util_host = darktable.develop->proxy.masks_flexi_host.module;
    if(util_host && util_host->expander)
    {
      _masks_utility_apply_collapsed(util_host, TRUE);

      GtkWidget *lbl = darktable.develop->proxy.masks_flexi_host.header_label;
      GtkWidget *levb = darktable.develop->proxy.masks_flexi_host.label_evb;
      if(!lbl)
      {
        GtkWidget *header = DTGTK_EXPANDER(util_host->expander)->header;
        GList *children = gtk_container_get_children(GTK_CONTAINER(header));
        for(GList *c = children; c; c = g_list_next(c))
        {
          if(GTK_IS_EVENT_BOX(c->data))
          {
            GtkWidget *child = gtk_bin_get_child(GTK_BIN(c->data));
            if(GTK_IS_LABEL(child))
            {
              lbl = child;
              levb = GTK_WIDGET(c->data);
              darktable.develop->proxy.masks_flexi_host.header_label = lbl;
              darktable.develop->proxy.masks_flexi_host.label_evb = levb;
              break;
            }
          }
        }
        g_list_free(children);
      }

      if(lbl && GTK_IS_LABEL(lbl))
      {
        gchar *markup = _model_masks_panel_header_markup(NULL, NULL, TRUE);
        gtk_label_set_markup(GTK_LABEL(lbl), markup);
        g_free(markup);
      }

      if(util_host->arrow)
      {
        gtk_widget_set_sensitive(util_host->arrow, FALSE);
        gtk_widget_set_tooltip_text(util_host->arrow, _("disabled because no module is selected"));
      }
      if(levb)
      {
        gtk_widget_set_sensitive(levb, FALSE);
        gtk_widget_set_tooltip_text(levb, _("disabled because no module is selected"));
      }
    }
    if(util_host && util_host->preset_label && GTK_IS_LABEL(util_host->preset_label))
      gtk_label_set_text(GTK_LABEL(util_host->preset_label), "");

    _masks_flexi_host_reconfigure();
    // dev->gui_module is already updated to the new focus target (or NULL)
    // by the time this runs -- see dt_iop_gui_set_focus in imageop.c, which
    // sets it before calling lose_focus on the outgoing module
    dt_iop_module_t *next = darktable.develop->gui_module;
    dt_iop_gui_blend_data_t *next_bd = next ? next->blend_data : NULL;
    // no handing over on teardown: gui_module there is not a module taking
    // focus but one on its way to being freed by the same loop, and re-showing
    // the panel for it undoes the hide darkroom's leave() already did -- which
    // left an empty flexi panel holding open a column of the lighttable
    const gboolean next_wants_host =
      handoff && next && next_bd && next_bd->masks_support;
    if(next_wants_host)
    {
      const int pos = _masks_panel_position();
      if(pos == MASKS_PANEL_POS_CANVAS)
      {
        const gboolean next_mask_active =
          next->blend_params && next->blend_params->mask_mode != DEVELOP_MASK_DISABLED;
        const dt_masks_panel_state_t state =
          _model_masks_panel_state(pos, TRUE, TRUE, next->expanded,
                                   next_mask_active, _masks_panel_collapsed_pref());
        dt_ui_flexi_panel_set_icon(darktable.gui->ui, state.corner_icon_active,
                                   _mask_mode_label(next->blend_params ? next->blend_params->mask_mode : 0));
        dt_ui_flexi_panel_set_collapsed(darktable.gui->ui,
                                        state.panel_collapsed,
                                        TRUE, FALSE);
      }
      else
      {
        dt_ui_flexi_panel_set_icon(darktable.gui->ui, FALSE, NULL);
        dt_ui_flexi_panel_set_collapsed(darktable.gui->ui, TRUE, FALSE, FALSE);
      }
    }
    else
    {
      // nothing is focused anymore (or the focused module has no masking):
      // hide the panel and its corner icon entirely
      dt_ui_flexi_panel_set_icon(darktable.gui->ui, FALSE, NULL);
      dt_ui_flexi_panel_set_collapsed(darktable.gui->ui, TRUE, FALSE, FALSE);
    }
  }
}

void _masks_flexi_release(dt_iop_module_t *module)
{
  _masks_flexi_release_full(module, TRUE);
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
// Guarantee the host shows exactly one module's panel.
//
// hosted_module is what normally does this: relocate releases it before
// installing its own widgets. But it is a single pointer maintained by hand
// across focus changes, position changes and module teardown, and nothing ever
// removes a header from a host except the release that pairs with it. So any
// path that loses track of a module whose header is still parented in a host
// -- a module destroyed while hosted, a focus change that never reached us --
// strands that header there permanently, and the user gets two stacked panel
// headers with a single body under them.
//
// This asks the widget tree instead of trusting the pointer: any *live* module
// still parented in a host, other than the one taking over, is released
// properly. It logs when it fires, because it firing means one of those paths
// is still wrong and the log names the module that got left behind.
static void _release_stray_hosted(dt_iop_module_t *keep)
{
  for(GList *m = darktable.develop->iop; m; m = g_list_next(m))
  {
    dt_iop_module_t *other = m->data;
    if(other == keep || !other->blend_data) continue;
    if(!_widgets_are_hosted(other->blend_data)) continue;

    dt_print(DT_DEBUG_MASKS,
             "[masks] flexi panel: '%s' was still hosted when '%s' took it over"
             " -- releasing it (its header would have been stranded)",
             other->op, keep->op);
    _masks_flexi_release(other);
  }
}

void _masks_flexi_relocate(dt_iop_module_t *module)
{
  if(!module || !module->blend_data) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd->relocatable_box) return;

  const int pos = _masks_panel_position();
  const uint32_t mask_mode = module->blend_params->mask_mode;
  const gboolean is_focused = darktable.develop->gui_module == module;
  const gboolean has_masking = bd->masks_support;
  const gboolean mask_active = mask_mode != DEVELOP_MASK_DISABLED;
  const dt_masks_panel_state_t state =
    _model_masks_panel_state(pos, is_focused, has_masking, module->expanded,
                             mask_active, _masks_panel_collapsed_pref());

  GtkWidget *target = NULL;
  if(state.want_hosted && pos == MASKS_PANEL_POS_UTILITY)
  {
    dt_lib_module_t *host = darktable.develop->proxy.masks_flexi_host.module;
    GtkBox *content_box = darktable.develop->proxy.masks_flexi_host.content_box;
    if(host && content_box) target = GTK_WIDGET(content_box);
  }
  else if(state.want_hosted) // LEFT / RIGHT
  {
    target = dt_ui_flexi_panel_content(darktable.gui->ui);
    dt_ui_flexi_panel_set_side(darktable.gui->ui, _masks_panel_side_right());
  }

  if(!target)
  {
    if(pos == MASKS_PANEL_POS_EMBEDDED && is_focused && has_masking)
    {
      dt_iop_module_t *prev = darktable.develop->proxy.masks_flexi_host.hosted_module;
      const gboolean focus_changed = (prev != module);
      if(prev && prev != module) _masks_flexi_release(prev);
      darktable.develop->proxy.masks_flexi_host.hosted_module = module;

      if(bd->masks_blend_header && gtk_widget_get_parent(bd->masks_blend_header) != GTK_WIDGET(bd->relocatable_box))
      {
        _reparent_into(bd->masks_blend_header, GTK_WIDGET(bd->relocatable_box), FALSE, FALSE);
        gtk_box_reorder_child(bd->relocatable_box, bd->masks_blend_header, 0);
      }
      if(bd->masks_panel_body && gtk_widget_get_parent(GTK_WIDGET(bd->masks_panel_body)) != GTK_WIDGET(bd->relocatable_box))
      {
        _reparent_into(GTK_WIDGET(bd->masks_panel_body), GTK_WIDGET(bd->relocatable_box), FALSE, FALSE);
        gtk_box_reorder_child(bd->relocatable_box, GTK_WIDGET(bd->masks_panel_body), 1);
      }

      if(bd->masks_right_cluster && GTK_IS_WIDGET(bd->masks_right_cluster))
      {
        if(bd->showmask && GTK_IS_WIDGET(bd->showmask) && gtk_widget_get_parent(bd->showmask) != bd->masks_right_cluster)
          _reparent_into(bd->showmask, bd->masks_right_cluster, FALSE, FALSE);
        if(bd->mask_enable_toggle && GTK_IS_WIDGET(bd->mask_enable_toggle) && gtk_widget_get_parent(bd->mask_enable_toggle) != bd->masks_right_cluster)
          _reparent_into(bd->mask_enable_toggle, bd->masks_right_cluster, FALSE, FALSE);
      }
      if(bd->masks_blend_header && GTK_IS_WIDGET(bd->masks_blend_header))
        gtk_widget_set_visible(bd->masks_blend_header, TRUE);

      if(bd->iopw && GTK_IS_WIDGET(bd->iopw))
      {
        _reparent_into(GTK_WIDGET(bd->relocatable_box), bd->iopw, FALSE, FALSE);
        gtk_widget_set_visible(GTK_WIDGET(bd->relocatable_box), module->expanded);
      }

      gtk_widget_set_visible(bd->flexi_inline_collapse_btn, TRUE);
      _masks_header_apply_side(bd, FALSE);
      if(focus_changed)
        _masks_embedded_apply_collapsed(module, _masks_panel_collapsed_pref());
      gtk_widget_hide(bd->masks_options_btn);  // options open on the toggle's right-click now
      dt_gui_add_class(bd->masks_blend_header, "blending-tabs-embedded");
      if(bd->masks_blend_header_label && GTK_IS_LABEL(bd->masks_blend_header_label))
        gtk_label_set_text(GTK_LABEL(bd->masks_blend_header_label), _("blend mask"));
      return;
    }

    // an expanded-but-unfocused module must not show its full blend/mask
    // panel inline, whatever the position preference (see
    // _masks_flexi_release, which gates visibility on real focus)
    _masks_flexi_release(module);
    return;
  }

  dt_iop_module_t *prev = darktable.develop->proxy.masks_flexi_host.hosted_module;
  const gboolean focus_changed = (prev != module);
  if(prev && prev != module) _masks_flexi_release(prev);
  _release_stray_hosted(module);

  darktable.develop->proxy.masks_flexi_host.hosted_module = module;

  if(pos == MASKS_PANEL_POS_CANVAS)
  {
    GtkWidget *hdr_target = dt_ui_flexi_panel_header(darktable.gui->ui);
    GtkWidget *cnt_target = dt_ui_flexi_panel_content(darktable.gui->ui);
    if(hdr_target && bd->masks_blend_header)
    {
      _reparent_into(bd->masks_blend_header, hdr_target, FALSE, FALSE);
      gtk_widget_show(bd->masks_blend_header);
    }
    if(cnt_target && bd->masks_panel_body)
    {
      _reparent_into(GTK_WIDGET(bd->masks_panel_body), cnt_target, FALSE, FALSE);
      gtk_widget_show(GTK_WIDGET(bd->masks_panel_body));
    }
  }
  else
  {
    if(bd->masks_blend_header && gtk_widget_get_parent(bd->masks_blend_header) != GTK_WIDGET(bd->relocatable_box))
    {
      _reparent_into(bd->masks_blend_header, GTK_WIDGET(bd->relocatable_box), FALSE, FALSE);
      gtk_box_reorder_child(bd->relocatable_box, bd->masks_blend_header, 0);
    }
    if(bd->masks_panel_body && gtk_widget_get_parent(GTK_WIDGET(bd->masks_panel_body)) != GTK_WIDGET(bd->relocatable_box))
    {
      _reparent_into(GTK_WIDGET(bd->masks_panel_body), GTK_WIDGET(bd->relocatable_box), FALSE, FALSE);
      gtk_box_reorder_child(bd->relocatable_box, GTK_WIDGET(bd->masks_panel_body), 1);
    }
    _reparent_into(GTK_WIDGET(bd->relocatable_box), target, FALSE, FALSE);
    gtk_widget_show(GTK_WIDGET(bd->relocatable_box));
  }

  // hosted: the host itself collapses (grid panel to its corner icon, utility
  // lib to its expander header), so the body is never folded here -- undo any
  // embedded fold the box is carrying over
  if(bd->masks_panel_body)
    gtk_widget_set_visible(GTK_WIDGET(bd->masks_panel_body), TRUE);
  _masks_flexi_host_reconfigure();

  if(pos == MASKS_PANEL_POS_UTILITY)
  {
    GtkBox *toggle_box = darktable.develop->proxy.masks_flexi_host.toggle_box;
    if(toggle_box)
    {
      _reparent_into(bd->mask_enable_toggle, GTK_WIDGET(toggle_box), FALSE, FALSE);
      gtk_widget_set_valign(bd->mask_enable_toggle, GTK_ALIGN_CENTER);
      gtk_widget_show(bd->mask_enable_toggle);
      gtk_widget_show(GTK_WIDGET(toggle_box));
    }
    GtkBox *actions_box = darktable.develop->proxy.masks_flexi_host.actions_box;
    if(actions_box)
    {
      _reparent_into(bd->showmask, GTK_WIDGET(actions_box), FALSE, FALSE);
      gtk_widget_set_valign(bd->showmask, GTK_ALIGN_CENTER);
      const gboolean is_mask_enabled = (module->blend_params->mask_mode != DEVELOP_MASK_DISABLED);
      gtk_widget_set_visible(bd->showmask, is_mask_enabled && !module->hide_enable_button);
      gtk_widget_show(GTK_WIDGET(actions_box));
    }
    gtk_widget_set_visible(bd->masks_blend_header, FALSE);

    dt_lib_module_t *host = darktable.develop->proxy.masks_flexi_host.module;
    if(host && host->expander)
    {
      GtkWidget *lbl = darktable.develop->proxy.masks_flexi_host.header_label;
      GtkWidget *levb = darktable.develop->proxy.masks_flexi_host.label_evb;
      if(!lbl)
      {
        GtkWidget *header = DTGTK_EXPANDER(host->expander)->header;
        GList *children = gtk_container_get_children(GTK_CONTAINER(header));
        for(GList *c = children; c; c = g_list_next(c))
        {
          if(GTK_IS_EVENT_BOX(c->data))
            {
            GtkWidget *child = gtk_bin_get_child(GTK_BIN(c->data));
            if(GTK_IS_LABEL(child))
            {
              lbl = child;
              levb = GTK_WIDGET(c->data);
              darktable.develop->proxy.masks_flexi_host.header_label = lbl;
              darktable.develop->proxy.masks_flexi_host.label_evb = levb;
              break;
            }
          }
        }
        g_list_free(children);
      }

      if(lbl && GTK_IS_LABEL(lbl))
      {
        gchar *markup = _model_masks_panel_header_markup(module ? module->name() : NULL,
                                                         module ? dt_iop_get_instance_name(module) : NULL,
                                                         TRUE);
        gtk_label_set_markup(GTK_LABEL(lbl), markup);
        g_free(markup);
      }

      if(host->arrow)
      {
        gtk_widget_set_sensitive(host->arrow, TRUE);
        gtk_widget_set_tooltip_text(host->arrow, _("show module"));
      }
      if(levb)
      {
        gtk_widget_set_sensitive(levb, TRUE);
        gtk_widget_set_tooltip_text(levb, _("blend mask"));
      }
    }
    if(host && host->preset_label)
      gtk_widget_set_visible(host->preset_label, FALSE);

    // the shared state, like the other two positions. Previously this derived
    // expansion from mask_mode alone, so a relocate (a focus change, a mode
    // change) re-expanded a lib the user had just folded -- and, since
    // dt_lib_gui_set_expanded persists, overwrote the folded state as it went.
    if(host && focus_changed)
      _masks_utility_apply_collapsed(host, state.panel_collapsed);
  }
  else
  {
    if(bd->masks_right_cluster)
    {
      if(gtk_widget_get_parent(bd->showmask) != bd->masks_right_cluster)
        _reparent_into(bd->showmask, bd->masks_right_cluster, FALSE, FALSE);
      if(gtk_widget_get_parent(bd->mask_enable_toggle) != bd->masks_right_cluster)
        _reparent_into(bd->mask_enable_toggle, bd->masks_right_cluster, FALSE, FALSE);
    }
    gtk_widget_set_visible(bd->masks_blend_header, TRUE);
    // hosted elsewhere now -- drop the embedded inset, the host already
    // provides its own (see darktable.css's "#blending-tabs.blending-tabs-embedded")
    dt_gui_remove_class(bd->masks_blend_header, "blending-tabs-embedded");
  }

  if(bd->masks_blend_header_label && GTK_IS_LABEL(bd->masks_blend_header_label))
  {
    const gboolean is_hosted = (pos == MASKS_PANEL_POS_CANVAS);
    gchar *markup = _model_masks_panel_header_markup(module ? module->name() : NULL,
                                                     module ? dt_iop_get_instance_name(module) : NULL,
                                                     is_hosted);
    if(is_hosted)
      gtk_label_set_markup(GTK_LABEL(bd->masks_blend_header_label), markup);
    else
      gtk_label_set_text(GTK_LABEL(bd->masks_blend_header_label), markup);
    g_free(markup);
  }

  // in the utility lib, that lib's own header hamburger is repurposed to
  // this same options menu (see masks_flexi_host.c's view_enter and
  // dt_iop_gui_blend_masks_options_popup) -- don't show a second, redundant
  // one in the mode-select row too
  gtk_widget_hide(bd->masks_options_btn);  // options open on the toggle's right-click now

  if(pos == MASKS_PANEL_POS_CANVAS)
  {
    dt_ui_flexi_panel_set_icon(darktable.gui->ui, state.corner_icon_active,
                               _mask_mode_label(mask_mode));
    // the shared state again -- applying it, not deciding it, so persist=FALSE.
    //
    // Also when the panel simply disagrees with the state, not only on a focus
    // change: expanding a module that already has focus is a relocate with
    // focus_changed FALSE, and state.panel_collapsed has just gone from TRUE
    // (a collapsed module never shows the panel, see _model_masks_panel_state)
    // to the user's preference. Gated on focus alone, nothing applied that, so
    // the first module you expanded came up with its panel still hidden and
    // only a detour through another module brought it back.
    //
    const gboolean panel_disagrees =
      state.panel_collapsed != dt_ui_flexi_panel_is_collapsed(darktable.gui->ui);
    if(focus_changed || panel_disagrees)
      dt_ui_flexi_panel_set_collapsed(darktable.gui->ui, state.panel_collapsed,
                                      TRUE, FALSE);

    // no in-header collapse arrow out on the canvas: a small button on a
    // floating panel is a poor target for the one thing you always want to be
    // able to do to it, and the darkroom toolbar's mask-panel button (which
    // shows whether the panel is out) does the same job at a fixed, learnable
    // position. It stays for the embedded position, where the header is the only
    // control there is.
    gtk_widget_set_visible(bd->flexi_inline_collapse_btn, FALSE);
    _masks_header_apply_side(bd, dt_ui_flexi_panel_is_right(darktable.gui->ui));
  }
  else
  {
    dt_ui_flexi_panel_set_icon(darktable.gui->ui, FALSE, NULL);
    dt_ui_flexi_panel_set_collapsed(darktable.gui->ui, TRUE, FALSE, FALSE);

    if(pos == MASKS_PANEL_POS_EMBEDDED)
    {
      gtk_widget_set_visible(bd->flexi_inline_collapse_btn, TRUE);
      _masks_header_apply_side(bd, FALSE);
    }
    else // MASKS_PANEL_POS_UTILITY
    {
      gtk_widget_set_visible(bd->flexi_inline_collapse_btn, FALSE);
    }
  }

  // focus moved, or the hosted module's masking changed: the toolbox button
  // reports both ("is there a panel to show" and "is it showing")
  dt_iop_gui_blend_masks_panel_sync_toolbox();
}

// ---- position preference ---------------------------------------------------

static void _masks_panel_position_activate(GtkToggleButton *mi, dt_iop_module_t *module)
{
  // a real GtkRadioButton group, which fires "toggled" on both the item losing
  // the selection and the one gaining it -- only act on the latter. (The menu
  // this replaced had to use plain check *menu items* and enforce exclusion by
  // hand, because the theme styles no "radio" node for menu items; radio
  // buttons in a popover are styled and used elsewhere, see global_toolbox.c.)
  if(darktable.gui->reset || !gtk_toggle_button_get_active(mi)) return;

  const int pos = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(mi), "dt-panel-pos"));
  dt_conf_set_int("plugins/darkroom/blend/masks_panel_position", pos);

  // update the utility-mode host lib's own visibility for the new position
  _masks_flexi_host_reconfigure();

  // leaving the separate-panel (left/right) mechanism entirely: force it
  // fully hidden (not just emptied) rather than leaving an empty panel
  // visible -- _masks_flexi_relocate()'s own release path only re-applies
  // whatever visibility it already had, which isn't enough here
  if(pos != MASKS_PANEL_POS_CANVAS)
    dt_ui_flexi_panel_set_collapsed(darktable.gui->ui, TRUE, FALSE, FALSE);

  // repositioning is a deliberate user action -- make sure the result is
  // actually visible, in every position: unfold the panel and store that,
  // before the relocate below applies it. Overriding _masks_flexi_relocate's
  // "no mask -> fold to the corner icon / the collapsed header" is the point:
  // explicitly picking a position should show what was picked.
  _masks_panel_set_collapsed_pref(FALSE);

  // decide where this (focused) module's content should live now
  if(module)
  {
    _masks_flexi_relocate(module);

    switch(pos)
    {
    case MASKS_PANEL_POS_CANVAS:
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
}

// appends a "blend mask panel position" section to `box` -- radios under a
// section label, so the current choice is visible at a glance
void _add_masks_panel_position_box(GtkWidget *box, dt_iop_module_t *module)
{
  GtkWidget *header = gtk_label_new(_("blend mask panel position"));
  gtk_label_set_justify(GTK_LABEL(header), GTK_JUSTIFY_CENTER);
  dt_gui_add_class(header, "dt_section_label");
  gtk_widget_set_tooltip_text(header, _("where the blend mask panel (groups, elements, refinements)"
              " is shown.\n"
              "moving to/from the utility module or a separate panel takes effect"
              " the next time the panel is rebuilt (e.g. after reopening darkroom)."));
  dt_gui_box_add(box, header);

  static const struct
  {
    int pos;
    const char *label;
  } items[] = {
    { MASKS_PANEL_POS_EMBEDDED, N_("embedded within each module (default)") },
    { MASKS_PANEL_POS_UTILITY, N_("utility module, left panel") },
    // one entry, not one per side: which edge it opens on is no longer part of
    // the choice, it is whichever edge the user last opened it on
    { MASKS_PANEL_POS_CANVAS, N_("separate panel, beside the canvas") },
  };

  const int cur_pos = _masks_panel_position();
  GtkWidget *group = NULL;
  GtkWidget *radios[G_N_ELEMENTS(items)];

  // states first, handlers after: setting the active radio while building would
  // otherwise read as the user choosing a position and relocate the panel
  ++darktable.gui->reset;
  for(size_t i = 0; i < G_N_ELEMENTS(items); i++)
  {
    radios[i] = gtk_radio_button_new_with_label_from_widget(
      group ? GTK_RADIO_BUTTON(group) : NULL, _(items[i].label));
    if(!group) group = radios[i];
    g_object_set_data(G_OBJECT(radios[i]), "dt-panel-pos", GINT_TO_POINTER(items[i].pos));
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(radios[i]), items[i].pos == cur_pos);
    dt_gui_box_add(box, radios[i]);
  }
  --darktable.gui->reset;

  for(size_t i = 0; i < G_N_ELEMENTS(items); i++)
    g_signal_connect(G_OBJECT(radios[i]), "toggled",
                     G_CALLBACK(_masks_panel_position_activate), module);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
