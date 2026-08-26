/*
    This file is part of darktable,
    Copyright (C) 2013-2026 darktable developers.

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

#include "common/debug.h"
#include "control/conf.h"
#include "control/control.h"
#include "develop/blend.h"
#include "develop/imageop.h"
#include "develop/masks.h"

// a flattened scratch-group entry's own parentid is the id of its *real*
// structural parent (see dt_masks_group_ungroup, masks/masks.c), not
// necessarily the module's own top group -- for a committed AI-mask bundle's
// child (see _register_vectorized_forms, masks/object.c), that is the
// bundle's own formid. Returns the bundle form if `fpt` is one of its
// children, else NULL. Used throughout this file to treat a bundle as one
// coordinated unit on the canvas (select/highlight/drag/grow-shrink), while
// individual bezier-node dragging still targets just the one child.
static dt_masks_form_t *_bundle_parent_of(const dt_masks_point_group_t *fpt)
{
  dt_masks_form_t *parent = dt_masks_get_from_id(darktable.develop, fpt->parentid);
  return (parent && (parent->type & DT_MASKS_OBJECT)) ? parent : NULL;
}

// after a bundle-wide edit (coordinated resize/drag) has mutated every
// child's own points directly, force-rebuild each child's own display buffer
// in `gui->points` -- dt_masks_gui_form_create (unlike its "only if the pipe
// hash changed" _test_create sibling) always recomputes unconditionally, the
// same call path.c's own scroll/drag handlers use after mutating one form.
static void _bundle_refresh_children(dt_masks_form_t *scratch_grp,
                                     const dt_masks_form_t *bundle,
                                     dt_masks_form_gui_t *gui,
                                     const dt_iop_module_t *module)
{
  int pos = 0;
  for(GList *l = scratch_grp->points; l; l = g_list_next(l), pos++)
  {
    const dt_masks_point_group_t *pt = l->data;
    if(pt->parentid != bundle->formid) continue;
    dt_masks_form_t *child = dt_masks_get_from_id(darktable.develop, pt->formid);
    if(child) dt_masks_gui_form_create(child, gui, pos, module);
  }
}

static int _group_events_mouse_scrolled(dt_iop_module_t *module,
                                        const float pzx,
                                        const float pzy,
                                        const int up,
                                        const uint32_t state,
                                        dt_masks_form_t *form,
                                        const int unused1,
                                        dt_masks_form_gui_t *gui,
                                        const int unused)
{
  if(gui->group_edited >= 0)
  {
    // we get the form
    dt_masks_point_group_t *fpt = g_list_nth_data(form->points, gui->group_edited);
    dt_masks_form_t *sel = dt_masks_get_from_id(darktable.develop, fpt->formid);
    if(!sel || !sel->functions) return 0;

    // Bundle-coordinated scroll gestures. dt_modifier_is is an *exact* match
    // among Shift+Control+Alt/Meta, so the four modifier states (none, ctrl,
    // shift, ctrl+shift) must be told apart the same way path.c's own
    // _path_events_mouse_scrolled does -- as one mutually exclusive
    // if/else-if chain -- rather than four independent conditions that can
    // silently overlap (e.g. "not exactly ctrl" AND "not exactly shift" is
    // also true for ctrl+shift together, which an earlier version of this
    // function got wrong: it hijacked ctrl+shift's legacy-resize gesture into
    // plain resize instead of leaving it alone).
    dt_masks_form_t *bundle = _bundle_parent_of(fpt);
    if(bundle && dt_modifier_is(state, GDK_CONTROL_MASK))
    {
      // ctrl+scroll (opacity): the panel's own inline opacity control for a
      // bundle row edits the bundle's own membership entry in the module's
      // top group (its overall contribution to the composite) via
      // dt_masks_form_change_opacity(bundle, module's group formid, ...) --
      // not any child's own internal per-membership opacity within the
      // bundle, which is invisible bookkeeping nothing else in the UI
      // exposes. Drive the exact same call here.
      const float amount = up ? 0.05f : -0.05f;
      dt_masks_form_change_opacity(bundle, module->blend_params->mask_id, amount);
      dt_masks_iop_update(module);
      dt_control_queue_redraw_center();
      return 1;
    }
    if(bundle && !dt_modifier_is(state, GDK_CONTROL_MASK))
    {
      if(dt_modifier_is(state, GDK_SHIFT_MASK) && bundle->functions && bundle->functions->modify_property)
      {
        // shift+scroll (feather): drives the bundle's own FEATHER case
        // (_object_bundle_modify_property) directly, exactly as the panel's
        // own feather slider does -- every child, the scrolled one included,
        // scaled by the one ratio a scroll tick represents (matching
        // dt_masks_change_size's own step), through the identical call. No
        // child is special-cased against its siblings.
        const float ratio = up ? 1.0f / 0.97f : 0.97f;
        float sum = 0.0f, minv = 0.0f, maxv = 0.0f;
        int count = 0;
        bundle->functions->modify_property(bundle, DT_MASKS_PROPERTY_FEATHER, 1.0f, ratio,
                                           &sum, &count, &minv, &maxv);
        dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
        _bundle_refresh_children(form, bundle, gui, module);
        dt_masks_iop_update(module);
        dt_control_queue_redraw_center();
        return 1;
      }
      else if(dt_modifier_is(state, GDK_CONTROL_MASK | GDK_SHIFT_MASK)
              && gui->edit_mode == DT_MASKS_EDIT_FULL
              && bundle->functions && bundle->functions->modify_property)
      {
        // ctrl+shift (legacy centroid resize, path.c's _path_resize_centroid)
        // is the exact same affine scale-about-center operation as the
        // panel's own SIZE slider -- same dt_masks_change_size step ratio,
        // same math, just driven by a scroll tick instead of a dragged
        // value. Drives the bundle's own coordinated SIZE case directly, the
        // same way plain-scroll resize and shift-scroll feather already do.
        const float ratio = up ? 1.0f / 0.97f : 0.97f;
        float sum = 0.0f, minv = 0.0f, maxv = 0.0f;
        int count = 0;
        bundle->functions->modify_property(bundle, DT_MASKS_PROPERTY_SIZE, 1.0f, ratio,
                                           &sum, &count, &minv, &maxv);
        dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
        _bundle_refresh_children(form, bundle, gui, module);
        dt_masks_iop_update(module);
        dt_control_queue_redraw_center();
        return 1;
      }
      else if(gui->edit_mode == DT_MASKS_EDIT_FULL
              && bundle->functions && bundle->functions->resize && bundle->functions->resize_get)
      {
        // plain scroll (grow/shrink): drives the bundle's own coordinated
        // resize (same cached-baseline mechanism as the panel's "shrink or
        // grow" slider, see _object_bundle_resize/object.c).
        const gboolean use_percent =
          !g_strcmp0(dt_conf_get_string_const("masks/path_resize_unit"), "% of path size");
        float amount = 0.0f;
        bundle->functions->resize_get(bundle, use_percent, &amount);
        const int new_amount = (int)roundf(amount) + (up ? 1 : -1);
        if(bundle->functions->resize(bundle, new_amount, use_percent))
        {
          dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
          _bundle_refresh_children(form, bundle, gui, module);
          dt_masks_iop_update(module);
          dt_control_queue_redraw_center();
        }
        return 1;
      }
    }
    // anything not handled above (ordinary, non-bundle members) falls
    // straight through to the child's own handler, unchanged.
    return sel->functions->mouse_scrolled(module, pzx, pzy, up, state, sel,
                                          fpt->parentid, gui, gui->group_edited);
  }
  return 0;
}

static int _group_events_button_pressed(dt_iop_module_t *module,
                                        const float pzx,
                                        const float pzy,
                                        const double pressure,
                                        const int which,
                                        const int type,
                                        const uint32_t state,
                                        dt_masks_form_t *form,
                                        const int unused1,
                                        dt_masks_form_gui_t *gui,
                                        const int unused2)
{
  if(gui->group_edited != gui->group_selected)
  {
    // we set the selected form in edit mode
    gui->group_edited = gui->group_selected;
    // we initialise some variable
    gui->dx = gui->dy = 0.0f;
    gui->form_selected = FALSE;
    gui->border_selected = FALSE;
    gui->form_dragging = FALSE;
    gui->form_rotating = FALSE;
    gui->source_rotating = FALSE;
    gui->counter_rotate_source = FALSE;
    gui->pivot_selected = FALSE;
    gui->point_border_selected = -1;
    gui->seg_selected = -1;
    gui->point_selected = -1;
    gui->feather_selected = -1;
    gui->point_border_dragging = -1;
    gui->seg_dragging = -1;
    gui->feather_dragging = -1;
    gui->point_dragging = -1;

    dt_control_queue_redraw_center();
    return 1;
  }
  if(gui->group_edited >= 0)
  {
    // we get the form
    dt_masks_point_group_t *fpt = g_list_nth_data(form->points, gui->group_edited);
    dt_masks_form_t *sel = dt_masks_get_from_id(darktable.develop, fpt->formid);
    if(!sel) return 0;
    if(sel->functions)
    {
      // did we asked for feather only?
      if(dt_modifier_is(state, GDK_SHIFT_MASK) ^ gui->select_only_border)
      {
        // then make sure we try to select the feather point
        gui->select_only_border = dt_modifier_is(state, GDK_SHIFT_MASK);
        sel->functions->mouse_moved(module, pzx, pzy, pressure,
                                    which, dt_dev_get_zoom_scale_full(), sel, fpt->parentid,
                                    gui, gui->group_edited);
      }

      const int ret = sel->functions->button_pressed(module, pzx, pzy, pressure, which, type, state, sel,
                                                     fpt->parentid, gui, gui->group_edited);
      // a plain click just selected this child individually (dt_masks_select_form,
      // called from inside the child's own button_pressed) -- if it belongs to
      // an AI-mask bundle, promote that selection to the whole bundle instead,
      // so canvas clicks always select the bundle as one unit, matching the
      // panel row's own selection (see dt_group_events_post_expose's highlight
      // logic below, which highlights every child sharing the selected parent).
      if(darktable.develop->mask_form_selected_id == sel->formid)
      {
        dt_masks_form_t *bundle = _bundle_parent_of(fpt);
        if(bundle) dt_masks_select_form(module, bundle);
      }
      return ret;
    }
  }
  return 0;
}

static int _group_events_button_released(dt_iop_module_t *module,
                                         const float pzx,
                                         const float pzy,
                                         const int which,
                                         const uint32_t state,
                                         dt_masks_form_t *form,
                                         const int unused1,
                                         dt_masks_form_gui_t *gui,
                                         const int unused2)
{
  if(gui->group_edited >= 0)
  {
    // we get the form
    dt_masks_point_group_t *fpt = g_list_nth_data(form->points, gui->group_edited);
    dt_masks_form_t *sel = dt_masks_get_from_id(darktable.develop, fpt->formid);
    if(!sel || !sel->functions) return 0;

    // a bundle-wide rotation drag (see _bundle_rotate_step) deliberately
    // skips the panel-sync call on every mouse_moved tick -- a full rebuild
    // per tick during a live drag would be janky -- but the panel's own
    // rotation slider still needs to catch up once the drag actually ends,
    // the same way scroll-driven feather/size/opacity already do per tick.
    const gboolean was_rotating = gui->form_rotating;
    const dt_masks_form_t *bundle = was_rotating ? _bundle_parent_of(fpt) : NULL;
    const int ret = sel->functions->button_released(module, pzx, pzy, which, state, sel, fpt->parentid,
                                                     gui, gui->group_edited);
    if(bundle) dt_masks_iop_update(module);
    return ret;
  }
  return 0;
}

static inline gboolean _is_handling_form(dt_masks_form_gui_t *gui)
{
  return gui->form_dragging
    || gui->source_dragging
    || gui->gradient_toggling
    || gui->form_rotating
    || gui->source_rotating
    || (gui->point_edited != -1)
    || (gui->point_dragging != -1)
    || (gui->feather_dragging != -1)
    || (gui->point_border_dragging != -1)
    || (gui->seg_dragging != -1);
}

// canvas ctrl+drag rotation on a bundle child rotates the whole AI-mask
// bundle together about one shared center, exactly like the panel's own
// rotation slider (reuses _object_bundle_modify_property's ROTATION case,
// the already skew-free pixel-space rotation) -- the child's own
// screen-space, per-shape-centroid rotation (dt_masks_rotate_ctrl_points via
// its own gpt display buffer, see path.c's form_rotating branch) is bypassed
// entirely: this computes the angular sweep about the bundle's own shared
// center instead and lets modify_property apply it to every child. Returns 1
// (always handles the tick) once form_rotating + a bundle parent are found.
static int _bundle_rotate_step(dt_iop_module_t *module,
                               dt_masks_form_t *bundle,
                               dt_masks_form_t *scratch_grp,
                               dt_masks_form_gui_t *gui,
                               const float pzx,
                               const float pzy)
{
  float wd, ht, iwidth, iheight;
  dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);
  if(iwidth <= 0.0f || iheight <= 0.0f || wd <= 0.0f || ht <= 0.0f)
  {
    gui->dx = pzx;
    gui->dy = pzy;
    return 1;
  }

  // shared bundle center: pooled mean of every child's own corner points
  // (form space), same convention _object_bundle_modify_property itself uses
  double cx = 0.0, cy = 0.0;
  int npts = 0;
  for(GList *l = bundle->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    const dt_masks_form_t *child = dt_masks_get_from_id(darktable.develop, pt->formid);
    if(!child) continue;
    for(GList *p = child->points; p; p = g_list_next(p))
    {
      const dt_masks_point_path_t *pp = p->data;
      cx += pp->corner[0];
      cy += pp->corner[1];
      npts++;
    }
  }
  if(npts == 0)
  {
    gui->dx = pzx;
    gui->dy = pzy;
    return 1;
  }
  cx /= npts;
  cy /= npts;

  // forward-transform the shared center into backbuffer/screen space -- the
  // same space the mouse position and path.c's own rotation gesture both
  // measure their angular sweep in, so the drag pivots visually where the
  // bundle actually is on screen.
  float piv[2] = { (float)cx * iwidth, (float)cy * iheight };
  dt_dev_distort_transform(darktable.develop, piv, 1);

  const float cmx = pzx * wd, cmy = pzy * ht;
  const float pmx = gui->dx * wd, pmy = gui->dy * ht;
  float dv = atan2f(cmy - piv[1], cmx - piv[0]) - atan2f(pmy - piv[1], pmx - piv[0]);
  if(fabsf(dv) > M_PI_F) dv -= copysignf(DT_2PI_F, dv);
  const float dv_deg = rad2degf(dv);

  if(dv_deg != 0.0f && bundle->functions && bundle->functions->modify_property)
  {
    float sum = 0.0f, minv = 0.0f, maxv = 0.0f;
    int count = 0;
    bundle->functions->modify_property(bundle, DT_MASKS_PROPERTY_ROTATION, 0.0f, dv_deg,
                                       &sum, &count, &minv, &maxv);
    _bundle_refresh_children(scratch_grp, bundle, gui, module);
  }

  gui->dx = pzx;
  gui->dy = pzy;
  dt_control_queue_redraw_center();
  return 1;
}

static int _group_events_mouse_moved(dt_iop_module_t *module,
                                     const float pzx,
                                     const float pzy,
                                     const double pressure,
                                     const int which,
                                     const float zoom_scale,
                                     dt_masks_form_t *form,
                                     const int unused1,
                                     dt_masks_form_gui_t *gui,
                                     const int unused2)
{
  const float as = dt_masks_sensitive_dist(zoom_scale);

  // we first don't do anything if we are inside a scrolling session

  if(gui->scrollx != 0.0f && gui->scrolly != 0.0f)
  {
    const float as2 = 0.015f / zoom_scale;
    if((gui->scrollx - pzx < as2 && gui->scrollx - pzx > -as2)
       && (gui->scrolly - pzy < as2 && gui->scrolly - pzy > -as2))
      return 1;
    gui->scrollx = gui->scrolly = 0.0f;
  }

  // if a form is in edit mode and we are dragging, don't try to
  // select another form
  if(gui->group_edited >= 0 && _is_handling_form(gui))
  {
    // we get the form
    dt_masks_point_group_t *fpt = g_list_nth_data(form->points, gui->group_edited);
    dt_masks_form_t *sel = dt_masks_get_from_id(darktable.develop, fpt->formid);
    if(!sel) return 0;

    // a ctrl+drag rotation on a bundle child rotates the whole bundle about
    // its own shared center instead -- fully handled here, bypassing the
    // child's own per-shape rotation entirely (see _bundle_rotate_step).
    if(gui->form_rotating)
    {
      dt_masks_form_t *rot_bundle = _bundle_parent_of(fpt);
      if(rot_bundle)
        return _bundle_rotate_step(module, rot_bundle, form, gui, pzx, pzy);
    }

    // a whole-shape (body) drag on a bundle child should translate the whole
    // AI-mask bundle together, not just this one child -- path.c's own
    // form_dragging case (the only kind of drag this applies to; a node/
    // feather/segment drag is left alone, still purely per-child) recomputes
    // its per-tick delta from its own first point's corner each call, so
    // snapshotting that corner before and after the delegated call recovers
    // exactly the delta it just applied, which is then reapplied verbatim to
    // every sibling (a pure translation needs no per-child sign-awareness,
    // unlike SIZE/ROTATION).
    dt_masks_form_t *bundle = gui->form_dragging ? _bundle_parent_of(fpt) : NULL;
    float anchor_before[2] = { 0.0f, 0.0f };
    if(bundle && sel->points)
    {
      const dt_masks_point_path_t *p0 = sel->points->data;
      anchor_before[0] = p0->corner[0];
      anchor_before[1] = p0->corner[1];
    }

    int rep = 0;
    if(sel->functions)
      rep = sel->functions->mouse_moved(module, pzx, pzy, pressure, which, zoom_scale, sel, fpt->parentid,
                                        gui, gui->group_edited);

    if(bundle && sel->points)
    {
      const dt_masks_point_path_t *p0 = sel->points->data;
      const float dx = p0->corner[0] - anchor_before[0];
      const float dy = p0->corner[1] - anchor_before[1];
      if(dx != 0.0f || dy != 0.0f)
      {
        for(GList *l = bundle->points; l; l = g_list_next(l))
        {
          const dt_masks_point_group_t *cpt = l->data;
          if(cpt->formid == sel->formid) continue; // already moved above
          dt_masks_form_t *sib = dt_masks_get_from_id(darktable.develop, cpt->formid);
          if(!sib) continue;
          for(GList *pp = sib->points; pp; pp = g_list_next(pp))
          {
            dt_masks_point_path_t *pt = pp->data;
            pt->corner[0] += dx; pt->corner[1] += dy;
            pt->ctrl1[0] += dx; pt->ctrl1[1] += dy;
            pt->ctrl2[0] += dx; pt->ctrl2[1] += dy;
          }
        }
        _bundle_refresh_children(form, bundle, gui, module);
      }
    }

    if(rep) return 1;
    // if a point is in state editing, then we don't want that another
    // form can be selected
    if(gui->point_edited >= 0) return 0;
  }

  // now we check if we are near a form
  int pos = 0;
  gui->form_selected = gui->border_selected = FALSE;
  gui->source_selected = gui->source_dragging = FALSE;
  gui->pivot_selected = FALSE;
  gui->feather_selected = -1;
  gui->point_edited = gui->point_selected = -1;
  gui->seg_selected = -1;
  gui->point_border_selected = -1;
  gui->group_edited = gui->group_selected = -1;
  gui->select_only_border = dt_modifier_is(which, GDK_SHIFT_MASK);

  dt_masks_form_t *sel = NULL;
  dt_masks_point_group_t *sel_fpt = NULL;
  int sel_pos = 0;
  float sel_dist = FLT_MAX;

  for(GList *fpts = form->points; fpts; fpts = g_list_next(fpts))
  {
    dt_masks_point_group_t *fpt = fpts->data;
    dt_masks_form_t *frm = dt_masks_get_from_id(darktable.develop, fpt->formid);
    // a hidden or disabled shape is not editable on the canvas: skip it when picking the
    // form under the cursor (it also draws no outline).
    if(fpt->state & (DT_MASKS_STATE_HIDDEN | DT_MASKS_STATE_DISABLE)) { pos++; continue; }
    int inside, inside_border, near, inside_source;
    float dist = FLT_MAX;
    inside = inside_border = inside_source = 0;
    near = -1;

    float wd, ht;
    dt_masks_get_image_size(&wd, &ht, NULL, NULL);
    const float xx = pzx * wd,
                yy = pzy * ht;
    if(frm && frm->functions && frm->functions->get_distance)
      frm->functions->get_distance(xx, yy, as, gui, pos, g_list_length(frm->points),
                                   &inside, &inside_border, &near, &inside_source, &dist);

    if(inside || inside_border || near >= 0 || inside_source)
    {
      if(sel_dist > dist)
      {
        sel = frm;
        sel_dist = dist;
        sel_pos = pos;
        sel_fpt = fpt;
      }
    }
    pos++;
  }

  if(sel && sel->functions)
  {
    gui->group_edited = gui->group_selected = sel_pos;
    // canvas -> list hover sync: highlight the matching row (or collapsed cluster
    // header) in the in-module mask list. Only when the hovered shape changes.
    if(gui->canvas_hover_formid != sel_fpt->formid)
    {
      gui->canvas_hover_formid = sel_fpt->formid;
      dt_iop_gui_masks_hover_form(module, sel_fpt->formid);
    }
    return sel->functions->mouse_moved(module, pzx, pzy, pressure, which, zoom_scale,
                                       sel, sel_fpt->parentid, gui, gui->group_edited);
  }

  // nothing under the cursor: drop the list row hover highlight
  if(dt_is_valid_maskid(gui->canvas_hover_formid))
  {
    gui->canvas_hover_formid = INVALID_MASKID;
    dt_iop_gui_masks_hover_form(module, INVALID_MASKID);
  }

  dt_control_queue_redraw_center();
  return 0;
}

// is this formid (or, for an AI-mask bundle child, its parent bundle's own
// formid) one the in-module panel asked us to highlight (a hovered list row,
// or every member of a hovered cluster header)? The parentid check is what
// makes hovering/selecting a bundle's single panel row highlight every one
// of its children together, the same way a real cluster header already does.
static gboolean _panel_hovered(const dt_masks_form_gui_t *gui,
                               const dt_mask_id_t formid,
                               const dt_mask_id_t parentid)
{
  for(const GList *l = gui->panel_hover_formids; l; l = g_list_next(l))
  {
    const dt_mask_id_t id = GPOINTER_TO_INT(l->data);
    if(id == formid || id == parentid) return TRUE;
  }
  return FALSE;
}

// is this formid soloed or solo-edited in the panel? Unlike the hover sync above,
// this stays true regardless of what the mouse is doing, so a soloed shape keeps
// its canvas highlight while the user works elsewhere in the list.
static gboolean _panel_soloed(const dt_masks_form_gui_t *gui,
                              const dt_mask_id_t formid,
                              const dt_mask_id_t parentid)
{
  for(const GList *l = gui->solo_formids; l; l = g_list_next(l))
  {
    const dt_mask_id_t id = GPOINTER_TO_INT(l->data);
    if(id == formid || id == parentid) return TRUE;
  }
  return FALSE;
}

void dt_group_events_post_expose(cairo_t *cr,
                                 const float zoom_scale,
                                 dt_masks_form_t *form,
                                 dt_masks_form_gui_t *gui)
{
  // base_sel is the canvas hover: the shape currently under the cursor (or -1).
  // A hovered shape always wins; the persistent panel selection is only drawn
  // when nothing at all is being hovered (no canvas hover, no list-row hover).
  const int base_sel = gui->group_selected;
  const gboolean any_list_hover = gui->panel_hover_formids != NULL;

  // if the canvas-hovered/selected entry is a child of an AI-mask bundle,
  // every sibling shares its highlight too -- the bundle is one coordinated
  // unit (see _bundle_parent_of/masks/object.c), not N independent shapes.
  dt_mask_id_t base_sel_bundle = INVALID_MASKID;
  if(base_sel >= 0)
  {
    const dt_masks_point_group_t *base_fpt = g_list_nth_data(form->points, base_sel);
    const dt_masks_form_t *bundle = base_fpt ? _bundle_parent_of(base_fpt) : NULL;
    if(bundle) base_sel_bundle = bundle->formid;
  }

  int pos = 0;
  for(GList *fpts = form->points; fpts; fpts = g_list_next(fpts))
  {
    dt_masks_point_group_t *fpt = fpts->data;
    dt_masks_form_t *sel = dt_masks_get_from_id(darktable.develop, fpt->formid);
    if(!sel) return;
    // a hidden or disabled shape draws no outline/handles on the canvas (matches the
    // renderer, which excludes it from the composite). keep pos in step with
    // gui->points by skipping only the draw call.
    if(sel->functions && !(fpt->state & (DT_MASKS_STATE_HIDDEN | DT_MASKS_STATE_DISABLE)))
    {
      // decide whether this shape draws its own highlight (feather + anchors) by
      // posing as the selected group member for the duration of its post_expose
      // call only: a hovered list row/cluster member, else -- when nothing is
      // hovered -- the persistently selected shape.
      int eff = base_sel;
      if(_panel_hovered(gui, fpt->formid, fpt->parentid))
        eff = pos;
      else if(_panel_soloed(gui, fpt->formid, fpt->parentid))
        eff = pos;
      else if(!any_list_hover && base_sel < 0
              && dt_is_valid_maskid(gui->panel_selected_formid)
              && (fpt->formid == gui->panel_selected_formid
                  || fpt->parentid == gui->panel_selected_formid))
        eff = pos;
      else if(dt_is_valid_maskid(base_sel_bundle) && fpt->parentid == base_sel_bundle)
        eff = pos;
      gui->group_selected = eff;
      sel->functions->post_expose(cr, zoom_scale, gui, pos, g_list_length(sel->points));
      gui->group_selected = base_sel;
    }
    pos++;
  }
}

static void _inverse_mask(const dt_iop_module_t *const module,
                          const dt_dev_pixelpipe_iop_t *const piece,
                          dt_masks_form_t *const form,
                          float **buffer,
                          int *width,
                          int *height,
                          int *posx,
                          int *posy)
{
  // we create a new buffer
  const int wt = piece->iwidth;
  const int ht = piece->iheight;
  float *buf = dt_alloc_align_float((size_t)ht * wt);

  // we fill this buffer
  for(int yy = 0; yy < MIN(*posy, ht); yy++)
  {
    for(int xx = 0; xx < wt; xx++) buf[(size_t)yy * wt + xx] = 1.0f;
  }

  for(int yy = MAX(*posy, 0); yy < MIN(ht, (*posy) + (*height)); yy++)
  {
    for(int xx = 0; xx < MIN((*posx), wt); xx++)
      buf[(size_t)yy * wt + xx] = 1.0f;
    for(int xx = MAX((*posx), 0); xx < MIN(wt, (*posx) + (*width)); xx++)
      buf[(size_t)yy * wt + xx] = 1.0f - (*buffer)[((size_t)yy - (*posy)) * (*width) + xx - (*posx)];
    for(int xx = MAX((*posx) + (*width), 0); xx < wt; xx++)
      buf[(size_t)yy * wt + xx] = 1.0f;
  }

  for(int yy = MAX((*posy) + (*height), 0); yy < ht; yy++)
  {
    for(int xx = 0; xx < wt; xx++) buf[(size_t)yy * wt + xx] = 1.0f;
  }

  // we free the old buffer
  dt_free_align(*buffer);
  (*buffer) = buf;

  // we return correct values for positions;
  *posx = *posy = 0;
  *width = wt;
  *height = ht;
}

int dt_masks_group_get_mask(const dt_iop_module_t *const module,
                           const dt_dev_pixelpipe_iop_t *const piece,
                           dt_masks_form_t *const form,
                           float **buffer,
                           int *width,
                           int *height,
                           int *posx,
                           int *posy)
{
  // we allocate buffers and values
  const guint nb = g_list_length(form->points);
  if(nb == 0) return 0;

  float **bufs = calloc(nb, sizeof(float *));
  int *w = malloc(sizeof(int) * nb);
  int *h = malloc(sizeof(int) * nb);
  int *px = malloc(sizeof(int) * nb);
  int *py = malloc(sizeof(int) * nb);
  int *ok = malloc(sizeof(int) * nb);
  int *states = malloc(sizeof(int) * nb);
  float *op = malloc(sizeof(float) * nb);

  // and we get all masks
  int pos = 0;
  int nb_ok = 0;
  for(GList *fpts = form->points; fpts; fpts = g_list_next(fpts))
  {
    dt_masks_point_group_t *fpt = fpts->data;
    dt_masks_form_t *sel = dt_masks_get_from_id_ext(piece->pipe->forms, fpt->formid);
    // a hidden or disabled shape contributes nothing
    if(sel && !(fpt->state & (DT_MASKS_STATE_HIDDEN | DT_MASKS_STATE_DISABLE)))
    {
      ok[pos] = dt_masks_get_mask(module, piece, sel, &bufs[pos],
                                  &w[pos], &h[pos], &px[pos], &py[pos]);
      if(fpt->state & DT_MASKS_STATE_INVERSE)
      {
        double start = dt_get_wtime();
        _inverse_mask(module, piece, sel, &bufs[pos], &w[pos], &h[pos], &px[pos], &py[pos]);
        dt_print(DT_DEBUG_MASKS | DT_DEBUG_PERF,
                 "[masks %s] inverse took %0.04f sec",
                 sel->name, dt_get_lap_time(&start));
      }
      op[pos] = fpt->opacity;
      states[pos] = fpt->state;
      if(ok[pos]) nb_ok++;
    }
    else
    {
      // hidden or missing form: takes no slot in the composite
      ok[pos] = 0;
    }
    pos++;
  }
  if(nb_ok == 0) goto error;

  // now we get the min, max, width, height of the final mask
  int l = INT_MAX, r = INT_MIN, t = INT_MAX, b = INT_MIN;
  for(int i = 0; i < nb; i++)
  {
    if(!ok[i]) continue;
    l = MIN(l, px[i]);
    t = MIN(t, py[i]);
    r = MAX(r, px[i] + w[i]);
    b = MAX(b, py[i] + h[i]);
  }
  *posx = l;
  *posy = t;
  *width = r - l;
  *height = b - t;

  // we allocate the buffer
  *buffer = dt_alloc_align_float((size_t)(r - l) * (b - t));

  // and we copy each buffer inside, row by row
  // the first *visible* shape always composites as a plain copy onto the
  // (uninitialized) buffer, whatever its explicit operator: the rendered mask
  // must match the algebra of the visible shapes only (see _group_get_mask_roi).
  gboolean first_visible = TRUE;
  for(int i = 0; i < nb; i++)
  {
    if(!ok[i]) continue;  // hidden/missing form: nothing to composite
    double start = dt_get_debug_wtime();
    if(!first_visible && (states[i] & (DT_MASKS_STATE_UNION | DT_MASKS_STATE_SUM)))
    {
      for(int y = 0; y < h[i]; y++)
      {
        for(int x = 0; x < w[i]; x++)
        {
          (*buffer)[(py[i] + y - t) * (r - l) + px[i] + x - l]
              = fmaxf((*buffer)[(py[i] + y - t) * (r - l) + px[i] + x - l],
                      bufs[i][y * w[i] + x] * op[i]);
        }
      }
    }
    else if(!first_visible && (states[i] & DT_MASKS_STATE_INTERSECTION))
    {
      for(int y = 0; y < b - t; y++)
      {
        for(int x = 0; x < r - l; x++)
        {
          const float b1 = (*buffer)[y * (r - l) + x];
          float b2 = 0.0f;
          if(y + t - py[i] >= 0
             && y + t - py[i] < h[i]
             && x + l - px[i] >= 0
             && x + l - px[i] < w[i])
            b2 = bufs[i][(y + t - py[i]) * w[i] + x + l - px[i]];
          if(b1 > 0.0f && b2 > 0.0f)
            (*buffer)[y * (r - l) + x] = fminf(b1, b2 * op[i]);
          else
            (*buffer)[y * (r - l) + x] = 0.0f;
        }
      }
    }
    else if(!first_visible && (states[i] & DT_MASKS_STATE_DIFFERENCE))
    {
      for(int y = 0; y < h[i]; y++)
      {
        for(int x = 0; x < w[i]; x++)
        {
          const float b1 = (*buffer)[(py[i] + y - t) * (r - l) + px[i] + x - l];
          const float b2 = bufs[i][y * w[i] + x] * op[i];
          if(b1 > 0.0f && b2 > 0.0f)
            (*buffer)[(py[i] + y - t) * (r - l) + px[i] + x - l] = b1 * (1.0f - b2);
        }
      }
    }
    else if(!first_visible && (states[i] & DT_MASKS_STATE_EXCLUSION))
    {
      for(int y = 0; y < h[i]; y++)
      {
        for(int x = 0; x < w[i]; x++)
        {
          const float b1 = (*buffer)[(py[i] + y - t) * (r - l) + px[i] + x - l];
          const float b2 = bufs[i][y * w[i] + x] * op[i];
          if(b1 > 0.0f && b2 > 0.0f)
            (*buffer)[(py[i] + y - t) * (r - l) + px[i] + x - l] =
              fmaxf((1.0f - b1) * b2, b1 * (1.0f - b2));
          else
            (*buffer)[(py[i] + y - t) * (r - l) + px[i] + x - l]
                = fmaxf((*buffer)[(py[i] + y - t) * (r - l) + px[i] + x - l],
                        bufs[i][y * w[i] + x] * op[i]);
        }
      }
    }
    else if(!first_visible && (states[i] & DT_MASKS_STATE_MULTIPLY))
    {
      // multiply the accumulator by this shape; outside the shape b2 = 0, so
      // the product is 0 there (iterate the whole region, like intersection).
      for(int y = 0; y < b - t; y++)
      {
        for(int x = 0; x < r - l; x++)
        {
          float b2 = 0.0f;
          if(y + t - py[i] >= 0
             && y + t - py[i] < h[i]
             && x + l - px[i] >= 0
             && x + l - px[i] < w[i])
            b2 = bufs[i][(y + t - py[i]) * w[i] + x + l - px[i]];
          (*buffer)[y * (r - l) + x] *= b2 * op[i];
        }
      }
    }
    else // if we are here, this mean that we just have to copy the shape and null other parts
    {
      for(int y = 0; y < b - t; y++)
      {
        for(int x = 0; x < r - l; x++)
        {
          float b2 = 0.0f;
          if(y + t - py[i] >= 0
             && y + t - py[i] < h[i]
             && x + l - px[i] >= 0
             && x + l - px[i] < w[i])
            b2 = bufs[i][(y + t - py[i]) * w[i] + x + l - px[i]];
          (*buffer)[y * (r - l) + x] = b2 * op[i];
        }
      }
    }

    dt_print(DT_DEBUG_MASKS | DT_DEBUG_PERF,
             "[masks %d] combine took %0.04f sec",
             i, dt_get_lap_time(&start));
    first_visible = FALSE;
  }

  free(op);
  free(states);
  free(ok);
  free(py);
  free(px);
  free(h);
  free(w);
  for(int i = 0; i < nb; i++) dt_free_align(bufs[i]);
  free(bufs);
  return 1;

error:
  free(op);
  free(states);
  free(ok);
  free(py);
  free(px);
  free(h);
  free(w);
  for(int i = 0; i < nb; i++) dt_free_align(bufs[i]);
  free(bufs);
  return 0;
}

static void _combine_masks_union(float *const restrict dest,
                                 float *const restrict newmask,
                                 const size_t npixels,
                                 const float opacity,
                                 const int inverted)
{
  if(inverted)
  {
    DT_OMP_FOR_SIMD(dt_omp_sharedconst(dest, newmask) aligned(dest, newmask : 64))
    for(size_t index = 0; index < npixels; index++)
    {
      const float mask = opacity * (1.0f - newmask[index]);
      dest[index] = MAX(dest[index], mask);
    }
  }
  else
  {
    DT_OMP_FOR_SIMD(aligned(dest, newmask : 64))
    for(size_t index = 0; index < npixels; index++)
    {
      const float mask = opacity * newmask[index];
      dest[index] = MAX(dest[index], mask);
    }
  }
}

static void _combine_masks_intersect(float *const restrict dest,
                                     float *const restrict newmask,
                                     const size_t npixels,
                                     const float opacity,
                                     const int inverted)
{
  if(inverted)
  {
    DT_OMP_FOR_SIMD(aligned(dest, newmask : 64))
    for(size_t index = 0; index < npixels; index++)
    {
      const float mask = opacity * (1.0f - newmask[index]);
      dest[index] = MIN(MAX(dest[index], 0.0f), MAX(mask, 0.0f));
    }
  }
  else
  {
    DT_OMP_FOR_SIMD(aligned(dest, newmask : 64))
    for(size_t index = 0; index < npixels; index++)
    {
      const float mask = opacity * newmask[index];
      dest[index] = MIN(MAX(dest[index], 0.0f), MAX(mask, 0.0f));
    }
  }
}

DT_OMP_DECLARE_SIMD()
static inline int both_positive(const float val1, const float val2)
{
  // this needs to be a separate inline function to convince the compiler to vectorize
  return (val1 > 0.0f) && (val2 > 0.0f);
}

static void _combine_masks_difference(float *const restrict dest,
                                      float *const restrict newmask,
                                      const size_t npixels,
                                      const float opacity,
                                      const int inverted)
{
  if(inverted)
  {
    DT_OMP_FOR_SIMD(aligned(dest, newmask : 64))
    for(size_t index = 0; index < npixels; index++)
    {
      const float mask = opacity * (1.0f - newmask[index]);
      dest[index] *= (1.0f - mask * both_positive(dest[index],mask));
    }
  }
  else
  {
    DT_OMP_FOR_SIMD(aligned(dest, newmask : 64))
    for(size_t index = 0; index < npixels; index++)
    {
      const float mask = opacity * newmask[index];
      dest[index] *= (1.0f - mask * both_positive(dest[index],mask));
    }
  }
}

static void _combine_masks_sum(float *const restrict dest,
                               float *const restrict newmask,
                               const size_t npixels,
                               const float opacity,
                               const int inverted)
{
  if(inverted)
  {
    DT_OMP_FOR_SIMD(aligned(dest, newmask : 64))
    for(size_t index = 0; index < npixels; index++)
    {
      const float mask = opacity * (1.0f - newmask[index]);
      dest[index] = MIN(1.0f, dest[index] + mask);
    }
  }
  else
  {
    DT_OMP_FOR_SIMD(aligned(dest, newmask : 64))
    for(size_t index = 0; index < npixels; index++)
    {
      const float mask = opacity * newmask[index];
      dest[index] = MIN(1.0f, dest[index] + mask);
    }
  }
}

static void _combine_masks_exclusion(float *const restrict dest,
                                     float *const restrict newmask,
                                     const size_t npixels,
                                     const float opacity,
                                     const int inverted)
{
  if(inverted)
  {
    DT_OMP_FOR_SIMD(aligned(dest, newmask : 64))
    for(size_t index = 0; index < npixels; index++)
    {
      const float mask = opacity * (1.0f - newmask[index]);
      const float pos = both_positive(dest[index], mask);
      const float neg = (1.0f - pos);
      const float b1 = dest[index];
      dest[index] = pos * MAX((1.0f - b1) * mask,
                              b1 * (1.0f - mask)) + neg * MAX(b1, mask);
    }
  }
  else
  {
    DT_OMP_FOR_SIMD(aligned(dest, newmask : 64))
    for(size_t index = 0; index < npixels; index++)
    {
      const float mask = opacity * newmask[index];
      const float pos = both_positive(dest[index], mask);
      const float neg = (1.0f - pos);
      const float b1 = dest[index];
      dest[index] = pos * MAX((1.0f - b1) * mask, b1 * (1.0f - mask)) + neg * MAX(b1, mask);
    }
  }
}

static void _combine_masks_multiply(float *const restrict dest,
                                    float *const restrict newmask,
                                    const size_t npixels,
                                    const float opacity,
                                    const int inverted)
{
  // multiply the running accumulator by this shape, the way legacy parametric
  // masks combine. Onto the empty base this is degenerate (0), so the
  // first-visible-as-add rule promotes a base multiply to a plain copy.
  if(inverted)
  {
    DT_OMP_FOR_SIMD(aligned(dest, newmask : 64))
    for(int index = 0; index < npixels; index++)
    {
      const float mask = opacity * (1.0f - newmask[index]);
      dest[index] *= mask;
    }
  }
  else
  {
    DT_OMP_FOR_SIMD(aligned(dest, newmask : 64))
    for(int index = 0; index < npixels; index++)
    {
      const float mask = opacity * newmask[index];
      dest[index] *= mask;
    }
  }
}

// soft union ("screen"): a+b-ab. Like union it is associative/commutative with
// the empty mask as identity, but it is *not* idempotent, so feathered overlaps
// build up smoothly instead of leaving the crease that max() produces. Used as
// the optional within-group combiner on the flexi group-fold path.
static void _combine_masks_screen(float *const restrict dest,
                                  float *const restrict newmask,
                                  const size_t npixels,
                                  const float opacity,
                                  const int inverted)
{
  if(inverted)
  {
    DT_OMP_FOR_SIMD(aligned(dest, newmask : 64))
    for(int index = 0; index < npixels; index++)
    {
      const float mask = opacity * (1.0f - newmask[index]);
      const float d = dest[index];
      dest[index] = d + mask - d * mask;
    }
  }
  else
  {
    DT_OMP_FOR_SIMD(aligned(dest, newmask : 64))
    for(int index = 0; index < npixels; index++)
    {
      const float mask = opacity * newmask[index];
      const float d = dest[index];
      dest[index] = d + mask - d * mask;
    }
  }
}

// Composite a finished group sub-mask into the accumulator with the group's
// own operator, exactly once (opacity/invert already baked into `grp`, so
// op=1, inverted=0). Never called for the base (bottom) group -- its own
// operator is never evaluated at all, see the base-group handling in
// _group_get_mask_roi_flexi.
static void _flexi_apply_group_op(float *const restrict buffer,
                                  float *const restrict grp,
                                  const size_t npixels,
                                  const guint group_op)
{
  if(group_op & DT_MASKS_STATE_UNION)
    _combine_masks_union(buffer, grp, npixels, 1.0f, 0);
  else if(group_op & DT_MASKS_STATE_INTERSECTION)
    _combine_masks_intersect(buffer, grp, npixels, 1.0f, 0);
  else if(group_op & DT_MASKS_STATE_DIFFERENCE)
    _combine_masks_difference(buffer, grp, npixels, 1.0f, 0);
  else if(group_op & DT_MASKS_STATE_SUM)
    _combine_masks_sum(buffer, grp, npixels, 1.0f, 0);
  else if(group_op & DT_MASKS_STATE_EXCLUSION)
    _combine_masks_exclusion(buffer, grp, npixels, 1.0f, 0);
  else if(group_op & DT_MASKS_STATE_MULTIPLY)
    _combine_masks_multiply(buffer, grp, npixels, 1.0f, 0);
  else if(group_op & DT_MASKS_STATE_OP_SCREEN)
    _combine_masks_screen(buffer, grp, npixels, 1.0f, 0);
  else
    _combine_masks_union(buffer, grp, npixels, 1.0f, 0);
}

// true iff every pixel is (within float rounding) exactly 1.0 -- a rendered
// mask member that changes nothing, used to keep a parametric channel still
// sitting at its full/base range from counting as an "active" group member
// (see the nb_members bookkeeping in _group_get_mask_roi_flexi below).
static gboolean _mask_buffer_is_uniform_one(const float *const restrict buffer,
                                            const size_t npixels)
{
  for(size_t i = 0; i < npixels; i++)
    if(buffer[i] < 0.9999f) return FALSE;
  return TRUE;
}

// Flexi group-composition fold (group-composition model, flexi masks only):
// consecutive *visible* shapes sharing one operator form a "group". A group's
// members are combined into a sub-mask by union (default) or screen — both
// order-independent, so a group is an unordered bag of shapes. That sub-mask is
// refined once (per-group refinement, stored broadcast on the members), then
// composited into the result with the group's operator a single time. An empty
// group (no visible members) contributes nothing (identity), so an empty
// intersect group never blanks the mask. The classic sequential fold below is
// left untouched, so legacy (non-flexi) masks render byte-identically.
static int _group_get_mask_roi_flexi(const dt_iop_module_t *const restrict module,
                                     const dt_dev_pixelpipe_iop_t *const restrict piece,
                                     dt_masks_form_t *const form,
                                     const dt_iop_roi_t *const roi,
                                     float *const restrict buffer)
{
  const int width = roi->width;
  const int height = roi->height;
  const size_t npixels = (size_t)width * height;

  float *const restrict bufs = dt_alloc_align_float(npixels);  // one raw shape
  float *const restrict grp  = dt_alloc_align_float(npixels);  // group sub-mask
  if(bufs == NULL || grp == NULL)
  {
    dt_free_align(bufs);
    dt_free_align(grp);
    return 0;
  }

  memset(buffer, 0, npixels * sizeof(float));

  // transient (non-serialized, flexi-only) refinement bypass: the GUI sets these
  // flags on the module's blend_data and triggers a reprocess. bypass_all skips
  // every group's refinement; bypass_cid skips just the selected group's run
  // (identified by its bottom member = run head = the group id).
  const dt_iop_gui_blend_data_t *const bd =
    module ? (const dt_iop_gui_blend_data_t *)module->blend_data : NULL;
  const gboolean bypass_all = bd && bd->refine_bypass_all;
  const dt_mask_id_t bypass_cid =
    (bd && bd->refine_bypass_group) ? bd->panel_selected_group_cid : INVALID_MASKID;

  int nb_groups = 0;  // how many groups have composited into `buffer`
  GList *fpts = form->points;
  while(fpts)
  {
    // skip hidden/absent shapes; the first usable one starts a new group
    dt_masks_point_group_t *const head = fpts->data;
    if((head->state & DT_MASKS_STATE_HIDDEN)
       || !dt_masks_get_from_id_ext(piece->pipe->forms, head->formid))
    {
      fpts = g_list_next(fpts);
      continue;
    }

    const guint group_op = head->state & DT_MASKS_STATE_OP;
    // a bypassed group is skipped whole: its members are still walked (so the
    // run boundary is found and the next group starts in the right place) but
    // none of their masks are rendered and nothing is composited, exactly as
    // if the group were not there. Its real operator is still in `group_op`,
    // untouched, so un-bypassing restores it.
    const gboolean bypassed = (group_op & DT_MASKS_STATE_OP_BYPASS) != 0;
    // within-group combine mode (how members fold together): union (default),
    // screen (soft union), intersect (min), or multiply (true per-pixel
    // product). Read from the run head, which carries the broadcast flag.
    const gboolean screen = (head->state & DT_MASKS_STATE_SCREEN) != 0;
    const gboolean isect  = (head->state & DT_MASKS_STATE_ISECT) != 0;
    const gboolean within_multiply = (head->state & DT_MASKS_STATE_WITHIN_MULTIPLY) != 0;
    // per-group refinement is broadcast onto every member, so the head carries a
    // copy. Only a GROUP-scoped one applies to the whole group -- an ELEMENT one
    // belongs to that member alone and is applied to its own mask in the fold
    // below. Reading the head unconditionally (as this used to) meant the head's
    // element refinement leaked over the entire group while every other member's
    // was dropped, and soloing a member made its own refinement work only because
    // hiding the rest promoted it to head.
    dt_masks_refinement_t group_refine = { 0 };
    if(head->refinement.enabled == DT_MASKS_REFINE_GROUP)
      group_refine = head->refinement;

    // build the group sub-mask by folding all consecutive visible members that
    // share this operator. Intersect and multiply seed at 1.0 (everything,
    // then min/multiply each member in); union/screen seed at 0.0 (nothing,
    // then max/soft-union in). (a bypassed group folds nothing into `grp`, so
    // it needs no seed either)
    if(!bypassed)
    {
      if(isect || within_multiply)
        for(size_t i = 0; i < npixels; i++) grp[i] = 1.0f;
      else
        memset(grp, 0, npixels * sizeof(float));
    }
    int nb_members = 0;  // members whose mask actually folded into `grp`
    int nb_seen = 0;     // members belonging to this run, renderable or not
    while(fpts)
    {
      dt_masks_point_group_t *const m = fpts->data;
      if(m->state & DT_MASKS_STATE_HIDDEN)
      {
        fpts = g_list_next(fpts);
        continue;
      }
      // a different operator -- or a group_start marker on a same-operator head
      // (first-class groups) -- ends this group and starts the next one. The
      // run-boundary test counts every member seen, not just the ones that
      // rendered: a bypassed group renders none of them, and even in a live
      // group an unrenderable head must not let the next group's head slip in.
      if(nb_seen > 0
         && (((m->state & DT_MASKS_STATE_OP) != group_op) || m->group_start))
        break;
      nb_seen++;
      if(bypassed || (m->state & DT_MASKS_STATE_DISABLE))  // nothing to render, just walk to the end of the run
      {
        fpts = g_list_next(fpts);
        continue;
      }
      dt_masks_form_t *const sel =
        dt_masks_get_from_id_ext(piece->pipe->forms, m->formid);
      if(!sel)
      {
        fpts = g_list_next(fpts);
        continue;
      }

      memset(bufs, 0, npixels * sizeof(float));
      if(dt_masks_get_mask_roi(module, piece, sel, roi, bufs))
      {
        // this member's own refinement, applied to its raw mask before inversion
        // and compositing -- the same point the classic renderer applies it (see
        // _group_get_mask_roi below). No-op unless this member carries one.
        gboolean elem_bypassed = FALSE;
        if(bd && bd->masks_refine_bypassed && g_hash_table_lookup(bd->masks_refine_bypassed, GUINT_TO_POINTER((guint32)m->formid)))
          elem_bypassed = TRUE;
        if(m->refinement.enabled == DT_MASKS_REFINE_ELEMENT && !elem_bypassed)
          dt_develop_blend_refine_form_mask((dt_iop_module_t *)module,
                                            (dt_dev_pixelpipe_iop_t *)piece,
                                            bufs, roi, &m->refinement);

        const float op = m->opacity;
        const int inverted = (m->state & DT_MASKS_STATE_INVERSE);
        if(isect)                _combine_masks_intersect(grp, bufs, npixels, op, inverted);
        else if(screen)          _combine_masks_screen   (grp, bufs, npixels, op, inverted);
        else if(within_multiply) _combine_masks_multiply (grp, bufs, npixels, op, inverted);
        else                     _combine_masks_union    (grp, bufs, npixels, op, inverted);
        // a parametric channel still sitting at its base/full-range state (or
        // one whose refinement scope happens to cover nothing) renders as a
        // uniform, fully-opaque buffer -- exactly a no-op, indistinguishable
        // in its effect from the member not being there at all. Checked on
        // the rendered result (after refinement, above) rather than by
        // inspecting the form's own range fields, so it also covers a
        // refinement that empties out an otherwise-narrowed channel. Not
        // counting it here means a group made up entirely of such members is
        // treated the same as a truly empty one by the nb_members==0 check
        // below, which is what lets the "no active mask element -> fully
        // opaque, no yellow overlay" fallback (see nb_groups==0 further down)
        // apply while the user is still setting up a fresh channel, instead
        // of showing a yellow wall that has nothing to do with their actual
        // (not yet narrowed) selection.
        const gboolean is_uniform_noop =
          (sel->type & DT_MASKS_PARAMETRIC) && _mask_buffer_is_uniform_one(bufs, npixels);
        if(!is_uniform_noop) nb_members++;
      }
      fpts = g_list_next(fpts);
    }

    if(bypassed) continue;         // disabled group → contributes nothing
    if(nb_members == 0) continue;  // empty group → identity

    // per-group refinement, applied once to the finished sub-mask (skipped when
    // this group -- or every group -- is bypassed for preview)
    const gboolean group_bypassed =
      bypass_all || (dt_is_valid_maskid(bypass_cid) && head->formid == bypass_cid)
      || (bd && bd->masks_refine_bypassed && g_hash_table_lookup(bd->masks_refine_bypassed, GUINT_TO_POINTER((guint32)head->formid | 0x80000000U)));
    if(group_refine.enabled && !group_bypassed)
      dt_develop_blend_refine_form_mask((dt_iop_module_t *)module,
                                        (dt_dev_pixelpipe_iop_t *)piece,
                                        grp, roi, &group_refine);

    // invert-output (true group invert, see DT_MASKS_STATE_OP_INVERT):
    // applied to this run's finished sub-mask, after its members have folded
    // and any group refinement has run, but before it composites onto the
    // accumulator below -- so a difference-op group seeding the accumulator
    // (the `continue` case right below) also seeds it already inverted.
    if(group_op & DT_MASKS_STATE_OP_INVERT)
      for(size_t i = 0; i < npixels; i++) grp[i] = 1.0f - grp[i];

    // group-level opacity (see dt_masks_point_group_t.group_opacity): a
    // persistent, multiplicative gain on this run's own finished sub-mask,
    // applied on top of -- not instead of -- each member's own independent
    // opacity (already folded into `grp` above; the two multiply together).
    // Read from the head, same convention as every other broadcast run-level
    // field (state/refinement/name). Applied after invert-output, for the
    // same reason element opacity multiplies a shape's already-inverted mask
    // in _combine_masks_union et al: it scales the run's actual finished
    // contribution, whatever its state, not some pre-invert intermediate.
    for(size_t i = 0; i < npixels; i++) grp[i] *= head->group_opacity;

    if(nb_groups == 0)
    {
      // the base group has no predecessor to combine with, so its own
      // operator is never evaluated: its finished sub-mask becomes the
      // initial accumulator directly, whatever operator happens to be shown
      // on it (every operator's own identity element reduces to exactly this
      // anyway -- union/sum/exclusion/screen from empty and
      // intersect/multiply from full all equal `grp` unchanged; only
      // difference has no identity element at all, so this is also its
      // fallback). Invert the group (or its members) for the complement
      // instead.
      memcpy(buffer, grp, npixels * sizeof(float));
    }
    else
    {
      _flexi_apply_group_op(buffer, grp, npixels, group_op);
    }
    nb_groups++;
  }

  if(nb_groups == 0)
  {
    // no group actually contributed anything (every group hidden, bypassed,
    // or member-less) -- this must render as "no active mask element", which
    // in dt is a fully opaque mask (the module stays 100% active), matching
    // the `mode_drawn && !form` fallback in dt_develop_blend. Leaving
    // `buffer` at its initial all-zero state here would instead silently
    // disable the module, which is not classic's convention.
    for(size_t i = 0; i < npixels; i++) buffer[i] = 1.0f;
  }

  dt_free_align(bufs);
  dt_free_align(grp);
  return nb_groups != 0;
}

int dt_masks_group_get_mask_roi(const dt_iop_module_t *const restrict module,
                               const dt_dev_pixelpipe_iop_t *const restrict piece,
                               dt_masks_form_t *const form,
                               const dt_iop_roi_t *const roi,
                               float *const restrict buffer)
{
  if(!form->points) return 0;

  // flexi masks use the group-composition fold; legacy masks fall through to
  // the classic sequential fold below, byte-identically.
  const dt_develop_blend_params_t *const bp =
    piece ? (const dt_develop_blend_params_t *)piece->blendop_data : NULL;
  if(bp && (bp->mask_mode & DEVELOP_MASK_FLEXI))
    return _group_get_mask_roi_flexi(module, piece, form, roi, buffer);

  double start = dt_get_debug_wtime();
  int nb_ok = 0;

  const int width = roi->width;
  const int height = roi->height;
  const size_t npixels = (size_t)width * height;

  // we need to allocate a zeroed temporary buffer for intermediate
  // creation of individual shapes
  float *const restrict bufs = dt_alloc_align_float(npixels);
  if(bufs == NULL) return 0;

  // start from an empty result so a hidden/absent base form does not leave the
  // first composited shape reading uninitialized memory
  memset(buffer, 0, npixels * sizeof(float));

  // and we get all masks
  for(GList *fpts = form->points; fpts; fpts = g_list_next(fpts))
  {
    dt_masks_point_group_t *fpt = fpts->data;
    // a hidden or disabled shape contributes nothing
    if(fpt->state & (DT_MASKS_STATE_HIDDEN | DT_MASKS_STATE_DISABLE)) continue;
    dt_masks_form_t *sel = dt_masks_get_from_id_ext(piece->pipe->forms, fpt->formid);

    if(sel)
    {
      // ensure that we start with a zeroed buffer regardless of what
      // was previously written into 'bufs'
      memset(bufs, 0, npixels*sizeof(float));
      const int ok = dt_masks_get_mask_roi(module, piece, sel, roi, bufs);
      const float op = fpt->opacity;
      const int state = fpt->state;

      if(darktable.dump_pfm_module)
      {
        char *filename = g_strdup_printf("mask-%d", fpt->formid);
        dt_dump_pfm(filename,
                    bufs,
                    width,
                    height,
                    sizeof(float),
                    module->op);
        g_free(filename);
      }

      if(ok)
      {
        // optional per-shape refinement, applied to the raw shape mask before
        // inversion and compositing. No-op (and zero cost) unless this shape
        // has refinement enabled, so existing masks render unchanged.
        if(fpt->refinement.enabled)
          dt_develop_blend_refine_form_mask((dt_iop_module_t *)module,
                                            (dt_dev_pixelpipe_iop_t *)piece,
                                            bufs, roi, &fpt->refinement);

        // first see if we need to invert this shape
        const int inverted = (state & DT_MASKS_STATE_INVERSE);

        // the first *visible* shape always composites as ADD onto the empty
        // accumulator, whatever its explicit operator says: the rendered mask
        // must match the algebra of the visible shapes only. e.g. hiding the
        // base promotes the next visible shape to the implicit base (add),
        // so [add][intersect][union] with the first two hidden renders as the
        // third shape alone. (nb_ok == 0 means nothing has composited yet.)
        if(nb_ok == 0 || (state & DT_MASKS_STATE_UNION))
        {
          _combine_masks_union(buffer, bufs, npixels, op, inverted);
        }
        else if(state & DT_MASKS_STATE_INTERSECTION)
        {
          _combine_masks_intersect(buffer, bufs, npixels, op, inverted);
        }
        else if(state & DT_MASKS_STATE_DIFFERENCE)
        {
          _combine_masks_difference(buffer, bufs, npixels, op, inverted);
        }
        else if(state & DT_MASKS_STATE_SUM)
        {
          _combine_masks_sum(buffer, bufs, npixels, op, inverted);
        }
        else if(state & DT_MASKS_STATE_EXCLUSION)
        {
          _combine_masks_exclusion(buffer, bufs, npixels, op, inverted);
        }
        else if(state & DT_MASKS_STATE_MULTIPLY)
        {
          _combine_masks_multiply(buffer, bufs, npixels, op, inverted);
        }
        else // if we are here, this mean that we just have to copy
             // the shape and null other parts
        {
          DT_OMP_FOR_SIMD(aligned(buffer, bufs : 64))
          for(size_t index = 0; index < npixels; index++)
          {
            buffer[index] = op * (inverted ? (1.0f - bufs[index]) : bufs[index]);
          }
        }

        dt_print(DT_DEBUG_MASKS | DT_DEBUG_PERF,
                 "[masks %d] combine took %0.04f sec",
                 nb_ok, dt_get_lap_time(&start));

        nb_ok++;
      }
    }

    if(darktable.dump_pfm_module)
    {
      char *filename = g_strdup_printf("mask-combined-%d", fpt->formid);
      dt_dump_pfm(filename,
                  buffer,
                  width,
                  height,
                  sizeof(float),
                  module->op);
      g_free(filename);
    }
  }
  // and we free the intermediate buffer
  dt_free_align(bufs);

  return nb_ok != 0;
}

int dt_masks_group_render_roi(dt_iop_module_t *module,
                              dt_dev_pixelpipe_iop_t *piece,
                              dt_masks_form_t *form,
                              const dt_iop_roi_t *roi,
                              float *buffer)
{
  if(!form) return 0;

  double start = dt_get_debug_wtime();
  const int ok = dt_masks_get_mask_roi(module, piece, form, roi, buffer);

  dt_print(DT_DEBUG_MASKS | DT_DEBUG_PERF,
           "[masks] render all masks took %0.04f sec",
           dt_get_lap_time(&start));
  return ok;
}

static GSList *_group_setup_mouse_actions(const dt_masks_form_t *const form)
{
  GSList *lm = NULL;
  // initialize the mask of seen shapes to the set of flags which
  // aren't actually shapes
  dt_masks_type_t seen_types = (DT_MASKS_GROUP | DT_MASKS_CLONE | DT_MASKS_NON_CLONE);
  // iterate over the shapes in the group, adding the mouse_action for
  // each distinct type of shape

  for(GList *fpts = form->points; fpts; fpts = g_list_next(fpts))
  {
    dt_masks_point_group_t *fpt = fpts->data;
    dt_masks_form_t *sel = dt_masks_get_from_id(darktable.develop, fpt->formid);
    if(!sel || (sel->type & ~seen_types) == 0)
      continue;
    if(sel->functions && sel->functions->setup_mouse_actions)
    {
      GSList *new_actions = sel->functions->setup_mouse_actions(sel);
      lm = g_slist_concat(lm, new_actions);
      seen_types |= sel->type;
    }
  }
  return lm;
}

void dt_masks_group_duplicate_points(dt_develop_t *const dev,
                                     dt_masks_form_t *const base,
                                     dt_masks_form_t *const dest)
{
  for(GList *pts = base->points; pts; pts = g_list_next(pts))
  {
    dt_masks_point_group_t *pt = pts->data;
    dt_masks_point_group_t *npt = calloc(1, sizeof(dt_masks_point_group_t));

    npt->formid = dt_masks_form_duplicate(dev, pt->formid);
    npt->parentid = dest->formid;
    npt->state = pt->state;
    npt->opacity = pt->opacity;
    npt->refinement = pt->refinement;
    npt->group_opacity = pt->group_opacity;
    dest->points = g_list_append(dest->points, npt);
  }
}

// The function table for groups.  This must be public, i.e. no "static" keyword.
const dt_masks_functions_t dt_masks_functions_group = {
  .point_struct_size = sizeof(struct dt_masks_point_group_t),
  .sanitize_config = NULL,
  .setup_mouse_actions = _group_setup_mouse_actions,
  .set_form_name = NULL,
  .set_hint_message = NULL,
  .duplicate_points = dt_masks_group_duplicate_points,
  .initial_source_pos = NULL,
  .get_distance = NULL,
  .get_points = NULL,
  .get_points_border = NULL,
  .get_mask = dt_masks_group_get_mask,
  .get_mask_roi = dt_masks_group_get_mask_roi,
  .get_area = NULL,
  .get_source_area = NULL,
  .mouse_moved = _group_events_mouse_moved,
  .mouse_scrolled = _group_events_mouse_scrolled,
  .button_pressed = _group_events_button_pressed,
  .button_released = _group_events_button_released,
//TODO:  .post_expose = _group_events_post_expose
};


// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
