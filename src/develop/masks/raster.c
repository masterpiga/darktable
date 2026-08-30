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
#include "develop/blend.h"
#include "develop/imageop.h"
#include "develop/masks.h"
#include "develop/pixelpipe.h"

#include <float.h>

/* A raster mask (another module's output mask) as a first-class drawn-mask
 * form.
 *
 * A raster form (dt_masks_point_raster_t) references an upstream module's
 * output mask by (op, instance, id). Placing it in a module's mask group lets
 * the raster be composited as an element -- combined with shapes and
 * parametric channels by the usual operators (union/intersection/...), exactly
 * like any other element -- instead of being an exclusive whole-mask mode.
 *
 * This is purely additive: the module's built-in exclusive raster mode (mask
 * mode RASTER) and its UI are untouched, so existing edits render identically.
 * A raster form only ever exists in edits created after this feature.
 *
 * Rendering reuses the pipe's existing raster-mask fetch/distortion machinery
 * (dt_dev_get_raster_mask), which returns the source mask already distorted to
 * the requesting module's output roi. Because the group is rendered on the CPU
 * even in the OpenCL pipe, a raster element works identically on GPU.
 *
 * The dependency (so each source module stores its mask and the pipe orders
 * correctly) is registered by _reconcile_raster_form_users() in imageop.c, at
 * commit_params time -- so it also takes effect on edit reload, with no GUI
 * action. It is per-element: a module may hold several raster elements, each
 * naming a different upstream source. It is deliberately independent of
 * blend_params.raster_mask_*, which stays reserved for the exclusive whole-mask
 * RASTER mode; nothing writes those fields on behalf of a raster form, and
 * module->raster_mask.sink.source is therefore NOT this form's source. */

static void _raster_set_form_name(dt_masks_form_t *const form, const size_t nb)
{
  // prefix must match _form_type_prefix / _kind_name(DT_MASKS_RASTER) in
  // blend_gui.c so the mask-list row strips it cleanly from the display name
  snprintf(form->name, sizeof(form->name), "%s #%d", _("raster mask"), (int)nb);
}

static GSList *_raster_setup_mouse_actions(const dt_masks_form_t *const form)
{
  // no canvas interaction; configured from the side panel
  return NULL;
}

/* A raster form has no on-canvas geometry, but the group event/expose
 * dispatchers (src/develop/masks/group.c) call some vtable entries on every
 * group member without a per-function NULL check. We therefore provide explicit
 * no-op stubs rather than leaving those slots NULL, so a raster form can sit in
 * a shown group without crashing. The form is never the "closest" form
 * (get_distance returns a huge distance), so it is never picked for direct
 * interaction. Mirrors the parametric form (see parametric.c). */

static void _raster_post_expose(cairo_t *const cr,
                                const float zoom_scale,
                                dt_masks_form_gui_t *const gui,
                                const int nb,
                                const int index)
{
  // nothing to draw
}

static void _raster_get_distance(const float x,
                                 const float y,
                                 const float as,
                                 dt_masks_form_gui_t *const gui,
                                 const int index,
                                 const int num_points,
                                 gboolean *const inside,
                                 gboolean *const inside_border,
                                 int *const near,
                                 gboolean *const inside_source,
                                 float *const dist)
{
  // never the closest form to the pointer
  if(inside) *inside = FALSE;
  if(inside_border) *inside_border = FALSE;
  if(near) *near = -1;
  if(inside_source) *inside_source = FALSE;
  if(dist) *dist = FLT_MAX;
}

static int _raster_get_points_border(dt_develop_t *const dev,
                                     dt_masks_form_t *const form,
                                     float **const points,
                                     int *const points_count,
                                     float **const border,
                                     int *const border_count,
                                     const int source,
                                     const dt_iop_module_t *const module)
{
  // no geometric outline
  return 0;
}

static void _raster_duplicate_points(dt_develop_t *const dev,
                                     dt_masks_form_t *const base,
                                     dt_masks_form_t *const dest)
{
  for(GList *pts = base->points; pts; pts = g_list_next(pts))
  {
    dt_masks_point_raster_t *p = malloc(sizeof(dt_masks_point_raster_t));
    memcpy(p, pts->data, sizeof(dt_masks_point_raster_t));
    dest->points = g_list_append(dest->points, p);
  }
}

// resolve the form's source (op + instance) to a live module in the pipe
static dt_iop_module_t *_raster_resolve_source(const dt_iop_module_t *const module,
                                               const dt_masks_point_raster_t *const p)
{
  if(!module || !module->dev || !p->source[0]) return NULL;
  for(GList *iter = module->dev->iop; iter; iter = g_list_next(iter))
  {
    dt_iop_module_t *iop = iter->data;
    if(dt_iop_module_is(iop, p->source) && iop->multi_priority == p->instance) return iop;
  }
  return NULL;
}

/* An unresolvable raster element renders as all-zero, and reports success.
 *
 * The distinction matters more than it looks. Returning 0 means "this member
 * did not render", and _group_get_mask_roi_flexi() then does not count it --
 * so a group whose only member is an unresolvable raster comes out with
 * nb_members == 0, which trips the deliberate "no active mask element"
 * fallback in dt_develop_blend_process() and fills the mask with 1.0. The
 * module would apply at full strength across the whole image.
 *
 * That fallback is right for what it was written for (a group the user is
 * still building, where a yellow wall would hide the image they are placing
 * shapes on). It is wrong here: the classic renderer's raster branch fills
 * 0.0f when dt_dev_get_raster_mask() hands back NULL, so a raster mask whose
 * source is gone means the module contributes *nothing*. Rendering zero and
 * counting the member keeps that, and keeps it for the case migration cannot
 * see either -- a source module deleted after the fact, which resolves fine
 * today and not tomorrow.
 *
 * Found by replaying real edits: 5 in the harvested corpus carry mask mode
 * RASTER with an empty source (a source module removed at some point), and
 * every one of them flipped from "module does nothing" to "module applies
 * everywhere". */
static int _raster_unresolved(float *const buffer, const dt_iop_roi_t *const roi)
{
  memset(buffer, 0, (size_t)roi->width * roi->height * sizeof(float));
  return 1;
}

gboolean dt_masks_raster_is_unresolved(const dt_iop_module_t *module,
                                       const dt_dev_pixelpipe_iop_t *piece,
                                       const dt_masks_form_t *form)
{
  if(!form || !(form->type & DT_MASKS_RASTER) || !form->points) return FALSE;

  const dt_iop_module_t *source = _raster_resolve_source(module, form->points->data);
  if(!source) return TRUE;

  // Whether the source is on. Inside a pipe its piece is the authority --
  // module->enabled is not maintained in an export pipe (a source that is on
  // for the darkroom can read as off there, and vice versa), so a piece-less
  // check would answer for the wrong pipe.
  gboolean enabled = source->enabled;
  if(piece && piece->pipe)
  {
    const dt_dev_pixelpipe_iop_t *source_piece = NULL;
    for(GList *n = piece->pipe->nodes; n; n = g_list_next(n))
    {
      const dt_dev_pixelpipe_iop_t *cand = n->data;
      if(cand->module == source)
      {
        source_piece = cand;
        break;
      }
    }
    // in this pipe the module does not exist at all
    if(!source_piece) return TRUE;
    enabled = source_piece->enabled;
  }
  if(!enabled) return TRUE;

  // ...and whether it publishes anything. An enabled module with no mask of its
  // own and no IOP_FLAGS_WRITE_RASTER never puts a mask in the table, so this
  // element has nothing to read however healthy the reference looks. Same test
  // dt_dev_get_raster_mask() makes before it gives up (pixelpipe_hb.c).
  const dt_develop_mask_mode_t mask_mode =
    source->blend_params ? source->blend_params->mask_mode : DEVELOP_MASK_DISABLED;
  const gboolean writes_masks = (mask_mode > DEVELOP_MASK_ENABLED)
                             || (source->flags() & IOP_FLAGS_WRITE_RASTER);
  return !writes_masks;
}

static int _raster_get_mask_roi(const dt_iop_module_t *const module,
                                const dt_dev_pixelpipe_iop_t *const piece,
                                dt_masks_form_t *const form,
                                const dt_iop_roi_t *const roi,
                                float *const buffer)
{
  if(!form->points) return 0;
  const dt_masks_point_raster_t *const p = form->points->data;

  dt_iop_module_t *source = _raster_resolve_source(module, p);
  if(!source)
  {
    dt_print(DT_DEBUG_MASKS, "[masks] raster form %d: source '%s' not found in pipe",
             form->formid, p->source);
    return _raster_unresolved(buffer, roi);
  }

  // dt_dev_get_raster_mask returns the source mask already distorted to the
  // requesting piece's output roi (== the group render roi here), or NULL if
  // the mask is not (yet) available. free_mask tells us whether the buffer was
  // freshly allocated (distorted) and must be released.
  gboolean free_mask = FALSE;
  float *raster = dt_dev_get_raster_mask((dt_dev_pixelpipe_iop_t *)piece, source, p->id,
                                         module, &free_mask);
  if(!raster)
  {
    dt_print(DT_DEBUG_MASKS, "[masks] raster form %d: no raster mask from '%s' id=%d",
             form->formid, p->source, p->id);
    return _raster_unresolved(buffer, roi);
  }

  const size_t npix = (size_t)roi->width * roi->height;
  // hand the raw 0..1 mask to the group compositor; opacity and the per-element
  // invert (DT_MASKS_STATE_INVERSE) are applied there like any other element
  memcpy(buffer, raster, npix * sizeof(float));

  if(free_mask) dt_free_align(raster);
  return 1;
}

// The function table for raster masks. Most geometric/mouse callbacks are
// unused; the form is a pure pixel reference to another module's mask.
const dt_masks_functions_t dt_masks_functions_raster = {
  .point_struct_size = sizeof(struct dt_masks_point_raster_t),
  .sanitize_config = NULL,
  .setup_mouse_actions = _raster_setup_mouse_actions,
  .set_form_name = _raster_set_form_name,
  .set_hint_message = NULL,
  .modify_property = NULL,
  .duplicate_points = _raster_duplicate_points,
  .initial_source_pos = NULL,
  .get_distance = _raster_get_distance,
  .get_points = NULL,
  .get_points_border = _raster_get_points_border,
  .get_mask = NULL,
  .get_mask_roi = _raster_get_mask_roi,
  .get_area = NULL,
  .get_source_area = NULL,
  .mouse_moved = NULL,
  .mouse_scrolled = NULL,
  .button_pressed = NULL,
  .button_released = NULL,
  .post_expose = _raster_post_expose
};
