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

#include <float.h>

/* A parametric (blendif) mask as a first-class drawn-mask form.
 *
 * A parametric form carries its own copy of the blendif channel configuration
 * (dt_masks_point_parametric_t), so several independent parametric masks can
 * live in one module's mask group and be combined with the usual operators
 * (union/intersection/difference/sum/exclusion/inverse), exactly like shapes.
 *
 * This is purely additive: the module's single built-in parametric mask and
 * its UI are untouched, so existing edits render identically. A parametric
 * form only ever exists in edits created after this feature.
 *
 * Rendering reuses the existing per-colorspace make_mask functions by
 * temporarily pointing the piece's blendop data at this form's blendif config.
 * Those functions read the module's input/output image, which is exposed to
 * the group renderer through the transient blend_refine_* context on piece
 * (set in dt_develop_blend_process). In the OpenCL pipe the images live on the
 * device and are not exposed here, so the parametric mask falls back to fully
 * opaque (mask = 1) on GPU -- a known limitation shared with per-shape
 * feathering. */

// localized channel label for a single-channel form ("Lightness", "chroma", ...),
// or a generic fallback for the legacy multi-channel form (no single-channel
// binding, so it has no one channel to name itself after). Exported (see
// blend.h) so the mask-list rename UI (blend_gui.c) can recompute this same
// prefix from the form's type every time, rather than parsing it out of
// form->name -- that keeps the prefix stable across repeated renames.
const char *dt_masks_parametric_type_label(const dt_masks_form_t *const form)
{
  const dt_masks_point_parametric_t *p = form->points ? form->points->data : NULL;
  if(p && p->single)
  {
    const dt_iop_gui_blendif_channel_t *channels =
      dt_develop_blendif_channels_for_csp((int)p->colorspace);
    int nch = 0;
    if(channels) while(channels[nch].label) nch++;
    if(channels && (int)p->channel < nch)
      return _(channels[p->channel].label);
  }
  return _("parametric");
}

static void _parametric_set_form_name(dt_masks_form_t *const form, const size_t nb)
{
  // a single-channel form's name leads with its channel -- "parametric" in
  // front of that is redundant, unlike the legacy multi-channel form
  snprintf(form->name, sizeof(form->name), "%s #%d",
           dt_masks_parametric_type_label(form), (int)nb);
}

static GSList *_parametric_setup_mouse_actions(const dt_masks_form_t *const form)
{
  // no canvas interaction; configured from the side panel
  return NULL;
}

/* A parametric form has no on-canvas geometry, but the group event/expose
 * dispatchers (src/develop/masks/group.c) call some vtable entries on every
 * group member without a per-function NULL check. We therefore provide explicit
 * no-op stubs rather than leaving those slots NULL, so a parametric form can
 * sit in a shown group without crashing. The form is never the "closest" form
 * (get_distance returns a huge distance), so it is never picked for direct
 * interaction. */

static void _parametric_post_expose(cairo_t *const cr,
                                    const float zoom_scale,
                                    dt_masks_form_gui_t *const gui,
                                    const int nb,
                                    const int index)
{
  // nothing to draw
}

static void _parametric_get_distance(const float x,
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

static int _parametric_get_points_border(dt_develop_t *const dev,
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

static void _parametric_duplicate_points(dt_develop_t *const dev,
                                         dt_masks_form_t *const base,
                                         dt_masks_form_t *const dest)
{
  for(GList *pts = base->points; pts; pts = g_list_next(pts))
  {
    dt_masks_point_parametric_t *p = malloc(sizeof(dt_masks_point_parametric_t));
    memcpy(p, pts->data, sizeof(dt_masks_point_parametric_t));
    dest->points = g_list_append(dest->points, p);
  }
}

static int _parametric_get_mask_roi(const dt_iop_module_t *const module,
                                    const dt_dev_pixelpipe_iop_t *const piece,
                                    dt_masks_form_t *const form,
                                    const dt_iop_roi_t *const roi,
                                    float *const buffer)
{
  if(!form->points) return 0;
  const dt_masks_point_parametric_t *const p = form->points->data;

  const size_t npix = (size_t)roi->width * roi->height;
  // start fully opaque; make_mask multiplies the channel response into it
  for(size_t k = 0; k < npix; k++) buffer[k] = 1.0f;

  // guide images come from the transient blend context (CPU pipe only)
  const float *const a = piece->blend_refine_guide_in;
  const float *const b = piece->blend_refine_guide_out;
  if(!a || !b)
  {
    dt_print(DT_DEBUG_MASKS,
             "[masks] parametric form %d: no guide image available, mask = 1",
             form->formid);
    return 1;
  }

  // temporarily expose this form's blendif config as the piece's blendop data,
  // with full opacity (the group compositor applies the form opacity later).
  dt_dev_pixelpipe_iop_t *const pc = (dt_dev_pixelpipe_iop_t *)piece;
  dt_develop_blend_params_t *const saved = pc->blendop_data;
  if(!saved) return 1;

  dt_develop_blend_params_t tmp = *saved;
  tmp.blendif = p->blendif;
  memcpy(tmp.blendif_parameters, p->blendif_parameters,
         sizeof(tmp.blendif_parameters));
  memcpy(tmp.blendif_boost_factors, p->blendif_boost_factors,
         sizeof(tmp.blendif_boost_factors));
  tmp.opacity = 100.0f;
  // the blendif make_mask functions short-circuit to a uniform mask unless the
  // blend mode advertises a parametric mask. A parametric *form* is evaluated
  // while the module is in flexi/drawn mode (no CONDITIONAL bit), so force the
  // bit on for this scratch copy; otherwise every form would render as opaque.
  tmp.mask_mode = DEVELOP_MASK_ENABLED | DEVELOP_MASK_CONDITIONAL;
  pc->blendop_data = &tmp;

  const dt_iop_roi_t *const rin =
    piece->blend_refine_roi_in ? piece->blend_refine_roi_in : roi;
  const dt_iop_roi_t *const rout =
    piece->blend_refine_roi_out ? piece->blend_refine_roi_out : roi;

  dt_print(DT_DEBUG_MASKS,
           "[masks] parametric form %d: make_mask csp=%d blendif=0x%x",
           form->formid, saved->blend_cst, p->blendif);

  switch(saved->blend_cst)
  {
    case DEVELOP_BLEND_CS_LAB:
      dt_develop_blendif_lab_make_mask(pc, a, b, rin, rout, buffer);
      break;
    case DEVELOP_BLEND_CS_RGB_DISPLAY:
      dt_develop_blendif_rgb_hsl_make_mask(pc, a, b, rin, rout, buffer);
      break;
    case DEVELOP_BLEND_CS_RGB_SCENE:
      dt_develop_blendif_rgb_jzczhz_make_mask(pc, a, b, rin, rout, buffer);
      break;
    case DEVELOP_BLEND_CS_RAW:
      dt_develop_blendif_raw_make_mask(pc, a, b, rin, rout, buffer);
      break;
    default:
      break;
  }

  pc->blendop_data = saved;
  return 1;
}

// The function table for parametric masks. Most geometric/mouse callbacks are
// unused; the form is evaluated purely from pixel values.
const dt_masks_functions_t dt_masks_functions_parametric = {
  .point_struct_size = sizeof(struct dt_masks_point_parametric_t),
  .sanitize_config = NULL,
  .setup_mouse_actions = _parametric_setup_mouse_actions,
  .set_form_name = _parametric_set_form_name,
  .set_hint_message = NULL,
  .modify_property = NULL,
  .duplicate_points = _parametric_duplicate_points,
  .initial_source_pos = NULL,
  .get_distance = _parametric_get_distance,
  .get_points = NULL,
  .get_points_border = _parametric_get_points_border,
  .get_mask = NULL,
  .get_mask_roi = _parametric_get_mask_roi,
  .get_area = NULL,
  .get_source_area = NULL,
  .mouse_moved = NULL,
  .mouse_scrolled = NULL,
  .button_pressed = NULL,
  .button_released = NULL,
  .post_expose = _parametric_post_expose
};

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
