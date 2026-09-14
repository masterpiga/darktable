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

#include "develop/masks.h"
#include "bauhaus/bauhaus.h"
#include "common/debug.h"
#include "control/conf.h"
#include "control/control.h"
#include "develop/blend.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"

#pragma GCC diagnostic ignored "-Wshadow"

dt_masks_form_t *dt_masks_dup_masks_form(const dt_masks_form_t *form)
{
  if(!form) return NULL;

  dt_masks_form_t *new_form = malloc(sizeof(struct dt_masks_form_t));
  memcpy(new_form, form, sizeof(struct dt_masks_form_t));

  // then duplicate the GList *points

  GList* newpoints = NULL;

  if(form->points)
  {
    const int size_item = (form->functions) ? form->functions->point_struct_size : 0;

    if(size_item != 0)
    {
      for(GList *pt = form->points; pt; pt = g_list_next(pt))
      {
        void *item = malloc(size_item);
        memcpy(item, pt->data, size_item);
        newpoints = g_list_prepend(newpoints, item);
      }
    }
  }
  // list was built in reverse order, so un-reverse it
  new_form->points = g_list_reverse(newpoints);

  return new_form;
}

static void *_dup_masks_form_cb(const void *formdata, const gpointer user_data)
{
  // duplicate the main form struct
  const dt_masks_form_t *form = (dt_masks_form_t *)formdata;
  const dt_masks_form_t *uform = (dt_masks_form_t *)user_data;
  const dt_masks_form_t *f = uform == NULL || form->formid != uform->formid ? form : uform;
  return (void *)dt_masks_dup_masks_form(f);
}

// duplicate the list of forms, replace item in the list with form with the same formid
GList *dt_masks_dup_forms_deep(GList *forms,
                               dt_masks_form_t *form)
{
  return (GList *)g_list_copy_deep(forms, _dup_masks_form_cb, (gpointer)form);
}

static int _get_opacity(const dt_masks_form_gui_t *gui,
                        const dt_masks_form_t *form)
{
  const dt_masks_point_group_t *fpt = g_list_nth_data(form->points, gui->group_edited);
  const dt_masks_form_t *sel = dt_masks_get_from_id(darktable.develop, fpt->formid);
  if(!sel) return 0;

  const dt_mask_id_t formid = sel->formid;

  // look for apacity
  const dt_masks_form_t *grp = dt_masks_get_from_id(darktable.develop, fpt->parentid);
  if(!grp || !(grp->type & DT_MASKS_GROUP)) return 0;

  int opacity = 0;
  for(GList *fpts = grp->points; fpts; fpts = g_list_next(fpts))
  {
    const dt_masks_point_group_t *fpt = fpts->data;
    if(fpt->formid == formid)
    {
      opacity = fpt->opacity * 100;
      break;
    }
  }

  return opacity;
}

static dt_masks_type_t _get_all_types_in_group(const dt_masks_form_t *form,
                                               const int depth)
{
  // a marker resolves to no form
  if(!form || depth > DT_MASKS_NESTING_MAX) return 0;
  if(form->type & (DT_MASKS_GROUP | DT_MASKS_OBJECT))
  {
    dt_masks_type_t tp = 0;
    for(GList *l = form->points; l; l = g_list_next(l))
    {
      const dt_masks_point_group_t *pt = l->data;
      const dt_masks_form_t *f = dt_masks_get_from_id(darktable.develop, pt->formid);
      tp |= _get_all_types_in_group(f, depth + 1);
    }
    return tp;
  }
  else
  {
    return form->type;
  }
}

GSList *dt_masks_mouse_actions(const dt_masks_form_t *form)
{
  const dt_masks_type_t formtype = _get_all_types_in_group(form, 0);
  GSList *lm = NULL;

  if(form->functions && form->functions->setup_mouse_actions)
  {
    lm = form->functions->setup_mouse_actions(form);
  }
  // add the common action(s) shared by all shapes
  if(formtype != 0)
  {
    lm = dt_mouse_action_create_simple(lm, DT_MOUSE_ACTION_RIGHT, 0,
                                       _("[SHAPE] remove shape"));
  }

  return lm;
}

static void _set_hinter_message(const dt_masks_form_gui_t *gui,
                                const dt_masks_form_t *form)
{
  char msg[512] = "";

  const int ftype = form->type;

  int opacity = 100;

  const dt_masks_form_t *sel = form;
  const dt_masks_point_group_t *fpt = NULL;
  if((ftype & DT_MASKS_GROUP) && (gui->group_edited >= 0))
  {
    // we get the selected form
    fpt = g_list_nth_data(form->points, gui->group_edited);
    sel = dt_masks_get_from_id(darktable.develop, fpt->formid);
    if(!sel) return;

    opacity = _get_opacity(gui, form);
  }
  else
  {
    opacity = (int)(dt_conf_get_float("plugins/darkroom/masks/opacity") * 100);
  }

  if(sel->functions && sel->functions->set_hint_message)
  {
    // pass the selected sub-form (sel), not the outer form: when editing a
    // group member, `form` is the group, so its type/points would not describe
    // the shape the hint is about (e.g. whether it is a clone with a source)
    sel->functions->set_hint_message(gui, sel, opacity, msg, sizeof(msg));
  }

  // a path of an AI object: say how its paths are reached one by one
  const dt_masks_form_t *object =
    fpt ? dt_masks_get_from_id(darktable.develop, fpt->parentid) : NULL;
  if(object && (object->type & DT_MASKS_OBJECT))
  {
    if(msg[0]) g_strlcat(msg, "\n", sizeof(msg));
    g_strlcat(msg,
              gui->entered_object == object->formid
                ? _("inside the AI object: its paths are edited and removed one by one,"
                    " click outside it to leave")
                : _("double-click to edit and remove the AI object's paths one by one"),
              sizeof(msg));
  }

  dt_control_hinter_message(msg);
}

void dt_masks_init_form_gui(dt_masks_form_gui_t *gui)
{
  memset(gui, 0, sizeof(dt_masks_form_gui_t));

  gui->posx = gui->posy = -1.0f;
  gui->posx_source = gui->posy_source = -1.0f;
  gui->source_pos_type = DT_MASKS_SOURCE_POS_RELATIVE_TEMP;
  gui->panel_hover_formids = NULL;
  gui->panel_selected_formid = INVALID_MASKID;
  gui->canvas_hover_formid = INVALID_MASKID;
  gui->entered_object = INVALID_MASKID;
}

void dt_masks_gui_form_create(dt_masks_form_t *form,
                              dt_masks_form_gui_t *gui,
                              const int index,
                              const dt_iop_module_t *module)
{
  const int npoints = g_list_length(gui->points);

  if(npoints == index)
    gui->points = g_list_append(gui->points, calloc(1, sizeof(dt_masks_form_gui_points_t)));
  else if(npoints > index)
    dt_masks_gui_form_remove(form, gui, index);
  else
    return;

  dt_masks_form_gui_points_t *gpt = g_list_nth_data(gui->points, index);

  if(dt_masks_get_points_border(darktable.develop, form,
                                &gpt->points, &gpt->points_count,
                                &gpt->border, &gpt->border_count, 0, NULL))
  {
    if(form->type & DT_MASKS_CLONE)
      dt_masks_get_points_border(darktable.develop, form,
                                 &gpt->source, &gpt->source_count, NULL, NULL, 1, module);
    gui->pipe_hash = darktable.develop->preview_pipe->backbuf_hash;
    gui->formid = form->formid;
  }
}

void dt_masks_form_gui_points_free(const gpointer data)
{
  if(!data) return;

  dt_masks_form_gui_points_t *gpt = (dt_masks_form_gui_points_t *)data;

  dt_free_align(gpt->points);
  dt_free_align(gpt->border);
  dt_free_align(gpt->source);
  free(gpt);
}

void dt_masks_gui_form_remove(dt_masks_form_t *form,
                              dt_masks_form_gui_t *gui,
                              const int index)
{
  dt_masks_form_gui_points_t *gpt = g_list_nth_data(gui->points, index);
  gui->pipe_hash = DT_INVALID_HASH;
  gui->formid = NO_MASKID;

  if(gpt)
  {
    gpt->points_count = gpt->border_count = gpt->source_count = 0;
    dt_free_align(gpt->points);
    gpt->points = NULL;
    dt_free_align(gpt->border);
    gpt->border = NULL;
    dt_free_align(gpt->source);
    gpt->source = NULL;
  }
}

// Maximum fraction of the image dimension by which a dragged mask node /
// anchor / clone source may be pushed outside the image.
#define DT_MASKS_MOVE_MARGIN 0.5f

void dt_masks_clamp_move_pts(float *pts, const float wd, const float ht)
{
  const float mx = DT_MASKS_MOVE_MARGIN * wd;
  const float my = DT_MASKS_MOVE_MARGIN * ht;
  pts[0] = fminf(fmaxf(pts[0], -mx), wd + mx);
  pts[1] = fminf(fmaxf(pts[1], -my), ht + my);
}

void dt_masks_gui_form_test_create(dt_masks_form_t *form,
                                   dt_masks_form_gui_t *gui,
                                   const dt_iop_module_t *module)
{
  // we test if the image has changed
  if(gui->pipe_hash != DT_INVALID_HASH)
  {
    if(gui->pipe_hash != darktable.develop->preview_pipe->backbuf_hash)
    {
      dt_print(DT_DEBUG_EXPOSE, "[dt_masks_gui_form_test_create] refreshes mask visualizer");
      gui->pipe_hash = DT_INVALID_HASH;
      gui->formid = NO_MASKID;
      g_list_free_full(gui->points, dt_masks_form_gui_points_free);
      gui->points = NULL;
    }
  }

  // we create the spots if needed
  if(gui->pipe_hash == DT_INVALID_HASH)
  {
    if(form->type & DT_MASKS_GROUP)
    {
      int pos = 0;
      for(GList *fpts = form->points; fpts;  fpts = g_list_next(fpts))
      {
        const dt_masks_point_group_t *fpt = fpts->data;
        dt_masks_form_t *sel = dt_masks_get_from_id(darktable.develop, fpt->formid);
        if(!sel) return;
        dt_masks_gui_form_create(sel, gui, pos, module);
        pos++;
      }
    }
    else
      dt_masks_gui_form_create(form, gui, 0, module);
  }
}

// does `id` name a form in `forms`, or a group marker in one of them? The two
// share one id space: a form taking a marker's id would be found wherever the
// marker's formid is looked up
static gboolean _id_taken(GList *forms, const dt_mask_id_t id)
{
  for(const GList *f = forms; f; f = g_list_next(f))
  {
    const dt_masks_form_t *ff = f->data;
    if(ff->formid == id) return TRUE;
    if(!(ff->type & DT_MASKS_GROUP)) continue;
    for(const GList *p = ff->points; p; p = g_list_next(p))
    {
      const dt_masks_point_group_t *pt = p->data;
      if(dt_masks_point_is_marker(pt) && pt->formid == id) return TRUE;
    }
  }
  return FALSE;
}

static void _check_id(dt_masks_form_t *form)
{
  dt_mask_id_t nid = 100;
  while(_id_taken(darktable.develop->forms, form->formid)) form->formid = nid++;
}

static void _set_group_name_from_module(const dt_iop_module_t *module,
                                        dt_masks_form_t *grp)
{
  gchar *module_label = dt_history_item_get_name(module);
  snprintf(grp->name, sizeof(grp->name), _("group `%s'"), module_label);
  g_free(module_label);
}

static dt_masks_form_t *_group_create(dt_develop_t *dev,
                                      const dt_iop_module_t *module,
                                      const dt_masks_type_t type)
{
  dt_masks_form_t* grp = dt_masks_create(type);
  _set_group_name_from_module(module, grp);
  _check_id(grp);
  dev->forms = g_list_append(dev->forms, grp);
  module->blend_params->mask_id = grp->formid;
  return grp;
}

dt_masks_form_t *dt_masks_module_group_create(dt_develop_t *dev, dt_iop_module_t *module)
{
  return _group_create(dev, module, DT_MASKS_GROUP);
}

static dt_masks_form_t *_group_from_module(const dt_develop_t *dev,
                                           const dt_iop_module_t *module)
{
  return dt_masks_get_from_id(dev, module->blend_params->mask_id);
}

static gboolean _form_is_in_group(const dt_develop_t *dev,
                                  const dt_masks_form_t *group,
                                  const dt_mask_id_t maskid,
                                  const int depth)
{
  if(depth > DT_MASKS_NESTING_MAX) return FALSE;
  for(const GList *iter = group->points; iter; iter = g_list_next(iter))
  {
    const dt_masks_point_group_t *pt = iter->data;
    if(pt->formid == maskid) return TRUE;

    const dt_masks_form_t *child = dt_masks_get_from_id(dev, pt->formid);
    if(child && (child->type & DT_MASKS_GROUP))
    {
      if(_form_is_in_group(dev, child, maskid, depth + 1)) return TRUE;
    }
  }
  return FALSE;
}

gboolean dt_masks_is_in_module(const dt_mask_id_t maskid, const dt_iop_module_t *module)
{
  if(!dt_is_valid_maskid(maskid) || !module) return FALSE;

  if(maskid == module->blend_params->mask_id) return TRUE;

  const dt_masks_form_t *root = dt_masks_get_from_id(module->dev, module->blend_params->mask_id);
  if(root && (root->type & DT_MASKS_GROUP))
    return _form_is_in_group(module->dev, root, maskid, 0);

  return FALSE;
}

void dt_masks_register_forms(dt_develop_t *dev,
                             GList *forms)
{
  for(GList *l = forms;
      l;
      l = g_list_next(l))
  {
    dt_masks_form_t *form = l->data;
    dev->forms = g_list_append(dev->forms, form);
  }

  dt_dev_add_masks_history_item(dev, NULL, TRUE);
}

void dt_masks_assign_unique_name(dt_develop_t *dev, dt_masks_form_t *form)
{
  // mask nb will be at least the length of the list

  // count only the same forms to have a clean numbering
  guint nb = 0;
  for(GList *l = dev->forms; l; l = g_list_next(l))
  {
    const dt_masks_form_t *f = l->data;
    if(f->type == form->type)
      nb++;
  }

  gboolean exist = FALSE;

  // check that we do not have duplicate, in case some masks have been
  // removed we can have hole and so nb could already exists.
  do
  {
    exist = FALSE;
    nb++;

    if(form->functions && form->functions->set_form_name)
      form->functions->set_form_name(form, nb);

    for(GList *l = dev->forms; l; l = g_list_next(l))
    {
      const dt_masks_form_t *f = l->data;
      if(!strcmp(f->name, form->name))
      {
        exist = TRUE;
        break;
      }
    }
  } while(exist);
}

// the opacity a freshly added shape starts at. Parametric/raster channels
// have no on-canvas "set opacity" gesture, so remembering a shape's last
// opacity for them would be surprising -- always start those fully opaque.
// For drawn shapes, the "sticky opacity" option (masks panel hamburger ->
// options) controls whether the last-used opacity (see
// dt_masks_form_change_opacity) is carried over, or every new shape starts
// fully opaque instead.
static float _new_shape_default_opacity(const dt_masks_type_t type)
{
  if(type & (DT_MASKS_PARAMETRIC | DT_MASKS_RASTER)) return 1.0f;
  const float op = dt_conf_get_float("plugins/darkroom/masks/opacity");
  if(dt_conf_get_bool("plugins/darkroom/masks/opacity_not_sticky"))
    dt_conf_set_float("plugins/darkroom/masks/opacity", 1.0f);
  return op;
}

dt_masks_point_group_t *dt_masks_group_insert_point(dt_develop_t *dev,
                                                    dt_iop_module_t *module,
                                                    dt_masks_form_t *form)
{
  // is there already a masks group for this module ?
  dt_masks_form_t *grp = _group_from_module(dev, module);
  if(!grp)
  {
    // we create a new group
    if(form->type & (DT_MASKS_CLONE | DT_MASKS_NON_CLONE))
      grp = _group_create(dev, module, DT_MASKS_GROUP | DT_MASKS_CLONE);
    else
      grp = _group_create(dev, module, DT_MASKS_GROUP);
  }
  // we add the form in this group
  dt_masks_point_group_t *grpt = calloc(1, sizeof(dt_masks_point_group_t));
  grpt->formid = form->formid;
  grpt->parentid = grp->formid;
  grpt->state = DT_MASKS_STATE_SHOW | DT_MASKS_STATE_USE;
  grpt->opacity = _new_shape_default_opacity(form->type);
  grpt->group_opacity = 1.0f;

  if(module->blend_params->mask_mode & DEVELOP_MASK_FLEXI)
  {
    // a flexi element takes its group's settings from the group's marker, and
    // a flexi list starts with one (see dt_masks_group_ensure_marker). Its own
    // operator is only what the classic fold would read
    dt_masks_group_ensure_marker(dev->forms, grp);
    grpt->state |= DT_MASKS_STATE_UNION;
    // the panel's target group: on top of it, which for an empty group is
    // right after its marker. Without one the element lands in the top group
    dt_iop_gui_blend_data_t *bd = module->blend_data;
    GList *after = NULL;
    if(bd && bd->insert_active && dt_is_valid_maskid(bd->insert_after_fid))
      for(GList *l = grp->points; l && !after; l = g_list_next(l))
        if(((dt_masks_point_group_t *)l->data)->formid == bd->insert_after_fid) after = l;
    grp->points = after ? g_list_insert_before(grp->points, after->next, grpt)
                        : g_list_append(grp->points, grpt);
  }
  else
  {
    if(grp->points) grpt->state |= dt_masks_get_default_operator(form);
    grp->points = g_list_append(grp->points, grpt);
  }
  return grpt;
}

void dt_masks_group_insert_member(dt_develop_t *dev,
                                  dt_iop_module_t *module,
                                  dt_masks_form_t *form,
                                  dt_masks_form_gui_t *gui)
{
  dt_masks_group_insert_point(dev, module, form);
  // we save the group
  dt_dev_add_masks_history_item(dev, module, TRUE);
  if(gui)
  {
    gui->panel_selected_formid = form->formid;
  }
  else if(dev && dev->form_gui)
  {
    dev->form_gui->panel_selected_formid = form->formid;
  }
  if(module && module->blend_data)
  {
    dt_iop_gui_blend_data_t *bd = module->blend_data;
    bd->panel_selected_formid = form->formid;
  }
  // we update module gui
  if(gui) dt_masks_iop_update(module);
}

void dt_masks_gui_form_save_creation(dt_develop_t *dev,
                                     dt_iop_module_t *module,
                                     dt_masks_form_t *form,
                                     dt_masks_form_gui_t *gui)
{
  // we check if the id is already registered
  _check_id(form);

  if(gui) gui->creation = FALSE;

  dt_masks_assign_unique_name(dev, form);

  dev->forms = g_list_append(dev->forms, form);

  dt_dev_add_masks_history_item(dev, module, TRUE);

  if(module) dt_masks_group_insert_member(dev, module, form, gui);

  // show the form if needed
  if(gui)
  {
    dev->form_gui->formid = form->formid;
    dev->form_gui->panel_selected_formid = form->formid;
  }
  if(module && module->blend_data)
  {
    dt_iop_gui_blend_data_t *bd = module->blend_data;
    bd->panel_selected_formid = form->formid;
  }
}

int dt_masks_form_duplicate(dt_develop_t *dev, const dt_mask_id_t formid)
{
  // we create a new empty form
  dt_masks_form_t *fbase = dt_masks_get_from_id(dev, formid);
  if(!fbase) return -1;
  dt_masks_form_t *fdest = dt_masks_create(fbase->type);
  _check_id(fdest);

  // we copy the base values
  fdest->source[0] = fbase->source[0];
  fdest->source[1] = fbase->source[1];
  fdest->source[2] = fbase->source[2];
  fdest->version = fbase->version;
  snprintf(fdest->name, sizeof(fdest->name), _("copy of `%s'"), fbase->name);

  darktable.develop->forms = g_list_append(dev->forms, fdest);

  // we copy all the points
  if(fbase->functions && fbase->functions->duplicate_points)
    fbase->functions->duplicate_points(dev, fbase, fdest);

  // we save the form
  dt_dev_add_masks_history_item(dev, NULL, TRUE);

  // and we return its id
  return fdest->formid;
}

dt_mask_id_t dt_masks_form_copy(dt_develop_t *dev, const dt_mask_id_t formid)
{
  dt_masks_form_t *base = dt_masks_get_from_id(dev, formid);
  if(!base) return INVALID_MASKID;
  dt_masks_form_t *dest = dt_masks_create(base->type);
  if(!dest) return INVALID_MASKID;
  _check_id(dest);
  memcpy(dest->source, base->source, sizeof(dest->source));
  dest->version = base->version;
  g_strlcpy(dest->name, base->name, sizeof(dest->name));
  dev->forms = g_list_append(dev->forms, dest);

  // a container's members are copied too: dt_masks_group_duplicate_points
  // would do the same, but records a history item per member
  if(base->type & (DT_MASKS_GROUP | DT_MASKS_OBJECT))
  {
    for(GList *l = base->points; l; l = g_list_next(l))
    {
      const dt_masks_point_group_t *pt = l->data;
      if(dt_masks_point_is_marker(pt))
      {
        dt_masks_group_copy_marker(dev->forms, dest, pt);
        continue;
      }
      const dt_mask_id_t nid = dt_masks_form_copy(dev, pt->formid);
      if(!dt_is_valid_maskid(nid)) continue;
      dt_masks_point_group_t *npt = malloc(sizeof(dt_masks_point_group_t));
      memcpy(npt, pt, sizeof(dt_masks_point_group_t));
      npt->formid = nid;
      npt->parentid = dest->formid;
      dest->points = g_list_append(dest->points, npt);
    }
  }
  else if(base->functions && base->functions->duplicate_points)
    base->functions->duplicate_points(dev, base, dest);

  return dest->formid;
}

int dt_masks_get_points_border(dt_develop_t *dev,
                               dt_masks_form_t *form,
                               float **points,
                               int *points_count,
                               float **border,
                               int *border_count,
                               const int source,
                               const dt_iop_module_t *module)
{
  if(form->functions && form->functions->get_points_border)
  {
    return form->functions->get_points_border(dev, form, points, points_count,
                                              border, border_count, source,
                                              module);
  }
  return 0;
}

int dt_masks_get_area(const dt_iop_module_t *module,
                      const dt_dev_pixelpipe_iop_t *piece,
                      dt_masks_form_t *form,
                      int *width,
                      int *height,
                      int *posx,
                      int *posy)
{
  if(form->functions && form->functions->get_area)
    return form->functions->get_area(module, piece, form, width, height, posx, posy);

  return 0;
}

int dt_masks_get_source_area(dt_iop_module_t *module,
                             dt_dev_pixelpipe_iop_t *piece,
                             dt_masks_form_t *form,
                             int *width,
                             int *height,
                             int *posx,
                             int *posy)
{
  *width = *height = *posx = *posy = 0;

  // must be a clone form
  if(form->type & DT_MASKS_CLONE)
  {
    if(form->functions && form->functions->get_source_area)
      return form->functions->get_source_area(module, piece, form, width, height,
                                              posx, posy);
  }
  return 0;
}

int dt_masks_version(void)
{
  return DEVELOP_MASKS_VERSION;
}

static int _masks_legacy_params_v1_to_v2(const dt_develop_t *dev, void *params)
{
  /*
   * difference: before v2 images were originally rotated on load, and then
   * maybe in flip iop
   * after v2: images are only rotated in flip iop.
   */

  dt_masks_form_t *m = (dt_masks_form_t *)params;

  const dt_image_orientation_t ori = dt_image_orientation(&dev->image_storage);

  if(ori == ORIENTATION_NONE)
  {
    // image is not rotated, we're fine!
    m->version = 2;
    return 0;
  }
  else
  {
    if(dev->iop == NULL) return 1;

    dt_iop_module_t *module = dt_iop_get_module_from_list(dev->iop, "flip");

    if(module == NULL) return 1;

    dt_dev_pixelpipe_iop_t piece = { 0 };

    module->init_pipe(module, NULL, &piece);
    module->commit_params(module, module->default_params, NULL, &piece);

    piece.buf_in.width = 1;
    piece.buf_in.height = 1;

    GList *p = m->points;

    if(!p) return 1;

    if(m->type & DT_MASKS_CIRCLE)
    {
      dt_masks_point_circle_t *circle = p->data;
      module->distort_backtransform(module, &piece, circle->center, 1);
    }
    else if(m->type & DT_MASKS_PATH)
    {
      for(; p; p = g_list_next(p))
      {
        dt_masks_point_path_t *path = p->data;
        module->distort_backtransform(module, &piece, path->corner, 1);
        module->distort_backtransform(module, &piece, path->ctrl1, 1);
        module->distort_backtransform(module, &piece, path->ctrl2, 1);
      }
    }
    else if(m->type & DT_MASKS_GRADIENT)
    { // TODO: new ones have wrong rotation.
      dt_masks_point_gradient_t *gradient = p->data;
      module->distort_backtransform(module, &piece, gradient->anchor, 1);

      if(ori == ORIENTATION_ROTATE_180_DEG)
        gradient->rotation -= 180.0f;
      else if(ori == ORIENTATION_ROTATE_CCW_90_DEG)
        gradient->rotation -= 90.0f;
      else if(ori == ORIENTATION_ROTATE_CW_90_DEG)
        gradient->rotation -= -90.0f;
    }
    else if(m->type & DT_MASKS_ELLIPSE)
    {
      dt_masks_point_ellipse_t *ellipse = p->data;
      module->distort_backtransform(module, &piece, ellipse->center, 1);

      if(ori & ORIENTATION_SWAP_XY)
      {
        const float y = ellipse->radius[0];
        ellipse->radius[0] = ellipse->radius[1];
        ellipse->radius[1] = y;
      }
    }
    else if(m->type & DT_MASKS_BRUSH)
    {
      for(; p; p = g_list_next(p))
      {
        dt_masks_point_brush_t *brush = p->data;
        module->distort_backtransform(module, &piece, brush->corner, 1);
        module->distort_backtransform(module, &piece, brush->ctrl1, 1);
        module->distort_backtransform(module, &piece, brush->ctrl2, 1);
      }
    }

    if(m->type & DT_MASKS_CLONE)
    {
      // NOTE: can be: DT_MASKS_CIRCLE, DT_MASKS_ELLIPSE, DT_MASKS_PATH
      module->distort_backtransform(module, &piece, m->source, 1);
    }

    m->version = 2;

    return 0;
  }
}

static void _masks_legacy_params_v2_to_v3_transform(const dt_image_t *img,
                                                      float *points)
{
  const float w = img->width;
  const float h = img->height;
  const float cx = img->crop_x;
  const float cy = img->crop_y;
  const float cw = img->p_width;
  const float ch = img->p_height;

  /*
   * masks coordinates are normalized, so we need to:
   * 1. de-normalize them by image original cropped dimensions
   * 2. un-crop them by adding top-left crop coordinates
   * 3. normalize them by the image fully uncropped dimensions
   */
  points[0] = ((points[0] * cw) + cx) / w;
  points[1] = ((points[1] * ch) + cy) / h;
}

static void _masks_legacy_params_v2_to_v3_transform_only_rescale
  (const dt_image_t *img,
   float *points,
   const size_t points_count)
{
  const float w = img->width;
  const float h = img->height;
  const float cw = img->p_width;
  const float ch = img->p_height;

  /*
   * masks coordinates are normalized, so we need to:
   * 1. de-normalize them by minimal of image original cropped dimensions
   * 2. normalize them by the minimal of image fully uncropped dimensions
   */
  for(size_t i = 0; i < points_count; i++)
    points[i] = ((points[i] * MIN(cw, ch))) / MIN(w, h);
}

static int _masks_legacy_params_v2_to_v3(const dt_develop_t *dev, void *params)
{
  /*
   * difference: before v3 images were originally cropped on load
   * after v3: images are cropped in rawprepare iop.
   */

  dt_masks_form_t *m = (dt_masks_form_t *)params;

  const dt_image_t *img = &(dev->image_storage);

  if(img->p_width == img->width
     && img->p_height == img->height)
  {
    // image has no "raw cropping", we're fine!
    m->version = 3;
    return 0;
  }
  else
  {
    GList *p = m->points;

    if(!p) return 1;

    if(m->type & DT_MASKS_CIRCLE)
    {
      dt_masks_point_circle_t *circle = p->data;
      _masks_legacy_params_v2_to_v3_transform(img, circle->center);
      _masks_legacy_params_v2_to_v3_transform_only_rescale(img, &circle->radius, 1);
      _masks_legacy_params_v2_to_v3_transform_only_rescale(img, &circle->border, 1);
    }
    else if(m->type & DT_MASKS_PATH)
    {
      for(; p; p = g_list_next(p))
      {
        dt_masks_point_path_t *path = p->data;
        _masks_legacy_params_v2_to_v3_transform(img, path->corner);
        _masks_legacy_params_v2_to_v3_transform(img, path->ctrl1);
        _masks_legacy_params_v2_to_v3_transform(img, path->ctrl2);
        _masks_legacy_params_v2_to_v3_transform_only_rescale(img, path->border, 2);
      }
    }
    else if(m->type & DT_MASKS_GRADIENT)
    {
      dt_masks_point_gradient_t *gradient = p->data;
      _masks_legacy_params_v2_to_v3_transform(img, gradient->anchor);
    }
    else if(m->type & DT_MASKS_ELLIPSE)
    {
      dt_masks_point_ellipse_t *ellipse = p->data;
      _masks_legacy_params_v2_to_v3_transform(img, ellipse->center);
      _masks_legacy_params_v2_to_v3_transform_only_rescale(img, ellipse->radius, 2);
      _masks_legacy_params_v2_to_v3_transform_only_rescale(img, &ellipse->border, 1);
    }
    else if(m->type & DT_MASKS_BRUSH)
    {
      for(; p;  p = g_list_next(p))
      {
        dt_masks_point_brush_t *brush = p->data;
        _masks_legacy_params_v2_to_v3_transform(img, brush->corner);
        _masks_legacy_params_v2_to_v3_transform(img, brush->ctrl1);
        _masks_legacy_params_v2_to_v3_transform(img, brush->ctrl2);
        _masks_legacy_params_v2_to_v3_transform_only_rescale(img, brush->border, 2);
      }
    }

    if(m->type & DT_MASKS_CLONE)
    {
      // NOTE: can be: DT_MASKS_CIRCLE, DT_MASKS_ELLIPSE, DT_MASKS_PATH
      _masks_legacy_params_v2_to_v3_transform(img, m->source);
    }

    m->version = 3;

    return 0;
  }
}

static int _masks_legacy_params_v3_to_v4(dt_develop_t *dev, void *params)
{
  /*
   * difference affecting ellipse
   * up to v3: only equidistant feathering
   * after v4: choice between equidistant and proportional feathering
   * type of feathering is defined in new flags parameter
   */

  dt_masks_form_t *m = (dt_masks_form_t *)params;

  GList *p = m->points;

  if(!p) return 1;

  if(m->type & DT_MASKS_ELLIPSE)
  {
    dt_masks_point_ellipse_t *ellipse = p->data;
    ellipse->flags = DT_MASKS_ELLIPSE_EQUIDISTANT;
  }

  m->version = 4;

  return 0;
}


static int _masks_legacy_params_v4_to_v5(dt_develop_t *dev, void *params)
{
  /*
   * difference affecting gradient
   * up to v4: only linear gradient (relative to input image)
   * after v5: curved gradients
   */

  dt_masks_form_t *m = (dt_masks_form_t *)params;

  GList *p = m->points;

  if(!p) return 1;

  if(m->type & DT_MASKS_GRADIENT)
  {
    dt_masks_point_gradient_t *gradient = p->data;
    gradient->curvature = 0.0f;
  }

  m->version = 5;

  return 0;
}

static int _masks_legacy_params_v5_to_v6(dt_develop_t *dev, void *params)
{
  /*
   * difference affecting gradient
   * up to v5: linear transition
   * after v5: linear or sigmoidal transition
   */

  dt_masks_form_t *m = (dt_masks_form_t *)params;

  GList *p = m->points;

  if(!p) return 1;

  if(m->type & DT_MASKS_GRADIENT)
  {
    dt_masks_point_gradient_t *gradient = p->data;
    gradient->state = DT_MASKS_GRADIENT_STATE_LINEAR;
  }

  m->version = 6;

  return 0;
}

static int dt_masks_legacy_params_v6_to_v7(dt_develop_t *dev, void *params)
{
  /*
   * masks v7 appended an optional per-shape refinement block to the group
   * point struct (dt_masks_point_group_t.refinement). The block is already
   * zero-filled at read time (enabled == 0), which disables it and keeps
   * rendering identical to v6. Nothing to convert; just bump the version.
   */
  dt_masks_form_t *m = (dt_masks_form_t *)params;
  m->version = 7;
  return 0;
}

static int dt_masks_legacy_params_v7_to_v8(dt_develop_t *dev, void *params)
{
  /*
   * masks v8 appended an optional custom group-name field to the group point
   * struct (dt_masks_point_group_t.name), for flexi first-class groups. The
   * field is already zero-filled at read time (empty string), which means no
   * custom name and keeps rendering identical to v7. Nothing to convert;
   * just bump the version.
   */
  dt_masks_form_t *m = (dt_masks_form_t *)params;
  m->version = 8;
  return 0;
}

static int dt_masks_legacy_params_v8_to_v9(dt_develop_t *dev, void *params)
{
  /*
   * masks v9 appended a persistent, multiplicative group-level opacity to
   * the group point struct (dt_masks_point_group_t.group_opacity), broadcast
   * to every member of a run the same way refinement/name are. Unlike those,
   * 0.0 (what the read-time zero-fill above leaves it at) is not a neutral
   * value for a multiplicative gain -- it would silently zero out every
   * pre-v9 group's mask -- so every group point explicitly gets the
   * identity value (1.0) here instead.
   */
  dt_masks_form_t *m = (dt_masks_form_t *)params;
  if(m->type & DT_MASKS_GROUP)
    for(GList *l = m->points; l; l = g_list_next(l))
      ((dt_masks_point_group_t *)l->data)->group_opacity = 1.0f;
  m->version = 9;
  return 0;
}

static int dt_masks_legacy_params_v9_to_v10(dt_develop_t *dev, void *params)
{
  /*
   * masks v10 replaced the temporary DT_MASKS_STATE_GROUP_BREAK bit
   * (borrowed from the group point's `state` field) with a real, dedicated
   * field, dt_masks_point_group_t.group_start -- see that enum value's own
   * comment. Zero-fill is neutral for the new field (0 = "no explicit
   * break," same as "bit not set"), so there is nothing to backfill; the
   * only work here is carrying forward any break that was actually set in
   * the old bit, and clearing that bit from `state` so it doesn't linger
   * as stale data once nothing reads it from there anymore.
   */
  dt_masks_form_t *m = (dt_masks_form_t *)params;
  if(m->type & DT_MASKS_GROUP)
    for(GList *l = m->points; l; l = g_list_next(l))
    {
      dt_masks_point_group_t *pt = (dt_masks_point_group_t *)l->data;
      if(pt->state & DT_MASKS_STATE_GROUP_BREAK)
      {
        pt->group_start = 1;
        pt->state &= ~DT_MASKS_STATE_GROUP_BREAK;
      }
    }
  m->version = 10;
  return 0;
}

int dt_masks_legacy_params(dt_develop_t *dev,
                           void *params,
                           const int old_version,
                           const int new_version)
{
  // sequential upgrade chain: apply every step from old_version up to
  // new_version in order. Each dt_masks_legacy_params_vN_to_vN+1 bumps the
  // form version and returns non-zero on failure.
  if(old_version < 1 || old_version > new_version) return 1;

  int res = 0;
  if(!res && old_version < 2 && new_version >= 2)
    res = _masks_legacy_params_v1_to_v2(dev, params);
  if(!res && old_version < 3 && new_version >= 3)
    res = _masks_legacy_params_v2_to_v3(dev, params);
  if(!res && old_version < 4 && new_version >= 4)
    res = _masks_legacy_params_v3_to_v4(dev, params);
  if(!res && old_version < 5 && new_version >= 5)
    res = _masks_legacy_params_v4_to_v5(dev, params);
  if(!res && old_version < 6 && new_version >= 6)
    res = _masks_legacy_params_v5_to_v6(dev, params);
  if(!res && old_version < 7 && new_version >= 7)
    res = dt_masks_legacy_params_v6_to_v7(dev, params);
  if(!res && old_version < 8 && new_version >= 8)
    res = dt_masks_legacy_params_v7_to_v8(dev, params);
  if(!res && old_version < 9 && new_version >= 9)
    res = dt_masks_legacy_params_v8_to_v9(dev, params);
  if(!res && old_version < 10 && new_version >= 10)
    res = dt_masks_legacy_params_v9_to_v10(dev, params);

  return res;
}

static dt_mask_id_t form_id = 0;

dt_mask_id_t dt_masks_new_marker_id(GList *forms)
{
  dt_mask_id_t id = time(NULL) + form_id++;
  while(_id_taken(forms, id)) id = time(NULL) + form_id++;
  return id;
}

dt_masks_point_group_t *dt_masks_marker_new(GList *forms,
                                            const dt_masks_form_t *grp,
                                            const int state)
{
  dt_masks_point_group_t *m = calloc(1, sizeof(dt_masks_point_group_t));
  if(!m) return NULL;
  m->formid = dt_masks_new_marker_id(forms);
  m->parentid = grp ? grp->formid : NO_MASKID;
  m->state = DT_MASKS_STATE_GROUP_MARKER
             | (state & (DT_MASKS_STATE_OP | DT_MASKS_STATE_WITHIN));
  if(!(m->state & DT_MASKS_STATE_OP_COMBINE)) m->state |= DT_MASKS_STATE_UNION;
  m->opacity = 1.0f;
  m->group_opacity = 1.0f;
  return m;
}

dt_masks_point_group_t *dt_masks_group_copy_marker(GList *forms,
                                                   dt_masks_form_t *dest,
                                                   const dt_masks_point_group_t *marker)
{
  dt_masks_point_group_t *pt = malloc(sizeof(dt_masks_point_group_t));
  if(!pt) return NULL;
  memcpy(pt, marker, sizeof(dt_masks_point_group_t));
  pt->formid = dt_masks_new_marker_id(forms);
  pt->parentid = dest->formid;
  dest->points = g_list_append(dest->points, pt);
  return pt;
}

gboolean dt_masks_group_ensure_marker(GList *forms, dt_masks_form_t *grp)
{
  if(!grp || !(grp->type & DT_MASKS_GROUP) || (grp->type & DT_MASKS_CLONE)) return FALSE;
  if(grp->points && dt_masks_point_is_marker(grp->points->data)) return FALSE;
  dt_masks_point_group_t *m = dt_masks_marker_new(forms, grp, DT_MASKS_STATE_UNION);
  if(!m) return FALSE;
  grp->points = g_list_prepend(grp->points, m);
  return TRUE;
}

// does the point at node `l` of a classic list start a new run? The
// non-union operators are the ones classic applies once per member, see
// _normalize_group in migrate_legacy.c for why `split_nonunion` exists
static gboolean _classic_run_starts(GList *l, const gboolean split_nonunion)
{
  if(!l->prev) return TRUE;
  const dt_masks_point_group_t *pt = l->data;
  const dt_masks_point_group_t *below = l->prev->data;
  // MULTIPLY is absent: no classic shape carries it, see _normalize_group
  const int non_union = DT_MASKS_STATE_INTERSECTION | DT_MASKS_STATE_DIFFERENCE
                        | DT_MASKS_STATE_SUM | DT_MASKS_STATE_EXCLUSION;
  return pt->group_start
         || (split_nonunion && (pt->state & non_union))
         || dt_masks_eff_group_op(pt->state) != dt_masks_eff_group_op(below->state);
}

// the marker id of the run headed by `head` in group `grp`: derived from the
// two, so the same run marked in two history snapshots gets the same id. The
// panel keys a group's selection, number and expanded state on it
static dt_mask_id_t _run_marker_id(GList *forms, const dt_mask_id_t grp, const dt_mask_id_t head)
{
  const gint64 key = ((gint64)grp << 32) | (guint32)head;
  // positive, and above the small ids _check_id hands out on a collision
  dt_mask_id_t id = (dt_mask_id_t)((g_int64_hash(&key) & 0x3fffffffu) | 0x40000000u);
  while(_id_taken(forms, id)) id = (dt_mask_id_t)(((guint32)id + 1u) | 0x40000000u);
  return id;
}

static gboolean _has_marker(const dt_masks_form_t *grp)
{
  for(const GList *l = grp->points; l; l = g_list_next(l))
    if(dt_masks_point_is_marker(l->data)) return TRUE;
  return FALSE;
}

static int _combine_op(const int state)
{
  return dt_masks_eff_group_op(state) & DT_MASKS_STATE_OP_COMBINE;
}

// a group that only unions its members in, as every classic run is
static gboolean _marker_is_plain(const dt_masks_point_group_t *mk)
{
  return !(mk->state & (DT_MASKS_STATE_OP_DISABLE | DT_MASKS_STATE_OP_INVERT
                        | DT_MASKS_STATE_WITHIN))
         && mk->group_opacity == 1.0f
         && mk->refinement.enabled == DT_MASKS_REFINE_OFF;
}

static dt_masks_point_group_t *_plain_marker(GList *forms,
                                             const dt_masks_form_t *grp,
                                             const dt_mask_id_t head)
{
  dt_masks_point_group_t *mk = dt_masks_marker_new(forms, grp, DT_MASKS_STATE_UNION);
  mk->formid = _run_marker_id(forms, grp->formid, head);
  return mk;
}

// Push "s * (invert ? 1 - r : r)" of a nested group's result r into one of its
// groups, `first` being its bottom one. 1 - (acc op x) is (1 - acc) op' x',
// where x is inverted for union, intersection and multiply and kept for
// difference and screen; sum and exclusion have no such dual. s * (acc op x)
// scales x for union and intersection, and nothing for difference and
// multiply, where the scaled acc carries it. With `apply` FALSE, only checks
static gboolean _push_into_run(dt_masks_point_group_t *mk,
                               const gboolean first,
                               const gboolean invert,
                               const float scale,
                               const gboolean apply)
{
  int op = _combine_op(mk->state);
  gboolean flip = invert;
  if(invert && !first)
  {
    if(op == DT_MASKS_STATE_UNION) op = DT_MASKS_STATE_INTERSECTION;
    else if(op == DT_MASKS_STATE_INTERSECTION) op = DT_MASKS_STATE_UNION;
    else if(op == DT_MASKS_STATE_MULTIPLY) op = DT_MASKS_STATE_OP_SCREEN;
    else if(op == DT_MASKS_STATE_DIFFERENCE)
    {
      op = DT_MASKS_STATE_OP_SCREEN;
      flip = FALSE;
    }
    else if(op == DT_MASKS_STATE_OP_SCREEN)
    {
      op = DT_MASKS_STATE_DIFFERENCE;
      flip = FALSE;
    }
    else
      return FALSE;
  }
  // a group's opacity scales its inverted output, so the two do not commute
  if(flip && mk->group_opacity != 1.0f) return FALSE;
  const gboolean scales =
    first || op == DT_MASKS_STATE_UNION || op == DT_MASKS_STATE_INTERSECTION;
  if(scale != 1.0f && !scales
     && op != DT_MASKS_STATE_DIFFERENCE && op != DT_MASKS_STATE_MULTIPLY)
    return FALSE;

  if(apply)
  {
    mk->state = (mk->state & ~DT_MASKS_STATE_OP_COMBINE) | op;
    if(flip) mk->state ^= DT_MASKS_STATE_OP_INVERT;
    if(scales) mk->group_opacity *= scale;
  }
  return TRUE;
}

/* Replace the member at `ml` of `grp`, a nested group just converted from
   classic, by that group's own groups, where this renders the same mask.
   Classic nests freely, and a drawn + parametric migration nests the whole
   drawn group once more, but most of it is not needed by the flexi fold:
   masks_revamp_nested_groups.md, Q2.

   The member's inversion and opacity, then its group's invert-output and
   opacity, act on the nested result r as s * (inv ? 1 - r : r). It dissolves:
   - when the nested list is one group: that group takes the member's place,
     the transform going onto its own invert-output and opacity;
   - when the member is the bottom of `grp`: the nested groups are spliced in,
     the transform pushed into each (_push_into_run);
   - when every nested group has the operator of the member's group, which is
     union, intersection, sum or multiply, and nothing inverts.
   A member sharing a plain union group with others is split off first, and
   the pieces merged back where they end up plain unions again.

   On success *resume is the node after what was spliced in. */
static gboolean _dissolve_member(GList *forms,
                                 dt_masks_form_t *grp,
                                 GList *ml,
                                 GList **resume)
{
  dt_masks_point_group_t *m = ml->data;
  const dt_masks_form_t *child = dt_masks_get_from_id_ext(forms, m->formid);
  if(!child || child == grp || !(child->type & DT_MASKS_GROUP)
     || (child->type & (DT_MASKS_CLONE | DT_MASKS_OBJECT))
     || !child->points || !dt_masks_point_is_marker(child->points->data))
    return FALSE;
  if(m->state & (DT_MASKS_STATE_HIDDEN | DT_MASKS_STATE_DISABLE)) return FALSE;
  if(m->refinement.enabled != DT_MASKS_REFINE_OFF) return FALSE;

  GList *hl = ml->prev;
  while(hl && !dt_masks_point_is_marker(hl->data)) hl = hl->prev;
  if(!hl) return FALSE;
  const dt_masks_point_group_t *head = hl->data;
  GList *end = ml->next;
  while(end && !dt_masks_point_is_marker(end->data)) end = end->next;
  if(head->state & DT_MASKS_STATE_OP_DISABLE) return FALSE;
  if(head->refinement.enabled != DT_MASKS_REFINE_OFF) return FALSE;

  // regrouping a group's members is exact only for a plain union
  const gboolean split_before = hl->next != ml;
  const gboolean split_after = ml->next != end;
  const gboolean bottom = hl == grp->points;
  if((split_before || split_after)
     && !(_marker_is_plain(head)
          && (bottom || _combine_op(head->state) == DT_MASKS_STATE_UNION)))
    return FALSE;

  gboolean inv = (m->state & DT_MASKS_STATE_INVERSE) != 0;
  float s = m->opacity;
  if(head->state & DT_MASKS_STATE_OP_INVERT)
  {
    if(s != 1.0f) return FALSE;
    inv = !inv;
  }
  s *= head->group_opacity;

  const gboolean base = bottom && !split_before;
  const int op = split_before ? DT_MASKS_STATE_UNION : _combine_op(head->state);

  int n = 0;
  for(const GList *l = child->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    if(!dt_masks_point_is_marker(pt)) continue;
    if(pt->state & DT_MASKS_STATE_OP_DISABLE) return FALSE;
    n++;
  }

  enum { ONE_GROUP, BOTTOM, SAME_OP } rule;
  if(n == 1)
  {
    const dt_masks_point_group_t *k = child->points->data;
    if(inv && k->group_opacity != 1.0f) return FALSE;
    rule = ONE_GROUP;
  }
  else if(base)
  {
    // the nested bottom group must be the one that renders first: were it
    // empty, the next one would take its place with a pushed-down operator
    if(inv || s != 1.0f)
    {
      gboolean live = FALSE;
      for(const GList *l = child->points->next;
          l && !dt_masks_point_is_marker(l->data) && !live;
          l = g_list_next(l))
      {
        const dt_masks_point_group_t *pt = l->data;
        live = !(pt->state & (DT_MASKS_STATE_HIDDEN | DT_MASKS_STATE_DISABLE))
               && dt_masks_get_from_id_ext(forms, pt->formid);
      }
      if(!live) return FALSE;
    }
    for(const GList *l = child->points; l; l = g_list_next(l))
    {
      dt_masks_point_group_t *pt = l->data;
      if(dt_masks_point_is_marker(pt)
         && !_push_into_run(pt, l == child->points, inv, s, FALSE))
        return FALSE;
    }
    rule = BOTTOM;
  }
  else
  {
    if(inv) return FALSE;
    if(op != DT_MASKS_STATE_UNION && op != DT_MASKS_STATE_INTERSECTION
       && op != DT_MASKS_STATE_SUM && op != DT_MASKS_STATE_MULTIPLY)
      return FALSE;
    // sum clamps at 1 after each group, and multiply takes one scale only
    if(s != 1.0f && op != DT_MASKS_STATE_UNION && op != DT_MASKS_STATE_INTERSECTION)
      return FALSE;
    for(const GList *l = child->points->next; l; l = g_list_next(l))
    {
      const dt_masks_point_group_t *pt = l->data;
      if(dt_masks_point_is_marker(pt) && _combine_op(pt->state) != op) return FALSE;
    }
    rule = SAME_OP;
  }

  // split the member off into a group of its own
  GList *after = ml->next;
  GList *hb = NULL;
  if(split_after)
  {
    grp->points = g_list_insert_before(
      grp->points, after,
      _plain_marker(forms, grp, ((dt_masks_point_group_t *)after->data)->formid));
    hb = after->prev;
  }
  GList *gh = hl;
  if(split_before)
  {
    grp->points = g_list_insert_before(grp->points, ml, _plain_marker(forms, grp, m->formid));
    gh = ml->prev;
  }

  GList *first_new = NULL;
  if(rule == ONE_GROUP)
  {
    dt_masks_point_group_t *gm = gh->data;
    const dt_masks_point_group_t *k = child->points->data;
    const gboolean k_inv = ((k->state & DT_MASKS_STATE_OP_INVERT) != 0) != inv;
    gm->state = DT_MASKS_STATE_GROUP_MARKER | op | (k->state & DT_MASKS_STATE_WITHIN)
                | (k_inv ? DT_MASKS_STATE_OP_INVERT : 0);
    gm->group_opacity = k->group_opacity * s;
    gm->refinement = k->refinement;
    if(!gm->name[0]) g_strlcpy(gm->name, k->name, sizeof(gm->name));
    first_new = gh;
  }

  for(const GList *l = child->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    const gboolean marker = dt_masks_point_is_marker(pt);
    if(marker && rule == ONE_GROUP) continue;

    dt_masks_point_group_t *cp = malloc(sizeof(dt_masks_point_group_t));
    memcpy(cp, pt, sizeof(dt_masks_point_group_t));
    cp->parentid = grp->formid;
    if(marker)
    {
      const gboolean first = l == child->points;
      cp->formid = _run_marker_id(forms, grp->formid, pt->formid);
      if(rule == BOTTOM)
        _push_into_run(cp, first, inv, s, TRUE);
      else
      {
        if(first) cp->state = (cp->state & ~DT_MASKS_STATE_OP_COMBINE) | op;
        if(op == DT_MASKS_STATE_UNION || op == DT_MASKS_STATE_INTERSECTION)
          cp->group_opacity *= s;
      }
      if(first && !cp->name[0])
        g_strlcpy(cp->name, ((dt_masks_point_group_t *)gh->data)->name, sizeof(cp->name));
    }
    grp->points = g_list_insert_before(grp->points, ml, cp);
    if(marker && !first_new) first_new = ml->prev;
  }

  free(m);
  grp->points = g_list_delete_link(grp->points, ml);
  if(rule != ONE_GROUP)
  {
    free(gh->data);
    grp->points = g_list_delete_link(grp->points, gh);
  }

  // the plain unions around a split member are one group again
  if(split_before && _marker_is_plain(first_new->data)
     && _combine_op(((dt_masks_point_group_t *)first_new->data)->state) == DT_MASKS_STATE_UNION)
  {
    free(first_new->data);
    grp->points = g_list_delete_link(grp->points, first_new);
  }
  if(hb)
  {
    GList *ll = hb->prev;
    while(!dt_masks_point_is_marker(ll->data)) ll = ll->prev;
    if(_marker_is_plain(ll->data)
       && (ll == grp->points
           || _combine_op(((dt_masks_point_group_t *)ll->data)->state) == DT_MASKS_STATE_UNION))
    {
      free(hb->data);
      grp->points = g_list_delete_link(grp->points, hb);
    }
  }

  *resume = after;
  return TRUE;
}

// `grp` and the groups below it that have no markers yet: the classic ones a
// pass converts. Collected before marking, since a group nested twice is
// marked where it is met first and has to dissolve where it is met again too
static void _collect_classic(GList *forms,
                             const dt_masks_form_t *grp,
                             GHashTable *classic,
                             const int depth)
{
  if(!grp || !(grp->type & DT_MASKS_GROUP) || depth > DT_MASKS_NESTING_MAX) return;
  if(!_has_marker(grp)) g_hash_table_add(classic, GINT_TO_POINTER(grp->formid));
  for(const GList *l = grp->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    if(dt_masks_point_is_marker(pt)) continue;
    const dt_masks_form_t *child = dt_masks_get_from_id_ext(forms, pt->formid);
    if(child != grp) _collect_classic(forms, child, classic, depth + 1);
  }
}

static gboolean _mark_runs(GList *forms,
                           dt_masks_form_t *grp,
                           const gboolean split_nonunion,
                           const int depth,
                           GHashTable *classic)
{
  if(!grp || !(grp->type & DT_MASKS_GROUP) || (grp->type & DT_MASKS_CLONE)) return FALSE;
  // a malformed or cyclic tree must not spin here; nesting is shallow
  if(depth > DT_MASKS_NESTING_MAX) return FALSE;

  gboolean changed = FALSE;
  gboolean marked = FALSE;
  for(GList *l = grp->points; l && !marked; l = g_list_next(l))
    marked = dt_masks_point_is_marker(l->data);

  if(!grp->points)
  {
    grp->points = g_list_append(NULL, dt_masks_marker_new(forms, grp, DT_MASKS_STATE_UNION));
    changed = TRUE;
  }
  else if(!marked)
  {
    GList *l = grp->points;
    while(l)
    {
      GList *end = g_list_next(l);
      while(end && !_classic_run_starts(end, split_nonunion)) end = g_list_next(end);

      // the settings are the ones the flexi fold reads for this run: off its
      // first member that is shown and resolves (_group_get_mask_roi_flexi)
      const dt_masks_point_group_t *src = l->data;
      for(GList *m = l; m != end; m = g_list_next(m))
      {
        const dt_masks_point_group_t *pt = m->data;
        if(!(pt->state & DT_MASKS_STATE_HIDDEN) && dt_masks_get_from_id_ext(forms, pt->formid))
        {
          src = pt;
          break;
        }
      }

      dt_masks_point_group_t *marker =
        dt_masks_marker_new(forms, grp,
                            dt_masks_eff_group_op(src->state)
                              | (src->state & DT_MASKS_STATE_WITHIN));
      marker->formid =
        _run_marker_id(forms, grp->formid, ((dt_masks_point_group_t *)l->data)->formid);
      g_strlcpy(marker->name, src->name, sizeof(marker->name));
      marker->group_opacity = src->group_opacity;
      if(src->refinement.enabled == DT_MASKS_REFINE_GROUP) marker->refinement = src->refinement;

      // the group's settings are the marker's now. A member keeps only what is
      // its own and joins as a plain union element, like any flexi element
      for(GList *m = l; m != end; m = g_list_next(m))
      {
        dt_masks_point_group_t *pt = m->data;
        pt->state = (pt->state & ~(DT_MASKS_STATE_OP | DT_MASKS_STATE_WITHIN))
                    | DT_MASKS_STATE_UNION;
        pt->name[0] = '\0';
        pt->group_opacity = 1.0f;
        pt->group_start = 0;
        if(pt->refinement.enabled == DT_MASKS_REFINE_GROUP)
          memset(&pt->refinement, 0, sizeof(pt->refinement));
      }
      grp->points = g_list_insert_before(grp->points, l, marker);
      l = end;
    }
    changed = TRUE;
  }

  // a member can itself be a group, folded by the same algebra. One converted
  // from classic here is dissolved where it can be; a group already marked,
  // flexi-authored, is left as its author nested it
  for(GList *l = grp->points; l;)
  {
    GList *next = g_list_next(l);
    const dt_masks_point_group_t *pt = l->data;
    dt_masks_form_t *child =
      dt_masks_point_is_marker(pt) ? NULL : dt_masks_get_from_id_ext(forms, pt->formid);
    if(child && child != grp && (child->type & DT_MASKS_GROUP))
    {
      changed |= _mark_runs(forms, child, split_nonunion, depth + 1, classic);
      if(g_hash_table_contains(classic, GINT_TO_POINTER(child->formid))
         && _dissolve_member(forms, grp, l, &next))
        changed = TRUE;
    }
    l = next;
  }
  return changed;
}

// a copy of the nested group `src` for the group `parent`, its form and marker
// ids its own. They are derived from the ids copied, so every history snapshot
// holding the same tree gets the same ones
static dt_masks_form_t *_copy_group(GList *forms,
                                    const dt_masks_form_t *src,
                                    const dt_mask_id_t parent)
{
  dt_masks_form_t *copy = malloc(sizeof(dt_masks_form_t));
  memcpy(copy, src, sizeof(dt_masks_form_t));
  copy->points = NULL;
  copy->formid = _run_marker_id(forms, parent, src->formid);
  // `forms` holds the parent, so appending keeps the caller's head
  forms = g_list_append(forms, copy);
  for(const GList *l = src->points; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = malloc(sizeof(dt_masks_point_group_t));
    memcpy(pt, l->data, sizeof(dt_masks_point_group_t));
    pt->parentid = copy->formid;
    if(dt_masks_point_is_marker(pt))
      pt->formid = _run_marker_id(forms, copy->formid, pt->formid);
    copy->points = g_list_append(copy->points, pt);
  }
  return copy;
}

// Within one mask a group has one parent: the panel keys a group's rows on its
// id, and editing one place of it would edit the other. Classic can hold the
// same nested group twice, so each reference past the first gets a copy. A
// group shared with another module's mask stays shared
static gboolean _unshare_nested(GList *forms,
                                dt_masks_form_t *grp,
                                GHashTable *seen,
                                const int depth)
{
  if(!grp || !(grp->type & DT_MASKS_GROUP) || depth > DT_MASKS_NESTING_MAX) return FALSE;
  gboolean changed = FALSE;
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = l->data;
    if(dt_masks_point_is_marker(pt)) continue;
    dt_masks_form_t *child = dt_masks_get_from_id_ext(forms, pt->formid);
    if(!child || child == grp || !(child->type & DT_MASKS_GROUP)
       || (child->type & (DT_MASKS_CLONE | DT_MASKS_OBJECT)))
      continue;
    if(g_hash_table_contains(seen, GINT_TO_POINTER(child->formid)))
    {
      child = _copy_group(forms, child, grp->formid);
      pt->formid = child->formid;
      changed = TRUE;
    }
    g_hash_table_add(seen, GINT_TO_POINTER(child->formid));
    changed |= _unshare_nested(forms, child, seen, depth + 1);
  }
  return changed;
}

gboolean dt_masks_group_mark_classic_runs(GList *forms,
                                          dt_masks_form_t *grp,
                                          const gboolean split_nonunion)
{
  GHashTable *classic = g_hash_table_new(NULL, NULL);
  _collect_classic(forms, grp, classic, 0);
  gboolean changed = _mark_runs(forms, grp, split_nonunion, 0, classic);
  g_hash_table_destroy(classic);

  GHashTable *seen = g_hash_table_new(NULL, NULL);
  changed |= _unshare_nested(forms, grp, seen, 0);
  g_hash_table_destroy(seen);
  return changed;
}

static dt_masks_point_group_t *_find_marker(GList *forms,
                                            dt_masks_form_t *grp,
                                            const dt_mask_id_t cid,
                                            dt_masks_form_t **owner,
                                            const int depth)
{
  if(!grp || !(grp->type & DT_MASKS_GROUP) || depth > DT_MASKS_NESTING_MAX) return NULL;
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = l->data;
    if(dt_masks_point_is_marker(pt))
    {
      if(pt->formid != cid) continue;
      if(owner) *owner = grp;
      return pt;
    }
    dt_masks_form_t *child = dt_masks_get_from_id_ext(forms, pt->formid);
    if(child == grp) continue;
    dt_masks_point_group_t *mk = _find_marker(forms, child, cid, owner, depth + 1);
    if(mk) return mk;
  }
  return NULL;
}

dt_masks_point_group_t *dt_masks_group_find_marker(GList *forms,
                                                   dt_masks_form_t *root,
                                                   const dt_mask_id_t cid,
                                                   dt_masks_form_t **owner)
{
  return _find_marker(forms, root, cid, owner, 0);
}

static const dt_masks_point_raster_t *_find_raster_of(GList *forms,
                                                      const dt_masks_form_t *grp,
                                                      const dt_iop_module_t *source,
                                                      const dt_mask_id_t id,
                                                      const gboolean any_id,
                                                      const int depth)
{
  if(!grp || !(grp->type & DT_MASKS_GROUP) || depth > DT_MASKS_NESTING_MAX) return NULL;
  for(const GList *l = grp->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    if(!pt || dt_masks_point_is_marker(pt)) continue;
    const dt_masks_form_t *f = dt_masks_get_from_id_ext(forms, pt->formid);
    if(!f || f == grp) continue;
    if(f->type & DT_MASKS_GROUP)
    {
      const dt_masks_point_raster_t *rp =
        _find_raster_of(forms, f, source, id, any_id, depth + 1);
      if(rp) return rp;
    }
    else if((f->type & DT_MASKS_RASTER) && f->points)
    {
      const dt_masks_point_raster_t *rp = f->points->data;
      if((any_id || rp->id == id) && dt_iop_module_is(source, rp->source)
         && source->multi_priority == rp->instance)
        return rp;
    }
  }
  return NULL;
}

const dt_masks_point_raster_t *dt_masks_group_find_raster_of(GList *forms,
                                                             const dt_masks_form_t *grp,
                                                             const dt_iop_module_t *source,
                                                             const dt_mask_id_t id,
                                                             const gboolean any_id)
{
  return _find_raster_of(forms, grp, source, id, any_id, 0);
}

dt_masks_form_t *dt_masks_create(const dt_masks_type_t type)
{
  dt_masks_form_t *form = calloc(1, sizeof(dt_masks_form_t));
  if(!form) return NULL;

  form->type = type;
  form->version = dt_masks_version();
  form->formid = time(NULL) + form_id++;

  if(type & DT_MASKS_CIRCLE)
    form->functions = &dt_masks_functions_circle;
  else if(type & DT_MASKS_ELLIPSE)
    form->functions = &dt_masks_functions_ellipse;
  else if(type & DT_MASKS_BRUSH)
    form->functions = &dt_masks_functions_brush;
  else if(type & DT_MASKS_PATH)
    form->functions = &dt_masks_functions_path;
  else if(type & DT_MASKS_GRADIENT)
    form->functions = &dt_masks_functions_gradient;
  else if(type & DT_MASKS_GROUP)
    form->functions = &dt_masks_functions_group;
  else if(type & DT_MASKS_PARAMETRIC)
    form->functions = &dt_masks_functions_parametric;
  else if(type & DT_MASKS_RASTER)
    form->functions = &dt_masks_functions_raster;
#ifdef HAVE_AI
  else if(type & DT_MASKS_OBJECT)
    form->functions = &dt_masks_functions_object;
#endif

  if(form->functions && form->functions->sanitize_config)
    form->functions->sanitize_config(type);

  return form;
}

dt_masks_form_t *dt_masks_create_ext(const dt_masks_type_t type)
{
  dt_masks_form_t *form = dt_masks_create(type);

  // all forms created here are registered in
  // darktable.develop->allforms for later cleanup
  if(form)
    darktable.develop->allforms = g_list_append(darktable.develop->allforms, form);

  return form;
}

void dt_masks_replace_current_forms(dt_develop_t *dev, GList *forms)
{
  GList *forms_tmp = dt_masks_dup_forms_deep(forms, NULL);

  while(dev->forms)
  {
    darktable.develop->allforms =
      g_list_append(darktable.develop->allforms, dev->forms->data);
    dev->forms = g_list_delete_link(dev->forms, dev->forms);
  }

  dev->forms = forms_tmp;
}

dt_masks_form_t *dt_masks_get_from_id_ext(GList *forms, const dt_mask_id_t id)
{
  for(; forms; forms = g_list_next(forms))
  {
    dt_masks_form_t *form = forms->data;
    if(form->formid == id) return form;
  }
  return NULL;
}

dt_masks_form_t *dt_masks_get_from_id(const dt_develop_t *dev, const dt_mask_id_t id)
{
  return dt_masks_get_from_id_ext(dev->forms, id);
}

void dt_masks_read_masks_history(dt_develop_t *dev, const dt_imgid_t imgid)
{
  dt_dev_history_item_t *hist_item = NULL;
  const dt_dev_history_item_t *hist_item_last = NULL;
  int num_prev = -1;

  sqlite3_stmt *stmt;
  // clang-format off
  if(dev->snapshot_id == -1)
  {
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT imgid, formid, form, name, version, points, points_count, source, num"
      " FROM main.masks_history"
      " WHERE imgid = ?1"
      "   AND num < ?2"
      " ORDER BY num",
      -1, &stmt, NULL);
    DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, imgid);
    DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, dev->history_end);
  }
  else
  {
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT imgid, formid, form, name, version, points, points_count, source, num"
      " FROM memory.snapshot_masks_history"
      " WHERE id = ?1"
      "   AND num < ?2"
      " ORDER BY num",
      -1, &stmt, NULL);
    DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, dev->snapshot_id);
    DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, dev->history_end);
  }
  // clang-format on

  while(sqlite3_step(stmt) == SQLITE_ROW)
  {
    // db record: 0-img, 1-formid, 2-form_type, 3-name, 4-version,
    // 5-points, 6-points_count, 7-source, 8-num

    // we get the values

    const dt_mask_id_t formid = sqlite3_column_int(stmt, 1);
    const int num = sqlite3_column_int(stmt, 8);
    const dt_masks_type_t type = sqlite3_column_int(stmt, 2);
    dt_masks_form_t *form = dt_masks_create(type);
    form->formid = formid;
    const char *name = (const char *)sqlite3_column_text(stmt, 3);
    g_strlcpy(form->name, name, sizeof(form->name));
    form->version = sqlite3_column_int(stmt, 4);
    form->points = NULL;
    const int nb_points = sqlite3_column_int(stmt, 6);
    const int source_bytes = sqlite3_column_bytes(stmt, 7);
    if(source_bytes == sizeof(float) * 2)
    {
      memcpy(form->source, sqlite3_column_blob(stmt, 7), sizeof(float) * 2);
      form->source[2] = 0.0f;
    }
    else if(source_bytes == sizeof(float) * 3)
    {
      memcpy(form->source, sqlite3_column_blob(stmt, 7), sizeof(float) * 3);
    }
    else if(source_bytes == sizeof(float) * 4)
    {
      // migration: old format stored an unused scale field as source[3]; drop it
      memcpy(form->source, sqlite3_column_blob(stmt, 7), sizeof(float) * 3);
    }
    else
    {
      memset(form->source, 0, sizeof(float) * 3);
    }

    // and now we "read" the blob
    if(form->functions)
    {
      const char *const ptbuf = (char *)sqlite3_column_blob(stmt, 5);
      const size_t point_size = form->functions->point_struct_size;

      // the on-disk stride may be smaller than the current struct when an
      // older edit predates a field being appended to a point struct. So
      // far this affects group points, which gained the per-shape
      // refinement block in masks v7, the custom group-name field in masks
      // v8, the persistent group-opacity field in masks v9, and the
      // first-class group_start field in masks v10. We read the historic
      // stride and zero-fill the remainder so older masks load with
      // refinements disabled, no custom name, and no explicit group break
      // (all neutral at 0). The zero-filled group_opacity is NOT neutral (0
      // would zero out the whole group) -- the version migration below
      // fixes it up to 1.0 explicitly; group_start's zero-fill is neutral
      // (see its own comment in masks.h), but the migration still has to
      // carry forward any break that was set in the old, pre-v10 bit.
      size_t read_size = point_size;
      if(type & DT_MASKS_GROUP)
      {
        if(form->version < 7)
          read_size = offsetof(dt_masks_point_group_t, refinement);
        else if(form->version < 8)
          read_size = offsetof(dt_masks_point_group_t, name);
        else if(form->version < 9)
          read_size = offsetof(dt_masks_point_group_t, group_opacity);
        else if(form->version < 10)
          read_size = offsetof(dt_masks_point_group_t, group_start);
      }

      for(int i = 0; i < nb_points; i++)
      {
        char *point = calloc(1, point_size);
        memcpy(point, ptbuf + i * read_size, MIN(read_size, point_size));
        form->points = g_list_append(form->points, point);
      }
    }

    if(form->version != dt_masks_version())
    {
      if(dt_masks_legacy_params(dev, form, form->version, dt_masks_version()))
      {
        const char *fname =
          dev->image_storage.filename + strlen(dev->image_storage.filename);
        while(fname > dev->image_storage.filename && *fname != '/') fname--;
        if(fname > dev->image_storage.filename) fname++;

        dt_print(DT_DEBUG_ALWAYS,
                 "[_dev_read_masks_history] %s (imgid `%i'):"
                 " mask version mismatch: history is %d, darktable is %d",
                 fname, imgid, form->version, dt_masks_version());
        dt_control_log(_("%s: mask version mismatch: %d != %d"),
                       fname, dt_masks_version(), form->version);

        continue;
      }
    }

    // if this is a new history entry let's find it
    if(num_prev != num)
    {
      hist_item = NULL;
      for(GList *history = dev->history; history; history = g_list_next(history))
      {
        dt_dev_history_item_t *hitem = history->data;
        if(hitem->num == num)
        {
          hist_item = hitem;
          break;
        }
      }
      num_prev = num;
    }
    // add the form to the history entry
    if(hist_item)
    {
      hist_item->forms = g_list_append(hist_item->forms, form);
    }
    else
      dt_print(DT_DEBUG_ALWAYS,
               "[_dev_read_masks_history] can't find history entry %i"
               " while adding mask %s(%i)",
               num, form->name, formid);

    if(num < dev->history_end) hist_item_last = hist_item;
  }
  sqlite3_finalize(stmt);

  // and we update the current forms snapshot
  dt_masks_replace_current_forms(dev, (hist_item_last)?hist_item_last->forms:NULL);
}

void dt_masks_write_masks_history_item(const dt_imgid_t imgid,
                                       const int num,
                                       const dt_masks_form_t *form)
{
  sqlite3_stmt *stmt;

  // write the form into the database
  // clang-format off
  DT_DEBUG_SQLITE3_PREPARE_V2
    (dt_database_get(darktable.db),
     "INSERT INTO main.masks_history (imgid, num, formid, form, name,"
     "                                version, points, points_count,source)"
     " VALUES (?1, ?9, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
     -1, &stmt, NULL);
  // clang-format on
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, imgid);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 9, num);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, form->formid);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 3, form->type);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 4, form->name, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_BLOB(stmt, 8, form->source, 3 * sizeof(float), SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 5, form->version);
  if(form->functions)
  {
    const size_t point_size = form->functions->point_struct_size;
    const guint nb = g_list_length(form->points);
    char *const restrict ptbuf = malloc(nb * point_size);
    int pos = 0;
    for(GList *points = form->points; points; points = g_list_next(points))
    {
      memcpy(ptbuf + pos, points->data, point_size);
      pos += point_size;
    }
    DT_DEBUG_SQLITE3_BIND_BLOB(stmt, 6, ptbuf, nb * point_size, SQLITE_TRANSIENT);
    DT_DEBUG_SQLITE3_BIND_INT(stmt, 7, nb);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    free(ptbuf);
  }
}

void dt_masks_free_form(dt_masks_form_t *form)
{
  if(!form) return;
  g_list_free_full(form->points, free);
  form->points = NULL;
  free(form);
}

gboolean dt_masks_events_mouse_leave(dt_iop_module_t *module)
{
  dt_develop_t *dev = darktable.develop;
  if(dev->form_gui)
  {
    dt_masks_form_gui_t *gui = dev->form_gui;
    float zoom_x, zoom_y;
    dt_dev_get_viewport_params(&dev->full, NULL, NULL, &zoom_x, &zoom_y);

    float wd, ht;
    dt_masks_get_image_size(&wd, &ht, NULL, NULL);
    gui->posx = (.5f + zoom_x) * wd;
    gui->posy = (.5f + zoom_y) * ht;

    dt_control_hinter_message("");
  }
  return FALSE;
}

gboolean dt_masks_events_mouse_enter(dt_iop_module_t *module)
{
  return FALSE;
}

// return true in case of something has been exposed
gboolean dt_masks_events_mouse_moved(dt_iop_module_t *module,
                                     const float pzx,
                                     const float pzy,
                                     const double pressure,
                                     const int which,
                                     const float zoom_scale)
{
/*  For UI responsivness we want to avoid further processing if possible,
    so we do some tests and possibly return immediately.
    *module is either the module having focus or the mask manager (module == NULL)

    We can skip further processing if we have
    1. a module and it's not enabled
    2. the mask manager and it is not expanded
*/
  const gboolean skipped = (module && !module->enabled);

  dt_print(DT_DEBUG_VERBOSE,
    "[dt_masks_events_mouse_moved] %s %s",
    module ? module->so->op : "mask manager",
    skipped ? "skipped" : "");

  if(skipped) return FALSE;

  dt_masks_form_gui_t *gui = darktable.develop->form_gui;
  dt_masks_form_t *form = darktable.develop->form_visible;

  if(gui)
  {
    // This assume that if this event is generated the mouse is over
    // the center window
    float wd, ht;
    dt_masks_get_image_size(&wd, &ht, NULL, NULL);
    gui->posx = pzx * wd;
    gui->posy = pzy * ht;
  }

  // form->points can be mutated below (dragging a node); pixelpipe worker
  // threads deep-copy dev->forms concurrently in dt_dev_pixelpipe_process, so
  // this must be serialized against that read (see history_mutex there).
  dt_pthread_mutex_lock(&darktable.develop->history_mutex);

  int rep = 0;
  if(form->functions)
    rep = form->functions->mouse_moved(module, pzx, pzy, pressure, which, zoom_scale, form, 0, gui, 0);

  dt_pthread_mutex_unlock(&darktable.develop->history_mutex);

  if(gui) _set_hinter_message(gui, form);

  return rep != 0;
}

gboolean dt_masks_events_button_released(dt_iop_module_t *module,
                                         const float pzx,
                                         const float pzy,
                                         const int which,
                                         const uint32_t state,
                                         const float zoom_scale)
{
  dt_develop_t *dev = darktable.develop;
  dt_masks_form_t *form = dev->form_visible;
  dt_masks_form_gui_t *gui = dev->form_gui;

  DT_ENTER_GUI_UPDATE();
  if(dev->mask_form_selected_id)
    dt_dev_masks_selection_change(dev, module, dev->mask_form_selected_id);
  DT_LEAVE_GUI_UPDATE();

  gboolean ret = FALSE;
  if(form->functions)
  {
    // serialized against the pixelpipe's dt_masks_dup_forms_deep read of
    // dev->forms/form->points, see history_mutex use in dt_dev_pixelpipe_process.
    dt_pthread_mutex_lock(&dev->history_mutex);
    ret = form->functions->button_released(module, pzx, pzy, which, state, form, 0, gui, 0);
    form->functions->mouse_moved(module, pzx, pzy, 0, which, zoom_scale, form, 0, gui, 0);
    dt_pthread_mutex_unlock(&dev->history_mutex);
  }

  return ret;
}

gboolean dt_masks_events_button_pressed(dt_iop_module_t *module,
                                        const float pzx,
                                        const float pzy,
                                        const double pressure,
                                        const int which,
                                        const int type,
                                        const uint32_t state)
{
  dt_masks_form_t *form = darktable.develop->form_visible;
  dt_masks_form_gui_t *gui = darktable.develop->form_gui;

  // allow to select a shape inside an iop
  if(gui && which == GDK_BUTTON_PRIMARY)
  {
    const dt_masks_form_t *sel = NULL;

    if((gui->form_selected
        || gui->source_selected
        || gui->point_selected
        || gui->seg_selected
        || gui->feather_selected)
       && !gui->creation && gui->group_edited >= 0)
    {
      // we get the selected form
      const dt_masks_point_group_t *fpt = g_list_nth_data(form->points, gui->group_edited);
      if(fpt)
      {
        sel = dt_masks_get_from_id(darktable.develop, fpt->formid);
      }
    }

    dt_masks_select_form(module, sel);
  }

  if(form->functions)
  {
    // serialized against the pixelpipe's dt_masks_dup_forms_deep read of
    // dev->forms/form->points, see history_mutex use in dt_dev_pixelpipe_process.
    dt_pthread_mutex_lock(&darktable.develop->history_mutex);
    const gboolean ret = form->functions->button_pressed(
                           module, pzx, pzy, pressure, which, type, state, form, 0, gui, 0) ||
                         which == 3; // swallow right-clicks so right-drag rotate is disabled
    dt_pthread_mutex_unlock(&darktable.develop->history_mutex);
    return ret;
  }
  return FALSE;
}

gboolean dt_masks_scroll_over_mask(void)
{
  const dt_masks_form_t *form = darktable.develop->form_visible;
  const dt_masks_form_gui_t *gui = darktable.develop->form_gui;
  if(!form || !gui)
    return FALSE;

  // During shape creation, scroll adjusts size/border and stays over the mask.
  // Paths/polygons are built node by node and do not use scroll here.
  if(gui->creation)
    return !(form->type & DT_MASKS_PATH);

  // Otherwise form_visible is the group. group_edited is the active sub-form or
  // -1 when none. Scroll should count only when a sub-form is active.
  return gui->group_edited >= 0;
}

gboolean dt_masks_events_mouse_scrolled(dt_iop_module_t *module,
                                        const float pzx,
                                        const float pzy,
                                        const gboolean up,
                                        const uint32_t state)
{
  dt_masks_form_t *form = darktable.develop->form_visible;
  dt_masks_form_gui_t *gui = darktable.develop->form_gui;

  gboolean ret = FALSE;
  const gboolean incr = dt_mask_scroll_increases(up);

  if(form->functions)
  {
    // serialized against the pixelpipe's dt_masks_dup_forms_deep read of
    // dev->forms/form->points, see history_mutex use in dt_dev_pixelpipe_process.
    dt_pthread_mutex_lock(&darktable.develop->history_mutex);
    ret = (form->functions->mouse_scrolled(module, pzx, pzy,
                                          incr ? 1 : 0,
                                          state, form, 0, gui, 0)) != 0;
    dt_pthread_mutex_unlock(&darktable.develop->history_mutex);
  }

  if(gui)
  {
    // Do not update brush opacity here; it is mask density.
    if(gui->creation && dt_modifier_is(state, GDK_CONTROL_MASK))
    {
      float opacity = dt_conf_get_float("plugins/darkroom/masks/opacity");
      const float amount = incr ? 0.05f : -0.05f;

      opacity = CLAMP(opacity + amount, 0.05f, 1.0f);
      dt_conf_set_float("plugins/darkroom/masks/opacity", opacity);

      dt_toast_log(_("opacity: %.0f%%"), opacity * 100);
      dt_dev_masks_list_change(darktable.develop);

      ret = TRUE;
    }

    _set_hinter_message(gui, form);
  }

  return ret;
}

// visualize mask from viewport
void dt_masks_events_post_expose(const dt_iop_module_t *module,
                                 cairo_t *cr,
                                 const int32_t width,
                                 const int32_t height,
                                 const float pzx,
                                 const float pzy,
                                 const float zoom_scale)
{
  const dt_develop_t *dev = darktable.develop;
  dt_masks_form_t *form = dev->form_visible;
  dt_masks_form_gui_t *gui = dev->form_gui;
  if(!gui) return;
  if(!form) return;

  cairo_save(cr);
  cairo_set_source_rgb(cr, .3, .3, .3);

  cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);

  // we update the form if needed
  // add preview when creating a circle, ellipse and gradient
  if(!(((form->type & DT_MASKS_CIRCLE)
        || (form->type & DT_MASKS_ELLIPSE)
        || (form->type & DT_MASKS_GRADIENT))
       && gui->creation))
    dt_masks_gui_form_test_create(form, gui, module);

  // draw form
  if(form->type & DT_MASKS_GROUP)
    dt_group_events_post_expose(cr, zoom_scale, form, gui);
  else if(form->functions)
    form->functions->post_expose(cr, zoom_scale, gui, 0, g_list_length(form->points));

  cairo_restore(cr);
}

void dt_masks_clear_form_gui(const dt_develop_t *dev)
{
  if(!dev->form_gui) return;
  if(dev->form_gui->scratchpad_cleanup)
  {
    dev->form_gui->scratchpad_cleanup(dev->form_gui);
    dev->form_gui->scratchpad_cleanup = NULL;
  }
  g_list_free_full(dev->form_gui->points, dt_masks_form_gui_points_free);
  dev->form_gui->points = NULL;
  dt_masks_dynbuf_free(dev->form_gui->guipoints);
  dev->form_gui->guipoints = NULL;
  dt_masks_dynbuf_free(dev->form_gui->guipoints_payload);
  dev->form_gui->guipoints_payload = NULL;
  dev->form_gui->guipoints_count = 0;
  dev->form_gui->pipe_hash = DT_INVALID_HASH;
  dev->form_gui->formid = 0;
  dev->form_gui->dx = dev->form_gui->dy = 0.0f;
  dev->form_gui->scrollx = dev->form_gui->scrolly = 0.0f;
  dev->form_gui->form_selected = dev->form_gui->border_selected =
    dev->form_gui->form_dragging = dev->form_gui->form_rotating =
    dev->form_gui->border_toggling = dev->form_gui->gradient_toggling = FALSE;
  dev->form_gui->source_selected = dev->form_gui->source_dragging = FALSE;
  dev->form_gui->source_rotating = dev->form_gui->counter_rotate_source = FALSE;
  dev->form_gui->rotate_about_source = FALSE;
  dev->form_gui->pivot_selected = FALSE;
  dev->form_gui->point_border_selected = dev->form_gui->seg_selected =
    dev->form_gui->point_selected = dev->form_gui->feather_selected = -1;
  dev->form_gui->point_border_dragging = dev->form_gui->seg_dragging =
    dev->form_gui->feather_dragging = dev->form_gui->point_dragging = -1;
  dev->form_gui->creation_closing_form = dev->form_gui->creation = FALSE;
  if(dt_conf_get_bool("plugins/darkroom/masks/opacity_not_sticky"))
    dt_conf_set_float("plugins/darkroom/masks/opacity", 1.0f);
  dev->form_gui->pressure_sensitivity = DT_MASKS_PRESSURE_OFF;
  dev->form_gui->creation_module = NULL;
  dev->form_gui->point_edited = -1;

  dev->form_gui->group_edited = -1;
  dev->form_gui->group_selected = -1;
  dev->form_gui->edit_mode = DT_MASKS_EDIT_OFF;
  // allow to select a shape inside an iop
  dt_masks_select_form(NULL, NULL);
}

void dt_masks_change_form_gui(dt_masks_form_t *newform)
{
  const dt_masks_form_t *old = darktable.develop->form_visible;

  // the module whose flexi-panel pending-row placeholder (see
  // _build_masks_list's pending-row synthesis in blend_gui.c) is on screen
  // right now. Captured before dt_masks_clear_form_gui() below wipes both
  // creation and creation_module.
  dt_iop_module_t *const was_creating =
    (darktable.develop->form_gui && darktable.develop->form_gui->creation)
      ? darktable.develop->form_gui->creation_module
      : NULL;

  dt_masks_clear_form_gui(darktable.develop);
  darktable.develop->form_visible = newform;

  /* update sticky accels window */
  if(newform != old
     && darktable.view_manager->accels_window.window
     && darktable.view_manager->accels_window.sticky)
    dt_view_accels_refresh(darktable.view_manager);

  if(newform && newform->type != DT_MASKS_GROUP)
    darktable.develop->form_gui->creation = TRUE;

  DT_ENTER_GUI_UPDATE();
  dt_dev_masks_selection_change(darktable.develop, NULL, 0);
  DT_LEAVE_GUI_UPDATE();

  // creation just ended, or moved to another module: drop the pending row the
  // old owner is still showing. The shape modules' right-click-cancel handlers
  // refresh the panel themselves, but abandoning creation any other way (a
  // click outside the canvas, focusing another module) reaches none of them and
  // used to leave the placeholder behind with nothing to dismiss it.
  if(was_creating
     && (!darktable.develop->form_gui->creation
         || darktable.develop->form_gui->creation_module != was_creating))
    dt_iop_gui_blend_masks_creation_ended(was_creating);
}

void dt_masks_reset_form_gui(void)
{
  dt_masks_change_form_gui(NULL);
  const dt_iop_module_t *m = dt_dev_gui_module();
  if(m
     && (m->flags() & IOP_FLAGS_SUPPORTS_BLENDING)
     && !(m->flags() & IOP_FLAGS_NO_MASKS)
     && m->blend_data)
  {
    dt_iop_gui_blend_data_t *bd = m->blend_data;
    bd->masks_shown = DT_MASKS_EDIT_OFF;

    if(bd->masks_edit)
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_edit), 0);

    for(int n = 0; n < DEVELOP_MASKS_NB_SHAPES; n++)
      if(bd->masks_shapes[n])
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_shapes[n]), 0);
  }
}

void dt_masks_reset_show_masks_icons(void)
{
  if(darktable.develop->first_load) return;
  for(GList *modules = darktable.develop->iop; modules; modules = g_list_next(modules))
  {
    const dt_iop_module_t *m = modules->data;
    if(m
       && (m->flags() & IOP_FLAGS_SUPPORTS_BLENDING)
       && !(m->flags() & IOP_FLAGS_NO_MASKS))
    {
      dt_iop_gui_blend_data_t *bd = m->blend_data;
      if(!bd) break;  // TODO: this doesn't look right. Why do we
                      // break the while look as soon as one module
                      // has no blend_data?
      bd->masks_shown = DT_MASKS_EDIT_OFF;
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_edit), FALSE);
      gtk_widget_queue_draw(bd->masks_edit);
      for(int n = 0; n < DEVELOP_MASKS_NB_SHAPES; n++)
      {
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_shapes[n]), 0);
        gtk_widget_queue_draw(bd->masks_shapes[n]);
      }
    }
  }
}

dt_masks_edit_mode_t dt_masks_get_edit_mode(void)
{
  return darktable.develop->form_gui
    ? darktable.develop->form_gui->edit_mode
    : DT_MASKS_EDIT_OFF;
}

static inline gboolean _masks_is_restricted_mode(void)
{
  return dt_masks_get_edit_mode() == DT_MASKS_EDIT_RESTRICTED;
}

void dt_masks_set_edit_mode(dt_iop_module_t *module,
                            const dt_masks_edit_mode_t value)
{
  if(!module) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd) return;

  dt_masks_form_t *grp = NULL;
  dt_masks_form_t *form =
    dt_masks_get_from_id(module->dev, module->blend_params->mask_id);

  if(value && form)
  {
    grp = dt_masks_create_ext(DT_MASKS_GROUP);
    grp->formid = NO_MASKID;
    dt_masks_group_ungroup(grp, form);
  }

  const dt_masks_edit_mode_t old_shown = bd->masks_shown;

  // leaving edit mode cancels any solo-edit so its toggle does not linger active
  if(value == DT_MASKS_EDIT_OFF)
  {
    bd->soloedit_formid = INVALID_MASKID;
  }

  bd->masks_shown = value;
  dt_masks_change_form_gui(grp);
  darktable.develop->form_gui->edit_mode = value;

  DT_ENTER_GUI_UPDATE();
  dt_dev_masks_selection_change(darktable.develop, NULL,
                                value && form ? form->formid : NO_MASKID);
  DT_LEAVE_GUI_UPDATE();

  if(bd->masks_support)
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_edit),
                                 value == DT_MASKS_EDIT_OFF ? FALSE : TRUE);

  // Hiding (or changing) the mask overlay shrinks the editable canvas back to
  // the image, so a viewport that was panned to reach off-image handles is now
  // stale. Re-validate it: this no-op clamp snaps the pan back into bounds so
  // the picture re-centres immediately instead of lingering off-centre (which
  // also leaves the renderer churning) until the next manual pan.
  //
  // Not during a bulk history refresh: that reaches us from the
  // dt_iop_gui_update() loop in dt_dev_pop_history_items(), which holds
  // dev->history_mutex, whereas dt_dev_zoom_move() takes global_mutex and then
  // history_mutex -- the reverse of the order a pixelpipe worker uses, so the
  // two deadlock. Nothing is lost when hiding the overlay: after a history
  // reload the dirty pipe runs the image-only clamp itself. Worker validation
  // deliberately does not inspect mutable GUI overlay points.
  // Also avoid if we didn't change edit mode
  if(!darktable.develop->history_updating && old_shown != value)
    dt_dev_zoom_move(&darktable.develop->full, DT_ZOOM_MOVE, 0.0f, 0, 0.0f, 0.0f, TRUE);

  dt_control_queue_redraw_center();
}

void dt_masks_set_edit_mode_single_form(dt_iop_module_t *module,
                                        const dt_mask_id_t formid,
                                        const dt_masks_edit_mode_t value)
{
  if(!module) return;

  dt_masks_form_t *grp = dt_masks_create_ext(DT_MASKS_GROUP);

  const dt_mask_id_t grid = module->blend_params->mask_id;
  const dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, formid);
  if(form)
  {
    dt_masks_point_group_t *fpt = calloc(1, sizeof(dt_masks_point_group_t));
    fpt->formid = formid;
    fpt->parentid = grid;
    fpt->state = DT_MASKS_STATE_SHOW | DT_MASKS_STATE_USE;
    fpt->opacity = 1.0f;
    grp->points = g_list_append(grp->points, fpt);
  }

  dt_masks_form_t *grp2 = dt_masks_create_ext(DT_MASKS_GROUP);
  grp2->formid = NO_MASKID;
  dt_masks_group_ungroup(grp2, grp);
  dt_masks_change_form_gui(grp2);
  darktable.develop->form_gui->edit_mode = value;

  DT_ENTER_GUI_UPDATE();
  dt_dev_masks_selection_change(darktable.develop, NULL, value && form ? formid : NO_MASKID);
  DT_LEAVE_GUI_UPDATE();

  dt_control_queue_redraw_center();
}

void dt_masks_set_edit_mode_forms(dt_iop_module_t *module,
                                  GList *formids,
                                  const dt_masks_edit_mode_t value)
{
  if(!module) return;

  dt_masks_form_t *grp = dt_masks_create_ext(DT_MASKS_GROUP);
  const dt_mask_id_t grid = module->blend_params->mask_id;

  // build a scratch group holding only the requested forms, so only their
  // outlines/handles become editable on the canvas. The actual mask computation
  // (blend_params->mask_id) is untouched, so every shape still composites.
  dt_mask_id_t first = NO_MASKID;
  for(GList *l = formids; l; l = g_list_next(l))
  {
    const dt_mask_id_t formid = GPOINTER_TO_INT(l->data);
    const dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, formid);
    if(!form) continue;
    if(!dt_is_valid_maskid(first)) first = formid;
    dt_masks_point_group_t *fpt = calloc(1, sizeof(dt_masks_point_group_t));
    fpt->formid = formid;
    fpt->parentid = grid;
    fpt->state = DT_MASKS_STATE_SHOW | DT_MASKS_STATE_USE;
    fpt->opacity = 1.0f;
    grp->points = g_list_append(grp->points, fpt);
  }

  dt_masks_form_t *grp2 = dt_masks_create_ext(DT_MASKS_GROUP);
  grp2->formid = NO_MASKID;
  dt_masks_group_ungroup(grp2, grp);
  dt_masks_change_form_gui(grp2);
  darktable.develop->form_gui->edit_mode = value;

  DT_ENTER_GUI_UPDATE();
  dt_dev_masks_selection_change(darktable.develop, NULL,
                                value && dt_is_valid_maskid(first) ? first : NO_MASKID);
  DT_LEAVE_GUI_UPDATE();

  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(bd)
  {
    bd->masks_shown = value;
    if(bd->masks_support)
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_edit),
                                   value == DT_MASKS_EDIT_OFF ? FALSE : TRUE);
  }

  dt_control_queue_redraw_center();
}

void dt_masks_iop_edit_toggle_callback(GtkToggleButton *togglebutton,
                                       dt_iop_module_t *module)
{
  if(!module) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(module->blend_params->mask_id == NO_MASKID)
  {
    bd->masks_shown = DT_MASKS_EDIT_OFF;
    return;
  }

  // reset the gui
  dt_masks_set_edit_mode(
    module,
    (bd->masks_shown == DT_MASKS_EDIT_OFF ? DT_MASKS_EDIT_FULL : DT_MASKS_EDIT_OFF));
}

void dt_masks_group_update_name(dt_iop_module_t *module)
{
  dt_masks_form_t *grp = _group_from_module(darktable.develop, module);
  if(!grp)
    return;

  _set_group_name_from_module(module, grp);
  dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
  dt_masks_iop_update(module);
}

void dt_masks_group_add_members_of(dt_masks_form_t *grp, const dt_masks_form_t *src_grp)
{
  for(GList *points = src_grp->points; points; points = g_list_next(points))
  {
    const dt_masks_point_group_t *pt = points->data;
    if(dt_masks_point_is_marker(pt))
    {
      dt_masks_group_copy_marker(darktable.develop->forms, grp, pt);
      continue;
    }
    const dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, pt->formid);
    // shapes are shared with the source, but a parametric channel is copied:
    // two modules hardly want the same range, and a shared one would move both
    if(form && (form->type & DT_MASKS_PARAMETRIC))
      form = dt_masks_get_from_id(darktable.develop,
                                  dt_masks_form_copy(darktable.develop, form->formid));
    if(form)
    {
      dt_masks_point_group_t *grpt = dt_masks_group_add_form(grp, form);
      if(grpt)
      {
        grpt->state = pt->state;
        grpt->opacity = pt->opacity;
      }
    }
  }
}

void dt_masks_iop_use_same_as(dt_iop_module_t *module,
                              dt_iop_module_t *src)
{
  if(!module || !src) return;

  // we get the source group
  const dt_mask_id_t srcid = src->blend_params->mask_id;
  dt_masks_form_t *src_grp = dt_masks_get_from_id(darktable.develop, srcid);
  if(!src_grp || src_grp->type != DT_MASKS_GROUP) return;

  // is there already a masks group for this module ?
  dt_masks_form_t *grp = _group_from_module(darktable.develop, module);
  if(!grp)
  {
    grp = _group_create(darktable.develop, module, DT_MASKS_GROUP);
  }
  dt_masks_group_add_members_of(grp, src_grp);

  // we save the group
  dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
}

void dt_masks_iop_update(dt_iop_module_t *module)
{
  if(!module) return;

  dt_iop_gui_update(module);
  dt_iop_gui_update_masks(module);
}

void dt_masks_form_remove(dt_iop_module_t *module,
                          dt_masks_form_t *grp,
                          dt_masks_form_t *form)
{
  if(!form) return;
  const dt_mask_id_t id = form->formid;
  // a committed AI-mask bundle (DT_MASKS_OBJECT, see _register_vectorized_forms
  // in masks/object.c) is a valid parent too: canvas node-editing routes a
  // whole-shape delete here with `grp` resolved from the flattened scratch
  // group's own parentid (see dt_masks_group_ungroup), which for a bundle
  // child is the bundle's own formid, not the module's real top group.
  if(grp && !(grp->type & (DT_MASKS_GROUP | DT_MASKS_OBJECT))) return;

  if(!(form->type & (DT_MASKS_CLONE|DT_MASKS_NON_CLONE))
     && grp)
  {
    // we try to remove the form from the masks group
    gboolean ok = FALSE;
    for(GList *forms = grp->points; forms; forms = g_list_next(forms))
    {
      dt_masks_point_group_t *grpt = forms->data;
      if(grpt->formid == id)
      {
        ok = TRUE;
        grp->points = g_list_remove(grp->points, grpt);
        free(grpt);
        break;
      }
    }
    if(ok)
    {
      dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
      dt_masks_iop_update(module);
    }
    if(ok && grp->points == NULL) dt_masks_form_remove(module, NULL, grp);
    return;
  }

  if(form->type & DT_MASKS_GROUP && form->type & DT_MASKS_CLONE)
  {
    // when removing a cloning group the children have to be removed,
    // too, as they won't be shown in the mask manager and are thus
    // not accessible afterwards.
    while(form->points)
    {
      const dt_masks_point_group_t *group_child = form->points->data;
      dt_masks_form_t *child =
        dt_masks_get_from_id(darktable.develop, group_child->formid);
      dt_masks_form_remove(module, form, child);
      // no need to do anything to form->points, the recursive call
      // will have removed child from the list
    }
  }

  // if we are here that mean we have to permanently delete this form
  // we drop the form from all modules
  gboolean form_removed = FALSE;
  for(GList *iops = darktable.develop->iop; iops; iops = g_list_next(iops))
  {
    dt_iop_module_t *m = iops->data;
    if(m->flags() & IOP_FLAGS_SUPPORTS_BLENDING)
    {
      // is the form the base group of the iop ?
      if(id == m->blend_params->mask_id)
      {
        dt_print(DT_DEBUG_MASKS,
                 "[masks] dt_masks_form_remove '%s': mask_id %d->NO_MASKID"
                 " (form %d permanently removed)",
                 m->op, m->blend_params->mask_id, id);
        m->blend_params->mask_id = NO_MASKID;
        dt_masks_iop_update(m);
        dt_dev_add_history_item(darktable.develop, m, TRUE);
      }
      else
      {
        dt_masks_form_t *iopgrp = _group_from_module(darktable.develop, m);
        if(iopgrp && (iopgrp->type & DT_MASKS_GROUP))
        {
          gboolean ok = FALSE;
          GList *forms = iopgrp->points;
          while(forms)
          {
            dt_masks_point_group_t *grpt = forms->data;
            if(grpt->formid == id)
            {
              ok = TRUE;
              iopgrp->points = g_list_remove(iopgrp->points, grpt);
              free(grpt);
              forms = iopgrp->points; // jump back to start of list
              continue;
            }
            forms = g_list_next(forms); // advance to next form
          }
          if(ok)
          {
            form_removed = TRUE;
            dt_masks_iop_update(m);
            if(iopgrp->points == NULL) dt_masks_form_remove(m, NULL, iopgrp);
          }
        }
      }
    }
  }
  // we drop the form from the general list
  for(GList *forms = darktable.develop->forms; forms; forms = g_list_next(forms))
  {
    const dt_masks_form_t *f = forms->data;
    if(f->formid == id)
    {
      darktable.develop->forms = g_list_remove(darktable.develop->forms, f);
      form_removed = TRUE;
      break;
    }
  }
  if(form_removed) dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
}

dt_masks_remove_target_t dt_masks_remove_shape_target(const dt_iop_module_t *module,
                                                      dt_masks_form_t **form,
                                                      dt_mask_id_t *parentid,
                                                      const gboolean whole)
{
  // an AI object is selected as one unit, so removing it by a path removes
  // the object, unless the user stepped into it. Removing a single path
  // (inside the object, node by node, or from its row in the panel) edits
  // the object instead, down to its last path, which takes the object away
  // like the panel's delete: emptying it would remove it from every module
  // that links it
  dt_masks_form_t *object = dt_masks_get_from_id(darktable.develop, *parentid);
  if(object && (object->type & DT_MASKS_OBJECT) && module && module->blend_params)
  {
    const dt_masks_form_gui_t *gui = darktable.develop->form_gui;
    const gboolean one_path = !whole || (gui && gui->entered_object == object->formid);
    if(one_path && !g_list_shorter_than(object->points, 2)) return DT_MASKS_REMOVE_PATH;
    *form = object;
    *parentid = module->blend_params->mask_id;
  }

  // a shape of this module's own flexi mask goes the way the panel's delete
  // takes it: it leaves this module only, even when another module links it,
  // and emptying its group leaves the group in place
  if(module && module->blend_data && module->blend_params
     && (module->blend_params->mask_mode & DEVELOP_MASK_FLEXI)
     && *parentid == module->blend_params->mask_id
     && !((*form)->type & (DT_MASKS_CLONE | DT_MASKS_NON_CLONE)))
    return DT_MASKS_REMOVE_ELEMENT;
  return DT_MASKS_REMOVE_FROM_PARENT;
}

void dt_masks_remove_shape(dt_iop_module_t *module,
                           dt_masks_form_t *form,
                           dt_mask_id_t parentid,
                           const gboolean whole)
{
  if(dt_masks_remove_shape_target(module, &form, &parentid, whole) == DT_MASKS_REMOVE_ELEMENT)
    dt_iop_gui_blend_delete_element(module, form->formid);
  else
    dt_masks_form_remove(module, dt_masks_get_from_id(darktable.develop, parentid), form);
}

float dt_masks_form_change_opacity(dt_masks_form_t *form,
                                   const dt_mask_id_t parentid,
                                   const float amount)
{
  if(!form) return 0;
  dt_masks_form_t *grp = dt_masks_get_from_id(darktable.develop, parentid);
  // a committed AI-mask bundle (DT_MASKS_OBJECT, see _register_vectorized_forms
  // in masks/object.c) is a valid parent too: ctrl+scroll on a bundle child
  // resolves `parentid` to the bundle's own formid (see dt_masks_group_ungroup's
  // flattening), not the module's real top group.
  if(!grp || !(grp->type & (DT_MASKS_GROUP | DT_MASKS_OBJECT))) return 0;

  // we first need to test if the opacity can be set to the form
  if(form->type & DT_MASKS_GROUP) return 0;
  const dt_mask_id_t id = form->formid;

  // so we change the value inside the group
  for(GList *fpts = grp->points; fpts; fpts = g_list_next(fpts))
  {
    dt_masks_point_group_t *fpt = fpts->data;
    if(fpt->formid == id)
    {
      // 0, not the 0.05 floor this used to carry (upstream c646d7e959): that
      // clamp existed because a 0% shape was silently indistinguishable from a
      // live one, and the on-canvas toast below plus the mask panel's
      // low-opacity warning badge now say so out loud. The default opacity for
      // *new* shapes is still floored (see dt_masks_events_mouse_scrolled) --
      // there, a forgotten 0 makes every shape drawn afterwards invisible.
      const float opacity = CLAMP(fpt->opacity + amount, 0.0f, 1.0f);
      if(opacity != fpt->opacity)
      {
        fpt->opacity = opacity;
        dt_toast_log(_("opacity: %.0f%%"), opacity * 100);
        dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
      }
      return opacity;
    }
  }
  return 0;
}

void dt_masks_form_move(dt_masks_form_t *grp,
                        const dt_mask_id_t formid,
                        const gboolean up)
{
  if(!grp || !(grp->type & DT_MASKS_GROUP)) return;

  // we search the form in the group
  dt_masks_point_group_t *grpt = NULL;
  guint pos = 0;
  for(GList *fpts = grp->points; fpts; fpts = g_list_next(fpts))
  {
    dt_masks_point_group_t *fpt = fpts->data;
    if(fpt->formid == formid)
    {
      grpt = fpt;
      break;
    }
    pos++;
  }

  // we remove the form and read it
  if(grpt)
  {
    if(!up && pos == 0) return;
    if(up && pos == g_list_length(grp->points) - 1) return;

    grp->points = g_list_remove(grp->points, grpt);
    if(!up)
      pos -= 1;
    else
      pos += 1;
    grp->points = g_list_insert(grp->points, grpt, pos);
  }
}

static int _find_in_group(const dt_masks_form_t *grp,
                          const dt_mask_id_t formid,
                          const int depth)
{
  if(!(grp->type & DT_MASKS_GROUP) || depth > DT_MASKS_NESTING_MAX) return 0;
  if(grp->formid == formid) return 1;

  int nb = 0;
  for(GList *forms = grp->points; forms; forms = g_list_next(forms))
  {
    const dt_masks_point_group_t *grpt = forms->data;
    const dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, grpt->formid);
    if(form)
    {
      if(form->type & DT_MASKS_GROUP) nb += _find_in_group(form, formid, depth + 1);
    }
  }
  return nb;
}

dt_masks_state_t dt_masks_get_default_operator(const dt_masks_form_t *form)
{
  // parametric forms are ordinary group members now: a new one adopts the
  // selected group's operator (the shared default_operator pref, set when a group
  // or staged group is the active target), exactly like a drawn shape. With no
  // group selected (pref unset / "automatic") fall back to multiply, the
  // historic sensible default for a parametric "limit" mask. This only ever
  // affects flexi forms (parametric-as-form does not exist in legacy edits).
  if(form && (form->type & (DT_MASKS_PARAMETRIC | DT_MASKS_RASTER)))
  {
    const char *pop = dt_conf_get_string_const("plugins/darkroom/masks/default_operator");
    if(pop && *pop)
    {
      if(!strcmp(pop, "union")) return DT_MASKS_STATE_UNION;
      if(!strcmp(pop, "intersection")) return DT_MASKS_STATE_INTERSECTION;
      if(!strcmp(pop, "difference")) return DT_MASKS_STATE_DIFFERENCE;
      if(!strcmp(pop, "sum")) return DT_MASKS_STATE_SUM;
      if(!strcmp(pop, "exclusion")) return DT_MASKS_STATE_EXCLUSION;
      if(!strcmp(pop, "multiply")) return DT_MASKS_STATE_MULTIPLY;
    }
    return DT_MASKS_STATE_MULTIPLY;
  }

  // the user can pick a default composition operator for newly added
  // shapes in the mask manager. when unset (or "automatic") we keep the
  // historic behavior: brushes default to sum, everything else to union.
  // this only affects forms added from now on, never existing edits.
  const char *op = dt_conf_get_string_const("plugins/darkroom/masks/default_operator");
  if(op && *op)
  {
    if(!strcmp(op, "union")) return DT_MASKS_STATE_UNION;
    if(!strcmp(op, "intersection")) return DT_MASKS_STATE_INTERSECTION;
    if(!strcmp(op, "difference")) return DT_MASKS_STATE_DIFFERENCE;
    if(!strcmp(op, "sum")) return DT_MASKS_STATE_SUM;
    if(!strcmp(op, "exclusion")) return DT_MASKS_STATE_EXCLUSION;
    if(!strcmp(op, "multiply")) return DT_MASKS_STATE_MULTIPLY;
  }
  // "automatic" / unset → historic default
  const dt_masks_state_t st =
    (form && form->type == DT_MASKS_BRUSH) ? DT_MASKS_STATE_SUM : DT_MASKS_STATE_UNION;
  dt_print(DT_DEBUG_MASKS, "[masks] default operator for new form (pref='%s') -> 0x%x",
           (op && *op) ? op : "automatic", st);
  return st;
}

dt_masks_point_group_t *dt_masks_group_add_form(dt_masks_form_t *grp,
                                                const dt_masks_form_t *form)
{
  // add a form to group and check for self inclusion

  if(!(grp->type & DT_MASKS_GROUP)) return NULL;
  // either the form to add is not a group, so no risk
  // or we go through all points of form to see if we find a ref to grp->formid
  if(!(form->type & DT_MASKS_GROUP) || _find_in_group(form, grp->formid, 0) == 0)
  {
    dt_masks_point_group_t *grpt = calloc(1, sizeof(dt_masks_point_group_t));
    grpt->formid = form->formid;
    grpt->parentid = grp->formid;
    grpt->state = DT_MASKS_STATE_SHOW | DT_MASKS_STATE_USE;
    if(grp->points) grpt->state |= dt_masks_get_default_operator(form);
    grpt->opacity = _new_shape_default_opacity(form->type);
    grpt->group_opacity = 1.0f;
    grp->points = g_list_append(grp->points, grpt);
    return grpt;
  }

  dt_control_log(_("masks can not contain themselves"));
  return NULL;
}

// Is `formid` one of the ids in `formids` (a GList of GINT_TO_POINTER ids)?
static gboolean _id_in_list(GList *formids, const dt_mask_id_t formid)
{
  for(GList *l = formids; l; l = g_list_next(l))
    if(GPOINTER_TO_INT(l->data) == formid) return TRUE;
  return FALSE;
}

void dt_masks_group_set_state(dt_masks_form_t *grp,
                              GList *formids,
                              const dt_masks_state_t bits,
                              const gboolean set)
{
  if(!grp || !(grp->type & DT_MASKS_GROUP)) return;
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = l->data;
    if(!_id_in_list(formids, pt->formid)) continue;
    if(set)
      pt->state |= bits;
    else
      pt->state &= ~bits;
  }
}

void dt_masks_group_isolate_state(dt_masks_form_t *grp,
                                  GList *formids,
                                  const dt_masks_state_t bits)
{
  if(!grp || !(grp->type & DT_MASKS_GROUP)) return;
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = l->data;
    // formids == NULL is the "solo off" case: nothing is singled out any more,
    // so the bits come off everywhere. Note this is NOT the same as treating
    // every point as a non-member, which would set the bits everywhere and
    // hide the entire group instead.
    const gboolean keep = !formids || _id_in_list(formids, pt->formid);
    if(keep)
      pt->state &= ~bits;
    else
      pt->state |= bits;
  }
}

static void _ungroup(dt_masks_form_t *dest_grp,
                     dt_masks_form_t *grp,
                     const int depth)
{
  if(!grp || !dest_grp || depth > DT_MASKS_NESTING_MAX) return;
  // a committed, multi-path AI-mask bundle (DT_MASKS_OBJECT, see
  // _register_vectorized_forms in masks/object.c) is recursed into exactly
  // like a real nested DT_MASKS_GROUP below: its own children get flattened
  // into dest_grp individually, so the canvas's on-screen node-editing (which
  // always operates on this flattened scratch copy, see dt_masks_set_edit_mode)
  // can drag each sub-path's own points directly, delegating to that child's
  // own event handlers (path.c) with zero changes needed there -- the actual
  // mask math (module->blend_params->mask_id's real, unflattened group) is
  // untouched by this, so the panel's own coordinated feather/size/rotation
  // controls for the bundle keep working exactly as before.
  if(!(grp->type & (DT_MASKS_GROUP | DT_MASKS_OBJECT))
     || !(dest_grp->type & DT_MASKS_GROUP))
    return;

  for(GList *forms = grp->points; forms; forms = g_list_next(forms))
  {
    const dt_masks_point_group_t *grpt = forms->data;
    dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, grpt->formid);
    if(form)
    {
      if(form->type & (DT_MASKS_GROUP | DT_MASKS_OBJECT))
      {
        _ungroup(dest_grp, form, depth + 1);
      }
      else
      {
        dt_masks_point_group_t *fpt = calloc(1, sizeof(dt_masks_point_group_t));
        fpt->formid = grpt->formid;
        fpt->parentid = grpt->parentid;
        fpt->state = grpt->state;
        fpt->opacity = grpt->opacity;
        fpt->group_opacity = grpt->group_opacity;
        dest_grp->points = g_list_append(dest_grp->points, fpt);
      }
    }
  }
}

void dt_masks_group_ungroup(dt_masks_form_t *dest_grp,
                            dt_masks_form_t *grp)
{
  _ungroup(dest_grp, grp, 0);
}

dt_hash_t dt_masks_group_hash(dt_hash_t hash, dt_masks_form_t *form)
{
  return dt_masks_group_hash_ext(hash, form,
                                 darktable.develop ? darktable.develop->forms : NULL);
}

static dt_hash_t _group_hash(dt_hash_t hash,
                             dt_masks_form_t *form,
                             GList *forms_list,
                             const int depth)
{
  if(!form || depth > DT_MASKS_NESTING_MAX) return hash;
  // basic infos
  hash = dt_hash(hash, &form->type, sizeof(dt_masks_type_t));
  hash = dt_hash(hash, &form->formid, sizeof(dt_mask_id_t));
  hash = dt_hash(hash, &form->version, sizeof(int));
  hash = dt_hash(hash, &form->source, sizeof(float) * 3);

  for(const GList *forms = form->points; forms; forms = g_list_next(forms))
  {
    // a committed AI-mask bundle (DT_MASKS_OBJECT, see _register_vectorized_forms
    // in masks/object.c) has a ->points list structurally identical to a
    // group's (dt_masks_point_group_t referencing real child forms) -- it
    // must recurse the same way, or a child's own geometry (e.g. a canvas
    // node drag, see masks/masks.c's dt_masks_group_ungroup flattening) never
    // reaches this hash: the else branch below would hash only the bundle's
    // own point-group entries (formid/state/opacity), which a node drag never
    // touches, leaving blend.c's drawn-mask cache keyed on a stale hash that
    // never changes -- the edit becomes invisible until something else (e.g.
    // undo, which bypasses that cache) forces a full reprocess.
    if(form->type & (DT_MASKS_GROUP | DT_MASKS_OBJECT))
    {
      const dt_masks_point_group_t *grpt = forms->data;
      // a marker refers to no form, but holds the settings its group renders
      // with. Its name is left out: renaming a group changes no pixel
      if(dt_masks_point_is_marker(grpt))
      {
        hash = dt_hash(hash, &grpt->state, sizeof(int));
        hash = dt_hash(hash, &grpt->group_opacity, sizeof(float));
        hash = dt_hash(hash, &grpt->refinement, sizeof(dt_masks_refinement_t));
        continue;
      }
      dt_masks_form_t *f = dt_masks_get_from_id_ext(forms_list, grpt->formid);
      if(f)
      {
        // state & opacity
        hash = dt_hash(hash, &grpt->state, sizeof(int));
        hash = dt_hash(hash, &grpt->opacity, sizeof(float));
        // group-level opacity (masks v9) multiplies the group's finished
        // sub-mask in the group renderer (see _group_get_mask_roi_flexi in
        // group.c), so it is a rendering input exactly as the per-shape
        // opacity above is. Left out, dragging the group opacity slider
        // produced no visible change until some unrelated edit forced a
        // reprocess. Zero-filled on pre-v9 blobs, so neutral for old edits.
        hash = dt_hash(hash, &grpt->group_opacity, sizeof(float));
        // per-shape/per-group refinement (masks v7) is a rendering input consumed
        // by the group renderer, so it must feed the pixelpipe cache hash too;
        // zero-filled for legacy blobs, so this is neutral for old edits.
        hash = dt_hash(hash, &grpt->refinement, sizeof(dt_masks_refinement_t));
        hash = _group_hash(hash, f, forms_list, depth + 1);
      }
    }
    else if(form->functions)
    {
      hash = dt_hash(hash, forms->data, form->functions->point_struct_size);
    }
  }
  return hash;
}

dt_hash_t dt_masks_group_hash_ext(dt_hash_t hash,
                                  dt_masks_form_t *form,
                                  GList *forms_list)
{
  return _group_hash(hash, form, forms_list, 0);
}

// adds formid to used array
// if formid is a group or an AI object it adds all the forms that belong to it
static void _cleanup_unused_recurs(GList *forms,
                                   const dt_mask_id_t formid,
                                   int *used,
                                   const int nb,
                                   const int depth)
{
  if(depth > DT_MASKS_NESTING_MAX) return;
  // first, we search for the formid in used table
  for(int i = 0; i < nb; i++)
  {
    if(used[i] == 0)
    {
      // we store the formid
      used[i] = formid;
      break;
    }
    if(used[i] == formid) break;
  }

  // if the form is a group or an AI object, we iterate through the sub-forms:
  // an object's paths are separate forms that only it refers to
  const dt_masks_form_t *form = dt_masks_get_from_id_ext(forms, formid);
  if(form && (form->type & (DT_MASKS_GROUP | DT_MASKS_OBJECT)))
  {
    for(GList *grpts = form->points; grpts; grpts = g_list_next(grpts))
    {
      const dt_masks_point_group_t *grpt = grpts->data;
      // a marker's id names no form, and `used` has room for one id per form
      if(dt_masks_point_is_marker(grpt)) continue;
      _cleanup_unused_recurs(forms, grpt->formid, used, nb, depth + 1);
    }
  }
}

static gboolean _object_is_empty(GList *forms, const dt_masks_form_t *obj)
{
  for(const GList *p = obj->points; p; p = g_list_next(p))
    if(dt_masks_get_from_id_ext(forms, ((dt_masks_point_group_t *)p->data)->formid))
      return FALSE;
  return TRUE;
}

int dt_masks_prune_empty_objects(GList **forms)
{
  int removed = 0;
  GList *l = *forms;
  while(l)
  {
    dt_masks_form_t *obj = l->data;
    l = g_list_next(l);
    if(!(obj->type & DT_MASKS_OBJECT) || !_object_is_empty(*forms, obj)) continue;

    for(GList *g = *forms; g; g = g_list_next(g))
    {
      dt_masks_form_t *grp = g->data;
      if(!(grp->type & DT_MASKS_GROUP)) continue;
      GList *p = grp->points;
      while(p)
      {
        GList *next = g_list_next(p);
        dt_masks_point_group_t *pt = p->data;
        if(pt->formid == obj->formid)
        {
          grp->points = g_list_delete_link(grp->points, p);
          free(pt);
        }
        p = next;
      }
    }
    *forms = g_list_remove(*forms, obj);
    // freed with the rest of the dropped forms, as by the cleanup below
    darktable.develop->allforms = g_list_append(darktable.develop->allforms, obj);
    removed++;
  }
  return removed;
}

// removes from _forms, the snapshot history item `start` carries, every form
// no module mask in effect at a position in [start, end) reaches: the items
// from start on, and the earlier ones no later item of the same module (up
// to start) has replaced. A replaced item's mask is no longer applied from
// start on, and the positions before start read an earlier snapshot, so
// keeping its forms here only resurrects shapes nothing uses.
static int _masks_cleanup_unused(GList **_forms,
                                 GList *history_list,
                                 const int start,
                                 const int end)
{
  int masks_removed = dt_masks_prune_empty_objects(_forms) ? 1 : 0;
  GList *forms = *_forms;

  // we create a table to store the ids of used forms
  const guint nbf = g_list_length(forms);
  int *used = calloc(nbf, sizeof(int));

  int num = 0;
  for(GList *history = history_list;
      history && num < end && used;
      history = g_list_next(history), num++)
  {
    const dt_dev_history_item_t *hist = history->data;
    const dt_develop_blend_params_t *blend_params = hist->blend_params;
    if(!blend_params || !dt_is_valid_maskid(blend_params->mask_id)) continue;

    gboolean replaced = FALSE;
    if(num < start && hist->module)
    {
      int later = num + 1;
      for(const GList *l = g_list_next(history);
          l && later <= start;
          l = g_list_next(l), later++)
      {
        if(((const dt_dev_history_item_t *)l->data)->module == hist->module)
        {
          replaced = TRUE;
          break;
        }
      }
    }
    if(!replaced)
      _cleanup_unused_recurs(forms, blend_params->mask_id, used, nbf, 0);
  }

  // and we delete all unused forms
  GList *shapes = forms;
  while(shapes && used)
  {
    dt_masks_form_t *f = shapes->data;
    gboolean found = FALSE;
    for(int i = 0; i < nbf; i++)
    {
      if(used[i] == f->formid)
      {
        found = TRUE;
        break;
      }
      if(used[i] == 0) break;
    }

    shapes = g_list_next(shapes); // need to get 'next' now, because
                                  // we may be removing the current
                                  // node

    if(found == FALSE)
    {
      forms = g_list_remove(forms, f);
      // and add it to allforms for cleanup
      darktable.develop->allforms = g_list_append(darktable.develop->allforms, f);
      masks_removed = 1;
    }
  }

  free(used);

  *_forms = forms;

  return masks_removed;
}

// removes all unused forms from every forms snapshot in history. Each
// snapshot only has to keep what the positions reading it use, from its own
// item up to the next snapshot, so going back in history stays exact. Every
// item carrying forms is a snapshot, not just mask_manager ones: a module's
// own item carries one whenever its masks were recorded with it (see
// dt_dev_add_masks_history_item_ext), and history replay reads those too.
// A snapshot cleaned down to nothing becomes NULL, which replay reads as "no
// snapshot here", so an older one takes over: shapes only that older one
// keeps can come back, and only compressing history removes them.
void dt_masks_cleanup_unused_from_list(GList *history_list)
{
  int num = g_list_length(history_list) - 1;
  int end = num + 1;

  for(const GList *history = g_list_last(history_list);
      history;
      history = g_list_previous(history), num--)
  {
    dt_dev_history_item_t *hist = history->data;
    if(hist->forms)
    {
      _masks_cleanup_unused(&hist->forms, history_list, num, end);
      end = num;
    }
  }
}

void dt_masks_cleanup_unused(dt_develop_t *dev)
{
  dt_masks_change_form_gui(NULL);

  // we remove the forms from history
  dt_masks_cleanup_unused_from_list(dev->history);

  // and we save all that
  GList *forms = NULL;
  dt_iop_module_t *module = NULL;
  int num = 0;
  for(const GList *history = dev->history;
      history && num < dev->history_end;
      history = g_list_next(history))
  {
    const dt_dev_history_item_t *hist = history->data;

    if(hist->forms) forms = hist->forms;
    if(hist->module
       && strcmp(hist->op_name, "mask_manager") != 0)
      module = hist->module;

    num++;
  }

  dt_masks_replace_current_forms(dev, forms);

  if(module)
    dt_dev_add_history_item(dev, module, module->enabled);
  else
    dt_dev_add_masks_history_item(dev, NULL, TRUE);
}

gboolean dt_masks_point_in_form_exact(const float x,
                                      const float y,
                                      const float *points,
                                      const int points_start,
                                      const int points_count)
{
  // we use ray casting algorithm to avoid most problems with
  // horizontal segments, y should be rounded as int so that there's
  // very little chance than y==points...

  if(points_count > 2 + points_start)
  {
    const int start = (points[points_start * 2] == DT_INVALID_COORDINATE
                       && points[points_start * 2 + 1] != DT_INVALID_COORDINATE)
         ? points[points_start * 2 + 1]
         : points_start;

    int nb = 0;

    for(int i = start, next = start + 1; i < points_count;)
    {
      const float y1 = points[i * 2 + 1];
      const float y2 = points[next * 2 + 1];
      //if we need to skip points (in case of deleted point, because
      //of self-intersection)
      if(points[next * 2] == DT_INVALID_COORDINATE)
      {
        next = (y2 == DT_INVALID_COORDINATE) ? start : (int)y2;
        continue;
      }
      if(((y <= y2 && y > y1)
          || (y >= y2 && y < y1))
         && (points[i * 2] > x))
        nb++;

      if(next == start) break;
      i = next++;
      if(next >= points_count) next = start;
    }
    return (nb & 1) != 0;
  }
  return FALSE;
}

gboolean dt_masks_point_in_form_near(const float x,
                                     const float y,
                                     const float *points,
                                     const int points_start,
                                     const int points_count,
                                     const float distance,
                                     int *near)
{
  // we use ray casting algorithm to avoid most problems with
  // horizontal segments.

  const float distance2 = sqf(distance);

  *near = -1;

  if(points_count > 2 + points_start)
  {
    const int start = (points[points_start * 2] == DT_INVALID_COORDINATE
                       && points[points_start * 2 + 1] != DT_INVALID_COORDINATE)
      ? points[points_start * 2 + 1]
      : points_start;

    int nb = 0;
    for(int i = start, next = start + 1; i < points_count;)
    {
      const float x1 = points[i * 2];
      const float y1 = points[i * 2 + 1];
      const float y2 = points[next * 2 + 1];
      const float dd = sqf(x1 - x) + sqf(y1 - y);

      if(dd < distance2)
        *near = i * 2;

      //if we need to jump to skip points (in case of deleted point,
      //because of self-intersection)
      if(points[next * 2] == DT_INVALID_COORDINATE)
      {
        next = (y2 == DT_INVALID_COORDINATE) ? start : (int)y2;
        continue;
      }
      if((y <= y2 && y > y1)
         || (y >= y2 && y < y1))
      {
        if(x1 > x)
          nb++;
      }

      if(next == start) break;
      i = next++;
      if(next >= points_count)
        next = start;
    }
    return (nb & 1) != 0;
  }
  return FALSE;
}

float dt_masks_drag_factor(dt_masks_form_gui_t *gui,
                           const int index,
                           const int k,
                           const gboolean border)
{
  // we need the reference points
  dt_masks_form_gui_points_t *gpt = g_list_nth_data(gui->points, index);

  if(!gpt) return 0.0f;

  const float *boundary = border ? gpt->border : gpt->points;
  const float xref = gpt->points[0];
  const float yref = gpt->points[1];
  const float rx = boundary[k * 2] - xref;
  const float ry = boundary[k * 2 + 1] - yref;
  const float deltax = gui->posx + gui->dx - xref;
  const float deltay = gui->posy + gui->dy - yref;

  // we remap dx, dy to the right values, as it will be used in next
  // movements
  gui->dx = xref - gui->posx;
  gui->dy = yref - gui->posy;

  const float r = dt_fast_hypotf(rx, ry);
  const float d = (rx * deltax + ry * deltay) / r;
  const float s = fmaxf(r > 0.0f ? (r + d) / r : 0.0f, 0.0f);

  return s;
}

float dt_masks_change_size(const gboolean up,
                           const float value,
                           const float min,
                           const float max)
{
  const float v =
    up
    ? value / 0.97f
    : value * 0.97f;

  return CLAMP(v, min, max);
}

float dt_masks_change_rotation(const gboolean up,
                               const float value,
                               const gboolean is_degree)
{
  const float step = 40.f;
  const float incr = is_degree ? 360.f / step : DT_2PI_F / step;
  const float max  = is_degree ? 360.0        : M_PI_F;
  const float v =
    up
    ? value + incr
    : value - incr;

  if(is_degree)
    return fmodf(v + max, max);
  else
  {
    return v > max ? v - (2.0f * max) : v;
  }
}

// allow to select a shape inside an iop
void dt_masks_select_form(dt_iop_module_t *module,
                          const dt_masks_form_t *sel)
{
  gboolean selection_changed = FALSE;

  if(sel)
  {
    if(sel->formid != darktable.develop->mask_form_selected_id)
    {
      darktable.develop->mask_form_selected_id = sel->formid;
      selection_changed = TRUE;
    }
    // clicking a shape on the canvas always selects it -- including
    // re-clicking the already-selected shape -- and never deselects; a
    // shape can only be deselected by clicking empty canvas or another
    // shape's own row/title in the panel.
  }
  else
  {
    if(darktable.develop->mask_form_selected_id != 0)
    {
      darktable.develop->mask_form_selected_id = 0;
      selection_changed = TRUE;
    }
  }
  if(selection_changed)
  {
    if(!module && darktable.develop->mask_form_selected_id == 0)
      module = dt_dev_gui_module();
    if(module)
    {
      if(module->masks_selection_changed)
        module->masks_selection_changed(module, darktable.develop->mask_form_selected_id);
      // mirror the canvas selection into the flexi mask list (highlight its row)
      dt_iop_gui_masks_select_form(module, darktable.develop->mask_form_selected_id);
    }
  }
}

// draw a cross where the source position of a clone mask will be created
void dt_masks_draw_clone_source_pos(cairo_t *cr,
                                    const float zoom_scale,
                                    const float x,
                                    const float y)
{
  const float dx = 3.5f / zoom_scale;
  const float dy = 3.5f / zoom_scale;

  double dashed[] = { 4.0, 4.0 };
  dashed[0] /= zoom_scale;
  dashed[1] /= zoom_scale;

  cairo_set_dash(cr, dashed, 0, 0);
  const double lwidth = (dt_iop_canvas_not_sensitive(darktable.develop) ? 0.5 : 1.0) / zoom_scale;
  cairo_set_line_width(cr, 3.0 * lwidth);
  cairo_set_source_rgba(cr, .3, .3, .3, .8);

  cairo_move_to(cr, x + dx, y);
  cairo_line_to(cr, x - dx, y);
  cairo_move_to(cr, x, y + dy);
  cairo_line_to(cr, x, y - dy);
  cairo_stroke_preserve(cr);

  cairo_set_line_width(cr, lwidth);
  cairo_set_source_rgba(cr, .8, .8, .8, .8);
  cairo_stroke(cr);
}

// sets if the initial source position for a clone mask will be
// absolute or relative, based on mouse position and key state
void dt_masks_set_source_pos_initial_state(dt_masks_form_gui_t *gui,
                                           const uint32_t state,
                                           const float pzx,
                                           const float pzy)
{
  if(dt_modifier_is(state, GDK_SHIFT_MASK | GDK_CONTROL_MASK))
    gui->source_pos_type = DT_MASKS_SOURCE_POS_ABSOLUTE;
  else if(dt_modifier_is(state, GDK_SHIFT_MASK))
    gui->source_pos_type = DT_MASKS_SOURCE_POS_RELATIVE_TEMP;
  else
    dt_print(DT_DEBUG_ALWAYS,
             "[dt_masks_set_source_pos_initial_state] unknown state for setting masks position type");

  // both source types record an absolute position, for the relative
  // type, the first time is used the position is recorded, the second
  // time a relative position is calculated based on that one
  float wd, ht;
  dt_masks_get_image_size(&wd, &ht, NULL, NULL);
  gui->posx_source = pzx * wd;
  gui->posy_source = pzy * ht;
}

// set the initial source position value for a clone mask
void dt_masks_set_source_pos_initial_value(dt_masks_form_gui_t *gui,
                                           const int mask_type,
                                           dt_masks_form_t *form,
                                           const float pzx,
                                           const float pzy)
{
  float wd, ht, iwidth, iheight;
  dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);

  // if this is the first time the relative pos is used
  if(gui->source_pos_type == DT_MASKS_SOURCE_POS_RELATIVE_TEMP)
  {
    // if it has not been defined by the user, set some default
    if(gui->posx_source == -1.0f && gui->posy_source == -1.0f)
    {
      if(form->functions && form->functions->initial_source_pos)
      {
        form->functions->initial_source_pos(iwidth, iheight, &gui->posx_source, &gui->posy_source);
      }
      else
        dt_print(DT_DEBUG_ALWAYS, "[dt_masks_set_source_pos_initial_value]"
                 " unsupported masks type when calculating source position initial value\n");

      float pts[2] = { pzx * wd + gui->posx_source, pzy * ht + gui->posy_source };
      dt_dev_distort_backtransform(darktable.develop, pts, 1);

      form->source[0] = pts[0] / iwidth;
      form->source[1] = pts[1] / iheight;
    }
    else
    {
      // if a position was defined by the user, use the absolute value
      // the first time
      float pts[2] = { gui->posx_source, gui->posy_source };
      dt_dev_distort_backtransform(darktable.develop, pts, 1);

      form->source[0] = pts[0] / iwidth;
      form->source[1] = pts[1] / iheight;

      gui->posx_source = gui->posx_source - pzx * wd;
      gui->posy_source = gui->posy_source - pzy * ht;
    }

    gui->source_pos_type = DT_MASKS_SOURCE_POS_RELATIVE;
  }
  else if(gui->source_pos_type == DT_MASKS_SOURCE_POS_RELATIVE)
  {
    // original pos was already defined and relative value calculated,
    // just use it
    float pts[2] = { pzx * wd + gui->posx_source,
                     pzy * ht + gui->posy_source };
    dt_dev_distort_backtransform(darktable.develop, pts, 1);

    form->source[0] = pts[0] / iwidth;
    form->source[1] = pts[1] / iheight;
  }
  else if(gui->source_pos_type == DT_MASKS_SOURCE_POS_ABSOLUTE)
  {
    // an absolute position was defined by the user
    float pts_src[2] = { gui->posx_source, gui->posy_source };
    dt_dev_distort_backtransform(darktable.develop, pts_src, 1);

    form->source[0] = pts_src[0] / iwidth;
    form->source[1] = pts_src[1] / iheight;
  }
  else
    dt_print(DT_DEBUG_ALWAYS, "[dt_masks_set_source_pos_initial_value]"
             " unknown source position type\n");
}

// calculates the source position value for preview drawing, on cairo coordinates
void dt_masks_calculate_source_pos_value(const dt_masks_form_gui_t *gui,
                                         const int mask_type,
                                         const float initial_xpos,
                                         const float initial_ypos,
                                         const float xpos,
                                         const float ypos,
                                         float *px,
                                         float *py,
                                         const int adding)
{
  float wd, ht, iwidth, iheight;
  dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);

  float x = 0.0f, y = 0.0f;

  if(gui->source_pos_type == DT_MASKS_SOURCE_POS_RELATIVE)
  {
    x = xpos + gui->posx_source;
    y = ypos + gui->posy_source;
  }
  else if(gui->source_pos_type == DT_MASKS_SOURCE_POS_RELATIVE_TEMP)
  {
    if(gui->posx_source == -1.0f && gui->posy_source == -1.0f)
    {
#if 0 //TODO: replace individual cases with this generic one (will
      //require passing 'form' through multiple layers...)
      if(form->functions && form->functions->initial_source_pos)
      {
        form->functions->initial_source_pos(iwidth, iheight, &x, &y);
        x += xpos;
        y += ypos;
      }
#else
      if(mask_type & DT_MASKS_CIRCLE)
      {
        dt_masks_functions_circle.initial_source_pos(iwidth, iheight, &x, &y);
        x += xpos;
        y += ypos;
      }
      else if(mask_type & DT_MASKS_ELLIPSE)
      {
        dt_masks_functions_ellipse.initial_source_pos(iwidth, iheight, &x, &y);
        x += xpos;
        y += ypos;
      }
      else if(mask_type & DT_MASKS_PATH)
      {
        dt_masks_functions_path.initial_source_pos(iwidth, iheight, &x, &y);
        x += xpos;
        y += ypos;
      }
      else if(mask_type & DT_MASKS_BRUSH)
      {
        dt_masks_functions_brush.initial_source_pos(iwidth, iheight, &x, &y);
        x += xpos;
        y += ypos;
      }
#endif
      else
        dt_print(DT_DEBUG_ALWAYS, "[dt_masks_calculate_source_pos_value]"
                 " unsupported masks type when calculating source position value\n");
    }
    else
    {
      x = gui->posx_source;
      y = gui->posy_source;
    }
  }
  else if(gui->source_pos_type == DT_MASKS_SOURCE_POS_ABSOLUTE)
  {
    // if the user is actually adding the mask follow the cursor
    if(adding)
    {
      x = xpos + gui->posx_source - initial_xpos;
      y = ypos + gui->posy_source - initial_ypos;
    }
    else
    {
      // if not added yet set the start position
      x = gui->posx_source;
      y = gui->posy_source;
    }
  }
  else
    dt_print(DT_DEBUG_ALWAYS,
             "[dt_masks_calculate_source_pos_value]"
             " unknown source position type for setting source position value\n");

  *px = x;
  *py = y;
}

void dt_masks_draw_anchor(cairo_t *cr,
                          const gboolean selected,
                          const float zoom_scale,
                          const float x,
                          const float y)
{
  const float anchor_size = DT_PIXEL_APPLY_DPI(selected ? 8.0f : 5.0f) / zoom_scale;

  cairo_set_dash(cr, NULL, 0, 0);
  dt_draw_set_color_overlay(cr, TRUE, 0.8);
  cairo_rectangle(cr,
                  x - (anchor_size * 0.5f),
                  y - (anchor_size * 0.5f),
                  anchor_size,
                  anchor_size);
  cairo_fill_preserve(cr);
  const double lwidth = (dt_iop_canvas_not_sensitive(darktable.develop) ? 0.5 : 1.0) / zoom_scale;
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(selected ? 2.0 : 1.0) * lwidth);
  dt_draw_set_color_overlay(cr, FALSE, 0.8);
  cairo_stroke(cr);
}

void dt_masks_draw_ctrl(cairo_t *cr,
                        const float x,
                        const float y,
                        const float zoom_scale,
                        const gboolean selected)
{
  const float ctrl_size = DT_PIXEL_APPLY_DPI(selected ? 3.0f : 1.5f) / zoom_scale;

  cairo_arc(cr, x, y, ctrl_size, 0, 2.0 * M_PI);

  dt_draw_set_color_overlay(cr, TRUE, 0.8);
  cairo_fill_preserve(cr);

  const double lwidth = (dt_iop_canvas_not_sensitive(darktable.develop) ? 0.5 : 1.0) / zoom_scale;
  cairo_set_line_width(cr, lwidth);
  dt_draw_set_color_overlay(cr, FALSE, 0.8);
  cairo_stroke(cr);
}

void dt_masks_draw_arrow(cairo_t *cr,
                         const float from_x,
                         const float from_y,
                         const float to_x,
                         const float to_y,
                         const float zoom_scale,
                         const gboolean touch_dest)
{
  const float dx = from_x - to_x;
  const float dy = from_y - to_y;
  const float arrow_size = DT_PIXEL_APPLY_DPI(24.0f);

  const float arrow_scale = arrow_size / sqrtf(3.f * zoom_scale);

  const gboolean draw_arrow = TRUE;

  float cangle = atanf(dx / dy);

  if(dy > 0)
    cangle = M_PI_2 - cangle;
  else
    cangle = -M_PI_2 - cangle;

  // move a bit away from the path
  const float x = to_x + (touch_dest
                          ? 0.f
                          : 5.f * cosf(cangle) / zoom_scale);

  const float y = to_y + (touch_dest
                          ? 0.f
                          : 5.f * sinf(cangle) / zoom_scale);

  cairo_move_to(cr, from_x, from_y); // start
  cairo_line_to(cr, x, y);           // end + a bit of space

  // no arrow when size too small
  if(draw_arrow)
  {
    // then draw to line for the arrow itself

    cairo_move_to(cr,
                  x + arrow_scale * cosf(cangle + (0.4)),
                  y + arrow_scale * sinf(cangle + (0.4)));

    cairo_line_to(cr, x, y);

    cairo_line_to(cr,
                  x + arrow_scale * cosf(cangle - (0.4)),
                  y + arrow_scale * sinf(cangle - (0.4)));
  }
}

void dt_masks_stroke_arrow(cairo_t *cr,
                           const dt_masks_form_gui_t *gui,
                           const int group,
                           const float zoom_scale)
{
  const double dashed[] = { 0, 0 };
  cairo_set_dash(cr, dashed, 0, 0);

  const double lwidth = (dt_iop_canvas_not_sensitive(darktable.develop) ? 0.5 : 1.0) / zoom_scale;
  if((gui->group_selected == group) && (gui->form_selected || gui->form_dragging))
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(2.5) * lwidth);
  else
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.5) * lwidth);

  dt_draw_set_color_overlay(cr, FALSE, 0.8);
  cairo_stroke_preserve(cr);

  if((gui->group_selected == group) && (gui->form_selected || gui->form_dragging))
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0) * lwidth);
  else
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(0.5) * lwidth);

  dt_draw_set_color_overlay(cr, TRUE, 0.8);
  cairo_stroke(cr);
}

void dt_masks_closest_point(const int count,
                            const int nb_ctrl,
                            const float *points,
                            const float px,
                            const float py,
                            float *x,
                            float *y)
{
  float dist = FLT_MAX;
  *x = px;
  *y = py;

  for(int i = nb_ctrl; i < count; i++)
  {
    const float dx = points[i * 2] - px;
    const float dy = points[i * 2 + 1] - py;

    const float d = sqf(dx*dx + dy*dy);
    if(d < dist)
    {
      *x = points[i * 2];
      *y = points[i * 2 + 1];
      dist = d;
    }
  }
}

void dt_masks_rotate_ctrl_points(dt_develop_t *dev,
                                 const float *const gpt_points,
                                 const int points_count,
                                 const int nb,
                                 const float cx,
                                 const float cy,
                                 const float cos_a,
                                 const float sin_a,
                                 const float iwidth,
                                 const float iheight,
                                 float *const out)
{
  // the control points are the first nb*3 (x,y) pairs of the display buffer
  const int nctrl = nb * 3;
  if(nctrl < 1 || points_count < nctrl)
    return;

  // rotate every control point around the screen pivot into a scratch buffer,
  // then back-transform the whole batch in one pipe traversal (cheaper and more
  // accurate than inverting the pipe per point)
  float *const scr = dt_alloc_align_float((size_t)nctrl * 2);
  if(!scr)
    return;

  for(int i = 0; i < nctrl; i++)
  {
    const float rx = gpt_points[i * 2] - cx;
    const float ry = gpt_points[i * 2 + 1] - cy;
    scr[i * 2] = cx + rx * cos_a - ry * sin_a;
    scr[i * 2 + 1] = cy + rx * sin_a + ry * cos_a;
  }

  dt_dev_distort_backtransform(dev, scr, nctrl);

  for(int i = 0; i < nctrl; i++)
  {
    out[i * 2] = scr[i * 2] / iwidth;
    out[i * 2 + 1] = scr[i * 2 + 1] / iheight;
  }

  dt_free_align(scr);
}

void dt_masks_line_stroke(cairo_t *cr,
                          const gboolean border,
                          const gboolean source,
                          const gboolean selected,
                          const float zoom_scale)
{
  const double size_border     = DT_PIXEL_APPLY_DPI(1.0);
  const double size_source     = DT_PIXEL_APPLY_DPI(1.5);
  const double size_mask       = DT_PIXEL_APPLY_DPI(1.7);
  const double factor_selected = DT_PIXEL_APPLY_DPI(1.5);

  double dashed[] = { DT_PIXEL_APPLY_DPI(4.0), DT_PIXEL_APPLY_DPI(4.0) };
  dashed[0] /= zoom_scale;
  dashed[1] /= zoom_scale;
  const int len = sizeof(dashed) / sizeof(dashed[0]);

  double dashed_restricted[] = { DT_PIXEL_APPLY_DPI(8.0), DT_PIXEL_APPLY_DPI(12.0) };
  dashed_restricted[0] /= zoom_scale;
  dashed_restricted[1] /= zoom_scale;

  const gboolean restricted = _masks_is_restricted_mode();

  // first the background draw, darker
  if(restricted && !border)
    dt_draw_set_color_overlay(cr, FALSE, 0.1);
  else
    dt_draw_set_color_overlay(cr, FALSE, selected ? 0.8 : 0.5);

  cairo_set_dash(cr, dashed, border ? len : 0, 0);

  const double lwidth = (dt_iop_canvas_not_sensitive(darktable.develop) ? 0.5 : 1.0) / zoom_scale;
  const double line_width =
    ((border ? size_border : (source ? size_source : size_mask))
     * (selected ? factor_selected : 1.0)) * lwidth;

  cairo_set_line_width(cr, line_width);

  cairo_stroke_preserve(cr);

  // second the foreground draw, lighter (same size as darker if selected)
  cairo_set_line_width(cr, (line_width / (selected && !border ? 1.0 : 2.0)));

  if(restricted && !border)
  {
    cairo_set_dash(cr, dashed_restricted, len, 4);
    dt_draw_set_color_overlay(cr, TRUE, 1.0);
  }
  else if(!source)
  {
    dt_draw_set_color_overlay(cr, TRUE, selected ? 0.9 : 0.6);
    cairo_set_dash(cr, dashed, border ? len : 0, 4);
  }

  cairo_stroke(cr);
}

#include "detail.c"

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
