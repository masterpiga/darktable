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

#include "develop/masks/postedit_internal.h"

#include "common/darktable.h"
#include "develop/blend_gui_internal.h"
#include "develop/masks.h"

#include <float.h>
#include <string.h>

// ---------------------------------------------------------------------------
// the pokes
// ---------------------------------------------------------------------------


static const char *const _poke_name[POKE_N] =
{
  "op:union",     "op:intersection", "op:difference", "op:sum",
  "op:exclusion", "op:multiply",     "op:screen",
  "within:union", "within:screen",   "within:intersect", "within:multiply",
  "group:bypass", "group:invert",    "group:opacity",    "group:refine",
  "elem:disable", "elem:hidden",     "elem:inverse",     "elem:opacity",
  "elem:refine",  "elem:break",
};

const char *_poke_label(const poke_t k)
{
  return (k < POKE_N) ? _poke_name[k] : "?";
}

// A refinement that visibly changes any mask it is applied to, whatever the
// shapes are: a blur wide enough to move a feathered edge at the replay's
// 512px scale, plus a contrast lift so a mask that is already smooth still
// responds. The exact values do not matter -- both sides get the same one --
// only that the control is not silently inert.
static const dt_masks_refinement_t _refine_probe =
{
  .enabled = DT_MASKS_REFINE_OFF, // set per poke
  .details = 0.0f,
  .feathering_radius = 0.0f,
  .feathering_guide = DEVELOP_MASK_GUIDE_OUT_BEFORE_BLUR,
  .blur_radius = 8.0f,
  .contrast = 0.3f,
  .brightness = 0.0f,
};

/* Apply one poke to the member index range [first, last] of `points`.

   A run-level poke is broadcast across the whole range, which starts at the
   group's marker, the record the fold reads. An element-level poke is passed
   first == last. A group break is no poke: it inserts a marker
   (_break_before). */
void _apply_poke(GList *points, const poke_t k,
                 const int first, const int last)
{
  int i = 0;
  for(GList *l = points; l; l = g_list_next(l), i++)
  {
    if(i < first || i > last) continue;
    dt_masks_point_group_t *pt = l->data;

    switch(k)
    {
      // the combining operator replaces whatever one is there, leaving the
      // bypass/invert modifiers alone -- the panel's operator menu does the
      // same, which is the whole reason DT_MASKS_STATE_OP_COMBINE exists
      // apart from DT_MASKS_STATE_OP
      case POKE_OP_UNION:
      case POKE_OP_INTERSECTION:
      case POKE_OP_DIFFERENCE:
      case POKE_OP_SUM:
      case POKE_OP_EXCLUSION:
      case POKE_OP_MULTIPLY:
      case POKE_OP_SCREEN:
      {
        static const dt_masks_state_t ops[] =
          { DT_MASKS_STATE_UNION,     DT_MASKS_STATE_INTERSECTION,
            DT_MASKS_STATE_DIFFERENCE, DT_MASKS_STATE_SUM,
            DT_MASKS_STATE_EXCLUSION,  DT_MASKS_STATE_MULTIPLY,
            DT_MASKS_STATE_OP_SCREEN };
        pt->state = (pt->state & ~(int)DT_MASKS_STATE_OP_COMBINE)
                    | (int)ops[k - POKE_OP_UNION];
        break;
      }

      case POKE_WITHIN_UNION:
      case POKE_WITHIN_SCREEN:
      case POKE_WITHIN_ISECT:
      case POKE_WITHIN_MULTIPLY:
      {
        static const dt_masks_state_t within[] =
          { DT_MASKS_STATE_NONE, DT_MASKS_STATE_SCREEN,
            DT_MASKS_STATE_ISECT, DT_MASKS_STATE_WITHIN_MULTIPLY };
        pt->state = (pt->state & ~(int)DT_MASKS_STATE_WITHIN)
                    | (int)within[k - POKE_WITHIN_UNION];
        break;
      }

      case POKE_GROUP_BYPASS:  pt->state |= DT_MASKS_STATE_OP_DISABLE; break;
      case POKE_GROUP_INVERT:  pt->state |= DT_MASKS_STATE_OP_INVERT;  break;
      case POKE_GROUP_OPACITY: pt->group_opacity = 0.5f;               break;
      case POKE_GROUP_REFINE:
        pt->refinement = _refine_probe;
        pt->refinement.enabled = DT_MASKS_REFINE_GROUP;
        break;

      case POKE_ELEM_DISABLE: pt->state |= DT_MASKS_STATE_DISABLE; break;
      case POKE_ELEM_HIDDEN:  pt->state |= DT_MASKS_STATE_HIDDEN;  break;
      case POKE_ELEM_INVERSE: pt->state |= DT_MASKS_STATE_INVERSE; break;
      case POKE_ELEM_OPACITY: pt->opacity = 0.5f;                  break;
      case POKE_ELEM_REFINE:
        pt->refinement = _refine_probe;
        pt->refinement.enabled = DT_MASKS_REFINE_ELEMENT;
        break;

      default: break;
    }
  }
}

// ---------------------------------------------------------------------------
// the shape controls
// ---------------------------------------------------------------------------

static const char *const _geom_name[GEOM_N] =
{
  "geom:translate", "geom:node",     "geom:size",       "geom:feather",
  "geom:hardness",  "geom:rotation", "geom:curvature",  "geom:compression",
};

const char *_geom_label(const geom_t g)
{
  return (g < GEOM_N) ? _geom_name[g] : "?";
}

/** the size of one point of `form`, or 0 if the form has no editable geometry */
static size_t _point_size(const dt_masks_form_t *form)
{
  if(!form || !form->functions) return 0;
  // a group's "points" are dt_masks_point_group_t, which the pokes own; a
  // raster or parametric element has no geometry of its own at all
  if(form->type & (DT_MASKS_GROUP | DT_MASKS_RASTER | DT_MASKS_PARAMETRIC))
    return 0;
  return (size_t)form->functions->point_struct_size;
}

GList *_geom_snapshot(const dt_masks_form_t *form)
{
  const size_t sz = _point_size(form);
  if(!sz) return NULL;

  GList *out = NULL;
  for(GList *l = form->points; l; l = g_list_next(l))
  {
    void *copy = malloc(sz);
    if(!copy) { g_list_free_full(out, free); return NULL; }
    memcpy(copy, l->data, sz);
    out = g_list_append(out, copy);
  }
  return out;
}

void _geom_restore(dt_masks_form_t *form, GList *snapshot)
{
  if(!snapshot) return;
  // wholesale rather than element-wise: the list is put back exactly as it was
  // even if a control had added or dropped a node, and the caller cannot then
  // leak the snapshot by forgetting to free it
  g_list_free_full(form->points, free);
  form->points = snapshot;
}

/** shift every coordinate of `form` by (dx, dy), in normalized image space */
static void _translate(dt_masks_form_t *form, const float dx, const float dy)
{
  for(GList *l = form->points; l; l = g_list_next(l))
  {
    if(form->type & DT_MASKS_CIRCLE)
    {
      dt_masks_point_circle_t *p = l->data;
      p->center[0] += dx; p->center[1] += dy;
    }
    else if(form->type & DT_MASKS_ELLIPSE)
    {
      dt_masks_point_ellipse_t *p = l->data;
      p->center[0] += dx; p->center[1] += dy;
    }
    else if(form->type & DT_MASKS_GRADIENT)
    {
      dt_masks_point_gradient_t *p = l->data;
      p->anchor[0] += dx; p->anchor[1] += dy;
    }
    else if(form->type & DT_MASKS_PATH)
    {
      // the control points move with the corner, or the shape shears instead of
      // translating -- which is what dragging a whole path does in the canvas
      dt_masks_point_path_t *p = l->data;
      p->corner[0] += dx; p->corner[1] += dy;
      p->ctrl1[0] += dx;  p->ctrl1[1] += dy;
      p->ctrl2[0] += dx;  p->ctrl2[1] += dy;
    }
    else if(form->type & DT_MASKS_BRUSH)
    {
      dt_masks_point_brush_t *p = l->data;
      p->corner[0] += dx; p->corner[1] += dy;
      p->ctrl1[0] += dx;  p->ctrl1[1] += dy;
      p->ctrl2[0] += dx;  p->ctrl2[1] += dy;
    }
  }
}

/** drag the first node of a path or brush, deforming the shape rather than
    moving it. Returns FALSE for a shape that has no nodes to drag. */
static gboolean _drag_node(dt_masks_form_t *form, const float dx, const float dy)
{
  if(!form->points) return FALSE;
  if(form->type & DT_MASKS_PATH)
  {
    dt_masks_point_path_t *p = form->points->data;
    p->corner[0] += dx; p->corner[1] += dy;
    p->ctrl1[0] += dx;  p->ctrl1[1] += dy;
    p->ctrl2[0] += dx;  p->ctrl2[1] += dy;
    return TRUE;
  }
  if(form->type & DT_MASKS_BRUSH)
  {
    dt_masks_point_brush_t *p = form->points->data;
    p->corner[0] += dx; p->corner[1] += dy;
    p->ctrl1[0] += dx;  p->ctrl1[1] += dy;
    p->ctrl2[0] += dx;  p->ctrl2[1] += dy;
    return TRUE;
  }
  return FALSE;
}

gboolean _apply_geom(dt_masks_form_t *form, const geom_t g)
{
  const size_t sz = _point_size(form);
  if(!sz || !form->points) return FALSE;

  GList *before = _geom_snapshot(form);
  if(!before) return FALSE;

  switch(g)
  {
    // 5% of the frame, which at the replay's 512px long edge is 25 pixels: far
    // enough to move a shape onto or off its neighbour (the point of doing
    // this at all), small enough that a shape near the border stays in frame
    case GEOM_TRANSLATE: _translate(form, 0.05f, 0.03f); break;
    case GEOM_NODE:      _drag_node(form, 0.04f, -0.03f); break;

    default:
    {
      if(!form->functions->modify_property) break;

      /* The panel's sliders, driven through the same entry point the panel
         uses. Two conventions live behind it and the shapes disagree about
         which they follow: size, feather, hardness and compression scale by
         new/old, while rotation and curvature add (new - old). Passing the
         wrong pair would still be safe -- the change is confirmed against the
         point data below, not assumed -- but it would make a live control look
         inert, so each gets the pair its implementations actually read. */
      static const struct { dt_masks_property_t prop; float old_val, new_val; }
      _prop[GEOM_N] =
      {
        [GEOM_SIZE]        = { DT_MASKS_PROPERTY_SIZE,        1.0f, 1.25f },
        [GEOM_FEATHER]     = { DT_MASKS_PROPERTY_FEATHER,     1.0f, 1.40f },
        [GEOM_HARDNESS]    = { DT_MASKS_PROPERTY_HARDNESS,    1.0f, 1.30f },
        [GEOM_COMPRESSION] = { DT_MASKS_PROPERTY_COMPRESSION, 1.0f, 1.30f },
        [GEOM_ROTATION]    = { DT_MASKS_PROPERTY_ROTATION,    0.0f, 15.0f },
        [GEOM_CURVATURE]   = { DT_MASKS_PROPERTY_CURVATURE,   0.0f, 0.25f },
      };

      float sum = 0.0f, mn = -FLT_MAX, mx = FLT_MAX;
      int count = 0;
      form->functions->modify_property(form, _prop[g].prop, _prop[g].old_val,
                                       _prop[g].new_val, &sum, &count, &mn, &mx);
      break;
    }
  }

  /* Did it actually do anything? Asked of the bytes rather than taken from
     modify_property's `count`, which reports that a shape implements the
     property and not that the value moved: every implementation clamps, so a
     shape already at its maximum size accepts the call and stays put. A check
     that counted those as coverage would be overstating what it swept. */
  gboolean changed = FALSE;
  GList *a = before, *b = form->points;
  for(; a && b; a = g_list_next(a), b = g_list_next(b))
    if(memcmp(a->data, b->data, sz)) { changed = TRUE; break; }
  if(!changed && (a || b)) changed = TRUE;   // a node was added or dropped

  g_list_free_full(before, free);
  return changed;
}

// ---------------------------------------------------------------------------
// steps: one panel action, addressed to part of a group
// ---------------------------------------------------------------------------

// the index of the bottom or top member, skipping markers: an element step
// addressed to a marker would edit no element, and removing or moving one
// would merge or reorder groups instead
static int _member_index(dt_masks_form_t *grp, const gboolean top)
{
  int i = 0, found = -1;
  for(GList *l = grp->points; l; l = g_list_next(l), i++)
    if(!dt_masks_point_is_marker(l->data))
    {
      found = i;
      if(!top) break;
    }
  return found;
}

gboolean _resolve_scope(dt_masks_form_t *grp, const scope_t s,
                        int *first, int *last)
{
  const int n = (int)g_list_length(grp->points);
  if(n == 0) return FALSE;

  switch(s)
  {
    case SCOPE_FIRST:
    case SCOPE_LAST:
      *first = *last = _member_index(grp, s == SCOPE_LAST);
      return *first >= 0;
    case SCOPE_RUN:
    default:
      // the run the first member belongs to, which ends at the next member
      // the panel reports as starting one
      *first = 0;
      *last = n - 1;
      {
        int i = 0;
        for(GList *l = grp->points; l; l = g_list_next(l), i++)
        {
          if(i > 0 && _starts_group(l)) { *last = i - 1; break; }
        }
      }
      return TRUE;
  }
}

/* A group break: a copy of the enclosing group's marker before member `idx`,
   so the new group keeps the settings its members were folded with. Nothing
   happens where `idx` already heads its group, or is in none */
static void _break_before(dt_develop_t *dev, dt_masks_form_t *grp,
                          const int idx)
{
  GList *node = g_list_nth(grp->points, idx);
  if(!node || dt_masks_point_is_marker(node->data)) return;
  GList *m = node->prev;
  while(m && !dt_masks_point_is_marker(m->data)) m = m->prev;
  if(!m || m == node->prev) return;

  dt_masks_point_group_t *pt = malloc(sizeof(dt_masks_point_group_t));
  if(!pt) return;
  memcpy(pt, m->data, sizeof(dt_masks_point_group_t));
  pt->formid = dt_masks_new_marker_id(dev ? dev->forms : NULL);
  grp->points = g_list_insert_before(grp->points, node, pt);
}

void _apply_step(dt_develop_t *dev, dt_masks_form_t *grp, const step_t *st)
{
  int first = 0, last = 0;
  if(!_resolve_scope(grp, st->s, &first, &last)) return;

  if(st->kind == STEP_POKE)
  {
    if(st->k == POKE_ELEM_BREAK)
      _break_before(dev, grp, first);
    else
      _apply_poke(grp->points, st->k, first, last);
    return;
  }

  if(st->kind == STEP_GEOM)
  {
    /* The shape the member refers to, resolved through the dev rather than
       carried in the step: a check may hold several form trees for the same
       edit (one in memory, one just read back out of the database), and a step
       that named a form pointer would edit the wrong one. Resolving by member
       index in each arm independently is the discipline every other step
       follows. */
    const dt_masks_point_group_t *pt = g_list_nth_data(grp->points, first);
    dt_masks_form_t *shape =
      (pt && dev) ? dt_masks_get_from_id(dev, pt->formid) : NULL;
    if(shape) _apply_geom(shape, (geom_t)st->k);
    return;
  }

  if(g_list_length(grp->points) < 2) return;

  GList *node = g_list_nth(grp->points, first);
  if(!node) return;

  if(st->kind == STEP_REMOVE)
  {
    free(node->data);
    grp->points = g_list_delete_link(grp->points, node);
  }
  else if(st->kind == STEP_MOVE_UP && node->prev
          && !(dt_masks_point_is_marker(node->prev->data) && !node->prev->prev))
  {
    // swapping the payloads reorders the run without disturbing the list
    // nodes, which is all the fold reads. Past a marker it moves the member
    // into the group below; the bottom group has none, and a member before
    // the first marker would sit in no group at all
    gpointer tmp = node->data;
    node->data = node->prev->data;
    node->prev->data = tmp;
  }
}

const char *_step_label(const step_t *st)
{
  switch(st->kind)
  {
    case STEP_REMOVE:  return "remove";
    case STEP_MOVE_UP: return "reorder";
    case STEP_GEOM:    return _geom_label((geom_t)st->k);
    default:           return _poke_label(st->k);
  }
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
