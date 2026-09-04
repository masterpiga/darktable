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

#include "develop/masks/postedit.h"

#include "develop/masks/postedit_internal.h"

#include "common/darktable.h"
#include "develop/blend.h"
#include "develop/blend_gui_internal.h"
#include "develop/imageop.h"
#include "develop/masks.h"
#include "develop/masks/harvest_read.h"
#include "develop/masks/verify_internal.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <float.h>
#include <glib/gstdio.h>
#include <json-glib/json-glib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

// Both sides run the identical code over identical data apart from the
// operator bits under test, so a real match is bit-exact. Anything above this
// is a genuine divergence, not float noise -- the same reasoning (and the same
// single-threaded replay) as in verify.c.
#define POSTEDIT_EPS 1e-6

// ---------------------------------------------------------------------------
// JSON helpers (same shape as verify.c's: a missing or wrongly-typed member
// takes the default rather than aborting on a hand-edited harvest file)
// ---------------------------------------------------------------------------

static gint64 _obj_int(JsonObject *o, const char *k, const gint64 dflt)
{
  if(!o || !json_object_has_member(o, k)) return dflt;
  JsonNode *n = json_object_get_member(o, k);
  if(!n || json_node_get_node_type(n) != JSON_NODE_VALUE) return dflt;
  return json_node_get_int(n);
}

static const char *_obj_str(JsonObject *o, const char *k, const char *dflt)
{
  if(!o || !json_object_has_member(o, k)) return dflt;
  JsonNode *n = json_object_get_member(o, k);
  if(!n || json_node_get_node_type(n) != JSON_NODE_VALUE) return dflt;
  const char *v = json_node_get_string(n);
  return v ? v : dflt;
}

// ---------------------------------------------------------------------------
// the pokes
// ---------------------------------------------------------------------------


/* The run-level modifiers that a combination sweep crosses with each other and
   with the within-group modes (phase C below). These four are what make a run
   of two members behave differently from two runs of one: a refinement applied
   once to a joined sub-mask rather than twice to two halves, an opacity or an
   invert-output applied once rather than twice, a bypass skipping both members
   rather than one. So they are exactly where a partition disagreement that
   survives every single control could still show itself. */
#define COMBO_BYPASS  (1 << 0)
#define COMBO_INVERT  (1 << 1)
#define COMBO_OPACITY (1 << 2)
#define COMBO_REFINE  (1 << 3)
#define COMBO_MASK_N  16

/* Where a comparison is tallied. Five blocks: the single controls, the same
   run-level controls again but applied to a run that only exists once a group
   break has been made (phase B -- those runs are unreachable from the
   un-poked partition, so nothing else addresses them), one bucket per
   modifier subset for the combination sweep, and then the shape controls
   (phase D), once on their own and once crossed with an intersection.

   The intersection is not decoration. Whether two shapes meet is what decides
   whether an intersection or a difference computes anything at all, so on a
   corpus of shapes that happen not to overlap those operators are swept
   inertly however many times they are poked. Moving a shape is the only thing
   that changes that, which is why the crossed bucket exists alongside the
   plain one. */
#define TALLY_SINGLE(k)     (k)
#define TALLY_BREAK(k)      (POKE_N + (k))
#define TALLY_COMBO(mask)   (POKE_N + POKE_FIRST_ELEM + (mask))
#define TALLY_GEOM_BASE     (POKE_N + POKE_FIRST_ELEM + COMBO_MASK_N)
#define TALLY_GEOM(g)       (TALLY_GEOM_BASE + (g))
#define TALLY_GEOM_ISECT(g) (TALLY_GEOM_BASE + GEOM_N + (g))
#define TALLY_N             (TALLY_GEOM_BASE + 2 * GEOM_N)

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

/** the label a tally slot prints under, into `buf` */
static const char *_tally_name(const int slot, char *buf, const size_t n)
{
  if(slot < POKE_N) return _poke_name[slot];
  if(slot < POKE_N + POKE_FIRST_ELEM)
  {
    g_snprintf(buf, n, "break+%s", _poke_name[slot - POKE_N]);
    return buf;
  }
  if(slot >= TALLY_GEOM_BASE + GEOM_N)
  {
    g_snprintf(buf, n, "%s+isect",
               _geom_label((geom_t)(slot - TALLY_GEOM_BASE - GEOM_N)));
    return buf;
  }
  if(slot >= TALLY_GEOM_BASE)
    return _geom_label((geom_t)(slot - TALLY_GEOM_BASE));
  const int mask = slot - (POKE_N + POKE_FIRST_ELEM);
  g_snprintf(buf, n, "combo:%s%s%s%s",
             (mask & COMBO_BYPASS)  ? "bypass "  : "",
             (mask & COMBO_INVERT)  ? "invert "  : "",
             (mask & COMBO_OPACITY) ? "opacity " : "",
             (mask & COMBO_REFINE)  ? "refine "  : "");
  // trailing space from the last flag, and the empty subset, which is never used
  const size_t l = strlen(buf);
  if(l && buf[l - 1] == ' ') buf[l - 1] = '\0';
  return buf;
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

   A run-level poke is broadcast across the whole range because that is what
   the panel does with it -- the fold reads it back from the run's head, but
   every member carries a copy so that any one of them can represent the group
   (see dt_masks_point_group_t's own comments on name/refinement/group_opacity).
   An element-level poke is passed first == last. */
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
      case POKE_ELEM_BREAK:   pt->group_start = 1;                 break;

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

gboolean _resolve_scope(dt_masks_form_t *grp, const scope_t s,
                        int *first, int *last)
{
  const int n = (int)g_list_length(grp->points);
  if(n == 0) return FALSE;

  switch(s)
  {
    case SCOPE_FIRST: *first = *last = 0;     return TRUE;
    case SCOPE_LAST:  *first = *last = n - 1; return TRUE;
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

void _apply_step(dt_develop_t *dev, dt_masks_form_t *grp, const step_t *st)
{
  int first = 0, last = 0;
  if(!_resolve_scope(grp, st->s, &first, &last)) return;

  if(st->kind == STEP_POKE)
  {
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
  else if(st->kind == STEP_MOVE_UP && node->prev)
  {
    // swapping the payloads reorders the run without disturbing the list
    // nodes, which is all the fold reads
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

// ---------------------------------------------------------------------------
// snapshotting the mutable state of every group in the replay
// ---------------------------------------------------------------------------

// exactly the fields a poke can touch, so restoring one puts the edit back
// where it started without deep-copying any form
typedef struct
{
  int state;
  float opacity;
  float group_opacity;
  int group_start;
  dt_masks_refinement_t refinement;
} pt_snap_t;

/** every group form in the replay, in dev->forms order. Caller frees the
    list (not the forms). */
static GList *_all_groups(dt_develop_t *dev)
{
  GList *out = NULL;
  for(GList *l = dev->forms; l; l = g_list_next(l))
  {
    dt_masks_form_t *f = l->data;
    if(f && (f->type & DT_MASKS_GROUP)) out = g_list_append(out, f);
  }
  return out;
}

static int _count_points(GList *groups)
{
  int n = 0;
  for(GList *g = groups; g; g = g_list_next(g))
    n += (int)g_list_length(((dt_masks_form_t *)g->data)->points);
  return n;
}

static void _snap_all(GList *groups, pt_snap_t *out)
{
  int i = 0;
  for(GList *g = groups; g; g = g_list_next(g))
    for(GList *l = ((dt_masks_form_t *)g->data)->points; l; l = g_list_next(l), i++)
    {
      const dt_masks_point_group_t *pt = l->data;
      out[i].state = pt->state;
      out[i].opacity = pt->opacity;
      out[i].group_opacity = pt->group_opacity;
      out[i].group_start = pt->group_start;
      out[i].refinement = pt->refinement;
    }
}

static void _restore_all(GList *groups, const pt_snap_t *in)
{
  int i = 0;
  for(GList *g = groups; g; g = g_list_next(g))
    for(GList *l = ((dt_masks_form_t *)g->data)->points; l; l = g_list_next(l), i++)
    {
      dt_masks_point_group_t *pt = l->data;
      pt->state = in[i].state;
      pt->opacity = in[i].opacity;
      pt->group_opacity = in[i].group_opacity;
      pt->group_start = in[i].group_start;
      pt->refinement = in[i].refinement;
    }
}

// ---------------------------------------------------------------------------
// per-edit report
// ---------------------------------------------------------------------------

typedef enum
{
  POSTEDIT_SKIPPED = 0,
  POSTEDIT_IDENTICAL,
  POSTEDIT_DIFFERENT,
  POSTEDIT_ERROR,
} postedit_result_t;

typedef struct
{
  postedit_result_t result;
  const char *skip_reason;
  gboolean repeat;
  int pokes;             // comparisons actually made
  int pokes_differing;   // ... of which disagreed
  int pokes_live;        // ... of which changed the mask at all
  double worst_diff;
  int worst_ctl;         // tally slot of the worst divergence, -1 if none
  // per-poke tallies, folded into the run summary
  int per_ctl[TALLY_N];
  int per_ctl_diff[TALLY_N];
  int per_ctl_live[TALLY_N];
} postedit_report_t;

typedef struct
{
  int total, identical, different, skipped, error;
  int skipped_already_normal;  // canon(G) == G, so nothing could differ
  int pokes, pokes_differing, pokes_live;
  double worst_diff;
  int worst_index;
  int per_ctl[TALLY_N];
  int per_ctl_diff[TALLY_N];
  int per_ctl_live[TALLY_N];
} postedit_stats_t;

// ---------------------------------------------------------------------------
// one edit
// ---------------------------------------------------------------------------

// one contiguous stretch of a group's point list: a whole run, or a single
// element when first == last
typedef struct { GList *points; int first, last; } run_t;

// one control change: which control, and the stretch it applies to
typedef struct { GList *points; poke_t k; int first, last; } mut_t;

/** every run of `grp` under the state currently loaded, appended to `out` */
static void _collect_runs(dt_masks_form_t *grp, GArray *out)
{
  int i = 0, first = 0;
  for(GList *l = grp->points; l; l = g_list_next(l), i++)
  {
    if(i > 0 && _starts_group(l))
    {
      const run_t rr = { grp->points, first, i - 1 };
      g_array_append_val(out, rr);
      first = i;
    }
  }
  if(i > 0)
  {
    const run_t rr = { grp->points, first, i - 1 };
    g_array_append_val(out, rr);
  }
}

// everything one comparison needs, so the three sweep phases can share it
typedef struct
{
  replay_t *r;
  GList *groups;
  const pt_snap_t *orig;
  const pt_snap_t *canon;
  size_t npix;
  const float *base;
  postedit_report_t *rep;
  gboolean any_diff;
} sweep_t;

/** Apply one configuration -- a set of control changes -- to the migrated
    group and to the normalized one, render both, and tally the comparison
    under `slot`. The two sides get the identical changes at the identical
    member indices, so the only thing that can differ is how the renderer reads
    them. */
static void _compare_config(sweep_t *sw, const mut_t *muts, const int nmuts,
                            const int slot)
{
  postedit_report_t *rep = sw->rep;

  _restore_all(sw->groups, sw->orig);
  for(int i = 0; i < nmuts; i++)
    _apply_poke(muts[i].points, muts[i].k, muts[i].first, muts[i].last);
  float *a = _render_mask(sw->r, NULL);

  _restore_all(sw->groups, sw->canon);
  for(int i = 0; i < nmuts; i++)
    _apply_poke(muts[i].points, muts[i].k, muts[i].first, muts[i].last);
  float *b = _render_mask(sw->r, NULL);

  rep->pokes++;
  rep->per_ctl[slot]++;

  if((a == NULL) != (b == NULL))
  {
    // one side published a mask and the other did not: a divergence in its own
    // right, and not one a pixel metric can express
    rep->pokes_differing++;
    rep->per_ctl_diff[slot]++;
    sw->any_diff = TRUE;
    if(rep->worst_ctl < 0) rep->worst_ctl = slot;
    rep->worst_diff = 1.0;
  }
  else if(a && b)
  {
    const double d = _max_abs_diff(a, b, sw->npix);
    if(d > POSTEDIT_EPS)
    {
      rep->pokes_differing++;
      rep->per_ctl_diff[slot]++;
      sw->any_diff = TRUE;
      if(d > rep->worst_diff) { rep->worst_diff = d; rep->worst_ctl = slot; }
    }
    // did this configuration do anything at all on this edit? Reported, never
    // failed on: plenty of controls are legitimately inert (difference on a
    // group whose shapes do not overlap, an operator on the base run the fold
    // never evaluates), and a check that failed on those would be asserting
    // something untrue.
    if(sw->base && _max_abs_diff(a, sw->base, sw->npix) > POSTEDIT_EPS)
    {
      rep->pokes_live++;
      rep->per_ctl_live[slot]++;
    }
  }

  dt_free_align(a);
  dt_free_align(b);
}

static void _postedit_edit(JsonObject *edit, postedit_report_t *rep)
{
  memset(rep, 0, sizeof(*rep));
  rep->result = POSTEDIT_SKIPPED;
  rep->worst_ctl = -1;

  JsonObject *bo = json_object_get_object_member(edit, "blend");
  if(!bo) { rep->skip_reason = "no blend object"; return; }

  dt_develop_blend_params_t bp;
  dt_masks_harvest_read_blend_params(bo, &bp);

  // same guard as --verify-masks: an already-flexi edit was never migrated, so
  // there is no migrated-vs-fresh question to ask about it
  if(bp.mask_mode & DEVELOP_MASK_FLEXI)
  {
    rep->skip_reason = "already flexi";
    return;
  }

  JsonObject *img = json_object_get_object_member(edit, "image");
  const int full_w = (int)_obj_int(img, "width", 0);
  const int full_h = (int)_obj_int(img, "height", 0);
  int w = full_w, h = full_h;
  if(w <= 0 || h <= 0) { rep->skip_reason = "no image dimensions"; return; }

  if(w > VERIFY_MAX_EDGE || h > VERIFY_MAX_EDGE)
  {
    const double s = (double)VERIFY_MAX_EDGE / (double)MAX(w, h);
    w = MAX(8, (int)(w * s));
    h = MAX(8, (int)(h * s));
  }

  JsonArray *fa = json_object_has_member(edit, "forms")
    ? json_object_get_array_member(edit, "forms") : NULL;
  GList *forms = fa ? dt_masks_harvest_read_forms(fa) : NULL;
  if(fa && json_array_get_length(fa) > 0 && !forms)
  {
    rep->skip_reason = "forms could not be reconstructed";
    return;
  }

  const char *op = _obj_str(edit, "operation", NULL);

  replay_t r;
  const char *init_err =
    _replay_init(&r, op, &bp, forms, full_w, full_h, w, h);
  if(init_err)
  {
    rep->result = POSTEDIT_ERROR;
    rep->skip_reason = init_err;
    return;
  }

  const size_t npix = (size_t)w * h;

  if(!dt_masks_migrate_classic_to_flexi(&r.module, r.module.blend_params, -1))
  {
    rep->result = POSTEDIT_ERROR;
    rep->skip_reason = "migration declined";
    _replay_cleanup(&r);
    return;
  }

  GList *groups = _all_groups(&r.dev);
  const int npts = _count_points(groups);
  if(!groups || npts == 0)
  {
    rep->skip_reason = "no group to edit";
    g_list_free(groups);
    _replay_cleanup(&r);
    return;
  }

  pt_snap_t *orig = calloc((size_t)npts, sizeof(pt_snap_t));
  pt_snap_t *canon = calloc((size_t)npts, sizeof(pt_snap_t));
  if(!orig || !canon)
  {
    rep->result = POSTEDIT_ERROR;
    rep->skip_reason = "out of memory";
    free(orig); free(canon);
    g_list_free(groups);
    _replay_cleanup(&r);
    return;
  }

  _snap_all(groups, orig);
  // canon: the panel's own normalization, applied to every group including the
  // nested ones. This is the group a user building the same shapes from
  // scratch in the panel would have.
  for(GList *g = groups; g; g = g_list_next(g))
    _normalize_group_operators(g->data);
  _snap_all(groups, canon);
  _restore_all(groups, orig);

  // Nothing to sweep: the two sides are literally the same data, so every
  // comparison below would be trivially equal. Exact, not a heuristic.
  if(memcmp(orig, canon, (size_t)npts * sizeof(pt_snap_t)) == 0)
  {
    rep->skip_reason = "already normalized";
    free(orig); free(canon);
    g_list_free(groups);
    _replay_cleanup(&r);
    return;
  }

  /* Every run in every group, under whatever state is loaded when this is
     called. Computed on the canon side and used for both -- _starts_group()
     reads the effective operator, which normalization only writes out rather
     than changes, so the partition is identical either way. Addressing both
     sides by the same member indices is what makes the comparison
     apples-to-apples: the only thing left that can differ is how the renderer
     reads what was written. */
  GArray *runs = g_array_new(FALSE, FALSE, sizeof(run_t));
  GArray *elems = g_array_new(FALSE, FALSE, sizeof(run_t));
  _restore_all(groups, canon);
  for(GList *g = groups; g; g = g_list_next(g))
  {
    dt_masks_form_t *grp = g->data;
    _collect_runs(grp, runs);
    int i = 0;
    for(GList *l = grp->points; l; l = g_list_next(l), i++)
    {
      const run_t ee = { grp->points, i, i };
      g_array_append_val(elems, ee);
    }
  }
  _restore_all(groups, orig);

  // the un-poked canon render, so a poke can be reported as live or inert
  float *base = _render_mask(&r, NULL);

  sweep_t sw = { &r, groups, orig, canon, npix, base, rep, FALSE };

  // --- phase A: one control at a time --------------------------------------
  for(int k = 0; k < POKE_N; k++)
  {
    GArray *targets = (k >= POKE_FIRST_ELEM) ? elems : runs;
    for(guint t = 0; t < targets->len; t++)
    {
      const run_t tg = g_array_index(targets, run_t, t);
      const mut_t m = { tg.points, (poke_t)k, tg.first, tg.last };
      _compare_config(&sw, &m, 1, TALLY_SINGLE(k));
    }
  }

  /* --- phase B: run controls on a run that only exists after a break -------

     Phase A addresses runs from the un-poked partition, so a run created by
     DT_MASKS_STATE's group_start marker is never the target of a run-level
     control there -- and a run-level control read from a run head is exactly
     the shape of the bug this whole check exists for. Break at each position
     in turn, re-derive the partition with the break in place, and sweep the
     run controls over what that produces.

     Only the two runs the break itself created are swept, not every run in the
     group. That is exact rather than a budget: a break at `e` splits one run
     into the one ending at e-1 and the one starting at e, and leaves every
     other run's extent and head untouched, so poking one of those would be
     comparing the head phase A already compared, reading the same bits the
     same way. Sweeping them all would make this phase quadratic in the group
     size for no extra discrimination -- a thirty-shape drawn group would cost
     more than the rest of the harvest put together. */
  for(GList *g = groups; g; g = g_list_next(g))
  {
    dt_masks_form_t *grp = g->data;
    const int n = (int)g_list_length(grp->points);
    // a break on the bottom point is meaningless -- it has nothing below it to
    // be broken away from, and _normalize_group_operators() clears it there
    for(int e = 1; e < n; e++)
    {
      _restore_all(groups, canon);
      _apply_poke(grp->points, POKE_ELEM_BREAK, e, e);
      GArray *broken = g_array_new(FALSE, FALSE, sizeof(run_t));
      _collect_runs(grp, broken);
      _restore_all(groups, orig);

      for(guint t = 0; t < broken->len; t++)
      {
        const run_t tg = g_array_index(broken, run_t, t);
        // the two the break made: the new head at `e`, and the run truncated
        // just below it
        if(tg.first != e && tg.last != e - 1) continue;
        for(int k = 0; k < POKE_FIRST_ELEM; k++)
        {
          const mut_t m[2] = { { grp->points, POKE_ELEM_BREAK, e, e },
                               { tg.points, (poke_t)k, tg.first, tg.last } };
          _compare_config(&sw, m, 2, TALLY_BREAK(k));
        }
      }
      g_array_free(broken, TRUE);
    }
  }

  /* --- phase C: the run-level modifiers, in every combination --------------

     Each of bypass, invert-output, group opacity and group refinement is
     applied once per run, so all four distinguish "one run of two members"
     from "two runs of one" -- which is what a partition disagreement produces.
     Crossed with the within-group mode, which is the other control read from
     the head, and which only means anything for a run with more than one
     member. Every non-empty subset, because the fold branches on them in
     sequence (bypass short-circuits, the within mode picks the seed, the
     refinement runs on the folded sub-mask, invert after that, opacity after
     that) and a slip could live in any one of those orderings. */
  for(guint t = 0; t < runs->len; t++)
  {
    const run_t tg = g_array_index(runs, run_t, t);
    for(int mask = 1; mask < COMBO_MASK_N; mask++)
      for(int wm = 0; wm < 4; wm++)
      {
        mut_t m[5];
        int nm = 0;
        m[nm++] = (mut_t){ tg.points, (poke_t)(POKE_WITHIN_UNION + wm),
                           tg.first, tg.last };
        if(mask & COMBO_BYPASS)
          m[nm++] = (mut_t){ tg.points, POKE_GROUP_BYPASS, tg.first, tg.last };
        if(mask & COMBO_INVERT)
          m[nm++] = (mut_t){ tg.points, POKE_GROUP_INVERT, tg.first, tg.last };
        if(mask & COMBO_OPACITY)
          m[nm++] = (mut_t){ tg.points, POKE_GROUP_OPACITY, tg.first, tg.last };
        if(mask & COMBO_REFINE)
          m[nm++] = (mut_t){ tg.points, POKE_GROUP_REFINE, tg.first, tg.last };
        _compare_config(&sw, m, nm, TALLY_COMBO(mask));
      }
  }

  /* --- phase D: the shape controls ----------------------------------------

     Everything above changes how members are combined; nothing above changes
     what they are. That leaves the harvested geometry as a fixed input, and it
     is not a neutral one: on a corpus where the shapes do not overlap, an
     intersection has nothing to intersect and a difference nothing to remove,
     so those operators are swept over and over while computing the same empty
     answer. Moving a shape is what turns them live.

     One comparison per (shape, control) on its own, and one crossed with an
     intersection over the first run. The geometry is applied to the form,
     which both arms share -- deliberately: this is not asking whether a shape
     edit is faithful (roundtrip.c and persist.c ask that), it is asking
     whether the migrated and from-scratch partitions still agree once the mask
     underneath them has changed shape.

     Controls a shape does not implement are skipped rather than tallied.
     _apply_geom() reports that from the point data, so a slider that is
     accepted and then clamped back to where it was does not count as swept. */
  for(GList *g = groups; g; g = g_list_next(g))
  {
    dt_masks_form_t *grp = g->data;
    for(GList *p = grp->points; p; p = g_list_next(p))
    {
      const dt_masks_point_group_t *pt = p->data;
      dt_masks_form_t *shape = dt_masks_get_from_id(&r.dev, pt->formid);
      if(!shape) continue;

      for(int gk = 0; gk < GEOM_N; gk++)
      {
        GList *geom_snap = _geom_snapshot(shape);
        if(!geom_snap) break;              // nothing editable on this form
        if(!_apply_geom(shape, (geom_t)gk))
        {
          _geom_restore(shape, geom_snap);
          continue;
        }

        /* The two comparisons want DIFFERENT baselines, and getting that
           wrong makes one of them report nothing.

           For the plain one the question is "did the shape edit change the
           mask", so it is measured against `base` -- the canon render at the
           harvested geometry. Measuring it against a same-geometry render
           instead makes its live count identically zero by construction: with
           no poke applied, "did this configuration differ from the un-poked
           canon render" is the same question as "did the two arms disagree",
           which is the thing being asserted to be no. A column that can only
           ever read 0 is worse than no column.

           For the crossed one the question is "did the operator change the
           mask, under this geometry", so it is measured against the un-poked
           render AT this geometry -- otherwise the shape edit alone would
           make every configuration read as live and the operator's own effect
           would be invisible. */
        _restore_all(groups, canon);
        float *geom_base = _render_mask(&r, NULL);

        _compare_config(&sw, NULL, 0, TALLY_GEOM(gk));

        const float *saved_base = sw.base;
        sw.base = geom_base;
        const mut_t m = { grp->points, POKE_OP_INTERSECTION,
                          0, (int)g_list_length(grp->points) - 1 };
        _compare_config(&sw, &m, 1, TALLY_GEOM_ISECT(gk));
        sw.base = saved_base;

        dt_free_align(geom_base);
        _geom_restore(shape, geom_snap);
      }
    }
  }

  const gboolean any_diff = sw.any_diff;
  _restore_all(groups, orig);
  dt_free_align(base);

  rep->result = rep->pokes == 0 ? POSTEDIT_SKIPPED
                : any_diff ? POSTEDIT_DIFFERENT : POSTEDIT_IDENTICAL;
  if(rep->pokes == 0) rep->skip_reason = "no controls to sweep";

  g_array_free(runs, TRUE);
  g_array_free(elems, TRUE);
  free(orig);
  free(canon);
  g_list_free(groups);
  _replay_cleanup(&r);
}

// ---------------------------------------------------------------------------
// the run
// ---------------------------------------------------------------------------

static const char *_result_name(const postedit_result_t r)
{
  switch(r)
  {
    case POSTEDIT_IDENTICAL: return "identical";
    case POSTEDIT_DIFFERENT: return "DIFFERENT";
    case POSTEDIT_ERROR:     return "error";
    default:                 return "skipped";
  }
}

gboolean dt_masks_postedit_harvest_section(const char *json_path, FILE *rf)
{
  setvbuf(stdout, NULL, _IOLBF, 0);

#ifdef _OPENMP
  // single-threaded for the same reason as verify.c: a reduction whose float
  // addition order depends on thread scheduling makes the last bits of the
  // mask move between runs, and this check compares at 1e-6
  omp_set_num_threads(1);
#endif

  GError *err = NULL;
  JsonParser *parser = dt_masks_harvest_load(json_path, &err);
  if(!parser)
  {
    fprintf(stderr, "[postedit] cannot read %s: %s\n",
            json_path, err ? err->message : "unknown error");
    g_clear_error(&err);
    return FALSE;
  }

  JsonNode *root = json_parser_get_root(parser);
  JsonObject *ro = root ? json_node_get_object(root) : NULL;
  JsonArray *edits = ro && json_object_has_member(ro, "edits")
    ? json_object_get_array_member(ro, "edits") : NULL;
  if(!edits)
  {
    fprintf(stderr, "[postedit] %s has no \"edits\" array\n", json_path);
    g_object_unref(parser);
    return FALSE;
  }

  postedit_stats_t st;
  memset(&st, 0, sizeof(st));
  st.worst_index = -1;

  if(rf) fprintf(rf, "\n  \"source\": \"%s\",\n  \"edits\": [", json_path);
  gboolean first_report = TRUE;

  const guint n = json_array_get_length(edits);
  printf("[postedit] sweeping the panel's controls over %u harvested edits"
         " from %s\n", n, json_path);

  // exact repeats are swept once and their verdict reused, the same bookkeeping
  // --verify-masks does (see dt_masks_harvest_edit_key)
  GHashTable *seen =
    g_hash_table_new_full(g_str_hash, g_str_equal, g_free, g_free);
  int swept_unique = 0;

  for(guint i = 0; i < n; i++)
  {
    JsonObject *edit = json_array_get_object_element(edits, i);
    if(!edit) continue;

    postedit_report_t rep;
    gchar *key = dt_masks_harvest_edit_key(edit);
    const postedit_report_t *cached = key ? g_hash_table_lookup(seen, key) : NULL;
    if(cached)
    {
      rep = *cached;
      rep.repeat = TRUE;
      g_free(key);
    }
    else
    {
      if(darktable.unmuted & DT_DEBUG_MASKS)
        printf("[postedit] edit %u op=%s\n", i,
               _obj_str(edit, "operation", "?"));
      _postedit_edit(edit, &rep);
      rep.repeat = FALSE;
      swept_unique++;
      if(key)
      {
        postedit_report_t *store = malloc(sizeof(postedit_report_t));
        if(store) { *store = rep; g_hash_table_insert(seen, key, store); }
        else g_free(key);
      }
    }

    st.total++;
    switch(rep.result)
    {
      case POSTEDIT_IDENTICAL: st.identical++; break;
      case POSTEDIT_DIFFERENT: st.different++; break;
      case POSTEDIT_ERROR:     st.error++;     break;
      default:                 st.skipped++;   break;
    }
    if(rep.result == POSTEDIT_SKIPPED && rep.skip_reason
       && !strcmp(rep.skip_reason, "already normalized"))
      st.skipped_already_normal++;

    st.pokes += rep.pokes;
    st.pokes_differing += rep.pokes_differing;
    st.pokes_live += rep.pokes_live;
    for(int k = 0; k < TALLY_N; k++)
    {
      st.per_ctl[k] += rep.per_ctl[k];
      st.per_ctl_diff[k] += rep.per_ctl_diff[k];
      st.per_ctl_live[k] += rep.per_ctl_live[k];
    }
    if(rep.worst_diff > st.worst_diff)
    {
      st.worst_diff = rep.worst_diff;
      st.worst_index = (int)i;
    }

    if(rf)
    {
      char wbuf[64];
      gchar *worst = rep.worst_ctl >= 0
        ? g_strdup_printf("\"%s\"", _tally_name(rep.worst_ctl, wbuf, sizeof(wbuf)))
        : g_strdup("null");
      fprintf(rf, "%s\n    {\"index\": %u, \"operation\": \"%s\","
                  " \"result\": \"%s\", \"repeat\": %s, \"pokes\": %d,"
                  " \"pokes_differing\": %d, \"pokes_live\": %d,"
                  " \"worst_diff\": %.9g, \"worst_control\": %s,"
                  " \"reason\": \"%s\"}",
              first_report ? "" : ",", i, _obj_str(edit, "operation", "?"),
              _result_name(rep.result), rep.repeat ? "true" : "false",
              rep.pokes, rep.pokes_differing, rep.pokes_live, rep.worst_diff,
              worst, rep.skip_reason ? rep.skip_reason : "");
      g_free(worst);
      first_report = FALSE;
    }
  }

  g_hash_table_destroy(seen);
  g_object_unref(parser);

  const gboolean passed = (st.different == 0 && st.error == 0);

  printf("[postedit]\n");
  printf("[postedit] edits             : %d  (%d swept, %d reused as repeats)\n",
         st.total, swept_unique, st.total - swept_unique);
  printf("[postedit]   identical       : %d\n", st.identical);
  printf("[postedit]   DIFFERENT       : %d\n", st.different);
  printf("[postedit]   skipped         : %d  (of which %d already normalized,"
         " i.e. nothing to compare)\n", st.skipped, st.skipped_already_normal);
  printf("[postedit]   errors          : %d\n", st.error);
  printf("[postedit]\n");
  printf("[postedit] configurations compared : %d\n", st.pokes);
  printf("[postedit]   disagreed              : %d\n", st.pokes_differing);
  printf("[postedit]   changed the mask       : %d  (the rest are legitimately"
         " inert on their edit)\n", st.pokes_live);
  if(st.worst_index >= 0)
    printf("[postedit]   worst difference       : %.9g (edit %d)\n",
           st.worst_diff, st.worst_index);
  printf("[postedit]\n");
  printf("[postedit] per control                            compared  disagreed  live\n");
  for(int k = 0; k < TALLY_N; k++)
  {
    // an unused slot (the empty modifier subset) or a phase this corpus never
    // reached; printing it would only pad the table
    if(st.per_ctl[k] == 0) continue;
    char nbuf[64];
    printf("[postedit]   %-36s %8d  %9d  %4d\n",
           _tally_name(k, nbuf, sizeof(nbuf)),
           st.per_ctl[k], st.per_ctl_diff[k], st.per_ctl_live[k]);
  }

  if(rf)
  {
    fprintf(rf, "\n  ],\n  \"summary\": {\n");
    fprintf(rf, "    \"total\": %d,\n", st.total);
    fprintf(rf, "    \"identical\": %d,\n", st.identical);
    fprintf(rf, "    \"different\": %d,\n", st.different);
    fprintf(rf, "    \"skipped\": %d,\n", st.skipped);
    fprintf(rf, "    \"skipped_already_normalized\": %d,\n",
            st.skipped_already_normal);
    fprintf(rf, "    \"errors\": %d,\n", st.error);
    fprintf(rf, "    \"pokes\": %d,\n", st.pokes);
    fprintf(rf, "    \"pokes_differing\": %d,\n", st.pokes_differing);
    fprintf(rf, "    \"pokes_live\": %d,\n", st.pokes_live);
    fprintf(rf, "    \"worst_diff\": %.9g,\n", st.worst_diff);
    fprintf(rf, "    \"worst_index\": %d,\n", st.worst_index);
    fprintf(rf, "    \"per_control\": {");
    gboolean first_ctl = TRUE;
    for(int k = 0; k < TALLY_N; k++)
    {
      if(st.per_ctl[k] == 0) continue;
      char nbuf[64];
      fprintf(rf, "%s\n      \"%s\": {\"compared\": %d, \"differing\": %d,"
                  " \"live\": %d}",
              first_ctl ? "" : ",", _tally_name(k, nbuf, sizeof(nbuf)),
              st.per_ctl[k], st.per_ctl_diff[k], st.per_ctl_live[k]);
      first_ctl = FALSE;
    }
    fprintf(rf, "\n    },\n");
    fprintf(rf, "    \"passed\": %s\n  }\n", passed ? "true" : "false");
  }

  return passed;
}

gboolean dt_masks_postedit_harvest(const char *json_path, const char *report_path)
{
  FILE *rf = report_path ? g_fopen(report_path, "wb") : NULL;
  if(report_path && !rf)
    fprintf(stderr, "[postedit] cannot write report to %s\n", report_path);

  if(rf) fputs("{", rf);
  const gboolean ok = dt_masks_postedit_harvest_section(json_path, rf);
  if(rf)
  {
    fputs("}\n", rf);
    fclose(rf);
    printf("[postedit] per-edit report written to %s\n", report_path);
  }
  return ok;
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
