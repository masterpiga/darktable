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

#include "develop/masks/persist.h"

#include "common/darktable.h"
#include "develop/blend_gui_internal.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/masks.h"
#include "develop/masks/harvest_read.h"
#include "develop/masks/postedit_internal.h"
#include "develop/masks/scratch_image.h"
#include "develop/masks/verify_internal.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <glib/gstdio.h>
#include <json-glib/json-glib.h>
#include <stdio.h>
#include <string.h>

// the scratch image both arms are driven through, as in roundtrip.c: real ids
// start at 1 and this runs against a throwaway database, so nothing collides
#define PERSIST_IMGID 1

// Both arms run the identical blend over data that should be identical, so a
// real match is bit-exact; the same reasoning as verify.c and postedit.c.
#define PERSIST_EPS 1e-6

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
// the sequences
// ---------------------------------------------------------------------------

/* A sequence is a list of steps: see postedit_internal.h for the step
   vocabulary itself (pokes, shape controls, and the two list operations), and
   persist.h for why each step's scope is resolved against the group as it
   stands in each arm independently rather than pinned once up front. */
typedef struct
{
  const char *name;
  const char *seam;   // what a divergence here would mean, for the report
  int n;
  step_t step[3];
} seq_t;

/* The sequences, grouped by the seam each block covers.

   A sequence is short on purpose. What is being asked is whether a *single*
   save is transparent; a longer chain multiplies the opportunities but tests
   the same thing, and it makes a failure harder to read. Where three steps
   appear it is because the seam genuinely needs them: something must create a
   boundary, something must cross the save, and something must read the
   boundary back. */
static const seq_t _sequences[] =
{
  // ---- one edit, one save. The floor: if these do not hold, nothing longer
  // means anything, and a failure names the single control that did not
  // survive rather than an interaction.
  { "single:op-difference", "a between-group operator does not survive a save",
    1, { { POKE_OP_DIFFERENCE, SCOPE_RUN } } },
  { "single:within-isect", "a within-group mode does not survive a save",
    1, { { POKE_WITHIN_ISECT, SCOPE_RUN } } },
  { "single:group-opacity", "a group opacity does not survive a save",
    1, { { POKE_GROUP_OPACITY, SCOPE_RUN } } },
  { "single:group-refine", "a group refinement does not survive a save",
    1, { { POKE_GROUP_REFINE, SCOPE_RUN } } },
  { "single:group-invert", "an invert-output does not survive a save",
    1, { { POKE_GROUP_INVERT, SCOPE_RUN } } },
  { "single:group-bypass", "a bypass does not survive a save",
    1, { { POKE_GROUP_BYPASS, SCOPE_RUN } } },
  { "single:elem-break", "a group break does not survive a save",
    1, { { POKE_ELEM_BREAK, SCOPE_LAST } } },
  { "single:elem-disable", "a disabled element does not survive a save",
    1, { { POKE_ELEM_DISABLE, SCOPE_LAST } } },
  { "single:elem-opacity", "an element opacity does not survive a save",
    1, { { POKE_ELEM_OPACITY, SCOPE_FIRST } } },

  // ---- a run boundary created by an operator change, then read back through
  // a control the fold takes from the run's head. This is issue #21905 with
  // the user in place of migration: the boundary is only implied by the
  // operator, so if the save stores the operator without the boundary the
  // second edit addresses a group the renderer does not have.
  { "boundary:difference then within", "a run boundary implied by an operator is not stored",
    2, { { POKE_OP_DIFFERENCE, SCOPE_RUN }, { POKE_WITHIN_ISECT, SCOPE_RUN } } },
  { "boundary:intersection then opacity", "a run boundary implied by an operator is not stored",
    2, { { POKE_OP_INTERSECTION, SCOPE_RUN }, { POKE_GROUP_OPACITY, SCOPE_RUN } } },
  { "boundary:sum then invert", "a run boundary implied by an operator is not stored",
    2, { { POKE_OP_SUM, SCOPE_RUN }, { POKE_GROUP_INVERT, SCOPE_RUN } } },
  { "boundary:exclusion then refine", "a run boundary implied by an operator is not stored",
    2, { { POKE_OP_EXCLUSION, SCOPE_RUN }, { POKE_GROUP_REFINE, SCOPE_RUN } } },

  // ---- the same, but with the boundary made explicitly. group_start is the
  // one field with no other representation: an operator can be re-derived from
  // the state bits, a break cannot, so if anything is going to be dropped by
  // the writer it is this.
  { "break:then within", "an explicit group break is not stored",
    2, { { POKE_ELEM_BREAK, SCOPE_LAST }, { POKE_WITHIN_MULTIPLY, SCOPE_RUN } } },
  { "break:then opacity", "an explicit group break is not stored",
    2, { { POKE_ELEM_BREAK, SCOPE_LAST }, { POKE_GROUP_OPACITY, SCOPE_RUN } } },
  { "break:then operator", "an explicit group break is not stored",
    2, { { POKE_ELEM_BREAK, SCOPE_LAST }, { POKE_OP_DIFFERENCE, SCOPE_RUN } } },

  // ---- migration's own output, built on. The first step changes nothing
  // structural, so the second is applied to a group whose boundaries and
  // disable bits came from _split_nonunion_runs() and
  // _repair_base_case_overwrite() and have now been through storage once.
  { "migrated:opacity then refine", "migration's markers are lost by a save that did not touch them",
    2, { { POKE_ELEM_OPACITY, SCOPE_FIRST }, { POKE_GROUP_REFINE, SCOPE_RUN } } },
  { "migrated:disable then within", "the base-case repair's disable bits are lost by a save",
    2, { { POKE_ELEM_DISABLE, SCOPE_LAST }, { POKE_WITHIN_ISECT, SCOPE_RUN } } },

  // ---- a run-level modifier set before the save and read after it. These
  // four distinguish one run of two members from two runs of one, so a
  // partition that failed to survive shows itself here even when the operator
  // bits came back intact.
  { "modifier:bypass then within", "a modifier and the partition it reads disagree across a save",
    2, { { POKE_GROUP_BYPASS, SCOPE_RUN }, { POKE_WITHIN_ISECT, SCOPE_RUN } } },
  { "modifier:invert then operator", "a modifier and the partition it reads disagree across a save",
    2, { { POKE_GROUP_INVERT, SCOPE_RUN }, { POKE_OP_DIFFERENCE, SCOPE_RUN } } },
  { "modifier:bypass then opacity", "a modifier and the partition it reads disagree across a save",
    2, { { POKE_GROUP_BYPASS, SCOPE_RUN }, { POKE_GROUP_OPACITY, SCOPE_RUN } } },

  // ---- the member list itself. Deleting a shape and reordering rows are
  // ordinary panel actions, and a run is a maximal stretch of the list, so
  // both move boundaries that a later control then reads back. Nothing in the
  // poke vocabulary can express either, which is why they are here: without
  // them the checks are silent about a whole axis of what the panel can do.
  { "structural:remove then within", "a deletion moves a run boundary the save does not carry",
    2, { { POKE_N, SCOPE_LAST, STEP_REMOVE }, { POKE_WITHIN_ISECT, SCOPE_RUN } } },
  { "structural:remove then opacity", "a deletion moves a run boundary the save does not carry",
    2, { { POKE_N, SCOPE_FIRST, STEP_REMOVE }, { POKE_GROUP_OPACITY, SCOPE_RUN } } },
  { "structural:reorder then operator", "a reorder moves a run boundary the save does not carry",
    2, { { POKE_N, SCOPE_LAST, STEP_MOVE_UP }, { POKE_OP_DIFFERENCE, SCOPE_RUN } } },
  { "structural:reorder then within", "a reorder moves a run boundary the save does not carry",
    2, { { POKE_N, SCOPE_LAST, STEP_MOVE_UP }, { POKE_WITHIN_ISECT, SCOPE_RUN } } },
  { "structural:operator then remove", "a deletion after an operator change loses the boundary",
    2, { { POKE_OP_DIFFERENCE, SCOPE_RUN }, { POKE_N, SCOPE_LAST, STEP_REMOVE } } },
  { "structural:break then reorder", "a reorder across an explicit break loses it",
    2, { { POKE_ELEM_BREAK, SCOPE_LAST }, { POKE_N, SCOPE_LAST, STEP_MOVE_UP } } },

  /* ---- the shapes themselves. Everything above edits how members combine;
     these edit what they are, which is the one part of a mask with a per-type
     serialised representation of its own -- a blob of dt_masks_point_<type>_t
     in masks_history, written by code the harvested forms never exercise,
     because they arrive already-serialised and go back out unchanged.

     A shape control is therefore the only step here whose *first* half is at
     risk: a poke that fails to persist loses a state bit, while a geometry
     edit that fails to persist loses the shape. path.c's resize is the sharp
     case -- it keeps a cached baseline next to the points, so a save has to
     either carry that or reconstruct it, and a shape whose baseline came back
     wrong renders at the wrong size on the second open and at the right one on
     the first.

     Paired with a control that reads a run boundary, for the same reason as
     every other block: a lost boundary and a lost shape look nothing alike in
     the report, and pairing them costs one step. */
  { "geom:translate then within", "a moved shape does not survive a save",
    2, { GEOM_STEP(GEOM_TRANSLATE, SCOPE_FIRST), { POKE_WITHIN_ISECT, SCOPE_RUN } } },
  { "geom:size then operator", "a resized shape does not survive a save",
    2, { GEOM_STEP(GEOM_SIZE, SCOPE_FIRST), { POKE_OP_DIFFERENCE, SCOPE_RUN } } },
  { "geom:feather then opacity", "a re-feathered shape does not survive a save",
    2, { GEOM_STEP(GEOM_FEATHER, SCOPE_FIRST), { POKE_GROUP_OPACITY, SCOPE_RUN } } },
  { "geom:node then break", "a dragged node does not survive a save",
    2, { GEOM_STEP(GEOM_NODE, SCOPE_FIRST), { POKE_ELEM_BREAK, SCOPE_LAST } } },
  { "geom:rotation then within", "a rotated shape does not survive a save",
    2, { GEOM_STEP(GEOM_ROTATION, SCOPE_FIRST), { POKE_WITHIN_MULTIPLY, SCOPE_RUN } } },
  { "geom:translate then translate", "a shape edited twice across two saves drifts",
    2, { GEOM_STEP(GEOM_TRANSLATE, SCOPE_FIRST),
         GEOM_STEP(GEOM_TRANSLATE, SCOPE_FIRST) } },
  { "geom:size then size", "a resize baseline is rebuilt from the stored shape",
    2, { GEOM_STEP(GEOM_SIZE, SCOPE_FIRST), GEOM_STEP(GEOM_SIZE, SCOPE_FIRST) } },
  { "geom:remove then translate", "a shape edit after a deletion addresses the wrong member",
    2, { { POKE_N, SCOPE_FIRST, STEP_REMOVE },
         GEOM_STEP(GEOM_TRANSLATE, SCOPE_FIRST) } },

  // ---- three steps, where the seam needs one: make a boundary, cross a save
  // with an unrelated change, then read the boundary back. Two saves, and the
  // middle edit is what stops the third from being a repeat of the first.
  { "chain:operator break opacity", "a boundary does not survive an intervening save",
    3, { { POKE_OP_DIFFERENCE, SCOPE_RUN },
         { POKE_ELEM_BREAK, SCOPE_LAST },
         { POKE_GROUP_OPACITY, SCOPE_RUN } } },
  { "chain:break operator invert", "a boundary does not survive an intervening save",
    3, { { POKE_ELEM_BREAK, SCOPE_LAST },
         { POKE_OP_INTERSECTION, SCOPE_RUN },
         { POKE_GROUP_INVERT, SCOPE_RUN } } },
  { "chain:refine operator within", "a refinement's scope moves across a save",
    3, { { POKE_GROUP_REFINE, SCOPE_RUN },
         { POKE_OP_SUM, SCOPE_RUN },
         { POKE_WITHIN_ISECT, SCOPE_RUN } } },
};

#define SEQ_N ((int)(sizeof(_sequences) / sizeof(_sequences[0])))

// ---------------------------------------------------------------------------
// driving the scratch image
// ---------------------------------------------------------------------------

/** The module's own mask group in `dev`, or NULL.

    Read from the module rather than from dev->forms at large: dev->forms is
    per image and every masks_history row is a cumulative snapshot, so it
    routinely carries groups belonging to other modules and groups orphaned by
    earlier edits (see roundtrip.c, which learned the same thing the hard way). */
static dt_masks_form_t *_target_group(dt_develop_t *dev,
                                      const dt_develop_blend_params_t *bp)
{
  if(!bp || !(bp->mask_mode & DEVELOP_MASK_FLEXI)) return NULL;
  if(!dt_is_valid_maskid(bp->mask_id)) return NULL;
  dt_masks_form_t *grp = dt_masks_get_from_id(dev, bp->mask_id);
  return (grp && (grp->type & DT_MASKS_GROUP)) ? grp : NULL;
}

/** Every group the module renders through, the top one first and its nested
    groups after, in a deterministic order.

    Sweeping only the top group was the earlier behaviour, on the grounds that
    masks_history stores one flat row per form so a nested group traverses the
    same storage code. That is true of the storage half and wrong about the
    rest: a sequence here pokes a control and then reads the *partition* back,
    and a nested group is where the partition is easiest to get wrong, because
    the fold recurses into the child while taking run state from the parent's
    head. 5.7% of harvested edits carry one (19% in some libraries).

    Order is a breadth-first walk over member formids in list order, so both
    arms resolve the same index to the same group without either being handed
    a partition the other did not compute. Depth-bounded and deduplicated: a
    malformed or cyclic tree must not spin. */
static GList *_all_groups(dt_develop_t *dev, const dt_develop_blend_params_t *bp)
{
  dt_masks_form_t *top = _target_group(dev, bp);
  if(!top) return NULL;

  GList *out = g_list_append(NULL, top);
  for(GList *l = out; l; l = g_list_next(l))
  {
    if(g_list_position(out, l) > 64) break;   // bound on a malformed tree
    const dt_masks_form_t *grp = l->data;
    for(GList *p = grp->points; p; p = g_list_next(p))
    {
      const dt_masks_point_group_t *pt = p->data;
      dt_masks_form_t *child = dt_masks_get_from_id(dev, pt->formid);
      if(child && (child->type & DT_MASKS_GROUP) && !g_list_find(out, child))
        out = g_list_append(out, child);
    }
  }
  return out;
}

/** the `index`-th group of that walk, or NULL when the tree has fewer */
static dt_masks_form_t *_group_at(dt_develop_t *dev,
                                  const dt_develop_blend_params_t *bp,
                                  const int index)
{
  GList *all = _all_groups(dev, bp);
  dt_masks_form_t *grp = g_list_nth_data(all, index);
  g_list_free(all);
  return grp;
}

/** Every group member's classic-visible marker state, as a comparable string.

    Only the two fields migration writes into a reused classic tree: the state
    bits (of which DT_MASKS_STATE_DISABLE is the one the classic fold reads)
    and group_start. Members are keyed by formid and sorted, so a difference
    here is a difference in markers and not in list order. */
static gint _digest_cmp(gconstpointer a, gconstpointer b)
{
  return strcmp(a, b);
}

static gchar *_marker_digest(GList *forms)
{
  GList *rows = NULL;
  for(GList *f = forms; f; f = g_list_next(f))
  {
    const dt_masks_form_t *form = f->data;
    if(!(form->type & DT_MASKS_GROUP)) continue;
    for(GList *p = form->points; p; p = g_list_next(p))
    {
      const dt_masks_point_group_t *pt = p->data;
      rows = g_list_prepend(rows, g_strdup_printf("%d:%d:%d:%d;", form->formid,
                                                  pt->formid, pt->state,
                                                  pt->group_start));
    }
  }
  rows = g_list_sort(rows, _digest_cmp);
  GString *g = g_string_new(NULL);
  for(GList *l = rows; l; l = g_list_next(l)) g_string_append(g, l->data);
  g_list_free_full(rows, g_free);
  return g_string_free(g, FALSE);
}

/** Read the scratch image back through the real history reader.

    `bp_out` receives a copy of the effective blend_params and `forms_out` a
    deep copy of the form tree, both owned by the caller, so that the dev can
    be torn down immediately. `op_out` receives the module name, which points
    into a static-lifetime string on the history item and is copied by the
    caller if it needs to outlive this. Returns FALSE if nothing masked came
    back. */
static gboolean _read_state(dt_develop_blend_params_t *bp_out,
                            GList **forms_out,
                            char *op_out,
                            const size_t op_len)
{
  dt_develop_t dev;
  dt_dev_init(&dev, FALSE);
  // dt_dev_init leaves dev->iop NULL and dt_dev_read_history_ext refuses to do
  // anything without it
  dev.iop = dt_iop_load_modules(&dev);
  // no_image: there is no raw file behind the scratch row, and the
  // default-module machinery this flag skips would add auto-applied modules
  // that have nothing to do with the mask
  dt_masks_scratch_claim_image(&dev, PERSIST_IMGID);
  dt_dev_read_history_ext(&dev, PERSIST_IMGID, TRUE);

  gboolean ok = FALSE;
  GList *last = g_list_last(dev.history);
  if(last)
  {
    const dt_dev_history_item_t *h = last->data;
    if(h->blend_params)
    {
      memcpy(bp_out, h->blend_params, sizeof(dt_develop_blend_params_t));
      g_strlcpy(op_out, h->op_name, op_len);
      *forms_out = dt_masks_dup_forms_deep(dev.forms, NULL);
      ok = TRUE;
    }
  }

  dt_dev_cleanup(&dev);
  return ok;
}

/** One step of the persisted arm: read, apply the step, write back.

    The write is not a bare dt_dev_write_history_ext(). _dev_write_history_item()
    persists a history item's OWN forms snapshot, and a freshly-read stack has
    none -- only _dev_add_history_item_ext() fills it, by deep-copying
    dev->forms. So the sequence has to be the real one: pop the stack onto the
    modules, change the mask, then add a masks history item, which is what
    snapshots dev->forms. Calling the writer straight after a read instead
    wipes masks_history and stores nothing, which reads from the outside
    exactly like the change being lost -- roundtrip.c documents the same trap.

    Returns FALSE if the image came back with no flexi mask to edit. */
static gboolean _read_poke_write(const step_t *st, const int group_index)
{
  dt_develop_t dev;
  dt_dev_init(&dev, FALSE);
  dev.iop = dt_iop_load_modules(&dev);
  dt_masks_scratch_claim_image(&dev, PERSIST_IMGID);
  dt_dev_read_history_ext(&dev, PERSIST_IMGID, TRUE);
  dt_dev_pop_history_items_ext(&dev, dev.history_end);

  gboolean ok = FALSE;
  for(GList *m = dev.iop; m; m = g_list_next(m))
  {
    dt_iop_module_t *mod = m->data;
    if(!_target_group(&dev, mod->blend_params)) continue;
    dt_masks_form_t *grp = _group_at(&dev, mod->blend_params, group_index);
    if(!grp) break;

    {
      _apply_step(&dev, grp, st);
      // dt_dev_add_masks_history_item_ext, NOT the plain variant: only the
      // masks one passes include_masks = TRUE down to
      // _dev_add_history_item_ext, and only that snapshots dev->forms into the
      // item. The plain variant appends an item with forms == NULL, which
      // stores no masks at all.
      dt_dev_add_masks_history_item_ext(&dev, mod, FALSE, TRUE);
      ok = TRUE;
    }
    break;
  }

  if(ok) dt_dev_write_history_ext(&dev, PERSIST_IMGID);
  dt_dev_cleanup(&dev);
  return ok;
}

/** Put the scratch image back to its just-migrated state: seed the classic
    history again and open it once, which is what runs migration.

    `bp_out` and `forms_out` (either may be NULL) receive the migrated state as
    it stands IN MEMORY, before anything has been read back. That distinction
    is the whole point of this check. The in-memory state is what the user sees
    in the darkroom on the first open; whether the database now holds enough to
    reconstruct it is precisely the open question, so an arm that wants "the
    first open" must take it from here and not from a re-read. Taking it from a
    re-read instead is how the first version of this file managed to pass with
    the half-persisted migration reinstated: both arms were then sitting on the
    far side of the loss, agreeing with each other about the wrong mask.

    Returns FALSE if the row could not be seeded. */
static gboolean _reset_to_migrated(const char *op, const int mp, const int bv,
                                   const int w, const int h,
                                   const dt_develop_blend_params_t *bp,
                                   GList *forms,
                                   dt_develop_blend_params_t *bp_out,
                                   GList **forms_out)
{
  dt_masks_scratch_wipe_history(PERSIST_IMGID);
  dt_masks_scratch_seed_image(PERSIST_IMGID, w, h);
  // the iop-order entry must exist before the history row referencing it, or
  // dt_dev_read_history_ext() drops the row without a word (scratch_image.h)
  if(op) dt_masks_scratch_seed_iop_order(PERSIST_IMGID, op, mp);
  if(!op || !dt_masks_scratch_seed_history(PERSIST_IMGID, 0, op, mp, bv, bp, forms))
    return FALSE;

  // Opening the image is what migrates it, and -- since the half-persisted
  // migration fix -- what stores the result. Deliberately a plain read with no
  // edit of our own: the baseline both arms start from has to be what a user
  // gets by opening the image and nothing more.
  dt_develop_t dev;
  dt_dev_init(&dev, FALSE);
  dev.iop = dt_iop_load_modules(&dev);
  dt_masks_scratch_claim_image(&dev, PERSIST_IMGID);
  dt_dev_read_history_ext(&dev, PERSIST_IMGID, TRUE);

  gboolean ok = TRUE;
  if(bp_out || forms_out)
  {
    ok = FALSE;
    GList *last = g_list_last(dev.history);
    if(last)
    {
      const dt_dev_history_item_t *hi = last->data;
      if(hi->blend_params)
      {
        if(bp_out) memcpy(bp_out, hi->blend_params, sizeof(dt_develop_blend_params_t));
        if(forms_out) *forms_out = dt_masks_dup_forms_deep(dev.forms, NULL);
        ok = TRUE;
      }
    }
  }

  dt_dev_cleanup(&dev);
  return ok;
}

// ---------------------------------------------------------------------------
// rendering a stored state
// ---------------------------------------------------------------------------

/** Point an already-initialised replay at a different mask, taking ownership
    of `forms`. _render_mask() re-reads r->dev.forms into the pipe every time,
    so nothing else needs updating. */
static void _install_state(replay_t *r,
                           const dt_develop_blend_params_t *bp,
                           GList *forms)
{
  g_list_free_full(r->dev.forms, (GDestroyNotify)dt_masks_free_form);
  r->dev.forms = forms;
  // into the module's own allocation: it owns that buffer and frees it on
  // cleanup, so repointing it would double-free
  memcpy(r->module.blend_params, bp, sizeof(dt_develop_blend_params_t));
}

// ---------------------------------------------------------------------------
// one edit
// ---------------------------------------------------------------------------

typedef enum
{
  PERSIST_OK = 0,
  PERSIST_DIFFERENT,
  PERSIST_SKIPPED,
  PERSIST_ERROR
} persist_result_t;

typedef struct
{
  persist_result_t result;
  const char *skip_reason;
  int compared;
  int disagreed;
  int live;             // sequences that actually changed the mask
  int worst_seq;        // index into _sequences[], or -1
  double worst_diff;

  /* The classic fold, over what came back out of the database.

     --verify-masks asks the same question of the forms as migration left them
     in *memory* (see edit_report_t.restore_* there, and the argument for why
     the markers are render-neutral for classic). This asks it of the forms as
     *storage* returned them, which is the half that check cannot reach: it
     proves not only that the markers are safe for a classic reader but that
     they survive the write in a shape a classic reader can still make sense
     of. Both readers are real -- a migration that fails closed after
     _queue_group_split() has already marked the tree, and an older darktable
     opening the same masks_history rows.

     `db_marked` says whether the round trip actually brought markers back,
     i.e. whether the comparison is evidence or a tautology. */
  gboolean db_ran;
  gboolean db_marked;
  double db_max_diff;

  // how many groups the module renders through: 1 for a flat mask, more when
  // the top group has nested ones. Reported so a run says how much of the
  // nested surface it actually reached.
  int groups;
} persist_report_t;

typedef struct
{
  int compared;
  int disagreed;
  int live;
} seq_tally_t;

static void _persist_edit(JsonObject *edit,
                          persist_report_t *rep,
                          seq_tally_t *tally)
{
  memset(rep, 0, sizeof(*rep));
  rep->result = PERSIST_SKIPPED;
  rep->worst_seq = -1;

  JsonObject *bo = json_object_get_object_member(edit, "blend");
  if(!bo) { rep->skip_reason = "no blend object"; return; }

  dt_develop_blend_params_t classic_bp;
  dt_masks_harvest_read_blend_params(bo, &classic_bp);

  // an already-flexi edit was never migrated; the question here is about a
  // migrated mask being built on, so there is nothing to ask about it
  if(classic_bp.mask_mode & DEVELOP_MASK_FLEXI)
  {
    rep->skip_reason = "already flexi";
    return;
  }

  JsonObject *img = json_object_get_object_member(edit, "image");
  const int full_w = (int)_obj_int(img, "width", 0);
  const int full_h = (int)_obj_int(img, "height", 0);
  if(full_w <= 0 || full_h <= 0) { rep->skip_reason = "no image dimensions"; return; }

  int w = full_w, h = full_h;
  if(w > VERIFY_MAX_EDGE || h > VERIFY_MAX_EDGE)
  {
    const double s = (double)VERIFY_MAX_EDGE / (double)MAX(w, h);
    w = MAX(8, (int)(w * s));
    h = MAX(8, (int)(h * s));
  }

  JsonArray *fa = json_object_has_member(edit, "forms")
    ? json_object_get_array_member(edit, "forms") : NULL;
  GList *classic_forms = fa ? dt_masks_harvest_read_forms(fa) : NULL;
  if(fa && json_array_get_length(fa) > 0 && !classic_forms)
  {
    rep->skip_reason = "forms could not be reconstructed";
    return;
  }

  const char *op = _obj_str(edit, "operation", NULL);
  const int mp = (int)_obj_int(edit, "multi_priority", 0);
  const int bv = (int)_obj_int(edit, "blendop_version", 14);

  if(!_reset_to_migrated(op, mp, bv, full_w, full_h, &classic_bp, classic_forms,
                         NULL, NULL))
  {
    g_list_free_full(classic_forms, (GDestroyNotify)dt_masks_free_form);
    rep->skip_reason = "history row could not be seeded";
    return;
  }

  // the replay renders; it is built once and repointed at each state below.
  // Seeded with the harvested classic params so that a raster edit gets its
  // synthetic source attached from the fields it names -- no poke here touches
  // those, so the source stays right for every render.
  replay_t r;
  const char *init_err =
    _replay_init(&r, op, &classic_bp,
                 dt_masks_dup_forms_deep(classic_forms, NULL),
                 full_w, full_h, w, h);
  if(init_err)
  {
    g_list_free_full(classic_forms, (GDestroyNotify)dt_masks_free_form);
    rep->result = PERSIST_ERROR;
    rep->skip_reason = init_err;
    return;
  }

  const size_t npix = (size_t)w * h;

  /* --- the classic fold, over what storage returned ------------------------

     Render the edit exactly as authored -- classic blend_params, classic forms,
     nothing migrated -- and then again with the forms the database handed back
     after migration wrote its markers into them, still through the classic
     blend_params. If storage carries the markers in a shape a classic reader
     can still make sense of, those two masks are the same.

     Done before the sequence sweep and on its own renders, so it is answered
     even for an edit the sweep later has to skip. */
  {
    _install_state(&r, &classic_bp, dt_masks_dup_forms_deep(classic_forms, NULL));
    float *classic_authored = _render_mask(&r, NULL);
    gchar *authored_digest = _marker_digest(r.dev.forms);

    dt_develop_blend_params_t stored_bp;
    GList *stored_forms = NULL;
    char obuf[128] = { 0 };
    if(classic_authored && _read_state(&stored_bp, &stored_forms, obuf, sizeof(obuf)))
    {
      gchar *stored_digest = _marker_digest(stored_forms);
      rep->db_marked = strcmp(authored_digest, stored_digest) != 0;
      g_free(stored_digest);

      /* the stored forms, read through the *classic* params: mask_id is the
         same group either way -- a drawn-only migration reuses it verbatim,
         and a drawn+parametric one leaves the original drawn group in place
         under its own id and points the new top group elsewhere. */
      _install_state(&r, &classic_bp, stored_forms);
      float *classic_stored = _render_mask(&r, NULL);
      if(classic_stored)
      {
        rep->db_ran = TRUE;
        rep->db_max_diff = _max_abs_diff(classic_authored, classic_stored, npix);
        dt_free_align(classic_stored);
      }
    }
    g_free(authored_digest);
    dt_free_align(classic_authored);
  }

  // the baseline: the migrated mask as the first open leaves it in memory,
  // with nothing poked. Used only to tell a sequence that genuinely changed
  // something from one that was inert on this edit, so that a pass is not
  // reported as evidence when both arms rendered the same untouched mask.
  dt_develop_blend_params_t bp;
  GList *forms = NULL;
  char opbuf[128] = { 0 };
  if(!_reset_to_migrated(op, mp, bv, full_w, full_h, &classic_bp, classic_forms,
                         &bp, &forms))
  {
    rep->result = PERSIST_ERROR;
    rep->skip_reason = "the seeded image came back with no history";
    goto out;
  }
  _install_state(&r, &bp, forms);

  // a mask this check cannot poke: no group means no run and no member
  if(!_target_group(&r.dev, &bp))
  {
    rep->skip_reason = "no group to edit";
    goto out;
  }

  float *base = _render_mask(&r, NULL);
  if(!base)
  {
    rep->result = PERSIST_ERROR;
    rep->skip_reason = "the blend published no mask";
    goto out;
  }

  rep->result = PERSIST_OK;

  // how many groups the module renders through, counted once on the un-poked
  // tree: no poke here creates or destroys a group, so the count is stable and
  // both arms resolve the same index to the same group
  GList *g0 = _all_groups(&r.dev, &bp);
  const int ngroups = (int)g_list_length(g0);
  g_list_free(g0);
  rep->groups = ngroups;

  for(int gi = 0; gi < ngroups; gi++)
  for(int q = 0; q < SEQ_N; q++)
  {
    const seq_t *seq = &_sequences[q];

    /* ---- arm A is the session that never closes: seed the classic edit,
       open it once, and from then on only change things. Reseeding per
       sequence also undoes what the previous sequence's arm B wrote, so each
       sequence starts where it says it does. */
    if(!_reset_to_migrated(op, mp, bv, full_w, full_h, &classic_bp, classic_forms,
                           &bp, &forms))
      continue;
    _install_state(&r, &bp, forms);
    dt_masks_form_t *grp = _group_at(&r.dev, &bp, gi);
    if(!grp) continue;

    for(int s = 0; s < seq->n; s++)
      _apply_step(&r.dev, grp, &seq->step[s]);
    float *a = _render_mask(&r, NULL);

    /* ---- arm B is the same edits with the image closed and reopened between
       every one of them. Back to the same starting point first. */
    gboolean b_ok = _reset_to_migrated(op, mp, bv, full_w, full_h,
                                       &classic_bp, classic_forms, NULL, NULL);
    for(int s = 0; b_ok && s < seq->n; s++)
      b_ok = _read_poke_write(&seq->step[s], gi);

    float *b = NULL;
    if(b_ok && _read_state(&bp, &forms, opbuf, sizeof(opbuf)))
    {
      _install_state(&r, &bp, forms);
      b = _render_mask(&r, NULL);
    }

    if(a && b)
    {
      rep->compared++;
      tally[q].compared++;

      const gboolean live = _max_abs_diff(a, base, npix) > PERSIST_EPS;
      if(live) { rep->live++; tally[q].live++; }

      const double d = _max_abs_diff(a, b, npix);
      if(d > PERSIST_EPS)
      {
        rep->disagreed++;
        tally[q].disagreed++;
        rep->result = PERSIST_DIFFERENT;
        if(d > rep->worst_diff) { rep->worst_diff = d; rep->worst_seq = q; }
      }
    }
    else if(!a || !b_ok)
    {
      // an arm that could not be built at all is an error, not a pass: a
      // missing render compares equal to nothing and would otherwise vanish
      rep->result = PERSIST_ERROR;
      rep->skip_reason = "an arm could not be rendered";
    }

    dt_free_align(a);
    dt_free_align(b);
  }

  dt_free_align(base);

out:
  _replay_cleanup(&r);
  g_list_free_full(classic_forms, (GDestroyNotify)dt_masks_free_form);
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------

static const char *_result_name(const persist_result_t r)
{
  switch(r)
  {
    case PERSIST_OK:        return "identical";
    case PERSIST_DIFFERENT: return "different";
    case PERSIST_SKIPPED:   return "skipped";
    default:                return "error";
  }
}

/* See roundtrip.c: a plain block rather than do{...}while(0), because
   `continue` would bind to the do-while and fall straight through into the
   code the skip exists to avoid. Never follow a call with an `else`. */
#define PERSIST_SKIP(why)                                                       \
  {                                                                             \
    skipped++;                                                                  \
    if(rf)                                                                      \
    {                                                                           \
      fprintf(rf, "%s\n    {\"index\": %u, \"result\": \"skipped\","             \
                  " \"reason\": \"%s\"}", first_report ? "" : ",", i, (why));   \
      first_report = FALSE;                                                     \
    }                                                                           \
    continue;                                                                   \
  }

gboolean dt_masks_persist_harvest_section(const char *json_path, FILE *rf)
{
  setvbuf(stdout, NULL, _IOLBF, 0);

#ifdef _OPENMP
  /* Single-threaded for the same reason as verify.c and postedit.c: a
     reduction whose float addition order depends on thread scheduling makes
     the last bits of the mask move between runs, and this compares at 1e-6.

     Not optional here, and it was missing at first. Inside --check-masks the
     sections ahead of this one had already set it, so the figures were stable
     and the omission invisible; run on its own, the live count wandered
     between 353 and 370 over three runs of identical input. The verdict
     (0 disagreed) was never affected -- the two arms run the same code over
     the same data and stay bit-identical -- but the liveness count is what
     says the sweep was not vacuous, so an unstable one makes the pass
     unreadable. */
  omp_set_num_threads(1);
#endif

  GError *err = NULL;
  JsonParser *parser = dt_masks_harvest_load(json_path, &err);
  if(!parser)
  {
    fprintf(stderr, "[persist] cannot read %s: %s\n",
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
    fprintf(stderr, "[persist] %s has no \"edits\" array\n", json_path);
    g_object_unref(parser);
    return FALSE;
  }

  const guint n = json_array_get_length(edits);
  printf("[persist] saving and reopening between edits, over %u harvested"
         " edits from %s\n", n, json_path);

  if(rf) fprintf(rf, "\n  \"source\": \"%s\",\n  \"edits\": [", json_path);
  gboolean first_report = TRUE;

  seq_tally_t *tally = calloc((size_t)SEQ_N, sizeof(seq_tally_t));
  if(!tally)
  {
    g_object_unref(parser);
    return FALSE;
  }

  /* Exact repeats reuse the first occurrence's verdict rather than being
     replayed again -- see dt_masks_harvest_edit_key(). Every occurrence is
     still counted and reported. */
  GHashTable *seen = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, g_free);
  // `distinct` counts every edit actually replayed, skips included; `swept`
  // only the ones that got as far as a comparison. Printing `total - distinct`
  // as the repeat count mixed the two and went negative as soon as a corpus
  // skipped more than it swept (zisoft: "11 (229 distinct, -218 repeats)").
  int distinct = 0, swept_distinct = 0;

  int total = 0, identical = 0, different = 0, skipped = 0, errors = 0;
  int compared = 0, disagreed = 0, live = 0, vacuous = 0;
  int db_compared = 0, db_marked = 0, db_different = 0;
  int nested_edits = 0, groups_swept = 0;
  double db_worst = 0.0;
  int db_worst_index = -1;

  for(guint i = 0; i < n; i++)
  {
    JsonObject *edit = json_array_get_object_element(edits, i);
    if(!edit) continue;

    gchar *key = dt_masks_harvest_edit_key(edit);
    const persist_report_t *cached = key ? g_hash_table_lookup(seen, key) : NULL;
    if(cached)
    {
      g_free(key);
      switch(cached->result)
      {
        case PERSIST_OK:        total++; identical++; break;
        case PERSIST_DIFFERENT: total++; different++; break;
        case PERSIST_SKIPPED:   skipped++;            break;
        default:                total++; errors++;    break;
      }
      compared += cached->compared;
      disagreed += cached->disagreed;
      live += cached->live;
      if(cached->compared == 0 && cached->result == PERSIST_OK) vacuous++;
      if(cached->db_ran)
      {
        db_compared++;
        if(cached->db_marked) db_marked++;
        if(cached->db_max_diff > PERSIST_EPS) db_different++;
      }
      if(rf)
      {
        fprintf(rf, "%s\n    {\"index\": %u, \"result\": \"%s\","
                    " \"repeat\": true}",
                first_report ? "" : ",", i, _result_name(cached->result));
        first_report = FALSE;
      }
      continue;
    }

    persist_report_t rep;
    _persist_edit(edit, &rep, tally);
    distinct++;

    if(rep.result == PERSIST_SKIPPED)
    {
      if(key)
      {
        persist_report_t *store = malloc(sizeof(persist_report_t));
        if(store) { *store = rep; g_hash_table_insert(seen, key, store); }
        else g_free(key);
      }
      PERSIST_SKIP(rep.skip_reason ? rep.skip_reason : "unspecified");
    }

    total++;
    swept_distinct++;
    groups_swept += rep.groups;
    if(rep.groups > 1) nested_edits++;
    compared += rep.compared;
    disagreed += rep.disagreed;
    live += rep.live;

    /* An edit that produced no comparison at all is reported, not counted as
       a pass. It cannot have failed, which is exactly the problem: a silent
       zero here would be indistinguishable from 24 sequences agreeing. */
    if(rep.compared == 0 && rep.result != PERSIST_ERROR) vacuous++;

    if(rep.db_ran)
    {
      db_compared++;
      if(rep.db_marked) db_marked++;
      if(rep.db_max_diff > PERSIST_EPS)
      {
        db_different++;
        printf("[persist] CLASSIC CHANGED at edit %u (%s): the stored forms"
               " render differently through classic blend params, by %.6f\n",
               i, _obj_str(edit, "operation", "?"), rep.db_max_diff);
        if(rep.db_max_diff > db_worst)
        {
          db_worst = rep.db_max_diff;
          db_worst_index = (int)i;
        }
      }
    }

    if(rep.result == PERSIST_OK) identical++;
    else if(rep.result == PERSIST_DIFFERENT)
    {
      different++;
      const seq_t *worst = rep.worst_seq >= 0 ? &_sequences[rep.worst_seq] : NULL;
      printf("[persist] DIFFERENT at edit %u (%s): %d/%d sequences disagree,"
             " worst '%s' by %.6f -- %s\n",
             i, _obj_str(edit, "operation", "?"), rep.disagreed, rep.compared,
             worst ? worst->name : "?", rep.worst_diff,
             worst ? worst->seam : "?");
    }
    else errors++;

    if(rf)
    {
      const seq_t *worst = rep.worst_seq >= 0 ? &_sequences[rep.worst_seq] : NULL;
      fprintf(rf, "%s\n    {\"index\": %u, \"operation\": \"%s\","
                  " \"result\": \"%s\", \"repeat\": false,"
                  " \"compared\": %d, \"disagreed\": %d, \"live\": %d",
              first_report ? "" : ",", i, _obj_str(edit, "operation", "?"),
              _result_name(rep.result), rep.compared, rep.disagreed, rep.live);
      if(rep.db_ran)
        fprintf(rf, ", \"classic_over_stored_marked\": %s,"
                    " \"classic_over_stored_diff\": %.9g",
                rep.db_marked ? "true" : "false", rep.db_max_diff);
      if(worst)
        fprintf(rf, ", \"worst_sequence\": \"%s\", \"seam\": \"%s\","
                    " \"worst_diff\": %.9f",
                worst->name, worst->seam, rep.worst_diff);
      if(rep.result == PERSIST_ERROR && rep.skip_reason)
        fprintf(rf, ", \"error\": \"%s\"", rep.skip_reason);
      fputc('}', rf);
      first_report = FALSE;
    }

    if(key)
    {
      persist_report_t *store = malloc(sizeof(persist_report_t));
      if(store) { *store = rep; g_hash_table_insert(seen, key, store); }
      else g_free(key);
    }

    if((i + 1) % 50 == 0) printf("[persist]   %u/%u ...\n", i + 1, n);
  }

  g_object_unref(parser);
  g_hash_table_destroy(seen);

  /* A stored tree the classic fold makes a different mask of is a failure of
     the run, not a footnote: the markers are persisted now, so a fail-closed
     migration or an older darktable reads exactly that. */
  const gboolean passed = different == 0 && errors == 0 && vacuous == 0
                          && db_different == 0;

  if(rf)
  {
    fputs("\n  ],\n  \"summary\": {\n", rf);
    fprintf(rf, "    \"passed\": %s,\n", passed ? "true" : "false");
    fprintf(rf, "    \"harvested\": %u,\n", n);
    fprintf(rf, "    \"swept\": %d,\n", total);
    fprintf(rf, "    \"distinct_edits\": %d,\n", distinct);
    fprintf(rf, "    \"distinct_swept\": %d,\n", swept_distinct);
    fprintf(rf, "    \"identical\": %d,\n", identical);
    fprintf(rf, "    \"different\": %d,\n", different);
    fprintf(rf, "    \"errors\": %d,\n", errors);
    fprintf(rf, "    \"skipped\": %d,\n", skipped);
    fprintf(rf, "    \"sequences_compared\": %d,\n", compared);
    fprintf(rf, "    \"sequences_disagreed\": %d,\n", disagreed);
    fprintf(rf, "    \"sequences_live\": %d,\n", live);
    fprintf(rf, "    \"swept_nothing\": %d,\n", vacuous);
    fprintf(rf, "    \"groups_swept\": %d,\n", groups_swept);
    fprintf(rf, "    \"edits_with_nested_group\": %d,\n", nested_edits);
    fprintf(rf, "    \"classic_over_stored_compared\": %d,\n", db_compared);
    fprintf(rf, "    \"classic_over_stored_marked\": %d,\n", db_marked);
    fprintf(rf, "    \"classic_over_stored_different\": %d,\n", db_different);
    fprintf(rf, "    \"classic_over_stored_worst_diff\": %.9g,\n", db_worst);
    fputs("    \"per_sequence\": [", rf);
    for(int q = 0; q < SEQ_N; q++)
      fprintf(rf, "%s\n      {\"name\": \"%s\", \"seam\": \"%s\","
                  " \"compared\": %d, \"disagreed\": %d, \"live\": %d}",
              q ? "," : "", _sequences[q].name, _sequences[q].seam,
              tally[q].compared, tally[q].disagreed, tally[q].live);
    fputs("\n    ]\n  }", rf);
  }

  printf("[persist]\n");
  printf("[persist] edits             : %d swept  (%d distinct swept,"
         " %d reused as repeats; %d edits replayed in all)\n",
         total, swept_distinct, total - swept_distinct, distinct);
  printf("[persist]   identical       : %d\n", identical);
  printf("[persist]   DIFFERENT       : %d\n", different);
  printf("[persist]   skipped         : %d\n", skipped);
  printf("[persist]   errors          : %d\n", errors);
  printf("[persist]   swept NOTHING   : %d  (no sequence could be compared;"
         " would pass vacuously)\n", vacuous);
  printf("[persist]\n");
  printf("[persist] the classic fold, over the forms storage returned:\n");
  printf("[persist]   re-rendered     : %d\n", db_compared);
  printf("[persist]   of those, the round trip brought markers back : %d"
         "  (the rest prove nothing)\n", db_marked);
  printf("[persist]   CLASSIC CHANGED : %d", db_different);
  if(db_different) printf("   worst %.6f at edit %d", db_worst, db_worst_index);
  printf("\n");
  printf("[persist]\n");
  printf("[persist] groups swept       : %d  (%d edits had a nested group)\n",
         groups_swept, nested_edits);
  printf("[persist] sequences compared : %d\n", compared);
  printf("[persist]   disagreed        : %d\n", disagreed);
  printf("[persist]   changed the mask : %d  (the rest are legitimately inert"
         " on their edit)\n", live);
  printf("[persist]\n");
  printf("[persist] per sequence                          compared  disagreed  live\n");
  for(int q = 0; q < SEQ_N; q++)
    printf("[persist]   %-38s %8d %10d %5d\n", _sequences[q].name,
           tally[q].compared, tally[q].disagreed, tally[q].live);

  free(tally);
  return passed;
}

#undef PERSIST_SKIP

gboolean dt_masks_persist_harvest(const char *json_path, const char *report_path)
{
  FILE *rf = report_path ? g_fopen(report_path, "wb") : NULL;
  if(rf) fputs("{", rf);
  const gboolean ok = dt_masks_persist_harvest_section(json_path, rf);
  if(rf)
  {
    fputs("\n}\n", rf);
    fclose(rf);
    printf("[persist] per-edit report written to %s\n", report_path);
  }
  return ok;
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
