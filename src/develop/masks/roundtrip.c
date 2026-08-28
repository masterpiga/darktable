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

#include "develop/masks/roundtrip.h"

#include "common/darktable.h"
#include "common/debug.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/masks.h"
#include "develop/masks/harvest_read.h"
#include "develop/masks/scratch_image.h"

#include <json-glib/json-glib.h>
#include <sqlite3.h>
#include <stdio.h>

// A single scratch image reused for every edit, wiped between them. Real
// darktable ids start at 1 and this runs against a throwaway database (the
// caller is expected to pass --library :memory:), so there is nothing to
// collide with.
#define ROUNDTRIP_IMGID 1

// ---------------------------------------------------------------------------
// snapshotting: everything the mask depends on, as comparable text
// ---------------------------------------------------------------------------

static gint _form_by_id(gconstpointer a, gconstpointer b)
{
  return ((const dt_masks_form_t *)a)->formid - ((const dt_masks_form_t *)b)->formid;
}

/** Render the mask-relevant state of a loaded dev as a canonical string.

    Text rather than a struct comparison so that a mismatch names the field
    that moved: this test exists to catch persistence bugs, and "group_start
    0 vs 1 on member 3" is an answer, where "the snapshots differ" is another
    investigation.

    Forms are emitted in formid order, not list order. The list order out of
    the database is not part of the mask's meaning (members are ordered inside
    their group's point list, which *is* emitted in order), and sorting keeps
    an incidental reordering from being reported as a round-trip failure. */
static gchar *_snapshot(dt_develop_t *dev)
{
  GString *s = g_string_new(NULL);

  // dev->history, not dev->iop. dt_dev_read_history_ext() fills the history
  // stack; a module's own blend_params are only updated when that stack is
  // *popped* onto it, which needs pipes this test has no reason to build.
  // Snapshotting dev->iop instead produced an empty module list for every
  // edit -- and two empty lists compare equal, so the whole comparison passed
  // vacuously. The history item is also the thing dt_dev_write_history_ext()
  // actually writes back, which makes it the right object to compare anyway.
  // Only the LAST history item, not every one. Simulating the user's edit
  // appends an item, so comparing whole stacks would report that append as a
  // round-trip difference -- and the appended item is the point, not an
  // artifact. The last item is the effective state, which is what a reload has
  // to reproduce.
  GList *last = g_list_last(dev->history);
  for(GList *m = last; m; m = NULL)
  {
    const dt_dev_history_item_t *h = m->data;
    const dt_develop_blend_params_t *bp = h->blend_params;
    if(!bp || bp->mask_mode == DEVELOP_MASK_DISABLED) continue;
    g_string_append_printf(s,
      "module %s.%d enabled=%d mask_mode=%u mask_id=%d combine=%u opacity=%.6f\n"
      "  blendif=%u feather=%.6f/%u blur=%.6f contrast=%.6f brightness=%.6f details=%.6f\n"
      "  raster=%s.%d id=%d inv=%d\n",
      h->op_name, h->multi_priority, h->enabled ? 1 : 0,
      bp->mask_mode, bp->mask_id, bp->mask_combine, bp->opacity,
      bp->blendif, bp->feathering_radius, bp->feathering_guide, bp->blur_radius,
      bp->contrast, bp->brightness, bp->details,
      bp->raster_mask_source, bp->raster_mask_instance,
      bp->raster_mask_id, bp->raster_mask_invert ? 1 : 0);
  }

  GList *sorted = g_list_copy(dev->forms);
  sorted = g_list_sort(sorted, _form_by_id);
  for(GList *f = sorted; f; f = g_list_next(f))
  {
    const dt_masks_form_t *form = f->data;
    g_string_append_printf(s, "form %d type=%d version=%d points=%u\n",
                           form->formid, form->type, form->version,
                           g_list_length(form->points));
    if(!(form->type & DT_MASKS_GROUP)) continue;
    for(GList *p = form->points; p; p = g_list_next(p))
    {
      const dt_masks_point_group_t *pt = p->data;
      g_string_append_printf(s,
        "  member %d parent=%d state=%d opacity=%.6f group_opacity=%.6f"
        " group_start=%d refine=%d\n",
        pt->formid, pt->parentid, pt->state, pt->opacity,
        pt->group_opacity, pt->group_start, pt->refinement.enabled);
    }
  }
  g_list_free(sorted);

  return g_string_free(s, FALSE);
}

/** Does every non-union member start its own run?

    Comparing load #1 with load #2 catches state that changes across a save,
    and nothing else -- a migration that produced the *same wrong* tree both
    times would pass it. That is a real blind spot, because the two loads take
    genuinely different paths (the first migrates from classic blend_params,
    the second finds them already flexi and no-ops), and the interesting
    failure is precisely one of them silently doing nothing.

    So each load is also checked against the invariant the fix exists to
    establish: the flexi fold applies a run's operator once per run, classic
    applies it once per member, and they agree only when every non-union member
    heads its own run (see _split_nonunion_runs in migrate_legacy.c). Checking
    it on both loads means a normalization that failed to run is caught even
    when it fails identically twice.

    Returns a description of the first violation, or NULL. */
static gchar *_check_group_runs(dt_develop_t *dev,
                                const dt_mask_id_t formid,
                                const int depth)
{
  if(depth > 8) return NULL;
  dt_masks_form_t *grp = dt_masks_get_from_id(dev, formid);
  if(!grp || !(grp->type & DT_MASKS_GROUP)) return NULL;

  const int non_union = DT_MASKS_STATE_INTERSECTION | DT_MASKS_STATE_DIFFERENCE
                      | DT_MASKS_STATE_SUM | DT_MASKS_STATE_EXCLUSION;

  for(GList *p = grp->points; p; p = g_list_next(p))
  {
    const dt_masks_point_group_t *pt = p->data;
    if((pt->state & non_union) && !pt->group_start)
      return g_strdup_printf("group %d member %d has a non-union operator"
                             " (state=%d) but does not start a run",
                             grp->formid, pt->formid, pt->state);
    gchar *deeper = _check_group_runs(dev, pt->formid, depth + 1);
    if(deeper) return deeper;
  }
  return NULL;
}

static gchar *_check_run_invariant(dt_develop_t *dev)
{
  // Walked from each module's own mask_id rather than over dev->forms at
  // large, and that distinction is not pedantry -- checking every form was the
  // first thing tried and it reported a violation on a group no module
  // references. dev->forms is per *image*, and every masks_history row is a
  // full cumulative snapshot (see migrate_legacy.c's header), so it routinely
  // carries groups belonging to other modules and groups orphaned by earlier
  // edits. Those never render, migration never touches them, and holding them
  // to a rendering invariant is meaningless.
  GList *last = g_list_last(dev->history);
  for(GList *m = last; m; m = NULL)
  {
    const dt_dev_history_item_t *h = m->data;
    const dt_develop_blend_params_t *bp = h->blend_params;
    if(!bp || !(bp->mask_mode & DEVELOP_MASK_FLEXI)) continue;
    if(!dt_is_valid_maskid(bp->mask_id)) continue;
    gchar *v = _check_group_runs(dev, bp->mask_id, 0);
    if(v) return v;
  }
  return NULL;
}

/** Load the scratch image through the real history reader and snapshot it.
    `written_back` optionally receives whether the write path ran. */
static gchar *_load_and_snapshot(const gboolean write_back, gchar **violation)
{
  dt_develop_t dev;
  dt_dev_init(&dev, FALSE);
  // dt_dev_init leaves dev->iop NULL and dt_dev_read_history_ext refuses to do
  // anything without it
  dev.iop = dt_iop_load_modules(&dev);

  // no_image = TRUE: there is no raw file behind the scratch row, and the
  // default-module machinery that flag skips would add auto-applied modules
  // that have nothing to do with what is being measured
  dt_dev_read_history_ext(&dev, ROUNDTRIP_IMGID, TRUE);

  gchar *snap = _snapshot(&dev);
  if(violation) *violation = _check_run_invariant(&dev);

  if(write_back)
  {
    // Simulate the user opening the image and touching the mask, because that
    // -- not the load -- is what writes.
    //
    // _dev_write_history_item() persists a history item's OWN forms snapshot
    // (dt_dev_history_item_t.forms), and a freshly-read stack has none: only
    // _dev_add_history_item_ext() fills it, by deep-copying the live
    // dev->forms. Calling dt_dev_write_history_ext() straight after a read
    // therefore wipes masks_history and writes nothing back, which is what the
    // first version of this test did -- it reported the normalization as lost
    // when really it had never been offered for saving.
    //
    // The real sequence is: pop the stack onto the modules (so they carry the
    // migrated blend_params), then add a history item, which snapshots
    // dev->forms *after* dt_masks_normalize_flexi_groups() has run on it.
    dt_dev_pop_history_items_ext(&dev, dev.history_end);

    for(GList *m = dev.iop; m; m = g_list_next(m))
    {
      dt_iop_module_t *mod = m->data;
      if(mod->blend_params && (mod->blend_params->mask_mode & DEVELOP_MASK_FLEXI))
      {
        // dt_dev_add_masks_history_item_ext, NOT dt_dev_add_history_item_ext:
        // only the masks variant passes include_masks=TRUE down to
        // _dev_add_history_item_ext, and only that snapshots dev->forms into
        // the item. The plain variant appends an item with forms == NULL,
        // which writes no masks at all -- indistinguishable, from the
        // outside, from the normalization being lost.
        dt_dev_add_masks_history_item_ext(&dev, mod, FALSE, TRUE);
        break;
      }
    }

    dt_dev_write_history_ext(&dev, ROUNDTRIP_IMGID);
  }

  dt_dev_cleanup(&dev);
  return snap;
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------

/** first line that differs, for the report */
static gchar *_first_difference(const char *a, const char *b)
{
  gchar **la = g_strsplit(a, "\n", -1);
  gchar **lb = g_strsplit(b, "\n", -1);
  gchar *out = NULL;
  for(int i = 0; la[i] || lb[i]; i++)
  {
    const char *x = la[i] ? la[i] : "(end)";
    const char *y = lb[i] ? lb[i] : "(end)";
    if(strcmp(x, y))
    {
      out = g_strdup_printf("line %d: after load '%s' | after reload '%s'", i + 1, x, y);
      break;
    }
    if(!la[i] || !lb[i]) break;
  }
  g_strfreev(la);
  g_strfreev(lb);
  return out ? out : g_strdup("(no line-level difference found)");
}

// a harvested edit this tool cannot round-trip. Recorded rather than silently
// dropped, so the report accounts for every index in the harvest.
#define ROUNDTRIP_SKIP(why)                                                     \
  do {                                                                          \
    skipped++;                                                                  \
    if(rf)                                                                      \
    {                                                                           \
      fprintf(rf, "%s\n    {\"index\": %u, \"result\": \"skipped\","             \
                  " \"reason\": \"%s\"}", first_report ? "" : ",", i, (why));   \
      first_report = FALSE;                                                     \
    }                                                                           \
    continue;                                                                   \
  } while(0)

gboolean dt_masks_roundtrip_harvest_section(const char *json_path, FILE *rf)
{
  setvbuf(stdout, NULL, _IOLBF, 0);

  GError *err = NULL;
  // accepts the .gz the contributor actually sent, as well as a plain file
  JsonParser *parser = dt_masks_harvest_load(json_path, &err);
  if(!parser)
  {
    fprintf(stderr, "[roundtrip] cannot read %s: %s\n",
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
    fprintf(stderr, "[roundtrip] %s has no \"edits\" array\n", json_path);
    g_object_unref(parser);
    return FALSE;
  }

  const guint n = json_array_get_length(edits);
  printf("[roundtrip] round-tripping %u harvested edits from %s\n", n, json_path);

  if(rf) fprintf(rf, "\n  \"source\": \"%s\",\n  \"edits\": [", json_path);
  gboolean first_report = TRUE;

  int total = 0, same = 0, differ = 0, skipped = 0, errors = 0;
  int no_module = 0, multi_instance = 0;

  for(guint i = 0; i < n; i++)
  {
    JsonObject *edit = json_array_get_object_element(edits, i);
    if(!edit) continue;

    JsonObject *bo = json_object_get_object_member(edit, "blend");
    if(!bo) ROUNDTRIP_SKIP("no blend object");

    dt_develop_blend_params_t bp;
    dt_masks_harvest_read_blend_params(bo, &bp);

    // an already-flexi edit has no migration to survive; the round trip would
    // be testing the plain history reader, which is not what this is for
    if(bp.mask_mode & DEVELOP_MASK_FLEXI) ROUNDTRIP_SKIP("already flexi");

    JsonObject *img = json_object_get_object_member(edit, "image");
    const int w = img && json_object_has_member(img, "width")
      ? (int)json_object_get_int_member(img, "width") : 0;
    const int h = img && json_object_has_member(img, "height")
      ? (int)json_object_get_int_member(img, "height") : 0;
    if(w <= 0 || h <= 0) ROUNDTRIP_SKIP("no image dimensions");

    JsonArray *fa = json_object_has_member(edit, "forms")
      ? json_object_get_array_member(edit, "forms") : NULL;
    GList *forms = fa ? dt_masks_harvest_read_forms(fa) : NULL;
    if(fa && json_array_get_length(fa) > 0 && !forms)
      ROUNDTRIP_SKIP("forms could not be reconstructed");

    const char *op = json_object_has_member(edit, "operation")
      ? json_object_get_string_member(edit, "operation") : NULL;
    const int mp = json_object_has_member(edit, "multi_priority")
      ? (int)json_object_get_int_member(edit, "multi_priority") : 0;
    const int bv = json_object_has_member(edit, "blendop_version")
      ? (int)json_object_get_int_member(edit, "blendop_version") : 14;

    dt_masks_scratch_wipe_history(ROUNDTRIP_IMGID);
    dt_masks_scratch_seed_image(ROUNDTRIP_IMGID, w, h);
    // the iop-order entry must exist before the history row referencing it,
    // or dt_dev_read_history_ext() drops the row (see scratch_image.h)
    if(op) dt_masks_scratch_seed_iop_order(ROUNDTRIP_IMGID, op, mp);
    const gboolean seeded =
      op && dt_masks_scratch_seed_history(ROUNDTRIP_IMGID, 0, op, mp, bv, &bp, forms);
    g_list_free_full(forms, (GDestroyNotify)dt_masks_free_form);

    if(!seeded) ROUNDTRIP_SKIP("history row could not be seeded");

    total++;

    // load #1: the real migration runs here (the stored blendop_version is
    // classic), then writes everything back
    gchar *v1 = NULL, *v2 = NULL;
    gchar *snap1 = _load_and_snapshot(TRUE, &v1);
    // load #2: blendop_params are flexi now, so migration no-ops and the
    // stored forms have to carry what load #1 derived in memory
    gchar *snap2 = _load_and_snapshot(FALSE, &v2);

    if(mp > 0) multi_instance++;

    /* A guard against passing vacuously, not a statistic.
       dt_dev_read_history_ext() drops a history row whose (operation,
       multi_priority) has no iop-order entry, without saying so -- and two
       snapshots of a dev with no modules in it compare equal no matter what
       migration did. That is precisely how multi-instance edits used to be
       "tested" here. If this count is ever non-zero the run proves nothing
       about those edits, so it is reported next to the pass count rather than
       left for someone to notice. */
    if(snap1 && !strstr(snap1, "module ")) no_module++;

    gchar *diff = NULL;
    if(!snap1 || !snap2)
    {
      errors++;
    }
    else if(v1 || v2)
    {
      differ++;
      diff = g_strdup_printf("run-boundary invariant violated after %s: %s",
                             v1 ? "load" : "reload", v1 ? v1 : v2);
      printf("[roundtrip] INVARIANT at edit %u (%s): %s\n", i, op, diff);
    }
    else if(!strcmp(snap1, snap2))
    {
      same++;
    }
    else
    {
      differ++;
      diff = _first_difference(snap1, snap2);
      printf("[roundtrip] DIFFERENT at edit %u (%s): %s\n", i, op, diff);
    }

    if(rf)
    {
      gchar *esc = diff ? g_strescape(diff, NULL) : NULL;
      fprintf(rf, "%s\n    {\"index\": %u, \"operation\": \"%s\", \"mask_mode\": %u,"
                  " \"result\": \"%s\"%s%s%s}",
              first_report ? "" : ",", i, op, bp.mask_mode,
              diff ? "different" : (snap1 && snap2 ? "same" : "error"),
              esc ? ", \"first_difference\": \"" : "", esc ? esc : "", esc ? "\"" : "");
      g_free(esc);
      first_report = FALSE;
    }

    g_free(v1);
    g_free(v2);
    g_free(diff);
    g_free(snap1);
    g_free(snap2);

    if((i + 1) % 250 == 0) printf("[roundtrip]   %u/%u ...\n", i + 1, n);
  }

  g_object_unref(parser);

  const gboolean passed = differ == 0 && errors == 0 && no_module == 0;

  // the report carries every figure the summary below prints, so it can be read
  // on its own without the terminal output of the run that produced it
  if(rf)
  {
    fputs("\n  ],\n  \"summary\": {\n", rf);
    fprintf(rf, "    \"passed\": %s,\n", passed ? "true" : "false");
    fprintf(rf, "    \"harvested\": %u,\n", n);
    fprintf(rf, "    \"round_tripped\": %d,\n", total);
    fprintf(rf, "    \"unchanged\": %d,\n", same);
    fprintf(rf, "    \"different\": %d,\n", differ);
    fprintf(rf, "    \"errors\": %d,\n", errors);
    fprintf(rf, "    \"skipped\": %d,\n", skipped);
    fprintf(rf, "    \"multi_instance\": %d,\n", multi_instance);
    fprintf(rf, "    \"loaded_with_no_module\": %d\n", no_module);
    fputs("  }", rf);
  }

  printf("[roundtrip]\n");
  printf("[roundtrip] round-tripped   : %d\n", total);
  printf("[roundtrip]   unchanged     : %d\n", same);
  printf("[roundtrip]   DIFFERENT     : %d\n", differ);
  printf("[roundtrip]   errors        : %d\n", errors);
  printf("[roundtrip]   skipped       : %d  (already flexi, or unreconstructable)\n",
         skipped);
  printf("[roundtrip]   of those, multi-instance (multi_priority > 0) : %d\n",
         multi_instance);
  printf("[roundtrip]   loaded with NO module at all (would pass vacuously) : %d\n",
         no_module);

  return passed;
}

#undef ROUNDTRIP_SKIP

gboolean dt_masks_roundtrip_harvest(const char *json_path, const char *report_path)
{
  FILE *rf = report_path ? g_fopen(report_path, "wb") : NULL;
  if(rf) fputs("{", rf);
  const gboolean ok = dt_masks_roundtrip_harvest_section(json_path, rf);
  if(rf)
  {
    fputs("\n}\n", rf);
    fclose(rf);
    printf("[roundtrip] per-edit report written to %s\n", report_path);
  }
  return ok;
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
