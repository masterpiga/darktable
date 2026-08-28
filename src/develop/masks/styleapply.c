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

#include "develop/masks/styleapply.h"

#include "common/darktable.h"
#include "common/debug.h"
#include "common/history.h"
#include "common/iop_order.h"
#include "common/styles.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/masks.h"
#include "develop/masks/harvest_read.h"
#include "develop/masks/scratch_image.h"

#include <json-glib/json-glib.h>
#include <stdio.h>

// Same reasoning as --roundtrip-masks: one scratch image, wiped between edits,
// only ever safe against `--library :memory:`.
#define STYLEAPPLY_IMGID 1

// how many items of one module a single style under test may carry: one for an
// ordinary edit, two for one the user kept as a second instance
#define STYLEAPPLY_MAX_ITEMS 2

// ---------------------------------------------------------------------------
// the host edit: what is already on the image when the style lands
// ---------------------------------------------------------------------------

typedef struct _host_t
{
  char operation[32];
  int blendop_version;
  dt_develop_blend_params_t bp;
  GList *forms;
  gboolean valid;
} _host_t;

/** Pick the host from the corpus: the first drawn-mask edit that actually
    carries forms.

    Drawn-mask on purpose. The host's job is to give the image a real
    masks_history snapshot and a real mask_id resolving into it *before* the
    style arrives, so that "did the style disturb what was already there" is a
    meaningful question. A parametric-only host would be migrated into a
    synthesized form too, which muddles which of the two synthesis paths a
    failure came from. */
static gboolean _pick_host(JsonArray *edits, _host_t *host)
{
  const guint n = json_array_get_length(edits);
  for(guint i = 0; i < n; i++)
  {
    JsonObject *edit = json_array_get_object_element(edits, i);
    if(!edit) continue;
    JsonObject *bo = json_object_get_object_member(edit, "blend");
    if(!bo) continue;

    dt_develop_blend_params_t bp;
    dt_masks_harvest_read_blend_params(bo, &bp);
    if(bp.mask_mode & DEVELOP_MASK_FLEXI) continue;
    if(!(bp.mask_mode & DEVELOP_MASK_MASK)) continue;
    if(bp.mask_mode & (DEVELOP_MASK_CONDITIONAL | DEVELOP_MASK_RASTER)) continue;

    JsonArray *fa = json_object_has_member(edit, "forms")
      ? json_object_get_array_member(edit, "forms") : NULL;
    if(!fa || json_array_get_length(fa) == 0) continue;
    GList *forms = dt_masks_harvest_read_forms(fa);
    if(!forms) continue;

    const char *op = json_object_has_member(edit, "operation")
      ? json_object_get_string_member(edit, "operation") : NULL;
    if(!op) { g_list_free_full(forms, (GDestroyNotify)dt_masks_free_form); continue; }

    g_strlcpy(host->operation, op, sizeof(host->operation));
    host->blendop_version = json_object_has_member(edit, "blendop_version")
      ? (int)json_object_get_int_member(edit, "blendop_version") : 14;
    host->bp = bp;
    host->forms = forms;
    host->valid = TRUE;
    return TRUE;
  }
  return FALSE;
}

// ---------------------------------------------------------------------------
// describing what a load found, so a failure names the thing that moved
// ---------------------------------------------------------------------------

/** The state of one module's mask, as text. `resolved` reports whether the
    mask_id actually finds a group with content in dev->forms -- which is the
    whole question for a style, since the immediate migration path writes its
    synthesized form to dev->forms only. */
static gchar *_describe_mask(dt_develop_t *dev,
                             const dt_develop_blend_params_t *bp,
                             gboolean *resolved)
{
  if(resolved) *resolved = FALSE;
  if(!bp) return g_strdup("(no blend params)");

  GString *s = g_string_new(NULL);
  g_string_append_printf(s, "mask_mode=%u mask_id=%d", bp->mask_mode, bp->mask_id);

  if(!dt_is_valid_maskid(bp->mask_id))
  {
    /* No mask_id is not automatically a failure -- two migration outcomes
       legitimately end with no form at all, and both are correct:

         - plain uniform flexi (ENABLED | FLEXI, mask_id cleared), the
           normalization dt_masks_migrate_classic_to_flexi() applies to a
           module whose classic mask_mode carried no mask bits;

         - plain uniform classic (ENABLED alone), which is what
           _migrate_parametric_only() writes for the DT_COND_CONSTANT and
           DT_COND_PASSTHROUGH branches: a parametric mask with no active
           channel is a constant, so it is migrated to a uniform blend (with
           opacity forced to 0 when that constant is transparent) rather than
           to a form that would compute the same constant per pixel.

       This distinction cost a false positive: requiring the FLEXI bit here
       reported all 46 parametric-only edits in the corpus as lost masks, when
       what they had actually done was migrate correctly. What makes a mask
       lost is a mask_mode that says a form is in play and a mask_id that does
       not find one -- checked below, not here. */
    const uint32_t needs_form =
      DEVELOP_MASK_MASK | DEVELOP_MASK_CONDITIONAL | DEVELOP_MASK_RASTER;
    if(resolved) *resolved = !(bp->mask_mode & needs_form);
    g_string_append(s, " (no form -- uniform blend)");
    return g_string_free(s, FALSE);
  }

  const dt_masks_form_t *grp = dt_masks_get_from_id(dev, bp->mask_id);
  if(!grp)
  {
    g_string_append(s, " -> DANGLING (no such form)");
    return g_string_free(s, FALSE);
  }

  g_string_append_printf(s, " -> form type=%d members=%u [",
                         grp->type, g_list_length(grp->points));
  for(GList *p = grp->points; p; p = g_list_next(p))
  {
    const dt_masks_point_group_t *pt = p->data;
    g_string_append_printf(s, "%s%d:state=%d,start=%d,op=%.4f",
                           p == grp->points ? "" : " ",
                           pt->formid, pt->state, pt->group_start, pt->opacity);
  }
  g_string_append_c(s, ']');

  if(resolved) *resolved = (grp->points != NULL);
  return g_string_free(s, FALSE);
}

/** Find the last history item for `operation` at `multi_priority`, or NULL.
    A negative `multi_priority` matches any instance. */
static const dt_dev_history_item_t *_history_item_for(dt_develop_t *dev,
                                                      const char *operation,
                                                      const int multi_priority)
{
  const dt_dev_history_item_t *found = NULL;
  for(GList *l = dev->history; l; l = g_list_next(l))
  {
    const dt_dev_history_item_t *h = l->data;
    if(!h || strcmp(h->op_name, operation)) continue;
    if(multi_priority >= 0 && h->multi_priority != multi_priority) continue;
    found = h;
  }
  return found;
}

// ---------------------------------------------------------------------------
// the three phases
// ---------------------------------------------------------------------------

/** Read the scratch image through the real history reader. Caller cleans up. */
static void _open_scratch_dev(dt_develop_t *dev)
{
  dt_dev_init(dev, FALSE);
  // dt_dev_init leaves dev->iop NULL and dt_dev_read_history_ext refuses to do
  // anything without it
  dev->iop = dt_iop_load_modules(dev);
  // no_image = TRUE: there is no raw file behind the scratch row, and the
  // default-module machinery that flag skips would add auto-applied modules
  // that have nothing to do with what is being measured
  dt_dev_read_history_ext(dev, STYLEAPPLY_IMGID, TRUE);
}

/** Phase 1: read the seeded classic rows and write them back as flexi, exactly
    as --roundtrip-masks does, so the image on disk is a normal migrated image
    before any style touches it. Reports the host's mask state, which is what
    the style must not damage. */
static void _settle(const _host_t *host, gchar **host_desc)
{
  dt_develop_t dev;
  _open_scratch_dev(&dev);

  // pop the stack onto the modules, then snapshot dev->forms into a history
  // item -- dt_dev_write_history_ext() persists the *item's* forms, and a
  // freshly-read stack has none (see roundtrip.c for the long version).
  // Every migrated module needs its own snapshot, not just the first: the
  // driver may have seeded a second module here.
  dt_dev_pop_history_items_ext(&dev, dev.history_end);
  for(GList *m = dev.iop; m; m = g_list_next(m))
  {
    dt_iop_module_t *mod = m->data;
    if(mod->enabled && mod->blend_params
       && (mod->blend_params->mask_mode & DEVELOP_MASK_FLEXI))
      dt_dev_add_masks_history_item_ext(&dev, mod, FALSE, TRUE);
  }
  dt_dev_write_history_ext(&dev, STYLEAPPLY_IMGID);

  const dt_dev_history_item_t *h = _history_item_for(&dev, host->operation, 0);
  *host_desc = _describe_mask(&dev, h ? h->blend_params : NULL, NULL);

  dt_dev_cleanup(&dev);
}

/** Phase 2: apply one classic edit to the settled image as a style.

    This reproduces dt_styles_apply_style_item() (src/common/styles.c) rather
    than calling it, because calling it means building a dt_style_item_t and
    standing up the styles tables around it, none of which is what is under
    test. The two calls that matter are reproduced exactly, in order, with the
    same arguments the real function passes:

      - dt_develop_blend_legacy_params(), which is where flexi migration runs
        for a style -- it delegates to the _ext variant with history_num = -1,
        the immediate branch;

      - dt_history_merge_module_into_history() with dev_src == NULL, which is
        the argument styles pass and which decides whether the resulting
        history item gets a forms snapshot.

      - dt_ioppr_update_for_style_items(), which is what allocates the instances
        the style's items land on. This is deliberately not second-guessed: a
        style item's stored multi_priority is *not* the instance it ends up at,
        because the target image has its own instances to fit around, so
        darktable renumbers and derives the matching iop_order. Calling the real
        function is the only way to land where darktable would, and it is what
        creates the iop-order entry for a second instance -- without one the
        history row is written and then silently dropped on the next read.

    Everything else dt_styles_apply_style_item() does concerns module params,
    versions and the flip/spots special cases, none of which touches masks.

    `n_items` is how many items of this same module the style carries, which is
    the only way a style ever reaches a second instance. Note it is not the
    user-facing "append" mode: _styles_apply_to_image_ext() passes append=FALSE
    to both of the calls above unconditionally, so a style always *replaces*
    what is on the image; the instance count comes from the style itself, i.e.
    from an image that had the module twice when the style was captured. (This
    matters, and cost a wrong first attempt: forcing append=TRUE instead makes
    dt_ioppr_update_for_style_items() allocate instance 1 while
    dt_history_merge_module_into_history() still replaces instance 0, so the
    result lands somewhere the caller is not looking.)

    Returns FALSE if the style could not be applied at all; on success reports
    through `landed` which instances the items ended up at, since that is where
    the caller has to go looking for the results. */
static gboolean _apply_as_style(const char *operation,
                                const int blendop_version,
                                const dt_develop_blend_params_t *classic,
                                const int n_items,
                                int *landed)
{
  dt_develop_t dev;
  _open_scratch_dev(&dev);
  dt_dev_pop_history_items_ext(&dev, dev.history_end);

  dt_iop_module_t *mod_src = dt_iop_get_module_by_op_priority(dev.iop, operation, -1);
  if(!mod_src)
  {
    dt_dev_cleanup(&dev);
    return FALSE;
  }

  /* Resolve instances and iop-orders the way the real style path does. The
     multi_name is empty because the harvest records no user-authored text --
     that only matters in that it leaves _ioppr_update_for_entries()'s
     force-append branch alone. params_size must be non-zero or the entry is
     treated as an auto-init module and gets no iop-order at all. */
  dt_style_item_t si[STYLEAPPLY_MAX_ITEMS] = { { 0 } };
  GList *si_list = NULL;
  for(int k = 0; k < n_items; k++)
  {
    si[k].operation = (gchar *)operation;
    si[k].multi_name = (gchar *)"";
    si[k].multi_priority = k;
    si[k].params_size = mod_src->params_size;
    si_list = g_list_append(si_list, &si[k]);
  }
  dt_ioppr_update_for_style_items(&dev, si_list, FALSE);
  g_list_free(si_list);

  // modules_used is shared across the items, exactly as the loop in
  // _styles_apply_to_image_ext() shares it -- it is what stops the second item
  // from replacing the module the first one just claimed
  GList *modules_used = NULL;
  gboolean ok = TRUE;

  for(int k = 0; k < n_items; k++)
  {
    dt_iop_module_t *module = calloc(1, sizeof(dt_iop_module_t));
    if(!module) { ok = FALSE; break; }

    module->dev = &dev;
    if(dt_iop_load_module(module, mod_src->so, &dev))
    {
      free(module);
      ok = FALSE;
      break;
    }
    module->instance = mod_src->instance;
    module->multi_priority = si[k].multi_priority;
    module->iop_order = si[k].iop_order;
    module->enabled = TRUE;
    memcpy(module->params, module->default_params, module->params_size);
    if(landed) landed[k] = si[k].multi_priority;

    // the migration call. Note this is the same shape styles.c uses: the stored
    // version is classic, so the equality test there fails and this branch runs.
    if(blendop_version == dt_develop_blend_version())
      memcpy(module->blend_params, classic, sizeof(dt_develop_blend_params_t));
    else
      dt_develop_blend_legacy_params(module, classic, blendop_version,
                                     module->blend_params, dt_develop_blend_version(),
                                     sizeof(dt_develop_blend_params_t));

    dt_history_merge_module_into_history(&dev, NULL, module, &modules_used,
                                         FALSE, FALSE);

    dt_iop_cleanup_module(module);
    free(module);
  }
  g_list_free(modules_used);

  // and this is what styles.c does next: write history and forms to db
  if(ok) dt_dev_write_history_ext(&dev, STYLEAPPLY_IMGID);

  dt_dev_cleanup(&dev);
  return ok;
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------

/* A harvested edit this tool cannot apply as a style. Recorded rather than
   silently dropped, so the report accounts for every index in the harvest.

   A plain block, NOT the usual do{...}while(0) -- see the note on
   ROUNDTRIP_SKIP: `continue` binds to the nearest enclosing loop, and
   do/while(0) is one, so the idiom turns the skip into a fall-through. */
#define STYLEAPPLY_SKIP(why)                                                    \
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

gboolean dt_masks_styleapply_harvest_section(const char *json_path,
                                             FILE *rf,
                                             gboolean *ran)
{
  setvbuf(stdout, NULL, _IOLBF, 0);
  if(ran) *ran = TRUE;

  GError *err = NULL;
  // accepts the .gz the contributor actually sent, as well as a plain file
  JsonParser *parser = dt_masks_harvest_load(json_path, &err);
  if(!parser)
  {
    fprintf(stderr, "[styleapply] cannot read %s: %s\n",
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
    fprintf(stderr, "[styleapply] %s has no \"edits\" array\n", json_path);
    g_object_unref(parser);
    return FALSE;
  }

  _host_t host = { .valid = FALSE };
  if(!_pick_host(edits, &host))
  {
    /* Not a failure: this check needs a drawn-mask edit from the corpus to
       stand in for "a mask already on the image", and a library that happens to
       contain none simply has nothing to apply a style onto. Reporting it as a
       failed check would tell a contributor their masks are broken when what
       actually happened is that this question does not arise for them. */
    printf("[styleapply] %s has no drawn-mask edit to use as the host"
           " -- nothing to check\n", json_path);
    if(ran) *ran = FALSE;
    if(rf)
      fprintf(rf, "\n  \"source\": \"%s\",\n  \"edits\": [],\n"
                  "  \"summary\": {\n"
                  "    \"ran\": false,\n"
                  "    \"reason\": \"no drawn-mask edit in the corpus"
                  " to use as a host\"\n  }", json_path);
    g_object_unref(parser);
    return TRUE;
  }

  const guint n = json_array_get_length(edits);
  printf("[styleapply] applying %u harvested edits as styles onto a '%s' host\n",
         n, host.operation);

  if(rf) fprintf(rf, "\n  \"source\": \"%s\",\n  \"host\": \"%s\",\n  \"edits\": [",
                 json_path, host.operation);
  gboolean first_report = TRUE;

  int total = 0, ok_count = 0, dangling = 0, host_lost = 0, skipped = 0, errors = 0;
  int same_op = 0, drawn_in_style = 0, not_carried = 0;
  int multi_item = 0, landed_second = 0, no_module = 0;

  for(guint i = 0; i < n; i++)
  {
    JsonObject *edit = json_array_get_object_element(edits, i);
    if(!edit) continue;

    JsonObject *bo = json_object_get_object_member(edit, "blend");
    if(!bo) STYLEAPPLY_SKIP("no blend object");

    dt_develop_blend_params_t bp;
    dt_masks_harvest_read_blend_params(bo, &bp);
    // an already-flexi edit has no migration to survive
    if(bp.mask_mode & DEVELOP_MASK_FLEXI) STYLEAPPLY_SKIP("already flexi");

    const char *op = json_object_has_member(edit, "operation")
      ? json_object_get_string_member(edit, "operation") : NULL;
    if(!op) STYLEAPPLY_SKIP("no operation");

    JsonObject *img = json_object_get_object_member(edit, "image");
    const int w = img && json_object_has_member(img, "width")
      ? (int)json_object_get_int_member(img, "width") : 0;
    const int h = img && json_object_has_member(img, "height")
      ? (int)json_object_get_int_member(img, "height") : 0;
    if(w <= 0 || h <= 0) STYLEAPPLY_SKIP("no image dimensions");

    const int bv = json_object_has_member(edit, "blendop_version")
      ? (int)json_object_get_int_member(edit, "blendop_version") : 14;

    const gboolean collides = !strcmp(op, host.operation);
    if(collides) same_op++;

    /* An edit the user kept as a second instance is applied inside a style that
       carries the module twice, which is the only way a style ever reaches a
       second instance: style application always replaces rather than appends,
       so the instance count comes from the style itself -- from an image that
       had the module twice when the style was captured. Both items then have to
       migrate, both have to persist, and neither may end up standing on the
       other's form. */
    const int mp = json_object_has_member(edit, "multi_priority")
      ? (int)json_object_get_int_member(edit, "multi_priority") : 0;
    const int n_items = mp > 0 ? 2 : 1;

    // A style never carries drawn geometry: masks_history is per image, and
    // data.db's style_items has no forms column -- dt_styles_create_from_image()
    // copies history rows only. So a classic style whose mask_mode has
    // DEVELOP_MASK_MASK arrives with a mask_id that resolves to nothing, on
    // this branch and on master alike. That is not a migration regression, and
    // these edits are counted separately so the headline number is not dominated
    // by a pre-existing property of styles. They are still checked: whatever
    // migration decides to do with an unresolvable drawn mask, the result must
    // not claim a form that is not there.
    if(bp.mask_mode & DEVELOP_MASK_MASK) drawn_in_style++;

    // phase 0: a fresh image carrying the classic host edit
    dt_masks_scratch_wipe_history(STYLEAPPLY_IMGID);
    dt_masks_scratch_seed_image(STYLEAPPLY_IMGID, w, h);
    if(!dt_masks_scratch_seed_history(STYLEAPPLY_IMGID, 0, host.operation, 0,
                                      host.blendop_version, &host.bp, host.forms))
      STYLEAPPLY_SKIP("host history row could not be seeded");

    // phase 1: settle it -- migrate and write back, so the style lands on a
    // normal already-migrated image
    gchar *host_before = NULL;
    _settle(&host, &host_before);

    total++;
    if(n_items > 1) multi_item++;

    // phase 2: apply the edit under test as a style
    int landed[STYLEAPPLY_MAX_ITEMS] = { 0 };
    if(!_apply_as_style(op, bv, &bp, n_items, landed))
    {
      errors++;
      g_free(host_before);
      continue;
    }
    int max_landed = 0;
    for(int k = 0; k < n_items; k++) max_landed = MAX(max_landed, landed[k]);
    if(max_landed > 0) landed_second++;

    // phase 3: reload from the database and see what actually persisted
    dt_develop_t dev;
    _open_scratch_dev(&dev);

    /* Every item the style carried has to have arrived. The verdict is taken
       from the *worst* of them, so a style whose second instance was lost
       cannot be reported as ok on the strength of its first. */
    gboolean style_resolved = TRUE;
    gboolean vanished = FALSE;
    gboolean id_preserved = TRUE;
    gchar *style_desc = NULL;
    int worst_mp = landed[0];

    for(int k = 0; k < n_items; k++)
    {
      gboolean k_resolved = FALSE;
      const dt_dev_history_item_t *ih = _history_item_for(&dev, op, landed[k]);
      gchar *d = _describe_mask(&dev, ih ? ih->blend_params : NULL, &k_resolved);

      const gboolean k_same_id =
        ih && ih->blend_params && ih->blend_params->mask_id == bp.mask_id;

      // keep the first failing item's description, else the first item's
      if(!style_desc || (style_resolved && !k_resolved))
      {
        g_free(style_desc);
        style_desc = d;
        if(!k_resolved) worst_mp = landed[k];
      }
      else
        g_free(d);

      if(!ih) vanished = TRUE;
      if(!k_resolved) style_resolved = FALSE;
      if(!k_same_id) id_preserved = FALSE;
    }

    /* A drawn-ONLY style is the one case where a dangling mask is the correct,
       pre-existing outcome rather than a regression, and it is recognised
       precisely rather than by mask_mode alone: migration reuses the classic
       mask_id verbatim for drawn-only (see _dispatch()), so the mask_id that
       comes back must be the very same id the style was saved with -- an id
       that names a form belonging to the image the style was created from, and
       that no style has ever carried. Master behaves identically here. If the
       id had *changed*, migration would have synthesized something and lost
       it, which is a real failure, so the equality is part of the test. */
    const gboolean drawn_only =
      (bp.mask_mode & DEVELOP_MASK_MASK)
      && !(bp.mask_mode & (DEVELOP_MASK_CONDITIONAL | DEVELOP_MASK_RASTER));
    const gboolean expected_dangling =
      !style_resolved && drawn_only && id_preserved;

    // whatever the style did not land on must be exactly as it was
    gchar *host_after = NULL;
    gboolean host_ok = TRUE;
    if(!collides)
    {
      const dt_dev_history_item_t *hh = _history_item_for(&dev, host.operation, 0);
      host_after = _describe_mask(&dev, hh ? hh->blend_params : NULL, NULL);
      host_ok = host_before && host_after && !strcmp(host_before, host_after);
    }

    dt_dev_cleanup(&dev);

    /* A style item that landed on no module at all is the failure a
       single-instance harness could not see: the history row is written, then
       dropped on the next read for want of an iop-order entry, and two absent
       masks compare equal no matter what migration did. Named separately so it
       cannot be read as an ordinary dangling mask. */
    /* `verdict` is prose for a human reading the report; `outcome` is the
       stable slug tools aggregate on. Both, because the two audiences want
       different things and collapsing them means one of them loses: an
       aggregator matching on the prose breaks silently the moment the wording
       is improved (which is exactly how 584 known-good rows were once counted
       as failures). */
    const char *verdict;
    const char *outcome;
    if(vanished)
    {
      no_module++;
      verdict = "style item landed on no module at all";
      outcome = "no_module";
      printf("[styleapply] NO MODULE at edit %u (%s, instance %d):"
             " history row did not survive the reload\n", i, op, worst_mp);
    }
    else if(!host_ok)
    {
      host_lost++;
      verdict = "host mask disturbed";
      outcome = "host_disturbed";
      printf("[styleapply] HOST DISTURBED at edit %u (%s):\n"
             "             before: %s\n"
             "             after : %s\n",
             i, op, host_before, host_after);
    }
    else if(expected_dangling)
    {
      not_carried++;
      verdict = "drawn-only style, form never carried (same on master)";
      outcome = "not_carried";
    }
    else if(!style_resolved)
    {
      dangling++;
      verdict = "style mask lost";
      outcome = "style_mask_lost";
      printf("[styleapply] DANGLING at edit %u (%s, instance %d): %s\n",
             i, op, worst_mp, style_desc);
    }
    else
    {
      ok_count++;
      verdict = "ok";
      outcome = "ok";
    }

    if(rf)
    {
      gchar *esc = g_strescape(style_desc, NULL);
      fprintf(rf, "%s\n    {\"index\": %u, \"operation\": \"%s\", \"mask_mode\": %u,"
                  " \"same_op_as_host\": %s, \"style_items\": %d,"
                  " \"harvested_multi_priority\": %d, \"max_landed_instance\": %d,"
                  " \"outcome\": \"%s\", \"result\": \"%s\","
                  " \"style_mask\": \"%s\"}",
              first_report ? "" : ",", i, op, bp.mask_mode,
              collides ? "true" : "false", n_items, mp, max_landed, outcome,
              verdict, esc);
      g_free(esc);
      first_report = FALSE;
    }

    g_free(style_desc);
    g_free(host_before);
    g_free(host_after);

    if((i + 1) % 250 == 0) printf("[styleapply]   %u/%u ...\n", i + 1, n);
  }

  g_list_free_full(host.forms, (GDestroyNotify)dt_masks_free_form);
  g_object_unref(parser);

  const gboolean passed =
    dangling == 0 && host_lost == 0 && no_module == 0 && errors == 0;

  // the report carries every figure the summary below prints, so it can be read
  // on its own without the terminal output of the run that produced it
  if(rf)
  {
    fputs("\n  ],\n  \"summary\": {\n", rf);
    fprintf(rf, "    \"ran\": true,\n");
    fprintf(rf, "    \"passed\": %s,\n", passed ? "true" : "false");
    fprintf(rf, "    \"harvested\": %u,\n", n);
    fprintf(rf, "    \"applied_as_style\": %d,\n", total);
    fprintf(rf, "    \"ok\": %d,\n", ok_count);
    fprintf(rf, "    \"style_mask_lost\": %d,\n", dangling);
    fprintf(rf, "    \"host_disturbed\": %d,\n", host_lost);
    fprintf(rf, "    \"no_module_at_all\": %d,\n", no_module);
    fprintf(rf, "    \"drawn_only_not_carried\": %d,\n", not_carried);
    fprintf(rf, "    \"errors\": %d,\n", errors);
    fprintf(rf, "    \"skipped\": %d,\n", skipped);
    fprintf(rf, "    \"same_op_as_host\": %d,\n", same_op);
    fprintf(rf, "    \"drawn_mask_in_style\": %d,\n", drawn_in_style);
    fprintf(rf, "    \"multi_instance_styles\": %d,\n", multi_item);
    fprintf(rf, "    \"second_instance_allocated\": %d\n", landed_second);
    fputs("  }", rf);
  }

  printf("[styleapply]\n");
  printf("[styleapply] applied as style : %d\n", total);
  printf("[styleapply]   ok             : %d\n", ok_count);
  printf("[styleapply]   STYLE MASK LOST: %d  (migrated form never persisted)\n", dangling);
  printf("[styleapply]   HOST DISTURBED : %d\n", host_lost);
  printf("[styleapply]   NO MODULE AT ALL: %d  (row dropped on reload;"
         " would otherwise pass vacuously)\n", no_module);
  printf("[styleapply]   drawn-only, form never carried by any style"
         " (same on master, not a failure) : %d\n", not_carried);
  printf("[styleapply]   errors         : %d\n", errors);
  printf("[styleapply]   skipped        : %d  (already flexi, or unreconstructable)\n",
         skipped);
  printf("[styleapply]   (of those applied, %d target the host module itself,"
         " where replacing the host mask is correct)\n", same_op);
  printf("[styleapply]   (and %d carry a drawn mask, which no style can ever"
         " carry forms for -- true on master too)\n", drawn_in_style);
  printf("[styleapply]   multi-instance edits, applied in a style carrying the"
         " module twice : %d\n", multi_item);
  printf("[styleapply]     of those, a second instance was actually allocated  "
         "         : %d\n", landed_second);

  return passed;
}

#undef STYLEAPPLY_SKIP

gboolean dt_masks_styleapply_harvest(const char *json_path, const char *report_path)
{
  FILE *rf = report_path ? g_fopen(report_path, "wb") : NULL;
  if(rf) fputs("{", rf);
  const gboolean ok = dt_masks_styleapply_harvest_section(json_path, rf, NULL);
  if(rf)
  {
    fputs("\n}\n", rf);
    fclose(rf);
    printf("[styleapply] per-edit report written to %s\n", report_path);
  }
  return ok;
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
