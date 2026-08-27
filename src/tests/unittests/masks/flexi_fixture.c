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

#include "flexi_fixture.h"
#include "control/conf.h"

// defined in the generated conf_gen.h, compiled into lib_darktable
extern void dt_confgen_init(void);

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <cmocka.h>

#include <stdio.h>
#include <string.h>

dt_develop_t flexi_dev;
dt_iop_module_t flexi_module;
dt_iop_gui_blend_data_t flexi_bd;
dt_develop_blend_params_t flexi_bp;

static dt_masks_form_t *_grp = NULL;
// the element forms the group's points refer to. dt_masks_get_from_id() must
// find these, since the model resolves a formid to a form to tell a shape from
// a parametric element.
static GList *_forms = NULL;

#define FLEXI_GROUP_ID 1000

static dt_masks_state_t _op_from_letter(const char c)
{
  switch(c)
  {
    case 'u': return DT_MASKS_STATE_UNION;
    case 'i': return DT_MASKS_STATE_INTERSECTION;
    case 'd': return DT_MASKS_STATE_DIFFERENCE;
    case 'x': return DT_MASKS_STATE_EXCLUSION;
    case 's': return DT_MASKS_STATE_SUM;
    default: fail_msg("unknown operator letter '%c' in layout string", c);
  }
  return DT_MASKS_STATE_UNION; // unreachable; keeps the compiler quiet
}

static char _letter_from_op(const dt_masks_state_t op)
{
  switch(op)
  {
    case DT_MASKS_STATE_UNION: return 'u';
    case DT_MASKS_STATE_INTERSECTION: return 'i';
    case DT_MASKS_STATE_DIFFERENCE: return 'd';
    case DT_MASKS_STATE_EXCLUSION: return 'x';
    case DT_MASKS_STATE_SUM: return 's';
    default: return '?';
  }
}

// a minimal shape form so dt_masks_get_from_id() resolves this element. Type
// matters only where the model distinguishes shapes from parametric elements;
// tests that need a parametric element set the type themselves afterwards.
static void _add_form(const dt_mask_id_t fid)
{
  dt_masks_form_t *f = calloc(1, sizeof(dt_masks_form_t));
  f->formid = fid;
  f->type = DT_MASKS_CIRCLE;
  snprintf(f->name, sizeof(f->name), "circle #%d", (int)fid);
  _forms = g_list_append(_forms, f);
}

dt_masks_form_t *flexi_build(const char *layout)
{
  flexi_teardown();

  memset(&flexi_dev, 0, sizeof(flexi_dev));
  memset(&flexi_module, 0, sizeof(flexi_module));
  memset(&flexi_bd, 0, sizeof(flexi_bd));
  memset(&flexi_bp, 0, sizeof(flexi_bp));

  _grp = calloc(1, sizeof(dt_masks_form_t));
  _grp->formid = FLEXI_GROUP_ID;
  _grp->type = DT_MASKS_GROUP;

  gboolean first_group = TRUE;
  gchar **groups = g_strsplit(layout, "|", -1);
  for(int g = 0; groups[g]; g++)
  {
    gchar *spec = g_strstrip(g_strdup(groups[g]));
    if(!*spec)
    {
      g_free(spec);
      continue;
    }
    assert_true(spec[1] == ':');
    const dt_masks_state_t op = _op_from_letter(spec[0]);

    gboolean first_member = TRUE;
    gchar **ids = g_strsplit(spec + 2, ",", -1);
    for(int m = 0; ids[m]; m++)
    {
      gchar *idstr = g_strstrip(g_strdup(ids[m]));
      if(!*idstr)
      {
        g_free(idstr);
        continue;
      }
      const dt_mask_id_t fid = (dt_mask_id_t)atoi(idstr);
      g_free(idstr);

      dt_masks_point_group_t *pt = calloc(1, sizeof(dt_masks_point_group_t));
      pt->formid = fid;
      pt->parentid = FLEXI_GROUP_ID;
      pt->state = op | DT_MASKS_STATE_USE;
      pt->opacity = 1.0f;
      // the bottom-most point of the whole list cannot carry a break -- it
      // starts a group by virtue of being first (see _starts_group)
      pt->group_start = (first_member && !first_group) ? 1 : 0;

      _grp->points = g_list_append(_grp->points, pt);
      _add_form(fid);
      first_member = FALSE;
    }
    g_strfreev(ids);
    g_free(spec);
    first_group = FALSE;
  }
  g_strfreev(groups);

  _forms = g_list_append(_forms, _grp);

  flexi_dev.forms = _forms;
  flexi_bp.mask_id = FLEXI_GROUP_ID;
  flexi_bp.mask_mode = DEVELOP_MASK_ENABLED | DEVELOP_MASK_FLEXI;
  flexi_module.blend_params = &flexi_bp;
  flexi_module.blend_data = &flexi_bd;
  flexi_module.dev = &flexi_dev;
  // every mask-id field starts INVALID, not zero -- a zeroed blend_data would
  // read as "element 0 is soloed" (see the matching initialisation in
  // blend_gui.c's panel setup)
  flexi_bd.panel_selected_formid = INVALID_MASKID;
  flexi_bd.panel_selected_group_cid = INVALID_MASKID;
  flexi_bd.solo_formid = INVALID_MASKID;
  flexi_bd.soloedit_formid = INVALID_MASKID;
  flexi_bd.solo_group_key = 0;

  darktable.develop = &flexi_dev;
  return _grp;
}

dt_masks_form_t *flexi_group(void)
{
  return _grp;
}

char *flexi_layout(void)
{
  GString *s = g_string_new(NULL);
  for(GList *l = _grp ? _grp->points : NULL; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    // partition through the same predicate the panel and renderer use, not
    // through pt->group_start directly -- see the header comment
    if(_starts_group(l))
    {
      if(s->len) g_string_append(s, " | ");
      g_string_append_printf(s, "%c:", _letter_from_op(_eff_group_op(pt->state)));
    }
    else
      g_string_append_c(s, ',');
    g_string_append_printf(s, "%d", (int)pt->formid);
  }
  return g_string_free(s, FALSE);
}

dt_masks_empty_group_t *flexi_add_empty(const dt_masks_state_t op,
                                        const dt_mask_id_t below_fid)
{
  dt_masks_empty_group_t *eg = _empty_group_new(op, DT_MASKS_STATE_NONE, below_fid);
  flexi_bd.empty_groups = g_list_append(flexi_bd.empty_groups, eg);
  return eg;
}

char *flexi_visual_order(void)
{
  GString *s = g_string_new(NULL);
  GList *order = _masks_visual_group_order(&flexi_module);
  for(GList *l = order; l; l = g_list_next(l))
  {
    const _dt_masks_order_item_t *it = l->data;
    if(s->len) g_string_append(s, " | ");
    if(it->is_empty)
    {
      // a staged group has no members to name, so show only its operator
      g_string_append_printf(s, "[%c]", _letter_from_op(_eff_group_op(it->eg->op)));
    }
    else
    {
      GList *run = _selected_group_formids(_grp, it->cid);
      const dt_masks_point_group_t *head = _group_point(_grp, it->cid);
      g_string_append_printf(s, "%c:", _letter_from_op(_eff_group_op(head->state)));
      // _selected_group_formids returns the run top-down; print bottom-up to
      // match the layout strings
      GList *rev = g_list_reverse(g_list_copy(run));
      for(GList *m = rev; m; m = g_list_next(m))
        g_string_append_printf(s, "%s%d", m == rev ? "" : ",",
                               GPOINTER_TO_INT(m->data));
      g_list_free(rev);
      g_list_free(run);
    }
  }
  g_list_free_full(order, g_free);
  return g_string_free(s, FALSE);
}

void flexi_assert_order_(const char *expect, const char *file, const int line)
{
  char *got = flexi_visual_order();
  if(strcmp(got, expect) != 0)
  {
    print_error("%s:%d: visual order mismatch\n  expected: %s\n  actual:   %s\n",
                file, line, expect, got);
    g_free(got);
    fail();
  }
  g_free(got);
}

void flexi_set_ordinal(const dt_mask_id_t cid, const int ord)
{
  if(!flexi_bd.group_ordinals)
    flexi_bd.group_ordinals = g_hash_table_new(g_direct_hash, g_direct_equal);
  g_hash_table_insert(flexi_bd.group_ordinals, GINT_TO_POINTER(cid),
                      GINT_TO_POINTER(ord));
}

int flexi_get_ordinal(const dt_mask_id_t cid)
{
  if(!flexi_bd.group_ordinals) return 0;
  return GPOINTER_TO_INT(
    g_hash_table_lookup(flexi_bd.group_ordinals, GINT_TO_POINTER(cid)));
}

void flexi_assert_layout_(const char *expect, const char *file, const int line)
{
  char *got = flexi_layout();
  if(strcmp(got, expect) != 0)
  {
    // print both before failing: cmocka's string diff alone is hard to read
    // for these, and the layout is the whole point of the assertion
    print_error("%s:%d: layout mismatch\n  expected: %s\n  actual:   %s\n",
                file, line, expect, got);
    g_free(got);
    fail();
  }
  g_free(got);
}

static gchar *_conf_path = NULL;

void flexi_conf_init(void)
{
  if(darktable.conf) return;
  _conf_path = g_build_filename(g_get_tmp_dir(), "flexi_unittest_rc", NULL);
  // start from a clean slate every run, so one test's writes cannot leak into
  // the next run's expectations
  g_unlink(_conf_path);
  darktable.conf = calloc(1, sizeof(dt_conf_t));
  // the defaults/min/max table, generated from darktableconfig.xml into
  // conf_gen.h and compiled into lib_darktable. dt_conf_init sanitizes values
  // against it and dt_conf_get_* falls back to it for unset keys, so without
  // this every lookup hits a NULL table.
  dt_confgen_init();
  dt_conf_init(darktable.conf, _conf_path, FALSE, NULL);
}

void flexi_conf_cleanup(void)
{
  if(!darktable.conf) return;
  dt_conf_cleanup(darktable.conf);
  free(darktable.conf);
  darktable.conf = NULL;
  if(_conf_path)
  {
    g_unlink(_conf_path);
    g_free(_conf_path);
    _conf_path = NULL;
  }
}

void flexi_teardown(void)
{
  if(_grp)
  {
    g_list_free_full(_grp->points, free);
    _grp->points = NULL;
  }
  for(GList *l = _forms; l; l = g_list_next(l))
    if(l->data != _grp) free(l->data);
  g_list_free(_forms);
  _forms = NULL;
  free(_grp);
  _grp = NULL;

  for(GList *l = flexi_bd.empty_groups; l; l = g_list_next(l))
  {
    dt_masks_empty_group_t *eg = l->data;
    g_free(eg->name);
    free(eg);
  }
  g_list_free(flexi_bd.empty_groups);
  flexi_bd.empty_groups = NULL;
  if(flexi_bd.group_ordinals)
  {
    g_hash_table_destroy(flexi_bd.group_ordinals);
    flexi_bd.group_ordinals = NULL;
  }

  darktable.develop = NULL;
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
