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

  int g = 0;
  gchar **groups = g_strsplit(layout, "|", -1);
  for(int k = 0; groups[k]; k++)
  {
    gchar *spec = g_strstrip(g_strdup(groups[k]));
    if(!*spec)
    {
      g_free(spec);
      continue;
    }
    // "[x]" is an empty group, "x:ids" one with elements
    const gboolean empty = spec[0] == '[';
    const dt_masks_state_t op = _op_from_letter(empty ? spec[1] : spec[0]);
    if(!empty) assert_true(spec[1] == ':');

    dt_masks_point_group_t *marker = calloc(1, sizeof(dt_masks_point_group_t));
    marker->formid = FLEXI_GID(g++);
    marker->parentid = FLEXI_GROUP_ID;
    marker->state = DT_MASKS_STATE_GROUP_MARKER | op;
    marker->opacity = 1.0f;
    marker->group_opacity = 1.0f;
    _grp->points = g_list_append(_grp->points, marker);

    gchar **ids = g_strsplit(empty ? "" : spec + 2, ",", -1);
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
      pt->state = DT_MASKS_STATE_USE | DT_MASKS_STATE_UNION;
      pt->opacity = 1.0f;
      pt->group_opacity = 1.0f;
      _grp->points = g_list_append(_grp->points, pt);
      _add_form(fid);
    }
    g_strfreev(ids);
    g_free(spec);
  }
  g_strfreev(groups);

  _forms = g_list_append(_forms, _grp);

  flexi_dev.forms = _forms;
  flexi_bp.mask_id = FLEXI_GROUP_ID;
  flexi_bp.mask_mode = DEVELOP_MASK_ENABLED | DEVELOP_MASK_FLEXI;
  flexi_module.blend_params = &flexi_bp;
  flexi_module.blend_data = &flexi_bd;
  flexi_module.dev = &flexi_dev;
  flexi_bd.module = &flexi_module;
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

dt_masks_form_t *flexi_build_classic(const char *layout)
{
  dt_masks_form_t *grp = flexi_build(layout);
  dt_masks_state_t op = DT_MASKS_STATE_UNION;
  gboolean first_member = FALSE;
  GList *l = grp->points;
  while(l)
  {
    GList *next = g_list_next(l);
    dt_masks_point_group_t *pt = l->data;
    if(dt_masks_point_is_marker(pt))
    {
      op = pt->state & DT_MASKS_STATE_OP;
      // the bottom point of the whole list cannot carry a break
      first_member = l != grp->points;
      free(pt);
      grp->points = g_list_delete_link(grp->points, l);
    }
    else
    {
      pt->state = (pt->state & ~DT_MASKS_STATE_OP) | op;
      pt->group_start = first_member ? 1 : 0;
      first_member = FALSE;
    }
    l = next;
  }
  return grp;
}

dt_masks_form_t *flexi_group(void)
{
  return _grp;
}

char *flexi_layout(void)
{
  return flexi_layout_of(_grp);
}

char *flexi_layout_of(const dt_masks_form_t *g)
{
  GString *s = g_string_new(NULL);
  gboolean open_group = FALSE; // a group whose "x:" is written, with members
  char pending = 0;            // a group whose marker is read, with none yet
  for(GList *l = g ? g->points : NULL; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    // partition through the same predicate the panel uses -- see the header
    if(_starts_group(l))
    {
      if(pending) g_string_append_printf(s, "%s[%c]", s->len ? " | " : "", pending);
      pending = _letter_from_op(_eff_group_op(pt->state) & DT_MASKS_STATE_OP_COMBINE);
      open_group = FALSE;
      continue;
    }
    if(pending)
    {
      g_string_append_printf(s, "%s%c:", s->len ? " | " : "", pending);
      pending = 0;
      open_group = TRUE;
    }
    else if(open_group)
      g_string_append_c(s, ',');
    g_string_append_printf(s, "%d", (int)pt->formid);
  }
  if(pending) g_string_append_printf(s, "%s[%c]", s->len ? " | " : "", pending);
  return g_string_free(s, FALSE);
}

static char _letter_from_within(const int state)
{
  switch(state & DT_MASKS_STATE_WITHIN)
  {
    case 0: return 'u';
    case DT_MASKS_STATE_ISECT: return 'i';
    case DT_MASKS_STATE_WITHIN_DIFFERENCE: return 'd';
    case DT_MASKS_STATE_WITHIN_EXCLUSION: return 'x';
    case DT_MASKS_STATE_WITHIN_SUM: return 's';
    case DT_MASKS_STATE_WITHIN_MULTIPLY: return 'm';
    case DT_MASKS_STATE_SCREEN: return 'o';
    default: return '?';
  }
}

static void _append_settings(GString *s, const gboolean inverted, const float opacity)
{
  if(inverted) g_string_append_c(s, '~');
  if(opacity != 1.0f) g_string_append_printf(s, "@%g", opacity);
}

static void _tree_into(GString *s, const dt_masks_form_t *g, const int depth)
{
  gboolean open = FALSE;  // a group's "{" is written
  gboolean first = TRUE;  // nothing written in it yet
  for(GList *l = g->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    if(dt_masks_point_is_marker(pt))
    {
      if(open) g_string_append(s, "} | ");
      g_string_append_c(s, _letter_from_within(pt->state));
      _append_settings(s, (pt->state & DT_MASKS_STATE_OP_INVERT) != 0, pt->group_opacity);
      g_string_append_c(s, '{');
      open = TRUE;
      first = TRUE;
      continue;
    }
    if(!first) g_string_append_c(s, ',');
    first = FALSE;
    const dt_masks_form_t *f = dt_masks_get_from_id_ext(flexi_dev.forms, pt->formid);
    if(f && f != g && (f->type & DT_MASKS_GROUP) && depth < DT_MASKS_NESTING_MAX)
      _tree_into(s, f, depth + 1);
    else
      g_string_append_printf(s, "%d", (int)pt->formid);
    _append_settings(s, (pt->state & DT_MASKS_STATE_INVERSE) != 0, pt->opacity);
  }
  if(open) g_string_append_c(s, '}');
}

char *flexi_tree_of(const dt_masks_form_t *g)
{
  GString *s = g_string_new(NULL);
  if(g) _tree_into(s, g, 0);
  return g_string_free(s, FALSE);
}

void flexi_assert_tree_(const dt_masks_form_t *g,
                        const char *expect,
                        const char *file,
                        const int line)
{
  char *got = flexi_tree_of(g);
  if(strcmp(got, expect) != 0)
  {
    print_error("%s:%d: tree mismatch\n  expected: %s\n  actual:   %s\n", file, line, expect,
                got);
    g_free(got);
    fail();
  }
  g_free(got);
}

dt_masks_state_t flexi_group_op_of(const dt_mask_id_t fid)
{
  const dt_masks_point_group_t *marker = _group_point(_grp, _group_cid_of_form(_grp, fid));
  return marker ? _eff_group_op(marker->state) & DT_MASKS_STATE_OP_COMBINE : 0;
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
  flexi_assert_layout_of_(_grp, expect, file, line);
}

void flexi_assert_layout_of_(const dt_masks_form_t *g,
                             const char *expect,
                             const char *file,
                             const int line)
{
  char *got = flexi_layout_of(g);
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
