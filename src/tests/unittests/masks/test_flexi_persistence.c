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

// Mask blob version migration (dt_masks_legacy_params).
//
// This is the code that carries every already-saved edit forward when the
// group point struct gains a field. It is worth testing carefully for three
// reasons:
//
//   * it runs against data nobody can regenerate -- a user's existing library;
//   * it fails silently. A missed fixup does not crash or log, it just renders
//     a subtly (or catastrophically) different mask; and
//   * it is the one place where a zero-filled field is NOT automatically safe.
//     Fields appended to dt_masks_point_group_t are read at the historic
//     stride and zero-filled (see dt_masks_read_forms_ext), which is neutral
//     for most of them -- but group_opacity is multiplicative, so a zero-fill
//     would silently blank out every pre-v9 group's mask.
//
// The read-time stride selection itself is SQLite-coupled and not reachable
// here; what these tests cover is the migration chain that runs after it, on a
// form whose points have already been zero-filled the way that reader leaves
// them.

#include "flexi_fixture.h"

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <cmocka.h>

static int _teardown(void **state)
{
  flexi_teardown();
  return 0;
}

// a group whose points look the way dt_masks_read_forms_ext leaves them for an
// edit saved at `version`: everything appended after that version zero-filled.
static dt_masks_form_t *_build_legacy(const char *layout, const int version)
{
  dt_masks_form_t *grp = flexi_build(layout);
  grp->version = version;
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = l->data;
    if(version < 7) memset(&pt->refinement, 0, sizeof(pt->refinement));
    if(version < 8) memset(pt->name, 0, sizeof(pt->name));
    if(version < 9) pt->group_opacity = 0.0f;
    if(version < 10) pt->group_start = 0;
  }
  return grp;
}

static void _migrate(dt_masks_form_t *grp)
{
  const int rc = dt_masks_legacy_params(&flexi_dev, grp, grp->version,
                                        dt_masks_version());
  assert_int_equal(rc, 0);
  assert_int_equal(grp->version, dt_masks_version());
}

// ---------------------------------------------------------------------------
// v8 -> v9: group_opacity
// ---------------------------------------------------------------------------

// The dangerous one. group_opacity multiplies the group's finished sub-mask,
// so the 0.0 left by the read-time zero-fill is not neutral -- it is "erase
// this group entirely". Every pre-v9 group point must come out at 1.0.
static void test_v9_gives_every_group_point_unit_opacity(void **state)
{
  dt_masks_form_t *grp = _build_legacy("u:1,2 | i:3,4", 8);
  _migrate(grp);

  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    if(pt->group_opacity != 1.0f)
      fail_msg("form %d migrated from v8 with group_opacity %f -- a pre-v9 "
               "group would render blank", (int)pt->formid, pt->group_opacity);
  }
}

// the same fixup must apply however far back the edit came from, since the
// chain runs every intermediate step in order
static void test_v9_fixup_applies_from_every_older_version(void **state)
{
  for(int v = 6; v <= 8; v++)
  {
    dt_masks_form_t *grp = _build_legacy("u:1,2", v);
    _migrate(grp);
    for(GList *l = grp->points; l; l = g_list_next(l))
      assert_float_equal(((dt_masks_point_group_t *)l->data)->group_opacity,
                         1.0f, 1e-6);
    flexi_teardown();
  }
}

// an edit already at v9 or later keeps whatever the user actually set --
// the fixup must not stomp a real value
static void test_v9_does_not_overwrite_an_explicit_opacity(void **state)
{
  dt_masks_form_t *grp = _build_legacy("u:1,2", 9);
  for(GList *l = grp->points; l; l = g_list_next(l))
    ((dt_masks_point_group_t *)l->data)->group_opacity = 0.25f;

  _migrate(grp);

  for(GList *l = grp->points; l; l = g_list_next(l))
    assert_float_equal(((dt_masks_point_group_t *)l->data)->group_opacity,
                       0.25f, 1e-6);
}

// ---------------------------------------------------------------------------
// v9 -> v10: the GROUP_BREAK bit becomes the group_start field
// ---------------------------------------------------------------------------

// A break set in the old borrowed state bit must survive as the new field, or
// every multi-group edit saved under v9 collapses into one group on load.
static void test_v10_carries_break_bit_into_group_start(void **state)
{
  dt_masks_form_t *grp = _build_legacy("u:1,2 | u:3,4", 9);
  // re-encode the partition the way v9 stored it: the break lived in `state`
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = l->data;
    if(pt->formid == 3) pt->state |= DT_MASKS_STATE_GROUP_BREAK;
    pt->group_start = 0;
  }

  _migrate(grp);

  assert_int_equal(_group_point(grp, 3)->group_start, 1);
  // and the stale bit is cleared, so nothing reads it from `state` later
  assert_int_equal(_group_point(grp, 3)->state & DT_MASKS_STATE_GROUP_BREAK, 0);
  // the partition the user saved is the partition they get back
  assert_layout("u:1,2 | u:3,4");
}

// the inverse: no break bit anywhere means one run, not several
static void test_v10_without_break_bit_yields_one_group(void **state)
{
  dt_masks_form_t *grp = _build_legacy("u:1,2,3", 9);
  _migrate(grp);
  assert_layout("u:1,2,3");
}

// pre-v10 edits could not have two adjacent same-operator groups (there was no
// way to express it), so operator changes alone must still partition them --
// this is the back-compat path in _starts_group
static void test_v10_operator_change_still_partitions_old_edits(void **state)
{
  dt_masks_form_t *grp = _build_legacy("u:1,2 | i:3", 9);
  for(GList *l = grp->points; l; l = g_list_next(l))
    ((dt_masks_point_group_t *)l->data)->group_start = 0;

  _migrate(grp);
  assert_layout("u:1,2 | i:3");
}

// ---------------------------------------------------------------------------
// chain properties
// ---------------------------------------------------------------------------

// migrating is idempotent: a form already at the current version comes out
// unchanged, structurally and by hash
static void test_migration_is_idempotent(void **state)
{
  dt_masks_form_t *grp = _build_legacy("u:1,2 | i:3,4", 8);
  _migrate(grp);
  const dt_hash_t once = dt_masks_group_hash(DT_INITHASH, grp);
  char *layout = flexi_layout();

  _migrate(grp);
  assert_int_equal(dt_masks_group_hash(DT_INITHASH, grp), once);
  assert_layout(layout);
  g_free(layout);
}

// nonsensical version pairs are refused rather than half-applied
static void test_migration_rejects_impossible_versions(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2");
  // newer than current: nothing can be done with it
  assert_int_not_equal(dt_masks_legacy_params(&flexi_dev, grp,
                                              dt_masks_version() + 1,
                                              dt_masks_version()), 0);
  // version 0 is not a thing
  assert_int_not_equal(dt_masks_legacy_params(&flexi_dev, grp, 0,
                                              dt_masks_version()), 0);
}

// the migration must not renumber, reorder or drop members
static void test_migration_preserves_membership(void **state)
{
  dt_masks_form_t *grp = _build_legacy("u:1,2 | i:3,4 | d:5", 6);
  const guint before = g_list_length(grp->points);
  _migrate(grp);
  assert_int_equal(g_list_length(grp->points), before);
  assert_layout("u:1,2 | i:3,4 | d:5");
}

// refinement is zero-filled for pre-v7 edits, and zero means disabled -- an old
// mask must not come back with feathering or blur switched on
static void test_pre_v7_refinement_stays_disabled(void **state)
{
  dt_masks_form_t *grp = _build_legacy("u:1,2 | i:3", 6);
  _migrate(grp);
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    assert_int_equal(pt->refinement.enabled, 0);
    assert_float_equal(pt->refinement.feathering_radius, 0.0f, 1e-6);
    assert_float_equal(pt->refinement.blur_radius, 0.0f, 1e-6);
  }
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_teardown(test_v9_gives_every_group_point_unit_opacity, _teardown),
    cmocka_unit_test_teardown(test_v9_fixup_applies_from_every_older_version, _teardown),
    cmocka_unit_test_teardown(test_v9_does_not_overwrite_an_explicit_opacity, _teardown),
    cmocka_unit_test_teardown(test_v10_carries_break_bit_into_group_start, _teardown),
    cmocka_unit_test_teardown(test_v10_without_break_bit_yields_one_group, _teardown),
    cmocka_unit_test_teardown(test_v10_operator_change_still_partitions_old_edits, _teardown),
    cmocka_unit_test_teardown(test_migration_is_idempotent, _teardown),
    cmocka_unit_test_teardown(test_migration_rejects_impossible_versions, _teardown),
    cmocka_unit_test_teardown(test_migration_preserves_membership, _teardown),
    cmocka_unit_test_teardown(test_pre_v7_refinement_stays_disabled, _teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
