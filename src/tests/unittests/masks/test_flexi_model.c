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

// Behavioural regression tests for the flexi masks panel's model layer:
// grouping, drag-and-drop and selection, expressed as layout strings (see
// flexi_fixture.h). These run headless -- no GTK, no display -- because every
// behaviour asserted here lives in functions that take a mask group and plain
// values, not widgets.
//
// What this suite deliberately does NOT cover: anything that is a property of
// GTK itself rather than of the panel's logic -- event propagation between
// nested widgets, CSS rendering, tooltip delivery, widget packing. Those bugs
// are real (the double-fire on group_block's release handler was exactly one)
// but reproducing them requires real GTK event delivery against a real widget
// tree, which needs a display and is brittle enough that it would cost more
// trust than it earns. They stay a manual checklist; see README.md.

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

// ---------------------------------------------------------------------------
// the layout DSL itself -- if these are wrong every other test lies
// ---------------------------------------------------------------------------

static void test_layout_roundtrip(void **state)
{
  const char *cases[] = {
    "u:1",
    "u:1,2,3",
    "u:1,2 | i:3",
    "u:1 | i:2 | d:3 | x:4 | s:5",
    NULL,
  };
  for(int i = 0; cases[i]; i++)
  {
    flexi_build(cases[i]);
    assert_layout(cases[i]);
    flexi_teardown();
  }
}

// Two adjacent groups sharing one operator must stay two groups. This is the
// entire reason group_start exists as a stored field: before it, the partition
// was inferred from operator changes alone, so same-op neighbours silently
// merged.
static void test_adjacent_same_op_groups_stay_separate(void **state)
{
  flexi_build("u:1,2 | u:3");
  assert_layout("u:1,2 | u:3");

  GList *heads = _group_partition_heads(flexi_group());
  assert_int_equal(g_list_length(heads), 2);
  assert_int_equal(GPOINTER_TO_INT(heads->data), 1);
  assert_int_equal(GPOINTER_TO_INT(heads->next->data), 3);
  g_list_free(heads);
}

// ---------------------------------------------------------------------------
// group membership queries
// ---------------------------------------------------------------------------

static void test_cid_of_form_is_run_head(void **state)
{
  flexi_build("u:1,2 | i:3,4");
  dt_masks_form_t *grp = flexi_group();

  assert_int_equal(_group_cid_of_form(grp, 1), 1);
  assert_int_equal(_group_cid_of_form(grp, 2), 1);
  assert_int_equal(_group_cid_of_form(grp, 3), 3);
  assert_int_equal(_group_cid_of_form(grp, 4), 3);
  assert_int_equal(_group_cid_of_form(grp, 99), INVALID_MASKID);
}

static void test_selected_group_formids(void **state)
{
  flexi_build("u:1,2 | i:3,4,5");
  GList *run = _selected_group_formids(flexi_group(), 4);
  assert_int_equal(g_list_length(run), 3);
  g_list_free(run);

  run = _selected_group_formids(flexi_group(), 1);
  assert_int_equal(g_list_length(run), 2);
  g_list_free(run);
}

// ---------------------------------------------------------------------------
// the key snapshot/apply pair -- the mechanism every reorder relies on
// ---------------------------------------------------------------------------

// Reordering points must not repartition them. Snapshot, move a point within
// its own group, re-stamp: same groups, new order.
static void test_keys_survive_intra_group_reorder(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2,3 | i:4");

  GHashTable *keys = _group_keys_snapshot(grp);
  dt_masks_point_group_t *pt = _group_point(grp, 1);
  grp->points = g_list_remove(grp->points, pt);
  grp->points = g_list_insert(grp->points, pt, 2);
  _group_keys_apply(grp, keys);
  g_hash_table_destroy(keys);

  assert_layout("u:2,3,1 | i:4");
}

// A member absent from the key map inherits the key of the point below it, so
// a newly added shape merges into the group it sits on top of rather than
// starting a group of its own.
static void test_keys_absent_member_joins_group_below(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3");

  GHashTable *keys = _group_keys_snapshot(grp);
  dt_masks_point_group_t *pt = calloc(1, sizeof(dt_masks_point_group_t));
  pt->formid = 9;
  pt->state = DT_MASKS_STATE_INTERSECTION | DT_MASKS_STATE_USE;
  pt->opacity = 1.0f;
  grp->points = g_list_append(grp->points, pt); // on top of the whole list
  _group_keys_apply(grp, keys);
  g_hash_table_destroy(keys);

  assert_layout("u:1,2 | i:3,9");
}

// ---------------------------------------------------------------------------
// drag and drop: element onto element
// ---------------------------------------------------------------------------

static void test_drop_element_into_other_group(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3,4");

  // drop 1 onto 3, landing above it
  assert_true(_model_drop_element_onto_element(&flexi_module, grp, 1, 3, TRUE));
  assert_layout("u:2 | i:3,1,4");
}

static void test_drop_element_below_target(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3,4");

  assert_true(_model_drop_element_onto_element(&flexi_module, grp, 1, 3, FALSE));
  assert_layout("u:2 | i:1,3,4");
}

// The dragged element adopts its new group's operator -- otherwise it would
// keep its old one and split the group it just joined in two.
static void test_drop_adopts_target_operator(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | d:3");

  _model_drop_element_onto_element(&flexi_module, grp, 1, 3, TRUE);
  const dt_masks_point_group_t *moved = _group_point(grp, 1);
  assert_int_equal(_eff_group_op(moved->state), DT_MASKS_STATE_DIFFERENCE);
  assert_layout("u:2 | d:3,1");
}

static void _assert_group_count(dt_masks_form_t *grp, const int expect)
{
  GList *heads = _group_partition_heads(grp);
  const int n = g_list_length(heads);
  g_list_free(heads);
  if(n != expect)
  {
    char *got = flexi_layout();
    print_error("expected %d groups, found %d: %s\n", expect, n, got);
    g_free(got);
    fail();
  }
}

// The reported bug: moving an element from group A to group B produced a third
// group C. Whatever the cause, the invariant is simple and worth pinning --
// a move between two existing groups never changes the group count.
//
// Both drop directions are exercised deliberately: dropping *below* the target
// splits the target's run around the newcomer, so it is the direction that
// actually detects a lost partition re-stamp; dropping above it appends to the
// run's end and survives the same fault unnoticed. Testing only one direction
// here would have let the original bug through.
static void test_drop_between_groups_never_creates_a_third(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3,4");
  _model_drop_element_onto_element(&flexi_module, grp, 1, 3, TRUE);
  _assert_group_count(grp, 2);
  assert_layout("u:2 | i:3,1,4");
  flexi_teardown();

  grp = flexi_build("u:1,2 | i:3,4");
  _model_drop_element_onto_element(&flexi_module, grp, 1, 3, FALSE);
  _assert_group_count(grp, 2);
  assert_layout("u:2 | i:1,3,4");
}

// ...including when the two groups share an operator, where the partition is
// carried entirely by group_start and a lost key would merge or split them.
static void test_drop_between_same_op_groups_keeps_both(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | u:3,4");

  _model_drop_element_onto_element(&flexi_module, grp, 1, 3, TRUE);
  assert_layout("u:2 | u:3,1,4");

  GList *heads = _group_partition_heads(grp);
  assert_int_equal(g_list_length(heads), 2);
  g_list_free(heads);
}

// Dragging the bottom group's target -- the user's report specifically
// mentioned the bottom group as the drop target.
static void test_drop_onto_bottom_group(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3,4");

  _model_drop_element_onto_element(&flexi_module, grp, 3, 1, FALSE);
  assert_layout("u:3,1,2 | i:4");

  GList *heads = _group_partition_heads(grp);
  assert_int_equal(g_list_length(heads), 2);
  g_list_free(heads);
}

// Emptying a group leaves an empty-group placeholder behind, so the group does
// not silently vanish when its last member is dragged out.
static void test_drop_emptying_group_leaves_placeholder(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1 | i:2,3");

  assert_int_equal(g_list_length(flexi_bd.empty_groups), 0);
  _model_drop_element_onto_element(&flexi_module, grp, 1, 2, TRUE);
  assert_layout("i:2,1,3");
  assert_int_equal(g_list_length(flexi_bd.empty_groups), 1);
}

static void test_drop_onto_self_is_rejected(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3");

  assert_false(_model_drop_element_onto_element(&flexi_module, grp, 2, 2, TRUE));
  assert_layout("u:1,2 | i:3");
}

static void test_drop_of_unknown_element_is_rejected(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2");

  assert_false(_model_drop_element_onto_element(&flexi_module, grp, 77, 1, TRUE));
  assert_layout("u:1,2");
}

// ---------------------------------------------------------------------------
// selection state after a gesture
// ---------------------------------------------------------------------------

// A moved element stays selected, and its recorded group follows it to its new
// group -- otherwise the panel highlights the group it came from.
static void test_drop_keeps_element_selected_in_new_group(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3,4");

  _model_drop_element_onto_element(&flexi_module, grp, 1, 3, TRUE);
  assert_int_equal(flexi_bd.panel_selected_formid, 1);
  assert_int_equal(flexi_bd.panel_selected_group_cid, _group_cid_of_form(grp, 1));
  assert_int_equal(flexi_bd.panel_selected_group_cid, 3);
}

// ---------------------------------------------------------------------------
// the selection state machine
//
// The contract is that every reachable state is one click away. Each test
// below is one step of a click sequence a user can actually perform, applied
// through the same decision functions the click handlers use.
// ---------------------------------------------------------------------------

// apply a decision to the fixture's blend_data, the way the real callers do
static void _apply(const dt_masks_panel_sel_t s)
{
  flexi_bd.panel_selected_formid = s.formid;
  flexi_bd.panel_selected_group_cid = s.group_cid;
}

static void _click_element(const dt_mask_id_t id)
{
  _apply(_model_click_element(&flexi_bd, flexi_group(), id));
}

static void _click_group(const dt_mask_id_t cid)
{
  _apply(_model_click_group(&flexi_bd, cid));
}

static void test_click_group_selects_it(void **state)
{
  flexi_build("u:1,2 | i:3");
  _click_group(1);
  assert_int_equal(flexi_bd.panel_selected_group_cid, 1);
  assert_int_equal(flexi_bd.panel_selected_formid, INVALID_MASKID);
}

static void test_click_selected_group_clears_selection(void **state)
{
  flexi_build("u:1,2 | i:3");
  _click_group(1);
  _click_group(1);
  assert_int_equal(flexi_bd.panel_selected_group_cid, INVALID_MASKID);
  assert_int_equal(flexi_bd.panel_selected_formid, INVALID_MASKID);
}

// selecting an element selects its group too -- the two levels are nested,
// not independent
static void test_click_element_selects_element_and_its_group(void **state)
{
  flexi_build("u:1,2 | i:3,4");
  _click_element(4);
  assert_int_equal(flexi_bd.panel_selected_formid, 4);
  assert_int_equal(flexi_bd.panel_selected_group_cid, 3);
}

// the case that motivated the change: deselecting an element must leave its
// GROUP selected, so getting back to the group is not a second click
static void test_click_selected_element_falls_back_to_its_group(void **state)
{
  flexi_build("u:1,2 | i:3,4");
  _click_element(4);
  _click_element(4);
  assert_int_equal(flexi_bd.panel_selected_formid, INVALID_MASKID);
  assert_int_equal(flexi_bd.panel_selected_group_cid, 3);
}

// ...and one more click on that group then clears everything
static void test_element_then_group_reaches_empty_selection(void **state)
{
  flexi_build("u:1,2 | i:3,4");
  _click_element(4);
  _click_element(4);      // -> group 3
  _click_group(3);        // -> nothing
  assert_int_equal(flexi_bd.panel_selected_formid, INVALID_MASKID);
  assert_int_equal(flexi_bd.panel_selected_group_cid, INVALID_MASKID);
}

static void test_click_other_element_switches_directly(void **state)
{
  flexi_build("u:1,2 | i:3,4");
  _click_element(4);
  _click_element(1); // a different group's element, in one click
  assert_int_equal(flexi_bd.panel_selected_formid, 1);
  assert_int_equal(flexi_bd.panel_selected_group_cid, 1);
}

static void test_click_other_group_switches_directly(void **state)
{
  flexi_build("u:1,2 | i:3,4");
  _click_group(1);
  _click_group(3);
  assert_int_equal(flexi_bd.panel_selected_group_cid, 3);
}

// ---------------------------------------------------------------------------
// operator normalisation
// ---------------------------------------------------------------------------

// the base (bottom) point has nothing below it, so a break marker there is
// meaningless -- one arriving via a reorder must be cleared, or the partition
// reads wrong from the bottom up
static void test_normalize_clears_break_on_base_point(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2");
  ((dt_masks_point_group_t *)grp->points->data)->group_start = 1;
  _normalize_group_operators(grp);
  assert_int_equal(((dt_masks_point_group_t *)grp->points->data)->group_start, 0);
}

// back-compat: a point carrying no operator bit at all reads as union
static void test_normalize_defaults_missing_operator_to_union(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2");
  dt_masks_point_group_t *pt = _group_point(grp, 2);
  pt->state &= ~DT_MASKS_STATE_OP;
  _normalize_group_operators(grp);
  assert_int_equal(_eff_group_op(pt->state), DT_MASKS_STATE_UNION);
}

// bypass is a modifier layered on an operator, not an operator -- a bypassed
// group must keep the operator it goes back to
static void test_normalize_keeps_operator_under_bypass(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | d:3");
  dt_masks_point_group_t *pt = _group_point(grp, 3);
  pt->state |= DT_MASKS_STATE_OP_BYPASS;
  _normalize_group_operators(grp);
  assert_int_equal(pt->state & DT_MASKS_STATE_OP_COMBINE, DT_MASKS_STATE_DIFFERENCE);
}

// normalising must not repartition: it reads each point's neighbour state, so
// mutating operators inside the same loop can misdetect a run boundary
static void test_normalize_preserves_partition(void **state)
{
  flexi_build("u:1,2 | u:3,4 | i:5");
  _normalize_group_operators(flexi_group());
  assert_layout("u:1,2 | u:3,4 | i:5");
}

// ---------------------------------------------------------------------------
// solo / mute primitives (dt_masks_group_set_state / _isolate_state)
// ---------------------------------------------------------------------------

static gboolean _hidden(const dt_mask_id_t fid)
{
  return (_group_point(flexi_group(), fid)->state & DT_MASKS_STATE_HIDDEN) != 0;
}

static void test_isolate_state_hides_everything_else(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3,4");
  GList *keep = g_list_prepend(NULL, GINT_TO_POINTER(3));

  dt_masks_group_isolate_state(grp, keep, DT_MASKS_STATE_HIDDEN);
  g_list_free(keep);

  assert_true(_hidden(1));
  assert_true(_hidden(2));
  assert_false(_hidden(3));
  assert_true(_hidden(4));
}

// the inversion that matters: a NULL keep-list means "solo off" -- clear the
// bit everywhere -- NOT "keep nothing", which would hide every element
static void test_isolate_state_null_list_clears_everywhere(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3");
  for(GList *l = grp->points; l; l = g_list_next(l))
    ((dt_masks_point_group_t *)l->data)->state |= DT_MASKS_STATE_HIDDEN;

  dt_masks_group_isolate_state(grp, NULL, DT_MASKS_STATE_HIDDEN);

  assert_false(_hidden(1));
  assert_false(_hidden(2));
  assert_false(_hidden(3));
}

static void test_isolate_state_soloing_a_whole_group(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3,4");
  GList *keep = g_list_prepend(NULL, GINT_TO_POINTER(4));
  keep = g_list_prepend(keep, GINT_TO_POINTER(3));

  dt_masks_group_isolate_state(grp, keep, DT_MASKS_STATE_HIDDEN);
  g_list_free(keep);

  assert_true(_hidden(1));
  assert_true(_hidden(2));
  assert_false(_hidden(3));
  assert_false(_hidden(4));
}

static void test_set_state_targets_only_listed_members(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3");
  GList *ids = g_list_prepend(NULL, GINT_TO_POINTER(2));

  dt_masks_group_set_state(grp, ids, DT_MASKS_STATE_HIDDEN, TRUE);
  assert_false(_hidden(1));
  assert_true(_hidden(2));
  assert_false(_hidden(3));

  dt_masks_group_set_state(grp, ids, DT_MASKS_STATE_HIDDEN, FALSE);
  assert_false(_hidden(2));
  g_list_free(ids);
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_teardown(test_layout_roundtrip, _teardown),
    cmocka_unit_test_teardown(test_adjacent_same_op_groups_stay_separate, _teardown),
    cmocka_unit_test_teardown(test_cid_of_form_is_run_head, _teardown),
    cmocka_unit_test_teardown(test_selected_group_formids, _teardown),
    cmocka_unit_test_teardown(test_keys_survive_intra_group_reorder, _teardown),
    cmocka_unit_test_teardown(test_keys_absent_member_joins_group_below, _teardown),
    cmocka_unit_test_teardown(test_drop_element_into_other_group, _teardown),
    cmocka_unit_test_teardown(test_drop_element_below_target, _teardown),
    cmocka_unit_test_teardown(test_drop_adopts_target_operator, _teardown),
    cmocka_unit_test_teardown(test_drop_between_groups_never_creates_a_third, _teardown),
    cmocka_unit_test_teardown(test_drop_between_same_op_groups_keeps_both, _teardown),
    cmocka_unit_test_teardown(test_drop_onto_bottom_group, _teardown),
    cmocka_unit_test_teardown(test_drop_emptying_group_leaves_placeholder, _teardown),
    cmocka_unit_test_teardown(test_drop_onto_self_is_rejected, _teardown),
    cmocka_unit_test_teardown(test_drop_of_unknown_element_is_rejected, _teardown),
    cmocka_unit_test_teardown(test_drop_keeps_element_selected_in_new_group, _teardown),
    cmocka_unit_test_teardown(test_click_group_selects_it, _teardown),
    cmocka_unit_test_teardown(test_click_selected_group_clears_selection, _teardown),
    cmocka_unit_test_teardown(test_click_element_selects_element_and_its_group, _teardown),
    cmocka_unit_test_teardown(test_click_selected_element_falls_back_to_its_group, _teardown),
    cmocka_unit_test_teardown(test_element_then_group_reaches_empty_selection, _teardown),
    cmocka_unit_test_teardown(test_click_other_element_switches_directly, _teardown),
    cmocka_unit_test_teardown(test_click_other_group_switches_directly, _teardown),
    cmocka_unit_test_teardown(test_normalize_clears_break_on_base_point, _teardown),
    cmocka_unit_test_teardown(test_normalize_defaults_missing_operator_to_union, _teardown),
    cmocka_unit_test_teardown(test_normalize_keeps_operator_under_bypass, _teardown),
    cmocka_unit_test_teardown(test_normalize_preserves_partition, _teardown),
    cmocka_unit_test_teardown(test_isolate_state_hides_everything_else, _teardown),
    cmocka_unit_test_teardown(test_isolate_state_null_list_clears_everywhere, _teardown),
    cmocka_unit_test_teardown(test_isolate_state_soloing_a_whole_group, _teardown),
    cmocka_unit_test_teardown(test_set_state_targets_only_listed_members, _teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
