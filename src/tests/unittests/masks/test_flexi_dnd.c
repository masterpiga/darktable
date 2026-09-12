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

// The drag-and-drop paths other than element-onto-element (which lives in
// test_flexi_model.c alongside the grouping primitives it exercises), and the
// group operations the panel's menus make: adding, emptying, deleting and
// merging a group.
//
// The panel offers four distinct drops, each with its own target type so they
// cannot interfere: an element onto another element, an element onto a group
// (empty or not), a whole same-kind cluster onto either of those, and a whole
// group reordered against another group. A group is its marker followed by its
// elements, so every one of them is a move within one list: what they have to
// get right is where the dragged thing lands, and that no group's settings or
// place change on the way.

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

static void _assert_group_count(const int expect)
{
  GList *heads = _group_partition_heads(flexi_group());
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

static void _name_group(dt_masks_form_t *grp, const dt_mask_id_t cid, const char *name)
{
  g_strlcpy(_group_point(grp, cid)->name, name, sizeof(_group_point(grp, cid)->name));
}

// ---------------------------------------------------------------------------
// element onto a group
// ---------------------------------------------------------------------------

// the element joins the group, landing on top of it
static void test_element_onto_group_header(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3,4");
  assert_true(_model_drop_element_onto_group(&flexi_module, grp, 1, FLEXI_GID(1)));
  assert_layout("u:2 | i:3,4,1");
  _assert_group_count(2);
}

// the group can also be named by any of its elements
static void test_element_onto_group_named_by_a_member(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3,4");
  assert_true(_model_drop_element_onto_group(&flexi_module, grp, 1, 3));
  assert_layout("u:2 | i:3,4,1");
}

// the group's operator is its marker's: an element that joins reads it
static void test_element_onto_group_header_adopts_operator(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | d:3");
  _model_drop_element_onto_group(&flexi_module, grp, 2, FLEXI_GID(1));
  assert_int_equal(flexi_group_op_of(2), DT_MASKS_STATE_DIFFERENCE);
}

// dropping an element on the header of the group it is already in is a no-op,
// not a reorder -- otherwise a stray drag silently shuffles the group
static void test_element_onto_its_own_group_header_is_a_noop(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2,3 | i:4");
  assert_false(_model_drop_element_onto_group(&flexi_module, grp, 2, FLEXI_GID(0)));
  assert_layout("u:1,2,3 | i:4");
}

// the group an element leaves stays, empty, where it was: groups are only
// removed explicitly
static void test_element_onto_group_header_leaves_its_group(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1 | i:2,3");
  _model_drop_element_onto_group(&flexi_module, grp, 1, FLEXI_GID(1));
  assert_layout("[u] | i:2,3,1");
  _assert_group_count(2);
}

static void test_element_onto_invalid_group_is_rejected(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2");
  assert_false(_model_drop_element_onto_group(&flexi_module, grp, 1, INVALID_MASKID));
  assert_false(_model_drop_element_onto_group(&flexi_module, grp, 1, 12345));
  assert_layout("u:1,2");
}

// a group's marker is not an element: it cannot be dropped into another group
static void test_marker_is_not_moved_as_an_element(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1 | i:2");
  assert_false(_model_drop_element_onto_group(&flexi_module, grp, FLEXI_GID(0),
                                              FLEXI_GID(1)));
  assert_layout("u:1 | i:2");
}

// ---------------------------------------------------------------------------
// element onto an empty group
// ---------------------------------------------------------------------------

// an empty group is a group like any other: the element lands in it
static void test_element_fills_an_empty_group(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2,3 | [d]");
  assert_true(_model_drop_element_onto_group(&flexi_module, grp, 3, FLEXI_GID(1)));
  assert_layout("u:1,2 | d:3");
}

static void test_element_fills_an_empty_bottom_group(void **state)
{
  dt_masks_form_t *grp = flexi_build("[i] | u:1,2,3");
  assert_true(_model_drop_element_onto_group(&flexi_module, grp, 3, FLEXI_GID(0)));
  assert_layout("i:3 | u:1,2");
}

// the sole member of a group dropped onto the empty group below it: the two
// swap contents, not places
static void test_filling_the_group_below_keeps_their_order(void **state)
{
  dt_masks_form_t *grp = flexi_build("[d] | u:1 | i:2");
  assert_true(_model_drop_element_onto_group(&flexi_module, grp, 1, FLEXI_GID(0)));
  assert_layout("d:1 | [u] | i:2");
}

// ...and the same move upwards
static void test_filling_the_group_above_keeps_their_order(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1 | [d] | i:2");
  assert_true(_model_drop_element_onto_group(&flexi_module, grp, 1, FLEXI_GID(1)));
  assert_layout("[u] | d:1 | i:2");
}

// filling a group keeps the number it showed while empty: its id did not change
static void test_filling_a_group_keeps_its_number(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2,3 | [d]");
  flexi_set_ordinal(FLEXI_GID(1), 4);
  _model_drop_element_onto_group(&flexi_module, grp, 3, FLEXI_GID(1));
  assert_int_equal(flexi_get_ordinal(_group_cid_of_form(grp, 3)), 4);
}

// ---------------------------------------------------------------------------
// moves never change a group's settings
// ---------------------------------------------------------------------------

// a group's name, and every other setting, lives on its marker: moving
// elements between groups renames none of them
static void test_moving_elements_renames_no_group(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1 | [d] | i:2");
  _name_group(grp, FLEXI_GID(0), "sky");
  _name_group(grp, FLEXI_GID(2), "trees");

  assert_true(_model_drop_element_onto_group(&flexi_module, grp, 1, FLEXI_GID(1)));
  assert_true(_model_drop_element_onto_element(&flexi_module, grp, 2, 1, TRUE));
  assert_layout("[u] | d:1,2 | [i]");
  assert_string_equal(_group_point(grp, FLEXI_GID(0))->name, "sky");
  assert_string_equal(_group_point(grp, FLEXI_GID(1))->name, "");
  assert_string_equal(_group_point(grp, FLEXI_GID(2))->name, "trees");
}

static void test_moving_elements_leaves_every_group_setting(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1 | i:2,3");
  dt_masks_point_group_t *from = _group_point(grp, FLEXI_GID(0));
  dt_masks_point_group_t *to = _group_point(grp, FLEXI_GID(1));
  from->group_opacity = 0.8f;
  to->state |= DT_MASKS_STATE_SCREEN;
  to->group_opacity = 0.5f;
  to->refinement = (dt_masks_refinement_t){ .enabled = DT_MASKS_REFINE_GROUP,
                                            .blur_radius = 3.0f };
  const dt_masks_point_group_t from_before = *from, to_before = *to;

  assert_true(_model_drop_element_onto_element(&flexi_module, grp, 1, 2, FALSE));
  assert_int_equal(_group_cid_of_form(grp, 1), FLEXI_GID(1));
  assert_memory_equal(from, &from_before, sizeof(from_before));
  assert_memory_equal(to, &to_before, sizeof(to_before));
}

// an element's own refinement is its own, whatever group it is in
static void test_moving_keeps_the_elements_own_refinement(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,4 | i:2,3");
  _group_point(grp, 1)->refinement = (dt_masks_refinement_t){
    .enabled = DT_MASKS_REFINE_ELEMENT, .blur_radius = 7.0f };

  assert_true(_model_drop_element_onto_element(&flexi_module, grp, 1, 2, TRUE));
  assert_int_equal(_group_point(grp, 1)->refinement.enabled, DT_MASKS_REFINE_ELEMENT);
  assert_float_equal(_group_point(grp, 1)->refinement.blur_radius, 7.0f, 1e-6);
}

// a row drop moving the bottom group's sole member up, under the first element
// of the group above: that group keeps its place, as does the one emptied
static void test_row_drop_up_from_a_sole_member_keeps_group_order(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1 | i:2,3");
  assert_true(_model_drop_element_onto_element(&flexi_module, grp, 1, 2, FALSE));
  assert_layout("[u] | i:1,2,3");
}

// ---------------------------------------------------------------------------
// whole-cluster drops
// ---------------------------------------------------------------------------

static GList *_ids(const int a, const int b)
{
  GList *l = g_list_append(NULL, GINT_TO_POINTER(a));
  return g_list_append(l, GINT_TO_POINTER(b));
}

// every member moves together, as one contiguous block, keeping their relative
// order
static void test_cluster_onto_group_header(void **state)
{
  flexi_build("u:1,2,3 | i:4");
  GList *ids = _ids(2, 3);

  assert_true(_masks_cluster_move(&flexi_module, ids, FLEXI_GID(1), TRUE, FALSE));
  g_list_free(ids);
  assert_layout("u:1 | i:4,2,3");
  _assert_group_count(2);
}

static void test_cluster_onto_element_row(void **state)
{
  flexi_build("u:1,2,3 | i:4,5");
  GList *ids = _ids(2, 3);

  assert_true(_masks_cluster_move(&flexi_module, ids, 4, FALSE, TRUE));
  g_list_free(ids);
  assert_layout("u:1 | i:4,2,3,5");
  _assert_group_count(2);
}

static void test_cluster_onto_an_empty_group(void **state)
{
  flexi_build("u:1,2,3 | [d]");
  GList *ids = _ids(2, 3);

  assert_true(_masks_cluster_move(&flexi_module, ids, FLEXI_GID(1), TRUE, FALSE));
  g_list_free(ids);
  assert_layout("u:1 | d:2,3");
}

// moving out every member of a group leaves it, empty, where it was
static void test_cluster_emptying_group_leaves_it(void **state)
{
  flexi_build("u:1,2 | i:3,4");
  GList *ids = _ids(3, 4);

  _masks_cluster_move(&flexi_module, ids, FLEXI_GID(0), TRUE, FALSE);
  g_list_free(ids);
  assert_layout("u:1,2,3,4 | [i]");
}

static void test_cluster_move_with_no_members_is_rejected(void **state)
{
  flexi_build("u:1,2");
  assert_false(_masks_cluster_move(&flexi_module, NULL, FLEXI_GID(0), TRUE, FALSE));
  assert_layout("u:1,2");
}

// ---------------------------------------------------------------------------
// whole-group reorder
// ---------------------------------------------------------------------------

static void test_reorder_group_above_another(void **state)
{
  flexi_build("u:1,2 | i:3 | d:4");
  // move the union group above the difference group
  assert_true(_masks_reorder_groups(&flexi_module, FLEXI_GID(0), FLEXI_GID(2), TRUE));
  assert_layout("i:3 | d:4 | u:1,2");
  _assert_group_count(3);
}

static void test_reorder_group_below_another(void **state)
{
  flexi_build("u:1,2 | i:3 | d:4");
  assert_true(_masks_reorder_groups(&flexi_module, FLEXI_GID(2), FLEXI_GID(0), FALSE));
  assert_layout("d:4 | u:1,2 | i:3");
  _assert_group_count(3);
}

// a group moves as a unit -- its members keep their relative order
static void test_reorder_keeps_members_together_and_ordered(void **state)
{
  flexi_build("u:1,2,3 | i:4");
  _masks_reorder_groups(&flexi_module, FLEXI_GID(0), FLEXI_GID(1), TRUE);
  assert_layout("i:4 | u:1,2,3");
}

// two same-operator groups that end up adjacent stay two groups
static void test_reorder_does_not_merge_same_op_neighbours(void **state)
{
  flexi_build("u:1,2 | i:3 | u:4");
  _masks_reorder_groups(&flexi_module, FLEXI_GID(1), FLEXI_GID(2), TRUE);
  assert_layout("u:1,2 | u:4 | i:3");
  _assert_group_count(3);
}

static void test_reorder_onto_itself_is_rejected(void **state)
{
  flexi_build("u:1,2 | i:3");
  assert_false(_masks_reorder_groups(&flexi_module, FLEXI_GID(0), FLEXI_GID(0), TRUE));
  assert_layout("u:1,2 | i:3");
}

// a group is named by its own id, not by an element's
static void test_reorder_with_element_id_is_rejected(void **state)
{
  flexi_build("u:1,2 | i:3");
  assert_false(_masks_reorder_groups(&flexi_module, 2, FLEXI_GID(1), TRUE));
  assert_layout("u:1,2 | i:3");
}

// empty groups have their place in the same order, and move like any group
static void test_reorder_past_an_empty_group(void **state)
{
  flexi_build("u:1,2 | [d] | i:3");
  assert_true(_masks_reorder_groups(&flexi_module, FLEXI_GID(0), FLEXI_GID(2), TRUE));
  assert_layout("[d] | i:3 | u:1,2");
}

static void test_empty_group_can_be_reordered(void **state)
{
  flexi_build("[d] | u:1,2 | i:3");
  assert_true(_masks_reorder_groups(&flexi_module, FLEXI_GID(0), FLEXI_GID(2), TRUE));
  assert_layout("u:1,2 | i:3 | [d]");
}

// ---------------------------------------------------------------------------
// adding, emptying, deleting and merging groups
// ---------------------------------------------------------------------------

static void test_add_group_above_a_group(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1 | i:2");
  const dt_mask_id_t cid = _model_add_group(grp, DT_MASKS_STATE_DIFFERENCE, FLEXI_GID(0),
                                            FALSE);
  assert_layout("u:1 | [d] | i:2");
  assert_true(dt_masks_point_is_marker(_group_point(grp, cid)));
}

static void test_add_group_below_a_group(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1 | i:2");
  _model_add_group(grp, DT_MASKS_STATE_DIFFERENCE, FLEXI_GID(1), TRUE);
  assert_layout("u:1 | [d] | i:2");
  _model_add_group(grp, DT_MASKS_STATE_SUM, FLEXI_GID(0), TRUE);
  assert_layout("[s] | u:1 | [d] | i:2");
}

// with no group to add it next to, a group goes on top, or at the bottom
static void test_add_group_with_no_group_named(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1");
  _model_add_group(grp, DT_MASKS_STATE_DIFFERENCE, INVALID_MASKID, FALSE);
  _model_add_group(grp, DT_MASKS_STATE_SUM, INVALID_MASKID, TRUE);
  assert_layout("[s] | u:1 | [d]");
}

// a new group is live, whatever it was asked for
static void test_added_group_is_never_bypassed(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1");
  const dt_mask_id_t cid =
    _model_add_group(grp, DT_MASKS_STATE_DIFFERENCE | DT_MASKS_STATE_OP_BYPASS,
                     INVALID_MASKID, FALSE);
  assert_false(_group_point(grp, cid)->state & DT_MASKS_STATE_OP_BYPASS);
}

static void test_emptying_a_group_keeps_it(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1 | i:2,3 | d:4");
  _name_group(grp, FLEXI_GID(1), "sky");
  GList *gone = _model_empty_group(grp, FLEXI_GID(1));
  assert_int_equal(g_list_length(gone), 2);
  g_list_free(gone);
  assert_layout("u:1 | [i] | d:4");
  assert_string_equal(_group_point(grp, FLEXI_GID(1))->name, "sky");
}

static void test_deleting_a_group_takes_its_elements(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1 | i:2,3 | d:4");
  GList *gone = _model_delete_group(grp, FLEXI_GID(1));
  assert_int_equal(g_list_length(gone), 2);
  g_list_free(gone);
  assert_layout("u:1 | d:4");
  assert_null(_group_point(grp, FLEXI_GID(1)));
}

// the bottom group's elements join the group above when it is deleted: that
// one becomes the bottom group
static void test_deleting_the_bottom_group(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1 | i:2");
  g_list_free(_model_delete_group(grp, FLEXI_GID(0)));
  assert_layout("i:2");
}

// merging a group down makes its elements the group below's
static void test_merge_group_down(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1 | i:2,3 | d:4");
  assert_true(_model_merge_group_down(grp, FLEXI_GID(1)));
  assert_layout("u:1,2,3 | d:4");
  assert_int_equal(flexi_group_op_of(2), DT_MASKS_STATE_UNION);
}

static void test_merge_of_the_bottom_group_is_rejected(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1 | i:2");
  assert_false(_model_merge_group_down(grp, FLEXI_GID(0)));
  assert_layout("u:1 | i:2");
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_teardown(test_element_onto_group_header, _teardown),
    cmocka_unit_test_teardown(test_element_onto_group_named_by_a_member, _teardown),
    cmocka_unit_test_teardown(test_element_onto_group_header_adopts_operator, _teardown),
    cmocka_unit_test_teardown(test_element_onto_its_own_group_header_is_a_noop, _teardown),
    cmocka_unit_test_teardown(test_element_onto_group_header_leaves_its_group, _teardown),
    cmocka_unit_test_teardown(test_element_onto_invalid_group_is_rejected, _teardown),
    cmocka_unit_test_teardown(test_marker_is_not_moved_as_an_element, _teardown),
    cmocka_unit_test_teardown(test_element_fills_an_empty_group, _teardown),
    cmocka_unit_test_teardown(test_element_fills_an_empty_bottom_group, _teardown),
    cmocka_unit_test_teardown(test_filling_the_group_below_keeps_their_order, _teardown),
    cmocka_unit_test_teardown(test_filling_the_group_above_keeps_their_order, _teardown),
    cmocka_unit_test_teardown(test_filling_a_group_keeps_its_number, _teardown),
    cmocka_unit_test_teardown(test_moving_elements_renames_no_group, _teardown),
    cmocka_unit_test_teardown(test_moving_elements_leaves_every_group_setting, _teardown),
    cmocka_unit_test_teardown(test_moving_keeps_the_elements_own_refinement, _teardown),
    cmocka_unit_test_teardown(test_row_drop_up_from_a_sole_member_keeps_group_order, _teardown),
    cmocka_unit_test_teardown(test_cluster_onto_group_header, _teardown),
    cmocka_unit_test_teardown(test_cluster_onto_element_row, _teardown),
    cmocka_unit_test_teardown(test_cluster_onto_an_empty_group, _teardown),
    cmocka_unit_test_teardown(test_cluster_emptying_group_leaves_it, _teardown),
    cmocka_unit_test_teardown(test_cluster_move_with_no_members_is_rejected, _teardown),
    cmocka_unit_test_teardown(test_reorder_group_above_another, _teardown),
    cmocka_unit_test_teardown(test_reorder_group_below_another, _teardown),
    cmocka_unit_test_teardown(test_reorder_keeps_members_together_and_ordered, _teardown),
    cmocka_unit_test_teardown(test_reorder_does_not_merge_same_op_neighbours, _teardown),
    cmocka_unit_test_teardown(test_reorder_onto_itself_is_rejected, _teardown),
    cmocka_unit_test_teardown(test_reorder_with_element_id_is_rejected, _teardown),
    cmocka_unit_test_teardown(test_reorder_past_an_empty_group, _teardown),
    cmocka_unit_test_teardown(test_empty_group_can_be_reordered, _teardown),
    cmocka_unit_test_teardown(test_add_group_above_a_group, _teardown),
    cmocka_unit_test_teardown(test_add_group_below_a_group, _teardown),
    cmocka_unit_test_teardown(test_add_group_with_no_group_named, _teardown),
    cmocka_unit_test_teardown(test_added_group_is_never_bypassed, _teardown),
    cmocka_unit_test_teardown(test_emptying_a_group_keeps_it, _teardown),
    cmocka_unit_test_teardown(test_deleting_a_group_takes_its_elements, _teardown),
    cmocka_unit_test_teardown(test_deleting_the_bottom_group, _teardown),
    cmocka_unit_test_teardown(test_merge_group_down, _teardown),
    cmocka_unit_test_teardown(test_merge_of_the_bottom_group_is_rejected, _teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
