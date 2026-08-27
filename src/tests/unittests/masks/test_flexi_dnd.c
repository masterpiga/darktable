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
// test_flexi_model.c alongside the grouping primitives it exercises).
//
// The panel offers five distinct drops, each with its own target type so they
// cannot interfere: an element onto another element, an element onto a group
// header, an element onto a staged (empty) group, a whole same-kind cluster
// onto either of those, and a whole group reordered against another group.
// They share the snapshot/re-stamp machinery but differ in where the dragged
// thing lands and what it adopts, which is exactly where they can drift apart.
//
// Group reorder is the intricate one: it rebuilds the unified visual order of
// real runs *and* staged empties, splices the dragged group into it, then
// re-derives both the points order and every empty group's anchor from the
// result.

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

// ---------------------------------------------------------------------------
// element onto a group header
// ---------------------------------------------------------------------------

// the element joins the target's run, landing on top of it and adopting its
// operator
static void test_element_onto_group_header(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3,4");
  assert_true(_model_drop_element_onto_group(&flexi_module, grp, 1, 3));
  assert_layout("u:2 | i:3,4,1");
  _assert_group_count(2);
}

static void test_element_onto_group_header_adopts_operator(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | d:3");
  _model_drop_element_onto_group(&flexi_module, grp, 2, 3);
  assert_int_equal(_eff_group_op(_group_point(grp, 2)->state),
                   DT_MASKS_STATE_DIFFERENCE);
}

// dropping an element on the header of the group it is already in is a no-op,
// not a reorder -- otherwise a stray drag silently shuffles the run
static void test_element_onto_its_own_group_header_is_a_noop(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2,3 | i:4");
  assert_false(_model_drop_element_onto_group(&flexi_module, grp, 2, 1));
  assert_layout("u:1,2,3 | i:4");
}

// the base shape must stay at the bottom of the list: the drop clamps its
// insertion index so a move into the bottom group cannot displace it
static void test_element_onto_group_header_never_displaces_the_base(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1 | i:2,3");
  _model_drop_element_onto_group(&flexi_module, grp, 2, 1);
  // 1 remains the bottom-most point
  assert_int_equal(((dt_masks_point_group_t *)grp->points->data)->formid, 1);
}

static void test_element_onto_group_header_leaves_placeholder(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1 | i:2,3");
  assert_int_equal(g_list_length(flexi_bd.empty_groups), 0);
  _model_drop_element_onto_group(&flexi_module, grp, 1, 2);
  assert_int_equal(g_list_length(flexi_bd.empty_groups), 1);
}

static void test_element_onto_invalid_group_is_rejected(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2");
  assert_false(_model_drop_element_onto_group(&flexi_module, grp, 1,
                                              INVALID_MASKID));
  assert_layout("u:1,2");
}

// ---------------------------------------------------------------------------
// element onto a staged (empty) group -- realisation
// ---------------------------------------------------------------------------

// the staged group becomes real: the element adopts its operator, the
// placeholder is consumed, and the element heads its own new run
static void test_element_realizes_an_empty_group(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2,3");
  dt_masks_empty_group_t *eg = flexi_add_empty(DT_MASKS_STATE_DIFFERENCE, 1);

  assert_true(_model_drop_element_onto_empty(&flexi_module, grp, 3, eg));
  assert_layout("u:1,2 | d:3");
  assert_int_equal(g_list_length(flexi_bd.empty_groups), 0);
}

// a staged group carries a number while it has no members; realising it must
// keep that number rather than renumbering the group
static void test_realizing_preserves_the_staged_ordinal(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2,3");
  dt_masks_empty_group_t *eg = flexi_add_empty(DT_MASKS_STATE_DIFFERENCE, 1);
  eg->ordinal = 4;

  _model_drop_element_onto_empty(&flexi_module, grp, 3, eg);
  assert_int_equal(flexi_get_ordinal(3), 4);
}

// ...and its name, if it was named while still empty
static void test_realizing_preserves_the_staged_name(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2,3");
  dt_masks_empty_group_t *eg = flexi_add_empty(DT_MASKS_STATE_INTERSECTION, 1);
  eg->name = g_strdup("sky");

  _model_drop_element_onto_empty(&flexi_module, grp, 3, eg);
  assert_string_equal(_group_point(grp, 3)->name, "sky");
}

// ...and refinement staged before the group had anywhere to put it
static void test_realizing_adopts_staged_refinement(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2,3");
  dt_masks_empty_group_t *eg = flexi_add_empty(DT_MASKS_STATE_INTERSECTION, 1);
  eg->refinement.enabled = DT_MASKS_REFINE_GROUP;
  eg->refinement.blur_radius = 5.0f;

  _model_drop_element_onto_empty(&flexi_module, grp, 3, eg);
  assert_int_equal(_group_point(grp, 3)->refinement.enabled, DT_MASKS_REFINE_GROUP);
  assert_float_equal(_group_point(grp, 3)->refinement.blur_radius, 5.0f, 1e-6);
}

// an unanchored staged group (below_fid INVALID) realises at the very bottom
static void test_realizing_unanchored_empty_lands_at_the_bottom(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2,3");
  dt_masks_empty_group_t *eg = flexi_add_empty(DT_MASKS_STATE_INTERSECTION,
                                               INVALID_MASKID);

  _model_drop_element_onto_empty(&flexi_module, grp, 3, eg);
  assert_int_equal(((dt_masks_point_group_t *)grp->points->data)->formid, 3);
}

// dropping onto a placeholder the panel does not own must be refused rather
// than realising a group that is not in the list
static void test_realizing_an_unregistered_empty_is_rejected(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2");
  dt_masks_empty_group_t stray = { .op = DT_MASKS_STATE_UNION,
                                   .below_fid = INVALID_MASKID,
                                   .opacity = 1.0f };
  assert_false(_model_drop_element_onto_empty(&flexi_module, grp, 1, &stray));
  assert_layout("u:1,2");
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

  assert_true(_masks_cluster_move(&flexi_module, ids, 4, TRUE, FALSE));
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

// members adopt the target group's operator, like a single element does
static void test_cluster_adopts_target_operator(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2,3 | d:4");
  GList *ids = _ids(2, 3);

  _masks_cluster_move(&flexi_module, ids, 4, TRUE, FALSE);
  g_list_free(ids);
  assert_int_equal(_eff_group_op(_group_point(grp, 2)->state),
                   DT_MASKS_STATE_DIFFERENCE);
  assert_int_equal(_eff_group_op(_group_point(grp, 3)->state),
                   DT_MASKS_STATE_DIFFERENCE);
}

// moving out every member of a run leaves a placeholder, exactly as emptying a
// group one element at a time does
static void test_cluster_emptying_group_leaves_placeholder(void **state)
{
  flexi_build("u:1,2 | i:3,4");
  GList *ids = _ids(3, 4);

  _masks_cluster_move(&flexi_module, ids, 1, TRUE, FALSE);
  g_list_free(ids);
  assert_layout("u:1,2,3,4");
  assert_int_equal(g_list_length(flexi_bd.empty_groups), 1);
}

static void test_cluster_move_with_no_members_is_rejected(void **state)
{
  flexi_build("u:1,2");
  assert_false(_masks_cluster_move(&flexi_module, NULL, 1, TRUE, FALSE));
  assert_layout("u:1,2");
}

// ---------------------------------------------------------------------------
// whole-group reorder
// ---------------------------------------------------------------------------

static void test_reorder_group_above_another(void **state)
{
  flexi_build("u:1,2 | i:3 | d:4");
  // move the union group above the difference group
  assert_true(_masks_reorder_groups(&flexi_module, FALSE, 1, NULL,
                                    FALSE, 4, NULL, TRUE));
  assert_layout("i:3 | d:4 | u:1,2");
  _assert_group_count(3);
}

static void test_reorder_group_below_another(void **state)
{
  flexi_build("u:1,2 | i:3 | d:4");
  assert_true(_masks_reorder_groups(&flexi_module, FALSE, 4, NULL,
                                    FALSE, 1, NULL, FALSE));
  assert_layout("d:4 | u:1,2 | i:3");
  _assert_group_count(3);
}

// a group moves as a unit -- its members keep their relative order
static void test_reorder_keeps_members_together_and_ordered(void **state)
{
  flexi_build("u:1,2,3 | i:4");
  _masks_reorder_groups(&flexi_module, FALSE, 1, NULL, FALSE, 4, NULL, TRUE);
  assert_layout("i:4 | u:1,2,3");
}

// reordering must not merge two same-operator groups that end up adjacent --
// the partition is carried by group_start, and the re-derivation must preserve it
static void test_reorder_does_not_merge_same_op_neighbours(void **state)
{
  flexi_build("u:1,2 | i:3 | u:4");
  _masks_reorder_groups(&flexi_module, FALSE, 3, NULL, FALSE, 4, NULL, TRUE);
  assert_layout("u:1,2 | u:4 | i:3");
  _assert_group_count(3);
}

static void test_reorder_onto_itself_is_rejected(void **state)
{
  flexi_build("u:1,2 | i:3");
  assert_false(_masks_reorder_groups(&flexi_module, FALSE, 1, NULL,
                                     FALSE, 1, NULL, TRUE));
  assert_layout("u:1,2 | i:3");
}

// passing a member id that is not its run's head must not match a group -- the
// contract is that *_cid is the head formid
static void test_reorder_with_non_head_id_is_rejected(void **state)
{
  flexi_build("u:1,2 | i:3");
  assert_false(_masks_reorder_groups(&flexi_module, FALSE, 2, NULL,
                                     FALSE, 3, NULL, TRUE));
  assert_layout("u:1,2 | i:3");
}

// ---------------------------------------------------------------------------
// reorder with staged groups in the mix
// ---------------------------------------------------------------------------

// staged groups occupy positions in the same visual order as real ones, so a
// reorder has to place them and re-derive their anchors
static void test_visual_order_interleaves_staged_groups(void **state)
{
  flexi_build("u:1,2 | i:3");
  flexi_add_empty(DT_MASKS_STATE_DIFFERENCE, 1); // anchored onto the union run
  assert_order("u:1,2 | [d] | i:3");
}

static void test_unanchored_staged_group_sits_at_the_bottom(void **state)
{
  flexi_build("u:1,2 | i:3");
  flexi_add_empty(DT_MASKS_STATE_DIFFERENCE, INVALID_MASKID);
  assert_order("[d] | u:1,2 | i:3");
}

// moving a real group past a staged one must re-anchor the staged group rather
// than leave it pointing at a run that is no longer below it
static void test_reorder_reanchors_staged_group(void **state)
{
  flexi_build("u:1,2 | i:3");
  dt_masks_empty_group_t *eg = flexi_add_empty(DT_MASKS_STATE_DIFFERENCE, 1);
  assert_order("u:1,2 | [d] | i:3");

  // move the union group to the top; the staged group must follow the order,
  // not the formid it happened to be anchored to
  assert_true(_masks_reorder_groups(&flexi_module, FALSE, 1, NULL,
                                    FALSE, 3, NULL, TRUE));
  // whatever the resulting order is, every staged group's anchor must name a
  // run that really sits below it
  GList *order = _masks_visual_group_order(&flexi_module);
  dt_mask_id_t seen_below = INVALID_MASKID;
  for(GList *l = order; l; l = g_list_next(l))
  {
    const _dt_masks_order_item_t *it = l->data;
    if(it->is_empty)
      assert_int_equal(it->eg->below_fid, seen_below);
    else
      seen_below = it->cid;
  }
  g_list_free_full(order, g_free);
  (void)eg;
}

// a staged group can itself be dragged, so it is not stuck where it was created
static void test_staged_group_can_be_reordered(void **state)
{
  flexi_build("u:1,2 | i:3");
  dt_masks_empty_group_t *eg = flexi_add_empty(DT_MASKS_STATE_DIFFERENCE,
                                               INVALID_MASKID);
  assert_order("[d] | u:1,2 | i:3");

  assert_true(_masks_reorder_groups(&flexi_module, TRUE, INVALID_MASKID, eg,
                                    FALSE, 3, NULL, TRUE));
  assert_order("u:1,2 | i:3 | [d]");
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_teardown(test_element_onto_group_header, _teardown),
    cmocka_unit_test_teardown(test_element_onto_group_header_adopts_operator, _teardown),
    cmocka_unit_test_teardown(test_element_onto_its_own_group_header_is_a_noop, _teardown),
    cmocka_unit_test_teardown(test_element_onto_group_header_never_displaces_the_base, _teardown),
    cmocka_unit_test_teardown(test_element_onto_group_header_leaves_placeholder, _teardown),
    cmocka_unit_test_teardown(test_element_onto_invalid_group_is_rejected, _teardown),
    cmocka_unit_test_teardown(test_element_realizes_an_empty_group, _teardown),
    cmocka_unit_test_teardown(test_realizing_preserves_the_staged_ordinal, _teardown),
    cmocka_unit_test_teardown(test_realizing_preserves_the_staged_name, _teardown),
    cmocka_unit_test_teardown(test_realizing_adopts_staged_refinement, _teardown),
    cmocka_unit_test_teardown(test_realizing_unanchored_empty_lands_at_the_bottom, _teardown),
    cmocka_unit_test_teardown(test_realizing_an_unregistered_empty_is_rejected, _teardown),
    cmocka_unit_test_teardown(test_cluster_onto_group_header, _teardown),
    cmocka_unit_test_teardown(test_cluster_onto_element_row, _teardown),
    cmocka_unit_test_teardown(test_cluster_adopts_target_operator, _teardown),
    cmocka_unit_test_teardown(test_cluster_emptying_group_leaves_placeholder, _teardown),
    cmocka_unit_test_teardown(test_cluster_move_with_no_members_is_rejected, _teardown),
    cmocka_unit_test_teardown(test_reorder_group_above_another, _teardown),
    cmocka_unit_test_teardown(test_reorder_group_below_another, _teardown),
    cmocka_unit_test_teardown(test_reorder_keeps_members_together_and_ordered, _teardown),
    cmocka_unit_test_teardown(test_reorder_does_not_merge_same_op_neighbours, _teardown),
    cmocka_unit_test_teardown(test_reorder_onto_itself_is_rejected, _teardown),
    cmocka_unit_test_teardown(test_reorder_with_non_head_id_is_rejected, _teardown),
    cmocka_unit_test_teardown(test_visual_order_interleaves_staged_groups, _teardown),
    cmocka_unit_test_teardown(test_unanchored_staged_group_sits_at_the_bottom, _teardown),
    cmocka_unit_test_teardown(test_reorder_reanchors_staged_group, _teardown),
    cmocka_unit_test_teardown(test_staged_group_can_be_reordered, _teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
