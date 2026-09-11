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
#include "develop/develop.h"

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
// element chevrons under "auto-expand selected": the click decides what is
// open, not the selection it also makes
// ---------------------------------------------------------------------------

// a chevron click with the option on, applied the way _element_chevron_clicked
// applies it
static dt_masks_chevron_click_t _chevron(const dt_mask_id_t id, const gboolean expanded)
{
  const dt_masks_chevron_click_t c =
    _model_element_chevron_click(&flexi_bd, id, expanded, TRUE);
  flexi_bd.masks_last_expanded_elem = c.last_expanded;
  return c;
}

static void test_chevron_expand_collapses_previous(void **state)
{
  flexi_build("u:1,2,3");
  flexi_bd.masks_last_expanded_elem = INVALID_MASKID;
  assert_int_equal(_chevron(1, TRUE).collapse, INVALID_MASKID);
  assert_int_equal(flexi_bd.masks_last_expanded_elem, 1);
  assert_int_equal(_chevron(2, TRUE).collapse, 1);
  assert_int_equal(flexi_bd.masks_last_expanded_elem, 2);
}

// the reported bug: collapsing the open row must stick, not be re-opened
static void test_chevron_collapse_forgets_the_open_row(void **state)
{
  flexi_build("u:1,2,3");
  flexi_bd.masks_last_expanded_elem = 2;
  assert_int_equal(_chevron(2, FALSE).collapse, INVALID_MASKID);
  assert_int_equal(flexi_bd.masks_last_expanded_elem, INVALID_MASKID);
}

static void test_chevron_collapse_of_other_row_keeps_the_open_one(void **state)
{
  flexi_build("u:1,2,3");
  flexi_bd.masks_last_expanded_elem = 2;
  assert_int_equal(_chevron(1, FALSE).collapse, INVALID_MASKID);
  assert_int_equal(flexi_bd.masks_last_expanded_elem, 2);
}

static void test_chevron_reexpand_open_row_collapses_nothing(void **state)
{
  flexi_build("u:1,2,3");
  flexi_bd.masks_last_expanded_elem = 1;
  assert_int_equal(_chevron(1, TRUE).collapse, INVALID_MASKID);
  assert_int_equal(flexi_bd.masks_last_expanded_elem, 1);
}

static void test_chevron_without_auto_expand_moves_nothing(void **state)
{
  flexi_build("u:1,2,3");
  flexi_bd.masks_last_expanded_elem = 2;
  const dt_masks_chevron_click_t c =
    _model_element_chevron_click(&flexi_bd, 1, TRUE, FALSE);
  assert_int_equal(c.collapse, INVALID_MASKID);
  assert_int_equal(c.last_expanded, 2);
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

// ---------------------------------------------------------------------------
// linking and copying shapes between two modules' masks
// ---------------------------------------------------------------------------

#define OTHER_GROUP_ID 9000

// a second module, with a mask of its own over the fixture's forms
static dt_iop_module_t _other;
static dt_develop_blend_params_t _other_bp;
static dt_masks_form_t *_other_grp;

// a bare form in the fixture's list, freed by flexi_teardown()
static dt_masks_form_t *_new_form(const dt_mask_id_t fid, const dt_masks_type_t type)
{
  dt_masks_form_t *f = calloc(1, sizeof(dt_masks_form_t));
  f->formid = fid;
  f->type = type;
  snprintf(f->name, sizeof(f->name), "form #%d", (int)fid);
  flexi_dev.forms = g_list_append(flexi_dev.forms, f);
  return f;
}

// give the second module a mask: one union group holding `fids` bottom-up,
// each a circle unless it already exists. Both modules make up the pipe
static void _other_module(const dt_mask_id_t *fids, const int n)
{
  memset(&_other, 0, sizeof(_other));
  memset(&_other_bp, 0, sizeof(_other_bp));
  _other_grp = _new_form(OTHER_GROUP_ID, DT_MASKS_GROUP);
  for(int k = 0; k < n; k++)
  {
    if(!dt_masks_get_from_id(&flexi_dev, fids[k])) _new_form(fids[k], DT_MASKS_CIRCLE);
    dt_masks_point_group_t *pt = calloc(1, sizeof(dt_masks_point_group_t));
    pt->formid = fids[k];
    pt->parentid = OTHER_GROUP_ID;
    pt->state = DT_MASKS_STATE_UNION | DT_MASKS_STATE_USE;
    pt->opacity = 1.0f;
    _other_grp->points = g_list_append(_other_grp->points, pt);
  }
  _other_bp.mask_id = OTHER_GROUP_ID;
  _other.blend_params = &_other_bp;
  _other.dev = &flexi_dev;
  flexi_dev.iop = g_list_append(g_list_append(NULL, &flexi_module), &_other);
}

// what the panel does when a group is selected: new elements land on top of
// the group whose top member is `top`, taking its operator
static void _aim_at(const dt_mask_id_t top, const dt_masks_state_t op)
{
  flexi_bd.insert_active = TRUE;
  flexi_bd.insert_op = op;
  flexi_bd.insert_within = 0;
  flexi_bd.insert_after_fid = top;
  flexi_bd.insert_realize_empty = FALSE;
}

static int _users(const dt_mask_id_t fid)
{
  GList *users = _model_form_users(fid);
  const int n = g_list_length(users);
  g_list_free(users);
  return n;
}

static int _teardown_linking(void **state)
{
  if(_other_grp)
  {
    g_list_free_full(_other_grp->points, free);
    _other_grp->points = NULL;
    _other_grp = NULL;
  }
  g_list_free(flexi_dev.iop);
  flexi_dev.iop = NULL;
  flexi_conf_cleanup();
  flexi_teardown();
  return 0;
}

static void test_link_lands_in_target_group_in_order(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,2 | i:3");
  const dt_mask_id_t other[] = { 11, 12 };
  _other_module(other, 2);
  _aim_at(3, DT_MASKS_STATE_INTERSECTION);

  GList *fids = _model_module_shapes(&_other, INVALID_MASKID);
  GList *added = _model_import_forms(&flexi_module, &_other, fids, FALSE);
  assert_int_equal(g_list_length(added), 2);
  assert_layout("u:1,2 | i:3,11,12");
  // one form in two masks: that is the link
  assert_int_equal(_users(11), 2);
  assert_int_equal(_users(12), 2);
  g_list_free(added);
  g_list_free(fids);
}

static void test_link_skips_what_the_mask_already_uses(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,2 | i:3");
  const dt_mask_id_t other[] = { 11 };
  _other_module(other, 1);
  _aim_at(3, DT_MASKS_STATE_INTERSECTION);

  GList *fids = g_list_prepend(NULL, GINT_TO_POINTER(11));
  g_list_free(_model_import_forms(&flexi_module, &_other, fids, FALSE));
  _aim_at(11, DT_MASKS_STATE_INTERSECTION);
  assert_null(_model_import_forms(&flexi_module, &_other, fids, FALSE));
  assert_null(_model_import_forms(&flexi_module, &_other, fids, TRUE));
  assert_layout("u:1,2 | i:3,11");
  g_list_free(fids);
}

// the element looks as it does where it comes from; only the operator is the
// target group's
static void test_link_keeps_the_source_look(void **state)
{
  flexi_conf_init();
  flexi_build("u:1 | i:3");
  const dt_mask_id_t other[] = { 11 };
  _other_module(other, 1);
  dt_masks_point_group_t *spt = _other_grp->points->data;
  spt->opacity = 0.4f;
  spt->state |= DT_MASKS_STATE_INVERSE;
  _aim_at(3, DT_MASKS_STATE_INTERSECTION);

  GList *fids = g_list_prepend(NULL, GINT_TO_POINTER(11));
  g_list_free(_model_import_forms(&flexi_module, &_other, fids, FALSE));
  g_list_free(fids);

  const dt_masks_point_group_t *pt = _group_point(flexi_group(), 11);
  assert_non_null(pt);
  assert_float_equal(pt->opacity, 0.4f, 1e-6f);
  assert_true(pt->state & DT_MASKS_STATE_INVERSE);
  assert_int_equal(pt->state & DT_MASKS_STATE_OP, DT_MASKS_STATE_INTERSECTION);
}

static void test_copy_is_a_new_independent_form(void **state)
{
  flexi_conf_init();
  flexi_build("u:1 | i:3");
  const dt_mask_id_t other[] = { 11 };
  _other_module(other, 1);
  _aim_at(3, DT_MASKS_STATE_INTERSECTION);

  GList *fids = g_list_prepend(NULL, GINT_TO_POINTER(11));
  GList *added = _model_import_forms(&flexi_module, &_other, fids, TRUE);
  g_list_free(fids);
  assert_int_equal(g_list_length(added), 1);
  const dt_mask_id_t nid = GPOINTER_TO_INT(added->data);
  g_list_free(added);

  assert_int_not_equal(nid, 11);
  assert_null(_group_point(flexi_group(), 11));
  assert_int_equal(_group_cid_of_form(flexi_group(), nid), 3);
  assert_int_equal(_users(11), 1);
  assert_int_equal(_users(nid), 1);
  assert_string_equal(dt_masks_get_from_id(&flexi_dev, nid)->name, "form #11");
}

// an AI object's copy owns copies of its paths: sharing them would leave the
// two objects linked underneath
static void test_copy_of_object_copies_its_paths(void **state)
{
  flexi_conf_init();
  flexi_build("u:1");
  const dt_mask_id_t other[] = { 11 };
  _other_module(other, 1);
  dt_masks_form_t *obj = dt_masks_get_from_id(&flexi_dev, 11);
  obj->type = DT_MASKS_OBJECT;
  for(dt_mask_id_t m = 21; m <= 22; m++)
  {
    _new_form(m, DT_MASKS_PATH);
    dt_masks_point_group_t *mpt = calloc(1, sizeof(dt_masks_point_group_t));
    mpt->formid = m;
    mpt->parentid = 11;
    obj->points = g_list_append(obj->points, mpt);
  }
  _aim_at(1, DT_MASKS_STATE_UNION);

  GList *fids = g_list_prepend(NULL, GINT_TO_POINTER(11));
  GList *added = _model_import_forms(&flexi_module, &_other, fids, TRUE);
  g_list_free(fids);
  assert_int_equal(g_list_length(added), 1);
  const dt_masks_form_t *copy = dt_masks_get_from_id(&flexi_dev, GPOINTER_TO_INT(added->data));
  g_list_free(added);

  assert_non_null(copy);
  assert_true(copy->type & DT_MASKS_OBJECT);
  assert_int_equal(g_list_length(copy->points), 2);
  for(const GList *l = copy->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *mpt = l->data;
    assert_true(mpt->formid != 21 && mpt->formid != 22);
    assert_int_equal(mpt->parentid, copy->formid);
    assert_non_null(dt_masks_get_from_id(&flexi_dev, mpt->formid));
  }
  g_list_free_full(obj->points, free);
  obj->points = NULL;
}

static void test_module_shapes_follow_groups_and_skip_other_kinds(void **state)
{
  flexi_conf_init();
  flexi_build("u:1");
  const dt_mask_id_t other[] = { 11, 12, 13 };
  _other_module(other, 3);
  dt_masks_get_from_id(&flexi_dev, 12)->type = DT_MASKS_PARAMETRIC;
  // 13 heads a second group
  dt_masks_point_group_t *head = g_list_nth_data(_other_grp->points, 2);
  head->group_start = 1;
  head->state = DT_MASKS_STATE_INTERSECTION | DT_MASKS_STATE_USE;

  GList *all = _model_module_shapes(&_other, INVALID_MASKID);
  assert_int_equal(g_list_length(all), 2);
  assert_int_equal(GPOINTER_TO_INT(all->data), 11);
  assert_int_equal(GPOINTER_TO_INT(all->next->data), 13);
  g_list_free(all);

  GList *second = _model_module_shapes(&_other, 13);
  assert_int_equal(g_list_length(second), 1);
  assert_int_equal(GPOINTER_TO_INT(second->data), 13);
  g_list_free(second);
}

static void test_unlink_gives_this_module_its_own_copy(void **state)
{
  flexi_conf_init();
  flexi_build("u:1 | i:3");
  const dt_mask_id_t other[] = { 11 };
  _other_module(other, 1);
  _aim_at(3, DT_MASKS_STATE_INTERSECTION);
  GList *fids = g_list_prepend(NULL, GINT_TO_POINTER(11));
  g_list_free(_model_import_forms(&flexi_module, &_other, fids, FALSE));
  g_list_free(fids);
  flexi_bd.panel_selected_formid = 11;

  const dt_mask_id_t nid = _model_unlink_form(&flexi_module, 11);
  assert_true(dt_is_valid_maskid(nid));
  assert_int_not_equal(nid, 11);
  assert_null(_group_point(flexi_group(), 11));
  assert_int_equal(_group_cid_of_form(flexi_group(), nid), 3);
  assert_non_null(_group_point(_other_grp, 11));
  assert_int_equal(_users(11), 1);
  assert_int_equal(flexi_bd.panel_selected_formid, nid);
}

// a group is known by its head's id: unlinking the head must not renumber it
static void test_unlink_of_group_head_keeps_its_number(void **state)
{
  flexi_conf_init();
  flexi_build("u:1 | i:11");
  const dt_mask_id_t other[] = { 11 };
  _other_module(other, 1);
  flexi_set_ordinal(11, 2);
  flexi_bd.panel_selected_group_cid = 11;

  const dt_mask_id_t nid = _model_unlink_form(&flexi_module, 11);
  assert_int_equal(_group_cid_of_form(flexi_group(), nid), nid);
  assert_int_equal(flexi_get_ordinal(nid), 2);
  assert_int_equal(flexi_bd.panel_selected_group_cid, nid);
}

// the link indicator: the chain icon and the "unlink" entry show together
static void test_link_shows_only_for_shared_elements_but_raster(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,2,3");
  const dt_mask_id_t other[] = { 1, 3 };
  _other_module(other, 2);
  dt_masks_get_from_id(&flexi_dev, 3)->type = DT_MASKS_RASTER;

  assert_true(_model_form_is_linked(dt_masks_get_from_id(&flexi_dev, 1)));
  assert_false(_model_form_is_linked(dt_masks_get_from_id(&flexi_dev, 2)));
  // a raster element reads another module's mask; nothing of it is editable
  assert_false(_model_form_is_linked(dt_masks_get_from_id(&flexi_dev, 3)));
  assert_false(_model_form_is_linked(NULL));
}

// channels are only ever copied, but one shared by an older duplicate still
// shows the link, so it can be unlinked
static void test_shared_parametric_channel_shows_the_link(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,2");
  const dt_mask_id_t other[] = { 2 };
  _other_module(other, 1);
  dt_masks_get_from_id(&flexi_dev, 2)->type = DT_MASKS_PARAMETRIC;
  assert_true(_model_form_is_linked(dt_masks_get_from_id(&flexi_dev, 2)));
}

// another module linking or unlinking an element of this mask leaves the mask
// untouched, but its chain icon changes: the list must not skip the rebuild
static void test_list_signature_follows_links_made_elsewhere(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,2");
  const dt_mask_id_t other[] = { 11 };
  _other_module(other, 1);
  const dt_hash_t alone = _masks_list_signature(&flexi_module);
  assert_true(alone == _masks_list_signature(&flexi_module));

  dt_masks_point_group_t *pt = calloc(1, sizeof(dt_masks_point_group_t));
  pt->formid = 2;
  pt->parentid = OTHER_GROUP_ID;
  pt->state = DT_MASKS_STATE_UNION | DT_MASKS_STATE_USE;
  pt->opacity = 1.0f;
  _other_grp->points = g_list_append(_other_grp->points, pt);
  assert_true(_masks_list_signature(&flexi_module) != alone);

  _other_grp->points = g_list_remove(_other_grp->points, pt);
  free(pt);
  assert_true(_masks_list_signature(&flexi_module) == alone);
}

static void test_list_signature_ignores_links_among_others(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,2");
  const dt_mask_id_t other[] = { 11 };
  _other_module(other, 1);
  const dt_hash_t before = _masks_list_signature(&flexi_module);

  _new_form(12, DT_MASKS_CIRCLE);
  dt_masks_point_group_t *pt = calloc(1, sizeof(dt_masks_point_group_t));
  pt->formid = 12;
  pt->parentid = OTHER_GROUP_ID;
  _other_grp->points = g_list_append(_other_grp->points, pt);
  assert_true(_masks_list_signature(&flexi_module) == before);
}

// a duplicated instance shares its shapes with the original but gets its own
// channels, looking as they did there
static void test_duplicate_shares_shapes_and_copies_channels(void **state)
{
  flexi_conf_init();
  flexi_build("u:1");
  const dt_mask_id_t other[] = { 11, 12 };
  _other_module(other, 2);
  dt_masks_get_from_id(&flexi_dev, 12)->type = DT_MASKS_PARAMETRIC;
  dt_masks_point_group_t *spt = g_list_nth_data(_other_grp->points, 1);
  spt->opacity = 0.3f;
  spt->state = DT_MASKS_STATE_INTERSECTION | DT_MASKS_STATE_USE;

  dt_masks_form_t *dup = _new_form(8000, DT_MASKS_GROUP);
  dt_masks_group_add_members_of(dup, _other_grp);
  assert_int_equal(g_list_length(dup->points), 2);
  const dt_masks_point_group_t *shape = dup->points->data;
  const dt_masks_point_group_t *channel = dup->points->next->data;
  assert_int_equal(shape->formid, 11);
  assert_int_not_equal(channel->formid, 12);
  assert_true(dt_masks_get_from_id(&flexi_dev, channel->formid)->type & DT_MASKS_PARAMETRIC);
  assert_float_equal(channel->opacity, 0.3f, 1e-6f);
  assert_int_equal(channel->state, spt->state);
  // the original keeps its own channel
  assert_int_equal(_users(12), 1);
  g_list_free_full(dup->points, free);
  dup->points = NULL;
}

// ---------------------------------------------------------------------------
// AI objects: stepping into one on the canvas, and what a delete takes away
// ---------------------------------------------------------------------------

static dt_masks_form_gui_t _gui;

static void _with_gui(const dt_mask_id_t entered)
{
  memset(&_gui, 0, sizeof(_gui));
  _gui.entered_object = entered;
  flexi_dev.form_gui = &_gui;
}

// make `oid` an AI object of `n` paths numbered from `first`
static dt_masks_form_t *_make_object(const dt_mask_id_t oid,
                                     const dt_mask_id_t first,
                                     const int n)
{
  dt_masks_form_t *obj = dt_masks_get_from_id(&flexi_dev, oid);
  if(!obj) obj = _new_form(oid, DT_MASKS_OBJECT);
  obj->type = DT_MASKS_OBJECT;
  for(int k = 0; k < n; k++)
  {
    _new_form(first + k, DT_MASKS_PATH);
    dt_masks_point_group_t *mpt = calloc(1, sizeof(dt_masks_point_group_t));
    mpt->formid = first + k;
    mpt->parentid = oid;
    mpt->state = DT_MASKS_STATE_UNION | DT_MASKS_STATE_USE;
    mpt->opacity = 1.0f;
    obj->points = g_list_append(obj->points, mpt);
  }
  return obj;
}

static int _teardown_objects(void **state)
{
  for(GList *l = flexi_dev.forms; l; l = g_list_next(l))
  {
    dt_masks_form_t *f = l->data;
    if(!(f->type & DT_MASKS_OBJECT)) continue;
    g_list_free_full(f->points, free);
    f->points = NULL;
  }
  flexi_dev.form_gui = NULL;
  return _teardown_linking(state);
}

static void test_double_click_on_object_steps_in(void **state)
{
  flexi_conf_init();
  flexi_build("u:1");
  _with_gui(INVALID_MASKID);
  assert_true(dt_masks_gui_step_object(NULL, &_gui, 5, TRUE, TRUE));
  assert_int_equal(_gui.entered_object, 5);
}

// inside, a double-click is an ordinary click on one of the paths
static void test_double_click_inside_object_is_ordinary(void **state)
{
  flexi_conf_init();
  flexi_build("u:1");
  _with_gui(5);
  assert_false(dt_masks_gui_step_object(NULL, &_gui, 5, TRUE, TRUE));
  assert_int_equal(_gui.entered_object, 5);
}

static void test_click_inside_object_stays_in(void **state)
{
  flexi_conf_init();
  flexi_build("u:1");
  _with_gui(5);
  assert_false(dt_masks_gui_step_object(NULL, &_gui, 5, TRUE, FALSE));
  assert_int_equal(_gui.entered_object, 5);
}

// the click that steps out still does what it would have done
static void test_click_outside_object_steps_out(void **state)
{
  flexi_conf_init();
  flexi_build("u:1");
  _with_gui(5);
  assert_false(dt_masks_gui_step_object(NULL, &_gui, INVALID_MASKID, TRUE, FALSE));
  assert_int_equal(_gui.entered_object, INVALID_MASKID);
  _with_gui(5);
  assert_false(dt_masks_gui_step_object(NULL, &_gui, 6, TRUE, FALSE));
  assert_int_equal(_gui.entered_object, INVALID_MASKID);
}

static void test_double_click_on_other_object_moves_into_it(void **state)
{
  flexi_conf_init();
  flexi_build("u:1");
  _with_gui(5);
  assert_true(dt_masks_gui_step_object(NULL, &_gui, 6, TRUE, TRUE));
  assert_int_equal(_gui.entered_object, 6);
}

// a right-click inside removes a path: it must neither step out nor in
static void test_secondary_button_never_steps(void **state)
{
  flexi_conf_init();
  flexi_build("u:1");
  _with_gui(5);
  assert_false(dt_masks_gui_step_object(NULL, &_gui, INVALID_MASKID, FALSE, FALSE));
  assert_int_equal(_gui.entered_object, 5);
  _with_gui(INVALID_MASKID);
  assert_false(dt_masks_gui_step_object(NULL, &_gui, 6, FALSE, TRUE));
  assert_int_equal(_gui.entered_object, INVALID_MASKID);
}

static void test_double_click_on_plain_shape_does_nothing(void **state)
{
  flexi_conf_init();
  flexi_build("u:1");
  _with_gui(INVALID_MASKID);
  assert_false(dt_masks_gui_step_object(NULL, &_gui, INVALID_MASKID, TRUE, TRUE));
  assert_int_equal(_gui.entered_object, INVALID_MASKID);
}

// hover, drag, scroll and rotate move an object's paths together, except
// inside the object the user stepped into
// "clean up unused shapes": an object's paths are only reached through the
// object, so they are used whenever the object is
static void test_cleanup_keeps_the_paths_of_a_used_object(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,5");
  _make_object(5, 21, 2);

  dt_develop_blend_params_t bp = { 0 };
  bp.mask_id = grp->formid;
  dt_dev_history_item_t hist = { 0 };
  hist.blend_params = &bp;
  hist.forms = g_list_copy(flexi_dev.forms);
  GList *history = g_list_append(NULL, &hist);

  dt_masks_cleanup_unused_from_list(history);

  assert_non_null(dt_masks_get_from_id_ext(hist.forms, 5));
  assert_non_null(dt_masks_get_from_id_ext(hist.forms, 21));
  assert_non_null(dt_masks_get_from_id_ext(hist.forms, 22));

  g_list_free(history);
  g_list_free(hist.forms);
  // removed forms are only moved there; the fixture owns them
  g_list_free(flexi_dev.allforms);
  flexi_dev.allforms = NULL;
}

// an object none of whose paths exists is dropped from the group and the
// forms, and the member above it keeps its own group
static void test_empty_object_is_removed(void **state)
{
  flexi_build("u:1 | u:5,6");
  dt_masks_form_t *obj = _make_object(5, 21, 0);

  assert_int_equal(dt_masks_prune_empty_objects(&flexi_dev.forms), 1);
  assert_layout("u:1 | u:6");
  assert_null(dt_masks_get_from_id(&flexi_dev, 5));

  // back to the fixture, which frees it
  g_list_free(flexi_dev.allforms);
  flexi_dev.allforms = NULL;
  flexi_dev.forms = g_list_append(flexi_dev.forms, obj);
}

static void test_object_with_a_path_is_kept(void **state)
{
  flexi_build("u:1 | u:5,6");
  _make_object(5, 21, 1);

  assert_int_equal(dt_masks_prune_empty_objects(&flexi_dev.forms), 0);
  assert_layout("u:1 | u:5,6");
  assert_non_null(dt_masks_get_from_id(&flexi_dev, 5));
}

static void test_object_paths_act_as_one_until_stepped_in(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,5");
  dt_masks_form_t *obj = _make_object(5, 21, 2);
  const dt_masks_point_group_t path = { .formid = 21, .parentid = 5 };
  const dt_masks_point_group_t plain = { .formid = 1, .parentid = flexi_bp.mask_id };

  _with_gui(INVALID_MASKID);
  assert_ptr_equal(dt_masks_bundle_of(&path), obj);
  assert_null(dt_masks_bundle_of(&plain));
  _gui.entered_object = 5;
  assert_null(dt_masks_bundle_of(&path));
  _gui.entered_object = 6;
  assert_ptr_equal(dt_masks_bundle_of(&path), obj);
}

static dt_masks_remove_target_t _removes(const dt_mask_id_t fid,
                                         const dt_mask_id_t parent,
                                         const gboolean whole,
                                         dt_mask_id_t *taken,
                                         dt_mask_id_t *from)
{
  dt_masks_form_t *form = dt_masks_get_from_id(&flexi_dev, fid);
  dt_mask_id_t parentid = parent;
  const dt_masks_remove_target_t t =
    dt_masks_remove_shape_target(&flexi_module, &form, &parentid, whole);
  *taken = form->formid;
  *from = parentid;
  return t;
}

// an object is one element: a right-click on any of its paths removes it
// from this module, the way the panel's delete does
static void test_right_click_on_object_path_removes_the_object(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,5");
  _make_object(5, 21, 2);
  _with_gui(INVALID_MASKID);
  dt_mask_id_t taken, from;
  assert_int_equal(_removes(22, 5, TRUE, &taken, &from), DT_MASKS_REMOVE_ELEMENT);
  assert_int_equal(taken, 5);
  assert_int_equal(from, flexi_bp.mask_id);
}

static void test_right_click_inside_object_removes_one_path(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,5");
  _make_object(5, 21, 2);
  _with_gui(5);
  dt_mask_id_t taken, from;
  assert_int_equal(_removes(22, 5, TRUE, &taken, &from), DT_MASKS_REMOVE_PATH);
  assert_int_equal(taken, 22);
  assert_int_equal(from, 5);
}

// deleting nodes down to a path's minimum removes it gradually, stepped in or not
static void test_node_deletion_removes_one_path(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,5");
  _make_object(5, 21, 2);
  _with_gui(INVALID_MASKID);
  dt_mask_id_t taken, from;
  assert_int_equal(_removes(21, 5, FALSE, &taken, &from), DT_MASKS_REMOVE_PATH);
  assert_int_equal(taken, 21);
  assert_int_equal(from, 5);
}

// emptying the object would take it from every module that links it: its
// last path takes the object, from this module only
static void test_last_path_removes_the_object(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,5");
  _make_object(5, 21, 1);
  _with_gui(5);
  dt_mask_id_t taken, from;
  assert_int_equal(_removes(21, 5, TRUE, &taken, &from), DT_MASKS_REMOVE_ELEMENT);
  assert_int_equal(taken, 5);
  assert_int_equal(_removes(21, 5, FALSE, &taken, &from), DT_MASKS_REMOVE_ELEMENT);
  assert_int_equal(taken, 5);
  assert_int_equal(from, flexi_bp.mask_id);
}

// no canvas edit session: a gesture on the object still takes it whole
static void test_object_removal_without_canvas_session(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,5");
  _make_object(5, 21, 2);
  dt_mask_id_t taken, from;
  assert_int_equal(_removes(21, 5, TRUE, &taken, &from), DT_MASKS_REMOVE_ELEMENT);
  assert_int_equal(taken, 5);
}

// canvas and panel delete the same way: this module's instance only
static void test_canvas_delete_of_shape_is_the_panel_delete(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,2");
  const dt_mask_id_t other[] = { 2 };
  _other_module(other, 1);
  _with_gui(INVALID_MASKID);
  dt_mask_id_t taken, from;
  assert_int_equal(_removes(2, flexi_bp.mask_id, TRUE, &taken, &from),
                   DT_MASKS_REMOVE_ELEMENT);
  assert_int_equal(taken, 2);
  // outside this module's flexi mask a shape just leaves its parent
  assert_int_equal(_removes(2, OTHER_GROUP_ID, TRUE, &taken, &from),
                   DT_MASKS_REMOVE_FROM_PARENT);
  assert_int_equal(from, OTHER_GROUP_ID);
}

// the panel has a row for the object, none for its paths
static void test_canvas_path_selects_its_object_row(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,5");
  _make_object(5, 21, 2);
  assert_int_equal(_model_panel_formid_for(&flexi_module, 22), 5);
  assert_int_equal(_model_panel_formid_for(&flexi_module, 5), 5);
  assert_int_equal(_model_panel_formid_for(&flexi_module, 1), 1);
  assert_int_equal(_model_panel_formid_for(&flexi_module, 77), 77);
  assert_int_equal(_model_panel_formid_for(&flexi_module, INVALID_MASKID), INVALID_MASKID);
}

// ---------------------------------------------------------------------------
// the refinement panel's target when its element disappears
// ---------------------------------------------------------------------------

static void _scope(const int kind, const dt_mask_id_t fid)
{
  flexi_bd.masks_refine_scope_kind = kind;
  flexi_bd.masks_refine_scope_formid = fid;
}

// an element removed by a route that does not reselect (canvas, an AI
// object's last path, undo) must not stay the refinement target
static void test_refine_scope_of_removed_element_falls_back_to_its_group(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,2 | i:3");
  _scope(REFINE_SCOPE_ELEMENT, 9);
  flexi_bd.panel_selected_formid = 9;
  flexi_bd.panel_selected_group_cid = 3;

  assert_true(_model_refine_scope_prune(&flexi_module));
  assert_int_equal(flexi_bd.panel_selected_formid, INVALID_MASKID);
  assert_int_equal(flexi_bd.panel_selected_group_cid, 3);
  _model_refine_scope_from_selection(&flexi_module);
  assert_int_equal(flexi_bd.masks_refine_scope_kind, REFINE_SCOPE_GROUP);
  assert_int_equal(flexi_bd.masks_refine_scope_formid, 3);
}

static void test_refine_scope_with_nothing_left_is_the_whole_mask(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,2");
  _scope(REFINE_SCOPE_GROUP, 9);
  flexi_bd.panel_selected_group_cid = 9;

  assert_true(_model_refine_scope_prune(&flexi_module));
  assert_int_equal(flexi_bd.panel_selected_group_cid, INVALID_MASKID);
  _model_refine_scope_from_selection(&flexi_module);
  assert_int_equal(flexi_bd.masks_refine_scope_kind, REFINE_SCOPE_GLOBAL);
  assert_int_equal(flexi_bd.masks_refine_scope_formid, INVALID_MASKID);
}

static void test_refine_scope_of_present_element_is_kept(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,2");
  _scope(REFINE_SCOPE_ELEMENT, 2);
  flexi_bd.panel_selected_formid = 2;
  flexi_bd.panel_selected_group_cid = 1;

  assert_false(_model_refine_scope_prune(&flexi_module));
  assert_int_equal(flexi_bd.panel_selected_formid, 2);
  assert_int_equal(flexi_bd.panel_selected_group_cid, 1);
  _scope(REFINE_SCOPE_GLOBAL, INVALID_MASKID);
  assert_false(_model_refine_scope_prune(&flexi_module));
}

static void test_refine_scope_follows_the_selection(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,2 | i:3");
  flexi_bd.panel_selected_formid = 2;
  flexi_bd.panel_selected_group_cid = 1;
  _model_refine_scope_from_selection(&flexi_module);
  assert_int_equal(flexi_bd.masks_refine_scope_kind, REFINE_SCOPE_ELEMENT);
  assert_int_equal(flexi_bd.masks_refine_scope_formid, 2);

  flexi_bd.panel_selected_formid = INVALID_MASKID;
  flexi_bd.panel_selected_group_cid = INVALID_MASKID;
  flexi_bd.selected_empty = flexi_add_empty(DT_MASKS_STATE_UNION, INVALID_MASKID);
  _model_refine_scope_from_selection(&flexi_module);
  assert_int_equal(flexi_bd.masks_refine_scope_kind, REFINE_SCOPE_EMPTY_GROUP);
}

// ---------------------------------------------------------------------------
// raster elements: named after their source until given a name of their own
// ---------------------------------------------------------------------------

static dt_iop_module_so_t _this_so, _other_so;
static const char *_exposure_name(void) { return "exposure"; }
static const char *_this_name(void) { return "contrast"; }

// the other module is an exposure instance, and form `fid` of this mask a
// raster element reading its mask, named by its type alone as when added
static dt_masks_form_t *_raster_of_other(const dt_mask_id_t fid)
{
  const dt_mask_id_t other[] = { 11 };
  _other_module(other, 1);
  g_strlcpy(_other_so.op, "exposure", sizeof(_other_so.op));
  _other.so = &_other_so;
  _other.name = _exposure_name;
  g_strlcpy(_this_so.op, "bilat", sizeof(_this_so.op));
  flexi_module.so = &_this_so;
  flexi_module.name = _this_name;

  dt_masks_form_t *f = dt_masks_get_from_id(&flexi_dev, fid);
  f->type = DT_MASKS_RASTER;
  dt_masks_point_raster_t *p = calloc(1, sizeof(dt_masks_point_raster_t));
  g_strlcpy(p->source, "exposure", sizeof(p->source));
  f->points = g_list_append(NULL, p);
  g_strlcpy(f->name, _("raster mask"), sizeof(f->name));
  return f;
}

static void _rename_other(const char *name)
{
  g_strlcpy(_other.multi_name, name, sizeof(_other.multi_name));
  _other.multi_name_hand_edited = TRUE;
}

static void _assert_shows(const dt_masks_form_t *f, const char *expect)
{
  gchar *shown = _form_display_name(f);
  assert_string_equal(shown, expect);
  g_free(shown);
}

static int _teardown_raster(void **state)
{
  for(GList *l = flexi_dev.forms; l; l = g_list_next(l))
  {
    dt_masks_form_t *f = l->data;
    if(!(f->type & DT_MASKS_RASTER)) continue;
    g_list_free_full(f->points, free);
    f->points = NULL;
  }
  return _teardown_linking(state);
}

static void test_raster_element_follows_its_source_name(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,7");
  const dt_masks_form_t *f = _raster_of_other(7);
  gchar *before = dt_history_item_get_name(&_other);
  _assert_shows(f, before);

  _rename_other("sky");
  gchar *after = dt_history_item_get_name(&_other);
  assert_string_not_equal(after, before);
  _assert_shows(f, after);
  g_free(before);
  g_free(after);
}

static void test_renamed_raster_element_keeps_its_name(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,7");
  dt_masks_form_t *f = _raster_of_other(7);
  assert_true(_model_rename_form(f, "mine"));
  _assert_shows(f, "mine");
  _rename_other("sky");
  _assert_shows(f, "mine");
}

// emptying the name hands it back to the source; other kinds need a name
static void test_emptied_raster_name_follows_the_source_again(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,7");
  dt_masks_form_t *f = _raster_of_other(7);
  assert_true(_model_rename_form(f, "mine"));
  assert_true(_model_rename_form(f, ""));
  gchar *label = dt_history_item_get_name(&_other);
  _assert_shows(f, label);
  g_free(label);

  dt_masks_form_t *circle = dt_masks_get_from_id(&flexi_dev, 1);
  gchar *was = g_strdup(circle->name);
  assert_false(_model_rename_form(circle, ""));
  assert_string_equal(circle->name, was);
  g_free(was);
}

static void test_raster_source_is_found_by_operation_and_instance(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,7");
  dt_masks_form_t *f = _raster_of_other(7);
  assert_ptr_equal(dt_masks_raster_source(f), &_other);
  assert_null(dt_masks_raster_source(dt_masks_get_from_id(&flexi_dev, 1)));
  assert_null(dt_masks_raster_source(NULL));

  // a source that is gone leaves the row its operation to show
  ((dt_masks_point_raster_t *)f->points->data)->instance = 3;
  assert_null(dt_masks_raster_source(f));
  _assert_shows(f, "exposure");
}

// raster elements used to store their source's name as it was then
static void test_old_raster_name_follows_its_source(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,7");
  dt_masks_form_t *f = _raster_of_other(7);
  gchar *label = dt_history_item_get_name(&_other);
  g_snprintf(f->name, sizeof(f->name), "%s %s", _("raster mask"), label);
  g_free(label);
  _model_raster_names_follow_sources(flexi_group());
  assert_string_equal(f->name, _("raster mask"));

  g_snprintf(f->name, sizeof(f->name), "%s mine", _("raster mask"));
  _model_raster_names_follow_sources(flexi_group());
  _assert_shows(f, "mine");
}

// renaming the source changes nothing in this mask but what its row shows
// ---------------------------------------------------------------------------
// add target: the add shape / parametric / import buttons are enabled exactly
// when this resolves
// ---------------------------------------------------------------------------

static void _no_selection(void)
{
  flexi_bd.selected_empty = NULL;
  flexi_bd.panel_selected_group_cid = INVALID_MASKID;
}

static void test_add_target_is_the_only_group(void **state)
{
  flexi_build("u:1,2");
  _no_selection();

  const dt_masks_add_target_t t = _resolve_add_target(&flexi_module);
  assert_true(t.valid);
  assert_true(t.implicit);
  assert_int_equal(t.cid, 1);
  assert_null(t.empty);
}

static void test_add_target_ignores_a_stale_selection_with_one_group(void **state)
{
  flexi_build("u:1,2");
  _no_selection();
  flexi_bd.panel_selected_group_cid = 9;

  const dt_masks_add_target_t t = _resolve_add_target(&flexi_module);
  assert_true(t.valid);
  assert_true(t.implicit);
  assert_int_equal(t.cid, 1);
}

static void test_add_target_is_ambiguous_with_two_groups(void **state)
{
  flexi_build("u:1 | i:2");
  _no_selection();
  assert_int_equal(_group_count(&flexi_module), 2);
  assert_false(_resolve_add_target(&flexi_module).valid);

  flexi_bd.panel_selected_group_cid = 2;
  const dt_masks_add_target_t t = _resolve_add_target(&flexi_module);
  assert_true(t.valid);
  assert_false(t.implicit);
  assert_int_equal(t.cid, 2);
}

static void test_add_target_counts_a_staged_group(void **state)
{
  flexi_build("u:1");
  _no_selection();
  dt_masks_empty_group_t *eg = _empty_group_new(DT_MASKS_STATE_UNION, 0, INVALID_MASKID);
  flexi_bd.empty_groups = g_list_append(flexi_bd.empty_groups, eg);
  assert_int_equal(_group_count(&flexi_module), 2);
  assert_false(_resolve_add_target(&flexi_module).valid);

  flexi_bd.selected_empty = eg;
  const dt_masks_add_target_t t = _resolve_add_target(&flexi_module);
  assert_true(t.valid);
  assert_false(t.implicit);
  assert_ptr_equal(t.empty, eg);
  _no_selection();
}

// a point whose form is missing from dev->forms, starting its own run
static void _add_dangling_run(dt_masks_form_t *grp, const gboolean at_bottom)
{
  dt_masks_point_group_t *pt = calloc(1, sizeof(dt_masks_point_group_t));
  pt->formid = 99;
  pt->state = DT_MASKS_STATE_USE | DT_MASKS_STATE_UNION;
  pt->opacity = 1.0f;
  if(at_bottom)
  {
    // the old bottom point now sits above another one: keep it a run head
    ((dt_masks_point_group_t *)grp->points->data)->group_start = 1;
    grp->points = g_list_prepend(grp->points, pt);
  }
  else
  {
    pt->group_start = 1;
    grp->points = g_list_append(grp->points, pt);
  }
}

// the panel drops a group none of whose members resolve, so it must not make
// the one group it does show ambiguous
static void test_add_target_skips_a_group_the_panel_hides(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2");
  _no_selection();
  _add_dangling_run(grp, FALSE);

  assert_int_equal(_group_count(&flexi_module), 1);
  const dt_masks_add_target_t t = _resolve_add_target(&flexi_module);
  assert_true(t.valid);
  assert_int_equal(t.cid, 1);
}

static void test_add_target_is_the_shown_group_above_a_hidden_one(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2");
  _no_selection();
  _add_dangling_run(grp, TRUE);

  assert_int_equal(_group_count(&flexi_module), 1);
  const dt_masks_add_target_t t = _resolve_add_target(&flexi_module);
  assert_true(t.valid);
  assert_int_equal(t.cid, 1);
}

static void test_list_signature_follows_source_rename(void **state)
{
  flexi_conf_init();
  flexi_build("u:1,7");
  dt_masks_form_t *f = _raster_of_other(7);
  const dt_hash_t before = _masks_list_signature(&flexi_module);
  _rename_other("sky");
  assert_true(_masks_list_signature(&flexi_module) != before);

  assert_true(_model_rename_form(f, "mine"));
  const dt_hash_t named = _masks_list_signature(&flexi_module);
  _rename_other("trees");
  assert_true(_masks_list_signature(&flexi_module) == named);
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
    cmocka_unit_test_teardown(test_chevron_expand_collapses_previous, _teardown),
    cmocka_unit_test_teardown(test_chevron_collapse_forgets_the_open_row, _teardown),
    cmocka_unit_test_teardown(test_chevron_collapse_of_other_row_keeps_the_open_one, _teardown),
    cmocka_unit_test_teardown(test_chevron_reexpand_open_row_collapses_nothing, _teardown),
    cmocka_unit_test_teardown(test_chevron_without_auto_expand_moves_nothing, _teardown),
    cmocka_unit_test_teardown(test_normalize_clears_break_on_base_point, _teardown),
    cmocka_unit_test_teardown(test_normalize_defaults_missing_operator_to_union, _teardown),
    cmocka_unit_test_teardown(test_normalize_keeps_operator_under_bypass, _teardown),
    cmocka_unit_test_teardown(test_normalize_preserves_partition, _teardown),
    cmocka_unit_test_teardown(test_isolate_state_hides_everything_else, _teardown),
    cmocka_unit_test_teardown(test_isolate_state_null_list_clears_everywhere, _teardown),
    cmocka_unit_test_teardown(test_isolate_state_soloing_a_whole_group, _teardown),
    cmocka_unit_test_teardown(test_set_state_targets_only_listed_members, _teardown),
    cmocka_unit_test_teardown(test_link_lands_in_target_group_in_order, _teardown_linking),
    cmocka_unit_test_teardown(test_link_skips_what_the_mask_already_uses, _teardown_linking),
    cmocka_unit_test_teardown(test_link_keeps_the_source_look, _teardown_linking),
    cmocka_unit_test_teardown(test_copy_is_a_new_independent_form, _teardown_linking),
    cmocka_unit_test_teardown(test_copy_of_object_copies_its_paths, _teardown_linking),
    cmocka_unit_test_teardown(test_module_shapes_follow_groups_and_skip_other_kinds,
                              _teardown_linking),
    cmocka_unit_test_teardown(test_unlink_gives_this_module_its_own_copy, _teardown_linking),
    cmocka_unit_test_teardown(test_unlink_of_group_head_keeps_its_number, _teardown_linking),
    cmocka_unit_test_teardown(test_link_shows_only_for_shared_elements_but_raster,
                              _teardown_linking),
    cmocka_unit_test_teardown(test_shared_parametric_channel_shows_the_link, _teardown_linking),
    cmocka_unit_test_teardown(test_list_signature_follows_links_made_elsewhere,
                              _teardown_linking),
    cmocka_unit_test_teardown(test_list_signature_ignores_links_among_others, _teardown_linking),
    cmocka_unit_test_teardown(test_duplicate_shares_shapes_and_copies_channels,
                              _teardown_linking),
    cmocka_unit_test_teardown(test_double_click_on_object_steps_in, _teardown_objects),
    cmocka_unit_test_teardown(test_double_click_inside_object_is_ordinary, _teardown_objects),
    cmocka_unit_test_teardown(test_click_inside_object_stays_in, _teardown_objects),
    cmocka_unit_test_teardown(test_click_outside_object_steps_out, _teardown_objects),
    cmocka_unit_test_teardown(test_double_click_on_other_object_moves_into_it,
                              _teardown_objects),
    cmocka_unit_test_teardown(test_secondary_button_never_steps, _teardown_objects),
    cmocka_unit_test_teardown(test_double_click_on_plain_shape_does_nothing, _teardown_objects),
    cmocka_unit_test_teardown(test_object_paths_act_as_one_until_stepped_in, _teardown_objects),
    cmocka_unit_test_teardown(test_right_click_on_object_path_removes_the_object,
                              _teardown_objects),
    cmocka_unit_test_teardown(test_right_click_inside_object_removes_one_path,
                              _teardown_objects),
    cmocka_unit_test_teardown(test_node_deletion_removes_one_path, _teardown_objects),
    cmocka_unit_test_teardown(test_last_path_removes_the_object, _teardown_objects),
    cmocka_unit_test_teardown(test_object_removal_without_canvas_session, _teardown_objects),
    cmocka_unit_test_teardown(test_canvas_delete_of_shape_is_the_panel_delete,
                              _teardown_objects),
    cmocka_unit_test_teardown(test_canvas_path_selects_its_object_row, _teardown_objects),
    cmocka_unit_test_teardown(test_refine_scope_of_removed_element_falls_back_to_its_group,
                              _teardown_objects),
    cmocka_unit_test_teardown(test_refine_scope_with_nothing_left_is_the_whole_mask,
                              _teardown_objects),
    cmocka_unit_test_teardown(test_refine_scope_of_present_element_is_kept, _teardown_objects),
    cmocka_unit_test_teardown(test_refine_scope_follows_the_selection, _teardown_objects),
    cmocka_unit_test_teardown(test_raster_element_follows_its_source_name, _teardown_raster),
    cmocka_unit_test_teardown(test_renamed_raster_element_keeps_its_name, _teardown_raster),
    cmocka_unit_test_teardown(test_emptied_raster_name_follows_the_source_again,
                              _teardown_raster),
    cmocka_unit_test_teardown(test_raster_source_is_found_by_operation_and_instance,
                              _teardown_raster),
    cmocka_unit_test_teardown(test_old_raster_name_follows_its_source, _teardown_raster),
    cmocka_unit_test_teardown(test_list_signature_follows_source_rename, _teardown_raster),
    cmocka_unit_test_teardown(test_cleanup_keeps_the_paths_of_a_used_object,
                              _teardown_objects),
    cmocka_unit_test_teardown(test_empty_object_is_removed, _teardown_objects),
    cmocka_unit_test_teardown(test_object_with_a_path_is_kept, _teardown_objects),
    cmocka_unit_test_teardown(test_add_target_is_the_only_group, _teardown),
    cmocka_unit_test_teardown(test_add_target_ignores_a_stale_selection_with_one_group,
                              _teardown),
    cmocka_unit_test_teardown(test_add_target_is_ambiguous_with_two_groups, _teardown),
    cmocka_unit_test_teardown(test_add_target_counts_a_staged_group, _teardown),
    cmocka_unit_test_teardown(test_add_target_skips_a_group_the_panel_hides, _teardown),
    cmocka_unit_test_teardown(test_add_target_is_the_shown_group_above_a_hidden_one,
                              _teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
