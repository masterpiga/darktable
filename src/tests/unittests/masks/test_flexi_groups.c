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

// Group identity and isolation state.
//
// Three clusters of behaviour that share the property of being invisible until
// they go wrong, and of being about *which group is which* rather than about
// pixels:
//
//   * the solo family -- solo, group-solo and solo-edit are mutually exclusive,
//     and at most one element OR one group is soloed at any time;
//   * group numbering -- a group's displayed number is an identity, so it must
//     survive the group emptying and refilling, and must not be reused while a
//     peer still shows it;
//   * refinement scope -- which level (whole mask, group, element) a refinement
//     applies to, and the disjoint key spaces the bypass set uses.

#include "flexi_fixture.h"
#include "develop/pixelpipe.h"

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <cmocka.h>

static int _teardown(void **state)
{
  flexi_teardown();
  return 0;
}

static gboolean _hidden(const dt_mask_id_t fid)
{
  return (_group_point(flexi_group(), fid)->state & DT_MASKS_STATE_HIDDEN) != 0;
}

static gboolean _any_hidden(void)
{
  for(GList *l = flexi_group()->points; l; l = g_list_next(l))
    if(((dt_masks_point_group_t *)l->data)->state & DT_MASKS_STATE_HIDDEN)
      return TRUE;
  return FALSE;
}

// ---------------------------------------------------------------------------
// solo: at most one thing at a time
// ---------------------------------------------------------------------------

static void test_solo_element_hides_the_others(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3");
  _model_toggle_solo_form(&flexi_module, grp, 2);

  assert_int_equal(flexi_bd.solo_formid, 2);
  assert_true(_hidden(1));
  assert_false(_hidden(2));
  assert_true(_hidden(3));
}

static void test_solo_same_element_again_turns_it_off(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3");
  _model_toggle_solo_form(&flexi_module, grp, 2);
  _model_toggle_solo_form(&flexi_module, grp, 2);

  assert_int_equal(flexi_bd.solo_formid, INVALID_MASKID);
  assert_false(_any_hidden());
}

// soloing a second element moves the solo rather than adding one
static void test_soloing_another_element_moves_the_solo(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3");
  _model_toggle_solo_form(&flexi_module, grp, 2);
  _model_toggle_solo_form(&flexi_module, grp, 3);

  assert_int_equal(flexi_bd.solo_formid, 3);
  assert_true(_hidden(1));
  assert_true(_hidden(2));
  assert_false(_hidden(3));
}

static void test_solo_group_hides_other_groups(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3,4");
  GList *members = g_list_append(g_list_append(NULL, GINT_TO_POINTER(3)),
                                 GINT_TO_POINTER(4));
  _model_toggle_solo_group(&flexi_module, grp, 3, members);
  g_list_free(members);

  assert_int_equal(flexi_bd.solo_group_key, 3);
  assert_true(_hidden(1));
  assert_true(_hidden(2));
  assert_false(_hidden(3));
  assert_false(_hidden(4));
}

// an element solo and a group solo cannot both be active
static void test_group_solo_cancels_element_solo(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3,4");
  _model_toggle_solo_form(&flexi_module, grp, 1);
  assert_int_equal(flexi_bd.solo_formid, 1);

  GList *members = g_list_append(NULL, GINT_TO_POINTER(3));
  _model_toggle_solo_group(&flexi_module, grp, 3, members);
  g_list_free(members);

  assert_int_equal(flexi_bd.solo_formid, INVALID_MASKID);
  assert_int_equal(flexi_bd.solo_group_key, 3);
}

static void test_element_solo_cancels_group_solo(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3,4");
  GList *members = g_list_append(NULL, GINT_TO_POINTER(3));
  _model_toggle_solo_group(&flexi_module, grp, 3, members);
  g_list_free(members);
  assert_int_equal(flexi_bd.solo_group_key, 3);

  _model_toggle_solo_form(&flexi_module, grp, 1);
  assert_int_equal(flexi_bd.solo_group_key, 0);
  assert_int_equal(flexi_bd.solo_formid, 1);
}

// ---------------------------------------------------------------------------
// solo vs solo-edit: mutually exclusive, and they mean different things
// ---------------------------------------------------------------------------

// solo-edit narrows what is *editable*; it must not hide anything
static void test_solo_edit_hides_nothing(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3");
  const dt_masks_solo_canvas_t c = _model_toggle_soloedit(&flexi_module, grp, 2);

  assert_int_equal(c, DT_MASKS_SOLO_CANVAS_ONE);
  assert_int_equal(flexi_bd.soloedit_formid, 2);
  assert_false(_any_hidden());
}

// turning on solo-edit drops any active solo AND restores every element's
// visibility -- solo-edit isolates what is editable, not what is shown
static void test_solo_edit_cancels_solo_and_unhides(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3");
  _model_toggle_solo_form(&flexi_module, grp, 2);
  assert_true(_any_hidden());

  _model_toggle_soloedit(&flexi_module, grp, 1);

  assert_int_equal(flexi_bd.solo_formid, INVALID_MASKID);
  assert_int_equal(flexi_bd.solo_group_key, 0);
  assert_int_equal(flexi_bd.soloedit_formid, 1);
  assert_false(_any_hidden());
}

static void test_solo_edit_cancels_group_solo_and_unhides(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3,4");
  GList *members = g_list_append(NULL, GINT_TO_POINTER(3));
  _model_toggle_solo_group(&flexi_module, grp, 3, members);
  g_list_free(members);
  assert_true(_any_hidden());

  _model_toggle_soloedit(&flexi_module, grp, 1);

  assert_int_equal(flexi_bd.solo_group_key, 0);
  assert_false(_any_hidden());
}

// and the other direction: turning on solo drops any active solo-edit
static void test_solo_cancels_solo_edit(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3");
  _model_toggle_soloedit(&flexi_module, grp, 1);
  assert_int_equal(flexi_bd.soloedit_formid, 1);

  const dt_masks_solo_canvas_t c = _model_toggle_solo_form(&flexi_module, grp, 3);
  assert_int_equal(flexi_bd.soloedit_formid, INVALID_MASKID);
  // the caller is told to restore whole-group canvas editing
  assert_int_equal(c, DT_MASKS_SOLO_CANVAS_FULL);
}

static void test_group_solo_cancels_solo_edit(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3,4");
  _model_toggle_soloedit(&flexi_module, grp, 1);

  GList *members = g_list_append(NULL, GINT_TO_POINTER(3));
  const dt_masks_solo_canvas_t c =
    _model_toggle_solo_group(&flexi_module, grp, 3, members);
  g_list_free(members);

  assert_int_equal(flexi_bd.soloedit_formid, INVALID_MASKID);
  assert_int_equal(c, DT_MASKS_SOLO_CANVAS_FULL);
}

// solo-edit on the same element again turns it off and restores full editing
static void test_solo_edit_toggles_off(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2");
  _model_toggle_soloedit(&flexi_module, grp, 1);
  const dt_masks_solo_canvas_t c = _model_toggle_soloedit(&flexi_module, grp, 1);

  assert_int_equal(flexi_bd.soloedit_formid, INVALID_MASKID);
  assert_int_equal(c, DT_MASKS_SOLO_CANVAS_FULL);
}

static void _assert_one_isolation_mode(const char *after)
{
  int active = 0;
  if(dt_is_valid_maskid(flexi_bd.solo_formid)) active++;
  if(flexi_bd.solo_group_key != 0) active++;
  if(dt_is_valid_maskid(flexi_bd.soloedit_formid)) active++;
  if(active > 1)
    fail_msg("after %s: %d isolation modes active at once "
             "(solo=%d group=%u soloedit=%d)", after, active,
             (int)flexi_bd.solo_formid, flexi_bd.solo_group_key,
             (int)flexi_bd.soloedit_formid);
}

// Never more than one of the three at a time -- checked after EVERY toggle,
// not just at the end of a sequence. Checking only at the end lets a violation
// hide whenever the last toggle happens to be the one that clears the others,
// which is exactly what a solo-edit-last ordering does.
static void test_at_most_one_isolation_mode_is_ever_active(void **state)
{
  // every ordering of the three toggles, so no single sequence can mask a
  // one-directional failure to cancel
  const int orders[6][3] = { {0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0} };

  for(int o = 0; o < 6; o++)
  {
    dt_masks_form_t *grp = flexi_build("u:1,2 | i:3,4");
    GList *members = g_list_append(NULL, GINT_TO_POINTER(3));

    for(int step = 0; step < 3; step++)
    {
      switch(orders[o][step])
      {
        case 0:
          _model_toggle_solo_form(&flexi_module, grp, 1);
          _assert_one_isolation_mode("solo element");
          break;
        case 1:
          _model_toggle_solo_group(&flexi_module, grp, 3, members);
          _assert_one_isolation_mode("solo group");
          break;
        default:
          _model_toggle_soloedit(&flexi_module, grp, 2);
          _assert_one_isolation_mode("solo-edit");
          break;
      }
    }
    g_list_free(members);
    flexi_teardown();
  }
}

// soloing an element that a previous solo had hidden must not leave a solo-edit
// pointing at something invisible
static void test_solo_edit_cleared_when_its_element_gets_hidden(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3");
  _model_toggle_soloedit(&flexi_module, grp, 1);
  // solo a different element -- 1 becomes hidden, so its solo-edit is stale
  const dt_masks_solo_canvas_t c = _model_toggle_solo_form(&flexi_module, grp, 3);

  assert_true(_hidden(1));
  assert_int_equal(flexi_bd.soloedit_formid, INVALID_MASKID);
  assert_int_equal(c, DT_MASKS_SOLO_CANVAS_FULL);
}

static void test_solo_of_unknown_element_is_rejected(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2");
  _model_toggle_solo_form(&flexi_module, grp, 77);
  assert_int_equal(flexi_bd.solo_formid, INVALID_MASKID);
  assert_false(_any_hidden());
}

// ---------------------------------------------------------------------------
// per-element disable, the third isolation-shaped control
// ---------------------------------------------------------------------------

// DISABLE is a separate, persistent bit -- unlike solo it is stored per element
// and survives; it must not be entangled with the HIDDEN bit solo uses
static void test_disable_is_independent_of_solo(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3");
  _group_point(grp, 2)->state |= DT_MASKS_STATE_DISABLE;

  _model_toggle_solo_form(&flexi_module, grp, 1);
  _model_toggle_solo_form(&flexi_module, grp, 1); // and off again

  // solo cleared every HIDDEN bit, but DISABLE is untouched
  assert_false(_any_hidden());
  assert_int_not_equal(_group_point(grp, 2)->state & DT_MASKS_STATE_DISABLE, 0);
}

static void test_group_bypass_is_independent_of_disable(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3");
  dt_masks_point_group_t *pt = _group_point(grp, 3);
  pt->state |= DT_MASKS_STATE_OP_BYPASS;
  pt->state |= DT_MASKS_STATE_DISABLE;

  // both bits coexist, and neither disturbs the operator underneath
  assert_int_equal(pt->state & DT_MASKS_STATE_OP_COMBINE, DT_MASKS_STATE_INTERSECTION);
  assert_int_not_equal(pt->state & DT_MASKS_STATE_OP_BYPASS, 0);
  assert_int_not_equal(pt->state & DT_MASKS_STATE_DISABLE, 0);
}

// ---------------------------------------------------------------------------
// group numbering
// ---------------------------------------------------------------------------

static void test_ordinal_max_is_per_operator(void **state)
{
  flexi_build("u:1,2 | i:3 | u:4");
  flexi_set_ordinal(1, 1); // union 1
  flexi_set_ordinal(4, 2); // union 2
  flexi_set_ordinal(3, 1); // intersection 1

  const int uop = _op_index_for_state(DT_MASKS_STATE_UNION);
  const int iop = _op_index_for_state(DT_MASKS_STATE_INTERSECTION);
  assert_int_equal(_group_ord_max_for_op(&flexi_module, uop), 2);
  assert_int_equal(_group_ord_max_for_op(&flexi_module, iop), 1);
}

// a staged group holds a number too, so a new group must not reuse it
static void test_ordinal_max_counts_staged_groups(void **state)
{
  flexi_build("u:1,2");
  flexi_set_ordinal(1, 1);
  dt_masks_empty_group_t *eg = flexi_add_empty(DT_MASKS_STATE_UNION, 1);
  eg->ordinal = 5;

  const int uop = _op_index_for_state(DT_MASKS_STATE_UNION);
  assert_int_equal(_group_ord_max_for_op(&flexi_module, uop), 5);
}

static void test_ordinal_of_cid_reads_back(void **state)
{
  flexi_build("u:1,2 | i:3");
  flexi_set_ordinal(3, 7);
  assert_int_equal(_group_ordinal_of_cid(&flexi_module, 3), 7);
}

// numbers whose group no longer exists are dropped, so a series can restart at
// 1 once emptied -- and the table does not grow across edits
static void test_pruning_drops_numbers_of_vanished_groups(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3");
  flexi_set_ordinal(1, 1);
  flexi_set_ordinal(3, 1);

  // dissolve the intersection group by moving its only member into the union
  _model_drop_element_onto_group(&flexi_module, grp, 3, 1);
  // the placeholder left behind keeps the group alive, so drop it too
  flexi_teardown();

  flexi_build("u:1,2");
  flexi_set_ordinal(1, 1);
  flexi_set_ordinal(99, 4); // a group that no longer exists
  _prune_group_ordinals(&flexi_module);

  assert_int_equal(flexi_get_ordinal(1), 1);
  assert_int_equal(flexi_get_ordinal(99), 0);
}

// a group solo whose group has been dissolved must not linger -- left stale,
// every row keeps reading "some group is soloed and it isn't me" and dims
static void test_pruning_clears_stale_group_solo(void **state)
{
  flexi_build("u:1,2");
  flexi_bd.solo_group_key = 99; // no run is headed by 99

  _prune_stale_solo(&flexi_module);
  assert_int_equal(flexi_bd.solo_group_key, 0);
}

static void test_pruning_keeps_a_live_group_solo(void **state)
{
  flexi_build("u:1,2 | i:3");
  flexi_bd.solo_group_key = 3; // 3 heads a real run

  _prune_stale_solo(&flexi_module);
  assert_int_equal(flexi_bd.solo_group_key, 3);
}

// a group solo keyed on a member that is not its run's head is stale too
static void test_pruning_clears_solo_keyed_on_non_head(void **state)
{
  flexi_build("u:1,2 | i:3");
  flexi_bd.solo_group_key = 2; // 2 is a member, not a head

  _prune_stale_solo(&flexi_module);
  assert_int_equal(flexi_bd.solo_group_key, 0);
}

// ---------------------------------------------------------------------------
// refinement scope
// ---------------------------------------------------------------------------

// the three key spaces share one table and must never collide: an element key
// and a group key built from the same id are different keys
static void test_refine_key_spaces_are_disjoint(void **state)
{
  for(dt_mask_id_t id = 1; id < 64; id++)
  {
    const guint32 ek = dt_masks_refine_key_element(id);
    const guint32 gk = dt_masks_refine_key_group(id);
    assert_int_not_equal(ek, gk);
    assert_int_not_equal(ek, DT_MASKS_REFINE_KEY_GLOBAL);
    assert_int_not_equal(gk, DT_MASKS_REFINE_KEY_GLOBAL);
  }
}

// the group space is marked by the top bit, which no mask id ever uses
static void test_refine_group_key_uses_the_flag_bit(void **state)
{
  const guint32 gk = dt_masks_refine_key_group(5);
  assert_int_not_equal(gk & DT_MASKS_REFINE_KEY_GROUP_FLAG, 0);
  assert_int_equal(gk & ~DT_MASKS_REFINE_KEY_GROUP_FLAG, 5u);
  assert_int_equal(dt_masks_refine_key_element(5) & DT_MASKS_REFINE_KEY_GROUP_FLAG,
                   0u);
}

// element and group refinement are stored the same way but mean different
// things: element applies before compositing, group applies once to the
// composited group mask
static void test_refine_scope_element_vs_group(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3,4");

  _group_point(grp, 1)->refinement.enabled = DT_MASKS_REFINE_ELEMENT;
  _group_point(grp, 1)->refinement.blur_radius = 2.0f;
  // group scope is a broadcast copy onto every member of the run
  _group_point(grp, 3)->refinement.enabled = DT_MASKS_REFINE_GROUP;
  _group_point(grp, 3)->refinement.blur_radius = 4.0f;
  _group_point(grp, 4)->refinement.enabled = DT_MASKS_REFINE_GROUP;
  _group_point(grp, 4)->refinement.blur_radius = 4.0f;

  assert_int_equal(_group_point(grp, 1)->refinement.enabled, DT_MASKS_REFINE_ELEMENT);
  assert_int_equal(_group_point(grp, 2)->refinement.enabled, DT_MASKS_REFINE_OFF);
  assert_int_equal(_group_point(grp, 3)->refinement.enabled, DT_MASKS_REFINE_GROUP);
  assert_int_equal(_group_point(grp, 4)->refinement.enabled, DT_MASKS_REFINE_GROUP);
}

// OFF is the zero value, so a zero-filled (pre-v7) point reads as no refinement
static void test_refine_off_is_the_zero_value(void **state)
{
  assert_int_equal(DT_MASKS_REFINE_OFF, 0);
  dt_masks_form_t *grp = flexi_build("u:1,2");
  assert_int_equal(_group_point(grp, 1)->refinement.enabled, DT_MASKS_REFINE_OFF);
}

// a moved element carries its own element-scoped refinement with it
static void test_element_refinement_follows_a_move(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2 | i:3");
  _group_point(grp, 2)->refinement.enabled = DT_MASKS_REFINE_ELEMENT;
  _group_point(grp, 2)->refinement.contrast = 0.5f;

  _model_drop_element_onto_element(&flexi_module, grp, 2, 3, TRUE);

  assert_int_equal(_group_point(grp, 2)->refinement.enabled, DT_MASKS_REFINE_ELEMENT);
  assert_float_equal(_group_point(grp, 2)->refinement.contrast, 0.5f, 1e-6);
}

// ---------------------------------------------------------------------------
// the refinement-bypass snapshot
//
// The GUI holds bypassed keys in a hash table; commit_params copies the subset
// that applies to this mask into a sorted array on the pipe piece, which the
// renderer then binary-searches. Three things can go wrong quietly: the array
// not being sorted (making the search miss), keys that are not mask ids leaking
// in, and the hash not being canonical (so the cache misses or over-holds).
// ---------------------------------------------------------------------------

static dt_dev_pixelpipe_iop_t *_piece = NULL;

static void _bypass(const guint32 key)
{
  if(!flexi_bd.masks_refine_bypassed)
    flexi_bd.masks_refine_bypassed =
      g_hash_table_new(g_direct_hash, g_direct_equal);
  g_hash_table_insert(flexi_bd.masks_refine_bypassed, GUINT_TO_POINTER(key),
                      GINT_TO_POINTER(1));
}

static void _commit(void)
{
  if(!_piece) _piece = calloc(1, sizeof(dt_dev_pixelpipe_iop_t));
  _piece->blendop_data = &flexi_bp;
  dt_masks_refine_bypass_commit(&flexi_module, _piece);
}

static int _bypass_teardown(void **state)
{
  if(_piece)
  {
    dt_masks_refine_bypass_cleanup(&_piece->refine_bypass);
    free(_piece);
    _piece = NULL;
  }
  if(flexi_bd.masks_refine_bypassed)
  {
    g_hash_table_destroy(flexi_bd.masks_refine_bypassed);
    flexi_bd.masks_refine_bypassed = NULL;
  }
  flexi_teardown();
  return 0;
}

static void test_bypass_snapshot_finds_every_committed_key(void **state)
{
  flexi_build("u:1,2 | i:3,4");
  _bypass(DT_MASKS_REFINE_KEY_GLOBAL);
  _bypass(dt_masks_refine_key_element(2));
  _bypass(dt_masks_refine_key_group(3));
  _commit();

  assert_true(dt_masks_refine_bypass_lookup(&_piece->refine_bypass,
                                            DT_MASKS_REFINE_KEY_GLOBAL));
  assert_true(dt_masks_refine_bypass_lookup(&_piece->refine_bypass,
                                            dt_masks_refine_key_element(2)));
  assert_true(dt_masks_refine_bypass_lookup(&_piece->refine_bypass,
                                            dt_masks_refine_key_group(3)));
  // and nothing else
  assert_false(dt_masks_refine_bypass_lookup(&_piece->refine_bypass,
                                             dt_masks_refine_key_element(1)));
  assert_false(dt_masks_refine_bypass_lookup(&_piece->refine_bypass,
                                             dt_masks_refine_key_group(2)));
}

// the renderer binary-searches, so the committed array must be sorted -- with
// enough keys that an unsorted array would actually make the search miss
static void test_bypass_snapshot_is_searchable_at_scale(void **state)
{
  flexi_build("u:1,2,3,4,5,6,7,8 | i:9,10,11,12");
  for(dt_mask_id_t id = 1; id <= 12; id++)
  {
    _bypass(dt_masks_refine_key_element(id));
    _bypass(dt_masks_refine_key_group(id));
  }
  _commit();

  // sorted, as the binary search requires
  for(int i = 1; i < _piece->refine_bypass.nkeys; i++)
    if(_piece->refine_bypass.keys[i - 1] >= _piece->refine_bypass.keys[i])
      fail_msg("committed bypass keys are not sorted at index %d", i);

  // every one is findable -- element and group keys interleave across the
  // top-bit boundary, which is exactly where a search bug would show
  for(dt_mask_id_t id = 1; id <= 12; id++)
  {
    if(!dt_masks_refine_bypass_lookup(&_piece->refine_bypass,
                                      dt_masks_refine_key_element(id)))
      fail_msg("element key for %d not found in the snapshot", (int)id);
    if(!dt_masks_refine_bypass_lookup(&_piece->refine_bypass,
                                      dt_masks_refine_key_group(id)))
      fail_msg("group key for %d not found in the snapshot", (int)id);
  }
}

// staged groups are keyed in the GUI table by their own pointer, which must
// never be mistaken for a mask id and copied into the snapshot
static void test_bypass_snapshot_excludes_staged_group_keys(void **state)
{
  flexi_build("u:1,2");
  dt_masks_empty_group_t *eg = flexi_add_empty(DT_MASKS_STATE_INTERSECTION, 1);
  _bypass(GPOINTER_TO_UINT(eg));
  _bypass(dt_masks_refine_key_element(1));
  _commit();

  // only the real element key made it across
  assert_int_equal(_piece->refine_bypass.nkeys, 1);
  assert_true(dt_masks_refine_bypass_lookup(&_piece->refine_bypass,
                                            dt_masks_refine_key_element(1)));
}

static void test_bypass_lookup_on_an_empty_snapshot(void **state)
{
  flexi_build("u:1,2");
  _commit(); // nothing bypassed
  assert_false(dt_masks_refine_bypass_lookup(&_piece->refine_bypass,
                                             dt_masks_refine_key_element(1)));
  assert_false(dt_masks_refine_bypass_lookup(&_piece->refine_bypass,
                                             DT_MASKS_REFINE_KEY_GLOBAL));
  assert_int_equal(dt_masks_refine_bypass_hash(&_piece->refine_bypass), DT_INITHASH);
}

// a classic (non-flexi) mask has no bypass feature at all; the snapshot must
// come back empty rather than carrying stale GUI state
static void test_bypass_snapshot_is_flexi_only(void **state)
{
  flexi_build("u:1,2");
  _bypass(dt_masks_refine_key_element(1));
  flexi_bp.mask_mode = DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK; // classic
  _commit();

  assert_int_equal(_piece->refine_bypass.nkeys, 0);
  assert_false(dt_masks_refine_bypass_lookup(&_piece->refine_bypass,
                                             dt_masks_refine_key_element(1)));
}

// the hash feeds mask-cache invalidation, so it must depend on the set and not
// on the order the keys happened to be inserted in
static void test_bypass_hash_is_canonical(void **state)
{
  flexi_build("u:1,2 | i:3");
  _bypass(dt_masks_refine_key_element(3));
  _bypass(dt_masks_refine_key_element(1));
  _commit();
  const dt_hash_t first = dt_masks_refine_bypass_hash(&_piece->refine_bypass);

  _bypass_teardown(NULL);

  flexi_build("u:1,2 | i:3");
  _bypass(dt_masks_refine_key_element(1)); // opposite insertion order
  _bypass(dt_masks_refine_key_element(3));
  _commit();
  assert_int_equal(dt_masks_refine_bypass_hash(&_piece->refine_bypass), first);
}

static void test_bypass_hash_changes_with_the_set(void **state)
{
  flexi_build("u:1,2 | i:3");
  _bypass(dt_masks_refine_key_element(1));
  _commit();
  const dt_hash_t one = dt_masks_refine_bypass_hash(&_piece->refine_bypass);

  _bypass(dt_masks_refine_key_element(3));
  _commit();
  assert_int_not_equal(dt_masks_refine_bypass_hash(&_piece->refine_bypass), one);
}

// an element key and a group key built from the same id must not collide in
// the snapshot either -- the disjointness tested above has to survive commit
static void test_bypass_snapshot_keeps_key_spaces_apart(void **state)
{
  flexi_build("u:1,2 | i:3");
  _bypass(dt_masks_refine_key_group(3));
  _commit();

  assert_true(dt_masks_refine_bypass_lookup(&_piece->refine_bypass,
                                            dt_masks_refine_key_group(3)));
  assert_false(dt_masks_refine_bypass_lookup(&_piece->refine_bypass,
                                             dt_masks_refine_key_element(3)));
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_teardown(test_solo_element_hides_the_others, _teardown),
    cmocka_unit_test_teardown(test_solo_same_element_again_turns_it_off, _teardown),
    cmocka_unit_test_teardown(test_soloing_another_element_moves_the_solo, _teardown),
    cmocka_unit_test_teardown(test_solo_group_hides_other_groups, _teardown),
    cmocka_unit_test_teardown(test_group_solo_cancels_element_solo, _teardown),
    cmocka_unit_test_teardown(test_element_solo_cancels_group_solo, _teardown),
    cmocka_unit_test_teardown(test_solo_edit_hides_nothing, _teardown),
    cmocka_unit_test_teardown(test_solo_edit_cancels_solo_and_unhides, _teardown),
    cmocka_unit_test_teardown(test_solo_edit_cancels_group_solo_and_unhides, _teardown),
    cmocka_unit_test_teardown(test_solo_cancels_solo_edit, _teardown),
    cmocka_unit_test_teardown(test_group_solo_cancels_solo_edit, _teardown),
    cmocka_unit_test_teardown(test_solo_edit_toggles_off, _teardown),
    cmocka_unit_test_teardown(test_at_most_one_isolation_mode_is_ever_active, _teardown),
    cmocka_unit_test_teardown(test_solo_edit_cleared_when_its_element_gets_hidden, _teardown),
    cmocka_unit_test_teardown(test_solo_of_unknown_element_is_rejected, _teardown),
    cmocka_unit_test_teardown(test_disable_is_independent_of_solo, _teardown),
    cmocka_unit_test_teardown(test_group_bypass_is_independent_of_disable, _teardown),
    cmocka_unit_test_teardown(test_ordinal_max_is_per_operator, _teardown),
    cmocka_unit_test_teardown(test_ordinal_max_counts_staged_groups, _teardown),
    cmocka_unit_test_teardown(test_ordinal_of_cid_reads_back, _teardown),
    cmocka_unit_test_teardown(test_pruning_drops_numbers_of_vanished_groups, _teardown),
    cmocka_unit_test_teardown(test_pruning_clears_stale_group_solo, _teardown),
    cmocka_unit_test_teardown(test_pruning_keeps_a_live_group_solo, _teardown),
    cmocka_unit_test_teardown(test_pruning_clears_solo_keyed_on_non_head, _teardown),
    cmocka_unit_test_teardown(test_refine_key_spaces_are_disjoint, _teardown),
    cmocka_unit_test_teardown(test_refine_group_key_uses_the_flag_bit, _teardown),
    cmocka_unit_test_teardown(test_refine_scope_element_vs_group, _teardown),
    cmocka_unit_test_teardown(test_refine_off_is_the_zero_value, _teardown),
    cmocka_unit_test_teardown(test_element_refinement_follows_a_move, _teardown),
    cmocka_unit_test_teardown(test_bypass_snapshot_finds_every_committed_key, _bypass_teardown),
    cmocka_unit_test_teardown(test_bypass_snapshot_is_searchable_at_scale, _bypass_teardown),
    cmocka_unit_test_teardown(test_bypass_snapshot_excludes_staged_group_keys, _bypass_teardown),
    cmocka_unit_test_teardown(test_bypass_lookup_on_an_empty_snapshot, _bypass_teardown),
    cmocka_unit_test_teardown(test_bypass_snapshot_is_flexi_only, _bypass_teardown),
    cmocka_unit_test_teardown(test_bypass_hash_is_canonical, _bypass_teardown),
    cmocka_unit_test_teardown(test_bypass_hash_changes_with_the_set, _bypass_teardown),
    cmocka_unit_test_teardown(test_bypass_snapshot_keeps_key_spaces_apart, _bypass_teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
