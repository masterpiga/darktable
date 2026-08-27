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

// The classic -> flexi migration case table (masks/migrate_legacy.c).
//
// The pixel suite under src/tests/masking/flexi/ proves that migrated masks
// *render* the same as classic ones, for the combinations it has fixtures for.
// It cannot reach the rest of the table: several bit combinations are
// unreachable from any GUI, and so cannot be produced by hand-authoring an
// edit, yet can perfectly well arrive in a stored or foreign XMP. What those
// degrade to is a decision, not an accident, and it is asserted here.
//
// These tests check *structure* -- the resulting mask_mode and the forms
// synthesized -- which is cheap, exhaustive, and complementary to the pixel
// suite's much narrower but deeper guarantee.
//
// history_num is -1 throughout: that is the no-database path, where synthesis
// happens inline instead of being deferred into dev->pending_flexi_migrations
// for the darkroom loader to write under the final history row.

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

// a module carrying a classic blend_params with `mode` set, ready to migrate
static void _classic(const uint32_t mode)
{
  flexi_build("u:1,2");           // gives us a real group at FLEXI_GROUP_ID
  g_strlcpy(flexi_module.op, "exposure", sizeof(flexi_module.op)); // logs only
  flexi_bp.mask_mode = mode;
  flexi_bp.blendif = 0;
  // a real colorspace: _classify_conditional derives its channel-polarity mask
  // from blend_cst, and an unset one has no channels at all, which makes every
  // blendif classify as degenerate (see the degenerate-branch tests below)
  flexi_bp.blend_cst = DEVELOP_BLEND_CS_RGB_SCENE;
  flexi_bp.mask_combine = DEVELOP_COMBINE_NORM_EXCL;
  flexi_bp.opacity = 1.0f;
  flexi_bp.raster_mask_source[0] = '\0';
}

// a blendif value with one genuinely active channel for the fixture's
// colorspace -- the DT_COND_REAL branch, i.e. a parametric mask that actually
// restricts something
static uint32_t _active_channel_bit(void)
{
  const dt_iop_gui_blendif_channel_t *ch =
    dt_develop_blendif_channels_for_csp(DEVELOP_BLEND_CS_RGB_SCENE);
  assert_non_null(ch);
  return 1u << ch[0].param_channels[0];
}

static gboolean _migrate(void)
{
  return dt_masks_migrate_classic_to_flexi(&flexi_module, &flexi_bp, -1);
}

// how many forms of `type` the fixture's dev now holds
static int _count_forms(const dt_masks_type_t type)
{
  int n = 0;
  for(GList *l = flexi_dev.forms; l; l = g_list_next(l))
    if(((dt_masks_form_t *)l->data)->type & type) n++;
  return n;
}

static void _assert_flexi(void)
{
  if(!(flexi_bp.mask_mode & DEVELOP_MASK_FLEXI))
    fail_msg("mask_mode 0x%x is not flexi after migration", flexi_bp.mask_mode);
  if(!(flexi_bp.mask_mode & DEVELOP_MASK_ENABLED))
    fail_msg("mask_mode 0x%x lost ENABLED during migration", flexi_bp.mask_mode);
}

// ---------------------------------------------------------------------------
// cases 0-1: nothing to migrate
// ---------------------------------------------------------------------------

static void test_disabled_stays_disabled(void **state)
{
  _classic(DEVELOP_MASK_DISABLED);
  assert_true(_migrate());
  assert_int_equal(flexi_bp.mask_mode, DEVELOP_MASK_DISABLED);
}

// a uniform-opacity blend has no form to point at, but is normalized to a
// flexi state so "every mask_mode is DISABLED or flexi" holds afterwards
static void test_uniform_enabled_becomes_flexi(void **state)
{
  _classic(DEVELOP_MASK_ENABLED);
  assert_true(_migrate());
  _assert_flexi();
}

// ---------------------------------------------------------------------------
// case 2: drawn only -- zero transform
// ---------------------------------------------------------------------------

// flexi renders a drawn group through the identical code path, so the group is
// reused verbatim: no new form, and mask_id untouched
static void test_drawn_only_reuses_the_group(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK);
  const dt_mask_id_t before = flexi_bp.mask_id;
  const int forms_before = g_list_length(flexi_dev.forms);

  assert_true(_migrate());
  _assert_flexi();
  assert_int_equal(flexi_bp.mask_id, before);
  assert_int_equal((int)g_list_length(flexi_dev.forms), forms_before);
}

// defensive: a mask_id that resolves to nothing must still migrate cleanly --
// flexi's "no form" fallback matches classic's, so nothing is fabricated
static void test_drawn_with_dangling_mask_id(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK);
  flexi_bp.mask_id = 4242; // no such form
  assert_true(_migrate());
  _assert_flexi();
}

// ---------------------------------------------------------------------------
// case 3: parametric only -- synthesized as a form
// ---------------------------------------------------------------------------

static void test_parametric_only_synthesizes_a_parametric_form(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_CONDITIONAL);
  flexi_bp.blendif = _active_channel_bit();
  const int before = _count_forms(DT_MASKS_PARAMETRIC);

  assert_true(_migrate());
  _assert_flexi();
  assert_int_equal(_count_forms(DT_MASKS_PARAMETRIC), before + 1);
}

// ---------------------------------------------------------------------------
// case 4: drawn AND parametric
// ---------------------------------------------------------------------------

// the classic renderer multiplies the two together, so the drawn group is left
// untouched and a parametric element is stacked onto it
static void test_drawn_and_parametric_stacks_a_parametric_element(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK | DEVELOP_MASK_CONDITIONAL);
  flexi_bp.blendif = _active_channel_bit();
  const int before = _count_forms(DT_MASKS_PARAMETRIC);

  assert_true(_migrate());
  _assert_flexi();
  assert_int_equal(_count_forms(DT_MASKS_PARAMETRIC), before + 1);
}

// ---------------------------------------------------------------------------
// case 5: raster
// ---------------------------------------------------------------------------

static void test_raster_synthesizes_a_raster_form(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_RASTER);
  g_strlcpy(flexi_bp.raster_mask_source, "colorbalancergb",
            sizeof(flexi_bp.raster_mask_source));
  flexi_bp.raster_mask_instance = 0;
  flexi_bp.raster_mask_id = 0;
  const int before = _count_forms(DT_MASKS_RASTER);

  assert_true(_migrate());
  _assert_flexi();
  assert_int_equal(_count_forms(DT_MASKS_RASTER), before + 1);
}

// classic stores the raster inversion in its own field; the new struct has no
// such field, so it must move onto the point's own INVERSE state bit
static void test_raster_inversion_moves_onto_the_state_bit(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_RASTER);
  g_strlcpy(flexi_bp.raster_mask_source, "colorbalancergb",
            sizeof(flexi_bp.raster_mask_source));
  flexi_bp.raster_mask_invert = TRUE;

  assert_true(_migrate());
  _assert_flexi();

  // find the group the migration pointed us at, and check its raster member
  dt_masks_form_t *grp = dt_masks_get_from_id(&flexi_dev, flexi_bp.mask_id);
  assert_non_null(grp);
  gboolean found_inverted = FALSE;
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    const dt_masks_form_t *f = dt_masks_get_from_id(&flexi_dev, pt->formid);
    if(f && (f->type & DT_MASKS_RASTER) && (pt->state & DT_MASKS_STATE_INVERSE))
      found_inverted = TRUE;
  }
  assert_true(found_inverted);
}

// ---------------------------------------------------------------------------
// case 6: RASTER combined with MASK / CONDITIONAL -- unreachable from the GUI
// ---------------------------------------------------------------------------

// the classic renderer is an if/else chain where raster wins outright, so
// faithfully reproducing it means the other bits' data is dropped, not merged.
// No GUI can produce this; a stored or foreign XMP can.
static void test_raster_wins_over_drawn(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_RASTER | DEVELOP_MASK_MASK);
  g_strlcpy(flexi_bp.raster_mask_source, "colorbalancergb",
            sizeof(flexi_bp.raster_mask_source));
  const int para_before = _count_forms(DT_MASKS_PARAMETRIC);

  assert_true(_migrate());
  _assert_flexi();
  assert_int_equal(_count_forms(DT_MASKS_RASTER), 1);
  assert_int_equal(_count_forms(DT_MASKS_PARAMETRIC), para_before);
}

static void test_raster_wins_over_parametric(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_RASTER | DEVELOP_MASK_CONDITIONAL);
  g_strlcpy(flexi_bp.raster_mask_source, "colorbalancergb",
            sizeof(flexi_bp.raster_mask_source));
  flexi_bp.blendif = _active_channel_bit();
  const int para_before = _count_forms(DT_MASKS_PARAMETRIC);

  assert_true(_migrate());
  _assert_flexi();
  assert_int_equal(_count_forms(DT_MASKS_RASTER), 1);
  // the parametric data is dropped, matching how classic already rendered it
  assert_int_equal(_count_forms(DT_MASKS_PARAMETRIC), para_before);
}

static void test_raster_wins_over_both(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_RASTER | DEVELOP_MASK_MASK
           | DEVELOP_MASK_CONDITIONAL);
  g_strlcpy(flexi_bp.raster_mask_source, "colorbalancergb",
            sizeof(flexi_bp.raster_mask_source));
  assert_true(_migrate());
  _assert_flexi();
  assert_int_equal(_count_forms(DT_MASKS_RASTER), 1);
}

// ---------------------------------------------------------------------------
// case 7: a mode bit with ENABLED clear
// ---------------------------------------------------------------------------

// every GUI mode button writes ENABLED alongside its mode bit, so this cannot
// occur from any code path -- but the renderer already treats it as equivalent,
// so migration normalizes it rather than rejecting the edit
static void test_mode_bit_without_enabled_gets_enabled(void **state)
{
  const uint32_t modes[] = { DEVELOP_MASK_MASK, DEVELOP_MASK_CONDITIONAL,
                             DEVELOP_MASK_RASTER };
  for(size_t i = 0; i < sizeof(modes) / sizeof(*modes); i++)
  {
    _classic(modes[i]); // deliberately no ENABLED
    flexi_bp.mask_mode = modes[i]; // _classic() set it, but be explicit
    flexi_bp.blendif = _active_channel_bit();
    g_strlcpy(flexi_bp.raster_mask_source, "colorbalancergb",
              sizeof(flexi_bp.raster_mask_source));
    assert_true(_migrate());
    _assert_flexi();
    flexi_teardown();
  }
}

// ---------------------------------------------------------------------------
// degenerate parametrics: no channel does anything
// ---------------------------------------------------------------------------

// A blendif whose channels cancel out (or where none is active at all) has no
// mask to build: it collapses to a plain uniform blend with no form, and for
// the "always zero" parity to opacity 0, which reproduces "contributes
// nothing" exactly. No parametric form is synthesized for it.
static void test_degenerate_parametric_collapses_to_uniform(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_CONDITIONAL);
  flexi_bp.blendif = 0; // no channel active
  const int before = _count_forms(DT_MASKS_PARAMETRIC);

  assert_true(_migrate());
  assert_int_equal(_count_forms(DT_MASKS_PARAMETRIC), before);
  assert_int_equal(flexi_bp.mask_id, NO_MASKID);
  // no classic bit survives, whichever uniform parity it landed on
  assert_int_equal(flexi_bp.mask_mode & DEVELOP_MASK_CONDITIONAL, 0);
}

// a RAW colorspace has no canceling-channel mechanism at all, so every
// parametric there is degenerate by construction
static void test_parametric_in_raw_colorspace_is_degenerate(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_CONDITIONAL);
  flexi_bp.blend_cst = DEVELOP_BLEND_CS_RAW;
  const int before = _count_forms(DT_MASKS_PARAMETRIC);

  assert_true(_migrate());
  assert_int_equal(_count_forms(DT_MASKS_PARAMETRIC), before);
  assert_int_equal(flexi_bp.mask_mode & DEVELOP_MASK_CONDITIONAL, 0);
}

// ---------------------------------------------------------------------------
// case 8: already flexi
// ---------------------------------------------------------------------------

// edits created under the POC, before the version bump shipped, are nominally
// still at the old blend version but already carry FLEXI -- they must pass
// through untouched rather than being migrated a second time
static void test_already_flexi_passes_through(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_FLEXI);
  const dt_mask_id_t id_before = flexi_bp.mask_id;
  const int forms_before = g_list_length(flexi_dev.forms);

  assert_true(_migrate());
  assert_int_equal(flexi_bp.mask_mode, DEVELOP_MASK_ENABLED | DEVELOP_MASK_FLEXI);
  assert_int_equal(flexi_bp.mask_id, id_before);
  assert_int_equal((int)g_list_length(flexi_dev.forms), forms_before);
}

// ---------------------------------------------------------------------------
// whole-table sweep and idempotency
// ---------------------------------------------------------------------------

// every one of the sixteen reachable bit combinations must leave mask_mode in
// a state the renderer understands: DISABLED, or ENABLED together with FLEXI.
// Nothing may be left carrying a classic mode bit.
static void test_every_bit_combination_lands_in_a_valid_state(void **state)
{
  for(uint32_t bits = 0; bits < 16; bits++)
  {
    const uint32_t mode =
        ((bits & 1) ? DEVELOP_MASK_ENABLED : 0)
      | ((bits & 2) ? DEVELOP_MASK_MASK : 0)
      | ((bits & 4) ? DEVELOP_MASK_CONDITIONAL : 0)
      | ((bits & 8) ? DEVELOP_MASK_RASTER : 0);

    _classic(mode);
    g_strlcpy(flexi_bp.raster_mask_source, "colorbalancergb",
              sizeof(flexi_bp.raster_mask_source));
    _migrate();

    const uint32_t out = flexi_bp.mask_mode;
    // Valid end states: DISABLED, ENABLED|FLEXI, or -- where a degenerate
    // parametric collapses to a constant -- a plain uniform ENABLED blend with
    // no form (see _migrate_parametric_only's DT_COND_CONSTANT /
    // DT_COND_PASSTHROUGH branches). What must never survive is a classic mode
    // bit: that would be a half-migrated edit.
    const gboolean valid =
      (out == DEVELOP_MASK_DISABLED)
      || (out == DEVELOP_MASK_ENABLED)
      || ((out & DEVELOP_MASK_ENABLED) && (out & DEVELOP_MASK_FLEXI));
    if(!valid)
      fail_msg("mask_mode 0x%x migrated to 0x%x, which is not a valid end state",
               mode, out);
    if(out & (DEVELOP_MASK_MASK | DEVELOP_MASK_CONDITIONAL | DEVELOP_MASK_RASTER))
      fail_msg("mask_mode 0x%x migrated to 0x%x, which still carries a classic "
               "mode bit", mode, out);

    flexi_teardown();
  }
}

// migrating an already-migrated edit is a no-op -- the FLEXI guard at the top
// makes this structural rather than something a runtime flag has to enforce
static void test_migration_is_idempotent(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_CONDITIONAL);
  flexi_bp.blendif = _active_channel_bit();
  assert_true(_migrate());

  const uint32_t mode_once = flexi_bp.mask_mode;
  const int forms_once = g_list_length(flexi_dev.forms);

  assert_true(_migrate());
  assert_int_equal(flexi_bp.mask_mode, mode_once);
  assert_int_equal((int)g_list_length(flexi_dev.forms), forms_once);
}

// a module with no dev cannot synthesize forms; it must decline rather than
// half-migrate
static void test_module_without_dev_does_not_half_migrate(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_CONDITIONAL);
  flexi_module.dev = NULL;
  const uint32_t before = flexi_bp.mask_mode;

  _migrate();
  // either fully migrated or left classic -- never a mix of both
  const uint32_t out = flexi_bp.mask_mode;
  const gboolean clean = (out == before)
                         || ((out & DEVELOP_MASK_FLEXI)
                             && !(out & (DEVELOP_MASK_MASK | DEVELOP_MASK_CONDITIONAL
                                         | DEVELOP_MASK_RASTER)));
  if(!clean)
    fail_msg("mask_mode left half-migrated: 0x%x -> 0x%x", before, out);
  flexi_module.dev = &flexi_dev;
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_teardown(test_disabled_stays_disabled, _teardown),
    cmocka_unit_test_teardown(test_uniform_enabled_becomes_flexi, _teardown),
    cmocka_unit_test_teardown(test_drawn_only_reuses_the_group, _teardown),
    cmocka_unit_test_teardown(test_drawn_with_dangling_mask_id, _teardown),
    cmocka_unit_test_teardown(test_parametric_only_synthesizes_a_parametric_form, _teardown),
    cmocka_unit_test_teardown(test_drawn_and_parametric_stacks_a_parametric_element, _teardown),
    cmocka_unit_test_teardown(test_raster_synthesizes_a_raster_form, _teardown),
    cmocka_unit_test_teardown(test_raster_inversion_moves_onto_the_state_bit, _teardown),
    cmocka_unit_test_teardown(test_raster_wins_over_drawn, _teardown),
    cmocka_unit_test_teardown(test_raster_wins_over_parametric, _teardown),
    cmocka_unit_test_teardown(test_raster_wins_over_both, _teardown),
    cmocka_unit_test_teardown(test_mode_bit_without_enabled_gets_enabled, _teardown),
    cmocka_unit_test_teardown(test_degenerate_parametric_collapses_to_uniform, _teardown),
    cmocka_unit_test_teardown(test_parametric_in_raw_colorspace_is_degenerate, _teardown),
    cmocka_unit_test_teardown(test_already_flexi_passes_through, _teardown),
    cmocka_unit_test_teardown(test_every_bit_combination_lands_in_a_valid_state, _teardown),
    cmocka_unit_test_teardown(test_migration_is_idempotent, _teardown),
    cmocka_unit_test_teardown(test_module_without_dev_does_not_half_migrate, _teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
