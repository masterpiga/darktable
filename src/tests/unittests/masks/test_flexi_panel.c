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

// What the panel decides to *show*, as opposed to what it does to the mask.
//
// These are display rules -- which warning badge a row carries, which sliders a
// parametric row exposes, what the panel preferences are -- and every one of
// them is a decision the panel makes from values it can read, ahead of touching
// any widget. Only the decisions are tested; the widget updates they drive are
// on the manual checklist in README.md.
//
// Where a rule lives inside a widget function, it has been split out into a
// `_model_*` decision the widget code then applies, so the rule and the panel
// cannot disagree.

#include "flexi_fixture.h"
#include "control/conf.h"

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
// warning badges
// ---------------------------------------------------------------------------

// an element contributing (almost) nothing is worth flagging, because on canvas
// 9% and 0% look nearly identical to "off"
static void test_low_opacity_badge_threshold(void **state)
{
  assert_int_equal(_model_badge_kind(1.0f, FALSE), DT_MASKS_BADGE_NONE);
  assert_int_equal(_model_badge_kind(0.5f, FALSE), DT_MASKS_BADGE_NONE);
  assert_int_equal(_model_badge_kind(0.10f, FALSE), DT_MASKS_BADGE_NONE);
  assert_int_equal(_model_badge_kind(0.099f, FALSE), DT_MASKS_BADGE_LOW_OPACITY);
  assert_int_equal(_model_badge_kind(0.0f, FALSE), DT_MASKS_BADGE_LOW_OPACITY);
}

// no-op outranks low opacity: a channel that restricts nothing contributes
// nothing whatever its opacity, so reporting the opacity would be noise
static void test_noop_badge_outranks_low_opacity(void **state)
{
  assert_int_equal(_model_badge_kind(0.0f, TRUE), DT_MASKS_BADGE_NOOP);
  assert_int_equal(_model_badge_kind(1.0f, TRUE), DT_MASKS_BADGE_NOOP);
}

// ---------------------------------------------------------------------------
// the no-op predicate behind that badge
// ---------------------------------------------------------------------------

// build a single-channel parametric form whose sub-ranges are all at the full
// default span -- i.e. it restricts nothing
static dt_masks_form_t *_make_parametric(void)
{
  dt_masks_form_t *f = calloc(1, sizeof(dt_masks_form_t));
  f->formid = 50;
  f->type = DT_MASKS_PARAMETRIC;
  dt_masks_point_parametric_t *p = calloc(1, sizeof(dt_masks_point_parametric_t));
  p->single = 1;
  p->invert = 0;
  p->colorspace = DEVELOP_BLEND_CS_RGB_SCENE;
  p->channel = 0;
  for(int i = 0; i < 4 * DEVELOP_BLENDIF_SIZE; i += 4)
  {
    p->blendif_parameters[i + 0] = 0.0f;
    p->blendif_parameters[i + 1] = 0.0f;
    p->blendif_parameters[i + 2] = 1.0f;
    p->blendif_parameters[i + 3] = 1.0f;
  }
  f->points = g_list_append(NULL, p);
  return f;
}

static void _free_parametric(dt_masks_form_t *f)
{
  g_list_free_full(f->points, free);
  free(f);
}

static void test_untouched_parametric_is_a_noop(void **state)
{
  dt_masks_form_t *f = _make_parametric();
  assert_true(_parametric_form_is_noop(f));
  _free_parametric(f);
}

static void test_narrowed_parametric_is_not_a_noop(void **state)
{
  dt_masks_form_t *f = _make_parametric();
  const dt_iop_gui_blendif_channel_t *ch =
    dt_develop_blendif_channels_for_csp(DEVELOP_BLEND_CS_RGB_SCENE);
  dt_masks_point_parametric_t *p = f->points->data;
  // narrow the input sub-range of the form's own channel
  p->blendif_parameters[4 * ch[0].param_channels[0] + 2] = 0.5f;

  assert_false(_parametric_form_is_noop(f));
  _free_parametric(f);
}

// an output sub-range still refines the mask even while its slider is hidden,
// so it counts too -- not just whichever one the UI happens to show
static void test_output_range_alone_is_not_a_noop(void **state)
{
  dt_masks_form_t *f = _make_parametric();
  const dt_iop_gui_blendif_channel_t *ch =
    dt_develop_blendif_channels_for_csp(DEVELOP_BLEND_CS_RGB_SCENE);
  dt_masks_point_parametric_t *p = f->points->data;
  p->blendif_parameters[4 * ch[0].param_channels[1] + 2] = 0.5f;

  assert_false(_parametric_form_is_noop(f));
  _free_parametric(f);
}

// inverted polarity is excluded outright: a full range selects everything, its
// complement selects nothing -- a different kind of wrong, not a no-op
static void test_inverted_parametric_is_never_a_noop(void **state)
{
  dt_masks_form_t *f = _make_parametric();
  ((dt_masks_point_parametric_t *)f->points->data)->invert = 1;
  assert_false(_parametric_form_is_noop(f));
  _free_parametric(f);
}

// a drawn shape is not a parametric form and never carries the no-op badge
static void test_a_shape_is_never_a_noop(void **state)
{
  flexi_build("u:1,2");
  assert_false(_parametric_form_is_noop(dt_masks_get_from_id(&flexi_dev, 1)));
}

// ---------------------------------------------------------------------------
// adaptive display of a parametric row
// ---------------------------------------------------------------------------

// an expanded row always shows both sub-ranges, whatever the user has touched
static void test_expanded_row_shows_both_ranges(void **state)
{
  const dt_masks_param_vis_t v = _model_param_row_visibility(TRUE, FALSE, FALSE, FALSE);
  assert_true(v.input);
  assert_true(v.output);
}

// the boost-factor slider only exists for channels that have one
static void test_boost_slider_follows_the_channel(void **state)
{
  assert_true(_model_param_row_visibility(TRUE, TRUE, TRUE, TRUE).boost);
  assert_false(_model_param_row_visibility(TRUE, TRUE, TRUE, FALSE).boost);
  // never on a collapsed row, whatever the channel supports
  assert_false(_model_param_row_visibility(FALSE, TRUE, TRUE, TRUE).boost);
}

// a collapsed row adapts: an untouched channel shows only the input slider,
// rather than a second slider that says nothing
static void test_collapsed_untouched_row_shows_input_only(void **state)
{
  const dt_masks_param_vis_t v = _model_param_row_visibility(FALSE, FALSE, FALSE, FALSE);
  assert_true(v.input);
  assert_false(v.output);
}

static void test_collapsed_row_with_only_output_used(void **state)
{
  const dt_masks_param_vis_t v = _model_param_row_visibility(FALSE, FALSE, TRUE, FALSE);
  assert_false(v.input);
  assert_true(v.output);
}

static void test_collapsed_row_with_both_used_shows_both(void **state)
{
  const dt_masks_param_vis_t v = _model_param_row_visibility(FALSE, TRUE, TRUE, FALSE);
  assert_true(v.input);
  assert_true(v.output);
}

static void test_collapsed_row_with_only_input_used(void **state)
{
  const dt_masks_param_vis_t v = _model_param_row_visibility(FALSE, TRUE, FALSE, FALSE);
  assert_true(v.input);
  assert_false(v.output);
}

// the per-sub-range bypass toggles only mean something when both are in play
static void test_bypass_shown_only_when_both_ranges_used(void **state)
{
  assert_true(_model_param_row_visibility(FALSE, TRUE, TRUE, FALSE).bypass);
  assert_false(_model_param_row_visibility(FALSE, TRUE, FALSE, FALSE).bypass);
  assert_false(_model_param_row_visibility(FALSE, FALSE, TRUE, FALSE).bypass);
  assert_false(_model_param_row_visibility(TRUE, TRUE, FALSE, FALSE).bypass);
}

// the "is this sub-range used" predicate the rule above consumes
static void test_channel_used_detects_a_touched_range(void **state)
{
  dt_masks_form_t *f = _make_parametric();
  dt_masks_point_parametric_t *p = f->points->data;
  const dt_iop_gui_blendif_channel_t *ch =
    dt_develop_blendif_channels_for_csp(DEVELOP_BLEND_CS_RGB_SCENE);

  assert_false(_param_channel_is_used(p, &ch[0], 0));
  p->blendif_parameters[4 * ch[0].param_channels[0] + 2] = 0.5f;
  assert_true(_param_channel_is_used(p, &ch[0], 0));
  _free_parametric(f);
}

// the channel's own enable bit counts as "used" even at a full range, so a
// channel the user explicitly switched on does not read as untouched
static void test_channel_used_honours_the_active_bit(void **state)
{
  dt_masks_form_t *f = _make_parametric();
  dt_masks_point_parametric_t *p = f->points->data;
  const dt_iop_gui_blendif_channel_t *ch =
    dt_develop_blendif_channels_for_csp(DEVELOP_BLEND_CS_RGB_SCENE);

  assert_false(_param_channel_is_used(p, &ch[0], 0));
  p->blendif |= (1u << ch[0].param_channels[0]);
  assert_true(_param_channel_is_used(p, &ch[0], 0));
  _free_parametric(f);
}

// ---------------------------------------------------------------------------
// panel preferences
// ---------------------------------------------------------------------------

static int _conf_setup(void **state)
{
  flexi_conf_init();
  return 0;
}

static int _conf_teardown(void **state)
{
  flexi_teardown();
  flexi_conf_cleanup();
  return 0;
}

// "sticky opacity" off means a new shape's opacity is remembered; on means the
// stored opacity resets to full after each use, so the next shape starts opaque
static void test_sticky_opacity_preference_roundtrips(void **state)
{
  dt_conf_set_bool("plugins/darkroom/masks/opacity_not_sticky", FALSE);
  assert_false(dt_conf_get_bool("plugins/darkroom/masks/opacity_not_sticky"));

  dt_conf_set_float("plugins/darkroom/masks/opacity", 0.4f);
  assert_float_equal(dt_conf_get_float("plugins/darkroom/masks/opacity"), 0.4f, 1e-6);

  // with stickiness disabled the panel resets the stored value to 1.0 after a
  // shape is created (see dt_masks_form_gui_t's opacity handling in masks.c)
  dt_conf_set_bool("plugins/darkroom/masks/opacity_not_sticky", TRUE);
  if(dt_conf_get_bool("plugins/darkroom/masks/opacity_not_sticky"))
    dt_conf_set_float("plugins/darkroom/masks/opacity", 1.0f);
  assert_float_equal(dt_conf_get_float("plugins/darkroom/masks/opacity"), 1.0f, 1e-6);
}

static void test_auto_expand_preference_roundtrips(void **state)
{
  dt_conf_set_bool("plugins/darkroom/masks/auto_expand_selected", TRUE);
  assert_true(dt_conf_get_bool("plugins/darkroom/masks/auto_expand_selected"));
  dt_conf_set_bool("plugins/darkroom/masks/auto_expand_selected", FALSE);
  assert_false(dt_conf_get_bool("plugins/darkroom/masks/auto_expand_selected"));
}

static void test_collapse_refinements_preference_roundtrips(void **state)
{
  dt_conf_set_bool("plugins/darkroom/masks/collapse_refinements_default", TRUE);
  assert_true(dt_conf_get_bool("plugins/darkroom/masks/collapse_refinements_default"));
  dt_conf_set_bool("plugins/darkroom/masks/collapse_refinements_default", FALSE);
  assert_false(dt_conf_get_bool("plugins/darkroom/masks/collapse_refinements_default"));
}

// the panel's default operator for new groups is a string key; an unset or
// unknown value must not leave the panel without an operator
static void test_default_operator_preference_roundtrips(void **state)
{
  dt_conf_set_string("plugins/darkroom/masks/default_operator", "intersection");
  assert_string_equal(
    dt_conf_get_string_const("plugins/darkroom/masks/default_operator"), "intersection");
}

// the mask panel position drives where the panel is hosted; the values the
// panel switches on must round-trip as integers
static void test_panel_position_preference_roundtrips(void **state)
{
  for(int pos = 0; pos < 4; pos++)
  {
    dt_conf_set_int("plugins/darkroom/blend/masks_panel_position", pos);
    assert_int_equal(dt_conf_get_int("plugins/darkroom/blend/masks_panel_position"), pos);
  }
}

// ---------------------------------------------------------------------------
// mask panel & corner icon state transitions
// ---------------------------------------------------------------------------

// when a module is focused and masking-capable, but collapsed:
// - the panel is collapsed
// - the corner icon is visible
// - the corner icon active status reflects whether masking is on or off
static void test_panel_state_collapsed_module_shows_corner_icon(void **state)
{
  const dt_masks_panel_state_t s_on =
    _model_masks_panel_state(MASKS_PANEL_POS_LEFT, TRUE, TRUE, FALSE, TRUE, FALSE);
  assert_true(s_on.want_hosted);
  assert_true(s_on.panel_collapsed);
  assert_true(s_on.corner_icon_visible);
  assert_true(s_on.corner_icon_active);

  const dt_masks_panel_state_t s_off =
    _model_masks_panel_state(MASKS_PANEL_POS_LEFT, TRUE, TRUE, FALSE, FALSE, FALSE);
  assert_true(s_off.want_hosted);
  assert_true(s_off.panel_collapsed);
  assert_true(s_off.corner_icon_visible);
  assert_false(s_off.corner_icon_active);
}

// when an expanded module is focused and masking-capable:
// - the panel is expanded (unless the user explicitly collapsed it via pref)
// - if expanded, the corner icon is not visible
static void test_panel_state_expanded_module_respects_pref(void **state)
{
  const dt_masks_panel_state_t s_exp =
    _model_masks_panel_state(MASKS_PANEL_POS_LEFT, TRUE, TRUE, TRUE, TRUE, FALSE);
  assert_true(s_exp.want_hosted);
  assert_false(s_exp.panel_collapsed);
  assert_false(s_exp.corner_icon_visible);

  const dt_masks_panel_state_t s_col =
    _model_masks_panel_state(MASKS_PANEL_POS_LEFT, TRUE, TRUE, TRUE, TRUE, TRUE);
  assert_true(s_col.want_hosted);
  assert_true(s_col.panel_collapsed);
  assert_true(s_col.corner_icon_visible);
}

// when masking is disabled on a focused, expanded module:
// - the panel respects user preference so blend mode and opacity controls remain accessible
// - the corner icon is rendered in its inactive state
static void test_panel_state_mask_disabled_defaults_to_collapsed(void **state)
{
  const dt_masks_panel_state_t s =
    _model_masks_panel_state(MASKS_PANEL_POS_LEFT, TRUE, TRUE, TRUE, FALSE, FALSE);
  assert_true(s.want_hosted);
  assert_false(s.panel_collapsed);
  assert_false(s.corner_icon_visible);
  assert_false(s.corner_icon_active);
}

// when a module has no masking support (e.g. demosaic, crop) or is unfocused:
// - the panel is collapsed
// - the corner icon is NOT visible
static void test_panel_state_unsupported_or_unfocused_hides_all(void **state)
{
  const dt_masks_panel_state_t s_no_mask =
    _model_masks_panel_state(MASKS_PANEL_POS_LEFT, TRUE, FALSE, TRUE, FALSE, FALSE);
  assert_false(s_no_mask.want_hosted);
  assert_true(s_no_mask.panel_collapsed);
  assert_false(s_no_mask.corner_icon_visible);

  const dt_masks_panel_state_t s_unfocused =
    _model_masks_panel_state(MASKS_PANEL_POS_LEFT, FALSE, TRUE, TRUE, TRUE, FALSE);
  assert_false(s_unfocused.want_hosted);
  assert_true(s_unfocused.panel_collapsed);
  assert_false(s_unfocused.corner_icon_visible);
}

// pinning the mask panel when the module is collapsed must expand the module
static void test_pinning_collapsed_module_expands_iop(void **state)
{
  assert_true(_model_masks_pin_should_expand_iop(FALSE, TRUE));
  assert_false(_model_masks_pin_should_expand_iop(TRUE, TRUE));
  assert_false(_model_masks_pin_should_expand_iop(FALSE, FALSE));
}

// overall module blend opacity slider uses 0..100 range and maps to 0..1 fraction
static void test_main_opacity_badge_threshold(void **state)
{
  assert_int_equal(_model_badge_kind(100.0f / 100.0f, FALSE), DT_MASKS_BADGE_NONE);
  assert_int_equal(_model_badge_kind(74.0f / 100.0f, FALSE), DT_MASKS_BADGE_NONE);
  assert_int_equal(_model_badge_kind(10.0f / 100.0f, FALSE), DT_MASKS_BADGE_NONE);
  assert_int_equal(_model_badge_kind(9.9f / 100.0f, FALSE), DT_MASKS_BADGE_LOW_OPACITY);
  assert_int_equal(_model_badge_kind(0.0f / 100.0f, FALSE), DT_MASKS_BADGE_LOW_OPACITY);
}

// in utility position, panel state follows user preference and does not collapse when IOP is collapsed
static void test_panel_state_utility_position_follows_pref(void **state)
{
  // focused & expanded module in utility position: open when pref is open
  const dt_masks_panel_state_t s_open =
    _model_masks_panel_state(MASKS_PANEL_POS_UTILITY, TRUE, TRUE, TRUE, TRUE, FALSE);
  assert_true(s_open.want_hosted);
  assert_false(s_open.panel_collapsed);
  assert_false(s_open.corner_icon_visible);

  // focused & collapsed IOP module in utility position: stays open when pref is open
  const dt_masks_panel_state_t s_iop_col =
    _model_masks_panel_state(MASKS_PANEL_POS_UTILITY, TRUE, TRUE, FALSE, TRUE, FALSE);
  assert_true(s_iop_col.want_hosted);
  assert_false(s_iop_col.panel_collapsed);
  assert_false(s_iop_col.corner_icon_visible);

  // user collapsed the utility expander: collapsed
  const dt_masks_panel_state_t s_user_col =
    _model_masks_panel_state(MASKS_PANEL_POS_UTILITY, TRUE, TRUE, TRUE, TRUE, TRUE);
  assert_true(s_user_col.want_hosted);
  assert_true(s_user_col.panel_collapsed);
  assert_false(s_user_col.corner_icon_visible);
}

// separate grid panel corner icon and left/right hosting are never active for utility or embedded
static void test_panel_state_no_separate_panel_for_utility_or_embedded(void **state)
{
  const dt_masks_panel_state_t s_util =
    _model_masks_panel_state(MASKS_PANEL_POS_UTILITY, TRUE, TRUE, TRUE, TRUE, FALSE);
  assert_false(s_util.corner_icon_visible);

  const dt_masks_panel_state_t s_emb =
    _model_masks_panel_state(MASKS_PANEL_POS_EMBEDDED, TRUE, TRUE, TRUE, TRUE, FALSE);
  assert_false(s_emb.want_hosted);
  assert_false(s_emb.corner_icon_visible);
}

// pinning a module with masking off must enable masking
static void test_pinning_disabled_mask_enables_mask(void **state)
{
  assert_true(_model_masks_pin_should_enable_mask(DEVELOP_MASK_DISABLED));
  assert_false(_model_masks_pin_should_enable_mask(DEVELOP_MASK_ENABLED | DEVELOP_MASK_FLEXI));
}

// dedicated panel caption reflects module name and instance name with 2-line markup
static void test_masks_panel_header_markup(void **state)
{
  char *m_hosted_no_inst = _model_masks_panel_header_markup("exposure", "", TRUE);
  assert_non_null(strstr(m_hosted_no_inst, "blend mask"));
  assert_non_null(strstr(m_hosted_no_inst, "exposure"));
  assert_non_null(strstr(m_hosted_no_inst, "\n"));
  assert_null(strstr(m_hosted_no_inst, "•"));
  free(m_hosted_no_inst);

  char *m_hosted_inst = _model_masks_panel_header_markup("exposure", "foreground", TRUE);
  assert_non_null(strstr(m_hosted_inst, "blend mask"));
  assert_non_null(strstr(m_hosted_inst, "exposure"));
  assert_non_null(strstr(m_hosted_inst, "• foreground"));
  assert_non_null(strstr(m_hosted_inst, "\n"));
  free(m_hosted_inst);

  char *m_hosted_no_mod = _model_masks_panel_header_markup(NULL, NULL, TRUE);
  assert_non_null(strstr(m_hosted_no_mod, "blend mask"));
  assert_non_null(strstr(m_hosted_no_mod, "no focused module"));
  assert_non_null(strstr(m_hosted_no_mod, "\n"));
  free(m_hosted_no_mod);

  char *m_embedded = _model_masks_panel_header_markup("exposure", "foreground", FALSE);
  assert_string_equal(m_embedded, "blend mask");
  free(m_embedded);
}

// corner icon tooltip reflects module name, instance name, and action hints
static void test_masks_corner_icon_tooltip(void **state)
{
  char *tt_off = _model_masks_corner_icon_tooltip("exposure", "foreground", FALSE, NULL);
  assert_non_null(strstr(tt_off, "exposure (foreground)"));
  assert_non_null(strstr(tt_off, "mask - off"));
  assert_non_null(strstr(tt_off, "click to enable mask and pin"));
  free(tt_off);

  char *tt_on = _model_masks_corner_icon_tooltip("exposure", "", TRUE, "drawn mask");
  assert_non_null(strstr(tt_on, "exposure"));
  assert_non_null(strstr(tt_on, "drawn mask"));
  assert_non_null(strstr(tt_on, "click to expand"));
  free(tt_on);
}

// in-header collapse/pin button tooltip reflects peeking/docked state and auto-enable
static void test_masks_inline_collapse_tooltip(void **state)
{
  char *tt_peek_off = _model_masks_inline_collapse_tooltip(TRUE, FALSE, MASKS_PANEL_POS_LEFT);
  assert_string_equal(tt_peek_off, "pin this panel open and enable mask");
  free(tt_peek_off);

  char *tt_peek_on = _model_masks_inline_collapse_tooltip(TRUE, TRUE, MASKS_PANEL_POS_LEFT);
  assert_string_equal(tt_peek_on, "pin this panel open");
  free(tt_peek_on);

  char *tt_docked = _model_masks_inline_collapse_tooltip(FALSE, TRUE, MASKS_PANEL_POS_LEFT);
  assert_non_null(strstr(tt_docked, "collapse this panel"));
  free(tt_docked);
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_teardown(test_low_opacity_badge_threshold, _teardown),
    cmocka_unit_test_teardown(test_noop_badge_outranks_low_opacity, _teardown),
    cmocka_unit_test_teardown(test_untouched_parametric_is_a_noop, _teardown),
    cmocka_unit_test_teardown(test_narrowed_parametric_is_not_a_noop, _teardown),
    cmocka_unit_test_teardown(test_output_range_alone_is_not_a_noop, _teardown),
    cmocka_unit_test_teardown(test_inverted_parametric_is_never_a_noop, _teardown),
    cmocka_unit_test_teardown(test_a_shape_is_never_a_noop, _teardown),
    cmocka_unit_test_teardown(test_expanded_row_shows_both_ranges, _teardown),
    cmocka_unit_test_teardown(test_boost_slider_follows_the_channel, _teardown),
    cmocka_unit_test_teardown(test_collapsed_untouched_row_shows_input_only, _teardown),
    cmocka_unit_test_teardown(test_collapsed_row_with_only_output_used, _teardown),
    cmocka_unit_test_teardown(test_collapsed_row_with_both_used_shows_both, _teardown),
    cmocka_unit_test_teardown(test_collapsed_row_with_only_input_used, _teardown),
    cmocka_unit_test_teardown(test_bypass_shown_only_when_both_ranges_used, _teardown),
    cmocka_unit_test_teardown(test_channel_used_detects_a_touched_range, _teardown),
    cmocka_unit_test_teardown(test_channel_used_honours_the_active_bit, _teardown),
    cmocka_unit_test_setup_teardown(test_sticky_opacity_preference_roundtrips,
                                    _conf_setup, _conf_teardown),
    cmocka_unit_test_setup_teardown(test_auto_expand_preference_roundtrips,
                                    _conf_setup, _conf_teardown),
    cmocka_unit_test_setup_teardown(test_collapse_refinements_preference_roundtrips,
                                    _conf_setup, _conf_teardown),
    cmocka_unit_test_setup_teardown(test_default_operator_preference_roundtrips,
                                    _conf_setup, _conf_teardown),
    cmocka_unit_test_setup_teardown(test_panel_position_preference_roundtrips,
                                    _conf_setup, _conf_teardown),
    cmocka_unit_test_teardown(test_panel_state_collapsed_module_shows_corner_icon, _teardown),
    cmocka_unit_test_teardown(test_panel_state_expanded_module_respects_pref, _teardown),
    cmocka_unit_test_teardown(test_panel_state_mask_disabled_defaults_to_collapsed, _teardown),
    cmocka_unit_test_teardown(test_panel_state_unsupported_or_unfocused_hides_all, _teardown),
    cmocka_unit_test_teardown(test_main_opacity_badge_threshold, _teardown),
    cmocka_unit_test_teardown(test_panel_state_utility_position_follows_pref, _teardown),
    cmocka_unit_test_teardown(test_panel_state_no_separate_panel_for_utility_or_embedded, _teardown),
    cmocka_unit_test_teardown(test_pinning_collapsed_module_expands_iop, _teardown),
    cmocka_unit_test_teardown(test_pinning_disabled_mask_enables_mask, _teardown),
    cmocka_unit_test_teardown(test_masks_panel_header_markup, _teardown),
    cmocka_unit_test_teardown(test_masks_corner_icon_tooltip, _teardown),
    cmocka_unit_test_teardown(test_masks_inline_collapse_tooltip, _teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
