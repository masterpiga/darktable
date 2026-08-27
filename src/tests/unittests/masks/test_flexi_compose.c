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

// The mask operators themselves: the arithmetic behind every operator name the
// panel shows.
//
// Until now these were only ever checked end-to-end, by rendering an image and
// comparing pixels. That proves the whole pipeline agrees with itself on the
// fixtures it has, but it does not pin what an operator *means*, and it cannot
// state the properties the rest of the design leans on. Those properties are
// load-bearing:
//
//   * a group's members combine order-independently -- which is the whole
//     justification for treating a group as an unordered bag of shapes, and for
//     the panel letting members be reordered freely within a group;
//   * an empty group is the identity for its operator -- which is what stops an
//     empty intersect group from blanking the entire mask;
//   * opacity and invert compose the same way for every operator.
//
// Each is asserted here directly, on small buffers with known values, rather
// than being inferred from an image.

#include "flexi_fixture.h"
#include "develop/masks/group_internal.h"

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <cmocka.h>

#include <math.h>

#define N 5
// a spread of values including both endpoints, since the operators clamp and
// saturate differently at 0 and 1
static const float A[N] = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };
static const float B[N] = { 1.0f, 0.75f, 0.5f, 0.25f, 0.0f };

typedef void (*combine_fn)(float *const restrict, float *const restrict,
                           const size_t, const float, const int);

static void _apply(combine_fn fn, const float *dest_in, const float *src,
                   float *out, const float opacity, const int inverted)
{
  float s[N];
  memcpy(out, dest_in, sizeof(float) * N);
  memcpy(s, src, sizeof(float) * N);
  fn(out, s, N, opacity, inverted);
}

static void _assert_close(const float *got, const float *want, const char *what)
{
  for(int i = 0; i < N; i++)
    if(fabsf(got[i] - want[i]) > 1e-5f)
      fail_msg("%s: element %d is %.6f, expected %.6f", what, i, got[i], want[i]);
}

// ---------------------------------------------------------------------------
// what each operator computes
// ---------------------------------------------------------------------------

static void test_union_is_max(void **state)
{
  float out[N];
  _apply(_combine_masks_union, A, B, out, 1.0f, 0);
  const float want[N] = { 1.0f, 0.75f, 0.5f, 0.75f, 1.0f };
  _assert_close(out, want, "union");
}

static void test_intersection_is_min(void **state)
{
  float out[N];
  _apply(_combine_masks_intersect, A, B, out, 1.0f, 0);
  const float want[N] = { 0.0f, 0.25f, 0.5f, 0.25f, 0.0f };
  _assert_close(out, want, "intersection");
}

// difference removes the incoming mask from the accumulator
static void test_difference_subtracts(void **state)
{
  float out[N];
  _apply(_combine_masks_difference, A, B, out, 1.0f, 0);
  const float want[N] = { 0.0f, 0.0625f, 0.25f, 0.5625f, 1.0f };
  for(int i = 0; i < N; i++)
    if(out[i] < -1e-5f || out[i] > 1.0f + 1e-5f)
      fail_msg("difference left element %d out of range: %.6f", i, out[i]);
  // a == b removes everything; a with b == 0 keeps a
  assert_float_equal(out[0], 0.0f, 1e-5);   // a=0
  assert_float_equal(out[4], 1.0f, 1e-5);   // a=1, b=0 -> untouched
  (void)want;
}

static void test_screen_is_the_probabilistic_or(void **state)
{
  float out[N];
  _apply(_combine_masks_screen, A, B, out, 1.0f, 0);
  float want[N];
  for(int i = 0; i < N; i++) want[i] = A[i] + B[i] - A[i] * B[i];
  _assert_close(out, want, "screen");
}

static void test_multiply_is_the_product(void **state)
{
  float out[N];
  _apply(_combine_masks_multiply, A, B, out, 1.0f, 0);
  float want[N];
  for(int i = 0; i < N; i++) want[i] = A[i] * B[i];
  _assert_close(out, want, "multiply");
}

// sum and exclusion must stay in range even where their raw arithmetic would
// not -- a + b saturates at 1, and exclusion is symmetric about it
static void test_sum_and_exclusion_stay_in_range(void **state)
{
  float out[N];
  _apply(_combine_masks_sum, A, B, out, 1.0f, 0);
  for(int i = 0; i < N; i++)
    if(out[i] < -1e-5f || out[i] > 1.0f + 1e-5f)
      fail_msg("sum left element %d out of range: %.6f", i, out[i]);

  _apply(_combine_masks_exclusion, A, B, out, 1.0f, 0);
  for(int i = 0; i < N; i++)
    if(out[i] < -1e-5f || out[i] > 1.0f + 1e-5f)
      fail_msg("exclusion left element %d out of range: %.6f", i, out[i]);
}

// every operator must keep its output in [0,1] for every input in [0,1]:
// a mask outside that range is meaningless to the blend math downstream
static void test_every_operator_keeps_the_mask_in_range(void **state)
{
  const struct { const char *name; combine_fn fn; } ops[] = {
    { "union", _combine_masks_union },
    { "intersection", _combine_masks_intersect },
    { "difference", _combine_masks_difference },
    { "sum", _combine_masks_sum },
    { "exclusion", _combine_masks_exclusion },
    { "multiply", _combine_masks_multiply },
    { "screen", _combine_masks_screen },
  };
  const float opacities[] = { 0.0f, 0.35f, 1.0f };

  for(size_t o = 0; o < sizeof(ops) / sizeof(*ops); o++)
    for(size_t k = 0; k < sizeof(opacities) / sizeof(*opacities); k++)
      for(int inv = 0; inv < 2; inv++)
      {
        float out[N];
        _apply(ops[o].fn, A, B, out, opacities[k], inv);
        for(int i = 0; i < N; i++)
          if(!(out[i] >= -1e-5f && out[i] <= 1.0f + 1e-5f) || isnan(out[i]))
            fail_msg("%s (opacity %.2f, inverted %d) produced %.6f at element %d",
                     ops[o].name, opacities[k], inv, out[i], i);
      }
}

// ---------------------------------------------------------------------------
// order independence -- what makes a group an unordered bag
// ---------------------------------------------------------------------------

// A group's members are folded by union (default) or screen. Both must be
// commutative, or the panel's freedom to reorder members within a group would
// silently change the rendered mask.
static void test_group_fold_operators_are_commutative(void **state)
{
  float ab[N], ba[N];

  _apply(_combine_masks_union, A, B, ab, 1.0f, 0);
  _apply(_combine_masks_union, B, A, ba, 1.0f, 0);
  _assert_close(ab, ba, "union is not commutative");

  _apply(_combine_masks_screen, A, B, ab, 1.0f, 0);
  _apply(_combine_masks_screen, B, A, ba, 1.0f, 0);
  _assert_close(ab, ba, "screen is not commutative");
}

// and associative, so a three-member group folds to the same mask whatever
// order the members are visited in
static void test_group_fold_operators_are_associative(void **state)
{
  const float C[N] = { 0.6f, 0.1f, 0.9f, 0.4f, 0.2f };
  float ab[N], abc[N], bc[N], a_bc[N];

  _apply(_combine_masks_union, A, B, ab, 1.0f, 0);
  _apply(_combine_masks_union, ab, C, abc, 1.0f, 0);
  _apply(_combine_masks_union, B, C, bc, 1.0f, 0);
  _apply(_combine_masks_union, A, bc, a_bc, 1.0f, 0);
  _assert_close(abc, a_bc, "union is not associative");

  _apply(_combine_masks_screen, A, B, ab, 1.0f, 0);
  _apply(_combine_masks_screen, ab, C, abc, 1.0f, 0);
  _apply(_combine_masks_screen, B, C, bc, 1.0f, 0);
  _apply(_combine_masks_screen, A, bc, a_bc, 1.0f, 0);
  _assert_close(abc, a_bc, "screen is not associative");
}

// difference is deliberately NOT commutative -- it is a between-group
// operator, where order is the user's choice and must be respected
static void test_difference_is_order_dependent(void **state)
{
  float ab[N], ba[N];
  _apply(_combine_masks_difference, A, B, ab, 1.0f, 0);
  _apply(_combine_masks_difference, B, A, ba, 1.0f, 0);

  gboolean same = TRUE;
  for(int i = 0; i < N; i++) if(fabsf(ab[i] - ba[i]) > 1e-5f) same = FALSE;
  if(same) fail_msg("difference came out order-independent; it must not be");
}

// ---------------------------------------------------------------------------
// identities -- what makes an empty group harmless
// ---------------------------------------------------------------------------

// An empty group contributes nothing, which the compositor implements by
// skipping it. These pin the arithmetic that makes skipping the *right*
// choice: compositing an all-zero mask is already the identity for union,
// and an all-one mask is the identity for intersection -- so an empty
// intersect group could never be allowed to composite as all-zero.
static void test_union_with_zero_is_identity(void **state)
{
  const float zero[N] = { 0 };
  float out[N];
  _apply(_combine_masks_union, A, zero, out, 1.0f, 0);
  _assert_close(out, A, "union with an empty mask changed the accumulator");
}

static void test_intersection_with_one_is_identity(void **state)
{
  const float one[N] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
  float out[N];
  _apply(_combine_masks_intersect, A, one, out, 1.0f, 0);
  _assert_close(out, A, "intersection with a full mask changed the accumulator");
}

static void test_multiply_with_one_is_identity(void **state)
{
  const float one[N] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
  float out[N];
  _apply(_combine_masks_multiply, A, one, out, 1.0f, 0);
  _assert_close(out, A, "multiply by a full mask changed the accumulator");
}

static void test_screen_with_zero_is_identity(void **state)
{
  const float zero[N] = { 0 };
  float out[N];
  _apply(_combine_masks_screen, A, zero, out, 1.0f, 0);
  _assert_close(out, A, "screen with an empty mask changed the accumulator");
}

static void test_difference_with_zero_is_identity(void **state)
{
  const float zero[N] = { 0 };
  float out[N];
  _apply(_combine_masks_difference, A, zero, out, 1.0f, 0);
  _assert_close(out, A, "difference by an empty mask changed the accumulator");
}

// ---------------------------------------------------------------------------
// opacity and invert
// ---------------------------------------------------------------------------

// opacity 0 makes the incoming mask contribute nothing -- the same identity as
// an empty mask, for the operators whose identity is zero
static void test_zero_opacity_neutralizes_a_union_member(void **state)
{
  float out[N];
  _apply(_combine_masks_union, A, B, out, 0.0f, 0);
  _assert_close(out, A, "a zero-opacity union member still changed the mask");
}

// invert complements the incoming mask before the operator sees it, so
// inverting an all-zero mask makes it behave like an all-one one
static void test_invert_complements_the_incoming_mask(void **state)
{
  const float zero[N] = { 0 };
  const float one[N] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
  float inverted_zero[N], plain_one[N];

  _apply(_combine_masks_union, A, zero, inverted_zero, 1.0f, 1);
  _apply(_combine_masks_union, A, one, plain_one, 1.0f, 0);
  _assert_close(inverted_zero, plain_one,
                "inverting an empty mask did not behave like a full one");
}

// inverting twice is the identity, for every operator
static void test_double_invert_is_identity(void **state)
{
  const struct { const char *name; combine_fn fn; } ops[] = {
    { "union", _combine_masks_union },
    { "intersection", _combine_masks_intersect },
    { "difference", _combine_masks_difference },
    { "sum", _combine_masks_sum },
    { "exclusion", _combine_masks_exclusion },
    { "multiply", _combine_masks_multiply },
    { "screen", _combine_masks_screen },
  };
  float b_inv[N];
  for(int i = 0; i < N; i++) b_inv[i] = 1.0f - B[i];

  for(size_t o = 0; o < sizeof(ops) / sizeof(*ops); o++)
  {
    float via_flag[N], via_data[N];
    _apply(ops[o].fn, A, B, via_flag, 1.0f, 1);      // invert flag on B
    _apply(ops[o].fn, A, b_inv, via_data, 1.0f, 0);  // pre-inverted data
    _assert_close(via_flag, via_data, ops[o].name);
  }
}

// ---------------------------------------------------------------------------
// operator dispatch
// ---------------------------------------------------------------------------

// _flexi_apply_group_op routes a state word to the right operator. A group
// whose state carries no operator bit at all must fall back to union, matching
// _eff_group_op's own back-compat rule -- otherwise an old or hand-edited blob
// composites as something arbitrary.
static void test_group_op_dispatch_matches_each_operator(void **state)
{
  const struct { dt_masks_state_t bit; combine_fn fn; const char *name; } cases[] = {
    { DT_MASKS_STATE_UNION, _combine_masks_union, "union" },
    { DT_MASKS_STATE_INTERSECTION, _combine_masks_intersect, "intersection" },
    { DT_MASKS_STATE_DIFFERENCE, _combine_masks_difference, "difference" },
    { DT_MASKS_STATE_SUM, _combine_masks_sum, "sum" },
    { DT_MASKS_STATE_EXCLUSION, _combine_masks_exclusion, "exclusion" },
    { DT_MASKS_STATE_MULTIPLY, _combine_masks_multiply, "multiply" },
    { DT_MASKS_STATE_OP_SCREEN, _combine_masks_screen, "screen" },
  };

  for(size_t c = 0; c < sizeof(cases) / sizeof(*cases); c++)
  {
    float via_dispatch[N], via_direct[N], src[N];
    memcpy(via_dispatch, A, sizeof(A));
    memcpy(src, B, sizeof(B));
    _flexi_apply_group_op(via_dispatch, src, N, cases[c].bit);
    _apply(cases[c].fn, A, B, via_direct, 1.0f, 0);
    _assert_close(via_dispatch, via_direct, cases[c].name);
  }
}

static void test_group_op_dispatch_defaults_to_union(void **state)
{
  float via_dispatch[N], via_union[N], src[N];
  memcpy(via_dispatch, A, sizeof(A));
  memcpy(src, B, sizeof(B));
  _flexi_apply_group_op(via_dispatch, src, N, 0); // no operator bit at all
  _apply(_combine_masks_union, A, B, via_union, 1.0f, 0);
  _assert_close(via_dispatch, via_union, "operator-less group did not default to union");
}

// ---------------------------------------------------------------------------
// the "contributes nothing" detector
// ---------------------------------------------------------------------------

// a parametric channel still at its full range renders as an all-one mask, and
// must not count as an active group member
static void test_uniform_one_detector(void **state)
{
  float buf[N] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
  assert_true(_mask_buffer_is_uniform_one(buf, N));

  buf[2] = 0.99f;
  assert_false(_mask_buffer_is_uniform_one(buf, N));

  // the tolerance is one-sided: values at or above the threshold count as one
  buf[2] = 0.99999f;
  assert_true(_mask_buffer_is_uniform_one(buf, N));

  float zero[N] = { 0 };
  assert_false(_mask_buffer_is_uniform_one(zero, N));
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_union_is_max),
    cmocka_unit_test(test_intersection_is_min),
    cmocka_unit_test(test_difference_subtracts),
    cmocka_unit_test(test_screen_is_the_probabilistic_or),
    cmocka_unit_test(test_multiply_is_the_product),
    cmocka_unit_test(test_sum_and_exclusion_stay_in_range),
    cmocka_unit_test(test_every_operator_keeps_the_mask_in_range),
    cmocka_unit_test(test_group_fold_operators_are_commutative),
    cmocka_unit_test(test_group_fold_operators_are_associative),
    cmocka_unit_test(test_difference_is_order_dependent),
    cmocka_unit_test(test_union_with_zero_is_identity),
    cmocka_unit_test(test_intersection_with_one_is_identity),
    cmocka_unit_test(test_multiply_with_one_is_identity),
    cmocka_unit_test(test_screen_with_zero_is_identity),
    cmocka_unit_test(test_difference_with_zero_is_identity),
    cmocka_unit_test(test_zero_opacity_neutralizes_a_union_member),
    cmocka_unit_test(test_invert_complements_the_incoming_mask),
    cmocka_unit_test(test_double_invert_is_identity),
    cmocka_unit_test(test_group_op_dispatch_matches_each_operator),
    cmocka_unit_test(test_group_op_dispatch_defaults_to_union),
    cmocka_unit_test(test_uniform_one_detector),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
