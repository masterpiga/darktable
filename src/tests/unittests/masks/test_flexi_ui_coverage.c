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

// Is every mask state the panel can reach also a state the harvest checks
// sweep?
//
// This deliberately does NOT test the GUI. Whether a widget is wired to the
// right callback, whether a click lands, whether a toggle repaints -- none of
// that is here, and none of it is what the mask checks are for. The question
// is narrower and answerable: --postedit-masks and --persist-masks sweep a
// fixed vocabulary of control changes (postedit_internal.h), and that
// vocabulary is a hand-written list. If the panel can put a member into a
// state no poke produces, then every one of those checks is silent about it --
// not because the mask is fine, but because nothing ever looked.
//
// So the property is a containment, not a behaviour:
//
//     { member states reachable from the panel } is a subset of
//     { member states some poke produces } union { render-irrelevant }
//
// and the exemption half is checkable rather than asserted: a bit is
// render-irrelevant when neither the fold (masks/group.c) nor the blend
// (blend.c) reads it, which is a grep anyone can repeat, recorded per entry
// below.
//
// TWO AXES, ONLY ONE OF THEM HERE
//
// The panel changes a mask in two different ways, and they need different
// coverage arguments:
//
//   - per-member fields (an operator, an opacity, a refinement, a break).
//     That is what the poke vocabulary is, and what this file pins.
//   - the member LIST (delete a shape, reorder rows). A run is a maximal
//     stretch of the list, so both move boundaries -- but neither is
//     expressible as a field change, so no poke can produce them. Those are
//     covered instead by --persist-masks' structural sequences (STEP_REMOVE
//     and STEP_MOVE_UP in persist.c), which is where they belong: what they
//     break is a boundary surviving a save, not a field.
//
// So this file is not the whole containment argument, only the field half of
// it. The list half has no static tripwire, because adding a list operation
// does not change any struct this could pin.
//
// The tripwire that makes this survive contact with future work is
// _the_struct_has_not_grown(): dt_masks_point_group_t's size is pinned, so
// adding a field to it fails this test until someone classifies the field as
// covered or exempt. That is blunt on purpose -- the failure mode being
// guarded against is a new panel control whose state nothing sweeps, and the
// only reliable moment to catch it is when the field it writes appears.

#include "flexi_fixture.h"

#include "develop/masks/postedit_internal.h"

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <cmocka.h>

// ---------------------------------------------------------------------------
// what the panel can write
// ---------------------------------------------------------------------------

/* Every per-member state bit a panel control can set or clear, with where it
   does it. Kept as source references rather than prose so the claim can be
   re-checked: `grep -n` the line, confirm it still writes that bit.

   `covered` says whether a poke has to produce it. The exempt ones are exempt
   because the renderer never reads them, which is the only reason that holds
   up -- a bit the fold ignores cannot make a mask wrong however it is set. */
static const struct
{
  int bit;
  gboolean covered;
  const char *where;
} _panel_bits[] =
{
  // the operator menu writes the whole DT_MASKS_STATE_OP field at once
  // (blend_gui.c _group_op_apply, and the drag-and-drop paths that copy an
  // operator between rows)
  { DT_MASKS_STATE_UNION,           TRUE,  "blend_gui.c:_group_op_apply" },
  { DT_MASKS_STATE_INTERSECTION,    TRUE,  "blend_gui.c:_group_op_apply" },
  { DT_MASKS_STATE_DIFFERENCE,      TRUE,  "blend_gui.c:_group_op_apply" },
  { DT_MASKS_STATE_EXCLUSION,       TRUE,  "blend_gui.c:_group_op_apply" },
  { DT_MASKS_STATE_SUM,             TRUE,  "blend_gui.c:_group_op_apply" },
  { DT_MASKS_STATE_MULTIPLY,        TRUE,  "blend_gui.c:_group_op_apply" },
  { DT_MASKS_STATE_OP_SCREEN,       TRUE,  "blend_gui.c:_group_op_apply" },

  // the within-group combine mode
  { DT_MASKS_STATE_SCREEN,          TRUE,  "blend_gui.c within-mode callback" },
  { DT_MASKS_STATE_ISECT,           TRUE,  "blend_gui.c within-mode callback" },
  { DT_MASKS_STATE_WITHIN_MULTIPLY, TRUE,  "blend_gui.c within-mode callback" },

  // broadcast across a run by dt_masks_group_set_state()
  { DT_MASKS_STATE_OP_DISABLE,      TRUE,  "blend_gui.c:9990 (bypass)" },
  { DT_MASKS_STATE_OP_INVERT,       TRUE,  "blend_gui.c:10127 (invert output)" },

  // per element
  { DT_MASKS_STATE_DISABLE,         TRUE,  "blend_gui.c:6570 (enable/disable)" },
  { DT_MASKS_STATE_HIDDEN,          TRUE,  "blend_gui.c:4988 (hide)" },
  { DT_MASKS_STATE_INVERSE,         TRUE,  "blend_gui.c:10102 (invert shape)" },

  /* Exempt: neither masks/group.c nor blend.c reads these, so no setting of
     them can change a rendered mask.

     SHOW is canvas visibility -- whether the shape is drawn on screen while
     editing. The fold's skip test is
     `state & (DT_MASKS_STATE_HIDDEN | DT_MASKS_STATE_DISABLE)` (group.c:1414,
     and the flexi fold at group.c:1229/1242); SHOW appears in neither.

     USE is set once when a member is created (blend_gui.c:7756) and never
     cleared by any control.

     GROUP_BREAK is the historical bit that the first-class `group_start`
     field replaced (see its comment in masks.h); nothing reads it any more. */
  { DT_MASKS_STATE_SHOW,            FALSE, "blend_gui.c:5028, canvas only" },
  { DT_MASKS_STATE_USE,             FALSE, "blend_gui.c:7756, set at creation" },
  { DT_MASKS_STATE_GROUP_BREAK,     FALSE, "superseded by group_start" },
};

#define PANEL_BITS_N ((int)(sizeof(_panel_bits) / sizeof(_panel_bits[0])))

// ---------------------------------------------------------------------------
// what the pokes actually write
// ---------------------------------------------------------------------------

/** the union of every state bit some poke sets, and which non-state fields
    some poke changes */
static void _poke_coverage(int *bits,
                           gboolean *opacity,
                           gboolean *group_opacity,
                           gboolean *refinement,
                           gboolean *group_start)
{
  *bits = 0;
  *opacity = *group_opacity = *refinement = *group_start = FALSE;

  for(poke_t k = 0; k < POKE_N; k++)
  {
    // a pristine member with every operator bit already set, so a poke that
    // *replaces* a field (the operator menu, the within mode) shows up as a
    // change in the bits it clears as well as the ones it sets
    dt_masks_point_group_t pt = { 0 };
    pt.state = DT_MASKS_STATE_USE | DT_MASKS_STATE_SHOW;
    pt.opacity = 1.0f;
    pt.group_opacity = 1.0f;
    const dt_masks_point_group_t before = pt;

    GList *one = g_list_append(NULL, &pt);
    _apply_poke(one, k, 0, 0);
    g_list_free(one);

    *bits |= (pt.state ^ before.state);
    if(pt.opacity != before.opacity) *opacity = TRUE;
    if(pt.group_opacity != before.group_opacity) *group_opacity = TRUE;
    if(pt.refinement.enabled != before.refinement.enabled) *refinement = TRUE;
    if(pt.group_start != before.group_start) *group_start = TRUE;
  }
}

/* Every state bit the panel can write is either produced by some poke or
   declared render-irrelevant. A new control that sets a bit neither of those
   covers fails here, which is the point: the harvest checks would otherwise
   pass on it in silence. */
static void test_every_panel_state_bit_is_swept_or_exempt(void **state)
{
  int covered = 0;
  gboolean op, gop, ref, gst;
  _poke_coverage(&covered, &op, &gop, &ref, &gst);

  for(int i = 0; i < PANEL_BITS_N; i++)
  {
    if(!_panel_bits[i].covered) continue;
    if(!(covered & _panel_bits[i].bit))
      fail_msg("panel can set state bit 0x%x (%s) but no poke produces it",
               _panel_bits[i].bit, _panel_bits[i].where);
  }
}

/* ... and the exempt ones really are absent from the vocabulary. If a poke
   starts producing one, either the bit became render-relevant (and the
   exemption above is now a lie) or the poke is writing something it should
   not. Either way it wants a human. */
static void test_exempt_bits_are_not_quietly_swept(void **state)
{
  int covered = 0;
  gboolean op, gop, ref, gst;
  _poke_coverage(&covered, &op, &gop, &ref, &gst);

  for(int i = 0; i < PANEL_BITS_N; i++)
  {
    if(_panel_bits[i].covered) continue;
    if(covered & _panel_bits[i].bit)
      fail_msg("state bit 0x%x (%s) is declared render-irrelevant but a poke"
               " sets it -- reclassify it or stop poking it",
               _panel_bits[i].bit, _panel_bits[i].where);
  }
}

/* The non-state fields the panel writes: per-shape opacity (blend_gui.c:3926),
   group opacity (10187), refinement (3523/3533/3540/14714) and the group
   break (3134/3170/5031/7763). `name` is deliberately absent -- it is a label
   the renderer never reads. */
static void test_every_panel_written_field_is_swept(void **state)
{
  int covered = 0;
  gboolean op, gop, ref, gst;
  _poke_coverage(&covered, &op, &gop, &ref, &gst);

  assert_true(op);   // blend_gui.c:3926
  assert_true(gop);  // blend_gui.c:10187
  assert_true(ref);  // blend_gui.c:3523
  assert_true(gst);  // blend_gui.c:3134
}

/* The tripwire.

   A new field on dt_masks_point_group_t is how a new panel control arrives,
   and a control whose field no poke touches is invisible to every harvest
   check. Pinning the size forces whoever adds one to come here and say which
   it is: swept, or render-irrelevant.

   If this fires, do NOT just bump the number. Add the field to _panel_bits or
   to the field test above, or record why the renderer cannot see it. */
static void test_the_struct_has_not_grown(void **state)
{
  // formid + parentid + state + opacity + refinement + name[128]
  // + group_opacity + group_start
  assert_int_equal(sizeof(dt_masks_point_group_t),
                   sizeof(dt_mask_id_t) * 2
                   + sizeof(int)
                   + sizeof(float)
                   + sizeof(dt_masks_refinement_t)
                   + 128
                   + sizeof(float)
                   + sizeof(int));
}

int main(void)
{
  const struct CMUnitTest tests[] =
  {
    cmocka_unit_test(test_every_panel_state_bit_is_swept_or_exempt),
    cmocka_unit_test(test_exempt_bits_are_not_quietly_swept),
    cmocka_unit_test(test_every_panel_written_field_is_swept),
    cmocka_unit_test(test_the_struct_has_not_grown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
