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

/** Give `channel` a genuinely partial selection range.

    The four stored values per channel are the slider handles. All zeros --
    what a freshly-zeroed fixture has -- is not a partial range but an empty
    one, and [0,0,1,1] (darktable's default) is the full range; both classify
    as degenerate and collapse to a uniform blend, so a test using either would
    never reach the branch it meant to test. */
static void _set_partial_range(const int channel)
{
  float *const p = flexi_bp.blendif_parameters + 4 * channel;
  p[0] = 0.2f;
  p[1] = 0.3f;
  p[2] = 0.7f;
  p[3] = 0.8f;
}

/** Two active channels, each with a real partial range.

    One channel is enough to reach the DT_COND_REAL branch, but not enough to
    exercise the combine algebra: DEVELOP_COMBINE_INCL governs how channels
    combine with each other as well as how the parametric mask combines with
    the drawn one, so with a single channel the inclusive and exclusive cases
    can coincide and a test would pass without distinguishing them. */
static uint32_t _two_active_channel_bits(void)
{
  const dt_iop_gui_blendif_channel_t *ch =
    dt_develop_blendif_channels_for_csp(DEVELOP_BLEND_CS_RGB_SCENE);
  assert_non_null(ch);
  assert_non_null(ch[1].name);

  const int c0 = ch[0].param_channels[0];
  const int c1 = ch[1].param_channels[0];
  _set_partial_range(c0);
  _set_partial_range(c1);
  return (1u << c0) | (1u << c1);
}

/** Apply the normalisation the legacy pipeline guarantees before flexi
    migration ever sees the params.

    dt_develop_blend_legacy_params_ext() calls _fix_masks_combine() in every
    single version branch, and only then runs the flexi migration -- so a
    *drawn* mask can never reach migration with DEVELOP_COMBINE_INV still set;
    it has already been rewritten to DEVELOP_COMBINE_MASKS_POS (or cancelled
    against an existing one). These tests call the migration directly, which
    bypasses that, so they have to reproduce the precondition or they would be
    asserting against inputs the real code path cannot produce. */
static void _apply_legacy_combine_fix(void)
{
  if(!(flexi_bp.mask_mode & DEVELOP_MASK_MASK)) return;

  const gboolean m_inv = (flexi_bp.mask_combine & DEVELOP_COMBINE_INV) != 0;
  const gboolean m_pos = (flexi_bp.mask_combine & DEVELOP_COMBINE_MASKS_POS) != 0;
  if(m_inv && !m_pos)
  {
    flexi_bp.mask_combine &= ~(uint32_t)DEVELOP_COMBINE_INV;
    flexi_bp.mask_combine |= DEVELOP_COMBINE_MASKS_POS;
  }
  else if(m_inv && m_pos)
  {
    flexi_bp.mask_combine &= ~(uint32_t)DEVELOP_COMBINE_INV;
    flexi_bp.mask_combine &= ~(uint32_t)DEVELOP_COMBINE_MASKS_POS;
  }
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

// A group inherited from a classic edit has no combine operator on its bottom
// member: that is what dt_masks_group_add_form() gives a group's first shape,
// and the migration reuses the point list verbatim. Both partitioners have to
// read it as union, or they disagree about where the group ends -- the panel
// showing one group whose within-group mode, opacity, refinement and
// invert-output all read from a head the fold has split off into a run of its
// own, so every one of those controls silently does nothing (#21905).
static void test_classic_head_without_an_operator_keeps_its_group(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK);
  dt_masks_form_t *grp = flexi_group();

  // as classic left it: USE|SHOW on the bottom member, no operator at all
  dt_masks_point_group_t *head = grp->points->data;
  head->state &= ~(int)DT_MASKS_STATE_OP;

  assert_true(_migrate());

  // the fold's boundary test (dt_masks_point_breaks_run, masks.h) and the
  // panel's (_starts_group, blend_gui.c) both keep the member above in the run
  const dt_masks_point_group_t *above = grp->points->next->data;
  assert_false(dt_masks_point_breaks_run(above, dt_masks_eff_group_op(head->state)));
  assert_false(_starts_group(grp->points->next));
}

// ... and it still has to read as union once the group is bypassed or its
// output inverted. Those are modifiers layered on an operator, not operators
// themselves, so an operator-less head carrying one is still missing its
// combine bit -- resolving the whole of DT_MASKS_STATE_OP instead would make
// the head look like it had an operator after all, and split the group again
// the moment the user touched either control (found by --postedit-masks).
static void test_a_modifier_is_not_an_operator(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK);
  dt_masks_form_t *grp = flexi_group();
  dt_masks_point_group_t *head = grp->points->data;
  dt_masks_point_group_t *above = grp->points->next->data;
  head->state &= ~(int)DT_MASKS_STATE_OP;

  assert_true(_migrate());

  const int modifiers[] = { DT_MASKS_STATE_OP_DISABLE, DT_MASKS_STATE_OP_INVERT };
  for(int m = 0; m < 2; m++)
  {
    // the panel broadcasts a modifier across every member of the run, so both
    // ends carry it -- which is exactly when the two readings can disagree
    head->state |= modifiers[m];
    above->state |= modifiers[m];
    assert_false(dt_masks_point_breaks_run(above, dt_masks_eff_group_op(head->state)));
    assert_false(_starts_group(grp->points->next));
    head->state &= ~modifiers[m];
    above->state &= ~modifiers[m];
  }
}

// defensive: a mask_id that resolves to nothing must still migrate cleanly --
// flexi's "no form" fallback matches classic's, so nothing is fabricated

/* Every stored history snapshot gets normalized, not only the newest.

   dt_dev_read_history_ext() upgrades EVERY history item's blend_params to
   FLEXI (the legacy conversion runs inside its per-row loop) and stores them
   all. If normalization only reached the newest, every earlier item was left
   as flexi params over an unnormalized tree -- #21905, preserved at that
   history position -- and dt_dev_pop_history_items_ext() renders exactly that
   tree when the history slider goes back past the newest mask edit.

   Two items here, each with its own deep copy of the forms, the older one NOT
   the snapshot owner. Both must come out with the run boundary marked. */
static void test_every_history_snapshot_is_normalized(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK);
  dt_masks_form_t *grp = flexi_group();

  // a member carrying a non-union operator: the split has to give it its own
  // run, and the classic tree does not say so
  dt_masks_point_group_t *above = grp->points->next->data;
  above->state = (above->state & ~(int)DT_MASKS_STATE_OP_COMBINE)
                 | (int)DT_MASKS_STATE_DIFFERENCE;
  above->group_start = 0;

  assert_true(_migrate());

  /* The fixture hand-builds its group, so it has no ->functions; a real one
     gets dt_masks_functions_group from dt_masks_create(). That matters here
     and nowhere else in this file: dt_masks_dup_masks_form() copies a form's
     points only when functions->point_struct_size is set, so without it every
     snapshot below would be a group with no members and the test would be
     asserting on nothing. */
  for(GList *f = flexi_dev.forms; f; f = g_list_next(f))
  {
    dt_masks_form_t *form = f->data;
    if(form->type & DT_MASKS_GROUP) form->functions = &dt_masks_functions_group;
  }

  // two items, older first, each with its own copy of the tree -- as
  // dt_dev_add_masks_history_item_ext() would have left them
  dt_dev_history_item_t older = { 0 }, newer = { 0 };
  older.num = 0;
  newer.num = 1;
  older.forms = dt_masks_dup_forms_deep(flexi_dev.forms, NULL);
  newer.forms = dt_masks_dup_forms_deep(flexi_dev.forms, NULL);
  older.blend_params = &flexi_bp;
  newer.blend_params = &flexi_bp;
  flexi_dev.history = g_list_append(NULL, &older);
  flexi_dev.history = g_list_append(flexi_dev.history, &newer);
  flexi_dev.history_end = 2;

  // clear the markers the inline pass already wrote, so the assertions below
  // can only pass if this call put them back on BOTH snapshots
  for(GList *h = flexi_dev.history; h; h = g_list_next(h))
  {
    const dt_dev_history_item_t *it = h->data;
    dt_masks_form_t *g = dt_masks_get_from_id_ext(it->forms, flexi_bp.mask_id);
    assert_non_null(g);
    for(GList *p = g->points; p; p = g_list_next(p))
      ((dt_masks_point_group_t *)p->data)->group_start = 0;
  }

  flexi_dev.pending_flexi_group_splits =
    g_list_append(NULL, GINT_TO_POINTER(flexi_bp.mask_id));
  dt_masks_normalize_flexi_groups(&flexi_dev);

  for(GList *h = flexi_dev.history; h; h = g_list_next(h))
  {
    const dt_dev_history_item_t *it = h->data;
    dt_masks_form_t *g = dt_masks_get_from_id_ext(it->forms, flexi_bp.mask_id);
    assert_non_null(g);
    assert_true((g->type & DT_MASKS_GROUP) != 0);
    assert_int_equal(2, g_list_length(g->points));
    const dt_masks_point_group_t *m = g->points->next->data;
    assert_int_equal(1, m->group_start);
  }

  g_list_free_full(older.forms, (void (*)(void *))dt_masks_free_form);
  g_list_free_full(newer.forms, (void (*)(void *))dt_masks_free_form);
  g_list_free(flexi_dev.history);
  flexi_dev.history = NULL;
  flexi_dev.history_end = 0;
}

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

// ... but not when there is nothing to invert. A raster whose source module was
// removed can never resolve; classic reads its invert flag only inside the
// branch that got a mask back, so with no source it fills 0.0f and the module
// contributes nothing. Carrying the bit across would make the element render as
// 1.0 everywhere instead -- the module going from doing nothing to applying at
// full strength, which is what a real harvested edit did (kofa_1: the whole mask
// off by 1.0, the image by 15.9).
static void test_raster_inversion_is_dropped_when_the_source_is_gone(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_RASTER);
  flexi_bp.raster_mask_source[0] = '\0';
  flexi_bp.raster_mask_id = -1;
  flexi_bp.raster_mask_invert = TRUE;

  assert_true(_migrate());
  _assert_flexi();

  dt_masks_form_t *grp = dt_masks_get_from_id(&flexi_dev, flexi_bp.mask_id);
  assert_non_null(grp);
  gboolean saw_raster = FALSE;
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    const dt_masks_form_t *f = dt_masks_get_from_id(&flexi_dev, pt->formid);
    if(f && (f->type & DT_MASKS_RASTER))
    {
      saw_raster = TRUE;
      assert_false(pt->state & DT_MASKS_STATE_INVERSE);
    }
  }
  assert_true(saw_raster);
}

// ---------------------------------------------------------------------------
// the broken-raster rule
// ---------------------------------------------------------------------------

// Migration can only drop the inversion for a source that is already gone when
// it runs. A source deleted *afterwards* leaves a form that looks perfectly
// well-formed, so the renderer has to make the same call at render time -- it
// asks dt_masks_raster_is_unresolved(), and the group fold skips the inversion
// when it says yes. This pins the predicate that decision rests on.

static dt_masks_form_t *_raster_form(const char *source, const int instance)
{
  dt_masks_form_t *f = calloc(1, sizeof(dt_masks_form_t));
  f->type = DT_MASKS_RASTER;
  f->formid = 4242;
  dt_masks_point_raster_t *p = calloc(1, sizeof(dt_masks_point_raster_t));
  g_strlcpy(p->source, source, sizeof(p->source));
  p->instance = instance;
  p->id = 0;
  f->points = g_list_append(f->points, p);
  return f;
}

static void _free_raster_form(dt_masks_form_t *f)
{
  g_list_free_full(f->points, free);
  free(f);
}

// no module flags in particular: what makes the stand-in a raster writer here
// is its own mask_mode, the other half of the same test in the predicate
static int _no_flags(void)
{
  return 0;
}

// a source module sitting in the pipe at the instance the form names, switched
// on and carrying a mask of its own -- i.e. one that genuinely publishes a
// raster mask, which is the only state the predicate calls resolved
static void _push_source_module(dt_iop_module_t *mod,
                                dt_iop_module_so_t *so,
                                dt_develop_blend_params_t *bp,
                                const char *op,
                                const int instance)
{
  memset(mod, 0, sizeof(*mod));
  memset(so, 0, sizeof(*so));
  memset(bp, 0, sizeof(*bp));
  g_strlcpy(so->op, op, sizeof(so->op));
  mod->so = so;
  mod->multi_priority = instance;
  mod->enabled = TRUE;
  mod->flags = _no_flags;
  bp->mask_mode = DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK;
  mod->blend_params = bp;
  flexi_dev.iop = g_list_append(flexi_dev.iop, mod);
}

static void test_raster_with_no_source_is_unresolved(void **state)
{
  dt_masks_form_t *f = _raster_form("", 0);
  assert_true(dt_masks_raster_is_unresolved(&flexi_module, NULL, f));
  _free_raster_form(f);
}

static void test_raster_whose_module_is_absent_is_unresolved(void **state)
{
  dt_masks_form_t *f = _raster_form("colorbalancergb", 0);
  // nothing in the pipe at all: the module was deleted after this form was made
  assert_true(dt_masks_raster_is_unresolved(&flexi_module, NULL, f));
  _free_raster_form(f);
}

static void test_raster_resolves_to_a_live_module(void **state)
{
  dt_iop_module_t mod;
  dt_iop_module_so_t so;
  dt_develop_blend_params_t bp;
  _push_source_module(&mod, &so, &bp, "colorbalancergb", 0);

  dt_masks_form_t *f = _raster_form("colorbalancergb", 0);
  assert_false(dt_masks_raster_is_unresolved(&flexi_module, NULL, f));
  _free_raster_form(f);

  // the same op at a different instance is a different module, and does not
  // satisfy a form that names instance 0
  dt_masks_form_t *other = _raster_form("colorbalancergb", 3);
  assert_true(dt_masks_raster_is_unresolved(&flexi_module, NULL, other));
  _free_raster_form(other);

  g_list_free(flexi_dev.iop);
  flexi_dev.iop = NULL;
}

// a switched-off source publishes nothing: dt_dev_get_raster_mask() drops its
// mask (and deletes it as stale), so the element is just as unable to draw
// anything as one whose module was deleted outright
static void test_raster_from_a_disabled_module_is_unresolved(void **state)
{
  dt_iop_module_t mod;
  dt_iop_module_so_t so;
  dt_develop_blend_params_t bp;
  _push_source_module(&mod, &so, &bp, "colorbalancergb", 0);
  mod.enabled = FALSE;

  dt_masks_form_t *f = _raster_form("colorbalancergb", 0);
  assert_true(dt_masks_raster_is_unresolved(&flexi_module, NULL, f));

  // and it comes back the moment the source is switched on again -- the state
  // is transient in the user's hands, so the predicate must not latch
  mod.enabled = TRUE;
  assert_false(dt_masks_raster_is_unresolved(&flexi_module, NULL, f));

  _free_raster_form(f);
  g_list_free(flexi_dev.iop);
  flexi_dev.iop = NULL;
}

// an enabled module with no mask of its own and no IOP_FLAGS_WRITE_RASTER never
// puts anything in the raster table, so a reference to it is equally empty
static void test_raster_from_a_module_that_writes_no_mask_is_unresolved(void **state)
{
  dt_iop_module_t mod;
  dt_iop_module_so_t so;
  dt_develop_blend_params_t bp;
  _push_source_module(&mod, &so, &bp, "colorbalancergb", 0);
  bp.mask_mode = DEVELOP_MASK_ENABLED;   // uniform opacity: no mask published

  dt_masks_form_t *f = _raster_form("colorbalancergb", 0);
  assert_true(dt_masks_raster_is_unresolved(&flexi_module, NULL, f));
  _free_raster_form(f);

  g_list_free(flexi_dev.iop);
  flexi_dev.iop = NULL;
}

// the rule is about raster forms only: everything else resolves vacuously, so a
// shape or a parametric channel must never be reported as broken
static void test_non_raster_forms_are_never_unresolved(void **state)
{
  dt_masks_form_t *grp = flexi_build("u:1,2");
  assert_non_null(grp);
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    const dt_masks_form_t *f = dt_masks_get_from_id(&flexi_dev, pt->formid);
    assert_false(dt_masks_raster_is_unresolved(&flexi_module, NULL, f));
  }
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

// ---------------------------------------------------------------------------
// mask_combine: the INV / INCL / MASKS_POS algebra, exhaustively
//
// mask_combine is three bits, so the whole space is eight values and can be
// enumerated rather than sampled. That is worth doing deliberately, because
// these are the branches least likely to be reached by accident: in a real
// 2468-edit library, INCL appeared 8 times and INV not once. There is no need
// to wait for a contributor who happens to use them -- the space is small
// enough to cover outright, and covering it is strictly better than sampling.
//
// Two preconditions have to be reproduced or these tests assert against inputs
// the real code path cannot produce (see _apply_legacy_combine_fix), and one
// classification rule has to be respected: with INCL set, _classify_conditional
// flips the polarity of *every* channel in the colourspace's mask, so any
// channel left inactive becomes a "canceling" channel and the whole config
// collapses to DT_COND_CONSTANT. Reaching DT_COND_REAL with INCL therefore
// requires *all* channels active, not merely some.
// ---------------------------------------------------------------------------

/** Every channel of the fixture's colourspace, active, each with a partial
    range -- the only way to reach DT_COND_REAL while INCL is set. */
static uint32_t _all_channel_bits(void)
{
  const uint32_t mask = DEVELOP_BLENDIF_RGB_MASK; // fixture is RGB_SCENE
  for(int c = 0; c < DEVELOP_BLENDIF_SIZE; c++)
    if(mask & (1u << c)) _set_partial_range(c);
  return mask;
}

/** A blendif guaranteed to classify as DT_COND_REAL for this `incl`. */
static uint32_t _real_blendif(const gboolean incl)
{
  return incl ? _all_channel_bits() : _two_active_channel_bits();
}

// Where INV/INCL actually mean something -- a parametric mask -- migration must
// not leave either set. They are classic's way of spelling inversion, and the
// blendif evaluators read them directly; a survivor would be applied twice,
// once by the flexi renderer and once inside the synthesized parametric form.
//
// Drawn-only is deliberately not covered: with no parametric mask there is no
// blendif evaluation to read either bit, so INCL is inert there and migration
// leaves it alone. (INV cannot even arrive -- _fix_masks_combine has already
// rewritten it upstream.)
static void test_migration_never_leaves_inv_or_incl_set(void **state)
{
  const uint32_t modes[] = {
    DEVELOP_MASK_ENABLED | DEVELOP_MASK_CONDITIONAL,
    DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK | DEVELOP_MASK_CONDITIONAL,
  };

  for(size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); m++)
    for(uint32_t combine = 0; combine < 8; combine++)
    {
      _classic(modes[m]);
      flexi_bp.mask_combine = combine;
      flexi_bp.blendif = _real_blendif((combine & DEVELOP_COMBINE_INCL) != 0);
      _apply_legacy_combine_fix();
      assert_true(_migrate());

      if(flexi_bp.mask_combine & (DEVELOP_COMBINE_INV | DEVELOP_COMBINE_INCL))
        fail_msg("mode 0x%x combine 0x%x migrated to combine 0x%x, which still "
                 "carries INV or INCL", modes[m], combine, flexi_bp.mask_combine);

      flexi_teardown();
    }
}

// Parametric-only, reaching DT_COND_REAL: the whole of INV and INCL folds onto
// MASKS_POS as their inequality. INCL pre-flips each channel's own polarity
// bit, which accounts for its contribution, leaving INV to be re-expressed;
// with both set they cancel.
static void test_parametric_only_folds_inv_and_incl_onto_masks_pos(void **state)
{
  for(uint32_t combine = 0; combine < 8; combine++)
  {
    const gboolean incl = (combine & DEVELOP_COMBINE_INCL) != 0;
    _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_CONDITIONAL);
    flexi_bp.mask_combine = combine;
    flexi_bp.blendif = _real_blendif(incl);
    // no drawn mask here, so the legacy fix is a no-op; kept for symmetry
    _apply_legacy_combine_fix();
    const gboolean inv = (flexi_bp.mask_combine & DEVELOP_COMBINE_INV) != 0;
    assert_true(_migrate());

    const gboolean expect = (incl != inv);
    const gboolean got =
      (flexi_bp.mask_combine & DEVELOP_COMBINE_MASKS_POS) != 0;

    if(got != expect)
      fail_msg("parametric-only combine 0x%x (incl=%d inv=%d): expected "
               "MASKS_POS=%d after migration, got %d",
               combine, incl, inv, expect, got);

    flexi_teardown();
  }
}

// INCL with only *some* channels active is not a partial selection at all: the
// polarity flip turns every inactive channel into a canceling one, and classic
// replaces the whole mask buffer with a constant. Migration has to reproduce
// that, not the per-channel curve the configuration appears to describe.
//
// This is the case that makes "just set INCL and two channels" the wrong way
// to build a test for the INCL algebra -- it never reaches it.
static void test_inclusive_with_partial_channels_collapses_to_a_constant(void **state)
{
  for(uint32_t inv_bit = 0; inv_bit < 2; inv_bit++)
  {
    const uint32_t combine = DEVELOP_COMBINE_INCL
                           | (inv_bit ? DEVELOP_COMBINE_INV : 0);
    _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_CONDITIONAL);
    flexi_bp.mask_combine = combine;
    flexi_bp.blendif = _two_active_channel_bits(); // deliberately not all
    assert_true(_migrate());

    // a plain uniform blend: no form, no mask id, no leftover polarity
    assert_int_equal(flexi_bp.mask_mode, DEVELOP_MASK_ENABLED);
    assert_int_equal(flexi_bp.mask_id, NO_MASKID);
    assert_int_equal(_count_forms(DT_MASKS_PARAMETRIC), 0);
    assert_false(flexi_bp.mask_combine & DEVELOP_COMBINE_MASKS_POS);

    // classic's fill is opac = (INV==0)^(INCL==0); with INCL set that is
    // opaque exactly when INV is not
    const gboolean opaque = (inv_bit == 0);
    if(opaque) assert_true(flexi_bp.opacity != 0.0f);
    else assert_float_equal(flexi_bp.opacity, 0.0f, 1e-9);

    flexi_teardown();
  }
}

// Drawn AND parametric, reaching DT_COND_REAL: the composite inversion is
// INV xor INCL, and that is what lands on MASKS_POS. The drawn side's own
// inversion (MASKS_POS xor INCL) is carried separately, on the drawn element's
// state bit -- invert(d)*p is not invert(d*p), so the two cannot share a flag.
//
// Expectations are computed from the mask_combine that survives
// _apply_legacy_combine_fix(), not from the raw loop value: with a drawn mask
// present, INV has already been rewritten before migration ever sees it.
static void test_drawn_and_parametric_folds_composite_invert_onto_masks_pos(void **state)
{
  for(uint32_t combine = 0; combine < 8; combine++)
  {
    const gboolean incl = (combine & DEVELOP_COMBINE_INCL) != 0;
    _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK | DEVELOP_MASK_CONDITIONAL);
    flexi_bp.mask_combine = combine;
    flexi_bp.blendif = _real_blendif(incl);
    _apply_legacy_combine_fix();

    const uint32_t effective = flexi_bp.mask_combine;
    const gboolean eff_inv = (effective & DEVELOP_COMBINE_INV) != 0;
    assert_true(_migrate());

    const gboolean expect = (eff_inv != incl);
    const gboolean got =
      (flexi_bp.mask_combine & DEVELOP_COMBINE_MASKS_POS) != 0;

    if(got != expect)
      fail_msg("drawn+parametric combine 0x%x -> effective 0x%x (incl=%d "
               "inv=%d): expected MASKS_POS=%d after migration, got %d",
               combine, effective, incl, eff_inv, expect, got);

    flexi_teardown();
  }
}

// Every combine value must still land in a valid end state -- the same
// invariant test_every_bit_combination_lands_in_a_valid_state asserts across
// mask_mode, now crossed with the combine bits it holds fixed.
static void test_every_combine_value_lands_in_a_valid_state(void **state)
{
  for(uint32_t bits = 0; bits < 16; bits++)
    for(uint32_t combine = 0; combine < 8; combine++)
    {
      const uint32_t mode =
          ((bits & 1) ? DEVELOP_MASK_ENABLED : 0)
        | ((bits & 2) ? DEVELOP_MASK_MASK : 0)
        | ((bits & 4) ? DEVELOP_MASK_CONDITIONAL : 0)
        | ((bits & 8) ? DEVELOP_MASK_RASTER : 0);

      _classic(mode);
      flexi_bp.mask_combine = combine;
      flexi_bp.blendif = _real_blendif((combine & DEVELOP_COMBINE_INCL) != 0);
      _apply_legacy_combine_fix();
      g_strlcpy(flexi_bp.raster_mask_source, "colorbalancergb",
                sizeof(flexi_bp.raster_mask_source));
      _migrate();

      const uint32_t out = flexi_bp.mask_mode;
      const gboolean valid =
        (out == DEVELOP_MASK_DISABLED)
        || (out == DEVELOP_MASK_ENABLED)
        || ((out & DEVELOP_MASK_ENABLED) && (out & DEVELOP_MASK_FLEXI));
      if(!valid)
        fail_msg("mode 0x%x combine 0x%x migrated to 0x%x, not a valid end state",
                 mode, combine, out);
      if(out & (DEVELOP_MASK_MASK | DEVELOP_MASK_CONDITIONAL | DEVELOP_MASK_RASTER))
        fail_msg("mode 0x%x combine 0x%x migrated to 0x%x, which still carries "
                 "a classic mode bit", mode, combine, out);

      flexi_teardown();
    }
}

// ---------------------------------------------------------------------------
// NO_MASKS modules keep their parametric mask across migration
//
// IOP_FLAGS_NO_MASKS (retouch, spots) means the module consumes drawn forms
// itself inside process(), so the blend must not also render the forms behind
// mask_id. That is about *drawn* masks -- such a module can still carry a
// parametric blend mask, which classic evaluates in make_mask() with no group
// involved.
//
// Migration moves that parametric config into a form inside a flexi group, and
// the group is exactly what the gate used to refuse to render. The mask
// collapsed to a flat opacity, silently: structurally the migration was
// perfect, and every one of the 200 tests here passed. It took replaying real
// edits (--verify-masks) to see it, on 24 of 2466, every one retouch in
// parametric-only mode.
//
// This pins the distinction cheaply so it cannot come back without a failure.
// ---------------------------------------------------------------------------

static int _flags_no_masks(void)
{
  return IOP_FLAGS_SUPPORTS_BLENDING | IOP_FLAGS_NO_MASKS;
}

static int _flags_ordinary(void)
{
  return IOP_FLAGS_SUPPORTS_BLENDING;
}

static void test_no_masks_module_blocks_a_classic_drawn_group(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK);
  flexi_module.flags = _flags_no_masks;
  // the original protection: a drawn mask on such a module must not render,
  // or the blend paints the module's own shapes
  assert_false(dt_blend_may_render_group(&flexi_module,
                                         DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK));
}

static void test_no_masks_module_still_renders_a_flexi_group(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_CONDITIONAL);
  flexi_module.flags = _flags_no_masks;
  // ...but a flexi group is not those shapes, and refusing it is what threw
  // away the migrated parametric mask
  assert_true(dt_blend_may_render_group(&flexi_module,
                                        DEVELOP_MASK_ENABLED | DEVELOP_MASK_FLEXI));
}

static void test_ordinary_module_always_renders_its_group(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK);
  flexi_module.flags = _flags_ordinary;
  assert_true(dt_blend_may_render_group(&flexi_module,
                                        DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK));
  assert_true(dt_blend_may_render_group(&flexi_module,
                                        DEVELOP_MASK_ENABLED | DEVELOP_MASK_FLEXI));
}

/** The end-to-end shape of the bug: a parametric-only edit on a NO_MASKS
    module migrates to a flexi group, and that group must be renderable --
    otherwise the parametric mask it now lives in is unreachable. */
static void test_parametric_on_no_masks_module_stays_renderable(void **state)
{
  _classic(DEVELOP_MASK_ENABLED | DEVELOP_MASK_CONDITIONAL);
  flexi_module.flags = _flags_no_masks;
  flexi_bp.blendif = _two_active_channel_bits();
  assert_true(_migrate());
  _assert_flexi();

  // the parametric config now lives in a synthesized form...
  assert_true(_count_forms(DT_MASKS_PARAMETRIC) > 0);
  // ...reachable only through the group, which must therefore render
  assert_true(dt_blend_may_render_group(&flexi_module, flexi_bp.mask_mode));
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
    cmocka_unit_test_teardown(test_classic_head_without_an_operator_keeps_its_group,
                              _teardown),
    cmocka_unit_test_teardown(test_a_modifier_is_not_an_operator, _teardown),
    cmocka_unit_test_teardown(test_every_history_snapshot_is_normalized, _teardown),
    cmocka_unit_test_teardown(test_drawn_with_dangling_mask_id, _teardown),
    cmocka_unit_test_teardown(test_parametric_only_synthesizes_a_parametric_form, _teardown),
    cmocka_unit_test_teardown(test_drawn_and_parametric_stacks_a_parametric_element, _teardown),
    cmocka_unit_test_teardown(test_raster_synthesizes_a_raster_form, _teardown),
    cmocka_unit_test_teardown(test_raster_inversion_moves_onto_the_state_bit, _teardown),
    cmocka_unit_test_teardown(test_raster_inversion_is_dropped_when_the_source_is_gone,
                              _teardown),
    cmocka_unit_test_teardown(test_raster_with_no_source_is_unresolved, _teardown),
    cmocka_unit_test_teardown(test_raster_whose_module_is_absent_is_unresolved, _teardown),
    cmocka_unit_test_teardown(test_raster_resolves_to_a_live_module, _teardown),
    cmocka_unit_test_teardown(test_raster_from_a_disabled_module_is_unresolved, _teardown),
    cmocka_unit_test_teardown(test_raster_from_a_module_that_writes_no_mask_is_unresolved,
                              _teardown),
    cmocka_unit_test_teardown(test_non_raster_forms_are_never_unresolved, _teardown),
    cmocka_unit_test_teardown(test_raster_wins_over_drawn, _teardown),
    cmocka_unit_test_teardown(test_raster_wins_over_parametric, _teardown),
    cmocka_unit_test_teardown(test_raster_wins_over_both, _teardown),
    cmocka_unit_test_teardown(test_mode_bit_without_enabled_gets_enabled, _teardown),
    cmocka_unit_test_teardown(test_degenerate_parametric_collapses_to_uniform, _teardown),
    cmocka_unit_test_teardown(test_parametric_in_raw_colorspace_is_degenerate, _teardown),
    cmocka_unit_test_teardown(test_already_flexi_passes_through, _teardown),
    cmocka_unit_test_teardown(test_every_bit_combination_lands_in_a_valid_state, _teardown),
    cmocka_unit_test_teardown(test_migration_never_leaves_inv_or_incl_set, _teardown),
    cmocka_unit_test_teardown(test_parametric_only_folds_inv_and_incl_onto_masks_pos, _teardown),
    cmocka_unit_test_teardown(test_inclusive_with_partial_channels_collapses_to_a_constant, _teardown),
    cmocka_unit_test_teardown(test_drawn_and_parametric_folds_composite_invert_onto_masks_pos, _teardown),
    cmocka_unit_test_teardown(test_every_combine_value_lands_in_a_valid_state, _teardown),
    cmocka_unit_test_teardown(test_no_masks_module_blocks_a_classic_drawn_group, _teardown),
    cmocka_unit_test_teardown(test_no_masks_module_still_renders_a_flexi_group, _teardown),
    cmocka_unit_test_teardown(test_ordinary_module_always_renders_its_group, _teardown),
    cmocka_unit_test_teardown(test_parametric_on_no_masks_module_stays_renderable, _teardown),
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
