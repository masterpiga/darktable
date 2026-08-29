/*
    This file is part of darktable,
    Copyright (C) 2013-2026 darktable developers.

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

/* One-time conversion of a module's classic (pre-flexi) mask_mode --
 * DEVELOP_MASK_MASK / _CONDITIONAL / _RASTER / _MASK_CONDITIONAL -- into the
 * flexi representation (DEVELOP_MASK_FLEXI), so flexi becomes the only mask
 * editor a module ever needs and the classic mode-specific UI/rendering can
 * eventually be retired.
 *
 * Entry point: dt_masks_migrate_classic_to_flexi(), called from
 * dt_develop_blend_legacy_params_ext() (blend.c) as the tail of every
 * successful blend-params version upgrade -- see that function's own comment
 * for why it runs unconditionally after every branch, not just one version
 * step.
 *
 * Design constraints (see masks_revamp_flexi_migration_plan.md for the full
 * case-by-case rationale):
 *
 *  - DEVELOP_MASK_MASK needs no transformation at all: flexi renders a drawn
 *    group through the exact same code path as classic (mode_drawn covers
 *    both bits, see blend.c), so reusing mask_id verbatim and just flipping
 *    the mode bit is already correct.
 *
 *  - DEVELOP_MASK_CONDITIONAL and DEVELOP_MASK_RASTER live entirely as
 *    scalar fields on blend_params in classic mode, outside the form tree --
 *    they are synthesized here as new DT_MASKS_PARAMETRIC / DT_MASKS_RASTER
 *    form elements.
 *
 *  - DEVELOP_MASK_MASK_CONDITIONAL (drawn AND parametric, combined by
 *    multiplication in the classic renderer) synthesizes a new parametric
 *    element and stacks it onto the *existing*, untouched drawn group via
 *    DT_MASKS_STATE_MULTIPLY -- the between-group operator built for exactly
 *    this purpose (see its comment in masks.h).
 *
 *  - Fail closed: on any allocation/synthesis failure, or for a combination
 *    the flexi data model cannot cleanly reproduce (see the DEVELOP_COMBINE_*
 *    handling below), bp is left with its original classic mask_mode
 *    untouched and the module keeps rendering through the classic path --
 *    never silently drop a mask.
 *
 *  - Persistence: forms created here are appended to module->dev->forms (so
 *    the paths that later snapshot dev->forms into a fresh history item --
 *    style application, live preset application -- pick them up) and, when a
 *    real history-stack `num` is known (the darkroom-load path), written
 *    directly into main.masks_history. The latter is required, not optional:
 *    dt_masks_read_masks_history() replaces dev->forms wholesale from the DB
 *    right after the whole history-load loop finishes, which would otherwise
 *    silently discard anything only sitting in dev->forms in memory (see
 *    dt_dev_read_history_ext() in develop.c).
 *
 *  - Every *existing* main.masks_history row belongs to a whole cumulative
 *    snapshot: darktable writes each history item's forms as a full copy of
 *    dev->forms as it stood at that step (see dt_dev_write_history_item() in
 *    develop.c), which is why a classic drawn mask made in an early step is
 *    still found by every later step's own masks_history rows. A newly
 *    synthesized form has no such history -- writing it only under the row
 *    that created it would make it vanish again the moment
 *    dt_masks_read_masks_history() (which only ever looks at the *current*
 *    step, dev->history_end - 1) re-reads the image, unless that also
 *    happens to be the last step. So for the darkroom-load path (a real
 *    history_num), synthesis is deferred to dt_masks_finish_flexi_migrations()
 *    and written directly under history_end - 1 instead -- see its own
 *    comment, and dev->pending_flexi_migrations in develop.h, for why.
 */

#include "common/darktable.h"
#include "common/debug.h"
#include "control/control.h"
#include "develop/blend.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/masks.h"

// ---------------------------------------------------------------------------
// construction helpers
// ---------------------------------------------------------------------------

/* One reused classic drawn group, queued for the post-read pass (see
   _queue_group_split() and dev->pending_flexi_group_splits in develop.h).
   Carries the pre-migration blend params so that pass can also *undo* the
   migration for a group flexi cannot render faithfully. */
typedef struct dt_masks_pending_split_t
{
  dt_mask_id_t mask_id;
  dt_iop_module_t *module;
  dt_develop_blend_params_t *bp;    // the live params the migration wrote
  dt_develop_blend_params_t classic; // what they held before it
} dt_masks_pending_split_t;

/* Classic applies a shape's combine operator once per SHAPE; the flexi fold
 * applies it once per RUN.
 *
 * The flexi fold (_group_get_mask_roi_flexi in group.c) partitions grp->points
 * into maximal same-operator runs, folds each run's members together with the
 * run's *within-group* mode (SCREEN/ISECT/WITHIN_MULTIPLY; none set = union,
 * i.e. max) and composites the finished sub-mask onto the accumulator with the
 * run's between-group operator -- once. The classic sequential fold instead
 * walks the list applying each member's own operator to the accumulator
 * directly. Same operators (DT_MASKS_STATE_OP_COMBINE is exactly the classic
 * set), applied a different number of times.
 *
 * That is invisible for union, because max is idempotent as well as
 * associative: max'ing a run together and then max'ing it in once equals
 * max'ing each member in individually. It is very visible for the others --
 * SUM (a+b) compounds per application, and INTERSECTION is min(acc, max(e1,e2))
 * one way and min(acc, e1, e2) the other. A real 48-brush mask at 0.1 opacity
 * reached 0.6202 under classic and 0.1723 after migration.
 *
 * Giving every non-union member its own run restores per-member application
 * exactly (verified: the 27 edits in the harvested corpus that diverged all
 * became identical, worst residual 2.98e-08). Consecutive union members are
 * deliberately left merged -- they are already equivalent, and splitting them
 * would turn a 48-stroke mask into 48 one-shape groups in the panel for no
 * behavioural gain.
 *
 * Idempotent: re-marking a member that already starts a run changes nothing,
 * which is what lets this run on every load and also on a group that has since
 * been written back to the database in split form. */
static void _split_nonunion_runs(dt_develop_t *dev,
                                 dt_masks_form_t *grp,
                                 const int depth)
{
  if(!grp || !(grp->type & DT_MASKS_GROUP)) return;
  // a malformed/cyclic tree must not spin here; classic nesting is shallow
  if(depth > 8) return;

  // MULTIPLY is deliberately absent: no classic drawn shape carries it (it is
  // the operator migration itself attaches to a synthesized parametric run),
  // and that run is built already-correct by _migrate_drawn_and_parametric.
  const int non_union = DT_MASKS_STATE_INTERSECTION
                      | DT_MASKS_STATE_DIFFERENCE
                      | DT_MASKS_STATE_SUM
                      | DT_MASKS_STATE_EXCLUSION;

  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = l->data;
    if(pt->state & non_union) pt->group_start = 1;

    // A member can itself be a group, and rendering one recurses back into
    // dt_masks_group_get_mask_roi() -- which reads the *module's* blend_params,
    // now flexi, so the nested group is folded by the run algebra too and needs
    // the same treatment. Missing this left 4 of the 27 corpus divergences
    // unfixed when the split was applied only to the top-level group.
    dt_masks_form_t *child = dt_masks_get_from_id(dev, pt->formid);
    if(child && (child->type & DT_MASKS_GROUP))
      _split_nonunion_runs(dev, child, depth + 1);
  }
}

/* Does any group in this tree hold a member classic renders as a REPLACE?
 *
 * A member with no DT_MASKS_STATE_OP_COMBINE bit at all is the base entry
 * dt_masks_group_add_form() creates for a group's first shape (`if(grp->points)
 * state |= default_operator` -- the first member gets none, having nothing to
 * combine with). Classic's fold special-cases exactly that position:
 * `nb_ok == 0 || (state & UNION)` unions it onto the empty accumulator. But
 * when such a member ends up *after* one that already rendered, classic falls
 * through its entire if/else chain to the final else -- `buffer[i] = op * mask[i]`
 * -- which REPLACES the accumulator and discards every earlier member.
 *
 * Flexi cannot express that. _flexi_apply_group_op() (group.c) maps an
 * operator-less run head to union, and the panel agrees: blend_gui.c repairs
 * `(state & OP_COMBINE) == NONE` to DT_MASKS_STATE_UNION on sight, calling it
 * back-compat that is "never valid for new edits". The same bit pattern means
 * replace to the classic renderer and union to every part of flexi -- so a mask
 * whose earlier members classic throws away would come back after migration, at
 * full strength.
 *
 * This is the trigger behind the nested-group failures the harvest campaign
 * found. Grouping shapes is what produces a second operator-less member, which
 * is why every failure had nested groups and no failure lacked them -- but
 * nesting was never the fault: 849 of thad's 858 nested-group edits migrate
 * exactly. Across seven contributed libraries, 52 edits carry this pattern
 * inside their own mask group and 9 rendered differently; the other 43 only
 * because whatever replaced the discarded members happened to cover them.
 *
 * So: fail closed, per this file's standing rule. The module keeps its classic
 * mask_mode and its classic renderer -- exactly what it uses on master today.
 * A mask flexi renders wrong is worse than one flexi does not yet own.
 *
 * Hidden and disabled members are skipped on both sides of the count: neither
 * renders, so neither can be the earlier member that turns a later one into a
 * replace, nor the replace itself. */
static gboolean _group_has_replace_member(dt_develop_t *dev,
                                          const dt_mask_id_t mask_id,
                                          const int depth)
{
  // same bound as _split_nonunion_runs: a malformed/cyclic tree must not spin
  if(depth > 8) return FALSE;

  const dt_masks_form_t *grp = dt_masks_get_from_id(dev, mask_id);
  if(!grp || !(grp->type & DT_MASKS_GROUP)) return FALSE;

  int live = 0;
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    if(!(pt->state & (DT_MASKS_STATE_HIDDEN | DT_MASKS_STATE_DISABLE)))
    {
      if(live > 0 && (pt->state & DT_MASKS_STATE_OP_COMBINE) == DT_MASKS_STATE_NONE)
        return TRUE;
      live++;
    }
    // a member can itself be a group, folded by the same algebra
    if(_group_has_replace_member(dev, pt->formid, depth + 1)) return TRUE;
  }
  return FALSE;
}

/* Queue a reused classic drawn group for the normalization above, and do it
 * once now.
 *
 * Both halves are needed, for different callers. Doing it now covers the paths
 * that never re-read from the database and instead snapshot dev->forms as it
 * stands (style application, live preset application). Queueing covers the
 * darkroom-load path, where dt_masks_read_masks_history() replaces dev->forms
 * wholesale straight after migration and would discard the in-memory work --
 * see dev->pending_flexi_group_splits.
 *
 * The queue entry also carries what it takes to *undo* the migration later.
 * The replace-member check above needs the group's real member list, and on the
 * darkroom-load path there is no point before synthesis where that exists:
 * drawn-only migrates inline while dev->forms still holds the previous image's
 * forms, and drawn+parametric is deferred to dt_masks_finish_flexi_migrations(),
 * which by construction runs *before* dt_masks_read_masks_history(). So the
 * decision is made where the forms finally are -- in
 * dt_masks_normalize_flexi_groups() -- and failing closed there means putting
 * back the blend params the migration overwrote. */
static void _queue_group_split(dt_iop_module_t *module,
                               const dt_mask_id_t mask_id,
                               dt_develop_blend_params_t *bp,
                               const dt_develop_blend_params_t *const classic)
{
  if(!module->dev || !dt_is_valid_maskid(mask_id)) return;

  _split_nonunion_runs(module->dev, dt_masks_get_from_id(module->dev, mask_id), 0);

  dt_masks_pending_split_t *entry = malloc(sizeof(dt_masks_pending_split_t));
  if(!entry) return;
  entry->mask_id = mask_id;
  entry->module = module;
  entry->bp = bp;
  entry->classic = *classic;
  module->dev->pending_flexi_group_splits =
    g_list_append(module->dev->pending_flexi_group_splits, entry);
}

static dt_masks_point_group_t *_new_group_point(const dt_mask_id_t formid,
                                                const int state)
{
  dt_masks_point_group_t *pt = calloc(1, sizeof(dt_masks_point_group_t));
  if(!pt) return NULL;
  pt->formid = formid;
  pt->state = state;
  // a remembered "last used" opacity has no meaning for a member migration
  // synthesizes -- always start fully opaque, the same convention already
  // used for DT_MASKS_PARAMETRIC/DT_MASKS_RASTER members added interactively
  // (see dt_masks_gui_form_save_creation() in masks.c). For the wrapper
  // entry that re-references an *existing* drawn group (the
  // MASK_CONDITIONAL case below), 1.0 is required for correctness, not just
  // convention: anything less would attenuate the drawn mask by an amount
  // classic rendering never applied.
  pt->opacity = 1.0f;
  // classic has no equivalent of the persistent, multiplicative group-level
  // opacity flexi groups carry (see dt_masks_point_group_t.group_opacity) --
  // 1.0 (no effect) keeps a migrated group's mask bit-identical to classic.
  pt->group_opacity = 1.0f;
  return pt;
}

static void _migration_failed(const dt_iop_module_t *module, const char *why);

// appends `form` to the live in-memory forms list, and -- only when a real
// history-stack position is known -- writes it straight into
// main.masks_history too (see the file header comment for why both are
// needed depending on the caller).
static void
_persist_form(dt_iop_module_t *module, dt_masks_form_t *form, const int history_num)
{
  module->dev->forms = g_list_append(module->dev->forms, form);
  if(history_num >= 0)
    dt_masks_write_masks_history_item(module->dev->image_storage.id, history_num, form);
}

// Builds one DT_MASKS_PARAMETRIC element per channel of `colorspace` that is
// active in `blendif` (already had any DEVELOP_COMBINE_INCL polarity flip
// applied by the caller -- see _channel_polarity_mask), each a proper
// single-channel form (see dt_masks_point_parametric_t::single in blend.h)
// with its own channel-specific lead icon, label and progressive name --
// exactly as if that channel had been added by hand, one at a time (see
// _add_parametric_channel / dt_masks_assign_unique_name).
//
// Deliberately flat: the flexi editing panel has no notion of a group
// nested inside another group (_form_kind() in blend_gui.c does not include
// DT_MASKS_GROUP among the kinds a row can display), so every element this
// returns is meant to be added directly as a member of the caller's own
// group, not wrapped. The caller broadcasts DT_MASKS_STATE_WITHIN_MULTIPLY
// across all of them when there is more than one (true per-pixel
// multiplication, dest *= member -- exactly classic's own `mask *= factor`
// per channel, see e.g. dt_develop_blendif_rgb_jzczhz_make_mask() in
// blendif_rgb_jzczhz.c). Any *composite*-level invert (from
// DEVELOP_COMBINE_INV/_INCL) belongs on the module's own blend_params as
// DEVELOP_COMBINE_MASKS_POS instead of on any one member -- inverting a
// single channel is not the same as inverting their product (invert(a)*b !=
// invert(a*b) in general), and unlike DEVELOP_COMBINE_INV/_INCL,
// DEVELOP_COMBINE_MASKS_POS is never read inside a parametric form's own
// evaluation (_parametric_get_mask_roi() in parametric.c copies mask_combine
// onto a scratch blend_params, but nothing in blendif_lab.c/_rgb_hsl.c/
// _rgb_jzczhz.c/_raw.c ever tests DEVELOP_COMBINE_MASKS_POS) -- only
// dt_develop_blend_process()'s own post-fold check (blend.c) consumes it,
// exactly once, on the whole rendered group.
//
// Returns the list of forms to persist and add as group members, or NULL
// (with *ok FALSE) on allocation failure, after freeing every form already
// built. None of them are added to module->dev->forms yet, so a caller that
// goes on to fail some later allocation of its own can still discard
// everything cleanly (see _migrate_parametric_only /
// _migrate_drawn_and_parametric).
static GList *_build_channel_forms(dt_iop_module_t *module,
                                   const int32_t blend_cst,
                                   const uint32_t blendif,
                                   const float *const blendif_parameters,
                                   const float *const blendif_boost_factors,
                                   gboolean *ok)
{
  *ok = TRUE;

  const dt_iop_gui_blendif_channel_t *channels =
    dt_develop_blendif_channels_for_csp((int)blend_cst);
  int nch = 0;
  if(channels)
    while(channels[nch].label) nch++;

  int active_ch[DEVELOP_BLENDIF_SIZE];
  gboolean active_out[DEVELOP_BLENDIF_SIZE];
  int n_active = 0;
  for(int ch = 0; ch < nch && n_active < DEVELOP_BLENDIF_SIZE; ch++)
  {
    // param_channels[] holds plain slot indices (e.g. DEVELOP_BLENDIF_L_in
    // == 0), not bitmasks -- the actual activity bit is 1 << slot (see
    // blendif_lab.c: `blendif & (1 << DEVELOP_BLENDIF_L_in)`)
    const gboolean in_active = (blendif & (1u << channels[ch].param_channels[0])) != 0;
    const gboolean out_active = (blendif & (1u << channels[ch].param_channels[1])) != 0;
    if(in_active || out_active)
    {
      active_ch[n_active] = ch;
      active_out[n_active] = out_active;
      n_active++;
    }
  }

  GList *out = NULL;
  for(int i = 0; i < n_active; i++)
  {
    dt_masks_form_t *form = dt_masks_create(DT_MASKS_PARAMETRIC);
    dt_masks_point_parametric_t *p =
      form ? calloc(1, sizeof(dt_masks_point_parametric_t)) : NULL;
    if(!form || !p)
    {
      dt_masks_free_form(form);
      g_list_free_full(out, (GDestroyNotify)dt_masks_free_form);
      _migration_failed(module, "allocation failure");
      *ok = FALSE;
      return NULL;
    }
    // keep only this one channel's own active + polarity bits
    const uint32_t ch_mask = (1u << channels[active_ch[i]].param_channels[0])
                             | (1u << channels[active_ch[i]].param_channels[1]);
    p->blendif = blendif & (ch_mask | (ch_mask << 16));
    memcpy(p->blendif_parameters, blendif_parameters,
           4 * DEVELOP_BLENDIF_SIZE * sizeof(float));
    memcpy(p->blendif_boost_factors, blendif_boost_factors,
           DEVELOP_BLENDIF_SIZE * sizeof(float));
    p->colorspace = (uint32_t)blend_cst;
    p->single = 1;
    p->channel = (uint32_t)active_ch[i];
    p->in_out = active_out[i] ? 1u : 0u;
    form->points = g_list_append(form->points, p);
    dt_masks_assign_unique_name(module->dev, form);
    out = g_list_append(out, form);
  }
  return out;
}

// appends one dt_masks_point_group_t per form in `forms` to `*points`,
// broadcasting DT_MASKS_STATE_WITHIN_MULTIPLY across all of them when there
// is more than one (their own run's *within*-run fold, see
// _build_channel_forms). `extra_state` is ORed onto every point on top of
// that -- e.g. DT_MASKS_STATE_MULTIPLY, the *between*-group operator a
// caller needs when these channels' own run must itself multiply into a
// pre-existing member outside this list (see
// _migrate_drawn_and_parametric's drawn_pt). Returns FALSE (and leaves
// `*points` unmodified beyond what it already had) on allocation failure.
static gboolean _append_channel_points(GList *forms,
                                       const dt_mask_id_t parentid,
                                       const int extra_state,
                                       GList **points)
{
  const gboolean within_multiply = forms && forms->next;
  GList *added = NULL;
  for(GList *l = forms; l; l = g_list_next(l))
  {
    const dt_masks_form_t *form = l->data;
    dt_masks_point_group_t *pt = _new_group_point(
      form->formid, DT_MASKS_STATE_SHOW | DT_MASKS_STATE_USE | extra_state
                      | (within_multiply ? DT_MASKS_STATE_WITHIN_MULTIPLY : 0));
    if(!pt)
    {
      g_list_free_full(added, free);
      return FALSE;
    }
    pt->parentid = parentid;
    added = g_list_append(added, pt);
  }
  *points = g_list_concat(*points, added);
  return TRUE;
}

// once a classic module's blendif config has been copied into a synthesized
// DT_MASKS_PARAMETRIC form, the top-level copy left on `n` is not just inert
// leftover data -- dt_develop_blend_process() (blend.c) unconditionally runs
// one more make_mask() pass after rendering the (now flexi) drawn group,
// using `n`'s own mask_mode/mask_combine/blendif. Once migrated, mode_mode
// no longer has DEVELOP_MASK_CONDITIONAL set, so every blendif_*_make_mask()
// variant takes their "mask is not conditional" fallback there -- which
// still honors DEVELOP_COMBINE_INV unconditionally (see e.g.
// dt_develop_blendif_rgb_jzczhz_make_mask() in blendif_rgb_jzczhz.c) and
// inverts the mask a *second* time on top of the correctly-inverted value
// the synthesized form's own render already produced. Harmless when INV was
// never set (the fallback then only re-multiplies by opacity, a no-op at
// 100%), silently wrong whenever it was: clear it here, along with the
// now-fully-superseded blendif fields themselves, so that stray pass is
// left with nothing to act on.
static void _clear_toplevel_blendif(dt_develop_blend_params_t *n)
{
  n->mask_combine &= ~(uint32_t)(DEVELOP_COMBINE_INV | DEVELOP_COMBINE_INCL);
  n->blendif = 0;
  memset(n->blendif_parameters, 0, sizeof(n->blendif_parameters));
  memset(n->blendif_boost_factors, 0, sizeof(n->blendif_boost_factors));
}

// DEVELOP_COMBINE_INCL is not a simple "invert the final result" flag: every
// blendif_*_make_mask() variant that supports it (blendif_lab.c,
// blendif_rgb_hsl.c, blendif_rgb_jzczhz.c -- not blendif_raw.c, which never
// reads it at all) XORs *every channel's own polarity bit* with a
// colorspace-specific mask *before* computing the per-channel selection --
// see e.g. `d->blendif ^ (mask_inclusive ? DEVELOP_BLENDIF_RGB_MASK << 16 :
// 0)` in dt_develop_blendif_rgb_jzczhz_make_mask(). Reproducing INCL in a
// synthesized DT_MASKS_PARAMETRIC form means applying that same XOR to the
// copied `blendif` value up front, not treating INCL as an outer invert.
static uint32_t _channel_polarity_mask(const int32_t blend_cst)
{
  switch(blend_cst)
  {
  case DEVELOP_BLEND_CS_LAB: return DEVELOP_BLENDIF_Lab_MASK;
  case DEVELOP_BLEND_CS_RGB_DISPLAY:
  case DEVELOP_BLEND_CS_RGB_SCENE: return DEVELOP_BLENDIF_RGB_MASK;
  default: return 0; // RAW (and anything else): INCL has no channel-polarity effect there
  }
}

// every blendif_*_make_mask() variant that supports INCL (blendif_lab.c,
// blendif_rgb_hsl.c, blendif_rgb_jzczhz.c -- not blendif_raw.c, which reads
// neither INCL nor this branching at all) actually has *three* distinct
// behaviors, not the two ("real formula" vs "one wholesale constant") this
// migration originally assumed:
//
//  - DT_COND_REAL: at least one channel is genuinely active, and none of
//    the *other* (inactive) channels got spuriously flagged "canceling" by
//    INCL's polarity flip (see _channel_polarity_mask()'s comment). The
//    ordinary per-channel computation runs.
//
//  - DT_COND_PASSTHROUGH: no channel is active at all, *and* INCL didn't
//    flip any of them into a canceling state (i.e. INCL is unset, or the
//    colorspace has no channels to flip -- RAW always lands here). Classic
//    takes the *first* branch of the outer if/else in make_mask() --
//    `mask[x] = opacity * mask[x]` (or `opacity * (1 - mask[x])` if INV) --
//    which *multiplies the incoming buffer*, not replaces it. For resolved
//    drawn content that incoming value is the real, spatially-varying drawn
//    mask: this preserves it exactly (scaled by opacity, optionally
//    inverted by INV alone -- INCL never enters this formula). Only when
//    the incoming value is itself already a flat fallback constant (no
//    drawn content, or no drawn mode at all) does the result reduce to a
//    constant too -- and by a *different* rule than DT_COND_CONSTANT below,
//    since it's driven by whichever bit produced that incoming constant
//    (INCL for pure-parametric, MASKS_POS for drawn+parametric-with-no-
//    content), not by INCL^INV. Missing this distinction (treating
//    "!any_channel_active" as unconditionally the same wholesale-constant
//    case as "canceling_channel") broke real fixtures during development --
//    modules that select "drawn & parametric" mode but never actually
//    configure a channel, effectively using the drawn shape alone.
//
//  - DT_COND_CONSTANT: canceling_channel is set (which requires at least
//    one inactive channel to have been flipped by INCL -- see
//    _channel_polarity_mask()) -- classic takes the *second* branch,
//    `dt_iop_image_fill()`, which *replaces* the buffer wholesale with
//    `opac = ((INV==0)^(INCL==0)) ? global_opacity : 0.0f`, discarding
//    everything: the parametric curve, and (when reached from a mode_drawn
//    call) any already-rendered drawn geometry too. Confirmed empirically
//    against every INCL-with-a-partial-channel-config combination this
//    migration was tested against.
typedef enum
{
  DT_COND_REAL,
  DT_COND_PASSTHROUGH,
  DT_COND_CONSTANT,
} dt_cond_branch_t;

static dt_cond_branch_t _classify_conditional(const int32_t blend_cst,
                                              const uint32_t blendif,
                                              const gboolean incl)
{
  const uint32_t mask = _channel_polarity_mask(blend_cst);
  if(!mask) return DT_COND_PASSTHROUGH; // RAW: no canceling-channel mechanism at all

  const uint32_t any_channel_active = blendif & mask;
  const uint32_t flipped = blendif ^ (incl ? (mask << 16) : 0);
  const uint32_t canceling_channel = (flipped >> 16) & ~flipped & mask;

  if(canceling_channel) return DT_COND_CONSTANT;
  if(any_channel_active) return DT_COND_REAL;
  return DT_COND_PASSTHROUGH;
}

static void _migration_failed(const dt_iop_module_t *module, const char *why)
{
  dt_print(DT_DEBUG_ALWAYS, "[masks] classic->flexi migration failed for module '%s': %s",
           module->op, why);
  dt_control_log(_("%s: failed to migrate mask to flexi, mask kept in classic mode"),
                 module->op);
}

// does `mask_id` resolve to a form with actual content? Mirrors exactly the
// condition dt_develop_blend_process() itself uses to decide whether a drawn
// mask has something to render (`form && form->points`, see blend.c) -- not
// "is it specifically typed DT_MASKS_GROUP", since the renderer does not
// require that either.
//
// dev->forms cannot be trusted for this while inside the darkroom
// history-load loop: dt_masks_read_masks_history() (which populates it from
// this image's own masks_history rows) does not run until the entire
// per-row loop finishes, so at legacy-params time dev->forms is still
// whatever was left over from a previous image (or NULL). A real
// history_num signals exactly that context, so query main.masks_history
// directly in that case; everywhere else (style/preset application, both of
// which operate on a dev whose forms are already fully loaded) dev->forms is
// reliable and cheaper to use.
static gboolean _mask_id_has_content(dt_iop_module_t *module,
                                     const dt_mask_id_t mask_id,
                                     const int history_num)
{
  if(!dt_is_valid_maskid(mask_id)) return FALSE;

  if(history_num < 0)
  {
    const dt_masks_form_t *form = dt_masks_get_from_id(module->dev, mask_id);
    return form && form->points != NULL;
  }

  sqlite3_stmt *stmt;
  DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                              "SELECT points_count FROM main.masks_history"
                              " WHERE imgid = ?1 AND formid = ?2"
                              " ORDER BY num DESC LIMIT 1",
                              -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, module->dev->image_storage.id);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, mask_id);
  gboolean has_content = FALSE;
  if(sqlite3_step(stmt) == SQLITE_ROW) has_content = sqlite3_column_int(stmt, 0) > 0;
  sqlite3_finalize(stmt);
  return has_content;
}

// ---------------------------------------------------------------------------
// per-case synthesis
// ---------------------------------------------------------------------------

// DEVELOP_MASK_CONDITIONAL (pure parametric, no drawn shapes): one group
// holding one legacy-style (single=0, full multi-channel) DT_MASKS_PARAMETRIC
// element, with the classic blendif configuration copied verbatim.
static gboolean _migrate_parametric_only(dt_iop_module_t *module,
                                         const dt_develop_blend_params_t *const o,
                                         dt_develop_blend_params_t *n,
                                         const int history_num)
{
  const gboolean incl = (o->mask_combine & DEVELOP_COMBINE_INCL) != 0;
  const gboolean inv = (o->mask_combine & DEVELOP_COMBINE_INV) != 0;
  const dt_cond_branch_t branch = _classify_conditional(o->blend_cst, o->blendif, incl);

  // both degenerate branches collapse to a constant here (there's no drawn
  // content this function ever deals with -- see _migrate_drawn_and_
  // parametric()'s own, separate DT_COND_PASSTHROUGH handling for the case
  // where real geometry is in play): no form needed at all, just a plain
  // uniform blend (module applies everywhere) or, for "always zero", the
  // same with opacity forced to 0 (opacity multiplies the mask everywhere
  // in the blend math, so opacity=0 reproduces "contributes nothing"
  // exactly). The two branches use *different* parities on purpose: classic
  // reaches them through different code (DT_COND_CONSTANT's `opac =
  // (INV==0)^(INCL==0)`, computed inside make_mask(); DT_COND_PASSTHROUGH's
  // `opacity * mask_in` / `opacity * (1-mask_in)`, with the classic pure-
  // parametric fallback `mask_in = INCL ? 0 : 1` fed in from
  // dt_develop_blend_process() *before* make_mask() even runs) -- they only
  // happen to agree when a real channel is active (DT_COND_REAL below).
  gboolean opaque;
  if(branch == DT_COND_CONSTANT)
    opaque = (incl != inv);
  else if(branch == DT_COND_PASSTHROUGH)
    opaque = (incl == inv);
  if(branch != DT_COND_REAL)
  {
    _clear_toplevel_blendif(n);
    n->mask_mode = DEVELOP_MASK_ENABLED;
    n->mask_id = NO_MASKID;
    n->mask_combine &= ~(uint32_t)DEVELOP_COMBINE_MASKS_POS;
    if(!opaque) n->opacity = 0.0f;
    return TRUE;
  }

  dt_masks_form_t *grp = dt_masks_create(DT_MASKS_GROUP);
  if(!grp)
  {
    _migration_failed(module, "allocation failure");
    return FALSE;
  }

  // DEVELOP_COMBINE_INCL pre-flips every channel's own polarity bit (see
  // _channel_polarity_mask()'s comment) -- bake that into the copied
  // `blendif` value itself, so the synthesized form's per-channel selection
  // matches classic's exactly. Only reached here (past the constant check
  // above) when every channel of the colorspace is simultaneously active,
  // so this flip cannot leave any channel both flipped and inactive.
  const uint32_t flipped_blendif =
    o->blendif ^ (incl ? (_channel_polarity_mask(o->blend_cst) << 16) : 0);

  gboolean built_ok = FALSE;
  GList *param_forms =
    _build_channel_forms(module, o->blend_cst, flipped_blendif, o->blendif_parameters,
                         o->blendif_boost_factors, &built_ok);
  if(!built_ok)
  {
    dt_masks_free_form(grp);
    return FALSE;
  }

  if(!_append_channel_points(param_forms, grp->formid, 0, &grp->points))
  {
    dt_masks_free_form(grp);
    g_list_free_full(param_forms, (GDestroyNotify)dt_masks_free_form);
    _migration_failed(module, "allocation failure");
    return FALSE;
  }

  for(GList *l = param_forms; l; l = g_list_next(l))
    _persist_form(module, l->data, history_num);
  g_list_free(param_forms);
  _persist_form(module, grp, history_num);

  _clear_toplevel_blendif(n);
  n->mask_mode = DEVELOP_MASK_ENABLED | DEVELOP_MASK_FLEXI;
  n->mask_id = grp->formid;
  // With the channel-polarity flip above already accounting for INCL's own
  // contribution, what's left is DEVELOP_COMBINE_INV -- classic applies it
  // *after* the per-channel computation, inverting the whole result (see
  // e.g. dt_develop_blendif_rgb_jzczhz_make_mask() in
  // blendif_rgb_jzczhz.c: `mask = opacity * (1 - mask)` vs `mask = opacity *
  // mask`). That composite-level invert goes on the module's own
  // DEVELOP_COMBINE_MASKS_POS (dt_develop_blend_process()'s post-fold check
  // in blend.c, applied once to the whole rendered group) rather than any
  // one channel's own membership -- inverting a single channel before the
  // product is not the same as inverting the product itself. This bit is
  // never read by _parametric_get_mask_roi() (parametric.c)'s own scratch
  // copy of blend_params -- unlike DEVELOP_COMBINE_INV/_INCL, no
  // blendif_*_make_mask() variant ever tests DEVELOP_COMBINE_MASKS_POS -- so
  // there is no risk of inheriting it a second time here. INCL and INV turn
  // out to be interchangeable at this final step too -- toggling either one
  // alone inverts the classic result, and toggling both together cancels
  // back to normal -- so this is driven by their XOR, not INV alone.
  n->mask_combine &= ~(uint32_t)DEVELOP_COMBINE_MASKS_POS;
  if(incl != inv) n->mask_combine |= DEVELOP_COMBINE_MASKS_POS;
  return TRUE;
}

// DEVELOP_MASK_RASTER: one group holding one DT_MASKS_RASTER element,
// referencing the same source the classic raster_mask_* fields already
// describe. Those fields are deliberately left untouched -- the pipe's
// raster-mask dependency registration (dt_iop_commit_blend_params() /
// _reconcile_raster_form_users() in imageop.c) already walks the flexi form
// tree independently of mask_mode, and rendering a DT_MASKS_RASTER form
// (masks/raster.c) reads them directly too.
static gboolean _migrate_raster(dt_iop_module_t *module,
                                const dt_develop_blend_params_t *const o,
                                dt_develop_blend_params_t *n,
                                const int history_num)
{
  dt_masks_form_t *raster_form = dt_masks_create(DT_MASKS_RASTER);
  dt_masks_form_t *grp = dt_masks_create(DT_MASKS_GROUP);
  if(!raster_form || !grp)
  {
    dt_masks_free_form(raster_form);
    dt_masks_free_form(grp);
    _migration_failed(module, "allocation failure");
    return FALSE;
  }

  dt_masks_point_raster_t *rp = calloc(1, sizeof(dt_masks_point_raster_t));
  if(!rp)
  {
    dt_masks_free_form(raster_form);
    dt_masks_free_form(grp);
    _migration_failed(module, "allocation failure");
    return FALSE;
  }
  g_strlcpy(rp->source, o->raster_mask_source, sizeof(rp->source));
  rp->instance = o->raster_mask_instance;
  rp->id = o->raster_mask_id;
  raster_form->points = g_list_append(raster_form->points, rp);

  // classic applies raster_mask_invert inline (mask[k] = (1-raster[k])*opacity,
  // see dt_develop_blend_process() in blend.c); the flexi group fold applies a
  // member's own DT_MASKS_STATE_INVERSE bit with the identical formula (see
  // _combine_masks_union() in group.c), so this is an exact equivalent, not
  // an approximation.
  int state = DT_MASKS_STATE_SHOW | DT_MASKS_STATE_USE;
  if(o->raster_mask_invert) state |= DT_MASKS_STATE_INVERSE;

  dt_masks_point_group_t *pt = _new_group_point(raster_form->formid, state);
  if(!pt)
  {
    dt_masks_free_form(raster_form);
    dt_masks_free_form(grp);
    _migration_failed(module, "allocation failure");
    return FALSE;
  }
  pt->parentid = grp->formid;
  grp->points = g_list_append(grp->points, pt);

  _persist_form(module, raster_form, history_num);
  _persist_form(module, grp, history_num);

  n->mask_mode = DEVELOP_MASK_ENABLED | DEVELOP_MASK_FLEXI;
  n->mask_id = grp->formid;

  /* classic's raster branch reads NONE of mask_combine: it is an `else if`
     ahead of the drawn/parametric branch in dt_develop_blend_process() (see
     blend.c), so the mask is exactly raster * opacity -- MASKS_POS never
     inverts it, INV never reaches it (the blendif_*_make_mask() call that
     consumes INV lives in the drawn/parametric branch and is not run), and
     INCL only ever feeds a fallback fill that branch also owns.
     Post-migration the group goes *through* that drawn/parametric branch, so
     every one of those bits would suddenly apply to a mask classic rendered
     without them.

     MASKS_POS is not hypothetical: exactly two edits across seven contributed
     libraries pair raster with it, and before this both rendered fully
     inverted (max_diff 1.0). The other two are cleared for the same reason,
     ahead of a corpus that happens to contain them. */
  n->mask_combine &= ~(uint32_t)(DEVELOP_COMBINE_INV
                                 | DEVELOP_COMBINE_INCL
                                 | DEVELOP_COMBINE_MASKS_POS);
  return TRUE;
}

// DEVELOP_MASK_MASK_CONDITIONAL: drawn AND parametric, combined by
// multiplication in the classic renderer. Stacks a new parametric element
// onto the *existing*, untouched drawn group via DT_MASKS_STATE_MULTIPLY.
static gboolean _migrate_drawn_and_parametric(dt_iop_module_t *module,
                                              const dt_develop_blend_params_t *const o,
                                              dt_develop_blend_params_t *n,
                                              const int history_num)
{
  // classic: with no resolvable drawn mask, the "no form" fallback fills
  // 1.0/0.0 depending on DEVELOP_COMBINE_MASKS_POS ("inverted"), *then*
  // multiplies by the parametric result -- see dt_develop_blend_process() in
  // blend.c. That fallback fill plays exactly the same role MASKS_POS's own
  // outer fill plays in the pure-parametric (DEVELOP_MASK_CONDITIONAL-alone)
  // case, just driven by a different bit -- so the same case analysis
  // applies: whether the final result is a real (normal-or-inverted)
  // parametric mask, or a hard constant, depends on whether MASKS_POS and
  // INCL *agree*:
  //
  //  - MASKS_POS == INCL (both set or both unset): the fallback fill and
  //    INCL's own channel-polarity flip work out to a real mask again,
  //    inverted on their XOR with INV -- exactly _migrate_parametric_only()'s
  //    own rule, and MASKS_POS drops out of the picture entirely (there's no
  //    real drawn content for it to describe). Delegate to it directly.
  //  - MASKS_POS != INCL: everything collapses to a hard constant,
  //    independent of the parametric config -- opaque (module applies
  //    uniformly) if INCL XOR INV is set, zero (module is a no-op) if not.
  //    No form is representable *or* needed here: migrate to a plain
  //    uniform blend (no masking at all) at the module's own opacity for
  //    "opaque", or the same with opacity forced to 0 for "zero" -- opacity
  //    multiplies the mask everywhere in the blend math, so opacity=0
  //    reproduces "contributes nothing" exactly.
  if(!_mask_id_has_content(module, o->mask_id, history_num))
  {
    const gboolean masks_pos = (o->mask_combine & DEVELOP_COMBINE_MASKS_POS) != 0;
    const gboolean incl = (o->mask_combine & DEVELOP_COMBINE_INCL) != 0;
    if(masks_pos == incl) return _migrate_parametric_only(module, o, n, history_num);

    const gboolean inv = (o->mask_combine & DEVELOP_COMBINE_INV) != 0;
    const gboolean opaque = (incl != inv);
    _clear_toplevel_blendif(n);
    n->mask_mode = DEVELOP_MASK_ENABLED;
    n->mask_id = NO_MASKID;
    n->mask_combine &= ~(uint32_t)DEVELOP_COMBINE_MASKS_POS;
    if(!opaque) n->opacity = 0.0f;
    return TRUE;
  }

  {
    const gboolean incl = (o->mask_combine & DEVELOP_COMBINE_INCL) != 0;
    const gboolean inv = (o->mask_combine & DEVELOP_COMBINE_INV) != 0;
    const dt_cond_branch_t branch = _classify_conditional(o->blend_cst, o->blendif, incl);

    if(branch == DT_COND_CONSTANT)
    {
      // classic's canceling-channel fallback (dt_iop_image_fill()) replaces
      // the *entire* mask buffer wholesale, discarding the just-rendered
      // drawn geometry along with the parametric curve -- so drawn content
      // being present changes nothing about whether, or to what, this
      // collapses. Confirmed empirically: drawn+parametric with a partial-
      // channel INCL config renders identically to the pure-constant
      // no-content case with the same INCL/INV, regardless of the drawn
      // shape.
      const gboolean opaque = (incl != inv);
      _clear_toplevel_blendif(n);
      n->mask_mode = DEVELOP_MASK_ENABLED;
      n->mask_id = NO_MASKID;
      n->mask_combine &= ~(uint32_t)DEVELOP_COMBINE_MASKS_POS;
      if(!opaque) n->opacity = 0.0f;
      return TRUE;
    }

    if(branch == DT_COND_PASSTHROUGH)
    {
      // no active parametric channel at all (and INCL didn't flip any
      // inactive one into canceling, or the colorspace has none to flip):
      // classic's *first* branch multiplies the already-rendered drawn
      // value `d` by opacity, optionally inverting -- `d` itself is passed
      // through untouched, not replaced, so the parametric side of this
      // mask contributes nothing at all and this collapses to a drawn-only
      // migration (same as _dispatch()'s DEVELOP_MASK_MASK-alone case:
      // reuse mask_id verbatim, no new form). The only wrinkle is that this
      // branch's own INV sits *after* MASKS_POS's drawn-invert
      // (`d' = MASKS_POS ? 1-d : d`, applied unconditionally before
      // make_mask() ever runs) rather than being independent of it the way
      // it is in the DT_COND_REAL case below -- `INV ? 1-d' : d'` -- which
      // collapses to a *single* invert driven by MASKS_POS^INV applied
      // directly to `d` (verified algebraically: every one of the 4
      // MASKS_POS/INV combinations reduces to exactly that XOR). So MASKS_POS
      // and INV fold onto the very same drawn_pt state DEVELOP_MASK_MASK's
      // own invert already uses, with no separate parametric member needed.
      const gboolean masks_pos = (o->mask_combine & DEVELOP_COMBINE_MASKS_POS) != 0;
      _clear_toplevel_blendif(n);
      n->mask_mode = DEVELOP_MASK_ENABLED | DEVELOP_MASK_FLEXI;
      // n->mask_id is already o->mask_id -- reused verbatim, so this is a
      // drawn-only migration in every respect and needs the same run-boundary
      // normalization that _dispatch()'s DEVELOP_MASK_MASK-alone case applies.
      // (DT_COND_CONSTANT just above does not: it sets mask_id to NO_MASKID,
      // so no group is rendered at all.)
      _queue_group_split(module, o->mask_id, n, o);
      if(masks_pos != inv)
        n->mask_combine |= DEVELOP_COMBINE_MASKS_POS;
      else
        n->mask_combine &= ~(uint32_t)DEVELOP_COMBINE_MASKS_POS;
      return TRUE;
    }

    // DT_COND_REAL with INCL set (the only way to reach this while INCL is
    // on: every channel of the colorspace is simultaneously active, the
    // only config that avoids the constant collapse above) -- with real,
    // spatially-varying drawn content `d`, classic's inclusive formula works
    // out to 1-(1-d)*temp when INV=0, (1-d)*temp when INV=1 (see the INV
    // comment below for the non-INCL derivation this generalizes). Both
    // reduce to the *same* multiply-fold construction below, just with an
    // extra XOR(incl) folded into the two invert decisions it already makes
    // (verified algebraically against all 4 INCL/INV combinations; reduces
    // to exactly the existing non-INCL formula when incl=0):
    //   invert_drawn     ^= incl
    //   invert_composite ^= incl
    // plus a pre-flip of the synthesized parametric form's own blendif
    // polarity bits (same trick _migrate_parametric_only() already uses for
    // the pure-parametric INCL case), since incl also flips the per-channel
    // curve *evaluation* itself, independent of the outer formula choice
    // above. No new flexi combine operator or nesting needed.
  }

  dt_masks_form_t *top_grp = dt_masks_create(DT_MASKS_GROUP);
  if(!top_grp)
  {
    _migration_failed(module, "allocation failure");
    return FALSE;
  }

  const gboolean incl = (o->mask_combine & DEVELOP_COMBINE_INCL) != 0;
  const uint32_t flipped_blendif =
    o->blendif ^ (incl ? (_channel_polarity_mask(o->blend_cst) << 16) : 0);

  gboolean built_ok = FALSE;
  GList *param_forms =
    _build_channel_forms(module, o->blend_cst, flipped_blendif, o->blendif_parameters,
                         o->blendif_boost_factors, &built_ok);
  if(!built_ok)
  {
    dt_masks_free_form(top_grp);
    return FALSE;
  }

  // classic has two *independent* inversions in this mode: DEVELOP_COMBINE_
  // MASKS_POS inverts the drawn portion alone, strictly before the
  // parametric multiply (dt_develop_blend_process(), blend.c: applied to
  // `mask` right after dt_masks_group_render_roi() / the classic drawn-form
  // branch, before the make_mask() call that folds in the parametric
  // channels); DEVELOP_COMBINE_INV inverts the *already-multiplied*
  // composite, one step later, inside make_mask() itself (see e.g.
  // dt_develop_blendif_rgb_jzczhz_make_mask() in blendif_rgb_jzczhz.c:
  // `mask = opacity * (1 - mask*temp_mask)` vs `mask = opacity *
  // mask*temp_mask`, where by that point `mask` already reflects any
  // MASKS_POS inversion). invert(d)*p != invert(d*p) in general, so these
  // need two different translations, not one:
  //
  //  - MASKS_POS moves onto the wrapper entry that re-references the drawn
  //    group (applied by the fold before the multiply, exactly matching
  //    classic ordering, see _combine_masks_union() in group.c).
  //
  //  - INV has no per-member equivalent (it applies to the *whole* fold
  //    result, not to either operand alone) -- but dt_develop_blend_process()
  //    already provides exactly that hook for a mode_drawn module: the same
  //    MASKS_POS check runs *again*, unconditionally, right after
  //    dt_masks_group_render_roi() returns the whole drawn+parametric
  //    composite in one call (this is what the drawn-only migration case
  //    relies on unchanged). So INV is translated onto the *module's own*
  //    mask_combine as MASKS_POS instead, letting that existing post-fold
  //    check invert the composite as a whole -- independent of, and not to
  //    be confused with, whatever the original MASKS_POS did (already fully
  //    consumed above, and always cleared below).
  //
  //  - INCL (see the DT_COND_REAL comment above) generalizes both of the
  //    above with a simple XOR: it's only ever reached here alongside a
  //    genuinely spatially-varying `temp` (every channel active, so no
  //    canceling-channel collapse), and the two classic inclusive-formula
  //    variants are exactly the non-inclusive ones with drawn and composite
  //    both additionally flipped.
  const gboolean invert_drawn =
    ((o->mask_combine & DEVELOP_COMBINE_MASKS_POS) != 0) ^ incl;
  const gboolean invert_composite = ((o->mask_combine & DEVELOP_COMBINE_INV) != 0) ^ incl;
  int drawn_state = DT_MASKS_STATE_SHOW | DT_MASKS_STATE_USE;
  if(invert_drawn) drawn_state |= DT_MASKS_STATE_INVERSE;

  dt_masks_point_group_t *drawn_pt = _new_group_point(o->mask_id, drawn_state);
  if(!drawn_pt)
  {
    dt_masks_free_form(top_grp);
    g_list_free_full(param_forms, (GDestroyNotify)dt_masks_free_form);
    _migration_failed(module, "allocation failure");
    return FALSE;
  }
  drawn_pt->parentid = top_grp->formid;
  top_grp->points = g_list_append(top_grp->points, drawn_pt);

  // drawn_pt references the *original* classic drawn group, which is rendered
  // by recursing back into dt_masks_group_get_mask_roi() -- and that recursion
  // reads the module's (now flexi) blend_params, so the inner group is folded
  // by the flexi run algebra too. It therefore needs the same run-boundary
  // normalization as the drawn-only case above.
  _queue_group_split(module, o->mask_id, n, o);

  // the channel elements form their own run, separate from drawn_pt's (a
  // different operator always starts a new run, see the run-boundary test
  // in _group_get_mask_roi_flexi) -- DT_MASKS_STATE_MULTIPLY is that run's
  // *between*-group operator, multiplying its own (possibly multi-channel,
  // WITHIN_MULTIPLY-folded) result into the accumulator drawn_pt already
  // copied in as the first run.
  if(!_append_channel_points(param_forms, top_grp->formid, DT_MASKS_STATE_MULTIPLY,
                             &top_grp->points))
  {
    dt_masks_free_form(top_grp);
    g_list_free_full(param_forms, (GDestroyNotify)dt_masks_free_form);
    _migration_failed(module, "allocation failure");
    return FALSE;
  }

  for(GList *l = param_forms; l; l = g_list_next(l))
    _persist_form(module, l->data, history_num);
  g_list_free(param_forms);
  _persist_form(module, top_grp, history_num);

  _clear_toplevel_blendif(n);
  n->mask_mode = DEVELOP_MASK_ENABLED | DEVELOP_MASK_FLEXI;
  n->mask_id = top_grp->formid;
  // the original MASKS_POS (if any) is now fully represented by drawn_pt's
  // own state above; clear it unconditionally so it does not *also* trigger
  // dt_develop_blend_process()'s post-fold check, then -- independently --
  // set that same bit if INV asked for the composite-level invert instead.
  n->mask_combine &= ~(uint32_t)DEVELOP_COMBINE_MASKS_POS;
  if(invert_composite) n->mask_combine |= DEVELOP_COMBINE_MASKS_POS;
  return TRUE;
}

// dispatches to the right case builder by precedence, mirroring
// dt_develop_blend_process() exactly: raster is checked first and, if set,
// wins outright over any MASK/CONDITIONAL bits also present (an if/else-if
// chain in the renderer, not independent contributions) -- so a
// non-standard combination normalizes to pure raster here too, matching
// what the renderer already does with it. Used both for the immediate path
// (history_num < 0) and, deferred, from dt_masks_finish_flexi_migrations().
static gboolean _dispatch(dt_iop_module_t *module,
                          const dt_develop_blend_params_t *const o,
                          dt_develop_blend_params_t *n,
                          const int history_num)
{
  gboolean ok;

  if(o->mask_mode & DEVELOP_MASK_RASTER)
  {
    if((o->mask_mode & (DEVELOP_MASK_MASK | DEVELOP_MASK_CONDITIONAL)))
      dt_print(
        DT_DEBUG_ALWAYS,
        "[masks] module '%s': non-standard mask_mode 0x%x (RASTER combined with "
        "MASK/CONDITIONAL) -- migrating as pure raster, matching how it already renders",
        module->op, o->mask_mode);
    ok = _migrate_raster(module, o, n, history_num);
  }
  else if((o->mask_mode & DEVELOP_MASK_MASK) && (o->mask_mode & DEVELOP_MASK_CONDITIONAL))
  {
    ok = _migrate_drawn_and_parametric(module, o, n, history_num);
  }
  else if(o->mask_mode & DEVELOP_MASK_MASK)
  {
    // Drawn only: the form tree is reused verbatim, mask_id and all -- but
    // NOT untouched. `mode_drawn` covers both DEVELOP_MASK_MASK and
    // DEVELOP_MASK_FLEXI in blend.c, so the two modes reach the same *call*;
    // they do not reach the same renderer. dt_masks_group_get_mask_roi()
    // dispatches on the FLEXI bit to a different fold, which applies each
    // member's combine operator once per run rather than once per member.
    // _queue_group_split() marks the run boundaries that make the two agree;
    // see its comment for why that is all it takes.
    n->mask_mode = DEVELOP_MASK_ENABLED | DEVELOP_MASK_FLEXI;
    _queue_group_split(module, o->mask_id, n, o);
    ok = TRUE;
  }
  else // DEVELOP_MASK_CONDITIONAL alone
  {
    ok = _migrate_parametric_only(module, o, n, history_num);
  }

  // every mode button in the GUI always writes ENABLED together with its
  // mode bit (see _blendop_masks_mode_callback() in blend_gui.c), so this is
  // unreachable from any current code path -- but if foreign/hand-edited
  // data has a mode bit set without ENABLED, normalize it: the renderer
  // already treats that the same as ENABLED|<mode> (mask_mode ==
  // DEVELOP_MASK_ENABLED is an *exact* equality check gating the uniform
  // path, so this can only ever add a bit that was already implicitly in
  // effect, never change behavior).
  if(ok) n->mask_mode |= DEVELOP_MASK_ENABLED;

  return ok;
}

// a queued dt_masks_migrate_classic_to_flexi() that needs real form
// synthesis and has a real history_num (see dev->pending_flexi_migrations
// in develop.h and this file's header comment for why it is deferred).
typedef struct _pending_flexi_migration_t
{
  dt_iop_module_t *module;
  dt_develop_blend_params_t classic; // bp's original, pre-migration snapshot
  dt_develop_blend_params_t *bp;     // the live params to update once resolved
} _pending_flexi_migration_t;

gboolean dt_masks_migrate_classic_to_flexi(dt_iop_module_t *module,
                                           dt_develop_blend_params_t *bp,
                                           const int history_num)
{
  if(!module) return TRUE;

  // already flexi (edits created under the POC, before the version bump that
  // gates this migration shipped): nothing to do.
  if(bp->mask_mode & DEVELOP_MASK_FLEXI) return TRUE;

  // no classic mask mode set at all: either disabled (nothing to do), or
  // plain uniform ENABLED -- already renders identically to an empty flexi
  // group (see blend.c's "no form defined" fallback fill), so normalize it
  // explicitly rather than leaving a raw classic value in bp. Keeps "every
  // module's mask_mode is DISABLED or a flexi state" a true invariant with
  // no exception, which the mode-select UI relies on.
  if(!(bp->mask_mode
       & (DEVELOP_MASK_MASK | DEVELOP_MASK_CONDITIONAL | DEVELOP_MASK_RASTER)))
  {
    if(bp->mask_mode & DEVELOP_MASK_ENABLED)
    {
      bp->mask_mode = DEVELOP_MASK_ENABLED | DEVELOP_MASK_FLEXI;
      bp->mask_id = NO_MASKID;
    }
    return TRUE;
  }

  // no dev context to synthesize/persist forms into -- e.g.
  // dt_develop_blend_legacy_params_from_so(), used only for converting a
  // built-in preset's blend params blob at module registration time, with no
  // real image and module->dev == NULL. Nothing meaningful to migrate there
  // (built-in presets carry no drawn geometry); stay classic.
  if(!module->dev) return TRUE;

  const dt_develop_blend_params_t o = *bp;

  /* Fail closed on a group flexi cannot render faithfully, before anything is
     written (see _group_has_replace_member()). Only when dev->forms can be
     trusted -- history_num < 0 is exactly the "not inside the darkroom
     history-load loop" condition _mask_id_has_content() documents, where
     dev->forms still holds whatever the previous image left behind and a
     lookup by id could match a stale form. The darkroom path makes the same
     decision later instead, in dt_masks_normalize_flexi_groups(), which is the
     first point where the group's members actually exist.

     Returns TRUE, not FALSE: nothing failed. The migration looked at the mask
     and declined it, exactly as the no-dev-context case above does, and bp
     keeps its classic mask_mode. FALSE means "synthesis broke", which asks the
     caller to fail the whole blend legacy upgrade -- far too big a hammer for
     a mask that renders correctly the way it already is. */
  if(history_num < 0
     && (o.mask_mode & DEVELOP_MASK_MASK)
     && !(o.mask_mode & DEVELOP_MASK_RASTER)
     && _group_has_replace_member(module->dev, o.mask_id, 0))
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[masks] module '%s': mask group %d has a member classic renders as"
             " a replace, which flexi cannot express -- keeping the mask in"
             " classic mode", module->op, o.mask_id);
    return TRUE;
  }

  // drawn-only needs no new form at all (see _dispatch()), so it is always
  // safe to do immediately -- the existing form it reuses is already
  // correctly cumulative in the pre-existing data, whichever row created it.
  const gboolean needs_new_form =
    (o.mask_mode & (DEVELOP_MASK_CONDITIONAL | DEVELOP_MASK_RASTER)) != 0;

  if(needs_new_form && history_num >= 0)
  {
    // defer: dt_masks_finish_flexi_migrations() knows the *final*
    // history_end and runs before dt_masks_read_masks_history(), so the
    // form it writes actually survives being read back.
    _pending_flexi_migration_t *pending = malloc(sizeof(_pending_flexi_migration_t));
    if(!pending)
    {
      _migration_failed(module, "allocation failure");
      // clear the mask rather than leave a raw classic value in bp -- the
      // module's blend_mode/opacity keep working, just uniformly (same
      // invariant Phase 0.5 establishes for every migration outcome).
      bp->mask_mode = DEVELOP_MASK_ENABLED | DEVELOP_MASK_FLEXI;
      bp->mask_id = NO_MASKID;
      return TRUE;
    }
    pending->module = module;
    pending->classic = o;
    pending->bp = bp;
    module->dev->pending_flexi_migrations =
      g_list_append(module->dev->pending_flexi_migrations, pending);

    // mask_id is left untouched (still the pre-migration value) until
    // dt_masks_finish_flexi_migrations() resolves it -- harmless, since
    // nothing renders or otherwise reads bp between now and then, still
    // within the same dt_dev_read_history_ext() call.
    bp->mask_mode = DEVELOP_MASK_ENABLED | DEVELOP_MASK_FLEXI;
    return TRUE;
  }

  return _dispatch(module, &o, bp, history_num);
}

// dt_masks_read_masks_history()'s notion of "current" is not literally
// history_end - 1: its loop only ever visits rows that actually exist in
// main.masks_history, so hist_item_last ends up being whichever *existing*
// row has the highest num below history_end -- which is earlier than
// history_end - 1 whenever the last few history steps did not themselves
// touch masks (nothing re-snapshots dev->forms into a step that never calls
// dt_dev_add_masks_history_item). Writing a synthesized form under a bare
// history_end - 1 that has no prior masks_history rows would silently create
// a *new* highest num, hijacking hist_item_last away from that real
// cumulative snapshot -- so every other module's mask, correctly resolving
// via the old hist_item_last, would stop resolving. Writing under the same
// num that already holds it keeps everything on the one shared snapshot.
static int _current_masks_history_num(const dt_develop_t *dev)
{
  sqlite3_stmt *stmt;
  DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                              "SELECT MAX(num) FROM main.masks_history"
                              " WHERE imgid = ?1 AND num < ?2",
                              -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, dev->image_storage.id);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, dev->history_end);
  int num = dev->history_end - 1; // fallback: no masks data exists yet at all
  if(sqlite3_step(stmt) == SQLITE_ROW && sqlite3_column_type(stmt, 0) != SQLITE_NULL)
    num = sqlite3_column_int(stmt, 0);
  sqlite3_finalize(stmt);
  return num;
}

void dt_masks_finish_flexi_migrations(dt_develop_t *dev)
{
  if(!dev->pending_flexi_migrations) return;

  const int history_num = _current_masks_history_num(dev);

  for(GList *l = dev->pending_flexi_migrations; l; l = g_list_next(l))
  {
    _pending_flexi_migration_t *pending = l->data;

    if(!_dispatch(pending->module, &pending->classic, pending->bp, history_num))
    {
      // allocation failure inside _dispatch(): undo the optimistic
      // mask_mode flip from dt_masks_migrate_classic_to_flexi(), but don't
      // revert all the way back to the classic snapshot -- clear the mask
      // (uniform, blend stays on) and log, same as the other
      // allocation-failure site above, so this never leaves a raw classic
      // value in bp either.
      *pending->bp = pending->classic;
      pending->bp->mask_mode = DEVELOP_MASK_ENABLED | DEVELOP_MASK_FLEXI;
      pending->bp->mask_id = NO_MASKID;
    }
    free(pending);
  }

  g_list_free(dev->pending_flexi_migrations);
  dev->pending_flexi_migrations = NULL;
}

/* Run-boundary normalization for classic drawn groups reused by a migration.

   Must run AFTER dt_masks_read_masks_history(), which is the whole reason this
   is separate from dt_masks_finish_flexi_migrations() (that one runs before it,
   because it writes new forms the read then picks up). This one adjusts groups
   that already exist in the database, so anything it does before the read is
   discarded by it.

   Deliberately writes nothing back. The stored group keeps the classic shape
   list exactly as authored -- which keeps the conversion reversible, matters
   for the classic-restore path, and means a migration never rewrites a user's
   form data. The markers only reach the database if the user edits the image,
   at which point dt_dev_write_history_ext() rewrites masks_history from
   dev->forms wholesale and picks them up. Re-deriving them on every load until
   then costs one pass over the point list, and _split_nonunion_runs() is
   idempotent, so a group that HAS been written back is simply re-marked to the
   same value. */
void dt_masks_normalize_flexi_groups(dt_develop_t *dev)
{
  if(!dev->pending_flexi_group_splits) return;

  for(GList *l = dev->pending_flexi_group_splits; l; l = g_list_next(l))
  {
    dt_masks_pending_split_t *entry = l->data;

    /* This is also the first moment the group's real member list exists on the
       darkroom-load path, so it is where the fail-closed check has to happen
       (see _group_has_replace_member()). Putting the classic params back is the
       whole undo: drawn-only changed nothing but mask_mode, and for the other
       two cases the snapshot restores mask_id/mask_combine/blendif as well.
       Whatever those cases synthesized stays in dev->forms, referenced by
       nothing -- which is exactly what dt_masks_cleanup_unused() already
       prunes, and is why no bespoke teardown is needed here. */
    if(_group_has_replace_member(dev, entry->mask_id, 0))
    {
      *entry->bp = entry->classic;
      dt_print(DT_DEBUG_ALWAYS,
               "[masks] module '%s': mask group %d has a member classic renders"
               " as a replace, which flexi cannot express -- keeping the mask"
               " in classic mode",
               entry->module->op, entry->mask_id);
      continue;
    }

    _split_nonunion_runs(dev, dt_masks_get_from_id(dev, entry->mask_id), 0);
  }

  g_list_free_full(dev->pending_flexi_group_splits, free);
  dev->pending_flexi_group_splits = NULL;
}
