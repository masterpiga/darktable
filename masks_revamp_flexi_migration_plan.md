# Migrating classic masks to flexi

A version-bump migration that rewrites every pre-flexi drawn, parametric, and
raster mask into the flexi group representation — with an exhaustive case
table, a fail-closed safety rule, and a validation pass against the existing
masking integration tests.

**Status:** implemented — `src/develop/masks/migrate_legacy.c`
**Hook:** `DEVELOP_BLEND_VERSION` 14 → 15 (done; the macro is at 15)
**Scope:** Phase A (this doc) + Phase B outline

> **This document describes shipped code.** The case table in §3 was written
> ahead of the implementation and has been corrected against it — where the
> two disagreed, the code won, because in each case it had worked out a
> subtlety the plan had not. Those corrections are called out inline. Keep
> this file and `migrate_legacy.c` in step: the code's own comments carry the
> algebra, this file carries the map.

---

## 1. Why this exists

Flexi masks already render every classic drawn shape through the identical
code path used by `DEVELOP_MASK_MASK` — the two modes share one boolean gate,
`mode_drawn`, and one storage layer, the `mask_id` → `DT_MASKS_GROUP` form
tree. Parametric (`DEVELOP_MASK_CONDITIONAL`) and raster
(`DEVELOP_MASK_RASTER`) masks, though, live as scalar fields directly on
`blend_params` — outside the form tree entirely — and nothing converted them
into flexi's `DT_MASKS_PARAMETRIC` / `DT_MASKS_RASTER` form types.

The goal is to close that gap completely: open any pre-existing edit, in any
legacy mode, and have it render through flexi with the same pixels, before
the legacy mode toggle buttons and rendering branches are deleted in a later
pass. That is what `migrate_legacy.c` now does.

## 2. Where migration hooks in

Flexi originally shipped without bumping `DEVELOP_BLEND_VERSION` — it reused a
free bit (`1<<4`) on the existing `mask_mode` field, so the layout never
changed. That meant the fast path in `dt_dev_read_history_ext` saw every stored
classic edit as version-current and `memcpy`d it straight through —
`dt_develop_blend_legacy_params()` never ran for them at all.

> **Decision (done).** `DEVELOP_BLEND_VERSION` is 15, with an `old_version ==
> 14` step in `dt_develop_blend_legacy_params()` (`src/develop/blend.c`). This
> is the only hook that fires unconditionally for every already-persisted edit,
> and it's covered for free at all three places that call this function the
> same way: darkroom history load, style application (`styles.c`), and preset
> application (`presets.c`).

> **Correction to the above.** The migration is *not* attached to the
> `old_version == 14` branch specifically. Every version branch converts to the
> current layout, so `dt_develop_blend_legacy_params_ext()` calls
> `dt_masks_migrate_classic_to_flexi()` once at the tail, after whichever
> branch ran — a v9 edit is migrated as surely as a v14 one. See that
> function's own comment for why, and for why a migration failure is
> deliberately *not* propagated as a legacy-params failure (the generic
> fallback is `default_blendop_params`, which would silently drop a still-valid
> classic mask).

> **Required adjacent fix (done).** The `old_version == 13` branch hardcoded
> the literal `14` instead of the version macro, and would have gone dead the
> instant `DEVELOP_BLEND_VERSION` became 15, silently orphaning every v13 edit.
> Every branch now tests `new_version == DEVELOP_BLEND_VERSION`.

## 3. The case table

Every reachable value of `mask_mode & (ENABLED | MASK | CONDITIONAL |
RASTER)` — sixteen bit combinations, all sixteen accounted for. Cases 6 and 7
are unreachable through any current or historical GUI path, but stored /
foreign data must still degrade correctly.

| # | Mode | Output | Recipe |
|---|---|---|---|
| 0 | `DISABLED` | unchanged | No masking configured. Nothing to migrate. |
| 1 | `ENABLED` only (uniform) | `ENABLED\|FLEXI` | Opacity-only blend, no form lookup at all. **Corrected:** the plan said to leave this as plain `ENABLED`, reasoning that `FLEXI` would be a bit with nothing to point at. The code sets `ENABLED\|FLEXI` with `mask_id = NO_MASKID` instead, so that "every module's `mask_mode` is `DISABLED` or a flexi state" is an invariant with no exceptions — which the mode-select UI relies on. Flexi's own no-form fallback renders this identically. |
| 2 | `MASK`, valid group | `ENABLED\|FLEXI` | **Zero transform.** `mask_id` reused verbatim — the group's existing per-shape combine bits are already valid flexi operators by construction. |
| 2a/2b | `MASK`, invalid/non-group `mask_id` | `ENABLED\|FLEXI` | Defensive. No form is fabricated — the flexi "no form" fallback matches the classic one exactly since both read the same `mode_drawn` gate. Logged, not treated as an error. |
| 3 | `CONDITIONAL` | `ENABLED\|FLEXI`, or uniform | **Synthesize, but only when there is something to synthesize.** `_classify_conditional()` sorts the blendif into three branches first — see the sub-table below. Only `DT_COND_REAL` builds forms. **Corrected on two counts:** (a) the plan said an unconfigured (all-default) blendif is still synthesized "not optimized away"; the code collapses it instead, because it is provably not a no-op-shaped mask but *no mask at all* — see below. (b) The plan said one `single = 0` multi-channel form; the code builds **one single-channel form per active channel** (`p->single = 1`), joined within their own run by `DT_MASKS_STATE_WITHIN_MULTIPLY`, so each channel is separately visible and editable in the panel. |
| 4 | `MASK_CONDITIONAL` | `ENABLED\|FLEXI`, or drawn-only, or uniform | **Synthesize.** Drawn *and* parametric, combined by multiplication in the classic renderer: a wrapper group holds the existing drawn group (untouched) with the synthesized channel run joined onto it by `DT_MASKS_STATE_MULTIPLY` — the between-group operator built to mirror exactly this legacy combine. **Corrected:** the plan's one-line "if `mask_id` is invalid this collapses to case 3 unless `INCL`" understates it. The real decision tree is in the sub-table below; in particular a *passthrough* parametric collapses to a plain drawn-only migration with `MASKS_POS` and `INV` folded into a single drawn invert, and `INCL` is handled by XORing two invert decisions rather than by a separate code path. |
| 5 | `RASTER` | `ENABLED\|FLEXI` | **Synthesize.** One group holding one `DT_MASKS_RASTER` element (`source`, `instance`, `id` copied from the classic `raster_mask_*` fields). Inversion moves onto the point's own `DT_MASKS_STATE_INVERSE` bit, since the new struct has no dedicated invert field. |
| 6 | `RASTER` + `MASK`/`CONDITIONAL` | `ENABLED\|FLEXI` | Unreachable via GUI. Migrates as pure case 5 — the classic renderer is an `if / else if` chain where raster wins outright, so faithfully reproducing that means the other bits and their data are dropped, not merged. Logged as an anomaly when encountered. |
| 7 | any mode bit, `ENABLED` clear | as matching case, `\|ENABLED` forced | Unreachable from the GUI (every mode button writes `ENABLED` and its mode bit together), but the renderer already treats it as equivalent to the same bits with `ENABLED` set. Migrated per its matching case above, with `ENABLED` forced on the output. |
| 8 | `FLEXI` already set | pass-through | Edits created under the current POC, before the version bump ships. Guarded at the very top of the migration function — returns immediately, untouched. |

### 3.1 Classifying a parametric mask (cases 3 and 4)

A classic parametric config does not always describe a mask. `_classify_conditional(blend_cst, blendif, incl)`
sorts it into three branches, and only one of them has geometry to migrate:

| Branch | When | Migrates to |
|---|---|---|
| `DT_COND_REAL` | at least one channel of the colorspace is active, and no channel cancels | one single-channel `DT_MASKS_PARAMETRIC` form per active channel, folded together by `WITHIN_MULTIPLY` |
| `DT_COND_CONSTANT` | a channel's polarity bits cancel it out (`INCL` can flip an inactive channel into this state) | **no form.** A plain uniform blend: opacity untouched if `INCL != INV`, forced to `0.0` otherwise |
| `DT_COND_PASSTHROUGH` | no channel is active at all, or the colorspace has no canceling mechanism (RAW) | **no form.** Case 3: uniform blend as above. Case 4: drawn-only migration, `mask_id` reused verbatim, with `MASKS_POS ^ INV` folded into a single drawn invert |

**Why collapsing is right, and the plan was wrong.** The plan's instinct was
that synthesizing an all-default form is harmless because it "renders as a
provable no-op". That is true of a form whose *range* covers everything — but
these two branches are not that. In classic, they never reach the per-channel
curve at all: `DT_COND_CONSTANT` replaces the whole mask buffer wholesale via
`dt_iop_image_fill()`, and `DT_COND_PASSTHROUGH` passes the incoming value
through untouched. Synthesizing a form for either would fabricate a mask where
classic had none, and in the `INCL != INV` case would render *inverted* against
the original. Opacity `0.0` reproduces "contributes nothing" exactly, because
opacity multiplies the mask everywhere in the blend math.

Case 4 adds one decision ahead of this, when `mask_id` resolves to no content:
if `MASKS_POS == INCL` the drawn fallback fill and `INCL`'s channel-polarity
flip cancel out and it delegates straight to case 3; if they differ, everything
collapses to a hard constant regardless of the parametric config.

`DT_COND_REAL` with `INCL` set is reachable only when every channel of the
colorspace is simultaneously active. It needs no new operator or nesting: the
same multiply-fold construction applies with `incl` XORed into both invert
decisions (`invert_drawn`, `invert_composite`) plus a pre-flip of the
synthesized form's own polarity bits. The algebra behind that — and behind the
`MASKS_POS ^ INV` fold above — is verified in comments at each branch in
`_migrate_drawn_and_parametric()`; it is not re-derived here.

## 4. Fail closed, not silent

The generic `legacy_params` failure convention in this codebase falls back to
`default_blendop_params` — for every other kind of params blob, that's
reasonable, since there's nothing better to fall back to. For a masking
migration, it's the wrong default: it would silently delete the user's mask.

> **Rule.** On a synthesis failure — allocation, or a combination the flexi
> model cannot cleanly reproduce — the migration leaves `mask_mode` in its
> original classic value. Because Phase A keeps the classic rendering code
> alive, "stay classic" is a fully functional fallback. The failure is logged
> via `dt_control_log` and `dt_print(DT_DEBUG_ALWAYS, …)`, and the generic
> default-params fallback is deliberately bypassed: `dt_develop_blend_legacy_params_ext()`
> returns success with the untouched classic data rather than propagating a
> migration failure as a layout failure.

> **The one exception.** If allocating the *deferral record* fails (the
> `history_num >= 0` path, where synthesis is postponed to
> `dt_masks_finish_flexi_migrations()`), the code cannot stay classic — the
> version has already moved and there is nothing left to carry the classic
> config forward. It clears the mask instead: `ENABLED|FLEXI` with
> `mask_id = NO_MASKID`, so blend mode and opacity keep working uniformly.
> That is a lost mask, not a preserved one, and it is the deliberate lesser
> evil against a half-migrated edit.

> **Idempotency.** Guaranteed structurally, not by a runtime flag — but not
> for the reason first written here. The plan claimed the step "only ever
> fires for `old_version == 14` data"; it does not, because the migration runs
> at the tail of *every* version branch (see §2). What actually guarantees
> idempotency is case 8's guard: the migration returns immediately if `FLEXI`
> is already set, and every path that migrates sets it. An already-migrated
> edit therefore cannot be migrated twice, whatever version it arrives as.

## 5. The code, and where it hooks in

`src/develop/masks/migrate_legacy.c` is kept separate from the already-large
`masks.c` and from `blend.c` so it can be reasoned about — and eventually
retired — on its own. It exports `dt_masks_migrate_classic_to_flexi()`,
dispatching into one static helper per case family.

**Persistence works differently from the plan.** The plan expected a shared
`_migrate_persist_form()` writing each synthesized form into
`main.masks_history` at the creating row's `num`, following `spots.c`'s
`legacy_params`. That does not survive a reload. Every existing
`masks_history` row is a *whole cumulative snapshot* of `dev->forms` as it
stood at that step, and `dt_masks_read_masks_history()` only ever reads the
current step — so a form written under the row that created it vanishes on the
next load unless that row happens to be the last one.

So on the darkroom-load path (a real `history_num`) synthesis is **deferred**:
the classic params are queued on `dev->pending_flexi_migrations`, and
`dt_masks_finish_flexi_migrations()` — which knows the final `history_end` and
runs before `dt_masks_read_masks_history()` — creates the forms and writes them
under `history_end - 1`. Paths with no history context (`history_num < 0`, e.g.
style and preset application) synthesize inline into `dev->forms` instead.

**Raster pipe-ordering** was the one piece of Phase-A work the plan flagged as
not optional, and it is handled — though not by the proposed
`dt_masks_tree_uses_raster()` helper. `_reconcile_raster_form_users()` in
`imageop.c` walks each module's tree at commit time and registers a dependency
per `DT_MASKS_RASTER` element, independently of `mask_mode & DEVELOP_MASK_RASTER`
and of `blend_params.raster_mask_*` (which stay reserved for the exclusive
whole-mask RASTER mode). See the header comment in `masks/raster.c`.

**Files:**

- `src/develop/masks/migrate_legacy.c` — the migration
- `src/develop/blend.c` / `blend.h` — `DEVELOP_BLEND_VERSION` 15, the tail call
- `src/develop/develop.c` / `develop.h` — `pending_flexi_migrations`, load ordering
- `src/develop/imageop.c` — `_reconcile_raster_form_users()`
- `src/develop/masks.h` — prototypes
- `src/develop/masks/{raster,parametric}.c` — the form types synthesized

## 6. Validating against the existing test corpus

The `src/tests/integration/` submodule already carries classic-mode masking
fixtures for every combine shape darktable supports — and critically, none
of them use flexi yet, so they're clean baselines. Its own comparison is
perceptual (CIEDE2000 delta-E ≤ 2.3), which is the right bar for a rendering
change but not for this one: migration claims pixel-identical output, so
validation uses the suite's `count-diff-pixels` tool directly, threshold
zero, comparing each fixture rendered unmigrated against the same fixture
rendered after conversion.

| Fixture | Mode exercised | Covers |
|---|---|---|
| `0004-masks` | `MASK` | Grouped circle / path / gradient / ellipse / brush |
| `0081-mask-groups` | `MASK` | Grouped drawn shapes, case 2 |
| `0144-masks-combine-sum` | `MASK` | Sum combine mode |
| `0034-blending-modes-parametric` | `CONDITIONAL` | Pure blendif across blend modes, case 3 |
| `0116`–`0125-blending-*` | `CONDITIONAL` | Parametric in RGB display / scene / Lab colorspaces (10 fixtures) |
| `0087` / `0088-blendif-*` | `MASK` | Blendif AND/OR, difference/exclusive combine |
| `0025` / `0156-exposure-guided-filter*` | `MASK_CONDITIONAL` | Drawn + parametric combined, case 4 |
| `0090-mask-combine` | mixed | Multi-module combine, four modules |
| `0091-mask-combine-intersection-inverted` | `MASK` | Inverted intersection combine |
| `0167-raster-mask` | `RASTER` | bilat sourced from colorbalancergb's mask, case 5 |

**Two suites now exist, and they cover different things.**

`src/tests/masking/flexi/` is the pixel suite this section describes: hand-packed
classic XMPs rendered through `darktable-cli`, compared exactly against
checked-in expectations. It is the only thing that can prove migration is
*pixel-identical*, and it is where the case-4 algebra was actually validated.
See its own README and `masks_revamp_flexi_migration_benchmark.md` for the
investigation history behind the matrix.

`src/tests/unittests/masks/test_flexi_migrate.c` covers the case table
*structurally* — the resulting `mask_mode` and the forms synthesized — with no
rendering at all. It reaches what the pixel suite cannot: all sixteen bit
combinations are swept, including the ones no GUI can produce but a stored or
foreign XMP can (case 6's RASTER+MASK/CONDITIONAL, case 7's cleared ENABLED
bit), plus the degenerate-parametric collapses of §3.1, a dangling `mask_id`,
and idempotency. It runs headless in under a second.

The division is deliberate: pixels for "does it render the same", structure for
"does every reachable input land somewhere valid". Neither subsumes the other.

**Still uncovered.** A forced-allocation-failure test for the fail-closed rule.
The deferred path (`history_num >= 0`, `dt_masks_finish_flexi_migrations()`) —
the unit tests drive the inline path only, and the pixel suite exercises the
deferred one but cannot assert on its intermediate state. Style and preset
application, which route through the same function but not through
`dt_dev_read_history_ext`.

**Run commands:**

```sh
# the flexi migration pixel suite
cd src/tests/masking/flexi && ./run.sh

# re-validate against a from-scratch pre-migration build
./run.sh --pristine <path-to-pristine-darktable-cli>

# the structural suite
ctest --test-dir build-test -R flexi_migrate --output-on-failure

# the pre-existing integration corpus
cd src/tests/integration && ./run            # full suite
./run 0167-raster-mask                       # one test
./run --op=bilat                             # everything blending a module
```

The before/after comparison the plan called for does exist: `run.sh --pristine`
renders each fixture twice — once with a build that has the migration stashed
out, once with the current binary — and diffs them directly, which is exactly
the two-renderer shape none of the integration suite's own tooling assumes.

---

## Phase B — outline only

Once migration is proven against the full corpus above plus a read-only dry
run over a real library: the four retired mode toggle buttons and their
callbacks in `blend_gui.c`, the non-flexi rendering branches in `blend.c`,
and the `raster_mask_*` scalar fields on `blend_params` themselves, behind a
further version bump. (Raster dependency registration already runs off the
tree-walk — see §5 — so that precondition is met.)

> **Sequencing note.** Phase B is *not* the next step. Classic has to be
> restored first, because that is what upstream `master` ships — see
> `masks_revamp_classic_restore_plan.md` — and the multi-release rollout that
> eventually reaches Phase B is in `masks_revamp_transition_plan.md`. In
> particular, migration currently runs unconditionally on every blend-params
> upgrade; restoring classic means gating it, or the restored classic UI opens
> every existing edit and finds nothing classic left to edit.

`dt_masks_migrate_classic_to_flexi()` is not on this list. Like every other
step in `dt_develop_blend_legacy_params`, it's the permanent bridge for old
files — not legacy code waiting to be deleted alongside the UI it replaces.
