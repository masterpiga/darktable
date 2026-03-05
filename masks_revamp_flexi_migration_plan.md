# Migrating classic masks to flexi

A version-bump migration that rewrites every pre-flexi drawn, parametric, and
raster mask into the flexi group representation — with an exhaustive case
table, a fail-closed safety rule, and a validation pass against the existing
masking integration tests.

**Status:** proposed, unimplemented
**Hook:** `DEVELOP_BLEND_VERSION` 14 → 15
**Scope:** Phase A (this doc) + Phase B outline

---

## 1. Why this exists

Flexi masks already render every classic drawn shape through the identical
code path used by `DEVELOP_MASK_MASK` — the two modes share one boolean gate,
`mode_drawn`, and one storage layer, the `mask_id` → `DT_MASKS_GROUP` form
tree. Parametric (`DEVELOP_MASK_CONDITIONAL`) and raster
(`DEVELOP_MASK_RASTER`) masks, though, still live as scalar fields directly on
`blend_params` — outside the form tree entirely — with no code anywhere that
converts them into flexi's `DT_MASKS_PARAMETRIC` / `DT_MASKS_RASTER` form
types.

The goal is to close that gap completely: open any pre-existing edit, in any
legacy mode, and have it render through flexi with the same pixels, before
the legacy mode toggle buttons and rendering branches are deleted in a later
pass.

## 2. Where migration hooks in

Flexi shipped without bumping `DEVELOP_BLEND_VERSION` — it reused a free bit
(`1<<4`) on the existing `mask_mode` field, so the layout never changed. That
means the fast path in `dt_dev_read_history_ext` (`develop.c:2649`) currently
sees every stored classic edit as version-current and `memcpy`s it straight
through — `dt_develop_blend_legacy_params()` never runs for them at all.

> **Decision.** Bump `DEVELOP_BLEND_VERSION` from 14 to 15, and add the
> conversion as the `old_version == 14` step inside
> `dt_develop_blend_legacy_params()` (`src/develop/blend.c`). This is the only
> hook that fires unconditionally for every already-persisted edit, and it's
> covered for free at all three places that already call this function the
> same way: darkroom history load, style application (`styles.c:711`), and
> preset application (`presets.c:1097`).

> **Required adjacent fix.** The current `old_version == 13 && new_version ==
> 14` branch at `blend.c:2451` hardcodes the literal `14` instead of the
> version macro. Left alone, that branch goes dead the instant
> `DEVELOP_BLEND_VERSION` becomes 15 — silently orphaning every v13 edit. Must
> become `new_version == DEVELOP_BLEND_VERSION` as part of this change, not
> after.

## 3. The case table

Every reachable value of `mask_mode & (ENABLED | MASK | CONDITIONAL |
RASTER)` — sixteen bit combinations, all sixteen accounted for. Cases 6 and 7
are unreachable through any current or historical GUI path, but stored /
foreign data must still degrade correctly.

| # | Mode | Output | Recipe |
|---|---|---|---|
| 0 | `DISABLED` | unchanged | No masking configured. Nothing to migrate. |
| 1 | `ENABLED` only (uniform) | unchanged | Opacity-only blend, no form lookup at all. Left as plain `ENABLED` — setting `FLEXI` here would add a bit with nothing for it to point at. |
| 2 | `MASK`, valid group | `ENABLED\|FLEXI` | **Zero transform.** `mask_id` reused verbatim — the group's existing per-shape combine bits are already valid flexi operators by construction. |
| 2a/2b | `MASK`, invalid/non-group `mask_id` | `ENABLED\|FLEXI` | Defensive. No form is fabricated — the flexi "no form" fallback matches the classic one exactly since both read the same `mode_drawn` gate. Logged, not treated as an error. |
| 3 | `CONDITIONAL` | `ENABLED\|FLEXI` | **Synthesize.** One group holding one `DT_MASKS_PARAMETRIC` element — `single = 0` (legacy full multi-channel form), with `blendif` / `blendif_parameters` / `blendif_boost_factors` copied verbatim. An unconfigured (all-default) blendif is still synthesized, not optimized away — it renders as a provable no-op and keeps the logic simpler to verify. |
| 4 | `MASK_CONDITIONAL` | `ENABLED\|FLEXI` | **Synthesize.** Drawn *and* parametric, combined by multiplication in the classic renderer. A new wrapper group stacks the existing drawn group (untouched) beneath a new parametric element joined by `DT_MASKS_STATE_MULTIPLY` — the operator built and commented specifically to mirror this legacy combine. If `mask_id` is invalid, this collapses to case 3, unless `DEVELOP_COMBINE_INCL` was set — a rare, deliberately separate sub-case that migrates to a degenerate always-masked-out group. |
| 5 | `RASTER` | `ENABLED\|FLEXI` | **Synthesize.** One group holding one `DT_MASKS_RASTER` element (`source`, `instance`, `id` copied from the classic `raster_mask_*` fields). Inversion moves onto the point's own `DT_MASKS_STATE_INVERSE` bit, since the new struct has no dedicated invert field. |
| 6 | `RASTER` + `MASK`/`CONDITIONAL` | `ENABLED\|FLEXI` | Unreachable via GUI. Migrates as pure case 5 — the classic renderer is an `if / else if` chain where raster wins outright, so faithfully reproducing that means the other bits and their data are dropped, not merged. Logged as an anomaly when encountered. |
| 7 | any mode bit, `ENABLED` clear | as matching case, `\|ENABLED` forced | Unreachable from the GUI (every mode button writes `ENABLED` and its mode bit together), but the renderer already treats it as equivalent to the same bits with `ENABLED` set. Migrated per its matching case above, with `ENABLED` forced on the output. |
| 8 | `FLEXI` already set | pass-through | Edits created under the current POC, before the version bump ships. Guarded at the very top of the migration function — returns immediately, untouched. |

## 4. Fail closed, not silent

The generic `legacy_params` failure convention in this codebase falls back to
`default_blendop_params` — for every other kind of params blob, that's
reasonable, since there's nothing better to fall back to. For a masking
migration, it's the wrong default: it would silently delete the user's mask.

> **Rule.** On any synthesis failure — allocation, DB write, anything — the
> migration leaves `mask_mode` in its original classic value and returns
> success, not failure. Because Phase A keeps the classic rendering code
> alive, "stay classic" is a fully functional fallback. The failure is
> logged via `dt_control_log` and `dt_print(DT_DEBUG_ALWAYS, …)`, and the
> generic default-params fallback is deliberately bypassed for this specific
> failure path.

> **Idempotency.** Guaranteed structurally, not by a runtime flag: the step
> only ever fires for `old_version == 14` data, and every writer of v15 data
> — new modules and the migration itself — always stamps
> `DEVELOP_BLEND_VERSION == 15`. An edit can't re-enter the branch once
> migrated. Case 8's guard handles the one exception: pre-bump flexi edits
> nominally still at version 14.

## 5. New code, exact integration points

One new file, `src/develop/masks/migrate_legacy.c`, kept separate from the
already-large `masks.c` and from `blend.c` so it can be reasoned about — and
eventually retired — on its own. It exports
`dt_masks_migrate_classic_to_flexi()`, dispatching into one static helper per
case family, plus a shared `_migrate_persist_form()` that writes each
synthesized form into `main.masks_history` at the correct history `num`,
following the pattern already proven by `spots.c`'s own `legacy_params`
(which does exactly this: create a form, write it to history, rewire
`blend_params->mask_id`).

One piece of Phase-A work is not optional: readers that gate raster-mask
pipe-order dependency registration on `mask_mode & DEVELOP_MASK_RASTER`
specifically must be extended to also fire when a `DT_MASKS_RASTER` form
exists anywhere in the module's flexi tree — otherwise a migrated raster
mask keeps rendering correctly by coincidence while silently losing its
pipe-ordering guarantee. A new helper, `dt_masks_tree_uses_raster()`, covers
this.

**Files touched:**

- `src/develop/blend.c` — edit (version bump step, v13→14 literal fix)
- `src/develop/blend.h` — edit (`DEVELOP_BLEND_VERSION`)
- `src/develop/masks.h` — edit (new prototypes)
- `src/develop/masks/migrate_legacy.c` — **new**
- `src/develop/masks/raster.c` — reference
- `src/develop/masks/parametric.c` — reference
- `src/iop/spots.c` — pattern to follow
- `src/develop/develop.c` — reference (history load ordering)
- `src/common/styles.c` — reference (style-apply path)
- `src/gui/presets.c` — reference (preset-apply path)

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

**Gaps the existing suite doesn't cover.** Hand-crafted fixtures needed for
the defensive cases: RASTER combined with MASK (case 6), the
ENABLED-bit-clear variant (case 7), a corrupted `mask_id` (case 2a/2b), and
the `DEVELOP_COMBINE_INCL` degenerate branch (case 4, incl. sub-case). A
forced-allocation-failure test confirms the fail-closed rule holds. Style
and preset application paths get one fixture each, since they route through
the same function but not through `dt_dev_read_history_ext`.

**Run commands:**

```sh
# single test
./run 0167-raster-mask

# every test attaching a blend op to a given module
./run --op=bilat

# full suite
./run
```

No script exists yet for before/after comparison of the *same* edit under
two renderers — the plan is a small wrapper reusing `count-diff-pixels`
against two `darktable-cli` invocations, since none of the suite's tooling
assumes that shape of comparison today.

---

## Phase B — outline only

Once migration is proven against the full corpus above plus a read-only dry
run over a real library: the four retired mode toggle buttons and their
callbacks in `blend_gui.c`, the non-flexi rendering branches in `blend.c`,
and — once raster dependency registration is switched to the tree-walk added
in Phase A — the `raster_mask_*` scalar fields on `blend_params` themselves,
behind a further version bump.

`dt_masks_migrate_classic_to_flexi()` is not on this list. Like every other
step in `dt_develop_blend_legacy_params`, it's the permanent bridge for old
files — not legacy code waiting to be deleted alongside the UI it replaces.
