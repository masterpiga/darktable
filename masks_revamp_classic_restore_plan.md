# Restoring classic masks, and isolating flexi from it

The branch went all-in on flexi deliberately — to have a clear picture of the
end state, and to keep flexi code from entangling itself with classic. Both
goals are served. But master ships classic, so classic has to come back before
any of this merges, and it has to come back in a form where flexi changes
*cannot* reach it by accident.

**Status:** proposed
**Companion docs:** `masks_revamp_upstream_plan.md` (PR sequence),
`masks_revamp_transition_plan.md` (multi-release rollout),
`masks_revamp_flexi_migration_plan.md` (data conversion)

---

## 1. What classic actually lost

Measured against `master`, not estimated. The good news is that the damage is
confined to the GUI layer and falls into four cohesive clusters.

### 1.1 The engine is intact

`blend.c`'s classic rendering paths were **not** removed. `mode_parametric`
and `raster` are untouched; `mode_drawn` was only widened:

```c
const gboolean mode_drawn = mask_mode & (DEVELOP_MASK_MASK | DEVELOP_MASK_FLEXI);
```

Every classic `mask_mode` bit still exists with its original value, and
`DEVELOP_MASK_FLEXI` is an additive `1 << 4`. `develop.h`'s `proxy.masks`
struct is untouched. **No engine restoration work is needed** — this is the
single biggest thing in our favour and it is why the rest is tractable.

### 1.2 One deleted file

`src/libs/masks.c` — the classic mask manager lib — was deleted and its
`CMakeLists.txt` entry replaced with `masks_flexi_host`. A clean
`git checkout master -- src/libs/masks.c` revert, plus restoring the CMake
entry *alongside* rather than instead of the flexi host.

*Risk to check, not assume:* `masks.h` reordered the `dt_masks_state_t`
declarations (values unchanged) and added new form types. Master's
`libs/masks.c` should compile against it unmodified, but that needs proving,
not asserting.

### 1.3 Seven fields removed from `dt_iop_gui_blend_data_t`

```
GList *masks_modes;  GList *masks_modes_toggles;  GtkWidget *selected_mask_mode;
GtkWidget *colorpicker;  GtkWidget *colorpicker_set_values;
gboolean output_channels_shown;  GtkWidget *channel_boost_factor_slider;
```

`colorpicker_set_values` did not vanish — it moved from `bd->` into flexi's
per-parametric-editor struct (`blend_gui.c:909`). That is the pattern
throughout: classic's single-panel state became flexi's per-element state.

### 1.4 Thirty-nine functions removed from `blend_gui.c`

`blend_gui.c` went 3,716 → 16,698 lines. The functions present on master and
absent on the branch group cleanly into three subsystems:

| Cluster | Count | Functions |
|---|---|---|
| **Classic mode buttons** | 7 | `_blendop_masks_modes_{none_clicked,uni,drawn,param,both,raster}_toggled`, `_blendop_masks_modes_toggle` |
| **Classic blendif editor** | 22 | `dt_iop_gui_{init,update}_blendif`, `_blendop_blendif_{tab_switch,update_tab,sliders_callback,polarity_callback,boost_factor_callback,feathering_callback,details_callback,invert,reset,…}`, `_blendif_{scale,show/hide/clean_output_channels,are_output_channels_used,colorpicker_cst}`, `_get_boost_factor`, `_update_gradient_slider_pickers` |
| **Classic raster combo** | 5 | `dt_iop_gui_{init,update}_raster`, `_raster_{combo_populate,polarity_callback,value_changed_callback}` |

Three subsystems, one file, one struct. That is the whole restoration surface.

### 1.5 Migration is unconditional

`dt_masks_migrate_classic_to_flexi()` runs as the tail of *every* successful
blend-params upgrade. With classic restored, that has to become conditional —
otherwise the restored classic UI opens every existing edit and finds nothing
classic left to edit.

---

## 2. The isolation architecture

The restoration approach and the isolation requirement are the same question,
because *how* classic comes back determines whether flexi can leak into it.

### 2.1 Restore classic by reverting the file, not by re-adding the code

**Recommendation: `src/develop/blend_gui.c` returns to master byte-for-byte,
and all flexi code moves into new translation units.**

The alternative — re-adding the 39 functions into the current 16.7k-line file
— leaves classic and flexi sharing one file, one set of file-statics, and one
review surface. That is precisely the percolation risk, institutionalised.

Reverting instead buys three things at once:

- **Zero classic regression, by construction.** Not "we tested it" — the
  classic diff against upstream is *empty*. A reviewer confirms it with
  `git diff master -- src/develop/blend_gui.c`.
- **A trivially reviewable PR.** Classic contributes nothing to the diff, so
  100% of reviewer attention goes to flexi, which is where it is needed.
- **Isolation the compiler enforces.** Classic's TU does not `#include
  "blend_gui_internal.h"`, so it *cannot* name a flexi symbol. Discipline is
  not required; the build fails.

**The cost, stated honestly:** any genuine bug fix the branch made to shared
classic code is discarded by the revert. That is the right outcome, not a
regrettable one — a real classic fix belongs upstream on its own merits, as
its own small PR, exactly like Batch 1's `exif.cc` and `history.c` fixes. It
should not ride into master inside a 17k-line masks feature.

*Triage step:* before reverting, diff the ~34 shared statics between the two
versions and sort each change into "flexi needed this" (goes to flexi's copy)
or "this is a classic bug fix" (goes to a Batch 1 PR). This is the one part of
the restoration that requires judgement rather than mechanics.

### 2.2 File layout

```
blend_gui.c            reverted to master; classic panel + shared entry points.
                       The ONLY branch change: the dispatch hook (§2.3).
blend_gui_internal.h   the flexi-side seam. Already exists; keep it small.
blend_gui_flexi*.c     all flexi code. Never included by blend_gui.c.
blend_gui_blendif.c    (new) the pure blendif helpers both sides use — §2.4.
```

### 2.3 One dispatch point, not many

Classic and flexi share the public entry points (`dt_iop_gui_init_blending`,
`dt_iop_gui_update_blending`, `dt_iop_gui_cleanup_blending`,
`dt_iop_gui_blending_lose_focus`, `dt_iop_gui_blending_reload_defaults`).
Each gets a single early branch on the editor mode, delegating to a flexi
function declared in `blend_gui_internal.h`. Five small, identical-shaped,
individually reviewable insertions — and they are the complete allowlist of
what may differ from master in that file.

### 2.4 Share the pure, duplicate the stateful

~34 file-static helpers exist in both versions, and about half are blendif
related. They are not all the same kind of thing, and the sharing decision
should follow that:

- **Pure functions — share.** `_blendif_cook`, `_blendif_scale_print_*`,
  `_blendif_print_digits_*`, `_blendif_scale`, `_log10_scale_callback`,
  `_magnifier_scale_callback`. Stateless math and formatting over blendif
  values. A change to these *should* change both — that is not percolation,
  it is correctness. Move them to `blend_gui_blendif.c` with a narrow header.
- **State-touching functions — duplicate.** `_blendop_masks_mode_callback`,
  `_blendif_change_blend_colorspace`, `_blendop_blendif_showmask_clicked`, and
  every widget callback. These read and write `bd`, history, and the pipe.
  Flexi keeps its own copies. The duplication is the point: it is what makes
  the two panels independently changeable.

The line is mechanical enough to state as a rule: **if it takes a `GtkWidget*`
or a `dt_iop_module_t*`, it does not get shared.**

### 2.5 Make cross-use visible in the struct

Split `dt_iop_gui_blend_data_t`'s panel state:

```c
struct { GList *masks_modes; GList *masks_modes_toggles; GtkWidget *selected_mask_mode;
         GtkWidget *colorpicker; /* … */ } classic;
struct { /* flexi panel state */ } flexi;
/* genuinely shared fields stay at top level */
```

This does not prevent anything — it makes every crossing legible at the point
of use and in a diff. `bd->flexi.something` appearing in classic's file is
obvious to a reviewer in a way that `bd->something` never is.

### 2.6 The guard that actually works

A CI check asserting that `git diff master -- src/develop/blend_gui.c`
contains nothing outside the §2.3 dispatch allowlist. Cheap to write, and it
is the only mechanism here that catches the mistake *at the moment it is
made* rather than at review. Everything else in this section reduces the
probability; this one closes the loop.

---

## 3. Migration becomes conditional

With classic restored, migration must stop being automatic. Per the transition
plan's Stage 1, it becomes explicit and per-module — the user converts a mask
when they choose to.

Mechanically:
- Gate the `dt_masks_migrate_classic_to_flexi()` call in
  `dt_develop_blend_legacy_params_ext()` on an explicit request rather than
  running it for every upgrade.
- Hold `DEVELOP_BLEND_VERSION` at 14 for edits that were never converted, so
  a user who never touches flexi keeps writing blobs older darktable can read.
- Keep the fail-closed rule exactly as it is — it becomes *more* valuable once
  classic is a live fallback rather than a theoretical one.

The `old_version == 14` branch in `blend.c` and all of `migrate_legacy.c` stay
as they are. Only the trigger changes.

---

## 4. Sequence

1. **Triage the shared statics** (§2.1) — sort branch changes into
   flexi-needed vs classic-fix. Judgement work; everything else waits on it.
2. **Extract flexi out of `blend_gui.c`** into `blend_gui_flexi*.c`, leaving
   the file at master's content plus the five dispatch hooks.
3. **Restore `libs/masks.c`** and its CMake entry alongside the flexi host;
   confirm it compiles unmodified against the new `masks.h`.
4. **Restore the seven `dt_iop_gui_blend_data_t` fields**, in a `classic`
   sub-struct (§2.5).
5. **Restore the three function clusters** (§1.4) — verbatim from master.
6. **Gate migration** (§3).
7. **Add the CI guard** (§2.6).
8. **Verify:** classic diff against master is empty outside the allowlist;
   the flexi suite still passes; a manual pass over classic's mode buttons,
   blendif editor, and raster combo confirms upstream behaviour.
9. **Split out any classic bug fixes** found in step 1 as their own Batch 1
   PRs against upstream master.

Steps 2 and 5 are the bulk of the work and are largely mechanical. Step 1 is
the only one that can produce surprises, which is why it goes first.
