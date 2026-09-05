# Staging masks_revamp for upstream (darktable master)

**Revised 2026-09-05.** The previous revision assumed classic and flexi would
coexist for several releases behind a mode button. That is no longer the plan,
and most of its PR sequence described a tree that no longer exists.

## The decision this plan is built on

**Flexi replaces classic. There is no coexistence stage, no mode button, no
opt-in preference, no "convert this mask" action.**

What made a staged transition look necessary was uncertainty about whether
migration preserves existing edits. That uncertainty has been retired by
measurement (§3): the branch now carries a migration verification campaign over
14 contributed libraries and 61,332 real edits with zero failures, plus 261
headless unit tests and a 44-scenario pixel suite. A migration we can show is
correct does not need a user-facing escape hatch — it needs the panel swapped
and the old one deleted, which is what the branch already does.

**What is *not* being deleted**, and must not be confused with the panel swap:

- **`migrate_legacy.c`.** Like every other step in
  `dt_develop_blend_legacy_params`, it is the permanent bridge for old files.
- **Classic *storage*.** No schema change: groups are inferred runs over the
  existing flat `points` list, `_MULTIPLY`/`_SCREEN`/`GROUP_BREAK` reuse
  previously-unused `state` bits, and `group_start` is a dedicated appended
  field carried by the masks-format version bump (v10). See
  `masks_revamp_data_model.md`.
- **The classic renderer** — but *not* because migration needs it as a
  fallback. See below; this is a deliberate deferral, not a dependency.

So "swap the panel" is precisely: delete `libs/masks.c`, land the flexi panel,
and bump `DEVELOP_BLEND_VERSION` 14 → 15 so every edit migrates on load.

### Is the classic renderer subsumed?

For the drawn/parametric fold: **yes, functionally.**

- Migration has **no semantic refusal left**. The one class that used to fail
  closed — a classic "replace"-position member, an operator-less member sitting
  above others, which classic renders by discarding everything below it — is now
  *repaired* rather than refused: `_repair_base_case_overwrite()`
  (`migrate_legacy.c:199`) disables exactly the members classic discarded, so
  both folds render what classic always rendered. `_group_has_replace_member()`
  no longer exists. Every remaining `return FALSE` in the file is
  `_migration_failed(module, "allocation failure")` plus one invalid-`mask_id`
  guard, i.e. OOM.
- **Nothing in a migrated tree can reach the classic fold.**
  `dt_masks_group_get_mask_roi()` (`group.c:1405`) dispatches on the *module's*
  `blend_params`, and every recursive call site passes `piece` through, so
  nested subgroups inside a migrated tree take the flexi fold too.
- **Classic has no future readers.** Merely opening or exporting an image
  persists `mask_mode = FLEXI` (`migrate_legacy.c:1264`, the #21905 note), so
  the conversion is one-way per file from the first load.

Three reasons it still stays in the tree for now, in order of weight:

1. **It is the verification suite's oracle.** `verify.c` renders classic and
   migrated in the same binary and diffs them (`max_diff` = "CPU: classic vs
   migrated", `verify.c:1034`); `--check-masks`, `--postedit-masks` and
   `src/tests/masking/flexi` all rest on that comparison. Delete the classic
   fold and §3's entire evidence base stops being re-runnable — including by a
   future contributor investigating a migration bug we have not seen. This is
   the argument that actually decides it.
2. **Cheap retreat.** With the fold present, a field defect is fixed by a guard
   that keeps one configuration classic. Without it, it is a revert.
3. **`DEVELOP_MASK_RASTER` still has live guards** — `mode_raster`
   (`blend_gui.c:1098`), `show_flexi_ui = !mode_raster` (1111), and the classic
   raster branch in `blend.c`. Migration converts raster to a `DT_MASKS_RASTER`
   form (`migrate_legacy.c:1061`) and nothing sets the classic raster bit
   afterwards, so these look vestigial — but that needs verifying, not
   assuming, before anything is removed.

Deleting the classic fold is therefore a **separate, later change** (Batch 5),
whose first job is to say what replaces the oracle. It is not a prerequisite
for, or a casualty of, the panel swap.

## 1. What is actually on the branch today

83 commits, 251 files, ~52k insertions over `master` (docs and test fixtures
included). The code footprint:

| Area | Size | Note |
|---|---:|---|
| `develop/blend_gui.c` | +17,678 | the panel; the bulk of the diff |
| `develop/masks/{verify,harvest,postedit,persist,undo,roundtrip,styleapply,probe_image,scratch_image,check}.c` | ~9,500 | migration verification tooling, shipped behind CLI flags |
| `gui/gtk.c` / `gtk.h` | +1,563 / +45 | panel placement, hot edges, resize handle, growth toast |
| `develop/masks_gui_panel_host.c` | +1,290 | panel hosting |
| `develop/masks/migrate_legacy.c` | +1,379 | classic → flexi conversion |
| `develop/masks/group.c` | +942 | flexi fold (`_group_get_mask_roi_flexi`) |
| `develop/blend.{c,h}` | +692 / +654 | flexi branches, model |
| `common/darktable.c` | +538 | the `--*-masks` CLI verbs |
| `develop/masks_gui_presets.c` | +584 | |
| `develop/masks/{object,parametric,raster}.c` | +985 | object/parametric/raster as forms |
| `libs/masks.c` | **−2,513** | classic mask manager, **deleted** |
| `libs/masks_flexi_host.c` | +310 | its replacement |
| `data/themes/darktable.css` | +1,102 | |
| `src/tests/unittests/masks/**` | ~6,300 | 261 cmocka tests, 12 suites |
| `src/tests/masking/flexi/**` | 44 XMPs + refs | pixel suite |

Versions: `DEVELOP_BLEND_VERSION` 15, `DEVELOP_MASKS_VERSION` 10.
New config keys: `plugins/darkroom/masks/{default_operator,solo_edit_mode,
auto_expand_selected,show_panel_handle}` and
`plugins/darkroom/blend/preview_channel_on_hover`.

## 2. Things on the branch that must **not** reach a PR

Decide these before opening anything, because they are what makes a diff look
unreviewable:

- **The 15 tracked working documents in the repo root**
  (`masks_revamp_*.md`, `branch_analysis_*.md`,
  `classic_opencl_blend_findings.md`, …) plus
  `masks_revamp_migration_ledger.json` (17k lines),
  `migration_failures.json*`, `classic_opencl_outliers.*`. These are our
  engineering record, not darktable's. Keep them on the branch; exclude them
  from every PR. Anything a reviewer genuinely needs becomes a
  `dev-doc/` page written for them.
- **`data/masks_corpus.db` (19 MB)** and **`dev-doc/flexi_masks/demo.mp4`
  (4.4 MB)** plus ~20 PNG screenshots. A 19 MB binary in `data/` ships to every
  user; it cannot go upstream as-is. Host the corpus out of tree and have the
  tooling fetch or accept a path. Trim the doc media hard.
- **`.github/workflows/*` fork gating** (`|| github.repository ==
  'masterpiga/darktable'`) — branch convenience, drop from the PR.
- **`darktable.flexi_test_mode`** (`common/darktable.c:1270`) — the
  default-TRUE library isolation safeguard for branch testers. It is
  deliberately there and stays there *on the branch*; it does not go upstream,
  where the feature is the shipped default and the real library is the point.

## 3. The evidence that licenses the swap

This is the argument the PR description leads with, not an appendix.

| Layer | What it proves | Where |
|---|---|---|
| 261 cmocka tests, 12 suites | model, grouping, DnD, selection, cache-hash contract, persistence, migration case table, operator algebra — **structural only, no pixels** | `src/tests/unittests/masks/` |
| 44-scenario pixel suite | migrated render == classic render on the export pipe; plus two route-equivalence controls (K1/K1C, K2/K2C) that hold without a pristine build | `src/tests/masking/flexi/run.sh` |
| Corpus campaign | 14 libraries, 61,332 edits, 7,932 distinct configuration *shapes*, **0 failures** → failure rate < 0.038% (1 in 2,648) at 95% confidence | `masks_revamp_migration_confidence.md` |
| `--persist-masks` | a save/reload between two edits does not change the mask | `masks/persist.c` |
| `--postedit-masks` | a migrated group edited through every panel control matches the from-scratch equivalent — the only check that looks *past* migration | `masks/postedit.c` |
| `--undo-masks`, `--verify-masks`, `--check-masks` | undo/geometry, targeted replay of known failures, per-corpus harvest | `masks/{undo,verify,check}.c` |

Two real migration bugs were found by this campaign and fixed
(`masks_revamp_migration_failures.md`); both were invisible to all 261
structural tests. Three classic-blending defects were found alongside it, two
already merged upstream, one measured and deliberately not shipped
(`upstreamed_rendering_fixes_from_flexi_migration.md`).

One result of the suite worth carrying into the PR description, because a
reviewer will otherwise find it and draw the wrong conclusion: harvesting the
fixture's own sidecars and running the migration oracle over them
(`--harvest-masks-xmp xmps/ F && --verify-masks F`) reports **3 DIFFERENT out of
49** — `J5`, `J6`, `J7`. Those three encode `DT_MASKS_REFINE_GROUP`, a flexi-only
refinement scope, into pre-flexi blobs. The whole per-shape refinement block is
this branch's masks v7 (upstream master is still at v6, with no refinement field
at all), so no released darktable ever wrote scope=2 and no real classic edit can
carry it; the classic fold reads the field as a plain bool
(`if(fpt->refinement.enabled)`, `masks/group.c:1470`) and so applies it per
element, while the flexi fold applies it once per group. That is the two folds
disagreeing about an input classic cannot produce, not a migration defect. Every
classic-authorable scenario in the fixture verifies identical, and
`CLASSIC CHANGED: 0` confirms the fail-closed path.

Known gaps to state plainly rather than paper over: coverage is thin for
`inv|*` combine flags, `drawn|parametric|raster`, and non-clone circle forms
(§"Coverage gaps" of the confidence doc); contributors, not edits, are the real
sampling unit and there are 14 of them; and
`src/tests/integration` reports OK on macOS without ever running deltae, so it
is not pixel coverage on this machine.

## 4. Staging principles

- **Foundation before feature.** Generic infra and widget fixes that don't
  depend on flexi land first, in small PRs. They are the fastest to review and
  they shrink the swap.
- **Engine before UI, and dormant.** The data model, the flexi fold, and
  `migrate_legacy.c` can all land while `DEVELOP_BLEND_VERSION` stays 14.
  Nothing sets `DEVELOP_MASK_FLEXI`, so rendering is provably unchanged for
  every user; the unit tests and the CLI verbs call the new code directly, so
  it is not untested dead code either.
- **Exactly one PR changes behaviour.** The version bump, the panel, and the
  deletion of `libs/masks.c` are one atomic change. Splitting them produces
  intermediate merged states where migrated data has no UI that can express it
  — the worst possible thing to hand a nightly user.
- **One logical capability per PR, not one file per PR.**

## 5. PR sequence

### Batch 1 — independent fixes (largely done)

| PR | Status |
|---|---|
| `exif.cc`: O(n²) → O(n) XMP tag deletion | **merged** |
| `history.c`: include `masks_history` in the edit-history hash ([#21896](https://github.com/darktable-org/darktable/pull/21896)) | **merged** |
| `bauhaus.c`: dragging in the popup's precise-entry mode on GTK3 ([#21894](https://github.com/darktable-org/darktable/pull/21894)) | **merged** |
| Better mask editing near/outside image borders ([#21382](https://github.com/darktable-org/darktable/pull/21382)) | **merged** (caused #21594/#21606, fixed; #21602 was pre-existing) |
| OpenCL publishes a stale raster mask (finding 1) | **merged** |
| JzCzhz hue divergence (finding 2) | **merged** |

Still to extract, all flexi-neutral and independently useful:

- **1a — widget layer:** `dtgtk/gradientslider.{c,h}` marker redesign,
  `dtgtk/paint.{c,h}` new glyphs, bauhaus indicator tweaks. Visible on any
  existing slider.
- **1b — pipeline caching fixes marked ⬆ YES in
  `masks_revamp_perf_findings.md`:** A14 (`usedetails` order-0 flush on every
  `synch_all`), A12 (full-pipe cache budget `mipmap_memory/4 → /2`), A9
  (`dt_iop_commit_blend_params` reporting a new raster on every commit), A13
  (per-module rasterized drawn-mask cache, CPU path), and
  `imagebuf.c`'s negative-RoI fast-path fix. These fix master today.
- **1c — `darkroom.c` module-focus restore** and other small behavioural fixes
  the branch accumulated that have nothing to do with masks.
- **1e — `count-diff-pixels` uint8 underflow** (in the `darktable-tests`
  submodule, so a separate PR there). `np.abs(arr1 - arr2)` runs on the uint8
  arrays PIL returns, so a -1 difference wraps to 255 and every `--threshold`
  comparison reads it as a large one. Harmless at the default threshold of 0,
  which is why nobody noticed, but it makes the option unusable — and it cost a
  full bisect here before being spotted.

- **1d — the unrecognised-newer-blend-version fallback.** Today an unknown
  `blendop_version` silently becomes `default_blendop_params` (i.e. the mask
  vanishes) rather than being flagged. It cannot help *this* transition, since
  released versions already behave that way, but it is exactly the shape of the
  rest of this batch and it helps the next bump.

### Batch 2 — panel-hosting GUI infrastructure

**PR 2a — `gui/gtk.{c,h}`:** panel placement (embedded/utility/left/right),
hot edges, the resize handle and its `show_panel_handle` preference, the
"cannot grow further" toast, and the overlay suppression during mask editing.
This is ~1.6k lines of generic GTK work in the app's own chrome; it is
reviewable on its own terms and demonstrable without a single mask.

**PR 2b — `develop/masks_gui_panel_host.c` + `libs/masks_flexi_host.c` +
`libs/CMakeLists.txt`:** the lib that hosts a repositionable panel, with stub
content. Isolates "can this live in a repositionable panel" from "what's in
it". `libs/masks.c` is untouched here.

### Batch 3 — engine and migration, dormant

**PR 3a — the model and the renderer.** `develop/masks.h`,
`masks/{masks,group}.c` + `group_internal.h`, new
`masks/{parametric,raster}.c`, `masks/object.c`, `blend.{c,h}`,
`pixelpipe_hb.{c,h}`, `develop.{c,h}`, `imageop.c`, the new config keys. Ships
with the model-level cmocka suites (`test_flexi_model`, `test_flexi_compose`,
`test_flexi_groups`, `test_flexi_cache`, `test_flexi_persistence`) and
`src/tests/unittests/masks/{flexi_fixture.*,CMakeLists.txt}`. `BUILD_TESTING`
and `libcmocka` are already the project's convention. Version stays 14; nothing
sets the flexi bit; zero rendering change, and the integration suite proves it.

**PR 3b — migration.** `masks/migrate_legacy.c` plus `test_flexi_migrate` and
the `src/tests/masking/flexi/` pixel suite. Still not invoked from
`dt_develop_blend_legacy_params` — only from tests and the CLI verbs — so it
converts nothing yet, but a reviewer can run `run.sh` and see migrated pixels
match classic. This is the PR whose description carries §3's evidence table.

**PR 3c — the verification tooling.** `masks/{harvest,verify,check,persist,
postedit,undo,roundtrip,styleapply,probe_image,scratch_image}.c` and the
`--*-masks` verbs in `common/darktable.c`: ~9.5k lines that ship in the normal
binary. **This one needs a maintainer decision before it is written**, and the
question should be asked in PR 3b rather than sprung as a diff:

1. upstream it whole, as the reproducibility story for the migration
   (contributors can re-run the campaign);
2. upstream a reduced core (`harvest` + `verify` + `check`) and keep the
   post-edit/undo/style harnesses on the branch;
3. keep all of it out of tree, and cite results only.

Recommendation: **(2)**. `--harvest-masks` and `--verify-masks` are what let a
user with a broken edit hand us a reproducer; the rest is our development
harness. Whatever the answer, the corpus DB does not ship (§2).

### Batch 4 — the swap

**PR 4 — flexi replaces classic.** `develop/blend_gui.c` (+17.7k) and
`blend_gui_internal.h`, the real content of the PR 2b host,
`masks_gui_presets.c`, `data/themes/darktable.css` and the two chunk themes,
the `DEVELOP_BLEND_VERSION` 14 → 15 bump that turns migration on, the deletion
of `libs/masks.c`, the remaining panel-behaviour suites (`test_flexi_panel`,
`test_flexi_dnd`, `test_flexi_controls`, `test_flexi_ui_coverage`,
`test_flexi_raster_prune`), `dev-doc/flexi_masks/` (trimmed) and the
`RELEASE_NOTES.md` entry.

This is one PR because every part of it is load-bearing for the others, and
because any decomposition ships a half-swapped state. It is large; the
mitigation is that Batches 1–3 have already removed everything separable from
it, and that `blend_gui.c`'s new code is covered by headless tests a reviewer
can run.

**If maintainers refuse a PR this size**, the fallback is an in-tree overlap —
*not* a user-facing one: land the panel over several PRs with flexi reachable
only via an undocumented preference and `libs/masks.c` still present, then a
final small PR that flips the default, bumps the version and deletes classic.
Users never see two mask UIs; only the tree carries both, and only for the
weeks the review takes. Offer this only if asked.

### Batch 5 — after the swap

Not part of upstreaming, tracked so they don't get lost:

- OpenCL path for the per-module rendered-mask cache (perf findings D4).
- On-device mask compositing/feather (D5) — mask rendering is CPU-only even on
  the OpenCL pipe (B4).
- Widen corpus coverage on the strata §3 names as thin.
- **Retire the classic fold** (`group.c`'s sequential branch, `blend.c`'s
  classic mask branches, the `mode_raster` guards once confirmed vestigial).
  Functionally subsumed already; blocked on deciding what becomes the
  verification suite's oracle, since `verify.c` and the pixel suite currently
  diff migrated *against* classic. Options: freeze reference renders produced by
  the last build that had both, or keep the classic fold compiled only under
  `BUILD_TESTING`.

## 6. Verification per PR

- **Batch 1:** existing suites plus the manual smoke each fix implies (slider
  redraw for 1a; `-d perf` / `-d pipe` before-and-after for the caching fixes
  in 1b).
- **Batch 2:** manual GUI walkthrough of every panel position and of the resize
  handle, on all three platforms if possible — this is the batch most likely to
  break a window manager we don't run.
- **Batch 3a/3b:** `ctest -R flexi` (261 tests), `src/tests/masking/flexi/run.sh`
  + `verify_effect.sh`, `src/tests/integration/run` on Linux (**not** macOS, where
  deltae never runs), and a full `--check-masks` pass over the corpus.
- **Batch 4:** all of the above, plus a manual walkthrough per capability the
  panel offers, plus at least one real library migrated and edited in anger by
  someone who is not us.

State in each PR description what was run and what was not, and disclose AI
assistance.
