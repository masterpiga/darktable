# Staging masks_revamp for upstream (darktable master)

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

Deleting the classic fold is therefore a **separate, later change** (after the
swap, §5),
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

- **Ship the fixes master already wants, first.** The performance and widget
  work on this branch fixes bugs `master` has today. It is the fastest to
  review, it needs none of the rest, and it buys credibility for the batches
  that follow.
- **Engine before UI, and dormant.** The data model, the flexi fold, and
  `migrate_legacy.c` can all land while `DEVELOP_BLEND_VERSION` stays 14.
  Nothing sets `DEVELOP_MASK_FLEXI`, so rendering is provably unchanged for
  every user; the unit tests and the CLI verbs call the new code directly, so
  it is not untested dead code either.
- **No batch writes code whose only purpose is to be deleted.** The panel host
  is never taught to carry classic; it lands with a placeholder (Batch 6) that
  the swap replaces (Batch 7).
- **Exactly one PR changes mask behaviour.** The version bump, the flexi panel,
  and the deletion of `libs/masks.c` are one atomic change. Splitting them
  produces intermediate merged states where migrated data has no UI that can
  express it — the worst possible thing to hand a nightly user.
- **One logical capability per PR, not one file per PR.**

## 5. PR sequence

Seven batches. 1–2 fix `master`; 3–6 add dormant code and dormant UI; 7 is the
swap and the only batch a user's existing edits notice.

### Batch 1 — performance fixes, already implemented

Section 1 of `masks_revamp_perf_findings.md`. All `[core]`: they fix `master`
today and none of them mentions flexi.

- **1a — U1, `usedetails` flushes nearly the whole pipe on every `synch_all`.**
  Fix is written and on the branch (`pixelpipe_hb.{c,h}`, the
  `synch_no_detail_invalidate` guard). **Read the two porting traps in the
  findings doc before touching it** — the scharr clear and the flush that
  follows are a load-bearing pair, and the decision must be made from
  buffer presence, not from a `want_detail_mask` compare. Runtime neutrality of
  detail masks is still unverified (findings §U1); do that before opening.
- **1b — U3, per-module rendered-mask cache.** CPU path done
  (`piece->drawn_mask_cache`, `blend.c:954`); the OpenCL path is unwritten. Two
  options: hold the PR until `dt_develop_blend_process_cl` is mirrored, or land
  CPU-only and follow up. Prefer mirroring first — a cache that exists on one
  path invites exactly the CPU/OpenCL divergence this tree keeps finding.
- **1c — `imagebuf.c` negative-RoI fast path.** Small and self-contained.

Not in this batch: **U2** (`toneequal` invalidating its downstream tail) is a
`master` bug whose root cause is confirmed but whose fix is unwritten, and it is
not masks work — its own PR, whenever someone writes it. **U4** (on-device mask
compositing) and **U5** (interactive downscaling) are unstarted; see "beyond the
sequence" below.

*Dropped from earlier revisions of this list:* the raster-mask commit
invalidation landed upstream as
[#21519](https://github.com/darktable-org/darktable/pull/21519), and the
full-pipe cache budget was superseded by `cdc84de4d3`.

### Batch 2 — general UI and correctness fixes, already implemented

Already merged, kept here as the batch's ledger:

| PR | Status |
|---|---|
| `exif.cc`: O(n²) → O(n) XMP tag deletion | **merged** |
| `history.c`: include `masks_history` in the edit-history hash ([#21896](https://github.com/darktable-org/darktable/pull/21896)) | **merged** |
| `bauhaus.c`: dragging in the popup's precise-entry mode on GTK3 ([#21894](https://github.com/darktable-org/darktable/pull/21894)) | **merged** |
| Better mask editing near/outside image borders ([#21382](https://github.com/darktable-org/darktable/pull/21382)) | **merged** (caused #21594/#21606, fixed; #21602 was pre-existing) |
| OpenCL publishes a stale raster mask (finding 1) | **merged** |
| JzCzhz hue divergence (finding 2) | **merged** |

Still to extract, all flexi-neutral and independently useful:

- **2a — widget layer:** `dtgtk/gradientslider.{c,h}` marker redesign,
  `dtgtk/paint.{c,h}` new glyphs, bauhaus indicator tweaks. Visible on any
  existing slider, so it carries its own screenshots.
- **2b — `darkroom.c` module-focus restore** and the other small behavioural
  fixes the branch accumulated that have nothing to do with masks.
- **2c — the unrecognised-newer-blend-version fallback.** Today an unknown
  `blendop_version` silently becomes `default_blendop_params` (i.e. the mask
  vanishes) rather than being flagged. It cannot help *this* transition, since
  released versions already behave that way, but it helps the next bump and it
  is the same shape as the rest of this batch.
- **2d — `count-diff-pixels` uint8 underflow** (in the `darktable-tests`
  submodule, so a separate PR there). `np.abs(arr1 - arr2)` runs on the uint8
  arrays PIL returns, so a -1 difference wraps to 255 and every `--threshold`
  comparison reads it as a large one. Harmless at the default threshold of 0,
  which is why nobody noticed, but it makes the option unusable — and it cost a
  full bisect here before being spotted.

### Batch 3 — flexi data model and renderer, dormant

`develop/masks.h`, `masks/{masks,group}.c` + `group_internal.h`, new
`masks/{parametric,raster}.c`, `masks/object.c`, `blend.{c,h}`,
`pixelpipe_hb.{c,h}`, `develop.{c,h}`, `imageop.c`, the new config keys. Ships
with the model-level cmocka suites (`test_flexi_model`, `test_flexi_compose`,
`test_flexi_groups`, `test_flexi_cache`, `test_flexi_persistence`) and
`src/tests/unittests/masks/{flexi_fixture.*,CMakeLists.txt}`. `BUILD_TESTING`
and `libcmocka` are already the project's convention.

Version stays 14; nothing sets the flexi bit; zero rendering change, and the
integration suite proves it. This PR also carries `masks_revamp_data_model.md`,
rewritten as a `dev-doc/` page, since it is what makes the rest reviewable.

### Batch 4 — migration, still not invoked

`masks/migrate_legacy.c` plus `test_flexi_migrate` and the
`src/tests/masking/flexi/` pixel suite. Still not called from
`dt_develop_blend_legacy_params` — only from tests — so it converts nothing yet,
but a reviewer can run `run.sh` and see migrated pixels match classic.

This is the PR whose description carries §3's evidence table, including the
`J5`/`J6`/`J7` result and the stated coverage gaps. Note the ordering
consequence: at this point the campaign's *results* are being cited, but the
tooling that produced them lands in Batch 5. Say so in the description and link
the follow-up, rather than letting a reviewer discover that `--verify-masks`
isn't in the tree they just read.

### Batch 5 — verification tooling (reduced core)

**Decided: upstream the reduced core.** `masks/{harvest,verify,check}.c`, the
`probe_image`/`scratch_image` support they need, and the corresponding
`--harvest-masks` / `--verify-masks` / `--check-masks` verbs in
`common/darktable.c`. These are what let a user with a broken edit hand us a
reproducer, and what lets a future contributor re-run the migration oracle.

Staying on the branch: `masks/{postedit,undo,roundtrip,styleapply}.c` and their
verbs. They are our development harness — they exercise panel controls and undo,
so they will not even be meaningful upstream until Batch 7 lands, and they would
add several thousand lines to a binary that ships to every user for no reader
outside this team.

Two constraints on this batch regardless:

- **`data/masks_corpus.db` does not ship** (§2). The tooling must accept a path
  or fetch, and the PR must work with no corpus present.
- The verbs must be inert in a normal run: no startup cost, no new threads, no
  behaviour change when the flags are absent.

If maintainers would rather have none of it, fall back to citing results only
and keep the whole harness on the branch — but ask in Batch 4's PR, not here.

### Batch 6 — relocation logic, placeholder content

Everything about *where* a mask panel can live, with nothing real in it yet.
Isolates "can this live in a repositionable panel" — the question most likely to
break a window manager we don't run — from "what's in it".

**One PR, not two.** An earlier revision split this into `gui/gtk.c` chrome and
the host lib. That split does not survive contact with the code: `gtk.c` builds
`flexi_header` and `flexi_content` as empty containers explicitly documented as
"where blend_gui.c reparents flexi masks header/content into"
(`gtk.c:144-145`), the resize handle sizes that body, and the slivers and halos
(`_flexi_build_sliver`, `_flexi_build_halo`, `_flexi_slivers_update`) exist to
*reveal* it. Nothing decides the panel is visible except the host, and nothing
is in it except what the host reparents. Landing `gtk.c` alone would ship an
empty container with a hot edge onto nothing and a handle that resizes nothing
— untestable, and a reviewer would rightly ask what it is for.

So Batch 6 is:

- **`gui/gtk.{c,h}`** — the `DT_UI_PANEL_FLEXI` placement (`gtk.h:396`), the
  centerrow docking (`_flexi_dock_reorder`, `gtk.c:3760`), hot edges, the resize
  handle and its `show_panel_handle` preference, the "cannot grow further"
  toast, and the overlay suppression during mask editing. Note that
  `DT_UI_PANEL_FLEXI` is deliberately *not* one of the collapsible column panels
  (`gtk.c:3573`); it is an overlay on the canvas, which is what makes the
  slivers necessary.
- **`develop/masks_gui_panel_host.c` + `libs/masks_flexi_host.c` +
  `libs/CMakeLists.txt`** — the host lib and the reparenting machinery, the
  positions (embedded/utility/left/right), the
  `plugins/darkroom/blend/masks_panel_position` preference and its menu
  (`_add_masks_panel_position_box`), the darkroom toolbar toggle
  (`dt_iop_gui_blend_masks_panel_toggle` /
  `dt_iop_gui_blend_masks_panel_sync_toolbox`, `masks_gui_panel_host.c:139,150`)
  and the collapsed-corner affordance.
- **the dummy content** the host reparents into `flexi_header`/`flexi_content`,
  standing in for what `blend_gui.c` will supply in Batch 7.
- **`test_flexi_panel`**, whose model-level helpers (`_model_masks_panel_state`,
  `_model_masks_corner_icon_tooltip`) are pure functions and test fine against a
  placeholder.

`libs/masks.c` is untouched; classic keeps its own UI exactly as it is.

Split it into *commits* along those lines if it helps review, but not into PRs.

**Flag-guarded: it does not reach users.** With the flag off — the default, and
the only state a user ever sees — the lib registers but never shows: no toolbar
button, no position preference in the UI, no panel. With it on, the panel
appears with dummy content and every position, hot edge and resize behaviour is
exercisable. Consistent with Batches 3–5, which also ship dormant.

Make it a **runtime** guard (a `dt_conf_get_bool` on an undocumented key read
where the lib decides visibility), not a CMake option: a maintainer can then
exercise the panel on a normal build without a rebuild, `test_flexi_panel` runs
either way, and there is no build-system change to unpick later. Because the key
is deliberately undocumented, it gets **no** entry in
`data/darktableconfig.xml.in` — say so in the PR description, since that is
otherwise the reviewer's first objection.

Batch 7 deletes the flag, the guard and the dummy content together with the
placeholder.

**What Batch 6 does not prove.** The reparenting code lands here, but its real
client — `bd->relocatable_box`, `bd->masks_blend_header`, `bd->masks_panel_body`,
all built by the new `blend_gui.c` — only arrives in Batch 7
(`_masks_flexi_release_full()`, `masks_gui_panel_host.c:696-753`). A placeholder
reparents cleanly in a way a live module UI with focus changes, expanders and
DnD may not. Two further behaviours are simply inert against dummy content: the
halo poll stands down while on-canvas mask editing is armed
(`_flexi_proximity_poll` / `_flexi_shape_highlighted`, `gtk.c:36,1805`), and
`flexi_mask_active`/`flexi_mask_label` describe a mask that does not exist yet.
Treat 6 as evidence about window chrome and panel mechanics, not about hosting a
live mask UI, and expect fixes to the host in 7.

### Batch 7 — flexi replaces classic

`develop/blend_gui.c` (+17.7k) and `blend_gui_internal.h`, the real content of
the Batch 6 host (replacing the placeholder), `masks_gui_presets.c`,
`data/themes/darktable.css` and the two chunk themes, the
`DEVELOP_BLEND_VERSION` 14 → 15 bump that turns migration on, the deletion of
`libs/masks.c`, the remaining panel-behaviour suites (`test_flexi_dnd`,
`test_flexi_controls`, `test_flexi_ui_coverage`, `test_flexi_raster_prune`),
the forced pre-migration backup (see below), `dev-doc/flexi_masks/` (trimmed)
and the `RELEASE_NOTES.md` entry.

One PR because every part of it is load-bearing for the others, and because any
decomposition ships a half-swapped state: migrated data with no UI that can
express it. It is large; the mitigation is that Batches 1–6 have already removed
everything separable from it — the engine, the migration, the tooling and now
all of the panel chrome — and that `blend_gui.c`'s new code is covered by
headless tests a reviewer can run.

#### Migration mitigation strategy

**A pre-migration backup, on the DB-snapshot precedent.**
darktable already takes *mandatory version-upgrade snapshots* of `library.db`
regardless of the `database/create_snapshot` preference
(`database.c:5177,5225`, and the preference's own longdescription says so).
A blend-version bump is not a schema upgrade, so it does not currently
trigger one — force it for the 14 → 15 bump, and alongside it copy each
image's existing sidecar to `<file>.xmp.pre-flexi` before the first
write-back. Restoring the sidecar (or the snapshot) and opening with an old
darktable then works exactly as it did. No schema change, no format risk,
nothing permanent, and it is a pattern maintainers already accept. Its one
real gap: `write_sidecar_files` can be "never", which is why the forced DB
snapshot is the load-bearing half and the XMP copy the convenience.

### Beyond the sequence

Not part of upstreaming, tracked so they don't get lost:

- On-device (OpenCL) mask compositing and feather (perf findings U4) — mask
  rendering is CPU-only even on the OpenCL pipe (C3). Large effort; the
  group-fold *operators* are flexi-specific but the underlying limitation is
  `master`'s.
- Interactive downscaling during slider drag (U5).
- The branch-only items in §2 of the findings doc (N1–N3).
- Widen corpus coverage on the strata §3 names as thin.
- **Retire the classic fold** (`group.c`'s sequential branch, `blend.c`'s
  classic mask branches, the `mode_raster` guards once confirmed vestigial).
  Functionally subsumed already; blocked on deciding what becomes the
  verification suite's oracle, since `verify.c` and the pixel suite currently
  diff migrated *against* classic. Options: freeze reference renders produced by
  the last build that had both, or keep the classic fold compiled only under
  `BUILD_TESTING`.

## 6. Verification per PR

- **Batch 1:** `src/tests/integration/run` on Linux (**not** macOS, where deltae
  never runs) — these change the pipe, so pixel neutrality is the whole claim —
  plus `-d perf` / `-d pipe` before-and-after numbers in the description, and
  for U3 an explicit CPU-vs-OpenCL comparison of a plain drawn mask. Also the
  per-shape-details repro named in the findings doc (set a details threshold,
  move a handle, watch for "detail mask blending error").
- **Batch 2:** existing suites plus the manual smoke each fix implies — slider
  redraw for 2a, focus behaviour for 2b.
- **Batch 3/4:** `ctest -R flexi`, `src/tests/masking/flexi/run.sh` +
  `verify_effect.sh`, `src/tests/integration/run` on Linux, and a full
  `--check-masks` pass over the corpus (run locally; the corpus does not ship).
  For Batch 3 the load-bearing claim is *no rendering change at all*, so the
  integration suite matters more than the flexi tests.
- **Batch 5:** each verb exercised from a clean checkout with no corpus present,
  and a run of the full binary with no `--*-masks` flag to show it is inert.
- **Batch 6:** `ctest -R flexi_panel`; with the flag on, a manual walkthrough of
  every panel position and of the resize handle, on all three platforms if
  possible — this is the batch most likely to break a window manager we don't
  run; and with the flag off, confirmation that the UI is byte-for-byte the
  darkroom users have today. State explicitly that the walkthrough ran against
  dummy content, not a live module UI.
- **Batch 7:** all of the above, plus a manual walkthrough per capability the
  panel offers *in every position*, plus at least one real library migrated and
  edited in anger by someone who is not us.

State in each PR description what was run and what was not, and disclose AI
assistance.
