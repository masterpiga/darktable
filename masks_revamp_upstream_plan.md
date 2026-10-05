# Flexi masks: upstreaming plan

The series lives on `upstream-flexi-model`, in the worktree
`darktable-worktrees/upstream-flexi`, on top of master `61dea294be`. Every
commit builds with `USE_AI` on and off, and `ctest -R "flexi|probe"` passes
(13/13) on the last one. Draft PR descriptions for each commit, with
reproductions for the bug fixes, are in `masks_revamp_pr_descriptions.md`
(local, not tracked).

## What changes

The mask manager is replaced. A module's mask becomes a tree of groups: each
group folds its elements with one operator (maximum, screen, sum, minimum,
product, difference or exclusion) and has its own opacity, inversion and
refinement. Elements can be drawn shapes, parametric (blendif) channels,
raster masks or AI objects, so one panel expresses all of them.

Existing edits are converted when they are loaded. There is no coexistence
mode and no opt-in: `libs/masks.c` is deleted, and the new panel lives in the
module's blending section, in a utility module or in a panel over the
darkroom canvas.

Against master:

- **Masks format v6 → v7.** The group point gains per-shape refinement, a group
  name, a group opacity and a preset note, appended to the struct (16 → 240
  bytes). The v6 → v7 step sets the group opacity to 1.0; zero-fill is neutral
  for the rest. New form types: `DT_MASKS_PARAMETRIC`, `DT_MASKS_RASTER`. New
  `state` bits for the group operators and modes, all previously unused.
- **Blend params v14 → v15.** Same layout. The bump makes every older edit go
  through `dt_develop_blend_legacy_params_ext`, where the migration runs. The
  mask lock takes over a reserved field.
- **Migration** (`masks/migrate_legacy.c`). Converts the classic flat list into
  groups that render the same pixels, parametric and raster masks into
  elements. It cannot fail except on allocation. Where classic ignored part of
  a mask (members below an operator-less "replace" member, a duplicate
  reference that renders nothing), the migrated mask drops it, so the panel
  shows what renders. A bottom shape set to intersection or difference, which
  classic combines with an empty mask and so renders as nothing, becomes a
  union at 0% opacity, and a drawn mask with nothing to draw (an empty group,
  or groups of empty groups), which classic renders as an empty mask, becomes
  a plain blend at 0% opacity. Conversion is one-way: the edit is written back
  as flexi.
- **Renderer.** A second fold (`_group_get_mask_roi_flexi`, `group.c`)
  selected by `DEVELOP_MASK_FLEXI` on the module. The classic fold stays: it is
  the reference the verification tools and the pixel suite compare against.
- **UI.** The panel (`blend_gui.c`), its hosts and canvas placement (`gtk.c`,
  `masks_gui_panel_host.c`, `libs/masks_flexi_host.c`), a darkroom toolbar
  toggle (`masks_gui_toolbar.c`), slider and glyph changes in `dtgtk/` and
  `bauhaus/`, CSS, and eleven preferences under `plugins/darkroom/masks/` and
  `plugins/darkroom/blend/`. The built-in group presets and their notes are a
  data file, `data/masks_group_presets.json`. A bash build step
  (`tools/generate_masks_presets_strings.sh`) extracts their strings for
  translation into a header that only `xgettext` reads (`po/POTFILES.in`), as
  `tools/generate_styles_string.sh` does for styles. The panel uses event
  controllers throughout, except raw `scroll-event` on GTK3, as master does.
- **Tools.** `--harvest-masks`, `--harvest-masks-xmp` and `--verify-masks` let
  a user hand over a reproducer for a mask that migrated wrong, and let anyone
  re-run the migration check. `--check-masks`, `--roundtrip-masks`,
  `--styleapply-masks`, `--persist-masks`, `--undo-masks` and `--lock-masks`
  run a harvest through the database trip, style application, panel edits,
  undo and the lock. They do nothing unless their flag is given.
- **Docs.** `dev-doc/masks_data_model.md` (the data model) and
  `dev-doc/flexi_masks/styling.md` (the panel's styling contract for themes).

## Commit sequence

Ten commits. Commits 1–5 and 9 apply to master on their own, so they can go
as separate PRs ahead of the rest. Commits 6–8 change nothing a user sees:
`DEVELOP_BLEND_VERSION` stays 14, the migration is not called, and nothing
sets `DEVELOP_MASK_FLEXI`, so every mask renders through the classic fold as
today. Commit 10 switches everything on at once, because any split would ship
migrated edits with no panel that can show them.

| # | commit | subject | code (`src/`, `data/`) | tests |
|---|--------|---------|---:|---:|
| 1 | `00a9b76f49` | develop: do not dereference freed modules in the chroma cache | +28 / −2 | |
| 2 | `30f038fc0c` | ras2vect: give vectorized path points a nonzero feather | +4 / −1 | |
| 3 | `184896c18a` | masks: draw shapes on the canvas with a contrasting edge | +56 / −25 | |
| 4 | `7ef0494817` | masks: cancel the shape being drawn with Escape | +33 | |
| 5 | `3ec16167c4` | develop: move history compress and truncate out of the history module | +53 / −45 | |
| 6 | `4de17deac4` | masks: add the masks v7 format for flexi masks | +213 / −59 | |
| 7 | `d74c08d5a1` | masks: add the flexi mask engine | +5.0k / −0.2k | |
| 8 | `c417960bd8` | masks: add tools to verify the classic to flexi migration | +3.7k | +0.8k |
| 9 | `c1ef9597be` | bauhaus, dtgtk: widget support for the flexi masks panel | +0.9k / −0.2k | |
| 10 | `dc041b452f` | masks: replace the classic masks with flexi masks | +30.9k / −5.6k | +17.2k |

In all: about **+40.9k / −6.2k** lines of code, +18k of tests (cmocka suites
and the 46-scenario pixel suite in `src/tests/masking/flexi/`), and 0.4k of
docs and tools. The panel in `blend_gui.c` is the largest part, at
+16.3k / −2.1k; the CSS adds 1.9k and `gtk.c` 1.6k.

### Bug fixes and behavior changes

1. **Chroma cache.** Leaving the darkroom freed the modules that
   `dev->chroma` points at; the next darkroom entry dereferenced them in
   `dt_dev_reset_chroma()`. A use-after-free, usually silent.
2. **Vectorized paths.** Paths from the AI object tool and from rasterfile's
   "vectorize" have a zero feather, which pins the mask manager's feather
   slider at 0.
3. **Canvas outlines.** The dark outline is black at full overlay contrast
   and vanishes over a dark picture.

Inside larger commits:

- 7: `dt_iop_copy_image_roi()` read before its input for a negative RoI
  offset (crop's `distort_mask()` in a zoomed view). Also on the standalone
  branch `fix_copy_image_roi_bounds`, so it can go separately
- 9: slider popups get room past both ends of the bar, so a click there
  sets the minimum or maximum instead of closing the popup
- 10: the expander no longer scrolls a focused module back to its header
  when its height changes. Trade-off: a module growing past the bottom of
  the panel is no longer scrolled into view either

### What commit 10 holds

- the switch: `dt_develop_blend_legacy_params_ext` runs the migration and
  `DEVELOP_BLEND_VERSION` goes to 15; `libs/masks.c` is deleted
- the panel, its hosts and toolbar toggle, presets (JSON plus the
  string-extraction step), CSS, preferences, shape-tool changes
- the canvas following the panel: hover and selection mirrored both ways,
  solo and solo edit, AI objects moving as one unit
- the mask lock: locked masks survive reset, presets, styles and paste
- the expander change above
- the panel's unit suites, the pixel suite, and the CLI checks other than
  harvest and verify

## Evidence

From the `masks_revamp` branch; not re-run on the upstream series.

| Check | Result |
|---|---|
| 13 cmocka suites | structure, grouping, operators, drag and drop, cache hash, persistence, migration cases, panel styling; no pixels. Pass on the upstream series |
| pixel suite against a stock master build | 38/46 bit-identical; the other 8 use v7-only fields (per-shape refinement), which master cannot read |
| pixel suite against its reference images | 46/46 |
| corpus: 14 contributors, 63,157 edits, 8,203 distinct configuration shapes | 0 migration failures: failure rate below 0.037% (1 in 2,738) at 95% confidence |

The corpus check compares migrated renders against the classic fold in the
same binary, which follows master's rules. The bound treats each distinct
configuration shape as an independent sample; the shapes come from 14
contributors. The harvested corpus stays out of tree, and the tools take a
path to it; `masks_revamp` carries it as `data/masks_corpus.db`, which is not
in the upstream series.

It depends on one fix to master, which has landed (a87e42fc82):
`dt_gradient_lookup()` extrapolated below zero for a negative table index, so
a gradient mask applied its module in reverse along its outer edge.

**A master bug this exposes.** Master's classic fold never clears its output,
and `dt_develop_blend_process` allocates the mask without initializing it. A
top-level group whose bottom shape uses intersection or difference therefore
combines with uninitialized memory. Fresh allocations usually come back zeroed,
so such a shape usually contributes nothing, and that is what the migration
reproduces. The same holds for a mask group with nothing to draw: master
blends with the buffer as allocated, in practice an empty mask. The classic
fold here clears its output first, in both cases.

## Checks per commit

- **1–5, 9:** the reproductions in the PR descriptions, before and after; for
  3 and 9, screenshots
- **6:** `src/tests/integration` with OpenCL off and on, unchanged; v6 masks
  round-trip through the v6 → v7 step
- **7:** `src/tests/integration`, unchanged; a render through a negative RoI
  offset
- **8:** `ctest -R probe`; each tool from a clean checkout with no corpus; a
  normal run with none of the flags
- **10:** all of the above, plus `ctest -R flexi`, the pixel suite through
  `--verify-masks` and on the migrated load path, and `run.sh --pristine`
  against a stock master build; a walkthrough of every panel control in every
  panel position, on Linux, macOS and Windows; a real library migrated and
  edited by someone outside the branch

## Later

- Retire the classic fold. It first needs a replacement reference for the
  verification tools, either frozen renders or the fold built only with
  `BUILD_TESTING`.
- Composite masks on the GPU. Mask rendering is CPU-only today, on master as
  well.
