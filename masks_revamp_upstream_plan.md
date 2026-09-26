# Flexi masks: upstreaming plan

## What changes

The mask manager is replaced. A module's mask becomes a tree of groups: each
group folds its elements with one operator and has its own opacity, inversion
and refinement. Elements can be drawn shapes, parametric (blendif) channels or
raster masks, so one panel expresses all three.

Existing edits are converted when they are loaded. There is no coexistence
mode and no opt-in: `libs/masks.c` is deleted, and the new panel lives in the
module's blending section or in a movable panel on the darkroom canvas.

Against master:

- **Masks format v6 → v7.** The group point gains per-shape refinement, a group
  name and a group opacity, appended to the struct. The v6 → v7 step sets the
  group opacity to 1.0; zero-fill is neutral for the rest. New form types:
  `DT_MASKS_PARAMETRIC`, `DT_MASKS_RASTER`, `DT_MASKS_OBJECT`. New `state` bits
  for the group operators and modes, all previously unused.
- **Blend params v14 → v15.** Same layout. The bump makes every older edit go
  through `dt_develop_blend_legacy_params`, where the migration runs. The mask
  lock takes over a reserved field.
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
- **UI.** The panel (`blend_gui.c`), its host and canvas placement (`gtk.c`,
  `masks_gui_panel_host.c`), a darkroom toolbar toggle, slider and glyph
  changes in `dtgtk/` and `bauhaus/`, CSS, and ten preferences under
  `plugins/darkroom/masks/` and `plugins/darkroom/blend/`.
- **Tools.** `--harvest-masks`, `--verify-masks` and `--check-masks` let a user
  hand over a reproducer for a mask that migrated wrong, and let anyone re-run
  the migration check. They do nothing unless their flag is given.

### Size

Code going upstream (`src/` and `data/`, without tests and docs): about
**+42.6k / −5.6k lines**. The panel in `blend_gui.c` is the largest part, at
about 18k.

| PR | Files | Lines |
|---|---|---:|
| 1 model | `masks.h`, `masks/masks.c`, `blend.h` | +3.0k / −0.4k |
| 2 engine | `masks/group.c`, `group_internal.h` | +1.0k / −0.1k |
| | `masks/{parametric,raster,object}.c` | +1.0k / −0.1k |
| | `blend.c`, `pixelpipe_hb`, `imageop`, `develop`, `blendop.cl`, `blends/*`, `imagebuf.c` | +1.6k / −0.2k |
| | `masks/migrate_legacy.c` | +1.7k |
| 3 tests | `masks/{harvest,verify,check,persist,undo,roundtrip,styleapply,lockcheck,postedit,probe_image,scratch_image}` and the CLI flags | +9.5k |
| 4 UI | `blend_gui.c`, `blend_gui_internal.h`, `masks_gui_presets.c` | +18.1k / −2.0k |
| | `gtk.c`, panel host, darkroom toolbar | +3.5k |
| | CSS | +1.6k / −0.1k |
| | `dtgtk/*`, `bauhaus/*` | +1.1k / −0.1k |
| | `history.c`, color picker, shape tools, preferences, small touches | +0.5k / −0.1k |
| | `libs/masks.c` | −2.5k |

Tests add ~11.1k lines of cmocka suites and a 44-scenario pixel suite.

## PR sequence

Four PRs: model, engine, tests, UI. The first three change nothing a user
sees: `DEVELOP_BLEND_VERSION` stays 14, the migration is not called, and
nothing sets `DEVELOP_MASK_FLEXI`, so every mask renders through the classic
fold as today. The fourth switches everything on at once, because any split
would ship migrated edits with no panel that can show them.

### 1. Model

The masks v7 format and its v6 → v7 step, the new form types and `state`
bits, `DEVELOP_MASK_FLEXI`, and a `dev-doc/` page describing the data model.

The only visible effect: masks are written as v7, which older darktable
versions cannot read.

### 2. Engine

The flexi fold and its dispatch, render paths for the new form types, the
blend and pixelpipe changes (cache hashing, raster-user pruning), and
`migrate_legacy.c`, which is compiled but not called. Also a fix to
`dt_iop_copy_image_roi`, whose per-line fast path read before the start of the
input buffer for a negative RoI offset. That bug is on master today.

### 3. Tests and verification tools

The unit suites for model, fold, cache hashing, persistence and migration;
the pixel suite `src/tests/masking/flexi/`; and the three CLI tools. The
harvested corpus (19 MB) stays out of tree; the tools take a path.

The migration is not wired in yet, so the pixel suite checks it through
`--verify-masks`, which calls it directly.

### 4. UI, and the switch

In order, each commit building:

1. widget layer: gradient slider markers, new glyphs, expander, bauhaus
2. panel host and canvas placement, darkroom toolbar toggle
3. the panel, its presets, CSS, preferences, shape-tool changes
4. mask lock: locked masks survive reset, presets, styles and paste
5. the switch: call the migration from `dt_develop_blend_legacy_params_ext`,
   bump `DEVELOP_BLEND_VERSION` to 15, delete `libs/masks.c`, take a forced
   pre-migration backup (below), and stop turning an unknown newer
   `blendop_version` silently into default params
6. panel test suites, user documentation, `RELEASE_NOTES.md`

#### Pre-migration backup

darktable already takes a mandatory snapshot of `library.db` on schema
upgrades, whatever `database/create_snapshot` says. The blend bump is not a
schema upgrade: force the snapshot for it, and copy each sidecar to
`<file>.xmp.pre-flexi` before the first write-back. Restoring either and
opening in an older darktable gives back the classic edit. The snapshot is the
part that matters, because sidecar writing can be off.

## Evidence

| Check | Result |
|---|---|
| 12 cmocka suites | structure, grouping, operators, cache hash, persistence, migration cases; no pixels |
| pixel suite against a stock master build | 38/46 bit-identical; the other 8 use v7-only fields (per-shape refinement), which master cannot read |
| pixel suite against its reference images | 46/46 |
| corpus: 14 contributors, 63,157 edits, 8,203 distinct configuration shapes | 0 migration failures: failure rate below 0.037% (1 in 2,738) at 95% confidence |

The corpus check compares migrated renders against the classic fold in the
same binary, which follows master's rules. The bound treats each distinct
configuration shape as an independent sample; the shapes come from 14
contributors.

It depends on one fix to master, submitted separately: `dt_gradient_lookup()`
extrapolated below zero for a negative table index, so a gradient mask
applied its module in reverse along its outer edge.

**A master bug this exposes.** Master's classic fold never clears its output,
and `dt_develop_blend_process` allocates the mask without initializing it. A
top-level group whose bottom shape uses intersection or difference therefore
combines with uninitialized memory. Fresh allocations usually come back zeroed,
so such a shape usually contributes nothing, and that is what the migration
reproduces. The same holds for a mask group with nothing to draw: master
blends with the buffer as allocated, in practice an empty mask. The classic
fold here clears its output first, in both cases.

## Checks per PR

- **1:** `src/tests/integration` with OpenCL off and on, unchanged; v6 masks
  round-trip through the v6 → v7 step.
- **2:** `src/tests/integration`, unchanged; a render through a negative RoI
  offset.
- **3:** `ctest -R flexi`; the pixel suite through `--verify-masks`, and
  `run.sh --pristine` against a stock master build; each tool
  from a clean checkout with no corpus; a normal run with none of the flags.
- **4:** all of the above, plus the pixel suite on the migrated load path; a
  walkthrough of every panel control in every panel position, on Linux,
  macOS and Windows; a real library migrated and edited by someone outside
  the branch.

## Later

- Retire the classic fold. It first needs a replacement reference for the
  verification tools, either frozen renders or the fold built only with
  `BUILD_TESTING`.
- Composite masks on the GPU. Mask rendering is CPU-only today, on master as
  well.
