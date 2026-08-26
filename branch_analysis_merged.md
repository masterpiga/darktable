# Comprehensive Code Quality Analysis: `masks_revamp` Branch (Unified Report)

This document is the unified, authoritative code quality and architecture review of the `masks_revamp` branch (`master...masks_revamp`). It synthesizes structural decomposition, data model design, concurrency safety, code duplication, readability, and verified build/test findings into a single prioritized roadmap.

All facts and claims cite exact file and line locations verified in the active codebase, following the directives in [AGENTS.md](file:///Users/dudo/Documents/Coding/darktable/AGENTS.md). Section 5 states the limits of that verification.

---

## 1. Scope and Verification Summary

### Branch Metrics
- **Commits**: 5 commits (`master...masks_revamp`)
- **Diff Stat**: 143 files changed, ~28,000 insertions, ~4,100 deletions
- **Key File Changes**:
  - [src/develop/blend_gui.c](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c): 3,716 lines -> **16,752 lines** (431 functions)
  - `dt_iop_gui_blend_data_t` in [src/develop/blend.h](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend.h#L394-L787): 55 fields -> **137 fields**
  - `src/libs/masks.c`: 2,513 lines deleted (partially replaced by [src/libs/masks_flexi_host.c](file:///Users/dudo/Documents/Coding/darktable/src/libs/masks_flexi_host.c), 182 lines; the rest was absorbed into `blend_gui.c`)

### Verification Status
- **Build**: builds via `./build_dudo.sh Debug` (`build/CMakeCache.txt` shows `CMAKE_BUILD_TYPE:STRING=Debug`, with fresh artifacts in `~/Applications/darktable-masks_revamp/bin/`). Compiler warning output was not audited.
- **Test Suite**: `src/tests/masking/flexi/` holds 29 XMP scenarios with 29 matching expected PNGs, driven by `verify_effect.sh` / `run.sh`; the reported run produced valid `PARTIAL` spatial masks or the expected collapsed constants for all of them.
- **Not registered with the build system**: `src/tests/CMakeLists.txt` contains only `add_subdirectory(unittests)` and never references `masking/`. The flexi suite is therefore invisible to CTest and CI, and runs only when invoked by hand. This matches how `src/tests/integration/` works upstream, so it is not a defect, but it does mean the suite protects nothing automatically.

---

## 2. Core Findings by Quality Dimension

### 2.1 Concurrency & Layering: Pixelpipe Threads Reading GTK GUI State
*This is the highest-priority defect: a correctness and thread-safety hazard.*

> **Resolved** -- see `branch_analysis_worklog.md` §1. The bypass set is now
> snapshotted into `piece->refine_bypass` at `commit_params` time and the
> renderer reads only that. Note that two of the three pieces of state described
> below (`refine_bypass_all`, `refine_bypass_group`) were never assigned
> anywhere, so the live surface was smaller than stated here.

- **Locations**:
  - [src/develop/masks/group.c:1167-1171, 1261, 1302](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/group.c#L1167-L1171)
  - [src/develop/blend.c:313-318, 795-800](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend.c#L313-L318)
  - [src/develop/blend.h:792-805](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend.h#L792-L805)
- **Mechanism**:
  - In `_group_get_mask_roi_flexi` ([group.c:1167](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/group.c#L1167)) and `_flexi_global_refine_bypassed` ([blend.c:313](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend.c#L313)), pixelpipe worker threads inspect `module->blend_data` (`dt_iop_gui_blend_data_t *`) and call `g_hash_table_lookup(bd->masks_refine_bypassed, ...)` without synchronization.
  - `dt_masks_refine_bypass_hash` ([blend.h:792](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend.h#L792)) iterates this `GHashTable` directly during cache hash computation ([blend.c:795-800](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend.c#L795-L800)).
  - The hash table is created/modified on the GTK main thread in [blend_gui.c:3522-3526](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L3522-L3526) and destroyed in [blend_gui.c:15721](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L15721).
- **Impact**:
  - *Data race*: concurrent `g_hash_table_insert` on the UI thread during worker-thread lookup/iteration is undefined behavior.
  - *Teardown race*: leaving the darkroom view destroys the hash table while in-flight pixelpipe jobs may still be dereferencing it.
  - *Export is not affected*: `blend_data` is allocated only in `dt_iop_gui_init_blending` ([blend_gui.c:16356](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L16356)), so in `darktable-cli`, export and thumbnail pipes it is NULL and the bypass silently reads as "off". That is the intended behavior (the flag is deliberately never serialized, [blend.c:305-308](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend.c#L305-L308)) — it is noted here only so the fix preserves it, not as a defect.
- **Remedy**: Respect darktable's threading boundary. Snapshot the transient bypass state into pipe-local data (`piece->blendop_data` or committed blend parameters) during `commit_params()` on the main thread, and derive the cache hash from that snapshot, rather than querying live GTK pointers during pixel processing.
- **Related**: the bypass table multiplexes three key spaces by bit arithmetic with no shared helper — element = `formid`, group = `formid | 0x80000000U`, global = `0` — written independently at [blend_gui.c:3456](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L3456) and [group.c:1302](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/group.c#L1302). See §2.5C.

---

### 2.2 Decomposition: Monolithic Complexity in `blend_gui.c`
- **Location**: [src/develop/blend_gui.c:1-16752](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L1-L16752)
- **Mechanism**:
  - [blend_gui.c](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c) contains 431 functions and mixes at least eight distinct responsibilities (classic blendif GUI, scoped refinement, shape properties, inline opacity popups, DnD, group-layout presets, the inline parametric row editor, the mask tree list, and panel relocation).
  - The top functions are massive: `_build_masks_list` (844 lines, [line 13801](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L13801)), `_make_shape_row` (452 lines, [line 13180](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L13180)), `dt_iop_gui_init_blending` (400 lines, [line 16347](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L16347)), `dt_iop_gui_init_masks` (336 lines, [line 15361](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L15361)), `_make_pending_shape_row` (278 lines, [line 10695](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L10695)).
  - `_build_masks_list` conflates **model mutation** (realizing empty groups, lines 13891-13965), **reconciliation** (scaffold seeding, ordinal assignment, selection seeding, lines 13982-14046), and **widget packing** (lines 14052-14644). Drawing a shape only takes effect *because a rebuild happened to run*.
- **Impact**:
  - Requires 61 static forward declarations due to out-of-order definitions.
  - Full destroy-and-rebuild churn forces seven interconnected guard flags (eight fields) to avoid re-entrancy and use-after-free crashes: `masks_rebuild_suppressed` ([blend.h:748](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend.h#L748)), `masks_rebuild_pending`, `masks_rebuild_idle_id`, `masks_list_sig` ([blend.h:593](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend.h#L593)), `masks_skip_group_select_release` + `masks_skip_group_select_release_time`, `masks_suppress_toggle_select`, `masks_row_click_handled`.
- **Remedy**:
  1. Separate `_build_masks_list` into `_masks_panel_reconcile()` (pure model logic, callable directly at mutation points) and `_masks_panel_pack()` (pure widget tree construction). This is also what retires most of the guard flags above.
  2. Move localized property changes (name edits, solo toggles, opacity updates) to in-place widget updates, reserving the full rebuild strictly for structural list mutations. The `masks_row_map` formid->row hash ([blend.h:591](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend.h#L591)) already exists to make this cheap.
  3. Decompose [blend_gui.c](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c) into cohesive modules under `src/develop/masks/gui/`:
     - `blend_gui.c`: Core blend modes, colorspaces, and IOP container lifecycle.
     - `masks_gui_list.c`: Tree list widget, cluster folding, row packing, and rebuild scheduling.
     - `masks_gui_dnd.c`: Drag-and-drop state machine and handlers.
     - `masks_gui_param_editor.c`: Inline single-channel parametric editor and range slider.
     - `masks_gui_properties.c`: Shape property expanders, resize handlers, and inline opacity.
     - `masks_gui_refine.c`: Refinement controls and scoped targeting.
     - `masks_gui_presets.c`: Group-layout preset capture/apply and its `data.presets` persistence.

---

### 2.3 Reusability & DRY (Code Duplication)

#### A. Duplicate Group Headers & Shape Rows
- **Locations**:
  - `_pack_empty_group_header` ([blend_gui.c:10979](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L10979), 249 lines) vs populated group headers built inline in `_build_masks_list`
  - `_make_pending_shape_row` ([blend_gui.c:10695](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L10695), 278 lines) vs `_make_shape_row` ([blend_gui.c:13180](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L13180), 452 lines)
- **Finding**:
  - `_pack_empty_group_header` is a structural clone of the populated group header — same `_pack_row_header` layout, same CSS classes, same 50dpi title request, same ellipsize + `max_width_chars(1)` pair, same drag source/dest wiring. It contains 8 explicit comments stating it matches "exactly" ([lines 11007, 11015, 11025, 11065, 11099, 11132, 11150, 11205](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L11007)). The two are kept in sync by comment, not by code.
  - `_make_pending_shape_row` duplicates drag handles, name labels, and opacity slider wiring, including a copy-pasted duplicate comment ([lines 10713-10721](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L10713-L10721) has the same "opacity: universal across every shape kind" note written twice back to back).
- **Remedy**:
  - Unify group headers into a single `_make_group_header(module, desc)` accepting a descriptor struct (`op`, `within`, `opacity`, `name`, `ordinal`, `is_base`, `selected`, `member_ids or NULL`).
  - Unify shape row generation so pending rows share layout helpers with committed rows.

#### B. Drag-and-Drop Handler Duplication (~23 functions, ~650 lines)
- **Locations**: [blend_gui.c:6359-6750](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L6359-L6750) and [blend_gui.c:10207-10600](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L10207-L10600)
- **Finding**: roughly 23 drag-get, drag-received, drop and drop-feedback functions (`_masks_row_drag_get`, `_masks_group_drag_get`, `_masks_cluster_drag_get`, `_masks_empty_drag_get`, `_masks_shape_to_group_drop`, `_masks_cluster_to_group_drop`, `_masks_cluster_row_drop`, `_masks_shape_to_empty_drop`, `_masks_cluster_to_empty_drop`, `_masks_group_to_empty_drop`, `_masks_empty_reorder_drop`, `_element_row_drag_received`, `_masks_empty_header_drag_received`, and the motion/leave/begin helpers) totalling ~650 lines. The largest are modest on their own (`_masks_shape_to_empty_drop` 83, `_masks_cluster_to_empty_drop` 87, `_masks_shape_to_group_drop` 71), but they repeat the same skeleton: payload parsing, vertical coordinate hit-testing, `grp->points` detach/splice, operator normalization, and a history commit.
- **Remedy**: Unify into a single polymorphic drag payload descriptor (`dt_masks_drag_payload_t`) and one drop-target execution engine (`_masks_move_payload_to_target()`). The value here is the collapsed state machine, not the raw line count.

#### C. Stale Fork of `src/libs/masks.c` Property Tables
- **Locations**: [blend_gui.c:1023, 3950, 3975, 4119, 4150, 4320, 4423, 10713](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L1023), [masks/object.c:2172](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/object.c#L2172)
- **Finding**: `_blend_masks_properties` ([blend_gui.c:3954](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L3954)) and `_paint_resize_unit` ([blend_gui.c:4119](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L4119)) are forks of code from `src/libs/masks.c`, which **this branch deletes**. Nine comments cite the deleted file as the reference authority, one of them instructing the reader to "keep the two in sync if either changes" ([blend_gui.c:3950-3953](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L3950)) and another describing itself as an "exact copy" ([4119](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L4119)). Per AGENTS.md, a stale comment is worse than none.
- **Remedy**: Move the property metadata table and resize painting helpers into `src/develop/masks/masks.c` behind public `dt_masks_*` accessors. Do not simply reword the comments — the duplication is the finding.

#### D. Menu & Popup Boilerplate
- **Locations**:
  - Menu construction clusters: [blend_gui.c:672-876](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L672-L876), [2288-2466](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L2288-L2466), [8043-8084](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L8043-L8084), [9858-10136](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L9858-L10136), [11332-11406](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L11332-L11406), [15180-15197](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L15180-L15197) — 43 menu/menu-item constructions across ~6 sites.
  - Bauhaus static popups: [blend_gui.c:4750-4850, 12250-12400](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L4750-L4850), supported by the `static_popup` branch in [bauhaus.c:632-711](file:///Users/dudo/Documents/Coding/darktable/src/bauhaus/bauhaus.c#L632-L711) and `dt_bauhaus_widget_show_popup` ([bauhaus.c:3514](file:///Users/dudo/Documents/Coding/darktable/src/bauhaus/bauhaus.c#L3514)).
- **Finding**:
  - The menus are not as boilerplate-heavy as first assessed — a `_add_menu_section_header` helper already exists and items are ~8-11 lines each. What actually repeats, roughly 40 times, is the triple `g_object_set_data_full(item, "formids", g_list_copy(formids), g_list_free)` + `g_signal_connect(...)` + `gtk_menu_shell_append(...)` (e.g. `_build_group_actions_menu`, [blend_gui.c:9899](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L9899)).
  - The static hover-preview hook is a public API expressed entirely as magic `g_object_set_data` string keys, documented as such in the header itself ([bauhaus.h:202-215](file:///Users/dudo/Documents/Coding/darktable/src/bauhaus/bauhaus.h#L202-L215)), set at [blend_gui.c:12364, 12386-12388](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L12364).
- **Remedy**: A local `_menu_add_item(menu, label, tooltip, formids, cb, module)` helper covering the repeated triple; and promote the static popup/hover-preview hooks to a typed Bauhaus API (`dt_bauhaus_slider_set_static_hover_preview()`), keeping the string keys private to `bauhaus.c`.

---

### 2.4 Generality & Data Model Design

#### A. Generality Gap in `DT_UI_PANEL_FLEXI` Window Column
- **Location**: [src/gui/gtk.c:3549-3700](file:///Users/dudo/Documents/Coding/darktable/src/gui/gtk.c#L3549-L3700), [src/gui/gtk.h:389-488](file:///Users/dudo/Documents/Coding/darktable/src/gui/gtk.h#L389-L488)
- **Finding**:
  - `gtk.c` adds ~450 lines of feature-specific panel code, including custom resize-handle callbacks (`_flexi_handle_button_callback`, `_flexi_handle_motion_callback`, `_flexi_handle_cursor_callback`) that parallel the existing `_panel_handle_*` handlers. The header comment at [gtk.c:3549-3555](file:///Users/dudo/Documents/Coding/darktable/src/gui/gtk.c#L3549-L3555) states the duplication as a deliberate choice.
  - Crucially, `DT_UI_PANEL_FLEXI` is excluded from the width arbitration array at [gtk.c:3411](file:///Users/dudo/Documents/Coding/darktable/src/gui/gtk.c#L3411) (`side_panels[] = { DT_UI_PANEL_LEFT, DT_UI_PANEL_RIGHT }`). Since the flexi panel is a real column inside `centerrow`, left/right panel resizing can over-commit window width — a behavioral consequence, not only an aesthetic one.
- **Remedy**: Generalize into a reusable `DT_UI_PANEL_EXTRA` column, fold it into `side_panels[]`, and route handle events through table-driven dispatch instead of `strcmp` chains. All masks-specific behavior (corner icon, tooltip, collapse-on-mask-off) belongs in `masks_flexi_host.c` / `blend_gui.c`.

#### B. Fragile Invariants in the Inferred-Group Representation
- **Locations**: [group.c:1175-1245](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/group.c#L1175-L1245), [blend_gui.c:8478-8600](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L8478-L8600), [blend_gui.c:13891-13965](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L13891-L13965)
- **Finding**: Groups are not stored entities. They are inferred as maximal runs of same-operator points in the flat `grp->points` list, segmented by the `group_start` flag (and, for pre-v10 blobs, the historic `DT_MASKS_STATE_GROUP_BREAK` bit). Every group-level attribute — operator/modifier state, refinement, custom name, group opacity — is *broadcast* onto each member of the run, by hand-rolled loops in roughly 12 places in `blend_gui.c`. Staged empty groups (`dt_masks_empty_group_t`) exist only in UI memory (`bd->empty_groups`), anchored by `below_fid`, creating a second, parallel group representation that the persisted model knows nothing about.
- **Impact**: Any member mutation (delete, reorder, drag, merge) must manually re-broadcast attributes and repair `group_start` on neighbours; group reordering is intricate `GList` slicing performed directly inside UI callbacks. The invariants are documented but unenforced, so a missed broadcast is silent.
- **Remedy**: Encapsulate run and boundary manipulation as data-model primitives in `src/develop/masks/group.c` — `dt_masks_group_get_runs()`, `dt_masks_group_move_run()`, `dt_masks_group_set_run_properties()` (one broadcast implementation) — and have the UI call those rather than splicing raw `GList` nodes. This is also the prerequisite for deciding the long-standing empty-groups persistence question.

#### C. Parametric and Raster Form Engine Invariants
- **Locations**:
  - [src/develop/masks/parametric.c:167-198](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/parametric.c#L167-L198)
  - [src/develop/masks/raster.c:124-135, 146](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/raster.c#L124-L135)
- **Finding**:
  - In `_parametric_get_mask_roi`, the renderer builds a stack struct `dt_develop_blend_params_t tmp`, points `pc->blendop_data` at it, calls `make_mask`, then restores the saved pointer. `piece` is pipe-local, so this is not a cross-thread race, but it leaves a pipe structure pointing at a stack local for the duration of the call, is hostile to re-entrancy, and requires casting away `const` on `piece`.
  - In `_raster_get_mask_roi`, `_raster_resolve_source` walks `module->dev->iop` doing `dt_iop_module_is(iop, p->source)` string comparisons on **every mask evaluation**, rather than resolving once at commit time.
- **Remedy**:
  - Provide a single-channel evaluation helper taking explicit channel/parameter pointers, so no `blendop_data` mocking is needed.
  - Pre-resolve the raster source module pointer in `commit_params()`. The branch already keeps `blend_params.raster_mask_*` in sync with the raster element for the pipe's existing dependency machinery, which is the natural place to cache it.

#### D. Branch Scaffolding in Core Startup -- WITHDRAWN, this finding was wrong
> **Not a defect. Do not act on this.** `flexi_test_mode` defaulting to TRUE is
> a deliberate safeguard: it keeps an experimental masks branch away from a
> tester's real photo library, for testers who just double-click the app and
> never pass a flag. Calling it "scaffolding to remove before upstreaming" was a
> misreading of intent -- it is a feature of a test build, and it stays. The
> assignment now carries a DO NOT comment saying so. Retained below only as a
> record of what was originally claimed.

- **Location**: [src/common/darktable.c:1130, 1593-1620](file:///Users/dudo/Documents/Coding/darktable/src/common/darktable.c#L1130), [src/common/image.c:3022-3032](file:///Users/dudo/Documents/Coding/darktable/src/common/image.c#L3022-L3032)
- **Finding**: `darktable.flexi_test_mode = TRUE;` ([darktable.c:1130](file:///Users/dudo/Documents/Coding/darktable/src/common/darktable.c#L1130)) is the unconditional default, redirecting all database and config operations to `/tmp/flexi_mask_test` and sidecars to `*_flexi_test.xmp`. It also changes what `--configdir` and `--library` mean: in test mode they only select what to seed *from*, never where writes land. `--no-flexi-test-mode` ([darktable.c:1228](file:///Users/dudo/Documents/Coding/darktable/src/common/darktable.c#L1228)) is the only way out.
- **Remedy**: Remove before upstream submission, or invert to an opt-in CLI flag defaulting to `FALSE`. The recursive directory copier (`_flexi_test_mode_copy_dir_recursive`, [darktable.c:996](file:///Users/dudo/Documents/Coding/darktable/src/common/darktable.c#L996)) belongs in `common/file_location.c` if it is kept at all.

---

### 2.5 Readability & Maintainability

#### A. Stringly-Typed Widget State
- **Location**: [src/develop/blend_gui.c](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c)
- **Finding**: 388 `g_object_set_data`/`get_data` calls across ~85 distinct string keys (`"formid"` 34x, `"eg"` 20x, `"group-key"` 18x, `"module"` 14x, `"formids"` 13x, down to one-offs like `"skip-auto-expand"` and `"precise-anchor-marker-x"`). None are `#define`d and none document ownership, although some keys carry a `GDestroyNotify` and lookalikes do not. The `"eg"` key stores raw `dt_masks_empty_group_t *` pointers owned by `bd->empty_groups`, while `bd->selected_empty` and `bd->insert_empty` are declared as bare `void *` in [blend.h:628, 659](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend.h#L628).
- **Remedy**: Define named constants in a header (`#define MASK_KEY_FORMID "formid"`) with documented pointee type and ownership, and forward-declare `dt_masks_empty_group_t` in `masks.h` so those two fields can be typed and compiler-checked.

#### B. Dead Code with Live NULL Dereferences
> **Resolved, and partly re-diagnosed** -- see `branch_analysis_worklog.md` §2.
> The two sites below turned out to be *unreachable* (their only callers were
> dead branches of `blend_color_picker_apply`) and would have **segfaulted**,
> not warned, since `bd->channel` is never assigned. Two different sites in the
> same cluster were genuinely live and are the ones that were actually costing
> GTK criticals: `_blendop_blendif_disp_alternative_worker` (reached by pressing
> `a` on a parametric row's slider) and `_blendif_change_blend_colorspace`.

- **Locations**:
  - [blend_gui.c:1670-1672](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L1670-L1672): calls `gtk_widget_set_sensitive(GTK_WIDGET(data->channel_boost_factor_slider), ...)` and `dt_bauhaus_slider_set(...)` on a field the header documents as permanently NULL.
  - [blend_gui.c:1344](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L1344): calls `gtk_notebook_get_tab_label(bd->channel_tabs, gtk_notebook_get_nth_page(bd->channel_tabs, tab))` where `bd->channel_tabs` is never assigned anywhere in the tree.
  - Both sit under `_blendop_blendif_update_tab` ([blend_gui.c:1576](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L1576)), still reachable from `blend_color_picker_apply` ([2166](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L2166)) and [1312](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L1312). Neither should crash — `DT_BAUHAUS_WIDGET(NULL)` bails and the `gtk_*` calls hit `g_return_if_fail` — but each pass should emit several GTK criticals.
  - Fully dead struct members retained with a "kept for struct-layout stability" rationale that does not apply to a heap-allocated GUI struct: `blendif_section`, `blendif_header`, `masks_import_label` (declaration only); `masks_param_op`, `masks_param_op_box`, `masks_elements_header`, `masks_elements_box`, `masks_elements_title` (declaration plus a single `= NULL` in cleanup).
- **Remedy**: Delete the classic-tab editor functions together with the retired fields, rather than adding NULL guards around them.

#### C. Code Style & Invariants
- **Magic bit arithmetic**: `formid | 0x80000000U` multiplexes element and group keys in the refinement-bypass table, written inline at [blend_gui.c:3456](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L3456) and [group.c:1302](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/group.c#L1302). Introduce `DT_MASKS_MAKE_GROUP_KEY(cid)` / `DT_MASKS_IS_GROUP_KEY(key)` in `masks.h`.
- **Unenforced state invariants**: in `dt_masks_state_t` ([masks.h](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks.h)), `DT_MASKS_STATE_WITHIN` must hold exactly one of three bits and `DT_MASKS_STATE_OP_COMBINE` exactly one of seven. Both are documented in prose only. A `_state_set_within()` setter plus an assert would make them checkable.
- **Bit ordering**: `dt_masks_state_t` bits are declared out of numeric order (8, 9, 10, 11, 12, 15, 13, 14, 17, 16), making "which bit is free" a search rather than a glance.
- **Non-ASCII punctuation in translations**: [blend_gui.c:3859, 7492, 10445](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L3859) plus the middle-dot group-caption separator use em-dashes, ellipses and middle dots inside `_()` strings. Per AGENTS.md this is not cosmetic: `po/` msgids are keyed to plain-ASCII American English.
- **Line length**: 424 added lines exceed 90 columns (334 in `blend_gui.c`); `group.c:1261` and `group.c:1302` at 130+ and 160+ characters are the worst outside it.
- **Preference registration is correct**: the seven new conf keys are declared in [data/darktableconfig.xml.in](file:///Users/dudo/Documents/Coding/darktable/data/darktableconfig.xml.in), which AGENTS.md requires and which changes of this size commonly miss.

---

## 3. Prioritized Refactoring Roadmap

```
+-------------------------------------------------------------------------+
| Phase 1: Correctness & Thread Safety (Critical)                         |
| - Snapshot refinement bypass state in commit_params() (fix group.c/     |
|   blend.c races), preserving the NULL-blend_data = no-bypass export     |
|   behaviour.                                                            |
| - Delete dead classic-tab editor code and its NULL-deref call sites.    |
| - (withdrawn: flexi_test_mode stays TRUE, it is a tester safeguard)     |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| Phase 2: Structural Decomposition (Upstream Enabler)                    |
| - Split _build_masks_list into _masks_panel_reconcile & _pack.          |
| - Move localized row updates off the full-rebuild path.                 |
| - Decompose blend_gui.c (16.7k lines) into:                             |
|   * blend_gui.c (classic & core blendop)                                |
|   * masks_gui_list.c (flexi tree list & rebuilds)                       |
|   * masks_gui_dnd.c (unified DnD controller)                            |
|   * masks_gui_param_editor.c (inline parametric row editor)             |
|   * masks_gui_properties.c (shape property expanders)                   |
|   * masks_gui_refine.c (refinement controls)                            |
|   * masks_gui_presets.c (group-layout preset persistence)               |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| Phase 3: Duplication Elimination & Generality (DRY & Robustness)        |
| - Promote run/broadcast manipulation to group.c data-model primitives.  |
| - Unify group headers (_make_group_header) and shape rows.              |
| - Consolidate DnD handlers into unified payload/dispatcher.             |
| - Move metadata property tables to develop/masks/masks.c.               |
| - Generalize DT_UI_PANEL_FLEXI and integrate with width arbitration.    |
| - Replace parametric stack mocking; pre-resolve raster source module.   |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| Phase 4: Polish & Style Compliance (Clean Upstream PRs)                 |
| - Define named constants for g_object_set_data string keys.             |
| - Add group-key macros; reorder dt_masks_state_t bits; assert the       |
|   WITHIN / OP_COMBINE mutual-exclusion invariants.                      |
| - Convert non-ASCII punctuation in translatable strings to ASCII.       |
| - Enforce <90 column formatting per AGENTS.md.                          |
+-------------------------------------------------------------------------+
```

### Upstream PR Staging
Per [masks_revamp_upstream_plan.md](file:///Users/dudo/Documents/Coding/darktable/masks_revamp_upstream_plan.md):
1. **Batch 1 (infra / widget fixes)**: standalone bauhaus fixes, gradient slider polish, and history hash fixes can land immediately.
2. **Batch 2 (headless engine)**: `masks.h`, `group.c`, `parametric.c`, `raster.c` and `migrate_legacy.c` can go as a self-contained PR with test coverage — which is also the argument for registering the flexi suite with CTest (§1).
3. **Batch 3 (UI)**: Phase 2's subdivision of `blend_gui.c` is a precondition, not a follow-up. A 15k-line addition to one file cannot be reviewed.

---

## 4. Master Action Items Matrix

| Subsystem | Area | Exact Location | Issue | Priority | Target Refactoring |
|---|---|---|---|---|---|
| **Pixelpipe / Blend** | Thread Safety | [group.c:1167](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/group.c#L1167), [blend.c:313](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend.c#L313) | Worker threads query live UI `GHashTable` | **High** | Snapshot bypass state at `commit_params()` |
| **Blend GUI** | Decomposition | [blend_gui.c:1-16752](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L1) | 16.7k-line monolithic file (431 functions, 61 fwd decls) | **High** | Split into modular `src/develop/masks/gui/` files |
| **Core Init** | ~~Safety / Config~~ | [darktable.c:1130](file:///Users/dudo/Documents/Coding/darktable/src/common/darktable.c#L1130) | ~~Default-on `/tmp/flexi_mask_test` redirection~~ | **Withdrawn** | Intentional tester safeguard - leave as is |
| **GUI Lifecycle** | Readability / State | [blend_gui.c:13801](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L13801) | `_build_masks_list` mixes mutation, reconcile & packing; 7 guard flags | **Medium** | Split into `_reconcile` / `_pack`; in-place row updates |
| **Masks Data Model** | Generality | [group.c:1175](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/group.c#L1175), [blend_gui.c:8478](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L8478) | Inferred runs + ~12 hand-rolled attribute broadcasts | **Medium** | `dt_masks_group_*_run()` primitives in `group.c` |
| **Drag & Drop** | Reusability | [blend_gui.c:6359, 10207](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L6359) | ~23 handlers / ~650 lines repeating one skeleton | **Medium** | Unify payload & drop target dispatcher |
| **Group / Row UI** | Reusability | [blend_gui.c:10695, 10979](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L10695) | Cloned headers (empty vs real) & rows (pending vs real) | **Medium** | Unify via `_make_group_header` descriptor |
| **Window Layout** | Generality | [gtk.c:3411, 3549](file:///Users/dudo/Documents/Coding/darktable/src/gui/gtk.c#L3411) | `DT_UI_PANEL_FLEXI` excluded from width arbitration | **Medium** | Generalize to `DT_UI_PANEL_EXTRA` in `side_panels[]` |
| **Parametric / Raster** | Generality | [parametric.c:167](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/parametric.c#L167), [raster.c:124](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/raster.c#L124) | `blendop_data` stack mocking; source resolved per evaluation | **Medium** | Clean channel evaluator; pre-resolved pointer |
| **Property Tables** | Architecture | [blend_gui.c:3954](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L3954) | Fork of the deleted `src/libs/masks.c`; 9 stale citations | **Medium** | Move to `src/develop/masks/masks.c` |
| **Test Suite** | Coverage | [src/tests/CMakeLists.txt](file:///Users/dudo/Documents/Coding/darktable/src/tests/CMakeLists.txt) | 29-scenario flexi suite unknown to CTest/CI | **Low** | Register `masking/` or wire into CI script |
| **Bauhaus Popups** | Reusability | [bauhaus.h:202](file:///Users/dudo/Documents/Coding/darktable/src/bauhaus/bauhaus.h#L202), [bauhaus.c:632](file:///Users/dudo/Documents/Coding/darktable/src/bauhaus/bauhaus.c#L632) | Magic string keys as the public hover-preview API | **Low** | Promote to typed Bauhaus setter |
| **Widget Keys** | Readability | [blend_gui.c](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c) | 388 ad-hoc `g_object_set_data` calls, ~85 keys | **Low** | Centralize `#define MASK_KEY_*` constants |
| **Dead Code** | Hygiene | [blend_gui.c:1344, 1670](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L1344) | Dereferencing permanently-NULL fields | **Low** | Delete legacy tab callbacks & retired fields |
| **Style & Strings** | Standards | [blend_gui.c:3859, 7492](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend_gui.c#L3859), [masks.h](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks.h) | Non-ASCII punctuation in `_()`, >90 columns, unenforced bit invariants | **Low** | ASCII strings, reorder enum, add asserts |

---

## 5. Verification Limits

- Claims about file contents, line numbers, function sizes and call sites were checked against the working tree.
- The **GTK criticals** predicted in §2.5B were read from the code, not observed at runtime. Confirm by activating a parametric row's color picker and watching stderr.
- The **width over-commit** in §2.4A follows from `side_panels[]` excluding the flexi column; it was not reproduced interactively.
- The build was confirmed present and Debug-configured; **compiler warning output was not audited**, so "builds cleanly" is stated only as "builds".
- The flexi mask suite's pass result is carried over from the reported run; it was not re-executed for this review, and neither was `src/tests/integration/`. No pixel-level claim here rests on a run performed during the review itself.
