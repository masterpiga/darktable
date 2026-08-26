# masks_revamp — code quality analysis

Read-only review of `master...masks_revamp`, focused on generality, reusability,
readability and decomposition. No files were changed to produce it. Every claim
is cited to a file and line that was actually read; see *What was not verified*
at the end for the limits.

## Scope of the branch

5 commits, 143 files, ~28k insertions. Excluding tests, the diff is dominated by
one file:

| File | master | branch |
|---|---|---|
| `src/develop/blend_gui.c` | 3,716 lines | **16,752 lines**, 431 functions |
| `dt_iop_gui_blend_data_t` (`src/develop/blend.h:394-787`) | 55 fields | **137 fields** |
| `src/libs/masks.c` | 2,513 lines | deleted |

The new mask *model* code (`masks/parametric.c`, `masks/raster.c`,
`masks/migrate_legacy.c`, `masks/group.c`) is in good shape: small functions, the
form-type vtable pattern is reused correctly, and the neutrality reasoning is
documented where it matters. The problems are concentrated in the GUI layer and
in three places where the GUI has leaked into non-GUI code.

## 1. Decomposition: `blend_gui.c` is now three unrelated subsystems in one file

431 functions, and the top five are 844/452/400/336/307 lines:

- `_build_masks_list` (`blend_gui.c:13801`) — **844 lines**
- `_make_shape_row` (`blend_gui.c:13180`) — 452
- `dt_iop_gui_init_blending` (`blend_gui.c:16347`) — 400
- `_make_pending_shape_row` (`blend_gui.c:10695`) — 278
- `_pack_empty_group_header` (`blend_gui.c:10979`) — 249

`_build_masks_list` is not just long, it mixes three responsibilities that should
not be in a widget builder:

1. **Model mutation** — the "realize an empty group into a real run" block
   (`blend_gui.c:13891-13965`) rewrites `grp->points`, re-anchors `below_fid`s,
   broadcasts refinement and names onto members, and reassigns ordinals. Drawing
   a shape only takes effect *because a rebuild happened to run*.
2. **Reconciliation** — scaffold seeding (`13982`), ordinal prune/assign
   (`13996`), stale-solo prune, selection seeding (`14029`).
3. **Widget packing** — the remaining ~500 lines.

The natural split is `_masks_panel_reconcile(module)` (pure model, testable,
callable from the mutation sites directly) followed by
`_masks_panel_pack(module)`. That also removes the reason the file needs
`masks_rebuild_suppressed` / `masks_rebuild_pending` / `masks_rebuild_idle_id` /
`masks_list_sig` (`blend.h:593`, `748`, `754`, `761`) as four separate guards
around one call.

Beyond that, the file holds at least three separable units that share almost
nothing:

- classic blendif editor + blend modes (roughly lines 1-3000)
- the flexi mask panel (roughly 3000-15200) — the mask-manager replacement
- panel relocation / hosting (`_masks_flexi_relocate`, `blend_gui.c:16177`)

Splitting into `blend_gui.c` / `masks_panel.c` / `masks_panel_dnd.c` would make
the diff reviewable upstream. Right now no reviewer can meaningfully read a
15k-line addition to one file.

## 2. Reusability: three near-duplicate implementations kept in sync by comment

**a) Empty-group header vs. real-group header.** `_pack_empty_group_header`
(`blend_gui.c:10979`) is a structural clone of the real-group header built inline
inside `_build_masks_list`. It carries at least eight comments of the form
"exactly matching a populated group's own header (see its own labevt further
down)" — `11007-11012`, `11015`, `11025`, `11065`, `11099`, `11132`, `11150`,
`11205`. Both build the same
`_pack_row_header(hdr, handle, labevt, opacity_inner, badge_stack, within_sel, ...)`
layout with the same CSS classes, the same 50dpi title request, the same
ellipsize + `max_width_chars(1)` pair, the same drag source/dest set. One
`_make_group_header(module, desc)` taking a small descriptor (`op`, `within`,
`opacity`, `name`, `ordinal`, `is_base`, `selected`, `member_ids or NULL`) would
collapse both.

**b) Pending shape row vs. real shape row.** `_make_pending_shape_row`
(`blend_gui.c:10695`, 278 lines) duplicates the drag handle, name label and
opacity slider construction of `_make_shape_row` (`blend_gui.c:13180`, 452
lines). The duplication is visible even in the comments — `10713-10721` has the
same "opacity: universal across every shape kind" comment written twice back to
back, one long form and one short, a copy-paste leftover.

**c) The mask-properties table was forked out of the deleted lib.**
`_blend_masks_properties` (`blend_gui.c:3954`) says:

> This metadata table mirrors src/libs/masks.c's file-local `_masks_properties`
> exactly … It can't be shared directly since the original is `static const` in
> another translation unit; keep the two in sync if either changes.

That file no longer exists on this branch. There are **9 comments citing
`src/libs/masks.c`** as the authority for current behavior: `blend_gui.c:1023`,
`3950`, `3975`, `4119` ("Exact copy of src/libs/masks.c's own
`_paint_resize_unit`"), `4150`, `4320`, `4423`, `10713`, and
`masks/object.c:2172`. AGENTS.md is explicit that a stale comment is worse than
none. The fix is not to edit the comments: it is to move the property table and
`_paint_resize_unit` into `src/develop/masks/masks.c` behind a public accessor,
which is where they belonged once the lib was deleted.

## 3. Generality: a feature-specific panel hardcoded into `src/gui/gtk.c`

The branch adds ~450 lines to the generic window infrastructure for one feature:
`DT_UI_PANEL_FLEXI` plus
`dt_ui_flexi_panel_content/_set_side/_set_collapsed/_is_collapsed/_set_icon`
(`gui/gtk.h`, `gui/gtk.c:3549-3700`). The header comment states the duplication
outright (`gtk.c:3549-3555`):

> own resize-handle callbacks (not sharing
> `_panel_handle_button_callback`/`_motion_callback`'s left/right/bottom
> dispatch), own conf-backed width … Deliberately not touching
> `_handle_panel_widths`/`_panel_set_side_panel_width`'s L/R-only
> width-arbitration logic.

So `_flexi_handle_button_callback` / `_flexi_handle_motion_callback` /
`_flexi_handle_cursor_callback` (`gtk.c:3564-3610`) are a parallel copy of the
existing handlers, and the flexi column is excluded from the width arbitration at
`gtk.c:3411`, where `side_panels[] = { DT_UI_PANEL_LEFT, DT_UI_PANEL_RIGHT }` is
what reserves room for the center view. Since the flexi panel *is* a real column
inside `centerrow`, dragging the left or right panel can over-commit the window
width — that is a behavioral consequence of the non-generality, not just an
aesthetic one.

Two options, in order of preference:

- Make it a generic extra column (`DT_UI_PANEL_EXTRA`, name-driven conf key), add
  it to `side_panels[]`, and reuse `_panel_handle_*` by making the handle-name
  dispatch table-driven instead of `strcmp` chains. All masks-specific behavior
  (icon, tooltip, collapse-on-mask-off) moves to `libs/masks_flexi_host.c` /
  `blend_gui.c`.
- If a fourth column really is masks-only, at minimum share
  `_panel_set_side_panel_width` and include it in the arbitration array.

**Related, smaller generality gap:** the bauhaus hover-preview hook is a public
API expressed entirely as magic `g_object_set_data` string keys — the header
itself documents *"Set it with `g_object_set_data(G_OBJECT(widget),
"dt-bauhaus-static-hover-preview", hook)`"* (`bauhaus/bauhaus.h:202-215`), read at
`bauhaus.c:708-711`, used at `blend_gui.c:12386`. A shared widget library should
expose `dt_bauhaus_slider_set_static_hover_preview(w, fn, data)` and keep the
keys private.

## 4. Layering: pixelpipe worker threads read GUI-owned state

This is the finding to fix first, because it is a correctness hazard and not only
a design one. The transient refinement-bypass preview is implemented by reaching
from the render path into `module->blend_data`:

- `masks/group.c:1167-1171` — `bd->refine_bypass_all`,
  `bd->panel_selected_group_cid`
- `masks/group.c:1261`, `masks/group.c:1302` —
  `g_hash_table_lookup(bd->masks_refine_bypassed, …)`
- `blend.c:313-318` — same table
- `blend.c:795-800` — folds `dt_masks_refine_bypass_hash(bd)` into the drawn-mask
  cache key
- `blend.h:792-805` — iterates the table inline in a header

The same table is created and written on the GUI thread at `blend_gui.c:3522-3526`
and destroyed at `blend_gui.c:15721`. Concurrent `g_hash_table_insert` against
lookups/iteration from a pipe thread is undefined, and the destroy at
darkroom-leave races any pipe still finishing. This is the same class of bug
already fixed once in `masks.c` (the unlocked `dev->forms` mutation racing the
pixelpipe's deep-copy read), and it deserves the same treatment: snapshot the
bypass set into `piece->blendop_data` in `commit_params`, so the renderer reads
pipe-local data and the cache hash comes from that snapshot.

Two smaller things in the same area:

- The bypass table encodes three key spaces by bit-twiddling with no shared
  helper: element = `formid`, group = `formid | 0x80000000U`, global = `0`. The
  literal is written independently at `blend_gui.c:3456` and `masks/group.c:1302`.
  One `_refine_bypass_key(kind, formid)` inline in `masks.h` removes the class of
  bug where the two drift.
- `masks/group.c:1261` and `1302` are 130+ and 160+ characters — the file's only
  real style outliers.

## 5. Readability: stringly-typed widget state

There are **388** `g_object_set_data`/`get_data` calls in `blend_gui.c` across
**~85 distinct string keys** — `"formid"` (34), `"eg"` (20), `"group-key"` (18),
`"module"` (14), `"formids"` (13), down to one-offs like `"skip-auto-expand"`,
`"badge-noop"`, `"precise-anchor-marker-x"`. None are `#define`d, none are
documented in one place, and several encode ownership (`g_object_set_data_full`
with `g_list_free`, `blend_gui.c:13214`) while lookalikes do not.

The `"eg"` key is the worst case, because it stores a raw
`dt_masks_empty_group_t *` on a widget while `bd->empty_groups` owns and frees
those nodes — and the struct is so unavailable at that point that
`bd->selected_empty` and `bd->insert_empty` are declared as bare `void *`
(`blend.h:628`, `blend.h:659`). Forward-declaring the type in `masks.h` and
typing those two fields costs nothing and lets the compiler check the ~20 `"eg"`
round-trips.

Minimum viable improvement: a `masks_panel_keys.h` block of
`#define MASK_KEY_FORMID "formid"` … with a one-line comment per key stating the
pointee type and who owns it.

## 6. Dead code that is not merely dead

`dt_iop_gui_blend_data_t` retains several fields documented as *"permanently NULL
now (kept for struct-layout stability)"* (`blend.h:470-477`). That rationale does
not hold — `blend_data` is heap-allocated by core and IOPs are rebuilt from the
same tree, so there is no ABI to preserve. And two of those permanently-NULL
fields are still dereferenced on a live path:

- `blend_gui.c:1670-1672` calls
  `gtk_widget_set_sensitive(GTK_WIDGET(data->channel_boost_factor_slider), …)`
  and `dt_bauhaus_slider_set(…)` unguarded, on a field the header says is always
  NULL.
- `blend_gui.c:1344` calls
  `gtk_notebook_get_tab_label(bd->channel_tabs, gtk_notebook_get_nth_page(bd->channel_tabs, tab))`
  — and `bd->channel_tabs` is never assigned anywhere in the tree (grep shows
  reads only, plus `channel_tabs_csp`, which is a different field).

Both sit under `_blendop_blendif_update_tab` (`blend_gui.c:1576`), which is still
called from `blend_color_picker_apply` (`2166`) and from `1312`. Neither should
crash — `DT_BAUHAUS_WIDGET(NULL)` returns NULL and the function bails, `gtk_*`
emit `g_return_if_fail` — but each pass should print several GTK criticals. This
was **not run**, so treat it as "read from the code, needs a console check": pick
a color from a parametric row's picker and watch stderr. If confirmed, the fix is
to delete the classic-tab editor functions along with the fields, not to add NULL
guards.

Also fully dead: `blendif_section`, `blendif_header`, `masks_import_label`
(declaration + comment only); `masks_param_op`, `masks_param_op_box`,
`masks_elements_header`, `masks_elements_box`, `masks_elements_title`
(declaration + one `= NULL` assignment in cleanup, nothing else).

## 7. Branch-only scaffolding in core init -- WITHDRAWN

**This finding was wrong; do not act on it.** `flexi_test_mode` defaulting to
TRUE is a deliberate safeguard that keeps an experimental masks branch away from
a tester's real photo library, for testers who just double-click the app and
never pass a flag. It stays as it is. The original text is kept below only as a
record of what was claimed.


`common/darktable.c` gains ~157 lines of `--flexi-test-mode`: a **default-on**
redirect of configdir, `library.db` and XMP writes to a hardcoded
`/tmp/flexi_mask_test`, plus a recursive directory copier. It works and it is
well documented, but it is development scaffolding hardwired into the
application's core startup, and it silently changes what `--configdir` and
`--library` mean. It must not reach a PR. If the seeding-and-scratch behavior is
worth keeping, it belongs behind an opt-in flag that defaults *off*, and the
copier belongs in `common/file_location.c`.

On the positive side, the branch does add the seven new preference entries to
`data/darktableconfig.xml.in`, which AGENTS.md requires and which is the step
most changes of this size miss.

## 8. Smaller items

- **Style** (AGENTS.md "lines under 90 characters"): 424 added lines exceed 90
  columns, 334 of them in `blend_gui.c`.
- **Non-ASCII punctuation** (AGENTS.md "Plain ASCII punctuation"): 15 added lines
  use en/em dashes, ellipses, middle dots or arrows. Four are inside `_()`
  strings, which matters beyond style because it fixes the msgid:
  `blend_gui.c:3859` ("- click to toggle"), `7492` ("save current layout as
  preset..."), `10445` ("empty group - select it..."), and the middle-dot
  separator in group captions.
- **`dt_masks_state_t` bit ordering** (`masks.h`): new bits are declared out of
  numeric order (8, 9, 10, 11, 12, 15, 13, 14, 17, 16), which makes "what's the
  next free bit" a search rather than a glance. The mutual-exclusion invariants
  (`DT_MASKS_STATE_WITHIN` = exactly one of three; `DT_MASKS_STATE_OP_COMBINE` =
  exactly one of seven) are documented but unenforced — a `_state_set_within()`
  helper plus an assert would be cheap.
- **Group-level field broadcast**: `state`, `refinement`, `name`,
  `group_opacity` are each "broadcast onto every member of the run" by
  hand-rolled loops in ~12 places (e.g. `blend_gui.c:13924-13946`). One
  `_group_broadcast(grp, cid, setter, data)` would make the convention
  enforceable rather than remembered.
- **Test suite not registered**: `src/tests/masking/flexi/` has 30+ XMP cases,
  expected PNGs and a `run.sh`, but `src/tests/CMakeLists.txt` only does
  `add_subdirectory(unittests)` and never references `masking/`. That matches how
  `integration/` works upstream, so it is not wrong, but the suite is invisible
  to CTest and to CI.

## Suggested order of work

1. Snapshot the refinement-bypass state into pipe-local data (§4) — correctness,
   and it also unblocks any later parallelism work.
2. Delete the classic-tab editor remnants and the permanently-NULL fields (§6) —
   small, self-contained, removes live NULL dereferences.
3. Split `_build_masks_list` into reconcile + pack (§1) — the precondition for
   everything else, and it retires four rebuild guard flags.
4. Unify the two group headers and the two shape rows (§2a, §2b), and move the
   property table out of `blend_gui.c` (§2c).
5. Generalize `DT_UI_PANEL_FLEXI` into an extra column and fold it into the width
   arbitration (§3).
6. The style/string sweep (§8). (§7 is withdrawn -- `flexi_test_mode` stays.)

Steps 3-5 are what make the branch reviewable upstream; 1-2 are worth doing
regardless of whether it ever leaves this fork.

## What was not verified

The branch was not built and darktable was not run, so the GTK-critical claims in
§6 and the width-arbitration consequence in §3 are read from the code, not
observed. `src/tests/masking/flexi/run.sh` and the integration suite were not
run, so there is no statement here about pixel-level behavior. Every other claim
is a grep or a read of the cited line.
