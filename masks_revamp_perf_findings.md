# masks_revamp — caching & responsiveness: open items

Living document. Tracks the caching / UI-responsiveness work on the `masks_revamp` (Flexi mask)
branch that is **not yet done**, in two buckets:

1. **[§1] Upstreamable to `master`** — fixes that exist on this branch (or are specified here)
   for bugs that `master` also has.
2. **[§2] Still missing** — work not started or only half done on the branch.

Items already fixed *and* already on `master` have been removed; see git history of this file
if you need the old diagnoses.

**Verified against `master` at `52435b9a0c` (2026-09-06)**, which is an ancestor of
`masks_revamp` — so every line reference below was read out of the current tree. Re-read the
`master` code before acting on any §1 item: two earlier candidates in this file were overtaken
by upstream rewrites that solved the same bug a different way.

**Status legend:** 🎁 fix exists on the branch, ready to port · 🔬 root cause confirmed, fix not
written · 🧭 under investigation · 💡 proposed, not started · ◐ half done

**Scope tag:** `[masks]` = owned by masks_revamp · `[core]` = pre-existing darktable
pixelpipe/cache behaviour surfaced during this work.

---

## 1. Upstreamable to `master`

### U1 🎁 `[core]` `usedetails` flushes nearly the whole pipe on every `synch_all`
**The bug on `master`:** `dt_dev_pixelpipe_synch_all` calls `dt_dev_clear_scharr_mask(pipe)` and
resets `want_detail_mask = FALSE` at its top
([pixelpipe_hb.c:776-777](src/develop/pixelpipe_hb.c#L776)); history replay then re-requests the
detail mask in `_dev_pixelpipe_synch`, which (flag now false) calls
`dt_dev_pixelpipe_cache_invalidate_later(pipe, gen ? gen->iop_order : 0, "usedetails ")`
([pixelpipe_hb.c:664-668](src/develop/pixelpipe_hb.c#L664)). `gen` is `demosaic` for raws, which
sits near the head of the pipe, so this is very nearly a full flush. The in-code comment already
flags it ("Can this somehow be avoided?").

**Trigger:** any masked module with global `bp->details` ≠ 0. Cost is real: on the reporting
user's XMP, toggling the mask overlay recomputed the whole pipe (incl. `colorin` and an upstream
`atrous` at ~0.9 s) on *every* toggle and *every* mask edit — ~1.5 s → ~0.4 s once fixed.

**The fix (on the branch, `pixelpipe_hb.c` / `.h`):** stop freeing the scharr in `synch_all`;
suppress the per-module `usedetails` flush during replay behind a new `pipe->synch_no_detail_invalidate`
guard; after replay, decide from the **actual scharr-buffer presence** and flush at order 0 only
when the buffer is *needed but missing* (`"usedetails build "`) or *no longer needed but present*
(`"usedetails drop "`). A mere detail-threshold slider move touches neither — the mask hash
changes, so the masked module invalidates on its own, with no full flush.

**Why it's safe:** the scharr only changes when its producer reprocesses. `rawprepare` writes it
WB-independently (`rawmode=FALSE`, [rawprepare.c:445](src/iop/rawprepare.c#L445)) and `demosaic`
is downstream of `temperature`, so no scharr input can change without the producer rerunning and
rewriting it. Per-piece distortion caches are still dropped each `synch_all` (hash-guarded); only
the scharr buffer is preserved.

**Two traps for whoever ports this** — both were hit here:
- **Do not just skip the flush.** `synch_all`'s scharr clear and the flush that follows are a
  **load-bearing pair**: the flush is what invalidates the producer's cacheline and forces it to
  regenerate the buffer it just freed. Skipping only the flush leaves `pipe->scharr.data == NULL`
  forever, and per-shape detail refinement then reports `dt_control_log("detail mask blending
  error")` ([blend.c:271,302](src/develop/blend.c#L271)). Repro: set a per-shape *details
  threshold*, then move a shape handle. You must remove the clear as well.
- **Do not decide from a cross-`synch_all` `want_detail_mask` compare.** It flickers mid-drag and
  is reset by node rebuilds, which caused spurious full flushes on every slider move in the first
  cut. Test the buffer, not the flag.

**Not yet verified at runtime:** neutrality of detail masks, and absence of the "detail mask
blending error" above. Do this before opening the PR.

### U2 🔬 `[core]` `toneequal` invalidates its whole downstream tail on every overlay toggle
Pre-existing on `master`, unrelated to masks — it only surfaced here once U1 removed the other
causes, and it is now the residual cost in the ~1.5 s mask-overlay toggle.

`toneequal` caches its luminance mask in GUI state and, when it deems it stale
(`saved_hash != hash || !luminance_valid`, [toneequal.c:1108/1123](src/iop/toneequal.c#L1108)),
recomputes it and calls `dt_dev_pixelpipe_cache_invalidate_later(piece->pipe, self->iop_order,
"toneequal: ")` ([toneequal.c:1140](src/iop/toneequal.c#L1140)) — wiping every cacheline
downstream of iop_order 3000. Toggling a *downstream* module's mask overlay cannot change
toneequal's input, so the recompute is spurious.

**Proven by bisection:** with `toneequal` disabled, the same toggle invalidates only
`refresh: after ioporder=6200` (the focused module and its tail), serves the rest from cache, and
the hit rate goes 0.00 → 0.75.

**Still to pin down:** which of the two triggers fires — a spurious `luminance_valid` reset (set
by `invalidate_luminance_cache`, [toneequal.c:624](src/iop/toneequal.c#L624), called from
`gui_update` and the auto-adjust quads) or a genuinely volatile `hash`. Fix belongs in its own
PR, not the masks work.

### U3 ✔ `[core/masks]` Per-module rendered-mask cache — CPU and OpenCL
A module re-rasterizes its drawn mask (`dt_masks_group_render_roi`) from scratch every time it
reprocesses, even when the mask is unchanged. True on `master` too, so the concept upstreams;
the exact hooks here are branch-shaped.

**Done:** `piece->drawn_mask_cache` memoizes the raw render output, keyed on
`dt_masks_group_hash_ext(form, pipe->forms) + refine-bypass hash + roi_out + mask_mode`, with
`src_hash = pipe->scharr.hash`. Cleared in `_clear_piece_mask_caches` (piece destroy + scharr
rewrite), which bounds memory. Both blend paths go through one function,
`_render_drawn_mask_cached()` ([blend.c:772](src/develop/blend.c#L772)), called from
`dt_develop_blend_process` ([:1031](src/develop/blend.c#L1031)) and
`dt_develop_blend_process_cl` ([:1647](src/develop/blend.c#L1647)) — the group renderer runs on
the host in both pipes, so a cached buffer is valid for either, and sharing the function is what
keeps a CPU/OpenCL divergence from creeping in. Other files:
[pixelpipe_hb.h:101](src/develop/pixelpipe_hb.h#L101),
[pixelpipe_hb.c:3822](src/develop/pixelpipe_hb.c#L3822).

On the CL path the cache sits after the guide readback, which only happens when
`_group_needs_host_guides()` is true — exactly the case the cache declines to serve, so the two
never interact; a failed readback also leaves `cacheable` false.

**It was inert in the darkroom until 2026-09-07, on both paths.**
`dt_dev_pixelpipe_synch_all` cleared every piece's mask caches wholesale
([pixelpipe_hb.c:883](src/develop/pixelpipe_hb.c#L883)), and a synch_all runs before essentially
every interactive render — every history change, every mask edit, every overlay toggle. An
instrumented darkroom session (10 full-pipe `blend with form CL0` renders at a fixed roi, mask
overlay on) logged **zero** hits. The blanket clear was written for the *distortion* caches,
which the comment there calls "hash-guarded and cheap"; `drawn_mask_cache` was added to the same
helper and inherited it, though it is the one cache whose entire premise is that refilling it is
not cheap. Three changes came out of that:

- `_clear_piece_distortion_caches()` splits detail+raster from drawn; synch_all clears only the
  first two. The drawn cache is still freed with the piece and on scharr clear.
- surviving synch_all means the buffer is *retained* (~183 MB per masked piece at 45 MP), so it
  now allocates through `dt_dev_pixelpipe_prepare_mask_cache()` /
  `dt_dev_pixelpipe_clear_mask_cache()` (the former statics, now exported) instead of raw
  `dt_alloc_align_float`. It is therefore counted in `pipe->mask_cache_size`, visible to the
  pipe-cache trimming, and honours the `_use_mask_cache()` low-memory opt-out (≥ 6 GB available)
  that the detail and raster caches already obeyed.
- the key now resolves group members through the *pipe's* form list
  (`dt_masks_group_hash_ext`, new) rather than `darktable.develop`'s. An unresolvable member
  contributes nothing to the hash, which then collapses to the group's own
  type/formid/version/source and stops tracking the mask — the trap recorded in
  `verify.c:837`. Harmless while synch_all wiped the cache every render; not harmless now, and
  it also covers second-window pinned devs and exports of an image other than the open one.

**Correctness argument:** the render output depends on module
*pixels* only via guided-filter feathering and parametric-as-form members, both gated out by
`!_group_needs_host_guides(form, piece)` (no cheap stable hash for those guides); and via
per-shape *details* refinement, which depends on the scharr buffer and is captured by `src_hash`.
Global post-ops (feather/blur/tone/global details) and invert are applied *after* the cached
point, so they run fresh and need not be in the key; global blend opacity is applied later too,
so opacity slides reuse the mask. `suppress_mask`/`uniform` short-circuit before the render
block.

**Verified (macOS, OpenCL on an available GPU device):**

- `src/tests/masking/flexi/run.sh` (CPU): 44/46, unchanged. The two failures, `F1`/`F2`,
  reproduce identically with the change stashed out — the documented JzCzhz build variance, not
  this change.
- All 44 fixture XMPs re-rendered through `darktable-cli` with `opencl=TRUE`, before and after
  the change: **byte-identical**. That covers the CL miss path only, since one export renders
  each piece once.
- The **CL hit path** is exercised by `--verify-masks`, whose `_render_mask_cl()`
  ([verify.c:530](src/develop/masks/verify.c#L530)) calls `dt_develop_blend_process_cl` on the
  same `piece` once per replayed edit. Over the 49-edit fixture harvest: `CPU vs GPU gap,
  migrated: 1.01e-06`, `edits where migration widened that gap by >1/255: 0`, `CLASSIC CHANGED:
  0`, and the 3 DIFFERENT are the known `J5`/`J6`/`J7` `DT_MASKS_REFINE_GROUP` cases. A stale
  buffer served across edits would have shown up as a large CPU-vs-GPU gap.
- `ctest -R flexi` in `build-tests`: 11/11. (`test_filmicrgb` does not link on macOS —
  `ld: unknown options: --wrap=…`, pre-existing and unrelated.)
- All of the above re-run after the synch_all/accounting/hash changes: same numbers.
- **Interactive, both directions**, two instrumented darkroom sessions on the OpenCL pipe:
  - *sliders only* (a downstream module's slider, then exposure's own): 5 `exposure` blend
    renders, **4 hits** — one miss to populate, everything after served. Three on
    `CL0 [full HQ]` and one on `CPU [preview]`, so the shared entry serves both devices and both
    pipes in a real session, not just in the harness.
  - *node dragging*: 124 `dt_masks_events_mouse_moved`, 10 renders, **0 hits**. Every render
    lands immediately after a drag burst ends, i.e. the mask had just changed, so a miss is
    correct at each one. Had the key been missing shape geometry this log would have been all
    hits and the shape would have looked frozen on canvas.
  - cost removed, same roi (5520x8288, one circle, 45 MP): mask ready in **75.5 ms** on the
    miss, **12.0-12.2 ms** on a hit. The residue is the 183 MB memcpy out of the cache.

**Payoff and limits:** spares rasterization when a module reprocesses with an unchanged mask —
chiefly while the mask overlay is shown (pipe cache is off downstream of focus, so every
downstream masked module re-renders each frame) and when a non-mask slider on a masked module
moves. It does **not** remove the downstream *pixel* processing under mask overlay; that is C1,
and it is by design.

### U4 💡 `[core/masks]` On-device (OpenCL) mask compositing / feather
Mask rendering is CPU-only even on the OpenCL pipe (C3). In the reporting user's profile the
masked `exposure.2` + `exposure` cost ~0.8 s wall / ~7 s CPU **each** in mask compositing,
dominating a ~2.8 s recompute, while `agx` itself was ~0.17 s on GPU.

Move the group fold and the guided-filter feather onto the GPU. Large effort. The group-fold
*operators* are branch-specific, but the underlying "masks composite on CPU only" limitation is
`master`'s too, so the core of this upstreams.

### U5 💡 `[core]` Interactive downscaling during slider drag
Process at preview scale while dragging, full resolution on release.

---

## 2. Still missing (branch-only)

### N1 💡 `[masks]` `_reconcile_raster_form_users` runs once per replayed history item
An instrumented run showed ~40 identical `(agx → exposure.2)` reconcile calls per commit — one
per history item replayed by `synch_all`, times the pipes. Now cheap (hash lookups, no
invalidation), but redundant: reconciliation only needs to run once per module per synch, from
the final committed state. Low priority; do it if `synch_all` ever shows up in a profile.

### N2 💡 `[masks]` Fine-grained widget-diff reconciliation of `_build_masks_list`
The panel currently skips a rebuild entirely when `_masks_list_signature` is unchanged, which
covers the common case. The full per-widget reuse/move/destroy diff is still unwritten, and needs
interactive GTK testing (DnD / revealer / parametric-editor lifecycles) to be worth attempting.

### N3 💡 `[masks]` Remaining plan follow-ups
- slider-drag history debounce (plan item 2.2)
- skip the mask-manager *lib* rebuild that every `dt_dev_add_masks_history_item` still triggers
  in flexi mode (2.4)
- consolidate the direct/deferred rebuild call sites (3.4)

---

## 3. Context needed to read the above

### C1 `[core]` The mask overlay disables the pipe cache entirely — by design
With `pipe->mask_display` set, `dt_dev_pixelpipe_cache_available` returns FALSE
([pixelpipe_cache.c:178](src/develop/pixelpipe_cache.c#L178)) and cachelines are stored with
`DT_INVALID_HASH` ([:349-350](src/develop/pixelpipe_cache.c#L349)). So with the overlay ON,
returning to a previous value can never be a cache hit. Intentional: `pipe->mask_display` is
excluded from the piece hash. Do not "fix" this; work around it (U3).

### C2 `[core]` How the full-pipe cache budget is actually computed
Full pipe = 64 lines (`darktable.pipe_cache ? 64 : DT_PIPECACHE_MIN`) bounded by
`dt_get_available_mem() / cache->mem_fraction`, with `mem_fraction = 8` for the FULL pipe
([pixelpipe_hb.c:263](src/develop/pixelpipe_hb.c#L263),
[pixelpipe_cache.c:506,532](src/develop/pixelpipe_cache.c#L506)); `checkmem` evicts oldest lines
over budget. Each full-res RGBA-float buffer is large (6984×4660 ≈ 520 MB), so deep pipes churn.

`dt_get_available_mem()` is `MAX(512MB, (total_memory - cl_uni_memory)/1024 * fractions[4*level + 0])`
([darktable.c:2590-2599](src/common/darktable.c#L2590)) — the **first** number of the resource
tuple (the third is `_get_mipmap_size`, which no longer feeds the pipe cache at all). Index 0 is
`128 / 512 / 700 / 16384` for small/default/large/unrestricted
([darktable.c:1937-1942](src/common/darktable.c#L1937)), so **default → large does raise the pipe
cache budget**, by ~1.37×.
- Tuning without code: with the app closed, raise the **1st** number in `darktablerc`, e.g.
  `resource_large=900 16 128 900`, then select `large`.

### C3 `[core]` With OpenCL, intermediate GPU outputs aren't host-cached
Device buffers aren't copied back for the cache except the focused module's pinned input
([pixelpipe_hb.c:2661-2708](src/develop/pixelpipe_hb.c#L2661)), so the pipe re-executes
top-to-bottom on each edit. Cheap for GPU modules, expensive for the CPU-side mask work in U4.

---

## 4. Diagnostic playbook

- **Per-module timing:** `darktable -d perf` → `processed <module> … took Ns`.
- **Cache decisions:** `darktable -d pipe` → `cache HIT`, `importance hints … focus
  important_in`, `pipe cache check … Freed: invalid NMB`, and crucially
  `pipecache invalidate|flush <reason>` — the reason string names the invalidator
  (`blend new raster:`, `set raster:`, `refresh:`, `usedetails `, `toneequal: `).
- **Memory eviction:** `-d pipe -d memory` → `pipe cache check … limit=NMB`.
- **Isolate masks vs pipe:** move a *non-mask* slider on the same module A→B→A; if it lags
  identically, the cost is the pipe cache, not the mask code.
- **Isolate the detail path (U1):** set all `details` to 0; if the lag drops, U1 is involved.
- **Isolate toneequal (U2):** disable `toneequal`; if the invalidation collapses to
  `refresh: after ioporder=<focused>`, U2 is involved.
