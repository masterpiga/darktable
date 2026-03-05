# masks_revamp — caching & responsiveness findings

Living document. Tracks every caching / UI-responsiveness issue found while hardening the
`masks_revamp` (Flexi mask) branch, plus the fixes landed and the ones still proposed.
Update as new findings appear.

**Status legend:** ✅ fixed this session · 🔬 root cause confirmed, fix proposed · 🧭 under
investigation · 💡 proposed optimization (not started) · ℹ️ pre-existing context (not a
regression)

**Scope tag:** `[masks]` = introduced/owned by masks_revamp · `[core]` = pre-existing
darktable pixelpipe/cache behaviour surfaced during this work.

**⬆ Upstream impact** — every caching/efficiency/responsiveness item records whether the
affected code also exists on `master`, i.e. whether the fix is worth upstreaming:
- **⬆ YES** — the code/bug exists on `master`; fix applies there too.
- **⬆ no** — branch-only (the code doesn't exist on `master`).
- **⬆ n/a** — context/observation, not a fix.

---

## 0. TL;DR of the live investigation — ROOT CAUSE FOUND & FIXED

A user reported 2–3 s lag toggling a mask value (feathering on an `agx` mask, `agx` last in a
deep pipe) even returning to a previously-computed value. Profiling (`-d perf`, `-d pipe`)
established that the whole pipe re-executed on every mask edit, with
`pipe cache check … Freed: invalid 4933MB` — the entire pipe cache explicitly invalidated on
every commit. The focused-module input pin worked (`importance hints … agx … focus
important_in`), but the pinned buffer was invalidated before it could be reused.

`-d pipe` named the culprit:

```
pipecache invalidate [full HQ]  blend new raster: 7 cachelines after ioporder=2700, blend cache
pipecache invalidate [full HQ]  blend new raster: 6 cachelines after ioporder=2600, blend cache
```

`exposure.1` is ioporder 2500, so 2600/2700 are `exposure.2`/`exposure` — the raster mask
**sources**. Invalidating "after ioporder=2600" wiped the whole tail, which is exactly why
every recompute restarted at `exposure.2`.

**Two independent bugs, both fixed; both made every mask commit report a bogus "new raster
user", invalidating the pipe cache downstream of the raster source:**
1. **[A8]** `[masks]` — a `GINT_TO_POINTER(0)`/NULL truthiness bug in the raster-**form**
   reconciliation. Fixing it made *whole-mask* refinement instant.
2. **[A9]** `[core, ⬆ upstreamable]` — the **legacy** raster path returned `candidate`
   unconditionally, ignoring its own `new` flag. This only bit the `synch_all` path (full
   history replay), which is why *group-level* refinement stayed slow after A8. Fixing it made
   group-level refinement snappy too.

Verified by instrumentation: exactly one legitimate `NEW (invalidates)` at startup, `already`
thereafter. **[A10]** records a plausible-but-wrong hypothesis that the log disproved.

---

## A. Fixed this session

### A11 ❌ `[core]` Attempted `usedetails` flush-skip — **REVERTED (caused "detail mask blending error")**
**Attempt:** skip the `usedetails` order-0 full flush during `synch_all` when the detail
requirement is unchanged; invalidate once at the end only on a `want_detail_mask` toggle
(guard `pipe->synch_no_detail_invalidate`, new struct field).
**Why it broke:** `synch_all` *also* calls `dt_dev_clear_scharr_mask(pipe)` unconditionally at
its top ([pixelpipe_hb.c:692](src/develop/pixelpipe_hb.c#L692)), which **frees**
`pipe->scharr.data`. That buffer is (re)produced *only* when its producer module
(demosaic / rawprepare, `IOP_FLAGS_WRITE_DETAILS`) runs `process` →
`dt_dev_write_scharr_mask`. On `master` the order-0 flush that fires right after the clear is
what invalidates the producer's cacheline and forces it to reprocess and **regenerate** the
scharr buffer. The clear and the flush are a **load-bearing pair**. Removing the flush in the
steady state left the buffer freed but never regenerated → the per-shape detail refinement read
`p->scharr.data == NULL` → `dt_control_log("detail mask blending error")`
([blend.c:271,302](src/develop/blend.c#L271)). Repro: set a per-shape *details threshold*, then
move a shape handle (→ `synch_all`).
**Reverted** to master behaviour (`git checkout` of `pixelpipe_hb.c/.h`; they carried only this
change).
**Why C2 is not a one-line flush-skip (for whoever retries D2):** the scharr content depends on
producer-stage inputs (raw data, ROI, `dsc.temperature.coeffs` / WB, `rawmode`). When any of
those change the producer reprocesses on its own (hash miss) and rewrites the buffer, so the
`synch_all` clear is *usually* redundant — **but** there are cases (notably sraw scharr written
by `rawprepare`, whose params don't include WB, on a WB change) where the producer may not
reprocess yet the scharr must change; master's clear+flush covers those conservatively. A sound
C2 must (a) stop unconditionally clearing the scharr in `synch_all`, and (b) add a real
"scharr is stale" test (or force just the producer piece to reprocess) rather than skipping the
flush. That's an investigation, not a quick fix — deferred. The perf cost it targets only bites
when detail refinement is actively in use.

### A14 ✅ `[core]` `usedetails` order-0 flush on every `synch_all` — **FIXED CORRECTLY** — **⬆ YES** — *implements C2/D2*
The real cause of the **~2 s mask-overlay toggle** lag once a detail mask is in use (found via the
user's XMP + `-d perf`): toggling the overlay calls `dt_iop_refresh_center` → sets
`DT_DEV_PIPE_SYNCH` → `synch_all`, which reset `want_detail_mask=FALSE`; replay then re-requested it
via `dt_dev_pixelpipe_usedetails` → `cache_invalidate_later(pipe, 0, "usedetails ")` = **full-pipe
flush**. The whole pipe (incl. `colorin`, upstream `atrous` ~0.9 s) recomputed on every toggle *and*
every mask edit. (This is why A11's naive skip broke: `synch_all` *also* freed the scharr and relied
on this flush to rebuild it.)
**Correct fix:** stop freeing the scharr in `synch_all`; suppress the per-module `usedetails` flush
during replay (guard `pipe->synch_no_detail_invalidate`); after replay decide from the **actual
scharr-buffer presence** (NOT a cross-`synch_all` `want_detail_mask` compare — that flickers mid-drag
and is reset by node rebuilds, which caused spurious per-slider-move full flushes in the first cut):
flush at order 0 only when the buffer is *needed but missing* (`"usedetails build "` — first process /
after a node rebuild) or *no longer needed but present* (`"usedetails drop "` — clear it). A mere
detail-threshold slider move touches neither (scharr unchanged): the mask hash changes, so the masked
module invalidates on its own — no full flush. Safe because the scharr only changes when its
producer reprocesses: **rawprepare writes it WB-independently** (`rawmode=FALSE` →
[rawprepare.c:445](src/iop/rawprepare.c#L445)) and **demosaic is downstream of `temperature`**, so no
scharr input can change without the producer reprocessing (and rewriting it). Per-piece distortion
caches are still dropped each `synch_all` (hash-guarded); only the scharr buffer is preserved.
Files: [pixelpipe_hb.c](src/develop/pixelpipe_hb.c) (`dt_dev_pixelpipe_synch_all`,
`dt_dev_pixelpipe_usedetails`, init), [pixelpipe_hb.h](src/develop/pixelpipe_hb.h) (guard field).
**Effect:** overlay toggle drops from full-pipe (~1.5 s) to just the focused module + its tail
(`agx` ~0.4 s); `atrous`/`colorin` stay cached. **Upstream: YES** (verbatim on master).
**NEEDS runtime verify:** neutrality of detail masks + no "detail mask blending error" (the A11 failure).

### A12 ✅ `[core]` Raised the full-pipe cache budget (`mipmap_memory/4 → /2`) — **⬆ YES** — *implements D3*
`dt_dev_pixelpipe_init` capped the FULL pipe cache at `MAX(64MB, mipmap_memory/4)`
([pixelpipe_hb.c:254](src/develop/pixelpipe_hb.c#L254)). For large images a single full-res RGBA
intermediate is ~0.5 GB, so `/4` held barely one or two → deep pipes churned and A→B→A was rarely a
hit. Doubled to `/2` so ~2× more intermediates survive between edits. One-token change; the line is
verbatim on `master`. Trade-off: more RAM for the pipe cache (bounded by the resource level).
**Upstream: YES** (could also be exposed as a dedicated setting rather than hard-coded).

### A13 ✅ `[masks/core]` Per-module rasterized drawn-mask cache — **⬆ YES (concept)** — *implements D4 (CPU path)*
A module re-rasterizes its drawn mask (`dt_masks_group_render_roi`) from scratch every time it
(re)processes, even when the mask is unchanged — the standing cost behind B4. Added
`piece->drawn_mask_cache` (reusing `dt_dev_distorted_mask_cache_t`) that memoizes the raw render
output. **Key = `dt_masks_group_hash(form) + roi_out`; src_hash = `pipe->scharr.hash`.**
**Correctness (the part that bit C2 — verified here):** the render output depends on module *pixels*
only via (a) guided-filter feathering and (b) parametric-as-form members — both gated out by
`!_group_needs_host_guides(form, piece)` (those guides have no cheap stable hash); and (c) per-shape
*details* refinement, which depends on the scharr buffer — captured by `src_hash`. Global post-ops
(feather/blur/tone/**global** details) and invert are applied *after* the cached point, so they run
fresh and need not be in the key; global blend opacity is applied later too (so opacity slides reuse
the mask). `suppress_mask`/`uniform` short-circuit before the render block, so no interaction.
Cache is cleared in `_clear_piece_mask_caches` (piece destroy + scharr rewrite), bounding memory.
**Payoff:** spares rasterization when a module reprocesses with an unchanged mask — chiefly while the
**mask overlay is shown** (pipe cache disabled downstream of focus, so every downstream masked module
re-renders each frame) and when a **non-mask slider on a masked module** moves. It does *not* remove
the downstream *pixel* processing under mask overlay (that's B1, separate).
File: [src/develop/blend.c](src/develop/blend.c) (CPU `dt_develop_blend_process`),
[src/develop/pixelpipe_hb.h](src/develop/pixelpipe_hb.h)/[.c](src/develop/pixelpipe_hb.c).
**TODO:** mirror into the OpenCL blend path (`dt_develop_blend_process_cl`, ~[blend.c:1300](src/develop/blend.c#L1300));
verify neutrality (plain drawn mask renders byte-identical). **Upstream: concept YES** (master also
re-rasterizes every process); the exact hooks are branch-shaped.

### A10 ❌ `[masks]` "synch_all replay transiently unregisters raster users" — **HYPOTHESIS DISPROVEN**
Recorded so nobody re-derives it. After A8, group-level refinement (but not whole-mask) still
invalidated. The hypothesis was that `synch_all`'s full-history replay transiently *unregistered*
the raster user (via `_reconcile_raster_form_users`'s else-branch with `grp == NULL`, or the
legacy remove at [imageop.c:2063](src/develop/imageop.c#L2063)), so the final re-registration
looked "new".
**Instrumentation disproved it.** A `-d pipe -d masks -d verbose` run showed exactly **one**
`raster form register … -> NEW (invalidates)` (the legitimate first registration at startup),
then `present=1 old=0 -> already` on every subsequent call, and **zero**
`raster form unregister` lines. Registration is stable across replay.
**Actual cause: A9.** The residual invalidation came from the legacy path's unconditional
`return candidate;`, which fires for any module carrying a legacy raster sink on every
*full-history* replay (`synch_all`) but not on `synch_top` (which commits only the top item, and
`agx` has no legacy sink) — precisely the whole-mask vs. group-level asymmetry.
**Confirmed fixed:** user reports group-level feathering is now snappy.

### D9 💡 `[masks]` `_reconcile_raster_form_users` runs once per replayed history item — **⬆ no**
The instrumented run showed ~40 identical `(agx → exposure.2)` reconcile calls per commit — one
per history item replayed by `synch_all`, times the pipes. Now cheap (hash lookups, no
invalidation), but redundant: reconciliation only needs to run once per module per synch, from
the final committed state. Low priority; worth doing if `synch_all` ever shows up in a profile.

### A9 ✅ `[core]` `dt_iop_commit_blend_params` reported a "new raster" on every commit — **⬆ YES** — *the group-level fix*
The legacy raster-sink path computed `const gboolean new = g_hash_table_insert(...)` but used it
only for a debug print, then did `return candidate;` **unconditionally**. The caller
(`dt_iop_commit_params`) treats a non-NULL return as "a source gained a new user" and calls
`dt_dev_pixelpipe_cache_invalidate_later(pipe, new_raster->iop_order, "blend new raster: ")` —
so **any module using a legacy raster mask invalidated its source's downstream cache on every
history commit**. It also discarded `_reconcile_raster_form_users`'s return, so a genuinely-new
raster *form* source never invalidated when a legacy sink was also present.
**Fix:** return `candidate` only when the registration is genuinely `new`; otherwise return the
form-reconcile result; when both are new, report the one with the **earlier** `iop_order` so the
invalidation covers both.
File: [src/develop/imageop.c](src/develop/imageop.c) (`dt_iop_commit_blend_params`).
**Upstream: YES.** `master` has the identical `new`-computed-then-ignored + `return candidate;`
pattern and invalidates on it. Master gates the invalidation on
`blendop_params->mask_mode & DEVELOP_MASK_RASTER` — true for exactly the raster users that hit
this path — so **master wipes the pipe cache downstream of a raster source on every history
commit**. Strong upstream candidate.

#### Exact upstream patch (against `master`)
On this branch the fix spans `imageop.c:2068-2081`, but most of that is branch-only
(`_reconcile_raster_form_users` / the `form_raster` iop_order preference). The upstreamable
essence is the `if(!new)` guard. On `master` it is a **one-liner at `imageop.c:1962`**:
```diff
-        return candidate;
+        // Only report a *genuinely new* registration. The caller uses a non-NULL
+        // return to invalidate the pipe cache downstream of the source; returning
+        // `candidate` unconditionally wipes that cache on every history commit.
+        return new ? candidate : NULL;
```
Safety verified on `master`: the return value is captured only at `imageop.c:2186`
(`new_raster`) and consumed only at `imageop.c:2241-2242` (the invalidation); the other call
sites (82, 431, 2330) discard it. Master's own doc comment (`imageop.c:1920-1921`) already
specifies the function "either returns NULL or the source module".
*Reviewer caveat for the commit message:* `g_hash_table_insert` reports **key** novelty, not
value change, so a *retarget* to a different `raster_mask_id` on the same source would not
invalidate. This cannot occur in practice — a source exports exactly one raster slot
(`BLEND_RASTER_ID == 0`), so `raster_mask_id` is effectively constant.

### A8 ✅ `[masks]` Raster-form reconciliation invalidated the pipe cache on every commit — **⬆ no**
`_reconcile_raster_form_users` decided "is this raster user already registered?" with:
```c
const gpointer old_value = g_hash_table_lookup(cand->raster_mask.source.users, module);
const gboolean already = old_value && GPOINTER_TO_INT(old_value) == (int)want;
```
`want` is *always* `BLEND_RASTER_ID == 0` for a raster form (the function's own comment says
so), and `GINT_TO_POINTER(0)` is **NULL** — so `g_hash_table_lookup` returns NULL whether the
key is absent or present-with-value-0. `already` was therefore **always FALSE**, making every
commit report a new raster user → `dt_dev_pixelpipe_cache_invalidate_later(pipe,
new_raster->iop_order, "blend new raster: ")` → the whole pipe downstream of the raster source
invalidated on *every mask edit*.
**Fix:** probe key presence explicitly with `g_hash_table_lookup_extended` so a stored value of
0 is distinguishable from an absent key.
File: [src/develop/imageop.c](src/develop/imageop.c) (`_reconcile_raster_form_users`).
**Upstream: no.** `_reconcile_raster_form_users` is branch-only (raster-as-form). `master`
registers via `g_hash_table_insert`'s *return value* (imageop.c:1951), which correctly reports
"key newly inserted" and never hits this trap. (Minor latent difference: master's form can't
detect a *retarget* — value change on an existing key — only new keys; not a live bug there
since a retarget changes the source module, hence the hash key.)
*Consequence:* the `set raster:` invalidations seen alongside are secondary —
`dt_iop_piece_set_raster` only invalidates when the source actually reprocesses and rewrites
its mask ([imageop.c:3734](src/develop/imageop.c#L3734)). With A8 fixed the sources stay cached,
don't reprocess, and don't rewrite. Correctness is preserved: genuinely editing a source's mask
changes its own hash → it reprocesses → `set raster:` correctly invalidates downstream.

### A1 ✅ `[masks]` Per-shape/group refinement missing from the render cache hash — **⬆ no**
`dt_masks_group_hash` hashed a group point's `state` + `opacity` but not `refinement`, so
refinement edits (blur/feather/contrast/details/brightness) did not change any `piece->hash`
and could be served stale. **Fix:** hash `grpt->refinement` alongside state/opacity.
File: [src/develop/masks/masks.c](src/develop/masks/masks.c) (`dt_masks_group_hash`).
**Upstream: no** — `dt_masks_refinement_t` does not exist on `master` (per-shape refinement is
a branch feature).

### A2 ✅ `[masks]` Parametric-as-form & per-group feather broken on the OpenCL pipe — **⬆ no**
On the GPU pipe `blend_refine_guide_in/out` were NULL, so parametric-form masks rendered a
uniform 1.0 and per-group guided-filter feathering was silently skipped (CPU/GPU divergence).
**Fix:** predicate `_group_needs_host_guides` + read the guide images back to host
(`dt_opencl_copy_image_to_host`) only when a group needs them; plain drawn shapes keep the
no-readback fast path.
File: [src/develop/blend.c](src/develop/blend.c) (`dt_develop_blend_process_cl`).
**Upstream: no** — `blend_refine_guide_in/out` and `DT_MASKS_PARAMETRIC` are branch-only.

### A3 ✅ `[masks]` Multi-form opacity commit fired N history items per gesture — **⬆ no**
`_props_row_apply` looped over every targeted form calling `dt_masks_form_change_opacity`, each
committing a full history item (3-pipe synch + panel rebuild) — N× per drag, ×per tick.
**Fix:** mutate all forms' opacity in place, commit exactly once after the loop.
File: [src/develop/blend_gui.c](src/develop/blend_gui.c) (`_props_row_apply`).
**Upstream: no** — the multi-form loop is flexi-only. (`dt_masks_form_change_opacity` still
self-commits on `master`, which is correct there: it's called once per gesture.)

### A4 ✅ `[masks]` Solo-group did a full panel rebuild — **⬆ no**
`_toggle_solo_group` did a full `_build_masks_list` rebuild where `_toggle_solo_form` already
did a cheap in-place refresh. **Fix:** solo-group now uses `_refresh_all_shape_rows` +
`_sync_solo_canvas_highlight`. Stays persistent + undoable.
File: [src/develop/blend_gui.c](src/develop/blend_gui.c). **Upstream: no** — flexi-only.

### A5 ✅ `[masks]` Per-hover recursive tree walks → O(1) map — **⬆ no**
Every canvas-hover motion ran several full recursive walks of the nested `masks_list_box` tree.
**Fix:** a `formid → row` GHashTable (`bd->masks_row_map`) rebuilt alongside the panel; O(1)
lookups with a tree-walk fallback.
Files: [src/develop/blend_gui.c](src/develop/blend_gui.c), [src/develop/blend.h](src/develop/blend.h).
**Upstream: no** — flexi panel only.

### A6 ✅ `[masks]` Deferred panel rebuilds not de-duplicated — **⬆ no**
One gesture could enqueue several `g_idle_add(_rebuild_masks_list_idle)` full rebuilds.
**Fix:** `_queue_masks_list_rebuild` behind a single `masks_rebuild_pending` guard; 18 call
sites routed through it. **Upstream: no** — flexi panel only.

### A7 ✅ `[masks]` `_build_masks_list` full teardown on every mutation (reconcile-by-skip) — **⬆ no**
**Fix:** `_masks_list_signature` hashes everything the tree is built from; when unchanged the
rebuild is skipped entirely. Hoisted loop-invariant `_group_count` out of the per-group loop
(O(groups×points) → O(points)). Fine-grained per-widget diff deliberately deferred (see D7).
File: [src/develop/blend_gui.c](src/develop/blend_gui.c). **Upstream: no** — flexi panel only.

---

## B. Caching context established during the investigation

### B1 ℹ️ `[core]` The mask overlay disables the pipe cache entirely — **⬆ n/a** (by design)
With `pipe->mask_display` set, `dt_dev_pixelpipe_cache_available` returns FALSE
([pixelpipe_cache.c:182](src/develop/pixelpipe_cache.c#L182)) and cachelines are stored with
`DT_INVALID_HASH` ([:353-354](src/develop/pixelpipe_cache.c#L353)). So with the overlay ON,
returning to a previous value can never be a cache hit. Intentional: `pipe->mask_display` is
excluded from the piece hash ([pixelpipe_hb.c:2066](src/develop/pixelpipe_hb.c#L2066)).

### B2 ℹ️ `[core]` Full-pipe cache is memory-bounded; the resource dropdown barely helps — **⬆ YES**
Full pipe = 64 lines but capped by `memlimit = MAX(64MB, mipmap_memory/4)`
([pixelpipe_hb.c:254](src/develop/pixelpipe_hb.c#L254) — **present verbatim on master**);
`checkmem` evicts oldest lines over budget ([pixelpipe_cache.c:473](src/develop/pixelpipe_cache.c#L473)).
For large images each full-res RGBA-float buffer is huge (6984×4660 ≈ 520 MB), so deep pipes
churn. **Gotcha:** the *resource level* dropdown's `large` uses the **same** mipmap fraction
(128/1024) as `default` — only `small` differs ([darktable.c:1814-1818](src/common/darktable.c#L1814)) —
so default→large does **not** raise this budget. Both facts hold on `master`.
- Workaround (no code): with the app closed, set in `darktablerc` e.g.
  `resource_large=700 16 512 900` (3rd number = mipmap fraction), then pick `large`.

### B3 ℹ️ `[core]` With OpenCL, intermediate GPU outputs aren't host-cached — **⬆ n/a**
Device buffers aren't copied back for the cache except the focused module's pinned input
([pixelpipe_hb.c:2661-2708](src/develop/pixelpipe_hb.c#L2661)), so the pipe re-executes
top-to-bottom each edit (cheap for GPU modules).

### B5 🔬 `[core]` Tone equalizer invalidates the whole tail below it on every overlay toggle — **⬆ YES (pre-existing, not masks_revamp)**
Residual cause of the ~1.5 s mask-overlay-toggle latency after C2/D3/D4 landed. `toneequal`
caches its luminance mask in GUI state and, when it deems it stale
(`saved_hash != hash || !luminance_valid`, [toneequal.c:1109/1125](src/iop/toneequal.c#L1109)),
recomputes it and calls `dt_dev_pixelpipe_cache_invalidate_later(pipe, self->iop_order,
"toneequal: ")` ([toneequal.c:1134](src/iop/toneequal.c#L1134)) — wiping every cacheline
downstream of `toneequal` (iop_order 3000): `colorin → channelmixerrgb → atrous → agx`.
Toggling a *downstream* module's mask overlay (agx) should not change toneequal's input, so the
recompute is spurious/wasteful. `invalidate_luminance_cache` (sets `luminance_valid=FALSE`,
[toneequal.c:622](src/iop/toneequal.c#L622)) is called from `gui_update` + the auto-adjust
quads; the exact toggle trigger (spurious `luminance_valid` reset vs a volatile `hash`) is not
yet pinned down. **Proven by bisection:** with toneequal disabled, the same toggle invalidates
only `refresh: after ioporder=6200` (agx+tail), serves the rest from cache, hit rate 0.00→0.75.
**Pre-existing on master** (stock `toneequal.c`), independent of the Flexi panel; only surfaced
because C2/D3/D4 removed the other causes. Out of scope for the masks hardening; upstreamable as
its own fix. See B4.

### B4 🔬 `[core]` Mask rendering is CPU-only even on the OpenCL pipe — **⬆ YES**
`dt_masks_group_render_roi` always runs on CPU; the result is uploaded to the device
([blend.c:1263](src/develop/blend.c#L1263)). In the user's profile the masked modules
`exposure.2` + `exposure` cost ~0.8 s wall / ~7 s CPU **each** (their CPU mask compositing),
dominating a ~2.8 s recompute — while `agx` itself is ~0.17 s on GPU. Also true on `master`.
See D4/D5.

---

## C. Root-cause findings — fix proposed

### C2 ✅ `[core]` `usedetails` wipes the whole cache on every commit when details are in use — **⬆ YES** — **FIXED, see A14**
(A11 was a failed first attempt — reverted. A14 is the correct fix: preserve the scharr across
`synch_all` so no regeneration flush is needed.)
`synch_all` resets `want_detail_mask = FALSE` ([pixelpipe_hb.c:692](src/develop/pixelpipe_hb.c#L692))
then history replay re-requests it via `dt_dev_pixelpipe_usedetails`, which (flag now false)
calls `dt_dev_pixelpipe_cache_invalidate_later(pipe, 0, "usedetails ")` — a **full flush**
([pixelpipe_hb.c:785](src/develop/pixelpipe_hb.c#L785)). The in-code comment already flags this
("Can this somehow be avoided?"). Trigger: any masked module with global `bp->details` ≠ 0 (on
`master`), or additionally a per-shape `refinement.details` ≠ 0 via `_blend_group_wants_details`
(branch-only extra trigger).
**Not** the current user's lag (confirmed: no `details requested` in their log), but a real
full-cache wipe for anyone using detail refinement.
**Upstream: YES** — `dt_dev_pixelpipe_usedetails` and the `synch_all` reset exist verbatim on
`master`; only the extra branch-side trigger is new.
- **Fix direction:** preserve `want_detail_mask` across `synch_all`; suppress the `usedetails`
  invalidation during history replay and invalidate once at the end only if the detail-mask
  requirement actually toggled.

---

## D. Proposed optimizations (not started)

### D2 ✅ `[core]` Fix `synch_all`/`usedetails` spurious full-flush (implements C2) — **⬆ YES** — **DONE, see A14**
### D3 ✅ `[core]` Raise the full-pipe cache budget for deep pipes — **⬆ YES** — **DONE, see A12**
### D4 ◐ `[core/masks]` Per-module rendered-mask cache — **⬆ YES** — **CPU path DONE (A13); OpenCL path TODO**
### D5 💡 `[core/masks]` On-device (OpenCL) mask compositing / feather — **⬆ YES**
Move the group fold + guided-filter feather onto the GPU so OpenCL actually accelerates masked
modules (B4). Large effort. The group-fold operators are branch-specific, but the underlying
"masks composite on CPU only" limitation is `master`'s too.
### D6 💡 `[core]` Interactive downscaling during slider drag — **⬆ YES**
Process at preview scale while dragging, full-res on release.
### D7 💡 `[masks]` Fine-grained widget-diff reconciliation of `_build_masks_list` — **⬆ no**
The full per-widget reuse/move/destroy diff (beyond the A7 signature-skip). Needs interactive
GTK testing (DnD/revealer/parametric-editor lifecycles).
### D8 💡 `[masks]` Remaining plan follow-ups — **⬆ no**
2.2 slider-drag history debounce · 2.4 skip the mask-manager *lib* rebuild that every
`dt_dev_add_masks_history_item` still triggers in flexi mode · 3.4 direct/deferred rebuild
call-site consolidation.

---

## E. Diagnostic playbook

- **Per-module timing:** `darktable -d perf` → `processed <module> … took Ns`.
- **Cache decisions:** `darktable -d pipe` → `cache HIT`, `importance hints … focus
  important_in`, `pipe cache check … Freed: invalid NMB`, and crucially
  `pipecache invalidate|flush <reason>` — the reason string names the invalidator
  (`blend new raster:`, `set raster:`, `refresh:`, `usedetails `).
- **Memory eviction:** `-d pipe -d memory` → `pipe cache check … limit=NMB`.
- **Isolate masks vs pipe:** move a *non-mask* slider on the same module A→B→A; if it lags
  identically, the cost is the pipe cache, not the mask code.
- **Isolate detail path (C2):** set all `details` to 0; if lag drops, C2 is involved.
- **Verify A8:** with `-d pipe`, moving a mask slider must **no longer** print
  `blend new raster:` on every commit, and the tail (`exposure.2` …) must stop recomputing.
