# masks_revamp — caching & responsiveness: open items

Living document. Tracks the caching / UI-responsiveness work on the `masks_revamp` (Flexi mask)
branch that is **not yet done**, in two buckets:

1. **[§1] Upstreamable to `master`** — fixes that exist on this branch (or are specified here)
   for bugs that `master` also has.
2. **[§2] Still missing** — work not started or only half done on the branch.

Items already fixed *and* already on `master` have been removed; see git history of this file
if you need the old diagnoses.

**Verified against `master` at `620e80c1e0` (2026-09-09)**, which is an ancestor of
`masks_revamp` — so every line reference below was read out of the current tree. Re-read the
`master` code before acting on any §1 item: several earlier candidates in this file were
overtaken by upstream rewrites that solved the same bug a different way.

**Merged upstream since the last revision.** The three fixes that made up §1 have all landed on
`master`, so their diagnoses are gone from this file (git history has them). Kept here as a
ledger, because older notes still refer to them by number (the staging plan now keeps its own
copy of this in §7, "already upstreamed"):

| Was | Upstream commit | Note |
|---|---|---|
| **U1** `usedetails` flushes nearly the whole pipe on every `synch_all` | `917646d131` *pixelpipe: keep the scharr detail mask across synch_all* | same design (stop freeing the scharr, decide from **buffer presence**), but gated by a `replaying` parameter on `_dev_pixelpipe_synch` rather than the branch's `pipe->synch_no_detail_invalidate` field. That field is now dead and has been removed from the branch. |
| **U2** `toneequal` invalidates its downstream tail on every overlay toggle | `50e0964f68` *toneequal: do not self-invalidate preview pixelpipe cache during processing* | the open question ("which of the two triggers fires") turned out to be the wrong question: the invalidation was unnecessary altogether, and the fix is its one-line deletion. |
| **U3** per-module rendered-mask cache, CPU and OpenCL | `a108707551` *pixelpipe: per-module rendered mask cache* | landed with both paths through one `_render_drawn_mask_cached()`. The branch keeps a small flexi-specific residual on top (the `_group_needs_host_guides` gate for parametric-as-form members). |

U4 and U5 keep their numbers so the plan's cross-references stay valid.

**Status legend:** 🎁 fix exists on the branch, ready to port · 🔬 root cause confirmed, fix not
written · 🧭 under investigation · 💡 proposed, not started · ◐ half done

**Scope tag:** `[masks]` = owned by masks_revamp · `[core]` = pre-existing darktable
pixelpipe/cache behaviour surfaced during this work.

---

## 1. Upstreamable to `master`

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
([pixelpipe_cache.c:174,180](src/develop/pixelpipe_cache.c#L174)) and cachelines are stored with
`DT_INVALID_HASH` ([:268](src/develop/pixelpipe_cache.c#L268)). So with the overlay ON,
returning to a previous value can never be a cache hit. Intentional: `pipe->mask_display` is
excluded from the piece hash. Do not "fix" this; work around it — which is what the rendered-mask
cache (U3, now upstream as `a108707551`) does.

### C2 `[core]` How the full-pipe cache budget is actually computed
Full pipe = 64 lines (`darktable.pipe_cache ? 64 : DT_PIPECACHE_MIN`) bounded by
`dt_get_available_mem() / cache->mem_fraction`, with `mem_fraction = 8` for the FULL pipe
([pixelpipe_hb.c:325](src/develop/pixelpipe_hb.c#L325),
[pixelpipe_cache.c:506,532](src/develop/pixelpipe_cache.c#L506)); `checkmem` evicts oldest lines
over budget. Each full-res RGBA-float buffer is large (6984×4660 ≈ 520 MB), so deep pipes churn.

`dt_get_available_mem()` is `MAX(512MB, (total_memory - cl_uni_memory)/1024 * fractions[4*level + 0])`
([darktable.c:3126-3135](src/common/darktable.c#L3126)) — the **first** number of the resource
tuple (the third is `_get_mipmap_size`, which no longer feeds the pipe cache at all). Index 0 is
`128 / 512 / 700 / 16384` for small/default/large/unrestricted
([darktable.c:2373-2378](src/common/darktable.c#L2373)), so **default → large does raise the pipe
cache budget**, by ~1.37×.
- Tuning without code: with the app closed, raise the **1st** number in `darktablerc`, e.g.
  `resource_large=900 16 128 900`, then select `large`.

### C3 `[core]` With OpenCL, intermediate GPU outputs aren't host-cached
Device buffers aren't copied back for the cache except the focused module's pinned input
([pixelpipe_hb.c:2483,2597,3176](src/develop/pixelpipe_hb.c#L2483)), so the pipe re-executes
top-to-bottom on each edit. Cheap for GPU modules, expensive for the CPU-side mask work in U4.

---

## 4. Diagnostic playbook

- **Per-module timing:** `darktable -d perf` → `processed <module> … took Ns`.
- **Cache decisions:** `darktable -d pipe` → `cache HIT`, `importance hints … focus
  important_in`, `pipe cache check … Freed: invalid NMB`, and crucially
  `pipecache invalidate|flush <reason>` — the reason string names the invalidator
  (`blend new raster:`, `set raster:`, `refresh:`, `usedetails `, `usedetails build `).
  `toneequal: ` no longer exists; that invalidation was deleted by `50e0964f68`.
- **Memory eviction:** `-d pipe -d memory` → `pipe cache check … limit=NMB`.
- **Isolate masks vs pipe:** move a *non-mask* slider on the same module A→B→A; if it lags
  identically, the cost is the pipe cache, not the mask code.
- **Isolate the detail path:** set all `details` to 0; if the lag drops, the scharr/`usedetails`
  path is involved (fixed upstream by `917646d131`, but the repro still tells you where to look).
