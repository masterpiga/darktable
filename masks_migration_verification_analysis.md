# Classic → Flexi Mask Migration Verification Analysis

A thorough audit of the verification architecture ([masks_revamp_verification_report.md](file:///Users/dudo/Documents/Coding/darktable/masks_revamp_verification_report.md)), the four CLI tools in `src/develop/masks/`, and the underlying migration implementation in `src/develop/` was conducted.

---

## 1. Executive Verdict

**The approach is mathematically sound, remarkably thorough, and genuinely does what it says on the tin.** 

Unlike standard structural unit tests that merely assert data-structure shapes, this pipeline performs end-to-end pixel rendering comparisons across both CPU and OpenCL pipelines, models drift baselines, tests database serialization round-trips, and isolates non-informative ("inert") masks.

The strategy of distributing the read-only scraper to 10–20 power users and replaying their JSON harvests locally is **safe, privacy-preserving, and highly effective for surfacing real-world edge cases**.

However, several **blind spots, edge cases, and campaign-level oversights** were identified during the audit that must be accounted for.

---

## 2. Component-by-Component Validation

### 2.1 The Scraper (`--harvest-masks`, [harvest.c](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/harvest.c))

* **Safety & Database Integrity ([harvest.c:L501-517](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/harvest.c#L501-L517)):**
  The database is opened strictly read-only twice over: using SQLite URI mode `file:...?mode=ro` and SQLite open flags `SQLITE_OPEN_READONLY | SQLITE_OPEN_URI`. It executes before darktable's database initialization, ensuring zero locks, zero WAL writes, and zero risk of corrupting user catalogs.
* **Privacy & User Anonymity ([harvest.c:L42-49](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/harvest.c#L42-L49), [L264-266](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/harvest.c#L264-L266), [L404](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/harvest.c#L404)):**
  No raw pixel data, image filenames, folder paths, EXIF metadata, timestamps, tags, ratings, or user-typed shape/group names are extracted. Group `name` is explicitly omitted during decoding. Geometry is normalized to unit floats $[0, 1]$. The JSON emitter is written by hand rather than via reflective serialization, preventing accidental leaks.
* **Form History Deduplication ([harvest.c:L58-64](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/harvest.c#L58-L64)):**
  The query `SELECT ... MAX(num) FROM masks_history WHERE imgid = ?1 AND num <= ?2 GROUP BY formid` correctly models darktable's cumulative snapshot architecture and avoids dangling mask IDs.

---

### 2.2 The Verification Engine (`--verify-masks`, [verify.c](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/verify.c))

* **Production Blend Replay ([verify.c:L331-366](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/verify.c#L331-L366), [L383-427](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/verify.c#L383-L427)):**
  Instead of simulating the mask transfer function in isolation, both the classic and migrated states are evaluated directly through production [dt_develop_blend_process()](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend.c) and [dt_develop_blend_process_cl()](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend.c).
* **Dual-Path GPU Replay & Gap Widening ([verify.c:L770-785](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/verify.c#L770-L785), [L950-968](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/verify.c#L950-L968)):**
  Because CPU and OpenCL blend implementations diverge slightly in float precision and ordering even in stock darktable, judging GPU migration by whether it *widened* the pre-existing classic CPU/GPU gap (`dev_diff_after - dev_diff_before > 1/255`) avoids false regression flags while catching true OpenCL divergence.
* **Color Management & Profile Plumbing ([verify.c:L674-723](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/verify.c#L674-L723)):**
  The replay harness wires an explicit Linear Rec2020 working, output, and input profile into `r->pipe` and `r->dev.iop_order_list`. This prevents blendif parametric masks from aborting due to missing color spaces.
* **Reproducibility ([verify.c:L1011-1028](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/verify.c#L1011-L1028)):**
  `omp_set_num_threads(1)` eliminates thread-scheduling float reassociation drift across runs.

---

### 2.3 The Probe Image ([probe_image.c](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/probe_image.c))

* **Excitation of Parametric Channels:**
  The probe uses MurmurHash3-based deterministic noise, van der Corput low-discrepancy sequences across 16px tiles, 5-octave fBm texture, coprime multi-scale edge half-planes, a saturation booster, and an exposure ladder spanning $[-6\text{ EV}, +2\text{ EV}]$.
* **Inert vs. Live Filtering ([verify.c:L441-450](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/verify.c#L441-L450), [L1108-1115](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/verify.c#L1108-L1115)):**
  Edits resulting in flat/uniform masks are flagged as `inert` and excluded from live pass rates, preventing vacuous $0 == 0$ false passes.

---

### 2.4 Persistence Round-Trip Testing (`--roundtrip-masks`, [roundtrip.c](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/roundtrip.c))

* **Two-Stage Load Pipeline ([roundtrip.c:L460-470](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/roundtrip.c#L460-L470)):**
  - **Load #1:** Replays classic history from SQLite → runs migration → writes flexi history and forms back to SQLite via [dt_dev_write_history_ext()](file:///Users/dudo/Documents/Coding/darktable/src/develop/develop.c).
  - **Load #2:** Re-reads the newly written flexi history from SQLite.
* **Non-Union Run Invariant ([roundtrip.c:L250-272](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/roundtrip.c#L250-L272)):**
  Validates that every non-union combine operator (`DIFFERENCE`, `INTERSECTION`, `SUM`, `EXCLUSION`) has `group_start = 1` across both loads.

---

## 3. Gaps, Blind Spots & What Was Overlooked

While the core mechanics are robust, the following gaps in the power-user campaign and testing harness should be addressed:

### Gap 1: Power Users with Older History Entries (`blendop_version <= 10`)
* **Mechanism:** In [harvest.c:L597-601](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/harvest.c#L597-L601):
  ```c
  if(!bp || bp_bytes != (int)sizeof(dt_develop_blend_params_t))
  {
    skipped_size++;
    continue;
  }
  ```
* **Issue:** darktable $\le 3.8$ stored shorter blend parameter structs (prior to boost factors and feathering/detail extensions). Because `harvest.c` does not initialize the full IOP module stack, it cannot run [dt_develop_blend_legacy_params_ext()](file:///Users/dudo/Documents/Coding/darktable/src/develop/blend.c#L2726) during harvesting and silently skips these rows with `skipped_size++`.
* **Impact:** Power users with libraries dating back 5–10 years may have thousands of edits skipped unless those images were previously opened or modified in modern darktable versions.
* **Action:** Document in the user instructions that `--harvest-masks` will report `skipped (old blendop)`, and verify whether that count is high for long-time users.

---

### Gap 2: Mask Presets & Styles (`data.db`) are Never Harvested
* **Mechanism:** The scraper only scans `library.db` (`main.history` and `main.masks_history`).
* **Issue:** Power users heavily utilize **saved mask presets** and **styles** containing classic drawn/parametric masks stored in `data.db` (`main.presets`, `main.style_items`).
* **Impact:** A power user could have zero mask issues in their active images, yet have broken classic mask presets or styles upon upgrading to darktable with flexi masks.
* **Action:** Consider extending `harvest.c` to accept `--data-db data.db` or run a targeted preset check.

---

### Gap 3: Sidecar XMP Workflows
* **Issue:** Many power users configure darktable with ephemeral or frequently wiped `library.db` files and rely on `.xmp` sidecars alongside their RAW files.
* **Impact:** Running `--harvest-masks` on their active `library.db` might yield very few edits if their collection is distributed across external disks and un-imported XMPs.
* **Action:** Provide clear instructions for users to point to their main catalog database, or advise them to import their archive into a temporary test library before harvesting.

---

### Gap 4: Multi-Instance Modules in Round-Trip Testing
* **Mechanism:** In [roundtrip.c:L130-140](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/roundtrip.c#L130-L140), `multi_priority` is hardcoded to `0` when inserting scratch history:
  ```c
  // multi_priority is forced to 0, NOT the harvested value...
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 7, 0);
  ```
* **Reason:** Secondary module instances lack entries in the default IOP order of an empty scratch image, causing `dt_dev_read_history_ext()` to reject them.
* **Issue:** In `--verify-masks`, multi-instance raster masks *are* resolved ([verify.c:L542](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/verify.c#L542)), but in `--roundtrip-masks`, the persistence of secondary instances is normalized away.
* **Impact:** Any serialization bug specific to multi-instance mask references across save/load would not be caught by `--roundtrip-masks`.

---

### Gap 5: Resolution Downscaling & Sub-Pixel Guides
* **Mechanism:** [verify.c:L54](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/verify.c#L54) clamps replay images to `VERIFY_MAX_EDGE = 512` while scaling `roi.scale = width / full_width`.
* **Issue:** On 45–60 MP images with extremely tight feathering radii ($<1.0\text{ px}$), downscaling to 512px can compress the filter radius to fractional sub-pixel extents where guided-filter box blurs clamp or behave differently than at full export resolution.
* **Impact:** Minor discrepancies in high-frequency detail thresholds or sub-pixel edge feathering may be smoothed out at 512px.

---

### Gap 6: Display / Color Space Assumptions in Replay
* **Mechanism:** [verify.c:L689-723](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/verify.c#L689-L723) sets `DT_COLORSPACE_LIN_REC2020` as the pipeline working space.
* **Context:** In classic darktable, modules operating in legacy display-referred RGB (`DEVELOP_BLEND_CS_RGB_DISPLAY`) or Lab (`DEVELOP_BLEND_CS_LAB`) evaluate blendif against specific color spaces.
* **Validation:** While the replay correctly initializes the profile structures, classic edits authored under legacy sRGB or custom display profiles are replayed in Linear Rec2020. This is valid for testing mathematical consistency (since classic and flexi both evaluate under the same replayed profile), but does not test custom user ICC profiles.

---

## 4. Summary of Verification Coverage

| Category | Replay Mechanism | Covered? | Limitations / Notes |
|---|---|:---:|---|
| **Drawn-only Masks** | GPU & CPU Production Render | **Yes** | Reuses form tree; verifies run-split normalization |
| **Pure Parametric** | GPU & CPU Production Render + Probe | **Yes** | Fully covers channel polarity, boosts, EV range |
| **Drawn + Parametric** | GPU & CPU Production Render + Probe | **Yes** | Validates multiply stacking and composite invert |
| **Raster Masks** | Synthetic Upstream Source Module | **Partial** | Covers invert, opacity, refinement; does *not* cover geometric transforms (lens/crop) |
| **Combine Operators** | Sequential vs. Run Fold Test | **Yes** | Solved via Bug #5 split normalization |
| **Save / Load Cycle** | SQLite Round-Trip Replay | **Yes** | Invariant check on non-union runs |
| **Older Edits ($\le 3.8$)**| `--harvest-masks` Read | **No** | Struct size mismatch skips un-migrated legacy rows |
| **Styles & Presets** | Harvest from `data.db` | **No** | Scraper only targets `library.db` |
| **GUI & Interactions** | Headless Replay | **No** | Excluded by design (focuses on rendering engine) |

---

## 5. Actionable Recommendations for the User Campaign

1. **User Execution Guidance:**
   Instruct participating power users on how to target custom library locations:
   ```bash
   darktable --harvest-masks my_harvest.json --library /path/to/library.db
   ```
2. **Review Harvest Logs for Skipped Versions:**
   When receiving JSON files from users, check the `"summary"` object:
   - If `"skipped_unsupported_blendop_size"` is high, explain to the user that older un-migrated edits from previous major versions were bypassed.
3. **Focus on Rare Combine Modes:**
   Prioritize analyzing JSONs where `"mask_combine"` contains non-zero `"inverted"` or `"inclusive"` counts, as these exercise the most intricate XOR polarity-folding paths in [migrate_legacy.c](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/migrate_legacy.c).
4. **Spot-Check Non-Union Multi-Brush Masks:**
   Review edits flagged with high brush/path counts using non-union operators (`DIFFERENCE`, `INTERSECTION`, `SUM`) to ensure [dt_masks_normalize_flexi_groups()](file:///Users/dudo/Documents/Coding/darktable/src/develop/masks/migrate_legacy.c#L1128) successfully eliminates all run-aggregation discrepancies.
