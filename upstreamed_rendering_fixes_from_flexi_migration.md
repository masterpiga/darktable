# Three classic-blending rendering defects found while verifying the flexi mask migration

All three are **pre-existing in `master`**, unrelated to the mask revamp. They
were found by a tool built to verify something else, and each is worth fixing
upstream on its own merits. In every case the symptom is the same: darktable
renders the same edit differently depending on whether OpenCL is enabled.

| | what | state |
|---|---|---|
| 1 | the OpenCL path publishes a stale raster mask | **fixed and merged** (`f54402ea96`) |
| 2 | JzCzhz hue is ill-conditioned in the shared JzAzBz transform | diagnosed; fix written and measured in isolation, not implemented |
| 3 | mask feathering is near-singular for grey guides at guide weight 100 | diagnosed, localised and **measured**; the fix is a rendering decision, not a numerical one |

They are ordered by how much evidence stands behind them, strongest first.

The tool (`--verify-masks` on the `masks_revamp` branch) replays a harvested
mask edit four ways -- classic and migrated, each on the CPU and on OpenCL --
and compares the resulting masks. Every measurement below comes from seven
contributed libraries, 42,275 masked edits. (Two more libraries have since been
checked, with the finding-1 fix in place: they add 10,275 edits and 12
divergences, all of them finding 2.) It was built to prove the migration is faithful; the four-way replay also
makes classic disagreeing *with itself* across CPU and OpenCL visible, which is
what this document is about.

Throughout, "diverges" means classic's own CPU and OpenCL renders of the same
edit differ by more than 1/255 at some pixel.


## Finding 1 -- the OpenCL path publishes a stale raster mask (FIXED, merged)

Merged as `f54402ea96`. Kept here for the record, and because the evidence
below is what separates this cause from finding 2.

### The problem

`dt_develop_blend_process_cl()` used to end with:

```c
if(dt_iop_piece_is_raster_mask_used(piece, BLEND_RASTER_ID))
{
  // get back final mask from the device as the raster mask
  if(!raster)                                    // <-- the bug
  {
    err = dt_opencl_copy_image_to_host(devid, mask, dev_mask, ...);
    ...
  }
  dt_iop_piece_set_raster(piece, mask, roi_in, roi_out);
}
```

The `if(!raster)` guard skips the device-to-host readback when this module's own
mask *came from* a raster mask, on the reasoning that such a mask is built on
the host and uploaded, so the host buffer already holds it.

That is true only while nothing touches it on the device afterwards. The mask
post-processing immediately above -- feathering, blur, tone curve -- does
exactly that: it reads and writes `dev_mask`. With any of those active, the host
buffer is still the **unrefined** mask, and that is what gets published.

So a module that consumes a raster mask, refines it, and is itself a raster
source for a later module hands every downstream consumer a different mask than
it blended with -- and a different one than the CPU path publishes for the same
edit. Chained raster masks with feathering are ordinary practice.

### The evidence

Classic's own CPU-vs-OpenCL disagreement, bucketed by mask mode *and* by active
post-processing (42,078 edits replayed on both paths):

| mask mode | post-processing | diverge | edits | rate |
|---|---|---:|---:|---:|
| raster | **any** (feather / blur / tone) | 61 | 61 | **100%** |
| raster | none | 0 | 590 | **0%** |
| drawn | any combination | 0 | 6,747 | 0% |
| drawn + parametric (no active channel) | any combination | 0 | 5,905 | 0% |

The third and fourth rows matter as much as the first: drawn masks with the same
feathering, blur and tone curve do not diverge, in over twelve thousand edits.
That rules out `guided_filter_cl`, `dt_gaussian_blur_cl` and
`blendop_mask_tone_curve` as causes, and leaves the publication step. The
100%/0% split on the raster rows follows exactly from the code: with no
post-processing the shortcut is valid, so those 590 edits are clean.

### The fix

Read back unconditionally. Delete the `if(!raster)` guard. This is what was
merged.

### Evidence that it fixes it

A 76-edit corpus assembled from every *distinct* raster configuration in the
seven libraries:

| | divergent edits | worst classic CPU/GPU gap |
|---|---:|---:|
| before | 42 of 76 | 0.940 |
| after | **0** of 76 | 0.000153 |

Across six of the seven full libraries (41,730 edits), total divergent edits
1,373 -> 1,293. All 80 removed are raster edits with post-processing.

### Does it break existing edits?

**CPU renders: no change at all.** The CPU path already published the
post-processed mask; this only makes OpenCL do the same.

**OpenCL renders: yes, they change** -- for the affected configuration (raster
mask + feathering/blur/tone curve, consumed downstream), and they change *to*
what the CPU already produced. There is no "existing correct render" being lost:
the two pipes disagreed, and one of them was wrong by its own code's stated
intent.

**Mitigation: none needed.** This is a plain defect with no defensible previous
behaviour to preserve.


## Finding 2 -- JzCzhz hue diverges by design, not by drift

### The problem

The remaining 1,293 divergent edits are almost entirely
`blend_cst = rgb_scene`: 1,426 of 1,443 outliers in the original measurement,
against 3 in `rgb_display` and 14 in `Lab`. Within them the JzCzhz **hue**
channel (`hz_in`) is active in 1,218, and is the *only* active channel in 694.

The obvious hypothesis -- that `data/kernels/blendop.cl` and
`src/develop/blends/blendif_*.c` are hand-maintained twins that drifted -- is
**wrong**. They agree. The divergence is numerical amplification in the shared
JzAzBz transform, and it is severe:

| stage | CPU vs OpenCL |
|---|---:|
| LMS, before the two power laws | 1.19e-07 relative (1 ulp -- identical) |
| L'M'S', after `pow(., 134.034375)` | **1.85e-05 relative** (~155x) |
| Iz | 7.4e-07 absolute |
| **Az / Bz** | **5.6e-06 absolute** |

Two things compound. The PQ exponent 134.034375 multiplies any relative
difference in its argument by 134 -- so one ulp anywhere upstream (a `dot()`
that contracts to FMA, `pow`'s last bit) becomes 1.3e-05. Then the A matrix
computes `az = 3.524L' - 4.0667M' + 0.5427S'`, whose coefficients sum to zero:
for near-neutral colours it is a difference of three near-equal numbers, so the
relative error lands as absolute error on a quantity that is itself small.

Hue is `atan2(Bz, Az) / 2pi`, so it absorbs all of it: the hue uncertainty is
roughly `9e-7 / Cz` turns. Measured, on 400,000 samples:

| Cz | hue max difference |
|---|---:|
| < 1e-5 | 0.045 |
| 1e-5 .. 1e-4 | 0.017 |
| 1e-4 .. 1e-3 | 0.0016 |
| > 1e-2 | 7.9e-05 |

Jz survives (it is a sum, not a difference) and Cz mostly survives; hue does
not. This is not a coding error on either side -- **no conforming float32
implementation can agree with another here**, so making the kernel "more
accurate" does not help. It was checked: Apple's OpenCL `pow` already matches
the host `powf` to 1 ulp on this input range.

### Severity as it stands

Over 1,443 outliers: median worst pixel 0.0136 (about 3.5 8-bit steps), median
mean 7.7e-06 over the frame, median 16.8% of the frame differing at all. So the
typical case is a broad, faint difference -- invisible over the area it covers,
visible only where it peaks. A small minority (29 edits) reach a full 1.0 at a
handful of pixels on a mask boundary.

### The proposed fix

**Reformulate `dt_XYZ_2_JzAzBz` -- and its OpenCL twin -- to carry the
differences through the power laws instead of forming three near-equal values
and subtracting them at the end.**

Write the outputs in the algebraically identical cancellation-free form

```
az = 3.524(L'-M') + 0.542708(S'-M')
bz = 0.199076(L'-S') - 1.096799(S'-M')
```

and obtain each difference without ever computing it as a subtraction of
computed values:

- `L - M` etc. from the *matrix rows* (`(M[0]-M[1]) . XYZ'`), not from the
  computed LMS;
- through each power law with `x^k - y^k = y^k * expm1(k * log1p((x-y)/y))`,
  which is accurate because `log1p` and `expm1` are accurate near zero;
- through the rational PQ step with `y_a - y_b = (c2 - c1 c3)(t_a - t_b) /
  ((1 + c3 t_a)(1 + c3 t_b))`, again exact in form.

Fall back to the current path where any LMS component is non-positive
(out-of-gamut), which the current code already clamps.

### Evidence that it fixes it

Implemented on both sides in a standalone harness (verbatim copies of
darktable's host and kernel code, no darktable build involved), 200,000 samples
on an Apple M4 Pro:

| | Az/Bz max abs difference | hue max difference |
|---|---:|---:|
| current formulation | 6.27e-06 | 1.68e-03 |
| cancellation-free | **4.49e-07** | **3.13e-05** |

A 14x improvement on Az/Bz and **54x on hue**. It is also more accurate in
absolute terms, not merely more consistent -- less cancellation means closer to
the true value on both paths.

**This has been validated in isolation, not yet implemented in darktable.**
That is deliberate: `dt_XYZ_2_JzAzBz` is used well beyond blending
(`colorbalancergb`, `colorequal`, `diffuse`, the JzCzhz colour picker), so it
belongs in its own upstream change with its own review, not bundled into a mask
branch.

### Does it break existing edits?

**Yes -- and much less than the bug it removes.** This one changes both the CPU
and the OpenCL render, because it changes a shared conversion.

Measured, host-current vs host-reformulated on the same 200,000 samples:

| | value |
|---|---:|
| mean hue change | 1.43e-05 |
| max hue change | 0.0068 |
| samples changing by more than 1/255 | 3 of 199,773 (**0.002%**) |

Compare that with the CPU/OpenCL disagreement it removes (max 1.68e-03 today,
and up to 0.045 at low chroma). **The change to any single render is of the same
order as, or smaller than, the amount by which the two pipes currently disagree
with each other.** For the pixels most affected -- near-neutral ones -- there is
no stable "existing render" to preserve in the first place: today they render
one way with OpenCL enabled and another way without.

**In which way it changes.** Only near-neutral colours move, and only in hue;
Jz and Cz are essentially untouched. Anything a user selected by a hue band and
could actually see the colour of is unaffected. The visible effect is confined
to whether desaturated pixels fall inside or outside a hue selection -- which is
currently decided by float noise.

**Mitigation options, worst to best:**

1. *Version-gate it* behind a `blend_params` field, the way `feather_version`
   already gates a rendering change, so historical edits keep the old
   conversion. **Not recommended**: it preserves the divergence for exactly the
   edits that have it, which defeats the purpose, and it only covers blending
   while the same conversion is used by several modules that have no such field.
2. *Apply it only in the blend path*, leaving other JzAzBz consumers alone.
   Cheaper to review, but leaves darktable with two JzAzBz transforms that
   disagree with each other -- trading one consistency problem for another.
3. *Apply it everywhere and ship it.* **Recommended.** It is a strictly more
   accurate evaluation of the same function; the render change is far below the
   noise it removes; and it is the only option that leaves one JzAzBz in the
   codebase.

A separate, orthogonal option was tried and **rejected**: gating the hue channel
to zero below a chroma threshold, on the grounds that hue is meaningless there.
It is defensible as a quality change (a hue selection over a flat grey area
currently renders as speckle) but it does **not** fix the consistency problem --
measured on a full library it reduced the worst-pixel divergence by about 10%
and the affected area not at all, because the hue error is broad rather than
confined to the achromatic axis. Recorded here so nobody spends the day on it
twice.


## Finding 3 -- mask feathering solves a near-singular system, and the two implementations resolve it differently

### The problem

Mask feathering is a guided filter (`src/common/guided_filter.c`, and its
separate OpenCL implementation in the same file plus
`data/kernels/guided_filter.cl`). Two things about how blending calls it put it
in a bad numerical regime:

1. The guide is pre-multiplied by `guide_weight`, which for an RGB blend at
   `feather_version == 0` is **100** (`_get_guide_weight`, blend.c). Guide
   values therefore sit around 1e2, and their squares around 1e4 -- far more for
   scene-referred highlights.
2. The covariance matrix is built in the textbook-unstable form
   `Var = E[x^2] - E[x]^2` (`Sigma_0_0 = varpx[VAR_RR] - guide_r*guide_r + eps`
   and its five siblings), and then inverted by Cramer's rule. Over a flat or
   near-achromatic guide that matrix is close to singular -- see "What the
   divergence actually is" below, where measurement corrects the obvious first
   guess about which of these two facts is doing the damage.

The regularizer does not rescue it: `eps` is **absolute** (1.0 at
`feather_version == 0`), while the quantity it regularizes scales as
`guide_weight^2`. At weight 100 it is four orders of magnitude smaller than the
matrix entries, so a near-flat guide leaves the 3x3 solve close to singular, and
Cramer's rule amplifies whatever the cancellation left behind.

Nothing here is *wrong* on either path. The CPU keeps the whole expression in
registers inside one loop, where the compiler may contract to FMA; the OpenCL
path computes each product in its own kernel, rounds it into a float image, and
subtracts in the next. Same algebra, different rounding, on an expression that
cannot afford any.

### The evidence

All from one harvested edit (`colorbalancergb`, parametric mask, feathering
radius 10, `feather_version 0`, `rgb_scene`), replayed on both pipes. "Gap"
is CPU vs OpenCL on the same edit; the migrated column isolates the filter,
because a migrated mask is built on the host for both pipes and the filter is
then the only thing left that differs.

| variant | CPU max | classic gap | migrated gap |
|---|---:|---:|---:|
| as harvested | 0 | 0.06092 | 0.06488 |
| **feathering off** | 0 | 0.00341 | **0.00000** |
| blur off | 0 | 0.06421 | 0.06419 |
| feathering and blur off | 0 | 0.00367 | **0.00000** |

With feathering off the migrated edit is bit-identical across CPU and OpenCL.
The blur is irrelevant. The whole 0.065 is the guided filter.

Sweeping the two parameters the diagnosis predicts:

| `guide_weight` | r=2 | r=5 | r=10 | r=20 | r=40 |
|---|---:|---:|---:|---:|---:|
| **100** (`feather_version 0`) | 0.0642 | 0.0642 | 0.0642 | 0.00073 | 0.00017 |
| **10** (`feather_version 1`) | 0.00154 | 0.00154 | 0.00154 | 0.00010 | 0.00003 |

Dropping the guide weight by 10x drops the divergence by **41x**, with
everything else identical -- which is what an `eps` that does not scale with
`guide_weight^2` predicts, and which is not explainable by the filter's own
behaviour. Larger radii recover as well: a wider window admits more genuine
variation, so the matrix moves away from singular.

It is not confined to parametric masks or to migration. In one library's classic
edits, plain **drawn-only** masks show the same signature, and only when
feathered:

| classic drawn-only | n | median gap | max | over 1/255 |
|---|---:|---:|---:|---:|
| feathered | 480 | 0 | 0.00523 | 1 |
| not feathered | 260 | 0 | 0.00000 | 0 |

### How much of the corpus is exposed

Across 61,387 masked edits from 14 libraries:

| | edits | share |
|---|---:|---:|
| feathered at all | 38,151 | **62.1%** |
| feathered, `feather_version 0`, RGB (guide weight 100) | 19,169 | **31.2%** |
| feathered, `feather_version 1` (guide weight 10) | 17,728 | 28.9% |

So roughly a third of all masked edits sit in the badly-conditioned
configuration. The divergence is usually far below 1/255 -- it needs a guide
that is flat where the mask has structure -- but the exposure is not a corner
case, and every one of those edits renders differently with OpenCL on than off
by some amount.

### What the divergence actually is

The first guess -- that this is plain float32 cancellation in
`E[x^2] - E[x]^2` -- is not sufficient, and measurement rules it out. The
cancellation leaves an absolute error of order `2^-24 * mean^2`, which for a
mid-grey guide at weight 100 is about 2e-05, i.e. five orders of magnitude below
`eps = 1.0`. Rounding alone cannot be decided by a term that small.

What makes it matter is that **the 3x3 system is near-singular to begin with**.
The guide is RGB, and for any achromatic or near-achromatic region R, G and B
are the same signal: the covariance matrix collapses towards rank 1, and the
only thing keeping it invertible is `eps` on the diagonal. At guide weight 100
the matrix entries are ~1e4 while `eps` is 1, so the regularization is
relatively 1e-4 -- and Cramer's rule then amplifies whatever each
implementation's rounding left behind. Grey and near-grey guide regions are
ordinary, not exotic, which is why this is visible across a third of the corpus.

That also explains the `feather_version` result: version 1 does not merely
rescale, it makes `eps` relatively 25x larger, which is a real conditioning
improvement, and the divergence falls 41x.

### The proposed fix, and what measuring it showed

The regularization has to be **relative to the guide's scale** rather than
absolute. Implemented as a floor on the covariance diagonal proportional to what
was subtracted (`floor * mean^2`, added identically in `guided_filter.c` and
`guided_filter.cl`), swept on the edit above:

| floor | migrated CPU/OpenCL gap |
|---|---:|
| 0 (current master) | 0.0642 |
| 2^-20 | 0.0279 |
| 1e-05 | 0.0197 |
| 1e-04 | 0.0070 |
| 1e-03 | **0.00184** |
| 1e-02 | 0.00064 |

Monotonic, with no threshold at which it snaps clean: the disagreement is
governed by how well-conditioned the solve is, exactly as the diagnosis says.
Over the whole akgt94 corpus (2,521 edits), a 1e-03 floor takes the worst
migrated CPU/OpenCL gap from **0.0649 to 0.00955**, a 6.8x improvement, and
moves 12 edits from "differs" to "identical".

**The uncomfortable part, stated plainly.** A floor large enough to fix
consistency is large enough to change rendering. At 1e-03 and a mid-grey guide
the added term is ~40% of `eps` -- that is not a rounding correction, it is a
different amount of feathering in flat regions. Conversely a floor small enough
to be invisible (2^-20 adds 0.04% of `eps`) only buys 2.3x, leaving the gap at
0.028, still 7x over 1/255. There is no value that is both invisible and
sufficient, because the quantity being stabilized *is* the filter's own
behaviour near flat guides.

So this cannot be fixed the way finding 2 can. It is not a more accurate way to
evaluate the same function; it is a decision about how strongly to regularize,
and any answer that fixes the CPU/OpenCL disagreement changes what feathering
does on flat, near-achromatic guides.

### Recommendation

Ship a `feather_version 2` whose `eps` is proportional to `guide_weight^2`
(equivalently: apply the floor above with a value around 1e-03), leaving
existing edits on their current version and their current rendering. This is the
mechanism `feather_version 1` already used, for the same reason, and the numbers
above give the constant a basis rather than a guess.

What should *not* be done is to add a small floor ungated and call the problem
solved: the measurement above shows that buys less than a factor of three and
leaves the disagreement well above the visible threshold.

### Does it break existing edits?

Version-gated, no -- that is the point of gating it. Existing edits keep
`feather_version 0` or `1` and render exactly as they do now, including their
CPU/OpenCL disagreement. New edits get the conditioned parameters.

Left ungated it would break them, mildly but genuinely: flat, near-achromatic
regions would feather slightly differently. That is a real change to how the
filter behaves, not a change in its rounding.

The experiment was implemented, measured and then **reverted** -- the branch
carries none of it, and the numbers above are all that was wanted from it.


## What this means for flexi

None of the three bugs is caused by migration, and the first two cannot occur in
a migrated edit at all.

Flexi renders its whole group -- `DT_MASKS_PARAMETRIC` members included -- with
`dt_masks_group_render_roi()` on the host, in *both* pipes, and uploads the
result like any drawn mask (`mode_drawn` includes `DEVELOP_MASK_FLEXI`).
Migration clears `DEVELOP_MASK_CONDITIONAL`, so the blendif kernel evaluates
nothing, and a migrated raster mask becomes a group element that takes the
normal drawn branch -- with its readback. Classic keeps two implementations of
the parametric mask and a special case for raster; flexi keeps one of each.

Consequences:

- **Finding 1 cannot occur** in a migrated edit: there is no raster shortcut.
  Confirmed by the verifier -- `dev_diff_after` (migrated CPU vs migrated GPU) is
  ~0 on every one of the 61 edits where classic diverges.
- **Finding 2 cannot occur** in a migrated edit: the JzCzhz conversion runs once,
  on the host, for both pipes. Fixing it upstream would still be worth doing --
  every non-mask consumer of JzAzBz has the same problem -- but flexi masks do
  not depend on it.
- **Finding 3 reaches flexi in full.** Feathering is the same guided filter
  wherever the mask came from, so a flexi mask feathered at guide weight 100
  diverges exactly as a classic one does. Migration neither causes nor cures it:
  it does not touch `feathering_radius` or `feather_version`, and a migrated
  edit keeps whatever the classic one had. This is the one finding here that is
  worth fixing *for* flexi rather than merely around it.
- The price flexi pays for the first two is a device-to-host readback of
  `dev_in`/`dev_out` when a group contains a parametric member or per-shape
  feathering (`_group_needs_host_guides()`), and a residual CPU/GPU gap of
  0.0025 (Apple) / 0.00093 (AMD) from the refinement stages, which still have
  separate implementations. Both below 1/255 across the whole corpus.
- None of the three should be charged to the migration statistics, and the
  verifier now proves that rather than asserting it. Where migration appears to
  widen an edit's own CPU/GPU gap, it re-renders the migrated pair with the mask
  post-processing switched off and reports `dev_diff_after_nopost`: if the gap
  survives it is migration's, and if it collapses the amplifier was a stage
  classic runs too. Over 61,332 edits the count of widenings that survive is
  **0**.


## Reproducing

Both need a build of the `masks_revamp` branch, and a machine with a working
OpenCL device (the run prints `[verify] OpenCL device N acquired`; without one
every GPU figure is 0, which looks like agreement but is really no test).

```
darktable --library :memory: --verify-masks <harvest.json.gz>
```

writes a per-edit report next to the input. The classic-diverges-with-itself
condition is `max_diff <= 1/255 && gpu_max_diff > 1/255`, equivalently
`dev_diff_before > 1/255` with `dev_diff_after ~ 0`.

- Finding 1: needs a build from before `f54402ea96` to reproduce at all.
  `classic_opencl_outliers.json.gz` in the repo root reproduces the general
  divergence; a raster-specific corpus is trivial to regenerate by
  filtering any harvest for `mask_mode` containing `raster` and a non-zero
  `feathering_radius`, `blur_radius`, `contrast` or `brightness`.
- Finding 2: the standalone differential harnesses used for the numbers above
  (`jzdiff.c`, `stage.c`, `powtest.c`, `stable.c`) need only `clang` and
  `-framework OpenCL` (or any OpenCL SDK); they contain verbatim copies of both
  darktable implementations and no darktable dependency.
- Finding 3: take any feathered edit and vary one field at a time in the
  harvest JSON -- `feathering_radius` to 0 (the gap goes to 0), and
  `feather_version` to 1 (the gap drops ~40x) -- then re-run the command above.
  Both tables in that section were produced that way, from a single harvested
  edit, with no code changes at all.

Reproduced on Apple M4 Pro (Apple OpenCL 1.2) and, for finding 2's signature, on
AMD gfx1150 with `OPENCL FAST MODE: NO` -- so none of them is a vendor compiler
issue, and `-cl-fast-relaxed-math` is not involved.
