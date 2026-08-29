# Classic blending: CPU/OpenCL divergence

Handoff note. This is **not** a masks_revamp bug — it is a pre-existing defect in
darktable's *classic* blending on OpenCL, found incidentally by the migration
verifier on the `masks_revamp` branch. It should be reproducible on `master`.

Everything below is evidence gathered on 2026-08-28, plus the second-device
confirmation of 2026-08-29 (section 2). Where something is a hypothesis rather
than a measurement, it says so.


## 1. The finding

Replaying real users' mask configurations through the production blend path,
four ways per edit (classic/migrated x CPU/OpenCL), turns up **274 edits out of
5954** where the classic mask renders differently on CPU than on OpenCL. Six of
them differ by the full range (1.0 on a 0..1 mask); the median is ~0.014, i.e.
about 3.5 8-bit steps — visible.

In every one of the 274:

- the **CPU** renders classic and migrated identically (`max_diff <= 1/255`);
- the **migrated GPU** render agrees with the CPU (`dev_diff_after` ~ 0);
- only the **classic GPU** render is the outlier (`dev_diff_before` ~
  `gpu_max_diff`).

So the odd one out is classic-on-GPU, and the other three agree.

**Severity, in practical terms.** A mask value is how strongly the module applies
at that pixel, 0 = not at all, 1 = fully. Over all six corpora (section 7):

| | value | what it means |
|---|---|---|
| frequency | 1,443 / 42,275 edits (**3.4%**) | of real masked edits; 0% for drawn-only, 4.7-12.5% for modes using a parametric component |
| typical (median) | 0.014 at the worst pixel, 7.7e-06 as a mean | ~3.5 steps on a 0-255 scale where it peaks, but two thousandths of a step averaged over the ~17% of the frame it touches -- see the severity table below |
| worst observed | 1.0 | fully masked vs not masked at all: the module applies at full strength where it should be absent, or vice versa |

**How much of the image is affected — now measured.** `_max_abs_diff()` alone
returns the single largest disagreement anywhere in the frame, so a 1.0 was
consistent with half the photo being wrong *and* with one pixel landing on the
wrong side of a threshold. `verify.c` now reports `gpu_mean_diff` and
`gpu_differing_pixels` beside it, and over all 1,443 outliers in the seven
corpora the answer splits into two quite different populations:

| | worst pixel | mean over frame | share of frame differing |
|---|---:|---:|---:|
| all 1,443 outliers | median 0.0136 | median 7.7e-06 | median **16.8%**, p90 42% |
| the 29 full-range (1.0) ones | 1.0 | up to 0.336 | median **0.003%**, max 4.6% |

So the typical outlier is the opposite of what the worst-pixel figure suggests:
a *broad* region -- around a sixth of the frame -- differing by an amount whose
mean is 7.7e-06, two thousandths of an 8-bit step, with a handful of pixels
reaching ~3.5 steps. Imperceptible over the area it covers, visible only where
it peaks.

The full-range cases are the reverse and are genuinely thin: 1.0 at the worst
pixel over a median 0.003% of the frame, i.e. a handful of pixels on a mask
boundary. That is consistent with the threshold sensitivity of group B in
section 2 -- a small numeric difference landing on either side of a comparison
-- and it means "six edits differ by the full range" should not be read as six
badly wrong images.

Neither population is a reason to leave the bug unfixed, but they call for
different debugging: the broad-and-faint majority looks like an arithmetic
difference, the narrow-and-total minority like a branch taken differently.

Replay resolution: the verifier renders at 512px on the long edge (about
175,000 pixels for a 3:2 frame), so the percentages above are of that canvas.

## 2. What is established, and what is not

**Established.** 274 distinct configurations from three independently
contributed libraries diverge, across 15 modules. Zero CPU failures and zero
cases where migration widened an edit's own CPU/GPU gap, in all three corpora --
so this is not the migration moving anything.

**Also established: it reproduces on a second, unrelated OpenCL stack.** The
corpus counts above were all gathered on one device:

```
DEVICE: 'Apple M4 Pro'   PLATFORM: Apple   OpenCL 1.2   DRIVER VERSION 1.2 1.0
```

A harvest file contains only mask *specifications* -- the rendering happens
locally -- so what varied between the corpora was the mask configurations, not
the GPU. That left open whether this was a darktable bug or an Apple OpenCL
compiler bug. (An earlier note in the working log claimed cross-hardware
reproduction on the strength of one contributor's higher rate; that reasoning was
wrong, and is superseded by the actual second-device run below rather than
merely retracted.)

The 35-edit corpus of section 5 has since been replayed on:

```
DEVICE: 'gfx1150'   PLATFORM: AMD Accelerated Parallel Processing
DEVICE VERSION: OpenCL 2.0 AMD-APP (3679.0)   DRIVER VERSION: 3679.0 (PAL,LC)
OPENCL FAST MODE: NO
```

Windows, AMD RDNA, LLVM/PAL compiler -- nothing shared with the Apple stack. 33
of the 35 still diverge, with the same signature: `max_diff == 0` on every edit,
worst classic CPU/GPU gap 1.0, worst migrated gap 0.00093, `dev_gap_widened` 0.
**This is a darktable bug, not an Apple compiler bug.** `OPENCL FAST MODE: NO`
also rules out `-cl-fast-relaxed-math` as the cause.

**Two mechanisms, not one.** Comparing the two runs edit by edit splits the
corpus cleanly:

| group | behaviour | edits |
|---|---|---|
| **A** | Apple and AMD diverge by the *same amount to 5 significant figures* | 1, 4, 6, 7, 8, 10, 12, 13, 16, 17, 21, 22, 23, 33, 34 |
| **B** | magnitude is vendor-dependent, ratio 0.000 to 5.603 | the rest |

Group A is the important one. Two different compilers on two different ISAs do
not agree to five digits by accident, so for those edits the divergence is a
deterministic, source-level difference between the kernel and its host
counterpart -- reproducible anywhere, and the right place to start debugging.

Group B varies with the device in both directions. Two Apple outliers fall below
1/255 on AMD (edit 5, `diffuse`, 1.0 -> 0.0000212; edit 9, `colorbalancergb`,
0.664 -> 0.000309), while others get materially worse (`colorequal` 14/15,
0.081 -> 0.454; `blurs` 30, 5.0x; `highpass` 26, 2.0x). That is consistent with a
second, precision-sensitive effect: a small numeric difference landing on either
side of a threshold and amplifying. It explains why the outlier rate varies
between corpora and why some cases reach full range.

Caveat on the AMD run: it was built from `dc7361be49`, 7 commits behind the
`masks_revamp` HEAD that produced the Apple numbers (hence the missing `source`
and `summary` keys in its report, added later by `e7e06fb8f2`). The `verify.c`
diff across that range touches only report serialisation and gzip input --
`_verify_edit()`, `_max_abs_diff()`, the tolerance and the single-threading are
unchanged -- so the two runs are directly comparable. Re-running both on the same
commit is cheap and worth doing before this goes upstream.


## 3. The discriminator

Not every failing `--verify-masks` verdict is this bug. The test that separates
it from a genuine migration failure, using the per-edit report fields:

| field | meaning | this bug |
|---|---|---|
| `max_diff` | CPU: classic vs migrated | `<= 1/255` |
| `gpu_max_diff` | GPU: classic vs migrated | `> 1/255` |
| `dev_diff_before` | CPU vs GPU, **classic** | ~ `gpu_max_diff` |
| `dev_diff_after` | CPU vs GPU, **migrated** | ~ 0 |

`dev_gap_widened` in the run summary counts edits where migration made an edit's
own CPU/GPU gap worse. It is **0** across all corpora; a non-zero value there
would be a migration regression and a different investigation.


## 4. The two causes

**Both are now identified.** Section 4 originally carried a single hypothesis
("it is somewhere in the blendif kernels"). Splitting the corpus by mask mode
*and* by which post-processing is active separates it into two unrelated
defects, one of which is an outright bug with a one-line fix and the other of
which is not a coding error at all. What follows keeps the original evidence
that led there, then states each cause and how it was confirmed.

### 4.0 The split that separates them

Classic's own CPU-vs-GPU disagreement (`dev_diff_before > 1/255`), over all
42,078 edits that ran on both paths, bucketed by mask mode and active mask
post-processing:

| mask mode | parametric channels | post-processing | diverge | edits | rate |
|---|---|---|---:|---:|---:|
| raster | none | **any** (feather / blur / tone) | 61 | 61 | **100%** |
| raster | none | none | 0 | 590 | **0%** |
| drawn | none | any combination | 0 | 6,747 | 0% |
| drawn + parametric | none | any combination | 0 | 5,905 | 0% |
| drawn + parametric | yes | none | 389 | 888 | 43.8% |
| drawn + parametric | yes | feather | 697 | 21,627 | 3.2% |
| parametric | yes | various | ~50 | ~700 | 4-29% |

Two populations that share nothing: one keyed on **raster + post-processing**
and completely deterministic, one keyed on **an active parametric channel** and
probabilistic. Note the third and fourth rows: drawn masks with feathering,
blur and tone curve do *not* diverge, in thousands of edits. That rules the
post-processing implementations themselves (`guided_filter_cl`,
`dt_gaussian_blur_cl`, `blendop_mask_tone_curve`) out as a cause, which is what
makes the 100% raster row so pointed.

### 4.1 Cause A: the OpenCL path publishes a stale raster mask (FIXED)

`dt_develop_blend_process_cl()` ended with:

```c
if(dt_iop_piece_is_raster_mask_used(piece, BLEND_RASTER_ID))
{
  // get back final mask from the device as the raster mask
  if(!raster)
  {
    err = dt_opencl_copy_image_to_host(devid, mask, dev_mask, ...);
    ...
  }
  dt_iop_piece_set_raster(piece, mask, roi_in, roi_out);
}
```

The `if(!raster)` guard skips the device-to-host readback when this module's own
mask *came from* a raster mask, on the reasoning that such a mask is built on
the host and uploaded, so the host buffer already holds it. That is true only
while nothing touches it on the device afterwards -- and the mask
post-processing immediately above does exactly that, reading and writing
`dev_mask`. With feathering, blur or the tone curve active, the host buffer is
still the **unrefined** mask, and that is what gets published.

So a module that consumes a raster mask, refines it, and is itself used as a
raster source hands every downstream consumer a different mask than it blended
with, and a different one than the CPU path publishes for the same edit. It
also explains the 100%/0% split exactly: with no post-processing the shortcut is
valid, which is why those 590 edits are clean.

The fix is to read back unconditionally (blend.c, this branch). Confirmed on a
76-edit corpus built from every distinct raster configuration in the seven
libraries: **42 divergent before, 0 after**, worst classic CPU/GPU gap
0.94 -> 0.000153.

This one is worth reporting upstream on its own. It is not exotic -- chained
raster masks with feathering are ordinary practice -- and the symptom (a
downstream module masked by the *unfeathered* shape, but only with OpenCL on)
is the kind of thing users report as "the GPU render looks different" without
ever finding the cause.

### 4.2 Cause B: hue is ill-conditioned near the achromatic axis

The remaining population is `blend_cst = rgb_scene` almost exclusively: 1,426 of
the 1,443 outliers, against 3 in `rgb_display` and 14 in `Lab`. Within it,
`hz_in` -- the JzCzhz **hue** channel -- is active in 1,218 of them, and 694 have
hue as their *only* active channel.

A standalone differential (`jzdiff.c`, no darktable build involved: verbatim
copies of `dt_XYZ_2_JzAzBz`/`dt_JzAzBz_2_JzCzhz` from
`colorspaces_inline_conversions.h` and of `XYZ_to_JzAzBz`/`JzAzBz_to_JzCzhz`
from `data/kernels/colorspace.h`, fed identical XYZ D65 values) settles what
drifted, on 400,000 samples on the Apple M4 Pro:

| channel | max difference | mean | samples over 1/255 |
|---|---:|---:|---:|
| Jz | 4.6e-07 | 5.3e-09 | 0 |
| Cz | 7.5e-06 | 7.5e-08 | 0 |
| **hz** | **0.045** | 1.0e-04 | 1.14% |

And the hue disagreement is entirely confined to low chroma:

| Cz | samples | hue max | over 1/255 |
|---|---:|---:|---:|
| < 1e-6 | 15,788 | 0.045 | 1.39% |
| 1e-6 .. 1e-5 | 25,210 | 0.0095 | 1.22% |
| 1e-5 .. 1e-4 | 83,284 | 0.017 | 4.82% |
| 1e-4 .. 1e-3 | 40,961 | 0.0016 | **0%** |
| 1e-3 .. 1e-2 | 159,549 | 0.0009 | **0%** |
| > 1e-2 | 75,208 | 7.9e-05 | **0%** |

**So there is no formula drift.** The two implementations agree on Jz and Cz to
around 1e-6, and on hue to within 1e-4 wherever hue means anything. They differ
only where `Az` and `Bz` are both within ~1e-9 of zero, and the angle
`atan2(Bz, Az)` is therefore arbitrary -- the worst single sample is a *black*
pixel (XYZ = 0, Cz = 1e-17) where the CPU says hue 0.436 and the GPU says 0.482.
Both are meaningless; neither is wrong.

That is the mechanism for the dominant population, and it also explains the
shape of the severity table in section 1: a broad region of near-neutral pixels
(shadows, sky, anything desaturated) each differing by a tiny amount, with the
occasional pixel where a hue-band slider lands on opposite sides of its edge.

The residual full-range cases on `Jz_in` or `Cz_in` alone (90 and 26 edits) are
the second-order version of the same thing: those channels agree to 5e-7, but
`_blendif_compute_factor` makes a hard `<=` comparison against a slider limit,
so a 5e-7 input difference at a slider edge swings the factor between 0 and 1.
That is the "group B" threshold sensitivity of section 2, seen from the other
end.

**A fix here is a rendering decision, not a bug fix**, and belongs upstream
rather than in this branch. The honest options are to gate the hue channel's
factor where chroma is below the point at which hue is defined (both paths, and
arguably a quality improvement in its own right -- a hue selection on a black
region is currently speckle), or to accept the divergence as noise on
ill-conditioned input. Chasing ulps in the conversion will not help: the inputs
that disagree are ones where no amount of precision produces a meaningful
answer.

### 4.3 The original evidence

Outlier rate broken down by classic mask mode, over 5954 edits:

| mask mode | outliers | edits | rate |
|---|---:|---:|---:|
| `uniform\|drawn` | **0** | 1190 | **0.0%** |
| `uniform\|drawn\|parametric` | 174 | 3701 | 4.7% |
| `uniform\|parametric` | 53 | 686 | 7.7% |
| `uniform\|raster` | 47 | 377 | 12.5% |

**Drawn-only masks never diverge** -- on this bug. (Do not read that as "drawn
masks are fine": three later corpora turned up nine drawn-only edits where
*migration itself* renders the wrong mask, on both devices equally. Those are a
separate defect, see `masks_revamp_migration_failures.md`, and they are excluded
from the counts here by the `max_diff <= 1/255` condition in the discriminator
above.) The clean split here is exactly what the code predicts: in
`dt_develop_blend_process_cl()` (src/develop/blend.c:1303) the drawn mask is
rasterised **on the host** by `dt_masks_group_render_roi()` and uploaded, so its
geometry cannot differ from the CPU run. The parametric mask is the part
evaluated in-kernel, via `kernel_blendop_mask_*` fed by `dev_blendif_params`
from `dt_develop_blendif_process_parameters()`.

**Hypothesis (superseded by 4.2):** the divergence lives in the blendif OpenCL
kernels, in `data/kernels/blendop.cl`, against their host counterparts in
`src/develop/blends/blendif_*.c`. Half right: it is reached through those
kernels, but nothing in them has drifted -- the disagreement is in the JzCzhz
hue of near-achromatic pixels, which is ill-conditioned rather than wrong.

The shape of it is that classic has **two** implementations of the parametric
mask -- host C for the CPU pipe, an OpenCL kernel for the GPU pipe -- which are
maintained by hand and have drifted. Flexi has **one**: `mode_drawn` is
`mask_mode & (DEVELOP_MASK_MASK | DEVELOP_MASK_FLEXI)` (blend.c:1343), so a flexi
group is rendered by `dt_masks_group_render_roi()` on the host in *both* pipes,
`DT_MASKS_PARAMETRIC` members included, and the result is uploaded like any drawn
mask. Migration clears `DEVELOP_MASK_CONDITIONAL`, so the blendif kernel has
nothing left to evaluate. That is why the migrated render agrees with the CPU.

It is mask *generation* that is unified, not the whole path: feathering, blur and
the details refinement still have separate CPU and GPU implementations
(`guided_filter_cl()` vs the host guided filter), `kernel_mask` still runs to
apply global opacity and combine, and a parametric form evaluates against
`dev_in`/`dev_out` read back from the device, which can itself differ slightly
from the CPU-computed image. That residue is what the non-zero migrated CPU/GPU
gap is (0.00093 on AMD, 0.0025 on Apple) -- below 1/255 everywhere in the corpus,
i.e. invisible, with `dev_gap_widened` 0 on both vendors.

~~Raster's 12.5% is consistent with the same root cause rather than a second
one: a raster mask is produced by another module's blend, so a producer with a
parametric mask that ran on GPU stores an already-divergent mask.~~ **Wrong** --
this was the assumption flagged as open question #2, and checking it is what
turned up cause A. Raster's divergence has nothing to do with the producer's
mask: it is the stale readback of section 4.1, gated on post-processing rather
than on anything the producer did.

Module counts (a symptom of which modules the contributors used with parametric
masks, not necessarily of which are implicated):

```
colorbalancergb 127   channelmixerrgb 42   diffuse 38   atrous 13   exposure 12
primaries 11   retouch 10   colorequal 9   highpass 4   colorharmonizer 2
denoiseprofile 2   blurs 1   sharpen 1   rgbcurve 1   censorize 1
```


## 5. Reproducing

A minimal corpus is checked in as `classic_opencl_outliers.json.gz` (35 edits: the
worst three per module across all three libraries, 15 modules, six of them at
the full 1.0 difference). `classic_opencl_outliers.provenance.json` maps each
back to the contributor and index it came from.

```
darktable --library :memory: --verify-masks classic_opencl_outliers.json.gz
```

Runs in seconds and writes `classic_opencl_outliers.json.report.json`. Compare
`dev_diff_before` against `dev_diff_after` per edit. Check the run actually used
the GPU: it prints either `[verify] OpenCL device N acquired` or `[verify] no
OpenCL device: CPU blend only`, and in the CPU-only case every `gpu_max_diff` is
0, which looks like "no divergence" but is really "no test".

On Windows, `darktable.exe` redirects stdout to
`%USERPROFILE%\Documents\Darktable\darktable-log.txt` and calls `FreeConsole()`,
so the terminal stays silent; redirect to a file (`> verify.log 2>&1`) to keep
the output, and use `start /wait` since cmd does not wait for a GUI-subsystem
binary. `darktable-cli` is not an alternative -- it requires input and output
filenames and exits before `--verify-masks` is parsed. Note also that if the
report file cannot be opened (Windows Controlled Folder Access protects
`Downloads`/`Documents` from unsigned local builds), the run still completes
successfully and writes nothing, with no error: confirm the
`[verify] per-edit report written to ...` line.

Single worst cases, all with `gpu_max_diff == 1.0`, `max_diff == 0`:

| corpus | index | module |
|---|---:|---|
| dudo | 371 | retouch |
| leonidas | 163 | rgbcurve |
| leonidas | 1256, 1258 | channelmixerrgb |
| gwbarn | 10 | colorbalancergb |
| gwbarn | 949 | diffuse |

The full corpora are not in the repo (they are 40-150 MB each); the harvest
files live outside it. `--verify-masks` accepts gzipped input directly.

Relevant machinery, all on the `masks_revamp` branch:

- `src/develop/masks/verify.c` -- the four-way replay. `_verify_edit()` renders
  and `_max_abs_diff()` compares; the mask itself is recovered from the real
  blend via `pipe->store_all_raster_masks`, so nothing about it is recomputed
  by the test.
- The replay is forced single-threaded (`omp_set_num_threads(1)`) for
  reproducibility -- see the comment at verify.c:1011. Do not remove it while
  chasing this; float reassociation across threads moved four verdicts by up to
  0.1 between runs.
- The image under the mask is a generated probe (`probe_image.h`), never a
  user's photo.


## 6. Suggested next steps

1. ~~Reproduce on a non-Apple OpenCL device.~~ **Done** -- see section 2. It
   reproduces on AMD; this is a darktable bug.
2. ~~Confirm the blendif-kernel hypothesis by dumping the mask from both
   paths.~~ **Done, and it was not the kernels** -- see section 4.2. Bucketing
   the corpus by mask mode *and* by active post-processing, rather than by mask
   mode alone, split the population in two; a standalone differential of the
   two JzCzhz implementations then showed they agree everywhere hue is
   defined.
3. ~~Treat group B as a separate question.~~ Resolved as the same thing seen at
   the other end: `_blendif_compute_factor` compares against slider limits with
   a hard `<=`, so a 5e-7 input difference at a slider edge swings the factor
   between 0 and 1. Group A is the hue population, where the input difference
   is large enough (up to 0.045) that both vendors land the same side.
4. ~~Check whether raster is downstream of the same cause.~~ **Done, and it is
   not** -- it is cause A, an unconditional-readback bug in
   `dt_develop_blend_process_cl()`, fixed and confirmed on a 76-edit corpus
   (42 divergent -> 0). This is the piece to take upstream first: it is a
   plain defect with a one-line fix, independent of everything else here.
5. ~~Measure the affected area, not just the worst pixel.~~ **Done** -- see the
   table in section 1. The split it revealed is worth carrying into the
   debugging: chase a **broad, faint** case and a **full-range, tiny-area** case
   separately, since they are unlikely to have the same cause.
6. Once localised, this is an upstream `master` bug report, not a masks_revamp
   one. The verifier can produce a clean before/after per edit for the issue,
   now on two vendors.


## 7. Provenance of the numbers

| corpus | edits | outliers | CPU failures | gap widened |
|---|---:|---:|---:|---:|
| dudo | 2429 | 50 | 0 | 0 |
| leonidas | 1722 | 126 | 0 | 0 |
| gwbarn | 1803 | 98 | 0 | 0 |

Three further corpora have since been checked -- `thad` (27,693 edits, 774
outliers), `christian_pfister` (7,765 edits, 305) and `zisoft` (281 edits, 5) --
bringing the totals to 41,730 edits and 1,358 classic-GPU outliers, still with
zero `dev_gap_widened` anywhere. The mask-mode pattern in section 4 was
recomputed over the first three corpora only; it has not been re-derived over
all six, though nothing in the later runs contradicts it.

Contributed mask data is specifications only -- no image content, filenames or
user text. See `masks_revamp_migration_confidence.md` for the campaign's own
running numbers, which count *migration* reliability and deliberately exclude
these outliers.
