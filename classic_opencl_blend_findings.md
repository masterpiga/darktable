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
| frequency | 1,358 / 41,730 edits (**3.3%**) | of real masked edits; 0% for drawn-only, 4.7-12.5% for modes using a parametric component |
| typical (median) | 0.014 | ~1.4 percentage points of module strength, about 3.5 steps on a 0-255 scale -- a faint edge or band in smooth gradients, invisible in texture |
| worst observed | 1.0 | fully masked vs not masked at all: the module applies at full strength where it should be absent, or vice versa |

**What that does NOT say: how much of the image is affected.** `_max_abs_diff()`
returns the single largest disagreement anywhere in the frame, so a 1.0 is
consistent with half the photo being wrong *and* with one pixel landing on the
wrong side of a threshold on a mask boundary. The per-edit rows carry
`mean_diff` and `differing_pixels` only for the CPU comparison, not for the GPU
one, so the affected *area* is currently unmeasured. Given the threshold
sensitivity documented in section 2 (group B), some of the full-range cases are
plausibly thin boundary effects rather than whole regions -- but that is a guess,
not a measurement.

Closing that gap is a small change and worth doing before this goes upstream: add
`gpu_mean_diff` and `gpu_differing_pixels` next to the existing CPU fields in
`verify.c` and re-run the corpora. "3.3% of edits, median 1.4% strength error,
N% of pixels affected" is a much harder claim to wave away than a worst-pixel
figure alone.


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


## 4. The lead

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

**Hypothesis (untested):** the divergence lives in the blendif OpenCL kernels,
in `data/kernels/blendop.cl`, against their host counterparts in
`src/develop/blends/blendif_*.c`.

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

Raster's 12.5% is consistent with the same root cause rather than a second one:
a raster mask is produced by another module's blend, so a producer with a
parametric mask that ran on GPU stores an already-divergent mask. Worth
confirming rather than assuming -- **open question #2**.

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
2. **Confirm the blendif-kernel hypothesis** by dumping the mask from both paths
   -- `-d masks -d opencl`, or `--dump-diff-pipe`, which exists for exactly this.
   Use a **group A** case: deterministic across vendors, so anything found is
   real rather than device noise. Edit 1 (`rgbcurve`, = leonidas #163) is both
   full-range and group A, which makes it the single best target.
3. **Treat group B as a separate question**, and only after A is understood. If
   fixing the group A divergence also removes the group B spread, there was one
   cause all along; if it does not, there is a genuinely precision-sensitive
   comparison in the kernel to find.
4. **Check whether raster is downstream of the same cause** by testing a raster
   consumer whose producer has a drawn-only mask; the hypothesis predicts no
   divergence there.
5. **Measure the affected area**, not just the worst pixel -- add
   `gpu_mean_diff` and `gpu_differing_pixels` to the per-edit report in
   `verify.c` and re-run the corpora. See the severity note at the end of
   section 1: without it, a `1.0` verdict cannot be told apart from a
   single-pixel boundary artefact, and that is the first thing an upstream
   reviewer will ask.
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
