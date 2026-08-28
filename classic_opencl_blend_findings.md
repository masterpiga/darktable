# Classic blending: CPU/OpenCL divergence

Handoff note. This is **not** a masks_revamp bug — it is a pre-existing defect in
darktable's *classic* blending on OpenCL, found incidentally by the migration
verifier on the `masks_revamp` branch. It should be reproducible on `master`.

Everything below is evidence gathered on 2026-08-28. Where something is a
hypothesis rather than a measurement, it says so.


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


## 2. What is established, and what is not

**Established.** 274 distinct configurations from three independently
contributed libraries diverge, across 15 modules. Zero CPU failures and zero
cases where migration widened an edit's own CPU/GPU gap, in all three corpora --
so this is not the migration moving anything.

**NOT established: that this reproduces on other hardware.** A harvest file
contains only mask *specifications*; the rendering all happens locally. Every
one of these runs was on a single device:

```
DEVICE: 'Apple M4 Pro'   PLATFORM: Apple   OpenCL 1.2   DRIVER VERSION 1.2 1.0
```

What varies between the corpora is the mask configurations, not the GPU. An
earlier note in the working log claimed cross-hardware reproduction on the
strength of one contributor's higher rate; that was wrong and is retracted here.
**Confirming this on a non-Apple OpenCL stack is open question #1** -- it decides
whether this is a darktable kernel bug or an Apple OpenCL compiler bug, and those
have very different fixes.


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
`src/develop/blends/blendif_*.c`. That also explains why the *migrated* render
agrees with the CPU -- flexi represents a parametric mask as a
`DT_MASKS_PARAMETRIC` form, which the group renderer evaluates on the host, so
the migrated path stops calling the blendif kernel at all.

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
`dev_diff_before` against `dev_diff_after` per edit.

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

1. **Reproduce on a non-Apple OpenCL device.** Decides kernel bug vs Apple
   compiler bug. Nothing else is worth much until this is answered.
2. **Confirm the blendif-kernel hypothesis** by taking one full-range case
   (leonidas #163, rgbcurve) and dumping the mask from both paths --
   `-d masks -d opencl`, or `--dump-diff-pipe`, which exists for exactly this.
3. **Check whether raster is downstream of the same cause** by testing a raster
   consumer whose producer has a drawn-only mask; the hypothesis predicts no
   divergence there.
4. Once localised, this is an upstream `master` bug report, not a masks_revamp
   one. The verifier can produce a clean before/after per edit for the issue.


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
