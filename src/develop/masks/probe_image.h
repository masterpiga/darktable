/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

// A synthetic "probe" image for replaying harvested masks.
//
// The mask-migration verifier (see harvest.c) needs an image to render
// harvested masks against. It cannot use the user's own photo -- we
// deliberately never collect those -- and it must not use an image that makes
// masks vacuous. A parametric mask that selects, say, the top decile of
// luminance produces an all-zero mask on a picture that never gets that
// bright, and an all-zero mask compares equal to another all-zero mask no
// matter how badly the migration mangled it. Every such case is a test that
// silently passes.
//
// So the probe is generated, not shipped -- and the standard it is held to is
// deliberately not "whatever some library of real edits happens to use".
//
// That distinction matters more than it first appears. It is tempting to
// profile a large collection of real masks, see which channels they select on
// and how far the sliders travel, and build a probe that covers exactly that.
// The result would be a probe tuned to one person's habits: someone who never
// selects on hue, or never raises a boost factor, generates evidence that
// looks like proof those ranges do not need covering. The next user's library
// then quietly falls outside what was ever verified, and the failure mode is
// the silent one again -- vacuous passes, not visible errors.
//
// The bar is therefore set from the code and the colour space rather than from
// anyone's data: cover every channel blendif offers, across every value that
// channel can physically take.
//
// Rather than trying to hit each derived channel directly in its own
// perceptual space, the probe covers the linear-RGB cube densely: hue, chroma,
// luminance and the per-channel values are all functions of RGB, so covering
// the cube covers all of them at once, using the pipeline's own colour maths
// rather than a reimplementation of it here. What counts as adequate coverage
// is derived the same way: test_probe_image.c sweeps the cube to discover what
// each channel can actually take, and holds the probe to that.
//
// Two further properties do not follow from colour coverage, and are required
// separately:
//
//   - guided-filter feathering and detail masks both read structure out of the
//     guide image, so on a smooth image they degenerate towards no-ops and
//     their comparisons pass vacuously however wrong the migration was. The
//     probe therefore carries hard edges at several scales, and texture at
//     every octave a wavelet decomposition can look at.
//
//   - drawn masks sit at arbitrary normalised positions, so coverage has to be
//     spatially homogeneous: it is not enough for the image as a whole to span
//     the cube if the region under some particular ellipse is flat. The probe
//     is tiled so that each individual tile already sweeps a full 2D slice of
//     the cube, and neighbouring tiles walk the remaining axes on
//     low-discrepancy sequences, so any local window of a few tiles is close
//     to full coverage of the diffuse range.
//
// The image is scene-referred linear RGB, and deliberately contains values
// above 1.0 (a per-tile exposure ladder), because the working space is
// scene-referred and blendif boost factors reach up there.
//
// Generation is fully deterministic -- integer hash noise, no rand() -- so the
// same size always yields the same probe on every machine. That is what makes
// a verification run reproducible, and what lets a harvested file collected on
// one machine be replayed on another.
//
// The claim that this probe is actually adequate is not taken on faith: it is
// measured, per channel, by test_probe_image.c in the masks unit-test suite.

#include <glib.h>
#include <stddef.h>

G_BEGIN_DECLS

/** Fill `buf` with the probe image: `width` * `height` pixels, 4 floats each
    (RGBx, linear scene-referred; the 4th channel is set to 0). `buf` must hold
    at least width*height*4 floats. Deterministic for a given size. */
void dt_masks_probe_generate(float *const buf,
                             const int width,
                             const int height);

/** Allocate and generate in one step; free with dt_free_align(). Returns NULL
    on allocation failure or non-positive dimensions. */
float *dt_masks_probe_new(const int width, const int height);

G_END_DECLS

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
