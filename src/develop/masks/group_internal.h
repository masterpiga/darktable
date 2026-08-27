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

// Internal seam for the group compositor (masks/group.c).
//
// These are the mask operators themselves -- the arithmetic behind every
// operator name the panel shows. They were file-static; they are declared here
// so the compositor test suite (src/tests/unittests/masks/test_flexi_compose.c)
// can pin their semantics directly, on small buffers with known values, instead
// of only inferring them from rendered images.
//
// That matters because these functions define behaviour the rest of the design
// leans on and states as fact: that a group's members combine order-
// independently (which is what lets a group be an unordered bag of shapes),
// that an empty group is the identity for its operator (which is what stops an
// empty intersect group blanking the whole mask), and that opacity and invert
// compose the same way for every operator.
//
// Not public API: no IOP or pipe code should reach for these. Everything
// outside group.c composites through dt_masks_group_render_roi().

#include <glib.h>
#include <stddef.h>

G_BEGIN_DECLS

/* Composite `newmask` into the `dest` accumulator, in place, over `npixels`.
   `opacity` scales the incoming mask; `inverted` complements it first (i.e.
   uses 1 - newmask). dest is both input and output. */

void _combine_masks_union(float *const restrict dest,
                          float *const restrict newmask,
                          const size_t npixels,
                          const float opacity,
                          const int inverted);
void _combine_masks_intersect(float *const restrict dest,
                              float *const restrict newmask,
                              const size_t npixels,
                              const float opacity,
                              const int inverted);
void _combine_masks_difference(float *const restrict dest,
                               float *const restrict newmask,
                               const size_t npixels,
                               const float opacity,
                               const int inverted);
void _combine_masks_sum(float *const restrict dest,
                        float *const restrict newmask,
                        const size_t npixels,
                        const float opacity,
                        const int inverted);
void _combine_masks_exclusion(float *const restrict dest,
                              float *const restrict newmask,
                              const size_t npixels,
                              const float opacity,
                              const int inverted);
void _combine_masks_multiply(float *const restrict dest,
                             float *const restrict newmask,
                             const size_t npixels,
                             const float opacity,
                             const int inverted);
void _combine_masks_screen(float *const restrict dest,
                           float *const restrict newmask,
                           const size_t npixels,
                           const float opacity,
                           const int inverted);

/** composite a finished group sub-mask with the group's own operator, once
    (opacity/invert are already baked into it, so op=1, inverted=0) */
void _flexi_apply_group_op(float *const restrict buffer,
                           float *const restrict grp,
                           const size_t npixels,
                           const guint group_op);

/** is every pixel (within float rounding) exactly 1.0? Used to keep a
    parametric channel still at its full range from counting as an active
    group member. */
gboolean _mask_buffer_is_uniform_one(const float *const restrict buffer,
                                     const size_t npixels);

G_END_DECLS

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
