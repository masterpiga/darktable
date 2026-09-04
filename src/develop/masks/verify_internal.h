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

// The replay harness, shared between the harvest checks that render.
//
// verify.c owns it: it builds a dt_develop_t, an iop module, a pixelpipe and
// one piece around a harvested edit, over the generated probe image, and
// renders the mask through the real dt_develop_blend_process(). postedit.c
// needs exactly the same thing -- the difference between the two checks is
// what they do to the mask between renders, not how they render it.
//
// Everything here was file-static in verify.c and is declared only because a
// second caller now lives in another translation unit. Nothing outside
// src/develop/masks/ may use it: it exists for the --harvest-masks tooling,
// not for the pipeline.

#include "develop/blend.h"
#include "develop/imageop.h"
#include "develop/masks.h"
#include "develop/pixelpipe.h"

#include <glib.h>

G_BEGIN_DECLS

// masks are stored normalised, so the replay renders at a bounded size rather
// than the edit's own: 512 on the long edge keeps every shape's proportions
// and every parametric channel's behaviour while making a few thousand
// renders affordable
#define VERIFY_MAX_EDGE 512

typedef struct
{
  dt_develop_t dev;
  dt_iop_module_t module;
  dt_dev_pixelpipe_t pipe;
  dt_dev_pixelpipe_iop_t piece;
  gboolean module_loaded;
  gboolean dev_mutex_ready;
  dt_iop_roi_t roi;
  float *probe;
  // What the module under test "produced": the probe with a synthetic effect
  // applied (see _make_module_output). The blend mixes this with `probe`
  // according to the mask, so it is both what makes the rendered image respond
  // to the mask at all and what gives the blendif `_out` channels something of
  // their own to select on.
  float *modout;
  float *out;

  // the upstream module a raster edit reads its mask from, present only for
  // raster edits (see _attach_raster_source)
  dt_iop_module_t source_module;
  dt_dev_pixelpipe_iop_t source_piece;
  gboolean source_loaded;

  // OpenCL device to replay the GPU blend on, or -1 when this build/run has
  // no usable device (the CPU comparison still stands on its own)
  int devid;

  // whatever darktable.develop pointed at before this replay claimed it, put
  // back on cleanup -- see _replay_init
  dt_develop_t *saved_develop;

  // the canvas editing state a shape's modify_property() reads; dev.form_gui
  // points at it. See _replay_init for why it cannot simply be NULL.
  dt_masks_form_gui_t form_gui;
} replay_t;

/** Build a replay around one harvested edit. Returns NULL on success, or a
    static string naming what could not be set up. */
const char *_replay_init(replay_t *r,
                         const char *operation,
                         const dt_develop_blend_params_t *bp,
                         GList *forms,
                         const int full_width,
                         const int full_height,
                         const int width,
                         const int height);

/** Render the mask for the current blend_params/forms, into a caller-owned
    copy. Returns NULL if the blend published nothing. `image` (may be NULL)
    receives the rendered RGBA image the same way. */
float *_render_mask(replay_t *r, float **image);

/** free everything the replay allocated */
void _replay_cleanup(replay_t *r);

/** is every value in `m` the same? */
gboolean _is_uniform(const float *m, const size_t n);

/** the largest absolute difference between two buffers */
double _max_abs_diff(const float *a, const float *b, const size_t n);

G_END_DECLS

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
