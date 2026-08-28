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

#include "common/image.h"
#include "develop/blend.h"

#include <glib.h>

/* Putting a harvested edit into a throwaway database as *classic* history, so
 * that reading it back runs the real migration.
 *
 * Shared by --roundtrip-masks and --styleapply-masks, which both need an image
 * that exists only for the duration of one comparison. Every one of these
 * writes to the database, so they are only ever safe against a scratch library
 * (`--library :memory:`), never a real catalogue.
 */

/** Wipe history, masks_history and module_order for `imgid`. */
void dt_masks_scratch_wipe_history(const dt_imgid_t imgid);

/** Create (or replace) the scratch image row. There are no NOT NULL
    constraints on main.images, so only the few columns the history reader
    actually consults are filled. history_end is 1: callers seed a single
    history row, at num 0. */
void dt_masks_scratch_seed_image(const dt_imgid_t imgid,
                                 const int width,
                                 const int height);

/** Write one history row at `num`, plus its forms under the same num.

    `op_params` are the module's *defaults*, not the user's: the harvest
    deliberately records no module parameters (they are user data and say
    nothing about masks). That substitution is sound because nothing under test
    reads them -- migration and the mask code work on blend_params and the form
    tree -- but they cannot simply be omitted, since the history reader runs
    each module's own legacy_params on them and drops the row outright if the
    blob is the wrong size.

    `blendop_version` is the harvested one, so the row genuinely arrives as
    classic and the real migration runs on read.

    `multi_priority` is written as given. A second instance only survives the
    read if the image also has an iop-order entry for it -- see
    dt_masks_scratch_seed_iop_order(), which callers must invoke first for any
    non-zero multi_priority.

    Returns FALSE if the module is unknown or the insert failed. */
gboolean dt_masks_scratch_seed_history(const dt_imgid_t imgid,
                                       const int num,
                                       const char *operation,
                                       const int multi_priority,
                                       const int blendop_version,
                                       const dt_develop_blend_params_t *bp,
                                       GList *forms);

/** Give `imgid` an iop-order list that contains (operation, multi_priority).

    Only needed for a second or later instance. A scratch image otherwise gets
    the *default* iop order, which has one entry per module at instance 0 only;
    dt_ioppr_get_iop_order() returns INT_MAX for any other instance ("cannot get
    iop-order for X instance N"), and dt_dev_read_history_ext() then drops the
    row without a word. That silently produced a dev with no modules at all,
    and comparisons between two empty module lists pass no matter what
    migration did -- so this is not cosmetic: without it a multi-instance edit
    is not being tested, it is being skipped.

    Mirrors dt_ioppr_insert_module_instance(): the new entry goes immediately
    before the highest-instance entry already present for that operation, which
    is where darktable itself puts a duplicated module. A no-op for
    multi_priority 0. */
void dt_masks_scratch_seed_iop_order(const dt_imgid_t imgid,
                                     const char *operation,
                                     const int multi_priority);

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
