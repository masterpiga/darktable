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

// Reconstruction of a harvested edit (see harvest.h) back into live darktable
// structures. Shared by the two tools that consume a harvest file --
// --verify-masks (verify.h) and --roundtrip-masks (roundtrip.h) -- so that
// both read the format through exactly the same code. A second reader would be
// a second set of beliefs about the format, and a disagreement between them
// would show up as a difference in whichever tool was wrong.

#include "develop/blend.h"
#include "develop/masks.h"

#include <glib.h>
#include <json-glib/json-glib.h>

G_BEGIN_DECLS

/** Load a harvest file into a JsonParser, transparently decompressing it if it
    is gzipped. --harvest-masks writes FILE and FILE.gz and asks contributors to
    send the second, so the compressed one is what usually arrives; whether it
    is compressed is decided by the magic number, not the extension. Returns
    NULL on failure with `error` set; caller owns the parser. */
JsonParser *dt_masks_harvest_load(const char *path, GError **error);

/** A key identifying everything about one harvested edit that a replay depends
    on: the module, the blend params, the forms and the image dimensions --
    everything except where it sat in the harvest.

    Real libraries are full of exact repeats: a preset or a copied history
    applied across hundreds of images stores the same mask specification each
    time. 61% of the edits in the seven contributed corpora are byte-identical
    to an earlier one, 74% in the largest. Replaying them is not extra evidence
    -- the same specification rendered against the same probe by the same
    single-threaded code cannot produce a different answer -- so the checks
    render each distinct edit once and reuse the verdict for its repeats, while
    still counting every occurrence in the statistics.

    Deliberately content-based rather than shape-based. The configuration
    *shape* used for the reliability statistics (tools/masks_migration_confidence.py)
    excludes geometry and parameter values on purpose, and 330 of the 5,521
    recorded shapes contain more than one distinct outcome -- so sampling a few
    edits per shape would be sampling, with a real chance of missing the
    minority. Equal content is the only equivalence that is free.

    Caller owns the returned string. */
gchar *dt_masks_harvest_edit_key(JsonObject *edit);

/** Rebuild the form list for one harvested edit from its "forms" array.
    Returns NULL if anything is unreconstructable, so a malformed record is
    skipped rather than replayed as something subtly different. Caller owns the
    list (free with dt_masks_free_form). */
GList *dt_masks_harvest_read_forms(JsonArray *forms_arr);

/** Rebuild the classic blend params for one harvested edit from its "blend"
    object. Zeroes `p` first, so absent members take their zero value. */
void dt_masks_harvest_read_blend_params(JsonObject *b,
                                        dt_develop_blend_params_t *p);

G_END_DECLS

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
