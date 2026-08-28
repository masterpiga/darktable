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
