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

// rebuilding a harvested edit (see harvest.h) into darktable structures,
// shared by every tool that reads a harvest file, so that all read the format
// through the same code: a second reader could disagree with the first

#include "develop/blend.h"
#include "develop/masks.h"

#include <glib.h>
#include <json-glib/json-glib.h>

G_BEGIN_DECLS

/** load a harvest file into a JsonParser, decompressing it when it is
    gzipped, as the copy contributors are asked to send is. The magic number
    decides, not the extension. NULL on failure, with `error` set; the caller
    owns the parser */
JsonParser *dt_masks_harvest_load(const char *path, GError **error);

/** a key for everything a replay of one harvested edit depends on: the
    module, the blend params, the forms and the image size, but not its place
    in the harvest. A preset or a copied history stores the same mask on many
    images, and a repeat rendered the same way cannot render differently, so
    the checks replay each distinct edit once and count its repeats. Keyed on
    content, not on the configuration's shape: edits of one shape can render
    differently, so sampling by shape could miss one. The caller owns the
    string */
gchar *dt_masks_harvest_edit_key(JsonObject *edit);

/** member `k` of `o`; `dflt` if `o` is NULL or the member is absent, null or
    not a plain value */
gint64 dt_masks_harvest_obj_int(JsonObject *o, const char *k, const gint64 dflt);
float dt_masks_harvest_obj_float(JsonObject *o, const char *k, const float dflt);
const char *dt_masks_harvest_obj_str(JsonObject *o, const char *k, const char *dflt);

/** rebuild the forms of one harvested edit from its "forms" array. NULL if
    anything cannot be rebuilt, so that a malformed record is skipped rather
    than replayed as something else. The caller owns the list (free it with
    dt_masks_free_form) */
GList *dt_masks_harvest_read_forms(JsonArray *forms_arr);

/** rebuild the blend params of one harvested edit from its "blend" object.
    `p` is zeroed first, so absent members are 0 */
void dt_masks_harvest_read_blend_params(JsonObject *b,
                                        dt_develop_blend_params_t *p);

G_END_DECLS

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
