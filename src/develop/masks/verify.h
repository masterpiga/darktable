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

// Migration verification: does a real edit render the same mask after being
// migrated to the flexi model as it did before?
//
// Run as `darktable --verify-masks harvest.json`, on a file produced by
// --harvest-masks (see harvest.h).
//
// WHY THIS EXISTS SEPARATELY FROM THE UNIT TESTS
//
// The masks unit suite is large and exhaustive over the enumerable input
// space -- every mask_mode bit combination, every mask_combine value, the
// whole INV/INCL/MASKS_POS algebra. But every one of those assertions is
// *structural*: that a parametric form was synthesized, that no classic mode
// bit survived, that MASKS_POS came out equal to INV^INCL. Not one of them
// looks at a pixel.
//
// So they establish that migration produces well-formed, internally
// consistent output. They cannot establish that it produces the *same mask*.
// A migration could satisfy every one of them and still shift a feathering
// guide, composite two shapes in the wrong order, or apply an opacity at the
// wrong level -- and nothing would notice. That is the failure this file is
// for, and it is the failure that actually matters, because it is silent:
// the edit loads, the mask looks plausible, and the module applies somewhere
// the user did not ask for.
//
// HOW IT AVOIDS TESTING ITSELF
//
// The obvious way to write this -- compute "what the mask should be" here and
// compare -- would be worthless. It would encode this file's beliefs about
// classic blending, which are the same beliefs migrate_legacy.c encodes, so a
// wrong belief would cancel out and the comparison would pass.
//
// Instead both renders go through dt_develop_blend_process(), the actual
// production blend path, unmodified. The mask is recovered from it by setting
// pipe->store_all_raster_masks, which makes the blend publish its finished
// mask into piece->raster_masks (see the tail of dt_develop_blend_process).
// Nothing about the mask is computed here; the same function that renders the
// user's photo renders both sides of the comparison.
//
// The image underneath is the generated probe (see probe_image.h) rather than
// the user's photo, which we deliberately never collect. That substitution is
// sound because masks are stored in normalised coordinates and, for the
// parametric ones, evaluated against whatever pixels are present -- but it is
// also the point where this whole exercise can quietly become vacuous, since
// two all-zero masks compare equal however wrong the migration was. Hence the
// probe's own coverage tests, and hence the "inert" bucket in the report
// below: an edit whose mask is uniform before migration proves nothing, and
// is counted separately rather than being allowed to pad the pass rate.

#include <glib.h>
#include <stdio.h>

G_BEGIN_DECLS

/** Replay every edit in the harvest file at `json_path`, rendering its mask
    before and after migration, and report. If `report_path` is non-NULL a
    detailed per-edit JSON report is written there, carrying both the per-edit
    rows and a "summary" object holding every figure the run prints, so the
    file stands on its own without the terminal output.

    Returns TRUE if every non-skipped edit rendered identically. */
gboolean dt_masks_verify_harvest(const char *json_path,
                                 const char *report_path);

/** Same run, writing its report as the *body* of an already-open JSON object
    (`"source"`, `"edits"`, `"summary"` members, no enclosing braces) so several
    tools can be composed into one document -- see dt_masks_check_harvest().
    `rf` may be NULL to run without a report. */
gboolean dt_masks_verify_harvest_section(const char *json_path, FILE *rf);

G_END_DECLS

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
