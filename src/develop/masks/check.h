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

// One command for the whole migration check.
//
// --roundtrip-masks, --verify-masks and --styleapply-masks each answer a
// different question about the same harvest file (see their own headers for
// why none of the three subsumes the others). Run separately they are three
// invocations producing three report files, and a verdict that lives partly in
// each and partly in the terminal output -- which is exactly the shape that
// makes a contributor's run hard to act on once the terminal is gone.
//
// `--check-masks harvest.json` runs all three against one file and writes a
// single self-contained FILE.check.json:
//
//   { "source": ..., "darktable_version": ...,
//     "roundtrip":  { "edits": [...], "summary": {...} },
//     "verify":     { "edits": [...], "summary": {...} },
//     "styleapply": { "edits": [...], "summary": {...} },
//     "summary":    { "passed": bool, per-tool pass flags } }
//
// Order is deliberate: roundtrip first because it is the cheapest and names
// the field that broke, verify second because it is the expensive pixel-level
// pass, styleapply last. All three always run -- an early failure must not
// hide what the other two would have found, since a contributor's file may not
// come back a second time.
//
// Like --roundtrip-masks and --styleapply-masks it drives the real history
// writer against a scratch image id, so it needs `--library :memory:` and must
// never be pointed at a real catalogue.

#include <glib.h>

G_BEGIN_DECLS

/** Run the round-trip, verification and style-application checks over the
    harvest file at `json_path`, writing one combined report to `report_path`
    (may be NULL).

    Returns TRUE only if all three passed. */
gboolean dt_masks_check_harvest(const char *json_path, const char *report_path);

G_END_DECLS

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
