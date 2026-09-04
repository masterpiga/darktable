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

// Does closing and reopening the image between two edits change the mask?
//
// Run as `darktable --library :memory: --persist-masks harvest.json`, on a
// --harvest-masks file. It WRITES to the database, so it is only ever safe
// against a scratch library.
//
// WHY THIS CHECK EXISTS
//
// Migration used to be persisted by halves. Reading a classic edit upgraded
// blend_params to mask_mode = FLEXI and that reached the database on every
// open, but the run-boundary markers dt_masks_normalize_flexi_groups() derives
// did not: they live on dev->forms, a deep copy, while the writer walks each
// history item's own snapshot. Migration only runs while the stored
// blendop_version is old, so the second load found a current-version flexi
// edit, normalized nothing, and rendered the mask wrong -- permanently, and
// without the user ever editing anything. Opening or exporting was enough.
//
// Not one of the other checks could see it. --verify-masks, --postedit-masks
// and --styleapply-masks all work in memory, on a single migration; they never
// ask the database anything. --roundtrip-masks does save and reload, but it
// compares *stored state*, so it can only catch a field that changes across
// the trip -- and it is checked against an invariant written by hand, which
// covers the run boundaries and nothing else.
//
// So the property here is about pixels, and needs no notion of what the mask
// ought to be:
//
//     for every sequence of panel edits e1..en applied to a migrated mask,
//         render(en(...e1(G))) == render(en(...(save/reload)...e1(G)))
//
// A save and a reload between two edits is something the user can do at any
// moment and darktable gives no indication of; if it changes the mask, some
// part of the mask's meaning is not being stored. That is exactly the bug
// above, stated without reference to migration at all -- which is the point,
// because the fix for it moved the markers into storage, so from now on they
// are *read* rather than re-derived and their idempotency stops being
// exercised on every load.
//
// WHAT IS SWEPT
//
// Not all combinations -- a bounded set of short sequences aimed at the seams
// where a save can lose something, listed in _sequences[] with the seam each
// one covers. In outline:
//
//   - a run boundary the user creates (an operator change, a group break) and
//     then reads back through a run-level control on the next edit. This is
//     the shape of the original bug, with the user in place of migration
//   - the base-case repair's disable bits, written by migration and then built
//     on
//   - the run-level modifiers (bypass, invert-output, group opacity, group
//     refinement) set on one side of a save and read on the other. Those four
//     are what make one run of two members behave differently from two runs of
//     one, so a partition that did not survive the save shows up in them
//   - single-step sequences as a floor: if one edit does not survive one save,
//     nothing longer means anything
//
// Both arms address members by resolving the scope (a run, the first element,
// the last) against their OWN current state rather than a partition computed
// once. That is deliberate and makes the check strictly more sensitive: if the
// save lost a boundary, the reloaded arm's next edit lands on a different set
// of members -- which is precisely what the user would experience, their next
// click going somewhere else. When the save is transparent the two states are
// identical and the resolution trivially agrees.
//
// Only the module's own top-level group is swept, not the nested ones. The
// storage path does not distinguish them -- masks_history stores one flat row
// per form, group or not -- so a nested group buys another traversal of the
// same code.
//
// WHAT IT CATCHES, MEASURED
//
// Reinstating the half-persisted migration (skipping _sync_forms_to_history()
// in migrate_legacy.c) makes it fail on 9 of the 22 swept edits of the
// migration_failures corpus, 169 of 528 sequences, with 21 of the 24 sequences
// firing on at least one edit.
//
// The three that never fire are the three built on a bypass:
// single:group-bypass, modifier:bypass-then-within and
// modifier:bypass-then-opacity. That is not a gap to close but a fact about
// the control: a bypassed group contributes nothing whichever way the members
// were partitioned, so no partition disagreement can show through one. They
// are kept because a bypass that failed to survive a save at all would show up
// against the un-poked baseline, which is a different failure and one nothing
// else here would see.
//
// CPU ONLY, for the reason given in postedit.h: the OpenCL blend consumes a
// mask the CPU built, so a GPU replay would exercise the same fold twice.

#include <glib.h>
#include <stdio.h>

G_BEGIN_DECLS

/** Replay every edit in the harvest file at `json_path`, applying each edit
    sequence twice -- once wholly in memory, once with a save and a reload
    between every step -- and comparing the rendered masks. If `report_path` is
    non-NULL a per-edit JSON report is written there.

    Returns TRUE if every swept edit matched on every sequence. */
gboolean dt_masks_persist_harvest(const char *json_path,
                                  const char *report_path);

/** Same run, writing its report as the *body* of an already-open JSON object
    (`"source"`, `"edits"`, `"summary"` members, no enclosing braces) so it can
    be composed into one document -- see dt_masks_check_harvest(). `rf` may be
    NULL to run without a report. */
gboolean dt_masks_persist_harvest_section(const char *json_path, FILE *rf);

G_END_DECLS

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
