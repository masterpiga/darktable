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

// Mask harvesting: export every mask configuration in a library to a JSON
// file, so that migration to the flexi mask model can be verified against real
// edits instead of only against ones we thought to invent.
//
// Run as `darktable --harvest-masks out.json`, honouring --library and
// --configdir to pick which library to read.
//
// TWO PROPERTIES THIS FILE IS RESPONSIBLE FOR
//
// 1. It is strictly read-only on the user's library.
//
//    This is not a matter of care while writing the SQL. A normal darktable
//    startup opens the library read-write, takes a lock on it, and will
//    silently upgrade its schema if it was made by an older version -- so
//    merely pointing an ordinary run at someone's real library modifies it.
//    Harvesting therefore does not use darktable's own database handle at all.
//    It opens its own connection with SQLITE_OPEN_READONLY on a `file:...?
//    mode=ro&immutable=0` URI, runs before dt_database_init() is ever reached,
//    and exits without going near the rest of startup.
//
// 2. Nothing in the output identifies the user, their files, or their subjects.
//
//    We are asking strangers to send us this file, and "trust us" is not a
//    reasonable thing to ask. So the format is plain JSON with every value
//    decoded into named fields -- no base64, no opaque blobs -- specifically so
//    that anyone can open it in a text editor and confirm for themselves what
//    it does and does not contain. Everything in it is a number or a darktable
//    module name.
//
//    Getting that right takes more than not selecting the filename column.
//    Free text hides in three places, and all three are stripped here:
//
//      - `images.filename` / film roll paths, which leak names, places and
//        folder structure. Never queried.
//      - `masks_history.name`, e.g. "group `tone equalizer - eyes'". Usually
//        auto-generated, but the user can rename a shape to anything at all.
//        Dropped; the reader regenerates a canonical name from the type.
//      - `dt_masks_point_group_t.name[128]`, a user-typed group name that
//        lives *inside the points blob*. This one is the reason the format
//        decodes blobs rather than shipping them: a base64 dump of a group's
//        points would have carried it, and it would not have been visible to
//        anyone auditing the file.
//
//    Image identity is reduced to width and height, which is all the verifier
//    needs (masks are stored in normalised coordinates; the pixels underneath
//    are irrelevant and are replaced by a generated probe -- see
//    probe_image.h). Image ids are renumbered sequentially so they cannot be
//    correlated with anything outside the file.

#include <glib.h>

G_BEGIN_DECLS

/** Harvest every mask-bearing history entry in the library at `library_path`
    into a JSON file at `output_path`.

    Opens its own read-only connection; never writes to, locks, or upgrades the
    library. Returns TRUE on success. Progress and a summary go to stdout, so
    the user can see what was collected before deciding whether to share it. */
gboolean dt_masks_harvest_library(const char *library_path,
                                  const char *output_path);

G_END_DECLS

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
