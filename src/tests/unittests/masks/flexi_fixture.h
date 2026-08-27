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

// Mock environment for the flexi masks panel's model layer.
//
// The panel's group model -- which shapes form which groups, what each group's
// operator is, what is selected -- is a pure structure: a dt_masks_form_t of
// type DT_MASKS_GROUP whose `points` list holds one dt_masks_point_group_t per
// element, ordered bottom-up, partitioned into groups by the `group_start`
// flag (see _starts_group in blend_gui.c). No GTK widget is involved in any of
// it. The only global the model reaches for is darktable.develop, and only to
// resolve a formid to a form via dt_masks_get_from_id -- which just walks
// dev->forms.
//
// So the whole mock is: a dt_develop_t holding a forms list, an iop module
// pointing at it, and a blend_data for the panel's own scratch state. No
// gtk_init, no display, no database, no pixelpipe.
//
// LAYOUT STRINGS
//
// Building points lists by hand makes tests unreadable, so scenarios are
// written as layout strings that mirror what the panel shows, bottom group
// first:
//
//     "u:1,2 | i:3"
//
// is a union group holding elements 1 and 2 (1 at the bottom), then an
// intersection group holding element 3 above it. Operator letters are
// u(nion), i(ntersection), d(ifference), x = e(x)clusion, s(um).
//
// flexi_build() turns such a string into a live group; flexi_layout()
// serialises a live group back into one. A test is then a round trip through
// the model:
//
//     flexi_build("u:1,2 | i:3");
//     _model_drop_element_onto_element(mod, grp, 1, 3, TRUE);
//     assert_layout("u:2 | i:3,1");
//
// Serialising through _starts_group (rather than reading group_start
// directly) is deliberate: it is the same partition function the panel and
// the renderer use, so a layout assertion tests what the user will actually
// see, not what the flags happen to say.

#include "common/darktable.h"
#include "develop/blend.h"
#include "develop/blend_gui_internal.h"
#include "develop/imageop.h"
#include "develop/masks.h"

#include <glib.h>

// the fixture's live objects, valid between flexi_build() and flexi_teardown()
extern dt_develop_t flexi_dev;
extern dt_iop_module_t flexi_module;
extern dt_iop_gui_blend_data_t flexi_bd;
extern dt_develop_blend_params_t flexi_bp;

/** build a mask group from a layout string; returns the group. */
dt_masks_form_t *flexi_build(const char *layout);

/** the group built by the last flexi_build() */
dt_masks_form_t *flexi_group(void);

/** serialise the current group back to a layout string. Caller frees. */
char *flexi_layout(void);

/** stage an empty (member-less) group anchored above the run containing
    `below_fid` (INVALID_MASKID = unanchored, i.e. bottom of the list), and
    register it on the fixture's blend_data. Returns it; owned by the fixture. */
dt_masks_empty_group_t *flexi_add_empty(const dt_masks_state_t op,
                                        const dt_mask_id_t below_fid);

/** serialise the visual group order -- real runs and staged empties together,
    bottom-up -- as e.g. "u:1,2 | [i] | d:3". Caller frees. */
char *flexi_visual_order(void);

/** cmocka assertion on flexi_visual_order(). */
void flexi_assert_order_(const char *expect, const char *file, const int line);
#define assert_order(expect) flexi_assert_order_((expect), __FILE__, __LINE__)

/** remember `ord` as the displayed number of the group headed by `cid` */
void flexi_set_ordinal(const dt_mask_id_t cid, const int ord);
/** the remembered number for `cid`, or 0 */
int flexi_get_ordinal(const dt_mask_id_t cid);

/** bring up a scratch darktable.conf backed by a temp file, so tests can
    exercise code that reads panel preferences. Opt-in: only the suites that
    need it call this, and it is torn down by flexi_conf_cleanup(). */
void flexi_conf_init(void);
void flexi_conf_cleanup(void);

/** free everything the fixture allocated. Safe to call twice. */
void flexi_teardown(void);

/** cmocka assertion: current layout equals `expect`, with a readable diff. */
void flexi_assert_layout_(const char *expect, const char *file, const int line);
#define assert_layout(expect) flexi_assert_layout_((expect), __FILE__, __LINE__)

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
