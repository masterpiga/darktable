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
// type DT_MASKS_GROUP whose `points` list holds, bottom-up, one marker per
// group (see DT_MASKS_STATE_GROUP_MARKER) followed by that group's elements.
// No GTK widget is involved in any of it. The only global the model reaches
// for is darktable.develop, and only to resolve a formid to a form via
// dt_masks_get_from_id -- which just walks dev->forms.
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
//     "u:1,2 | [d] | i:3"
//
// is a union group holding elements 1 and 2 (1 at the bottom), an empty
// difference group above it, then an intersection group holding element 3.
// Operator letters are u(nion), i(ntersection), d(ifference), x = e(x)clusion,
// s(um).
//
// flexi_build() turns such a string into a live group, giving group n of the
// string the id FLEXI_GID(n); flexi_layout() serialises a live group back into
// one. A test is then a round trip through the model:
//
//     flexi_build("u:1,2 | i:3");
//     _model_drop_element_onto_element(mod, grp, 1, 3, TRUE);
//     assert_layout("u:2 | i:3,1");
//
// Serialising through _starts_group (rather than reading the marker bit
// directly) is deliberate: it is the same partition function the panel uses,
// so a layout assertion tests what the user will actually see.

#include "common/darktable.h"
#include "develop/blend.h"
#include "develop/blend_gui_internal.h"
#include "develop/imageop.h"
#include "develop/masks.h"

#include <glib.h>

// the id flexi_build() gives the n-th group of its layout string, bottom-up
#define FLEXI_GID(n) ((dt_mask_id_t)(900 + (n)))

// the fixture's live objects, valid between flexi_build() and flexi_teardown()
extern dt_develop_t flexi_dev;
extern dt_iop_module_t flexi_module;
extern dt_iop_gui_blend_data_t flexi_bd;
extern dt_develop_blend_params_t flexi_bp;

/** build a mask group from a layout string; returns the group. */
dt_masks_form_t *flexi_build(const char *layout);

/** the same group the way a classic edit, or a flexi one stored before group
    markers, holds it: no markers, each member carrying its group's operator,
    and a later group's first member its group_start */
dt_masks_form_t *flexi_build_classic(const char *layout);

/** the group built by the last flexi_build() */
dt_masks_form_t *flexi_group(void);

/** serialise the current group back to a layout string. Caller frees. */
char *flexi_layout(void);
/** the same for any group form, such as a nested group. Caller frees. */
char *flexi_layout_of(const dt_masks_form_t *g);

/** serialise group form `g` as a tree, the way a mask whose every group folds
    its members with one operator holds it (masks_revamp_nested_groups.md,
    Q8): "d{u{1,2},3}" is a difference group whose base is a union group of 1
    and 2, with 3 subtracted. A group's letter is its within-group operator:
    u(nion), i(ntersection), d(ifference), x = e(x)clusion, s(um),
    m(ultiply), o = screen. A trailing "~" is an inverted group or element,
    "@0.5" a group's or element's opacity. A list still holding several
    groups shows them separated by " | ". Caller frees. */
char *flexi_tree_of(const dt_masks_form_t *g);
/** cmocka assertion: the tree of group form `g` equals `expect`. */
void flexi_assert_tree_(const dt_masks_form_t *g,
                        const char *expect,
                        const char *file,
                        const int line);
#define assert_tree(g, expect) flexi_assert_tree_((g), (expect), __FILE__, __LINE__)

/** the between-group operator of the group element `fid` is in */
dt_masks_state_t flexi_group_op_of(const dt_mask_id_t fid);

/** remember `ord` as the displayed number of group `cid` */
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
/** cmocka assertion: the layout of group form `g` equals `expect`. */
void flexi_assert_layout_of_(const dt_masks_form_t *g,
                             const char *expect,
                             const char *file,
                             const int line);
#define assert_layout_of(g, expect) flexi_assert_layout_of_((g), (expect), __FILE__, __LINE__)

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
