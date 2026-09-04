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

// Does *editing* a migrated mask behave like editing the mask the panel shows?
//
// Run as `darktable --postedit-masks harvest.json`, on a --harvest-masks file.
//
// WHY THE OTHER CHECKS CANNOT ANSWER THIS
//
// --roundtrip-masks, --verify-masks and --styleapply-masks all judge a
// migrated edit *as authored*: does it survive a save/load, does it render the
// mask it rendered before, does it still apply as a style. All three stop at
// the moment migration finishes. None of them ever touches a control
// afterwards.
//
// That leaves a whole class of bug outside every one of them, and it is not
// hypothetical: issue #21905 was exactly this. A group inherited from a
// classic edit has no combine operator on its bottom member, and the fold and
// the panel resolved that differently -- the panel reading it as union, the
// fold as "no operator", so the fold split one group into two runs. The
// migrated edit rendered *identically* either way, because the split runs
// recombine to the same mask, which is why --verify-masks passed it and would
// have passed it on any number of harvested edits. The damage only appeared
// once the user changed something: every control that a run reads from its
// head -- within-group combine mode, group opacity, group refinement,
// invert-output -- was being applied to a group the renderer did not have, and
// silently did nothing.
//
// So the property this file checks is one step further on than the others:
//
//     for every panel-reachable edit P, and every migrated group G,
//         render(P(G)) == render(P(canon(G)))
//
// where canon(G) is the same group with the panel's own normalization applied
// (_normalize_group_operators() in blend_gui.c: every member's effective
// operator written out explicitly). canon(G) is what a group built from
// scratch in the panel looks like for the same shapes and the same operators,
// so the property is precisely "editing a migrated group does what editing the
// equivalent fresh group does".
//
// It is not circular. canon() is a transformation of the *data*, derived from
// the panel's reading of it; the comparison is on pixels out of the production
// blend. Nothing here computes what the mask ought to be.
//
// WHAT IS SWEPT
//
// Every group reachable from the module's mask group, nested ones included,
// in three phases:
//
//   A  one control at a time. Per run: the seven between-group operators, the
//      four within-group combine modes, bypass, invert-output, group opacity,
//      a group-scoped refinement. Per element: disable, hide, invert, opacity,
//      an element-scoped refinement, the first-class group break.
//
//   B  the run controls again, on a run that only exists once a break has been
//      made. Phase A addresses runs from the un-poked partition, so a run a
//      break created is never the target of a run-level control there -- and a
//      run-level control read from a run head is the exact shape of the bug
//      this file exists for. Only the two runs the break itself created are
//      swept: it splits one run in two and leaves every other run's extent and
//      head untouched, so the rest would repeat phase A.
//
//   C  the run-level modifiers in every combination: all fifteen non-empty
//      subsets of {bypass, invert-output, group opacity, group refinement},
//      each crossed with the four within-group modes. Those four are what make
//      one run of two members behave differently from two runs of one -- a
//      refinement applied once to a joined sub-mask rather than twice to two
//      halves, an opacity or an invert applied once rather than twice -- so
//      they are where a partition disagreement that survives every single
//      control could still show itself. The bug this check found on its first
//      run needed a modifier set to become visible at all.
//
// Combinations beyond those are deliberately not swept, and the reason is an
// argument rather than a budget. G and canon(G) differ in exactly one thing:
// the missing combine bit. A control either overwrites it -- the seven
// operators, after which the two sides are byte-identical data and anything
// composed on top stays identical -- or leaves it alone, in which case the
// delta between the two sides is unchanged. So composing controls cannot make
// the *data* differ in a new way; it can only drive the fold down a path
// neither reaches alone, and phases B and C are where those paths are.
//
// Each configuration is applied identically to both sides -- same member
// indices, same values -- so the only thing that can differ is how the
// renderer reads them. Every partition used to address a configuration comes
// from the panel's own _starts_group(), and is the same for both sides by
// construction, since normalization only writes out what _eff_group_op()
// already reported.
//
// An edit whose canon(G) is byte-identical to G is skipped rather than swept:
// both sides would be literally the same data, so every comparison is trivially
// equal. That is an exact argument, not a sampling heuristic, and the count is
// reported.
//
// CPU ONLY, deliberately. Group folding happens on the CPU for the GPU path
// too -- the OpenCL blend consumes a mask the CPU built -- so a second GPU
// replay would exercise the same fold twice and answer nothing this does not.

#include <glib.h>
#include <stdio.h>

G_BEGIN_DECLS

/** Replay every edit in the harvest file at `json_path`, sweeping the panel's
    controls over each migrated group and comparing against the same sweep on
    the normalized group. If `report_path` is non-NULL a per-edit JSON report
    is written there, carrying a per-control breakdown as well as the rows.

    Returns TRUE if every swept edit matched on every poke. */
gboolean dt_masks_postedit_harvest(const char *json_path,
                                   const char *report_path);

/** Same run, writing its report as the *body* of an already-open JSON object
    (`"source"`, `"edits"`, `"summary"` members, no enclosing braces) so it can
    be composed into one document -- see dt_masks_check_harvest(). `rf` may be
    NULL to run without a report. */
gboolean dt_masks_postedit_harvest_section(const char *json_path, FILE *rf);

G_END_DECLS

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
