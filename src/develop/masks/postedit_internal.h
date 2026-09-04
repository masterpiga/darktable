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

// The panel's controls, as something a check can apply to a group.
//
// postedit.c owns this: it enumerates every control the masks panel offers and
// applies one to a range of a group's point list, which is what a check needs
// to stand in for a user changing something. persist.c needs the same
// vocabulary -- the difference between the two checks is what they do around
// the change (postedit compares against a from-scratch group, persist compares
// against a save/reload), not what the change is.
//
// Sharing it is not just economy. If the two files each kept their own idea of
// what "set the group opacity" means they could drift, and the weaker of the
// two would then be reporting on a control the panel no longer has.
//
// Nothing outside src/develop/masks/ may use this: it exists for the
// --harvest-masks tooling, not for the GUI or the pipeline.

#include "develop/masks.h"

#include <glib.h>

G_BEGIN_DECLS

typedef enum
{
  // per run, broadcast across every member of it -- the fold reads all of
  // these from the run's head
  POKE_OP_UNION = 0,
  POKE_OP_INTERSECTION,
  POKE_OP_DIFFERENCE,
  POKE_OP_SUM,
  POKE_OP_EXCLUSION,
  POKE_OP_MULTIPLY,
  POKE_OP_SCREEN,
  POKE_WITHIN_UNION,
  POKE_WITHIN_SCREEN,
  POKE_WITHIN_ISECT,
  POKE_WITHIN_MULTIPLY,
  POKE_GROUP_BYPASS,
  POKE_GROUP_INVERT,
  POKE_GROUP_OPACITY,
  POKE_GROUP_REFINE,
  // per element
  POKE_ELEM_DISABLE,
  POKE_ELEM_HIDDEN,
  POKE_ELEM_INVERSE,
  POKE_ELEM_OPACITY,
  POKE_ELEM_REFINE,
  POKE_ELEM_BREAK,
  POKE_N
} poke_t;

// the first element-level poke; everything below it addresses a whole run
#define POKE_FIRST_ELEM POKE_ELEM_DISABLE

/* The other half of what the panel can do to a mask: change a SHAPE.

   A poke changes how a member is combined. None of them touches the geometry
   the member refers to, and geometry is not a detail the checks can wave away:
   which shapes overlap is what decides whether an intersection or a difference
   has anything to compute, so a whole class of operator is only ever exercised
   inertly on a corpus where the harvested shapes happen not to meet. Moving one
   shape onto another turns those from "compared, inert" into "compared, live".

   For the storage checks the stake is different and larger: geometry is the one
   part of a mask with a per-type serialised representation (a blob of
   dt_masks_point_<type>_t in masks_history), so a shape that has been edited is
   the only thing that exercises writing a point struct the harvest did not
   supply. path.c's resize even keeps a cached baseline alongside the points --
   state that a save has to either carry or reconstruct.

   These are the panel's own shape controls, driven the way the panel drives
   them: through functions->modify_property() for the sliders, and by moving
   points for the drags, which have no property. */
typedef enum
{
  GEOM_TRANSLATE = 0,   // drag the whole shape across the image
  GEOM_NODE,            // drag one node of a path/brush, deforming it
  GEOM_SIZE,
  GEOM_FEATHER,
  GEOM_HARDNESS,
  GEOM_ROTATION,
  GEOM_CURVATURE,
  GEOM_COMPRESSION,
  GEOM_N
} geom_t;

/** the shape control's label, for reports */
const char *_geom_label(const geom_t g);

/** Apply one shape control to `form`, which must be a leaf shape.

    Returns TRUE only if the form's points actually changed. Most shapes
    implement only some of the properties -- a circle has no rotation, a
    gradient no feather -- and modify_property() silently ignores the rest, so a
    caller that tallied every call would be reporting coverage it does not have.
    The return value is compared over the real point data rather than taken from
    modify_property's `count`, because a property can be accepted and then
    clamped back to where it already was. */
gboolean _apply_geom(dt_masks_form_t *form, const geom_t g);

/** A copy of `form`'s point list, for restoring it after a geometry sweep.
    Returns NULL for a form with no point_struct_size (a group, a raster or a
    parametric element), which is also what _apply_geom refuses to touch. */
GList *_geom_snapshot(const dt_masks_form_t *form);

/** Put a _geom_snapshot() back and free it. */
void _geom_restore(dt_masks_form_t *form, GList *snapshot);

// ---------------------------------------------------------------------------
// one panel action, addressed to part of a group
// ---------------------------------------------------------------------------

/* A poke and a geometry control say WHAT to change; a step says what and
   WHERE, and adds the two things that are neither -- deleting a member and
   reordering the list.

   This lives here rather than in the check that first needed it because two
   now do (--persist-masks and --undo-masks) and a third will. Each keeping its
   own would let them drift, and the weaker one would then be reporting on a
   panel that no longer exists -- the same argument the poke vocabulary above
   is shared for. */
typedef enum
{
  SCOPE_RUN = 0,    // the whole run the first member belongs to
  SCOPE_FIRST,      // the first element on its own
  SCOPE_LAST        // the last element on its own
} scope_t;

typedef enum
{
  STEP_POKE = 0,   // change a member's own state (poke_t in `k`)
  STEP_REMOVE,     // delete the member the scope names
  STEP_MOVE_UP,    // swap it with the member below it in the list
  STEP_GEOM,       // edit the SHAPE the member refers to (geom_t in `k`)
} step_kind_t;

/* `k` carries the poke for STEP_POKE and the geom_t for STEP_GEOM -- the two
   never appear in the same step, and a second field would have to be spelled
   out in every sequence initialiser just to say "unused". STEP_POKE is 0 so
   the sequences written before the other kinds existed keep their two-field
   initialisers. */
typedef struct { poke_t k; scope_t s; step_kind_t kind; } step_t;

#define GEOM_STEP(g, sc) { (poke_t)(g), (sc), STEP_GEOM }

/** Resolve a step's scope against `grp` as it stands, into [first, last].
    Returns FALSE if the group has no members to address. */
gboolean _resolve_scope(dt_masks_form_t *grp, const scope_t s,
                        int *first, int *last);

/** Apply one step to `grp` as it currently stands.

    `dev` resolves member formids to shapes, and is only read by STEP_GEOM.

    A structural step is a no-op on a group with a single member: removing it
    would leave an empty group and reordering it has nothing to swap with, and
    neither is a state the panel can produce either. */
void _apply_step(dt_develop_t *dev, dt_masks_form_t *grp, const step_t *st);

/** the step's label, for reports */
const char *_step_label(const step_t *st);

/** the control's label, for reports */
const char *_poke_label(const poke_t k);

/** Apply one poke to the member index range [first, last] of `points`.

    A run-level poke is broadcast across the whole range because that is what
    the panel does with it -- the fold reads it back from the run's head, but
    every member carries a copy so that any one of them can represent the group
    (see dt_masks_point_group_t's own comments on name/refinement/group_opacity).
    An element-level poke is passed first == last. */
void _apply_poke(GList *points, const poke_t k, const int first, const int last);

G_END_DECLS

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
