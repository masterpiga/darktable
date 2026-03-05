# Flexi masks: data model and expressivity

This document is about two additive changes to the existing masks data
model — turning raster and parametric masks into ordinary list members, and
letting a single list express more than one group instead of one long
chain — and the expressivity each one buys. Classic's entire vocabulary
sits strictly inside flexi's: everything a classic mask could express,
flexi expresses the same way, and flexi can express a great deal classic
never could. It covers how both changes are encoded, why they're encoded
that way, why classic turns out to be a strict subset of flexi, and what's
still open.

## Two things called "group" — pick one apart before anything else

Classic darktable already has a `DT_MASKS_GROUP` form: a form type whose
stored blob is a flat, ordered list of member shapes, each carrying its own
combine operator. Every classic multi-shape mask is one of these. This
document is not proposing a new form type; `DT_MASKS_GROUP` itself is
untouched.

What flexi adds is a second, finer-grained notion of grouping *inside* a
`DT_MASKS_GROUP`'s member list: a maximal stretch of adjacent members in
that list that share an effective operator (or are explicitly marked as one
unit — more on that below), which the UI and the renderer treat as a single
thing with its own name, opacity, refinement, invert, and bypass. A
`DT_MASKS_GROUP` with five members might contain one such stretch or three,
depending on how those members' operators line up. Classic had no concept
of this at all: it only ever had the flat list, folded strictly in order.

This document uses `DT_MASKS_GROUP` (the type name, set in code) whenever
it means the stored form — the data structure. It uses **group**, plain and
lowercase, the way a darktable user would say it: the thing they create,
name, and combine in the panel — the maximal-run concept above. The two
usually coincide (one `DT_MASKS_GROUP` form, one group), which is exactly
why the words collide; they stop coinciding the moment a `DT_MASKS_GROUP`
holds more than one group, which is one of the two changes this document
covers. Where the distinction matters, this document says which one it
means; where it doesn't (most of the time), "group" alone is enough.

## The one-line version

Flexi doesn't add a new kind of mask, and it doesn't change what a
`DT_MASKS_GROUP` form is. It makes two additive changes to what that
form's existing member list can do: raster and parametric masks become
ordinary list members instead of single, mutually-exclusive references
hanging off the module, and a single list can be read as more than one
group instead of one long chain folded left to right. No new table, no new
form entity, no new blob layout for the common case. Both changes are
entirely in what gets read out of bits that were already there, or added to
the *end* of structs that were already there.

## Motivation: what flexi is for

Classic masking folds a `DT_MASKS_GROUP`'s flat list of shapes strictly
left to right, each shape carrying its own combine operator. To know what
the resulting mask is, you have to replay the whole sequence: the meaning
of shape 5 depends on the accumulator state left behind by shapes 1 through
4. Nothing in the data represents "these three shapes are one idea" — that
grouping exists only in the user's head, and it's lost the moment they
reopen the module later and have to reconstruct it from the shape order.

Flexi's data model represents combining regions as buckets instead: a group
is an unordered set of members that fold together first (union by default,
or a different within-group combine mode if set), producing one sub-mask,
which then composites into the overall result with its own single
operator. The mental model is "here are my buckets, here's how the buckets
combine" instead of "replay this sequence and track the running total" — a
flat, two-level structure instead of an N-deep fold, matching how the panel
presents it (grouped rows, not a bare list).

Flexi makes a second change that this document hasn't introduced yet and
that has nothing to do with grouping: it turns a raster-mask reference and
a single parametric channel into ordinary members of a `DT_MASKS_GROUP`'s
list — `DT_MASKS_RASTER` and `DT_MASKS_PARAMETRIC`, one point struct each,
sitting in the member list next to circles and brushes (the mechanics are
in "Groups aren't stored" below). Classic has neither: raster is one
scalar reference hanging off the module, and parametric is one shared
multi-channel config, not a member of anything. Once they're list members,
each one is independently addressable — it has its own operator, its own
invert, its own position in the list — the same way a circle or a brush
always has. And critically, a *channel* becomes an element, not a
*configuration*: nothing stops the same channel (say, lightness) from
appearing as two separate elements with two different curves and ranges,
each combined differently with the rest of the mask — `lightness_a ∪
(lightness_b ∩ raster(retouch, spot 1))` is an ordinary list, not a special
case.

Two independent things are new as a result, and it's worth keeping them
apart: what becomes possible purely because raster and parametric turned
into list members, and what becomes possible because the list can be
subdivided into groups. The first doesn't need grouping at all.

**New, because raster and parametric are now ordinary members of the
list** — this is a change to what a list can *contain*, not to how the
list is read:

- **Multiple raster masks, combined.** Classic has exactly one raster
  reference — source module, instance, mask id, invert flag — four scalar
  fields, a single slot, and raster is its own mode, mutually exclusive with
  drawn or parametric (the only *combination* classic's mode enum defines at
  all is drawn-and-parametric — raster combines with nothing else). Flexi's
  raster element is a first-class member of a `DT_MASKS_GROUP`'s list, so a
  mask can reference several other modules' raster outputs — or the same
  module's raster output more than once, e.g. inverted once and not the
  other time — and combine them: `raster(retouch, spot 1) ∪
  raster(retouch, spot 2)`. This would work the same way even in a single
  ungrouped flat fold; it has nothing to do with subdividing the list, only
  with raster no longer being a single scalar reference on the side.
- **Multiple parametric elements, including multiple instances of the same
  channel, with real per-element combine logic, not one fixed AND.**
  Classic has one blendif config; every active channel computes its own
  per-pixel factor and they're all multiplied together — a hard-wired
  intersection across the active channel set, with no way to say "channel A
  instead of channel B" rather than "channel A and channel B," and no way
  to use the same channel twice with two different curves. Flexi's
  single-channel parametric elements are independent list members, so
  `red-channel ∪ green-channel`, `saturation ∩ ¬lightness`, or even
  `lightness(curve 1) ∪ lightness(curve 2)` — the same channel, two
  different ranges, combined by union instead of both being forced to
  narrow the same AND — are all ordinary masks, not something classic's one
  shared multi-channel curve editor can express regardless of how its
  channels are configured. Again, this follows from parametric elements
  becoming ordinary members with their own operator — a flat, ungrouped
  fold of them would already give you this; grouping is not what's doing
  the work.

Mixing raster, parametric, and drawn shapes freely in the same group — as
opposed to just in the same flat list — *does* depend on grouping, since
"the same group" is the concept being defined; that part belongs with the
list below.

**New, because the list can be subdivided into groups** — this is the
second of the two data-model changes this document covers; these all
depend on there being more than one group inside a single
`DT_MASKS_GROUP`:

- **Real parenthesized combination, not a flat left-to-right fold.**
  Classic folds shapes sequentially — each shape's own operator applies
  against whatever the accumulator holds so far, strictly in list order.
  There is no way to say "these two shapes act as one unit before the next
  operator sees them" — `A ∪ B ∩ C ∪ D` is unavoidably
  `(((A ∪ B) ∩ C) ∪ D)`, order-dependent all the way through. Flexi's
  groups are real parentheses: `(A ∪ B) ∩ (C ∪ D)` is two groups inside one
  `DT_MASKS_GROUP`, each folding its own members first, composited once
  each — order between groups still matters (composition isn't commutative
  in general), but *within* a group it doesn't, and the grouping itself is
  explicit instead of an accident of draw order.
- **Refining a composed result, not just one shape.** Classic's mask
  refinement (feathering/contrast/brightness/details) is per-shape, and
  that's the only scope it can ever mean, because classic has no group to
  refine. Flexi lets refinement target either one member's own mask before
  it folds in, or the *finished, already-composed* multi-shape group mask
  once — smoothing the seam between three unioned shapes as a single step
  has no classic equivalent, since classic never has more than one shape's
  mask in hand at a time to refine.
- **A group-level opacity multiplier.** A group-level opacity scales a
  whole group's finished sub-mask, on top of each member's own independent
  opacity — dialing back "the union of these four shapes" as one unit
  without touching any shape's individual weight. Classic only ever had
  per-shape opacity; there was no group to attach a second multiplier to.
- **Naming, muting, and soloing a group.** Classic's per-shape list worked
  one shape at a time — hide a shape, name a shape. Flexi's group-level
  name, mute/solo, and bypass (temporarily skip a whole group without
  deleting or renumbering anything) operate on a group as a unit, because a
  group is now something the UI and the data model both recognize as one
  thing, not N shapes that happen to share an operator.

## No schema change

The database table that stores masks is unchanged: one row per form, an
image id, a form id, a form type, a name, a struct-version number, and a raw
binary blob of "points" plus a count. That blob is just an array of
whatever fixed-size C struct that form type uses — a circle's points, a
path's points, or, for a `DT_MASKS_GROUP` form, one small record per
member: form id, parent id, an integer state field, a float opacity, and a
few more fields described below. That's the whole persistence story for
`DT_MASKS_GROUP` itself — unchanged. Flexi rides entirely inside the
per-member state field, an integer that was already a bitfield carrying
"is this member active," "is it visible," "is it inverted," and the classic
union/intersection/difference/sum/exclusion operator choice. Flexi bolts on
a handful of new bits in the same field for its own concerns — hidden,
screen, multiply, intersect, bypass, invert-the-group — all in
previously-unused high bits. Every one of them defaults to 0 in an edit
that predates it, and 0 was already the neutral "classic behavior" value
for that bit, so old blobs decode with zero special-casing and render
bit-identically.

The per-member record itself also grew a few fields over time, appended to
the end: a per-shape refinement block, a name (broadcast across a group so
any member of it can carry the group's display name), a group-level
opacity multiplier, and a group-boundary marker (see below). That's a
per-*type* struct-version number, not a database schema bump, and it's
handled the same additive way: whatever reads an old blob knows the
historic size for each old struct version, reads that many bytes per point,
and zero-fills the tail. Zero happens to be neutral for refinement, name,
and the group-boundary marker; it isn't for the opacity multiplier (0 would
mute the whole group), so that one field is explicitly back-filled to the
identity value for old data instead of being left at its zero-fill.

The other half of the picture is one new bit in the existing "mask mode"
field that already lived on every module's blend parameters — the field
that says whether a module's mask is off, drawn, parametric, raster, or
some combination. Flexi just adds one more value to that set. Same struct,
same size, no schema change needed for that either — flexi is additive to
a field that was already a bitmask.

### The enums and structs, before and after

The diffs below show the actual shape of the change: nothing removed,
everything appended.

The per-member state bitfield — classic already had the top half (a
member's own display/operator flags); flexi appends the bottom half, most
of it describing the group that member belongs to rather than the member
alone:

```diff
 typedef enum dt_masks_state_t
 {
   DT_MASKS_STATE_NONE          = 0,
   DT_MASKS_STATE_USE           = 1 << 0,
   DT_MASKS_STATE_SHOW          = 1 << 1,
   DT_MASKS_STATE_INVERSE       = 1 << 2,
   DT_MASKS_STATE_UNION         = 1 << 3,
   DT_MASKS_STATE_INTERSECTION  = 1 << 4,
   DT_MASKS_STATE_DIFFERENCE    = 1 << 5,
   DT_MASKS_STATE_EXCLUSION     = 1 << 6,
   DT_MASKS_STATE_SUM           = 1 << 7,
+  DT_MASKS_STATE_HIDDEN           = 1 << 8,  // this member is skipped entirely
+  DT_MASKS_STATE_SCREEN           = 1 << 9,  // within-group: combine by soft union vs...
+  DT_MASKS_STATE_MULTIPLY         = 1 << 10, // ...between-group: combine by multiplication
+  DT_MASKS_STATE_ISECT            = 1 << 12, // within-group: combine by intersection (min)
+  DT_MASKS_STATE_WITHIN_MULTIPLY  = 1 << 15, // within-group: combine by true multiplication
+  DT_MASKS_STATE_OP_SCREEN        = 1 << 13, // between-group: combine by soft union
+  DT_MASKS_STATE_OP_BYPASS        = 1 << 14, // whole group temporarily skipped
+  DT_MASKS_STATE_OP_INVERT        = 1 << 16, // invert the group's own finished sub-mask
 } dt_masks_state_t;
```

The per-member record stored inside a `DT_MASKS_GROUP` — classic already
had the first four fields; flexi appends the rest, one at a time as needs
came up:

```diff
 typedef struct dt_masks_point_group_t
 {
   dt_mask_id_t formid;
   dt_mask_id_t parentid;
   int state;
   float opacity;
+  dt_masks_refinement_t refinement;  // per-shape or per-group refinement
+  char name[128];                    // group display name, broadcast to every member
+  float group_opacity;               // group-level opacity multiplier
+  int group_start;                   // explicit group-boundary marker
 } dt_masks_point_group_t;
```

The mask-mode field on a module's blend parameters — classic already
enumerated every mode a mask could be in; flexi is one more value in the
same bitmask:

```diff
 typedef enum dt_develop_mask_mode_t
 {
   DEVELOP_MASK_DISABLED         = 0,
   DEVELOP_MASK_ENABLED          = 1,
   DEVELOP_MASK_MASK             = 1 << 1,  // drawn mask
   DEVELOP_MASK_CONDITIONAL      = 1 << 2,  // parametric mask
   DEVELOP_MASK_RASTER           = 1 << 3,  // raster mask
   DEVELOP_MASK_MASK_CONDITIONAL = (DEVELOP_MASK_MASK | DEVELOP_MASK_CONDITIONAL),
+  DEVELOP_MASK_FLEXI            = 1 << 4,  // flexi mask
 } dt_develop_mask_mode_t;
```

The set of form types a `DT_MASKS_GROUP`'s member list can hold — classic
already had shapes and nested `DT_MASKS_GROUP`s; flexi adds two form types
that behave like shapes for storage and composition purposes, even though
they don't draw anything:

```diff
 typedef enum dt_masks_type_t
 {
   DT_MASKS_NONE      = 0,
   DT_MASKS_CIRCLE    = 1 << 0,
   DT_MASKS_PATH      = 1 << 1,
   DT_MASKS_GROUP     = 1 << 2,
   DT_MASKS_CLONE     = 1 << 3,
   DT_MASKS_GRADIENT  = 1 << 4,
   DT_MASKS_ELLIPSE   = 1 << 5,
   DT_MASKS_BRUSH     = 1 << 6,
   DT_MASKS_NON_CLONE = 1 << 7,
   DT_MASKS_OBJECT    = 1 << 8,
+  DT_MASKS_PARAMETRIC = 1 << 9,   // a parametric (blendif) mask as a first-class form
+  DT_MASKS_RASTER     = 1 << 10,  // a raster mask (another module's output) as a first-class form
 } dt_masks_type_t;
```

Every one of these diffs is purely additive — nothing existing moved,
shrank, or changed meaning. That's what keeps old data readable without a
rewrite: an old blob is a valid prefix of what a new reader expects, not a
different, incompatible shape.

## Groups aren't stored — they're read off a DT_MASKS_GROUP's list

A `DT_MASKS_GROUP` is stored, as it always was: it's an ordinary form row
with its own form id, referenced from a module's blend parameters. What is
*not* stored anywhere is where one group ends and the next begins as a
separate entity — there's no group id, no group row. Rendering and editing
code walks the member list and decides a new group starts wherever the
*effective* operator changes, or wherever a member has the group-boundary
marker set. That second condition exists because two adjacent groups with
the *same* operator (say, two independent "union" groups back to back) are
otherwise indistinguishable from one bigger group — the marker is a
deliberate boundary flag for exactly that case, and it's a dedicated field
rather than a bit inside the shared state field: a group boundary is a
property of the *group*, not of the shape that happens to sit at its edge,
so it gets its own field instead of sharing one that's already doing
per-shape display/operator/polarity duty. Older edits, from before the
dedicated field existed, only had the borrowed bit; that bit is read into
the dedicated field the moment such a form is loaded, so those edits render
exactly as they always did.

Two new form *types* ride the same mechanism: a first-class parametric
element (a self-contained copy of a blendif channel config, so several
parametric elements can coexist and combine like shapes) and a first-class
raster element (a reference to another module's raster output — source op,
instance, mask id — so it composites like a shape instead of being a
single module-wide scalar reference). Both are just new form-type bits with
their own point struct, added to a `DT_MASKS_GROUP`'s flat list exactly
like a circle or a brush would be. Nothing about the storage cares that a
member happens to be "a parametric mask" instead of "a shape."

## Rendering: folding groups instead of chaining shapes

The flexi renderer walks a `DT_MASKS_GROUP`'s member list once. For each
detected group, it: unions (or screens) every visible member into one
sub-mask, applies the group's own refinement once, then composites *that*
into the accumulator with the group's operator, one time. An empty or
fully-hidden group contributes nothing — union/screen's identity element —
so a hidden or bypassed intersect group doesn't blank the whole mask, which
was possible in classic's ordered chain (an intersect shape anywhere in the
sequence zeroes everything composited before it). The classic sequential
fold is untouched, selected whenever the flexi mode bit is absent, so
legacy masks still render through the exact path they always did.

## Classic is a strict subset of flexi

The claim: everything classic can express, flexi can express the same way,
and flexi can express plenty that classic never could (the whole point of
"Motivation" above). Nothing about reading an old, classic-only edit
through flexi's model changes what an image looks like — classic's
vocabulary just turns out to be a small corner of flexi's. This was checked
both analytically — by construction, walking every classic configuration to
its flexi equivalent — and empirically — by rendering both and comparing
pixels. Classic has a small, enumerable space of configurations (which of
the four mask-mode combinations, times which of three independent polarity
flags are set), so "every classic mask" is not an appeal to intuition; it's
a finite matrix that was actually walked.

**How classic masks map onto flexi, in pseudo-code.** Classic's mask is one
`DT_MASKS_GROUP`'s flat, ordered list of shapes, each carrying its own
operator, folded left to right; on top of that sits a parametric (blendif)
config that multiplies in per-pixel, and a couple of global polarity flags.
A representative case — several shapes with different operators, plus a
multi-channel parametric mask, plus an inverted result — looks like this in
classic:

```
# classic: one DT_MASKS_GROUP, flat fold, parametric multiplies in, global invert
mask = 0
mask = union(mask, circleA)
mask = intersect(mask, pathB)
mask = difference(mask, circleC)
mask = mask * parametric(channel_L) * parametric(channel_C)   # one config, channels ANDed
result = invert(mask)                                          # global INV flag
```

The same result in flexi is still one `DT_MASKS_GROUP`, now holding two
groups instead of one flat fold, with the parametric channels split into
independent elements instead of one shared config, and the whole thing
invertible per-group instead of only globally:

```
# flexi: one DT_MASKS_GROUP, two groups, composited once each; parametric channels are peers
group1 = union(circleA, ...)              # circleA starts the group
group1 = intersect(group1, pathB)         # still one group, no boundary marker
group1 = difference(group1, circleC)

group2 = intersect(parametric(channel_L), parametric(channel_C))  # explicit peer group
      # (classic's "all active channels multiply" becomes an ordinary
      # intersection between two ordinary group members)

result = invert(multiply(group1, group2))   # OP_INVERT on the composing group
```

Nothing here is a special case in the data model — `group1` and `group2`
are both just "a group of members with an operator," exactly like every
other group in any `DT_MASKS_GROUP`; the parametric channels didn't need a
different kind of container from the shapes, just more members in a group
of their own.

A second representative case is classic's `INCL` flag, which flips whether
a parametric channel means "pixels where this channel matches" or "pixels
where it doesn't," and does so *at the same time* as inverting the drawn
content and/or inverting the final composite — three flags that interact
rather than stack cleanly:

```
# classic: three polarity flags, applied in a fixed, hard-coded order
drawn = brushA                                    # MASKS_POS may invert this
drawn = invert(drawn) if MASKS_POS else drawn
param = parametric(channel_S)                     # INCL may invert this
param = invert(param) if INCL else param
mask = drawn * param
result = invert(mask) if INV else mask
```

Flexi represents this as two ordinary groups (each holding one member) and
one ordinary per-group invert flag, with `INCL`'s effect folded into which
member gets built inverted rather than staying a separate runtime flag:

```
# flexi: same shapes, polarity resolved once at conversion time, not per-render
drawn_group = maybe_invert(brushA, MASKS_POS XOR INCL)
param_group = maybe_invert(parametric(channel_S), INCL)   # element-level invert
result = maybe_invert(multiply(drawn_group, param_group), INV XOR INCL)
```

Every classic configuration reduces to some instance of these two shapes —
one `DT_MASKS_GROUP` holding groups that combine shapes/parametric
elements/raster references with per-group operators, invert, and bypass —
which is exactly flexi's general vocabulary, not a set of special cases
bolted on to cover classic.

**Empirical verification methodology.** For each of the four mask-mode
combinations crossed with each combination of the three polarity flags, a
synthetic test image and a hand-built classic mask configuration were
rendered twice: once with a build that only knows the pre-flexi, classic
rendering path (mask mode never gets the flexi bit, so the old code path
runs untouched), and once with the current build, which converts that same
stored configuration to flexi on load and renders it through the new
group-fold path. The two output images were then compared pixel by pixel
and required to be bit-identical, not just visually close. Every
combination came back bit-identical on the first correct attempt except
one, which exposed a genuine bug in the polarity algebra above (an
under-handled interaction between `INCL` and the other two flags); once
fixed, that case also came back bit-identical. All of these scenarios are
kept as a permanent regression suite, so the same matrix gets re-verified
on every future change to either rendering path.

## Pros and cons of the approach actually taken

**Pro:** no forced conversion and no library-wide rewrite step. An older
darktable can still open a flexi-authored XMP for a module that never used
any flexi-only element — it just sees classic bits it understands. Even for
masks that do use flexi elements, the raw data sits in ordinary point blobs
an old reader would skip past (unknown form ids simply aren't referenced),
not something that corrupts or crashes it.

**Pro:** every additive bit is additive in practice, not just in name —
there is no code path where getting flexi wrong corrupts a classic edit,
because the classic renderer never looks at the new bits at all. Two
independent rendering paths sharing one storage format is a smaller trust
boundary than one shared path with a mode flag threaded through it.

**Con:** the whole design leans on encoding structure as a *sequence*
(operator runs in a flat list) instead of a real hierarchy. That's cheap to
read and free to store, but it means group membership is implicit — move
one point past a group-boundary marker with the wrong tool and you've
silently merged or split a group. Every mutation path has to be careful
about this; it's not enforced by the type system, only by discipline in the
editing code, which does funnel every mutation through a small number of
"re-derive and re-stamp the whole partition" helpers, precisely to keep
that discipline in one place rather than re-implemented at each call site.

**Con:** the per-shape state field carries a lot of independent concerns at
once — display flags, polarity, the between-group operator, the
within-group combine mode, mute, bypass, invert. It's comfortably inside a
32-bit field today, but every future *group*-level concept has to make a
deliberate choice: does this belong on the shared state field (if it's
genuinely per-shape), or does it need its own field the way the
group-boundary marker and the group-level opacity do (if it's genuinely
per-group)? Getting that judgment call wrong is exactly how a per-shape
field ends up carrying group-level meaning by accident.

## Alternatives that were available and weren't taken

**A real groups table**, first-class rows with their own id, referencing
member form ids — the group-level equivalent of what `DT_MASKS_GROUP`
already is for its members. This is the "obviously correct" relational
design, and it's what a from-scratch implementation would do. It was passed
over because it's the one design that *isn't* free: it needs a schema
change, a version bump on read for every existing image, and a real
answer for "what happens to a group row when its last member is deleted" —
none of which the flat-list-plus-marker approach has to solve.

**A JSON/serialized blob for the whole `DT_MASKS_GROUP`**, replacing the
fixed-size struct-per-point blob. More flexible, easier to extend without
the "struct grew, read historic stride" dance — but slower to parse at
scale (a heavily masked image can have hundreds of points), harder to
bulk-copy in and out, and a much bigger diff against the existing masks I/O
code for a feature that, so far, hasn't needed the extra flexibility.

**Skipping the dedicated group-boundary field and reusing a spare state bit
instead** (the approach the other additive flags above all take). It
wasn't taken for exactly the reason called out in "Pros and cons": a group
boundary is a property of the group, not of the shape sitting at its edge,
and every future group-level concept deserves that same judgment call made
deliberately rather than defaulted to "find a spare bit."

## Open questions

**No enforced invariant for group boundaries.** Group-boundary correctness
depends on every mutation path routing through one of the "re-derive and
re-stamp the whole partition" helpers; nothing in the type system stops a
new call site from mutating a `DT_MASKS_GROUP`'s member list directly and
skipping that step. Whether that's an acceptable amount of discipline to
require, or whether it should be enforced more structurally, is open.

**Future group-level concepts.** The group-boundary marker and the
group-level opacity each got their own field instead of a state bit because
they're properties of the group, not the shape at its edge. Whether that
standard should be written down explicitly (so it's applied consistently to
whatever comes next) or left as implicit precedent is open.
