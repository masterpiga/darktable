# Transitioning users from classic masks to flexi

How flexi replaces classic masks *across releases* — staged so that no stage
can lose a user's mask, and each stage has an explicit gate before the next.

**Status:** proposed
**Companion docs:** `masks_revamp_flexi_migration_plan.md` (the data
conversion itself), `masks_revamp_upstream_plan.md` (the PR sequence)

---

## 1. What this doc adds

The two companion docs cover *conversion* (how a classic blob becomes a flexi
form tree) and *staging* (how the diff reaches upstream as reviewable PRs).
Neither covers the transition: what happens to the edits already in a
quarter-million user libraries, over which releases, and what we do if it
goes wrong.

`masks_revamp_upstream_plan.md` sketches four stages as an aside ("e.g.:
default to flexi for new masks / offer migrate-to-flexi UI / auto-migrate /
remove classic"). That ordering is right. This doc makes it a plan: entry and
exit criteria per stage, the safety net each one needs, and the surfaces
beyond the darkroom that also carry blend params.

## 2. The constraint everything follows from

**Migration is a one-way door at file granularity, and the door is silent.**

`dt_dev_read_history_ext` treats a blob whose `blendop_version` it doesn't
recognise as unconvertible: `dt_develop_blend_legacy_params_ext` returns
failure for `old_version > new_version` (no branch matches), and
`develop.c:2709` falls back to `default_blendop_params`. For a user who
downgrades darktable, or opens a shared XMP on an older release, that is not
"the mask appears as classic" — it is **the mask is gone**, with no dialog and
no log line they will see. `DEVELOP_MASKS_VERSION` 10 is a second, independent
door on the masks blob.

This is true of every version bump darktable has ever done, and is normally
acceptable because the blast radius is one module's parameters. Here the blast
radius is *every masked edit in the library*.

Two properties of the current implementation soften it, and both should be
preserved deliberately rather than by accident:

- **Conversion is in-memory until the image is opened in darkroom.** Per
  `migrate_legacy.c`'s own header comment, the write into `main.masks_history`
  happens only on the darkroom-load path, where a real history `num` exists.
  Thumbnail regeneration and export convert for rendering without persisting.
  So the door closes per image, at a moment the user chose, not in a
  library-wide sweep on first launch. *(Worth an explicit test — it is load-bearing
  for this whole plan and currently rests on a comment.)*
- **Migration fails closed.** On any synthesis failure the module keeps its
  classic `mask_mode` and renders through the classic path. Never a dropped
  mask.

**Corollary for sequencing.** UI adoption is reversible and per-user; data
migration is irreversible and per-file. Ship the reversible thing first and
the irreversible thing last, with the widest possible evidence base behind it.
The branch today does the opposite — flexi is opt-in-ish while migration is
unconditional.

## 3. Stages

### Stage 0 — prove equivalence at library scale (pre-release)

The flexi suite is 37 scenarios in one colorspace on the export pipe. That is
the right shape of test but not the right *n* for an irreversible conversion.

**Do:** a bulk offline differ. For every image in a real library with
`mask_mode != 0`, render twice — classic build and flexi build, same
`darktable-cli` invocation — and compare with `count-diff-pixels` at threshold
zero, the same tooling the migration plan already specifies for its fixtures.
This is a script, not a feature; it never ships.

**Gate:** zero non-identical images across a corpus in the thousands, spanning
all four classic modes, or every non-identity explained and either fixed or
accepted in writing.

### Stage 1 — flexi ships, opt-in, nothing auto-converts

Classic stays the default editor. Flexi is reached by a preference, and
conversion becomes **explicit and per-module**: a "convert this mask to flexi"
action in the module's mask UI.

**Why explicit:** every user whose files cross the version door in this stage
chose to cross it, one mask at a time, and can be told what it means. That is
the difference between a bug report saying "flexi ate my masks" and one saying
"I converted and don't like the result."

**Needs:**
- A first-conversion confirmation stating plainly that the edit will not be
  readable by older darktable versions.
- The blend-params version bump held back — a user who never converts should
  still be writing v14 blobs. This is the one genuinely awkward piece of
  engineering in the plan and it is the price of the stage; see §5.

**Gate:** one full release cycle. Defect reports against flexi-converted edits
no worse than the classic baseline, and no report of a mask lost or changed by
conversion.

### Stage 2 — flexi is the default editor for *new* masks

New masks are created in flexi. Existing masks still open classic until
converted. Nothing converts on its own.

This is the stage where new users stop learning classic at all, which is what
actually retires it — well before any code is deleted.

**Gate:** one release cycle as the new-mask default. Documentation complete.
The Stage 0 differ re-run against a corpus that now includes real
user-authored flexi edits.

### Stage 3 — auto-migrate on darkroom open

What the branch implements today. Existing edits convert when opened.

**Do not ship this in the same release as anything else risky.** It is the
only irreversible stage, and if it has to be backed out, edits already
converted are stranded on a version no shipped release understands.

**Needs, in order of importance:**
1. **A flexi→classic downgrade converter, as project insurance.** Not a user
   feature — a tool that exists so that "revert the flexi release" is a
   recoverable decision rather than an abandonment of everyone who upgraded.
   It need only handle the representable subset: a flexi tree that is a single
   run of drawn shapes with classic-expressible combine operators maps cleanly
   back to `DEVELOP_MASK_MASK` + `mask_id`. Trees using between-group
   operators, parametric-as-form, or raster-as-form are not representable and
   should refuse rather than approximate. That subset is the overwhelming
   majority of migrated edits, because it is what migration case 2 produces —
   *zero transform*.
2. **A pre-migration backup prompt**, once, on first launch of the release:
   back up `library.db` and sidecars before the first image is opened.
3. **Release notes that lead with the one-way door**, not with the feature.

**Gate:** Stages 0–2 gates all still green, plus the downgrade converter
existing and tested against the Stage 0 corpus round-tripped.

### Stage 4 — remove classic

Delete the classic mode buttons and their callbacks in `blend_gui.c`, the
non-flexi rendering branches in `blend.c`, `libs/masks.c`, and eventually the
`raster_mask_*` scalars — as already outlined as Phase B of the migration plan.

`migrate_legacy.c` is **not** on this list. Like every other step in
`dt_develop_blend_legacy_params`, it is the permanent bridge for old files.

## 4. Surfaces beyond the darkroom

Blend params travel further than the history stack, and each of these routes
through the same `legacy_params` path — so each is a way for the version door
to catch someone who never opened a masked image:

| Surface | Risk | Handling |
|---|---|---|
| **Styles** (`styles.c:711`) | Highest. Styles are *shared on forums*. A style created on flexi, applied on an older release, drops the mask silently. | Stage 1 conversion prompt must mention it. Consider stamping styles with the blend version in their display name or import warning. |
| **Presets** (`presets.c:1097`) | Same mechanism, but presets rarely leave one machine. | Covered by the same converter; no extra work. |
| **XMP sidecars** | The dual-boot and cloud-sync case. Two darktable versions over one directory. | Release notes; nothing technical available. |
| **Thumbnails / export** | Convert in memory only, no persistence. | Verify this holds (§2) and keep it. |

## 5. One upstream fix worth doing regardless

The silent `default_blendop_params` fallback for an *unrecognised newer*
version is bad behaviour independent of flexi — it is how darktable has always
handled downgrades. It can't be retrofitted into already-released versions, so
it doesn't help this transition; it helps the next one.

Distinguishing "older version I can convert" from "newer version I cannot
understand", and for the latter logging loudly and flagging the image rather
than substituting defaults, is a small, self-contained, flexi-neutral change
of exactly the shape that Batch 1 of the upstream plan is already collecting.

## 6. What to do next, concretely

1. Correct the stale status line in `masks_revamp_flexi_migration_plan.md`
   ("proposed, unimplemented" — it is implemented).
2. Write the Stage 0 bulk differ and run it against a real library. Everything
   in this plan is downstream of its result.
3. Decide Stage 1 vs Stage 3 for the *first* released version — i.e. whether
   the version bump ships gated or unconditional. This is the single decision
   the rest of the plan hangs on, and it is a judgement about appetite for
   irreversibility, not a technical question.
4. Build the flexi→classic downgrade converter before Stage 3, not after.
