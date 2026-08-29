# Migration failures found by the harvest campaign

**Status: FIXED (2026-08-29). Kept as the record of what the campaign found,
what actually caused it, and what the reproducer is now guarding.**

Six contributed libraries, 41,730 edits, turned up **11 edits (8 distinct
configuration shapes) where migration rendered a different mask than classic
did on the CPU**. Every one was a full-range failure (`max_diff = 1.0`), up to
174,592 differing pixels and a mean difference of 0.75 -- not float noise, a
visibly wrong mask.

Both are fixed in `migrate_legacy.c`. All seven libraries now re-check clean:
the worst CPU difference anywhere is 1.1e-03, well under the 1/255 visibility
threshold, and the bound moved from 0.290% to 0.060% (1 in 1,661). The two
headings below name what the campaign *thought* it had found; each now leads
with what the cause turned out to be.

Both devices agree with each other on these (`dev_diff_before` ~
`dev_diff_after` ~ 0), so this is not the classic-OpenCL divergence documented
in `classic_opencl_blend_findings.md`. It is migration.

Reproducer: `migration_failures.json.gz` (11 failures + 15 passing controls),
with `migration_failures.provenance.json` mapping each back to its contributor
and index.

```
darktable --library :memory: --verify-masks migration_failures.json.gz
```

The controls matter: they are configurations that look like the failures on
every coarse feature and pass anyway, so a proposed fix has to explain both
sides, not just make the failures go away.


## Bug A -- not nested groups: a member classic renders as a *replace*

**Cause.** A member with no `DT_MASKS_STATE_OP_COMBINE` bit is what
`dt_masks_group_add_form()` gives a group's *first* shape (`if(grp->points)
state |= default_operator` -- the first gets none, having nothing to combine
with). Classic's fold special-cases that position: `nb_ok == 0 || (state &
UNION)` unions it onto the empty accumulator. But when such a member ends up
*after* one that already rendered, classic falls through its whole if/else
chain to the final else -- `buffer[i] = op * mask[i]` -- which **replaces** the
accumulator and discards every earlier member.

Flexi cannot express that. `_flexi_apply_group_op()` maps an operator-less run
head to union, and the panel agrees: `blend_gui.c` repairs `(state &
OP_COMBINE) == NONE` to `DT_MASKS_STATE_UNION` on sight, calling it back-compat
that is "never valid for new edits". So the same bit pattern means *replace* to
the classic renderer and *union* to every part of flexi, and the mask classic
threw away came back after migration at full strength.

Nesting was only a correlate: grouping shapes is what produces a **second**
operator-less member. That is why every failure had nested groups and none
lacked them, while 849 of thad's 858 nested-group edits migrated exactly.

**Fix: fail closed** (`_group_has_replace_member()`), per this file's standing
rule -- the module keeps its classic mask_mode and classic renderer, which is
what it uses on master today. Across the seven libraries 52 edits (0.12%) carry
the pattern inside their own mask group; 9 rendered differently and the other 43
passed only because whatever replaced the discarded members happened to cover
them. All 52 now stay classic.

The awkward part is *when* it can be decided. On darkroom load the group's
members do not exist yet at migration time -- drawn-only migrates inline while
`dev->forms` still holds the previous image, and drawn+parametric is deferred to
a pass that by construction runs before the masks are read. So the check runs at
the one point where the members do exist, `dt_masks_normalize_flexi_groups()`,
and undoes the migration from a snapshot the split queue now carries. Style and
preset application, where `dev->forms` is loaded, decide up front instead.

### What the campaign originally observed

9 edits, all from `thad`, all `mask_mode = 3` (`uniform|drawn`), no parametric
and no raster involved. The module's mask is a group **whose members are
themselves groups**:

```
form 1780145258  GROUP  points=2        <- module's mask_id
   member 1780145119  (a GROUP)
   member 1780145168  (a GROUP)
form 1780145119  GROUP  points=1
   member 1780145107  (a path)
form 1780145168  GROUP  points=1
   member 1780145161  (a path)
```

Affected modules: colorbalancergb x4, diffuse x2, sharpen x2,
contrastntexture x1 (thad indices 8223, 8225, 8617, 8620, 21710, 21711, 21727,
21728, 26998).

**Necessary but not sufficient.** Every failure has nested groups and no failure
lacks them -- but thad alone holds 858 nested-group edits and 849 of them pass.
The failures all carry >= 2 members at the root group (2 to 4, never 1) where
676 of the 849 passes have exactly 1, which narrows it without separating it:
173 passing edits also have >= 2 members. Coarse feature analysis has gone as
far as it can; isolating the trigger needs an actual mask diff on one case.

Suggested first case: thad #21710 (colorbalancergb, 93,251 differing pixels,
structure shown above), against the controls in the reproducer.

**No earlier corpus contained a nested group at all.** dudo, leonidas and
gwbarn between them -- 5,991 edits -- exercise the case zero times. It took a
27,693-edit library to reach it.


## Bug B -- raster mask combined with MASKS_POS

**Cause.** Classic's raster branch is an `else if` *ahead* of the
drawn/parametric branch in `dt_develop_blend_process()`, so it reads none of
`mask_combine`: the mask is exactly `raster * opacity`. MASKS_POS never inverts
it, INV never reaches it (the `blendif_*_make_mask()` call that consumes INV
lives in the branch that does not run), and INCL only feeds a fallback fill that
branch owns. After migration the group renders *through* that later branch,
where MASKS_POS inverts it -- hence `max_diff = 1.0`, a fully inverted mask.

**Fix.** `_migrate_raster()` clears `INV | INCL | MASKS_POS`, since classic
consumed none of them. Both edits now render at 3.9e-05.

### What the campaign originally observed

2 edits, from `christian_pfister` (#7253, #7254, both colorbalancergb),
`mask_mode = 9` (`uniform|raster`) with `mask_combine = 4`
(`DEVELOP_COMBINE_MASKS_POS`).

**This one is total.** Across all six corpora there are exactly **2** edits with
that combination, and **both fail**:

| raster? | masks_pos? | failures | edits |
|---|---|---:|---:|
| no | no | 7 | 38,039 |
| no | yes | 2 | 2,912 |
| **yes** | **yes** | **2** | **2** |
| yes | no | 0 | 580 |

Raster alone is fine (580 edits, 0 failures) and MASKS_POS alone is nearly fine;
it is the pair that breaks. A 2/2 sample is far too small to call the rate, but
it is enough to say the combination is worth reading the migration code for
directly rather than sampling further -- and the small sample is itself the
point, since this combination is rare enough that no amount of ordinary testing
would have produced it.

Note both edits also carry a `mask_id` resolving to a real drawn group even
though `mask_mode` has no `DEVELOP_MASK_MASK` bit. Whether migration is
mishandling the raster path, the stale drawn group, or their interaction is
open.


## A ninth failure that was never real

For one round the document reported nine failing shapes, the extra one being
`exposure` / `uniform|flexi` from three style-apply rows in `dudo`. It was an
artefact of the harness, and the note is kept because the way it hid is worth
remembering.

`ROUNDTRIP_SKIP` and `STYLEAPPLY_SKIP` ended in `continue`, wrapped in the
usual `do { ... } while(0)`. `continue` binds to the nearest enclosing loop and
`do/while(0)` is one, so it left the macro instead of the edit loop and
execution fell through into the code the skip existed to avoid. Every skipped
edit was therefore *also* judged, and each affected report carried two rows for
that index. `--verify-masks` was unaffected: it skips by returning from
`_verify_edit()`, so it has no loop to mis-bind to.

The aggregator then indexed report rows by harvest index with a dict
comprehension, so the second row silently won -- and the second row is the
judgement the skip was there to prevent. Three already-flexi `dudo` edits with
stale mask ids were charged to migration as lost masks.

Both macros are now plain blocks, and `_index_rows()` in the aggregator refuses
a report that carries two rows for one index rather than resolving it. All
seven corpora were re-checked and re-recorded on the fixed build; the bound
returned to 0.290%.


## Why these were not caught earlier

- 205 existing unit tests pass on all 11. None of them looks at a pixel.
- The synthetic suite covers the enumerable input space -- mask mode bits,
  combine values, the INV/INCL algebra -- but not *structures* users build by
  hand, like a group of groups, nor rare mode/flag pairings like raster with
  MASKS_POS.
- Five of the six contributed libraries do not contain either case.

That is the argument for the campaign, in one line: the failures were found by
the only method that could have found them.


## Where this leaves the numbers

`masks_revamp_migration_confidence.md` reports **0 failing shapes out of 4,975,
bounding the migration failure rate below 0.060% (1 in 1,661) at 95%
confidence** -- down from 0.290% while the two bugs stood.

Do not read that as "migration is now proven correct". It is the same evidence
as before with two known defects removed; the standing caveat still holds, and
seven contributors remains the real limit on what the number can say.

The reproducer stays in the tree as a regression test. Re-run it after any
change to `migrate_legacy.c` or to the group fold:

```
darktable --library :memory: --verify-masks migration_failures.json.gz
```

It should report 26 replayed, 0 different. The 15 controls are the half that
matters: they look like the failures on every coarse feature and passed even
while the bugs were live, so a change that "fixes" the failures by breaking the
controls is caught here.
