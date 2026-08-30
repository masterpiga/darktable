#!/usr/bin/env python3
#
#   This file is part of darktable,
#   Copyright (C) 2026 darktable developers.
#
#   darktable is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   darktable is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with darktable.  If not, see <http://www.gnu.org/licenses/>.
"""Aggregate contributed mask-migration checks into a reliability bound.

Takes the (harvest.json, harvest.json.check.json) pairs collected from
contributors and answers the only question the campaign can actually answer:

    given everything we have tested, how large could the migration failure
    rate still be?

Usage:
    tools/masks_migration_confidence.py harvest1.json harvest2.json ...
    tools/masks_migration_confidence.py --json out.json harvest*.json

Each argument is a harvest file; its report is looked up as <harvest>.check.json
(or the older <harvest>.report.json + .roundtrip.json + .styleapply.json trio).


WHY THIS IS NOT JUST passed/total
---------------------------------

Three things make the naive pass rate wrong, and all three inflate it:

1.  Edits within one library are not independent trials. A user applies one
    preset, or copies one history stack, across hundreds of images: the corpus
    contains the same mask configuration many times over. Counting those as
    separate successes claims evidence that was never gathered. So the unit of
    observation here is a distinct configuration *shape* -- see _shape() -- and
    repeats collapse onto it. The edit count is still reported alongside, for
    context, but it is never what the interval is computed from.

2.  An edit whose classic mask is uniform ("inert") proves nothing: two
    constant masks compare equal however wrong the migration was. Those are
    excluded from the denominator rather than allowed to pad it.

3.  A failing *verify* verdict is not automatically a migration failure. The
    corpus so far shows a large population of edits where the CPU renders
    classic and migrated identically, the migrated GPU render agrees with the
    CPU, and only the *classic* GPU render is the outlier -- a pre-existing
    OpenCL bug in classic blending that migration exposes rather than causes.
    Those are counted in their own category, with their own bound, instead of
    being either silently dropped or charged to migration.

The interval itself is Clopper-Pearson (exact binomial), one-sided upper. With
zero observed failures it degenerates to the familiar rule of three: the 95%
upper bound on the failure rate is about 3/n. That is the honest shape of the
result -- "we have not found a failure" is evidence bounded by how much was
looked at, and this says how much.


WHAT IT STILL CANNOT TELL YOU
-----------------------------

Contributors are the real sampling unit and there are very few of them, so
even the shape-level interval is optimistic: two libraries from the same kind
of user explore the same corner of the input space. The per-stratum table
exists for exactly that reason -- a global number hides that one feature has
thousands of shapes behind it and another has four from a single person.
Strata below MIN_USEFUL_N are flagged rather than given a bound that would
read as reassurance.
"""

import argparse
import collections
import datetime
import gzip
import hashlib
import json
import os
import sys

# the running ledger, and the document regenerated from it, both at repo root
# alongside the other masks_revamp_* working documents
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LEDGER = os.path.join(REPO, "masks_revamp_migration_ledger.json")
DEFAULT_DOC = os.path.join(REPO, "masks_revamp_migration_confidence.md")
LEDGER_VERSION = 1

# a stratum with fewer distinct shapes than this gets no bound: at n=5 the 95%
# upper bound with zero failures is still ~45%, which is not a statement worth
# printing next to one derived from n=2000
MIN_USEFUL_N = 30

# mirrors VERIFY_EPS_EQUIVALENT in src/develop/masks/verify.c: one 8-bit step,
# below which a difference cannot be seen
EPS = 1.0 / 255.0

MASK_MODE_NAMES = {
    0: "disabled",
    1: "uniform",
    2: "drawn",
    4: "parametric",
    8: "raster",
    16: "flexi",
}

FORM_TYPE_NAMES = {
    1: "circle",
    2: "path",
    4: "group",
    8: "clone",
    16: "gradient",
    32: "ellipse",
    64: "brush",
    128: "non-clone",
}


def _bits(value, names):
    """Decode a bitmask into its set names, e.g. 7 -> 'uniform|drawn|parametric'."""
    out = [n for b, n in sorted(names.items()) if b and (value & b)]
    return "|".join(out) if out else str(value)


def _combine_name(value):
    parts = ["inv" if value & 1 else "norm", "incl" if value & 2 else "excl"]
    if value & 4:
        parts.append("masks_pos")
    return "|".join(parts)


# ---------------------------------------------------------------------------
# the unit of observation
# ---------------------------------------------------------------------------


def _shape(edit):
    """A configuration's *shape*: everything migration actually branches on.

    Deliberately not the exact configuration. Migration dispatches on mask
    modes, combine flags, form types and group structure; the geometry inside a
    circle rides along without selecting a different path. Keying on geometry
    too would leave the "same style pasted onto 500 images" case counted 500
    times, which is precisely the correlation that has to be collapsed.

    The cost is that two edits with the same shape and different geometry
    collapse into one trial even though a degenerate geometry could in
    principle fail alone. That direction is the safe one: it makes the interval
    wider, never narrower.
    """
    blend = edit.get("blend") or {}
    forms = edit.get("forms") or []

    types = collections.Counter(f.get("type") for f in forms)
    # group structure: how many members each group carries, and with which
    # operator states -- the flexi fold's own input
    groups = []
    for f in forms:
        pts = f.get("points")
        if isinstance(pts, list) and f.get("type", 0) & 4:
            groups.append(tuple(sorted(p.get("state", 0) for p in pts
                                       if isinstance(p, dict))))

    return (
        edit.get("operation"),
        blend.get("mask_mode"),
        blend.get("mask_combine"),
        blend.get("blend_cst"),
        1 if (edit.get("multi_priority") or 0) > 0 else 0,
        tuple(sorted(types.items())),
        tuple(sorted(groups)),
    )


# ---------------------------------------------------------------------------
# classifying one edit's outcome
# ---------------------------------------------------------------------------


def _verify_outcome(row):
    """Bucket a verify row.

    Returns one of:
      'inert'          -- uniform classic mask, proves nothing either way
      'ok'             -- migration preserved the mask
      'cpu_fail'       -- migration changed the mask on the CPU: a real failure
      'gpu_regression' -- migration widened this edit's own CPU/GPU gap
      'classic_gpu'    -- classic GPU render is the outlier, migration is not

    The last is the discriminator that keeps a pre-existing OpenCL bug in
    *classic* blending from being charged to migration. It is only reached when
    the CPU comparison is clean and the migrated render is the one agreeing
    with the CPU; anything else lands in a failure bucket.
    """
    if row.get("result") == "skipped":
        return "skipped"

    # Before the inert bucket, not after it. "Inert" means the *classic* mask is
    # uniform, which normally makes the comparison uninformative -- a flat mask
    # matches a flat mask however the two renderers got there. It stops being
    # uninformative the moment migration disagrees with it: a uniform classic
    # mask the migrated render does not reproduce is not weak evidence, it is a
    # module that went from doing nothing to applying everywhere. Testing inert
    # first swallowed exactly that case.
    if row.get("max_diff", 0) > EPS:
        return "cpu_fail"

    if row.get("inert"):
        return "inert"

    if row.get("gpu_ran"):
        before = row.get("dev_diff_before", 0.0)
        after = row.get("dev_diff_after", 0.0)
        if after - before > EPS:
            return "gpu_regression"
        if row.get("gpu_max_diff", 0.0) > EPS:
            return "classic_gpu"

    return "ok"


def _roundtrip_outcome(row):
    result = row.get("result")
    if result == "skipped":
        return "skipped"
    return "ok" if result == "same" else "fail"


# styleapply's per-edit rows carry a stable slug in "outcome"; older reports,
# written before that field existed, carry only the human-readable sentence in
# "result". Mapped explicitly rather than matched loosely -- an aggregator that
# guesses at prose silently miscounts, which is how 584 known-good rows were
# once charged to migration as failures.
_STYLEAPPLY_PROSE = {
    "ok": "ok",
    "drawn-only style, form never carried (same on master)": "not_carried",
    "style mask lost": "style_mask_lost",
    "host mask disturbed": "host_disturbed",
    "style item landed on no module at all": "no_module",
}


def _styleapply_outcome(row):
    result = row.get("result")
    if result == "skipped":
        return "skipped"

    outcome = row.get("outcome")
    if outcome is None:
        outcome = _STYLEAPPLY_PROSE.get(result)
        if outcome is None:
            raise SystemExit(
                "styleapply row %s has an unrecognised result %r. Refusing to "
                "guess whether that is a pass or a failure -- re-run the check "
                "with a current build, whose reports carry a stable \"outcome\" "
                "field." % (row.get("index"), result))

    # "a style never carries drawn geometry" is a property of styles that holds
    # on master too, not a migration outcome -- see styleapply.h. The edit is
    # still a real trial: it asserts migration did not leave the module claiming
    # a form that is not there.
    return "ok" if outcome in ("ok", "not_carried") else "fail"


# ---------------------------------------------------------------------------
# Clopper-Pearson one-sided upper bound
# ---------------------------------------------------------------------------


def _log_beta(a, b):
    import math
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b), by continued fraction.

    Written out rather than pulled from scipy so this runs against a bare
    interpreter -- contributors' reports should be aggregatable without a
    numerics stack being installed first.
    """
    import math
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)

    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return math.exp(a * math.log(x) + b * math.log1p(-x) - _log_beta(a, b)) * h / a


def upper_bound(failures, n, confidence=0.95):
    """One-sided Clopper-Pearson upper bound on the failure probability.

    With failures == 0 this is 1 - (1-confidence)**(1/n), i.e. the rule of
    three at 95%: roughly 3/n.
    """
    if n <= 0:
        return None
    if failures >= n:
        return 1.0
    if failures == 0:
        return 1.0 - (1.0 - confidence) ** (1.0 / n)

    alpha = 1.0 - confidence
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        # P(X <= failures | p=mid) = I_{1-mid}(n-failures, failures+1)
        tail = _betainc(n - failures, failures + 1, 1.0 - mid)
        if tail > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _read_json(path):
    """Load a JSON file, transparently decompressing a gzipped one.

    Decided by the magic number rather than the extension, matching
    dt_masks_harvest_load() in src/develop/masks/verify.c: the .gz a
    contributor sends may well arrive renamed by whatever service carried it.
    """
    with open(path, "rb") as fp:
        magic = fp.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    with opener(path, "rt", encoding="utf-8") as fp:
        return json.load(fp)


def _stem(harvest_path):
    """Report paths hang off the harvest path with any trailing .gz removed --
    see _masks_report_path() in src/common/darktable.c, which names them the
    same way so a corpus checked compressed and one checked unpacked produce
    the same report filenames."""
    return harvest_path[:-3] if harvest_path.endswith(".gz") else harvest_path


def _load_report(harvest_path):
    """Return {'roundtrip': [...], 'verify': [...], 'styleapply': [...]}.

    Accepts the combined --check-masks report, falling back to the three
    single-tool reports so corpora checked before --check-masks existed are
    still aggregatable.
    """
    stem = _stem(harvest_path)
    combined = stem + ".check.json"
    if os.path.exists(combined):
        doc = _read_json(combined)
        return ({k: (doc.get(k) or {}).get("edits", []) for k in
                 ("roundtrip", "verify", "styleapply")},
                doc.get("darktable_version"))

    out = {}
    for key, suffix in (("verify", ".report.json"),
                        ("roundtrip", ".roundtrip.json"),
                        ("styleapply", ".styleapply.json")):
        path = stem + suffix
        out[key] = _read_json(path).get("edits", []) if os.path.exists(path) else []
    return out, None


class Corpus:
    def __init__(self, harvest_path, label=None):
        self.path = harvest_path
        base = os.path.basename(_stem(harvest_path))
        self.name = label or (base[:-5] if base.endswith(".json") else base)
        self.edits = _read_json(harvest_path).get("edits", [])
        self.reports, self.dt_version = _load_report(harvest_path)
        if not any(self.reports.values()):
            raise SystemExit(
                "%s: no report found -- run `darktable --library :memory: "
                "--check-masks %s` first" % (self.name, harvest_path))


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

CHECKS = (
    ("roundtrip", _roundtrip_outcome),
    ("verify", _verify_outcome),
    ("styleapply", _styleapply_outcome),
)

FAILURE_OUTCOMES = {"fail", "cpu_fail", "gpu_regression"}


class Shape:
    """One distinct configuration shape, and everything seen about it."""

    __slots__ = ("key", "edits", "contributors", "outcomes", "features")

    def __init__(self, key, features):
        self.key = key
        self.features = features
        self.edits = 0
        self.contributors = set()
        self.outcomes = {name: collections.Counter() for name, _ in CHECKS}

    def verdict(self, check):
        """A shape fails a check if any edit sharing it failed."""
        counts = self.outcomes[check]
        if any(counts[o] for o in FAILURE_OUTCOMES):
            return "fail"
        if counts["classic_gpu"]:
            return "classic_gpu"
        if counts["ok"]:
            return "ok"
        return "untested"  # inert or skipped only


def _features(edit):
    blend = edit.get("blend") or {}
    forms = edit.get("forms") or []
    return {
        "operation": edit.get("operation"),
        "mask_mode": blend.get("mask_mode"),
        "mask_combine": blend.get("mask_combine"),
        "form_types": sorted({f.get("type") for f in forms}),
        "multi_instance": (edit.get("multi_priority") or 0) > 0,
    }


def _shape_id(key):
    """A stable identifier for a shape, so the ledger can accumulate across
    runs and across corpora. Hash of the canonical key rather than the key
    itself: it has to survive being written to JSON, where a tuple would not
    round-trip, and it keeps the ledger a fixed size per shape."""
    canon = json.dumps(key, sort_keys=True, default=str)
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:16]


def _index_rows(corpus_name, check, rows):
    """Index one check's report rows by harvest index, refusing duplicates.

    Each index must appear at most once: a check either skips an edit or
    judges it. Two rows for one index means the check emitted a skip and then
    ran the edit anyway, which a dict comprehension would resolve silently by
    keeping whichever row came last -- and the row that comes last is the one
    the skip was there to prevent. That is not hypothetical: a build made from
    a dirty tree emitted both for every "already flexi" edit, and three of
    dudo's stale mask ids were charged to migration as lost masks because the
    skip that should have excluded them was the row that got dropped.

    Raising is the point. A report is cheap to regenerate and the bound is
    quoted to contributors, so a malformed one has to stop the run rather than
    shift a number nobody would think to re-derive."""
    out = {}
    for row in rows:
        index = row.get("index")
        if index in out:
            raise SystemExit(
                "%s: %s reports two rows for edit %s (%r and %r). The report is "
                "malformed -- re-run `darktable --library :memory: --check-masks` "
                "with a build of the current tree, and record that."
                % (corpus_name, check, index,
                   out[index].get("result"), row.get("result")))
        out[index] = row
    return out


def digest(corpus):
    """Everything the ledger needs to keep about one corpus.

    Per shape: how many of its edits this corpus held, and the outcome counts
    per check. That is enough to recompute every interval later without the
    harvest -- which matters, because the harvests are large (150 MB is
    typical) and are not what gets kept."""
    shapes = {}
    features = {}
    by_index = {check: _index_rows(corpus.name, check, corpus.reports.get(check, []))
                for check, _ in CHECKS}

    for index, edit in enumerate(corpus.edits):
        key = _shape(edit)
        sid = _shape_id(key)
        features.setdefault(sid, _features(edit))
        entry = shapes.setdefault(sid, {"edits": 0, "outcomes": {}})
        entry["edits"] += 1
        for check, classify in CHECKS:
            row = by_index[check].get(index)
            if row is not None:
                counts = entry["outcomes"].setdefault(check, {})
                outcome = classify(row)
                counts[outcome] = counts.get(outcome, 0) + 1

    return {
        "recorded": datetime.date.today().isoformat(),
        "edits": len(corpus.edits),
        "darktable_version": corpus.dt_version,
        "shapes": shapes,
    }, features


def load_ledger(path):
    if not os.path.exists(path):
        return {"version": LEDGER_VERSION, "corpora": {}, "shape_features": {}}
    doc = _read_json(path)
    if doc.get("version") != LEDGER_VERSION:
        raise SystemExit("%s: ledger version %s, expected %d"
                         % (path, doc.get("version"), LEDGER_VERSION))
    return doc


def save_ledger(path, ledger):
    """One line per shape, not one per field. The ledger is committed and grows
    with every contributor, so it is written compactly -- but still line-broken
    at the shape, so a diff shows which shapes a re-check moved rather than one
    unreadable line."""
    def dump(obj):
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))

    with open(path, "w") as fp:
        fp.write("{\n")
        fp.write(' "version": %d,\n' % ledger["version"])
        fp.write(' "corpora": {\n')
        names = sorted(ledger["corpora"])
        for ci, name in enumerate(names):
            entry = ledger["corpora"][name]
            fp.write('  %s: {\n' % dump(name))
            for key in ("recorded", "edits", "darktable_version"):
                fp.write('   %s: %s,\n' % (dump(key), dump(entry.get(key))))
            fp.write('   "shapes": {\n')
            sids = sorted(entry["shapes"])
            for si, sid in enumerate(sids):
                fp.write('    %s: %s%s\n' % (dump(sid), dump(entry["shapes"][sid]),
                                             "," if si + 1 < len(sids) else ""))
            fp.write('   }\n')
            fp.write('  }%s\n' % ("," if ci + 1 < len(names) else ""))
        fp.write(' },\n')
        fp.write(' "shape_features": {\n')
        sids = sorted(ledger["shape_features"])
        for si, sid in enumerate(sids):
            fp.write('  %s: %s%s\n' % (dump(sid), dump(ledger["shape_features"][sid]),
                                       "," if si + 1 < len(sids) else ""))
        fp.write(' }\n')
        fp.write("}\n")


def record(ledger, corpora):
    """Merge corpora into the ledger, replacing any earlier record of the same
    one. Replacing rather than adding is the whole point: a corpus re-checked
    after a fix must not have its old outcomes counted alongside its new ones,
    and re-running the same file twice must not double its weight."""
    for corpus in corpora:
        entry, features = digest(corpus)
        previously = corpus.name in ledger["corpora"]
        ledger["corpora"][corpus.name] = entry
        ledger["shape_features"].update(features)
        print("  %-32s %5d edits, %4d shapes  (%s)"
              % (corpus.name, entry["edits"], len(entry["shapes"]),
                 "replaced" if previously else "new"))

    # drop features no surviving corpus refers to
    live = set()
    for entry in ledger["corpora"].values():
        live |= set(entry["shapes"])
    ledger["shape_features"] = {k: v for k, v in ledger["shape_features"].items()
                                if k in live}
    return ledger


def shapes_from_ledger(ledger):
    """Rebuild the Shape objects the summariser works on, from the union of
    every recorded corpus. A shape seen by several contributors is one trial,
    with all their outcomes merged onto it."""
    shapes = {}
    for name, entry in ledger["corpora"].items():
        for sid, rec in entry["shapes"].items():
            shape = shapes.get(sid)
            if shape is None:
                features = ledger["shape_features"].get(sid, {})
                shape = shapes[sid] = Shape(sid, features)
            shape.edits += rec["edits"]
            shape.contributors.add(name)
            for check, counts in rec.get("outcomes", {}).items():
                for outcome, n in counts.items():
                    shape.outcomes[check][outcome] += n
    return shapes


def failing_shapes(shapes):
    """The shapes that failed a check, for the document's own account of them.

    A document that reports "8 migration failures" and says nothing about what
    they are is not a summary, it is a number: whoever reads it still has to go
    and find out whether they are understood or merely counted."""
    out = []
    for shape in shapes.values():
        if not any(shape.verdict(c) == "fail" for c, _ in CHECKS):
            continue
        out.append({"operation": shape.features.get("operation"),
                    "mask_mode": shape.features.get("mask_mode"),
                    "mask_combine": shape.features.get("mask_combine"),
                    "contributors": len(shape.contributors)})
    out.sort(key=lambda f: (str(f["operation"]), f["mask_mode"] or 0,
                            f["mask_combine"] or 0))
    return out


def _strata(shape):
    """Every stratum this shape belongs to. A shape counts in several."""
    out = [("overall", "all configurations")]
    mode = shape.features.get("mask_mode")
    if mode is not None:
        out.append(("mask mode", _bits(mode, MASK_MODE_NAMES)))
    combine = shape.features.get("mask_combine")
    if combine is not None:
        out.append(("mask combine", _combine_name(combine)))
    for t in shape.features.get("form_types") or []:
        if t is not None:
            out.append(("form type", _bits(t, FORM_TYPE_NAMES)))
    out.append(("instance", "second instance"
                if shape.features.get("multi_instance") else "first instance"))
    return out


def summarise(shapes, confidence):
    table = collections.defaultdict(
        lambda: {"shapes": 0, "edits": 0, "failures": 0,
                 "classic_gpu": 0, "untested": 0, "contributors": set()})

    for shape in shapes.values():
        # a shape is a trial only if at least one check actually exercised it
        verdicts = {c: shape.verdict(c) for c, _ in CHECKS}
        tested = [v for v in verdicts.values() if v != "untested"]
        for stratum in _strata(shape):
            row = table[stratum]
            row["contributors"] |= shape.contributors
            row["edits"] += shape.edits
            if not tested:
                row["untested"] += 1
                continue
            row["shapes"] += 1
            if any(v == "fail" for v in verdicts.values()):
                row["failures"] += 1
            elif any(v == "classic_gpu" for v in verdicts.values()):
                row["classic_gpu"] += 1

    out = {}
    for stratum, row in table.items():
        n, f = row["shapes"], row["failures"]
        out[stratum] = {
            "shapes": n,
            "edits": row["edits"],
            "failures": f,
            "classic_gpu_outliers": row["classic_gpu"],
            "untested_shapes": row["untested"],
            "contributors": len(row["contributors"]),
            "failure_rate_upper_bound":
                upper_bound(f, n, confidence) if n >= MIN_USEFUL_N else None,
        }
    return out




# ---------------------------------------------------------------------------
# the running document
# ---------------------------------------------------------------------------

GROUPS = ("mask mode", "form type", "mask combine", "instance")


def _cell(name):
    """A stratum name inside a markdown table cell. The names are bitmask
    decodings like "uniform|drawn|parametric", and a bare pipe ends the cell --
    inside backticks included, which GFM does not exempt."""
    return "`%s`" % str(name).replace("|", "\\|")


def _rank(kv):
    """Sort key for a stratum row: most shapes first, then by name.

    The name tiebreak is not cosmetic. Without it, rows with equal shape counts
    come out in dict insertion order, which differs between a run that just
    recorded a corpus and one that reloaded the ledger from JSON -- so
    regenerating the document produced a spurious git diff every time."""
    return (-kv[1]["shapes"], str(kv[0]))


def _pct(bound):
    if bound is None:
        return "_too few_"
    return "%.3f%% (1 in %s)" % (bound * 100.0, f"{int(1 / bound):,}")


def _table(rows, confidence):
    out = ["| | shapes | edits | failures | contributors | %d%% upper bound |"
           % round(confidence * 100),
           "|---|---:|---:|---:|---:|---|"]
    for name, r in sorted(rows, key=_rank):
        out.append("| %s | %d | %d | %d | %d | %s |"
                   % (_cell(name), r["shapes"], r["edits"], r["failures"],
                      r["contributors"], _pct(r["failure_rate_upper_bound"])))
    return out


def render_doc(ledger, summary, confidence, failing=()):
    overall = summary[("overall", "all configurations")]
    corpora = ledger["corpora"]
    n_c = len(corpora)
    L = []

    L.append("# Mask migration reliability")
    L.append("")
    L.append("<!-- GENERATED FILE -- do not edit by hand.")
    L.append("     Regenerate with:")
    L.append("       tools/masks_migration_confidence.py --record HARVEST.json[.gz]")
    L.append("     which merges a newly checked corpus into")
    L.append("     masks_revamp_migration_ledger.json and rewrites this file. -->")
    L.append("")
    L.append("_Last updated %s, over %d contributed librar%s._"
             % (datetime.date.today().isoformat(), n_c,
                "y" if n_c == 1 else "ies"))
    L.append("")

    bound = overall["failure_rate_upper_bound"]
    L.append("## Where we stand")
    L.append("")
    if bound is None:
        L.append("Not enough data yet to bound the failure rate "
                 "(%d shapes, need %d)." % (overall["shapes"], MIN_USEFUL_N))
    else:
        L.append("**%d migration failure%s in %d distinct configuration shapes"
                 " &rarr; the failure rate is below %s at %d%% confidence.**"
                 % (overall["failures"], "" if overall["failures"] == 1 else "s",
                    overall["shapes"], _pct(bound), round(confidence * 100)))
    L.append("")
    L.append("| | |")
    L.append("|---|---:|")
    L.append("| contributed libraries | %d |" % n_c)
    L.append("| harvested edits | %d |" % sum(c["edits"] for c in corpora.values()))
    L.append("| distinct configuration shapes | %d |" % overall["shapes"])
    L.append("| migration failures | %d |" % overall["failures"])
    L.append("| classic-GPU outliers | %d |" % overall["classic_gpu_outliers"])
    L.append("| shapes proving nothing (inert/skipped) | %d |"
             % overall["untested_shapes"])
    L.append("")
    L.append("Classic-GPU outliers are counted separately on purpose: there the CPU")
    L.append("renders classic and migrated identically and only the *classic* GPU")
    L.append("render disagrees, which is a pre-existing OpenCL bug in classic")
    L.append("blending that migration exposes rather than causes.")
    L.append("")

    L.append("## What was measured")
    L.append("")
    L.append("The unit is a distinct configuration **shape** -- operation, mask mode,")
    L.append("combine flags, form-type multiset and group structure, i.e. everything")
    L.append("migration branches on -- not an edit. One preset applied across")
    L.append("hundreds of images is one thing tested, not hundreds, and counting")
    L.append("edits would claim several times the evidence actually gathered.")
    L.append("Geometry is deliberately not part of the shape: it rides along without")
    L.append("selecting a different code path, and including it would leave the")
    L.append("correlated case uncollapsed.")
    L.append("")
    L.append("Intervals are one-sided Clopper-Pearson. With zero observed failures")
    L.append("that degenerates to the rule of three, a bound of about 3/n; with")
    L.append("failures observed it widens accordingly, which is the interval doing")
    L.append("its job rather than the measurement regressing.")
    L.append("")

    if overall["failures"]:
        L.append("## Known failures")
        L.append("")
        L.append("The failing shapes are characterised, reproducible and **open**. See")
        L.append("`masks_revamp_migration_failures.md` for the analysis and")
        L.append("`migration_failures.json.gz` for a reproducer carrying passing")
        L.append("controls alongside them.")
        L.append("")
        L.append("| operation | mask mode | combine | contributors |")
        L.append("|---|---|---|---:|")
        for f in failing:
            L.append("| `%s` | %s | %s | %d |"
                     % (f["operation"],
                        _cell(_bits(f["mask_mode"] or 0, MASK_MODE_NAMES)),
                        _cell(_combine_name(f["mask_combine"] or 0)),
                        f["contributors"]))
        L.append("")
        L.append("A failing *shape* can stand for several failing edits: repeats of the")
        L.append("same configuration collapse onto it, exactly as passing repeats do.")
        L.append("")

    L.append("## Contributed corpora")
    L.append("")
    L.append("| corpus | recorded | edits | shapes | darktable |")
    L.append("|---|---|---:|---:|---|")
    for name in sorted(corpora):
        c = corpora[name]
        L.append("| `%s` | %s | %d | %d | %s |"
                 % (name, c.get("recorded", "?"), c["edits"], len(c["shapes"]),
                    c.get("darktable_version") or "-"))
    L.append("")

    for group in GROUPS:
        rows = [(name, r) for (g, name), r in summary.items() if g == group]
        if not rows:
            continue
        L.append("## By %s" % group)
        L.append("")
        L.extend(_table(rows, confidence))
        L.append("")

    thin = sorted(((name, r) for (g, name), r in summary.items()
                   if g != "overall" and r["failure_rate_upper_bound"] is None),
                  key=lambda kv: (kv[1]["shapes"], str(kv[0])))
    L.append("## Coverage gaps")
    L.append("")
    if not thin:
        L.append("Every stratum now has at least %d shapes behind it." % MIN_USEFUL_N)
    else:
        L.append("These strata have fewer than %d shapes, so no bound is quoted for"
                 % MIN_USEFUL_N)
        L.append("them -- at n=5 a zero-failure bound is still ~45%, which would read")
        L.append("as reassurance it has not earned. **This is the list to ask")
        L.append("contributors for.**")
        L.append("")
        L.append("| stratum | shapes | contributors |")
        L.append("|---|---:|---:|")
        for name, r in thin:
            L.append("| %s | %d | %d |"
                     % (_cell(name), r["shapes"], r["contributors"]))
    L.append("")

    L.append("## What this still cannot tell you")
    L.append("")
    L.append("Contributors are the real sampling unit and there are %d of them."
             % n_c)
    L.append("Shapes within one library stay correlated even after collapsing, so")
    L.append("the headline bound is optimistic as a statement about darktable users")
    L.append("at large. More *libraries* widen coverage far faster than more edits")
    L.append("from the same one.")
    L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# terminal output
# ---------------------------------------------------------------------------


def _print_group(title, rows, confidence):
    print("\n%s" % title)
    print("  %-26s %7s %7s %6s %8s  %s"
          % ("", "shapes", "edits", "fails", "contrib",
             "%d%% upper bound" % round(confidence * 100)))
    for name, r in sorted(rows, key=_rank):
        print("  %-26s %7d %7d %6d %8d  %s"
              % (name[:26], r["shapes"], r["edits"], r["failures"],
                 r["contributors"], _pct(r["failure_rate_upper_bound"])))


def report(ledger, summary, confidence):
    overall = summary[("overall", "all configurations")]
    n_c = len(ledger["corpora"])
    print("\nmask migration reliability, over %d contributed librar%s"
          % (n_c, "y" if n_c == 1 else "ies"))
    for name in sorted(ledger["corpora"]):
        c = ledger["corpora"][name]
        print("  %-32s %5d edits, %4d shapes  (recorded %s)"
              % (name, c["edits"], len(c["shapes"]), c.get("recorded", "?")))

    print("\n  distinct configuration shapes : %d  (from %d edits)"
          % (overall["shapes"], overall["edits"]))
    print("  migration failures            : %d" % overall["failures"])
    print("  classic-GPU outliers          : %d  (pre-existing OpenCL bug in"
          " classic blending, not migration)" % overall["classic_gpu_outliers"])
    print("  shapes proving nothing        : %d  (inert or skipped)"
          % overall["untested_shapes"])
    bound = overall["failure_rate_upper_bound"]
    if bound is not None:
        print("\n  => %d failure(s) in %d independent shapes: the migration failure"
              % (overall["failures"], overall["shapes"]))
        print("     rate is below %s at %d%% confidence."
              % (_pct(bound), round(confidence * 100)))

    for group in GROUPS:
        rows = [(name, r) for (g, name), r in summary.items() if g == group]
        if rows:
            _print_group("by %s:" % group, rows, confidence)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog="Harvests may be gzipped; the .gz a contributor sends is read "
               "directly.")
    ap.add_argument("harvests", nargs="*",
                    help="contributed harvest files (.json or .json.gz)")
    ap.add_argument("--record", action="store_true",
                    help="merge these harvests into the ledger and rewrite the "
                         "document (without it, nothing is written)")
    ap.add_argument("--label", action="append", default=[],
                    help="name to record a corpus under, in argument order "
                         "(default: the harvest's filename stem)")
    ap.add_argument("--ledger", default=DEFAULT_LEDGER)
    ap.add_argument("--doc", default=DEFAULT_DOC)
    ap.add_argument("--confidence", type=float, default=0.95)
    args = ap.parse_args()

    ledger = load_ledger(args.ledger)

    if args.harvests:
        labels = args.label + [None] * (len(args.harvests) - len(args.label))
        corpora = [Corpus(p, lab) for p, lab in zip(args.harvests, labels)]
        if args.record:
            print("recording into %s:" % os.path.relpath(args.ledger, REPO))
            record(ledger, corpora)
            save_ledger(args.ledger, ledger)
        else:
            # a dry run: report on these corpora alone, leaving the ledger be
            scratch = {"version": LEDGER_VERSION, "corpora": {}, "shape_features": {}}
            record(scratch, corpora)
            ledger = scratch
    elif not ledger["corpora"]:
        raise SystemExit("no corpora recorded yet, and none given. Run with a "
                         "harvest file, and --record to keep it.")

    shapes = shapes_from_ledger(ledger)
    summary = summarise(shapes, args.confidence)
    report(ledger, summary, args.confidence)

    if args.record:
        with open(args.doc, "w") as fp:
            fp.write(render_doc(ledger, summary, args.confidence,
                                failing_shapes(shapes)))
        print("\nledger   : %s" % os.path.relpath(args.ledger, REPO))
        print("document : %s" % os.path.relpath(args.doc, REPO))
    elif args.harvests:
        print("\n(dry run -- pass --record to merge this into %s)"
              % os.path.relpath(args.ledger, REPO))

    overall = summary[("overall", "all configurations")]
    return 0 if overall["failures"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
