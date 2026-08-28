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
import json
import os
import sys

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
    if row.get("inert"):
        return "inert"

    if row.get("max_diff", 0) > EPS:
        return "cpu_fail"

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


def _load_report(harvest_path):
    """Return {'roundtrip': [...], 'verify': [...], 'styleapply': [...]}.

    Accepts the combined --check-masks report, falling back to the three
    single-tool reports so corpora checked before --check-masks existed are
    still aggregatable.
    """
    combined = harvest_path + ".check.json"
    if os.path.exists(combined):
        with open(combined) as fp:
            doc = json.load(fp)
        return {k: (doc.get(k) or {}).get("edits", []) for k in
                ("roundtrip", "verify", "styleapply")}

    out = {}
    for key, suffix in (("verify", ".report.json"),
                        ("roundtrip", ".roundtrip.json"),
                        ("styleapply", ".styleapply.json")):
        path = harvest_path + suffix
        if os.path.exists(path):
            with open(path) as fp:
                out[key] = json.load(fp).get("edits", [])
        else:
            out[key] = []
    return out


class Corpus:
    def __init__(self, harvest_path):
        self.name = os.path.basename(harvest_path)
        with open(harvest_path) as fp:
            self.edits = json.load(fp).get("edits", [])
        self.reports = _load_report(harvest_path)
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


def aggregate(corpora):
    shapes = {}
    for corpus in corpora:
        for index, edit in enumerate(corpus.edits):
            key = _shape(edit)
            shape = shapes.get(key)
            if shape is None:
                shape = shapes[key] = Shape(key, _features(edit))
            shape.edits += 1
            shape.contributors.add(corpus.name)
            for check, classify in CHECKS:
                for row in corpus.reports.get(check, []):
                    if row.get("index") == index:
                        shape.outcomes[check][classify(row)] += 1
                        break
    return shapes


def _strata(shape):
    """Every stratum this shape belongs to. A shape counts in several."""
    out = [("overall", "all configurations")]
    mode = shape.features["mask_mode"]
    if mode is not None:
        out.append(("mask mode", _bits(mode, MASK_MODE_NAMES)))
    combine = shape.features["mask_combine"]
    if combine is not None:
        out.append(("mask combine", _combine_name(combine)))
    for t in shape.features["form_types"]:
        if t is not None:
            out.append(("form type", _bits(t, FORM_TYPE_NAMES)))
    out.append(("instance", "second instance" if shape.features["multi_instance"]
                else "first instance"))
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
# output
# ---------------------------------------------------------------------------


def _print_group(title, rows, confidence):
    print("\n%s" % title)
    print("  %-26s %7s %7s %6s %9s  %s"
          % ("", "shapes", "edits", "fails", "contrib", "%d%% upper bound"
             % round(confidence * 100)))
    for name, r in sorted(rows, key=lambda kv: -kv[1]["shapes"]):
        bound = r["failure_rate_upper_bound"]
        if bound is None:
            shown = "-- too few (n<%d)" % MIN_USEFUL_N
        else:
            shown = "%.3f%%  (1 in %s)" % (bound * 100.0, f"{int(1/bound):,}")
        print("  %-26s %7d %7d %6d %9d  %s"
              % (name[:26], r["shapes"], r["edits"], r["failures"],
                 r["contributors"], shown))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("harvests", nargs="+", help="contributed harvest .json files")
    ap.add_argument("--json", metavar="FILE", help="also write the aggregate here")
    ap.add_argument("--confidence", type=float, default=0.95)
    args = ap.parse_args()

    corpora = [Corpus(p) for p in args.harvests]
    shapes = aggregate(corpora)
    summary = summarise(shapes, args.confidence)

    print("mask migration reliability, aggregated over %d contributed librar%s"
          % (len(corpora), "y" if len(corpora) == 1 else "ies"))
    for corpus in corpora:
        print("  %-40s %6d edits" % (corpus.name, len(corpus.edits)))

    overall = summary[("overall", "all configurations")]
    print("\n  distinct configuration shapes : %d  (from %d edits)"
          % (overall["shapes"], overall["edits"]))
    print("  migration failures            : %d" % overall["failures"])
    print("  classic-GPU outliers          : %d  (pre-existing OpenCL bug in"
          " classic blending, not migration)" % overall["classic_gpu_outliers"])
    print("  shapes proving nothing        : %d  (inert or skipped)"
          % overall["untested_shapes"])
    bound = overall["failure_rate_upper_bound"]
    if bound is not None:
        print("\n  => with %d failure(s) in %d independent shapes, the true"
              % (overall["failures"], overall["shapes"]))
        print("     migration failure rate is below %.3f%% (about 1 in %s)"
              " at %d%% confidence."
              % (bound * 100.0, f"{int(1/bound):,}",
                 round(args.confidence * 100)))

    for group in ("mask mode", "form type", "mask combine", "instance"):
        rows = [(name, r) for (g, name), r in summary.items() if g == group]
        if rows:
            _print_group("by %s:" % group, rows, args.confidence)

    thin = sorted(((name, r) for (g, name), r in summary.items()
                   if g != "overall" and r["failure_rate_upper_bound"] is None),
                  key=lambda kv: kv[1]["shapes"])
    if thin:
        print("\nnot enough data to bound (these are what to ask contributors for):")
        for name, r in thin:
            print("  %-26s %3d shapes from %d contributor(s)"
                  % (name, r["shapes"], r["contributors"]))

    if len(corpora) < 5:
        print("\nNOTE: %d contributed librar%s. Configurations within one library"
              % (len(corpora), "y" if len(corpora) == 1 else "ies"))
        print("      are correlated even after collapsing to shapes, so the bound"
              " above is")
        print("      optimistic as a statement about darktable users at large."
              " More libraries")
        print("      widen coverage far faster than more edits from the same one.")

    if args.json:
        with open(args.json, "w") as fp:
            json.dump({"corpora": [c.name for c in corpora],
                       "confidence": args.confidence,
                       "strata": {"%s: %s" % k: v for k, v in summary.items()}},
                      fp, indent=1)
        print("\naggregate written to %s" % args.json)

    return 0 if overall["failures"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
