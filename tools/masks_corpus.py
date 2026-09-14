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
"""The contributed mask corpus, as one committable SQLite file.

    tools/masks_corpus.py build   corpus.db harvest1.json.gz ...
    tools/masks_corpus.py add     corpus.db harvest9.json.gz
    tools/masks_corpus.py verify  corpus.db harvest1.json.gz ...
    tools/masks_corpus.py export  corpus.db outdir/ [library ...]
    tools/masks_corpus.py check   corpus.db outdir/ [library ...] --darktable BIN
    tools/masks_corpus.py stats   corpus.db

`check` runs --check-masks over the corpus in a pool of processes and writes,
per library, the harvest and the <harvest>.check.json a whole-library run
would have written, so masks_migration_confidence.py --record takes them as
they are.

WHY

--harvest-masks files are what contributors send, and they are big: twelve
libraries came to 25 MB compressed, 343 MB expanded, which is more than anyone
wants in a git repository and more than anyone wants to shuffle around to run a
check. Almost all of it is repetition. An edit records the module's whole
dev->forms snapshot, so the same drawn group is written out again for every
module that masks with it and every image the preset was applied to.

Measured over the twelve libraries we have:

    57144 edits          ->  22469 distinct   (39%)
    135540 form records  ->  33838 distinct   (25%)
    57144 blend records  ->  14043 distinct   (25%)

so the corpus is roughly four copies of itself. Content-addressing what repeats
and packing the numbers as the C floats they already are gets 25 MB down to
18 MB on disk and 5.9 MB as git stores it, with nothing thrown away.

Commit the .db, not a .gz of it: git compresses blobs anyway, so the repository
cost is the same, while an uncompressed file can be queried with sqlite3
directly and -- because `add` appends rather than rewriting -- deltas against
the previous version instead of replacing it wholesale.

WHAT IS AND IS NOT DROPPED

Dropped, because it is derived from a sibling and reconstructed on export: the
name strings a harvest carries next to a numeric code -- `type_name`,
`mask_mode_names`, `blend_cst_name`.

`points_count` looks like it belongs on that list and does not. For a form the
harvester decoded it is len(points); for one it could not decode it is the
count the stored header claimed, and the disagreement between the two is the
entire content of the record. It is stored.

Kept, in full: everything else.

That includes the two things it would have been easy to throw away. The first
is ORDER. dt_masks_harvest_edit_key() excludes `index` and `image_index` from
the identity of an edit, and the checks collapse exact repeats anyway, so
storing occurrence counts instead of occurrences would have kept every verdict
-- while permuting the exported file, moving which instance is the "first" and
which the "repeat", and making every report that names an edit by index name a
different one. edit_instance keeps one row per occurrence in the library's own
order, which costs about 0.2 MB and makes an export the file that went in.

The second is the forms the HARVESTER could not decode. Those carry
`points_error` instead of points, and 659 of them exist. "this library contains
masks darktable itself cannot read" is a finding; turning them into ordinary
forms here would erase it.

WHY float32 IS LOSSLESS

Every number in a harvest came out of a C float and was printed with %.9g, so
storing float32 loses nothing that the file ever held. Checked rather than
assumed, over all twelve libraries: of 1917843 float leaves, the number of
values for which `'%.9g' % float32(x) != text` is zero.

Note that a double comparison will NOT show this. `0.858859181` parses to a
double that differs from the float32 in the low bits while denoting the same
float32, so comparing the doubles reports a loss that is not there. `verify`
projects both sides through float32 for exactly this reason.

WHY NOT JUST GZIP HARDER

Because it was tried. SQLite is not compressed, so the first cut of this file
was 9.06 MB gzipped against 8.09 MB for the same data as normalised JSON --
binary floats are dense but incompressible, while repeated float TEXT gzips
beautifully. Byte-shuffle and delta filters made it worse. What actually paid
was removing the content-address columns (32 random bytes per row, plus an
index holding them again), packing the blend params, and keying the point
schema on (type, version).
"""

import argparse
import collections
import glob
import gzip
import hashlib
import json
import os
import struct
import sqlite3
import sys

FORMAT_VERSION = 1

# ---------------------------------------------------------------------------
# what the packer knows
# ---------------------------------------------------------------------------

# Every member of a harvested record this tool can carry. Anything else stops
# the build.
#
# This list is the whole correctness argument for the file. The packer walks a
# fixed set of members and writes columns for them, so a member it has never
# seen would be dropped without a word -- and the corpus would then be a lossy
# copy that still round-trips, still renders, and still compares equal to
# itself. Three fields were found this way while this was written
# (`dimensions_known`, `points_error`, `points_blob_bytes`), each only because
# a verification run failed and was chased down. A fourth would not have been
# noticed. So: unknown member, hard error, add it here deliberately.
KNOWN_EDIT = {'index', 'image_index', 'image', 'operation', 'multi_priority',
              'enabled', 'blendop_version', 'blend', 'forms'}
KNOWN_IMAGE = {'width', 'height', 'dimensions_known'}
KNOWN_FORM = {'formid', 'type', 'type_name', 'version', 'source',
              'points_count', 'points', 'points_error', 'points_blob_bytes'}

# Reconstructible from a numeric sibling, so not stored. Restored on export.
#
# `points_count` is deliberately NOT in here, although it looks like it belongs:
# for a form the harvester decoded it is exactly len(points), but for one it
# could not decode it is the count the stored header CLAIMED, and the
# disagreement with the blob length is the entire content of the record.
# Treating it as derived silently rewrote 83 of those to 0.
DERIVED = {'type_name', 'mask_mode_names', 'blend_cst_name'}

# The blend member that is a string rather than a number; it gets its own
# column instead of going through the packer.
BLEND_TEXT = 'raster_mask_source'

FMT = {'i': '<i', 'q': '<q', 'f': '<f'}
SZ = {'i': 4, 'q': 8, 'f': 4}
I32_MIN, I32_MAX = -2 ** 31, 2 ** 31 - 1

SCHEMA = """
PRAGMA page_size=4096;

CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE library(id INTEGER PRIMARY KEY, name TEXT UNIQUE, dt_version TEXT,
  format_version INT, blend_version INT, masks_version INT, added TEXT);

CREATE TABLE image(id INTEGER PRIMARY KEY, width INT, height INT,
  dimensions_known INT);

-- How a point of a given (type, version) is laid out in the `points` blob, as
-- {name: [kind, count]} with kind in i/q/f. Keyed on the VERSION as well as
-- the type because which fields a point carries is decided by the masks blob
-- version, not by the shape: a v8 group point has a `refinement` a v6 one does
-- not. Keying on type alone made the reader invent a zero-filled refinement
-- for every v6 group point -- semantically harmless (zero-fill is what
-- disabled means) but not the same record, and this file's whole claim is that
-- it is the same record.
CREATE TABLE point_schema(type INT, version INT, spec TEXT,
  PRIMARY KEY(type, version));

-- the same, for blend params, which are one flat record
CREATE TABLE blend_schema(seq INTEGER PRIMARY KEY, name TEXT, kind TEXT, n INT);

CREATE TABLE blend(id INTEGER PRIMARY KEY, raster_source TEXT, v BLOB);

CREATE TABLE form(id INTEGER PRIMARY KEY, formid INT, type INT, version INT,
  sx REAL, sy REAL, sz REAL,
  -- the point count the stored header CLAIMED. The same as the number of
  -- points for a form that decoded; for one that did not, it is the count that
  -- disagreed with the blob length, which is the whole reason the record
  -- exists -- so it cannot be derived from `points` and is stored.
  npoints INT,
  points BLOB,
  -- a form the HARVESTER could not decode: it recorded the failure instead of
  -- the points, and the checks skip the edit that contains it
  points_error TEXT, points_blob_bytes INT);

-- an edit's forms array, as its form ids packed little-endian uint32
CREATE TABLE form_set(id INTEGER PRIMARY KEY, members BLOB);

CREATE TABLE edit(id INTEGER PRIMARY KEY, operation TEXT, blendop_version INT,
  multi_priority INT, enabled INT, image_id INT, blend_id INT, form_set_id INT);

-- One row per edit AS IT APPEARED, in the library's own order.
--
-- Storing counts instead would be smaller and would keep every verdict, since
-- the checks collapse exact repeats anyway. It would also permute the exported
-- file: occurrences of one edit would come out grouped, so which instance is
-- the "first" and which the "repeat" moves, and every report that names an
-- edit by index would name a different one. Order is cheap (four ints per
-- occurrence, and they are near-sorted so they compress away), so the corpus
-- keeps it and an export is the file that went in.
--
-- image_index is carried for the same reason. Nothing reads it today -- the
-- harvester writes it and no check consults it -- but it is data a
-- contributor sent, and it costs a column.
CREATE TABLE edit_instance(library_id INT, seq INT, edit_id INT, image_index INT,
  PRIMARY KEY(library_id, seq)) WITHOUT ROWID;
"""

# Dedup indexes. Created for a build or an append and dropped again before the
# file is written out: they exist to make INSERT find an existing row, and
# nothing reads them afterwards. Keeping them would put a second copy of every
# points blob in the file for no gain.
DEDUP_INDEXES = [
    "CREATE UNIQUE INDEX ix_image ON image(width,height,dimensions_known)",
    "CREATE UNIQUE INDEX ix_blend ON blend(raster_source,v)",
    "CREATE UNIQUE INDEX ix_form ON form(formid,type,version,sx,sy,sz,npoints,"
    "points,points_error,points_blob_bytes)",
    "CREATE UNIQUE INDEX ix_form_set ON form_set(members)",
    "CREATE UNIQUE INDEX ix_edit ON edit(operation,blendop_version,"
    "multi_priority,enabled,image_id,blend_id,form_set_id)",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k not in DERIVED}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def _flat(rec, pre=''):
    """one level of nesting flattened to dotted keys (group points' refinement)"""
    out = {}
    for k, v in rec.items():
        if isinstance(v, dict):
            out.update(_flat(v, pre + k + '.'))
        else:
            out[pre + k] = v
    return out


def _unflat(rec):
    out = {}
    for k, v in rec.items():
        if '.' in k:
            a, b = k.split('.', 1)
            out.setdefault(a, {})[b] = v
        else:
            out[k] = v
    return out


def _kind(values):
    """How a field has to be stored, given every value ever seen for it.

    int32 unless something does not fit -- `blendif` is a uint32 bitfield and
    overflows -- and float32 as soon as one value is fractional.
    """
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in values):
        return 'f'
    return 'i' if all(I32_MIN <= x <= I32_MAX for x in values) else 'q'


def _promote(a, b):
    if a == b:
        return a
    return 'f' if 'f' in (a, b) else 'q'


def _pack(rec, spec):
    """`rec` (already flattened) packed per `spec`, a list of (name, kind, n)"""
    out = bytearray()
    for name, kind, n in spec:
        v = rec.get(name, 0 if n == 1 else [0] * n)
        vs = list(v) if isinstance(v, list) else [v]
        vs += [0] * (n - len(vs))
        for x in vs:
            out += struct.pack(FMT[kind], float(x) if kind == 'f' else int(x))
    return bytes(out)


def _unpack(blob, spec, count):
    out = []
    off = 0
    for _ in range(count):
        rec = {}
        for name, kind, n in spec:
            vals = []
            for _ in range(n):
                vals.append(struct.unpack_from(FMT[kind], blob, off)[0])
                off += SZ[kind]
            rec[name] = vals[0] if n == 1 else vals
        out.append(rec)
    return out


def _check_known(edit, where):
    def bad(got, known, what):
        extra = set(got) - known
        if extra:
            raise SystemExit(
                f"{where}: unknown {what} member(s) {sorted(extra)}.\n"
                f"  The packer would drop them silently. Add them to KNOWN_* in\n"
                f"  {os.path.relpath(__file__)}, give them a column, and rebuild.")
    bad(edit, KNOWN_EDIT, 'edit')
    bad(edit.get('image') or {}, KNOWN_IMAGE, 'image')
    for f in edit.get('forms') or []:
        bad(f, KNOWN_FORM, 'form')


def _load_harvest(path):
    opener = gzip.open if open(path, 'rb').read(2) == b'\x1f\x8b' else open
    with opener(path, 'rt') as fh:
        return json.load(fh)


def _edit_body(edit):
    """the edit as it is stored: derived and positional members removed"""
    return {k: _strip(v) for k, v in edit.items()
            if k not in ('index', 'image_index')}


def _edit_key(body):
    """the identity of an edit, matching dt_masks_harvest_edit_key()"""
    sig = json.dumps([body.get(m) for m in
                      ("operation", "blendop_version", "multi_priority",
                       "enabled", "image", "blend", "forms")],
                     sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(sig.encode()).hexdigest()


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------

def _infer_schemas(edits, point_schema=None, blend_schema=None):
    """Widen (or create) the packing schemas to cover `edits`.

    Widening rather than replacing, so `add` can bring a library whose floats
    land in a field an earlier one only ever had integers in.
    """
    ps = collections.defaultdict(dict)
    for k, v in (point_schema or {}).items():
        ps[k] = dict(v)
    bs = dict(blend_schema or {})

    for body in edits:
        for k, v in (body.get('blend') or {}).items():
            if k == BLEND_TEXT:
                continue
            vs = v if isinstance(v, list) else [v]
            kind = _kind(vs)
            cur = bs.get(k)
            if cur is None:
                bs[k] = [kind, len(vs)]
            else:
                cur[0] = _promote(cur[0], kind)
                cur[1] = max(cur[1], len(vs))
        for f in body.get('forms') or []:
            tv = (f['type'], f['version'])
            ps.setdefault(tv, {})
            for pt in f.get('points') or []:
                for k, v in _flat(pt).items():
                    vs = v if isinstance(v, list) else [v]
                    kind = _kind(vs)
                    cur = ps[tv].get(k)
                    if cur is None:
                        ps[tv][k] = [kind, len(vs)]
                    else:
                        cur[0] = _promote(cur[0], kind)
                        cur[1] = max(cur[1], len(vs))
    ps = {t: {k: v for k, v in sorted(s.items())} for t, s in ps.items()}
    bs = {k: v for k, v in sorted(bs.items())}
    return ps, bs


def _read_schemas(con):
    ps = {(t, ver): json.loads(s) for t, ver, s in
          con.execute("SELECT type,version,spec FROM point_schema")}
    bs = {n: [k, c] for _, n, k, c in
          con.execute("SELECT seq,name,kind,n FROM blend_schema ORDER BY seq")}
    return ps, bs


def _write_schemas(con, ps, bs):
    con.execute("DELETE FROM point_schema")
    con.execute("DELETE FROM blend_schema")
    for (t, ver), spec in sorted(ps.items()):
        con.execute("INSERT INTO point_schema VALUES(?,?,?)",
                    (t, ver, json.dumps(spec)))
    for i, (name, (kind, n)) in enumerate(bs.items()):
        con.execute("INSERT INTO blend_schema VALUES(?,?,?,?)", (i, name, kind, n))


def _spec(schema_entry):
    return [(k, v[0], v[1]) for k, v in schema_entry.items()]


# ---------------------------------------------------------------------------
# build / add
# ---------------------------------------------------------------------------

def _intern(con, sql, select_sql, row, cache, cache_key):
    if cache_key in cache:
        return cache[cache_key]
    hit = con.execute(select_sql, row).fetchone()
    rid = hit[0] if hit else con.execute(sql, row).lastrowid
    cache[cache_key] = rid
    return rid


def _ingest(con, sources, ps, bs, verbose=True):
    """Insert every edit of every source, deduplicating against what is there."""
    for stmt in DEDUP_INDEXES:
        con.execute(stmt)

    cache = {}
    added_edits = 0
    for path in sources:
        d = _load_harvest(path)
        name = os.path.basename(path)
        if con.execute("SELECT 1 FROM library WHERE name=?", (name,)).fetchone():
            raise SystemExit(f"{name}: already in the corpus. Remove it first, or "
                             f"rename the harvest if it is genuinely a new library.")
        lib = con.execute(
            "INSERT INTO library(name,dt_version,format_version,blend_version,"
            "masks_version,added) VALUES(?,?,?,?,?,date('now'))",
            (name, d.get('darktable_version'), d.get('format_version'),
             d.get('current_blend_version'), d.get('current_masks_version'))).lastrowid

        instances = []
        for edit in d.get('edits', []):
            _check_known(edit, name)
            body = _edit_body(edit)

            im = body.get('image') or {}
            irow = (im.get('width'), im.get('height'), im.get('dimensions_known'))
            iid = _intern(con,
                          "INSERT INTO image(width,height,dimensions_known) VALUES(?,?,?)",
                          "SELECT id FROM image WHERE width IS ? AND height IS ?"
                          " AND dimensions_known IS ?", irow, cache, ('i',) + irow)

            b = body.get('blend') or {}
            brow = (b.get(BLEND_TEXT, ''), _pack(b, _spec(bs)))
            bid = _intern(con, "INSERT INTO blend(raster_source,v) VALUES(?,?)",
                          "SELECT id FROM blend WHERE raster_source IS ? AND v IS ?",
                          brow, cache, ('b',) + brow)

            fids = []
            for f in body.get('forms') or []:
                pts = f.get('points') or []
                frow = (f['formid'], f['type'], f['version'],
                        *(f.get('source') or [0, 0, 0]),
                        f.get('points_count', len(pts)),
                        None if 'points_error' in f
                        else _pack_points(f, ps),
                        f.get('points_error'), f.get('points_blob_bytes'))
                fids.append(_intern(
                    con,
                    "INSERT INTO form(formid,type,version,sx,sy,sz,npoints,points,"
                    "points_error,points_blob_bytes) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    "SELECT id FROM form WHERE formid IS ? AND type IS ? AND"
                    " version IS ? AND sx IS ? AND sy IS ? AND sz IS ? AND"
                    " npoints IS ? AND points IS ? AND points_error IS ?"
                    " AND points_blob_bytes IS ?", frow, cache, ('f',) + frow))

            members = struct.pack('<%dI' % len(fids), *fids)
            sid = _intern(con, "INSERT INTO form_set(members) VALUES(?)",
                          "SELECT id FROM form_set WHERE members IS ?",
                          (members,), cache, ('s', members))

            erow = (body.get('operation'), body.get('blendop_version'),
                    body.get('multi_priority'), body.get('enabled'), iid, bid, sid)
            before = con.execute("SELECT COUNT(*) FROM edit").fetchone()[0]
            eid = _intern(con,
                          "INSERT INTO edit(operation,blendop_version,multi_priority,"
                          "enabled,image_id,blend_id,form_set_id) VALUES(?,?,?,?,?,?,?)",
                          "SELECT id FROM edit WHERE operation IS ? AND"
                          " blendop_version IS ? AND multi_priority IS ? AND"
                          " enabled IS ? AND image_id IS ? AND blend_id IS ?"
                          " AND form_set_id IS ?", erow, cache, ('e',) + erow)
            if con.execute("SELECT COUNT(*) FROM edit").fetchone()[0] != before:
                added_edits += 1
            instances.append((lib, edit.get('index', len(instances)), eid,
                              edit.get('image_index')))

        con.executemany("INSERT INTO edit_instance VALUES(?,?,?,?)", instances)
        if verbose:
            print(f"  {name}: {len(instances)} edits, "
                  f"{len(set(i[2] for i in instances))} distinct")

    for stmt in DEDUP_INDEXES:
        con.execute("DROP INDEX " + stmt.split()[3])
    return added_edits


def _pack_points(f, ps):
    spec = _spec(ps[(f['type'], f['version'])])
    out = bytearray()
    for pt in f.get('points') or []:
        out += _pack(_flat(pt), spec)
    return bytes(out)


def cmd_build(args):
    if os.path.exists(args.db):
        os.remove(args.db)
    con = sqlite3.connect(args.db)
    con.executescript(SCHEMA)
    con.execute("INSERT INTO meta VALUES('format_version',?)", (str(FORMAT_VERSION),))
    con.execute("INSERT INTO meta VALUES('generator',?)",
                (os.path.basename(__file__),))

    bodies = []
    for p in args.sources:
        for e in _load_harvest(p).get('edits', []):
            _check_known(e, os.path.basename(p))
            bodies.append(_edit_body(e))
    ps, bs = _infer_schemas(bodies)
    _write_schemas(con, ps, bs)
    print(f"packing {len(args.sources)} libraries")
    _ingest(con, args.sources, ps, bs)
    con.commit()
    con.execute("VACUUM")
    con.close()
    _report(args.db)


def cmd_add(args):
    con = sqlite3.connect(args.db)
    ps, bs = _read_schemas(con)

    bodies = []
    for p in args.sources:
        for e in _load_harvest(p).get('edits', []):
            _check_known(e, os.path.basename(p))
            bodies.append(_edit_body(e))

    # A new library can put a fractional value in a field every earlier one had
    # only integers in. Widening the schema changes the layout, so everything
    # already stored has to be repacked -- which is why this is checked rather
    # than assumed.
    ps2, bs2 = _infer_schemas(bodies, ps, bs)
    if ps2 != ps or bs2 != bs:
        raise SystemExit(
            "this library needs a wider packing schema than the corpus has "
            "(a field that was integer-valued everywhere else is fractional "
            "here, or a new (type, version) appeared).\n"
            "  Rebuild instead:  masks_corpus.py build <db> <all harvests>")

    print(f"adding {len(args.sources)} librar{'y' if len(args.sources)==1 else 'ies'}")
    _ingest(con, args.sources, ps, bs)
    con.commit()
    con.execute("VACUUM")
    con.close()
    _report(args.db)


# ---------------------------------------------------------------------------
# read back
# ---------------------------------------------------------------------------

def _read_all(con, with_derived=False):
    ps, bs = _read_schemas(con)
    bspec = _spec(bs)

    forms = {}
    for (fid, formid, t, ver, sx, sy, sz, npt, blob, perr,
         pbb) in con.execute(
            "SELECT id,formid,type,version,sx,sy,sz,npoints,points,points_error,"
            "points_blob_bytes FROM form"):
        f = {'formid': formid, 'type': t, 'version': ver, 'source': [sx, sy, sz]}
        f['points_count'] = npt
        if with_derived:
            f['type_name'] = _type_name(t)
        if perr is not None:
            f['points_error'] = perr
            f['points_blob_bytes'] = pbb
        else:
            f['points'] = [_unflat(r) for r in
                           _unpack(blob or b'', _spec(ps.get((t, ver), {})), npt)]
        forms[fid] = f

    blends = {}
    for bid, rs, v in con.execute("SELECT id,raster_source,v FROM blend"):
        b = _unpack(v, bspec, 1)[0]
        b[BLEND_TEXT] = rs
        if with_derived:
            b['mask_mode_names'] = [nm for bit, nm in _MASK_MODE_NAMES
                                    if b.get('mask_mode', 0) & bit]
            b['blend_cst_name'] = _CST_NAMES.get(b.get('blend_cst'), 'unknown')
        blends[bid] = b

    images = {}
    for i, w, hh, dk in con.execute(
            "SELECT id,width,height,dimensions_known FROM image"):
        im = {'width': w, 'height': hh}
        if dk is not None:
            im['dimensions_known'] = dk
        images[i] = im

    fsets = {i: list(struct.unpack('<%dI' % (len(m) // 4), m))
             for i, m in con.execute("SELECT id,members FROM form_set")}

    edits = {}
    for eid, op, bv, mp, en, iid, bid, sid in con.execute(
            "SELECT id,operation,blendop_version,multi_priority,enabled,"
            "image_id,blend_id,form_set_id FROM edit"):
        edits[eid] = {'image': images[iid], 'operation': op,
                      'multi_priority': mp, 'enabled': en,
                      'blendop_version': bv, 'blend': blends[bid],
                      'forms': [forms[f] for f in fsets[sid]]}
    return edits


def _f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def _norm(o):
    """Canonical form for comparison.

    Floats are projected to float32, because that is what the value IS: the
    harvest prints a C float with %.9g, and parsing that decimal into a double
    gives a number that differs from the float32 in the low bits while denoting
    the same float32. Comparing the doubles reports a loss that is not there --
    the first version of `verify` did exactly that and declared 56912 of 57144
    edits unreconstructable.

    Integers above 2^24 stay integers: a mask_id or a formid would be destroyed
    by a float32 round trip. Smaller ones are projected so that a field written
    `1` in one edit and `1.0` in another compares equal.
    """
    if isinstance(o, dict):
        return {k: _norm(v) for k, v in sorted(o.items())}
    if isinstance(o, list):
        return [_norm(v) for v in o]
    if isinstance(o, bool):
        return int(o)
    if isinstance(o, float):
        return _f32(o)
    if isinstance(o, int):
        return o if abs(o) > (1 << 24) else _f32(o)
    return o


def _rebuild_library(con, rebuilt, lid):
    """the library's edits, in order, decorated exactly as a harvest is"""
    out = []
    for seq, eid, iidx in con.execute(
            "SELECT seq,edit_id,image_index FROM edit_instance"
            " WHERE library_id=? ORDER BY seq", (lid,)):
        e = json.loads(json.dumps(rebuilt[eid]))   # a copy per occurrence
        e['index'] = seq
        if iidx is not None:
            e['image_index'] = iidx
        out.append(e)
    return out


def cmd_verify(args):
    """Prove the corpus reproduces its sources, edit for edit, field for field.

    Not "the stored columns match": the whole file the harvester wrote, derived
    name strings and positional indices included, rebuilt from the corpus and
    compared against the original in order. That is the claim worth making --
    a check that only compared what it chose to store would pass on a corpus
    that had dropped something, which is exactly how `points_count` and
    `dimensions_known` got lost and found again while this was written.
    """
    con = sqlite3.connect(args.db)
    rebuilt = _read_all(con, with_derived=True)
    lids = {n: i for i, n in con.execute("SELECT id,name FROM library")}

    total = missing = differing = 0
    problems = []
    for path in args.sources:
        name = os.path.basename(path)
        if name not in lids:
            problems.append(f"{name}: not in the corpus")
            continue
        src = _load_harvest(path).get('edits', [])
        got = _rebuild_library(con, rebuilt, lids[name])
        if len(src) != len(got):
            problems.append(f"{name}: {len(src)} edits in the harvest, "
                            f"{len(got)} rebuilt")
            continue
        for i, (a, b) in enumerate(zip(src, got)):
            total += 1
            if _norm(a) != _norm(b):
                differing += 1
                if len(problems) < 5:
                    fields = [k for k in set(a) | set(b)
                              if _norm(a.get(k)) != _norm(b.get(k))]
                    problems.append(f"{name}[{i}]: differs in {sorted(fields)}")

    ok = not differing and not problems
    print(f"libraries             : {len(args.sources)}")
    print(f"edits compared        : {total}")
    print(f"rebuilt identically   : {total - differing}")
    print(f"DIFFERING             : {differing}")
    for line in problems[:5]:
        print(f"  {line}")
    print(f"verdict               : {'the corpus reproduces its sources' if ok else 'MISMATCH'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

# The name strings a harvest carries next to each numeric code. Not stored --
# they are derived -- so they are rebuilt here, and they have to be rebuilt
# EXACTLY: these mirror _form_type_name(), _emit_mask_mode() and _cst_name() in
# src/develop/masks/harvest.c, in the same order and with the same spelling.
#
# Including the spellings that are not what one would pick: the colour space is
# "Lab", not "lab", and a type the harvester has no name for is "unknown" --
# object, parametric and raster forms fall through to it, so listing them here
# would produce a file the harvester never writes. Both were caught by
# comparing an export against its source, which is why that comparison is worth
# running after touching this.
_TYPE_NAMES = [(1 << 0, 'circle'), (1 << 1, 'path'), (1 << 2, 'group'),
               (1 << 4, 'gradient'), (1 << 5, 'ellipse'), (1 << 6, 'brush')]
_MASK_MODE_NAMES = [(1 << 0, 'enabled'), (1 << 1, 'drawn'), (1 << 2, 'parametric'),
                    (1 << 3, 'raster'), (1 << 4, 'flexi')]   # blend.h: FLEXI is 1<<4
_CST_NAMES = {0: 'none', 1: 'raw', 2: 'Lab', 3: 'rgb_display', 4: 'rgb_scene'}


def _type_name(t):
    for bit, name in _TYPE_NAMES:
        if t & bit:
            return name
    return 'unknown'


def cmd_export(args):
    con = sqlite3.connect(args.db)
    rebuilt = _read_all(con, with_derived=True)
    libs = {i: (n, dv, fv, bv, mv) for i, n, dv, fv, bv, mv in con.execute(
        "SELECT id,name,dt_version,format_version,blend_version,masks_version"
        " FROM library")}
    per_lib = collections.defaultdict(list)
    for lid, seq, eid, iidx in con.execute(
            "SELECT library_id,seq,edit_id,image_index FROM edit_instance"
            " ORDER BY library_id,seq"):
        per_lib[lid].append((seq, eid, iidx))

    os.makedirs(args.outdir, exist_ok=True)
    wanted = set(args.libraries or [])
    written = []
    for lid, rows in sorted(per_lib.items()):
        name, dv, fv, bv, mv = libs[lid]
        if wanted and name not in wanted and os.path.splitext(name)[0] not in wanted:
            continue
        edits = []
        for seq, eid, iidx in rows:
            e = json.loads(json.dumps(rebuilt[eid]))   # a copy per occurrence
            e['index'] = seq
            if iidx is not None:
                e['image_index'] = iidx
            edits.append(e)
        doc = {'format': 'darktable mask harvest',
               'format_version': fv, 'darktable_version': dv,
               'current_blend_version': bv, 'current_masks_version': mv,
               'exported_from': os.path.basename(args.db),
               'edits': edits}
        out = os.path.join(args.outdir, name if name.endswith('.gz') else name + '.gz')
        with gzip.open(out, 'wt') as fh:
            json.dump(doc, fh, separators=(',', ':'))
        written.append((out, len(edits)))
        print(f"  {out}  {len(edits)} edits")
    if not written:
        print("nothing exported"
              + (f" -- no library matched {args.libraries}" if args.libraries else ""))
        return 1
    return 0


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

# One --check-masks process per chunk, never threads inside one. The checks
# pin OpenMP to one thread for reproducibility (verify.c), hold an OpenCL
# device for the whole run, and drive the real history reader and writer
# against scratch images at fixed ids in the process's own library, so a
# second thread would share all of that. Processes share nothing but the
# scratch root flexi test mode puts under g_get_tmp_dir(), and a private
# TMPDIR per worker separates that too.
#
# Every occurrence of an edit goes to the same chunk, in library order. The
# checks replay the first occurrence and reuse its verdict for the rest, so
# keeping them together makes each "repeat" flag and each distinct count come
# out exactly as a whole-library run would print them.

SECTIONS = ('roundtrip', 'verify', 'styleapply', 'persist', 'undo')

# How each summary member merges across chunks. Every member the C side writes
# has to be listed: one it has never seen stops the merge rather than being
# summed by default, which would be wrong for the first maximum added.
_SUM = 'sum'
_MAX = 'max'
_ALL = 'all'
_EQUAL = 'equal'
MERGE = {
    'roundtrip': {
        'passed': _ALL,
        **{k: _SUM for k in ('harvested', 'round_tripped', 'distinct_edits',
                             'unchanged', 'different', 'errors', 'skipped',
                             'multi_instance', 'loaded_with_no_module')},
    },
    'verify': {
        'passed': _ALL,
        **{k: _SUM for k in ('harvested', 'replayed', 'distinct_edits',
                             'identical', 'equivalent', 'different', 'skipped',
                             'errors', 'live', 'live_identical',
                             'live_equivalent', 'live_different', 'inert',
                             'gpu_compared', 'image_compared',
                             'gpu_image_compared', 'dev_gap_widened',
                             'dev_gap_widened_own')},
        'worst_dev_gap_classic': _MAX,
        'worst_dev_gap_migrated': _MAX,
    },
    'styleapply': {
        'passed': _ALL,
        'ran': _EQUAL,
        'reason': _EQUAL,
        **{k: _SUM for k in ('harvested', 'applied_as_style', 'distinct_edits',
                             'ok', 'style_mask_lost', 'host_disturbed',
                             'no_module_at_all', 'drawn_only_not_carried',
                             'errors', 'skipped', 'same_op_as_host',
                             'drawn_mask_in_style', 'multi_instance_styles',
                             'second_instance_allocated')},
    },
    'persist': {
        'passed': _ALL,
        **{k: _SUM for k in ('harvested', 'swept', 'distinct_edits',
                             'distinct_swept', 'identical', 'different',
                             'errors', 'skipped', 'sequences_compared',
                             'sequences_disagreed', 'sequences_live',
                             'swept_nothing', 'groups_swept',
                             'edits_with_nested_group')},
    },
    'undo': {
        'passed': _ALL,
        **{k: _SUM for k in ('harvested', 'swept', 'distinct_edits',
                             'distinct_swept', 'identical', 'different',
                             'errors', 'skipped', 'cycles_compared',
                             'cycles_disagreed', 'cycles_live', 'undo_failed',
                             'redo_failed', 'swept_nothing', 'groups_swept',
                             'edits_with_nested_group')},
        'worst_diff': _MAX,
    },
}

# A worst value together with the edit it came from and that edit's other
# figures. verify.c keeps the first edit strictly above the running worst,
# starting from 0 at index -1, so the merge keeps the largest value and, on a
# tie, the lowest library position.
WORST = {
    'verify': [
        ('worst_cpu_diff', 'worst_cpu_diff_index',
         ('worst_cpu_mean_diff', 'worst_cpu_differing_pixels')),
        ('worst_gpu_diff', 'worst_gpu_diff_index',
         ('worst_gpu_mean_diff', 'worst_gpu_differing_pixels')),
        ('worst_image_diff', 'worst_image_diff_index',
         ('worst_image_mean_diff', 'worst_image_differing_pixels')),
        ('worst_gpu_image_diff', 'worst_gpu_image_diff_index',
         ('worst_gpu_image_mean_diff', 'worst_gpu_image_differing_pixels')),
    ],
}

# per-sequence and per-action tallies: summed entry by entry, keyed on a name
TALLY = {
    'persist': ('per_sequence', 'name', ('compared', 'disagreed', 'live')),
    'undo': ('per_action', 'action',
             ('compared', 'disagreed', 'undo_failed', 'redo_failed', 'live')),
}

# mirrors blend.h: what _pick_host() in styleapply.c asks of a host
_MASK_DRAWN, _MASK_CONDITIONAL, _MASK_RASTER, _MASK_FLEXI = 2, 4, 8, 16

# How many host candidates a chunk carries. _pick_host() takes the first
# candidate whose forms it can rebuild, and a form the harvester could not
# decode is the only way that fails, so a handful past the first clean one is
# already more than it can need.
HOST_CANDIDATES_PAST_CLEAN = 8


def _host_candidates(rebuilt, rows):
    """The library's possible style-apply hosts, in order, each edit once.

    Only the conditions that can be read off the record are applied here; the
    rest (whether the forms rebuild) is left to _pick_host() itself, which runs
    over this list instead of the chunk's own edits. Deduplicated by edit,
    since an edit that fails as a host fails again at every repeat."""
    out, seen, clean = [], set(), 0
    for _, eid, _ in rows:
        if eid in seen:
            continue
        seen.add(eid)
        e = rebuilt[eid]
        mode = int((e.get('blend') or {}).get('mask_mode', 0))
        forms = e.get('forms') or []
        if (mode & _MASK_FLEXI or not mode & _MASK_DRAWN
                or mode & (_MASK_CONDITIONAL | _MASK_RASTER)
                or not forms or not e.get('operation')):
            continue
        out.append(e)
        if not any('points_error' in f for f in forms):
            clean += 1
            if clean >= HOST_CANDIDATES_PAST_CLEAN:
                break
    return out


def _chunks(rows, size):
    """Library positions cut into chunks of about `size` distinct edits.

    All positions of one edit land in the same chunk, and each chunk lists its
    positions in library order."""
    by_edit = collections.OrderedDict()
    for pos, (_, eid, _) in enumerate(rows):
        by_edit.setdefault(eid, []).append(pos)
    out, cur, n = [], [], 0
    for positions in by_edit.values():
        cur.extend(positions)
        n += 1
        if n >= size:
            out.append(sorted(cur))
            cur, n = [], 0
    if cur:
        out.append(sorted(cur))
    return out


def _harvest_doc(con_meta, edits, **extra):
    name, dv, fv, bv, mv = con_meta
    doc = {'format': 'darktable mask harvest',
           'format_version': fv, 'darktable_version': dv,
           'current_blend_version': bv, 'current_masks_version': mv}
    doc.update(extra)
    doc['edits'] = edits
    return doc


def _occurrence(rebuilt, seq, eid, iidx):
    e = json.loads(json.dumps(rebuilt[eid]))   # a copy per occurrence
    e['index'] = seq
    if iidx is not None:
        e['image_index'] = iidx
    return e


def _merge_summary(section, parts, remap):
    """One section's summary from its chunks' summaries. `remap[k]` turns
    chunk k's positions into library positions."""
    rules = MERGE[section]
    worst = WORST.get(section, [])
    worst_keys = {k for v, i, c in worst for k in (v, i, *c)}
    tally = TALLY.get(section)
    out = {}
    for k, part in enumerate(parts):
        for key in part:
            if key not in rules and key not in worst_keys \
                    and not (tally and key == tally[0]):
                raise SystemExit(
                    f"{section}.summary.{key}: no merge rule. Add it to MERGE,\n"
                    f"  WORST or TALLY in {os.path.relpath(__file__)} -- summing\n"
                    f"  it by default would be wrong the first time it is not a count.")
    for key, rule in rules.items():
        vals = [p[key] for p in parts if key in p]
        if not vals:
            continue
        if rule == _SUM:
            out[key] = sum(vals)
        elif rule == _MAX:
            out[key] = max(vals)
        elif rule == _ALL:
            out[key] = all(vals)
        elif any(v != vals[0] for v in vals):
            raise SystemExit(f"{section}.summary.{key}: chunks disagree {sorted(set(map(str, vals)))}")
        else:
            out[key] = vals[0]
    for vkey, ikey, companions in worst:
        best = None
        for k, p in enumerate(parts):
            if vkey not in p or p[ikey] < 0:
                continue
            cand = (p[vkey], -remap[k][p[ikey]], k)
            if best is None or cand > best:
                best = cand
        if best is None:
            src = next((p for p in parts if vkey in p), None)
            if src is not None:
                out.update({vkey: src[vkey], ikey: -1,
                            **{c: src[c] for c in companions}})
        else:
            p = parts[best[2]]
            out.update({vkey: p[vkey], ikey: -best[1],
                        **{c: p[c] for c in companions}})
    if tally:
        # undo lists only the actions a chunk compared, so the order is that of
        # first sighting rather than undo.c's own; nothing reads it by position
        member, name, fields = tally
        merged = collections.OrderedDict()
        for p in parts:
            for entry in p.get(member, []):
                m = merged.get(entry[name])
                if m is None:
                    merged[entry[name]] = dict(entry)
                else:
                    for f in fields:
                        m[f] += entry[f]
        out[member] = list(merged.values())
    return out


def _merge_reports(source, reports, remaps, expected_positions):
    """The chunks' --check-masks reports as the report a whole-library run
    would have written: rows renumbered to library positions and in library
    order, summaries merged by MERGE/WORST/TALLY."""
    doc = {'source': source}
    versions = {r.get('darktable_version') for r in reports}
    if len(versions) != 1:
        raise SystemExit(f"{source}: chunks were checked by different builds {sorted(map(str, versions))}")
    doc['darktable_version'] = versions.pop()
    for section in SECTIONS:
        secs = [r.get(section) or {} for r in reports]
        out = {'source': source}
        hosts = {s['host'] for s in secs if 'host' in s}
        if len(hosts) > 1:
            raise SystemExit(f"{source}: chunks picked different style-apply hosts {sorted(hosts)}")
        if hosts:
            out['host'] = hosts.pop()
        rows = []
        for k, s in enumerate(secs):
            for row in s.get('edits', []):
                row = dict(row)
                row['index'] = remaps[k][row['index']]
                rows.append(row)
        rows.sort(key=lambda r: r['index'])
        got = [r['index'] for r in rows]
        if got and got != expected_positions:
            raise SystemExit(f"{source}: {section} does not report every edit exactly once "
                             f"({len(got)} rows for {len(expected_positions)} edits)")
        out['edits'] = rows
        out['summary'] = _merge_summary(section, [s.get('summary') or {} for s in secs], remaps)
        doc[section] = out
    ran = {bool(doc['styleapply']['summary'].get('ran', True))}
    doc['summary'] = {
        'passed': all(doc[s]['summary'].get('passed', False) for s in SECTIONS),
        'roundtrip_passed': doc['roundtrip']['summary'].get('passed', False),
        'verify_passed': doc['verify']['summary'].get('passed', False),
        'styleapply_ran': ran.pop(),
        'styleapply_passed': doc['styleapply']['summary'].get('passed', False),
        'persist_passed': doc['persist']['summary'].get('passed', False),
        'undo_passed': doc['undo']['summary'].get('passed', False),
    }
    return doc


def cmd_check(args):
    import concurrent.futures
    import queue
    import shutil
    import subprocess
    import tempfile
    import time

    # a run takes a while and is usually redirected to a log someone tails
    sys.stdout.reconfigure(line_buffering=True)
    dt = os.path.abspath(args.darktable)
    if not os.access(dt, os.X_OK):
        raise SystemExit(f"{args.darktable}: not an executable darktable binary")
    outdir = os.path.abspath(args.outdir)
    work = os.path.join(outdir, 'chunks')
    os.makedirs(work, exist_ok=True)

    con = sqlite3.connect(args.db)
    rebuilt = _read_all(con, with_derived=True)
    libs = {i: (n, dv, fv, bv, mv) for i, n, dv, fv, bv, mv in con.execute(
        "SELECT id,name,dt_version,format_version,blend_version,masks_version"
        " FROM library")}
    per_lib = collections.defaultdict(list)
    for lid, seq, eid, iidx in con.execute(
            "SELECT library_id,seq,edit_id,image_index FROM edit_instance"
            " ORDER BY library_id,seq"):
        per_lib[lid].append((seq, eid, iidx))
    con.close()

    wanted = set(args.libraries or [])
    plan = []       # (library name, harvest path, positions, [chunk jobs])
    jobs = []       # (chunk path, number of distinct edits)
    for lid, rows in sorted(per_lib.items()):
        meta = libs[lid]
        name = meta[0]
        stem = name[:-3] if name.endswith('.gz') else name
        # the contributor's name alone will do: "akgt94" for masks_harvest.akgt94.json.gz
        short = os.path.splitext(stem)[0]
        for prefix in ('masks_harvest.', 'masks_harvest_'):
            if short.startswith(prefix):
                short = short[len(prefix):]
        if wanted and not wanted & {name, stem, os.path.splitext(stem)[0], short}:
            continue
        occ = [_occurrence(rebuilt, *r) for r in rows]

        # the whole library too: the ledger reads its edits next to the report
        harvest = os.path.join(outdir, stem + '.gz')
        with gzip.open(harvest, 'wt') as fh:
            json.dump(_harvest_doc(meta, occ, exported_from=os.path.basename(args.db)),
                      fh, separators=(',', ':'))

        cands = _host_candidates(rebuilt, rows)
        chunk_jobs = []
        for c, positions in enumerate(_chunks(rows, args.chunk)):
            path = os.path.join(work, f"{os.path.splitext(stem)[0]}.c{c:04d}.json.gz")
            with gzip.open(path, 'wt') as fh:
                json.dump(_harvest_doc(meta, [occ[p] for p in positions],
                                       exported_from=os.path.basename(args.db),
                                       styleapply_host_candidates=cands),
                          fh, separators=(',', ':'))
            distinct = len({rows[p][1] for p in positions})
            chunk_jobs.append((path, positions))
            jobs.append((path, distinct))
        plan.append((name, harvest, list(range(len(rows))), chunk_jobs))
        print(f"  {stem}: {len(rows)} edits, {len({r[1] for r in rows})} distinct,"
              f" {len(chunk_jobs)} chunks")
    if not plan:
        print("nothing to check"
              + (f" -- no library matched {args.libraries}" if args.libraries else ""))
        return 1

    # Each worker slot gets its own scratch root, reused across its chunks. The
    # darktablerc of the shared scratch root is copied in when there is one, so
    # a pooled run starts from the configuration a plain --check-masks run on
    # this machine would.
    rc = args.rc or os.path.join(tempfile.gettempdir(), 'flexi_mask_test',
                                 'config', 'darktablerc')
    slots = queue.Queue()
    for k in range(args.jobs):
        tmp = os.path.join(outdir, 'tmp', f'slot{k:02d}')
        cfg = os.path.join(tmp, 'flexi_mask_test', 'config')
        os.makedirs(cfg, exist_ok=True)
        if os.path.exists(rc) and not os.path.exists(os.path.join(cfg, 'darktablerc')):
            shutil.copy(rc, cfg)
        slots.put(tmp)
    print(f"{len(jobs)} chunks over {args.jobs} processes"
          + (f", darktablerc from {rc}" if os.path.exists(rc) else ", fresh configuration"))

    def run(path, extra=()):
        report = path[:-3] + '.check.json'
        if args.resume and not extra and os.path.exists(report):
            return path, 'reused'
        tmp = slots.get()
        try:
            env = dict(os.environ, TMPDIR=tmp + os.sep)
            with open(path[:-8] + '.log', 'w') as log:
                rc_ = subprocess.call([dt, *extra, '--library', ':memory:',
                                       '--check-masks', path],
                                      cwd=outdir, env=env, stdout=log,
                                      stderr=subprocess.STDOUT)
        finally:
            slots.put(tmp)
        # 1 is a check that ran and failed; anything else left no usable report
        if rc_ not in (0, 1) or not os.path.exists(report):
            return path, f'crashed ({rc_})'
        return path, 'ok' if rc_ == 0 else 'failed'

    def pool(paths, extra=()):
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as ex:
            for done, fut in enumerate(concurrent.futures.as_completed(
                    [ex.submit(run, p, extra) for p in paths]), 1):
                path, st = fut.result()
                status[path] = st
                if st.startswith('crashed') or done % max(1, len(paths) // 20) == 0 \
                        or done == len(paths):
                    print(f"  {done}/{len(paths)} chunks, {time.time() - t0:.0f}s"
                          + (f"  -- {os.path.basename(path)} {st}"
                             if st.startswith('crashed') else ''))

    def gpu(path):
        """True/False for whether verify got an OpenCL device, None if unknown"""
        try:
            with open(path[:-8] + '.log', errors='replace') as log:
                text = log.read()
        except OSError:
            return None
        if '[verify] no OpenCL device' in text:
            return False
        return True if '[verify] OpenCL device' in text else None

    t0 = time.time()
    status = {}
    # largest first, so the pool does not end waiting on one long chunk
    pool([p for p, _ in sorted(jobs, key=lambda j: -j[1])])

    # A process that came up without an OpenCL device replays verify on the CPU
    # only and still passes, so the GPU half of the check would be missing
    # without a word. Seen once, on the first chunk of each worker, with twelve
    # processes on fourteen cores; not reproducible on its own. Where the
    # machine has a device, such a chunk is run again, with -d opencl so the log
    # says why if it happens twice.
    cpu_only = [p for p in status if not status[p].startswith('crashed') and gpu(p) is False]
    if cpu_only and any(gpu(p) for p in status):
        print(f"  {len(cpu_only)} chunk(s) ran verify without an OpenCL device; running again")
        pool(cpu_only, ('-d', 'opencl'))
        for p in cpu_only:
            if gpu(p) is False:
                status[p] = 'crashed (no OpenCL device twice)'
                print(f"  -- {os.path.basename(p)}: still no OpenCL device, see its .log")

    crashed = [p for p, st in status.items() if st.startswith('crashed')]
    all_passed = True
    print()
    for name, harvest, positions, chunk_jobs in plan:
        stem = os.path.basename(harvest)[:-3]
        if any(p in crashed for p, _ in chunk_jobs):
            print(f"  {stem:44s} INCOMPLETE (a chunk crashed; see its .log)")
            all_passed = False
            continue
        reports = [_load_harvest(p[:-3] + '.check.json') for p, _ in chunk_jobs]
        doc = _merge_reports(harvest, reports, [pos for _, pos in chunk_jobs], positions)
        with open(os.path.join(outdir, stem + '.check.json'), 'w') as fh:
            json.dump(doc, fh, indent=2)
        s = doc['summary']
        failed = [k[:-7] for k in ('roundtrip_passed', 'verify_passed',
                                   'styleapply_passed', 'persist_passed',
                                   'undo_passed') if not s[k]]
        all_passed &= s['passed']
        print(f"  {stem:44s} {'PASSED' if s['passed'] else 'FAILED: ' + ', '.join(failed)}")
    print(f"\nreports and harvests in {outdir}  ({time.time() - t0:.0f}s)")
    if crashed:
        return 2
    return 0 if all_passed else 1


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def _report(db):
    con = sqlite3.connect(db)
    n = lambda t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    occ = n('edit_instance')
    print(f"\n{db}  {os.path.getsize(db)/1048576:.2f} MB")
    print(f"  libraries {n('library')}   edits {occ} ({n('edit')} distinct)")
    print(f"  forms {n('form')}   blends {n('blend')}   "
          f"form sets {n('form_set')}   images {n('image')}")
    con.close()


def cmd_stats(args):
    _report(args.db)
    con = sqlite3.connect(args.db)
    print("\n  per library:")
    for name, e, d in con.execute(
            "SELECT l.name, COUNT(*), COUNT(DISTINCT i.edit_id) FROM library l"
            " JOIN edit_instance i ON i.library_id=l.id GROUP BY l.id ORDER BY 2 DESC"):
        print(f"    {name:44s} {e:7d} edits  {d:6d} distinct")
    return 0


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('build', help='pack harvests into a new corpus')
    p.add_argument('db'); p.add_argument('sources', nargs='+')
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser('add', help='add a library to an existing corpus')
    p.add_argument('db'); p.add_argument('sources', nargs='+')
    p.set_defaults(fn=cmd_add)

    p = sub.add_parser('verify', help='prove the corpus reconstructs the harvests')
    p.add_argument('db'); p.add_argument('sources', nargs='+')
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser('export', help='write harvest files back out')
    p.add_argument('db'); p.add_argument('outdir')
    p.add_argument('libraries', nargs='*')
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser('check', help='run --check-masks over the corpus in parallel')
    p.add_argument('db'); p.add_argument('outdir')
    p.add_argument('libraries', nargs='*')
    p.add_argument('--darktable', required=True, help='the darktable binary to check with')
    p.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 2),
                   help='processes at once (default: cores - 2)')
    p.add_argument('--chunk', type=int, default=100,
                   help='distinct edits per process (default: 100)')
    p.add_argument('--rc', help='darktablerc to start each process from (default:'
                   ' the one in the flexi test mode scratch root, if any)')
    p.add_argument('--resume', action='store_true',
                   help='keep chunk reports already in outdir from an earlier run')
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser('stats', help='what is in the corpus')
    p.add_argument('db')
    p.set_defaults(fn=cmd_stats)

    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == '__main__':
    main()
