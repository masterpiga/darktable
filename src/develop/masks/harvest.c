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

#include "develop/masks/harvest.h"

#include "common/darktable.h"
#include "develop/blend.h"
#include "develop/masks.h"

#include <gio/gio.h>
#include <inttypes.h>
#include <sqlite3.h>
#include <stdio.h>
#include <string.h>

#define HARVEST_FORMAT_VERSION 1

// ---------------------------------------------------------------------------
// JSON emission
//
// Hand-rolled rather than pulled from a library, because the output has a
// property no generic serialiser can be asked to guarantee: every value in it
// must be one we deliberately decided to include. Writing the fields out by
// hand means a field can only appear here if someone typed its name, which is
// the point -- a reflective serialiser over the structs would happily emit the
// user-typed group name this format exists to leave out.
// ---------------------------------------------------------------------------

typedef struct
{
  FILE *f;
  int indent;
  gboolean need_comma;
} json_t;

static void _j_indent(json_t *j)
{
  if(j->need_comma) fputs(",", j->f);
  fputs("\n", j->f);
  for(int i = 0; i < j->indent; i++) fputs("  ", j->f);
  j->need_comma = TRUE;
}

static void _j_open(json_t *j, const char *key, const char brace)
{
  _j_indent(j);
  if(key) fprintf(j->f, "\"%s\": %c", key, brace);
  else fprintf(j->f, "%c", brace);
  j->indent++;
  j->need_comma = FALSE;
}

static void _j_close(json_t *j, const char brace)
{
  j->indent--;
  fputs("\n", j->f);
  for(int i = 0; i < j->indent; i++) fputs("  ", j->f);
  fprintf(j->f, "%c", brace);
  j->need_comma = TRUE;
}

static void _j_int(json_t *j, const char *key, const int64_t v)
{
  _j_indent(j);
  fprintf(j->f, "\"%s\": %" PRId64, key, v);
}

/** %.9g round-trips a float exactly, so the verifier replays the same value
    the user's edit held rather than a rounded one. */
static void _j_float(json_t *j, const char *key, const float v)
{
  _j_indent(j);
  if(isfinite(v)) fprintf(j->f, "\"%s\": %.9g", key, (double)v);
  else fprintf(j->f, "\"%s\": null", key); // JSON has no NaN/Inf
}

/** Emit a string, escaped. Only ever used for values this file chose: module
    operation names, type names, and fixed format strings. Never for anything
    read out of a user-editable column. */
static void _j_str(json_t *j, const char *key, const char *v)
{
  _j_indent(j);
  fprintf(j->f, "\"%s\": \"", key);
  for(const char *p = v; p && *p; p++)
  {
    if(*p == '"' || *p == '\\') fprintf(j->f, "\\%c", *p);
    else if((unsigned char)*p < 0x20) fprintf(j->f, "\\u%04x", *p);
    else fputc(*p, j->f);
  }
  fputs("\"", j->f);
}

static void _j_float_array(json_t *j, const char *key, const float *v, const int n)
{
  _j_indent(j);
  fprintf(j->f, "\"%s\": [", key);
  for(int i = 0; i < n; i++)
    fprintf(j->f, "%s%.9g", i ? ", " : "", isfinite(v[i]) ? (double)v[i] : 0.0);
  fputs("]", j->f);
}

// ---------------------------------------------------------------------------
// decoding helpers
// ---------------------------------------------------------------------------

/** Human-readable list of the mask-mode bits, so a reader does not have to
    look up what 6 means. */
static void _emit_mask_mode(json_t *j, const uint32_t mode)
{
  _j_int(j, "mask_mode", mode);
  _j_indent(j);
  fprintf(j->f, "\"mask_mode_names\": [");
  const struct { uint32_t bit; const char *name; } bits[] = {
    { DEVELOP_MASK_ENABLED,     "enabled" },
    { DEVELOP_MASK_MASK,        "drawn" },
    { DEVELOP_MASK_CONDITIONAL, "parametric" },
    { DEVELOP_MASK_RASTER,      "raster" },
    { DEVELOP_MASK_FLEXI,       "flexi" },
  };
  gboolean first = TRUE;
  for(size_t i = 0; i < sizeof(bits) / sizeof(bits[0]); i++)
    if(mode & bits[i].bit)
    {
      fprintf(j->f, "%s\"%s\"", first ? "" : ", ", bits[i].name);
      first = FALSE;
    }
  fputs("]", j->f);
}

static const char *_cst_name(const int32_t cst)
{
  switch(cst)
  {
    case DEVELOP_BLEND_CS_NONE:        return "none";
    case DEVELOP_BLEND_CS_RAW:         return "raw";
    case DEVELOP_BLEND_CS_LAB:         return "Lab";
    case DEVELOP_BLEND_CS_RGB_DISPLAY: return "rgb_display";
    case DEVELOP_BLEND_CS_RGB_SCENE:   return "rgb_scene";
    default:                           return "unknown";
  }
}

static const char *_form_type_name(const int type)
{
  // the primitive bit, ignoring the clone/non-clone qualifiers
  if(type & DT_MASKS_CIRCLE)   return "circle";
  if(type & DT_MASKS_PATH)     return "path";
  if(type & DT_MASKS_GROUP)    return "group";
  if(type & DT_MASKS_GRADIENT) return "gradient";
  if(type & DT_MASKS_ELLIPSE)  return "ellipse";
  if(type & DT_MASKS_BRUSH)    return "brush";
  return "unknown";
}

/** Bytes of a stored group point for a given masks version.

    Mirrors the stride selection in dt_masks_read_masks_history(): group points
    have grown fields over time (refinement in v7, name in v8, group_opacity in
    v9, group_start in v10) and an older blob is shorter than the current
    struct. Reading with the wrong stride would silently misinterpret every
    field after the first, so this has to track that function exactly. */
static size_t _group_point_stride(const int version)
{
  if(version < 7)  return offsetof(dt_masks_point_group_t, refinement);
  if(version < 8)  return offsetof(dt_masks_point_group_t, name);
  if(version < 9)  return offsetof(dt_masks_point_group_t, group_opacity);
  if(version < 10) return offsetof(dt_masks_point_group_t, group_start);
  return sizeof(dt_masks_point_group_t);
}

static size_t _point_stride(const int type, const int version)
{
  if(type & DT_MASKS_GROUP)    return _group_point_stride(version);
  if(type & DT_MASKS_CIRCLE)   return sizeof(dt_masks_point_circle_t);
  if(type & DT_MASKS_ELLIPSE)  return sizeof(dt_masks_point_ellipse_t);
  if(type & DT_MASKS_PATH)     return sizeof(dt_masks_point_path_t);
  if(type & DT_MASKS_BRUSH)    return sizeof(dt_masks_point_brush_t);
  if(type & DT_MASKS_GRADIENT) return sizeof(dt_masks_point_gradient_t);
  return 0;
}

// ---------------------------------------------------------------------------
// per-form emission
// ---------------------------------------------------------------------------

/** Emit one decoded point. Every branch here lists the fields explicitly; in
    particular the group branch emits everything in dt_masks_point_group_t
    *except* `name`, which is user-typed free text. */
static void _emit_point(json_t *j,
                        const int type,
                        const int version,
                        const void *p)
{
  _j_open(j, NULL, '{');

  if(type & DT_MASKS_CIRCLE)
  {
    const dt_masks_point_circle_t *c = p;
    _j_float_array(j, "center", c->center, 2);
    _j_float(j, "radius", c->radius);
    _j_float(j, "border", c->border);
  }
  else if(type & DT_MASKS_ELLIPSE)
  {
    const dt_masks_point_ellipse_t *e = p;
    _j_float_array(j, "center", e->center, 2);
    _j_float_array(j, "radius", e->radius, 2);
    _j_float(j, "rotation", e->rotation);
    _j_float(j, "border", e->border);
    _j_int(j, "flags", e->flags);
  }
  else if(type & DT_MASKS_PATH)
  {
    const dt_masks_point_path_t *q = p;
    _j_float_array(j, "corner", q->corner, 2);
    _j_float_array(j, "ctrl1", q->ctrl1, 2);
    _j_float_array(j, "ctrl2", q->ctrl2, 2);
    _j_float_array(j, "border", q->border, 2);
    _j_int(j, "state", q->state);
  }
  else if(type & DT_MASKS_BRUSH)
  {
    const dt_masks_point_brush_t *b = p;
    _j_float_array(j, "corner", b->corner, 2);
    _j_float_array(j, "ctrl1", b->ctrl1, 2);
    _j_float_array(j, "ctrl2", b->ctrl2, 2);
    _j_float_array(j, "border", b->border, 2);
    _j_float(j, "density", b->density);
    _j_float(j, "hardness", b->hardness);
    _j_int(j, "state", b->state);
  }
  else if(type & DT_MASKS_GRADIENT)
  {
    const dt_masks_point_gradient_t *g = p;
    _j_float_array(j, "anchor", g->anchor, 2);
    _j_float(j, "rotation", g->rotation);
    _j_float(j, "compression", g->compression);
    _j_float(j, "steepness", g->steepness);
    _j_float(j, "curvature", g->curvature);
    _j_int(j, "state", g->state);
  }
  else if(type & DT_MASKS_GROUP)
  {
    const dt_masks_point_group_t *g = p;
    _j_int(j, "formid", g->formid);
    _j_int(j, "parentid", g->parentid);
    _j_int(j, "state", g->state);
    _j_float(j, "opacity", g->opacity);

    // `name` is deliberately absent: it is user-typed free text living inside
    // the points blob. See harvest.h.

    if(version >= 7)
    {
      _j_open(j, "refinement", '{');
      _j_int(j, "enabled", g->refinement.enabled);
      _j_float(j, "feathering_radius", g->refinement.feathering_radius);
      _j_int(j, "feathering_guide", g->refinement.feathering_guide);
      _j_float(j, "blur_radius", g->refinement.blur_radius);
      _j_float(j, "contrast", g->refinement.contrast);
      _j_float(j, "brightness", g->refinement.brightness);
      _j_float(j, "details", g->refinement.details);
      _j_close(j, '}');
    }
    if(version >= 9) _j_float(j, "group_opacity", g->group_opacity);
    if(version >= 10) _j_int(j, "group_start", g->group_start);
  }

  _j_close(j, '}');
}

// ---------------------------------------------------------------------------
// coverage tally
//
// A harvest file is worth collecting in proportion to what it contains that we
// have not seen before, and that is not visible by eye in 42MB of JSON. So the
// scan tallies the configurations that are *rare* -- the ones a synthetic
// fixture has to stand in for today because no real edit exercises them.
//
// The inverted and inclusive mask-combine modes are the specific reason this
// exists. Between them they select the XOR-folding and constant-collapse
// branches of the migration, and in the library this was first developed
// against, INV appears zero times and INCL eight. That is not a quirk of one
// library so much as of the settings themselves -- almost nobody changes them
// -- but "almost nobody" is not nobody, and a contributor who does use them
// holds the only real test data for those branches in existence.
//
// Reporting the tally on stdout means such a contributor is told their file is
// unusual, and reporting it in the JSON means an incoming file can be triaged
// on its rare-case counts without parsing all of it.
// ---------------------------------------------------------------------------

#define HARVEST_MAX_VERSION 24

typedef struct
{
  int combine_inv, combine_incl, combine_masks_pos;
  int case_drawn_only, case_parametric_only, case_drawn_and_parametric;
  int case_raster, case_already_flexi;
  int uses_feathering, uses_details, uses_blur, uses_contrast_or_brightness;
  int per_shape_refinement, custom_group_opacity, explicit_group_start;
  int blendop_version[HARVEST_MAX_VERSION];
  int masks_version[HARVEST_MAX_VERSION];
  int form_type_circle, form_type_ellipse, form_type_path;
  int form_type_brush, form_type_gradient, form_type_group;
} harvest_stats_t;

static void _tally_version(int *bucket, const int v)
{
  if(v >= 0 && v < HARVEST_MAX_VERSION) bucket[v]++;
}

/** Emit a version histogram as an object keyed by version number, listing only
    the versions actually present. */
static void _emit_version_histogram(json_t *j, const char *key, const int *bucket)
{
  _j_open(j, key, '{');
  for(int v = 0; v < HARVEST_MAX_VERSION; v++)
    if(bucket[v])
    {
      char name[16];
      snprintf(name, sizeof(name), "%d", v);
      _j_int(j, name, bucket[v]);
    }
  _j_close(j, '}');
}

/** Read the forms attached to one history entry and emit them. */
static int _emit_forms(json_t *j,
                       sqlite3 *db,
                       const int imgid,
                       const int num,
                       harvest_stats_t *st)
{
  sqlite3_stmt *stmt;
  // Every form *visible* at this history position, not just the ones written
  // at it.
  //
  // masks_history stores a row when a form is created or changed, under the
  // history entry that changed it. A later entry that merely references an
  // existing mask writes no row of its own, so selecting `num = ?2` returns
  // nothing for it -- and the edit then replays with a dangling mask_id and no
  // geometry, which renders as the "no form" fallback instead of as the user's
  // actual mask.
  //
  // That is what dt_masks_read_masks_history() does too (it reads every row
  // with num < history_end and lets later rows for the same formid win), and
  // getting it wrong here produced 22 spurious "the migration changed this
  // mask" results whose real cause was that half the geometry was missing.
  //
  // SQLite's documented bare-column behaviour applies: with MAX(num) in the
  // select list, the other columns come from the row that supplied the
  // maximum, which is exactly the latest version of each form.
  const char *q = "SELECT formid, form, version, points, points_count, source,"
                  "       MAX(num)"
                  " FROM masks_history WHERE imgid = ?1 AND num <= ?2"
                  " GROUP BY formid"
                  " ORDER BY formid";
  if(sqlite3_prepare_v2(db, q, -1, &stmt, NULL) != SQLITE_OK) return 0;
  sqlite3_bind_int(stmt, 1, imgid);
  sqlite3_bind_int(stmt, 2, num);

  int n = 0;
  _j_open(j, "forms", '[');
  while(sqlite3_step(stmt) == SQLITE_ROW)
  {
    const int formid = sqlite3_column_int(stmt, 0);
    const int type = sqlite3_column_int(stmt, 1);
    const int version = sqlite3_column_int(stmt, 2);

    _tally_version(st->masks_version, version);
    if(type & DT_MASKS_CIRCLE)   st->form_type_circle++;
    if(type & DT_MASKS_ELLIPSE)  st->form_type_ellipse++;
    if(type & DT_MASKS_PATH)     st->form_type_path++;
    if(type & DT_MASKS_BRUSH)    st->form_type_brush++;
    if(type & DT_MASKS_GRADIENT) st->form_type_gradient++;
    if(type & DT_MASKS_GROUP)    st->form_type_group++;
    const void *pts = sqlite3_column_blob(stmt, 3);
    const int pts_bytes = sqlite3_column_bytes(stmt, 3);
    const int pts_count = sqlite3_column_int(stmt, 4);
    const void *src = sqlite3_column_blob(stmt, 5);
    const int src_bytes = sqlite3_column_bytes(stmt, 5);

    _j_open(j, NULL, '{');
    _j_int(j, "formid", formid);
    _j_int(j, "type", type);
    _j_str(j, "type_name", _form_type_name(type));
    _j_int(j, "version", version);

    // `masks_history.name` is deliberately not selected: user-renameable.

    // source point, for clone/heal forms. Historic rows stored 2, 3 or 4
    // floats (the 4-float form carried an unused scale field); normalise to
    // the three the current code keeps, as dt_masks_read_masks_history does.
    float source[3] = { 0.0f, 0.0f, 0.0f };
    if(src && src_bytes >= (int)(sizeof(float) * 2))
      memcpy(source, src, MIN((size_t)src_bytes, sizeof(float) * 3));
    _j_float_array(j, "source", source, 3);

    const size_t stride = _point_stride(type, version);
    _j_int(j, "points_count", pts_count);

    if(stride > 0 && pts && pts_count > 0
       && (size_t)pts_bytes >= stride * (size_t)pts_count)
    {
      _j_open(j, "points", '[');
      for(int i = 0; i < pts_count; i++)
      {
        // Zero-fill the tail exactly as the loader does, so a point from an
        // older masks version is reported with the same neutral defaults the
        // reader would give it rather than with uninitialised memory.
        char point[sizeof(dt_masks_point_group_t)];
        memset(point, 0, sizeof(point));
        memcpy(point, (const char *)pts + stride * (size_t)i, stride);
        _emit_point(j, type, version, point);

        if(type & DT_MASKS_GROUP)
        {
          const dt_masks_point_group_t *g = (const dt_masks_point_group_t *)point;
          if(version >= 7 && g->refinement.enabled) st->per_shape_refinement++;
          if(version >= 9 && g->group_opacity != 1.0f) st->custom_group_opacity++;
          if(version >= 10 && g->group_start) st->explicit_group_start++;
        }
      }
      _j_close(j, ']');
    }
    else
    {
      // A blob that does not match its declared stride and count is not
      // something to guess at: say so in the output rather than emitting
      // fabricated geometry.
      _j_int(j, "points_blob_bytes", pts_bytes);
      _j_str(j, "points_error", "blob size does not match stride * count");
    }

    _j_close(j, '}');
    n++;
  }
  _j_close(j, ']');
  sqlite3_finalize(stmt);
  return n;
}

// ---------------------------------------------------------------------------
// blend params
// ---------------------------------------------------------------------------

static void _emit_blend_params(json_t *j, const dt_develop_blend_params_t *b)
{
  _j_open(j, "blend", '{');
  _emit_mask_mode(j, b->mask_mode);
  _j_int(j, "blend_cst", b->blend_cst);
  _j_str(j, "blend_cst_name", _cst_name(b->blend_cst));
  _j_int(j, "blend_mode", b->blend_mode);
  _j_float(j, "blend_parameter", b->blend_parameter);
  _j_float(j, "opacity", b->opacity);
  _j_int(j, "mask_combine", b->mask_combine);
  _j_int(j, "mask_id", b->mask_id);
  _j_int(j, "blendif", b->blendif);
  _j_float(j, "feathering_radius", b->feathering_radius);
  _j_int(j, "feathering_guide", b->feathering_guide);
  _j_float(j, "blur_radius", b->blur_radius);
  _j_float(j, "contrast", b->contrast);
  _j_float(j, "brightness", b->brightness);
  _j_float(j, "details", b->details);
  _j_int(j, "feather_version", b->feather_version);
  _j_float_array(j, "blendif_parameters", b->blendif_parameters,
                 4 * DEVELOP_BLENDIF_SIZE);
  _j_float_array(j, "blendif_boost_factors", b->blendif_boost_factors,
                 DEVELOP_BLENDIF_SIZE);
  // raster_mask_source is a module operation name (e.g. "retouch"), not user
  // text -- the set of possible values is fixed by the build.
  _j_str(j, "raster_mask_source", b->raster_mask_source);
  _j_int(j, "raster_mask_instance", b->raster_mask_instance);
  _j_int(j, "raster_mask_id", b->raster_mask_id);
  _j_int(j, "raster_mask_invert", b->raster_mask_invert ? 1 : 0);
  _j_close(j, '}');
}

// ---------------------------------------------------------------------------
// the harvest
// ---------------------------------------------------------------------------

/** Write a gzip copy of `src_path` at `dst_path`.

    The point is that both files exist afterwards. The person who ran this is
    being asked to send us the result, and being asked to read it first -- so
    the readable JSON stays, and the compressed copy is the one to upload. It
    is a large ratio for this data (a 143 MB harvest goes to about 12 MB), which
    is the difference between "attach it to a forum post" and "find a file
    host".

    Streamed rather than read into memory: these files run to hundreds of MB.
    Failure is not fatal to the harvest -- the JSON is already safely written,
    so the caller reports it and carries on. */
static gboolean _gzip_file(const char *src_path,
                           const char *dst_path,
                           GError **err)
{
  GFile *src = g_file_new_for_path(src_path);
  GFile *dst = g_file_new_for_path(dst_path);
  gboolean ok = FALSE;

  GFileInputStream *in = g_file_read(src, NULL, err);
  if(in)
  {
    GFileOutputStream *out =
      g_file_replace(dst, NULL, FALSE, G_FILE_CREATE_NONE, NULL, err);
    if(out)
    {
      GZlibCompressor *comp =
        g_zlib_compressor_new(G_ZLIB_COMPRESSOR_FORMAT_GZIP, -1);
      GOutputStream *zout =
        g_converter_output_stream_new(G_OUTPUT_STREAM(out), G_CONVERTER(comp));

      ok = g_output_stream_splice(zout, G_INPUT_STREAM(in),
                                  G_OUTPUT_STREAM_SPLICE_CLOSE_SOURCE
                                  | G_OUTPUT_STREAM_SPLICE_CLOSE_TARGET,
                                  NULL, err) >= 0;

      g_object_unref(zout);
      g_object_unref(comp);
      g_object_unref(out);
    }
    g_object_unref(in);
  }

  g_object_unref(src);
  g_object_unref(dst);
  return ok;
}

/** Size of a file in bytes, or -1. */
static goffset _file_size(const char *path)
{
  GFile *f = g_file_new_for_path(path);
  GFileInfo *info = g_file_query_info(f, G_FILE_ATTRIBUTE_STANDARD_SIZE,
                                      G_FILE_QUERY_INFO_NONE, NULL, NULL);
  const goffset size = info ? g_file_info_get_size(info) : -1;
  if(info) g_object_unref(info);
  g_object_unref(f);
  return size;
}

gboolean dt_masks_harvest_library(const char *library_path,
                                  const char *output_path)
{
  sqlite3 *db = NULL;

  // Read-only, and belt-and-braces about it: the URI says mode=ro and the open
  // flags say SQLITE_OPEN_READONLY. Either alone would do; both together mean
  // a future edit cannot quietly drop the guarantee by touching one of them.
  gchar *uri = g_strdup_printf("file:%s?mode=ro", library_path);
  const int rc = sqlite3_open_v2(uri, &db,
                                 SQLITE_OPEN_READONLY | SQLITE_OPEN_URI, NULL);
  g_free(uri);

  if(rc != SQLITE_OK)
  {
    fprintf(stderr, "[harvest] cannot open library read-only: %s\n  %s\n",
            library_path, db ? sqlite3_errmsg(db) : "(no handle)");
    if(db) sqlite3_close(db);
    return FALSE;
  }

  FILE *f = g_fopen(output_path, "wb");
  if(!f)
  {
    fprintf(stderr, "[harvest] cannot write output file: %s\n", output_path);
    sqlite3_close(db);
    return FALSE;
  }

  json_t j = { .f = f, .indent = 0, .need_comma = FALSE };

  fputs("{", f);
  j.indent = 1;
  j.need_comma = FALSE;

  _j_str(&j, "format", "darktable-mask-harvest");
  _j_int(&j, "format_version", HARVEST_FORMAT_VERSION);
  _j_str(&j, "darktable_version", darktable_package_version);
  _j_int(&j, "current_blend_version", DEVELOP_BLEND_VERSION);
  _j_int(&j, "current_masks_version", dt_masks_version());
  _j_str(&j, "contents",
         "Mask configurations only. No file names, no folder or film-roll "
         "names, no shape or group names, no module instance names, no image "
         "content, no thumbnails, no timestamps, no EXIF. Image identity is "
         "reduced to pixel dimensions and a sequential index.");

  _j_open(&j, "edits", '[');

  sqlite3_stmt *stmt;
  // Only what is needed, and nothing free-text: no filename, no multi_name.
  const char *query =
    "SELECT h.imgid, h.num, h.operation, h.blendop_params, h.blendop_version,"
    "       h.multi_priority, h.enabled, i.width, i.height"
    " FROM history h JOIN images i ON i.id = h.imgid"
    " WHERE h.blendop_params IS NOT NULL"
    " ORDER BY h.imgid, h.num";

  if(sqlite3_prepare_v2(db, query, -1, &stmt, NULL) != SQLITE_OK)
  {
    fprintf(stderr, "[harvest] query failed: %s\n", sqlite3_errmsg(db));
    fclose(f);
    sqlite3_close(db);
    return FALSE;
  }

  harvest_stats_t st;
  memset(&st, 0, sizeof(st));

  int scanned = 0, harvested = 0, skipped_no_mask = 0, skipped_size = 0;
  int total_forms = 0, last_imgid = -1, image_index = -1, images_with_masks = 0;
  gboolean counted_this_image = FALSE;

  while(sqlite3_step(stmt) == SQLITE_ROW)
  {
    scanned++;
    const int imgid = sqlite3_column_int(stmt, 0);
    const int num = sqlite3_column_int(stmt, 1);
    const char *operation = (const char *)sqlite3_column_text(stmt, 2);
    const void *bp = sqlite3_column_blob(stmt, 3);
    const int bp_bytes = sqlite3_column_bytes(stmt, 3);
    const int bp_version = sqlite3_column_int(stmt, 4);
    const int multi_priority = sqlite3_column_int(stmt, 5);
    const int enabled = sqlite3_column_int(stmt, 6);
    const int width = sqlite3_column_int(stmt, 7);
    const int height = sqlite3_column_int(stmt, 8);

    if(imgid != last_imgid)
    {
      last_imgid = imgid;
      image_index++;
      counted_this_image = FALSE;
    }

    // Only entries whose blob is the size this build's struct expects can be
    // decoded field by field. An older, shorter blendop version would need
    // dt_develop_blend_legacy_params to interpret, which needs a module and a
    // pipe -- out of scope for a read-only scan. Those are counted and
    // reported rather than silently dropped, so a library full of them is
    // visible as such instead of looking like a library with no masks.
    if(!bp || bp_bytes != (int)sizeof(dt_develop_blend_params_t))
    {
      skipped_size++;
      continue;
    }

    dt_develop_blend_params_t blend;
    memcpy(&blend, bp, sizeof(blend));

    // No mask of any kind -> nothing for the migration to do, nothing to
    // verify. Uniform-opacity blends are not interesting here.
    const uint32_t interesting = DEVELOP_MASK_MASK | DEVELOP_MASK_CONDITIONAL
                               | DEVELOP_MASK_RASTER | DEVELOP_MASK_FLEXI;
    if(!(blend.mask_mode & interesting))
    {
      skipped_no_mask++;
      continue;
    }

    if(!counted_this_image)
    {
      images_with_masks++;
      counted_this_image = TRUE;
    }

    _j_open(&j, NULL, '{');
    _j_int(&j, "index", harvested);
    // A sequential index, not the database id: nothing here can be correlated
    // with anything outside this file.
    _j_int(&j, "image_index", image_index);
    _j_open(&j, "image", '{');
    _j_int(&j, "width", width);
    _j_int(&j, "height", height);
    _j_close(&j, '}');

    _j_str(&j, "operation", operation ? operation : "");
    _j_int(&j, "multi_priority", multi_priority);
    _j_int(&j, "enabled", enabled);
    _j_int(&j, "blendop_version", bp_version);

    // tally the rare configurations (see harvest_stats_t)
    if(blend.mask_combine & DEVELOP_COMBINE_INV) st.combine_inv++;
    if(blend.mask_combine & DEVELOP_COMBINE_INCL) st.combine_incl++;
    if(blend.mask_combine & DEVELOP_COMBINE_MASKS_POS) st.combine_masks_pos++;

    const gboolean has_drawn = (blend.mask_mode & DEVELOP_MASK_MASK) != 0;
    const gboolean has_param = (blend.mask_mode & DEVELOP_MASK_CONDITIONAL) != 0;
    if(blend.mask_mode & DEVELOP_MASK_FLEXI) st.case_already_flexi++;
    else if(blend.mask_mode & DEVELOP_MASK_RASTER) st.case_raster++;
    else if(has_drawn && has_param) st.case_drawn_and_parametric++;
    else if(has_drawn) st.case_drawn_only++;
    else if(has_param) st.case_parametric_only++;

    if(blend.feathering_radius != 0.0f) st.uses_feathering++;
    if(blend.details != 0.0f) st.uses_details++;
    if(blend.blur_radius != 0.0f) st.uses_blur++;
    if(blend.contrast != 0.0f || blend.brightness != 0.0f)
      st.uses_contrast_or_brightness++;

    _tally_version(st.blendop_version, bp_version);

    _emit_blend_params(&j, &blend);
    total_forms += _emit_forms(&j, db, imgid, num, &st);

    _j_close(&j, '}');
    harvested++;
  }
  sqlite3_finalize(stmt);

  _j_close(&j, ']');

  _j_open(&j, "summary", '{');
  _j_int(&j, "history_entries_scanned", scanned);
  _j_int(&j, "edits_harvested", harvested);
  _j_int(&j, "images_with_masks", images_with_masks);
  _j_int(&j, "forms_harvested", total_forms);
  _j_int(&j, "skipped_no_mask", skipped_no_mask);
  _j_int(&j, "skipped_unsupported_blendop_size", skipped_size);
  _j_close(&j, '}');

  // What is in here that we may not have seen before. Kept as its own section
  // so an incoming file can be triaged on these counts without reading the
  // whole of it.
  _j_open(&j, "coverage", '{');
  _j_open(&j, "mask_combine", '{');
  _j_int(&j, "inverted", st.combine_inv);
  _j_int(&j, "inclusive", st.combine_incl);
  _j_int(&j, "drawn_mask_polarity", st.combine_masks_pos);
  _j_close(&j, '}');
  _j_open(&j, "cases", '{');
  _j_int(&j, "drawn_only", st.case_drawn_only);
  _j_int(&j, "parametric_only", st.case_parametric_only);
  _j_int(&j, "drawn_and_parametric", st.case_drawn_and_parametric);
  _j_int(&j, "raster", st.case_raster);
  _j_int(&j, "already_flexi", st.case_already_flexi);
  _j_close(&j, '}');
  _j_open(&j, "refinements", '{');
  _j_int(&j, "feathering", st.uses_feathering);
  _j_int(&j, "details", st.uses_details);
  _j_int(&j, "blur", st.uses_blur);
  _j_int(&j, "contrast_or_brightness", st.uses_contrast_or_brightness);
  _j_int(&j, "per_shape_refinement", st.per_shape_refinement);
  _j_int(&j, "custom_group_opacity", st.custom_group_opacity);
  _j_int(&j, "explicit_group_start", st.explicit_group_start);
  _j_close(&j, '}');
  _j_open(&j, "form_types", '{');
  _j_int(&j, "circle", st.form_type_circle);
  _j_int(&j, "ellipse", st.form_type_ellipse);
  _j_int(&j, "path", st.form_type_path);
  _j_int(&j, "brush", st.form_type_brush);
  _j_int(&j, "gradient", st.form_type_gradient);
  _j_int(&j, "group", st.form_type_group);
  _j_close(&j, '}');
  _emit_version_histogram(&j, "blendop_versions", st.blendop_version);
  _emit_version_histogram(&j, "masks_versions", st.masks_version);
  _j_close(&j, '}');

  j.indent = 0;
  fputs("\n}\n", f);
  fclose(f);
  sqlite3_close(db);

  printf("[harvest] read-only scan of %s\n", library_path);
  printf("[harvest]   history entries scanned : %d\n", scanned);
  printf("[harvest]   edits with masks        : %d\n", harvested);
  printf("[harvest]   images involved         : %d\n", images_with_masks);
  printf("[harvest]   mask forms              : %d\n", total_forms);
  if(skipped_size)
    printf("[harvest]   skipped (old blendop)   : %d\n", skipped_size);
  printf("[harvest] wrote %s\n", output_path);

  // and a compressed copy alongside it, so sharing the result does not need a
  // second tool the person may not have (notably on Windows)
  gchar *gz_path = g_strconcat(output_path, ".gz", NULL);
  GError *gz_err = NULL;
  const gboolean gz_ok = _gzip_file(output_path, gz_path, &gz_err);
  if(gz_ok)
  {
    const goffset raw = _file_size(output_path);
    const goffset gz = _file_size(gz_path);
    if(raw > 0 && gz > 0)
    {
      gchar *raw_s = g_format_size(raw);
      gchar *gz_s = g_format_size(gz);
      printf("[harvest] wrote %s  (%s, from %s)\n", gz_path, gz_s, raw_s);
      g_free(raw_s);
      g_free(gz_s);
    }
    else
      printf("[harvest] wrote %s\n", gz_path);
  }
  else
  {
    printf("[harvest] could not write %s: %s\n"
           "[harvest] (not a problem -- the JSON above is complete;"
           " compress it yourself if you like)\n",
           gz_path, gz_err ? gz_err->message : "unknown error");
  }
  g_clear_error(&gz_err);
  // Say plainly when a library holds one of the configurations we have no real
  // test data for, so the person who ran it knows their file is worth sending
  // even if the totals look unremarkable.
  if(st.combine_inv || st.combine_incl)
  {
    printf("[harvest]\n[harvest] This library contains rarely-used mask combine modes:\n");
    if(st.combine_inv)
      printf("[harvest]   inverted mask combine : %d edit(s)\n", st.combine_inv);
    if(st.combine_incl)
      printf("[harvest]   inclusive mask combine: %d edit(s)\n", st.combine_incl);
    printf("[harvest] These exercise migration paths that almost no real edit "
           "reaches,\n[harvest] so this file is especially useful to us.\n");
  }
  printf("[harvest]\n");
  printf("[harvest] The output is plain JSON containing only numbers and "
         "module names.\n"
         "[harvest] It has no file names, folder names, shape or group names, "
         "or image content.\n"
         "[harvest] Please open it and check before sharing it.\n");
  if(gz_ok)
    printf("[harvest] Read %s, then send %s -- they hold the same thing.\n",
           output_path, gz_path);
  g_free(gz_path);

  return TRUE;
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
