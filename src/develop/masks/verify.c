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

#include "develop/masks/verify.h"

#include "common/darktable.h"
#include "develop/blend.h"
#include "develop/imageop.h"
#include "develop/masks.h"
#include "develop/masks/harvest_read.h"
#include "develop/masks/probe_image.h"
#include "common/iop_profile.h"
#include "develop/pixelpipe.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <gio/gio.h>
#include <glib/gstdio.h>
#include <json-glib/json-glib.h>
#include <math.h>
#include <stdio.h>

// A migrated mask is expected to be *bit* identical in the ordinary case: the
// migration is meant to re-express the same computation, not to approximate
// it. But several paths legitimately reassociate float arithmetic -- an
// opacity that used to multiply once at the end now multiplies per element,
// for instance -- so an exact comparison would report noise as breakage.
//
// These thresholds separate the three answers worth distinguishing: identical
// (nothing moved), equivalent (moved by less than the mask's own 8-bit
// representable step, so nothing a user could see), and different.
//
// The shape of the test is darktable's own, taken from the integration suite's
// `deltae`: a result fails when EITHER the worst pixel exceeds the tolerance OR
// the mean over the whole frame exceeds a third of it
// (`max_dE > MAX_DELTA_E or mean_dE > MAX_DELTA_E / 3`), with a much tighter
// max deciding "identical". Two conditions rather than one, because a worst-
// pixel figure on its own answers whether anything differs and nothing about
// how much: one pixel landing on the far side of a threshold on a mask
// boundary and half the frame being wrong both report 1.0. The mean is what
// tells them apart -- it is the magnitude weighted by the area it covers -- and
// `differing_pixels` in the per-edit rows says how much of the frame took part.
//
// Same structure, mask units instead of delta-E: a mask is 0..1 module
// strength, so the tolerance is the 8-bit step it is stored and displayed at.
#define VERIFY_EPS_IDENTICAL 1e-6f
#define VERIFY_EPS_EQUIVALENT (1.0f / 255.0f)
#define VERIFY_EPS_EQUIVALENT_MEAN (VERIFY_EPS_EQUIVALENT / 3.0f)

// Replaying at full sensor resolution would spend most of the run in
// rasterisation for no extra discrimination -- a difference in mask geometry
// shows up at any resolution. The harvested aspect ratio is preserved, since
// masks are stored normalised and a wrong aspect would distort every shape.
#define VERIFY_MAX_EDGE 512

typedef enum
{
  VERIFY_IDENTICAL = 0,
  VERIFY_EQUIVALENT,
  VERIFY_DIFFERENT,
  VERIFY_SKIPPED,
  VERIFY_ERROR,
} verify_result_t;

typedef struct
{
  int total;
  int identical, equivalent, different, skipped, error;
  int inert_before;      // classic mask uniform: comparison proves nothing
  int live;              // classic mask genuinely varies: a real test
  int live_identical, live_equivalent, live_different;
  double worst_max_diff;
  int worst_index;
  // the same edit's mean and differing-pixel count, so the headline number can
  // be read as a magnitude and not just as "something differs somewhere"
  double worst_mean_diff;
  int worst_differing_pixels;

  int gpu_compared;              // edits where both GPU renders succeeded
  double worst_gpu_diff;         // GPU: worst classic-vs-migrated
  int worst_gpu_index;
  double worst_gpu_mean_diff;
  int worst_gpu_differing_pixels;

  // what those mask differences did to the rendered image
  int image_compared;
  double worst_image_diff, worst_image_mean_diff;
  int worst_image_differing_pixels, worst_image_index;
  int gpu_image_compared;
  double worst_gpu_image_diff, worst_gpu_image_mean_diff;
  int worst_gpu_image_differing_pixels, worst_gpu_image_index;
  double worst_dev_before;       // worst CPU/GPU gap on classic edits
  double worst_dev_after;        // ... and on migrated ones
  int dev_gap_widened;           // migrated gap worse than classic by >1/255
  // ... and how many of those survive with the mask post-processing switched
  // off, i.e. are migration's to answer for rather than a downstream stage's
  // (see dev_diff_after_nopost)
  int dev_gap_widened_own;
} verify_stats_t;

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

static gint64 _obj_int(JsonObject *o, const char *k, const gint64 dflt)
{
  if(!json_object_has_member(o, k)) return dflt;
  JsonNode *n = json_object_get_member(o, k);
  if(JSON_NODE_HOLDS_NULL(n)) return dflt;
  return json_node_get_int(n);
}

static float _obj_float(JsonObject *o, const char *k, const float dflt)
{
  if(!json_object_has_member(o, k)) return dflt;
  JsonNode *n = json_object_get_member(o, k);
  if(JSON_NODE_HOLDS_NULL(n)) return dflt;
  return (float)json_node_get_double(n);
}

static const char *_obj_str(JsonObject *o, const char *k, const char *dflt)
{
  if(!json_object_has_member(o, k)) return dflt;
  JsonNode *n = json_object_get_member(o, k);
  if(JSON_NODE_HOLDS_NULL(n)) return dflt;
  return json_node_get_string(n);
}

/** read a float array member into `out`, up to `n` entries */
static void _obj_float_array(JsonObject *o, const char *k, float *out, const int n)
{
  if(!json_object_has_member(o, k)) return;
  JsonArray *a = json_object_get_array_member(o, k);
  if(!a) return;
  const int len = MIN(n, (int)json_array_get_length(a));
  for(int i = 0; i < len; i++) out[i] = (float)json_array_get_double_element(a, i);
}

// ---------------------------------------------------------------------------
// reconstruction
// ---------------------------------------------------------------------------

/** Rebuild one point of a form from its decoded JSON. Mirrors _emit_point() in
    harvest.c field for field; the one field the harvest deliberately omits
    (a group's user-typed name) is left zeroed, which is exactly what the
    loader does for an edit that has none. */
static void _read_point(JsonObject *p, const int type, void *out)
{
  if(type & DT_MASKS_CIRCLE)
  {
    dt_masks_point_circle_t *c = out;
    _obj_float_array(p, "center", c->center, 2);
    c->radius = _obj_float(p, "radius", 0.0f);
    c->border = _obj_float(p, "border", 0.0f);
  }
  else if(type & DT_MASKS_ELLIPSE)
  {
    dt_masks_point_ellipse_t *e = out;
    _obj_float_array(p, "center", e->center, 2);
    _obj_float_array(p, "radius", e->radius, 2);
    e->rotation = _obj_float(p, "rotation", 0.0f);
    e->border = _obj_float(p, "border", 0.0f);
    e->flags = (int)_obj_int(p, "flags", 0);
  }
  else if(type & DT_MASKS_PATH)
  {
    dt_masks_point_path_t *q = out;
    _obj_float_array(p, "corner", q->corner, 2);
    _obj_float_array(p, "ctrl1", q->ctrl1, 2);
    _obj_float_array(p, "ctrl2", q->ctrl2, 2);
    _obj_float_array(p, "border", q->border, 2);
    q->state = (int)_obj_int(p, "state", 0);
  }
  else if(type & DT_MASKS_BRUSH)
  {
    dt_masks_point_brush_t *b = out;
    _obj_float_array(p, "corner", b->corner, 2);
    _obj_float_array(p, "ctrl1", b->ctrl1, 2);
    _obj_float_array(p, "ctrl2", b->ctrl2, 2);
    _obj_float_array(p, "border", b->border, 2);
    b->density = _obj_float(p, "density", 1.0f);
    b->hardness = _obj_float(p, "hardness", 1.0f);
    b->state = (int)_obj_int(p, "state", 0);
  }
  else if(type & DT_MASKS_GRADIENT)
  {
    dt_masks_point_gradient_t *g = out;
    _obj_float_array(p, "anchor", g->anchor, 2);
    g->rotation = _obj_float(p, "rotation", 0.0f);
    g->compression = _obj_float(p, "compression", 0.0f);
    g->steepness = _obj_float(p, "steepness", 0.0f);
    g->curvature = _obj_float(p, "curvature", 0.0f);
    g->state = (int)_obj_int(p, "state", 0);
  }
  else if(type & DT_MASKS_GROUP)
  {
    dt_masks_point_group_t *g = out;
    g->formid = (dt_mask_id_t)_obj_int(p, "formid", INVALID_MASKID);
    g->parentid = (dt_mask_id_t)_obj_int(p, "parentid", INVALID_MASKID);
    g->state = (int)_obj_int(p, "state", 0);
    g->opacity = _obj_float(p, "opacity", 1.0f);
    g->group_opacity = _obj_float(p, "group_opacity", 1.0f);
    g->group_start = (int)_obj_int(p, "group_start", 0);

    if(json_object_has_member(p, "refinement"))
    {
      JsonObject *r = json_object_get_object_member(p, "refinement");
      if(r)
      {
        g->refinement.enabled = (int)_obj_int(r, "enabled", 0);
        g->refinement.feathering_radius = _obj_float(r, "feathering_radius", 0.0f);
        g->refinement.feathering_guide = (int)_obj_int(r, "feathering_guide", 0);
        g->refinement.blur_radius = _obj_float(r, "blur_radius", 0.0f);
        g->refinement.contrast = _obj_float(r, "contrast", 0.0f);
        g->refinement.brightness = _obj_float(r, "brightness", 0.0f);
        g->refinement.details = _obj_float(r, "details", 0.0f);
      }
    }
  }
}

/** Load a harvest file, transparently decompressing a gzipped one.

    --harvest-masks writes both FILE and FILE.gz and asks contributors to read
    the first and send the second (it is ~12x smaller), so the file that
    actually arrives is nearly always compressed. Requiring it to be unpacked
    first put a manual step between receiving a contribution and checking it,
    for no reason: the magic number says which it is.

    Detected by content rather than by extension, so a .gz that was renamed on
    the way through a file-sharing service, or a plain file that kept the
    suffix, both still work. */
JsonParser *dt_masks_harvest_load(const char *path, GError **error)
{
  JsonParser *parser = json_parser_new();

  guchar magic[2] = { 0, 0 };
  FILE *probe = g_fopen(path, "rb");
  const gboolean gzipped =
    probe && fread(magic, 1, 2, probe) == 2 && magic[0] == 0x1f && magic[1] == 0x8b;
  if(probe) fclose(probe);

  if(!gzipped)
  {
    if(json_parser_load_from_file(parser, path, error)) return parser;
    g_object_unref(parser);
    return NULL;
  }

  GFile *file = g_file_new_for_path(path);
  GFileInputStream *raw = g_file_read(file, NULL, error);
  gboolean ok = FALSE;
  if(raw)
  {
    GZlibDecompressor *decomp = g_zlib_decompressor_new(G_ZLIB_COMPRESSOR_FORMAT_GZIP);
    GInputStream *plain =
      g_converter_input_stream_new(G_INPUT_STREAM(raw), G_CONVERTER(decomp));
    ok = json_parser_load_from_stream(parser, plain, NULL, error);
    g_object_unref(plain);
    g_object_unref(decomp);
    g_object_unref(raw);
  }
  g_object_unref(file);

  if(ok) return parser;
  g_object_unref(parser);
  return NULL;
}

gchar *dt_masks_harvest_edit_key(JsonObject *edit)
{
  // "index" and "image_index" are deliberately absent: they say where the edit
  // sat in the harvest, which changes nothing about what it renders.
  static const char *const members[] =
    { "operation", "blendop_version", "multi_priority", "enabled",
      "image", "blend", "forms", NULL };

  GString *acc = g_string_new(NULL);
  JsonGenerator *gen = json_generator_new();
  for(int i = 0; members[i]; i++)
  {
    JsonNode *n = json_object_get_member(edit, members[i]);
    if(n)
    {
      json_generator_set_root(gen, n);
      gsize len = 0;
      gchar *txt = json_generator_to_data(gen, &len);
      if(txt) g_string_append_len(acc, txt, (gssize)len);
      g_free(txt);
    }
    // a separator so absent and empty members cannot alias into each other
    g_string_append_c(acc, 0x1f);
  }
  g_object_unref(gen);

  gchar *key = g_compute_checksum_for_string(G_CHECKSUM_SHA256, acc->str, acc->len);
  g_string_free(acc, TRUE);
  return key;
}

/** Rebuild the form list for one edit. Returns NULL if anything is
    unreconstructable, so a malformed record is skipped rather than replayed as
    something subtly different from what it recorded. */
GList *dt_masks_harvest_read_forms(JsonArray *forms_arr)
{
  GList *forms = NULL;

  for(guint i = 0; i < json_array_get_length(forms_arr); i++)
  {
    JsonObject *fo = json_array_get_object_element(forms_arr, i);
    if(!fo) goto fail;

    const int type = (int)_obj_int(fo, "type", 0);
    dt_masks_form_t *form = dt_masks_create(type);
    if(!form) goto fail;

    form->formid = (dt_mask_id_t)_obj_int(fo, "formid", INVALID_MASKID);
    form->version = (int)_obj_int(fo, "version", dt_masks_version());
    snprintf(form->name, sizeof(form->name), "form %d", form->formid);
    _obj_float_array(fo, "source", form->source, 3);

    if(json_object_has_member(fo, "points_error"))
    {
      dt_masks_free_form(form);
      goto fail;
    }

    if(json_object_has_member(fo, "points") && form->functions)
    {
      JsonArray *pts = json_object_get_array_member(fo, "points");
      const size_t psize = form->functions->point_struct_size;
      for(guint k = 0; pts && k < json_array_get_length(pts); k++)
      {
        JsonObject *po = json_array_get_object_element(pts, k);
        if(!po) continue;
        void *point = calloc(1, psize);
        if(!point) continue;
        _read_point(po, type, point);
        form->points = g_list_append(form->points, point);
      }
    }

    // Everything is replayed at the current masks version: the harvest already
    // decoded each blob with the historic stride and the loader's zero-fill
    // rules, so what we hold is the post-read state, not the on-disk one.
    form->version = dt_masks_version();

    forms = g_list_append(forms, form);
  }
  return forms;

fail:
  g_list_free_full(forms, (GDestroyNotify)dt_masks_free_form);
  return NULL;
}

void dt_masks_harvest_read_blend_params(JsonObject *b, dt_develop_blend_params_t *p)
{
  memset(p, 0, sizeof(*p));
  p->mask_mode = (uint32_t)_obj_int(b, "mask_mode", 0);
  p->blend_cst = (int32_t)_obj_int(b, "blend_cst", 0);
  p->blend_mode = (uint32_t)_obj_int(b, "blend_mode", 0);
  p->blend_parameter = _obj_float(b, "blend_parameter", 0.0f);
  p->opacity = _obj_float(b, "opacity", 100.0f);
  p->mask_combine = (uint32_t)_obj_int(b, "mask_combine", 0);
  p->mask_id = (dt_mask_id_t)_obj_int(b, "mask_id", INVALID_MASKID);
  p->blendif = (uint32_t)_obj_int(b, "blendif", 0);
  p->feathering_radius = _obj_float(b, "feathering_radius", 0.0f);
  p->feathering_guide = (uint32_t)_obj_int(b, "feathering_guide", 0);
  p->blur_radius = _obj_float(b, "blur_radius", 0.0f);
  p->contrast = _obj_float(b, "contrast", 0.0f);
  p->brightness = _obj_float(b, "brightness", 0.0f);
  p->details = _obj_float(b, "details", 0.0f);
  p->feather_version = (uint32_t)_obj_int(b, "feather_version", 0);
  _obj_float_array(b, "blendif_parameters", p->blendif_parameters,
                   4 * DEVELOP_BLENDIF_SIZE);
  _obj_float_array(b, "blendif_boost_factors", p->blendif_boost_factors,
                   DEVELOP_BLENDIF_SIZE);
  const char *src = _obj_str(b, "raster_mask_source", "");
  g_strlcpy(p->raster_mask_source, src ? src : "", sizeof(p->raster_mask_source));
  p->raster_mask_instance = (int)_obj_int(b, "raster_mask_instance", 0);
  p->raster_mask_id = (dt_mask_id_t)_obj_int(b, "raster_mask_id", INVALID_MASKID);
  p->raster_mask_invert = _obj_int(b, "raster_mask_invert", 0) ? TRUE : FALSE;
}

// ---------------------------------------------------------------------------
// the replay harness
// ---------------------------------------------------------------------------

typedef struct
{
  dt_develop_t dev;
  dt_iop_module_t module;
  dt_dev_pixelpipe_t pipe;
  dt_dev_pixelpipe_iop_t piece;
  gboolean module_loaded;
  gboolean dev_mutex_ready;
  dt_iop_roi_t roi;
  float *probe;
  // What the module under test "produced": the probe with a synthetic effect
  // applied (see _make_module_output). The blend mixes this with `probe`
  // according to the mask, so it is both what makes the rendered image respond
  // to the mask at all and what gives the blendif `_out` channels something of
  // their own to select on.
  float *modout;
  float *out;

  // the upstream module a raster edit reads its mask from, present only for
  // raster edits (see _attach_raster_source)
  dt_iop_module_t source_module;
  dt_dev_pixelpipe_iop_t source_piece;
  gboolean source_loaded;

  // OpenCL device to replay the GPU blend on, or -1 when this build/run has
  // no usable device (the CPU comparison still stands on its own)
  int devid;
} replay_t;

/** The OpenCL device the GPU replays run on, locked once for the whole run
    rather than per edit -- locking and releasing 2466 times would dominate the
    runtime and tells us nothing extra. -1 when unavailable, in which case the
    run reports CPU-only results and says so rather than silently narrowing. */
static int _verify_devid = -1;

/** The output of the module the mask is attached to: the probe at +1 EV.
    Not decoration -- two things depend on the module having actually done
    something, and until this existed it had not.

    The blend computes `out = in * (1 - mask) + module_out * mask`
    (blendif_*.c). Seeding `out` with a copy of the input, as this harness did,
    makes the blend a no-op for every mask: the rendered image is the probe
    whatever the mask says, so there is no image-level effect to measure at all.

    And the parametric channels come in `_in`/`_out` pairs -- blendif evaluates
    the second half against the module's output (see the DEVELOP_BLENDIF_*_out
    reads in blendif_rgb_jzczhz.c). With output equal to input, every `_out`
    channel was silently exercised as a duplicate of its `_in` counterpart, so
    half the parametric channel space was never really tested.

    +1 EV, i.e. a doubling, because exposure is the archetypal masked
    adjustment and because in the scene-linear probe it is exactly `in * 2`:
    the per-pixel effect size `|module_out - in|` is then the image value
    itself -- non-zero everywhere except true black, and never larger than 1.
    That last part matters for reading the numbers: since the image difference
    is the mask difference scaled by that effect, the mask metric is an upper
    bound on the image metric, which is why the verdict stays on the mask.

    Deliberately not clipped: the pipeline is float and scene-linear values
    above 1.0 are ordinary, and clipping would flatten the effect to zero over
    the probe's whole upper half. */
static float *_make_module_output(const float *const probe, const size_t npix)
{
  float *m = dt_alloc_align_float(npix * 4);
  if(!m) return NULL;
  for(size_t i = 0; i < npix * 4; i++) m[i] = probe[i] * 2.0f;
  return m;
}

/** the mask dt_develop_blend_process() published for the last render */
static const float *_published_mask(replay_t *r)
{
  return g_hash_table_lookup(r->piece.raster_masks,
                             GINT_TO_POINTER(BLEND_RASTER_ID));
}

/** Render the mask for the current blend_params/forms, into a caller-owned
    copy. Returns NULL if the blend published nothing. */
static float *_render_mask(replay_t *r, float **image)
{
  const size_t npix = (size_t)r->roi.width * r->roi.height;
  if(image) *image = NULL;

  // the blend writes into `out`, mixing it with the input by the mask -- so it
  // starts as what the module produced, not as a copy of the input (see
  // _make_module_output)
  memcpy(r->out, r->modout, sizeof(float) * npix * 4);

  // pipe->forms is what the drawn/flexi group lookup walks, and migration has
  // may have added forms to dev->forms since the last render
  r->pipe.forms = r->dev.forms;

  // Production wiring, not a shortcut around it. The classic raster branch
  // does not read blend_params->raster_mask_source at all -- it follows
  // module->raster_mask.sink.source, a resolved module pointer that only
  // dt_iop_commit_blend_params() ever sets. Setting that pointer by hand here
  // would have been the harness deciding what the pipe should have resolved;
  // calling the real function instead means the classic side finds its source
  // exactly as it does in a live pipe, and the flexi side gets its raster
  // *form* elements registered by the same call (_reconcile_raster_form_users).
  // Run for every edit, raster or not, so there is one wiring path rather than
  // a special case that only the raster edits exercise.
  dt_develop_blend_params_t committed = *r->module.blend_params;
  dt_iop_commit_blend_params(&r->module, &committed, &r->pipe);

  dt_develop_blend_process(&r->module, &r->piece, r->probe, r->out,
                           &r->roi, &r->roi);

  const float *m = _published_mask(r);
  if(!m) return NULL;

  float *copy = dt_alloc_align_float(npix);
  if(!copy) return NULL;
  memcpy(copy, m, sizeof(float) * npix);

  // the blended image the mask actually produced, for the severity half of the
  // comparison
  if(image)
  {
    *image = dt_alloc_align_float(npix * 4);
    if(*image) memcpy(*image, r->out, sizeof(float) * npix * 4);
  }
  return copy;
}

/** The same render, through dt_develop_blend_process_cl.

    Not an optional extra. dt_develop_blend_process_cl is a *separate,
    hand-maintained* implementation of the same branch structure -- its own
    comments say so ("kept in sync by hand") -- and every mask number this tool
    produced before this existed came from the CPU function alone. Most users
    run OpenCL, so a migration verified only on the CPU is verified on the path
    fewer people take. The one bug this immediately found (mode_parametric where
    the CPU tests mode_drawn, see blend.c) had been sitting in a branch that
    migration itself makes unreachable, so no amount of CPU replay would ever
    have reached it.

    The mask comes back the same way as on the CPU: the tail of the CL function
    copies the finished mask off the device and publishes it through
    dt_iop_piece_set_raster(), so nothing here re-implements the readback. */
static float *_render_mask_cl(replay_t *r, float **image)
{
  if(image) *image = NULL;
#ifdef HAVE_OPENCL
  if(r->devid < 0) return NULL;

  const size_t npix = (size_t)r->roi.width * r->roi.height;
  const int w = r->roi.width, h = r->roi.height;

  float *copy = NULL;
  cl_mem dev_in = dt_opencl_alloc_device(r->devid, w, h, sizeof(float) * 4);
  cl_mem dev_out = dt_opencl_alloc_device(r->devid, w, h, sizeof(float) * 4);
  if(!dev_in || !dev_out) goto done;

  // same starting state as the CPU render: output begins as a copy of the
  // input, which is what a module that did nothing would have produced
  if(dt_opencl_write_host_to_image(r->devid, r->probe, dev_in, w, h, sizeof(float) * 4)
     != CL_SUCCESS) goto done;
  if(dt_opencl_write_host_to_image(r->devid, r->modout, dev_out, w, h, sizeof(float) * 4)
     != CL_SUCCESS) goto done;

  r->pipe.forms = r->dev.forms;
  r->pipe.devid = r->devid;

  dt_develop_blend_params_t committed = *r->module.blend_params;
  dt_iop_commit_blend_params(&r->module, &committed, &r->pipe);

  if(!dt_develop_blend_process_cl(&r->module, &r->piece, dev_in, dev_out,
                                  &r->roi, &r->roi))
    goto done;

  const float *m = _published_mask(r);
  if(!m) goto done;

  copy = dt_alloc_align_float(npix);
  if(copy) memcpy(copy, m, sizeof(float) * npix);

  if(image)
  {
    float *img = dt_alloc_align_float(npix * 4);
    // a failed readback leaves *image NULL, which the caller treats as
    // "no image comparison here" rather than comparing against garbage
    if(img
       && dt_opencl_copy_image_to_host(r->devid, img, dev_out, w, h,
                                       sizeof(float) * 4) == CL_SUCCESS)
      *image = img;
    else
      dt_free_align(img);
  }

done:
  dt_opencl_release_mem_object(dev_in);
  dt_opencl_release_mem_object(dev_out);
  r->pipe.devid = DT_DEVICE_CPU;
  return copy;
#else
  return NULL;
#endif
}

/** How two masks differ: the worst deviation, the mean over every pixel, and
    how many pixels differ at all.

    Max alone answers "is there a difference" and nothing about its size -- one
    stray pixel and a wholly inverted mask both report 1.0. The mean and the
    differing-pixel count are what separate those, so they are collected for the
    GPU comparisons on the same footing as the CPU one rather than left to a
    reader's imagination. */
typedef struct _diff_stats_t
{
  double max;
  double mean;
  int differing;
} _diff_stats_t;

static _diff_stats_t _diff_stats(const float *a, const float *b, const size_t n)
{
  _diff_stats_t st = { 0.0, 0.0, 0 };
  double sum = 0.0;
  for(size_t i = 0; i < n; i++)
  {
    const double d = fabs((double)a[i] - (double)b[i]);
    if(d > st.max) st.max = d;
    sum += d;
    if(d > VERIFY_EPS_IDENTICAL) st.differing++;
  }
  st.mean = n ? sum / (double)n : 0.0;
  return st;
}

/** The same three statistics over a rendered image rather than a mask.

    RGB only: the fourth float of each pixel is not image content, and letting
    it into a mean would dilute every number by a quarter.

    This is the other half of what the integration suite measures. A mask
    difference is the more sensitive signal -- it is the module's strength, so
    it registers wherever the mask moved at all -- while what a user could
    actually see is that difference scaled by how much the module changes the
    pixel underneath. Both are reported, because either alone misleads: the
    mask number alone cannot say whether anything visible happened, and the
    image number alone hides a mask error in regions where this particular
    synthetic effect happens to be small. */
static _diff_stats_t _diff_stats_rgb(const float *a, const float *b, const size_t npix)
{
  _diff_stats_t st = { 0.0, 0.0, 0 };
  double sum = 0.0;
  for(size_t i = 0; i < npix; i++)
  {
    double worst_ch = 0.0;
    for(int c = 0; c < 3; c++)
    {
      const double d = fabs((double)a[i * 4 + c] - (double)b[i * 4 + c]);
      if(d > worst_ch) worst_ch = d;
      sum += d;
    }
    if(worst_ch > st.max) st.max = worst_ch;
    // one pixel, counted once, if any of its channels moved -- the same rule
    // count-diff-pixels applies in the integration suite
    if(worst_ch > VERIFY_EPS_IDENTICAL) st.differing++;
  }
  st.mean = npix ? sum / (double)(npix * 3) : 0.0;
  return st;
}

/** worst absolute deviation between two masks */
static double _max_abs_diff(const float *a, const float *b, const size_t n)
{
  return _diff_stats(a, b, n).max;
}

/** is this mask the same value everywhere? A uniform mask makes the comparison
    vacuous -- it would match another uniform mask regardless of what migration
    did to the configuration that produced it. */
static gboolean _is_uniform(const float *m, const size_t n)
{
  if(n == 0) return TRUE;
  for(size_t i = 1; i < n; i++)
    if(fabsf(m[i] - m[0]) > VERIFY_EPS_IDENTICAL) return FALSE;
  return TRUE;
}

/** Find the module shared-object for an operation name. */
static dt_iop_module_so_t *_find_so(const char *op)
{
  if(!op || !*op) return NULL;
  for(GList *l = darktable.iop; l; l = g_list_next(l))
  {
    dt_iop_module_so_t *so = l->data;
    if(so && !strcmp(so->op, op)) return so;
  }
  return NULL;
}

static void _replay_cleanup(replay_t *r)
{
  if(r->module_loaded) dt_iop_cleanup_module(&r->module);
  if(r->source_loaded) dt_iop_cleanup_module(&r->source_module);
  if(r->dev_mutex_ready) dt_pthread_mutex_destroy(&r->dev.history_mutex);
  if(r->piece.raster_masks) g_hash_table_destroy(r->piece.raster_masks);
  if(r->source_piece.raster_masks) g_hash_table_destroy(r->source_piece.raster_masks);
  g_list_free(r->pipe.nodes);
  g_list_free(r->dev.iop);
  dt_free_align(r->probe);
  dt_free_align(r->modout);
  dt_free_align(r->out);
  g_list_free_full(r->dev.forms, (GDestroyNotify)dt_masks_free_form);
  memset(r, 0, sizeof(*r));
}

/** The mask the upstream module is pretending to have produced.

    Deliberately not flat and not derived from the probe: a raster mask is an
    *input* to everything under test here, so it wants shape of its own --
    enough variation that an inversion, an opacity, a blur or a tone curve each
    leave a distinguishable trace, and enough of the range actually reached
    (exact 0 and exact 1 both occur) that a polarity error cannot hide in the
    interior. The smooth radial falloff paired with the probe's own hard edges
    is also what gives the guided-filter feathering something real to work
    with -- it needs a mask that disagrees with the image. */
static float *_synthetic_raster_mask(const int w, const int h)
{
  float *m = dt_alloc_align_float((size_t)w * h);
  if(!m) return NULL;

  const float cx = 0.5f * (float)(w - 1);
  const float cy = 0.5f * (float)(h - 1);
  const float norm = 1.0f / sqrtf(cx * cx + cy * cy);

  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
    {
      const float dx = ((float)x - cx) * norm;
      const float dy = ((float)y - cy) * norm;
      // radial soft disc, saturating to exactly 1 near the centre and exactly
      // 0 in the corners; the diagonal term breaks the symmetry so a
      // transpose-style error cannot pass
      const float rad = sqrtf(dx * dx + dy * dy) * 1.6f;
      const float diag = 0.15f * (((float)x / (float)MAX(1, w - 1))
                                  - ((float)y / (float)MAX(1, h - 1)));
      const float t = CLAMPF(1.0f - rad + diag, 0.0f, 1.0f);
      // smoothstep, so the interior has real gradient rather than a cone
      m[(size_t)y * w + x] = t * t * (3.0f - 2.0f * t);
    }
  return m;
}

/** Give the replay the upstream piece a raster edit reads from.

    Both the classic raster branch of dt_develop_blend_process() and the flexi
    DT_MASKS_RASTER form (masks/raster.c) resolve their mask through the same
    dt_dev_get_raster_mask(), which walks pipe->nodes for the source piece and
    dev->iop for the source module. With neither populated it returns NULL on
    both sides, and the comparison degenerates to "two empty masks match" --
    which is why these edits used to be skipped outright rather than counted.

    Standing up the source for real is what makes the 123 raster edits
    testable, and it is honest to do it this way: the fetch itself is shared
    code exercised identically by both sides, so what the comparison actually
    isolates is the part that does differ -- classic applying opacity and
    raster_mask_invert inline versus the flexi group compositing the same
    raster as an element -- with the global refinements running downstream of
    both. Returns a skip reason, or NULL on success. */
static const char *_attach_raster_source(replay_t *r,
                                         const dt_develop_blend_params_t *bp)
{
  dt_iop_module_so_t *src_so = _find_so(bp->raster_mask_source);
  if(!src_so) return "raster source module not in this build";

  if(dt_iop_load_module_by_so(&r->source_module, src_so, &r->dev))
    return "raster source instance could not be loaded";
  r->source_loaded = TRUE;
  r->source_module.dev = &r->dev;
  r->source_module.multi_priority = bp->raster_mask_instance;

  // The source has to sit strictly earlier in the pipe or dt_dev_get_raster_mask
  // refuses the fetch outright (and pops a dt_control_log about it). In the
  // edit this came from it necessarily did -- darktable would not have let the
  // user pick it otherwise -- so pinning it just below the target reproduces
  // the real arrangement rather than inventing a favourable one. Asking the
  // order list would not do: the harvested instance number often has no entry,
  // and a source and target of the same op differ only by instance.
  r->source_module.iop_order = r->module.iop_order - 1.0;

  // dt_dev_get_raster_mask discards (and deletes) masks from a source that is
  // disabled or does not write raster masks, so the stand-in has to look like
  // a module that genuinely published one.
  if(r->source_module.blend_params)
    r->source_module.blend_params->mask_mode =
      DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK;

  float *raster = _synthetic_raster_mask(r->roi.width, r->roi.height);
  if(!raster) return "raster mask allocation failure";

  r->source_piece.pipe = &r->pipe;
  r->source_piece.module = &r->source_module;
  r->source_piece.enabled = TRUE;
  r->source_piece.colors = 4;
  r->source_piece.iscale = 1.0f;
  r->source_piece.processed_roi_in = r->roi;
  r->source_piece.processed_roi_out = r->roi;
  r->source_piece.raster_masks =
    g_hash_table_new_full(g_direct_hash, g_direct_equal, NULL, dt_free_align_ptr);
  g_hash_table_insert(r->source_piece.raster_masks,
                      GINT_TO_POINTER(bp->raster_mask_id), raster);

  // source first: the fetch walks nodes forward from the source to the target,
  // and stops as soon as it reaches the target module
  r->pipe.nodes = g_list_append(r->pipe.nodes, &r->source_piece);
  r->pipe.nodes = g_list_append(r->pipe.nodes, &r->piece);
  // _raster_resolve_source() (masks/raster.c) matches op + multi_priority
  // against dev->iop, which is how the flexi side finds the same module the
  // classic side reaches through blend_params
  r->dev.iop = g_list_append(r->dev.iop, &r->source_module);
  r->dev.iop = g_list_append(r->dev.iop, &r->module);

  return NULL;
}

/** Stand up the minimum a blend needs: a real module instance for the
    harvested operation, a dev holding the forms, and a pipe/piece pair
    carrying the blend params.

    The module has to be a genuine instance rather than a hand-filled struct.
    dt_develop_blend_process() calls through it -- self->flags() at minimum,
    and the blend colourspace is decided by the module's own
    blend_colorspace() -- so a stub would either crash (it did) or, worse,
    silently replay every edit in the wrong colour space. Loading the module
    the edit actually names is also what makes the replay faithful: an edit on
    a Lab module and one on a scene-referred RGB module take different paths
    through the blendif code. */
static const char *_replay_init(replay_t *r,
                             const char *operation,
                             const dt_develop_blend_params_t *bp,
                             GList *forms,
                             const int full_width,
                             const int full_height,
                             const int width,
                             const int height)
{
  memset(r, 0, sizeof(*r));

  r->devid = _verify_devid;

  dt_iop_module_so_t *so = _find_so(operation);
  if(!so) return "module not in this build";

  r->dev.forms = forms;
  // The mask dispatchers in masks.c take dev->history_mutex when they mutate
  // dev->forms (they race the pixelpipe's deep-copy read otherwise), and
  // migration goes through them. A zeroed dt_develop_t has an uninitialised
  // mutex, which aborts on first lock rather than failing quietly.
  //
  // It has to be RECURSIVE, exactly as dt_dev_init() creates it (develop.c):
  // these call paths do re-enter, so a default mutex does not abort -- it
  // deadlocks, which looks like the verifier hanging rather than like a bug.
  pthread_mutexattr_t recursive_locking;
  pthread_mutexattr_init(&recursive_locking);
  pthread_mutexattr_settype(&recursive_locking, PTHREAD_MUTEX_RECURSIVE);
  dt_pthread_mutex_init(&r->dev.history_mutex, &recursive_locking);
  pthread_mutexattr_destroy(&recursive_locking);
  r->dev_mutex_ready = TRUE;

  // note the sense: this returns TRUE on *failure* (see its callers in
  // imageop.c and blend.c, which all read it that way)
  if(dt_iop_load_module_by_so(&r->module, so, &r->dev))
    return "module instance could not be loaded";
  r->module_loaded = TRUE;
  r->module.dev = &r->dev;

  if(!r->module.blend_params)
  {
    dt_iop_cleanup_module(&r->module);
    memset(r, 0, sizeof(*r));
    return "module has no blend_params";
  }
  // copy into the module's own allocation rather than repointing it: the
  // module owns that buffer and frees it on cleanup
  memcpy(r->module.blend_params, bp, sizeof(dt_develop_blend_params_t));

  r->pipe.forms = forms;
  r->pipe.type = DT_DEV_PIXELPIPE_EXPORT; // never the focused GUI pipe
  // the *full* image dimensions: mask geometry is stored normalised against
  // these, so they must be the original size even though we rasterise smaller
  r->pipe.iwidth = full_width;
  r->pipe.iheight = full_height;
  // makes dt_develop_blend_process() publish its finished mask instead of
  // discarding it -- this is how the mask is recovered without touching the
  // blend code itself
  r->pipe.store_all_raster_masks = TRUE;

  r->piece.pipe = &r->pipe;
  r->piece.module = &r->module;
  r->piece.blendop_data = r->module.blend_params;
  r->piece.colors = 4;
  r->piece.enabled = TRUE;
  // Must be 1.0, not left zeroed. Radius-style parameters are converted to
  // pixels as `roi_out->scale / piece->iscale` (see the feathering call sites
  // in blend.c), so a zero iscale divides by zero and asks the guided filter
  // for an effectively infinite window -- which does not crash, it just runs
  // forever, and reads exactly like a deadlock.
  r->piece.iscale = 1.0f;
  r->piece.raster_masks =
    g_hash_table_new_full(g_direct_hash, g_direct_equal, NULL, dt_free_align_ptr);

  // Colour management, without which the whole comparison is quietly hollow.
  //
  // The per-channel branch of every blendif_*_make_mask() calls
  // dt_develop_blendif_init_masking_profile() and, if it cannot get a profile,
  // *returns leaving the mask untouched*. On a pipe with no profile that means
  // parametric masks are never evaluated at all -- and, worse, the two sides
  // fail asymmetrically: classic still carries DEVELOP_MASK_CONDITIONAL and so
  // enters that branch and bails, while a migrated edit has had CONDITIONAL
  // folded away, takes the early "not conditional" path instead, and applies
  // global opacity. The result is a clean, entirely spurious "the migration
  // changed this mask" on every parametric edit -- which is exactly what the
  // first runs reported.
  //
  // So the dev needs an iop-order list (the profile lookup asks where this
  // module sits relative to colorin/colorout) and the pipe needs real profile
  // info. Linear Rec2020 is darktable's own default working space.
  r->dev.iop_order_list = darktable.iop_order_list;
  r->module.iop_order =
    dt_ioppr_get_iop_order(r->dev.iop_order_list, r->module.op, r->module.multi_priority);

  dt_ioppr_set_pipe_work_profile_info(&r->dev, &r->pipe,
                                      DT_COLORSPACE_LIN_REC2020, "", DT_INTENT_PERCEPTUAL);
  dt_ioppr_set_pipe_output_profile_info(&r->dev, &r->pipe,
                                        DT_COLORSPACE_LIN_REC2020, "", DT_INTENT_PERCEPTUAL);

  // The INPUT profile matters too, and for a reason worth spelling out: with a
  // scene-referred blend colourspace, dt_develop_blendif_init_masking_profile()
  // asks dt_ioppr_get_pipe_current_profile_info(), which picks input / work /
  // output by comparing this module's iop_order against colorin's and
  // colorout's. Those two lookups fail on this replay's order list -- the
  // "cannot get iop-order for colorin instance 0" line -- so both come back
  // INT_MAX, every module compares as "before colorin", and the *input*
  // profile is the one actually consulted.
  //
  // Leaving it unset made the profile lookup fail, and the two blend paths
  // then diverge in a way that quietly hollowed out the GPU comparison: the
  // CPU's make_mask returns early for a non-conditional mask, before it ever
  // needs a profile, and keeps the drawn mask -- while the OpenCL kernel tests
  // `use_work_profile == 0` in its first line and returns *without writing the
  // mask at all*, leaving zeros. Both sides of a GPU comparison then came out
  // zero and agreed, which is a pass that proves nothing.
  // Set on the pipe directly rather than through
  // dt_ioppr_set_pipe_input_profile_info(): that setter consults the image
  // cache for the dev's imgid to reconcile the EXIF colourspace, and this
  // replay has no image behind it, so it dereferences a NULL image and
  // crashes. The list entry is all the profile lookup above actually needs.
  r->pipe.input_profile_info =
    dt_ioppr_add_profile_info_to_list(&r->dev, DT_COLORSPACE_LIN_REC2020, "",
                                      DT_INTENT_PERCEPTUAL);

  r->roi.x = 0;
  r->roi.y = 0;
  r->roi.width = width;
  r->roi.height = height;
  // the downscale actually applied, so radii shrink with the raster
  r->roi.scale = full_width > 0 ? (float)width / (float)full_width : 1.0f;

  // the raster fetch reads these off the target piece, both to decide whether
  // an intermediate module would have to distort the mask (equal rois: none
  // does) and to check the mask it hands back matches the requested size
  r->piece.processed_roi_in = r->roi;
  r->piece.processed_roi_out = r->roi;

  if(bp->raster_mask_source[0])
  {
    const char *raster_err = _attach_raster_source(r, bp);
    if(raster_err)
    {
      _replay_cleanup(r);
      return raster_err;
    }
  }

  r->probe = dt_masks_probe_new(width, height);
  r->modout = _make_module_output(r->probe, (size_t)width * height);
  r->out = dt_alloc_align_float((size_t)width * height * 4);
  if(!r->probe || !r->out)
  {
    _replay_cleanup(r);
    return "buffer allocation failure";
  }
  return NULL;
}

// ---------------------------------------------------------------------------
// one edit
// ---------------------------------------------------------------------------

typedef struct
{
  verify_result_t result;
  const char *skip_reason;
  gboolean inert;
  // this edit is byte-identical to an earlier one and reused its verdict
  // rather than being rendered again (see dt_masks_harvest_edit_key)
  gboolean repeat;
  double max_diff;          // CPU: classic vs migrated -- the original verdict
  double mean_diff;
  int differing_pixels;

  // GPU replay. Present only when an OpenCL device was available; `gpu_ran`
  // says whether these numbers mean anything.
  gboolean gpu_ran;
  double gpu_max_diff;      // GPU: classic vs migrated
  double gpu_mean_diff;     // ... and how large that difference actually is
  int gpu_differing_pixels;

  // the same comparisons on the rendered image, i.e. what the mask difference
  // actually did to pixels (see _diff_stats_rgb)
  gboolean image_compared, gpu_image_compared;
  double image_max_diff, image_mean_diff;
  int image_differing_pixels;
  double gpu_image_max_diff, gpu_image_mean_diff;
  int gpu_image_differing_pixels;
  // CPU-vs-GPU disagreement, measured on *both* sides of the migration.
  //
  // The after-value alone would be unreadable. The two blend implementations
  // are not bit-identical to begin with -- different math, different order,
  // the GPU running some steps in kernels the CPU does in scalar code -- so
  // some CPU/GPU gap is expected on any edit, migrated or not. What would be a
  // real defect is migration *widening* that gap. Recording the classic gap as
  // a baseline is what makes the migrated gap interpretable instead of just
  // alarming.
  double dev_diff_before;
  double dev_diff_after;

  /* The migrated gap again, with the mask post-processing switched off --
     measured only for the few edits where the gap widened past the threshold,
     which is what makes the extra pair of renders affordable.
     `nopost_ran` says whether the number means anything.

     A widened gap has two possible authors and the two numbers above cannot
     tell them apart. Either migration made the migrated pipeline itself
     inconsistent across CPU and OpenCL -- a real defect -- or a stage that runs
     *after* the mask, identically on both sides of the migration, diverges
     between CPU and OpenCL and merely got handed a slightly different input.
     Feathering is the one that matters: it is a guided filter with separate CPU
     and OpenCL implementations, and it amplifies. Re-rendering without it
     answers the question by measurement: if the gap survives, migration owns
     it; if it collapses to nothing, the post-processing does. */
  gboolean nopost_ran;
  double dev_diff_after_nopost;
} edit_report_t;

static void _verify_edit(JsonObject *edit, edit_report_t *rep)
{
  memset(rep, 0, sizeof(*rep));
  rep->result = VERIFY_SKIPPED;

  JsonObject *bo = json_object_get_object_member(edit, "blend");
  if(!bo) { rep->skip_reason = "no blend object"; return; }

  dt_develop_blend_params_t bp;
  dt_masks_harvest_read_blend_params(bo, &bp);

  // Already-migrated edits have nothing to prove: case 8's FLEXI guard makes
  // migration a no-op, so both renders would be the same call.
  if(bp.mask_mode & DEVELOP_MASK_FLEXI)
  {
    rep->skip_reason = "already flexi";
    return;
  }

  JsonObject *img = json_object_get_object_member(edit, "image");
  const int full_w = img ? (int)_obj_int(img, "width", 0) : 0;
  const int full_h = img ? (int)_obj_int(img, "height", 0) : 0;
  int w = full_w, h = full_h;
  if(w <= 0 || h <= 0) { rep->skip_reason = "no image dimensions"; return; }

  // scale down, preserving aspect (masks are normalised, so a wrong aspect
  // would distort every shape)
  if(w > VERIFY_MAX_EDGE || h > VERIFY_MAX_EDGE)
  {
    const double s = (double)VERIFY_MAX_EDGE / (double)MAX(w, h);
    w = MAX(8, (int)(w * s));
    h = MAX(8, (int)(h * s));
  }

  JsonArray *fa = json_object_has_member(edit, "forms")
    ? json_object_get_array_member(edit, "forms") : NULL;
  GList *forms = fa ? dt_masks_harvest_read_forms(fa) : NULL;
  if(fa && json_array_get_length(fa) > 0 && !forms)
  {
    rep->skip_reason = "forms could not be reconstructed";
    return;
  }

  replay_t r;
  const char *init_err =
    _replay_init(&r, _obj_str(edit, "operation", NULL), &bp, forms,
                 full_w, full_h, w, h);
  if(init_err)
  {
    rep->result = VERIFY_ERROR;
    rep->skip_reason = init_err;
    return;
  }

  const size_t npix = (size_t)w * h;

  // --- before migration -------------------------------------------------
  float *before_img = NULL, *after_img = NULL;
  float *before_cl_img = NULL, *after_cl_img = NULL;
  float *before = _render_mask(&r, &before_img);
  if(!before)
  {
    rep->result = VERIFY_ERROR;
    rep->skip_reason = "classic render produced no mask";
    _replay_cleanup(&r);
    return;
  }

  rep->inert = _is_uniform(before, npix);

  // the same classic edit on the GPU, before anything is migrated: this is the
  // baseline the post-migration CPU/GPU gap gets judged against
  float *before_cl = _render_mask_cl(&r, &before_cl_img);

  // --- migrate ----------------------------------------------------------
  if(!dt_masks_migrate_classic_to_flexi(&r.module, r.module.blend_params, -1))
  {
    rep->result = VERIFY_ERROR;
    rep->skip_reason = "migration declined";
    dt_free_align(before);
    dt_free_align(before_cl);
    dt_free_align(before_img);
    dt_free_align(before_cl_img);
    _replay_cleanup(&r);
    return;
  }

  // --- after migration --------------------------------------------------
  float *after = _render_mask(&r, &after_img);
  if(!after)
  {
    rep->result = VERIFY_ERROR;
    rep->skip_reason = "flexi render produced no mask";
    dt_free_align(before);
    dt_free_align(before_cl);
    dt_free_align(before_img);
    dt_free_align(before_cl_img);
    _replay_cleanup(&r);
    return;
  }

  float *after_cl = _render_mask_cl(&r, &after_cl_img);

  // Only meaningful when *both* GPU renders succeeded. If one side rendered
  // and the other did not, that asymmetry is itself worth reporting rather
  // than being averaged into a number, so it is counted as an error below.
  if(before_cl && after_cl && (darktable.unmuted & DT_DEBUG_MASKS))
  {
    // temporary triage dump
    float mn[4], mx[4]; double sm[4];
    const float *bufs[4] = { before, after, before_cl, after_cl };
    const char *nm[4] = { "cpu_classic", "cpu_flexi ", "gpu_classic", "gpu_flexi " };
    for(int k = 0; k < 4; k++)
    {
      mn[k] = mx[k] = bufs[k][0]; sm[k] = 0.0;
      for(size_t i = 0; i < npix; i++)
      { mn[k] = fminf(mn[k], bufs[k][i]); mx[k] = fmaxf(mx[k], bufs[k][i]); sm[k] += bufs[k][i]; }
      printf("[verify]      %s min=%.4f max=%.4f mean=%.4f\n", nm[k], mn[k], mx[k], sm[k]/npix);
    }
    printf("[verify]      mask_mode %u -> %u, mask_id %d -> %d, combine %u -> %u\n",
           bp.mask_mode, r.module.blend_params->mask_mode,
           bp.mask_id, r.module.blend_params->mask_id,
           bp.mask_combine, r.module.blend_params->mask_combine);
  }

  if(before_cl && after_cl)
  {
    rep->gpu_ran = TRUE;
    const _diff_stats_t g = _diff_stats(before_cl, after_cl, npix);
    rep->gpu_max_diff = g.max;
    rep->gpu_mean_diff = g.mean;
    rep->gpu_differing_pixels = g.differing;

    if(before_cl_img && after_cl_img)
    {
      const _diff_stats_t gi = _diff_stats_rgb(before_cl_img, after_cl_img, npix);
      rep->gpu_image_compared = TRUE;
      rep->gpu_image_max_diff = gi.max;
      rep->gpu_image_mean_diff = gi.mean;
      rep->gpu_image_differing_pixels = gi.differing;
    }
    rep->dev_diff_before = _max_abs_diff(before, before_cl, npix);
    rep->dev_diff_after = _max_abs_diff(after, after_cl, npix);

    // Only when the gap actually widened: re-render the migrated pair with the
    // mask post-processing off, to find out whether migration or a shared
    // downstream stage owns the widening (see dev_diff_after_nopost). Migration
    // leaves these fields alone -- feathering and friends stay in blend_params
    // for a migrated edit exactly as they were -- so zeroing them here disables
    // the same stages on both sides, and _render_mask commits the params afresh
    // on every call.
    if(rep->dev_diff_after - rep->dev_diff_before > VERIFY_EPS_EQUIVALENT)
    {
      dt_develop_blend_params_t *const p = r.module.blend_params;
      const float keep_feather = p->feathering_radius;
      const float keep_blur = p->blur_radius;
      const float keep_contrast = p->contrast;
      const float keep_brightness = p->brightness;
      const float keep_details = p->details;

      p->feathering_radius = 0.0f;
      p->blur_radius = 0.0f;
      p->contrast = 0.0f;
      p->brightness = 0.0f;
      p->details = 0.0f;

      float *np = _render_mask(&r, NULL);
      float *np_cl = _render_mask_cl(&r, NULL);
      if(np && np_cl)
      {
        rep->nopost_ran = TRUE;
        rep->dev_diff_after_nopost = _max_abs_diff(np, np_cl, npix);
      }
      dt_free_align(np);
      dt_free_align(np_cl);

      p->feathering_radius = keep_feather;
      p->blur_radius = keep_blur;
      p->contrast = keep_contrast;
      p->brightness = keep_brightness;
      p->details = keep_details;
    }
  }

  // --- compare ----------------------------------------------------------
  const _diff_stats_t c = _diff_stats(before, after, npix);
  const double max_d = c.max;

  if(before_img && after_img)
  {
    const _diff_stats_t ci = _diff_stats_rgb(before_img, after_img, npix);
    rep->image_compared = TRUE;
    rep->image_max_diff = ci.max;
    rep->image_mean_diff = ci.mean;
    rep->image_differing_pixels = ci.differing;
  }

  rep->max_diff = c.max;
  rep->mean_diff = c.mean;
  rep->differing_pixels = c.differing;

  if((darktable.unmuted & DT_DEBUG_MASKS) && max_d > VERIFY_EPS_EQUIVALENT)
  {
    float bmin = before[0], bmax = before[0], amin = after[0], amax = after[0];
    double bsum = 0.0, asum = 0.0;
    for(size_t i = 0; i < npix; i++)
    {
      bmin = fminf(bmin, before[i]); bmax = fmaxf(bmax, before[i]); bsum += before[i];
      amin = fminf(amin, after[i]);  amax = fmaxf(amax, after[i]);  asum += after[i];
    }
    printf("[verify]   DIFF before[min=%.4f max=%.4f mean=%.4f] "
           "after[min=%.4f max=%.4f mean=%.4f]\n",
           bmin, bmax, bsum / npix, amin, amax, asum / npix);
    printf("[verify]        mask_mode %u -> %u, blend_cst %d -> %d, "
           "opacity %.1f -> %.1f, mask_id %d -> %d, forms %d -> %d\n",
           bp.mask_mode, r.module.blend_params->mask_mode,
           bp.blend_cst, r.module.blend_params->blend_cst,
           bp.opacity, r.module.blend_params->opacity,
           bp.mask_id, r.module.blend_params->mask_id,
           g_list_length(forms), g_list_length(r.dev.forms));
  }

  // The verdict is the worst of what was actually measured. Two facts have to
  // clear the bar, not one:
  //
  //  - migration preserved the mask on the GPU as well as on the CPU
  //    (gpu_max_diff), and
  //  - migration did not *widen* the CPU/GPU gap. Judged against this edit's
  //    own classic baseline rather than against zero, because the two blend
  //    implementations already disagree slightly on unmigrated edits and
  //    calling that a migration failure would be wrong. A little headroom
  //    (one more 8-bit step) keeps ordinary kernel noise from being reported
  //    as a regression.
  double verdict_d = max_d;
  double verdict_mean = c.mean;
  if(rep->gpu_ran)
  {
    if(rep->gpu_max_diff > verdict_d)
    {
      verdict_d = rep->gpu_max_diff;
      verdict_mean = rep->gpu_mean_diff;
    }
    // The widening is a max-vs-max quantity: it compares two worst-pixel gaps,
    // so there is no mean that belongs with it. It therefore only ever raises
    // the max side of the test, and is left out of the mean side rather than
    // paired with a number measuring something else.
    const double widened = rep->dev_diff_after - rep->dev_diff_before;
    if(widened > VERIFY_EPS_EQUIVALENT) verdict_d = MAX(verdict_d, widened);
  }

  // `deltae`'s rule: over tolerance on the worst pixel, or over a third of it
  // on average, is a real difference; well under it everywhere is identical.
  if(verdict_d <= VERIFY_EPS_IDENTICAL) rep->result = VERIFY_IDENTICAL;
  else if(verdict_d <= VERIFY_EPS_EQUIVALENT
          && verdict_mean <= VERIFY_EPS_EQUIVALENT_MEAN) rep->result = VERIFY_EQUIVALENT;
  else rep->result = VERIFY_DIFFERENT;

  // one GPU render succeeding while the other failed is a real asymmetry --
  // exactly the shape the NO_MASKS bug had -- so it must not pass quietly
  if(_verify_devid >= 0 && !rep->gpu_ran && (before_cl || after_cl))
  {
    rep->result = VERIFY_ERROR;
    rep->skip_reason = before_cl ? "GPU rendered classic but not migrated"
                                 : "GPU rendered migrated but not classic";
  }

  dt_free_align(before);
  dt_free_align(after);
  dt_free_align(before_cl);
  dt_free_align(after_cl);
  dt_free_align(before_img);
  dt_free_align(after_img);
  dt_free_align(before_cl_img);
  dt_free_align(after_cl_img);
  _replay_cleanup(&r);
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------

static const char *_result_name(const verify_result_t r)
{
  switch(r)
  {
    case VERIFY_IDENTICAL:  return "identical";
    case VERIFY_EQUIVALENT: return "equivalent";
    case VERIFY_DIFFERENT:  return "DIFFERENT";
    case VERIFY_SKIPPED:    return "skipped";
    default:                return "ERROR";
  }
}

gboolean dt_masks_verify_harvest_section(const char *json_path, FILE *rf)
{
  // line-buffered: a crash mid-replay must not swallow the progress output
  // that says which edit it was on
  setvbuf(stdout, NULL, _IOLBF, 0);

#ifdef _OPENMP
  // Single-threaded, deliberately, and not for safety -- for reproducibility.
  //
  // Several stages of the blend reduce over pixels in parallel, so the order
  // of float additions depends on thread scheduling and the mask comes out
  // differing in the last bits from one run to the next. Measured across two
  // full 2466-edit runs, 4 edits changed verdict between them purely from
  // that: three by less than 0.004, but one by 0.1, which is far above any
  // threshold worth setting and looked exactly like a real migration bug.
  // Forced to one thread, all four are identical and stable across repeated
  // runs.
  //
  // A verifier whose answer moves between runs cannot be used to investigate
  // anything, so the cost (a slower pass) buys the only property that makes
  // the output actionable. This is also why the tolerance below stays tight:
  // with the nondeterminism removed there is no float noise left to absorb.
  omp_set_num_threads(1);
#endif

#ifdef HAVE_OPENCL
  // One device for the whole run. DT_DEV_PIXELPIPE_EXPORT matches the pipe
  // type the replay declares, so the device priority preferences resolve the
  // same way an export would.
  if(darktable.opencl && darktable.opencl->inited)
    _verify_devid = dt_opencl_lock_device(DT_DEV_PIXELPIPE_EXPORT);
  if(_verify_devid >= 0)
    printf("[verify] OpenCL device %d acquired: GPU blend will be replayed too\n",
           _verify_devid);
  else
    printf("[verify] no OpenCL device: CPU blend only\n");
#else
  printf("[verify] built without OpenCL: CPU blend only\n");
#endif

  GError *err = NULL;
  // accepts the .gz the contributor actually sent, as well as a plain file
  JsonParser *parser = dt_masks_harvest_load(json_path, &err);
  if(!parser)
  {
    fprintf(stderr, "[verify] cannot read %s: %s\n",
            json_path, err ? err->message : "unknown error");
    g_clear_error(&err);
    return FALSE;
  }

  JsonNode *root = json_parser_get_root(parser);
  JsonObject *ro = root ? json_node_get_object(root) : NULL;
  JsonArray *edits = ro && json_object_has_member(ro, "edits")
    ? json_object_get_array_member(ro, "edits") : NULL;
  if(!edits)
  {
    fprintf(stderr, "[verify] %s has no \"edits\" array\n", json_path);
    g_object_unref(parser);
    return FALSE;
  }

  verify_stats_t st;
  memset(&st, 0, sizeof(st));
  st.worst_index = -1;
  st.worst_gpu_index = -1;
  st.worst_image_index = -1;
  st.worst_gpu_image_index = -1;

  if(rf) fprintf(rf, "\n  \"source\": \"%s\",\n  \"edits\": [", json_path);
  gboolean first_report = TRUE;

  const guint n = json_array_get_length(edits);
  printf("[verify] replaying %u harvested edits from %s\n", n, json_path);

  /* Exact repeats are rendered once (see dt_masks_harvest_edit_key). The
     verdict is stored against the edit's content key and reused, so every
     occurrence is still counted, reported and aggregated exactly as if it had
     been replayed -- only the four renders are skipped. */
  GHashTable *seen =
    g_hash_table_new_full(g_str_hash, g_str_equal, g_free, g_free);
  int replayed_unique = 0;

  for(guint i = 0; i < n; i++)
  {
    JsonObject *edit = json_array_get_object_element(edits, i);
    if(!edit) continue;

    edit_report_t rep;
    gchar *key = dt_masks_harvest_edit_key(edit);
    const edit_report_t *cached = key ? g_hash_table_lookup(seen, key) : NULL;
    if(cached)
    {
      rep = *cached;
      rep.repeat = TRUE;
      g_free(key);
    }
    else
    {
      // -d masks names each edit before replaying it: when one of these wedges
      // or crashes, the last line printed is the only thing that says which
      // configuration did it.
      if(darktable.unmuted & DT_DEBUG_MASKS)
        printf("[verify] edit %u op=%s mask_mode=%d\n", i,
               _obj_str(edit, "operation", "?"),
               (int)_obj_int(json_object_get_object_member(edit, "blend"),
                             "mask_mode", -1));
      _verify_edit(edit, &rep);
      rep.repeat = FALSE;
      replayed_unique++;
      if(key)
      {
        edit_report_t *store = malloc(sizeof(edit_report_t));
        if(store) { *store = rep; g_hash_table_insert(seen, key, store); }
        else g_free(key);
      }
    }

    st.total++;
    switch(rep.result)
    {
      case VERIFY_IDENTICAL:  st.identical++; break;
      case VERIFY_EQUIVALENT: st.equivalent++; break;
      case VERIFY_DIFFERENT:  st.different++; break;
      case VERIFY_SKIPPED:    st.skipped++; break;
      default:                st.error++; break;
    }

    if(rep.result == VERIFY_IDENTICAL || rep.result == VERIFY_EQUIVALENT
       || rep.result == VERIFY_DIFFERENT)
    {
      if(rep.inert) st.inert_before++;
      else
      {
        st.live++;
        if(rep.result == VERIFY_IDENTICAL) st.live_identical++;
        else if(rep.result == VERIFY_EQUIVALENT) st.live_equivalent++;
        else st.live_different++;
      }
      if(rep.max_diff > st.worst_max_diff)
      {
        st.worst_max_diff = rep.max_diff;
        st.worst_index = (int)i;
        st.worst_mean_diff = rep.mean_diff;
        st.worst_differing_pixels = rep.differing_pixels;
      }
      if(rep.image_compared)
      {
        st.image_compared++;
        if(rep.image_max_diff > st.worst_image_diff)
        {
          st.worst_image_diff = rep.image_max_diff;
          st.worst_image_mean_diff = rep.image_mean_diff;
          st.worst_image_differing_pixels = rep.image_differing_pixels;
          st.worst_image_index = (int)i;
        }
      }
      if(rep.gpu_image_compared)
      {
        st.gpu_image_compared++;
        if(rep.gpu_image_max_diff > st.worst_gpu_image_diff)
        {
          st.worst_gpu_image_diff = rep.gpu_image_max_diff;
          st.worst_gpu_image_mean_diff = rep.gpu_image_mean_diff;
          st.worst_gpu_image_differing_pixels = rep.gpu_image_differing_pixels;
          st.worst_gpu_image_index = (int)i;
        }
      }
      if(rep.gpu_ran)
      {
        st.gpu_compared++;
        if(rep.gpu_max_diff > st.worst_gpu_diff)
        {
          st.worst_gpu_diff = rep.gpu_max_diff;
          st.worst_gpu_index = (int)i;
          st.worst_gpu_mean_diff = rep.gpu_mean_diff;
          st.worst_gpu_differing_pixels = rep.gpu_differing_pixels;
        }
        st.worst_dev_before = MAX(st.worst_dev_before, rep.dev_diff_before);
        st.worst_dev_after = MAX(st.worst_dev_after, rep.dev_diff_after);
        if(rep.dev_diff_after - rep.dev_diff_before > VERIFY_EPS_EQUIVALENT)
        {
          st.dev_gap_widened++;
          // the migrated pipeline disagreeing with itself once nothing runs
          // after the mask is migration's own inconsistency; a widening that
          // vanishes here was amplification by a stage classic runs too
          if(!rep.nopost_ran || rep.dev_diff_after_nopost > VERIFY_EPS_EQUIVALENT)
            st.dev_gap_widened_own++;
        }
      }
    }

    // skipped edits are written too: the report has to account for every edit
    // in the harvest, or reading it means reconciling it against the terminal
    // output to find out what happened to the missing indices
    if(rf)
    {
      fprintf(rf, "%s\n    {\"index\": %u, \"operation\": \"%s\", \"result\": \"%s\","
                  " \"inert\": %s, \"max_diff\": %.9g, \"mean_diff\": %.9g,"
                  " \"differing_pixels\": %d, \"gpu_ran\": %s,"
                  " \"gpu_max_diff\": %.9g, \"gpu_mean_diff\": %.9g,"
                  " \"gpu_differing_pixels\": %d,"
                              " \"repeat\": %s, \"image_compared\": %s, \"image_max_diff\": %.9g,"
                  " \"image_mean_diff\": %.9g, \"image_differing_pixels\": %d,"
                  " \"gpu_image_compared\": %s, \"gpu_image_max_diff\": %.9g,"
                  " \"gpu_image_mean_diff\": %.9g,"
                  " \"gpu_image_differing_pixels\": %d,"
                  " \"dev_diff_before\": %.9g,"
                  " \"dev_diff_after\": %.9g,"
                  " \"nopost_ran\": %s, \"dev_diff_after_nopost\": %.9g%s%s%s}",
              first_report ? "" : ",", i,
              _obj_str(edit, "operation", "?"),
              _result_name(rep.result),
              rep.inert ? "true" : "false",
              rep.max_diff, rep.mean_diff, rep.differing_pixels,
              rep.gpu_ran ? "true" : "false",
              rep.gpu_max_diff, rep.gpu_mean_diff, rep.gpu_differing_pixels,
              rep.repeat ? "true" : "false",
              rep.image_compared ? "true" : "false",
              rep.image_max_diff, rep.image_mean_diff, rep.image_differing_pixels,
              rep.gpu_image_compared ? "true" : "false",
              rep.gpu_image_max_diff, rep.gpu_image_mean_diff,
              rep.gpu_image_differing_pixels,
              rep.dev_diff_before, rep.dev_diff_after,
              rep.nopost_ran ? "true" : "false", rep.dev_diff_after_nopost,
              rep.skip_reason ? ", \"reason\": \"" : "",
              rep.skip_reason ? rep.skip_reason : "",
              rep.skip_reason ? "\"" : "");
      first_report = FALSE;
    }

    if((i + 1) % 250 == 0)
      printf("[verify]   %u/%u ...\n", i + 1, n);
  }

  g_object_unref(parser);

  const gboolean passed = st.different == 0 && st.error == 0;

  // Every number the summary below prints also goes into the report, so the
  // file is self-contained: reading a run must not require having kept the
  // terminal output that went with it.
  if(rf)
  {
    fputs("\n  ],\n  \"summary\": {\n", rf);
    fprintf(rf, "    \"passed\": %s,\n", passed ? "true" : "false");
    fprintf(rf, "    \"harvested\": %u,\n", n);
    fprintf(rf, "    \"replayed\": %d,\n", st.total);
    fprintf(rf, "    \"distinct_edits\": %d,\n", replayed_unique);
    fprintf(rf, "    \"identical\": %d,\n", st.identical);
    fprintf(rf, "    \"equivalent\": %d,\n", st.equivalent);
    fprintf(rf, "    \"different\": %d,\n", st.different);
    fprintf(rf, "    \"skipped\": %d,\n", st.skipped);
    fprintf(rf, "    \"errors\": %d,\n", st.error);
    fprintf(rf, "    \"live\": %d,\n", st.live);
    fprintf(rf, "    \"live_identical\": %d,\n", st.live_identical);
    fprintf(rf, "    \"live_equivalent\": %d,\n", st.live_equivalent);
    fprintf(rf, "    \"live_different\": %d,\n", st.live_different);
    fprintf(rf, "    \"inert\": %d,\n", st.inert_before);
    fprintf(rf, "    \"worst_cpu_diff\": %.9g,\n", st.worst_max_diff);
    fprintf(rf, "    \"worst_cpu_diff_index\": %d,\n", st.worst_index);
    fprintf(rf, "    \"worst_cpu_mean_diff\": %.9g,\n", st.worst_mean_diff);
    fprintf(rf, "    \"worst_cpu_differing_pixels\": %d,\n", st.worst_differing_pixels);
    fprintf(rf, "    \"gpu_compared\": %d,\n", st.gpu_compared);
    fprintf(rf, "    \"worst_gpu_diff\": %.9g,\n", st.worst_gpu_diff);
    fprintf(rf, "    \"worst_gpu_diff_index\": %d,\n", st.worst_gpu_index);
    fprintf(rf, "    \"worst_gpu_mean_diff\": %.9g,\n", st.worst_gpu_mean_diff);
    fprintf(rf, "    \"worst_gpu_differing_pixels\": %d,\n", st.worst_gpu_differing_pixels);
    fprintf(rf, "    \"image_compared\": %d,\n", st.image_compared);
    fprintf(rf, "    \"worst_image_diff\": %.9g,\n", st.worst_image_diff);
    fprintf(rf, "    \"worst_image_diff_index\": %d,\n", st.worst_image_index);
    fprintf(rf, "    \"worst_image_mean_diff\": %.9g,\n", st.worst_image_mean_diff);
    fprintf(rf, "    \"worst_image_differing_pixels\": %d,\n",
            st.worst_image_differing_pixels);
    fprintf(rf, "    \"gpu_image_compared\": %d,\n", st.gpu_image_compared);
    fprintf(rf, "    \"worst_gpu_image_diff\": %.9g,\n", st.worst_gpu_image_diff);
    fprintf(rf, "    \"worst_gpu_image_diff_index\": %d,\n", st.worst_gpu_image_index);
    fprintf(rf, "    \"worst_gpu_image_mean_diff\": %.9g,\n",
            st.worst_gpu_image_mean_diff);
    fprintf(rf, "    \"worst_gpu_image_differing_pixels\": %d,\n",
            st.worst_gpu_image_differing_pixels);
    fprintf(rf, "    \"worst_dev_gap_classic\": %.9g,\n", st.worst_dev_before);
    fprintf(rf, "    \"worst_dev_gap_migrated\": %.9g,\n", st.worst_dev_after);
    fprintf(rf, "    \"dev_gap_widened\": %d,\n", st.dev_gap_widened);
    fprintf(rf, "    \"dev_gap_widened_own\": %d\n", st.dev_gap_widened_own);
    fputs("  }", rf);
  }

  g_hash_table_destroy(seen);

  printf("[verify]\n");
  printf("[verify] replayed          : %d  (%d distinct, %d exact repeats reused)\n",
         st.total, replayed_unique, st.total - replayed_unique);
  printf("[verify]   identical       : %d\n", st.identical);
  printf("[verify]   equivalent      : %d"
         "  (worst pixel below 1/255 and mean below 1/765, invisible)\n",
         st.equivalent);
  printf("[verify]   DIFFERENT       : %d\n", st.different);
  printf("[verify]   skipped         : %d\n", st.skipped);
  printf("[verify]   errors          : %d\n", st.error);
  printf("[verify]\n");
  // The distinction that decides what the run is worth: a comparison between
  // two uniform masks would have passed no matter what migration did.
  printf("[verify] of those actually compared:\n");
  printf("[verify]   live (mask varies)   : %d   -> %d identical, %d equivalent, %d different\n",
         st.live, st.live_identical, st.live_equivalent, st.live_different);
  printf("[verify]   inert (uniform mask) : %d   (proves nothing either way)\n",
         st.inert_before);
  if(st.worst_index >= 0)
    printf("[verify] worst CPU difference: %.9g at edit %d"
           " (mean %.9g over %d differing pixels)\n",
           st.worst_max_diff, st.worst_index,
           st.worst_mean_diff, st.worst_differing_pixels);

  if(st.image_compared && st.worst_image_index >= 0)
    printf("[verify] worst image difference: %.9g at edit %d"
           " (mean %.9g over %d differing pixels)\n",
           st.worst_image_diff, st.worst_image_index,
           st.worst_image_mean_diff, st.worst_image_differing_pixels);

  if(st.gpu_compared)
  {
    printf("[verify]\n");
    printf("[verify] GPU (OpenCL blend), %d edits replayed on both paths:\n",
           st.gpu_compared);
    printf("[verify]   migration on GPU, worst difference : %.9g at edit %d"
           " (mean %.9g over %d differing pixels)\n",
           st.worst_gpu_diff, st.worst_gpu_index,
           st.worst_gpu_mean_diff, st.worst_gpu_differing_pixels);
    // Reported side by side on purpose. The absolute CPU/GPU gap is not a
    // defect -- two separate implementations of the same blend never agree to
    // the last bit -- so the number that matters is whether migration made it
    // worse, not how large it is.
    printf("[verify]   CPU vs GPU gap, classic  : %.9g  (pre-existing baseline)\n",
           st.worst_dev_before);
    printf("[verify]   CPU vs GPU gap, migrated : %.9g\n", st.worst_dev_after);
    printf("[verify]   edits where migration widened that gap by >1/255 : %d\n",
           st.dev_gap_widened);
    if(st.dev_gap_widened)
      printf("[verify]     of those, still widened with mask post-processing off"
             " (migration's own) : %d\n", st.dev_gap_widened_own);
  }
  else
    printf("[verify] GPU: not replayed (no OpenCL device)\n");

#ifdef HAVE_OPENCL
  if(_verify_devid >= 0)
  {
    dt_opencl_unlock_device(_verify_devid);
    _verify_devid = -1;
  }
#endif

  return passed;
}

gboolean dt_masks_verify_harvest(const char *json_path, const char *report_path)
{
  FILE *rf = report_path ? g_fopen(report_path, "wb") : NULL;
  if(rf) fputs("{", rf);
  const gboolean ok = dt_masks_verify_harvest_section(json_path, rf);
  if(rf)
  {
    fputs("\n}\n", rf);
    fclose(rf);
    printf("[verify] per-edit report written to %s\n", report_path);
  }
  return ok;
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
