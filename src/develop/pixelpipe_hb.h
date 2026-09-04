/*
    This file is part of darktable,
    Copyright (C) 2009-2026 darktable developers.

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

#include "common/atomic.h"
#include "common/image.h"
#include "common/iop_order.h"
#include "control/conf.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/pixelpipe_cache.h"
#include "imageio/imageio_common.h"

G_BEGIN_DECLS

#define DT_PIPECACHE_MIN 2

/** cached distorted mask at a geometric module's output boundary.
 *  used to avoid re-distorting masks from scratch when multiple
 *  downstream modules request the same mask type. */
typedef struct dt_dev_distorted_mask_cache_t
{
  float *data;      // the cached distorted mask at this piece's output
  size_t size;      // allocated size of data, accounted in pipe->mask_cache_size
  dt_iop_roi_t roi; // the roi this mask corresponds to (piece->processed_roi_out)
  dt_hash_t hash;     // hash of pipe/geometry state for invalidation
  dt_hash_t src_hash; // hash of source data (e.g. threshold) for invalidation
} dt_dev_distorted_mask_cache_t;

/** pipe-local snapshot of the flexi mask refinement-bypass preview state.
 *
 *  The live state is a GHashTable owned by the module's GUI
 *  (dt_iop_gui_blend_data_t.masks_refine_bypassed) and mutated on the GTK
 *  thread. Pixelpipe worker threads must not touch it, so it is copied here
 *  by dt_masks_refine_bypass_commit() during commit_params. Keys are built
 *  with dt_masks_refine_key_*() (develop/blend.h); the array is sorted so
 *  lookups can bisect and the hash is order-independent. */
typedef struct dt_dev_refine_bypass_t
{
  guint32 *keys;  // sorted bypass keys, or NULL when nothing is bypassed
  int nkeys;
} dt_dev_refine_bypass_t;

typedef struct dt_dev_pixelpipe_iop_t
{
  struct dt_iop_module_t *module;  // the module in the dev operation stack
  struct dt_dev_pixelpipe_t *pipe; // the pipe this piece belongs to
  void *data;                      // to be used by the module to store stuff per pipe piece
  void *blendop_data;              // to be used by the module to store blendop per pipe piece
  gboolean enabled; // used to disable parts of the pipe for export, independent on module itself.

  dt_dev_request_flags_t request_histogram;              // (bitwise) set if you want an histogram captured
  dt_dev_histogram_collection_params_t histogram_params; // set histogram generation params
  uint32_t *histogram; // pointer to histogram data; histogram_bins_count bins with 4 channels each
  dt_dev_histogram_stats_t histogram_stats; // stats of captured histogram
  uint32_t histogram_max[4];                // maximum levels in histogram, one per channel

  float iscale;                   // input actually just downscaled buffer? iscale*iwidth = actual width
  int iwidth, iheight;            // width and height of input buffer
  dt_hash_t hash;                 // hash of params and enabled.
  int bpc;                        // bits per channel, 32 means float
  int colors;                     // how many colors per pixel
  dt_iop_roi_t buf_in;            // theoretical full buffer regions of interest, as passed through modify_roi_out
  dt_iop_roi_t buf_out;
  dt_iop_roi_t processed_roi_in;  // the actual roi that was used for processing the piece
  dt_iop_roi_t processed_roi_out;
  gboolean process_cl_ready;      // set this to FALSE in commit_params to temporarily disable the use of process_cl
  gboolean process_tiling_ready;  // set this to FALSE in commit_params to temporarily disable tiling

  // the following are used internally for caching:
  dt_iop_buffer_dsc_t dsc_in;
  dt_iop_buffer_dsc_t dsc_out;
  uint8_t xtrans[6][6];
  uint32_t filters;
  GHashTable *raster_masks;

  // cached distorted masks at geometric module boundaries
  dt_dev_distorted_mask_cache_t detail_mask_cache;
  dt_dev_distorted_mask_cache_t raster_mask_cache;
  // cached output of dt_masks_group_render_roi (the rasterized drawn mask,
  // before global post-ops/invert). hash = dt_masks_group_hash + roi_out;
  // src_hash = pipe->scharr.hash (covers per-shape details refinement).
  // only populated when the group needs no host guides (no guided-filter
  // feathering / parametric member), whose result depends on module pixels.
  dt_dev_distorted_mask_cache_t drawn_mask_cache;

  // transient guides used by optional per-shape mask refinement (feathering)
  // inside the group renderer. They point into the module in/out buffers and
  // are only valid for the duration of dt_develop_blend_process; NULL when no
  // shape in the group requests refinement.
  const float *blend_refine_guide_in;
  const float *blend_refine_guide_out;
  const dt_iop_roi_t *blend_refine_roi_in;
  const dt_iop_roi_t *blend_refine_roi_out;

  // GUI-owned refinement bypass state, snapshotted at commit time so the
  // renderer never reads live GTK data from a worker thread
  dt_dev_refine_bypass_t refine_bypass;
} dt_dev_pixelpipe_iop_t;

typedef enum dt_dev_pixelpipe_change_t
{
  DT_DEV_PIPE_UNCHANGED   = 0,      // no event
  DT_DEV_PIPE_TOP_CHANGED = 1 << 0, // only params of top element changed
  DT_DEV_PIPE_REMOVE      = 1 << 1, // possibly elements of the pipe have to be removed
  DT_DEV_PIPE_SYNCH       = 1 << 2, // all nodes up to end need to be synched,
                                    // but no removal of module pieces is necessary
  DT_DEV_PIPE_ZOOMED      = 1 << 3  // zoom event, preview pipe does not need changes
} dt_dev_pixelpipe_change_t;

typedef enum dt_dev_pixelpipe_status_t
{
  DT_DEV_PIXELPIPE_DIRTY = 0,   // history stack changed or image new
  DT_DEV_PIXELPIPE_RUNNING = 1, // pixelpipe is running
  DT_DEV_PIXELPIPE_VALID = 2,   // pixelpipe has finished; valid result
  DT_DEV_PIXELPIPE_INVALID = 3  // pixelpipe has finished; invalid result
} dt_dev_pixelpipe_status_t;

/* dt_dev_pixelpipe_stopper_t is used for shutdown in dt_dev_pixelpipe_t.
    By design we can write atomically on a pipe->shutdown to request an early exit
    of the pixepipe process _dev_pixelpipe_process_rec().

    When setting pipe->shutdown we should use dt_dev_pixelpipe_set_shutdown(),
    it uses dt_atomic_CAS_int so having only one shutdown request per pipe run.
    The "expected" state is DT_DEV_PIXELPIPE_PROCESSING, in other states the
    set_shutdown() request is ignored.
    As we never have valid output data we always invalidate output cacheline.

    A summary about how these shutdown modes are supposed to work:

    DT_DEV_PIXELPIPE_STOP_NO
    Set whenever we initialize or clean up the pipe, means "pipe is idle"

    DT_DEV_PIXELPIPE_PROCESSING
    Set whenever a pipe is started making this different from idling mode STOP_NO.
    Please note there is a very small timelap after the pipe thread has started.

    DT_DEV_PIXELPIPE_STOP_NODES
    Set if the pipe should stop as the pipe nodes are changed.

    DT_DEV_PIXELPIPE_STOP_HQ
    Used to switch between darkroom HQ modes.

    DT_DEV_PIXELPIPE_STOP_ZOOM
    A request to restart with different darkroom position or scale.
    We might get back to last zoom setting pretty soon so we keep cachlines
    as we do for above shutdown modes.

    DT_DEV_PIXELPIPE_STOP_DATA
    A request to restart with different module parameters,
    writing back input cl_mem to host for a faster restart if possible.

    DT_DEV_PIXELPIPE_STOP_PIECE
    A module has stopped within the piece process() variants.
    As we missed processing the correct output and all following modules
    will give different results accordingly we clear cachelines for following
    modules (possibly writing back input cl_mem to host for a faster restart).
*/

typedef enum dt_dev_pixelpipe_stopper_t
{
  DT_DEV_PIXELPIPE_STOP_NO = 0,
  DT_DEV_PIXELPIPE_PROCESSING,
  DT_DEV_PIXELPIPE_STOP_NODES,
  DT_DEV_PIXELPIPE_STOP_HQ,
  DT_DEV_PIXELPIPE_STOP_ZOOM,
  DT_DEV_PIXELPIPE_STOP_DATA,
  DT_DEV_PIXELPIPE_STOP_PIECE,
} dt_dev_pixelpipe_stopper_t;

typedef struct dt_dev_detail_mask_t
{
  dt_iop_roi_t roi;
  dt_hash_t hash;
  float *data;
  size_t size;
} dt_dev_detail_mask_t;

/**
 * this encapsulates the pixelpipe.
 * a develop module will need several of these:
 * for previews and full blits to cairo and for
 * the export function.
 */
typedef struct dt_dev_pixelpipe_t
{
  // store history/zoom caches
  dt_dev_pixelpipe_cache_t cache;
  // set to an iop_order to invalidate cachelines >= given order before next pixelpipe run
  uint32_t cache_obsolete_order;
  uint64_t runs; // used only for pixelpipe cache statistics
  // input buffer
  float *input;
  // width and height of input buffer
  int iwidth, iheight;
  // input actually just downscaled buffer? iscale*iwidth = actual width
  float iscale;
  // dimensions of processed buffer
  int processed_width, processed_height;

  // this one actually contains the expected output format,
  // and should be modified by process*(), if necessary.
  dt_iop_buffer_dsc_t dsc;

  /** work profile info of the image */
  struct dt_iop_order_iccprofile_info_t *work_profile_info;
  /** input profile info **/
  struct dt_iop_order_iccprofile_info_t *input_profile_info;
  /** output profile info **/
  struct dt_iop_order_iccprofile_info_t *output_profile_info;
  /** used only as a cache-identity tag to invalidate the cache **/
  struct dt_iop_order_iccprofile_info_t *export_profile_info;

  // instances of pixelpipe, stored in GList of dt_dev_pixelpipe_iop_t
  GList *nodes;
  // event flag
  dt_dev_pixelpipe_change_t changed;
  // pipe status
  dt_dev_pixelpipe_status_t status;
  gboolean loading;
  gboolean input_changed;
  // backbuffer (output)
  uint8_t *backbuf;
  size_t backbuf_size;
  int backbuf_width, backbuf_height;
  float backbuf_scale;
  dt_dev_zoom_pos_t backbuf_zoom_pos;
  dt_hash_t backbuf_hash;
  dt_pthread_mutex_t mutex, backbuf_mutex, busy_mutex;
  int final_width, final_height;

  // the data for the luminance mask are kept in a buffer written by demosaic or rawprepare
  // as we have to scale the mask later we keep size at that stage
  gboolean want_detail_mask;
  // set only while dt_dev_pixelpipe_synch_all replays history: suppresses the
  // per-module usedetails order-0 flush; synch_all invalidates once at the end
  // and only if the detail requirement actually toggled.
  gboolean synch_no_detail_invalidate;
  struct dt_dev_detail_mask_t scharr;

  // avoid cached data for processed module
  gboolean nocache;

  dt_imgid_t output_imgid;
  /* Testing for shutting down and a running pixelpipe
     can be used in various ways defined in dt_dev_pixelpipe_stopper_t, in all cases the
       running pipe is stopped asap
     If we don't use one of the enum values this is interpreted as the iop_order of the module
     that has set this in case of an error condition or other reasons that request a re-run of the pipe.
     In those cases we assume cachelines after this module and the input of the stopper module
     are not valid cachelines any more so the pixelpipe takes care of this.
  */
  dt_atomic_int shutdown;
  // opencl enabled for this pixelpipe?
  gboolean opencl_enabled;
  // opencl error detected?
  gboolean opencl_error;
  // running in a tiling context?
  gboolean tiling;
  // should this pixelpipe display a mask in the end?
  dt_dev_pixelpipe_display_mask_t mask_display;
  // should this pixelpipe completely suppressed the blendif module?
  gboolean bypass_blendif;
  // input data based on this timestamp:
  int input_timestamp;
  uint32_t average_delay;
  dt_dev_pixelpipe_type_t type;
  // the final output pixel format this pixelpipe will be converted to
  dt_imageio_levels_t levels;
  // opencl device that has been locked for this pipe.
  int devid;
  // image struct as it was when the pixelpipe was initialized. copied to avoid race conditions.
  dt_image_t image;
  // the user might choose to overwrite the output color space and rendering intent.
  dt_colorspaces_color_profile_type_t icc_type;
  gchar *icc_filename;
  dt_iop_color_intent_t icc_intent;
  // snapshot of modules
  GList *iop;
  // snapshot of modules iop_order
  GList *iop_order_list;
  // snapshot of mask list
  GList *forms;
  // the masks generated in the pipe for later reusal are inside dt_dev_pixelpipe_iop_t
  gboolean store_all_raster_masks;
  // module blending cache
  float *bcache_data;
  dt_hash_t bcache_hash;
  size_t bcache_size;

  // reusable ping-pong buffers for mask distortion walks
  float *mask_distort_buf[2];
  size_t mask_distort_buf_size[2];
  // sum of all per-piece detail/raster mask caches currently allocated in this pipe
  size_t mask_cache_size;
} dt_dev_pixelpipe_t;

struct dt_develop_t;

static inline gboolean dt_pipe_is_fast(const dt_dev_pixelpipe_t *pipe)
{
  return (pipe->type & DT_DEV_PIXELPIPE_FAST);
}
static inline gboolean dt_pipe_is_full(const dt_dev_pixelpipe_t *pipe)
{
  return (pipe->type & DT_DEV_PIXELPIPE_FULL);
}
static inline gboolean dt_pipe_is_thumb(const dt_dev_pixelpipe_t *pipe)
{
  return (pipe->type & DT_DEV_PIXELPIPE_THUMBNAIL);
}
static inline gboolean dt_pipe_is_export(const dt_dev_pixelpipe_t *pipe)
{
  return (pipe->type & DT_DEV_PIXELPIPE_EXPORT);
}
static inline gboolean dt_pipe_is_basic(const dt_dev_pixelpipe_t *pipe)
{
  return (pipe->type & DT_DEV_PIXELPIPE_BASIC);
}
static inline gboolean dt_pipe_is_canvas(const dt_dev_pixelpipe_t *pipe)
{
  return (pipe->type & DT_DEV_PIXELPIPE_CANVAS);
}
static inline gboolean dt_pipe_is_preview(const dt_dev_pixelpipe_t *pipe)
{
  return (pipe->type & DT_DEV_PIXELPIPE_PREVIEW);
}
static inline gboolean dt_pipe_is_preview2(const dt_dev_pixelpipe_t *pipe)
{
  return (pipe->type & DT_DEV_PIXELPIPE_PREVIEW2);
}
static inline gboolean dt_pipe_is_screen(const dt_dev_pixelpipe_t *pipe)
{
  return (pipe->type & DT_DEV_PIXELPIPE_SCREEN);
}
static inline gboolean dt_pipe_is_image(const dt_dev_pixelpipe_t *pipe)
{
  return (pipe->type & DT_DEV_PIXELPIPE_IMAGE);
}
static inline gboolean dt_pipe_is_image_final(const dt_dev_pixelpipe_t *pipe)
{
  return (pipe->type & DT_DEV_PIXELPIPE_IMAGE_FINAL);
}
static inline gboolean dt_pipe_no_mask_display(const dt_dev_pixelpipe_t *pipe)
{
  return pipe->mask_display == DT_DEV_PIXELPIPE_DISPLAY_NONE;
}
static inline gboolean dt_pipe_mask_display(const dt_dev_pixelpipe_t *pipe)
{
  return pipe->mask_display != DT_DEV_PIXELPIPE_DISPLAY_NONE;
}
static inline gboolean dt_pipe_processing(dt_dev_pixelpipe_t *pipe)
{
  return dt_atomic_get_int(&pipe->shutdown) == DT_DEV_PIXELPIPE_PROCESSING;
}
static inline gboolean dt_pipe_started(dt_dev_pixelpipe_t *pipe)
{
  return dt_atomic_get_int(&pipe->shutdown) >= DT_DEV_PIXELPIPE_PROCESSING;
}

// report pipe->type as textual string
const char *dt_dev_pixelpipe_type_to_str(const dt_dev_pixelpipe_type_t pipe_type);
// return pipe->shutdown as textual
const char *dt_dev_pixelpipe_shutdown_to_str(const dt_dev_pixelpipe_stopper_t stopper);

// sets pipe->shutdown in atomic CAS mode so only one mode is possible per pipe run
void dt_dev_pixelpipe_set_shutdown(dt_dev_pixelpipe_t *pipe, const dt_dev_pixelpipe_stopper_t stopper);
// Is there a pending shutdown request for the piece's pipe?
gboolean dt_dev_piece_shutdown(dt_dev_pixelpipe_iop_t *piece, const gboolean test);

// inits the pixelpipe with plain passthrough input/output and empty input and default caching settings.
gboolean dt_dev_pixelpipe_init(dt_dev_pixelpipe_t *pipe);
// inits the preview pixelpipe with plain passthrough input/output and empty input and default caching
// settings.
gboolean dt_dev_pixelpipe_init_preview(dt_dev_pixelpipe_t *pipe);
gboolean dt_dev_pixelpipe_init_preview2(dt_dev_pixelpipe_t *pipe);
// inits the pixelpipe with settings optimized for full-image export
// (no history stack cache)
gboolean dt_dev_pixelpipe_init_export(dt_dev_pixelpipe_t *pipe,
                                      const int32_t width,
                                      const int32_t height,
                                      const int levels,
                                      const gboolean store_masks);
// inits the pixelpipe with settings optimized for thumbnail export
// (no history stack cache)
gboolean dt_dev_pixelpipe_init_thumbnail(dt_dev_pixelpipe_t *pipe,
                                         const int32_t width,
                                         const int32_t height);
// inits all but the pixel caches, so you can't actually process an
// image (just get dimensions and distortions)
gboolean dt_dev_pixelpipe_init_dummy(dt_dev_pixelpipe_t *pipe,
                                     const int32_t width,
                                     const int32_t height);
// inits the pixelpipe with given cacheline size and number of
// entries. returns TRUE in case of success
gboolean dt_dev_pixelpipe_init_cached(dt_dev_pixelpipe_t *pipe,
                                      const size_t size,
                                      const int32_t entries,
                                      const int32_t fraction);
// returns available memory for the pipe
size_t dt_get_available_pipe_mem(const dt_dev_pixelpipe_t *pipe);
// constructs a new input buffer from given RGB float array.
void dt_dev_pixelpipe_set_input(dt_dev_pixelpipe_t *pipe,
                                struct dt_develop_t *dev,
                                float *input,
                                const int width,
                                const int height,
                                const float iscale);
// set some metadata for colorout to avoid race conditions.
void dt_dev_pixelpipe_set_icc(dt_dev_pixelpipe_t *pipe,
                              const dt_colorspaces_color_profile_type_t icc_type,
                              const gchar *icc_filename,
                              const dt_iop_color_intent_t icc_intent);

// returns the dimensions of the full image after processing.
void dt_dev_pixelpipe_get_dimensions(dt_dev_pixelpipe_t *pipe,
                                     struct dt_develop_t *dev,
                                     const int width_in,
                                     const int height_in,
                                     int *width,
                                     int *height);

// destroys all allocated data.
void dt_dev_pixelpipe_cleanup(dt_dev_pixelpipe_t *pipe);

// wrapper for cleanup_nodes, create_nodes, synch_all and synch_top,
// decides upon changed event which one to take on. also locks
// dev->history_mutex.
void dt_dev_pixelpipe_change(dt_dev_pixelpipe_t *pipe, struct dt_develop_t *dev);
// cleanup all nodes except clean input/output
void dt_dev_pixelpipe_cleanup_nodes(dt_dev_pixelpipe_t *pipe);
// sync with develop_t history stack from scratch (new node added, have to pop old ones)
void dt_dev_pixelpipe_create_nodes(dt_dev_pixelpipe_t *pipe, struct dt_develop_t *dev);
// sync with develop_t history stack by just copying the top item params (same op, new params on top)
void dt_dev_pixelpipe_synch_all(dt_dev_pixelpipe_t *pipe, struct dt_develop_t *dev);
// adjust output node according to history stack (history pop event)
void dt_dev_pixelpipe_synch_top(dt_dev_pixelpipe_t *pipe, struct dt_develop_t *dev);
// force a rebuild of the pipe, needed when a module order is changed for example
void dt_dev_pixelpipe_rebuild(struct dt_develop_t *dev);

/* Drop phantom entries -- deleted, disabled or de-synced consumers -- from
   `module`'s raster mask user table, judging every consumer from its node in
   THIS pipe.

   Called by synch_all and synch_top, which is where it belongs; declared here
   only so it can be tested directly. It has to answer for two consumer shapes
   that look nothing alike (the exclusive raster sink, and a DT_MASKS_RASTER
   form element inside a mask group) and for a pipe whose module state is stale
   by design (the export pipe, where piece->enabled is authoritative and
   module->enabled is not). Getting one cell of that matrix wrong drops a live
   consumer's mask, silently and only on export -- which is what regression
   0167-raster-mask was. Reaching it through synch_all would mean standing up a
   history stack to test a decision that reads none of it. */
void dt_dev_pixelpipe_prune_stale_raster_users(dt_dev_pixelpipe_t *pipe,
                                               struct dt_iop_module_t *module);

// process region of interest of pixels. returns TRUE if pipe was altered during processing.
gboolean dt_dev_pixelpipe_process(dt_dev_pixelpipe_t *pipe,
                             struct dt_develop_t *dev,
                             const int x,
                             const int y,
                             const int width,
                             const int height,
                             const float scale,
                             const int devid);
// convenience method that does not gamma-compress the image.
gboolean dt_dev_pixelpipe_process_no_gamma(dt_dev_pixelpipe_t *pipe,
                                      struct dt_develop_t *dev,
                                      const int x,
                                      const int y,
                                      const int width,
                                      const int height,
                                      const float scale);

// disable given op and all that comes after it in the pipe:
void dt_dev_pixelpipe_disable_after(dt_dev_pixelpipe_t *pipe, const char *op);
// disable given op and all that comes before it in the pipe:
void dt_dev_pixelpipe_disable_before(dt_dev_pixelpipe_t *pipe, const char *op);

// helper function to pass a raster mask through a (so far) processed pipe
float *dt_dev_get_raster_mask(dt_dev_pixelpipe_iop_t *piece,
                              const struct dt_iop_module_t *raster_mask_source,
                              const dt_mask_id_t raster_mask_id,
                              const struct dt_iop_module_t *target_module,
                              gboolean *free_mask);
// some helper functions related to the details mask interface
void dt_dev_clear_scharr_mask(dt_dev_pixelpipe_t *pipe);

gboolean dt_dev_write_scharr_mask(dt_dev_pixelpipe_iop_t *piece,
                                  float *const rgb,
                                  const dt_iop_roi_t *const roi_in,
                                  const gboolean mode);
#ifdef HAVE_OPENCL
int dt_dev_write_scharr_mask_cl(dt_dev_pixelpipe_iop_t *piece,
                                const cl_mem in,
                                const dt_iop_roi_t *const roi_in,
                                const gboolean mode);
#endif

void dt_dev_prepare_piece_cfa(dt_dev_pixelpipe_iop_t *piece, const dt_iop_roi_t *roi);

/* specialized version of dt_print for pixelpipe debugging */
void dt_print_pipe_ext(const char *title,
                       const dt_dev_pixelpipe_t *pipe,
                       const struct dt_iop_module_t *mod,
                       const int device,
                       const dt_iop_roi_t *roi_in,
                       const dt_iop_roi_t *roi_out,
                       const char *msg, ...)
  __attribute__((format(printf, 7, 8)));

// helper function writing the pipe-processed ctmask data to dest
float *dt_dev_distort_detail_mask(dt_dev_pixelpipe_iop_t *piece,
                                  const float *src,
                                  const struct dt_iop_module_t *target_module,
                                  const dt_hash_t src_hash);

dt_hash_t dt_dev_pixelpipe_piece_hash(dt_dev_pixelpipe_iop_t *piece,
                                      const dt_iop_roi_t *roi,
                                      const gboolean include);

G_END_DECLS

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
