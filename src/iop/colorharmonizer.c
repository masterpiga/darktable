/*
    This file is part of darktable,
    Copyright (C) 2010-2024 darktable developers.

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
#include "common/iop_profile.h"
#include "common/math.h"
#include "bauhaus/bauhaus.h"
#include "common/colorspaces.h"
#include "common/colorspaces_inline_conversions.h"
#include "common/chromatic_adaptation.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "gui/color_picker_proxy.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"
#include "libs/lib.h"
#include "common/color_harmony.h"
#include "common/opencl.h"

#include <gtk/gtk.h>
#include <math.h>
#include <stdlib.h>

#define COLORHARMONIZER_HUE_BINS 360
#define COLORHARMONIZER_MAX_NODES 4

DT_MODULE_INTROSPECTION(4, dt_iop_colorharmonizer_params_t)

typedef enum dt_iop_colorharmonizer_rule_t
{
  DT_COLORHARMONIZER_MONOCHROMATIC = 0,           // $DESCRIPTION: "monochromatic"
  DT_COLORHARMONIZER_ANALOGOUS = 1,               // $DESCRIPTION: "analogous"
  DT_COLORHARMONIZER_ANALOGOUS_COMPLEMENTARY = 2, // $DESCRIPTION: "analogous complementary"
  DT_COLORHARMONIZER_COMPLEMENTARY = 3,           // $DESCRIPTION: "complementary"
  DT_COLORHARMONIZER_SPLIT_COMPLEMENTARY = 4,     // $DESCRIPTION: "split complementary"
  DT_COLORHARMONIZER_DYAD = 5,                    // $DESCRIPTION: "dyad"
  DT_COLORHARMONIZER_TRIAD = 6,                   // $DESCRIPTION: "triad"
  DT_COLORHARMONIZER_TETRAD = 7,                  // $DESCRIPTION: "tetrad"
  DT_COLORHARMONIZER_SQUARE = 8,                  // $DESCRIPTION: "square"
  DT_COLORHARMONIZER_CUSTOM = 9                   // $DESCRIPTION: "custom"
} dt_iop_colorharmonizer_rule_t;

typedef struct dt_iop_colorharmonizer_params_t
{
  dt_iop_colorharmonizer_rule_t rule; // $DEFAULT: DT_COLORHARMONIZER_COMPLEMENTARY $DESCRIPTION: "harmony rule"
  float anchor_hue;                   // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.1 $DESCRIPTION: "anchor hue"
  float effect_strength;              // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "effect strength"
  float protect_neutral;              // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.2 $DESCRIPTION: "protect neutral"
  float zone_width;                   // $MIN: 0.25 $MAX: 4.0 $DEFAULT: 1.0 $DESCRIPTION: "effect width"
  float custom_hue[4];               // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "custom node hue"
  int   num_custom_nodes;            // $MIN: 2 $MAX: 4 $DEFAULT: 4 $DESCRIPTION: "active nodes"
} dt_iop_colorharmonizer_params_t;

typedef struct dt_iop_colorharmonizer_gui_data_t
{
  GtkWidget *rule, *anchor_hue, *effect_strength, *protect_neutral, *zone_width;
  GtkWidget *set_from_vectorscope, *sync_to_vectorscope, *auto_detect;
  GtkWidget *actions_box;                                // "auto detect" + "set from vectorscope" row
  GtkWidget *num_custom_nodes_slider;                    // node count (custom mode only)
  GtkWidget *swatches_area;                              // horizontal swatches row (non-custom mode)
  GtkWidget *node_swatch[COLORHARMONIZER_MAX_NODES];     // swatches inside swatches_area
  GtkWidget *custom_swatch[COLORHARMONIZER_MAX_NODES];   // swatches inside custom rows
  GtkWidget *custom_hue_slider[COLORHARMONIZER_MAX_NODES];
  GtkWidget *custom_row[COLORHARMONIZER_MAX_NODES];
  float      hue_histogram[COLORHARMONIZER_HUE_BINS];
  gboolean   histogram_valid;
  GMutex     histogram_lock;
} dt_iop_colorharmonizer_gui_data_t;

typedef struct dt_iop_colorharmonizer_global_data_t
{
  int kernel_colorharmonizer;
} dt_iop_colorharmonizer_global_data_t;

const char *name()
{
  return _("color harmonizer");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description
    (self,
     _("harmonize colors toward a selected palette in perceptual space"),
     _("creative color grading"),
     _("linear, RGB, scene-referred"),
     _("JzAzBz / JzCzhz (perceptual)"),
     _("linear, RGB, scene-referred"));
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}

int default_group()
{
  return IOP_GROUP_COLOR;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  // We work in RGB pipeline, convert internally to JzAzBz.
  return IOP_CS_RGB;
}

int legacy_params(dt_iop_module_t *self,
                  const void *const old_params,
                  const int old_version,
                  void **new_params,
                  int32_t *new_params_size,
                  int *new_version)
{
  if(old_version == 2)
  {
    typedef struct
    {
      dt_iop_colorharmonizer_rule_t rule;
      float anchor_hue;
      float effect_strength;
      float protect_neutral;
      float zone_width;
    } v2_params_t;

    typedef struct
    {
      dt_iop_colorharmonizer_rule_t rule;
      float anchor_hue;
      float effect_strength;
      float protect_neutral;
      float zone_width;
      float custom_hue[4];
    } v3_params_t;

    const v2_params_t *o = old_params;
    v3_params_t *n = malloc(sizeof(v3_params_t));

    n->rule            = o->rule;
    n->anchor_hue      = o->anchor_hue;
    n->effect_strength = o->effect_strength;
    n->protect_neutral = o->protect_neutral;
    n->zone_width      = o->zone_width;
    n->custom_hue[0]   = 0.0f;
    n->custom_hue[1]   = 0.25f;
    n->custom_hue[2]   = 0.5f;
    n->custom_hue[3]   = 0.75f;

    *new_params      = n;
    *new_params_size = sizeof(v3_params_t);
    *new_version     = 3;
    return 0;
  }
  if(old_version == 3)
  {
    typedef struct
    {
      dt_iop_colorharmonizer_rule_t rule;
      float anchor_hue;
      float effect_strength;
      float protect_neutral;
      float zone_width;
      float custom_hue[4];
    } v3_params_t;

    const v3_params_t *o = old_params;
    dt_iop_colorharmonizer_params_t *n = malloc(sizeof(dt_iop_colorharmonizer_params_t));

    n->rule             = o->rule;
    n->anchor_hue       = o->anchor_hue;
    n->effect_strength  = o->effect_strength;
    n->protect_neutral  = o->protect_neutral;
    n->zone_width       = o->zone_width;
    for(int i = 0; i < 4; i++) n->custom_hue[i] = o->custom_hue[i];
    n->num_custom_nodes = 4;

    *new_params      = n;
    *new_params_size = sizeof(dt_iop_colorharmonizer_params_t);
    *new_version     = 4;
    return 0;
  }
  return 1;
}

void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  memcpy(piece->data, p1, self->params_size);
}

// Compute a Gaussian-weighted hue shift toward all harmony nodes.
//
// Rather than snapping to the single nearest node (which creates a hard zone
// boundary at the angular midpoint between adjacent nodes — visible as harsh
// colour transitions in smooth gradients), every node contributes a weighted
// pull whose strength decays smoothly with angular distance.
// Pull the pixel hue toward the nearest harmony node.
// Returns a hue delta scaled by the Gaussian proximity to that node.
//
//   Narrow zone (< 1): Gaussian drops off quickly → only hues very close to a
//                       node are attracted; distant hues are barely shifted.
//   Default zone (1):  Gaussian tapers to ~14 % at the midpoint between nodes.
//   Wide zone   (> 1): Gaussian stays high across the full hue circle → broad,
//                       global correction; all hues are pulled noticeably.
//
// Using the nearest-node approach (max weight) rather than a weighted average
// avoids the cancellation artefact that occurs when opposing nodes (e.g.
// complementary) pull in opposite directions and nearly neutralize each other.
static inline float get_weighted_hue_shift(float px_hue, const float *nodes, int num_nodes,
                                           float zone_width_factor)
{
  const float sigma = zone_width_factor * 0.5f / (float)num_nodes;
  const float inv_2sigma2 = 1.0f / (2.0f * sigma * sigma);

  float max_weight = 0.0f;
  float best_diff  = 0.0f;

  for(int i = 0; i < num_nodes; i++)
  {
    float d = fabsf(px_hue - nodes[i]);
    if(d > 0.5f) d = 1.0f - d;

    const float w = expf(-d * d * inv_2sigma2);

    if(w > max_weight)
    {
      max_weight = w;
      float diff = nodes[i] - px_hue;
      if(diff > 0.5f)       diff -= 1.0f;
      else if(diff < -0.5f) diff += 1.0f;
      best_diff = diff;
    }
  }

  // max_weight acts as a proximity gate:
  //   wide zone  → high weight for all pixels  → broad effect
  //   narrow zone → low weight far from nodes  → selective effect
  return max_weight * best_diff;
}

static inline float wrap_hue(float h)
{
  h = fmodf(h, 1.0f);
  if (h < 0.0f) h += 1.0f;
  return h;
}

static inline void get_harmony_nodes(dt_iop_colorharmonizer_rule_t rule, float anchor_hue,
                                     const float *custom_hue, int custom_n,
                                     float *nodes, int *num_nodes)
{
  switch(rule)
  {
    case DT_COLORHARMONIZER_MONOCHROMATIC:
      nodes[0] = anchor_hue;
      *num_nodes = 1;
      break;
    case DT_COLORHARMONIZER_ANALOGOUS:
      nodes[0] = anchor_hue;
      nodes[1] = wrap_hue(anchor_hue - 0.083333f); // -30 deg
      nodes[2] = wrap_hue(anchor_hue + 0.083333f); // +30 deg
      *num_nodes = 3;
      break;
    case DT_COLORHARMONIZER_ANALOGOUS_COMPLEMENTARY:
      nodes[0] = anchor_hue;
      nodes[1] = wrap_hue(anchor_hue - 0.083333f); // -30 deg
      nodes[2] = wrap_hue(anchor_hue + 0.083333f); // +30 deg
      nodes[3] = wrap_hue(anchor_hue + 0.5f);      // +180 deg
      *num_nodes = 4;
      break;
    case DT_COLORHARMONIZER_COMPLEMENTARY:
      nodes[0] = anchor_hue;
      nodes[1] = wrap_hue(anchor_hue + 0.5f);
      *num_nodes = 2;
      break;
    case DT_COLORHARMONIZER_SPLIT_COMPLEMENTARY:
      nodes[0] = anchor_hue;
      nodes[1] = wrap_hue(anchor_hue + 0.416667f); // +150 deg
      nodes[2] = wrap_hue(anchor_hue + 0.583333f); // +210 deg
      *num_nodes = 3;
      break;
    case DT_COLORHARMONIZER_DYAD:
      // anchor_hue is the symmetry axis; the two members sit ±30° around it,
      // matching the vectorscope harmony guide definition.
      nodes[0] = wrap_hue(anchor_hue - 0.083333f); // -30 deg
      nodes[1] = wrap_hue(anchor_hue + 0.083333f); // +30 deg
      *num_nodes = 2;
      break;
    case DT_COLORHARMONIZER_TRIAD:
      nodes[0] = anchor_hue;
      nodes[1] = wrap_hue(anchor_hue + 0.333333f); // +120 deg
      nodes[2] = wrap_hue(anchor_hue + 0.666667f); // +240 deg
      *num_nodes = 3;
      break;
    case DT_COLORHARMONIZER_TETRAD:
      // Two dyad pairs symmetric around anchor_hue; anchor itself is not a member node,
      // matching the vectorscope harmony guide definition.
      nodes[0] = wrap_hue(anchor_hue - 0.083333f); // -30 deg
      nodes[1] = wrap_hue(anchor_hue + 0.083333f); // +30 deg
      nodes[2] = wrap_hue(anchor_hue + 0.416667f); // +150 deg
      nodes[3] = wrap_hue(anchor_hue + 0.583333f); // +210 deg
      *num_nodes = 4;
      break;
    case DT_COLORHARMONIZER_SQUARE:
      nodes[0] = anchor_hue;
      nodes[1] = wrap_hue(anchor_hue + 0.25f);     // +90 deg
      nodes[2] = wrap_hue(anchor_hue + 0.5f);      // +180 deg
      nodes[3] = wrap_hue(anchor_hue + 0.75f);     // +270 deg
      *num_nodes = 4;
      break;
    case DT_COLORHARMONIZER_CUSTOM:
      // Each node position is independently user-defined.
      for(int i = 0; i < custom_n; i++)
        nodes[i] = custom_hue ? custom_hue[i] : 0.0f;
      *num_nodes = custom_n;
      break;
    default:
      nodes[0] = anchor_hue;
      *num_nodes = 1;
      break;
  }
}

void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const ivoid,
             void *const ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  dt_iop_colorharmonizer_params_t *p = piece->data;
  const size_t ch = piece->colors;

  if(!dt_iop_have_required_input_format(4, self, piece->colors, ivoid, ovoid, roi_in, roi_out))
    return;

  // Pre-calculate target nodes based on the harmony rule.
  float nodes[COLORHARMONIZER_MAX_NODES] = {0};
  int num_nodes = 1;
  get_harmony_nodes(p->rule, p->anchor_hue, p->custom_hue, p->num_custom_nodes, nodes, &num_nodes);

  // To convert pipeline working RGB space to XYZ, we need the matrix
  const dt_iop_order_iccprofile_info_t *const work_profile = dt_ioppr_get_pipe_work_profile_info(piece->pipe);
  if(!work_profile) return;

  DT_OMP_FOR()
  for(int j = 0; j < roi_out->height; j++)
  {
    const float *in = ((const float *)ivoid) + (size_t)ch * roi_in->width * j;
    float *out = ((float *)ovoid) + (size_t)ch * roi_out->width * j;

    for(int i = 0; i < roi_out->width; i++, in += ch, out += ch)
    {
      // 1. Pipeline RGB -> XYZ
      dt_aligned_pixel_t px_rgb, px_xyz;
      for_each_channel(c) px_rgb[c] = fmaxf(in[c], 0.0f); // Ensure positive
      px_rgb[3] = 0.0f;
      dt_apply_transposed_color_matrix(px_rgb, work_profile->matrix_in_transposed, px_xyz);

      // 2. XYZ (D50) -> XYZ (D65) -> JzAzBz -> JzCzhz
      dt_aligned_pixel_t px_xyz_d65, px_jzazbz, px_jzczhz;
      XYZ_D50_to_D65(px_xyz, px_xyz_d65);
      dt_XYZ_2_JzAzBz(px_xyz_d65, px_jzazbz);
      dt_JzAzBz_2_JzCzhz(px_jzazbz, px_jzczhz);

      // Hue is JzCzhz[2], Chroma is JzCzhz[1]
      float hue = px_jzczhz[2];
      float chroma = px_jzczhz[1];

      // Protect neutrals: reduce effect where chroma is low.
      // Uses a hyperbolic formula (no hard ceiling) with quadratic slider mapping
      // so sensitivity is distributed evenly across the full slider range.
      // protect_neutral = 0 -> full effect on all pixels
      // protect_neutral = 1 -> strong suppression even on moderately saturated colors
      const float t = p->protect_neutral;
      const float cutoff = t * t * t * 0.3f;
      const float chroma_weight = chroma / (chroma + cutoff + 1e-5f);

      // Soft weighted pull toward harmony nodes (smooth across zone boundaries)
      const float hue_shift = get_weighted_hue_shift(hue, nodes, num_nodes, p->zone_width);
      const float pull_amount = p->effect_strength * chroma_weight;
      float new_hue = wrap_hue(hue + hue_shift * pull_amount);

      px_jzczhz[2] = new_hue;

      // 3. JzCzhz -> JzAzBz -> XYZ (D65) -> XYZ (D50)
      dt_JzCzhz_2_JzAzBz(px_jzczhz, px_jzazbz);
      dt_JzAzBz_2_XYZ(px_jzazbz, px_xyz_d65);
      XYZ_D65_to_D50(px_xyz_d65, px_xyz);

      // 4. XYZ -> Pipeline RGB
      dt_aligned_pixel_t px_rgb_out;
      dt_apply_transposed_color_matrix(px_xyz, work_profile->matrix_out_transposed, px_rgb_out);
      for_each_channel(c) out[c] = px_rgb_out[c];
      out[3] = in[3]; // Copy alpha
    }
  }

  // Build a chroma-weighted hue histogram from the input for auto-detection.
  // Runs only on the preview pipe (small, representative, updated frequently).
  if(self->gui_data && (piece->pipe->type & DT_DEV_PIXELPIPE_PREVIEW) == DT_DEV_PIXELPIPE_PREVIEW)
  {
    dt_iop_colorharmonizer_gui_data_t *g = self->gui_data;
    float local_histo[COLORHARMONIZER_HUE_BINS] = { 0.0f };

    for(int j = 0; j < roi_in->height; j++)
    {
      const float *src = ((const float *)ivoid) + (size_t)ch * roi_in->width * j;
      for(int i = 0; i < roi_in->width; i++, src += ch)
      {
        dt_aligned_pixel_t px_h, px_xyz_h, px_xyz_d65_h, px_jzazbz_h, px_jzczhz_h;
        for_each_channel(c) px_h[c] = fmaxf(src[c], 0.0f);
        px_h[3] = 0.0f;
        dt_apply_transposed_color_matrix(px_h, work_profile->matrix_in_transposed, px_xyz_h);
        XYZ_D50_to_D65(px_xyz_h, px_xyz_d65_h);
        dt_XYZ_2_JzAzBz(px_xyz_d65_h, px_jzazbz_h);
        dt_JzAzBz_2_JzCzhz(px_jzazbz_h, px_jzczhz_h);

        const float chroma_h = px_jzczhz_h[1];
        if(chroma_h > 0.01f)
        {
          const int bin = (int)(px_jzczhz_h[2] * COLORHARMONIZER_HUE_BINS) % COLORHARMONIZER_HUE_BINS;
          local_histo[bin] += chroma_h;
        }
      }
    }

    g_mutex_lock(&g->histogram_lock);
    memcpy(g->hue_histogram, local_histo, sizeof(local_histo));
    g->histogram_valid = TRUE;
    g_mutex_unlock(&g->histogram_lock);
  }
}

void init(dt_iop_module_t *self)
{
  dt_iop_default_init(self);

  // Set sensible defaults for custom harmony nodes (evenly spread)
  dt_iop_colorharmonizer_params_t *p = self->default_params;
  p->custom_hue[0] = 0.0f;
  p->custom_hue[1] = 0.25f;
  p->custom_hue[2] = 0.5f;
  p->custom_hue[3] = 0.75f;
  p->num_custom_nodes = 4;
  memcpy(self->params, self->default_params, self->params_size);
}

void cleanup(dt_iop_module_t *self)
{
  dt_iop_default_cleanup(self);
}

void init_global(dt_iop_module_so_t *self)
{
  const int program = 40; // colorharmonizer.cl in programs.conf
  dt_iop_colorharmonizer_global_data_t *gd = malloc(sizeof(dt_iop_colorharmonizer_global_data_t));
  self->data = gd;
  gd->kernel_colorharmonizer = dt_opencl_create_kernel(program, "colorharmonizer");
}

void cleanup_global(dt_iop_module_so_t *self)
{
  dt_iop_colorharmonizer_global_data_t *gd = self->data;
  dt_opencl_free_kernel(gd->kernel_colorharmonizer);
  free(self->data);
  self->data = NULL;
}

#ifdef HAVE_OPENCL
int process_cl(dt_iop_module_t *self,
               dt_dev_pixelpipe_iop_t *piece,
               cl_mem dev_in, cl_mem dev_out,
               const dt_iop_roi_t *const roi_in,
               const dt_iop_roi_t *const roi_out)
{
  const dt_iop_colorharmonizer_params_t *const p = piece->data;
  const dt_iop_colorharmonizer_global_data_t *const gd = self->global_data;

  cl_int err = DT_OPENCL_DEFAULT_ERROR;

  if(piece->colors != 4)
    return err;

  const int devid = piece->pipe->devid;
  const int width  = roi_out->width;
  const int height = roi_out->height;

  const dt_iop_order_iccprofile_info_t *const work_profile = dt_ioppr_get_pipe_work_profile_info(piece->pipe);
  if(!work_profile) return err;

  // Premultiply: RGB_pipeline -> XYZ_D65 = XYZ_D50_to_D65 @ RGB_to_XYZ_D50
  dt_colormatrix_t input_matrix;
  dt_colormatrix_mul(input_matrix, XYZ_D50_to_D65_CAT16, work_profile->matrix_in);

  // Premultiply: XYZ_D65 -> RGB_pipeline = XYZ_D50_to_RGB @ XYZ_D65_to_D50
  dt_colormatrix_t output_matrix;
  dt_colormatrix_mul(output_matrix, work_profile->matrix_out, XYZ_D65_to_D50_CAT16);

  cl_mem input_matrix_cl  = dt_opencl_copy_host_to_device_constant(devid, 12 * sizeof(float), input_matrix);
  cl_mem output_matrix_cl = dt_opencl_copy_host_to_device_constant(devid, 12 * sizeof(float), output_matrix);

  // Harmony nodes
  float nodes[COLORHARMONIZER_MAX_NODES] = { 0.f };
  int num_nodes = 1;
  get_harmony_nodes(p->rule, p->anchor_hue, p->custom_hue, p->num_custom_nodes, nodes, &num_nodes);
  cl_mem nodes_cl = dt_opencl_copy_host_to_device_constant(devid, COLORHARMONIZER_MAX_NODES * sizeof(float), nodes);

  if(input_matrix_cl == NULL || output_matrix_cl == NULL || nodes_cl == NULL)
  {
    err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    goto error;
  }

  err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_colorharmonizer, width, height,
    CLARG(dev_in), CLARG(dev_out),
    CLARG(width), CLARG(height),
    CLARG(input_matrix_cl), CLARG(output_matrix_cl),
    CLARG(nodes_cl), CLARG(num_nodes),
    CLARG(p->zone_width), CLARG(p->effect_strength), CLARG(p->protect_neutral));

error:
  dt_opencl_release_mem_object(input_matrix_cl);
  dt_opencl_release_mem_object(output_matrix_cl);
  dt_opencl_release_mem_object(nodes_cl);
  return err;
}
#endif

#define ANGLE_SHIFT -30.f
#define CONVENTIONAL_DEG_TO_YRG_RAD(x) (deg2radf(x + ANGLE_SHIFT))

static inline void prepare_RGB_Yrg_matrices(const dt_iop_order_iccprofile_info_t *const profile,
                                            dt_colormatrix_t input_matrix, dt_colormatrix_t output_matrix)
{
  dt_colormatrix_t temp_matrix;

  // Prepare the RGB (D50) -> XYZ D50 -> XYZ D65 -> LMS 2006 matrix
  dt_colormatrix_mul(temp_matrix, XYZ_D50_to_D65_CAT16, profile->matrix_in);
  dt_colormatrix_mul(input_matrix, XYZ_D65_to_LMS_2006_D65, temp_matrix);

  // Prepare the LMS 2006 -> XYZ D65 -> XYZ D50 -> RGB matrix (D50)
  dt_colormatrix_mul(temp_matrix, XYZ_D65_to_D50_CAT16, LMS_2006_D65_to_XYZ_D65);
  dt_colormatrix_mul(output_matrix, profile->matrix_out, temp_matrix);
}

static void _YchToRGB(dt_aligned_pixel_t *RGB_out, const float chroma, const float hue,
                      const dt_iop_order_iccprofile_info_t *output_profile,
                      const dt_colormatrix_t output_matrix_LMS_to_RGB)
{
  dt_aligned_pixel_t RGB_linear = { 0.f };
  dt_aligned_pixel_t Ych;
  make_Ych(0.75f, chroma, hue, Ych);
  dt_aligned_pixel_t XYZ_D65 = { 0.f };
  dt_aligned_pixel_t XYZ_D50 = { 0.f };
  Ych_to_XYZ(Ych, XYZ_D65);
  XYZ_D65_to_D50(XYZ_D65, XYZ_D50);
  dt_apply_transposed_color_matrix(XYZ_D50, output_profile->matrix_out_transposed, RGB_linear);
  // normalize to the brightest value available at this hue and chroma
  const float max_RGB = max3f(RGB_linear);
  for_each_channel(c) RGB_linear[c] = MAX(RGB_linear[c] / max_RGB, 0.f);
  // Apply nonlinear LUT if necessary
  if(output_profile->nonlinearlut)
    dt_ioppr_apply_trc(RGB_linear, *RGB_out, output_profile->lut_out, output_profile->unbounded_coeffs_out,
                       output_profile->lutsize);
  else
    memcpy(*RGB_out, RGB_linear, sizeof(RGB_linear));
}

static inline float _clip_chroma_black(const float coeffs[3], const float cos_h, const float sin_h)
{
  // N.B. this is the same as clip_chroma_white_raw() but with target value = 0.
  // This allows eliminating some computation.

  // Get chroma that brings one component of target RGB to zero.
  // coeffs are the transformation coeffs to get one components (R, G or B) from input LMS.
  // i.e. it is a row of the LMS -> RGB transformation matrix.
  // See tools/derive_filmic_v6_gamut_mapping.py for derivation of these equations.
  const float denominator = coeffs[0] * (0.979381443298969f * cos_h + 0.391752577319588f * sin_h)
                            + coeffs[1] * (0.0206185567010309f * cos_h + 0.608247422680412f * sin_h)
                            - coeffs[2] * (cos_h + sin_h);

  // this channel won't limit the chroma
  if(denominator == 0.f) return FLT_MAX;

  const float numerator = -0.427506877216495f * (coeffs[0] + 0.856492345150334f * coeffs[1] + 0.554995960637719f * coeffs[2]);
  const float max_chroma = numerator / denominator;
  return max_chroma >= 0.f ? max_chroma : FLT_MAX;
}

static inline float Ych_max_chroma_without_negatives(const dt_colormatrix_t matrix_out,
                                                     const float cos_h, const float sin_h)
{
  const float chroma_R_black = _clip_chroma_black(matrix_out[0], cos_h, sin_h);
  const float chroma_G_black = _clip_chroma_black(matrix_out[1], cos_h, sin_h);
  const float chroma_B_black = _clip_chroma_black(matrix_out[2], cos_h, sin_h);
  return MIN(MIN(chroma_R_black, chroma_G_black), chroma_B_black);
}

static void _paint_hue_slider(const dt_iop_order_iccprofile_info_t *output_profile,
                               const dt_colormatrix_t output_matrix_LMS_to_RGB,
                               GtkWidget *slider)
{
  for(int i = 0; i < DT_BAUHAUS_SLIDER_MAX_STOPS; i++)
  {
    const float stop = ((float)i / (float)(DT_BAUHAUS_SLIDER_MAX_STOPS - 1));
    const float h = CONVENTIONAL_DEG_TO_YRG_RAD(stop * 360.f);
    const float max_chroma = Ych_max_chroma_without_negatives(output_matrix_LMS_to_RGB, cosf(h), sinf(h));
    dt_aligned_pixel_t RGB;
    _YchToRGB(&RGB, MIN(0.2f, max_chroma), h, output_profile, output_matrix_LMS_to_RGB);
    dt_bauhaus_slider_set_stop(slider, stop, RGB[0], RGB[1], RGB[2]);
  }
}

static void _paint_all_hue_sliders(const dt_iop_order_iccprofile_info_t *output_profile,
                                    const dt_colormatrix_t output_matrix_LMS_to_RGB,
                                    const dt_iop_colorharmonizer_gui_data_t *const g)
{
  _paint_hue_slider(output_profile, output_matrix_LMS_to_RGB, g->anchor_hue);
  for(int i = 0; i < COLORHARMONIZER_MAX_NODES; i++)
    _paint_hue_slider(output_profile, output_matrix_LMS_to_RGB, g->custom_hue_slider[i]);
}

static void _push_to_vectorscope(dt_iop_module_t *self)
{
  dt_iop_colorharmonizer_params_t *p = self->params;
  dt_color_harmony_guide_t guide;
  dt_lib_histogram_get_harmony(darktable.lib, &guide);

  if(p->rule == DT_COLORHARMONIZER_CUSTOM)
  {
    // Custom: provide absolute-angle nodes; type is left as NONE so the
    // vectorscope's own UI shows no standard rule selected.
    guide.type     = DT_COLOR_HARMONY_NONE;
    guide.custom_n = p->num_custom_nodes;
    for(int i = 0; i < p->num_custom_nodes; i++)
      guide.custom_angles[i] = p->custom_hue[i];
  }
  else
  {
    guide.type        = (dt_color_harmony_type_t)(p->rule + 1);
    guide.rotation    = (int)roundf(p->anchor_hue * 360.0f) % 360;
    guide.custom_n    = 0;
  }

  dt_lib_histogram_set_harmony(darktable.lib, &guide);
  dt_lib_histogram_set_scope(darktable.lib, 0); // 0 = DT_LIB_HISTOGRAM_SCOPE_VECTORSCOPE
}

static void _sync_to_vectorscope_toggled(GtkToggleButton *button, dt_iop_module_t *self)
{
  if(gtk_toggle_button_get_active(button))
  {
    _push_to_vectorscope(self);
  }
  else
  {
    // Custom guides only live in memory — clear them from the vectorscope when
    // sync is disabled so they don't linger as stale overlays.
    dt_iop_colorharmonizer_params_t *p = self->params;
    if(p->rule == DT_COLORHARMONIZER_CUSTOM)
    {
      dt_color_harmony_guide_t guide;
      dt_lib_histogram_get_harmony(darktable.lib, &guide);
      guide.custom_n = 0;
      guide.type     = DT_COLOR_HARMONY_NONE;
      dt_lib_histogram_set_harmony(darktable.lib, &guide);
    }
  }
}

// Render the hue for a single harmony node swatch.
// Converts a normalized hue [0,1] to an sRGB color using JzCzhz.
static void _hue_to_srgb(float hue, float *r, float *g, float *b)
{
  const float Jz = 0.1f;
  const float Cz = 0.05f;
  dt_aligned_pixel_t px_jzczhz = { Jz, Cz, hue, 0.0f };
  dt_aligned_pixel_t px_jzazbz, px_xyz_d65, px_xyz_d50, px_rgb;

  dt_JzCzhz_2_JzAzBz(px_jzczhz, px_jzazbz);
  dt_JzAzBz_2_XYZ(px_jzazbz, px_xyz_d65);
  for(int c = 0; c < 3; c++) px_xyz_d65[c] /= 100.0f;
  XYZ_D65_to_D50(px_xyz_d65, px_xyz_d50);
  dt_XYZ_to_sRGB(px_xyz_d50, px_rgb);

  *r = CLAMP(px_rgb[0], 0.0f, 1.0f);
  *g = CLAMP(px_rgb[1], 0.0f, 1.0f);
  *b = CLAMP(px_rgb[2], 0.0f, 1.0f);
}

static gboolean _swatch_draw_callback(GtkWidget *widget, cairo_t *cr, gpointer user_data)
{
  dt_iop_module_t *self = user_data;
  dt_iop_colorharmonizer_params_t *p = self->params;
  const int idx = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(widget), "swatch-index"));

  float hue = 0.0f;
  gboolean show_color = TRUE;

  if(p->rule == DT_COLORHARMONIZER_CUSTOM)
  {
    hue = p->custom_hue[idx];
  }
  else
  {
    float nodes[COLORHARMONIZER_MAX_NODES];
    int num_nodes = 0;
    get_harmony_nodes(p->rule, p->anchor_hue, NULL, COLORHARMONIZER_MAX_NODES, nodes, &num_nodes);
    if(idx < num_nodes)
      hue = nodes[idx];
    else
      show_color = FALSE; // unused slot: draw neutral grey
  }

  if(!show_color)
  {
    cairo_set_source_rgb(cr, 0.25, 0.25, 0.25);
    cairo_paint(cr);
    return TRUE;
  }

  float r, g, b;
  _hue_to_srgb(hue, &r, &g, &b);
  cairo_set_source_rgb(cr, r, g, b);
  cairo_paint(cr);
  return TRUE;
}

static void _custom_hue_changed(GtkWidget *widget, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;

  // Deactivate any active color picker when the slider is moved directly.
  dt_iop_color_picker_reset(self, TRUE);

  dt_iop_colorharmonizer_params_t *p = self->params;
  dt_iop_colorharmonizer_gui_data_t *g = self->gui_data;
  const int idx = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(widget), "slider-index"));

  p->custom_hue[idx] = dt_bauhaus_slider_get(widget);

  gtk_widget_queue_draw(g->custom_swatch[idx]);
  gtk_widget_queue_draw(g->node_swatch[idx]);
  dt_dev_add_history_item(self->dev, self, TRUE);

  if(g->sync_to_vectorscope
     && gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->sync_to_vectorscope)))
    _push_to_vectorscope(self);
}

void gui_changed(dt_iop_module_t *self, GtkWidget *widget, void *previous)
{
  dt_iop_colorharmonizer_gui_data_t *g = self->gui_data;
  if(!g) return;

  dt_iop_colorharmonizer_params_t *p = self->params;
  const gboolean is_custom = (p->rule == DT_COLORHARMONIZER_CUSTOM);

  // Toggle between layouts depending on mode
  gtk_widget_set_visible(g->anchor_hue, !is_custom);
  gtk_widget_set_visible(g->swatches_area, !is_custom);
  gtk_widget_set_visible(g->actions_box, !is_custom);
  gtk_widget_set_visible(g->num_custom_nodes_slider, is_custom);

  if(is_custom)
  {
    // Show only the active custom rows
    const int n = CLAMP(p->num_custom_nodes, 2, COLORHARMONIZER_MAX_NODES);
    for(int i = 0; i < COLORHARMONIZER_MAX_NODES; i++)
      gtk_widget_set_visible(g->custom_row[i], i < n);
  }
  else
  {
    for(int i = 0; i < COLORHARMONIZER_MAX_NODES; i++)
      gtk_widget_set_visible(g->custom_row[i], FALSE);

    // Show only as many swatches as the rule needs
    float nodes[COLORHARMONIZER_MAX_NODES];
    int num_nodes = 0;
    get_harmony_nodes(p->rule, p->anchor_hue, NULL, COLORHARMONIZER_MAX_NODES, nodes, &num_nodes);
    for(int i = 0; i < COLORHARMONIZER_MAX_NODES; i++)
      gtk_widget_set_visible(g->node_swatch[i], i < num_nodes);
  }

  // Redraw all swatches (both sets; only the visible ones will paint)
  for(int i = 0; i < COLORHARMONIZER_MAX_NODES; i++)
  {
    gtk_widget_queue_draw(g->node_swatch[i]);
    gtk_widget_queue_draw(g->custom_swatch[i]);
  }

  // In custom mode, sync slider values to params (they are editable)
  if(is_custom)
  {
    ++darktable.gui->reset;
    for(int i = 0; i < COLORHARMONIZER_MAX_NODES; i++)
      dt_bauhaus_slider_set(g->custom_hue_slider[i], p->custom_hue[i]);
    --darktable.gui->reset;
  }

  // Paint hue slider gradients
  const dt_iop_order_iccprofile_info_t *output_profile = NULL;
  if(self->dev && self->dev->full.pipe)
    output_profile = dt_ioppr_get_pipe_output_profile_info(self->dev->full.pipe);

  if(!output_profile || !dt_is_valid_colormatrix(output_profile->matrix_out[0][0]))
  {
    output_profile = dt_ioppr_add_profile_info_to_list(self->dev, DT_COLORSPACE_SRGB, "",
                                                       DT_INTENT_RELATIVE_COLORIMETRIC);
  }
  dt_colormatrix_t input_matrix = { { 0.f } };
  dt_colormatrix_t output_matrix = { { 0.f } };
  prepare_RGB_Yrg_matrices(output_profile, input_matrix, output_matrix);
  _paint_all_hue_sliders(output_profile, output_matrix, g);

  // Auto-sync rule / anchor hue (or the custom rule selection) to the vectorscope
  if(widget && (widget == g->rule || widget == g->anchor_hue || widget == g->num_custom_nodes_slider)
     && g->sync_to_vectorscope
     && gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->sync_to_vectorscope)))
    _push_to_vectorscope(self);
}

void gui_update(dt_iop_module_t *self)
{
  dt_iop_colorharmonizer_gui_data_t *g = self->gui_data;
  dt_iop_colorharmonizer_params_t *p = self->params;

  dt_bauhaus_combobox_set(g->rule, p->rule);
  dt_bauhaus_slider_set(g->anchor_hue, p->anchor_hue);
  dt_bauhaus_slider_set(g->effect_strength, p->effect_strength);
  dt_bauhaus_slider_set(g->protect_neutral, p->protect_neutral);
  dt_bauhaus_slider_set(g->zone_width, p->zone_width);

  ++darktable.gui->reset;
  for(int i = 0; i < COLORHARMONIZER_MAX_NODES; i++)
    dt_bauhaus_slider_set(g->custom_hue_slider[i], p->custom_hue[i]);
  --darktable.gui->reset;

  gui_changed(self, NULL, NULL);
  dt_iop_color_picker_reset(self, TRUE);
}

// Convert a picked pixel color (pipeline RGB) to a normalized JzCzhz hue [0, 1].
// Convert a picked pixel color to a normalized JzCzhz hue [0, 1].
// Returns FALSE only if the work profile is unavailable.
static gboolean _picked_color_to_hue(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, float *out_hue)
{
  const dt_iop_order_iccprofile_info_t *const work_profile = pipe->work_profile_info;
  if(!work_profile) return FALSE;

  dt_aligned_pixel_t px_rgb, px_xyz;
  for(int c = 0; c < 3; c++) px_rgb[c] = fmaxf(self->picked_color[c], 0.0f);
  px_rgb[3] = 0.0f;

  const float (*const matrix_in)[4] = (const float (*)[4])work_profile->matrix_in;
  px_xyz[0] = matrix_in[0][0]*px_rgb[0] + matrix_in[0][1]*px_rgb[1] + matrix_in[0][2]*px_rgb[2];
  px_xyz[1] = matrix_in[1][0]*px_rgb[0] + matrix_in[1][1]*px_rgb[1] + matrix_in[1][2]*px_rgb[2];
  px_xyz[2] = matrix_in[2][0]*px_rgb[0] + matrix_in[2][1]*px_rgb[1] + matrix_in[2][2]*px_rgb[2];
  px_xyz[3] = 0.0f;

  dt_aligned_pixel_t px_xyz_d65, px_jzazbz, px_jzczhz;
  XYZ_D50_to_D65(px_xyz, px_xyz_d65);
  dt_XYZ_2_JzAzBz(px_xyz_d65, px_jzazbz);
  dt_JzAzBz_2_JzCzhz(px_jzazbz, px_jzczhz);

  *out_hue = px_jzczhz[2];
  return TRUE;
}

void color_picker_apply(dt_iop_module_t *self, GtkWidget *picker, dt_dev_pixelpipe_t *pipe)
{
  dt_iop_colorharmonizer_gui_data_t *g = self->gui_data;
  dt_iop_colorharmonizer_params_t *p = self->params;

  float hue = 0.0f;
  if(!_picked_color_to_hue(self, pipe, &hue)) return;

  if(picker == g->anchor_hue)
  {
    p->anchor_hue = hue;
    ++darktable.gui->reset;
    dt_bauhaus_slider_set(g->anchor_hue, p->anchor_hue);
    --darktable.gui->reset;

    // Update swatch display (both sets)
    for(int i = 0; i < COLORHARMONIZER_MAX_NODES; i++)
    {
      gtk_widget_queue_draw(g->custom_swatch[i]);
      gtk_widget_queue_draw(g->node_swatch[i]);
    }

    dt_dev_add_history_item(self->dev, self, TRUE);
    return;
  }

  for(int i = 0; i < COLORHARMONIZER_MAX_NODES; i++)
  {
    if(picker == g->custom_hue_slider[i])
    {
      p->custom_hue[i] = hue;
      ++darktable.gui->reset;
      dt_bauhaus_slider_set(g->custom_hue_slider[i], hue);
      --darktable.gui->reset;
      gtk_widget_queue_draw(g->custom_swatch[i]);
      dt_dev_add_history_item(self->dev, self, TRUE);
      return;
    }
  }
}

// Score how well a harmony rule+anchor covers the image's hue histogram.
// Returns a coverage fraction in [0,1]: the fraction of chromatic energy that
// falls within the Gaussian attraction zones of the harmony nodes.
// Uses the same sigma formula as the main algorithm (zone_width_factor = 1.0).
static float _score_harmony(const float *histo, int num_bins,
                              dt_iop_colorharmonizer_rule_t rule, float anchor_hue)
{
  float nodes[COLORHARMONIZER_MAX_NODES];
  int num_nodes = 1;
  get_harmony_nodes(rule, anchor_hue, NULL, COLORHARMONIZER_MAX_NODES, nodes, &num_nodes);

  const float sigma = 0.5f / (float)num_nodes;
  const float inv_2sigma2 = 1.0f / (2.0f * sigma * sigma);

  float total   = 0.0f;
  float covered = 0.0f;

  for(int b = 0; b < num_bins; b++)
  {
    if(histo[b] <= 0.0f) continue;
    const float h = (b + 0.5f) / (float)num_bins;

    float max_w = 0.0f;
    for(int i = 0; i < num_nodes; i++)
    {
      float d = fabsf(h - nodes[i]);
      if(d > 0.5f) d = 1.0f - d;
      const float w = expf(-d * d * inv_2sigma2);
      if(w > max_w) max_w = w;
    }

    covered += histo[b] * max_w;
    total   += histo[b];
  }

  return (total > 1e-6f) ? (covered / total) : 0.0f;
}

// Analyse a chroma-weighted hue histogram and return the harmony rule and
// anchor hue that best explain the existing color distribution (i.e. the
// combination that already covers the most chromatic energy).
static void _auto_detect_harmony(const float *histo, int num_bins,
                                  dt_iop_colorharmonizer_rule_t *best_rule,
                                  float *best_anchor)
{
  // Smooth the histogram with three passes of a circular box filter to
  // suppress noise from individual pixels before scoring.
  float smooth[COLORHARMONIZER_HUE_BINS];
  memcpy(smooth, histo, num_bins * sizeof(float));
  for(int pass = 0; pass < 3; pass++)
  {
    float tmp[COLORHARMONIZER_HUE_BINS];
    for(int b = 0; b < num_bins; b++)
    {
      const int prev = (b - 1 + num_bins) % num_bins;
      const int next = (b + 1) % num_bins;
      tmp[b] = (smooth[prev] + smooth[b] + smooth[next]) * (1.0f / 3.0f);
    }
    memcpy(smooth, tmp, num_bins * sizeof(float));
  }

  float best_score = -1.0f;
  *best_rule   = DT_COLORHARMONIZER_COMPLEMENTARY;
  *best_anchor = 0.0f;

  // Only test predefined (non-custom) rules
  const int num_rules = DT_COLORHARMONIZER_SQUARE + 1;
  const int num_steps = 72; // 5° resolution — 9 × 72 = 648 combinations, trivially fast

  for(int r = 0; r < num_rules; r++)
  {
    for(int a = 0; a < num_steps; a++)
    {
      const float anchor = (float)a / (float)num_steps;
      const float score  = _score_harmony(smooth, num_bins,
                                          (dt_iop_colorharmonizer_rule_t)r, anchor);
      if(score > best_score)
      {
        best_score   = score;
        *best_rule   = (dt_iop_colorharmonizer_rule_t)r;
        *best_anchor = anchor;
      }
    }
  }
}

static void _auto_detect_callback(GtkButton *button, dt_iop_module_t *self)
{
  dt_iop_colorharmonizer_gui_data_t *g = self->gui_data;
  dt_iop_colorharmonizer_params_t   *p = self->params;

  g_mutex_lock(&g->histogram_lock);
  if(!g->histogram_valid)
  {
    g_mutex_unlock(&g->histogram_lock);
    dt_control_log(_("no histogram available yet — wait for the preview to finish processing"));
    return;
  }
  float histo[COLORHARMONIZER_HUE_BINS];
  memcpy(histo, g->hue_histogram, sizeof(histo));
  g_mutex_unlock(&g->histogram_lock);

  dt_iop_colorharmonizer_rule_t best_rule;
  float best_anchor;
  _auto_detect_harmony(histo, COLORHARMONIZER_HUE_BINS, &best_rule, &best_anchor);

  p->rule       = best_rule;
  p->anchor_hue = best_anchor;

  ++darktable.gui->reset;
  dt_bauhaus_combobox_set(g->rule, p->rule);
  dt_bauhaus_slider_set(g->anchor_hue, p->anchor_hue);
  --darktable.gui->reset;

  dt_iop_gui_changed(DT_ACTION(self), NULL, NULL);
  dt_dev_add_history_item(self->dev, self, TRUE);

  if(g->sync_to_vectorscope
     && gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->sync_to_vectorscope)))
    _push_to_vectorscope(self);
}

static void _set_from_vectorscope_callback(GtkButton *button, dt_iop_module_t *self)
{
  dt_iop_colorharmonizer_params_t *params = (dt_iop_colorharmonizer_params_t *)self->params;
  dt_iop_colorharmonizer_gui_data_t *g = (dt_iop_colorharmonizer_gui_data_t *)self->gui_data;
  dt_color_harmony_guide_t guide;

  dt_lib_histogram_get_harmony(darktable.lib, &guide);

  if(guide.type != DT_COLOR_HARMONY_NONE)
  {
    params->rule = (dt_iop_colorharmonizer_rule_t)(guide.type - 1);
    params->anchor_hue = guide.rotation / 360.0f;

    // update GUI without triggering infinite pipe recursion
    ++darktable.gui->reset;
    dt_bauhaus_combobox_set(g->rule, params->rule);
    dt_bauhaus_slider_set(g->anchor_hue, params->anchor_hue);
    --darktable.gui->reset;

    dt_iop_gui_changed(DT_ACTION(self), NULL, NULL);
    dt_dev_add_history_item(self->dev, self, TRUE);
  }

  // select vectorscope in histogram view
  dt_lib_histogram_set_scope(darktable.lib, 0); // 0 = DT_LIB_HISTOGRAM_SCOPE_VECTORSCOPE
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_colorharmonizer_gui_data_t *g = IOP_GUI_ALLOC(colorharmonizer);
  self->widget = dt_gui_vbox();

  g->rule = dt_bauhaus_combobox_from_params(self, "rule");
  gtk_widget_set_tooltip_text(g->rule,
    _("harmony rule that defines which hues are considered 'in harmony'.\n"
      "\n"
      "monochromatic: a single hue family — only one node.\n"
      "analogous: three adjacent hues spaced 30° apart — naturalistic, cohesive.\n"
      "analogous complementary: analogous triad plus its complement — rich but balanced.\n"
      "complementary: two hues opposite on the wheel — high contrast, vibrant.\n"
      "split complementary: one hue and the two hues flanking its complement — vivid yet less stark.\n"
      "dyad: two hues separated by 60° — gentle contrast.\n"
      "triad: three hues evenly spaced 120° apart — balanced, colorful.\n"
      "tetrad: four hues in two complementary pairs spaced 60° — complex, needs restraint.\n"
      "square: four hues evenly spaced 90° apart — strong and varied.\n"
      "custom: define all four harmony node hues independently."));

  gtk_box_pack_start(GTK_BOX(self->widget),
                     dt_ui_section_label_new(_("Rule controls")), FALSE, FALSE, 0);

  g->anchor_hue = dt_color_picker_new(self, DT_COLOR_PICKER_AREA,
                                      dt_bauhaus_slider_from_params(self, "anchor_hue"));
  dt_bauhaus_slider_set_feedback(g->anchor_hue, 0);
  dt_bauhaus_slider_set_format(g->anchor_hue, "°");
  gtk_widget_set_tooltip_text(g->anchor_hue,
    _("the primary 'key' hue of the harmony — the first node from which all others are derived.\n"
      "\n"
      "drag the slider or use the color picker (eyedropper) to sample a dominant hue directly\n"
      "from the image. the remaining harmony nodes are computed automatically from this hue\n"
      "and the selected rule.\n"
      "\n"
      "tip: pick a skin tone, a sky, or another dominant subject color to anchor the palette."));

  // Node count slider — shown in custom mode only; placed before swatches/rows so it
  // controls how many are visible.
  g->num_custom_nodes_slider = dt_bauhaus_slider_from_params(self, "num_custom_nodes");
  dt_bauhaus_slider_set_digits(g->num_custom_nodes_slider, 0);
  gtk_widget_set_tooltip_text(g->num_custom_nodes_slider,
    _("number of active harmony nodes in custom mode (2–4).\n"
      "nodes are numbered from the top; inactive nodes are hidden."));

  // Horizontal swatches area — shown in non-custom mode (read-only node colors)
  g->swatches_area = dt_gui_hbox();
  for(int i = 0; i < COLORHARMONIZER_MAX_NODES; i++)
  {
    g->node_swatch[i] = gtk_drawing_area_new();
    gtk_widget_set_size_request(g->node_swatch[i],
                                DT_PIXEL_APPLY_DPI(24), DT_PIXEL_APPLY_DPI(24));
    g_object_set_data(G_OBJECT(g->node_swatch[i]), "swatch-index", GINT_TO_POINTER(i));
    g_signal_connect(G_OBJECT(g->node_swatch[i]), "draw",
                     G_CALLBACK(_swatch_draw_callback), self);
    gtk_box_pack_start(GTK_BOX(g->swatches_area), g->node_swatch[i], TRUE, TRUE, 0);
  }
  gtk_box_pack_start(GTK_BOX(self->widget), g->swatches_area, FALSE, FALSE, 0);

  // Vertical swatch rows — one per harmony node (max 4), shown in custom mode only
  for(int i = 0; i < COLORHARMONIZER_MAX_NODES; i++)
  {
    GtkWidget *row = dt_gui_hbox();

    // Color swatch (small square showing the node's hue)
    g->custom_swatch[i] = gtk_drawing_area_new();
    gtk_widget_set_size_request(g->custom_swatch[i],
                                DT_PIXEL_APPLY_DPI(24), DT_PIXEL_APPLY_DPI(24));
    g_object_set_data(G_OBJECT(g->custom_swatch[i]), "swatch-index", GINT_TO_POINTER(i));
    g_signal_connect(G_OBJECT(g->custom_swatch[i]), "draw",
                     G_CALLBACK(_swatch_draw_callback), self);
    gtk_box_pack_start(GTK_BOX(row), g->custom_swatch[i], FALSE, FALSE, 0);

    // Hue slider with color picker
    GtkWidget *slider = dt_bauhaus_slider_new_with_range_and_feedback(
        self, 0.0f, 1.0f, 0.001f, 0.0f, 3, 0);
    dt_bauhaus_slider_set_format(slider, "°");
    g_object_set_data(G_OBJECT(slider), "slider-index", GINT_TO_POINTER(i));
    g_signal_connect(G_OBJECT(slider), "value-changed",
                     G_CALLBACK(_custom_hue_changed), self);
    gtk_widget_set_tooltip_text(slider,
      _("hue of this harmony node (normalized 0–1, i.e. 0°–360°).\n"
        "only active in 'custom' harmony mode.\n"
        "use the color picker to sample the desired hue from the image."));

    // Wrap slider with color picker quad button
    g->custom_hue_slider[i] = dt_color_picker_new(self, DT_COLOR_PICKER_AREA, slider);

    gtk_box_pack_start(GTK_BOX(row), g->custom_hue_slider[i], TRUE, TRUE, 0);

    gtk_box_pack_start(GTK_BOX(self->widget), row, FALSE, FALSE, 0);
    g->custom_row[i] = row;
  }

  // "auto detect" and "set from vectorscope" on one row
  g->actions_box = dt_gui_hbox();
  g->auto_detect = gtk_button_new_with_label(_("auto detect"));
  gtk_widget_set_tooltip_text(g->auto_detect,
    _("analyze the image's hue distribution and automatically suggest the harmony rule\n"
      "and anchor hue that best match its existing color palette.\n"
      "\n"
      "the detection scores every rule and anchor combination against a chroma-weighted\n"
      "histogram of the preview image, then selects the combination that already covers\n"
      "the most chromatic energy — i.e. requires the least correction.\n"
      "\n"
      "the result replaces the current rule and anchor hue. use effect strength to control\n"
      "how strongly the remaining off-palette colors are pulled toward the detected palette."));
  g->set_from_vectorscope = gtk_button_new_with_label(_("set from vectorscope"));
  gtk_widget_set_tooltip_text(g->set_from_vectorscope,
    _("import the harmony rule and anchor hue currently displayed in the vectorscope.\n"
      "also switches the histogram panel to the vectorscope view if it is not already active."));
  gtk_box_pack_start(GTK_BOX(g->actions_box), g->auto_detect, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(g->actions_box), g->set_from_vectorscope, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), g->actions_box, FALSE, FALSE, 0);

  g_signal_connect(G_OBJECT(g->auto_detect), "clicked", G_CALLBACK(_auto_detect_callback), self);
  g_signal_connect(G_OBJECT(g->set_from_vectorscope), "clicked",
                   G_CALLBACK(_set_from_vectorscope_callback), self);

  g->histogram_valid = FALSE;
  g_mutex_init(&g->histogram_lock);

  gtk_box_pack_start(GTK_BOX(self->widget),
                     dt_ui_section_label_new(_("Effect controls")), FALSE, FALSE, 0);

  g->effect_strength = dt_bauhaus_slider_from_params(self, "effect_strength");
  gtk_widget_set_tooltip_text(g->effect_strength,
    _("how strongly off-harmony hues are pulled toward the nearest harmony node.\n"
      "\n"
      "0: no effect — colors are left unchanged.\n"
      "low values (0.1–0.3): subtle nudge, colors lean toward the palette without\n"
      "  losing their original character.\n"
      "high values (0.7–1.0): aggressive correction, most hues converge noticeably\n"
      "  onto the harmony nodes.\n"
      "\n"
      "the shift is weighted by each pixel's chroma: fully desaturated pixels (grays)\n"
      "are never affected regardless of this setting."));

  g->zone_width = dt_bauhaus_slider_from_params(self, "zone_width");
  dt_bauhaus_slider_set_digits(g->zone_width, 2);
  gtk_widget_set_tooltip_text(g->zone_width,
    _("controls the angular reach of each harmony node's attraction.\n"
      "\n"
      "the attraction is Gaussian — strongest at the node center and tapering smoothly\n"
      "outward. zone width scales the standard deviation of this Gaussian linearly.\n"
      "\n"
      "narrow (< 1): only hues very close to a node are pulled — selective, precise.\n"
      "  useful when colors are already near-harmonic and need only a gentle correction.\n"
      "default (1): each node's influence tapers to near-zero at the midpoint between\n"
      "  adjacent nodes — clean separation with smooth transitions.\n"
      "wide (> 1): attraction zones overlap — broader, more global hue shift.\n"
      "  useful for strongly discordant images or when a painterly look is desired."));

  g->protect_neutral = dt_bauhaus_slider_from_params(self, "protect_neutral");
  dt_bauhaus_slider_set_digits(g->protect_neutral, 2);
  gtk_widget_set_tooltip_text(g->protect_neutral,
    _("shields low-chroma (desaturated) colors from the hue correction.\n"
      "\n"
      "0: no protection — grays, skin highlights, and muted tones are all shifted.\n"
      "low values (0.1–0.3): only near-neutral grays are exempted; pastels are still affected.\n"
      "mid values (0.4–0.6): muted and pastel colors are increasingly spared.\n"
      "high values (0.7–1.0): only vivid, saturated colors are corrected; everything else\n"
      "  is left close to its original hue.\n"
      "\n"
      "the weighting is smooth and hyperbolic — there is no hard cutoff. protection grows\n"
      "gradually from zero and the slider response is distributed evenly across its range."));

  // "keep vectorscope in sync" at the bottom
  g->sync_to_vectorscope = gtk_check_button_new_with_label(_("keep vectorscope in sync"));
  gtk_widget_set_tooltip_text(g->sync_to_vectorscope,
    _("when enabled, the vectorscope harmony overlay is updated automatically every time\n"
      "the harmony rule or anchor hue changes in this module.\n"
      "disable to adjust the module without disturbing the vectorscope display."));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->sync_to_vectorscope), TRUE);
  gtk_box_pack_start(GTK_BOX(self->widget), g->sync_to_vectorscope, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->sync_to_vectorscope), "toggled",
                   G_CALLBACK(_sync_to_vectorscope_toggled), self);
}

void gui_cleanup(dt_iop_module_t *self)
{
  dt_iop_colorharmonizer_gui_data_t *g = self->gui_data;
  if(g) g_mutex_clear(&g->histogram_lock);
  dt_iop_default_cleanup(self);
}
