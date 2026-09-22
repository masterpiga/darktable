/*
    This file is part of darktable,
    Copyright (C) 2012-2026 darktable developers.

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
#include "common/gdk_event_utils.h"

#include "develop/blend.h"
#include "develop/blend_gui_internal.h"
#include "bauhaus/bauhaus.h"
#include "common/database.h"
#include "common/debug.h"
#include "common/dtpthread.h"
#include "common/math.h"
#include "common/opencl.h"
#include "common/iop_profile.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "develop/masks.h"
#include "develop/tiling.h"
#include "dtgtk/button.h"
#include "dtgtk/expander.h"
#include "dtgtk/togglebutton.h"
#include "dtgtk/gradientslider.h"
#include "gui/draw.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"
#include "gui/preferences.h"
#include "libs/lib.h"
#include "gui/presets.h"

#include <assert.h>
#include <gmodule.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define NEUTRAL_GRAY 0.5

const dt_introspection_type_enum_tuple_t dt_develop_blend_mode_names[]
    = { { NC_("blendmode", "normal"),
          DEVELOP_BLEND_NORMAL2 },
        { NC_("blendmode", "average"),
          DEVELOP_BLEND_AVERAGE },
        { NC_("blendmode", "difference"),
          DEVELOP_BLEND_DIFFERENCE2 },

        { NC_("blendmode", "normal bounded"),
          DEVELOP_BLEND_BOUNDED },
        { NC_("blendmode", "lighten"),
          DEVELOP_BLEND_LIGHTEN },
        { NC_("blendmode", "darken"),
          DEVELOP_BLEND_DARKEN },
        { NC_("blendmode", "screen"),
          DEVELOP_BLEND_SCREEN },

        { NC_("blendmode", "multiply"),
          DEVELOP_BLEND_MULTIPLY },
        { NC_("blendmode", "divide"),
          DEVELOP_BLEND_DIVIDE },
        { NC_("blendmode", "addition"),
          DEVELOP_BLEND_ADD },
        { NC_("blendmode", "subtract"),
          DEVELOP_BLEND_SUBTRACT },
        { NC_("blendmode", "geometric mean"),
          DEVELOP_BLEND_GEOMETRIC_MEAN },
        { NC_("blendmode", "harmonic mean"),
          DEVELOP_BLEND_HARMONIC_MEAN },

        { NC_("blendmode", "overlay"),
          DEVELOP_BLEND_OVERLAY },
        { NC_("blendmode", "softlight"),
          DEVELOP_BLEND_SOFTLIGHT },
        { NC_("blendmode", "hardlight"),
          DEVELOP_BLEND_HARDLIGHT },
        { NC_("blendmode", "vividlight"),
          DEVELOP_BLEND_VIVIDLIGHT },
        { NC_("blendmode", "linearlight"),
          DEVELOP_BLEND_LINEARLIGHT },
        { NC_("blendmode", "pinlight"),
          DEVELOP_BLEND_PINLIGHT },

        { NC_("blendmode", "lightness"),
          DEVELOP_BLEND_LIGHTNESS },
        { NC_("blendmode", "chromaticity"),
          DEVELOP_BLEND_CHROMATICITY },

        { NC_("blendmode", "Lab lightness"),
          DEVELOP_BLEND_LAB_LIGHTNESS },
        { NC_("blendmode", "Lab a-channel"),
          DEVELOP_BLEND_LAB_A },
        { NC_("blendmode", "Lab b-channel"),
          DEVELOP_BLEND_LAB_B },
        { NC_("blendmode", "Lab color"),
          DEVELOP_BLEND_LAB_COLOR },

        { NC_("blendmode", "RGB red channel"),
          DEVELOP_BLEND_RGB_R },
        { NC_("blendmode", "RGB green channel"),
          DEVELOP_BLEND_RGB_G },
        { NC_("blendmode", "RGB blue channel"),
          DEVELOP_BLEND_RGB_B },
        { NC_("blendmode", "HSV value"),
          DEVELOP_BLEND_HSV_VALUE },
        { NC_("blendmode", "HSV color"),
          DEVELOP_BLEND_HSV_COLOR },

        { NC_("blendmode", "hue"),
          DEVELOP_BLEND_HUE },
        { NC_("blendmode", "color"),
          DEVELOP_BLEND_COLOR },
        { NC_("blendmode", "coloradjustment"),
          DEVELOP_BLEND_COLORADJUST },

        /** deprecated blend modes: make them available as legacy
         * history stacks might want them */

        { NC_("blendmode", "difference (deprecated)"),
          DEVELOP_BLEND_DIFFERENCE },
        { NC_("blendmode", "subtract inverse (deprecated)"),
          DEVELOP_BLEND_SUBTRACT_INVERSE },
        { NC_("blendmode", "divide inverse (deprecated)"),
          DEVELOP_BLEND_DIVIDE_INVERSE },
        { NC_("blendmode", "Lab L-channel (deprecated)"),
          DEVELOP_BLEND_LAB_L },
        { } };

const dt_introspection_type_enum_tuple_t dt_develop_blend_mode_flag_names[]
    = { { NC_("blendoperation", "normal"), 0 },
        { NC_("blendoperation", "reverse"), DEVELOP_BLEND_REVERSE },
        { } };

const dt_introspection_type_enum_tuple_t dt_develop_blend_colorspace_names[]
    = { { N_("default"),
          DEVELOP_BLEND_CS_NONE },
        { N_("RAW"),
          DEVELOP_BLEND_CS_RAW },
        { N_("Lab"),
          DEVELOP_BLEND_CS_LAB },
        { N_("RGB (display)"),
          DEVELOP_BLEND_CS_RGB_DISPLAY },
        { N_("RGB (scene)"),
          DEVELOP_BLEND_CS_RGB_SCENE },
        { } };

const dt_introspection_type_enum_tuple_t dt_develop_mask_mode_names[] = {
  { N_("off"), DEVELOP_MASK_DISABLED },
  { N_("uniformly"), DEVELOP_MASK_ENABLED },
  { N_("drawn mask"), DEVELOP_MASK_MASK | DEVELOP_MASK_ENABLED },
  { N_("parametric mask"), DEVELOP_MASK_CONDITIONAL | DEVELOP_MASK_ENABLED },
  { N_("raster mask"), DEVELOP_MASK_RASTER | DEVELOP_MASK_ENABLED },
  { N_("drawn & parametric mask"), DEVELOP_MASK_MASK_CONDITIONAL | DEVELOP_MASK_ENABLED },
  { N_("flexi mask"), DEVELOP_MASK_FLEXI | DEVELOP_MASK_ENABLED },
  {}
};

const dt_introspection_type_enum_tuple_t dt_develop_combine_masks_names[]
    = { { N_("exclusive"),            DEVELOP_COMBINE_NORM_EXCL },
        { N_("inclusive"),            DEVELOP_COMBINE_NORM_INCL },
        { N_("exclusive & inverted"), DEVELOP_COMBINE_INV_EXCL },
        { N_("inclusive & inverted"), DEVELOP_COMBINE_INV_INCL },
        { } };

const dt_introspection_type_enum_tuple_t dt_develop_feathering_guide_names[]
    = { { N_("output before blur"), DEVELOP_MASK_GUIDE_OUT_BEFORE_BLUR },
        { N_("input before blur"),  DEVELOP_MASK_GUIDE_IN_BEFORE_BLUR },
        { N_("output after blur"),  DEVELOP_MASK_GUIDE_OUT_AFTER_BLUR },
        { N_("input after blur"),   DEVELOP_MASK_GUIDE_IN_AFTER_BLUR },
        { } };

const dt_introspection_type_enum_tuple_t dt_develop_invert_mask_names[]
    = { { N_("off"), DEVELOP_COMBINE_NORM },
        { N_("on"), DEVELOP_COMBINE_INV },
        { } };

const dt_iop_gui_blendif_colorstop_t _gradient_L[]
    = { { 0.0f,   { 0, 0, 0, 1.0 } },
        { 0.125f, { NEUTRAL_GRAY / 8, NEUTRAL_GRAY / 8, NEUTRAL_GRAY / 8, 1.0 } },
        { 0.25f,  { NEUTRAL_GRAY / 4, NEUTRAL_GRAY / 4, NEUTRAL_GRAY / 4, 1.0 } },
        { 0.5f,   { NEUTRAL_GRAY / 2, NEUTRAL_GRAY / 2, NEUTRAL_GRAY / 2, 1.0 } },
        { 1.0f,   { NEUTRAL_GRAY, NEUTRAL_GRAY, NEUTRAL_GRAY, 1.0 } } };

// The values for "a" are generated in the following way:
//   Lab (with L=[90 to 68], b=0, and a=[-56 to 56]
//    -> sRGB (D65 linear) -> normalize with MAX(R,G,B) = 0.75
const dt_iop_gui_blendif_colorstop_t _gradient_a[] = {
    { 0.000f, { 0.0112790f, 0.7500000f, 0.5609999f, 1.0f } },
    { 0.250f, { 0.2888855f, 0.7500000f, 0.6318934f, 1.0f } },
    { 0.375f, { 0.4872486f, 0.7500000f, 0.6825501f, 1.0f } },
    { 0.500f, { 0.7500000f, 0.7499399f, 0.7496052f, 1.0f } },
    { 0.625f, { 0.7500000f, 0.5054633f, 0.5676756f, 1.0f } },
    { 0.750f, { 0.7500000f, 0.3423850f, 0.4463195f, 1.0f } },
    { 1.000f, { 0.7500000f, 0.1399815f, 0.2956989f, 1.0f } },
};

// The values for "b" are generated in the following way:
//   Lab (with L=[58 to 62], a=0, and b=[-65 to 65]
//    -> sRGB (D65 linear) -> normalize with MAX(R,G,B) = 0.75
const dt_iop_gui_blendif_colorstop_t _gradient_b[] = {
    { 0.000f, { 0.0162050f, 0.1968228f, 0.7500000f, 1.0f } },
    { 0.250f, { 0.2027354f, 0.3168822f, 0.7500000f, 1.0f } },
    { 0.375f, { 0.3645722f, 0.4210476f, 0.7500000f, 1.0f } },
    { 0.500f, { 0.6167146f, 0.5833379f, 0.7500000f, 1.0f } },
    { 0.625f, { 0.7500000f, 0.6172369f, 0.5412091f, 1.0f } },
    { 0.750f, { 0.7500000f, 0.5590797f, 0.3071980f, 1.0f } },
    { 1.000f, { 0.7500000f, 0.4963975f, 0.0549797f, 1.0f } },
};

const dt_iop_gui_blendif_colorstop_t _gradient_gray[]
    = { { 0.0f,   { 0, 0, 0, 1.0 } },
        { 0.125f, { NEUTRAL_GRAY / 8, NEUTRAL_GRAY / 8, NEUTRAL_GRAY / 8, 1.0 } },
        { 0.25f,  { NEUTRAL_GRAY / 4, NEUTRAL_GRAY / 4, NEUTRAL_GRAY / 4, 1.0 } },
        { 0.5f,   { NEUTRAL_GRAY / 2, NEUTRAL_GRAY / 2, NEUTRAL_GRAY / 2, 1.0 } },
        { 1.0f,   { NEUTRAL_GRAY, NEUTRAL_GRAY, NEUTRAL_GRAY, 1.0 } } };

const dt_iop_gui_blendif_colorstop_t _gradient_red[] = {
    { 0.000f, { 0.0000000f, 0.0000000f, 0.0000000f, 1.0f } },
    { 0.125f, { 0.0937500f, 0.0000000f, 0.0000000f, 1.0f } },
    { 0.250f, { 0.1875000f, 0.0000000f, 0.0000000f, 1.0f } },
    { 0.500f, { 0.3750000f, 0.0000000f, 0.0000000f, 1.0f } },
    { 1.000f, { 0.7500000f, 0.0000000f, 0.0000000f, 1.0f } }
};

const dt_iop_gui_blendif_colorstop_t _gradient_green[] = {
    { 0.000f, { 0.0000000f, 0.0000000f, 0.0000000f, 1.0f } },
    { 0.125f, { 0.0000000f, 0.0937500f, 0.0000000f, 1.0f } },
    { 0.250f, { 0.0000000f, 0.1875000f, 0.0000000f, 1.0f } },
    { 0.500f, { 0.0000000f, 0.3750000f, 0.0000000f, 1.0f } },
    { 1.000f, { 0.0000000f, 0.7500000f, 0.0000000f, 1.0f } }
};

const dt_iop_gui_blendif_colorstop_t _gradient_blue[] = {
    { 0.000f, { 0.0000000f, 0.0000000f, 0.0000000f, 1.0f } },
    { 0.125f, { 0.0000000f, 0.0000000f, 0.0937500f, 1.0f } },
    { 0.250f, { 0.0000000f, 0.0000000f, 0.1875000f, 1.0f } },
    { 0.500f, { 0.0000000f, 0.0000000f, 0.3750000f, 1.0f } },
    { 1.000f, { 0.0000000f, 0.0000000f, 0.7500000f, 1.0f } }
};

// The chroma values are displayed in a gradient from {0.5,0.5,0.5} to {0.5,0.0,0.5} (pink)
const dt_iop_gui_blendif_colorstop_t _gradient_chroma[] = {
    { 0.000f, { 0.5000000f, 0.5000000f, 0.5000000f, 1.0f } },
    { 0.125f, { 0.5000000f, 0.4375000f, 0.5000000f, 1.0f } },
    { 0.250f, { 0.5000000f, 0.3750000f, 0.5000000f, 1.0f } },
    { 0.500f, { 0.5000000f, 0.2500000f, 0.5000000f, 1.0f } },
    { 1.000f, { 0.5000000f, 0.0000000f, 0.5000000f, 1.0f } }
};

// The hue values for LCh are generated in the following way:
//   LCh (with L=65 and C=37) -> sRGB (D65 linear) -> normalize with MAX(R,G,B) = 0.75
// Please keep in sync with the display in the gamma module
const dt_iop_gui_blendif_colorstop_t _gradient_LCh_hue[] = {
    { 0.000f, { 0.7500000f, 0.2200405f, 0.4480174f, 1.0f } },
    { 0.104f, { 0.7500000f, 0.2475123f, 0.2488547f, 1.0f } },
    { 0.200f, { 0.7500000f, 0.3921083f, 0.2017670f, 1.0f } },
    { 0.295f, { 0.7500000f, 0.7440329f, 0.3011876f, 1.0f } },
    { 0.377f, { 0.3813996f, 0.7500000f, 0.3799668f, 1.0f } },
    { 0.503f, { 0.0747526f, 0.7500000f, 0.7489037f, 1.0f } },
    { 0.650f, { 0.0282981f, 0.3736209f, 0.7500000f, 1.0f } },
    { 0.803f, { 0.2583821f, 0.2591069f, 0.7500000f, 1.0f } },
    { 0.928f, { 0.7500000f, 0.2788102f, 0.7492077f, 1.0f } },
    { 1.000f, { 0.7500000f, 0.2200405f, 0.4480174f, 1.0f } },
};

// The hue values for HSL are generated in the following way:
//   HSL (with S=0.5 and L=0.5) -> any RGB(linear) -> (normalize with MAX(R,G,B) = 0.75)
// Please keep in sync with the display in the gamma module
const dt_iop_gui_blendif_colorstop_t _gradient_HSL_hue[] = {
    { 0.000f, { 0.7500000f, 0.2500000f, 0.2500000f, 1.0f } },
    { 0.167f, { 0.7500000f, 0.7500000f, 0.2500000f, 1.0f } },
    { 0.333f, { 0.2500000f, 0.7500000f, 0.2500000f, 1.0f } },
    { 0.500f, { 0.2500000f, 0.7500000f, 0.7500000f, 1.0f } },
    { 0.667f, { 0.2500000f, 0.2500000f, 0.7500000f, 1.0f } },
    { 0.833f, { 0.7500000f, 0.2500000f, 0.7500000f, 1.0f } },
    { 1.000f, { 0.7500000f, 0.2500000f, 0.2500000f, 1.0f } },
};

// The hue values for JzCzhz are generated in the following way:
//   JzCzhz (with Jz=0.011 and Cz=0.01) -> sRGB(D65 linear)
//     -> normalize with MAX(R,G,B) = 0.75
// Please keep in sync with the display in the gamma module
const dt_iop_gui_blendif_colorstop_t _gradient_JzCzhz_hue[] = {
    { 0.000f, { 0.7500000f, 0.1946971f, 0.3697612f, 1.0f } },
    { 0.082f, { 0.7500000f, 0.2278141f, 0.2291548f, 1.0f } },
    { 0.150f, { 0.7500000f, 0.3132381f, 0.1653960f, 1.0f } },
    { 0.275f, { 0.7483232f, 0.7500000f, 0.1939316f, 1.0f } },
    { 0.378f, { 0.2642865f, 0.7500000f, 0.2642768f, 1.0f } },
    { 0.570f, { 0.0233180f, 0.7493543f, 0.7500000f, 1.0f } },
    { 0.650f, { 0.1119025f, 0.5116763f, 0.7500000f, 1.0f } },
    { 0.762f, { 0.3331225f, 0.3337235f, 0.7500000f, 1.0f } },
    { 0.883f, { 0.7464700f, 0.2754816f, 0.7500000f, 1.0f } },
    { 1.000f, { 0.7500000f, 0.1946971f, 0.3697612f, 1.0f } },
};

enum _channel_indexes
{
  CHANNEL_INDEX_L = 0,
  CHANNEL_INDEX_a = 1,
  CHANNEL_INDEX_b = 2,
  CHANNEL_INDEX_C = 3,
  CHANNEL_INDEX_h = 4,
  CHANNEL_INDEX_g = 0,
  CHANNEL_INDEX_R = 1,
  CHANNEL_INDEX_G = 2,
  CHANNEL_INDEX_B = 3,
  CHANNEL_INDEX_H = 4,
  CHANNEL_INDEX_S = 5,
  CHANNEL_INDEX_l = 6,
  CHANNEL_INDEX_Jz = 4,
  CHANNEL_INDEX_Cz = 5,
  CHANNEL_INDEX_hz = 6,
};

dt_masks_form_t *_module_mask_group(dt_iop_module_t *module);
dt_masks_point_group_t *_group_point(dt_masks_form_t *grp, const dt_mask_id_t id);
static gboolean _module_has_drawn_shapes(const dt_iop_module_t *module);
static void _queue_masks_list_rebuild(dt_iop_module_t *module);
static void _queue_link_peers_rebuild(const dt_iop_module_t *module);
static void _auto_expand_selected_row(dt_iop_module_t *module, const dt_mask_id_t id);

static gboolean _blendif_blend_parameter_enabled(dt_develop_blend_colorspace_t csp,
                                                 const dt_develop_blend_mode_t mode)
{
  if(csp == DEVELOP_BLEND_CS_RGB_SCENE)
  {
    switch(mode & ~DEVELOP_BLEND_REVERSE)
    {
      case DEVELOP_BLEND_ADD:
      case DEVELOP_BLEND_MULTIPLY:
      case DEVELOP_BLEND_SUBTRACT:
      case DEVELOP_BLEND_SUBTRACT_INVERSE:
      case DEVELOP_BLEND_DIVIDE:
      case DEVELOP_BLEND_DIVIDE_INVERSE:
      case DEVELOP_BLEND_RGB_R:
      case DEVELOP_BLEND_RGB_G:
      case DEVELOP_BLEND_RGB_B:
        return TRUE;
      default:
        return FALSE;
    }
  }
  return FALSE;
}

// core boost-factor lookup, parameterized on an explicit boost-factors array
// and channel-descriptor array instead of a fixed dt_iop_gui_blend_data_t --
// shared by the module-wide shared editor (bp/data->channel) and per-row
// parametric-form editors (p->blendif_boost_factors/dt_develop_blendif_channels_for_csp),
// see _param_row_boost_factor (per-row).
static inline float _get_boost_factor_ex(const float *blendif_boost_factors,
                                         const dt_iop_gui_blendif_channel_t *channels,
                                         const int channel,
                                         const int in_out)
{
  return exp2f(blendif_boost_factors[channels[channel].param_channels[in_out]]);
}

// normalize a raw picked pixel into each channel's [0,1] display range,
// boost-factor corrected. Parameterized on boost_factors/channels (see
// _get_boost_factor_ex) so it works identically for the module-wide shared
// editor and for a per-row parametric-form editor (whose boost factors live
// in that form's own dt_masks_point_parametric_t, not the module's bp).
static void _blendif_scale_ex(const float *blendif_boost_factors,
                              const dt_iop_gui_blendif_channel_t *channels,
                              dt_iop_colorspace_type_t cst,
                              const float *in,
                              float *out,
                              const dt_iop_order_iccprofile_info_t *work_profile,
                              const int in_out)
{
  out[0] = out[1] = out[2] = out[3] = out[4] = out[5] = out[6] = out[7] = -1.0f;

#define BOOST(idx) _get_boost_factor_ex(blendif_boost_factors, channels, idx, in_out)

  switch(cst)
  {
    case IOP_CS_LAB:
      out[CHANNEL_INDEX_L] = (in[0] / BOOST(0)) / 100.0f;
      out[CHANNEL_INDEX_a] = ((in[1] / BOOST(1)) + 128.0f) / 256.0f;
      out[CHANNEL_INDEX_b] = ((in[2] / BOOST(2)) + 128.0f) / 256.0f;
      break;
    case IOP_CS_RGB:
      if(work_profile == NULL)
        out[CHANNEL_INDEX_g] = 0.3f * in[0] + 0.59f * in[1] + 0.11f * in[2];
      else
        out[CHANNEL_INDEX_g] = dt_ioppr_get_rgb_matrix_luminance
          (in, work_profile->matrix_in,
           work_profile->lut_in,
           work_profile->unbounded_coeffs_in,
           work_profile->lutsize,
           work_profile->nonlinearlut);
      out[CHANNEL_INDEX_g] = out[CHANNEL_INDEX_g] / BOOST(0);
      out[CHANNEL_INDEX_R] = in[0] / BOOST(1);
      out[CHANNEL_INDEX_G] = in[1] / BOOST(2);
      out[CHANNEL_INDEX_B] = in[2] / BOOST(3);
      break;
    case IOP_CS_LCH:
      out[CHANNEL_INDEX_C] = (in[1] / BOOST(3)) / (128.0f * M_SQRT2_F);
      out[CHANNEL_INDEX_h] = in[2] / BOOST(4);
      break;
    case IOP_CS_HSL:
      out[CHANNEL_INDEX_H] = in[0] / BOOST(4);
      out[CHANNEL_INDEX_S] = in[1] / BOOST(5);
      out[CHANNEL_INDEX_l] = in[2] / BOOST(6);
      break;
    case IOP_CS_JZCZHZ:
      out[CHANNEL_INDEX_Jz] = in[0] / BOOST(4);
      out[CHANNEL_INDEX_Cz] = in[1] / BOOST(5);
      out[CHANNEL_INDEX_hz] = in[2] / BOOST(6);
      break;
    default:
      break;
  }
#undef BOOST
}

static void _blendif_cook(const dt_iop_colorspace_type_t cst,
                          const float *in,
                          float *out,
                          const dt_iop_order_iccprofile_info_t *const work_profile)
{
  out[0] = out[1] = out[2] = out[3] = out[4] = out[5] = out[6] = out[7] = -1.0f;

  switch(cst)
  {
    case IOP_CS_LAB:
      out[CHANNEL_INDEX_L] = in[0];
      out[CHANNEL_INDEX_a] = in[1];
      out[CHANNEL_INDEX_b] = in[2];
      break;
    case IOP_CS_RGB:
      if(work_profile == NULL)
        out[CHANNEL_INDEX_g] = (0.3f * in[0] + 0.59f * in[1] + 0.11f * in[2]) * 100.0f;
      else
        out[CHANNEL_INDEX_g] = dt_ioppr_get_rgb_matrix_luminance
          (in, work_profile->matrix_in,
           work_profile->lut_in,
           work_profile->unbounded_coeffs_in,
           work_profile->lutsize,
           work_profile->nonlinearlut) * 100.0f;
      out[CHANNEL_INDEX_R] = in[0] * 100.0f;
      out[CHANNEL_INDEX_G] = in[1] * 100.0f;
      out[CHANNEL_INDEX_B] = in[2] * 100.0f;
      break;
    case IOP_CS_LCH:
      out[CHANNEL_INDEX_C] = in[1] / (128.0f * M_SQRT2_F) * 100.0f;
      out[CHANNEL_INDEX_h] = in[2] * 360.0f;
      break;
    case IOP_CS_HSL:
      out[CHANNEL_INDEX_H] = in[0] * 360.0f;
      out[CHANNEL_INDEX_S] = in[1] * 100.0f;
      out[CHANNEL_INDEX_l] = in[2] * 100.0f;
      break;
    case IOP_CS_JZCZHZ:
      out[CHANNEL_INDEX_Jz] = in[0] * 100.0f;
      out[CHANNEL_INDEX_Cz] = in[1] * 100.0f;
      out[CHANNEL_INDEX_hz] = in[2] * 360.0f;
      break;
    default:
      break;
  }
}

static inline int _blendif_print_digits_default(const float value)
{
  int digits;
  if(value < 0.0001f) digits = 0;
  else if(value < 0.01f) digits = 2;
  else if(value < 0.999f) digits = 1;
  else digits = 0;

  return digits;
}

static inline int _blendif_print_digits_ab(const float value)
{
  int digits;
  if(fabsf(value) < 10.0f) digits = 1;
  else digits = 0;

  return digits;
}

static void _blendif_scale_print_ab(const float value,
                                    const float boost_factor,
                                    char *string,
                                    int n)
{
  const float scaled = (value * 256.0f - 128.0f) * boost_factor;
  snprintf(string, n, "%-5.*f", _blendif_print_digits_ab(scaled), scaled);
}

static void _blendif_scale_print_hue(const float value,
                                     const float boost_factor,
                                     char *string,
                                     const int n)
{
  snprintf(string, n, "%-5.0f", value * 360.0f);
}

static void _blendif_scale_print_default(const float value,
                                         const float boost_factor,
                                         char *string,
                                         const int n)
{
  const float scaled = value * boost_factor;
  snprintf(string, n, "%-5.*f", _blendif_print_digits_default(scaled), scaled * 100.0f);
}

static void _add_wrapped_box(GtkWidget *container,
                             GtkBox *box,
                             gchar *help_url)
{
  GtkWidget *event_box = gtk_event_box_new();
  GtkWidget *revealer = gtk_revealer_new();
  gtk_container_add(GTK_CONTAINER(revealer), GTK_WIDGET(box));
  gtk_container_add(GTK_CONTAINER(event_box), revealer);
  gtk_container_add(GTK_CONTAINER(container), event_box);
  // event box is needed so that one can click into the area to get help
  dt_gui_add_help_link(event_box, help_url);
  gtk_widget_set_name(GTK_WIDGET(box), "blending-box");
}

static void _box_set_visible(GtkBox *box, gboolean visible)
{
  if(!box) return;

  GtkRevealer *revealer = GTK_REVEALER(gtk_widget_get_parent(GTK_WIDGET(box)));
  gtk_revealer_set_transition_duration(revealer,
                                       dt_conf_get_int("darkroom/ui/transition_duration"));
  gtk_revealer_set_reveal_child(revealer, visible);
}



// re-home a widget into a new parent (no-op if already there), preserving its
// shown state. Used by the panel host to move the whole flexi panel between
// its possible homes (see masks_gui_panel_host.c)
void _reparent_into(GtkWidget *w,
                    GtkWidget *parent,
                    const gboolean at_end,
                    const gboolean expand)
{
  if(!w || !parent || !GTK_IS_WIDGET(w) || !GTK_IS_BOX(parent)) return;
  GtkWidget *cur = gtk_widget_get_parent(w);
  if(cur == parent) return;

  const gboolean was_visible = gtk_widget_get_visible(w);

  g_object_ref(w);
  if(cur) gtk_container_remove(GTK_CONTAINER(cur), w);

  if(at_end)
    gtk_box_pack_end(GTK_BOX(parent), w, expand, expand, 0);
  else
  {
    if(expand) gtk_widget_set_hexpand(w, TRUE);
    dt_gui_box_add(parent, w);
  }

  if(was_visible) gtk_widget_show(w);
  g_object_unref(w);
}

// an expanding, zero-content spacer: takes whatever width the row has left
// over, so the runs of buttons either side of it are pushed apart
static GtkWidget *_pack_stretch(GtkWidget *box)
{
  GtkWidget *stretch = dt_gui_hbox();
  gtk_widget_show(stretch);
  dt_gui_box_add(box, dt_gui_expand(stretch));
  return stretch;
}

// a fixed spacer that only separates two runs of buttons, without competing
// for the row's slack: one icon wide, from .mask-row-gap in darktable.css
static GtkWidget *_pack_gap(GtkWidget *box)
{
  GtkWidget *gap = dt_gui_hbox();
  dt_gui_add_class(gap, "mask-row-gap");
  gtk_widget_show(gap);
  dt_gui_box_add(box, gap);
  return gap;
}

// defined much further down (grouping shape rows / naming clusters); forward
// declared here so the import menu can group its shapes by kind the same way
// the mask list clusters same-kind elements
static guint _form_kind(const dt_masks_form_t *form);
static const char *_kind_name(const guint kind, const gboolean plural);
// defined further up (module.c-adjacent helpers); forward declared here so
// the import menu can look up which module (if any) currently uses a form.
void _build_masks_list(dt_iop_module_t *module);
// defined further down, with the rest of the raster element, group naming and
// refinement code
static void _add_raster_mask(dt_iop_module_t *self,
                             dt_iop_module_t *src,
                             const dt_mask_id_t id);
static const char *_group_custom_name(dt_masks_form_t *grp, const dt_mask_id_t cid);
static const char *_within_name(const dt_masks_state_t within);
static void _flexi_refine_follow_selection(dt_iop_gui_blend_data_t *bd);
void _refresh_canvas_edit(dt_iop_module_t *module);
static dt_mask_id_t _mask_group_cid(dt_iop_module_t *module);
static void _select_mask_group_if_none(dt_iop_gui_blend_data_t *bd);

// ---- linking and copying elements between modules' masks -----------------
// A shape or AI object can sit in several modules' masks at once: each mask's
// point refers to the same form, so editing the form changes it in all of
// them, while the point's own opacity, operator, invert state and refinement
// stay per module. That is a link. A copy is a new form, independent from the
// start. Parametric channels are only ever copied: the same thresholds select
// something else in another module's pixels

// a shape or AI object: what can be linked. A parametric or raster element
// is not, and neither is a clone/heal source nor a whole group
static gboolean _form_is_shape(const dt_masks_form_t *f)
{
  return f
         && !(f->type & (DT_MASKS_GROUP | DT_MASKS_CLONE | DT_MASKS_NON_CLONE
                         | DT_MASKS_PARAMETRIC | DT_MASKS_RASTER))
         && _form_kind(f);
}

GList *_model_form_users(const dt_mask_id_t fid)
{
  GList *users = NULL;
  for(GList *l = darktable.develop ? darktable.develop->iop : NULL; l; l = g_list_next(l))
  {
    dt_iop_module_t *m = l->data;
    dt_masks_form_t *grp = _module_mask_group(m);
    if(grp && _group_point(grp, fid)) users = g_list_append(users, m);
  }
  return users;
}

// append the shapes of the nested group `grp`, at any depth, bottom-up
static void _append_nested_shapes(const dt_masks_form_t *grp, GList **out, const int depth)
{
  if(!grp || depth > DT_MASKS_NESTING_MAX) return;
  for(const GList *l = grp->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    if(dt_masks_point_is_marker(pt)) continue;
    const dt_masks_form_t *f = dt_masks_get_from_id(darktable.develop, pt->formid);
    if(f && f != grp && (f->type & DT_MASKS_GROUP))
      _append_nested_shapes(f, out, depth + 1);
    else if(_form_is_shape(f))
      *out = g_list_append(*out, GINT_TO_POINTER(pt->formid));
  }
}

GList *_model_module_shapes(dt_iop_module_t *src, const dt_mask_id_t cid)
{
  dt_masks_form_t *grp = _module_mask_group(src);
  GList *out = NULL;
  gboolean in_run = !dt_is_valid_maskid(cid);
  for(GList *l = grp ? grp->points : NULL; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    if(dt_is_valid_maskid(cid) && _starts_group(l)) in_run = pt->formid == cid;
    if(!in_run || _starts_group(l)) continue;
    const dt_masks_form_t *f = dt_masks_get_from_id(darktable.develop, pt->formid);
    // a nested group's shapes are the group's
    if(f && (f->type & DT_MASKS_GROUP))
      _append_nested_shapes(f, &out, 1);
    else if(_form_is_shape(f))
      out = g_list_append(out, GINT_TO_POINTER(pt->formid));
  }
  return out;
}

GList *_model_import_forms(dt_iop_module_t *module,
                           dt_iop_module_t *src,
                           GList *fids,
                           const gboolean copy)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *src_grp = src ? _module_mask_group(src) : NULL;
  GList *added = NULL;
  for(GList *l = fids; l; l = g_list_next(l))
  {
    const dt_mask_id_t fid = GPOINTER_TO_INT(l->data);
    dt_masks_form_t *grp = _module_mask_group(module);
    if(grp && _group_point(grp, fid)) continue;

    const dt_mask_id_t id = copy ? dt_masks_form_copy(darktable.develop, fid) : fid;
    dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, id);
    if(!form) continue;
    dt_masks_point_group_t *pt = dt_masks_group_insert_point(darktable.develop, module, form);
    if(!pt) continue;

    // it looks the way it does in the mask it comes from; the operator is the
    // target group's, set by the insertion above
    const dt_masks_point_group_t *spt = src_grp ? _group_point(src_grp, fid) : NULL;
    if(spt)
    {
      pt->opacity = spt->opacity;
      pt->state = (pt->state & ~DT_MASKS_STATE_INVERSE) | (spt->state & DT_MASKS_STATE_INVERSE);
      pt->refinement = spt->refinement;
    }

    // the next one lands above this one, in the same group
    if(bd && bd->insert_active) bd->insert_after_fid = id;
    added = g_list_append(added, GINT_TO_POINTER(id));
  }
  return added;
}

// move a hash table entry keyed by form id over to another id
static void _remap_formid_key(GHashTable *table, const dt_mask_id_t from, const dt_mask_id_t to)
{
  gpointer value = NULL;
  if(table && g_hash_table_lookup_extended(table, GINT_TO_POINTER(from), NULL, &value))
  {
    g_hash_table_steal(table, GINT_TO_POINTER(from));
    g_hash_table_insert(table, GINT_TO_POINTER(to), value);
  }
}

// defined further down, next to the row index it walks
static int _model_form_uses_in_mask(dt_iop_module_t *module, const dt_mask_id_t fid);

dt_mask_id_t _model_unlink_form_point(dt_iop_module_t *module,
                                      const dt_mask_id_t fid,
                                      dt_masks_point_group_t *pt)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!pt) pt = grp ? _group_point(grp, fid) : NULL;
  if(!pt) return INVALID_MASKID;
  const dt_mask_id_t nid = dt_masks_form_copy(darktable.develop, fid);
  if(!dt_is_valid_maskid(nid)) return INVALID_MASKID;
  pt->formid = nid;

  // the panel knows elements by id: carry each reference over, or the copy
  // loses its selection and its expanded state. Groups are known by their
  // markers, which an element's unlinking leaves alone.
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  // where the mask still holds another reference to the original, that state
  // describes a row that is still there: moving it to the copy would take the
  // selection and the expanded state away from the row that kept the shape
  if(!bd || _model_form_uses_in_mask(module, fid) > 0) return nid;
  dt_mask_id_t *refs[] = {
    &bd->panel_selected_formid,    &bd->solo_formid,
    &bd->soloedit_formid,          &bd->masks_last_expanded_elem,
    &bd->masks_refine_scope_formid, &bd->insert_after_fid,
  };
  for(size_t k = 0; k < G_N_ELEMENTS(refs); k++)
    if(*refs[k] == fid) *refs[k] = nid;
  _remap_formid_key(bd->masks_props_expanded, fid, nid);
  return nid;
}

// unlink whichever reference this module's mask holds first. A mask that holds
// the shape once -- every case but a within-mask link -- has only that one
dt_mask_id_t _model_unlink_form(dt_iop_module_t *module, const dt_mask_id_t fid)
{
  return _model_unlink_form_point(module, fid, NULL);
}

// what a pick in the import menu does. Its target carries `a` and `b`, as
// noted per op; a source module travels as an index into the menu's module
// table (see _masks_import_module_index), since a GVariant cannot carry it
typedef enum _masks_import_op_t
{
  _IMPORT_LINK_ONE = 0,    // a: source module (-1: none), b: the form
  _IMPORT_COPY_ONE,
  _IMPORT_LINK_GROUP,      // a: source module, b: its group's head (INVALID_MASKID: all)
  _IMPORT_COPY_GROUP,
  _IMPORT_COPY_PARAMETRIC, // a: source module, b: the form
  _IMPORT_ADD_MASK,        // a: raster source (see _raster_sources_collect)
  _IMPORT_USE_MASK,
} _masks_import_op_t;

typedef struct _masks_raster_source_entry_t
{
  dt_iop_module_t *src;
  dt_mask_id_t id;
  char *name;
} _masks_raster_source_entry_t;

static void _raster_source_entry_free(gpointer data)
{
  _masks_raster_source_entry_t *entry = data;
  if(entry)
  {
    g_free(entry->name);
    g_free(entry);
  }
}

// every mask another module offers as a raster source, in pipe order: those
// upstream of `module` into `usable`, those downstream, which are processed
// after it and so never available to it, into `later`
static void _raster_sources_collect(dt_iop_module_t *module, GPtrArray *usable, GPtrArray *later)
{
  gboolean past = FALSE;
  for(GList *iter = darktable.develop->iop; iter; iter = g_list_next(iter))
  {
    dt_iop_module_t *iop = iter->data;
    if(iop == module)
    {
      past = TRUE;
      continue;
    }
    if(!iop->raster_mask.source.masks) continue;

    GHashTableIter masks_iter;
    gpointer key, value;
    g_hash_table_iter_init(&masks_iter, iop->raster_mask.source.masks);
    while(g_hash_table_iter_next(&masks_iter, &key, &value))
    {
      // the mask's available identifier (module display name, or the mask
      // name/path for an external source): the same string the whole-mask
      // raster picker shows (see _raster_combo_populate / dt_iop_advertise_rastermask)
      _masks_raster_source_entry_t *entry = g_new0(_masks_raster_source_entry_t, 1);
      entry->src = iop;
      entry->id = GPOINTER_TO_INT(key);
      entry->name = g_strdup(value ? (const char *)value : iop->name());
      g_ptr_array_add(past ? later : usable, entry);
    }
  }
}

// commit shapes or parametric channels brought in from another module
static void _masks_import_forms(dt_iop_module_t *module,
                                dt_iop_module_t *src,
                                GList *fids,
                                const gboolean copy)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  GList *added = _model_import_forms(module, src, fids, copy);
  if(!added) return;
  const dt_mask_id_t last = GPOINTER_TO_INT(g_list_last(added)->data);
  g_list_free(added);
  dt_print(DT_DEBUG_MASKS, "[masks] %s %d element(s) into '%s'", copy ? "copied" : "linked",
           g_list_length(fids), module->op);

  bd->panel_selected_formid = last;
  if(darktable.develop->form_gui) darktable.develop->form_gui->panel_selected_formid = last;
  dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
  _queue_link_peers_rebuild(module);
  dt_masks_iop_update(module);
  dt_masks_set_edit_mode(module, DT_MASKS_EDIT_FULL);
}

static gboolean _mask_has_elements(const dt_masks_form_t *grp);

// "use the mask of": this module's mask becomes a single raster element
// reading the source's mask. Everything else in it goes, after asking
static void _masks_use_mask_of(dt_iop_module_t *module,
                               const _masks_raster_source_entry_t *entry)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(_mask_has_elements(_module_mask_group(module)))
  {
    if(!dt_gui_show_yes_no_dialog(
         _("replace the mask?"), "",
         _("this removes every element from this module's mask and uses the mask"
           " of %s in their place"),
         entry->name))
      return;
    _masks_reset_mask_core(module);
  }
  // the reset left no group to aim at: the raster element starts the mask
  bd->insert_active = FALSE;
  _add_raster_mask(module, entry->src, entry->id);
  _flexi_refine_follow_selection(bd);
  _refresh_canvas_edit(module);
}

static void _masks_import_pick_action(GSimpleAction *action, GVariant *parameter, gpointer user_data)
{
  GtkWidget *btn = GTK_WIDGET(user_data);
  dt_iop_module_t *module = g_object_get_data(G_OBJECT(btn), "module");
  GPtrArray *mods = g_object_get_data(G_OBJECT(btn), "import_modules");
  GPtrArray *rasters = g_object_get_data(G_OBJECT(btn), "import_rasters");
  int op = 0, a = -1, b = INVALID_MASKID;
  g_variant_get(parameter, "(iii)", &op, &a, &b);
  if(darktable.gui->active_popover_menu)
    gtk_popover_popdown(GTK_POPOVER(darktable.gui->active_popover_menu));
  if(!module || !module->blend_data) return;

  if(op == _IMPORT_ADD_MASK || op == _IMPORT_USE_MASK)
  {
    if(!rasters || a < 0 || a >= (int)rasters->len) return;
    const _masks_raster_source_entry_t *entry = g_ptr_array_index(rasters, a);
    if(op == _IMPORT_USE_MASK)
      _masks_use_mask_of(module, entry);
    else
      _add_raster_mask(module, entry->src, entry->id);
    return;
  }

  dt_iop_module_t *src = (mods && a >= 0 && a < (int)mods->len) ? g_ptr_array_index(mods, a) : NULL;
  const gboolean whole = op == _IMPORT_LINK_GROUP || op == _IMPORT_COPY_GROUP;
  if(whole && !src) return;
  GList *fids = whole ? _model_module_shapes(src, b) : g_list_prepend(NULL, GINT_TO_POINTER(b));
  const gboolean copy =
    op == _IMPORT_COPY_ONE || op == _IMPORT_COPY_GROUP || op == _IMPORT_COPY_PARAMETRIC;
  _masks_import_forms(module, src, fids, copy);
  g_list_free(fids);
}

static void _masks_import_append(GMenu *menu,
                                 const char *label,
                                 const _masks_import_op_t op,
                                 const int a,
                                 const int b)
{
  GMenuItem *it = g_menu_item_new(label ? label : "", NULL);
  g_menu_item_set_action_and_target_value(it, "masks_import.pick",
                                          g_variant_new("(iii)", (int)op, a, b));
  g_menu_append_item(menu, it);
  g_object_unref(it);
}

#define _IMPORT_MAX_KIND_BUCKETS 16

// find (or create, appending to menu) the submenu for a given shape kind
static GMenu *_masks_import_kind_bucket(
  GMenu *menu, guint *kinds, GMenu **submenus, int *n_buckets, const guint kind)
{
  for(int k = 0; k < *n_buckets; k++)
    if(kinds[k] == kind) return submenus[k];
  if(*n_buckets >= _IMPORT_MAX_KIND_BUCKETS) return NULL;

  GMenu *sub = g_menu_new();
  kinds[*n_buckets] = kind;
  submenus[*n_buckets] = sub;
  (*n_buckets)++;

  g_menu_append_submenu(menu, _kind_name(kind, TRUE), G_MENU_MODEL(sub));
  return sub;
}

// index of `m` in the menu's module table, adding it if new
static int _masks_import_module_index(GPtrArray *mods, dt_iop_module_t *m)
{
  if(!m) return -1;
  for(guint k = 0; k < mods->len; k++)
    if(g_ptr_array_index(mods, k) == m) return (int)k;
  g_ptr_array_add(mods, m);
  return (int)mods->len - 1;
}

// a path that is one of an AI object's members: the object is the element,
// importing one of its paths on its own would split it apart
static gboolean _masks_import_is_object_member(const dt_mask_id_t formid)
{
  for(GList *l = darktable.develop->forms; l; l = g_list_next(l))
  {
    const dt_masks_form_t *f = l->data;
    if(!(f->type & DT_MASKS_OBJECT)) continue;
    for(GList *p = f->points; p; p = g_list_next(p))
      if(((dt_masks_point_group_t *)p->data)->formid == formid) return TRUE;
  }
  return FALSE;
}

// shapes of `src`'s mask (see _model_module_shapes) the mask `grp` does not
// use yet
static GList *_masks_import_candidates(dt_iop_module_t *src,
                                       const dt_mask_id_t cid,
                                       dt_masks_form_t *grp)
{
  GList *fids = _model_module_shapes(src, cid);
  for(GList *l = fids; l;)
  {
    GList *next = g_list_next(l);
    if(grp && _group_point(grp, GPOINTER_TO_INT(l->data))) fids = g_list_delete_link(fids, l);
    l = next;
  }
  return fids;
}

// the group headed by `cid` in `src`'s mask, named the way its own panel
// names it
static gchar *_masks_import_group_label(dt_iop_module_t *src,
                                        dt_masks_form_t *sgrp,
                                        const dt_mask_id_t cid)
{
  if(src->blend_data && cid == _mask_group_cid(src)) return g_strdup(_("whole mask"));
  const char *custom = _group_custom_name(sgrp, cid);
  if(custom) return g_strdup(custom);
  const dt_masks_point_group_t *head = _group_point(sgrp, cid);
  const char *op = _within_name(head ? head->state : 0);
  // numbers live in the module's own panel data, and are handed out on first
  // request in the order they are asked for: bottom-up, as here, is how that
  // panel numbers them itself
  return src->blend_data ? g_strdup_printf("%s-%d", op, _group_ordinal_of_cid(src, cid))
                         : g_strdup(op);
}

// "link shapes" or "copy shapes": every shape and AI object that another
// module's mask uses, or that no mask does, and that this module's mask does
// not use yet, reachable by the module it comes from and by its kind. Added
// to `menu` as one section. Returns how many there are
static int _masks_import_fill_shapes(GMenu *menu,
                                     dt_iop_module_t *module,
                                     GPtrArray *mods,
                                     const gboolean copy)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  const _masks_import_op_t one = copy ? _IMPORT_COPY_ONE : _IMPORT_LINK_ONE;
  const _masks_import_op_t group = copy ? _IMPORT_COPY_GROUP : _IMPORT_LINK_GROUP;
  GMenu *by_module = g_menu_new();
  GMenu *by_type = g_menu_new();
  guint kinds[_IMPORT_MAX_KIND_BUCKETS];
  GMenu *kind_submenus[_IMPORT_MAX_KIND_BUCKETS];
  int n_kinds = 0;
  int n = 0;

  for(GList *l = darktable.develop->iop; l; l = g_list_next(l))
  {
    dt_iop_module_t *src = l->data;
    if(src == module) continue;
    GList *shapes = _masks_import_candidates(src, INVALID_MASKID, grp);
    if(!shapes) continue;
    dt_masks_form_t *sgrp = _module_mask_group(src);
    const int a = _masks_import_module_index(mods, src);
    GMenu *sub = g_menu_new();

    GMenu *whole = g_menu_new();
    _masks_import_append(whole, _("all shapes"), group, a, INVALID_MASKID);
    // one entry per group, worth offering only when there is more than one
    GMenu *groups = g_menu_new();
    int n_groups = 0;
    for(GList *p = sgrp->points; p; p = g_list_next(p))
    {
      if(!_starts_group(p)) continue;
      const dt_mask_id_t cid = ((dt_masks_point_group_t *)p->data)->formid;
      GList *members = _masks_import_candidates(src, cid, grp);
      if(!members) continue;
      g_list_free(members);
      gchar *label = _masks_import_group_label(src, sgrp, cid);
      _masks_import_append(groups, label, group, a, cid);
      g_free(label);
      n_groups++;
    }
    if(n_groups > 1)
      g_menu_append_submenu(whole, _("all shapes from group"), G_MENU_MODEL(groups));
    g_object_unref(groups);
    g_menu_append_section(sub, NULL, G_MENU_MODEL(whole));
    g_object_unref(whole);

    GMenu *each = g_menu_new();
    for(GList *s = shapes; s; s = g_list_next(s))
    {
      const dt_mask_id_t fid = GPOINTER_TO_INT(s->data);
      const dt_masks_form_t *f = dt_masks_get_from_id(darktable.develop, fid);
      _masks_import_append(each, f->name, one, a, fid);
    }
    g_menu_append_section(sub, NULL, G_MENU_MODEL(each));
    g_object_unref(each);

    gchar *mlabel = dt_history_item_get_name(src);
    g_menu_append_submenu(by_module, mlabel, G_MENU_MODEL(sub));
    g_free(mlabel);
    g_object_unref(sub);
    n += g_list_length(shapes);
    g_list_free(shapes);
  }

  // every shape once, by kind, plus the ones no module uses at all
  GMenu *unused = g_menu_new();
  int n_unused = 0;
  for(GList *l = darktable.develop->forms; l; l = g_list_next(l))
  {
    const dt_masks_form_t *f = l->data;
    if(!_form_is_shape(f) || _masks_import_is_object_member(f->formid)) continue;
    if(grp && _group_point(grp, f->formid)) continue;
    GList *users = _model_form_users(f->formid);
    dt_iop_module_t *owner = users ? users->data : NULL;
    g_list_free(users);
    if(!owner)
    {
      _masks_import_append(unused, f->name, one, -1, f->formid);
      n_unused++;
      n++;
    }
    GMenu *bucket =
      _masks_import_kind_bucket(by_type, kinds, kind_submenus, &n_kinds, _form_kind(f));
    if(bucket)
      _masks_import_append(bucket, f->name, one, _masks_import_module_index(mods, owner),
                           f->formid);
  }
  if(n_unused)
    g_menu_append_submenu(by_module, _("not currently used"), G_MENU_MODEL(unused));
  g_object_unref(unused);

  if(n)
  {
    GMenu *sec = g_menu_new();
    g_menu_append_submenu(sec, _("by source module"), G_MENU_MODEL(by_module));
    g_menu_append_submenu(sec, _("by type"), G_MENU_MODEL(by_type));
    g_menu_append_section(menu, NULL, G_MENU_MODEL(sec));
    g_object_unref(sec);
  }
  g_object_unref(by_module);
  g_object_unref(by_type);
  for(int k = 0; k < n_kinds; k++) g_object_unref(kind_submenus[k]);
  return n;
}

// "copy parametric channel": every other module's parametric channels. One
// set up in another blend colorspace is listed but cannot be picked: its
// stored channel would be read through this module's channel table (see
// _parametric_get_mask_roi in masks/parametric.c). Returns how many there are
static int _masks_import_fill_parametric(GMenu *menu, dt_iop_module_t *module, GPtrArray *mods)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  const uint32_t csp = (uint32_t)module->blend_params->blend_cst;
  int n = 0;
  for(GList *l = darktable.develop->iop; l; l = g_list_next(l))
  {
    dt_iop_module_t *src = l->data;
    dt_masks_form_t *sgrp = src == module ? NULL : _module_mask_group(src);
    if(!sgrp) continue;
    GMenu *ok = g_menu_new();
    GMenu *other = g_menu_new();
    int n_ok = 0, n_other = 0;
    for(GList *p = sgrp->points; p; p = g_list_next(p))
    {
      const dt_mask_id_t fid = ((dt_masks_point_group_t *)p->data)->formid;
      const dt_masks_form_t *f = dt_masks_get_from_id(darktable.develop, fid);
      if(!f || !(f->type & DT_MASKS_PARAMETRIC) || !f->points) continue;
      if(grp && _group_point(grp, fid)) continue;
      const dt_masks_point_parametric_t *pp = f->points->data;
      if(pp->colorspace == csp)
      {
        _masks_import_append(ok, f->name, _IMPORT_COPY_PARAMETRIC,
                             _masks_import_module_index(mods, src), fid);
        n_ok++;
      }
      else
      {
        g_menu_append(other, f->name, "masks_import.unavailable");
        n_other++;
      }
    }
    if(n_ok || n_other)
    {
      GMenu *sub = g_menu_new();
      if(n_ok) g_menu_append_section(sub, NULL, G_MENU_MODEL(ok));
      if(n_other)
        g_menu_append_section(sub, _("other blend colorspace"), G_MENU_MODEL(other));
      gchar *mlabel = dt_history_item_get_name(src);
      g_menu_append_submenu(menu, mlabel, G_MENU_MODEL(sub));
      g_free(mlabel);
      g_object_unref(sub);
      n += n_ok + n_other;
    }
    g_object_unref(ok);
    g_object_unref(other);
  }
  return n;
}

// "add the mask of" / "use the mask of": the raster sources from
// _raster_sources_collect, the downstream ones listed but not pickable.
// Returns how many there are
static int _masks_import_fill_raster(GMenu *menu,
                                     GPtrArray *usable,
                                     GPtrArray *later,
                                     const _masks_import_op_t op)
{
  if(usable->len)
  {
    GMenu *ok = g_menu_new();
    for(guint k = 0; k < usable->len; k++)
    {
      const _masks_raster_source_entry_t *entry = g_ptr_array_index(usable, k);
      _masks_import_append(ok, entry->name, op, (int)k, 0);
    }
    g_menu_append_section(menu, NULL, G_MENU_MODEL(ok));
    g_object_unref(ok);
  }
  if(later->len)
  {
    GMenu *na = g_menu_new();
    for(guint k = 0; k < later->len; k++)
    {
      const _masks_raster_source_entry_t *entry = g_ptr_array_index(later, k);
      g_menu_append(na, entry->name, "masks_import.unavailable");
    }
    g_menu_append_section(menu, _("processed later in the pipe"), G_MENU_MODEL(na));
    g_object_unref(na);
  }
  return (int)(usable->len + later->len);
}

// removing a shape from a module's own group only detaches it from that
// group (see dt_masks_form_remove's grp != NULL branch in masks.c) -- it
// stays in darktable.develop->forms, unused, until something purges it:
// dt_masks_cleanup_unused, offered from this menu since that is where the
// clutter shows (it is exactly what ends up in the "not currently used"
// bucket of "by source module").
static void _masks_import_cleanup_action(GSimpleAction *action, GVariant *parameter, gpointer user_data)
{
  dt_iop_module_t *module = (dt_iop_module_t *)user_data;
  dt_masks_cleanup_unused(darktable.develop);
  dt_control_log(_("unused shapes removed"));
  _build_masks_list(module);
  if(darktable.gui->active_popover_menu)
    gtk_popover_popdown(GTK_POPOVER(darktable.gui->active_popover_menu));
}

// the reach of the cleanup above ends at the history stack: a shape only an
// earlier, since replaced step still uses has to stay for that step. Removing
// those too means dropping the steps, so this compresses history first, after
// saying so -- it takes undo and redo history with it.
static void _masks_import_compress_action(GSimpleAction *action, GVariant *parameter, gpointer user_data)
{
  if(darktable.gui->active_popover_menu)
    gtk_popover_popdown(GTK_POPOVER(darktable.gui->active_popover_menu));
  if(!dt_gui_show_yes_no_dialog(
       _("compress history and clean up unused shapes?"), "",
       _("this compresses the history stack, dropping every earlier step and"
         " anything you could redo, then deletes every shape no module uses.\n\n"
         "shapes still used by earlier steps can only be removed this way.")))
    return;
  dt_dev_history_truncate(darktable.develop, TRUE);
  dt_masks_cleanup_unused(darktable.develop);
  dt_control_log(_("history compressed and unused shapes removed"));
  // reloading history can rebuild module instances: go through the focused
  // module rather than the one this menu was opened on
  dt_iop_module_t *module = darktable.develop->gui_module;
  if(module && module->blend_data) _build_masks_list(module);
}

// the import menu: everything that brings in what another module already has.
// Shapes and AI objects are linked or copied (see the linking section above),
// parametric channels copied, and another module's whole mask arrives as a
// raster element, always live: added to the selected group, or replacing this
// module's mask outright. Submenu entries get no tooltips (see
// _popover_menu_apply_tooltips in gui/gtk.c), so section captions carry the
// explanations instead
static gboolean
_masks_import_btn_press(GtkWidget *btn, GdkEventButton *ev, dt_iop_module_t *module)
{
  if(ev->button != GDK_BUTTON_PRIMARY) return FALSE;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd) return FALSE;
  dt_iop_request_focus(module);

  if(gtk_widget_get_action_group(btn, "masks_import") == NULL)
  {
    GActionEntry pick_entries[] = {
      { "pick", _masks_import_pick_action, "(iii)", NULL },
    };
    GActionEntry module_entries[] = {
      { "cleanup",  _masks_import_cleanup_action,  NULL, NULL },
      { "compress", _masks_import_compress_action, NULL, NULL },
    };
    GSimpleActionGroup *sag = g_simple_action_group_new();
    g_action_map_add_action_entries(G_ACTION_MAP(sag), pick_entries,
                                    G_N_ELEMENTS(pick_entries), btn);
    g_action_map_add_action_entries(G_ACTION_MAP(sag), module_entries,
                                    G_N_ELEMENTS(module_entries), module);
    // entries shown for context but not pickable point here
    GSimpleAction *unavailable = g_simple_action_new("unavailable", NULL);
    g_simple_action_set_enabled(unavailable, FALSE);
    g_action_map_add_action(G_ACTION_MAP(sag), G_ACTION(unavailable));
    g_object_unref(unavailable);
    gtk_widget_insert_action_group(btn, "masks_import", G_ACTION_GROUP(sag));
    g_object_unref(sag);
  }
  g_object_set_data(G_OBJECT(btn), "module", module);

  GPtrArray *mods = g_ptr_array_new();
  GPtrArray *usable = g_ptr_array_new_with_free_func(_raster_source_entry_free);
  GPtrArray *later = g_ptr_array_new_with_free_func(_raster_source_entry_free);
  _raster_sources_collect(module, usable, later);

  GMenu *menu = g_menu_new();

  GMenu *link_menu = g_menu_new();
  GMenu *copy_menu = g_menu_new();
  const int n_shapes = _masks_import_fill_shapes(link_menu, module, mods, FALSE);
  _masks_import_fill_shapes(copy_menu, module, mods, TRUE);
  GMenu *param_menu = g_menu_new();
  const int n_param =
    bd->blendif_support ? _masks_import_fill_parametric(param_menu, module, mods) : 0;
  // the captions sit on the top level, so the difference between linking and
  // copying is read before either submenu is opened
  if(n_shapes)
  {
    GMenu *sec_link = g_menu_new();
    g_menu_append_submenu(sec_link, _("link shapes"), G_MENU_MODEL(link_menu));
    g_menu_append_section(menu, _("shared: editing one changes it everywhere"),
                          G_MENU_MODEL(sec_link));
    g_object_unref(sec_link);
  }
  if(n_shapes || n_param)
  {
    GMenu *sec_copy = g_menu_new();
    if(n_shapes)
      g_menu_append_submenu(sec_copy, _("copy shapes"), G_MENU_MODEL(copy_menu));
    if(n_param)
      g_menu_append_submenu(sec_copy, _("copy parametric channel"), G_MENU_MODEL(param_menu));
    g_menu_append_section(menu, _("independent copies"), G_MENU_MODEL(sec_copy));
    g_object_unref(sec_copy);
  }
  g_object_unref(link_menu);
  g_object_unref(copy_menu);
  g_object_unref(param_menu);

  GMenu *add_menu = g_menu_new();
  GMenu *use_menu = g_menu_new();
  const int n_raster = _masks_import_fill_raster(add_menu, usable, later, _IMPORT_ADD_MASK);
  _masks_import_fill_raster(use_menu, usable, later, _IMPORT_USE_MASK);
  if(n_raster)
  {
    GMenu *sec_mask = g_menu_new();
    g_menu_append_submenu(sec_mask, _("add the mask of"), G_MENU_MODEL(add_menu));
    g_menu_append_submenu(sec_mask, _("use the mask of"), G_MENU_MODEL(use_menu));
    g_menu_append_section(menu, _("another module's whole mask, kept up to date"),
                          G_MENU_MODEL(sec_mask));
    g_object_unref(sec_mask);
  }
  g_object_unref(add_menu);
  g_object_unref(use_menu);

  if(!n_shapes && !n_param && !n_raster)
    g_menu_append(menu, _("nothing to import"), "masks_import.unavailable");

  GMenu *cleanup_sec = g_menu_new();
  g_menu_append(cleanup_sec, _("clean up unused shapes"), "masks_import.cleanup");
  g_menu_append(cleanup_sec, _("compress history and clean up unused shapes"),
                "masks_import.compress");
  g_menu_append_section(menu, NULL, G_MENU_MODEL(cleanup_sec));
  g_object_unref(cleanup_sec);

  // the picks resolve their targets through these until the next popup
  g_object_set_data_full(G_OBJECT(btn), "import_modules", mods,
                         (GDestroyNotify)g_ptr_array_unref);
  g_object_set_data_full(G_OBJECT(btn), "import_rasters", usable,
                         (GDestroyNotify)g_ptr_array_unref);
  g_ptr_array_unref(later);

  darktable.gui->active_popover_menu = dt_gui_popover_menu_from_model(btn, menu);
  gtk_popover_popup(GTK_POPOVER(darktable.gui->active_popover_menu));
  g_object_unref(menu);
  return TRUE;
}

// edit on canvas and solo edit, as one run on the panel header between two
// fixed gaps: they act on the canvas, wherever the panel is hosted. Packed
// once, into the box the header reserved for them (see masks_header_edit_box)
static void _pack_header_edit_run(dt_iop_gui_blend_data_t *bd)
{
  GtkWidget *run = bd->masks_header_edit_box;
  if(!run) return;
  _pack_gap(run);
  dt_gui_box_add(run, bd->masks_edit, bd->soloedit_mode);
  _pack_gap(run);
  // soloedit_mode carries no_show_all and is shown by the mask-mode update
  gtk_widget_show(bd->masks_edit);
  gtk_widget_show(run);
}

// per-row parametric mask editor: every parametric channel row owns its own
// slider/picker/boost widgets, bound directly to that form's own
// dt_masks_point_parametric_t (see _build_param_row_editor). Declared here
// (rather than where it is built, near _make_shape_row) so early functions
// like _masks_param_inout_toggled can reach into a row's own editor by formid.
typedef struct dt_masks_param_row_editor_t
{
  dt_mask_id_t formid;
  dt_iop_module_t *module;
  dt_iop_gui_blendif_filter_t filter[2]; // input = 0, output = 1; no polarity widget
  GtkWidget *boost_box;
  GtkWidget *boost_slider;
  // both real, functional pickers -- kept alive but never shown; a single
  // visible master_picker button below stands in for both (see
  // _param_row_master_picker_pressed), routing plain/shift clicks to
  // colorpicker_set_values (set range from input/output) and ctrl clicks to
  // colorpicker (pick GUI color, point/area), so the row's action cluster
  // only spends one slot instead of two.
  GtkWidget *colorpicker;
  GtkWidget *colorpicker_set_values;
  GtkWidget *master_picker;
  // the user asked that expanding a parametric row's existing in/out chevron
  // also reveal the opacity slider (parametric rows get no separate properties
  // expander of their own -- see _make_props_row_toggle's callers). Gated by
  // the same p->in_out bit as filter[1].box/boost_box, in
  // _update_param_row_visibility. Delta-applied via the shared
  // _props_row_apply, same as every other row kind's opacity control.
  GtkWidget *opacity_box;
  GtkWidget *opacity_slider;
  float opacity_last_value;
  // collapsed state docks the input slider directly onto the row's own header
  // bar (see _make_shape_row) instead of the below-row editor -- this is that
  // slot, wired up from _make_shape_row after this editor is built (NULL until
  // then, and left NULL entirely for a legacy multi-channel form, which is
  // still edited via the shared tabbed editor, not one of these per-row ones).
  // Always visible and always the row's sole expanding child (see
  // _make_shape_row's packing of evbox/header_slot) -- whether or not the
  // slider is currently docked inside it, so the name label never has to
  // fight it for a share of the header's free width (that used to be done by
  // toggling the name's own GtkBox "expand" child property at dock time,
  // which left a dead gap whenever the two were briefly out of sync). See
  // _update_param_row_header_dock.
  GtkWidget *header_slot;
  // the row's own full-width outer box (icon/name/slider/actions), wired up
  // from _make_shape_row exactly like header_slot above -- used only to size
  // and place the precise-value popup (see _bauhaus_whisker_popup_rect)
  // against the row's real on-screen width/position, not the header_slot's
  // (which starts only after the name label).
  GtkWidget *row;
  // the row's picker button (master_picker's own wrapper, see
  // _build_param_row_editor's picker_box_out) permanently docked as
  // header_slot's fixed-width first child, immediately left of whichever
  // slider is currently docked there -- see _update_param_row_header_dock.
  // Never reparented itself, unlike the input/opacity sliders.
  GtkWidget *header_picker;
  GtkWidget *sliders_grid;
  GtkWidget *input_lbl;
  GtkWidget *input_slot;
  // the bypass "eye" and the fixed-width box it lives in, laid over the right
  // end of the slider row (input_slot/output_slot are those overlays). The two
  // are separate because they hide on different conditions: the box goes only
  // with its row, while the eye inside it comes and goes with whether the
  // channel has both sub-ranges in play. See _make_param_bypass_slot.
  GtkWidget *input_bypass_btn;
  GtkWidget *input_bypass_slot;
  GtkWidget *output_lbl;
  GtkWidget *output_slot;
  GtkWidget *output_bypass_btn;
  GtkWidget *output_bypass_slot;
  GtkWidget *name_evbox;
} dt_masks_param_row_editor_t;

static void _update_param_row_display(dt_masks_param_row_editor_t *ed);
static void _update_param_row_visibility(dt_masks_param_row_editor_t *ed);
static gboolean _param_row_picker_apply(dt_iop_module_t *module,
                                        GtkWidget *picker,
                                        dt_dev_pixelpipe_t *pipe);

// per-row/group "properties" inline expander (see the Phase-3-replacement
// block below, near _blend_masks_properties): every shape/raster/group row
// gets its own permanently-built (but conditionally hidden) editor box docked
// directly below it, mirroring dt_masks_param_row_editor_t's pattern exactly.
typedef struct dt_masks_props_row_editor_t
{
  dt_iop_module_t *module;
  dt_mask_id_t formid;   // single element's own id, or a group's head/cid
  gboolean is_group;     // TRUE => target is _selected_group_formids(grp, formid)
  gboolean opacity_only; // TRUE => build only the opacity control
  GtkWidget *widget[DT_MASKS_PROPERTY_LAST];
  float last_value[DT_MASKS_PROPERTY_LAST];
  // a relative (ratio) property has no fixed "no change" absolute value the
  // way an additive one does -- its double-click reset target is instead the
  // shape's own size/feather/etc. as first seen by this row, captured once
  // (see _props_row_populate) rather than re-synced on every reopen, so it
  // reads as "undo edits made in this sitting", closest available proxy for
  // "reset to how it was created" without persisting new per-shape state.
  gboolean relative_baseline_set;
  // path-only shrink/grow control, mirroring the removed mask manager's own resize_amount
  // (see the block near _blend_masks_properties below) -- NULL for a group row
  // or an opacity-only row, and hidden at runtime for anything but a path.
  GtkWidget *resize_widget;
  guint resize_timer;       // debounce source id (0 = none)
  gboolean resize_updating; // guard: programmatic slider change, don't commit
} dt_masks_props_row_editor_t;

static void _refine_scope_combo_rebuild(dt_iop_module_t *module);
static void _update_add_target_sensitivity(dt_iop_module_t *module);
static void _update_refine_sensitivity(dt_iop_module_t *module);
static void _set_group_target_ext(dt_iop_module_t *module,
                                  const dt_mask_id_t cid,
                                  const dt_mask_id_t keep_entered);
static void _set_group_target(dt_iop_module_t *module, const dt_mask_id_t cid);
static void _set_form_target_ext(dt_iop_module_t *module,
                                 const dt_mask_id_t id,
                                 const gboolean auto_expand);
static void _set_form_target(dt_iop_module_t *module, const dt_mask_id_t id);
static void _element_chevron_clicked(dt_iop_module_t *module,
                                     const dt_mask_id_t id,
                                     const gboolean expanded);
int _within_index_for_state(const int state);
dt_mask_id_t _group_cid_of_form(dt_masks_form_t *grp, const dt_mask_id_t fid);
static void _paint_param_inout(cairo_t *cr,
                               const gint x,
                               const gint y,
                               const gint w,
                               const gint h,
                               const gint flags,
                               void *data);
// appends a "presets" section (group-layout presets) directly to `menu` --
// see the full definition near _flexi_layout_apply
// detach members from a module's mask group without dt_masks_form_remove's
// nested history/GUI update and its "group just emptied" destruction cascade
static void _detach_group_members(dt_masks_form_t *grp, GList *fids);
static void _recompute_insert_hint(dt_iop_module_t *module);
static void _blendif_options_callback(GtkButton *button, dt_iop_module_t *module);
static gboolean _op_is_bypassed(const int state);
static GtkWidget *_find_row_by_formid(GtkWidget *w, const dt_mask_id_t formid);
int _group_ordinal_of_cid(dt_iop_module_t *module, const dt_mask_id_t cid);
static guint _form_kind(const dt_masks_form_t *form);
static const char *_kind_name(const guint kind, const gboolean plural);
static DTGTKCairoPaintIconFunc _kind_icon_paint(const guint kind);
static DTGTKCairoPaintIconFunc _op_paint_for_state(const int state);
static GtkWidget *_make_channel_handle(const char *code, const char *tooltip);
static const char *_form_type_prefix(const dt_masks_form_t *form);
static GtkWidget *_make_pending_shape_row(dt_iop_module_t *module, dt_masks_form_t *form);
static gboolean _shape_props_subpanel(void);
static void _props_panel_show(dt_iop_gui_blend_data_t *bd);

// with the mask off there is nothing for the panel's controls to act on, so
// the whole panel body goes insensitive rather than merely inert -- the panel
// can still be expanded (the mask-off state is reachable and legible either
// way), it just stops offering controls that would silently do nothing.
// Deliberately left live: the on/off toggle, the collapse arrow and the
// options hamburger, all in the header -- those are the ways out of this
// state, and the options menu is where the panel position is chosen.
static void _masks_panel_apply_enabled_state(dt_iop_gui_blend_data_t *data,
                                             const gboolean mask_enabled)
{
  if(data->masks_panel_body)
    gtk_widget_set_sensitive(GTK_WIDGET(data->masks_panel_body), mask_enabled);
  // the theme's insensitive colors would make an off mask's groups and
  // elements unreadable, and they are still meant to be read. Only the list:
  // the controls around it keep looking unavailable (see masks-list-off in
  // darktable.css)
  if(data->masks_list_box)
  {
    if(mask_enabled)
      dt_gui_remove_class(GTK_WIDGET(data->masks_list_box), "masks-list-off");
    else
      dt_gui_add_class(GTK_WIDGET(data->masks_list_box), "masks-list-off");
  }

  // the rest are header controls. Sensitivity is set on the widgets
  // themselves, not on the cluster holding them, because the hamburger shares
  // that cluster and must stay usable while the mask is off.
  const gboolean has_drawn = _module_has_drawn_shapes(data->module);
  if(data->masks_edit) gtk_widget_set_sensitive(data->masks_edit, mask_enabled && has_drawn);
}

static void _blendop_masks_mode_callback(const dt_develop_mask_mode_t mask_mode,
                                         dt_iop_gui_blend_data_t *data)
{
  dt_develop_blend_params_t *bp = data->module->blend_params;
  if(bp->mask_mode != mask_mode)
    dt_print(DT_DEBUG_MASKS,
             "[masks] _blendop_masks_mode_callback '%s': mask_mode 0x%x->0x%x",
             data->module->op, bp->mask_mode, mask_mode);
  const gboolean was_enabled = bp->mask_mode & DEVELOP_MASK_ENABLED;
  bp->mask_mode = mask_mode;

  const gboolean mask_enabled = mask_mode & DEVELOP_MASK_ENABLED;
  const gboolean mode_raster = mask_mode & DEVELOP_MASK_RASTER;
  const gboolean mode_drawn = mask_mode & DEVELOP_MASK_MASK;
  const gboolean mode_flexi = !mode_raster && (mask_enabled || (mask_mode & DEVELOP_MASK_FLEXI));
  const gboolean mode_parametric = mask_mode & DEVELOP_MASK_CONDITIONAL;
  // flexi reuses the drawn-group toolbar/renderer, so the drawn-mask panel and
  // refinement controls appear for it too.
  const gboolean mode_drawn_or_flexi = mode_drawn || mode_flexi;
  // with the mask off the panel shows what switching it on would give -- the
  // same controls, greyed out (see _masks_panel_apply_enabled_state) -- rather
  // than folding to nothing and leaving an empty panel behind for anyone who
  // expands it anyway. Switching on always lands in flexi (see
  // _blendop_mask_enable), so flexi is the layout to preview.
  const gboolean show_mask_ui = !mode_raster;
  const gboolean show_flexi_ui = !mode_raster;

  _box_set_visible(data->blend_box, TRUE);
  _masks_panel_apply_enabled_state(data, mask_enabled);

  if(data->masks_blend_header)
  {
    if(mask_enabled)
      dt_gui_add_class(data->masks_blend_header, "mask-enabled");
    else
      dt_gui_remove_class(data->masks_blend_header, "mask-enabled");
  }
  // the docked panel is styled as a module, active while its mask is on
  if(darktable.develop->proxy.masks_flexi_host.hosted_module == data->module)
    dt_ui_flexi_panel_set_active(darktable.gui->ui, mask_enabled);

  dt_iop_advertise_rastermask(data->module, mask_mode);

  if(mask_enabled
     && ((data->masks_inited && mode_drawn_or_flexi)
         || (data->blendif_inited && mode_parametric)))
  {
    if(data->blendif_inited && mode_parametric)
    {
      dt_bauhaus_combobox_set_from_value(data->masks_combine_combo,
         bp->mask_combine & (DEVELOP_COMBINE_INV | DEVELOP_COMBINE_INCL));
    }
    gtk_widget_set_visible(GTK_WIDGET(data->masks_combine_combo), data->blendif_inited && mode_parametric);

    /*
     * if this iop is operating in raw space, it has only 1 channel per pixel,
     * thus there is no alpha channel where we would normally store mask
     * that would get displayed if following button have been pressed.
     *
     * TODO: revisit if/once there semi-raw iops (e.g temperature) with blending
     */
    if(data->module->blend_colorspace(data->module, NULL, NULL) == IOP_CS_RAW)
    {
      data->module->request_mask_display = DT_DEV_PIXELPIPE_DISPLAY_NONE;
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->showmask), FALSE);

      // disable also guided-filters on RAW based color space
      gtk_widget_set_sensitive(data->masks_feathering_guide_combo, FALSE);
      gtk_widget_hide(GTK_WIDGET(data->masks_feathering_guide_combo));
      gtk_widget_set_sensitive(data->feathering_radius_slider, FALSE);
      gtk_widget_hide(GTK_WIDGET(data->feathering_radius_slider));
      gtk_widget_set_sensitive(data->brightness_slider, FALSE);
      gtk_widget_hide(GTK_WIDGET(data->brightness_slider));
      gtk_widget_set_sensitive(data->contrast_slider, FALSE);
      gtk_widget_hide(GTK_WIDGET(data->contrast_slider));
      gtk_widget_set_sensitive(data->details_slider, FALSE);
      gtk_widget_hide(GTK_WIDGET(data->details_slider));
    }

    _box_set_visible(data->refine_box, TRUE);
  }
  else
  {
    // mask off: still shown, greyed, as the preview described above
    _box_set_visible(data->refine_box,
                     !mask_enabled || (data->raster_inited && mode_raster));
  }

  if(data->masks_inited && show_mask_ui)
  {
    // flexi-only widgets: new-shape operator selector, add-parametric button,
    // and the per-shape composition list. classic drawn mask keeps the vanilla
    // toolbar.
    if(data->masks_param_channels_box)
      gtk_widget_set_visible(data->masks_param_channels_box,
                             show_flexi_ui && data->blendif_support);
    gtk_widget_set_visible(data->masks_toolbar, show_flexi_ui);
    if(data->soloedit_mode) gtk_widget_set_visible(data->soloedit_mode, show_flexi_ui);
    gtk_widget_set_visible(GTK_WIDGET(data->masks_list_box), show_flexi_ui);
    // only for a live mask: with the mask off the list keeps whatever it last
    // held, greyed out, rather than being rebuilt from a group nothing is
    // using. A list never built is built anyway, so switching the mask on or
    // off never changes the groups it shows
    if(mode_flexi || data->masks_list_sig == DT_INVALID_HASH)
      _build_masks_list(data->module);
    _box_set_visible(data->masks_box, TRUE);
    _props_panel_show(data);

    if(!mask_enabled)
    {
      // the panel is a preview of what switching the mask on would give, so
      // nothing of it belongs on canvas (this used to fall to the classic
      // branch below, which is now reached only by a live classic mask)
      for(int n = 0; n < DEVELOP_MASKS_NB_SHAPES; n++)
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->masks_shapes[n]), FALSE);
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->masks_edit), FALSE);
      dt_masks_set_edit_mode(data->module, DT_MASKS_EDIT_OFF);
    }
  }
  else if(data->masks_inited)
  {
    for(int n = 0; n < DEVELOP_MASKS_NB_SHAPES; n++)
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->masks_shapes[n]), FALSE);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->masks_edit), FALSE);
    dt_masks_set_edit_mode(data->module, DT_MASKS_EDIT_OFF);
    _box_set_visible(data->masks_box, FALSE);
    _box_set_visible(data->props_panel_box, FALSE);
  }
  else if(data->masks_support)
  {
    for(int n = 0; n < DEVELOP_MASKS_NB_SHAPES; n++)
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->masks_shapes[n]), FALSE);
    _box_set_visible(data->masks_box, FALSE);
    _box_set_visible(data->props_panel_box, FALSE);
  }

  _box_set_visible(data->raster_box, data->raster_inited && mode_raster);

  // leaving flexi: drop flexi-only selection/staging state
  if(!mode_flexi)
  {
    data->panel_selected_formid = INVALID_MASKID;
    data->panel_selected_group_cid = INVALID_MASKID;
    data->insert_active = FALSE;
  }

  if(data->blendif_inited && mode_parametric)
  {
    _box_set_visible(data->blendif_box, TRUE);
  }
  else if(data->blendif_inited)
  {
    /* switch off color picker */
    dt_iop_color_picker_reset(data->module, FALSE);

    _box_set_visible(data->blendif_box, FALSE);
  }
  else
  {
    _box_set_visible(data->blendif_box, FALSE);
  }

  dt_dev_add_history_item(darktable.develop, data->module, TRUE);

  // rebuild the accelerators
  dt_iop_connect_accels_multi(data->module->so);

  // switching the mask on is an act of reaching for its controls, so unfold
  if(mask_enabled != was_enabled && mask_enabled)
  {
    _masks_panel_set_collapsed_pref(FALSE);
  }

  // mode just changed (possibly into/out of flexi) while this module was
  // already focused -- dt_iop_request_focus() above is a no-op in that case,
  // so re-evaluate the flexi panel's host placement here too
  _masks_flexi_relocate(data->module);
}

static void _blendop_blend_mode_callback(GtkWidget *combo,
                                         dt_iop_gui_blend_data_t *data)
{
  DT_GUARD_GUI_UPDATE();

  dt_develop_blend_params_t *bp = data->module->blend_params;
  const dt_develop_blend_mode_t new_blend_mode =
    GPOINTER_TO_INT(dt_bauhaus_combobox_get_data(combo));

  if(new_blend_mode != (bp->blend_mode & DEVELOP_BLEND_MODE_MASK))
  {
    bp->blend_mode = new_blend_mode | (bp->blend_mode & DEVELOP_BLEND_REVERSE);

    if(_blendif_blend_parameter_enabled(data->blend_modes_csp, bp->blend_mode))
    {
      gtk_widget_show(data->blend_mode_parameter_slider);
    }
    else
    {
      bp->blend_parameter = 0.0f;
      dt_bauhaus_slider_set(data->blend_mode_parameter_slider, bp->blend_parameter);
      gtk_widget_hide(data->blend_mode_parameter_slider);
    }
    dt_dev_add_history_item(darktable.develop, data->module, TRUE);
  }
}

static void _blendop_blend_order_clicked(GtkGestureSingle *gesture,
                                             gint n_press,
                                             gdouble x,
                                             gdouble y,
                                             dt_iop_module_t *module)
{
  DT_GUARD_GUI_UPDATE();

  GtkWidget *button = dt_gui_get_widget(gesture);

  dt_develop_blend_params_t *bp = module->blend_params;
  const gboolean active = !(bp->blend_mode & DEVELOP_BLEND_REVERSE);

  if(!active)
    bp->blend_mode &= ~DEVELOP_BLEND_REVERSE;
  else
    bp->blend_mode |= DEVELOP_BLEND_REVERSE;

  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(button), active);

  dt_dev_add_history_item(darktable.develop, module, TRUE);
  dt_control_queue_redraw_widget(GTK_WIDGET(button));
}

static void _blendop_masks_combine_callback(GtkWidget *combo,
                                            dt_iop_gui_blend_data_t *data)
{
  dt_develop_blend_params_t *const bp = data->module->blend_params;

  const uint32_t combine =
    GPOINTER_TO_UINT(dt_bauhaus_combobox_get_data(data->masks_combine_combo));
  bp->mask_combine &= ~(DEVELOP_COMBINE_INV | DEVELOP_COMBINE_INCL);
  bp->mask_combine |= combine;

  // inverts the parametric mask channels that are not used
  if(data->blendif_support && data->blendif_inited)
  {
    const uint32_t mask =
      data->csp == DEVELOP_BLEND_CS_LAB
      ? DEVELOP_BLENDIF_Lab_MASK
      : DEVELOP_BLENDIF_RGB_MASK;

    const uint32_t unused_channels = mask & ~bp->blendif;

    bp->blendif &= ~(unused_channels << 16);
    if(bp->mask_combine & DEVELOP_COMBINE_INCL)
    {
      bp->blendif |= unused_channels << 16;
    }
    // the shared tabbed editor that used to be refreshed here is gone; the
    // history item below re-runs gui_update, which repaints the per-row
    // parametric editors from their own forms
  }

  dt_dev_add_history_item(darktable.develop, data->module, TRUE);
}

static float _log10_scale_callback(GtkWidget *self,
                                  const float inval,
                                  const int dir)
{
  float outval = .0f;
  const float tiny = 1.0e-4f;

  switch(dir)
  {
    case GRADIENT_SLIDER_SET:
      outval = (log10(CLAMP(inval, 0.0001f, 1.0f)) + 4.0f) / 4.0f;
      break;
    case GRADIENT_SLIDER_GET:
      outval = CLAMP(exp(M_LN10 * (4.0f * inval - 4.0f)), 0.0f, 1.0f);
      if(outval <= tiny) outval = 0.0f;
      if(outval >= 1.0f - tiny) outval = 1.0f;
      break;
    default:
      outval = inval;
  }
  return outval;
}


static float _magnifier_scale_callback(GtkWidget *self,
                                      const float inval,
                                      const int dir)
{
  const float range = 6.0f;
  const float invrange = 1.0f/range;
  const float scale = tanh(range * 0.5f);
  const float invscale = 1.0f/scale;
  const float eps = 1.0e-6f;
  const float tiny = 1.0e-4f;

  float outval = .0f;
  switch(dir)
  {
    case GRADIENT_SLIDER_SET:
      outval = (invscale * tanh(range *
                                (CLAMP(inval, 0.0f, 1.0f) - 0.5f)) + 1.0f) * 0.5f;
      if(outval <= tiny) outval = 0.0f;
      if(outval >= 1.0f - tiny) outval = 1.0f;
      break;
    case GRADIENT_SLIDER_GET:
      outval = invrange * atanh((2.0f *
                                 CLAMP(inval, eps, 1.0f - eps) - 1.0f) * scale) + 0.5f;
      if(outval <= tiny) outval = 0.0f;
      if(outval >= 1.0f - tiny) outval = 1.0f;
      break;
    default:
      outval = inval;
  }
  return outval;
}

// defined below, next to the other per-row editor helpers
static const dt_masks_param_row_editor_t *_param_row_editor_resolve(
  GtkWidget *widget, const dt_iop_gui_blendif_channel_t **channels_out, int *ch_out);

// toggle a slider's alternative (log / magnifier) display scale and restate its
// head label to match. The slider always belongs to a per-row parametric editor
// now -- the shared tabbed editor this used to also serve is gone, and with it
// bd->filter[], whose head labels were NULL here (silently dropping the "(log)"
// / "(zoom)" suffix this function exists to show).
static int _blendop_blendif_disp_alternative_worker(GtkWidget *widget,
                                                    dt_iop_module_t *module,
                                                    const int mode,
                                                    float (*scale_callback)(GtkWidget*, float, int),
                                                    const char *label)
{
  GtkDarktableGradientSlider *slider = (GtkDarktableGradientSlider *)widget;

  dtgtk_gradient_slider_multivalue_set_scale_callback
    (slider,
     (mode == 1) ? scale_callback : NULL);

  const dt_iop_gui_blendif_channel_t *channels;
  int ch;
  const dt_masks_param_row_editor_t *ed =
    _param_row_editor_resolve(widget, &channels, &ch);
  if(ed)
  {
    const int in_out = (widget == GTK_WIDGET(ed->filter[1].slider)) ? 1 : 0;
    gchar *text = g_strdup_printf("%s%s", (in_out == 0) ? _("input") : _("output"),
                                  (mode == 1) ? label : "");
    if(ed->filter[in_out].head) gtk_label_set_text(ed->filter[in_out].head, text);
    // the compact layout shows a second copy of the same head label beside the
    // slider (see _apply_param_row_filter_layout); keep the two in step
    if(ed->filter[in_out].head_compact)
      gtk_label_set_text(ed->filter[in_out].head_compact, text);
    g_free(text);
  }

  return (mode == 1) ? 1 : 0;
}

static int _blendop_blendif_disp_alternative_mag(GtkWidget *widget,
                                                 dt_iop_module_t *module,
                                                 const int mode)
{
  return _blendop_blendif_disp_alternative_worker
    (widget, module, mode, _magnifier_scale_callback, _(" (zoom)"));
}

static int _blendop_blendif_disp_alternative_log(GtkWidget *widget,
                                                 dt_iop_module_t *module,
                                                 const int mode)
{
  return _blendop_blendif_disp_alternative_worker
    (widget, module, mode, _log10_scale_callback, _(" (log)"));
}

// parameterized on an explicit channel index instead of reading bd->tab, so a
// per-row parametric-form editor can call it with p->channel (see
// _param_row_picker_colorspace) while the shared editor keeps using bd->tab.
static dt_iop_colorspace_type_t
_picker_colorspace_for_channel(const dt_develop_blend_colorspace_t channel_tabs_csp,
                               const int channel)
{
  dt_iop_colorspace_type_t picker_cst = IOP_CS_NONE;

  if(channel_tabs_csp == DEVELOP_BLEND_CS_RGB_DISPLAY)
  {
    if(channel < 4)
      picker_cst = IOP_CS_RGB;
    else
      picker_cst = IOP_CS_HSL;
  }
  else if(channel_tabs_csp == DEVELOP_BLEND_CS_RGB_SCENE)
  {
    if(channel < 4)
      picker_cst = IOP_CS_RGB;
    else
      picker_cst = IOP_CS_JZCZHZ;
  }
  else if(channel_tabs_csp == DEVELOP_BLEND_CS_LAB)
  {
    if(channel < 3)
      picker_cst = IOP_CS_LAB;
    else
      picker_cst = IOP_CS_LCH;
  }

  return picker_cst;
}

static dt_iop_colorspace_type_t
_blendop_blendif_get_picker_colorspace(dt_iop_gui_blend_data_t *bd)
{
  return _picker_colorspace_for_channel(bd->channel_tabs_csp, bd->tab);
}

static inline int _blendif_print_digits_picker(const float value)
{
  return (value < 10.0f) ? 2 : 1;
}

// NB: the former _blendop_blendif_details_callback / _blendop_blendif_feathering_callback
// were folded into the scoped-refinement handler _refine_control_changed (see the
// Phase 2 block near the mask group helpers); its GLOBAL-scope path reproduces
// their exact behaviour (details zero-cross reprocess, feather_version bump).

static void _blendop_blendif_showmask_clicked(
  GtkGestureSingle *gesture, gint n_press, gdouble x, gdouble y, dt_iop_module_t *module)
{
  DT_GUARD_GUI_UPDATE();

  if(dt_gui_current_button(gesture) != GDK_BUTTON_PRIMARY) return;

  GtkWidget *button = dt_gui_get_widget(gesture);

  const gboolean has_mask_display =
    module->request_mask_display
    & (DT_DEV_PIXELPIPE_DISPLAY_MASK | DT_DEV_PIXELPIPE_DISPLAY_CHANNEL);

  module->request_mask_display &=
    ~(DT_DEV_PIXELPIPE_DISPLAY_MASK | DT_DEV_PIXELPIPE_DISPLAY_CHANNEL
      | DT_DEV_PIXELPIPE_DISPLAY_ANY);

  GdkModifierType state = dt_gui_current_state(gesture);

  if(dt_modifier_is(state, GDK_CONTROL_MASK | GDK_SHIFT_MASK))
    module->request_mask_display |=
      (DT_DEV_PIXELPIPE_DISPLAY_MASK | DT_DEV_PIXELPIPE_DISPLAY_CHANNEL);
  else if(dt_modifier_is(state, GDK_SHIFT_MASK))
    module->request_mask_display |= DT_DEV_PIXELPIPE_DISPLAY_CHANNEL;
  else if(dt_modifier_is(state, GDK_CONTROL_MASK))
    module->request_mask_display |= DT_DEV_PIXELPIPE_DISPLAY_MASK;
  else
    module->request_mask_display |=
      (has_mask_display ? DT_DEV_PIXELPIPE_DISPLAY_NONE : DT_DEV_PIXELPIPE_DISPLAY_MASK);

  gtk_toggle_button_set_active
    (GTK_TOGGLE_BUTTON(button),
     module->request_mask_display != DT_DEV_PIXELPIPE_DISPLAY_NONE);

  if(module->off) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(module->off), TRUE);

  DT_ENTER_GUI_UPDATE();

  // (re)set the header mask indicator too
  if(module->mask_indicator)
    gtk_toggle_button_set_active
      (GTK_TOGGLE_BUTTON(module->mask_indicator),
       module->request_mask_display != DT_DEV_PIXELPIPE_DISPLAY_NONE);

  DT_LEAVE_GUI_UPDATE();

  dt_iop_request_focus(module);
  dt_iop_refresh_center(module);
}

static void _update_mask_enable_toggle_tooltip(GtkWidget *toggle, const gboolean enabled)
{
  if(!toggle) return;
  // the right-click half is the only advertisement the blending options get:
  // no panel position shows a preferences icon for them any more (see
  // _blendop_mask_enable_toggled)
  gtk_widget_set_tooltip_text(toggle,
                              enabled
                              ? _("mask enabled\nclick to disable\nright-click for blending options")
                              : _("mask disabled\nclick to enable\nright-click for blending options"));
}

// force the blend mask on (flexi), no-op if it already has some mask
// content -- used by entry points ("add shape" / "add parametric channel" /
// "add raster element") that need the group evaluated even if the user
// hadn't switched masking on yet (see bd->mask_enable_toggle for the
// user-facing on/off control, which is the only other way into this state)
static void _blendop_mask_enable(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *data = module->blend_data;
  if(module->blend_params->mask_mode
     & (DEVELOP_MASK_MASK | DEVELOP_MASK_FLEXI | DEVELOP_MASK_RASTER))
    return;

  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->mask_enable_toggle), TRUE);
  _update_mask_enable_toggle_tooltip(data->mask_enable_toggle, TRUE);
  _blendop_masks_mode_callback(DEVELOP_MASK_ENABLED | DEVELOP_MASK_FLEXI, data);
  dt_iop_add_remove_mask_indicator(module, TRUE);
  gtk_widget_set_visible(data->showmask, TRUE);

  const int pos = _masks_panel_position();
  if(pos == MASKS_PANEL_POS_UTILITY)
  {
    dt_lib_module_t *host = darktable.develop->proxy.masks_flexi_host.module;
    if(host) dt_lib_gui_set_expanded(host, TRUE);
  }

  DT_ENTER_GUI_UPDATE();
  if(module->mask_indicator)
    gtk_toggle_button_set_active(
      GTK_TOGGLE_BUTTON(module->mask_indicator),
      gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(data->showmask)));
  DT_LEAVE_GUI_UPDATE();
}

// public wrapper around _blendop_mask_enable above -- see blend.h. bd's own
// mask_enable_toggle is driven by a click *gesture* (see
// _blendop_mask_enable_toggled below), not a GtkToggleButton "toggled"
// signal, so a caller in another translation unit (gtk.c's flexi corner
// icon) can't just flip the toggle's active state and expect the usual
// enabling side effects to follow -- it needs this real entry point instead.
void dt_iop_gui_blend_mask_enable(dt_iop_module_t *module)
{
  if(!module || !module->blend_data) return;
  dt_iop_request_focus(module);
  _blendop_mask_enable(module);
}

void dt_iop_gui_blend_sync_pending_ai_sliders(dt_iop_module_t *module)
{
#ifdef HAVE_AI
  dt_iop_gui_blend_data_t *bd = module ? module->blend_data : NULL;
  if(!bd || !bd->pending_ai_smoothing_slider || !bd->pending_ai_cleanup_slider) return;

  float smoothing = 0.0f;
  int cleanup = 0;
  if(!dt_masks_object_creation_get_preview_params(&smoothing, &cleanup)) return;

  DT_ENTER_GUI_UPDATE();
  dt_bauhaus_slider_set(bd->pending_ai_smoothing_slider, smoothing);
  dt_bauhaus_slider_set(bd->pending_ai_cleanup_slider, (float)cleanup);
  DT_LEAVE_GUI_UPDATE();
  bd->pending_ai_smoothing_last = smoothing;
  bd->pending_ai_cleanup_last = (float)cleanup;
#else
  (void)module;
#endif
}

// the single on/off toggle for the blend mask (see bd->mask_enable_toggle):
// with flexi as the only mask type left, "on" and "pick a mask type" are
// the same action -- picking it with nothing added yet behaves exactly like
// classic's old "uniformly" (see blend.c's "no form defined" fallback fill).
// note this is NOT redundant with an off module: DEVELOP_MASK_DISABLED skips
// the blend-compositing step entirely (see pixelpipe_hb.c), while an empty
// flexi mask still engages it with a full/uniform mask, so blend mode and
// opacity keep having an effect.
static void _blendop_mask_enable_toggled(
  GtkGestureSingle *gesture, gint n_press, gdouble x, gdouble y, dt_iop_module_t *module)
{
  DT_GUARD_GUI_UPDATE();
  const guint pressed = dt_gui_current_button(gesture);
  GtkWidget *button = dt_gui_get_widget(gesture);

  // no panel position shows a preferences icon of its own any more (see
  // _masks_header_apply_side), so the blending options hang off a right-click
  // here, the way the guide settings hang off the guides icon in the toolbar
  if(pressed == GDK_BUTTON_SECONDARY)
  {
    dt_iop_request_focus(module);
    _blendif_options_callback(GTK_BUTTON(button), module);
    return;
  }
  if(pressed != GDK_BUTTON_PRIMARY) return;

  dt_iop_gui_blend_data_t *data = module->blend_data;

  dt_iop_request_focus(module);

  if(!gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(button)))
  {
    _blendop_mask_enable(module);
  }
  else
  {
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(button), FALSE);
    _update_mask_enable_toggle_tooltip(button, FALSE);
    gtk_widget_set_visible(data->showmask, FALSE);
    _blendop_masks_mode_callback(DEVELOP_MASK_DISABLED, data);
    dt_iop_add_remove_mask_indicator(module, FALSE);
  }

  dt_control_hinter_message("");
}

static void _blendop_masks_add_shape(GtkGestureSingle *gesture,
                                         gint n_press,
                                         gdouble x,
                                         gdouble y,
                                         dt_iop_module_t *self)
{
  GtkWidget *widget = dt_gui_get_widget(gesture);

  dt_iop_gui_blend_data_t *bd = self->blend_data;

  const GdkModifierType state = dt_gui_current_state(gesture);
  const gboolean continuous = dt_modifier_is(state, GDK_CONTROL_MASK);

  // find out who we are
  int this = -1;
  for(int n = 0; n < DEVELOP_MASKS_NB_SHAPES; n++)
  {
    if(widget == bd->masks_shapes[n])
    {
      this = n;
      break;
    }
  }

  if(this < 0) return;

#ifdef HAVE_AI
  if(bd->masks_type[this] == DT_MASKS_OBJECT && !dt_masks_object_available())
  {
    dt_control_log(_("AI model is not available. Check preferences > AI"));
    return;
  }
#endif

  _blendop_mask_enable(self);

  // if the clicked shape is already armed, clicking it again disarms it
  if(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget))
     && darktable.develop->form_gui
     && darktable.develop->form_gui->creation
     && darktable.develop->form_gui->creation_module == self)
  {
    darktable.develop->form_gui->creation_continuous = FALSE;
    darktable.develop->form_gui->creation_continuous_module = NULL;
    for(int n = 0; n < DEVELOP_MASKS_NB_SHAPES; n++)
      if(bd->masks_shapes[n])
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_shapes[n]), FALSE);
    dt_masks_change_form_gui(NULL);
    dt_control_queue_redraw_center();
    return;
  }

  // set all shape buttons to inactive
  for(int n = 0; n < DEVELOP_MASKS_NB_SHAPES; n++)
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_shapes[n]), FALSE);

  // we want to be sure that the iop has focus
  dt_iop_request_focus(self);
  dt_iop_color_picker_reset(self, FALSE);
  bd->masks_shown = DT_MASKS_EDIT_FULL;
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_edit), FALSE);
  // we create the new form
  dt_masks_form_t *form = dt_masks_create(bd->masks_type[this]);
  dt_masks_change_form_gui(form);
  darktable.develop->form_gui->creation_module = self;
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(widget), TRUE);
  // make the pending-row placeholder appear immediately (see
  // _build_masks_list's pending-row synthesis / _masks_list_signature)
  _queue_masks_list_rebuild(self);

  if(continuous)
  {
    darktable.develop->form_gui->creation_continuous = TRUE;
    darktable.develop->form_gui->creation_continuous_module = self;
  }

  dt_control_queue_redraw_center();
}

static void _blendop_masks_show_and_edit(GtkGestureSingle *gesture,
                                             gint n_press,
                                             gdouble x,
                                             gdouble y,
                                             dt_iop_module_t *self)
{
  darktable.develop->form_gui->creation_continuous = FALSE;
  darktable.develop->form_gui->creation_continuous_module = NULL;

  dt_iop_gui_blend_data_t *bd = self->blend_data;

  dt_iop_request_focus(self);

  DT_ENTER_GUI_UPDATE();

  dt_iop_color_picker_reset(self, FALSE);

  GdkModifierType state = dt_gui_current_state(gesture);

  if(_module_has_drawn_shapes(self))
  {
    const gboolean control_button_pressed =
      dt_modifier_is(state, GDK_CONTROL_MASK);

    switch(bd->masks_shown)
    {
      case DT_MASKS_EDIT_FULL:
        bd->masks_shown = control_button_pressed
          ? DT_MASKS_EDIT_RESTRICTED
          : DT_MASKS_EDIT_OFF;
        break;

      case DT_MASKS_EDIT_RESTRICTED:
        bd->masks_shown = !control_button_pressed
          ? DT_MASKS_EDIT_FULL
          : DT_MASKS_EDIT_OFF;
        break;

      default:
      case DT_MASKS_EDIT_OFF:
        bd->masks_shown = control_button_pressed
          ? DT_MASKS_EDIT_RESTRICTED
          : DT_MASKS_EDIT_FULL;
    }
  }
  else
  {
    bd->masks_shown = DT_MASKS_EDIT_OFF;
    /* remove hinter messages */
    dt_control_hinter_message("");
  }

  gtk_toggle_button_set_active
    (GTK_TOGGLE_BUTTON(bd->masks_edit), bd->masks_shown != DT_MASKS_EDIT_OFF);
  dt_masks_set_edit_mode(self, bd->masks_shown);

  // set all add shape buttons to inactive
  for(int n = 0; n < DEVELOP_MASKS_NB_SHAPES; n++)
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_shapes[n]), FALSE);

  DT_LEAVE_GUI_UPDATE();
}

// A blend-level color pick. The two shared-editor pickers this used to also
// serve (bd->colorpicker / bd->colorpicker_set_values) went away with the
// classic tabbed blendif editor and were never rebuilt, so both were NULL and
// neither branch could ever match a real picker widget. Every pick that
// reaches this now belongs to a parametric row's own editor.
gboolean blend_color_picker_apply(dt_iop_module_t *module,
                                  GtkWidget *picker,
                                  dt_dev_pixelpipe_t *pipe)
{
  return _param_row_picker_apply(module, picker, pipe);
}

// how many parametric elements the module's mask currently holds.
//
// A parametric form stores the channel layout of the colorspace it was authored
// in (dt_masks_point_parametric_t.colorspace, "the colorspace the form was made
// in"), but the renderer evaluates every form against the module's *current*
// blend_cst -- see the switch in _parametric_get_mask_roi (masks/parametric.c),
// which has to, because the pixel data it is handed is in that colorspace and
// nothing else. So a form cannot survive a colorspace change: its stored
// channel bits would be reinterpreted under a different channel table, and the
// panel would go on displaying "a" with a Lab gradient while the pipe computed
// some unrelated RGB channel.
//
// Neither remapping the channels (Lab a/b have no RGB-display counterpart) nor
// silently dropping the forms is honest, so a colorspace change offers the user
// the one honest option -- delete them -- and does nothing unless they accept
// (see _blendif_change_blend_colorspace).
//
// Referenced by id rather than by pointer: removing a form can also collapse
// the group that held it (see dt_masks_form_remove's emptied-group branch), so
// a pointer collected here can be dangling by the time the next one is removed.
// Both ends are re-resolved at the point of use instead.
typedef struct _parametric_ref_t
{
  dt_mask_id_t grpid;   // the group holding it, which is where it is removed from
  dt_mask_id_t formid;
} _parametric_ref_t;

// every parametric element in the mask, at any nesting depth -- one inside a
// subgroup is just as unable to survive the switch as a top-level one
static void _collect_parametric_forms(dt_masks_form_t *grp,
                                      GList **out,
                                      const int depth)
{
  if(!grp || depth > DT_MASKS_NESTING_MAX) return;
  for(const GList *l = grp->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *const pt = l->data;
    dt_masks_form_t *const f = dt_masks_get_from_id(darktable.develop, pt->formid);
    if(!f) continue;

    if(f->type & DT_MASKS_PARAMETRIC)
    {
      _parametric_ref_t *const ref = malloc(sizeof(_parametric_ref_t));
      ref->grpid = grp->formid;
      ref->formid = f->formid;
      *out = g_list_prepend(*out, ref);
    }
    else if(f->type & (DT_MASKS_GROUP | DT_MASKS_OBJECT))
      _collect_parametric_forms(f, out, depth + 1);
  }
}

static gboolean _blendif_change_blend_colorspace(dt_iop_module_t *module,
                                                 dt_develop_blend_colorspace_t cst)
{
  switch(cst)
  {
    case DEVELOP_BLEND_CS_RAW:
    case DEVELOP_BLEND_CS_LAB:
    case DEVELOP_BLEND_CS_RGB_DISPLAY:
    case DEVELOP_BLEND_CS_RGB_SCENE:
      break;
    default:
      cst = dt_develop_blend_default_module_blend_colorspace(module);
      break;
  }
  if(cst != module->blend_params->blend_cst)
  {
    // Parametric elements cannot come along (see _collect_parametric_forms):
    // deleting them is the only honest outcome, so ask, and switch nothing at
    // all unless the user agrees. This is the authority for every path into a
    // colorspace change -- the menu, a shortcut, a future caller -- rather than
    // something the menu enforces by disabling its own entries.
    GList *parametrics = NULL;
    _collect_parametric_forms(_module_mask_group(module), &parametrics, 0);
    if(parametrics)
    {
      const int n = g_list_length(parametrics);
      const gboolean confirmed = dt_gui_show_yes_no_dialog(
        ngettext("remove parametric element?",
                 "remove parametric elements?", n), "",
        ngettext("this mask has %d parametric element. it stores the channels of"
                 " the colorspace it was created in, and those channels do not"
                 " exist in another colorspace, so it cannot be carried over.\n\n"
                 "change the blend colorspace and remove it?",
                 "this mask has %d parametric elements. they store the channels of"
                 " the colorspace they were created in, and those channels do not"
                 " exist in another colorspace, so they cannot be carried over.\n\n"
                 "change the blend colorspace and remove them?", n), n);
      if(!confirmed)
      {
        g_list_free_full(parametrics, free);
        return FALSE;
      }

      dt_masks_clear_form_gui(darktable.develop);
      for(const GList *l = parametrics; l; l = g_list_next(l))
      {
        const _parametric_ref_t *const ref = l->data;
        // re-resolved per iteration: an earlier removal may have taken the
        // group with it (see the _parametric_ref_t comment)
        dt_masks_form_t *const owner = dt_masks_get_from_id(darktable.develop, ref->grpid);
        dt_masks_form_t *const form = dt_masks_get_from_id(darktable.develop, ref->formid);
        if(owner && form) dt_masks_form_remove(module, owner, form);
      }
      g_list_free_full(parametrics, free);
      dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);

      // the panel's selection and its cached list signature can still name the
      // forms just deleted; dt_iop_gui_update() below rebuilds from these
      dt_iop_gui_blend_data_t *const bdp = module->blend_data;
      if(bdp)
      {
        if(bdp->panel_selected_formid != INVALID_MASKID)
          bdp->panel_selected_formid = INVALID_MASKID;
        bdp->masks_list_sig = DT_INVALID_HASH;
      }
    }

    dt_develop_blend_init_blendif_parameters(module->blend_params, cst);

    // look for last history item for this module with the selected
    // blending mode to copy parametric mask settings
    for(const GList *history = g_list_last(darktable.develop->history);
        history;
        history = g_list_previous(history))
    {
      const dt_dev_history_item_t *data = history->data;
      if(data->module == module && data->blend_params->blend_cst == cst)
      {
        const dt_develop_blend_params_t *hp = data->blend_params;
        dt_develop_blend_params_t *np = module->blend_params;

        np->blend_mode = hp->blend_mode;
        np->blend_parameter = hp->blend_parameter;
        np->blendif = hp->blendif;
        memcpy(np->blendif_parameters,
               hp->blendif_parameters, sizeof(hp->blendif_parameters));
        memcpy(np->blendif_boost_factors,
               hp->blendif_boost_factors, sizeof(hp->blendif_boost_factors));
        break;
      }
    }

    dt_iop_gui_blend_data_t *bd = module->blend_data;
    const dt_iop_colorspace_type_t cst_old = _blendop_blendif_get_picker_colorspace(bd);
    dt_dev_add_new_history_item(darktable.develop, module, FALSE);
    dt_iop_gui_update(module);

    // re-arm a picker that is currently up, so it samples in the new
    // colorspace. This used to test bd->colorpicker/bd->colorpicker_set_values
    // directly, but those belonged to the removed shared editor and were NULL:
    // the pickers that can be live now are a parametric row's own, so ask the
    // picker proxy which module is picking instead of naming widgets.
    if(cst_old != _blendop_blendif_get_picker_colorspace(bd)
       && dt_iop_color_picker_get_active_cst(module) != IOP_CS_NONE)
    {
      dt_iop_color_picker_set_cst(bd->module, _blendop_blendif_get_picker_colorspace(bd));
      dt_dev_reprocess_all(bd->module->dev);
      dt_control_queue_redraw();
    }

    return TRUE;
  }
  return FALSE;
}

static void _masks_opacity_sticky_toggled(GtkToggleButton *mi, dt_iop_module_t *module)
{
  // the checkbox reads "sticky" (on = remember last opacity for new shapes),
  // the conf key is stored inverted (absent/FALSE = sticky, the default) so
  // it needs no preferences.xml entry -- see _new_shape_default_opacity in
  // masks.c, which is the actual place this is consumed.
  const gboolean not_sticky = !gtk_toggle_button_get_active(mi);
  dt_conf_set_bool("plugins/darkroom/masks/opacity_not_sticky", not_sticky);
  if(not_sticky) dt_conf_set_float("plugins/darkroom/masks/opacity", 1.0f);
}

// the channel-preview mode, defined further down with the rest of the hover
// machinery; the panel options menu is built up here
static gboolean _preview_on_hover_is_on(void);
static void _preview_on_hover_set(const gboolean on);

// "auto-expand selected" (masks panel hamburger -> options): whatever is
// selected -- a group, an element, or an element and the group holding it --
// is the one thing expanded, and whatever the option expanded before is
// collapsed. See _auto_expand_selected_row / _auto_expand_selected_group.
static gboolean _auto_expand_selected(void)
{
  return dt_conf_get_bool("plugins/darkroom/masks/auto_expand_selected");
}

// "use sliders for opacity" (nested under auto-expand in the same menu):
// opacity leaves every row and group header and becomes a full slider at the
// top of each expanded panel instead. Defaults off, and is in effect only
// while auto-expand is on -- see _model_opacity_sliders_in_effect, which is
// also what greys the checkbox out.
static gboolean _opacity_sliders(void)
{
  return _model_opacity_sliders_in_effect(
    _auto_expand_selected(), dt_conf_get_bool("plugins/darkroom/masks/opacity_sliders"));
}

// "shape properties in subpanel" (same menu): a drawn shape's properties leave
// its row for a collapsible section of their own (see _props_panel_sync)
static gboolean _shape_props_subpanel(void)
{
  return dt_conf_get_bool("plugins/darkroom/masks/shape_props_subpanel");
}

// the expander options below are all read at row-build time from a conf key,
// not from anything _masks_list_signature hashes (see _make_props_row_toggle,
// _make_shape_row, the group header build) -- without invalidating the cached
// signature here, toggling one would have no visible effect until something
// unrelated next moved the signature.
static void _masks_rebuild_for_option(dt_iop_module_t *module)
{
  if(!module) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(bd) bd->masks_list_sig = DT_INVALID_HASH;
  _queue_masks_list_rebuild(module);
}

static void _masks_auto_expand_selected_toggled(GtkToggleButton *mi,
                                                dt_iop_module_t *module)
{
  const gboolean on = gtk_toggle_button_get_active(mi);
  dt_conf_set_bool("plugins/darkroom/masks/auto_expand_selected", on);
  // the rebuild below applies the option to every row whose expanded state is
  // pure GUI state (see _make_props_row_toggle's build-time rule), but a
  // parametric row's is its stored in_out field, which only
  // _auto_expand_selected_row knows how to move -- so switching the option on
  // with a parametric element selected has to expand it here and now.
  dt_iop_gui_blend_data_t *bd = module ? module->blend_data : NULL;
  if(on && bd) _auto_expand_selected_row(module, bd->panel_selected_formid);
  // "use sliders for opacity" only means anything while this is on, and says so
  // by greying out -- keep that live while the menu is open, rather than only
  // correcting itself the next time it is built
  GtkWidget *child = g_object_get_data(G_OBJECT(mi), "dependent-option");
  if(child) gtk_widget_set_sensitive(child, on);
  _masks_rebuild_for_option(module);
}

static void _masks_opacity_sliders_toggled(GtkToggleButton *mi,
                                           dt_iop_module_t *module)
{
  dt_conf_set_bool("plugins/darkroom/masks/opacity_sliders",
                   gtk_toggle_button_get_active(mi));
  _masks_rebuild_for_option(module);
}

static void _masks_shape_props_subpanel_toggled(GtkToggleButton *mi,
                                                dt_iop_module_t *module)
{
  dt_conf_set_bool("plugins/darkroom/masks/shape_props_subpanel",
                   gtk_toggle_button_get_active(mi));
  // the rebuild refills the subpanel, and shows or hides it (see _props_panel_sync)
  _masks_rebuild_for_option(module);
}

static void _masks_collapse_refinements_default_toggled(GtkToggleButton *mi,
                                                        dt_iop_module_t *module)
{
  dt_conf_set_bool("plugins/darkroom/masks/collapse_refinements_default",
                   gtk_toggle_button_get_active(mi));
}

static void _masks_show_panel_handle_toggled(GtkToggleButton *mi,
                                             dt_iop_module_t *module)
{
  dt_conf_set_bool("plugins/darkroom/masks/show_panel_handle",
                   gtk_toggle_button_get_active(mi));
  // the handle is built once, at view init, so it has to be told
  dt_ui_flexi_panel_update_handle(darktable.gui->ui);
}

static void _masks_preview_on_hover_toggled(GtkToggleButton *mi,
                                            dt_iop_module_t *module)
{
  _preview_on_hover_set(gtk_toggle_button_get_active(mi));
}

// appends the "options" section to `box` -- behavioural toggles for the blend
// mask panel that don't fit the position or colorspace sections. Check buttons
// under a dt_section_label, the way the other toolbar preference popovers are
// laid out (see global_toolbox.c's overlay settings).
#define _MASKS_OPT_CHECK(var, label, tip, active, cb)                             \
  GtkWidget *var = gtk_check_button_new_with_label(label);                        \
  gtk_widget_set_tooltip_text(var, tip);                                          \
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(var), active);                   \
  g_signal_connect(G_OBJECT(var), "toggled", G_CALLBACK(cb), module);             \
  dt_gui_box_add(box, var);

static void _add_masks_panel_options_box(GtkWidget *box, dt_iop_module_t *module)
{
  GtkWidget *header = gtk_label_new(_("options"));
  gtk_label_set_justify(GTK_LABEL(header), GTK_JUSTIFY_CENTER);
  dt_gui_add_class(header, "dt_section_label");
  gtk_widget_set_tooltip_text(header,
                              _("behavioural options for the blend mask panel."));
  dt_gui_box_add(box, header);

  // the checkbox reads "sticky" (on = remember last opacity for new shapes),
  // the conf key is stored inverted (absent/FALSE = sticky, the default) so
  // it needs no preferences.xml entry -- see _new_shape_default_opacity in
  // masks.c, which is the actual place this is consumed.
  _MASKS_OPT_CHECK(
    sticky, _("sticky opacity"),
    _("when enabled (default), a newly added shape starts at the opacity"
      " last used by any shape, so adjusting opacity once carries over to"
      " every shape you add afterwards.\n"
      "when disabled, every newly added shape starts at 100% opacity,"
      " regardless of what opacity was last used."),
    !dt_conf_get_bool("plugins/darkroom/masks/opacity_not_sticky"),
    _masks_opacity_sticky_toggled)

  _MASKS_OPT_CHECK(
    autoexpand, _("auto-expand selected"),
    _("when enabled (default), whatever you select is the one thing expanded,"
      " and whatever was expanded before is collapsed. selecting a group shows"
      " its elements; selecting an element shows its controls, and its group"
      " with it. applies to every element that can be expanded -- shapes and"
      " parametric elements, plus raster masks once \"use sliders for opacity\""
      " below gives them a slider to show.\n"
      "selecting something with nothing to expand leaves whatever is open"
      " open, rather than collapsing the panel down to nothing.\n"
      "when disabled, everything is expanded and collapsed by hand."),
    _auto_expand_selected(), _masks_auto_expand_selected_toggled)

  // "use sliders for opacity" hangs off auto-expand above: it is only in
  // effect while that is on (see _opacity_sliders), so it is indented under it
  // and goes insensitive with it rather than sitting there as an equal that
  // silently does nothing. An insensitive widget receives no events and so
  // shows no tooltip of its own -- the event box around it does, which is the
  // one moment the explanation matters most.
  GtkWidget *opacity_sliders = gtk_check_button_new_with_label(
    _("use sliders for opacity"));
  gtk_toggle_button_set_active(
    GTK_TOGGLE_BUTTON(opacity_sliders),
    dt_conf_get_bool("plugins/darkroom/masks/opacity_sliders"));
  gtk_widget_set_sensitive(opacity_sliders, dt_conf_get_bool(
                             "plugins/darkroom/masks/auto_expand_selected"));
  g_signal_connect(G_OBJECT(opacity_sliders), "toggled",
                   G_CALLBACK(_masks_opacity_sliders_toggled), module);
  g_object_set_data(G_OBJECT(autoexpand), "dependent-option", opacity_sliders);

  GtkWidget *opacity_sliders_slot = gtk_event_box_new();
  gtk_container_add(GTK_CONTAINER(opacity_sliders_slot), opacity_sliders);
  // indented in code rather than by a css class: GtkEventBox allocates itself
  // without a css gadget, so it is one of the widgets GTK3 silently ignores a
  // css margin on. Same indent the dependent check buttons in
  // export_metadata.c use
  gtk_widget_set_margin_start(opacity_sliders_slot, DT_PIXEL_APPLY_DPI(10));
  gtk_widget_set_tooltip_text(
    opacity_sliders_slot,
    _("requires auto-expand above, which is what keeps the panel holding the"
      " slider open.\n"
      "when enabled, opacity leaves the row and group headers and becomes a"
      " full slider at the top of every expanded panel, elements and groups"
      " alike.\n"
      "opacity is a raster mask's only property, so this is also what makes"
      " raster masks expandable at all: with it enabled they carry the same"
      " chevron as every other element.\n"
      "disabled by default."));
  dt_gui_box_add(box, opacity_sliders_slot);

  _MASKS_OPT_CHECK(
    props_subpanel, _("shape properties in subpanel"),
    _("when enabled, the properties of the selected shape (size, feather,"
      " hardness, rotation and the like) are shown in a collapsible section of"
      " their own, between the mask list and the refinements, instead of"
      " expanding under the shape's row. while a shape is being drawn, the"
      " section holds its creation controls and opens by itself; its"
      " placeholder row still shows the group it lands in.\n"
      "the section is empty and disabled while anything but a shape is"
      " selected: parametric elements, raster masks and groups keep their"
      " controls in the list.\n"
      "disabled by default."),
    _shape_props_subpanel(), _masks_shape_props_subpanel_toggled)

  _MASKS_OPT_CHECK(
    hover, _("preview channel under cursor"),
    _("when enabled, resting the pointer on one of the add-parametric-element"
      " buttons displays that channel in the center view, so you can see what"
      " a channel looks like before adding an element for it.\n"
      "the preview only starts after a short pause, so passing over the"
      " buttons on the way elsewhere costs nothing.\n"
      "disabled by default."),
    _preview_on_hover_is_on(),
    _masks_preview_on_hover_toggled)

  _MASKS_OPT_CHECK(
    showhandle, _("show the mask panel's resize handle"),
    _("when enabled (default), the edge of the mask panel floating over the"
      " canvas carries a visible handle with an arrow showing which way the"
      " panel folds away.\n"
      "when disabled, the handle is invisible, like the main panels', and still"
      " resizes the panel by dragging and hides it on a click."),
    dt_conf_get_bool("plugins/darkroom/masks/show_panel_handle"),
    _masks_show_panel_handle_toggled)

  _MASKS_OPT_CHECK(
    collapse, _("collapse refinements by default"),
    _("when enabled, newly selected masks, groups, and elements start with their"
      " refinements section collapsed by default.\n"
      "disabled by default."),
    dt_conf_get_bool("plugins/darkroom/masks/collapse_refinements_default"),
    _masks_collapse_refinements_default_toggled)
}
#undef _MASKS_OPT_CHECK

// a radio in `box`, grouped with `group` (NULL starts a new group), carrying
// `data` under `key` for the toggled handler to read back
static GtkWidget *_masks_pref_radio(GtkWidget *box,
                                    GtkWidget *group,
                                    const gchar *label,
                                    const gchar *key,
                                    const int data,
                                    const gboolean active)
{
  GtkWidget *rb = gtk_radio_button_new_with_label_from_widget(
    group ? GTK_RADIO_BUTTON(group) : NULL, label);
  g_object_set_data(G_OBJECT(rb), key, GINT_TO_POINTER(data));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(rb), active);
  dt_gui_box_add(box, rb);
  return rb;
}

static GtkWidget *_masks_pref_section(GtkWidget *box, const gchar *title, const gchar *tip)
{
  GtkWidget *lb = gtk_label_new(title);
  gtk_label_set_justify(GTK_LABEL(lb), GTK_JUSTIFY_CENTER);
  dt_gui_add_class(lb, "dt_section_label");
  if(tip) gtk_widget_set_tooltip_text(lb, tip);
  dt_gui_box_add(box, lb);
  return lb;
}

static void _blendif_colorspace_radio_toggled(GtkToggleButton *rb,
                                              dt_iop_module_t *module)
{
  if(darktable.gui->reset || !gtk_toggle_button_get_active(rb)) return;
  const dt_develop_blend_colorspace_t cst =
    GPOINTER_TO_INT(g_object_get_data(G_OBJECT(rb), "dt-blend-cst"));
  if(_blendif_change_blend_colorspace(module, cst))
    gtk_widget_queue_draw(module->widget);
}

// The blend mask panel's settings, laid out as a popover of sections the way the
// darkroom toolbar's other preference popovers are (guides, global toolbox): a
// dt_section_label per section, radios for the exclusive choices and check
// buttons for the toggles. This replaced a GtkMenu of check items, which worked
// but looked nothing like the neighbouring popovers it sits between.
static void _blendif_options_callback(GtkButton *button,
                                      dt_iop_module_t *module)
{
  const dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd) return;

  // the blendif color-space section is only meaningful where blendif is supported;
  // the popover itself also opens on masks-only modules (for the other sections)
  const gboolean blendif_ok = bd->blendif_support && bd->blendif_inited;

  GtkWidget *pop = gtk_popover_new(GTK_WIDGET(button));
  GtkWidget *box = dt_gui_vbox();
  gtk_container_add(GTK_CONTAINER(pop), box);

  _masks_pref_section(box, _("blend mask panel settings"), NULL);

  // blend colorspace
  const dt_develop_blend_colorspace_t module_cst =
    dt_develop_blend_default_module_blend_colorspace(module);
  const dt_develop_blend_colorspace_t module_blend_cst = module->blend_params->blend_cst;

  if(blendif_ok
     && (module_cst == DEVELOP_BLEND_CS_LAB || module_cst == DEVELOP_BLEND_CS_RGB_DISPLAY
         || module_cst == DEVELOP_BLEND_CS_RGB_SCENE))
  {
    _masks_pref_section(box, _("blend colorspace"), NULL);

    // every entry here stays live even when the mask holds parametric elements
    // that cannot survive the switch: _blendif_change_blend_colorspace asks
    // about those and deletes them on a yes.
    ++darktable.gui->reset;
    GtkWidget *g = _masks_pref_radio(box, NULL, _("default"), "dt-blend-cst",
                                     DEVELOP_BLEND_CS_NONE,
                                     module_blend_cst == DEVELOP_BLEND_CS_NONE);
    GtkWidget *radios[4] = { g, NULL, NULL, NULL };
    int n = 1;
    // only offer Lab blending on a Lab module, to avoid using it at the wrong
    // place (it should not be active for RGB modules before colorin/after colorout)
    if(module_cst == DEVELOP_BLEND_CS_LAB)
      radios[n++] = _masks_pref_radio(box, g, _("Lab"), "dt-blend-cst",
                                      DEVELOP_BLEND_CS_LAB,
                                      module_blend_cst == DEVELOP_BLEND_CS_LAB);
    radios[n++] = _masks_pref_radio(box, g, _("RGB (display)"), "dt-blend-cst",
                                    DEVELOP_BLEND_CS_RGB_DISPLAY,
                                    module_blend_cst == DEVELOP_BLEND_CS_RGB_DISPLAY);
    radios[n++] = _masks_pref_radio(box, g, _("RGB (scene)"), "dt-blend-cst",
                                    DEVELOP_BLEND_CS_RGB_SCENE,
                                    module_blend_cst == DEVELOP_BLEND_CS_RGB_SCENE);
    --darktable.gui->reset;
    // connected only after the initial states are set, so building the popover
    // does not look like the user picking a colorspace
    for(int i = 0; i < n; i++)
      g_signal_connect(G_OBJECT(radios[i]), "toggled",
                       G_CALLBACK(_blendif_colorspace_radio_toggled), module);
  }

  if(bd->masks_support)
  {
    _add_masks_panel_position_box(box, module);
    _add_masks_panel_options_box(box, module);
  }

  gtk_widget_show_all(box);
  gtk_popover_popup(GTK_POPOVER(pop));

  // the anchor is not always a DtGtkButton: this also opens from a right-click
  // on the darkroom toolbar's mask-panel toggle, which is a DtGtkToggleButton
  if(DTGTK_IS_BUTTON(button)) dtgtk_button_set_active(DTGTK_BUTTON(button), FALSE);
}

void dt_iop_gui_blend_masks_options_popup(GtkButton *button, gpointer user_data)
{
  dt_iop_module_t *module = darktable.develop ? darktable.develop->gui_module : NULL;
  if(!module) module = darktable.develop->proxy.masks_flexi_host.hosted_module;
  if(module)
  {
    _blendif_options_callback(button, module);
  }
  else
  {
    // no focused module: the panel-wide settings still apply, so show those
    // alone. Same popover shape as the full one, minus every section that needs
    // a module to talk about.
    GtkWidget *pop = gtk_popover_new(GTK_WIDGET(button));
    GtkWidget *box = dt_gui_vbox();
    gtk_container_add(GTK_CONTAINER(pop), box);
    _masks_pref_section(box, _("blend mask panel settings"), NULL);
    _add_masks_panel_position_box(box, NULL);
    _add_masks_panel_options_box(box, NULL);
    gtk_widget_show_all(box);
    gtk_popover_popup(GTK_POPOVER(pop));
    if(DTGTK_IS_BUTTON(button)) dtgtk_button_set_active(DTGTK_BUTTON(button), FALSE);
  }
}

// resolve the DT_DEV_PIXELPIPE_DISPLAY_* channel bit for a flexi parametric
// row's own slider -- each row is single-channel, so there is no shared
// "current tab" to read (that was the removed classic editor's model, see
// dt_iop_gui_init_blendif's removal note on data->channel/data->tab in
// dt_iop_gui_blend_data_t, blend.h). `widget` must be tagged with its owning
// dt_masks_param_row_editor_t via "param-row-editor" (see
// _build_param_row_editor). Returns FALSE (leaving *channel_out untouched) if
// the row/form cannot be resolved, so callers degrade to "no channel view"
// instead of dereferencing anything -- this is what used to crash on
// 'c'/'C'/'m'/'M' (and shift-hover) over a row's slider.
// shared resolver: finds the flexi parametric row (and its channel-table
// entry) owning `widget`, tagged via "param-row-editor" (see
// _build_param_row_editor). Used by _param_row_editor_channel below and by
// the key-press handler's alt-display case, both of which used to read the
// removed classic editor's data->channel[data->tab] (see the removal note on
// dt_iop_gui_blend_data_t in blend.h) -- permanently NULL now, hence the crash.
static const dt_masks_param_row_editor_t *_param_row_editor_resolve(
  GtkWidget *widget, const dt_iop_gui_blendif_channel_t **channels_out, int *ch_out)
{
  const dt_masks_param_row_editor_t *ed =
    g_object_get_data(G_OBJECT(widget), "param-row-editor");
  if(!ed) return NULL;
  dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, ed->formid);
  const dt_masks_point_parametric_t *p = form && form->points ? form->points->data : NULL;
  if(!p) return NULL;
  const dt_iop_gui_blendif_channel_t *channels =
    dt_develop_blendif_channels_for_csp((int)p->colorspace);
  if(!channels) return NULL;
  int nch = 0;
  while(channels[nch].label) nch++;
  if((int)p->channel < 0 || (int)p->channel >= nch) return NULL;
  *channels_out = channels;
  *ch_out = (int)p->channel;
  return ed;
}

static gboolean _param_row_editor_channel(GtkWidget *widget,
                                          dt_dev_pixelpipe_display_mask_t *channel_out)
{
  const dt_iop_gui_blendif_channel_t *channels;
  int ch;
  const dt_masks_param_row_editor_t *ed =
    _param_row_editor_resolve(widget, &channels, &ch);
  if(!ed) return FALSE;
  dt_dev_pixelpipe_display_mask_t channel = channels[ch].display_channel;
  if(widget == GTK_WIDGET(ed->filter[1].slider))
    channel |= DT_DEV_PIXELPIPE_DISPLAY_OUTPUT;
  *channel_out = channel;
  return TRUE;
}

// toggle channel/mask view
static void _blendop_blendif_channel_mask_view_toggle
  (GtkWidget *widget,
   dt_iop_module_t *module,
   const dt_dev_pixelpipe_display_mask_t mode)
{
  dt_dev_pixelpipe_display_mask_t new_request_mask_display =
    module->request_mask_display;

  // toggle mode
  if(module->request_mask_display & mode)
    new_request_mask_display &= ~mode;
  else
    new_request_mask_display |= mode;

  new_request_mask_display &= ~DT_DEV_PIXELPIPE_DISPLAY_ANY;

  // in case user requests channel display: get the channel
  if(new_request_mask_display & DT_DEV_PIXELPIPE_DISPLAY_CHANNEL)
  {
    dt_dev_pixelpipe_display_mask_t channel;
    if(_param_row_editor_channel(widget, &channel))
    {
      new_request_mask_display &= ~DT_DEV_PIXELPIPE_DISPLAY_ANY;
      new_request_mask_display |= channel;
    }
    else
      new_request_mask_display &= ~DT_DEV_PIXELPIPE_DISPLAY_CHANNEL;
  }

  if(new_request_mask_display != module->request_mask_display)
  {
    module->request_mask_display = new_request_mask_display;
    dt_iop_refresh_center(module);
  }
}


// "preview channel under cursor" is a latched mode rather than a held key:
// while it is on, hovering an "add channel" button shows that channel on the
// center view, and leaving restores whatever was displayed before. The state is
// global (one working mode, not a per-module setting) so it survives moving
// between modules; it lives in the panel options menu, and
// _shortcut_toggle_preview_on_hover makes it bindable.
#define BLEND_PREVIEW_ON_HOVER_CONF "plugins/darkroom/blend/preview_channel_on_hover"

// how long the pointer has to rest on a button before the preview fires. Each
// preview costs a pipeline reprocess, so sweeping the pointer across the row of
// channel buttons on the way somewhere else must not queue one per button --
// only a deliberate pause asks for anything. 100ms did that job but was below
// the threshold where a delay is noticed at all, so the preview read as firing
// on contact and there was no way to cross the row without triggering the one
// button the pointer happened to slow down over.
#define BLEND_PREVIEW_ON_HOVER_DWELL_MS 500

static gboolean _preview_on_hover_is_on(void)
{
  return dt_conf_get_bool(BLEND_PREVIEW_ON_HOVER_CONF);
}

// drop a pending dwell timer, if any. Must be called from every path that
// tears down or unhovers, or the timer fires into freed blend_data.
static void _preview_on_hover_cancel_dwell(dt_iop_gui_blend_data_t *bd)
{
  if(!bd || !bd->preview_dwell_timer) return;
  g_source_remove(bd->preview_dwell_timer);
  bd->preview_dwell_timer = 0;
}

// bring module->request_mask_display in line with the mode and what is hovered:
// show the hovered channel while both hold, and restore what was displayed
// before as soon as either stops holding
static void _preview_on_hover_apply(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd) return;

  // hovered_channel_display is NONE when the widget's channel could not be
  // resolved; there is nothing to preview then
  const gboolean want_preview =
    _preview_on_hover_is_on()
    && bd->hovered_channel_widget
    && bd->hovered_channel_display != DT_DEV_PIXELPIPE_DISPLAY_NONE;

  dt_dev_pixelpipe_display_mask_t wanted;

  dt_pthread_mutex_lock(&bd->lock);
  if(want_preview)
  {
    // first frame of a preview: remember what to come back to
    if(!bd->hover_preview_active) bd->save_for_leave = module->request_mask_display;
    bd->hover_preview_active = TRUE;
    wanted = DT_DEV_PIXELPIPE_DISPLAY_CHANNEL | bd->hovered_channel_display;
  }
  else
  {
    // with no preview up there is nothing of ours to take down: leave
    // request_mask_display to whoever else set it
    wanted = bd->hover_preview_active
               ? bd->save_for_leave
               : module->request_mask_display;
    bd->hover_preview_active = FALSE;
  }
  dt_pthread_mutex_unlock(&bd->lock);

  if(module->request_mask_display != wanted)
  {
    module->request_mask_display = wanted;
    dt_iop_refresh_center(module);
  }
}

static gboolean _preview_on_hover_dwell_elapsed(gpointer user_data)
{
  dt_iop_module_t *module = user_data;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd) return G_SOURCE_REMOVE;

  bd->preview_dwell_timer = 0;
  _preview_on_hover_apply(module);
  return G_SOURCE_REMOVE;
}

// shared by both hovered widget kinds: the parametric range sliders below and
// the "add channel" buttons in _rebuild_param_channel_buttons, which resolve
// their channel differently. Only the buttons carry a channel to preview; the
// sliders pass NONE and are here purely to register the hover.
static void _preview_on_hover_enter(dt_iop_module_t *module,
                                    GtkWidget *widget,
                                    const dt_dev_pixelpipe_display_mask_t channel)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;

  bd->hovered_channel_widget = widget;
  bd->hovered_channel_display = channel;
  gtk_widget_grab_focus(widget);

  // re-arm on every enter, so sweeping across the button row only ever fires
  // for the button the pointer actually settles on. Whatever was previewed
  // before stays up meanwhile, rather than flickering off and on.
  _preview_on_hover_cancel_dwell(bd);
  if(_preview_on_hover_is_on() && channel != DT_DEV_PIXELPIPE_DISPLAY_NONE)
    bd->preview_dwell_timer = g_timeout_add(BLEND_PREVIEW_ON_HOVER_DWELL_MS,
                                            _preview_on_hover_dwell_elapsed, module);
  else
    _preview_on_hover_apply(module);  // nothing to show: take any preview down now
}

static void _preview_on_hover_leave(dt_iop_module_t *module, GtkWidget *widget)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;

  // a leave for a widget we are no longer tracking would undo the enter that
  // has since taken over (adjacent buttons deliver enter before leave)
  if(bd->hovered_channel_widget != widget) return;

  bd->hovered_channel_widget = NULL;
  _preview_on_hover_cancel_dwell(bd);
  _preview_on_hover_apply(module);
}

// set the mode and let whichever module is hovering an add-channel button
// right now pick the change up without moving the pointer. Shared by the panel
// options menu item and the shortcut action (_shortcut_toggle_preview_on_hover).
static void _preview_on_hover_set(const gboolean on)
{
  dt_conf_set_bool(BLEND_PREVIEW_ON_HOVER_CONF, on);

  for(GList *m = darktable.develop ? darktable.develop->iop : NULL;
      m;
      m = g_list_next(m))
  {
    dt_iop_module_t *mod = m->data;
    dt_iop_gui_blend_data_t *bd = mod->blend_data;
    _preview_on_hover_cancel_dwell(bd);
    _preview_on_hover_apply(mod);
    // the channel buttons say whether resting on them previews, so their
    // tooltips have to be rebuilt when that stops being true
    if(bd && bd->masks_param_channels_inner) _update_add_target_sensitivity(mod);
  }
}

static void _blendop_blendif_enter_cb(GtkEventControllerMotion *controller,
                                      double x, double y,
                                      dt_iop_module_t *module)
{
  DT_GUARD_GUI_UPDATE();

  // the sliders are where the work happens: previewing on the way to grabbing
  // one costs a pipeline reprocess nobody asked for, so they only register the
  // hover (the a/m keys need it, see _blendop_blendif_key_press_cb) and leave
  // the channel unresolved -- _preview_on_hover_apply reads NONE as "nothing to
  // preview". The add-channel buttons are the ones that preview.
  _preview_on_hover_enter(module, dt_gui_get_widget(controller),
                          DT_DEV_PIXELPIPE_DISPLAY_NONE);
}

static void _blendop_blendif_leave_cb(GtkEventControllerMotion *controller,
                                      dt_iop_module_t *module)
{
  DT_GUARD_GUI_UPDATE();

  _preview_on_hover_leave(module, dt_gui_get_widget(controller));
}

static gboolean _blendop_blendif_key_press_cb(GtkEventControllerKey *controller,
                                              guint keyval,
                                              guint keycode,
                                              GdkModifierType state,
                                              dt_iop_module_t *module)
{
  if(dt_atomic_get_int(&darktable.gui->reset) != 0) return FALSE;

  GtkWidget *widget = dt_gui_get_widget(controller);
  dt_iop_gui_blend_data_t *data = module->blend_data;
  if(data->hovered_channel_widget != widget) return FALSE;
  gboolean handled = FALSE;

  switch(keyval)
  {
    case GDK_KEY_a:
    case GDK_KEY_A:
    {
      const dt_iop_gui_blendif_channel_t *channels;
      int ch;
      const dt_masks_param_row_editor_t *row_ed =
        _param_row_editor_resolve(widget, &channels, &ch);
      // data->altmode is sized [8][2] (the old classic tab-count ceiling);
      // reused here indexed by the row's own channel instead of a tab.
      if(row_ed && ch >= 0 && ch < 8 && channels[ch].altdisplay)
      {
        const int io = (widget == GTK_WIDGET(row_ed->filter[1].slider)) ? 1 : 0;
        data->altmode[ch][io] =
          channels[ch].altdisplay(widget, module, data->altmode[ch][io] + 1);
      }
      handled = TRUE;
      break;
    }
    case GDK_KEY_m:
    case GDK_KEY_M:
      _blendop_blendif_channel_mask_view_toggle
        (widget, module,
         DT_DEV_PIXELPIPE_DISPLAY_MASK);
      handled = TRUE;
      break;
  }

  if(handled)
    dt_iop_request_focus(module);

  return handled;
}


#define COLORSTOPS(gradient) sizeof(gradient) / sizeof(dt_iop_gui_blendif_colorstop_t), \
                             gradient

const dt_iop_gui_blendif_channel_t Lab_channels[]
    = { { N_("L"), N_("add a parametric element of type L"), 1.0f / 100.0f,
            COLORSTOPS(_gradient_L), TRUE, 0.0f,
          { DEVELOP_BLENDIF_L_in, DEVELOP_BLENDIF_L_out }, DT_DEV_PIXELPIPE_DISPLAY_L,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("lightness") },
        { N_("a"), N_("add a parametric element of type a"), 1.0f / 256.0f,
          COLORSTOPS(_gradient_a), TRUE, 0.0f,
          { DEVELOP_BLENDIF_A_in, DEVELOP_BLENDIF_A_out }, DT_DEV_PIXELPIPE_DISPLAY_a,
          _blendif_scale_print_ab, _blendop_blendif_disp_alternative_mag,
          N_("green/red") },
        { N_("b"), N_("add a parametric element of type b"), 1.0f / 256.0f,
          COLORSTOPS(_gradient_b), TRUE, 0.0f,
          { DEVELOP_BLENDIF_B_in, DEVELOP_BLENDIF_B_out }, DT_DEV_PIXELPIPE_DISPLAY_b,
          _blendif_scale_print_ab, _blendop_blendif_disp_alternative_mag,
          N_("blue/yellow") },
        { N_("C"), N_("add a parametric element of type C"), 1.0f / 100.0f,
          COLORSTOPS(_gradient_chroma),
          TRUE, 0.0f,
          { DEVELOP_BLENDIF_C_in, DEVELOP_BLENDIF_C_out }, DT_DEV_PIXELPIPE_DISPLAY_LCH_C,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("saturation") },
        { N_("h"), N_("add a parametric element of type h"), 1.0f / 360.0f,
          COLORSTOPS(_gradient_LCh_hue),
          FALSE, 0.0f,
          { DEVELOP_BLENDIF_h_in, DEVELOP_BLENDIF_h_out }, DT_DEV_PIXELPIPE_DISPLAY_LCH_h,
          _blendif_scale_print_hue, NULL, N_("hue") },
        { NULL } };

const dt_iop_gui_blendif_channel_t rgb_channels[]
    = { { N_("g"), N_("add a parametric element of type g"), 1.0f / 255.0f,
            COLORSTOPS(_gradient_gray), TRUE, 0.0f,
          { DEVELOP_BLENDIF_GRAY_in, DEVELOP_BLENDIF_GRAY_out },
          DT_DEV_PIXELPIPE_DISPLAY_GRAY,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("gray") },
        { N_("R"), N_("add a parametric element of type R"), 1.0f / 255.0f,
          COLORSTOPS(_gradient_red), TRUE, 0.0f,
          { DEVELOP_BLENDIF_RED_in, DEVELOP_BLENDIF_RED_out },
          DT_DEV_PIXELPIPE_DISPLAY_R,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("red") },
        { N_("G"), N_("add a parametric element of type G"), 1.0f / 255.0f,
          COLORSTOPS(_gradient_green), TRUE, 0.0f,
          { DEVELOP_BLENDIF_GREEN_in, DEVELOP_BLENDIF_GREEN_out },
          DT_DEV_PIXELPIPE_DISPLAY_G,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("green") },
        { N_("B"), N_("add a parametric element of type B"), 1.0f / 255.0f,
          COLORSTOPS(_gradient_blue), TRUE, 0.0f,
          { DEVELOP_BLENDIF_BLUE_in, DEVELOP_BLENDIF_BLUE_out },
          DT_DEV_PIXELPIPE_DISPLAY_B,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("blue") },
        { N_("H"), N_("add a parametric element of type H"), 1.0f / 360.0f,
          COLORSTOPS(_gradient_HSL_hue),
          FALSE, 0.0f,
          { DEVELOP_BLENDIF_H_in, DEVELOP_BLENDIF_H_out },
          DT_DEV_PIXELPIPE_DISPLAY_HSL_H,
          _blendif_scale_print_hue, NULL,
          N_("hue") },
        { N_("S"), N_("add a parametric element of type S"), 1.0f / 100.0f,
          COLORSTOPS(_gradient_chroma),
          FALSE, 0.0f,
          { DEVELOP_BLENDIF_S_in, DEVELOP_BLENDIF_S_out },
          DT_DEV_PIXELPIPE_DISPLAY_HSL_S,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("chroma") },
        { N_("L"), N_("add a parametric element of type L"), 1.0f / 100.0f,
          COLORSTOPS(_gradient_gray),
          FALSE, 0.0f,
          { DEVELOP_BLENDIF_l_in, DEVELOP_BLENDIF_l_out },
          DT_DEV_PIXELPIPE_DISPLAY_HSL_l,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("luminance") },
        { NULL } };

const dt_iop_gui_blendif_channel_t rgbj_channels[]
    = { { N_("g"), N_("add a parametric element of type g"), 1.0f / 255.0f,
            COLORSTOPS(_gradient_gray), TRUE, 0.0f,
          { DEVELOP_BLENDIF_GRAY_in, DEVELOP_BLENDIF_GRAY_out },
          DT_DEV_PIXELPIPE_DISPLAY_GRAY,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("gray") },
        { N_("R"), N_("add a parametric element of type R"), 1.0f / 255.0f,
          COLORSTOPS(_gradient_red), TRUE, 0.0f,
          { DEVELOP_BLENDIF_RED_in, DEVELOP_BLENDIF_RED_out },
          DT_DEV_PIXELPIPE_DISPLAY_R,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("red") },
        { N_("G"), N_("add a parametric element of type G"), 1.0f / 255.0f,
          COLORSTOPS(_gradient_green), TRUE, 0.0f,
          { DEVELOP_BLENDIF_GREEN_in, DEVELOP_BLENDIF_GREEN_out },
          DT_DEV_PIXELPIPE_DISPLAY_G,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("green") },
        { N_("B"), N_("add a parametric element of type B"), 1.0f / 255.0f,
          COLORSTOPS(_gradient_blue), TRUE, 0.0f,
          { DEVELOP_BLENDIF_BLUE_in, DEVELOP_BLENDIF_BLUE_out },
          DT_DEV_PIXELPIPE_DISPLAY_B,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("blue") },
        { N_("Jz"), N_("add a parametric element of type Jz"), 1.0f / 100.0f,
          COLORSTOPS(_gradient_gray),
          TRUE, -6.64385619f, // cf. _blend_init_blendif_boost_parameters
          { DEVELOP_BLENDIF_Jz_in, DEVELOP_BLENDIF_Jz_out },
          DT_DEV_PIXELPIPE_DISPLAY_JzCzhz_Jz,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("luminance") },
        { N_("Cz"), N_("add a parametric element of type Cz"), 1.0f / 100.0f,
          COLORSTOPS(_gradient_chroma),
          TRUE, -6.64385619f, // cf. _blend_init_blendif_boost_parameters
          { DEVELOP_BLENDIF_Cz_in, DEVELOP_BLENDIF_Cz_out },
          DT_DEV_PIXELPIPE_DISPLAY_JzCzhz_Cz,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("chroma") },
        { N_("hz"), N_("add a parametric element of type hz"), 1.0f / 360.0f,
          COLORSTOPS(_gradient_JzCzhz_hue),
          FALSE, 0.0f,
          { DEVELOP_BLENDIF_hz_in, DEVELOP_BLENDIF_hz_out },
          DT_DEV_PIXELPIPE_DISPLAY_JzCzhz_hz,
          _blendif_scale_print_hue, NULL,
          N_("hue") },
        { NULL } };

// the channel descriptor array for a blend colorspace (NULL-terminated). Mirrors
// the switch in dt_iop_gui_update_blendif; used by the single-channel parametric
// machinery (add-buttons + editor lock) to enumerate / index channels.
// exported (see blend.h) so parametric.c can label a single-channel form's
// name after its channel without duplicating the per-colorspace arrays above
const dt_iop_gui_blendif_channel_t *dt_develop_blendif_channels_for_csp(const int csp)
{
  switch(csp)
  {
  case DEVELOP_BLEND_CS_LAB: return Lab_channels;
  case DEVELOP_BLEND_CS_RGB_DISPLAY: return rgb_channels;
  case DEVELOP_BLEND_CS_RGB_SCENE: return rgbj_channels;
  default: return NULL;
  }
}

const char *slider_tooltip[] =
  { N_("adjustment based on input image received by this module:\n"
       "- upper markers: full opacity (100% mask)\n"
       "- lower markers: zero opacity (0% mask)\n"
       "- between upper/lower markers: opacity transition\n\n"
       "drag marker to adjust (shift+drag to move range)\n"
       "right-click marker for precise numeric entry\n"
       "double-click to reset\n"
       "press 'm' to toggle mask view\n"
       "press 'a' to toggle display modes"),
    N_("adjustment based on unblended output of this module:\n"
       "- upper markers: full opacity (100% mask)\n"
       "- lower markers: zero opacity (0% mask)\n"
       "- between upper/lower markers: opacity transition\n\n"
       "drag marker to adjust (shift+drag to move range)\n"
       "right-click marker for precise numeric entry\n"
       "double-click to reset\n"
       "press 'm' to toggle mask view\n"
       "press 'a' to toggle display modes") };

static void _rebuild_param_channel_buttons(dt_iop_module_t *module);

void dt_iop_gui_update_masks(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_develop_blend_params_t *bp = module->blend_params;

  if(!bd || !bd->masks_support || !bd->masks_inited) return;

  DT_ENTER_GUI_UPDATE();

  /* update masks state */
  const gboolean flexi = bp->mask_mode & DEVELOP_MASK_FLEXI;
  dt_masks_form_t *grp =
    dt_masks_get_from_id(darktable.develop, module->blend_params->mask_id);
  // classic drawn mode has nothing to edit on canvas without shapes
  if(!flexi && !(grp && (grp->type & DT_MASKS_GROUP) && grp->points))
  {
    bd->masks_shown = DT_MASKS_EDIT_OFF;
    dt_masks_set_edit_mode(module, DT_MASKS_EDIT_OFF);
  }

  if(bd->masks_support)
  {
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_edit),
                                 bd->masks_shown != DT_MASKS_EDIT_OFF);
  }

  // update buttons status
  for(int n = 0; n < DEVELOP_MASKS_NB_SHAPES; n++)
  {
    if(module->dev->form_gui && module->dev->form_visible
       && module->dev->form_gui->creation
       && module->dev->form_gui->creation_module == module
       && (module->dev->form_visible->type & bd->masks_type[n]))
    {
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_shapes[n]), TRUE);
    }
    else
    {
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_shapes[n]), FALSE);
    }
  }

  DT_LEAVE_GUI_UPDATE();

  // a panel/history/image update may have swapped the mask group out from under
  // us (and does not go through _build_masks_list); resync the scope combo and
  // reload the refinement controls for the active scope.
  _refine_scope_combo_rebuild(module);
}

// ===========================================================================
// In-module per-shape composition list + parametric (blendif) forms (Phase 3)
// ---------------------------------------------------------------------------
// A compact list of the module's mask-group shapes is shown in the module's own
// mask section, each row carrying a composition-operator chooser and an inverse
// toggle, plus reordering. Parametric masks can be added like a shape and, when
// selected, bind the module's existing parametric channel editor to that form
// (see _blendif_commit). All of this only ever touches new parametric forms /
// explicit user actions, so legacy edits are unaffected.

// destroys and rebuilds the whole mask-list widget tree, including the very
// widget a drag-and-drop just landed on. Doing that synchronously from inside
// a "drag-data-received" handler races the macOS (quartz) backend's own
// teardown of the just-finished NSDraggingSession -- gtk_drag_finish() returns
// before Cocoa is fully done referencing the source view, and destroying it
// right away has been observed to abort a *later* drag deep inside
// _gdk_quartz_window_drag_begin. Deferred to the next main-loop iteration
// (after the drag machinery has fully unwound) via g_idle_add instead of a
// direct call, for every DnD receive handler that rebuilds the list.
static gboolean _rebuild_masks_list_idle(gpointer user_data)
{
  dt_iop_module_t *module = user_data;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  // clear the pending guard *before* rebuilding so any request raised during the
  // rebuild itself still queues a fresh pass rather than being dropped.
  if(bd)
  {
    bd->masks_rebuild_pending = FALSE;
    bd->masks_rebuild_idle_id = 0;
  }
  _build_masks_list(module);
  return G_SOURCE_REMOVE;
}

// Queue a single deferred mask-list rebuild, coalescing repeated requests within
// one main-loop turn: one user gesture can raise several rebuild requests (a DnD
// receive that reorders and reselects, an op that also emits a history item),
// and each raw g_idle_add would otherwise run a full teardown/rebuild. The guard
// is cleared when the idle fires (see _rebuild_masks_list_idle). The source id is
// also kept so dt_iop_gui_cleanup_blending can cancel it if the module is torn
// down before the idle gets a chance to run (darkroom exit/app quit) -- an idle
// callback left dangling past teardown dereferences already-destroyed widgets.
static void _queue_masks_list_rebuild(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(bd)
  {
    if(bd->masks_rebuild_pending) return;
    bd->masks_rebuild_pending = TRUE;
    bd->masks_rebuild_idle_id = g_idle_add(_rebuild_masks_list_idle, (gpointer)module);
    return;
  }
  g_idle_add(_rebuild_masks_list_idle, (gpointer)module);
}

// the modules sharing a form all show its chain icon, so a change of who uses
// it (import, unlink, delete) must reach their panels too. Their signatures
// fold the users (see _masks_list_signature), so the unaffected ones skip
static void _queue_link_peers_rebuild(const dt_iop_module_t *module)
{
  for(GList *l = darktable.develop->iop; l; l = g_list_next(l))
  {
    dt_iop_module_t *m = l->data;
    if(m != module && m->blend_data) _queue_masks_list_rebuild(m);
  }
}

// defined below (needs _group_point etc.); declared here so the group's
// "solo" menu item can call it directly.
static void _toggle_solo_group(dt_iop_module_t *module, const dt_mask_id_t cid);
// defined below (needs _param_row_point etc.); declared here so
// _solo_badge_form_press (which comes first in the file) can call it
// directly to clear solo-edit from a click on its own status badge.
static void _toggle_soloedit(dt_iop_module_t *module, const dt_mask_id_t id);
static void _toggle_element_disable(dt_iop_module_t *module, const dt_mask_id_t id);

static gboolean _form_has_drawn_shape(const dt_masks_form_t *form, const int depth)
{
  if(!form || depth > DT_MASKS_NESTING_MAX) return FALSE;
  if(form->type & (DT_MASKS_CIRCLE | DT_MASKS_PATH | DT_MASKS_GRADIENT | DT_MASKS_ELLIPSE
                   | DT_MASKS_BRUSH
#ifdef HAVE_AI
                   | DT_MASKS_OBJECT
#endif
     ))
    return TRUE;

  if(form->type & DT_MASKS_GROUP)
  {
    for(const GList *l = form->points; l; l = g_list_next(l))
    {
      const dt_masks_point_group_t *pt = l->data;
      const dt_masks_form_t *child = dt_masks_get_from_id(darktable.develop, pt->formid);
      if(_form_has_drawn_shape(child, depth + 1)) return TRUE;
    }
  }
  return FALSE;
}

static gboolean _module_has_drawn_shapes(const dt_iop_module_t *module)
{
  if(!module || !module->blend_params) return FALSE;
  const dt_masks_form_t *grp =
    dt_masks_get_from_id(darktable.develop, module->blend_params->mask_id);
  return _form_has_drawn_shape(grp, 0);
}

dt_masks_form_t *_module_mask_group(dt_iop_module_t *module)
{
  if(!module || !module->blend_params) return NULL;
  dt_masks_form_t *grp =
    dt_masks_get_from_id(darktable.develop, module->blend_params->mask_id);
  return (grp && (grp->type & DT_MASKS_GROUP)) ? grp : NULL;
}

// A flexi mask with no group form yet shows one empty group all the same (see
// _masks_panel_pack). The form, with that group's marker, is created here, the
// first time something is written to the group, and not when the panel merely
// shows it: only a masks history item records a new form, and a module's own
// history item would record a mask id naming nothing. Callers commit with the
// module, so its mask id is recorded too.
dt_masks_form_t *_module_flexi_group(dt_iop_module_t *module, dt_mask_id_t *cid)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp && module && darktable.develop)
    grp = dt_masks_module_group_create(darktable.develop, module);
  if(!grp) return NULL;
  dt_masks_group_ensure_marker(darktable.develop->forms, grp);
  if(cid && !dt_is_valid_maskid(*cid) && grp->points)
    *cid = ((dt_masks_point_group_t *)grp->points->data)->formid;
  return grp;
}

// does the mask have elements, not just groups?
static gboolean _mask_has_elements(const dt_masks_form_t *grp)
{
  for(const GList *l = grp ? grp->points : NULL; l; l = g_list_next(l))
    if(!dt_masks_point_is_marker(l->data)) return TRUE;
  return FALSE;
}

// the AI object the canvas is stepped into, or INVALID_MASKID
static dt_mask_id_t _entered_object(void)
{
  const dt_masks_form_gui_t *gui = darktable.develop ? darktable.develop->form_gui : NULL;
  return gui ? gui->entered_object : INVALID_MASKID;
}

// the paths of AI object `obj`, its marker aside
static int _object_path_count(const dt_masks_form_t *obj)
{
  int n = 0;
  for(const GList *l = obj ? obj->points : NULL; l; l = g_list_next(l))
    if(!dt_masks_point_is_marker(l->data)) n++;
  return n;
}

// the list node of point `id` -- a member or a marker -- of the mask `grp`, at
// any depth, and the group form whose list holds it. The list of `grp` itself
// is read directly first, so the paths of an AI object, whose form is no
// group, are found too
static GList *_point_node_owner(dt_masks_form_t *grp,
                                const dt_mask_id_t id,
                                dt_masks_form_t **owner)
{
  for(GList *l = grp ? grp->points : NULL; l; l = g_list_next(l))
    if(((dt_masks_point_group_t *)l->data)->formid == id)
    {
      if(owner) *owner = grp;
      return l;
    }
  GList *found = grp && (grp->type & DT_MASKS_GROUP)
                   ? dt_masks_group_find_node(darktable.develop ? darktable.develop->forms
                                                                : NULL,
                                              grp, id, owner)
                   : NULL;
  // an AI object is no group, so the walk above does not enter it. The one
  // stepped into shows its paths as rows of its own group, which act on their
  // points like any other rows (see _make_shape_row)
  const dt_mask_id_t entered = _entered_object();
  if(!found && dt_is_valid_maskid(entered) && entered != id && grp
     && grp->formid != entered)
  {
    dt_masks_form_t *obj = dt_masks_get_from_id(darktable.develop, entered);
    if(obj && _point_node_owner(grp, entered, NULL))
      found = _point_node_owner(obj, id, owner);
  }
  return found;
}

// the node of point `pt` itself at any depth, with the list holding it. A mask
// can hold the same shape twice, so where a row knows its own reference this
// finds that one, not the first with its form id as _point_node_owner does.
// `pt` is only compared, never read
static GList *_point_node_at(dt_masks_form_t *grp,
                             const dt_masks_point_group_t *pt,
                             dt_masks_form_t **owner,
                             const int depth)
{
  if(!grp || !pt || depth > DT_MASKS_NESTING_MAX) return NULL;
  for(GList *l = grp->points; l; l = g_list_next(l))
    if(l->data == pt)
    {
      if(owner) *owner = grp;
      return l;
    }
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *p = l->data;
    if(dt_masks_point_is_marker(p)) continue;
    dt_masks_form_t *f = dt_masks_get_from_id(darktable.develop, p->formid);
    if(f && f != grp && (f->type & (DT_MASKS_GROUP | DT_MASKS_OBJECT)))
    {
      GList *n = _point_node_at(f, pt, owner, depth + 1);
      if(n) return n;
    }
  }
  return NULL;
}

dt_masks_point_group_t *_group_point(dt_masks_form_t *grp, const dt_mask_id_t id)
{
  GList *node = _point_node_owner(grp, id, NULL);
  return node ? node->data : NULL;
}

static void _mask_points_into(dt_masks_form_t *grp, GList **out, const int depth)
{
  if(!grp || depth > DT_MASKS_NESTING_MAX) return;
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = l->data;
    *out = g_list_prepend(*out, pt);
    if(dt_masks_point_is_marker(pt)) continue;
    dt_masks_form_t *f = dt_masks_get_from_id(darktable.develop, pt->formid);
    if(f && f != grp && (f->type & DT_MASKS_GROUP)) _mask_points_into(f, out, depth + 1);
  }
}

// every point of the mask `grp` at any depth, bottom-up, a nested group's
// right after the member that holds it. Free the list, not the points
static GList *_mask_points(dt_masks_form_t *grp)
{
  GList *out = NULL;
  _mask_points_into(grp, &out, 0);
  return g_list_reverse(out);
}

// how many times this module's own mask references form `fid`, at any depth.
// One mask can hold the same shape twice (a group linking a shape another
// group defines), which _model_form_users cannot report: it counts a module
// once however many of its members point at the form
static int _model_form_uses_in_mask(dt_iop_module_t *module, const dt_mask_id_t fid)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp || !dt_is_valid_maskid(fid)) return 0;
  int n = 0;
  GList *pts = _mask_points(grp);
  for(GList *l = pts; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    if(!dt_masks_point_is_marker(pt) && pt->formid == fid) n++;
  }
  g_list_free(pts);
  return n;
}

// a member point and the index of its form in the flattened copy of the mask
// the canvas edits (dt_masks_group_ungroup)
typedef struct _canvas_point_t
{
  dt_masks_point_group_t *pt;
  int pos;
} _canvas_point_t;

static void _canvas_points_into(dt_masks_form_t *grp, GArray *out, int *pos, const int depth)
{
  if(!grp || depth > DT_MASKS_NESTING_MAX) return;
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = l->data;
    // a marker, or a member whose form is gone, has no form in the copy
    dt_masks_form_t *f = dt_masks_get_from_id(darktable.develop, pt->formid);
    if(!f) continue;
    const _canvas_point_t cp = { pt, *pos };
    g_array_append_val(out, cp);
    // a nested group or an AI object is replaced by its forms in the copy,
    // taking no index of its own
    if(f->type & (DT_MASKS_GROUP | DT_MASKS_OBJECT))
    {
      if(f != grp) _canvas_points_into(f, out, pos, depth + 1);
    }
    else
      (*pos)++;
  }
}

// every member point of the mask `grp` at any depth, in the canvas copy's
// order, with its form's index there. Free with g_array_free(a, TRUE)
static GArray *_canvas_points(dt_masks_form_t *grp)
{
  GArray *out = g_array_new(FALSE, FALSE, sizeof(_canvas_point_t));
  int pos = 0;
  _canvas_points_into(grp, out, &pos, 0);
  return out;
}

// the gain the groups around member `fid` apply to it: the opacity of its
// group, and at each enclosing level the nested group's own member opacity and
// the opacity of the group holding it (see _group_get_mask_roi_flexi)
static float _enclosing_gain(dt_masks_form_t *grp, const dt_mask_id_t fid)
{
  float gain = 1.0f;
  dt_mask_id_t id = fid;
  for(int depth = 0; depth <= DT_MASKS_NESTING_MAX; depth++)
  {
    dt_masks_form_t *owner = NULL;
    GList *node = _point_node_owner(grp, id, &owner);
    if(!node) break;
    if(id != fid) gain *= ((dt_masks_point_group_t *)node->data)->opacity;
    GList *marker = node;
    while(marker && !dt_masks_point_is_marker(marker->data)) marker = marker->prev;
    if(marker) gain *= ((dt_masks_point_group_t *)marker->data)->group_opacity;
    if(owner == grp) break;
    id = owner->formid;
  }
  return gain;
}

// is `id` one of the member ids `formids`, or a point of a nested group among
// them, at any depth?
static gboolean _members_hold(GList *formids, const dt_mask_id_t id)
{
  if(!dt_is_valid_maskid(id)) return FALSE;
  for(GList *l = formids; l; l = g_list_next(l))
  {
    const dt_mask_id_t fid = GPOINTER_TO_INT(l->data);
    if(fid == id) return TRUE;
    dt_masks_form_t *f = dt_masks_get_from_id(darktable.develop, fid);
    if(f && (f->type & DT_MASKS_GROUP) && _point_node_owner(f, id, NULL)) return TRUE;
  }
  return FALSE;
}

// a group's user-given name, or NULL if it has none: held by its marker
static const char *_group_custom_name(dt_masks_form_t *grp, const dt_mask_id_t cid)
{
  const dt_masks_point_group_t *pt = _group_point(grp, cid);
  return (pt && pt->name[0]) ? pt->name : NULL;
}

// ===========================================================================
// Phase 2: scoped mask refinement.
//
// The "mask refinement" sliders (details / feathering guide+radius / blur /
// brightness / contrast) can target one of three scopes chosen by the combo in
// the section header:
//   - GLOBAL    : blend_params->{details,...}, applied once on the final group
//                 mask. The legacy behaviour, and the default. Also the forced
//                 scope in classic/raster modes (the combo is flexi-only).
//   - ALL_SHAPES: one refinement set broadcast into every *drawn* (non-parametric)
//                 form's dt_masks_point_group_t.refinement.
//   - PARAMETRIC: a single parametric form's point refinement.
// The latter two reuse the existing per-shape refinement storage (masks v7) and
// its renderer hook (group.c, dt_develop_blend_refine_form_mask), so no schema,
// version or render change is needed -- a zero-filled (disabled) refinement is
// byte-identical to before.
// A group's own refinement (REFINE_SCOPE_GROUP) is held by its marker.
// The REFINE_SCOPE_* values live in blend_gui_internal.h.

// a group's displayed number (see its definition)
static int _group_ordinal_any(dt_iop_module_t *module, const dt_mask_id_t cid);

// enum values in the same order as dt_develop_feathering_guide_names[], so a
// combo index maps to the stored uint32 guide value (and back).
static const uint32_t _refine_guide_values[] = { DEVELOP_MASK_GUIDE_OUT_BEFORE_BLUR,
                                                 DEVELOP_MASK_GUIDE_IN_BEFORE_BLUR,
                                                 DEVELOP_MASK_GUIDE_OUT_AFTER_BLUR,
                                                 DEVELOP_MASK_GUIDE_IN_AFTER_BLUR };

// first drawn (non-parametric) form's point in the group, or NULL. Used to read
// back the "all shapes" refinement (every drawn point is kept in sync, so the
// first one is representative).
// A nested group is not a drawn form of its own: its shapes are, at any depth.
static dt_masks_point_group_t *_refine_first_drawn_point(dt_masks_form_t *grp)
{
  dt_masks_point_group_t *first = NULL;
  GList *pts = _mask_points(grp);
  for(GList *l = pts; l && !first; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = l->data;
    dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, pt->formid);
    if(form && !(form->type & (DT_MASKS_PARAMETRIC | DT_MASKS_GROUP))) first = pt;
  }
  g_list_free(pts);
  return first;
}

// a group point's effective operator for grouping: a missing operator (the base)
// reads as union, matching how the list folds runs. Shared with the renderer,
// which has to partition the list into exactly the same runs -- see
// dt_masks_eff_group_op() in masks.h for what goes wrong when it does not.
dt_masks_state_t _eff_group_op(const int state)
{
  return dt_masks_eff_group_op(state);
}

// A flexi group is its marker (see DT_MASKS_STATE_GROUP_MARKER) followed by
// its members, up to the next marker; grp->points is bottom-up. TRUE if list
// node `l` is a marker
gboolean _starts_group(GList *l)
{
  return l && dt_masks_point_is_marker(l->data);
}

// the list node of point `id` -- a member or a marker -- or NULL
static GList *_point_node(dt_masks_form_t *grp, const dt_mask_id_t id)
{
  return _point_node_owner(grp, id, NULL);
}

// the marker node of the group node `l` is in: `l` itself for a marker
static GList *_group_marker_node(GList *l)
{
  while(l && !_starts_group(l)) l = l->prev;
  return l;
}

// the last node of the group whose marker node is `marker`: its top member, or
// the marker itself when it has none. What goes on top of the group goes
// right after it
static GList *_group_last_node(GList *marker)
{
  GList *last = marker;
  for(GList *l = marker ? marker->next : NULL; l && !_starts_group(l); l = g_list_next(l))
    last = l;
  return last;
}

// put the points of `pts` (bottom-up, none of them in the list) right after
// node `at`, or at the bottom of the list for NULL, in their order
static void _insert_points_after(dt_masks_form_t *grp, GList *at, GList *pts)
{
  for(GList *l = pts; l; l = g_list_next(l))
  {
    grp->points = g_list_insert_before(grp->points, at ? at->next : grp->points, l->data);
    at = at ? at->next : grp->points;
  }
}

// how many groups the list of `grp` itself holds, nested ones not counted
static int _group_partition_count(const dt_masks_form_t *grp)
{
  int n = 0;
  for(GList *l = grp ? grp->points : NULL; l; l = g_list_next(l))
    if(_starts_group(l)) n++;
  return n;
}

GList *_group_partition_heads(dt_masks_form_t *grp)
{
  GList *out = NULL;
  for(GList *l = grp ? grp->points : NULL; l; l = g_list_next(l))
    if(_starts_group(l))
      out =
        g_list_prepend(out, GINT_TO_POINTER(((dt_masks_point_group_t *)l->data)->formid));
  return g_list_reverse(out);
}

GList *_selected_group_formids(dt_masks_form_t *grp, const dt_mask_id_t id)
{
  GList *marker = _group_marker_node(_point_node(grp, id));
  GList *out = NULL;
  for(GList *l = marker ? marker->next : NULL; l && !_starts_group(l); l = g_list_next(l))
    out =
      g_list_prepend(out, GINT_TO_POINTER(((dt_masks_point_group_t *)l->data)->formid));
  return out;
}

dt_mask_id_t _group_cid_of_form(dt_masks_form_t *grp, const dt_mask_id_t fid)
{
  GList *marker = _group_marker_node(_point_node(grp, fid));
  return marker ? ((dt_masks_point_group_t *)marker->data)->formid : INVALID_MASKID;
}

// is the group member `fid` is in bypassed, or a group holding that one at any
// depth? Its members then render nothing
static gboolean _member_group_bypassed(dt_masks_form_t *grp, const dt_mask_id_t fid)
{
  dt_mask_id_t id = fid;
  for(int depth = 0; depth <= DT_MASKS_NESTING_MAX; depth++)
  {
    dt_masks_form_t *owner = NULL;
    GList *marker = _group_marker_node(_point_node_owner(grp, id, &owner));
    if(!marker) return FALSE;
    if(((dt_masks_point_group_t *)marker->data)->state & DT_MASKS_STATE_OP_BYPASS)
      return TRUE;
    if(owner == grp) return FALSE;
    id = owner->formid;
  }
  return FALSE;
}

static dt_masks_form_t *_new_nested_group(void);
static void _add_nested_ref(dt_masks_form_t *owner, GList *at, dt_masks_form_t *sub);

dt_mask_id_t _model_add_group(dt_masks_form_t *grp,
                              const dt_masks_state_t op,
                              const dt_mask_id_t cid,
                              const gboolean below)
{
  if(!grp) return INVALID_MASKID;
  // a nested group is one group, so its sibling is a new nested group beside
  // it in its holder's list
  dt_masks_form_t *nested = _model_nested_group_of(grp, cid);
  if(nested)
  {
    dt_masks_form_t *holder = NULL;
    GList *ref = _point_node_owner(grp, nested->formid, &holder);
    dt_masks_form_t *sub = ref ? _new_nested_group() : NULL;
    if(!sub) return INVALID_MASKID;
    dt_masks_point_group_t *mk =
      dt_masks_marker_new(darktable.develop->forms, sub, op & DT_MASKS_STATE_OP_COMBINE);
    sub->points = g_list_append(NULL, mk);
    _add_nested_ref(holder, below ? ref->prev : ref, sub);
    return mk->formid;
  }
  // next to group `cid`, in the list that holds it
  dt_masks_form_t *owner = grp;
  GList *next_to = _group_marker_node(_point_node_owner(grp, cid, &owner));
  // a new group is live whatever it was added next to
  dt_masks_point_group_t *m =
    dt_masks_marker_new(darktable.develop ? darktable.develop->forms : NULL, owner,
                        op & DT_MASKS_STATE_OP_COMBINE);
  if(!m) return INVALID_MASKID;
  GList *at = next_to ? (below ? next_to->prev : _group_last_node(next_to))
                      : (below ? NULL : g_list_last(owner->points));
  GList *one = g_list_prepend(NULL, m);
  _insert_points_after(owner, at, one);
  g_list_free(one);
  return m->formid;
}

// how many levels of nested groups the member form `f` brings: 0 for a shape
static int _form_nesting(const dt_masks_form_t *f, const int depth)
{
  if(!f || !(f->type & DT_MASKS_GROUP) || depth > DT_MASKS_NESTING_MAX) return 0;
  int deepest = 0;
  for(const GList *l = f->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    if(dt_masks_point_is_marker(pt)) continue;
    const dt_masks_form_t *c = dt_masks_get_from_id(darktable.develop, pt->formid);
    if(c != f) deepest = MAX(deepest, _form_nesting(c, depth + 1));
  }
  return 1 + deepest;
}

// how many nested groups hold the list of `owner`: 0 for the mask's own
static int _list_depth(dt_masks_form_t *grp, dt_masks_form_t *owner)
{
  int depth = 0;
  while(owner && owner != grp && depth <= DT_MASKS_NESTING_MAX)
  {
    dt_masks_form_t *up = NULL;
    if(!_point_node_owner(grp, owner->formid, &up)) break;
    owner = up;
    depth++;
  }
  return depth;
}

// may the member `fid` move from the list of `from` into the list of `to`?
// A nested group never into itself or below itself, and nothing deeper than a
// walk of the mask follows (DT_MASKS_NESTING_MAX), unless it is no deeper than
// it already was
static gboolean _may_move_into(dt_masks_form_t *grp,
                               dt_masks_form_t *from,
                               dt_masks_form_t *to,
                               const dt_mask_id_t fid)
{
  if(from == to) return TRUE;
  dt_masks_form_t *f = dt_masks_get_from_id(darktable.develop, fid);
  if(f && (f->type & DT_MASKS_GROUP) && (f == to || _point_node_owner(f, to->formid, NULL)))
    return FALSE;
  const int n = _form_nesting(f, 0);
  const int depth = _list_depth(grp, to) + n;
  return depth <= DT_MASKS_NESTING_MAX || depth <= _list_depth(grp, from) + n;
}

// how many groups the list of `owner` holds
static int _list_group_count(const dt_masks_form_t *owner)
{
  int n = 0;
  for(GList *l = owner ? owner->points : NULL; l; l = g_list_next(l))
    if(_starts_group(l)) n++;
  return n;
}

// a new nested group form with no points, named and added to the image's
// forms. Named here: a group form has no set_form_name, which
// dt_masks_assign_unique_name needs
static dt_masks_form_t *_new_nested_group(void)
{
  dt_develop_t *dev = darktable.develop;
  dt_masks_form_t *sub = dt_masks_create(DT_MASKS_GROUP);
  if(!sub || !dev) return sub;
  int nb = 0;
  for(GList *l = dev->forms; l; l = g_list_next(l))
    if(((dt_masks_form_t *)l->data)->type == DT_MASKS_GROUP) nb++;
  gboolean taken;
  do
  {
    taken = FALSE;
    snprintf(sub->name, sizeof(sub->name), _("group #%d"), ++nb);
    for(GList *l = dev->forms; l && !taken; l = g_list_next(l))
      taken = !strcmp(((dt_masks_form_t *)l->data)->name, sub->name);
  } while(taken);
  dev->forms = g_list_append(dev->forms, sub);
  return sub;
}

// a plain reference to `sub` in the list of `owner`, right after node `at`
static void _add_nested_ref(dt_masks_form_t *owner, GList *at, dt_masks_form_t *sub)
{
  dt_masks_point_group_t *pt = calloc(1, sizeof(dt_masks_point_group_t));
  pt->formid = sub->formid;
  pt->parentid = owner->formid;
  pt->state = DT_MASKS_STATE_SHOW | DT_MASKS_STATE_USE | DT_MASKS_STATE_UNION;
  pt->opacity = 1.0f;
  pt->group_opacity = 1.0f;
  GList *one = g_list_prepend(NULL, pt);
  _insert_points_after(owner, at, one);
  g_list_free(one);
}

// make `sub` a member of the group whose marker node is `marker` in the list
// of `owner`, on top of its members
static void _add_nested_member(dt_masks_form_t *owner, GList *marker, dt_masks_form_t *sub)
{
  _add_nested_ref(owner, _group_last_node(marker), sub);
}

dt_mask_id_t _model_nest_new_group(dt_masks_form_t *grp,
                                   const dt_masks_state_t within,
                                   const dt_mask_id_t cid)
{
  dt_masks_form_t *owner = NULL;
  GList *marker = _group_marker_node(_point_node_owner(grp, cid, &owner));
  if(!marker || _list_depth(grp, owner) + 1 > DT_MASKS_NESTING_MAX) return INVALID_MASKID;
  dt_masks_form_t *sub = _new_nested_group();
  if(!sub) return INVALID_MASKID;
  dt_masks_point_group_t *m = dt_masks_marker_new(darktable.develop->forms, sub,
                                                  within & DT_MASKS_STATE_WITHIN);
  sub->points = g_list_append(NULL, m);
  _add_nested_member(owner, marker, sub);
  return m->formid;
}

// move the group whose marker node is `src`, in the list of `sowner`, into a
// new nested group as its one group, the reference to which goes into the
// list of `to` right after the point `after`. FALSE where it may not go: the
// list it leaves keeps a group, and its members land one level below the list
// of `to`, never beside or inside themselves and never deeper than
// DT_MASKS_NESTING_MAX
static gboolean _wrap_group(dt_masks_form_t *grp,
                            dt_masks_form_t *sowner,
                            GList *src,
                            dt_masks_form_t *to,
                            const dt_masks_point_group_t *after)
{
  if(_list_group_count(sowner) <= 1 || !after || after == src->data) return FALSE;
  const int depth = _list_depth(grp, to) + 1;
  if(depth > DT_MASKS_NESTING_MAX) return FALSE;
  for(GList *l = src->next; l && !_starts_group(l); l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    if(pt == after) return FALSE;
    dt_masks_form_t *f = dt_masks_get_from_id(darktable.develop, pt->formid);
    if(f && (f->type & DT_MASKS_GROUP)
       && (f == to || _point_node_owner(f, to->formid, NULL)))
      return FALSE;
    if(depth + _form_nesting(f, 0) > DT_MASKS_NESTING_MAX) return FALSE;
  }

  dt_masks_form_t *sub = _new_nested_group();
  if(!sub) return FALSE;
  // the group, its marker and members, becomes the nested group's one group
  GList *slice = NULL;
  for(GList *l = src; l && (l == src || !_starts_group(l)); l = g_list_next(l))
    slice = g_list_append(slice, l->data);
  for(GList *l = slice; l; l = g_list_next(l))
  {
    sowner->points = g_list_remove(sowner->points, l->data);
    ((dt_masks_point_group_t *)l->data)->parentid = sub->formid;
  }
  sub->points = slice;
  _add_nested_ref(to, g_list_find(to->points, after), sub);
  return TRUE;
}

gboolean _model_nest_group(dt_masks_form_t *grp,
                           const dt_mask_id_t src_cid,
                           const dt_mask_id_t dst_cid)
{
  if(!grp || src_cid == dst_cid) return FALSE;
  dt_masks_form_t *sowner = NULL, *downer = NULL;
  GList *src = _point_node_owner(grp, src_cid, &sowner);
  GList *dst = _group_marker_node(_point_node_owner(grp, dst_cid, &downer));
  if(!src || !_starts_group(src) || !dst || dst == src) return FALSE;
  return _wrap_group(grp, sowner, src, downer, _group_last_node(dst)->data);
}

// a new nested group folding with `within`, holding nothing but its marker
static dt_masks_form_t *_new_empty_nested_group(const dt_masks_state_t within)
{
  dt_masks_form_t *sub = _new_nested_group();
  if(!sub) return NULL;
  sub->points = g_list_append(NULL, dt_masks_marker_new(darktable.develop->forms, sub,
                                                        within & DT_MASKS_STATE_WITHIN));
  return sub;
}

// the marker of an empty group on top of the list of `owner`: what "compose"
// gives the user to fill
static dt_mask_id_t _add_empty_on_top(dt_masks_form_t *owner)
{
  dt_masks_form_t *e = _new_empty_nested_group(0);
  if(!e) return INVALID_MASKID;
  _add_nested_ref(owner, g_list_last(owner->points), e);
  return ((dt_masks_point_group_t *)e->points->data)->formid;
}

dt_mask_id_t _model_compose(dt_masks_form_t *grp,
                            const dt_masks_point_group_t *pt,
                            const dt_masks_state_t within)
{
  dt_masks_form_t *owner = NULL;
  GList *node = _point_node_at(grp, pt, &owner, 0);
  // the paths of an AI object move as the object, and a list of several
  // groups is only in an edit stored before one-group masks
  if(!node || !(owner->type & DT_MASKS_GROUP) || (owner->type & DT_MASKS_OBJECT)
     || (_starts_group(node) && _list_group_count(owner) != 1))
    return INVALID_MASKID;

  if(owner == grp && _starts_group(node))
  {
    // the mask's own group stays the mask: its members and settings move into
    // a new group at its bottom, and it starts over with `within` and none
    if(_form_nesting(grp, 0) > DT_MASKS_NESTING_MAX) return INVALID_MASKID;
    dt_masks_point_group_t *root = node->data;
    dt_masks_form_t *sub = _new_nested_group();
    if(!sub) return INVALID_MASKID;
    dt_masks_point_group_t *mk = malloc(sizeof(dt_masks_point_group_t));
    memcpy(mk, root, sizeof(dt_masks_point_group_t));
    mk->formid = dt_masks_new_marker_id(darktable.develop->forms);
    mk->parentid = sub->formid;
    // a bypassed mask offers no compose, and the bypass is the mask's anyway
    mk->state &= ~DT_MASKS_STATE_OP_DISABLE;
    GList *members = node->next;
    node->next = NULL;
    if(members) members->prev = NULL;
    for(GList *l = members; l; l = g_list_next(l))
      ((dt_masks_point_group_t *)l->data)->parentid = sub->formid;
    sub->points = g_list_prepend(members, mk);
    root->state = (root->state & ~(DT_MASKS_STATE_WITHIN | DT_MASKS_STATE_OP_INVERT))
                  | (within & DT_MASKS_STATE_WITHIN);
    root->group_opacity = 1.0f;
    memset(&root->refinement, 0, sizeof(root->refinement));
    _add_nested_ref(grp, node, sub);
    return _add_empty_on_top(grp);
  }

  // a group is composed through the reference its holder has to it
  if(_starts_group(node))
  {
    const dt_mask_id_t gid = owner->formid;
    node = _point_node_owner(grp, gid, &owner);
    if(!node) return INVALID_MASKID;
  }
  const dt_masks_point_group_t *ref = node->data;
  const int depth = _list_depth(grp, owner) + 1;
  const int below = _form_nesting(dt_masks_get_from_id(darktable.develop, ref->formid), 0);
  if(depth + MAX(1, below) > DT_MASKS_NESTING_MAX) return INVALID_MASKID;

  // the reference moves as it is, keeping its own settings, into a new group
  // that takes its place and applies nothing
  dt_masks_form_t *wrap = _new_empty_nested_group(within);
  if(!wrap) return INVALID_MASKID;
  _add_nested_ref(owner, node, wrap);
  owner->points = g_list_remove_link(owner->points, node);
  ((dt_masks_point_group_t *)node->data)->parentid = wrap->formid;
  wrap->points = g_list_concat(wrap->points, node);
  return _add_empty_on_top(wrap);
}

// the one member of the mask's own group, when that is a plain nested group
// nothing else holds, or NULL
static dt_masks_form_t *_sole_held_group(dt_masks_form_t *grp, dt_masks_point_group_t **ref)
{
  if(!grp || !grp->points || !dt_masks_point_is_marker(grp->points->data)
     || !grp->points->next || grp->points->next->next)
    return NULL;
  *ref = grp->points->next->data;
  dt_masks_form_t *sub = dt_masks_get_from_id(darktable.develop, (*ref)->formid);
  if(!sub || sub == grp || !(sub->type & DT_MASKS_GROUP)
     || (sub->type & (DT_MASKS_CLONE | DT_MASKS_OBJECT)) || !sub->points
     || !dt_masks_point_is_marker(sub->points->data))
    return NULL;
  // another module's mask keeps its members whoever holds it
  for(GList *m = darktable.develop->iop; m; m = g_list_next(m))
  {
    const dt_iop_module_t *mod = m->data;
    if(mod->blend_params && mod->blend_params->mask_id == sub->formid) return NULL;
  }
  int refs = 0;
  for(GList *f = darktable.develop->forms; f; f = g_list_next(f))
  {
    const dt_masks_form_t *g = f->data;
    if(!(g->type & DT_MASKS_GROUP)) continue;
    for(GList *l = g->points; l; l = g_list_next(l))
      if(((dt_masks_point_group_t *)l->data)->formid == sub->formid) refs++;
  }
  return refs == 1 ? sub : NULL;
}

gboolean _model_hoist_sole_group(dt_masks_form_t *grp, dt_masks_refinement_t *whole)
{
  dt_masks_point_group_t *ref = NULL;
  dt_masks_form_t *sub = _sole_held_group(grp, &ref);
  if(!sub || _list_group_count(sub) != 1) return FALSE;
  dt_masks_point_group_t *root = grp->points->data;
  dt_masks_point_group_t *mk = sub->points->data;
  // the mask's own group applies nothing, and the reference nothing
  if((root->state & (DT_MASKS_STATE_OP_DISABLE | DT_MASKS_STATE_OP_INVERT))
     || root->group_opacity != 1.0f || root->refinement.enabled != DT_MASKS_REFINE_OFF
     || ref->opacity != 1.0f || ref->refinement.enabled != DT_MASKS_REFINE_OFF
     || (ref->state & (DT_MASKS_STATE_INVERSE | DT_MASKS_STATE_HIDDEN | DT_MASKS_STATE_DISABLE)))
    return FALSE;
  // the mask's own group takes no name, and the group's bypass has nowhere to go
  if(mk->name[0] || (mk->state & DT_MASKS_STATE_OP_DISABLE)) return FALSE;
  // the group's refinement becomes the whole mask's, which runs after the
  // mask's invert and opacity where the group's ran before its own
  const gboolean refined = mk->refinement.enabled != DT_MASKS_REFINE_OFF;
  if(refined
     && (whole->enabled || (mk->state & DT_MASKS_STATE_OP_INVERT) || mk->group_opacity != 1.0f))
    return FALSE;

  root->state = (root->state & ~(DT_MASKS_STATE_WITHIN | DT_MASKS_STATE_OP_INVERT))
                | (mk->state & (DT_MASKS_STATE_WITHIN | DT_MASKS_STATE_OP_INVERT));
  root->group_opacity = mk->group_opacity;
  if(refined) *whole = mk->refinement;
  free(ref);
  grp->points = g_list_delete_link(grp->points, grp->points->next);
  GList *members = sub->points->next;
  sub->points->next = NULL;
  if(members) members->prev = NULL;
  for(GList *l = members; l; l = g_list_next(l))
    ((dt_masks_point_group_t *)l->data)->parentid = grp->formid;
  grp->points = g_list_concat(grp->points, members);
  return TRUE;
}

// take the members of the group whose marker node is `marker` out of the
// list, and the marker too with `with_marker`. Returns the members' ids,
// bottom-up
static GList *_take_group_points(dt_masks_form_t *grp, GList *marker, const gboolean with_marker)
{
  GList *ids = NULL;
  GList *l = marker->next;
  while(l && !_starts_group(l))
  {
    GList *next = g_list_next(l);
    ids = g_list_prepend(ids, GINT_TO_POINTER(((dt_masks_point_group_t *)l->data)->formid));
    free(l->data);
    grp->points = g_list_delete_link(grp->points, l);
    l = next;
  }
  if(with_marker)
  {
    free(marker->data);
    grp->points = g_list_delete_link(grp->points, marker);
  }
  return g_list_reverse(ids);
}

GList *_model_delete_group(dt_masks_form_t *grp, const dt_mask_id_t cid)
{
  dt_masks_form_t *owner = NULL;
  GList *marker = _group_marker_node(_point_node_owner(grp, cid, &owner));
  return marker ? _take_group_points(owner, marker, TRUE) : NULL;
}

GList *_model_empty_group(dt_masks_form_t *grp, const dt_mask_id_t cid)
{
  dt_masks_form_t *owner = NULL;
  GList *marker = _group_marker_node(_point_node_owner(grp, cid, &owner));
  return marker ? _take_group_points(owner, marker, FALSE) : NULL;
}

gboolean _model_merge_group_down(dt_masks_form_t *grp, const dt_mask_id_t cid)
{
  dt_masks_form_t *owner = NULL;
  GList *marker = _group_marker_node(_point_node_owner(grp, cid, &owner));
  if(!marker || !marker->prev) return FALSE;
  free(marker->data);
  owner->points = g_list_delete_link(owner->points, marker);
  return TRUE;
}

// read the six refinement controls into r. enabled is derived from whether any
// effective parameter is non-neutral, so committing an all-neutral refinement
// leaves enabled == 0 and the renderer keeps its byte-identical fast path.
static void _refine_read_controls(dt_iop_gui_blend_data_t *bd, dt_masks_refinement_t *r)
{
  r->details = dt_bauhaus_slider_get(bd->details_slider);
  const int gi = dt_bauhaus_combobox_get(bd->masks_feathering_guide_combo);
  r->feathering_guide =
    (gi >= 0 && gi < 4) ? _refine_guide_values[gi] : DEVELOP_MASK_GUIDE_OUT_BEFORE_BLUR;
  r->feathering_radius = dt_bauhaus_slider_get(bd->feathering_radius_slider);
  r->blur_radius = dt_bauhaus_slider_get(bd->blur_radius_slider);
  r->contrast = dt_bauhaus_slider_get(bd->contrast_slider);
  r->brightness = dt_bauhaus_slider_get(bd->brightness_slider);
  r->enabled = (r->details != 0.0f || r->feathering_radius != 0.0f
                || r->blur_radius != 0.0f || r->contrast != 0.0f || r->brightness != 0.0f)
                 ? 1
                 : 0;
}

// push a refinement struct into the six controls without triggering commits.
static void _refine_set_controls(dt_iop_gui_blend_data_t *bd,
                                 const dt_masks_refinement_t *r)
{
  bd->masks_refine_updating = TRUE;
  dt_bauhaus_slider_set(bd->details_slider, r->details);
  int gi = 0;
  for(int i = 0; i < 4; i++)
    if(_refine_guide_values[i] == r->feathering_guide)
    {
      gi = i;
      break;
    }
  dt_bauhaus_combobox_set(bd->masks_feathering_guide_combo, gi);
  dt_bauhaus_slider_set(bd->feathering_radius_slider, r->feathering_radius);
  dt_bauhaus_slider_set(bd->blur_radius_slider, r->blur_radius);
  dt_bauhaus_slider_set(bd->brightness_slider, r->brightness);
  dt_bauhaus_slider_set(bd->contrast_slider, r->contrast);
  bd->masks_refine_updating = FALSE;
}

// load the six controls from whatever scope is currently active.
static void _refine_populate(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_develop_blend_params_t *bp = module->blend_params;
  dt_masks_refinement_t r = { 0 };

  if(bd->masks_refine_scope_kind == REFINE_SCOPE_GLOBAL)
  {
    r.details = bp->details;
    r.feathering_guide = bp->feathering_guide;
    r.feathering_radius = bp->feathering_radius;
    r.blur_radius = bp->blur_radius;
    r.brightness = bp->brightness;
    r.contrast = bp->contrast;
  }
  else
  {
    dt_masks_form_t *grp = _module_mask_group(module);
    // ALL_SHAPES reads the first drawn point; GROUP reads the group's marker;
    // ELEMENT reads that one specific form directly
    const dt_masks_point_group_t *src =
      (bd->masks_refine_scope_kind == REFINE_SCOPE_ALL_SHAPES)
        ? _refine_first_drawn_point(grp)
        : _group_point(grp, bd->masks_refine_scope_formid);
    // ...but only when the stored value belongs to the scope being shown. One
    // member's point holds either its own element refinement or a broadcast copy
    // of its group's (see dt_masks_refine_scope_t); showing one in the other's
    // controls would report a refinement this scope does not have, and the next
    // slider move would rewrite it into the wrong scope.
    if(src)
    {
      const gboolean want_group = bd->masks_refine_scope_kind == REFINE_SCOPE_GROUP;
      const gboolean is_group = src->refinement.enabled == DT_MASKS_REFINE_GROUP;
      if(src->refinement.enabled == DT_MASKS_REFINE_OFF || want_group == is_group)
        r = src->refinement;
    }
  }
  _refine_set_controls(bd, &r);
}

static void _refine_update_header(dt_iop_module_t *module);

static inline gpointer _refine_scope_key(dt_iop_gui_blend_data_t *bd)
{
  if(!bd) return GUINT_TO_POINTER(0);
  if(bd->masks_refine_scope_kind == REFINE_SCOPE_ELEMENT)
    return GUINT_TO_POINTER(dt_masks_refine_key_element(bd->masks_refine_scope_formid));
  else if(bd->masks_refine_scope_kind == REFINE_SCOPE_GROUP)
    return GUINT_TO_POINTER(dt_masks_refine_key_group(bd->masks_refine_scope_formid));
  else
    return GUINT_TO_POINTER(DT_MASKS_REFINE_KEY_GLOBAL);
}

static void _refine_update_expanded_state(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module ? module->blend_data : NULL;
  if(!bd || !bd->masks_refine_toggle_btn) return;

  if(!bd->masks_refine_expanded)
    bd->masks_refine_expanded = g_hash_table_new(g_direct_hash, g_direct_equal);

  gpointer key = _refine_scope_key(bd);
  gpointer val = NULL;
  gboolean expanded =
    !dt_conf_get_bool("plugins/darkroom/masks/collapse_refinements_default");
  if(g_hash_table_lookup_extended(bd->masks_refine_expanded, key, NULL, &val))
    expanded = GPOINTER_TO_INT(val);

  bd->masks_refine_updating = TRUE;
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_refine_toggle_btn), expanded);
  dtgtk_togglebutton_set_paint(
    DTGTK_TOGGLEBUTTON(bd->masks_refine_toggle_btn), dtgtk_cairo_paint_solid_arrow,
    (expanded ? CPF_DIRECTION_DOWN : CPF_DIRECTION_LEFT), NULL);
  if(bd->masks_refine_expander
     && dtgtk_expander_get_expanded(DTGTK_EXPANDER(bd->masks_refine_expander))
          != expanded)
    dtgtk_expander_set_expanded(DTGTK_EXPANDER(bd->masks_refine_expander), expanded);
  if(bd->masks_refine_sliders_box)
    gtk_widget_set_visible(GTK_WIDGET(bd->masks_refine_sliders_box), expanded);
  bd->masks_refine_updating = FALSE;
}

static void _refine_toggle_toggled(GtkToggleButton *btn, gpointer user_data)
{
  dt_iop_module_t *module = (dt_iop_module_t *)user_data;
  dt_iop_gui_blend_data_t *bd = module ? module->blend_data : NULL;
  if(!bd || bd->masks_refine_updating) return;

  const gboolean active = gtk_toggle_button_get_active(btn);
  if(!bd->masks_refine_expanded)
    bd->masks_refine_expanded = g_hash_table_new(g_direct_hash, g_direct_equal);

  gpointer key = _refine_scope_key(bd);
  g_hash_table_insert(bd->masks_refine_expanded, key, GINT_TO_POINTER(active));

  dtgtk_togglebutton_set_paint(DTGTK_TOGGLEBUTTON(bd->masks_refine_toggle_btn),
                               dtgtk_cairo_paint_solid_arrow,
                               (active ? CPF_DIRECTION_DOWN : CPF_DIRECTION_LEFT), NULL);
  if(bd->masks_refine_expander)
    dtgtk_expander_set_expanded(DTGTK_EXPANDER(bd->masks_refine_expander), active);
  if(bd->masks_refine_sliders_box)
  {
    gtk_widget_set_visible(GTK_WIDGET(bd->masks_refine_sliders_box), active);
    gtk_widget_queue_resize(GTK_WIDGET(bd->masks_refine_sliders_box));
  }
}

static void _refine_bypass_toggled(GtkToggleButton *btn, gpointer user_data)
{
  dt_iop_module_t *module = (dt_iop_module_t *)user_data;
  dt_iop_gui_blend_data_t *bd = module ? module->blend_data : NULL;
  if(!bd || bd->masks_refine_updating) return;

  const gboolean bypassed = gtk_toggle_button_get_active(btn);
  if(!bd->masks_refine_bypassed)
    bd->masks_refine_bypassed = g_hash_table_new(g_direct_hash, g_direct_equal);

  gpointer key = _refine_scope_key(bd);
  g_hash_table_insert(bd->masks_refine_bypassed, key, GINT_TO_POINTER(bypassed));

  _update_refine_sensitivity(module);

  if(module->dev)
  {
    dt_dev_reprocess_all(module->dev);
    dt_control_queue_redraw();
  }
}

static void _refine_header_clicked(
  GtkGestureSingle *gesture, gint n_press, gdouble x, gdouble y, gpointer user_data)
{
  if(gtk_gesture_single_get_current_button(gesture) != GDK_BUTTON_PRIMARY) return;
  dt_iop_gui_blend_data_t *bd = (dt_iop_gui_blend_data_t *)user_data;
  if(!bd || !bd->masks_refine_toggle_btn) return;

  const gboolean active =
    gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(bd->masks_refine_toggle_btn));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_refine_toggle_btn), !active);
}

// flexi: the refinement controls follow the list selection — a selected shape
// (or parametric/raster element) targets only that one element; a selected
// group header (no specific element within it) targets the whole group;
// nothing selected targets global. Defined here so _update_row_selection
// (above the scope helpers) can drive it.
void _model_refine_scope_from_selection(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  const gboolean flexi = !(module->blend_params->mask_mode & DEVELOP_MASK_RASTER);
  if(flexi && dt_is_valid_maskid(bd->panel_selected_formid))
  {
    bd->masks_refine_scope_kind = REFINE_SCOPE_ELEMENT;
    bd->masks_refine_scope_formid = bd->panel_selected_formid;
  }
  // the mask's own group refines the whole mask: the module-wide refinement
  // every migrated edit keeps, applied after the mask is rendered. Its marker's
  // own group refinement is not reachable from the panel
  else if(flexi && dt_is_valid_maskid(bd->panel_selected_group_cid)
          && bd->panel_selected_group_cid != _mask_group_cid(module))
  {
    bd->masks_refine_scope_kind = REFINE_SCOPE_GROUP;
    bd->masks_refine_scope_formid = bd->panel_selected_group_cid;
  }
  else
  {
    bd->masks_refine_scope_kind = REFINE_SCOPE_GLOBAL;
    bd->masks_refine_scope_formid = INVALID_MASKID;
  }
}

// the refinement scope outlives its target when that is removed by a route
// that does not reselect (canvas, an AI object losing its last path, undo),
// and the caption kept naming it
gboolean _model_refine_scope_prune(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  const gboolean scope_gone =
    (bd->masks_refine_scope_kind == REFINE_SCOPE_ELEMENT
     || bd->masks_refine_scope_kind == REFINE_SCOPE_GROUP)
    && !_group_point(grp, bd->masks_refine_scope_formid);
  if(!scope_gone) return FALSE;
  if(!_group_point(grp, bd->panel_selected_formid))
    bd->panel_selected_formid = INVALID_MASKID;
  if(!_group_point(grp, bd->panel_selected_group_cid))
    bd->panel_selected_group_cid = INVALID_MASKID;
  return TRUE;
}

// retarget from the selection, then reload the refinement controls for it
static void _flexi_refine_follow_selection(dt_iop_gui_blend_data_t *bd)
{
  if(!bd || !bd->blend_inited || !bd->module) return;
  _model_refine_scope_from_selection(bd->module);
  _refine_populate(bd->module);
  // clicking a row (the lightweight _update_row_selection path, not a full
  // list rebuild) changes the scope kind above but does not otherwise touch
  // the caption -- without this it keeps showing whichever scope was active
  // before the click (typically "group" or "whole mask"), making element
  // selection look like a no-op even though it did retarget the sliders.
  _refine_update_header(bd->module);
  _update_refine_sensitivity(bd->module);
  _refine_update_expanded_state(bd->module);
}

// commit a control change in GLOBAL scope. This reproduces, field by field, the
// exact behaviour the old set_field bindings / blendif callbacks had, so classic
// and flexi-global refinement render bit-for-bit identically to before.
static void _refine_commit_global(dt_iop_gui_blend_data_t *bd, GtkWidget *w)
{
  dt_develop_blend_params_t *bp = bd->module->blend_params;

  if(w == bd->details_slider)
  {
    const float oldval = bp->details;
    bp->details = dt_bauhaus_slider_get(w);
    dt_dev_add_history_item(darktable.develop, bd->module, TRUE);
    if((oldval == 0.0f) && (bp->details != 0.0f))
    {
      dt_dev_reprocess_all(bd->module->dev);
      dt_control_queue_redraw();
    }
    return;
  }

  if(w == bd->masks_feathering_guide_combo)
  {
    const int gi = dt_bauhaus_combobox_get(w);
    if(gi >= 0 && gi < 4) bp->feathering_guide = _refine_guide_values[gi];
  }
  else if(w == bd->feathering_radius_slider)
  {
    bp->feathering_radius = dt_bauhaus_slider_get(w);
    if(bp->feather_version == 0) bp->feather_version = 1;
  }
  else if(w == bd->blur_radius_slider)
  {
    bp->blur_radius = dt_bauhaus_slider_get(w);
    if(bp->feather_version == 0) bp->feather_version = 1;
  }
  else if(w == bd->brightness_slider)
    bp->brightness = dt_bauhaus_slider_get(w);
  else if(w == bd->contrast_slider)
    bp->contrast = dt_bauhaus_slider_get(w);

  dt_dev_add_history_item(darktable.develop, bd->module, TRUE);
}

// commit a control change in a non-global (per-form) scope: write the refinement
// into the targeted point(s) and persist via a masks history item.
static void _refine_commit_nonglobal(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp) return;

  dt_masks_refinement_t r = { 0 };
  _refine_read_controls(bd, &r);
  // stamp which mask this refinement is for (see dt_masks_refine_scope_t).
  // ALL_SHAPES is a per-shape broadcast, not a group one: each shape refines
  // its own mask.
  if(r.enabled)
    r.enabled = (bd->masks_refine_scope_kind == REFINE_SCOPE_GROUP)
                  ? DT_MASKS_REFINE_GROUP
                  : DT_MASKS_REFINE_ELEMENT;

  if(bd->masks_refine_scope_kind == REFINE_SCOPE_ALL_SHAPES)
  {
    // broadcast to every drawn (non-parametric) form, nested groups' shapes
    // included (see _refine_first_drawn_point)
    GList *pts = _mask_points(grp);
    for(GList *l = pts; l; l = g_list_next(l))
    {
      dt_masks_point_group_t *pt = l->data;
      dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, pt->formid);
      if(form && !(form->type & (DT_MASKS_PARAMETRIC | DT_MASKS_GROUP))) pt->refinement = r;
    }
    g_list_free(pts);
  }
  else // REFINE_SCOPE_GROUP writes the group's marker, ELEMENT the one element
  {
    dt_masks_point_group_t *pt = _group_point(grp, bd->masks_refine_scope_formid);
    if(pt) pt->refinement = r;
  }

  dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
}

// shared value-changed handler for all six refinement controls.
static void _refine_control_changed(GtkWidget *w, dt_iop_gui_blend_data_t *bd)
{
  if(DT_IN_GUI_UPDATE() || !bd || !bd->blend_inited || bd->masks_refine_updating) return;
  if(bd->masks_refine_scope_kind == REFINE_SCOPE_GLOBAL)
    _refine_commit_global(bd, w);
  else
    _refine_commit_nonglobal(bd->module);
  _refine_update_header(bd->module);
}

// reset the refinement of the currently selected scope back to neutral. For
// GLOBAL this clears blend_params (mirroring _refine_commit_global, but all
// fields at once); for the per-form scopes the controls are zeroed and the
// neutral (enabled==0) refinement is broadcast, restoring the byte-identical
// renderer fast path.
// clear the module-wide ("whole mask") refinement back to neutral, in the
// caller's blend_params. Returns TRUE if `details` was non-zero: crossing it to
// zero has to rebuild the scharr-derived detail mask, which an ordinary history
// item does not force, so the caller owes a dt_dev_reprocess_all.
// Shared by the refinement panel's own reset button and "reset mask" -- the two
// must clear exactly the same fields, or resetting the mask leaves a
// whole-mask refinement behind that nothing on screen still accounts for.
static gboolean _refine_clear_global(dt_iop_module_t *module)
{
  dt_develop_blend_params_t *bp = module->blend_params;
  const gboolean had_details = bp->details != 0.0f;
  bp->details = 0.0f;
  bp->feathering_guide = _refine_guide_values[0];
  bp->feathering_radius = 0.0f;
  bp->blur_radius = 0.0f;
  bp->brightness = 0.0f;
  bp->contrast = 0.0f;
  return had_details;
}

// is any module-wide refinement actually set? (so "reset mask" can skip
// committing a history item when there is nothing to clear)
static gboolean _refine_global_is_set(const dt_iop_module_t *module)
{
  const dt_develop_blend_params_t *bp = module->blend_params;
  return bp->details != 0.0f || bp->feathering_radius != 0.0f || bp->blur_radius != 0.0f
         || bp->brightness != 0.0f || bp->contrast != 0.0f;
}

static void _refine_reset_clicked(GtkWidget *btn, dt_iop_gui_blend_data_t *bd)
{
  if(DT_IN_GUI_UPDATE() || !bd || !bd->blend_inited || bd->masks_refine_updating) return;

  // neutral refinement: all magnitudes zero, guide back to its first value
  dt_masks_refinement_t r = { 0 };
  r.feathering_guide = _refine_guide_values[0];
  _refine_set_controls(bd, &r); // updates the six controls, guarded (no commit)

  if(bd->masks_refine_scope_kind == REFINE_SCOPE_GLOBAL)
  {
    const gboolean had_details = _refine_clear_global(bd->module);
    dt_dev_add_history_item(darktable.develop, bd->module, TRUE);
    // details crossing to zero needs the same full reprocess the slider path does
    if(had_details)
    {
      dt_dev_reprocess_all(bd->module->dev);
      dt_control_queue_redraw();
    }
  }
  else
    _refine_commit_nonglobal(bd->module);

  _refine_update_header(bd->module);
}

static gboolean _icon_widget_draw(GtkWidget *w, cairo_t *cr, gpointer user_data)
{
  DTGTKCairoPaintIconFunc paint = (DTGTKCairoPaintIconFunc)user_data;
  if(!paint) return FALSE;
  GtkAllocation a;
  gtk_widget_get_allocation(w, &a);
  GdkRGBA c;
  GtkStyleContext *ctx = gtk_widget_get_style_context(w);
  const GtkStateFlags state = gtk_widget_get_state_flags(w);
  gtk_style_context_get_color(ctx, state, &c);
  cairo_set_source_rgba(cr, c.red, c.green, c.blue, c.alpha * 0.85);
  paint(cr, 0, 0, a.width, a.height, 0, NULL);
  return FALSE;
}

static GtkWidget *_make_icon_widget(DTGTKCairoPaintIconFunc paint)
{
  GtkWidget *da = gtk_drawing_area_new();
  gtk_widget_set_size_request(da, DT_PIXEL_APPLY_DPI(16), DT_PIXEL_APPLY_DPI(16));
  gtk_widget_set_valign(da, GTK_ALIGN_CENTER);
  g_signal_connect(G_OBJECT(da), "draw", G_CALLBACK(_icon_widget_draw), (gpointer)paint);
  return da;
}

// the refinement section caption mirrors the row being refined:
// Expander header shows "(element|group|whole mask) refinement",
// and when expanded, inner header row shows <icon> <label> <actions>.
static void _refine_update_header(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module ? module->blend_data : NULL;
  if(!bd) return;

  const char *section_title = _("whole mask refinement");
  if(bd->masks_refine_scope_kind == REFINE_SCOPE_ELEMENT)
    section_title = _("element refinement");
  else if(bd->masks_refine_scope_kind == REFINE_SCOPE_GROUP)
    section_title = _("group refinement");

  if(bd->masks_refine_section_label)
    gtk_label_set_text(GTK_LABEL(bd->masks_refine_section_label), section_title);

  if(bd->masks_refine_icon_box)
  {
    // Clear existing icon widget
    GList *children =
      gtk_container_get_children(GTK_CONTAINER(bd->masks_refine_icon_box));
    for(GList *c = children; c; c = g_list_next(c))
      gtk_widget_destroy(GTK_WIDGET(c->data));
    g_list_free(children);
  }

  gchar *name = NULL;
  GtkWidget *icon_w = NULL;

  if(bd->masks_refine_scope_kind == REFINE_SCOPE_ELEMENT)
  {
    dt_masks_form_t *form =
      dt_masks_get_from_id(darktable.develop, bd->masks_refine_scope_formid);
    if(form)
    {
      // the name as its row shows it: the icon below already says the type
      name = _form_display_name(form);
      if(form->type & DT_MASKS_PARAMETRIC)
      {
        const gchar *code = dt_masks_parametric_type_label(form);
        if(code) icon_w = _make_channel_handle(code, NULL);
      }
      else
      {
        const guint kind = _form_kind(form);
        DTGTKCairoPaintIconFunc paint = _kind_icon_paint(kind);
        if(paint) icon_w = _make_icon_widget(paint);
      }
    }
    else
    {
      name = g_strdup(_("shape"));
    }
  }
  else if(bd->masks_refine_scope_kind == REFINE_SCOPE_GROUP)
  {
    dt_masks_form_t *grp = _module_mask_group(module);
    const dt_masks_point_group_t *head = _group_point(grp, bd->masks_refine_scope_formid);
    const char *custom_name = _group_custom_name(grp, bd->masks_refine_scope_formid);
    name =
      custom_name
        ? g_strdup(custom_name)
        : g_strdup_printf("%s-%d", _within_name(head ? head->state : 0),
                          _group_ordinal_of_cid(module, bd->masks_refine_scope_formid));

    DTGTKCairoPaintIconFunc paint = _op_paint_for_state(head ? head->state : 0);
    if(paint) icon_w = _make_icon_widget(paint);
  }
  else
  {
    name = g_strdup(_("whole mask"));
    icon_w = _make_icon_widget(dtgtk_cairo_paint_masks_eye);
  }

  if(icon_w && bd->masks_refine_icon_box)
  {
    dt_gui_box_add(bd->masks_refine_icon_box, icon_w);
    gtk_widget_show_all(bd->masks_refine_icon_box);
  }

  if(bd->masks_refine_name_label)
    gtk_label_set_text(GTK_LABEL(bd->masks_refine_name_label), name ? name : "");
  g_free(name);

  // Update bypass button state
  gpointer key = _refine_scope_key(bd);
  gboolean bypassed = FALSE;
  if(bd->masks_refine_bypassed)
    bypassed = GPOINTER_TO_INT(g_hash_table_lookup(bd->masks_refine_bypassed, key));
  bd->masks_refine_updating = TRUE;
  if(bd->masks_refine_bypass_btn)
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_refine_bypass_btn),
                                 bypassed);
  bd->masks_refine_updating = FALSE;

  // Update indicator icon (shows whether current target has active refinements)
  dt_masks_refinement_t r = { 0 };
  _refine_read_controls(bd, &r);
  const gboolean has_refinement = (r.enabled != 0);
  if(bd->masks_refine_indicator_icon)
  {
    gtk_widget_set_opacity(bd->masks_refine_indicator_icon, has_refinement ? 1.0 : 0.25);
    gtk_widget_set_tooltip_text(bd->masks_refine_indicator_icon,
                                has_refinement
                                  ? _("refinements are active for this target")
                                  : _("no refinements for this target"));
  }

  _refine_update_expanded_state(module);
}

// (re)build the refinement-header group selector from the current mask group:
// title-only now (see _refine_update_header) -- no selector combo any more,
// the scope follows the mask list selection alone. Still refreshes the reset
// button's visibility (flexi-only) and the caption/sliders for the current
// scope, so it stays safe to call from every place that used to also rebuild
// the combo (list rebuild, target-sensitivity refresh).
static void _refine_scope_combo_rebuild(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd) return;

  const gboolean mode_flexi = module->blend_params->mask_mode & DEVELOP_MASK_FLEXI;

  if(bd->masks_refine_reset_btn)
    gtk_widget_set_visible(bd->masks_refine_reset_btn, mode_flexi);

  _flexi_refine_follow_selection(bd);
  _refine_update_header(module);

  _refine_populate(module);
}

// defined with the other badge helpers (it needs the row/run lookups), but
// called from _props_row_apply below on every opacity change
static void _refresh_lowop_badges(dt_iop_module_t *module);
static void _update_lowop_badge(GtkWidget *badge,
                                const float opacity,
                                const gboolean is_group,
                                const gboolean is_noop,
                                const char *noop_reason);
static void
_set_badge_active(GtkWidget *badge, gboolean active, const char *tooltip_when_active);
static const char *_solo_badge_tooltip(void);
enum
{
  MASK_SOLO_BADGE_NONE = 0,
  MASK_SOLO_BADGE_SOLO,
  MASK_SOLO_BADGE_DISABLE,
  MASK_SOLO_BADGE_BYPASS = MASK_SOLO_BADGE_DISABLE,
};
static void _set_solo_status_badge(GtkWidget *badge, int status);

// ===========================================================================
// Per-shape/raster/group/parametric inline "properties" expanders.
//
// Formerly a single, selection-following "element properties" panel (a
// visually separate sibling of the mask refinement panel above, scoped by
// masks_refine_scope_kind/_formid). Per the revamped design, every row now
// owns its own inline expander instead -- shapes and raster forms get a new
// toggle button next to their solo-edit slot, groups get one in their header,
// and parametric rows reuse their existing in/out chevron to also reveal
// opacity. This block keeps the classic mask manager's own delta-based commit
// machinery (modify_property/dt_masks_form_change_opacity) generalized to take
// an explicit per-row target instead of a single global scope.
//
// NOTE ON PROVENANCE. Comments in this file refer to "the removed mask
// manager": that is src/libs/masks.c, the lib this branch DELETED when the
// flexi panel replaced it. Those references record which behaviour a piece of
// code was written to reproduce -- they are not pointers to code you can go and
// read, and there is nothing left to keep in sync with.
//
// This metadata table began as a copy of that lib's file-local
// _masks_properties (name/format/min/max/relative/boolean per
// dt_masks_property_t). It is now the only copy, so it is authoritative rather
// than a mirror.
static const struct
{
  gchar *name;
  gchar *format;
  float min, max;
  gboolean relative;
  gboolean boolean;
} _blend_masks_properties[DT_MASKS_PROPERTY_LAST] = {
  [DT_MASKS_PROPERTY_OPACITY] = { N_("opacity"), "%", 0, 1, FALSE, FALSE },
  [DT_MASKS_PROPERTY_SIZE] = { N_("size"), "%", 0.0001, 1, TRUE, FALSE },
  [DT_MASKS_PROPERTY_HARDNESS] = { N_("hardness"), "%", 0.0001, 1, TRUE, FALSE },
  [DT_MASKS_PROPERTY_FEATHER] = { N_("fade-out border"), "%", 0.0001, 1, TRUE, FALSE },
  [DT_MASKS_PROPERTY_ROTATION] = { N_("rotation"), "°", 0, 360, FALSE, FALSE },
  [DT_MASKS_PROPERTY_CURVATURE] = { N_("curvature"), "%", -1, 1, FALSE, FALSE },
  [DT_MASKS_PROPERTY_COMPRESSION] = { N_("compression"), "%", 0.0001, 1, TRUE, FALSE },
  [DT_MASKS_PROPERTY_CLEANUP] = { N_("cleanup"), "", 0, 100, FALSE, FALSE },
  [DT_MASKS_PROPERTY_SMOOTHING] = { N_("smoothing"), "", 0, 1.3, FALSE, FALSE },
  [DT_MASKS_PROPERTY_REFINE] = { N_("refine mask boundary"), "", 0, 1, FALSE, TRUE },
};

// "use sliders for opacity" only takes effect while "auto-expand selected" is
// also on: the slider lives at the top of an expanded panel, and auto-expand is
// what guarantees the panel you are working in is the one open. Without it, the
// list starts collapsed, so moving opacity there would put it out of sight
// everywhere. The panel says so by nesting the checkbox under auto-expand and
// greying it out (see _add_masks_panel_options_box); this is the same rule
// applied to the layout itself, so a conf value left over from a session with
// auto-expand on cannot leak a half-applied layout.
gboolean _model_opacity_sliders_in_effect(const gboolean auto_expand,
                                          const gboolean use_sliders)
{
  return auto_expand && use_sliders;
}

// does an element row of this kind carry an expander of its own?
//
// A drawn shape (circle/path/... and AI objects alike) always has properties
// worth an expander, and a parametric row's in/out chevron is its expander. A
// raster mask has exactly one property -- opacity, since its modify_property is
// NULL -- which its row header shows inline, so it has nothing to expand and
// gets no chevron; "use sliders for opacity" moves that one property into the
// expanded panel, which is what gives a raster row something to show and, with
// it, a chevron like every other element's. Groups never come through here:
// their chevron reveals their members, not properties (see
// _group_expand_toggled).
gboolean _model_row_is_expandable(const dt_masks_type_t type,
                                  const gboolean opacity_sliders)
{
  if(type & DT_MASKS_RASTER) return opacity_sliders;
  return TRUE;
}

// the shared checkerboard-to-white track every opacity slider in this panel
// wears; defined further below, alongside the inline-opacity styling helpers.
static void _style_opacity_gradient(GtkWidget *slider);

// refresh the compact value label that follows an opacity slider after a
// programmatic (and therefore signal-less) set; defined further below,
// alongside the label itself.
static void _refresh_inline_opacity_label(GtkWidget *slider);

// apply a single property's new value to every form in `target_formids`,
// following the exact delta protocol the removed mask manager's _property_changed
// uses: modify_property takes (old_val -> new_val) and derives its own
// ratio/delta internally, so *last_value must be the previously *committed*
// value for this specific row/control, never the shape's raw current
// size/hardness/etc -- these are relative controls, not absolute readouts.
// Also drives the live on-canvas preview (dt_masks_gui_form_create) exactly
// as the classic manager does, using each shape's absolute position in the
// group's full points list. `target_formids` is not owned/freed here -- the
// caller builds and frees it (a single formid, or a whole group's run).
// both defined below, next to the row index they read: the rows showing one
// form, and the refresh that pushes an edit made in one of them into the rest
static GSList *_masks_rows_for_form(dt_iop_gui_blend_data_t *bd,
                                    const dt_mask_id_t formid);
static void _refresh_sibling_prop_rows(dt_iop_module_t *module,
                                       GList *formids,
                                       GtkWidget *src);

static void _props_row_apply(dt_iop_module_t *module,
                             GList *target_formids,
                             const int prop,
                             GtkWidget *widget,
                             float *last_value,
                             const gboolean allow_hide)
{
  dt_develop_t *dev = darktable.develop;
  dt_masks_form_gui_t *gui = dev->form_gui;
  dt_masks_form_t *grp = _module_mask_group(module);
  const gboolean is_bool = _blend_masks_properties[prop].boolean;

  if(!grp || !gui || !target_formids)
  {
    // only populate-style callers are allowed to hide a control -- an
    // interactive edit (allow_hide == FALSE) must never make its own widget
    // vanish out from under the user's drag; see the allow_hide comment below.
    if(allow_hide) gtk_widget_hide(widget);
    return;
  }

  const float value = is_bool
                        ? (float)gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget))
                        : dt_bauhaus_slider_get(widget);
  const float old_value = *last_value;

  int count = 0;
  float sum = 0;
  float min = _blend_masks_properties[prop].min, max = _blend_masks_properties[prop].max;
  if(!is_bool)
  {
    if(_blend_masks_properties[prop].relative)
    {
      max /= min;
      min /= _blend_masks_properties[prop].max;
    }
    else
    {
      max -= min;
      min -= _blend_masks_properties[prop].max;
    }
  }

  // at any depth: a row can edit a member of a nested group. Positions index
  // the canvas's copy of the group (see _canvas_points)
  GArray *pts = _canvas_points(grp);
  // the shapes already reshaped in this pass. A mask can reference one shape
  // several times, and geometry belongs to the shape, not to the reference:
  // applying it once per reference made a single slider step land two or three
  // times, which for a relative property (size) compounds into a value the
  // user never asked for
  GList *reshaped = NULL;
  for(guint k = 0; k < pts->len; k++)
  {
    dt_masks_point_group_t *fpt = g_array_index(pts, _canvas_point_t, k).pt;
    const int fpos = g_array_index(pts, _canvas_point_t, k).pos;
    if(!g_list_find(target_formids, GINT_TO_POINTER(fpt->formid))) continue;

    dt_masks_form_t *sel = dt_masks_get_from_id(dev, fpt->formid);
    if(!sel) continue;

    if(prop == DT_MASKS_PROPERTY_OPACITY)
    {
      // mutate opacity in place and commit exactly once after the loop (below),
      // instead of dt_masks_form_change_opacity's per-form history commit: a
      // group/cluster drag would otherwise fire one full history item (3-pipe
      // synch + panel rebuild) per member, multiplied again per drag tick.
      // the floor is 0, not the classic manager's 0.05: that clamp (upstream
      // c646d7e959, "0% means no effect anyway so better remove the shape")
      // existed because a fully transparent shape was indistinguishable from a
      // live one in the old flat list. This panel makes it visible instead --
      // an element or group under MASK_LOW_OPACITY_WARN carries a warning badge
      // (see _make_lowop_badge) -- so the slider can reach the end of the 0-100
      // range it advertises.
      const float new_opacity = CLAMP(fpt->opacity + (value - old_value), 0.0f, 1.0f);
      fpt->opacity = new_opacity;
      sum += new_opacity;
      max = fminf(max, 1.0f - new_opacity);
      min = fmaxf(min, 0.0f - new_opacity);
      ++count;
    }
    else if(sel->functions && sel->functions->modify_property)
    {
      if(g_list_find(reshaped, GINT_TO_POINTER(fpt->formid)))
      {
        // the shape was already changed through another of its references;
        // only this reference's own canvas copy still has to follow it
        if(value != old_value) dt_masks_gui_form_create(sel, gui, fpos, dev->gui_module);
        continue;
      }
      reshaped = g_list_prepend(reshaped, GINT_TO_POINTER(fpt->formid));
      const int saved_count = count;
      sel->functions->modify_property(sel, prop, old_value, value, &sum, &count, &min,
                                      &max);
      if(count != saved_count && value != old_value)
        dt_masks_gui_form_create(sel, gui, fpos, dev->gui_module);
    }
  }
  g_array_free(pts, TRUE);
  g_list_free(reshaped);

  // visibility ("does this property even apply to the current target set") is
  // decided only at populate time -- an interactive value-change (allow_hide
  // == FALSE) must never toggle it, or a transient count==0 mid-drag (e.g.
  // while a shape is being edited) would hide the very slider the user is
  // dragging, and it would stay hidden until the row's expander is reopened.
  if(allow_hide) gtk_widget_set_visible(widget, count != 0);
  if(!count) return;

  // dt_bauhaus_slider_set_soft_range/dt_bauhaus_slider_set and
  // gtk_toggle_button_set_active below re-emit "value-changed"/"toggled" on
  // this same widget, which would otherwise re-enter the row's own
  // changed-handler and recurse forever -- guard exactly like masks.c's own
  // DT_ENTER_GUI_UPDATE()/DT_LEAVE_GUI_UPDATE() around _property_changed (this
  // guard is what fixed a real stack-overflow crash from this same re-entrancy).
  DT_ENTER_GUI_UPDATE();
  if(is_bool)
  {
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(widget), (sum / count) > 0.5f);
    *last_value = (float)gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
  }
  else
  {
    if(_blend_masks_properties[prop].relative)
    {
      max *= sum / count;
      min *= sum / count;
    }
    else
    {
      max += sum / count;
      min += sum / count;
    }
    if(dt_isnan(min)) min = _blend_masks_properties[prop].min;
    if(dt_isnan(max)) max = _blend_masks_properties[prop].max;
    dt_bauhaus_slider_set_soft_range(widget, min, max);
    dt_bauhaus_slider_set(widget, sum / count);
    *last_value = dt_bauhaus_slider_get(widget);
  }
  DT_LEAVE_GUI_UPDATE();

  // the set above was silent (see _refresh_inline_opacity_label): where this
  // widget is the hidden slider behind a row header's compact opacity value,
  // and the value it settled on is not the one the user's own gesture left it
  // at (a clamp, or a soft-range recompute), that label has to be told by hand
  // or it keeps reading the number from before this call.
  _refresh_inline_opacity_label(widget);

  dt_control_queue_redraw_center();

  // an opacity change can push a row (or its whole group) across the
  // low-opacity threshold -- refresh the badges in place, on every drag tick.
  // Nothing else in the panel changes, so this must not be a rebuild.
  if(prop == DT_MASKS_PROPERTY_OPACITY)
  {
    _refresh_lowop_badges(module);
  }

  // the same shape can have a row under each group that references it, and
  // every one of those rows carries its own controls: show the new value in
  // all of them, not just the one the user is dragging
  if(value != old_value) _refresh_sibling_prop_rows(module, target_formids, widget);

  // commit exactly one history item for the whole gesture across every targeted
  // form, whatever the property -- opacity included (the OPACITY branch above no
  // longer self-commits per form, so a multi-form drag is now a single commit).
  if(value != old_value) dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
}

// Quad for the shrink/grow slider's unit toggle: always shows "%" inside a
// button-like square frame. Its active state (drawn brighter by bauhaus) tells
// whether % mode is engaged; the slider's own value format spells out the
// unit. Exact copy of the removed mask manager's own _paint_resize_unit -- file-local
// static there, so duplicated here rather than shared across the two TUs.
static void _props_paint_resize_unit(cairo_t *cr,
                                     const gint x,
                                     const gint y,
                                     const gint w,
                                     const gint h,
                                     const gint flags,
                                     void *data)
{
  const char *txt = "%";
  cairo_save(cr);

  const double side = MIN(w, h);
  const double fx = x + (w - side) / 2.0;
  const double fy = y + (h - side) / 2.0;

  PangoLayout *layout = pango_cairo_create_layout(cr);
  if(darktable.bauhaus->pango_font_desc)
    pango_layout_set_font_description(layout, darktable.bauhaus->pango_font_desc);
  pango_layout_set_text(layout, txt, -1);
  int tw = 0, th = 0;
  pango_layout_get_pixel_size(layout, &tw, &th);

  const double pad = DT_PIXEL_APPLY_DPI(1.0);
  const double avail = side - 2.0 * pad;
  const double scale = (tw > 0 && th > 0) ? fmin(avail / tw, avail / th) : 1.0;
  cairo_translate(cr, fx + (side - tw * scale) / 2.0, fy + (side - th * scale) / 2.0);
  cairo_scale(cr, scale, scale);
  pango_cairo_show_layout(cr, layout);
  g_object_unref(layout);
  cairo_restore(cr);
}

// Set the shape to the slider's absolute offset and commit one history item --
// mirrors the removed mask manager's own _resize_commit exactly, but scoped directly to
// this row's single shape (no "which selected path" ambiguity to resolve: the
// row already names one shape by construction).
static void _props_resize_commit(dt_masks_props_row_editor_t *ed)
{
  dt_develop_t *dev = darktable.develop;
  dt_masks_form_gui_t *gui = dev->form_gui;
  dt_masks_form_t *form = dt_masks_get_from_id(dev, ed->formid);
  if(!form || !gui || !form->functions || !form->functions->resize) return;

  const int amount = (int)roundf(dt_bauhaus_slider_get(ed->resize_widget));
  const gboolean pct = dt_bauhaus_widget_get_quad_active(ed->resize_widget);

  if(!form->functions->resize(form, amount, pct) && amount < 0)
    dt_control_log(_("shrink amount too large: the path would disappear"));

  // positions index the canvas's copy of the group (see _canvas_points)
  GArray *pts = _canvas_points(_module_mask_group(ed->module));
  int pos = 0;
  for(guint k = 0; k < pts->len; k++)
    if(g_array_index(pts, _canvas_point_t, k).pt->formid == ed->formid)
    {
      pos = g_array_index(pts, _canvas_point_t, k).pos;
      break;
    }
  g_array_free(pts, TRUE);

  dt_masks_gui_form_create(form, gui, pos, dev->gui_module);
  dt_dev_add_masks_history_item(dev, dev->gui_module, TRUE);
  dt_control_queue_redraw_center();
}

static gboolean _props_resize_timeout(gpointer data)
{
  dt_masks_props_row_editor_t *ed = data;
  ed->resize_timer = 0;
  _props_resize_commit(ed);
  return G_SOURCE_REMOVE;
}

// Debounce: morphing is expensive, so commit ~180 ms after the last change
// rather than on every slider tick -- same interval as masks.c's own slider.
static void _props_resize_schedule_commit(dt_masks_props_row_editor_t *ed)
{
  if(ed->resize_updating) return;
  if(ed->resize_timer) g_source_remove(ed->resize_timer);
  ed->resize_timer = g_timeout_add(180, _props_resize_timeout, ed);
}

static void _props_resize_amount_changed(GtkWidget *w, dt_masks_props_row_editor_t *ed)
{
  _props_resize_schedule_commit(ed);
}

// Reflect the current unit in the slider's value suffix (e.g. "5 px" / "5 %").
static void _props_resize_sync_unit(dt_masks_props_row_editor_t *ed)
{
  const gboolean pct = dt_bauhaus_widget_get_quad_active(ed->resize_widget);
  dt_bauhaus_slider_set_format(ed->resize_widget, pct ? " %" : " px");
}

// the unit toggle lives in the slider's quad; bauhaus flips the active flag
// before emitting "quad-pressed", so the new state is read directly. The unit
// preference is shared with the classic mask manager's own slider (same conf
// key), so switching it in either place keeps both in sync.
static void _props_resize_unit_quad(GtkWidget *w, dt_masks_props_row_editor_t *ed)
{
  const gboolean pct = dt_bauhaus_widget_get_quad_active(w);
  dt_conf_set_string("masks/path_resize_unit", pct ? "% of path size" : "pixels");
  _props_resize_sync_unit(ed);
  _props_resize_schedule_commit(ed);
}

// Refresh the shrink/grow slider for this row's shape: shown only for a path,
// mirroring the offset the path mask currently has applied (0 for a fresh
// shape, or whatever a scroll-wheel/previous resize left, or a size/feather/
// rotation edit having reset it -- see _props_row_control_changed). Called on
// populate and after every one of this row's own edits.
static void _props_resize_update(dt_masks_props_row_editor_t *ed)
{
  if(!ed->resize_widget) return;
  dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, ed->formid);
  const gboolean is_path = form && form->functions && form->functions->resize_get;

  if(is_path)
  {
    const gboolean pct = dt_bauhaus_widget_get_quad_active(ed->resize_widget);
    float amount = 0.0f;
    form->functions->resize_get(form, pct, &amount);

    // reflect the current offset without triggering a (re)commit
    if(ed->resize_timer)
    {
      g_source_remove(ed->resize_timer);
      ed->resize_timer = 0;
    }
    ed->resize_updating = TRUE;
    dt_bauhaus_slider_set(ed->resize_widget, roundf(amount));
    ed->resize_updating = FALSE;
  }
  gtk_widget_set_visible(ed->resize_widget, is_path);
}

// destroy-notify for a props row editor's "props-editor" data: cancels any
// pending debounced resize commit (see _props_resize_schedule_commit) before
// freeing, so a row torn down mid-debounce (list rebuild, shape deletion, ...)
// never fires a commit against a dangling ed pointer.
static void _props_row_editor_free(gpointer data)
{
  dt_masks_props_row_editor_t *ed = data;
  if(ed->resize_timer) g_source_remove(ed->resize_timer);
  g_free(ed);
}

// the explicit target formid list for a props row editor: its own single id,
// or (for a group row) every member of that group's run -- the same run
// _refine_commit_nonglobal broadcasts refinements to, via
// _selected_group_formids, so "the group" means the same set of shapes
// everywhere. Caller frees the returned list.
static GList *_props_row_target_formids(const dt_masks_props_row_editor_t *ed)
{
  if(!ed) return NULL;
  if(ed->is_group)
    return _selected_group_formids(_module_mask_group(ed->module), ed->formid);
  return g_list_prepend(NULL, GINT_TO_POINTER(ed->formid));
}

// (re)populate every one of this row's own controls -- called once right
// after construction and whenever the row's expander is opened. Sliders are
// never populated with a shape's absolute current value -- like the classic
// mask manager, they are delta/ratio controls that start from whatever they
// were last left at (ed->last_value[]), and re-running _props_row_apply with
// the unchanged current value is a neutral no-op that only recomputes which
// controls apply (count != 0) and their soft range for this row's own target.
static void _props_row_populate(dt_masks_props_row_editor_t *ed)
{
  if(!ed) return;
  GList *ids = _props_row_target_formids(ed);
  for(int i = 0; i < DT_MASKS_PROPERTY_LAST; i++)
    if(ed->widget[i])
      _props_row_apply(ed->module, ids, i, ed->widget[i], &ed->last_value[i], TRUE);
  g_list_free(ids);

  // capture each relative slider's own absolute reading (just applied above,
  // via _props_row_apply's "dt_bauhaus_slider_set(widget, sum / count)") the
  // first time this row is ever populated, and use it as the widget's own
  // double-click reset target from then on -- a ratio control's neutral
  // reading of 0 double-click-resets to "no change from wherever it is right
  // now", which is a genuine no-op and not useful; this makes double-click
  // instead undo whatever edits were made since the row was first opened.
  if(!ed->relative_baseline_set)
  {
    for(int i = 0; i < DT_MASKS_PROPERTY_LAST; i++)
      if(ed->widget[i] && _blend_masks_properties[i].relative)
        dt_bauhaus_slider_set_default(ed->widget[i], ed->last_value[i]);
    ed->relative_baseline_set = TRUE;
  }
}

// a shape referenced more than once in one mask has a row under each group
// that references it, each with its own copy of the controls. Push an edit
// made in one of them into the others, which otherwise keep reading the shape
// as it was before the edit until something rebuilds the list.
static void _refresh_sibling_prop_rows(dt_iop_module_t *module,
                                       GList *formids,
                                       GtkWidget *src)
{
  dt_iop_gui_blend_data_t *bd = module ? module->blend_data : NULL;
  if(!bd) return;
  // _props_row_populate re-enters _props_row_apply, which comes back here:
  // refresh the siblings of an edit, never the siblings of a refresh
  static gboolean refreshing = FALSE;
  if(refreshing) return;
  refreshing = TRUE;

  // a row's controls live in one of two editors: the expanded properties box,
  // and the header's own compact opacity control (see _make_shape_row)
  static const char *const editor_boxes[] = { "props-editor-box",
                                              "inline-opacity-editor-box" };
  for(GList *f = formids; f; f = g_list_next(f))
  {
    for(GSList *r = _masks_rows_for_form(bd, GPOINTER_TO_INT(f->data)); r; r = r->next)
    {
      for(size_t b = 0; b < G_N_ELEMENTS(editor_boxes); b++)
      {
        GtkWidget *box = g_object_get_data(G_OBJECT(r->data), editor_boxes[b]);
        dt_masks_props_row_editor_t *ed =
          box ? g_object_get_data(G_OBJECT(box), "props-editor") : NULL;
        if(!ed) continue;
        // the control the user is holding keeps the position they left it at;
        // repopulating it would fight the gesture (see _props_row_apply's own
        // allow_hide reasoning)
        gboolean owns_src = FALSE;
        for(int i = 0; i < DT_MASKS_PROPERTY_LAST; i++)
          if(ed->widget[i] == src) owns_src = TRUE;
        if(!owns_src) _props_row_populate(ed);
      }
    }
  }
  refreshing = FALSE;
}

// shared value-changed/toggled handler for a props row editor's controls. The
// control's own property index is stashed on the widget at construction time
// (see "dt-prop"), so one handler can serve all ten like _refine_control_changed
// does for the six refinement controls.
static void _props_row_control_changed(GtkWidget *widget, dt_masks_props_row_editor_t *ed)
{
  if(DT_IN_GUI_UPDATE() || !ed || !ed->module || !ed->module->blend_data
     || !((dt_iop_gui_blend_data_t *)ed->module->blend_data)->blend_inited)
    return;
  const int prop = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(widget), "dt-prop"));
  GList *ids = _props_row_target_formids(ed);
  _props_row_apply(ed->module, ids, prop, widget, &ed->last_value[prop], FALSE);
  g_list_free(ids);

  // a size/feather/rotation edit reshapes the path and drops its shrink/grow
  // baseline (see path.c); refresh the resize slider so it reads back 0 --
  // mirrors the removed mask manager's own _property_changed "reshaped" handling.
  if(ed->resize_widget
     && (prop == DT_MASKS_PROPERTY_SIZE || prop == DT_MASKS_PROPERTY_FEATHER
         || prop == DT_MASKS_PROPERTY_ROTATION))
    _props_resize_update(ed);
}

// build one row/group's own inline properties editor: either just the opacity
// control (raster/group rows) or all 10 classic mask-manager properties
// (shape rows; the ones a raster form's NULL modify_property naturally hides
// via the count==0 rule above still collapse down to opacity-only at runtime).
// Mirrors _build_param_row_editor's exact show_all-then-no_show_all
// sequencing: show every child at least once (so no_show_all does not
// permanently hide something that was never shown), *then* mark the whole box
// no_show_all so no ancestor's later show_all (e.g. the group-block reveal in
// _build_masks_list) can force it back open regardless of this row's own
// expander state.
static GtkWidget *_build_props_row_editor(dt_iop_module_t *module,
                                          const dt_mask_id_t formid,
                                          const gboolean is_group,
                                          const gboolean opacity_only,
                                          const gboolean exclude_opacity)
{
  dt_masks_props_row_editor_t *ed = g_malloc0(sizeof(dt_masks_props_row_editor_t));
  ed->module = module;
  ed->formid = formid;
  ed->is_group = is_group;
  ed->opacity_only = opacity_only;

  GtkWidget *box = dt_gui_vbox();
  for(int i = 0; i < DT_MASKS_PROPERTY_LAST; i++)
  {
    if(opacity_only && i != DT_MASKS_PROPERTY_OPACITY) continue;
    // opacity has its own always-visible inline slider in the row's header
    // now (shape/raster rows, see _make_shape_row) -- exclude it here so it
    // is not also editable a second time from this expander.
    if(exclude_opacity && i == DT_MASKS_PROPERTY_OPACITY) continue;

    GtkWidget *w;
    if(_blend_masks_properties[i].boolean)
    {
      w = gtk_check_button_new_with_label(_(_blend_masks_properties[i].name));
      ed->last_value[i] = (float)gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(w));
      g_object_set_data(G_OBJECT(w), "dt-prop", GINT_TO_POINTER(i));
      g_signal_connect(G_OBJECT(w), "toggled", G_CALLBACK(_props_row_control_changed),
                       ed);
    }
    else
    {
      // every property here, opacity included, is delta-applied off this
      // slider's own last known position (see _props_row_apply's "value -
      // old_value" and _props_row_populate, which seeds both the slider and
      // *last_value from the target's actual current value right after
      // building/reopening this row). For a relative property 0 is a
      // meaningful "no change" default to double-click-reset to -- but
      // opacity's own slider shows an *absolute* 0-100% position, so
      // double-clicking it must reset that position to 100%, not 0: with
      // *last_value already sitting at the target's real opacity, resetting
      // the widget's own default to 1.0 makes that delta land exactly on
      // "set to 100%" instead of "subtract the entire current opacity".
      float defval = 0.0;
      if(i == DT_MASKS_PROPERTY_OPACITY) defval = 1.0;
      // a relative (ratio) property's neutral "no change" reading is an
      // exact 0 -- modify_property's own ratio = (!old_val || !new_val) ?
      // 1.0f : new_val/old_val (e.g. circle.c) already treats a literal 0
      // reading as identity, precisely so the widget's own reset can use it.
      // But dt_bauhaus_slider_set() always clamps to the widget's *hard*
      // min, so constructing the slider with the property table's own
      // advertised min (0.0001, used elsewhere as this property's soft
      // display floor) would clamp a double-click reset to 0.0001 instead
      // of 0 -- which is not "no change", it is "ratio = 0.0001/old_value",
      // i.e. shrink the shape almost to nothing. Give the widget itself a
      // hard min of 0 for relative properties so the reset can actually
      // land on the true neutral value; the property table's 0.0001 is
      // still used unchanged everywhere else (soft-range floor math in
      // _props_row_apply, modify_property's own clamps, ...).
      const float widget_min =
        _blend_masks_properties[i].relative ? 0.0f : _blend_masks_properties[i].min;
      w = dt_bauhaus_slider_new_with_range(module, widget_min,
                                           _blend_masks_properties[i].max, 0, defval, 2);
      dt_bauhaus_widget_set_label(w, N_("blend"), _blend_masks_properties[i].name);
      dt_bauhaus_slider_set_format(w, _blend_masks_properties[i].format);
      dt_bauhaus_slider_set_digits(w, 2);
      if(_blend_masks_properties[i].relative) dt_bauhaus_slider_set_log_curve(w);
      ed->last_value[i] = dt_bauhaus_slider_get(w);
      g_object_set_data(G_OBJECT(w), "dt-prop", GINT_TO_POINTER(i));
      g_signal_connect(G_OBJECT(w), "value-changed",
                       G_CALLBACK(_props_row_control_changed), ed);
      // a bauhaus slider paints its own opaque pill background from its own
      // #bauhaus-slider CSS node, which would otherwise occlude the row's
      // hover/selection wash right where the slider sits (same fix already
      // applied to the boost-factor slider, see .mask-boost-factor-slider).
      dt_gui_add_class(w, "mask-props-slider");
      // no quad icon on any of these sliders -- without this the slider
      // reserves the quad's width unused, reading as narrower than the row
      // it sits in (same reasoning as the boost-factor slider's own call).
      dt_bauhaus_widget_set_quad_visibility(w, FALSE);
      // every opacity slider this panel shows carries the same checkerboard-
      // to-white track, whether it ends up inline in a row header (where
      // _style_inline_opacity_box applies it again, harmlessly) or leading an
      // expanded panel under "show opacity slider in expanded elements"
      if(i == DT_MASKS_PROPERTY_OPACITY) _style_opacity_gradient(w);
    }
    ed->widget[i] = w;
    dt_gui_box_add(box, w);
  }

  // path-only shrink/grow (outset/inset) control -- mirrors the removed mask manager's
  // own "shrink or grow" slider exactly (same conf-stored unit, same debounced
  // resize()/resize_get() calls into path.c's cache), just scoped to this row's
  // single shape instead of "whichever single path is selected". A group or
  // opacity-only row (raster/group headers) never gets one; _props_resize_update
  // hides it at runtime for anything but a path.
  if(!is_group && !opacity_only)
  {
    GtkWidget *w = dt_bauhaus_slider_new_with_range(module, -1000, 1000, 1, 0.0, 0);
    dt_bauhaus_widget_set_label(w, N_("blend"), N_("shrink or grow"));
    dt_bauhaus_slider_set_soft_range(w, -20, 20);
    dt_bauhaus_slider_set_format(w, "");
    gtk_widget_set_tooltip_text(
      w, _("grow (positive) or shrink (negative) the selected path,\n"
           "relative to its shape when selected; 0 restores the original"));
    g_signal_connect(G_OBJECT(w), "value-changed",
                     G_CALLBACK(_props_resize_amount_changed), ed);
    dt_gui_add_class(w, "mask-props-slider");

    // unit (px / %) toggle in the slider's quad -- kept visible, unlike the
    // other properties sliders above, since it is this control's own setting
    dt_bauhaus_widget_set_quad_paint(w, _props_paint_resize_unit, 0, NULL);
    dt_bauhaus_widget_set_quad_toggle(w, TRUE);
    {
      const char *unit = dt_conf_get_string_const("masks/path_resize_unit");
      dt_bauhaus_widget_set_quad_active(w, !g_strcmp0(unit, "% of path size"));
    }
    dt_bauhaus_widget_set_quad_tooltip(
      w, _("shrink/grow unit: image pixels (px) or % of path size - click to toggle"));
    g_signal_connect(G_OBJECT(w), "quad-pressed", G_CALLBACK(_props_resize_unit_quad),
                     ed);

    ed->resize_widget = w;
    _props_resize_sync_unit(ed);
    dt_gui_box_add(box, w);

    // "size" scales the shape live; "shrink or grow" insets/outsets its outline --
    // keep this slider right below "size", matching masks.c's own ordering,
    // instead of at the end of the property list.
    if(ed->widget[DT_MASKS_PROPERTY_SIZE])
    {
      GList *kids = gtk_container_get_children(GTK_CONTAINER(box));
      const gint size_pos = g_list_index(kids, ed->widget[DT_MASKS_PROPERTY_SIZE]);
      if(size_pos >= 0) gtk_box_reorder_child(GTK_BOX(box), w, size_pos + 1);
      g_list_free(kids);
    }
  }

  // id mirrors the class for direct CSS targeting alongside the existing
  // class-based rules (shared by every row kind's props editor instance)
  gtk_widget_set_name(box, "mask-props-row-editor");
  dt_gui_add_class(box, "mask-props-row-editor");

  // a pending debounced resize commit (see _props_resize_schedule_commit) must
  // not fire after this row is torn down (e.g. the list rebuilds, or the shape
  // is deleted, within the 180ms window) -- plain g_free would leave it armed
  // with a dangling ed pointer.
  g_object_set_data_full(G_OBJECT(box), "props-editor", ed, _props_row_editor_free);
  gtk_widget_show_all(box);
  gtk_widget_set_no_show_all(box, TRUE);

  _props_row_populate(ed);
  _props_resize_update(ed);
  return box;
}

// ---- "shape properties in subpanel" ----------------------------------------

// what the subpanel holds the editor of: the selected element, if it is a
// shape, one with geometry of its own. A parametric channel, a raster mask
// and a group are none, and neither is the AI object stepped into, which then
// shows as its group
dt_mask_id_t _model_props_panel_target(const dt_iop_gui_blend_data_t *bd)
{
  const dt_mask_id_t id = bd->panel_selected_formid;
  const dt_masks_form_t *f =
    dt_is_valid_maskid(id) ? dt_masks_get_from_id(darktable.develop, id) : NULL;
  if(!f || (f->type & (DT_MASKS_PARAMETRIC | DT_MASKS_RASTER | DT_MASKS_GROUP)))
    return INVALID_MASKID;
  if((f->type & DT_MASKS_OBJECT) && _entered_object() == id) return INVALID_MASKID;
  return id;
}

static void _props_panel_set_expanded(dt_iop_gui_blend_data_t *bd, const gboolean expanded)
{
  dtgtk_togglebutton_set_paint(DTGTK_TOGGLEBUTTON(bd->props_panel_toggle_btn),
                               dtgtk_cairo_paint_solid_arrow,
                               expanded ? CPF_DIRECTION_DOWN : CPF_DIRECTION_LEFT, NULL);
  dtgtk_expander_set_expanded(DTGTK_EXPANDER(bd->props_panel_expander), expanded);
  gtk_widget_set_visible(bd->props_panel_content, expanded);
}

static void _props_panel_toggled(GtkToggleButton *btn, dt_iop_gui_blend_data_t *bd)
{
  _props_panel_set_expanded(bd, gtk_toggle_button_get_active(btn));
}

static void _props_panel_header_clicked(
  GtkGestureSingle *gesture, gint n_press, gdouble x, gdouble y, gpointer user_data)
{
  if(gtk_gesture_single_get_current_button(gesture) != GDK_BUTTON_PRIMARY) return;
  dt_iop_gui_blend_data_t *bd = user_data;
  GtkToggleButton *btn = GTK_TOGGLE_BUTTON(bd->props_panel_toggle_btn);
  gtk_toggle_button_set_active(btn, !gtk_toggle_button_get_active(btn));
}

// the subpanel shows only while it holds something: with the option on, with
// the mask list shown, and with a shape selected or being drawn
static void _props_panel_show(dt_iop_gui_blend_data_t *bd)
{
  GtkWidget *pending = bd->pending_props_box;
  const gboolean filled = (pending && gtk_widget_get_parent(pending) == bd->props_panel_content)
                          || dt_is_valid_maskid(bd->props_panel_formid);
  _box_set_visible(bd->props_panel_box,
                   _shape_props_subpanel() && filled && bd->masks_list_box
                   && gtk_widget_get_visible(GTK_WIDGET(bd->masks_list_box)));
}

// fill the subpanel from the selection: the creation controls of a shape being
// drawn, opened, or else the selected shape's properties, or else nothing, and
// hidden. Kept while it still holds what the selection asks for, since the
// editor inside may be the one being dragged; `force` rebuilds it anyway, after
// the list was rebuilt from changed data
static void _props_panel_sync(dt_iop_module_t *module, const gboolean force)
{
  dt_iop_gui_blend_data_t *bd = module ? module->blend_data : NULL;
  if(!bd || !bd->props_panel_content) return;
  const gboolean on = _shape_props_subpanel();
  GtkWidget *pending = on ? bd->pending_props_box : NULL;
  const dt_mask_id_t target = pending || !on ? INVALID_MASKID : _model_props_panel_target(bd);
  const gboolean placed = pending && gtk_widget_get_parent(pending) == bd->props_panel_content;
  if(!force && (pending ? placed : target == bd->props_panel_formid)) return;

  GList *kids = gtk_container_get_children(GTK_CONTAINER(bd->props_panel_content));
  for(GList *k = kids; k; k = g_list_next(k))
    if(k->data != pending) gtk_widget_destroy(k->data);
  g_list_free(kids);
  bd->props_panel_formid = target;

  if(pending)
  {
    if(!placed) dt_gui_box_add(bd->props_panel_content, pending);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->props_panel_toggle_btn), TRUE);
  }
  else if(dt_is_valid_maskid(target))
  {
    // opacity stays in the row header unless "use sliders for opacity" has
    // moved it into the expanded controls, as for an expanded row
    dt_gui_box_add(bd->props_panel_content,
                   _build_props_row_editor(module, target, FALSE, FALSE, !_opacity_sliders()));
  }
  _props_panel_show(bd);
}

// defined further below (near the parametric row's own use of it); forward
// declared here so _make_props_row_toggle's shared chevron button can use the
// same icon.
static void _paint_param_inout(cairo_t *cr,
                               const gint x,
                               const gint y,
                               const gint w,
                               const gint h,
                               const gint flags,
                               void *data);

// toggled handler for the shared props-row chevron built by
// _make_props_row_toggle: flips the row's remembered expand state (keyed by
// its own target id, "props-key") and shows/hides its docked editor box
// ("props-editor-box") in place -- no rebuild needed.
static void _props_row_toggled(GtkWidget *btn, dt_iop_module_t *module)
{
  if(DT_IN_GUI_UPDATE()) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  const dt_mask_id_t key = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(btn), "props-key"));
  const gboolean active = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(btn));

  // a shape/raster row's own toggle has no bubbling ancestor to select it the
  // way a group's toggle does above -- select it explicitly here instead, if
  // it wasn't already selected (never deselect: same select-only rule as
  // every other action control, see _set_form_target).
  //
  // bd->masks_suppress_toggle_select guards this against a real recursion
  // bug: "auto-expand selected" (_auto_expand_selected_row)
  // programmatically flips OTHER rows' toggles off to enforce
  // single-expansion, with that flag set for the duration. Without this
  // guard, collapsing a non-selected row's toggle here would re-select it
  // (key != panel_selected_formid is true for exactly the rows being
  // collapsed) -- which calls _auto_expand_selected_row again for that row,
  // which collapses the previously-selected row's toggle, re-selecting
  // *that* one, and so on: two rows pinging the selection back and forth
  // forever, blowing the stack (observed as a SIGSEGV "excessive recursion"
  // crash). A plain DT_ENTER/LEAVE_GUI_UPDATE would also work here, except
  // this function already bails out entirely on DT_IN_GUI_UPDATE() (see
  // above), which would then also suppress the hash/visibility update this
  // programmatic toggle still needs -- hence a separate, narrower flag.
  const gboolean is_group =
    GPOINTER_TO_INT(g_object_get_data(G_OBJECT(btn), "props-is-group"));
  if(!bd->masks_suppress_toggle_select && !is_group)
    _element_chevron_clicked(module, key, active);

  if(!bd->masks_props_expanded)
    bd->masks_props_expanded = g_hash_table_new(g_direct_hash, g_direct_equal);
  g_hash_table_insert(bd->masks_props_expanded, GUINT_TO_POINTER(key),
                      GINT_TO_POINTER(active));

  GtkWidget *editor_box = g_object_get_data(G_OBJECT(btn), "props-editor-box");
  if(editor_box)
  {
    gtk_widget_set_visible(editor_box, active);
    gtk_widget_queue_resize(editor_box);
  }
}

// declared here so _group_expand_toggled can hand the option its new
// last-expanded group; defined below, once the group header lookup exists.
static void _collapse_auto_expanded_group(dt_iop_module_t *module,
                                          const dt_mask_id_t keep_cid);

// TRUE while _auto_expand_selected_group / _collapse_auto_expanded_group are
// driving group chevrons themselves. Those calls re-enter _group_expand_toggled
// -- a toggle emits "toggled" whoever flipped it -- which must still do its
// hash and visibility work, but must NOT read the flip as the user overriding
// the option: the nested collapse would otherwise clear the very
// last-expanded-group the enforcing call is in the middle of setting, and leave
// a bogus "the user just collapsed this" one-shot behind. Single-threaded GUI
// work, so a plain file-static is enough.
static gboolean _group_expand_enforcing = FALSE;

static void _group_expand_toggled(GtkToggleButton *btn, gpointer user_data)
{
  if(DT_IN_GUI_UPDATE()) return;
  dt_iop_module_t *module = (dt_iop_module_t *)user_data;
  dt_iop_gui_blend_data_t *bd = module ? module->blend_data : NULL;
  if(!bd) return;
  const guint cid = GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(btn), "props-key"));
  const gboolean active = gtk_toggle_button_get_active(btn);
  if(!bd->masks_props_expanded)
    bd->masks_props_expanded = g_hash_table_new(g_direct_hash, g_direct_equal);
  g_hash_table_insert(bd->masks_props_expanded, GUINT_TO_POINTER(cid),
                      GINT_TO_POINTER(active));
  GtkWidget *elem_box = g_object_get_data(G_OBJECT(btn), "elem-box");
  if(elem_box)
  {
    gtk_widget_set_visible(elem_box, active);
    gtk_widget_queue_resize(elem_box);
  }

  // a real click on the chevron is the user overriding "auto-expand selected"
  // by hand, so it decides what the option considers open from here on. This
  // click also bubbles up to the group header, which selects the group (see
  // _group_header_release) -- and the selection would run auto-expand again,
  // so tell it what just happened rather than let it undo the click. GTK emits
  // "toggled" from the button's own release handler, before the event reaches
  // the header, so this always lands first.
  if(_group_expand_enforcing || !_auto_expand_selected()) return;
  if(active)
  {
    _collapse_auto_expanded_group(module, (dt_mask_id_t)cid);
    bd->masks_last_expanded_group = (dt_mask_id_t)cid;
  }
  else
  {
    bd->masks_last_expanded_group = INVALID_MASKID;
    bd->masks_group_collapse_click = (dt_mask_id_t)cid;
  }
}

// a nested group row's chevron: shows or hides the row's own groups, and
// remembers it by the nested group's form id, as a group header's chevron
// does by the group's (see _group_expand_toggled). "auto-expand selected"
// leaves it alone: what it shows follows the selection through
// _reveal_nesting instead
static void _subgroup_expand_toggled(GtkToggleButton *btn, gpointer user_data)
{
  if(DT_IN_GUI_UPDATE()) return;
  dt_iop_module_t *module = (dt_iop_module_t *)user_data;
  dt_iop_gui_blend_data_t *bd = module ? module->blend_data : NULL;
  if(!bd) return;
  const gboolean active = gtk_toggle_button_get_active(btn);
  if(!bd->masks_props_expanded)
    bd->masks_props_expanded = g_hash_table_new(g_direct_hash, g_direct_equal);
  g_hash_table_insert(bd->masks_props_expanded, g_object_get_data(G_OBJECT(btn), "props-key"),
                      GINT_TO_POINTER(active));
  GtkWidget *elem_box = g_object_get_data(G_OBJECT(btn), "elem-box");
  if(elem_box)
  {
    gtk_widget_set_visible(elem_box, active);
    gtk_widget_queue_resize(elem_box);
  }
}

// which element "auto-expand selected" keeps open at build time: the
// current selection, if it is an element that can be expanded at all, and
// otherwise whatever was expanded last (see bd->masks_last_expanded_elem).
// Selecting a group, or a raster mask while "show opacity slider in expanded
// elements" is off, therefore leaves the open element open rather than
// collapsing the panel down to nothing.
dt_mask_id_t _model_auto_expand_anchor(const dt_iop_gui_blend_data_t *bd)
{
  const dt_mask_id_t sel = bd->panel_selected_formid;
  if(dt_is_valid_maskid(sel))
  {
    const dt_masks_form_t *f = dt_masks_get_from_id(darktable.develop, sel);
    // a nested group's chevron shows groups, which follow the group half
    // (see _reveal_nesting), not properties
    if(f && !(f->type & DT_MASKS_GROUP) && _model_row_is_expandable(f->type, _opacity_sliders()))
      return sel;
  }
  return bd->masks_last_expanded_elem;
}

// the same, one level up: which group "auto-expand selected" keeps open at
// build time. A group needs no expandability test -- every group has members
// to reveal -- so this is simply the selected group, falling back to whatever
// the option opened last.
dt_mask_id_t _model_auto_expand_group_anchor(const dt_iop_gui_blend_data_t *bd)
{
  if(dt_is_valid_maskid(bd->panel_selected_group_cid))
    return bd->panel_selected_group_cid;
  return bd->masks_last_expanded_group;
}

// a real click on an element row's chevron decides what is open, not the
// selection it also makes: expanding makes this row the one open element and
// collapses the previous one, collapsing it forgets it. Without the option,
// nothing moves.
dt_masks_chevron_click_t _model_element_chevron_click(const dt_iop_gui_blend_data_t *bd,
                                                      const dt_mask_id_t id,
                                                      const gboolean expanded,
                                                      const gboolean auto_expand)
{
  dt_masks_chevron_click_t c = { INVALID_MASKID, bd->masks_last_expanded_elem };
  if(!auto_expand) return c;
  if(expanded)
  {
    if(dt_is_valid_maskid(c.last_expanded) && c.last_expanded != id)
      c.collapse = c.last_expanded;
    c.last_expanded = id;
  }
  else if(c.last_expanded == id)
    c.last_expanded = INVALID_MASKID;
  return c;
}

// build the toggle button + docked editor pair shared by shape rows, raster
// rows, and group headers: a chevron styled like every other row expander
// (".mask-row-expander"), remembering its expanded state
// across rebuilds in bd->masks_props_expanded (keyed by `key` -- a shape's own
// formid, or a group's head/cid), mirroring bd->masks_cluster_expanded's exact
// pattern. Returns the toggle button; *editor_box_out receives the editor box
// to dock into the row/group layout (already built with its initial
// expanded/collapsed visibility applied).
static GtkWidget *_make_props_row_toggle(dt_iop_module_t *module,
                                         const dt_mask_id_t key,
                                         const gboolean is_group,
                                         const gboolean opacity_only,
                                         const gboolean exclude_opacity,
                                         const char *tooltip,
                                         GtkWidget **editor_box_out)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd->masks_props_expanded)
    bd->masks_props_expanded = g_hash_table_new(g_direct_hash, g_direct_equal);
  // "auto-expand selected" (masks panel hamburger -> options): while
  // enabled, expansion is strictly tied to _model_auto_expand_anchor -- the
  // selection if it can be expanded at all, else the last element that could
  // -- not bd->panel_selected_formid directly: selecting something with
  // nothing to expand (a group, or a raster row while "show opacity slider in
  // expanded elements" is off) must leave whichever element was last expanded
  // alone instead of collapsing it, so the panel does not visibly shift just
  // because the user picked such an element next (see
  // _auto_expand_selected_row, which maintains that field and performs the
  // matching in-place enforcement on selection change, since selection itself
  // never triggers a full rebuild). Groups are untouched -- this option only
  // ever affects element rows (is_group is FALSE at both call sites, but kept
  // explicit here for clarity).
  const gboolean auto_exp = _auto_expand_selected();
  const dt_mask_id_t anchor = _model_auto_expand_anchor(bd);
  const gboolean expanded = (!is_group && auto_exp)
                              ? (dt_is_valid_maskid(anchor) && key == anchor)
                              : GPOINTER_TO_INT(g_hash_table_lookup(
                                  bd->masks_props_expanded, GUINT_TO_POINTER(key)));
  if(!is_group && auto_exp && dt_is_valid_maskid(anchor) && key == anchor)
    bd->masks_last_expanded_elem = key;

  GtkWidget *editor_box =
    _build_props_row_editor(module, key, is_group, opacity_only, exclude_opacity);
  gtk_widget_set_visible(editor_box, expanded);

  GtkWidget *btn = dtgtk_togglebutton_new(_paint_param_inout, 0, NULL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(btn), expanded);
  // an expander (chevron), not a mode toggle -- .mask-row-expander in
  // darktable.css carries the shared chevron styling
  dt_gui_add_class(btn, "mask-row-expander");
  dt_gui_add_class(btn, "dt_transparent_background");
  gtk_widget_set_tooltip_text(btn, tooltip);
  g_object_set_data(G_OBJECT(btn), "props-key", GINT_TO_POINTER(key));
  g_object_set_data(G_OBJECT(btn), "props-editor-box", editor_box);
  g_object_set_data(G_OBJECT(btn), "props-is-group", GINT_TO_POINTER(is_group));
  g_signal_connect(G_OBJECT(btn), "toggled", G_CALLBACK(_props_row_toggled), module);

  if(editor_box_out) *editor_box_out = editor_box;
  return btn;
}

// live tooltip for an inline (label/value hidden) opacity slider, shared by
// shape/raster rows and the group header (which drives its own copy of this
// text directly, see _group_opacity_update_tooltip -- its value is not a
// plain _build_props_row_editor slider). Called once after construction to
// set the initial text, then again on every "value-changed" tick.
static void _inline_opacity_tooltip_changed(GtkWidget *w, gpointer user_data)
{
  gchar *tip = g_strdup_printf(_("opacity: %.0f%%"), dt_bauhaus_slider_get(w) * 100.0f);
  gtk_widget_set_tooltip_text(w, tip);
  g_free(tip);
}

// paint an opacity slider's baseline with the standard alpha-channel
// affordance -- a checkerboard (transparent) on the left fading into solid
// white (opaque) on the right, via bauhaus's own checker-gradient mode (see
// dt_bauhaus_slider_set_checker_gradient) -- so the track itself hints at
// what the value means instead of reading as just another plain slider.
// The "brighten up to the handle" fill feedback is redundant on top of a
// track that already fades to white on its own, so it is switched off here.
// Shared by every opacity slider this panel shows (shape/raster/group/
// parametric rows, see call sites).
static void _style_opacity_gradient(GtkWidget *slider)
{
  dt_bauhaus_slider_set_checker_gradient(slider, TRUE);
  dt_bauhaus_slider_set_feedback(slider, FALSE);
}

// style a _build_props_row_editor(..., opacity_only=TRUE) box for inline
// display in a row's own header instead of docked below it (shape/raster
// rows -- mirrors the group header's own opacity slider treatment, see the
// group header build): drop the box's own below-row margins and hide the
// slider's label/value (the tooltip above stands in for them on hover).
// Sizing within the header's free width is handled by the caller instead
// (see _control_column_size_allocate, which drives this box's own
// size-request from the row's width so the row's name column -- not this
// box -- ends up as the one that actually expands to fill). The box (not
// just its one slider child) is what the caller packs --
// reparenting just the slider out of it would orphan the
// dt_masks_props_row_editor_t the box's own destruction is tied to (see
// _build_props_row_editor's "props-editor" data), while the slider's signal
// handler keeps referencing it.
// the actual per-slider half of the styling above: hides the label/value,
// tags it for the "mask-inline-opacity" CSS + the gradient track, and wires
// up the tooltip. Factored out so a caller that already has a bare slider in
// hand (no _build_props_row_editor box around it -- e.g. the pending-row
// opacity slider, which is a plain conf-write control instead) can apply the
// exact same look without going through the box-shaped wrapper below.
static void _inline_opacity_update_label(GtkWidget *label, const float val)
{
  GtkWidget *slider = g_object_get_data(G_OBJECT(label), "opacity-slider");
  const float max_val = slider ? dt_bauhaus_slider_get_hard_max(slider) : 1.0f;
  const float pct = (max_val > 1.5f) ? val : (val * 100.0f);
  gchar *txt = g_strdup_printf("%.0f%%", pct);
  gtk_label_set_text(GTK_LABEL(label), txt);
  g_free(txt);
}

static void _inline_opacity_slider_changed_cb(GtkWidget *slider, gpointer user_data)
{
  GtkWidget *label = user_data;
  if(GTK_IS_LABEL(label))
    _inline_opacity_update_label(label, dt_bauhaus_slider_get(slider));
}

// push a slider's current value into the compact value label that follows it
// (see _make_inline_opacity_value_widget, which tags the slider with its
// label), for the one case the "value-changed" wiring above cannot cover.
//
// bauhaus deliberately does not emit "value-changed" while DT_IN_GUI_UPDATE()
// (see _slider_set_normalized in bauhaus.c), which is the guard every
// programmatic set in this panel is made under -- without it the set would
// re-enter the very handler that made it. The slider itself still repaints
// (the guard only suppresses the signal), so it is exactly the label that
// would be left reading the old number.
static void _refresh_inline_opacity_label(GtkWidget *slider)
{
  if(!slider) return;
  GtkWidget *label = g_object_get_data(G_OBJECT(slider), "opacity-value-label");
  if(GTK_IS_LABEL(label))
    _inline_opacity_update_label(label, dt_bauhaus_slider_get(slider));
}

static void _blend_opacity_slider_changed_cb(GtkWidget *slider, gpointer user_data)
{
  dt_iop_gui_blend_data_t *bd = user_data;
  if(!bd || !bd->blend_opacity_lowop_badge) return;
  const float val = dt_bauhaus_slider_get(slider);
  _update_lowop_badge(bd->blend_opacity_lowop_badge, val / 100.0f, FALSE, FALSE, NULL);
}

// the whisker popup's placement rules, over plain geometry: a square directly
// above or below the anchor (never over it, so the control points it drives
// stay visible while it is open), centred on where the caller asked, and held
// inside both the host panel and the work area. Split out from
// _bauhaus_whisker_popup_rect below so the rules can be tested without a
// display -- getting them wrong is invisible on the machine that wrote them
// and lands the popup somewhere useless on everyone else's.
GdkRectangle _model_whisker_popup_rect(const dt_masks_whisker_geom_t *g)
{
  const gint space_above = g->anchor.y - g->workarea.y;
  const gint space_below =
    (g->workarea.y + g->workarea.height) - (g->anchor.y + g->anchor.height);

  // below by preference: either it fits there, or there is at least as much
  // room below as above
  gint y = (space_below >= g->size + g->gap || space_below >= space_above)
             ? g->anchor.y + g->anchor.height + g->gap
             : g->anchor.y - g->gap - g->size;

  // neither side has the room -- a short screen, or an anchor near an edge.
  // Overlapping the anchor is bad; hanging off the work area is worse, since
  // the popup is then squashed to fit (GDK_ANCHOR_RESIZE_Y, see
  // _window_position in bauhaus.c) and a squashed color wheel stops being a
  // circle.
  y = CLAMP(y, g->workarea.y, g->workarea.y + g->workarea.height - g->size);

  // held to the panel rather than the work area: the panel is where the
  // controls are, and a popup that wandered onto the image would cover the
  // very thing the user is judging the change against
  const gint x = CLAMP(g->center_x - g->size / 2, g->panel_x,
                       g->panel_x + g->panel_w - g->size);

  const GdkRectangle rect = { x, y, g->size, g->size };
  return rect;
}

// where the whisker popup of `anchor` should sit, horizontally centred on
// `center_in_anchor` -- an x offset within the anchor, so callers do not each
// have to work out its position on screen.
//
// The result is in root (screen) coordinates, which is what
// dt_bauhaus_widget_set_popup_position() wants: the popup is pinned *before*
// it is shown, so bauhaus places it once, itself, in whatever coordinate
// space it actually anchors against. Moving the popup window by hand after
// the fact instead -- what this used to do -- only ever agreed with bauhaus's
// own idea of where the popup was on a single monitor with the main window at
// the screen origin; anywhere else the popup jumped back to the primary
// display the moment bauhaus repositioned it (see _window_position in
// bauhaus.c).
static gboolean _bauhaus_whisker_popup_rect(GtkWidget *anchor,
                                            const gint center_in_anchor,
                                            GdkRectangle *rect)
{
  GtkWidget *toplevel = gtk_widget_get_toplevel(anchor);
  GdkWindow *top_gdk =
    gtk_widget_is_toplevel(toplevel) ? gtk_widget_get_window(toplevel) : NULL;
  if(!top_gdk) return FALSE;

  gint top_x, top_y;
  gdk_window_get_origin(top_gdk, &top_x, &top_y);

  gint rx, ry;
  gtk_widget_translate_coordinates(anchor, toplevel, 0, 0, &rx, &ry);
  GtkAllocation alloc;
  gtk_widget_get_allocation(anchor, &alloc);

  dt_masks_whisker_geom_t g = { .anchor = { top_x + rx, top_y + ry,
                                            alloc.width, alloc.height },
                                .center_x = top_x + rx + center_in_anchor,
                                .size = DT_PIXEL_APPLY_DPI(180),
                                .gap = DT_PIXEL_APPLY_DPI(6) };

  GdkMonitor *mon =
    gdk_display_get_monitor_at_window(gdk_window_get_display(top_gdk), top_gdk);
  if(mon) gdk_monitor_get_workarea(mon, &g.workarea);

  g.panel_x = g.workarea.x;
  g.panel_w = g.workarea.width;
  if(dt_ui_panel_ancestor(darktable.gui->ui, DT_UI_PANEL_LEFT, anchor))
  {
    g.panel_x = top_x;
    g.panel_w = dt_ui_panel_get_size(darktable.gui->ui, DT_UI_PANEL_LEFT);
  }
  else if(dt_ui_panel_ancestor(darktable.gui->ui, DT_UI_PANEL_RIGHT, anchor))
  {
    g.panel_w = dt_ui_panel_get_size(darktable.gui->ui, DT_UI_PANEL_RIGHT);
    g.panel_x = top_x + gtk_widget_get_allocated_width(toplevel) - g.panel_w;
  }

  *rect = _model_whisker_popup_rect(&g);
  return TRUE;
}

// pin `slider`'s popup above/below `anchor`, centred on `center_in_anchor`
// (an x offset within the anchor), and open it there
static void _show_bauhaus_whisker_popup(GtkWidget *slider,
                                        GtkWidget *anchor,
                                        const gint center_in_anchor)
{
  GdkRectangle rect;
  if(_bauhaus_whisker_popup_rect(anchor, center_in_anchor, &rect))
    dt_bauhaus_widget_set_popup_position(slider, &rect);
  dt_bauhaus_widget_show_popup(slider);
}

static gboolean _inline_opacity_popup_idle(gpointer user_data)
{
  GtkWidget *evbox = user_data;
  if(!GTK_IS_WIDGET(evbox)) return G_SOURCE_REMOVE;
  GtkWidget *slider = g_object_get_data(G_OBJECT(evbox), "opacity-slider");
  if(!slider || !GTK_IS_WIDGET(slider)) return G_SOURCE_REMOVE;

  GtkAllocation alloc;
  gtk_widget_get_allocation(evbox, &alloc);
  _show_bauhaus_whisker_popup(slider, evbox, alloc.width / 2);
  return G_SOURCE_REMOVE;
}

static gboolean
_inline_opacity_button_press(GtkWidget *w, GdkEventButton *ev, gpointer user_data)
{
  GtkWidget *slider = g_object_get_data(G_OBJECT(w), "opacity-slider");
  if(!slider) return TRUE;

  dt_iop_module_t *module = g_object_get_data(G_OBJECT(w), "module");
  if(module && module->blend_data)
  {
    dt_iop_gui_blend_data_t *bd = module->blend_data;
    bd->masks_skip_group_select_release = TRUE;
    bd->masks_skip_group_select_release_time = ev->time;
    bd->masks_row_click_handled = TRUE;
  }

  if(ev->button == GDK_BUTTON_SECONDARY)
  {
    g_idle_add(_inline_opacity_popup_idle, w);
    return TRUE;
  }
  else if(ev->type == GDK_2BUTTON_PRESS && ev->button == GDK_BUTTON_PRIMARY)
  {
    const float max_val = dt_bauhaus_slider_get_hard_max(slider);
    dt_bauhaus_slider_set(slider, max_val > 1.5f ? 100.0f : 1.0f);
    return TRUE;
  }
  return TRUE;
}

static gboolean
_inline_opacity_button_release(GtkWidget *w, GdkEventButton *ev, gpointer user_data)
{
  return TRUE;
}

static gboolean
_inline_opacity_scroll(GtkWidget *w, GdkEventScroll *ev, gpointer user_data)
{
  GtkWidget *slider = g_object_get_data(G_OBJECT(w), "opacity-slider");
  if(!slider || !gtk_widget_is_sensitive(w)) return FALSE;

  GdkModifierType state = dt_gdk_event_get_state((GdkEvent *)ev);
  const gboolean is_ctrl = (state & GDK_CONTROL_MASK) != 0;
  const gboolean is_shift = (state & GDK_SHIFT_MASK) != 0;

  const float max_val = dt_bauhaus_slider_get_hard_max(slider);
  const gboolean is_100_scale = (max_val > 1.5f);

  float step = is_100_scale ? 5.0f : 0.05f;
  if(is_ctrl)
    step = is_100_scale ? 1.0f : 0.01f;
  else if(is_shift)
    step = is_100_scale ? 10.0f : 0.10f;
  else
  {
    if(dt_conf_get_bool("darkroom/ui/sidebar_scroll_default")) return FALSE;
  }

  double delta_x = 0.0, delta_y = 0.0;
  if(ev->direction == GDK_SCROLL_UP)
    delta_y = -1.0;
  else if(ev->direction == GDK_SCROLL_DOWN)
    delta_y = 1.0;
  else if(ev->direction == GDK_SCROLL_SMOOTH)
    gdk_event_get_scroll_deltas((GdkEvent *)ev, &delta_x, &delta_y);
  else
    return FALSE;

  double delta = (fabs(delta_x) > fabs(delta_y)) ? -delta_x : -delta_y;
  if(delta == 0.0) return FALSE;

  int dir = (delta > 0.0) ? 1 : -1;
  if(dt_conf_get_bool("masks_scroll_down_increases")) dir = -dir;

  const float current = dt_bauhaus_slider_get(slider);
  const float new_val = CLAMP(current + dir * step, 0.0f, is_100_scale ? 100.0f : 1.0f);
  dt_bauhaus_slider_set(slider, new_val);
  return TRUE;
}

static void _inline_opacity_enter(GtkEventControllerMotion *controller,
                                  gdouble x,
                                  gdouble y,
                                  gpointer user_data)
{
  (void)user_data;
  dt_gui_cursor_set(dt_gui_get_widget(controller), "ns-resize", "mask/opacity");
}

static void _inline_opacity_leave(GtkEventControllerMotion *controller,
                                  gpointer user_data)
{
  (void)user_data;
  dt_gui_cursor_set(dt_gui_get_widget(controller), NULL, "mask/opacity");
}

static void _inline_opacity_realize(GtkWidget *widget, gpointer user_data)
{
  (void)user_data;
  dt_gui_cursor_set(widget, "ns-resize", "mask/opacity");
}

static GtkWidget *_make_inline_opacity_value_widget(GtkWidget *slider,
                                                    dt_iop_module_t *module)
{
  GtkWidget *evbox = gtk_event_box_new();
  gtk_event_box_set_visible_window(GTK_EVENT_BOX(evbox), TRUE);
  gtk_widget_add_events(evbox, GDK_SCROLL_MASK | GDK_SMOOTH_SCROLL_MASK
                                 | GDK_BUTTON_PRESS_MASK | GDK_BUTTON_RELEASE_MASK
                                 | GDK_ENTER_NOTIFY_MASK | GDK_LEAVE_NOTIFY_MASK);

  GtkWidget *label = gtk_label_new("");
  gtk_label_set_width_chars(GTK_LABEL(label), 4);
  gtk_label_set_xalign(GTK_LABEL(label), 1.0f);
  if(slider)
  {
    g_object_set_data(G_OBJECT(label), "opacity-slider", slider);
    // the reverse link, for the sets bauhaus makes silently -- see
    // _refresh_inline_opacity_label
    g_object_set_data(G_OBJECT(slider), "opacity-value-label", label);
    _inline_opacity_update_label(label, dt_bauhaus_slider_get(slider));
    g_signal_connect(G_OBJECT(slider), "value-changed",
                     G_CALLBACK(_inline_opacity_slider_changed_cb), label);
  }
  gtk_container_add(GTK_CONTAINER(evbox), label);
  dt_gui_add_class(evbox, "mask-inline-opacity-value");
  gtk_widget_set_tooltip_text(evbox,
                              _("opacity (right-click for precise entry; "
                                "Ctrl/Shift+scroll to adjust; double-click to reset)"));

  g_object_set_data(G_OBJECT(evbox), "opacity-slider", slider);
  if(module) g_object_set_data(G_OBJECT(evbox), "module", module);
  g_signal_connect(G_OBJECT(evbox), "realize", G_CALLBACK(_inline_opacity_realize), NULL);
  dt_gui_connect_motion(evbox, NULL, _inline_opacity_enter, _inline_opacity_leave, NULL);
  g_signal_connect(G_OBJECT(evbox), "button-press-event",
                   G_CALLBACK(_inline_opacity_button_press), NULL);
  g_signal_connect(G_OBJECT(evbox), "button-release-event",
                   G_CALLBACK(_inline_opacity_button_release), NULL);
  g_signal_connect(G_OBJECT(evbox), "scroll-event", G_CALLBACK(_inline_opacity_scroll),
                   NULL);

  return evbox;
}

static GtkWidget *_style_inline_opacity_box(GtkWidget *box, dt_iop_module_t *module)
{
  dt_gui_remove_class(box, "mask-props-row-editor");
  GList *kids = gtk_container_get_children(GTK_CONTAINER(box));
  GtkWidget *slider = kids ? GTK_WIDGET(kids->data) : NULL;
  g_list_free(kids);
  if(!slider) return box;

  dt_bauhaus_widget_hide_label(slider);
  dt_gui_add_class(slider, "mask-inline-opacity");
  _style_opacity_gradient(slider);

  gtk_widget_set_no_show_all(box, TRUE);
  gtk_widget_hide(box);

  GtkWidget *val_widget = _make_inline_opacity_value_widget(slider, module);

  GtkWidget *container = dt_gui_hbox();
  dt_gui_box_add(container, box);
  gtk_box_pack_end(GTK_BOX(container), val_widget, TRUE, TRUE, 0);
  gtk_widget_set_halign(val_widget, GTK_ALIGN_END);
  gtk_widget_set_valign(val_widget, GTK_ALIGN_CENTER);

  return container;
}
// Shared by every row/header kind that has a name column and, optionally, a
// param slot and/or an inline opacity slider (element/raster/parametric rows
// via _make_shape_row, the pending/temporary row via _make_pending_shape_row,
// and group headers via _build_masks_list). By splitting the row into a
// 35% / 65% homogeneous grid (left 35%: handle + name; right 65%: icon/slider + badges),
// GTK resolves all column alignment natively in a single layout pass without
// Pack a row header: <icon/handle> <name> <badges> <opacity> <action_slot (18px)>
// - actions: within-group combine selector for groups, colorpicker for parametric, or
// NULL for shapes
// - trailing_control: inline opacity value widget
// - badge_stack: low-opacity / solo status badges
// - expander_toggle: expand/collapse arrow toggle button (or NULL)
static void _pack_row_header(GtkWidget *row,
                             GtkWidget *handle,
                             GtkWidget *name,
                             GtkWidget *trailing_control,
                             GtkWidget *badge_stack,
                             GtkWidget *actions,
                             GtkWidget *expander_toggle)
{
  GtkWidget *hbox = dt_gui_hbox();

  if(handle) dt_gui_box_add(hbox, handle);
  if(name)
  {
    dt_gui_box_add(hbox, dt_gui_expand(name));
  }

  // 1. Right-most slot: expander arrow (if present)
  if(expander_toggle)
  {
    dt_gui_add_class(expander_toggle, "mask-row-expander");
    dt_gui_add_class(expander_toggle, "dt_transparent_background");
    gtk_widget_set_valign(expander_toggle, GTK_ALIGN_CENTER);
    gtk_box_pack_end(GTK_BOX(hbox), expander_toggle, FALSE, FALSE, 0);
  }
  else if(trailing_control)
  {
    dt_gui_add_class(trailing_control, "mask-row-trailing-no-expander");
  }

  // 2. Opacity label (immediately to the left of the expander arrow)
  if(trailing_control)
  {
    gtk_box_pack_end(GTK_BOX(hbox), trailing_control, FALSE, FALSE, 0);
  }

  // 3. Action icon (within-group combine, picker, etc. - between badges and opacity)
  if(actions)
  {
    dt_gui_add_class(hbox, "mask-row-header-with-action");
    gtk_box_pack_end(GTK_BOX(hbox), actions, FALSE, FALSE, DT_PIXEL_APPLY_DPI(2));
  }
  else
  {
    dt_gui_add_class(hbox, "mask-row-header-no-action");
  }

  // 4. Badges (immediately to the left of action icon / opacity)
  if(badge_stack)
  {
    gtk_box_pack_end(GTK_BOX(hbox), badge_stack, FALSE, FALSE, DT_PIXEL_APPLY_DPI(2));
  }

  dt_gui_box_add(row, dt_gui_expand(hbox));
}

// Recursively walk a stored mask group, recording each *leaf* shape's effective
// hidden state (its own HIDDEN bit OR-ed with that of every enclosing group point)
// keyed by formid. A nested group -- e.g. a shape-set "used from" another module --
// is a single point in the stored group but gets flattened into its individual leaf
// shapes in dev->form_visible (dt_masks_group_ungroup recurses), each with the
// leaf's own formid/state. So the parent group-point's HIDDEN has to be pushed down
// to the leaves, or hiding/soloing such a set would leave its outlines drawn.
static void _collect_effective_hidden(dt_masks_form_t *grp,
                                      const gboolean inherited_hidden,
                                      GHashTable *hidden_by_formid,
                                      const int depth)
{
  if(!grp || !(grp->type & DT_MASKS_GROUP) || depth > DT_MASKS_NESTING_MAX) return;
  gboolean group_bypassed = FALSE;
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    // a bypassed group's members render nothing
    if(dt_masks_point_is_marker(pt))
    {
      group_bypassed = _op_is_bypassed(pt->state);
      continue;
    }
    const gboolean hidden =
      inherited_hidden || (pt->state & (DT_MASKS_STATE_HIDDEN | DT_MASKS_STATE_DISABLE))
      || group_bypassed;
    dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, pt->formid);
    if(form && (form->type & DT_MASKS_GROUP))
      _collect_effective_hidden(form, hidden, hidden_by_formid, depth + 1);
    else
      g_hash_table_insert(hidden_by_formid, GINT_TO_POINTER(pt->formid),
                          GINT_TO_POINTER(hidden ? 1 : 0));
  }
}

// The canvas edit overlay (dev->form_visible) is a flattened *copy* of the stored
// group, built once when edit mode is entered (dt_masks_group_ungroup copies each
// point's state). Toggling hide/solo mutates the stored group only, so the overlay
// would keep drawing the now-hidden shapes' outlines until edit mode is re-entered.
// Mirror the stored HIDDEN bits (flattened through nested groups) onto the matching
// overlay leaves by formid and redraw, so soloing/hiding restricts the visible
// outlines immediately. Also drop the panel selection if the selected shape just
// became hidden -- a hidden shape must not stay highlighted / drawn as selected.
static void _sync_hidden_to_form_visible(dt_iop_module_t *module)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  dt_masks_form_t *vis = darktable.develop ? darktable.develop->form_visible : NULL;
  if(!grp || !vis || !(vis->type & DT_MASKS_GROUP)) return;

  GHashTable *hidden = g_hash_table_new(g_direct_hash, g_direct_equal);
  _collect_effective_hidden(grp, FALSE, hidden, 0);

  for(GList *l = vis->points; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *vp = l->data;
    gpointer val = NULL;
    if(!g_hash_table_lookup_extended(hidden, GINT_TO_POINTER(vp->formid), NULL, &val))
      continue;
    if(GPOINTER_TO_INT(val))
      vp->state |= DT_MASKS_STATE_HIDDEN;
    else
      vp->state &= ~DT_MASKS_STATE_HIDDEN;
  }
  g_hash_table_destroy(hidden);

  // a hidden shape must not remain the selected/edited one (its row would stay
  // highlighted and its canvas outline drawn as selected)
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(bd && dt_is_valid_maskid(bd->panel_selected_formid))
  {
    const dt_masks_point_group_t *selp = _group_point(grp, bd->panel_selected_formid);
    if(selp && (selp->state & DT_MASKS_STATE_HIDDEN))
    {
      bd->panel_selected_formid = INVALID_MASKID;
      if(darktable.develop->form_gui)
        darktable.develop->form_gui->panel_selected_formid = INVALID_MASKID;
    }
  }

  dt_control_queue_redraw_center();
}

// install (once) the CSS that draws a border around the selected mask-list row.
// Done at runtime so it works regardless of the active theme; registered just
// above darktable's own theme provider (USER+1) so the border is not overridden.
// rows tagged "mask-row" may be nested inside cluster expanders, so the row
// lookups below walk the whole subtree under masks_list_box.
// the mask list's rows and headers carry a tag (object data such as "mask-row"
// or "mask-header") and sit nested in expanders and boxes at varying depths:
// these two walk the subtree under `w`, `w` excluded, to reach them.
// _foreach_tagged calls `fn` on each tagged widget and does not look inside it
static void _foreach_tagged(GtkWidget *w,
                            const char *tag,
                            void (*fn)(GtkWidget *tagged, gpointer data),
                            gpointer data)
{
  if(!GTK_IS_CONTAINER(w)) return;
  GList *kids = gtk_container_get_children(GTK_CONTAINER(w));
  for(GList *c = kids; c; c = g_list_next(c))
  {
    GtkWidget *child = c->data;
    if(g_object_get_data(G_OBJECT(child), tag))
      fn(child, data);
    else
      _foreach_tagged(child, tag, fn, data);
  }
  g_list_free(kids);
}

// the first tagged widget `match` accepts, or NULL
static GtkWidget *_find_tagged(GtkWidget *w,
                               const char *tag,
                               gboolean (*match)(GtkWidget *tagged, gconstpointer data),
                               gconstpointer data)
{
  if(!GTK_IS_CONTAINER(w)) return NULL;
  GtkWidget *found = NULL;
  GList *kids = gtk_container_get_children(GTK_CONTAINER(w));
  for(GList *c = kids; c && !found; c = g_list_next(c))
  {
    GtkWidget *child = c->data;
    if(g_object_get_data(G_OBJECT(child), tag) && match(child, data))
      found = child;
    else
      found = _find_tagged(child, tag, match, data);
  }
  g_list_free(kids);
  return found;
}

// the id a tagged widget carries under `key`
static inline dt_mask_id_t _widget_id(GtkWidget *w, const char *key)
{
  return GPOINTER_TO_INT(g_object_get_data(G_OBJECT(w), key));
}

// a group header's cid (see the header build in _build_masks_list)
static inline dt_mask_id_t _header_cid(GtkWidget *header)
{
  return (dt_mask_id_t)GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(header), "group-key"));
}

static gboolean _header_has_cid(GtkWidget *header, gconstpointer cid)
{
  return _header_cid(header) == GPOINTER_TO_INT(cid);
}

static void _paint_row_selection(GtkWidget *row, gpointer sel)
{
  const dt_mask_id_t id = GPOINTER_TO_INT(sel);
  if(dt_is_valid_maskid(id) && _widget_id(row, "formid") == id)
    dt_gui_add_class(row, "mask-list-row-selected");
  else
    dt_gui_remove_class(row, "mask-list-row-selected");
}

static void _apply_row_selection(GtkWidget *w, const dt_mask_id_t sel)
{
  _foreach_tagged(w, "mask-row", _paint_row_selection, GINT_TO_POINTER(sel));
}

// same idea as _apply_row_selection, but for a group's header (tagged "mask-header"
// at construction, with "group-key" holding its cid and "header-widget" the inner
// box the CSS class actually goes on -- see the header build in _build_masks_list).
// `cls` on a group's block, or `root_cls` on the mask's own group's block with
// `cls` on its header row
static void _paint_group_header(GtkWidget *header,
                                const char *cls,
                                const char *root_cls,
                                const gboolean on)
{
  GtkWidget *target = g_object_get_data(G_OBJECT(header), "header-widget");
  if(!target) target = header;
  // the mask's own group lights up its header row and its rail alone: its
  // block is the whole list, and shading it would shade everything in it
  if(g_object_get_data(G_OBJECT(target), "is-root"))
  {
    if(on) dt_gui_add_class(target, root_cls);
    else dt_gui_remove_class(target, root_cls);
    GtkWidget *row = g_object_get_data(G_OBJECT(header), "group-header-widget");
    if(row) target = row;
  }
  if(on)
    dt_gui_add_class(target, cls);
  else
    dt_gui_remove_class(target, cls);
}

static void _paint_group_selection(GtkWidget *header, gpointer sel)
{
  const dt_mask_id_t cid = GPOINTER_TO_INT(sel);
  _paint_group_header(header, "mask-list-row-selected", "mask-root-selected",
                      dt_is_valid_maskid(cid) && _header_cid(header) == cid);
}

static void _apply_group_selection(GtkWidget *w, const dt_mask_id_t sel)
{
  _foreach_tagged(w, "mask-header", _paint_group_selection, GINT_TO_POINTER(sel));
}

// everything holding the selected row or group header is selected by
// implication, up to the list `list`: the groups (their blocks carry
// "group-key") and the rows of nested groups and of the AI object stepped into
// (tagged "mask-row"). Shaded apart from the selection itself
static void _paint_ancestors_selected(GtkWidget *w, GtkWidget *list)
{
  for(GtkWidget *p = gtk_widget_get_parent(w); p && p != list; p = gtk_widget_get_parent(p))
  {
    if(g_object_get_data(G_OBJECT(p), "mask-row"))
      dt_gui_add_class(p, "mask-list-row-implied");
    else if(g_object_get_data(G_OBJECT(p), "group-key"))
    {
      GtkWidget *header = _find_tagged(p, "mask-header", _header_has_cid,
                                       g_object_get_data(G_OBJECT(p), "group-key"));
      if(header)
        _paint_group_header(header, "mask-list-row-implied", "mask-root-implied", TRUE);
    }
  }
}

static void _clear_row_implied(GtkWidget *row, gpointer data)
{
  dt_gui_remove_class(row, "mask-list-row-implied");
}

static void _clear_header_implied(GtkWidget *header, gpointer data)
{
  _paint_group_header(header, "mask-list-row-implied", "mask-root-implied", FALSE);
}

typedef struct _ancestor_walk_t
{
  GtkWidget *list;
  dt_mask_id_t formid, cid;
} _ancestor_walk_t;

static void _paint_row_ancestors(GtkWidget *row, gpointer data)
{
  const _ancestor_walk_t *a = data;
  if(_widget_id(row, "formid") == a->formid) _paint_ancestors_selected(row, a->list);
}

static void _paint_header_ancestors(GtkWidget *header, gpointer data)
{
  const _ancestor_walk_t *a = data;
  if(_header_cid(header) == a->cid) _paint_ancestors_selected(header, a->list);
}

// the ancestors of what is explicitly selected: element `formid`, or else
// group `cid`
static void _apply_ancestor_selection(GtkWidget *list,
                                      const dt_mask_id_t formid,
                                      const dt_mask_id_t cid)
{
  _foreach_tagged(list, "mask-row", _clear_row_implied, NULL);
  _foreach_tagged(list, "mask-header", _clear_header_implied, NULL);
  _ancestor_walk_t a = { list, formid, cid };
  if(dt_is_valid_maskid(formid))
    _foreach_tagged(list, "mask-row", _paint_row_ancestors, &a);
  else if(dt_is_valid_maskid(cid))
    _foreach_tagged(list, "mask-header", _paint_header_ancestors, &a);
}

// what is explicitly selected: the element, or else its group. With an
// element selected its group is selected too, but only by implication
static inline dt_mask_id_t _explicit_group_cid(const dt_iop_gui_blend_data_t *bd)
{
  return dt_is_valid_maskid(bd->panel_selected_formid) ? INVALID_MASKID
                                                       : bd->panel_selected_group_cid;
}

// same idea as _apply_group_selection, but toggles a group header's own solo
// badge (tagged "solo-badge" at construction) instead of the selection class.
// Needed because soloing an element (_toggle_solo_form) only refreshes element
// rows in place (_refresh_all_shape_rows) -- without this, clearing a group
// solo by soloing one of its own elements left that group's badge stuck on
// screen even though bd->solo_group_key had already gone back to 0.
static void _paint_group_solo_badge(GtkWidget *header, gpointer solo_key)
{
  const guint key = GPOINTER_TO_UINT(solo_key);
  GtkWidget *badge = g_object_get_data(G_OBJECT(header), "solo-badge");
  const gboolean bypassed = g_object_get_data(G_OBJECT(header), "group-bypassed") != NULL;
  if(badge)
    _set_solo_status_badge(badge, bypassed ? MASK_SOLO_BADGE_DISABLE
                                  : (key != 0 && key == (guint)_header_cid(header))
                                    ? MASK_SOLO_BADGE_SOLO
                                    : MASK_SOLO_BADGE_NONE);
}

static void _apply_group_solo_badges(GtkWidget *w, const guint solo_key)
{
  _foreach_tagged(w, "mask-header", _paint_group_solo_badge, GUINT_TO_POINTER(solo_key));
}

// same idea as _apply_group_solo_badges, but dims a group header while any
// solo is active -- without this, only element rows dimmed on solo, leaving a
// group's own header fully lit even though every shape inside it was
// solo-suppressed. The group that is itself the solo target (cid ==
// solo_group_key) must stay fully lit, not dim itself; an empty group never
// is one, so it dims.
typedef struct _header_dimming_t
{
  gboolean solo_active;
  guint solo_group_key;
} _header_dimming_t;

static void _dim_group_header(GtkWidget *header, gpointer data)
{
  const _header_dimming_t *d = data;
  const guint cid = (guint)_header_cid(header);
  GtkWidget *target = g_object_get_data(G_OBJECT(header), "group-header-widget");
  GtkWidget *within_sel = g_object_get_data(G_OBJECT(header), "within-sel-widget");
  GtkWidget *opacity_slider = g_object_get_data(G_OBJECT(header), "group-opacity-widget");
  const gboolean suppressed = d->solo_active && cid != d->solo_group_key;
  const gboolean bypassed = g_object_get_data(G_OBJECT(header), "group-bypassed") != NULL;
  if(target)
  {
    if(bypassed)
    {
      GtkWidget *ghandle = g_object_get_data(G_OBJECT(header), "ghandle-widget");
      GtkWidget *lbl_box = g_object_get_data(G_OBJECT(header), "title-label-box");
      GtkWidget *labevt = lbl_box ? gtk_widget_get_parent(lbl_box) : NULL;
      GtkWidget *opacity_inner = opacity_slider ? gtk_widget_get_parent(opacity_slider) : NULL;
      if(ghandle) gtk_widget_set_opacity(ghandle, 0.45);
      if(labevt) gtk_widget_set_opacity(labevt, 0.45);
      if(opacity_inner) gtk_widget_set_opacity(opacity_inner, 0.45);
      if(within_sel) gtk_widget_set_opacity(within_sel, 0.45);
      gtk_widget_set_opacity(target, 1.0);
    }
    else
    {
      gtk_widget_set_opacity(target, suppressed ? 0.45 : 1.0);
    }
  }
  if(within_sel) gtk_widget_set_sensitive(within_sel, !suppressed && !bypassed);
  if(opacity_slider) gtk_widget_set_sensitive(opacity_slider, !suppressed && !bypassed);
  // tag the *soloed* group's whole block so its own cluster headers stay lit
  // (they dim by default under .mask-solo-active -- see darktable.css); a
  // group is being shown in full, so nothing inside it should read as
  // suppressed. Other groups' blocks keep the tag off, so their clusters dim.
  GtkWidget *block = g_object_get_data(G_OBJECT(header), "header-widget");
  if(block)
  {
    if(d->solo_group_key != 0 && cid == d->solo_group_key)
      dt_gui_add_class(block, "mask-group-soloed");
    else
      dt_gui_remove_class(block, "mask-group-soloed");
  }
}

static void _apply_group_header_dimming(GtkWidget *w,
                                        const gboolean solo_active,
                                        const guint solo_group_key)
{
  _header_dimming_t d = { solo_active, solo_group_key };
  _foreach_tagged(w, "mask-header", _dim_group_header, &d);
}

// same tree-walk idea as _apply_group_header_dimming, but toggles one specific
// run's own operator-handle look in place, for "invert output"
// (_group_toggle_output_invert) -- a persistent, checkable state change that
// touches nothing structural (no row added/removed/reordered), so it does not
// need a full teardown+rebuild any more than an element's own INVERSE toggle
// does (see _invert_group_members's switch to _refresh_all_shape_rows).
static void
_apply_group_output_invert_icon(GtkWidget *w, const guint cid, const gboolean inverted)
{
  GtkWidget *header = _find_tagged(w, "mask-header", _header_has_cid, GUINT_TO_POINTER(cid));
  GtkWidget *ghandle = header ? g_object_get_data(G_OBJECT(header), "ghandle-widget") : NULL;
  if(!ghandle) return;
  if(inverted)
    dt_gui_add_class(ghandle, "mask-list-handle-inverted");
  else
    dt_gui_remove_class(ghandle, "mask-list-handle-inverted");
  // a fixed handle is transparent only while not inverted (see _pack_group)
  if(g_object_get_data(G_OBJECT(ghandle), "lead-static"))
  {
    GtkWidget *box = gtk_widget_get_parent(ghandle);
    if(inverted)
    {
      dt_gui_remove_class(ghandle, "mask-lead-static");
      if(box) dt_gui_remove_class(box, "mask-lead-static");
    }
    else
    {
      dt_gui_add_class(ghandle, "mask-lead-static");
      if(box) dt_gui_add_class(box, "mask-lead-static");
    }
  }
  gtk_widget_queue_draw(ghandle);
}

static gboolean _row_has_formid(GtkWidget *row, gconstpointer formid)
{
  return _widget_id(row, "formid") == GPOINTER_TO_INT(formid);
}

static GtkWidget *_find_row_by_formid(GtkWidget *w, const dt_mask_id_t formid)
{
  return _find_tagged(w, "mask-row", _row_has_formid, GINT_TO_POINTER(formid));
}

// every row showing form `formid`, in build order. One mask can reference the
// same shape more than once (a group linking a shape another group defines),
// and each reference gets its own row, so this is a list rather than one
// widget. Owned by the map; the caller does not free it
static GSList *_masks_rows_for_form(dt_iop_gui_blend_data_t *bd,
                                    const dt_mask_id_t formid)
{
  if(!bd || !bd->masks_row_map || !dt_is_valid_maskid(formid)) return NULL;
  return g_hash_table_lookup(bd->masks_row_map, GINT_TO_POINTER(formid));
}

// O(1) shape-row lookup by form id via the masks_row_map index (see blend.h),
// used by every per-formid whole-list lookup instead of a recursive tree walk.
// Falls back to the tree walk if the map is somehow cold, so behaviour is never
// worse than before. Where the same shape is referenced twice this returns the
// first of its rows: anything refreshing one row per reference wants
// _masks_row_for_point instead.
static GtkWidget *_masks_row_widget(dt_iop_gui_blend_data_t *bd,
                                    const dt_mask_id_t formid)
{
  if(!bd || !dt_is_valid_maskid(formid)) return NULL;
  GSList *rows = _masks_rows_for_form(bd, formid);
  GtkWidget *w = rows ? rows->data : NULL;
  if(!w && bd->masks_list_box)
    w = _find_row_by_formid(GTK_WIDGET(bd->masks_list_box), formid);
  return w;
}

// the row built for this exact reference. Each row is tagged with the member
// point it was built from (see _make_shape_row), compared by identity only:
// a pointer left over from a model edit that has not rebuilt the panel yet
// simply matches nothing, so it is never dereferenced. Without this, every
// reference to a twice-used shape resolved to the same row, and whichever one
// the walk visited last decided what that row displayed.
static GtkWidget *_masks_row_for_point(dt_iop_gui_blend_data_t *bd,
                                       const dt_masks_point_group_t *pt)
{
  if(!bd || !pt) return NULL;
  GSList *rows = _masks_rows_for_form(bd, pt->formid);
  for(GSList *r = rows; r; r = r->next)
    if(g_object_get_data(G_OBJECT(r->data), "row-point") == pt)
      return r->data;
  // only one row can be meant when the shape is referenced once (and this is
  // also the cold-map fallback); with several, a point that matches none of
  // them is stale, and painting an arbitrary row from it is what this avoids
  return rows && rows->next ? NULL : _masks_row_widget(bd, pt->formid);
}

// The panel's four DnD payload types. Named here because each one is written
// twice -- once in a GtkTargetEntry table below, once in the hover classifier
// (_dnd_hover_kind) that compares the negotiated target's name back against it.
// A typo in either copy fails silently, as a drag that simply never matches.
#define DND_TARGET_ROW "dt-mask-row"
#define DND_TARGET_GROUP "dt-mask-group"
#define DND_TARGET_CLUSTER "dt-mask-cluster"

// drag-and-drop reordering of rows. Each row's name widget is both a drag
// source and a drop target carrying the form id; dropping reorders grp->points.
static const GtkTargetEntry _mask_row_dnd[] = { { (gchar *)DND_TARGET_ROW,
                                                  GTK_TARGET_SAME_APP, 0 } };

// every badge kind (solo/solo-edit/low-opacity) is now always mapped, part of
// one fixed-size 3-cell stack packed into a row/header's own box (see
// _make_badge_stack), rather than each badge being packed as its own
// separate sibling and shown/hidden with gtk_widget_set_visible -- toggling
// a badge's visibility that way used to change the row's own packed-child
// count, so however many badges happened to be active shifted every other
// header control (name, slider, operator chips) sideways. An "active" flag
// (read by each badge's own draw handler below) now stands in for
// show/hide: inactive means painted as nothing, but the badge's cell in the
// stack -- and everything to its left in the row -- never moves. Clearing
// the tooltip alongside also keeps an inactive (blank) badge from hovering
// up a status message that no longer applies.
static void _set_badge_active(GtkWidget *badge,
                              const gboolean active,
                              const char *tooltip_when_active)
{
  if(!badge) return;
  g_object_set_data(G_OBJECT(badge), "badge-active", GINT_TO_POINTER(active));
  gtk_widget_set_tooltip_text(badge, active ? tooltip_when_active : NULL);
  gtk_widget_queue_draw(badge);
}

static gboolean _badge_is_active(GtkWidget *badge)
{
  return GPOINTER_TO_INT(g_object_get_data(G_OBJECT(badge), "badge-active"));
}

// shared tooltip text for the solo/solo-edit status -- same message whether
// the badge belongs to an element row or a group header, and whether it is
// being set at construction or refreshed in place later.
static const char *_solo_badge_tooltip(void)
{
  return _("soloed: only this is used\n"
           "click here to clear solo");
}

// solo and per-element disable are mutually exclusive, so one row/header can
// only ever be in one of these states at a time and they share a single badge
// slot instead of two (see _make_badge_stack). Solo-edit used to have a state
// here too; it is a mode now (see _soloedit_follow_selection), always applying
// to the selected row, so the selection highlight and the header toggle already
// say what it is doing. (MASK_SOLO_BADGE_* forward-declared above, with the
// other badge helper forward decls.)

static const char *_disable_badge_tooltip(void)
{
  return _("disabled: click to enable");
}

static int _solo_status_badge_get(GtkWidget *badge)
{
  return GPOINTER_TO_INT(g_object_get_data(G_OBJECT(badge), "badge-status"));
}

// set which of the mutually-exclusive states (if any) this badge shows,
// updating its tooltip to match -- MASK_SOLO_BADGE_NONE leaves the cell
// reserved but blank (see _solo_status_badge_draw).
static void _set_solo_status_badge(GtkWidget *badge, const int status)
{
  if(!badge) return;
  g_object_set_data(G_OBJECT(badge), "badge-status", GINT_TO_POINTER(status));
  gtk_widget_set_tooltip_text(
    badge, status == MASK_SOLO_BADGE_SOLO      ? _solo_badge_tooltip()
           : status == MASK_SOLO_BADGE_DISABLE ? _disable_badge_tooltip()
                                               : NULL);
  gtk_widget_queue_draw(badge);
}

// a small badge shown next to a soloed or disabled element/group's label, reusing the same light-bg/dark-fg swap (see .mask-power-solo).
static gboolean _solo_status_badge_draw(GtkWidget *w, cairo_t *cr, gpointer user_data)
{
  const int status = _solo_status_badge_get(w);
  if(status == MASK_SOLO_BADGE_NONE)
    return FALSE; // blank: reserve the cell, paint nothing
  GtkAllocation a;
  gtk_widget_get_allocation(w, &a);
  GtkStyleContext *ctx = gtk_widget_get_style_context(w);
  const GtkStateFlags state = gtk_widget_get_state_flags(w);

  gtk_render_background(ctx, cr, 0, 0, a.width, a.height);
  GdkRGBA c;
  gtk_style_context_get_color(ctx, state, &c);
  cairo_set_source_rgba(cr, c.red, c.green, c.blue, c.alpha);
  const gint pad = DT_PIXEL_APPLY_DPI(1);
  if(status == MASK_SOLO_BADGE_SOLO)
    dtgtk_cairo_paint_eye(cr, pad, pad, a.width - 2 * pad, a.height - 2 * pad, 0, NULL);
  else if(status == MASK_SOLO_BADGE_DISABLE)
    dtgtk_cairo_paint_eye_toggle(cr, pad, pad, a.width - 2 * pad, a.height - 2 * pad,
                                 CPF_ACTIVE, NULL);
  return FALSE;
}

// sized as one cell of the badge stack (see _make_badge_stack). Starts blank
// (MASK_SOLO_BADGE_NONE); callers set the initial status with
// _set_solo_status_badge.
static GtkWidget *_make_solo_status_badge(void)
{
  GtkWidget *badge = gtk_event_box_new();
  gtk_event_box_set_visible_window(GTK_EVENT_BOX(badge), TRUE);
  gtk_widget_set_app_paintable(badge, TRUE);
  gtk_widget_set_size_request(badge, DT_PIXEL_APPLY_DPI(8), DT_PIXEL_APPLY_DPI(8));
  dt_gui_add_class(badge, "mask-power-solo");
  g_signal_connect(G_OBJECT(badge), "draw", G_CALLBACK(_solo_status_badge_draw), NULL);
  return badge;
}

// --- low-opacity warning badge ----------------------------------------------
// Opacity can now go all the way to 0 (see the CLAMP in _props_row_apply): the
// classic manager's 0.05 floor was there only because a near-invisible shape
// used to be indistinguishable from a live one in the flat list. This badge is
// what replaces that floor -- an element or group under the threshold below
// says so on its own row, so "why is this shape doing nothing?" is answerable
// at a glance instead of by opening the properties expander.
#define MASK_LOW_OPACITY_WARN 0.10f

// same reason _solo_badge_draw paints by hand: GtkDarktableIcon never calls
// gtk_render_background, so a plain icon child would leave the CSS-styled badge
// background unpainted. dtgtk_cairo_paint_warning fills even-odd (a solid
// triangle with the exclamation mark knocked out of it), so it needs only the
// foreground colour -- .mask-lowop-warn supplies an amber one.
static gboolean _lowop_badge_draw(GtkWidget *w, cairo_t *cr, gpointer user_data)
{
  if(!_badge_is_active(w)) return FALSE;
  GtkAllocation a;
  gtk_widget_get_allocation(w, &a);
  GtkStyleContext *ctx = gtk_widget_get_style_context(w);
  const GtkStateFlags state = gtk_widget_get_state_flags(w);

  gtk_render_background(ctx, cr, 0, 0, a.width, a.height);
  const gint pad = DT_PIXEL_APPLY_DPI(1);
  // two different reasons share this one slot (see _update_lowop_badge): a
  // no-op element (still at its full/base range, contributes nothing at all)
  // takes precedence over a merely-low-opacity one. Drawn as a plain solid
  // red dot for now -- the switch-off glyph read too close to an open
  // slider handle at this size to be told apart at a glance; a filled disc
  // in a colour nothing else in the row uses is the placeholder until this
  // gets a considered icon.
  if(GPOINTER_TO_INT(g_object_get_data(G_OBJECT(w), "badge-noop")))
  {
    cairo_set_source_rgba(cr, 0.9, 0.15, 0.15, 1.0);
    const double cx = a.width / 2.0, cy = a.height / 2.0;
    const double r = (MIN(a.width, a.height) - 2 * pad) / 2.0;
    cairo_arc(cr, cx, cy, r, 0, 2 * G_PI);
    cairo_fill(cr);
  }
  else
  {
    GdkRGBA c;
    gtk_style_context_get_color(ctx, state, &c);
    cairo_set_source_rgba(cr, c.red, c.green, c.blue, c.alpha);
    dtgtk_cairo_paint_warning(cr, pad, pad, a.width - 2 * pad, a.height - 2 * pad, 0,
                              NULL);
  }
  return FALSE;
}

// starts inactive (blank); _refresh_lowop_badges reveals it in place, no
// list rebuild needed. Not clickable: it reports a value the row's own
// opacity slider owns, so there is nothing for a click to do.
static GtkWidget *_make_lowop_badge(void)
{
  GtkWidget *badge = gtk_event_box_new();
  gtk_event_box_set_visible_window(GTK_EVENT_BOX(badge), TRUE);
  gtk_widget_set_app_paintable(badge, TRUE);
  gtk_widget_set_size_request(badge, DT_PIXEL_APPLY_DPI(8), DT_PIXEL_APPLY_DPI(8));
  dt_gui_add_class(badge, "mask-lowop-warn");
  g_signal_connect(G_OBJECT(badge), "draw", G_CALLBACK(_lowop_badge_draw), NULL);
  return badge;
}

// pack the low-opacity warning badge and the (solo/solo-edit, mutually
// exclusive, see MASK_SOLO_BADGE_*) status badge into one fixed-size
// vertical stack ("stacked squares"), meant to be packed into a row/
// header's own box in place of where the badges used to be packed
// individually. Because every badge is now always mapped and merely blank
// while inactive (see the badge-active/badge-status comments above
// _set_badge_active/_set_solo_status_badge), this stack's own size never
// changes as badges turn on and off, so it reserves a constant slot and
// nothing else in the row shifts. The two squares and the gap between them
// (.mask-badge-stack in darktable.css) add up to no more than the lead
// handle's height, so the stack never makes a header taller than its handle.
static GtkWidget *_make_badge_stack(GtkWidget *lowop_badge, GtkWidget *solo_status_badge)
{
  GtkWidget *stack = dt_gui_vbox();
  gtk_widget_set_valign(stack, GTK_ALIGN_CENTER);
  dt_gui_add_class(stack, "mask-badge-stack");
  if(lowop_badge) dt_gui_box_add(stack, lowop_badge);
  if(solo_status_badge)
    dt_gui_box_add(stack, solo_status_badge);
  return stack;
}

// true iff `sel` is a single-channel parametric form still sitting at its
// full/base range ({0,0,1,1} per channel). A legacy multi-channel form
// (single == 0) has too many
// independent ranges to summarize as one badge, so it is never flagged here.
// `p->channel` indexes the colorspace's channel[] array, NOT the
// blendif_parameters slot directly -- that slot is
// channels[p->channel].param_channels[in_out] (same indirection every other
// reader of blendif_parameters in this file goes through, e.g.
// _blendif_scale_ex/the "changed" tab-label check above). Both input and
// output sub-ranges are checked: per dt_masks_point_parametric_t's own field
// comment, a non-empty output range still refines the mask even while its
// slider is hidden, so it must count too, not just whichever one the UI
// happens to show. Inverted polarity is excluded outright: a full range
// selects everything, but its complement selects nothing, which is a very
// different (and not currently detected/badged) kind of "wrong", not a no-op.
gboolean _parametric_form_is_noop(const dt_masks_form_t *const sel)
{
  if(!sel || !(sel->type & DT_MASKS_PARAMETRIC) || !sel->points) return FALSE;
  const dt_masks_point_parametric_t *const p = sel->points->data;
  if(!p->single || p->invert) return FALSE;
  const dt_iop_gui_blendif_channel_t *const channels =
    dt_develop_blendif_channels_for_csp((int)p->colorspace);
  if(!channels) return FALSE;
  for(int in_out = 0; in_out < 2; in_out++)
  {
    const int ch = channels[p->channel].param_channels[in_out];
    const float *const r = &p->blendif_parameters[4 * ch];
    if(r[0] != 0.0f || r[1] != 0.0f || r[2] != 1.0f || r[3] != 1.0f) return FALSE;
  }
  return TRUE;
}

// activate/deactivate one badge from the opacity it watches, and say the
// actual value in its tooltip -- "low" alone doesn't tell the user whether
// they are looking at 9% or 0%, and those read very differently on canvas.
// `is_noop` (an element only, never a group -- see the callers) takes
// precedence over the opacity check: a parametric channel still at its
// full/base range contributes nothing regardless of its opacity, so opacity
// is not even worth reporting once that's already true.
// Which badge a row should show, from the two values it watches. `is_noop`
// (an element only, never a group -- see the callers) takes precedence: a
// parametric channel still at its full/base range contributes nothing
// regardless of its opacity, so opacity is not even worth reporting once that
// is already true. Split from the widget update below so the rule can be
// tested without a row.
dt_masks_badge_kind_t _model_badge_kind(const float opacity, const gboolean is_noop)
{
  if(is_noop) return DT_MASKS_BADGE_NOOP;
  return (opacity < MASK_LOW_OPACITY_WARN) ? DT_MASKS_BADGE_LOW_OPACITY
                                           : DT_MASKS_BADGE_NONE;
}

// `noop_reason`, when non-NULL, replaces the default parametric wording with a
// reason of the caller's own. The badge itself is deliberately the same one:
// both cases are "this element is in the list but contributes nothing", which
// is what the badge means, and giving a broken raster its own glyph would add a
// second thing to learn for a state the user resolves the same way -- by fixing
// the row or removing it.
static void _update_lowop_badge(GtkWidget *badge,
                                const float opacity,
                                const gboolean is_group,
                                const gboolean is_noop,
                                const char *noop_reason)
{
  if(!badge) return;
  if(is_noop)
  {
    g_object_set_data(G_OBJECT(badge), "badge-noop", GINT_TO_POINTER(1));
    dt_gui_remove_class(badge, "mask-lowop-warn");
    dt_gui_add_class(badge, "mask-noop-warn");
    _set_badge_active(badge, TRUE,
                      noop_reason
                      ? noop_reason
                      : _("this channel's range still covers its entire span, so it "
                          "does not restrict the mask at all yet -- adjust it to "
                          "have an effect"));
    return;
  }
  g_object_set_data(G_OBJECT(badge), "badge-noop", GINT_TO_POINTER(0));
  dt_gui_remove_class(badge, "mask-noop-warn");
  dt_gui_add_class(badge, "mask-lowop-warn");
  const gboolean low = opacity < MASK_LOW_OPACITY_WARN;
  if(!low)
  {
    _set_badge_active(badge, FALSE, NULL);
    return;
  }
  gchar *tip =
    opacity <= 0.0f
      ? g_strdup(is_group ? _("opacity 0%: this group is fully transparent and\n"
                              "contributes nothing to the mask")
                          : _("opacity 0%: this element is fully transparent and\n"
                              "contributes nothing to the mask"))
      : g_strdup_printf(
          is_group ? _("opacity %.0f%%: this group has very little effect on the mask")
                   : _("opacity %.0f%%: this element has very little effect on the mask"),
          opacity * 100.0f);
  _set_badge_active(badge, TRUE, tip);
  g_free(tip);
}

// lightweight: refresh one shape/parametric row's toggle states and opacity
// from `pt`'s current state, in place -- no widget is destroyed or reparented.
// Used by the solo/invert handlers so a single click doesn't tear down and
// rebuild the whole list, which visibly flashes the panel (most noticeably
// the docked parametric editor, which gets parked home and re-docked on every
// rebuild). Now that real mute is gone, DT_MASKS_STATE_HIDDEN only ever
// reflects a transient solo, so a row stays fully interactive (selectable,
// draggable, soloable) regardless -- only its opacity dims.
static void _update_shape_row_state(dt_iop_gui_blend_data_t *bd,
                                    GtkWidget *row_vbox,
                                    const dt_masks_point_group_t *pt)
{
  if(!row_vbox) return;
  const gboolean elem_disabled = (pt->state & DT_MASKS_STATE_DISABLE) != 0;
  const gboolean group_bypassed =
    _member_group_bypassed(_module_mask_group(bd->module), pt->formid);
  const gboolean hidden =
    (pt->state & DT_MASKS_STATE_HIDDEN) || group_bypassed || elem_disabled;
  const gboolean inverse = pt->state & DT_MASKS_STATE_INVERSE;
  const gboolean solo = bd->solo_formid == pt->formid;

  // a soloed element stays highlighted like a hovered row, not just while the
  // mouse is over it -- a distinct class from the transient hover wash so it
  // survives hovering elsewhere in the list (see _clear_hover_classes).
  if(solo)
    dt_gui_add_class(row_vbox, "mask-list-row-solo");
  else
    dt_gui_remove_class(row_vbox, "mask-list-row-solo");

  GtkWidget *row = g_object_get_data(G_OBJECT(row_vbox), "row-hbox");
  GtkWidget *handle = g_object_get_data(G_OBJECT(row_vbox), "handle-widget");
  GtkWidget *name_evbox = g_object_get_data(G_OBJECT(row_vbox), "name-evbox");
  GtkWidget *action_icon = g_object_get_data(G_OBJECT(row_vbox), "action-icon");
  GtkWidget *solo_badge = g_object_get_data(G_OBJECT(row_vbox), "solo-badge");
  GtkWidget *opacity_box = g_object_get_data(G_OBJECT(row_vbox), "opacity-editor-box");
  GtkWidget *expand_toggle = g_object_get_data(G_OBJECT(row_vbox), "expand-toggle");

  if(solo_badge)
    _set_solo_status_badge(solo_badge, elem_disabled ? MASK_SOLO_BADGE_DISABLE
                                       : solo ? MASK_SOLO_BADGE_SOLO
                                              : MASK_SOLO_BADGE_NONE);

  if(handle)
  {
    if(inverse)
      dt_gui_add_class(handle, "mask-list-handle-inverted");
    else
      dt_gui_remove_class(handle, "mask-list-handle-inverted");
    gtk_widget_queue_draw(handle);
  }

  if(elem_disabled)
  {
    if(handle) gtk_widget_set_opacity(handle, 0.45);
    if(name_evbox) gtk_widget_set_opacity(name_evbox, 0.45);
    if(opacity_box) gtk_widget_set_opacity(opacity_box, 0.45);
    if(action_icon) gtk_widget_set_opacity(action_icon, 0.45);
    if(expand_toggle) gtk_widget_set_opacity(expand_toggle, 0.45);
    if(row) gtk_widget_set_opacity(row, 1.0);
  }
  else
  {
    if(handle) gtk_widget_set_opacity(handle, 1.0);
    if(name_evbox) gtk_widget_set_opacity(name_evbox, 1.0);
    if(opacity_box) gtk_widget_set_opacity(opacity_box, 1.0);
    if(action_icon) gtk_widget_set_opacity(action_icon, 1.0);
    if(expand_toggle) gtk_widget_set_opacity(expand_toggle, 1.0);
    const gboolean solo_hidden = (pt->state & DT_MASKS_STATE_HIDDEN) || group_bypassed;
    if(row) gtk_widget_set_opacity(row, solo_hidden ? 0.45 : 1.0);
  }

  // a solo-suppressed element's controls have no visible effect while another
  // element is soloed (this row contributes nothing to the composite) -- gray
  // them out too, not just dim the row. Only the editor boxes (sliders) are
  // made insensitive, never row_vbox/row itself: the row must stay draggable
  // and selectable.
  GtkWidget *param_box = g_object_get_data(G_OBJECT(row_vbox), "param-editor-box");
  if(param_box) gtk_widget_set_sensitive(param_box, !hidden);
  // a parametric row draws its polarity from this same INVERSE bit, but in its
  // own sliders' markers rather than the handle icon (see
  // _update_param_row_display / _param_row_inverted): refresh it here so every
  // caller flipping the bit stays consistent, whether it came through one row
  // (_invert_element) or all of them at once (_invert_group_members).
  dt_masks_param_row_editor_t *param_ed =
    param_box ? g_object_get_data(G_OBJECT(param_box), "param-editor") : NULL;
  if(param_ed) _update_param_row_display(param_ed);
  GtkWidget *props_box = g_object_get_data(G_OBJECT(row_vbox), "props-editor-box");
  if(props_box) gtk_widget_set_sensitive(props_box, !hidden);
  if(opacity_box) gtk_widget_set_sensitive(opacity_box, !hidden);
  // solo-edit only makes sense on a shape that is actually shown -- a
  // solo-suppressed shape contributes nothing to the composite, so nothing to
  // edit. There is no persistent solo-edit widget to grey out any more (it is
  // a menu item built fresh each time the row's actions menu opens, see
  // _build_shape_actions_menu); _clear_soloedit_if_hidden below still drops
  // an already-active solo-edit for the same reason.
}

// defined below (it needs the per-row/-header selection appliers); declared
// here so the in-place refresh can also settle the selection, which solo can
// clear out from under it.
static void _update_row_selection(dt_iop_gui_blend_data_t *bd);

// refresh every shape/parametric row currently in the list from the module's
// mask group, in place (see _update_shape_row_state) -- used by solo, which can
// flip the hidden state of every other row at once.
static void _refresh_all_shape_rows(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!bd || !bd->masks_list_box || !grp) return;
  GList *pts = _mask_points(grp);
  for(GList *l = pts; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = l->data;
    GtkWidget *row_vbox = _masks_row_for_point(bd, pt);
    if(row_vbox) _update_shape_row_state(bd, row_vbox, pt);
  }
  g_list_free(pts);
  // an element solo clears any active group solo (see _toggle_solo_form); make
  // sure a group header's own badge follows suit without a full rebuild
  _apply_group_solo_badges(GTK_WIDGET(bd->masks_list_box), bd->solo_group_key);
  const gboolean solo_active =
    dt_is_valid_maskid(bd->solo_formid) || bd->solo_group_key != 0;
  // same-kind cluster headers dim purely in CSS (#mask-cluster-header-row under
  // .mask-solo-active in darktable.css). They carry no per-row opacity of their
  // own, so a single state class on the list box is all the code needed --
  // clearing solo drops the class and restores them with no extra bookkeeping.
  if(solo_active)
    dt_gui_add_class(GTK_WIDGET(bd->masks_list_box), "mask-solo-active");
  else
    dt_gui_remove_class(GTK_WIDGET(bd->masks_list_box), "mask-solo-active");
  _apply_group_header_dimming(GTK_WIDGET(bd->masks_list_box), solo_active,
                              bd->solo_group_key);
  // callers reach here after _sync_hidden_to_form_visible, which drops the
  // panel selection when the selected element is the one that just became
  // hidden (see its own "a hidden shape must not remain the selected one").
  // _update_shape_row_state does not paint the selection -- only the solo
  // class -- so without this the deselected row kept its selected border until
  // something else forced a rebuild. That gap is exactly what a partial move
  // off the rebuild path costs: the rebuild used to repaint everything, so
  // each state an in-place path replaces it for has to be accounted for
  // explicitly.
  _update_row_selection(bd);
}

// the group's own persistent, multiplicative opacity (see
// dt_masks_point_group_t.group_opacity and the header's own inline slider),
// held by its marker. This is the group's own gain, independent of its
// members' own opacities -- each element's own low-opacity badge already
// accounts for the group it sits in (see _refresh_lowop_badges' effective-
// opacity walk below), so this deliberately does not re-derive anything from
// the members here.
static float _group_own_opacity(dt_masks_form_t *grp, const dt_mask_id_t cid)
{
  const dt_masks_point_group_t *pt = _group_point(grp, cid);
  return pt ? pt->group_opacity : 1.0f; // not found: nothing to warn about
}

// refresh every group header's low-opacity badge, in place. Headers are not in
// bd->masks_row_map (that indexes element rows only), so they are found by the
// same recursive walk _apply_group_solo_badges uses.
static void _paint_group_lowop_badge(GtkWidget *header, gpointer grp)
{
  GtkWidget *badge = g_object_get_data(G_OBJECT(header), "lowop-badge");
  if(badge)
    _update_lowop_badge(badge, _group_own_opacity(grp, _header_cid(header)), TRUE,
                        FALSE, NULL);
}

static void _apply_group_lowop_badges(GtkWidget *w, dt_masks_form_t *grp)
{
  _foreach_tagged(w, "mask-header", _paint_group_lowop_badge, grp);
}

// refresh every low-opacity badge in the panel (element rows and group headers)
// from the current opacities. Cheap and in-place -- no widget is created or
// destroyed -- so it can run on every tick of an opacity drag as well as at the
// end of a list rebuild.
static void _refresh_lowop_badges(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module ? module->blend_data : NULL;
  if(!bd) return;
  if(bd->blend_opacity_lowop_badge && module->blend_params)
  {
    _update_lowop_badge(bd->blend_opacity_lowop_badge,
                        module->blend_params->opacity / 100.0f,
                        FALSE, FALSE, NULL);
  }
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!bd->masks_list_box || !grp) return;
  // an element's overall (effective) opacity is its own value multiplied by
  // the gain of every group around it, nested ones included (see
  // _enclosing_gain)
  GList *pts = _mask_points(grp);
  for(GList *l = pts; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    if(dt_masks_point_is_marker(pt)) continue;
    const float run_group_opacity = _enclosing_gain(grp, pt->formid);
    GtkWidget *row_vbox = _masks_row_for_point(bd, pt);
    if(row_vbox)
    {
      const dt_masks_form_t *const sel =
        dt_masks_get_from_id(darktable.develop, pt->formid);
      // a raster element that cannot reach a mask can never contribute -- the
      // renderer draws it as zero and skips its inversion (see
      // dt_masks_raster_is_unresolved) -- so it earns the same badge as a
      // parametric channel that restricts nothing, with its own reason. The
      // wording covers both ways it gets there: a module that is gone (the row
      // is only removable) and one that is merely switched off or no longer
      // masking (fixable at the source, so do not tell the user to delete it).
      const gboolean raster_broken = dt_masks_raster_is_unresolved(module, NULL, sel);
      _update_lowop_badge(g_object_get_data(G_OBJECT(row_vbox), "lowop-badge"),
                          pt->opacity * run_group_opacity, FALSE,
                          raster_broken || _parametric_form_is_noop(sel),
                          raster_broken
                          ? _("this raster mask has no mask to read: the module it "
                              "came from is switched off, no longer carries a mask, "
                              "or is gone -- so this element selects nothing. Restore "
                              "the source module, or remove this element")
                          : NULL);
    }
  }
  g_list_free(pts);
  _apply_group_lowop_badges(GTK_WIDGET(bd->masks_list_box), grp);
}

void dt_iop_gui_blend_refresh_mask_badges(dt_iop_module_t *module)
{
  _refresh_lowop_badges(module);
}

// keep the canvas's persistent solo highlight (gui->solo_formids) in step with the
// panel's solo / solo-edit state. Unlike the hover sync (panel_hover_formids,
// cleared the moment the mouse moves elsewhere), this must survive the user
// working anywhere else in the panel or canvas, so it lives in its own list,
// recomputed here whenever solo state changes.
static void _sync_solo_canvas_highlight(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_gui_t *gui = darktable.develop->form_gui;
  if(!bd || !gui) return;
  GList *ids = NULL;
  if(dt_is_valid_maskid(bd->solo_formid))
    ids = g_list_prepend(ids, GINT_TO_POINTER(bd->solo_formid));
  if(dt_is_valid_maskid(bd->soloedit_formid))
    ids = g_list_prepend(ids, GINT_TO_POINTER(bd->soloedit_formid));
  if(bd->solo_group_key != 0)
  {
    dt_masks_form_t *grp = _module_mask_group(module);
    GList *members = _selected_group_formids(grp, (dt_mask_id_t)bd->solo_group_key);
    ids = g_list_concat(ids, members);
  }
  g_list_free(gui->solo_formids);
  gui->solo_formids = ids;
  dt_control_queue_redraw_center();
}

// lightweight: update only the selected-row border on the existing rows, without
// rebuilding the list (so it is safe to call from a button-press handler, where a
// full rebuild would destroy the row mid-press and break drag-and-drop). Also
// mirrors the persistent selection onto the canvas (gui->panel_selected_formid,
// drawn when nothing is hovered) and asks for a redraw.
// flexi: mirror the selected shape's group operator into the new-shape operator
// (defined lower, after the operator helpers).
static void _flexi_new_op_follow_selection(dt_iop_gui_blend_data_t *bd);

// solo-edit is a mode rather than a per-element action: while it is on, canvas
// editing follows the list selection, so only the selected shape's nodes and
// handles are grabbable and clicking down the list walks the isolation along
// with it. The state it drives (bd->soloedit_formid) and the way it is applied
// are unchanged -- this only replaces the trigger, so it goes through
// _toggle_soloedit rather than setting the canvas up itself. The header toggle
// is the whole indication that the mode is on; rows carry no solo-edit badge,
// since the isolated element is by definition the selected one.
#define MASKS_SOLOEDIT_MODE_CONF "plugins/darkroom/masks/solo_edit_mode"

static gboolean _soloedit_mode_is_on(void)
{
  return dt_conf_get_bool(MASKS_SOLOEDIT_MODE_CONF);
}

// what the mode would isolate for the current selection: the selected element,
// as long as it is a drawn shape. A parametric channel or a raster mask has no
// canvas geometry of its own to isolate (same carve-out the menu item had), and
// a group selection means "edit the whole group", which is the mode's own off
// state anyway.
dt_mask_id_t _model_soloedit_target(dt_iop_gui_blend_data_t *bd)
{
  if(!_soloedit_mode_is_on() || !dt_is_valid_maskid(bd->panel_selected_formid))
    return INVALID_MASKID;
  // solo and solo-edit stay mutually exclusive: while something is soloed the
  // mode stands down rather than cancelling the solo behind the user's back
  // (_model_toggle_soloedit would clear it). It re-applies on the next
  // selection change once the solo is off.
  if(dt_is_valid_maskid(bd->solo_formid) || bd->solo_group_key != 0)
    return INVALID_MASKID;

  const dt_masks_form_t *form =
    dt_masks_get_from_id(darktable.develop, bd->panel_selected_formid);
  if(!form || (form->type & (DT_MASKS_PARAMETRIC | DT_MASKS_RASTER)))
    return INVALID_MASKID;
  // a path of the AI object stepped into isolates the object: stepping in is
  // for editing its paths side by side, and narrowing to the path would also
  // rebuild the canvas under the press that selected it
  const dt_mask_id_t entered = _entered_object();
  dt_masks_form_t *obj =
    dt_is_valid_maskid(entered) ? dt_masks_get_from_id(darktable.develop, entered) : NULL;
  if(obj && _group_point(obj, bd->panel_selected_formid)) return entered;
  return bd->panel_selected_formid;
}

static void _soloedit_follow_selection(dt_iop_gui_blend_data_t *bd)
{
  // _toggle_soloedit repaints the rows (_refresh_all_shape_rows), whose
  // _update_row_selection lands straight back here in the middle of the toggle.
  // One gesture, one decision
  static gboolean applying = FALSE;
  if(applying) return;

  const dt_mask_id_t want = _model_soloedit_target(bd);
  if(bd->soloedit_formid == want) return;

  // narrowing the canvas edit scope tears down and rebuilds form_visible, which
  // drops the canvas selection (dt_masks_clear_form_gui): put it back
  // afterwards. The panel keeps its own (see dt_iop_gui_masks_select_form)
  const dt_mask_id_t canvas_sel = darktable.develop->mask_form_selected_id;

  applying = TRUE;
  // _toggle_soloedit is a toggle, so "off" means feeding it back the id that is
  // currently isolated
  _toggle_soloedit(bd->module,
                   dt_is_valid_maskid(want) ? want : bd->soloedit_formid);
  applying = FALSE;
  darktable.develop->mask_form_selected_id = canvas_sel;
}

// the solo-edit mode toggle, and the <blending> action bound to it. Global like
// the channel-preview mode next to it, so it survives moving between modules;
// applying it to every module also lets whichever one is showing its panel pick
// the change up without a further click.
static void _soloedit_mode_toggled(GtkGestureSingle *gesture,
                                   gint n_press,
                                   gdouble x,
                                   gdouble y,
                                   dt_iop_module_t *module)
{
  DT_GUARD_GUI_UPDATE();

  const gboolean on = !_soloedit_mode_is_on();
  dt_conf_set_bool(MASKS_SOLOEDIT_MODE_CONF, on);

  DT_ENTER_GUI_UPDATE();
  for(GList *m = darktable.develop ? darktable.develop->iop : NULL;
      m;
      m = g_list_next(m))
  {
    dt_iop_gui_blend_data_t *bd = ((dt_iop_module_t *)m->data)->blend_data;
    if(bd && bd->soloedit_mode)
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->soloedit_mode), on);
  }
  DT_LEAVE_GUI_UPDATE();

  for(GList *m = darktable.develop ? darktable.develop->iop : NULL;
      m;
      m = g_list_next(m))
  {
    dt_iop_gui_blend_data_t *bd = ((dt_iop_module_t *)m->data)->blend_data;
    if(bd && bd->masks_list_box) _soloedit_follow_selection(bd);
  }
}

static void _update_row_selection(dt_iop_gui_blend_data_t *bd)
{
  if(!bd || !bd->masks_list_box) return;
  // every route that clears the group selection ends here
  _select_mask_group_if_none(bd);
  // every group's element rows are nested inside masks_list_box (under their header)
  _apply_row_selection(GTK_WIDGET(bd->masks_list_box), bd->panel_selected_formid);
  _apply_group_selection(GTK_WIDGET(bd->masks_list_box), _explicit_group_cid(bd));
  _apply_ancestor_selection(GTK_WIDGET(bd->masks_list_box), bd->panel_selected_formid,
                            bd->panel_selected_group_cid);
  if(darktable.develop && darktable.develop->form_gui)
    darktable.develop->form_gui->panel_selected_formid = bd->panel_selected_formid;
  _flexi_new_op_follow_selection(bd);
  _flexi_refine_follow_selection(bd);
  _soloedit_follow_selection(bd);
  _props_panel_sync(bd->module, FALSE);
  dt_control_queue_redraw_center();
}

// drop the transient hover wash from every row / cluster header in the list.
static void _clear_hover_classes(GtkWidget *w)
{
  if(!GTK_IS_WIDGET(w)) return;
  dt_gui_remove_class(w, "mask-list-row-hover");
  if(!GTK_IS_CONTAINER(w)) return;
  GList *kids = gtk_container_get_children(GTK_CONTAINER(w));
  for(GList *c = kids; c; c = g_list_next(c)) _clear_hover_classes(c->data);
  g_list_free(kids);
}

// find a group header whose member set includes formid (group headers carry
// their member ids in "group-formids"). Used as the fallback below when a
// shape's own nested row cannot be found directly, so its group header is
// highlighted instead.
static gboolean _header_has_member(GtkWidget *header, gconstpointer formid)
{
  GList *members = g_object_get_data(G_OBJECT(header), "group-formids");
  return g_list_find(members, formid) != NULL;
}

static GtkWidget *_find_collapsed_cluster_header(GtkWidget *w, const dt_mask_id_t formid)
{
  return _find_tagged(w, "group-formids", _header_has_member, GINT_TO_POINTER(formid));
}

// canvas -> list selection sync: when a shape is selected on the canvas (click),
// highlight its row in the flexi mask list. No-op when there is no list
// (classic mode / no masks).
// a path of an AI object is selected through its object, whose row stands for
// it, unless the object is stepped into: its paths then have rows of their own
dt_mask_id_t _model_panel_formid_for(dt_iop_module_t *module, const dt_mask_id_t formid)
{
  if(!dt_is_valid_maskid(formid)) return INVALID_MASKID;
  dt_masks_form_t *mgrp = _module_mask_group(module);
  if(!mgrp || _group_point(mgrp, formid)) return formid;
  // an object in a nested group too
  dt_mask_id_t out = formid;
  GList *pts = _mask_points(mgrp);
  for(GList *l = pts; l; l = g_list_next(l))
  {
    dt_masks_form_t *f =
      dt_masks_get_from_id(darktable.develop, ((dt_masks_point_group_t *)l->data)->formid);
    if(f && (f->type & DT_MASKS_OBJECT) && _group_point(f, formid))
    {
      out = f->formid == _entered_object() ? formid : f->formid;
      break;
    }
  }
  g_list_free(pts);
  return out;
}

void dt_iop_gui_masks_select_form(dt_iop_module_t *module, const dt_mask_id_t formid)
{
  if(!module) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd || !bd->masks_list_box) return;
  const dt_mask_id_t id = _model_panel_formid_for(module, formid);
  // a canvas with nothing selected leaves the panel's selection alone: the
  // canvas drops its own on every rebuild (dt_masks_clear_form_gui), undo's
  // included (libs/history.c _pop_undo), while the user's deselects reach the
  // panel on their own (an empty-canvas click, dt_iop_gui_masks_clear_selection;
  // a delete, _clear_stale_formid_refs)
  if(!dt_is_valid_maskid(id) || bd->panel_selected_formid == id) return;
  bd->panel_selected_formid = id;

  // mirror the group the shape belongs to, same as a list click (_select_form),
  // so a canvas click also highlights/expands the group the shape lives in.
  // Unlike _set_group_target, panel_selected_formid is left set here so the
  // specific row within the group still gets its own highlight too.
  //
  // Only when the canvas actually names a shape, though. A cleared canvas
  // selection says nothing about which *group* is targeted -- the group is the
  // coarser, independent selection level -- and clobbering it here broke group
  // refinement outright: _set_group_target() sets panel_selected_group_cid and
  // then calls dt_masks_set_edit_mode(DT_MASKS_EDIT_FULL), whose canvas rebuild
  // syncs back through here with an invalid formid. That wiped the cid that had
  // just been set, so _flexi_refine_follow_selection() fell through to
  // REFINE_SCOPE_GLOBAL and clicking a group header only ever produced "whole
  // mask refinement". Clearing the group target stays an explicit user action
  // (clicking the selected header again, see _select_group).
  if(dt_is_valid_maskid(id))
  {
    dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, id);
    dt_masks_form_t *grp = _module_mask_group(module);
    bd->panel_selected_group_cid = (form && !(form->type & DT_MASKS_PARAMETRIC) && grp)
                                     ? _group_cid_of_form(grp, id)
                                     : INVALID_MASKID;
  }

  _update_row_selection(bd);
  _auto_expand_selected_row(module, id);
}

// canvas -> list hover sync: transiently highlight the row matching the shape
// under the cursor, or its group's header as a fallback. An invalid id just
// clears the hover wash.
void dt_iop_gui_masks_hover_form(dt_iop_module_t *module, const dt_mask_id_t formid)
{
  if(!module) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd || !bd->masks_list_box) return;
  GtkWidget *box = GTK_WIDGET(bd->masks_list_box);
  _clear_hover_classes(box);
  if(!dt_is_valid_maskid(formid)) return;
  // prefer the shape's own (nested) row; fall back to the group header that contains
  // it
  GtkWidget *target = _masks_row_widget(bd, formid);
  if(!target)
  {
    // a path of an AI object not stepped into: the object's row
    const dt_mask_id_t row_fid = _model_panel_formid_for(module, formid);
    if(row_fid != formid) target = _masks_row_widget(bd, row_fid);
  }
  if(!target) target = _find_collapsed_cluster_header(box, formid);
  if(target) dt_gui_add_class(target, "mask-list-row-hover");
}

void dt_iop_gui_masks_entered_object_changed(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module ? module->blend_data : NULL;
  if(!bd || !bd->masks_list_box) return;
  // the object's row turns into its group, or back (see _make_shape_row)
  _queue_masks_list_rebuild(module);
}

// the panel's way to step the canvas into AI object `id`, or out of the one it
// is in with INVALID_MASKID: the step a double-click on the object and a click
// outside it take on the canvas
static void _step_object(dt_iop_module_t *module, const dt_mask_id_t id)
{
  dt_masks_gui_step_object(module, darktable.develop ? darktable.develop->form_gui : NULL,
                           id, TRUE, dt_is_valid_maskid(id));
}

// the whole panel selection, groups included: unlike the element selection
// dt_iop_gui_masks_select_form mirrors, which leaves the group alone because
// canvas rebuilds pass through it too, this is only ever the user's click
void dt_iop_gui_masks_clear_selection(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module ? module->blend_data : NULL;
  if(!bd || !bd->masks_list_box) return;
  if(!dt_is_valid_maskid(bd->panel_selected_formid)
     && !dt_is_valid_maskid(bd->panel_selected_group_cid))
    return;
  bd->panel_selected_formid = INVALID_MASKID;
  bd->panel_selected_group_cid = INVALID_MASKID;
  _update_row_selection(bd);
  _update_add_target_sensitivity(module);
}

// icon for the parametric mask's "show output" toggle: a chevron pointing down
// when collapsed (only the input slider shown), up when expanded (the output
// slider is shown too). CPF_ACTIVE is set automatically by the togglebutton draw
// code to match the checked state, so one paint function covers both.
static void _paint_param_inout(cairo_t *cr,
                               const gint x,
                               const gint y,
                               const gint w,
                               const gint h,
                               const gint flags,
                               void *data)
{
  const gint dirmask =
    CPF_DIRECTION_UP | CPF_DIRECTION_DOWN | CPF_DIRECTION_LEFT | CPF_DIRECTION_RIGHT;
  const gint dir = (flags & CPF_ACTIVE) ? CPF_DIRECTION_DOWN : CPF_DIRECTION_LEFT;
  dtgtk_cairo_paint_solid_arrow(cr, x, y, w, h, (flags & ~dirmask) | dir, data);
}

// the expand/collapse toggle on a parametric mask's shape row (see
// _make_shape_row): same in/out semantics as legacy multi-channel blendif --
// input and output are independent, additive (AND) refinements on the same
// channel, not alternatives. p->in_out here controls both whether the output
// (and opacity) sliders are shown next to the input one, and the row's
// compact/full layout as one combined state: collapsed is a compact,
// input-only slider; expanded shows input/output/opacity all in full (see
// _update_param_row_visibility). p->in_out never touches p->blendif, so an
// output range set earlier keeps refining the mask even while its slider is
// hidden.
static void _masks_param_inout_toggled(GtkWidget *btn, dt_iop_module_t *module)
{
  if(DT_IN_GUI_UPDATE()) return;
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(btn), "formid"));
  dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, id);
  if(!form || !(form->type & DT_MASKS_PARAMETRIC) || !form->points) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  const uint32_t want = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(btn)) ? 1u : 0u;
  // selects the row too, if it wasn't (never deselects)
  _element_chevron_clicked(module, id, want != 0);
  dt_masks_point_parametric_t *p = form->points->data;

  if(p->in_out == want) return;
  p->in_out = want;
  dt_print(DT_DEBUG_MASKS, "[masks] parametric form %d: show_output=%u", id, want);
  gtk_widget_set_tooltip_text(
    btn, _("show/hide this channel's expanded controls (full input and output sliders)"));
  dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);

  // this row's editor is always present now (see _build_param_row_editor) --
  // just show/hide its output slider and boost box in place, no docking
  GtkWidget *row_vbox = _masks_row_widget(bd, id);
  GtkWidget *editor_box =
    row_vbox ? g_object_get_data(G_OBJECT(row_vbox), "param-editor-box") : NULL;
  dt_masks_param_row_editor_t *ed =
    editor_box ? g_object_get_data(G_OBJECT(editor_box), "param-editor") : NULL;
  if(ed) _update_param_row_display(ed);
}

// "bypassed": a crossed-out eye. dtgtk_cairo_paint_eye_toggle already draws
// exactly that, but only strikes the eye through when CPF_ACTIVE is set, and
// every call site here (the operator menu's pixbuf, the group handle) paints
// with flags 0 -- so force the flag rather than teach each of them about it.
static void _paint_masks_bypass(cairo_t *cr,
                                const gint x,
                                const gint y,
                                const gint w,
                                const gint h,
                                const gint flags,
                                void *data)
{
  dtgtk_cairo_paint_eye_toggle(cr, x, y, w, h, flags | CPF_ACTIVE, data);
}

// input channel icon for parametric editor rows (arrow entering module)
static void _paint_param_input(cairo_t *cr,
                               const gint x,
                               const gint y,
                               const gint w,
                               const gint h,
                               const gint flags,
                               void *data)
{
  cairo_save(cr);
  cairo_translate(cr, x, y);
  cairo_scale(cr, w, h);
  cairo_set_line_width(cr, 0.11);
  cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
  cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);

  // vertical target line / barrier on the right
  cairo_move_to(cr, 0.82, 0.18);
  cairo_line_to(cr, 0.82, 0.82);
  cairo_stroke(cr);

  // arrow shaft pointing into it
  cairo_move_to(cr, 0.15, 0.5);
  cairo_line_to(cr, 0.65, 0.5);
  cairo_stroke(cr);

  // arrowhead pointing right
  cairo_move_to(cr, 0.42, 0.27);
  cairo_line_to(cr, 0.65, 0.5);
  cairo_line_to(cr, 0.42, 0.73);
  cairo_stroke(cr);

  cairo_restore(cr);
}

// output channel icon for parametric editor rows (arrow leaving module)
static void _paint_param_output(cairo_t *cr,
                                const gint x,
                                const gint y,
                                const gint w,
                                const gint h,
                                const gint flags,
                                void *data)
{
  cairo_save(cr);
  cairo_translate(cr, x, y);
  cairo_scale(cr, w, h);
  cairo_set_line_width(cr, 0.11);
  cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
  cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);

  // vertical source line / barrier on the left
  cairo_move_to(cr, 0.18, 0.18);
  cairo_line_to(cr, 0.18, 0.82);
  cairo_stroke(cr);

  // arrow shaft pointing out of it
  cairo_move_to(cr, 0.35, 0.5);
  cairo_line_to(cr, 0.85, 0.5);
  cairo_stroke(cr);

  // arrowhead pointing right
  cairo_move_to(cr, 0.62, 0.27);
  cairo_line_to(cr, 0.85, 0.5);
  cairo_line_to(cr, 0.62, 0.73);
  cairo_stroke(cr);

  cairo_restore(cr);
}

// operator descriptors: the same icons (and order) the mask manager uses
static const struct
{
  dt_masks_state_t state;
  DTGTKCairoPaintIconFunc paint;
  const char *name;
  const char *tooltip;
} _masks_ops[] = {
  { DT_MASKS_STATE_UNION, dtgtk_cairo_paint_masks_union, N_("union"),
    N_("adds group to stack: highest opacity takes precedence") },
  { DT_MASKS_STATE_INTERSECTION, dtgtk_cairo_paint_masks_intersection,
    N_("intersection"),
    N_("restricts accumulated mask strictly to this group's area") },
  { DT_MASKS_STATE_DIFFERENCE, dtgtk_cairo_paint_masks_difference, N_("difference"),
    N_("subtracts group from accumulated mask, cutting holes where it overlaps") },
  { DT_MASKS_STATE_SUM, dtgtk_cairo_paint_masks_sum, N_("sum"),
    N_("adds opacities together, building up density toward solid coverage") },
  { DT_MASKS_STATE_EXCLUSION, dtgtk_cairo_paint_masks_exclusion, N_("exclusion"),
    N_("keeps areas covered by either stack or group alone, clearing overlaps") },
  { DT_MASKS_STATE_MULTIPLY, dtgtk_cairo_paint_masks_multiply, N_("multiply"),
    N_("scales down accumulated stack opacity by group opacity") },
  { DT_MASKS_STATE_OP_SCREEN, dtgtk_cairo_paint_tool_blur, N_("screen"),
    N_("blends group smoothly onto stack without forming harsh boundary seams") },
  // bypass is last so the loops below (which return the first bit they find)
  // keep reporting the group's real combining operator; it is a modifier on
  // top of one of the operators above, not one of them, and every menu that
  // offers "which operator does this group use" skips it -- only the
  // between-group chooser for an existing group offers it, as a toggle.
  { DT_MASKS_STATE_OP_BYPASS, _paint_masks_bypass, N_("disable"), NULL }
};

// a group's operator: how it folds its own members together, in list order
// (masks_revamp_nested_groups.md, Q8). union (max) is the neutral default;
// screen (a+b-ab) smooths feathered overlaps; intersect (min) is the AND;
// difference subtracts every member from the bottom one. Order matches the
// menu. `algebra` is what the menus show after the name: the fold of the mask
// so far `a` with the next member `b` (group.c _combine_masks_*)
static const struct
{
  dt_masks_state_t bit; // 0 = union (no within bit)
  DTGTKCairoPaintIconFunc paint;
  const char *name;
  const char *tooltip;
  const char *algebra;
} _within_modes[] = {
  { 0, dtgtk_cairo_paint_masks_union, N_("union"),
    N_("standard combination: highest opacity wins"), N_("max") },
  { DT_MASKS_STATE_SCREEN, dtgtk_cairo_paint_tool_blur, N_("screen"),
    N_("soft blend: feathered edges merge smoothly without harsh seams"), "a + b - ab" },
  { DT_MASKS_STATE_ISECT, dtgtk_cairo_paint_masks_intersection, N_("intersect"),
    N_("keeps only mutual overlap where all shapes coincide"), N_("min") },
  { DT_MASKS_STATE_WITHIN_MULTIPLY, dtgtk_cairo_paint_masks_multiply, N_("multiply"),
    N_("scales shape opacities against each other"), "a * b" },
  { DT_MASKS_STATE_WITHIN_SUM, dtgtk_cairo_paint_masks_sum, N_("sum"),
    N_("adds shape opacities together, clipped at full opacity"), "a + b" },
  { DT_MASKS_STATE_WITHIN_DIFFERENCE, dtgtk_cairo_paint_masks_difference, N_("difference"),
    N_("subtracts every element from the bottom one, cutting holes where they overlap"),
    "a * (1 - b)" },
  { DT_MASKS_STATE_WITHIN_EXCLUSION, dtgtk_cairo_paint_masks_exclusion, N_("exclusion"),
    N_("keeps areas covered by one element alone, clearing overlaps, "
       "from the bottom element up"), N_("xor") },
};

static DTGTKCairoPaintIconFunc _within_paint(const dt_masks_state_t within)
{
  if(within & DT_MASKS_STATE_ISECT) return dtgtk_cairo_paint_masks_intersection;
  if(within & DT_MASKS_STATE_SCREEN) return dtgtk_cairo_paint_tool_blur;
  if(within & DT_MASKS_STATE_WITHIN_MULTIPLY) return dtgtk_cairo_paint_masks_multiply;
  if(within & DT_MASKS_STATE_WITHIN_SUM) return dtgtk_cairo_paint_masks_sum;
  if(within & DT_MASKS_STATE_WITHIN_DIFFERENCE) return dtgtk_cairo_paint_masks_difference;
  if(within & DT_MASKS_STATE_WITHIN_EXCLUSION) return dtgtk_cairo_paint_masks_exclusion;
  return dtgtk_cairo_paint_masks_union;
}

static const char *_within_name(const dt_masks_state_t within)
{
  if(within & DT_MASKS_STATE_ISECT) return _("intersect");
  if(within & DT_MASKS_STATE_SCREEN) return _("screen");
  if(within & DT_MASKS_STATE_WITHIN_MULTIPLY) return _("multiply");
  if(within & DT_MASKS_STATE_WITHIN_SUM) return _("sum");
  if(within & DT_MASKS_STATE_WITHIN_DIFFERENCE) return _("difference");
  if(within & DT_MASKS_STATE_WITHIN_EXCLUSION) return _("exclusion");
  return _("union");
}

// is this group's between-group operator currently bypassed (group disabled)?
static gboolean _op_is_bypassed(const int state)
{
  return (state & DT_MASKS_STATE_OP_BYPASS) != 0;
}

// the icon for a group's operator chip. A bypassed group shows the bypass glyph
// instead of its own operator's -- that the group is switched off is the more
// important thing to read at a glance, and its name keeps showing the operator
// it will go back to (see _op_name_for_state).
static DTGTKCairoPaintIconFunc _op_paint_for_state(const int state)
{
  for(int i = 0; i < (int)(sizeof(_masks_ops) / sizeof(_masks_ops[0])); i++)
  {
    if(_masks_ops[i].state == DT_MASKS_STATE_OP_BYPASS) continue;
    if(state & _masks_ops[i].state) return _masks_ops[i].paint;
  }
  return dtgtk_cairo_paint_masks_union;
}

static GdkPixbuf *_op_pixbuf(DTGTKCairoPaintIconFunc paint)
{
  const int s = DT_PIXEL_APPLY_DPI(14);
  cairo_surface_t *cst = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, s, s);
  cairo_t *cr = cairo_create(cst);
  dt_gui_gtk_set_source_rgba(cr, DT_GUI_COLOR_BUTTON_FG, 1.0);
  paint(cr, 0, 0, s, s, 0, NULL);
  cairo_destroy(cr);
  guchar *data = cairo_image_surface_get_data(cst);
  dt_draw_cairo_to_gdk_pixbuf(data, s, s);
  GdkPixbuf *shared =
    gdk_pixbuf_new_from_data(data, GDK_COLORSPACE_RGB, TRUE, 8, s, s,
                             cairo_image_surface_get_stride(cst), NULL, NULL);
  GdkPixbuf *owned = gdk_pixbuf_copy(shared); // own the pixels, then drop the surface
  g_object_unref(shared);
  cairo_surface_destroy(cst);
  return owned;
}

// "add group": clicking the button opens an operator chooser; picking an
// operator stages a new (empty) group of that operator on top of the list. The
// button icon mirrors the operator the next shape will use (the pref read back by
// dt_masks_get_default_operator), which the staged group also sets.
// the add-group icon reflects bd->masks_new_group_op -- the operator the *user*
// last chose for the add-group button. It is deliberately NOT tied to the current
// selection (the icon only changes when the user picks an operator here), so it
// reads as "the kind of group the add-group button will create next".
static void _new_shape_op_update(GtkWidget *btn)
{
  // the add-group button is a fixed "+" affordance: clicking it opens the operator
  // chooser. The icon never reflects the selection or the chosen operator. It is a
  // filled circle with a cut-out plus.
  dtgtk_button_set_paint(DTGTK_BUTTON(btn), dtgtk_cairo_paint_plus, 0, NULL);
  gtk_widget_set_tooltip_text(btn, _("add a new group inside the selected group\n"
                                     "(or at the top of the mask, if none is selected)\n"
                                     "click to pick its operator"));
  gtk_widget_queue_draw(btn);
}

// stage an empty group of the chosen operator (defined after the helpers it
// relies on).
static void _stage_new_group(dt_iop_module_t *module, const int within);

// build a labelled "icon + name" menu item with an action target for an operator chooser
static GMenuItem *_op_gmenu_item_target(DTGTKCairoPaintIconFunc paint,
                                        const char *name,
                                        const char *tooltip,
                                        const char *action,
                                        const int target)
{
  GMenuItem *it = g_menu_item_new(_(name), NULL);
  g_menu_item_set_action_and_target_value(it, action, g_variant_new_int32(target));
  if(tooltip)
    g_menu_item_set_attribute(it, "tooltip", "s", _(tooltip));
  GdkPixbuf *pb = _op_pixbuf(paint);
  if(pb)
  {
    g_menu_item_set_icon(it, G_ICON(pb));
    g_object_unref(pb);
  }
  return it;
}

// the menu item for group operator `i` of _within_modes, named with its algebra.
// Every menu that picks a group's operator builds its items here
static GMenuItem *_within_mode_gmenu_item(const int i, const char *action, const int target)
{
  GMenuItem *it = _op_gmenu_item_target(_within_modes[i].paint, _within_modes[i].name,
                                        _within_modes[i].tooltip, action, target);
  // only the words are marked for translation: a formula reads the same in
  // every language, and comes back from _() as it went in
  gchar *label = g_strdup_printf("%s (%s)", _(_within_modes[i].name),
                                 _(_within_modes[i].algebra));
  g_menu_item_set_label(it, label);
  g_free(label);
  return it;
}

// the add-group operator chooser (_new_shape_op_press) is defined later, after the
// empty-group helpers, so it can disable operators that would create two adjacent
// same-operator groups given the current selection.
static gboolean _new_shape_op_press(GtkWidget *w, GdkEventButton *ev, gpointer u);

// operator selector: just the current-operator icon inside a bordered box, so it
// reads as a chooser (the border) rather than a plain icon button. No chevron --
// the border alone is the affordance. The inner icon button is returned via *inner.
static GtkWidget *
_make_op_combo(GtkWidget **inner, DTGTKCairoPaintIconFunc icon, GCallback press)
{
  GtkWidget *box = dt_gui_hbox();
  dt_gui_add_class(box, "mask-op-combo");
  GtkWidget *btn = dtgtk_button_new(icon, 0, NULL);
  gtk_widget_set_valign(btn, GTK_ALIGN_CENTER);
  dt_gui_box_add(box, btn);
  // use g_signal_connect_data directly: the checked g_signal_connect macro only
  // accepts a literal G_CALLBACK(func), not a GCallback variable.
  //
  // `press` is NULL for the base group, whose between-group operator cannot be
  // changed (see the is_base/is_base_group callers) -- it is a plain icon then,
  // with nothing to connect. Connecting NULL unconditionally made GLib log
  // "g_signal_connect_data: assertion 'c_handler != NULL' failed" for every base
  // group on every panel rebuild.
  if(press)
    g_signal_connect_data(G_OBJECT(btn), "button-press-event", press, btn, NULL, 0);
  // the wrapper box carries no_show_all (its visibility is driven by mode_flexi),
  // which also stops show_all from reaching the child: show it explicitly so the
  // box is not empty once it is made visible.
  gtk_widget_show(btn);
  if(inner) *inner = btn;
  return box;
}

// called after any (element or group) solo change: an active solo-edit whose
// element just became hidden by the new solo no longer has anything visible
// on canvas to edit -- it does not make sense to solo-edit something that
// isn't shown, so drop it and restore full-group canvas editability.
// No refresh of its own: both callers (_toggle_solo_form, _toggle_solo_group)
// run _refresh_all_shape_rows immediately afterwards, and solo-edit drives no
// row visual of its own any more -- the header toggle shows the mode, and the
// isolated element is always the selected one.
// state half; TRUE means the caller must restore full-group canvas editing
static gboolean _model_clear_soloedit_if_hidden(dt_iop_module_t *module,
                                                dt_masks_form_t *grp)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!dt_is_valid_maskid(bd->soloedit_formid)) return FALSE;
  const dt_masks_point_group_t *sp = _group_point(grp, bd->soloedit_formid);
  if(sp && (sp->state & DT_MASKS_STATE_HIDDEN))
  {
    bd->soloedit_formid = INVALID_MASKID;
    return TRUE;
  }
  return FALSE;
}


// solo a single element: show only this shape, hiding all the others; toggling
// off clears every hidden bit (solo is the only thing that ever sets
// DT_MASKS_STATE_HIDDEN now that real mute has been removed, so there is
// nothing else to preserve). Triggered from the row's own actions menu (see
// _build_shape_actions_menu) or by clicking its own solo badge to clear it,
// with the soloed state shown by a badge next to the name instead of a
// button icon (see _set_solo_status_badge / _update_shape_row_state).
// Model half of the element solo toggle -- the state machine only. Returns
// what the caller must then do to the canvas edit scope. The three isolation
// modes (solo, solo-edit, and per-element disable) are mutually exclusive by
// construction here rather than by convention at the call sites.
dt_masks_solo_canvas_t _model_toggle_solo_form(dt_iop_module_t *module,
                                               dt_masks_form_t *grp,
                                               const dt_mask_id_t id)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!grp || !_group_point(grp, id)) return DT_MASKS_SOLO_CANVAS_NONE;
  dt_masks_solo_canvas_t canvas = DT_MASKS_SOLO_CANVAS_NONE;

  if(bd->solo_formid == id)
  {
    dt_masks_group_isolate_state(grp, NULL, DT_MASKS_STATE_HIDDEN);
    bd->solo_formid = INVALID_MASKID;
    bd->solo_group_key = 0;
    dt_print(DT_DEBUG_MASKS, "[masks] solo off");
  }
  else
  {
    GList *one = g_list_prepend(NULL, GINT_TO_POINTER(id));
    dt_masks_group_isolate_state(grp, one, DT_MASKS_STATE_HIDDEN);
    g_list_free(one);
    bd->solo_formid = id;
    // only one thing is ever soloed: an element solo cancels any group solo
    bd->solo_group_key = 0;
    dt_print(DT_DEBUG_MASKS, "[masks] solo form %d", id);
    // solo and solo-edit are mutually exclusive (they now share one status
    // badge slot, see _make_badge_stack) -- soloing unconditionally drops
    // any active solo-edit, not just one whose element the new solo happens
    // to hide (see _model_clear_soloedit_if_hidden for that narrower case).
    if(dt_is_valid_maskid(bd->soloedit_formid))
    {
      bd->soloedit_formid = INVALID_MASKID;
      canvas = DT_MASKS_SOLO_CANVAS_FULL;
    }
  }
  if(_model_clear_soloedit_if_hidden(module, grp))
    canvas = DT_MASKS_SOLO_CANVAS_FULL;
  return canvas;
}

static void _toggle_solo_form(dt_iop_module_t *module, const dt_mask_id_t id)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp || !_group_point(grp, id)) return;

  if(_model_toggle_solo_form(module, grp, id) == DT_MASKS_SOLO_CANVAS_FULL)
    dt_masks_set_edit_mode(module, DT_MASKS_EDIT_FULL);
  dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  _sync_hidden_to_form_visible(module);
  // solo can flip every row's hidden state at once; refresh them all in place
  // instead of rebuilding the whole list (see _update_shape_row_state).
  _refresh_all_shape_rows(module);
  _sync_solo_canvas_highlight(module);
  // solo-edit stood down while this was soloed: taking the solo off isolates
  // the selection again (see _model_soloedit_target)
  _soloedit_follow_selection(module->blend_data);
}

// the badge only shows a click-to-clear affordance while it is actually
// showing one of the two states (see the MASK_SOLO_BADGE_* comment above
// _set_solo_status_badge). Since it is now always mapped (blank when neither
// is active, no longer hidden), a press has to check the status explicitly:
// without it, clicking the badge's blank cell would fall straight into
// _toggle_solo_form's else-branch and solo this element on instead of doing
// nothing. A click always means "clear whichever of the two is currently
// showing" -- no need to check which formid is current, just turn it off
// directly.
static gboolean
_solo_badge_form_press(GtkWidget *w, GdkEventButton *e, dt_iop_module_t *module)
{
  if(e->button != GDK_BUTTON_PRIMARY) return FALSE;
  const int status = _solo_status_badge_get(w);
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(w), "formid"));
  if(status == MASK_SOLO_BADGE_SOLO)
    _toggle_solo_form(module, id);
  else if(status == MASK_SOLO_BADGE_DISABLE)
    _toggle_element_disable(module, id);
  else
    return FALSE;
  return TRUE;
}

// flexi: clear any active solo (used wherever the whole selection/visibility
// state is reset, e.g. deleting the last remaining shape).
static void _masks_clear_solo_state(dt_iop_gui_blend_data_t *bd)
{
  bd->solo_formid = INVALID_MASKID;
  bd->solo_group_key = 0;
}

// after removing shapes from the mask the canvas still draws the outlines of the
// now-gone shapes: dt_masks_clear_form_gui clears the gui points but form_visible
// still points at the (stale) edit group, so the overlay is not refreshed until
// the next unrelated action (e.g. adding a shape). Rebuild the on-canvas edit
// overlay from what remains so the ghost outlines clear immediately.
void _refresh_canvas_edit(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(bd && bd->masks_shown != DT_MASKS_EDIT_OFF)
    dt_masks_set_edit_mode(module, bd->masks_shown); // rebuilds form_visible + redraws
  else
    dt_masks_change_form_gui(NULL); // clear the overlay
  dt_control_queue_redraw_center();
}

static void _toggle_element_disable(dt_iop_module_t *module, const dt_mask_id_t id)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp) return;
  dt_masks_point_group_t *pt = _group_point(grp, id);
  if(!pt) return;
  if(pt->state & DT_MASKS_STATE_DISABLE)
    pt->state &= ~DT_MASKS_STATE_DISABLE;
  else
    pt->state |= DT_MASKS_STATE_DISABLE;

  dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
  _sync_hidden_to_form_visible(module);
  // one bit on one point: refresh that row in place instead of tearing the
  // whole list down and rebuilding it. _update_shape_row_state renders every
  // DISABLE-dependent part of a row (status badge, dimmed handle/name/opacity/
  // action icon, insensitive editors) -- a strict superset of what
  // _make_shape_row sets from the same bit at construction time -- so a rebuild
  // has nothing to add here beyond the visible flash and the re-docking of any
  // open parametric editor. Mirrors _invert_element, which is the same
  // one-bit-on-one-point gesture and already took this path.
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  _update_shape_row_state(bd, _masks_row_widget(bd, id), pt);
  _refresh_canvas_edit(module);
}

// core of "reset mask": remove every element and every group but the one the
// mask always has, left empty, with no confirmation and no rebuild of its own
// -- callers that need those (the plain reset button, group-layout preset
// apply) add them on top. Factored out so a preset apply can reuse the exact
// same wipe instead of re-deriving it.
void _masks_reset_mask_core(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(grp && grp->points)
  {
    dt_masks_clear_form_gui(darktable.develop);
    // the points go, the group form stays: removing its elements through
    // dt_masks_form_remove would delete it with its last one, and the
    // module's mask with it (see _detach_group_members)
    g_list_free_full(grp->points, free);
    grp->points = NULL;
    dt_masks_group_ensure_marker(darktable.develop->forms, grp);
    dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
  }
  bd->panel_selected_group_cid = INVALID_MASKID;
  bd->panel_selected_formid = INVALID_MASKID;
  _masks_clear_solo_state(bd);

  // per-element and per-group refinements died with the shapes above (they live
  // in dt_masks_point_group_t), but the module-wide one lives in blend_params
  // and used to survive a reset: the mask was gone while its whole-mask
  // refinement stayed applied, with nothing left in the panel pointing at it.
  if(_refine_global_is_set(module))
  {
    const gboolean had_details = _refine_clear_global(module);
    dt_dev_add_history_item(darktable.develop, module, TRUE);
    if(had_details) // see _refine_clear_global
    {
      dt_dev_reprocess_all(module->dev);
      dt_control_queue_redraw();
    }
  }

  // the refinement panel's own per-formid scratch (which rows are bypassed,
  // which are expanded) is keyed by ids that no longer exist after the wipe
  if(bd->masks_refine_bypassed) g_hash_table_remove_all(bd->masks_refine_bypassed);
  if(bd->masks_refine_expanded) g_hash_table_remove_all(bd->masks_refine_expanded);
  if(bd->masks_props_expanded) g_hash_table_remove_all(bd->masks_props_expanded);
  bd->masks_refine_scope_kind = REFINE_SCOPE_GLOBAL;
  bd->masks_refine_scope_formid = INVALID_MASKID;
  _queue_link_peers_rebuild(module);
}

static void _masks_row_drag_get(GtkWidget *w,
                                GdkDragContext *ctx,
                                GtkSelectionData *sel,
                                guint info,
                                guint time,
                                gpointer user_data)
{
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(w), "formid"));
  gtk_selection_data_set(sel, gtk_selection_data_get_target(sel), 8, (const guchar *)&id,
                         sizeof(id));
}

// the selection a moved element leaves: the element, inside its new group. A
// parametric channel selects no group, as a click on it does not
static void _select_moved_element(dt_iop_module_t *module,
                                  dt_masks_form_t *grp,
                                  const dt_mask_id_t src)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  bd->panel_selected_formid = src;
  dt_masks_form_t *sform = dt_masks_get_from_id(darktable.develop, src);
  bd->panel_selected_group_cid = (sform && !(sform->type & DT_MASKS_PARAMETRIC))
                                   ? _group_cid_of_form(grp, src)
                                   : INVALID_MASKID;
}

// The model half of the element-onto-element drop, split out from the GTK
// handler below so that both the handler and the panel's model test suite
// (src/tests/unittests/masks/test_flexi_model.c) drive the exact same code --
// the gesture's meaning lives here, and nothing reimplements it. Everything
// GTK-shaped stays in the handler, which decodes the drag into this
// function's three plain arguments and commits the result afterwards.
//
// Mutates grp->points and the panel's selection; deliberately does NOT touch
// history, the pipe or the widget tree -- committing is the caller's job.
// `above` means the shape lands visually above the target, i.e. later in the
// bottom-up points list. Returns TRUE if anything moved.
gboolean _model_drop_element_onto_element(dt_iop_module_t *module,
                                          dt_masks_form_t *grp,
                                          const dt_mask_id_t src,
                                          const dt_mask_id_t dst,
                                          const gboolean above)
{
  if(!grp || src == dst) return FALSE;
  return _model_drop_point_onto_point(module, grp, _group_point(grp, src),
                                      _group_point(grp, dst), above);
}

gboolean _model_drop_point_onto_point(dt_iop_module_t *module,
                                      dt_masks_form_t *grp,
                                      const dt_masks_point_group_t *sp,
                                      const dt_masks_point_group_t *dp,
                                      const gboolean above)
{
  if(!grp || !sp || !dp || sp->formid == dp->formid) return FALSE;
  const dt_mask_id_t src = sp->formid;
  dt_masks_form_t *sowner = NULL, *downer = NULL;
  GList *s = _point_node_at(grp, sp, &sowner, 0);
  GList *d = _point_node_at(grp, dp, &downer, 0);
  // elements both: a group is dropped onto through its header. Across nesting
  // levels, only where it may go (see _may_move_into)
  if(!s || !d || _starts_group(s) || _starts_group(d)
     || !_may_move_into(grp, sowner, downer, src))
    return FALSE;

  // sitting among its members is all it takes to join dst's group: the
  // group's settings are its marker's. d->prev is at worst that marker
  dt_masks_point_group_t *spt = s->data;
  sowner->points = g_list_delete_link(sowner->points, s);
  spt->parentid = downer->formid;
  GList *one = g_list_prepend(NULL, spt);
  _insert_points_after(downer, above ? d : d->prev, one);
  g_list_free(one);

  // a moved element should stay selected at the end of the drag -- otherwise
  // it lands in its new spot with no visible indication of what just moved
  _select_moved_element(module, grp, src);
  return TRUE;
}

// a nested group dragged by its header, which lands beside an element row it
// is dropped on as an element would (see the header build in _pack_group)
static gboolean _drags_as_element(GdkDragContext *ctx)
{
  GtkWidget *source = gtk_drag_get_source_widget(ctx);
  return source && g_object_get_data(G_OBJECT(source), "drags-as-element");
}

// the reference the element row of widget `w` shows, while it is still in the
// mask and still shape `id`; else the first reference to `id`. A mask can hold
// a shape twice, and then its form id alone names the wrong row
static const dt_masks_point_group_t *_row_reference(dt_masks_form_t *grp,
                                                    GtkWidget *w,
                                                    const dt_mask_id_t id)
{
  GtkWidget *row = w ? g_object_get_data(G_OBJECT(w), "row-vbox") : NULL;
  const dt_masks_point_group_t *pt = row ? g_object_get_data(G_OBJECT(row), "row-point") : NULL;
  if(pt && _point_node_at(grp, pt, NULL, 0) && pt->formid == id) return pt;
  return _group_point(grp, id);
}

static void _masks_row_drag_received(GtkWidget *w,
                                     GdkDragContext *ctx,
                                     gint x,
                                     gint y,
                                     GtkSelectionData *sel,
                                     guint info,
                                     guint time,
                                     dt_iop_module_t *module)
{
  gboolean ok = FALSE;
  if(gtk_selection_data_get_length(sel) == (gint)sizeof(dt_mask_id_t))
  {
    dt_masks_form_t *grp = _module_mask_group(module);
    dt_mask_id_t src = *(const dt_mask_id_t *)gtk_selection_data_get_data(sel);
    // a nested group dragged by its header carries its group's id
    const gboolean nested = _drags_as_element(ctx);
    if(nested)
    {
      const dt_masks_form_t *sub = _model_nested_group_of(grp, src);
      src = sub ? sub->formid : INVALID_MASKID;
    }
    const dt_mask_id_t dst = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(w), "formid"));
    // rows display bottom-up: dropping on the top half of a row places the
    // shape visually above the target (= later in the list).
    const int h = gtk_widget_get_allocated_height(w);
    const gboolean above = (h > 0 && y < h / 2);
    const dt_masks_point_group_t *sp =
      nested ? _group_point(grp, src) : _row_reference(grp, gtk_drag_get_source_widget(ctx), src);
    ok = _model_drop_point_onto_point(module, grp, sp, _row_reference(grp, w, dst), above);
    if(ok)
    {
      dt_print(DT_DEBUG_MASKS, "[masks] form %d drag-moved near %d", src, dst);
      dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
    }
  }
  gtk_drag_finish(ctx, ok, FALSE, time);
  if(ok) _queue_masks_list_rebuild(module);
}

// group-header drag-and-drop: reorder a whole group, its marker and its
// members, as a unit. A separate target type from the per-shape row DnD so the
// two don't interfere.
static const GtkTargetEntry _mask_group_dnd[] = { { (gchar *)DND_TARGET_GROUP,
                                                    GTK_TARGET_SAME_APP, 0 } };

// a same-kind element cluster's own drag-and-drop: moves every one of its
// members together, as one contiguous block preserving their relative order --
// like a group drag, but for just this same-kind run within a group (see
// _masks_cluster_move). A separate target type from both the per-shape row
// and per-group DnD so all three coexist without interfering.
static const GtkTargetEntry _mask_cluster_dnd[] = { { (gchar *)DND_TARGET_CLUSTER,
                                                      GTK_TARGET_SAME_APP, 0 } };

// a group header accepts every kind of drop: a whole group (reorder), a single
// shape (drop onto a group to move the shape into it), and a whole cluster
// (move every member together). The receive handler routes on the entry info
// below.
enum
{
  DND_MASK_GROUP = 0,
  DND_MASK_ROW = 1,
  DND_MASK_CLUSTER = 2
};
static const GtkTargetEntry _mask_hdr_dnd[] = {
  { (gchar *)DND_TARGET_GROUP, GTK_TARGET_SAME_APP, DND_MASK_GROUP },
  { (gchar *)DND_TARGET_ROW, GTK_TARGET_SAME_APP, DND_MASK_ROW },
  { (gchar *)DND_TARGET_CLUSTER, GTK_TARGET_SAME_APP, DND_MASK_CLUSTER }
};

// The frame a group-level drop target belongs to: the widget the insertion line
// is drawn on, and the rectangle an above/below decision is measured against.
// A group is covered by several drop targets -- its header event box, its block,
// each element row, each cluster header -- and they all resolve to the same
// frame (the group block), which is what makes the group one target rather than
// a stack of them. Falls back to the widget itself for a target that belongs to
// no group.
static GtkWidget *_group_frame_of(GtkWidget *w)
{
  GtkWidget *f = g_object_get_data(G_OBJECT(w), "group-frame");
  if(!f) f = g_object_get_data(G_OBJECT(w), "header-widget");
  return f ? f : w;
}

// Where a group-reorder drop lands relative to the group under the pointer:
// TRUE = above it (later in the bottom-up list), FALSE = below.
//
// Measured against the group's frame, never the sub-widget that happened to
// receive the event -- each of those reports `y` relative to itself, so taking
// its own midpoint gave every sub-widget its own flip point. Dragging up
// through a single group then flipped the indicator repeatedly (below over the
// body's lower half, above over its upper half, below again over the header's
// lower half, above over its top half) instead of switching once at the
// group's middle.
//
// This is the only place the decision is made: the motion handler that draws
// the insertion line and the receive handlers that perform the move all call
// it, so the line and the drop that follows can never disagree.
static gboolean _group_drop_above(GtkWidget *w, const gint y)
{
  GtkWidget *f = _group_frame_of(w);
  gint fx = 0, fy = y;
  // translate_coordinates needs a common ancestor and realized widgets; when it
  // cannot answer, measure against the receiving widget rather than guess
  if(w != f && !gtk_widget_translate_coordinates(w, f, 0, y, &fx, &fy)) f = w, fy = y;
  const int h = gtk_widget_get_allocated_height(f);
  return h > 0 && fy < h / 2;
}

static void _masks_group_drag_get(GtkWidget *w,
                                  GdkDragContext *ctx,
                                  GtkSelectionData *sel,
                                  guint info,
                                  guint time,
                                  gpointer user_data)
{
  // the group's id, its marker's: an empty group has no member to name it by
  const dt_mask_id_t id = _header_cid(w);
  dt_print(DT_DEBUG_MASKS, "[masks dnd] group drag-data-get id=%d", id);
  gtk_selection_data_set(sel, gtk_selection_data_get_target(sel), 8, (const guchar *)&id,
                         sizeof(id));
}

// a cluster's DnD payload is every member's formid, packed as a plain array --
// order does not matter on the receive side (_masks_cluster_move re-derives the
// members' relative order from grp->points itself), so the "hover-formids" list
// already stashed on the header (see _pack_group_elements) is reused as-is.
static void _masks_cluster_drag_get(GtkWidget *w,
                                    GdkDragContext *ctx,
                                    GtkSelectionData *sel,
                                    guint info,
                                    guint time,
                                    gpointer user_data)
{
  GList *ids = g_object_get_data(G_OBJECT(w), "hover-formids");
  const int n = g_list_length(ids);
  dt_mask_id_t *buf = g_malloc_n(MAX(n, 1), sizeof(dt_mask_id_t));
  int i = 0;
  for(GList *l = ids; l; l = g_list_next(l)) buf[i++] = GPOINTER_TO_INT(l->data);
  dt_print(DT_DEBUG_MASKS, "[masks dnd] cluster drag-data-get n=%d", n);
  gtk_selection_data_set(sel, gtk_selection_data_get_target(sel), 8, (const guchar *)buf,
                         n * (int)sizeof(dt_mask_id_t));
  g_free(buf);
}

// unpack a cluster's DnD payload (see _masks_cluster_drag_get) back into a
// GList of formids. Caller frees.
static GList *_cluster_ids_from_selection(GtkSelectionData *sel)
{
  const gint len = gtk_selection_data_get_length(sel);
  if(len <= 0 || len % (gint)sizeof(dt_mask_id_t) != 0) return NULL;
  const dt_mask_id_t *buf = (const dt_mask_id_t *)gtk_selection_data_get_data(sel);
  const int n = len / (int)sizeof(dt_mask_id_t);
  GList *ids = NULL;
  for(int i = 0; i < n; i++) ids = g_list_prepend(ids, GINT_TO_POINTER(buf[i]));
  return ids;
}

// Select the group a drag just moved, once it has landed.
//
// Every element-level drop already does this for the element it moved ("a moved
// element should stay selected at the end of the drag -- otherwise it lands in
// its new spot with no visible indication of what just moved", see
// _masks_row_drag_received). Group-level drops did not, so a moved group landed
// unselected and the selection still pointed at whatever was selected before the
// drag -- which then silently decided where the next "add group" went.
static void _select_moved_group(dt_iop_module_t *module, const dt_mask_id_t cid)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd) return;
  bd->panel_selected_formid = INVALID_MASKID;
  bd->panel_selected_group_cid = cid;
}

static void _masks_group_drag_received(GtkWidget *w,
                                       GdkDragContext *ctx,
                                       gint x,
                                       gint y,
                                       GtkSelectionData *sel,
                                       guint info,
                                       guint time,
                                       dt_iop_module_t *module)
{
  gboolean ok = FALSE;
  if(gtk_selection_data_get_length(sel) == (gint)sizeof(dt_mask_id_t))
  {
    // the dragged group's id, and the id of the group under the pointer
    const dt_mask_id_t src = *(const dt_mask_id_t *)gtk_selection_data_get_data(sel);
    const dt_mask_id_t dst = _header_cid(w);
    if(dt_is_valid_maskid(dst))
    {
      const gboolean above = _group_drop_above(w, y);
      // with shift held, the group goes inside the one under the pointer
      const gboolean inside = dt_modifier_is(dt_key_modifier_state(), GDK_SHIFT_MASK);
      ok = _model_move_group(module, src, dst, above, inside);
      if(inside && !ok && src != dst)
        dt_control_log(_("this group cannot go inside that one: a group never goes"
                         " inside itself, and a list keeps its last group"));
      // a moved group stays selected, exactly as a moved element does (see
      // _masks_row_drag_received's own note): otherwise it lands in its new
      // spot with nothing indicating what just moved, and -- worse -- the
      // selection still points at whatever was selected beforehand, so the next
      // "add group" anchors above *that* group rather than the one just
      // dragged
      if(ok) _select_moved_group(module, src);
      // reordering groups reorders grp->points, i.e. the fold order the pipe
      // actually evaluates -- so it has to be committed exactly like an element
      // reorder does (see _masks_row_drag_received). Without this the model
      // moved but nothing invalidated the pipe, so the canvas kept the pre-drag
      // render until an unrelated event (a zoom) forced a recompute.
      if(ok) dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
    }
    dt_print(DT_DEBUG_MASKS, "[masks dnd] group received src=%d dst=%d ok=%d", src, dst,
             ok);
  }
  gtk_drag_finish(ctx, ok, FALSE, time);
  if(ok) _queue_masks_list_rebuild(module);
}

// a single shape dropped onto a group header: move it into that group, adopting
// the group's operator. It lands at the top of the target run.
// Model half of the element-onto-group-header drop -- same split as
// _model_drop_element_onto_element (see its comment). The element joins the
// target group's run, landing on top of it.
gboolean _model_drop_element_onto_group(dt_iop_module_t *module,
                                        dt_masks_form_t *grp,
                                        const dt_mask_id_t src,
                                        const dt_mask_id_t dst)
{
  if(!grp || src == dst) return FALSE;
  return _model_drop_point_onto_group(module, grp, _group_point(grp, src), dst);
}

gboolean _model_drop_point_onto_group(dt_iop_module_t *module,
                                      dt_masks_form_t *grp,
                                      const dt_masks_point_group_t *sp,
                                      const dt_mask_id_t dst)
{
  if(!grp || !sp || sp->formid == dst) return FALSE;
  const dt_mask_id_t src = sp->formid;
  dt_masks_form_t *sowner = NULL, *downer = NULL;
  GList *s = _point_node_at(grp, sp, &sowner, 0);
  GList *marker = _group_marker_node(_point_node_owner(grp, dst, &downer));
  // as for a drop onto an element
  if(!s || _starts_group(s) || !marker || !_may_move_into(grp, sowner, downer, src))
    return FALSE;
  // already in that group: nothing to do
  if(_group_marker_node(s) == marker) return FALSE;

  dt_masks_point_group_t *spt = s->data;
  sowner->points = g_list_delete_link(sowner->points, s);
  spt->parentid = downer->formid;
  GList *one = g_list_prepend(NULL, spt);
  _insert_points_after(downer, _group_last_node(marker), one);
  g_list_free(one);

  // a moved element should stay selected at the end of the drag
  _select_moved_element(module, grp, src);
  return TRUE;
}

static void _masks_shape_to_group_drop(GtkWidget *w,
                                       GdkDragContext *ctx,
                                       GtkSelectionData *sel,
                                       guint time,
                                       dt_iop_module_t *module)
{
  gboolean ok = FALSE;
  if(gtk_selection_data_get_length(sel) == (gint)sizeof(dt_mask_id_t))
  {
    const dt_mask_id_t src = *(const dt_mask_id_t *)gtk_selection_data_get_data(sel);
    dt_masks_form_t *grp = _module_mask_group(module);
    ok = _model_drop_point_onto_group(module, grp,
                                      _row_reference(grp, gtk_drag_get_source_widget(ctx), src),
                                      _header_cid(w));
    if(ok)
    {
      dt_print(DT_DEBUG_MASKS, "[masks] shape %d moved into group %d", src, _header_cid(w));
      dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
    }
  }
  gtk_drag_finish(ctx, ok, FALSE, time);
  if(ok) _queue_masks_list_rebuild(module);
}

// a whole cluster dropped onto a group header: move every member together,
// adopting the target group's operator, landing on top of its run (mirrors
// _masks_shape_to_group_drop, generalized to the cluster's whole member set).
static void _masks_cluster_to_group_drop(GtkWidget *w,
                                         GdkDragContext *ctx,
                                         GtkSelectionData *sel,
                                         guint time,
                                         dt_iop_module_t *module)
{
  GList *ids = _cluster_ids_from_selection(sel);
  const dt_mask_id_t dst = _header_cid(w);
  const gboolean ok =
    ids && dt_is_valid_maskid(dst) && _masks_cluster_move(module, ids, dst, TRUE, FALSE);
  g_list_free(ids);
  if(ok)
  {
    dt_print(DT_DEBUG_MASKS, "[masks] cluster moved near %d", dst);
    dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  }
  gtk_drag_finish(ctx, ok, FALSE, time);
  if(ok) _queue_masks_list_rebuild(module);
}

// a whole cluster dropped onto an element row: move every member together,
// landing directly above/below that row and adopting its group's operator
// (mirrors _masks_row_drag_received, generalized to the cluster's members).
static void _masks_cluster_row_drop(GtkWidget *w,
                                    GdkDragContext *ctx,
                                    gint y,
                                    GtkSelectionData *sel,
                                    guint time,
                                    dt_iop_module_t *module)
{
  GList *ids = _cluster_ids_from_selection(sel);
  const dt_mask_id_t dst = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(w), "formid"));
  gboolean ok = FALSE;
  if(ids && dt_is_valid_maskid(dst))
  {
    const int h = gtk_widget_get_allocated_height(w);
    const gboolean above = (h > 0 && y < h / 2);
    ok = _masks_cluster_move(module, ids, dst, FALSE, above);
  }
  g_list_free(ids);
  if(ok)
  {
    dt_print(DT_DEBUG_MASKS, "[masks] cluster moved near %d", dst);
    dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  }
  gtk_drag_finish(ctx, ok, FALSE, time);
  if(ok) _queue_masks_list_rebuild(module);
}

// a group header is a drop target for a whole group (reorder), a single shape
// (move into the group), or a whole cluster (move every member together).
// Route on the target entry info.
static void _masks_header_drag_received(GtkWidget *w,
                                        GdkDragContext *ctx,
                                        gint x,
                                        gint y,
                                        GtkSelectionData *sel,
                                        guint info,
                                        guint time,
                                        dt_iop_module_t *module)
{
  dt_print(DT_DEBUG_MASKS, "[masks dnd] header drag-data-received info=%u len=%d", info,
           gtk_selection_data_get_length(sel));
  if(info == DND_MASK_ROW)
    _masks_shape_to_group_drop(w, ctx, sel, time, module);
  else if(info == DND_MASK_CLUSTER)
    _masks_cluster_to_group_drop(w, ctx, sel, time, module);
  else
    _masks_group_drag_received(w, ctx, x, y, sel, info, time, module);
}

// an element row (evbox/row_evbox, tagged with its own group's "group-key" --
// see _make_shape_row) is *also* a drop target for a whole group, not just a
// shape: otherwise only the thin header row would accept such a drop, and
// dragging a group over any of a target group's own elements -- easy to do by
// accident -- would be silently rejected. A shape dropped here still reorders
// precisely next to this row (_masks_row_drag_received), unlike a shape
// dropped on the header (which just lands on top of the group).
static void _element_row_drag_received(GtkWidget *w,
                                       GdkDragContext *ctx,
                                       gint x,
                                       gint y,
                                       GtkSelectionData *sel,
                                       guint info,
                                       guint time,
                                       dt_iop_module_t *module)
{
  if(info == DND_MASK_ROW || (info == DND_MASK_GROUP && _drags_as_element(ctx)))
    _masks_row_drag_received(w, ctx, x, y, sel, info, time, module);
  else if(info == DND_MASK_CLUSTER)
    _masks_cluster_row_drop(w, ctx, y, sel, time, module);
  else
    _masks_group_drag_received(w, ctx, x, y, sel, info, time, module);
}

// point the panel's element selection at id -- the refinement / element-
// properties panels then edit just it -- while still updating the group
// context (add-target sensitivity, edit mode) the same way clicking the
// group's header would. A parametric row's own editor is always visible
// already (see _build_param_row_editor), no separate "open" step, but it
// still becomes the refinement/properties target. Never deselects: this is
// the "acting on an element selects it if it wasn't already selected" variant
// shared by the row's own action controls (see _row_click_press's
// ctrl+click invert); _select_form below adds the toggle-to-deselect behaviour
// for a genuine click on the title.
// same-kind drawn shapes (any kind except parametric/raster, see
// _pack_group_elements) fold into a collapsed expand/collapse cluster once
// there are enough of them -- AI mask (DT_MASKS_OBJECT) rows are not
// exempted from this any more than circle/path/brush rows are. A row inside
// a *collapsed* cluster is still reachable in the widget tree (a GtkRevealer
// keeps its child even while hidden), so toggling its own props expander
// still technically "works", but the user would never see it happen behind
// a collapsed cluster -- walk up from the row and force that cluster open
// too, mirroring _element_cluster_toggle's own reveal/arrow/hash-update
// triplet. A shape outside any cluster has no GtkRevealer ancestor short of
// masks_list_box, so this is a no-op for the common case.
// the expand/collapse chevron of the group header for `gcid`, or NULL: group
// headers are tagged "mask-header", carry their cid under "group-key", and
// hold their own toggle under "group-expand-toggle" (see the header build).
static GtkWidget *_find_group_expand_toggle(GtkWidget *w, const dt_mask_id_t gcid)
{
  GtkWidget *header = _find_tagged(w, "mask-header", _header_has_cid, GINT_TO_POINTER(gcid));
  return header ? g_object_get_data(G_OBJECT(header), "group-expand-toggle") : NULL;
}

static void _reveal_group_header(GtkWidget *w, const dt_mask_id_t gcid)
{
  GtkWidget *toggle = _find_group_expand_toggle(w, gcid);
  if(toggle && !gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)))
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle), TRUE);
}

// open every nested group row, and the group holding each such row, between
// the top of the list and point `id`, so a selection inside a nested group is
// never left folded away. Opening them is not the user choosing which group
// "auto-expand selected" keeps open, hence _group_expand_enforcing
static void _reveal_nesting(dt_iop_module_t *module, const dt_mask_id_t id)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!bd->masks_list_box || !dt_is_valid_maskid(id)) return;
  const gboolean was = _group_expand_enforcing;
  _group_expand_enforcing = TRUE;
  dt_mask_id_t cur = id;
  for(int depth = 0; depth <= DT_MASKS_NESTING_MAX; depth++)
  {
    dt_masks_form_t *owner = NULL;
    if(!_point_node_owner(grp, cur, &owner) || owner == grp) break;
    GtkWidget *row = _masks_row_widget(bd, owner->formid);
    GtkWidget *toggle = row ? g_object_get_data(G_OBJECT(row), "expand-toggle") : NULL;
    if(toggle && !gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)))
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle), TRUE);
    const dt_mask_id_t gcid = _group_cid_of_form(grp, owner->formid);
    if(dt_is_valid_maskid(gcid)) _reveal_group_header(GTK_WIDGET(bd->masks_list_box), gcid);
    cur = owner->formid;
  }
  _group_expand_enforcing = was;
}

// reveals all containers (cluster revealer and enclosing group) for a given row
static void
_reveal_containers_for_row(dt_iop_module_t *module, GtkWidget *row, const dt_mask_id_t id)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(row)
  {
    for(GtkWidget *w = row; w && w != GTK_WIDGET(bd->masks_list_box);
        w = gtk_widget_get_parent(w))
    {
      if(!GTK_IS_REVEALER(w)) continue;
      GtkRevealer *rev = GTK_REVEALER(w);
      if(!gtk_revealer_get_reveal_child(rev))
      {
        gtk_revealer_set_reveal_child(rev, TRUE);
        GtkWidget *arrow = g_object_get_data(G_OBJECT(w), "arrow");
        if(arrow)
        {
          dtgtk_button_set_paint(DTGTK_BUTTON(arrow), dtgtk_cairo_paint_dropdown, 0,
                                 NULL);
          gtk_widget_queue_draw(arrow);
        }
        const guint cid = GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(w), "cluster-key"));
        if(bd->masks_cluster_expanded)
          g_hash_table_insert(bd->masks_cluster_expanded, GUINT_TO_POINTER(cid),
                              GINT_TO_POINTER(TRUE));
      }
    }
  }

  dt_masks_form_t *grp = _module_mask_group(module);
  const dt_mask_id_t gcid = _group_cid_of_form(grp, id);
  if(dt_is_valid_maskid(gcid))
  {
    if(bd->masks_props_expanded)
      g_hash_table_insert(bd->masks_props_expanded, GUINT_TO_POINTER((guint)gcid),
                          GINT_TO_POINTER(TRUE));
    if(bd->masks_list_box) _reveal_group_header(GTK_WIDGET(bd->masks_list_box), gcid);
  }
  _reveal_nesting(module, id);
}

// an element row's own expander, whatever kind of row it is: a drawn shape's
// (or an expandable raster row's) props chevron, or a parametric row's in/out
// chevron. Every row that has one tags its row_vbox with it at build time (see
// _make_shape_row's "expand-toggle"), so this needs no per-kind knowledge and
// returns NULL for a row that has nothing to expand.
static GtkWidget *_row_expand_toggle(dt_iop_gui_blend_data_t *bd, const dt_mask_id_t id)
{
  GtkWidget *row = dt_is_valid_maskid(id) ? _masks_row_widget(bd, id) : NULL;
  return row ? g_object_get_data(G_OBJECT(row), "expand-toggle") : NULL;
}

// expand or collapse one element row, with none of the side effects a real
// click on its chevron carries.
//
// A shape/raster row's expanded state is pure GUI state, so its toggle can
// simply be flipped and _props_row_toggled left to do the rest. A parametric
// row's is not: its chevron is the in/out toggle, and in_out is a *stored*
// field of the form, so _masks_param_inout_toggled commits a mask history item
// every time it moves. Auto-expand is a side effect of merely selecting
// something -- landing an undo step (and a pipe reprocess) on every click
// through a list of parametric elements would be wrong -- so set the field and
// refresh the row's display here instead, with the handler guarded out. The
// value still persists with the next real edit; nothing in the pipe reads it
// (see _masks_param_inout_toggled: in_out never touches p->blendif).
static void _set_row_expanded(dt_iop_module_t *module,
                              const dt_mask_id_t id,
                              const gboolean expanded)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  GtkWidget *row = _masks_row_widget(bd, id);
  GtkWidget *toggle = row ? g_object_get_data(G_OBJECT(row), "expand-toggle") : NULL;
  if(!toggle) return;
  if(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)) == expanded) return;

  // a props chevron carries the editor box it drives, a nested group row's
  // the box of its groups; a parametric row's in/out chevron carries neither
  if(g_object_get_data(G_OBJECT(toggle), "props-editor-box")
     || g_object_get_data(G_OBJECT(toggle), "elem-box"))
  {
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle), expanded);
    return;
  }

  dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, id);
  if(!form || !(form->type & DT_MASKS_PARAMETRIC) || !form->points) return;
  dt_masks_point_parametric_t *p = form->points->data;
  p->in_out = expanded ? 1u : 0u;
  DT_ENTER_GUI_UPDATE(); // keep _masks_param_inout_toggled out of this
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle), expanded);
  DT_LEAVE_GUI_UPDATE();
  GtkWidget *editor_box = g_object_get_data(G_OBJECT(row), "param-editor-box");
  dt_masks_param_row_editor_t *ed =
    editor_box ? g_object_get_data(G_OBJECT(editor_box), "param-editor") : NULL;
  if(ed) _update_param_row_display(ed);
}

// "auto-expand selected" option (masks panel hamburger -> options):
// while enabled, exactly one element row -- the last-selected element that
// actually has an expander -- is ever expanded (see _make_props_row_toggle's
// matching build-time rule, which reads _model_auto_expand_anchor, not
// panel_selected_formid directly). Selection itself only ever goes through
// the row's lightweight in-place updater (_update_row_selection), never a
// full rebuild, so this enforces the same invariant immediately.
//
// Selecting something with nothing to expand -- a group, a raster row while
// "show opacity slider in expanded elements" is off, an invalid/cleared
// selection -- is deliberately a no-op here: `id` is only ever a *candidate*
// replacement, and this bails out before touching anything if `id` itself has
// no expander, so whatever was expanded before stays open instead of
// collapsing just because the user picked such an element next (which would
// otherwise visibly shift the panel for no reason). Every kind that does have
// one is treated alike -- drawn shapes, AI objects (DT_MASKS_OBJECT),
// parametric elements and expandable raster rows -- keyed only by the
// element's own form id, never by kind (see _row_expand_toggle).
static void _auto_expand_selected_row(dt_iop_module_t *module, const dt_mask_id_t id)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!_auto_expand_selected()) return;

  GtkWidget *row = dt_is_valid_maskid(id) ? _masks_row_widget(bd, id) : NULL;
  GtkWidget *toggle = _row_expand_toggle(bd, id);
  // `id` has nothing to expand -- leave the last-expanded element alone. A
  // nested group's chevron shows its groups, not properties: collapsing it for
  // the next selection could fold that selection away (see _reveal_nesting)
  if(!toggle || g_object_get_data(G_OBJECT(toggle), "elem-box")) return;

  if(bd->masks_last_expanded_elem == id
     && gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)))
    return; // already the one that's expanded

  // every _set_row_expanded below is a programmatic enforcement move, not a
  // user click -- guarded by masks_suppress_toggle_select so
  // _props_row_toggled's own "toggling this row's expander also selects it"
  // behavior (meant for a real click) does not fire back into
  // _set_form_target -> _auto_expand_selected_row for the row being
  // collapsed here, which would re-select it and recurse without end (see
  // _props_row_toggled's own comment on this exact failure mode). Not
  // DT_ENTER/LEAVE_GUI_UPDATE: _props_row_toggled bails out entirely on
  // DT_IN_GUI_UPDATE(), which would also block the hash/visibility update
  // these calls are made for in the first place.
  bd->masks_suppress_toggle_select = TRUE;

  // collapse only the previously-expanded element (there is at most one, by
  // construction) -- not "every other row": a row that was somehow left
  // expanded outside this mechanism is none of this function's business, only
  // the one it itself opened last.
  if(dt_is_valid_maskid(bd->masks_last_expanded_elem)
     && bd->masks_last_expanded_elem != id)
    _set_row_expanded(module, bd->masks_last_expanded_elem, FALSE);

  _reveal_containers_for_row(module, row, id);
  _set_row_expanded(module, id, TRUE);
  bd->masks_last_expanded_elem = id;

  bd->masks_suppress_toggle_select = FALSE;
}

// collapse whichever group "auto-expand selected" last opened, unless it is
// `keep_cid`. Split out so _group_expand_toggled can reuse it when a real
// click on a chevron takes over as the one open group.
static void _collapse_auto_expanded_group(dt_iop_module_t *module,
                                          const dt_mask_id_t keep_cid)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  const dt_mask_id_t prev = bd->masks_last_expanded_group;
  if(!dt_is_valid_maskid(prev) || prev == keep_cid || !bd->masks_list_box) return;
  // a group holding `keep_cid` in a nested group stays open to show it
  GList *prev_members = _selected_group_formids(_module_mask_group(module), prev);
  const gboolean holds_keep = _members_hold(prev_members, keep_cid);
  g_list_free(prev_members);
  if(holds_keep) return;
  GtkWidget *toggle =
    _find_group_expand_toggle(GTK_WIDGET(bd->masks_list_box), prev);
  if(toggle && gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)))
  {
    const gboolean was = _group_expand_enforcing;
    _group_expand_enforcing = TRUE;
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle), FALSE);
    _group_expand_enforcing = was;
  }
}

// the group half of "auto-expand selected": selecting a group opens it and
// closes the one opened before, exactly as _auto_expand_selected_row does for
// elements one level down. Selecting an *element* comes through here too --
// _set_form_target sets the group half of the selection first (see
// _set_group_target) -- so picking an element opens both its group and itself.
//
// A group's chevron reveals its members, not a properties panel, so this
// tracks its own "at most one open" state (bd->masks_last_expanded_group)
// rather than sharing the element one. Clearing the selection is a no-op, same
// candidate-only rule the element half follows: whatever is open stays open
// instead of the panel collapsing to nothing.
static void _auto_expand_selected_group(dt_iop_module_t *module,
                                        const dt_mask_id_t cid)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!_auto_expand_selected()) return;

  // a chevron click the user just made collapses the group *and* selects it,
  // which lands here -- honour the click instead of undoing it. One-shot,
  // cleared by whichever selection arrives first so it can never go stale.
  const dt_mask_id_t collapsed_by_click = bd->masks_group_collapse_click;
  bd->masks_group_collapse_click = INVALID_MASKID;

  if(!dt_is_valid_maskid(cid) || collapsed_by_click == cid) return;
  if(!bd->masks_list_box) return;

  GtkWidget *toggle = _find_group_expand_toggle(GTK_WIDGET(bd->masks_list_box), cid);
  if(!toggle) return;
  if(bd->masks_last_expanded_group == cid
     && gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)))
    return; // already the one that's open

  _collapse_auto_expanded_group(module, cid);
  if(!gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)))
  {
    _group_expand_enforcing = TRUE;
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle), TRUE);
    _group_expand_enforcing = FALSE;
  }
  _reveal_nesting(module, cid);
  bd->masks_last_expanded_group = cid;
}

// auto_expand=FALSE skips _auto_expand_selected_row's own collapse-the-
// previous-row/expand-this-one side effect -- used by a right-click (see
// _row_click_press): that side effect can change row heights (the
// previously-expanded row collapsing shifts every row below it), and doing
// that synchronously before the actions menu pops up leaves the menu
// anchored to where the header *was*, not where it ends up once the reflow
// lands -- a right-click is opening a context menu for whichever shape it
// landed on, not asking to see that shape's controls, so there is no reason
// for it to reflow the list at all. Every other caller still wants the
// normal interactive behaviour and goes through the plain _set_form_target
// wrapper below (auto_expand=TRUE).
// The panel's selection state machine, split out from the widget/canvas
// effects its callers apply, so the contract below can be tested without a
// display (see src/tests/unittests/masks/test_flexi_model.c). These decide
// *what* a click selects; _set_form_target / _set_group_target then apply it.
//
// Selection has two levels -- a group, and an element within it -- and the
// contract is that every reachable state is one click away:
//
//   click a group       -> that group selected
//   click it again      -> the mask's own group selected (it cannot be
//                          deselected: one group is always selected)
//   click the group of the selected element
//                       -> that group selected, the element dropped
//   click an element    -> that element selected, inside its group
//   click it again      -> the element is dropped, its GROUP stays selected
//   click elsewhere     -> that thing selected
//
// The element-deselect case is the subtle one: stepping out of an element
// lands in its group rather than clearing both levels at once. Clearing both
// made re-selecting the group after deselecting an element take two clicks.
dt_masks_panel_sel_t _model_click_element(const dt_iop_gui_blend_data_t *bd,
                                          dt_masks_form_t *grp,
                                          const dt_mask_id_t id)
{
  dt_masks_panel_sel_t s = { INVALID_MASKID, INVALID_MASKID };
  // an element's group is selected alongside it either way -- what differs is
  // whether the element itself survives the click
  s.group_cid = _group_cid_of_form(grp, id);
  if(bd->panel_selected_formid != id) s.formid = id;
  return s;
}

dt_masks_panel_sel_t _model_click_group(const dt_iop_gui_blend_data_t *bd,
                                        const dt_mask_id_t cid)
{
  dt_masks_panel_sel_t s = { INVALID_MASKID, INVALID_MASKID };
  // only a group selected by itself deselects: one selected because it holds
  // the selected element is selected in the element's place
  const gboolean deselect = dt_is_valid_maskid(bd->panel_selected_group_cid)
                            && bd->panel_selected_group_cid == cid
                            && !dt_is_valid_maskid(bd->panel_selected_formid);
  // deselecting lands on the mask's own group, which therefore stays selected
  // when clicked again
  s.group_cid = deselect ? _mask_group_cid(bd->module) : cid;
  return s;
}

static void _set_form_target_ext(dt_iop_module_t *module,
                                 const dt_mask_id_t id,
                                 const gboolean auto_expand)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  _set_group_target_ext(module, _group_cid_of_form(grp, id), id);
  bd->panel_selected_formid = id;
  if(dt_is_valid_maskid(id))
  {
    GtkWidget *row = _masks_row_widget(bd, id);
    _reveal_containers_for_row(module, row, id);
  }
  _update_row_selection(bd);
  if(auto_expand) _auto_expand_selected_row(module, id);
}

static void _set_form_target(dt_iop_module_t *module, const dt_mask_id_t id)
{
  _set_form_target_ext(module, id, TRUE);
}

// a real click on an element row's chevron, props or parametric in/out alike
// (shift+click on the row drives the same chevron). It also selects the row,
// but that selection must not run auto-expand: the option would re-open a row
// the click just collapsed, before the handler got to act on it.
static void _element_chevron_clicked(dt_iop_module_t *module,
                                     const dt_mask_id_t id,
                                     const gboolean expanded)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  const dt_masks_chevron_click_t c =
    _model_element_chevron_click(bd, id, expanded, _auto_expand_selected());
  if(dt_is_valid_maskid(c.collapse))
  {
    // programmatic: must not read as a click on that row's own chevron (see
    // _props_row_toggled for the recursion this flag prevents)
    const gboolean was = bd->masks_suppress_toggle_select;
    bd->masks_suppress_toggle_select = TRUE;
    _set_row_expanded(module, c.collapse, FALSE);
    bd->masks_suppress_toggle_select = was;
  }
  bd->masks_last_expanded_elem = c.last_expanded;
  if(bd->panel_selected_formid != id) _set_form_target_ext(module, id, FALSE);
  // the release that toggled the chevron goes on to the row's own click
  // surface, and _row_click_release would toggle an already selected row off
  bd->masks_skip_group_select_release = TRUE;
  bd->masks_skip_group_select_release_time = gtk_get_current_event_time();
}

// select an element by clicking its title. Clicking the title of an already-
// selected element deselects it (toggle), mirroring the group header's own
// title-click behaviour (see _select_group) -- only the title click routes
// through here, see _set_form_target above for the select-only variant.
//
// Deselecting an element drops back to its GROUP being selected, rather than to
// nothing. Selection has two levels (group, then element within it), so element
// and group are not two independent things a click has to clear separately:
// stepping out of an element lands you in its group, and clicking that group
// again clears everything. Every state is therefore one click away -- which it
// was not when this cleared both at once, since re-selecting the group after
// deselecting an element then took a second click.
static void _select_form(dt_iop_module_t *module, const dt_mask_id_t id)
{
  const dt_masks_panel_sel_t s =
    _model_click_element(module->blend_data, _module_mask_group(module), id);
  if(dt_is_valid_maskid(s.formid)) _set_form_target(module, s.formid);
  else _set_group_target(module, s.group_cid);
}

// stable type-label prefix for a form ("circle", "Lightness", ...), recomputed
// from form->type (and, for parametric, its channel) every time rather than
// parsed out of form->name -- so it survives repeated renames, see
// _row_click_press / _rename_commit.
static const char *_form_type_prefix(const dt_masks_form_t *form)
{
  if(form->type & DT_MASKS_PARAMETRIC) return dt_masks_parametric_type_label(form);
  return _kind_name(_form_kind(form), FALSE);
}

// form->name with its stable type prefix stripped -- the row's own icon (and,
// for parametric, the channel badge) already say what kind this is, so
// repeating it in the text would be redundant. Used both for the row label and
// to prefill the rename entry with only the editable part. Caller frees.
gchar *_form_display_name(const dt_masks_form_t *form)
{
  const char *prefix = _form_type_prefix(form);
  if(!prefix) prefix = "";
  const size_t plen = strlen(prefix);
  const char *rest = form->name;
  if(g_str_has_prefix(form->name, prefix)
     && (form->name[plen] == ' ' || form->name[plen] == '\0'))
  {
    rest = form->name + plen;
    while(*rest == ' ') rest++;
  }
  // a raster element named by its type alone shows its source's current name,
  // so renaming the source renames it; a name of its own stops that
  if((form->type & DT_MASKS_RASTER) && !*rest)
  {
    const dt_iop_module_t *src = dt_masks_raster_source(form);
    if(src) return dt_history_item_get_name(src);
    const dt_masks_point_raster_t *p = form->points ? form->points->data : NULL;
    return g_strdup(p ? p->source : "");
  }
  return g_strdup(rest);
}

gboolean _model_rename_form(dt_masks_form_t *form, const char *txt)
{
  if(!form || !txt) return FALSE;
  char name[sizeof(form->name)];
  // the entry edits only the part after the type prefix, so a rename replaces
  // the auto-assigned "#<id>" without ever dropping the "what is this"
  // indication. Emptying a raster element's name makes it follow its source
  if(*txt)
    g_snprintf(name, sizeof(name), "%s %s", _form_type_prefix(form), txt);
  else if(form->type & DT_MASKS_RASTER)
    g_strlcpy(name, _form_type_prefix(form), sizeof(name));
  else
    return FALSE;
  if(!strcmp(name, form->name)) return FALSE;
  g_strlcpy(form->name, name, sizeof(form->name));
  return TRUE;
}

// raster elements used to store their source's name as it was then; one still
// showing it follows the source from now on, like a new one
void _model_raster_names_follow_sources(dt_masks_form_t *grp)
{
  GList *pts = _mask_points(grp);
  for(GList *l = pts; l; l = g_list_next(l))
  {
    dt_masks_form_t *f =
      dt_masks_get_from_id(darktable.develop, ((dt_masks_point_group_t *)l->data)->formid);
    const dt_iop_module_t *src = f && (f->type & DT_MASKS_RASTER) ? dt_masks_raster_source(f) : NULL;
    if(!src) continue;
    gchar *label = dt_history_item_get_name(src);
    gchar *stored = g_strdup_printf("%s %s", _form_type_prefix(f), label);
    if(!strcmp(f->name, stored)) g_strlcpy(f->name, _form_type_prefix(f), sizeof(f->name));
    g_free(stored);
    g_free(label);
  }
  g_list_free(pts);
}

// a shared element shows the chain and offers "unlink". A raster element never
// does: it has nothing shared to edit, and unlinking it would change nothing
gboolean _model_form_is_linked(const dt_masks_form_t *form)
{
  if(!form || (form->type & DT_MASKS_RASTER)) return FALSE;
  GList *users = _model_form_users(form->formid);
  const gboolean linked = !g_list_shorter_than(users, 2);
  g_list_free(users);
  return linked;
}

// the tooltip of a linked element's chain icon, naming the other modules that
// use it. `uses_here` is how often this module's own mask references the form
// (see _model_form_uses_in_mask): a shape can be linked without any other
// module being involved. NULL when it is neither shared nor repeated
static gchar *_linked_tooltip(const dt_iop_module_t *module,
                              const dt_mask_id_t fid,
                              const dt_masks_form_t *form,
                              const int uses_here)
{
  GList *users = _model_form_users(fid);
  GString *names = g_string_new(NULL);
  for(GList *l = users; l; l = g_list_next(l))
  {
    if(l->data == module) continue;
    gchar *name = dt_history_item_get_name(l->data);
    if(names->len) g_string_append(names, ", ");
    g_string_append(names, name);
    g_free(name);
  }
  g_list_free(users);
  if(!names->len)
  {
    g_string_free(names, TRUE);
    // no other module uses it, but this mask can reference it more than once
    if(uses_here > 1)
      return g_strdup_printf(
        _("used %d times in this mask\n"
          "this shape is shared: editing it changes every row it appears in,\n"
          "while each row keeps its own opacity, operator and refinements"),
        uses_here);
    return NULL;
  }
  // parametric channels are only ever copied, but edits made before that rule
  // (duplicated instances used to share them) can still hold a shared one
  const char *format =
    (form->type & DT_MASKS_OBJECT)
      ? _("linked with %s\n"
          "this AI object is shared: editing it changes it in every module it is linked with,\n"
          "while its opacity, operator and refinements stay separate\n"
          "right-click and pick \"unlink\" to give this module its own copy")
    : (form->type & DT_MASKS_PARAMETRIC)
      ? _("linked with %s\n"
          "this channel is shared: changing its range changes it in every module it is"
          " linked with,\n"
          "while its opacity, operator and refinements stay separate\n"
          "right-click and pick \"unlink\" to give this module its own copy")
      : _("linked with %s\n"
          "this shape is shared: editing it changes it in every module it is linked with,\n"
          "while its opacity, operator and refinements stay separate\n"
          "right-click and pick \"unlink\" to give this module its own copy");
  gchar *tip = g_strdup_printf(format, names->str);
  g_string_free(names, TRUE);
  return tip;
}

// the chain of a linked element: a chip like the solo badge (see .mask-row-linked)
static gboolean _linked_badge_draw(GtkWidget *w, cairo_t *cr, gpointer user_data)
{
  GtkAllocation a;
  gtk_widget_get_allocation(w, &a);
  GtkStyleContext *ctx = gtk_widget_get_style_context(w);
  gtk_render_background(ctx, cr, 0, 0, a.width, a.height);
  GdkRGBA c;
  gtk_style_context_get_color(ctx, gtk_widget_get_state_flags(w), &c);
  cairo_set_source_rgba(cr, c.red, c.green, c.blue, c.alpha);
  const gint pad = DT_PIXEL_APPLY_DPI(1);
  dtgtk_cairo_paint_link(cr, pad, pad, a.width - 2 * pad, a.height - 2 * pad, 0, NULL);
  return TRUE;
}

static GtkWidget *_make_linked_badge(const char *tooltip)
{
  GtkWidget *badge = gtk_event_box_new();
  gtk_event_box_set_visible_window(GTK_EVENT_BOX(badge), TRUE);
  gtk_widget_set_app_paintable(badge, TRUE);
  gtk_widget_set_size_request(badge, DT_PIXEL_APPLY_DPI(11), DT_PIXEL_APPLY_DPI(11));
  gtk_widget_set_valign(badge, GTK_ALIGN_CENTER);
  dt_gui_add_class(badge, "mask-row-linked");
  gtk_widget_set_tooltip_text(badge, tooltip);
  g_signal_connect(G_OBJECT(badge), "draw", G_CALLBACK(_linked_badge_draw), NULL);
  return badge;
}

static void _rename_commit(GtkWidget *entry, dt_iop_module_t *module)
{
  if(g_object_get_data(G_OBJECT(entry), "done")) return; // guard double commit
  g_object_set_data(G_OBJECT(entry), "done", GINT_TO_POINTER(1));
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(entry), "formid"));
  dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, id);
  gchar *txt = g_strdup(gtk_entry_get_text(GTK_ENTRY(entry)));
  if(txt) g_strstrip(txt);
  if(_model_rename_form(form, txt))
  {
    dt_print(DT_DEBUG_MASKS, "[masks] form %d renamed to '%s'", id, form->name);
    dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  }
  g_free(txt);
  // deferred, not a direct _build_masks_list() call: this runs from inside
  // "activate"/"focus-out-event" dispatch on `entry`, a descendant of the
  // very row a synchronous rebuild would destroy out from under GTK's own
  // event propagation -- same crash class _queue_masks_list_rebuild's own
  // comment describes for DnD receive handlers, reachable here too since
  // e.g. a queued motion event for a nearby widget (a parametric row's own
  // gradient slider, see _blendop_blendif_enter_cb) can still be dispatched
  // against a now-dangling pointer after a synchronous teardown.
  _queue_masks_list_rebuild(module);
}

static gboolean _rename_focus_out(GtkWidget *entry, GdkEvent *e, dt_iop_module_t *module)
{
  _rename_commit(entry, module);
  return FALSE;
}

// _group_rename_commit/_group_rename_focus_out/_group_rename_key_press are
// defined further down (after dt_masks_empty_group_t's own definition, which
// they need the full type of, not just the forward declaration in scope
// here) -- see there for what they do.
static void _group_rename_commit(GtkWidget *entry, dt_iop_module_t *module);
static gboolean
_group_rename_focus_out(GtkWidget *entry, GdkEvent *e, dt_iop_module_t *module);
static gboolean
_group_rename_key_press(GtkWidget *entry, GdkEventKey *e, dt_iop_module_t *module);

// start inline rename on `evbox` (swap its label for an entry): shared by a
// ctrl+click on the row's name (_row_click_press) and the "rename" entry
// in the row's actions menu (_build_shape_actions_menu).
static void
_start_rename_element(GtkWidget *evbox, dt_iop_module_t *module, const dt_mask_id_t id)
{
  // same gesture as renaming a module (ctrl+click), for consistency -- see
  // _iop_plugin_header_button_release.
  GtkWidget *child = gtk_bin_get_child(GTK_BIN(evbox));
  if(child && GTK_IS_ENTRY(child))
  {
    // already renaming -- a fast repeated ctrl+click can re-enter here while
    // the entry from the first click is still focused. Destroying a focused
    // entry fires its focus-out-event synchronously, which commits the
    // rename and rebuilds the whole list (_build_masks_list) while GTK is
    // still unwinding the outer destroy call on this same row -- a reentrant
    // teardown that corrupts the tree it's still unparenting from and
    // crashes. Just re-focus the existing entry instead of destroying/
    // recreating it.
    gtk_widget_grab_focus(child);
    return;
  }
  if(child) gtk_widget_destroy(child);
  GtkWidget *entry = gtk_entry_new();
  // a stock GtkEntry carries its own border/padding, taller than the plain
  // label it replaces -- without this the row (and the whole panel) grows
  // by a few pixels for as long as the rename is in progress, then shrinks
  // back on commit. Frameless + a zero-padding CSS class (see
  // .mask-rename-entry in darktable.css) keeps the row's height stable.
  gtk_entry_set_has_frame(GTK_ENTRY(entry), FALSE);
  dt_gui_add_class(entry, "mask-rename-entry");
  // and wider: a stock entry asks for about 20 characters, where the label it
  // replaces asks for one (see gtk_label_set_max_width_chars in
  // _make_shape_row). It still fills the width the row gives it
  gtk_entry_set_width_chars(GTK_ENTRY(entry), 1);
  gtk_entry_set_max_width_chars(GTK_ENTRY(entry), 1);
  dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, id);
  if(form)
  {
    // prefill with just the part after the type prefix, so the prefix
    // itself is never in the editable text and can't be typed over
    gchar *rest = _form_display_name(form);
    gtk_entry_set_text(GTK_ENTRY(entry), rest);
    g_free(rest);
  }
  g_object_set_data(G_OBJECT(entry), "formid", GINT_TO_POINTER(id));
  gtk_container_add(GTK_CONTAINER(evbox), entry);
  g_signal_connect(G_OBJECT(entry), "activate", G_CALLBACK(_rename_commit), module);
  g_signal_connect(G_OBJECT(entry), "focus-out-event", G_CALLBACK(_rename_focus_out),
                   module);
  gtk_widget_show(entry);
  gtk_widget_grab_focus(entry);
}

// a deleted formid can be left behind in several bd fields that reference a
// specific shape by id (element selection, solo, solo-edit) -- none of the
// delete paths (_delete_single_shape, _group_reset_members, _group_delete,
// _delete_elements) used to clear these, so after deleting the
// individually-selected shape, _flexi_refine_follow_selection kept reading a
// "valid" (dt_is_valid_maskid) but now-nonexistent panel_selected_formid,
// landing refinement scope on ELEMENT for a shape that no longer resolves --
// seen as the refinement caption still naming the deleted shape and its
// controls reading as disabled, even though the group's own add-target
// selection (panel_selected_group_cid) was unaffected and
// adding a new shape kept working. Called once per deleted formid.
static void _clear_stale_formid_refs(dt_iop_gui_blend_data_t *bd, const dt_mask_id_t id)
{
  if(!bd || !dt_is_valid_maskid(id)) return;
  if(bd->panel_selected_formid == id) bd->panel_selected_formid = INVALID_MASKID;
  if(bd->solo_formid == id) bd->solo_formid = INVALID_MASKID;
  if(bd->soloedit_formid == id) bd->soloedit_formid = INVALID_MASKID;
  // "auto-expand selected" (see _auto_expand_selected_row): a stale
  // reference here just means the option's next selection won't find
  // anything to collapse, harmless, but leaving it wrong would misreport
  // which row _make_props_row_toggle expands on the next full rebuild.
  if(bd->masks_last_expanded_elem == id) bd->masks_last_expanded_elem = NO_MASKID;
  // the group half of the same option keys on a group's head formid, so it can
  // go stale the same way (a run's head is deleted, or the run is emptied)
  if(bd->masks_last_expanded_group == id) bd->masks_last_expanded_group = NO_MASKID;
  if(bd->masks_group_collapse_click == id) bd->masks_group_collapse_click = NO_MASKID;
}

// delete a single shape from the module's mask group: shared by a right-click
// on the row's name (_row_click_press) and the "delete" entry in the row's
// actions menu (_build_shape_actions_menu).
static void _delete_single_shape(dt_iop_module_t *module, const dt_mask_id_t id)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, id);
  if(!grp || !form) return;
  // deferred, so it sees the group after the removal
  _queue_link_peers_rebuild(module);
  _clear_stale_formid_refs(bd, id);
  dt_masks_clear_form_gui(darktable.develop);
  // Detach the point ourselves rather than calling dt_masks_form_remove():
  // that runs a nested history item + dt_masks_iop_update() between removing
  // the point and its own "did the group just empty?" test, and then acts on
  // `grp` across that reentry. Observed result was the module's whole mask
  // group being destroyed while a sibling member was still in it --
  // blend_params.mask_id reset to NO_MASKID and the FLEXI bit lost with the
  // history item that followed, so the panel emptied (or the group vanished
  // from it) while the pipe's own copy of the forms kept rendering the
  // survivor. _detach_group_members() touches only grp->points, with no
  // reentry and no destruction cascade. A group emptied this way stays: its
  // marker does.
  GList *one = g_list_prepend(NULL, GINT_TO_POINTER(id));
  _detach_group_members(grp, one);
  g_list_free(one);
  dt_print(DT_DEBUG_MASKS, "[masks] form %d deleted from panel", id);
  dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  _queue_masks_list_rebuild(module);
  _refresh_canvas_edit(module);
}

void dt_iop_gui_blend_delete_element(dt_iop_module_t *module, const dt_mask_id_t id)
{
  if(module && module->blend_data) _delete_single_shape(module, id);
}

// forward declared here (defined much further down, near _build_shape_actions_menu's
// other caller) so _row_click_press's own right-click can open the same menu
// without reordering half the file.
static void _build_shape_actions_menu(GtkWidget *anchor,
                                      dt_iop_module_t *module,
                                      const dt_mask_id_t id,
                                      GtkWidget *handle,
                                      GtkWidget *evbox);

// unified press/release handlers for a row's three "non-specific" click
// surfaces: the lead icon (handle), the name, and the row's own background
// (covering every gap between actual controls, e.g. the opacity slider) --
// connected identically to all three (see _make_shape_row) so a click has the
// exact same effect no matter which of the three it lands on, as long as it
// isn't a genuinely specific interactive widget in its own right (a slider,
// a colour picker, a badge, ...), each of which keeps its own distinct
// meaning. `w` (whichever of the three received the event) only needs its own
// "formid" tag to work here -- "handle-widget" and "name-evbox" are looked up
// from it too (each of the three carries all three tags, including a
// self-reference on whichever one it itself is, see _make_shape_row):
//  * ctrl+click:         rename
//  * shift+click:        toggle this element's properties/expanded view
//  * right-click:        open the actions menu
//  * plain click/release: select (toggles off if already selected)
// double-click-to-solo used to live here too (see _group_header_press's own
// comment for groups) -- dropped for the same reason: the double-click's
// first press already ran a full press/release cycle through
// _row_click_release's toggle-to-deselect branch, so by the time the second
// press arrived the element read as deselected even though it was the solo
// target, and force-selecting it back afterward was never fully reliable.
// Solo an element via its own solo badge, or the row's actions menu, instead.
// once the right-click actions menu closes -- an item was chosen, or the
// user clicked away/pressed Escape -- auto-expand the row it was opened on,
// if it's still selected and the option is on. Deferred to here rather than
// done up front when the menu opens (see _set_form_target_ext's
// auto_expand=FALSE in _row_click_press below) so the reflow this can cause
// never fights the menu's own popup position while it's open, but the user
// still doesn't have to click the row a second time afterward just to see
// its controls -- right-clicking alone should end up exactly where a plain
// click would have. "hide" fires for every dismissal path (item chosen,
// click-away, Escape) alike, unlike "deactivate" (fires before an item's own
// "activate" completes) or a per-item callback (would have to be repeated
// on every menu entry, including future ones).
static void _shape_popover_closed(GtkPopover *popover, gpointer user_data)
{
  dt_iop_module_t *module = (dt_iop_module_t *)user_data;
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(popover), "formid"));
  dt_iop_gui_blend_data_t *bd = module ? module->blend_data : NULL;
  if(bd && bd->panel_selected_formid == id) _auto_expand_selected_row(module, id);
}

static gboolean
_row_click_press(GtkWidget *w, GdkEventButton *ev, dt_iop_module_t *module)
{
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(w), "formid"));
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  // a fresh press always starts a new interaction -- clear any stale flag a
  // previous press's drag-begin set but whose release never arrived to
  // consume (e.g. a drag cancelled by Escape), so it cannot wrongly swallow
  // this press's own eventual release (see _row_drag_begin / masks_row_click_handled).
  bd->masks_row_click_handled = FALSE;
  if(bd->masks_skip_group_select_release_time != ev->time)
    bd->masks_skip_group_select_release = FALSE;
  if(ev->type == GDK_BUTTON_PRESS && ev->button == GDK_BUTTON_PRIMARY
     && dt_modifier_is(ev->state, GDK_CONTROL_MASK))
  {
    if(bd->panel_selected_formid != id) _set_form_target(module, id);
    GtkWidget *evbox = g_object_get_data(G_OBJECT(w), "name-evbox");
    _start_rename_element(evbox, module, id);
    return TRUE;
  }
  if(ev->type == GDK_BUTTON_PRESS && ev->button == GDK_BUTTON_SECONDARY)
  {
    // not a direct delete any more, so a stray right-click cannot destroy a
    // shape with no confirmation -- delete is one of the actions menu's own
    // items instead (see _build_shape_actions_menu).
    // auto_expand=FALSE: this selects the right-clicked shape (so the menu's
    // own actions target the right one) without also auto-expanding its
    // controls -- see _set_form_target_ext's own comment for why a right-
    // click reflowing the row out from under the about-to-open menu is
    // exactly the bug this avoids.
    if(bd->panel_selected_formid != id) _set_form_target_ext(module, id, FALSE);
    GtkWidget *handle = g_object_get_data(G_OBJECT(w), "handle-widget");
    GtkWidget *evbox = g_object_get_data(G_OBJECT(w), "name-evbox");
    _build_shape_actions_menu(w, module, id, handle, evbox);
    g_object_set_data(G_OBJECT(darktable.gui->active_popover_menu), "formid", GINT_TO_POINTER(id));
    g_signal_connect(G_OBJECT(darktable.gui->active_popover_menu), "closed", G_CALLBACK(_shape_popover_closed), module);
    GdkRectangle rect = { (int)ev->x, (int)ev->y, 1, 1 };
    gtk_popover_set_pointing_to(GTK_POPOVER(darktable.gui->active_popover_menu), &rect);
    gtk_popover_popup(GTK_POPOVER(darktable.gui->active_popover_menu));
    return TRUE;
  }
  // a plain primary press must return FALSE so this widget's own drag source
  // can arm (handle/evbox/row_evbox are all independently armed, see
  // _make_shape_row); selection happens on button-release instead (a release
  // is not delivered when a drag started, so dragging never also selects).
  // See _row_click_release.
  return FALSE;
}

// there is no longer a visible chevron button for a row's (or group's) own
// properties/expanded-view editor -- shift+click on any of a row's own
// non-specific click surfaces toggles it instead, by driving the
// still-alive-but-hidden toggle button the row's handle was tagged with at
// build time (see _make_shape_row / the group header block's "expand-toggle"
// data). No-op if untagged.
static void _toggle_expand_widget(GtkWidget *src)
{
  GtkWidget *btn = g_object_get_data(G_OBJECT(src), "expand-toggle");
  if(!btn) return;
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(btn),
                               !gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(btn)));
}

// a plain click on the handle/name arms a drag source (see _row_click_press's
// own comment) so the row can be dragged to reorder -- but that means the
// eventual "click" completion may arrive as a "drag-begin" signal instead of
// a button-release-event, either because the user genuinely started dragging,
// or (observed on macOS) because the drag source spuriously arms for what
// was, from the user's perspective, an ordinary click with no real movement.
// Either way, select the row right here rather than only ever on release, so
// a plain click that gets swallowed by the drag machinery still selects its
// row -- see masks_row_click_handled's own comment in blend.h for how this
// pairs with _row_click_release to avoid acting twice. Select-only (not the
// toggle _select_form uses) so starting a genuine drag on an already-selected
// row can never read as an accidental deselect the instant the drag begins.
static void _row_drag_begin(GtkWidget *w, GdkDragContext *dc, dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd) return;
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(w), "formid"));
  _set_form_target_ext(module, id, FALSE);
  bd->masks_row_click_handled = TRUE;
}

// matching release for _row_click_press's plain-click case. ctrl+click and
// right-click are both handled entirely on press and must not also do
// anything here.
//
// Always returns TRUE once it has actually acted (selected, toggled expand,
// or consumed a drag-begin's flag): the same callback is connected to all
// three of a row's click surfaces (handle, name-evbox, and the row_evbox
// background that wraps the whole row, see _make_shape_row), and an
// unconsumed button-release-event bubbles from whichever of the inner two
// (handle/evbox) received it up through its ancestors -- including
// row_evbox, which has this identical handler attached too. Returning FALSE
// here used to let that second, bubbled invocation run the same selection
// logic again with `w` now pointing at row_evbox, toggling the row right
// back off in the same click: this is why clicking the lead icon or the name
// silently did nothing while clicking the row's own empty background (which
// receives the event directly, with nothing above it to bubble the toggle
// into a second time) worked fine.
static gboolean
_row_click_release(GtkWidget *w, GdkEventButton *ev, dt_iop_module_t *module)
{
  if(ev->button != GDK_BUTTON_PRIMARY) return FALSE;
  if(dt_modifier_is(ev->state, GDK_CONTROL_MASK)) return FALSE;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(bd->masks_skip_group_select_release)
  {
    bd->masks_skip_group_select_release = FALSE;
    return TRUE;
  }
  // _row_drag_begin already selected this row for this same press -- see its
  // own comment. Consume the flag and stop, so this release cannot also
  // toggle the selection it just set (or run the shift-click branch a second
  // time for a gesture drag-begin already resolved).
  if(bd->masks_row_click_handled)
  {
    bd->masks_row_click_handled = FALSE;
    return TRUE;
  }
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(w), "formid"));
  if(dt_modifier_is(ev->state, GDK_SHIFT_MASK))
  {
    if(bd->panel_selected_formid != id) _set_form_target_ext(module, id, FALSE);
    _toggle_expand_widget(g_object_get_data(G_OBJECT(w), "handle-widget"));
    return TRUE;
  }
  _select_form(module, id);
  return TRUE;
}

// every row/header kind (element/parametric/raster row, real group header,
// empty-group header, cluster header) has its own unique widget name for CSS
// (#mask-shape-row, #mask-group-header-row, ...) but shares the ".mask-panel-row"
// class for exactly this kind of kind-agnostic lookup. Finds the row/header
// widget a crossing event box drives the hover wash for: for an element row it
// is the evbox's own parent (row_vbox wraps row_evbox); for a group/cluster
// header it is the evbox's own child (hdr_evbox wraps hdr). Returns NULL if
// neither shape matches.
static gboolean _is_mask_panel_row(GtkWidget *w)
{
  return w
         && gtk_style_context_has_class(gtk_widget_get_style_context(w),
                                        "mask-panel-row");
}

static GtkWidget *_row_widget_for_hover(GtkWidget *w)
{
  GtkWidget *parent = gtk_widget_get_parent(w);
  if(_is_mask_panel_row(parent)) return parent;
  if(GTK_IS_BIN(w))
  {
    GtkWidget *child = gtk_bin_get_child(GTK_BIN(w));
    if(_is_mask_panel_row(child)) return child;
  }
  return NULL;
}

// list -> canvas hover: hovering a mask-list row highlights its shape on the
// canvas; hovering a cluster header highlights every member shape. The hover
// target ids are carried on the event box as "hover-formids" (a one-element list
// for a single row, the whole member set for a cluster header). The box has a
// real window so crossings into its child buttons report GDK_NOTIFY_INFERIOR,
// which we ignore so the hover stays stable across the row's controls.
// Also drives the row's own hover wash in the list (mirroring the canvas ->
// list sync in dt_iop_gui_masks_hover_form), so hovering a row highlights it
// exactly like hovering its shape on the canvas does.
static gboolean _row_crossing(GtkWidget *w, GdkEventCrossing *ev, dt_iop_module_t *module)
{
  if(ev->detail == GDK_NOTIFY_INFERIOR) return FALSE;
  dt_masks_form_gui_t *gui = darktable.develop->form_gui;
  if(!gui) return FALSE;
  const gboolean entering = ev->type == GDK_ENTER_NOTIFY;
  // interacting with one of this row's controls must keep the shape highlighted
  // for as long as the interaction lasts, not just while the pointer happens to
  // sit inside the row: dragging a slider (or opening a bauhaus popup, which
  // takes a gtk grab) delivers a leave the moment the grab starts, and the drag
  // itself routinely carries the pointer well outside the row. Both are ignored
  // here -- the matching ungrab crossing, or the next real pointer crossing,
  // settles the hover once the interaction is over.
  if(!entering
     && (ev->mode == GDK_CROSSING_GRAB || ev->mode == GDK_CROSSING_GTK_GRAB
         || (ev->state & (GDK_BUTTON1_MASK | GDK_BUTTON2_MASK | GDK_BUTTON3_MASK))))
    return FALSE;
  g_list_free(gui->panel_hover_formids);
  gui->panel_hover_formids = NULL;
  if(entering)
    gui->panel_hover_formids =
      g_list_copy(g_object_get_data(G_OBJECT(w), "hover-formids"));
  // a leave is not always reliably paired with the matching enter (the pointer
  // can move from one row's own GdkWindow straight onto an adjacent row's
  // without a clean crossing sequence for the first one), which could leave a
  // stale hover wash stuck on a row indefinitely -- easily mistaken for that
  // row still being "selected", since both washes look alike. Unconditionally
  // clear every hover class first, then (re)apply it to the current target.
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(bd && bd->masks_list_box) _clear_hover_classes(GTK_WIDGET(bd->masks_list_box));
  GtkWidget *target = _row_widget_for_hover(w);
  if(target && entering) dt_gui_add_class(target, "mask-list-row-hover");
  dt_control_queue_redraw_center();
  return FALSE;
}

// --- group headers -----------------------------------------------------------
// Every group (a marker and its members) gets its own
// header in the list (see _starts_group / _build_masks_list); same-kind runs
// within one group are separately folded into a collapsible kind-cluster
// expander to keep the list manageable when there are many (e.g. tens of)
// brush strokes (see cluster_min in _pack_group_elements).

#define MASK_GROUP_MIN 1 // unused: every operator-group always gets its own header

// a group is named, and numbered, after how it combines its own members: a
// nested group has no between-group operator, and a top-level one is named
// the same way so the two read alike
int _within_index_for_state(const int state)
{
  if(state & DT_MASKS_STATE_SCREEN) return 1;
  if(state & DT_MASKS_STATE_ISECT) return 2;
  if(state & DT_MASKS_STATE_WITHIN_MULTIPLY) return 3;
  if(state & DT_MASKS_STATE_WITHIN_SUM) return 4;
  if(state & DT_MASKS_STATE_WITHIN_DIFFERENCE) return 5;
  if(state & DT_MASKS_STATE_WITHIN_EXCLUSION) return 6;
  return 0;
}

// commit a group's rename entry: the text is the group's name, held by its
// marker. The one group of a mask with no group form yet becomes real here
// (see _module_flexi_group)
static void _group_rename_commit(GtkWidget *entry, dt_iop_module_t *module)
{
  if(g_object_get_data(G_OBJECT(entry), "done")) return; // guard double commit
  g_object_set_data(G_OBJECT(entry), "done", GINT_TO_POINTER(1));
  gchar *txt = g_strdup(gtk_entry_get_text(GTK_ENTRY(entry)));
  if(txt) g_strstrip(txt);
  dt_mask_id_t cid = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(entry), "group-cid"));
  if(txt && *txt)
  {
    dt_masks_point_group_t *marker = _group_point(_module_flexi_group(module, &cid), cid);
    if(marker)
    {
      g_strlcpy(marker->name, txt, sizeof(marker->name));
      dt_print(DT_DEBUG_MASKS, "[masks] group %d renamed to '%s'", cid, txt);
      dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
    }
  }
  g_free(txt);
  // deferred -- see the same comment on _rename_commit
  _queue_masks_list_rebuild(module);
}

static gboolean
_group_rename_focus_out(GtkWidget *entry, GdkEvent *e, dt_iop_module_t *module)
{
  _group_rename_commit(entry, module);
  return FALSE;
}

// Escape abandons the edit and restores whatever text the entry started with
// (a custom name, or empty if there wasn't one yet) instead of committing --
// shared by both a populated and an empty group's rename entry. Sets the same
// "done" guard _group_rename_commit uses, so the focus-out event the
// subsequent rebuild's teardown fires on this entry does not also commit.
static gboolean
_group_rename_key_press(GtkWidget *entry, GdkEventKey *e, dt_iop_module_t *module)
{
  if(e->keyval != GDK_KEY_Escape) return FALSE;
  g_object_set_data(G_OBJECT(entry), "done", GINT_TO_POINTER(1));
  // nothing about the underlying data changes on cancel, so the list's own
  // signature doesn't move either -- without forcing it stale here, the
  // reconcile-by-skip check in _build_masks_list (see _masks_list_signature)
  // would see an unchanged signature and skip the rebuild entirely, leaving
  // this entry on screen forever (it was destroyed, not hidden, when the
  // rename began -- see _start_group_rename -- so there is no cheaper way
  // back to the label than a rebuild).
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(bd) bd->masks_list_sig = DT_INVALID_HASH;
  _queue_masks_list_rebuild(module);
  return TRUE;
}

void dt_iop_gui_blend_masks_creation_ended(dt_iop_module_t *module)
{
  if(!module || !module->blend_data) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(bd->masks_support)
  {
    for(int n = 0; n < DEVELOP_MASKS_NB_SHAPES; n++)
      if(bd->masks_shapes[n])
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_shapes[n]), FALSE);
  }
  // deferred, not direct: dt_masks_change_form_gui is itself called from the
  // middle of dt_masks_set_edit_mode, before edit_mode and selection state have
  // finished transitioning, and a synchronous rebuild there reenters
  // _build_masks_list on that half-set state (see the note in
  // dt_masks_change_form_gui). The idle fires once the caller has unwound.
  _queue_masks_list_rebuild(module);
}

// dev->forms/history was rewritten wholesale from under the panel (undo/redo,
// jump to a history step, style paste, snapshot restore, compress history, a
// module reset -- see dt_dev_reload_history_items). The groups came back with
// the forms; what has to start over is the panel's one-shot selection seeding
// and the signature it skips a rebuild on
void dt_iop_gui_blend_forms_reloaded(dt_iop_module_t *module)
{
  if(!module) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd) return;
  // only a module currently showing the flexi list has anything for the
  // rebuild below to fix -- reload rewrites every module's dev->forms
  // wholesale, but the overwhelming majority of modules were never in flexi
  // mode, so queuing a full masks-panel teardown+rebuild for all of them
  // regardless was pure waste: on a single undo this fired for every module in
  // the pipeline (~70 on a typical default pipeline), even though only one or
  // two had actually changed, and the resulting burst of simultaneous panel
  // rebuilds was observed to perturb the right panel's scroll position.
  const gboolean had_content = bd->masks_list_sig != DT_INVALID_HASH;
  const gboolean flexi =
    module->blend_params && (module->blend_params->mask_mode & DEVELOP_MASK_FLEXI);
  if(had_content || flexi)
  {
    bd->masks_list_sig = DT_INVALID_HASH;
    if(bd->masks_list_box) _queue_masks_list_rebuild(module);
  }
}

// move every member of a dragged cluster together, preserving their relative
// (bottom-up) order, to the position/group a drop indicates -- the same move
// _masks_row_drag_received / _masks_shape_to_group_drop do for one shape,
// generalized to a same-kind run's whole member set. `dst` is either a group's
// id (dst_is_group: the cluster lands on top of it) or the target row's own
// formid (drop lands directly above/below it, per `above`). Returns FALSE
// (no-op) if `member_ids` is empty or `dst` is itself one of the members.
gboolean _masks_cluster_move(dt_iop_module_t *module,
                             GList *member_ids,
                             const dt_mask_id_t dst,
                             const gboolean dst_is_group,
                             const gboolean above)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp || !member_ids) return FALSE;

  for(GList *l = member_ids; l; l = g_list_next(l))
    if(GPOINTER_TO_INT(l->data) == dst) return FALSE;
  dt_masks_form_t *owner = NULL;
  GList *d = _point_node_owner(grp, dst, &owner);
  if(!d || (dst_is_group ? !_group_marker_node(d) : _starts_group(d))) return FALSE;

  // recover the cluster's own relative order from the list holding dst (the
  // DnD payload itself carries no meaningful order, see
  // _masks_cluster_drag_get). Members in another list stay where they are,
  // as a single element dropped across nesting levels does
  GList *ordered = NULL;
  for(GList *l = owner->points; l; l = g_list_next(l))
    if(!_starts_group(l)
       && g_list_find(member_ids,
                      GINT_TO_POINTER(((dt_masks_point_group_t *)l->data)->formid)))
      ordered = g_list_append(ordered, l->data);
  if(!ordered) return FALSE;

  for(GList *l = ordered; l; l = g_list_next(l))
    owner->points = g_list_remove(owner->points, l->data);
  // on top of dst's group, or next to dst: a member, so d->prev is at worst
  // its group's marker
  GList *at = dst_is_group ? _group_last_node(_group_marker_node(d)) : above ? d : d->prev;
  _insert_points_after(owner, at, ordered);
  g_list_free(ordered);
  return TRUE;
}

// how many groups the panel shows: one per marker, or the one a mask with no
// group form yet shows (see _module_flexi_group). With a single group new
// elements land in it automatically; with several, one must be selected.
int _group_count(dt_iop_module_t *module)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp) return 1;
  int n = 0;
  GList *pts = _mask_points(grp);
  for(GList *l = pts; l; l = g_list_next(l))
    if(dt_masks_point_is_marker(l->data)) n++;
  g_list_free(pts);
  return n;
}

// how many groups share the list of group `cid`, itself included: the top
// group's list, or a nested group's. The last one of a list stays
static int _level_group_count(dt_masks_form_t *grp, const dt_mask_id_t cid)
{
  if(!grp) return 1;
  dt_masks_form_t *owner = grp;
  _point_node_owner(grp, cid, &owner);
  int n = 0;
  for(GList *l = owner->points; l; l = g_list_next(l))
    if(_starts_group(l)) n++;
  return n;
}

// remember a widget's tooltip text as set at construction time, so a later
// disabled-state update (_update_add_target_sensitivity) can append a hint
// without clobbering the button's own description
static void _stash_base_tooltip(GtkWidget *w)
{
  gchar *base = gtk_widget_get_tooltip_text(w);
  g_object_set_data_full(G_OBJECT(w), "dt-base-tooltip", base, g_free);
}

// append `hint` (may be "") to a widget's construction-time tooltip, replacing
// whatever hint was appended last time round
static void _append_tooltip_hint(GtkWidget *w, const char *hint)
{
  const char *base = g_object_get_data(G_OBJECT(w), "dt-base-tooltip");
  if(!base) return;
  gchar *tt = g_strconcat(base, hint, NULL);
  gtk_widget_set_tooltip_text(w, tt);
  g_free(tt);
}

// re-append (or drop) the disabled-state hint on a widget previously stashed
// with _stash_base_tooltip, matching its current sensitivity
static void
_restate_tooltip_hint(GtkWidget *w, const gboolean has_target, const char *no_target_hint)
{
  _append_tooltip_hint(w, has_target ? "" : no_target_hint);
}

// Which group a newly added element will land in.
//
// Normally the explicit panel selection. But when the mask has exactly one
// group there is nowhere else an element could go, so that group is the target
// whether or not it happens to be selected -- making the user click the only
// candidate first is pure ceremony. `implicit` records which of the two
// happened, so the add buttons can say which in their tooltips.
//
// Single source of truth for both halves of "where does this land": the button
// sensitivity/tooltips (_update_add_target_sensitivity) and the insertion
// itself (_recompute_insert_hint). Those derived it separately before, which is
// exactly how the enabled state and the actual destination drift apart.
dt_masks_add_target_t _resolve_add_target(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  dt_masks_add_target_t t = { INVALID_MASKID, FALSE, FALSE };

  if(dt_is_valid_maskid(bd->panel_selected_group_cid) && grp
     && _group_point(grp, bd->panel_selected_group_cid))
  {
    t.cid = bd->panel_selected_group_cid;
    t.valid = TRUE;
  }
  else if(_group_count(module) == 1 || _group_partition_count(grp) == 1)
  {
    // the sole group, or the mask's own when its list holds one: what the
    // mask holds at the top is that group's (masks_revamp_nested_groups.md,
    // Q8). A mask with no group form yet has no id for it, and its first
    // element creates it (see dt_masks_group_insert_point)
    GList *heads = _group_partition_heads(grp);
    t.cid = heads ? GPOINTER_TO_INT(heads->data) : INVALID_MASKID;
    g_list_free(heads);
    t.valid = TRUE;
    t.implicit = TRUE;
  }
  return t;
}

// whole-mask (global scope) refinement is always reachable: it operates on
// the final composited mask regardless of how many shapes exist, so it stays
// enabled unconditionally. A GROUP/EMPTY_GROUP/ALL_SHAPES-scoped refinement,
// by contrast, is only meaningful when its target actually has a member to
// refine -- an empty staged group, or a real group whose run has no member
// formids, contributes nothing to the mask, so refining it would just be a
// second, redundant place to do what the global controls already do (see
// _flexi_refine_follow_selection, which retargets this same widget set to
// whichever scope the current panel selection implies). Called after every
// masks-list rebuild, from dt_iop_gui_update_blending, and whenever the
// panel selection retargets the scope, so it tracks live add/remove of
// shapes and selection changes without needing the panel reopened.
static void _update_refine_sensitivity(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd) return;

  gboolean active = TRUE;
  if(bd->masks_refine_scope_kind == REFINE_SCOPE_GROUP)
  {
    dt_masks_form_t *grp = _module_mask_group(module);
    GList *ids = _selected_group_formids(grp, bd->masks_refine_scope_formid);
    active = ids != NULL;
    g_list_free(ids);
  }
  else if(bd->masks_refine_scope_kind == REFINE_SCOPE_ALL_SHAPES)
  {
    dt_masks_form_t *grp = _module_mask_group(module);
    active = grp && grp->points != NULL;
  }
  else if(bd->masks_refine_scope_kind == REFINE_SCOPE_ELEMENT)
  {
    // the targeted element can vanish (deleted) without the scope itself
    // being re-derived first -- e.g. deleting the very shape this scope
    // still points at leaves masks_refine_scope_formid stale until the next
    // selection change, so this must verify the point still exists rather
    // than assume ELEMENT scope always targets something real.
    dt_masks_form_t *grp = _module_mask_group(module);
    active = grp && _group_point(grp, bd->masks_refine_scope_formid) != NULL;
  }
  // REFINE_SCOPE_GLOBAL always targets something real

  // Check if refinement for current target is bypassed (disabled)
  gpointer key = _refine_scope_key(bd);
  gboolean bypassed = FALSE;
  if(bd->masks_refine_bypassed)
    bypassed = GPOINTER_TO_INT(g_hash_table_lookup(bd->masks_refine_bypassed, key));
  if(bypassed) active = FALSE;

  if(bd->masks_refine_section_label)
    _restate_tooltip_hint(bd->masks_refine_section_label, active,
                          _("\nrefinements cannot be applied to an empty group -- to "
                            "refine the whole image, deselect the group."));

  if(bd->masks_feathering_guide_combo)
    gtk_widget_set_sensitive(bd->masks_feathering_guide_combo, active);
  if(bd->feathering_radius_slider)
    gtk_widget_set_sensitive(bd->feathering_radius_slider, active);
  if(bd->blur_radius_slider) gtk_widget_set_sensitive(bd->blur_radius_slider, active);
  if(bd->brightness_slider) gtk_widget_set_sensitive(bd->brightness_slider, active);
  if(bd->contrast_slider) gtk_widget_set_sensitive(bd->contrast_slider, active);
  if(bd->details_slider) gtk_widget_set_sensitive(bd->details_slider, active);
}

// enable/disable the add-element controls (shapes, parametric channels, the
// combo) to match whether there is a target group for them to land in, and
// refresh the refinement-scope combo to match the current selection. Shared by
// _build_masks_list (full rebuild) and the lightweight, no-rebuild selection
// paths (_set_group_target, _select_group) so group selection never needs a
// full list rebuild just to keep these in step.
static void _update_add_target_sensitivity(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  // adding a shape / parametric mask / raster / imported shape targets the group
  // _resolve_add_target picks: the selected one, or the sole group when there is
  // only one (nothing to disambiguate). Only a real ambiguity -- several groups,
  // none selected -- disables the controls.
  const dt_masks_add_target_t target = _resolve_add_target(module);
  const gboolean has_target = target.valid;
  // say where the element will land, not just when it cannot land anywhere
  const char *hint = !has_target
                       ? _("\n(select a group first: there is more than one, so where"
                           " the new element goes is ambiguous)")
                     : target.implicit ? _("\n(added to the only group)")
                                       : _("\n(added to the selected group)");
  for(int n = 0; n < DEVELOP_MASKS_NB_SHAPES; n++)
    if(bd->masks_shapes[n])
    {
      gtk_widget_set_sensitive(bd->masks_shapes[n], has_target);
      _append_tooltip_hint(bd->masks_shapes[n], hint);
    }
  if(bd->masks_param_channels_box)
    gtk_widget_set_sensitive(bd->masks_param_channels_box, has_target);
  // the container above is also disabled, but each channel button is set
  // insensitive individually too (matching masks_shapes/raster/import) so its
  // own disabled-state tooltip stays reachable. These also say so when the
  // channel-preview mode is on, since then resting on one does something
  // beyond adding an element (see _preview_on_hover_enter).
  if(bd->masks_param_channels_inner)
  {
    gchar *ch_hint =
      _preview_on_hover_is_on()
        ? g_strconcat(hint,
                      _("\nrest the pointer here to preview this channel"),
                      NULL)
        : g_strdup(hint);
    for(GList *l =
          gtk_container_get_children(GTK_CONTAINER(bd->masks_param_channels_inner));
        l; l = g_list_delete_link(l, l))
    {
      gtk_widget_set_sensitive(GTK_WIDGET(l->data), has_target);
      _append_tooltip_hint(GTK_WIDGET(l->data), ch_hint);
    }
    g_free(ch_hint);
  }

  // import also adds elements to the target group, so it needs the same
  // target and the same explanation when there isn't one
  if(bd->masks_import_btn)
  {
    gtk_widget_set_sensitive(bd->masks_import_btn, has_target);
    gchar *tt = g_strconcat(_("link or copy shapes from other modules, copy their parametric\n"
                              "channels, or add or use another module's whole mask\n"
                              "(click to pick)"),
                            hint, NULL);
    gtk_widget_set_tooltip_text(bd->masks_import_btn, tt);
    g_free(tt);
  }

  // refresh the refinement scope combo: forms may have been added/removed/renamed,
  // or the selected group may have changed
  _refine_scope_combo_rebuild(module);
}

// recompute the insertion hint read by dt_masks_gui_form_save_creation from the
// current target. The target itself is resolved by _resolve_add_target, shared
// with the add-button sensitivity so the destination and the enabled state can
// never disagree -- including the "only one group, so no selection needed" case.
static void _recompute_insert_hint(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  bd->insert_active = FALSE;
  bd->insert_after_fid = INVALID_MASKID;

  // on top of the target group: after its top member, or after its marker when
  // it has none. The one group of a mask with no group form yet has neither,
  // and dt_masks_group_insert_point creates it with the element
  const dt_masks_add_target_t target = _resolve_add_target(module);
  GList *marker = target.valid ? _point_node(grp, target.cid) : NULL;
  if(marker)
  {
    bd->insert_active = TRUE;
    bd->insert_after_fid = ((dt_masks_point_group_t *)_group_last_node(marker)->data)->formid;
  }
}

// move group `src_cid`, its marker and its members, right above group
// `dst_cid` (`above`) or right below it. Returns FALSE (no-op) if the two are
// the same group or either is not a group
gboolean _masks_reorder_groups(dt_iop_module_t *module,
                               const dt_mask_id_t src_cid,
                               const dt_mask_id_t dst_cid,
                               const gboolean above)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  dt_masks_form_t *sowner = NULL, *downer = NULL;
  GList *src = _point_node_owner(grp, src_cid, &sowner);
  GList *dst = _point_node_owner(grp, dst_cid, &downer);
  if(!src || !dst || src == dst || !_starts_group(src) || !_starts_group(dst))
    return FALSE;

  GList *slice = NULL;
  for(GList *l = src; l && (l == src || !_starts_group(l)); l = g_list_next(l))
    slice = g_list_append(slice, l->data);
  // across nesting levels: the list it leaves keeps a group, and each member
  // may go where it lands (see _may_move_into)
  gboolean ok = sowner == downer || _list_group_count(sowner) > 1;
  for(GList *l = slice->next; l && ok; l = g_list_next(l))
    ok = _may_move_into(grp, sowner, downer, ((dt_masks_point_group_t *)l->data)->formid);
  if(!ok)
  {
    g_list_free(slice);
    return FALSE;
  }
  for(GList *l = slice; l; l = g_list_next(l))
  {
    sowner->points = g_list_remove(sowner->points, l->data);
    ((dt_masks_point_group_t *)l->data)->parentid = downer->formid;
  }
  _insert_points_after(downer, above ? _group_last_node(dst) : dst->prev, slice);
  g_list_free(slice);
  return TRUE;
}

dt_masks_form_t *_model_nested_group_of(dt_masks_form_t *grp, const dt_mask_id_t cid)
{
  dt_masks_form_t *owner = NULL;
  GList *node = _point_node_owner(grp, cid, &owner);
  return node && _starts_group(node) && owner && owner != grp && _list_group_count(owner) == 1
           ? owner
           : NULL;
}

// the group of the nested group `sub` goes into the list of `downer`, right
// above or below the group whose marker node is `dst`, and the reference to
// `sub` goes. The group keeps its own settings; the between-group operator it
// had no use for while nested becomes union. Only for a plain reference, the
// one a nested group shown as its group has (see _nested_as_group): the
// settings of any other reference apply on top of the group's own
static gboolean _unnest_group(dt_masks_form_t *grp,
                              dt_masks_form_t *sub,
                              GList *dst,
                              dt_masks_form_t *downer,
                              const gboolean above)
{
  dt_masks_form_t *holder = NULL;
  GList *ref = _point_node_owner(grp, sub->formid, &holder);
  if(!ref || !holder || !sub->points) return FALSE;
  const dt_masks_point_group_t *r = ref->data;
  if(r->opacity != 1.0f || r->refinement.enabled != DT_MASKS_REFINE_OFF
     || (r->state & (DT_MASKS_STATE_INVERSE | DT_MASKS_STATE_HIDDEN | DT_MASKS_STATE_DISABLE)))
    return FALSE;
  for(GList *l = sub->points->next; l; l = g_list_next(l))
    if(!_may_move_into(grp, sub, downer, ((dt_masks_point_group_t *)l->data)->formid))
      return FALSE;

  holder->points = g_list_delete_link(holder->points, ref);
  free((dt_masks_point_group_t *)r);
  GList *slice = sub->points;
  sub->points = NULL;
  dt_masks_point_group_t *mk = slice->data;
  mk->state = (mk->state & ~DT_MASKS_STATE_OP_COMBINE) | DT_MASKS_STATE_UNION;
  for(GList *l = slice; l; l = g_list_next(l))
    ((dt_masks_point_group_t *)l->data)->parentid = downer->formid;
  _insert_points_after(downer, above ? _group_last_node(dst) : dst->prev, slice);
  g_list_free(slice);
  return TRUE;
}

gboolean _model_move_group(dt_iop_module_t *module,
                           const dt_mask_id_t src_cid,
                           const dt_mask_id_t dst_cid,
                           const gboolean above,
                           const gboolean inside)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  dt_masks_form_t *sowner = NULL, *downer = NULL;
  GList *src = _point_node_owner(grp, src_cid, &sowner);
  GList *dst = _point_node_owner(grp, dst_cid, &downer);
  if(!src || !dst || src == dst || !_starts_group(src) || !_starts_group(dst))
    return FALSE;
  // a nested group holding one group is shown as that group, and moves as the
  // element it is in its holder's list
  dt_masks_form_t *snest = _model_nested_group_of(grp, src_cid);
  dt_masks_form_t *dnest = _model_nested_group_of(grp, dst_cid);
  if(inside)
    return snest ? _model_drop_element_onto_group(module, grp, snest->formid, dst_cid)
                 : _model_nest_group(grp, src_cid, dst_cid);
  if(dnest)
  {
    // beside a nested group is among its holder's elements
    if(snest)
      return _model_drop_element_onto_element(module, grp, snest->formid, dnest->formid,
                                              above);
    dt_masks_form_t *holder = NULL;
    GList *ref = _point_node_owner(grp, dnest->formid, &holder);
    return ref && ref->prev
           && _wrap_group(grp, sowner, src, holder, above ? ref->data : ref->prev->data);
  }
  if(snest) return _unnest_group(grp, snest, dst, downer, above);
  return _masks_reorder_groups(module, src_cid, dst_cid, above);
}

// highest number currently held by a live group of within-group mode `mode`
// (see _within_index_for_state), 0 if none. A new group takes one past this, so
// a number is never handed out while a peer still shows it, and a series
// restarts at 1 once its last group is gone.
int _group_ord_max_for_within(dt_iop_module_t *module, const int mode)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  int mx = 0;

  if(bd->group_ordinals)
  {
    // numbers are the mask's, nested groups' included
    GList *pts = _mask_points(grp);
    for(GList *l = pts; l; l = g_list_next(l))
    {
      const dt_masks_point_group_t *head = l->data;
      if(!dt_masks_point_is_marker(head)) continue;
      if(_within_index_for_state(head->state) != mode) continue;
      const int ord = GPOINTER_TO_INT(
        g_hash_table_lookup(bd->group_ordinals, GINT_TO_POINTER(head->formid)));
      if(ord > mx) mx = ord;
    }
    g_list_free(pts);
  }
  return mx;
}

// drop remembered numbers whose group no longer exists, so a series can restart
// at 1 once emptied (and the table does not grow across edits/images)
// a deleted/reshaped group can leave bd->solo_group_key pointing at a cid
// that no longer identifies any real run (see _clear_stale_formid_refs's own
// comment for the formid-keyed siblings of this same bug class -- solo_group_key
// is cid-keyed, not formid-keyed, so it needs its own check). Left stale, every
// row/header in the panel keeps reading "some group is soloed, and it isn't
// me" (see the dt_is_valid_maskid(bd->solo_formid) || bd->solo_group_key != 0
// dimming checks in both _pack_empty_group_header and the real-group header
// build), dimming everything to 45% opacity including a freshly emptied
// group's own header -- visible as a hard opacity seam against its own,
// undimmed pending-row body. Self-healing at rebuild (like
// _prune_group_ordinals) rather than chasing every mutation call site.
void _prune_stale_solo(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(bd->solo_group_key == 0) return;
  const dt_masks_point_group_t *pt =
    _group_point(_module_mask_group(module), (dt_mask_id_t)bd->solo_group_key);
  if(!pt || !dt_masks_point_is_marker(pt)) bd->solo_group_key = 0;
}

void _prune_group_ordinals(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd->group_ordinals) return;
  dt_masks_form_t *grp = _module_mask_group(module);

  GHashTableIter it;
  gpointer k, v;
  g_hash_table_iter_init(&it, bd->group_ordinals);
  while(g_hash_table_iter_next(&it, &k, &v))
  {
    const dt_masks_point_group_t *pt = _group_point(grp, GPOINTER_TO_INT(k));
    if(!pt || !dt_masks_point_is_marker(pt)) g_hash_table_iter_remove(&it);
  }
}

/* The group's displayed number. This is a remembered identity, assigned once
   and kept for as long as the group exists -- NOT a positional count. Numbering
   groups by position meant deleting one renumbered every survivor above it, so
   removing union-1 turned union-2 into union-1 and read as though the wrong
   group had been deleted.

   Groups keep their number in bd->group_ordinals, keyed by their id, which
   emptying and refilling a group leaves alone. */
static int _group_ordinal_any(dt_iop_module_t *module, const dt_mask_id_t cid)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;

  // the one group of a mask with no group form yet
  if(!dt_is_valid_maskid(cid)) return 1;
  const dt_masks_point_group_t *head = _group_point(_module_mask_group(module), cid);
  if(!head) return 0;
  // the mask's own group is "whole mask", not a numbered group: numbering it
  // would start its nested groups of the same mode at 2
  if(cid == _mask_group_cid(module)) return 0;

  if(!bd->group_ordinals)
    bd->group_ordinals = g_hash_table_new(g_direct_hash, g_direct_equal);

  int ord =
    GPOINTER_TO_INT(g_hash_table_lookup(bd->group_ordinals, GINT_TO_POINTER(cid)));
  if(ord <= 0)
  {
    ord = _group_ord_max_for_within(module, _within_index_for_state(head->state)) + 1;
    g_hash_table_insert(bd->group_ordinals, GINT_TO_POINTER(cid), GINT_TO_POINTER(ord));
  }
  return ord;
}

/* Give a number to every group that has none, walking in render order (bottom
   up) so a first build numbers groups the way they are stacked. Groups added
   later just take the next free number for their operator, wherever they sit --
   the number says which group this is, not where it sits. Called once per
   rebuild, after _prune_group_ordinals. */
static void _assign_group_ordinals(dt_iop_module_t *module)
{
  GList *pts = _mask_points(_module_mask_group(module));
  for(GList *l = pts; l; l = g_list_next(l))
    if(dt_masks_point_is_marker(l->data))
      _group_ordinal_any(module, ((dt_masks_point_group_t *)l->data)->formid);
  g_list_free(pts);
}

// 1-based per-operator ordinal of group `cid` (see _group_ordinal_any)
int _group_ordinal_of_cid(dt_iop_module_t *module, const dt_mask_id_t cid)
{
  return _group_ordinal_any(module, cid);
}

// "add group": a new empty group, selected, on top of the members of the
// group new elements go to -- the selected one, or the mask's own -- folding
// its members with the within-group operator `within`. A group is part of the
// mask now, so it is recorded
static void _stage_new_group(dt_iop_module_t *module, const int within)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  const dt_masks_add_target_t target = _resolve_add_target(module);
  dt_mask_id_t cid = target.valid ? target.cid : INVALID_MASKID;
  dt_masks_form_t *grp = _module_flexi_group(module, &cid);
  if(!grp) return;
  const dt_mask_id_t nid = _model_nest_new_group(grp, within, cid);
  if(!dt_is_valid_maskid(nid))
  {
    dt_control_log(_("this group is nested as deep as groups go"));
    return;
  }
  bd->panel_selected_formid = INVALID_MASKID;
  bd->panel_selected_group_cid = nid;
  bd->masks_new_group_op = within & DT_MASKS_STATE_WITHIN;
  dt_print(DT_DEBUG_MASKS, "[masks] add group %d inside group %d within=0x%x", nid, cid,
           within);
  dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
  _build_masks_list(module);
}

// select a real group by its header. The selected group is where the next drawn
// shape lands (it adopts the group's operator) and what the refinement controls
// target. Clicking the already selected group deselects it (selection toggles),
// so the user can reach a state with no group selected.
static void
_select_group(dt_iop_module_t *module, const dt_mask_id_t cid, const int op_state)
{
  (void)op_state;
  // any open parametric editor stays open across a group-target change --
  // it is bound to a specific form, not to which group is selected.
  const dt_masks_panel_sel_t s = _model_click_group(module->blend_data, cid);
  _set_group_target(module, s.group_cid);
}

// core of group selection: point the panel's "where do new elements go" target
// at the group cid (INVALID_MASKID to clear it), then update the header/row
// highlight and the dependent controls (add-element sensitivity, refinement
// scope combo) in place -- no list rebuild, so this never disturbs the GTK
// focus chain and never triggers the containing scrolled viewport to
// auto-scroll to a re-created widget (see _select_group / _select_form /
// _param_enter_edit, all of which funnel group selection through here instead
// of _build_masks_list).
// Selecting anything but the AI object stepped into steps out of it, as a click
// outside it on the canvas does; keep_entered is the element an element
// selection is on its way to, which may be that object.
static void _set_group_target_ext(dt_iop_module_t *module,
                                  const dt_mask_id_t cid,
                                  const dt_mask_id_t keep_entered)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  bd->panel_selected_formid = INVALID_MASKID;
  bd->panel_selected_group_cid = cid;
  _select_mask_group_if_none(bd);
  bd->masks_shown = DT_MASKS_EDIT_FULL;
  dt_masks_set_edit_mode(module, DT_MASKS_EDIT_FULL);
  // dt_masks_set_edit_mode(FULL) just rebuilt form_visible as the *whole*
  // group, which would silently widen an active solo-edit back to every
  // shape's outline (e.g. any group/header selection change routes through
  // here). Re-narrow immediately so solo-edit's canvas scope survives any
  // selection change while it's active.
  if(dt_is_valid_maskid(bd->soloedit_formid))
  {
    GList *one = g_list_prepend(NULL, GINT_TO_POINTER(bd->soloedit_formid));
    dt_masks_set_edit_mode_forms(module, one, DT_MASKS_EDIT_FULL);
    g_list_free(one);
  }
  _update_row_selection(bd);
  _update_add_target_sensitivity(module);
  // every group selection funnels through here, including the one an element
  // selection makes on its way to _set_form_target -- so this is the single
  // place the group half of "auto-expand selected" has to act
  _auto_expand_selected_group(module, bd->panel_selected_group_cid);
  const dt_mask_id_t entered = _entered_object();
  dt_masks_form_t *obj =
    dt_is_valid_maskid(entered) ? dt_masks_get_from_id(darktable.develop, entered) : NULL;
  // the object's own group and paths are inside it too
  const gboolean inside = keep_entered == entered
                          || (obj && ((dt_is_valid_maskid(keep_entered)
                                       && _group_point(obj, keep_entered))
                                      || (dt_is_valid_maskid(cid) && _group_point(obj, cid))));
  if(dt_is_valid_maskid(entered) && !inside) _step_object(module, INVALID_MASKID);
}

static void _set_group_target(dt_iop_module_t *module, const dt_mask_id_t cid)
{
  _set_group_target_ext(module, cid, INVALID_MASKID);
}

// flexi only: keep the insertion hint (where the next drawn shape lands) in step
// with the current selection on the no-rebuild selection path. The add-group icon
// is intentionally NOT touched here -- it only changes when the user picks an
// operator from the add-group menu.
static void _flexi_new_op_follow_selection(dt_iop_gui_blend_data_t *bd)
{
  if(!bd->module) return;
  if(bd->module->blend_params->mask_mode & DEVELOP_MASK_RASTER) return;
  _recompute_insert_hint(bd->module);
}

// the final click on an operator item of the add-group chooser
static void _new_shape_op_action(GSimpleAction *action, GVariant *parameter, gpointer user_data)
{
  dt_iop_module_t *module = (dt_iop_module_t *)user_data;
  const int idx = g_variant_get_int32(parameter);
  if(darktable.gui->active_popover_menu)
    gtk_popover_popdown(GTK_POPOVER(darktable.gui->active_popover_menu));
  if(module && idx >= 0 && idx < (int)(sizeof(_within_modes) / sizeof(_within_modes[0])))
    _stage_new_group(module, _within_modes[idx].bit);
}

// the add-group operator chooser: every operator a group can fold its members
// with
static gboolean _new_shape_op_press(GtkWidget *w, GdkEventButton *ev, gpointer u)
{
  GtkWidget *btn = u ? GTK_WIDGET(u) : w;

  // right-click: the group layout presets, which build a whole set of groups at
  // once. They live here rather than in the panel settings because that is what
  // they are -- a bulk version of this button, not a preference.
  if(ev->button == GDK_BUTTON_SECONDARY && ev->type == GDK_BUTTON_PRESS)
  {
    dt_iop_module_t *module = g_object_get_data(G_OBJECT(btn), "module");
    if(!module || !module->blend_data) return FALSE;
    if(module->blend_params->mask_mode & DEVELOP_MASK_RASTER) return FALSE;
    GMenu *pmenu = g_menu_new();
    _add_flexi_presets_menu(pmenu, btn, module);
    darktable.gui->active_popover_menu = dt_gui_popover_menu_from_model(btn, pmenu);
    gtk_popover_popup(GTK_POPOVER(darktable.gui->active_popover_menu));
    g_object_unref(pmenu);
    return TRUE;
  }

  if(ev->button != GDK_BUTTON_PRIMARY) return FALSE;

  dt_iop_module_t *module = g_object_get_data(G_OBJECT(btn), "module");
  GActionGroup *action_group = gtk_widget_get_action_group(btn, "masks_new_op");
  if(action_group == NULL)
  {
    GActionEntry action_entries[] =
    {
      { "add", _new_shape_op_action, "i", NULL },
    };
    action_group = G_ACTION_GROUP(g_simple_action_group_new());
    g_action_map_add_action_entries(G_ACTION_MAP(action_group), action_entries,
                                    G_N_ELEMENTS(action_entries), module);
    gtk_widget_insert_action_group(btn, "masks_new_op", action_group);
  }

  GMenu *menu = g_menu_new();
  for(int i = 0; i < (int)(sizeof(_within_modes) / sizeof(_within_modes[0])); i++)
  {
    GMenuItem *it = _within_mode_gmenu_item(i, "masks_new_op.add", i);
    g_menu_append_item(menu, it);
    g_object_unref(it);
  }
  darktable.gui->active_popover_menu = dt_gui_popover_menu_from_model(btn, menu);
  gtk_popover_popup(GTK_POPOVER(darktable.gui->active_popover_menu));
  g_object_unref(menu);
  return TRUE;
}

// the bits identifying a shape's kind (ignoring clone/state flags)
static guint _form_kind(const dt_masks_form_t *form)
{
  return form->type
         & (DT_MASKS_CIRCLE | DT_MASKS_PATH | DT_MASKS_GRADIENT | DT_MASKS_ELLIPSE
            | DT_MASKS_BRUSH | DT_MASKS_PARAMETRIC | DT_MASKS_RASTER
#ifdef HAVE_AI
            | DT_MASKS_OBJECT
#endif
         );
}

// manual collapse/expand: the disclosure triangle sits after the label so the
// operator/invert/hide controls stay aligned with the per-row layout. The
// clickable label area carries the body revealer, the triangle widget and the
// group key (so the expanded state survives a rebuild).
/* Detach every listed member from the module's mask group, and nothing else.

   This is deliberately NOT dt_masks_form_remove(module, grp, form). That
   function's grp != NULL branch does the same detach, but then adds:

     if(ok && grp->points == NULL) dt_masks_form_remove(module, NULL, grp);

   i.e. once the last point is gone it permanently deletes the *group form
   itself*, which resets blend_params.mask_id to NO_MASKID (masks.c). Emptying
   the group is exactly what both callers below do, so removing a group that
   happened to hold the mask's last shapes tore down the module's whole mask
   container: _module_mask_group() then returned NULL and the panel lost the
   anchor it renders from -- every group vanished at once. It is also directly
   contrary to _group_reset_members' purpose, which is to KEEP the group.

   Detaching leaves the shapes in dev->forms, unused, exactly as the upstream
   detach branch does; "delete unused shapes" purges them (see
   _masks_import_cleanup_unused). Callers record one masks history item and
   trigger one rebuild afterwards, so the per-removal history/update work that
   dt_masks_form_remove does (and that masks_rebuild_suppressed existed to mask)
   is not needed either. */
static void _detach_group_members(dt_masks_form_t *grp, GList *fids)
{
  // one point per listed id: a form listed twice is referenced twice
  for(GList *l = fids; l; l = g_list_next(l))
  {
    dt_masks_form_t *owner = NULL;
    GList *p = _point_node_owner(grp, GPOINTER_TO_INT(l->data), &owner);
    if(!p || _starts_group(p)) continue;
    free(p->data);
    owner->points = g_list_delete_link(owner->points, p);
  }
}

// members whose form is gone from dev->forms. They render nothing (see
// _group_get_mask_roi_flexi), but a run made only of them has no row to head
// it, so the panel can show neither that group nor the empty groups anchored
// on it, and they count as members when the last real one is deleted. Dropped
// like any other detach, once per point: a form can be referenced more than
// once. Returns how many went
int _model_prune_dangling_members(dt_masks_form_t *grp)
{
  GList *gone = NULL;
  // nested groups' members too: _detach_group_members takes each from the
  // list that holds it
  GList *pts = _mask_points(grp);
  for(GList *l = pts; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    // a group marker refers to no form by design
    if(dt_masks_point_is_marker(pt)) continue;
    if(!dt_masks_get_from_id(darktable.develop, pt->formid))
      gone = g_list_prepend(gone, GINT_TO_POINTER(pt->formid));
  }
  g_list_free(pts);
  const int n = g_list_length(gone);
  if(gone) _detach_group_members(grp, gone);
  g_list_free(gone);
  return n;
}

gboolean _model_ensure_a_group(dt_masks_form_t *grp)
{
  return dt_masks_group_ensure_marker(darktable.develop ? darktable.develop->forms : NULL, grp);
}

// remove elements from the module's mask; their groups stay, even emptied
static void _delete_elements(dt_iop_module_t *module, GList *fids)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp || !fids) return;
  dt_masks_clear_form_gui(darktable.develop);
  for(GList *l = fids; l; l = g_list_next(l))
    _clear_stale_formid_refs(bd, GPOINTER_TO_INT(l->data));
  _detach_group_members(grp, fids);
  dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  // deferred: this is called from a header's own press handler, which is
  // still mid-dispatch on `module`'s header widget -- rebuilding synchronously
  // here would destroy that widget out from under GTK's event propagation and
  // crash (same class of bug as the DnD teardown race, see _rebuild_masks_list_idle)
  _queue_masks_list_rebuild(module);
  _refresh_canvas_edit(module);
}

// "delete group": the group goes, and its elements with it. The last group
// stays: the mask always has one
// the nested group form whose only group is `cid`, or NULL: deleting that
// group deletes the nested group, the element its holder shows it as
static dt_masks_form_t *_sole_nested_group(dt_masks_form_t *grp, const dt_mask_id_t cid)
{
  const dt_masks_point_group_t *mk = grp ? _group_point(grp, cid) : NULL;
  if(!mk || mk->parentid == grp->formid || _level_group_count(grp, cid) > 1) return NULL;
  return dt_masks_get_from_id(darktable.develop, mk->parentid);
}

static void _group_delete(dt_iop_module_t *module, const dt_mask_id_t cid)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  const dt_masks_form_t *nested = _sole_nested_group(grp, cid);
  if(nested)
  {
    if(bd->panel_selected_group_cid == cid) bd->panel_selected_group_cid = INVALID_MASKID;
    _delete_single_shape(module, nested->formid);
    return;
  }
  if(!grp || _level_group_count(grp, cid) <= 1) return;
  dt_masks_clear_form_gui(darktable.develop);
  GList *fids = _model_delete_group(grp, cid);
  for(GList *l = fids; l; l = g_list_next(l))
    _clear_stale_formid_refs(bd, GPOINTER_TO_INT(l->data));
  g_list_free(fids);
  if(bd->panel_selected_group_cid == cid) bd->panel_selected_group_cid = INVALID_MASKID;
  dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
  // deferred, same reasoning as _delete_elements above
  _queue_masks_list_rebuild(module);
  _refresh_canvas_edit(module);
}

// "empty group": its elements go, the group stays where it is, selected, with
// its settings and its number
static void _group_reset_members(dt_iop_module_t *module, const dt_mask_id_t cid)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp) return;
  dt_masks_clear_form_gui(darktable.develop);
  GList *fids = _model_empty_group(grp, cid);
  for(GList *l = fids; l; l = g_list_next(l))
    _clear_stale_formid_refs(bd, GPOINTER_TO_INT(l->data));
  g_list_free(fids);
  bd->panel_selected_group_cid = cid;
  dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  // deferred, same reasoning as _delete_elements above
  _queue_masks_list_rebuild(module);
  _refresh_canvas_edit(module);
}

static void
_group_op_apply(dt_iop_module_t *module, const dt_mask_id_t cid, const dt_masks_state_t op);
static void _build_group_actions_menu(GtkWidget *anchor,
                                      dt_iop_module_t *module,
                                      const dt_mask_id_t cid,
                                      const gboolean is_base,
                                      GtkWidget *lbl_box);

// start inline rename on a group's title: swap `lbl_box`'s label child for an
// entry, same gesture as renaming an element (ctrl+click, see
// _start_rename_element) -- but the typed text names the group (see
// _group_rename_commit). Shared by ctrl+click on the header background/title
// (_group_header_press) and ctrl+click on the operator handle
// (_group_op_press): ctrl+click is now the one shared "rename" gesture
// regardless of which of a header's non-specific-widget areas it lands on
// (icon, title, or the empty gaps between them), matching the same rule
// shift+click (toggle properties) and right-click (open the actions menu)
// already follow.
static void _start_group_rename(GtkWidget *lbl_box,
                                dt_iop_module_t *module,
                                const dt_mask_id_t cid)
{
  // the mask's own group is labeled "whole mask", never by a name of its own
  if(!lbl_box || cid == _mask_group_cid(module)) return;
  GtkWidget *current = g_object_get_data(G_OBJECT(lbl_box), "title-child");
  if(current && GTK_IS_ENTRY(current))
  {
    // already renaming -- see _row_click_press for why a fast repeated
    // ctrl+click must re-focus rather than destroy/recreate the entry
    gtk_widget_grab_focus(current);
    return;
  }
  // renaming acts on the group, so it should select it too (never deselect --
  // same select-only rule every other action control follows, see
  // _update_add_target_sensitivity) and the selection should still be there
  // once the rename commits: neither commit path touches selection, and
  // committing only ever rebuilds the list (which preserves it), so
  // selecting here is the one place this needs to happen.
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(bd->panel_selected_group_cid != cid) _set_group_target(module, cid);
  const char *custom = _group_custom_name(_module_mask_group(module), cid);
  if(current) gtk_widget_destroy(current);
  GtkWidget *entry = gtk_entry_new();
  gtk_entry_set_has_frame(GTK_ENTRY(entry), FALSE);
  dt_gui_add_class(entry, "mask-rename-entry");
  // as narrow a request as the title label it replaces (see
  // _start_rename_element)
  gtk_entry_set_width_chars(GTK_ENTRY(entry), 1);
  gtk_entry_set_max_width_chars(GTK_ENTRY(entry), 1);
  if(custom) gtk_entry_set_text(GTK_ENTRY(entry), custom);
  g_object_set_data(G_OBJECT(entry), "group-cid", GINT_TO_POINTER(cid));
  g_object_set_data(G_OBJECT(lbl_box), "title-child", entry);
  dt_gui_box_add(lbl_box, dt_gui_expand(entry));
  gtk_box_reorder_child(GTK_BOX(lbl_box), entry, 0);
  g_signal_connect(G_OBJECT(entry), "activate", G_CALLBACK(_group_rename_commit), module);
  g_signal_connect(G_OBJECT(entry), "focus-out-event",
                   G_CALLBACK(_group_rename_focus_out), module);
  g_signal_connect(G_OBJECT(entry), "key-press-event",
                   G_CALLBACK(_group_rename_key_press), module);
  // the header (hdr_evbox, found by walking up to the ancestor tagged
  // "group-key" -- see its construction) is armed as a reorder drag source
  // whenever there are 2+ groups. GTK's own drag
  // recognizer can arm on a ctrl+click's press even with no real subsequent
  // movement (the exact same spurious-arm quirk _row_drag_begin documents
  // for element rows) and takes a pointer grab that steals keyboard focus
  // right back off the entry just grabbed below -- firing its focus-out
  // handler, which commits (on unchanged text) and destroys the entry a
  // moment after it appeared. Disarming the drag source for the duration of
  // the rename prevents that; it is safely re-armed for free the next time
  // the panel rebuilds, which every rename commit/cancel path already
  // triggers.
  for(GtkWidget *w = lbl_box; w; w = gtk_widget_get_parent(w))
    if(g_object_get_data(G_OBJECT(w), "group-key"))
    {
      gtk_drag_source_unset(w);
      break;
    }
  gtk_widget_show(entry);
  gtk_widget_grab_focus(entry);
}

// the header event box: a plain primary press must return FALSE so the group
// drag source can arm (the group is selected on release, see below).
// Right-click opens the operator/actions menu (see below).
static gboolean
_group_header_press(GtkWidget *w, GdkEventButton *e, dt_iop_module_t *module)
{
  // double-click used to solo the whole group here, but the preceding single
  // click's own release always ran first (selecting/deselecting the group)
  // before the second press could be recognized as a double-click, so
  // double-clicking an already-selected group reliably deselected it right
  // as it was soloed -- force-selecting afterward (tried first) didn't fully
  // resolve it either. Dropped rather than keep chasing it: a group can
  // still be soloed via its "solo" menu item (see _build_group_op_menu) or
  // the solo badge itself once active (see _solo_badge_group_press).
  // rename and the actions menu select the group first, as they do an element
  // (see _row_click_press), so the highlight shows what they act on
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  const dt_mask_id_t cid =
    (dt_mask_id_t)GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(w), "group-key"));
  const gboolean rename = e->type == GDK_BUTTON_PRESS && e->button == GDK_BUTTON_PRIMARY
                          && dt_modifier_is(e->state, GDK_CONTROL_MASK);
  const gboolean menu = e->button == GDK_BUTTON_SECONDARY;
  if((rename || menu) && bd->panel_selected_group_cid != cid)
    _set_group_target(module, cid);
  if(rename)
  {
    _start_group_rename(g_object_get_data(G_OBJECT(w), "title-label-box"), module, cid);
    return TRUE;
  }
  if(menu)
  {
    const gboolean is_base = g_object_get_data(G_OBJECT(w), "is-base-group") != NULL;
    GtkWidget *lbl_box = g_object_get_data(G_OBJECT(w), "title-label-box");
    _build_group_actions_menu(w, module, cid, is_base, lbl_box);
    GdkRectangle rect = { (int)e->x, (int)e->y, 1, 1 };
    gtk_popover_set_pointing_to(GTK_POPOVER(darktable.gui->active_popover_menu), &rect);
    gtk_popover_popup(GTK_POPOVER(darktable.gui->active_popover_menu));
    return TRUE;
  }
  bd->masks_skip_group_select_release = FALSE;
  return FALSE; // let the drag source arm; selection happens on release
}

// select the group on release (a release is not delivered when a drag started,
// so dragging a group never also selects it). A release that bubbled up from an
// action control (operator chip, ...) rather than a genuine click on the title
// takes the select-only branch instead: acting on the group selects it if it
// wasn't already selected, but never deselects it -- only a click on the title
// itself toggles selection off (see _select_group). Shift+click has no special
// meaning here any more: a group's opacity is always visible inline in the
// header now (see the header build below), so it just falls through to a
// plain select, same as an unmodified click.
static gboolean
_group_header_release(GtkWidget *w, GdkEventButton *e, dt_iop_module_t *module)
{
  if(e->button != GDK_BUTTON_PRIMARY) return FALSE;
  // ctrl+click is handled entirely on press (_group_header_press starts the
  // rename entry there) and must not also act here -- same guard
  // _row_click_release already has for the identical element-rename gesture.
  // Without it this release still ran _select_group, which can deselect the
  // group and queues a list rebuild that destroys the rename entry _start_
  // group_rename just created on the very same click, before the user can
  // type anything.
  if(dt_modifier_is(e->state, GDK_CONTROL_MASK)) return FALSE;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  const dt_mask_id_t cid =
    (dt_mask_id_t)GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(w), "group-key"));
  if(bd->masks_skip_group_select_release)
  {
    bd->masks_skip_group_select_release = FALSE;
    if(bd->panel_selected_group_cid != cid) _set_group_target(module, cid);
    return FALSE;
  }
  const int opstate = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(w), "group-op"));
  _select_group(module, cid, opstate);
  return FALSE;
}

// The group's BODY (its block) uses the two handlers above verbatim, so the body
// and the header cannot disagree about what a click means. But the header -- and
// every element row and editor -- sits INSIDE the block, and all of their
// handlers return FALSE so the drag source can arm. GTK therefore bubbles those
// clicks up to the block, where running the same toggle a second time undoes the
// first: clicking a group header selected the group and then instantly
// deselected it, so the header looked completely inert.
//
// Filter by delivery instead of by widget: act only on events GDK delivered to
// the block's OWN window, which is exactly the group body that no child covers
// -- the padding, the indent left of the element rows, the gaps between them.
// Anything a child already saw is left alone, and still reaches its own handler.
static gboolean _event_on_own_window(GtkWidget *w, const GdkEventButton *e)
{
  return e->window == gtk_widget_get_window(w);
}

static gboolean
_group_block_press(GtkWidget *w, GdkEventButton *e, dt_iop_module_t *module)
{
  if(!_event_on_own_window(w, e)) return FALSE;
  return _group_header_press(w, e, module);
}

static gboolean
_group_block_release(GtkWidget *w, GdkEventButton *e, dt_iop_module_t *module)
{
  if(!_event_on_own_window(w, e)) return FALSE;
  return _group_header_release(w, e, module);
}

// solo a whole group: show only its member shapes, hiding all others.
// Toggling off restores every hidden bit (solo is the only thing that ever
// sets DT_MASKS_STATE_HIDDEN now that real mute has been removed). Used to be
// the ctrl+click branch of a combined mute/solo "power" button; now triggered
// from the group's own "solo" menu item (see _build_group_op_menu) or by
// clicking its own solo badge to clear it, with the soloed state shown by a
// badge next to the label instead of a button icon.
// Model half of the group solo toggle; mirrors _model_toggle_solo_form.
dt_masks_solo_canvas_t _model_toggle_solo_group(dt_iop_module_t *module,
                                                dt_masks_form_t *grp,
                                                const guint key,
                                                GList *members)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!grp) return DT_MASKS_SOLO_CANVAS_NONE;
  dt_masks_solo_canvas_t canvas = DT_MASKS_SOLO_CANVAS_NONE;

  if(bd->solo_group_key == key)
  {
    dt_masks_group_isolate_state(grp, NULL, DT_MASKS_STATE_HIDDEN);
    bd->solo_group_key = 0;
  }
  else
  {
    dt_masks_group_isolate_state(grp, members, DT_MASKS_STATE_HIDDEN);
    // only one thing is ever soloed: a group solo cancels any element solo
    bd->solo_formid = INVALID_MASKID;
    bd->solo_group_key = key;
    // same mutual-exclusivity rule as _model_toggle_solo_form
    if(dt_is_valid_maskid(bd->soloedit_formid))
    {
      bd->soloedit_formid = INVALID_MASKID;
      canvas = DT_MASKS_SOLO_CANVAS_FULL;
    }
  }
  if(_model_clear_soloedit_if_hidden(module, grp))
    canvas = DT_MASKS_SOLO_CANVAS_FULL;
  return canvas;
}

static void _toggle_solo_group(dt_iop_module_t *module, const dt_mask_id_t cid)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp) return;

  GList *members = _selected_group_formids(grp, cid);
  const dt_masks_solo_canvas_t canvas =
    _model_toggle_solo_group(module, grp, (guint)cid, members);
  g_list_free(members);
  if(canvas == DT_MASKS_SOLO_CANVAS_FULL)
    dt_masks_set_edit_mode(module, DT_MASKS_EDIT_FULL);
  dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  _sync_hidden_to_form_visible(module);
  // group solo only flips hidden/dim state and solo badges, never the list
  // structure, so refresh every row (and the group-solo badges / empty-group
  // dimming) in place -- exactly like _toggle_solo_form -- instead of tearing
  // down and rebuilding the whole panel. _refresh_all_shape_rows mutates only
  // existing widgets, so it is safe to call synchronously here even though we
  // may still be inside the triggering menu item's own event dispatch (unlike
  // a full rebuild, which would destroy the widget mid-event).
  _refresh_all_shape_rows(module);
  _sync_solo_canvas_highlight(module);
  // same as _toggle_solo_form
  _soloedit_follow_selection(module->blend_data);
}

// same idea as _solo_badge_form_press: the group badge only shows solo (never
// solo-edit, groups have no such concept) while its group is the soloed one,
// so a click while it shows solo always means "un-solo". Must check the
// status first (see _solo_badge_form_press).
static gboolean
_solo_badge_group_press(GtkWidget *w, GdkEventButton *e, dt_iop_module_t *module)
{
  if(e->button != GDK_BUTTON_PRIMARY) return FALSE;
  const int status = _solo_status_badge_get(w);
  if(status == MASK_SOLO_BADGE_SOLO)
  {
    _toggle_solo_group(module, _header_cid(w));
    return TRUE;
  }
  else if(status == MASK_SOLO_BADGE_DISABLE)
  {
    _group_op_apply(module, _header_cid(w), DT_MASKS_STATE_OP_BYPASS);
    return TRUE;
  }
  return FALSE;
}

// set a group's within-group combine mode, on its marker. Union (no within
// bit) ⇒ byte-identical for groups that never touch this.
static void
_within_mode_apply(dt_iop_module_t *module, dt_mask_id_t cid, const dt_masks_state_t within)
{
  dt_masks_point_group_t *marker = _group_point(_module_flexi_group(module, &cid), cid);
  if(!marker) return;
  marker->state = (marker->state & ~DT_MASKS_STATE_WITHIN) | (within & DT_MASKS_STATE_WITHIN);
  dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
  _build_masks_list(module);
}

// the marker of the mask's own group: the first of its list, or INVALID_MASKID
// for a mask with no group form yet
static dt_mask_id_t _root_cid(dt_iop_module_t *module)
{
  const dt_masks_form_t *grp = _module_mask_group(module);
  return grp && grp->points && dt_masks_point_is_marker(grp->points->data)
           ? ((dt_masks_point_group_t *)grp->points->data)->formid
           : INVALID_MASKID;
}

// the mask's own group, the one every other group nests in: its list's only
// group. INVALID_MASKID for a mask with no group form yet, and for a list of
// several groups, which only an edit stored before one-group masks holds
static dt_mask_id_t _mask_group_cid(dt_iop_module_t *module)
{
  dt_masks_form_t *grp = module ? _module_mask_group(module) : NULL;
  if(!grp || _level_group_count(grp, INVALID_MASKID) != 1) return INVALID_MASKID;
  return _root_cid(module);
}

// one group is always selected: with none, the mask's own. That is also why it
// is the one group a click cannot deselect (see _model_click_group)
static void _select_mask_group_if_none(dt_iop_gui_blend_data_t *bd)
{
  if(!bd || dt_is_valid_maskid(bd->panel_selected_group_cid)) return;
  bd->panel_selected_group_cid = _mask_group_cid(bd->module);
}

static void _within_action(GSimpleAction *action, GVariant *parameter, gpointer user_data)
{
  GtkWidget *anchor = GTK_WIDGET(user_data);
  dt_iop_module_t *module = g_object_get_data(G_OBJECT(anchor), "module");
  const dt_mask_id_t cid = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(anchor), "within_cid"));
  const dt_masks_state_t within = (dt_masks_state_t)g_variant_get_int32(parameter);
  if(darktable.gui->active_popover_menu)
    gtk_popover_popdown(GTK_POPOVER(darktable.gui->active_popover_menu));
  if(module) _within_mode_apply(module, cid, within);
}

// build and show the within-group combine chooser (union / screen /
// intersect) for a group. Shared by a direct click on the chooser button and
// the "change within-group mode" shortcut.
static void _build_within_menu(GtkWidget *anchor, dt_iop_module_t *module, const dt_mask_id_t cid)
{
  g_object_set_data(G_OBJECT(anchor), "module", module);
  g_object_set_data(G_OBJECT(anchor), "within_cid", GINT_TO_POINTER(cid));

  GActionGroup *action_group = gtk_widget_get_action_group(anchor, "masks_within");
  if(action_group == NULL)
  {
    GActionEntry action_entries[] =
    {
      { "set", _within_action, "i", NULL },
    };
    action_group = G_ACTION_GROUP(g_simple_action_group_new());
    g_action_map_add_action_entries(G_ACTION_MAP(action_group), action_entries,
                                    G_N_ELEMENTS(action_entries), anchor);
    gtk_widget_insert_action_group(anchor, "masks_within", action_group);
  }

  GMenu *menu = g_menu_new();
  for(int i = 0; i < (int)(sizeof(_within_modes) / sizeof(_within_modes[0])); i++)
  {
    GMenuItem *it = _within_mode_gmenu_item(i, "masks_within.set", _within_modes[i].bit);
    g_menu_append_item(menu, it);
    g_object_unref(it);
  }

  darktable.gui->active_popover_menu = dt_gui_popover_menu_from_model(anchor, menu);
  gtk_popover_popup(GTK_POPOVER(darktable.gui->active_popover_menu));
  g_object_unref(menu);
}

static gboolean
_group_within_press(GtkWidget *widget, GdkEventButton *ev, gpointer user_data)
{
  if(ev->button != GDK_BUTTON_PRIMARY) return FALSE;
  GtkWidget *btn = user_data;
  dt_iop_module_t *module = g_object_get_data(G_OBJECT(btn), "module");
  if(!module) return TRUE;

  _build_within_menu(btn, module, _header_cid(btn));
  return TRUE;
}

// change the operator of a whole group, on its marker.
//
// `op` == DT_MASKS_STATE_OP_BYPASS is the one special case: bypass is a modifier
// on top of the group's operator, not an operator of its own, so it toggles the
// bypass bit and leaves the rest of the operator alone. Picking any real
// operator clears the bypass bit with the rest of the old one, which is what
// makes choosing an operator on a disabled group re-enable it.
static void
_group_op_apply(dt_iop_module_t *module, dt_mask_id_t cid, const dt_masks_state_t op)
{
  dt_masks_point_group_t *marker = _group_point(_module_flexi_group(module, &cid), cid);
  if(!marker) return;
  if(op == DT_MASKS_STATE_OP_BYPASS)
    marker->state ^= DT_MASKS_STATE_OP_BYPASS;
  else
    marker->state = (marker->state & ~DT_MASKS_STATE_OP) | op;
  dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
  // deferred: also reachable directly from the group's own operator-handle
  // press handler (_group_op_press's ctrl/shift-click), still mid-dispatch on
  // that widget -- see _delete_elements above for why this must not be
  // synchronous
  _queue_masks_list_rebuild(module);
  _refresh_canvas_edit(module);
}

// merge this group down into the group directly below it: its members join
// that group, whose settings then apply to them. No-op for the bottom group
// (nothing below it).
static void _merge_group_down(dt_iop_module_t *module, const dt_mask_id_t cid)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!_model_merge_group_down(grp, cid)) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(bd->panel_selected_group_cid == cid) bd->panel_selected_group_cid = INVALID_MASKID;
  dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  // deferred: also reachable directly from the group's own operator-handle
  // press handler (_group_op_press's shift-click), same reasoning as
  // _group_op_apply above
  _queue_masks_list_rebuild(module);
}

// the group header's drag handle doubles as its operator chip: shift+click
// merges the group down into the group below it (first-class groups). A
// plain click is left alone here (returns FALSE) so the handle's own drag
// source can still arm -- the operator chooser it used to open immediately
// opens instead on the matching release (see _group_op_release), once it is
// known no drag actually happened. That chooser is also where both invert
// actions below live now (see _build_group_op_menu) -- there used to be a
// ctrl+click shortcut straight to _invert_group_members here, dropped once
// the menu offered both invert actions explicitly (a bare ctrl+click could
// only ever reach one of the two, and picking which one silently would have
// been confusing now that they mean different things).
//
// flip every member's own inversion bit independently (not a group-wide state
// to set/clear): ON, OFF, ON becomes OFF, ON, OFF. A one-shot action, not a
// persistent "group is inverted" mode -- shared by the operator chooser's
// "invert all elements" entry and the "invert selected group" shortcut. Not
// the same operation as _group_toggle_output_invert below: inverting every
// member and folding is mathematically different from folding and then
// inverting the result, for anything but a single-member group (see
// DT_MASKS_STATE_OP_INVERT in masks.h).
static void _invert_group_members(dt_iop_module_t *module, const dt_mask_id_t cid)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  GList *members = _selected_group_formids(grp, cid);
  if(!members) return;
  for(GList *l = members; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = _group_point(grp, GPOINTER_TO_INT(l->data));
    if(!pt) continue;
    pt->state ^= DT_MASKS_STATE_INVERSE;
  }
  g_list_free(members);
  dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  // an INVERSE-only change touches no row's structure/position, just its own
  // look -- refresh every row in place (same mechanism _invert_element uses
  // for the per-shape gesture) instead of a full teardown+rebuild, which
  // would also needlessly flash the panel while this menu item's own popup
  // is still unwinding (see _rebuild_masks_list_idle's Quartz-teardown note
  // for the same class of hazard in a different gesture).
  _refresh_all_shape_rows(module);
}

// invert-output (DT_MASKS_STATE_OP_INVERT, "true" group invert): a persistent
// flag on the group's marker, unlike _invert_group_members' one-shot member
// flip. Individual members' own DT_MASKS_STATE_INVERSE bits are untouched --
// the two are independent.
static void _group_toggle_output_invert(dt_iop_module_t *module, const dt_mask_id_t cid)
{
  dt_masks_point_group_t *marker = _group_point(_module_mask_group(module), cid);
  if(!marker) return;
  marker->state ^= DT_MASKS_STATE_OP_INVERT;
  dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
  // like _invert_group_members's own switch away from a full rebuild: this
  // touches no row's structure, just this one group's own handle look --
  // update it in place instead of tearing down the whole panel.
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(bd && bd->masks_list_box)
    _apply_group_output_invert_icon(GTK_WIDGET(bd->masks_list_box), (guint)cid,
                                    (marker->state & DT_MASKS_STATE_OP_INVERT) != 0);
}

// the group header's opacity slider's tooltip stands in for its own hidden
// label/value (see the header build below), so it must track live drag
// ticks, not just report the value the slider was built with.
static void _group_opacity_update_tooltip(GtkWidget *slider, const float value)
{
  gchar *tip = g_strdup_printf(_("opacity: %.0f%%\n"
                                 "applied on top of -- not instead of -- each "
                                 "element's own opacity, the two multiply together"),
                               value * 100.0f);
  gtk_widget_set_tooltip_text(slider, tip);
  g_free(tip);
}

// pressing/dragging the slider is, like every other action control in a
// group header (the operator handle, the old properties chevron), not a
// click on the title -- it must not be able to deselect an already-selected
// group. A bauhaus widget's own click handling does not consume the
// underlying button-press-event, which still bubbles to hdr_evbox's press/
// release afterwards (see _group_op_press for the same reasoning applied to
// the operator handle), so arm the same select-only guard here before that
// happens.
static gboolean
_group_opacity_press(GtkWidget *w, GdkEventButton *ev, dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  bd->masks_skip_group_select_release = TRUE;
  bd->masks_skip_group_select_release_time = ev->time;
  return FALSE; // let the slider's own click/drag handling proceed untouched
}

// set the group's own persistent, multiplicative opacity (see
// dt_masks_point_group_t.group_opacity), on its marker -- an absolute value,
// unlike every other multi-target properties row (_props_row_apply's delta
// convention): a group header always represents exactly one group, so there is
// no multi-select ambiguity a delta needs to resolve.
static void _group_opacity_changed(GtkWidget *w, dt_iop_module_t *module)
{
  if(DT_IN_GUI_UPDATE()) return;
  dt_mask_id_t cid = _header_cid(w);
  dt_masks_point_group_t *marker = _group_point(_module_flexi_group(module, &cid), cid);
  if(!marker) return;
  const float value = dt_bauhaus_slider_get(w);
  marker->group_opacity = value;
  _group_opacity_update_tooltip(w, value);
  dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
  dt_control_queue_redraw_center();
  // an opacity change can push this group -- or, since it scales every
  // member's own effective opacity too, any of its elements -- across the
  // low-opacity threshold; refresh every badge in the panel, not just this
  // group's own (mirrors _props_row_apply's own call for the same reason).
  _refresh_lowop_badges(module);
}

// the group an actions-menu item acts on, and its module
static dt_iop_module_t *_group_act_target(gpointer u, dt_mask_id_t *cid)
{
  GtkWidget *anchor = GTK_WIDGET(u);
  *cid = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(anchor), "group_act_cid"));
  if(darktable.gui->active_popover_menu)
    gtk_popover_popdown(GTK_POPOVER(darktable.gui->active_popover_menu));
  return g_object_get_data(G_OBJECT(anchor), "module");
}

static void _group_act_disable(GSimpleAction *action, GVariant *param, gpointer u)
{
  dt_mask_id_t cid;
  dt_iop_module_t *module = _group_act_target(u, &cid);
  if(module) _group_op_apply(module, cid, DT_MASKS_STATE_OP_BYPASS);
}

static void _group_act_solo(GSimpleAction *action, GVariant *param, gpointer u)
{
  dt_mask_id_t cid;
  dt_iop_module_t *module = _group_act_target(u, &cid);
  if(module) _toggle_solo_group(module, cid);
}

static void _group_act_invert_output(GSimpleAction *action, GVariant *param, gpointer u)
{
  dt_mask_id_t cid;
  dt_iop_module_t *module = _group_act_target(u, &cid);
  if(module) _group_toggle_output_invert(module, cid);
}

static void _group_act_invert_elems(GSimpleAction *action, GVariant *param, gpointer u)
{
  dt_mask_id_t cid;
  dt_iop_module_t *module = _group_act_target(u, &cid);
  if(module) _invert_group_members(module, cid);
}

static void _group_act_rename(GSimpleAction *action, GVariant *param, gpointer u)
{
  GtkWidget *lbl_box = g_object_get_data(G_OBJECT(u), "group_act_lbl_box");
  dt_mask_id_t cid;
  dt_iop_module_t *module = _group_act_target(u, &cid);
  if(module && lbl_box) _start_group_rename(lbl_box, module, cid);
}

static void _group_act_merge_down(GSimpleAction *action, GVariant *param, gpointer u)
{
  dt_mask_id_t cid;
  dt_iop_module_t *module = _group_act_target(u, &cid);
  if(module) _merge_group_down(module, cid);
}

static void _group_act_empty(GSimpleAction *action, GVariant *param, gpointer u)
{
  dt_mask_id_t cid;
  dt_iop_module_t *module = _group_act_target(u, &cid);
  if(module) _group_reset_members(module, cid);
}

static void _group_act_delete(GSimpleAction *action, GVariant *param, gpointer u)
{
  dt_mask_id_t cid;
  dt_iop_module_t *module = _group_act_target(u, &cid);
  if(module) _group_delete(module, cid);
}

// the module's own ("whole mask") refinement, as a group holds one
static dt_masks_refinement_t _refine_of_module(dt_iop_module_t *module)
{
  const dt_develop_blend_params_t *bp = module->blend_params;
  dt_masks_refinement_t r = { 0 };
  r.enabled = _refine_global_is_set(module) ? DT_MASKS_REFINE_GROUP : DT_MASKS_REFINE_OFF;
  r.details = bp->details;
  r.feathering_guide = bp->feathering_guide;
  r.feathering_radius = bp->feathering_radius;
  r.blur_radius = bp->blur_radius;
  r.brightness = bp->brightness;
  r.contrast = bp->contrast;
  return r;
}

// "compose": the element or group of `pt` goes into a new group folding with
// `within`, under a new empty group, which is selected so the next shape
// drawn lands in it
static void _compose(dt_iop_module_t *module,
                     const dt_masks_point_group_t *pt,
                     const dt_masks_state_t within)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp || !pt) return;
  const gboolean whole = dt_masks_point_is_marker(pt) && pt->formid == _mask_group_cid(module);
  dt_masks_clear_form_gui(darktable.develop);
  const dt_mask_id_t eid = _model_compose(grp, pt, within);
  if(!dt_is_valid_maskid(eid))
  {
    dt_control_log(_("this group is nested as deep as groups go"));
    return;
  }

  // the whole mask's refinement refined what is now the group at its bottom,
  // and goes with it: the whole mask starts over. A refinement that group
  // already holds, which only a migrated edit can give the mask's own group,
  // leaves it where it is
  gboolean had_details = FALSE;
  if(whole && _refine_global_is_set(module))
  {
    const dt_masks_point_group_t *ref = grp->points->next->data;
    dt_masks_form_t *sub = dt_masks_get_from_id(darktable.develop, ref->formid);
    dt_masks_point_group_t *mk = sub ? sub->points->data : NULL;
    if(mk && mk->refinement.enabled == DT_MASKS_REFINE_OFF)
    {
      mk->refinement = _refine_of_module(module);
      had_details = _refine_clear_global(module);
    }
  }

  bd->panel_selected_formid = INVALID_MASKID;
  bd->panel_selected_group_cid = eid;
  // the list rebuild keeps the refinement controls on their old scope
  _flexi_refine_follow_selection(bd);
  dt_print(DT_DEBUG_MASKS, "[masks] compose %d within=0x%x, empty group %d", pt->formid,
           within, eid);
  // with the module: a moved whole-mask refinement is in its blend params
  dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
  if(had_details) // see _refine_clear_global
  {
    dt_dev_reprocess_all(module->dev);
    dt_control_queue_redraw();
  }
  _queue_masks_list_rebuild(module);
  _refresh_canvas_edit(module);
}

// the group form whose tree compose and simplify restructure for group `cid`:
// the mask's own, or a nested group's. NULL for an AI object, whose paths
// move as one
static dt_masks_form_t *_restructurable_group(dt_iop_module_t *module, const dt_mask_id_t cid)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  dt_masks_form_t *g = cid == _mask_group_cid(module) ? grp : _model_nested_group_of(grp, cid);
  return g && (g->type & DT_MASKS_GROUP) && !(g->type & DT_MASKS_OBJECT) ? g : NULL;
}

// "simplify": make the tree under group `cid` shallower where that renders
// the same mask (dt_masks_group_simplify). The whole mask also takes over a
// lone group it holds
static void _simplify(dt_iop_module_t *module, const dt_mask_id_t cid)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp) return;
  const gboolean whole = cid == _mask_group_cid(module);
  dt_masks_form_t *target = _restructurable_group(module, cid);
  if(!target) return;

  dt_masks_clear_form_gui(darktable.develop);
  dt_masks_refinement_t r = _refine_of_module(module);
  const gboolean had_refine = r.enabled != DT_MASKS_REFINE_OFF;
  gboolean changed = FALSE;
  for(int pass = 0; pass <= DT_MASKS_NESTING_MAX; pass++)
  {
    gboolean again = dt_masks_group_simplify(darktable.develop->forms, target);
    if(whole) again |= _model_hoist_sole_group(grp, &r);
    changed |= again;
    if(!again) break;
  }
  if(!changed)
  {
    dt_control_log(_("nothing to simplify"));
    return;
  }

  // a hoisted group's refinement is the whole mask's now
  if(!had_refine && r.enabled != DT_MASKS_REFINE_OFF)
  {
    dt_develop_blend_params_t *bp = module->blend_params;
    bp->details = r.details;
    bp->feathering_guide = r.feathering_guide;
    bp->feathering_radius = r.feathering_radius;
    bp->blur_radius = r.blur_radius;
    bp->brightness = r.brightness;
    bp->contrast = r.contrast;
  }

  bd->panel_selected_formid = INVALID_MASKID;
  bd->panel_selected_group_cid = cid;
  // a hoist moved a refinement into the whole mask's, which may be on screen
  _flexi_refine_follow_selection(bd);
  dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
  _queue_masks_list_rebuild(module);
  _refresh_canvas_edit(module);
}

// the "compose" entry opening `compose`, which it takes
static void _append_compose_submenu(GMenu *section, GMenu *compose)
{
  GMenuItem *it = g_menu_item_new_submenu(_("compose"), G_MENU_MODEL(compose));
  g_menu_item_set_attribute(it, "tooltip", "s",
                            _("use this as a building block for a larger mask, for example\n"
                              "to subtract from it or to intersect it with something else:\n"
                              "it goes into a new group of the chosen operator,\n"
                              "with an empty group on top to draw into"));
  g_menu_append_item(section, it);
  g_object_unref(it);
  g_object_unref(compose);
}

static void _group_act_compose(GSimpleAction *action, GVariant *param, gpointer u)
{
  dt_mask_id_t cid;
  dt_iop_module_t *module = _group_act_target(u, &cid);
  if(module)
    _compose(module, _group_point(_module_mask_group(module), cid),
             (dt_masks_state_t)g_variant_get_int32(param));
}

static void _group_act_simplify(GSimpleAction *action, GVariant *param, gpointer u)
{
  dt_mask_id_t cid;
  dt_iop_module_t *module = _group_act_target(u, &cid);
  if(module) _simplify(module, cid);
}

// the "compose" submenu: every operator but `current`, whose group composing
// would only add one more member to. `keep_current` offers it anyway, for a
// member past the base of a group folding in order: `a - (b - c)` is no
// `a - b - c`
static GMenu *_compose_menu(const char *action,
                            const dt_masks_state_t current,
                            const gboolean keep_current)
{
  GMenu *sub = g_menu_new();
  for(int i = 0; i < (int)(sizeof(_within_modes) / sizeof(_within_modes[0])); i++)
  {
    if(_within_modes[i].bit == (current & DT_MASKS_STATE_WITHIN) && !keep_current) continue;
    GMenuItem *it = _within_mode_gmenu_item(i, action, _within_modes[i].bit);
    g_menu_append_item(sub, it);
    g_object_unref(it);
  }
  return sub;
}

static void _build_group_actions_menu(GtkWidget *anchor,
                                      dt_iop_module_t *module,
                                      const dt_mask_id_t cid,
                                      const gboolean is_base,
                                      GtkWidget *lbl_box)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  const dt_masks_point_group_t *marker = _group_point(grp, cid);
  const gboolean bypassed = marker && _op_is_bypassed(marker->state);
  const gboolean output_inverted = marker && (marker->state & DT_MASKS_STATE_OP_INVERT);
  // what acts on the members has nothing to act on in an empty group
  GList *members = _selected_group_formids(grp, cid);
  const gboolean has_members = members != NULL;
  g_list_free(members);

  g_object_set_data(G_OBJECT(anchor), "module", module);
  g_object_set_data(G_OBJECT(anchor), "group_act_cid", GINT_TO_POINTER(cid));
  g_object_set_data(G_OBJECT(anchor), "group_act_lbl_box", lbl_box);

  GSimpleActionGroup *sag = g_simple_action_group_new();
  GActionMap *map = G_ACTION_MAP(sag);

  GSimpleAction *act_dis = g_simple_action_new_stateful("disable", NULL, g_variant_new_boolean(bypassed));
  g_signal_connect(act_dis, "activate", G_CALLBACK(_group_act_disable), anchor);
  g_action_map_add_action(map, G_ACTION(act_dis));

  GSimpleAction *act_solo = g_simple_action_new_stateful("solo", NULL, g_variant_new_boolean(bd->solo_group_key == (guint)cid));
  g_signal_connect(act_solo, "activate", G_CALLBACK(_group_act_solo), anchor);
  g_action_map_add_action(map, G_ACTION(act_solo));

  GSimpleAction *act_inv_out = g_simple_action_new_stateful("invert_output", NULL, g_variant_new_boolean(output_inverted));
  g_signal_connect(act_inv_out, "activate", G_CALLBACK(_group_act_invert_output), anchor);
  g_action_map_add_action(map, G_ACTION(act_inv_out));

  GActionEntry action_entries[] =
  {
    { "invert_elems", _group_act_invert_elems, NULL, NULL },
    { "compose",      _group_act_compose,      "i",  NULL },
    { "simplify",     _group_act_simplify,     NULL, NULL },
    { "rename",       _group_act_rename,       NULL, NULL },
    { "merge_down",   _group_act_merge_down,   NULL, NULL },
    { "empty",        _group_act_empty,        NULL, NULL },
    { "delete",       _group_act_delete,       NULL, NULL },
  };
  g_action_map_add_action_entries(map, action_entries, G_N_ELEMENTS(action_entries), anchor);

  // the one group a list cannot lose is the mask's own: every other group
  // nests in it, and the mask is that group (see _group_delete)
  const gboolean deletable = _level_group_count(_module_mask_group(module), cid) > 1
                             || _sole_nested_group(_module_mask_group(module), cid);

  gtk_widget_insert_action_group(anchor, "masks_group_act", G_ACTION_GROUP(sag));

  GMenu *menu = g_menu_new();

  // visibility
  GMenu *sec_vis = g_menu_new();
  g_menu_append(sec_vis, _("disable"), "masks_group_act.disable");
  if(!bypassed && has_members)
    g_menu_append(sec_vis, _("solo"), "masks_group_act.solo");
  g_menu_append_section(menu, _("visibility"), G_MENU_MODEL(sec_vis));
  g_object_unref(sec_vis);

  if(!bypassed && has_members)
  {
    // mask operations
    GMenu *sec_ops = g_menu_new();
    g_menu_append(sec_ops, _("invert output"), "masks_group_act.invert_output");
    g_menu_append(sec_ops, _("invert all elements"), "masks_group_act.invert_elems");
    if(_restructurable_group(module, cid))
    {
      GMenu *compose = _compose_menu("masks_group_act.compose", marker->state, FALSE);
      _append_compose_submenu(sec_ops, compose);
      GMenuItem *simplify = g_menu_item_new(_("simplify"), "masks_group_act.simplify");
      g_menu_item_set_attribute(simplify, "tooltip", "s",
                                _("remove the groups inside this one that change nothing:\n"
                                  "empty groups, groups holding a single element,\n"
                                  "and groups using the operator of the group holding them"));
      g_menu_append_item(sec_ops, simplify);
      g_object_unref(simplify);
    }
    g_menu_append_section(menu, _("mask operations"), G_MENU_MODEL(sec_ops));
    g_object_unref(sec_ops);
  }

  // edit
  GMenu *sec_edit = g_menu_new();
  if(cid != _mask_group_cid(module))
    g_menu_append(sec_edit, _("rename"), "masks_group_act.rename");
  if(!is_base && !bypassed && has_members)
    g_menu_append(sec_edit, _("merge elements into group below"), "masks_group_act.merge_down");
  if(has_members) g_menu_append(sec_edit, _("empty group"), "masks_group_act.empty");
  if(deletable) g_menu_append(sec_edit, _("delete group"), "masks_group_act.delete");
  g_menu_append_section(menu, _("edit"), G_MENU_MODEL(sec_edit));
  g_object_unref(sec_edit);

  darktable.gui->active_popover_menu = dt_gui_popover_menu_from_model(anchor, menu);
  g_object_unref(menu);
}

// solo-edit a single shape: only its outline/handles are editable on the
// canvas, while the full mask still computes so every shape's effect is still
// visible in the mask overlay. Toggling off restores editing of the whole group.
// Shared by the row's solo-edit toggle button and the "toggle solo-edit for
// current shape" shortcut.
// Model half of the solo-edit toggle; the third corner of the mutual
// exclusivity enforced by _model_toggle_solo_form / _model_toggle_solo_group.
dt_masks_solo_canvas_t _model_toggle_soloedit(dt_iop_module_t *module,
                                              dt_masks_form_t *grp,
                                              const dt_mask_id_t id)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(bd->soloedit_formid == id)
  {
    bd->soloedit_formid = INVALID_MASKID;
    return DT_MASKS_SOLO_CANVAS_FULL;
  }

  bd->soloedit_formid = id;
  // solo and solo-edit are mutually exclusive (see the matching clear in
  // _model_toggle_solo_form/_model_toggle_solo_group) -- drop any active solo
  // and restore every element's visibility, since solo-edit only isolates
  // what is editable, not what is shown.
  if(dt_is_valid_maskid(bd->solo_formid) || bd->solo_group_key != 0)
  {
    dt_masks_group_isolate_state(grp, NULL, DT_MASKS_STATE_HIDDEN);
    bd->solo_formid = INVALID_MASKID;
    bd->solo_group_key = 0;
  }
  return DT_MASKS_SOLO_CANVAS_ONE;
}

static void _toggle_soloedit(dt_iop_module_t *module, const dt_mask_id_t id)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  const gboolean had_solo =
    dt_is_valid_maskid(bd->solo_formid) || bd->solo_group_key != 0;
  const dt_masks_solo_canvas_t canvas =
    _model_toggle_soloedit(module, _module_mask_group(module), id);

  if(canvas == DT_MASKS_SOLO_CANVAS_ONE)
  {
    GList *one = g_list_prepend(NULL, GINT_TO_POINTER(id));
    dt_masks_set_edit_mode_forms(module, one, DT_MASKS_EDIT_FULL);
    g_list_free(one);
    if(had_solo) _sync_hidden_to_form_visible(module);
  }
  else
    dt_masks_set_edit_mode(module, DT_MASKS_EDIT_FULL);
  // solo-edit changes which shape the canvas lets you edit, never the list
  // structure. The refresh is for the solo it may have just cleared above
  // (_model_toggle_soloedit drops any active solo), whose hidden bits every row
  // paints from. It does not have to be deferred: _refresh_all_shape_rows only
  // mutates existing widgets, so it is safe synchronously even though the
  // selection change that drives it (see _soloedit_follow_selection) can arrive
  // mid-rebuild (same reasoning as _toggle_solo_group).
  _refresh_all_shape_rows(module);
}

// drop-target feedback: while a drag hovers a group header, wash its frame so the
// user sees which group an element would land in. The "row-frame" data points at
// the named header box (where the selection/hover CSS lives); the class is removed
// on leave (and the post-drop rebuild recreates the rows anyway).
static void _clear_drop_classes(GtkWidget *f)
{
  dt_gui_remove_class(f, "mask-list-row-drop");
  dt_gui_remove_class(f, "mask-list-row-drop-above");
  dt_gui_remove_class(f, "mask-list-row-drop-below");
}

// clear the drop feedback from `f` *and its siblings*. The insertion line is
// drawn on a canonical neighbour rather than always on the hovered group (see
// _canonical_drop_frame), so the widget wearing the class is not necessarily
// the one a later motion/leave event arrives on -- clearing only `f` would
// strand a line on the group next door.
static void _clear_group_drop_classes(GtkWidget *f)
{
  _clear_drop_classes(f);
  GtkWidget *parent = gtk_widget_get_parent(f);
  if(!GTK_IS_CONTAINER(parent)) return;
  GList *kids = gtk_container_get_children(GTK_CONTAINER(parent));
  for(GList *l = kids; l; l = g_list_next(l))
    if(l->data != f) _clear_drop_classes(GTK_WIDGET(l->data));
  g_list_free(kids);
}

// The gap between two adjacent groups is ONE insertion slot, but it has two
// names: "below the upper group" and "above the lower group". Drawing each on
// its own block's edge put two different lines a few pixels apart (the blocks
// carry a 4px margin between them), so a single slot read as two competing drop
// targets and the indicator appeared to jump as the pointer crossed the
// boundary.
//
// Collapse the two names to one: a slot is always drawn as the *top* edge of
// the group below it. Crossing between two groups then changes nothing on
// screen at all, because both sides resolve to the same widget and the same
// class. Only the bottom-most slot, which has no group below it, stays a
// "below" on the last group's own bottom edge.
//
// Purely presentational -- the drop itself still acts on the group actually
// under the pointer with its own above/below (the two describe the same gap, so
// they move the group to the same place). Nothing about the model changes here.
static GtkWidget *_canonical_drop_frame(GtkWidget *f, gboolean *above)
{
  if(*above) return f; // already "top edge of the group below the slot"

  GtkWidget *parent = gtk_widget_get_parent(f);
  if(!GTK_IS_CONTAINER(parent)) return f;

  // Find the neighbour by on-screen geometry, not by position in the child
  // list. The list packs blocks with gtk_box_pack_end, and reasoning about what
  // that implies for gtk_container_get_children's order is exactly the kind of
  // assumption that is easy to get backwards and hard to see in a screenshot --
  // allocations say where things actually are.
  GtkAllocation fa;
  gtk_widget_get_allocation(f, &fa);
  const int f_mid = fa.y + fa.height / 2;

  GtkWidget *below = NULL;
  int below_mid = 0;
  GList *kids = gtk_container_get_children(GTK_CONTAINER(parent));
  for(GList *l = kids; l; l = g_list_next(l))
  {
    GtkWidget *s = GTK_WIDGET(l->data);
    if(s == f || !gtk_widget_get_visible(s)) continue;
    GtkAllocation sa;
    gtk_widget_get_allocation(s, &sa);
    const int s_mid = sa.y + sa.height / 2;
    if(s_mid <= f_mid) continue;                                  // not below f on screen
    if(!below || s_mid < below_mid) below = s, below_mid = s_mid; // nearest one
  }
  g_list_free(kids);

  if(!below) return f; // f is the bottom-most group: keep its own bottom edge
  *above = TRUE;
  return below;
}

// What is hovering a drop target: a whole group (real or empty) being
// reordered, versus a single element (or a same-kind cluster) being moved into
// a group. The two want opposite feedback -- an insertion line at the edge it
// would land on, versus a highlight of the whole target group.
typedef enum
{
  DND_HOVER_OTHER = 0, // negotiated nothing we know: fall back to a plain highlight
  DND_HOVER_REORDER,   // DND_TARGET_GROUP
  DND_HOVER_ELEMENT    // DND_TARGET_ROW / DND_TARGET_CLUSTER
} dt_masks_dnd_hover_t;

// NB this is not free: gtk_drag_dest_find_target() negotiates against the drag
// pasteboard, which on quartz means a full type-list round trip per call. It
// runs on every motion event, so classify ONCE per event and pass the result
// down (see _group_drop_motion_kind) rather than re-deriving it in a callee.
static dt_masks_dnd_hover_t _dnd_hover_kind(GtkWidget *w, GdkDragContext *dc)
{
  const GdkAtom target = gtk_drag_dest_find_target(w, dc, NULL);
  if(target == GDK_NONE) return DND_HOVER_OTHER;
  gchar *name = gdk_atom_name(target);
  dt_masks_dnd_hover_t kind = DND_HOVER_OTHER;
  if(name)
  {
    if(!strcmp(name, DND_TARGET_GROUP))
      kind = DND_HOVER_REORDER;
    else if(!strcmp(name, DND_TARGET_ROW) || !strcmp(name, DND_TARGET_CLUSTER))
      kind = DND_HOVER_ELEMENT;
    g_free(name);
  }
  return kind;
}

// The body of _group_drop_motion, taking an already-classified hover kind so a
// caller that has classified the event itself does not pay for it twice.
static gboolean _group_drop_motion_kind(GtkWidget *w,
                                        gint y,
                                        GtkWidget *f,
                                        const dt_masks_dnd_hover_t kind)
{
  // siblings too: a reorder line is drawn on a canonical neighbour, not always
  // on this frame (see _canonical_drop_frame)
  _clear_group_drop_classes(f);

  // with shift held a group goes inside the target, which lights up whole, as
  // for an element dropped into it (see _masks_group_drag_received)
  if(kind == DND_HOVER_REORDER && !dt_modifier_is(dt_key_modifier_state(), GDK_SHIFT_MASK))
  {
    // rows display bottom-up: the top half means "land above this group". The
    // decision is _group_drop_above's alone -- the same call the receive
    // handlers make -- so the line drawn here and the move that follows cannot
    // disagree.
    gboolean above = _group_drop_above(w, y);
    // ...then draw that slot in its canonical place, so the gap between two
    // groups shows one line rather than one per neighbour
    GtkWidget *line = _canonical_drop_frame(f, &above);
    dt_gui_add_class(line,
                     above ? "mask-list-row-drop-above" : "mask-list-row-drop-below");
  }
  else
  {
    if(kind == DND_HOVER_ELEMENT)
    {
      // Auto-expand group if hovering a collapsed group
      GtkWidget *exp_toggle = g_object_get_data(G_OBJECT(w), "group-expand-toggle");
      if(exp_toggle && !gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(exp_toggle)))
      {
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(exp_toggle), TRUE);
      }
    }
    dt_gui_add_class(f, "mask-list-row-drop");
  }
  return FALSE; // let GTK_DEST_DEFAULT_MOTION still answer the drag status
}

static gboolean _group_drop_motion(
  GtkWidget *w, GdkDragContext *dc, gint x, gint y, guint time, gpointer frame)
{
  if(!frame) return FALSE;
  return _group_drop_motion_kind(w, y, GTK_WIDGET(frame), _dnd_hover_kind(w, dc));
}

static void
_group_drop_leave(GtkWidget *w, GdkDragContext *dc, guint time, gpointer frame)
{
  // siblings too, for the same reason as the motion handler: the line may be
  // wearing on a neighbouring group's block rather than this frame
  if(frame) _clear_group_drop_classes(GTK_WIDGET(frame));
}

static gboolean _element_drop_motion(
  GtkWidget *w, GdkDragContext *dc, gint x, gint y, guint time, gpointer user_data)
{
  GtkWidget *row_vbox = g_object_get_data(G_OBJECT(w), "row-vbox");
  if(!row_vbox) return FALSE;

  const dt_masks_dnd_hover_t kind = _dnd_hover_kind(w, dc);
  if(kind == DND_HOVER_REORDER && !_drags_as_element(dc))
  {
    // a whole group hovering an element row still means "reorder next to this
    // row's group", so hand it straight to the group-level feedback -- passing
    // the kind we already have, since re-deriving it would negotiate the drag
    // pasteboard a second time for this one motion event
    GtkWidget *group_frame = g_object_get_data(G_OBJECT(w), "group-frame");
    if(group_frame) return _group_drop_motion_kind(w, y, group_frame, kind);
    return FALSE;
  }

  _clear_drop_classes(row_vbox);
  const int h = gtk_widget_get_allocated_height(w);
  const gboolean above = (h > 0 && y < h / 2);
  dt_gui_add_class(row_vbox,
                   above ? "mask-list-row-drop-above" : "mask-list-row-drop-below");
  return FALSE;
}

static void
_element_drop_leave(GtkWidget *w, GdkDragContext *dc, guint time, gpointer user_data)
{
  GtkWidget *row_vbox = g_object_get_data(G_OBJECT(w), "row-vbox");
  if(row_vbox) _clear_drop_classes(row_vbox);
  GtkWidget *group_frame = g_object_get_data(G_OBJECT(w), "group-frame");
  if(group_frame) _clear_drop_classes(group_frame);
}

// a group drag has begun: a drag is not a click, so suppress the button-release
// that selects the group. Some platforms (notably macOS) still deliver a release
// to the drag source when the drag ends, which would otherwise toggle the group's
// selection right after a reorder.
static void _group_drag_begin(GtkWidget *w, GdkDragContext *dc, dt_iop_module_t *module)
{
  dt_print(DT_DEBUG_MASKS, "[masks dnd] group drag-begin");
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(bd)
  {
    bd->masks_skip_group_select_release = TRUE;
    bd->masks_group_op_drag_started = TRUE;
  }
}

static GtkWidget *_make_drag_handle(DTGTKCairoPaintIconFunc kind_paint,
                                    gboolean enabled,
                                    const char *tooltip);

#ifdef HAVE_AI
// shared "value-changed" handler for the two pending-row AI sliders
// (smoothing/cleanup). Applies as a delta against this widget's own last
// value, same convention dt_masks_object_creation_apply_property /
// _object_modify_property already use -- deliberately bypasses
// _props_row_apply entirely (see _make_pending_shape_row's own comment),
// and never triggers a masks-list rebuild, so an in-progress drag on this
// slider is never interrupted.
static void _pending_ai_slider_changed(GtkWidget *widget, dt_iop_module_t *module)
{
  if(DT_IN_GUI_UPDATE()) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd) return;
  const dt_masks_property_t prop =
    (dt_masks_property_t)GPOINTER_TO_INT(g_object_get_data(G_OBJECT(widget), "dt-prop"));
  const float new_val = dt_bauhaus_slider_get(widget);
  float *last = (prop == DT_MASKS_PROPERTY_SMOOTHING) ? &bd->pending_ai_smoothing_last
                                                      : &bd->pending_ai_cleanup_last;
  const float old_val = *last;
  *last = new_val;
  dt_masks_object_creation_apply_property(prop, old_val, new_val);
}
#endif

// "value-changed" handler shared by every pending-row slider that edits a
// shape-creation conf default directly (the same conf keys each shape's own
// _*_events_mouse_scrolled reads/writes while gui->creation is set -- see
// e.g. circle.c's DT_MASKS_CONF(form->type, circle, size)/border, or
// masks.c's own "plugins/darkroom/masks/opacity"). These are absolute
// values, not deltas, and there is no committed form to call
// modify_property on yet, so this writes the conf key straight from the
// slider's own reading and asks the canvas to redraw -- exactly what the
// scroll-wheel gesture already does for the same key, just from a slider
// instead. The conf key string is stashed on the widget at construction
// time; it is always one of DT_MASKS_CONF's own string-literal expansions,
// so no ownership/copy is needed.
static void _pending_conf_slider_changed(GtkWidget *widget, gpointer user_data)
{
  if(DT_IN_GUI_UPDATE()) return;
  const char *key = g_object_get_data(G_OBJECT(widget), "dt-conf-key");
  dt_conf_set_float(key, dt_bauhaus_slider_get(widget));
  dt_control_queue_redraw_center();
}

// ellipse-only variant of the above: its "size" conf key is radius_a, but
// the scroll gesture that this slider mirrors
// (_ellipse_events_mouse_scrolled's plain-scroll branch) scales radius_b by
// the same factor to keep the aspect ratio -- a plain single-key write would
// silently stretch the ellipse as the slider moves. The two conf keys are
// stashed on the widget as "dt-conf-key" (radius_a) / "dt-conf-key2"
// (radius_b) at construction time.
static void _pending_ellipse_size_changed(GtkWidget *widget, gpointer user_data)
{
  if(DT_IN_GUI_UPDATE()) return;
  const char *key_a = g_object_get_data(G_OBJECT(widget), "dt-conf-key");
  const char *key_b = g_object_get_data(G_OBJECT(widget), "dt-conf-key2");
  const float old_a = dt_conf_get_float(key_a);
  const float new_a = dt_bauhaus_slider_get(widget);
  if(old_a > 0.0f)
  {
    const float factor = new_a / old_a;
    dt_conf_set_float(key_b, dt_conf_get_float(key_b) * factor);
  }
  dt_conf_set_float(key_a, new_a);
  dt_control_queue_redraw_center();
}

// builds one such slider, seeded from the conf key's current value.
static GtkWidget *_pending_conf_slider_new(dt_iop_module_t *module,
                                           const char *key,
                                           const char *label,
                                           const float min,
                                           const float max,
                                           const int digits,
                                           const char *format,
                                           const char *tooltip)
{
  GtkWidget *w =
    dt_bauhaus_slider_new_with_range(module, min, max, 0, dt_conf_get_float(key), digits);
  dt_bauhaus_widget_set_label(w, N_("blend"), label);
  if(format) dt_bauhaus_slider_set_format(w, format);
  if(tooltip) gtk_widget_set_tooltip_text(w, tooltip);
  g_object_set_data(G_OBJECT(w), "dt-conf-key", (gpointer)key);
  g_signal_connect(G_OBJECT(w), "value-changed", G_CALLBACK(_pending_conf_slider_changed),
                   NULL);
  dt_gui_add_class(w, "mask-props-slider");
  dt_bauhaus_widget_set_quad_visibility(w, FALSE);
  return w;
}

// synthesizes a disposable, dashed-border placeholder row for the single
// shape currently being drawn (dev->form_gui->creation): not backed by a
// real dt_masks_point_group_t (that only exists once the shape commits), so
// it carries no rename/drag/delete/in-out controls -- just enough to show
// the user which group it will land in, plus a set of live property sliders
// (see below). Torn down and rebuilt like any other row whenever
// _build_masks_list runs (see _masks_list_signature's pending-state hash
// fold), never edited in place.
//
// Property sliders shown here mirror whichever conf-backed defaults each
// shape's own _*_events_mouse_scrolled already reads/writes while
// gui->creation is set (see e.g. circle.c's DT_MASKS_CONF(type, circle,
// size)/border) -- these are the only properties that genuinely make sense
// before commit: they are absolute "what will the next shape be created
// with" values, not the relative/delta edits _build_props_row_editor
// applies to an already-committed shape's own points. Two are handled
// specially: opacity (universal, adjusts the sticky-default conf, see
// _new_shape_default_opacity) and, for DT_MASKS_OBJECT (the AI object
// tool), smoothing/cleanup, which call dt_masks_object_creation_apply_property
// directly since that shape's "size" is potrace-vectorized rather than
// conf-seeded. Brush additionally gets pressure-sensitivity/stroke-
// smoothing preference combos (see the DT_MASKS_BRUSH branch below): unlike
// every slider here, those are *only* ever meaningful pre-commit (they
// affect how the in-progress stroke is captured/simplified, never the
// result afterward), so unlike every other property they are deliberately
// absent from _build_props_row_editor entirely, not just duplicated here.
//
// None of this goes through _props_row_apply/_build_props_row_editor: that
// shared machinery only ever applies a property to forms already found in
// grp->points (_props_row_apply's own loop is `for(GList *fpts =
// grp->points; ...)`), and this form is, by definition, not committed yet.
static gboolean _pending_shape_click(GtkWidget *w, GdkEventButton *e, gpointer user_data)
{
  // absorb clicks on the pending row background so reaching for sliders does
  // not bubble up to group_block and deselect or disarm the shape
  return TRUE;
}

static GtkWidget *_make_pending_shape_row(dt_iop_module_t *module, dt_masks_form_t *form)
{
  const guint kind = _form_kind(form);

  GtkWidget *row = dt_gui_hbox();
  GtkWidget *handle = _make_drag_handle(
    _kind_icon_paint(kind), FALSE,
    _("this shape has not been added yet -- finish drawing it on canvas to add it"));
  g_signal_connect(G_OBJECT(handle), "button-press-event",
                   G_CALLBACK(_pending_shape_click), NULL);
  g_signal_connect(G_OBJECT(handle), "button-release-event",
                   G_CALLBACK(_pending_shape_click), NULL);

  gchar *text = g_strdup_printf(_("new %s"), _kind_name(kind, FALSE));
  GtkWidget *name = gtk_label_new(text);
  g_free(text);
  gtk_label_set_xalign(GTK_LABEL(name), 0.0f);
  gtk_label_set_ellipsize(GTK_LABEL(name), PANGO_ELLIPSIZE_MIDDLE);
  gtk_label_set_max_width_chars(GTK_LABEL(name), 1);
  dt_gui_add_class(name, "mask-row-name");
  gtk_widget_set_tooltip_text(
    name,
    _("this shape has not been added yet -- finish drawing it on canvas to add it"));

  // opacity: universal across every shape kind, mirroring the removed mask manager's
  // own "prop == DT_MASKS_PROPERTY_OPACITY && gui->creation" case, which
  // adjusts the *sticky default* opacity conf (the value the shape actually
  // gets baked in with on commit, see masks.c's _new_shape_default_opacity)
  // rather than a real shape's opacity -- there is no real shape yet. Hidden
  // when "sticky opacity" is off (the options menu's "disable stickiness of
  // opacity"): every new shape gets 100% regardless of this conf then, so a
  // slider here would silently have no effect.
  // opacity: universal across every shape kind, shown directly in the header
  const float init_op = dt_conf_get_float("plugins/darkroom/masks/opacity");
  GtkWidget *opacity = dt_bauhaus_slider_new_with_range(
    module, _blend_masks_properties[DT_MASKS_PROPERTY_OPACITY].min,
    _blend_masks_properties[DT_MASKS_PROPERTY_OPACITY].max, 0, init_op, 2);
  dt_bauhaus_widget_set_label(opacity, N_("blend"),
                              _blend_masks_properties[DT_MASKS_PROPERTY_OPACITY].name);
  dt_bauhaus_slider_set_format(opacity,
                               _blend_masks_properties[DT_MASKS_PROPERTY_OPACITY].format);
  dt_bauhaus_widget_set_quad_visibility(opacity, FALSE);
  dt_bauhaus_widget_hide_label(opacity);
  g_object_set_data(G_OBJECT(opacity), "dt-conf-key",
                    (gpointer) "plugins/darkroom/masks/opacity");
  g_signal_connect(G_OBJECT(opacity), "value-changed",
                   G_CALLBACK(_pending_conf_slider_changed), NULL);
  dt_gui_add_class(opacity, "mask-props-slider");
  dt_gui_add_class(opacity, "mask-inline-opacity");
  _style_opacity_gradient(opacity);

  GtkWidget *val_widget = _make_inline_opacity_value_widget(opacity, module);
  gtk_widget_set_no_show_all(opacity, TRUE);
  gtk_widget_hide(opacity);

  GtkWidget *opacity_slot = dt_gui_hbox();
  dt_gui_box_add(opacity_slot, opacity);
  gtk_box_pack_end(GTK_BOX(opacity_slot), val_widget, TRUE, TRUE, 0);
  gtk_widget_set_halign(val_widget, GTK_ALIGN_END);
  gtk_widget_set_valign(opacity_slot, GTK_ALIGN_CENTER);

  // name column + opacity slot: the exact same shared layout code
  // _make_shape_row itself calls (see _pack_row_header) --
  // not a re-derivation of it -- so this row's name-column width and
  // slider cap/alignment math can never drift from a committed row's again.
  _pack_row_header(row, handle, name, opacity_slot,
                   _make_badge_stack(_make_lowop_badge(), _make_solo_status_badge()),
                   NULL, NULL);

  GtkWidget *row_vbox = dt_gui_vbox(row);
  gtk_widget_set_name(row_vbox, "mask-shape-row");
  dt_gui_add_class(row_vbox, "mask-panel-row");
  dt_gui_add_class(row_vbox, "mask-row-pending");

  // every property slider below docks into this box instead of row_vbox
  // directly, and it is named/classed exactly like _build_props_row_editor's
  // own box (see there) so a committed row's expanded properties and this
  // pending row's own properties get the identical CSS margins (.mask-props-
  // row-editor) -- same inset from the row's edges, same spacing between
  // sliders. Without this the two looked inconsistent: same sliders, but
  // sitting flush against row_vbox's own edges instead of inset like a real
  // row's properties editor.
  GtkWidget *props_box = dt_gui_vbox();
  gtk_widget_set_name(props_box, "mask-props-row-editor");
  dt_gui_add_class(props_box, "mask-props-row-editor");

  if(kind == DT_MASKS_PATH)
  {
    // path has no pre-commit "size" -- it is built up from individually
    // clicked/dragged nodes, not a seeded radius -- but each new node's
    // border does seed from this conf default (see
    // _path_events_button_pressed's own "masks_border" read), even though,
    // unlike every other shape here, path's own _path_events_mouse_scrolled
    // has no gui->creation branch to adjust it live by scrolling. This
    // slider is the only way to adjust it before commit either way.
    GtkWidget *feather = _pending_conf_slider_new(
      module, DT_MASKS_CONF(form->type, path, border),
      _blend_masks_properties[DT_MASKS_PROPERTY_FEATHER].name,
      _blend_masks_properties[DT_MASKS_PROPERTY_FEATHER].min,
      _blend_masks_properties[DT_MASKS_PROPERTY_FEATHER].max, 2,
      _blend_masks_properties[DT_MASKS_PROPERTY_FEATHER].format,
      _("fade-out border the next node placed on this path will start with."));
    dt_gui_box_add(props_box, feather);
  }
  else if(kind == DT_MASKS_CIRCLE)
  {
    GtkWidget *size =
      _pending_conf_slider_new(module, DT_MASKS_CONF(form->type, circle, size),
                               _blend_masks_properties[DT_MASKS_PROPERTY_SIZE].name,
                               _blend_masks_properties[DT_MASKS_PROPERTY_SIZE].min,
                               _blend_masks_properties[DT_MASKS_PROPERTY_SIZE].max, 2,
                               _blend_masks_properties[DT_MASKS_PROPERTY_SIZE].format,
                               _("radius of the next circle, before it is placed -- same "
                                 "as scrolling on canvas."));
    dt_gui_box_add(props_box, size);

    GtkWidget *feather = _pending_conf_slider_new(
      module, DT_MASKS_CONF(form->type, circle, border),
      _blend_masks_properties[DT_MASKS_PROPERTY_FEATHER].name,
      _blend_masks_properties[DT_MASKS_PROPERTY_FEATHER].min,
      _blend_masks_properties[DT_MASKS_PROPERTY_FEATHER].max, 2,
      _blend_masks_properties[DT_MASKS_PROPERTY_FEATHER].format,
      _("fade-out border of the next circle, before it is placed --\n"
        "same as shift+scrolling on canvas."));
    dt_gui_box_add(props_box, feather);
  }
  else if(kind == DT_MASKS_ELLIPSE)
  {
    // size (radius_a) is a special case: the scroll gesture scales radius_b
    // by the same factor to keep the aspect ratio, so a plain conf-write
    // slider (which would only ever touch radius_a) is not enough here --
    // see _pending_ellipse_size_changed.
    GtkWidget *size = dt_bauhaus_slider_new_with_range(
      module, _blend_masks_properties[DT_MASKS_PROPERTY_SIZE].min,
      _blend_masks_properties[DT_MASKS_PROPERTY_SIZE].max, 0,
      dt_conf_get_float(DT_MASKS_CONF(form->type, ellipse, radius_a)), 2);
    dt_bauhaus_widget_set_label(size, N_("blend"),
                                _blend_masks_properties[DT_MASKS_PROPERTY_SIZE].name);
    dt_bauhaus_slider_set_format(size,
                                 _blend_masks_properties[DT_MASKS_PROPERTY_SIZE].format);
    gtk_widget_set_tooltip_text(
      size,
      _("size of the next ellipse, before it is placed -- same as scrolling on canvas."));
    g_object_set_data(G_OBJECT(size), "dt-conf-key",
                      (gpointer)DT_MASKS_CONF(form->type, ellipse, radius_a));
    g_object_set_data(G_OBJECT(size), "dt-conf-key2",
                      (gpointer)DT_MASKS_CONF(form->type, ellipse, radius_b));
    g_signal_connect(G_OBJECT(size), "value-changed",
                     G_CALLBACK(_pending_ellipse_size_changed), NULL);
    dt_gui_add_class(size, "mask-props-slider");
    dt_bauhaus_widget_set_quad_visibility(size, FALSE);
    dt_gui_box_add(props_box, size);

    GtkWidget *feather = _pending_conf_slider_new(
      module, DT_MASKS_CONF(form->type, ellipse, border),
      _blend_masks_properties[DT_MASKS_PROPERTY_FEATHER].name,
      _blend_masks_properties[DT_MASKS_PROPERTY_FEATHER].min,
      _blend_masks_properties[DT_MASKS_PROPERTY_FEATHER].max, 2,
      _blend_masks_properties[DT_MASKS_PROPERTY_FEATHER].format,
      _("fade-out border of the next ellipse, before it is placed --\n"
        "same as shift+scrolling on canvas."));
    dt_gui_box_add(props_box, feather);

    GtkWidget *rotation =
      _pending_conf_slider_new(module, DT_MASKS_CONF(form->type, ellipse, rotation),
                               _blend_masks_properties[DT_MASKS_PROPERTY_ROTATION].name,
                               _blend_masks_properties[DT_MASKS_PROPERTY_ROTATION].min,
                               _blend_masks_properties[DT_MASKS_PROPERTY_ROTATION].max, 1,
                               _blend_masks_properties[DT_MASKS_PROPERTY_ROTATION].format,
                               _("rotation of the next ellipse, before it is placed --\n"
                                 "same as ctrl+shift+scrolling on canvas."));
    dt_gui_box_add(props_box, rotation);
  }
  else if(kind == DT_MASKS_GRADIENT)
  {
    GtkWidget *compression = _pending_conf_slider_new(
      module, DT_MASKS_CONF(form->type, gradient, compression),
      _blend_masks_properties[DT_MASKS_PROPERTY_COMPRESSION].name, 0.001f, 1.0f, 2, "%",
      _("compression of the next gradient, before it is placed --\n"
        "same as shift+scrolling on canvas."));
    dt_gui_box_add(props_box, compression);

    GtkWidget *curvature = _pending_conf_slider_new(
      module, DT_MASKS_CONF(form->type, gradient, curvature),
      _blend_masks_properties[DT_MASKS_PROPERTY_CURVATURE].name, -2.0f, 2.0f, 2, NULL,
      _("curvature of the next gradient, before it is placed --\n"
        "same as scrolling on canvas."));
    dt_gui_box_add(props_box, curvature);
  }
  else if(kind == DT_MASKS_BRUSH)
  {
    GtkWidget *size = _pending_conf_slider_new(
      module, DT_MASKS_CONF(form->type, brush, border),
      _blend_masks_properties[DT_MASKS_PROPERTY_SIZE].name,
      _blend_masks_properties[DT_MASKS_PROPERTY_SIZE].min,
      _blend_masks_properties[DT_MASKS_PROPERTY_SIZE].max, 2,
      _blend_masks_properties[DT_MASKS_PROPERTY_SIZE].format,
      _("width of the next brush stroke -- same as scrolling on canvas."));
    dt_gui_box_add(props_box, size);

    GtkWidget *hardness = _pending_conf_slider_new(
      module, DT_MASKS_CONF(form->type, brush, hardness),
      _blend_masks_properties[DT_MASKS_PROPERTY_HARDNESS].name,
      _blend_masks_properties[DT_MASKS_PROPERTY_HARDNESS].min,
      _blend_masks_properties[DT_MASKS_PROPERTY_HARDNESS].max, 2,
      _blend_masks_properties[DT_MASKS_PROPERTY_HARDNESS].format,
      _("hardness of the next brush stroke -- same as shift+scrolling on canvas."));
    dt_gui_box_add(props_box, hardness);
  }

#ifdef HAVE_AI
  if(kind == DT_MASKS_OBJECT)
  {
    dt_iop_gui_blend_data_t *bd = module->blend_data;
    float smoothing = 0.0f;
    int cleanup = 0;
    dt_masks_object_creation_get_preview_params(&smoothing, &cleanup);

    bd->pending_ai_smoothing_last = smoothing;
    GtkWidget *sm = dt_bauhaus_slider_new_with_range(
      module, _blend_masks_properties[DT_MASKS_PROPERTY_SMOOTHING].min,
      _blend_masks_properties[DT_MASKS_PROPERTY_SMOOTHING].max, 0, smoothing, 2);
    dt_bauhaus_widget_set_label(
      sm, N_("blend"), _blend_masks_properties[DT_MASKS_PROPERTY_SMOOTHING].name);
    dt_bauhaus_slider_set_format(
      sm, _blend_masks_properties[DT_MASKS_PROPERTY_SMOOTHING].format);
    dt_bauhaus_slider_set_digits(sm, 2);
    gtk_widget_set_tooltip_text(
      sm, _("how closely the traced outline follows the AI selection's raw edge.\n"
            "lower: a tighter, more angular fit to the selection.\n"
            "higher: a looser fit with smoother, more rounded corners.\n"
            "same as scrolling on the canvas while drawing."));
    g_object_set_data(G_OBJECT(sm), "dt-prop",
                      GINT_TO_POINTER(DT_MASKS_PROPERTY_SMOOTHING));
    g_signal_connect(G_OBJECT(sm), "value-changed",
                     G_CALLBACK(_pending_ai_slider_changed), module);
    dt_gui_box_add(props_box, sm);
    bd->pending_ai_smoothing_slider = sm;

    bd->pending_ai_cleanup_last = (float)cleanup;
    GtkWidget *cl = dt_bauhaus_slider_new_with_range(
      module, _blend_masks_properties[DT_MASKS_PROPERTY_CLEANUP].min,
      _blend_masks_properties[DT_MASKS_PROPERTY_CLEANUP].max, 0, (float)cleanup, 0);
    dt_bauhaus_widget_set_label(cl, N_("blend"),
                                _blend_masks_properties[DT_MASKS_PROPERTY_CLEANUP].name);
    dt_bauhaus_slider_set_format(
      cl, _blend_masks_properties[DT_MASKS_PROPERTY_CLEANUP].format);
    gtk_widget_set_tooltip_text(
      cl,
      _("discards small, stray outline fragments below this size (in traced pixels).\n"
        "higher: removes more small islands/holes, at the risk of dropping\n"
        "genuinely small parts of the selection.\n"
        "same as shift+scrolling on the canvas while drawing."));
    g_object_set_data(G_OBJECT(cl), "dt-prop",
                      GINT_TO_POINTER(DT_MASKS_PROPERTY_CLEANUP));
    g_signal_connect(G_OBJECT(cl), "value-changed",
                     G_CALLBACK(_pending_ai_slider_changed), module);
    dt_gui_box_add(props_box, cl);
    bd->pending_ai_cleanup_slider = cl;
  }
#endif

  // brush-only: pen-pressure sensitivity and stroke-simplification smoothing.
  // Unlike every other property here, these are *not* per-shape parameters --
  // they are global preferences (conf keys "pressure_sensitivity"/
  // "brush_smoothing") that only have any effect while the stroke is still
  // being captured (guipoints pressure handling and the Ramer-Douglas-Peucker
  // simplification that turns it into nodes on commit, both in brush.c). Once
  // the shape is committed its nodes are fixed, so changing either afterwards
  // would be a no-op -- hence they belong only on this pending row, never in
  // the post-commit properties editor (_build_props_row_editor never builds
  // them). dt_gui_preferences_enum binds straight to the conf key itself (no
  // per-shape modify_property/delta plumbing needed, unlike the AI sliders
  // above), so no extra state or "sync back" call is required. Pass the real
  // module as the action (like every other control on this row) rather than
  // NULL: dt_gui_preferences_enum's alignment/label-rendering path is keyed
  // off whether an action was given (NULL flips it into the label-less,
  // left-aligned "standalone widget with its own external GtkLabel" mode
  // Preferences-dialog grids use) -- without a real module the row's own
  // dt_bauhaus_widget_set_label call below still stores the label text, but
  // it never gets drawn.
  if(kind == DT_MASKS_BRUSH)
  {
    if(darktable.gui->have_pen_pressure)
    {
      GtkWidget *pressure =
        dt_gui_preferences_enum(DT_ACTION(module), "pressure_sensitivity");
      dt_bauhaus_widget_set_label(pressure, N_("blend"), N_("pressure"));
      dt_gui_box_add(props_box, pressure);
    }

    GtkWidget *smoothing = dt_gui_preferences_enum(DT_ACTION(module), "brush_smoothing");
    dt_bauhaus_widget_set_label(smoothing, N_("blend"), N_("smoothing"));
    dt_gui_box_add(props_box, smoothing);
  }

  // with "shape properties in subpanel" the row keeps only its header, in the
  // group the shape lands in, and the controls go to the subpanel, which the
  // rebuild building this row fills right after (see _props_panel_sync)
  if(_shape_props_subpanel())
  {
    gtk_widget_show_all(props_box);
    ((dt_iop_gui_blend_data_t *)module->blend_data)->pending_props_box = props_box;
  }
  else
    dt_gui_box_add(row_vbox, props_box);

  GtkWidget *pending_evbox = gtk_event_box_new();
  gtk_event_box_set_visible_window(GTK_EVENT_BOX(pending_evbox), TRUE);
  g_signal_connect(G_OBJECT(pending_evbox), "button-press-event",
                   G_CALLBACK(_pending_shape_click), NULL);
  g_signal_connect(G_OBJECT(pending_evbox), "button-release-event",
                   G_CALLBACK(_pending_shape_click), NULL);
  gtk_container_add(GTK_CONTAINER(pending_evbox), row_vbox);

  gtk_widget_show_all(pending_evbox);
  return pending_evbox;
}

// The event box wrapping a group header -- real or staged (empty) -- carrying
// the click and drag-and-drop wiring both kinds share. The drop-target list,
// the drag action, the drag-begin handler, and the tags a ctrl+click rename and
// the solo dimming look the header up by are identical for both; only which
// handlers receive the events, and what payload the header drags, differ.
//
// Built in one place so the two cannot drift apart. This skeleton is precisely
// the kind of code where a fix made to one header kind and not the other goes
// unnoticed: nothing about a drop-target list being one entry short is visible
// until someone drags the right thing onto the wrong header.
//
// `source_targets`/`drag_get` NULL means "not a drag source" -- a lone group has
// nowhere to reorder to. The caller still connects drag-motion/drag-leave
// itself: the two kinds deliberately highlight different widgets (a real group
// highlights its whole block so the group-reorder insertion line spans its full
// body; a staged one has only its header row), and for a real group that widget
// does not exist yet at this point.
static GtkWidget *_make_group_header_evbox(dt_iop_module_t *module,
                                           GtkWidget *hdr,
                                           GtkWidget *lbl_box,
                                           GCallback press,
                                           GCallback release,
                                           GCallback drag_received,
                                           const GtkTargetEntry *source_targets,
                                           GCallback drag_get)
{
  GtkWidget *evbox = gtk_event_box_new();
  gtk_event_box_set_visible_window(GTK_EVENT_BOX(evbox), TRUE);
  gtk_container_add(GTK_CONTAINER(evbox), hdr);

  // ctrl+click rename finds the title by this tag (see _group_header_press /
  // _empty_header_press)
  g_object_set_data(G_OBJECT(evbox), "title-label-box", lbl_box);
  // solo dimming must reach the header row itself, never an enclosing block --
  // the member rows already dim individually, so dimming a block would
  // double-dim them (see _apply_group_header_dimming)
  g_object_set_data(G_OBJECT(evbox), "group-header-widget", hdr);

  // g_signal_connect_data, not g_signal_connect: the checked macro only accepts
  // a literal G_CALLBACK(func), not a GCallback variable (same reason as
  // _make_op_combo's own note)
  g_signal_connect_data(G_OBJECT(evbox), "button-press-event", press, module, NULL, 0);
  g_signal_connect_data(G_OBJECT(evbox), "button-release-event", release, module, NULL,
                        0);

  // a header accepts a whole group (reorder), a single shape (move it into this
  // group) and an empty group (reorder) -- one target list covers all three
  gtk_drag_dest_set(evbox, GTK_DEST_DEFAULT_MOTION | GTK_DEST_DEFAULT_DROP, _mask_hdr_dnd,
                    G_N_ELEMENTS(_mask_hdr_dnd), GDK_ACTION_MOVE);
  g_signal_connect_data(G_OBJECT(evbox), "drag-data-received", drag_received, module,
                        NULL, 0);

  if(source_targets && drag_get)
  {
    // also a drag source for its own reorder, in addition to the grip handle in
    // column 0 -- grabbing anywhere on the row moves the group
    gtk_drag_source_set(evbox, GDK_BUTTON1_MASK, source_targets, 1, GDK_ACTION_MOVE);
    g_signal_connect_data(G_OBJECT(evbox), "drag-data-get", drag_get, NULL, NULL, 0);
    g_signal_connect(G_OBJECT(evbox), "drag-begin", G_CALLBACK(_group_drag_begin),
                     module);
  }
  return evbox;
}

// invert a single element's mask polarity (the drag handle's ctrl+click
// behaviour, extracted so the "invert selected element" shortcut can share it).
static void _invert_element(dt_iop_module_t *module, const dt_mask_id_t id)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  dt_masks_point_group_t *pt = grp ? _group_point(grp, id) : NULL;
  if(!pt) return;
  pt->state ^= DT_MASKS_STATE_INVERSE;
  dt_print(DT_DEBUG_MASKS, "[masks] form %d inverse=%d", id,
           !!(pt->state & DT_MASKS_STATE_INVERSE));
  dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  // update this row's own state in place -- a full rebuild here would tear
  // down and recreate the whole list (and re-dock the parametric editor if one
  // is open), which visibly flashes the panel for what is just one bit.
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  GtkWidget *row_vbox = _masks_row_widget(bd, id);
  // this also flips a parametric row's own slider markers, which carry the
  // polarity a shape row shows on its handle icon (see _update_shape_row_state)
  _update_shape_row_state(bd, row_vbox, pt);
}

// GtkCheckMenuItem does not auto-close its parent popup on toggle the way a
// plain GtkMenuItem's "activate" does (by design -- it lets a settings-style
// menu stay open across several checkbox flips) -- but every check item in
// *this* menu (disable/solo/solo-edit/invert) is a one-shot
// action, not a settings panel, and is meant to close the menu like every
// other entry here (see _row_click_press's right-click branch and
// _shape_menu_closed, which the menu's own "hide" signal drives). Without
// this, "hide" simply never fired for these four items -- the menu stayed
// open until a later, unrelated click dismissed it -- which is why deferred
// auto-expand-on-close only ever appeared to work for whichever item
// happened to be clicked last before that unrelated dismissal.
static void _shape_act_disable(GSimpleAction *action, GVariant *param, gpointer u)
{
  GtkWidget *anchor = GTK_WIDGET(u);
  dt_iop_module_t *module = g_object_get_data(G_OBJECT(anchor), "module");
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(anchor), "shape_act_id"));
  if(darktable.gui->active_popover_menu)
    gtk_popover_popdown(GTK_POPOVER(darktable.gui->active_popover_menu));
  if(module) _toggle_element_disable(module, id);
}

static void _shape_act_solo(GSimpleAction *action, GVariant *param, gpointer u)
{
  GtkWidget *anchor = GTK_WIDGET(u);
  dt_iop_module_t *module = g_object_get_data(G_OBJECT(anchor), "module");
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(anchor), "shape_act_id"));
  if(darktable.gui->active_popover_menu)
    gtk_popover_popdown(GTK_POPOVER(darktable.gui->active_popover_menu));
  if(module) _toggle_solo_form(module, id);
}

static void _shape_act_invert(GSimpleAction *action, GVariant *param, gpointer u)
{
  GtkWidget *anchor = GTK_WIDGET(u);
  dt_iop_module_t *module = g_object_get_data(G_OBJECT(anchor), "module");
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(anchor), "shape_act_id"));
  if(darktable.gui->active_popover_menu)
    gtk_popover_popdown(GTK_POPOVER(darktable.gui->active_popover_menu));
  if(module) _invert_element(module, id);
}

static void _shape_act_rename(GSimpleAction *action, GVariant *param, gpointer u)
{
  GtkWidget *anchor = GTK_WIDGET(u);
  dt_iop_module_t *module = g_object_get_data(G_OBJECT(anchor), "module");
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(anchor), "shape_act_id"));
  GtkWidget *evbox = g_object_get_data(G_OBJECT(anchor), "shape_act_evbox");
  if(darktable.gui->active_popover_menu)
    gtk_popover_popdown(GTK_POPOVER(darktable.gui->active_popover_menu));
  if(evbox && module) _start_rename_element(evbox, module, id);
}

// the panel's way into an AI object's paths and back out (see _step_object)
static void _shape_act_edit_paths(GSimpleAction *action, GVariant *param, gpointer u)
{
  GtkWidget *anchor = GTK_WIDGET(u);
  dt_iop_module_t *module = g_object_get_data(G_OBJECT(anchor), "module");
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(anchor), "shape_act_id"));
  if(darktable.gui->active_popover_menu)
    gtk_popover_popdown(GTK_POPOVER(darktable.gui->active_popover_menu));
  if(!module) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  const gboolean inside = _entered_object() == id;
  // the paths are picked on the canvas, so going in turns editing on there
  if(!inside && bd && bd->masks_edit
     && !gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(bd->masks_edit)))
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_edit), TRUE);
  _step_object(module, inside ? INVALID_MASKID : id);
}

// give this module its own copy of a linked shape or AI object: the other
// modules keep the original
static void _unlink_element(dt_iop_module_t *module,
                            const dt_mask_id_t id,
                            dt_masks_point_group_t *pt)
{
  dt_masks_clear_form_gui(darktable.develop);
  const dt_mask_id_t nid = _model_unlink_form_point(module, id, pt);
  if(!dt_is_valid_maskid(nid)) return;
  dt_print(DT_DEBUG_MASKS, "[masks] form %d unlinked in '%s' as %d", id, module->op, nid);
  dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
  _queue_masks_list_rebuild(module);
  _queue_link_peers_rebuild(module);
  _refresh_canvas_edit(module);
}

static void _shape_act_unlink(GSimpleAction *action, GVariant *param, gpointer u)
{
  GtkWidget *anchor = GTK_WIDGET(u);
  dt_iop_module_t *module = g_object_get_data(G_OBJECT(anchor), "module");
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(anchor), "shape_act_id"));
  // the row the menu was opened on, so a shape held twice unlinks the one the
  // user clicked (see _build_shape_actions_menu)
  dt_masks_point_group_t *pt = g_object_get_data(G_OBJECT(anchor), "shape_act_point");
  if(darktable.gui->active_popover_menu)
    gtk_popover_popdown(GTK_POPOVER(darktable.gui->active_popover_menu));
  if(module) _unlink_element(module, id, pt);
}

// a raster element's mask is edited where it is made: its source module gets
// the focus, expanded, with its mask on the canvas
static void _edit_raster_source(const dt_masks_form_t *form)
{
  dt_iop_module_t *src = dt_masks_raster_source(form);
  if(!src) return;
  if(!src->expanded)
    dt_iop_gui_set_expanded(src, TRUE, dt_conf_get_bool("darkroom/ui/single_module"));
  dt_iop_request_focus(src);
  dt_iop_gui_blend_data_t *sbd = src->blend_data;
  if(sbd && _module_mask_group(src))
  {
    sbd->masks_shown = DT_MASKS_EDIT_FULL;
    dt_masks_set_edit_mode(src, DT_MASKS_EDIT_FULL);
  }
}

static void _shape_act_edit_source(GSimpleAction *action, GVariant *param, gpointer u)
{
  GtkWidget *anchor = GTK_WIDGET(u);
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(anchor), "shape_act_id"));
  if(darktable.gui->active_popover_menu)
    gtk_popover_popdown(GTK_POPOVER(darktable.gui->active_popover_menu));
  _edit_raster_source(dt_masks_get_from_id(darktable.develop, id));
}

void dt_iop_gui_blend_module_renamed(dt_iop_module_t *module)
{
  // raster elements following this module's name show it in other panels
  if(module && darktable.develop) _queue_link_peers_rebuild(module);
}

static void _shape_act_delete(GSimpleAction *action, GVariant *param, gpointer u)
{
  GtkWidget *anchor = GTK_WIDGET(u);
  dt_iop_module_t *module = g_object_get_data(G_OBJECT(anchor), "module");
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(anchor), "shape_act_id"));
  if(darktable.gui->active_popover_menu)
    gtk_popover_popdown(GTK_POPOVER(darktable.gui->active_popover_menu));
  if(module) _delete_single_shape(module, id);
}

// the member point a shape actions menu acts on: the row's own, where it has one
static const dt_masks_point_group_t *_shape_act_point(GtkWidget *anchor,
                                                      dt_iop_module_t *module,
                                                      const dt_mask_id_t id)
{
  const dt_masks_point_group_t *pt = g_object_get_data(G_OBJECT(anchor), "shape_act_point");
  return pt ? pt : _group_point(_module_mask_group(module), id);
}

static void _shape_act_compose(GSimpleAction *action, GVariant *param, gpointer u)
{
  GtkWidget *anchor = GTK_WIDGET(u);
  dt_iop_module_t *module = g_object_get_data(G_OBJECT(anchor), "module");
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(anchor), "shape_act_id"));
  if(darktable.gui->active_popover_menu)
    gtk_popover_popdown(GTK_POPOVER(darktable.gui->active_popover_menu));
  if(module)
    _compose(module, _shape_act_point(anchor, module, id),
             (dt_masks_state_t)g_variant_get_int32(param));
}

// the "compose" submenu of member `pt`, or NULL where it cannot be composed:
// a path of an AI object moves with the object
static GMenu *_shape_compose_menu(dt_masks_form_t *grp, const dt_masks_point_group_t *pt)
{
  dt_masks_form_t *owner = NULL;
  GList *node = pt ? _point_node_at(grp, pt, &owner, 0) : NULL;
  if(!node || !(owner->type & DT_MASKS_GROUP) || (owner->type & DT_MASKS_OBJECT)) return NULL;
  GList *marker = _group_marker_node(node);
  const int within = marker ? ((dt_masks_point_group_t *)marker->data)->state & DT_MASKS_STATE_WITHIN
                            : 0;
  // composing the base with its group's operator adds a member, as it does
  // anywhere in a group whose members fold in any order
  const gboolean ordered =
    within & (DT_MASKS_STATE_WITHIN_DIFFERENCE | DT_MASKS_STATE_WITHIN_EXCLUSION);
  return _compose_menu("masks_shape_act.compose", within,
                       ordered && marker && marker->next != node);
}

static void _build_shape_actions_menu(GtkWidget *anchor,
                                      dt_iop_module_t *module,
                                      const dt_mask_id_t id,
                                      GtkWidget *handle,
                                      GtkWidget *evbox)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  // the exact reference this menu was opened on: one mask can hold the same
  // shape twice, and then the form id alone names neither row. The anchor is
  // the row's own box, which carries the member point it was built from (see
  // _make_shape_row); falling back to the first match keeps every other row
  // behaving as before
  dt_masks_point_group_t *row_pt = g_object_get_data(G_OBJECT(anchor), "row-point");
  const dt_masks_point_group_t *pt = row_pt ? row_pt
                                     : grp ? _group_point(grp, id) : NULL;

  const gboolean elem_disabled = pt && (pt->state & DT_MASKS_STATE_DISABLE);
  const gboolean elem_inverted = pt && (pt->state & DT_MASKS_STATE_INVERSE);

  g_object_set_data(G_OBJECT(anchor), "module", module);
  g_object_set_data(G_OBJECT(anchor), "shape_act_id", GINT_TO_POINTER(id));
  g_object_set_data(G_OBJECT(anchor), "shape_act_point", row_pt);
  g_object_set_data(G_OBJECT(anchor), "shape_act_evbox", evbox);

  GSimpleActionGroup *sag = g_simple_action_group_new();
  GActionMap *map = G_ACTION_MAP(sag);

  GSimpleAction *act_dis = g_simple_action_new_stateful("disable", NULL, g_variant_new_boolean(elem_disabled));
  g_signal_connect(act_dis, "activate", G_CALLBACK(_shape_act_disable), anchor);
  g_action_map_add_action(map, G_ACTION(act_dis));

  GSimpleAction *act_solo = g_simple_action_new_stateful("solo", NULL, g_variant_new_boolean(bd->solo_formid == id));
  g_signal_connect(act_solo, "activate", G_CALLBACK(_shape_act_solo), anchor);
  g_action_map_add_action(map, G_ACTION(act_solo));

  GSimpleAction *act_inv = g_simple_action_new_stateful("invert", NULL, g_variant_new_boolean(elem_inverted));
  g_signal_connect(act_inv, "activate", G_CALLBACK(_shape_act_invert), anchor);
  g_action_map_add_action(map, G_ACTION(act_inv));

  GActionEntry action_entries[] =
  {
    { "rename",      _shape_act_rename,      NULL, NULL },
    { "edit_source", _shape_act_edit_source, NULL, NULL },
    { "compose",     _shape_act_compose,     "i",  NULL },
    { "unlink",      _shape_act_unlink,      NULL, NULL },
    { "edit_paths",  _shape_act_edit_paths,  NULL, NULL },
    { "delete",      _shape_act_delete,      NULL, NULL },
  };
  g_action_map_add_action_entries(map, action_entries, G_N_ELEMENTS(action_entries), anchor);

  gtk_widget_insert_action_group(anchor, "masks_shape_act", G_ACTION_GROUP(sag));

  const dt_masks_form_t *elem = dt_masks_get_from_id(darktable.develop, id);
  // shared with another module, or held twice by this mask: either way this
  // row shows the same shape as some other row (see _model_form_uses_in_mask)
  const gboolean linked =
    _model_form_is_linked(elem) || _model_form_uses_in_mask(module, id) > 1;

  GMenu *menu = g_menu_new();

  GMenu *sec_vis = g_menu_new();
  g_menu_append(sec_vis, _("disable"), "masks_shape_act.disable");
  if(!elem_disabled)
    g_menu_append(sec_vis, _("solo"), "masks_shape_act.solo");
  g_menu_append_section(menu, _("visibility"), G_MENU_MODEL(sec_vis));
  g_object_unref(sec_vis);

  if(!elem_disabled)
  {
    GMenu *sec_ops = g_menu_new();
    g_menu_append(sec_ops, _("invert"), "masks_shape_act.invert");
    GMenu *compose = _shape_compose_menu(grp, pt);
    if(compose) _append_compose_submenu(sec_ops, compose);
    g_menu_append_section(menu, _("mask operations"), G_MENU_MODEL(sec_ops));
    g_object_unref(sec_ops);
  }

  GMenu *sec_edit = g_menu_new();
  const dt_iop_module_t *raster_src = dt_masks_raster_source(elem);
  if(raster_src)
  {
    gchar *src_name = dt_history_item_get_name(raster_src);
    gchar *tip = g_strdup_printf(_("focus %s and edit its mask on the canvas"), src_name);
    GMenuItem *it = g_menu_item_new(_("edit source mask"), "masks_shape_act.edit_source");
    g_menu_item_set_attribute(it, "tooltip", "s", tip);
    g_menu_append_item(sec_edit, it);
    g_object_unref(it);
    g_free(tip);
    g_free(src_name);
  }
  g_menu_append(sec_edit, _("rename"), "masks_shape_act.rename");
  if(linked)
  {
    GMenuItem *it = g_menu_item_new(_("unlink"), "masks_shape_act.unlink");
    g_menu_item_set_attribute(it, "tooltip", "s",
      _("give this row its own copy, so editing it no longer changes the other"
        " rows and modules it is linked with"));
    g_menu_append_item(sec_edit, it);
    g_object_unref(it);
  }
  // a single-path object already acts as that one path
  if(elem && (elem->type & DT_MASKS_OBJECT) && _object_path_count(elem) >= 2)
  {
    const gboolean inside = _entered_object() == id;
    GMenuItem *it = g_menu_item_new(inside ?_("stop editing individual paths")
                                           : _("edit individual paths"),
                                    "masks_shape_act.edit_paths");
    g_menu_item_set_attribute(it, "tooltip", "s",
      _("edit and remove the AI object's paths one by one on the canvas,"
        " like double-clicking it there"));
    g_menu_append_item(sec_edit, it);
    g_object_unref(it);
  }
  g_menu_append(sec_edit, _("delete"), "masks_shape_act.delete");
  g_menu_append_section(menu, _("edit"), G_MENU_MODEL(sec_edit));
  g_object_unref(sec_edit);

  darktable.gui->active_popover_menu = dt_gui_popover_menu_from_model(anchor, menu);
  g_object_unref(menu);
}

// human-readable name for a shape kind (the _form_kind bit), singular or plural.
// Used to label the same-kind element clusters ("3 circles").
static const char *_kind_name(const guint kind, const gboolean plural)
{
  switch(kind)
  {
  case DT_MASKS_CIRCLE: return plural ? _("circles") : _("circle");
  case DT_MASKS_ELLIPSE: return plural ? _("ellipses") : _("ellipse");
  case DT_MASKS_PATH: return plural ? _("paths") : _("path");
  case DT_MASKS_GRADIENT: return plural ? _("gradients") : _("gradient");
  case DT_MASKS_BRUSH: return plural ? _("brushes") : _("brush");
  case DT_MASKS_PARAMETRIC: return plural ? _("parametric masks") : _("parametric mask");
  case DT_MASKS_RASTER: return plural ? _("raster masks") : _("raster mask");
#ifdef HAVE_AI
  case DT_MASKS_OBJECT: return plural ? _("AI objects") : _("AI object");
#endif
  default: return plural ? _("shapes") : _("shape");
  }
}

// same glyph the add-toolbar button for this kind uses, so a row's icon
// matches the icon the user picked it from
static DTGTKCairoPaintIconFunc _kind_icon_paint(const guint kind)
{
  switch(kind)
  {
  case DT_MASKS_CIRCLE: return dtgtk_cairo_paint_masks_circle;
  case DT_MASKS_ELLIPSE: return dtgtk_cairo_paint_masks_ellipse;
  case DT_MASKS_PATH: return dtgtk_cairo_paint_masks_path;
  case DT_MASKS_GRADIENT: return dtgtk_cairo_paint_masks_gradient;
  case DT_MASKS_BRUSH: return dtgtk_cairo_paint_masks_brush;
  case DT_MASKS_PARAMETRIC: return dtgtk_cairo_paint_masks_parametric;
  case DT_MASKS_RASTER: return dtgtk_cairo_paint_masks_raster;
#ifdef HAVE_AI
  case DT_MASKS_OBJECT: return dtgtk_cairo_paint_masks_object;
#endif
  default: return NULL;
  }
}

static void _pack_group_elements(dt_iop_module_t *module,
                                 dt_masks_form_t *grp,
                                 GtkWidget *container,
                                 GList *fids,
                                 GList *group_formids,
                                 GtkWidget *group_frame);
static void _pack_subgroup(dt_iop_module_t *module, dt_masks_form_t *sub, GtkWidget *box);
static gboolean _nested_as_group(const dt_masks_point_group_t *pt, const dt_masks_form_t *form);

// --- drag handle ----------------------------------------------------------
// A small "grip" the user grabs to reorder a row. It is a plain *windowed* event
// box used directly as the drag source, so the button press lands on its own
// window and GTK's drag gesture always arms. Dragging a whole header/row instead
// is unreliable: the child label/button windows swallow the press before the
// row's drag source can see it.
//
// When the row/group's kind maps to a single icon (a shape row, a same-kind
// cluster header), that icon is drawn in the handle instead of the generic
// grip dots -- one slot doing double duty as both the drag affordance and the
// "what kind is this" indicator, instead of two separate icons competing for
// the same corner. Rows whose kind doesn't map to one icon (a real group can
// mix shape kinds; an empty group has none yet) keep the plain grip.
typedef struct _handle_icon_t
{
  DTGTKCairoPaintIconFunc paint;
} _handle_icon_t;

static gboolean _drag_handle_draw(GtkWidget *w, cairo_t *cr, gpointer user_data)
{
  const gboolean disabled =
    GPOINTER_TO_INT(g_object_get_data(G_OBJECT(w), "handle-disabled"));
  GtkAllocation a;
  gtk_widget_get_allocation(w, &a);
  GdkRGBA c;
  GtkStyleContext *ctx = gtk_widget_get_style_context(w);
  const GtkStateFlags state = gtk_widget_get_state_flags(w);

  // this widget is app-paintable (see _make_drag_handle), so its normal CSS
  // background is never drawn automatically -- paint it explicitly, so
  // .mask-list-handle-inverted (darktable.css) can swap this handle to a
  // light background / dark foreground, reading as a true colour inversion
  // rather than a colour tint over the icon.
  gtk_render_background(ctx, cr, 0, 0, a.width, a.height);
  gtk_style_context_get_color(ctx, state, &c);

  const _handle_icon_t *hi = g_object_get_data(G_OBJECT(w), "handle-icon");
  if(hi && hi->paint)
  {
    // a meaningful type icon needs to actually read, unlike the subtle grip dots
    cairo_set_source_rgba(cr, c.red, c.green, c.blue, c.alpha * (disabled ? 0.35 : 0.85));
    const gint pad = DT_PIXEL_APPLY_DPI(1);
    hi->paint(cr, pad, pad, a.width - 2 * pad, a.height - 2 * pad, 0, NULL);
    return FALSE;
  }

  // a disabled handle is drawn faint so it reads as "present but inactive"
  cairo_set_source_rgba(cr, c.red, c.green, c.blue, c.alpha * (disabled ? 0.16 : 0.5));
  const double r = MAX(1.0, DT_PIXEL_APPLY_DPI(1.1));
  const double dx = DT_PIXEL_APPLY_DPI(2.3);
  const double dy = DT_PIXEL_APPLY_DPI(3.2);
  const double cx = a.width * 0.5;
  const double cy = a.height * 0.5;
  for(int ix = -1; ix <= 1; ix += 2)
    for(int iy = -1; iy <= 1; iy++)
    {
      cairo_arc(cr, cx + ix * dx, cy + iy * dy, r, 0, 2.0 * M_PI);
      cairo_fill(cr);
    }
  return FALSE;
}

// build a drag-handle column. A glyph is always drawn (so columns line up and the
// affordance is visible on every reorderable row type): `kind_paint` when the row's
// kind maps to one icon, otherwise the generic grip dots (see _drag_handle_draw).
// When `enabled` is false it is drawn faint/disabled and is not a drag source.
// `tooltip` explains how to drag (enabled) or why the row cannot be moved/is not
// draggable (disabled). The caller wires the drag source + payload separately.
static GtkWidget *_make_drag_handle(DTGTKCairoPaintIconFunc kind_paint,
                                    gboolean enabled,
                                    const char *tooltip)
{
  GtkWidget *eb = gtk_event_box_new();
  gtk_event_box_set_visible_window(GTK_EVENT_BOX(eb), TRUE);
  gtk_widget_set_app_paintable(eb, TRUE);
  gtk_widget_set_size_request(eb, DT_PIXEL_APPLY_DPI(18), DT_PIXEL_APPLY_DPI(18));
  gtk_widget_set_valign(eb, GTK_ALIGN_CENTER);
  // a rounded plate behind every handle, always -- not just when inverted --
  // so a blocky icon (e.g. the raster mask checkerboard) reads as a rounded
  // chip like the rest of the panel instead of a bare rectangle (see
  // .mask-list-handle in darktable.css; _drag_handle_draw paints this
  // background itself since the handle is app-paintable)
  dt_gui_add_class(eb, "mask-list-handle");
  if(!enabled) g_object_set_data(G_OBJECT(eb), "handle-disabled", GINT_TO_POINTER(1));
  if(kind_paint)
  {
    _handle_icon_t *hi = g_malloc(sizeof(_handle_icon_t));
    hi->paint = kind_paint;
    g_object_set_data_full(G_OBJECT(eb), "handle-icon", hi, g_free);
  }
  if(tooltip) gtk_widget_set_tooltip_text(eb, tooltip);
  g_signal_connect(G_OBJECT(eb), "draw", G_CALLBACK(_drag_handle_draw), NULL);
  return eb;
}

// a parametric row's own lead handle: the channel code itself (e.g. "hz",
// "Cz") instead of a generic "this is a parametric mask" glyph -- every
// channel used the same icon, so it carried no information a glance at the
// row didn't already need the name for, and it duplicated the separate
// channel-code badge that used to sit next to it. This is a plain label (not
// app-paintable like _make_drag_handle's icon version), so .mask-list-handle
// / .mask-list-handle-inverted's background+text colour swap applies to it
// via ordinary CSS with no custom draw code needed.
static GtkWidget *_make_channel_handle(const char *code, const char *tooltip)
{
  GtkWidget *eb = gtk_event_box_new();
  gtk_event_box_set_visible_window(GTK_EVENT_BOX(eb), TRUE);
  // same square footprint as _make_drag_handle's icon plate, regardless of
  // how many characters the channel code has -- a one-off "hz" or "Cz" chip
  // must not read as a wider/differently-shaped column than every other
  // row's icon handle
  gtk_widget_set_size_request(eb, DT_PIXEL_APPLY_DPI(18), DT_PIXEL_APPLY_DPI(18));
  gtk_widget_set_valign(eb, GTK_ALIGN_CENTER);
  dt_gui_add_class(eb, "mask-list-handle");
  dt_gui_add_class(eb, "mask-channel-handle");
  GtkWidget *lbl = gtk_label_new(code);
  gtk_label_set_xalign(GTK_LABEL(lbl), 0.5f);
  gtk_label_set_justify(GTK_LABEL(lbl), GTK_JUSTIFY_CENTER);
  gtk_widget_set_halign(lbl, GTK_ALIGN_CENTER);
  gtk_widget_set_valign(lbl, GTK_ALIGN_CENTER);
  gtk_container_add(GTK_CONTAINER(eb), lbl);
  if(tooltip) gtk_widget_set_tooltip_text(eb, tooltip);
  return eb;
}

// ---- always-expanded per-row parametric mask editor -----------------------
// Every parametric channel row gets its own permanently-visible input/output
// slider pair, boost-factor slider and picker buttons, bound directly to that
// form's own dt_masks_point_parametric_t -- instead of the single editor
// widget set (bd->blendif_box et al, still used for classic/legacy
// multi-channel editing only) reparented under whichever one row was being
// edited. See _build_param_row_editor below. (dt_masks_param_row_editor_t
// itself is declared earlier in this file, near the other forward decls, so
// early functions like _masks_param_inout_toggled can use it too.)

// the single-channel form this editor owns, or NULL if it no longer exists
// (e.g. deleted from under it before the next rebuild tears the row down).
static dt_masks_point_parametric_t *
_param_row_point(const dt_masks_param_row_editor_t *ed)
{
  dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, ed->formid);
  if(!form || !(form->type & DT_MASKS_PARAMETRIC) || !form->points) return NULL;
  return form->points->data;
}

// is this row's own shape inverted? (the per-shape ctrl+click invert,
// DT_MASKS_STATE_INVERSE on its group point) -- flips the displayed slider polarity to
// match, same as the legacy shared editor's _param_single_inverted, but keyed on this
// row's own formid.
static gboolean _param_row_inverted(dt_iop_module_t *module, const dt_mask_id_t formid)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  const dt_masks_point_group_t *gp = grp ? _group_point(grp, formid) : NULL;
  return gp && (gp->state & DT_MASKS_STATE_INVERSE);
}

gboolean _param_channel_is_used(const dt_masks_point_parametric_t *p,
                                              const dt_iop_gui_blendif_channel_t *channel,
                                              const int in_out)
{
  if(!p || !channel) return FALSE;
  const int ch = channel->param_channels[in_out];
  const float *const r = &p->blendif_parameters[4 * ch];
  const gboolean is_default_range =
    (r[0] == 0.0f && r[1] == 0.0f && r[2] == 1.0f && r[3] == 1.0f);
  const gboolean bit_active = (p->blendif & (1u << ch)) != 0;
  return !is_default_range || bit_active;
}

// show/hide this row's input slider, output slider and boost-factor slider:
// - expanded (p->in_out != 0): show both input and output sliders + boost slider
// - collapsed (p->in_out == 0):
//     * both input & output used: show both input and output sliders (hide boost factor
//     only)
//     * only output used: show only output slider
//     * only input used (or no-op / default): show only input slider
// Which of a parametric row's controls are shown, from the channel's own state.
// A collapsed row adapts to which sub-ranges the user has actually touched, so
// an untouched channel does not show a slider that says nothing; an expanded
// row always shows both. Split from the widget update below so the rule can be
// tested without a row -- see test_flexi_panel.c.
//
// `opacity_slider_enabled` is the "show opacity slider in expanded elements"
// option. A parametric row's opacity control follows the same rule every other
// element row's does: the compact value in the row header is always there, and
// the full slider leading the expanded controls appears only when the option
// asks for it -- so, here, only when the row is expanded *and* it is on.
dt_masks_param_vis_t _model_param_row_visibility(const gboolean expanded,
                                                 const gboolean in_used,
                                                 const gboolean out_used,
                                                 const gboolean boost_enabled,
                                                 const gboolean opacity_slider_enabled)
{
  dt_masks_param_vis_t v = { TRUE, FALSE, FALSE, FALSE, FALSE };

  v.opacity = expanded && opacity_slider_enabled;

  if(expanded)
  {
    v.input = TRUE;
    v.output = TRUE;
    v.boost = boost_enabled;
  }
  else if(in_used && out_used)
  {
    v.input = TRUE;
    v.output = TRUE;
  }
  else if(!in_used && out_used)
  {
    v.input = FALSE;
    v.output = TRUE;
  }
  else
  {
    // only input used, or neither used (no-op default state)
    v.input = TRUE;
    v.output = FALSE;
  }

  // the per-sub-range bypass toggles only mean something when both are in play
  v.bypass = in_used && out_used;
  return v;
}

static void _param_slider_fit_eye(dt_masks_param_row_editor_t *ed);

static void _update_param_row_visibility(dt_masks_param_row_editor_t *ed)
{
  const dt_masks_point_parametric_t *p = _param_row_point(ed);
  if(!p) return;
  const dt_iop_gui_blendif_channel_t *channels =
    dt_develop_blendif_channels_for_csp(p->colorspace);
  const dt_iop_gui_blendif_channel_t *channel = channels ? &channels[p->channel] : NULL;

  const gboolean in_used = _param_channel_is_used(p, channel, 0);
  const gboolean out_used = _param_channel_is_used(p, channel, 1);
  const dt_masks_param_vis_t vis =
    _model_param_row_visibility(p->in_out != 0, in_used, out_used,
                                channel && channel->boost_factor_enabled,
                                _opacity_sliders());
  const gboolean show_input = vis.input;
  const gboolean show_output = vis.output;
  const gboolean show_boost = vis.boost;
  const gboolean show_bypass = vis.bypass;

  // the eye and the box it sits in hide on different conditions: the box
  // follows its slider row, while the eye follows show_bypass. The box keeps
  // its width either way, which is what the slider's margin is fitted to (see
  // _param_slider_fit_eye).
  if(ed->input_lbl) gtk_widget_set_visible(ed->input_lbl, show_input);
  if(ed->input_slot) gtk_widget_set_visible(ed->input_slot, show_input);
  if(ed->input_bypass_slot)
    gtk_widget_set_visible(ed->input_bypass_slot, show_input);
  if(ed->input_bypass_btn)
  {
    gtk_widget_set_visible(ed->input_bypass_btn, show_bypass);
    gtk_widget_set_opacity(ed->input_bypass_btn, 1.0);
    gtk_widget_set_sensitive(ed->input_bypass_btn, show_bypass);
  }
  if(ed->output_lbl) gtk_widget_set_visible(ed->output_lbl, show_output);
  if(ed->output_slot) gtk_widget_set_visible(ed->output_slot, show_output);
  if(ed->output_bypass_slot)
    gtk_widget_set_visible(ed->output_bypass_slot, show_output);
  if(ed->output_bypass_btn)
  {
    gtk_widget_set_visible(ed->output_bypass_btn, show_bypass);
    gtk_widget_set_opacity(ed->output_bypass_btn, 1.0);
    gtk_widget_set_sensitive(ed->output_bypass_btn, show_bypass);
  }

  if(ed->sliders_grid)
  {
    gtk_widget_set_visible(ed->sliders_grid, TRUE);
    gtk_widget_queue_resize(ed->sliders_grid);
  }
  if(ed->boost_box)
  {
    gtk_widget_set_visible(ed->boost_box, show_boost);
    gtk_widget_queue_resize(ed->boost_box);
  }
  if(ed->opacity_box)
  {
    gtk_widget_set_visible(ed->opacity_box, vis.opacity);
    gtk_widget_queue_resize(ed->opacity_box);
  }

  // the eye boxes may only just have been shown
  _param_slider_fit_eye(ed);
}

// refresh this row's own slider markers/values/labels/boost-slider display from
// its form's current values -- what the classic tabbed editor's own per-tab
// refresh used to do, scoped to one form's one channel (no bp scratch, no tab).
static void _update_param_row_display(dt_masks_param_row_editor_t *ed)
{
  const dt_masks_point_parametric_t *p = _param_row_point(ed);
  if(!p) return;
  const dt_iop_gui_blendif_channel_t *channels =
    dt_develop_blendif_channels_for_csp(p->colorspace);
  if(!channels) return;
  const dt_iop_gui_blendif_channel_t *channel = &channels[p->channel];
  const gboolean single_inv = _param_row_inverted(ed->module, ed->formid);

  DT_ENTER_GUI_UPDATE();
  for(int in_out = 1; in_out >= 0; in_out--)
  {
    const dt_develop_blendif_channels_t ch = channel->param_channels[in_out];
    dt_iop_gui_blendif_filter_t *sl = &ed->filter[in_out];
    const float *parameters = &p->blendif_parameters[4 * ch];
    const float *defaults =
      &ed->module->default_blendop_params->blendif_parameters[4 * ch];

    // a single-channel row has no polarity control of its own (sl->polarity is
    // NULL, see _build_param_row_editor) -- the shape's own ctrl+click invert
    // (single_inv, also driving the row's handle icon, see _invert_element) is
    // the one and only source of truth for polarity here. p->blendif's own
    // per-channel polarity bit is a leftover from the legacy multi-channel tab
    // editor and must stay at its canonical (non-inverted) default for a
    // single-channel form -- see _add_parametric_channel.
    const int polarity = single_inv ? 0 : 1;
    dtgtk_gradient_slider_multivalue_set_marker(sl->slider,
                                                polarity
                                                  ? GRADIENT_SLIDER_MARKER_LOWER_OPEN_BIG
                                                  : GRADIENT_SLIDER_MARKER_UPPER_OPEN_BIG,
                                                0);
    dtgtk_gradient_slider_multivalue_set_marker(
      sl->slider,
      polarity ? GRADIENT_SLIDER_MARKER_UPPER_FILLED_BIG
               : GRADIENT_SLIDER_MARKER_LOWER_FILLED_BIG,
      1);
    dtgtk_gradient_slider_multivalue_set_marker(
      sl->slider,
      polarity ? GRADIENT_SLIDER_MARKER_UPPER_FILLED_BIG
               : GRADIENT_SLIDER_MARKER_LOWER_FILLED_BIG,
      2);
    dtgtk_gradient_slider_multivalue_set_marker(sl->slider,
                                                polarity
                                                  ? GRADIENT_SLIDER_MARKER_LOWER_OPEN_BIG
                                                  : GRADIENT_SLIDER_MARKER_UPPER_OPEN_BIG,
                                                3);

    for(int k = 0; k < 4; k++)
    {
      dtgtk_gradient_slider_multivalue_set_value(sl->slider, parameters[k], k);
      dtgtk_gradient_slider_multivalue_set_resetvalue(sl->slider, defaults[k], k);
    }

    const float boost_factor =
      _get_boost_factor_ex(p->blendif_boost_factors, channels, p->channel, in_out);
    char range_text[4][256];
    for(int k = 0; k < 4; k++)
    {
      channel->scale_print(parameters[k], boost_factor, range_text[k],
                           sizeof(range_text[k]));
      gtk_label_set_text(sl->label[k], range_text[k]);
    }

    // compact mode hides these numeric labels entirely (see
    // _apply_param_row_filter_layout) -- surface the same range values on the
    // slider's own tooltip so hovering it in compact mode loses no information.
    gchar *full_tip =
      g_strdup_printf("%s: %s  %s  %s  %s\n\n%s", in_out ? _("output") : _("input"),
                      range_text[0], range_text[1], range_text[2], range_text[3],
                      _(slider_tooltip[in_out]));
    gtk_widget_set_tooltip_text(GTK_WIDGET(sl->slider), full_tip);
    g_free(full_tip);

    dtgtk_gradient_slider_multivalue_clear_stops(sl->slider);
    for(int k = 0; k < channel->numberstops; k++)
      dtgtk_gradient_slider_multivalue_set_stop(
        sl->slider, channel->colorstops[k].stoppoint, channel->colorstops[k].color);
    dtgtk_gradient_slider_multivalue_set_increment(sl->slider, channel->increment);
  }

  const gboolean boost_enabled = channel->boost_factor_enabled;
  if(boost_enabled)
    dt_bauhaus_slider_set(ed->boost_slider,
                          p->blendif_boost_factors[channel->param_channels[0]]
                            - channel->boost_factor_offset);

  if(ed->input_bypass_btn)
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(ed->input_bypass_btn),
                                 (p->disabled & 1) != 0);
  if(ed->output_bypass_btn)
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(ed->output_bypass_btn),
                                 (p->disabled & 2) != 0);

  DT_LEAVE_GUI_UPDATE();

  _update_param_row_visibility(ed);
}

// commit a blendif edit straight to this row's own form (no module->blend_params
// scratch involved) -- the per-row equivalent of _blendif_commit's parametric branch.
static void _param_form_commit(dt_iop_module_t *module, const dt_mask_id_t formid)
{
  dt_print(DT_DEBUG_MASKS, "[masks] parametric form %d: blendif edit committed", formid);
  dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  (void)module;
}

static void _param_channel_bypass_toggled(GtkToggleButton *btn, gpointer user_data)
{
  DT_GUARD_GUI_UPDATE();
  dt_masks_param_row_editor_t *ed = (dt_masks_param_row_editor_t *)user_data;
  if(!ed) return;
  dt_masks_point_parametric_t *p = _param_row_point(ed);
  if(!p) return;

  const int in_out = (btn == GTK_TOGGLE_BUTTON(ed->output_bypass_btn)) ? 1 : 0;
  const gboolean bypassed = gtk_toggle_button_get_active(btn);

  if(bypassed)
    p->disabled |= (1u << in_out);
  else
    p->disabled &= ~(1u << in_out);

  _param_form_commit(ed->module, ed->formid);
  _update_param_row_display(ed);
  _refresh_lowop_badges(ed->module);

  if(ed->module && ed->module->dev)
  {
    dt_dev_reprocess_all(ed->module->dev);
    dt_control_queue_redraw();
  }
}

// refreshes this row's own numeric labels from `slider`'s current marker
// positions, purely cosmetic -- unlike _param_row_slider_callback below, this
// never writes into the form's own persisted blendif_parameters or touches
// the tooltip/blendif-bit bookkeeping, so it is safe to call for a value that
// may still be discarded (see the hover-preview handler further down, which
// previews a node position without committing it).
static void
_update_param_row_range_labels_preview(dt_masks_param_row_editor_t *ed,
                                       GtkDarktableGradientSlider *slider,
                                       const dt_iop_gui_blendif_channel_t *channel,
                                       const float boost_factor,
                                       const int in_out)
{
  for(int k = 0; k < 4; k++)
  {
    const float value = dtgtk_gradient_slider_multivalue_get_value(slider, k);
    char range_text[256];
    channel->scale_print(value, boost_factor, range_text, sizeof(range_text));
    gtk_label_set_text(ed->filter[in_out].label[k], range_text);
  }
}

static void _param_row_slider_callback(GtkDarktableGradientSlider *slider,
                                       dt_masks_param_row_editor_t *ed)
{
  DT_GUARD_GUI_UPDATE();
  dt_masks_point_parametric_t *p = _param_row_point(ed);
  if(!p) return;
  const dt_iop_gui_blendif_channel_t *channels =
    dt_develop_blendif_channels_for_csp(p->colorspace);
  if(!channels) return;
  const dt_iop_gui_blendif_channel_t *channel = &channels[p->channel];

  const int in_out = (slider == ed->filter[1].slider) ? 1 : 0;
  const dt_develop_blendif_channels_t ch = channel->param_channels[in_out];

  // a manual drag on this row's own slider means the user is done with
  // whatever range this row's picker last set -- turn the picker off so it
  // doesn't keep overwriting the values being dragged on the next pick.
  if(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(ed->colorpicker))
     || gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(ed->colorpicker_set_values)))
    dt_iop_color_picker_reset(ed->module, FALSE);

  float *parameters = &p->blendif_parameters[4 * ch];
  for(int k = 0; k < 4; k++)
    parameters[k] = dtgtk_gradient_slider_multivalue_get_value(slider, k);

  const float boost_factor =
    _get_boost_factor_ex(p->blendif_boost_factors, channels, p->channel, in_out);
  char range_text[4][256];
  for(int k = 0; k < 4; k++)
  {
    channel->scale_print(parameters[k], boost_factor, range_text[k],
                         sizeof(range_text[k]));
    gtk_label_set_text(ed->filter[in_out].label[k], range_text[k]);
  }

  // keep the compact-mode tooltip (see _update_param_row_display) in sync with
  // a live drag too, not just the initial build
  gchar *full_tip =
    g_strdup_printf("%s: %s  %s  %s  %s\n\n%s", in_out ? _("output") : _("input"),
                    range_text[0], range_text[1], range_text[2], range_text[3],
                    _(slider_tooltip[in_out]));
  gtk_widget_set_tooltip_text(GTK_WIDGET(slider), full_tip);
  g_free(full_tip);

  if(parameters[1] == 0.0f && parameters[2] == 1.0f)
    p->blendif &= ~(1 << ch);
  else
    p->blendif |= (1 << ch);

  _param_form_commit(ed->module, ed->formid);
  _update_param_row_visibility(ed);
  // dragging a node can walk this element's own range into (or out of) a
  // no-op full span -- refresh its badge live, same as an opacity drag does
  // (see the DT_MASKS_PROPERTY_OPACITY case this function's own sibling
  // callback ends up feeding into); cheap and in-place, no list rebuild.
  _refresh_lowop_badges(ed->module);
}

static void _param_row_slider_reset_callback(GtkDarktableGradientSlider *slider,
                                             dt_masks_param_row_editor_t *ed)
{
  DT_GUARD_GUI_UPDATE();
  dt_masks_point_parametric_t *p = _param_row_point(ed);
  if(!p) return;
  const dt_iop_gui_blendif_channel_t *channels =
    dt_develop_blendif_channels_for_csp(p->colorspace);
  if(!channels) return;
  const dt_iop_gui_blendif_channel_t *channel = &channels[p->channel];

  const int in_out = (slider == ed->filter[1].slider) ? 1 : 0;
  const dt_develop_blendif_channels_t ch = channel->param_channels[in_out];

  // reset always clears polarity back to "not inverted" for this channel; the
  // per-shape invert (ctrl+click on the row handle) is what actually flips it
  p->blendif &= ~(1 << (16 + ch));

  _param_form_commit(ed->module, ed->formid);
  _update_param_row_display(ed);
  // a reset routinely lands this element's range back at the no-op full
  // span -- see the matching comment on _param_row_slider_callback above
  _refresh_lowop_badges(ed->module);
  // this row's own pickers are deferred (see DT_COLOR_PICKER_DEFERRED_AREA):
  // they normally resume from whatever box they last sampled, but a reset
  // means "start over" for the value they feed too, so forget that box --
  // the next pick waits for an entirely fresh selection instead of jumping
  // straight back to a leftover one that no longer has anything to do with
  // this now-reset range.
  dt_iop_color_picker_forget(ed->colorpicker_set_values);
  dt_iop_color_picker_forget(ed->colorpicker);
}

// this row's own range slider (see _build_param_row_filter) has no built-in
// equivalent of a plain bauhaus slider's right-click "type an exact value"
// popup (see _popup_show in bauhaus.c) -- dragging a node is the only way to
// set one of its four points. The three functions below add one, scoped to
// just these parametric-channel range sliders (see
// _param_row_slider_precise_press, connected in _build_param_row_editor)
// rather than touching GtkDarktableGradientSlider itself, which is shared
// well beyond flexi masks.
//
// A node's own stored position (gslider->position[k], what
// dtgtk_gradient_slider_multivalue_get/set_value read and write) lives in
// the same normalized [0,1] "display" domain channel->scale_print already
// formats for the row's own numeric labels (see _update_param_row_display) --
// _blendif_scale_ex, which normalizes a picked pixel into this same domain
// for the colour-picker feature, confirms it. scale_print itself is a
// one-way formatter with no matching parser, but only three implementations
// of it exist in this file (_blendif_scale_print_default/_ab/_hue, matched
// below by function-pointer identity), each a simple, exactly invertible
// formula -- so round-tripping a typed value back into [0,1] stays exact
// rather than needing a generic string-to-value parser for every channel
// kind that might ever be added.
float _param_row_slider_precise_display(const dt_iop_gui_blendif_channel_t *channel,
                                        const float boost_factor,
                                        const float frac)
{
  if(channel->scale_print == _blendif_scale_print_hue) return frac * 360.0f;
  if(channel->scale_print == _blendif_scale_print_ab)
    return (frac * 256.0f - 128.0f) * boost_factor;
  return frac * boost_factor * 100.0f; // _blendif_scale_print_default
}

float _param_row_slider_precise_parse(const dt_iop_gui_blendif_channel_t *channel,
                                      const float boost_factor,
                                      const float typed)
{
  if(channel->scale_print == _blendif_scale_print_hue) return typed / 360.0f;
  if(channel->scale_print == _blendif_scale_print_ab)
    return (typed / boost_factor + 128.0f) / 256.0f;
  return (typed / 100.0f) / boost_factor; // _blendif_scale_print_default
}

// looks up everything both a real commit (_param_row_slider_precise_value_changed)
// and a hover preview (_param_row_slider_precise_hover_preview) need to turn
// one of the popup's own bauhaus-slider values into this node's [0,1] channel
// fraction. Returns FALSE (nothing to do) if the row's own form/channel data
// went away mid-interaction.
static gboolean
_param_row_slider_precise_context(GtkWidget *slider,
                                  const float bauhaus_value,
                                  gint *k_out,
                                  float *newfrac_out,
                                  dt_masks_param_row_editor_t **ed_out,
                                  const dt_iop_gui_blendif_channel_t **channel_out,
                                  float *boost_factor_out,
                                  int *in_out_out)
{
  const gint k = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(slider), "precise-marker"));
  if(k < 0) return FALSE;
  dt_masks_param_row_editor_t *ed =
    g_object_get_data(G_OBJECT(slider), "param-row-editor");
  if(!ed) return FALSE;
  dt_masks_point_parametric_t *p = _param_row_point(ed);
  if(!p) return FALSE;
  const dt_iop_gui_blendif_channel_t *channels =
    dt_develop_blendif_channels_for_csp(p->colorspace);
  if(!channels) return FALSE;
  const dt_iop_gui_blendif_channel_t *channel = &channels[p->channel];
  const int in_out = (slider == GTK_WIDGET(ed->filter[1].slider)) ? 1 : 0;
  const float boost_factor =
    _get_boost_factor_ex(p->blendif_boost_factors, channels, p->channel, in_out);

  *k_out = k;
  *newfrac_out = _param_row_slider_precise_parse(channel, boost_factor, bauhaus_value);
  *ed_out = ed;
  *channel_out = channel;
  *boost_factor_out = boost_factor;
  *in_out_out = in_out;
  return TRUE;
}

// restores `slider`'s markers to this popup session's own baseline (see
// _param_row_slider_precise_open) before applying a new position for node
// `k` -- always computing from that one fixed baseline, never cumulatively
// from wherever a prior preview/drag tick happened to leave things, is what
// makes a neighbour that got pushed out of the way ease back once the
// pointer/drag reverses towards where this session started, instead of
// staying pushed forever (see _slider_move's own push logic in
// gradientslider.c, which has no such notion of "the position before this
// gesture began" on its own).
static void _param_row_slider_precise_restore_baseline(GtkWidget *slider)
{
  gdouble *baseline = g_object_get_data(G_OBJECT(slider), "precise-baseline");
  if(!baseline) return;
  DT_ENTER_GUI_UPDATE();
  dtgtk_gradient_slider_multivalue_set_values(DTGTK_GRADIENT_SLIDER(slider), baseline);
  DT_LEAVE_GUI_UPDATE();
}

// live-updates this popover's own node every time the embedded bauhaus
// slider's value is actually committed -- dragging it, scrolling it, typing
// into its own right-click popup, or the hover-preview settling (see
// _param_row_slider_precise_hover_settled) all funnel through this the same
// way any other bauhaus slider's edits do, so the node tracks it exactly like
// a normal slider-bound parameter rather than only committing once on close.
static void _param_row_slider_precise_value_changed(GtkWidget *bauhaus_slider,
                                                    GtkWidget *slider)
{
  if(DT_IN_GUI_UPDATE()) return;

  // this is a real commit: whatever the hover-preview debounce still had
  // pending is moot now, drop it rather than let it fire a redundant commit
  // a moment later
  const guint pending =
    GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(slider), "precise-hover-settle"));
  if(pending)
  {
    g_source_remove(pending);
    g_object_set_data(G_OBJECT(slider), "precise-hover-settle", NULL);
  }

  gint k;
  float newfrac;
  dt_masks_param_row_editor_t *ed;
  const dt_iop_gui_blendif_channel_t *channel;
  float boost_factor;
  int in_out;
  if(!_param_row_slider_precise_context(slider, dt_bauhaus_slider_get(bauhaus_slider), &k,
                                        &newfrac, &ed, &channel, &boost_factor, &in_out))
    return;

  _param_row_slider_precise_restore_baseline(slider);

  // emits "value-changed" itself (see
  // dtgtk_gradient_slider_multivalue_set_value_pushing), which _param_row_slider_callback
  // is already listening for -- so this commits through the exact same
  // persistence/label-refresh path a drag on the range slider's own node directly would.
  // The "_pushing" variant (not the plain
  // ..._set_value) matches a real drag's own behaviour: crossing an adjacent
  // node pushes it along instead of being hard-blocked at it (see
  // _param_row_slider_precise_open, whose embedded slider is given the full
  // channel range, not a neighbour-clamped one, precisely so this can happen).
  dtgtk_gradient_slider_multivalue_set_value_pushing(DTGTK_GRADIENT_SLIDER(slider),
                                                     newfrac, k);
}

// how long the pointer has to sit still over the popup's own slider before a
// hovered-but-not-yet-committed position actually gets committed (see
// _param_row_slider_precise_hover_preview/_settled below).
#define DT_MASKS_PRECISE_HOVER_SETTLE_MS 200

// fires once the pointer has stopped moving for DT_MASKS_PRECISE_HOVER_SETTLE_MS
// after a hover preview -- commits the previewed value for real, through the
// exact same bauhaus-slider "value-changed" path scrolling/dragging/typing
// already use (see _param_row_slider_precise_value_changed).
static gboolean _param_row_slider_precise_hover_settled(gpointer user_data)
{
  GtkWidget *bauhaus_slider = user_data;
  if(!GTK_IS_WIDGET(bauhaus_slider)) return G_SOURCE_REMOVE;
  GtkWidget *slider =
    g_object_get_data(G_OBJECT(bauhaus_slider), "precise-hover-preview-data");
  if(slider) g_object_set_data(G_OBJECT(slider), "precise-hover-settle", NULL);
  const float *value =
    g_object_get_data(G_OBJECT(bauhaus_slider), "precise-hover-last-value");
  // dt_bauhaus_slider_set() takes the same raw/unfactored domain `value`
  // already is (see _slider_normalized_to_value in bauhaus.c, which is what
  // produced it) -- dt_bauhaus_slider_set_val() instead expects the
  // factor+offset-applied public domain, so passing this same raw value
  // through it silently reinterpreted it in the wrong units whenever a
  // channel's slider had a non-trivial factor/offset, landing the commit at
  // the wrong position (seen as the control points "jumping" once the
  // pointer stopped, rather than settling where the hover preview left them).
  if(value) dt_bauhaus_slider_set(bauhaus_slider, *value);
  return G_SOURCE_REMOVE;
}

// dt_bauhaus_static_hover_preview_t hook (see bauhaus.h): fires on every
// pointer motion over the popup's own slider while no button is held.
// Previews node k moving to the hovered value -- and any neighbour it would
// push along -- on the row's own range slider, without touching the form's
// persisted parameters or the bauhaus slider's own committed value, then
// (re)arms the settle timer that turns this into a real commit once the
// pointer stops. A plain mouse-over that never pauses long enough to settle,
// or a popup dismissed (ESC) before it does, never touches anything real --
// see _param_row_slider_precise_closed, which discards it instead.
static void _param_row_slider_precise_hover_preview(GtkWidget *bauhaus_slider,
                                                    float value,
                                                    gpointer user_data)
{
  GtkWidget *slider = user_data;

  gint k;
  float newfrac;
  dt_masks_param_row_editor_t *ed;
  const dt_iop_gui_blendif_channel_t *channel;
  float boost_factor;
  int in_out;
  if(!_param_row_slider_precise_context(slider, value, &k, &newfrac, &ed, &channel,
                                        &boost_factor, &in_out))
    return;

  // move the popup's own displayed value/fill to track the hover position
  // too -- previously only the row's own markers moved during hover, so the
  // number shown in the popup itself stayed stuck at wherever it was before
  // the hover started. Guarded so this doesn't itself count as a commit (no
  // "value-changed", see _slider_set_normalized's own DT_IN_GUI_UPDATE check) --
  // that still only happens for real once the settle timer fires (see
  // _param_row_slider_precise_hover_settled) or the user actually clicks.
  DT_ENTER_GUI_UPDATE();
  dt_bauhaus_slider_set(bauhaus_slider, value);
  DT_LEAVE_GUI_UPDATE();

  _param_row_slider_precise_restore_baseline(slider);

  DT_ENTER_GUI_UPDATE();
  dtgtk_gradient_slider_multivalue_set_value_pushing(DTGTK_GRADIENT_SLIDER(slider),
                                                     newfrac, k);
  DT_LEAVE_GUI_UPDATE();

  // labels normally refresh from "value-changed" (see _param_row_slider_callback),
  // suppressed above since nothing is committed yet -- refresh them directly
  // instead, purely cosmetic
  _update_param_row_range_labels_preview(ed, DTGTK_GRADIENT_SLIDER(slider), channel,
                                         boost_factor, in_out);

  // this row lives in a different top-level window than the popup calling
  // this hook (see _param_row_slider_precise_open): a plain queue_draw here
  // only *requests* a redraw, and while the popup's own motion-notify stream
  // keeps firing back-to-back, the main loop's idle-priority redraw pass for
  // that other window never gets a turn -- the row visibly moves only once
  // the pointer stops and the stream lets up. Forcing the redraw synchronously
  // here, right when the position is known, is what actually makes it track
  // the pointer instead of only ever catching up at the end. gdk_window_process_updates
  // is deprecated (GTK4 has no equivalent -- the compositor's frame clock
  // replaces it), but there is no non-deprecated way to force a cross-window
  // redraw synchronously in GTK3, which is what this one, narrow case needs.
  // The popup's OWN window needs the identical forced flush for the identical
  // reason: even though the popup is the window the motion-notify stream is
  // itself arriving on, GTK never yields to its idle-priority redraw between
  // back-to-back motion events either, so dt_bauhaus_slider_set's own
  // queue_draw above (silent, see the DT_IN_GUI_UPDATE guard) sat un-rendered
  // the same way -- the popup's own number only ever caught up once the
  // pointer stopped, same symptom as the row.
  GdkWindow *slider_window = gtk_widget_get_window(slider);
  if(slider_window)
  {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    gdk_window_process_updates(slider_window, TRUE);
#pragma GCC diagnostic pop
  }
  GdkWindow *popup_window = gtk_widget_get_window(bauhaus_slider);
  if(popup_window && popup_window != slider_window)
  {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    gdk_window_process_updates(popup_window, TRUE);
#pragma GCC diagnostic pop
  }

  g_object_set_data(G_OBJECT(bauhaus_slider), "precise-hover-preview-data", slider);
  float *stored_value = g_new(float, 1);
  *stored_value = value;
  g_object_set_data_full(G_OBJECT(bauhaus_slider), "precise-hover-last-value",
                         stored_value, g_free);

  const guint pending =
    GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(slider), "precise-hover-settle"));
  if(pending) g_source_remove(pending);
  const guint handle =
    g_timeout_add(DT_MASKS_PRECISE_HOVER_SETTLE_MS,
                  _param_row_slider_precise_hover_settled, bauhaus_slider);
  g_object_set_data(G_OBJECT(slider), "precise-hover-settle", GUINT_TO_POINTER(handle));
}

// self-destructs the popover once GTK reports it closed and clears the two
// object-data slots _param_row_slider_precise_press uses to track "is a
// popover currently open, and for which node" -- so a stale pointer can
// never be read back out after this. Triggered either directly (Escape, an
// outside click on the popover itself) or by
// _param_row_slider_precise_popup_hidden below, once the embedded slider's
// own bauhaus popup -- the actual editing UI, see _param_row_slider_precise_open
// -- has closed.
static void _param_row_slider_precise_closed(GtkPopover *popover, GtkWidget *slider)
{
  // a hover preview still in flight (the settle debounce hasn't fired yet,
  // see _param_row_slider_precise_hover_preview) was never committed --
  // discard it and put the row's markers back exactly where this popup found
  // them (see _param_row_slider_precise_open). If the debounce already fired
  // at some point during this session, its commit stands; only an
  // in-flight, uncommitted preview gets thrown away here.
  const guint pending =
    GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(slider), "precise-hover-settle"));
  if(pending)
  {
    g_source_remove(pending);
    g_object_set_data(G_OBJECT(slider), "precise-hover-settle", NULL);
    _param_row_slider_precise_restore_baseline(slider);
  }
  g_object_set_data(G_OBJECT(slider), "precise-baseline", NULL);

  g_object_set_data(G_OBJECT(slider), "precise-popover", NULL);
  g_object_set_data(G_OBJECT(slider), "precise-marker", GINT_TO_POINTER(-1));
  // stop pinning this node's highlight now that its editor is gone (see
  // _param_row_slider_precise_open, which pins it on open).
  DTGTK_GRADIENT_SLIDER(slider)->pinned = -1;
  gtk_widget_queue_draw(slider);
  gtk_widget_destroy(GTK_WIDGET(popover));
}

// darktable.bauhaus->popup.window is the one floating window every bauhaus
// widget in the whole app shares for its right-click editing popup (only one
// can ever be open at a time) -- connected once, right when *our* popup opens
// below, so its "hide" (Enter, Escape, or an outside click, all handled
// entirely inside bauhaus.c) tells us editing this node is done and our own
// anchor popover should close along with it, instead of leaving it (and the
// row slider it's been standing in front of) open for a second, separate
// dismissal.
static void _param_row_slider_precise_popup_hidden(GtkWidget *bauhaus_popup_window,
                                                   GtkWidget *popover)
{
  g_signal_handlers_disconnect_by_func(bauhaus_popup_window,
                                       _param_row_slider_precise_popup_hidden, popover);
  if(GTK_IS_POPOVER(popover)) gtk_popover_popdown(GTK_POPOVER(popover));
}

// the embedded slider (see _param_row_slider_precise_open) is not realized
// yet at the point its anchor popover is first mapped, and _popup_show in
// bauhaus.c reads its GdkWindow to work out which toplevel the popup gets
// anchored to. Deferred to a plain idle instead of the "map" signal so this
// runs strictly after the size-allocate/realize pass that follows mapping
// (GTK services its own pending resizes at a higher priority than
// G_PRIORITY_DEFAULT_IDLE, so by the time this fires the widget is real).
static gboolean _param_row_slider_precise_open_idle(gpointer user_data)
{
  GtkWidget *bauhaus_slider = user_data;
  if(!GTK_IS_WIDGET(bauhaus_slider)) return G_SOURCE_REMOVE;
  GtkWidget *popover =
    g_object_get_data(G_OBJECT(bauhaus_slider), "precise-anchor-popover");
  GtkWidget *slider =
    g_object_get_data(G_OBJECT(bauhaus_slider), "precise-anchor-slider");
  const gint marker_x = GPOINTER_TO_INT(
    g_object_get_data(G_OBJECT(bauhaus_slider), "precise-anchor-marker-x"));
  if(slider && GTK_IS_WIDGET(slider))
    _show_bauhaus_whisker_popup(bauhaus_slider, slider, marker_x);
  else
    dt_bauhaus_widget_show_popup(bauhaus_slider);
  if(popover)
    g_signal_connect(G_OBJECT(darktable.bauhaus->popup.window), "hide",
                     G_CALLBACK(_param_row_slider_precise_popup_hidden), popover);
  return G_SOURCE_REMOVE;
}

// opens a small (practically invisible -- see above) anchor popover at node
// k's own position, holding one real bauhaus slider bound to that node's
// value, and immediately opens *that* slider's own right-click popup on it --
// so what the user actually sees and edits is a completely normal bauhaus
// value-entry popup, "the same logarithmic selection mode as [any other]
// value selector for sliders" the user asked for (see _popup_show in
// bauhaus.c), not a bespoke re-implementation of it, and not an extra click
// through some intermediate slider bar first.
static void _param_row_slider_precise_open(GtkWidget *slider,
                                           dt_masks_param_row_editor_t *ed,
                                           const gint k)
{
  dt_masks_point_parametric_t *p = _param_row_point(ed);
  if(!p) return;
  const dt_iop_gui_blendif_channel_t *channels =
    dt_develop_blendif_channels_for_csp(p->colorspace);
  if(!channels) return;
  const dt_iop_gui_blendif_channel_t *channel = &channels[p->channel];
  const int in_out = (slider == GTK_WIDGET(ed->filter[1].slider)) ? 1 : 0;
  const float boost_factor =
    _get_boost_factor_ex(p->blendif_boost_factors, channels, p->channel, in_out);

  GtkDarktableGradientSlider *gslider = DTGTK_GRADIENT_SLIDER(slider);

  // keep this node's own marker highlighted for as long as its editor is
  // open, regardless of where the pointer actually is/goes -- the popup now
  // deliberately opens away from the slider (see
  // _bauhaus_whisker_popup_rect), so the normal hover/drag-driven
  // highlight (see gradientslider.c's hovered_marker) would otherwise drop
  // as soon as the pointer leaves the slider. Cleared again in
  // _param_row_slider_precise_closed.
  gslider->pinned = k;
  gtk_widget_queue_draw(slider);

  // this node's own valid display-unit range: the channel's *overall* range
  // (matching _param_row_slider_precise_display's own formulas), not narrowed
  // to whatever an adjacent node currently allows -- a real drag on the
  // gradient slider itself is free to cross an adjacent node and push it
  // along (see _slider_move's FREE_MARKERS branch in gradientslider.c), so
  // this embedded slider must allow the same range, not hard-block at the
  // neighbour. Order is preserved instead by
  // dtgtk_gradient_slider_multivalue_set_value_pushing, called from
  // _param_row_slider_precise_value_changed on every change, which pushes the
  // neighbour rather than clamping against it -- exactly like a drag would.
  const double lo_frac = 0.0;
  const double hi_frac = 1.0;
  const gboolean is_hue = channel->scale_print == _blendif_scale_print_hue;
  const gboolean is_ab = channel->scale_print == _blendif_scale_print_ab;
  const float lo = _param_row_slider_precise_display(channel, boost_factor, lo_frac);
  const float hi = _param_row_slider_precise_display(channel, boost_factor, hi_frac);
  const float cur =
    _param_row_slider_precise_display(channel, boost_factor, gslider->position[k]);
  const int digits = is_hue ? 0 : 2;

  GtkWidget *bauhaus_slider =
    dt_bauhaus_slider_new_with_range(ed->module, lo, hi, 0, cur, digits);
  // exactly "°", no leading space: that string, together with a 360-wide
  // range, is what bauhaus matches on to give a slider the color-wheel popup
  // and the wrap-around past either end that an angle wants (see
  // _is_full_circle in bauhaus.c) -- the same treatment color balance rgb's
  // own hue sliders get. It is not a decoration to translate either, for the
  // same reason.
  dt_bauhaus_slider_set_format(bauhaus_slider, is_hue ? "°" : is_ab ? "" : "%");
  dt_bauhaus_widget_hide_label(bauhaus_slider);
  dt_bauhaus_widget_set_quad_visibility(bauhaus_slider, FALSE);
  dt_bauhaus_slider_set_val(bauhaus_slider, cur);
  // carry the channel's own gradient over from the row's range slider, so the
  // popup is colored like the track it is editing rather than a neutral bar.
  // The color wheel above needs it too: without stops it falls back to a bare
  // dial with no hues on it at all (see _draw_color_wheel in bauhaus.c).
  for(int s = 0; s < channel->numberstops; s++)
    dt_bauhaus_slider_set_stop(bauhaus_slider, channel->colorstops[s].stoppoint,
                               channel->colorstops[s].color.red,
                               channel->colorstops[s].color.green,
                               channel->colorstops[s].color.blue);
  // fill the popover's own width fully -- a bauhaus widget's halign otherwise
  // defaults to only claiming its own (narrow) natural width even inside an
  // expand+fill box slot.
  gtk_widget_set_hexpand(bauhaus_slider, TRUE);
  gtk_widget_set_halign(bauhaus_slider, GTK_ALIGN_FILL);
  // this popup opens away from the pointer on purpose (see
  // _bauhaus_whisker_popup_rect, positioned against the row's own
  // bounds, not the click point) -- tell bauhaus's own popup motion handler
  // not to apply its usual "opened at the pointer" assumptions (hover alone
  // dragging the value, auto-reject once the pointer strays too far from
  // where it opened): see the "static_popup" check in _window_motion_handle.
  g_object_set_data(G_OBJECT(bauhaus_slider), "dt-bauhaus-static-popup",
                    GINT_TO_POINTER(1));
  g_signal_connect(G_OBJECT(bauhaus_slider), "value-changed",
                   G_CALLBACK(_param_row_slider_precise_value_changed), slider);
  // static popups opt out of bauhaus's own "bare hover drags the value"
  // behaviour (see the comment right above), which is what made this control
  // click-and-drag-only -- restore a hover preview on top of that opt-out
  // instead of reverting it outright (removing "dt-bauhaus-static-popup"
  // would break the popup's own away-from-the-pointer positioning): this
  // hook is called with the value the pointer is over, without ever touching
  // the slider's own committed value, so it is up to
  // _param_row_slider_precise_hover_preview to decide what to preview and
  // when to actually commit (see its own comment, and the settle timer it
  // arms). This node's own baseline -- every marker's position as this popup
  // found it -- is snapshotted here too: both a real drag and a settled
  // hover preview recompute from this one fixed baseline (see
  // _param_row_slider_precise_restore_baseline), not cumulatively, so a
  // neighbour pushed out of the way eases back if the gesture reverses; and
  // _param_row_slider_precise_closed reverts to it if the popup is dismissed
  // (ESC) while a preview is still uncommitted.
  gdouble *baseline = g_new(gdouble, GRADIENT_SLIDER_MAX_POSITIONS);
  dtgtk_gradient_slider_multivalue_get_values(gslider, baseline);
  g_object_set_data_full(G_OBJECT(slider), "precise-baseline", baseline, g_free);
  g_object_set_data(G_OBJECT(bauhaus_slider), "dt-bauhaus-static-hover-preview",
                    (gpointer)_param_row_slider_precise_hover_preview);
  g_object_set_data(G_OBJECT(bauhaus_slider), "dt-bauhaus-static-hover-preview-data",
                    slider);

  GtkWidget *popover = gtk_popover_new(slider);
  // not modal: this popover is only ever an invisible-in-practice anchor for
  // the embedded slider's own popup (see _param_row_slider_precise_slider_mapped),
  // which opens a separate top-level window and manages its own grab/modality
  // entirely itself (see _popup_show in bauhaus.c) -- a *modal* anchor popover
  // grabs input for itself too, and that grab was winning over the bauhaus
  // popup's own, leaving the bauhaus popup visible but unable to receive any
  // clicks/keys at all.
  gtk_popover_set_modal(GTK_POPOVER(popover), FALSE);
  // fully transparent: this anchor (both its own chrome and the slider inside
  // it) is never meant to be seen at all -- it exists only to give the
  // embedded slider a real, mapped, on-screen position for its own popup to
  // open from (see _param_row_slider_precise_slider_mapped). A plain
  // gtk_widget_set_opacity() has no effect here: darktable.css sets
  // "popover { opacity: 1; ... }" (needed elsewhere for the tooltip on/off
  // shortcut), and that CSS rule always wins over the widget property -- so
  // this needs its own, more specific CSS override instead (see
  // "popover.mask-precise-anchor" in darktable.css). Not gtk_widget_hide,
  // so it stays mapped/positioned throughout.
  dt_gui_add_class(popover, "mask-precise-anchor");
  GtkWidget *box = dt_gui_hbox(bauhaus_slider);
  gtk_widget_set_size_request(box, DT_PIXEL_APPLY_DPI(160), -1);
  gtk_container_add(GTK_CONTAINER(popover), box);
  gtk_widget_show_all(box);

  // anchor at this node's own x position along the slider (not just centered
  // on the whole widget), mirroring a bauhaus slider's own popup opening
  // right over the value it edits.
  GtkAllocation alloc;
  gtk_widget_get_allocation(slider, &alloc);
  const int usable = MAX(alloc.width - gslider->margin_left - gslider->margin_right, 1);
  // span the anchor rect over the slider's full height (not just a 1px point
  // at mid-height) so GTK's automatic above/below placement clears the whole
  // slider instead of centering the popup on its vertical midpoint, which put
  // the popup's bottom half directly over the slider's top half.
  const GdkRectangle rect = { gslider->margin_left + (int)(gslider->position[k] * usable),
                              0, 1, alloc.height };
  gtk_popover_set_pointing_to(GTK_POPOVER(popover), &rect);

  g_signal_connect(G_OBJECT(popover), "closed",
                   G_CALLBACK(_param_row_slider_precise_closed), slider);
  g_object_set_data(G_OBJECT(bauhaus_slider), "precise-anchor-popover", popover);
  // consulted by _param_row_slider_precise_open_idle to place the real
  // bauhaus popup against the slider's own bounds instead of this anchor's.
  g_object_set_data(G_OBJECT(bauhaus_slider), "precise-anchor-slider", slider);
  // this node's own x within the slider, for _bauhaus_whisker_popup_rect to
  // center the real popup on (clamped to the mask panel's own bounds) instead
  // of centering it on the whole row -- computed now, from the slider's real
  // (already allocated) position, rather than recomputed later from the
  // row/marker fraction.
  g_object_set_data(G_OBJECT(bauhaus_slider), "precise-anchor-marker-x",
                    GINT_TO_POINTER(gslider->margin_left
                                    + (gint)(gslider->position[k] * usable)));

  g_object_set_data(G_OBJECT(slider), "precise-popover", popover);
  g_object_set_data(G_OBJECT(slider), "precise-marker", GINT_TO_POINTER(k));

  gtk_popover_popup(GTK_POPOVER(popover));
  g_idle_add(_param_row_slider_precise_open_idle, bauhaus_slider);
}

// right-click on one of this row's own range-slider nodes: instead of the
// widget's own built-in "toggle marker selection" behaviour (see the
// GDK_BUTTON_SECONDARY branch of _gradient_slider_button_press in
// dtgtk/gradientslider.c), pop up the precise-entry UI above for the node
// nearest the click, closing it again on a second right-click on the same
// node (toggle). Connected with a plain g_signal_connect, not _after:
// "button-press-event" is RUN_LAST, so a normally-connected handler runs
// *before* the widget's own class handler and, by returning TRUE here,
// fully replaces its right-click behaviour for this widget instead of also
// running alongside it (see g_signal_connect's own ordering guarantees).
static gboolean
_param_row_slider_precise_press(GtkWidget *widget, GdkEventButton *ev, gpointer user_data)
{
  if(ev->type != GDK_BUTTON_PRESS || ev->button != GDK_BUTTON_SECONDARY) return FALSE;

  dt_masks_param_row_editor_t *ed =
    g_object_get_data(G_OBJECT(widget), "param-row-editor");
  if(!ed) return FALSE;

  GtkDarktableGradientSlider *gslider = DTGTK_GRADIENT_SLIDER(widget);
  const gint k = gslider->active >= 0 ? gslider->active : gslider->selected;
  if(k < 0 || k >= gslider->positions) return FALSE;

  GtkWidget *existing = g_object_get_data(G_OBJECT(widget), "precise-popover");
  const gint existing_k =
    GPOINTER_TO_INT(g_object_get_data(G_OBJECT(widget), "precise-marker"));
  if(existing)
  {
    const gboolean same = (existing_k == k);
    // synchronously fires "closed" (see _param_row_slider_precise_closed),
    // which destroys it and clears both object-data slots before this
    // function goes on to read them again below
    gtk_popover_popdown(GTK_POPOVER(existing));
    if(same) return TRUE;
  }

  _param_row_slider_precise_open(widget, ed, k);
  return TRUE;
}

static void _param_row_boost_factor_callback(GtkWidget *slider,
                                             dt_masks_param_row_editor_t *ed)
{
  if(DT_IN_GUI_UPDATE()) return;
  dt_masks_point_parametric_t *p = _param_row_point(ed);
  if(!p) return;
  const dt_iop_gui_blendif_channel_t *channels =
    dt_develop_blendif_channels_for_csp(p->colorspace);
  if(!channels) return;
  const dt_iop_gui_blendif_channel_t *channel = &channels[p->channel];

  const float value = dt_bauhaus_slider_get(slider);
  for(int in_out = 1; in_out >= 0; in_out--)
  {
    const int ch = channel->param_channels[in_out];
    float off = 0.0f;
    if(p->colorspace == DEVELOP_BLEND_CS_LAB
       && (ch == DEVELOP_BLENDIF_A_in || ch == DEVELOP_BLENDIF_A_out
           || ch == DEVELOP_BLENDIF_B_in || ch == DEVELOP_BLENDIF_B_out))
      off = 0.5f;
    const float new_value = value + channel->boost_factor_offset;
    const float old_value = p->blendif_boost_factors[ch];
    const float factor = exp2f(old_value) / exp2f(new_value);
    float *parameters = &p->blendif_parameters[4 * ch];
    if(parameters[0] > 0.0f) parameters[0] = CLIP((parameters[0] - off) * factor + off);
    if(parameters[1] > 0.0f) parameters[1] = CLIP((parameters[1] - off) * factor + off);
    if(parameters[2] < 1.0f) parameters[2] = CLIP((parameters[2] - off) * factor + off);
    if(parameters[3] < 1.0f) parameters[3] = CLIP((parameters[3] - off) * factor + off);
    if(parameters[1] == 0.0f && parameters[2] == 1.0f) p->blendif &= ~(1 << ch);
    p->blendif_boost_factors[ch] = new_value;
  }
  _param_form_commit(ed->module, ed->formid);
  _update_param_row_display(ed);
}

// the parametric row's own opacity slider (packed alongside output/boost,
// under the same p->in_out gate -- see _update_param_row_visibility): commits
// via the same shared _props_row_apply every other row kind's opacity control
// uses, scoped to just this one form.
static void _param_row_opacity_changed(GtkWidget *widget, dt_masks_param_row_editor_t *ed)
{
  if(DT_IN_GUI_UPDATE() || !ed) return;
  GList *ids = g_list_prepend(NULL, GINT_TO_POINTER(ed->formid));
  _props_row_apply(ed->module, ids, DT_MASKS_PROPERTY_OPACITY, widget,
                   &ed->opacity_last_value, FALSE);
  g_list_free(ids);
}

// find the per-row editor struct owning `picker` (tagged "param-row-formid" at
// creation, see _build_param_row_editor), or NULL if `picker` is not one of
// this module's per-row picker buttons.
static dt_masks_param_row_editor_t *_param_row_editor_for_picker(dt_iop_module_t *module,
                                                                 GtkWidget *picker)
{
  const dt_mask_id_t formid =
    GPOINTER_TO_INT(g_object_get_data(G_OBJECT(picker), "param-row-formid"));
  if(!dt_is_valid_maskid(formid)) return NULL;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  GtkWidget *row_vbox = _masks_row_widget(bd, formid);
  GtkWidget *editor_box =
    row_vbox ? g_object_get_data(G_OBJECT(row_vbox), "param-editor-box") : NULL;
  return editor_box ? g_object_get_data(G_OBJECT(editor_box), "param-editor") : NULL;
}

// arms the picker for this row's own channel/colorspace before sampling
// starts -- without this, the pixelpipe samples/converts the picked pixel
// into whatever colorspace the picker was last armed for (e.g. the module's
// default, or a previous row's channel), which for a mismatched channel
// (say cst=RGB while this row is a JzCzhz "hz" channel) leaves that channel
// unwritten by _blendif_scale_ex and both bounds collapse to the same
// clamped default -- a zero-width range. Mirrors the classic shared
// editor's _update_gradient_slider_pickers, which does the same on toggle.
static void _update_param_row_slider_pickers(dt_masks_param_row_editor_t *ed);

static void _param_row_arm_picker_cst(GtkWidget *button, dt_masks_param_row_editor_t *ed)
{
  const dt_masks_point_parametric_t *p = _param_row_point(ed);
  if(!p) return;
  dt_iop_color_picker_set_cst(
    ed->module, _picker_colorspace_for_channel(
                  (dt_develop_blend_colorspace_t)p->colorspace, (int)p->channel));
  // also refresh (or clear) this row's picker marker/label to match the
  // button's new armed/disarmed state, mirroring the classic shared
  // editor's _update_gradient_slider_pickers, which does both in one call.
  _update_param_row_slider_pickers(ed);

  const gboolean btn_active = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(button));
  const dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, ed->formid);
  const char *element_name = (form && form->name[0]) ? form->name : _("parametric");

  if(button == ed->colorpicker_set_values)
  {
    const gboolean is_output =
      GPOINTER_TO_INT(g_object_get_data(G_OBJECT(button), "pick-output")) != 0;
    if(btn_active)
      dt_toast_log(is_output ? _("output picker of %s armed")
                             : _("input picker of %s armed"),
                   element_name);
    else
      dt_toast_log(is_output ? _("output picker of %s unarmed")
                             : _("input picker of %s unarmed"),
                   element_name);
  }
  else if(button == ed->colorpicker)
  {
    if(btn_active)
      dt_toast_log(_("color picker of %s armed"), element_name);
    else
      dt_toast_log(_("color picker of %s unarmed"), element_name);
  }

  // keep the one visible button's own look in sync with whichever of the two
  // real (hidden) pickers this "toggled" came from -- see
  // _param_row_master_picker_pressed.
  if(ed->master_picker)
  {
    const gboolean active =
      gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(ed->colorpicker))
      || gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(ed->colorpicker_set_values));
    if(!active && ed->colorpicker_set_values)
      g_object_set_data(G_OBJECT(ed->colorpicker_set_values), "pick-output", GINT_TO_POINTER(0));
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(ed->master_picker), active);
  }
}

// consolidated front-end for a parametric row's two color pickers (saves a
// slot in the row's action cluster): one visible button standing in for both
// hidden-but-functional real ones. CAPTURE-phase claim mirrors how
// dt_color_picker_new's own buttons suppress their default click-to-toggle
// behaviour (see _color_picker_new in color_picker_proxy.c) -- this button
// has no picker logic of its own, only the dispatch below, so its own
// click/toggle must never fire.
//   plain click   -> colorpicker_set_values, point mode is irrelevant (area-only);
//                    applied from the input range (see _param_row_picker_apply)
//   shift+click    -> same picker; applied from the output range instead
//                    (the modifier is read again at apply time, on the canvas
//                    pick/drag, not here)
//   ctrl+click     -> colorpicker, point mode
//   ctrl+right-click -> colorpicker, area mode
// claim the event sequence in CAPTURE phase so master_picker's own internal
// GtkGestureMultiPress (BUBBLE phase, click-to-toggle) never runs -- same
// pattern as color_picker_proxy.c's _color_picker_new (that one's own
// _gesture_begin_claim is static to that file, hence this local copy).
static void _param_row_master_picker_begin_claim(GtkGesture *gesture,
                                                 GdkEventSequence *sequence,
                                                 gpointer user_data)
{
  gtk_gesture_set_sequence_state(gesture, sequence, GTK_EVENT_SEQUENCE_CLAIMED);
}

static void _param_row_master_picker_pressed(GtkGesture *gesture,
                                             gint n_press,
                                             gdouble x,
                                             gdouble y,
                                             dt_masks_param_row_editor_t *ed)
{
  const gboolean ctrl = dt_modifier_is(dt_key_modifier_state(), GDK_CONTROL_MASK);
  const gboolean shift = dt_modifier_is(dt_key_modifier_state(), GDK_SHIFT_MASK);
  const gboolean right =
    gtk_gesture_single_get_current_button(GTK_GESTURE_SINGLE(gesture))
    == GDK_BUTTON_SECONDARY;
  if(ctrl)
    dt_color_picker_click(ed->colorpicker, right);
  else if(!right)
  {
    g_object_set_data(G_OBJECT(ed->colorpicker_set_values), "pick-output",
                      GINT_TO_POINTER(shift));
    dt_color_picker_click(ed->colorpicker_set_values, FALSE);
  }
}

// per-row equivalent of _update_gradient_slider_pickers -- the plain "pick
// GUI color" button doesn't change any value, it only moves the little
// picker-mean/min/max marker on this row's own slider (and its text label)
// to reflect where the just-sampled color falls on this row's channel.
static void _update_param_row_slider_pickers(dt_masks_param_row_editor_t *ed)
{
  const dt_masks_point_parametric_t *p = _param_row_point(ed);
  if(!p) return;
  const dt_iop_gui_blendif_channel_t *channels =
    dt_develop_blendif_channels_for_csp(p->colorspace);
  if(!channels) return;

  dt_iop_module_t *module = ed->module;
  float *raw_mean, *raw_min, *raw_max;

  DT_ENTER_GUI_UPDATE();

  for(int in_out = 1; in_out >= 0; in_out--)
  {
    if(in_out)
    {
      raw_mean = module->picked_output_color;
      raw_min = module->picked_output_color_min;
      raw_max = module->picked_output_color_max;
    }
    else
    {
      raw_mean = module->picked_color;
      raw_min = module->picked_color_min;
      raw_max = module->picked_color_max;
    }

    dt_iop_gui_blendif_filter_t *sl = &ed->filter[in_out];

    if((gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(ed->colorpicker))
        || gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(ed->colorpicker_set_values)))
       && (raw_min[0] != FLT_MAX))
    {
      float picker_mean[8], picker_min[8], picker_max[8];
      float cooked[8];

      const dt_iop_colorspace_type_t cst = _picker_colorspace_for_channel(
        (dt_develop_blend_colorspace_t)p->colorspace, (int)p->channel);
      const dt_iop_order_iccprofile_info_t *work_profile =
        ((dt_develop_blend_colorspace_t)p->colorspace == DEVELOP_BLEND_CS_RGB_SCENE)
          ? dt_ioppr_get_pipe_current_profile_info(module, module->dev->full.pipe)
          : dt_ioppr_get_iop_work_profile_info(module, module->dev->iop);

      _blendif_scale_ex(p->blendif_boost_factors, channels, cst, raw_mean, picker_mean,
                        work_profile, in_out);
      _blendif_scale_ex(p->blendif_boost_factors, channels, cst, raw_min, picker_min,
                        work_profile, in_out);
      _blendif_scale_ex(p->blendif_boost_factors, channels, cst, raw_max, picker_max,
                        work_profile, in_out);
      _blendif_cook(cst, raw_mean, cooked, work_profile);

      const int tab = (int)p->channel;
      gchar *text =
        g_strdup_printf("(%.*f)", _blendif_print_digits_picker(cooked[tab]), cooked[tab]);

      dtgtk_gradient_slider_multivalue_set_picker_meanminmax(
        sl->slider, CLAMP(picker_mean[tab], 0.0f, 1.0f),
        CLAMP(picker_min[tab], 0.0f, 1.0f), CLAMP(picker_max[tab], 0.0f, 1.0f));
      gtk_label_set_text(sl->picker_label, text);

      g_free(text);
    }
    else
    {
      dtgtk_gradient_slider_multivalue_set_picker(sl->slider, NAN);
      gtk_label_set_text(sl->picker_label, "");
    }
  }

  DT_LEAVE_GUI_UPDATE();
}

// per-row equivalent of blend_color_picker_apply's two branches -- applies a
// picked color straight into this row's own form (no bp scratch), mirroring
// the legacy math (see _blendif_scale_ex) with p->channel standing in for the
// shared editor's bd->tab.
static gboolean _param_row_picker_apply(dt_iop_module_t *module,
                                        GtkWidget *picker,
                                        dt_dev_pixelpipe_t *pipe)
{
  dt_masks_param_row_editor_t *ed = _param_row_editor_for_picker(module, picker);
  if(!ed) return FALSE;
  dt_masks_point_parametric_t *p = _param_row_point(ed);
  if(!p) return FALSE;
  const dt_iop_gui_blendif_channel_t *channels =
    dt_develop_blendif_channels_for_csp(p->colorspace);
  if(!channels) return FALSE;

  if(picker == ed->colorpicker_set_values)
  {
    DT_TRY_GUI_UPDATE(TRUE);

    const int tab = (int)p->channel;
    dt_aligned_pixel_t raw_min, raw_max;
    float picker_min[8] DT_ALIGNED_PIXEL, picker_max[8] DT_ALIGNED_PIXEL;
    dt_aligned_pixel_t picker_values;

    // shift (not ctrl) picks the output range -- ctrl is now the consolidated
    // picker button's own modifier for the OTHER picker (see
    // _param_row_master_picker_pressed)
    const gboolean armed_shift =
      GPOINTER_TO_INT(g_object_get_data(G_OBJECT(picker), "pick-output"));
    const gboolean current_shift =
      dt_modifier_is(dt_key_modifier_state(), GDK_SHIFT_MASK);
    const int in_out = (armed_shift || current_shift) ? 1 : 0;

    if(in_out)
    {
      for(size_t i = 0; i < 4; i++)
      {
        raw_min[i] = module->picked_output_color_min[i];
        raw_max[i] = module->picked_output_color_max[i];
      }
    }
    else
    {
      for(size_t i = 0; i < 4; i++)
      {
        raw_min[i] = module->picked_color_min[i];
        raw_max[i] = module->picked_color_max[i];
      }
    }

    const dt_iop_gui_blendif_channel_t *channel = &channels[p->channel];
    const dt_develop_blendif_channels_t ch = channel->param_channels[in_out];
    dt_iop_gui_blendif_filter_t *sl = &ed->filter[in_out];
    float *parameters = &p->blendif_parameters[4 * ch];

    // always derive from this row's own channel rather than trusting
    // dt_iop_color_picker_get_active_cst()'s stored state -- with several
    // rows' pickers sharing one module-wide picker object, that state only
    // reflects whichever row last armed it (see _param_row_arm_picker_cst),
    // which is unreliable to re-derive at apply time.
    const dt_iop_colorspace_type_t cst = _picker_colorspace_for_channel(
      (dt_develop_blend_colorspace_t)p->colorspace, (int)p->channel);
    const dt_iop_order_iccprofile_info_t *work_profile =
      ((dt_develop_blend_colorspace_t)p->colorspace == DEVELOP_BLEND_CS_RGB_SCENE)
        ? dt_ioppr_get_pipe_current_profile_info(module, pipe)
        : dt_ioppr_get_iop_work_profile_info(module, module->dev->iop);

    gboolean reverse_hues = FALSE;
    if(cst == IOP_CS_HSL && tab == CHANNEL_INDEX_H)
    {
      if((raw_max[3] - raw_min[3]) < (raw_max[0] - raw_min[0]) && raw_min[3] < 0.5f
         && raw_max[3] > 0.5f)
      {
        raw_max[0] = raw_max[3] < 0.5f ? raw_max[3] + 0.5f : raw_max[3] - 0.5f;
        raw_min[0] = raw_min[3] < 0.5f ? raw_min[3] + 0.5f : raw_min[3] - 0.5f;
        reverse_hues = TRUE;
      }
    }
    else if((cst == IOP_CS_LCH && tab == CHANNEL_INDEX_h)
            || (cst == IOP_CS_JZCZHZ && tab == CHANNEL_INDEX_hz))
    {
      if((raw_max[3] - raw_min[3]) < (raw_max[2] - raw_min[2]) && raw_min[3] < 0.5f
         && raw_max[3] > 0.5f)
      {
        raw_max[2] = raw_max[3] < 0.5f ? raw_max[3] + 0.5f : raw_max[3] - 0.5f;
        raw_min[2] = raw_min[3] < 0.5f ? raw_min[3] + 0.5f : raw_min[3] - 0.5f;
        reverse_hues = TRUE;
      }
    }

    _blendif_scale_ex(p->blendif_boost_factors, channels, cst, raw_min, picker_min,
                      work_profile, in_out);
    _blendif_scale_ex(p->blendif_boost_factors, channels, cst, raw_max, picker_max,
                      work_profile, in_out);

    const float feather = 0.01f;
    if(picker_min[tab] > picker_max[tab])
    {
      const float tmp = picker_min[tab];
      picker_min[tab] = picker_max[tab];
      picker_max[tab] = tmp;
    }

    picker_values[0] = CLAMP(picker_min[tab] - feather, 0.f, 1.f);
    picker_values[1] = CLAMP(picker_min[tab] + feather, 0.f, 1.f);
    picker_values[2] = CLAMP(picker_max[tab] - feather, 0.f, 1.f);
    picker_values[3] = CLAMP(picker_max[tab] + feather, 0.f, 1.f);

    if(picker_values[1] > picker_values[2])
    {
      picker_values[1] = CLAMP(picker_min[tab], 0.f, 1.f);
      picker_values[2] = CLAMP(picker_max[tab], 0.f, 1.f);
    }
    picker_values[0] = CLAMP(picker_values[0], 0.f, picker_values[1]);
    picker_values[3] = CLAMP(picker_values[3], picker_values[2], 1.f);

    for(int k = 0; k < 4; k++)
      dtgtk_gradient_slider_multivalue_set_value(sl->slider, picker_values[k], k);

    DT_LEAVE_GUI_UPDATE();

    for(int k = 0; k < 4; k++)
      parameters[k] = dtgtk_gradient_slider_multivalue_get_value(sl->slider, k);

    if(parameters[1] == 0.0f && parameters[2] == 1.0f)
      p->blendif &= ~(1 << ch);
    else
      p->blendif |= (1 << ch);

    // legacy also XORs in a whole-mask "invert" toggle (bp->mask_combine) here;
    // a single-channel form has no such global toggle (its own shape-level
    // invert is a separate axis, applied by the compositor), so reverse_hues
    // alone decides the picked range's polarity bit.
    if(reverse_hues)
      p->blendif |= 1 << (16 + ch);
    else
      p->blendif &= ~(1 << (16 + ch));

    _param_form_commit(module, ed->formid);
    _update_param_row_display(ed);
    // a picked area routinely takes this element's range off (or, after a
    // reset, back onto) the no-op full span -- same as a manual drag/reset
    // (see the matching call in _param_row_slider_callback/_reset_callback);
    // this path sets the range via dtgtk_gradient_slider_multivalue_set_value
    // directly rather than through that slider's own "value-changed", so
    // nothing else refreshes the badge for it.
    _refresh_lowop_badges(module);

    return TRUE;
  }
  else if(picker == ed->colorpicker)
  {
    DT_GUARD_GUI_UPDATE(TRUE);
    _update_param_row_slider_pickers(ed);
    return TRUE;
  }
  return FALSE;
}

// build one input-or-output slider bundle -- mirrors the shared editor's
// construction in dt_iop_gui_init_blendif, minus the per-slider polarity
// button (single-channel forms replace it with the row's own ctrl+click invert).

static void _build_param_row_filter(dt_iop_gui_blendif_filter_t *sl, const int in_out)
{
  sl->slider =
    DTGTK_GRADIENT_SLIDER_MULTIVALUE(dtgtk_gradient_slider_multivalue_new_with_name(
      4, in_out ? "mask-param-output-slider" : "mask-param-input-slider"));
  dt_gui_add_class(GTK_WIDGET(sl->slider), "mask-param-slider");
  sl->polarity = NULL;

  GtkWidget *label_box = gtk_grid_new();
  gtk_grid_set_column_homogeneous(GTK_GRID(label_box), TRUE);
  sl->label_box = label_box;

  sl->head = GTK_LABEL(dt_ui_label_new(in_out ? _("output") : _("input")));
  gtk_grid_attach(GTK_GRID(label_box), GTK_WIDGET(sl->head), 0, 0, 1, 1);

  GtkWidget *overlay = gtk_overlay_new();
  gtk_grid_attach(GTK_GRID(label_box), overlay, 1, 0, 3, 1);
  sl->values_box = overlay;

  sl->picker_label = GTK_LABEL(gtk_label_new(""));
  gtk_widget_set_name(GTK_WIDGET(sl->picker_label), "blend-data");
  gtk_label_set_xalign(sl->picker_label, .0);
  gtk_label_set_yalign(sl->picker_label, 1.0);
  gtk_container_add(GTK_CONTAINER(overlay), GTK_WIDGET(sl->picker_label));

  for(int k = 0; k < 4; k++)
  {
    sl->label[k] = GTK_LABEL(gtk_label_new(NULL));
    gtk_widget_set_name(GTK_WIDGET(sl->label[k]), "blend-data");
    gtk_label_set_xalign(sl->label[k], .35 + k * .65 / 3);
    gtk_label_set_yalign(sl->label[k], k % 2);
    gtk_overlay_add_overlay(GTK_OVERLAY(overlay), GTK_WIDGET(sl->label[k]));
  }

  gtk_widget_set_tooltip_text(GTK_WIDGET(sl->slider), _(slider_tooltip[in_out]));

  sl->head_compact = GTK_LABEL(dt_ui_label_new(in_out ? _("output") : _("input")));
  gtk_widget_set_tooltip_text(GTK_WIDGET(sl->head_compact), _(slider_tooltip[in_out]));
  sl->compact_row = NULL;
  sl->box = NULL;
}

// the eye floats over the right end of its slider row instead of taking a
// grid column of its own, and the slider gives back the eye's width less the
// room its bar keeps for a handle at 1.0: the bar then ends where the eye
// starts, flush with the header's opacity value above it, whatever the marker
// shape or font size. The handle at 1.0 reaches under the eye's left margin.
// Run on style changes, not size-allocate: a resize queued from inside an
// allocation is dropped, and the margin would wait for some later relayout.
static void _param_slider_fit_eye(dt_masks_param_row_editor_t *ed)
{
  GtkWidget *eyes[2] = { ed->input_bypass_slot, ed->output_bypass_slot };
  for(int i = 0; i < 2; i++)
  {
    GtkWidget *slider = GTK_WIDGET(ed->filter[i].slider);
    if(!slider || !eyes[i]) continue;
    gint eye_w = 0;
    gtk_widget_get_preferred_width(eyes[i], &eye_w, NULL);
    // no width yet (unstyled, or hidden with its row): the next style change
    // or _update_param_row_visibility tries again
    if(eye_w <= 0) continue;
    const int inset =
      dtgtk_gradient_slider_get_right_inset(DTGTK_GRADIENT_SLIDER(slider));
    const int margin = MAX(0, eye_w - inset);
    if(gtk_widget_get_margin_end(slider) != margin)
      gtk_widget_set_margin_end(slider, margin);
  }
}

// connected with g_signal_connect_object on the editor box, which owns ed
// (freed with it), so this is disconnected before ed goes away
static void _param_slider_style_updated(GtkWidget *widget,
                                        GtkWidget *wrap)
{
  dt_masks_param_row_editor_t *ed = g_object_get_data(G_OBJECT(wrap), "param-editor");
  if(ed) _param_slider_fit_eye(ed);
}

// the "temporarily disable this channel" eye that sits at the right end of a
// parametric row's input/output slider
static GtkWidget *_make_param_bypass_btn(const char *tooltip,
                                         dt_masks_param_row_editor_t *ed)
{
  GtkWidget *btn = dtgtk_togglebutton_new(dtgtk_cairo_paint_eye_toggle, 0, NULL);
  // -bypass-btn carries the look (dimmed until hovered or checked),
  // -param-bypass-btn the geometry: exactly the row header's expander button,
  // so the two read as one column rather than two similar icons that happen to
  // be near each other
  dt_gui_add_class(btn, "mask-refine-bypass-btn");
  dt_gui_add_class(btn, "mask-param-bypass-btn");
  gtk_widget_set_valign(btn, GTK_ALIGN_CENTER);
  gtk_widget_set_tooltip_text(btn, tooltip);
  g_signal_connect(G_OBJECT(btn), "toggled",
                   G_CALLBACK(_param_channel_bypass_toggled), ed);
  return btn;
}

// the fixed-width box the eye above lives in, laid over the right end of its
// slider row (see _param_slider_fit_eye).
//
// The eye comes and goes with whether the channel has both sub-ranges in play
// (see _update_param_row_visibility); the box keeps its width whether or not
// the eye is in it, so the slider's margin does not change as the user edits.
//
// That width is the row header's expander button, and the editor's right edge
// is flush with the header's (see .mask-param-row-editor in darktable.css), so
// the eyes stack directly under the expander.
static GtkWidget *_make_param_bypass_slot(GtkWidget *btn)
{
  GtkWidget *slot = dt_gui_hbox();
  dt_gui_add_class(slot, "mask-param-bypass-slot");
  // an overlay child fills the overlay unless told otherwise
  gtk_widget_set_halign(slot, GTK_ALIGN_END);
  gtk_widget_set_valign(slot, GTK_ALIGN_CENTER);
  dt_gui_box_add(slot, btn);
  return slot;
}

// build the always-visible per-row parametric editor for `form` (a single-channel
// parametric mask). Returns the wrapper widget (sliders + boost factor) to pack
// under the row; *picker_box_out receives the row's two color-picker buttons as
// a separate small box, meant to be packed into the row's own header/actions
// cluster instead (see _make_shape_row) -- they are per-channel controls, not
// part of the slider editor itself. The editor struct is attached to the
// returned wrap widget (freed automatically when the row is torn down by the
// next _build_masks_list rebuild); the picker box lives in the same row's
// widget subtree so both are destroyed together.
static GtkWidget *_build_param_row_editor(dt_iop_module_t *module,
                                          dt_masks_form_t *form,
                                          GtkWidget **picker_box_out)
{
  const dt_masks_point_parametric_t *p = form->points ? form->points->data : NULL;
  if(!p)
  {
    if(picker_box_out) *picker_box_out = NULL;
    return NULL;
  }

  dt_masks_param_row_editor_t *ed = g_malloc0(sizeof(dt_masks_param_row_editor_t));
  ed->formid = form->formid;
  ed->module = module;

  _build_param_row_filter(&ed->filter[0], 0);
  _build_param_row_filter(&ed->filter[1], 1);

  for(int in_out = 0; in_out < 2; in_out++)
  {
    dt_iop_gui_blendif_filter_t *sl = &ed->filter[in_out];
    g_signal_connect(G_OBJECT(sl->slider), "value-changed",
                     G_CALLBACK(_param_row_slider_callback), ed);
    g_signal_connect(G_OBJECT(sl->slider), "value-reset",
                     G_CALLBACK(_param_row_slider_reset_callback), ed);
    // back-reference so _param_row_editor_channel can resolve THIS row's own
    // channel from the slider alone, instead of the removed classic editor's shared
    // data->channel[data->tab] (permanently NULL now, see dt_iop_gui_blend_data_t in
    // blend.h -- dereferencing it here used to crash on 'c'/'C'/'m'/'M'/'a'/'A' while
    // hovering this slider).
    g_object_set_data(G_OBJECT(sl->slider), "param-row-editor", ed);
    dt_gui_connect_motion(sl->slider, NULL, _blendop_blendif_enter_cb,
                          _blendop_blendif_leave_cb, module);
    dt_gui_connect_key(sl->slider, _blendop_blendif_key_press_cb, module);
    // right-click: precise numeric entry for the nearest node (see
    // _param_row_slider_precise_press), replacing this widget's own built-in
    // right-click behaviour just for these range sliders.
    g_signal_connect(G_OBJECT(sl->slider), "button-press-event",
                     G_CALLBACK(_param_row_slider_precise_press), NULL);
  }

  // both real pickers stay fully functional (dt_color_picker_click below
  // arms them programmatically), just never shown -- see master_picker,
  // built after them, which is the row's one visible button.
  GtkWidget *picker_box = dt_gui_hbox();
  gtk_widget_set_size_request(picker_box, DT_PIXEL_APPLY_DPI(18), DT_PIXEL_APPLY_DPI(18));
  gtk_widget_set_valign(picker_box, GTK_ALIGN_CENTER);
  dt_gui_add_class(picker_box, "mask-within-combo");
  ed->colorpicker = dt_color_picker_new(module,
                                        DT_COLOR_PICKER_POINT_AREA | DT_COLOR_PICKER_IO
                                          | DT_COLOR_PICKER_DEFERRED_AREA,
                                        picker_box);
  gtk_widget_set_no_show_all(ed->colorpicker, TRUE);
  gtk_widget_hide(ed->colorpicker);
  g_object_set_data(G_OBJECT(ed->colorpicker), "param-row-formid",
                    GINT_TO_POINTER(ed->formid));
  g_signal_connect(G_OBJECT(ed->colorpicker), "toggled",
                   G_CALLBACK(_param_row_arm_picker_cst), ed);

  // deferred: don't sample a big default box the instant this arms (see
  // DT_COLOR_PICKER_DEFERRED_AREA) -- wait for the user's own drag on canvas,
  // so the range isn't set from ~96% of the image before they've picked
  // anything.
  ed->colorpicker_set_values = dt_color_picker_new(
    module, DT_COLOR_PICKER_AREA | DT_COLOR_PICKER_IO | DT_COLOR_PICKER_DEFERRED_AREA,
    picker_box);
  gtk_widget_set_no_show_all(ed->colorpicker_set_values, TRUE);
  gtk_widget_hide(ed->colorpicker_set_values);
  g_object_set_data(G_OBJECT(ed->colorpicker_set_values), "param-row-formid",
                    GINT_TO_POINTER(ed->formid));
  g_signal_connect(G_OBJECT(ed->colorpicker_set_values), "toggled",
                   G_CALLBACK(_param_row_arm_picker_cst), ed);

  // the one visible button standing in for both -- see
  // _param_row_master_picker_pressed for the modifier dispatch. Built the
  // same way dt_color_picker_new's own buttons are (dtgtk togglebutton +
  // CAPTURE-phase gesture claiming the press), since it needs the identical
  // "my handler fully owns click/toggle state" behaviour but with no
  // dt_iop_color_picker_t of its own to hand that off to.
  ed->master_picker = dtgtk_togglebutton_new(dtgtk_cairo_paint_colorpicker, 0, NULL);
  dt_gui_add_class(ed->master_picker, "dt_transparent_background");
  gtk_widget_set_valign(ed->master_picker, GTK_ALIGN_CENTER);
  gtk_widget_set_name(ed->master_picker, "keep-active");
  gtk_widget_set_tooltip_text(ed->master_picker,
                              _("click: set range from input\n"
                                "shift+click: set range from output\n"
                                "ctrl+click: pick GUI color (point)\n"
                                "ctrl+right-click: pick GUI color (area)"));
  GtkGesture *master_gesture = gtk_gesture_multi_press_new(ed->master_picker);
  gtk_event_controller_set_propagation_phase(GTK_EVENT_CONTROLLER(master_gesture),
                                             GTK_PHASE_CAPTURE);
  gtk_gesture_single_set_button(GTK_GESTURE_SINGLE(master_gesture), 0);
  dt_gui_add_controller(ed->master_picker, master_gesture);
  g_signal_connect(master_gesture, "pressed",
                   G_CALLBACK(_param_row_master_picker_pressed), ed);
  g_signal_connect(master_gesture, "begin",
                   G_CALLBACK(_param_row_master_picker_begin_claim), NULL);
  dt_gui_box_add(picker_box, ed->master_picker);

  ed->boost_slider = dt_bauhaus_slider_new_with_range(module, 0.0f, 18.0f, 0, 0.0f, 3);
  dt_bauhaus_slider_set_format(ed->boost_slider, _(" EV"));
  dt_bauhaus_widget_set_label(ed->boost_slider, N_("blend"), N_("boost factor"));
  dt_bauhaus_slider_set_soft_range(ed->boost_slider, 0.0, 3.0);
  // this slider has no quad icon, so hide the quad area entirely instead of
  // leaving an empty reserved patch to its right
  dt_bauhaus_widget_set_quad_visibility(ed->boost_slider, FALSE);
  gtk_widget_set_tooltip_text(
    ed->boost_slider,
    _("adjust the channel boost factor.\nincrease to allow matching values over 100%"));
  g_signal_connect(G_OBJECT(ed->boost_slider), "value-changed",
                   G_CALLBACK(_param_row_boost_factor_callback), ed);
  dt_gui_add_class(ed->boost_slider, "mask-boost-factor-slider");
  ed->boost_box = dt_gui_vbox(ed->boost_slider);
  dt_gui_add_class(ed->boost_box, "mask-boost-factor-box");

  // opacity slider: a parametric row's in/out chevron is its expander (unlike
  // shape/raster rows, which get their own separate one -- see
  // _make_props_row_toggle), so this is the slider that leads its expanded
  // controls, exactly as the shape rows' does. Shown only while "show opacity
  // slider in expanded elements" is on (see _model_param_row_visibility); the
  // row header's own compact opacity value is always there either way, and
  // reads this same slider (see _make_shape_row's parametric branch). Styled
  // like boost_box's labeled, below-row slider rather than the inline,
  // label-hidden treatment a row header uses, and delta-applied via the shared
  // _props_row_apply -- same protocol as every other row kind's opacity.
  ed->opacity_slider = dt_bauhaus_slider_new_with_range(
    module, _blend_masks_properties[DT_MASKS_PROPERTY_OPACITY].min,
    _blend_masks_properties[DT_MASKS_PROPERTY_OPACITY].max, 0, 1.0, 2);
  dt_bauhaus_widget_set_label(ed->opacity_slider, N_("blend"),
                              _blend_masks_properties[DT_MASKS_PROPERTY_OPACITY].name);
  dt_bauhaus_slider_set_format(ed->opacity_slider,
                               _blend_masks_properties[DT_MASKS_PROPERTY_OPACITY].format);
  dt_bauhaus_slider_set_digits(ed->opacity_slider, 2);
  // no quad icon -- see the same call for the shape/group properties sliders
  dt_bauhaus_widget_set_quad_visibility(ed->opacity_slider, FALSE);
  ed->opacity_last_value = dt_bauhaus_slider_get(ed->opacity_slider);
  g_object_set_data(G_OBJECT(ed->opacity_slider), "dt-prop",
                    GINT_TO_POINTER(DT_MASKS_PROPERTY_OPACITY));
  g_signal_connect(G_OBJECT(ed->opacity_slider), "value-changed",
                   G_CALLBACK(_param_row_opacity_changed), ed);
  // same background-occlusion fix as the shape/group properties sliders and
  // the boost-factor slider: without it this slider's own opaque pill paints
  // over the row's hover/selection wash.
  dt_gui_add_class(ed->opacity_slider, "mask-props-slider");
  _style_opacity_gradient(ed->opacity_slider);
  g_signal_connect(G_OBJECT(ed->opacity_slider), "value-changed",
                   G_CALLBACK(_inline_opacity_tooltip_changed), NULL);
  // opacity_box carries the same below-row margins boost_box does, since this
  // is now a slider the editor genuinely shows rather than a parking spot for
  // one docked elsewhere
  ed->opacity_box = dt_gui_vbox(ed->opacity_slider);
  dt_gui_add_class(ed->opacity_box, "mask-param-opacity-box");

  GtkWidget *sliders_grid = gtk_grid_new();
  gtk_grid_set_column_homogeneous(GTK_GRID(sliders_grid), FALSE);
  gtk_grid_set_column_spacing(GTK_GRID(sliders_grid), DT_PIXEL_APPLY_DPI(4));
  gtk_grid_set_row_spacing(GTK_GRID(sliders_grid), DT_PIXEL_APPLY_DPI(2));

  GtkWidget *input_lbl = _make_icon_widget(_paint_param_input);
  gtk_widget_set_tooltip_text(input_lbl, _(slider_tooltip[0]));
  dt_gui_add_class(input_lbl, "mask-param-channel-icon");
  gtk_grid_attach(GTK_GRID(sliders_grid), input_lbl, 0, 0, 1, 1);
  ed->input_lbl = input_lbl;

  GtkWidget *input_slot = dt_gui_hbox();
  gtk_widget_set_hexpand(input_slot, TRUE);
  gtk_widget_set_valign(GTK_WIDGET(ed->filter[0].slider), GTK_ALIGN_CENTER);
  dt_gui_box_add(input_slot, dt_gui_expand(ed->filter[0].slider));

  GtkWidget *input_bypass_btn =
    _make_param_bypass_btn(_("temporarily disable this input channel"), ed);
  ed->input_bypass_slot = _make_param_bypass_slot(input_bypass_btn);
  ed->input_bypass_btn = input_bypass_btn;

  // an overlay, not a grid column: the slider is a windowed widget, so the eye
  // would otherwise be painted over wherever the two share pixels
  GtkWidget *input_overlay = gtk_overlay_new();
  // the overlay is no_show_all (below), so nothing else shows what is in it
  gtk_widget_show_all(input_slot);
  gtk_container_add(GTK_CONTAINER(input_overlay), input_slot);
  gtk_overlay_add_overlay(GTK_OVERLAY(input_overlay), ed->input_bypass_slot);
  // clicks on the eye box's empty space, with the eye hidden, reach the slider
  gtk_overlay_set_overlay_pass_through(GTK_OVERLAY(input_overlay),
                                       ed->input_bypass_slot, TRUE);
  gtk_widget_set_hexpand(input_overlay, TRUE);
  gtk_grid_attach(GTK_GRID(sliders_grid), input_overlay, 1, 0, 1, 1);
  ed->input_slot = input_overlay;

  GtkWidget *output_lbl = _make_icon_widget(_paint_param_output);
  gtk_widget_set_tooltip_text(output_lbl, _(slider_tooltip[1]));
  dt_gui_add_class(output_lbl, "mask-param-channel-icon");
  gtk_grid_attach(GTK_GRID(sliders_grid), output_lbl, 0, 1, 1, 1);
  ed->output_lbl = output_lbl;

  GtkWidget *output_slot = dt_gui_hbox();
  gtk_widget_set_hexpand(output_slot, TRUE);
  gtk_widget_set_valign(GTK_WIDGET(ed->filter[1].slider), GTK_ALIGN_CENTER);
  dt_gui_box_add(output_slot, dt_gui_expand(ed->filter[1].slider));

  GtkWidget *output_bypass_btn =
    _make_param_bypass_btn(_("temporarily disable this output channel"), ed);
  ed->output_bypass_slot = _make_param_bypass_slot(output_bypass_btn);
  ed->output_bypass_btn = output_bypass_btn;

  // an overlay, not a grid column: the slider is a windowed widget, so the eye
  // would otherwise be painted over wherever the two share pixels
  GtkWidget *output_overlay = gtk_overlay_new();
  // the overlay is no_show_all (below), so nothing else shows what is in it
  gtk_widget_show_all(output_slot);
  gtk_container_add(GTK_CONTAINER(output_overlay), output_slot);
  gtk_overlay_add_overlay(GTK_OVERLAY(output_overlay), ed->output_bypass_slot);
  // clicks on the eye box's empty space, with the eye hidden, reach the slider
  gtk_overlay_set_overlay_pass_through(GTK_OVERLAY(output_overlay),
                                       ed->output_bypass_slot, TRUE);
  gtk_widget_set_hexpand(output_overlay, TRUE);
  gtk_grid_attach(GTK_GRID(sliders_grid), output_overlay, 1, 1, 1, 1);
  ed->output_slot = output_overlay;

  ed->sliders_grid = sliders_grid;

  // opacity leads the expanded controls, matching where a shape row's own
  // opacity slider sits in its props editor (see _build_props_row_editor,
  // where DT_MASKS_PROPERTY_OPACITY is the first property in the table)
  GtkWidget *wrap = dt_gui_vbox(ed->opacity_box, sliders_grid, ed->boost_box);
  // id mirrors the class for direct CSS targeting alongside the existing
  // class-based rules (shared by every parametric row's own editor instance)
  gtk_widget_set_name(wrap, "mask-param-row-editor");
  dt_gui_add_class(wrap, "mask-param-row-editor");

  _update_param_row_display(ed);
  g_object_set_data_full(G_OBJECT(wrap), "param-editor", ed, g_free);

  GtkWidget *fit_on_style[] = { GTK_WIDGET(ed->filter[0].slider), ed->input_bypass_slot,
                                GTK_WIDGET(ed->filter[1].slider), ed->output_bypass_slot };
  for(int i = 0; i < G_N_ELEMENTS(fit_on_style); i++)
    g_signal_connect_object(G_OBJECT(fit_on_style[i]), "style-updated",
                            G_CALLBACK(_param_slider_style_updated), wrap, 0);

  gtk_widget_show_all(wrap);
  gtk_widget_set_no_show_all(ed->input_lbl, TRUE);
  gtk_widget_set_no_show_all(ed->input_slot, TRUE);
  gtk_widget_set_no_show_all(ed->input_bypass_slot, TRUE);
  gtk_widget_set_no_show_all(ed->input_bypass_btn, TRUE);
  gtk_widget_set_no_show_all(ed->output_lbl, TRUE);
  gtk_widget_set_no_show_all(ed->output_slot, TRUE);
  gtk_widget_set_no_show_all(ed->output_bypass_slot, TRUE);
  gtk_widget_set_no_show_all(ed->output_bypass_btn, TRUE);
  gtk_widget_set_no_show_all(ed->boost_box, TRUE);
  gtk_widget_set_no_show_all(ed->opacity_box, TRUE);
  _update_param_row_visibility(ed);
  // establish the opacity slider's soft range/visibility for this specific
  // form the same neutral no-op way _props_row_populate does for every other
  // row kind -- done after the no_show_all sequencing above (not before), so
  // a hide here (count == 0, never happens for opacity but kept consistent)
  // cannot be undone by the show_all() call above.
  {
    GList *ids = g_list_prepend(NULL, GINT_TO_POINTER(ed->formid));
    _props_row_apply(module, ids, DT_MASKS_PROPERTY_OPACITY, ed->opacity_slider,
                     &ed->opacity_last_value, TRUE);
    g_list_free(ids);
  }
  // _props_row_apply above sets the slider's real value inside a
  // DT_ENTER_GUI_UPDATE()/DT_LEAVE_GUI_UPDATE() guard (to avoid a spurious
  // history commit on every row build), which suppresses "value-changed" --
  // so the tooltip's own handler (connected above) never sees this initial
  // set and would otherwise show a stale default ("0%", the slider's
  // as-constructed value) until the user's first drag. Sync it once here,
  // directly, now that the real value is in place.
  _inline_opacity_tooltip_changed(ed->opacity_slider, NULL);
  if(picker_box_out) *picker_box_out = picker_box;
  return wrap;
}

// build one element (shape) row: invert toggle | name (select / rename / delete /
// reorder / move-to-group via DnD) | hide | solo | solo-edit, wrapped in an event
// box that drives the canvas hover and carries the selection highlight. Returns
// the row's vertical container; a parametric row packs its own always-visible
// editor into it (see _build_param_row_editor).
// Make `w` respond to a click exactly as this element's row header does: the
// SAME two handlers, plus the three context keys they read off the widget they
// fire on. Used for every surface that is "inside the element but not its
// header" -- the row header event box itself, and the docked parametric /
// properties editors below it.
//
// Without this, a click on an element's expanded editor area was consumed by no
// one and bubbled up to the enclosing group's block, so clicking inside an
// element selected its GROUP. Element rows and their editors are separate
// windowed widgets (row_vbox between them is a windowless GtkBox, which only
// ever sees events its children did not take), so each surface has to be wired
// individually -- but to the same handlers, never to a second idea of what a
// click on an element means.
static void _wire_element_click_surface(GtkWidget *w,
                                        dt_iop_module_t *module,
                                        const dt_mask_id_t fid,
                                        GtkWidget *handle,
                                        GtkWidget *name_evbox)
{
  // _row_click_press/_release read all three off `w`: the id to act on, the
  // handle the right-click actions menu anchors to, and the entry ctrl+click
  // rename swaps in. Neither uses the event's coordinates, so it does not
  // matter that these surfaces have different origins.
  g_object_set_data(G_OBJECT(w), "formid", GINT_TO_POINTER(fid));
  g_object_set_data(G_OBJECT(w), "handle-widget", handle);
  g_object_set_data(G_OBJECT(w), "name-evbox", name_evbox);
  g_signal_connect(G_OBJECT(w), "button-press-event",
                   G_CALLBACK(_row_click_press), module);
  g_signal_connect(G_OBJECT(w), "button-release-event",
                   G_CALLBACK(_row_click_release), module);
}

// a nested group's body, around and between its groups: pressing there is not
// the start of a drag, and releasing selects the nested group's element, but
// never deselects it. Only events on the body's own window count, as for a
// group's block (see _event_on_own_window): its groups' clicks bubble up here
static gboolean _event_on_own_window(GtkWidget *w, const GdkEventButton *e);

static gboolean
_subgroup_body_press(GtkWidget *w, GdkEventButton *e, dt_iop_module_t *module)
{
  return _event_on_own_window(w, e) && e->button == GDK_BUTTON_PRIMARY;
}

static gboolean
_subgroup_body_release(GtkWidget *w, GdkEventButton *e, dt_iop_module_t *module)
{
  if(!_event_on_own_window(w, e) || e->button != GDK_BUTTON_PRIMARY) return FALSE;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  const dt_mask_id_t fid = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(w), "formid"));
  if(bd && bd->panel_selected_formid != fid) _set_form_target(module, fid);
  return TRUE;
}

// an element row's header line squares off onto its editor's rail while any
// of its editors shows (.mask-row-open in darktable.css), as a group's header
// does onto its elements' rail (see _sync_group_open)
static void _sync_element_open(GtkWidget *editor, GParamSpec *pspec, gpointer row_vbox)
{
  static const char *const keys[] = { "param-editor-box", "props-editor-box",
                                      "subgroup-box" };
  gboolean open = FALSE;
  for(size_t i = 0; i < G_N_ELEMENTS(keys); i++)
  {
    GtkWidget *e = g_object_get_data(G_OBJECT(row_vbox), keys[i]);
    if(e && gtk_widget_get_visible(e)) open = TRUE;
  }
  if(open)
    dt_gui_add_class(GTK_WIDGET(row_vbox), "mask-row-open");
  else
    dt_gui_remove_class(GTK_WIDGET(row_vbox), "mask-row-open");
}

static GtkWidget *_make_shape_row(dt_iop_module_t *module,
                                  dt_masks_point_group_t *fpt,
                                  dt_masks_form_t *form,
                                  GList *group_formids,
                                  GtkWidget *group_frame)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  const dt_mask_id_t fid = fpt->formid;
  // the group the row's element is in: a group dropped on the row lands by it
  const dt_mask_id_t gcid = _group_cid_of_form(_module_mask_group(module), fid);
  GtkWidget *row = dt_gui_hbox();

  // column 0 -- drag handle (the reliable drag source for moving the shape onto
  // another group), showing this shape's own kind icon (circle/path/...), or,
  // for a parametric row, its channel code (e.g. "hz", "Cz") instead -- every
  // channel used the same generic "parametric" glyph, which carried no
  // information the name didn't already have to say, and duplicated a
  // separate badge that used to sit next to it. One slot doubling as the drag
  // affordance and the "what kind is this" indicator either way (see
  // _make_drag_handle / _make_channel_handle).
  const guint kind = _form_kind(form);
  const gchar *channel_code =
    (form->type & DT_MASKS_PARAMETRIC) ? dt_masks_parametric_type_label(form) : NULL;
  // every row kind's handle opens a full actions menu on a plain click
  // instead of selecting/deselecting directly (see _row_click_press/
  // _release, _build_shape_actions_menu) -- the menu's own contents adapt to
  // what actually makes sense for this row kind (e.g. no "solo edit" for a
  // raster/parametric row).
  // a member that is a nested group shows its own groups under its row
  // and so is an AI object stepped into on the canvas: it shows as the group it
  // is, its paths rows of that group, edited one by one as they are there
  const gboolean is_subgroup = (form->type & DT_MASKS_GROUP)
                               || ((form->type & DT_MASKS_OBJECT) && _entered_object() == fid);
  const gboolean is_drawn_shape =
    !(form->type & (DT_MASKS_PARAMETRIC | DT_MASKS_RASTER | DT_MASKS_GROUP));
  // "use sliders for opacity": opacity is either the compact value in this
  // row's header or a full slider leading its expanded controls, never both.
  // Moving it into the expanded panel is also what gives a raster row anything
  // to expand, and so a chevron -- see _model_row_is_expandable. A parametric
  // row's own copy of this decision is applied in _model_param_row_visibility,
  // since its slider only appears once the row is actually expanded.
  const gboolean opacity_sliders = _opacity_sliders();
  const gboolean expandable = _model_row_is_expandable(form->type, opacity_sliders);
  // the header line alone, which an open row shades like a group's header
  // (see _sync_element_open)
  dt_gui_add_class(row, "mask-element-header");
  // one shared tooltip -- and, further down, one shared pair of click
  // handlers (_row_click_press/_row_click_release) -- for every one of this
  // row's "non-specific" click surfaces: the lead icon, the name, and the
  // row's own background (covering the gaps between actual controls, e.g.
  // the opacity slider). A click has the exact same effect no matter which
  // of the three it lands on, so there is no reason for their tooltips (or
  // their behaviour) to read differently any more. Freed once, after the
  // last of the three widgets that needs it is built (see row_evbox below).
  gchar *row_tip =
    (form->type & DT_MASKS_RASTER)
      ? g_strdup(expandable
                 ? _("click to select, click again to deselect\n"
                     "ctrl+click to rename\n"
                     "shift+click to show/hide this raster mask's opacity slider\n"
                     "right-click to open the actions menu "
                     "(invert, solo, rename, delete)\n"
                     "drag to rearrange, or onto a group to move "
                     "this raster mask into it")
                 : _("click to select, click again to deselect\n"
                     "ctrl+click to rename\n"
                     "right-click to open the actions menu "
                     "(invert, solo, rename, delete)\n"
                     "drag to rearrange, or onto a group to move "
                     "this raster mask into it"))
    : is_subgroup
      ? g_strdup(_("a group inside this group: its own groups are shown below it\n"
                   "click to select, click again to deselect\n"
                   "ctrl+click to rename\n"
                   "shift+click to show/hide its groups\n"
                   "right-click to open the actions menu "
                   "(invert, solo, rename, delete)\n"
                   "drag to rearrange"))
    : is_drawn_shape
      ? g_strdup(_("click to select, click again to deselect\n"
                   "ctrl+click to rename\n"
                   "shift+click to show/hide this shape's expanded controls\n"
                   "right-click to open the actions menu "
                   "(invert, solo, solo-edit, rename, delete)\n"
                   "drag to rearrange, or onto a different group to move"))
      : g_strdup(_("click to select, click again to deselect\n"
                   "ctrl+click to rename\n"
                   "shift+click to show/hide this channel's expanded controls\n"
                   "right-click to open the actions menu "
                   "(invert, solo, rename, delete)\n"
                   "drag to rearrange, or onto a group to move "
                   "this channel into it"));
  GtkWidget *handle =
    channel_code ? _make_channel_handle(channel_code, row_tip)
                 : _make_drag_handle((form->type & DT_MASKS_GROUP) ? dtgtk_cairo_paint_masks_multi
                                                                   : _kind_icon_paint(kind),
                                     TRUE, row_tip);
  // no separate invert button: invert is one of the actions menu's own items
  // now (right-click, see _build_shape_actions_menu). An inverted shape
  // fills its handle, mirroring .mask-op-inverted on groups.
  if(fpt->state & DT_MASKS_STATE_INVERSE)
    dt_gui_add_class(handle, "mask-list-handle-inverted");
  g_object_set_data(G_OBJECT(handle), "formid", GINT_TO_POINTER(fid));
  // self-reference, so _row_click_press/_release can resolve "handle-widget"
  // from any of the three widgets they're connected to alike (see below)
  g_object_set_data(G_OBJECT(handle), "handle-widget", handle);
  g_signal_connect(G_OBJECT(handle), "button-press-event", G_CALLBACK(_row_click_press),
                   module);
  g_signal_connect(G_OBJECT(handle), "button-release-event",
                   G_CALLBACK(_row_click_release), module);
  g_signal_connect(G_OBJECT(handle), "drag-data-get", G_CALLBACK(_masks_row_drag_get),
                   NULL);
  g_signal_connect(G_OBJECT(handle), "drag-begin", G_CALLBACK(_row_drag_begin), module);
  gtk_drag_source_set(handle, GDK_BUTTON1_MASK, _mask_row_dnd, 1, GDK_ACTION_MOVE);

  // name (expands): see _row_click_press/_row_click_release for the full set
  // of gestures, shared with the handle above and the row's own background
  // below. The selected row is shown by the border highlight (see row_vbox
  // below). The type prefix (e.g. "circle", "Cz") is stripped from the
  // displayed text -- the handle already says what kind this is (icon, or
  // channel code for a parametric row), so repeating it in the label would
  // be redundant (see _form_type_prefix).
  gchar *display_name = _form_display_name(form);
  GtkWidget *name = gtk_label_new(display_name);
  g_free(display_name);
  gtk_label_set_xalign(GTK_LABEL(name), 0.0f);
  gtk_label_set_ellipsize(GTK_LABEL(name), PANGO_ELLIPSIZE_MIDDLE);
  // ellipsize alone only kicks in once the label is squeezed below its own
  // natural (full-text) width -- without a cap on that natural width, a long
  // name's evbox (fixed at 50dpi via size_request, a *minimum* only) still
  // asks for its full un-ellipsized width whenever the row has room to grant
  // it, at the opacity slot's own expanding-child expense. Capping natural
  // width in characters here is what actually makes the 50dpi column, and
  // not the slider, absorb a long name.
  gtk_label_set_max_width_chars(GTK_LABEL(name), 1);
  // a little breathing room between the lead handle and the name, so the
  // text doesn't sit flush against the handle's own rounded plate
  dt_gui_add_class(name, "mask-row-name");
  GtkWidget *evbox = gtk_event_box_new();
  // the rename gesture (see _start_rename_element) swaps evbox's own child
  // for a GtkEntry in place, so evbox must contain the label alone -- the
  // solo badge is packed as evbox's own sibling in `row` instead (see below).
  gtk_container_add(GTK_CONTAINER(evbox), name);
  // hexpand is set below, once name_expand is known (a parametric row's name
  // must not claim any of the header's free width -- see name_expand).
  gtk_widget_set_tooltip_text(evbox, row_tip);
  g_object_set_data(G_OBJECT(handle), "name-evbox", evbox);
  // self-reference, mirroring handle's own above
  g_object_set_data(G_OBJECT(evbox), "name-evbox", evbox);
  g_object_set_data(G_OBJECT(evbox), "handle-widget", handle);
  g_object_set_data(G_OBJECT(evbox), "formid", GINT_TO_POINTER(fid));
  // also tagged with this row's own group's member ids, so a group/empty-group
  // drag dropped on this row (not just the group's header) still resolves to
  // the right group -- see _element_row_drag_received.
  if(group_formids)
    g_object_set_data_full(G_OBJECT(evbox), "group-formids", g_list_copy(group_formids),
                           (GDestroyNotify)g_list_free);
  g_object_set_data(G_OBJECT(evbox), "group-key", GINT_TO_POINTER(gcid));
  // the name is a drop target (drop another shape here to reorder, or a whole
  // group/empty group here to land next to this row), and -- like the grip
  // handle in column 0 -- also a drag source, so grabbing the name starts the
  // same reorder/move-to-group drag (a plain press returns FALSE, letting the
  // drag source arm; selection happens on release, see _row_click_release).
  gtk_drag_dest_set(evbox, GTK_DEST_DEFAULT_ALL, _mask_hdr_dnd,
                    G_N_ELEMENTS(_mask_hdr_dnd), GDK_ACTION_MOVE);
  g_signal_connect(G_OBJECT(evbox), "drag-data-received",
                   G_CALLBACK(_element_row_drag_received), module);
  g_signal_connect(G_OBJECT(evbox), "drag-data-get", G_CALLBACK(_masks_row_drag_get),
                   NULL);
  if(group_frame) g_object_set_data(G_OBJECT(evbox), "group-frame", group_frame);
  g_signal_connect(G_OBJECT(evbox), "drag-motion", G_CALLBACK(_element_drop_motion),
                   NULL);
  g_signal_connect(G_OBJECT(evbox), "drag-leave", G_CALLBACK(_element_drop_leave), NULL);
  gtk_drag_source_set(evbox, GDK_BUTTON1_MASK, _mask_row_dnd, 1, GDK_ACTION_MOVE);
  g_signal_connect(G_OBJECT(evbox), "drag-begin", G_CALLBACK(_row_drag_begin), module);
  g_signal_connect(G_OBJECT(evbox), "button-press-event", G_CALLBACK(_row_click_press),
                   module);
  g_signal_connect(G_OBJECT(evbox), "button-release-event",
                   G_CALLBACK(_row_click_release), module);

  // this slot holds a parametric row's "show output" toggle (see
  // _masks_param_inout_toggled), keeping alignment with the shape rows
  // above/below -- NULL for a drawn shape or a raster row (see their own
  // branches below): solo-edit is now reachable from a drawn shape's own
  // actions menu instead of a dedicated icon (see _build_shape_actions_menu).
  GtkWidget *soloedit;
  GtkWidget *param_editor = NULL;
  GtkWidget *param_picker_box = NULL;
  // properties expander: every drawn shape gets one (see
  // _make_props_row_toggle), and a raster row gets one too once "show opacity
  // slider in expanded elements" gives it something to show (see
  // _model_row_is_expandable). Parametric rows do not -- their existing
  // in/out toggle above (soloedit, in this branch) already reveals opacity too.
  GtkWidget *props_toggle = NULL;
  GtkWidget *props_editor_box = NULL;
  // the row's compact opacity value, shown inline in the header next to the
  // name -- mirrors the group header's own treatment, and every element kind
  // has one unless "use sliders for opacity" has moved opacity into the
  // expanded panel instead. A shape or raster row wraps its own hidden
  // props-editor slider (_style_inline_opacity_box); a parametric row reads
  // the slider in its editor instead, so the two can never disagree.
  GtkWidget *opacity_box = NULL;
  // the props editor inside opacity_box, which is only a wrapper (see
  // _style_inline_opacity_box) -- the editor is what owns the slider's
  // lifetime, so it has to be kept for as long as the row is alive.
  GtkWidget *inline_opacity_editor = NULL;
  // the row's own properties/expanded-view toggle, whichever widget that is
  // for this row kind (see below): the chevron the row header shows, which
  // shift+click on the lead handle or the title also drives programmatically
  // (see _row_click_release, and _auto_expand_selected_row). NULL for a row
  // with nothing to expand, which is a raster row while "show opacity slider
  // in expanded elements" is off.
  GtkWidget *expand_toggle = NULL;
  // a nested group's own groups, shown under its row (see _pack_subgroup)
  GtkWidget *subgroup_box = NULL;
  if(is_subgroup)
  {
    // its opacity and inversion apply to its finished mask, as a shape's do
    // to the shape's; its chevron shows its groups instead of properties
    soloedit = NULL;
    subgroup_box = dt_gui_vbox();
    gtk_widget_set_name(subgroup_box, "mask-subgroup-elements");
    dt_gui_add_class(subgroup_box, "masks-list");
    dt_gui_add_class(subgroup_box, "mask-group-elements");
    if(opacity_sliders)
    {
      GtkWidget *ex_op = _build_props_row_editor(module, fid, FALSE, TRUE, FALSE);
      dt_gui_add_class(ex_op, "mask-group-opacity-editor");
      dt_gui_box_add(subgroup_box, ex_op);
    }
    else
    {
      inline_opacity_editor = _build_props_row_editor(module, fid, FALSE, TRUE, FALSE);
      opacity_box = _style_inline_opacity_box(inline_opacity_editor, module);
    }
    _pack_subgroup(module, form, subgroup_box);

    // open unless closed by hand, and always while it holds the selection
    const gboolean holds_selection =
      (dt_is_valid_maskid(bd->panel_selected_formid)
       && _point_node_owner(form, bd->panel_selected_formid, NULL))
      || (dt_is_valid_maskid(bd->panel_selected_group_cid)
          && _point_node_owner(form, bd->panel_selected_group_cid, NULL));
    gpointer stored = NULL;
    const gboolean expanded =
      holds_selection || !bd->masks_props_expanded
      || !g_hash_table_lookup_extended(bd->masks_props_expanded, GINT_TO_POINTER(fid), NULL,
                                       &stored)
      || GPOINTER_TO_INT(stored);
    gtk_widget_set_visible(subgroup_box, expanded);
    expand_toggle = dtgtk_togglebutton_new(_paint_param_inout, 0, NULL);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(expand_toggle), expanded);
    dt_gui_add_class(expand_toggle, "dt_transparent_background");
    dt_gui_add_class(expand_toggle, "mask-row-expander");
    gtk_widget_set_valign(expand_toggle, GTK_ALIGN_CENTER);
    gtk_widget_set_tooltip_text(expand_toggle, _("show/hide this group's groups"));
    g_object_set_data(G_OBJECT(expand_toggle), "props-key", GINT_TO_POINTER(fid));
    g_object_set_data(G_OBJECT(expand_toggle), "elem-box", subgroup_box);
    g_signal_connect(G_OBJECT(expand_toggle), "toggled",
                     G_CALLBACK(_subgroup_expand_toggled), module);
  }
  else if(form->type & DT_MASKS_PARAMETRIC)
  {
    const dt_masks_point_parametric_t *p = form->points ? form->points->data : NULL;
    const gboolean out = p && p->in_out;
    soloedit = dtgtk_togglebutton_new(_paint_param_inout, 0, NULL);
    gtk_widget_set_valign(soloedit, GTK_ALIGN_CENTER);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(soloedit), out);
    // this is an expander (chevron down = expanded, left = collapsed)
    dt_gui_add_class(soloedit, "mask-row-expander");
    dt_gui_add_class(soloedit, "dt_transparent_background");
    gtk_widget_set_tooltip_text(
      soloedit,
      _("show/hide this channel's expanded controls (full input and output sliders)"));
    g_object_set_data(G_OBJECT(soloedit), "formid", GINT_TO_POINTER(fid));
    g_signal_connect(G_OBJECT(soloedit), "toggled",
                     G_CALLBACK(_masks_param_inout_toggled), module);
    expand_toggle = soloedit;
    // built here (rather than after row_vbox exists, below) so its two picker
    // buttons are ready to pack into the header actions cluster next to the
    // expander/power icons -- they are per-channel controls, not part of the
    param_editor = _build_param_row_editor(module, form, &param_picker_box);
    dt_masks_param_row_editor_t *ped =
      param_editor ? g_object_get_data(G_OBJECT(param_editor), "param-editor") : NULL;

    if(ped)
    {
      ped->name_evbox = evbox;

      if(ped->opacity_slider && !opacity_sliders)
      {
        opacity_box = _make_inline_opacity_value_widget(ped->opacity_slider, module);
        gtk_widget_set_halign(opacity_box, GTK_ALIGN_END);
        gtk_widget_set_valign(opacity_box, GTK_ALIGN_CENTER);
      }

      _update_param_row_visibility(ped);
    }
  }
  else if(form->type & DT_MASKS_RASTER)
  {
    // a raster mask has no on-canvas geometry to solo-edit, so this slot has
    // nothing to hold. Opacity is its only property (modify_property is NULL
    // for a raster form).
    soloedit = NULL;
    if(!opacity_sliders)
    {
      inline_opacity_editor = _build_props_row_editor(module, fid, FALSE, TRUE, FALSE);
      opacity_box = _style_inline_opacity_box(inline_opacity_editor, module);
    }
    // that inline value is all a raster row has, so by default there is
    // nothing behind an expander and the row carries no chevron. "show
    // opacity slider in expanded elements" gives it a full slider to show,
    // and with it the same expander every other element row has.
    if(expandable)
    {
      props_toggle = _make_props_row_toggle(
        module, fid, FALSE, TRUE, FALSE,
        _("show/hide this raster mask's opacity slider"), &props_editor_box);
      expand_toggle = props_toggle;
    }
  }
  else
  {
    // no more dedicated solo-edit icon -- solo-edit (like every other
    // gesture this row offers) is now also reachable from the actions menu
    // opened by a plain click on the row's own lead handle (see
    // _build_shape_actions_menu / _row_click_press).
    soloedit = NULL;

    // opacity is shown inline in the header next to the name (see
    // _style_inline_opacity_box) -- everything else this shape has (size,
    // hardness, feather, rotation, curvature, compression, cleanup,
    // smoothing, refine-mask-boundary) stays behind its own separate
    // expander. That expander excludes opacity, since the header already
    // shows it -- unless "use sliders for opacity" swaps the two round: a full
    // slider leading the expanded controls, and no compact value in the
    // header.
    if(!opacity_sliders)
    {
      inline_opacity_editor = _build_props_row_editor(module, fid, FALSE, TRUE, FALSE);
      opacity_box = _style_inline_opacity_box(inline_opacity_editor, module);
    }
    // "shape properties in subpanel" shows them there instead, for the
    // selected shape (see _props_panel_sync), so the row has nothing to expand
    if(!_shape_props_subpanel())
    {
      const char *props_tip =
        (form->type & DT_MASKS_OBJECT)
          ? _("show/hide this object's expanded controls (smoothing, cleanup, etc.)")
          : _("show/hide this shape's expanded controls (size, hardness, etc.)");
      props_toggle =
        _make_props_row_toggle(module, fid, FALSE, FALSE, !opacity_sliders,
                               props_tip, &props_editor_box);
      expand_toggle = props_toggle;
    }
  }

  if(expand_toggle)
  {
    g_object_set_data(G_OBJECT(handle), "expand-toggle", expand_toggle);
    g_object_set_data(G_OBJECT(evbox), "expand-toggle", expand_toggle);
  }

  // solo/solo-edit status badge (mutually exclusive, see MASK_SOLO_BADGE_*):
  // blank unless this element is currently soloed or solo-edited (see
  // _toggle_solo_form/_toggle_soloedit/_update_shape_row_state) -- occupies a
  // fixed cell in the row's own badge stack (see _make_badge_stack below)
  // regardless. Only the solo state is clickable to clear (see
  // _solo_badge_form_press) -- solo-edit is toggled from the actions menu.
  const gboolean elem_disabled = (fpt->state & DT_MASKS_STATE_DISABLE) != 0;
  GtkWidget *solo_badge = _make_solo_status_badge();
  _set_solo_status_badge(solo_badge, elem_disabled ? MASK_SOLO_BADGE_DISABLE
                                     : bd->solo_formid == fid ? MASK_SOLO_BADGE_SOLO
                                                              : MASK_SOLO_BADGE_NONE);
  g_object_set_data(G_OBJECT(solo_badge), "formid", GINT_TO_POINTER(fid));
  g_signal_connect(G_OBJECT(solo_badge), "button-press-event",
                   G_CALLBACK(_solo_badge_form_press), module);

  // low-opacity warning: blank unless this element's opacity is under
  // MASK_LOW_OPACITY_WARN. Its initial state is set by the
  // _refresh_lowop_badges call at the end of _build_masks_list, once the
  // row is registered in bd->masks_row_map (it isn't yet, here).
  GtkWidget *lowop_badge = _make_lowop_badge();

  // a linked element carries a chain icon ending its name column. It cannot
  // sit right after the text: the name is capped to one character of natural
  // width (see gtk_label_set_max_width_chars above) and only grows by
  // expanding into the whole column, so the icon goes at the column's end.
  // A raster element has nothing shared to edit, so it never shows one
  GtkWidget *name_slot = evbox;
  // linked across modules, or referenced more than once by this mask itself:
  // both are the same shape shown in two places, so both earn the chain icon
  const int uses_here = _model_form_uses_in_mask(module, fid);
  if(_model_form_is_linked(form) || uses_here > 1)
  {
    gchar *linked_tip = _linked_tooltip(module, fid, form, uses_here);
    if(linked_tip)
    {
      GtkWidget *chain = _make_linked_badge(linked_tip);
      g_free(linked_tip);
      name_slot = dt_gui_hbox(dt_gui_expand(evbox), chain);
      gtk_widget_show(chain);
      gtk_widget_show(name_slot);
    }
  }

  GtkWidget *action_icon = (form->type & DT_MASKS_PARAMETRIC) ? param_picker_box : NULL;
  _pack_row_header(row, handle, name_slot, opacity_box,
                   _make_badge_stack(lowop_badge, solo_badge), action_icon,
                   expand_toggle);

  // disabled elements dim their controls while keeping the badge at 1.0 opacity
  if(elem_disabled)
  {
    gtk_widget_set_opacity(handle, 0.45);
    gtk_widget_set_opacity(evbox, 0.45);
    if(opacity_box) gtk_widget_set_opacity(opacity_box, 0.45);
    if(action_icon) gtk_widget_set_opacity(action_icon, 0.45);
    gtk_widget_set_opacity(row, 1.0);
  }
  else if(fpt->state & DT_MASKS_STATE_HIDDEN)
  {
    gtk_widget_set_opacity(row, 0.45);
  }

  // an event box around the row drives the canvas hover feedback (labels +
  // highlight). It needs a real window (visible_window TRUE) so crossings
  // into the row's own buttons are reported as GDK_NOTIFY_INFERIOR and the
  // hover stays active over the whole row -- with an input-only box the hover
  // only triggered in the gaps between the child widgets.
  GtkWidget *row_evbox = gtk_event_box_new();
  gtk_event_box_set_visible_window(GTK_EVENT_BOX(row_evbox), TRUE);
  gtk_container_add(GTK_CONTAINER(row_evbox), row);
  // so a click landing in a gap between this row's own controls (see
  // _row_click_press/_row_click_release) has the exact same effect as
  // clicking the handle/name directly.
  _wire_element_click_surface(row_evbox, module, fid, handle, evbox);
  gtk_widget_set_tooltip_text(row_evbox, row_tip);
  g_free(row_tip); // last of the three widgets that needed it (handle, evbox, row_evbox)
  // hovering this row highlights just this shape on the canvas
  g_object_set_data_full(G_OBJECT(row_evbox), "hover-formids",
                         g_list_prepend(NULL, GINT_TO_POINTER(fid)),
                         (GDestroyNotify)g_list_free);
  gtk_widget_add_events(row_evbox, GDK_ENTER_NOTIFY_MASK | GDK_LEAVE_NOTIFY_MASK);
  g_signal_connect(G_OBJECT(row_evbox), "enter-notify-event", G_CALLBACK(_row_crossing),
                   module);
  g_signal_connect(G_OBJECT(row_evbox), "leave-notify-event", G_CALLBACK(_row_crossing),
                   module);
  // also a drop target (same reasoning as evbox above): the gaps must accept a
  // group/empty-group/shape drop too, not just reject it and block bubbling.
  if(group_formids)
    g_object_set_data_full(G_OBJECT(row_evbox), "group-formids",
                           g_list_copy(group_formids), (GDestroyNotify)g_list_free);
  g_object_set_data(G_OBJECT(row_evbox), "group-key", GINT_TO_POINTER(gcid));
  gtk_drag_dest_set(row_evbox, GTK_DEST_DEFAULT_ALL, _mask_hdr_dnd,
                    G_N_ELEMENTS(_mask_hdr_dnd), GDK_ACTION_MOVE);
  g_signal_connect(G_OBJECT(row_evbox), "drag-data-received",
                   G_CALLBACK(_element_row_drag_received), module);
  if(group_frame) g_object_set_data(G_OBJECT(row_evbox), "group-frame", group_frame);
  g_signal_connect(G_OBJECT(row_evbox), "drag-motion", G_CALLBACK(_element_drop_motion),
                   NULL);
  g_signal_connect(G_OBJECT(row_evbox), "drag-leave", G_CALLBACK(_element_drop_leave),
                   NULL);

  // each row gets its own vertical container so the parametric channel editor
  // can be docked directly underneath the row it belongs to (see below). The
  // border highlight is on this box (a GtkBox renders its CSS frame reliably;
  // a GtkEventBox does not) and carries the form id so _update_row_selection /
  // _dock_editor_under can find it without a rebuild.
  GtkWidget *row_vbox = dt_gui_vbox();
  // unique per-kind id (#mask-shape-row) -- shared by every element kind (drawn
  // shape, parametric, raster), which all go through this same function; see
  // .mask-panel-row in darktable.css for the base styling shared with every
  // other row/header kind in the panel.
  gtk_widget_set_name(row_vbox, "mask-shape-row");
  dt_gui_add_class(row_vbox, "mask-panel-row");
  dt_gui_box_add(row_vbox, row_evbox);
  g_object_set_data(G_OBJECT(row_evbox), "row-vbox", row_vbox);
  g_object_set_data(G_OBJECT(evbox), "row-vbox", row_vbox);
  g_object_set_data(G_OBJECT(handle), "row-vbox", row_vbox);

  g_object_set_data(G_OBJECT(row_vbox), "mask-row", GINT_TO_POINTER(1));
  g_object_set_data(G_OBJECT(row_vbox), "formid", GINT_TO_POINTER(fid));
  // which reference this row is: a mask can hold the same shape in two groups,
  // and then form id alone no longer identifies a row (see _masks_row_for_point)
  g_object_set_data(G_OBJECT(row_vbox), "row-point", fpt);
  // index this row for O(1) lookup by form id (see _masks_row_widget); the map is
  // cleared at the top of _build_masks_list, so entries never outlive their widget.
  // One entry per form holds every row built for it, in build order. The list is
  // stolen and re-inserted rather than looked up and updated in place: the map
  // frees the list it holds, and inserting over a head it already stores would
  // free the very list this row was just appended to.
  if(bd->masks_row_map)
  {
    GSList *rows = g_hash_table_lookup(bd->masks_row_map, GINT_TO_POINTER(fid));
    g_hash_table_steal(bd->masks_row_map, GINT_TO_POINTER(fid));
    rows = g_slist_append(rows, row_vbox);
    g_hash_table_insert(bd->masks_row_map, GINT_TO_POINTER(fid), rows);
  }
  // the header's compact opacity control is its own editor, separate from the
  // expanded properties box below: a sibling refresh has to find both
  if(inline_opacity_editor)
    g_object_set_data(G_OBJECT(row_vbox), "inline-opacity-editor-box",
                      inline_opacity_editor);
  // tag the row's own interactive widgets so _update_shape_row_state can refresh
  // them in place (toggle states, opacity) without a full list rebuild.
  g_object_set_data(G_OBJECT(row_vbox), "row-hbox", row);
  g_object_set_data(G_OBJECT(row_vbox), "handle-widget", handle);
  g_object_set_data(G_OBJECT(row_vbox), "name-evbox", evbox);
  if(action_icon) g_object_set_data(G_OBJECT(row_vbox), "action-icon", action_icon);
  g_object_set_data(G_OBJECT(row_vbox), "solo-badge", solo_badge);
  g_object_set_data(G_OBJECT(row_vbox), "lowop-badge", lowop_badge);
  if(expand_toggle) g_object_set_data(G_OBJECT(row_vbox), "expand-toggle", expand_toggle);
  // tag the properties editor box too (mirrors "param-editor-box" below) so
  // _update_shape_row_state can make it insensitive while this row is
  // solo-suppressed -- see the props_editor_box comment above (raster/shape
  // rows use it interchangeably, parametric rows have their own always-visible
  // "param-editor-box" instead).
  if(props_editor_box)
    g_object_set_data(G_OBJECT(row_vbox), "props-editor-box", props_editor_box);
  // same, for the always-visible inline opacity box (shape/raster rows) --
  // see _update_shape_row_state's own "opacity-editor-box" lookup
  if(opacity_box)
    g_object_set_data(G_OBJECT(row_vbox), "opacity-editor-box", opacity_box);
  if(dt_is_valid_maskid(bd->panel_selected_formid) && fid == bd->panel_selected_formid)
    dt_gui_add_class(row_vbox, "mask-list-row-selected");
  if(bd->solo_formid == fid)
    dt_gui_add_class(row_vbox, "mask-list-row-solo");

  // every parametric mask row gets its own permanently visible slider editor
  // (see _build_param_row_editor) -- no expand/collapse or docking needed
  if(param_editor)
  {
    // indent/inset entirely via CSS (.mask-param-row-editor's margin-left/
    // margin-right in darktable.css), not hardcoded here, so a theme can
    // restyle it without a rebuild
    // wrap in a real-window event box so hovering any of its sliders/pickers
    // (not just the row header above) also drives the row's hover highlight
    // -- a windowless box only sees crossings in the gaps between its child
    // widgets (same reasoning as row_evbox above).
    GtkWidget *param_evbox = gtk_event_box_new();
    gtk_event_box_set_visible_window(GTK_EVENT_BOX(param_evbox), TRUE);
    gtk_container_add(GTK_CONTAINER(param_evbox), param_editor);
    g_object_set_data_full(G_OBJECT(param_evbox), "hover-formids",
                           g_list_prepend(NULL, GINT_TO_POINTER(fid)),
                           (GDestroyNotify)g_list_free);
    gtk_widget_add_events(param_evbox, GDK_ENTER_NOTIFY_MASK | GDK_LEAVE_NOTIFY_MASK);
    g_signal_connect(G_OBJECT(param_evbox), "enter-notify-event",
                     G_CALLBACK(_row_crossing), module);
    g_signal_connect(G_OBJECT(param_evbox), "leave-notify-event",
                     G_CALLBACK(_row_crossing), module);
    // clicking the editor's own background selects this element, not its group
    _wire_element_click_surface(param_evbox, module, fid, handle, evbox);
    dt_gui_box_add(row_vbox, param_evbox);
    // "param-editor-box" must keep pointing at the editor itself (not the
    // hover wrapper): _masks_param_inout_toggled / _masks_param_compact_press
    // look up the "param-editor" data that _build_param_row_editor attached
    // to this exact widget.
    g_object_set_data(G_OBJECT(row_vbox), "param-editor-box", param_editor);
  }

  // shape/raster rows' own properties editor, docked and hover-wrapped the
  // same way the parametric editor above is (see _make_props_row_toggle for
  // the toggle that shows/hides it).
  if(props_editor_box)
  {
    // indent/inset entirely via CSS (.mask-props-row-editor's margin-left/
    // margin-right in darktable.css), not hardcoded here
    GtkWidget *props_evbox = gtk_event_box_new();
    gtk_event_box_set_visible_window(GTK_EVENT_BOX(props_evbox), TRUE);
    gtk_container_add(GTK_CONTAINER(props_evbox), props_editor_box);
    g_object_set_data_full(G_OBJECT(props_evbox), "hover-formids",
                           g_list_prepend(NULL, GINT_TO_POINTER(fid)),
                           (GDestroyNotify)g_list_free);
    gtk_widget_add_events(props_evbox, GDK_ENTER_NOTIFY_MASK | GDK_LEAVE_NOTIFY_MASK);
    g_signal_connect(G_OBJECT(props_evbox), "enter-notify-event",
                     G_CALLBACK(_row_crossing), module);
    g_signal_connect(G_OBJECT(props_evbox), "leave-notify-event",
                     G_CALLBACK(_row_crossing), module);
    // same as the parametric editor above: this is still inside the element
    _wire_element_click_surface(props_evbox, module, fid, handle, evbox);
    dt_gui_box_add(row_vbox, props_evbox);
  }

  // indented under the row by the rail every group's elements have (see
  // .mask-group-elements in darktable.css). Its groups are selected, dropped
  // onto and dimmed through their own headers, as the top list's are
  if(subgroup_box)
  {
    // its own window, so a click between its groups stays inside the element
    // instead of selecting the group holding it, which would step out of the AI
    // object stepped into (see _set_group_target_ext)
    GtkWidget *sub_evbox = gtk_event_box_new();
    gtk_event_box_set_visible_window(GTK_EVENT_BOX(sub_evbox), TRUE);
    gtk_container_add(GTK_CONTAINER(sub_evbox), subgroup_box);
    g_object_set_data(G_OBJECT(sub_evbox), "formid", GINT_TO_POINTER(fid));
    g_signal_connect(G_OBJECT(sub_evbox), "button-press-event",
                     G_CALLBACK(_subgroup_body_press), module);
    g_signal_connect(G_OBJECT(sub_evbox), "button-release-event",
                     G_CALLBACK(_subgroup_body_release), module);
    g_object_bind_property(subgroup_box, "visible", sub_evbox, "visible",
                           G_BINDING_SYNC_CREATE);
    dt_gui_box_add(row_vbox, sub_evbox);
    g_object_set_data(G_OBJECT(row_vbox), "subgroup-box", subgroup_box);
  }

  // disconnected with row_vbox, which is destroyed together with its editors
  GtkWidget *const editors[] = { param_editor, props_editor_box, subgroup_box };
  for(size_t i = 0; i < G_N_ELEMENTS(editors); i++)
    if(editors[i])
      g_signal_connect_object(editors[i], "notify::visible",
                              G_CALLBACK(_sync_element_open), row_vbox, 0);
  _sync_element_open(NULL, NULL, row_vbox);

  // this element contributes nothing to the mask -- it is disabled, it is
  // suppressed by a solo, or the group holding it is bypassed -- so none of
  // this row's controls can have any visible effect. Grey them out and dim the
  // row, exactly as _update_shape_row_state does for the same three states: it
  // is what every later refresh applies, so a freshly built row that skipped
  // one of them showed live controls on a row labelled disabled until
  // something else repainted it. Only the editors and the actions column are
  // made insensitive -- never row_vbox/row_evbox itself, so the row stays
  // selectable and draggable, the same carve-out solo makes.
  if(_member_group_bypassed(_module_mask_group(module), fid)
     || elem_disabled
     || (fpt->state & DT_MASKS_STATE_HIDDEN))
  {
    // a disabled row dims its own parts instead (see above), which keeps its
    // badge readable at full strength
    if(!elem_disabled) gtk_widget_set_opacity(row, 0.45);
    if(expand_toggle) gtk_widget_set_sensitive(expand_toggle, FALSE);
    if(action_icon) gtk_widget_set_sensitive(action_icon, FALSE);
    if(param_editor) gtk_widget_set_sensitive(param_editor, FALSE);
    if(props_editor_box) gtk_widget_set_sensitive(props_editor_box, FALSE);
    if(opacity_box) gtk_widget_set_sensitive(opacity_box, FALSE);
  }

  return row_vbox;
}

// Order-independent fold of a {key -> flag} GHashTable into an accumulator, so
// its contribution to the signature does not depend on GHashTable iteration
// order (which is unspecified).
static guint64 _fold_flag_table(GHashTable *t)
{
  if(!t) return 0;
  guint64 acc = 1469598103934665603ULL; // FNV offset basis, just a seed
  GHashTableIter it;
  gpointer k, v;
  g_hash_table_iter_init(&it, t);
  while(g_hash_table_iter_next(&it, &k, &v))
    acc += (guint64)GPOINTER_TO_INT(k) * 2654435761u + (guint64)(GPOINTER_TO_INT(v) != 0);
  return acc;
}

// A hash of everything _build_masks_list builds the tree from: the mask model
// (dt_masks_group_hash already folds every point's formid/state/opacity/
// refinement in order plus each leaf form's own config, so add/delete/reorder/
// operator/opacity/refinement/solo-via-HIDDEN and parametric config all move it)
// plus the UI-state the build consults (mask mode, selection/solo,
// cluster/props expansion). When this is unchanged since the last build the
// rebuilt tree would be byte-identical, so the whole teardown/rebuild can be
// skipped.
dt_hash_t _masks_list_signature(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);

  dt_hash_t sig = dt_masks_group_hash(DT_INITHASH, grp);

  // dt_masks_group_hash does not fold form->name or orphan point states, but
  // the rows display them, so fold each member form's state and name in points
  // order, nested groups' members included
  GList *pts = _mask_points(grp);
  for(GList *p = pts; p; p = g_list_next(p))
  {
    const dt_masks_point_group_t *pt = p->data;
    sig = dt_hash(sig, &pt->formid, sizeof(pt->formid));
    sig = dt_hash(sig, &pt->state, sizeof(pt->state));
    sig = dt_hash(sig, &pt->group_opacity, sizeof(pt->group_opacity));
    sig = dt_hash(sig, &pt->opacity, sizeof(pt->opacity));
    sig = dt_hash(sig, &pt->refinement, sizeof(pt->refinement));
    const dt_masks_form_t *f = dt_masks_get_from_id(darktable.develop, pt->formid);
    if(f)
    {
      // what the row shows, which for a raster element can be its source's name
      gchar *shown = _form_display_name(f);
      sig = dt_hash(sig, shown, strlen(shown));
      g_free(shown);
    }
    if(pt->name[0]) sig = dt_hash(sig, pt->name, strlen(pt->name));
    // the chain icon and its tooltip follow which modules share the form,
    // which another module changes without touching this group
    GList *users = _model_form_users(pt->formid);
    for(GList *u = users; u; u = g_list_next(u)) sig = dt_hash(sig, &u->data, sizeof(u->data));
    g_list_free(users);
  }
  g_list_free(pts);

  const uint32_t mode = module->blend_params->mask_mode;
  sig = dt_hash(sig, &mode, sizeof(mode));

  // selection / solo (drive per-row/per-header CSS classes and badges), and
  // the AI object stepped into, whose row then shows as its group (see
  // _make_shape_row): stepping in or out changes nothing else hashed here
  const int32_t ui[6] = {
    bd->panel_selected_formid,   bd->panel_selected_group_cid, bd->solo_formid,
    (int32_t)bd->solo_group_key, bd->soloedit_formid,          _entered_object()
  };
  sig = dt_hash(sig, ui, sizeof(ui));

  // cluster / props expansion (each drives a revealer's initial state)
  const guint64 folds[2] = { _fold_flag_table(bd->masks_cluster_expanded),
                             _fold_flag_table(bd->masks_props_expanded) };
  sig = dt_hash(sig, folds, sizeof(folds));

  // pending (uncommitted, on-canvas) shape being drawn for THIS module: not
  // itself part of grp->points, so dt_masks_group_hash above never sees it --
  // without this the pending-row synthesis in _build_masks_list would be
  // silently skipped on creation-start/creation-cancel, same signature-
  // omission trap already fixed twice elsewhere in this file. The shape's own
  // type is enough (no need for live geometry/smoothing/cleanup here -- the
  // pending row's sliders are updated in place, not by a rebuild, see
  // dt_iop_gui_blend_sync_pending_ai_sliders).
  {
    const dt_masks_form_gui_t *fg = darktable.develop->form_gui;
    const dt_masks_form_t *pending = (fg && fg->creation && fg->creation_module == module)
                                       ? darktable.develop->form_visible
                                       : NULL;
    const int32_t pending_type = pending ? (int32_t)pending->type : -1;
    sig = dt_hash(sig, &pending_type, sizeof(pending_type));
  }

  return sig;
}

// Model-side reconciliation for the mask panel: settle everything the panel
// derives from -- realizing a just-drawn shape into its staged group, seeding
// the foundation group, renumbering groups, dropping stale solo state and
// picking an initial selection -- before a single widget is built. Touches no
// widgets.
//
// Split out of _build_masks_list, which interleaved this with widget packing.
// That interleaving is why "drawing a shape takes effect" really meant "a
// rebuild happened to run": these mutations sat on the panel's render path
// rather than at the point of the edit.
//
// Returns FALSE when there is nothing to render at all.
static gboolean _masks_panel_reconcile(dt_iop_module_t *module,
                                       dt_masks_form_t *grp,
                                       const gboolean flexi)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;

  // an AI object left without paths (by an earlier cleanup, or a path form
  // lost otherwise) is never shown or edited again: drop it, and any member
  // whose form is gone (see _model_prune_dangling_members). A list that does
  // not start with a marker gets one (see dt_masks_group_ensure_marker). All
  // of it renders as before, so no history item is needed; the next one
  // carries the change
  if(flexi && darktable.develop)
  {
    dt_pthread_mutex_lock(&darktable.develop->history_mutex);
    const int pruned = dt_masks_prune_empty_objects(&darktable.develop->forms);
    const int dangling = _model_prune_dangling_members(grp);
    _model_ensure_a_group(grp);
    dt_pthread_mutex_unlock(&darktable.develop->history_mutex);
    if(pruned) _queue_link_peers_rebuild(module);
    if(dangling)
      dt_print(DT_DEBUG_MASKS, "[masks] '%s': dropped %d member(s) whose form is gone",
               module->op, dangling);
  }

  // group numbers are identities, not positions: forget the ones whose group is
  // gone, then number any group still without one (see _group_ordinal_any).
  // Before any packing, so every header and caption below reads the same,
  // already-settled number.
  if(flexi)
  {
    _prune_group_ordinals(module);
    _assign_group_ordinals(module);
  }
  _prune_stale_solo(module);
  _model_raster_names_follow_sources(grp);

  // one line per rebuild describing what the panel is about to render from, so a
  // panel that goes blank/empty can be told apart from a mask that really lost
  // its content
  dt_print(DT_DEBUG_MASKS,
           "[masks] panel rebuild '%s': mask_id=%d grp=%s points=%d flexi=%d",
           module->op, module->blend_params->mask_id, grp ? "ok" : "NULL",
           grp ? g_list_length(grp->points) : -1, flexi ? 1 : 0);
  // a flexi mask always shows a group: its own, or with no group form yet the
  // one it will have (see _module_flexi_group)
  if(!flexi) return FALSE;

  // one group is always selected: a selection whose group is gone (deleted,
  // merged, undone) falls back to the mask's own, as a deselect does. The
  // mask's own group then targets the whole mask's refinement (see
  // _model_refine_scope_from_selection)
  if(dt_is_valid_maskid(bd->panel_selected_group_cid)
     && !_group_point(grp, bd->panel_selected_group_cid))
    bd->panel_selected_group_cid = INVALID_MASKID;
  _select_mask_group_if_none(bd);

  // refresh the insert hint now (not just at the very end, its other call
  // site) so it reflects any selection change made just above, in time for
  // the pending-row placement below to target the right group.
  // Idempotent/side-effect-free to call twice in one pass.
  _recompute_insert_hint(module);
  return TRUE;
}

// a group header squares its bottom-left corner onto the rail below it while
// its elements show (.mask-group-open in darktable.css). Every route that
// opens or closes a group shows or hides its element box, so following that
// box covers them all
static void _sync_group_open(GtkWidget *elem_box, GParamSpec *pspec, gpointer hdr)
{
  GList *kids = gtk_container_get_children(GTK_CONTAINER(elem_box));
  if(gtk_widget_get_visible(elem_box) && kids)
    dt_gui_add_class(GTK_WIDGET(hdr), "mask-group-open");
  else
    dt_gui_remove_class(GTK_WIDGET(hdr), "mask-group-open");
  g_list_free(kids);
}

// one group of the panel: its header, and its element rows nested under it.
// `marker` holds the group's settings, and its members start at `first`
static void _pack_group(dt_iop_module_t *module,
                        dt_masks_form_t *grp,
                        const dt_masks_point_group_t *marker,
                        GList *first,
                        const gboolean is_base_group,
                        const int ngroups,
                        dt_masks_form_t *pending_form,
                        GtkWidget *container)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  const dt_masks_state_t op = _eff_group_op(marker->state);
  const guint cid = (guint)marker->formid;
  // the group's resolvable member ids, top-first (g_list_prepend)
  GList *formids = NULL;
  int run = 0;
  gboolean all_hidden = TRUE;
  for(GList *m = first; m && !_starts_group(m); m = g_list_next(m))
  {
    dt_masks_point_group_t *pm = m->data;
    if(!dt_masks_get_from_id(darktable.develop, pm->formid))
    {
      // a member referenced by the group but absent from dev->forms. The
      // renderer works off the pipe's own deep copy, so the mask keeps drawing
      // while the row silently vanishes here -- never silent now.
      dt_print(
        DT_DEBUG_MASKS,
        "[masks] panel: group member %d of '%s' not in dev->forms -- row dropped",
        pm->formid, module->op);
      continue;
    }
    formids = g_list_prepend(formids, GINT_TO_POINTER(pm->formid));
    run++;
    if(!(pm->state & DT_MASKS_STATE_HIDDEN)) all_hidden = FALSE;
  }
  const gboolean empty = run == 0;
  // an empty group can never be the solo target, so while any solo is active
  // it dims like a group whose every member is hidden
  if(empty) all_hidden = dt_is_valid_maskid(bd->solo_formid) || bd->solo_group_key != 0;
  const dt_masks_state_t group_within = marker->state & DT_MASKS_STATE_WITHIN;
  const int opstate = (int)op;
  // a bypassed group contributes nothing, so nothing inside it can have any
  // visible effect: everything below is built insensitive except the operator
  // handle, which is the way back (see the sensitivity block after `hdr`).
  const gboolean group_bypassed = _op_is_bypassed(opstate);
  // the mask's own group, when its list holds that one group: every other
  // group nests in it (masks_revamp_nested_groups.md, Q8). Its header is a
  // group header like any other, carrying the whole-mask actions, but it
  // cannot be deleted, moved or deselected
  const gboolean is_root = !grp || (grp == _module_mask_group(module) && ngroups == 1);
  // persistent "true" group invert (DT_MASKS_STATE_OP_INVERT, see
  // _group_toggle_output_invert) -- unlike group_bypassed this does not
  // affect what is built below (an inverted group still contributes to the
  // mask, just flipped), only the handle's look and its tooltip.
  const gboolean group_inverted = (opstate & DT_MASKS_STATE_OP_INVERT) != 0;

  // the group's operator is its lead handle's (see ghandle below): there is no
  // second operator to choose on the right
  GtkWidget *within_sel = NULL;

  // label: "<mode>-<id>" -- the group's within-group mode and its per-mode id
  // (shared with empty groups and the refinement caption). Once the group
  // is given a custom name (ctrl+click the title, masks v8) that replaces the
  // default label outright rather than being appended to it -- the "<op>-<id>"
  // form only exists as a placeholder until the user names the thing. No
  // disclosure triangle (groups don't expand).
  const int gord = _group_ordinal_of_cid(module, (dt_mask_id_t)cid);
  // the mask's own group is always "whole mask": it names what the header
  // stands for, so it cannot be renamed (see _start_group_rename)
  const char *custom_name = _group_custom_name(grp, (dt_mask_id_t)cid);
  gchar *txt = is_root       ? g_strdup(_("whole mask"))
               : custom_name ? g_strdup(custom_name)
                             : g_strdup_printf("%s-%d", _within_name(group_within), gord);
  GtkWidget *lbl = gtk_label_new(txt);
  g_free(txt);
  gtk_label_set_xalign(GTK_LABEL(lbl), 0.0f);
  // ellipsize, now that the title column has a fixed width (see labevt's
  // size request below) instead of taking whatever it needs -- same reason
  // an element row's own name label ellipsizes (see _make_shape_row). The
  // max-width-chars cap is what actually makes that fixed width stick
  // (see the matching comment on the element row's own name label) --
  // without it a long custom name still claims its full natural width
  // whenever the header has room, at the opacity slider's expense.
  gtk_label_set_ellipsize(GTK_LABEL(lbl), PANGO_ELLIPSIZE_MIDDLE);
  gtk_label_set_max_width_chars(GTK_LABEL(lbl), 1);
  // soloed: only this group is used -- shown by a badge, packed into a
  // fixed-size stack (see _make_badge_stack below) instead of a
  // button icon (see _make_solo_badge / _toggle_solo_group, triggered by
  // the header's own "solo" menu item, see _build_group_op_menu). Always present
  // (like an element row's own badge), just blank unless active, so
  // soloing an element elsewhere -- which only refreshes rows in place,
  // not headers, see _refresh_all_shape_rows/_apply_group_solo_badges --
  // can still clear a stale badge here without a full list rebuild.
  const gboolean group_solo = bd->solo_group_key == cid;
  const int badge_status = group_bypassed ? MASK_SOLO_BADGE_DISABLE
                           : group_solo   ? MASK_SOLO_BADGE_SOLO
                                          : MASK_SOLO_BADGE_NONE;
  GtkWidget *group_solo_badge = _make_solo_status_badge();
  _set_solo_status_badge(group_solo_badge, badge_status);
  g_object_set_data(G_OBJECT(group_solo_badge), "group-key", GUINT_TO_POINTER(cid));
  g_signal_connect(G_OBJECT(group_solo_badge), "button-press-event",
                   G_CALLBACK(_solo_badge_group_press), module);
  // low-opacity warning for the whole group, stacked with the solo badge and
  // driven the same way (blank by default, activated in place by
  // _refresh_lowop_badges, which also sets its initial state at the end of
  // this rebuild)
  GtkWidget *group_lowop_badge = _make_lowop_badge();
  // just the (possibly swapped-for-a-rename-entry) title now -- the badges
  // used to live in here too, but that let their visibility change this
  // box's own width, throwing off the fixed title column every other row
  // in the panel now shares (see labevt's size request below).
  GtkWidget *lbl_box = dt_gui_hbox();
  dt_gui_add_class(lbl_box, "mask-row-name");
  dt_gui_box_add(lbl_box, dt_gui_expand(lbl));
  // tagged so _group_header_press's ctrl+click can find (and later replace)
  // whichever of lbl / the rename entry currently occupies this slot
  g_object_set_data(G_OBJECT(lbl_box), "title-child", lbl);
  GtkWidget *labevt = gtk_event_box_new();
  // windowless: the label must not capture the button-press/motion stream, or the
  // header's group drag source (on hdr_evbox) never arms when the user grabs the
  // label text -- the natural place to grab a row to drag it.
  gtk_event_box_set_visible_window(GTK_EVENT_BOX(labevt), FALSE);
  gtk_container_add(GTK_CONTAINER(labevt), lbl_box);
  // expands to absorb whatever width the opacity/within-group slot below
  // doesn't need (see _control_column_size_allocate), same as an element
  // row's own name column -- the 50dpi request is just a floor so it never
  // gets squeezed to nothing on an unusually narrow/crowded row.
  gtk_widget_set_size_request(labevt, DT_PIXEL_APPLY_DPI(50), -1);
  gtk_widget_set_hexpand(labevt, TRUE);
  gtk_widget_set_tooltip_text(
    labevt, is_root ? _("the whole mask: every element and group is inside it\n"
                        "click to select it, which also refines the whole mask\n"
                        "right-click to open the mask's actions menu "
                        "(also reachable from the lead icon)")
            : empty ? _("empty group - select it, then draw a shape (or drop one here) to"
                        " fill it\n"
                        "ctrl+click to rename\n"
                        "drag the row to rearrange\n"
                        "right-click to open the group's actions menu")
                    : _("click to select this group\n"
                        "ctrl+click to rename\n"
                        "drag the row to rearrange, holding shift when dropping to"
                        " put it inside the group under the pointer\n"
                        "right-click to open the group's actions menu "
                        "(also reachable from the lead icon), which "
                        "includes \"solo\": use only this group"));

  // column 0 is the group's operator (ghandle below). With only one group in
  // its list there is nothing to reorder against, so dragging is disabled
  // (ngroups is computed once before the loop).
  const gboolean group_movable = ngroups >= 2;
  // a group nested in another is one element of its holder: it has no
  // between-group operator, its holder's within-group operator combines it
  // with its siblings (masks_revamp_nested_groups.md, Q7)
  const gboolean nested = grp && grp != _module_mask_group(module);
  // a nested group shown as its one group, in place of its element row (see
  // _nested_as_group): its header moves the nested group
  const dt_masks_point_group_t *nested_ref =
    nested && ngroups == 1 ? _group_point(_module_mask_group(module), grp->formid) : NULL;
  const gboolean shown_nested = nested_ref && _nested_as_group(nested_ref, grp);
  // the group's operator: how it folds its own elements, from the bottom one
  // up (masks_revamp_nested_groups.md, Q8)
  gchar *ghandle_tip = g_strdup_printf(
    is_root && group_bypassed ? _("mask operator: %s (disabled)\n"
                                  "the mask keeps its elements, but contributes nothing\n"
                                  "click to change the operator\n"
                                  "right-click for actions")
    : is_root ? _("mask operator: %s\n"
                  "how the mask combines its elements and groups, from the bottom"
                  " one up\n"
                  "click to change the operator\n"
                  "right-click for actions (solo, inverting and emptying)")
    : group_bypassed ? _("group operator: %s (disabled)\n"
                       "this group keeps its elements and its place, but contributes"
                       " nothing to the mask\n"
                       "click to change the operator\n"
                       "right-click for actions")
                   : _("group operator: %s\n"
                       "how this group combines its elements, from the bottom one up\n"
                       "click to change the operator\n"
                       "right-click for actions (solo, inverting, emptying and deleting)"),
    _within_name(group_within));
  GtkWidget *ghandle_btn = NULL;
  GtkWidget *ghandle =
    _make_op_combo(&ghandle_btn, _within_paint(group_within), G_CALLBACK(_group_within_press));
  dt_gui_remove_class(ghandle, "mask-op-combo");
  dt_gui_add_class(ghandle, "mask-within-combo");
  dt_gui_add_class(ghandle, "mask-group-lead-handle");
  gtk_widget_set_valign(ghandle, GTK_ALIGN_CENTER);

  if(group_inverted) dt_gui_add_class(ghandle_btn, "mask-list-handle-inverted");
  g_object_set_data(G_OBJECT(ghandle_btn), "module", module);
  if(is_base_group)
    g_object_set_data(G_OBJECT(ghandle_btn), "is-base-group", GINT_TO_POINTER(1));
  g_object_set_data(G_OBJECT(ghandle_btn), "title-label-box", lbl_box);
  g_object_set_data(G_OBJECT(ghandle_btn), "group-key", GUINT_TO_POINTER(cid));
  gtk_widget_set_tooltip_text(ghandle_btn, ghandle_tip);
  g_free(ghandle_tip);

  // opacity control: a persistent, multiplicative gain on this run's own
  // finished sub-mask (see dt_masks_point_group_t.group_opacity and
  // _group_get_mask_roi_flexi in group.c), applied on top of -- not instead
  // of -- each member's own independent opacity. Shown right next to the
  // group's name, unless "use sliders for opacity" has moved it into the
  // group's expanded contents as a full slider instead (see
  // _opacity_sliders). An absolute value bound
  // straight to the persisted field via _group_opacity_changed, unlike the
  // delta convention every multi-target properties row uses
  // (_props_row_apply): a group header always represents exactly one run,
  // so there is no multi-select ambiguity to resolve. The label/value are
  // hidden to keep the header compact -- the tooltip stands in for them
  // (see _group_opacity_update_tooltip), refreshed live on every drag tick.
  const gboolean show_group_opacity_slider = _opacity_sliders();
  const gboolean show_group_header_opacity = !show_group_opacity_slider;

  GtkWidget *group_opacity_slider = NULL;
  GtkWidget *group_val_widget = NULL;
  GtkWidget *group_opacity_inner = NULL;
  if(show_group_header_opacity)
  {
    group_opacity_slider = dt_bauhaus_slider_new_with_range(
      module, _blend_masks_properties[DT_MASKS_PROPERTY_OPACITY].min,
      _blend_masks_properties[DT_MASKS_PROPERTY_OPACITY].max, 0, 1.0f, 2);
    dt_bauhaus_widget_set_label(group_opacity_slider, N_("blend"), N_("opacity"));
    dt_bauhaus_slider_set_format(group_opacity_slider, "%");
    dt_bauhaus_slider_set_digits(group_opacity_slider, 2);
    dt_bauhaus_widget_set_quad_visibility(group_opacity_slider, FALSE);
    dt_bauhaus_widget_hide_label(group_opacity_slider);
    // the pill-background fix every other props slider needs (see
    // .mask-boost-factor-slider in darktable.css); no margins of its own
    // since this one sits inline in the header, not docked below a row.
    dt_gui_add_class(group_opacity_slider, "mask-props-slider");
    dt_gui_add_class(group_opacity_slider, "mask-inline-opacity");
    _style_opacity_gradient(group_opacity_slider);
    // a bauhaus slider's own natural height (line_height + baseline) is taller
    // than this row's other, icon-sized controls -- FILL (the GtkWidget
    // default) would stretch it to match the row instead of the other way
    // around, leaving it looking vertically off; centering it in whatever
    // height the row ends up with reads right instead.
    gtk_widget_set_valign(group_opacity_slider, GTK_ALIGN_CENTER);
    {
      const dt_masks_point_group_t *head_pt = _group_point(grp, (dt_mask_id_t)cid);
      const float go = head_pt ? head_pt->group_opacity : 1.0f;
      DT_ENTER_GUI_UPDATE(); // populate only -- must not fire _group_opacity_changed
      dt_bauhaus_slider_set(group_opacity_slider, go);
      DT_LEAVE_GUI_UPDATE();
      _group_opacity_update_tooltip(group_opacity_slider, go);
    }
    g_object_set_data(G_OBJECT(group_opacity_slider), "group-key", GUINT_TO_POINTER(cid));
    g_signal_connect(G_OBJECT(group_opacity_slider), "value-changed",
                     G_CALLBACK(_group_opacity_changed), module);
    g_signal_connect(G_OBJECT(group_opacity_slider), "button-press-event",
                     G_CALLBACK(_group_opacity_press), module);
  }

  GtkWidget *hdr = dt_gui_hbox();
  // unique per-kind id (#mask-group-header-row); .mask-panel-row is the
  // shared base styling class every row/header kind in the panel keeps
  gtk_widget_set_name(hdr, "mask-group-header-row");
  dt_gui_add_class(hdr, "mask-panel-row");
  // a subtle resting background distinct from plain element rows, so this
  // reads as a group heading even when nothing is selected (see
  // .mask-group-header in darktable.css)
  dt_gui_add_class(hdr, "mask-group-header");
  if(group_solo) dt_gui_add_class(hdr, "mask-list-row-solo");

  if(group_opacity_slider)
  {
    group_val_widget = _make_inline_opacity_value_widget(group_opacity_slider, module);
    gtk_widget_set_no_show_all(group_opacity_slider, TRUE);
    gtk_widget_hide(group_opacity_slider);

    group_opacity_inner = dt_gui_hbox();
    dt_gui_box_add(group_opacity_inner, group_opacity_slider);
    gtk_box_pack_end(GTK_BOX(group_opacity_inner), group_val_widget, FALSE, FALSE, 0);
    gtk_widget_set_halign(group_val_widget, GTK_ALIGN_END);
    gtk_widget_set_valign(group_opacity_inner, GTK_ALIGN_CENTER);
  }

  // a selection inside one of its nested groups is inside it too
  const gboolean has_selected = _members_hold(formids, bd->panel_selected_formid)
                                || _members_hold(formids, bd->panel_selected_group_cid);

  // "auto-expand selected", group half: exactly the anchor group is open --
  // the selected one, or, with nothing selected, whatever the option opened
  // last (see _auto_expand_selected_group, which enforces the same invariant
  // in place on every selection change, since selection never rebuilds the
  // list). With neither available there is nothing to anchor on, so this
  // falls back to the remembered per-group state below, which defaults a
  // group to open rather than leaving the panel showing bare headers.
  const dt_mask_id_t group_anchor = _model_auto_expand_group_anchor(bd);
  const gboolean group_auto_exp =
    _auto_expand_selected() && dt_is_valid_maskid(group_anchor);

  // with the anchor nested in it, this group has to be open to show it
  const gboolean group_expanded =
    group_auto_exp
      ? ((dt_mask_id_t)cid == group_anchor || _members_hold(formids, group_anchor))
      : (has_selected || !bd->masks_props_expanded
         || !g_hash_table_contains(bd->masks_props_expanded, GUINT_TO_POINTER(cid))
         || GPOINTER_TO_INT(
           g_hash_table_lookup(bd->masks_props_expanded, GUINT_TO_POINTER(cid))));

  if(group_auto_exp && group_expanded) bd->masks_last_expanded_group = (dt_mask_id_t)cid;

  if(group_expanded && bd->masks_props_expanded)
    g_hash_table_insert(bd->masks_props_expanded, GUINT_TO_POINTER(cid),
                        GINT_TO_POINTER(TRUE));

  // an empty group has nothing to show or hide
  GtkWidget *group_expand_toggle = NULL;
  if(!empty)
  {
    group_expand_toggle = dtgtk_togglebutton_new(_paint_param_inout, 0, NULL);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(group_expand_toggle), group_expanded);
    dt_gui_add_class(group_expand_toggle, "dt_transparent_background");
    dt_gui_add_class(group_expand_toggle, "mask-row-expander");
    gtk_widget_set_tooltip_text(group_expand_toggle,
                                _("show/hide this group's elements"));
    g_object_set_data(G_OBJECT(group_expand_toggle), "props-key", GUINT_TO_POINTER(cid));
    g_signal_connect(G_OBJECT(group_expand_toggle), "toggled",
                     G_CALLBACK(_group_expand_toggled), module);
  }

  _pack_row_header(hdr, ghandle, labevt, group_opacity_inner,
                   _make_badge_stack(group_lowop_badge, group_solo_badge), within_sel,
                   group_expand_toggle);
  // dimmed when the group contributes nothing: every element hidden, or the
  // whole group bypassed (in which case the badge stays at full opacity)
  if(group_bypassed)
  {
    gtk_widget_set_opacity(ghandle, 0.45);
    gtk_widget_set_opacity(labevt, 0.45);
    if(group_opacity_inner) gtk_widget_set_opacity(group_opacity_inner, 0.45);
    if(within_sel) gtk_widget_set_opacity(within_sel, 0.45);
  }
  else if(all_hidden)
  {
    gtk_widget_set_opacity(hdr, 0.45);
  }

  // a bypassed group has no effect on the mask, so none of its own controls
  // can either -- grey them out, exactly as a solo-suppressed element's
  // editors are (see _update_shape_row_state). The operator handle and the
  // header's own event box stay live: the handle is the only way back (it
  // opens the chooser, which collapses to "resume", see _build_group_op_menu)
  // and the header must still be selectable/draggable, the same carve-out
  // solo makes. Element rows inside the group are greyed by _make_shape_row,
  // which reads the bypass bit off each member's own state.
  if(group_bypassed)
  {
    if(group_opacity_slider) gtk_widget_set_sensitive(group_opacity_slider, FALSE);
    if(group_val_widget) gtk_widget_set_sensitive(group_val_widget, FALSE);
  }

  // an event box wraps the header so the canvas<->list hover sync can locate
  // this group by any member id (it carries "group-formids") and so clicking
  // it selects the group / right-clicking deletes it.
  GtkWidget *hdr_evbox = _make_group_header_evbox(
    module, hdr, lbl_box, G_CALLBACK(_group_header_press),
    G_CALLBACK(_group_header_release), G_CALLBACK(_masks_header_drag_received),
    group_movable || shown_nested ? _mask_group_dnd : NULL,
    group_movable || shown_nested ? G_CALLBACK(_masks_group_drag_get) : NULL);
  // the one group of a nested group has nothing to reorder against, but the
  // nested group itself moves: among its holder's elements when dropped on an
  // element row (see _drags_as_element), as a group elsewhere
  if(shown_nested) g_object_set_data(G_OBJECT(hdr_evbox), "drags-as-element", GINT_TO_POINTER(1));
  g_object_set_data_full(G_OBJECT(hdr_evbox), "group-formids", g_list_copy(formids),
                         (GDestroyNotify)g_list_free);
  g_object_set_data_full(G_OBJECT(hdr_evbox), "hover-formids", g_list_copy(formids),
                         (GDestroyNotify)g_list_free);
  gtk_widget_add_events(hdr_evbox, GDK_ENTER_NOTIFY_MASK | GDK_LEAVE_NOTIFY_MASK);
  g_signal_connect(G_OBJECT(hdr_evbox), "enter-notify-event", G_CALLBACK(_row_crossing),
                   module);
  g_signal_connect(G_OBJECT(hdr_evbox), "leave-notify-event", G_CALLBACK(_row_crossing),
                   module);

  // DnD (drop targets, and the drag source when group_movable) is wired by
  // _make_group_header_evbox above, shared with the staged-group header.
  // "group-formids", which _masks_group_drag_get reads at drag time, is set
  // just above.

  // selection / delete / reset: a plain primary press returns FALSE (so the group
  // drag source can arm), the group is selected on release, and right-click
  // deletes (shift+right-click resets) the group. The base group cannot be
  // deleted; tag it so the delete handlers refuse it.
  g_object_set_data(G_OBJECT(hdr_evbox), "group-key", GUINT_TO_POINTER(cid));
  g_object_set_data(G_OBJECT(hdr_evbox), "group-op", GINT_TO_POINTER(opstate));
  // "title-label-box" (ctrl+click rename) is tagged by
  // _make_group_header_evbox, shared with the staged-group header.
  // tagged so _apply_group_selection (a lightweight, no-rebuild selection update)
  // can find this header and toggle its highlight in place
  g_object_set_data(G_OBJECT(hdr_evbox), "mask-header", GINT_TO_POINTER(1));
  // tagged so _apply_group_solo_badges can find and toggle this header's own
  // solo badge in place too
  g_object_set_data(G_OBJECT(hdr_evbox), "solo-badge", group_solo_badge);
  // same, for the group's low-opacity warning (see _apply_group_lowop_badges)
  g_object_set_data(G_OBJECT(hdr_evbox), "lowop-badge", group_lowop_badge);
  if(is_base_group)
    g_object_set_data(G_OBJECT(hdr_evbox), "is-base-group", GINT_TO_POINTER(1));
  // press/release are connected by _make_group_header_evbox above

  // pack the header and this group's elements as one block: the header on top, the
  // element rows nested (indented) right below it. formids is top-first; the rows
  // render bottom-up (bottom member at the bottom). The whole block (not just the
  // header) is what "header-widget" points at below, so a selected group shades
  // its entire body, not just its header row.
  // An event box, not a plain GtkBox: a GtkBox is windowless and receives no
  // events at all, so clicking a group's body anywhere outside its header row
  // -- the padding, the indent to the left of the element rows, the gaps
  // between them -- used to do nothing. The block owns the group's whole
  // visual extent, so that is the area that should select it.
  //
  // The event box IS group_block (rather than a wrapper around it) on purpose:
  // everything below still refers to group_block for its CSS classes, its drop
  // target and drop-indicator classes, and its position among masks_list_box's
  // children -- which _canonical_drop_frame walks to find a group's neighbour
  // (see the one-insertion-slot work). Wrapping would have inserted a level
  // between the block and that sibling list and broken the drop indicator.
  // Children with their own windows (hdr_evbox, each row's own evbox) still
  // consume their clicks first; only what falls through reaches here.
  GtkWidget *group_block = gtk_event_box_new();
  gtk_event_box_set_visible_window(GTK_EVENT_BOX(group_block), TRUE);
  GtkWidget *block_inner = dt_gui_vbox();
  gtk_container_add(GTK_CONTAINER(group_block), block_inner);
  // id mirrors the class for direct CSS targeting alongside the existing
  // class-based rules (shared by every real group's own block instance)
  gtk_widget_set_name(group_block, "mask-group-block");
  dt_gui_add_class(group_block, "mask-group-block");
  dt_gui_box_add(block_inner, hdr_evbox);
  if(is_root)
  {
    dt_gui_add_class(group_block, "mask-root-block");
    g_object_set_data(G_OBJECT(group_block), "is-root", GINT_TO_POINTER(1));
  }
  g_object_set_data(G_OBJECT(hdr_evbox), "header-widget", group_block);
  // "header-widget" above targets the whole block (selection shades the
  // group's entire body); solo-suppression dimming (_apply_group_header_dimming)
  // must only dim the header row itself -- the member rows already dim
  // themselves individually via _update_shape_row_state, so dimming the
  // whole block here would double-dim them (compositing two 0.45 opacities).
  // "group-header-widget" (-> hdr) is tagged by _make_group_header_evbox.
  // so _apply_group_output_invert_icon can find and toggle this run's own
  // operator handle in place when "invert output" changes, the same way
  // "group-header-widget"/"header-widget" above let other in-place walkers
  // reach this header without a full rebuild.
  g_object_set_data(G_OBJECT(hdr_evbox), "ghandle-widget", ghandle_btn);
  // so _apply_group_header_dimming can grey these out in place too, not
  // just the header's opacity -- while another group/element is soloed,
  // this group contributes nothing, so its own controls should not be
  // editable either (previously only a bypassed group's controls were
  // disabled; a merely solo-suppressed group's stayed fully interactive
  // despite reading as disabled).
  g_object_set_data(G_OBJECT(hdr_evbox), "within-sel-widget", within_sel);
  g_object_set_data(G_OBJECT(hdr_evbox), "group-opacity-widget", group_opacity_slider);
  // read back by _apply_group_header_dimming's in-place refresh, so a solo
  // change elsewhere never re-enables a group that is independently
  // bypassed (bypass and solo-suppression are separate reasons a group's
  // controls stay grey, tracked and cleared independently)
  if(group_bypassed)
    g_object_set_data(G_OBJECT(hdr_evbox), "group-bypassed", GINT_TO_POINTER(1));

  // highlight the whole group block (not just the header) while a drag
  // (element or group) hovers it -- the group-reorder insertion line, in
  // particular, needs to span the group's full body so it reads as landing
  // above/below the group, not just above/below its header row
  g_signal_connect(G_OBJECT(hdr_evbox), "drag-motion", G_CALLBACK(_group_drop_motion),
                   group_block);
  g_signal_connect(G_OBJECT(hdr_evbox), "drag-leave", G_CALLBACK(_group_drop_leave),
                   group_block);
  g_object_set_data(G_OBJECT(hdr_evbox), "group-expand-toggle", group_expand_toggle);

  // the group's own block is ALSO a drop target in its own right, covering
  // every gap (margins/spacing between the header and its rows) that no
  // individual row or header widget occupies -- without this, moving the
  // pointer through those gaps flickered between "no drop" and "drop" as it
  // crossed from one child widget's bounds to the next (a child row/header's
  // own more specific drag-dest still wins whenever the pointer is directly
  // over it, since GTK always resolves to the topmost widget under the
  // pointer -- this only fills the cracks between them). A drop here is
  // routed exactly like a drop on the header itself.
  gtk_drag_dest_set(group_block, GTK_DEST_DEFAULT_MOTION | GTK_DEST_DEFAULT_DROP,
                    _mask_hdr_dnd, G_N_ELEMENTS(_mask_hdr_dnd), GDK_ACTION_MOVE);
  g_object_set_data_full(G_OBJECT(group_block), "group-formids", g_list_copy(formids),
                         (GDestroyNotify)g_list_free);
  g_object_set_data(G_OBJECT(group_block), "group-expand-toggle", group_expand_toggle);
  g_signal_connect(G_OBJECT(group_block), "drag-data-received",
                   G_CALLBACK(_masks_header_drag_received), module);
  g_signal_connect(G_OBJECT(group_block), "drag-motion", G_CALLBACK(_group_drop_motion),
                   group_block);
  g_signal_connect(G_OBJECT(group_block), "drag-leave", G_CALLBACK(_group_drop_leave),
                   group_block);

  // clicking the group's body selects/deselects it exactly as clicking its
  // header does -- the SAME two handlers, not a second implementation of
  // "what a click on a group means", so the two surfaces cannot drift apart
  // (ctrl+click rename and right-click actions come along for free, which is
  // the point). They read their context off the widget, so the block needs
  // the same keys the header carries; "group-formids" is already set above.
  g_object_set_data(G_OBJECT(group_block), "group-key", GUINT_TO_POINTER(cid));
  g_object_set_data(G_OBJECT(group_block), "group-op", GINT_TO_POINTER(opstate));
  g_object_set_data(G_OBJECT(group_block), "title-label-box", lbl_box);
  if(is_base_group)
    g_object_set_data(G_OBJECT(group_block), "is-base-group", GINT_TO_POINTER(1));
  g_signal_connect(G_OBJECT(group_block), "button-press-event",
                   G_CALLBACK(_group_block_press), module);
  g_signal_connect(G_OBJECT(group_block), "button-release-event",
                   G_CALLBACK(_group_block_release), module);

  // highlight the whole group block when its group is the selected one; the
  // mask's own, only its header row (see _paint_group_selection). Held by a
  // selected element, it is selected by implication instead (see
  // _apply_ancestor_selection)
  if(dt_is_valid_maskid(_explicit_group_cid(bd))
     && (dt_mask_id_t)cid == _explicit_group_cid(bd))
  {
    dt_gui_add_class(is_root ? hdr : group_block, "mask-list-row-selected");
    // the rail under it lights up with the header (see _paint_group_selection)
    if(is_root) dt_gui_add_class(group_block, "mask-root-selected");
  }

  GtkWidget *elem_box = dt_gui_vbox();
  // indent/inset entirely via CSS (.mask-group-elements's margin-left/
  // margin-right in darktable.css), not hardcoded here. id mirrors the
  // class for direct CSS targeting alongside the existing class-based rules
  // (shared by every real group's own elements-box instance)
  gtk_widget_set_name(elem_box, "mask-group-elements");
  dt_gui_add_class(elem_box, "masks-list");
  dt_gui_add_class(elem_box, "mask-group-elements");
  gtk_widget_set_visible(elem_box, empty || group_expanded);
  if(group_expand_toggle)
    g_object_set_data(G_OBJECT(group_expand_toggle), "elem-box", elem_box);

  // "use sliders for opacity": the group's opacity, as a full labeled slider
  // leading its expanded contents instead of the compact value its header
  // would otherwise carry. Packed before anything else, so it stays above
  // both the member rows (packed from the bottom, see _pack_group_elements)
  // and the pending-shape placeholder below. Drives the same persisted
  // group_opacity through _group_opacity_changed the header value does.
  if(show_group_opacity_slider)
  {
    GtkWidget *ex_op = dt_bauhaus_slider_new_with_range(
      module, _blend_masks_properties[DT_MASKS_PROPERTY_OPACITY].min,
      _blend_masks_properties[DT_MASKS_PROPERTY_OPACITY].max, 0, 1.0f, 2);
    dt_bauhaus_widget_set_label(ex_op, N_("blend"), N_("opacity"));
    dt_bauhaus_slider_set_format(ex_op, "%");
    dt_bauhaus_slider_set_digits(ex_op, 2);
    dt_bauhaus_widget_set_quad_visibility(ex_op, FALSE);
    dt_gui_add_class(ex_op, "mask-props-slider");
    _style_opacity_gradient(ex_op);
    {
      const dt_masks_point_group_t *head_pt = _group_point(grp, (dt_mask_id_t)cid);
      const float go = head_pt ? head_pt->group_opacity : 1.0f;
      DT_ENTER_GUI_UPDATE(); // populate only -- must not fire _group_opacity_changed
      dt_bauhaus_slider_set(ex_op, go);
      DT_LEAVE_GUI_UPDATE();
      _group_opacity_update_tooltip(ex_op, go);
    }
    g_object_set_data(G_OBJECT(ex_op), "group-key", GUINT_TO_POINTER(cid));
    g_signal_connect(G_OBJECT(ex_op), "value-changed",
                     G_CALLBACK(_group_opacity_changed), module);
    g_signal_connect(G_OBJECT(ex_op), "button-press-event",
                     G_CALLBACK(_group_opacity_press), module);
    if(group_bypassed) gtk_widget_set_sensitive(ex_op, FALSE);

    GtkWidget *ex_op_box = dt_gui_vbox(ex_op);
    dt_gui_add_class(ex_op_box, "mask-group-opacity-editor");
    dt_gui_box_add(elem_box, ex_op_box);
  }

  _pack_group_elements(module, grp, elem_box, g_list_reverse(g_list_copy(formids)),
                       formids, group_block);

  // if a shape is currently being drawn and this run is where it would land
  // (see _recompute_insert_hint), show its disposable placeholder row at the
  // top of this group's elements -- exactly where the real row lands once it
  // commits (a new element is inserted above the run's current top member).
  if(pending_form
     && (!grp
         || (bd->insert_active
             && _group_cid_of_form(grp, bd->insert_after_fid) == (dt_mask_id_t)cid)))
    dt_gui_box_add(elem_box, _make_pending_shape_row(module, pending_form));

  // an empty group shows its box only for what it holds anyway (the opacity
  // slider, a shape landing in it): bare, it would be a stub of rail
  if(empty)
  {
    GList *kids = gtk_container_get_children(GTK_CONTAINER(elem_box));
    gtk_widget_set_visible(elem_box, kids != NULL);
    g_list_free(kids);
  }

  dt_gui_box_add(block_inner, elem_box);
  // disconnected with hdr, which the block destroys together with elem_box
  g_signal_connect_object(elem_box, "notify::visible", G_CALLBACK(_sync_group_open), hdr, 0);
  _sync_group_open(elem_box, NULL, hdr);

  gtk_box_pack_end(GTK_BOX(container), group_block, FALSE, FALSE, 0);

  g_list_free(formids);
}

// the single shape currently being drawn on canvas for this module (if any)
// -- not a real grp->points member yet, rendered as a disposable placeholder
// row instead (see _make_pending_shape_row). NULL whenever nothing is being
// drawn, or it belongs to a different module.
static dt_masks_form_t *_pending_form(dt_iop_module_t *module)
{
  const dt_masks_form_gui_t *pending_fg = darktable.develop->form_gui;
  return (pending_fg && pending_fg->creation && pending_fg->creation_module == module)
           ? darktable.develop->form_visible
           : NULL;
}

// the groups of the nested group `sub`, packed into `box` under the row of the
// member that holds it, the way the top list packs into masks_list_box
static void _pack_subgroup(dt_iop_module_t *module, dt_masks_form_t *sub, GtkWidget *box)
{
  // a malformed tree can hold a group inside itself: stop where every
  // recursive walk of the mask stops. The panel is built on the GUI thread only
  static int depth = 0;
  if(depth >= DT_MASKS_NESTING_MAX) return;
  depth++;
  const int ngroups = _level_group_count(sub, INVALID_MASKID);
  dt_masks_form_t *pending_form = _pending_form(module);
  for(GList *l = sub->points; l; l = g_list_next(l))
    if(_starts_group(l))
      _pack_group(module, sub, l->data, l->next, l == sub->points, ngroups, pending_form, box);
  depth--;
}

// Widget-side: build the panel's row tree from the already-reconciled model.
// Every mutation happens in _masks_panel_reconcile above, so this only reads.
static void _masks_panel_pack(dt_iop_module_t *module, dt_masks_form_t *grp)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *pending_form = _pending_form(module);
  GtkWidget *list = GTK_WIDGET(bd->masks_list_box);

  DT_ENTER_GUI_UPDATE();

  if(!bd->masks_cluster_expanded)
    bd->masks_cluster_expanded = g_hash_table_new(g_direct_hash, g_direct_equal);

  // one block per group, each its header with its element rows nested under
  // it, packed from the end so the bottom group sits at the bottom. A nested
  // group's blocks sit under its member row the same way (see _pack_subgroup)
  const int ngroups = _level_group_count(grp, INVALID_MASKID);
  if(!grp)
  {
    // a mask with no group form yet shows the one group it will have, empty,
    // which its first edit creates (see _module_flexi_group)
    dt_masks_point_group_t none = { 0 };
    none.formid = INVALID_MASKID;
    none.state = DT_MASKS_STATE_GROUP_MARKER | DT_MASKS_STATE_UNION;
    none.group_opacity = 1.0f;
    _pack_group(module, NULL, &none, NULL, TRUE, ngroups, pending_form, list);
  }
  for(GList *l = grp ? grp->points : NULL; l; l = g_list_next(l))
    if(_starts_group(l))
      _pack_group(module, grp, l->data, l->next, l == grp->points, ngroups, pending_form,
                  list);

  // the box carries no_show_all (flexi-only), which makes gtk_widget_show_all on
  // the box itself a no-op; show each header explicitly, then reveal the box.
  GList *children = gtk_container_get_children(GTK_CONTAINER(bd->masks_list_box));
  for(GList *c = children; c; c = g_list_next(c))
    gtk_widget_show_all(GTK_WIDGET(c->data));
  g_list_free(children);
  gtk_widget_set_visible(GTK_WIDGET(bd->masks_list_box), TRUE);

  // a scope whose target is gone follows the surviving selection instead
  if(_model_refine_scope_prune(module)) _flexi_refine_follow_selection(bd);

  // keep the canvas mirror of the persistent selection in step with the rebuild
  if(darktable.develop && darktable.develop->form_gui)
    darktable.develop->form_gui->panel_selected_formid = bd->panel_selected_formid;

  DT_LEAVE_GUI_UPDATE();

  // the insertion hint must always reflect the current target after a rebuild
  _recompute_insert_hint(module);

  _update_add_target_sensitivity(module);
  _update_refine_sensitivity(module);
  _sync_solo_canvas_highlight(module);
  // badges are built hidden and revealed from the current opacities -- after the
  // show_all pass above (which cannot force them on, they carry no_show_all) and
  // after every row is registered in bd->masks_row_map
  _refresh_lowop_badges(module);
  // the rows paint their own selection as they are built, not what holds them
  _apply_ancestor_selection(GTK_WIDGET(bd->masks_list_box), bd->panel_selected_formid,
                            bd->panel_selected_group_cid);
  // the data the rows were rebuilt from changed, and so may the shape's editor
  _props_panel_sync(module, TRUE);
}

void _build_masks_list(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd || !bd->masks_list_box) return;
  if(bd->masks_rebuild_suppressed) return;

  if(bd->masks_rebuild_idle_id)
  {
    g_source_remove(bd->masks_rebuild_idle_id);
    bd->masks_rebuild_idle_id = 0;
  }
  bd->masks_rebuild_pending = FALSE;

  // reconcile-by-skip: if nothing the tree is built from has changed since the
  // last build, the rebuilt tree would be identical -- skip the whole teardown/
  // rebuild. Turns the many defensive/duplicate rebuild requests (see the ~30
  // call sites) into a cheap hash compare. DT_INVALID_HASH (fresh bd) never
  // matches, so the first build always runs.
  const dt_hash_t sig = _masks_list_signature(module);
  if(sig != DT_INVALID_HASH && sig == bd->masks_list_sig)
  {
    dt_print(DT_DEBUG_MASKS, "[masks] build skipped (signature unchanged, 0x%llx)",
             (unsigned long long)sig);
    return;
  }
  bd->masks_list_sig = sig;

  // rebuilding destroys the rows without delivering leave events, so clear any
  // pending hover feedback to avoid a highlight sticking on the canvas
  if(darktable.develop->form_gui)
  {
    g_list_free(darktable.develop->form_gui->panel_hover_formids);
    darktable.develop->form_gui->panel_hover_formids = NULL;
    darktable.develop->form_gui->canvas_hover_formid = INVALID_MASKID;
  }

  // A parametric row owns two color-picker buttons (see _build_param_row_editor),
  // and the wipe below destroys them. darktable.lib->proxy.colorpicker.picker_proxy
  // is a GLOBAL that keeps pointing at whichever picker was last activated,
  // including its ->colorpick widget -- so a rebuild while one of this panel's
  // pickers is active leaves that global holding a destroyed GtkWidget. The next
  // click on ANY picker then runs _color_picker_reset(prior_picker) on it
  // (DTGTK_IS_TOGGLEBUTTON reads the finalized GObject's class pointer) and
  // segfaults. This is unique to this panel: every other picker in darktable is
  // built once in gui_init and outlives everything, so nothing upstream ever had
  // to invalidate the global. Repro: activate a parametric row's picker, add a
  // second channel (rebuild), click the new row's picker.
  //
  // Must run BEFORE the wipe -- dt_iop_color_picker_reset unsets the picker's own
  // toggle widget, which has to still exist.
  dt_iop_color_picker_reset(module, FALSE);

  // the "add group" button (masks_new_op_box) now lives permanently in
  // masks_toolbar (see its field comment in blend.h), not in this list, so
  // the unconditional wipe below no longer needs to spare it first.
  dt_gui_container_remove_children(GTK_CONTAINER(bd->masks_list_box));

  // the pending-row sliders (if any) are children of masks_list_box and were
  // just destroyed by the wipe above -- forget the stale pointers so
  // dt_iop_gui_blend_sync_pending_ai_sliders can tell "no active session" from
  // "the row just hasn't been (re)built yet" apart. _make_pending_shape_row
  // repopulates these below if a pending row is actually built this pass.
  bd->pending_ai_smoothing_slider = NULL;
  bd->pending_ai_cleanup_slider = NULL;
  // the same for the creation controls a pending row built for the shape
  // properties subpanel: it still shows them until _props_panel_sync replaces them
  bd->pending_props_box = NULL;

  // reset the formid -> row index; it is repopulated as _make_shape_row builds
  // each row below. Cleared here (before any new rows) so it never holds a
  // pointer to a just-destroyed row.
  if(!bd->masks_row_map)
    bd->masks_row_map = g_hash_table_new_full(g_direct_hash, g_direct_equal, NULL,
                                              (GDestroyNotify)g_slist_free);
  else
    g_hash_table_remove_all(bd->masks_row_map);

  // module->blend_params is transiently reset to defaults and then walked back
  // up through the module's own history by dt_dev_pixelpipe_synch_all() (once
  // per pipe, main + preview) while holding dev->history_mutex the whole time
  // -- an unrelated masks edit on another module can trigger that walk on the
  // GUI thread via a nested pixelpipe_change while this rebuild is deferred via
  // g_idle_add, so an unguarded read here can catch mask_id/mask_mode mid-reset
  // and render the panel as if the mask were empty. Taking the same (recursive)
  // mutex for just this snapshot guarantees we only ever see a settled value.
  dt_pthread_mutex_lock(&darktable.develop->history_mutex);
  dt_masks_form_t *grp = _module_mask_group(module);
  const gboolean flexi = !(module->blend_params->mask_mode & DEVELOP_MASK_RASTER);
  dt_pthread_mutex_unlock(&darktable.develop->history_mutex);

  if(!_masks_panel_reconcile(module, grp, flexi))
  {
    gtk_widget_set_visible(GTK_WIDGET(bd->masks_list_box), FALSE);
    _recompute_insert_hint(module);
    _props_panel_sync(module, TRUE);
    return;
  }

  _masks_panel_pack(module, grp);
}

// expand/collapse a same-kind element cluster. Shared by the triangle button
// (still a direct press handler -- it is not itself a drag source) and the
// header background's release handler below (the header IS now a drag source,
// so its own press must return FALSE instead to let the drag arm; see
// _element_cluster_press).
static gboolean
_element_cluster_toggle(GtkWidget *w, GdkEventButton *e, dt_iop_module_t *module)
{
  if(e->button != GDK_BUTTON_PRIMARY) return FALSE;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  GtkRevealer *rev = g_object_get_data(G_OBJECT(w), "revealer");
  GtkWidget *arrow = g_object_get_data(G_OBJECT(w), "arrow");
  const guint key = GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(w), "cluster-key"));
  const gboolean now = !gtk_revealer_get_reveal_child(rev);
  gtk_revealer_set_reveal_child(rev, now);
  dtgtk_button_set_paint(DTGTK_BUTTON(arrow), dtgtk_cairo_paint_dropdown,
                         now ? 0 : CPF_DIRECTION_UP, NULL);
  gtk_widget_queue_draw(arrow);
  if(bd && bd->masks_cluster_expanded)
    g_hash_table_insert(bd->masks_cluster_expanded, GUINT_TO_POINTER(key),
                        GINT_TO_POINTER(now));
  return TRUE;
}

// the arrow toggles on its own press (see below); without this, that press's
// matching release is unhandled by the arrow and bubbles up to the header
// event box's own "button-release-event" (_element_cluster_toggle), toggling
// a second time and cancelling the first -- clicking the chevron would then
// visibly do nothing. Consuming the release here (primary button only, so a
// right-click still bubbles up for the header's own delete handling) stops
// that bubble.
static gboolean
_element_cluster_arrow_release(GtkWidget *w, GdkEventButton *e, gpointer user_data)
{
  return e->button == GDK_BUTTON_PRIMARY;
}

// right-click deletes every member of the cluster (like right-click on a
// group header deletes the whole group, see _group_header_press); a plain
// primary press must return FALSE so the drag source can arm (see the
// drag_source_set on hdr_evbox below) -- the toggle itself happens on
// release instead, same press/release split every other draggable
// row/header in this file uses (e.g. _row_click_press).
static gboolean
_element_cluster_press(GtkWidget *w, GdkEventButton *e, dt_iop_module_t *module)
{
  if(e->button == GDK_BUTTON_SECONDARY)
  {
    GList *members = g_object_get_data(G_OBJECT(w), "hover-formids");
    _delete_elements(module, members);
    return TRUE;
  }
  return FALSE;
}

// pack one group's element rows into `container`, nested under that group's header.
// `grp` is the group form whose list holds the group: the mask's own, or a
// nested group's. `fids` is the run's member ids bottom-up (consumed/freed here). Same-kind drawn
// shapes fold into expand/collapse clusters (no actions); parametric forms are never
// folded (each keeps its own inline editor). `group_formids`/`group_frame` let
// every element row also double as a group/empty-group reorder drop target (see
// _make_shape_row): otherwise only the thin header row would accept such a drop,
// and dragging a group over any of a target group's own elements -- very easy to
// do by accident, since the header row is thin -- would be silently rejected.
// A member that is a nested group, shown as a group of its own like a
// top-level one: it holds one group, and its reference carries nothing the
// group's header cannot show (migration moves a reference's opacity and
// inversion onto the group's marker where that is exact, masks.c
// _fold_nested_refs). Anything else keeps the element row that shows it
static gboolean _nested_as_group(const dt_masks_point_group_t *pt, const dt_masks_form_t *form)
{
  if(!(form->type & DT_MASKS_GROUP) || (form->type & (DT_MASKS_CLONE | DT_MASKS_OBJECT)))
    return FALSE;
  if(pt->opacity != 1.0f || pt->refinement.enabled != DT_MASKS_REFINE_OFF
     || (pt->state & (DT_MASKS_STATE_INVERSE | DT_MASKS_STATE_HIDDEN | DT_MASKS_STATE_DISABLE)))
    return FALSE;
  if(!form->points || !dt_masks_point_is_marker(form->points->data)) return FALSE;
  for(const GList *l = g_list_next(form->points); l; l = g_list_next(l))
    if(dt_masks_point_is_marker(l->data)) return FALSE;
  return TRUE;
}

static void _pack_group_elements(dt_iop_module_t *module,
                                 dt_masks_form_t *grp,
                                 GtkWidget *container,
                                 GList *fids,
                                 GList *group_formids,
                                 GtkWidget *group_frame)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!grp || !fids)
  {
    g_list_free(fids);
    return;
  }

  // build the element rows in bottom-up order, then fold adjacent same-kind runs
  // into expand/collapse clusters (a lone shape stays a plain row). Pack with
  // pack_end so the bottom member sits at the bottom.
  const int n = g_list_length(fids);
  GtkWidget **rows = g_malloc0_n(n, sizeof(GtkWidget *));
  guint *kinds = g_malloc0_n(n, sizeof(guint));
  dt_mask_id_t *fid_of = g_malloc0_n(n, sizeof(dt_mask_id_t));
  int nr = 0;
  for(GList *l = fids; l; l = g_list_next(l))
  {
    const dt_mask_id_t fid = GPOINTER_TO_INT(l->data);
    dt_masks_point_group_t *fpt = _group_point(grp, fid);
    dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, fid);
    if(!fpt || !form)
    {
      // header rendered but this element's row dropped -- "the group looks empty"
      dt_print(DT_DEBUG_MASKS,
               "[masks] panel: element %d of '%s' dropped (point=%s form=%s)", fid,
               module->op, fpt ? "ok" : "MISSING", form ? "ok" : "MISSING");
      continue;
    }
    if(_nested_as_group(fpt, form))
    {
      // packed as its own group, header and all, in this element's place
      rows[nr] = dt_gui_vbox();
      _pack_subgroup(module, form, rows[nr]);
    }
    else
      rows[nr] = _make_shape_row(module, fpt, form, group_formids, group_frame);
    kinds[nr] = _form_kind(form);
    fid_of[nr] = fid;
    nr++;
  }
  g_list_free(fids);

  // fold runs of >= 3 adjacent same-kind drawn elements into expand/collapse
  // clusters to cut clutter. Only adjacent ones: a group folds its members in
  // list order, so gathering scattered members would either misstate that order
  // or have to reorder the group. Dragging a member next to another kind thus
  // takes it out of its cluster. Parametric and raster forms are never
  // clustered (each has its own inline editor), nor are nested groups. pack_end
  // keeps the bottom member at the bottom.
  const int cluster_min = 3;
  for(int i = 0; i < nr;)
  {
    const guint kind = kinds[i];
    int count = 1;
    while(i + count < nr && kinds[i + count] == kind) count++;

    // kind 0 is a nested group (see _form_kind): each shows its own groups
    if(count < cluster_min || kind == DT_MASKS_PARAMETRIC || kind == DT_MASKS_RASTER
       || kind == 0)
    {
      gtk_box_pack_end(GTK_BOX(container), rows[i], FALSE, FALSE, 0);
      i++;
      continue;
    }

    // nested one level deeper than a plain (unclustered) element row, via CSS
    // (.mask-cluster-elements' own margin-left in darktable.css), so expanding
    // a cluster visually reads as revealing its members as its own children.
    GtkWidget *inner = dt_gui_vbox();
    // id mirrors the class for direct CSS targeting alongside the existing
    // class-based rules (shared by every cluster's own elements-box instance)
    gtk_widget_set_name(inner, "mask-cluster-elements");
    dt_gui_add_class(inner, "mask-cluster-elements");
    GList *member_fids = NULL;
    gboolean contains_selected = FALSE;
    for(int k = i; k < i + count; k++)
    {
      gtk_box_pack_end(GTK_BOX(inner), rows[k], FALSE, FALSE, 0);
      member_fids = g_list_prepend(member_fids, GINT_TO_POINTER(fid_of[k]));
      if(dt_is_valid_maskid(bd->panel_selected_formid)
         && fid_of[k] == bd->panel_selected_formid)
        contains_selected = TRUE;
    }

    // a same-kind cluster: a header that only expands/collapses (no actions). Keyed
    // by its first member fid so the expanded state survives a rebuild. Clusters
    // default to collapsed -- both the first time a kind reaches the clustering
    // threshold and on every rebuild until the user explicitly expands it -- so
    // only an explicit TRUE recorded in the hash table opens one.
    const guint cid = (guint)fid_of[i];
    const gboolean expanded =
      contains_selected
      || (g_hash_table_contains(bd->masks_cluster_expanded, GUINT_TO_POINTER(cid))
          && GPOINTER_TO_INT(
            g_hash_table_lookup(bd->masks_cluster_expanded, GUINT_TO_POINTER(cid))));
    if(expanded && bd->masks_cluster_expanded)
      g_hash_table_insert(bd->masks_cluster_expanded, GUINT_TO_POINTER(cid),
                          GINT_TO_POINTER(TRUE));

    gchar *txt = g_strdup_printf("%d %s", count, _kind_name(kind, TRUE));
    GtkWidget *lbl = gtk_label_new(txt);
    g_free(txt);
    gtk_label_set_xalign(GTK_LABEL(lbl), 0.0f);
    GtkWidget *arrow =
      dtgtk_button_new(dtgtk_cairo_paint_dropdown, expanded ? 0 : CPF_DIRECTION_UP, NULL);
    gtk_widget_set_valign(arrow, GTK_ALIGN_CENTER);
    // kind icon in the same column the member rows' drag handle occupies (see
    // _make_shape_row), so a collapsed cluster still shows what it is -- just
    // column-aligned; the actual drag source is hdr_evbox below (the whole
    // header row, like a group's), not this icon itself
    GtkWidget *kicon =
      _make_drag_handle(_kind_icon_paint(kind), TRUE, _kind_name(kind, FALSE));
    // label and disclosure triangle side-by-side
    GtkWidget *lblbox = dt_gui_hbox();
    gtk_box_set_spacing(GTK_BOX(lblbox), DT_PIXEL_APPLY_DPI(4));
    dt_gui_box_add(lblbox, lbl, arrow);
    GtkWidget *chdr = dt_gui_hbox();
    // unique per-kind id (#mask-cluster-header-row); .mask-panel-row is the
    // shared base styling class every row/header kind in the panel keeps
    gtk_widget_set_name(chdr, "mask-cluster-header-row");
    dt_gui_add_class(chdr, "mask-panel-row");
    dt_gui_box_add(chdr, kicon, dt_gui_expand(lblbox));
    GtkWidget *hdr_evbox = gtk_event_box_new();
    gtk_event_box_set_visible_window(GTK_EVENT_BOX(hdr_evbox), TRUE);
    gtk_container_add(GTK_CONTAINER(hdr_evbox), chdr);
    gtk_widget_set_tooltip_text(hdr_evbox, _("click to expand or collapse\n"
                                             "drag anywhere in the row to move every "
                                             "member together, like a single element\n"
                                             "right-click to delete every member"));

    GtkWidget *rev = gtk_revealer_new();
    gtk_container_add(GTK_CONTAINER(rev), inner);
    gtk_revealer_set_reveal_child(GTK_REVEALER(rev), expanded);

    // toggle from both the header background (event box) and the triangle itself:
    // the triangle is a button that consumes its own press, so without its own
    // handler clicking directly on it would do nothing (the fiddly part). The
    // header background is also this cluster's drag source (see below), so its
    // own press must return FALSE (arm the drag) and the toggle moves to
    // release instead -- a drag never delivers a release, so dragging the
    // cluster never also toggles it (same split _row_click_press/_release use).
    g_object_set_data(G_OBJECT(hdr_evbox), "revealer", rev);
    g_object_set_data(G_OBJECT(hdr_evbox), "arrow", arrow);
    g_object_set_data(G_OBJECT(hdr_evbox), "cluster-key", GUINT_TO_POINTER(cid));
    // mirrored onto the revealer itself so a member row can walk straight up
    // its own ancestor chain to find (and force-open) its enclosing cluster --
    // see _reveal_cluster_for_row -- without needing the sibling header widget.
    g_object_set_data(G_OBJECT(rev), "arrow", arrow);
    g_object_set_data(G_OBJECT(rev), "cluster-key", GUINT_TO_POINTER(cid));
    g_object_set_data_full(G_OBJECT(hdr_evbox), "hover-formids", member_fids,
                           (GDestroyNotify)g_list_free);
    gtk_widget_add_events(hdr_evbox, GDK_ENTER_NOTIFY_MASK | GDK_LEAVE_NOTIFY_MASK);
    g_signal_connect(G_OBJECT(hdr_evbox), "enter-notify-event", G_CALLBACK(_row_crossing),
                     module);
    g_signal_connect(G_OBJECT(hdr_evbox), "leave-notify-event", G_CALLBACK(_row_crossing),
                     module);
    g_signal_connect(G_OBJECT(hdr_evbox), "button-press-event",
                     G_CALLBACK(_element_cluster_press), module);
    g_signal_connect(G_OBJECT(hdr_evbox), "button-release-event",
                     G_CALLBACK(_element_cluster_toggle), module);
    // draggable as a block, moving every member together (see _masks_cluster_move):
    // "hover-formids" set just above already holds every member's formid, reused
    // as-is by _masks_cluster_drag_get.
    gtk_drag_source_set(hdr_evbox, GDK_BUTTON1_MASK, _mask_cluster_dnd, 1,
                        GDK_ACTION_MOVE);
    g_signal_connect(G_OBJECT(hdr_evbox), "drag-data-get",
                     G_CALLBACK(_masks_cluster_drag_get), NULL);
    g_object_set_data(G_OBJECT(arrow), "revealer", rev);
    g_object_set_data(G_OBJECT(arrow), "arrow", arrow);
    g_object_set_data(G_OBJECT(arrow), "cluster-key", GUINT_TO_POINTER(cid));
    g_signal_connect(G_OBJECT(arrow), "button-press-event",
                     G_CALLBACK(_element_cluster_toggle), module);
    g_signal_connect(G_OBJECT(arrow), "button-release-event",
                     G_CALLBACK(_element_cluster_arrow_release), NULL);

    GtkWidget *cbox = dt_gui_vbox();
    dt_gui_box_add(cbox, hdr_evbox, rev);
    gtk_box_pack_end(GTK_BOX(container), cbox, FALSE, FALSE, 0);

    // same "fill the cracks" fix as the group's own block above: the gaps
    // between this cluster's header and its (expanded) member rows have no
    // widget of their own, so without this the pointer flickered between "no
    // drop" and "drop" moving through them. A drop lands wherever a drop on
    // this cluster's ENCLOSING group would (group_formids/group_frame, not
    // this cluster's own member subset), same as every plain element row here
    // already does for hovering (see _make_shape_row).
    if(group_frame)
    {
      gtk_drag_dest_set(cbox, GTK_DEST_DEFAULT_MOTION | GTK_DEST_DEFAULT_DROP,
                        _mask_hdr_dnd, G_N_ELEMENTS(_mask_hdr_dnd), GDK_ACTION_MOVE);
      g_object_set_data_full(G_OBJECT(cbox), "group-formids", g_list_copy(group_formids),
                             (GDestroyNotify)g_list_free);
      g_object_set_data(G_OBJECT(cbox), "group-key",
                        GINT_TO_POINTER(_group_cid_of_form(grp, fid_of[i])));
      g_signal_connect(G_OBJECT(cbox), "drag-data-received",
                       G_CALLBACK(_masks_header_drag_received), module);
      g_signal_connect(G_OBJECT(cbox), "drag-motion", G_CALLBACK(_group_drop_motion),
                       group_frame);
      g_signal_connect(G_OBJECT(cbox), "drag-leave", G_CALLBACK(_group_drop_leave),
                       group_frame);
    }
    i += count;
  }

  g_free(rows);
  g_free(kinds);
  g_free(fid_of);

  const gboolean is_mask_enabled = (module->blend_params->mask_mode != DEVELOP_MASK_DISABLED);
  const gboolean has_drawn = _module_has_drawn_shapes(module);
  if(bd->masks_edit)
  {
    gtk_widget_set_sensitive(bd->masks_edit, is_mask_enabled && has_drawn);
    if(!has_drawn && gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(bd->masks_edit)))
    {
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_edit), FALSE);
      dt_masks_set_edit_mode(module, DT_MASKS_EDIT_OFF);
    }
  }
}

// create a SINGLE-CHANNEL parametric form and open it for inline editing.
// channel_idx indexes the module's blend-colorspace channel[] array; in_out picks
// input(0)/output(1). Seeded NEUTRAL (no channel bit active, whole-range params) so
// it has no effect until the slider is dragged. Each parametric form edits exactly
// one channel; several can be combined with the usual operators, like shapes.
static void
_add_parametric_channel(dt_iop_module_t *self, const int channel_idx, const int in_out)
{
  dt_iop_gui_blend_data_t *bd = self->blend_data;
  if(!bd->blendif_support)
  {
    dt_control_log(_("this module does not support parametric masks"));
    return;
  }

  // the add-parametric controls are flexi-only, so the module is normally already
  // in flexi; only if it is in neither flexi nor drawn mode do we switch it into
  // flexi so the group (and the parametric form inside it) is evaluated. Forcing
  // drawn mode here would hide the flexi-only panel and the new row.
  if(!(self->blend_params->mask_mode & (DEVELOP_MASK_MASK | DEVELOP_MASK_FLEXI)))
    _blendop_mask_enable(self);
  dt_iop_request_focus(self);

  dt_masks_form_t *form = dt_masks_create(DT_MASKS_PARAMETRIC);
  dt_masks_point_parametric_t *p = calloc(1, sizeof(dt_masks_point_parametric_t));
  const dt_develop_blend_params_t *dp = self->default_blendop_params;
  // seeded NEUTRAL (see comment above): no channel bit active AND no polarity
  // bit set, regardless of what the module's own default blend params happen
  // to carry in those bits -- a single-channel form has no UI to ever touch
  // polarity itself (see _update_param_row_display), so a nonzero inherited
  // bit here would silently desync the slider from the shape's own invert
  // state (and the handle icon) from the moment it is created.
  p->blendif = 0;
  memcpy(p->blendif_parameters, dp->blendif_parameters, sizeof(p->blendif_parameters));
  memcpy(p->blendif_boost_factors, dp->blendif_boost_factors,
         sizeof(p->blendif_boost_factors));
  p->colorspace = (uint32_t)self->blend_params->blend_cst;
  p->single = 1;
  p->channel = (uint32_t)channel_idx;
  p->in_out = (uint32_t)in_out;
  p->invert = 0;
  // new parametric channel masks start collapsed -- a compact, input-only
  // slider (p->in_out defaults to 0/input-only above); see
  // _update_param_row_visibility.
  form->points = g_list_append(form->points, p);

  dt_print(DT_DEBUG_MASKS,
           "[masks] add single-channel parametric form to '%s' (ch=%d io=%d)", self->op,
           channel_idx, in_out);

  // register + add to the module's group (group creation, numbering, default
  // operator from the mask-manager pref, history)
  dt_masks_gui_form_save_creation(darktable.develop, self, form, NULL);

  // build the list so the new form gets its own row -- its editor is always
  // visible/live, no separate "open for editing" step needed
  _build_masks_list(self);
}

// one-click "add parametric" channel button (flexi row). Adds a single-channel
// form for the button's channel, on the input sub-channel.
static void _param_channel_clicked(GtkButton *button, gpointer user_data)
{
  if(DT_IN_GUI_UPDATE()) return;
  dt_iop_module_t *self = user_data;
  const int ch = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(button), "param-channel"));
  _add_parametric_channel(self, ch, 0);
}

static void _param_channel_button_enter_cb(GtkEventControllerMotion *controller,
                                           double x, double y,
                                           dt_iop_module_t *module)
{
  DT_GUARD_GUI_UPDATE();

  GtkWidget *widget = dt_gui_get_widget(controller);
  dt_iop_gui_blend_data_t *bd = module->blend_data;

  // the button stands for one channel of the module's blend colorspace; the
  // row is rebuilt whenever that colorspace changes, so the index still fits
  dt_dev_pixelpipe_display_mask_t channel = DT_DEV_PIXELPIPE_DISPLAY_NONE;
  const int ch = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(widget), "param-channel"));
  const dt_iop_gui_blendif_channel_t *channels =
    dt_develop_blendif_channels_for_csp(bd->csp);
  if(channels && ch >= 0)
  {
    int nch = 0;
    while(channels[nch].label) nch++;
    if(ch < nch) channel = channels[ch].display_channel;
  }

  _preview_on_hover_enter(module, widget, channel);
}

static void _param_channel_button_leave_cb(GtkEventControllerMotion *controller,
                                           dt_iop_module_t *module)
{
  DT_GUARD_GUI_UPDATE();

  _preview_on_hover_leave(module, dt_gui_get_widget(controller));
}

// (re)build the flexi-only "add parametric" row: one flat, CSS-themeable button
// (styled like the add-shape buttons) per channel of the module's blend
// colorspace. Rebuilt only when the csp changes. The row's own visibility is
// toggled per mode by the mask-mode callbacks.
static void _rebuild_param_channel_buttons(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd->masks_param_channels_inner) return;
  if(bd->param_channels_csp == (int)bd->csp) return; // already built for this csp
  bd->param_channels_csp = (int)bd->csp;

  // dt_action_define_iop below keeps a per-instance referral to each button in
  // module->widget_list, with nothing to drop it again. Left in place, the old
  // buttons' referrals dangle, and the next dt_accel_connect_instance_iop (a
  // history step toggling the module, say) crashes on them. They are plain
  // buttons, so they sit ahead of the bauhaus tail at widget_list_bh
  GList *old = gtk_container_get_children(GTK_CONTAINER(bd->masks_param_channels_inner));
  for(GSList **link = &module->widget_list; *link && *link != module->widget_list_bh;)
  {
    dt_action_target_t *referral = (*link)->data;
    if(g_list_find(old, referral->target))
    {
      GSList *dead = *link;
      *link = dead->next;
      g_free(referral);
      g_slist_free_1(dead);
    }
    else
      link = &(*link)->next;
  }
  g_list_free(old);

  dt_gui_container_destroy_children(GTK_CONTAINER(bd->masks_param_channels_inner));

  const dt_iop_gui_blendif_channel_t *channels =
    dt_develop_blendif_channels_for_csp(bd->csp);
  if(!channels) return;

  int idx = 0;
  for(const dt_iop_gui_blendif_channel_t *ch = channels; ch->label; ch++, idx++)
  {
    GtkWidget *btn = gtk_button_new_with_label(_(ch->label));
    dt_gui_add_class(btn, "dt_transparent_background");
    dt_gui_add_class(btn, "mask-channel-add-btn");
    gtk_widget_set_tooltip_text(btn, _(ch->tooltip));
    _stash_base_tooltip(btn);
    g_object_set_data(G_OBJECT(btn), "param-channel", GINT_TO_POINTER(idx));
    g_signal_connect(G_OBJECT(btn), "clicked", G_CALLBACK(_param_channel_clicked),
                     module);
    dt_gui_connect_motion(btn, NULL, _param_channel_button_enter_cb,
                          _param_channel_button_leave_cb, module);
    gtk_widget_show(btn);
    dt_gui_box_add(bd->masks_param_channels_inner, btn);
    // makes each channel button individually shortcut-assignable, like the
    // add-shape buttons (dt_iop_togglebutton_new does this internally for
    // those; this is a plain gtk_button_new, rebuilt per csp, so it needs the
    // call explicitly every time it is (re)created)
    dt_action_define_iop(module, "blend`shapes", ch->label, btn, &dt_action_def_button);
  }
}

// add a raster mask element referencing the given upstream source module + mask
// id. Raster elements are first-class: several can coexist in a module's group,
// each referencing a different source, each composited with its own operator --
// exactly like shapes and parametric channels. The source->this-module
// dependency is wired at commit time by _reconcile_raster_form_users
// (imageop.c), which registers every raster FORM's source (and survives edit
// reload), so nothing here touches the single legacy blend_params raster sink.
static void _add_raster_mask(dt_iop_module_t *self,
                             dt_iop_module_t *src,
                             const dt_mask_id_t id)
{
  dt_iop_gui_blend_data_t *bd = self->blend_data;
  if(!bd->masks_support || !src) return;

  // as with the parametric add controls, make sure the module is in a mode where
  // the group (and the raster element inside it) is evaluated
  if(!(self->blend_params->mask_mode & (DEVELOP_MASK_MASK | DEVELOP_MASK_FLEXI)))
    _blendop_mask_enable(self);
  dt_iop_request_focus(self);

  // if the source is not already storing a raster mask for anyone, it must be
  // reprocessed so it starts storing one (its cache is otherwise valid and would
  // not recompute); the commit-time reconciliation registers us as a user first.
  const gboolean reprocess = !dt_iop_is_raster_mask_used(src, id);

  dt_masks_form_t *form = dt_masks_create(DT_MASKS_RASTER);
  dt_masks_point_raster_t *p = calloc(1, sizeof(dt_masks_point_raster_t));
  g_strlcpy(p->source, src->op, sizeof(p->source));
  p->instance = src->multi_priority;
  p->id = id;
  form->points = g_list_append(form->points, p);

  dt_print(DT_DEBUG_MASKS, "[masks] add raster form to '%s' from '%s' id=%d", self->op,
           src->op, id);

  // registers the form + adds it to the module's group (records masks history,
  // which reprocesses -> commits -> reconciles the raster source registration)
  dt_masks_gui_form_save_creation(darktable.develop, self, form, NULL);

  // named by its type alone, the element shows its source's current name (see
  // _form_display_name). Set AFTER save_creation, whose de-dup numbering names
  // it "raster mask #N"
  g_strlcpy(form->name, _("raster mask"), sizeof(form->name));
  dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);

  _build_masks_list(self);
  // full reprocess so the (possibly newly-used) source recomputes and stores its
  // mask, and so this module's commit re-runs the source reconciliation
  if(reprocess) dt_dev_reprocess_all(self->dev);
}

// ---- shortcut actions on "whatever is currently selected in the panel" -----
// These have no fixed on-screen widget (unlike the add-shape and
// add-parametric buttons, which are made shortcut-assignable directly via
// dt_action_define_iop above and in _rebuild_param_channel_buttons): they act
// on the module's current panel selection (bd->panel_selected_formid /
// panel_selected_group_cid), which changes as the user clicks around. Each one
// is a thin wrapper around the same helper the matching click handler already
// uses (see _toggle_element_hidden, _toggle_ids_hidden, _invert_element,
// _invert_group_members, _toggle_soloedit, _build_group_op_menu,
// _build_within_menu, _stage_new_group, _add_parametric_channel), so a
// keyboard shortcut and the matching mouse click always do exactly the same
// thing.
//
// dt_action_register's callback gets no per-instance context (see
// dt_action_t / DT_ACTION_TYPE_COMMAND in accelerators.c), so -- like the
// action-resolution the accelerator core itself falls back to -- every one of
// these resolves "the module to act on" via dt_dev_gui_module(): the one
// instance whose panel is currently expanded/focused. That is already the
// only instance whose mask panel selection is meaningful.

static void _shortcut_add_group_above_selected(dt_action_t *action)
{
  dt_iop_module_t *module = dt_dev_gui_module();
  if(!module || !module->blend_data) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd->masks_support || !bd->masks_inited) return;
  _stage_new_group(module, bd->masks_new_group_op);
}

static void _shortcut_invert_selected_group(dt_action_t *action)
{
  dt_iop_module_t *module = dt_dev_gui_module();
  if(!module || !module->blend_data) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp || !dt_is_valid_maskid(bd->panel_selected_group_cid)) return;
  _invert_group_members(module, bd->panel_selected_group_cid);
}

static void _shortcut_invert_selected_element(dt_action_t *action)
{
  dt_iop_module_t *module = dt_dev_gui_module();
  if(!module || !module->blend_data) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!dt_is_valid_maskid(bd->panel_selected_formid)) return;
  _invert_element(module, bd->panel_selected_formid);
}

static void _shortcut_toggle_soloedit(dt_action_t *action)
{
  dt_iop_module_t *module = dt_dev_gui_module();
  if(!module || !module->blend_data) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd->soloedit_mode) return;
  // drive the header toggle rather than bd->soloedit_formid directly: solo-edit
  // follows the selection now, so isolating one shape by hand would only last
  // until the next selection change put it back
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->soloedit_mode),
                               !_soloedit_mode_is_on());
}

static void _shortcut_change_group_mode(dt_action_t *action)
{
  dt_iop_module_t *module = dt_dev_gui_module();
  if(!module || !module->blend_data) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp || !dt_is_valid_maskid(bd->panel_selected_group_cid)) return;
  GtkWidget *anchor = bd->masks_list_box ? GTK_WIDGET(bd->masks_list_box) : module->widget;
  _build_within_menu(anchor, module, bd->panel_selected_group_cid);
}

// toggle "bypass" on the selected group: the keyboard counterpart of the bypass
// entry in the operator chooser (see _build_group_op_menu). Worth its own
// shortcut because it is the one operator meant to be flipped back and forth
// while judging an edit.
static void _shortcut_toggle_group_bypass(dt_action_t *action)
{
  dt_iop_module_t *module = dt_dev_gui_module();
  if(!module || !module->blend_data) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp || !dt_is_valid_maskid(bd->panel_selected_group_cid)) return;
  _group_op_apply(module, bd->panel_selected_group_cid, DT_MASKS_STATE_OP_BYPASS);
}

static void _shortcut_change_group_within_mode(dt_action_t *action)
{
  dt_iop_module_t *module = dt_dev_gui_module();
  if(!module || !module->blend_data) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp || !dt_is_valid_maskid(bd->panel_selected_group_cid)) return;
  GtkWidget *anchor = bd->masks_list_box ? GTK_WIDGET(bd->masks_list_box) : module->widget;
  _build_within_menu(anchor, module, bd->panel_selected_group_cid);
}

// the panel options (see _add_masks_panel_options_menu) are check items on a
// menu that is rebuilt from conf on every popup, so there is no persistent
// widget for dt_action_define_iop to bind to. These give each option a
// bindable shortcut anyway, flipping the same conf key the menu item does.
static void _shortcut_toggle_preview_on_hover(dt_action_t *action)
{
  _preview_on_hover_set(!_preview_on_hover_is_on());
}

static void _shortcut_toggle_sticky_opacity(dt_action_t *action)
{
  dt_conf_set_bool("plugins/darkroom/masks/opacity_not_sticky",
                   !dt_conf_get_bool("plugins/darkroom/masks/opacity_not_sticky"));
}

static void _shortcut_toggle_auto_expand_selected(dt_action_t *action)
{
  const gboolean on = !_auto_expand_selected();
  dt_conf_set_bool("plugins/darkroom/masks/auto_expand_selected", on);
  // read at row-build time, not hashed by _masks_list_signature -- same
  // invalidation (and same parametric-row kick) the menu item's own callback
  // does, see _masks_auto_expand_selected_toggled
  dt_iop_module_t *module = dt_dev_gui_module();
  if(module && module->blend_data)
  {
    dt_iop_gui_blend_data_t *bd = module->blend_data;
    if(on) _auto_expand_selected_row(module, bd->panel_selected_formid);
    _masks_rebuild_for_option(module);
  }
}

static void _shortcut_toggle_collapse_refinements(dt_action_t *action)
{
  dt_conf_set_bool(
    "plugins/darkroom/masks/collapse_refinements_default",
    !dt_conf_get_bool("plugins/darkroom/masks/collapse_refinements_default"));
}

// this one used to be a widget action bound to bd->flexi_inline_collapse_btn,
// but that button is deliberately hidden in the utility-lib position (the lib's
// own expander header collapses the panel there, see masks_gui_panel_host.c),
// and _process_action refuses to run a widget action whose target is invisible
// (dt_action_widget_invisible in gui/accelerators.c) -- so the shortcut was
// dead in exactly one of the four positions. A command action carries no
// widget to be gated on, and the click handler it calls already dispatches on
// masks_panel_position, utility included.
static void _shortcut_toggle_masks_panel(dt_action_t *action)
{
  dt_iop_gui_blend_masks_panel_toggle();
}

// register every panel-selection shortcut above under "<blending> / masks", the
// same shared tree the panel's own widget actions land in (dt_action_define_iop
// routes a "blend`masks" section to darktable.control->actions_blend). Not under
// module->so: none of these acts on a particular operation -- each resolves its
// module through dt_dev_gui_module() -- so an owner per operation only listed
// the same twelve entries again under every module that supports masking, mixed
// in among that module's own parameters. Called once per instance init;
// repeated registration of a path that already exists is expected and harmless
// (dt_action_register only fills in a node still typed as a section).
static void _register_masks_action_shortcuts(void)
{
  dt_action_t *masks = dt_action_section(&darktable.control->actions_blend, N_("masks"));

  dt_action_register(masks, N_("show/hide mask panel"),
                     _shortcut_toggle_masks_panel, 0, 0);
  dt_action_register(masks, N_("add group above selected group"),
                     _shortcut_add_group_above_selected, 0, 0);
  dt_action_register(masks, N_("invert selected group visibility"),
                     _shortcut_invert_selected_group, 0, 0);
  dt_action_register(masks, N_("invert selected element visibility"),
                     _shortcut_invert_selected_element, 0, 0);
  dt_action_register(masks, N_("toggle solo-edit for current shape"),
                     _shortcut_toggle_soloedit, 0, 0);
  dt_action_register(masks, N_("change mode for current group"),
                     _shortcut_change_group_mode, 0, 0);
  dt_action_register(masks, N_("bypass/resume current group"),
                     _shortcut_toggle_group_bypass, 0, 0);
  dt_action_register(masks, N_("change within-group mode for current group"),
                     _shortcut_change_group_within_mode, 0, 0);
  dt_action_register(masks, N_("preview channel under cursor"),
                     _shortcut_toggle_preview_on_hover, 0, 0);
  dt_action_register(masks, N_("sticky opacity"),
                     _shortcut_toggle_sticky_opacity, 0, 0);
  dt_action_register(masks, N_("auto-expand selected"),
                     _shortcut_toggle_auto_expand_selected, 0, 0);
  dt_action_register(masks, N_("collapse refinements by default"),
                     _shortcut_toggle_collapse_refinements, 0, 0);
}

void dt_iop_gui_init_masks(GtkWidget *blendw, dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;

  /* create and add masks support if module supports it */
  if(bd->masks_support)
  {
    bd->masks_shown = DT_MASKS_EDIT_OFF;

    // flexi-only: opens the import menu (see _masks_import_btn_press)
    bd->masks_import_btn = dtgtk_button_new(dtgtk_cairo_paint_import, 0, NULL);
    gtk_widget_set_tooltip_text(bd->masks_import_btn,
                                _("link or copy shapes from other modules, copy their parametric\n"
                                  "channels, or add or use another module's whole mask\n"
                                  "(click to pick)"));
    g_signal_connect(G_OBJECT(bd->masks_import_btn), "button-press-event",
                     G_CALLBACK(_masks_import_btn_press), module);

    // default operator for a newly added group
    bd->masks_new_group_op = DT_MASKS_STATE_UNION;

    // ---- masks_toolbar: the fixed two-row flexi toolbar for every "add an
    // element" action (see its field comment in blend.h for the rationale
    // and exact row contents). Built first (empty) so the add-group button
    // below has somewhere to go; the rest of its permanent (flexi-only)
    // children are appended further down, as each is built.
    GtkWidget *toolbar = dt_gui_vbox();
    gtk_widget_set_no_show_all(toolbar, TRUE);
    dt_gui_add_class(toolbar, "masks-toolbar");
    bd->masks_toolbar = toolbar;
    GtkWidget *toolbar_row1 = dt_gui_hbox();
    dt_gui_add_class(toolbar_row1, "masks-btn-row");
    gtk_widget_show(toolbar_row1);
    bd->masks_toolbar_row1 = toolbar_row1;
    dt_gui_box_add(toolbar, toolbar_row1);
    GtkWidget *toolbar_row2 = dt_gui_hbox();
    dt_gui_add_class(toolbar_row2, "masks-btn-row");
    gtk_widget_show(toolbar_row2);
    bd->masks_toolbar_row2 = toolbar_row2;
    dt_gui_box_add(toolbar, toolbar_row2);

    // "add group": a plain "+" that opens the operator chooser (its icon is a
    // fixed add affordance, it never reflects the selection). Row 1, right
    // after the shape buttons: it adds to the mask as they do
    bd->masks_new_op_box = _make_op_combo(&bd->masks_new_op, dtgtk_cairo_paint_plus,
                                          G_CALLBACK(_new_shape_op_press));
    // the add-group button is a plain "+" icon, not a bordered chooser: drop the
    // "mask-op-combo" border so there is no white outline around it
    dt_gui_remove_class(bd->masks_new_op_box, "mask-op-combo");
    g_object_set_data(G_OBJECT(bd->masks_new_op), "module", module);
    _new_shape_op_update(bd->masks_new_op);
    gtk_widget_set_tooltip_text(bd->masks_new_op_box,
                                _("add a new group inside the selected group\n"
                                  "(or at the top of the mask, if none is selected)\n"
                                  "click to pick its operator\n"
                                  "right-click for group layout presets, which"
                                  " build a whole set of groups at once"));
    gtk_widget_show(bd->masks_new_op_box);
    bd->masks_new_op_label = NULL; // retired (the button is icon-only now)

    // row 1's slack sits in front of everything, so its buttons keep together
    // at the right edge and only the leading gap grows with the panel
    _pack_stretch(toolbar_row1);

    // the runs that add to the mask, each a fixed gap apart: add a group,
    // add a shape, then import one from another module
    dt_gui_box_add(toolbar_row1, bd->masks_new_op_box);
    _pack_gap(toolbar_row1);

    // reserves row 1's position for shapes_box, which does not exist as a
    // toolbar child yet: it is built below and slotted in between these two
    // gaps (see the reorder there)
    _pack_gap(toolbar_row1);
    gtk_widget_show(bd->masks_import_btn);
    dt_gui_box_add(toolbar_row1, bd->masks_import_btn);

    // solo edit sits on the panel header, next to "edit on canvas": it is used
    // interactively, and its state has to be visible while editing. The channel
    // preview is a set-once mode, so it lives in the panel options menu instead
    // (see _add_masks_panel_options_menu).
    bd->soloedit_mode = dt_iop_togglebutton_new(
      module, "blend`tools", N_("solo edit the selected element"), NULL,
      G_CALLBACK(_soloedit_mode_toggled), FALSE, 0, 0,
      dtgtk_cairo_paint_soloedit, NULL);
    gtk_widget_set_tooltip_text
      (bd->soloedit_mode,
       _("solo edit the selected element\n"
         "while enabled, only the selected element's nodes and handles are\n"
         "editable on canvas; the other elements still contribute to the mask"));
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->soloedit_mode),
                                 _soloedit_mode_is_on());
    gtk_widget_set_no_show_all(bd->soloedit_mode, TRUE);

    // NB: each group's elements (shapes) are nested directly under that group's
    // header inside masks_list_box (built by _build_masks_list /
    // _pack_group_elements); there is no separate "elements" section.

    // ---- shapes box: the shape-add buttons, wrapped as one group so the
    // toolbar row can space them as a unit (packed into toolbar_row1 below)
    GtkWidget *shapes_box = dt_gui_hbox();
    bd->masks_shapes_box = shapes_box;

    // "edit on canvas": toggles the on-canvas editing overlay (the shape
    // controls). On the panel header, with solo edit (see below)
    bd->masks_edit = dt_iop_togglebutton_new(
      module, "blend`tools", N_("edit on canvas"),
      N_("edit on canvas in restricted mode (no moving or resizing of shapes)"),
      G_CALLBACK(_blendop_masks_show_and_edit), FALSE, 0, 0, dtgtk_cairo_paint_masks_eye,
      NULL);
    gtk_widget_set_tooltip_text(
      bd->masks_edit,
      _("edit drawn mask elements on canvas\n"
        "ctrl+click for restricted mode (no moving or resizing of shapes)"));

    bd->masks_type[0] = DT_MASKS_PATH;
    bd->masks_shapes[0] = dt_iop_togglebutton_new(
      module, "blend`shapes", N_("add path"), N_("add multiple paths"),
      G_CALLBACK(_blendop_masks_add_shape), FALSE, 0, 0, dtgtk_cairo_paint_masks_path,
      NULL);
    gtk_widget_show(bd->masks_shapes[0]);
    dt_gui_box_add(shapes_box, bd->masks_shapes[0]);
    _stash_base_tooltip(bd->masks_shapes[0]);

    bd->masks_type[1] = DT_MASKS_BRUSH;
    bd->masks_shapes[1] = dt_iop_togglebutton_new(
      module, "blend`shapes", N_("add brush"), N_("add multiple brush strokes"),
      G_CALLBACK(_blendop_masks_add_shape), FALSE, 0, 0, dtgtk_cairo_paint_masks_brush,
      NULL);
    gtk_widget_show(bd->masks_shapes[1]);
    dt_gui_box_add(shapes_box, bd->masks_shapes[1]);
    _stash_base_tooltip(bd->masks_shapes[1]);

    bd->masks_type[2] = DT_MASKS_CIRCLE;
    bd->masks_shapes[2] = dt_iop_togglebutton_new(
      module, "blend`shapes", N_("add circle"), N_("add multiple circles"),
      G_CALLBACK(_blendop_masks_add_shape), FALSE, 0, 0, dtgtk_cairo_paint_masks_circle,
      NULL);
    gtk_widget_show(bd->masks_shapes[2]);
    dt_gui_box_add(shapes_box, bd->masks_shapes[2]);
    _stash_base_tooltip(bd->masks_shapes[2]);

    bd->masks_type[3] = DT_MASKS_ELLIPSE;
    bd->masks_shapes[3] = dt_iop_togglebutton_new(
      module, "blend`shapes", N_("add ellipse"), N_("add multiple ellipses"),
      G_CALLBACK(_blendop_masks_add_shape), FALSE, 0, 0, dtgtk_cairo_paint_masks_ellipse,
      NULL);
    gtk_widget_show(bd->masks_shapes[3]);
    dt_gui_box_add(shapes_box, bd->masks_shapes[3]);
    _stash_base_tooltip(bd->masks_shapes[3]);

    bd->masks_type[4] = DT_MASKS_GRADIENT;
    bd->masks_shapes[4] = dt_iop_togglebutton_new(
      module, "blend`shapes", N_("add gradient"), N_("add multiple gradients"),
      G_CALLBACK(_blendop_masks_add_shape), FALSE, 0, 0, dtgtk_cairo_paint_masks_gradient,
      NULL);
    gtk_widget_show(bd->masks_shapes[4]);
    dt_gui_box_add(shapes_box, bd->masks_shapes[4]);
    _stash_base_tooltip(bd->masks_shapes[4]);

#ifdef HAVE_AI
    bd->masks_type[5] = DT_MASKS_OBJECT;
    bd->masks_shapes[5] =
      dt_iop_togglebutton_new(module, "blend`shapes", N_("add AI object"), NULL,
                              G_CALLBACK(_blendop_masks_add_shape), FALSE, 0, 0,
                              dtgtk_cairo_paint_masks_object, NULL);
    gtk_widget_show(bd->masks_shapes[5]);
    dt_gui_box_add(shapes_box, bd->masks_shapes[5]);
    _stash_base_tooltip(bd->masks_shapes[5]);
#endif

    // parametric (blendif) forms are added via the channel row below (flexi-only),
    // one flat button per channel of the module's blend colorspace.
    bd->panel_selected_formid = INVALID_MASKID;
    bd->panel_selected_group_cid = INVALID_MASKID;
    bd->insert_active = FALSE;
    bd->solo_formid = INVALID_MASKID;

    // ---- "add parametric" cluster (flexi-only, toolbar row 2, leftmost):
    // one flat button per channel of the module's blend colorspace,
    // populated lazily by _rebuild_param_channel_buttons once the csp is
    // known. Visibility is toggled per mode alongside the rest of the
    // flexi-only widgets.
    bd->masks_param_channels_box = dt_gui_hbox();
    bd->param_channels_csp = DEVELOP_BLEND_CS_NONE;
    gtk_widget_set_no_show_all(bd->masks_param_channels_box, TRUE);

    // the channel buttons live in an inner box (rebuilt per csp); it carries
    // no no_show_all of its own, so it stays realized -- the cluster's
    // visibility is driven by the outer box.
    bd->masks_param_channels_inner = dt_gui_hbox();
    dt_gui_box_add(bd->masks_param_channels_box, bd->masks_param_channels_inner);
    gtk_widget_show(bd->masks_param_channels_inner);
    // right-aligned on its own row
    _pack_stretch(toolbar_row2);
    dt_gui_box_add(toolbar_row2, bd->masks_param_channels_box);

    // the shape buttons take the slot reserved for them in row 1:
    // stretch(0) add-group(1) gap(2) [shapes_box] gap(4) import(5)
    gtk_widget_show(shapes_box);
    dt_gui_box_add(toolbar_row1, shapes_box);
    gtk_box_reorder_child(GTK_BOX(toolbar_row1), shapes_box, 3);

    // edit on canvas and solo edit, onto the panel header built before this
    _pack_header_edit_run(bd);

    // per-shape composition list (the groups), populated by _build_masks_list()
    // whenever the module is in flexi-mask mode.
    bd->masks_list_box = GTK_BOX(dt_gui_vbox());
    gtk_widget_set_no_show_all(GTK_WIDGET(bd->masks_list_box), TRUE);
    // unique id for the panel's own top-level list container, alongside the
    // existing "masks-list" class every nested list box in the panel shares
    gtk_widget_set_name(GTK_WIDGET(bd->masks_list_box), "masks-list-box");
    dt_gui_add_class(GTK_WIDGET(bd->masks_list_box), "masks-list");

    // layout: toolbar -> element list. The list opens on the mask's own
    // group, whose header carries the whole-mask actions
    bd->masks_box = GTK_BOX(dt_gui_vbox(toolbar, GTK_WIDGET(bd->masks_list_box)));
    _add_wrapped_box(blendw, bd->masks_box, "masks_drawn");

    bd->masks_inited = TRUE;
    _register_masks_action_shortcuts();
  }
}

void dt_iop_gui_cleanup_blending(dt_iop_module_t *module)
{
  if(!module->blend_data) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;

  // last resort only. The real release happens in dt_iop_gui_cleanup_module,
  // *before* it destroys the module's widget tree, because by the time we get
  // here the widgets a release would move may already be freed -- and the
  // header/body of a hosted panel are children of the host, so the destroy
  // does not take them with it (see dt_iop_gui_blend_masks_panel_release).
  //
  // What is left for this to handle is the case that release cannot: at app
  // quit the host itself may be torn down first, so bd->* point at dead
  // widgets (they are never nulled when a widget dies) and walking them is the
  // burst of GTK_IS_WIDGET criticals on exit. Reparenting a destroyed box into
  // a destroyed iopw achieves nothing anyway; only the host bookkeeping still
  // matters, so do just that.
  if(darktable.develop->proxy.masks_flexi_host.hosted_module == module)
  {
    if(bd->relocatable_box && GTK_IS_WIDGET(bd->relocatable_box))
      _masks_flexi_release(module);
    else
      darktable.develop->proxy.masks_flexi_host.hosted_module = NULL;
  }

  _preview_on_hover_cancel_dwell(bd);

  dt_pthread_mutex_lock(&bd->lock);
  // a queued masks-list rebuild (_queue_masks_list_rebuild) left pending past
  // this teardown would otherwise fire later on the main loop and dereference
  // the widgets/blend_data freed below -- observed live as a burst of
  // GTK_IS_WIDGET/GTK_IS_BOX critical warnings right at darkroom exit/app quit.
  if(bd->masks_rebuild_idle_id) g_source_remove(bd->masks_rebuild_idle_id);

  if(bd->masks_cluster_expanded) g_hash_table_destroy(bd->masks_cluster_expanded);
  if(bd->masks_props_expanded) g_hash_table_destroy(bd->masks_props_expanded);
  if(bd->masks_refine_expanded) g_hash_table_destroy(bd->masks_refine_expanded);
  if(bd->masks_refine_bypassed) g_hash_table_destroy(bd->masks_refine_bypassed);
  if(bd->masks_row_map) g_hash_table_destroy(bd->masks_row_map);
  if(bd->group_ordinals) g_hash_table_destroy(bd->group_ordinals);
  dt_pthread_mutex_unlock(&bd->lock);
  dt_pthread_mutex_destroy(&bd->lock);

  g_free(module->blend_data);
  module->blend_data = NULL;
}


static gboolean _add_blendmode_combo(GtkWidget *combobox,
                                     const dt_develop_blend_mode_t start,
                                     const dt_develop_blend_mode_t end)
{
  return dt_bauhaus_combobox_add_introspection(combobox,
                                               NULL,
                                               dt_develop_blend_mode_names,
                                               start,
                                               end);
}

static GtkWidget *_combobox_new_from_list(dt_iop_module_t *module,
                                          const gchar *label,
                                          const dt_introspection_type_enum_tuple_t *list,
                                          uint32_t *field,
                                          const gchar *tooltip)
{
  GtkWidget *combo = dt_bauhaus_combobox_new(module);

  if(field)
    dt_bauhaus_widget_set_field(combo, field, DT_INTROSPECTION_TYPE_ENUM);
  dt_action_t *ac = dt_bauhaus_widget_set_label(combo, N_("blend"), label);
  gtk_widget_set_tooltip_text(combo, tooltip);
  dt_bauhaus_combobox_add_introspection(combo, ac, list, list[0].value, -1);

  return combo;
}

void dt_iop_gui_update_blending(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_develop_blend_params_t *bp = module->blend_params;

  if(!(module->flags() & IOP_FLAGS_SUPPORTS_BLENDING)
     || !bd
     || !bd->blend_inited)
    return;

  DT_ENTER_GUI_UPDATE();

  // update color space from parameters
  const dt_develop_blend_colorspace_t default_csp =
    dt_develop_blend_default_module_blend_colorspace(module);
  switch(default_csp)
  {
    case DEVELOP_BLEND_CS_RAW:
      bd->csp = DEVELOP_BLEND_CS_RAW;
      break;
    case DEVELOP_BLEND_CS_LAB:
    case DEVELOP_BLEND_CS_RGB_DISPLAY:
    case DEVELOP_BLEND_CS_RGB_SCENE:
      switch(bp->blend_cst)
      {
        case DEVELOP_BLEND_CS_LAB:
        case DEVELOP_BLEND_CS_RGB_DISPLAY:
        case DEVELOP_BLEND_CS_RGB_SCENE:
          bd->csp = bp->blend_cst;
          break;
        default:
          bd->csp = default_csp;
          break;
      }
      break;
    case DEVELOP_BLEND_CS_NONE:
    default:
      bd->csp = DEVELOP_BLEND_CS_NONE;
      break;
  }

  const gboolean is_mask_enabled = (bp->mask_mode != DEVELOP_MASK_DISABLED);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->mask_enable_toggle),
                               is_mask_enabled);
  _update_mask_enable_toggle_tooltip(bd->mask_enable_toggle, is_mask_enabled);
  _masks_panel_apply_enabled_state(bd, is_mask_enabled);
  if(bd->masks_blend_header)
  {
    if(is_mask_enabled)
      dt_gui_add_class(bd->masks_blend_header, "mask-enabled");
    else
      dt_gui_remove_class(bd->masks_blend_header, "mask-enabled");
  }
  if(darktable.develop->proxy.masks_flexi_host.hosted_module == module)
    dt_ui_flexi_panel_set_active(darktable.gui->ui, is_mask_enabled);

  const gboolean has_mask_display =
    (module->request_mask_display != DT_DEV_PIXELPIPE_DISPLAY_NONE);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->showmask), has_mask_display);

  // details-threshold refinement (bp->details) carves a real, non-uniform
  // mask out of image detail even with no drawn/parametric/raster mask type
  // engaged at all (see dt_develop_blend_process's own `uniform` branch,
  // which now applies it) -- so the show-mask/suppress controls and the
  // header mask indicator should be reachable in that case too, not just
  // when a mask_mode type bit is set.
  const gboolean valid_masking =
    (bp->mask_mode & ~DEVELOP_MASK_ENABLED) || bp->details != 0.0f;

  // (un)set the mask indicator
  dt_iop_add_remove_mask_indicator(module, valid_masking);

  // initialization of blending modes
  if(bd->csp != bd->blend_modes_csp)
  {
    dt_bauhaus_combobox_clear(bd->blend_modes_combo);

    if(bd->csp == DEVELOP_BLEND_CS_LAB
       || bd->csp == DEVELOP_BLEND_CS_RGB_DISPLAY
       || bd->csp == DEVELOP_BLEND_CS_RAW )
    {
      dt_bauhaus_combobox_add_section(bd->blend_modes_combo, _("normal & difference"));
      _add_blendmode_combo(bd->blend_modes_combo,
                           DEVELOP_BLEND_NORMAL2, DEVELOP_BLEND_DIFFERENCE2);
      _add_blendmode_combo(bd->blend_modes_combo,
                           DEVELOP_BLEND_BOUNDED, DEVELOP_BLEND_BOUNDED);
      dt_bauhaus_combobox_add_section(bd->blend_modes_combo, _("lighten"));
      _add_blendmode_combo(bd->blend_modes_combo,
                           DEVELOP_BLEND_LIGHTEN, DEVELOP_BLEND_LIGHTEN);
      _add_blendmode_combo(bd->blend_modes_combo,
                           DEVELOP_BLEND_ADD, DEVELOP_BLEND_ADD);
      _add_blendmode_combo(bd->blend_modes_combo,
                           DEVELOP_BLEND_SCREEN, DEVELOP_BLEND_SCREEN);
      dt_bauhaus_combobox_add_section(bd->blend_modes_combo, _("darken"));
      _add_blendmode_combo(bd->blend_modes_combo,
                           DEVELOP_BLEND_DARKEN, DEVELOP_BLEND_DARKEN);
      _add_blendmode_combo(bd->blend_modes_combo,
                           DEVELOP_BLEND_SUBTRACT, DEVELOP_BLEND_SUBTRACT);
      _add_blendmode_combo(bd->blend_modes_combo,
                           DEVELOP_BLEND_MULTIPLY, DEVELOP_BLEND_MULTIPLY);
      dt_bauhaus_combobox_add_section(bd->blend_modes_combo, _("contrast enhancing"));
      _add_blendmode_combo(bd->blend_modes_combo,
                           DEVELOP_BLEND_OVERLAY, DEVELOP_BLEND_PINLIGHT);

      if(bd->csp == DEVELOP_BLEND_CS_LAB
         || bd->csp == DEVELOP_BLEND_CS_RGB_DISPLAY)
      {
        dt_bauhaus_combobox_add_section(bd->blend_modes_combo, _("color channel"));
        if(bd->csp == DEVELOP_BLEND_CS_LAB)
          _add_blendmode_combo(bd->blend_modes_combo,
                               DEVELOP_BLEND_LAB_LIGHTNESS, DEVELOP_BLEND_LAB_COLOR);
        else
          _add_blendmode_combo(bd->blend_modes_combo,
                               DEVELOP_BLEND_RGB_R, DEVELOP_BLEND_HSV_COLOR);
        _add_blendmode_combo(bd->blend_modes_combo,
                             DEVELOP_BLEND_HUE, DEVELOP_BLEND_COLORADJUST);

        dt_bauhaus_combobox_add_section(bd->blend_modes_combo,
                                        _("chromaticity & lightness"));
        _add_blendmode_combo(bd->blend_modes_combo,
                             DEVELOP_BLEND_LIGHTNESS, DEVELOP_BLEND_CHROMATICITY);
      }
    }
    else if(bd->csp == DEVELOP_BLEND_CS_RGB_SCENE)
    {
      dt_bauhaus_combobox_add_section(bd->blend_modes_combo, _("normal & arithmetic"));
      _add_blendmode_combo(bd->blend_modes_combo,
                           DEVELOP_BLEND_NORMAL2, DEVELOP_BLEND_DIFFERENCE2);
      _add_blendmode_combo(bd->blend_modes_combo,
                           DEVELOP_BLEND_MULTIPLY, DEVELOP_BLEND_HARMONIC_MEAN);
      dt_bauhaus_combobox_add_section(bd->blend_modes_combo, _("color channel"));
      _add_blendmode_combo(bd->blend_modes_combo,
                           DEVELOP_BLEND_RGB_R, DEVELOP_BLEND_RGB_B);
      dt_bauhaus_combobox_add_section(bd->blend_modes_combo, _("chromaticity & lightness"));
      _add_blendmode_combo(bd->blend_modes_combo,
                           DEVELOP_BLEND_LIGHTNESS, DEVELOP_BLEND_CHROMATICITY);
    }
    bd->blend_modes_csp = bd->csp;
  }

  dt_develop_blend_mode_t blend_mode = bp->blend_mode & DEVELOP_BLEND_MODE_MASK;

  if(!dt_bauhaus_combobox_set_from_value(bd->blend_modes_combo, blend_mode))
  {
    // add deprecated blend mode
    dt_bauhaus_combobox_add_section(bd->blend_modes_combo, _("deprecated"));
    if(!_add_blendmode_combo(bd->blend_modes_combo, blend_mode, blend_mode))
    {
      // should never happen: unknown blend mode
      dt_control_log(_("unknown blend mode '%d' in module '%s'"), blend_mode, module->op);
      bp->blend_mode = DEVELOP_BLEND_NORMAL2;
      blend_mode = DEVELOP_BLEND_NORMAL2;
    }

    dt_bauhaus_combobox_set_from_value(bd->blend_modes_combo, blend_mode);
  }

  const gboolean blend_mode_reversed =
    (bp->blend_mode & DEVELOP_BLEND_REVERSE) == DEVELOP_BLEND_REVERSE;

  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->blend_modes_blend_order),
                               blend_mode_reversed);

  dt_bauhaus_slider_set(bd->blend_mode_parameter_slider, bp->blend_parameter);
  gtk_widget_set_visible(bd->blend_mode_parameter_slider,
     _blendif_blend_parameter_enabled(bd->blend_modes_csp, bp->blend_mode));

  dt_bauhaus_combobox_set_from_value(bd->masks_combine_combo,
    bp->mask_combine & (DEVELOP_COMBINE_INV | DEVELOP_COMBINE_INCL));
  dt_bauhaus_slider_set(bd->opacity_slider, bp->opacity);
  if(bd->blend_opacity_lowop_badge)
    _update_lowop_badge(bd->blend_opacity_lowop_badge, bp->opacity / 100.0f, FALSE, FALSE, NULL);
  dt_bauhaus_combobox_set_from_value(bd->masks_feathering_guide_combo, bp->feathering_guide);
  dt_bauhaus_slider_set(bd->feathering_radius_slider, bp->feathering_radius);
  dt_bauhaus_slider_set(bd->blur_radius_slider, bp->blur_radius);
  dt_bauhaus_slider_set(bd->brightness_slider, bp->brightness);
  dt_bauhaus_slider_set(bd->contrast_slider, bp->contrast);
  dt_bauhaus_slider_set(bd->details_slider, bp->details);
  _update_refine_sensitivity(module);

  /* reset all alternative display modes for blendif */
  memset(bd->altmode, 0, sizeof(bd->altmode));

  // keep the flexi "add parametric" channel buttons in sync with the csp
  _rebuild_param_channel_buttons(module);

  dt_iop_gui_update_masks(module);

  /* now show hide controls as required */
  const dt_develop_mask_mode_t mask_mode = bp->mask_mode;
  const gboolean mask_enabled = mask_mode & DEVELOP_MASK_ENABLED;
  const gboolean mode_raster = mask_mode & DEVELOP_MASK_RASTER;
  const gboolean mode_drawn = mask_mode & DEVELOP_MASK_MASK;
  const gboolean mode_flexi = !mode_raster && (mask_enabled || (mask_mode & DEVELOP_MASK_FLEXI));
  const gboolean mode_parametric = mask_mode & DEVELOP_MASK_CONDITIONAL;
  // flexi reuses the drawn-group toolbar/renderer (see _blendop_masks_mode_callback)
  const gboolean mode_drawn_or_flexi = mode_drawn || mode_flexi;
  // mask off shows the flexi panel greyed out rather than an empty panel --
  // see _blendop_masks_mode_callback, which this mirrors
  const gboolean show_mask_ui = !mode_raster;
  const gboolean show_flexi_ui = !mode_raster;

  _box_set_visible(bd->blend_box, TRUE);

  const dt_image_t img = module->dev->image_storage;
  gtk_widget_set_visible(bd->details_slider, dt_image_is_rawprepare_supported(&img));

  if(mask_enabled
     && ((bd->masks_inited && mode_drawn_or_flexi)
         || (bd->blendif_inited && mode_parametric)))
  {
    gtk_widget_set_visible(GTK_WIDGET(bd->masks_combine_combo), bd->blendif_inited && mode_parametric);

    // flexi-only refinement embellishment (the per-target reset) never appears
    // in the classic drawn/parametric panels. Gated here authoritatively (the
    // target-suffix caption is handled in _refine_update_header), so the
    // classic refinement header stays vanilla.
    if(bd->masks_refine_reset_btn)
      gtk_widget_set_visible(bd->masks_refine_reset_btn, mode_flexi);

    /*
     * if this iop is operating in raw space, it has only 1 channel per pixel,
     * thus there is no alpha channel where we would normally store mask
     * that would get displayed if following button have been pressed.
     *
     * TODO: revisit if/once there semi-raw iops (e.g temperature) with blending
     */
    if(module->blend_colorspace(module, NULL, NULL) == IOP_CS_RAW)
    {
      module->request_mask_display = DT_DEV_PIXELPIPE_DISPLAY_NONE;
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->showmask), FALSE);
      // (re)set the header mask indicator too
      if(module->mask_indicator)
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(module->mask_indicator), FALSE);
    }

    _box_set_visible(bd->refine_box, TRUE);
  }
  else
  {
    module->request_mask_display = DT_DEV_PIXELPIPE_DISPLAY_NONE;
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->showmask), FALSE);
    // (re)set the header mask indicator too
    if(module->mask_indicator)
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(module->mask_indicator), FALSE);

    // mask off: still shown, greyed, as the preview described above
    _box_set_visible(bd->refine_box,
                     !mask_enabled || (bd->raster_inited && mode_raster));
  }

  if(bd->masks_inited && show_mask_ui)
  {
    // flexi-only widgets: new-shape operator selector, add-parametric button,
    // and the per-shape composition list (classic drawn mask stays vanilla)
    if(bd->masks_param_channels_box)
      gtk_widget_set_visible(bd->masks_param_channels_box,
                             show_flexi_ui && bd->blendif_support);
    gtk_widget_set_visible(bd->masks_toolbar, show_flexi_ui);
    if(bd->soloedit_mode) gtk_widget_set_visible(bd->soloedit_mode, show_flexi_ui);
    gtk_widget_set_visible(GTK_WIDGET(bd->masks_list_box), show_flexi_ui);
    _box_set_visible(bd->masks_box, TRUE);
    _props_panel_show(bd);
    // (re)build the per-shape composition list for this module's group -- only
    // for a live mask; with the mask off the list keeps what it last held
    // unless the group was deleted or emptied, or it was never built. An off
    // mask shows the groups switching it on would show, so the toggle never
    // changes the structure the panel displays
    dt_masks_form_t *grp = _module_mask_group(module);
    const gboolean has_group = grp && grp->points;
    const gboolean had_list = bd->masks_list_sig != DT_INVALID_HASH;
    if(mode_flexi || !has_group || !had_list) _build_masks_list(module);
    // and nothing of an off mask belongs on canvas (this used to fall to the
    // classic branch below, which is now reached only by a live classic mask)
    if(!mask_enabled) dt_masks_set_edit_mode(module, DT_MASKS_EDIT_OFF);
  }
  else if(bd->masks_inited)
  {
    dt_masks_set_edit_mode(module, DT_MASKS_EDIT_OFF);
    _box_set_visible(bd->masks_box, FALSE);
    _box_set_visible(bd->props_panel_box, FALSE);
  }
  else
  {
    _box_set_visible(bd->masks_box, FALSE);
    _box_set_visible(bd->props_panel_box, FALSE);
  }

  _box_set_visible(bd->raster_box, bd->raster_inited && mode_raster);

  if(bd->blendif_inited && mode_parametric)
  {
    _box_set_visible(bd->blendif_box, TRUE);
  }
  else if(bd->blendif_inited)
  {
    /* switch off color picker */
    dt_iop_color_picker_reset(module, FALSE);

    _box_set_visible(bd->blendif_box, FALSE);
  }
  else
  {
    _box_set_visible(bd->blendif_box, FALSE);
  }

  // modules that can't be toggled on/off in the first place (see
  // module->hide_enable_button) don't get a blend-mask on/off control either
  gtk_widget_set_visible(bd->mask_enable_toggle, !module->hide_enable_button);
  gtk_widget_hide(bd->masks_options_btn);  // options open on the toggle's right-click now
  gtk_widget_set_visible(bd->showmask, is_mask_enabled && !module->hide_enable_button);

  if(darktable.develop && darktable.develop->gui_module == module)
    _masks_flexi_relocate(module);

  DT_LEAVE_GUI_UPDATE();
}

// the mask overlay and "edit on canvas" mode the previously focused module had,
// handed over to the next focused module so masks of different modules can be
// compared and edited without switching them on each time
static dt_dev_pixelpipe_display_mask_t _focus_carried_mask_display = DT_DEV_PIXELPIPE_DISPLAY_NONE;
static dt_masks_edit_mode_t _focus_carried_edit = DT_MASKS_EDIT_OFF;
static guint _focus_carry_drop_source = 0;

// focus left the module and went nowhere: nothing is handed on
static gboolean _focus_carry_drop(gpointer user_data)
{
  _focus_carry_drop_source = 0;
  _focus_carried_mask_display = DT_DEV_PIXELPIPE_DISPLAY_NONE;
  _focus_carried_edit = DT_MASKS_EDIT_OFF;
  return G_SOURCE_REMOVE;
}

// a masked module without shapes has nothing to edit on canvas, so it holds the
// edit mode for the next module instead of dropping it (see
// dt_iop_gui_blending_lose_focus)
static dt_iop_module_t *_focus_edit_holder = NULL;

static void _carry_edit_to(dt_iop_module_t *module)
{
  const dt_masks_edit_mode_t carried = _focus_carried_edit;
  _focus_carried_edit = DT_MASKS_EDIT_OFF;
  _focus_edit_holder = NULL;

  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(carried == DT_MASKS_EDIT_OFF
     || !bd->masks_support
     || !bd->masks_edit
     || module->blend_params->mask_mode == DEVELOP_MASK_DISABLED
     || gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(bd->masks_edit)))
    return;

  if(!_module_has_drawn_shapes(module))
  {
    _focus_carried_edit = carried;
    _focus_edit_holder = module;
    return;
  }

  bd->masks_shown = carried;
  DT_ENTER_GUI_UPDATE();
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_edit), TRUE);
  DT_LEAVE_GUI_UPDATE();
  dt_masks_set_edit_mode(module, carried);
}

static void _carry_mask_display_to(dt_iop_module_t *module)
{
  const dt_dev_pixelpipe_display_mask_t carried = _focus_carried_mask_display;
  _focus_carried_mask_display = DT_DEV_PIXELPIPE_DISPLAY_NONE;

  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!carried
     || !module->enabled
     || module->hide_enable_button
     || module->blend_params->mask_mode == DEVELOP_MASK_DISABLED
     || module->blend_colorspace(module, NULL, NULL) == IOP_CS_RAW
     || module->request_mask_display != DT_DEV_PIXELPIPE_DISPLAY_NONE)
    return;

  module->request_mask_display = carried;
  DT_ENTER_GUI_UPDATE();
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->showmask), TRUE);
  if(module->mask_indicator)
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(module->mask_indicator), TRUE);
  DT_LEAVE_GUI_UPDATE();
  dt_iop_refresh_center(module);
}

void dt_iop_gui_blending_gain_focus(dt_iop_module_t *module)
{
  if(!module || !module->blend_data)
  {
    _focus_carried_mask_display = DT_DEV_PIXELPIPE_DISPLAY_NONE;
    _focus_carried_edit = DT_MASKS_EDIT_OFF;
    return;
  }
  _masks_flexi_relocate(module);
  _carry_mask_display_to(module);
  _carry_edit_to(module);
}

void dt_iop_gui_blending_lose_focus(dt_iop_module_t *module)
{
  DT_GUARD_GUI_UPDATE();
  if(!module) return;

  // stepping into an AI object (see dt_masks_form_gui_t.entered_object) is
  // part of editing this module's mask on the canvas: it ends with it
  _step_object(module, INVALID_MASKID);

  const gboolean has_mask_display =
    module->request_mask_display
    & (DT_DEV_PIXELPIPE_DISPLAY_MASK | DT_DEV_PIXELPIPE_DISPLAY_CHANNEL);

  const gboolean suppress = module->suppress_mask;

  if((module->flags() & IOP_FLAGS_SUPPORTS_BLENDING) && module->blend_data)
  {
    dt_iop_gui_blend_data_t *bd = module->blend_data;

    // a running hover preview is not the overlay the user asked for;
    // save_for_leave holds that one. Channel displays are per-module, so
    // only the plain mask overlay travels. A focus on no module keeps it until
    // idle: expanding with single_module collapses the old module first, which
    // focuses NULL just before the new one (see _gui_set_single_expanded)
    dt_pthread_mutex_lock(&bd->lock);
    const dt_dev_pixelpipe_display_mask_t shown =
      bd->hover_preview_active ? bd->save_for_leave : module->request_mask_display;
    dt_pthread_mutex_unlock(&bd->lock);
    _focus_carried_mask_display = shown & DT_DEV_PIXELPIPE_DISPLAY_MASK;
    const gboolean editing =
      bd->masks_support && bd->masks_edit
      && gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(bd->masks_edit));
    if(editing)
      _focus_carried_edit = bd->masks_shown;
    else if(module != _focus_edit_holder)
      _focus_carried_edit = DT_MASKS_EDIT_OFF;
    _focus_edit_holder = NULL;
    if(!darktable.develop->gui_module
       && (_focus_carried_mask_display || _focus_carried_edit)
       && !_focus_carry_drop_source)
      _focus_carry_drop_source = g_idle_add(_focus_carry_drop, NULL);

    // don't let the flexi masks panel content linger in a shared host once
    // its owning module loses focus
    if(darktable.develop->proxy.masks_flexi_host.hosted_module == module)
    {
      _masks_flexi_release(module);
    }

    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->showmask), FALSE);
    module->request_mask_display = DT_DEV_PIXELPIPE_DISPLAY_NONE;
    module->suppress_mask = FALSE;

    // (re)set the header mask indicator too
    DT_ENTER_GUI_UPDATE();
    if(module->mask_indicator)
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(module->mask_indicator), FALSE);
    DT_LEAVE_GUI_UPDATE();

    if(bd->masks_support)
    {
      // unselect all tools
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_edit), FALSE);
      dt_masks_set_edit_mode(module, DT_MASKS_EDIT_OFF);

      for(int k=0; k < DEVELOP_MASKS_NB_SHAPES; k++)
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_shapes[k]), FALSE);
    }

    // request_mask_display was just cleared above, so a hover preview that was
    // running has nothing left to restore
    _preview_on_hover_cancel_dwell(bd);
    bd->hovered_channel_widget = NULL;
    dt_pthread_mutex_lock(&bd->lock);
    bd->hover_preview_active = FALSE;
    bd->save_for_leave = DT_DEV_PIXELPIPE_DISPLAY_NONE;
    dt_pthread_mutex_unlock(&bd->lock);

    // reprocess main center image if needed
    if(has_mask_display || suppress)
      dt_iop_refresh_center(module);
  }
}

void dt_iop_gui_blending_reload_defaults(dt_iop_module_t *module)
{
  if(!module) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd || !bd->blendif_support || !bd->blendif_inited) return;
}

void dt_iop_gui_init_blending(GtkWidget *iopw,
                              dt_iop_module_t *module)
{
  /* create and add blend mode if module supports it */
  if(module->flags() & IOP_FLAGS_SUPPORTS_BLENDING)
  {
    DT_ENTER_GUI_UPDATE();
    --darktable.bauhaus->skip_accel;

    module->blend_data = g_malloc0(sizeof(dt_iop_gui_blend_data_t));
    dt_iop_gui_blend_data_t *bd = module->blend_data;
    dt_develop_blend_params_t *bp = module->blend_params;

    bd->iopw = iopw;
    bd->module = module;
    bd->csp = DEVELOP_BLEND_CS_NONE;
    bd->blend_modes_csp = DEVELOP_BLEND_CS_NONE;
    bd->channel_tabs_csp = DEVELOP_BLEND_CS_NONE;
    dt_iop_colorspace_type_t cst = module->blend_colorspace(module, NULL, NULL);
    bd->blendif_support = (cst == IOP_CS_LAB || cst == IOP_CS_RGB);
    // classic blendif's tabbed channel-editor widgets no longer get built here
    // (flexi replaced them with per-row editors), but blendif_inited is still
    // read everywhere as "blendif is usable for this module" -- it used to be
    // set at the end of the now-removed dt_iop_gui_init_blendif, so set it
    // here instead, gated the same way that function was.
    bd->blendif_inited = bd->blendif_support;
    bd->masks_support = !(module->flags() & IOP_FLAGS_NO_MASKS);

    dt_pthread_mutex_init(&bd->lock, NULL);
    dt_pthread_mutex_lock(&bd->lock);
    bd->save_for_leave = 0;
    dt_pthread_mutex_unlock(&bd->lock);

    // collapse control for the masking panel: in the separate flexi panel
    // (left/right) it folds the whole panel away to its canvas corner icon,
    // embedded in the module it folds the panel body away below this header
    // and doubles as the way back (see _flexi_inline_collapse_clicked, which
    // dispatches on the position, and sets the arrow direction and tooltip
    // to match). Hidden only in the utility-lib position, which collapses
    // via the lib's own expander header. A plain flat arrow with its own CSS
    // class, deliberately not styled like the on/off toggle next to it.
    bd->flexi_inline_collapse_btn =
      dtgtk_button_new(dtgtk_cairo_paint_solid_arrow, CPF_DIRECTION_LEFT, NULL);
    gtk_widget_set_name(bd->flexi_inline_collapse_btn, "flexi-inline-collapse");
    g_signal_connect(G_OBJECT(bd->flexi_inline_collapse_btn), "clicked",
                     G_CALLBACK(_flexi_inline_collapse_clicked), module);
    gtk_widget_set_no_show_all(bd->flexi_inline_collapse_btn, TRUE);
    gtk_widget_set_visible(bd->flexi_inline_collapse_btn, FALSE);
    gtk_widget_set_valign(bd->flexi_inline_collapse_btn, GTK_ALIGN_CENTER);
    // no dt_action_define_iop here: the matching shortcut is a command action
    // instead, so that it still works in the position that hides this button
    // (see _shortcut_toggle_masks_panel)

    // on/off toggle for the whole blend mask (DEVELOP_MASK_DISABLED vs
    // DEVELOP_MASK_ENABLED|DEVELOP_MASK_FLEXI) -- flexi is the only mask
    // type left, so there is nothing left to pick a "type" from, just on/off
    // (see _blendop_mask_enable_toggled)
    bd->mask_enable_toggle =
      dt_iop_togglebutton_new(module, "blend`masks", N_("mask enabled"), NULL,
                              G_CALLBACK(_blendop_mask_enable_toggled), FALSE, 0, 0,
                              dtgtk_cairo_paint_switch, NULL);
    _update_mask_enable_toggle_tooltip(bd->mask_enable_toggle, FALSE);
    // background always blends with the module's own background, on or off
    // -- only the glyph itself shows state
    dt_gui_add_class(bd->mask_enable_toggle, "dt_transparent_background");
    dt_gui_add_class(bd->mask_enable_toggle, "mask-enable-toggle");
    gtk_widget_set_valign(bd->mask_enable_toggle, GTK_ALIGN_CENTER);

    // its own id rather than "iop-panel-label": embedded, this caption has to
    // line up with the module's name beside it, but the two headers are not
    // interchangeable and only the embedded position wants that metric (see
    // "#blending-tabs-caption" in darktable.css)
    GtkWidget *caption_label = dt_ui_label_new(_("blend mask"));
    gtk_widget_set_name(caption_label, "blending-tabs-caption");
    bd->masks_blend_header_label = caption_label;

    // "blend mask" header, in one fixed reading order:
    //
    //   expander | title | <space> | show_mask_overlay | <gap> | edit on canvas
    //   | solo edit | <gap> | on/off toggle
    //
    // The expander (the panel-collapse arrow, embedded position only) leads;
    // the caption follows; everything after the space closes on the right,
    // grouped into right_cluster below. The space in the middle is simply what
    // is left between the start-packed and end-packed halves. When docked in
    // the separate *right* panel, the expander moves to the far right -- see
    // _masks_header_apply_side. The preferences gear stays hidden: the
    // blending options open on the on/off toggle's right-click.
    GtkWidget *gbox =
      dt_gui_hbox(bd->flexi_inline_collapse_btn, caption_label);
    dt_gui_add_class(gbox, "dt_section_label");
    dt_gui_add_help_link(gbox, "masks_blending");
    gtk_widget_set_name(gbox, "blending-tabs");
    // default to the embedded inset (see darktable.css's "#blending-tabs.
    // blending-tabs-embedded"); _masks_flexi_relocate toggles this off for
    // the two hosted positions, which already provide their own inset
    dt_gui_add_class(gbox, "blending-tabs-embedded");
    // the blending tabs' own header; the panel can host it (see
    // _masks_flexi_relocate)
    bd->masks_blend_header = gbox;

    GtkWidget *presets_button = bd->masks_options_btn =
      dtgtk_button_new_full(dtgtk_cairo_paint_preferences, 0, NULL,
                            &(dtgtk_button_config_t){
                              .tooltip = _("blending options"),
                            });
    gtk_widget_set_valign(presets_button, GTK_ALIGN_CENTER);
    if(bd->blendif_support || bd->masks_support)
    {
      g_signal_connect(G_OBJECT(presets_button), "clicked",
                       G_CALLBACK(_blendif_options_callback), module);
    }
    else
    {
      gtk_widget_set_sensitive(GTK_WIDGET(presets_button), FALSE);
    }
    dt_action_define_iop(module, "blend`masks", N_("preferences"),
                         presets_button, &dt_action_def_button);

    bd->showmask = dt_iop_togglebutton_new(
      module, "blend`tools", N_("display mask and/or color channel"), NULL,
      G_CALLBACK(_blendop_blendif_showmask_clicked), FALSE, 0, 0,
      dtgtk_cairo_paint_showmask, NULL);
    gtk_widget_set_valign(bd->showmask, GTK_ALIGN_CENTER);
    gtk_widget_set_tooltip_text
      (bd->showmask,
       _("display mask and/or color channel.\n"
         "ctrl+click to display mask,\n"
         "shift+click to display channel"));

    // edit on canvas and solo edit, with a gap either side: they are built with
    // the rest of the mask controls, and packed in by _pack_header_edit_run.
    // Hidden until then, so a module without masks shows no stray gaps
    bd->masks_header_edit_box = dt_gui_hbox();
    dt_gui_add_class(bd->masks_header_edit_box, "masks-btn-row");
    gtk_widget_set_no_show_all(bd->masks_header_edit_box, TRUE);

    // right-hand cluster: show_mask_overlay, edit run, preferences (hidden),
    // on/off toggle
    GtkWidget *right_cluster = bd->masks_right_cluster =
      dt_gui_hbox(bd->showmask, bd->masks_header_edit_box, presets_button,
                  bd->mask_enable_toggle);
    dt_gui_add_class(right_cluster, "masks-btn-row");
    gtk_widget_set_valign(right_cluster, GTK_ALIGN_CENTER);
    gtk_box_pack_end(GTK_BOX(gbox), right_cluster, FALSE, FALSE, 0);

    bd->blend_modes_combo = dt_bauhaus_combobox_new(module);
    dt_action_t * ac = dt_bauhaus_widget_set_label(bd->blend_modes_combo,
                                                   N_("blend"),
                                                   N_("mode"));
    dt_bauhaus_combobox_add_introspection(bd->blend_modes_combo, ac,
                                          dt_develop_blend_mode_names, -1, -1);
    gtk_widget_set_tooltip_text(bd->blend_modes_combo, _("choose blending mode"));

    g_signal_connect(G_OBJECT(bd->blend_modes_combo), "value-changed",
                     G_CALLBACK(_blendop_blend_mode_callback), bd);
    dt_gui_add_help_link(GTK_WIDGET(bd->blend_modes_combo),
                         "masks_blending_op");

    bd->blend_modes_blend_order = dt_iop_togglebutton_new
      (module, "blend`tools",
       N_("toggle blend order"), NULL,
       G_CALLBACK(_blendop_blend_order_clicked), FALSE,
       0, 0,
       dtgtk_cairo_paint_invert, NULL);
    gtk_widget_set_tooltip_text
      (bd->blend_modes_blend_order,
       _("toggle the blending order between the input and the output of the module,\n"
         "by default the output will be blended on top of the input,\n"
         "order can be reversed by clicking on the icon (input on top of output)"));

    bd->blend_mode_parameter_slider =
      dt_bauhaus_slider_new_with_range(module, -18.0f, 18.0f, 0, 0.0f, 3);
    dt_bauhaus_widget_set_field(bd->blend_mode_parameter_slider, &bp->blend_parameter, DT_INTROSPECTION_TYPE_FLOAT);
    dt_bauhaus_widget_set_label(bd->blend_mode_parameter_slider, N_("blend"), N_("fulcrum"));
    dt_bauhaus_slider_set_format(bd->blend_mode_parameter_slider, _(" EV"));
    dt_bauhaus_slider_set_soft_range(bd->blend_mode_parameter_slider, -3.0, 3.0);
    gtk_widget_set_tooltip_text(bd->blend_mode_parameter_slider,
                                _("adjust the fulcrum used by some blending"
                                  " operations"));
    gtk_widget_set_visible(bd->blend_mode_parameter_slider, FALSE);

    bd->opacity_slider = dt_bauhaus_slider_new_with_range(module, 0.0, 100.0, 0, 100.0, 0);
    dt_bauhaus_widget_set_field(bd->opacity_slider, &bp->opacity, DT_INTROSPECTION_TYPE_FLOAT);
    dt_bauhaus_widget_set_label(bd->opacity_slider, N_("blend"), N_("opacity"));
    dt_bauhaus_slider_set_format(bd->opacity_slider, "%");
    gtk_widget_set_tooltip_text(bd->opacity_slider,
                                _("set the opacity of the blending"));
    // no quad icon on this slider -- without this it reserves the quad's
    // width unused, reading as narrower than it needs to be (same reasoning
    // as the props/boost-factor sliders' own identical call).
    dt_bauhaus_widget_set_quad_visibility(bd->opacity_slider, FALSE);
    _style_opacity_gradient(bd->opacity_slider);
    dt_bauhaus_widget_hide_label(bd->opacity_slider);
    dt_gui_add_class(bd->opacity_slider, "blend-main-opacity-slider");
    module->fusion_slider = bd->opacity_slider;

    GtkWidget *opacity_header = dt_gui_hbox();
    GtkWidget *opacity_lbl = gtk_label_new(_("opacity"));
    gtk_label_set_xalign(GTK_LABEL(opacity_lbl), 0.0f);
    dt_gui_box_add(opacity_header, dt_gui_expand(opacity_lbl));

    bd->blend_opacity_lowop_badge = _make_lowop_badge();
    GtkWidget *val_widget = _make_inline_opacity_value_widget(bd->opacity_slider, module);

    GtkWidget *val_box = dt_gui_hbox();
    dt_gui_box_add(val_box, bd->blend_opacity_lowop_badge, val_widget);

    gtk_box_pack_end(GTK_BOX(opacity_header), val_box, FALSE, FALSE, 0);

    g_signal_connect(G_OBJECT(bd->opacity_slider), "value-changed",
                     G_CALLBACK(_blend_opacity_slider_changed_cb), bd);

    GtkWidget *opacity_box = dt_gui_vbox();
    dt_gui_add_class(opacity_box, "blend-main-opacity-box");
    dt_gui_box_add(opacity_box, opacity_header, bd->opacity_slider);

    bd->masks_combine_combo = _combobox_new_from_list
      (module,
       N_("combine masks"),
       dt_develop_combine_masks_names, NULL,
       _("how to combine individual drawn mask and different channels of parametric mask"));
    g_signal_connect(G_OBJECT(bd->masks_combine_combo), "value-changed",
                     G_CALLBACK(_blendop_masks_combine_callback), bd);
    dt_gui_add_help_link(GTK_WIDGET(bd->masks_combine_combo),
                         "masks_combined");

    bd->details_slider = dt_bauhaus_slider_new_with_range(module, -1.0f, 1.0f, 0, 0.0f, 2);
    dt_bauhaus_widget_set_label(bd->details_slider, N_("blend"), N_("details threshold"));
    dt_bauhaus_slider_set_format(bd->details_slider, "%");
    gtk_widget_set_tooltip_text
      (bd->details_slider,
       _("adjust the threshold for the details mask (using raw data),\n"
         "positive values select areas with strong details,\n"
         "negative values select flat areas"));
    dt_bauhaus_widget_set_quad_visibility(bd->details_slider, FALSE);
    g_signal_connect(G_OBJECT(bd->details_slider), "value-changed",
                     G_CALLBACK(_refine_control_changed), bd);

    // NB: the six "mask refinement" controls are deliberately *not* bound to
    // blend_params via dt_bauhaus_widget_set_field. They are driven by the
    // unified _refine_control_changed handler so they can target one of three
    // scopes (global / all shapes / a parametric form). In global scope it
    // writes blend_params exactly as the old set_field bindings did, so classic
    // and flexi-global refinement stay byte-identical.
    bd->masks_feathering_guide_combo = _combobox_new_from_list(
      module, N_("feathering guide"), dt_develop_feathering_guide_names, NULL,
      _("choose to guide mask by input or output image and\n"
        "choose to apply feathering before or after mask blur"));
    g_signal_connect(G_OBJECT(bd->masks_feathering_guide_combo), "value-changed",
                     G_CALLBACK(_refine_control_changed), bd);

    bd->feathering_radius_slider =
      dt_bauhaus_slider_new_with_range(module, 0.0, 250.0, 0, 0.0, 1);
    dt_bauhaus_widget_set_label(bd->feathering_radius_slider,
                                N_("blend"), N_("feathering radius"));
    dt_bauhaus_slider_set_format(bd->feathering_radius_slider, _(" px"));
    gtk_widget_set_tooltip_text(bd->feathering_radius_slider,
                                _("spatial radius of feathering"));
    dt_bauhaus_widget_set_quad_visibility(bd->feathering_radius_slider, FALSE);
    g_signal_connect(G_OBJECT(bd->feathering_radius_slider), "value-changed",
                     G_CALLBACK(_refine_control_changed), bd);

    bd->blur_radius_slider =
      dt_bauhaus_slider_new_with_range(module, 0.0, 100.0, 0, 0.0, 1);
    dt_bauhaus_widget_set_label(bd->blur_radius_slider, N_("blend"), N_("blurring radius"));
    dt_bauhaus_slider_set_format(bd->blur_radius_slider, _(" px"));
    gtk_widget_set_tooltip_text(bd->blur_radius_slider,
                                _("radius for gaussian blur of blend mask"));
    dt_bauhaus_widget_set_quad_visibility(bd->blur_radius_slider, FALSE);
    g_signal_connect(G_OBJECT(bd->blur_radius_slider), "value-changed",
                     G_CALLBACK(_refine_control_changed), bd);

    bd->brightness_slider = dt_bauhaus_slider_new_with_range(module, -1.0, 1.0, 0, 0.0, 2);
    dt_bauhaus_widget_set_label(bd->brightness_slider, N_("blend"),
                                N_("mask brightness"));
    dt_bauhaus_slider_set_format(bd->brightness_slider, "%");
    gtk_widget_set_tooltip_text
      (bd->brightness_slider,
       _("shifts and tilts the tone curve of the blend mask to adjust its brightness\n"
         "without affecting fully transparent/fully opaque regions"));
    dt_bauhaus_widget_set_quad_visibility(bd->brightness_slider, FALSE);
    g_signal_connect(G_OBJECT(bd->brightness_slider), "value-changed",
                     G_CALLBACK(_refine_control_changed), bd);

    bd->contrast_slider = dt_bauhaus_slider_new_with_range(module, -1.0, 1.0, 0, 0.0, 2);
    dt_bauhaus_widget_set_label(bd->contrast_slider, N_("blend"), N_("mask contrast"));
    dt_bauhaus_slider_set_format(bd->contrast_slider, "%");
    gtk_widget_set_tooltip_text
      (bd->contrast_slider,
       _("gives the tone curve of the blend mask an s-like shape to "
         "adjust its contrast"));
    dt_bauhaus_widget_set_quad_visibility(bd->contrast_slider, FALSE);
    g_signal_connect(G_OBJECT(bd->contrast_slider), "value-changed",
                     G_CALLBACK(_refine_control_changed), bd);

    // Expander header bar (darktable standard section expander):
    // shows "(element|group|whole mask) refinement" centered, and the solid arrow toggle
    // on the right.
    GtkWidget *destdisp_head = dt_gui_hbox();
    gtk_box_set_spacing(GTK_BOX(destdisp_head), DT_BAUHAUS_SPACE);
    dt_gui_add_class(destdisp_head, "dt_section_expander");
    dt_gui_add_class(destdisp_head, "mask-refine-section-expander");

    bd->masks_refine_indicator_icon =
      _make_icon_widget(dtgtk_cairo_paint_masks_vertgradient);
    gtk_widget_set_size_request(bd->masks_refine_indicator_icon, DT_PIXEL_APPLY_DPI(14),
                                DT_PIXEL_APPLY_DPI(14));
    gtk_widget_set_valign(bd->masks_refine_indicator_icon, GTK_ALIGN_CENTER);
    gtk_widget_set_halign(bd->masks_refine_indicator_icon, GTK_ALIGN_CENTER);
    gtk_widget_set_opacity(bd->masks_refine_indicator_icon, 0.25);
    gtk_widget_set_tooltip_text(bd->masks_refine_indicator_icon,
                                _("no refinements for this target"));

    GtkWidget *icon_evb = gtk_event_box_new();
    dt_gui_add_class(icon_evb, "mask-refine-indicator-box");
    gtk_container_add(GTK_CONTAINER(icon_evb), bd->masks_refine_indicator_icon);
    dt_gui_connect_click(icon_evb, _refine_header_clicked, NULL, bd);

    bd->masks_refine_section_label = dt_ui_section_label_new(_("whole mask refinement"));
    gtk_widget_set_tooltip_text(bd->masks_refine_section_label,
                                _("refinements follow the panel selection: an element, a "
                                  "group, or the whole mask if nothing is selected."));
    _stash_base_tooltip(bd->masks_refine_section_label);

    GtkWidget *header_evb = gtk_event_box_new();
    gtk_container_add(GTK_CONTAINER(header_evb), bd->masks_refine_section_label);
    dt_gui_connect_click(header_evb, _refine_header_clicked, NULL, bd);

    bd->masks_refine_toggle_btn =
      dtgtk_togglebutton_new(dtgtk_cairo_paint_solid_arrow, CPF_DIRECTION_DOWN, NULL);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_refine_toggle_btn), TRUE);
    dt_gui_add_class(bd->masks_refine_toggle_btn, "dt_ignore_fg_state");
    dt_gui_add_class(bd->masks_refine_toggle_btn, "dt_transparent_background");
    gtk_widget_set_tooltip_text(bd->masks_refine_toggle_btn,
                                _("toggle refinements section"));
    g_signal_connect(G_OBJECT(bd->masks_refine_toggle_btn), "toggled",
                     G_CALLBACK(_refine_toggle_toggled), module);

    dt_gui_box_add(destdisp_head, icon_evb, dt_gui_expand(header_evb));
    gtk_box_pack_end(GTK_BOX(destdisp_head), bd->masks_refine_toggle_btn, FALSE, FALSE,
                     0);

    // Inside the expanded section:
    // Top row showing: <icon> <label> <actions>
    GtkWidget *inner_header_row = dt_gui_hbox();
    gtk_box_set_spacing(GTK_BOX(inner_header_row), 4);
    dt_gui_add_class(inner_header_row, "mask-refine-inner-header");

    bd->masks_refine_icon_box = dt_gui_hbox();
    gtk_widget_set_valign(bd->masks_refine_icon_box, GTK_ALIGN_CENTER);

    bd->masks_refine_name_label = gtk_label_new(_("whole mask"));
    gtk_label_set_xalign(GTK_LABEL(bd->masks_refine_name_label), 0.0f);
    gtk_label_set_ellipsize(GTK_LABEL(bd->masks_refine_name_label), PANGO_ELLIPSIZE_END);
    gtk_widget_set_hexpand(bd->masks_refine_name_label, TRUE);
    dt_gui_add_class(bd->masks_refine_name_label, "mask-refine-header-name");

    dt_gui_box_add(inner_header_row, bd->masks_refine_icon_box, dt_gui_expand(bd->masks_refine_name_label));

    // Actions on the right of the inner header: [reset] [bypass toggle]
    bd->masks_refine_reset_btn = dtgtk_button_new(dtgtk_cairo_paint_reset, 0, NULL);
    gtk_widget_set_tooltip_text(bd->masks_refine_reset_btn,
                                _("reset the refinement of the current target"));
    g_signal_connect(G_OBJECT(bd->masks_refine_reset_btn), "clicked",
                     G_CALLBACK(_refine_reset_clicked), bd);
    gtk_widget_set_no_show_all(bd->masks_refine_reset_btn, TRUE);
    gtk_widget_set_visible(bd->masks_refine_reset_btn, FALSE);
    gtk_box_pack_end(GTK_BOX(inner_header_row), bd->masks_refine_reset_btn, FALSE, FALSE,
                     0);

    bd->masks_refine_bypass_btn =
      dtgtk_togglebutton_new(dtgtk_cairo_paint_eye_toggle, 0, NULL);
    dt_gui_add_class(bd->masks_refine_bypass_btn, "mask-refine-bypass-btn");
    gtk_widget_set_tooltip_text(
      bd->masks_refine_bypass_btn,
      _("temporarily disable the effect of refinements for this target"));
    g_signal_connect(G_OBJECT(bd->masks_refine_bypass_btn), "toggled",
                     G_CALLBACK(_refine_bypass_toggled), module);
    gtk_box_pack_end(GTK_BOX(inner_header_row), bd->masks_refine_bypass_btn, FALSE, FALSE,
                     0);

    bd->masks_refine_scope_kind = REFINE_SCOPE_GLOBAL;
    bd->masks_refine_scope_formid = INVALID_MASKID;

    // relocatable_box holds the "blend mask" header (gbox) plus everything
    // below it, and is the unit that _masks_flexi_relocate() moves between
    // iopw (embedded, the default) and a flexi masks panel host (utility lib
    // or separate grid panel) -- the header travels together with the rest
    // of the content, not left behind. gbox is packed directly here (not
    // inside blend_box below) so it stays visible even while the mask is
    // off -- it's the only way back on.
    bd->relocatable_box = GTK_BOX(dt_gui_vbox());
    dt_gui_box_add(iopw, GTK_WIDGET(bd->relocatable_box));
    dt_gui_box_add(bd->relocatable_box, gbox);
    // ...and everything below the header goes into masks_panel_body, one
    // wrapper the embedded position can fold away as a unit without
    // disturbing the mode-driven visibility of what is inside it (see its
    // field comment). Nothing else changes: mask_panel, which the rest of
    // this function fills, just points at the body instead of at
    // relocatable_box itself.
    bd->masks_panel_body = GTK_BOX(dt_gui_vbox());
    dt_gui_box_add(bd->relocatable_box, GTK_WIDGET(bd->masks_panel_body));
    GtkWidget *mask_panel = GTK_WIDGET(bd->masks_panel_body);

    GtkWidget *box = dt_gui_vbox();
    bd->blend_box = GTK_BOX(dt_gui_vbox(
      dt_gui_hbox(dt_gui_expand(bd->blend_modes_combo), bd->blend_modes_blend_order),
      bd->blend_mode_parameter_slider, opacity_box));
    _add_wrapped_box(box, bd->blend_box, NULL);

    dt_gui_box_add(mask_panel, box);
    dt_iop_gui_init_masks(mask_panel, module);

    // "shape properties in subpanel": a collapsible like the refinements', its
    // content filled from the selection (see _props_panel_sync). Built always,
    // shown only while the option is on (see _blendop_masks_mode_callback)
    {
      GtkWidget *head = dt_gui_hbox();
      gtk_box_set_spacing(GTK_BOX(head), DT_BAUHAUS_SPACE);
      dt_gui_add_class(head, "dt_section_expander");
      dt_gui_add_class(head, "mask-refine-section-expander");
      GtkWidget *label = dt_ui_section_label_new(_("shape properties"));
      gtk_widget_set_tooltip_text(
        label, _("the properties of the selected shape, or the creation controls of a"
                 " shape being drawn. empty while anything but a shape is selected."));
      GtkWidget *label_evb = gtk_event_box_new();
      gtk_container_add(GTK_CONTAINER(label_evb), label);
      dt_gui_connect_click(label_evb, _props_panel_header_clicked, NULL, bd);
      bd->props_panel_toggle_btn =
        dtgtk_togglebutton_new(dtgtk_cairo_paint_solid_arrow, CPF_DIRECTION_DOWN, NULL);
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->props_panel_toggle_btn), TRUE);
      dt_gui_add_class(bd->props_panel_toggle_btn, "dt_ignore_fg_state");
      dt_gui_add_class(bd->props_panel_toggle_btn, "dt_transparent_background");
      gtk_widget_set_tooltip_text(bd->props_panel_toggle_btn,
                                  _("toggle shape properties section"));
      g_signal_connect(G_OBJECT(bd->props_panel_toggle_btn), "toggled",
                       G_CALLBACK(_props_panel_toggled), bd);
      dt_gui_box_add(head, dt_gui_expand(label_evb));
      gtk_box_pack_end(GTK_BOX(head), bd->props_panel_toggle_btn, FALSE, FALSE, 0);

      bd->props_panel_content = dt_gui_vbox();
      gtk_widget_set_name(bd->props_panel_content, "collapsible");
      dt_gui_add_class(bd->props_panel_content, "mask-props-panel");
      bd->props_panel_expander = dtgtk_expander_new(head, bd->props_panel_content);
      dtgtk_expander_set_expanded(DTGTK_EXPANDER(bd->props_panel_expander), TRUE);
      gtk_widget_set_name(bd->props_panel_expander, "collapse-block");
      bd->props_panel_formid = INVALID_MASKID;
      bd->props_panel_box = GTK_BOX(dt_gui_vbox(bd->props_panel_expander));
      _add_wrapped_box(mask_panel, bd->props_panel_box, "masks_drawn");
    }

    bd->masks_refine_sliders_box = GTK_BOX(
      dt_gui_vbox(inner_header_row, bd->details_slider, bd->masks_feathering_guide_combo,
                  bd->feathering_radius_slider, bd->blur_radius_slider,
                  bd->brightness_slider, bd->contrast_slider));
    gtk_widget_set_name(GTK_WIDGET(bd->masks_refine_sliders_box), "collapsible");

    bd->masks_refine_expander =
      dtgtk_expander_new(destdisp_head, GTK_WIDGET(bd->masks_refine_sliders_box));
    dtgtk_expander_set_expanded(DTGTK_EXPANDER(bd->masks_refine_expander), TRUE);
    gtk_widget_set_name(bd->masks_refine_expander, "collapse-block");

    bd->refine_box = GTK_BOX(dt_gui_vbox(bd->masks_refine_expander));
    _add_wrapped_box(mask_panel, bd->refine_box, "masks_refinement");

    // the standalone "element properties" panel that used to live here is
    // gone -- per-shape/raster/group/parametric properties are now inline
    // expanders on each row instead (see _build_props_row_editor /
    // _make_props_row_toggle, wired from _make_shape_row and the group-header
    // block in _build_masks_list).

    gtk_widget_set_name(GTK_WIDGET(iopw), "blending-wrapper");

    // masks_panel_body's own visibility is the embedded collapse state, so it
    // must not be reset by an ancestor's show_all -- and the module expander
    // does exactly one right after this function returns (see
    // dt_iop_gui_set_expander). Same show-then-no_show_all sequencing as the
    // other collapsible wrappers here: every child is shown once (as that
    // expander-level show_all would have done anyway) before the wrapper
    // opts out of later ones.
    gtk_widget_show_all(GTK_WIDGET(bd->masks_panel_body));
    gtk_widget_set_no_show_all(GTK_WIDGET(bd->masks_panel_body), TRUE);

    bd->blend_inited = TRUE;

    ++darktable.bauhaus->skip_accel;
    DT_LEAVE_GUI_UPDATE();
  }
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
