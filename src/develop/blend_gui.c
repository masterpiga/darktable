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
static void _queue_masks_list_rebuild(dt_iop_module_t *module);
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
// shown state. Used to share widgets between the classic and flexi mask
// layouts.
void _reparent_into(GtkWidget *w,
                    GtkWidget *parent,
                    const gboolean at_end,
                    const gboolean expand)
{
  if(!w || !parent) return;
  GtkWidget *cur = gtk_widget_get_parent(w);
  if(cur == parent) return;

  g_object_ref(w);
  if(cur) gtk_container_remove(GTK_CONTAINER(cur), w);

  if(at_end)
    gtk_box_pack_end(GTK_BOX(parent), w, expand, expand, 0);
  else
    gtk_box_pack_start(GTK_BOX(parent), w, expand, expand, 0);
  g_object_unref(w);
}

// masks_toolbar (see its field comment in blend.h) is a plain, fixed
// two-row layout -- no dynamic wrap/reflow. Several dynamic approaches
// (GtkFlowBox, destroy-and-rebuild rows, per-widget reflow driven by
// "size-allocate") were each tried and rejected: GtkFlowBox's row/column
// space-distribution model spaced items apart instead of packing them
// tightly; destroying and recreating row GtkBoxes from inside a
// size-allocate handler raced with GTK's own layout pass and left the
// toolbar blank; and even a careful reflow-in-place scheme left several
// icon-drawn buttons (togglebuttons/dtgtk buttons using a custom cairo
// paint function, as opposed to plain-text GtkButtons) invisible until an
// unrelated event forced a redraw, for reasons that didn't resolve after
// several rounds of instrumentation. Row 1: add-group | shape buttons
// (masks_shapes_box) | add-raster. Row 2: parametric channel buttons
// (masks_param_channels_box) | import/reuse. If the panel is made
// extremely narrow, a row can clip -- that's preferable to any of the
// above failure modes.
static void _masks_toolbar_place_shapes_box(dt_iop_gui_blend_data_t *bd)
{
  _reparent_into(bd->masks_shapes_box, bd->masks_toolbar_row1, FALSE, FALSE);
  // slot 2: add-group(0) stretch(1) [shapes_box] stretch(3) raster(4)
  gtk_box_reorder_child(GTK_BOX(bd->masks_toolbar_row1), bd->masks_shapes_box, 2);
}

// an expanding, zero-content spacer: grows with the box so button clusters
// stay apart proportionally to the panel's width instead of hugging the left
static void _toolbar_pack_stretch(GtkWidget *box)
{
  GtkWidget *stretch = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
  gtk_widget_show(stretch);
  gtk_box_pack_start(GTK_BOX(box), stretch, TRUE, TRUE, 0);
}

// defined much further down (grouping shape rows / naming clusters); forward
// declared here so the import menu can group its "existing shape" entries by
// kind the same way the mask list clusters same-kind elements.
static guint _form_kind(const dt_masks_form_t *form);
static const char *_kind_name(const guint kind, const gboolean plural);
// defined further up (module.c-adjacent helpers); forward declared here so
// the import menu can look up which module (if any) currently uses a form.
void _build_masks_list(dt_iop_module_t *module);

// a picked menu entry just replays it on the (permanently hidden, headless)
// masks_combo: dt_bauhaus_combobox_set fires "value-changed" exactly as a
// real click on the combo's own popup would, so dt_masks_iop_value_changed_callback
// (connected once, in dt_iop_gui_init_masks) handles it completely unchanged.
static void _masks_import_pick(GtkMenuItem *item, gpointer user_data)
{
  dt_iop_gui_blend_data_t *bd = (dt_iop_gui_blend_data_t *)user_data;
  const int idx = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(item), "idx"));
  dt_bauhaus_combobox_set(bd->masks_combo, idx);
}

// build a single "idx"-carrying, _masks_import_pick-wired menu item
static GtkWidget *
_masks_import_menu_item(const char *label, const int idx, dt_iop_gui_blend_data_t *bd)
{
  GtkWidget *it = gtk_menu_item_new_with_label(label ? label : "");
  g_object_set_data(G_OBJECT(it), "idx", GINT_TO_POINTER(idx));
  g_signal_connect(G_OBJECT(it), "activate", G_CALLBACK(_masks_import_pick), bd);
  return it;
}

// marks a form-kind bucket holding whole other-module mask groups (imported
// as one composite "shape"), distinct from every real _form_kind() bit
#define _IMPORT_KIND_GROUP ((guint) - 1)
#define _IMPORT_MAX_KIND_BUCKETS 16

// find (or create, appending to menu) the submenu for a given shape kind
static GtkWidget *_masks_import_kind_bucket(
  GtkWidget *menu, guint *kinds, GtkWidget **submenus, int *n_buckets, const guint kind)
{
  for(int k = 0; k < *n_buckets; k++)
    if(kinds[k] == kind) return submenus[k];
  if(*n_buckets >= _IMPORT_MAX_KIND_BUCKETS) return NULL;

  GtkWidget *sub = gtk_menu_new();
  kinds[*n_buckets] = kind;
  submenus[*n_buckets] = sub;
  (*n_buckets)++;

  GtkWidget *it = gtk_menu_item_new_with_label(
    kind == _IMPORT_KIND_GROUP ? _("groups") : _kind_name(kind, TRUE));
  gtk_menu_item_set_submenu(GTK_MENU_ITEM(it), sub);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), it);
  return sub;
}

#define _IMPORT_MAX_MODULE_BUCKETS 32

// which module (if any) currently uses this form in its own mask group --
// used to group the import menu's "by source module" view. A form can only
// ever be a member of one group at a time in this UI (dt_masks_iop_combo_populate's
// own "existing shape" list already only offers forms unused by the *current*
// module, not forms unused by everyone), so the first match is the only one.
static dt_iop_module_t *_masks_import_form_owner(const dt_mask_id_t formid)
{
  for(GList *iter = darktable.develop->iop; iter; iter = g_list_next(iter))
  {
    dt_iop_module_t *m = iter->data;
    if(!(m->flags() & IOP_FLAGS_SUPPORTS_BLENDING) || (m->flags() & IOP_FLAGS_NO_MASKS))
      continue;
    dt_masks_form_t *grp = _module_mask_group(m);
    if(grp && _group_point(grp, formid)) return m;
  }
  return NULL;
}

// find (or create, appending to menu) the submenu for a given owning module
// (NULL = not currently used by any module, but still importable)
static GtkWidget *_masks_import_module_bucket(GtkWidget *menu,
                                              dt_iop_module_t **owners,
                                              GtkWidget **submenus,
                                              int *n_buckets,
                                              dt_iop_module_t *owner)
{
  for(int k = 0; k < *n_buckets; k++)
    if(owners[k] == owner) return submenus[k];
  if(*n_buckets >= _IMPORT_MAX_MODULE_BUCKETS) return NULL;

  GtkWidget *sub = gtk_menu_new();
  owners[*n_buckets] = owner;
  submenus[*n_buckets] = sub;
  (*n_buckets)++;

  gchar *label =
    owner ? dt_history_item_get_name(owner) : g_strdup(_("not currently used"));
  GtkWidget *it = gtk_menu_item_new_with_label(label);
  g_free(label);
  gtk_menu_item_set_submenu(GTK_MENU_ITEM(it), sub);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), it);
  return sub;
}

// removing a shape from a module's own group only detaches it from that
// group (see dt_masks_form_remove's grp != NULL branch in masks.c) -- it
// stays in darktable.develop->forms, unused, until something purges it. That
// purge already existed (dt_masks_cleanup_unused, wired to "delete unused
// shapes" in the classic mask manager panel's right-click menu); this just
// offers the same action from the flexi import menu, since that is where a
// user is now more likely to notice the clutter (it is exactly what ends up
// in the "not currently used" bucket of "by source module").
static void _masks_import_cleanup_unused(GtkMenuItem *item, gpointer user_data)
{
  dt_iop_module_t *module = (dt_iop_module_t *)user_data;
  dt_masks_cleanup_unused(darktable.develop);
  dt_control_log(_("unused shapes removed"));
  _build_masks_list(module);
}

// flexi's "import shape" trigger: like the add-group button, a single click
// shows the choices immediately as a plain popup menu -- no combobox is ever
// shown on screen. masks_combo itself stays hidden permanently and is used
// purely as a headless data source: dt_masks_iop_combo_populate (the same
// function the combo would call on its own popup open) fills its entries/ids,
// which are then just regrouped instead of shown as one flat list -- flat
// became unwieldy with more than a handful of entries (every shape in the
// image, plus one entry per other module in "use same shapes as"). Existing
// shapes get two parallel groupings, each just a different lens on the same
// entries -- "by source module" (which module currently uses each shape, or
// "not currently used") and "by type" (matching how the mask list itself
// clusters same-kind elements, see _pack_group_elements / _form_kind) -- so
// the user can navigate whichever way they already have in mind. Raster
// forms are dropped entirely: raster elements have their own dedicated
// add-raster button (with its own, more precise upstream-module picker, see
// _masks_raster_add_press), so listing them here too would just be the same
// targets reachable two different, inconsistently-named ways.
static gboolean
_masks_import_btn_press(GtkWidget *btn, GdkEventButton *ev, dt_iop_module_t *module)
{
  if(ev->button != GDK_BUTTON_PRIMARY) return FALSE;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd || !bd->masks_combo) return FALSE;

  dt_masks_iop_combo_populate(bd->masks_combo, &module);

  const int n = dt_bauhaus_combobox_length(bd->masks_combo);
  GtkWidget *by_type_menu = gtk_menu_new();
  GtkWidget *by_module_menu = gtk_menu_new();
  GtkWidget *reuse_menu = gtk_menu_new();
  guint kinds[_IMPORT_MAX_KIND_BUCKETS];
  GtkWidget *kind_submenus[_IMPORT_MAX_KIND_BUCKETS];
  int n_kinds = 0;
  dt_iop_module_t *owners[_IMPORT_MAX_MODULE_BUCKETS];
  GtkWidget *owner_submenus[_IMPORT_MAX_MODULE_BUCKETS];
  int n_owners = 0;
  int n_existing = 0, n_reuse = 0;

  // entry 0 is the fixed "import shape" placeholder itself (a permanent
  // no-op, see dt_masks_iop_value_changed_callback); every other entry with
  // id 0 is one of dt_masks_iop_combo_populate's own section dividers -- we
  // rebuild that grouping ourselves as submenus, so both kinds are skipped
  // here on the strength of the id's sign alone (positive = existing shape
  // formid, negative = -1*iop-index-1 for "use same shapes as").
  for(int i = 1; i < n; i++)
  {
    const int id = bd->masks_combo_ids ? bd->masks_combo_ids[i] : 0;
    if(id == 0) continue;
    const char *label = dt_bauhaus_combobox_get_entry(bd->masks_combo, i);

    if(id > 0)
    {
      const dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, id);
      if(form && (form->type & DT_MASKS_RASTER)) continue;

      const guint kind =
        (form && (form->type & DT_MASKS_GROUP)) ? _IMPORT_KIND_GROUP : _form_kind(form);
      GtkWidget *type_bucket =
        _masks_import_kind_bucket(by_type_menu, kinds, kind_submenus, &n_kinds, kind);
      if(type_bucket)
        gtk_menu_shell_append(GTK_MENU_SHELL(type_bucket),
                              _masks_import_menu_item(label, i, bd));

      dt_iop_module_t *owner = _masks_import_form_owner(id);
      GtkWidget *module_bucket = _masks_import_module_bucket(
        by_module_menu, owners, owner_submenus, &n_owners, owner);
      if(module_bucket)
        gtk_menu_shell_append(GTK_MENU_SHELL(module_bucket),
                              _masks_import_menu_item(label, i, bd));

      n_existing++;
    }
    else
    {
      gtk_menu_shell_append(GTK_MENU_SHELL(reuse_menu),
                            _masks_import_menu_item(label, i, bd));
      n_reuse++;
    }
  }

  GtkWidget *menu = gtk_menu_new();
  if(n_existing > 0)
  {
    GtkWidget *it_module =
      gtk_menu_item_new_with_label(_("add existing shape by source module"));
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(it_module), by_module_menu);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), it_module);

    GtkWidget *it_type = gtk_menu_item_new_with_label(_("add existing shape by type"));
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(it_type), by_type_menu);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), it_type);
  }
  else
  {
    gtk_widget_destroy(by_module_menu);
    gtk_widget_destroy(by_type_menu);
  }

  if(n_reuse > 0)
  {
    GtkWidget *it = gtk_menu_item_new_with_label(_("use same shapes as"));
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(it), reuse_menu);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), it);
  }
  else
    gtk_widget_destroy(reuse_menu);

  if(n_existing == 0 && n_reuse == 0)
  {
    GtkWidget *it = gtk_menu_item_new_with_label(_("nothing to import"));
    gtk_widget_set_sensitive(it, FALSE);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), it);
  }

  gtk_menu_shell_append(GTK_MENU_SHELL(menu), gtk_separator_menu_item_new());
  GtkWidget *cleanup_it = gtk_menu_item_new_with_label(_("clean up unused shapes"));
  gtk_widget_set_tooltip_text(cleanup_it,
                              _("remove every shape that no module currently uses\n"
                                "(the \"not currently used\" entries above)"));
  g_signal_connect(G_OBJECT(cleanup_it), "activate",
                   G_CALLBACK(_masks_import_cleanup_unused), module);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), cleanup_it);

  gtk_widget_show_all(menu);
  gtk_menu_popup_at_pointer(GTK_MENU(menu), (GdkEvent *)ev);
  return TRUE;
}

// apply the mask toolbar layout for the current mode. Classic restores the master
// two-row toolbar: combo row [import combo][invert], shapes row [edit][shapes].
// Flexi is compact: masks_toolbar takes over every "add an element" action
// (see its field comment in blend.h), while "edit on canvas" and the
// whole-mask "invert" toggle move up into the "mask elements" header.
// masks_combo stays wherever it already is (masks_combo_row) either way --
// in flexi it is a headless data source for masks_import_btn's popup menu
// and is never shown, so it does not need a visible home there; hiding
// masks_combo_row hides it along with everything else in that row. Every
// other shared widget is simply re-homed, so neither layout duplicates
// state. Called on every blending update (idempotent).
static void _masks_apply_layout(dt_iop_gui_blend_data_t *bd, const gboolean flexi)
{
  if(!bd->masks_combo_row || !bd->masks_shapes_row || !bd->masks_toolbar
     || !bd->masks_shapes_box)
    return;
  if(flexi)
  {
    _masks_toolbar_place_shapes_box(bd);
    // "edit on canvas" sits right after the "mask elements" label, with "invert"
    // right next to it -- both well clear of "reset" (packed separately, at the
    // header's far right), so a mis-click can't land on "reset" by mistake.
    if(bd->masks_groups_header)
    {
      _reparent_into(bd->masks_edit, bd->masks_groups_header, FALSE, FALSE);
      _reparent_into(bd->masks_polarity, bd->masks_groups_header, FALSE, FALSE);
    }
    gtk_widget_set_visible(bd->masks_toolbar, TRUE);
    gtk_widget_set_visible(bd->masks_combo_row, FALSE);
    gtk_widget_set_visible(bd->masks_shapes_row, FALSE);
  }
  else
  {
    _reparent_into(bd->masks_combo, bd->masks_combo_row, FALSE, TRUE);
    _reparent_into(bd->masks_polarity, bd->masks_combo_row, TRUE, FALSE);
    _reparent_into(bd->masks_edit, bd->masks_shapes_row, FALSE, FALSE);
    _reparent_into(bd->masks_shapes_box, bd->masks_shapes_row, FALSE, FALSE);
    // keep "edit" leftmost, the shapes box right of it
    gtk_box_reorder_child(GTK_BOX(bd->masks_shapes_row), bd->masks_edit, 0);
    gtk_widget_set_visible(bd->masks_toolbar, FALSE);
    gtk_widget_set_visible(bd->masks_combo_row, TRUE);
    gtk_widget_set_visible(bd->masks_shapes_row, TRUE);
  }
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
  // and place the precise-value popup (see _param_row_slider_precise_place)
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
  GtkWidget *input_bypass_btn;
  GtkWidget *output_lbl;
  GtkWidget *output_slot;
  GtkWidget *output_bypass_btn;
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
static void _empty_groups_clear(dt_iop_gui_blend_data_t *bd);
static void _update_add_target_sensitivity(dt_iop_module_t *module);
static void _update_refine_sensitivity(dt_iop_module_t *module);
static void _set_group_target(dt_iop_module_t *module, const dt_mask_id_t cid);
static void _set_form_target_ext(dt_iop_module_t *module,
                                 const dt_mask_id_t id,
                                 const gboolean auto_expand);
static void _set_form_target(dt_iop_module_t *module, const dt_mask_id_t id);
int _op_index_for_state(const int state);
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
static const char *_op_name_for_state(const int state);
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
// if removing `src` would empty its group, build an empty-group placeholder (same
// operator/screen/anchor) so the group persists instead of vanishing; else NULL.
// Must be called BEFORE the move (reads the pre-move layout). Defined with the
// empty-group machinery further down; forward-declared here via the struct tag so
// the single-shape move handlers above it can use it.
struct dt_masks_empty_group_t *_capture_emptied_group(dt_masks_form_t *grp,
                                                             const dt_mask_id_t src);

static void _blendop_masks_mode_callback(const dt_develop_mask_mode_t mask_mode,
                                         dt_iop_gui_blend_data_t *data)
{
  dt_develop_blend_params_t *bp = data->module->blend_params;
  if(bp->mask_mode != mask_mode)
    dt_print(DT_DEBUG_MASKS,
             "[masks] _blendop_masks_mode_callback '%s': mask_mode 0x%x->0x%x",
             data->module->op, bp->mask_mode, mask_mode);
  bp->mask_mode = mask_mode;

  const gboolean mask_enabled = mask_mode & DEVELOP_MASK_ENABLED;
  const gboolean mode_raster = mask_mode & DEVELOP_MASK_RASTER;
  const gboolean mode_drawn = mask_mode & DEVELOP_MASK_MASK;
  const gboolean mode_flexi = mask_mode & DEVELOP_MASK_FLEXI;
  const gboolean mode_parametric = mask_mode & DEVELOP_MASK_CONDITIONAL;
  // flexi reuses the drawn-group toolbar/renderer, so the drawn-mask panel and
  // refinement controls appear for it too.
  const gboolean mode_drawn_or_flexi = mode_drawn || mode_flexi;

  _box_set_visible(data->blend_box, mask_enabled);

  if(data->masks_blend_header)
  {
    if(mask_enabled)
      dt_gui_add_class(data->masks_blend_header, "mask-enabled");
    else
      dt_gui_remove_class(data->masks_blend_header, "mask-enabled");
  }

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
      gtk_widget_hide(GTK_WIDGET(data->showmask));
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->suppress), FALSE);
      gtk_widget_hide(GTK_WIDGET(data->suppress));

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
    else
    {
      gtk_widget_show(GTK_WIDGET(data->showmask));
      gtk_widget_show(GTK_WIDGET(data->suppress));
    }

    _box_set_visible(data->refine_box, TRUE);
  }
  else
  {
    _box_set_visible(data->refine_box, data->raster_inited && mode_raster);
  }

  if(data->masks_inited && mode_drawn_or_flexi)
  {
    // section caption reflects the mode: flexi drops the label (the combo value
    // "N shapes used" already says enough); classic keeps "drawn mask"
    dt_bauhaus_widget_set_label(data->masks_combo, N_("blend"),
                                mode_flexi ? "" : N_("drawn mask"));
    // flexi-only widgets: new-shape operator selector, add-parametric button,
    // and the per-shape composition list. classic drawn mask keeps the vanilla
    // toolbar.
    if(data->masks_reset_mask_btn)
      gtk_widget_set_visible(data->masks_reset_mask_btn, mode_flexi);
    if(data->masks_param_channels_box)
      gtk_widget_set_visible(data->masks_param_channels_box,
                             mode_flexi && data->blendif_support);
    if(data->masks_groups_header)
      gtk_widget_set_visible(data->masks_groups_header, mode_flexi);
    _masks_apply_layout(data, mode_flexi);
    gtk_widget_set_visible(GTK_WIDGET(data->masks_list_box), mode_flexi);
    if(mode_flexi) _build_masks_list(data->module);
    _box_set_visible(data->masks_box, TRUE);
  }
  else if(data->masks_inited)
  {
    for(int n = 0; n < DEVELOP_MASKS_NB_SHAPES; n++)
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->masks_shapes[n]), FALSE);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->masks_edit), FALSE);
    dt_masks_set_edit_mode(data->module, DT_MASKS_EDIT_OFF);
    // restore classic homes so invert / edit on canvas don't linger in the mask
    // elements header of the parametric/raster panels after leaving flexi
    _masks_apply_layout(data, FALSE);
    _box_set_visible(data->masks_box, FALSE);
  }
  else if(data->masks_support)
  {
    for(int n = 0; n < DEVELOP_MASKS_NB_SHAPES; n++)
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->masks_shapes[n]), FALSE);
    _box_set_visible(data->masks_box, FALSE);
  }

  _box_set_visible(data->raster_box, data->raster_inited && mode_raster);

  // leaving flexi: drop flexi-only selection/staging state
  if(!mode_flexi)
  {
    data->panel_selected_formid = INVALID_MASKID;
    data->panel_selected_group_cid = INVALID_MASKID;
    _empty_groups_clear(data);
    data->scaffold_seeded = FALSE;
    data->masks_selection_seeded = FALSE;
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
  gtk_widget_set_tooltip_text(toggle, enabled ? _("mask enabled\nclick to disable")
                                              : _("mask disabled\nclick to enable"));
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
  gtk_widget_set_visible(data->suppress, TRUE);

  const int pos = dt_conf_get_int("plugins/darkroom/blend/masks_panel_position");
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
  if(dt_gui_current_button(gesture) != GDK_BUTTON_PRIMARY) return;

  GtkWidget *button = dt_gui_get_widget(gesture);
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
    _blendop_masks_mode_callback(DEVELOP_MASK_DISABLED, data);
    dt_iop_add_remove_mask_indicator(module, FALSE);
    gtk_widget_set_visible(data->showmask, FALSE);
    gtk_widget_set_visible(data->suppress, FALSE);
    const int pos = dt_conf_get_int("plugins/darkroom/blend/masks_panel_position");
    if(pos == MASKS_PANEL_POS_UTILITY)
    {
      dt_lib_module_t *host = darktable.develop->proxy.masks_flexi_host.module;
      if(host) dt_lib_gui_set_expanded(host, FALSE);
    }
  }

  dt_control_hinter_message("");
}

static void _blendop_blendif_suppress_toggled(GtkGestureSingle *gesture,
                                                  gint n_press,
                                                  gdouble x,
                                                  gdouble y,
                                                  dt_iop_module_t *module)
{
  GtkWidget *togglebutton_w = dt_gui_get_widget(gesture);
  GtkToggleButton *togglebutton = GTK_TOGGLE_BUTTON(togglebutton_w);
  module->suppress_mask = !gtk_toggle_button_get_active(togglebutton);
  DT_GUARD_GUI_UPDATE();

  if(module->off) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(module->off), TRUE);
  dt_iop_request_focus(module);

  gtk_toggle_button_set_active(togglebutton, module->suppress_mask);

  dt_control_queue_redraw_widget(GTK_WIDGET(togglebutton));
  dt_iop_refresh_center(module);
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

  // set all shape buttons to inactive
  for(int n = 0; n < DEVELOP_MASKS_NB_SHAPES; n++)
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_shapes[n]), FALSE);

  // we want to be sure that the iop has focus
  dt_iop_request_focus(self);
  dt_iop_color_picker_reset(self, FALSE);
  bd->masks_shown = DT_MASKS_EDIT_FULL;
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(widget), TRUE);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_edit), FALSE);
  // we create the new form
  dt_masks_form_t *form = dt_masks_create(bd->masks_type[this]);
  dt_masks_change_form_gui(form);
  darktable.develop->form_gui->creation_module = self;
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

  dt_masks_form_t *grp = dt_masks_get_from_id(darktable.develop,
                                              self->blend_params->mask_id);
  if(grp && (grp->type & DT_MASKS_GROUP) && grp->points)
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

static void _blendop_masks_polarity_callback(GtkGestureSingle *gesture,
                                                 gint n_press,
                                                 gdouble x,
                                                 gdouble y,
                                                 dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();

  GtkWidget *togglebutton = dt_gui_get_widget(gesture);

  const int active = !gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(togglebutton));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(togglebutton), active);

  dt_develop_blend_params_t *bp = self->blend_params;

  if(active)
    bp->mask_combine |= DEVELOP_COMBINE_MASKS_POS;
  else
    bp->mask_combine &= ~DEVELOP_COMBINE_MASKS_POS;

  dt_dev_add_history_item(darktable.develop, self, TRUE);
  dt_control_queue_redraw_widget(togglebutton);
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
// silently dropping the forms is honest, so the switch is refused while any
// exist. Callers use this both to disable the menu entries and to explain why.
static int _module_parametric_form_count(dt_iop_module_t *module)
{
  const dt_masks_form_t *const grp = _module_mask_group(module);
  int n = 0;
  for(const GList *l = grp ? grp->points : NULL; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *const pt = l->data;
    const dt_masks_form_t *const f = dt_masks_get_from_id(darktable.develop, pt->formid);
    if(f && (f->type & DT_MASKS_PARAMETRIC)) n++;
  }
  return n;
}

// shared by every disabled blend-colorspace menu entry
static const char *_blend_cst_locked_tooltip(void)
{
  return _("the blend colorspace cannot be changed while the mask has parametric"
           " elements: each one stores the channels of the colorspace it was"
           " created in, and those channels do not exist in another one.\n"
           "remove the parametric elements first.");
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
    // the menu entries that lead here are already disabled in this case (see
    // _add_blend_colorspace_menu); this is the authority, so that any other
    // path -- a shortcut, a future caller -- cannot invalidate the forms
    // either. See _module_parametric_form_count for why the switch is refused.
    if(_module_parametric_form_count(module))
    {
      dt_control_log("%s", _blend_cst_locked_tooltip());
      return FALSE;
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

static void _blendif_select_colorspace(GtkMenuItem *menuitem,
                                       dt_iop_module_t *module)
{
  const dt_develop_blend_colorspace_t cst =
    GPOINTER_TO_INT(g_object_get_data(G_OBJECT(menuitem), "dt-blend-cst"));
  if(_blendif_change_blend_colorspace(module, cst))
  {
    gtk_widget_queue_draw(module->widget);
  }
}

static void _masks_opacity_sticky_toggled(GtkCheckMenuItem *mi, dt_iop_module_t *module)
{
  // the checkbox reads "sticky" (on = remember last opacity for new shapes),
  // the conf key is stored inverted (absent/FALSE = sticky, the default) so
  // it needs no preferences.xml entry -- see _new_shape_default_opacity in
  // masks.c, which is the actual place this is consumed.
  const gboolean not_sticky = !gtk_check_menu_item_get_active(mi);
  dt_conf_set_bool("plugins/darkroom/masks/opacity_not_sticky", not_sticky);
  if(not_sticky) dt_conf_set_float("plugins/darkroom/masks/opacity", 1.0f);
}

static void _masks_auto_expand_selected_toggled(GtkCheckMenuItem *mi,
                                                dt_iop_module_t *module)
{
  dt_conf_set_bool("plugins/darkroom/masks/auto_expand_selected",
                   gtk_check_menu_item_get_active(mi));
  // this option is read at row-build time from a conf key, not from anything
  // _masks_list_signature hashes (see _make_props_row_toggle) -- without
  // invalidating the cached signature here, toggling it would have no
  // visible effect until something unrelated next moved the signature.
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(bd) bd->masks_list_sig = DT_INVALID_HASH;
  _queue_masks_list_rebuild(module);
}

static void _masks_collapse_refinements_default_toggled(GtkCheckMenuItem *mi,
                                                        dt_iop_module_t *module)
{
  dt_conf_set_bool("plugins/darkroom/masks/collapse_refinements_default",
                   gtk_check_menu_item_get_active(mi));
}

// appends an "options" section directly to `menu` -- behavioural toggles
// for the flexi masks panel that don't fit the position/colorspace/presets
// sections above
static void _add_masks_panel_options_menu(GtkMenu *menu, dt_iop_module_t *module)
{
  GtkWidget *header = gtk_menu_item_new_with_label(_("options"));
  gtk_widget_set_sensitive(header, FALSE);
  gtk_widget_set_tooltip_text(header,
                              _("behavioural options for the flexi masks panel."));
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), header);

  GtkWidget *mi = gtk_check_menu_item_new_with_label(_("sticky opacity"));
  dt_gui_add_class(mi, "dt_transparent_background");
  gtk_widget_set_tooltip_text(
    mi, _("when enabled (default), a newly added shape starts at the opacity"
          " last used by any shape, so adjusting opacity once carries over to"
          " every shape you add afterwards.\n"
          "when disabled, every newly added shape starts at 100% opacity,"
          " regardless of what opacity was last used."));
  if(!dt_conf_get_bool("plugins/darkroom/masks/opacity_not_sticky"))
    gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(mi), TRUE);
  g_signal_connect(G_OBJECT(mi), "toggled", G_CALLBACK(_masks_opacity_sticky_toggled),
                   module);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);

  GtkWidget *ae = gtk_check_menu_item_new_with_label(_("auto-expand selected shape"));
  dt_gui_add_class(ae, "dt_transparent_background");
  gtk_widget_set_tooltip_text(
    ae, _("when enabled, the selected shape's expanded controls (size, hardness,"
          " etc.) are always shown, and every other shape's controls stay"
          " collapsed -- selecting a different shape expands it and collapses"
          " the previous one. only ever affects shapes, not groups.\n"
          "disabled by default."));
  if(dt_conf_get_bool("plugins/darkroom/masks/auto_expand_selected"))
    gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(ae), TRUE);
  g_signal_connect(G_OBJECT(ae), "toggled",
                   G_CALLBACK(_masks_auto_expand_selected_toggled), module);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), ae);

  GtkWidget *cr =
    gtk_check_menu_item_new_with_label(_("collapse refinements by default"));
  dt_gui_add_class(cr, "dt_transparent_background");
  gtk_widget_set_tooltip_text(
    cr, _("when enabled, newly selected masks, groups, and elements start with their"
          " refinements section collapsed by default.\n"
          "disabled by default."));
  if(dt_conf_get_bool("plugins/darkroom/masks/collapse_refinements_default"))
    gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(cr), TRUE);
  g_signal_connect(G_OBJECT(cr), "toggled",
                   G_CALLBACK(_masks_collapse_refinements_default_toggled), module);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), cr);
}

static void _blendif_options_callback(GtkButton *button,
                                      dt_iop_module_t *module)
{
  const dt_iop_gui_blend_data_t *bd = module->blend_data;

  if(!bd) return;

  // the blendif color-space section is only meaningful where blendif is supported;
  // the menu itself also opens on masks-only modules (for the mode-visibility items)
  const gboolean blendif_ok = bd->blendif_support && bd->blendif_inited;

  GtkWidget *mi;
  GtkMenu *menu = GTK_MENU(gtk_menu_new());
  // tracks whether anything has been appended yet, so a section separator is
  // only added between two sections, never as a leading/dangling line
  gboolean menu_has_items = FALSE;

  // add a section to switch blending color spaces
  const dt_develop_blend_colorspace_t module_cst =
    dt_develop_blend_default_module_blend_colorspace(module);
  const dt_develop_blend_colorspace_t module_blend_cst =
    module->blend_params->blend_cst;

  if(blendif_ok
     && (module_cst == DEVELOP_BLEND_CS_LAB || module_cst == DEVELOP_BLEND_CS_RGB_DISPLAY
         || module_cst == DEVELOP_BLEND_CS_RGB_SCENE))
  {
    // parametric elements pin the colorspace they were created in, so every
    // entry in this section is dead while any exist (see
    // _module_parametric_form_count). Disable them and say why on the section
    // header, which is the one widget here that stays hoverable -- GTK does not
    // deliver motion (hence tooltips) to insensitive widgets, so a tooltip on
    // the disabled entries themselves would never be seen.
    const gboolean cst_locked = _module_parametric_form_count(module) > 0;

    mi = gtk_menu_item_new_with_label(cst_locked ? _("blend colorspace (locked)")
                                                 : _("blend colorspace"));
    // the header is the only widget in this section left sensitive when locked,
    // purely so it can carry the explanation; it has no activate handler, so
    // clicking it just dismisses the menu.
    gtk_widget_set_sensitive(mi, cst_locked);
    if(cst_locked) gtk_widget_set_tooltip_text(mi, _blend_cst_locked_tooltip());
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);

    mi = gtk_menu_item_new_with_label(_("reset to default blend colorspace"));
    gtk_widget_set_sensitive(mi, !cst_locked);
    g_object_set_data_full(G_OBJECT(mi), "dt-blend-cst",
                           GINT_TO_POINTER(DEVELOP_BLEND_CS_NONE), NULL);
    g_signal_connect(G_OBJECT(mi), "activate",
                     G_CALLBACK(_blendif_select_colorspace), module);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);

    // only show Lab blending when the module is a Lab module to avoid
    // using it at the wrong place (Lab blending should not be
    // activated for RGB modules before colorin and after colorout)
    if(module_cst == DEVELOP_BLEND_CS_LAB)
    {
      mi = gtk_check_menu_item_new_with_label(_("Lab"));
      dt_gui_add_class(mi, "dt_transparent_background");
      if(module_blend_cst == DEVELOP_BLEND_CS_LAB)
      {
        gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(mi), TRUE);
        dt_gui_add_class(mi, "active_menu_item");
      }
      gtk_widget_set_sensitive(mi, !cst_locked);
      g_object_set_data_full(G_OBJECT(mi), "dt-blend-cst",
                             GINT_TO_POINTER(DEVELOP_BLEND_CS_LAB), NULL);
      g_signal_connect(G_OBJECT(mi), "activate",
                       G_CALLBACK(_blendif_select_colorspace), module);
      gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);
    }

    mi = gtk_check_menu_item_new_with_label(_("RGB (display)"));
    dt_gui_add_class(mi, "dt_transparent_background");
    if(module_blend_cst == DEVELOP_BLEND_CS_RGB_DISPLAY)
    {
      gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(mi), TRUE);
      dt_gui_add_class(mi, "active_menu_item");
    }
    gtk_widget_set_sensitive(mi, !cst_locked);
    g_object_set_data_full(G_OBJECT(mi), "dt-blend-cst",
                           GINT_TO_POINTER(DEVELOP_BLEND_CS_RGB_DISPLAY), NULL);
    g_signal_connect(G_OBJECT(mi), "activate",
                     G_CALLBACK(_blendif_select_colorspace), module);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);

    mi = gtk_check_menu_item_new_with_label(_("RGB (scene)"));
    dt_gui_add_class(mi, "dt_transparent_background");
    if(module_blend_cst == DEVELOP_BLEND_CS_RGB_SCENE)
    {
      gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(mi), TRUE);
      dt_gui_add_class(mi, "active_menu_item");
    }
    gtk_widget_set_sensitive(mi, !cst_locked);
    g_object_set_data_full(G_OBJECT(mi), "dt-blend-cst",
                           GINT_TO_POINTER(DEVELOP_BLEND_CS_RGB_SCENE), NULL);
    g_signal_connect(G_OBJECT(mi), "activate",
                     G_CALLBACK(_blendif_select_colorspace), module);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);

    menu_has_items = TRUE;
  }

  // "group layout presets" section (formerly a separate hamburger on the
  // "mask elements" header) -- only meaningful once the mask is actually
  // on, same as that button's old visibility
  if(bd->masks_support && (module->blend_params->mask_mode & DEVELOP_MASK_FLEXI))
  {
    if(menu_has_items)
      gtk_menu_shell_append(GTK_MENU_SHELL(menu), gtk_separator_menu_item_new());
    _add_flexi_presets_menu(menu, module);
    menu_has_items = TRUE;
  }

  // "options" section
  if(bd->masks_support)
  {
    if(menu_has_items)
      gtk_menu_shell_append(GTK_MENU_SHELL(menu), gtk_separator_menu_item_new());
    _add_masks_panel_options_menu(menu, module);
    menu_has_items = TRUE;
  }

  // "blend mask panel position" section
  if(bd->masks_support)
  {
    if(menu_has_items)
      gtk_menu_shell_append(GTK_MENU_SHELL(menu), gtk_separator_menu_item_new());
    _add_masks_panel_position_menu(menu, module);
  }

  dt_gui_menu_popup(menu,
                    GTK_WIDGET(button), GDK_GRAVITY_SOUTH_EAST, GDK_GRAVITY_NORTH_EAST);

  dtgtk_button_set_active(DTGTK_BUTTON(button), FALSE);
}

void dt_iop_gui_blend_masks_options_popup(GtkButton *button, gpointer user_data)
{
  dt_iop_module_t *module = darktable.develop->proxy.masks_flexi_host.hosted_module;
  if(module) _blendif_options_callback(button, module);
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

// activate channel/mask view
static void _blendop_blendif_channel_mask_view(GtkWidget *widget,
                                               dt_iop_module_t *module,
                                               const dt_dev_pixelpipe_display_mask_t mode)
{
  dt_dev_pixelpipe_display_mask_t new_request_mask_display =
    module->request_mask_display | mode;

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

  // only if something has changed: reprocess center view
  if(new_request_mask_display != module->request_mask_display)
  {
    module->request_mask_display = new_request_mask_display;
    dt_iop_refresh_center(module);
  }
}

// toggle channel/mask view
static void _blendop_blendif_channel_mask_view_toggle
  (GtkWidget *widget,
   dt_iop_module_t *module,
   const dt_dev_pixelpipe_display_mask_t mode)
{
  dt_iop_gui_blend_data_t *data = module->blend_data;

  dt_dev_pixelpipe_display_mask_t new_request_mask_display =
    module->request_mask_display & ~DT_DEV_PIXELPIPE_DISPLAY_STICKY;

  // toggle mode
  if(module->request_mask_display & mode)
    new_request_mask_display &= ~mode;
  else
    new_request_mask_display |= mode;

  dt_pthread_mutex_lock(&data->lock);
  if(new_request_mask_display & DT_DEV_PIXELPIPE_DISPLAY_STICKY)
    data->save_for_leave |= DT_DEV_PIXELPIPE_DISPLAY_STICKY;
  else
    data->save_for_leave &= ~DT_DEV_PIXELPIPE_DISPLAY_STICKY;
  dt_pthread_mutex_unlock(&data->lock);

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


// magic mode: if mouse cursor enters a gradient slider with shift
// and/or control pressed we enter channel display and/or mask display
// mode
static void _blendop_blendif_enter_cb(GtkEventControllerMotion *controller,
                                         double x, double y,
                                         dt_iop_module_t *module)
{
  if(dt_atomic_get_int(&darktable.gui->reset) != 0) return;

  GtkWidget *widget = dt_gui_get_widget(controller);
  dt_iop_gui_blend_data_t *data = module->blend_data;

  dt_dev_pixelpipe_display_mask_t mode = DT_DEV_PIXELPIPE_DISPLAY_NONE;

  const GdkModifierType state =
    dt_gui_get_current_event_state(GTK_EVENT_CONTROLLER(controller));
  {
    // depending on shift modifiers we activate channel and/or mask display
    if(dt_modifier_is(state, GDK_SHIFT_MASK | GDK_CONTROL_MASK))
    {
      mode = (DT_DEV_PIXELPIPE_DISPLAY_MASK | DT_DEV_PIXELPIPE_DISPLAY_CHANNEL);
    }
    else if(dt_modifier_is(state, GDK_SHIFT_MASK))
    {
      mode = DT_DEV_PIXELPIPE_DISPLAY_CHANNEL;
    }
    else if(dt_modifier_is(state, GDK_CONTROL_MASK))
    {
      mode = DT_DEV_PIXELPIPE_DISPLAY_MASK;
    }
  }

  dt_pthread_mutex_lock(&data->lock);
  if(mode && data->timeout_handle)
  {
    // purge any remaining timeout handlers
    g_source_remove(data->timeout_handle);
    data->timeout_handle = 0;
  }
  else if(!data->timeout_handle
          && !(data->save_for_leave & DT_DEV_PIXELPIPE_DISPLAY_STICKY))
  {
    // save request_mask_display to restore later
    data->save_for_leave =
      module->request_mask_display & ~DT_DEV_PIXELPIPE_DISPLAY_STICKY;
  }
  dt_pthread_mutex_unlock(&data->lock);

  _blendop_blendif_channel_mask_view(widget, module, mode);

  gtk_widget_grab_focus(widget);
}


// handler for delayed mask/channel display mode switch-off
static gboolean _blendop_blendif_leave_delayed(gpointer data)
{
  dt_iop_module_t *module = (dt_iop_module_t *)data;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  int reprocess = 0;

  dt_pthread_mutex_lock(&bd->lock);
  // restore saved request_mask_display and reprocess image
  if(bd->timeout_handle
     && (module->request_mask_display
         != (bd->save_for_leave & ~DT_DEV_PIXELPIPE_DISPLAY_STICKY)))
  {
    module->request_mask_display = bd->save_for_leave & ~DT_DEV_PIXELPIPE_DISPLAY_STICKY;
    reprocess = 1;
  }
  bd->timeout_handle = 0;
  dt_pthread_mutex_unlock(&bd->lock);

  if(reprocess)
    dt_iop_refresh_center(module);
  // return FALSE and thereby terminate the handler
  return FALSE;
}

// de-activate magic mode when leaving the gradient slider
static void _blendop_blendif_leave_cb(GtkEventControllerMotion *controller,
                                        dt_iop_module_t *module)
{
  if(dt_atomic_get_int(&darktable.gui->reset) != 0) return;

  dt_iop_gui_blend_data_t *data = module->blend_data;

  // do not immediately switch-off mask/channel display in case user
  // leaves gradient only briefly.  instead we activate a handler
  // function that gets triggered after some timeout
  dt_pthread_mutex_lock(&data->lock);
  if(!(module->request_mask_display & DT_DEV_PIXELPIPE_DISPLAY_STICKY)
     && !data->timeout_handle
     && (module->request_mask_display
         != (data->save_for_leave & ~DT_DEV_PIXELPIPE_DISPLAY_STICKY)))
      data->timeout_handle = g_timeout_add(1000, _blendop_blendif_leave_delayed, module);
  dt_pthread_mutex_unlock(&data->lock);
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
    case GDK_KEY_c:
      _blendop_blendif_channel_mask_view_toggle
        (widget, module, DT_DEV_PIXELPIPE_DISPLAY_CHANNEL);
      handled = TRUE;
      break;
    case GDK_KEY_C:
      _blendop_blendif_channel_mask_view_toggle
        (widget, module,
         DT_DEV_PIXELPIPE_DISPLAY_CHANNEL | DT_DEV_PIXELPIPE_DISPLAY_STICKY);
      handled = TRUE;
      break;
    case GDK_KEY_m:
    case GDK_KEY_M:
      _blendop_blendif_channel_mask_view_toggle
        (widget, module,
         DT_DEV_PIXELPIPE_DISPLAY_MASK);
      handled = TRUE;
  }

  if(handled)
    dt_iop_request_focus(module);

  return handled;
}


#define COLORSTOPS(gradient) sizeof(gradient) / sizeof(dt_iop_gui_blendif_colorstop_t), \
                             gradient

const dt_iop_gui_blendif_channel_t Lab_channels[]
    = { { N_("L"), N_("sliders for L channel"), 1.0f / 100.0f,
            COLORSTOPS(_gradient_L), TRUE, 0.0f,
          { DEVELOP_BLENDIF_L_in, DEVELOP_BLENDIF_L_out }, DT_DEV_PIXELPIPE_DISPLAY_L,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("lightness") },
        { N_("a"), N_("sliders for a channel"), 1.0f / 256.0f,
          COLORSTOPS(_gradient_a), TRUE, 0.0f,
          { DEVELOP_BLENDIF_A_in, DEVELOP_BLENDIF_A_out }, DT_DEV_PIXELPIPE_DISPLAY_a,
          _blendif_scale_print_ab, _blendop_blendif_disp_alternative_mag,
          N_("green/red") },
        { N_("b"), N_("sliders for b channel"), 1.0f / 256.0f,
          COLORSTOPS(_gradient_b), TRUE, 0.0f,
          { DEVELOP_BLENDIF_B_in, DEVELOP_BLENDIF_B_out }, DT_DEV_PIXELPIPE_DISPLAY_b,
          _blendif_scale_print_ab, _blendop_blendif_disp_alternative_mag,
          N_("blue/yellow") },
        { N_("C"), N_("sliders for chroma channel (of LCh)"), 1.0f / 100.0f,
          COLORSTOPS(_gradient_chroma),
          TRUE, 0.0f,
          { DEVELOP_BLENDIF_C_in, DEVELOP_BLENDIF_C_out }, DT_DEV_PIXELPIPE_DISPLAY_LCH_C,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("saturation") },
        { N_("h"), N_("sliders for hue channel (of LCh)"), 1.0f / 360.0f,
          COLORSTOPS(_gradient_LCh_hue),
          FALSE, 0.0f,
          { DEVELOP_BLENDIF_h_in, DEVELOP_BLENDIF_h_out }, DT_DEV_PIXELPIPE_DISPLAY_LCH_h,
          _blendif_scale_print_hue, NULL, N_("hue") },
        { NULL } };

const dt_iop_gui_blendif_channel_t rgb_channels[]
    = { { N_("g"), N_("sliders for gray value"), 1.0f / 255.0f,
            COLORSTOPS(_gradient_gray), TRUE, 0.0f,
          { DEVELOP_BLENDIF_GRAY_in, DEVELOP_BLENDIF_GRAY_out },
          DT_DEV_PIXELPIPE_DISPLAY_GRAY,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("gray") },
        { N_("R"), N_("sliders for red channel"), 1.0f / 255.0f,
          COLORSTOPS(_gradient_red), TRUE, 0.0f,
          { DEVELOP_BLENDIF_RED_in, DEVELOP_BLENDIF_RED_out },
          DT_DEV_PIXELPIPE_DISPLAY_R,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("red") },
        { N_("G"), N_("sliders for green channel"), 1.0f / 255.0f,
          COLORSTOPS(_gradient_green), TRUE, 0.0f,
          { DEVELOP_BLENDIF_GREEN_in, DEVELOP_BLENDIF_GREEN_out },
          DT_DEV_PIXELPIPE_DISPLAY_G,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("green") },
        { N_("B"), N_("sliders for blue channel"), 1.0f / 255.0f,
          COLORSTOPS(_gradient_blue), TRUE, 0.0f,
          { DEVELOP_BLENDIF_BLUE_in, DEVELOP_BLENDIF_BLUE_out },
          DT_DEV_PIXELPIPE_DISPLAY_B,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("blue") },
        { N_("H"), N_("sliders for hue channel (of HSL)"), 1.0f / 360.0f,
          COLORSTOPS(_gradient_HSL_hue),
          FALSE, 0.0f,
          { DEVELOP_BLENDIF_H_in, DEVELOP_BLENDIF_H_out },
          DT_DEV_PIXELPIPE_DISPLAY_HSL_H,
          _blendif_scale_print_hue, NULL,
          N_("hue") },
        { N_("S"), N_("sliders for chroma channel (of HSL)"), 1.0f / 100.0f,
          COLORSTOPS(_gradient_chroma),
          FALSE, 0.0f,
          { DEVELOP_BLENDIF_S_in, DEVELOP_BLENDIF_S_out },
          DT_DEV_PIXELPIPE_DISPLAY_HSL_S,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("chroma") },
        { N_("L"), N_("sliders for value channel (of HSL)"), 1.0f / 100.0f,
          COLORSTOPS(_gradient_gray),
          FALSE, 0.0f,
          { DEVELOP_BLENDIF_l_in, DEVELOP_BLENDIF_l_out },
          DT_DEV_PIXELPIPE_DISPLAY_HSL_l,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("luminance") },
        { NULL } };

const dt_iop_gui_blendif_channel_t rgbj_channels[]
    = { { N_("g"), N_("sliders for gray value"), 1.0f / 255.0f,
            COLORSTOPS(_gradient_gray), TRUE, 0.0f,
          { DEVELOP_BLENDIF_GRAY_in, DEVELOP_BLENDIF_GRAY_out },
          DT_DEV_PIXELPIPE_DISPLAY_GRAY,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("gray") },
        { N_("R"), N_("sliders for red channel"), 1.0f / 255.0f,
          COLORSTOPS(_gradient_red), TRUE, 0.0f,
          { DEVELOP_BLENDIF_RED_in, DEVELOP_BLENDIF_RED_out },
          DT_DEV_PIXELPIPE_DISPLAY_R,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("red") },
        { N_("G"), N_("sliders for green channel"), 1.0f / 255.0f,
          COLORSTOPS(_gradient_green), TRUE, 0.0f,
          { DEVELOP_BLENDIF_GREEN_in, DEVELOP_BLENDIF_GREEN_out },
          DT_DEV_PIXELPIPE_DISPLAY_G,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("green") },
        { N_("B"), N_("sliders for blue channel"), 1.0f / 255.0f,
          COLORSTOPS(_gradient_blue), TRUE, 0.0f,
          { DEVELOP_BLENDIF_BLUE_in, DEVELOP_BLENDIF_BLUE_out },
          DT_DEV_PIXELPIPE_DISPLAY_B,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("blue") },
        { N_("Jz"), N_("sliders for value channel (of JzCzhz)"), 1.0f / 100.0f,
          COLORSTOPS(_gradient_gray),
          TRUE, -6.64385619f, // cf. _blend_init_blendif_boost_parameters
          { DEVELOP_BLENDIF_Jz_in, DEVELOP_BLENDIF_Jz_out },
          DT_DEV_PIXELPIPE_DISPLAY_JzCzhz_Jz,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("luminance") },
        { N_("Cz"), N_("sliders for chroma channel (of JzCzhz)"), 1.0f / 100.0f,
          COLORSTOPS(_gradient_chroma),
          TRUE, -6.64385619f, // cf. _blend_init_blendif_boost_parameters
          { DEVELOP_BLENDIF_Cz_in, DEVELOP_BLENDIF_Cz_out },
          DT_DEV_PIXELPIPE_DISPLAY_JzCzhz_Cz,
          _blendif_scale_print_default, _blendop_blendif_disp_alternative_log,
          N_("chroma") },
        { N_("hz"), N_("sliders for hue channel (of JzCzhz)"), 1.0f / 360.0f,
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
  { N_("adjustment based on input received by this module:\n"
       "* range defined by upper markers: blend fully\n"
       "* range defined by lower markers: do not blend at all\n"
       "* range between adjacent upper/lower markers: blend gradually"),
    N_("adjustment based on unblended output of this module:\n"
       "* range defined by upper markers: blend fully\n"
       "* range defined by lower markers: do not blend at all\n"
       "* range between adjacent upper/lower markers: blend gradually") };

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
  dt_bauhaus_combobox_clear(bd->masks_combo);
  if(flexi)
  {
    // in flexi the combo is purely a shape importer: its entries import an
    // existing shape (or another module's shapes) into the selected group.
    dt_bauhaus_combobox_add(bd->masks_combo, _("import shape"));
  }
  else if(grp && (grp->type & DT_MASKS_GROUP) && grp->points)
  {
    char txt[512];
    const guint n = g_list_length(grp->points);
    snprintf(txt, sizeof(txt), ngettext("%d shape used", "%d shapes used", n), n);
    dt_bauhaus_combobox_add(bd->masks_combo, txt);
  }
  else
  {
    dt_bauhaus_combobox_add(bd->masks_combo, _("no mask used"));
    bd->masks_shown = DT_MASKS_EDIT_OFF;
    // reset the gui
    dt_masks_set_edit_mode(module, DT_MASKS_EDIT_OFF);
  }
  dt_bauhaus_combobox_set(bd->masks_combo, 0);

  if(bd->masks_support)
  {
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_edit),
                                 bd->masks_shown != DT_MASKS_EDIT_OFF);

    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_polarity),
                                 bp->mask_combine & DEVELOP_COMBINE_MASKS_POS);
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

  // classic mode keeps the import/shape combo always usable (the flexi rebuild
  // re-derives this from the group selection)
  if(!flexi) gtk_widget_set_sensitive(bd->masks_combo, TRUE);

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

// defined near the other group helpers (below _starts_group); declared here
// so _row_click_press (which comes first in the file) can call it directly
// instead of dt_masks_form_remove when deleting a run's last remaining member.
static void _group_reset_members(dt_iop_module_t *module, GList *fids, const int opstate);
// defined below (needs _group_point etc.); declared here so the group's
// "solo" menu item (_group_menu_toggle_solo) can call it directly.
static void _toggle_solo_group(dt_iop_module_t *module, const guint key, GList *members);
// defined below (needs _param_row_point etc.); declared here so
// _solo_badge_form_press (which comes first in the file) can call it
// directly to clear solo-edit from a click on its own status badge.
static void _toggle_soloedit(dt_iop_module_t *module, const dt_mask_id_t id);
static void _toggle_element_disable(dt_iop_module_t *module, const dt_mask_id_t id);

dt_masks_form_t *_module_mask_group(dt_iop_module_t *module)
{
  if(!module || !module->blend_params) return NULL;
  dt_masks_form_t *grp =
    dt_masks_get_from_id(darktable.develop, module->blend_params->mask_id);
  return (grp && (grp->type & DT_MASKS_GROUP)) ? grp : NULL;
}

dt_masks_point_group_t *_group_point(dt_masks_form_t *grp, const dt_mask_id_t id)
{
  for(GList *l = grp ? grp->points : NULL; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *q = l->data;
    if(q->formid == id) return q;
  }
  return NULL;
}

// a group's user-given name (masks v8), or NULL if it has none. Read off the
// cid member alone: renaming broadcasts the same text onto every member of
// the run (see _group_rename_commit), so any one of them -- cid included --
// reflects the whole group.
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
// REFINE_SCOPE_EMPTY_GROUP: a staged (member-less) group is selected. Same
// scope level as REFINE_SCOPE_GROUP, but the value lives in the empty group's
// own slot (dt_masks_empty_group_t.refinement) rather than in member points,
// and is adopted by the run once the group is realized.
enum
{
  REFINE_SCOPE_GLOBAL = 0,
  REFINE_SCOPE_ALL_SHAPES,
  REFINE_SCOPE_ELEMENT,
  REFINE_SCOPE_GROUP,
  REFINE_SCOPE_EMPTY_GROUP
};

// accessors for that slot. dt_masks_empty_group_t is defined much further down
// (with the rest of the empty-group machinery) and bd->selected_empty is a void*,
// so the refinement code up here reaches it through these instead of the struct.
static dt_masks_refinement_t _empty_group_refinement(const void *eg);
static void _empty_group_set_refinement(void *eg, const dt_masks_refinement_t *r);
static int _empty_group_op(const void *eg);
// eg's custom name (ctrl+click rename while still staged/empty), or NULL --
// same "same struct, reached through an opaque accessor" reason as the two
// above.
static const char *_empty_group_name(const void *eg);
// ordinal of an empty group (per-operator numbering shared with real groups)
struct dt_masks_empty_group_t;
static int _group_ordinal_any(dt_iop_module_t *module,
                              const dt_mask_id_t cid,
                              const struct dt_masks_empty_group_t *eg);

// enum values in the same order as dt_develop_feathering_guide_names[], so a
// combo index maps to the stored uint32 guide value (and back).
static const uint32_t _refine_guide_values[] = { DEVELOP_MASK_GUIDE_OUT_BEFORE_BLUR,
                                                 DEVELOP_MASK_GUIDE_IN_BEFORE_BLUR,
                                                 DEVELOP_MASK_GUIDE_OUT_AFTER_BLUR,
                                                 DEVELOP_MASK_GUIDE_IN_AFTER_BLUR };

// first drawn (non-parametric) form's point in the group, or NULL. Used to read
// back the "all shapes" refinement (every drawn point is kept in sync, so the
// first one is representative).
static dt_masks_point_group_t *_refine_first_drawn_point(dt_masks_form_t *grp)
{
  for(GList *l = grp ? grp->points : NULL; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = l->data;
    dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, pt->formid);
    if(form && !(form->type & DT_MASKS_PARAMETRIC)) return pt;
  }
  return NULL;
}

// a group point's effective operator for grouping: a missing operator (the base)
// reads as union, matching how the list folds runs.
dt_masks_state_t _eff_group_op(const int state)
{
  const dt_masks_state_t op = state & DT_MASKS_STATE_OP;
  return op ? op : DT_MASKS_STATE_UNION;
}

// first-class groups (see dt_masks_point_group_t.group_start). True if the point at
// list node `l` begins a new group. grp->points is ordered bottom-up, so l->prev is
// the point *below* l. A new group starts at the bottom point, at any point with
// group_start set, or -- for back-compat with edits made before the field existed --
// wherever the effective operator changes. Old edits have no break marked and never
// had two adjacent same-op groups, so detection is bit-identical for them.
gboolean _starts_group(GList *l)
{
  if(!l || !l->prev) return TRUE;
  const dt_masks_point_group_t *cur = l->data;
  const dt_masks_point_group_t *below = l->prev->data;
  if(cur->group_start) return TRUE;
  return _eff_group_op(cur->state) != _eff_group_op(below->state);
}

// is `fid` the only member of its run (maximal same-operator group)? Used when
// deleting a single shape (see _row_click_press) -- removing the last member
// of a run via dt_masks_form_remove would silently collapse the whole group
// (no placeholder left behind), so that case is routed through
// _group_reset_members instead, same as "shift+right-click the group header".
static gboolean
_group_sole_member(dt_masks_form_t *grp, const dt_mask_id_t fid, int *op_out)
{
  if(!grp) return FALSE;
  GList *l = grp->points;
  for(; l; l = g_list_next(l))
    if(((dt_masks_point_group_t *)l->data)->formid == fid) break;
  if(!l) return FALSE;

  GList *lo = l;
  while(!_starts_group(lo)) lo = lo->prev;
  if(op_out) *op_out = (int)_eff_group_op(((dt_masks_point_group_t *)lo->data)->state);

  for(GList *m = lo; m; m = g_list_next(m))
  {
    if(m != lo && _starts_group(m)) break;
    if(((dt_masks_point_group_t *)m->data)->formid != fid) return FALSE;
  }
  return TRUE;
}

// snapshot the current group partition as the list of head formids (bottom-up).
// Used to preserve which members form which group across an operator change that
// would otherwise merge same-op neighbours. Caller frees.
GList *_group_partition_heads(dt_masks_form_t *grp)
{
  GList *out = NULL;
  for(GList *l = grp ? grp->points : NULL; l; l = g_list_next(l))
    if(_starts_group(l))
      out =
        g_list_prepend(out, GINT_TO_POINTER(((dt_masks_point_group_t *)l->data)->formid));
  return g_list_reverse(out);
}

// re-stamp group_start so the partition described by `head_fids` survives, whatever
// the operators now are: every listed head (except the very bottom point, which
// cannot carry a break) gets the marker; every other point has it cleared.
static void _apply_partition_breaks(dt_masks_form_t *grp, GList *head_fids)
{
  gboolean first = TRUE;
  for(GList *l = grp ? grp->points : NULL; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = l->data;
    gboolean is_head = FALSE;
    for(GList *h = head_fids; h; h = g_list_next(h))
      if(GPOINTER_TO_INT(h->data) == pt->formid)
      {
        is_head = TRUE;
        break;
      }
    pt->group_start = (is_head && !first) ? 1 : 0;
    first = FALSE;
  }
}

// Like the head-formid snapshot above, but labels EVERY member with its group
// "key" = the formid of its run's bottom-most member (head). Used to preserve the
// partition across a single-shape move, where the moved shape may itself be a head
// (so the simple head-list snapshot would mis-track the group it leaves). Returns a
// GHashTable<formid,key>; caller destroys.
GHashTable *_group_keys_snapshot(dt_masks_form_t *grp)
{
  GHashTable *keys = g_hash_table_new(g_direct_hash, g_direct_equal);
  dt_mask_id_t key = INVALID_MASKID;
  for(GList *l = grp ? grp->points : NULL; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    if(_starts_group(l)) key = pt->formid; // a new run starts -> new key
    g_hash_table_insert(keys, GINT_TO_POINTER(pt->formid), GINT_TO_POINTER(key));
  }
  return keys;
}

// Re-stamp group_start from a key map: a point begins a group iff it is the bottom
// point or its key differs from the point below it. A member absent from the map
// inherits the key of the point below (so a freshly added shape merges into the
// group it sits on top of). Robust to a group's head having moved away.
void _group_keys_apply(dt_masks_form_t *grp, GHashTable *keys)
{
  dt_mask_id_t below = INVALID_MASKID;
  gboolean first = TRUE;
  for(GList *l = grp ? grp->points : NULL; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = l->data;
    const gpointer kp = g_hash_table_lookup(keys, GINT_TO_POINTER(pt->formid));
    const dt_mask_id_t key = kp ? GPOINTER_TO_INT(kp) : below;
    pt->group_start = (first || key == below) ? 0 : 1;
    below = key;
    first = FALSE;
  }
}

// formids of the contiguous same-operator run containing `sel` (i.e. the group as
// the list shows it). Caller frees the list.
GList *_selected_group_formids(dt_masks_form_t *grp, const dt_mask_id_t sel)
{
  if(!grp) return NULL;
  GList *node = NULL;
  for(GList *l = grp->points; l; l = g_list_next(l))
    if(((dt_masks_point_group_t *)l->data)->formid == sel)
    {
      node = l;
      break;
    }
  if(!node) return NULL;
  // walk down to this group's head (the bottom-most point that starts the group)
  GList *lo = node;
  while(!_starts_group(lo)) lo = lo->prev;
  // collect members upward until the next group begins
  GList *out = NULL;
  for(GList *l = lo; l; l = g_list_next(l))
  {
    if(l != lo && _starts_group(l)) break;
    out =
      g_list_prepend(out, GINT_TO_POINTER(((dt_masks_point_group_t *)l->data)->formid));
  }
  return out;
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
  else if(bd->masks_refine_scope_kind == REFINE_SCOPE_EMPTY_GROUP)
  {
    if(bd->selected_empty) r = _empty_group_refinement(bd->selected_empty);
  }
  else
  {
    dt_masks_form_t *grp = _module_mask_group(module);
    // ALL_SHAPES reads the first drawn point; GROUP reads its representative
    // member (a group's members are broadcast-synced, so any member reflects
    // the whole run); ELEMENT reads that one specific form directly
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
  else if(bd->masks_refine_scope_kind == REFINE_SCOPE_EMPTY_GROUP)
    // a staged group has no members and no id, so it is keyed by its own
    // address. It never reaches the renderer (see dt_masks_refine_bypass_commit).
    return (gpointer)bd->selected_empty;
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
static void _flexi_refine_follow_selection(dt_iop_gui_blend_data_t *bd)
{
  if(!bd || !bd->blend_inited || !bd->module) return;
  const gboolean flexi = bd->module->blend_params->mask_mode & DEVELOP_MASK_FLEXI;
  if(flexi && dt_is_valid_maskid(bd->panel_selected_formid))
  {
    bd->masks_refine_scope_kind = REFINE_SCOPE_ELEMENT;
    bd->masks_refine_scope_formid = bd->panel_selected_formid;
  }
  else if(flexi && dt_is_valid_maskid(bd->panel_selected_group_cid))
  {
    bd->masks_refine_scope_kind = REFINE_SCOPE_GROUP;
    bd->masks_refine_scope_formid = bd->panel_selected_group_cid;
  }
  else if(flexi && bd->selected_empty)
  {
    // a staged group with no members yet is still a selected *group*: it must
    // not fall through to global, or the sole empty group of a fresh/reset mask
    // would make "group" and "whole mask" scope indistinguishable (identical
    // caption, and deselecting it would look like a no-op).
    bd->masks_refine_scope_kind = REFINE_SCOPE_EMPTY_GROUP;
    bd->masks_refine_scope_formid = INVALID_MASKID;
  }
  else
  {
    bd->masks_refine_scope_kind = REFINE_SCOPE_GLOBAL;
    bd->masks_refine_scope_formid = INVALID_MASKID;
  }
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

  // a staged group has no members to write into: park the value on the empty
  // group itself. Nothing to persist (empty groups are UI-side state and render
  // nothing), and no history item -- the run adopts it on realize.
  if(bd->masks_refine_scope_kind == REFINE_SCOPE_EMPTY_GROUP)
  {
    if(!bd->selected_empty) return;
    dt_masks_refinement_t er = { 0 };
    _refine_read_controls(bd, &er);
    // a group's refinement, staged until the group has members to broadcast onto
    if(er.enabled) er.enabled = DT_MASKS_REFINE_GROUP;
    _empty_group_set_refinement(bd->selected_empty, &er);
    return;
  }

  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp) return;

  dt_masks_refinement_t r = { 0 };
  _refine_read_controls(bd, &r);
  // stamp which mask this refinement is for. A group's is broadcast onto every
  // member (there is no per-group storage), so without this the renderer cannot
  // tell a member's own refinement from a copy of its group's -- see
  // dt_masks_refine_scope_t. ALL_SHAPES is a per-shape broadcast, not a group
  // one: each shape refines its own mask.
  if(r.enabled)
    r.enabled = (bd->masks_refine_scope_kind == REFINE_SCOPE_GROUP)
                  ? DT_MASKS_REFINE_GROUP
                  : DT_MASKS_REFINE_ELEMENT;

  if(bd->masks_refine_scope_kind == REFINE_SCOPE_ALL_SHAPES)
  {
    // broadcast to every drawn (non-parametric) form
    for(GList *l = grp->points; l; l = g_list_next(l))
    {
      dt_masks_point_group_t *pt = l->data;
      dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, pt->formid);
      if(form && !(form->type & DT_MASKS_PARAMETRIC)) pt->refinement = r;
    }
  }
  else if(bd->masks_refine_scope_kind == REFINE_SCOPE_GROUP)
  {
    // broadcast to every member of the selected group's run
    GList *ids = _selected_group_formids(grp, bd->masks_refine_scope_formid);
    for(GList *l = ids; l; l = g_list_next(l))
    {
      dt_masks_point_group_t *pt = _group_point(grp, GPOINTER_TO_INT(l->data));
      if(pt) pt->refinement = r;
    }
    g_list_free(ids);
  }
  else // REFINE_SCOPE_ELEMENT: write only the single targeted element, no broadcast
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
  else if(bd->masks_refine_scope_kind == REFINE_SCOPE_GROUP
          || bd->masks_refine_scope_kind == REFINE_SCOPE_EMPTY_GROUP)
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
      name = g_strdup(form->name);
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
        : g_strdup_printf("%s-%d", _op_name_for_state(head ? head->state : 0),
                          _group_ordinal_of_cid(module, bd->masks_refine_scope_formid));

    DTGTKCairoPaintIconFunc paint = _op_paint_for_state(head ? head->state : 0);
    if(paint) icon_w = _make_icon_widget(paint);
  }
  else if(bd->masks_refine_scope_kind == REFINE_SCOPE_EMPTY_GROUP)
  {
    const char *custom_name = _empty_group_name(bd->selected_empty);
    name = (custom_name && custom_name[0])
             ? g_strdup(custom_name)
             : g_strdup_printf(
                 "%s-%d", _op_name_for_state(_empty_group_op(bd->selected_empty)),
                 _group_ordinal_any(module, INVALID_MASKID, bd->selected_empty));

    DTGTKCairoPaintIconFunc paint =
      _op_paint_for_state(_empty_group_op(bd->selected_empty));
    if(paint) icon_w = _make_icon_widget(paint);
  }
  else
  {
    name = g_strdup(_("whole mask"));
    icon_w = _make_icon_widget(dtgtk_cairo_paint_masks_eye);
  }

  if(icon_w && bd->masks_refine_icon_box)
  {
    gtk_box_pack_start(GTK_BOX(bd->masks_refine_icon_box), icon_w, FALSE, FALSE, 0);
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
static void
_set_badge_active(GtkWidget *badge, gboolean active, const char *tooltip_when_active);
static const char *_solo_badge_tooltip(void);
static const char *_soloedit_badge_tooltip(void);
enum
{
  MASK_SOLO_BADGE_NONE = 0,
  MASK_SOLO_BADGE_SOLO,
  MASK_SOLO_BADGE_SOLOEDIT,
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

  int count = 0, pos = 0;
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

  for(GList *fpts = grp->points; fpts; fpts = g_list_next(fpts), pos++)
  {
    dt_masks_point_group_t *fpt = fpts->data;
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
      const int saved_count = count;
      sel->functions->modify_property(sel, prop, old_value, value, &sum, &count, &min,
                                      &max);
      if(count != saved_count && value != old_value)
        dt_masks_gui_form_create(sel, gui, pos, dev->gui_module);
    }
  }

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

  dt_control_queue_redraw_center();

  // an opacity change can push a row (or its whole group) across the
  // low-opacity threshold -- refresh the badges in place, on every drag tick.
  // Nothing else in the panel changes, so this must not be a rebuild.
  if(prop == DT_MASKS_PROPERTY_OPACITY) _refresh_lowop_badges(module);

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

  dt_masks_form_t *grp = _module_mask_group(ed->module);
  int pos = 0;
  if(grp)
    for(GList *fpts = grp->points; fpts; fpts = g_list_next(fpts), pos++)
      if(((dt_masks_point_group_t *)fpts->data)->formid == ed->formid) break;

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
  // bug: "auto-expand selected shape" (_auto_expand_selected_row)
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
  if(!bd->masks_suppress_toggle_select && !is_group && bd->panel_selected_formid != key)
    _set_form_target(module, key);

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
}

// build the toggle button + docked editor pair shared by shape rows, raster
// rows, and group headers: a chevron styled like the parametric row's
// existing in/out toggle ("mask-inout-toggle"), remembering its expanded state
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
  // "auto-expand selected shape" (masks panel hamburger -> options): while
  // enabled, expansion is strictly tied to bd->masks_last_expanded_shape --
  // the most recently selected shape that actually has a props row -- not
  // bd->panel_selected_formid directly: selecting something without its own
  // props toggle (a parametric channel row, a group) must leave whichever
  // shape was last expanded alone instead of collapsing it, so the panel
  // does not visibly shift just because the user picked a non-shape element
  // (see _auto_expand_selected_row, which maintains this field and performs
  // the matching in-place enforcement on selection change, since selection
  // itself never triggers a full rebuild). Groups are untouched -- this
  // option only ever affects shape rows (is_group is always FALSE at the
  // one call site, but kept explicit here for clarity).
  const gboolean auto_exp =
    dt_conf_get_bool("plugins/darkroom/masks/auto_expand_selected");
  const gboolean expanded = (!is_group && auto_exp)
                              ? (dt_is_valid_maskid(bd->panel_selected_formid)
                                   ? (key == bd->panel_selected_formid)
                                   : (key == bd->masks_last_expanded_shape))
                              : GPOINTER_TO_INT(g_hash_table_lookup(
                                  bd->masks_props_expanded, GUINT_TO_POINTER(key)));
  if(!is_group && auto_exp && dt_is_valid_maskid(bd->panel_selected_formid)
     && key == bd->panel_selected_formid)
    bd->masks_last_expanded_shape = key;

  GtkWidget *editor_box =
    _build_props_row_editor(module, key, is_group, opacity_only, exclude_opacity);
  gtk_widget_set_visible(editor_box, expanded);

  GtkWidget *btn = dtgtk_togglebutton_new(_paint_param_inout, 0, NULL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(btn), expanded);
  // an expander (chevron), not a mode toggle -- same convention as the
  // parametric row's chevron (see .mask-inout-toggle in darktable.css)
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
  gchar *txt = g_strdup_printf("%.0f%%", val * 100.0f);
  gtk_label_set_text(GTK_LABEL(label), txt);
  g_free(txt);
}

static void _inline_opacity_slider_changed_cb(GtkWidget *slider, gpointer user_data)
{
  GtkWidget *label = user_data;
  if(GTK_IS_LABEL(label))
    _inline_opacity_update_label(label, dt_bauhaus_slider_get(slider));
}

static void _place_bauhaus_whisker_popup(GtkWidget *anchor, const gint center_x)
{
  GtkWidget *popup = darktable.bauhaus->popup.window;
  if(!popup || !GTK_IS_WIDGET(popup) || !gtk_widget_get_visible(popup)) return;

  GtkWidget *toplevel = gtk_widget_get_toplevel(anchor);
  GdkWindow *top_gdk =
    gtk_widget_is_toplevel(toplevel) ? gtk_widget_get_window(toplevel) : NULL;
  GdkWindow *popup_gdk = gtk_widget_get_window(popup);
  if(!top_gdk || !popup_gdk) return;

  gint top_x, top_y;
  gdk_window_get_origin(top_gdk, &top_x, &top_y);

  gint rx, ry;
  gtk_widget_translate_coordinates(anchor, toplevel, 0, 0, &rx, &ry);
  GtkAllocation alloc;
  gtk_widget_get_allocation(anchor, &alloc);

  const gint anchor_y = top_y + ry;
  const gint pop_size = DT_PIXEL_APPLY_DPI(180);
  gdk_window_resize(popup_gdk, pop_size, pop_size);

  GdkRectangle workarea = { 0 };
  GdkMonitor *mon =
    gdk_display_get_monitor_at_window(gdk_window_get_display(top_gdk), top_gdk);
  if(mon) gdk_monitor_get_workarea(mon, &workarea);

  const gint gap = DT_PIXEL_APPLY_DPI(6);
  const gint space_above = anchor_y - workarea.y;
  const gint space_below = (workarea.y + workarea.height) - (anchor_y + alloc.height);

  const gint pop_y = (space_below >= pop_size + gap || space_below >= space_above)
                       ? anchor_y + alloc.height + gap
                       : anchor_y - gap - pop_size;

  gint pop_x = center_x - pop_size / 2;
  gint panel_x = workarea.x, panel_w = workarea.width;
  if(dt_ui_panel_ancestor(darktable.gui->ui, DT_UI_PANEL_LEFT, anchor))
  {
    panel_x = top_x;
    panel_w = dt_ui_panel_get_size(darktable.gui->ui, DT_UI_PANEL_LEFT);
  }
  else if(dt_ui_panel_ancestor(darktable.gui->ui, DT_UI_PANEL_RIGHT, anchor))
  {
    panel_w = dt_ui_panel_get_size(darktable.gui->ui, DT_UI_PANEL_RIGHT);
    panel_x = top_x + gtk_widget_get_allocated_width(toplevel) - panel_w;
  }
  pop_x = CLAMP(pop_x, panel_x, panel_x + panel_w - pop_size);

  gdk_window_move(popup_gdk, pop_x, pop_y);

  darktable.bauhaus->popup.position.x = pop_x;
  darktable.bauhaus->popup.position.y = pop_y;
  darktable.bauhaus->popup.position.width = pop_size;
  darktable.bauhaus->popup.position.height = pop_size;
  darktable.bauhaus->popup.offset = 0;
  darktable.bauhaus->popup.offcut = 0;
}

static gboolean _inline_opacity_popup_idle(gpointer user_data)
{
  GtkWidget *evbox = user_data;
  if(!GTK_IS_WIDGET(evbox)) return G_SOURCE_REMOVE;
  GtkWidget *slider = g_object_get_data(G_OBJECT(evbox), "opacity-slider");
  if(!slider || !GTK_IS_WIDGET(slider)) return G_SOURCE_REMOVE;

  dt_bauhaus_widget_show_popup(slider);

  GtkWidget *top = gtk_widget_get_toplevel(evbox);
  gint sx = 0, sy = 0;
  gtk_widget_translate_coordinates(evbox, top, 0, 0, &sx, &sy);
  GdkWindow *top_gdk = gtk_widget_is_toplevel(top) ? gtk_widget_get_window(top) : NULL;
  gint tx = 0, ty = 0;
  if(top_gdk) gdk_window_get_origin(top_gdk, &tx, &ty);
  GtkAllocation alloc;
  gtk_widget_get_allocation(evbox, &alloc);
  const gint center_x = tx + sx + alloc.width / 2;

  _place_bauhaus_whisker_popup(evbox, center_x);
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
    dt_bauhaus_slider_set(slider, 1.0f);
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

  float step = 0.05f;
  if(is_ctrl)
    step = 0.01f;
  else if(is_shift)
    step = 0.10f;
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
  const float new_val = CLAMP(current + dir * step, 0.0f, 1.0f);
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
  gtk_label_set_width_chars(GTK_LABEL(label), 5);
  gtk_label_set_xalign(GTK_LABEL(label), 1.0f);
  if(slider)
  {
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

  GtkWidget *container = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
  gtk_box_pack_start(GTK_BOX(container), box, FALSE, FALSE, 0);
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
  GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);

  if(handle) gtk_box_pack_start(GTK_BOX(hbox), handle, FALSE, FALSE, 0);
  if(name)
  {
    gtk_widget_set_hexpand(name, TRUE);
    gtk_box_pack_start(GTK_BOX(hbox), name, TRUE, TRUE, 0);
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

  gtk_box_pack_start(GTK_BOX(row), hbox, TRUE, TRUE, 0);
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
                                      GHashTable *hidden_by_formid)
{
  if(!grp || !(grp->type & DT_MASKS_GROUP)) return;
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    const gboolean hidden =
      inherited_hidden || (pt->state & (DT_MASKS_STATE_HIDDEN | DT_MASKS_STATE_DISABLE))
      || _op_is_bypassed(pt->state);
    dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, pt->formid);
    if(form && (form->type & DT_MASKS_GROUP))
      _collect_effective_hidden(form, hidden, hidden_by_formid);
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
  _collect_effective_hidden(grp, FALSE, hidden);

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

// the head form is the base (no operator); every following form must carry one
void _normalize_group_operators(dt_masks_form_t *grp)
{
  // the bottom (foundation) group now keeps its own real operator (see the
  // fixed seed placeholder row / _flexi_apply_group_op in group.c) -- it is no
  // longer force-stamped to union. IMPORTANT: do not fold any "force this
  // point's operator" logic back into this loop -- _starts_group() reads the
  // *previous* point's already-normalized state, so mutating a point's own
  // operator here changes what the next iteration sees as its neighbour and
  // can misdetect a run boundary (this previously split a freshly-retagged
  // group in two: forcing the bottom point back to union made the next point
  // in the very same loop look like it started a new run).
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = l->data;
    if(l == grp->points)
    {
      pt->state &= ~DT_MASKS_STATE_SHOW;
      // the bottom point has nothing below it, so a break marker there is
      // meaningless -- clear any that travelled in on a reorder
      pt->group_start = 0;
    }
    else
    {
      pt->state |= DT_MASKS_STATE_SHOW;
    }
    // back-compat only: a point with no operator bit at all (never valid for
    // new edits) reads as union. Bypass does not count as an operator here --
    // it is a modifier layered on one, so a bypassed group must still carry
    // the operator it goes back to.
    if((pt->state & DT_MASKS_STATE_OP_COMBINE) == DT_MASKS_STATE_NONE)
      pt->state |= DT_MASKS_STATE_UNION;
  }
}

// install (once) the CSS that draws a border around the selected mask-list row.
// Done at runtime so it works regardless of the active theme; registered just
// above darktable's own theme provider (USER+1) so the border is not overridden.
// rows tagged "mask-row" may be nested inside cluster expanders, so the row
// lookups below walk the whole subtree under masks_list_box.
static void _apply_row_selection(GtkWidget *w, const dt_mask_id_t sel)
{
  if(!GTK_IS_CONTAINER(w)) return;
  GList *kids = gtk_container_get_children(GTK_CONTAINER(w));
  for(GList *c = kids; c; c = g_list_next(c))
  {
    GtkWidget *child = c->data;
    if(g_object_get_data(G_OBJECT(child), "mask-row"))
    {
      const dt_mask_id_t fid =
        GPOINTER_TO_INT(g_object_get_data(G_OBJECT(child), "formid"));
      if(dt_is_valid_maskid(sel) && fid == sel)
        dt_gui_add_class(child, "mask-list-row-selected");
      else
        dt_gui_remove_class(child, "mask-list-row-selected");
    }
    else
      _apply_row_selection(child, sel); // recurse into expanders / boxes
  }
  g_list_free(kids);
}

// same idea as _apply_row_selection, but for a group's header (tagged "mask-header"
// at construction, with "group-key" holding its cid and "header-widget" the inner
// box the CSS class actually goes on -- see the header build in _build_masks_list).
static void _apply_group_selection(GtkWidget *w, const dt_mask_id_t sel)
{
  if(!GTK_IS_CONTAINER(w)) return;
  GList *kids = gtk_container_get_children(GTK_CONTAINER(w));
  for(GList *c = kids; c; c = g_list_next(c))
  {
    GtkWidget *child = c->data;
    if(g_object_get_data(G_OBJECT(child), "mask-header"))
    {
      const dt_mask_id_t cid =
        (dt_mask_id_t)GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(child), "group-key"));
      GtkWidget *target = g_object_get_data(G_OBJECT(child), "header-widget");
      if(!target) target = child;
      if(dt_is_valid_maskid(sel) && cid == sel)
        dt_gui_add_class(target, "mask-list-row-selected");
      else
        dt_gui_remove_class(target, "mask-list-row-selected");
    }
    else
      _apply_group_selection(child, sel); // recurse into expanders / boxes
  }
  g_list_free(kids);
}

// same idea as _apply_group_selection, but toggles a group header's own solo
// badge (tagged "solo-badge" at construction) instead of the selection class.
// Needed because soloing an element (_toggle_solo_form) only refreshes element
// rows in place (_refresh_all_shape_rows) -- without this, clearing a group
// solo by soloing one of its own elements left that group's badge stuck on
// screen even though bd->solo_group_key had already gone back to 0.
static void _apply_group_solo_badges(GtkWidget *w, const guint solo_key)
{
  if(!GTK_IS_CONTAINER(w)) return;
  GList *kids = gtk_container_get_children(GTK_CONTAINER(w));
  for(GList *c = kids; c; c = g_list_next(c))
  {
    GtkWidget *child = c->data;
    if(g_object_get_data(G_OBJECT(child), "mask-header"))
    {
      const guint cid = GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(child), "group-key"));
      GtkWidget *badge = g_object_get_data(G_OBJECT(child), "solo-badge");
      const gboolean bypassed =
        g_object_get_data(G_OBJECT(child), "group-bypassed") != NULL;
      if(badge)
        _set_solo_status_badge(badge, bypassed ? MASK_SOLO_BADGE_DISABLE
                                      : (solo_key != 0 && solo_key == cid)
                                        ? MASK_SOLO_BADGE_SOLO
                                        : MASK_SOLO_BADGE_NONE);
    }
    else
      _apply_group_solo_badges(child, solo_key); // recurse into expanders / boxes
  }
  g_list_free(kids);
}

// same idea as _apply_group_selection, but dims an empty group's header (tagged
// "eg-header" at construction) while any solo is active -- an empty group has
// no members, so it can never itself be the solo target, and should dim
// exactly like a real group whose every member is solo-hidden. Needed for the
// same reason as _apply_group_solo_badges: soloing an element only refreshes
// element rows in place, never headers.
static void _apply_empty_group_dimming(GtkWidget *w, const gboolean solo_active)
{
  if(!GTK_IS_CONTAINER(w)) return;
  GList *kids = gtk_container_get_children(GTK_CONTAINER(w));
  for(GList *c = kids; c; c = g_list_next(c))
  {
    GtkWidget *child = c->data;
    if(g_object_get_data(G_OBJECT(child), "eg-header"))
    {
      // "group-header-widget" (-> hdr specifically), not "header-widget" (->
      // the whole block, used for selection shading -- see
      // _pack_empty_group_header/_apply_empty_selection): dimming must stay
      // on the header row alone, same split a real group's own header uses
      // (see _apply_group_header_dimming's own comment) -- dimming the whole
      // block would also dim a pending-shape placeholder row sitting under an
      // empty group's header, which is not a solo-suppression target.
      GtkWidget *target = g_object_get_data(G_OBJECT(child), "group-header-widget");
      if(!target) target = child;
      gtk_widget_set_opacity(target, solo_active ? 0.45 : 1.0);
    }
    else
      _apply_empty_group_dimming(child, solo_active); // recurse into expanders / boxes
  }
  g_list_free(kids);
}

// same idea as _apply_group_solo_badges, but dims a *real* group/cluster
// header (class "mask-group-header") while any solo is active, mirroring
// _apply_empty_group_dimming for empty groups -- without this, only empty
// group headers and individual element rows dimmed on solo, leaving a real
// group's own header fully lit even though every shape inside it was
// solo-suppressed. The group that is itself the solo target (cid ==
// solo_group_key) must stay fully lit, not dim itself.
static void _apply_group_header_dimming(GtkWidget *w,
                                        const gboolean solo_active,
                                        const guint solo_group_key)
{
  if(!GTK_IS_CONTAINER(w)) return;
  GList *kids = gtk_container_get_children(GTK_CONTAINER(w));
  for(GList *c = kids; c; c = g_list_next(c))
  {
    GtkWidget *child = c->data;
    if(g_object_get_data(G_OBJECT(child), "mask-header"))
    {
      const guint cid = GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(child), "group-key"));
      GtkWidget *target = g_object_get_data(G_OBJECT(child), "group-header-widget");
      GtkWidget *within_sel = g_object_get_data(G_OBJECT(child), "within-sel-widget");
      GtkWidget *opacity_slider =
        g_object_get_data(G_OBJECT(child), "group-opacity-widget");
      const gboolean suppressed = solo_active && cid != solo_group_key;
      const gboolean bypassed =
        g_object_get_data(G_OBJECT(child), "group-bypassed") != NULL;
      if(target)
      {
        if(bypassed)
        {
          GtkWidget *ghandle = g_object_get_data(G_OBJECT(child), "ghandle-widget");
          GtkWidget *lbl_box = g_object_get_data(G_OBJECT(child), "title-label-box");
          GtkWidget *labevt = lbl_box ? gtk_widget_get_parent(lbl_box) : NULL;
          GtkWidget *opacity_inner =
            opacity_slider ? gtk_widget_get_parent(opacity_slider) : NULL;
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
      if(opacity_slider)
        gtk_widget_set_sensitive(opacity_slider, !suppressed && !bypassed);
      // tag the *soloed* group's whole block so its own cluster headers stay lit
      // (they dim by default under .mask-solo-active -- see darktable.css); a
      // group is being shown in full, so nothing inside it should read as
      // suppressed. Other groups' blocks keep the tag off, so their clusters dim.
      GtkWidget *block = g_object_get_data(G_OBJECT(child), "header-widget");
      if(block)
      {
        if(solo_group_key != 0 && cid == solo_group_key)
          dt_gui_add_class(block, "mask-group-soloed");
        else
          dt_gui_remove_class(block, "mask-group-soloed");
      }
    }
    else
      _apply_group_header_dimming(child, solo_active,
                                  solo_group_key); // recurse into expanders / boxes
  }
  g_list_free(kids);
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
  if(!GTK_IS_CONTAINER(w)) return;
  GList *kids = gtk_container_get_children(GTK_CONTAINER(w));
  for(GList *c = kids; c; c = g_list_next(c))
  {
    GtkWidget *child = c->data;
    if(g_object_get_data(G_OBJECT(child), "mask-header"))
    {
      const guint this_cid =
        GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(child), "group-key"));
      if(this_cid == cid)
      {
        GtkWidget *ghandle = g_object_get_data(G_OBJECT(child), "ghandle-widget");
        if(ghandle)
        {
          if(inverted)
            dt_gui_add_class(ghandle, "mask-list-handle-inverted");
          else
            dt_gui_remove_class(ghandle, "mask-list-handle-inverted");
          gtk_widget_queue_draw(ghandle);
        }
      }
    }
    else
      _apply_group_output_invert_icon(child, cid,
                                      inverted); // recurse into expanders / boxes
  }
  g_list_free(kids);
}

// same idea as _apply_group_selection, but for empty-group headers (tagged
// "eg-header" at construction, with "eg" holding the dt_masks_empty_group_t*
// and "header-widget" the box the CSS class goes on -- see _pack_empty_group_header).
// Needed because selecting a real group through the lightweight, no-rebuild
// path (_set_group_target) clears bd->selected_empty in the data model but,
// without this, never removed the stale highlight left on a previously-
// selected empty group's header widget.
static void _apply_empty_selection(GtkWidget *w, const struct dt_masks_empty_group_t *sel)
{
  if(!GTK_IS_CONTAINER(w)) return;
  GList *kids = gtk_container_get_children(GTK_CONTAINER(w));
  for(GList *c = kids; c; c = g_list_next(c))
  {
    GtkWidget *child = c->data;
    if(g_object_get_data(G_OBJECT(child), "eg-header"))
    {
      const struct dt_masks_empty_group_t *eg = g_object_get_data(G_OBJECT(child), "eg");
      GtkWidget *target = g_object_get_data(G_OBJECT(child), "header-widget");
      if(!target) target = child;
      if(eg == sel)
        dt_gui_add_class(target, "mask-list-row-selected");
      else
        dt_gui_remove_class(target, "mask-list-row-selected");
    }
    else
      _apply_empty_selection(child, sel); // recurse into expanders / boxes
  }
  g_list_free(kids);
}

static GtkWidget *_find_row_by_formid(GtkWidget *w, const dt_mask_id_t formid)
{
  if(!GTK_IS_CONTAINER(w)) return NULL;
  GtkWidget *found = NULL;
  GList *kids = gtk_container_get_children(GTK_CONTAINER(w));
  for(GList *c = kids; c && !found; c = g_list_next(c))
  {
    GtkWidget *child = c->data;
    if(g_object_get_data(G_OBJECT(child), "mask-row")
       && GPOINTER_TO_INT(g_object_get_data(G_OBJECT(child), "formid")) == formid)
      found = child;
    else
      found = _find_row_by_formid(child, formid);
  }
  g_list_free(kids);
  return found;
}

// O(1) shape-row lookup by form id via the masks_row_map index (see blend.h),
// used by every per-formid whole-list lookup instead of a recursive tree walk.
// Falls back to the tree walk if the map is somehow cold, so behaviour is never
// worse than before.
static GtkWidget *_masks_row_widget(dt_iop_gui_blend_data_t *bd,
                                    const dt_mask_id_t formid)
{
  if(!bd || !dt_is_valid_maskid(formid)) return NULL;
  GtkWidget *w = bd->masks_row_map
                   ? g_hash_table_lookup(bd->masks_row_map, GINT_TO_POINTER(formid))
                   : NULL;
  if(!w && bd->masks_list_box)
    w = _find_row_by_formid(GTK_WIDGET(bd->masks_list_box), formid);
  return w;
}

// The panel's four DnD payload types. Named here because each one is written
// twice -- once in a GtkTargetEntry table below, once in the hover classifier
// (_dnd_hover_kind) that compares the negotiated target's name back against it.
// A typo in either copy fails silently, as a drag that simply never matches.
#define DND_TARGET_ROW "dt-mask-row"
#define DND_TARGET_GROUP "dt-mask-group"
#define DND_TARGET_EMPTY "dt-mask-empty"
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

static const char *_soloedit_badge_tooltip(void)
{
  return _("solo edited: only this element's nodes/handles are editable on canvas\n"
           "(other elements still contribute to the visible mask)\n"
           "click here to clear solo edit");
}

// solo and solo-edit are mutually exclusive (see the clearing logic in
// _toggle_solo_form/_toggle_solo_group/_toggle_soloedit), so one row/header
// can only ever be in one of these states at a time -- they share a single
// badge slot instead of two, saving a cell in the badge stack (see
// _make_badge_stack) and, for element rows, the vertical space that cell used
// to cost every row whether or not solo-edit was ever in play. (MASK_SOLO_BADGE_*
// forward-declared above, with the other badge helper forward decls.)

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
    badge, status == MASK_SOLO_BADGE_SOLO       ? _solo_badge_tooltip()
           : status == MASK_SOLO_BADGE_SOLOEDIT ? _soloedit_badge_tooltip()
           : status == MASK_SOLO_BADGE_DISABLE  ? _disable_badge_tooltip()
                                                : NULL);
  gtk_widget_queue_draw(badge);
}

// a small badge shown next to a soloed, solo-edited or disabled element/group's
// label, reusing the same light-bg/dark-fg swap (see .mask-power-solo).
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
  else if(status == MASK_SOLO_BADGE_SOLOEDIT)
    dtgtk_cairo_paint_soloedit(cr, pad, pad, a.width - 2 * pad, a.height - 2 * pad, 0,
                               NULL);
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
  gtk_widget_set_size_request(badge, DT_PIXEL_APPLY_DPI(11), DT_PIXEL_APPLY_DPI(11));
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
  gtk_widget_set_size_request(badge, DT_PIXEL_APPLY_DPI(11), DT_PIXEL_APPLY_DPI(11));
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
// nothing else in the row shifts. `spacing` (DPI-scaled) is left as a small
// gap between the two squares.
static GtkWidget *_make_badge_stack(GtkWidget *lowop_badge, GtkWidget *solo_status_badge)
{
  GtkWidget *stack = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_PIXEL_APPLY_DPI(2));
  gtk_widget_set_valign(stack, GTK_ALIGN_CENTER);
  dt_gui_add_class(stack, "mask-badge-stack");
  if(lowop_badge) gtk_box_pack_start(GTK_BOX(stack), lowop_badge, FALSE, FALSE, 0);
  if(solo_status_badge)
    gtk_box_pack_start(GTK_BOX(stack), solo_status_badge, FALSE, FALSE, 0);
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

static void _update_lowop_badge(GtkWidget *badge,
                                const float opacity,
                                const gboolean is_group,
                                const gboolean is_noop)
{
  if(!badge) return;
  if(is_noop)
  {
    g_object_set_data(G_OBJECT(badge), "badge-noop", GINT_TO_POINTER(1));
    dt_gui_remove_class(badge, "mask-lowop-warn");
    dt_gui_add_class(badge, "mask-noop-warn");
    _set_badge_active(badge, TRUE,
                      _("this channel's range still covers its entire span, so it "
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
  const gboolean hidden =
    (pt->state & DT_MASKS_STATE_HIDDEN) || _op_is_bypassed(pt->state) || elem_disabled;
  const gboolean inverse = pt->state & DT_MASKS_STATE_INVERSE;
  const gboolean solo = bd->solo_formid == pt->formid;

  // a soloed (or solo-edited) element stays highlighted like a hovered row, not
  // just while the mouse is over it -- a distinct class from the transient hover
  // wash so it survives hovering elsewhere in the list (see _clear_hover_classes).
  if(solo || bd->soloedit_formid == pt->formid)
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
                                       : bd->soloedit_formid == pt->formid
                                         ? MASK_SOLO_BADGE_SOLOEDIT
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
    const gboolean solo_hidden =
      (pt->state & DT_MASKS_STATE_HIDDEN) || _op_is_bypassed(pt->state);
    if(row) gtk_widget_set_opacity(row, solo_hidden ? 0.45 : 1.0);
  }

  // a solo-suppressed element's controls have no visible effect while another
  // element is soloed (this row contributes nothing to the composite) -- gray
  // them out too, not just dim the row. Only the editor boxes (sliders) are
  // made insensitive, never row_vbox/row itself: the row must stay draggable
  // and selectable.
  GtkWidget *param_box = g_object_get_data(G_OBJECT(row_vbox), "param-editor-box");
  if(param_box) gtk_widget_set_sensitive(param_box, !hidden);
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
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = l->data;
    GtkWidget *row_vbox = _masks_row_widget(bd, pt->formid);
    if(row_vbox) _update_shape_row_state(bd, row_vbox, pt);
  }
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
  _apply_empty_group_dimming(GTK_WIDGET(bd->masks_list_box), solo_active);
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

// every member formid of the run (group) headed by `cid` (the run's own head
// point, matching how _build_masks_list derives a group header's "cid" from
// first_fid). Caller frees. Shared by the solo-canvas sync below and by the
// panel-selection shortcuts (_group_run_members), which both need "every
// member of group X" without walking the run-boundary logic themselves.
static GList *_group_run_members(dt_masks_form_t *grp, const dt_mask_id_t cid)
{
  GList *out = NULL;
  gboolean in_run = FALSE;
  for(GList *l = grp ? grp->points : NULL; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = l->data;
    if(_starts_group(l)) in_run = pt->formid == cid;
    if(in_run) out = g_list_prepend(out, GINT_TO_POINTER(pt->formid));
  }
  return out;
}

// the group's own persistent, multiplicative opacity (see
// dt_masks_point_group_t.group_opacity and the header's own inline slider) --
// broadcast onto every member of the run, so any one of them (the head, by
// convention) reports it. This is the group's own gain, independent of its
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
static void _apply_group_lowop_badges(GtkWidget *w, dt_masks_form_t *grp)
{
  if(!GTK_IS_CONTAINER(w)) return;
  GList *kids = gtk_container_get_children(GTK_CONTAINER(w));
  for(GList *c = kids; c; c = g_list_next(c))
  {
    GtkWidget *child = c->data;
    if(g_object_get_data(G_OBJECT(child), "mask-header"))
    {
      const guint cid = GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(child), "group-key"));
      GtkWidget *badge = g_object_get_data(G_OBJECT(child), "lowop-badge");
      if(badge)
        _update_lowop_badge(badge, _group_own_opacity(grp, (dt_mask_id_t)cid), TRUE,
                            FALSE);
    }
    else
      _apply_group_lowop_badges(child, grp); // recurse into expanders / boxes
  }
  g_list_free(kids);
}

// refresh every low-opacity badge in the panel (element rows and group headers)
// from the current opacities. Cheap and in-place -- no widget is created or
// destroyed -- so it can run on every tick of an opacity drag as well as at the
// end of a list rebuild.
static void _refresh_lowop_badges(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module ? module->blend_data : NULL;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!bd || !bd->masks_list_box || !grp) return;
  // an element's overall (effective) opacity is its own value multiplied by
  // its containing run's group-level gain (see _group_get_mask_roi_flexi in
  // group.c) -- track the running group's own opacity as the list is walked
  // top-to-bottom, same as _group_run_members' run-boundary test.
  float run_group_opacity = 1.0f;
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    if(_starts_group(l)) run_group_opacity = pt->group_opacity;
    GtkWidget *row_vbox = _masks_row_widget(bd, pt->formid);
    if(row_vbox)
    {
      const dt_masks_form_t *const sel =
        dt_masks_get_from_id(darktable.develop, pt->formid);
      _update_lowop_badge(g_object_get_data(G_OBJECT(row_vbox), "lowop-badge"),
                          pt->opacity * run_group_opacity, FALSE,
                          _parametric_form_is_noop(sel));
    }
  }
  _apply_group_lowop_badges(GTK_WIDGET(bd->masks_list_box), grp);
}

// is cid the base (bottom-most) group? grp->points is ordered bottom-up (see
// _starts_group), so the very first point is always a run head and is the
// base group's own cid.
static gboolean _group_is_base(dt_masks_form_t *grp, const dt_mask_id_t cid)
{
  return grp && grp->points
         && ((dt_masks_point_group_t *)grp->points->data)->formid == cid;
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
    GList *members = _group_run_members(grp, (dt_mask_id_t)bd->solo_group_key);
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

static void _update_row_selection(dt_iop_gui_blend_data_t *bd)
{
  if(!bd || !bd->masks_list_box) return;
  // every group's element rows are nested inside masks_list_box (under their header)
  _apply_row_selection(GTK_WIDGET(bd->masks_list_box), bd->panel_selected_formid);
  _apply_group_selection(GTK_WIDGET(bd->masks_list_box), bd->panel_selected_group_cid);
  _apply_empty_selection(GTK_WIDGET(bd->masks_list_box), bd->selected_empty);
  if(darktable.develop && darktable.develop->form_gui)
    darktable.develop->form_gui->panel_selected_formid = bd->panel_selected_formid;
  _flexi_new_op_follow_selection(bd);
  _flexi_refine_follow_selection(bd);
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
static GtkWidget *_find_collapsed_cluster_header(GtkWidget *w, const dt_mask_id_t formid)
{
  if(!GTK_IS_CONTAINER(w)) return NULL;
  GtkWidget *found = NULL;
  GList *kids = gtk_container_get_children(GTK_CONTAINER(w));
  for(GList *c = kids; c && !found; c = g_list_next(c))
  {
    GtkWidget *child = c->data;
    // group headers carry their member ids; when a shape's own row cannot be
    // located directly, fall back to highlighting its enclosing group header.
    GList *members = g_object_get_data(G_OBJECT(child), "group-formids");
    if(members)
      for(GList *m = members; m; m = g_list_next(m))
        if(GPOINTER_TO_INT(m->data) == formid)
        {
          found = child;
          break;
        }
    if(!found) found = _find_collapsed_cluster_header(child, formid);
  }
  g_list_free(kids);
  return found;
}

// canvas -> list selection sync: when a shape is selected on the canvas (click),
// highlight its row in the flexi mask list. An invalid id clears the selection
// (e.g. clicking empty canvas, or toggling the selected shape off). No-op when
// there is no list (classic mode / no masks).
void dt_iop_gui_masks_select_form(dt_iop_module_t *module, const dt_mask_id_t formid)
{
  if(!module) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd || !bd->masks_list_box) return;
  const dt_mask_id_t id = dt_is_valid_maskid(formid) ? formid : INVALID_MASKID;
  if(bd->panel_selected_formid == id) return;
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
  if(!target) target = _find_collapsed_cluster_header(box, formid);
  if(target) dt_gui_add_class(target, "mask-list-row-hover");
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
  // acting on this row selects it if it wasn't already selected, but never
  // deselects it -- same select-only rule as every other action control
  // (see _set_form_target)
  if(bd->panel_selected_formid != id) _set_form_target(module, id);
  dt_masks_point_parametric_t *p = form->points->data;
  const uint32_t want = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(btn)) ? 1u : 0u;

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

// operator descriptors: the same icons (and order) the mask manager uses
static const struct
{
  dt_masks_state_t state;
  DTGTKCairoPaintIconFunc paint;
  const char *name;
} _masks_ops[] = {
  { DT_MASKS_STATE_UNION, dtgtk_cairo_paint_masks_union, N_("union") },
  { DT_MASKS_STATE_INTERSECTION, dtgtk_cairo_paint_masks_intersection,
    N_("intersection") },
  { DT_MASKS_STATE_DIFFERENCE, dtgtk_cairo_paint_masks_difference, N_("difference") },
  { DT_MASKS_STATE_SUM, dtgtk_cairo_paint_masks_sum, N_("sum") },
  { DT_MASKS_STATE_EXCLUSION, dtgtk_cairo_paint_masks_exclusion, N_("exclusion") },
  { DT_MASKS_STATE_MULTIPLY, dtgtk_cairo_paint_masks_multiply, N_("multiply") },
  { DT_MASKS_STATE_OP_SCREEN, dtgtk_cairo_paint_tool_blur, N_("screen") },
  // bypass is last so the loops below (which return the first bit they find)
  // keep reporting the group's real combining operator; it is a modifier on
  // top of one of the operators above, not one of them, and every menu that
  // offers "which operator does this group use" skips it -- only the
  // between-group chooser for an existing group offers it, as a toggle.
  { DT_MASKS_STATE_OP_BYPASS, _paint_masks_bypass, N_("disable") }
};

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

// the group's combining operator name, unaffected by bypass: a bypassed group
// keeps its identity (and its "<op>-<n>" default label) while disabled.
static const char *_op_name_for_state(const int state)
{
  for(int i = 0; i < (int)(sizeof(_masks_ops) / sizeof(_masks_ops[0])); i++)
    if((state & DT_MASKS_STATE_OP_COMBINE) & _masks_ops[i].state)
      return _(_masks_ops[i].name);
  return _("union");
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
  gtk_widget_set_tooltip_text(btn, _("add a new group above the selected group\n"
                                     "(ctrl+click to add it below instead)\n"
                                     "click to pick its operator"));
  gtk_widget_queue_draw(btn);
}

// stage an empty group of the chosen operator on top of the list (defined after
// the operator-index helpers it relies on).
static void _stage_new_group(dt_iop_module_t *module,
                             const int op_state,
                             const gboolean below_target);

// build a labelled "icon + name" menu item for an operator chooser
static GtkWidget *_op_menu_item(DTGTKCairoPaintIconFunc paint, const char *name)
{
  GtkWidget *it = gtk_menu_item_new();
  GtkWidget *hb = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, DT_PIXEL_APPLY_DPI(6));
  gtk_box_pack_start(GTK_BOX(hb), gtk_image_new_from_pixbuf(_op_pixbuf(paint)), FALSE,
                     FALSE, 0);
  gtk_box_pack_start(GTK_BOX(hb), gtk_label_new(_(name)), FALSE, FALSE, 0);
  gtk_container_add(GTK_CONTAINER(it), hb);
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
  GtkWidget *box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
  dt_gui_add_class(box, "mask-op-combo");
  GtkWidget *btn = dtgtk_button_new(icon, 0, NULL);
  gtk_widget_set_valign(btn, GTK_ALIGN_CENTER);
  gtk_box_pack_start(GTK_BOX(box), btn, FALSE, FALSE, 0);
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
// run _refresh_all_shape_rows immediately afterwards, which repaints the
// cleared row's badge and solo class from bd->soloedit_formid anyway. This
// used to force a full list rebuild here "because the solo-edit toggle
// button's checked state is only set at row-construction time" -- there is no
// such button any more, solo-edit is a check menu item built fresh each time a
// row's actions menu opens (see _build_shape_actions_menu), so the only thing
// the state still drives in a row is the badge/class pair above.
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
  else if(status == MASK_SOLO_BADGE_SOLOEDIT)
    _toggle_soloedit(module, id);
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

// core of "reset mask": remove every shape and drop every empty group, with no
// confirmation and no rebuild of its own -- callers that need those (the plain
// reset button, group-layout preset apply) add them on top. Factored out so a
// preset apply can reuse the exact same wipe instead of re-deriving it.
void _masks_reset_mask_core(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(grp && grp->points)
  {
    dt_masks_clear_form_gui(darktable.develop);
    GList *ids = NULL;
    for(GList *l = grp->points; l; l = g_list_next(l))
      ids =
        g_list_prepend(ids, GINT_TO_POINTER(((dt_masks_point_group_t *)l->data)->formid));
    for(GList *l = ids; l; l = g_list_next(l))
    {
      dt_masks_form_t *f =
        dt_masks_get_from_id(darktable.develop, GPOINTER_TO_INT(l->data));
      if(f) dt_masks_form_remove(module, grp, f);
    }
    g_list_free(ids);
    dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  }
  // drop every empty group and re-seed the scaffold on the next rebuild
  _empty_groups_clear(bd);
  bd->scaffold_seeded = FALSE;
  bd->masks_selection_seeded = FALSE;
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
}

// "reset mask": remove every shape and restore the virgin add/intersect/subtract
// scaffold. Destructive, so it asks for confirmation first.
static void _masks_reset_mask(GtkWidget *btn, dt_iop_module_t *module)
{
  if(DT_IN_GUI_UPDATE()) return;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(grp && grp->points
     && !dt_gui_show_yes_no_dialog(
       _("reset mask?"), "", _("this removes every shape from this mask. continue?")))
    return;

  _masks_reset_mask_core(module);
  _build_masks_list(module);
  // repopulate the refinement controls from whatever scope the reset settled
  // on -- the six sliders would otherwise keep displaying the values that were
  // just cleared out from under them
  _flexi_refine_follow_selection(module->blend_data);
  _refresh_canvas_edit(module);
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

// Post-condition shared by the two "move an element to a group" drops: the
// moved element must end up in the SAME run as the drop target. Both paths make
// that true by construction -- they copy the target's operator AND its partition
// key (see _group_keys_snapshot, which keys every run by its head formid) -- so
// a failure here means src and dst disagreed about which run dst is in. The one
// way that can happen is `dst` having been read from stale widget data: the
// "group-formids" a row/header carries is only refreshed by a full panel
// rebuild, and rebuilds are g_idle-deferred (_queue_masks_list_rebuild). A stale
// dst hands src a foreign key, _group_keys_apply stamps a group_start on it, and
// the element lands in a group of its own instead of joining the target.
//
// User-reported once, not reproducible: "moved an element from A to B and it
// ended up in C" (target was the bottom group). Left instrumented rather than
// guessed at -- this is cheap (two run lookups per drop) and prints the whole
// layout, which is what reconstructing the case actually needs.
static void _verify_element_joined(dt_masks_form_t *grp,
                                   const dt_mask_id_t src,
                                   const dt_mask_id_t dst,
                                   const char *where)
{
  if(!grp) return;
  const dt_mask_id_t src_cid = _group_cid_of_form(grp, src);
  const dt_mask_id_t dst_cid = _group_cid_of_form(grp, dst);
  if(src_cid == dst_cid) return;
  dt_print(DT_DEBUG_ALWAYS,
           "[masks] %s: element %d did NOT join target %d"
           " (landed in run %d, target is run %d) -- please report this layout:",
           where, src, dst, src_cid, dst_cid);
  int idx = 0;
  for(GList *l = grp->points; l; l = g_list_next(l), idx++)
  {
    const dt_masks_point_group_t *pt = l->data;
    dt_print(DT_DEBUG_ALWAYS, "[masks]   [%d] fid=%d op=0x%x group_start=%d%s%s", idx,
             pt->formid, (unsigned)(pt->state & DT_MASKS_STATE_OP), (int)pt->group_start,
             pt->formid == src ? "  <- moved" : "",
             pt->formid == dst ? "  <- target" : "");
  }
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

  dt_masks_point_group_t *spt = NULL;
  for(GList *l = grp->points; l; l = g_list_next(l))
    if(((dt_masks_point_group_t *)l->data)->formid == src)
    {
      spt = l->data;
      break;
    }
  if(!spt) return FALSE;

  // if this empties src's group, keep it alive as an empty-group placeholder
  struct dt_masks_empty_group_t *emptied = _capture_emptied_group(grp, src);
  // first-class groups: snapshot the partition, then make the dragged shape
  // join the target shape's group (adopt its key + operator). Re-stamping
  // from the key map keeps every OTHER group distinct even when operators
  // coincide, so dropping a shape between two same-op groups no longer merges
  // them. (The drop lands next to the target, i.e. inside its run.)
  GHashTable *keys = _group_keys_snapshot(grp);
  const gpointer dkey = g_hash_table_lookup(keys, GINT_TO_POINTER(dst));
  g_hash_table_insert(keys, GINT_TO_POINTER(src), dkey);
  const dt_masks_point_group_t *dpt = _group_point(grp, dst);
  if(dpt)
    spt->state = (spt->state & ~DT_MASKS_STATE_OP) | (dpt->state & DT_MASKS_STATE_OP);

  grp->points = g_list_remove(grp->points, spt);
  int tgt = 0, idx = 0;
  for(GList *l = grp->points; l; l = g_list_next(l), idx++)
    if(((dt_masks_point_group_t *)l->data)->formid == dst)
    {
      tgt = idx;
      break;
    }
  grp->points = g_list_insert(grp->points, spt, above ? tgt + 1 : tgt);
  _group_keys_apply(grp, keys);
  g_hash_table_destroy(keys);

  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(emptied) bd->empty_groups = g_list_append(bd->empty_groups, emptied);
  _normalize_group_operators(grp);
  _verify_element_joined(grp, src, dst, "row drop");

  // a moved element should stay selected at the end of the drag -- otherwise
  // it lands in its new spot with no visible indication of what just moved
  bd->panel_selected_formid = src;
  dt_masks_form_t *sform = dt_masks_get_from_id(darktable.develop, src);
  bd->panel_selected_group_cid = (sform && !(sform->type & DT_MASKS_PARAMETRIC))
                                   ? _group_cid_of_form(grp, src)
                                   : INVALID_MASKID;
  return TRUE;
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
    const dt_mask_id_t src = *(const dt_mask_id_t *)gtk_selection_data_get_data(sel);
    const dt_mask_id_t dst = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(w), "formid"));
    // rows display bottom-up: dropping on the top half of a row places the
    // shape visually above the target (= later in the list).
    const int h = gtk_widget_get_allocated_height(w);
    const gboolean above = (h > 0 && y < h / 2);
    ok = _model_drop_element_onto_element(module, _module_mask_group(module),
                                          src, dst, above);
    if(ok)
    {
      dt_print(DT_DEBUG_MASKS, "[masks] form %d drag-moved near %d", src, dst);
      dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
    }
  }
  gtk_drag_finish(ctx, ok, FALSE, time);
  if(ok) _queue_masks_list_rebuild(module);
}

// group-header drag-and-drop: reorder a whole group (its contiguous member run)
// as a unit. A separate target type from the per-shape row DnD so the two don't
// interfere. Same-op groups that end up adjacent simply merge (inferred runs).
static const GtkTargetEntry _mask_group_dnd[] = { { (gchar *)DND_TARGET_GROUP,
                                                    GTK_TARGET_SAME_APP, 0 } };

// empty-group drag-and-drop: reorder an empty (staged) group the same way a real
// group is reordered, so it is not stuck at whatever spot it was created at.
static const GtkTargetEntry _mask_empty_dnd[] = { { (gchar *)DND_TARGET_EMPTY,
                                                    GTK_TARGET_SAME_APP, 0 } };

// a same-kind element cluster's own drag-and-drop: moves every one of its
// members together, as one contiguous block preserving their relative order --
// like a group drag, but for just this same-kind run within a group (see
// _masks_cluster_move). A separate target type from both the per-shape row
// and per-group DnD so all three coexist without interfering.
static const GtkTargetEntry _mask_cluster_dnd[] = { { (gchar *)DND_TARGET_CLUSTER,
                                                      GTK_TARGET_SAME_APP, 0 } };

// a group header (real or empty) accepts every kind of drop: a whole real group
// (reorder), a single shape (drop onto a group to move the shape into it), an
// empty group (reorder), and a whole cluster (move every member together).
// The receive handler routes on the entry info below.
enum
{
  DND_MASK_GROUP = 0,
  DND_MASK_ROW = 1,
  DND_MASK_EMPTY = 2,
  DND_MASK_CLUSTER = 3
};
static const GtkTargetEntry _mask_hdr_dnd[] = {
  { (gchar *)DND_TARGET_GROUP, GTK_TARGET_SAME_APP, DND_MASK_GROUP },
  { (gchar *)DND_TARGET_ROW, GTK_TARGET_SAME_APP, DND_MASK_ROW },
  { (gchar *)DND_TARGET_EMPTY, GTK_TARGET_SAME_APP, DND_MASK_EMPTY },
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
  // any member formid identifies the group's run (rebuilt on the receive side)
  GList *ids = g_object_get_data(G_OBJECT(w), "group-formids");
  const dt_mask_id_t id = ids ? GPOINTER_TO_INT(ids->data) : INVALID_MASKID;
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

// an empty group has no formid of its own (nothing serialized to identify it
// by), so its DnD payload is its own GUI-side pointer instead -- safe because
// GTK_TARGET_SAME_APP never leaves this process, and the pointer stays valid
// for the (synchronous) lifetime of a single drag.
static void _masks_empty_drag_get(GtkWidget *w,
                                  GdkDragContext *ctx,
                                  GtkSelectionData *sel,
                                  guint info,
                                  guint time,
                                  gpointer user_data)
{
  gpointer eg = g_object_get_data(G_OBJECT(w), "eg");
  gtk_selection_data_set(sel, gtk_selection_data_get_target(sel), 8, (const guchar *)&eg,
                         sizeof(eg));
}

// index of the first point whose formid is in `ids`, and (via *last) the last;
// returns -1 if none found.
int _run_extent(dt_masks_form_t *grp, GList *ids, int *last)
{
  int first = -1, idx = 0;
  *last = -1;
  for(GList *l = grp->points; l; l = g_list_next(l), idx++)
  {
    const dt_mask_id_t fid = ((dt_masks_point_group_t *)l->data)->formid;
    for(GList *m = ids; m; m = g_list_next(m))
      if(GPOINTER_TO_INT(m->data) == fid)
      {
        if(first < 0) first = idx;
        *last = idx;
        break;
      }
  }
  return first;
}

// defined near the other unified-reorder helpers, after dt_masks_empty_group_t
// is declared; forward-declared here so the group/empty drag receive handlers
// (which appear earlier in the file) can call it.
gboolean _masks_reorder_groups(dt_iop_module_t *module,
                                      const gboolean src_is_empty,
                                      const dt_mask_id_t src_cid,
                                      struct dt_masks_empty_group_t *src_eg,
                                      const gboolean dst_is_empty,
                                      const dt_mask_id_t dst_cid,
                                      struct dt_masks_empty_group_t *dst_eg,
                                      const gboolean above);

// defined near the other unified-move helpers (below _capture_emptied_group_multi);
// forward-declared here so the cluster drag receive handlers (which appear
// earlier in the file) can call it.
gboolean _masks_cluster_move(dt_iop_module_t *module,
                                    GList *member_ids,
                                    const dt_mask_id_t dst,
                                    const gboolean dst_is_group,
                                    const gboolean above);

// Select the group a drag just moved, once it has landed.
//
// Every element-level drop already does this for the element it moved ("a moved
// element should stay selected at the end of the drag -- otherwise it lands in
// its new spot with no visible indication of what just moved", see
// _masks_row_drag_received). Group-level drops did not, so a moved group landed
// unselected and the selection still pointed at whatever was selected before the
// drag -- which then silently decided where the next "add group" went.
//
// The cid must be re-derived from the run head *after* the reorder; a cid read
// before it describes the old layout.
static void _select_moved_group(dt_iop_module_t *module, const dt_mask_id_t cid)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd) return;
  bd->selected_empty = NULL;
  bd->panel_selected_formid = INVALID_MASKID;
  bd->panel_selected_group_cid = cid;
}

// same, for a staged (empty) group: it has no members, so it is identified by
// its own pointer rather than a run head
static void _select_moved_empty_group(dt_iop_module_t *module, dt_masks_empty_group_t *eg)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd) return;
  bd->panel_selected_formid = INVALID_MASKID;
  bd->panel_selected_group_cid = INVALID_MASKID;
  bd->selected_empty = eg;
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
    const dt_mask_id_t src = *(const dt_mask_id_t *)gtk_selection_data_get_data(sel);
    GList *dst_ids = g_object_get_data(G_OBJECT(w), "group-formids");
    const dt_mask_id_t dst = dst_ids ? GPOINTER_TO_INT(dst_ids->data) : INVALID_MASKID;
    dt_masks_form_t *grp = _module_mask_group(module);
    if(grp && dt_is_valid_maskid(dst))
    {
      const gboolean above = _group_drop_above(w, y);
      ok = _masks_reorder_groups(module, FALSE, _group_cid_of_form(grp, src), NULL, FALSE,
                                 _group_cid_of_form(grp, dst), NULL, above);
      // a moved group stays selected, exactly as a moved element does (see
      // _masks_row_drag_received's own note): otherwise it lands in its new
      // spot with nothing indicating what just moved, and -- worse -- the
      // selection still points at whatever was selected beforehand, so the next
      // "add group" anchors above *that* group rather than the one just
      // dragged. Re-derived from the run head after the reorder, never reused
      // from before it.
      if(ok) _select_moved_group(module, _group_cid_of_form(grp, src));
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
  if(!grp || !dt_is_valid_maskid(dst) || src == dst) return FALSE;

  // the dropped shape is already in this group's run? then nothing to do
  GList *dst_run = _selected_group_formids(grp, dst);
  gboolean already = FALSE;
  for(GList *l = dst_run; l; l = g_list_next(l))
    if(GPOINTER_TO_INT(l->data) == src)
    {
      already = TRUE;
      break;
    }
  g_list_free(dst_run);

  dt_masks_point_group_t *sp = NULL;
  for(GList *l = grp->points; l; l = g_list_next(l))
    if(((dt_masks_point_group_t *)l->data)->formid == src)
    {
      sp = l->data;
      break;
    }
  const dt_masks_point_group_t *dp = _group_point(grp, dst);
  if(!sp || !dp || already) return FALSE;

  const dt_masks_state_t op = dp->state & DT_MASKS_STATE_OP;
  // if this empties src's group, keep it alive as an empty-group placeholder
  struct dt_masks_empty_group_t *emptied = _capture_emptied_group(grp, src);
  // first-class groups: preserve the partition across the move so the other
  // groups stay distinct even when operators coincide. The moved shape joins
  // the target group (adopts its key + operator).
  GHashTable *keys = _group_keys_snapshot(grp);
  g_hash_table_insert(keys, GINT_TO_POINTER(src),
                      g_hash_table_lookup(keys, GINT_TO_POINTER(dst)));
  grp->points = g_list_remove(grp->points, sp);
  // re-find the run extent (indices shifted after the removal) and land on top
  GList *dst_run2 = _selected_group_formids(grp, dst);
  int last = -1;
  const int firstidx = _run_extent(grp, dst_run2, &last);
  int at = (firstidx < 0) ? (int)g_list_length(grp->points) : last + 1;
  if(at < 1) at = 1; // never displace the base shape from the bottom
  sp->state = (sp->state & ~DT_MASKS_STATE_OP) | op;
  grp->points = g_list_insert(grp->points, sp, at);
  g_list_free(dst_run2);
  _group_keys_apply(grp, keys);
  g_hash_table_destroy(keys);

  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(emptied) bd->empty_groups = g_list_append(bd->empty_groups, emptied);
  _normalize_group_operators(grp);
  _verify_element_joined(grp, src, dst, "group-header drop");

  // a moved element should stay selected at the end of the drag
  bd->panel_selected_formid = src;
  dt_masks_form_t *sform = dt_masks_get_from_id(darktable.develop, src);
  bd->panel_selected_group_cid = (sform && !(sform->type & DT_MASKS_PARAMETRIC))
                                   ? _group_cid_of_form(grp, src)
                                   : INVALID_MASKID;
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
    GList *dst_ids = g_object_get_data(G_OBJECT(w), "group-formids");
    const dt_mask_id_t dst = dst_ids ? GPOINTER_TO_INT(dst_ids->data) : INVALID_MASKID;
    ok = _model_drop_element_onto_group(module, _module_mask_group(module), src, dst);
    if(ok)
    {
      dt_print(DT_DEBUG_MASKS, "[masks] shape %d moved into group of %d", src, dst);
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
  GList *dst_ids = g_object_get_data(G_OBJECT(w), "group-formids");
  const dt_mask_id_t dst = dst_ids ? GPOINTER_TO_INT(dst_ids->data) : INVALID_MASKID;
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

// defined near the other empty-group DnD handlers, after dt_masks_empty_group_t
// is declared; forward-declared here so the header dispatcher can route to it.
static void _masks_empty_reorder_drop(GtkWidget *w,
                                      GdkDragContext *ctx,
                                      gint y,
                                      GtkSelectionData *sel,
                                      guint time,
                                      dt_iop_module_t *module);

// a group header is a drop target for a whole real group (reorder), a single
// shape (move into the group), an empty group (reorder), or a whole cluster
// (move every member together). Route on the target entry info.
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
  else if(info == DND_MASK_EMPTY)
    _masks_empty_reorder_drop(w, ctx, y, sel, time, module);
  else if(info == DND_MASK_CLUSTER)
    _masks_cluster_to_group_drop(w, ctx, sel, time, module);
  else
    _masks_group_drag_received(w, ctx, x, y, sel, info, time, module);
}

// an element row (evbox/row_evbox, tagged with its own group's "group-formids"
// -- see _make_shape_row) is *also* a drop target for a whole group or empty
// group, not just a shape: otherwise only the thin header row would accept
// such a drop, and dragging a group over any of a target group's own elements
// -- easy to do by accident -- would be silently rejected. A shape dropped
// here still reorders precisely next to this row (_masks_row_drag_received),
// unlike a shape dropped on the header (which just lands on top of the group).
static void _element_row_drag_received(GtkWidget *w,
                                       GdkDragContext *ctx,
                                       gint x,
                                       gint y,
                                       GtkSelectionData *sel,
                                       guint info,
                                       guint time,
                                       dt_iop_module_t *module)
{
  if(info == DND_MASK_ROW)
    _masks_row_drag_received(w, ctx, x, y, sel, info, time, module);
  else if(info == DND_MASK_EMPTY)
    _masks_empty_reorder_drop(w, ctx, y, sel, time, module);
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
// find the props expand/collapse toggle button (see _make_props_row_toggle)
// tagged with "props-key" == key, searching only inside `root` (a single
// row's own widget subtree, not the whole panel -- cheap).
static GtkWidget *_find_props_toggle_in(GtkWidget *root, const dt_mask_id_t key)
{
  if(!root) return NULL;
  if(GTK_IS_TOGGLE_BUTTON(root) && g_object_get_data(G_OBJECT(root), "props-editor-box")
     && GPOINTER_TO_INT(g_object_get_data(G_OBJECT(root), "props-key")) == key)
    return root;
  if(!GTK_IS_CONTAINER(root)) return NULL;
  GtkWidget *found = NULL;
  GList *kids = gtk_container_get_children(GTK_CONTAINER(root));
  for(GList *c = kids; c && !found; c = g_list_next(c))
    found = _find_props_toggle_in(GTK_WIDGET(c->data), key);
  g_list_free(kids);
  return found;
}

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
static void _reveal_group_header(GtkWidget *w, const dt_mask_id_t gcid)
{
  if(!GTK_IS_CONTAINER(w)) return;
  GList *kids = gtk_container_get_children(GTK_CONTAINER(w));
  for(GList *c = kids; c; c = g_list_next(c))
  {
    GtkWidget *child = c->data;
    if(g_object_get_data(G_OBJECT(child), "mask-header"))
    {
      const dt_mask_id_t cid =
        (dt_mask_id_t)GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(child), "group-key"));
      if(cid == gcid)
      {
        GtkWidget *toggle = g_object_get_data(G_OBJECT(child), "group-expand-toggle");
        if(toggle && !gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)))
          gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle), TRUE);
      }
    }
    else
      _reveal_group_header(child, gcid);
  }
  g_list_free(kids);
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
}

// "auto-expand selected shape" option (masks panel hamburger -> options):
// while enabled, exactly one shape row -- the last-selected shape that
// actually has a props row -- is ever expanded (see _make_props_row_toggle's
// matching build-time rule, which reads bd->masks_last_expanded_shape, not
// panel_selected_formid directly). Selection itself only ever goes through
// the row's lightweight in-place updater (_update_row_selection), never a
// full rebuild, so this enforces the same invariant immediately.
//
// Selecting something without its own props toggle -- a parametric channel
// row, a group, an invalid/cleared selection -- is deliberately a no-op
// here: `id` is only ever a *candidate* replacement, and this bails out
// before touching anything if `id` itself has no props row to expand, so
// whatever shape was expanded before stays open instead of collapsing just
// because the user picked a non-shape element next (which would otherwise
// visibly shift the panel for no reason). Applies uniformly to every shape
// kind this panel shows, drawn or AI (DT_MASKS_OBJECT) alike -- both go
// through the exact same row/toggle construction (see _make_shape_row's
// generic "else" branch), keyed only by the shape's own form id, never by
// kind.
static void _auto_expand_selected_row(dt_iop_module_t *module, const dt_mask_id_t id)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!dt_conf_get_bool("plugins/darkroom/masks/auto_expand_selected")) return;

  GtkWidget *row = dt_is_valid_maskid(id) ? _masks_row_widget(bd, id) : NULL;
  GtkWidget *toggle = row ? _find_props_toggle_in(row, id) : NULL;
  if(!toggle)
    return; // `id` has no props row of its own -- leave the last-expanded shape alone

  if(bd->masks_last_expanded_shape == id
     && gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)))
    return; // already the one that's expanded

  // every gtk_toggle_button_set_active below is a programmatic enforcement
  // move, not a user click -- guarded by masks_suppress_toggle_select so
  // _props_row_toggled's own "toggling this row's expander also selects it"
  // behavior (meant for a real click) does not fire back into
  // _set_form_target -> _auto_expand_selected_row for the row being
  // collapsed here, which would re-select it and recurse without end (see
  // _props_row_toggled's own comment on this exact failure mode). Not
  // DT_ENTER/LEAVE_GUI_UPDATE: _props_row_toggled bails out entirely on
  // DT_IN_GUI_UPDATE(), which would also block the hash/visibility update
  // these calls are made for in the first place.
  bd->masks_suppress_toggle_select = TRUE;

  // collapse only the previously-expanded shape (there is at most one, by
  // construction) -- not "every other shape row": a row that was somehow
  // left expanded outside this mechanism is none of this function's
  // business, only the one it itself opened last.
  if(dt_is_valid_maskid(bd->masks_last_expanded_shape)
     && bd->masks_last_expanded_shape != id)
  {
    GtkWidget *prev_row = _masks_row_widget(bd, bd->masks_last_expanded_shape);
    GtkWidget *prev_toggle =
      prev_row ? _find_props_toggle_in(prev_row, bd->masks_last_expanded_shape) : NULL;
    if(prev_toggle && gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(prev_toggle)))
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(prev_toggle), FALSE);
  }

  _reveal_containers_for_row(module, row, id);
  if(!gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)))
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle), TRUE);
  bd->masks_last_expanded_shape = id;

  bd->masks_suppress_toggle_select = FALSE;
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
//   click it again      -> nothing selected
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
  const gboolean deselect = dt_is_valid_maskid(bd->panel_selected_group_cid)
                            && bd->panel_selected_group_cid == cid;
  s.group_cid = deselect ? INVALID_MASKID : cid;
  return s;
}

static void _set_form_target_ext(dt_iop_module_t *module,
                                 const dt_mask_id_t id,
                                 const gboolean auto_expand)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  _set_group_target(module, _group_cid_of_form(grp, id));
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
static gchar *_form_display_name(const dt_masks_form_t *form)
{
  const char *prefix = _form_type_prefix(form);
  const size_t plen = strlen(prefix);
  const char *rest = form->name;
  if(g_str_has_prefix(form->name, prefix)
     && (form->name[plen] == ' ' || form->name[plen] == '\0'))
  {
    rest = form->name + plen;
    while(*rest == ' ') rest++;
  }
  return g_strdup(rest);
}

static void _rename_commit(GtkWidget *entry, dt_iop_module_t *module)
{
  if(g_object_get_data(G_OBJECT(entry), "done")) return; // guard double commit
  g_object_set_data(G_OBJECT(entry), "done", GINT_TO_POINTER(1));
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(entry), "formid"));
  dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, id);
  gchar *txt = g_strdup(gtk_entry_get_text(GTK_ENTRY(entry)));
  if(txt) g_strstrip(txt);
  if(form && txt && *txt)
  {
    // the entry only edits the part after the type prefix (see
    // _row_click_press), so a rename replaces the auto-assigned "#<id>"
    // without ever dropping the "what is this" indication.
    g_snprintf(form->name, sizeof(form->name), "%s %s", _form_type_prefix(form), txt);
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
// delete paths (_delete_single_shape, _group_reset_members,
// _group_delete_shapes) used to clear these, so after deleting the
// individually-selected shape, _flexi_refine_follow_selection kept reading a
// "valid" (dt_is_valid_maskid) but now-nonexistent panel_selected_formid,
// landing refinement scope on ELEMENT for a shape that no longer resolves --
// seen as the refinement caption still naming the deleted shape and its
// controls reading as disabled, even though the group's own add-target
// selection (bd->selected_empty/panel_selected_group_cid) was unaffected and
// adding a new shape kept working. Called once per deleted formid.
static void _clear_stale_formid_refs(dt_iop_gui_blend_data_t *bd, const dt_mask_id_t id)
{
  if(!bd || !dt_is_valid_maskid(id)) return;
  if(bd->panel_selected_formid == id) bd->panel_selected_formid = INVALID_MASKID;
  if(bd->solo_formid == id) bd->solo_formid = INVALID_MASKID;
  if(bd->soloedit_formid == id) bd->soloedit_formid = INVALID_MASKID;
  // "auto-expand selected shape" (see _auto_expand_selected_row): a stale
  // reference here just means the option's next selection won't find
  // anything to collapse, harmless, but leaving it wrong would misreport
  // which row _make_props_row_toggle expands on the next full rebuild.
  if(bd->masks_last_expanded_shape == id) bd->masks_last_expanded_shape = NO_MASKID;
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

  int op = DT_MASKS_STATE_UNION;
  if(_group_sole_member(grp, id, &op))
  {
    // deleting the run's only remaining member would otherwise silently
    // collapse the whole group (dt_masks_form_remove drops an emptied
    // group entirely) -- leave an empty placeholder instead, same as
    // shift+right-click on the group header (see _group_reset_members,
    // which also does its own _clear_stale_formid_refs for `id`)
    GList *solo = g_list_prepend(NULL, GINT_TO_POINTER(id));
    _group_reset_members(module, solo, op);
    g_list_free(solo);
    return;
  }
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
  // survivor. _detach_group_members() touches only grp->points and re-stamps
  // the partition, with no reentry and no destruction cascade; the group's
  // last member is routed to _group_reset_members() above, so this branch
  // always leaves at least one member behind.
  GList *one = g_list_prepend(NULL, GINT_TO_POINTER(id));
  _detach_group_members(grp, one);
  g_list_free(one);
  dt_print(DT_DEBUG_MASKS, "[masks] form %d deleted from panel", id);
  dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  _queue_masks_list_rebuild(module);
  _refresh_canvas_edit(module);
}

#ifdef HAVE_AI
// "break into components": re-parent an AI-mask bundle's own children (see
// _register_vectorized_forms/object.c) as direct members of the containing
// module group, in the bundle's own place, preserving each child's own
// union/difference state; then drop the now-empty DT_MASKS_OBJECT wrapper.
// After this the former bundle members are ordinary independent shapes the
// user rearranges/edits by hand -- shared by the "break into components"
// entry in the row's actions menu (_build_shape_actions_menu).
static void _break_apart_ai_bundle(dt_iop_module_t *module, const dt_mask_id_t id)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  dt_masks_form_t *bundle = dt_masks_get_from_id(darktable.develop, id);
  if(!grp || !bundle || !(bundle->type & DT_MASKS_OBJECT) || !bundle->points) return;

  dt_masks_clear_form_gui(darktable.develop);

  // build the replacement points, one per child, in the bundle's own order
  GList *replacement = NULL;
  gboolean first = TRUE;
  for(GList *l = bundle->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *bpt = l->data;
    dt_masks_point_group_t *npt = calloc(1, sizeof(dt_masks_point_group_t));
    npt->formid = bpt->formid;
    npt->parentid = grp->formid;
    npt->state = DT_MASKS_STATE_SHOW | DT_MASKS_STATE_USE
                 | (bpt->state & DT_MASKS_STATE_DIFFERENCE ? DT_MASKS_STATE_DIFFERENCE
                                                           : DT_MASKS_STATE_UNION);
    npt->opacity = 1.0f;
    npt->group_opacity = 1.0f;
    // the whole run takes over the bundle's own former group-boundary marker
    // (only the first child continues it; the rest stay inside the same run)
    npt->group_start = first ? 1 : 0;
    first = FALSE;
    replacement = g_list_append(replacement, npt);
  }

  // splice `replacement` into grp->points where the bundle's own point was
  GList *pos = NULL;
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    if(pt->formid == id)
    {
      pos = l;
      break;
    }
  }
  if(pos)
  {
    for(GList *r = replacement; r; r = g_list_next(r))
      grp->points = g_list_insert_before(grp->points, pos, r->data);
    dt_masks_point_group_t *old_pt = pos->data;
    grp->points = g_list_remove(grp->points, old_pt);
    free(old_pt);
  }
  g_list_free(replacement);

  // drop the now-empty wrapper (its children are not touched -- they are
  // independent forms in dev->forms, now referenced directly by `grp`)
  darktable.develop->forms = g_list_remove(darktable.develop->forms, bundle);
  dt_masks_free_form(bundle);

  dt_print(DT_DEBUG_MASKS, "[masks] AI mask %d broken into components", id);
  dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  _queue_masks_list_rebuild(module);
  _refresh_canvas_edit(module);
}

static void _shape_menu_break_apart(GtkMenuItem *item, dt_iop_module_t *module)
{
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(item), "formid"));
  _break_apart_ai_bundle(module, id);
}
#endif // HAVE_AI

// forward declared here (defined much further down, near _build_shape_actions_menu's
// other caller) so _row_click_press's own right-click can open the same menu
// without reordering half the file.
static GtkWidget *_build_shape_actions_menu(dt_iop_module_t *module,
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
static void _shape_menu_closed(GtkWidget *menu, dt_iop_module_t *module)
{
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(menu), "formid"));
  dt_iop_gui_blend_data_t *bd = module->blend_data;
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
    GtkWidget *menu = _build_shape_actions_menu(module, id, handle, evbox);
    g_object_set_data(G_OBJECT(menu), "formid", GINT_TO_POINTER(id));
    g_signal_connect(G_OBJECT(menu), "hide", G_CALLBACK(_shape_menu_closed), module);
    gtk_menu_popup_at_pointer(GTK_MENU(menu), (GdkEvent *)ev);
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
// Every maximal run of consecutive same-operator points (a "group") gets its own
// header in the list (see _starts_group / _build_masks_list); same-kind runs
// within one group are separately folded into a collapsible kind-cluster
// expander to keep the list manageable when there are many (e.g. tens of)
// brush strokes (see cluster_min in _pack_group_elements).

#define MASK_GROUP_MIN 1 // unused: every operator-group always gets its own header

int _op_index_for_state(const int state)
{
  for(int i = 0; i < (int)(sizeof(_masks_ops) / sizeof(_masks_ops[0])); i++)
    if((state & DT_MASKS_STATE_OP_COMBINE) & _masks_ops[i].state) return i;
  return 0;
}

// ---- empty (staged) groups -------------------------------------------------
// A group with an operator but no members yet: the reset scaffold (add/intersect/
// subtract) and anything created by the "add group" button. They are UI-side
// state (an empty group has nothing to serialize); a shape drawn into one
// "realizes" it into a real operator-run. Each keeps its place via below_fid, a
// member id of the real run directly below it (INVALID = bottom of the list).
// dt_masks_empty_group_t now lives in blend_gui_internal.h -- masks_gui_presets.c
// needs its layout to capture/restore a group-layout preset.

// frees an empty group and its owned name -- use everywhere an empty group is
// discarded (free(eg) alone would leak eg->name) or as a GDestroyNotify
static void _empty_group_free(gpointer data)
{
  dt_masks_empty_group_t *eg = data;
  if(!eg) return;
  g_free(eg->name);
  free(eg);
}

// commit a group's rename entry -- handles both a populated group (its
// "group-cid" entry-data is valid, broadcasting the text onto every member of
// the run, see _group_custom_name) and a staged empty group (its "eg"
// entry-data is non-NULL, writing straight onto eg->name, adopted onto the
// run's member(s) once the group is realized -- see _masks_shape_to_empty_drop
// and the realize block in _build_masks_list). One shared path for both, so
// the two kinds cannot again drift apart the way they previously did (a
// missing ctrl+click release guard, a missing drag-source disarm, and
// eg->name being left out of _masks_list_signature's hash -- which silently
// ate Enter-to-commit for an empty group, since the rebuild that would swap
// the entry back for the label got skipped as a no-op change -- all three
// bugs existed in one copy and not the other before this consolidation).
static void _group_rename_commit(GtkWidget *entry, dt_iop_module_t *module)
{
  if(g_object_get_data(G_OBJECT(entry), "done")) return; // guard double commit
  g_object_set_data(G_OBJECT(entry), "done", GINT_TO_POINTER(1));
  dt_masks_empty_group_t *eg = g_object_get_data(G_OBJECT(entry), "eg");
  gchar *txt = g_strdup(gtk_entry_get_text(GTK_ENTRY(entry)));
  if(txt) g_strstrip(txt);
  if(eg)
  {
    dt_iop_gui_blend_data_t *bd = module->blend_data;
    if(g_list_find(bd->empty_groups, eg) && txt && *txt)
    {
      g_free(eg->name);
      eg->name = g_strdup(txt);
      dt_print(DT_DEBUG_MASKS, "[masks] empty group renamed to '%s'", txt);
    }
  }
  else
  {
    const dt_mask_id_t cid =
      GPOINTER_TO_INT(g_object_get_data(G_OBJECT(entry), "group-cid"));
    dt_masks_form_t *grp = _module_mask_group(module);
    if(grp && txt && *txt)
    {
      GList *ids = _selected_group_formids(grp, cid);
      for(GList *l = ids; l; l = g_list_next(l))
      {
        dt_masks_point_group_t *pt = _group_point(grp, GPOINTER_TO_INT(l->data));
        if(pt) g_strlcpy(pt->name, txt, sizeof(pt->name));
      }
      g_list_free(ids);
      dt_print(DT_DEBUG_MASKS, "[masks] group %d renamed to '%s'", cid, txt);
      dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
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

// see the forward declarations next to the REFINE_SCOPE_* enum
static dt_masks_refinement_t _empty_group_refinement(const void *eg)
{
  const dt_masks_empty_group_t *g = eg;
  dt_masks_refinement_t r = { 0 };
  if(g) r = g->refinement;
  return r;
}

static void _empty_group_set_refinement(void *eg, const dt_masks_refinement_t *r)
{
  dt_masks_empty_group_t *g = eg;
  if(g && r) g->refinement = *r;
}

static int _empty_group_op(const void *eg)
{
  const dt_masks_empty_group_t *g = eg;
  return g ? (int)g->op : 0;
}

static const char *_empty_group_name(const void *eg)
{
  const dt_masks_empty_group_t *g = eg;
  return (g && g->name && g->name[0]) ? g->name : NULL;
}

dt_masks_empty_group_t *_empty_group_new(const dt_masks_state_t op,
                                         const dt_masks_state_t within,
                                         const dt_mask_id_t below_fid)
{
  dt_masks_empty_group_t *eg = calloc(1, sizeof(dt_masks_empty_group_t));
  // an empty group renders nothing anyway, so it never carries the bypass
  // modifier -- emptying a bypassed group and refilling it gives a live group
  // of the same operator, rather than one that is mysteriously still disabled
  // with no chooser entry to re-enable it (the empty-group chooser omits bypass)
  eg->op = (op & DT_MASKS_STATE_OP_COMBINE) ? (op & DT_MASKS_STATE_OP_COMBINE)
                                            : DT_MASKS_STATE_UNION;
  eg->within = within & DT_MASKS_STATE_WITHIN;
  eg->below_fid = below_fid;
  // a brand-new group always starts fully opaque (see dt_masks_gui_form_save_creation);
  // overwritten by _flexi_layout_apply when the group comes from a saved preset
  eg->opacity = 1.0f;
  return eg;
}

static void _empty_groups_clear(dt_iop_gui_blend_data_t *bd)
{
  g_list_free_full(bd->empty_groups, _empty_group_free);
  bd->empty_groups = NULL;
  bd->selected_empty = NULL;
}

// dev->forms/history was rewritten wholesale from under the panel (undo/redo,
// jump to a history step, style paste, snapshot restore, compress history --
// see dt_dev_reload_history_items, the only caller). bd->empty_groups is
// GUI-only scratch state, never part of what got saved/restored: an empty
// group left behind by e.g. _group_reset_members (deleting a run's last
// member keeps an empty placeholder rather than collapsing the group) has no
// counterpart in the reloaded history, so after undoing that exact deletion
// the restored real group and the still-there placeholder would render as two
// rows for the same group. There is no way to tell which placeholders are
// still valid against a wholesale reload, so drop them all -- losing an
// unfilled placeholder group is a much smaller surprise than a phantom
// duplicate.
void dt_iop_gui_blend_forms_reloaded(dt_iop_module_t *module)
{
  if(!module) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd) return;
  // only a module that actually had placeholders to reconcile, or is
  // currently showing the flexi list at all, has anything for the rebuild
  // below to fix -- reload rewrites every module's dev->forms wholesale
  // (undo/redo, jump to history step, style paste, snapshot restore,
  // compress history), but the overwhelming majority of modules were never
  // in flexi mode and never had an empty-group placeholder, so queuing a
  // full masks-panel teardown+rebuild for all of them regardless was pure
  // waste: on a single undo this fired for every module in the pipeline
  // (~70 on a typical default pipeline), even though only one or two had
  // actually changed, and the resulting burst of simultaneous panel
  // rebuilds was observed to perturb the right panel's scroll position.
  const gboolean had_empties = bd->empty_groups != NULL;
  const gboolean flexi =
    module->blend_params && (module->blend_params->mask_mode & DEVELOP_MASK_FLEXI);
  _empty_groups_clear(bd);
  bd->insert_empty = NULL;
  bd->scaffold_seeded = FALSE;
  bd->masks_selection_seeded = FALSE;
  if(had_empties || flexi)
  {
    bd->masks_list_sig = DT_INVALID_HASH;
    if(bd->masks_list_box) _queue_masks_list_rebuild(module);
  }
}

// If `src` is the SOLE member of its group, removing it would leave the group empty.
// Rather than let the group vanish, build an empty-group placeholder carrying the
// group's operator, screen flag and anchor (the member directly below it, INVALID =
// bottom), so the group keeps its place in the list and the user can move shapes
// back into it. Returns NULL when src's group has other members (it survives the
// move). Reads the pre-move layout, so call it BEFORE removing src.
struct dt_masks_empty_group_t *_capture_emptied_group(dt_masks_form_t *grp,
                                                             const dt_mask_id_t src)
{
  GList *node = NULL;
  for(GList *l = grp ? grp->points : NULL; l; l = g_list_next(l))
    if(((dt_masks_point_group_t *)l->data)->formid == src)
    {
      node = l;
      break;
    }
  if(!node) return NULL;
  // src must be its run's head (nothing below in the same group) AND have nothing
  // above it in the same group -> it is the only member
  if(!_starts_group(node)) return NULL;
  if(node->next && !_starts_group(node->next)) return NULL;
  const dt_masks_point_group_t *sp = node->data;
  const dt_mask_id_t below =
    node->prev ? ((dt_masks_point_group_t *)node->prev->data)->formid : INVALID_MASKID;
  dt_masks_empty_group_t *eg =
    _empty_group_new(sp->state, sp->state & DT_MASKS_STATE_WITHIN, below);
  if(sp->name[0]) eg->name = g_strdup(sp->name);
  return eg;
}

// generalizes the above to a whole set of ids at once (a dragged cluster's
// members, which always belong to the same run -- see _pack_group_elements):
// if removing exactly this set would empty the run that contains them, capture
// it as an empty-group placeholder the same way a single-shape removal does,
// so the group doesn't just vanish. Returns NULL if some other member of that
// run is not in `ids` (the group survives) or `ids` is empty/unresolvable.
static struct dt_masks_empty_group_t *_capture_emptied_group_multi(dt_masks_form_t *grp,
                                                                   GList *ids)
{
  if(!ids) return NULL;
  GList *node = NULL;
  const dt_mask_id_t first_id = GPOINTER_TO_INT(ids->data);
  for(GList *l = grp ? grp->points : NULL; l; l = g_list_next(l))
    if(((dt_masks_point_group_t *)l->data)->formid == first_id)
    {
      node = l;
      break;
    }
  if(!node) return NULL;
  GList *lo = node;
  while(!_starts_group(lo)) lo = lo->prev;
  for(GList *l = lo; l; l = g_list_next(l))
  {
    if(l != lo && _starts_group(l)) break;
    const dt_mask_id_t fid = ((dt_masks_point_group_t *)l->data)->formid;
    gboolean found = FALSE;
    for(GList *m = ids; m; m = g_list_next(m))
      if(GPOINTER_TO_INT(m->data) == fid)
      {
        found = TRUE;
        break;
      }
    if(!found) return NULL; // some member of this run stays behind -- it survives
  }
  const dt_masks_point_group_t *headpt = lo->data;
  const dt_mask_id_t below =
    lo->prev ? ((dt_masks_point_group_t *)lo->prev->data)->formid : INVALID_MASKID;
  dt_masks_empty_group_t *eg =
    _empty_group_new(headpt->state, headpt->state & DT_MASKS_STATE_WITHIN, below);
  if(headpt->name[0]) eg->name = g_strdup(headpt->name);
  return eg;
}

// move every member of a dragged cluster together, preserving their relative
// (bottom-up) order, to the position/group a drop indicates -- the same move
// _masks_row_drag_received / _masks_shape_to_group_drop do for one shape,
// generalized to a same-kind run's whole member set. `dst` is either the
// target group's own head formid (dst_is_group) or the target row's own
// formid (drop lands directly above/below it, per `above`). Returns FALSE
// (no-op) if `member_ids` is empty or `dst` is itself one of the members.
gboolean _masks_cluster_move(dt_iop_module_t *module,
                                    GList *member_ids,
                                    const dt_mask_id_t dst,
                                    const gboolean dst_is_group,
                                    const gboolean above)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp || !member_ids) return FALSE;

  for(GList *l = member_ids; l; l = g_list_next(l))
    if(GPOINTER_TO_INT(l->data) == dst) return FALSE;

  // recover the cluster's own relative order from grp->points (the DnD payload
  // itself carries no meaningful order, see _masks_cluster_drag_get)
  GList *ordered = NULL;
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    const dt_mask_id_t fid = ((dt_masks_point_group_t *)l->data)->formid;
    for(GList *m = member_ids; m; m = g_list_next(m))
      if(GPOINTER_TO_INT(m->data) == fid)
      {
        ordered = g_list_append(ordered, l->data);
        break;
      }
  }
  if(!ordered) return FALSE;

  struct dt_masks_empty_group_t *emptied = _capture_emptied_group_multi(grp, member_ids);

  // preserve every OTHER group's partition (same trick as a single-shape move):
  // snapshot every point's group key, then remap every moved member's key to
  // the destination's, and re-stamp GROUP_BREAK from that map once the points
  // have actually been relocated.
  GHashTable *keys = _group_keys_snapshot(grp);
  const gpointer dkey = g_hash_table_lookup(keys, GINT_TO_POINTER(dst));
  for(GList *l = ordered; l; l = g_list_next(l))
    g_hash_table_insert(
      keys, GINT_TO_POINTER(((dt_masks_point_group_t *)l->data)->formid), dkey);

  const dt_masks_point_group_t *dpt = _group_point(grp, dst);
  const dt_masks_state_t dst_op =
    dpt ? (dpt->state & DT_MASKS_STATE_OP) : DT_MASKS_STATE_UNION;

  for(GList *l = ordered; l; l = g_list_next(l))
    grp->points = g_list_remove(grp->points, l->data);

  int at;
  if(dst_is_group)
  {
    GList *dst_run = _selected_group_formids(grp, dst);
    int last = -1;
    const int firstidx = _run_extent(grp, dst_run, &last);
    at = (firstidx < 0) ? (int)g_list_length(grp->points) : last + 1;
    g_list_free(dst_run);
  }
  else
  {
    int tgt = 0, idx = 0;
    for(GList *l = grp->points; l; l = g_list_next(l), idx++)
      if(((dt_masks_point_group_t *)l->data)->formid == dst)
      {
        tgt = idx;
        break;
      }
    at = above ? tgt + 1 : tgt;
  }
  if(at < 1) at = 1; // never displace the base shape from the bottom

  int pos = at;
  for(GList *l = ordered; l; l = g_list_next(l), pos++)
  {
    dt_masks_point_group_t *pt = l->data;
    pt->state = (pt->state & ~DT_MASKS_STATE_OP) | dst_op;
    grp->points = g_list_insert(grp->points, pt, pos);
  }
  g_list_free(ordered);

  _group_keys_apply(grp, keys);
  g_hash_table_destroy(keys);
  if(emptied) bd->empty_groups = g_list_append(bd->empty_groups, emptied);
  _normalize_group_operators(grp);
  return TRUE;
}

// total number of groups (real operator-runs + empty groups). Used to decide
// whether a default target exists: with a single group new elements land in it
// automatically; with several, one must be explicitly selected.
static int _group_count(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  int n = g_list_length(bd->empty_groups);
  GList *l = grp ? grp->points : NULL;
  while(l)
  {
    n++;
    GList *p = g_list_next(l);
    while(p && !_starts_group(p)) p = g_list_next(p);
    l = p;
  }
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
typedef struct dt_masks_add_target_t
{
  dt_masks_empty_group_t *empty; // staged (member-less) group, or NULL
  dt_mask_id_t cid;              // real group's cid, or INVALID_MASKID
  gboolean valid;
  gboolean implicit; // resolved from "only one group", not a selection
} dt_masks_add_target_t;

static dt_masks_add_target_t _resolve_add_target(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  dt_masks_add_target_t t = { NULL, INVALID_MASKID, FALSE, FALSE };

  if(bd->selected_empty)
  {
    t.empty = bd->selected_empty;
    t.valid = TRUE;
  }
  else if(dt_is_valid_maskid(bd->panel_selected_group_cid) && grp
          && _group_point(grp, bd->panel_selected_group_cid))
  {
    t.cid = bd->panel_selected_group_cid;
    t.valid = TRUE;
  }
  else if(_group_count(module) == 1)
  {
    // the sole group is either the one staged group or the one real run, whose
    // cid is its first point in grp->points order -- the same convention
    // _build_masks_list uses for group headers (see _group_cid_of_form)
    if(bd->empty_groups)
      t.empty = bd->empty_groups->data;
    else if(grp && grp->points)
      t.cid = ((const dt_masks_point_group_t *)grp->points->data)->formid;
    t.valid = t.empty != NULL || dt_is_valid_maskid(t.cid);
    t.implicit = t.valid;
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
  if(bd->masks_refine_scope_kind == REFINE_SCOPE_EMPTY_GROUP)
  {
    active = FALSE;
  }
  else if(bd->masks_refine_scope_kind == REFINE_SCOPE_GROUP)
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
  gtk_widget_set_sensitive(bd->masks_combo, has_target);
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
  // own disabled-state tooltip stays reachable
  if(bd->masks_param_channels_inner)
    for(GList *l =
          gtk_container_get_children(GTK_CONTAINER(bd->masks_param_channels_inner));
        l; l = g_list_delete_link(l, l))
    {
      gtk_widget_set_sensitive(GTK_WIDGET(l->data), has_target);
      _append_tooltip_hint(GTK_WIDGET(l->data), hint);
    }

  // raster and import/reuse also add an element to the target group, so they
  // need the same target and the same explanation when there isn't one
  if(bd->masks_raster_add_btn)
  {
    gtk_widget_set_sensitive(bd->masks_raster_add_btn, has_target);
    gchar *tt =
      g_strconcat(_("add a raster mask element: use another module's mask as an element\n"
                    "of this group, combined with the group's operator"),
                  hint, NULL);
    gtk_widget_set_tooltip_text(bd->masks_raster_add_btn, tt);
    g_free(tt);
  }
  if(bd->masks_import_btn)
  {
    gtk_widget_set_sensitive(bd->masks_import_btn, has_target);
    gchar *tt = g_strconcat(_("import an existing shape, or reuse another\n"
                              "module's mask (click to pick one)"),
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
  bd->insert_op = 0;
  bd->insert_within = 0;
  bd->insert_realize_empty = FALSE;
  bd->insert_empty = NULL;
  bd->insert_opacity = 1.0f;

  const dt_masks_add_target_t target = _resolve_add_target(module);

  if(target.empty)
  {
    const dt_masks_empty_group_t *eg = target.empty;
    bd->insert_active = TRUE;
    bd->insert_realize_empty = TRUE;
    bd->insert_op = eg->op;
    bd->insert_within = eg->within;
    bd->insert_opacity = eg->opacity;
    // land just above the run anchored below this empty group (its top member)
    if(grp && dt_is_valid_maskid(eg->below_fid))
    {
      GList *run = _selected_group_formids(grp, eg->below_fid);
      if(run) bd->insert_after_fid = GPOINTER_TO_INT(run->data); // head = top member
      g_list_free(run);
    }
  }
  else if(dt_is_valid_maskid(target.cid) && grp)
  {
    const dt_masks_point_group_t *pt = _group_point(grp, target.cid);
    if(pt)
    {
      bd->insert_active = TRUE;
      bd->insert_op = pt->state & DT_MASKS_STATE_OP;
      GList *run = _selected_group_formids(grp, target.cid);
      if(run)
      {
        bd->insert_after_fid = GPOINTER_TO_INT(run->data); // top member of the run
        // adopt the run's within-group combine mode (screen/intersect/union), but
        // only if every member already agrees on it; else fall back to union (0)
        const dt_masks_point_group_t *h = _group_point(grp, GPOINTER_TO_INT(run->data));
        const dt_masks_state_t within = h ? (h->state & DT_MASKS_STATE_WITHIN) : 0;
        bd->insert_within = within;
        for(GList *l = run; l; l = g_list_next(l))
        {
          const dt_masks_point_group_t *m = _group_point(grp, GPOINTER_TO_INT(l->data));
          if(!m || (m->state & DT_MASKS_STATE_WITHIN) != within)
          {
            bd->insert_within = 0;
            break;
          }
        }
        g_list_free(run);
      }
    }
  }
}

// the group id (selectable group identity) of the run containing form `fid`:
// the run's bottom member (its first point in grp->points order), matching how
// _build_masks_list keys group headers. INVALID if the form is not found.
dt_mask_id_t _group_cid_of_form(dt_masks_form_t *grp, const dt_mask_id_t fid)
{
  GList *run = _selected_group_formids(grp, fid);
  dt_mask_id_t cid = INVALID_MASKID;
  if(run)
  {
    cid = GPOINTER_TO_INT(g_list_last(run)->data);
    g_list_free(run);
  }
  return cid;
}

// -------- unified group / empty-group reorder --------
// below_fid alone cannot express every reorder a drag-and-drop gesture might
// ask for: an empty group's anchor is a formid of some real run, so it keeps
// tracking that run wherever the run's own position ends up -- there is no
// way to encode "this empty is now below the very run it is anchored to"
// without changing which run it is anchored to. Rather than special-case that
// (and every other combination -- empty past empty, real past its own empty,
// etc.), every drag/drop between two groups (real or empty) is resolved by
// building the single bottom-up list of every group exactly as
// _build_masks_list renders it, splicing the dragged one to its new position
// in that list, and re-deriving grp->points' run order + every empty's
// below_fid/relative order from the result. This always lets the user freely
// rearrange groups, whatever the current anchoring happens to be.

// one group (real run or empty group) in the unified bottom-up visual order.
// bottom-up list of every group (real run or empty group) in current visual
// order, exactly mirroring the packing order _build_masks_list uses. Caller
// frees every item and the list (g_list_free_full(..., g_free)).
GList *_masks_visual_group_order(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  GList *out = NULL;

  // bottom_empties: unanchored (or dangling-anchor) empties, in list order
  for(GList *e = bd->empty_groups; e; e = g_list_next(e))
  {
    dt_masks_empty_group_t *eg = e->data;
    const gboolean anchored =
      grp && dt_is_valid_maskid(eg->below_fid) && _group_point(grp, eg->below_fid);
    if(!anchored)
    {
      _dt_masks_order_item_t *it = g_malloc0(sizeof(_dt_masks_order_item_t));
      it->is_empty = TRUE;
      it->eg = eg;
      out = g_list_append(out, it);
    }
  }

  // groups pass: one item per real run (bottom-up), followed by any empties
  // anchored onto it (list order)
  GList *heads = _group_partition_heads(grp);
  for(GList *h = heads; h; h = g_list_next(h))
  {
    const dt_mask_id_t cid = GPOINTER_TO_INT(h->data);
    _dt_masks_order_item_t *it = g_malloc0(sizeof(_dt_masks_order_item_t));
    it->is_empty = FALSE;
    it->cid = cid;
    out = g_list_append(out, it);

    GList *run = _selected_group_formids(grp, cid);
    for(GList *e = bd->empty_groups; e; e = g_list_next(e))
    {
      dt_masks_empty_group_t *eg = e->data;
      gboolean match = FALSE;
      for(GList *m = run; m; m = g_list_next(m))
        if(GPOINTER_TO_INT(m->data) == eg->below_fid)
        {
          match = TRUE;
          break;
        }
      if(match)
      {
        _dt_masks_order_item_t *eit = g_malloc0(sizeof(_dt_masks_order_item_t));
        eit->is_empty = TRUE;
        eit->eg = eg;
        out = g_list_append(out, eit);
      }
    }
    g_list_free(run);
  }
  g_list_free(heads);
  return out;
}

// move the group identified by (src_is_empty, src_cid, src_eg) to sit directly
// above (`above` TRUE) or below (FALSE) the group identified by (dst_is_empty,
// dst_cid, dst_eg), in the unified bottom-up order (see _masks_visual_group_order),
// then re-derive grp->points' run order and every empty group's below_fid/
// relative order from the result. Returns FALSE (no-op) if src and dst are the
// same group or either is not found. For a real group, *_cid must be the run's
// own head formid (see _group_cid_of_form) -- any other member id will not match.
gboolean _masks_reorder_groups(dt_iop_module_t *module,
                                      const gboolean src_is_empty,
                                      const dt_mask_id_t src_cid,
                                      dt_masks_empty_group_t *src_eg,
                                      const gboolean dst_is_empty,
                                      const dt_mask_id_t dst_cid,
                                      dt_masks_empty_group_t *dst_eg,
                                      const gboolean above)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp) return FALSE;

  GList *order = _masks_visual_group_order(module);

  GList *src_node = NULL, *dst_node = NULL;
  for(GList *l = order; l; l = g_list_next(l))
  {
    _dt_masks_order_item_t *it = l->data;
    if(src_is_empty ? (it->is_empty && it->eg == src_eg)
                    : (!it->is_empty && it->cid == src_cid))
      src_node = l;
    if(dst_is_empty ? (it->is_empty && it->eg == dst_eg)
                    : (!it->is_empty && it->cid == dst_cid))
      dst_node = l;
  }
  if(!src_node || !dst_node || src_node == dst_node)
  {
    g_list_free_full(order, g_free);
    return FALSE;
  }

  _dt_masks_order_item_t *src_item = src_node->data;
  order = g_list_remove_link(order, src_node);
  order = above ? g_list_insert_before(order, dst_node->next, src_item)
                : g_list_insert_before(order, dst_node, src_item);
  g_list_free_1(src_node);

  // real run order: concatenate each run's own points (already bottom-up) in
  // the new relative order
  GList *heads_order = NULL;
  for(GList *l = order; l; l = g_list_next(l))
  {
    _dt_masks_order_item_t *it = l->data;
    if(!it->is_empty) heads_order = g_list_append(heads_order, GINT_TO_POINTER(it->cid));
  }
  GList *new_points = NULL;
  for(GList *l = heads_order; l; l = g_list_next(l))
  {
    // _selected_group_formids returns top-member-first (see its own callers,
    // e.g. "top member of the run" above) -- grp->points is bottom-up, so the
    // run must be walked in reverse here, or the run's own members land
    // reversed within grp->points. That put the true (bottom) head somewhere
    // above the run's start, and _apply_partition_breaks below then stamped a
    // break there -- splitting one dragged group into two on every reorder.
    GList *run = _selected_group_formids(grp, GPOINTER_TO_INT(l->data));
    run = g_list_reverse(run);
    for(GList *m = run; m; m = g_list_next(m))
    {
      dt_masks_point_group_t *pt = _group_point(grp, GPOINTER_TO_INT(m->data));
      if(pt) new_points = g_list_append(new_points, pt);
    }
    g_list_free(run);
  }
  g_list_free(grp->points); // frees only the list spine, not the point objects
  grp->points = new_points;
  _apply_partition_breaks(grp, heads_order);
  g_list_free(heads_order);
  _normalize_group_operators(grp);

  // empty groups: preserve relative order, recompute each one's below_fid by
  // scanning downward (toward the bottom) from its new position for the
  // nearest real item
  GList *new_empties = NULL;
  for(GList *l = order; l; l = g_list_next(l))
  {
    _dt_masks_order_item_t *it = l->data;
    if(!it->is_empty) continue;
    dt_mask_id_t below = INVALID_MASKID;
    for(GList *b = l->prev; b; b = g_list_previous(b))
    {
      _dt_masks_order_item_t *bit = b->data;
      if(!bit->is_empty)
      {
        below = bit->cid;
        break;
      }
    }
    it->eg->below_fid = below;
    new_empties = g_list_append(new_empties, it->eg);
  }
  g_list_free(bd->empty_groups);
  bd->empty_groups = new_empties;

  g_list_free_full(order, g_free);
  return TRUE;
}

// 1-based ordinal of a group within its OWN operator mode, counting bottom->top
// across BOTH real runs and empty groups (they interleave exactly as the render
// loop packs them: bottom-anchored empties first, then each real run bottom-up
// with the empties anchored onto it). Numbering is per-operator, so groups read as
// "union 1", "difference 1", "difference 2", ... Matches against a real run head
// `cid` OR an empty group `eg` (pass INVALID_MASKID / NULL for the one not looked
// up). Groups keep an id even while empty. Returns 0 if not found.
// highest number currently held by a live group of this operator (0 if none).
// A new group takes one past this, so a number is never handed out while a peer
// still shows it, and a series restarts at 1 once its last group is gone.
int _group_ord_max_for_op(dt_iop_module_t *module, const int opidx)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  int mx = 0;

  for(GList *e = bd->empty_groups; e; e = g_list_next(e))
  {
    const dt_masks_empty_group_t *g = e->data;
    if(_op_index_for_state(g->op) == opidx && g->ordinal > mx) mx = g->ordinal;
  }

  if(bd->group_ordinals)
    for(GList *l = grp ? grp->points : NULL; l; l = g_list_next(l))
    {
      if(!_starts_group(l)) continue;
      const dt_masks_point_group_t *head = l->data;
      if(_op_index_for_state(head->state) != opidx) continue;
      const int ord = GPOINTER_TO_INT(
        g_hash_table_lookup(bd->group_ordinals, GINT_TO_POINTER(head->formid)));
      if(ord > mx) mx = ord;
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
  dt_masks_form_t *grp = _module_mask_group(module);
  gboolean live = FALSE;
  for(GList *l = grp ? grp->points : NULL; l; l = g_list_next(l))
    if(_starts_group(l)
       && (guint)((dt_masks_point_group_t *)l->data)->formid == bd->solo_group_key)
    {
      live = TRUE;
      break;
    }
  if(!live) bd->solo_group_key = 0;
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
    const dt_mask_id_t cid = GPOINTER_TO_INT(k);
    gboolean live = FALSE;
    for(GList *l = grp ? grp->points : NULL; l; l = g_list_next(l))
      if(_starts_group(l) && ((dt_masks_point_group_t *)l->data)->formid == cid)
      {
        live = TRUE;
        break;
      }
    if(!live) g_hash_table_iter_remove(&it);
  }
}

/* The group's displayed number. This is a remembered identity, assigned once
   and kept for as long as the group exists -- NOT a positional count. Numbering
   groups by position meant deleting one renumbered every survivor above it, so
   removing union-1 turned union-2 into union-1 and read as though the wrong
   group had been deleted.

   Real groups keep their number in bd->group_ordinals (keyed by cid); empty
   groups keep theirs in the struct. The number is carried across the empty <->
   real transitions (see _group_reset_members and the two realize paths), so
   emptying and refilling a group does not renumber it either. */
static int _group_ordinal_any(dt_iop_module_t *module,
                              const dt_mask_id_t cid,
                              const dt_masks_empty_group_t *eg)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;

  if(eg)
  {
    dt_masks_empty_group_t *g = (dt_masks_empty_group_t *)eg;
    if(g->ordinal <= 0)
      g->ordinal = _group_ord_max_for_op(module, _op_index_for_state(g->op)) + 1;
    return g->ordinal;
  }

  if(!dt_is_valid_maskid(cid)) return 0;
  const dt_masks_point_group_t *head = _group_point(_module_mask_group(module), cid);
  if(!head) return 0;

  if(!bd->group_ordinals)
    bd->group_ordinals = g_hash_table_new(g_direct_hash, g_direct_equal);

  int ord =
    GPOINTER_TO_INT(g_hash_table_lookup(bd->group_ordinals, GINT_TO_POINTER(cid)));
  if(ord <= 0)
  {
    ord = _group_ord_max_for_op(module, _op_index_for_state(head->state)) + 1;
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
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);

  // bottom-anchored empties (no live anchor) render at the very bottom
  for(GList *e = bd->empty_groups; e; e = g_list_next(e))
  {
    dt_masks_empty_group_t *g = e->data;
    const gboolean anchored =
      grp && dt_is_valid_maskid(g->below_fid) && _group_point(grp, g->below_fid);
    if(!anchored) _group_ordinal_any(module, INVALID_MASKID, g);
  }

  // each real run bottom-up, then the empties anchored onto it
  GList *l = grp ? grp->points : NULL;
  while(l)
  {
    const dt_masks_point_group_t *head = l->data;
    GList *members = NULL;
    GList *p = l;
    while(p)
    {
      const dt_masks_point_group_t *m = p->data;
      if(p != l && _starts_group(p)) break;
      members = g_list_prepend(members, GINT_TO_POINTER(m->formid));
      p = g_list_next(p);
    }
    _group_ordinal_any(module, head->formid, NULL);
    for(GList *e = bd->empty_groups; e; e = g_list_next(e))
    {
      dt_masks_empty_group_t *g = e->data;
      for(GList *mm = members; mm; mm = g_list_next(mm))
        if(GPOINTER_TO_INT(mm->data) == g->below_fid)
        {
          _group_ordinal_any(module, INVALID_MASKID, g);
          break;
        }
    }
    g_list_free(members);
    l = p;
  }
}

// 1-based per-operator ordinal of the real group whose head formid == cid (see
// _group_ordinal_any).
int _group_ordinal_of_cid(dt_iop_module_t *module, const dt_mask_id_t cid)
{
  return _group_ordinal_any(module, cid, NULL);
}

// the head formid of the run directly below the run headed by `cid` (i.e. the
// next group down the stack), or INVALID_MASKID if cid's own run is already
// the base (bottom) group. `cid` must be a run's own head (bottom-most member),
// as every caller of this already has (_group_cid_of_form / panel_selected_group_cid
// always store the head). Used by _stage_new_group's "add below" placement.
static dt_mask_id_t _group_below_cid(dt_masks_form_t *grp, const dt_mask_id_t cid)
{
  if(!grp) return INVALID_MASKID;
  GList *node = NULL;
  for(GList *l = grp->points; l; l = g_list_next(l))
    if(((dt_masks_point_group_t *)l->data)->formid == cid)
    {
      node = l;
      break;
    }
  if(!node || !node->prev) return INVALID_MASKID;
  GList *lo = node->prev;
  while(!_starts_group(lo)) lo = lo->prev;
  return ((dt_masks_point_group_t *)lo->data)->formid;
}

// "add group": create an empty group and select it, anchored above or below
// the current target -- a real group, an empty group, or (nothing selected)
// above everything / at the very bottom -- depending on `below_target`. The
// add-group icon adopts the chosen operator (the one explicit place the icon
// is repainted from a user action).
static void
_stage_new_group(dt_iop_module_t *module, const int op_state, const gboolean below_target)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  dt_masks_state_t op = (op_state & DT_MASKS_STATE_OP) ? (op_state & DT_MASKS_STATE_OP)
                                                       : DT_MASKS_STATE_UNION;
  dt_mask_id_t below = INVALID_MASKID;
  GList *after = NULL; // node in empty_groups to insert the new one next to
  GList *before_node =
    NULL; // ...or to insert it *before*, see the "above a real group" case
  gboolean prepend_bottom = FALSE; // render as the bottom-most anchor-missing entry

  if(bd->selected_empty)
  {
    const dt_masks_empty_group_t *sel = bd->selected_empty;
    below = sel->below_fid; // share the anchor -- above/below only changes which
    after = g_list_find(bd->empty_groups, sel); // side of `sel` it lands on
  }
  else if(dt_is_valid_maskid(bd->panel_selected_group_cid))
  {
    below = below_target
              ? _group_below_cid(grp, bd->panel_selected_group_cid) // directly below it
              : bd->panel_selected_group_cid;                       // above it

    if(!below_target)
    {
      // "above the selected group" has to mean *immediately* above it, but the
      // anchor alone does not say that: several empty groups can share one
      // anchor, and _masks_panel_pack renders them in bd->empty_groups order
      // with gtk_box_pack_end -- so later in the list renders higher. Appending
      // therefore placed the new group above every empty already anchored to
      // this run, which with one such empty present landed it at the very top
      // of the panel instead of directly above the selected group.
      //
      // Going in before the first of them puts it lowest of that anchor's
      // empties, i.e. directly above the run. (The below_target case wants the
      // opposite -- directly *below* the selected group is the highest slot of
      // the run beneath it -- which is what appending already gives.)
      for(GList *e = bd->empty_groups; e; e = g_list_next(e))
      {
        const dt_masks_empty_group_t *eg_at = e->data;
        if(!dt_is_valid_maskid(eg_at->below_fid)) continue;
        // compare runs, not raw formids: an empty may be anchored to any member
        // of the run (see the match test in _masks_panel_pack), not just its head
        if(_group_cid_of_form(grp, eg_at->below_fid) == bd->panel_selected_group_cid)
        {
          before_node = e;
          break;
        }
      }
    }
  }
  else if(below_target)
  {
    // nothing selected: land below everything, same anchor as an empty list,
    // regardless of whatever groups already exist above it
    prepend_bottom = TRUE;
  }
  else if(grp && grp->points)
  {
    GList *last = g_list_last(grp->points); // above the topmost real run
    if(last) below = ((dt_masks_point_group_t *)last->data)->formid;
  }

  dt_masks_empty_group_t *eg = _empty_group_new(op, 0, below);
  if(prepend_bottom)
    // prepend so it renders as the bottom-most anchor-missing entry
    bd->empty_groups = g_list_prepend(bd->empty_groups, eg);
  else if(after)
    // same anchor as the selected empty group: insert on whichever side of it
    // renders above (after `after`) or below (before `after`) it
    bd->empty_groups = below_target
                         ? g_list_insert_before(bd->empty_groups, after, eg)
                         : g_list_insert_before(bd->empty_groups, after->next, eg);
  else if(before_node)
    // directly above the selected real group, below any empties already
    // anchored to it (see the loop that found this node)
    bd->empty_groups = g_list_insert_before(bd->empty_groups, before_node, eg);
  else
    bd->empty_groups = g_list_append(bd->empty_groups, eg);

  bd->panel_selected_formid = INVALID_MASKID;
  bd->panel_selected_group_cid = INVALID_MASKID;
  bd->selected_empty = eg;
  // the user explicitly chose this operator for the add-group button: update the icon
  bd->masks_new_group_op = op;
  if(bd->masks_new_op) _new_shape_op_update(bd->masks_new_op);
  dt_print(DT_DEBUG_MASKS, "[masks] add empty group op=0x%x below=%d below_target=%d", op,
           below, below_target);
  _build_masks_list(module);
}

// select an empty group (the next drawn shape realizes it). Clicking the already
// selected empty group deselects it (selection toggles), so the user can reach a
// state with no group selected.
static void _select_empty_group(dt_iop_module_t *module, dt_masks_empty_group_t *eg)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  // any open parametric editor stays open across a group-target change (see
  // _select_group); _build_masks_list re-docks it under its row itself.
  const gboolean deselect = (bd->selected_empty == eg);
  bd->panel_selected_formid = INVALID_MASKID;
  bd->panel_selected_group_cid = INVALID_MASKID;
  bd->selected_empty = deselect ? NULL : eg;
  bd->masks_shown = DT_MASKS_EDIT_FULL;
  dt_masks_set_edit_mode(module, DT_MASKS_EDIT_FULL);
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
// at cid (INVALID to clear it), then update the header/row highlight and the
// dependent controls (add-element sensitivity, refinement scope combo) in place
// -- no list rebuild, so this never disturbs the GTK focus chain and never
// triggers the containing scrolled viewport to auto-scroll to a re-created
// widget (see _select_group / _select_form / _param_enter_edit, all of which
// funnel group selection through here instead of _build_masks_list).
static void _set_group_target(dt_iop_module_t *module, const dt_mask_id_t cid)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  bd->panel_selected_formid = INVALID_MASKID;
  bd->selected_empty = NULL;
  bd->panel_selected_group_cid = cid;
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
}

// flexi only: keep the insertion hint (where the next drawn shape lands) in step
// with the current selection on the no-rebuild selection path. The add-group icon
// is intentionally NOT touched here -- it only changes when the user picks an
// operator from the add-group menu.
static void _flexi_new_op_follow_selection(dt_iop_gui_blend_data_t *bd)
{
  if(!bd->module) return;
  if(!(bd->module->blend_params->mask_mode & DEVELOP_MASK_FLEXI)) return;
  _recompute_insert_hint(bd->module);
}

// the final click on an operator item: plain click stages the new group above
// the current target, ctrl+click stages it below (or, with nothing selected,
// above everything vs. at the very bottom -- see _stage_new_group). "activate"
// carries no event of its own, so the modifier state is read off whichever
// event is currently being processed (the button-release on this very item).
static void _new_shape_op_activate(GtkMenuItem *item, gpointer user_data)
{
  GtkWidget *btn = g_object_get_data(G_OBJECT(item), "opbtn");
  dt_iop_module_t *module = g_object_get_data(G_OBJECT(btn), "module");
  const int idx = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(item), "opidx"));
  GdkModifierType state = 0;
  gtk_get_current_event_state(&state);
  const gboolean below = dt_modifier_is(state, GDK_CONTROL_MASK);
  if(module) _stage_new_group(module, _masks_ops[idx].state, below);
}

// the add-group operator chooser. With first-class groups two same-operator groups
// may sit adjacent (kept apart by GROUP_BREAK), so every operator is offered
// unconditionally -- including at the base (bottom): a base group's own
// operator is never evaluated at all (see _group_get_mask_roi_flexi), so it
// always contributes exactly its own mask regardless of which one is picked.
static gboolean _new_shape_op_press(GtkWidget *w, GdkEventButton *ev, gpointer u)
{
  if(ev->button != GDK_BUTTON_PRIMARY) return FALSE;
  GtkWidget *btn = u ? GTK_WIDGET(u) : w;

  GtkWidget *menu = gtk_menu_new();
  for(int i = 0; i < (int)(sizeof(_masks_ops) / sizeof(_masks_ops[0])); i++)
  {
    // bypass is not an operator a group can be created with -- it only
    // disables an existing one (see _build_group_op_menu)
    if(_masks_ops[i].state == DT_MASKS_STATE_OP_BYPASS) continue;
    GtkWidget *it = _op_menu_item(_masks_ops[i].paint, _masks_ops[i].name);
    gtk_widget_set_tooltip_text(it, _("click to add above the current target\n"
                                      "(or above everything, if none is selected)\n"
                                      "ctrl+click to add below it instead\n"
                                      "(or at the very bottom, if none is selected)"));
    g_object_set_data(G_OBJECT(it), "opbtn", btn);
    g_object_set_data(G_OBJECT(it), "opidx", GINT_TO_POINTER(i));
    g_signal_connect(G_OBJECT(it), "activate", G_CALLBACK(_new_shape_op_activate), NULL);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), it);
  }
  gtk_widget_show_all(menu);
  gtk_menu_popup_at_pointer(GTK_MENU(menu), (GdkEvent *)ev);
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
  // GROUP_BREAK lives on a run's head (its bottom-most point), so removing a run
  // can leave the runs that surrounded it touching with no marker between them --
  // if they share an operator they silently merge into one group. Snapshot the
  // partition first and re-stamp it after, exactly as every reorder/move path
  // does (_group_keys_apply is explicitly robust to a head having gone away).
  GHashTable *keys = _group_keys_snapshot(grp);

  for(GList *l = fids; l; l = g_list_next(l))
  {
    const dt_mask_id_t fid = GPOINTER_TO_INT(l->data);
    for(GList *p = grp->points; p; p = g_list_next(p))
    {
      dt_masks_point_group_t *pt = p->data;
      if(pt->formid == fid)
      {
        grp->points = g_list_remove(grp->points, pt);
        free(pt);
        break;
      }
    }
  }

  _group_keys_apply(grp, keys);
  g_hash_table_destroy(keys);
}

// remove every member shape of a group from the module's mask group. The group
// itself disappears (no empty group left behind) -- this is "delete group".
static void _group_delete_shapes(dt_iop_module_t *module, GList *fids)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp || !fids) return;
  dt_masks_clear_form_gui(darktable.develop);
  for(GList *l = fids; l; l = g_list_next(l))
    _clear_stale_formid_refs(bd, GPOINTER_TO_INT(l->data));
  _detach_group_members(grp, fids);
  dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  // deferred: this is called from the header's own press handler, which is
  // still mid-dispatch on `module`'s header widget -- rebuilding synchronously
  // here would destroy that widget out from under GTK's event propagation and
  // crash (same class of bug as the DnD teardown race, see _rebuild_masks_list_idle)
  _queue_masks_list_rebuild(module);
  _refresh_canvas_edit(module);
}

// "reset group": remove every member shape but keep the group as an empty group
// (a persistent drop target) in the same place and with the same operator.
static void _group_reset_members(dt_iop_module_t *module, GList *fids, const int opstate)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp || !fids) return;
  const dt_masks_state_t op =
    (opstate & DT_MASKS_STATE_OP) ? (opstate & DT_MASKS_STATE_OP) : DT_MASKS_STATE_UNION;
  // the anchor for the empty group is the point directly below the run's bottom
  // member (INVALID if the run sits at the very bottom of the list)
  dt_mask_id_t below = INVALID_MASKID;
  // the run's bottom-most member is also its cid, which is how the group's
  // remembered number is keyed -- capture it before the members are detached
  dt_mask_id_t cid = INVALID_MASKID;
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    const dt_mask_id_t fid = ((dt_masks_point_group_t *)l->data)->formid;
    gboolean member = FALSE;
    for(GList *m = fids; m; m = g_list_next(m))
      if(GPOINTER_TO_INT(m->data) == fid)
      {
        member = TRUE;
        break;
      }
    if(member)
    {
      cid = fid;
      if(l->prev) below = ((dt_masks_point_group_t *)l->prev->data)->formid;
      break; // first (bottom-most) member found
    }
  }
  // the group stays, so it keeps its number: emptying a group must not renumber
  // it (nor free its number for a peer, until it is really gone)
  const int keep_ord = dt_is_valid_maskid(cid) ? _group_ordinal_of_cid(module, cid) : 0;
  // the group survives this as an empty placeholder, so its refinement should
  // too: stash it before the members (which hold it) are removed. Members are
  // broadcast-synced, so any one of them reflects the whole run.
  dt_masks_refinement_t keep = { 0 };
  gchar *keep_name = NULL;
  {
    const dt_masks_point_group_t *any =
      fids ? _group_point(grp, GPOINTER_TO_INT(fids->data)) : NULL;
    if(any)
    {
      keep = any->refinement;
      if(any->name[0]) keep_name = g_strdup(any->name);
    }
  }
  dt_masks_clear_form_gui(darktable.develop);
  for(GList *l = fids; l; l = g_list_next(l))
    _clear_stale_formid_refs(bd, GPOINTER_TO_INT(l->data));
  // detach only -- see _detach_group_members: routing this through
  // dt_masks_form_remove() would delete the module's whole mask group as soon as
  // this run held its last shapes, which is the exact opposite of "keep the
  // group as an empty placeholder"
  _detach_group_members(grp, fids);
  dt_masks_empty_group_t *eg = _empty_group_new(op, 0, below);
  eg->refinement = keep;
  eg->ordinal = keep_ord;
  eg->name = keep_name;
  bd->empty_groups = g_list_append(bd->empty_groups, eg);
  bd->panel_selected_group_cid = INVALID_MASKID;
  bd->selected_empty = eg;
  dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  // deferred, same reasoning as _group_delete_shapes above
  _queue_masks_list_rebuild(module);
  _refresh_canvas_edit(module);
}

static void _close_shape_actions_menu(GtkWidget *item);
static void
_group_op_apply(dt_iop_module_t *module, GList *formids, const dt_masks_state_t op);
static GtkWidget *_build_group_between_op_menu(dt_iop_module_t *module,
                                               GList *formids,
                                               const gboolean is_base);
static GtkWidget *_build_group_actions_menu(dt_iop_module_t *module,
                                            GList *formids,
                                            const gboolean is_base,
                                            GtkWidget *lbl_box);

// start inline rename on a group's title: swap `lbl_box`'s label child for an
// entry, same gesture as renaming an element (ctrl+click, see
// _start_rename_element) -- but the typed text broadcasts onto every member
// of the run instead of a single form's name (see _group_rename_commit).
// Shared by ctrl+click on the header background/title (_group_header_press)
// and ctrl+click on the operator handle (_group_op_press): ctrl+click is now
// the one shared "rename" gesture regardless of which of a header's
// non-specific-widget areas it lands on (icon, title, or the empty gaps
// between them), matching the same rule shift+click (toggle properties) and
// right-click (open the actions menu) already follow.
// exactly one of cid/eg is the real target: cid valid + eg NULL for a
// populated group, cid invalid + eg non-NULL for a staged empty one (see
// _group_rename_commit, which reads the same pair back off the entry).
static void _start_group_rename(GtkWidget *lbl_box,
                                dt_iop_module_t *module,
                                const dt_mask_id_t cid,
                                dt_masks_empty_group_t *eg)
{
  if(!lbl_box) return;
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
  if(eg)
  {
    if(bd->selected_empty != eg) _select_empty_group(module, eg);
  }
  else if(bd->panel_selected_group_cid != cid)
    _set_group_target(module, cid);
  const char *custom =
    eg ? eg->name
       : (dt_is_valid_maskid(cid) ? _group_custom_name(_module_mask_group(module), cid)
                                  : NULL);
  if(current) gtk_widget_destroy(current);
  GtkWidget *entry = gtk_entry_new();
  gtk_entry_set_has_frame(GTK_ENTRY(entry), FALSE);
  dt_gui_add_class(entry, "mask-rename-entry");
  if(custom) gtk_entry_set_text(GTK_ENTRY(entry), custom);
  g_object_set_data(G_OBJECT(entry), "group-cid", GINT_TO_POINTER(cid));
  g_object_set_data(G_OBJECT(entry), "eg", eg);
  g_object_set_data(G_OBJECT(lbl_box), "title-child", entry);
  gtk_box_pack_start(GTK_BOX(lbl_box), entry, TRUE, TRUE, 0);
  gtk_box_reorder_child(GTK_BOX(lbl_box), entry, 0);
  g_signal_connect(G_OBJECT(entry), "activate", G_CALLBACK(_group_rename_commit), module);
  g_signal_connect(G_OBJECT(entry), "focus-out-event",
                   G_CALLBACK(_group_rename_focus_out), module);
  g_signal_connect(G_OBJECT(entry), "key-press-event",
                   G_CALLBACK(_group_rename_key_press), module);
  // the header (hdr_evbox, found by walking up to the ancestor tagged
  // "group-key" or "eg-header" -- see their construction) is armed as a
  // reorder drag source whenever there are 2+ groups. GTK's own drag
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
    if(g_object_get_data(G_OBJECT(w), "group-key")
       || g_object_get_data(G_OBJECT(w), "eg-header"))
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
  if(e->type == GDK_BUTTON_PRESS && e->button == GDK_BUTTON_PRIMARY
     && dt_modifier_is(e->state, GDK_CONTROL_MASK))
  {
    _start_group_rename(g_object_get_data(G_OBJECT(w), "title-label-box"), module,
                        GPOINTER_TO_INT(g_object_get_data(G_OBJECT(w), "group-key")),
                        NULL);
    return TRUE;
  }
  if(e->button == GDK_BUTTON_SECONDARY)
  {
    const gboolean is_base = g_object_get_data(G_OBJECT(w), "is-base-group") != NULL;
    GList *formids = g_object_get_data(G_OBJECT(w), "group-formids");
    GtkWidget *lbl_box = g_object_get_data(G_OBJECT(w), "title-label-box");
    GtkWidget *menu = _build_group_actions_menu(module, formids, is_base, lbl_box);
    gtk_menu_popup_at_pointer(GTK_MENU(menu), (GdkEvent *)e);
    return TRUE;
  }
  dt_iop_gui_blend_data_t *bd = module->blend_data;
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

static void _toggle_solo_group(dt_iop_module_t *module, const guint key, GList *members)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp) return;

  if(_model_toggle_solo_group(module, grp, key, members) == DT_MASKS_SOLO_CANVAS_FULL)
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
}

// same idea as _solo_badge_form_press: the group badge only shows solo (never
// solo-edit, groups have no such concept) while its group is the soloed one,
// so a click while it shows solo always means "un-solo" -- the off-branch of
// _toggle_solo_group never touches `members`, so NULL is safe. Must check the
// status first (see _solo_badge_form_press).
static gboolean
_solo_badge_group_press(GtkWidget *w, GdkEventButton *e, dt_iop_module_t *module)
{
  if(e->button != GDK_BUTTON_PRIMARY) return FALSE;
  const int status = _solo_status_badge_get(w);
  if(status == MASK_SOLO_BADGE_SOLO)
  {
    const guint key = GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(w), "group-key"));
    _toggle_solo_group(module, key, NULL);
    return TRUE;
  }
  else if(status == MASK_SOLO_BADGE_DISABLE)
  {
    GList *formids = g_object_get_data(G_OBJECT(w), "formids");
    _group_op_apply(module, formids, DT_MASKS_STATE_OP_BYPASS);
    return TRUE;
  }
  return FALSE;
}

// within-group combine modes: how a group folds its own members together, before
// the finished sub-mask is composited onto the stack by the group's operator.
// union (max) is the neutral default; screen (a+b-ab) smooths feathered overlaps;
// intersect (min) is the AND, e.g. to rebuild a legacy multi-channel parametric
// mask from single-channel parametric elements. Order matches the menu.
static const struct
{
  dt_masks_state_t bit; // 0 = union (no within bit)
  DTGTKCairoPaintIconFunc paint;
  const char *name;
} _within_modes[] = {
  { 0, dtgtk_cairo_paint_masks_union, N_("union") },
  { DT_MASKS_STATE_SCREEN, dtgtk_cairo_paint_tool_blur, N_("screen") },
  { DT_MASKS_STATE_ISECT, dtgtk_cairo_paint_masks_intersection, N_("intersect") },
  { DT_MASKS_STATE_WITHIN_MULTIPLY, dtgtk_cairo_paint_masks_multiply, N_("multiply") },
};

static DTGTKCairoPaintIconFunc _within_paint(const dt_masks_state_t within)
{
  if(within & DT_MASKS_STATE_ISECT) return dtgtk_cairo_paint_masks_intersection;
  if(within & DT_MASKS_STATE_SCREEN) return dtgtk_cairo_paint_tool_blur;
  if(within & DT_MASKS_STATE_WITHIN_MULTIPLY) return dtgtk_cairo_paint_masks_multiply;
  return dtgtk_cairo_paint_masks_union;
}

static const char *_within_name(const dt_masks_state_t within)
{
  if(within & DT_MASKS_STATE_ISECT) return _("intersect");
  if(within & DT_MASKS_STATE_SCREEN) return _("screen");
  if(within & DT_MASKS_STATE_WITHIN_MULTIPLY) return _("multiply");
  return _("union");
}

// set a group's within-group combine mode: broadcast the (mutually exclusive)
// within bits onto every member, so the renderer reads it from the run head.
// Union (no within bit) ⇒ byte-identical for groups that never touch this.
static void
_within_mode_apply(dt_iop_module_t *module, GList *formids, const dt_masks_state_t within)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp) return;
  for(GList *l = formids; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = _group_point(grp, GPOINTER_TO_INT(l->data));
    if(!pt) continue;
    pt->state = (pt->state & ~DT_MASKS_STATE_WITHIN) | (within & DT_MASKS_STATE_WITHIN);
  }
  dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  _build_masks_list(module);
}

static void _within_menu_activate(GtkMenuItem *item, dt_iop_module_t *module)
{
  GList *formids = g_object_get_data(G_OBJECT(item), "formids");
  const dt_masks_state_t within =
    GPOINTER_TO_INT(g_object_get_data(G_OBJECT(item), "within"));
  _within_mode_apply(module, formids, within);
}

// build (but do not show) the within-group combine chooser (union / screen /
// intersect) for a group. Shared by a direct click on the chooser button and
// the "change within-group mode" shortcut.
static GtkWidget *_build_within_menu(dt_iop_module_t *module, GList *formids)
{
  GtkWidget *menu = gtk_menu_new();
  for(int i = 0; i < (int)(sizeof(_within_modes) / sizeof(_within_modes[0])); i++)
  {
    GtkWidget *it = _op_menu_item(_within_modes[i].paint, _within_modes[i].name);
    g_object_set_data(G_OBJECT(it), "within", GINT_TO_POINTER(_within_modes[i].bit));
    g_object_set_data_full(G_OBJECT(it), "formids", g_list_copy(formids),
                           (GDestroyNotify)g_list_free);
    g_signal_connect(G_OBJECT(it), "activate", G_CALLBACK(_within_menu_activate), module);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), it);
  }
  gtk_widget_show_all(menu);
  return menu;
}

static gboolean
_group_within_press(GtkWidget *widget, GdkEventButton *ev, gpointer user_data)
{
  if(ev->button != GDK_BUTTON_PRIMARY) return FALSE;
  GtkWidget *btn = user_data;
  dt_iop_module_t *module = g_object_get_data(G_OBJECT(btn), "module");
  GList *formids = g_object_get_data(G_OBJECT(btn), "formids");
  if(!module) return TRUE;

  GtkWidget *menu = _build_within_menu(module, formids);
  dt_gui_menu_popup(GTK_MENU(menu), btn, GDK_GRAVITY_SOUTH_WEST, GDK_GRAVITY_NORTH_WEST);
  return TRUE;
}

// build the within-group combine chooser: a bordered chooser box (like the
// add-group operator combo) showing the current mode's icon. `within` is the
// group's aggregate mode (all members agree, else union). Packed on the RIGHT of
// the header so it never reads as the group's own (between-group) operator chip,
// which lives in the left-hand handle. Pass sensitive=FALSE for empty groups.
static GtkWidget *_make_within_selector(dt_iop_module_t *module,
                                        GList *formids,
                                        const dt_masks_state_t within,
                                        const gboolean sensitive)
{
  GtkWidget *inner = NULL;
  GtkWidget *box =
    _make_op_combo(&inner, _within_paint(within), G_CALLBACK(_group_within_press));
  // unlike the between-group operator chip, this selector should not read as
  // a bordered "combo" -- drop the border so it sits flush in the header.
  dt_gui_remove_class(box, "mask-op-combo");
  // the label next to it has hexpand and no margin of its own, so without this
  // the icon sits flush against the label text -- add breathing room (see
  // darktable.css)
  dt_gui_add_class(box, "mask-within-combo");
  // same footprint as the between-group operator chip (_make_drag_handle's
  // own 18dpi plate) -- the two read as a matched pair of operator icons
  // either side of the group's title/opacity, not a big chip and a small one.
  gtk_widget_set_size_request(box, DT_PIXEL_APPLY_DPI(18), DT_PIXEL_APPLY_DPI(18));
  gtk_widget_set_valign(box, GTK_ALIGN_CENTER);
  g_object_set_data(G_OBJECT(inner), "module", module);
  if(formids)
    g_object_set_data_full(G_OBJECT(inner), "formids", g_list_copy(formids),
                           (GDestroyNotify)g_list_free);
  gchar *tip =
    sensitive
      ? g_strdup_printf(_("within-group combine: %s\n"
                          "click to change how this group's shapes fold together:\n"
                          "union (max), screen (soft overlaps) or intersect (min/AND)"),
                        _within_name(within))
      : g_strdup(_("within-group combine (available once the group has shapes)"));
  gtk_widget_set_tooltip_text(inner, tip);
  g_free(tip);
  gtk_widget_set_sensitive(box, sensitive);
  return box;
}

// change the operator of a whole group: broadcast `op` onto every member, then
// normalize. First-class groups may share an operator with a neighbour, so we
// snapshot the partition and re-stamp the break markers afterwards -- the group
// keeps its identity even if it now matches the group above/below.
//
// `op` == DT_MASKS_STATE_OP_BYPASS is the one special case: bypass is a modifier
// on top of the group's operator, not an operator of its own, so it toggles the
// bypass bit and leaves the rest of the operator alone. Picking any real
// operator clears the bypass bit with the rest of the old one, which is what
// makes choosing an operator on a disabled group re-enable it.
static void
_group_op_apply(dt_iop_module_t *module, GList *formids, const dt_masks_state_t op)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp) return;
  const gboolean toggle_bypass = op == DT_MASKS_STATE_OP_BYPASS;
  // decide the wanted bypass state once, from one member (the bit is broadcast
  // across the whole group, so any member answers), so a group whose members
  // somehow disagree ends up consistent rather than half-toggled
  gboolean set_bypass = FALSE;
  if(toggle_bypass && formids)
  {
    for(GList *l = formids; l; l = g_list_next(l))
    {
      const dt_masks_point_group_t *pt = _group_point(grp, GPOINTER_TO_INT(l->data));
      if(pt)
      {
        set_bypass = !_op_is_bypassed(pt->state);
        break;
      }
    }
  }
  GList *heads = _group_partition_heads(grp);
  if(toggle_bypass)
    dt_masks_group_set_state(grp, formids, DT_MASKS_STATE_OP_BYPASS, set_bypass);
  else
    // not a bit broadcast but a field replacement (the operator is a value in
    // the DT_MASKS_STATE_OP bits, not a flag), so it stays hand-rolled here
    for(GList *l = formids; l; l = g_list_next(l))
    {
      dt_masks_point_group_t *pt = _group_point(grp, GPOINTER_TO_INT(l->data));
      if(pt) pt->state = (pt->state & ~DT_MASKS_STATE_OP) | op;
    }
  _apply_partition_breaks(grp, heads);
  g_list_free(heads);
  _normalize_group_operators(grp);
  dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
  // deferred: also reachable directly from the group's own operator-handle
  // press handler (_group_op_press's ctrl/shift-click), still mid-dispatch on
  // that widget -- see _group_delete_shapes above for why this must not be
  // synchronous
  _queue_masks_list_rebuild(module);
  _refresh_canvas_edit(module);
}

// merge this group down into the group directly below it: the members adopt the
// lower group's operator and this group's break marker is dropped, so the two runs
// fuse into one. No-op for the bottom group (nothing below it). Other groups keep
// their identity (their heads are re-stamped from the pre-merge partition).
static void _merge_group_down(dt_iop_module_t *module, GList *formids)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp || !formids) return;

  // locate this group's head node (the bottom-most member that starts the group)
  GList *head_node = NULL;
  for(GList *l = grp->points; l && !head_node; l = g_list_next(l))
  {
    if(!_starts_group(l)) continue;
    const dt_mask_id_t fid = ((dt_masks_point_group_t *)l->data)->formid;
    for(GList *f = formids; f; f = g_list_next(f))
      if(GPOINTER_TO_INT(f->data) == fid)
      {
        head_node = l;
        break;
      }
  }
  if(!head_node || !head_node->prev) return; // bottom group: nothing to merge into

  const dt_mask_id_t head_fid = ((dt_masks_point_group_t *)head_node->data)->formid;
  const dt_masks_state_t below_op =
    _eff_group_op(((dt_masks_point_group_t *)head_node->prev->data)->state);

  // the post-merge partition is the current one minus this group's head
  GList *heads = _group_partition_heads(grp);
  heads = g_list_remove(heads, GINT_TO_POINTER(head_fid));

  for(GList *f = formids; f; f = g_list_next(f))
  {
    dt_masks_point_group_t *pt = _group_point(grp, GPOINTER_TO_INT(f->data));
    if(pt) pt->state = (pt->state & ~DT_MASKS_STATE_OP) | below_op;
  }
  _apply_partition_breaks(grp, heads);
  g_list_free(heads);
  _normalize_group_operators(grp);
  dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  // deferred: also reachable directly from the group's own operator-handle
  // press handler (_group_op_press's shift-click), same reasoning as
  // _group_op_apply above
  _queue_masks_list_rebuild(module);
}

static void _group_op_menu_activate(GtkMenuItem *item, dt_iop_module_t *module)
{
  GList *formids = g_object_get_data(G_OBJECT(item), "formids");
  const dt_masks_state_t op = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(item), "op"));
  _group_op_apply(module, formids, op);
}

// "merge into group below" menu item activate: shares the same action that
// used to live behind the handle's own shift+click, before that gesture was
// dropped entirely (see _group_op_press)
static void _group_merge_down_menu_activate(GtkMenuItem *item, dt_iop_module_t *module)
{
  GList *formids = g_object_get_data(G_OBJECT(item), "formids");
  _merge_group_down(module, formids);
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
static void _invert_group_members(dt_iop_module_t *module, GList *members)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp || !members) return;
  for(GList *l = members; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = _group_point(grp, GPOINTER_TO_INT(l->data));
    if(!pt) continue;
    pt->state ^= DT_MASKS_STATE_INVERSE;
  }
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
// per-run flag, unlike _invert_group_members' one-shot member flip. Broadcast
// across every member of the run (same reason DT_MASKS_STATE_OP_BYPASS is in
// _group_op_apply) so the bit stays part of DT_MASKS_STATE_OP and travels
// correctly with the run through reorder/merge/split. Individual members'
// own DT_MASKS_STATE_INVERSE bits are untouched -- the two are independent.
static void _group_toggle_output_invert(dt_iop_module_t *module, GList *members)
{
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp || !members) return;
  const dt_masks_point_group_t *any = _group_point(grp, GPOINTER_TO_INT(members->data));
  if(!any) return;
  const gboolean set_invert = !(any->state & DT_MASKS_STATE_OP_INVERT);
  dt_masks_group_set_state(grp, members, DT_MASKS_STATE_OP_INVERT, set_invert);
  dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
  // like _invert_group_members's own switch away from a full rebuild: this
  // touches no row's structure, just this one run's own handle look -- update
  // it in place (members' tail is this run's head/cid, see the header build's
  // own formids ordering) instead of tearing down the whole panel.
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(bd && bd->masks_list_box)
  {
    const guint cid = GPOINTER_TO_UINT(g_list_last(members)->data);
    _apply_group_output_invert_icon(GTK_WIDGET(bd->masks_list_box), cid, set_invert);
  }
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

// set this run's own persistent, multiplicative opacity (see
// dt_masks_point_group_t.group_opacity) -- an absolute value, unlike every
// other multi-target properties row (_props_row_apply's delta convention): a
// group header always represents exactly one run, so there is no multi-select
// ambiguity a delta needs to resolve. Broadcast onto every member of the run,
// same convention as invert-output/bypass above (see _group_toggle_output_invert).
static void _group_opacity_changed(GtkWidget *w, dt_iop_module_t *module)
{
  if(DT_IN_GUI_UPDATE()) return;
  dt_masks_form_t *grp = _module_mask_group(module);
  GList *members = g_object_get_data(G_OBJECT(w), "formids");
  if(!grp || !members) return;
  const float value = dt_bauhaus_slider_get(w);
  for(GList *l = members; l; l = g_list_next(l))
  {
    dt_masks_point_group_t *pt = _group_point(grp, GPOINTER_TO_INT(l->data));
    if(pt) pt->group_opacity = value;
  }
  _group_opacity_update_tooltip(w, value);
  dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  dt_control_queue_redraw_center();
  // an opacity change can push this group -- or, since it scales every
  // member's own effective opacity too, any of its elements -- across the
  // low-opacity threshold; refresh every badge in the panel, not just this
  // group's own (mirrors _props_row_apply's own call for the same reason).
  _refresh_lowop_badges(module);
}

static void _group_invert_elements_menu_activate(GtkMenuItem *item,
                                                 dt_iop_module_t *module)
{
  GList *formids = g_object_get_data(G_OBJECT(item), "formids");
  _invert_group_members(module, formids);
}

static void _group_invert_output_menu_toggled(GtkCheckMenuItem *item,
                                              dt_iop_module_t *module)
{
  GList *formids = g_object_get_data(G_OBJECT(item), "formids");
  _close_shape_actions_menu(GTK_WIDGET(item));
  _group_toggle_output_invert(module, formids);
}

static void _group_menu_delete(GtkMenuItem *item, dt_iop_module_t *module)
{
  GList *formids = g_object_get_data(G_OBJECT(item), "formids");
  _group_delete_shapes(module, formids);
}

static void _group_menu_empty(GtkMenuItem *item, dt_iop_module_t *module)
{
  GList *formids = g_object_get_data(G_OBJECT(item), "formids");
  const int opstate = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(item), "op"));
  _group_reset_members(module, formids, opstate);
}

// mirrors _shape_menu_toggle_solo, but for a whole group.
static void _group_menu_toggle_solo(GtkCheckMenuItem *item, dt_iop_module_t *module)
{
  GList *formids = g_object_get_data(G_OBJECT(item), "formids");
  const guint key = GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(item), "group-key"));
  _close_shape_actions_menu(GTK_WIDGET(item));
  _toggle_solo_group(module, key, formids);
}

static gboolean
_group_between_op_press(GtkWidget *widget, GdkEventButton *ev, gpointer user_data)
{
  if(ev->button != GDK_BUTTON_PRIMARY) return FALSE;
  GtkWidget *btn = user_data;
  dt_iop_module_t *module = g_object_get_data(G_OBJECT(btn), "module");
  GList *formids = g_object_get_data(G_OBJECT(btn), "formids");
  const gboolean is_base = g_object_get_data(G_OBJECT(btn), "is-base-group") != NULL;
  if(!module) return TRUE;

  GtkWidget *menu = _build_group_between_op_menu(module, formids, is_base);
  dt_gui_menu_popup(GTK_MENU(menu), btn, GDK_GRAVITY_SOUTH_WEST, GDK_GRAVITY_NORTH_WEST);
  return TRUE;
}

static GtkWidget *_build_group_between_op_menu(dt_iop_module_t *module,
                                               GList *formids,
                                               const gboolean is_base)
{
  GtkWidget *menu = gtk_menu_new();
  for(int i = 0; i < (int)(sizeof(_masks_ops) / sizeof(_masks_ops[0])); i++)
  {
    if(_masks_ops[i].state == DT_MASKS_STATE_OP_BYPASS) continue;
    if(is_base) continue; // the base group's operator is a no-op
    GtkWidget *it = _op_menu_item(_masks_ops[i].paint, _masks_ops[i].name);
    g_object_set_data_full(G_OBJECT(it), "formids", g_list_copy(formids),
                           (GDestroyNotify)g_list_free);
    g_object_set_data(G_OBJECT(it), "op", GINT_TO_POINTER(_masks_ops[i].state));
    g_signal_connect(G_OBJECT(it), "activate", G_CALLBACK(_group_op_menu_activate),
                     module);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), it);
  }
  gtk_widget_show_all(menu);
  return menu;
}

static void _group_menu_toggle_bypass(GtkCheckMenuItem *item, dt_iop_module_t *module)
{
  GList *formids = g_object_get_data(G_OBJECT(item), "formids");
  _close_shape_actions_menu(GTK_WIDGET(item));
  _group_op_apply(module, formids, DT_MASKS_STATE_OP_BYPASS);
}

static void _group_menu_rename(GtkMenuItem *item, dt_iop_module_t *module)
{
  GtkWidget *lbl_box = g_object_get_data(G_OBJECT(item), "title-label-box");
  const dt_mask_id_t cid =
    (dt_mask_id_t)GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(item), "group-key"));
  dt_masks_empty_group_t *eg = g_object_get_data(G_OBJECT(item), "eg");
  if(lbl_box) _start_group_rename(lbl_box, module, cid, eg);
}

static void
_add_menu_section_header(GtkWidget *menu, const char *title, const gboolean add_separator)
{
  if(add_separator)
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), gtk_separator_menu_item_new());
  GtkWidget *header = gtk_menu_item_new_with_label(title);
  gtk_widget_set_sensitive(header, FALSE);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), header);
}

static GtkWidget *_build_group_actions_menu(dt_iop_module_t *module,
                                            GList *formids,
                                            const gboolean is_base,
                                            GtkWidget *lbl_box)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  const guint cid = formids ? GPOINTER_TO_UINT(g_list_last(formids)->data) : 0;
  const dt_masks_point_group_t *any =
    (grp && formids) ? _group_point(grp, (dt_mask_id_t)cid) : NULL;
  const gboolean bypassed = any && _op_is_bypassed(any->state);

  GtkWidget *menu = gtk_menu_new();

  // visibility
  _add_menu_section_header(menu, _("visibility"), FALSE);

  // disable toggle
  GtkWidget *bypass_it = gtk_check_menu_item_new_with_label(_("disable"));
  gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(bypass_it), bypassed);
  gtk_widget_set_tooltip_text(
    bypass_it, _("temporarily disable this group: it keeps its shapes, its operator "
                 "and its place in the stack, but contributes nothing to the mask"));
  g_object_set_data_full(G_OBJECT(bypass_it), "formids", g_list_copy(formids),
                         (GDestroyNotify)g_list_free);
  g_signal_connect(G_OBJECT(bypass_it), "toggled", G_CALLBACK(_group_menu_toggle_bypass),
                   module);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), bypass_it);

  if(!bypassed)
  {
    // solo the whole group (see _toggle_solo_group)
    GtkWidget *solo_it = gtk_check_menu_item_new_with_label(_("solo"));
    gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(solo_it),
                                   bd->solo_group_key == cid);
    g_object_set_data_full(G_OBJECT(solo_it), "formids", g_list_copy(formids),
                           (GDestroyNotify)g_list_free);
    g_object_set_data(G_OBJECT(solo_it), "group-key", GUINT_TO_POINTER(cid));
    g_signal_connect(G_OBJECT(solo_it), "toggled", G_CALLBACK(_group_menu_toggle_solo),
                     module);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), solo_it);

    // mask operations
    _add_menu_section_header(menu, _("mask operations"), TRUE);

    const gboolean output_inverted = any && (any->state & DT_MASKS_STATE_OP_INVERT);
    GtkWidget *invert_output_it = gtk_check_menu_item_new_with_label(_("invert output"));
    gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(invert_output_it),
                                   output_inverted);
    gtk_widget_set_tooltip_text(
      invert_output_it,
      _("invert this group's own finished mask before it combines with the "
        "stack below it -- a persistent state, shown on the group's own "
        "handle; the elements inside keep their own independent state"));
    g_object_set_data_full(G_OBJECT(invert_output_it), "formids", g_list_copy(formids),
                           (GDestroyNotify)g_list_free);
    g_signal_connect(G_OBJECT(invert_output_it), "toggled",
                     G_CALLBACK(_group_invert_output_menu_toggled), module);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), invert_output_it);

    GtkWidget *invert_elems_it = gtk_menu_item_new_with_label(_("invert all elements"));
    gtk_widget_set_tooltip_text(
      invert_elems_it, _("flip every element's own inversion bit independently\n"
                         "a one-shot action, not a persistent state -- not the same as "
                         "\"invert output\" except for a single-element group"));
    g_object_set_data_full(G_OBJECT(invert_elems_it), "formids", g_list_copy(formids),
                           (GDestroyNotify)g_list_free);
    g_signal_connect(G_OBJECT(invert_elems_it), "activate",
                     G_CALLBACK(_group_invert_elements_menu_activate), module);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), invert_elems_it);
  }

  // edit
  _add_menu_section_header(menu, _("edit"), TRUE);

  // rename
  GtkWidget *rename_it = gtk_menu_item_new_with_label(_("rename"));
  g_object_set_data(G_OBJECT(rename_it), "title-label-box", lbl_box);
  g_object_set_data(G_OBJECT(rename_it), "group-key", GUINT_TO_POINTER(cid));
  g_signal_connect(G_OBJECT(rename_it), "activate", G_CALLBACK(_group_menu_rename),
                   module);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), rename_it);

  // merge elements into group below
  if(!is_base && !bypassed)
  {
    GtkWidget *merge_it =
      gtk_menu_item_new_with_label(_("merge elements into group below"));
    g_object_set_data_full(G_OBJECT(merge_it), "formids", g_list_copy(formids),
                           (GDestroyNotify)g_list_free);
    g_signal_connect(G_OBJECT(merge_it), "activate",
                     G_CALLBACK(_group_merge_down_menu_activate), module);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), merge_it);
  }

  // empty group
  GtkWidget *empty_it = gtk_menu_item_new_with_label(_("empty group"));
  gtk_widget_set_tooltip_text(
    empty_it, _("remove every element from this group, keeping the (now empty) "
                "group itself in place as a drop target"));
  g_object_set_data_full(G_OBJECT(empty_it), "formids", g_list_copy(formids),
                         (GDestroyNotify)g_list_free);
  g_object_set_data(G_OBJECT(empty_it), "op", any ? GINT_TO_POINTER(any->state) : NULL);
  g_signal_connect(G_OBJECT(empty_it), "activate", G_CALLBACK(_group_menu_empty), module);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), empty_it);

  // delete group
  GtkWidget *delete_it = gtk_menu_item_new_with_label(_("delete group"));
  gtk_widget_set_tooltip_text(delete_it, _("delete this group and every element in it"));
  if(_group_count(module) <= 1) gtk_widget_set_sensitive(delete_it, FALSE);
  g_object_set_data_full(G_OBJECT(delete_it), "formids", g_list_copy(formids),
                         (GDestroyNotify)g_list_free);
  g_signal_connect(G_OBJECT(delete_it), "activate", G_CALLBACK(_group_menu_delete),
                   module);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), delete_it);

  gtk_widget_show_all(menu);
  return menu;
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
  // structure: the only thing it drives in the list is the solo badge and the
  // row's solo class on the old and new rows (see _update_shape_row_state),
  // both of which _refresh_all_shape_rows repaints from bd->soloedit_formid.
  // This is also why it no longer has to be deferred -- this is reachable
  // mid-dispatch from the row's own actions menu ("solo edit", see
  // _shape_menu_toggle_soloedit), and a full rebuild would have destroyed that
  // menu's own row underneath it; _refresh_all_shape_rows only mutates
  // existing widgets, so it is safe synchronously (same reasoning as
  // _toggle_solo_group).
  _refresh_all_shape_rows(module);
}

// change an empty group's operator from its header op chip
static void _empty_op_activate(GtkMenuItem *item, gpointer user_data)
{
  dt_iop_module_t *module = g_object_get_data(G_OBJECT(item), "module");
  dt_masks_empty_group_t *eg = g_object_get_data(G_OBJECT(item), "eg");
  const int idx = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(item), "opidx"));
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!g_list_find(bd->empty_groups, eg)) return; // stale
  eg->op = _masks_ops[idx].state;
  bd->selected_empty = eg; // keep it selected
  // note: changing an empty group's operator does NOT repaint the add-group icon
  // (that only follows the add-group menu, see _new_shape_op_update)
  _build_masks_list(module);
}

// "delete group" menu item's own handler (see _build_empty_group_menu) --
// pulled out of what used to be _empty_header_press's own right-click branch
// directly, now that right-click opens a menu instead of acting immediately.
static void _empty_menu_delete(GtkMenuItem *item, gpointer user_data)
{
  dt_iop_module_t *module = g_object_get_data(G_OBJECT(item), "module");
  dt_masks_empty_group_t *eg = g_object_get_data(G_OBJECT(item), "eg");
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!eg || !g_list_find(bd->empty_groups, eg)) return; // stale
  if(bd->selected_empty == eg) bd->selected_empty = NULL;
  bd->empty_groups = g_list_remove(bd->empty_groups, eg);
  _empty_group_free(eg);
  // deferred: this menu item is still mid-dispatch -- see _group_delete_shapes
  // above for why this must not be synchronous
  _queue_masks_list_rebuild(module);
}

static GtkWidget *_build_empty_group_between_op_menu(dt_iop_module_t *module,
                                                     dt_masks_empty_group_t *eg,
                                                     const gboolean is_base)
{
  GtkWidget *menu = gtk_menu_new();
  if(!is_base)
  {
    for(int i = 0; i < (int)(sizeof(_masks_ops) / sizeof(_masks_ops[0])); i++)
    {
      if(_masks_ops[i].state == DT_MASKS_STATE_OP_BYPASS) continue;
      GtkWidget *it = _op_menu_item(_masks_ops[i].paint, _masks_ops[i].name);
      g_object_set_data(G_OBJECT(it), "module", module);
      g_object_set_data(G_OBJECT(it), "eg", eg);
      g_object_set_data(G_OBJECT(it), "opidx", GINT_TO_POINTER(i));
      g_signal_connect(G_OBJECT(it), "activate", G_CALLBACK(_empty_op_activate), NULL);
      gtk_menu_shell_append(GTK_MENU_SHELL(menu), it);
    }
  }
  gtk_widget_show_all(menu);
  return menu;
}

static gboolean
_empty_between_op_press(GtkWidget *widget, GdkEventButton *ev, gpointer user_data)
{
  if(ev->button != GDK_BUTTON_PRIMARY) return FALSE;
  GtkWidget *btn = user_data;
  dt_iop_module_t *module = g_object_get_data(G_OBJECT(btn), "module");
  dt_masks_empty_group_t *eg = g_object_get_data(G_OBJECT(btn), "eg");
  const gboolean is_base = g_object_get_data(G_OBJECT(btn), "is-base-group") != NULL;
  if(!module || !eg) return TRUE;

  GtkWidget *menu = _build_empty_group_between_op_menu(module, eg, is_base);
  dt_gui_menu_popup(GTK_MENU(menu), btn, GDK_GRAVITY_SOUTH_WEST, GDK_GRAVITY_NORTH_WEST);
  return TRUE;
}

static GtkWidget *_build_empty_group_actions_menu(dt_iop_module_t *module,
                                                  dt_masks_empty_group_t *eg,
                                                  GtkWidget *lbl_box)
{
  GtkWidget *menu = gtk_menu_new();
  _add_menu_section_header(menu, _("edit"), FALSE);

  GtkWidget *rename_it = gtk_menu_item_new_with_label(_("rename"));
  g_object_set_data(G_OBJECT(rename_it), "title-label-box", lbl_box);
  g_object_set_data(G_OBJECT(rename_it), "eg", eg);
  g_object_set_data(G_OBJECT(rename_it), "group-key", GUINT_TO_POINTER(INVALID_MASKID));
  g_signal_connect(G_OBJECT(rename_it), "activate", G_CALLBACK(_group_menu_rename),
                   module);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), rename_it);

  GtkWidget *it = gtk_menu_item_new_with_label(_("delete group"));
  if(_group_count(module) <= 1) gtk_widget_set_sensitive(it, FALSE);
  g_object_set_data(G_OBJECT(it), "module", module);
  g_object_set_data(G_OBJECT(it), "eg", eg);
  g_signal_connect(G_OBJECT(it), "activate", G_CALLBACK(_empty_menu_delete), NULL);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), it);
  gtk_widget_show_all(menu);
  return menu;
}

// a plain primary press just clears any stale skip-select flag and returns
// FALSE, letting the drag source arm -- selection happens on release (see
// _empty_header_release), exactly like a real group's header. Right-click
// opens the actions menu above instead of acting immediately. Ctrl+click
// renames, matching a populated group's own header (_group_header_press).
static gboolean
_empty_header_press(GtkWidget *w, GdkEventButton *e, dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_empty_group_t *eg = g_object_get_data(G_OBJECT(w), "eg");
  if(!eg || !g_list_find(bd->empty_groups, eg)) return FALSE;
  if(e->type == GDK_BUTTON_PRESS && e->button == GDK_BUTTON_PRIMARY
     && dt_modifier_is(e->state, GDK_CONTROL_MASK))
  {
    _start_group_rename(g_object_get_data(G_OBJECT(w), "title-label-box"), module,
                        INVALID_MASKID, eg);
    return TRUE;
  }
  if(e->button == GDK_BUTTON_SECONDARY)
  {
    GtkWidget *lbl_box = g_object_get_data(G_OBJECT(w), "title-label-box");
    GtkWidget *menu = _build_empty_group_actions_menu(module, eg, lbl_box);
    gtk_menu_popup_at_pointer(GTK_MENU(menu), (GdkEvent *)e);
    return TRUE;
  }
  if(e->button != GDK_BUTTON_PRIMARY) return FALSE;
  bd->masks_skip_group_select_release = FALSE;
  return FALSE;
}

// matching release for _empty_header_press: select the group, unless a drag
// began in between (masks_skip_group_select_release got set by
// _group_drag_begin, shared with real groups), in which case this release
// just ends the drag and nothing else happens.
static gboolean
_empty_header_release(GtkWidget *w, GdkEventButton *e, dt_iop_module_t *module)
{
  if(e->button != GDK_BUTTON_PRIMARY) return FALSE;
  // ctrl+click is handled entirely on press (_empty_header_press starts the
  // rename entry there) and must not also act here -- same guard
  // _group_header_release has for a populated group's identical gesture
  // (missing here was exactly why ctrl+click-to-rename an empty group
  // flashed: this release still ran _select_empty_group right after).
  if(dt_modifier_is(e->state, GDK_CONTROL_MASK)) return FALSE;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(bd->masks_skip_group_select_release)
  {
    bd->masks_skip_group_select_release = FALSE;
    return FALSE;
  }
  dt_masks_empty_group_t *eg = g_object_get_data(G_OBJECT(w), "eg");
  if(!eg || !g_list_find(bd->empty_groups, eg)) return FALSE;
  _select_empty_group(module, eg);
  return TRUE;
}

// a shape dropped onto an empty group fills (realizes) it: the shape moves into
// the group's slot, adopts its operator/screen flag, and the empty group is
// dropped (its later siblings re-anchor onto the new run, mirroring the realize
// path in _build_masks_list).
// Model half of the element-onto-empty-group drop: the element leaves its old
// group and REALIZES the staged empty group, adopting its operator, within-group
// flag, staged refinement, ordinal and name. Same split as
// _model_drop_element_onto_element (see its comment).
gboolean _model_drop_element_onto_empty(dt_iop_module_t *module,
                                        dt_masks_form_t *grp,
                                        const dt_mask_id_t src,
                                        dt_masks_empty_group_t *eg)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!grp || !eg || !g_list_find(bd->empty_groups, eg)) return FALSE;

  dt_masks_point_group_t *sp = NULL;
  for(GList *l = grp->points; l; l = g_list_next(l))
    if(((dt_masks_point_group_t *)l->data)->formid == src)
    {
      sp = l->data;
      break;
    }
  if(!sp) return FALSE;

  const dt_masks_state_t op = (eg->op & DT_MASKS_STATE_OP)
                                ? (eg->op & DT_MASKS_STATE_OP)
                                : DT_MASKS_STATE_UNION;
  // if moving src empties its old group, keep that group alive as a placeholder
  struct dt_masks_empty_group_t *emptied = _capture_emptied_group(grp, src);
  // first-class groups: snapshot the partition so the group the shape leaves
  // stays distinct from its neighbours even when operators coincide. The shape
  // realizes a BRAND-NEW group, so it is forced to its own head below.
  GHashTable *keys = _group_keys_snapshot(grp);
  g_hash_table_insert(keys, GINT_TO_POINTER(src), GINT_TO_POINTER(src));
  grp->points = g_list_remove(grp->points, sp);
  sp->state = (sp->state & ~DT_MASKS_STATE_OP) | op;
  sp->state =
    (sp->state & ~DT_MASKS_STATE_WITHIN) | (eg->within & DT_MASKS_STATE_WITHIN);
  // position: just above the run anchored below this empty group; a
  // bottom-anchored empty (below INVALID) puts the shape at the bottom (base)
  int at = 0;
  if(dt_is_valid_maskid(eg->below_fid))
  {
    GList *run = _selected_group_formids(grp, eg->below_fid);
    int last = -1;
    const int firstidx = _run_extent(grp, run, &last);
    at = (firstidx < 0) ? (int)g_list_length(grp->points) : last + 1;
    g_list_free(run);
  }
  grp->points = g_list_insert(grp->points, sp, at);
  // re-anchor later siblings sharing this anchor, then drop the empty group
  GList *node = g_list_find(bd->empty_groups, eg);
  for(GList *l = node ? node->next : NULL; l; l = g_list_next(l))
  {
    dt_masks_empty_group_t *s = l->data;
    if(s->below_fid == eg->below_fid) s->below_fid = src;
  }
  if(bd->selected_empty == eg) bd->selected_empty = NULL;
  // adopt any refinement staged while the group had no members (see
  // dt_masks_empty_group_t.refinement); sp is the realized run's sole member
  if(eg->refinement.enabled) sp->refinement = eg->refinement;
  // and its number, so filling a group by drop does not renumber it
  if(eg->ordinal > 0)
  {
    if(!bd->group_ordinals)
      bd->group_ordinals = g_hash_table_new(g_direct_hash, g_direct_equal);
    g_hash_table_insert(bd->group_ordinals, GINT_TO_POINTER(sp->formid),
                        GINT_TO_POINTER(eg->ordinal));
  }
  // and its custom name, if it was named while still empty
  if(eg->name) g_strlcpy(sp->name, eg->name, sizeof(sp->name));
  bd->empty_groups = g_list_remove(bd->empty_groups, eg);
  _empty_group_free(eg);
  _group_keys_apply(grp, keys);
  g_hash_table_destroy(keys);
  if(emptied) bd->empty_groups = g_list_append(bd->empty_groups, emptied);
  // guarantee the realized shape is its own group head (cleared by normalize if
  // it lands at the very bottom, where a break is meaningless)
  sp->group_start = 1;
  _normalize_group_operators(grp);
  // a moved element should stay selected at the end of the drag
  bd->panel_selected_group_cid = _group_cid_of_form(grp, src); // select realized run
  bd->panel_selected_formid = src;
  return TRUE;
}

static void _masks_shape_to_empty_drop(GtkWidget *w,
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
    const dt_mask_id_t src = *(const dt_mask_id_t *)gtk_selection_data_get_data(sel);
    dt_masks_empty_group_t *eg = g_object_get_data(G_OBJECT(w), "eg");
    ok = _model_drop_element_onto_empty(module, _module_mask_group(module), src, eg);
    if(ok)
    {
      dt_print(DT_DEBUG_MASKS, "[masks] shape %d filled an empty group", src);
      dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
    }
  }
  gtk_drag_finish(ctx, ok, FALSE, time);
  if(ok) _queue_masks_list_rebuild(module);
}

// a whole cluster dropped onto an empty group fills (realizes) it, same as a
// single shape does above (_masks_shape_to_empty_drop) -- every member moves
// into the group's slot together, adopting its operator/screen flag, landing
// as one contiguous block in their original relative order, and the empty
// group placeholder is dropped.
static void _masks_cluster_to_empty_drop(GtkWidget *w,
                                         GdkDragContext *ctx,
                                         GtkSelectionData *sel,
                                         guint time,
                                         dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_empty_group_t *eg = g_object_get_data(G_OBJECT(w), "eg");
  GList *ids = _cluster_ids_from_selection(sel);
  gboolean ok = FALSE;
  if(eg && g_list_find(bd->empty_groups, eg) && ids)
  {
    dt_masks_form_t *grp = _module_mask_group(module);
    // recover the cluster's own relative (bottom-up) order from grp->points
    GList *ordered = NULL;
    for(GList *l = grp ? grp->points : NULL; l; l = g_list_next(l))
    {
      const dt_mask_id_t fid = ((dt_masks_point_group_t *)l->data)->formid;
      for(GList *m = ids; m; m = g_list_next(m))
        if(GPOINTER_TO_INT(m->data) == fid)
        {
          ordered = g_list_append(ordered, l->data);
          break;
        }
    }
    if(grp && ordered)
    {
      const dt_masks_state_t op = (eg->op & DT_MASKS_STATE_OP)
                                    ? (eg->op & DT_MASKS_STATE_OP)
                                    : DT_MASKS_STATE_UNION;
      // if moving the cluster empties its old group, keep that group alive as a
      // placeholder
      struct dt_masks_empty_group_t *emptied = _capture_emptied_group_multi(grp, ids);
      // the cluster realizes a brand-new group: force every member's key to the
      // block's own (bottom-most) formid so the whole run reads as one distinct group
      const dt_mask_id_t new_head = ((dt_masks_point_group_t *)ordered->data)->formid;
      GHashTable *keys = _group_keys_snapshot(grp);
      for(GList *l = ordered; l; l = g_list_next(l))
        g_hash_table_insert(keys,
                            GINT_TO_POINTER(((dt_masks_point_group_t *)l->data)->formid),
                            GINT_TO_POINTER(new_head));

      for(GList *l = ordered; l; l = g_list_next(l))
        grp->points = g_list_remove(grp->points, l->data);

      // position: just above the run anchored below this empty group; a
      // bottom-anchored empty (below INVALID) puts the block at the bottom (base)
      int at = 0;
      if(dt_is_valid_maskid(eg->below_fid))
      {
        GList *run = _selected_group_formids(grp, eg->below_fid);
        int last = -1;
        const int firstidx = _run_extent(grp, run, &last);
        at = (firstidx < 0) ? (int)g_list_length(grp->points) : last + 1;
        g_list_free(run);
      }
      int pos = at;
      for(GList *l = ordered; l; l = g_list_next(l), pos++)
      {
        dt_masks_point_group_t *pt = l->data;
        pt->state = (pt->state & ~DT_MASKS_STATE_OP) | op;
        pt->state =
          (pt->state & ~DT_MASKS_STATE_WITHIN) | (eg->within & DT_MASKS_STATE_WITHIN);
        grp->points = g_list_insert(grp->points, pt, pos);
      }

      // re-anchor later siblings sharing this anchor, then drop the empty group
      GList *node = g_list_find(bd->empty_groups, eg);
      for(GList *l = node ? node->next : NULL; l; l = g_list_next(l))
      {
        dt_masks_empty_group_t *s = l->data;
        if(s->below_fid == eg->below_fid) s->below_fid = new_head;
      }
      if(bd->selected_empty == eg) bd->selected_empty = NULL;
      bd->empty_groups = g_list_remove(bd->empty_groups, eg);
      _empty_group_free(eg);

      _group_keys_apply(grp, keys);
      g_hash_table_destroy(keys);
      if(emptied) bd->empty_groups = g_list_append(bd->empty_groups, emptied);
      // guarantee the realized block's own head starts a new run (cleared by
      // normalize if it lands at the very bottom, where a break is meaningless)
      ((dt_masks_point_group_t *)ordered->data)->group_start = 1;
      _normalize_group_operators(grp);

      bd->panel_selected_group_cid = _group_cid_of_form(grp, new_head);
      bd->panel_selected_formid = INVALID_MASKID;
      dt_print(DT_DEBUG_MASKS, "[masks] cluster filled an empty group");
      dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
      ok = TRUE;
    }
    g_list_free(ordered);
  }
  g_list_free(ids);
  gtk_drag_finish(ctx, ok, FALSE, time);
  if(ok) _queue_masks_list_rebuild(module);
}

// a whole real group was dropped on an empty group's header: reorder them via
// _masks_reorder_groups, which resolves this the same way as any other
// group/empty combination -- including landing directly on one of the
// dragged run's own anchored empties, which the below_fid model alone cannot
// express (an empty's anchor tracks its run wherever the run ends up).
static void _masks_group_to_empty_drop(GtkWidget *w,
                                       GdkDragContext *ctx,
                                       gint x,
                                       gint y,
                                       GtkSelectionData *sel,
                                       guint time,
                                       dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_empty_group_t *eg = g_object_get_data(G_OBJECT(w), "eg");
  gboolean ok = FALSE;
  if(eg && g_list_find(bd->empty_groups, eg)
     && gtk_selection_data_get_length(sel) == (gint)sizeof(dt_mask_id_t))
  {
    const dt_mask_id_t src = *(const dt_mask_id_t *)gtk_selection_data_get_data(sel);
    dt_masks_form_t *grp = _module_mask_group(module);
    if(grp)
    {
      const gboolean above = _group_drop_above(w, y);
      ok = _masks_reorder_groups(module, FALSE, _group_cid_of_form(grp, src), NULL, TRUE,
                                 INVALID_MASKID, eg, above);
      // the moved group stays selected, re-derived after the reorder -- see
      // _select_moved_group
      if(ok) _select_moved_group(module, _group_cid_of_form(grp, src));
      // a real group lands at the staged group's position, which can carry it
      // past other real groups (an empty group can sit between two of them), so
      // this reorders the evaluated fold order too and must be committed -- see
      // _masks_group_drag_received. (The mirror case, dragging an *empty* group,
      // moves nothing the pipe can see and is deliberately not committed.)
      if(ok) dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
    }
  }
  gtk_drag_finish(ctx, ok, FALSE, time);
  if(ok) _queue_masks_list_rebuild(module);
}

// an empty group was dropped on this header (real or empty): reorder them via
// _masks_reorder_groups, the same unified logic every group/empty combination
// goes through.
static void _masks_empty_reorder_drop(GtkWidget *w,
                                      GdkDragContext *ctx,
                                      gint y,
                                      GtkSelectionData *sel,
                                      guint time,
                                      dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  gboolean ok = FALSE;
  dt_masks_empty_group_t *src = NULL;
  if(gtk_selection_data_get_length(sel) == (gint)sizeof(gpointer))
  {
    memcpy(&src, gtk_selection_data_get_data(sel), sizeof(gpointer));
    if(src && g_list_find(bd->empty_groups, src))
    {
      dt_masks_form_t *grp = _module_mask_group(module);
      const gboolean above = _group_drop_above(w, y);

      GList *dst_formids = g_object_get_data(G_OBJECT(w), "group-formids");
      dt_masks_empty_group_t *dst_eg = g_object_get_data(G_OBJECT(w), "eg");
      const dt_mask_id_t dst_cid =
        dst_formids && grp ? _group_cid_of_form(grp, GPOINTER_TO_INT(dst_formids->data))
                           : INVALID_MASKID;

      if(dst_eg || dt_is_valid_maskid(dst_cid))
        ok = _masks_reorder_groups(module, TRUE, INVALID_MASKID, src, dst_eg != NULL,
                                   dst_cid, dst_eg, above);
      // the moved (staged) group stays selected -- see _select_moved_group
      if(ok) _select_moved_empty_group(module, src);
    }
  }
  gtk_drag_finish(ctx, ok, FALSE, time);
  if(ok) _queue_masks_list_rebuild(module);
}

// an empty group's header is a drop target for a shape (realize), a whole real
// group (land adjacent to this empty group), a whole cluster (realize with
// every member), or another empty group (reorder). Route on the target
// entry info.
static void _masks_empty_header_drag_received(GtkWidget *w,
                                              GdkDragContext *ctx,
                                              gint x,
                                              gint y,
                                              GtkSelectionData *sel,
                                              guint info,
                                              guint time,
                                              dt_iop_module_t *module)
{
  if(info == DND_MASK_ROW)
    _masks_shape_to_empty_drop(w, ctx, x, y, sel, info, time, module);
  else if(info == DND_MASK_GROUP)
    _masks_group_to_empty_drop(w, ctx, x, y, sel, time, module);
  else if(info == DND_MASK_CLUSTER)
    _masks_cluster_to_empty_drop(w, ctx, sel, time, module);
  else
    _masks_empty_reorder_drop(w, ctx, y, sel, time, module);
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
  DND_HOVER_REORDER,   // DND_TARGET_GROUP / DND_TARGET_EMPTY
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
    if(!strcmp(name, DND_TARGET_GROUP) || !strcmp(name, DND_TARGET_EMPTY))
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

  if(kind == DND_HOVER_REORDER)
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
  if(kind == DND_HOVER_REORDER)
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
static GtkWidget *_make_pending_shape_row(dt_iop_module_t *module, dt_masks_form_t *form)
{
  const guint kind = _form_kind(form);

  GtkWidget *row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
  GtkWidget *handle = _make_drag_handle(
    _kind_icon_paint(kind), FALSE,
    _("this shape has not been added yet -- finish drawing it on canvas to add it"));

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

  GtkWidget *opacity_slot = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
  gtk_box_pack_start(GTK_BOX(opacity_slot), opacity, FALSE, FALSE, 0);
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
    gtk_box_pack_start(GTK_BOX(props_box), feather, FALSE, FALSE, 0);
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
    gtk_box_pack_start(GTK_BOX(props_box), size, FALSE, FALSE, 0);

    GtkWidget *feather = _pending_conf_slider_new(
      module, DT_MASKS_CONF(form->type, circle, border),
      _blend_masks_properties[DT_MASKS_PROPERTY_FEATHER].name,
      _blend_masks_properties[DT_MASKS_PROPERTY_FEATHER].min,
      _blend_masks_properties[DT_MASKS_PROPERTY_FEATHER].max, 2,
      _blend_masks_properties[DT_MASKS_PROPERTY_FEATHER].format,
      _("fade-out border of the next circle, before it is placed --\n"
        "same as shift+scrolling on canvas."));
    gtk_box_pack_start(GTK_BOX(props_box), feather, FALSE, FALSE, 0);
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
    gtk_box_pack_start(GTK_BOX(props_box), size, FALSE, FALSE, 0);

    GtkWidget *feather = _pending_conf_slider_new(
      module, DT_MASKS_CONF(form->type, ellipse, border),
      _blend_masks_properties[DT_MASKS_PROPERTY_FEATHER].name,
      _blend_masks_properties[DT_MASKS_PROPERTY_FEATHER].min,
      _blend_masks_properties[DT_MASKS_PROPERTY_FEATHER].max, 2,
      _blend_masks_properties[DT_MASKS_PROPERTY_FEATHER].format,
      _("fade-out border of the next ellipse, before it is placed --\n"
        "same as shift+scrolling on canvas."));
    gtk_box_pack_start(GTK_BOX(props_box), feather, FALSE, FALSE, 0);

    GtkWidget *rotation =
      _pending_conf_slider_new(module, DT_MASKS_CONF(form->type, ellipse, rotation),
                               _blend_masks_properties[DT_MASKS_PROPERTY_ROTATION].name,
                               _blend_masks_properties[DT_MASKS_PROPERTY_ROTATION].min,
                               _blend_masks_properties[DT_MASKS_PROPERTY_ROTATION].max, 1,
                               _blend_masks_properties[DT_MASKS_PROPERTY_ROTATION].format,
                               _("rotation of the next ellipse, before it is placed --\n"
                                 "same as ctrl+shift+scrolling on canvas."));
    gtk_box_pack_start(GTK_BOX(props_box), rotation, FALSE, FALSE, 0);
  }
  else if(kind == DT_MASKS_GRADIENT)
  {
    GtkWidget *compression = _pending_conf_slider_new(
      module, DT_MASKS_CONF(form->type, gradient, compression),
      _blend_masks_properties[DT_MASKS_PROPERTY_COMPRESSION].name, 0.001f, 1.0f, 2, "%",
      _("compression of the next gradient, before it is placed --\n"
        "same as shift+scrolling on canvas."));
    gtk_box_pack_start(GTK_BOX(props_box), compression, FALSE, FALSE, 0);

    GtkWidget *curvature = _pending_conf_slider_new(
      module, DT_MASKS_CONF(form->type, gradient, curvature),
      _blend_masks_properties[DT_MASKS_PROPERTY_CURVATURE].name, -2.0f, 2.0f, 2, NULL,
      _("curvature of the next gradient, before it is placed --\n"
        "same as scrolling on canvas."));
    gtk_box_pack_start(GTK_BOX(props_box), curvature, FALSE, FALSE, 0);
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
    gtk_box_pack_start(GTK_BOX(props_box), size, FALSE, FALSE, 0);

    GtkWidget *hardness = _pending_conf_slider_new(
      module, DT_MASKS_CONF(form->type, brush, hardness),
      _blend_masks_properties[DT_MASKS_PROPERTY_HARDNESS].name,
      _blend_masks_properties[DT_MASKS_PROPERTY_HARDNESS].min,
      _blend_masks_properties[DT_MASKS_PROPERTY_HARDNESS].max, 2,
      _blend_masks_properties[DT_MASKS_PROPERTY_HARDNESS].format,
      _("hardness of the next brush stroke -- same as shift+scrolling on canvas."));
    gtk_box_pack_start(GTK_BOX(props_box), hardness, FALSE, FALSE, 0);
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
    gtk_box_pack_start(GTK_BOX(props_box), sm, FALSE, FALSE, 0);
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
    gtk_box_pack_start(GTK_BOX(props_box), cl, FALSE, FALSE, 0);
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
      gtk_box_pack_start(GTK_BOX(props_box), pressure, FALSE, FALSE, 0);
    }

    GtkWidget *smoothing = dt_gui_preferences_enum(DT_ACTION(module), "brush_smoothing");
    dt_bauhaus_widget_set_label(smoothing, N_("blend"), N_("smoothing"));
    gtk_box_pack_start(GTK_BOX(props_box), smoothing, FALSE, FALSE, 0);
  }

  gtk_box_pack_start(GTK_BOX(row_vbox), props_box, FALSE, FALSE, 0);

  gtk_widget_show_all(row_vbox);
  return row_vbox;
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

// render one empty group as a header, column-aligned with the shape rows and the
// real group headers: [mode chip] | label | [within-group chooser (disabled)].
// An empty group has no members, so there is nothing to solo/mute yet -- that
// column is simply omitted, same as a populated group. Clicking selects it;
// right-click removes it; dropping a shape onto it fills (realizes) the group.
static void _pack_empty_group_header(dt_iop_module_t *module,
                                     dt_masks_empty_group_t *eg,
                                     const gboolean is_base)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  const int opstate = eg->op;
  const gboolean selected = (bd->selected_empty == eg);

  // within-group combine chooser placeholder (disabled: no members to combine
  // yet), shown on the right to match the populated header layout
  GtkWidget *within_sel = _make_within_selector(module, NULL, eg->within, FALSE);

  // "<operator>-<id>" -- same format as a populated group's own title (see its
  // own labevt further down). It used to also append " · empty" here, but that
  // extra text roughly doubled the string length while sharing the exact same
  // fixed 50px column + PANGO_ELLIPSIZE_MIDDLE budget as every other row kind,
  // so the ellipsis reliably chewed the whole thing down to a meaningless tail
  // fragment (e.g. "…pty" from "empty"). The row already reads as an empty
  // placeholder via its own styling/tooltip, so the label doesn't need to say
  // it too. Empty groups carry an id too (shared per-operator numbering with
  // the real groups), so every group is identified by its id.
  const int gord = _group_ordinal_any(module, INVALID_MASKID, eg);
  gchar *egtxt = eg->name ? g_strdup(eg->name)
                          : g_strdup_printf("%s-%d", _op_name_for_state(opstate), gord);
  GtkWidget *lbl = gtk_label_new(egtxt);
  g_free(egtxt);
  gtk_label_set_xalign(GTK_LABEL(lbl), 0.0f);
  // ellipsize + a fixed title-column width, exactly matching a populated
  // group's own header (see its own labevt further down) -- without this,
  // an empty group's title claims whatever width it needs instead of the
  // same fixed column every other row shares, which is what threw off both
  // the icon-to-title spacing and the within-group chooser's own position
  // (packed right after this column, see opacity_inner below).
  gtk_label_set_ellipsize(GTK_LABEL(lbl), PANGO_ELLIPSIZE_MIDDLE);
  gtk_label_set_max_width_chars(GTK_LABEL(lbl), 1);
  // wrapped in a box, exactly like a populated group's own lbl_box, so
  // ctrl+click can swap the label for a rename entry in place (see
  // _empty_header_press / _group_rename_commit)
  GtkWidget *lbl_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
  dt_gui_add_class(lbl_box, "mask-row-name");
  gtk_box_pack_start(GTK_BOX(lbl_box), lbl, TRUE, TRUE, 0);
  g_object_set_data(G_OBJECT(lbl_box), "title-child", lbl);
  GtkWidget *labevt = gtk_event_box_new();
  // windowless: the label must not capture the button-press/motion stream, or
  // the header's drag source (on hdr_evbox) never arms when the user grabs the
  // label text -- the natural place to grab a row to drag it (see the real
  // group header's labevt for the same reasoning).
  gtk_event_box_set_visible_window(GTK_EVENT_BOX(labevt), FALSE);
  gtk_container_add(GTK_CONTAINER(labevt), lbl_box);
  gtk_widget_set_size_request(labevt, DT_PIXEL_APPLY_DPI(50), -1);
  // expands to absorb whatever width the opacity slider below doesn't need
  // (see _control_column_size_allocate) -- the 50dpi request above is just a
  // floor so it never gets squeezed to nothing on an unusually narrow row.
  gtk_widget_set_hexpand(labevt, TRUE);
  // the last remaining group cannot be removed (see _empty_header_press), so
  // don't offer it when this is the only one left
  gchar *egtip = g_strdup_printf(
    _("empty group - select it, then draw a shape (or drop one here) to fill it\n"
      "%s"
      "drag the row to rearrange\n"
      "ctrl+click to rename\n"
      "it has no shapes yet, so it contributes nothing%s"),
    _group_count(module) > 1 ? _("right-click to remove this group\n") : "",
    is_base ? _(", and its operator would have no effect once filled either -- see "
                "the lead icon's own tooltip")
            : "");
  gtk_widget_set_tooltip_text(labevt, egtip);
  g_free(egtip);
  g_object_set_data(G_OBJECT(labevt), "eg", eg);

  GtkWidget *hdr = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
  // unique per-kind id (#mask-empty-header-row) so CSS/lookups can target this
  // row kind directly, without combinator rules against other row kinds that
  // used to share the generic "mask-list-row" name; .mask-panel-row is the
  // shared class every row/header kind keeps for their common base styling.
  gtk_widget_set_name(hdr, "mask-empty-header-row");
  dt_gui_add_class(hdr, "mask-panel-row");
  // a subtle resting background distinct from plain element rows, so this
  // reads as a group heading even when nothing is selected (see
  // .mask-group-header in darktable.css)
  dt_gui_add_class(hdr, "mask-group-header");
  // NB: "mask-list-row-selected" is applied to `block` (below, once it exists),
  // not to `hdr` here -- same split a real group's header uses (group_block
  // vs hdr, see its own "header-widget"/"group-header-widget" tags): applying
  // it to hdr alone gave hdr its own independent 4-sided bright border,
  // visible as a seam at hdr's own bottom edge once a body (a real element,
  // or this row's own pending-shape placeholder) sits below it, instead of
  // one bright border wrapping the whole header+body block.
  // an empty group has no members, so it can never itself be the solo target --
  // while any solo is active it always dims, exactly like a real group whose
  // every member is hidden (see all_hidden in the real-group header build)
  if(dt_is_valid_maskid(bd->solo_formid) || bd->solo_group_key != 0)
    gtk_widget_set_opacity(hdr, 0.45);
  GtkWidget *ehandle_btn = NULL;
  GtkWidget *ehandle =
    _make_op_combo(&ehandle_btn, _op_paint_for_state(opstate),
                   is_base ? NULL : G_CALLBACK(_empty_between_op_press));
  dt_gui_remove_class(ehandle, "mask-op-combo");
  dt_gui_add_class(ehandle, "mask-within-combo");
  dt_gui_add_class(ehandle, "mask-group-lead-handle");
  if(is_base)
  {
    dt_gui_add_class(ehandle, "mask-lead-static");
    dt_gui_add_class(ehandle_btn, "mask-lead-static");
    dt_gui_add_class(ehandle_btn, "dt_no_hover");
  }
  gtk_widget_set_valign(ehandle, GTK_ALIGN_CENTER);

  g_object_set_data(G_OBJECT(ehandle_btn), "module", module);
  g_object_set_data(G_OBJECT(ehandle_btn), "eg", eg);
  if(is_base)
    g_object_set_data(G_OBJECT(ehandle_btn), "is-base-group", GINT_TO_POINTER(1));
  gtk_widget_set_tooltip_text(
    ehandle_btn,
    is_base
      ? _("between-group combine: the base group has no predecessor to combine with, "
          "so its operator has no effect -- it always contributes its own mask")
      : _("between-group combine: how this (once filled) group's mask "
          "will combine with the stack accumulated by every group below it\n"
          "click to change"));

  // a disabled opacity slider, matching a populated group's own header
  // exactly -- an empty group has no members to scale yet, but showing the row
  // without one just made it visibly thinner than every other row, not
  // meaningfully different in what it offers.
  GtkWidget *opacity_slider = dt_bauhaus_slider_new_with_range(
    module, _blend_masks_properties[DT_MASKS_PROPERTY_OPACITY].min,
    _blend_masks_properties[DT_MASKS_PROPERTY_OPACITY].max, 0, 1.0f, 2);
  dt_bauhaus_widget_set_label(opacity_slider, N_("blend"), N_("opacity"));
  dt_bauhaus_slider_set_format(opacity_slider, "%");
  dt_bauhaus_slider_set_digits(opacity_slider, 2);
  dt_bauhaus_widget_set_quad_visibility(opacity_slider, FALSE);
  dt_bauhaus_widget_hide_label(opacity_slider);
  dt_gui_add_class(opacity_slider, "mask-props-slider");
  dt_gui_add_class(opacity_slider, "mask-inline-opacity");
  _style_opacity_gradient(opacity_slider);
  gtk_widget_set_valign(opacity_slider, GTK_ALIGN_CENTER);
  DT_ENTER_GUI_UPDATE(); // populate only, no listener attached anyway
  dt_bauhaus_slider_set(opacity_slider, eg->opacity);
  DT_LEAVE_GUI_UPDATE();
  _group_opacity_update_tooltip(opacity_slider, eg->opacity);
  gtk_widget_set_sensitive(opacity_slider, FALSE);

  GtkWidget *val_widget = _make_inline_opacity_value_widget(opacity_slider, module);
  gtk_widget_set_sensitive(val_widget, FALSE);
  gtk_widget_set_no_show_all(opacity_slider, TRUE);
  gtk_widget_hide(opacity_slider);

  GtkWidget *opacity_inner = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
  gtk_box_pack_start(GTK_BOX(opacity_inner), opacity_slider, FALSE, FALSE, 0);
  gtk_box_pack_end(GTK_BOX(opacity_inner), val_widget, TRUE, TRUE, 0);
  gtk_widget_set_halign(val_widget, GTK_ALIGN_END);
  gtk_widget_set_valign(opacity_inner, GTK_ALIGN_CENTER);

  _pack_row_header(hdr, ehandle, labevt, opacity_inner,
                   _make_badge_stack(_make_lowop_badge(), _make_solo_status_badge()),
                   within_sel, NULL);

  // an event box wraps the header so clicking selects (release) and a shape,
  // real group, or another empty group can be dropped onto it; right-click
  // (press) removes it. It is also a drag source for its own reorder (like a
  // real group's hdr_evbox), so grabbing anywhere on the row moves it.
  // a lone group has nowhere to reorder to, so it is not a drag source
  const gboolean movable = _group_count(module) >= 2;
  GtkWidget *hdr_evbox = _make_group_header_evbox(
    module, hdr, lbl_box, G_CALLBACK(_empty_header_press),
    G_CALLBACK(_empty_header_release), G_CALLBACK(_masks_empty_header_drag_received),
    movable ? _mask_empty_dnd : NULL, movable ? G_CALLBACK(_masks_empty_drag_get) : NULL);
  g_object_set_data(G_OBJECT(hdr_evbox), "eg", eg);
  if(is_base) g_object_set_data(G_OBJECT(hdr_evbox), "is-base-group", GINT_TO_POINTER(1));
  // tagged so _apply_empty_selection (a lightweight, no-rebuild selection update)
  // can find this row and toggle its highlight in place
  g_object_set_data(G_OBJECT(hdr_evbox), "eg-header", GINT_TO_POINTER(1));
  // "group-header-widget" (-> hdr, set by the helper) is for solo dimming
  // (_apply_empty_group_dimming); "header-widget" (-> block, set below once it
  // exists) is for selection shading, matching the same two-tag split a real
  // group's own header uses -- keeps a selected empty group's highlight
  // wrapping its whole body (header + any pending-shape placeholder row)
  // instead of hdr's own edges alone.
  GtkWidget *block = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
  // a staged group is a group: its block carries the same id/class as a real
  // one, which is what gives it the shared spacing and -- the reason this is
  // unconditional rather than only when a pending row exists, as it used to be
  // -- the drop-indicator styling. Without the class the insertion line is
  // applied to a widget no rule matches, so dragging near a staged group showed
  // no indicator at all.
  gtk_widget_set_name(block, "mask-group-block");
  dt_gui_add_class(block, "mask-group-block");
  gtk_box_pack_start(GTK_BOX(block), hdr_evbox, FALSE, FALSE, 0);

  // highlight this group's frame while a drag hovers it. The frame is the whole
  // block, not just the header row: a staged group's block can also hold the
  // pending-shape placeholder row, and the insertion line has to span the
  // group's full body to read as landing above/below the group rather than
  // above/below its header. This also has to be the same widget
  // _group_drop_above measures against (it resolves "header-widget", set just
  // below) -- drawing the line on one rectangle while deciding above/below from
  // another is what made the indicator flip mid-group.
  g_signal_connect(G_OBJECT(hdr_evbox), "drag-motion", G_CALLBACK(_group_drop_motion),
                   block);
  g_signal_connect(G_OBJECT(hdr_evbox), "drag-leave", G_CALLBACK(_group_drop_leave),
                   block);
  // see the "header-widget"/"group-header-widget" split comment above: this
  // tag is what _apply_empty_selection shades, and it must be the whole
  // block (matching a real group's own group_block target) so a selected
  // empty group's highlight wraps its header and body as one card.
  g_object_set_data(G_OBJECT(hdr_evbox), "header-widget", block);
  if(selected) dt_gui_add_class(block, "mask-list-row-selected");

  // if a shape is currently being drawn and this is the empty group it would
  // land in (see _recompute_insert_hint), show its disposable placeholder row
  // right under this header -- exactly where the real row will appear once
  // it commits.
  {
    const dt_masks_form_gui_t *fg = darktable.develop->form_gui;
    dt_masks_form_t *pending = (fg && fg->creation && fg->creation_module == module)
                                 ? darktable.develop->form_visible
                                 : NULL;
    // bd->insert_empty is never actually assigned a real value anywhere in
    // this file (always NULL) -- the live target is bd->selected_empty, with
    // insert_empty only reserved as a fallback for a "no explicit selection"
    // case that isn't wired up yet (see its own field comment in blend.h and
    // the same fallback pattern in the "realize" block above). Mirror that
    // fallback here rather than relying on insert_empty alone.
    const dt_masks_empty_group_t *target_eg =
      bd->selected_empty ? bd->selected_empty : bd->insert_empty;
    if(pending && bd->insert_active && bd->insert_realize_empty && target_eg == eg)
    {
      // wrap in the same "mask-group-elements" indent a real group's elements
      // box carries (see _build_masks_list's own elem_box), so the pending
      // row reads as nested inside this group instead of sitting flush with
      // the header at the group's own indent level.
      GtkWidget *pending_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
      gtk_widget_set_name(pending_box, "mask-group-elements");
      dt_gui_add_class(pending_box, "masks-list");
      dt_gui_add_class(pending_box, "mask-group-elements");
      gtk_box_pack_start(GTK_BOX(pending_box), _make_pending_shape_row(module, pending),
                         FALSE, FALSE, 0);
      gtk_box_pack_start(GTK_BOX(block), pending_box, FALSE, FALSE, 0);
      // the "one visual card" framing this body needs (.mask-group-block) is
      // now applied unconditionally where the block is created above
    }
  }

  gtk_box_pack_end(GTK_BOX(bd->masks_list_box), block, FALSE, FALSE, 0);
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
  _update_shape_row_state(bd, row_vbox, pt);
  // if this is a parametric form, flip its own editor's slider markers to match
  GtkWidget *editor_box =
    row_vbox ? g_object_get_data(G_OBJECT(row_vbox), "param-editor-box") : NULL;
  dt_masks_param_row_editor_t *ed =
    editor_box ? g_object_get_data(G_OBJECT(editor_box), "param-editor") : NULL;
  if(ed) _update_param_row_display(ed);
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
static void _close_shape_actions_menu(GtkWidget *item)
{
  GtkWidget *menu = gtk_widget_get_ancestor(item, GTK_TYPE_MENU);
  if(menu) gtk_menu_popdown(GTK_MENU(menu));
}

static void _shape_menu_toggle_disable(GtkCheckMenuItem *item, dt_iop_module_t *module)
{
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(item), "formid"));
  _close_shape_actions_menu(GTK_WIDGET(item));
  _toggle_element_disable(module, id);
}

static void _shape_menu_toggle_invert(GtkCheckMenuItem *item, dt_iop_module_t *module)
{
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(item), "formid"));
  _invert_element(module, id);
  _close_shape_actions_menu(GTK_WIDGET(item));
}

static void _shape_menu_toggle_solo(GtkCheckMenuItem *item, dt_iop_module_t *module)
{
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(item), "formid"));
  _toggle_solo_form(module, id);
  _close_shape_actions_menu(GTK_WIDGET(item));
}

static void _shape_menu_toggle_soloedit(GtkCheckMenuItem *item, dt_iop_module_t *module)
{
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(item), "formid"));
  _toggle_soloedit(module, id);
  _close_shape_actions_menu(GTK_WIDGET(item));
}

static void _shape_menu_rename(GtkMenuItem *item, dt_iop_module_t *module)
{
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(item), "formid"));
  GtkWidget *evbox = g_object_get_data(G_OBJECT(item), "evbox");
  if(evbox) _start_rename_element(evbox, module, id);
}

static void _shape_menu_delete(GtkMenuItem *item, dt_iop_module_t *module)
{
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(item), "formid"));
  _delete_single_shape(module, id);
}

// build (but do not show) a shape row's actions menu: consolidates the row's
// various actions -- what used to be a separate solo-edit icon, plus invert,
// solo, shift+click properties, right-click delete and ctrl+click rename --
// into one discoverable menu, opened by a plain click on the row's own lead
// handle (mirrors a group's operator chip opening its own menu on click, see
// _build_group_op_menu). The individual gestures besides solo/solo-edit still
// work too; this is an additional, more discoverable way to reach the same
// actions, not a replacement for them.
static GtkWidget *_build_shape_actions_menu(dt_iop_module_t *module,
                                            const dt_mask_id_t id,
                                            GtkWidget *handle,
                                            GtkWidget *evbox)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  const dt_masks_point_group_t *pt = grp ? _group_point(grp, id) : NULL;
  // solo edit isolates on-canvas nodes/handles for editing -- meaningless for
  // a parametric channel or a raster mask, neither of which has any canvas
  // geometry of its own to isolate (see _toggle_soloedit).
  dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, id);
  const gboolean is_drawn_shape =
    form && !(form->type & (DT_MASKS_PARAMETRIC | DT_MASKS_RASTER));

  GtkWidget *menu = gtk_menu_new();

  // visibility section
  _add_menu_section_header(menu, _("visibility"), FALSE);

  const gboolean elem_disabled = pt && (pt->state & DT_MASKS_STATE_DISABLE);
  GtkWidget *disable_item = gtk_check_menu_item_new_with_label(_("disable"));
  gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(disable_item), elem_disabled);
  gtk_widget_set_tooltip_text(
    disable_item, _("temporarily disable this element: it keeps its properties and its "
                    "place in the group, but contributes nothing to the mask"));
  g_object_set_data(G_OBJECT(disable_item), "formid", GINT_TO_POINTER(id));
  g_signal_connect(G_OBJECT(disable_item), "toggled",
                   G_CALLBACK(_shape_menu_toggle_disable), module);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), disable_item);

  if(!elem_disabled)
  {
    GtkWidget *solo_item = gtk_check_menu_item_new_with_label(_("solo"));
    gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(solo_item), bd->solo_formid == id);
    g_object_set_data(G_OBJECT(solo_item), "formid", GINT_TO_POINTER(id));
    g_signal_connect(G_OBJECT(solo_item), "toggled", G_CALLBACK(_shape_menu_toggle_solo),
                     module);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), solo_item);

    if(is_drawn_shape)
    {
      GtkWidget *soloedit_item = gtk_check_menu_item_new_with_label(_("solo edit"));
      gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(soloedit_item),
                                     bd->soloedit_formid == id);
      g_object_set_data(G_OBJECT(soloedit_item), "formid", GINT_TO_POINTER(id));
      g_signal_connect(G_OBJECT(soloedit_item), "toggled",
                       G_CALLBACK(_shape_menu_toggle_soloedit), module);
      gtk_menu_shell_append(GTK_MENU_SHELL(menu), soloedit_item);
    }

    // mask operations section
    _add_menu_section_header(menu, _("mask operations"), TRUE);

    GtkWidget *invert_item = gtk_check_menu_item_new_with_label(_("invert"));
    gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(invert_item),
                                   pt && (pt->state & DT_MASKS_STATE_INVERSE));
    g_object_set_data(G_OBJECT(invert_item), "formid", GINT_TO_POINTER(id));
    g_signal_connect(G_OBJECT(invert_item), "toggled",
                     G_CALLBACK(_shape_menu_toggle_invert), module);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), invert_item);
  }

  // edit section
  _add_menu_section_header(menu, _("edit"), TRUE);

  GtkWidget *rename_item = gtk_menu_item_new_with_label(_("rename"));
  g_object_set_data(G_OBJECT(rename_item), "formid", GINT_TO_POINTER(id));
  g_object_set_data(G_OBJECT(rename_item), "evbox", evbox);
  g_signal_connect(G_OBJECT(rename_item), "activate", G_CALLBACK(_shape_menu_rename),
                   module);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), rename_item);

#ifdef HAVE_AI
  // a multi-path AI mask (see _register_vectorized_forms/object.c) offers an
  // escape hatch for users who want full manual control over its individual
  // paths instead of the bundle's coordinated feather/size/rotation.
  if(form && (form->type & DT_MASKS_OBJECT) && g_list_length(form->points) > 1)
  {
    GtkWidget *break_item = gtk_menu_item_new_with_label(_("break into components"));
    gtk_widget_set_tooltip_text(
      break_item, _("convert this AI mask's paths into ordinary, independently "
                    "editable shapes in this group"));
    g_object_set_data(G_OBJECT(break_item), "formid", GINT_TO_POINTER(id));
    g_signal_connect(G_OBJECT(break_item), "activate",
                     G_CALLBACK(_shape_menu_break_apart), module);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), break_item);
  }
#endif

  GtkWidget *delete_item = gtk_menu_item_new_with_label(_("delete"));
  g_object_set_data(G_OBJECT(delete_item), "formid", GINT_TO_POINTER(id));
  g_signal_connect(G_OBJECT(delete_item), "activate", G_CALLBACK(_shape_menu_delete),
                   module);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), delete_item);

  gtk_widget_show_all(menu);
  return menu;
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
                                 GtkWidget *container,
                                 GList *fids,
                                 GList *group_formids,
                                 GtkWidget *group_frame);

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

// dock whichever slider belongs in the row's own header bar right now: the
// compact input slider while collapsed (see _make_shape_row, which wires up
// ed->header_slot once), full-width like the header slot's own free space --
// or the opacity slider while expanded, right-aligned and capped instead,
// mirroring a shape/raster row's own inline opacity treatment -- "for symmetry
// and to save vertical space" per the user's request, so expanding a parametric
// row's properties no longer grows it by a whole extra line. Either slider's
// *other* home (compact_row for the input slider, opacity_box for opacity, see
// _apply_param_row_filter_layout / the opacity slider's own construction above)
// is where _reparent_into pulls it back from -- nothing needs an explicit
// "undock" step, it simply isn't asked to move this time. header_slot itself
// stays visible either way -- it is always the row's one expanding child (see
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
dt_masks_param_vis_t _model_param_row_visibility(const gboolean expanded,
                                                 const gboolean in_used,
                                                 const gboolean out_used,
                                                 const gboolean boost_enabled)
{
  dt_masks_param_vis_t v = { TRUE, FALSE, FALSE, FALSE };

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
                                channel && channel->boost_factor_enabled);
  const gboolean show_input = vis.input;
  const gboolean show_output = vis.output;
  const gboolean show_boost = vis.boost;
  const gboolean show_bypass = vis.bypass;

  if(ed->input_lbl) gtk_widget_set_visible(ed->input_lbl, show_input);
  if(ed->input_slot) gtk_widget_set_visible(ed->input_slot, show_input);
  if(ed->input_bypass_btn)
  {
    gtk_widget_set_visible(ed->input_bypass_btn, show_input);
    gtk_widget_set_opacity(ed->input_bypass_btn, show_bypass ? 1.0 : 0.0);
    gtk_widget_set_sensitive(ed->input_bypass_btn, show_bypass);
  }
  if(ed->output_lbl) gtk_widget_set_visible(ed->output_lbl, show_output);
  if(ed->output_slot) gtk_widget_set_visible(ed->output_slot, show_output);
  if(ed->output_bypass_btn)
  {
    gtk_widget_set_visible(ed->output_bypass_btn, show_output);
    gtk_widget_set_opacity(ed->output_bypass_btn, show_bypass ? 1.0 : 0.0);
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
  if(ed->opacity_box) gtk_widget_set_visible(ed->opacity_box, FALSE);
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
                      _("double-click to reset.\n"
                        "press 'a' to toggle available slider modes.\n"
                        "press 'c' to toggle view of channel data.\n"
                        "press 'm' to toggle mask view."));
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
                    _("double-click to reset.\n"
                      "press 'a' to toggle available slider modes.\n"
                      "press 'c' to toggle view of channel data.\n"
                      "press 'm' to toggle mask view."));
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
static float _param_row_slider_precise_display(
  const dt_iop_gui_blendif_channel_t *channel, const float boost_factor, const float frac)
{
  if(channel->scale_print == _blendif_scale_print_hue) return frac * 360.0f;
  if(channel->scale_print == _blendif_scale_print_ab)
    return (frac * 256.0f - 128.0f) * boost_factor;
  return frac * boost_factor * 100.0f; // _blendif_scale_print_default
}

static float _param_row_slider_precise_parse(const dt_iop_gui_blendif_channel_t *channel,
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

// the embedded slider (see _param_row_slider_precise_open) has no valid
// *allocated* on-screen size yet at the point it is first mapped -- bauhaus's
// own popup sizes itself off that allocation (see _popup_show in bauhaus.c,
// specifically its "p->width == 1" fallback for an as-yet-unallocated
// widget, which falls back to the *whole host panel's* width instead, wildly
// oversized for a single range-slider node) -- GTK only assigns the real
// allocation in the size-allocate pass that follows mapping, not atomically
// with it. Deferred to a plain idle instead of the "map" signal so this runs
// strictly after that pass has actually happened (GTK services its own
// pending resizes at a higher priority than G_PRIORITY_DEFAULT_IDLE, so by
// the time this fires the widget's allocation is the real one).
// bauhaus's own slider popup (see _popup_show in bauhaus.c) always opens
static gboolean _param_row_slider_precise_open_idle(gpointer user_data)
{
  GtkWidget *bauhaus_slider = user_data;
  if(!GTK_IS_WIDGET(bauhaus_slider)) return G_SOURCE_REMOVE;
  GtkWidget *popover =
    g_object_get_data(G_OBJECT(bauhaus_slider), "precise-anchor-popover");
  dt_bauhaus_widget_show_popup(bauhaus_slider);
  if(popover)
    g_signal_connect(G_OBJECT(darktable.bauhaus->popup.window), "hide",
                     G_CALLBACK(_param_row_slider_precise_popup_hidden), popover);
  GtkWidget *slider =
    g_object_get_data(G_OBJECT(bauhaus_slider), "precise-anchor-slider");
  const gint marker_x = GPOINTER_TO_INT(
    g_object_get_data(G_OBJECT(bauhaus_slider), "precise-anchor-marker-x"));
  if(slider && GTK_IS_WIDGET(slider)) _place_bauhaus_whisker_popup(slider, marker_x);
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
  // _param_row_slider_precise_place), so the normal hover/drag-driven
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
  dt_bauhaus_slider_set_format(bauhaus_slider, is_hue ? _(" °") : is_ab ? "" : "%");
  dt_bauhaus_widget_hide_label(bauhaus_slider);
  dt_bauhaus_widget_set_quad_visibility(bauhaus_slider, FALSE);
  dt_bauhaus_slider_set_val(bauhaus_slider, cur);
  // fill the popover's own width fully -- a bauhaus widget's halign otherwise
  // defaults to only claiming its own (narrow) natural width even inside an
  // expand+fill box slot.
  gtk_widget_set_hexpand(bauhaus_slider, TRUE);
  gtk_widget_set_halign(bauhaus_slider, GTK_ALIGN_FILL);
  // this popup opens away from the pointer on purpose (see
  // _param_row_slider_precise_place, positioned against the row's own
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
  // this node's own on-screen x, for _param_row_slider_precise_place to
  // center the real popup on (clamped to the mask panel's own bounds), not
  // the row -- computed now, from the slider's real (already allocated)
  // position, rather than recomputed later from the row/marker fraction.
  {
    GtkWidget *slider_top = gtk_widget_get_toplevel(slider);
    gint sx = 0, sy = 0;
    gtk_widget_translate_coordinates(slider, slider_top, 0, 0, &sx, &sy);
    GdkWindow *slider_top_gdk =
      gtk_widget_is_toplevel(slider_top) ? gtk_widget_get_window(slider_top) : NULL;
    gint tx = 0, ty = 0;
    if(slider_top_gdk) gdk_window_get_origin(slider_top_gdk, &tx, &ty);
    const gint marker_x =
      tx + sx + gslider->margin_left + (gint)(gslider->position[k] * usable);
    g_object_set_data(G_OBJECT(bauhaus_slider), "precise-anchor-marker-x",
                      GINT_TO_POINTER(marker_x));
  }

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
  // keep the one visible button's own look in sync with whichever of the two
  // real (hidden) pickers this "toggled" came from -- see
  // _param_row_master_picker_pressed.
  if(ed->master_picker)
  {
    const gboolean active =
      gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(ed->colorpicker))
      || gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(ed->colorpicker_set_values));
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
  const gboolean right =
    gtk_gesture_single_get_current_button(GTK_GESTURE_SINGLE(gesture))
    == GDK_BUTTON_SECONDARY;
  if(ctrl)
    dt_color_picker_click(ed->colorpicker, right);
  else if(!right)
    dt_color_picker_click(ed->colorpicker_set_values, FALSE);
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
    const int in_out = ((dt_key_modifier_state() == GDK_SHIFT_MASK) && p->in_out) ? 1 : 0;

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

  gtk_widget_set_tooltip_text(GTK_WIDGET(sl->slider),
                              _("double-click to reset.\n"
                                "press 'a' to toggle available slider modes.\n"
                                "press 'c' to toggle view of channel data.\n"
                                "press 'm' to toggle mask view."));
  gtk_widget_set_tooltip_text(GTK_WIDGET(sl->head), _(slider_tooltip[in_out]));

  sl->head_compact = GTK_LABEL(dt_ui_label_new(in_out ? _("output") : _("input")));
  gtk_widget_set_tooltip_text(GTK_WIDGET(sl->head_compact), _(slider_tooltip[in_out]));
  sl->compact_row = NULL;
  sl->box = NULL;
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
    // back-reference so _param_row_editor_channel (see
    // _blendop_blendif_channel_mask_view) can resolve THIS row's own channel from the
    // slider alone, instead of the removed classic editor's shared
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
  GtkWidget *picker_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
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
  gtk_box_pack_start(GTK_BOX(picker_box), ed->master_picker, FALSE, FALSE, 0);

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

  // opacity slider: per the user's spec, expanding a parametric row's
  // in/out chevron also reveals opacity (unlike shape/raster/group rows,
  // which get their own separate expander -- see _make_props_row_toggle).
  // Delta-applied via the shared _props_row_apply, same protocol as every
  // other row kind's opacity control. Docks into the row's own header_slot
  // while expanded, right alongside the between-groups-style inline sliders
  // shape/raster/group rows show -- "for symmetry and to save vertical
  // space" (see _update_param_row_header_dock) -- so it is styled the same
  // inline way (label/value hidden, tooltip stands in for them) rather than
  // the labeled, below-row style boost_box uses.
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
  dt_bauhaus_widget_hide_label(ed->opacity_slider);
  ed->opacity_last_value = dt_bauhaus_slider_get(ed->opacity_slider);
  g_object_set_data(G_OBJECT(ed->opacity_slider), "dt-prop",
                    GINT_TO_POINTER(DT_MASKS_PROPERTY_OPACITY));
  g_signal_connect(G_OBJECT(ed->opacity_slider), "value-changed",
                   G_CALLBACK(_param_row_opacity_changed), ed);
  // same background-occlusion fix as the shape/group properties sliders and
  // the boost-factor slider: without it this slider's own opaque pill paints
  // over the row's hover/selection wash. .mask-inline-opacity matches the
  // margin every other row kind's inline opacity slider uses.
  dt_gui_add_class(ed->opacity_slider, "mask-props-slider");
  dt_gui_add_class(ed->opacity_slider, "mask-inline-opacity");
  _style_opacity_gradient(ed->opacity_slider);
  g_signal_connect(G_OBJECT(ed->opacity_slider), "value-changed",
                   G_CALLBACK(_inline_opacity_tooltip_changed), NULL);
  // opacity_slider's only real home is header_slot, docked there while
  // expanded (see _update_param_row_header_dock) -- opacity_box is just a
  // parking spot for it the rest of the time (collapsed, or before this row
  // even has a header_slot yet), permanently hidden: unlike boost_box it is
  // never itself shown, so it carries none of boost_box's below-row margin
  // styling.
  ed->opacity_box = dt_gui_vbox(ed->opacity_slider);

  GtkWidget *sliders_grid = gtk_grid_new();
  gtk_grid_set_column_homogeneous(GTK_GRID(sliders_grid), FALSE);
  gtk_grid_set_column_spacing(GTK_GRID(sliders_grid), DT_PIXEL_APPLY_DPI(4));
  gtk_grid_set_row_spacing(GTK_GRID(sliders_grid), DT_PIXEL_APPLY_DPI(2));

  GtkWidget *input_lbl = dt_ui_label_new(_("input"));
  gtk_label_set_xalign(GTK_LABEL(input_lbl), 0.0f);
  dt_gui_add_class(input_lbl, "mask-param-channel-label");
  gtk_grid_attach(GTK_GRID(sliders_grid), input_lbl, 0, 0, 1, 1);
  ed->input_lbl = input_lbl;

  GtkWidget *input_slot = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
  gtk_widget_set_hexpand(input_slot, TRUE);
  gtk_widget_set_valign(GTK_WIDGET(ed->filter[0].slider), GTK_ALIGN_CENTER);
  gtk_box_pack_start(GTK_BOX(input_slot), GTK_WIDGET(ed->filter[0].slider), TRUE, TRUE,
                     0);
  gtk_grid_attach(GTK_GRID(sliders_grid), input_slot, 1, 0, 1, 1);
  ed->input_slot = input_slot;

  GtkWidget *input_bypass_btn =
    dtgtk_togglebutton_new(dtgtk_cairo_paint_eye_toggle, 0, NULL);
  dt_gui_add_class(input_bypass_btn, "mask-refine-bypass-btn");
  gtk_widget_set_valign(input_bypass_btn, GTK_ALIGN_CENTER);
  gtk_widget_set_tooltip_text(input_bypass_btn,
                              _("temporarily disable this input channel"));
  g_signal_connect(G_OBJECT(input_bypass_btn), "toggled",
                   G_CALLBACK(_param_channel_bypass_toggled), ed);
  gtk_grid_attach(GTK_GRID(sliders_grid), input_bypass_btn, 2, 0, 1, 1);
  ed->input_bypass_btn = input_bypass_btn;

  GtkWidget *output_lbl = dt_ui_label_new(_("output"));
  gtk_label_set_xalign(GTK_LABEL(output_lbl), 0.0f);
  dt_gui_add_class(output_lbl, "mask-param-channel-label");
  gtk_grid_attach(GTK_GRID(sliders_grid), output_lbl, 0, 1, 1, 1);
  ed->output_lbl = output_lbl;

  GtkWidget *output_slot = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
  gtk_widget_set_hexpand(output_slot, TRUE);
  gtk_widget_set_valign(GTK_WIDGET(ed->filter[1].slider), GTK_ALIGN_CENTER);
  gtk_box_pack_start(GTK_BOX(output_slot), GTK_WIDGET(ed->filter[1].slider), TRUE, TRUE,
                     0);
  gtk_grid_attach(GTK_GRID(sliders_grid), output_slot, 1, 1, 1, 1);
  ed->output_slot = output_slot;

  GtkWidget *output_bypass_btn =
    dtgtk_togglebutton_new(dtgtk_cairo_paint_eye_toggle, 0, NULL);
  dt_gui_add_class(output_bypass_btn, "mask-refine-bypass-btn");
  gtk_widget_set_valign(output_bypass_btn, GTK_ALIGN_CENTER);
  gtk_widget_set_tooltip_text(output_bypass_btn,
                              _("temporarily disable this output channel"));
  g_signal_connect(G_OBJECT(output_bypass_btn), "toggled",
                   G_CALLBACK(_param_channel_bypass_toggled), ed);
  gtk_grid_attach(GTK_GRID(sliders_grid), output_bypass_btn, 2, 1, 1, 1);
  ed->output_bypass_btn = output_bypass_btn;

  ed->sliders_grid = sliders_grid;

  GtkWidget *wrap = dt_gui_vbox(sliders_grid, ed->boost_box, ed->opacity_box);
  // id mirrors the class for direct CSS targeting alongside the existing
  // class-based rules (shared by every parametric row's own editor instance)
  gtk_widget_set_name(wrap, "mask-param-row-editor");
  dt_gui_add_class(wrap, "mask-param-row-editor");

  _update_param_row_display(ed);
  g_object_set_data_full(G_OBJECT(wrap), "param-editor", ed, g_free);

  gtk_widget_show_all(wrap);
  gtk_widget_set_no_show_all(ed->input_lbl, TRUE);
  gtk_widget_set_no_show_all(ed->input_slot, TRUE);
  gtk_widget_set_no_show_all(ed->input_bypass_btn, TRUE);
  gtk_widget_set_no_show_all(ed->output_lbl, TRUE);
  gtk_widget_set_no_show_all(ed->output_slot, TRUE);
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

static GtkWidget *_make_shape_row(dt_iop_module_t *module,
                                  dt_masks_point_group_t *fpt,
                                  dt_masks_form_t *form,
                                  GList *group_formids,
                                  GtkWidget *group_frame)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  const dt_mask_id_t fid = fpt->formid;
  GtkWidget *row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);

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
  // raster/parametric row, no "toggle expanded controls" for a raster row,
  // which has no separate expander any more).
  const gboolean is_drawn_shape = !(form->type & (DT_MASKS_PARAMETRIC | DT_MASKS_RASTER));
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
      ? g_strdup(_("click to select, click again to deselect\n"
                   "ctrl+click to rename\n"
                   "right-click to open the actions menu "
                   "(invert, solo, rename, delete)\n"
                   "drag to rearrange, or onto a group to move "
                   "this raster mask into it"))
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
  GtkWidget *handle = channel_code
                        ? _make_channel_handle(channel_code, row_tip)
                        : _make_drag_handle(_kind_icon_paint(kind), TRUE, row_tip);
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
  // properties expander: every shape and raster row gets one (see
  // _make_props_row_toggle); parametric rows do not -- their existing in/out
  // toggle above (soloedit, in this branch) already reveals opacity too.
  // The chevron button itself is no longer shown on any row kind (see
  // expand_toggle below) -- it stays alive, hidden, purely as the state
  // holder its own "toggled" handler already knows how to drive.
  GtkWidget *props_toggle = NULL;
  GtkWidget *props_editor_box = NULL;
  // shape/raster rows' own opacity slider, shown directly, always, inline in
  // the header next to the name (see _style_inline_opacity_box) -- mirrors
  // the group header's own treatment. NULL for a parametric row, which
  // already reveals opacity through its own in/out chevron instead.
  GtkWidget *opacity_box = NULL;
  // the row's own properties/expanded-view toggle, whichever widget that is
  // for this row kind (see below) -- shift+click on the lead handle or the
  // title (see _row_click_release / _row_click_release) drives it
  // programmatically instead of a visible chevron button.
  GtkWidget *expand_toggle = NULL;
  if(form->type & DT_MASKS_PARAMETRIC)
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

      if(ped->opacity_slider)
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
    // a raster mask has no on-canvas geometry to solo-edit, and opacity (its
    // only property -- modify_property is NULL for a raster form) is now
    // always shown inline in the header instead of behind an expander, so
    // this slot has nothing left to hold.
    soloedit = NULL;
    opacity_box = _style_inline_opacity_box(
      _build_props_row_editor(module, fid, FALSE, TRUE, FALSE), module);
  }
  else
  {
    // no more dedicated solo-edit icon -- solo-edit (like every other
    // gesture this row offers) is now also reachable from the actions menu
    // opened by a plain click on the row's own lead handle (see
    // _build_shape_actions_menu / _row_click_press).
    soloedit = NULL;

    // opacity is shown directly, always, inline in the header next to the
    // name (see _style_inline_opacity_box) -- everything else this shape has
    // (size, hardness, feather, rotation, curvature, compression, cleanup,
    // smoothing, refine-mask-boundary) stays behind its own separate
    // expander, excluding opacity so it is not editable a second time from
    // there -- see _make_props_row_toggle.
    opacity_box = _style_inline_opacity_box(
      _build_props_row_editor(module, fid, FALSE, TRUE, FALSE), module);
    const char *props_tip =
      (form->type & DT_MASKS_OBJECT)
        ? _("show/hide this object's expanded controls (smoothing, cleanup, etc.)")
        : _("show/hide this shape's expanded controls (size, hardness, etc.)");
    props_toggle = _make_props_row_toggle(module, fid, FALSE, FALSE, TRUE, props_tip,
                                          &props_editor_box);
    expand_toggle = props_toggle;
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
                                     : bd->soloedit_formid == fid
                                       ? MASK_SOLO_BADGE_SOLOEDIT
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

  GtkWidget *action_icon = (form->type & DT_MASKS_PARAMETRIC) ? param_picker_box : NULL;
  _pack_row_header(row, handle, evbox, opacity_box,
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
  GtkWidget *row_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
  // unique per-kind id (#mask-shape-row) -- shared by every element kind (drawn
  // shape, parametric, raster), which all go through this same function; see
  // .mask-panel-row in darktable.css for the base styling shared with every
  // other row/header kind in the panel.
  gtk_widget_set_name(row_vbox, "mask-shape-row");
  dt_gui_add_class(row_vbox, "mask-panel-row");
  gtk_box_pack_start(GTK_BOX(row_vbox), row_evbox, FALSE, FALSE, 0);
  g_object_set_data(G_OBJECT(row_evbox), "row-vbox", row_vbox);
  g_object_set_data(G_OBJECT(evbox), "row-vbox", row_vbox);

  g_object_set_data(G_OBJECT(row_vbox), "mask-row", GINT_TO_POINTER(1));
  g_object_set_data(G_OBJECT(row_vbox), "formid", GINT_TO_POINTER(fid));
  // index this row for O(1) lookup by form id (see _masks_row_widget); the map is
  // cleared at the top of _build_masks_list, so entries never outlive their widget.
  if(bd->masks_row_map)
    g_hash_table_insert(bd->masks_row_map, GINT_TO_POINTER(fid), row_vbox);
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
  if(bd->solo_formid == fid || bd->soloedit_formid == fid)
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
    gtk_box_pack_start(GTK_BOX(row_vbox), param_evbox, FALSE, FALSE, 0);
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
    gtk_box_pack_start(GTK_BOX(row_vbox), props_evbox, FALSE, FALSE, 0);
  }

  // this element belongs to a bypassed group (the bypass bit is broadcast onto
  // every member, so fpt carries it): the group contributes nothing to the mask,
  // so none of this row's controls can have any visible effect. Grey them out
  // and dim the row, exactly as a solo-suppressed row is (see
  // _update_shape_row_state). Only the editors and the actions column are made
  // insensitive -- never row_vbox/row_evbox itself, so the row stays selectable
  // and draggable, the same carve-out solo makes.
  if(_op_is_bypassed(fpt->state))
  {
    gtk_widget_set_opacity(row, 0.45);
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

static void _consolidate_cluster_in_group(dt_masks_form_t *grp,
                                          const dt_mask_id_t *fids_in_cluster,
                                          int count)
{
  if(!grp || count < 2) return;
  int base_pos = -1, idx = 0;
  for(GList *l = grp->points; l; l = g_list_next(l), idx++)
  {
    const dt_masks_point_group_t *pt = l->data;
    for(int j = 0; j < count; j++)
    {
      if(pt->formid == fids_in_cluster[j])
      {
        base_pos = idx;
        break;
      }
    }
    if(base_pos >= 0) break;
  }
  if(base_pos < 0) return;

  GList *pts = NULL;
  for(int j = 0; j < count; j++)
  {
    for(GList *l = grp->points; l; l = g_list_next(l))
    {
      dt_masks_point_group_t *pt = l->data;
      if(pt->formid == fids_in_cluster[j])
      {
        pts = g_list_append(pts, pt);
        break;
      }
    }
  }

  gboolean already_contiguous = TRUE;
  idx = base_pos;
  for(GList *l = pts; l; l = g_list_next(l), idx++)
  {
    GList *node = g_list_nth(grp->points, idx);
    if(!node || node->data != l->data)
    {
      already_contiguous = FALSE;
      break;
    }
  }

  if(!already_contiguous)
  {
    for(GList *l = pts; l; l = g_list_next(l))
      grp->points = g_list_remove(grp->points, l->data);

    int pos = base_pos;
    for(GList *l = pts; l; l = g_list_next(l), pos++)
      grp->points = g_list_insert(grp->points, l->data, pos);
  }

  g_list_free(pts);
}

// A hash of everything _build_masks_list builds the tree from: the mask model
// (dt_masks_group_hash already folds every point's formid/state/opacity/
// refinement in order plus each leaf form's own config, so add/delete/reorder/
// operator/opacity/refinement/solo-via-HIDDEN and parametric config all move it)
// plus the UI-state the build consults (mask mode, empty-group scaffolding,
// selection/solo, cluster/props expansion, and the transient realize/seed
// triggers). When this is unchanged since the last build the rebuilt tree would
// be byte-identical, so the whole teardown/rebuild can be skipped. The
// realize/seed/auto-select blocks the build runs before packing are no-ops in
// steady state, and any pending one is flagged here (insert_realized_fid /
// scaffold_seeded), so a top-of-function signature is safe.
static dt_hash_t _masks_list_signature(dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);

  dt_hash_t sig = dt_masks_group_hash(DT_INITHASH, grp);

  // dt_masks_group_hash does not fold form->name or orphan point states, but
  // the rows display them, so fold each member form's state and name in points order.
  for(GList *p = grp ? grp->points : NULL; p; p = g_list_next(p))
  {
    const dt_masks_point_group_t *pt = p->data;
    sig = dt_hash(sig, &pt->formid, sizeof(pt->formid));
    sig = dt_hash(sig, &pt->state, sizeof(pt->state));
    sig = dt_hash(sig, &pt->group_start, sizeof(pt->group_start));
    sig = dt_hash(sig, &pt->group_opacity, sizeof(pt->group_opacity));
    sig = dt_hash(sig, &pt->opacity, sizeof(pt->opacity));
    sig = dt_hash(sig, &pt->refinement, sizeof(pt->refinement));
    const dt_masks_form_t *f = dt_masks_get_from_id(darktable.develop, pt->formid);
    if(f && f->name[0]) sig = dt_hash(sig, f->name, strlen(f->name));
    if(pt->name[0]) sig = dt_hash(sig, pt->name, strlen(pt->name));
  }

  const uint32_t mode = module->blend_params->mask_mode;
  sig = dt_hash(sig, &mode, sizeof(mode));

  // empty groups (order matters: bottom-anchored ones pack first)
  for(GList *e = bd->empty_groups; e; e = g_list_next(e))
  {
    const dt_masks_empty_group_t *eg = e->data;
    const int32_t v[3] = { (int32_t)eg->op, (int32_t)eg->within, (int32_t)eg->below_fid };
    sig = dt_hash(sig, v, sizeof(v));
    // staged group refinement: drives the refinement caption/sliders for a
    // selected empty group, so a change must move the signature
    sig = dt_hash(sig, &eg->refinement, sizeof(eg->refinement));
    // the header displays eg->name (see its own labevt build) exactly like a
    // populated group's custom pt->name is folded in above -- without this a
    // rename left the signature unchanged (nothing else about the panel
    // moved), so the rebuild that swaps the rename entry back for the label
    // got skipped as a no-op, and Enter appeared to do nothing.
    if(eg->name && eg->name[0]) sig = dt_hash(sig, eg->name, strlen(eg->name));
  }

  // selection / solo (drive per-row/per-header CSS classes and badges) and the
  // transient realize/seed triggers the build acts on before packing.
  const int32_t ui[7] = {
    bd->panel_selected_formid,   bd->panel_selected_group_cid, bd->solo_formid,
    (int32_t)bd->solo_group_key, bd->soloedit_formid,          bd->insert_realized_fid,
    (int32_t)bd->scaffold_seeded
  };
  sig = dt_hash(sig, ui, sizeof(ui));
  // bd->selected_empty: which empty group (if any) is selected. Not covered by
  // the "empty groups" loop above (that hashes each group's op/within/below_fid,
  // not which one is selected) -- without this, re-selecting a *different*
  // already-existing empty group leaves the signature unchanged whenever
  // nothing else about the panel moved, so the rebuild that would move the
  // selection highlight gets skipped and the previous group's row stays
  // (wrongly) marked as selected. Hashing the pointer itself is fine: it only
  // needs to change value when the selected group changes within this session.
  sig = dt_hash(sig, &bd->selected_empty, sizeof(bd->selected_empty));

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

  // realize: a shape was just drawn into the selected empty group. Drop the empty
  // group, re-anchor the empties that sat above it onto the new run, and select
  // the realized run so continued drawing keeps filling it.
  if(flexi && grp && dt_is_valid_maskid(bd->insert_realized_fid))
  {
    const dt_mask_id_t newfid = bd->insert_realized_fid;
    bd->insert_realized_fid = INVALID_MASKID;
    // the realized empty is the explicitly selected one, or (no selection) the
    // foundation empty that the insert hint defaulted to
    dt_masks_empty_group_t *eg =
      bd->selected_empty ? bd->selected_empty : bd->insert_empty;
    // refinement staged on the group while it had no members (see
    // dt_masks_empty_group_t.refinement); carried over to the realized run below
    dt_masks_refinement_t staged = { 0 };
    // likewise its number, so filling a group does not renumber it
    int staged_ord = 0;
    // and its custom name, if it was named while still empty
    gchar *staged_name = NULL;
    if(eg && g_list_find(bd->empty_groups, eg))
    {
      staged = eg->refinement;
      staged_ord = eg->ordinal;
      staged_name = eg->name;
      eg->name =
        NULL; // ownership moves to staged_name, don't let _empty_group_free take it
      GList *node = g_list_find(bd->empty_groups, eg);
      for(GList *l = node ? node->next : NULL; l; l = g_list_next(l))
      {
        dt_masks_empty_group_t *s = l->data;
        if(s->below_fid == eg->below_fid) s->below_fid = newfid;
      }
      bd->empty_groups = g_list_remove(bd->empty_groups, eg);
      _empty_group_free(eg);
    }
    bd->selected_empty = NULL;
    bd->insert_empty = NULL;
    // adopt the staged refinement: broadcast onto every member of the new run,
    // exactly as REFINE_SCOPE_GROUP does for an already-populated group
    if(staged.enabled)
    {
      GList *ids = _selected_group_formids(grp, newfid);
      for(GList *l = ids; l; l = g_list_next(l))
      {
        dt_masks_point_group_t *pt = _group_point(grp, GPOINTER_TO_INT(l->data));
        if(pt) pt->refinement = staged;
      }
      g_list_free(ids);
    }
    // adopt the staged name the same way -- broadcast onto every member of the
    // new run, mirroring _group_rename_commit's own broadcast for a real group
    if(staged_name)
    {
      GList *ids = _selected_group_formids(grp, newfid);
      for(GList *l = ids; l; l = g_list_next(l))
      {
        dt_masks_point_group_t *pt = _group_point(grp, GPOINTER_TO_INT(l->data));
        if(pt) g_strlcpy(pt->name, staged_name, sizeof(pt->name));
      }
      g_list_free(ids);
      g_free(staged_name);
    }
    // select the realized run (its cid = its bottom member, the run's lowest in
    // grp->points order)
    GList *run = _selected_group_formids(grp, newfid);
    if(run)
    {
      GList *bottom = g_list_last(run);
      const dt_mask_id_t newcid = GPOINTER_TO_INT(bottom->data);
      bd->panel_selected_group_cid = newcid;
      // hand the empty group's number to the run it just became
      if(staged_ord > 0)
      {
        if(!bd->group_ordinals)
          bd->group_ordinals = g_hash_table_new(g_direct_hash, g_direct_equal);
        g_hash_table_insert(bd->group_ordinals, GINT_TO_POINTER(newcid),
                            GINT_TO_POINTER(staged_ord));
      }
      g_list_free(run);
    }
  }

  // virgin flexi mask: seed only the base (union) group. An empty / reset mask has
  // exactly one group -- the permanent foundation -- so the first element drawn or
  // imported lands in it automatically (no selection needed). Further groups are
  // added explicitly with the "+" button.
  //
  // NB: this deliberately does NOT test bd->scaffold_seeded. That latch only
  // exists so a scaffold the user dismissed does not come back on every rebuild,
  // and it cannot apply here: the foundation group is permanent (right-clicking
  // it to delete is refused), so "flexi mode with no group at
  // all" is never a state the user can ask for -- it is only ever reached
  // accidentally. Deleting the last remaining group empties grp->points, which
  // makes dt_masks_form_remove() destroy the group form and reset
  // blend_params.mask_id to NO_MASKID (masks.c); with the latch already set from
  // the first build, nothing was re-seeded and have_content below went FALSE,
  // hiding the entire list until an explicit "add group" brought it back.
  if(flexi && (!grp || !grp->points) && !bd->empty_groups)
  {
    bd->empty_groups = g_list_append(
      bd->empty_groups, _empty_group_new(DT_MASKS_STATE_UNION, 0, INVALID_MASKID));
    bd->scaffold_seeded = TRUE;
  }

  // group numbers are identities, not positions: forget the ones whose group is
  // gone, then number any group still without one (see _group_ordinal_any). Runs
  // after the realize/seeding blocks above -- which create and retire groups --
  // and before any packing, so every header and caption below reads the same,
  // already-settled number.
  if(flexi)
  {
    _prune_group_ordinals(module);
    _assign_group_ordinals(module);
  }
  _prune_stale_solo(module);

  const gboolean have_content = (grp && grp->points) || bd->empty_groups;

  // one line per rebuild describing what the panel is about to render from, so a
  // panel that goes blank/empty can be told apart from a mask that really lost
  // its content (the two look identical on screen, see the drop logs below)
  dt_print(DT_DEBUG_MASKS,
           "[masks] panel rebuild '%s': mask_id=%d grp=%s points=%d empties=%d"
           " flexi=%d content=%d",
           module->op, module->blend_params->mask_id, grp ? "ok" : "NULL",
           grp ? g_list_length(grp->points) : -1, g_list_length(bd->empty_groups),
           flexi ? 1 : 0, have_content ? 1 : 0);
  if(!have_content) return FALSE;

  // a group can be deselected (selection toggles), so the panel may legitimately
  // have nothing selected -- even when there is only one group in total, so
  // that case is not special-cased into a forced *re*selection on every rebuild
  // (with nothing selected, refinement targets the whole mask -- see
  // _flexi_refine_follow_selection -- and new elements still default to the
  // sole group -- see _update_add_target_sensitivity / _recompute_insert_hint's
  // own single-group fallback). But the very first time the panel has content,
  // default-select the sole group once (masks_selection_seeded), so opening the
  // panel is ready to add elements without an extra click; any later explicit
  // deselect sticks since this does not run again.
  if(!bd->masks_selection_seeded)
  {
    bd->masks_selection_seeded = TRUE;
    if(!dt_is_valid_maskid(bd->panel_selected_group_cid) && !bd->selected_empty)
    {
      GList *heads = _group_partition_heads(grp);
      const int n_real = g_list_length(heads);
      const int n_empty = g_list_length(bd->empty_groups);
      if(n_real + n_empty == 1)
      {
        if(n_real == 1)
          bd->panel_selected_group_cid = GPOINTER_TO_INT(heads->data);
        else
          bd->selected_empty = bd->empty_groups->data;
      }
      g_list_free(heads);
    }
  }

  // refresh the insert hint now (not just at the very end, its other call
  // site) so it reflects any selection change made just above, in time for
  // the pending-row placement below to target the right group/empty group.
  // Idempotent/side-effect-free to call twice in one pass.
  _recompute_insert_hint(module);
  return TRUE;
}

// Widget-side: build the panel's row tree from the already-reconciled model.
// Every mutation this used to perform now happens in _masks_panel_reconcile
// above, so this only reads.
static void
_masks_panel_pack(dt_iop_module_t *module, dt_masks_form_t *grp, const gboolean flexi)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;

  // the single shape currently being drawn on canvas for this module (if
  // any) -- not a real grp->points member yet, rendered as a disposable
  // placeholder row instead (see _make_pending_shape_row). NULL whenever
  // nothing is being drawn, or it belongs to a different module.
  const dt_masks_form_gui_t *pending_fg = darktable.develop->form_gui;
  dt_masks_form_t *pending_form =
    (pending_fg && pending_fg->creation && pending_fg->creation_module == module)
      ? darktable.develop->form_visible
      : NULL;

  DT_ENTER_GUI_UPDATE();

  if(!bd->masks_cluster_expanded)
    bd->masks_cluster_expanded = g_hash_table_new(g_direct_hash, g_direct_equal);

  // empty groups whose anchor is missing (or INVALID) render at the very bottom;
  // pack them first. The rest are packed right above their anchor run in the loop.
  GList *bottom_empties = NULL;
  if(flexi)
    for(GList *e = bd->empty_groups; e; e = g_list_next(e))
    {
      dt_masks_empty_group_t *eg = e->data;
      gboolean anchored = FALSE;
      if(grp && dt_is_valid_maskid(eg->below_fid))
        anchored = (_group_point(grp, eg->below_fid) != NULL);
      if(!anchored) bottom_empties = g_list_append(bottom_empties, eg);
    }
  for(GList *e = bottom_empties; e; e = g_list_next(e))
    // unanchored empties always sit below every real group (see the loop
    // just above), so the very first one is the true structural base --
    // matching is_base_group's own "l == grp->points" check for a real
    // group's own head, just on this separate bottom_empties list instead
    _pack_empty_group_header(module, e->data, e == bottom_empties);
  g_list_free(bottom_empties);

  // groups pass: one header per maximal same-operator run, with that run's element
  // rows nested (indented) directly under the header -- each group owns its elements
  // inline (no separate elements panel). grp->points is bottom-up (head = base, no
  // operator); we walk it run by run and pack each group block with pack_end so the
  // base stays at the bottom and newer groups appear on top.
  dt_masks_state_t prev_op = 0; // operator of the run just below (for chooser disabling)
  // the total group count is loop-invariant, so compute it once here instead of
  // re-walking grp->points for every group header inside the loop below.
  const int ngroups = _group_count(module);
  GList *l = grp ? grp->points : NULL;
  while(l)
  {
    const gboolean is_base_group = (l == grp->points);
    const dt_masks_state_t op = _eff_group_op(((dt_masks_point_group_t *)l->data)->state);

    // collect this run's resolvable member ids + aggregate state. first_fid is the
    // bottom member (first in points order) and serves as the group's cid.
    GList *formids = NULL; // member ids, top-first (g_list_prepend)
    int run = 0;
    dt_mask_id_t first_fid = INVALID_MASKID;
    gboolean all_hidden = TRUE, all_screen = TRUE, all_isect = TRUE;
    GList *m = l;
    for(; m; m = g_list_next(m))
    {
      dt_masks_point_group_t *pm = m->data;
      if(m != l && _starts_group(m)) break;
      dt_masks_form_t *fm = dt_masks_get_from_id(darktable.develop, pm->formid);
      if(!fm)
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
      if(!dt_is_valid_maskid(first_fid)) first_fid = pm->formid;
      formids = g_list_prepend(formids, GINT_TO_POINTER(pm->formid));
      run++;
      if(!(pm->state & DT_MASKS_STATE_HIDDEN)) all_hidden = FALSE;
      if(!(pm->state & DT_MASKS_STATE_SCREEN)) all_screen = FALSE;
      if(!(pm->state & DT_MASKS_STATE_ISECT)) all_isect = FALSE;
    }
    // the group's aggregate within-group combine mode: all members must agree,
    // else it reads as union (the mixed/neutral case)
    const dt_masks_state_t group_within =
      all_isect ? DT_MASKS_STATE_ISECT : (all_screen ? DT_MASKS_STATE_SCREEN : 0);
    // m now points to the head of the next run (or NULL)
    const dt_masks_state_t next_op =
      m ? _eff_group_op(((dt_masks_point_group_t *)m->data)->state) : 0;

    if(run == 0)
    {
      // every member of this run was unresolvable, so the whole group header is
      // skipped -- this is what "the group disappeared from the panel" looks like
      dt_print(DT_DEBUG_MASKS,
               "[masks] panel: '%s' run starting at %d has no resolvable member"
               " -- group header dropped",
               module->op, ((dt_masks_point_group_t *)l->data)->formid);
      l = m;
      continue;
    }

    const guint cid = (guint)first_fid;
    const int opstate = (int)op;
    // a bypassed group contributes nothing, so nothing inside it can have any
    // visible effect: everything below is built insensitive except the operator
    // handle, which is the way back (see the sensitivity block after `hdr`).
    const gboolean group_bypassed = _op_is_bypassed(opstate);
    // persistent "true" group invert (DT_MASKS_STATE_OP_INVERT, see
    // _group_toggle_output_invert) -- unlike group_bypassed this does not
    // affect what is built below (an inverted group still contributes to the
    // mask, just flipped), only the handle's look and its tooltip.
    const gboolean group_inverted = (opstate & DT_MASKS_STATE_OP_INVERT) != 0;

    // first-class groups: two same-operator groups may sit adjacent (kept apart by
    // GROUP_BREAK), so the chooser no longer disables neighbouring operators --
    // which is why prev_op/next_op are computed but unused, and why the handle no
    // longer carries a "disabled-ops" mask (it was always 0, and nothing read it).
    (void)prev_op;
    (void)next_op;

    // within-group combine chooser (union / screen / intersect) -- packed on the
    // RIGHT of the header (see below) so it is not confused with the group's own
    // between-group operator chip, which is the handle in column 0.
    GtkWidget *within_sel = _make_within_selector(module, formids, group_within, TRUE);

    // label: "<operator>-<id>" -- the operator name and the group's per-operator
    // id (shared with empty groups and the refinement caption). Once the group
    // is given a custom name (ctrl+click the title, masks v8) that replaces the
    // default label outright rather than being appended to it -- the "<op>-<id>"
    // form only exists as a placeholder until the user names the thing. No
    // disclosure triangle (groups don't expand).
    const int gord = _group_ordinal_of_cid(module, (dt_mask_id_t)cid);
    const char *custom_name = _group_custom_name(grp, (dt_mask_id_t)cid);
    gchar *txt = custom_name
                   ? g_strdup(custom_name)
                   : g_strdup_printf("%s-%d", _op_name_for_state(opstate), gord);
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
    g_object_set_data_full(G_OBJECT(group_solo_badge), "formids", g_list_copy(formids),
                           (GDestroyNotify)g_list_free);
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
    GtkWidget *lbl_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    dt_gui_add_class(lbl_box, "mask-row-name");
    gtk_box_pack_start(GTK_BOX(lbl_box), lbl, TRUE, TRUE, 0);
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
    gtk_widget_set_tooltip_text(labevt, _("click to select this group\n"
                                          "ctrl+click to rename\n"
                                          "drag the row to rearrange\n"
                                          "right-click to open the group's actions menu "
                                          "(also reachable from the lead icon), which "
                                          "includes \"solo\": use only this group"));

    // column 0 -- drag handle, doubling as the operator chip (ctrl+click invert,
    // shift+click show/hide this group's opacity (same as the title, see
    // labevt's tooltip above), plain click changes the operator -- see
    // _group_op_press/_group_op_release). The base (bottom) group is no
    // longer a fixed foundation: any group can end up there (see the seed
    // placeholder row below the list), and every operator can be picked for
    // it -- its own operator is simply never evaluated (see
    // _group_get_mask_roi_flexi), so it always contributes exactly its own
    // mask, whatever is shown here. With only one group there is nothing to
    // reorder against, so dragging is disabled (ngroups is computed once
    // before the loop).
    const gboolean group_movable = ngroups >= 2;
    // explicit about what this operator actually does -- it combines this
    // group's own (within-group) mask onto the stack accumulated by every
    // group below it, the same way the within-group chooser spells out what
    // union/screen/intersect mean for a group's own members (see
    // _make_within_selector) -- "click to change this group's operator" alone
    // did not make that relationship clear.
    // the base group is a special case worth spelling out unconditionally,
    // not just in the "click to change" line: it has no predecessor to
    // combine with, so its own operator is never evaluated at all -- it
    // always just contributes its own mask, whatever operator is shown on
    // it. Invert (this group's output, or an individual element) is how to
    gchar *ghandle_tip =
      is_base_group
        ? g_strdup(
            group_bypassed
              ? _("between-group combine: bypassed\n"
                  "this group is disabled: it keeps its shapes and its place, "
                  "but contributes nothing to the mask\n"
                  "the base group has no predecessor to combine with, "
                  "so its operator has no effect")
              : _("between-group combine: the base group has no predecessor to combine "
                  "with, "
                  "so its operator has no effect -- it always contributes its own mask"))
        : g_strdup_printf(
            group_bypassed
              ? _("between-group combine: %s (bypassed)\n"
                  "this group is disabled: it keeps its shapes and its "
                  "place, but contributes nothing to the mask\n"
                  "click to change operator\n"
                  "right-click for actions")
              : _("between-group combine: %s\n"
                  "how this group's mask combines with the "
                  "stack accumulated by every group below it\n"
                  "click to change operator\n"
                  "right-click for actions (solo, inverting, emptying and deleting)"),
            _op_name_for_state(opstate));
    GtkWidget *ghandle_btn = NULL;
    GtkWidget *ghandle =
      _make_op_combo(&ghandle_btn, _op_paint_for_state(opstate),
                     is_base_group ? NULL : G_CALLBACK(_group_between_op_press));
    dt_gui_remove_class(ghandle, "mask-op-combo");
    dt_gui_add_class(ghandle, "mask-within-combo");
    dt_gui_add_class(ghandle, "mask-group-lead-handle");
    if(is_base_group)
    {
      dt_gui_add_class(ghandle, "mask-lead-static");
      dt_gui_add_class(ghandle_btn, "mask-lead-static");
      dt_gui_add_class(ghandle_btn, "dt_no_hover");
    }
    gtk_widget_set_valign(ghandle, GTK_ALIGN_CENTER);

    if(group_inverted) dt_gui_add_class(ghandle_btn, "mask-list-handle-inverted");
    if(!group_movable) gtk_widget_set_opacity(ghandle, 0.6);
    g_object_set_data(G_OBJECT(ghandle_btn), "module", module);
    g_object_set_data_full(G_OBJECT(ghandle_btn), "formids", g_list_copy(formids),
                           (GDestroyNotify)g_list_free);
    if(is_base_group)
      g_object_set_data(G_OBJECT(ghandle_btn), "is-base-group", GINT_TO_POINTER(1));
    g_object_set_data(G_OBJECT(ghandle_btn), "title-label-box", lbl_box);
    g_object_set_data(G_OBJECT(ghandle_btn), "group-key", GUINT_TO_POINTER(cid));
    gtk_widget_set_tooltip_text(ghandle_btn, ghandle_tip);
    g_free(ghandle_tip);

    // opacity control: a persistent, multiplicative gain on this run's own
    // finished sub-mask (see dt_masks_point_group_t.group_opacity and
    // _group_get_mask_roi_flexi in group.c), applied on top of -- not instead
    // of -- each member's own independent opacity. Shown directly, always,
    // right next to the group's name instead of behind a properties expander
    // (there is nothing else to show for a group, unlike shape/raster/
    // parametric rows, see _make_props_row_toggle). An absolute value bound
    // straight to the persisted field via _group_opacity_changed, unlike the
    // delta convention every multi-target properties row uses
    // (_props_row_apply): a group header always represents exactly one run,
    // so there is no multi-select ambiguity to resolve. The label/value are
    // hidden to keep the header compact -- the tooltip stands in for them
    // (see _group_opacity_update_tooltip), refreshed live on every drag tick.
    GtkWidget *group_opacity_slider = dt_bauhaus_slider_new_with_range(
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
    g_object_set_data_full(G_OBJECT(group_opacity_slider), "formids",
                           g_list_copy(formids), (GDestroyNotify)g_list_free);
    g_signal_connect(G_OBJECT(group_opacity_slider), "value-changed",
                     G_CALLBACK(_group_opacity_changed), module);
    g_signal_connect(G_OBJECT(group_opacity_slider), "button-press-event",
                     G_CALLBACK(_group_opacity_press), module);

    GtkWidget *hdr = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    // unique per-kind id (#mask-group-header-row); .mask-panel-row is the
    // shared base styling class every row/header kind in the panel keeps
    gtk_widget_set_name(hdr, "mask-group-header-row");
    dt_gui_add_class(hdr, "mask-panel-row");
    // a subtle resting background distinct from plain element rows, so this
    // reads as a group heading even when nothing is selected (see
    // .mask-group-header in darktable.css)
    dt_gui_add_class(hdr, "mask-group-header");
    if(group_solo) dt_gui_add_class(hdr, "mask-list-row-solo");

    GtkWidget *group_val_widget =
      _make_inline_opacity_value_widget(group_opacity_slider, module);
    gtk_widget_set_no_show_all(group_opacity_slider, TRUE);
    gtk_widget_hide(group_opacity_slider);

    GtkWidget *group_opacity_inner = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_box_pack_start(GTK_BOX(group_opacity_inner), group_opacity_slider, FALSE, FALSE,
                       0);
    gtk_box_pack_end(GTK_BOX(group_opacity_inner), group_val_widget, FALSE, FALSE, 0);
    gtk_widget_set_halign(group_val_widget, GTK_ALIGN_END);
    gtk_widget_set_valign(group_opacity_inner, GTK_ALIGN_CENTER);

    const gboolean has_selected =
      (dt_is_valid_maskid(bd->panel_selected_formid)
       && g_list_find(formids, GINT_TO_POINTER(bd->panel_selected_formid)) != NULL);

    const gboolean group_expanded =
      has_selected || !bd->masks_props_expanded
      || !g_hash_table_contains(bd->masks_props_expanded, GUINT_TO_POINTER(cid))
      || GPOINTER_TO_INT(
        g_hash_table_lookup(bd->masks_props_expanded, GUINT_TO_POINTER(cid)));

    if(group_expanded && bd->masks_props_expanded)
      g_hash_table_insert(bd->masks_props_expanded, GUINT_TO_POINTER(cid),
                          GINT_TO_POINTER(TRUE));

    GtkWidget *group_expand_toggle = dtgtk_togglebutton_new(_paint_param_inout, 0, NULL);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(group_expand_toggle), group_expanded);
    dt_gui_add_class(group_expand_toggle, "dt_transparent_background");
    dt_gui_add_class(group_expand_toggle, "mask-row-expander");
    gtk_widget_set_tooltip_text(group_expand_toggle,
                                _("show/hide this group's elements"));
    g_object_set_data(G_OBJECT(group_expand_toggle), "props-key", GUINT_TO_POINTER(cid));
    g_signal_connect(G_OBJECT(group_expand_toggle), "toggled",
                     G_CALLBACK(_group_expand_toggled), module);

    _pack_row_header(hdr, ghandle, labevt, group_opacity_inner,
                     _make_badge_stack(group_lowop_badge, group_solo_badge), within_sel,
                     group_expand_toggle);
    // dimmed when the group contributes nothing: every element hidden, or the
    // whole group bypassed (in which case the badge stays at full opacity)
    if(group_bypassed)
    {
      gtk_widget_set_opacity(ghandle, 0.45);
      gtk_widget_set_opacity(labevt, 0.45);
      gtk_widget_set_opacity(group_opacity_inner, 0.45);
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
      gtk_widget_set_sensitive(within_sel, FALSE);
      gtk_widget_set_sensitive(group_opacity_slider, FALSE);
      gtk_widget_set_sensitive(group_val_widget, FALSE);
    }

    // an event box wraps the header so the canvas<->list hover sync can locate
    // this group by any member id (it carries "group-formids") and so clicking
    // it selects the group / right-clicking deletes it.
    GtkWidget *hdr_evbox = _make_group_header_evbox(
      module, hdr, lbl_box, G_CALLBACK(_group_header_press),
      G_CALLBACK(_group_header_release), G_CALLBACK(_masks_header_drag_received),
      group_movable ? _mask_group_dnd : NULL,
      group_movable ? G_CALLBACK(_masks_group_drag_get) : NULL);
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
    GtkWidget *block_inner = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_container_add(GTK_CONTAINER(group_block), block_inner);
    // id mirrors the class for direct CSS targeting alongside the existing
    // class-based rules (shared by every real group's own block instance)
    gtk_widget_set_name(group_block, "mask-group-block");
    dt_gui_add_class(group_block, "mask-group-block");
    gtk_box_pack_start(GTK_BOX(block_inner), hdr_evbox, FALSE, FALSE, 0);
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

    // highlight the whole group block when its group is the selected one
    if(dt_is_valid_maskid(bd->panel_selected_group_cid)
       && (dt_mask_id_t)cid == bd->panel_selected_group_cid)
      dt_gui_add_class(group_block, "mask-list-row-selected");

    GtkWidget *elem_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    // indent/inset entirely via CSS (.mask-group-elements's margin-left/
    // margin-right in darktable.css), not hardcoded here. id mirrors the
    // class for direct CSS targeting alongside the existing class-based rules
    // (shared by every real group's own elements-box instance)
    gtk_widget_set_name(elem_box, "mask-group-elements");
    dt_gui_add_class(elem_box, "masks-list");
    dt_gui_add_class(elem_box, "mask-group-elements");
    gtk_widget_set_visible(elem_box, group_expanded);
    g_object_set_data(G_OBJECT(group_expand_toggle), "elem-box", elem_box);
    _pack_group_elements(module, elem_box, g_list_reverse(g_list_copy(formids)), formids,
                         group_block);

    // if a shape is currently being drawn and this run is where it would land
    // (see _recompute_insert_hint), show its disposable placeholder row at the
    // top of this group's elements -- exactly where the real row lands once it
    // commits (a new element is inserted above the run's current top member).
    if(pending_form && bd->insert_active && !bd->insert_realize_empty
       && g_list_find(formids, GINT_TO_POINTER(bd->insert_after_fid)))
      gtk_box_pack_start(GTK_BOX(elem_box), _make_pending_shape_row(module, pending_form),
                         FALSE, FALSE, 0);

    gtk_box_pack_start(GTK_BOX(block_inner), elem_box, FALSE, FALSE, 0);

    gtk_box_pack_end(GTK_BOX(bd->masks_list_box), group_block, FALSE, FALSE, 0);

    // empty groups anchored above this run sit just above it in the list
    for(GList *e = bd->empty_groups; e; e = g_list_next(e))
    {
      dt_masks_empty_group_t *eg = e->data;
      gboolean match = FALSE;
      for(GList *mm = formids; mm; mm = g_list_next(mm))
        if(GPOINTER_TO_INT(mm->data) == eg->below_fid)
        {
          match = TRUE;
          break;
        }
      // anchored above a specific real run, so never the bottom-most (base)
      // slot -- only an unanchored empty (handled in the bottom_empties loop
      // above) can be that
      if(match) _pack_empty_group_header(module, eg, FALSE);
    }

    g_list_free(formids);
    prev_op = op;
    l = m;
  }

  // the box carries no_show_all (flexi-only), which makes gtk_widget_show_all on
  // the box itself a no-op; show each header explicitly, then reveal the box.
  GList *children = gtk_container_get_children(GTK_CONTAINER(bd->masks_list_box));
  for(GList *c = children; c; c = g_list_next(c))
    gtk_widget_show_all(GTK_WIDGET(c->data));
  g_list_free(children);
  gtk_widget_set_visible(GTK_WIDGET(bd->masks_list_box), TRUE);

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

  // reset the formid -> row index; it is repopulated as _make_shape_row builds
  // each row below. Cleared here (before any new rows) so it never holds a
  // pointer to a just-destroyed row.
  if(!bd->masks_row_map)
    bd->masks_row_map = g_hash_table_new(g_direct_hash, g_direct_equal);
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
  const gboolean flexi = module->blend_params->mask_mode & DEVELOP_MASK_FLEXI;
  dt_pthread_mutex_unlock(&darktable.develop->history_mutex);

  if(!_masks_panel_reconcile(module, grp, flexi))
  {
    gtk_widget_set_visible(GTK_WIDGET(bd->masks_list_box), FALSE);
    _recompute_insert_hint(module);
    return;
  }

  _masks_panel_pack(module, grp, flexi);
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
    _group_delete_shapes(module, members);
    return TRUE;
  }
  return FALSE;
}

// pack one group's element rows into `container`, nested under that group's header.
// `fids` is the run's member ids bottom-up (consumed/freed here). Same-kind drawn
// shapes fold into expand/collapse clusters (no actions); parametric forms are never
// folded (each keeps its own inline editor). `group_formids`/`group_frame` let
// every element row also double as a group/empty-group reorder drop target (see
// _make_shape_row): otherwise only the thin header row would accept such a drop,
// and dragging a group over any of a target group's own elements -- very easy to
// do by accident, since the header row is thin -- would be silently rejected.
static void _pack_group_elements(dt_iop_module_t *module,
                                 GtkWidget *container,
                                 GList *fids,
                                 GList *group_formids,
                                 GtkWidget *group_frame)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
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
    rows[nr] = _make_shape_row(module, fpt, form, group_formids, group_frame);
    kinds[nr] = _form_kind(form);
    fid_of[nr] = fid;
    nr++;
  }
  g_list_free(fids);

  // fold same-kind elements into expand/collapse clusters to cut clutter: any drawn
  // kind with >= 3 members becomes a single cluster gathering all of its members
  // (even when they are not adjacent in the run). Kinds with fewer members -- and
  // parametric masks, which are never clustered (each has its own inline editor) --
  // stay as individual rows. A cluster (or a lone row) is emitted at the position of
  // the kind's first (bottom-most, since fids is bottom-up) member; pack_end keeps
  // the bottom member at the bottom, reproducing the run order.
  const int cluster_min = 3;
  gboolean *emitted = g_malloc0_n(nr, sizeof(gboolean));
  for(int i = 0; i < nr; i++)
  {
    if(emitted[i]) continue;
    const guint kind = kinds[i];
    int count = 0;
    for(int k = i; k < nr; k++)
      if(kinds[k] == kind) count++;

    if(count < cluster_min || kind == DT_MASKS_PARAMETRIC || kind == DT_MASKS_RASTER)
    {
      // a sparse (or parametric/raster) kind: this member is a plain row; the rest
      // land at their own spots
      gtk_box_pack_end(GTK_BOX(container), rows[i], FALSE, FALSE, 0);
      emitted[i] = TRUE;
      continue;
    }

    // nested one level deeper than a plain (unclustered) element row, via CSS
    // (.mask-cluster-elements' own margin-left in darktable.css), so expanding
    // a cluster visually reads as revealing its members as its own children.
    GtkWidget *inner = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    // id mirrors the class for direct CSS targeting alongside the existing
    // class-based rules (shared by every cluster's own elements-box instance)
    gtk_widget_set_name(inner, "mask-cluster-elements");
    dt_gui_add_class(inner, "mask-cluster-elements");
    GList *member_fids = NULL;
    dt_mask_id_t *cluster_fids = g_malloc_n(count, sizeof(dt_mask_id_t));
    int ci = 0;
    gboolean contains_selected = FALSE;
    for(int k = i; k < nr; k++)
    {
      if(kinds[k] != kind) continue;
      gtk_box_pack_end(GTK_BOX(inner), rows[k], FALSE, FALSE, 0);
      emitted[k] = TRUE;
      member_fids = g_list_prepend(member_fids, GINT_TO_POINTER(fid_of[k]));
      cluster_fids[ci++] = fid_of[k];
      if(dt_is_valid_maskid(bd->panel_selected_formid)
         && fid_of[k] == bd->panel_selected_formid)
        contains_selected = TRUE;
    }
    _consolidate_cluster_in_group(grp, cluster_fids, count);
    g_free(cluster_fids);

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
    GtkWidget *lblbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, DT_PIXEL_APPLY_DPI(4));
    gtk_box_pack_start(GTK_BOX(lblbox), lbl, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(lblbox), arrow, FALSE, FALSE, 0);
    GtkWidget *chdr = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    // unique per-kind id (#mask-cluster-header-row); .mask-panel-row is the
    // shared base styling class every row/header kind in the panel keeps
    gtk_widget_set_name(chdr, "mask-cluster-header-row");
    dt_gui_add_class(chdr, "mask-panel-row");
    gtk_box_pack_start(GTK_BOX(chdr), kicon, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(chdr), lblbox, TRUE, TRUE, 0);
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

    GtkWidget *cbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_box_pack_start(GTK_BOX(cbox), hdr_evbox, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(cbox), rev, FALSE, FALSE, 0);
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
      g_signal_connect(G_OBJECT(cbox), "drag-data-received",
                       G_CALLBACK(_masks_header_drag_received), module);
      g_signal_connect(G_OBJECT(cbox), "drag-motion", G_CALLBACK(_group_drop_motion),
                       group_frame);
      g_signal_connect(G_OBJECT(cbox), "drag-leave", G_CALLBACK(_group_drop_leave),
                       group_frame);
    }
  }
  g_free(emitted);

  g_free(rows);
  g_free(kinds);
  g_free(fid_of);
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

// hovering a channel button alone does not preview anything -- exactly like
// hovering a legacy blendif slider (see _blendop_blendif_enter/_key_press):
// hovering only grabs focus and remembers the prior request_mask_display, so
// that a 'c' key press while hovering can toggle that channel's mask on, and
// leaving (after a short delay, so moving between adjacent buttons doesn't
// flicker) restores what was showing before.
static gboolean _param_channel_button_enter(GtkWidget *widget,
                                            GdkEventCrossing *event,
                                            gpointer user_data)
{
  dt_iop_module_t *module = user_data;
  dt_iop_gui_blend_data_t *bd = module->blend_data;

  dt_pthread_mutex_lock(&bd->lock);
  if(bd->timeout_handle)
  {
    g_source_remove(bd->timeout_handle);
    bd->timeout_handle = 0;
  }
  else if(!(bd->save_for_leave & DT_DEV_PIXELPIPE_DISPLAY_STICKY))
  {
    bd->save_for_leave = module->request_mask_display & ~DT_DEV_PIXELPIPE_DISPLAY_STICKY;
  }
  dt_pthread_mutex_unlock(&bd->lock);

  gtk_widget_grab_focus(widget);
  return FALSE;
}

// press 'c' while hovering a channel button to toggle a preview of that
// channel's mask -- same key, same gesture as the legacy blendif sliders
// (see _blendop_blendif_key_press's GDK_KEY_c case).
static gboolean
_param_channel_button_key_press(GtkWidget *widget, GdkEventKey *event, gpointer user_data)
{
  if(event->keyval != GDK_KEY_c && event->keyval != GDK_KEY_C) return FALSE;

  dt_iop_module_t *module = user_data;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  const int ch = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(widget), "param-channel"));
  const dt_iop_gui_blendif_channel_t *channels =
    dt_develop_blendif_channels_for_csp(bd->csp);
  if(!channels) return FALSE;
  int nch = 0;
  while(channels[nch].label) nch++;
  if(ch < 0 || ch >= nch) return FALSE;

  const dt_dev_pixelpipe_display_mask_t mode =
    DT_DEV_PIXELPIPE_DISPLAY_CHANNEL | channels[ch].display_channel;

  module->request_mask_display =
    (module->request_mask_display & ~DT_DEV_PIXELPIPE_DISPLAY_STICKY) == mode
      ? DT_DEV_PIXELPIPE_DISPLAY_NONE
      : mode;
  dt_iop_refresh_center(module);
  return TRUE;
}

static gboolean _param_channel_button_leave(GtkWidget *widget,
                                            GdkEventCrossing *event,
                                            gpointer user_data)
{
  dt_iop_module_t *module = user_data;
  dt_iop_gui_blend_data_t *bd = module->blend_data;

  dt_pthread_mutex_lock(&bd->lock);
  if(!(module->request_mask_display & DT_DEV_PIXELPIPE_DISPLAY_STICKY)
     && !bd->timeout_handle)
    bd->timeout_handle = g_timeout_add(1000, _blendop_blendif_leave_delayed, module);
  dt_pthread_mutex_unlock(&bd->lock);

  return FALSE;
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
    gtk_widget_add_events(btn, GDK_ENTER_NOTIFY_MASK | GDK_LEAVE_NOTIFY_MASK);
    g_signal_connect(G_OBJECT(btn), "enter-notify-event",
                     G_CALLBACK(_param_channel_button_enter), module);
    g_signal_connect(G_OBJECT(btn), "leave-notify-event",
                     G_CALLBACK(_param_channel_button_leave), module);
    g_signal_connect(G_OBJECT(btn), "key-press-event",
                     G_CALLBACK(_param_channel_button_key_press), module);
    gtk_widget_show(btn);
    gtk_box_pack_start(GTK_BOX(bd->masks_param_channels_inner), btn, FALSE, FALSE, 0);
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
                             const dt_mask_id_t id,
                             const char *srcname)
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

  // name the element after its source: the row shows the identifier the user
  // picked in the menu. Done AFTER save_creation because that calls the form's
  // set_form_name ("raster mask #N") during its de-dup numbering, which would
  // otherwise clobber this. form name = "<prefix> <srcname>" so
  // _form_display_name (strips the "raster mask" prefix) leaves the source name.
  if(srcname && *srcname)
  {
    snprintf(form->name, sizeof(form->name), "%s %s", _("raster mask"), srcname);
    dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  }

  _build_masks_list(self);
  // full reprocess so the (possibly newly-used) source recomputes and stores its
  // mask, and so this module's commit re-runs the source reconciliation
  if(reprocess) dt_dev_reprocess_all(self->dev);
}

static void _raster_menu_activate(GtkMenuItem *item, gpointer user_data)
{
  dt_iop_module_t *self = user_data;
  dt_iop_module_t *src = g_object_get_data(G_OBJECT(item), "raster-src");
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(item), "raster-id"));
  const char *srcname = g_object_get_data(G_OBJECT(item), "raster-name");
  _add_raster_mask(self, src, id, srcname);
}

// pop up a menu of every upstream module that offers a raster mask, mirroring
// the whole-mask raster source list (see _raster_combo_populate). Picking one
// adds (or retargets) this module's raster mask element.
static gboolean
_masks_raster_add_press(GtkWidget *button, GdkEventButton *event, dt_iop_module_t *module)
{
  if(DT_IN_GUI_UPDATE()) return TRUE;

  GtkWidget *menu = gtk_menu_new();
  int count = 0;
  for(GList *iter = darktable.develop->iop; iter; iter = g_list_next(iter))
  {
    dt_iop_module_t *iop = iter->data;
    if(iop == module) break; // only modules earlier in the pipe can be a source

    GHashTableIter masks_iter;
    gpointer key, value;
    g_hash_table_iter_init(&masks_iter, iop->raster_mask.source.masks);
    while(g_hash_table_iter_next(&masks_iter, &key, &value))
    {
      const dt_mask_id_t id = GPOINTER_TO_INT(key);
      // the mask's available identifier (module display name, or the mask
      // name/path for an external source): the same string the classic raster
      // picker shows (see _raster_combo_populate / dt_iop_advertise_rastermask)
      const char *name = value ? (const char *)value : iop->name();
      GtkWidget *mi = gtk_menu_item_new_with_label(name);
      g_object_set_data(G_OBJECT(mi), "raster-src", iop);
      g_object_set_data(G_OBJECT(mi), "raster-id", GINT_TO_POINTER(id));
      g_object_set_data_full(G_OBJECT(mi), "raster-name", g_strdup(name), g_free);
      g_signal_connect(G_OBJECT(mi), "activate", G_CALLBACK(_raster_menu_activate),
                       module);
      gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);
      count++;
    }
  }

  if(count == 0)
  {
    gtk_widget_destroy(menu);
    dt_control_log(_("no raster mask is available from an earlier module.\n"
                     "enable a mask on a module above this one first."));
    return TRUE;
  }

  gtk_widget_show_all(menu);
  dt_gui_menu_popup(GTK_MENU(menu), button, GDK_GRAVITY_SOUTH_WEST,
                    GDK_GRAVITY_NORTH_WEST);
  return TRUE;
}

// ---- shortcut actions on "whatever is currently selected in the panel" -----
// These have no fixed on-screen widget (unlike the add-shape/add-raster/
// add-parametric buttons, which are made shortcut-assignable directly via
// dt_action_define_iop above and in _rebuild_param_channel_buttons): they act
// on the module's current panel selection (bd->panel_selected_formid /
// panel_selected_group_cid), which changes as the user clicks around. Each one
// is a thin wrapper around the same helper the matching click handler already
// uses (see _toggle_element_hidden, _toggle_ids_hidden, _invert_element,
// _invert_group_members, _toggle_soloedit, _build_group_op_menu,
// _build_within_menu, _stage_new_group, _add_parametric_channel,
// _masks_raster_add_press), so a keyboard shortcut and the matching mouse
// click always do exactly the same thing.
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
  _stage_new_group(module, bd->masks_new_group_op, FALSE);
}

static void _shortcut_add_raster_mask(dt_action_t *action)
{
  dt_iop_module_t *module = dt_dev_gui_module();
  if(!module || !module->blend_data) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  if(!bd->masks_support || !bd->masks_inited) return;
  _masks_raster_add_press(NULL, NULL, module);
}

static void _shortcut_invert_selected_group(dt_action_t *action)
{
  dt_iop_module_t *module = dt_dev_gui_module();
  if(!module || !module->blend_data) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp || !dt_is_valid_maskid(bd->panel_selected_group_cid)) return;
  GList *members = _group_run_members(grp, bd->panel_selected_group_cid);
  _invert_group_members(module, members);
  g_list_free(members);
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
  if(!dt_is_valid_maskid(bd->panel_selected_formid)) return;
  _toggle_soloedit(module, bd->panel_selected_formid);
}

static void _shortcut_change_group_mode(dt_action_t *action)
{
  dt_iop_module_t *module = dt_dev_gui_module();
  if(!module || !module->blend_data) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp || !dt_is_valid_maskid(bd->panel_selected_group_cid)) return;
  GList *members = _group_run_members(grp, bd->panel_selected_group_cid);
  GtkWidget *menu = _build_group_between_op_menu(
    module, members, _group_is_base(grp, bd->panel_selected_group_cid));
  g_list_free(members);
  // no click event to anchor to -- pop up at the pointer, same fallback
  // dt_gui_menu_popup uses for any other keyboard-triggered menu
  dt_gui_menu_popup(GTK_MENU(menu), NULL, 0, 0);
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
  GList *members = _group_run_members(grp, bd->panel_selected_group_cid);
  _group_op_apply(module, members, DT_MASKS_STATE_OP_BYPASS);
  g_list_free(members);
}

static void _shortcut_change_group_within_mode(dt_action_t *action)
{
  dt_iop_module_t *module = dt_dev_gui_module();
  if(!module || !module->blend_data) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;
  dt_masks_form_t *grp = _module_mask_group(module);
  if(!grp || !dt_is_valid_maskid(bd->panel_selected_group_cid)) return;
  GList *members = _group_run_members(grp, bd->panel_selected_group_cid);
  GtkWidget *menu = _build_within_menu(module, members);
  g_list_free(members);
  dt_gui_menu_popup(GTK_MENU(menu), NULL, 0, 0);
}

// register every panel-selection shortcut above under module->so (the shared
// operation-type action tree, not the instance's own), so each shows up ONCE
// in shortcut preferences regardless of how many instances of this operation
// exist -- exactly like the widget actions dt_action_define_iop registers
// elsewhere in this file. Called once per instance init; repeated
// registration under the same "so" section is expected and harmless (see
// dt_action_section/dt_action_define).
static void _register_masks_action_shortcuts(dt_iop_module_t *module)
{
  dt_action_register(DT_ACTION(module->so), N_("add group above selected group"),
                     _shortcut_add_group_above_selected, 0, 0);
  dt_action_register(DT_ACTION(module->so), N_("add raster mask to current group"),
                     _shortcut_add_raster_mask, 0, 0);
  dt_action_register(DT_ACTION(module->so), N_("invert selected group visibility"),
                     _shortcut_invert_selected_group, 0, 0);
  dt_action_register(DT_ACTION(module->so), N_("invert selected element visibility"),
                     _shortcut_invert_selected_element, 0, 0);
  dt_action_register(DT_ACTION(module->so), N_("toggle solo-edit for current shape"),
                     _shortcut_toggle_soloedit, 0, 0);
  dt_action_register(DT_ACTION(module->so), N_("change mode for current group"),
                     _shortcut_change_group_mode, 0, 0);
  dt_action_register(DT_ACTION(module->so), N_("bypass/resume current group"),
                     _shortcut_toggle_group_bypass, 0, 0);
  dt_action_register(DT_ACTION(module->so),
                     N_("change within-group mode for current group"),
                     _shortcut_change_group_within_mode, 0, 0);
}

void dt_iop_gui_init_masks(GtkWidget *blendw, dt_iop_module_t *module)
{
  dt_iop_gui_blend_data_t *bd = module->blend_data;

  /* create and add masks support if module supports it */
  if(bd->masks_support)
  {
    bd->masks_combo_ids = NULL;
    bd->masks_shown = DT_MASKS_EDIT_OFF;

    bd->masks_combo = dt_bauhaus_combobox_new(module);
    dt_bauhaus_widget_set_label(bd->masks_combo, N_("blend"), N_("drawn mask"));
    // left-align the value ("N shapes used" in flexi) instead of the default right
    dt_bauhaus_combobox_set_selected_text_align(bd->masks_combo,
                                                DT_BAUHAUS_COMBOBOX_ALIGN_LEFT);
    // this is an action menu (each entry adds/uses a shape), not a value picker:
    // mute scroll so spinning the wheel over the open popup doesn't fire
    // value-changed per tick (which would add a shape per tick). The selection
    // is committed once, on click / popup close.
    dt_bauhaus_combobox_mute_scrolling(bd->masks_combo);

    dt_bauhaus_combobox_add(bd->masks_combo, _("no mask used"));
    g_signal_connect(G_OBJECT(bd->masks_combo), "value-changed",
                     G_CALLBACK(dt_masks_iop_value_changed_callback), module);
    dt_bauhaus_combobox_add_populate_fct(bd->masks_combo, dt_masks_iop_combo_populate);

    // flexi-only: in flexi, masks_combo is never actually shown (see
    // _masks_apply_layout) -- it stays alive purely as the headless data
    // source (entries/ids/value-changed) for this compact button, whose
    // click shows the same choices immediately as a plain popup menu, like
    // the add-group button, instead of asking for a second click to open a
    // combo the user never otherwise sees.
    bd->masks_import_btn = dtgtk_button_new(dtgtk_cairo_paint_import, 0, NULL);
    gtk_widget_set_tooltip_text(bd->masks_import_btn,
                                _("import an existing shape, or reuse another\n"
                                  "module's mask (click to pick one)"));
    g_signal_connect(G_OBJECT(bd->masks_import_btn), "button-press-event",
                     G_CALLBACK(_masks_import_btn_press), module);

    // ---- combo header row (classic two-row toolbar): the mask source/import combo
    // + the whole-mask "invert" toggle. In flexi this row is hidden by
    // _masks_apply_layout (masks_combo stays put, unused/invisible; invert
    // moves onto the "mask elements" header instead). Section-label styling =
    // text with a line below, matching every other section header.
    GtkWidget *hbox = dt_gui_hbox(dt_gui_expand(bd->masks_combo));
    dt_gui_add_class(hbox, "dt_section_label");
    bd->masks_combo_row = hbox;

    bd->masks_polarity =
      dt_iop_togglebutton_new(module, "blend`tools", N_("invert mask"), NULL,
                              G_CALLBACK(_blendop_masks_polarity_callback), FALSE, 0, 0,
                              dtgtk_cairo_paint_mask_invert, NULL);
    dtgtk_togglebutton_set_paint(DTGTK_TOGGLEBUTTON(bd->masks_polarity),
                                 dtgtk_cairo_paint_mask_invert, 0, NULL);
    dt_gui_add_class(bd->masks_polarity, "dt_ignore_fg_state");
    // dt_ignore_fg_state above suppresses the generic checked-button highlight
    // (see its own comment in darktable.css), so this button needs its own
    // explicit "on" state instead -- the same light-background/dark-foreground
    // swap a single inverted element/group's own icon gets (.mask-list-handle-
    // inverted, .mask-power-solo), so "invert mask" reads the same way those do
    // rather than giving no visual feedback at all when active.
    dt_gui_add_class(bd->masks_polarity, "mask-invert-toggle");
    // classic home: right end of the combo row. In flexi it moves onto the "mask
    // elements" header instead (see _masks_apply_layout).
    gtk_box_pack_end(GTK_BOX(hbox), bd->masks_polarity, FALSE, FALSE, 0);

    // ---- groups header (flexi-only): a section divider labelled "mask elements",
    // with "edit on canvas" right after the label, and "invert"/"reset" on the far
    // right (re-homed here by _masks_apply_layout / packed below). Section-label
    // styling (text above, line below), like the other headers. The label does NOT
    // expand, so "edit on canvas" sits immediately to its right; the line still
    // spans the full width (the border is on the hbox, not the label).
    GtkWidget *groups_label = dt_ui_label_new(_("mask elements"));
    gtk_widget_show(groups_label);
    // spacing before "edit on canvas" (packed here later by _masks_apply_layout)
    // so it doesn't sit flush against the label -- see .mask-elements-label
    dt_gui_add_class(groups_label, "mask-elements-label");
    GtkWidget *groups_hdr = dt_gui_hbox(groups_label);
    dt_gui_add_class(groups_hdr, "dt_section_label");
    gtk_widget_set_no_show_all(groups_hdr, TRUE);
    bd->masks_groups_header = groups_hdr;

    // default operator for a newly added group
    bd->masks_new_group_op = DT_MASKS_STATE_UNION;

    // ---- masks_toolbar: the fixed two-row flexi toolbar for every "add an
    // element" action (see its field comment in blend.h for the rationale
    // and exact row contents). Built first (empty) so the add-group button
    // below has somewhere to go; the rest of its permanent (flexi-only)
    // children are appended further down, as each is built.
    GtkWidget *toolbar = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_box_set_spacing(GTK_BOX(toolbar), DT_PIXEL_APPLY_DPI(3));
    gtk_widget_set_no_show_all(toolbar, TRUE);
    dt_gui_add_class(toolbar, "masks-toolbar");
    bd->masks_toolbar = toolbar;
    GtkWidget *toolbar_row1 = dt_gui_hbox();
    gtk_box_set_spacing(GTK_BOX(toolbar_row1), DT_PIXEL_APPLY_DPI(3));
    gtk_widget_show(toolbar_row1);
    bd->masks_toolbar_row1 = toolbar_row1;
    gtk_box_pack_start(GTK_BOX(toolbar), toolbar_row1, FALSE, FALSE, 0);
    GtkWidget *toolbar_row2 = dt_gui_hbox();
    gtk_box_set_spacing(GTK_BOX(toolbar_row2), DT_PIXEL_APPLY_DPI(3));
    gtk_widget_show(toolbar_row2);
    bd->masks_toolbar_row2 = toolbar_row2;
    gtk_box_pack_start(GTK_BOX(toolbar), toolbar_row2, FALSE, FALSE, 0);

    // "add group": a plain "+" that opens the operator chooser (its icon is a
    // fixed add affordance, it never reflects the selection). Row 1, leftmost.
    bd->masks_new_op_box = _make_op_combo(&bd->masks_new_op, dtgtk_cairo_paint_plus,
                                          G_CALLBACK(_new_shape_op_press));
    // the add-group button is a plain "+" icon, not a bordered chooser: drop the
    // "mask-op-combo" border so there is no white outline around it
    dt_gui_remove_class(bd->masks_new_op_box, "mask-op-combo");
    g_object_set_data(G_OBJECT(bd->masks_new_op), "module", module);
    _new_shape_op_update(bd->masks_new_op);
    gtk_widget_set_tooltip_text(bd->masks_new_op_box,
                                _("add a new group above the selected group\n"
                                  "(or above everything, if none is selected)\n"
                                  "ctrl+click to add it below instead\n"
                                  "click to pick its operator"));
    gtk_widget_show(bd->masks_new_op_box);
    gtk_box_pack_start(GTK_BOX(toolbar_row1), bd->masks_new_op_box, FALSE, FALSE, 0);
    bd->masks_new_op_label = NULL; // retired (the button is icon-only now)

    // clusters are separated by a single expanding stretch so the gap grows
    // with the panel instead of the buttons just sitting at the left edge.
    // this one ends up on shapes_box's *left* once
    // _masks_toolbar_place_shapes_box reorders shapes_box in between.
    _toolbar_pack_stretch(toolbar_row1);

    // reserves row 1's position right after shapes_box (which does not exist
    // as a toolbar child yet -- it starts out in masks_shapes_row, classic
    // default -- and only moves here once flexi mode is entered; see
    // _masks_toolbar_place_shapes_box). this stretch ends up on shapes_box's
    // *right*.
    _toolbar_pack_stretch(toolbar_row1);

    // "reset mask": clears every shape and restores the scaffold. Far right.
    bd->masks_reset_mask_btn = dtgtk_button_new(dtgtk_cairo_paint_reset, 0, NULL);
    gtk_widget_set_tooltip_text(bd->masks_reset_mask_btn,
                                _("reset the mask: remove every shape"));
    g_signal_connect(G_OBJECT(bd->masks_reset_mask_btn), "clicked",
                     G_CALLBACK(_masks_reset_mask), module);
    gtk_widget_set_no_show_all(bd->masks_reset_mask_btn, TRUE);

    // the right side of the groups header holds the whole-mask actions: "invert"
    // (re-homed here by _masks_apply_layout) and "reset". Mask layout presets
    // moved into the "blend mask" header's hamburger menu (see
    // _add_flexi_presets_menu), so there is no separate presets button here
    // anymore.
    gtk_box_pack_end(GTK_BOX(groups_hdr), bd->masks_reset_mask_btn, FALSE, FALSE, 0);

    // NB: each group's elements (shapes) are nested directly under that group's
    // header inside masks_list_box (built by _build_masks_list /
    // _pack_group_elements); there is no separate "elements" section.

    // ---- shapes box: the shape-add buttons, wrapped so the whole group is
    // re-homed as a unit between the classic shapes row and the flexi
    // toolbar (see _masks_toolbar_place_shapes_box).
    GtkWidget *shapes_box = dt_gui_hbox();
    bd->masks_shapes_box = shapes_box;

    // "edit on canvas": toggles the on-canvas editing overlay (the shape controls).
    // classic puts it leftmost on the shapes row; flexi moves it onto the "groups"
    // header. Created parentless and re-homed by _masks_apply_layout.
    bd->masks_edit = dt_iop_togglebutton_new(
      module, "blend`tools", N_("edit on canvas"),
      N_("edit on canvas in restricted mode (no moving or resizing of shapes)"),
      G_CALLBACK(_blendop_masks_show_and_edit), FALSE, 0, 0, dtgtk_cairo_paint_masks_eye,
      NULL);

    bd->masks_type[0] = DT_MASKS_PATH;
    bd->masks_shapes[0] = dt_iop_togglebutton_new(
      module, "blend`shapes", N_("add path"), N_("add multiple paths"),
      G_CALLBACK(_blendop_masks_add_shape), FALSE, 0, 0, dtgtk_cairo_paint_masks_path,
      NULL);
    gtk_widget_show(bd->masks_shapes[0]);
    gtk_box_pack_start(GTK_BOX(shapes_box), bd->masks_shapes[0], FALSE, FALSE, 0);
    _stash_base_tooltip(bd->masks_shapes[0]);

    bd->masks_type[1] = DT_MASKS_BRUSH;
    bd->masks_shapes[1] = dt_iop_togglebutton_new(
      module, "blend`shapes", N_("add brush"), N_("add multiple brush strokes"),
      G_CALLBACK(_blendop_masks_add_shape), FALSE, 0, 0, dtgtk_cairo_paint_masks_brush,
      NULL);
    gtk_widget_show(bd->masks_shapes[1]);
    gtk_box_pack_start(GTK_BOX(shapes_box), bd->masks_shapes[1], FALSE, FALSE, 0);
    _stash_base_tooltip(bd->masks_shapes[1]);

    bd->masks_type[2] = DT_MASKS_CIRCLE;
    bd->masks_shapes[2] = dt_iop_togglebutton_new(
      module, "blend`shapes", N_("add circle"), N_("add multiple circles"),
      G_CALLBACK(_blendop_masks_add_shape), FALSE, 0, 0, dtgtk_cairo_paint_masks_circle,
      NULL);
    gtk_widget_show(bd->masks_shapes[2]);
    gtk_box_pack_start(GTK_BOX(shapes_box), bd->masks_shapes[2], FALSE, FALSE, 0);
    _stash_base_tooltip(bd->masks_shapes[2]);

    bd->masks_type[3] = DT_MASKS_ELLIPSE;
    bd->masks_shapes[3] = dt_iop_togglebutton_new(
      module, "blend`shapes", N_("add ellipse"), N_("add multiple ellipses"),
      G_CALLBACK(_blendop_masks_add_shape), FALSE, 0, 0, dtgtk_cairo_paint_masks_ellipse,
      NULL);
    gtk_widget_show(bd->masks_shapes[3]);
    gtk_box_pack_start(GTK_BOX(shapes_box), bd->masks_shapes[3], FALSE, FALSE, 0);
    _stash_base_tooltip(bd->masks_shapes[3]);

    bd->masks_type[4] = DT_MASKS_GRADIENT;
    bd->masks_shapes[4] = dt_iop_togglebutton_new(
      module, "blend`shapes", N_("add gradient"), N_("add multiple gradients"),
      G_CALLBACK(_blendop_masks_add_shape), FALSE, 0, 0, dtgtk_cairo_paint_masks_gradient,
      NULL);
    gtk_widget_show(bd->masks_shapes[4]);
    gtk_box_pack_start(GTK_BOX(shapes_box), bd->masks_shapes[4], FALSE, FALSE, 0);
    _stash_base_tooltip(bd->masks_shapes[4]);

#ifdef HAVE_AI
    bd->masks_type[5] = DT_MASKS_OBJECT;
    bd->masks_shapes[5] =
      dt_iop_togglebutton_new(module, "blend`shapes", N_("add AI object"), NULL,
                              G_CALLBACK(_blendop_masks_add_shape), FALSE, 0, 0,
                              dtgtk_cairo_paint_masks_object, NULL);
    gtk_widget_show(bd->masks_shapes[5]);
    gtk_box_pack_start(GTK_BOX(shapes_box), bd->masks_shapes[5], FALSE, FALSE, 0);
    _stash_base_tooltip(bd->masks_shapes[5]);
#endif

    // parametric (blendif) forms are added via the channel row below (flexi-only),
    // one flat button per channel of the module's blend colorspace.
    bd->panel_selected_formid = INVALID_MASKID;
    bd->panel_selected_group_cid = INVALID_MASKID;
    bd->empty_groups = NULL;
    bd->selected_empty = NULL;
    bd->scaffold_seeded = FALSE;
    bd->masks_selection_seeded = FALSE;
    bd->insert_active = FALSE;
    bd->insert_realize_empty = FALSE;
    bd->insert_realized_fid = INVALID_MASKID;
    bd->solo_formid = INVALID_MASKID;

    // "add raster": an icon button (the same raster-mask glyph the rows use),
    // its own toolbar cluster -- a raster element brings in another module's
    // mask, same as import/reuse below, rather than drawing something new.
    // Row 1, rightmost.
    bd->masks_raster_add_btn = dtgtk_button_new(dtgtk_cairo_paint_masks_raster, 0, NULL);
    gtk_widget_set_tooltip_text(
      bd->masks_raster_add_btn,
      _("add a raster mask element: use another module's mask as an element\n"
        "of this group, combined with the group's operator"));
    g_signal_connect(G_OBJECT(bd->masks_raster_add_btn), "button-press-event",
                     G_CALLBACK(_masks_raster_add_press), module);
    gtk_widget_show(bd->masks_raster_add_btn);
    gtk_box_pack_start(GTK_BOX(toolbar_row1), bd->masks_raster_add_btn, FALSE, FALSE, 0);
    // makes the button assignable a shortcut like the shape-add buttons above
    // (those go through dt_iop_togglebutton_new, which does this internally --
    // this button is a plain dtgtk_button_new, so it needs the call explicitly)
    dt_action_define_iop(module, "blend`shapes", N_("add raster mask"),
                         bd->masks_raster_add_btn, &dt_action_def_button);

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
    gtk_box_pack_start(GTK_BOX(bd->masks_param_channels_box),
                       bd->masks_param_channels_inner, FALSE, FALSE, 0);
    gtk_widget_show(bd->masks_param_channels_inner);
    gtk_box_pack_start(GTK_BOX(toolbar_row2), bd->masks_param_channels_box, FALSE, FALSE,
                       0);

    _toolbar_pack_stretch(toolbar_row2);

    // "import/reuse shape": row 2, rightmost (see masks_import_btn's own
    // construction, earlier in this function, for the button itself).
    gtk_widget_show(bd->masks_import_btn);
    gtk_box_pack_start(GTK_BOX(toolbar_row2), bd->masks_import_btn, FALSE, FALSE, 0);

    // ---- shapes row (classic two-row toolbar): "show & edit elements" leftmost,
    // then the shapes box. The initial (classic) home; _masks_apply_layout re-homes
    // edit + shapes_box for flexi.
    GtkWidget *abox = dt_gui_hbox();
    bd->masks_shapes_row = abox;
    gtk_box_pack_start(GTK_BOX(abox), bd->masks_edit, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(abox), shapes_box, FALSE, FALSE, 0);

    // per-shape composition list (the groups), populated by _build_masks_list()
    // whenever the module is in flexi-mask mode.
    bd->masks_list_box = GTK_BOX(gtk_box_new(GTK_ORIENTATION_VERTICAL, 0));
    gtk_widget_set_no_show_all(GTK_WIDGET(bd->masks_list_box), TRUE);
    // unique id for the panel's own top-level list container, alongside the
    // existing "masks-list" class every nested list box in the panel shares
    gtk_widget_set_name(GTK_WIDGET(bd->masks_list_box), "masks-list-box");
    dt_gui_add_class(GTK_WIDGET(bd->masks_list_box), "masks-list"); // gap above the list

    // layout: "mask elements" header → toolbar → group list → classic combo
    // row → classic shapes row. The toolbar sits right under the header,
    // above the group list, so a freshly added element's row is right below
    // where it was added. _masks_apply_layout re-homes the shared widgets and
    // toggles row visibility per mode, so classic shows the master two-row
    // toolbar (combo row + shapes row) and flexi shows the header + toolbar +
    // group list.
    bd->masks_box = GTK_BOX(
      dt_gui_vbox(groups_hdr, toolbar, GTK_WIDGET(bd->masks_list_box), hbox, abox));
    _add_wrapped_box(blendw, bd->masks_box, "masks_drawn");

    bd->masks_inited = TRUE;
    _register_masks_action_shortcuts(module);
  }
}

void dt_iop_gui_cleanup_blending(dt_iop_module_t *module)
{
  if(!module->blend_data) return;
  dt_iop_gui_blend_data_t *bd = module->blend_data;

  // make sure this module's flexi content isn't left owning a shared host's
  // content box once its widgets (relocatable_box included) are destroyed.
  //
  // Unlike every other caller, this one can run *after* the widgets are gone:
  // when the panel is hosted elsewhere (the masks_flexi_host utility lib, or
  // the separate grid panel), relocatable_box is a child of that owner, not of
  // this module's iopw -- and at app quit that owner is torn down on its own
  // schedule, which may be first. _masks_flexi_release would then walk dangling
  // GtkWidget pointers (bd->* is never nulled when a widget dies), which is
  // what the burst of GTK_IS_WIDGET criticals on exit is. Reparenting a
  // destroyed box back into a destroyed iopw achieves nothing anyway; only the
  // host bookkeeping still matters, so do just that.
  if(darktable.develop->proxy.masks_flexi_host.hosted_module == module)
  {
    if(bd->relocatable_box && GTK_IS_WIDGET(bd->relocatable_box))
      _masks_flexi_release(module);
    else
      darktable.develop->proxy.masks_flexi_host.hosted_module = NULL;
  }

  dt_pthread_mutex_lock(&bd->lock);
  if(bd->timeout_handle)
    g_source_remove(bd->timeout_handle);
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
  free(bd->masks_combo_ids);
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
  if(bd->masks_blend_header)
  {
    if(is_mask_enabled)
      dt_gui_add_class(bd->masks_blend_header, "mask-enabled");
    else
      dt_gui_remove_class(bd->masks_blend_header, "mask-enabled");
  }

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
  // also hide the eye and showmask buttons for uniform blend
  gtk_widget_set_visible(bd->showmask, valid_masking);
  gtk_widget_set_visible(bd->suppress, valid_masking);

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
  const gboolean mode_flexi = mask_mode & DEVELOP_MASK_FLEXI;
  const gboolean mode_parametric = mask_mode & DEVELOP_MASK_CONDITIONAL;
  // flexi reuses the drawn-group toolbar/renderer (see _blendop_masks_mode_callback)
  const gboolean mode_drawn_or_flexi = mode_drawn || mode_flexi;

  _box_set_visible(bd->blend_box, mask_enabled);

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
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->suppress), FALSE);
      // (re)set the header mask indicator too
      if(module->mask_indicator)
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(module->mask_indicator), FALSE);
      gtk_widget_hide(GTK_WIDGET(bd->showmask));
      gtk_widget_hide(GTK_WIDGET(bd->suppress));
    }
    else
    {
      gtk_widget_show(GTK_WIDGET(bd->showmask));
      gtk_widget_show(GTK_WIDGET(bd->suppress));
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
    module->suppress_mask = FALSE;
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->suppress), FALSE);

    _box_set_visible(bd->refine_box, bd->raster_inited && mode_raster);
  }

  if(bd->masks_inited && mode_drawn_or_flexi)
  {
    // section caption reflects the mode: flexi drops the label (the combo value
    // "N shapes used" already says enough); classic keeps "drawn mask"
    dt_bauhaus_widget_set_label(bd->masks_combo, N_("blend"),
                                mode_flexi ? "" : N_("drawn mask"));
    // flexi-only widgets: new-shape operator selector, add-parametric button,
    // and the per-shape composition list (classic drawn mask stays vanilla)
    if(bd->masks_reset_mask_btn)
      gtk_widget_set_visible(bd->masks_reset_mask_btn, mode_flexi);
    if(bd->masks_param_channels_box)
      gtk_widget_set_visible(bd->masks_param_channels_box,
                             mode_flexi && bd->blendif_support);
    if(bd->masks_groups_header)
      gtk_widget_set_visible(bd->masks_groups_header, mode_flexi);
    _masks_apply_layout(bd, mode_flexi);
    gtk_widget_set_visible(GTK_WIDGET(bd->masks_list_box), mode_flexi);
    _box_set_visible(bd->masks_box, TRUE);
    // (re)build the per-shape composition list for this module's group
    if(mode_flexi) _build_masks_list(module);
  }
  else if(bd->masks_inited)
  {
    dt_masks_set_edit_mode(module, DT_MASKS_EDIT_OFF);

    // restore the classic homes so "invert" / "edit on canvas" never linger in the
    // "mask elements" header of a parametric/raster-only panel after leaving flexi
    _masks_apply_layout(bd, FALSE);
    _box_set_visible(bd->masks_box, FALSE);
  }
  else
  {
    _box_set_visible(bd->masks_box, FALSE);
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
  gtk_widget_set_visible(bd->masks_options_btn, !module->hide_enable_button);

  DT_LEAVE_GUI_UPDATE();
}

void dt_iop_gui_blending_gain_focus(dt_iop_module_t *module)
{
  if(!module || !module->blend_data) return;
  _masks_flexi_relocate(module);
}

void dt_iop_gui_blending_lose_focus(dt_iop_module_t *module)
{
  DT_GUARD_GUI_UPDATE();
  if(!module) return;

  const gboolean has_mask_display =
    module->request_mask_display
    & (DT_DEV_PIXELPIPE_DISPLAY_MASK | DT_DEV_PIXELPIPE_DISPLAY_CHANNEL);

  const gboolean suppress = module->suppress_mask;

  if((module->flags() & IOP_FLAGS_SUPPORTS_BLENDING) && module->blend_data)
  {
    dt_iop_gui_blend_data_t *bd = module->blend_data;

    // don't let the flexi masks panel content linger in a shared host once
    // its owning module loses focus
    if(darktable.develop->proxy.masks_flexi_host.hosted_module == module)
    {
      _masks_flexi_release(module);
    }

    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->showmask), FALSE);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->suppress), FALSE);
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

    dt_pthread_mutex_lock(&bd->lock);
    bd->save_for_leave = DT_DEV_PIXELPIPE_DISPLAY_NONE;
    if(bd->timeout_handle)
    {
      // purge any remaining timeout handlers
      g_source_remove(bd->timeout_handle);
      bd->timeout_handle = 0;
    }
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
    bd->timeout_handle = 0;
    bd->save_for_leave = 0;
    dt_pthread_mutex_unlock(&bd->lock);

    // collapse control for the separate flexi masks panel (left/right) --
    // only shown while hosted there (see _masks_flexi_relocate); a plain
    // flat arrow with its own CSS class, deliberately not styled like the
    // on/off toggle next to it
    bd->flexi_inline_collapse_btn =
      dtgtk_button_new(dtgtk_cairo_paint_solid_arrow, CPF_DIRECTION_LEFT, NULL);
    gtk_widget_set_name(bd->flexi_inline_collapse_btn, "flexi-inline-collapse");
    gtk_widget_set_tooltip_text(bd->flexi_inline_collapse_btn,
                                _("collapse this panel; click the icon it leaves behind\n"
                                  "on the canvas to bring it back"));
    g_signal_connect(G_OBJECT(bd->flexi_inline_collapse_btn), "clicked",
                     G_CALLBACK(_flexi_inline_collapse_clicked), NULL);
    gtk_widget_set_no_show_all(bd->flexi_inline_collapse_btn, TRUE);
    gtk_widget_set_visible(bd->flexi_inline_collapse_btn, FALSE);

    // on/off toggle for the whole blend mask (DEVELOP_MASK_DISABLED vs
    // DEVELOP_MASK_ENABLED|DEVELOP_MASK_FLEXI) -- flexi is the only mask
    // type left, so there is nothing left to pick a "type" from, just on/off
    // (see _blendop_mask_enable_toggled)
    bd->mask_enable_toggle =
      dt_iop_togglebutton_new(module, "blend`masks", N_("mask enabled"), NULL,
                              G_CALLBACK(_blendop_mask_enable_toggled), FALSE, 0, 0,
                              dtgtk_cairo_paint_masks_panel, NULL);
    _update_mask_enable_toggle_tooltip(bd->mask_enable_toggle, FALSE);
    // background always blends with the module's own background, on or off
    // -- only the glyph itself shows state
    dt_gui_add_class(bd->mask_enable_toggle, "dt_transparent_background");
    dt_gui_add_class(bd->mask_enable_toggle, "mask-enable-toggle");

    GtkWidget *caption_label = dt_ui_label_new(_("blend mask"));
    gtk_widget_set_margin_start(caption_label, DT_PIXEL_APPLY_DPI(4));

    // "blend mask" header: the panel-collapse arrow (hidden unless hosted in
    // a side panel) and the on/off toggle sit to the left of the caption;
    // the hamburger menu and the display/suppress eye icons (grouped into
    // right_cluster below) sit at the far right -- this replaces the old
    // classic mode-select row entirely, since flexi is the only mask type
    // left to switch on. left/right_cluster group their contents into a
    // single pack_end/pack_start unit apiece purely for ordering; the
    // header's own left/right inset (matching the module's content width)
    // comes from a real margin on gbox itself, in darktable.css's
    // "#blending-tabs" rule -- gbox is a sibling of the module's own
    // .dt_plugin_ui_main content box, not a descendant of it, so nothing
    // upstream already insets it.
    GtkWidget *left_cluster = bd->masks_left_cluster =
      dt_gui_hbox(bd->flexi_inline_collapse_btn, bd->mask_enable_toggle);

    GtkWidget *gbox = dt_gui_hbox(left_cluster, caption_label);
    dt_gui_add_class(gbox, "dt_section_label");
    dt_gui_add_help_link(gbox, "masks_blending");
    gtk_widget_set_name(gbox, "blending-tabs");
    // default to the embedded inset (see darktable.css's "#blending-tabs.
    // blending-tabs-embedded"); _masks_flexi_relocate toggles this off for
    // the two hosted positions, which already provide their own inset
    dt_gui_add_class(gbox, "blending-tabs-embedded");
    // flexi re-homes the whole-mask "invert" + "show & edit elements" toggles into
    // this header (see _masks_apply_layout)
    bd->masks_blend_header = gbox;

    // right-hand cluster: hamburger menu, then display/suppress eyes,
    // packed as a single pack_end unit so their relative order stays fixed
    // regardless of which of them is currently visible
    GtkWidget *right_cluster = bd->masks_right_cluster = dt_gui_hbox();
    gtk_box_pack_end(GTK_BOX(gbox), right_cluster, FALSE, FALSE, 0);

    GtkWidget *presets_button = bd->masks_options_btn =
      dtgtk_button_new_full(dtgtk_cairo_paint_presets, 0, NULL,
                            &(dtgtk_button_config_t){
                              .tooltip = _("blending options"),
                            });
    if(bd->blendif_support || bd->masks_support)
    {
      g_signal_connect(G_OBJECT(presets_button), "clicked",
                       G_CALLBACK(_blendif_options_callback), module);
    }
    else
    {
      gtk_widget_set_sensitive(GTK_WIDGET(presets_button), FALSE);
    }
    // pack_end, and before showmask/suppress below, so it claims the true
    // rightmost slot within the cluster (see "A should be on the very right")
    gtk_box_pack_end(GTK_BOX(right_cluster), presets_button, FALSE, FALSE, 0);

    bd->showmask = dt_iop_togglebutton_new(
      module, "blend`tools", N_("display mask and/or color channel"), NULL,
      G_CALLBACK(_blendop_blendif_showmask_clicked), FALSE, 0, 0,
      dtgtk_cairo_paint_showmask, right_cluster);
    gtk_widget_set_tooltip_text
      (bd->showmask,
       _("display mask and/or color channel.\n"
         "ctrl+click to display mask,\n"
         "shift+click to display channel.\n"
         "hover over parametric mask slider to select channel for display"));
    dt_gui_add_class(bd->showmask, "dt_transparent_background");

    bd->suppress = dt_iop_togglebutton_new(
      module, "blend`tools", N_("temporarily switch off blend mask"), NULL,
      G_CALLBACK(_blendop_blendif_suppress_toggled), FALSE, 0, 0,
      dtgtk_cairo_paint_eye_toggle, right_cluster);
    gtk_widget_set_tooltip_text
      (bd->suppress,
       _("temporarily switch off blend mask.\n"
         "only for module in focus"));
    dt_gui_add_class(bd->suppress, "dt_transparent_background");

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
    module->fusion_slider = bd->opacity_slider;

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
    GtkWidget *destdisp_head = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, DT_BAUHAUS_SPACE);
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

    gtk_box_pack_start(GTK_BOX(destdisp_head), icon_evb, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(destdisp_head), header_evb, TRUE, TRUE, 0);
    gtk_box_pack_end(GTK_BOX(destdisp_head), bd->masks_refine_toggle_btn, FALSE, FALSE,
                     0);

    // Inside the expanded section:
    // Top row showing: <icon> <label> <actions>
    GtkWidget *inner_header_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
    dt_gui_add_class(inner_header_row, "mask-refine-inner-header");

    bd->masks_refine_icon_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_widget_set_valign(bd->masks_refine_icon_box, GTK_ALIGN_CENTER);

    bd->masks_refine_name_label = gtk_label_new(_("whole mask"));
    gtk_label_set_xalign(GTK_LABEL(bd->masks_refine_name_label), 0.0f);
    gtk_label_set_ellipsize(GTK_LABEL(bd->masks_refine_name_label), PANGO_ELLIPSIZE_END);
    gtk_widget_set_hexpand(bd->masks_refine_name_label, TRUE);
    dt_gui_add_class(bd->masks_refine_name_label, "mask-refine-header-name");

    gtk_box_pack_start(GTK_BOX(inner_header_row), bd->masks_refine_icon_box, FALSE, FALSE,
                       0);
    gtk_box_pack_start(GTK_BOX(inner_header_row), bd->masks_refine_name_label, TRUE, TRUE,
                       0);

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
    GtkWidget *mask_panel = GTK_WIDGET(bd->relocatable_box);

    GtkWidget *box = dt_gui_vbox();
    bd->blend_box = GTK_BOX(dt_gui_vbox(
      dt_gui_hbox(dt_gui_expand(bd->blend_modes_combo), bd->blend_modes_blend_order),
      bd->blend_mode_parameter_slider, bd->opacity_slider));
    _add_wrapped_box(box, bd->blend_box, NULL);

    dt_gui_box_add(mask_panel, box);
    dt_iop_gui_init_masks(mask_panel, module);

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
