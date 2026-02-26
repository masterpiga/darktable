/*
    This file is part of darktable,
    Copyright (C) 2024 darktable developers.

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

#include "common.h"
#include "colorspace.h"

// Wrap hue to [0, 1)
static inline float wrap_hue(float h)
{
  h = fmod(h, 1.0f);
  if(h < 0.0f) h += 1.0f;
  return h;
}

// Pull the pixel hue toward the nearest harmony node, scaled by Gaussian proximity.
// Wide zone → high weight for all pixels → broad effect.
// Narrow zone → low weight far from nodes → selective effect.
static inline float get_weighted_hue_shift(const float px_hue,
                                           constant const float *const nodes,
                                           const int num_nodes,
                                           const float zone_width_factor)
{
  const float sigma = zone_width_factor * 0.5f / (float)num_nodes;
  const float inv_2sigma2 = 1.0f / (2.0f * sigma * sigma);

  float max_weight = 0.0f;
  float best_diff  = 0.0f;

  for(int i = 0; i < num_nodes; i++)
  {
    float d = fabs(px_hue - nodes[i]);
    if(d > 0.5f) d = 1.0f - d;

    const float w = exp(-d * d * inv_2sigma2);

    if(w > max_weight)
    {
      max_weight = w;
      float diff = nodes[i] - px_hue;
      if(diff > 0.5f)       diff -= 1.0f;
      else if(diff < -0.5f) diff += 1.0f;
      best_diff = diff;
    }
  }

  return max_weight * best_diff;
}

// JzCzhz -> JzAzBz (inverse of JzAzBz_to_JzCzhz in colorspace.h)
static inline float4 JzCzhz_to_JzAzBz(const float4 JzCzhz)
{
  const float angle = JzCzhz.z * 2.0f * M_PI_F;
  float4 JzAzBz;
  JzAzBz.x = JzCzhz.x;
  JzAzBz.y = JzCzhz.y * cos(angle);
  JzAzBz.z = JzCzhz.y * sin(angle);
  JzAzBz.w = JzCzhz.w;
  return JzAzBz;
}

kernel void colorharmonizer(read_only image2d_t in,
                            write_only image2d_t out,
                            const int width,
                            const int height,
                            constant const float *const matrix_in,
                            constant const float *const matrix_out,
                            constant const float *const nodes,
                            const int num_nodes,
                            const float zone_width,
                            const float effect_strength,
                            const float protect_neutral)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const float4 pix_in = fmax(0.0f, read_imagef(in, sampleri, (int2)(x, y)));

  // 1. Pipeline RGB -> XYZ D65 (premultiplied matrix: D50_to_D65 @ RGB_to_XYZ_D50)
  float4 XYZ_D65 = matrix_product_float4(pix_in, matrix_in);

  // 2. XYZ D65 -> JzAzBz -> JzCzhz
  float4 JzAzBz = XYZ_to_JzAzBz(XYZ_D65);
  float4 JzCzhz = JzAzBz_to_JzCzhz(JzAzBz);

  const float hue    = JzCzhz.z;
  const float chroma = JzCzhz.y;

  // 3. Protect neutrals
  const float t = protect_neutral;
  const float cutoff = t * t * t * 0.3f;
  const float chroma_weight = chroma / (chroma + cutoff + 1e-5f);

  // 4. Weighted hue shift toward harmony nodes
  const float hue_shift  = get_weighted_hue_shift(hue, nodes, num_nodes, zone_width);
  const float pull_amount = effect_strength * chroma_weight;
  JzCzhz.z = wrap_hue(hue + hue_shift * pull_amount);

  // 5. JzCzhz -> JzAzBz -> XYZ D65
  JzAzBz = JzCzhz_to_JzAzBz(JzCzhz);
  XYZ_D65 = JzAzBz_2_XYZ(JzAzBz);

  // 6. XYZ D65 -> Pipeline RGB (premultiplied matrix: XYZ_D50_to_RGB @ D65_to_D50)
  float4 pix_out = matrix_product_float4(XYZ_D65, matrix_out);
  pix_out.w = pix_in.w;

  write_imagef(out, (int2)(x, y), pix_out);
}
