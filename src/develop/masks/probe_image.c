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

#include "develop/masks/probe_image.h"

#include "common/darktable.h"
#include "common/math.h"

#include <math.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// deterministic integer hash noise
//
// Everything random-looking in the probe comes from here. It has to be an
// integer hash rather than rand(): the probe must come out bit-identical on
// every machine, or a harvested file collected by one user cannot be replayed
// by another, and a verification failure could never be reproduced.
// ---------------------------------------------------------------------------

static inline uint32_t _hash_u32(uint32_t x)
{
  // finaliser from MurmurHash3; good avalanche, no state, no libc
  x ^= x >> 16;
  x *= 0x85ebca6bu;
  x ^= x >> 13;
  x *= 0xc2b2ae35u;
  x ^= x >> 16;
  return x;
}

static inline uint32_t _hash3(const int32_t x, const int32_t y, const uint32_t seed)
{
  return _hash_u32(((uint32_t)x * 0x9e3779b1u) ^ ((uint32_t)y * 0x85ebca77u) ^ (seed * 0xc2b2ae3du));
}

/** hashed value in [0,1) at integer lattice point (x,y) */
static inline float _value_at(const int32_t x, const int32_t y, const uint32_t seed)
{
  return (float)(_hash3(x, y, seed) >> 8) * (1.0f / 16777216.0f);
}

static inline float _smoothstep(const float t)
{
  return t * t * (3.0f - 2.0f * t);
}

/** bilinear value noise with smoothstep interpolation, lattice period `scale` */
static inline float _value_noise(const float x, const float y, const uint32_t seed)
{
  const float fx = floorf(x), fy = floorf(y);
  const int32_t ix = (int32_t)fx, iy = (int32_t)fy;
  const float tx = _smoothstep(x - fx), ty = _smoothstep(y - fy);

  const float v00 = _value_at(ix, iy, seed);
  const float v10 = _value_at(ix + 1, iy, seed);
  const float v01 = _value_at(ix, iy + 1, seed);
  const float v11 = _value_at(ix + 1, iy + 1, seed);

  const float a = v00 + (v10 - v00) * tx;
  const float b = v01 + (v11 - v01) * tx;
  return a + (b - a) * ty;
}

/** fractional Brownian motion: `octaves` doublings of frequency, halvings of
    amplitude. Returns roughly [0,1], mean ~0.5. This is what gives the detail
    mask (which is a wavelet decomposition) something to find at every scale it
    looks at -- a single-frequency noise would light up one wavelet band and
    leave the rest flat. */
static inline float _fbm(const float x,
                         const float y,
                         const int octaves,
                         const uint32_t seed)
{
  float sum = 0.0f, amp = 0.5f, norm = 0.0f;
  float fx = x, fy = y;
  for(int o = 0; o < octaves; o++)
  {
    sum += amp * _value_noise(fx, fy, seed + (uint32_t)o * 0x1000193u);
    norm += amp;
    amp *= 0.5f;
    fx *= 2.0f;
    fy *= 2.0f;
  }
  return norm > 0.0f ? sum / norm : 0.0f;
}

// ---------------------------------------------------------------------------
// low-discrepancy sequences
//
// Used to walk the third colour axis (blue) and the exposure ladder across
// tiles. A low-discrepancy sequence rather than a hash because we want the
// *first few* tiles a small mask covers to already be well spread, which is
// exactly the property these have and a hash does not.
// ---------------------------------------------------------------------------

/** van der Corput radical inverse in base 2 */
static inline float _radical_inverse_2(uint32_t n)
{
  n = (n << 16) | (n >> 16);
  n = ((n & 0x55555555u) << 1) | ((n & 0xaaaaaaaau) >> 1);
  n = ((n & 0x33333333u) << 2) | ((n & 0xccccccccu) >> 2);
  n = ((n & 0x0f0f0f0fu) << 4) | ((n & 0xf0f0f0f0u) >> 4);
  n = ((n & 0x00ff00ffu) << 8) | ((n & 0xff00ff00u) >> 8);
  return (float)n * 2.3283064365386963e-10f; // / 2^32
}

/** van der Corput radical inverse in base 3 */
static inline float _radical_inverse_3(uint32_t n)
{
  float inv = 1.0f / 3.0f, r = 0.0f, f = inv;
  while(n)
  {
    r += (float)(n % 3u) * f;
    n /= 3u;
    f *= inv;
  }
  return r;
}

// ---------------------------------------------------------------------------
// the probe
// ---------------------------------------------------------------------------

/** Tile edge in pixels.

    This is the probe's main tuning knob, and it trades off the two coverage
    properties against each other. Each tile sweeps red and green internally,
    so a larger tile resolves that sweep more finely; but blue and exposure
    only change *between* tiles, so a larger tile means a small drawn mask
    sees fewer distinct slices of the colour cube.

    16px is chosen from the measured failure of the second property: at 48px a
    window one eighth of the image across spans barely two tiles, and the
    local-coverage test measured Jz reaching only 6 of 32 bins there. At 16px
    the same window spans about seven tiles each way, and the per-pixel noise
    is more than enough to fill in a 16-step sweep. Clamped for small images so
    the tiling never degenerates to a single tile. */
static inline int _tile_size(const int width, const int height)
{
  const int smaller = MIN(width, height);
  int t = 16;
  if(smaller < 8 * t) t = MAX(4, smaller / 8);
  return t;
}

void dt_masks_probe_generate(float *const buf,
                             const int width,
                             const int height)
{
  if(!buf || width <= 0 || height <= 0) return;

  const int tile = _tile_size(width, height);
  const int ntx = (width + tile - 1) / tile;

  // noise lattice density: fixed in *image* fractions rather than pixels, so
  // that the same structure appears at every probe size and a mask harvested
  // against one size behaves comparably at another.
  const float nscale = 24.0f;

  DT_OMP_FOR()
  for(int y = 0; y < height; y++)
  {
    const int ty = y / tile;
    const float v = (float)(y % tile) / (float)tile; // [0,1) within tile

    for(int x = 0; x < width; x++)
    {
      const int tx = x / tile;
      const float u = (float)(x % tile) / (float)tile;

      const uint32_t n = (uint32_t)(ty * ntx + tx);

      // Base layer: each tile sweeps a full (R,G) slice of the linear-RGB
      // cube, and the slice's blue level walks a base-2 radical inverse
      // across tiles. So one tile alone already spans all of red and all of
      // green, and a handful of neighbouring tiles span blue too.
      float rgb[3] = { u, v, _radical_inverse_2(n + 1u) };

      // Texture, at every scale the detail mask can look at. Different seed
      // per channel, so the noise moves through colour space rather than only
      // along the neutral axis -- otherwise it would add luminance coverage
      // but no hue coverage.
      const float nx = (float)x / (float)width * nscale;
      const float ny = (float)y / (float)height * nscale;
      for(int c = 0; c < 3; c++)
        rgb[c] += 0.18f * (_fbm(nx, ny, 5, 0x51ed270bu + (uint32_t)c * 0x9e3779b9u) - 0.5f);

      // Hard edges, in two families.
      //
      // The tile grid itself already provides a regular one: R and G both
      // jump from 1 back to 0 at every tile boundary. That is a strong,
      // perfectly sharp edge lattice -- but it is a single scale, perfectly
      // periodic, and strictly axis-aligned, which is about as unlike
      // photographic structure as an edge can be. Guided filtering is
      // sensitive to edge orientation and spacing, so a probe carrying only
      // that lattice would exercise one degenerate case and call it edge
      // coverage.
      //
      // The second family is therefore deliberately unlike the first in all
      // three respects: the cell sizes are chosen coprime to the tile size so
      // the cells never line up with tile boundaries, each cell is offset from
      // the origin, and each is split by a half-plane at a hashed angle rather
      // than along an axis. That yields sharp steps at arbitrary orientations
      // and arbitrary positions.
      //
      // The alignment point is not hypothetical: the first version of this
      // used cell = tile * (2 << level), which made every cell boundary land
      // exactly on a tile boundary, so the whole family added no edge the
      // lattice did not already have.
      // Four levels rather than two, because the balance matters and was
      // measured wrong at two: the tile lattice puts an axis-aligned step
      // every `tile` pixels in both directions, which is a far denser
      // population than a handful of cell boundaries can offset. The
      // orientation test found the diagonal bins holding a sixth of the share
      // they needed. Four families of finer cells bring off-axis edges up to
      // the same order as the lattice's.
      //
      // The per-family factors are correspondingly gentler, since up to four
      // of them now multiply on the same pixel.
      static const int _cell_bias[4] = { 5, 7, 11, 13 };
      for(int level = 0; level < 4; level++)
      {
        const int cell = tile * (level + 1) + _cell_bias[level]; // coprime to tile
        const int cx = (x + 13 * level) / cell;
        const int cy = (y + 29 * level) / cell;
        const uint32_t h = _hash3(cx, cy, 0xa511e9b3u + (uint32_t)level);

        // half-plane through the cell centre, at a hashed angle
        const float angle = (float)(h >> 8) * (DT_2PI_F / 16777216.0f);
        const float dx = (float)x - ((float)cx * cell + cell * 0.5f);
        const float dy = (float)y - ((float)cy * cell + cell * 0.5f);
        if(dx * cosf(angle) + dy * sinf(angle) > 0.0f)
        {
          const float k = (h & 2u) ? 0.6f : 1.5f;
          for(int c = 0; c < 3; c++) rgb[c] *= k;
        }
      }

      // Saturation ladder.
      //
      // The base sweep is a rectangular walk of the RGB cube, so it reaches
      // the cube's corners only at exact tile corners, and the additive noise
      // above pulls even those back towards neutral. The most saturated
      // colours the colour space can express are therefore never produced,
      // and the coverage test measured exactly that: the top bins of Cz were
      // unreachable. This pushes a quarter of the tiles away from their own
      // mean, towards (and past) the gamut boundary; the clamp to zero at the
      // end of the loop is what makes an out-of-gamut push land on the
      // boundary rather than outside it.
      const float s2 = _radical_inverse_2(n * 3u + 7u); // decorrelated from blue
      if(s2 > 0.75f)
      {
        const float mean = (rgb[0] + rgb[1] + rgb[2]) / 3.0f;
        const float k = 1.0f + (s2 - 0.75f) * (3.0f / 0.25f); // up to 4x
        for(int c = 0; c < 3; c++) rgb[c] = mean + k * (rgb[c] - mean);
      }

      // Exposure ladder, spanning [-6, +2] EV over half the tiles.
      //
      // Upwards because the working space is scene-referred: boost factors let
      // a blendif slider address values well above 1.0, and a probe clipped
      // there would make every such selection vacuous.
      //
      // Downwards because the base layer cannot get dark on its own. Red and
      // green sweep independently and blue walks its own sequence, so a truly
      // dark pixel needs all three near zero at once -- which happens only at
      // a tile corner, and even there the additive noise lifts it back up. The
      // deep shadows were measured missing: the coverage test found the bottom
      // bin of LAB L unreachable until this ladder was extended below zero EV.
      const float e3 = _radical_inverse_3(n + 1u);
      if(e3 > 0.5f)
      {
        const float ev = (e3 - 0.5f) * (8.0f / 0.5f) - 6.0f; // [-6, +2] EV
        const float gain = exp2f(ev);
        for(int c = 0; c < 3; c++) rgb[c] *= gain;
      }

      float *const px = buf + ((size_t)y * width + x) * 4;
      for(int c = 0; c < 3; c++) px[c] = MAX(0.0f, rgb[c]);
      px[3] = 0.0f;
    }
  }
}

float *dt_masks_probe_new(const int width, const int height)
{
  if(width <= 0 || height <= 0) return NULL;
  float *const buf = dt_alloc_align_float((size_t)width * height * 4);
  if(!buf) return NULL;
  dt_masks_probe_generate(buf, width, height);
  return buf;
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
