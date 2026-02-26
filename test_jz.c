#include <stdio.h>
#include <math.h>

typedef float dt_aligned_pixel_t[4] __attribute__((aligned(16)));

// define macros so colorspaces inline conversions compiles
#define DT_OMP_DECLARE_SIMD(...) 
#define for_each_channel(c) for(int c=0; c<3; c++)
#define CLIP(x) ((x) < 0.0f ? 0.0f : ((x) > 1.0f ? 1.0f : (x)))

#include "src/common/colorspaces.h"
#include "src/common/colorspaces_inline_conversions.h"

int main() {
  float Jz = 0.1f;
  float Cz = 0.06f;
  for (int h=0; h<6; h++) {
    float hue = h * 60.0f * M_PI / 180.0f;
    dt_aligned_pixel_t jzczhz = {Jz, Cz, hue, 0};
    dt_aligned_pixel_t jzazbz, xyz, rgb;
    dt_JzCzhz_2_JzAzBz(jzczhz, jzazbz);
    dt_JzAzBz_2_XYZ(jzazbz, xyz);
    dt_XYZ_to_sRGB(xyz, rgb);
    printf("Hue %d: RGB = %.3f, %.3f, %.3f\n", h*60, rgb[0], rgb[1], rgb[2]);
  }
}
