// What does a given Cz mean in visible terms? Anchor the proposed hue gate
// against the smallest chroma an 8-bit image can even represent.
#include <stdio.h>
#include <math.h>
#define DT_2PI_F 6.28318530717958647693f
static void XYZ_2_JzAzBz(const float X[4], float J[4]);
static void JzAzBz_2_JzCzhz(const float J[4], float o[4])
{ float h = atan2f(J[2], J[1]) / DT_2PI_F; o[0]=J[0]; o[1]=hypotf(J[1],J[2]); o[2]= h>=0?h:1+h; }
#include "hostjz.inc"
// linear Rec2020 -> XYZ D65
static void rec2020_to_xyz(const float rgb[3], float xyz[4])
{
  const float m[3][3] = {{0.6369580f,0.1446169f,0.1688810f},
                         {0.2627002f,0.6779981f,0.0593017f},
                         {0.0000000f,0.0280727f,1.0609851f}};
  for(int i=0;i<3;i++) xyz[i]=m[i][0]*rgb[0]+m[i][1]*rgb[1]+m[i][2]*rgb[2];
  xyz[3]=0;
}
static float cz_of(float r, float g, float b)
{
  float rgb[3]={r,g,b}, xyz[4], J[4]={0,0,0,0}, C[4]={0,0,0,0};
  rec2020_to_xyz(rgb,xyz); XYZ_2_JzAzBz(xyz,J); JzAzBz_2_JzCzhz(J,C); return C[1];
}
int main(void)
{
  printf("Cz for a neutral patch nudged off-axis by one 8-bit step (sRGB-ish encoding):\n");
  const float lin[] = { 0.0025f, 0.018f, 0.18f, 0.5f, 1.0f };
  const char *nm[] = { "deep shadow", "shadow", "mid grey 18%", "light", "white" };
  for(int i=0;i<5;i++)
  {
    const float v = lin[i];
    // one 8-bit code value at this luminance, converted to a linear delta
    const float enc = powf(v, 1.0f/2.2f);
    const float step = powf(enc + 1.0f/255.0f, 2.2f) - v;
    printf("  %-13s Y=%-8.4g  neutral Cz=%-10.3g  +1 code on R: Cz=%.4g\n",
           nm[i], v, cz_of(v,v,v), cz_of(v+step,v,v));
  }
  printf("\nCz for visibly tinted patches at mid grey:\n");
  for(float t = 0.01f; t <= 0.2001f; t *= 2.0f)
    printf("  R +%4.1f%%  Cz=%.4g\n", t*100.0f, cz_of(0.18f*(1+t),0.18f,0.18f));
  return 0;
}
