// Differential test: darktable's host JzCzhz conversion vs its OpenCL kernel
// counterpart, on identical inputs. Both sides are verbatim copies of the
// shipped code (src/common/colorspaces_inline_conversions.h and
// data/kernels/colorspace.h). No darktable build involved.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <OpenCL/opencl.h>

#define DT_2PI_F 6.28318530717958647693f
#define N 400000

// ---------------- host side (colorspaces_inline_conversions.h) -------------
static void host_XYZ_2_JzAzBz(const float XYZ_D65[4], float JzAzBz[4])
{
  const float b = 1.15f, g = 0.66f;
  const float c1 = 0.8359375f, c2 = 18.8515625f, c3 = 18.6875f;
  const float n = 0.159301758f, p = 134.034375f;
  const float d = -0.56f, d0 = 1.6295499532821566e-11f;
  static const float M_t[3][4] = {
      { 0.41478972f, -0.2015100f, -0.0166008f, 0.0f },
      { 0.57999900f,  1.1206490f,  0.2648000f, 0.0f },
      { 0.01464800f,  0.0531008f,  0.6684799f, 0.0f },
  };
  static const float A_t[3][4] = {
      { 0.5f,       3.524000f,  0.199076f, 0.0f },
      { 0.5f,      -4.066708f,  1.096799f, 0.0f },
      { 0.0f,       0.542708f, -1.295875f, 0.0f },
  };
  float XYZ[4] = { 0, 0, 0, 0 };
  XYZ[0] = b * XYZ_D65[0] - (b - 1.0f) * XYZ_D65[2];
  XYZ[1] = g * XYZ_D65[1] - (g - 1.0f) * XYZ_D65[0];
  XYZ[2] = XYZ_D65[2];

  float LMS[4] = { 0, 0, 0, 0 };
  for(int i = 0; i < 3; i++)
    LMS[i] = M_t[0][i] * XYZ[0] + M_t[1][i] * XYZ[1] + M_t[2][i] * XYZ[2];
  for(int i = 0; i < 3; i++)
  {
    LMS[i] = powf(fmaxf(LMS[i] / 10000.f, 0.0f), n);
    LMS[i] = powf((c1 + c2 * LMS[i]) / (1.0f + c3 * LMS[i]), p);
  }
  for(int i = 0; i < 3; i++)
    JzAzBz[i] = A_t[0][i] * LMS[0] + A_t[1][i] * LMS[1] + A_t[2][i] * LMS[2];
  JzAzBz[0] = fmaxf(((1.0f + d) * JzAzBz[0]) / (1.0f + d * JzAzBz[0]) - d0, 0.f);
}

static void host_JzAzBz_2_JzCzhz(const float J[4], float out[4])
{
  float var_H = atan2f(J[2], J[1]) / DT_2PI_F;
  out[0] = J[0];
  out[1] = hypotf(J[1], J[2]);
  out[2] = var_H >= 0.0f ? var_H : 1.0f + var_H;
}

// ---------------- device side (data/kernels/colorspace.h) ------------------
static const char *ksrc =
"#define DT_2PI_F 6.28318530717958647693f\n"
"static inline float dt_fast_hypot(const float x, const float y){ return hypot(x,y); }\n"
"static inline float4 XYZ_to_JzAzBz(float4 XYZ_D65)\n"
"{\n"
"  const float4 M[3] = { { 0.41478972f, 0.579999f, 0.0146480f, 0.0f },\n"
"                        { -0.2015100f, 1.120649f, 0.0531008f, 0.0f },\n"
"                        { -0.0166008f, 0.264800f, 0.6684799f, 0.0f } };\n"
"  const float4 A[3] = { { 0.5f, 0.5f, 0.0f, 0.0f },\n"
"                        { 3.524000f, -4.066708f, 0.542708f, 0.0f },\n"
"                        { 0.199076f, 1.096799f, -1.295875f, 0.0f } };\n"
"  float4 temp1, temp2;\n"
"  temp1.x = 1.15f * XYZ_D65.x - 0.15f * XYZ_D65.z;\n"
"  temp1.y = 0.66f * XYZ_D65.y + 0.34f * XYZ_D65.x;\n"
"  temp1.z = XYZ_D65.z;\n"
"  temp1.w = 0.f;\n"
"  temp2.x = dot(M[0], temp1);\n"
"  temp2.y = dot(M[1], temp1);\n"
"  temp2.z = dot(M[2], temp1);\n"
"  temp2.w = 0.f;\n"
"  temp2 = pow(fmax(temp2 / 10000.f, 0.0f), 0.159301758f);\n"
"  temp2 = pow((0.8359375f + 18.8515625f * temp2) / (1.0f + 18.6875f * temp2), 134.034375f);\n"
"  temp1.x = dot(A[0], temp2);\n"
"  temp1.y = dot(A[1], temp2);\n"
"  temp1.z = dot(A[2], temp2);\n"
"  temp1.x = fmax(0.44f * temp1.x / (1.0f - 0.56f * temp1.x) - 1.6295499532821566e-11f, 0.f);\n"
"  return temp1;\n"
"}\n"
"static inline float4 JzAzBz_to_JzCzhz(float4 JzAzBz)\n"
"{\n"
"  const float h = atan2(JzAzBz.z, JzAzBz.y) / DT_2PI_F;\n"
"  float4 JzCzhz;\n"
"  JzCzhz.x = JzAzBz.x;\n"
"  JzCzhz.y = dt_fast_hypot(JzAzBz.y, JzAzBz.z);\n"
"  JzCzhz.z = (h >= 0.0f) ? h : 1.0f + h;\n"
"  JzCzhz.w = JzAzBz.w;\n"
"  return JzCzhz;\n"
"}\n"
"__kernel void run(__global const float4 *in, __global float4 *out)\n"
"{\n"
"  int i = get_global_id(0);\n"
"  out[i] = JzAzBz_to_JzCzhz(XYZ_to_JzAzBz(in[i]));\n"
"}\n";

int main(void)
{
  float *in = malloc(sizeof(float) * 4 * N);
  float *gpu = malloc(sizeof(float) * 4 * N);
  float *cpu = malloc(sizeof(float) * 4 * N);

  // XYZ D65 values spanning the range a scene-referred pipe produces, with a
  // deliberate concentration near the achromatic axis where hue is ill-conditioned
  srandom(1234);
  for(int i = 0; i < N; i++)
  {
    const float Y = powf((float)(random() % 100000) / 100000.0f, 3.0f) * 4.0f;
    const float spread = (i % 4 == 0) ? 1e-4f : 0.35f; // a quarter near-neutral
    const float dx = ((float)(random() % 200000) / 100000.0f - 1.0f) * spread;
    const float dz = ((float)(random() % 200000) / 100000.0f - 1.0f) * spread;
    in[4 * i + 0] = Y * (0.95047f + dx);
    in[4 * i + 1] = Y;
    in[4 * i + 2] = Y * (1.08883f + dz);
    in[4 * i + 3] = 0.0f;
  }
  for(int i = 0; i < N; i++)
  {
    float J[4] = { 0, 0, 0, 0 };
    host_XYZ_2_JzAzBz(in + 4 * i, J);
    host_JzAzBz_2_JzCzhz(J, cpu + 4 * i);
  }

  cl_platform_id plat;
  cl_device_id dev;
  clGetPlatformIDs(1, &plat, NULL);
  clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
  char name[256];
  clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
  cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, NULL);
  cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, NULL);
  cl_program prog = clCreateProgramWithSource(ctx, 1, &ksrc, NULL, NULL);
  cl_int e = clBuildProgram(prog, 1, &dev, "", NULL, NULL);
  if(e != CL_SUCCESS)
  {
    char log[16384];
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
    fprintf(stderr, "build failed:\n%s\n", log);
    return 1;
  }
  cl_kernel k = clCreateKernel(prog, "run", NULL);
  cl_mem bi = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             sizeof(float) * 4 * N, in, NULL);
  cl_mem bo = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * 4 * N, NULL, NULL);
  clSetKernelArg(k, 0, sizeof(cl_mem), &bi);
  clSetKernelArg(k, 1, sizeof(cl_mem), &bo);
  size_t gs = N;
  clEnqueueNDRangeKernel(q, k, 1, NULL, &gs, NULL, 0, NULL, NULL);
  clEnqueueReadBuffer(q, bo, CL_TRUE, 0, sizeof(float) * 4 * N, gpu, 0, NULL, NULL);

  printf("device: %s   samples: %d\n", name, N);
  {
    // hue disagreement stratified by chroma
    const double edges[] = { 0.0, 1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4, 3e-4, 1e-3, 1e9 };
    for(int b = 0; b < 11; b++)
    {
      double mx = 0, sum = 0; long cnt = 0, big = 0;
      for(int i = 0; i < N; i++)
      {
        const double Cz = cpu[4*i+1];
        if(!(Cz >= edges[b] && Cz < edges[b+1])) continue;
        double d = fabs((double)cpu[4*i+2] - (double)gpu[4*i+2]);
        double alt = 1.0 - d; if(alt < d) d = alt;
        cnt++; sum += d; if(d > mx) mx = d; if(d > 1.0/255.0) big++;
      }
      if(cnt) printf("  Cz in [%.0e,%.0e): n=%ld  hue max=%.4g mean=%.3g  >1/255: %ld (%.2f%%)\n",
                     edges[b], edges[b+1], cnt, mx, sum/cnt, big, 100.0*big/cnt);
    }
  }
  const char *lbl[3] = { "Jz", "Cz", "hz" };
  for(int c = 0; c < 3; c++)
  {
    double mx = 0, sum = 0;
    int nbig = 0, worst = 0;
    for(int i = 0; i < N; i++)
    {
      double d = fabs((double)cpu[4 * i + c] - (double)gpu[4 * i + c]);
      if(c == 2) { double alt = 1.0 - d; if(alt < d) d = alt; } // hue wraps
      sum += d;
      if(d > mx) { mx = d; worst = i; }
      if(d > 1.0 / 255.0) nbig++;
    }
    printf("%s: max=%.6g  mean=%.3g  pixels>1/255: %d (%.3f%%)\n",
           lbl[c], mx, sum / N, nbig, 100.0 * nbig / N);
    printf("     worst sample XYZ=(%.6g %.6g %.6g) cpu=%.9g gpu=%.9g Cz=%.4g\n",
           in[4 * worst], in[4 * worst + 1], in[4 * worst + 2],
           cpu[4 * worst + c], gpu[4 * worst + c], cpu[4 * worst + 1]);
  }
  return 0;
}
