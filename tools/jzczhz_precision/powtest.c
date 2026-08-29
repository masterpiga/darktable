// Can the OpenCL side evaluate y^134.034375 as accurately as the host powf?
// Compares candidate formulations against a double-precision reference and
// against the host powf, on the y range the JzAzBz PQ curve actually produces.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <OpenCL/opencl.h>
#define N 200000
#define P 134.034375f
static const char *ksrc =
"#define P 134.034375f\n"
"__kernel void run(__global const float *y, __global float *o0, __global float *o1,\n"
"                  __global float *o2, __global float *o3, __global float *o4)\n"
"{ int i=get_global_id(0); float v=y[i];\n"
"  o0[i] = pow(v, P);\n"
"  o1[i] = powr(v, P);\n"
"  o2[i] = exp2(P * log2(v));\n"
"  o3[i] = exp(P * log1p(v - 1.0f));\n"
"  // compensated: split P*log(v) into a high and a low part so the exponent\n"
"  // multiplication does not throw away bits before exp() sees them\n"
"  { float l = log1p(v - 1.0f);\n"
"    float hi = P * l;\n"
"    float lo = fma(P, l, -hi);\n"
"    o4[i] = exp(hi) * (1.0f + lo); }\n"
"}\n";
int main(void)
{
  float *y=malloc(4*N); double *ref=malloc(8*N); float *hostp=malloc(4*N);
  srandom(3);
  for(int i=0;i<N;i++){
    // the PQ curve's output range: (c1 + c2 t)/(1 + c3 t), t in [0,~2]
    double t = (double)(random()%1000000)/1000000.0 * 2.0;
    double v = (0.8359375 + 18.8515625*t)/(1.0 + 18.6875*t);
    y[i]=(float)v; ref[i]=pow((double)y[i], (double)P); hostp[i]=powf(y[i], P);
  }
  cl_platform_id pl; cl_device_id d; clGetPlatformIDs(1,&pl,NULL);
  clGetDeviceIDs(pl,CL_DEVICE_TYPE_GPU,1,&d,NULL);
  cl_context c=clCreateContext(NULL,1,&d,NULL,NULL,NULL);
  cl_command_queue q=clCreateCommandQueue(c,d,0,NULL);
  cl_program p=clCreateProgramWithSource(c,1,&ksrc,NULL,NULL);
  if(clBuildProgram(p,1,&d,"",NULL,NULL)!=CL_SUCCESS){char L[8192];clGetProgramBuildInfo(p,d,CL_PROGRAM_BUILD_LOG,8192,L,NULL);puts(L);return 1;}
  cl_kernel k=clCreateKernel(p,"run",NULL);
  cl_mem bi=clCreateBuffer(c,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,4*N,y,NULL);
  cl_mem bo[5]; float *out[5];
  for(int j=0;j<5;j++){ bo[j]=clCreateBuffer(c,CL_MEM_WRITE_ONLY,4*N,NULL,NULL); out[j]=malloc(4*N); }
  clSetKernelArg(k,0,8,&bi);
  for(int j=0;j<5;j++) clSetKernelArg(k,j+1,8,&bo[j]);
  size_t g=N; clEnqueueNDRangeKernel(q,k,1,NULL,&g,NULL,0,NULL,NULL);
  for(int j=0;j<5;j++) clEnqueueReadBuffer(q,bo[j],CL_TRUE,0,4*N,out[j],0,NULL,NULL);
  const char *nm[5]={"pow(v,P)        (current)","powr(v,P)","exp2(P*log2 v)","exp(P*log1p(v-1))","compensated log1p+fma"};
  double hr=0; for(int i=0;i<N;i++){ if(ref[i]>1e-30){double e=fabs(hostp[i]-ref[i])/ref[i]; if(e>hr)hr=e;} }
  printf("host powf vs double reference : max rel %.3g\n\n", hr);
  for(int j=0;j<5;j++){
    double mr=0, mh=0;
    for(int i=0;i<N;i++){
      if(ref[i]<=1e-30) continue;
      double e=fabs(out[j][i]-ref[i])/ref[i]; if(e>mr)mr=e;
      double f=fabs((double)out[j][i]-(double)hostp[i])/ref[i]; if(f>mh)mh=f;
    }
    printf("%-26s max rel vs double %.3g   max rel vs host powf %.3g\n", nm[j], mr, mh);
  }
  return 0;
}
