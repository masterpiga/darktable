// Where does the CPU/GPU difference in JzAzBz actually enter?
// Emits the two intermediate stages plus the final Az/Bz from both sides.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <OpenCL/opencl.h>
#define N 200000
static const float M_t[3][4] = {{0.41478972f,-0.2015100f,-0.0166008f,0},
                                {0.57999900f, 1.1206490f, 0.2648000f,0},
                                {0.01464800f, 0.0531008f, 0.6684799f,0}};
static const float A_t[3][4] = {{0.5f, 3.524000f, 0.199076f,0},
                                {0.5f,-4.066708f, 1.096799f,0},
                                {0.0f, 0.542708f,-1.295875f,0}};
static void host(const float X[4], float lms_raw[3], float lms_p[3], float ab[3])
{
  float XYZ[3];
  XYZ[0]=1.15f*X[0]-0.15f*X[2]; XYZ[1]=0.66f*X[1]+0.34f*X[0]; XYZ[2]=X[2];
  for(int i=0;i<3;i++) lms_raw[i]=M_t[0][i]*XYZ[0]+M_t[1][i]*XYZ[1]+M_t[2][i]*XYZ[2];
  for(int i=0;i<3;i++){
    float t=powf(fmaxf(lms_raw[i]/10000.f,0.f),0.159301758f);
    lms_p[i]=powf((0.8359375f+18.8515625f*t)/(1.0f+18.6875f*t),134.034375f);
  }
  for(int i=0;i<3;i++) ab[i]=A_t[0][i]*lms_p[0]+A_t[1][i]*lms_p[1]+A_t[2][i]*lms_p[2];
}
static const char *ksrc =
"__kernel void run(__global const float4 *in, __global float4 *lmsraw, __global float4 *lmsp, __global float4 *ab)\n"
"{ int i=get_global_id(0);\n"
"  const float4 M[3]={{0.41478972f,0.579999f,0.0146480f,0},{-0.2015100f,1.120649f,0.0531008f,0},{-0.0166008f,0.264800f,0.6684799f,0}};\n"
"  const float4 A[3]={{0.5f,0.5f,0.0f,0},{3.524000f,-4.066708f,0.542708f,0},{0.199076f,1.096799f,-1.295875f,0}};\n"
"  float4 X=in[i], t1, t2;\n"
"  t1.x=1.15f*X.x-0.15f*X.z; t1.y=0.66f*X.y+0.34f*X.x; t1.z=X.z; t1.w=0.f;\n"
"  t2.x=dot(M[0],t1); t2.y=dot(M[1],t1); t2.z=dot(M[2],t1); t2.w=0.f;\n"
"  lmsraw[i]=t2;\n"
"  t2=pow(fmax(t2/10000.f,0.0f),0.159301758f);\n"
"  t2=pow((0.8359375f+18.8515625f*t2)/(1.0f+18.6875f*t2),134.034375f);\n"
"  lmsp[i]=t2;\n"
"  float4 o; o.x=dot(A[0],t2); o.y=dot(A[1],t2); o.z=dot(A[2],t2); o.w=0.f; ab[i]=o; }\n";
int main(void)
{
  float *in=malloc(16*N),*graw=malloc(16*N),*gp=malloc(16*N),*gab=malloc(16*N);
  float *hraw=malloc(12*N),*hp=malloc(12*N),*hab=malloc(12*N);
  srandom(7);
  for(int i=0;i<N;i++){
    float Y=powf((float)(random()%100000)/100000.f,3.f)*4.f;
    float s=0.35f;
    in[4*i+0]=Y*(0.95047f+((float)(random()%200000)/100000.f-1.f)*s);
    in[4*i+1]=Y;
    in[4*i+2]=Y*(1.08883f+((float)(random()%200000)/100000.f-1.f)*s);
    in[4*i+3]=0;
    host(in+4*i,hraw+3*i,hp+3*i,hab+3*i);
  }
  cl_platform_id pl; cl_device_id d; clGetPlatformIDs(1,&pl,NULL);
  clGetDeviceIDs(pl,CL_DEVICE_TYPE_GPU,1,&d,NULL);
  cl_context c=clCreateContext(NULL,1,&d,NULL,NULL,NULL);
  cl_command_queue q=clCreateCommandQueue(c,d,0,NULL);
  cl_program p=clCreateProgramWithSource(c,1,&ksrc,NULL,NULL);
  if(clBuildProgram(p,1,&d,"",NULL,NULL)!=CL_SUCCESS){char L[8192];clGetProgramBuildInfo(p,d,CL_PROGRAM_BUILD_LOG,8192,L,NULL);puts(L);return 1;}
  cl_kernel k=clCreateKernel(p,"run",NULL);
  cl_mem bi=clCreateBuffer(c,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,16*N,in,NULL);
  cl_mem b1=clCreateBuffer(c,CL_MEM_WRITE_ONLY,16*N,NULL,NULL);
  cl_mem b2=clCreateBuffer(c,CL_MEM_WRITE_ONLY,16*N,NULL,NULL);
  cl_mem b3=clCreateBuffer(c,CL_MEM_WRITE_ONLY,16*N,NULL,NULL);
  clSetKernelArg(k,0,8,&bi);clSetKernelArg(k,1,8,&b1);clSetKernelArg(k,2,8,&b2);clSetKernelArg(k,3,8,&b3);
  size_t g=N; clEnqueueNDRangeKernel(q,k,1,NULL,&g,NULL,0,NULL,NULL);
  clEnqueueReadBuffer(q,b1,CL_TRUE,0,16*N,graw,0,NULL,NULL);
  clEnqueueReadBuffer(q,b2,CL_TRUE,0,16*N,gp,0,NULL,NULL);
  clEnqueueReadBuffer(q,b3,CL_TRUE,0,16*N,gab,0,NULL,NULL);
  double r1=0,r2=0,a1=0,a2=0;
  for(int i=0;i<N;i++)for(int ch=0;ch<3;ch++){
    double h=hraw[3*i+ch],gg=graw[4*i+ch];
    if(fabs(h)>1e-12){double e=fabs(h-gg)/fabs(h); if(e>r1)r1=e;}
    h=hp[3*i+ch];gg=gp[4*i+ch];
    if(fabs(h)>1e-12){double e=fabs(h-gg)/fabs(h); if(e>r2)r2=e;}
    h=hab[3*i+ch];gg=gab[4*i+ch];
    double ae=fabs(h-gg);
    if(ch==0){ if(ae>a1)a1=ae; } else { if(ae>a2)a2=ae; }
  }
  printf("max RELATIVE diff, LMS (before the powers) : %.3g\n", r1);
  printf("max RELATIVE diff, L'M'S' (after pow^134)  : %.3g\n", r2);
  printf("max ABSOLUTE diff, Iz                      : %.3g\n", a1);
  printf("max ABSOLUTE diff, Az/Bz                   : %.3g\n", a2);
  return 0;
}
