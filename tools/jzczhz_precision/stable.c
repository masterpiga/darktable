// Is the JzCzhz CPU/GPU divergence removable? Compares the current formulation
// against a cancellation-free one, on both the host and the GPU.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <OpenCL/opencl.h>
#define N 200000
#define NN 0.159301758f
#define PP 134.034375f
#define C1 0.8359375f
#define C2 18.8515625f
#define C3 18.6875f
static const float M[3][3]={{0.41478972f,0.579999f,0.0146480f},
                            {-0.2015100f,1.120649f,0.0531008f},
                            {-0.0166008f,0.264800f,0.6684799f}};
static void prime(const float X[3], float o[3])
{ o[0]=1.15f*X[0]-0.15f*X[2]; o[1]=0.66f*X[1]+0.34f*X[0]; o[2]=X[2]; }
static void host_cur(const float X[3], float ab[2])
{
  float x[3]; prime(X,x); float L[3];
  for(int i=0;i<3;i++) L[i]=M[i][0]*x[0]+M[i][1]*x[1]+M[i][2]*x[2];
  for(int i=0;i<3;i++){ float t=powf(fmaxf(L[i]/10000.f,0.f),NN);
                        L[i]=powf((C1+C2*t)/(1.0f+C3*t),PP); }
  ab[0]=3.524000f*L[0]-4.066708f*L[1]+0.542708f*L[2];
  ab[1]=0.199076f*L[0]+1.096799f*L[1]-1.295875f*L[2];
}
// cancellation-free: carry the DIFFERENCES through the two power laws instead
// of forming three near-equal values and subtracting them at the end
static void host_stable(const float X[3], float ab[2])
{
  float x[3]; prime(X,x);
  float L[3];
  for(int i=0;i<3;i++) L[i]=M[i][0]*x[0]+M[i][1]*x[1]+M[i][2]*x[2];
  // differences taken on the matrix rows, so no cancellation of computed values
  const float dLM=(M[0][0]-M[1][0])*x[0]+(M[0][1]-M[1][1])*x[1]+(M[0][2]-M[1][2])*x[2];
  const float dSM=(M[2][0]-M[1][0])*x[0]+(M[2][1]-M[1][1])*x[1]+(M[2][2]-M[1][2])*x[2];
  const float dLS=(M[0][0]-M[2][0])*x[0]+(M[0][1]-M[2][1])*x[1]+(M[0][2]-M[2][2])*x[2];
  for(int i=0;i<3;i++) if(L[i]<1e-20f) { host_cur(X,ab); return; } // fall back out of gamut
  float t[3], y[3], p[3];
  for(int i=0;i<3;i++){ t[i]=powf(L[i]/10000.f,NN); y[i]=(C1+C2*t[i])/(1.0f+C3*t[i]); p[i]=powf(y[i],PP); }
  const float k=(C2-C1*C3);
  #define DT(a,b,d) (t[b]*expm1f(NN*log1pf((d)/L[b])))
  const float tLM=DT(0,1,dLM), tSM=DT(2,1,dSM), tLS=DT(0,2,dLS);
  #define DY(a,b,dt) ((k*(dt))/((1.0f+C3*t[a])*(1.0f+C3*t[b])))
  const float yLM=DY(0,1,tLM), ySM=DY(2,1,tSM), yLS=DY(0,2,tLS);
  const float pLM=p[1]*expm1f(PP*log1pf(yLM/y[1]));
  const float pSM=p[1]*expm1f(PP*log1pf(ySM/y[1]));
  const float pLS=p[2]*expm1f(PP*log1pf(yLS/y[2]));
  ab[0]=3.524000f*pLM+0.542708f*pSM;
  ab[1]=0.199076f*pLS-1.096799f*pSM;
}
static const char *ksrc =
"#define NN 0.159301758f\n#define PP 134.034375f\n#define C1 0.8359375f\n#define C2 18.8515625f\n#define C3 18.6875f\n"
"__kernel void run(__global const float4 *in, __global float2 *cur, __global float2 *stb)\n"
"{ int i=get_global_id(0); float4 X=in[i];\n"
"  const float3 Mr[3]={(float3)(0.41478972f,0.579999f,0.0146480f),(float3)(-0.2015100f,1.120649f,0.0531008f),(float3)(-0.0166008f,0.264800f,0.6684799f)};\n"
"  float3 x=(float3)(1.15f*X.x-0.15f*X.z, 0.66f*X.y+0.34f*X.x, X.z);\n"
"  float3 L=(float3)(dot(Mr[0],x),dot(Mr[1],x),dot(Mr[2],x));\n"
"  float3 t,y,p;\n"
"  t=pow(fmax(L/10000.f,0.0f),NN); y=(C1+C2*t)/(1.0f+C3*t); p=pow(y,PP);\n"
"  cur[i]=(float2)(3.524000f*p.x-4.066708f*p.y+0.542708f*p.z, 0.199076f*p.x+1.096799f*p.y-1.295875f*p.z);\n"
"  if(L.x<1e-20f||L.y<1e-20f||L.z<1e-20f){ stb[i]=cur[i]; return; }\n"
"  float dLM=dot(Mr[0]-Mr[1],x), dSM=dot(Mr[2]-Mr[1],x), dLS=dot(Mr[0]-Mr[2],x);\n"
"  float k=(C2-C1*C3);\n"
"  float tLM=t.y*expm1(NN*log1p(dLM/L.y)), tSM=t.y*expm1(NN*log1p(dSM/L.y)), tLS=t.z*expm1(NN*log1p(dLS/L.z));\n"
"  float yLM=(k*tLM)/((1.0f+C3*t.x)*(1.0f+C3*t.y));\n"
"  float ySM=(k*tSM)/((1.0f+C3*t.z)*(1.0f+C3*t.y));\n"
"  float yLS=(k*tLS)/((1.0f+C3*t.x)*(1.0f+C3*t.z));\n"
"  float pLM=p.y*expm1(PP*log1p(yLM/y.y)), pSM=p.y*expm1(PP*log1p(ySM/y.y)), pLS=p.z*expm1(PP*log1p(yLS/y.z));\n"
"  stb[i]=(float2)(3.524000f*pLM+0.542708f*pSM, 0.199076f*pLS-1.096799f*pSM); }\n";
int main(void)
{
  float *in=malloc(16*N),*hc=malloc(8*N),*hs=malloc(8*N),*gc=malloc(8*N),*gs=malloc(8*N);
  srandom(11);
  for(int i=0;i<N;i++){
    float Y=powf((float)(random()%100000)/100000.f,3.f)*4.f, s=0.35f;
    float X3[3]={Y*(0.95047f+((float)(random()%200000)/100000.f-1.f)*s),Y,
                 Y*(1.08883f+((float)(random()%200000)/100000.f-1.f)*s)};
    in[4*i]=X3[0];in[4*i+1]=X3[1];in[4*i+2]=X3[2];in[4*i+3]=0;
    host_cur(X3,hc+2*i); host_stable(X3,hs+2*i);
  }
  cl_platform_id pl;cl_device_id d;clGetPlatformIDs(1,&pl,NULL);clGetDeviceIDs(pl,CL_DEVICE_TYPE_GPU,1,&d,NULL);
  cl_context c=clCreateContext(NULL,1,&d,NULL,NULL,NULL);cl_command_queue q=clCreateCommandQueue(c,d,0,NULL);
  cl_program p=clCreateProgramWithSource(c,1,&ksrc,NULL,NULL);
  if(clBuildProgram(p,1,&d,"",NULL,NULL)!=CL_SUCCESS){char L[8192];clGetProgramBuildInfo(p,d,CL_PROGRAM_BUILD_LOG,8192,L,NULL);puts(L);return 1;}
  cl_kernel k=clCreateKernel(p,"run",NULL);
  cl_mem bi=clCreateBuffer(c,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,16*N,in,NULL);
  cl_mem b1=clCreateBuffer(c,CL_MEM_WRITE_ONLY,8*N,NULL,NULL),b2=clCreateBuffer(c,CL_MEM_WRITE_ONLY,8*N,NULL,NULL);
  clSetKernelArg(k,0,8,&bi);clSetKernelArg(k,1,8,&b1);clSetKernelArg(k,2,8,&b2);
  size_t g=N;clEnqueueNDRangeKernel(q,k,1,NULL,&g,NULL,0,NULL,NULL);
  clEnqueueReadBuffer(q,b1,CL_TRUE,0,8*N,gc,0,NULL,NULL);
  clEnqueueReadBuffer(q,b2,CL_TRUE,0,8*N,gs,0,NULL,NULL);
  double mc=0,ms=0,hcz=0,hsz=0; long big_c=0,big_s=0;
  for(int i=0;i<N;i++){
    double a=fabs(hc[2*i]-gc[2*i]),b=fabs(hc[2*i+1]-gc[2*i+1]);
    if(a>mc)mc=a; if(b>mc)mc=b;
    double a2=fabs(hs[2*i]-gs[2*i]),b2=fabs(hs[2*i+1]-gs[2*i+1]);
    if(a2>ms)ms=a2; if(b2>ms)ms=b2;
    double Cc=hypot(hc[2*i],hc[2*i+1]);
    double h1=atan2(hc[2*i+1],hc[2*i]), h2=atan2(gc[2*i+1],gc[2*i]);
    double dh=fabs(h1-h2)/(2*M_PI); if(dh>0.5)dh=1-dh; if(dh>hcz)hcz=dh; if(dh>1.0/255)big_c++;
    double h3=atan2(hs[2*i+1],hs[2*i]), h4=atan2(gs[2*i+1],gs[2*i]);
    double dh2=fabs(h3-h4)/(2*M_PI); if(dh2>0.5)dh2=1-dh2; if(dh2>hsz)hsz=dh2; if(dh2>1.0/255)big_s++;
    (void)Cc;
  }
  {
    double hmax=0, hmean=0; long n=0, big=0;
    for(int i=0;i<N;i++){
      double h1=atan2(hc[2*i+1],hc[2*i]), h2=atan2(hs[2*i+1],hs[2*i]);
      double dh=fabs(h1-h2)/(2*M_PI); if(dh>0.5)dh=1-dh;
      double Cz=hypot(hc[2*i],hc[2*i+1]);
      if(Cz<1e-9) continue;
      n++; hmean+=dh; if(dh>hmax)hmax=dh; if(dh>1.0/255)big++;
    }
    printf("HOST current vs HOST cancellation-free (what an existing render would see):\n");
    printf("  hue: max %.4g  mean %.3g  samples>1/255 %ld of %ld (%.3f%%)\n\n", hmax, hmean/n, big, n, 100.0*big/n);
  }
  printf("CPU vs GPU, current formulation    : Az/Bz max abs %.3g   hue max %.4g   samples>1/255 %ld\n",mc,hcz,big_c);
  printf("CPU vs GPU, cancellation-free      : Az/Bz max abs %.3g   hue max %.4g   samples>1/255 %ld\n",ms,hsz,big_s);
  return 0;
}
