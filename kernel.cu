// -*- compile-command: "nvcc arch sm_50 -Xptxas=-v -cubin kernel.cu"; -*-

//
//
//

#ifdef __cplusplus
extern "C" {
#endif

#include "assert_cuda.h"

#ifdef __cplusplus
}
#endif


#include "SourceOfLight.h"
#include "screen.h"
#include "Spheres.h"
#include "math_help.h"
#include <math.h>

#include <stdio.h>

#define SPEC_REF_CONST 0.2f
#define DIFF_REF_CONST 0.2f
#define AMB_REF_CONST 0.2f
#define ALPHA 20


//
//
//

#define PXL_KERNEL_THREADS_PER_BLOCK  256 // enough for 4Kx2 monitor

//
//
//

surface<void,cudaSurfaceType2D> surf;

//
//
//

union pxl_rgbx_24
{
  uint1       b32;

  struct {
    unsigned  r  : 8;
    unsigned  g  : 8;
    unsigned  b  : 8;
    unsigned  na : 8;
  };
};

//
//
//

__device__ bool isHit(float3 center, float1 radious, float3 pixel, float3 screenNorm)
{
    float3 diff = vectorDiff(center, pixel);
    float3 cross = crossProduct(screenNorm, diff);

    if (lenSquared(cross) <= radious.x * radious.x) {
        return true;
    }

    return false;
}


__device__ float hitPointDst(float3 center, float1 radious, float3 pixel, float3 screenNorm)
{
    float3 diff = vectorDiff(pixel, center);
    float temp = dotProduct(screenNorm, diff);

    return -temp - sqrtf(temp * temp - lenSquared(diff) + radious.x * radious.x);
}


__device__ union pxl_rgbx_24 colorCalculate(float3 hitPoint, float3 center, float r, float g, float b, float3 screenNorm, LightSource_t sol)
{
    float res_r, res_g, res_b;

    float3 N = vectorDiff(hitPoint, center);
    vectorNormalize(&N);

    float3 V = vectorNegate(screenNorm);
    vectorNormalize(&V);

    res_r = res_g = res_b = AMB_REF_CONST * 1;

    for (int i = 0; i < sol.cnt; i++) {
        float3 L = vectorDiff(sol.coor[i], hitPoint);
        vectorNormalize(&L);

        float3 R = vectorDiff(vectorMul(N, 2 * dotProduct(L, N)), L);

        float coef1 = maxf(dotProduct(L, N), 0.0f) * DIFF_REF_CONST;

        float coef2 = maxf(dotProduct(R, V), 0.0f);
        if (coef2 != 0) {
            coef2 = SPEC_REF_CONST * powf(coef2, ALPHA);
        }

        float i_r = sol.r[i] * r;
        float i_g = sol.g[i] * g;
        float i_b = sol.b[i] * b;

        res_r +=  coef1 * i_r + coef2 * i_r;

        //printf("r = %f", res_r);
        res_g += coef1 * i_g + coef2 * i_g;
        res_b += coef1 * i_b + coef2 * i_b;
    }

    union pxl_rgbx_24  rgbx;
    rgbx.r = (unsigned int)(minf(res_r, 1.0f) * 255.0f);
    rgbx.g = (unsigned int)(minf(res_g, 1.0f) * 255.0f);
    rgbx.b = (unsigned int)(minf(res_b, 1.0f) * 255.0f);

    return rgbx;
}

//
//
//

extern "C"
__global__
void
pxl_kernel(const int width, const int height, Spheres_t spheres, Screen_t* screen, LightSource_t sol)
{
  // pixel coordinates
  const int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
  const int x   = idx % width - width/2;
  const int y   = idx / width - height/2;

  union pxl_rgbx_24  rgbx;
  float hitPointDist = INFINITY;
  int index = 0;

  float3 pixel = make_float3(screen->middle.x, screen->middle.y + x, screen->middle.z + y);

  rgbx.na = 255;

  int id = idx % spheres.cnt;
  for (int i = 0; i < spheres.cnt; i++)
  {
      id = (id + 1) % spheres.cnt;
      if (isHit(spheres.centers[id], spheres.radiouses[id], pixel, screen->normalVec))
      {
          float d = hitPointDst(spheres.centers[id], spheres.radiouses[id], pixel, screen->normalVec);
          
          if (hitPointDist > d) {
              hitPointDist = d;
              index = id;
          }
      }
  }

  if (hitPointDist == INFINITY) {
      rgbx.r = 0;
      rgbx.g = 0;
      rgbx.b = 0;
  }
  else {
      float3 hitPoint = vectorAdd(pixel, vectorMul(screen->normalVec, hitPointDist));

      rgbx = colorCalculate(hitPoint,
          spheres.centers[index],
          spheres.r[index],
          spheres.g[index],
          spheres.b[index],
          screen->normalVec,
          sol);
  }

#if 0

  /*// pixel color
  const int          t    = (unsigned int)clock() / 1100000; // 1.1 GHz
  const int          xt   = (idx + t) % width;
  const unsigned int ramp = (unsigned int)(((float)xt / (float)(width-1)) * 255.0f + 0.5f);
  const unsigned int bar  = ((y + t) / 32) & 3;

  union pxl_rgbx_24  rgbx;

  rgbx.r  = (bar == 0) || (bar == 1) ? ramp : 0;
  rgbx.g  = (bar == 0) || (bar == 2) ? ramp : 0;
  rgbx.b  = (bar == 0) || (bar == 3) ? ramp : 0; 
  rgbx.na = 255;*/

#else // DRAW A RED BORDER TO VALIDATE FLIPPED BLIT

  /*const bool        border = (x == 0) || (x == width - 1) || (y == 0) || (y == height - 1);
  union pxl_rgbx_24 rgbx   = { border ? 0xFF0000FF : 0xFF000000 };*/
  
#endif

  surf2Dwrite(rgbx.b32, // even simpler: (unsigned int)clock()
    surf,
    (x + width/2)*sizeof(rgbx),
    y + height/2,
    cudaBoundaryModeZero); // squelches out-of-bound writes
}

//
//
//

extern "C"
cudaError_t
pxl_kernel_launcher(cudaArray_const_t array,
                    const int         width,
                    const int         height,
                    Spheres_t         d_spheres,
                    Screen_t         *screen,
                    LightSource_t     sol,
                    cudaEvent_t       event,
                    cudaStream_t      stream)
{
  cudaError_t cuda_err;

  // cuda_err = cudaEventRecord(event,stream);

  cuda_err = cuda(BindSurfaceToArray(surf,array));

  if (cuda_err)
    return cuda_err;

  const int blocks = (width * height + PXL_KERNEL_THREADS_PER_BLOCK - 1) / PXL_KERNEL_THREADS_PER_BLOCK;

  // cuda_err = cudaEventRecord(event,stream);

  if (blocks > 0)
    pxl_kernel<<<blocks,PXL_KERNEL_THREADS_PER_BLOCK,0,stream>>>(width,height, d_spheres, screen, sol);

  // cuda_err = cudaStreamWaitEvent(stream,event,0);
  
  return cudaSuccess;
}

//
//
//
