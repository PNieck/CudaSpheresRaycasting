#ifndef __HELP_MATH_H__
#define __HELP_MATH_H__


#include <cuda_runtime.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>


inline __device__ float maxf(float a, float b)
{
	return a > b ? a : b;
}


inline __device__ float minf(float a, float b)
{
	return a < b ? a : b;
}


inline __device__ float3 vectorAdd(float3 v1, float3 v2)
{
	return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}


inline __device__ float3 crossProduct(float3 v1, float3 v2)
{
	return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}


inline __device__ float dotProduct(float3 v1, float3 v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}


inline __device__ float3 vectorDiff(float3 v1, float3 v2)
{
	return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}


inline __device__ void vectorMulSelf(float3* v, float d)
{
	v->x *= d;
	v->y *= d;
	v->z *= d;
}


inline __device__ float3 vectorMul(float3 v, float d)
{
	return make_float3(v.x * d, v.y * d, v.z * d);
}


inline __device__ float lenSquared(float3 v)
{
	return v.x * v.x + v.y * v.y + v.z * v.z;
}


inline __device__ float vectorLen(float3 v)
{
	return sqrtf(lenSquared(v));
}


inline __device__ void vectorNormalize(float3* v)
{
	float invlen = 1.0f / vectorLen(*v);
	vectorMulSelf(v, invlen);
}


inline __device__ float3 vectorNegate(float3 v)
{
	return make_float3(-v.x, -v.y, -v.z);
}


inline void randomInit()
{
	srand(time(NULL));
}


inline int randomInt(int max, int min)
{
	return rand() % (max - min + 1) + min;
}


inline float randomFloat0_to_1()
{
	return (float)rand() / (float)RAND_MAX;
}

#endif
