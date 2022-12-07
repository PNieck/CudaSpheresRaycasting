#ifndef __SOL_H__
#define __SOL_H__


#include "cuda_runtime.h"


typedef struct LightSource_t {
	float3* coor;

	float* r, * g, * b;

	int cnt;
} LightSource_t;


int lightSourceInit(LightSource_t* spheres);


void lightSourceFree(LightSource_t* spheres);


int deviceLightSourceAlloc(LightSource_t* h_spheres, LightSource_t* d_spheres);


int deviceLightSourceInit(LightSource_t* d_spheres);


void deviceLightSourceFree(LightSource_t* d_spheres);


#endif