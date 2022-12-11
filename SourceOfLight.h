#ifndef __SOL_H__
#define __SOL_H__


#include "cuda_runtime.h"


typedef struct LightSource_t {
	float3* coor;

	float* r, * g, * b;

	int cnt;
} LightSource_t;


int lightSourceInit(LightSource_t* sol);


void lightSourceFree(LightSource_t* sol);


int deviceLightSourceAlloc(LightSource_t* h_sol, LightSource_t* d_sol);


int deviceLightSourceInit(LightSource_t* d_sol);


void deviceLightSourceFree(LightSource_t* d_sol);


#endif