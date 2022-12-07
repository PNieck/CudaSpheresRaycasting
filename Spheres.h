#ifndef __SPHERES_H__
#define __SPHERES_H__


#include "cuda_runtime.h"


typedef struct Spheres_t {
	float3* centers;
	float1* radiouses;

	float* r, *g, *b;

	int cnt;
} Spheres_t;


int spheresInit(Spheres_t* spheres);


void spheresFree(Spheres_t* spheres);


int deviceSpheresAlloc(Spheres_t* h_spheres, Spheres_t* d_spheres);


int deviceSpheresInit(Spheres_t* d_spheres);


void deviceSpheresFree(Spheres_t* d_spheres);


#endif