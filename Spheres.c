#include "Spheres.h"

#include <stdlib.h>
#include <time.h>

#include "math_help.h"


#define SPHERES_NUM 1000

#define MAX_POS 1000
#define MIN_POS -1000

#define MAX_RADIOUS 50
#define MIN_RADIOUS 5


int spheresInit(Spheres_t* spheres)
{
	spheres->centers = malloc(sizeof(float3) * SPHERES_NUM);  
	spheres->radiouses = malloc(sizeof(float1) * SPHERES_NUM);
	spheres->r = malloc(sizeof(float) * SPHERES_NUM);
	spheres->g = malloc(sizeof(float) * SPHERES_NUM);
	spheres->b = malloc(sizeof(float) * SPHERES_NUM);

	if (spheres->centers == NULL || spheres->radiouses == NULL ||
		spheres->r ==NULL || spheres->g == NULL || spheres->b == NULL) {
		free(spheres->centers);
		free(spheres->radiouses);

		free(spheres->r);
		free(spheres->g);
		free(spheres->b);

		return -1;
	}

	spheres->cnt = SPHERES_NUM;

	randomInit();

	for (int i = 0; i < SPHERES_NUM; i++)
	{
		spheres->centers[i] = make_float3(randomInt(MAX_POS, MIN_POS),
										  randomInt(MAX_POS, MIN_POS),
										  randomInt(MAX_POS, MIN_POS));
		spheres->radiouses[i] = make_float1(randomInt(MAX_RADIOUS, MIN_RADIOUS));
		spheres->r[i] = randomFloat0_to_1();
		spheres->g[i] = randomFloat0_to_1();
		spheres->b[i] = randomFloat0_to_1();
	}

	return 0;
}


void spheresFree(Spheres_t* spheres)
{
	if (spheres == NULL) {
		return;
	}
	
	free(spheres->centers);
	free(spheres->radiouses);
	
	free(spheres->r);
	free(spheres->g);
	free(spheres->b);

	spheres->cnt = 0;
}


int deviceSpheresAlloc(Spheres_t* h_spheres, Spheres_t* d_spheres)
{
	cudaError_t err;

	err = cudaMalloc(&d_spheres->centers, sizeof(float3) * h_spheres->cnt);
	if (err != cudaSuccess) {
		return -1;
	}

	err = cudaMalloc(&d_spheres->radiouses, sizeof(float1) * h_spheres->cnt);
	if (err != cudaSuccess) {
		cudaFree(d_spheres->centers);
		return -1;
	}

	err = cudaMalloc(&d_spheres->r, sizeof(float) * h_spheres->cnt);
	if (err != cudaSuccess) {
		cudaFree(d_spheres->centers);
		cudaFree(d_spheres->radiouses);
		return -1;
	}

	err = cudaMalloc(&d_spheres->g, sizeof(float) * h_spheres->cnt);
	if (err != cudaSuccess) {
		cudaFree(d_spheres->centers);
		cudaFree(d_spheres->radiouses);
		cudaFree(d_spheres->r);
		return -1;
	}

	err = cudaMalloc(&d_spheres->b, sizeof(float) * h_spheres->cnt);
	if (err != cudaSuccess) {
		cudaFree(d_spheres->centers);
		cudaFree(d_spheres->radiouses);
		cudaFree(d_spheres->r);
		cudaFree(d_spheres->g);
		return -1;
	}

	err = cudaMemcpy(d_spheres->centers, h_spheres->centers, sizeof(float3) * h_spheres->cnt, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		deviceSpheresFree(d_spheres);
		return -1;
	}

	err = cudaMemcpy(d_spheres->radiouses, h_spheres->radiouses, sizeof(float1) * h_spheres->cnt, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		deviceSpheresFree(d_spheres);
		return -1;
	}

	err = cudaMemcpy(d_spheres->r, h_spheres->r, sizeof(float) * h_spheres->cnt, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		deviceSpheresFree(d_spheres);
		return -1;
	}

	err = cudaMemcpy(d_spheres->g, h_spheres->g, sizeof(float) * h_spheres->cnt, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		deviceSpheresFree(d_spheres);
		return -1;
	}

	err = cudaMemcpy(d_spheres->b, h_spheres->b, sizeof(float) * h_spheres->cnt, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		deviceSpheresFree(d_spheres);
		return -1;
	}

	d_spheres->cnt = h_spheres->cnt;

	return 0;
}


int deviceSpheresInit(Spheres_t* spheres)
{
	Spheres_t h_spheres;
	int err;

	err = spheresInit(&h_spheres);
	if (err != 0) {
		return -1;
	}

	err = deviceSpheresAlloc(&h_spheres, spheres);
	spheresFree(&h_spheres);

	return err == 0 ? 0 : -1;
}


void deviceSpheresFree(Spheres_t* d_spheres)
{
	cudaFree(d_spheres->centers);
	cudaFree(d_spheres->radiouses);

	cudaFree(d_spheres->r);
	cudaFree(d_spheres->g);
	cudaFree(d_spheres->b);
}
