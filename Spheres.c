#include "Spheres.h"

#include <stdlib.h>


#define SPHERES_NUM 100


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
	
	spheres->centers[0] = make_float3(0, 0, 0); 
	spheres->radiouses[0] = make_float1(100);
	spheres->r[0] = 255 / 255;
	spheres->g[0] = 17 / 255;
	spheres->b[0] = 0 / 255;

	spheres->centers[1] = make_float3(0, 200, 0);
	spheres->radiouses[1] = make_float1(50);
	spheres->r[1] = 13 / 255;
	spheres->g[1] = 255 / 255;
	spheres->b[1] = 0 / 255;

	spheres->centers[2] = make_float3(50, -50, 150);
	spheres->radiouses[2] = make_float1(30);
	spheres->r[2] = 0 / 255;
	spheres->g[2] = 21 / 255;
	spheres->b[2] = 255 / 255;

	for (int i = 3; i < SPHERES_NUM; i++)
	{
		spheres->centers[i] = make_float3(50, -50, 1500);
		spheres->radiouses[i] = make_float1(30);
		spheres->r[i] = 255 / 255;
		spheres->g[i] = 255 / 255;
		spheres->b[i] = 255 / 255;
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
