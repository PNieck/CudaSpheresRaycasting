#include "SourceOfLight.h"

#include <stdlib.h>
#include "math_help.h"

#include <stdio.h>


#define SOL_NUM 10

#define MAX_POS_SOL 1200
#define MIN_POS_SOL -1200


int lightSourceInit(LightSource_t* sol)
{
	sol->coor = malloc(sizeof(float3) * SOL_NUM);
	sol->r = malloc(sizeof(float) * SOL_NUM);
	sol->g = malloc(sizeof(float) * SOL_NUM);
	sol->b = malloc(sizeof(float) * SOL_NUM);

	if (sol->coor == NULL ||
		sol->r == NULL || sol->g == NULL || sol->b == NULL) {
		free(sol->coor);

		free(sol->r);
		free(sol->g);
		free(sol->b);

		return -1;
	}

	sol->cnt = SOL_NUM;

	randomInit();

	for (int i = 0; i < SOL_NUM; i++)
	{
		sol->coor[i] = make_float3(randomInt(MAX_POS_SOL, MIN_POS_SOL),
								   randomInt(MAX_POS_SOL, MIN_POS_SOL),
								   randomInt(MAX_POS_SOL, MIN_POS_SOL));

		printf("coor.x: %f coor.y: %f, coor.z: %f\n", sol->coor[i].x, sol->coor[i].y, sol->coor[i].z);

		sol->r[i] = randomFloat0_to_1();
		sol->g[i] = randomFloat0_to_1();
		sol->b[i] = randomFloat0_to_1();

		printf("r: %f, g; %f, b; %f\n", sol->r[i], sol->g[i], sol->b[i]);
	}
	/*sol->coor[0] = make_float3(-200, 400, 0);
	sol->r[0] = 255 / 255;
	sol->g[0] = 255 / 255;
	sol->b[0] = 255 / 255;*/

	return 0;
}


void lightSourceFree(LightSource_t* sol)
{
	if (sol == NULL) {
		return;
	}

	free(sol->coor);

	free(sol->r);
	free(sol->g);
	free(sol->b);

	sol->cnt = 0;
}


int deviceLightSourceAlloc(LightSource_t* h_sol, LightSource_t* d_sol)
{
	cudaError_t err;

	err = cudaMalloc(&d_sol->coor, sizeof(float3) * h_sol->cnt);
	if (err != cudaSuccess) {
		return -1;
	}

	err = cudaMalloc(&d_sol->r, sizeof(float) * h_sol->cnt);
	if (err != cudaSuccess) {
		cudaFree(d_sol->coor);
		return -1;
	}

	err = cudaMalloc(&d_sol->g, sizeof(float) * h_sol->cnt);
	if (err != cudaSuccess) {
		cudaFree(d_sol->coor);
		cudaFree(d_sol->r);
		return -1;
	}

	err = cudaMalloc(&d_sol->b, sizeof(float) * h_sol->cnt);
	if (err != cudaSuccess) {
		cudaFree(d_sol->coor);
		cudaFree(d_sol->r);
		cudaFree(d_sol->g);
		return -1;
	}

	err = cudaMemcpy(d_sol->coor, h_sol->coor, sizeof(float3) * h_sol->cnt, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		deviceLightSourceFree(d_sol);
		return -1;
	}

	err = cudaMemcpy(d_sol->r, h_sol->r, sizeof(float) * h_sol->cnt, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		deviceLightSourceFree(d_sol);
		return -1;
	}

	err = cudaMemcpy(d_sol->g, h_sol->g, sizeof(float) * h_sol->cnt, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		deviceLightSourceFree(d_sol);
		return -1;
	}

	err = cudaMemcpy(d_sol->b, h_sol->b, sizeof(float) * h_sol->cnt, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		deviceLightSourceFree(d_sol);
		return -1;
	}

	d_sol->cnt = h_sol->cnt;

	return 0;
}


int deviceLightSourceInit(LightSource_t* d_sol)
{
	LightSource_t h_sol;
	int err;

	err = lightSourceInit(&h_sol);
	if (err != 0) {
		return -1;
	}

	err = deviceLightSourceAlloc(&h_sol, d_sol);
	lightSourceFree(&h_sol);

	return err == 0 ? 0 : -1;
}


void deviceLightSourceFree(LightSource_t* d_sol)
{
	cudaFree(d_sol->coor);

	cudaFree(d_sol->r);
	cudaFree(d_sol->g);
	cudaFree(d_sol->b);
}
