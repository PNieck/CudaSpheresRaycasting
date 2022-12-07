#include "Screen.h"

#include "cuda_runtime.h"
#include <math.h>


#define ONE_DEGREE 0.01745329252f // one degree in radians


void moveOneDegree(Screen_t* s) {
	float rot_re = cosf(ONE_DEGREE);
	float rot_im = cosf(ONE_DEGREE);

	s->middle.x = s->middle.x * rot_re - s->middle.y * rot_im;
	s->middle.y = s->middle.x * rot_im + s->middle.y * rot_re;
}


void initScreen(Screen_t* screen)
{
	screen->middle = make_float3(-400, 0, 0);
	screen->normalVec = make_float3(1, 0, 0);
}


Screen_t* screenOnManagedAlloc()
{
	Screen_t* result;
	cudaError_t err;

	err = cudaMallocManaged(&result, sizeof(result), cudaMemAttachGlobal);
	if (err != cudaSuccess) {
		return NULL;
	}

	initScreen(result);

	return result;
}


void screenManagedFree(Screen_t* screen)
{
	cudaFree(screen);
}
