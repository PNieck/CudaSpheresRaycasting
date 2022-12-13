#include "Screen.h"

#include "cuda_runtime.h"
#define _USE_MATH_DEFINES
#include <math.h>


#define ONE_DEGREE 0.01745329252f // one degree in radians


static Screen_t template;
static float degree;


void moveOneDegree(Screen_t* s)
{
	degree += ONE_DEGREE;

	if (degree > 2 * M_PI) {
		degree -= 2 * M_PI;
	}

	float cosAlpha = cosf(degree);
	float sinAlpha = cosf(degree);

	Screen_t tmp;
	tmp.middle.x = template.middle.x * cosAlpha - template.middle.y * sinAlpha;
	tmp.middle.y = template.middle.x * sinAlpha + template.middle.y * cosAlpha;

	tmp.normalVec.x = template.normalVec.x * cosAlpha - template.normalVec.y * sinAlpha;
	tmp.normalVec.y = template.normalVec.x * sinAlpha + template.normalVec.y * cosAlpha;

	//s->middle.x = template.middle.x * cosAlpha - template.middle.y * sinAlpha;
	//s->middle.y = template.middle.x * sinAlpha + template.middle.y * cosAlpha;

	//s->normalVec.x = template.normalVec.x * cosAlpha - template.normalVec.y * sinAlpha;
	//s->normalVec.y = template.normalVec.y * sinAlpha + template.normalVec.y * cosAlpha;

	*s = tmp;
}


static void writeInitValue(Screen_t* screen)
{
	screen->middle = make_float3(-1500, 0, 0);
	screen->normalVec = make_float3(1, 0, 0);
}


void initScreen(Screen_t* screen)
{
	writeInitValue(&template);
	degree = 0;

	*screen = template;
}


Screen_t* screenOnManagedAlloc()
{
	Screen_t* result;
	cudaError_t err;

	err = cudaMallocManaged(&result, sizeof(Screen_t), cudaMemAttachGlobal);
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


int deviceScreenInit(Screen_t* d_screen)
{
	Screen_t tmp;
	initScreen(&tmp);

	cudaMalloc(&d_screen, sizeof(Screen_t));
	cudaMemcpy(d_screen, &tmp, sizeof(Screen_t), cudaMemcpyHostToDevice);
}


void deviceScreenFree(Screen_t* d_screen)
{
	cudaFree(d_screen); 
}
