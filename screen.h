#ifndef __SCREEN_H__
#define __SCREEN_H__


#include "cuda_runtime.h"
#define _USE_MATH_DEFINES
#include <math.h>


#define ONE_DEGREE 0.01745329252f


typedef struct Screen_t {
	float3 normalVec;
	float3 middle;
} Screen_t;


void moveOneDegree(Screen_t* s);


void initScreen(Screen_t* screen);


Screen_t* screenOnManagedAlloc();


void screenManagedFree(Screen_t* screen);


int deviceScreenInit(Screen_t* d_screen);


void deviceScreenFree(Screen_t* d_screen);


inline void nextDegree(float* degree)
{
	*degree += ONE_DEGREE;

	if (*degree > 2 * M_PI) {
		*degree -= 2 * M_PI;
	}
}

#endif
