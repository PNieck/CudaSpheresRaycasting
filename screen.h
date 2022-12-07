#ifndef __SCREEN_H__
#define __SCREEN_H__


#include "cuda_runtime.h"


typedef struct Screen_t {
	float3 normalVec;
	float3 middle;
} Screen_t;


void moveOneDegree(Screen_t* s);


void initScreen(Screen_t* screen);


Screen_t* screenOnManagedAlloc();


void screenManagedFree(Screen_t* screen);


#endif
