#include "cuda_runtime.h"
#include <math.h>


#define ONE_DEGREE 0.01745329252f // one degree in radians


typedef struct Screen {
	float3 normalVec;
	float3 middle;
} Screen;

__host__ __device__ void moveOneDegree(Screen* s) {
	static const float rot_re = cosf(ONE_DEGREE);
	static const float rot_im = cosf(ONE_DEGREE);

	s->middle.x = s->middle.x * rot_re - s->middle.y * rot_im;
	s->middle.y = s->middle.x * rot_im + s->middle.y * rot_re;


}
