#pragma once
#define CUDAError(error) if(error != NULL) { std::cout << cudaGetErrorString(error) << '\n'; }
#define CAST_PIXEL(buffer) CastPixel(buffer)

#include"DIB.cuh"
#include"3DMath.cuh"

__device__ __host__ inline DWORD* CastPixel(void* ptr)
{
	return reinterpret_cast<DWORD*>(ptr);
}

__device__ __host__ inline void AdjustPointToScreen(INT2& point, unsigned int width, unsigned int height)
{
	point.x = (width - point.x) - 1;
	point.y = (height - point.y) - 1;

	return;
}

__device__ __host__ inline INT2 NDCToScreen(float x, float y, unsigned int width, unsigned int height)
{
	INT2 point(x * width, y * height);

	AdjustPointToScreen(point, width/2, height/2);

	return point;
}

template<typename _Ty>
__device__ __host__ inline void Swap(_Ty& t0, _Ty& t1)
{
	_Ty t = t0;

	t0 = t1;
	t1 = t;

	return;
}