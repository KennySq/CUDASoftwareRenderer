#pragma once
#define CUDAError(error) if(error != NULL) { std::cout << cudaGetErrorString(error) << '\n'; }
#define CAST_PIXEL(buffer) CastPixel(buffer)

#include"DIB.cuh"
#include"3DMath.cuh"

__device__ __host__ inline DWORD* CastPixel(void* ptr)
{
	return reinterpret_cast<DWORD*>(ptr);
}

__device__ __host__ inline bool IsOutofScreen(const INT2& point, unsigned int width, unsigned int height)
{
	return point.x < width&& point.x >= 0 && point.y < height&& point.y >= 0 ? false : true;
}

__device__ __host__ inline bool IsOutofScreen(int x, int y, unsigned int width, unsigned int height)
{
	return ((x < width - 1) && (x >= 0) && (y < height - 1) && (y >= 0)) ? false : true;
}

__device__ __host__ inline unsigned int PointToIndex(const INT2& point, unsigned int width)
{
	return (point.y * width) + point.x;
}

__device__ __host__ inline void ClampClipSpace(INT2& point, unsigned int width, unsigned int height)
{
	Clamp<int>(point.x, 0, width - 1);
	Clamp<int>(point.y, 0, height - 1);

	return;
}

__device__ __host__ inline void AdjustPointToScreen(INT2& point, unsigned int width, unsigned int height)
{
	point.x = (width - point.x) - 1;
	point.y = (height - point.y) - 1;

	return;
}

__device__ __host__ inline FLOAT3 HomogeneousToNDC(const FLOAT4& position)
{
	return FLOAT3(position.x / position.w, position.y / position.w, position.z / position.w);
}

__device__ __host__ inline INT2 NDCToClipSpace(FLOAT3 ndc, unsigned int width, unsigned int height)
{
	INT2 pixelCoord = INT2(ndc.x * width / ndc.z, ndc.y * height / ndc.z);

	AdjustPointToScreen(pixelCoord, width/2, height/2);

	return pixelCoord;
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