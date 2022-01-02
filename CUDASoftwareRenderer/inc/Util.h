#pragma once
#define CUDAError(error) if(error != NULL) { std::cout << cudaGetErrorString(error) << '\n'; }
#define CAST_PIXEL(buffer) CastPixel(buffer)

struct DIB;
struct INT2;

__device__ __host__ inline DWORD* CastPixel(void* ptr)
{
	return reinterpret_cast<DWORD*>(ptr);
}

__device__ __host__ inline void AdjustPointToScreen(const std::shared_ptr<DIB> dib, INT2& point)
{
	unsigned int width = dib->GetWidth();
	unsigned int height = dib->GetHeight();

	point.x = (width - point.x) - 1;
	point.y = (height - point.y) - 1;

	return;
}