#include<pch.h>
#include"DeviceMemory.cuh"

#include"Util.h"

__device__ void* gGlobalMemory = nullptr;

DeviceMemory::DeviceMemory(long long initSize)
	: mVirtual(nullptr), mOffset(0)
{
	
	cudaDeviceSynchronize();

	cudaDeviceProp deviceInfo{};
	int deviceAddr;
	
	cudaError_t error = cudaGetDevice(&deviceAddr);
	CUDAError(error);

	error = cudaGetDeviceProperties(&deviceInfo, deviceAddr);
	CUDAError(error);
	cudaDeviceSynchronize();

	size_t requestSize = deviceInfo.totalGlobalMem / 2;

	if (initSize != -1)
	{
		requestSize = initSize;
	}


	error = cudaMalloc(reinterpret_cast<void**>(&gGlobalMemory), requestSize);
	CUDAError(error);

	if (error != NULL)
	{
		requestSize /= 2;
		error = cudaMalloc(reinterpret_cast<void**>(&gGlobalMemory), requestSize);

		CUDAError(error);
	
		assert(error == NULL);
	}

	mVirtual = gGlobalMemory;
	mSize = requestSize;
	return;
}

DeviceMemory::~DeviceMemory()
{
	cudaError_t error = cudaFree(mVirtual);
	CUDAError(error);
}