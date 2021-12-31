#include<pch.h>
#include"DeviceMemory.cuh"

#include"Util.h"

__device__ void* gGlobalMemory = nullptr;

DeviceMemory::DeviceMemory(size_t initSize)
	: mSize(initSize), mVirtual(nullptr), mOffset(0)
{
	
	cudaDeviceSynchronize();

	cudaDeviceProp deviceInfo{};
	int deviceAddr;
	
	cudaError_t error = cudaGetDevice(&deviceAddr);
	CUDAError(error);

	error = cudaGetDeviceProperties(&deviceInfo, deviceAddr);
	CUDAError(error);

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
	
	return;
}

std::shared_ptr<DeviceEntity> DeviceMemory::Alloc(size_t size)
{
	size_t ptr = reinterpret_cast<size_t>(mVirtual) + mOffset;

	void* casted = reinterpret_cast<void*>(ptr);
	std::shared_ptr<DeviceEntity> entity = std::make_shared<DeviceEntity>(casted, size);

	MemoryBlock block = MemoryBlock(entity, mOffset);

	mOffset += size;

	mMemoryBlocks.push_back(block);

	return entity;
}
