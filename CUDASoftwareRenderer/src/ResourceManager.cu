#include<pch.h>
#include"ResourceManager.cuh"
#include"DeviceMemory.cuh"
#include"DeviceTexture.cuh"
#include"DeviceBuffer.cuh"
#include"Util.h"

std::shared_ptr<ResourceManager> ResourceManager::mInstance = nullptr;
__device__ DWORD* deviceTextureBuffer;
ResourceManager::ResourceManager()
	: mMemory(std::make_shared<DeviceMemory>(-1))
{
}

__global__ void KernelCopyFromFileTexture(DWORD* dst, DWORD* src, unsigned int width)
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y)
		+ (threadIdx.y * blockDim.x) + threadIdx.x;

	unsigned int x = threadId % width;
	unsigned int y = threadId / width;

	unsigned int index = PointToIndex(INT2(x, y), width);

	BYTE sb0 = (src[threadId] & 0xFF000000) >> 24;
	BYTE sb1 = (src[threadId] & 0x00FF0000) >> 16;
	BYTE sb2 = (src[threadId] & 0x0000FF00) >> 8;
	BYTE sb3 = (src[threadId] & 0x000000FF) >> 0;

	dst[threadId] |= (sb0 << 24);
	dst[threadId] |= (sb1 << 0);
	dst[threadId] |= (sb2 << 8);
	dst[threadId] |= (sb3 << 16);

}

std::shared_ptr<DeviceTexture> ResourceManager::CreateTextureFromFile(const char* path)
{
	unsigned char* buffer;
	size_t bufferSize;

	unsigned int width, height;
	
	unsigned int error = lodepng_decode_file(&buffer, &width, &height, path, LCT_RGBA, 8);
	size_t byteSize = (size_t)width * height * sizeof(DWORD);

	void* ptr = mMemory->Alloc(byteSize);

	std::shared_ptr<DeviceTexture> texture = std::make_shared<DeviceTexture>(ptr, byteSize);

	cudaError_t cuError = cudaMalloc(reinterpret_cast<void**>(&deviceTextureBuffer), byteSize);
	CUDAError(cuError);

	cuError = cudaMemcpy(deviceTextureBuffer, buffer, byteSize, cudaMemcpyHostToDevice);
	CUDAError(cuError);

	dim3 block = dim3(32, 32, 1);
	dim3 grid = dim3(width / block.x, height / block.y, 1);

	KernelCopyFromFileTexture<<<grid,block>>>(CAST_PIXEL(ptr), deviceTextureBuffer, width);
	
	cudaFree(deviceTextureBuffer);
	return texture;
}

std::shared_ptr<DeviceTexture> ResourceManager::CreateTexture2D(unsigned int width, unsigned height)
{
	size_t byteSize = (size_t)width * height * sizeof(DWORD);

	void* ptr = mMemory->Alloc(byteSize);

	std::shared_ptr<DeviceTexture> texture = std::make_shared<DeviceTexture>(ptr, byteSize);

	texture->mWidth = width;
	texture->mHeight;

	return texture;
}

std::shared_ptr<DeviceBuffer> ResourceManager::CreateBuffer(size_t stride, unsigned int count, void* subResource)
{
	size_t size = stride * count;
	void* ptr = mMemory->Alloc(size);

	std::shared_ptr<DeviceBuffer> buffer = std::make_shared<DeviceBuffer>(ptr, size);

	buffer->mStride = stride;

	if (subResource != nullptr)
	{
		cudaMemcpy(ptr, subResource, size, cudaMemcpyHostToDevice);
	}

	return buffer;
}
