#include<pch.h>
#include"ResourceManager.cuh"
#include"DeviceMemory.cuh"
#include"DeviceTexture.cuh"
#include"DeviceBuffer.cuh"

std::shared_ptr<ResourceManager> ResourceManager::mInstance = nullptr;
ResourceManager::ResourceManager()
	: mMemory(std::make_shared<DeviceMemory>(-1))
{
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
