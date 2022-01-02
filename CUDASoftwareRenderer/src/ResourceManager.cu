#include<pch.h>
#include"ResourceManager.cuh"
#include"DeviceMemory.cuh"
#include"DeviceTexture.cuh"

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